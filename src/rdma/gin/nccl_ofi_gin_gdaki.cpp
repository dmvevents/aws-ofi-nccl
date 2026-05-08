/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GDAKI plugin for the GIN API — EFA backend.
 *
 * This file implements the GDAKI-specific entry points:
 *   - createContext / destroyContext
 *   - get_properties
 *   - queryLastError
 *
 * Shared plugin APIs (init, devices, listen, connect, regMrSym,
 * regMrSymDmaBuf, deregMrSym, closeColl, closeListen, ginProgress,
 * finalize) are reused from the proxy-side implementations in
 * nccl_ofi_gin_api.cpp, which at init time copies the proxy-plugin
 * pointers into the GDAKI plugin table.
 *
 * ## Struct-layout guarantee
 *
 * The device handle returned from createContext (bytes 0..71) is
 * binary-layout-identical to NVIDIA's ncclGinGdakiGPUContext. DeepEP casts
 * the pointer to that struct and reads signals_table.buffer[sigId] via __ldg;
 * if the first 72 bytes drift, cross-node barrier takes a CUDA illegal
 * memory access fault. See docs/ROOT-CAUSE-struct-alias-gap-2026-05-08.md.
 */

#include "config.h"

#include "rdma/gin/nccl_ofi_gin_gdaki.h"
#include "nccl_ofi.h"
#include "nccl_ofi_api.h"
#include "nccl_ofi_param.h"

bool nccl_ofi_gin_gdaki_enabled()
{
	return ofi_nccl_gin_gdaki.get();
}

static ncclResult_t nccl_ofi_gin_gdaki_get_properties(int dev, ncclNetProperties_v12_t *props)
{
	nccl_ofi_properties_t ofi_properties;
	ncclResult_t ret = nccl_net_ofi_get_properties(dev, &ofi_properties);
	if (ret != ncclSuccess) {
		return ret;
	}

	props->name = ofi_properties.name;
	props->pciPath = ofi_properties.pci_path;
	props->guid = ofi_properties.guid;
	props->ptrSupport = NCCL_PTR_HOST;
	if (ofi_properties.hmem_support) {
		props->ptrSupport |= NCCL_PTR_CUDA;
	}
	if (ofi_properties.dmabuf_support) {
		props->ptrSupport |= NCCL_PTR_DMABUF;
	}

	props->regIsGlobal = ofi_properties.regIsGlobal;
	props->forceFlush = 0;
	props->speed = ofi_properties.port_speed;
	props->port = ofi_properties.port_number;
	props->latency = ofi_properties.latency;
	props->maxComms = ofi_properties.max_communicators;
	props->maxRecvs = ofi_properties.max_group_receives;
	props->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
	props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
	props->vProps.ndevs = 1;
	props->vProps.devs[0] = dev;
	props->maxP2pBytes = ofi_properties.max_p2p_bytes;
	props->maxCollBytes = ofi_properties.max_coll_bytes;
	props->maxMultiRequestSize = 1;
	props->railId = -1;
	props->planeId = -1;

	return ncclSuccess;
}

#if HAVE_EFA_DP_DIRECT && HAVE_DECL_FI_EFA_GDA_OPS

#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_ext_efa.h>

#include <efa_cuda_dp.h>

#include "nccl_ofi_cuda.h"
#include "rdma/gin/nccl_ofi_gin.h"
#include "rdma/gin/nccl_ofi_gin_gdaki_ctx.h"

/* Helper: CUDA-allocate + zero. Returns nullptr on failure. */
static void *cuda_alloc_zeroed(size_t size)
{
	void *ptr = nullptr;
	cudaError_t err = cudaMalloc(&ptr, size);
	if (err != cudaSuccess) {
		NCCL_OFI_WARN("gin GDAKI: cudaMalloc(%zu) failed: %s",
			      size, cudaGetErrorString(err));
		return nullptr;
	}
	err = cudaMemset(ptr, 0, size);
	if (err != cudaSuccess) {
		NCCL_OFI_WARN("gin GDAKI: cudaMemset failed: %s", cudaGetErrorString(err));
		cudaFree(ptr);
		return nullptr;
	}
	return ptr;
}

static inline void cuda_free(void *ptr)
{
	if (ptr) {
		cudaFree(ptr);
	}
}

/* Open an efa-direct fi_info targeting the same EFA device as the proxy
 * transport. The domain name comes from the existing proxy fi_info. */
static struct fi_info *gdaki_get_efa_direct_info(const char *domain_name)
{
	struct fi_info *hints = fi_allocinfo();
	if (hints == nullptr) {
		throw std::runtime_error("fi_allocinfo failed");
	}

	hints->caps = FI_MSG | FI_RMA | FI_WRITE | FI_READ |
		      FI_REMOTE_WRITE | FI_REMOTE_READ |
		      FI_SEND | FI_RECV | FI_SOURCE;
	hints->mode = FI_CONTEXT2;
	hints->ep_attr->type = FI_EP_RDM;

	hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR |
				      FI_MR_ALLOCATED | FI_MR_PROV_KEY;
	hints->domain_attr->threading = FI_THREAD_SAFE;
	hints->domain_attr->data_progress = FI_PROGRESS_AUTO;
	hints->domain_attr->control_progress = FI_PROGRESS_AUTO;

	hints->domain_attr->name = strdup(domain_name);
	hints->fabric_attr->name = strdup("efa-direct");

	struct fi_info *results = nullptr;
	int ret = fi_getinfo(FI_VERSION(2, 0), nullptr, nullptr, 0ULL, hints, &results);
	fi_freeinfo(hints);
	if (ret != 0 || results == nullptr) {
		throw std::runtime_error(std::string("fi_getinfo efa-direct failed: ") +
					 fi_strerror(-ret));
	}
	return results;
}

/* Register a GPU buffer on the efa-direct domain using dmabuf when the GPU
 * driver supports it, falling back to legacy FI_HMEM registration otherwise.
 */
static void gdaki_mr_reg_gpu(struct fid_domain *domain, void *ptr, size_t size,
			     struct fid_mr **mr_out)
{
	int dmabuf_fd = -1;
	size_t dmabuf_offset = 0;
	int rc = nccl_net_ofi_gpu_get_dma_buf_fd(ptr, size,
						 &dmabuf_fd, &dmabuf_offset);

	struct iovec iov = { ptr, size };
	struct fi_mr_attr attr = {};
	attr.mr_iov = &iov;
	attr.iov_count = 1;
	attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE |
		      FI_REMOTE_READ | FI_REMOTE_WRITE;
	attr.iface = FI_HMEM_CUDA;

	uint64_t flags = 0;
	struct fi_mr_dmabuf dmabuf = {};
	if (rc == 0 && dmabuf_fd >= 0) {
		dmabuf.fd = dmabuf_fd;
		dmabuf.offset = dmabuf_offset;
		dmabuf.len = size;
		dmabuf.base_addr = ptr;
		attr.dmabuf = &dmabuf;
		flags = FI_MR_DMABUF;
	}

	int ret = fi_mr_regattr(domain, &attr, flags, mr_out);
	if (ret != 0) {
		throw std::runtime_error(std::string("fi_mr_regattr GPU buf: ") +
					 fi_strerror(-ret));
	}
}

/* Tear down everything in ctx. Safe to call on partial init because every
 * field is zeroed on allocation. */
static void gdaki_destroy_ctx(struct nccl_ofi_gin_gdaki_context *ctx)
{
	if (ctx == nullptr) {
		return;
	}

	if (ctx->d_qp) {
		efa_cuda_destroy_qp(ctx->d_qp);
	}
	if (ctx->d_cq) {
		efa_cuda_destroy_cq(ctx->d_cq);
	}

	cuda_free(ctx->d_handle);
	cuda_free(ctx->d_signals_rkeys);
	cuda_free(ctx->d_peer_signal_vas);
	cuda_free(ctx->d_address_handles);
	cuda_free(ctx->d_remote_qpns);
	cuda_free(ctx->d_qkeys);
	cuda_free(ctx->d_peer_locks);

	if (ctx->signals_mr) {
		fi_close(&ctx->signals_mr->fid);
	}
	cuda_free(ctx->signals_buffer_dev);
	if (ctx->sink_mr) {
		fi_close(&ctx->sink_mr->fid);
	}
	cuda_free(ctx->sink_buffer_dev);

	if (ctx->ofi_ep) {
		fi_close(&ctx->ofi_ep->fid);
	}
	if (ctx->ofi_cq) {
		fi_close(&ctx->ofi_cq->fid);
	}
	if (ctx->ofi_av) {
		fi_close(&ctx->ofi_av->fid);
	}
	if (ctx->ofi_domain) {
		fi_close(&ctx->ofi_domain->fid);
	}
	if (ctx->ofi_fabric) {
		fi_close(&ctx->ofi_fabric->fid);
	}
	if (ctx->ofi_info) {
		fi_freeinfo(ctx->ofi_info);
	}

	delete ctx;
}

#endif /* HAVE_EFA_DP_DIRECT && HAVE_DECL_FI_EFA_GDA_OPS */

static ncclResult_t nccl_ofi_gin_gdaki_createContext(void *collComm,
						     ncclGinConfig_v13_t *config,
						     void **ginCtx,
						     ncclNetDeviceHandle_v11_t **devHandle)
{
#if !(HAVE_EFA_DP_DIRECT && HAVE_DECL_FI_EFA_GDA_OPS)
	(void)collComm; (void)config; (void)ginCtx; (void)devHandle;
	NCCL_OFI_WARN("gin GDAKI: createContext requires efa-dp-direct and "
		      "FI_EFA_GDA_OPS; plugin built without one or both");
	return ncclInternalError;
#else
	if (collComm == nullptr || config == nullptr ||
	    ginCtx == nullptr || devHandle == nullptr) {
		NCCL_OFI_WARN("gin GDAKI: createContext received NULL argument");
		return ncclInvalidArgument;
	}

	auto *put_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(collComm);
	int nranks = put_comm->get_nranks();
	int rank = put_comm->get_rank();
	int nSignals = (config->nSignals > 0) ? config->nSignals : nranks;

	auto *ctx = new (std::nothrow) nccl_ofi_gin_gdaki_context();
	if (ctx == nullptr) {
		return ncclSystemError;
	}
	memset(ctx, 0, sizeof(*ctx));
	ctx->nSignals = nSignals;
	ctx->nCounters = config->nCounters;
	ctx->nranks = nranks;
	ctx->rank = rank;

	try {
		/* ---- 1. Open efa-direct fabric + domain ---- */

		auto *plugin = nccl_net_ofi_get_plugin();
		auto *device = plugin->get_device(put_comm->get_dev());
		if (device == nullptr) {
			throw std::runtime_error("get_device returned null");
		}
		struct fi_info *proxy_info = device->get_ofi_info(0);
		if (proxy_info == nullptr || proxy_info->domain_attr == nullptr) {
			throw std::runtime_error("proxy fi_info missing domain_attr");
		}
		ctx->ofi_info = gdaki_get_efa_direct_info(proxy_info->domain_attr->name);

		int ret = fi_fabric(ctx->ofi_info->fabric_attr, &ctx->ofi_fabric, nullptr);
		if (ret != 0) {
			throw std::runtime_error(std::string("fi_fabric: ") + fi_strerror(-ret));
		}
		ret = fi_domain(ctx->ofi_fabric, ctx->ofi_info, &ctx->ofi_domain, nullptr);
		if (ret != 0) {
			throw std::runtime_error(std::string("fi_domain: ") + fi_strerror(-ret));
		}

		/* ---- 2. Open CQ, AV, endpoint ---- */

		struct fi_cq_attr cq_attr = {};
		cq_attr.format = FI_CQ_FORMAT_DATA;
		cq_attr.size = 1024;
		ret = fi_cq_open(ctx->ofi_domain, &cq_attr, &ctx->ofi_cq, nullptr);
		if (ret != 0) {
			throw std::runtime_error(std::string("fi_cq_open: ") + fi_strerror(-ret));
		}

		struct fi_av_attr av_attr = {};
		av_attr.type = FI_AV_TABLE;
		ret = fi_av_open(ctx->ofi_domain, &av_attr, &ctx->ofi_av, nullptr);
		if (ret != 0) {
			throw std::runtime_error(std::string("fi_av_open: ") + fi_strerror(-ret));
		}

		ret = fi_endpoint(ctx->ofi_domain, ctx->ofi_info, &ctx->ofi_ep, nullptr);
		if (ret != 0) {
			throw std::runtime_error(std::string("fi_endpoint: ") + fi_strerror(-ret));
		}
		ret = fi_ep_bind(ctx->ofi_ep, &ctx->ofi_cq->fid, FI_TRANSMIT | FI_RECV);
		if (ret != 0) {
			throw std::runtime_error(std::string("fi_ep_bind CQ: ") + fi_strerror(-ret));
		}
		ret = fi_ep_bind(ctx->ofi_ep, &ctx->ofi_av->fid, 0);
		if (ret != 0) {
			throw std::runtime_error(std::string("fi_ep_bind AV: ") + fi_strerror(-ret));
		}
		ret = fi_enable(ctx->ofi_ep);
		if (ret != 0) {
			throw std::runtime_error(std::string("fi_enable: ") + fi_strerror(-ret));
		}

		/* ---- 3. Open GDA ops extension ---- */

		struct fi_efa_ops_gda *gda_ops = nullptr;
		ret = fi_open_ops(&ctx->ofi_domain->fid, FI_EFA_GDA_OPS, 0,
				  (void **)&gda_ops, nullptr);
		if (ret != 0 || gda_ops == nullptr) {
			throw std::runtime_error(std::string("fi_open_ops FI_EFA_GDA_OPS: ") +
						 fi_strerror(-ret));
		}
		ctx->gda_ops = gda_ops;

		/* ---- 4. Query QP/CQ attributes + create efa-dp-direct QP/CQ ---- */

		struct fi_efa_wq_attr sq_attr = {}, rq_attr = {};
		ret = gda_ops->query_qp_wqs(ctx->ofi_ep, &sq_attr, &rq_attr);
		if (ret != 0) {
			throw std::runtime_error(std::string("query_qp_wqs: ") + fi_strerror(-ret));
		}

		struct fi_efa_cq_attr efa_cq_attr = {};
		ret = gda_ops->query_cq(ctx->ofi_cq, &efa_cq_attr);
		if (ret != 0) {
			throw std::runtime_error(std::string("query_cq: ") + fi_strerror(-ret));
		}

		/* SQ buffer and doorbell are PCIe MMIO. cudaHostRegister with
		 * cudaHostRegisterIoMemory makes them visible to GPU kernels. */
		cudaError_t cu;
		cu = cudaHostRegister(sq_attr.buffer,
				     (size_t)sq_attr.num_entries * sq_attr.entry_size,
				     cudaHostRegisterIoMemory);
		if (cu != cudaSuccess) {
			throw std::runtime_error(std::string("cudaHostRegister SQ buffer: ") +
						 cudaGetErrorString(cu));
		}
		cu = cudaHostRegister(sq_attr.doorbell, 4096, cudaHostRegisterIoMemory);
		if (cu != cudaSuccess) {
			throw std::runtime_error(std::string("cudaHostRegister SQ doorbell: ") +
						 cudaGetErrorString(cu));
		}

		struct efa_cuda_qp_attrs qp_attrs = {};
		qp_attrs.sq_buffer = sq_attr.buffer;
		qp_attrs.rq_buffer = rq_attr.buffer;
		qp_attrs.sq_doorbell = sq_attr.doorbell;
		qp_attrs.rq_doorbell = rq_attr.doorbell;
		qp_attrs.sq_num_entries = sq_attr.num_entries;
		qp_attrs.sq_entry_size = sq_attr.entry_size;
		qp_attrs.sq_max_batch = sq_attr.max_batch;
		qp_attrs.rq_num_entries = rq_attr.num_entries;
		qp_attrs.rq_entry_size = rq_attr.entry_size;

		ctx->d_qp = efa_cuda_create_qp(&qp_attrs, sizeof(qp_attrs));
		if (ctx->d_qp == nullptr) {
			throw std::runtime_error("efa_cuda_create_qp failed");
		}

		struct efa_cuda_cq_attrs cq_attrs = {};
		cq_attrs.buffer = efa_cq_attr.buffer;
		cq_attrs.num_entries = efa_cq_attr.num_entries;
		cq_attrs.entry_size = efa_cq_attr.entry_size;

		ctx->d_cq = efa_cuda_create_cq(&cq_attrs, sizeof(cq_attrs));
		if (ctx->d_cq == nullptr) {
			throw std::runtime_error("efa_cuda_create_cq failed");
		}

		/* ---- 5. Allgather endpoint addresses + per-peer (AHN, QPN, qkey) ---- */

		size_t ep_addr_len = 0;
		fi_getname(&ctx->ofi_ep->fid, nullptr, &ep_addr_len);
		std::vector<uint8_t> all_addrs((size_t)nranks * ep_addr_len, 0);
		ret = fi_getname(&ctx->ofi_ep->fid,
				 &all_addrs[(size_t)rank * ep_addr_len], &ep_addr_len);
		if (ret != 0) {
			throw std::runtime_error(std::string("fi_getname: ") + fi_strerror(-ret));
		}
		ret = put_comm->get_ag_comm().all_gather(all_addrs.data(), ep_addr_len);
		if (ret != 0) {
			throw std::runtime_error("allgather of ep addresses failed");
		}

		std::vector<uint16_t> h_ahns(nranks);
		std::vector<uint16_t> h_qpns(nranks);
		std::vector<uint32_t> h_qkeys(nranks);

		for (int i = 0; i < nranks; i++) {
			fi_addr_t fi_addr;
			ret = fi_av_insert(ctx->ofi_av,
					   &all_addrs[(size_t)i * ep_addr_len], 1,
					   &fi_addr, 0, nullptr);
			if (ret != 1) {
				throw std::runtime_error("fi_av_insert failed for rank " +
							 std::to_string(i));
			}
			uint16_t ahn = 0, remote_qpn = 0;
			uint32_t remote_qkey = 0;
			ret = gda_ops->query_addr(ctx->ofi_ep, fi_addr,
						  &ahn, &remote_qpn, &remote_qkey);
			if (ret != 0) {
				throw std::runtime_error("query_addr failed for rank " +
							 std::to_string(i));
			}
			h_ahns[i] = ahn;
			h_qpns[i] = remote_qpn;
			h_qkeys[i] = remote_qkey;
		}

		/* Upload per-peer tables to GPU. */
		ctx->d_address_handles = static_cast<uint16_t *>(
			cuda_alloc_zeroed((size_t)nranks * sizeof(uint16_t)));
		ctx->d_remote_qpns = static_cast<uint16_t *>(
			cuda_alloc_zeroed((size_t)nranks * sizeof(uint16_t)));
		ctx->d_qkeys = static_cast<uint32_t *>(
			cuda_alloc_zeroed((size_t)nranks * sizeof(uint32_t)));
		if (!ctx->d_address_handles || !ctx->d_remote_qpns || !ctx->d_qkeys) {
			throw std::runtime_error("cudaMalloc for per-peer tables failed");
		}
		cu = cudaMemcpy(ctx->d_address_handles, h_ahns.data(),
				(size_t)nranks * sizeof(uint16_t), cudaMemcpyHostToDevice);
		if (cu != cudaSuccess) {
			throw std::runtime_error("cudaMemcpy address_handles failed");
		}
		cu = cudaMemcpy(ctx->d_remote_qpns, h_qpns.data(),
				(size_t)nranks * sizeof(uint16_t), cudaMemcpyHostToDevice);
		if (cu != cudaSuccess) {
			throw std::runtime_error("cudaMemcpy remote_qpns failed");
		}
		cu = cudaMemcpy(ctx->d_qkeys, h_qkeys.data(),
				(size_t)nranks * sizeof(uint32_t), cudaMemcpyHostToDevice);
		if (cu != cudaSuccess) {
			throw std::runtime_error("cudaMemcpy qkeys failed");
		}

		/* Per-peer spinlock array (zero-init). Used by override's Put
		 * signal-only branch to serialize same-peer WRs across concurrent
		 * threads within one SM. Closes the EFA SRD reorder race where
		 * two WRs to the same 8-byte slot could be delivered out of
		 * order and the older stamp overwrites the newer. */
		ctx->d_peer_locks = static_cast<uint32_t *>(
			cuda_alloc_zeroed((size_t)nranks * sizeof(uint32_t)));
		if (!ctx->d_peer_locks) {
			throw std::runtime_error("cudaMalloc for peer_locks failed");
		}

		/* ---- 6. Allocate + register signals_buffer (fixes the struct-alias
		 *        bug that caused 2026-05-08 cross-node CUDA illegal addr) ---- */

		ctx->signals_buffer_size = (size_t)nSignals * sizeof(uint64_t);
		ctx->signals_buffer_dev = cuda_alloc_zeroed(ctx->signals_buffer_size);
		if (ctx->signals_buffer_dev == nullptr) {
			throw std::runtime_error("cudaMalloc signals_buffer failed");
		}
		gdaki_mr_reg_gpu(ctx->ofi_domain, ctx->signals_buffer_dev,
				 ctx->signals_buffer_size, &ctx->signals_mr);

		ctx->signals_lkey = (__be32)gda_ops->get_mr_lkey(ctx->signals_mr);
		uint64_t my_signals_rkey = fi_mr_key(ctx->signals_mr);

		/* ---- 7. Allgather signals rkeys + base VAs across peers ---- */

		std::vector<uint64_t> h_signals_rkeys(nranks, 0);
		std::vector<uint64_t> h_peer_signal_vas(nranks, 0);
		h_signals_rkeys[rank] = my_signals_rkey;
		h_peer_signal_vas[rank] = (uint64_t)ctx->signals_buffer_dev;

		ret = put_comm->get_ag_comm().all_gather(h_signals_rkeys.data(),
							 sizeof(uint64_t));
		if (ret != 0) {
			throw std::runtime_error("allgather of signals rkeys failed");
		}
		ret = put_comm->get_ag_comm().all_gather(h_peer_signal_vas.data(),
							 sizeof(uint64_t));
		if (ret != 0) {
			throw std::runtime_error("allgather of signals VAs failed");
		}

		/* rkeys on the wire are __be32 even though fi_mr_key returns uint64_t.
		 * efa-dp-direct expects the lower 32 bits. */
		std::vector<__be32> h_signals_rkeys_be(nranks);
		for (int i = 0; i < nranks; i++) {
			h_signals_rkeys_be[i] = (__be32)h_signals_rkeys[i];
		}

		ctx->d_signals_rkeys = static_cast<__be32 *>(
			cuda_alloc_zeroed((size_t)nranks * sizeof(__be32)));
		ctx->d_peer_signal_vas = static_cast<uint64_t *>(
			cuda_alloc_zeroed((size_t)nranks * sizeof(uint64_t)));
		if (!ctx->d_signals_rkeys || !ctx->d_peer_signal_vas) {
			throw std::runtime_error("cudaMalloc for signals tables failed");
		}
		cu = cudaMemcpy(ctx->d_signals_rkeys, h_signals_rkeys_be.data(),
				(size_t)nranks * sizeof(__be32), cudaMemcpyHostToDevice);
		if (cu != cudaSuccess) {
			throw std::runtime_error("cudaMemcpy signals_rkeys failed");
		}
		cu = cudaMemcpy(ctx->d_peer_signal_vas, h_peer_signal_vas.data(),
				(size_t)nranks * sizeof(uint64_t), cudaMemcpyHostToDevice);
		if (cu != cudaSuccess) {
			throw std::runtime_error("cudaMemcpy peer_signal_vas failed");
		}

		/* ---- 8. Allocate + register the 8-byte sink buffer ---- */

		ctx->sink_buffer_dev = cuda_alloc_zeroed(sizeof(uint64_t));
		if (ctx->sink_buffer_dev == nullptr) {
			throw std::runtime_error("cudaMalloc sink_buffer failed");
		}
		gdaki_mr_reg_gpu(ctx->ofi_domain, ctx->sink_buffer_dev,
				 sizeof(uint64_t), &ctx->sink_mr);
		ctx->sink_lkey = (__be32)gda_ops->get_mr_lkey(ctx->sink_mr);

		/* ---- 9. Populate + upload device handle (NVIDIA-compat layout) ---- */

		struct nccl_ofi_gin_gdaki_dev_handle h = {};
		/* NVIDIA-compat prefix (bytes 0..71). */
		h.gdqp = ctx->d_qp;                                /* [0]   */
		h.companion_gdqp = nullptr;                        /* [8]   */
		/* counters_table zero: not used for v1 signal path (spec D2). */
		h.signals_table.buffer =
			static_cast<uint64_t *>(ctx->signals_buffer_dev); /* [40] */
		h.signals_table.rkeys = ctx->d_signals_rkeys;      /* [48]  */
		h.signals_table.lkey = ctx->signals_lkey;          /* [56]  */
		h.signals_table.offset = 0;                        /* [60]  */
		h.sink_buffer_lkey = ctx->sink_lkey;               /* [64]  */
		/* EFA extension (bytes 72+). Read only by our override. */
		h.address_handles = ctx->d_address_handles;        /* [72]  */
		h.remote_qpns = ctx->d_remote_qpns;                /* [80]  */
		h.qkey = ctx->d_qkeys;                             /* [88]  */
		h.peer_signal_vas = ctx->d_peer_signal_vas;        /* [96]  */
		h.nranks = nranks;                                 /* [104] */
		h.rank = rank;                                     /* [108] */
		h.cq = ctx->d_cq;                                  /* [112] */
		h.peer_locks = ctx->d_peer_locks;                  /* [120] */

		ctx->d_handle = static_cast<nccl_ofi_gin_gdaki_dev_handle *>(
			cuda_alloc_zeroed(sizeof(h)));
		if (ctx->d_handle == nullptr) {
			throw std::runtime_error("cudaMalloc device handle failed");
		}
		cu = cudaMemcpy(ctx->d_handle, &h, sizeof(h), cudaMemcpyHostToDevice);
		if (cu != cudaSuccess) {
			throw std::runtime_error("cudaMemcpy device handle failed");
		}

		/* ---- 10. Populate host-side ncclNetDeviceHandle ---- */

		auto *dev_handle = static_cast<ncclNetDeviceHandle_v11_t *>(
			calloc(1, sizeof(ncclNetDeviceHandle_v11_t)));
		if (dev_handle == nullptr) {
			throw std::runtime_error("calloc ncclNetDeviceHandle failed");
		}
		dev_handle->netDeviceType = NCCL_NET_DEVICE_GIN_GDAKI;
		dev_handle->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
		dev_handle->handle = ctx->d_handle;
		dev_handle->size = sizeof(struct nccl_ofi_gin_gdaki_dev_handle);
		dev_handle->needsProxyProgress = 0;

		*ginCtx = ctx;
		*devHandle = dev_handle;
		put_comm->set_gdaki_ctx(ctx);

		NCCL_OFI_INFO(NCCL_NET,
			      "gin GDAKI: createContext done (nranks=%d rank=%d nSignals=%d "
			      "nCounters=%d sq_entries=%u sq_entry_size=%u cq_entries=%u)",
			      nranks, rank, nSignals, config->nCounters,
			      sq_attr.num_entries, sq_attr.entry_size,
			      efa_cq_attr.num_entries);
		return ncclSuccess;

	} catch (const std::exception &e) {
		NCCL_OFI_WARN("gin GDAKI: createContext failed: %s", e.what());
		gdaki_destroy_ctx(ctx);
		return ncclSystemError;
	}
#endif /* HAVE_EFA_DP_DIRECT && HAVE_DECL_FI_EFA_GDA_OPS */
}

static ncclResult_t nccl_ofi_gin_gdaki_destroyContext(void *ginCtx)
{
#if !(HAVE_EFA_DP_DIRECT && HAVE_DECL_FI_EFA_GDA_OPS)
	(void)ginCtx;
	return ncclSuccess;
#else
	gdaki_destroy_ctx(static_cast<nccl_ofi_gin_gdaki_context *>(ginCtx));
	return ncclSuccess;
#endif
}

/*
 * GDAKI-native regMrSym. DeepEP dispatch calls this per window
 * (per token buffer, per scale MR). We need to:
 *   1. Let the proxy-side regMrSym do its job (symmetric bootstrap).
 *   2. Additionally register the buffer on OUR efa-direct domain so the
 *      GPU kernel can post RDMA_WRITEs targeting it.
 *   3. Allgather per-peer rkeys AND per-peer base VAs across ranks.
 *   4. Build an mr_handle whose layout matches what the override expects:
 *        { __be32* rkeys_ptr; __be32 lkey; int32_t nranks;
 *          __be32 rkeys[nranks]; uint64_t peer_bases[nranks]; }
 *   5. Wrap it in nccl_ofi_gin_gdaki_mr_reg { fid_mr*, mr_handle* }
 *      and publish through *ginHandle.
 *
 * Before T6 (2026-05-08 evening) we inherited the proxy regMrSym which
 * returned a proxy-domain handle through ginHandle — our override read
 * that and got garbage rkeys/VAs, failing dispatch with
 * "CPU side received count: 0 0 0 0". The override's peer_mr_base()
 * helper expects peer_bases to follow rkeys in-allocation; previous
 * reference impls (anshumang fork) wrote only rkeys, missing peer_bases.
 * This implementation writes both.
 */
#if HAVE_EFA_DP_DIRECT && HAVE_DECL_FI_EFA_GDA_OPS

static ncclResult_t gdaki_reg_mr_common(void *collComm, void *data, size_t size,
					int type, uint64_t mrFlags,
					int dmabuf_fd, size_t dmabuf_offset,
					void **mhandle, void **ginHandle)
{
	if (collComm == nullptr || data == nullptr || mhandle == nullptr ||
	    ginHandle == nullptr) {
		return ncclInvalidArgument;
	}

	auto *put_comm = static_cast<nccl_ofi_rdma_gin_put_comm *>(collComm);
	auto *ctx = static_cast<nccl_ofi_gin_gdaki_context *>(put_comm->get_gdaki_ctx());

	if (ctx == nullptr) {
		/* createContext not yet called — fall back to proxy-only. */
		if (dmabuf_fd >= 0) {
			return nccl_ofi_gin_regMrSymDmaBuf(collComm, data, size, type,
							   dmabuf_offset, dmabuf_fd,
							   mrFlags, mhandle, ginHandle);
		}
		return nccl_ofi_gin_regMrSym(collComm, data, size, type,
					     mrFlags, mhandle, ginHandle);
	}

	/* Step 1: proxy-side registration (bootstrap). */
	ncclResult_t pret;
	if (dmabuf_fd >= 0) {
		pret = nccl_ofi_gin_regMrSymDmaBuf(collComm, data, size, type,
						   dmabuf_offset, dmabuf_fd,
						   mrFlags, mhandle, ginHandle);
	} else {
		pret = nccl_ofi_gin_regMrSym(collComm, data, size, type,
					     mrFlags, mhandle, ginHandle);
	}
	if (pret != ncclSuccess) {
		return pret;
	}
	/* We are about to overwrite *ginHandle with our own wrapper. The
	 * proxy's ginHandle pointer is stashed in *mhandle (same object),
	 * so the proxy deregMrSym path still works via mhandle. */

	int nranks = ctx->nranks;
	int rank = ctx->rank;
	auto *gda_ops = static_cast<struct fi_efa_ops_gda *>(ctx->gda_ops);

	try {
		/* Step 2: register on our efa-direct domain. For CUDA memory,
		 * prefer dmabuf to avoid GDRCopy path; fall back to FI_HMEM. */
		struct fid_mr *mr = nullptr;
		struct iovec iov = {data, size};
		struct fi_mr_attr attr = {};
		attr.mr_iov = &iov;
		attr.iov_count = 1;
		attr.access = FI_SEND | FI_RECV | FI_READ | FI_WRITE |
			      FI_REMOTE_READ | FI_REMOTE_WRITE;
		if (type == NCCL_PTR_CUDA) {
			attr.iface = FI_HMEM_CUDA;
		}

		uint64_t flags = 0;
		struct fi_mr_dmabuf dmabuf = {};
		int probed_dmabuf_fd = dmabuf_fd;
		size_t probed_dmabuf_offset = dmabuf_offset;
		if (type == NCCL_PTR_CUDA && probed_dmabuf_fd < 0) {
			int rc = nccl_net_ofi_gpu_get_dma_buf_fd(
				data, size, &probed_dmabuf_fd, &probed_dmabuf_offset);
			if (rc != 0) {
				probed_dmabuf_fd = -1;
			}
		}
		if (probed_dmabuf_fd >= 0) {
			dmabuf.fd = probed_dmabuf_fd;
			dmabuf.offset = probed_dmabuf_offset;
			dmabuf.len = size;
			dmabuf.base_addr = data;
			attr.dmabuf = &dmabuf;
			flags = FI_MR_DMABUF;
		}

		int ret = fi_mr_regattr(ctx->ofi_domain, &attr, flags, &mr);
		if (ret != 0) {
			throw std::runtime_error(std::string("fi_mr_regattr GDAKI window: ") +
						 fi_strerror(-ret));
		}

		uint32_t lkey_val = (uint32_t)gda_ops->get_mr_lkey(mr);
		uint64_t rkey_val = fi_mr_key(mr);
		uint64_t va_val = (uint64_t)data;

		/* Step 3: allgather per-peer rkeys AND per-peer base VAs. */
		std::vector<uint64_t> all_rkeys(nranks, 0);
		std::vector<uint64_t> all_vas(nranks, 0);
		all_rkeys[rank] = rkey_val;
		all_vas[rank] = va_val;
		ret = put_comm->get_ag_comm().all_gather(all_rkeys.data(),
							 sizeof(uint64_t));
		if (ret != 0) {
			fi_close(&mr->fid);
			throw std::runtime_error("allgather of window rkeys failed");
		}
		ret = put_comm->get_ag_comm().all_gather(all_vas.data(),
							 sizeof(uint64_t));
		if (ret != 0) {
			fi_close(&mr->fid);
			throw std::runtime_error("allgather of window VAs failed");
		}

		/* Step 4: allocate mr_handle with exact layout override expects:
		 *   struct { __be32 *rkeys_ptr; __be32 lkey; int32_t nranks; }
		 *   then __be32 rkeys[nranks]
		 *   then uint64_t peer_bases[nranks]
		 * All in one allocation so free() works.
		 *
		 * rkeys[] must be 4-byte aligned (it is, from sizeof struct).
		 * peer_bases[] must be 8-byte aligned — rkeys_tail is at
		 * sizeof(hdr) + nranks*4. We pad to 8-byte boundary if nranks is
		 * odd, so peer_bases starts on 8-byte alignment. */
		size_t hdr_size = sizeof(struct nccl_ofi_gin_gdaki_mr_handle);
		size_t rkeys_bytes = (size_t)nranks * sizeof(__be32);
		size_t pad = (rkeys_bytes % 8 == 0) ? 0 : 4;
		size_t vas_bytes = (size_t)nranks * sizeof(uint64_t);
		size_t handle_size = hdr_size + rkeys_bytes + pad + vas_bytes;

		auto *gdaki_handle = static_cast<struct nccl_ofi_gin_gdaki_mr_handle *>(
			calloc(1, handle_size));
		if (!gdaki_handle) {
			fi_close(&mr->fid);
			throw std::runtime_error("calloc mr_handle failed");
		}

		__be32 *rkeys_tail = reinterpret_cast<__be32 *>(
			reinterpret_cast<uintptr_t>(gdaki_handle) + hdr_size);
		uint64_t *vas_tail = reinterpret_cast<uint64_t *>(
			reinterpret_cast<uintptr_t>(gdaki_handle) +
			hdr_size + rkeys_bytes + pad);

		gdaki_handle->rkeys = rkeys_tail;   /* points INSIDE same alloc */
		gdaki_handle->lkey = (__be32)lkey_val;
		gdaki_handle->nranks = nranks;
		for (int i = 0; i < nranks; i++) {
			rkeys_tail[i] = (__be32)all_rkeys[i];
			vas_tail[i] = all_vas[i];
		}

		/* Step 5: wrap + publish. The wrapper shape
		 *   { void *mr_opaque; mr_handle *handle; }
		 * matches ncclGinGdakiMrRegDevice in the override. */
		auto *reg = new (std::nothrow) struct nccl_ofi_gin_gdaki_mr_reg();
		if (!reg) {
			free(gdaki_handle);
			fi_close(&mr->fid);
			throw std::runtime_error("new mr_reg failed");
		}
		reg->mr = mr;
		reg->handle = gdaki_handle;

		*ginHandle = reg;  /* OVERRIDE the proxy ginHandle */
		return ncclSuccess;

	} catch (const std::exception &e) {
		NCCL_OFI_WARN("gin GDAKI: regMrSym failed: %s", e.what());
		return ncclSystemError;
	}
}

static ncclResult_t nccl_ofi_gin_gdaki_regMrSym(void *collComm, void *data, size_t size,
						int type, uint64_t mrFlags,
						void **mhandle, void **ginHandle)
{
	return gdaki_reg_mr_common(collComm, data, size, type, mrFlags,
				   /*dmabuf_fd=*/-1, /*dmabuf_offset=*/0,
				   mhandle, ginHandle);
}

static ncclResult_t nccl_ofi_gin_gdaki_regMrSymDmaBuf(void *collComm, void *data, size_t size,
						      int type, uint64_t offset, int fd,
						      uint64_t mrFlags, void **mhandle,
						      void **ginHandle)
{
	return gdaki_reg_mr_common(collComm, data, size, type, mrFlags,
				   fd, offset, mhandle, ginHandle);
}

static ncclResult_t nccl_ofi_gin_gdaki_deregMrSym(void *collComm, void *mhandle)
{
	/* Proxy deregistration on mhandle (which remains the proxy's symm handle).
	 * The GDAKI-side wrapper *ginHandle was published separately and NCCL
	 * has its own lifecycle for that pointer. */
	return nccl_ofi_gin_deregMrSym(collComm, mhandle);
}

#else  /* !HAVE_EFA_DP_DIRECT || !HAVE_DECL_FI_EFA_GDA_OPS */

static ncclResult_t nccl_ofi_gin_gdaki_regMrSym(void *collComm, void *data, size_t size,
						int type, uint64_t mrFlags,
						void **mhandle, void **ginHandle)
{
	return nccl_ofi_gin_regMrSym(collComm, data, size, type, mrFlags,
				     mhandle, ginHandle);
}

static ncclResult_t nccl_ofi_gin_gdaki_regMrSymDmaBuf(void *collComm, void *data, size_t size,
						      int type, uint64_t offset, int fd,
						      uint64_t mrFlags, void **mhandle,
						      void **ginHandle)
{
	return nccl_ofi_gin_regMrSymDmaBuf(collComm, data, size, type,
					   offset, fd, mrFlags, mhandle, ginHandle);
}

static ncclResult_t nccl_ofi_gin_gdaki_deregMrSym(void *collComm, void *mhandle)
{
	return nccl_ofi_gin_deregMrSym(collComm, mhandle);
}

#endif /* HAVE_EFA_DP_DIRECT && HAVE_DECL_FI_EFA_GDA_OPS */

static ncclResult_t nccl_ofi_gin_gdaki_queryLastError(void *ginCtx, bool *hasError)
{
	(void)ginCtx;
	*hasError = false;
	return ncclSuccess;
}

/*
 * GDAKI plugin. Shared APIs (init, devices, listen, connect, regMrSym,
 * regMrSymDmaBuf, deregMrSym, closeColl, closeListen, ginProgress, finalize)
 * are copied from the proxy plugin at init time by nccl_ofi_gin_api.cpp.
 * GDAKI-specific entry points live above. iput/iputSignal/iget/iflush/test
 * are nullptr — no CPU involvement in GDAKI mode.
 */
ncclGin_v13_t nccl_ofi_gin_gdaki_plugin = {
	.name = "Libfabric_GDAKI",
	.init = nullptr,
	.devices = nullptr,
	.getProperties = nccl_ofi_gin_gdaki_get_properties,
	.listen = nullptr,
	.connect = nullptr,
	.createContext = nccl_ofi_gin_gdaki_createContext,
	.regMrSym = nccl_ofi_gin_gdaki_regMrSym,
	.regMrSymDmaBuf = nccl_ofi_gin_gdaki_regMrSymDmaBuf,
	.deregMrSym = nccl_ofi_gin_gdaki_deregMrSym,
	.destroyContext = nccl_ofi_gin_gdaki_destroyContext,
	.closeColl = nullptr,
	.closeListen = nullptr,
	.iput = nullptr,
	.iputSignal = nullptr,
	.iget = nullptr,
	.iflush = nullptr,
	.test = nullptr,
	.ginProgress = nullptr,
	.queryLastError = nccl_ofi_gin_gdaki_queryLastError,
	.finalize = nullptr
};
