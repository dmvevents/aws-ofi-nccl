/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Host-side state for the GIN GDAKI data path on EFA.
 *
 * Plugin-internal. This header is NOT included from device code.
 * Kept separate from nccl_ofi_gin_gdaki_dev.h so that the device header
 * stays free of libfabric and plugin-internal types.
 */

#ifndef NCCL_OFI_GIN_GDAKI_CTX_H_
#define NCCL_OFI_GIN_GDAKI_CTX_H_

#include "config.h"

#include <stdint.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>

#include "rdma/gin/nccl_ofi_gin_gdaki_dev.h"

/* Opaque forward decls. efa_cuda_dp.h and fi_ext_efa.h are pulled in only
 * in the .cpp file (gated by HAVE_EFA_DP_DIRECT && HAVE_DECL_FI_EFA_GDA_OPS).
 */
struct fi_efa_ops_gda;

/**
 * Host-side state associated with one createContext call.
 *
 * createContext returns this pointer as the opaque ginCtx.
 * destroyContext consumes it to tear everything down. Every field is
 * zeroed on allocation so teardown is safe on partial init.
 */
struct nccl_ofi_gin_gdaki_context {
	/* ---- libfabric efa-direct resources ---- */

	struct fid_fabric *ofi_fabric;
	struct fid_domain *ofi_domain;
	struct fid_ep     *ofi_ep;
	struct fid_av     *ofi_av;
	struct fid_cq     *ofi_cq;

	/* fi_info used to open the above. Held for teardown + attr queries.  */
	struct fi_info    *ofi_info;

	/* GDA ops function table (FI_EFA_GDA_OPS). Retained for get_mr_lkey
	 * in regMrSym and for any later attribute queries. */
	struct fi_efa_ops_gda *gda_ops;

	/* ---- GPU buffers owned by this context ---- */

	/* signals_buffer: uint64_t[nSignals] in GPU memory. The buffer
	 * DeepEP reads from as signals_table.buffer[sigId]. */
	void          *signals_buffer_dev;
	size_t         signals_buffer_size;
	struct fid_mr *signals_mr;       /* efa-direct MR for the above  */
	__be32         signals_lkey;     /* lkey on local domain         */

	/* sink_buffer: 8 bytes in GPU memory; used as SGE source for signal
	 * WRITEs that don't carry payload. */
	void          *sink_buffer_dev;
	struct fid_mr *sink_mr;
	__be32         sink_lkey;

	/* GPU-resident per-peer tables populated in createContext and
	 * pointed to from dev_handle. */
	__be32   *d_signals_rkeys;    /* __be32[nranks]  */
	uint64_t *d_peer_signal_vas;  /* uint64_t[nranks]*/
	uint16_t *d_address_handles;  /* uint16_t[nranks]*/
	uint16_t *d_remote_qpns;      /* uint16_t[nranks]*/
	uint32_t *d_qkeys;            /* uint32_t[nranks]*/
	uint32_t *d_peer_locks;       /* uint32_t[nranks] spinlocks */

	/* ---- efa-dp-direct QP / CQ device objects ---- */

	/* We use one shared QP for the whole context (SRD is connectionless,
	 * peer addressing lives in d_address_handles above). */
	struct efa_cuda_qp *d_qp;
	struct efa_cuda_cq *d_cq;

	/* ---- Device-visible handle (the pointer DeepEP casts) ---- */

	struct nccl_ofi_gin_gdaki_dev_handle *d_handle;

	/* ---- Config + identity ---- */

	int nSignals;
	int nCounters;
	int nranks;
	int rank;
};

/**
 * Host-side wrapper for a GDAKI memory registration.
 *
 * Returned as the ginHandle from regMrSym. Contains the efa-direct fid_mr
 * for deregistration plus the device-visible handle holding lkey + rkeys.
 */
struct nccl_ofi_gin_gdaki_mr_reg {
	struct fid_mr *mr;
	/* Heap block: header + rkeys[nranks] tail. */
	struct nccl_ofi_gin_gdaki_mr_handle *handle;
};

#endif /* NCCL_OFI_GIN_GDAKI_CTX_H_ */
