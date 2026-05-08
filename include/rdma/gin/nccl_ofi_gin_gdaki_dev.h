/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Device-visible types for the GIN GDAKI data path on EFA.
 *
 * This header defines the layout of the device handle returned from
 * createContext and consumed by kernel-side GIN Put/PutValue/Signal paths.
 *
 * ## Binary-layout compatibility
 *
 * Bytes 0..71 of nccl_ofi_gin_gdaki_dev_handle are binary-layout-identical
 * to NVIDIA's struct ncclGinGdakiGPUContext as declared in
 *   nccl_device/gin/gdaki/gin_gdaki_device_host_common.h
 *
 * | Offset | Field                         | Purpose                     |
 * | ------ | ----------------------------- | --------------------------- |
 * | 0      | gdqp                          | QP pointer (EFA: our qp)    |
 * | 8      | companion_gdqp                | Unused (nullptr)            |
 * | 16..39 | counters_table.{buffer,...}   | Unused (zero) for v1        |
 * | 40..63 | signals_table.{buffer,...}    | Populated by createContext  |
 * | 64     | sink_buffer_lkey              | Local sink MR lkey          |
 *
 * The reason for this constraint is in docs/ROOT-CAUSE-struct-alias-gap-2026-05-08.md:
 * DeepEP and NVIDIA gin device code read through a
 * reinterpret_cast<ncclGinGdakiGPUContext*>(ctx.handle), so the first 72 bytes
 * of the struct this plugin returns must match NVIDIA's layout byte-for-byte.
 *
 * Bytes 72+ hold EFA-specific extensions (per-peer address handles, remote
 * QPNs, q-keys) that our custom device code reads to build WQEs. NVIDIA code
 * never reads past byte 71.
 *
 * Keep this header free of libfabric and plugin-internal types so it can be
 * included from both host (C++17) and device (CUDA) translation units.
 */

#ifndef NCCL_OFI_GIN_GDAKI_DEV_H_
#define NCCL_OFI_GIN_GDAKI_DEV_H_

#include <stddef.h>
#include <stdint.h>
#include <linux/types.h>

/* Forward declarations of efa-dp-direct types. efa_cuda_dp.h is only pulled
 * in when HAVE_EFA_DP_DIRECT is set, because the header transitively includes
 * <cuda_runtime.h>. */
struct efa_cuda_qp;
struct efa_cuda_cq;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Per-buffer global table. Mirrors NVIDIA's
 * ncclGinGdakiGlobalGPUBufferTable<T> with an explicit offset field to match
 * the NVIDIA layout exactly (their template has implicit tail padding that
 * the device code reads via .offset).
 */
struct nccl_ofi_gin_gdaki_buffer_table_u64 {
	uint64_t *buffer;  /* GPU-resident uint64_t[N]                     */
	__be32   *rkeys;   /* GPU-resident __be32[nranks] per-peer rkeys   */
	__be32    lkey;    /* lkey of buffer on the local efa-direct domain*/
	uint32_t  offset;  /* byte offset within the parent MR (usually 0) */
};

/**
 * GDAKI memory registration handle returned as the ginHandle from regMrSym.
 * Allocated in host memory. The kernel receives this as a ncclGinWindow_t
 * (a plain void*). Binary-layout identical to NVIDIA's ncclGinGdakiMemHandle
 * plus a trailing flexible-array rkeys table.
 */
struct nccl_ofi_gin_gdaki_mr_handle {
	__be32 *rkeys;          /* GPU-resident __be32[nranks] per-peer rkeys  */
	__be32  lkey;           /* lkey on the local efa-direct domain         */
	int32_t nranks;          /* size of the rkeys array                    */
	/* rkeys[] storage follows in the same allocation (for free()).         */
};

/**
 * Device-visible handle returned from createContext.
 *
 * Allocated in GPU memory. The pointer is stored in
 * ncclNetDeviceHandle_v11_t::handle.
 *
 * Bytes 0..71: binary-layout-compat with NVIDIA's ncclGinGdakiGPUContext.
 * Bytes 72+:   EFA-specific extension, readable only by our device code.
 *
 * All pointer members refer to GPU-accessible memory.
 */
struct nccl_ofi_gin_gdaki_dev_handle {
	/* ---- NVIDIA-compat prefix (bytes 0..71) ---- */

	/* [0]  QP object. For EFA-SRD we share one qp across peers (SRD is
	 *      connectionless; per-peer addressing lives in address_handles
	 *      below). NVIDIA's device code indexes "gdqp + peer" but our
	 *      override overrides that path and reads address_handles[peer]
	 *      instead. */
	struct efa_cuda_qp *gdqp;                                      /* [0]  */

	/* [8]  Unused for v1. Kept for layout compat with NVIDIA. */
	struct efa_cuda_qp *companion_gdqp;                            /* [8]  */

	/* [16] counters_table. Zero-initialized for v1 (barrier uses signals).*/
	struct nccl_ofi_gin_gdaki_buffer_table_u64 counters_table;     /* [16] */

	/* [40] signals_table. DeepEP reads signals_table.buffer[sigId] via __ldg.
	 *      MUST be populated with a real GPU uint64_t[nSignals] + allgathered
	 *      per-peer rkeys, or cross-node barrier will take a CUDA illegal
	 *      memory access fault. This was the 2026-05-08 root cause. */
	struct nccl_ofi_gin_gdaki_buffer_table_u64 signals_table;      /* [40] */

	/* [64] Local sink buffer lkey. Used as SGE lkey for signal WRITEs that
	 *      don't carry payload. */
	__be32 sink_buffer_lkey;                                       /* [64] */

	/* [68] Padding to keep the NVIDIA-compat prefix at exactly 72 bytes,
	 *      so bytes 72+ begin on an 8-byte boundary for the pointer fields
	 *      that follow. */
	uint32_t _pad_nvidia_prefix;                                   /* [68] */

	/* ---- EFA extension (bytes 72+). Read ONLY by our device code. ---- */

	/* [72] Per-peer EFA address-handle numbers. */
	uint16_t *address_handles;                                     /* [72] */

	/* [80] Per-peer remote QP numbers. */
	uint16_t *remote_qpns;                                         /* [80] */

	/* [88] Per-peer Q keys. */
	uint32_t *qkey;                                                /* [88] */

	/* [96] Per-peer base VAs of the remote signals buffer. Sender uses
	 *      peer_signal_vas[peer] + signalId*sizeof(uint64_t) as the WR
	 *      remote address. If the efa-direct domain publishes offsets
	 *      relative to MR base (FI_MR_PROV_KEY semantics), each entry
	 *      holds the peer's signals_buffer MR base.  */
	uint64_t *peer_signal_vas;                                     /* [96] */

	/* [104] Cached nranks + rank for kernel-side bounds checks. */
	int32_t nranks;                                                /* [104]*/
	int32_t rank;                                                  /* [108]*/

	/* [112] EFA CQ (for kernel-side completion polling). Same lifetime as
	 *       gdqp above. */
	struct efa_cuda_cq *cq;                                        /* [112]*/

	/* [120] Reserved for future use. Total size: 128 bytes. */
	void *_reserved0;                                              /* [120]*/
};

#ifdef __cplusplus
} /* extern "C" */

/* Binary-layout guarantees for the NVIDIA-compat prefix. If any of these
 * fire, the struct has drifted from NVIDIA's ncclGinGdakiGPUContext and
 * cross-node DeepEP will fault. */
static_assert(sizeof(struct nccl_ofi_gin_gdaki_buffer_table_u64) == 24,
	      "buffer_table_u64 must be 24 bytes for NVIDIA layout compat");
static_assert(offsetof(struct nccl_ofi_gin_gdaki_dev_handle, gdqp) == 0,
	      "gdqp must be at offset 0");
static_assert(offsetof(struct nccl_ofi_gin_gdaki_dev_handle, companion_gdqp) == 8,
	      "companion_gdqp must be at offset 8");
static_assert(offsetof(struct nccl_ofi_gin_gdaki_dev_handle, counters_table) == 16,
	      "counters_table must be at offset 16");
static_assert(offsetof(struct nccl_ofi_gin_gdaki_dev_handle, signals_table) == 40,
	      "signals_table must be at offset 40 (NVIDIA layout — DeepEP reads here)");
static_assert(offsetof(struct nccl_ofi_gin_gdaki_dev_handle, sink_buffer_lkey) == 64,
	      "sink_buffer_lkey must be at offset 64");
static_assert(offsetof(struct nccl_ofi_gin_gdaki_dev_handle, address_handles) == 72,
	      "EFA extension begins at offset 72 (must be 8-byte aligned)");
static_assert(sizeof(struct nccl_ofi_gin_gdaki_dev_handle) == 128,
	      "dev_handle total size is 128 bytes");
#endif

#endif /* NCCL_OFI_GIN_GDAKI_DEV_H_ */
