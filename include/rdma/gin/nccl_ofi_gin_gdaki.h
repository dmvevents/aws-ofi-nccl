/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 */

#ifndef NCCL_OFI_GIN_GDAKI_H_
#define NCCL_OFI_GIN_GDAKI_H_

#include "nccl_ofi.h"

/*
 * Return true if GDAKI mode is requested via OFI_NCCL_GIN_GDAKI=1 env var.
 */
bool nccl_ofi_gin_gdaki_enabled();

/*
 * The GDAKI plugin. Shared functions (init, devices, listen, connect)
 * are nullptr and get copied from the proxy plugin at init time.
 */
extern ncclGin_v13_t nccl_ofi_gin_gdaki_plugin;

/*
 * Proxy-side regMr / deregMr forward decls. GDAKI regMr wraps these:
 * proxy impl runs first for the bootstrap + mhandle path; then
 * nccl_ofi_gin_gdaki_regMrSym appends efa-direct registration +
 * per-peer rkey/VA allgather + publishes its own ginHandle.
 * These are DEFINED in nccl_ofi_gin_api.cpp (non-static).
 */
ncclResult_t nccl_ofi_gin_regMrSym(void *collComm, void *data, size_t size, int type,
				   uint64_t mrFlags, void **mhandle, void **ginHandle);
ncclResult_t nccl_ofi_gin_regMrSymDmaBuf(void *collComm, void *data, size_t size, int type,
					 uint64_t offset, int fd, uint64_t mrFlags,
					 void **mhandle, void **ginHandle);
ncclResult_t nccl_ofi_gin_deregMrSym(void *collComm, void *mhandle);

#endif /* NCCL_OFI_GIN_GDAKI_H_ */
