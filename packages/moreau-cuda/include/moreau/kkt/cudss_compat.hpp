#pragma once

#include <cudss.h>

/**
 * cuDSS version compatibility (0.7 vs 0.8).
 *
 * cuDSS 0.8 introduced a dedicated cudssDataType_t for matrix creation
 * (no implicit conversion from cudaDataType_t), split the single CSR
 * index type into separate offset and index types, and replaced the
 * pivot types CUDSS_PIVOT_COL/ROW with a new enum where
 * CUDSS_PIVOT_DIAGONAL is the strategy for symmetric indefinite
 * matrices.
 */
#if CUDSS_VERSION >= 800
#define MOREAU_CUDSS_R_64F CUDSS_R_64F
// offsetType, indexType, valueType for cudssMatrixCreateCsr
#define MOREAU_CUDSS_CSR_I64_F64 CUDSS_R_64I, CUDSS_R_64I, CUDSS_R_64F
#define MOREAU_CUDSS_PIVOT_ON CUDSS_PIVOT_DIAGONAL
#else
#define MOREAU_CUDSS_R_64F CUDA_R_64F
// indexType, valueType for cudssMatrixCreateCsr
#define MOREAU_CUDSS_CSR_I64_F64 CUDA_R_64I, CUDA_R_64F
#define MOREAU_CUDSS_PIVOT_ON CUDSS_PIVOT_COL
#endif
