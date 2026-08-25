/**
 * @file kernels.cuh
 * @brief CUDA kernels for matrix equilibration
 */

#pragma once

#include "moreau/matrix/csr.hpp"
#include "moreau/vector/vector.hpp"
#include <cuda_runtime.h>
#include <cstdint>

namespace moreau {

// Computes symmetric row infinity-norms of P stored as full symmetric matrix, for a batch
void compute_batch_row_norms_P(
    const CSR& P,
    const int64_t* d_row_of,
    BatchedVector& d_row_norms,
    cudaStream_t stream
);

void compute_batch_col_norms_A_noreset(
    const CSR& A,
    BatchedVector& d_col_norms,
    cudaStream_t stream
);

void compute_batch_row_norms_A(
    const CSR& A,
    const int64_t* d_row_of,
    BatchedVector& d_row_norms,
    cudaStream_t stream
);

// Process dwork: handle zeros, rsqrt, and clip
void process_dwork(
    BatchedVector& d_dwork,
    BatchedVector& d_d,
    double scale_min,
    double scale_max,
    int64_t n,
    int64_t batch_size,
    cudaStream_t stream = 0
);

// Process ework: handle zeros, rsqrt, and clip
void process_ework(
    BatchedVector& d_ework,
    BatchedVector& d_e,
    double scale_min,
    double scale_max,
    int64_t m,
    int64_t batch_size,
    cudaStream_t stream = 0
);

// Fused equilibration: process both dwork and ework in one kernel launch
void process_equilibration(
    BatchedVector& d_dwork,
    BatchedVector& d_d,
    BatchedVector& d_ework,
    BatchedVector& d_e,
    double scale_min,
    double scale_max,
    int64_t n,
    int64_t m,
    int64_t batch_size,
    cudaStream_t stream = 0
);

// Left-only scale sparse matrix
void lscale(
    int64_t nnz,
    int64_t nrows,
    int64_t batch_size,
    const int64_t* d_row_of,
    double* d_vals,
    const double* d_L,
    cudaStream_t stream = 0
);

// Hadamard (element-wise) product
void hadamard(
    double* d_array,
    const double* d_scale_array,
    int64_t n,
    int64_t batch_size,
    cudaStream_t stream = 0
);

// Left-right scale sparse matrix
void lrscale(
    int64_t nnz,
    int64_t nrows,
    int64_t ncols,
    int64_t batch_size,
    const int64_t* d_row_of,
    const int64_t* d_colind,
    double* d_vals,
    const double* d_L,
    const double* d_R,
    cudaStream_t stream = 0
);


// Helper function to scale problem data (matches Clarabel's scale_data)
// Version without bounds
void scale_data(
    // Matrix P data
    CSR& P,
    const int64_t* d_P_row_of,
    // Matrix A data
    CSR& A,
    const int64_t* d_A_row_of,
    // Vectors q and b
    BatchedVector& q,
    BatchedVector& b,
    // Scaling vectors
    const BatchedVector* d,  // Can be nullptr if not using
    const BatchedVector* e,
    cudaStream_t stream = 0
);

void conditional_scale_cost(
    CSR& P,
    BatchedVector& q,
    BatchedVector& c,
    const BatchedVector& d_mean_col_norm_P,
    const BatchedVector& d_inf_norm_q,
    double scale_min,
    double scale_max,
    cudaStream_t stream = 0
);

void rectify_cone_equilibration(
    BatchedVector& d_ework,
    const BatchedVector& d_e,
    int64_t cone_start,
    int64_t cone_size,
    cudaStream_t stream = 0
);

// Rectify SOC cone equilibration for variable-size SOC cones:
// ework[i] = mean(e_cone) / e[i] for each SOC cone block
void rectify_soc_cone_equilibration(
    BatchedVector& d_ework,
    const BatchedVector& d_e,
    int64_t cone_start,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_dims,
    int64_t numSocCones,
    cudaStream_t stream = 0,
    const int64_t* d_soc_sz_offsets = nullptr
);

// Rectify the Ruiz `d` vector on direct-x SOC cones so that each
// cone's `x[J]` sees a uniform scalar scaling. For each direct-x SOC
// cone computes `g = geom_mean(d[indices])` (via log/exp to avoid
// overflow) and sets `dwork[idx] = g / d[idx]` for `idx ∈ indices`,
// leaving all other entries of `dwork` untouched (caller fills to 1.0
// before the call). Nonneg direct-x cones are skipped (per-entry
// scaling already preserves the orthant).
//
// Mirror of the CPU direct-x equilibration loop in
// `problemdata.rs::equilibrate` (lines 309-342) — CPU uses geometric
// mean there (different from the arithmetic mean used by slack SOC
// `rectify_equilibration`; see CPU socone.rs::rectify_equilibration).
// One block per (batch, cone), single-threaded; dense SOC dim ≤ 4.
void rectify_xcone_soc_d_equilibration(
    BatchedVector& d_dwork,
    const BatchedVector& d_d,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t numXCones,
    int64_t n,
    cudaStream_t stream = 0
);

// Rectify PSD cone equilibration for variable-size PSD cones:
// ework[i] = mean(e_cone) / e[i] for each PSD cone block
// d_psd_dims are matrix dimensions (svec_dim = dim*(dim+1)/2)
void rectify_psd_cone_equilibration(
    BatchedVector& d_ework,
    const BatchedVector& d_e,
    int64_t cone_start,
    const int64_t* d_psd_dims,
    int64_t numPsdCones,
    cudaStream_t stream = 0,
    const int64_t* d_psd_sz_offsets = nullptr
);

// Rectify GenPowerCone equilibration for variable-size cone blocks:
// ework[i] = mean(e_cone) / e[i] for each GenPowerCone block
void rectify_genpow_cone_equilibration(
    BatchedVector& d_ework,
    const BatchedVector& d_e,
    int64_t cone_start,
    const int64_t* d_genPowerOffsets,
    const int64_t* d_genPowerDim1s,
    const int64_t* d_genPowerDim2s,
    int64_t numGenPowerCones,
    cudaStream_t stream = 0,
    const int64_t* d_genPowerSzOffsets = nullptr
);

// Compute element-wise reciprocal: output[i] = 1.0 / input[i]
void compute_reciprocal(
    const BatchedVector& d_input,
    BatchedVector& d_output,
    cudaStream_t stream = 0
);

void set_array(
    BatchedVector& d_array,
    double value,
    int64_t start_idx,
    int64_t length,
    int64_t batch_size,
    cudaStream_t stream = 0
);

} // namespace moreau
