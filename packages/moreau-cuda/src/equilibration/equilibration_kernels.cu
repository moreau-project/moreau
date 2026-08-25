/**
 * @file kernels.cu
 * @brief CUDA kernel implementations for matrix equilibration
 */

#include "moreau/equilibration/equilibration_kernels.cuh"
#include "moreau/matrix/csr.hpp"
#include "moreau/vector/vector.hpp"
#include "moreau/cuda/utils.cuh"
#include "moreau/profiling/profiler.hpp"
#include <cassert>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>

// Compute element-wise reciprocal
__global__ void compute_reciprocal_kernel(
    const double* __restrict__ input,
    double* __restrict__ output,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (idx >= n || b >= batch_size) return;
    
    int64_t pos = b * n + idx;
    output[pos] = 1.0 / input[pos];
}

// CUDA kernel for set_array
__global__ void set_array_kernel(
    double* __restrict__ d_array,
    double value,
    int64_t start_idx,
    int64_t length,
    int64_t n,
    int64_t batch_size
) {
    int b = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < length && b < batch_size) {
        // row-major: offset (start_idx + tid) in batch b
        int64_t idx = b * n + (start_idx + tid);
        d_array[idx] = value;
    }
}

// --- device kernels ---
// For P (full symmetric matrix) - INFINITY NORM
// Each entry only contributes to the row it's in (no symmetric contribution needed
// since both (i,j) and (j,i) are explicitly stored in the full matrix)
__global__ void sym_row_norm_inf_full_batch(
    int64_t n, int64_t batch_size, int64_t nnzP,
    const int64_t* __restrict__ row_of,
    const int64_t* __restrict__ colind,
    const double* __restrict__ vals,
    double* __restrict__ row_norms)
{
    int64_t e = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (e >= nnzP || b >= batch_size) return;

    int64_t i = row_of[e];
    double abs_val = fabs(vals[b * nnzP + e]);

    // Full matrix: each entry only contributes to its own row
    atomicMaxDouble(&row_norms[b * n + i], abs_val);
}

__global__ void csr_col_norm_inf_nnz_batch(
    int64_t n_out,        // ← Dimension of col_norms (5 in your case)
    int64_t n_cols,       // ← Number of columns in A (13 in your case)  
    int64_t batch_size, 
    int64_t nnzA,
    const int64_t* __restrict__ colind,
    const double* __restrict__ vals,
    double* __restrict__ col_norms)
{
    int64_t e = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (e >= nnzA || b >= batch_size) return;

    int64_t j = colind[e];
    double abs_val = fabs(vals[b * nnzA + e]);

    atomicMaxDouble(&col_norms[b * n_out + j], abs_val);  // Use n_out here!
}

// For A (general m x n) - row INFINITY norms, nnz-level parallelism
__global__ void csr_row_norm_inf_nnz_batch(
    int64_t m, int64_t batch_size, int64_t nnzA,
    const int64_t* __restrict__ row_of,
    const double* __restrict__ vals,
    double* __restrict__ row_norms)
{
    int64_t e = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (e >= nnzA || b >= batch_size) return;

    int64_t i = row_of[e];
    double abs_val = fabs(vals[b * nnzA + e]);

    atomicMaxDouble(&row_norms[b * m + i], abs_val);
}


// Process dwork: zero->1, rsqrt, then clip based on d
__global__ void process_dwork_kernel(
    double* __restrict__ dwork,
    double* __restrict__ d,
    double scale_min,
    double scale_max,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (idx >= n || b >= batch_size) return;

    int64_t pos = b * n + idx;
    
    // Step 1: Zero rows/columns should not get scaled (set to 1.0)
    double val = dwork[pos];
    if (val == 0.0) {
        val = 1.0;
    }
    
    // Step 2: Reciprocal square root
    double scale = rsqrt(val);  // 1/sqrt(val)
    
    // Step 3: Clip based on cumulative scaling with d
    double min_val = scale_min / d[pos];
    double max_val = scale_max / d[pos];
    scale = fmax(min_val, fmin(scale, max_val));
    
    // Store result and update d and q
    dwork[pos] = scale;
    d[pos] *= scale;
}

// Process ework: zero->1, rsqrt, then clip based on e
__global__ void process_ework_kernel(
    double* __restrict__ ework,
    double* __restrict__ e,
    double scale_min,
    double scale_max,
    int64_t m,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b_idx = blockIdx.y;
    if (idx >= m || b_idx >= batch_size) return;

    int64_t pos = b_idx * m + idx;

    // Step 1: Zero rows/columns should not get scaled (set to 1.0)
    double val = ework[pos];
    if (val == 0.0) {
        val = 1.0;
    }

    // Step 2: Reciprocal square root
    double scale = rsqrt(val);  // 1/sqrt(val)

    // Step 3: Clip based on cumulative scaling with e
    double min_val = scale_min / e[pos];
    double max_val = scale_max / e[pos];
    scale = fmax(min_val, fmin(scale, max_val));

    // Store result and update e and b
    ework[pos] = scale;
    e[pos] *= scale;
}

// Fused equilibration: process both dwork and ework in one kernel launch
// Each thread handles one element from either d or e vector
__global__ void process_equilibration_kernel(
    double* __restrict__ dwork,
    double* __restrict__ d,
    double* __restrict__ ework,
    double* __restrict__ e,
    double scale_min,
    double scale_max,
    int64_t n,
    int64_t m,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (b >= batch_size) return;

    // Process dwork (size n)
    if (idx < n) {
        int64_t pos = b * n + idx;
        double val = dwork[pos];
        if (val == 0.0) val = 1.0;
        double scale = rsqrt(val);
        double min_val = scale_min / d[pos];
        double max_val = scale_max / d[pos];
        scale = fmax(min_val, fmin(scale, max_val));
        dwork[pos] = scale;
        d[pos] *= scale;
    }

    // Process ework (size m)
    if (idx < m) {
        int64_t pos = b * m + idx;
        double val = ework[pos];
        if (val == 0.0) val = 1.0;
        double scale = rsqrt(val);
        double min_val = scale_min / e[pos];
        double max_val = scale_max / e[pos];
        scale = fmax(min_val, fmin(scale, max_val));
        ework[pos] = scale;
        e[pos] *= scale;
    }
}


// Left-only scale sparse matrix in CSR format (nnz-level)
// Performs: vals[b,e] *= L[b,i] where i = row_of[e]
__global__ void lscale_csr_kernel(
    int64_t nnz,
    int64_t batch_size,
    int64_t nrows,
    const int64_t* __restrict__ row_of,
    double* __restrict__ vals,
    const double* __restrict__ L)  // [batch_size * nrows]
{
    int64_t e = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (e >= nnz || b >= batch_size) return;

    int64_t i = row_of[e];
    vals[b * nnz + e] *= L[b * nrows + i];
}

// Hadamard (element-wise) product: array *= scale_array
__global__ void hadamard_kernel(
    double* __restrict__ array,
    const double* __restrict__ scale_array,
    int64_t n,
    int64_t batch_size)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (idx >= n || b >= batch_size) return;
    
    int64_t pos = b * n + idx;
    array[pos] *= scale_array[pos];
}

// Left-right scale sparse matrix in CSR format (nnz-level)
// Performs: vals[b,e] *= L[b,i] * R[b,j] where i = row_of[e], j = colind[e]
__global__ void lrscale_csr_kernel(
    int64_t nnz,
    int64_t batch_size,
    int64_t nrows,
    int64_t ncols,
    const int64_t* __restrict__ row_of,
    const int64_t* __restrict__ colind,
    double* __restrict__ vals,
    const double* __restrict__ L,  // [batch_size * nrows]
    const double* __restrict__ R)  // [batch_size * ncols]
{
    int64_t e = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t b = blockIdx.y;
    if (e >= nnz || b >= batch_size) return;

    int64_t i = row_of[e];
    int64_t j = colind[e];

    vals[b * nnz + e] *= L[b * nrows + i] * R[b * ncols + j];
}

__global__ void conditional_scale_cost_kernel(
    double* __restrict__ P_vals,
    double* __restrict__ q,
    double* __restrict__ c,
    const double* __restrict__ mean_col_norm_P,
    const double* __restrict__ inf_norm_q,
    double scale_min,
    double scale_max,
    int64_t nnzP,
    int64_t n,
    int64_t batch_size)
{
    int64_t b = blockIdx.x;
    if (b >= batch_size) return;
    
    double mean_norm = mean_col_norm_P[b];
    double inf_q = inf_norm_q[b];
    
    if (mean_norm == 0.0 || inf_q == 0.0) return;
    
    double scale_cost = fmax(inf_q, mean_norm);
    double ctmp = 1.0 / scale_cost;
    ctmp = fmax(scale_min / c[b], fmin(ctmp, scale_max / c[b]));
    
    // Scale P values for this batch
    for (int64_t i = threadIdx.x; i < nnzP; i += blockDim.x) {
        P_vals[b * nnzP + i] *= ctmp;
    }
    
    // Scale q values for this batch
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        q[b * n + i] *= ctmp;
    }
    
    // Scale c (only first thread)
    if (threadIdx.x == 0) {
        c[b] *= ctmp;
    }
}

// Rectify cone equilibration: ework[i] = mean(e_cone) / e[i]
__global__ void rectify_cone_equilibration_kernel(
    double* __restrict__ ework,
    const double* __restrict__ e,
    int64_t cone_start,
    int64_t cone_size,
    int64_t batch_size,
    int64_t m)
{
    int64_t b = blockIdx.y;
    int64_t local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= batch_size || local_idx >= cone_size) return;
    
    // Each 3-element group is one cone
    int64_t cone_idx = local_idx / 3;
    int64_t elem_idx = local_idx % 3;
    int64_t cone_base = b * m + cone_start + cone_idx * 3;
    
    // Compute mean of this 3-element cone
    double e0 = e[cone_base];
    double e1 = e[cone_base + 1];
    double e2 = e[cone_base + 2];
    double mean = (e0 + e1 + e2) / 3.0;
    
    // Set ework[i] = mean / e[i]
    int64_t global_idx = cone_base + elem_idx;
    double e_val = (elem_idx == 0) ? e0 : ((elem_idx == 1) ? e1 : e2);
    ework[global_idx] = mean / e_val;
}

// Rectify SOC cone equilibration for variable-size SOC blocks:
// ework[i] = mean(e_cone) / e[i] for each SOC cone block
__global__ void rectify_soc_cone_equilibration_kernel(
    double* __restrict__ ework,
    const double* __restrict__ e,
    int64_t cone_start,
    int64_t batch_size,
    int64_t m,
    const int64_t* __restrict__ d_soc_offsets,
    const int64_t* __restrict__ d_soc_dims,
    int64_t numSocCones,
    const int64_t* __restrict__ d_soc_sz_offsets
) {
    int64_t b = blockIdx.y;
    int64_t cone_idx = blockIdx.x;

    if (b >= batch_size || cone_idx >= numSocCones || cone_idx < 0) return;
    if (d_soc_dims == nullptr) return;

    int64_t dim = d_soc_dims[cone_idx];
    if (dim <= 0) return;

    // Use d_soc_sz_offsets for constraint-space access (original positions in s/z)
    const int64_t* offsets = d_soc_sz_offsets ? d_soc_sz_offsets : d_soc_offsets;
    int64_t soc_off = offsets[cone_idx];
    int64_t base = b * m + cone_start + soc_off;

    double mean = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
        mean += e[base + i];
    }
    mean /= static_cast<double>(dim);

    for (int64_t i = 0; i < dim; ++i) {
        ework[base + i] = mean / e[base + i];
    }
}

// PSD cone equilibration rectification: like SOC but dim means matrix dim,
// not svec dim. Group size = dim*(dim+1)/2.
__global__ void rectify_psd_cone_equilibration_kernel(
    double* __restrict__ ework,
    const double* __restrict__ e,
    int64_t cone_start,
    int64_t batch_size,
    int64_t m,
    const int64_t* __restrict__ d_psd_dims,
    int64_t numPsdCones,
    const int64_t* __restrict__ d_psd_sz_offsets
) {
    int64_t b = blockIdx.y;
    int64_t cone_idx = blockIdx.x;

    if (b >= batch_size || cone_idx >= numPsdCones) return;

    int64_t mat_dim = d_psd_dims[cone_idx];
    int64_t svec_dim = mat_dim * (mat_dim + 1) / 2;
    if (svec_dim <= 0) return;

    int64_t sz_off = d_psd_sz_offsets[cone_idx];
    int64_t base = b * m + cone_start + sz_off;

    double mean = 0.0;
    for (int64_t i = 0; i < svec_dim; ++i) {
        mean += e[base + i];
    }
    mean /= static_cast<double>(svec_dim);

    for (int64_t i = 0; i < svec_dim; ++i) {
        double e_val = e[base + i];
        ework[base + i] = (e_val > 0.0) ? mean / e_val : 1.0;
    }
}

// Rectify GenPowerCone equilibration for variable-size cone blocks:
// ework[i] = mean(e_cone) / e[i] for each GenPowerCone block
// Mirrors CPU: rectify_equilibration -> delta = mean(e) / e (scalar equilibration)
__global__ void rectify_genpow_cone_equilibration_kernel(
    double* __restrict__ ework,
    const double* __restrict__ e,
    int64_t cone_start,
    int64_t batch_size,
    int64_t m,
    const int64_t* __restrict__ d_genPowerOffsets,
    const int64_t* __restrict__ d_genPowerDim1s,
    const int64_t* __restrict__ d_genPowerDim2s,
    int64_t numGenPowerCones,
    const int64_t* __restrict__ d_genPowerSzOffsets
) {
    int64_t b = blockIdx.y;
    int64_t cone_idx = blockIdx.x;

    if (b >= batch_size || cone_idx >= numGenPowerCones) return;

    int64_t dim = d_genPowerDim1s[cone_idx] + d_genPowerDim2s[cone_idx];
    if (dim <= 0) return;

    int64_t sz_off = d_genPowerSzOffsets[cone_idx];
    int64_t base = b * m + cone_start + sz_off;

    double mean = 0.0;
    for (int64_t i = 0; i < dim; ++i) {
        mean += e[base + i];
    }
    mean /= static_cast<double>(dim);

    for (int64_t i = 0; i < dim; ++i) {
        ework[base + i] = mean / e[base + i];
    }
}

namespace moreau {
    void compute_batch_row_norms_P(
        const CSR& P,
        const int64_t* d_row_of,
        BatchedVector& d_row_norms,
        cudaStream_t stream)
    {
        if (P.nnz() == 0 || P.n() == 0) {
            d_row_norms.setToConstant(0.0, stream);
            return;
        }
        d_row_norms.setToConstant(0.0, stream);

        dim3 blk(256, 1, 1);
        dim3 grd((P.nnz() + blk.x - 1) / blk.x, P.batchSize(), 1);
        // P is stored as full symmetric matrix
        MOREAU_KERNEL_LAUNCH(sym_row_norm_inf_full_batch, grd, blk, 0, stream,
            P.n(), P.batchSize(), P.nnz(), d_row_of, P.colIndices(), P.values(), d_row_norms.data());

    }

    void compute_batch_col_norms_A_noreset(
        const CSR& A,
        BatchedVector& d_col_norms,
        cudaStream_t stream)
    {
        if (A.nnz() == 0 || A.m() == 0) {
            d_col_norms.setToConstant(0.0, stream);
            return;
        }
        dim3 blk(256, 1, 1);
        dim3 grd((A.nnz() + blk.x - 1) / blk.x, A.batchSize(), 1);
        MOREAU_KERNEL_LAUNCH(csr_col_norm_inf_nnz_batch, grd, blk, 0, stream,
            d_col_norms.n(),
            A.n(),
            A.batchSize(),
            A.nnz(),
            A.colIndices(),
            A.values(),
            d_col_norms.data());
    
    }

    void compute_batch_row_norms_A(
        const CSR& A,
        const int64_t* d_row_of,
        BatchedVector& d_row_norms,
        cudaStream_t stream)
    {
        if (A.nnz() == 0 || A.n() == 0) {
            d_row_norms.setToConstant(0.0, stream);
            return;
        }
        d_row_norms.setToConstant(0.0, stream);

        dim3 blk(256, 1, 1);
        dim3 grd((A.nnz() + blk.x - 1) / blk.x, A.batchSize(), 1);
        MOREAU_KERNEL_LAUNCH(csr_row_norm_inf_nnz_batch, grd, blk, 0, stream,
            A.n(), A.batchSize(), A.nnz(), d_row_of, A.values(), d_row_norms.data());

    }

    void process_dwork(
        BatchedVector& d_dwork,
        BatchedVector& d_d,
        double scale_min,
        double scale_max,
        int64_t n,
        int64_t batch_size,
        cudaStream_t stream)
    {
        if (n == 0 || batch_size == 0) return;
        dim3 blk(256, 1, 1);
        dim3 grd((n + blk.x - 1) / blk.x, batch_size, 1);
        MOREAU_KERNEL_LAUNCH(process_dwork_kernel, grd, blk, 0, stream,
            d_dwork.data(), d_d.data(), scale_min, scale_max, n, batch_size);
    }

    void process_ework(
        BatchedVector& d_ework,
        BatchedVector& d_e,
        double scale_min,
        double scale_max,
        int64_t m,
        int64_t batch_size,
        cudaStream_t stream)
    {
        if (m == 0 || batch_size == 0) return;
        dim3 blk(256, 1, 1);
        dim3 grd((m + blk.x - 1) / blk.x, batch_size, 1);
        MOREAU_KERNEL_LAUNCH(process_ework_kernel, grd, blk, 0, stream,
            d_ework.data(), d_e.data(), scale_min, scale_max, m, batch_size);
    }

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
        cudaStream_t stream)
    {
        if (batch_size == 0 || (n == 0 && m == 0)) return;
        int64_t max_dim = (n > m) ? n : m;
        dim3 blk(256, 1, 1);
        dim3 grd((max_dim + blk.x - 1) / blk.x, batch_size, 1);
        MOREAU_KERNEL_LAUNCH(process_equilibration_kernel, grd, blk, 0, stream,
            d_dwork.data(), d_d.data(), d_ework.data(), d_e.data(),
            scale_min, scale_max, n, m, batch_size);
    }


    void lscale(
        int64_t nnz,
        int64_t nrows,
        int64_t batch_size,
        const int64_t* d_row_of,
        double* d_vals,
        const double* d_L,
        cudaStream_t stream)
    {
        if (nnz == 0 || nrows == 0 || batch_size == 0) return;
        dim3 blk(256, 1, 1);
        dim3 grd((nnz + blk.x - 1) / blk.x, batch_size, 1);
        MOREAU_KERNEL_LAUNCH(lscale_csr_kernel, grd, blk, 0, stream,
            nnz, batch_size, nrows, d_row_of, d_vals, d_L);
    }

    void hadamard(
        double* d_array,
        const double* d_scale_array,
        int64_t n,
        int64_t batch_size,
        cudaStream_t stream)
    {
        if (n == 0 || batch_size == 0) return;
        dim3 blk(256, 1, 1);
        dim3 grd((n + blk.x - 1) / blk.x, batch_size, 1);
        MOREAU_KERNEL_LAUNCH(hadamard_kernel, grd, blk, 0, stream, d_array, d_scale_array, n, batch_size);
    }

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
        cudaStream_t stream)
    {
        if (nnz == 0 || nrows == 0 || ncols == 0 || batch_size == 0) return;
        dim3 blk(256, 1, 1);
        dim3 grd((nnz + blk.x - 1) / blk.x, batch_size, 1);
        // FIXED: Pass parameters in correct order to match kernel signature
        MOREAU_KERNEL_LAUNCH(lrscale_csr_kernel, grd, blk, 0, stream,
            nnz, batch_size, nrows, ncols, d_row_of, d_colind, d_vals, d_L, d_R);
    }


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
        // Scaling vectors — either may be nullptr to skip that side.
        const BatchedVector* d,
        const BatchedVector* e,
        cudaStream_t stream
    ) {
        // P: left-right scale by d (symmetric). Untouched if d is null.
        if (d != nullptr && P.nnz() > 0) {
            lrscale(P.nnz(), P.n(), P.n(), P.batchSize(),
                    d_P_row_of, P.colIndices(), P.values(), d->data(), d->data(), stream);
        }

        // A: row scale by e, column scale by d. Handle all four
        // combinations since callers pass (d, e=null) for direct-x
        // rectification (rows unchanged, columns scaled via d[J] geom
        // mean) and (d=null, e) for slack rectification (rows scaled,
        // columns unchanged).
        if (A.nnz() > 0) {
            if (d != nullptr && e != nullptr) {
                lrscale(A.nnz(), A.n(), A.m(), A.batchSize(),
                        d_A_row_of, A.colIndices(), A.values(),
                        e->data(), d->data(), stream);
            } else if (d != nullptr) {
                // Column-only scale by d: A ← A · diag(d), i.e.
                // A[i,j] *= d[j]. Reuse lscale by feeding it A's
                // colIndices (the per-nnz column) as the "row_of"
                // mapping and d as the scaling vector; lscale's
                // inner loop is `vals[e] *= L[row_of[e]]`, which
                // under this substitution becomes `vals[e] *=
                // d[col_of[e]]` — exactly column scaling.
                // nrows for the kernel must match len(L) = len(d) = n
                // (primal-space) = A.m() (CSR convention: m() = cols).
                lscale(A.nnz(), A.m(), A.batchSize(),
                       A.colIndices(), A.values(), d->data(), stream);
            } else if (e != nullptr) {
                // Row-only scale by e.
                lscale(A.nnz(), A.n(), A.batchSize(),
                       d_A_row_of, A.values(), e->data(), stream);
            }
        }

        // q: scale by d if present.
        if (d != nullptr) {
            hadamard(q.data(), d->data(), q.n(), q.batchSize(), stream);
        }

        // b: scale by e if present. Empty b (m=0) harmlessly no-ops
        // inside hadamard since b.n() == 0.
        if (e != nullptr) {
            hadamard(b.data(), e->data(), b.n(), b.batchSize(), stream);
        }
    }

    void conditional_scale_cost(
        CSR& P,
        BatchedVector& q,
        BatchedVector& c,
        const BatchedVector& d_mean_col_norm_P,
        const BatchedVector& d_inf_norm_q,
        double scale_min,
        double scale_max,
        cudaStream_t stream
    ) {
        dim3 blk(256, 1, 1);
        dim3 grd(P.batchSize(), 1, 1);
        MOREAU_KERNEL_LAUNCH(conditional_scale_cost_kernel, grd, blk, 0, stream,
            P.values(), q.data(), c.data(), d_mean_col_norm_P.data(), d_inf_norm_q.data(),
            scale_min, scale_max, P.nnz(), P.n(), P.batchSize());
    }

    void rectify_cone_equilibration(
        BatchedVector& d_ework,
        const BatchedVector& d_e,
        int64_t cone_start,
        int64_t cone_size,
        cudaStream_t stream
    ) {
        if (cone_size == 0) return;

        dim3 blk(256, 1, 1);
        dim3 grd((cone_size + blk.x - 1) / blk.x, d_ework.batchSize(), 1);
        MOREAU_KERNEL_LAUNCH(rectify_cone_equilibration_kernel, grd, blk, 0, stream,
            d_ework.data(), d_e.data(), cone_start, cone_size, d_ework.batchSize(), d_ework.n());
    }

    void rectify_soc_cone_equilibration(
        BatchedVector& d_ework,
        const BatchedVector& d_e,
        int64_t cone_start,
        const int64_t* d_soc_offsets,
        const int64_t* d_soc_dims,
        int64_t numSocCones,
        cudaStream_t stream,
        const int64_t* d_soc_sz_offsets
    ) {
        if (numSocCones == 0) return;

        dim3 blk(1, 1, 1);
        dim3 grd(numSocCones, d_ework.batchSize(), 1);
        MOREAU_KERNEL_LAUNCH(rectify_soc_cone_equilibration_kernel, grd, blk, 0, stream,
            d_ework.data(), d_e.data(), cone_start, d_ework.batchSize(), d_ework.n(),
            d_soc_offsets, d_soc_dims, numSocCones, d_soc_sz_offsets);
    }

    // Direct-x SOC `d`-rectification. Mirror of the CPU equilibrate loop
    // in problemdata.rs:309-337: for each direct-x SOC cone, replace
    // `d[J]` with `g = geom_mean(d[J])` (uniform scalar) to preserve
    // `x[J] ∈ SOC` under the scaling `x̃ = D⁻¹·x`. Writes
    // `dwork[idx] = g / d[idx]` at each cone's indices; other entries
    // of dwork must be pre-filled to 1.0 by the caller.
    //
    // One (batch, cone) block, single-threaded (dense SOC dim ≤ 4).
    // Nonneg direct-x cones are skipped: per-entry scaling preserves
    // the positive orthant, so they don't need rectification.
    __global__ void rectify_xcone_soc_d_equilibration_kernel(
        double* __restrict__ dwork,
        const double* __restrict__ d,
        const int64_t* __restrict__ d_xcone_kinds,
        const int64_t* __restrict__ d_xcone_dims,
        const int64_t* __restrict__ d_xcone_numel_offsets,
        const int64_t* __restrict__ d_xcone_indices,
        int64_t n,
        int64_t batch_size
    ) {
        if (threadIdx.x != 0) return;
        const int64_t c = blockIdx.x;
        const int64_t b = blockIdx.y;
        if (b >= batch_size) return;

        const int64_t kind = d_xcone_kinds[c];
        // Cones requiring uniform x scaling (Cone::requires_uniform_x_scaling):
        //   SOC (kind=1, any dim — both dense ≤4 and sparse >4)
        //   PSD (kind=2, any dim)
        //   Exp (kind=3, dim=3) — invariant under uniform but not per-coord
        //   Power (kind=4, dim=3) — same
        //   GenPower (kind=5, dim=dim1+dim2) — same
        // Nonneg (kind=0) is invariant under per-entry scaling, so skip it.
        if (kind != 1 /* SOC */ && kind != 2 /* PSD */ &&
            kind != 3 /* Exp */ && kind != 4 /* Power */ &&
            kind != 5 /* GenPower */) return;
        const int64_t dim = d_xcone_dims[c];
        if (dim <= 0) return;

        const int64_t num_off = d_xcone_numel_offsets[c];
        const int64_t d_off   = b * n;

        // Geometric mean via log/exp to avoid product overflow/underflow
        // on ill-scaled problems. Matches CPU logsafe() convention — any
        // non-positive d[idx] would already have been caught by the Ruiz
        // loop's scale_min clamp, so a plain log is safe here.
        double ln_sum = 0.0;
        for (int64_t p = 0; p < dim; ++p) {
            const int64_t idx = d_xcone_indices[num_off + p];
            ln_sum += log(d[d_off + idx]);
        }
        const double geom_mean = exp(ln_sum / static_cast<double>(dim));

        for (int64_t p = 0; p < dim; ++p) {
            const int64_t idx = d_xcone_indices[num_off + p];
            dwork[d_off + idx] = geom_mean / d[d_off + idx];
        }
    }

    void rectify_xcone_soc_d_equilibration(
        BatchedVector& d_dwork,
        const BatchedVector& d_d,
        const int64_t* d_xcone_kinds,
        const int64_t* d_xcone_dims,
        const int64_t* d_xcone_numel_offsets,
        const int64_t* d_xcone_indices,
        int64_t numXCones,
        int64_t n,
        cudaStream_t stream
    ) {
        if (numXCones == 0) return;
        dim3 blk(1, 1, 1);
        dim3 grd(numXCones, d_dwork.batchSize(), 1);
        MOREAU_KERNEL_LAUNCH(rectify_xcone_soc_d_equilibration_kernel, grd, blk, 0, stream,
            d_dwork.data(), d_d.data(),
            d_xcone_kinds, d_xcone_dims,
            d_xcone_numel_offsets, d_xcone_indices,
            n, d_dwork.batchSize());
    }

    void rectify_psd_cone_equilibration(
        BatchedVector& d_ework,
        const BatchedVector& d_e,
        int64_t cone_start,
        const int64_t* d_psd_dims,
        int64_t numPsdCones,
        cudaStream_t stream,
        const int64_t* d_psd_sz_offsets
    ) {
        if (numPsdCones == 0) return;

        dim3 blk(1, 1, 1);
        dim3 grd(numPsdCones, d_ework.batchSize(), 1);
        MOREAU_KERNEL_LAUNCH(rectify_psd_cone_equilibration_kernel, grd, blk, 0, stream,
            d_ework.data(), d_e.data(), cone_start, d_ework.batchSize(), d_ework.n(),
            d_psd_dims, numPsdCones, d_psd_sz_offsets);
    }

    void rectify_genpow_cone_equilibration(
        BatchedVector& d_ework,
        const BatchedVector& d_e,
        int64_t cone_start,
        const int64_t* d_genPowerOffsets,
        const int64_t* d_genPowerDim1s,
        const int64_t* d_genPowerDim2s,
        int64_t numGenPowerCones,
        cudaStream_t stream,
        const int64_t* d_genPowerSzOffsets
    ) {
        if (numGenPowerCones == 0) return;

        dim3 blk(1, 1, 1);
        dim3 grd(numGenPowerCones, d_ework.batchSize(), 1);
        MOREAU_KERNEL_LAUNCH(rectify_genpow_cone_equilibration_kernel, grd, blk, 0, stream,
            d_ework.data(), d_e.data(), cone_start, d_ework.batchSize(), d_ework.n(),
            d_genPowerOffsets, d_genPowerDim1s, d_genPowerDim2s, numGenPowerCones, d_genPowerSzOffsets);
    }

    void set_array(
        BatchedVector& d_array,
        double value,
        int64_t start_idx,
        int64_t length,
        int64_t batch_size,
        cudaStream_t stream
    ) {
        if (length == 0 || batch_size == 0) return;
        // Kernel launch configuration
        dim3 blk(256, 1, 1);
        dim3 grd((length + blk.x - 1) / blk.x, batch_size, 1);

        MOREAU_KERNEL_LAUNCH(set_array_kernel, grd, blk, 0, stream,
            d_array.data(),
            value,
            start_idx,
            length,
            d_array.n(), // Total n for stride
            batch_size
        );
    }

    void compute_reciprocal(
        const BatchedVector& d_input,
        BatchedVector& d_output,
        cudaStream_t stream
    ) {
        if (d_input.n() == 0 || d_input.batchSize() == 0) return;
        dim3 blk(256, 1, 1);
        dim3 grd((d_input.n() + blk.x - 1) / blk.x, d_input.batchSize(), 1);
        MOREAU_KERNEL_LAUNCH(compute_reciprocal_kernel, grd, blk, 0, stream, d_input.data(), d_output.data(), d_input.n(), d_input.batchSize());
    }

} // namespace moreau
