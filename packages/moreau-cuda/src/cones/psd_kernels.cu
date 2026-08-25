/**
 * @file psd_kernels.cu
 * @brief CUDA kernel implementations for PSD (SDP) cone operations
 *
 * Implements PSD cone scaling using cuSOLVER (Cholesky, SVD, eigendecomp)
 * and cuBLAS (matrix multiply). Custom kernels for svec packing/unpacking
 * and symmetric Kronecker product.
 */

#include "moreau/cones/psd_kernels.cuh"
#include "moreau/cones/cones.hpp"
#include "moreau/profiling/profiler.hpp"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cmath>
#include <cfloat>
#include <vector>

namespace moreau {

// ============================================================================
// Device helper functions for PSD cone
// ============================================================================

/// Triangular number: n(n+1)/2
__host__ __device__ inline int64_t triangular_number(int64_t n) {
    return n * (n + 1) / 2;
}

/// Diagonal index in svec: k(k+3)/2 = position of (k,k) element
__host__ __device__ inline int64_t triangular_index(int64_t k) {
    return k * (k + 3) / 2;
}

/// Convert (row, col) with row <= col to upper-triangular linear index
/// Column-major upper triangle: (i,j) maps to j(j+1)/2 + i
__host__ __device__ inline int64_t coord_to_svec_index(int64_t i, int64_t j) {
    return j * (j + 1) / 2 + i;
}

static constexpr double SQRT2 = 1.4142135623730951;
static constexpr double INV_SQRT2 = 0.7071067811865476;

// ============================================================================
// svec ↔ mat conversion kernels
// ============================================================================

/**
 * @brief Unpack svec to dense symmetric matrix (column-major)
 *
 * svec uses upper-triangle column-major order:
 * (0,0), (0,1), (1,1), (0,2), (1,2), (2,2), ...
 * Off-diagonal elements are scaled by 1/√2 in svec → multiply by √2 to recover.
 */
__global__ void svec_to_mat_kernel(
    double* __restrict__ mat,          // output: [batchSize * totalMatSqDim]
    const double* __restrict__ svec,   // input:  [batchSize * totalSvecDim]
    const double* __restrict__ src_vec, // s or z vector [batchSize * m]
    int64_t src_offset,   // offset of PSD cones in src_vec
    int64_t m_total,      // total constraints per batch
    const int64_t* __restrict__ d_dims,
    const int64_t* __restrict__ d_svec_offsets,
    const int64_t* __restrict__ d_sz_offsets,
    const int64_t* __restrict__ d_matsq_offsets,
    int64_t numPsdCones,
    bool from_src_vec     // if true, read from src_vec; else from svec
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numPsdCones) return;

    int64_t n = d_dims[cone];
    int64_t matsq_off = d_matsq_offsets[cone];
    int64_t svec_off;
    const double* src;

    if (from_src_vec) {
        svec_off = d_sz_offsets[cone];
        src = src_vec + batch * m_total + src_offset + svec_off;
    } else {
        svec_off = d_svec_offsets[cone];
        src = svec + batch * d_svec_offsets[numPsdCones] + svec_off;
    }

    double* dst = mat + batch * d_matsq_offsets[numPsdCones] + matsq_off;

    // Each thread handles one or more (i,j) pairs
    int64_t svec_dim = n * (n + 1) / 2;
    for (int64_t idx = threadIdx.x; idx < svec_dim; idx += blockDim.x) {
        // Map linear svec index to (row, col) where row <= col
        // Column-major upper triangle: find col such that col*(col+1)/2 <= idx
        int64_t col = 0;
        int64_t acc = 0;
        while (acc + col + 1 <= idx) {
            acc += col + 1;
            col++;
        }
        int64_t row = idx - acc;

        double val = src[idx];
        if (row != col) {
            // Off-diagonal: svec stores with √2 scaling, recover by dividing by √2
            val *= INV_SQRT2;
        }
        // Column-major storage
        dst[col * n + row] = val;
        dst[row * n + col] = val;  // symmetric
    }
}

/**
 * @brief Pack dense symmetric matrix (column-major) to svec
 *
 * Inverse of svec_to_mat: diagonal stored as-is, off-diagonal scaled by 1/√2.
 */
__global__ void mat_to_svec_kernel(
    double* __restrict__ svec,          // output: [batchSize * totalSvecDim] or into dest_vec
    const double* __restrict__ mat,     // input:  [batchSize * totalMatSqDim]
    double* __restrict__ dest_vec,      // s or z vector [batchSize * m]
    int64_t dest_offset,   // offset of PSD cones in dest_vec
    int64_t m_total,
    const int64_t* __restrict__ d_dims,
    const int64_t* __restrict__ d_svec_offsets,
    const int64_t* __restrict__ d_sz_offsets,
    const int64_t* __restrict__ d_matsq_offsets,
    int64_t numPsdCones,
    bool to_dest_vec       // if true, write to dest_vec; else to svec
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numPsdCones) return;

    int64_t n = d_dims[cone];
    int64_t matsq_off = d_matsq_offsets[cone];
    const double* src = mat + batch * d_matsq_offsets[numPsdCones] + matsq_off;

    double* dst;
    if (to_dest_vec) {
        int64_t sz_off = d_sz_offsets[cone];
        dst = dest_vec + batch * m_total + dest_offset + sz_off;
    } else {
        int64_t svec_off = d_svec_offsets[cone];
        dst = svec + batch * d_svec_offsets[numPsdCones] + svec_off;
    }

    int64_t svec_dim = n * (n + 1) / 2;
    for (int64_t idx = threadIdx.x; idx < svec_dim; idx += blockDim.x) {
        int64_t col = 0;
        int64_t acc = 0;
        while (acc + col + 1 <= idx) {
            acc += col + 1;
            col++;
        }
        int64_t row = idx - acc;

        double val = src[col * n + row];
        if (row != col) {
            val *= SQRT2;
        }
        dst[idx] = val;
    }
}

// ============================================================================
// Symmetric Kronecker product kernel
// ============================================================================

/**
 * @brief Compute upper triangle of symmetric Kronecker product: Hs = skron(A)
 *
 * For n×n symmetric matrix A, computes the svec_dim×svec_dim Hessian.
 * Output is upper triangle of svec_dim×svec_dim stored linearly (packed).
 *
 * Formula:
 * For pairs (i,j) with i<=j and (k,l) with k<=l:
 *   Hs[(i,j),(k,l)] = A[i,k]*A[j,l] + A[i,l]*A[j,k]
 * with √2 scaling for off-diagonal svec indices.
 *
 * Grid: (numBlocks, batchSize, numCones) — parallelizes across output entries.
 */
__global__ void skron_kernel(
    double* __restrict__ Hs,           // output: [batchSize * totalPsdHsEntries]
    const double* __restrict__ RRt,    // input:  [batchSize * totalMatSqDim] (R·R^T matrices)
    const int64_t* __restrict__ d_dims,
    const int64_t* __restrict__ d_Hs_offsets,
    const int64_t* __restrict__ d_matsq_offsets,
    int64_t numPsdCones
) {
    int64_t cone = blockIdx.z;
    int64_t batch = blockIdx.y;
    if (cone >= numPsdCones) return;

    int64_t n = d_dims[cone];
    int64_t Hs_off = d_Hs_offsets[cone];
    int64_t matsq_off = d_matsq_offsets[cone];
    int64_t totalMatSqDim = d_matsq_offsets[numPsdCones];
    int64_t totalHsEntries = d_Hs_offsets[numPsdCones];

    const double* A = RRt + batch * totalMatSqDim + matsq_off;
    double* out = Hs + batch * totalHsEntries + Hs_off;

    int64_t svec_dim = n * (n + 1) / 2;
    int64_t hs_size = svec_dim * (svec_dim + 1) / 2;

    // Global thread ID across all blocks for this (batch, cone)
    int64_t flat = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (flat >= hs_size) return;

    // Map flat index to (row_svec, col_svec) in upper triangle using O(1) formula
    // flat = col_svec*(col_svec+1)/2 + row_svec, so col_svec ≈ sqrt(2*flat)
    int64_t col_svec = (int64_t)((sqrt(1.0 + 8.0 * (double)flat) - 1.0) * 0.5);
    // Correct for floating-point rounding
    while (col_svec * (col_svec + 1) / 2 > flat) col_svec--;
    while ((col_svec + 1) * (col_svec + 2) / 2 <= flat) col_svec++;
    int64_t row_svec = flat - col_svec * (col_svec + 1) / 2;

    // Map svec indices to matrix (i,j) pairs: svec_idx = j*(j+1)/2 + i
    // Use O(1) formula: j ≈ (sqrt(1+8*idx) - 1) / 2
    int64_t j_row = (int64_t)((sqrt(1.0 + 8.0 * (double)row_svec) - 1.0) * 0.5);
    while (j_row * (j_row + 1) / 2 > row_svec) j_row--;
    while ((j_row + 1) * (j_row + 2) / 2 <= row_svec) j_row++;
    int64_t i_row = row_svec - j_row * (j_row + 1) / 2;

    int64_t j_col = (int64_t)((sqrt(1.0 + 8.0 * (double)col_svec) - 1.0) * 0.5);
    while (j_col * (j_col + 1) / 2 > col_svec) j_col--;
    while ((j_col + 1) * (j_col + 2) / 2 <= col_svec) j_col++;
    int64_t i_col = col_svec - j_col * (j_col + 1) / 2;

    int64_t i = i_row, j = j_row, k = i_col, l = j_col;

    // Compute A[i,k]*A[j,l] + A[i,l]*A[j,k]  (column-major)
    double val = A[k * n + i] * A[l * n + j] + A[l * n + i] * A[k * n + j];

    // Apply svec scaling: off-diagonal svec entries are scaled by √2,
    // which introduces factors in the Kronecker product.
    // Both off-diag: val already correct (2 from sum * (1/√2)² = 1)
    // One off-diag: multiply by 1/√2
    // Both diagonal: multiply by 0.5 (sum gives 2× the desired value)
    if (i == j && k == l) {
        val *= 0.5;
    } else if (i == j || k == l) {
        val *= INV_SQRT2;
    }

    out[flat] = val;
}

// ============================================================================
// PSD cone initialization kernel
// ============================================================================

/**
 * @brief Initialize PSD cone variables to identity: s = z = svec(I)
 */
__global__ void initPsdConesKernel(
    double* __restrict__ s,
    double* __restrict__ z,
    int64_t psd_offset,
    const int64_t* __restrict__ d_dims,
    const int64_t* __restrict__ d_sz_offsets,
    int64_t numPsdCones,
    int64_t m_total
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numPsdCones) return;

    int64_t n = d_dims[cone];
    int64_t sz_off = d_sz_offsets[cone];
    int64_t svec_dim = n * (n + 1) / 2;
    int64_t base = batch * m_total + psd_offset + sz_off;

    for (int64_t idx = threadIdx.x; idx < svec_dim; idx += blockDim.x) {
        s[base + idx] = 0.0;
        z[base + idx] = 0.0;
    }
    __syncthreads();

    // Set diagonal entries to 1.0
    for (int64_t k = threadIdx.x; k < n; k += blockDim.x) {
        int64_t diag_idx = triangular_index(k);
        s[base + diag_idx] = 1.0;
        z[base + diag_idx] = 1.0;
    }
}

// ============================================================================
// PSD affine_ds kernel
// ============================================================================

/**
 * @brief Compute affine ds for PSD cones: ds = svec(diag(λ²))
 *
 * Only diagonal entries of the svec are nonzero:
 * ds[triangular_index(k)] = λ[k]² for k = 0..n-1
 */
__global__ void psd_affine_ds_kernel(
    double* __restrict__ ds,
    const double* __restrict__ psd_lambda,
    int64_t psd_offset,
    const int64_t* __restrict__ d_dims,
    const int64_t* __restrict__ d_sz_offsets,
    const int64_t* __restrict__ d_mat_offsets,
    int64_t numPsdCones,
    int64_t totalPsdSvecDim,
    int64_t totalPsdMatDim,
    int64_t m_total
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numPsdCones) return;

    int64_t n = d_dims[cone];
    int64_t sz_off = d_sz_offsets[cone];
    int64_t mat_off = d_mat_offsets[cone];
    int64_t svec_dim = n * (n + 1) / 2;
    int64_t base = batch * m_total + psd_offset + sz_off;
    const double* lambda = psd_lambda + batch * totalPsdMatDim + mat_off;

    // Zero everything first
    for (int64_t i = threadIdx.x; i < svec_dim; i += blockDim.x) {
        ds[base + i] = 0.0;
    }
    __syncthreads();

    // Set diagonal entries to λ²
    for (int64_t k = threadIdx.x; k < n; k += blockDim.x) {
        ds[base + triangular_index(k)] = lambda[k] * lambda[k];
    }
}

// ============================================================================
// PSD Δs_from_Δz_offset: out = W^T(λ \ ds)
// ============================================================================

/**
 * @brief Device kernel: λ_inv_circ_op for PSD cones
 *
 * Computes X[i,j] = 2·Z[i,j] / (λ[i] + λ[j]) where Z is a dense n×n matrix.
 * Operates in-place on mat (column-major).
 */
__global__ void psd_lambda_inv_circ_kernel(
    double* __restrict__ mat,          // [n×n] dense matrix (column-major), modified in-place
    const double* __restrict__ lambda, // [n] eigenvalues
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int64_t i = idx % n;
    int64_t j = idx / n;
    double denom = lambda[i] + lambda[j];
    mat[idx] = (fabs(denom) > 1e-15) ? 2.0 * mat[idx] / denom : 0.0;
}

/**
 * @brief Compute Δs_from_Δz_offset for PSD cones: out = W^T(λ \ ds)
 *
 * This is a symmetric cone operation:
 * 1. Convert ds (svec) → dense matrix D
 * 2. Compute λ \ D: X[i,j] = 2·D[i,j] / (λ[i] + λ[j])
 * 3. Apply W^T: result = R · X · R^T
 * 4. Convert back to svec
 */
void psd_ds_from_dz_offset(
    Cones& cones,
    double* out,
    const double* ds,
    int64_t psd_offset,
    int64_t m_total,
    cudaStream_t stream
) {
    if (cones.numPsdCones == 0) return;

    cusolverDnHandle_t cusolver = cones.cusolverH_;
    cublasHandle_t cublas = cones.cublasH_;
    cusolverDnSetStream(cusolver, stream);
    cublasSetStream_v2(cublas, stream);

    int64_t totalMatSqDim = cones.totalPsdMatSqDim;
    int64_t totalMatDim = cones.totalPsdMatDim;
    double alpha_one = 1.0, beta_zero = 0.0;

    // Step 1: Convert ds (svec in s/z vector) → dense matrices
    dim3 grid(cones.batchSize, cones.numPsdCones);
    MOREAU_KERNEL_LAUNCH(svec_to_mat_kernel, grid, 256, 0, stream,
        cones.psd_work_mat1.data(), nullptr, ds, psd_offset, m_total,
        cones.d_psd_dims, cones.d_psd_svec_offsets, cones.d_psd_sz_offsets,
        cones.d_psd_matsq_offsets, cones.numPsdCones, true
    );

    // Process each cone
    int64_t matsq_off = 0, mat_off = 0;
    for (int64_t cone = 0; cone < cones.numPsdCones; cone++) {
        int64_t n = cones.psdConeDims[cone];
        if (n <= 0) { matsq_off += n * n; mat_off += n; continue; }
        int64_t n2 = n * n;

        for (int64_t b = 0; b < cones.batchSize; b++) {
            double* D_ptr = cones.psd_work_mat1.data() + b * totalMatSqDim + matsq_off;
            double* work1 = cones.psd_work_mat2.data() + b * totalMatSqDim + matsq_off;
            double* work2 = cones.psd_work_mat3.data() + b * totalMatSqDim + matsq_off;
            double* R_ptr = cones.psd_R.data() + b * totalMatSqDim + matsq_off;
            double* lambda_ptr = cones.psd_lambda.data() + b * totalMatDim + mat_off;

            // Step 2: λ_inv_circ: D[i,j] = 2·D[i,j] / (λ[i] + λ[j])
            int blk = (n2 + 255) / 256;
            MOREAU_KERNEL_LAUNCH(psd_lambda_inv_circ_kernel, blk, 256, 0, stream, D_ptr, lambda_ptr, n);

            // Step 3: W^T * (λ\ds) = R · D · R^T
            // work1 = D · R^T
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                          n, n, n, &alpha_one, D_ptr, n, R_ptr, n, &beta_zero, work1, n);
            // work2 = R · work1 = R · D · R^T
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, n, n, &alpha_one, R_ptr, n, work1, n, &beta_zero, work2, n);
        }

        matsq_off += n2;
        mat_off += n;
    }

    // Step 4: Convert dense matrices → svec in out vector
    MOREAU_KERNEL_LAUNCH(mat_to_svec_kernel, grid, 256, 0, stream,
        nullptr, cones.psd_work_mat3.data(), out, psd_offset, m_total,
        cones.d_psd_dims, cones.d_psd_svec_offsets, cones.d_psd_sz_offsets,
        cones.d_psd_matsq_offsets, cones.numPsdCones, true
    );
}

// ============================================================================
// PSD mul_Hs kernel
// ============================================================================

/**
 * @brief Multiply by PSD Hessian: y = Hs * x
 *
 * Hs is stored as upper triangle of svec_dim × svec_dim.
 * y[i] = sum_j Hs[min(i,j), max(i,j)] * x[j]
 */
__global__ void psd_mul_Hs_kernel(
    double* __restrict__ y,
    const double* __restrict__ x,
    const double* __restrict__ psd_Hs,
    int64_t psd_offset,
    const int64_t* __restrict__ d_dims,
    const int64_t* __restrict__ d_sz_offsets,
    const int64_t* __restrict__ d_Hs_offsets,
    int64_t numPsdCones,
    int64_t totalPsdHsEntries,
    int64_t m_total
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numPsdCones) return;

    int64_t n = d_dims[cone];
    int64_t sz_off = d_sz_offsets[cone];
    int64_t Hs_off = d_Hs_offsets[cone];
    int64_t svec_dim = n * (n + 1) / 2;
    int64_t base = batch * m_total + psd_offset + sz_off;
    const double* Hs = psd_Hs + batch * totalPsdHsEntries + Hs_off;
    const double* xp = x + base;
    double* yp = y + base;

    // Each thread computes one row of the matrix-vector product
    for (int64_t row = threadIdx.x; row < svec_dim; row += blockDim.x) {
        double sum = 0.0;
        for (int64_t col = 0; col < svec_dim; col++) {
            // Access upper triangle: min(row, col), max(row, col)
            int64_t r = (row <= col) ? row : col;
            int64_t c = (row <= col) ? col : row;
            int64_t hs_idx = c * (c + 1) / 2 + r;
            sum += Hs[hs_idx] * xp[col];
        }
        yp[row] = sum;
    }
}

// ============================================================================
// PSD margins kernel
// ============================================================================

/**
 * @brief Compute minimum eigenvalue for PSD cone margins
 *
 * Uses a simple iterative approach or Cholesky test for small matrices.
 * For production, this would use cuSOLVER eigendecomp, but we inline
 * a simple version for correctness.
 */
__global__ void psd_margins_kernel(
    const double* __restrict__ z,
    double* __restrict__ d_min_margin,
    double* __restrict__ d_pos_margin,
    int64_t psd_offset,
    const int64_t* __restrict__ d_dims,
    const int64_t* __restrict__ d_sz_offsets,
    int64_t numPsdCones,
    int64_t m_total,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    // Thread 0 per batch computes margins for all PSD cones
    if (threadIdx.x == 0) {
        double local_min = 1e300;
        double local_pos = 0.0;

        for (int64_t cone = 0; cone < numPsdCones; cone++) {
            int64_t n = d_dims[cone];
            int64_t sz_off = d_sz_offsets[cone];
            int64_t base = batch * m_total + psd_offset + sz_off;

            // For PSD margin, we need min eigenvalue of mat(z)
            // Conservative approximation: use min diagonal element
            // (actual eigenvalue ≤ min diagonal for PD matrices)
            for (int64_t k = 0; k < n; k++) {
                double diag_val = z[base + triangular_index(k)];
                if (diag_val < local_min) local_min = diag_val;
                if (diag_val > 0.0) local_pos += diag_val;
            }
        }

        atomicMin((unsigned long long*)d_min_margin,
                  __double_as_longlong(local_min));
        atomicAdd(d_pos_margin, local_pos);
    }
}

/**
 * @brief Apply PSD scaled unit shift: z[triangular_index(k)] += alpha
 */
__global__ void psd_scaled_unit_shift_kernel(
    double* __restrict__ z,
    double alpha,
    int64_t psd_offset,
    const int64_t* __restrict__ d_dims,
    const int64_t* __restrict__ d_sz_offsets,
    int64_t numPsdCones,
    int64_t m_total
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numPsdCones) return;

    int64_t n = d_dims[cone];
    int64_t sz_off = d_sz_offsets[cone];
    int64_t base = batch * m_total + psd_offset + sz_off;

    for (int64_t k = threadIdx.x; k < n; k += blockDim.x) {
        z[base + triangular_index(k)] += alpha;
    }
}

/**
 * @brief Device kernel: subtract σμ from diagonal entries of PSD svec
 */
__global__ void psd_subtract_sigma_mu_kernel(
    double* __restrict__ shift,
    const double* __restrict__ sigma_mu,
    int64_t psd_offset,
    const int64_t* __restrict__ d_dims,
    const int64_t* __restrict__ d_sz_offsets,
    int64_t numPsdCones,
    int64_t m_total
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numPsdCones) return;

    int64_t n = d_dims[cone];
    int64_t sz_off = d_sz_offsets[cone];
    int64_t base = batch * m_total + psd_offset + sz_off;
    double sigma_mu_val = sigma_mu[batch];

    for (int64_t k = threadIdx.x; k < n; k += blockDim.x) {
        shift[base + triangular_index(k)] -= sigma_mu_val;
    }
}

/**
 * @brief PSD combined_ds_shift: shift = W^{-T}(step_s) ∘ W(step_z) - σμ·e
 *
 * For PSD cones (symmetric cone):
 * 1. step_z ← W * step_z  (= R^T · mat(step_z) · R)
 * 2. step_s ← W^{-T} * step_s  (= Rinv · mat(step_s) · Rinv^T)
 * 3. shift = circ_op(step_s, step_z) = (S·Z + Z·S)/2  (Jordan product)
 * 4. shift -= σμ · e  (identity in svec form)
 */
void psd_combined_ds_shift(
    Cones& cones,
    double* shift,
    double* step_z,
    double* step_s,
    const double* sigma_mu,
    int64_t psd_offset,
    int64_t m_total,
    cudaStream_t stream
) {
    if (cones.numPsdCones == 0) return;

    cublasHandle_t cublas = cones.cublasH_;
    cublasSetStream_v2(cublas, stream);

    int64_t totalMatSqDim = cones.totalPsdMatSqDim;
    double alpha_one = 1.0, alpha_half = 0.5, beta_zero = 0.0, beta_one = 1.0;

    dim3 grid(cones.batchSize, cones.numPsdCones);

    // Convert step_z and step_s (svec in s/z vector) → dense matrices
    // work_mat1 = mat(step_z), work_mat2 = mat(step_s)
    MOREAU_KERNEL_LAUNCH(svec_to_mat_kernel, grid, 256, 0, stream,
        cones.psd_work_mat1.data(), nullptr, step_z, psd_offset, m_total,
        cones.d_psd_dims, cones.d_psd_svec_offsets, cones.d_psd_sz_offsets,
        cones.d_psd_matsq_offsets, cones.numPsdCones, true
    );
    MOREAU_KERNEL_LAUNCH(svec_to_mat_kernel, grid, 256, 0, stream,
        cones.psd_work_mat2.data(), nullptr, step_s, psd_offset, m_total,
        cones.d_psd_dims, cones.d_psd_svec_offsets, cones.d_psd_sz_offsets,
        cones.d_psd_matsq_offsets, cones.numPsdCones, true
    );

    // Process each cone
    int64_t matsq_off = 0;
    for (int64_t cone = 0; cone < cones.numPsdCones; cone++) {
        int64_t n = cones.psdConeDims[cone];
        if (n <= 0) { matsq_off += n * n; continue; }
        int64_t n2 = n * n;

        for (int64_t b = 0; b < cones.batchSize; b++) {
            double* Z_ptr = cones.psd_work_mat1.data() + b * totalMatSqDim + matsq_off; // mat(step_z)
            double* S_ptr = cones.psd_work_mat2.data() + b * totalMatSqDim + matsq_off; // mat(step_s)
            double* work = cones.psd_work_mat3.data() + b * totalMatSqDim + matsq_off;
            double* R_ptr = cones.psd_R.data() + b * totalMatSqDim + matsq_off;
            double* Rinv_ptr = cones.psd_Rinv.data() + b * totalMatSqDim + matsq_off;

            // Step 1: step_z ← W * step_z = R^T · Z · R
            // work = R^T · Z
            cublasDgemm_v2(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                          n, n, n, &alpha_one, R_ptr, n, Z_ptr, n, &beta_zero, work, n);
            // Z = work · R = R^T · Z · R
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, n, n, &alpha_one, work, n, R_ptr, n, &beta_zero, Z_ptr, n);

            // Step 2: step_s ← W^{-T} * step_s = Rinv · S · Rinv^T
            // work = S · Rinv^T
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                          n, n, n, &alpha_one, S_ptr, n, Rinv_ptr, n, &beta_zero, work, n);
            // S = Rinv · work = Rinv · S · Rinv^T
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, n, n, &alpha_one, Rinv_ptr, n, work, n, &beta_zero, S_ptr, n);

            // Step 3: circ_op(S, Z) = (S·Z + Z·S) / 2
            // work = S · Z
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, n, n, &alpha_half, S_ptr, n, Z_ptr, n, &beta_zero, work, n);
            // work += 0.5 * Z · S  →  work = (S·Z + Z·S)/2
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, n, n, &alpha_half, Z_ptr, n, S_ptr, n, &beta_one, work, n);
            // Result in work (= psd_work_mat3)
        }

        matsq_off += n2;
    }

    // Convert W*step_z back to svec (modifies step_z in-place)
    MOREAU_KERNEL_LAUNCH(mat_to_svec_kernel, grid, 256, 0, stream,
        nullptr, cones.psd_work_mat1.data(), step_z, psd_offset, m_total,
        cones.d_psd_dims, cones.d_psd_svec_offsets, cones.d_psd_sz_offsets,
        cones.d_psd_matsq_offsets, cones.numPsdCones, true
    );
    // Convert W^{-T}*step_s back to svec (modifies step_s in-place)
    MOREAU_KERNEL_LAUNCH(mat_to_svec_kernel, grid, 256, 0, stream,
        nullptr, cones.psd_work_mat2.data(), step_s, psd_offset, m_total,
        cones.d_psd_dims, cones.d_psd_svec_offsets, cones.d_psd_sz_offsets,
        cones.d_psd_matsq_offsets, cones.numPsdCones, true
    );
    // Convert circ_op result to svec → shift
    MOREAU_KERNEL_LAUNCH(mat_to_svec_kernel, grid, 256, 0, stream,
        nullptr, cones.psd_work_mat3.data(), shift, psd_offset, m_total,
        cones.d_psd_dims, cones.d_psd_svec_offsets, cones.d_psd_sz_offsets,
        cones.d_psd_matsq_offsets, cones.numPsdCones, true
    );

    // Step 4: shift -= σμ·e (subtract from diagonal entries)
    MOREAU_KERNEL_LAUNCH(psd_subtract_sigma_mu_kernel, grid, 256, 0, stream,
        shift, sigma_mu, psd_offset,
        cones.d_psd_dims, cones.d_psd_sz_offsets,
        cones.numPsdCones, m_total
    );
}

// ============================================================================
// Device kernels for reciprocal sqrt and column scaling
// ============================================================================

/**
 * @brief Compute Λ^{-1/2} on device: out[i] = 1/√(in[i]) if in[i] > tol, else 0
 */
__global__ void compute_lambdaisqrt_kernel(
    double* __restrict__ out, const double* __restrict__ in, int64_t n
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Clamp lambda to prevent Λ^{-1/2} from becoming too large.
        // Without this, small lambda values (~1e-3) produce rsqrt ~31.6,
        // which amplifies numerical noise in R and Rinv matrix products.
        double lambda_clamped = fmax(in[i], 1e-10);
        out[i] = rsqrt(lambda_clamped);
    }
}

/**
 * @brief Zero the strict upper triangle of a column-major n×n matrix.
 * After dpotrf with LOWER fill mode, the upper triangle contains stale input data.
 * This kernel clears it so subsequent GEMM operations are correct.
 */
__global__ void zero_upper_triangle_kernel(double* __restrict__ M, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int64_t row = idx % n;
    int64_t col = idx / n;
    if (row < col) M[idx] = 0.0;
}

/**
 * @brief Scale columns of matrix M by vector scale: M[:,k] *= scale[k]
 * M is n×n column-major. Each thread handles one element.
 */
__global__ void scale_columns_by_vector_kernel(
    double* __restrict__ M, const double* __restrict__ scale, int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int64_t col = idx / n;
    M[idx] *= scale[col];
}

/**
 * @brief Scale rows of matrix M by vector scale: M[k,:] *= scale[k]
 * M is n×n column-major. Each thread handles one element.
 */
__global__ void scale_rows_by_vector_kernel(
    double* __restrict__ M, const double* __restrict__ scale, int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int64_t row = idx % n;
    M[idx] *= scale[row];
}

/**
 * @brief Set an n×n column-major matrix to identity on device.
 */
__global__ void set_identity_matrix_kernel(double* __restrict__ M, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int64_t row = idx % n;
    int64_t col = idx / n;
    M[idx] = (row == col) ? 1.0 : 0.0;
}

/**
 * @brief Set all elements of a device vector to a constant.
 */
__global__ void set_constant_kernel(double* __restrict__ v, int64_t n, double val) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = val;
}

/**
 * @brief Per-batch finite check.
 * For each non-finite data[idx], sets d_flags[idx / batch_stride] to 0
 * via atomicMin. d_flags must be a length-batchSize array initialized to 1.
 */
__global__ void check_finite_kernel_impl(const double* __restrict__ data, int64_t n,
                                         int64_t batch_stride, int32_t* __restrict__ d_flags) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && !isfinite(data[idx])) {
        atomicMin(&d_flags[idx / batch_stride], 0);
    }
}

/**
 * @brief Device-side syevd-failure mark.
 * Reads the per-iter cusolverDnDsyevd info; if non-zero, clears the
 * per-batch scaling-success flag. Replaces the per-iter
 * cudaStreamSynchronize+cudaMemcpy round-trip — failures now propagate
 * via the same scaling_success flag the slack PSD path uses, so the
 * IPM picks them up at the normal per-iter check without any host sync.
 * Downstream NaN propagation through the GEMM chain is caught by a
 * post-loop check_finite_kernel over the xcone PSD Hs block.
 */
__global__ void psd_syevd_mark_failure_kernel(
    const int* __restrict__ d_info,
    int32_t* __restrict__ scaling_success_b
) {
    if (*d_info != 0 && scaling_success_b != nullptr) {
        *scaling_success_b = 0;
    }
}

/**
 * @brief Compute sqrt of positive values: out[i] = sqrt(max(in[i], 0))
 * Used for converting eigenvalues of M^T*M to singular values of M.
 */
__global__ void sqrt_positive_kernel(double* __restrict__ v, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        v[i] = (v[i] > 0.0) ? sqrt(v[i]) : 0.0;
    }
}

/**
 * @brief Scale columns of M by 1/sigma: M[:,k] /= sigma[k]
 * If sigma[k] is near zero, set column to zero.
 */
__global__ void scale_columns_by_inv_kernel(double* __restrict__ M, const double* __restrict__ sigma, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int64_t col = idx / n;
    double s = sigma[col];
    M[idx] = (s > 1e-15) ? M[idx] / s : 0.0;
}

// ============================================================================
// Host-side PSD scaling implementation
// ============================================================================

/**
 * @brief Inner cuSOLVER/cuBLAS orchestration for the NT scaling factorization.
 *
 * Shared between slack PSD (`update_psd_scaling`) and direct-x PSD
 * (`update_xcones_psd_scaling`). The caller is responsible for filling
 * `mat_S` and `mat_Z` before invoking — the slack path runs `svec_to_mat`
 * over flat slack vectors; the direct-x path gathers `x[J]` and reads
 * `z_x` flat (with the primal↔dual swap, `mat_S` ← z_x, `mat_Z` ← x[J]).
 *
 * Outputs (all base pointers, indexed at `[b * totalMatSqDim + matsq_off]` or
 * `[b * totalMatDim + mat_off]`):
 *  - R, Rinv, lambda, Lambdaisqrt : NT scaling factors per cone
 *  - Hs : svec×svec upper-tri Hessian, packed via `skron_kernel` at the end
 *
 * Layout:
 *  - dims_host[c] = matrix side length k_c
 *  - matsq_offset accumulates k²; mat_offset accumulates k (per-batch)
 *  - d_dims, d_matsq_offsets, d_Hs_offsets must be aligned with dims_host
 */
static bool psd_scaling_inner_orchestrate(
    cusolverDnHandle_t cusolver,
    cublasHandle_t cublas,
    int* d_info,
    double* d_work,
    size_t work_size,
    double* d_syevd_work,
    size_t syevd_work_size,
    int64_t batchSize,
    int64_t numCones,
    int64_t totalMatSqDim,
    int64_t totalMatDim,
    int64_t totalHsEntries,
    const std::vector<int64_t>& dims_host,
    double* mat_S,
    double* mat_Z,
    double* mat_work,
    double* R_base,
    double* Rinv_base,
    double* lambda_base,
    double* Lambdaisqrt_base,
    double* Hs_base,
    const int64_t* d_dims,
    const int64_t* d_matsq_offsets,
    const int64_t* d_Hs_offsets,
    int32_t* d_scaling_success,
    cudaStream_t stream
) {
    cusolverDnSetStream(cusolver, stream);
    cublasSetStream_v2(cublas, stream);

    int64_t matsq_offset = 0;
    int64_t mat_offset = 0;

    // Phase 1: Run Cholesky + eigendecomp + GEMM for all cones/batch elements.
    // cuSOLVER operations share workspace so batch elements are sequential on the
    // same stream, but we minimize host-device syncs by deferring error checks.
    for (int64_t cone = 0; cone < numCones; cone++) {
        int64_t n = dims_host[cone];
        if (n == 0) continue;
        int64_t n2 = n * n;
        int blk_n2 = (n2 + 255) / 256;

        for (int64_t b = 0; b < batchSize; b++) {
            double* S_ptr = mat_S + b * totalMatSqDim + matsq_offset;
            double* Z_ptr = mat_Z + b * totalMatSqDim + matsq_offset;
            double* work_ptr = mat_work + b * totalMatSqDim + matsq_offset;
            double* R_ptr = R_base + b * totalMatSqDim + matsq_offset;
            double* Rinv_ptr = Rinv_base + b * totalMatSqDim + matsq_offset;
            double* lambda_ptr = lambda_base + b * totalMatDim + mat_offset;
            double* Lisqrt_ptr = Lambdaisqrt_base + b * totalMatDim + mat_offset;

            double alpha_one = 1.0, beta_zero = 0.0;

            // Step 2: Cholesky S = L1*L1^T and Z = L2*L2^T
            // Both use shared workspace so they run sequentially on the stream.
            // We skip per-Cholesky sync — if Cholesky fails (s/z not interior),
            // the subsequent eigendecomp will produce NaN/negative eigenvalues
            // which are caught by sqrt_positive_kernel (clamped to 0).
            cudaMemcpyAsync(R_ptr, S_ptr, sizeof(double) * n2,
                           cudaMemcpyDeviceToDevice, stream);
            cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_LOWER,
                            n, R_ptr, n, d_work, work_size, d_info);
            // dpotrf LOWER only writes the lower triangle; zero the upper triangle
            // so that subsequent GEMM reads correct data
            MOREAU_KERNEL_LAUNCH(zero_upper_triangle_kernel, blk_n2, 256, 0, stream, R_ptr, n);

            cudaMemcpyAsync(Rinv_ptr, Z_ptr, sizeof(double) * n2,
                           cudaMemcpyDeviceToDevice, stream);
            cusolverDnDpotrf(cusolver, CUBLAS_FILL_MODE_LOWER,
                            n, Rinv_ptr, n, d_work, work_size, d_info);
            MOREAU_KERNEL_LAUNCH(zero_upper_triangle_kernel, blk_n2, 256, 0, stream, Rinv_ptr, n);

            // Step 3: M = L2^T * L1 → work_ptr
            cublasDgemm_v2(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                          n, n, n,
                          &alpha_one, Rinv_ptr, n, R_ptr, n,
                          &beta_zero, work_ptr, n);

            // Step 4: Compute singular values/vectors of M via eigendecomposition
            // of M^T*M. This is more numerically robust than cusolverDnDgesvd,
            // which can return incorrect singular values for matrices with
            // degenerate (repeated) singular values.
            //
            // M^T * M → S_ptr (symmetric PSD)
            cublasDgemm_v2(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                          n, n, n,
                          &alpha_one, work_ptr, n, work_ptr, n,
                          &beta_zero, S_ptr, n);

            // Eigendecomposition of M^T*M: eigenvalues → lambda_ptr, eigenvectors → S_ptr.
            // Failure handling is fully device-side: psd_syevd_mark_failure_kernel
            // reads d_info on the stream and clears scaling_success[b] on
            // non-zero. Downstream GEMMs run unconditionally; any NaN/inf they
            // produce is caught by the post-cone-loop check_finite_kernel over
            // xcone_psd_Hs (same pattern as the slack PSD path). Saves a
            // cudaStreamSynchronize+cudaMemcpy round-trip per (cone, batch).
            cusolverDnDsyevd(cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                            n, S_ptr, n, lambda_ptr,
                            d_syevd_work, syevd_work_size, d_info);
            if (d_scaling_success) {
                MOREAU_KERNEL_LAUNCH(psd_syevd_mark_failure_kernel, 1, 1, 0, stream,
                    d_info, &d_scaling_success[b]);
            }

            // lambda = sqrt(eigenvalues of M^T*M) = singular values of M
            // V = eigenvectors of M^T*M (columns of S_ptr)
            int blk_lam = (n + 255) / 256;
            MOREAU_KERNEL_LAUNCH(sqrt_positive_kernel, blk_lam, 256, 0, stream, lambda_ptr, n);

            // Copy V to Z_ptr; syevd gives V (column-major eigenvectors)
            cudaMemcpyAsync(Z_ptr, S_ptr, sizeof(double) * n2,
                           cudaMemcpyDeviceToDevice, stream);

            // Compute U = M * V * diag(1/σ) → S_ptr
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, n, n,
                          &alpha_one, work_ptr, n, Z_ptr, n,
                          &beta_zero, S_ptr, n);
            MOREAU_KERNEL_LAUNCH(scale_columns_by_inv_kernel, blk_n2, 256, 0, stream,
                S_ptr, lambda_ptr, n);

            // Step 5: Compute Λ^{-1/2} on device (no host transfer)
            MOREAU_KERNEL_LAUNCH(compute_lambdaisqrt_kernel, blk_lam, 256, 0, stream,
                Lisqrt_ptr, lambda_ptr, n);

            // Step 5b: R = L1*V*Λ^{-1/2}
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, n, n,
                          &alpha_one, R_ptr, n, Z_ptr, n,
                          &beta_zero, work_ptr, n);
            MOREAU_KERNEL_LAUNCH(scale_columns_by_vector_kernel, blk_n2, 256, 0, stream,
                work_ptr, Lisqrt_ptr, n);
            cudaMemcpyAsync(R_ptr, work_ptr, sizeof(double) * n2,
                           cudaMemcpyDeviceToDevice, stream);

            // Step 6: Rinv = Λ^{-1/2}*U^T*L2^T
            cublasDgemm_v2(cublas, CUBLAS_OP_T, CUBLAS_OP_T,
                          n, n, n,
                          &alpha_one, S_ptr, n, Rinv_ptr, n,
                          &beta_zero, work_ptr, n);
            MOREAU_KERNEL_LAUNCH(scale_rows_by_vector_kernel, blk_n2, 256, 0, stream,
                work_ptr, Lisqrt_ptr, n);
            cudaMemcpyAsync(Rinv_ptr, work_ptr, sizeof(double) * n2,
                           cudaMemcpyDeviceToDevice, stream);

            // Step 7: RR^T = R * R^T → S_ptr for skron
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                          n, n, n,
                          &alpha_one, R_ptr, n, R_ptr, n,
                          &beta_zero, work_ptr, n);
            cudaMemcpyAsync(S_ptr, work_ptr, sizeof(double) * n2,
                           cudaMemcpyDeviceToDevice, stream);
        }

        matsq_offset += n * n;
        mat_offset += n;
    }

    // Step 8: Compute Hs = skron(RR^T) for all cones
    // Parallelize across output entries: each thread computes one Hs entry.
    {
        // Find max Hs size across all cones to determine grid.x
        int64_t max_hs_size = 0;
        for (int64_t c = 0; c < numCones; c++) {
            int64_t cn = dims_host[c];
            int64_t svd = cn * (cn + 1) / 2;
            int64_t hs = svd * (svd + 1) / 2;
            if (hs > max_hs_size) max_hs_size = hs;
        }
        if (max_hs_size > 0) {
            int threads = 256;
            int64_t num_blocks = (max_hs_size + threads - 1) / threads;
            dim3 grid((unsigned int)num_blocks, (unsigned int)batchSize, (unsigned int)numCones);
            MOREAU_KERNEL_LAUNCH(skron_kernel, grid, threads, 0, stream,
                Hs_base,
                mat_S,  // Contains RR^T for each cone
                d_dims,
                d_Hs_offsets,
                d_matsq_offsets,
                numCones
            );
        }
    }
    (void)totalHsEntries;  // currently consumed indirectly through d_Hs_offsets[numCones]

    return true;
}

/**
 * @brief Copy each PSD x-cone's Hs slice from the contiguous PSD-only
 * workspace into the per-x-cone slot of `xcone_Hs`. After this scatter,
 * `refresh_xcone_hs` (KKT scatter) sees PSD entries laid out alongside
 * SOC/Nonneg entries in xcone_Hs, indexed by `d_xcone_hs_offsets[c]`.
 */
// Forward declarations for kernels defined later in this file but used by
// the direct-x PSD step-math functions above.
__global__ void lrscale_kernel(double* __restrict__ M, const double* __restrict__ scale, int64_t n);
__global__ void min_eigval_to_alpha_kernel(
    double* __restrict__ alpha, const double* __restrict__ eigvals, int64_t n, double alpha_max);

__global__ void scatter_xcone_psd_hs_kernel(
    double* __restrict__ xcone_Hs,                  // [batchSize * totalXConeHsEntries]
    const double* __restrict__ xcone_psd_Hs,        // [batchSize * totalXPsdHsEntries]
    const int64_t* __restrict__ d_xcone_kinds,      // [numXCones]
    const int64_t* __restrict__ d_xcone_psd_idx,    // [numXCones]
    const int64_t* __restrict__ d_xcone_hs_offsets, // [numXCones+1]
    const int64_t* __restrict__ d_xcone_psd_hs_offsets, // [numXPsdCones+1]
    const int64_t* __restrict__ d_xcone_dims,       // [numXCones]
    int64_t numXCones,
    int64_t totalXConeHsEntries,
    int64_t totalXPsdHsEntries
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numXCones) return;
    if (static_cast<XConeKind>(d_xcone_kinds[cone]) != XConeKind::PSD) return;

    int64_t psd_idx = d_xcone_psd_idx[cone];
    if (psd_idx < 0) return;

    int64_t dst_off = d_xcone_hs_offsets[cone];
    int64_t src_off = d_xcone_psd_hs_offsets[psd_idx];
    int64_t hs_entries = d_xcone_hs_offsets[cone + 1] - dst_off;
    // Defensive: also matches d_xcone_psd_hs_offsets[psd_idx+1] - src_off.

    double* dst = xcone_Hs + batch * totalXConeHsEntries + dst_off;
    const double* src = xcone_psd_Hs + batch * totalXPsdHsEntries + src_off;

    for (int64_t i = threadIdx.x; i < hs_entries; i += blockDim.x) {
        dst[i] = src[i];
    }
}

/**
 * @brief Direct-x PSD: load svec → mat with two source modes.
 *
 * Two modes:
 *  - `gather_from_x = false`: read from the flat direct-x dual `z_x`
 *    (shape `[batchSize * totalXConeNumel]`) starting at offset
 *    `d_xcone_psd_in_full_offsets[cone]`.
 *  - `gather_from_x = true`: gather from primal `x` (shape `[batchSize * n_primal]`)
 *    via `d_xcone_indices` starting at `d_xcone_psd_in_full_offsets[cone]`.
 *
 * Used to populate `mat_S` (← z_x, "slack-side primal" under the swap)
 * and `mat_Z` (← gathered x[J], "slack-side dual" under the swap) for
 * the direct-x PSD scaling.
 */
__global__ void xcone_psd_svec_to_mat_kernel(
    double* __restrict__ mat,
    const double* __restrict__ xcone_src,                          // for non-gather mode
    const double* __restrict__ x,                                   // for gather mode
    int64_t totalXConeNumel,
    int64_t n_primal,
    const int64_t* __restrict__ d_xcone_psd_k,
    const int64_t* __restrict__ d_xcone_psd_matsq_offsets,
    const int64_t* __restrict__ d_xcone_psd_in_full_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t numXPsdCones,
    bool gather_from_x
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numXPsdCones) return;

    int64_t k = d_xcone_psd_k[cone];
    int64_t matsq_off = d_xcone_psd_matsq_offsets[cone];
    int64_t totalMatSqDim = d_xcone_psd_matsq_offsets[numXPsdCones];
    int64_t in_off = d_xcone_psd_in_full_offsets[cone];
    int64_t svec_dim = k * (k + 1) / 2;

    double* dst = mat + batch * totalMatSqDim + matsq_off;

    for (int64_t idx = threadIdx.x; idx < svec_dim; idx += blockDim.x) {
        int64_t col = 0, acc = 0;
        while (acc + col + 1 <= idx) { acc += col + 1; col++; }
        int64_t row = idx - acc;

        double val;
        if (gather_from_x) {
            int64_t src_idx = d_xcone_indices[in_off + idx];
            val = x[batch * n_primal + src_idx];
        } else {
            val = xcone_src[batch * totalXConeNumel + in_off + idx];
        }
        if (row != col) val *= INV_SQRT2;
        dst[col * k + row] = val;
        dst[row * k + col] = val;
    }
}

/**
 * @brief Direct-x PSD NT scaling.
 *
 * Mirrors `update_psd_scaling` but for direct-x PSD cones. Implements the
 * primal↔dual swap: passes `z_x` as the "slack-side primal" (mat_S) and
 * gathered `x[J]` as the "slack-side dual" (mat_Z), so the slack NT
 * machinery produces `xcone_psd_R/Rinv/lambda/Lambdaisqrt/Hs` that the
 * direct-x (1,1) Hessian block needs.
 *
 * @param x        Primal x [batchSize * n_primal]
 * @param z_x      Direct-x dual, flat over all xcones [batchSize * totalXConeNumel]
 * @param n_primal Length of x per batch element
 * @return true if scaling succeeded for all cones
 */
bool update_xcones_psd_scaling(
    Cones& cones,
    const double* x,
    const double* z_x,
    int64_t n_primal,
    cudaStream_t stream
) {
    if (cones.numXPsdCones == 0) return true;

    const int64_t batchSize = cones.batchSize;
    const int64_t numCones = cones.numXPsdCones;
    const int64_t totalMatSqDim = cones.totalXPsdMatSqDim;
    const int64_t totalMatDim = cones.totalXPsdMatDim;
    const int64_t totalHsEntries = cones.totalXPsdHsEntries;

    double* mat_S = cones.xcone_psd_work_mat1.data();
    double* mat_Z = cones.xcone_psd_work_mat2.data();
    double* mat_work = cones.xcone_psd_work_mat3.data();

    // Step 1: load mat_S ← z_x (flat) and mat_Z ← gather x[J]. The PRIMAL↔DUAL
    // SWAP at the slack-call boundary means mat_S plays the role of the slack
    // primal (s) and mat_Z plays the role of the slack dual (z).
    {
        dim3 grid(batchSize, numCones);
        int threads = 256;
        MOREAU_KERNEL_LAUNCH(xcone_psd_svec_to_mat_kernel, grid, threads, 0, stream,
            mat_S, z_x, nullptr,
            cones.totalXConeNumel, n_primal,
            cones.d_xcone_psd_k, cones.d_xcone_psd_matsq_offsets,
            cones.d_xcone_psd_in_full_offsets, cones.d_xcone_indices,
            numCones, /*gather_from_x=*/false);
        MOREAU_KERNEL_LAUNCH(xcone_psd_svec_to_mat_kernel, grid, threads, 0, stream,
            mat_Z, nullptr, x,
            cones.totalXConeNumel, n_primal,
            cones.d_xcone_psd_k, cones.d_xcone_psd_matsq_offsets,
            cones.d_xcone_psd_in_full_offsets, cones.d_xcone_indices,
            numCones, /*gather_from_x=*/true);
    }

    // dims_host: matrix side length k for each direct-x PSD cone.
    // Per-cone matrix side length k. Cached on the host at construction
    // (xconePsdDimsHost) — pulling from the device each iteration costs a
    // D→H copy + stream sync (violates the "no host↔device transfers in the
    // iteration loop" invariant).
    const std::vector<int64_t>& dims_host = cones.xconePsdDimsHost;

    bool ok = psd_scaling_inner_orchestrate(
        cones.cusolverH_, cones.cublasH_,
        cones.d_psd_info_, cones.d_psd_work_, cones.psd_work_size_,
        cones.d_psd_syevd_work_, cones.psd_syevd_work_size_,
        batchSize, numCones, totalMatSqDim, totalMatDim, totalHsEntries,
        dims_host,
        mat_S, mat_Z, mat_work,
        cones.xcone_psd_R.data(), cones.xcone_psd_Rinv.data(),
        cones.xcone_psd_lambda.data(), cones.xcone_psd_Lambdaisqrt.data(),
        cones.xcone_psd_Hs.data(),
        cones.d_xcone_psd_k, cones.d_xcone_psd_matsq_offsets,
        cones.d_xcone_psd_hs_offsets,
        // Pass the per-batch scaling-success flag so syevd failures
        // mark the batch via the device-side mark_failure kernel. The
        // post-loop check_finite_kernel below catches downstream NaN
        // that might survive the unconditional GEMM chain.
        cones.d_scaling_success,
        stream);

    // Defensive: NaN/inf may have propagated through the unconditional
    // GEMM chain on syevd failure (sqrt of negative eigval, division by
    // zero singular value, etc.). Flag any batch with non-finite xcone
    // Hs entries — mirrors what the slack PSD path does after computing
    // psd_Hs (see line ~2242).
    if (cones.d_scaling_success && totalHsEntries > 0) {
        check_finite_kernel(
            cones.xcone_psd_Hs.data(),
            batchSize * totalHsEntries,
            totalHsEntries,
            cones.d_scaling_success,
            stream);
    }

    // Scatter: copy each PSD x-cone's Hs slice from the contiguous PSD-only
    // `xcone_psd_Hs` into the per-x-cone slot in `xcone_Hs`. The existing
    // refresh_xcone_hs path then picks it up via H_xcone_hs_idx unchanged.
    if (ok && cones.numXCones > 0 && cones.totalXConeHsEntries > 0) {
        MOREAU_KERNEL_LAUNCH(scatter_xcone_psd_hs_kernel, dim3((unsigned int)batchSize, (unsigned int)cones.numXCones), 256, 0, stream,
            cones.xcone_Hs.data(), cones.xcone_psd_Hs.data(),
            cones.d_xcone_kinds, cones.d_xcone_psd_idx,
            cones.d_xcone_hs_offsets, cones.d_xcone_psd_hs_offsets,
            cones.d_xcone_dims,
            cones.numXCones,
            cones.totalXConeHsEntries, cones.totalXPsdHsEntries);
    }
    return ok;
}

// ============================================================================
// Direct-x PSD step math
// ============================================================================

/**
 * Compute svec column-major upper-tri index for matrix entry (row, col)
 * with row ≤ col: idx = col*(col+1)/2 + row.
 */
__device__ __forceinline__ int64_t xcone_psd_svec_idx(int64_t row, int64_t col) {
    return col * (col + 1) / 2 + row;
}

/**
 * @brief Direct-x PSD affine_ds: step_rhs.z_x[J] := svec(diag(λ²)).
 *
 * Zeros the entire svec slot then sets diagonal entries. Layout matches
 * slack `psd_affine_ds_kernel` exactly, just writes into the direct-x
 * dual buffer at the per-cone start `d_xcone_psd_in_full_offsets[psd_idx]`.
 */
__global__ void xcone_fill_step_rhs_zx_psd_affine_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ xcone_psd_lambda,
    const int64_t* __restrict__ d_xcone_psd_k,
    const int64_t* __restrict__ d_xcone_psd_mat_offsets,
    const int64_t* __restrict__ d_xcone_psd_in_full_offsets,
    int64_t numXPsdCones,
    int64_t totalXConeNumel,
    int64_t totalXPsdMatDim
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numXPsdCones) return;

    int64_t k = d_xcone_psd_k[cone];
    int64_t mat_off = d_xcone_psd_mat_offsets[cone];
    int64_t in_off = d_xcone_psd_in_full_offsets[cone];
    int64_t svec_dim = k * (k + 1) / 2;
    int64_t base = batch * totalXConeNumel + in_off;
    const double* lambda = xcone_psd_lambda + batch * totalXPsdMatDim + mat_off;

    // Zero everything first
    for (int64_t i = threadIdx.x; i < svec_dim; i += blockDim.x) {
        step_rhs_z_x[base + i] = 0.0;
    }
    __syncthreads();

    // Diagonal entries: λ² at svec slot (col*(col+1)/2 + col) for col = 0..k-1
    for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
        step_rhs_z_x[base + xcone_psd_svec_idx(col, col)] = lambda[col] * lambda[col];
    }
}

void fill_step_rhs_zx_psd_affine(
    Cones& cones,
    double* step_rhs_z_x,
    cudaStream_t stream
) {
    if (cones.numXPsdCones == 0) return;
    dim3 grid(static_cast<unsigned int>(cones.batchSize),
              static_cast<unsigned int>(cones.numXPsdCones));
    MOREAU_KERNEL_LAUNCH(xcone_fill_step_rhs_zx_psd_affine_kernel, grid, 256, 0, stream,
        step_rhs_z_x, cones.xcone_psd_lambda.data(),
        cones.d_xcone_psd_k, cones.d_xcone_psd_mat_offsets,
        cones.d_xcone_psd_in_full_offsets,
        cones.numXPsdCones, cones.totalXConeNumel, cones.totalXPsdMatDim);
}

/**
 * @brief Helper: gather x[J] into a dense matrix mat (column-major) for one
 * PSD x-cone, applying svec scaling (off-diagonals scaled by 1/√2 to recover
 * matrix value).
 */
__global__ void xcone_psd_gather_step_x_to_mat_kernel(
    double* __restrict__ mat,                                // [batchSize * totalXPsdMatSqDim]
    const double* __restrict__ x_src,                         // [batchSize * n_primal]
    int64_t n_primal,
    const int64_t* __restrict__ d_xcone_psd_k,
    const int64_t* __restrict__ d_xcone_psd_matsq_offsets,
    const int64_t* __restrict__ d_xcone_psd_in_full_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t numXPsdCones
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numXPsdCones) return;

    int64_t k = d_xcone_psd_k[cone];
    int64_t matsq_off = d_xcone_psd_matsq_offsets[cone];
    int64_t totalMatSqDim = d_xcone_psd_matsq_offsets[numXPsdCones];
    int64_t in_off = d_xcone_psd_in_full_offsets[cone];
    int64_t svec_dim = k * (k + 1) / 2;
    double* dst = mat + batch * totalMatSqDim + matsq_off;

    for (int64_t idx = threadIdx.x; idx < svec_dim; idx += blockDim.x) {
        int64_t col = 0, acc = 0;
        while (acc + col + 1 <= idx) { acc += col + 1; col++; }
        int64_t row = idx - acc;
        int64_t src_idx = d_xcone_indices[in_off + idx];
        double val = x_src[batch * n_primal + src_idx];
        if (row != col) val *= INV_SQRT2;
        dst[col * k + row] = val;
        dst[row * k + col] = val;
    }
}

/**
 * @brief Helper: load z_x slice into a dense matrix mat (column-major) for one
 * PSD x-cone, applying svec scaling.
 */
__global__ void xcone_psd_load_zx_to_mat_kernel(
    double* __restrict__ mat,
    const double* __restrict__ zx_src,                        // [batchSize * totalXConeNumel]
    int64_t totalXConeNumel,
    const int64_t* __restrict__ d_xcone_psd_k,
    const int64_t* __restrict__ d_xcone_psd_matsq_offsets,
    const int64_t* __restrict__ d_xcone_psd_in_full_offsets,
    int64_t numXPsdCones
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numXPsdCones) return;

    int64_t k = d_xcone_psd_k[cone];
    int64_t matsq_off = d_xcone_psd_matsq_offsets[cone];
    int64_t totalMatSqDim = d_xcone_psd_matsq_offsets[numXPsdCones];
    int64_t in_off = d_xcone_psd_in_full_offsets[cone];
    int64_t svec_dim = k * (k + 1) / 2;
    double* dst = mat + batch * totalMatSqDim + matsq_off;

    for (int64_t idx = threadIdx.x; idx < svec_dim; idx += blockDim.x) {
        int64_t col = 0, acc = 0;
        while (acc + col + 1 <= idx) { acc += col + 1; col++; }
        int64_t row = idx - acc;
        double val = zx_src[batch * totalXConeNumel + in_off + idx];
        if (row != col) val *= INV_SQRT2;
        dst[col * k + row] = val;
        dst[row * k + col] = val;
    }
}

/**
 * @brief Pack a dense symmetric matrix into svec form ADDED into step_rhs.z_x[J].
 *
 * For each pair (row ≤ col), step_rhs[in_off + svec_idx(row,col)] += val
 * with val = (row == col ? mat[row,row] : √2 · mat[row,col]).
 */
__global__ void xcone_psd_add_mat_to_zx_svec_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ mat,
    int64_t totalXConeNumel,
    const int64_t* __restrict__ d_xcone_psd_k,
    const int64_t* __restrict__ d_xcone_psd_matsq_offsets,
    const int64_t* __restrict__ d_xcone_psd_in_full_offsets,
    int64_t numXPsdCones
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numXPsdCones) return;

    int64_t k = d_xcone_psd_k[cone];
    int64_t matsq_off = d_xcone_psd_matsq_offsets[cone];
    int64_t totalMatSqDim = d_xcone_psd_matsq_offsets[numXPsdCones];
    int64_t in_off = d_xcone_psd_in_full_offsets[cone];
    int64_t svec_dim = k * (k + 1) / 2;
    const double* src = mat + batch * totalMatSqDim + matsq_off;
    double* dst = step_rhs_z_x + batch * totalXConeNumel + in_off;

    for (int64_t idx = threadIdx.x; idx < svec_dim; idx += blockDim.x) {
        int64_t col = 0, acc = 0;
        while (acc + col + 1 <= idx) { acc += col + 1; col++; }
        int64_t row = idx - acc;
        double v = src[col * k + row];
        if (row != col) v *= SQRT2;
        dst[idx] += v;
    }
}

/**
 * @brief Subtract σμ from each diagonal svec entry (for combined-step shift).
 */
__global__ void xcone_psd_subtract_sigma_mu_diag_kernel(
    double* __restrict__ step_rhs_z_x,
    const double* __restrict__ sigma_mu,
    int64_t totalXConeNumel,
    const int64_t* __restrict__ d_xcone_psd_k,
    const int64_t* __restrict__ d_xcone_psd_in_full_offsets,
    int64_t numXPsdCones
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numXPsdCones) return;

    int64_t k = d_xcone_psd_k[cone];
    int64_t in_off = d_xcone_psd_in_full_offsets[cone];
    double sm = sigma_mu[batch];
    double* dst = step_rhs_z_x + batch * totalXConeNumel + in_off;

    for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
        dst[xcone_psd_svec_idx(col, col)] -= sm;
    }
}

/**
 * @brief Scale the entire step_aff_z_x svec block by Mehrotra `m` (one block per cone).
 */
__global__ void xcone_psd_scale_zx_by_m_kernel(
    double* __restrict__ dst_mat,                  // copy of step_aff_z_x scaled by m, in mat form
    const double* __restrict__ step_aff_z_x,
    const double* __restrict__ mehrotra_m,
    int64_t totalXConeNumel,
    const int64_t* __restrict__ d_xcone_psd_k,
    const int64_t* __restrict__ d_xcone_psd_matsq_offsets,
    const int64_t* __restrict__ d_xcone_psd_in_full_offsets,
    int64_t numXPsdCones
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numXPsdCones) return;

    int64_t k = d_xcone_psd_k[cone];
    int64_t matsq_off = d_xcone_psd_matsq_offsets[cone];
    int64_t totalMatSqDim = d_xcone_psd_matsq_offsets[numXPsdCones];
    int64_t in_off = d_xcone_psd_in_full_offsets[cone];
    int64_t svec_dim = k * (k + 1) / 2;
    double* dst = dst_mat + batch * totalMatSqDim + matsq_off;
    double mb = mehrotra_m[batch];

    for (int64_t idx = threadIdx.x; idx < svec_dim; idx += blockDim.x) {
        int64_t col = 0, acc = 0;
        while (acc + col + 1 <= idx) { acc += col + 1; col++; }
        int64_t row = idx - acc;
        double val = mb * step_aff_z_x[batch * totalXConeNumel + in_off + idx];
        if (row != col) val *= INV_SQRT2;
        dst[col * k + row] = val;
        dst[row * k + col] = val;
    }
}

/**
 * @brief Direct-x PSD combined-step shift.
 *
 * Mirrors slack `psd_combined_ds_shift`. Computes:
 *   shift_mat = (R·step_aff_x_J·R^T) ⊙_λ (Rinv·(m·step_aff_z_x)·Rinv^T) − σμ·I
 * where ⊙_λ is the operation `X[i,j] = (X[i,j]·Y[i,j] + ...) / something`
 * — actually the slack version implements the symmetric Jordan product
 * via `lambda_inv_circ`. For a self-adjoint cone with the Mehrotra step
 * (where the W-scaled directions are diagonalized in λ), the standard
 * formula reduces to:
 *
 *   shift = svec(R·X·R^T) ⊙ svec(R^{-T}·Y·R^{-1}) - σμ·svec(I)
 *
 * Here we follow the slack PSD orchestration:
 *   1. mat_X = R^T · X · R   (X = mat(step_aff_x_J))     ← psd_work_mat1
 *   2. mat_Y = Rinv · Y · Rinv^T   (Y = mat(m·step_aff_z_x))   ← psd_work_mat2
 *   3. lrscale by Λ^{-1/2} (so each becomes "λ-circ-normalized")
 *   4. compose via λ_inv_circ
 *
 * For simplicity and correctness we implement the slack pattern in full.
 */
void add_combined_ds_shift_psd(
    Cones& cones,
    double* step_rhs_z_x,
    const double* step_aff_x,
    const double* step_aff_z_x,
    const double* sigma_mu,
    const double* mehrotra_m,
    int64_t n_primal,
    cudaStream_t stream
) {
    if (cones.numXPsdCones == 0) return;

    cusolverDnHandle_t cusolver = cones.cusolverH_;
    cublasHandle_t cublas = cones.cublasH_;
    cusolverDnSetStream(cusolver, stream);
    cublasSetStream_v2(cublas, stream);

    const int64_t batchSize = cones.batchSize;
    const int64_t numCones = cones.numXPsdCones;
    const int64_t totalMatSqDim = cones.totalXPsdMatSqDim;

    // Read dims for host iteration
    // Per-cone matrix side length k. Cached on the host at construction
    // (xconePsdDimsHost) — pulling from the device each iteration costs a
    // D→H copy + stream sync (violates the "no host↔device transfers in the
    // iteration loop" invariant).
    const std::vector<int64_t>& dims_host = cones.xconePsdDimsHost;

    // Step 1: load mat_X ← step_aff_x[J] (gathered) into work_mat1
    {
        dim3 grid((unsigned int)batchSize, (unsigned int)numCones);
        MOREAU_KERNEL_LAUNCH(xcone_psd_gather_step_x_to_mat_kernel, grid, 256, 0, stream,
            cones.xcone_psd_work_mat1.data(), step_aff_x, n_primal,
            cones.d_xcone_psd_k, cones.d_xcone_psd_matsq_offsets,
            cones.d_xcone_psd_in_full_offsets, cones.d_xcone_indices, numCones);
        // Step 2: load mat_Y ← (m · step_aff_z_x) flat into work_mat2
        MOREAU_KERNEL_LAUNCH(xcone_psd_scale_zx_by_m_kernel, grid, 256, 0, stream,
            cones.xcone_psd_work_mat2.data(), step_aff_z_x, mehrotra_m,
            cones.totalXConeNumel,
            cones.d_xcone_psd_k, cones.d_xcone_psd_matsq_offsets,
            cones.d_xcone_psd_in_full_offsets, numCones);
    }

    // Per-(cone, batch): R^T·X·R → work_mat1; Rinv·Y·Rinv^T → work_mat2;
    // λ-symmetrize via Λ^{-1/2} scaling and λ_inv_circ; combine to shift_mat.
    // Per-cone, per-batch, mirroring slack `psd_combined_ds_shift`:
    //   1. mat_Z = R^T · X · R              (X = mat(step_aff_x_J))
    //   2. mat_S = Rinv · Y · Rinv^T         (Y = mat(m · step_aff_z_x))
    //   3. shift_mat = (mat_S · mat_Z + mat_Z · mat_S) / 2
    // No λ-rescaling; no λ-inv-circ. The σμ·I subtraction happens via a
    // separate kernel below.
    int64_t matsq_off = 0;
    double alpha_one = 1.0, alpha_half = 0.5, beta_zero = 0.0, beta_one = 1.0;
    for (int64_t cone = 0; cone < numCones; ++cone) {
        int64_t n = dims_host[cone];
        if (n <= 0) { continue; }
        int64_t n2 = n * n;

        for (int64_t b = 0; b < batchSize; ++b) {
            double* X_mat = cones.xcone_psd_work_mat1.data() + b * totalMatSqDim + matsq_off;
            double* Y_mat = cones.xcone_psd_work_mat2.data() + b * totalMatSqDim + matsq_off;
            double* work = cones.xcone_psd_work_mat3.data() + b * totalMatSqDim + matsq_off;
            double* R_ptr = cones.xcone_psd_R.data() + b * totalMatSqDim + matsq_off;
            double* Rinv_ptr = cones.xcone_psd_Rinv.data() + b * totalMatSqDim + matsq_off;

            // Step 1: X_mat ← R^T · X · R   (overwrite X_mat, no temp shuffle)
            cublasDgemm_v2(cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                           &alpha_one, R_ptr, n, X_mat, n, &beta_zero, work, n);
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &alpha_one, work, n, R_ptr, n, &beta_zero, X_mat, n);

            // Step 2: Y_mat ← Rinv · Y · Rinv^T
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n,
                           &alpha_one, Y_mat, n, Rinv_ptr, n, &beta_zero, work, n);
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &alpha_one, Rinv_ptr, n, work, n, &beta_zero, Y_mat, n);

            // Step 3: work ← (Y_mat · X_mat + X_mat · Y_mat) / 2
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &alpha_half, Y_mat, n, X_mat, n, &beta_zero, work, n);
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &alpha_half, X_mat, n, Y_mat, n, &beta_one, work, n);

            // Stash work into X_mat so the post-loop pack-add kernel can find it.
            cudaMemcpyAsync(X_mat, work, sizeof(double) * (size_t)n2,
                            cudaMemcpyDeviceToDevice, stream);
        }
        matsq_off += n * n;
    }

    // Pack each cone's X_mat (= shift_mat) into svec form, ADD into step_rhs.z_x[J],
    // then subtract σμ from the diagonal.
    {
        dim3 grid((unsigned int)batchSize, (unsigned int)numCones);
        MOREAU_KERNEL_LAUNCH(xcone_psd_add_mat_to_zx_svec_kernel, grid, 256, 0, stream,
            step_rhs_z_x, cones.xcone_psd_work_mat1.data(),
            cones.totalXConeNumel,
            cones.d_xcone_psd_k, cones.d_xcone_psd_matsq_offsets,
            cones.d_xcone_psd_in_full_offsets, numCones);
        MOREAU_KERNEL_LAUNCH(xcone_psd_subtract_sigma_mu_diag_kernel, grid, 256, 0, stream,
            step_rhs_z_x, sigma_mu, cones.totalXConeNumel,
            cones.d_xcone_psd_k, cones.d_xcone_psd_in_full_offsets, numCones);
    }
}

/**
 * @brief Direct-x PSD step length: per-cone eigendecomp of W-transformed
 * (step.x[J], step.z_x) and atomic-min reduce into alpha_z, alpha_s.
 *
 * Mirrors slack `psd_step_length`. With the swap, step_x → "dz" (primal
 * boundary) and step_z_x → "ds" (dual boundary).
 */
void xcone_step_length_psd_reduce(
    Cones& cones,
    double* alpha_s,
    double* alpha_z,
    const double* /*var_x*/,           // current iterate not needed (unlike SOC closed-form);
    const double* /*var_z_x*/,         // step length is computed on transformed steps
    const double* step_x,
    const double* step_z_x,
    int64_t n_primal,
    double max_step,
    cudaStream_t stream
) {
    if (cones.numXPsdCones == 0) return;

    cusolverDnHandle_t cusolver = cones.cusolverH_;
    cublasHandle_t cublas = cones.cublasH_;
    cusolverDnSetStream(cusolver, stream);
    cublasSetStream_v2(cublas, stream);

    const int64_t batchSize = cones.batchSize;
    const int64_t numCones = cones.numXPsdCones;
    const int64_t totalMatSqDim = cones.totalXPsdMatSqDim;

    // Per-cone matrix side length k. Cached on the host at construction
    // (xconePsdDimsHost) — pulling from the device each iteration costs a
    // D→H copy + stream sync (violates the "no host↔device transfers in the
    // iteration loop" invariant).
    const std::vector<int64_t>& dims_host = cones.xconePsdDimsHost;

    // Stage step_x → mat (work_mat1), step_z_x → mat (work_mat2).
    {
        dim3 grid((unsigned int)batchSize, (unsigned int)numCones);
        MOREAU_KERNEL_LAUNCH(xcone_psd_gather_step_x_to_mat_kernel, grid, 256, 0, stream,
            cones.xcone_psd_work_mat1.data(), step_x, n_primal,
            cones.d_xcone_psd_k, cones.d_xcone_psd_matsq_offsets,
            cones.d_xcone_psd_in_full_offsets, cones.d_xcone_indices, numCones);
        MOREAU_KERNEL_LAUNCH(xcone_psd_load_zx_to_mat_kernel, grid, 256, 0, stream,
            cones.xcone_psd_work_mat2.data(), step_z_x, cones.totalXConeNumel,
            cones.d_xcone_psd_k, cones.d_xcone_psd_matsq_offsets,
            cones.d_xcone_psd_in_full_offsets, numCones);
    }

    // Per-cone, per-batch: apply R/Rinv transform then eigendecomp.
    // Reuses one scalar workspace per batch from the slack code's pattern.
    int64_t matsq_off = 0, mat_off = 0;
    double alpha_one = 1.0, beta_zero = 0.0;

    // Need a per-batch eigvals buffer. The slack path uses cones.psd_work_svec
    // (sized totalPsdSvecDim). We reuse it; if slack PSD is empty fall back
    // to the pre-allocated xcone scratch (`xcone_psd_eigvals_scratch`,
    // sized [batchSize][max(xcone_psd_k)] in Cones::initialize) so we don't
    // cudaMalloc inside the IPM loop (CLAUDE.md rule 2).
    bool have_slack_svec = (cones.psd_work_svec.n() > 0);
    double* d_eigvals_buf = nullptr;
    int64_t eigvals_stride = 0;
    if (have_slack_svec) {
        d_eigvals_buf = cones.psd_work_svec.data();
        eigvals_stride = cones.totalPsdSvecDim;
    } else {
        d_eigvals_buf = cones.xcone_psd_eigvals_scratch.data();
        eigvals_stride = cones.xcone_psd_eigvals_scratch.n();
    }

    for (int64_t cone = 0; cone < numCones; ++cone) {
        int64_t n = dims_host[cone];
        if (n <= 0) { continue; }
        int64_t n2 = n * n;
        int blk_n2 = (n2 + 255) / 256;

        for (int64_t b = 0; b < batchSize; ++b) {
            double* R_ptr = cones.xcone_psd_R.data() + b * totalMatSqDim + matsq_off;
            double* Rinv_ptr = cones.xcone_psd_Rinv.data() + b * totalMatSqDim + matsq_off;
            double* Lisqrt_ptr = cones.xcone_psd_Lambdaisqrt.data() + b * cones.totalXPsdMatDim + mat_off;
            double* dx_mat = cones.xcone_psd_work_mat1.data() + b * totalMatSqDim + matsq_off;
            double* dz_mat = cones.xcone_psd_work_mat2.data() + b * totalMatSqDim + matsq_off;
            double* work = cones.xcone_psd_work_mat3.data() + b * totalMatSqDim + matsq_off;
            double* eigvals = d_eigvals_buf + b * eigvals_stride;
            int* d_info = cones.d_psd_info_;

            // α_z (primal side, from dx = step_x[J]):
            //   d = R^T · dx · R, then lrscale by Λ^{-1/2}, eigendecomp.
            cublasDgemm_v2(cublas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                           &alpha_one, R_ptr, n, dx_mat, n, &beta_zero, work, n);
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &alpha_one, work, n, R_ptr, n, &beta_zero, dx_mat, n);
            MOREAU_KERNEL_LAUNCH(lrscale_kernel, blk_n2, 256, 0, stream, dx_mat, Lisqrt_ptr, n);
            cudaMemsetAsync(d_info, 0, sizeof(int), stream);
            cusolverDnDsyevd(cusolver, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                             n, dx_mat, n, eigvals,
                             cones.d_psd_syevd_work_, cones.psd_syevd_work_size_, d_info);
            MOREAU_KERNEL_LAUNCH(min_eigval_to_alpha_kernel, 1, 1, 0, stream, alpha_z + b, eigvals, n, max_step);

            // α_s (dual side, from dz = step_z_x):
            //   d = Rinv · dz · Rinv^T, then lrscale by Λ^{-1/2}, eigendecomp.
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n,
                           &alpha_one, dz_mat, n, Rinv_ptr, n, &beta_zero, work, n);
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &alpha_one, Rinv_ptr, n, work, n, &beta_zero, dz_mat, n);
            MOREAU_KERNEL_LAUNCH(lrscale_kernel, blk_n2, 256, 0, stream, dz_mat, Lisqrt_ptr, n);
            cudaMemsetAsync(d_info, 0, sizeof(int), stream);
            cusolverDnDsyevd(cusolver, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                             n, dz_mat, n, eigvals,
                             cones.d_psd_syevd_work_, cones.psd_syevd_work_size_, d_info);
            MOREAU_KERNEL_LAUNCH(min_eigval_to_alpha_kernel, 1, 1, 0, stream, alpha_s + b, eigvals, n, max_step);
        }

        matsq_off += n * n;
        mat_off += n;
    }

}

/**
 * @brief Scatter mat_out (column-major matrix per PSD x-cone) into workx
 * via xc.indices, applying svec scaling. Each (row, col) with row ≤ col
 * contributes to workx at problem-space index `xc.indices[svec_pos(row, col)]`,
 * where the svec_pos is the column-major upper-tri encoding.
 *
 * Off-diagonal entries get factor √2 (matching mat_to_svec convention).
 */
__global__ void xcone_psd_scatter_mat_to_workx_kernel(
    double* __restrict__ workx,                    // [batchSize * n_primal]
    const double* __restrict__ mat,                // [batchSize * totalXPsdMatSqDim]
    int64_t n_primal,
    const int64_t* __restrict__ d_xcone_psd_k,
    const int64_t* __restrict__ d_xcone_psd_matsq_offsets,
    const int64_t* __restrict__ d_xcone_psd_in_full_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t numXPsdCones
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numXPsdCones) return;

    int64_t k = d_xcone_psd_k[cone];
    int64_t matsq_off = d_xcone_psd_matsq_offsets[cone];
    int64_t totalMatSqDim = d_xcone_psd_matsq_offsets[numXPsdCones];
    int64_t in_off = d_xcone_psd_in_full_offsets[cone];
    int64_t svec_dim = k * (k + 1) / 2;
    const double* src = mat + batch * totalMatSqDim + matsq_off;

    for (int64_t idx = threadIdx.x; idx < svec_dim; idx += blockDim.x) {
        int64_t col = 0, acc = 0;
        while (acc + col + 1 <= idx) { acc += col + 1; col++; }
        int64_t row = idx - acc;
        double val = src[col * k + row];
        if (row != col) val *= SQRT2;
        int64_t prob_idx = d_xcone_indices[in_off + idx];
        atomicAdd(&workx[batch * n_primal + prob_idx], -val);
    }
}

void subtract_xcone_combined_from_workx_psd(
    Cones& cones,
    double* workx,
    const double* rhs_z_x,
    int64_t n_primal,
    cudaStream_t stream
) {
    if (cones.numXPsdCones == 0) return;

    cusolverDnHandle_t cusolver = cones.cusolverH_;
    cublasHandle_t cublas = cones.cublasH_;
    cusolverDnSetStream(cusolver, stream);
    cublasSetStream_v2(cublas, stream);

    const int64_t batchSize = cones.batchSize;
    const int64_t numCones = cones.numXPsdCones;
    const int64_t totalMatSqDim = cones.totalXPsdMatSqDim;

    // Per-cone matrix side length k. Cached on the host at construction
    // (xconePsdDimsHost) — pulling from the device each iteration costs a
    // D→H copy + stream sync (violates the "no host↔device transfers in the
    // iteration loop" invariant).
    const std::vector<int64_t>& dims_host = cones.xconePsdDimsHost;

    // Step 1: load rhs.z_x[J] flat → mat_D (work_mat1)
    {
        dim3 grid((unsigned int)batchSize, (unsigned int)numCones);
        MOREAU_KERNEL_LAUNCH(xcone_psd_load_zx_to_mat_kernel, grid, 256, 0, stream,
            cones.xcone_psd_work_mat1.data(), rhs_z_x, cones.totalXConeNumel,
            cones.d_xcone_psd_k, cones.d_xcone_psd_matsq_offsets,
            cones.d_xcone_psd_in_full_offsets, numCones);
    }

    // Per-(cone, batch): λ-inv-circ in-place, then mat_out = R · mat_D · R^T into work_mat3
    int64_t matsq_off = 0, mat_off = 0;
    double alpha_one = 1.0, beta_zero = 0.0;
    for (int64_t cone = 0; cone < numCones; ++cone) {
        int64_t n = dims_host[cone];
        if (n <= 0) { continue; }
        int64_t n2 = n * n;
        int blk_n2 = (n2 + 255) / 256;

        for (int64_t b = 0; b < batchSize; ++b) {
            double* D_mat = cones.xcone_psd_work_mat1.data() + b * totalMatSqDim + matsq_off;
            double* work = cones.xcone_psd_work_mat2.data() + b * totalMatSqDim + matsq_off;
            double* out_mat = cones.xcone_psd_work_mat3.data() + b * totalMatSqDim + matsq_off;
            double* R_ptr = cones.xcone_psd_R.data() + b * totalMatSqDim + matsq_off;
            double* lambda_ptr = cones.xcone_psd_lambda.data() + b * cones.totalXPsdMatDim + mat_off;

            // λ-inv-circ in-place: D[i,j] *= 2/(λ[i]+λ[j])
            MOREAU_KERNEL_LAUNCH(psd_lambda_inv_circ_kernel, blk_n2, 256, 0, stream, D_mat, lambda_ptr, n);

            // work = R · D
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                           &alpha_one, R_ptr, n, D_mat, n, &beta_zero, work, n);
            // out = work · R^T = R · D · R^T
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n,
                           &alpha_one, work, n, R_ptr, n, &beta_zero, out_mat, n);
        }
        matsq_off += n * n;
        mat_off += n;
    }

    // Step 3: scatter mat_out into workx via xc.indices, with svec scaling.
    {
        dim3 grid((unsigned int)batchSize, (unsigned int)numCones);
        MOREAU_KERNEL_LAUNCH(xcone_psd_scatter_mat_to_workx_kernel, grid, 256, 0, stream,
            workx, cones.xcone_psd_work_mat3.data(), n_primal,
            cones.d_xcone_psd_k, cones.d_xcone_psd_matsq_offsets,
            cones.d_xcone_psd_in_full_offsets, cones.d_xcone_indices,
            numCones);
    }
}

/**
 * @brief Add α to each PSD x-cone's diagonal svec entries of `x[J]`.
 *
 * For each PSD x-cone with svec layout indices, `x[xc.indices[diag_pos]] += α`
 * for diag_pos = 0, 2, 5, 9, ... (positions of the diagonal in column-major
 * upper-tri svec). Used by `shift_x_into_psd_interior` after the host-side
 * eigendecomp determines the per-cone shift amount.
 */
__global__ void xcone_psd_add_alpha_to_x_diag_kernel(
    double* __restrict__ x,
    const double* __restrict__ alpha_per_cone,         // [batchSize * numXPsdCones]
    int64_t n_primal,
    const int64_t* __restrict__ d_xcone_psd_k,
    const int64_t* __restrict__ d_xcone_psd_in_full_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t numXPsdCones
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numXPsdCones) return;

    int64_t k = d_xcone_psd_k[cone];
    int64_t in_off = d_xcone_psd_in_full_offsets[cone];
    double a = alpha_per_cone[batch * numXPsdCones + cone];

    for (int64_t col = threadIdx.x; col < k; col += blockDim.x) {
        // diag svec position: (col+1)*(col+2)/2 - 1
        int64_t diag_svec_pos = (col + 1) * (col + 2) / 2 - 1;
        int64_t prob_idx = d_xcone_indices[in_off + diag_svec_pos];
        // d_xcone_indices is disjoint across cones (asserted at
        // construction), so different `col` values in different cones
        // map to distinct prob_idx values within a batch slot. The
        // atomic was unnecessary; plain += is safe.
        x[batch * n_primal + prob_idx] += a;
    }
}

void shift_x_into_psd_interior(
    Cones& cones,
    double* x,
    int64_t n_primal,
    double total_degree,
    cudaStream_t stream
) {
    if (cones.numXPsdCones == 0) return;

    cusolverDnHandle_t cusolver = cones.cusolverH_;
    cusolverDnSetStream(cusolver, stream);

    const int64_t batchSize = cones.batchSize;
    const int64_t numCones = cones.numXPsdCones;
    const int64_t totalMatSqDim = cones.totalXPsdMatSqDim;

    // Per-cone matrix side length k. Cached on the host at construction
    // (xconePsdDimsHost) — pulling from the device each iteration costs a
    // D→H copy + stream sync (violates the "no host↔device transfers in the
    // iteration loop" invariant).
    const std::vector<int64_t>& dims_host = cones.xconePsdDimsHost;

    // Per (batch, cone): gather x[J] → matrix → eigendecomp → host eigvals
    // → compute α → write to alpha_per_cone host array → upload + add.
    std::vector<double> alpha_host(batchSize * numCones, 0.0);

    int64_t matsq_off = 0;
    for (int64_t cone = 0; cone < numCones; ++cone) {
        int64_t n = dims_host[cone];
        if (n <= 0) continue;
        int64_t n2 = n * n;
        int blk_n2 = (n2 + 255) / 256;
        (void)blk_n2;

        for (int64_t b = 0; b < batchSize; ++b) {
            // Gather x[J] (one batch, one cone) into a host-side matrix.
            // We do this via gather kernel into work_mat1, then download.
            double* d_mat = cones.xcone_psd_work_mat1.data() + b * totalMatSqDim + matsq_off;
            // The gather kernel reads ALL batches and ALL cones; for simplicity
            // gather everything once outside this batch loop. But ordering matters
            // here: we want sequential cuSolver calls. Instead, gather per (b, cone).
            // Use a single-threaded gather via host launch.
            // Easier: download x's slice for this cone, build matrix on host.
            std::vector<int64_t> indices_host(n * (n + 1) / 2);
            cudaMemcpyAsync(indices_host.data(),
                            cones.d_xcone_indices +
                                /*in_off for this cone*/ 0,
                            sizeof(int64_t) * 0, cudaMemcpyDeviceToHost, stream);
            // Host walks d_xcone_psd_in_full_offsets and d_xcone_indices.
            // Cache them once before the loop.
            (void)d_mat;
            (void)indices_host;
        }
        matsq_off += n2;
    }

    // Cleaner implementation: do all the host-side gather/eigendecomp work
    // up front, compute alpha values, then do one device kernel launch.
    // Read d_xcone_psd_in_full_offsets and d_xcone_indices once.
    std::vector<int64_t> in_full_offsets(numCones);
    cudaMemcpyAsync(in_full_offsets.data(), cones.d_xcone_psd_in_full_offsets,
                    sizeof(int64_t) * numCones, cudaMemcpyDeviceToHost, stream);
    int64_t total_indices = cones.totalXConeNumel;
    std::vector<int64_t> all_indices(total_indices);
    cudaMemcpyAsync(all_indices.data(), cones.d_xcone_indices,
                    sizeof(int64_t) * total_indices, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Per (batch, cone): build host-side matrix from x[J], eigendecomp on GPU,
    // read eigvals, compute α.
    matsq_off = 0;
    for (int64_t cone = 0; cone < numCones; ++cone) {
        int64_t n = dims_host[cone];
        if (n <= 0) { matsq_off += n * n; continue; }
        int64_t n2 = n * n;
        int64_t in_off = in_full_offsets[cone];
        int64_t svec_dim = n * (n + 1) / 2;

        for (int64_t b = 0; b < batchSize; ++b) {
            // Download x[J] entries for this cone.
            std::vector<double> xj(svec_dim);
            for (int64_t i = 0; i < svec_dim; ++i) {
                int64_t prob_idx = all_indices[in_off + i];
                cudaMemcpyAsync(&xj[i], x + b * n_primal + prob_idx,
                                sizeof(double), cudaMemcpyDeviceToHost, stream);
            }
            cudaStreamSynchronize(stream);

            // Build dense matrix from svec (column-major upper-tri).
            std::vector<double> mat_h(n2, 0.0);
            for (int64_t idx = 0; idx < svec_dim; ++idx) {
                int64_t col = 0, acc = 0;
                while (acc + col + 1 <= idx) { acc += col + 1; col++; }
                int64_t row = idx - acc;
                double v = xj[idx];
                if (row != col) v *= 0.7071067811865476;  // INV_SQRT2
                mat_h[col * n + row] = v;
                mat_h[row * n + col] = v;
            }

            // Upload to device and eigendecompose.
            double* d_mat = cones.xcone_psd_work_mat1.data() + b * totalMatSqDim + matsq_off;
            double* d_eig = cones.xcone_psd_work_mat3.data() + b * totalMatSqDim + matsq_off;
            cudaMemcpy(d_mat, mat_h.data(), sizeof(double) * n2, cudaMemcpyHostToDevice);
            cudaMemset(cones.d_psd_info_, 0, sizeof(int));
            cusolverDnDsyevd(cusolver,
                             CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                             n, d_mat, n, d_eig,
                             cones.d_psd_syevd_work_, cones.psd_syevd_work_size_,
                             cones.d_psd_info_);
            cudaDeviceSynchronize();

            std::vector<double> eigs(n);
            cudaMemcpy(eigs.data(), d_eig, sizeof(double) * n, cudaMemcpyDeviceToHost);

            // Compute (min_margin, pos_margin, target, α) using CPU's
            // _shift_single_cone_to_interior algorithm.
            double min_margin = eigs[0];
            double pos_margin = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                if (eigs[i] < min_margin) min_margin = eigs[i];
                if (eigs[i] > 0) pos_margin += eigs[i];
            }
            double degree = total_degree > 0 ? total_degree : 1.0;
            double target = std::max(1.0, pos_margin * 0.1 / degree);
            double alpha = 0.0;
            if (min_margin <= 0.0) {
                alpha = -min_margin + target;
            } else if (min_margin < target) {
                alpha = target - min_margin;
            }
            alpha_host[b * numCones + cone] = alpha;
        }
        matsq_off += n2;
    }

    // Upload α and dispatch the diag-add kernel.
    double* d_alpha = nullptr;
    cudaMalloc(&d_alpha, sizeof(double) * batchSize * numCones);
    cudaMemcpyAsync(d_alpha, alpha_host.data(),
                    sizeof(double) * batchSize * numCones,
                    cudaMemcpyHostToDevice, stream);

    dim3 grid((unsigned int)batchSize, (unsigned int)numCones);
    int threads = 32;
    MOREAU_KERNEL_LAUNCH(xcone_psd_add_alpha_to_x_diag_kernel, grid, threads, 0, stream,
        x, d_alpha, n_primal,
        cones.d_xcone_psd_k, cones.d_xcone_psd_in_full_offsets,
        cones.d_xcone_indices, numCones);

    cudaStreamSynchronize(stream);
    cudaFree(d_alpha);
}

/**
 * @brief Direct-x PSD: y[J flat] := -Hs · gather(x[J]) - offset[J flat]
 *
 * One block per (batch, cone). Gather Δx from lhs_x via d_xcone_indices,
 * apply the dense svec_dim×svec_dim Hessian Hs (stored upper-tri in
 * xcone_Hs), then subtract `offset[J flat]` (which is var_z_x[J] for
 * affine, or precomputed c_J_combined for combined).
 */
__global__ void recover_dz_x_psd_kernel(
    double* __restrict__ dz_x,
    const double* __restrict__ lhs_x,
    const double* __restrict__ offset_z_x,           // var_z_x for affine, c_J_combined for combined
    const double* __restrict__ xcone_Hs,
    const int64_t* __restrict__ d_xcone_kinds,
    const int64_t* __restrict__ d_xcone_dims,
    const int64_t* __restrict__ d_xcone_numel_offsets,
    const int64_t* __restrict__ d_xcone_hs_offsets,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t n_primal,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (static_cast<XConeKind>(d_xcone_kinds[cone]) != XConeKind::PSD) return;

    int64_t svec_dim = d_xcone_dims[cone];
    int64_t num_off = d_xcone_numel_offsets[cone];
    int64_t hs_off = d_xcone_hs_offsets[cone];
    int64_t x_off  = batch * n_primal;
    int64_t zx_off = batch * totalXConeNumel + num_off;
    int64_t hs_off_b = batch * totalXConeHsEntries + hs_off;

    // Per-thread: compute one row of (Hs · Δx[J]_gathered)
    for (int64_t row = threadIdx.x; row < svec_dim; row += blockDim.x) {
        double sum = 0.0;
        for (int64_t col = 0; col < svec_dim; ++col) {
            int64_t r = (row <= col) ? row : col;
            int64_t c = (row <= col) ? col : row;
            int64_t hs_idx = c * (c + 1) / 2 + r;
            int64_t prob_idx_col = d_xcone_indices[num_off + col];
            sum += xcone_Hs[hs_off_b + hs_idx] * lhs_x[x_off + prob_idx_col];
        }
        dz_x[zx_off + row] = -sum - offset_z_x[zx_off + row];
    }
}

void recover_dz_x_affine_psd(
    Cones& cones,
    double* dz_x,
    const double* lhs_x,
    const double* var_z_x,
    int64_t n_primal,
    cudaStream_t stream
) {
    if (cones.numXPsdCones == 0 || cones.numXCones == 0) return;
    dim3 grid((unsigned int)cones.batchSize, (unsigned int)cones.numXCones);
    int threads = 64;
    MOREAU_KERNEL_LAUNCH(recover_dz_x_psd_kernel, grid, threads, 0, stream,
        dz_x, lhs_x, var_z_x, cones.xcone_Hs.data(),
        cones.d_xcone_kinds, cones.d_xcone_dims,
        cones.d_xcone_numel_offsets, cones.d_xcone_hs_offsets,
        cones.d_xcone_indices,
        n_primal, cones.totalXConeNumel, cones.totalXConeHsEntries);
}

void recover_dz_x_combined_psd(
    Cones& cones,
    double* dz_x,
    const double* lhs_x,
    const double* rhs_z_x,
    int64_t n_primal,
    cudaStream_t stream
) {
    if (cones.numXPsdCones == 0 || cones.numXCones == 0) return;

    // Compute c_J_combined = svec(R · λ_inv_circ(mat(rhs.z_x[J])) · R^T)
    // into work_mat3 (one cone at a time, reusing the same per-cone matsq layout).
    cusolverDnHandle_t cusolver = cones.cusolverH_;
    cublasHandle_t cublas = cones.cublasH_;
    cusolverDnSetStream(cusolver, stream);
    cublasSetStream_v2(cublas, stream);

    const int64_t batchSize = cones.batchSize;
    const int64_t numCones = cones.numXPsdCones;
    const int64_t totalMatSqDim = cones.totalXPsdMatSqDim;

    // Per-cone matrix side length k. Cached on the host at construction
    // (xconePsdDimsHost) — pulling from the device each iteration costs a
    // D→H copy + stream sync (violates the "no host↔device transfers in the
    // iteration loop" invariant).
    const std::vector<int64_t>& dims_host = cones.xconePsdDimsHost;

    // Step 1: load rhs.z_x[J] flat → mat (work_mat1).
    {
        dim3 grid((unsigned int)batchSize, (unsigned int)numCones);
        MOREAU_KERNEL_LAUNCH(xcone_psd_load_zx_to_mat_kernel, grid, 256, 0, stream,
            cones.xcone_psd_work_mat1.data(), rhs_z_x, cones.totalXConeNumel,
            cones.d_xcone_psd_k, cones.d_xcone_psd_matsq_offsets,
            cones.d_xcone_psd_in_full_offsets, numCones);
    }

    int64_t matsq_off = 0, mat_off = 0;
    double alpha_one = 1.0, beta_zero = 0.0;
    for (int64_t cone = 0; cone < numCones; ++cone) {
        int64_t nk = dims_host[cone];
        if (nk <= 0) continue;
        int64_t n2 = nk * nk;
        int blk_n2 = (n2 + 255) / 256;

        for (int64_t b = 0; b < batchSize; ++b) {
            double* D_mat = cones.xcone_psd_work_mat1.data() + b * totalMatSqDim + matsq_off;
            double* work = cones.xcone_psd_work_mat2.data() + b * totalMatSqDim + matsq_off;
            double* out_mat = cones.xcone_psd_work_mat3.data() + b * totalMatSqDim + matsq_off;
            double* R_ptr = cones.xcone_psd_R.data() + b * totalMatSqDim + matsq_off;
            double* lambda_ptr = cones.xcone_psd_lambda.data() + b * cones.totalXPsdMatDim + mat_off;

            MOREAU_KERNEL_LAUNCH(psd_lambda_inv_circ_kernel, blk_n2, 256, 0, stream, D_mat, lambda_ptr, nk);
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, nk, nk, nk,
                           &alpha_one, R_ptr, nk, D_mat, nk, &beta_zero, work, nk);
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T, nk, nk, nk,
                           &alpha_one, work, nk, R_ptr, nk, &beta_zero, out_mat, nk);
        }
        matsq_off += n2;
        mat_off += nk;
    }

    // c_J_combined as svec entries → reusable Cones workspace (no per-call
    // cudaMalloc/cudaFree, no stream sync). Zero each call since we only
    // write into PSD x-cone slots and the kernel reads the full layout.
    double* d_c_combined = cones.xcone_psd_combined_scratch.data();
    cudaMemsetAsync(d_c_combined, 0,
                    sizeof(double) * batchSize * cones.totalXConeNumel, stream);
    {
        dim3 grid((unsigned int)batchSize, (unsigned int)numCones);
        MOREAU_KERNEL_LAUNCH(xcone_psd_add_mat_to_zx_svec_kernel, grid, 256, 0, stream,
            d_c_combined, cones.xcone_psd_work_mat3.data(),
            cones.totalXConeNumel,
            cones.d_xcone_psd_k, cones.d_xcone_psd_matsq_offsets,
            cones.d_xcone_psd_in_full_offsets, numCones);
    }

    // Now dispatch the recovery kernel using d_c_combined as the offset.
    {
        dim3 grid((unsigned int)cones.batchSize, (unsigned int)cones.numXCones);
        int threads = 64;
        MOREAU_KERNEL_LAUNCH(recover_dz_x_psd_kernel, grid, threads, 0, stream,
            dz_x, lhs_x, d_c_combined, cones.xcone_Hs.data(),
            cones.d_xcone_kinds, cones.d_xcone_dims,
            cones.d_xcone_numel_offsets, cones.d_xcone_hs_offsets,
            cones.d_xcone_indices,
            n_primal, cones.totalXConeNumel, cones.totalXConeHsEntries);
    }
}

// Slack-PSD entry point: builds mat_S/mat_Z from flat slack vectors with
// `psd_offset` and per-cone `sz_offsets`, then delegates to the orchestrator.
//
// Reports per-batch numerical failure by writing d_scaling_success[b] = 0
// when the orchestrator hit an eigendecomp fallback in batch b, and again
// (defensively) after computing Hs by running check_finite_kernel — a
// non-finite Hs in batch k will flag exactly that batch.
void update_psd_scaling(
    Cones& cones,
    const double* s,
    const double* z,
    int64_t m_total,
    int32_t* d_scaling_success,
    cudaStream_t stream
) {
    if (cones.numPsdCones == 0) return;

    const int64_t batchSize = cones.batchSize;
    const int64_t numCones = cones.numPsdCones;
    const int64_t totalMatSqDim = cones.totalPsdMatSqDim;
    const int64_t totalMatDim = cones.totalPsdMatDim;

    // Compute PSD offset in s/z
    int64_t psd_offset = cones.numZeroCones + cones.numNonnegCones
                       + cones.totalSocDim + cones.numExpCones * 3
                       + cones.numPowerCones * 3;

    // Work matrices: mat1 = S, mat2 = Z, mat3 = scratch
    double* mat_S = cones.psd_work_mat1.data();
    double* mat_Z = cones.psd_work_mat2.data();
    double* mat_work = cones.psd_work_mat3.data();

    // Step 1: Convert svec(s) → S and svec(z) → Z
    {
        dim3 grid(batchSize, numCones);
        int threads = 256;
        MOREAU_KERNEL_LAUNCH(svec_to_mat_kernel, grid, threads, 0, stream,
            mat_S, nullptr, s, psd_offset, m_total,
            cones.d_psd_dims, cones.d_psd_svec_offsets, cones.d_psd_sz_offsets,
            cones.d_psd_matsq_offsets, numCones, true
        );
        MOREAU_KERNEL_LAUNCH(svec_to_mat_kernel, grid, threads, 0, stream,
            mat_Z, nullptr, z, psd_offset, m_total,
            cones.d_psd_dims, cones.d_psd_svec_offsets, cones.d_psd_sz_offsets,
            cones.d_psd_matsq_offsets, numCones, true
        );
    }

    std::vector<int64_t> dims_host(cones.psdConeDims.begin(), cones.psdConeDims.end());

    psd_scaling_inner_orchestrate(
        cones.cusolverH_, cones.cublasH_,
        cones.d_psd_info_, cones.d_psd_work_, cones.psd_work_size_,
        cones.d_psd_syevd_work_, cones.psd_syevd_work_size_,
        batchSize, numCones, totalMatSqDim, totalMatDim, cones.totalPsdHsEntries,
        dims_host,
        mat_S, mat_Z, mat_work,
        cones.psd_R.data(), cones.psd_Rinv.data(),
        cones.psd_lambda.data(), cones.psd_Lambdaisqrt.data(),
        cones.psd_Hs.data(),
        cones.d_psd_dims, cones.d_psd_matsq_offsets, cones.d_psd_Hs_offsets,
        d_scaling_success,
        stream);

    // Per-batch finite check over the computed Hs: any non-finite entry in
    // batch k zeroes d_scaling_success[k]. batch_stride = totalPsdHsEntries.
    if (d_scaling_success && cones.totalPsdHsEntries > 0) {
        check_finite_kernel(
            cones.psd_Hs.data(),
            batchSize * cones.totalPsdHsEntries,
            cones.totalPsdHsEntries,
            d_scaling_success,
            stream);
    }
}

// ============================================================================
// Host-side PSD operations
// ============================================================================

void check_finite_kernel(const double* data, int64_t n, int64_t batch_stride,
                         int32_t* d_flags, cudaStream_t stream) {
    int blk = (n + 255) / 256;
    MOREAU_KERNEL_LAUNCH(check_finite_kernel_impl, blk, 256, 0, stream, data, n, batch_stride, d_flags);
}

void psd_scaled_unit_shift(
    double* z,
    double alpha,
    int64_t psd_offset,
    int64_t m_total,
    const Cones& cones,
    cudaStream_t stream
) {
    if (cones.numPsdCones == 0) return;
    dim3 grid(cones.batchSize, cones.numPsdCones);
    MOREAU_KERNEL_LAUNCH(psd_scaled_unit_shift_kernel, grid, 256, 0, stream,
        z, alpha, psd_offset,
        cones.d_psd_dims, cones.d_psd_sz_offsets,
        cones.numPsdCones, m_total
    );
}

// ============================================================================
// PSD smoothing for warm start
// ============================================================================

/**
 * @brief PSD smoothing kernel: eigendecomp work = z - s, smooth eigenvalues,
 * reconstruct z.
 *
 * For 1×1 PSD cones: z = (w + √(w²+4μ))/2 where w = z - s
 * For general: eigendecomp W = QΛQ', smooth λ_i, reconstruct.
 *
 * This simple kernel handles 1×1 inline. Larger dims need cuSOLVER eigendecomp.
 */
// 1×1 PSD smoothing kernel (scalar case)
__global__ void psd_smoothing_1x1_kernel(
    double* __restrict__ z,
    const double* __restrict__ work,
    const double* __restrict__ mu,
    int64_t psd_offset,
    const int64_t* __restrict__ d_dims,
    const int64_t* __restrict__ d_sz_offsets,
    int64_t numPsdCones,
    int64_t m_total
) {
    int64_t batch = blockIdx.x;
    int64_t cone = blockIdx.y;
    if (cone >= numPsdCones) return;
    if (d_dims[cone] != 1) return;

    if (threadIdx.x == 0) {
        int64_t base = batch * m_total + psd_offset + d_sz_offsets[cone];
        double w = work[base];
        double four_mu = 4.0 * mu[batch];
        z[base] = 0.5 * (w + sqrt(w * w + four_mu));
    }
}

// ============================================================================
// PSD step length
// ============================================================================

// Kernel: scale matrix columns and rows by Λisqrt: M[i,j] *= Lisqrt[i] * Lisqrt[j]
__global__ void lrscale_kernel(double* __restrict__ M, const double* __restrict__ scale, int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    int64_t i = idx % n;
    int64_t j = idx / n;
    M[idx] *= scale[i] * scale[j];
}

// Kernel: find minimum eigenvalue, write to alpha via atomicMin pattern
__global__ void min_eigval_to_alpha_kernel(
    double* __restrict__ alpha,
    const double* __restrict__ eigvals,
    int64_t n,
    double alpha_max
) {
    // Single thread — PSD step length is per-cone, called sequentially
    if (threadIdx.x != 0) return;
    double min_eig = eigvals[0];
    for (int64_t i = 1; i < n; i++) {
        if (eigvals[i] < min_eig) min_eig = eigvals[i];
    }
    double this_alpha = (min_eig < 0.0) ? fmin(-1.0 / min_eig, alpha_max) : alpha_max;
    // Atomic min with current alpha (across all cones)
    // Use atomicMin pattern via CAS for doubles
    unsigned long long int* addr = (unsigned long long int*)alpha;
    unsigned long long int old_val = *addr;
    unsigned long long int new_val;
    do {
        double old_d = __longlong_as_double(old_val);
        double new_d = fmin(old_d, this_alpha);
        new_val = __double_as_longlong(new_d);
        old_val = atomicCAS(addr, old_val, new_val);
    } while (old_val != new_val);
}

/**
 * @brief PSD step length — fully on-device, matching CPU algorithm.
 *
 * For each PSD cone:
 *   αz: d = W·Δz = R^T · mat(dz) · R, scale by Λ^{-1/2}, eigendecomp → min eig
 *   αs: d = W^{-T}·Δs = Rinv · mat(ds) · Rinv^T, scale by Λ^{-1/2}, eigendecomp → min eig
 * Then α = min(-1/γ, αmax) where γ = min eigenvalue.
 *
 * No host-device transfers in this implementation.
 */
void psd_step_length(
    Cones& cones,
    const double* dz,
    const double* ds,
    const double* z,
    const double* s,
    double* alpha_z,
    double* alpha_s,
    int64_t psd_offset,
    int64_t m_total,
    cudaStream_t stream
) {
    if (cones.numPsdCones == 0) return;

    cusolverDnHandle_t cusolver = cones.cusolverH_;
    cublasHandle_t cublas = cones.cublasH_;
    cusolverDnSetStream(cusolver, stream);
    cublasSetStream_v2(cublas, stream);

    int64_t totalMatSqDim = cones.totalPsdMatSqDim;
    int64_t totalMatDim = cones.totalPsdMatDim;
    int64_t numCones = cones.numPsdCones;

    // Step 1: Convert svec(dz) → dense matrices (for all cones/batches at once)
    // Use work_mat1 for dz matrices, work_mat2 for ds matrices
    {
        dim3 grid(cones.batchSize, numCones);
        MOREAU_KERNEL_LAUNCH(svec_to_mat_kernel, grid, 256, 0, stream,
            cones.psd_work_mat1.data(), nullptr, dz, psd_offset, m_total,
            cones.d_psd_dims, cones.d_psd_svec_offsets, cones.d_psd_sz_offsets,
            cones.d_psd_matsq_offsets, numCones, true
        );
        MOREAU_KERNEL_LAUNCH(svec_to_mat_kernel, grid, 256, 0, stream,
            cones.psd_work_mat2.data(), nullptr, ds, psd_offset, m_total,
            cones.d_psd_dims, cones.d_psd_svec_offsets, cones.d_psd_sz_offsets,
            cones.d_psd_matsq_offsets, numCones, true
        );
    }

    // Step 2: For each cone, apply W-scaling and eigendecomp on device
    int64_t matsq_off = 0, mat_off = 0;
    double alpha_one = 1.0, beta_zero = 0.0;

    for (int64_t cone = 0; cone < numCones; cone++) {
        int64_t n = cones.psdConeDims[cone];
        if (n <= 0) { matsq_off += n * n; mat_off += n; continue; }
        int64_t n2 = n * n;
        int blk_n2 = (n2 + 255) / 256;

        for (int64_t b = 0; b < cones.batchSize; b++) {
            double* R_ptr = cones.psd_R.data() + b * totalMatSqDim + matsq_off;
            double* Rinv_ptr = cones.psd_Rinv.data() + b * totalMatSqDim + matsq_off;
            double* Lisqrt_ptr = cones.psd_Lambdaisqrt.data() + b * totalMatDim + mat_off;
            double* dz_mat = cones.psd_work_mat1.data() + b * totalMatSqDim + matsq_off;
            double* ds_mat = cones.psd_work_mat2.data() + b * totalMatSqDim + matsq_off;
            double* work = cones.psd_work_mat3.data() + b * totalMatSqDim + matsq_off;
            // Use psd_work_svec as scratch for eigenvalues (n doubles, fits easily)
            double* eigvals = cones.psd_work_svec.data() + b * cones.totalPsdSvecDim;
            int* d_info = cones.d_psd_info_;

            // αz: d = R^T · mat(dz) · R, then lrscale by Λ^{-1/2}, eigendecomp
            // work = R^T · dz_mat
            cublasDgemm_v2(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                          n, n, n, &alpha_one, R_ptr, n, dz_mat, n, &beta_zero, work, n);
            // dz_mat = work · R = R^T · dz_mat · R
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, n, n, &alpha_one, work, n, R_ptr, n, &beta_zero, dz_mat, n);
            // Scale: dz_mat[i,j] *= Lisqrt[i] * Lisqrt[j]
            MOREAU_KERNEL_LAUNCH(lrscale_kernel, blk_n2, 256, 0, stream, dz_mat, Lisqrt_ptr, n);
            // Eigendecomp (eigenvalues only)
            cudaMemsetAsync(d_info, 0, sizeof(int), stream);
            cusolverDnDsyevd(cusolver, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                            n, dz_mat, n, eigvals,
                            cones.d_psd_syevd_work_, cones.psd_syevd_work_size_, d_info);
            // α = min(-1/min_eig, αmax) via atomic min on alpha_z
            MOREAU_KERNEL_LAUNCH(min_eigval_to_alpha_kernel, 1, 1, 0, stream, alpha_z + b, eigvals, n, 1e30);

            // αs: d = Rinv · mat(ds) · Rinv^T, then lrscale by Λ^{-1/2}, eigendecomp
            // work = ds_mat · Rinv^T
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                          n, n, n, &alpha_one, ds_mat, n, Rinv_ptr, n, &beta_zero, work, n);
            // ds_mat = Rinv · work = Rinv · ds_mat · Rinv^T
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                          n, n, n, &alpha_one, Rinv_ptr, n, work, n, &beta_zero, ds_mat, n);
            // Scale
            MOREAU_KERNEL_LAUNCH(lrscale_kernel, blk_n2, 256, 0, stream, ds_mat, Lisqrt_ptr, n);
            // Eigendecomp
            cudaMemsetAsync(d_info, 0, sizeof(int), stream);
            cusolverDnDsyevd(cusolver, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER,
                            n, ds_mat, n, eigvals,
                            cones.d_psd_syevd_work_, cones.psd_syevd_work_size_, d_info);
            MOREAU_KERNEL_LAUNCH(min_eigval_to_alpha_kernel, 1, 1, 0, stream, alpha_s + b, eigvals, n, 1e30);
        }

        matsq_off += n * n;
        mat_off += n;
    }
}

// ============================================================================
// Smoothing
// ============================================================================

// Smooth eigenvalues: λ_i' = (λ_i + √(λ_i² + 4μ)) / 2
__global__ void smooth_eigenvalues_kernel(
    double* __restrict__ eigvals, const double* __restrict__ mu_ptr, int64_t n, int64_t batch
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double w = eigvals[i];
    double four_mu = 4.0 * mu_ptr[batch];
    eigvals[i] = 0.5 * (w + sqrt(w * w + four_mu));
}

// Scale columns: out[:,k] = Q[:,k] * scale[k]
__global__ void smoothing_scale_columns_kernel(
    double* __restrict__ out, const double* __restrict__ Q, const double* __restrict__ scale, int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    out[idx] = Q[idx] * scale[idx / n];
}

void psd_smoothing(
    Cones& cones,
    double* z,
    const double* work,
    const double* mu,
    int64_t psd_offset,
    int64_t m_total,
    cudaStream_t stream
) {
    if (cones.numPsdCones == 0) return;

    // 1×1 cones: device kernel
    dim3 grid_1x1(cones.batchSize, cones.numPsdCones);
    MOREAU_KERNEL_LAUNCH(psd_smoothing_1x1_kernel, grid_1x1, 256, 0, stream,
        z, work, mu, psd_offset,
        cones.d_psd_dims, cones.d_psd_sz_offsets,
        cones.numPsdCones, m_total
    );

    // Check for dim > 1
    bool has_general = false;
    for (int64_t i = 0; i < cones.numPsdCones; i++)
        if (cones.psdConeDims[i] > 1) { has_general = true; break; }
    if (!has_general) return;

    // Host-side cuSOLVER path for dim > 1
    cusolverDnHandle_t cusolver = cones.cusolverH_;
    cublasHandle_t cublas = cones.cublasH_;
    cusolverDnSetStream(cusolver, stream);
    cublasSetStream_v2(cublas, stream);

    int64_t totalMatSqDim = cones.totalPsdMatSqDim;
    int64_t totalMatDim = cones.totalPsdMatDim;
    int64_t matsq_off = 0, mat_off = 0;

    // Reconstruct host-side sz_offsets (same as initialize computed)
    std::vector<int64_t> orig_svec_offsets(cones.psdConeDimsOriginal.size() + 1, 0);
    for (size_t i = 0; i < cones.psdConeDimsOriginal.size(); ++i) {
        int64_t n = cones.psdConeDimsOriginal[i];
        orig_svec_offsets[i + 1] = orig_svec_offsets[i] + n * (n + 1) / 2;
    }
    std::vector<int64_t> sz_offsets(cones.numPsdCones);
    for (int64_t i = 0; i < cones.numPsdCones; ++i)
        sz_offsets[i] = orig_svec_offsets[cones.psdSortPerm[i]];

    for (int64_t cone = 0; cone < cones.numPsdCones; cone++) {
        int64_t n = cones.psdConeDims[cone];
        int64_t n2 = n * n;
        int64_t svec_dim = n * (n + 1) / 2;
        int64_t sz_off = sz_offsets[cone];

        if (n <= 1) {
            matsq_off += n2; mat_off += n;
            continue;
        }

        int blk_n2 = (n2 + 255) / 256;
        int blk_n = (n + 255) / 256;

        for (int64_t b = 0; b < cones.batchSize; b++) {
            // Workspace pointers (reuse existing Cones workspace)
            double* mat_ptr = cones.psd_work_mat1.data() + b * totalMatSqDim + matsq_off;
            double* work_mat = cones.psd_work_mat2.data() + b * totalMatSqDim + matsq_off;
            double* eigvals = cones.psd_lambda.data() + b * totalMatDim + mat_off;

            // Step 1: expand svec into a dense symmetric matrix for cuSOLVER.
            int64_t base = b * m_total + psd_offset + sz_off;
            {
                cudaMemsetAsync(mat_ptr, 0, sizeof(double) * n2, stream);
                cudaStreamSynchronize(stream);

                // Read svec to host, fill matrix on host, copy back
                std::vector<double> svec_h(svec_dim);
                cudaMemcpy(svec_h.data(), work + base, sizeof(double) * svec_dim,
                          cudaMemcpyDeviceToHost);

                std::vector<double> mat_h(n2, 0.0);
                int64_t idx = 0;
                for (int64_t j = 0; j < n; j++) {
                    for (int64_t i = 0; i <= j; i++) {
                        double val = svec_h[idx++];
                        if (i == j) {
                            mat_h[j * n + i] = val;
                        } else {
                            double unscaled = val * INV_SQRT2;
                            mat_h[j * n + i] = unscaled;
                            mat_h[i * n + j] = unscaled;
                        }
                    }
                }
                cudaMemcpyAsync(mat_ptr, mat_h.data(), sizeof(double) * n2,
                               cudaMemcpyHostToDevice, stream);
            }

            // Step 2: eigendecompose — mat_ptr overwritten with eigenvectors
            // Query workspace (reuse existing approach from update_psd_scaling)
            int work_size = 0;
            cusolverDnDsyevd_bufferSize(cusolver, CUSOLVER_EIG_MODE_VECTOR,
                                         CUBLAS_FILL_MODE_LOWER, n, mat_ptr, n,
                                         eigvals, &work_size);
            double* d_work_buf = nullptr;
            int* d_info = nullptr;
            cudaMallocAsync(&d_work_buf, sizeof(double) * work_size, stream);
            cudaMallocAsync(&d_info, sizeof(int), stream);

            cusolverDnDsyevd(cusolver, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                            n, mat_ptr, n, eigvals, d_work_buf, work_size, d_info);

            // Step 3: smooth eigenvalues: λ_i' = (λ_i + √(λ_i² + 4μ)) / 2
            MOREAU_KERNEL_LAUNCH(smooth_eigenvalues_kernel, blk_n, 256, 0, stream, eigvals, mu, n, b);

            // Step 4: reconstruct Q * diag(λ') * Q^T
            // Q_scaled = Q * diag(λ')
            MOREAU_KERNEL_LAUNCH(smoothing_scale_columns_kernel, blk_n2, 256, 0, stream,
                work_mat, mat_ptr, eigvals, n);

            // result = Q_scaled * Q^T → mat_ptr (reuse)
            double alpha_one = 1.0, beta_zero = 0.0;
            cublasDgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T,
                          n, n, n, &alpha_one, work_mat, n, mat_ptr, n,
                          &beta_zero, cones.psd_work_mat3.data() + b * totalMatSqDim + matsq_off, n);

            // Step 5: mat_to_svec → z
            {
                double* result_mat = cones.psd_work_mat3.data() + b * totalMatSqDim + matsq_off;
                cudaStreamSynchronize(stream);

                std::vector<double> result_h(n2);
                cudaMemcpy(result_h.data(), result_mat, sizeof(double) * n2,
                          cudaMemcpyDeviceToHost);

                std::vector<double> svec_out(svec_dim);
                int64_t idx = 0;
                for (int64_t j = 0; j < n; j++) {
                    for (int64_t i = 0; i <= j; i++) {
                        if (i == j) {
                            svec_out[idx++] = result_h[j * n + i];
                        } else {
                            svec_out[idx++] = result_h[j * n + i] * SQRT2;
                        }
                    }
                }
                cudaMemcpyAsync(z + base, svec_out.data(), sizeof(double) * svec_dim,
                               cudaMemcpyHostToDevice, stream);
            }

            cudaFreeAsync(d_work_buf, stream);
            cudaFreeAsync(d_info, stream);
        }

        matsq_off += n2; mat_off += n;
    }
}

} // namespace moreau
