/**
 * @file kernels.cuh
 * @brief CUDA kernel declarations for residual computations
 *
 * This header defines GPU kernels for computing KKT residuals
 * and infeasibility certificates.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace moreau {

// Forward declarations
struct Residuals;
struct CSR;
struct BatchedVector;

/**
 * @brief Accumulate x[J]·z_x into per-batch dot_sz.
 *
 * Direct-x cones introduce x[J]'z_x complementarity into the central
 * μ = (s'z + x[J]'z_x + τ·κ) / (degree + 1). Rather than plumb a
 * second scalar through every call site, we fold the direct-x
 * contribution into the same dot_sz slot that slack s'z writes.
 *
 *   dot_sz[b] += Σ_k x[b][d_xcone_indices[k]] * z_x[b][k]
 */
void accumulate_dot_xJ_zx(
    double* dot_sz,
    const double* x,
    const double* z_x,
    const int64_t* d_xcone_indices,
    int64_t batchSize,
    int64_t n,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0
);

/**
 * @brief Scatter-add direct-x cone duals into the primal-infeasibility
 *        residual rx_inf.
 *
 * The HSDE primal-infeasibility certificate fires on
 * `‖A^T z − Σ_J E_J^T z_x‖ → 0` with `b^T z < 0`. To make the
 * downstream `rx_inf`-norm test capture that certificate residual on
 * direct-x problems, we fold the `+Σ_J E_J^T z_x` contribution into
 * `rx_inf` (the original SpMV leaves it as `−A^T z`). The full residual
 * `rx = rx_inf − Px − qτ` then naturally inherits the direct-x term.
 *
 * Indices across x-cones are disjoint by API contract so no atomics are
 * needed.
 *
 *   rx_inf[batch][d_xcone_indices[k]] += z_x[batch][k]
 *     for k ∈ [0, totalXConeNumel)
 */
void scatter_add_z_x_to_rx_inf(
    double* rx_inf,
    const double* z_x,
    const int64_t* d_xcone_indices,
    int64_t batchSize,
    int64_t n,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0
);

/**
 * @brief Complete residuals computation (no bounds)
 *
 * Computes:
 * - rx = rx_inf - Px - q*τ
 * - rz = rz_inf - b*τ
 * - rτ = qx + bz + κ + xPx/τ
 */
void completeResidualsNoBounds(
    int64_t n, int64_t m, int64_t batchSize,
    const double* d_rx_inf, const double* d_Px, const double* d_q, const double* d_tau, double* d_rx,
    const double* d_rz_inf, const double* d_b, double* d_rz,
    const double* d_dot_qx, const double* d_dot_bz, const double* d_dot_xPx, const double* d_kappa, double* d_rtau,
    cudaStream_t stream = 0
);

/**
 * @brief Fused: compute 4 dot products + complete residuals in one kernel launch
 *
 * Computes:
 * - dot_qx = q'x, dot_bz = b'z, dot_sz = s'z, dot_xPx = x'Px
 * - rx = rx_inf - Px - q*τ
 * - rz = rz_inf - b*τ
 * - rτ = dot_qx + dot_bz + κ + dot_xPx/τ
 *
 * Eliminates 5 kernel launches (4 dots + complete residuals) into 1.
 */
void fusedCompleteResidualsAndDots(
    int64_t n, int64_t m, int64_t batchSize,
    const double* d_q, const double* d_x,
    const double* d_b, const double* d_z,
    const double* d_s,
    const double* d_Px,
    const double* d_rx_inf, double* d_rx,
    const double* d_rz_inf, double* d_rz,
    const double* d_tau, const double* d_kappa,
    double* d_rtau,
    double* d_dot_qx, double* d_dot_bz,
    double* d_dot_sz, double* d_dot_xPx,
    cudaStream_t stream = 0
);

/**
 * @brief Compute per-batch dot products: dot[b] = x[b]^T * y[b]
 *
 * @param n Vector dimension
 * @param batchSize Number of batches
 * @param x First vector [n * batchSize]
 * @param y Second vector [n * batchSize]
 * @param result Output dot products [batchSize]
 * @param stream CUDA stream
 */
void dotProductBatched(
    int64_t n, int64_t batchSize,
    const double* x, const double* y, double* result,
    cudaStream_t stream = 0
);

/**
 * @brief Batched CSR SpMV: y[b] = alpha * A[b] * x[b] + beta * y[b]
 *
 * @param nrows Number of rows
 * @param ncols Number of columns
 * @param nnz Number of nonzeros
 * @param batchSize Number of batches
 * @param rowOffsets CSR row offsets [nrows+1]
 * @param colIndices CSR column indices [nnz]
 * @param values CSR values [nnz * batchSize]
 * @param x Input vector [ncols * batchSize]
 * @param y Output vector [nrows * batchSize]
 * @param alpha Scalar multiplier for Ax
 * @param beta Scalar multiplier for y
 * @param stream CUDA stream
 */
void csrSpMVBatched(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* rowOffsets, const int64_t* colIndices, const double* values,
    const double* x, double* y,
    double alpha, double beta,
    cudaStream_t stream = 0
);

/**
 * @brief Batched CSR SpMV with y initialization: y[b] = alpha * A[b] * x[b] + y_init[b]
 *
 * Eliminates a separate D2D memcpy by initializing y from y_init in the same kernel.
 * Equivalent to: memcpy(y, y_init), then csrSpMVBatched(..., alpha, 1.0)
 */
void csrSpMVBatchedWithInit(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* rowOffsets, const int64_t* colIndices, const double* values,
    const double* x, double* y, const double* y_init,
    double alpha,
    cudaStream_t stream = 0
);

/**
 * @brief Batched CSR SpMV with transpose: y[b] = alpha * A[b]^T * x[b] + beta * y[b]
 *
 * Uses precomputed row indices to eliminate binary search overhead.
 *
 * @param nrows Number of rows in A (ncols in A^T)
 * @param ncols Number of columns in A (nrows in A^T)
 * @param nnz Number of nonzeros
 * @param batchSize Number of batches
 * @param rowOf Precomputed row index for each nonzero [nnz]
 * @param colIndices CSR column indices [nnz]
 * @param values CSR values [nnz * batchSize]
 * @param x Input vector [nrows * batchSize] (rows of A)
 * @param y Output vector [ncols * batchSize] (cols of A)
 * @param alpha Scalar multiplier for A^T x
 * @param beta Scalar multiplier for y
 * @param stream CUDA stream
 */
void csrSpMVTransposeBatched(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* rowOf, const int64_t* colIndices, const double* values,
    const double* x, double* y,
    double alpha, double beta,
    cudaStream_t stream = 0
);

/**
 * @brief CSC-based batched transpose SpMV: y[b] = alpha * A[b]^T * x[b] + beta * y[b]
 *
 * Uses precomputed CSC structure (A^T in CSR format) for row-based parallelism
 * without atomics. Each thread handles one column of A (= one row of A^T).
 *
 * @param nrows Number of rows in A
 * @param ncols Number of columns in A (= output dimension)
 * @param nnz Number of nonzeros
 * @param batchSize Number of batches
 * @param At_rowOffsets CSC column offsets [ncols+1]
 * @param At_colIndices CSC row indices [nnz]
 * @param At_val_perm Permutation mapping A^T positions to A.values positions [nnz]
 * @param A_values Original A values [nnz * batchSize]
 * @param x Input vector [nrows * batchSize]
 * @param y Output vector [ncols * batchSize]
 * @param alpha Scalar multiplier for A^T x
 * @param beta Scalar multiplier for y
 * @param stream CUDA stream
 */
void csrSpMVTransposeBatched_CSC(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* At_rowOffsets, const int64_t* At_colIndices,
    const int64_t* At_val_perm, const double* A_values,
    const double* x, double* y,
    double alpha, double beta,
    cudaStream_t stream = 0
);

/**
 * @brief Broadcast CSR SpMV: y[b] = alpha * A * x[b] + beta * y[b]
 *
 * Single matrix A is broadcast across all batches.
 *
 * @param nrows Number of rows
 * @param ncols Number of columns
 * @param nnz Number of nonzeros
 * @param batchSize Number of batches
 * @param rowOffsets CSR row offsets [nrows+1]
 * @param colIndices CSR column indices [nnz]
 * @param values CSR values [nnz] (NOT batched - single copy)
 * @param x Input vector [ncols * batchSize] (batched)
 * @param y Output vector [nrows * batchSize] (batched)
 * @param alpha Scalar multiplier for Ax
 * @param beta Scalar multiplier for y
 * @param stream CUDA stream
 */
void csrSpMVBroadcast(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* rowOffsets, const int64_t* colIndices, const double* values,
    const double* x, double* y,
    double alpha, double beta,
    cudaStream_t stream = 0
);

/**
 * @brief Broadcast CSR SpMV with transpose: y[b] = alpha * A^T * x[b] + beta * y[b]
 *
 * Single matrix A is broadcast across all batches.
 *
 * @param nrows Number of rows in A (ncols in A^T)
 * @param ncols Number of columns in A (nrows in A^T)
 * @param nnz Number of nonzeros
 * @param batchSize Number of batches
 * @param rowOffsets CSR row offsets [nrows+1]
 * @param colIndices CSR column indices [nnz]
 * @param values CSR values [nnz] (NOT batched - single copy)
 * @param x Input vector [nrows * batchSize] (batched, rows of A)
 * @param y Output vector [ncols * batchSize] (batched, cols of A)
 * @param alpha Scalar multiplier for A^T x
 * @param beta Scalar multiplier for y
 * @param stream CUDA stream
 */
void csrSpMVTransposeBroadcast(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* rowOffsets, const int64_t* colIndices, const double* values,
    const double* x, double* y,
    double alpha, double beta,
    cudaStream_t stream = 0
);

} // namespace moreau
