/**
 * @file kernels.cu
 * @brief CUDA kernel implementations for residual computations
 *
 * Implements GPU kernels for computing KKT residuals
 * and infeasibility certificates.
 */

#include "moreau/residuals/residuals_kernels.cuh"
#include "moreau/residuals/residuals.hpp"
#include "moreau/matrix/csr.hpp"
#include "moreau/vector/vector.hpp"
#include "moreau/profiling/profiler.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

namespace moreau {

// Forward declaration for beta-scaling helper used in empty SpMV paths
__global__ void csrSpMVTransposeBatchedKernel_InitY(
    int64_t ncols, int64_t batchSize,
    double* __restrict__ y, double beta
);

// Accumulate x[J]·z_x into dot_sz using a warp-synchronous reduction.
__global__ void accumulate_dot_xJ_zx_kernel(
    double* __restrict__ dot_sz,                     // [batchSize]
    const double* __restrict__ x,                    // [batchSize * n]
    const double* __restrict__ z_x,                  // [batchSize * totalXConeNumel]
    const int64_t* __restrict__ d_xcone_indices,     // [totalXConeNumel]
    int64_t n,
    int64_t totalXConeNumel)
{
    const int64_t batch = blockIdx.x;
    const int64_t x_off  = batch * n;
    const int64_t zx_off = batch * totalXConeNumel;

    __shared__ double partial[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    double sum = 0.0;
    for (int64_t k = threadIdx.x; k < totalXConeNumel; k += blockDim.x) {
        const int64_t idx = d_xcone_indices[k];
        sum += x[x_off + idx] * z_x[zx_off + k];
    }

    // Warp reduction.
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, off);
    }
    if (lane == 0) partial[warp] = sum;
    __syncthreads();

    // First warp reduces partials.
    if (warp == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        double v = (threadIdx.x < num_warps) ? partial[threadIdx.x] : 0.0;
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffff, v, off);
        }
        if (threadIdx.x == 0) {
            dot_sz[batch] += v;
        }
    }
}

void accumulate_dot_xJ_zx(
    double* dot_sz,
    const double* x,
    const double* z_x,
    const int64_t* d_xcone_indices,
    int64_t batchSize,
    int64_t n,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (totalXConeNumel <= 0 || batchSize <= 0) return;
    const int threads = 256;
    dim3 grid(static_cast<unsigned int>(batchSize));
    accumulate_dot_xJ_zx_kernel<<<grid, threads, 0, stream>>>(
        dot_sz, x, z_x, d_xcone_indices, n, totalXConeNumel);
}

__global__ void scatter_add_z_x_to_rx_inf_kernel(
    double* __restrict__ rx_inf,                     // [batchSize * n]
    const double* __restrict__ z_x,                  // [batchSize * totalXConeNumel]
    const int64_t* __restrict__ d_xcone_indices,     // [totalXConeNumel]
    int64_t n,
    int64_t totalXConeNumel)
{
    const int64_t batch = blockIdx.x;
    const int64_t rx_off = batch * n;
    const int64_t zx_off = batch * totalXConeNumel;
    for (int64_t k = threadIdx.x; k < totalXConeNumel; k += blockDim.x) {
        const int64_t idx = d_xcone_indices[k];
        rx_inf[rx_off + idx] += z_x[zx_off + k];
    }
}

void scatter_add_z_x_to_rx_inf(
    double* rx_inf,
    const double* z_x,
    const int64_t* d_xcone_indices,
    int64_t batchSize,
    int64_t n,
    int64_t totalXConeNumel,
    cudaStream_t stream)
{
    if (totalXConeNumel <= 0 || batchSize <= 0) return;
    int threads = static_cast<int>(
        totalXConeNumel < 256 ? totalXConeNumel : 256);
    dim3 grid(static_cast<unsigned int>(batchSize));
    scatter_add_z_x_to_rx_inf_kernel<<<grid, threads, 0, stream>>>(
        rx_inf, z_x, d_xcone_indices, n, totalXConeNumel);
}

/**
 * @brief Complete residuals computation (no bounds)
 *
 * Computes:
 * - rx = rx_inf - Px - q*τ
 * - rz = rz_inf - b*τ
 * - rτ = qx + bz + κ + xPx/τ
 */
__global__ void completeResidualsNoBoundsKernel(
    int64_t n, int64_t m, int64_t batchSize,
    const double* __restrict__ rx_inf, const double* __restrict__ Px, const double* __restrict__ q, const double* __restrict__ tau, double* __restrict__ rx,
    const double* __restrict__ rz_inf, const double* __restrict__ b, double* __restrict__ rz,
    const double* __restrict__ dot_qx, const double* __restrict__ dot_bz, const double* __restrict__ dot_xPx, const double* __restrict__ kappa, double* __restrict__ rtau
) {
    int64_t batch = blockIdx.x;
    int64_t tid = threadIdx.x;

    if (batch >= batchSize) return;

    int64_t batch_offset_n = batch * n;
    int64_t batch_offset_m = batch * m;
    double tau_val = tau[batch];

    // Compute rx = rx_inf - Px - q*τ
    for (int64_t i = tid; i < n; i += blockDim.x) {
        rx[batch_offset_n + i] = rx_inf[batch_offset_n + i]
                                - Px[batch_offset_n + i]
                                - q[batch_offset_n + i] * tau_val;
    }

    // Compute rz = rz_inf - b*τ
    for (int64_t i = tid; i < m; i += blockDim.x) {
        rz[batch_offset_m + i] = rz_inf[batch_offset_m + i]
                                - b[batch_offset_m + i] * tau_val;
    }

    // Compute rτ = qx + bz + κ + xPx/τ (only one thread per batch)
    if (tid == 0) {
        rtau[batch] = dot_qx[batch] + dot_bz[batch] + kappa[batch] + (dot_xPx[batch] / tau_val);
    }
}

void completeResidualsNoBounds(
    int64_t n, int64_t m, int64_t batchSize,
    const double* d_rx_inf, const double* d_Px, const double* d_q, const double* d_tau, double* d_rx,
    const double* d_rz_inf, const double* d_b, double* d_rz,
    const double* d_dot_qx, const double* d_dot_bz, const double* d_dot_xPx, const double* d_kappa, double* d_rtau,
    cudaStream_t stream
) {
    int64_t threadsPerBlock = 256;
    int64_t numBlocks = batchSize;

    MOREAU_KERNEL_LAUNCH(completeResidualsNoBoundsKernel, numBlocks, threadsPerBlock, 0, stream,
        n, m, batchSize,
        d_rx_inf, d_Px, d_q, d_tau, d_rx,
        d_rz_inf, d_b, d_rz,
        d_dot_qx, d_dot_bz, d_dot_xPx, d_kappa, d_rtau);
}

/**
 * @brief Fused residual completion + 4 dot products kernel
 *
 * Computes in one launch:
 * - dot_qx = q'x, dot_bz = b'z, dot_sz = s'z, dot_xPx = x'Px (via reduction)
 * - rx = rx_inf - Px - q*τ
 * - rz = rz_inf - b*τ
 * - rτ = dot_qx + dot_bz + κ + dot_xPx/τ
 *
 * One block per batch. Grid-strides over max(n,m) elements.
 * Eliminates 5 kernel launches (4 dots + complete residuals) into 1.
 */
__global__ void fusedCompleteResidualsAndDotsKernel(
    int64_t n, int64_t m, int64_t batchSize,
    const double* __restrict__ q, const double* __restrict__ x_vec,
    const double* __restrict__ b_vec, const double* __restrict__ z_vec,
    const double* __restrict__ s_vec,
    const double* __restrict__ Px,
    const double* __restrict__ rx_inf, double* __restrict__ rx,
    const double* __restrict__ rz_inf, double* __restrict__ rz,
    const double* __restrict__ tau, const double* __restrict__ kappa,
    double* __restrict__ rtau,
    double* __restrict__ dot_qx, double* __restrict__ dot_bz,
    double* __restrict__ dot_sz, double* __restrict__ dot_xPx
) {
    int64_t batch = blockIdx.x;
    int64_t tid = threadIdx.x;
    if (batch >= batchSize) return;

    int64_t batch_n = batch * n;
    int64_t batch_m = batch * m;
    double tau_val = tau[batch];

    // Accumulate 4 dot products via grid-stride loop
    double sum_qx = 0.0, sum_xPx = 0.0;
    for (int64_t i = tid; i < n; i += blockDim.x) {
        int64_t idx = batch_n + i;
        double qi = q[idx];
        double xi = x_vec[idx];
        double Pxi = Px[idx];
        sum_qx += qi * xi;
        sum_xPx += xi * Pxi;
        // Compute rx while we're reading these values
        rx[idx] = rx_inf[idx] - Pxi - qi * tau_val;
    }

    double sum_bz = 0.0, sum_sz = 0.0;
    for (int64_t i = tid; i < m; i += blockDim.x) {
        int64_t idx = batch_m + i;
        double bi = b_vec[idx];
        double zi = z_vec[idx];
        double si = s_vec[idx];
        sum_bz += bi * zi;
        sum_sz += si * zi;
        // Compute rz while we're reading these values
        rz[idx] = rz_inf[idx] - bi * tau_val;
    }

    // Shared memory reduction for 4 dot products
    __shared__ double sh_qx[256], sh_bz[256], sh_sz[256], sh_xPx[256];
    sh_qx[tid] = sum_qx;
    sh_bz[tid] = sum_bz;
    sh_sz[tid] = sum_sz;
    sh_xPx[tid] = sum_xPx;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sh_qx[tid] += sh_qx[tid + s];
            sh_bz[tid] += sh_bz[tid + s];
            sh_sz[tid] += sh_sz[tid + s];
            sh_xPx[tid] += sh_xPx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        dot_qx[batch] = sh_qx[0];
        dot_bz[batch] = sh_bz[0];
        dot_sz[batch] = sh_sz[0];
        dot_xPx[batch] = sh_xPx[0];
        rtau[batch] = sh_qx[0] + sh_bz[0] + kappa[batch] + (sh_xPx[0] / tau_val);
    }
}

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
    cudaStream_t stream
) {
    if (batchSize == 0) return;

    int64_t threadsPerBlock = 256;
    int64_t numBlocks = batchSize;

    MOREAU_KERNEL_LAUNCH(fusedCompleteResidualsAndDotsKernel, numBlocks, threadsPerBlock, 0, stream,
        n, m, batchSize,
        d_q, d_x, d_b, d_z, d_s, d_Px,
        d_rx_inf, d_rx, d_rz_inf, d_rz,
        d_tau, d_kappa, d_rtau,
        d_dot_qx, d_dot_bz, d_dot_sz, d_dot_xPx);
}

/**
 * @brief Kernel to compute per-batch dot products
 */
__global__ void dotProductBatchedKernel(
    int64_t n, int64_t batchSize,
    const double* __restrict__ x, const double* __restrict__ y, double* __restrict__ result
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    // Each block computes one dot product for its batch
    __shared__ double shared[256];
    int64_t tid = threadIdx.x;
    int64_t batch_offset = batch * n;

    // Parallel reduction within block
    double sum = 0.0;
    for (int64_t i = tid; i < n; i += blockDim.x) {
        sum += x[batch_offset + i] * y[batch_offset + i];
    }
    shared[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (int64_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        result[batch] = shared[0];
    }
}

void dotProductBatched(
    int64_t n, int64_t batchSize,
    const double* x, const double* y, double* result,
    cudaStream_t stream
) {
    int64_t threadsPerBlock = 256;
    int64_t numBlocks = batchSize;

    MOREAU_KERNEL_LAUNCH(dotProductBatchedKernel, numBlocks, threadsPerBlock, 0, stream,
        n, batchSize, x, y, result);
}

/**
 * @brief Batched CSR SpMV kernel: y[b] = alpha * A[b] * x[b] + beta * y[b]
 */
__global__ void csrSpMVBatchedKernel(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* __restrict__ rowOffsets, const int64_t* __restrict__ colIndices, const double* __restrict__ values,
    const double* __restrict__ x, double* __restrict__ y,
    double alpha, double beta
) {
    int64_t batch = blockIdx.y;
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batchSize || row >= nrows) return;

    int64_t batch_offset_val = batch * nnz;
    int64_t batch_offset_x = batch * ncols;
    int64_t batch_offset_y = batch * nrows;

    int64_t row_start = rowOffsets[row];
    int64_t row_end = rowOffsets[row + 1];

    double sum = 0.0;
    for (int64_t idx = row_start; idx < row_end; ++idx) {
        int64_t col = colIndices[idx];
        sum += values[batch_offset_val + idx] * x[batch_offset_x + col];
    }

    if (beta == 0.0) {
        y[batch_offset_y + row] = alpha * sum;
    } else {
        y[batch_offset_y + row] = alpha * sum + beta * y[batch_offset_y + row];
    }
}

void csrSpMVBatched(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* rowOffsets, const int64_t* colIndices, const double* values,
    const double* x, double* y,
    double alpha, double beta,
    cudaStream_t stream
) {
    if (nrows == 0 || batchSize == 0) return;
    if (nnz == 0) {
        // Preserve beta semantics for empty matrices
        if (beta == 0.0) {
            cudaMemsetAsync(y, 0, sizeof(double) * nrows * batchSize, stream);
        } else if (beta != 1.0) {
            // y = beta * y
            dim3 blk(256, 1, 1);
            dim3 grd((nrows + blk.x - 1) / blk.x, batchSize, 1);
            MOREAU_KERNEL_LAUNCH(csrSpMVTransposeBatchedKernel_InitY, grd, blk, 0, stream,
                nrows, batchSize, y, beta);
        }
        return;
    }
    int64_t threadsPerBlock = 256;
    int64_t numBlocksRows = (nrows + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(numBlocksRows, batchSize);
    dim3 blockDim(threadsPerBlock);

    MOREAU_KERNEL_LAUNCH(csrSpMVBatchedKernel, gridDim, blockDim, 0, stream,
        nrows, ncols, nnz, batchSize,
        rowOffsets, colIndices, values,
        x, y, alpha, beta);
}

// ============================================================================
// SpMV with y initialization from separate source (eliminates D2D memcpy)
// ============================================================================
__global__ void csrSpMVBatchedWithInitKernel(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* __restrict__ rowOffsets, const int64_t* __restrict__ colIndices, const double* __restrict__ values,
    const double* __restrict__ x, double* __restrict__ y, const double* __restrict__ y_init,
    double alpha
) {
    int64_t batch = blockIdx.y;
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batchSize || row >= nrows) return;

    int64_t batch_offset_val = batch * nnz;
    int64_t batch_offset_x = batch * ncols;
    int64_t batch_offset_y = batch * nrows;

    int64_t row_start = rowOffsets[row];
    int64_t row_end = rowOffsets[row + 1];

    double sum = 0.0;
    for (int64_t idx = row_start; idx < row_end; ++idx) {
        int64_t col = colIndices[idx];
        sum += values[batch_offset_val + idx] * x[batch_offset_x + col];
    }

    y[batch_offset_y + row] = alpha * sum + y_init[batch_offset_y + row];
}

void csrSpMVBatchedWithInit(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* rowOffsets, const int64_t* colIndices, const double* values,
    const double* x, double* y, const double* y_init,
    double alpha,
    cudaStream_t stream
) {
    if (nrows == 0 || batchSize == 0) return;
    if (nnz == 0) {
        // No matrix entries, just copy y_init
        cudaMemcpyAsync(y, y_init, sizeof(double) * nrows * batchSize,
                        cudaMemcpyDeviceToDevice, stream);
        return;
    }
    int64_t threadsPerBlock = 256;
    int64_t numBlocksRows = (nrows + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(numBlocksRows, batchSize);
    dim3 blockDim(threadsPerBlock);

    MOREAU_KERNEL_LAUNCH(csrSpMVBatchedWithInitKernel, gridDim, blockDim, 0, stream,
        nrows, ncols, nnz, batchSize,
        rowOffsets, colIndices, values,
        x, y, y_init, alpha);
}

/**
 * @brief Batched CSR SpMV transpose kernel: y[b] = alpha * A[b]^T * x[b] + beta * y[b]
 *
 * Optimized version using atomics with element-based parallelization.
 * Each thread processes one non-zero element, computing: y[col] += alpha * value * x[row]
 *
 * This approach is much more efficient than the previous column-based approach:
 * - O(nnz) total work instead of O(nrows × nnz)
 * - Coalesced memory access for values and column indices
 * - Minimal atomic contention for sparse matrices
 */
__global__ void csrSpMVTransposeBatchedKernel_InitY(
    int64_t ncols, int64_t batchSize,
    double* __restrict__ y, double beta
) {
    int64_t batch = blockIdx.y;
    int64_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batchSize || col_idx >= ncols) return;

    int64_t batch_offset_y = batch * ncols;

    // Initialize output with beta * y
    if (beta == 0.0) {
        y[batch_offset_y + col_idx] = 0.0;
    } else {
        y[batch_offset_y + col_idx] *= beta;
    }
}

__global__ void csrSpMVTransposeBatchedKernel(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* __restrict__ rowOf, const int64_t* __restrict__ colIndices, const double* __restrict__ values,
    const double* __restrict__ x, double* __restrict__ y,
    double alpha
) {
    // Each thread handles one non-zero element across all batches
    int64_t batch = blockIdx.y;
    int64_t elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batchSize || elem_idx >= nnz) return;

    // Use precomputed row index (eliminates O(log n) binary search)
    int64_t row = rowOf[elem_idx];
    int64_t col = colIndices[elem_idx];

    int64_t batch_offset_val = batch * nnz;
    int64_t batch_offset_x = batch * nrows;
    int64_t batch_offset_y = batch * ncols;

    // Compute contribution: y[col] += alpha * value * x[row]
    double contrib = alpha * values[batch_offset_val + elem_idx] * x[batch_offset_x + row];
    atomicAdd(&y[batch_offset_y + col], contrib);
}

void csrSpMVTransposeBatched(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* rowOf, const int64_t* colIndices, const double* values,
    const double* x, double* y,
    double alpha, double beta,
    cudaStream_t stream
) {
    if (ncols == 0 || batchSize == 0) return;
    if (nnz == 0) {
        if (beta == 0.0) {
            cudaMemsetAsync(y, 0, sizeof(double) * ncols * batchSize, stream);
        } else if (beta != 1.0) {
            int64_t threadsPerBlock = 256;
            int64_t numBlocksCols = (ncols + threadsPerBlock - 1) / threadsPerBlock;
            dim3 gridDimInit(numBlocksCols, batchSize);
            dim3 blockDimInit(threadsPerBlock);

            MOREAU_KERNEL_LAUNCH(csrSpMVTransposeBatchedKernel_InitY, gridDimInit, blockDimInit, 0, stream,
                ncols, batchSize, y, beta);
        }
        return;
    }
    int64_t threadsPerBlock = 256;

    // Step 1: Initialize y with beta * y
    // Optimization: use cudaMemsetAsync when beta == 0, which is the common case
    if (beta == 0.0) {
        cudaMemsetAsync(y, 0, sizeof(double) * ncols * batchSize, stream);
    } else {
        int64_t numBlocksCols = (ncols + threadsPerBlock - 1) / threadsPerBlock;
        dim3 gridDimInit(numBlocksCols, batchSize);
        dim3 blockDimInit(threadsPerBlock);

        MOREAU_KERNEL_LAUNCH(csrSpMVTransposeBatchedKernel_InitY, gridDimInit, blockDimInit, 0, stream,
            ncols, batchSize, y, beta);
    }

    // Step 2: Accumulate contributions from non-zero elements (using precomputed row indices)
    int64_t numBlocksNnz = (nnz + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDimMain(numBlocksNnz, batchSize);
    dim3 blockDimMain(threadsPerBlock);

    MOREAU_KERNEL_LAUNCH(csrSpMVTransposeBatchedKernel, gridDimMain, blockDimMain, 0, stream,
        nrows, ncols, nnz, batchSize,
        rowOf, colIndices, values,
        x, y, alpha);
}

// ============================================================================
// CSC-based transpose SpMV (atomic-free)
// ============================================================================
// Uses precomputed CSC structure (A^T in CSR format).
// Each thread handles one row of A^T (= one column of A), no atomics needed.
__global__ void csrSpMVTransposeBatched_CSC_Kernel(
    int64_t ncols_A,    // = nrows of A^T
    int64_t nnz,
    int64_t batchSize,
    const int64_t* __restrict__ At_rowOffsets,   // [ncols_A + 1]
    const int64_t* __restrict__ At_colIndices,   // [nnz] — row indices of A
    const int64_t* __restrict__ At_val_perm,     // [nnz] — position in A.values
    const double* __restrict__ A_values,         // [nnz * batchSize] — original A values
    const double* __restrict__ x,                // [nrows_A * batchSize] — input
    double* __restrict__ y,                      // [ncols_A * batchSize] — output
    double alpha,
    double beta,
    int64_t nrows_A                 // for x indexing
) {
    int64_t batch = blockIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;  // column of A = row of A^T

    if (batch >= batchSize || col >= ncols_A) return;

    int64_t batch_val = batch * nnz;
    int64_t batch_x = batch * nrows_A;
    int64_t batch_y = batch * ncols_A;

    int64_t row_start = At_rowOffsets[col];
    int64_t row_end = At_rowOffsets[col + 1];

    double sum = 0.0;
    for (int64_t idx = row_start; idx < row_end; ++idx) {
        int64_t row_of_A = At_colIndices[idx];
        int64_t val_pos = At_val_perm[idx];
        sum += A_values[batch_val + val_pos] * x[batch_x + row_of_A];
    }

    if (beta == 0.0) {
        y[batch_y + col] = alpha * sum;
    } else {
        y[batch_y + col] = alpha * sum + beta * y[batch_y + col];
    }
}

void csrSpMVTransposeBatched_CSC(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* At_rowOffsets, const int64_t* At_colIndices,
    const int64_t* At_val_perm, const double* A_values,
    const double* x, double* y,
    double alpha, double beta,
    cudaStream_t stream
) {
    if (ncols == 0 || batchSize == 0) return;
    if (nnz == 0) {
        if (beta == 0.0) {
            cudaMemsetAsync(y, 0, sizeof(double) * ncols * batchSize, stream);
        } else if (beta != 1.0) {
            int64_t threadsPerBlock = 256;
            int64_t numBlocksCols = (ncols + threadsPerBlock - 1) / threadsPerBlock;
            dim3 gridDimInit(numBlocksCols, batchSize);
            dim3 blockDimInit(threadsPerBlock);
            MOREAU_KERNEL_LAUNCH(csrSpMVTransposeBatchedKernel_InitY, gridDimInit, blockDimInit, 0, stream,
                ncols, batchSize, y, beta);
        }
        return;
    }

    int64_t threadsPerBlock = 256;
    int64_t numBlocksCols = (ncols + threadsPerBlock - 1) / threadsPerBlock;
    dim3 gridDim(numBlocksCols, batchSize);
    dim3 blockDim(threadsPerBlock);

    MOREAU_KERNEL_LAUNCH(csrSpMVTransposeBatched_CSC_Kernel, gridDim, blockDim, 0, stream,
        ncols, nnz, batchSize,
        At_rowOffsets, At_colIndices, At_val_perm, A_values,
        x, y, alpha, beta, nrows);
}

/**
 * @brief Broadcast CSR SpMV kernel: y[b] = alpha * A * x[b] + beta * y[b]
 *
 * Matrix A is NOT batched (single copy broadcast across all batches).
 */
__global__ void csrSpMVBroadcastKernel(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* __restrict__ rowOffsets, const int64_t* __restrict__ colIndices, const double* __restrict__ values,
    const double* __restrict__ x, double* __restrict__ y,
    double alpha, double beta
) {
    int64_t batch = blockIdx.y;
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batchSize || row >= nrows) return;

    // Values are NOT batched - use same values for all batches
    int64_t batch_offset_x = batch * ncols;
    int64_t batch_offset_y = batch * nrows;

    int64_t row_start = rowOffsets[row];
    int64_t row_end = rowOffsets[row + 1];

    double sum = 0.0;
    for (int64_t idx = row_start; idx < row_end; ++idx) {
        int64_t col = colIndices[idx];
        sum += values[idx] * x[batch_offset_x + col];  // values[idx] NOT batched
    }

    if (beta == 0.0) {
        y[batch_offset_y + row] = alpha * sum;
    } else {
        y[batch_offset_y + row] = alpha * sum + beta * y[batch_offset_y + row];
    }
}

void csrSpMVBroadcast(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* rowOffsets, const int64_t* colIndices, const double* values,
    const double* x, double* y,
    double alpha, double beta,
    cudaStream_t stream
) {
    if (nrows == 0 || batchSize == 0) return;
    if (nnz == 0) {
        if (beta == 0.0) {
            cudaMemsetAsync(y, 0, sizeof(double) * nrows * batchSize, stream);
        } else if (beta != 1.0) {
            int64_t threadsPerBlock = 256;
            int64_t numBlocksRows = (nrows + threadsPerBlock - 1) / threadsPerBlock;
            dim3 gridDimInit(numBlocksRows, batchSize);
            dim3 blockDimInit(threadsPerBlock);
            MOREAU_KERNEL_LAUNCH(csrSpMVTransposeBatchedKernel_InitY, gridDimInit, blockDimInit, 0, stream,
                nrows, batchSize, y, beta);
        }
        return;
    }
    int64_t threadsPerBlock = 256;
    int64_t numBlocksRows = (nrows + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(numBlocksRows, batchSize);
    dim3 blockDim(threadsPerBlock);

    MOREAU_KERNEL_LAUNCH(csrSpMVBroadcastKernel, gridDim, blockDim, 0, stream,
        nrows, ncols, nnz, batchSize,
        rowOffsets, colIndices, values,
        x, y, alpha, beta);
}

/**
 * @brief Broadcast CSR SpMV transpose kernel: y[b] = alpha * A^T * x[b] + beta * y[b]
 *
 * Matrix A is NOT batched (single copy broadcast across all batches).
 */
__global__ void csrSpMVTransposeBroadcastKernel(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* __restrict__ rowOffsets, const int64_t* __restrict__ colIndices, const double* __restrict__ values,
    const double* __restrict__ x, double* __restrict__ y,
    double alpha, double beta
) {
    int64_t batch = blockIdx.y;
    int64_t col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batchSize || col_idx >= ncols) return;

    // Values are NOT batched - use same values for all batches
    int64_t batch_offset_x = batch * nrows;
    int64_t batch_offset_y = batch * ncols;

    // Initialize output with beta * y
    if (beta == 0.0) {
        y[batch_offset_y + col_idx] = 0.0;
    } else {
        y[batch_offset_y + col_idx] *= beta;
    }

    // Iterate through all rows and accumulate contributions to this column
    for (int64_t row = 0; row < nrows; ++row) {
        int64_t row_start = rowOffsets[row];
        int64_t row_end = rowOffsets[row + 1];

        for (int64_t idx = row_start; idx < row_end; ++idx) {
            if (colIndices[idx] == col_idx) {
                y[batch_offset_y + col_idx] += alpha * values[idx] * x[batch_offset_x + row];  // values[idx] NOT batched
                break;
            }
        }
    }
}

void csrSpMVTransposeBroadcast(
    int64_t nrows, int64_t ncols, int64_t nnz, int64_t batchSize,
    const int64_t* rowOffsets, const int64_t* colIndices, const double* values,
    const double* x, double* y,
    double alpha, double beta,
    cudaStream_t stream
) {
    if (ncols == 0 || batchSize == 0) return;
    if (nnz == 0) {
        if (beta == 0.0) {
            cudaMemsetAsync(y, 0, sizeof(double) * ncols * batchSize, stream);
        } else if (beta != 1.0) {
            int64_t threadsPerBlock = 256;
            int64_t numBlocksCols = (ncols + threadsPerBlock - 1) / threadsPerBlock;
            dim3 gridDimInit(numBlocksCols, batchSize);
            dim3 blockDimInit(threadsPerBlock);
            MOREAU_KERNEL_LAUNCH(csrSpMVTransposeBatchedKernel_InitY, gridDimInit, blockDimInit, 0, stream,
                ncols, batchSize, y, beta);
        }
        return;
    }
    int64_t threadsPerBlock = 256;
    int64_t numBlocksCols = (ncols + threadsPerBlock - 1) / threadsPerBlock;

    dim3 gridDim(numBlocksCols, batchSize);
    dim3 blockDim(threadsPerBlock);

    MOREAU_KERNEL_LAUNCH(csrSpMVTransposeBroadcastKernel, gridDim, blockDim, 0, stream,
        nrows, ncols, nnz, batchSize,
        rowOffsets, colIndices, values,
        x, y, alpha, beta);
}

} // namespace moreau
