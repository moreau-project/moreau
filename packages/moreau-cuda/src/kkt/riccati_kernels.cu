/**
 * @file riccati_kernels.cu
 * @brief CUDA kernels for batched block-tridiagonal (Riccati) KKT solver
 */

#include "moreau/kkt/riccati_kernels.cuh"
#include "moreau/profiling/profiler.hpp"
#include <cusolverDn.h>
#include <vector>

namespace moreau {

// ============================================================================
// h_inv computation
// ============================================================================

static __global__ void compute_h_inv_impl(
    double* __restrict__ h_inv,
    const double* __restrict__ nonneg_w,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t m,
    int64_t batchSize,
    double reg_eps)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * m;
    if (idx >= total) return;

    int64_t b = idx / m;
    int64_t r = idx % m;

    if (r < numZeroCones) {
        h_inv[idx] = 1.0 / reg_eps;
    } else if (r < numZeroCones + numNonnegCones) {
        int64_t nonneg_idx = b * numNonnegCones + (r - numZeroCones);
        double w = nonneg_w[nonneg_idx];
        double w2 = w * w;
        h_inv[idx] = (w2 > 1e-300) ? (1.0 / w2) : (1.0 / reg_eps);
    } else {
        h_inv[idx] = 1.0 / reg_eps;
    }
}

void compute_h_inv_kernel(
    double* h_inv, const double* nonneg_w,
    int64_t numZeroCones, int64_t numNonnegCones,
    int64_t m, int64_t batchSize, double reg_eps, cudaStream_t stream)
{
    int64_t total = batchSize * m;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(compute_h_inv_impl, blocks, threads, 0, stream,
        h_inv, nonneg_w, numZeroCones, numNonnegCones, m, batchSize, reg_eps);
}

// ============================================================================
// Zero blocks
// ============================================================================

static __global__ void zero_blocks_impl(
    double* __restrict__ D_data, double* __restrict__ L_data,
    int64_t D_total, int64_t L_total,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_D = batchSize * D_total;
    int64_t total_L = batchSize * L_total;
    if (idx < total_D) D_data[idx] = 0.0;
    if (idx < total_L) L_data[idx] = 0.0;
}

void zero_blocks_kernel(
    double* D_data, double* L_data,
    int64_t D_total_elems, int64_t L_total_elems,
    int64_t batchSize, cudaStream_t stream)
{
    int64_t total = batchSize * D_total_elems;
    if (batchSize * L_total_elems > total) total = batchSize * L_total_elems;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(zero_blocks_impl, blocks, threads, 0, stream,
        D_data, L_data, D_total_elems, L_total_elems, batchSize);
}

// ============================================================================
// Scatter P values into D and L blocks via pre-computed scatter map
// ============================================================================

static __global__ void scatter_P_impl(
    double* __restrict__ D_data, double* __restrict__ L_data,
    const double* __restrict__ P_values,
    const int32_t* __restrict__ p_type, const int32_t* __restrict__ p_block,
    const int32_t* __restrict__ p_li, const int32_t* __restrict__ p_lj,
    const int32_t* __restrict__ p_symmetric, const int32_t* __restrict__ p_idx,
    const int64_t* __restrict__ D_offsets, const int64_t* __restrict__ L_offsets,
    const int32_t* __restrict__ block_sizes,
    int64_t D_total, int64_t L_total,
    int64_t n_scatter, int64_t nnzP, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * n_scatter;
    if (tid >= total) return;

    int64_t b = tid / n_scatter;
    int64_t s = tid % n_scatter;

    double v = P_values[b * nnzP + p_idx[s]];
    int32_t bi = p_block[s];
    int32_t li = p_li[s];
    int32_t lj = p_lj[s];

    if (p_type[s] == 0) {
        // Diagonal block (col-major)
        int32_t d = block_sizes[bi];
        int64_t offset = b * D_total + D_offsets[bi] + li + (int64_t)lj * d;
        atomicAdd(&D_data[offset], v);
        if (p_symmetric[s]) {
            int64_t offset_sym = b * D_total + D_offsets[bi] + lj + (int64_t)li * d;
            atomicAdd(&D_data[offset_sym], v);
        }
    } else {
        // Sub-diagonal block L[bi] (col-major)
        int32_t rows_L = block_sizes[bi + 1];
        int64_t offset = b * L_total + L_offsets[bi] + li + (int64_t)lj * rows_L;
        atomicAdd(&L_data[offset], v);
    }
}

void scatter_P_to_blocks(
    double* D_data, double* L_data,
    const double* P_values,
    const int32_t* p_type, const int32_t* p_block,
    const int32_t* p_li, const int32_t* p_lj,
    const int32_t* p_symmetric, const int32_t* p_idx,
    const int64_t* D_offsets, const int64_t* L_offsets,
    const int32_t* block_sizes,
    int64_t D_total, int64_t L_total,
    int64_t n_scatter, int64_t nnzP, int64_t batchSize,
    cudaStream_t stream)
{
    if (n_scatter == 0) return;
    int64_t total = batchSize * n_scatter;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(scatter_P_impl, blocks, threads, 0, stream,
        D_data, L_data, P_values,
        p_type, p_block, p_li, p_lj, p_symmetric, p_idx,
        D_offsets, L_offsets, block_sizes,
        D_total, L_total, n_scatter, nnzP, batchSize);
}

// ============================================================================
// A'H^{-1}A assembly: one thread per (batch, row_of_A)
// ============================================================================

static __global__ void assemble_AHA_impl(
    double* __restrict__ D_data, double* __restrict__ L_data,
    const double* __restrict__ A_values,
    const double* __restrict__ h_inv,
    const int32_t* __restrict__ col_to_block,
    const int32_t* __restrict__ block_offsets,
    const int64_t* __restrict__ D_offsets, const int64_t* __restrict__ L_offsets,
    const int32_t* __restrict__ block_sizes,
    const int32_t* __restrict__ a_csr_row_start,
    const int32_t* __restrict__ a_csr_cols,
    const int32_t* __restrict__ a_csr_to_csc,
    int64_t n, int64_t m, int64_t nnzA,
    int64_t D_total, int64_t L_total,
    int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * m;
    if (tid >= total) return;

    int64_t b = tid / m;
    int64_t r = tid % m;

    double hr = h_inv[b * m + r];
    int32_t row_start = a_csr_row_start[r];
    int32_t row_end = a_csr_row_start[r + 1];
    if (row_start >= row_end) return;

    // For each pair of nonzeros in this row
    for (int32_t ii = row_start; ii < row_end; ++ii) {
        int32_t ci = a_csr_cols[ii];
        int32_t bi = col_to_block[ci];
        int32_t li = ci - block_offsets[bi];
        // A values in CSR order; the csr_to_csc map is used for CSC-ordered
        // access but here we access via CSR index directly since values are
        // stored in CSR order
        double ai = A_values[b * nnzA + ii];
        double ai_hr = ai * hr;

        for (int32_t jj = row_start; jj < row_end; ++jj) {
            int32_t cj = a_csr_cols[jj];
            int32_t bj = col_to_block[cj];
            int32_t lj = cj - block_offsets[bj];
            double aj = A_values[b * nnzA + jj];
            double contrib = ai_hr * aj;

            if (bi == bj) {
                int32_t d = block_sizes[bi];
                int64_t offset = b * D_total + D_offsets[bi] + li + (int64_t)lj * d;
                atomicAdd(&D_data[offset], contrib);
            } else if (bj + 1 == bi) {
                // Only store lower sub-diagonal: bi > bj, so L[bj]
                // L[bj]: shape (block_sizes[bj+1], block_sizes[bj]) = (size_bi, size_bj)
                // Entry (li, lj) in col-major
                int32_t rows_L = block_sizes[bj + 1];
                int64_t offset = b * L_total + L_offsets[bj] + li + (int64_t)lj * rows_L;
                atomicAdd(&L_data[offset], contrib);
            }
            // Skip bi + 1 == bj case — handled by the symmetric bj + 1 == bi case
        }
    }
}

void assemble_AHA_to_blocks(
    double* D_data, double* L_data,
    const double* A_values, const double* h_inv,
    const int32_t* col_to_block, const int32_t* block_offsets,
    const int64_t* D_offsets, const int64_t* L_offsets,
    const int32_t* block_sizes,
    const int32_t* a_csr_row_start, const int32_t* a_csr_cols,
    const int32_t* a_csr_to_csc,
    int64_t n, int64_t m, int64_t nnzA,
    int64_t D_total, int64_t L_total,
    int64_t batchSize, cudaStream_t stream)
{
    if (nnzA == 0) return;
    int64_t total = batchSize * m;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(assemble_AHA_impl, blocks, threads, 0, stream,
        D_data, L_data, A_values, h_inv,
        col_to_block, block_offsets,
        D_offsets, L_offsets, block_sizes,
        a_csr_row_start, a_csr_cols, a_csr_to_csc,
        n, m, nnzA, D_total, L_total, batchSize);
}

// ============================================================================
// Fused AHA+P assembly: one thread per (batch, output_element)
// No atomicAdd — each output element is computed by exactly one thread.
// Output elements are enumerated as D elements [0..D_total), then L elements
// [D_total..D_total+L_total). The offset is implicit: k for D, k-D_total for L.
//
// Precomputed on host:
//   pair_ptr[k] .. pair_ptr[k+1] = range into pair arrays
//   pair_ij[p].x/y = (csr_i, csr_j) packed, pair_row[p] = row
//   p_val_idx[k] = index into P_values for this output (-1 = no P contribution)
// ============================================================================

static __global__ void assemble_fused_impl(
    double* __restrict__ D_data,
    double* __restrict__ L_data,
    const double* __restrict__ A_values,
    const double* __restrict__ h_inv,
    const double* __restrict__ P_values,
    const int32_t* __restrict__ pair_ptr,       // [n_outputs+1]
    const int2* __restrict__ pair_ij,           // [n_pairs] packed (csr_i, csr_j)
    const int32_t* __restrict__ pair_row,       // [n_pairs]
    const int32_t* __restrict__ p_val_idx,      // [n_outputs] P index or -1
    int64_t n_outputs,
    int64_t D_total, int64_t L_total,
    int64_t nnzA, int64_t nnzP, int64_t m,
    int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * n_outputs;
    if (tid >= total) return;

    int64_t b = tid / n_outputs;
    int64_t k = tid % n_outputs;

    int32_t p_start = pair_ptr[k];
    int32_t p_end = pair_ptr[k + 1];

    double val = 0.0;
    int64_t a_base = b * nnzA;
    int64_t h_base = b * m;
    for (int32_t p = p_start; p < p_end; ++p) {
        int2 ij = pair_ij[p];
        int32_t r = pair_row[p];
        val += A_values[a_base + ij.x] * h_inv[h_base + r] * A_values[a_base + ij.y];
    }

    // Add P contribution if present
    int32_t pidx = p_val_idx[k];
    if (pidx >= 0) {
        val += P_values[b * nnzP + pidx];
    }

    // D elements are [0..D_total), L elements are [D_total..D_total+L_total)
    if (k < D_total) {
        D_data[b * D_total + k] = val;
    } else {
        L_data[b * L_total + (k - D_total)] = val;
    }
}

void assemble_fused(
    double* D_data, double* L_data,
    const double* A_values, const double* h_inv,
    const double* P_values,
    const int32_t* pair_ptr,
    const int2* pair_ij,
    const int32_t* pair_row,
    const int32_t* p_val_idx,
    int64_t n_outputs,
    int64_t D_total, int64_t L_total,
    int64_t nnzA, int64_t nnzP, int64_t m,
    int64_t batchSize,
    cudaStream_t stream)
{
    if (n_outputs == 0) return;
    int64_t total = batchSize * n_outputs;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(assemble_fused_impl, blocks, threads, 0, stream,
        D_data, L_data, A_values, h_inv, P_values,
        pair_ptr, pair_ij, pair_row, p_val_idx,
        n_outputs, D_total, L_total, nnzA, nnzP, m, batchSize);
}

// ============================================================================
// Schur RHS: rhs = rx + A' * diag(h_inv) * rz
// Uses CSC structure but values stored in CSR order via csc_to_csr map
// ============================================================================

static __global__ void form_schur_rhs_impl(
    double* __restrict__ rhs_out,
    const double* __restrict__ rx, const double* __restrict__ rz,
    const double* __restrict__ h_inv, const double* __restrict__ A_values,
    const int64_t* __restrict__ A_colptr, const int32_t* __restrict__ A_rowval,
    const int32_t* __restrict__ csc_to_csr,
    const int32_t* __restrict__ new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * n;
    if (tid >= total) return;

    int64_t b = tid / n;
    int64_t j = tid % n;  // band position
    // rx is in original column order; gather the entry at this band position.
    int64_t j_old = new_to_old ? new_to_old[j] : j;

    double sum = rx[b * n + j_old];

    for (int64_t idx = A_colptr[j]; idx < A_colptr[j + 1]; ++idx) {
        int32_t r = A_rowval[idx];
        int32_t csr_idx = csc_to_csr[idx];
        sum += A_values[b * nnzA + csr_idx] * h_inv[b * m + r] * rz[b * m + r];
    }

    rhs_out[b * n + j] = sum;
}

void form_schur_rhs_kernel(
    double* rhs_out, const double* rx, const double* rz,
    const double* h_inv, const double* A_values,
    const int64_t* A_colptr, const int32_t* A_rowval,
    const int32_t* csc_to_csr,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream)
{
    int64_t total = batchSize * n;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(form_schur_rhs_impl, blocks, threads, 0, stream,
        rhs_out, rx, rz, h_inv, A_values, A_colptr, A_rowval,
        csc_to_csr, new_to_old, n, m, nnzA, batchSize);
}

// ============================================================================
// Z recovery: z = h_inv * (A*x - rz)
// Uses CSR view, values in CSR order (direct index)
// ============================================================================

static __global__ void recover_z_impl(
    double* __restrict__ z_out, const double* __restrict__ x_sol, const double* __restrict__ rz,
    const double* __restrict__ h_inv, const double* __restrict__ A_values,
    const int32_t* __restrict__ a_csr_row_start, const int32_t* __restrict__ a_csr_cols,
    const int32_t* __restrict__ a_csr_to_csc,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * m;
    if (tid >= total) return;

    int64_t b = tid / m;
    int64_t r = tid % m;

    double ax = 0.0;
    for (int32_t idx = a_csr_row_start[r]; idx < a_csr_row_start[r + 1]; ++idx) {
        int32_t col = a_csr_cols[idx];
        // Values stored in CSR order, so direct index
        ax += A_values[b * nnzA + idx] * x_sol[b * n + col];
    }

    z_out[b * m + r] = h_inv[b * m + r] * (ax - rz[b * m + r]);
}

void recover_z_kernel(
    double* z_out, const double* x_sol, const double* rz,
    const double* h_inv, const double* A_values,
    const int32_t* a_csr_row_start, const int32_t* a_csr_cols,
    const int32_t* a_csr_to_csc,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream)
{
    int64_t total = batchSize * m;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(recover_z_impl, blocks, threads, 0, stream,
        z_out, x_sol, rz, h_inv, A_values,
        a_csr_row_start, a_csr_cols, a_csr_to_csc,
        n, m, nnzA, batchSize);
}

// ============================================================================
// Add epsilon to diagonal
// ============================================================================

static __global__ void add_eps_diagonal_impl(
    double* __restrict__ data,
    int64_t total_per_batch,
    int64_t block_offset,
    int32_t block_size,
    double eps,
    int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * block_size;
    if (tid >= total) return;

    int64_t b = tid / block_size;
    int32_t k = tid % block_size;

    // Col-major diagonal: (k,k) at offset k*(block_size+1)
    data[b * total_per_batch + block_offset + k * (block_size + 1)] += eps;
}

void add_eps_diagonal(
    double* data, int64_t total_per_batch, int64_t block_offset,
    int32_t block_size, double eps, int64_t batchSize, cudaStream_t stream)
{
    int64_t total = batchSize * block_size;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(add_eps_diagonal_impl, blocks, threads, 0, stream,
        data, total_per_batch, block_offset, block_size, eps, batchSize);
}

// ============================================================================
// Copy block data between strided buffers
// ============================================================================

static __global__ void copy_block_impl(
    double* __restrict__ dst, const double* __restrict__ src,
    int64_t block_elems,
    int64_t dst_stride, int64_t src_stride,
    int64_t dst_offset, int64_t src_offset,
    int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * block_elems;
    if (tid >= total) return;

    int64_t b = tid / block_elems;
    int64_t e = tid % block_elems;

    dst[b * dst_stride + dst_offset + e] = src[b * src_stride + src_offset + e];
}

void copy_block_data(
    double* dst, const double* src,
    int64_t block_elems,
    int64_t dst_stride, int64_t src_stride,
    int64_t dst_offset, int64_t src_offset,
    int64_t batchSize, cudaStream_t stream)
{
    int64_t total = batchSize * block_elems;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(copy_block_impl, blocks, threads, 0, stream,
        dst, src, block_elems, dst_stride, src_stride,
        dst_offset, src_offset, batchSize);
}

// ============================================================================
// Transpose block
// ============================================================================

static __global__ void transpose_block_impl(
    double* __restrict__ dst, const double* __restrict__ src,
    int64_t dst_stride, int64_t src_stride,
    int64_t dst_offset, int64_t src_offset,
    int32_t src_rows, int32_t src_cols,
    int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * (int64_t)src_rows * src_cols;
    if (tid >= total) return;

    int64_t b = tid / ((int64_t)src_rows * src_cols);
    int64_t elem = tid % ((int64_t)src_rows * src_cols);
    int32_t col = (int32_t)(elem / src_rows);
    int32_t row = (int32_t)(elem % src_rows);

    double val = src[b * src_stride + src_offset + row + (int64_t)col * src_rows];
    dst[b * dst_stride + dst_offset + col + (int64_t)row * src_cols] = val;
}

void transpose_block_data(
    double* dst, const double* src,
    int64_t dst_stride, int64_t src_stride,
    int64_t dst_offset, int64_t src_offset,
    int32_t src_rows, int32_t src_cols,
    int64_t batchSize, cudaStream_t stream)
{
    int64_t total = batchSize * (int64_t)src_rows * src_cols;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(transpose_block_impl, blocks, threads, 0, stream,
        dst, src, dst_stride, src_stride, dst_offset, src_offset,
        src_rows, src_cols, batchSize);
}

// ============================================================================
// Forward/backward substitution matvec
// ============================================================================

static __global__ void forward_sub_matvec_impl(
    double* __restrict__ y_data, const double* __restrict__ L_data, const double* __restrict__ tmp,
    int64_t y_stride, int64_t L_total,
    int64_t y_offset_i, int64_t L_offset_im1,
    int32_t si, int32_t si_prev,
    int64_t tmp_stride, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * si;
    if (tid >= total) return;

    int64_t b = tid / si;
    int32_t row = tid % si;

    double sum = 0.0;
    for (int32_t k = 0; k < si_prev; ++k) {
        // L[i-1] is (si, si_prev) col-major
        sum += L_data[b * L_total + L_offset_im1 + row + (int64_t)k * si] *
               tmp[b * tmp_stride + k];
    }

    y_data[b * y_stride + y_offset_i + row] -= sum;
}

void forward_sub_matvec(
    double* y_data, const double* L_data, const double* tmp,
    int64_t y_stride, int64_t L_total,
    int64_t y_offset_i, int64_t L_offset_im1,
    int32_t si, int32_t si_prev, int64_t tmp_stride,
    int64_t batchSize, cudaStream_t stream)
{
    int64_t total = batchSize * si;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(forward_sub_matvec_impl, blocks, threads, 0, stream,
        y_data, L_data, tmp, y_stride, L_total,
        y_offset_i, L_offset_im1, si, si_prev, tmp_stride, batchSize);
}

static __global__ void backward_sub_matvec_impl(
    double* __restrict__ y_data, const double* __restrict__ L_data, const double* __restrict__ x_next,
    int64_t y_stride, int64_t L_total, int64_t x_stride,
    int64_t y_offset_i, int64_t L_offset_i, int64_t x_offset_ip1,
    int32_t si, int32_t si_next, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * si;
    if (tid >= total) return;

    int64_t b = tid / si;
    int32_t row = tid % si;

    double sum = 0.0;
    for (int32_t k = 0; k < si_next; ++k) {
        // L[i] is (si_next, si) col-major
        // L[i]'(row, k) = L[i](k, row) = L_data[... + k + row * si_next]
        sum += L_data[b * L_total + L_offset_i + k + (int64_t)row * si_next] *
               x_next[b * x_stride + x_offset_ip1 + k];
    }

    y_data[b * y_stride + y_offset_i + row] -= sum;
}

void backward_sub_matvec(
    double* y_data, const double* L_data, const double* x_next,
    int64_t y_stride, int64_t L_total, int64_t x_stride,
    int64_t y_offset_i, int64_t L_offset_i, int64_t x_offset_ip1,
    int32_t si, int32_t si_next, int64_t batchSize, cudaStream_t stream)
{
    int64_t total = batchSize * si;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(backward_sub_matvec_impl, blocks, threads, 0, stream,
        y_data, L_data, x_next, y_stride, L_total, x_stride,
        y_offset_i, L_offset_i, x_offset_ip1, si, si_next, batchSize);
}

// ============================================================================
// Batch pointer setup
// ============================================================================

static __global__ void setup_batch_pointers_impl(
    double** ptr_array, double* __restrict__ base_data,
    const int64_t* __restrict__ offsets,
    int64_t total_per_batch, int64_t nblocks, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * nblocks;
    if (tid >= total) return;

    int64_t b = tid / nblocks;
    int64_t block = tid % nblocks;

    ptr_array[tid] = base_data + b * total_per_batch + offsets[block];
}

void setup_batch_pointers_kernel(
    double** ptr_array, double* base_data, const int64_t* offsets,
    int64_t total_per_batch, int64_t nblocks, int64_t batchSize,
    cudaStream_t stream)
{
    int64_t total = batchSize * nblocks;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(setup_batch_pointers_impl, blocks, threads, 0, stream,
        ptr_array, base_data, offsets, total_per_batch, nblocks, batchSize);
}

// ============================================================================
// Scatter A values from CSR to CSC order
// ============================================================================

static __global__ void scatter_A_csr_to_csc_impl(
    double* __restrict__ A_csc, const double* __restrict__ A_csr,
    const int32_t* __restrict__ csr_to_csc,
    int64_t nnzA, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * nnzA;
    if (tid >= total) return;

    int64_t b = tid / nnzA;
    int64_t i = tid % nnzA;

    int32_t csc_pos = csr_to_csc[i];
    A_csc[b * nnzA + csc_pos] = A_csr[b * nnzA + i];
}

void scatter_A_csr_to_csc(
    double* A_csc_values, const double* A_csr_values,
    const int32_t* csr_to_csc_map,
    int64_t nnzA, int64_t batchSize, cudaStream_t stream)
{
    if (nnzA == 0) return;
    int64_t total = batchSize * nnzA;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(scatter_A_csr_to_csc_impl, blocks, threads, 0, stream,
        A_csc_values, A_csr_values, csr_to_csc_map, nnzA, batchSize);
}

// ============================================================================
// Pack/unpack KKT RHS/solution
// ============================================================================

static __global__ void unpack_kkt_rhs_impl(
    double* __restrict__ rx, double* __restrict__ rz, const double* __restrict__ rhs,
    int64_t n, int64_t m, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (tid >= total) return;

    int64_t b = tid / N;
    int64_t i = tid % N;

    if (i < n) {
        rx[b * n + i] = rhs[b * N + i];
    } else {
        rz[b * m + (i - n)] = rhs[b * N + i];
    }
}

void unpack_kkt_rhs(
    double* rx, double* rz, const double* rhs,
    int64_t n, int64_t m, int64_t batchSize, cudaStream_t stream)
{
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(unpack_kkt_rhs_impl, blocks, threads, 0, stream,
        rx, rz, rhs, n, m, batchSize);
}

static __global__ void pack_kkt_sol_impl(
    double* __restrict__ sol, const double* __restrict__ x, const double* __restrict__ z,
    int64_t n, int64_t m, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (tid >= total) return;

    int64_t b = tid / N;
    int64_t i = tid % N;

    if (i < n) {
        sol[b * N + i] = x[b * n + i];
    } else {
        sol[b * N + i] = z[b * m + (i - n)];
    }
}

void pack_kkt_sol(
    double* sol, const double* x, const double* z,
    int64_t n, int64_t m, int64_t batchSize, cudaStream_t stream)
{
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(pack_kkt_sol_impl, blocks, threads, 0, stream,
        sol, x, z, n, m, batchSize);
}

// ============================================================================
// Negate vector
// ============================================================================

static __global__ void negate_vector_impl(
    double* __restrict__ dst, const double* __restrict__ src, int64_t total_elems)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_elems) return;
    dst[tid] = -src[tid];
}

void negate_vector(
    double* dst, const double* src,
    int64_t total_elems, cudaStream_t stream)
{
    if (total_elems == 0) return;
    int threads = 256;
    int blocks = (int)((total_elems + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(negate_vector_impl, blocks, threads, 0, stream, dst, src, total_elems);
}

// ============================================================================
// Scatter a band-ordered x-vector to original column order:
//   dst[b*n + new_to_old[i]] = src[b*n + i]
// ============================================================================

static __global__ void permute_scatter_impl(
    double* __restrict__ dst, const double* __restrict__ src,
    const int32_t* __restrict__ new_to_old, int64_t n, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * n;
    if (tid >= total) return;
    int64_t b = tid / n;
    int64_t i = tid % n;
    dst[b * n + new_to_old[i]] = src[b * n + i];
}

void permute_scatter(
    double* dst, const double* src, const int32_t* new_to_old,
    int64_t n, int64_t batchSize, cudaStream_t stream)
{
    int64_t total = batchSize * n;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(permute_scatter_impl, blocks, threads, 0, stream,
        dst, src, new_to_old, n, batchSize);
}

// ============================================================================
// Fused kernels: eliminate unpack/pack/negate kernel launches
// ============================================================================

// form_schur_rhs reading directly from interleaved [n+m] buffer, with optional negate
static __global__ void form_schur_rhs_interleaved_impl(
    double* __restrict__ rhs_out,
    const double* __restrict__ interleaved_rhs,  // [batchSize * (n+m)]
    const double* __restrict__ h_inv, const double* __restrict__ A_values,
    const int64_t* __restrict__ A_colptr, const int32_t* __restrict__ A_rowval,
    const int32_t* __restrict__ csc_to_csr,
    const int32_t* __restrict__ new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    double rx_sign)  // +1.0 or -1.0
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * n;
    if (tid >= total) return;

    int64_t b = tid / n;
    int64_t j = tid % n;  // band position
    int64_t N = n + m;
    // rx (x-part of the interleaved RHS) is in original column order.
    int64_t j_old = new_to_old ? new_to_old[j] : j;

    double sum = rx_sign * interleaved_rhs[b * N + j_old];

    for (int64_t idx = A_colptr[j]; idx < A_colptr[j + 1]; ++idx) {
        int32_t r = A_rowval[idx];
        int32_t csr_idx = csc_to_csr[idx];
        sum += A_values[b * nnzA + csr_idx] * h_inv[b * m + r] * interleaved_rhs[b * N + n + r];
    }

    rhs_out[b * n + j] = sum;
}

void form_schur_rhs_interleaved_kernel(
    double* rhs_out,
    const double* interleaved_rhs,
    const double* h_inv, const double* A_values,
    const int64_t* A_colptr, const int32_t* A_rowval,
    const int32_t* csc_to_csr,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    double rx_sign,
    cudaStream_t stream)
{
    int64_t total = batchSize * n;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(form_schur_rhs_interleaved_impl, blocks, threads, 0, stream,
        rhs_out, interleaved_rhs, h_inv, A_values, A_colptr, A_rowval,
        csc_to_csr, new_to_old, n, m, nnzA, batchSize, rx_sign);
}

// recover_z and pack x,z into interleaved [n+m] output
// rz is read from interleaved_rhs at offset n within each batch element
static __global__ void recover_z_and_pack_impl(
    double* __restrict__ interleaved_sol,  // [batchSize * (n+m)]
    const double* __restrict__ x_sol,
    const double* __restrict__ interleaved_rhs,  // [batchSize * (n+m)], rz at offset n
    const double* __restrict__ h_inv, const double* __restrict__ A_values,
    const int32_t* __restrict__ a_csr_row_start, const int32_t* __restrict__ a_csr_cols,
    const int32_t* __restrict__ new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (tid >= total) return;

    int64_t b = tid / N;
    int64_t i = tid % N;

    if (i < n) {
        // x_sol is in band order; scatter to the original column slot of output.
        int64_t i_old = new_to_old ? new_to_old[i] : i;
        interleaved_sol[b * N + i_old] = x_sol[b * n + i];
    } else {
        // Compute z and write to interleaved output
        int64_t r = i - n;
        double ax = 0.0;
        for (int32_t idx = a_csr_row_start[r]; idx < a_csr_row_start[r + 1]; ++idx) {
            int32_t col = a_csr_cols[idx];
            ax += A_values[b * nnzA + idx] * x_sol[b * n + col];
        }
        // rz is at position [b * N + n + r] in interleaved_rhs
        interleaved_sol[b * N + i] = h_inv[b * m + r] * (ax - interleaved_rhs[b * N + n + r]);
    }
}

void recover_z_and_pack_kernel(
    double* interleaved_sol,
    const double* x_sol,
    const double* interleaved_rhs,
    const double* h_inv, const double* A_values,
    const int32_t* a_csr_row_start, const int32_t* a_csr_cols,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream)
{
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(recover_z_and_pack_impl, blocks, threads, 0, stream,
        interleaved_sol, x_sol, interleaved_rhs, h_inv, A_values,
        a_csr_row_start, a_csr_cols, new_to_old, n, m, nnzA, batchSize);
}

// form_schur_rhs with negate flag (reads from separate rx, rz buffers)
static __global__ void form_schur_rhs_signed_impl(
    double* __restrict__ rhs_out,
    const double* __restrict__ rx, const double* __restrict__ rz,
    const double* __restrict__ h_inv, const double* __restrict__ A_values,
    const int64_t* __restrict__ A_colptr, const int32_t* __restrict__ A_rowval,
    const int32_t* __restrict__ csc_to_csr,
    const int32_t* __restrict__ new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    double rx_sign)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * n;
    if (tid >= total) return;

    int64_t b = tid / n;
    int64_t j = tid % n;  // band position
    int64_t j_old = new_to_old ? new_to_old[j] : j;  // rx in original column order

    double sum = rx_sign * rx[b * n + j_old];

    for (int64_t idx = A_colptr[j]; idx < A_colptr[j + 1]; ++idx) {
        int32_t r = A_rowval[idx];
        int32_t csr_idx = csc_to_csr[idx];
        sum += A_values[b * nnzA + csr_idx] * h_inv[b * m + r] * rz[b * m + r];
    }

    rhs_out[b * n + j] = sum;
}

void form_schur_rhs_signed_kernel(
    double* rhs_out,
    const double* rx, const double* rz,
    const double* h_inv, const double* A_values,
    const int64_t* A_colptr, const int32_t* A_rowval,
    const int32_t* csc_to_csr,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    double rx_sign,
    cudaStream_t stream)
{
    int64_t total = batchSize * n;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(form_schur_rhs_signed_impl, blocks, threads, 0, stream,
        rhs_out, rx, rz, h_inv, A_values, A_colptr, A_rowval,
        csc_to_csr, new_to_old, n, m, nnzA, batchSize, rx_sign);
}

// 2-RHS: both from interleaved [n+m] buffers (for solve2)
static __global__ void form_schur_rhs2_both_interleaved_impl(
    double* __restrict__ rhs_out1, double* __restrict__ rhs_out2,
    const double* __restrict__ interleaved_rhs1,  // [batchSize * (n+m)]
    const double* __restrict__ interleaved_rhs2,  // [batchSize * (n+m)]
    const double* __restrict__ h_inv, const double* __restrict__ A_values,
    const int64_t* __restrict__ A_colptr, const int32_t* __restrict__ A_rowval,
    const int32_t* __restrict__ csc_to_csr,
    const int32_t* __restrict__ new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * n;
    if (tid >= total) return;

    int64_t b = tid / n;
    int64_t j = tid % n;  // band position
    int64_t N = n + m;
    int64_t j_old = new_to_old ? new_to_old[j] : j;  // x-parts in original order

    double sum1 = interleaved_rhs1[b * N + j_old];
    double sum2 = interleaved_rhs2[b * N + j_old];

    for (int64_t idx = A_colptr[j]; idx < A_colptr[j + 1]; ++idx) {
        int32_t r = A_rowval[idx];
        int32_t csr_idx = csc_to_csr[idx];
        double a_val = A_values[b * nnzA + csr_idx];
        double h = h_inv[b * m + r];
        sum1 += a_val * h * interleaved_rhs1[b * N + n + r];
        sum2 += a_val * h * interleaved_rhs2[b * N + n + r];
    }

    rhs_out1[b * n + j] = sum1;
    rhs_out2[b * n + j] = sum2;
}

void form_schur_rhs2_both_interleaved_kernel(
    double* rhs_out1, double* rhs_out2,
    const double* interleaved_rhs1,
    const double* interleaved_rhs2,
    const double* h_inv, const double* A_values,
    const int64_t* A_colptr, const int32_t* A_rowval,
    const int32_t* csc_to_csr,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream)
{
    int64_t total = batchSize * n;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(form_schur_rhs2_both_interleaved_impl, blocks, threads, 0, stream,
        rhs_out1, rhs_out2, interleaved_rhs1, interleaved_rhs2,
        h_inv, A_values, A_colptr, A_rowval, csc_to_csr, new_to_old,
        n, m, nnzA, batchSize);
}

// 2-RHS: recover z for both, pack both into interleaved [n+m] outputs (for solve2)
static __global__ void recover_z2_and_pack2_impl(
    double* __restrict__ interleaved_sol1,  // [batchSize * (n+m)]
    double* __restrict__ interleaved_sol2,  // [batchSize * (n+m)]
    const double* __restrict__ x_sol1, const double* __restrict__ x_sol2,
    const double* __restrict__ interleaved_rhs1,  // [batchSize * (n+m)], rz1 at offset n
    const double* __restrict__ interleaved_rhs2,  // [batchSize * (n+m)], rz2 at offset n
    const double* __restrict__ h_inv, const double* __restrict__ A_values,
    const int32_t* __restrict__ a_csr_row_start, const int32_t* __restrict__ a_csr_cols,
    const int32_t* __restrict__ new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (tid >= total) return;

    int64_t b = tid / N;
    int64_t i = tid % N;

    if (i < n) {
        // x_sol* in band order; scatter to original column slot of each output.
        int64_t i_old = new_to_old ? new_to_old[i] : i;
        interleaved_sol1[b * N + i_old] = x_sol1[b * n + i];
        interleaved_sol2[b * N + i_old] = x_sol2[b * n + i];
    } else {
        int64_t r = i - n;
        double ax1 = 0.0, ax2 = 0.0;
        for (int32_t idx = a_csr_row_start[r]; idx < a_csr_row_start[r + 1]; ++idx) {
            int32_t col = a_csr_cols[idx];
            double a_val = A_values[b * nnzA + idx];
            ax1 += a_val * x_sol1[b * n + col];
            ax2 += a_val * x_sol2[b * n + col];
        }
        double h = h_inv[b * m + r];
        interleaved_sol1[b * N + i] = h * (ax1 - interleaved_rhs1[b * N + n + r]);
        interleaved_sol2[b * N + i] = h * (ax2 - interleaved_rhs2[b * N + n + r]);
    }
}

void recover_z2_and_pack2_kernel(
    double* interleaved_sol1, double* interleaved_sol2,
    const double* x_sol1, const double* x_sol2,
    const double* interleaved_rhs1, const double* interleaved_rhs2,
    const double* h_inv, const double* A_values,
    const int32_t* a_csr_row_start, const int32_t* a_csr_cols,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream)
{
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(recover_z2_and_pack2_impl, blocks, threads, 0, stream,
        interleaved_sol1, interleaved_sol2, x_sol1, x_sol2,
        interleaved_rhs1, interleaved_rhs2,
        h_inv, A_values, a_csr_row_start, a_csr_cols, new_to_old,
        n, m, nnzA, batchSize);
}

// 2-RHS: read RHS1 from interleaved, RHS2 from separate buffers (for solve_combined)
static __global__ void form_schur_rhs2_interleaved_impl(
    double* __restrict__ rhs_out1, double* __restrict__ rhs_out2,
    const double* __restrict__ interleaved_rhs1,  // [batchSize * (n+m)]
    const double* __restrict__ rx2, const double* __restrict__ rz2,
    const double* __restrict__ h_inv, const double* __restrict__ A_values,
    const int64_t* __restrict__ A_colptr, const int32_t* __restrict__ A_rowval,
    const int32_t* __restrict__ csc_to_csr,
    const int32_t* __restrict__ new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * n;
    if (tid >= total) return;

    int64_t b = tid / n;
    int64_t j = tid % n;  // band position
    int64_t N = n + m;
    int64_t j_old = new_to_old ? new_to_old[j] : j;  // both x-parts in original order

    double sum1 = interleaved_rhs1[b * N + j_old];
    double sum2 = rx2[b * n + j_old];

    for (int64_t idx = A_colptr[j]; idx < A_colptr[j + 1]; ++idx) {
        int32_t r = A_rowval[idx];
        int32_t csr_idx = csc_to_csr[idx];
        double a_val = A_values[b * nnzA + csr_idx];
        double h = h_inv[b * m + r];
        sum1 += a_val * h * interleaved_rhs1[b * N + n + r];
        sum2 += a_val * h * rz2[b * m + r];
    }

    rhs_out1[b * n + j] = sum1;
    rhs_out2[b * n + j] = sum2;
}

void form_schur_rhs2_interleaved_kernel(
    double* rhs_out1, double* rhs_out2,
    const double* interleaved_rhs1,
    const double* rx2, const double* rz2,
    const double* h_inv, const double* A_values,
    const int64_t* A_colptr, const int32_t* A_rowval,
    const int32_t* csc_to_csr,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream)
{
    int64_t total = batchSize * n;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(form_schur_rhs2_interleaved_impl, blocks, threads, 0, stream,
        rhs_out1, rhs_out2, interleaved_rhs1, rx2, rz2,
        h_inv, A_values, A_colptr, A_rowval, csc_to_csr, new_to_old,
        n, m, nnzA, batchSize);
}

// 2-RHS: recover z for both, pack RHS1 into interleaved, copy RHS2 to separate buffers
// rz1 is read from interleaved_rhs1 at offset n; rz2 from separate buffer
static __global__ void recover_z2_and_pack_impl(
    double* __restrict__ interleaved_sol1,  // [batchSize * (n+m)]
    double* __restrict__ z_out2,            // [batchSize * m]
    const double* __restrict__ x_sol1, const double* __restrict__ x_sol2,
    const double* __restrict__ interleaved_rhs1,  // [batchSize * (n+m)], rz1 at offset n
    const double* __restrict__ rz2,
    const double* __restrict__ h_inv, const double* __restrict__ A_values,
    const int32_t* __restrict__ a_csr_row_start, const int32_t* __restrict__ a_csr_cols,
    const int32_t* __restrict__ new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (tid >= total) return;

    int64_t b = tid / N;
    int64_t i = tid % N;

    if (i < n) {
        // x_sol1 in band order; scatter to original column slot of the output.
        int64_t i_old = new_to_old ? new_to_old[i] : i;
        interleaved_sol1[b * N + i_old] = x_sol1[b * n + i];
    } else {
        // Compute z for both RHS and write
        int64_t r = i - n;
        double ax1 = 0.0, ax2 = 0.0;
        for (int32_t idx = a_csr_row_start[r]; idx < a_csr_row_start[r + 1]; ++idx) {
            int32_t col = a_csr_cols[idx];
            double a_val = A_values[b * nnzA + idx];
            ax1 += a_val * x_sol1[b * n + col];
            ax2 += a_val * x_sol2[b * n + col];
        }
        double h = h_inv[b * m + r];
        // rz1 from interleaved_rhs1 at position [b * N + n + r]
        interleaved_sol1[b * N + i] = h * (ax1 - interleaved_rhs1[b * N + n + r]);
        z_out2[b * m + r] = h * (ax2 - rz2[b * m + r]);
    }
}

void recover_z2_and_pack_kernel(
    double* interleaved_sol1, double* z_out2,
    const double* x_sol1, const double* x_sol2,
    const double* interleaved_rhs1, const double* rz2,
    const double* h_inv, const double* A_values,
    const int32_t* a_csr_row_start, const int32_t* a_csr_cols,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream)
{
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(recover_z2_and_pack_impl, blocks, threads, 0, stream,
        interleaved_sol1, z_out2, x_sol1, x_sol2, interleaved_rhs1, rz2,
        h_inv, A_values, a_csr_row_start, a_csr_cols, new_to_old,
        n, m, nnzA, batchSize);
}

// ============================================================================
// Small batched Cholesky (one thread per batch element, for block_size <= 32)
// ============================================================================

static __global__ void cholesky_small_kernel(
    double* __restrict__ data, int64_t total_per_batch, int64_t block_offset,
    int32_t n, int* __restrict__ info, double eps, int64_t batchSize)
{
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    double* S = data + b * total_per_batch + block_offset;

    // In-place lower-triangular Cholesky with fused eps-regularization: S = L*L'
    // S is n x n col-major
    for (int32_t j = 0; j < n; ++j) {
        double sum = S[j + j * n] + eps;  // fused regularization
        for (int32_t k = 0; k < j; ++k) {
            double ljk = S[j + k * n];
            sum -= ljk * ljk;
        }
        if (sum <= 0.0) {
            info[b] = j + 1;
            return;
        }
        double ljj = sqrt(sum);
        S[j + j * n] = ljj;
        double inv_ljj = 1.0 / ljj;

        for (int32_t i = j + 1; i < n; ++i) {
            double s = S[i + j * n];
            for (int32_t k = 0; k < j; ++k)
                s -= S[i + k * n] * S[j + k * n];
            S[i + j * n] = s * inv_ljj;
        }
        for (int32_t i = 0; i < j; ++i)
            S[i + j * n] = 0.0;
    }

    info[b] = 0;
}

static void cholesky_small_batched(
    double* data, int64_t total_per_batch, int64_t block_offset,
    int32_t n, int* d_info, double eps, int64_t batchSize, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((batchSize + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(cholesky_small_kernel, blocks, threads, 0, stream,
        data, total_per_batch, block_offset, n, d_info, eps, batchSize);
}

// ============================================================================
// Thread-block smem kernels: one CUDA thread block per batch element
// GEMM-only Schur complement using precomputed L^{-1}, parallel GEMV solve.
// For medium blocks (8 < d <= smem limit, queried from device).
// ============================================================================


// Maximum block size for smem path, queried at runtime from device properties.
// Factorize needs (5*d² + 1) doubles of shared memory.
// Falls back to gmem path for larger blocks.
static int query_smem_max_block() {
    int device;
    cudaGetDevice(&device);
    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // Factorize is the binding constraint: (5*d² + 1) * sizeof(double) <= max_smem
    // Factorize needs 4 d×d matrices in smem (cur, prev_inv, work, w2) + 1 scalar.
    // L_blk is read directly from global memory (L2-cached).
    int d = (int)sqrt((max_smem / sizeof(double) - 1) / 4.0);
    // Clamp: don't go below 24 (warp path) or above the hard smem block limit
    if (d < 24) d = 24;
    if (d > kRiccatiMaxSmemBlock) d = kRiccatiMaxSmemBlock;
    return d;
}

int riccati_smem_max_block() {
    static int val = query_smem_max_block();
    return val;
}

static int get_smem_max_block() {
    return riccati_smem_max_block();
}
static constexpr int TB_SMEM_FACT_THREADS = 128;
static constexpr int TB_SMEM_SOLVE_THREADS = 128;

// Thread-block factorize: GEMM-only Schur complement + L^{-1} computation
// Optimizations: 128 threads, warp-parallel diagonal reduction,
// fused Cholesky + L^{-1} with pointer swaps.
// Shared memory: [s_matA: d²] [s_matB: d²] [s_matC: d²] [s_matD: d²] [scalar: 1]
// 4 matrices in smem (cur, prev_inv, work, w2). L_blk read from global (L2-cached).
// S_inv_data receives L^{-1} for each block (used by solve kernel for parallel GEMV)
static __global__ void tb_smem_factorize_kernel(
    double* __restrict__ S_inv_data, const double* __restrict__ D_data, const double* __restrict__ L_data,
    const int64_t* __restrict__ D_offsets, const int64_t* __restrict__ L_offsets,
    const int32_t* __restrict__ block_sizes,
    int64_t D_total, int64_t L_total,
    int64_t nblocks, int* __restrict__ info, int64_t batchSize,
    int32_t max_block, double reg_eps)
{
    int64_t b = blockIdx.x;
    if (b >= batchSize) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    extern __shared__ double smem[];
    int32_t md2 = max_block * max_block;
    double* s_matA    = smem;
    double* s_matB    = smem + md2;
    double* s_matC    = smem + 2 * md2;
    double* s_matD    = smem + 3 * md2;
    double* s_scalar  = smem + 4 * md2;

    double* s_cur = s_matA;
    double* s_prev_inv = s_matB;
    double* s_work = s_matC;
    // s_matD used as W2 scratch during Schur complement

    for (int64_t blk = 0; blk < nblocks; ++blk) {
        int32_t si = block_sizes[blk];
        int32_t si2 = si * si;

        // Load D[blk] into s_cur
        const double* D_blk = D_data + b * D_total + D_offsets[blk];
        for (int e = tid; e < si2; e += nthreads)
            s_cur[e] = D_blk[e];

        if (blk > 0) {
            int32_t sp = block_sizes[blk - 1];

            // L_blk read directly from global memory (L2-cached, saves d² smem)
            const double* L_blk = L_data + b * L_total + L_offsets[blk - 1];
            __syncthreads();

            // Step 1: W = L_prev^{-1} * L_sub^T  (sp×si, fully parallel)
            for (int e = tid; e < sp * si; e += nthreads) {
                int r = e % sp;
                int c = e / sp;
                double sum = 0.0;
                for (int32_t k = 0; k <= r; ++k)
                    sum += s_prev_inv[r + k * sp] * L_blk[c + k * si];
                s_work[e] = sum;
            }
            __syncthreads();

            // Step 2: W2 = L_prev^{-T} * W  (sp×si, fully parallel)
            double* s_w2 = s_matD;
            for (int e = tid; e < sp * si; e += nthreads) {
                int r = e % sp;
                int c = e / sp;
                double sum = 0.0;
                for (int32_t k = r; k < sp; ++k)
                    sum += s_prev_inv[k + r * sp] * s_work[k + c * sp];
                s_w2[e] = sum;
            }
            __syncthreads();

            // Step 3: s_cur -= L_sub * W2  (si×si, fully parallel)
            for (int e = tid; e < si2; e += nthreads) {
                int row = e % si;
                int col = e / si;
                double sum = 0.0;
                for (int32_t k = 0; k < sp; ++k)
                    sum += L_blk[row + k * si] * s_w2[k + col * sp];
                s_cur[e] -= sum;
            }
            __syncthreads();
        } else {
            __syncthreads();
        }

        // Fused Cholesky + L^{-1} with warp-parallel diagonal reduction.
        // Warp 0 reduces the diagonal sum in parallel; all threads do
        // sub-diagonal + L^{-1} work.
        for (int32_t j = 0; j < si; ++j) {
            if (warp_id == 0) {
                double partial = 0.0;
                for (int32_t k = lane; k < j; k += 32) {
                    double ljk = s_cur[j + k * si];
                    partial += ljk * ljk;
                }
                for (int offset = 16; offset > 0; offset >>= 1)
                    partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);

                if (lane == 0) {
                    double s = s_cur[j + j * si] + reg_eps - partial;
                    if (s <= 0.0) {
                        info[b] = blk * 100 + j + 1;
                        s_scalar[0] = -1.0;
                    } else {
                        double ljj = sqrt(s);
                        s_cur[j + j * si] = ljj;
                        s_scalar[0] = 1.0 / ljj;
                    }
                }
            }
            __syncthreads();
            if (s_scalar[0] < 0.0) return;
            double inv_ljj = s_scalar[0];

            // Cholesky sub-diagonal AND L^{-1} row j, interleaved
            int n_chol = si - j - 1;
            int n_inv = si;
            int total_work = n_chol + n_inv;
            for (int w = tid; w < total_work; w += nthreads) {
                if (w < n_chol) {
                    int i = j + 1 + w;
                    double s = s_cur[i + j * si];
                    for (int32_t k = 0; k < j; ++k)
                        s -= s_cur[i + k * si] * s_cur[j + k * si];
                    s_cur[i + j * si] = s * inv_ljj;
                } else {
                    int c = w - n_chol;
                    double val = (j == c) ? 1.0 : 0.0;
                    for (int32_t k = 0; k < j; ++k)
                        val -= s_cur[j + k * si] * s_work[k + c * si];
                    s_work[j + c * si] = val * inv_ljj;
                }
            }
            __syncthreads();
        }

        // Write L^{-1} to global (for solve kernel)
        double* Sinv_global = S_inv_data + b * D_total + D_offsets[blk];
        for (int e = tid; e < si2; e += nthreads)
            Sinv_global[e] = s_work[e];

        // Pointer swap: s_work has L^{-1}, becomes s_prev_inv for next block
        double* t = s_work;
        s_work = s_prev_inv;
        s_prev_inv = t;
        __syncthreads();
    }

    if (tid == 0) info[b] = 0;
}

// Thread-block solve: uses precomputed L^{-1} for parallel GEMV
// S^{-1} * y = L^{-T} * (L^{-1} * y)
// Shared memory: [Linv: d²] [tmp: d] [tmp2: d]
static __global__ void tb_smem_solve_kernel(
    double* __restrict__ lhsx, const double* __restrict__ S_inv_data, const double* __restrict__ L_data,
    const int64_t* __restrict__ D_offsets, const int64_t* __restrict__ L_offsets,
    const int32_t* __restrict__ block_offsets, const int32_t* __restrict__ block_sizes,
    int64_t D_total, int64_t L_total,
    int64_t n, int64_t nblocks, int64_t batchSize,
    int32_t max_block)
{
    int64_t b = blockIdx.x;
    if (b >= batchSize) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    double* y = lhsx + b * n;
    extern __shared__ double smem[];

    int32_t md2 = max_block * max_block;
    double* s_Linv = smem;
    double* s_tmp = smem + md2;
    double* s_tmp2 = s_tmp + max_block;

    // Forward substitution
    for (int64_t blk = 0; blk < nblocks; ++blk) {
        int32_t si = block_sizes[blk];
        if (blk > 0) {
            int32_t sp = block_sizes[blk - 1];
            int32_t bo_prev = block_offsets[blk - 1];
            int32_t bo_cur = block_offsets[blk];

            // Load L^{-1} and y
            int32_t sp2 = sp * sp;
            for (int e = tid; e < sp2; e += nthreads)
                s_Linv[e] = S_inv_data[b * D_total + D_offsets[blk - 1] + e];
            for (int k = tid; k < sp; k += nthreads)
                s_tmp[k] = y[bo_prev + k];
            __syncthreads();

            // z = L^{-1} * y (lower triangular GEMV)
            for (int r = tid; r < sp; r += nthreads) {
                double val = 0.0;
                for (int32_t k = 0; k <= r; ++k)
                    val += s_Linv[r + k * sp] * s_tmp[k];
                s_tmp2[r] = val;
            }
            __syncthreads();

            // result = L^{-T} * z (upper triangular GEMV)
            for (int r = tid; r < sp; r += nthreads) {
                double val = 0.0;
                for (int32_t k = r; k < sp; ++k)
                    val += s_Linv[k + r * sp] * s_tmp2[k];
                s_tmp[r] = val;
            }
            __syncthreads();

            // y[blk] -= L * (S^{-1} * y[blk-1])
            const double* L_block = L_data + b * L_total + L_offsets[blk - 1];
            for (int r = tid; r < si; r += nthreads) {
                double sum = 0.0;
                for (int32_t k = 0; k < sp; ++k)
                    sum += L_block[r + k * si] * s_tmp[k];
                y[bo_cur + r] -= sum;
            }
            __syncthreads();
        }
    }

    // Backward substitution
    for (int64_t blk = nblocks - 1; blk >= 0; --blk) {
        int32_t si = block_sizes[blk];
        int32_t bo = block_offsets[blk];

        if (blk < nblocks - 1) {
            int32_t sn = block_sizes[blk + 1];
            int32_t bo_next = block_offsets[blk + 1];
            const double* L_block = L_data + b * L_total + L_offsets[blk];

            for (int k = tid; k < sn; k += nthreads)
                s_tmp[k] = y[bo_next + k];
            __syncthreads();

            for (int r = tid; r < si; r += nthreads) {
                double sum = 0.0;
                for (int32_t k = 0; k < sn; ++k)
                    sum += L_block[k + r * sn] * s_tmp[k];
                y[bo + r] -= sum;
            }
            __syncthreads();
        }

        // Apply S^{-1} = L^{-T} * L^{-1}
        int32_t si2 = si * si;
        for (int e = tid; e < si2; e += nthreads)
            s_Linv[e] = S_inv_data[b * D_total + D_offsets[blk] + e];
        for (int k = tid; k < si; k += nthreads)
            s_tmp[k] = y[bo + k];
        __syncthreads();

        for (int r = tid; r < si; r += nthreads) {
            double val = 0.0;
            for (int32_t k = 0; k <= r; ++k)
                val += s_Linv[r + k * si] * s_tmp[k];
            s_tmp2[r] = val;
        }
        __syncthreads();

        for (int r = tid; r < si; r += nthreads) {
            double val = 0.0;
            for (int32_t k = r; k < si; ++k)
                val += s_Linv[k + r * si] * s_tmp2[k];
            y[bo + r] = val;
        }
        __syncthreads();
    }
}

// ============================================================================
// Multi-RHS solve kernel: 2 RHS with shared L^{-1} loading
// Shared memory: [Linv: d²] [tmp1: d] [tmp2: d] [tmp2_1: d] [tmp2_2: d]
// ============================================================================

static __global__ void tb_smem_solve2_kernel(
    double* __restrict__ lhsx1, double* __restrict__ lhsx2,
    const double* __restrict__ S_inv_data, const double* __restrict__ L_data,
    const int64_t* __restrict__ D_offsets, const int64_t* __restrict__ L_offsets,
    const int32_t* __restrict__ block_offsets, const int32_t* __restrict__ block_sizes,
    int64_t D_total, int64_t L_total,
    int64_t n, int64_t nblocks, int64_t batchSize,
    int32_t max_block)
{
    int64_t b = blockIdx.x;
    if (b >= batchSize) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    double* y1 = lhsx1 + b * n;
    double* y2 = lhsx2 + b * n;
    extern __shared__ double smem[];

    int32_t md2 = max_block * max_block;
    double* s_Linv = smem;
    double* s_tmp1 = smem + md2;
    double* s_tmp2 = s_tmp1 + max_block;
    double* s_tmp2_1 = s_tmp2 + max_block;
    double* s_tmp2_2 = s_tmp2_1 + max_block;

    // Forward substitution
    for (int64_t blk = 0; blk < nblocks; ++blk) {
        int32_t si = block_sizes[blk];
        if (blk > 0) {
            int32_t sp = block_sizes[blk - 1];
            int32_t bo_prev = block_offsets[blk - 1];
            int32_t bo_cur = block_offsets[blk];

            // Load L^{-1} and both y vectors
            int32_t sp2 = sp * sp;
            for (int e = tid; e < sp2; e += nthreads)
                s_Linv[e] = S_inv_data[b * D_total + D_offsets[blk - 1] + e];
            for (int k = tid; k < sp; k += nthreads) {
                s_tmp1[k] = y1[bo_prev + k];
                s_tmp2[k] = y2[bo_prev + k];
            }
            __syncthreads();

            // z = L^{-1} * y (both RHS, lower triangular GEMV)
            for (int r = tid; r < sp; r += nthreads) {
                double val1 = 0.0, val2 = 0.0;
                for (int32_t k = 0; k <= r; ++k) {
                    double linv = s_Linv[r + k * sp];
                    val1 += linv * s_tmp1[k];
                    val2 += linv * s_tmp2[k];
                }
                s_tmp2_1[r] = val1;
                s_tmp2_2[r] = val2;
            }
            __syncthreads();

            // result = L^{-T} * z (both RHS, upper triangular GEMV)
            for (int r = tid; r < sp; r += nthreads) {
                double val1 = 0.0, val2 = 0.0;
                for (int32_t k = r; k < sp; ++k) {
                    double linv = s_Linv[k + r * sp];
                    val1 += linv * s_tmp2_1[k];
                    val2 += linv * s_tmp2_2[k];
                }
                s_tmp1[r] = val1;
                s_tmp2[r] = val2;
            }
            __syncthreads();

            // y[blk] -= L * (S^{-1} * y[blk-1])
            const double* L_block = L_data + b * L_total + L_offsets[blk - 1];
            for (int r = tid; r < si; r += nthreads) {
                double sum1 = 0.0, sum2 = 0.0;
                for (int32_t k = 0; k < sp; ++k) {
                    double l_val = L_block[r + k * si];
                    sum1 += l_val * s_tmp1[k];
                    sum2 += l_val * s_tmp2[k];
                }
                y1[bo_cur + r] -= sum1;
                y2[bo_cur + r] -= sum2;
            }
            __syncthreads();
        }
    }

    // Backward substitution
    for (int64_t blk = nblocks - 1; blk >= 0; --blk) {
        int32_t si = block_sizes[blk];
        int32_t bo = block_offsets[blk];

        if (blk < nblocks - 1) {
            int32_t sn = block_sizes[blk + 1];
            int32_t bo_next = block_offsets[blk + 1];
            const double* L_block = L_data + b * L_total + L_offsets[blk];

            for (int k = tid; k < sn; k += nthreads) {
                s_tmp1[k] = y1[bo_next + k];
                s_tmp2[k] = y2[bo_next + k];
            }
            __syncthreads();

            for (int r = tid; r < si; r += nthreads) {
                double sum1 = 0.0, sum2 = 0.0;
                for (int32_t k = 0; k < sn; ++k) {
                    double l_val = L_block[k + r * sn];
                    sum1 += l_val * s_tmp1[k];
                    sum2 += l_val * s_tmp2[k];
                }
                y1[bo + r] -= sum1;
                y2[bo + r] -= sum2;
            }
            __syncthreads();
        }

        // Apply S^{-1} = L^{-T} * L^{-1} — load L^{-1} once, solve both RHS
        int32_t si2 = si * si;
        for (int e = tid; e < si2; e += nthreads)
            s_Linv[e] = S_inv_data[b * D_total + D_offsets[blk] + e];
        for (int k = tid; k < si; k += nthreads) {
            s_tmp1[k] = y1[bo + k];
            s_tmp2[k] = y2[bo + k];
        }
        __syncthreads();

        for (int r = tid; r < si; r += nthreads) {
            double val1 = 0.0, val2 = 0.0;
            for (int32_t k = 0; k <= r; ++k) {
                double linv = s_Linv[r + k * si];
                val1 += linv * s_tmp1[k];
                val2 += linv * s_tmp2[k];
            }
            s_tmp2_1[r] = val1;
            s_tmp2_2[r] = val2;
        }
        __syncthreads();

        for (int r = tid; r < si; r += nthreads) {
            double val1 = 0.0, val2 = 0.0;
            for (int32_t k = r; k < si; ++k) {
                double linv = s_Linv[k + r * si];
                val1 += linv * s_tmp2_1[k];
                val2 += linv * s_tmp2_2[k];
            }
            y1[bo + r] = val1;
            y2[bo + r] = val2;
        }
        __syncthreads();
    }
}

// Multi-RHS form_schur_rhs: compute 2 Schur RHS vectors in one kernel
static __global__ void form_schur_rhs2_impl(
    double* __restrict__ rhs_out1, double* __restrict__ rhs_out2,
    const double* __restrict__ rx1, const double* __restrict__ rz1,
    const double* __restrict__ rx2, const double* __restrict__ rz2,
    const double* __restrict__ h_inv, const double* __restrict__ A_values,
    const int64_t* __restrict__ A_colptr, const int32_t* __restrict__ A_rowval,
    const int32_t* __restrict__ csc_to_csr,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * n;
    if (tid >= total) return;

    int64_t b = tid / n;
    int64_t j = tid % n;

    double sum1 = rx1[b * n + j];
    double sum2 = rx2[b * n + j];

    for (int64_t idx = A_colptr[j]; idx < A_colptr[j + 1]; ++idx) {
        int32_t r = A_rowval[idx];
        int32_t csr_idx = csc_to_csr[idx];
        double a_val = A_values[b * nnzA + csr_idx];
        double h = h_inv[b * m + r];
        sum1 += a_val * h * rz1[b * m + r];
        sum2 += a_val * h * rz2[b * m + r];
    }

    rhs_out1[b * n + j] = sum1;
    rhs_out2[b * n + j] = sum2;
}

void form_schur_rhs2_kernel(
    double* rhs_out1, double* rhs_out2,
    const double* rx1, const double* rz1,
    const double* rx2, const double* rz2,
    const double* h_inv, const double* A_values,
    const int64_t* A_colptr, const int32_t* A_rowval,
    const int32_t* csc_to_csr,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream)
{
    int64_t total = batchSize * n;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(form_schur_rhs2_impl, blocks, threads, 0, stream,
        rhs_out1, rhs_out2, rx1, rz1, rx2, rz2,
        h_inv, A_values, A_colptr, A_rowval, csc_to_csr,
        n, m, nnzA, batchSize);
}

// Multi-RHS Z recovery: compute 2 z vectors in one kernel
static __global__ void recover_z2_impl(
    double* __restrict__ z_out1, double* __restrict__ z_out2,
    const double* __restrict__ x_sol1, const double* __restrict__ x_sol2,
    const double* __restrict__ rz1, const double* __restrict__ rz2,
    const double* __restrict__ h_inv, const double* __restrict__ A_values,
    const int32_t* __restrict__ a_csr_row_start, const int32_t* __restrict__ a_csr_cols,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * m;
    if (tid >= total) return;

    int64_t b = tid / m;
    int64_t r = tid % m;

    double ax1 = 0.0, ax2 = 0.0;
    for (int32_t idx = a_csr_row_start[r]; idx < a_csr_row_start[r + 1]; ++idx) {
        int32_t col = a_csr_cols[idx];
        double a_val = A_values[b * nnzA + idx];
        ax1 += a_val * x_sol1[b * n + col];
        ax2 += a_val * x_sol2[b * n + col];
    }

    double h = h_inv[b * m + r];
    z_out1[b * m + r] = h * (ax1 - rz1[b * m + r]);
    z_out2[b * m + r] = h * (ax2 - rz2[b * m + r]);
}

void recover_z2_kernel(
    double* z_out1, double* z_out2,
    const double* x_sol1, const double* x_sol2,
    const double* rz1, const double* rz2,
    const double* h_inv, const double* A_values,
    const int32_t* a_csr_row_start, const int32_t* a_csr_cols,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream)
{
    int64_t total = batchSize * m;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(recover_z2_impl, blocks, threads, 0, stream,
        z_out1, z_out2, x_sol1, x_sol2, rz1, rz2,
        h_inv, A_values, a_csr_row_start, a_csr_cols,
        n, m, nnzA, batchSize);
}

// Multi-RHS unpack: unpack 2 interleaved [n+m] RHS into 4 separate vectors
static __global__ void unpack_kkt_rhs2_impl(
    double* __restrict__ rx1, double* __restrict__ rz1, double* __restrict__ rx2, double* __restrict__ rz2,
    const double* __restrict__ rhs1, const double* __restrict__ rhs2,
    int64_t n, int64_t m, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (tid >= total) return;

    int64_t b = tid / N;
    int64_t i = tid % N;

    if (i < n) {
        rx1[b * n + i] = rhs1[b * N + i];
        rx2[b * n + i] = rhs2[b * N + i];
    } else {
        int64_t j = i - n;
        rz1[b * m + j] = rhs1[b * N + i];
        rz2[b * m + j] = rhs2[b * N + i];
    }
}

void unpack_kkt_rhs2(
    double* rx1, double* rz1, double* rx2, double* rz2,
    const double* rhs1, const double* rhs2,
    int64_t n, int64_t m, int64_t batchSize, cudaStream_t stream)
{
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(unpack_kkt_rhs2_impl, blocks, threads, 0, stream,
        rx1, rz1, rx2, rz2, rhs1, rhs2, n, m, batchSize);
}

// Multi-RHS pack: pack 4 separate vectors into 2 interleaved [n+m] solutions
static __global__ void pack_kkt_sol2_impl(
    double* __restrict__ sol1, double* __restrict__ sol2,
    const double* __restrict__ x1, const double* __restrict__ z1,
    const double* __restrict__ x2, const double* __restrict__ z2,
    int64_t n, int64_t m, int64_t batchSize)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (tid >= total) return;

    int64_t b = tid / N;
    int64_t i = tid % N;

    if (i < n) {
        sol1[b * N + i] = x1[b * n + i];
        sol2[b * N + i] = x2[b * n + i];
    } else {
        int64_t j = i - n;
        sol1[b * N + i] = z1[b * m + j];
        sol2[b * N + i] = z2[b * m + j];
    }
}

void pack_kkt_sol2(
    double* sol1, double* sol2,
    const double* x1, const double* z1,
    const double* x2, const double* z2,
    int64_t n, int64_t m, int64_t batchSize, cudaStream_t stream)
{
    int64_t N = n + m;
    int64_t total = batchSize * N;
    if (total == 0) return;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(pack_kkt_sol2_impl, blocks, threads, 0, stream,
        sol1, sol2, x1, z1, x2, z2, n, m, batchSize);
}


// ============================================================================
// Warp-cooperative fused kernels: one warp (32 threads) per batch element
// Handles blocks up to WARP_MAX_BLOCK using shared memory.
// ============================================================================

static constexpr int WARP_MAX_BLOCK = 24;
static constexpr int WARP_SIZE = 32;
// Shared memory per warp for factorize: 1 work matrix (d_prev * d)
// Max size: WARP_MAX_BLOCK^2 = 576 doubles = 4608 bytes per warp
static constexpr int WARP_FACT_SMEM_DOUBLES = WARP_MAX_BLOCK * WARP_MAX_BLOCK;

// Warp-cooperative factorize: one warp per batch element
static __global__ void warp_factorize_kernel(
    double* __restrict__ S_data, const double* __restrict__ D_data, const double* __restrict__ L_data,
    const int64_t* __restrict__ D_offsets, const int64_t* __restrict__ L_offsets,
    const int32_t* __restrict__ block_sizes,
    int64_t D_total, int64_t L_total,
    int64_t nblocks, int* __restrict__ info, int64_t batchSize,
    double reg_eps)
{
    // One warp per batch element
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int local_warp = threadIdx.x / WARP_SIZE;

    if (warp_id >= batchSize) return;
    int64_t b = warp_id;

    // Shared memory: each warp gets its own work matrix
    extern __shared__ double smem[];
    double* work = smem + local_warp * WARP_FACT_SMEM_DOUBLES;

    // Copy D -> S (parallel across warp)
    for (int64_t e = lane; e < D_total; e += WARP_SIZE)
        S_data[b * D_total + e] = D_data[b * D_total + e];
    __syncwarp();

    for (int64_t blk = 0; blk < nblocks; ++blk) {
        int32_t si = block_sizes[blk];
        double* Si = S_data + b * D_total + D_offsets[blk];

        if (blk > 0) {
            int32_t si_prev = block_sizes[blk - 1];
            const double* Sprev = S_data + b * D_total + D_offsets[blk - 1];
            const double* Li = L_data + b * L_total + L_offsets[blk - 1];
            int32_t work_elems = si_prev * si;
            int32_t Si_elems = si * si;

            // Load L' into work (transpose): work[r + c*si_prev] = Li[c + r*si]
            for (int e = lane; e < work_elems; e += WARP_SIZE) {
                int32_t r = e % si_prev;
                int32_t c = e / si_prev;
                work[e] = Li[c + r * si];
            }
            __syncwarp();

            // Forward triangular solve: Sprev * tmp = work (column by column)
            // Sprev is lower triangular, stored col-major in global mem
            for (int32_t col = 0; col < si; ++col) {
                for (int32_t r = 0; r < si_prev; ++r) {
                    // Compute dot product: sum_k Sprev[r,k] * work[k, col] for k < r
                    double val = 0.0;
                    for (int32_t k = lane; k < r; k += WARP_SIZE)
                        val += Sprev[r + k * si_prev] * work[k + col * si_prev];
                    // Warp reduce
                    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
                        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
                    if (lane == 0)
                        work[r + col * si_prev] = (work[r + col * si_prev] - val) / Sprev[r + r * si_prev];
                    __syncwarp();  // Ensure lane 0's write is visible to all lanes
                }
            }

            // Backward triangular solve: Sprev' * tmp2 = tmp
            for (int32_t col = 0; col < si; ++col) {
                for (int32_t r = si_prev - 1; r >= 0; --r) {
                    double val = 0.0;
                    for (int32_t k = r + 1 + (int32_t)lane; k < si_prev; k += WARP_SIZE)
                        val += Sprev[k + r * si_prev] * work[k + col * si_prev];
                    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
                        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
                    if (lane == 0)
                        work[r + col * si_prev] = (work[r + col * si_prev] - val) / Sprev[r + r * si_prev];
                    __syncwarp();  // Ensure lane 0's write is visible
                }
            }

            // S[blk] -= L[blk-1] * work  (GEMM: si × si -= si × si_prev * si_prev × si)
            // Each thread computes a subset of output elements
            for (int e = lane; e < Si_elems; e += WARP_SIZE) {
                int32_t r = e % si;
                int32_t c = e / si;
                double sum = 0.0;
                for (int32_t k = 0; k < si_prev; ++k)
                    sum += Li[r + k * si] * work[k + c * si_prev];
                Si[r + c * si] -= sum;
            }
            __syncwarp();
        }

        // Regularize (parallel)
        for (int32_t k = lane; k < si; k += WARP_SIZE)
            Si[k + k * si] += reg_eps;
        __syncwarp();

        // In-place Cholesky — inherently sequential per column, but row updates are parallel
        for (int32_t j = 0; j < si; ++j) {
            // Compute diagonal: S[j,j] -= sum_k S[j,k]^2
            double diag_sum = 0.0;
            for (int32_t k = lane; k < j; k += WARP_SIZE) {
                double ljk = Si[j + k * si];
                diag_sum += ljk * ljk;
            }
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
                diag_sum += __shfl_down_sync(0xFFFFFFFF, diag_sum, offset);

            double ljj = 0.0;
            if (lane == 0) {
                double s = Si[j + j * si] - diag_sum;
                if (s <= 0.0) {
                    info[b] = (int)(blk * 100 + j + 1);
                    // Signal failure to all lanes
                    ljj = -1.0;
                } else {
                    ljj = sqrt(s);
                    Si[j + j * si] = ljj;
                }
            }
            ljj = __shfl_sync(0xFFFFFFFF, ljj, 0);
            if (ljj <= 0.0) return;
            double inv_ljj = 1.0 / ljj;

            // Update column: S[i,j] = (S[i,j] - sum_k S[i,k]*S[j,k]) / ljj
            for (int32_t i = j + 1 + (int32_t)lane; i < si; i += WARP_SIZE) {
                double s = Si[i + j * si];
                for (int32_t k = 0; k < j; ++k)
                    s -= Si[i + k * si] * Si[j + k * si];
                Si[i + j * si] = s * inv_ljj;
            }
            // Zero upper triangle
            for (int32_t i = lane; i < j; i += WARP_SIZE)
                Si[i + j * si] = 0.0;
            __syncwarp();
        }
    }

    if (lane == 0) info[b] = 0;
}

// Warp-cooperative solve: one warp per batch element
static __global__ void warp_solve_kernel(
    double* __restrict__ lhsx, const double* __restrict__ S_data, const double* __restrict__ L_data,
    const int64_t* __restrict__ D_offsets, const int64_t* __restrict__ L_offsets,
    const int32_t* __restrict__ block_offsets, const int32_t* __restrict__ block_sizes,
    int64_t D_total, int64_t L_total,
    int64_t n, int64_t nblocks, int64_t batchSize)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    int local_warp = threadIdx.x / WARP_SIZE;

    if (warp_id >= batchSize) return;
    int64_t b = warp_id;

    extern __shared__ double smem[];
    double* work = smem + local_warp * WARP_MAX_BLOCK;

    double* y = lhsx + b * n;

    // Forward substitution
    for (int64_t blk = 0; blk < nblocks; ++blk) {
        int32_t si = block_sizes[blk];

        if (blk > 0) {
            int32_t si_prev = block_sizes[blk - 1];
            const double* Sprev = S_data + b * D_total + D_offsets[blk - 1];
            const double* Li = L_data + b * L_total + L_offsets[blk - 1];

            // Load y[blk-1] into work
            for (int32_t k = lane; k < si_prev; k += WARP_SIZE)
                work[k] = y[block_offsets[blk - 1] + k];
            __syncwarp();

            // Forward solve: Sprev * tmp = work
            for (int32_t r = 0; r < si_prev; ++r) {
                double val = 0.0;
                for (int32_t k = lane; k < r; k += WARP_SIZE)
                    val += Sprev[r + k * si_prev] * work[k];
                for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
                    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
                if (lane == 0)
                    work[r] = (work[r] - val) / Sprev[r + r * si_prev];
                __syncwarp();
            }

            // Backward solve: Sprev' * tmp2 = tmp
            for (int32_t r = si_prev - 1; r >= 0; --r) {
                double val = 0.0;
                for (int32_t k = r + 1 + (int32_t)lane; k < si_prev; k += WARP_SIZE)
                    val += Sprev[k + r * si_prev] * work[k];
                for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
                    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
                if (lane == 0)
                    work[r] = (work[r] - val) / Sprev[r + r * si_prev];
                __syncwarp();
            }

            // y[blk] -= L[blk-1] * work
            for (int32_t r = lane; r < si; r += WARP_SIZE) {
                double sum = 0.0;
                for (int32_t k = 0; k < si_prev; ++k)
                    sum += Li[r + k * si] * work[k];
                y[block_offsets[blk] + r] -= sum;
            }
            __syncwarp();
        }
    }

    // Backward substitution
    for (int64_t blk = nblocks - 1; blk >= 0; --blk) {
        int32_t si = block_sizes[blk];

        if (blk < nblocks - 1) {
            int32_t si_next = block_sizes[blk + 1];
            const double* Li = L_data + b * L_total + L_offsets[blk];

            // y[blk] -= L[blk]' * x[blk+1]
            for (int32_t r = lane; r < si; r += WARP_SIZE) {
                double sum = 0.0;
                for (int32_t k = 0; k < si_next; ++k)
                    sum += Li[k + r * si_next] * y[block_offsets[blk + 1] + k];
                y[block_offsets[blk] + r] -= sum;
            }
            __syncwarp();
        }

        // x[blk] = S[blk]^{-1} * y[blk]
        const double* Si = S_data + b * D_total + D_offsets[blk];
        for (int32_t k = lane; k < si; k += WARP_SIZE)
            work[k] = y[block_offsets[blk] + k];
        __syncwarp();

        // Forward solve
        for (int32_t r = 0; r < si; ++r) {
            double val = 0.0;
            for (int32_t k = lane; k < r; k += WARP_SIZE)
                val += Si[r + k * si] * work[k];
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            if (lane == 0)
                work[r] = (work[r] - val) / Si[r + r * si];
            __syncwarp();
        }

        // Backward solve
        for (int32_t r = si - 1; r >= 0; --r) {
            double val = 0.0;
            for (int32_t k = r + 1 + (int32_t)lane; k < si; k += WARP_SIZE)
                val += Si[k + r * si] * work[k];
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1)
                val += __shfl_down_sync(0xFFFFFFFF, val, offset);
            if (lane == 0)
                work[r] = (work[r] - val) / Si[r + r * si];
            __syncwarp();
        }

        // Write back
        for (int32_t k = lane; k < si; k += WARP_SIZE)
            y[block_offsets[blk] + k] = work[k];
        __syncwarp();
    }
}

// ============================================================================
// Block-tridiagonal Cholesky factorization (host-side loop, batched GPU ops)
// ============================================================================

// GPU kernel to setup pointer arrays — avoids host→device memcpy + sync
static __global__ void setup_ptrs_kernel(
    double** ptrs, double* __restrict__ base, int64_t stride, int64_t offset,
    int64_t batchSize)
{
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;
    ptrs[b] = base + b * stride + offset;
}

static void setup_ptrs(double** d_ptrs, double* base, int64_t stride,
                       int64_t offset, int64_t batchSize, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((batchSize + threads - 1) / threads);
    MOREAU_KERNEL_LAUNCH(setup_ptrs_kernel, blocks, threads, 0, stream,
        d_ptrs, base, stride, offset, batchSize);
}

bool riccati_factorize_blocks(
    double* S_data, const double* D_data, double* L_data,
    double* work_data, double** d_ptr_A, double** d_ptr_B, double** d_ptr_C,
    const int64_t* h_D_offsets, const int64_t* h_L_offsets,
    const int32_t* h_block_sizes,
    const int64_t* d_D_offsets, const int64_t* d_L_offsets,
    const int32_t* d_block_sizes,
    int64_t D_total_elems, int64_t L_total_elems,
    int64_t nblocks, int32_t max_block,
    int64_t batchSize,
    cublasHandle_t cublas_handle,
    cusolverDnHandle_t cusolver_handle,
    int* d_info,
    cudaStream_t stream,
    double** d_S_block_ptrs,
    double** d_L_block_ptrs,
    double** d_work_block_ptrs,
    double reg_eps)
{
    // tb_smem: one threadblock per batch element, parallelizes within each problem.
    // smem path: uses shared memory for all work matrices.
    // Max block size determined by device shared memory limit.
    int smem_max = get_smem_max_block();
    if (max_block <= smem_max && batchSize >= 1) {
        size_t smem_bytes = (4 * (size_t)max_block * max_block + 1) * sizeof(double);
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(tb_smem_factorize_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        }
        MOREAU_KERNEL_LAUNCH(tb_smem_factorize_kernel, (int)batchSize, TB_SMEM_FACT_THREADS, smem_bytes, stream,
            S_data, D_data, L_data, d_D_offsets, d_L_offsets, d_block_sizes,
            D_total_elems, L_total_elems, nblocks, d_info, batchSize, max_block, reg_eps);
        return true;
    }

    // Blocks exceed shared memory capacity — should not reach here.
    // Auto-dispatch sends large blocks to cuDSS; explicit Riccati rejects at construction.
    return false;
}

// ============================================================================
// Block-tridiagonal solve
// ============================================================================

void riccati_solve_blocks(
    double* lhsx, const double* S_data, const double* L_data,
    double* work_data, double** d_ptr_A, double** d_ptr_B,
    const int64_t* h_D_offsets, const int64_t* h_L_offsets,
    const int32_t* h_block_offsets, const int32_t* h_block_sizes,
    const int64_t* d_D_offsets, const int64_t* d_L_offsets,
    const int32_t* d_block_offsets, const int32_t* d_block_sizes,
    int64_t D_total_elems, int64_t L_total_elems,
    int64_t n, int64_t nblocks, int32_t max_block,
    int64_t batchSize,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    double** d_S_block_ptrs,
    double** d_work_block_ptrs)
{
    if (max_block <= get_smem_max_block() && batchSize >= 1) {
        size_t smem_bytes = ((size_t)max_block * max_block + 2 * (size_t)max_block) * sizeof(double);
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(tb_smem_solve_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        }
        MOREAU_KERNEL_LAUNCH(tb_smem_solve_kernel, (int)batchSize, TB_SMEM_SOLVE_THREADS, smem_bytes, stream,
            lhsx, S_data, L_data, d_D_offsets, d_L_offsets,
            d_block_offsets, d_block_sizes,
            D_total_elems, L_total_elems, n, nblocks, batchSize, max_block);
        return;
    }

    // Should not reach here — auto-dispatch sends large blocks to cuDSS.
}

// ============================================================================
// Multi-RHS block-tridiagonal solve (2 RHS simultaneously)
// ============================================================================

void riccati_solve2_blocks(
    double* lhsx1, double* lhsx2,
    const double* S_data, const double* L_data,
    double* work_data, double** d_ptr_A, double** d_ptr_B,
    const int64_t* h_D_offsets, const int64_t* h_L_offsets,
    const int32_t* h_block_offsets, const int32_t* h_block_sizes,
    const int64_t* d_D_offsets, const int64_t* d_L_offsets,
    const int32_t* d_block_offsets, const int32_t* d_block_sizes,
    int64_t D_total_elems, int64_t L_total_elems,
    int64_t n, int64_t nblocks, int32_t max_block,
    int64_t batchSize,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    double** d_S_block_ptrs,
    double** d_work_block_ptrs)
{
    if (max_block <= get_smem_max_block() && batchSize >= 1) {
        size_t smem_bytes = ((size_t)max_block * max_block + 4 * (size_t)max_block) * sizeof(double);
        if (smem_bytes > 48 * 1024) {
            cudaFuncSetAttribute(tb_smem_solve2_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
        }
        MOREAU_KERNEL_LAUNCH(tb_smem_solve2_kernel, (int)batchSize, TB_SMEM_SOLVE_THREADS, smem_bytes, stream,
            lhsx1, lhsx2, S_data, L_data, d_D_offsets, d_L_offsets,
            d_block_offsets, d_block_sizes,
            D_total_elems, L_total_elems, n, nblocks, batchSize, max_block);
        return;
    }

    // Should not reach here — auto-dispatch sends large blocks to cuDSS.
}

} // namespace moreau

