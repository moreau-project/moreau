/**
 * @file riccati_kernels.cuh
 * @brief CUDA kernel declarations and host-callable wrappers for batched Riccati KKT solver
 *
 * All functions declared here are implemented in riccati_kernels.cu and can be called
 * from both .cpp and .cu files. No __global__ kernels are exposed — only host wrappers.
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdint>

namespace moreau {

// ============================================================================
// Device capability queries
// ============================================================================

// Hard upper bound on the Riccati smem block size, independent of device.
// riccati_smem_max_block() never exceeds this; host-side structure detection
// uses it (without touching the device) to bound the admissible bandwidth.
constexpr int kRiccatiMaxSmemBlock = 128;

// Returns the maximum Riccati block size that fits in shared memory on the
// current device. Blocks larger than this use the global-memory fallback path.
// Cached after first call.
int riccati_smem_max_block();

// ============================================================================
// Kernel wrapper functions (callable from .cpp files)
// ============================================================================

void compute_h_inv_kernel(
    double* h_inv,
    const double* nonneg_w,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t m,
    int64_t batchSize,
    double reg_eps,
    cudaStream_t stream);

void zero_blocks_kernel(
    double* D_data,
    double* L_data,
    int64_t D_total_elems,
    int64_t L_total_elems,
    int64_t batchSize,
    cudaStream_t stream);

void scatter_P_to_blocks(
    double* D_data,
    double* L_data,
    const double* P_values,
    const int32_t* p_type,
    const int32_t* p_block,
    const int32_t* p_li,
    const int32_t* p_lj,
    const int32_t* p_symmetric,
    const int32_t* p_idx,
    const int64_t* D_offsets,
    const int64_t* L_offsets,
    const int32_t* block_sizes,
    int64_t D_total,
    int64_t L_total,
    int64_t n_scatter,
    int64_t nnzP,
    int64_t batchSize,
    cudaStream_t stream);

void assemble_AHA_to_blocks(
    double* D_data,
    double* L_data,
    const double* A_values,
    const double* h_inv,
    const int32_t* col_to_block,
    const int32_t* block_offsets,
    const int64_t* D_offsets,
    const int64_t* L_offsets,
    const int32_t* block_sizes,
    const int32_t* a_csr_row_start,
    const int32_t* a_csr_cols,
    const int32_t* a_csr_to_csc,
    int64_t n, int64_t m,
    int64_t nnzA,
    int64_t D_total, int64_t L_total,
    int64_t batchSize,
    cudaStream_t stream);

void assemble_AHA_scatter(
    double* D_data, double* L_data,
    const double* A_values, const double* h_inv,
    const int64_t* out_offset,
    const int32_t* out_is_L,
    const int32_t* pair_ptr,
    const int32_t* pair_csr_i,
    const int32_t* pair_csr_j,
    const int32_t* pair_row,
    int64_t n_outputs,
    int64_t D_total, int64_t L_total,
    int64_t nnzA, int64_t m,
    int64_t batchSize,
    cudaStream_t stream);

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
    cudaStream_t stream);

void form_schur_rhs_kernel(
    double* rhs_out,
    const double* rx,
    const double* rz,
    const double* h_inv,
    const double* A_values,
    const int64_t* A_colptr,
    const int32_t* A_rowval,
    const int32_t* csc_to_csr,
    const int32_t* new_to_old,  // band->original column order (null = identity)
    int64_t n, int64_t m,
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream);

void recover_z_kernel(
    double* z_out,
    const double* x_sol,
    const double* rz,
    const double* h_inv,
    const double* A_values,
    const int32_t* a_csr_row_start,
    const int32_t* a_csr_cols,
    const int32_t* a_csr_to_csc,
    int64_t n, int64_t m,
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream);

void add_eps_diagonal(
    double* data,
    int64_t total_per_batch,
    int64_t block_offset,
    int32_t block_size,
    double eps,
    int64_t batchSize,
    cudaStream_t stream);

void copy_block_data(
    double* dst, const double* src,
    int64_t block_elems,
    int64_t dst_stride, int64_t src_stride,
    int64_t dst_offset, int64_t src_offset,
    int64_t batchSize,
    cudaStream_t stream);

void transpose_block_data(
    double* dst, const double* src,
    int64_t dst_stride, int64_t src_stride,
    int64_t dst_offset, int64_t src_offset,
    int32_t src_rows, int32_t src_cols,
    int64_t batchSize,
    cudaStream_t stream);

void forward_sub_matvec(
    double* y_data,
    const double* L_data,
    const double* tmp,
    int64_t y_stride, int64_t L_total,
    int64_t y_offset_i, int64_t L_offset_im1,
    int32_t si, int32_t si_prev,
    int64_t tmp_stride,
    int64_t batchSize,
    cudaStream_t stream);

void backward_sub_matvec(
    double* y_data,
    const double* L_data,
    const double* x_next,
    int64_t y_stride, int64_t L_total, int64_t x_stride,
    int64_t y_offset_i, int64_t L_offset_i, int64_t x_offset_ip1,
    int32_t si, int32_t si_next,
    int64_t batchSize,
    cudaStream_t stream);

void setup_batch_pointers_kernel(
    double** ptr_array,
    double* base_data,
    const int64_t* offsets,
    int64_t total_per_batch,
    int64_t nblocks,
    int64_t batchSize,
    cudaStream_t stream);

/**
 * @brief Scatter A values from CSR order to CSC order
 *
 * d_A_csc_values[b*nnzA + csc_idx] = A_csr_values[b*nnzA + csr_idx]
 * where csc_idx = csr_to_csc_map[csr_idx]
 */
void scatter_A_csr_to_csc(
    double* A_csc_values,
    const double* A_csr_values,
    const int32_t* csr_to_csc_map,
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream);

/**
 * @brief Pack/unpack between interleaved [n+m] layout and separate [n], [m] buffers
 */
void unpack_kkt_rhs(
    double* rx, double* rz,
    const double* rhs,
    int64_t n, int64_t m,
    int64_t batchSize,
    cudaStream_t stream);

void pack_kkt_sol(
    double* sol,
    const double* x, const double* z,
    int64_t n, int64_t m,
    int64_t batchSize,
    cudaStream_t stream);

void negate_vector(
    double* dst, const double* src,
    int64_t total_elems,
    cudaStream_t stream);

// Scatter a band-ordered x-vector to original column order:
//   dst[b*n + new_to_old[i]] = src[b*n + i]
void permute_scatter(
    double* dst, const double* src, const int32_t* new_to_old,
    int64_t n, int64_t batchSize,
    cudaStream_t stream);

// All kernels below take `new_to_old` (band position -> original column). When
// non-null, rx/q reads gather through it and x-solution writes scatter through
// it, so the band-space solve interoperates with original-order RHS/solution
// buffers. Pass nullptr for the identity (unpermuted) order.

// Fused: form_schur_rhs reading directly from interleaved [n+m] buffer
void form_schur_rhs_interleaved_kernel(
    double* rhs_out,
    const double* interleaved_rhs,
    const double* h_inv, const double* A_values,
    const int64_t* A_colptr, const int32_t* A_rowval,
    const int32_t* csc_to_csr,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    double rx_sign,
    cudaStream_t stream);

// Fused: recover_z and pack x,z into interleaved [n+m] output
// rz is read from interleaved_rhs at offset n within each batch element
void recover_z_and_pack_kernel(
    double* interleaved_sol,
    const double* x_sol,
    const double* interleaved_rhs,
    const double* h_inv, const double* A_values,
    const int32_t* a_csr_row_start, const int32_t* a_csr_cols,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream);

// form_schur_rhs with sign flag on rx (eliminates negate_vector)
void form_schur_rhs_signed_kernel(
    double* rhs_out,
    const double* rx, const double* rz,
    const double* h_inv, const double* A_values,
    const int64_t* A_colptr, const int32_t* A_rowval,
    const int32_t* csc_to_csr,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    double rx_sign,
    cudaStream_t stream);

// 2-RHS: form_schur_rhs with both RHS from interleaved [n+m] buffers (for solve2)
void form_schur_rhs2_both_interleaved_kernel(
    double* rhs_out1, double* rhs_out2,
    const double* interleaved_rhs1,
    const double* interleaved_rhs2,
    const double* h_inv, const double* A_values,
    const int64_t* A_colptr, const int32_t* A_rowval,
    const int32_t* csc_to_csr,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream);

// 2-RHS: recover z for both, pack both into interleaved [n+m] outputs (for solve2)
void recover_z2_and_pack2_kernel(
    double* interleaved_sol1, double* interleaved_sol2,
    const double* x_sol1, const double* x_sol2,
    const double* interleaved_rhs1, const double* interleaved_rhs2,
    const double* h_inv, const double* A_values,
    const int32_t* a_csr_row_start, const int32_t* a_csr_cols,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream);

// 2-RHS: form_schur_rhs with RHS1 from interleaved, RHS2 from separate buffers
void form_schur_rhs2_interleaved_kernel(
    double* rhs_out1, double* rhs_out2,
    const double* interleaved_rhs1,
    const double* rx2, const double* rz2,
    const double* h_inv, const double* A_values,
    const int64_t* A_colptr, const int32_t* A_rowval,
    const int32_t* csc_to_csr,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream);

// 2-RHS: recover z for both, pack RHS1 into interleaved, copy RHS2 to separate buffers
// rz1 read from interleaved_rhs1 at offset n; rz2 from separate buffer
void recover_z2_and_pack_kernel(
    double* interleaved_sol1, double* z_out2,
    const double* x_sol1, const double* x_sol2,
    const double* interleaved_rhs1, const double* rz2,
    const double* h_inv, const double* A_values,
    const int32_t* a_csr_row_start, const int32_t* a_csr_cols,
    const int32_t* new_to_old,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream);

/**
 * @brief Block-tridiagonal Cholesky factorization
 *
 * Sequential over blocks, batched cuBLAS/cuSOLVER within each block.
 * D -> S copy, then for each block:
 *   if i > 0: S[i] -= L[i-1] * S[i-1]^{-1} * L[i-1]'
 *   S[i] += eps*I
 *   Cholesky(S[i])
 *
 * @return true on success, false if any Cholesky fails
 */
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
    double** d_S_block_ptrs = nullptr,
    double** d_L_block_ptrs = nullptr,
    double** d_work_block_ptrs = nullptr,
    double reg_eps = 1e-8);

/**
 * @brief Block-tridiagonal forward+backward solve
 *
 * Forward:  for i=0..nblocks-1: if i>0: y[i] -= L[i-1] * S[i-1]^{-1} y[i-1]
 * Backward: for i=nblocks-1..0: if i<nblocks-1: y[i] -= L[i]' x[i+1]; x[i] = S[i]^{-1} y[i]
 */
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
    double** d_S_block_ptrs = nullptr,
    double** d_work_block_ptrs = nullptr);

/**
 * @brief Multi-RHS block-tridiagonal solve (2 RHS simultaneously)
 *
 * Processes 2 right-hand sides with a single pass over the S/L factored data.
 * For fused paths, loads S[blk] into shared memory once and solves both RHS,
 * halving memory traffic for the Cholesky factor.
 */
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
    double** d_S_block_ptrs = nullptr,
    double** d_work_block_ptrs = nullptr);

/**
 * @brief Multi-RHS Schur complement RHS formation (2 RHS)
 */
void form_schur_rhs2_kernel(
    double* rhs_out1, double* rhs_out2,
    const double* rx1, const double* rz1,
    const double* rx2, const double* rz2,
    const double* h_inv, const double* A_values,
    const int64_t* A_colptr, const int32_t* A_rowval,
    const int32_t* csc_to_csr,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream);

/**
 * @brief Multi-RHS Z recovery (2 RHS)
 */
void recover_z2_kernel(
    double* z_out1, double* z_out2,
    const double* x_sol1, const double* x_sol2,
    const double* rz1, const double* rz2,
    const double* h_inv, const double* A_values,
    const int32_t* a_csr_row_start, const int32_t* a_csr_cols,
    int64_t n, int64_t m, int64_t nnzA, int64_t batchSize,
    cudaStream_t stream);

/**
 * @brief Multi-RHS unpack/pack (2 RHS)
 */
void unpack_kkt_rhs2(
    double* rx1, double* rz1, double* rx2, double* rz2,
    const double* rhs1, const double* rhs2,
    int64_t n, int64_t m, int64_t batchSize, cudaStream_t stream);

void pack_kkt_sol2(
    double* sol1, double* sol2,
    const double* x1, const double* z1,
    const double* x2, const double* z2,
    int64_t n, int64_t m, int64_t batchSize, cudaStream_t stream);

} // namespace moreau
