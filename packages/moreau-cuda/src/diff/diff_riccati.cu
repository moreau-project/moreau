/**
 * @file diff_riccati.cu
 * @brief Implementation of DiffRiccatiData for J'J backward pass
 *
 * Assembly uses precomputed scatter maps (built on CPU at construction time)
 * for parallel GPU assembly at solve time — one thread per output element
 * per batch, zero host↔device transfers during solve.
 */

#include "moreau/diff/diff_riccati.hpp"
#include "moreau/kkt/riccati_kernels.cuh"
#include "moreau/cuda/utils.hpp"
#include "moreau/cuda/status_utils.hpp"
#include <algorithm>
#include <numeric>
#include <cassert>
#include <stdexcept>
#include <unordered_map>

namespace moreau {

// ============================================================================
// GPU assembly kernel for backward J'J+εI blocks
// ============================================================================
// Each thread computes one output element k for one batch element b.
// Accumulates PP, ADA, PA, AA' contributions + diagonal corrections.

static __global__ void backward_assemble_kernel(
    double* __restrict__ D_data,
    double* __restrict__ L_data,
    const double* __restrict__ P_values,
    const double* __restrict__ A_values,
    const double* __restrict__ nonneg_H,
    // PP pairs
    const int32_t* __restrict__ pp_pair_ptr,
    const int2* __restrict__ pp_pair_ij,
    // ADA pairs
    const int32_t* __restrict__ ada_pair_ptr,
    const int2* __restrict__ ada_pair_ij,
    const int32_t* __restrict__ ada_pair_row,
    // PA pairs
    const int32_t* __restrict__ pa_pair_ptr,
    const int32_t* __restrict__ pa_pair_pidx,
    const int32_t* __restrict__ pa_pair_aidx,
    const int32_t* __restrict__ pa_pair_row,
    // AA' pairs
    const int32_t* __restrict__ aa_pair_ptr,
    const int2* __restrict__ aa_pair_ij,
    // AA' self-terms (diagonal)
    const int32_t* __restrict__ aa_self_ptr,
    const int32_t* __restrict__ aa_self_idx,
    // Diagonal flags
    const int32_t* __restrict__ diag_type,
    const int32_t* __restrict__ diag_row,
    // Dimensions
    int64_t n_outputs,
    int64_t D_total, int64_t L_total,
    int64_t nnzP, int64_t nnzA,
    int64_t numZeroCones, int64_t numNonnegCones,
    double reg_eps,
    int64_t batchSize)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = batchSize * n_outputs;
    if (tid >= total) return;

    int64_t b = tid / n_outputs;
    int64_t k = tid % n_outputs;

    int64_t p_base = b * nnzP;
    int64_t a_base = b * nnzA;

    // Helper: get h value for constraint r
    auto get_h = [&](int32_t r) -> double {
        if (r < numZeroCones) return 1.0;
        if (r < numZeroCones + numNonnegCones)
            return nonneg_H[b * numNonnegCones + (r - numZeroCones)];
        return 1.0;
    };

    double val = 0.0;

    // 1. PP pairs: Σ P[pair.x]*P[pair.y]
    {
        int32_t start = pp_pair_ptr[k], end = pp_pair_ptr[k + 1];
        for (int32_t p = start; p < end; ++p) {
            int2 ij = pp_pair_ij[p];
            val += P_values[p_base + ij.x] * P_values[p_base + ij.y];
        }
    }

    // 2. ADA pairs: Σ D₁[r]*A[pair.x]*A[pair.y]
    {
        int32_t start = ada_pair_ptr[k], end = ada_pair_ptr[k + 1];
        for (int32_t p = start; p < end; ++p) {
            int2 ij = ada_pair_ij[p];
            int32_t r = ada_pair_row[p];
            double h = get_h(r);
            double d1 = (h + reg_eps) / (1.0 + h + reg_eps);
            val += d1 * A_values[a_base + ij.x] * A_values[a_base + ij.y];
        }
    }

    // 3. PA pairs: PA' contribution (pidx>=0) or A'D₂ contribution (pidx<0)
    {
        int32_t start = pa_pair_ptr[k], end = pa_pair_ptr[k + 1];
        for (int32_t p = start; p < end; ++p) {
            int32_t pidx = pa_pair_pidx[p];
            int32_t aidx = pa_pair_aidx[p];
            int32_t r = pa_pair_row[p];
            double h = get_h(r);
            if (pidx >= 0) {
                // PA' pair: P[pidx]*A[aidx]
                val += P_values[p_base + pidx] * A_values[a_base + aidx];
            } else {
                // A'D₂ pair: D₂[r]*A[aidx]
                double d2 = reg_eps / (1.0 + h + reg_eps);
                val += d2 * A_values[a_base + aidx];
            }
        }
    }

    // 4. AA' pairs: Σ A[pair.x]*A[pair.y]
    {
        int32_t start = aa_pair_ptr[k], end = aa_pair_ptr[k + 1];
        for (int32_t p = start; p < end; ++p) {
            int2 ij = aa_pair_ij[p];
            val += A_values[a_base + ij.x] * A_values[a_base + ij.y];
        }
    }

    // 5. AA' self-terms (diagonal): Σ A[idx]²
    {
        int32_t start = aa_self_ptr[k], end = aa_self_ptr[k + 1];
        for (int32_t p = start; p < end; ++p) {
            double a = A_values[a_base + aa_self_idx[p]];
            val += a * a;
        }
    }

    // 6. Diagonal corrections
    {
        int32_t dt = diag_type[k];
        if (dt == 1) {
            // xx diagonal: +ε
            val += reg_eps;
        } else if (dt == 2) {
            // ww diagonal: +D₃[r] + ε
            int32_t r = diag_row[k];
            double h = get_h(r);
            double d3 = 2.0 - (1.0 + h) * (1.0 + h) / (1.0 + h + reg_eps);
            val += d3 + reg_eps;
        }
    }

    // Write to D or L
    if (k < D_total) {
        D_data[b * D_total + k] = val;
    } else {
        L_data[b * L_total + (k - D_total)] = val;
    }
}

// ============================================================================
// Constructor
// ============================================================================

DiffRiccatiData::DiffRiccatiData(
    int64_t n_, int64_t m_, int64_t batchSize_,
    const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP_,
    const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA_,
    const Cones& cones,
    const std::vector<int32_t>& x_block_sizes,
    const std::vector<int32_t>& perm,
    cudaStream_t stream)
    : n(n_), m(m_), nxw(n_ + m_), batchSize(batchSize_),
      nblocks(x_block_sizes.size()),
      h_x_block_sizes(x_block_sizes),
      nnzP(nnzP_), nnzA(nnzA_),
      numZeroCones(cones.numZeroCones),
      numNonnegCones(cones.numNonnegCones)
{
    // Column order (band position -> original x-variable); identity if no perm.
    // The x-variable->(block,local) maps and the scatter maps read this so the
    // backward operates in the same band space the forward solved in.
    h_new_to_old.resize(n);
    if ((int64_t)perm.size() == n) {
        h_new_to_old.assign(perm.begin(), perm.end());
    } else {
        for (int64_t i = 0; i < n; ++i) h_new_to_old[i] = (int32_t)i;
    }

    // Store P and A CSR structure on host
    h_P_ro.assign(P_ro, P_ro + n + 1);
    h_P_ci.assign(P_ci, P_ci + nnzP);
    h_A_ro.assign(A_ro, A_ro + m + 1);
    h_A_ci.assign(A_ci, A_ci + nnzA);

    // Compute x-block offsets
    h_x_block_offsets.resize(nblocks + 1);
    h_x_block_offsets[0] = 0;
    for (int64_t i = 0; i < nblocks; ++i)
        h_x_block_offsets[i + 1] = h_x_block_offsets[i] + h_x_block_sizes[i];
    assert(h_x_block_offsets[nblocks] == n);

    // Assign constraints to blocks
    assign_constraints_to_blocks(A_ro, A_ci, nnzA);

    // Compute combined block sizes
    h_block_sizes.resize(nblocks);
    h_block_offsets.resize(nblocks + 1);
    h_w_block_offsets.resize(nblocks + 1);
    h_block_offsets[0] = 0;
    h_w_block_offsets[0] = 0;
    for (int64_t i = 0; i < nblocks; ++i) {
        h_block_sizes[i] = h_x_block_sizes[i] + h_w_block_sizes[i];
        h_block_offsets[i + 1] = h_block_offsets[i] + h_block_sizes[i];
        h_w_block_offsets[i + 1] = h_w_block_offsets[i] + h_w_block_sizes[i];
    }
    assert(h_block_offsets[nblocks] == nxw);
    assert(h_w_block_offsets[nblocks] == m);

    max_block = *std::max_element(h_block_sizes.begin(), h_block_sizes.end());

    // D and L offsets
    h_D_offsets.resize(nblocks + 1);
    h_D_offsets[0] = 0;
    for (int64_t i = 0; i < nblocks; ++i)
        h_D_offsets[i + 1] = h_D_offsets[i] + (int64_t)h_block_sizes[i] * h_block_sizes[i];
    D_total_elems = h_D_offsets[nblocks];

    h_L_offsets.resize(nblocks);
    h_L_offsets[0] = 0;
    for (int64_t i = 0; i < nblocks - 1; ++i)
        h_L_offsets[i + 1] = h_L_offsets[i] +
            (int64_t)h_block_sizes[i + 1] * h_block_sizes[i];
    L_total_elems = (nblocks > 1) ? h_L_offsets[nblocks - 1] : 0;

    // Upload helpers
    auto upload_i32 = [](device_unique_ptr<int32_t>& ptr, const int32_t* data, size_t count) {
        int32_t* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(int32_t) * count));
        ptr.reset(tmp);
        CUDA_THROW(cudaMemcpy(tmp, data, sizeof(int32_t) * count, cudaMemcpyHostToDevice));
    };
    auto upload_i64 = [](device_unique_ptr<int64_t>& ptr, const int64_t* data, size_t count) {
        int64_t* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(int64_t) * count));
        ptr.reset(tmp);
        CUDA_THROW(cudaMemcpy(tmp, data, sizeof(int64_t) * count, cudaMemcpyHostToDevice));
    };
    auto alloc_double = [](device_unique_ptr<double>& ptr, size_t count) {
        double* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(double) * count));
        ptr.reset(tmp);
        CUDA_THROW(cudaMemset(tmp, 0, sizeof(double) * count));
    };

    upload_i32(d_block_sizes, h_block_sizes.data(), nblocks);
    upload_i32(d_block_offsets, h_block_offsets.data(), nblocks + 1);
    upload_i64(d_D_offsets, h_D_offsets.data(), nblocks + 1);
    upload_i64(d_L_offsets, h_L_offsets.data(), nblocks);

    // Allocate dense block data
    alloc_double(d_D, batchSize * D_total_elems);
    if (L_total_elems > 0) alloc_double(d_L, batchSize * L_total_elems);
    alloc_double(d_S, batchSize * D_total_elems);
    alloc_double(d_work, batchSize * (int64_t)max_block * max_block);

    alloc_double(d_lhs, batchSize * nxw);
    alloc_double(d_lhs2, batchSize * nxw);

    {
        double* tmp;
        CUDA_THROW(cudaMalloc(&tmp, sizeof(double) * batchSize * std::max(nnzA, (int64_t)1)));
        d_A_values.reset(tmp);
        CUDA_THROW(cudaMalloc(&tmp, sizeof(double) * batchSize * std::max(nnzP, (int64_t)1)));
        d_P_values.reset(tmp);
    }

    // cuBLAS and pointer arrays
    {
        double** tmp;
        CUDA_THROW(cudaMalloc(&tmp, sizeof(double*) * batchSize)); d_ptr_A.reset(tmp);
        CUDA_THROW(cudaMalloc(&tmp, sizeof(double*) * batchSize)); d_ptr_B.reset(tmp);
        CUDA_THROW(cudaMalloc(&tmp, sizeof(double*) * batchSize)); d_ptr_C.reset(tmp);
    }
    cublas_handle = nullptr;
    CUBLAS_THROW(cublasCreate(&cublas_handle));

    CUSOLVER_THROW(cusolverDnCreate(&cusolver_handle));
    {
        CUSOLVER_THROW(cusolverDnDpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                                                   max_block, nullptr, max_block, &cusolver_work_size));
        double* tmp;
        CUDA_THROW(cudaMalloc(&tmp, sizeof(double) * cusolver_work_size));
        d_cusolver_work.reset(tmp);
        int* tmp_info;
        CUDA_THROW(cudaMalloc(&tmp_info, sizeof(int) * batchSize));
        d_cusolver_info.reset(tmp_info);
    }

    // Pre-compute S, L, work pointer arrays
    {
        int64_t work_stride = (int64_t)max_block * max_block;

        std::vector<double*> h_S_ptrs(nblocks * batchSize);
        for (int64_t i = 0; i < nblocks; ++i)
            for (int64_t b = 0; b < batchSize; ++b)
                h_S_ptrs[i * batchSize + b] = d_S.get() + b * D_total_elems + h_D_offsets[i];
        {
            double** tmp2; CUDA_THROW(cudaMalloc(&tmp2, sizeof(double*) * nblocks * batchSize));
            d_S_block_ptrs.reset(tmp2);
            CUDA_THROW(cudaMemcpy(tmp2, h_S_ptrs.data(), sizeof(double*) * nblocks * batchSize, cudaMemcpyHostToDevice));
        }

        if (nblocks > 1) {
            std::vector<double*> h_L_ptrs((nblocks - 1) * batchSize);
            for (int64_t i = 0; i < nblocks - 1; ++i)
                for (int64_t b = 0; b < batchSize; ++b)
                    h_L_ptrs[i * batchSize + b] = d_L.get() + b * L_total_elems + h_L_offsets[i];
            double** tmp2; CUDA_THROW(cudaMalloc(&tmp2, sizeof(double*) * (nblocks - 1) * batchSize));
            d_L_block_ptrs.reset(tmp2);
            CUDA_THROW(cudaMemcpy(tmp2, h_L_ptrs.data(), sizeof(double*) * (nblocks - 1) * batchSize, cudaMemcpyHostToDevice));
        }

        std::vector<double*> h_work_ptrs(batchSize);
        for (int64_t b = 0; b < batchSize; ++b)
            h_work_ptrs[b] = d_work.get() + b * work_stride;
        {
            double** tmp2; CUDA_THROW(cudaMalloc(&tmp2, sizeof(double*) * batchSize));
            d_work_block_ptrs.reset(tmp2);
            CUDA_THROW(cudaMemcpy(tmp2, h_work_ptrs.data(), sizeof(double*) * batchSize, cudaMemcpyHostToDevice));
        }
    }

    // Build GPU mapping arrays for RHS formation / solution unpacking
    {
        // Indexed by original x-variable: map each to its band block + local slot.
        std::vector<int32_t> h_x_to_block(n), h_x_local(n);
        for (int64_t bi = 0; bi < nblocks; ++bi)
            for (int32_t bp = h_x_block_offsets[bi]; bp < h_x_block_offsets[bi + 1]; ++bp) {
                int32_t orig = h_new_to_old[bp];
                h_x_to_block[orig] = (int32_t)bi;
                h_x_local[orig] = bp - h_x_block_offsets[bi];
            }
        upload_i32(d_x_to_block, h_x_to_block.data(), n);
        upload_i32(d_x_local_idx, h_x_local.data(), n);

        std::vector<int32_t> h_w_local(m);
        std::vector<int32_t> w_count(nblocks, 0);
        for (int64_t r = 0; r < m; ++r) {
            int32_t bi = h_constraint_block[r];
            h_w_local[r] = h_x_block_sizes[bi] + w_count[bi];
            w_count[bi]++;
        }
        upload_i32(d_w_to_block, h_constraint_block.data(), m);
        upload_i32(d_w_local_idx, h_w_local.data(), m);
    }

    // Upload CSR structure to device
    upload_i64(d_A_ro, A_ro, m + 1);
    upload_i64(d_A_ci, A_ci, nnzA);
    upload_i64(d_P_ro, P_ro, n + 1);
    upload_i64(d_P_ci, P_ci, nnzP);

    // Build CSC of A and upload
    {
        std::vector<int32_t> csc_colptr(n + 1, 0);
        for (int64_t r = 0; r < m; ++r)
            for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p)
                csc_colptr[A_ci[p] + 1]++;
        for (int64_t c = 0; c < n; ++c) csc_colptr[c + 1] += csc_colptr[c];

        std::vector<int32_t> csc_rowidx(nnzA), csc_to_csr(nnzA);
        std::vector<int32_t> fill(n, 0);
        for (int64_t r = 0; r < m; ++r) {
            for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p) {
                int32_t c = (int32_t)A_ci[p];
                int32_t pos = csc_colptr[c] + fill[c]++;
                csc_rowidx[pos] = (int32_t)r;
                csc_to_csr[pos] = (int32_t)p;
            }
        }

        upload_i32(d_A_csc_colptr, csc_colptr.data(), n + 1);
        upload_i32(d_A_csc_rowidx, csc_rowidx.data(), nnzA);
        upload_i32(d_A_csc_to_csr, csc_to_csr.data(), nnzA);
    }

    // Build scatter maps
    build_scatter_maps(P_ro, P_ci, A_ro, A_ci);

    cudaStreamSynchronize(stream);
}

DiffRiccatiData::~DiffRiccatiData() {
    if (cusolver_handle) cusolverDnDestroy(cusolver_handle);
    if (cublas_handle) cublasDestroy(cublas_handle);
}

// ============================================================================
// Constraint-to-block assignment
// ============================================================================

void DiffRiccatiData::assign_constraints_to_blocks(
    const int64_t* A_ro, const int64_t* A_ci, int64_t /*nnzA*/)
{
    std::vector<int32_t> x_to_block(n);  // indexed by original x-variable
    for (int64_t bi = 0; bi < nblocks; ++bi)
        for (int32_t bp = h_x_block_offsets[bi]; bp < h_x_block_offsets[bi + 1]; ++bp)
            x_to_block[h_new_to_old[bp]] = (int32_t)bi;

    h_constraint_block.resize(m);
    h_w_block_sizes.resize(nblocks, 0);

    for (int64_t r = 0; r < m; ++r) {
        int32_t max_bi = 0;
        for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p)
            max_bi = std::max(max_bi, x_to_block[A_ci[p]]);
        h_constraint_block[r] = max_bi;
        h_w_block_sizes[max_bi]++;
    }
}

// ============================================================================
// Scatter map construction (CPU, runs once at construction)
// ============================================================================

void DiffRiccatiData::build_scatter_maps(
    const int64_t* P_ro, const int64_t* P_ci,
    const int64_t* A_ro, const int64_t* A_ci)
{
    n_asm_outputs = D_total_elems + L_total_elems;
    if (n_asm_outputs == 0) return;

    // Precompute mappings
    std::vector<int32_t> x_to_block(n);
    for (int64_t bi = 0; bi < nblocks; ++bi)
        for (int32_t c = h_x_block_offsets[bi]; c < h_x_block_offsets[bi + 1]; ++c)
            x_to_block[c] = (int32_t)bi;

    std::vector<int32_t> w_local_idx(m);
    {
        std::vector<int32_t> w_count(nblocks, 0);
        for (int64_t r = 0; r < m; ++r) {
            int32_t bi = h_constraint_block[r];
            w_local_idx[r] = h_x_block_sizes[bi] + w_count[bi];
            w_count[bi]++;
        }
    }

    // Build CSC of A for column intersection
    std::vector<int32_t> csc_col_start(n + 1, 0);
    for (int64_t r = 0; r < m; ++r)
        for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p)
            csc_col_start[A_ci[p] + 1]++;
    for (int64_t c = 0; c < n; ++c) csc_col_start[c + 1] += csc_col_start[c];
    std::vector<int32_t> csc_rows(nnzA), csc_csr_idx(nnzA);
    {
        std::vector<int32_t> fill(n, 0);
        for (int64_t r = 0; r < m; ++r)
            for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p) {
                int32_t c = (int32_t)A_ci[p];
                int32_t pos = csc_col_start[c] + fill[c]++;
                csc_rows[pos] = (int32_t)r;
                csc_csr_idx[pos] = (int32_t)p;
            }
    }

    // Build CSC of P for column intersection
    std::vector<int32_t> pcsc_col_start(n + 1, 0);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t p = P_ro[i]; p < P_ro[i + 1]; ++p)
            pcsc_col_start[P_ci[p] + 1]++;
    for (int64_t c = 0; c < n; ++c) pcsc_col_start[c + 1] += pcsc_col_start[c];
    std::vector<int32_t> pcsc_rows(nnzP), pcsc_csr_idx(nnzP);
    {
        std::vector<int32_t> fill(n, 0);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t p = P_ro[i]; p < P_ro[i + 1]; ++p) {
                int32_t c = (int32_t)P_ci[p];
                int32_t pos = pcsc_col_start[c] + fill[c]++;
                pcsc_rows[pos] = (int32_t)i;
                pcsc_csr_idx[pos] = (int32_t)p;
            }
    }

    // Per-output arrays
    std::vector<int32_t> h_pp_ptr(n_asm_outputs + 1, 0);
    std::vector<int2> h_pp_ij;
    std::vector<int32_t> h_ada_ptr(n_asm_outputs + 1, 0);
    std::vector<int2> h_ada_ij;
    std::vector<int32_t> h_ada_row;
    std::vector<int32_t> h_pa_ptr(n_asm_outputs + 1, 0);
    std::vector<int32_t> h_pa_pidx, h_pa_aidx, h_pa_row;
    std::vector<int32_t> h_aa_ptr(n_asm_outputs + 1, 0);
    std::vector<int2> h_aa_ij;
    std::vector<int32_t> h_aa_self_ptr_v(n_asm_outputs + 1, 0);
    std::vector<int32_t> h_aa_self_idx_v;

    std::vector<int32_t> h_diag_type_v(n_asm_outputs, 0);
    std::vector<int32_t> h_diag_row_v(n_asm_outputs, -1);

    // Helper: given output element k, what are the block-local (row, col) and blocks?
    // We iterate D then L blocks in the same order as memory layout.

    int64_t out_idx = 0;

    // Lambda to compute output element's (bi_row, bi_col, li_row, li_col, is_x_row, is_x_col)
    // For D block bi: elements are (li, lj) in [0, sz)×[0, sz), col-major k = li + lj*sz
    //   li < x_size[bi] → x-variable, global index = x_block_offsets[bi] + li
    //   li >= x_size[bi] → w-variable, constraint index computed from w mapping

    // Build reverse w mapping: for each block, list of constraint indices
    std::vector<std::vector<int64_t>> w_in_block(nblocks);
    for (int64_t r = 0; r < m; ++r)
        w_in_block[h_constraint_block[r]].push_back(r);

    // Helper: given (block, local_idx), return (is_x, global_idx)
    // is_x=true: global_idx is x-variable index
    // is_x=false: global_idx is constraint index
    // For an x-slot, return the *original* column at that band position so the
    // P/A/CSC lookups below (all in original-column space) stay consistent.
    auto local_to_global = [&](int32_t bi, int32_t li) -> std::pair<bool, int64_t> {
        if (li < h_x_block_sizes[bi])
            return {true, h_new_to_old[h_x_block_offsets[bi] + li]};
        int32_t wi = li - h_x_block_sizes[bi];
        return {false, w_in_block[bi][wi]};
    };

    // D blocks
    for (int64_t blk = 0; blk < nblocks; ++blk) {
        int32_t sz = h_block_sizes[blk];
        for (int32_t lj = 0; lj < sz; ++lj) {
            for (int32_t li = 0; li < sz; ++li) {
                auto [is_x_i, gi] = local_to_global(blk, li);
                auto [is_x_j, gj] = local_to_global(blk, lj);

                h_pp_ptr[out_idx] = (int32_t)h_pp_ij.size();
                h_ada_ptr[out_idx] = (int32_t)h_ada_ij.size();
                h_pa_ptr[out_idx] = (int32_t)h_pa_pidx.size();
                h_aa_ptr[out_idx] = (int32_t)h_aa_ij.size();
                h_aa_self_ptr_v[out_idx] = (int32_t)h_aa_self_idx_v.size();

                if (is_x_i && is_x_j) {
                    // xx block: P² + A'D₁A + εI
                    int64_t alpha = gi, beta = gj;

                    // P² pairs: iterate P[α,:], for each γ check P[β,γ] via CSC.
                    for (int64_t pa = P_ro[alpha]; pa < P_ro[alpha + 1]; ++pa) {
                        int64_t gamma = P_ci[pa];
                        // Look for P[β,γ] in CSC column γ
                        for (int32_t q = pcsc_col_start[gamma]; q < pcsc_col_start[gamma + 1]; ++q) {
                            if (pcsc_rows[q] == (int32_t)beta) {
                                h_pp_ij.push_back(make_int2((int32_t)pa, pcsc_csr_idx[q]));
                                break;
                            }
                        }
                    }

                    // A'D₁A pairs: Σ_r D₁[r]*A[r,α]*A[r,β]
                    // Intersect CSC columns α and β
                    int32_t pi = csc_col_start[alpha], pi_end = csc_col_start[alpha + 1];
                    int32_t pj = csc_col_start[beta], pj_end = csc_col_start[beta + 1];
                    while (pi < pi_end && pj < pj_end) {
                        if (csc_rows[pi] < csc_rows[pj]) ++pi;
                        else if (csc_rows[pi] > csc_rows[pj]) ++pj;
                        else {
                            h_ada_ij.push_back(make_int2(csc_csr_idx[pi], csc_csr_idx[pj]));
                            h_ada_row.push_back(csc_rows[pi]);
                            ++pi; ++pj;
                        }
                    }

                    // εI on diagonal
                    if (alpha == beta)
                        h_diag_type_v[out_idx] = 1;

                } else if (is_x_i && !is_x_j) {
                    // xw block: PA' + A'D₂
                    int64_t alpha = gi, r = gj;

                    // PA' part: Σ_γ P[α,γ]*A[r,γ]
                    for (int64_t pa = P_ro[alpha]; pa < P_ro[alpha + 1]; ++pa) {
                        int64_t gamma = P_ci[pa];
                        for (int64_t pb = A_ro[r]; pb < A_ro[r + 1]; ++pb) {
                            if (A_ci[pb] == gamma) {
                                h_pa_pidx.push_back((int32_t)pa);
                                h_pa_aidx.push_back((int32_t)pb);
                                h_pa_row.push_back((int32_t)r);
                                break;
                            }
                        }
                    }

                    // A'D₂ part: D₂[r]*A[r,α]
                    for (int64_t pb = A_ro[r]; pb < A_ro[r + 1]; ++pb) {
                        if (A_ci[pb] == alpha) {
                            h_pa_pidx.push_back(-1);  // marker for A'D₂
                            h_pa_aidx.push_back((int32_t)pb);
                            h_pa_row.push_back((int32_t)r);
                            break;
                        }
                    }

                } else if (!is_x_i && is_x_j) {
                    // wx block: AP + D₂A (transpose of xw)
                    int64_t r = gi, alpha = gj;

                    // AP part: Σ_γ A[r,γ]*P[α,γ]
                    for (int64_t pa = P_ro[alpha]; pa < P_ro[alpha + 1]; ++pa) {
                        int64_t gamma = P_ci[pa];
                        for (int64_t pb = A_ro[r]; pb < A_ro[r + 1]; ++pb) {
                            if (A_ci[pb] == gamma) {
                                h_pa_pidx.push_back((int32_t)pa);
                                h_pa_aidx.push_back((int32_t)pb);
                                h_pa_row.push_back((int32_t)r);
                                break;
                            }
                        }
                    }

                    // D₂A part: D₂[r]*A[r,α]
                    for (int64_t pb = A_ro[r]; pb < A_ro[r + 1]; ++pb) {
                        if (A_ci[pb] == alpha) {
                            h_pa_pidx.push_back(-1);
                            h_pa_aidx.push_back((int32_t)pb);
                            h_pa_row.push_back((int32_t)r);
                            break;
                        }
                    }

                } else {
                    // ww block: AA' + D₃ + εI
                    int64_t r = gi, s = gj;

                    if (r == s) {
                        // Diagonal: D₃[r] + ε + Σ_γ A[r,γ]²
                        h_diag_type_v[out_idx] = 2;
                        h_diag_row_v[out_idx] = (int32_t)r;
                        // Self-terms: A[r,γ]² for each γ
                        for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p)
                            h_aa_self_idx_v.push_back((int32_t)p);
                    } else {
                        // Off-diagonal: Σ_γ A[r,γ]*A[s,γ]
                        for (int64_t pa = A_ro[r]; pa < A_ro[r + 1]; ++pa) {
                            int64_t gamma = A_ci[pa];
                            for (int64_t pb = A_ro[s]; pb < A_ro[s + 1]; ++pb) {
                                if (A_ci[pb] == gamma) {
                                    h_aa_ij.push_back(make_int2((int32_t)pa, (int32_t)pb));
                                    break;
                                }
                            }
                        }
                    }
                }

                out_idx++;
            }
        }
    }

    // L blocks
    for (int64_t blk = 0; blk < nblocks - 1; ++blk) {
        int32_t sz_cur = h_block_sizes[blk];
        int32_t sz_next = h_block_sizes[blk + 1];
        for (int32_t lj = 0; lj < sz_cur; ++lj) {
            for (int32_t li = 0; li < sz_next; ++li) {
                auto [is_x_i, gi] = local_to_global(blk + 1, li);
                auto [is_x_j, gj] = local_to_global(blk, lj);

                h_pp_ptr[out_idx] = (int32_t)h_pp_ij.size();
                h_ada_ptr[out_idx] = (int32_t)h_ada_ij.size();
                h_pa_ptr[out_idx] = (int32_t)h_pa_pidx.size();
                h_aa_ptr[out_idx] = (int32_t)h_aa_ij.size();
                h_aa_self_ptr_v[out_idx] = (int32_t)h_aa_self_idx_v.size();

                if (is_x_i && is_x_j) {
                    // xx L-block: P² + A'D₁A
                    int64_t alpha = gi, beta = gj;

                    // P² pairs
                    for (int64_t pa = P_ro[alpha]; pa < P_ro[alpha + 1]; ++pa) {
                        int64_t gamma = P_ci[pa];
                        for (int32_t q = pcsc_col_start[gamma]; q < pcsc_col_start[gamma + 1]; ++q) {
                            if (pcsc_rows[q] == (int32_t)beta) {
                                h_pp_ij.push_back(make_int2((int32_t)pa, pcsc_csr_idx[q]));
                                break;
                            }
                        }
                    }

                    // A'D₁A pairs
                    int32_t pi = csc_col_start[alpha], pi_end = csc_col_start[alpha + 1];
                    int32_t pj = csc_col_start[beta], pj_end = csc_col_start[beta + 1];
                    while (pi < pi_end && pj < pj_end) {
                        if (csc_rows[pi] < csc_rows[pj]) ++pi;
                        else if (csc_rows[pi] > csc_rows[pj]) ++pj;
                        else {
                            h_ada_ij.push_back(make_int2(csc_csr_idx[pi], csc_csr_idx[pj]));
                            h_ada_row.push_back(csc_rows[pi]);
                            ++pi; ++pj;
                        }
                    }

                } else if (is_x_i && !is_x_j) {
                    // xw L-block
                    int64_t alpha = gi, r = gj;

                    for (int64_t pa = P_ro[alpha]; pa < P_ro[alpha + 1]; ++pa) {
                        int64_t gamma = P_ci[pa];
                        for (int64_t pb = A_ro[r]; pb < A_ro[r + 1]; ++pb) {
                            if (A_ci[pb] == gamma) {
                                h_pa_pidx.push_back((int32_t)pa);
                                h_pa_aidx.push_back((int32_t)pb);
                                h_pa_row.push_back((int32_t)r);
                                break;
                            }
                        }
                    }

                    for (int64_t pb = A_ro[r]; pb < A_ro[r + 1]; ++pb) {
                        if (A_ci[pb] == alpha) {
                            h_pa_pidx.push_back(-1);
                            h_pa_aidx.push_back((int32_t)pb);
                            h_pa_row.push_back((int32_t)r);
                            break;
                        }
                    }

                } else if (!is_x_i && is_x_j) {
                    // wx L-block
                    int64_t r = gi, alpha = gj;

                    for (int64_t pa = P_ro[alpha]; pa < P_ro[alpha + 1]; ++pa) {
                        int64_t gamma = P_ci[pa];
                        for (int64_t pb = A_ro[r]; pb < A_ro[r + 1]; ++pb) {
                            if (A_ci[pb] == gamma) {
                                h_pa_pidx.push_back((int32_t)pa);
                                h_pa_aidx.push_back((int32_t)pb);
                                h_pa_row.push_back((int32_t)r);
                                break;
                            }
                        }
                    }

                    for (int64_t pb = A_ro[r]; pb < A_ro[r + 1]; ++pb) {
                        if (A_ci[pb] == alpha) {
                            h_pa_pidx.push_back(-1);
                            h_pa_aidx.push_back((int32_t)pb);
                            h_pa_row.push_back((int32_t)r);
                            break;
                        }
                    }

                } else {
                    // ww L-block
                    int64_t r = gi, s = gj;
                    for (int64_t pa = A_ro[r]; pa < A_ro[r + 1]; ++pa) {
                        int64_t gamma = A_ci[pa];
                        for (int64_t pb = A_ro[s]; pb < A_ro[s + 1]; ++pb) {
                            if (A_ci[pb] == gamma) {
                                h_aa_ij.push_back(make_int2((int32_t)pa, (int32_t)pb));
                                break;
                            }
                        }
                    }
                }

                out_idx++;
            }
        }
    }

    assert(out_idx == n_asm_outputs);

    // Finalize CSR pointers
    h_pp_ptr[n_asm_outputs] = (int32_t)h_pp_ij.size();
    h_ada_ptr[n_asm_outputs] = (int32_t)h_ada_ij.size();
    h_pa_ptr[n_asm_outputs] = (int32_t)h_pa_pidx.size();
    h_aa_ptr[n_asm_outputs] = (int32_t)h_aa_ij.size();
    h_aa_self_ptr_v[n_asm_outputs] = (int32_t)h_aa_self_idx_v.size();

    n_pp_pairs = (int64_t)h_pp_ij.size();
    n_ada_pairs = (int64_t)h_ada_ij.size();
    n_pa_pairs = (int64_t)h_pa_pidx.size();
    n_aa_pairs = (int64_t)h_aa_ij.size();
    n_aa_self = (int64_t)h_aa_self_idx_v.size();

    // Upload all scatter maps to device
    auto upload_i32_vec = [](device_unique_ptr<int32_t>& ptr, const std::vector<int32_t>& v) {
        if (v.empty()) return;
        int32_t* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(int32_t) * v.size()));
        ptr.reset(tmp);
        CUDA_THROW(cudaMemcpy(tmp, v.data(), sizeof(int32_t) * v.size(), cudaMemcpyHostToDevice));
    };
    auto upload_int2_vec = [](device_unique_ptr<int2>& ptr, const std::vector<int2>& v) {
        if (v.empty()) return;
        int2* tmp; CUDA_THROW(cudaMalloc(&tmp, sizeof(int2) * v.size()));
        ptr.reset(tmp);
        CUDA_THROW(cudaMemcpy(tmp, v.data(), sizeof(int2) * v.size(), cudaMemcpyHostToDevice));
    };

    upload_i32_vec(d_pp_pair_ptr, h_pp_ptr);
    upload_int2_vec(d_pp_pair_ij, h_pp_ij);
    upload_i32_vec(d_ada_pair_ptr, h_ada_ptr);
    upload_int2_vec(d_ada_pair_ij, h_ada_ij);
    upload_i32_vec(d_ada_pair_row, h_ada_row);
    upload_i32_vec(d_pa_pair_ptr, h_pa_ptr);
    upload_i32_vec(d_pa_pair_pidx, h_pa_pidx);
    upload_i32_vec(d_pa_pair_aidx, h_pa_aidx);
    upload_i32_vec(d_pa_pair_row, h_pa_row);
    upload_i32_vec(d_aa_pair_ptr, h_aa_ptr);
    upload_int2_vec(d_aa_pair_ij, h_aa_ij);
    upload_i32_vec(d_aa_self_ptr, h_aa_self_ptr_v);
    upload_i32_vec(d_aa_self_idx, h_aa_self_idx_v);
    upload_i32_vec(d_diag_type, h_diag_type_v);
    upload_i32_vec(d_diag_row, h_diag_row_v);
}

// ============================================================================
// Populate
// ============================================================================

void DiffRiccatiData::populate(const double* P_values, const double* A_values, cudaStream_t stream) {
    if (nnzP > 0)
        CUDA_THROW(cudaMemcpyAsync(d_P_values.get(), P_values,
                       sizeof(double) * batchSize * nnzP,
                       cudaMemcpyDeviceToDevice, stream));
    if (nnzA > 0)
        CUDA_THROW(cudaMemcpyAsync(d_A_values.get(), A_values,
                       sizeof(double) * batchSize * nnzA,
                       cudaMemcpyDeviceToDevice, stream));
}

// ============================================================================
// GPU assembly
// ============================================================================

void DiffRiccatiData::assemble_blocks(const double* d_nonneg_H, cudaStream_t stream) {
    if (n_asm_outputs == 0) return;

    int64_t total = batchSize * n_asm_outputs;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);

    backward_assemble_kernel<<<blocks, threads, 0, stream>>>(
        d_D.get(), d_L.get(),
        d_P_values.get(), d_A_values.get(),
        d_nonneg_H,
        d_pp_pair_ptr.get(), d_pp_pair_ij.get(),
        d_ada_pair_ptr.get(), d_ada_pair_ij.get(), d_ada_pair_row.get(),
        d_pa_pair_ptr.get(), d_pa_pair_pidx.get(), d_pa_pair_aidx.get(), d_pa_pair_row.get(),
        d_aa_pair_ptr.get(), d_aa_pair_ij.get(),
        d_aa_self_ptr.get(), d_aa_self_idx.get(),
        d_diag_type.get(), d_diag_row.get(),
        n_asm_outputs, D_total_elems, L_total_elems,
        nnzP, nnzA,
        numZeroCones, numNonnegCones,
        reg_eps_, batchSize);
}

// ============================================================================
// Assemble and factorize
// ============================================================================

bool DiffRiccatiData::assemble_and_factorize(const double* nonneg_H, cudaStream_t stream) {
    assemble_blocks(nonneg_H, stream);

    return riccati_factorize_blocks(
        d_S.get(), d_D.get(), d_L.get(),
        d_work.get(), d_ptr_A.get(), d_ptr_B.get(), d_ptr_C.get(),
        h_D_offsets.data(), h_L_offsets.data(), h_block_sizes.data(),
        d_D_offsets.get(), d_L_offsets.get(), d_block_sizes.get(),
        D_total_elems, L_total_elems,
        nblocks, max_block, batchSize,
        cublas_handle, cusolver_handle, d_cusolver_info.get(),
        stream,
        d_S_block_ptrs.get(), d_L_block_ptrs.get(), d_work_block_ptrs.get(),
        reg_eps_);
}

// ============================================================================
// Solve
// ============================================================================

void DiffRiccatiData::solve(double* rhs_sol, cudaStream_t stream) {
    riccati_solve_blocks(
        rhs_sol, d_S.get(), d_L.get(),
        d_work.get(), d_ptr_A.get(), d_ptr_B.get(),
        h_D_offsets.data(), h_L_offsets.data(),
        h_block_offsets.data(), h_block_sizes.data(),
        d_D_offsets.get(), d_L_offsets.get(),
        d_block_offsets.get(), d_block_sizes.get(),
        D_total_elems, L_total_elems,
        nxw, nblocks, max_block, batchSize,
        cublas_handle, stream,
        d_S_block_ptrs.get(), d_work_block_ptrs.get());
}

void DiffRiccatiData::solve2(double* rhs_sol1, double* rhs_sol2, cudaStream_t stream) {
    riccati_solve2_blocks(
        rhs_sol1, rhs_sol2, d_S.get(), d_L.get(),
        d_work.get(), d_ptr_A.get(), d_ptr_B.get(),
        h_D_offsets.data(), h_L_offsets.data(),
        h_block_offsets.data(), h_block_sizes.data(),
        d_D_offsets.get(), d_L_offsets.get(),
        d_block_offsets.get(), d_block_sizes.get(),
        D_total_elems, L_total_elems,
        nxw, nblocks, max_block, batchSize,
        cublas_handle, stream,
        d_S_block_ptrs.get(), d_work_block_ptrs.get());
}

size_t DiffRiccatiData::memoryUsage() const noexcept {
    return sizeof(double) * batchSize * (D_total_elems * 2 + L_total_elems +
        (int64_t)max_block * max_block + nxw * 2 + nnzP + nnzA);
}

} // namespace moreau
