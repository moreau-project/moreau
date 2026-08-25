/**
 * @file diff_riccati.hpp
 * @brief Riccati-based backward pass for HSDE differentiation
 *
 * Solves (J'J + εI)*lam = -rhs_bar using block-tridiagonal Cholesky,
 * where J is the HSDE Jacobian (without τ row/column, handled via bordering).
 *
 * After eliminating λ_u (trivial diagonal block), the reduced system in
 * (λ_x, λ_w) has the block-tridiagonal structure:
 *
 *   [Mxx  Mxw] [λ_x]   [f_x]
 *   [Mwx  Mww] [λ_w] = [f_w]
 *
 * where:
 *   Mxx = P² + A'*D₁*A + εI   (D₁ = diag((h+ε)/(1+h+ε)))
 *   Mxw = P*A' + A'*D₂         (D₂ = diag(ε/(1+h+ε)))
 *   Mwx = Mxw'
 *   Mww = A*A' + D₃ + εI       (D₃ = diag(2 - (1+h)²/(1+h+ε)))
 *
 * The block size at each MPC time step is (d_x + d_w) where
 * d_x = forward block size and d_w = constraints assigned to that step.
 *
 * The τ variable is handled via bordering: solve 2 RHS, combine with
 * scalar equation. Recovery of λ_u uses the eliminated diagonal relation.
 *
 * Assembly uses a precomputed scatter map (built on CPU in constructor,
 * executed on GPU at solve time) — one thread per output element per batch.
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "moreau/cuda/utils.hpp"
#include "moreau/vector/vector.hpp"
#include "moreau/cones/cones.hpp"

namespace moreau {

/**
 * @brief Block-tridiagonal data for backward pass in (x,w) space.
 */
struct DiffRiccatiData {
    int64_t n;           // Number of primal variables (x-space)
    int64_t m;           // Number of constraints (w-space)
    int64_t nxw;         // n + m (total variables in reduced system)
    int64_t batchSize;
    int64_t nblocks;
    int32_t max_block;

    // Block structure in combined (x,w) space
    std::vector<int32_t> h_x_block_sizes;    // x-part size per block
    std::vector<int32_t> h_w_block_sizes;    // w-part size per block
    std::vector<int32_t> h_block_sizes;      // total size per block
    std::vector<int32_t> h_block_offsets;    // cumulative offsets in combined space
    std::vector<int32_t> h_x_block_offsets;  // cumulative x offsets
    std::vector<int32_t> h_w_block_offsets;  // cumulative w offsets

    device_unique_ptr<int32_t> d_block_sizes;
    device_unique_ptr<int32_t> d_block_offsets;

    // D and L block data (same layout as RiccatiKKTData)
    std::vector<int64_t> h_D_offsets;
    std::vector<int64_t> h_L_offsets;
    device_unique_ptr<int64_t> d_D_offsets;
    device_unique_ptr<int64_t> d_L_offsets;
    int64_t D_total_elems;
    int64_t L_total_elems;

    device_unique_ptr<double> d_D;      // Dense diagonal blocks [batchSize * D_total_elems]
    device_unique_ptr<double> d_L;      // Dense off-diagonal blocks [batchSize * L_total_elems]
    device_unique_ptr<double> d_S;      // Cholesky factors [batchSize * D_total_elems]
    device_unique_ptr<double> d_work;   // Workspace [batchSize * max_block²]

    // Solution vectors in (x,w) space [batchSize * nxw]
    device_unique_ptr<double> d_lhs;    // RHS/solution vector (also used as 3rd RHS)
    device_unique_ptr<double> d_lhs2;   // Second RHS for bordering

    // cuBLAS/cuSOLVER handles
    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    device_unique_ptr<double> d_cusolver_work;
    int cusolver_work_size;
    device_unique_ptr<int> d_cusolver_info;

    // Pointer arrays for batched operations
    device_unique_ptr<double*> d_ptr_A;
    device_unique_ptr<double*> d_ptr_B;
    device_unique_ptr<double*> d_ptr_C;
    device_unique_ptr<double*> d_S_block_ptrs;
    device_unique_ptr<double*> d_L_block_ptrs;
    device_unique_ptr<double*> d_work_block_ptrs;

    // Problem data (copied from solver, needed for assembly)
    device_unique_ptr<double> d_P_values;  // [batchSize * nnzP]
    device_unique_ptr<double> d_A_values;  // [batchSize * nnzA]
    int64_t nnzP, nnzA;

    // Host-side P and A CSR structure
    std::vector<int64_t> h_P_ro, h_P_ci;
    std::vector<int64_t> h_A_ro, h_A_ci;

    // Constraint-to-block mapping [m]
    std::vector<int32_t> h_constraint_block;  // which block each constraint belongs to

    // Column order that exposes the band: new_to_old[bp] = original x-variable at
    // band position bp. Empty/identity unless the forward solve reordered. The
    // x-variable -> (block, local) maps and the scatter maps bake this in, so the
    // RHS/solution kernels (which index by original x-variable) need no change.
    std::vector<int32_t> h_new_to_old;

    // Cone info
    int64_t numZeroCones;
    int64_t numNonnegCones;

    double reg_eps_ = 1e-8;

    // ================================================================
    // GPU scatter map for assembly (precomputed in constructor)
    // ================================================================
    // Unified scatter map: for each output element k in D/L, stores a list
    // of contribution pairs. Each pair encodes a product of two matrix values
    // (from P and/or A) optionally weighted by a constraint-dependent factor.
    //
    // Pair types:
    //   PP:  P[pair.x] * P[pair.y]                      (P² contribution)
    //   ADA: A[pair.x] * D₁[row] * A[pair.y]            (A'D₁A contribution)
    //   PA:  P[pair.x] * A[pair.y]                       (PA' contribution)
    //   AD:  D₂[row] * A[pair.x]  (pair.y unused)       (A'D₂ contribution)
    //   AA:  A[pair.x] * A[pair.y]                       (AA' contribution)
    //
    // Each type is stored in its own section of the pair array for efficiency.
    // Diagonal corrections (εI, D₃) are applied via flags per output element.

    int64_t n_asm_outputs;   // D_total_elems + L_total_elems

    // PP pairs (for P² contribution to xx blocks)
    device_unique_ptr<int32_t> d_pp_pair_ptr;  // [n_asm_outputs + 1]
    device_unique_ptr<int2>    d_pp_pair_ij;   // [n_pp_pairs] CSR indices into P
    int64_t n_pp_pairs;

    // ADA pairs (for A'D₁A contribution to xx blocks) — same structure as forward AHA
    device_unique_ptr<int32_t> d_ada_pair_ptr;  // [n_asm_outputs + 1]
    device_unique_ptr<int2>    d_ada_pair_ij;   // [n_ada_pairs] CSR indices into A
    device_unique_ptr<int32_t> d_ada_pair_row;  // [n_ada_pairs] constraint row
    int64_t n_ada_pairs;

    // PA pairs (for PA' + A'D₂ cross-term contribution to xw blocks)
    // Each pair: P[pair.x] * A[pair.y] (for PA' part)
    // Plus: D₂[row] * A[pair.x] (pair.y=-1 for A'D₂ part)
    device_unique_ptr<int32_t> d_pa_pair_ptr;  // [n_asm_outputs + 1]
    device_unique_ptr<int32_t> d_pa_pair_pidx; // [n_pa_pairs] P CSR index
    device_unique_ptr<int32_t> d_pa_pair_aidx; // [n_pa_pairs] A CSR index
    device_unique_ptr<int32_t> d_pa_pair_row;  // [n_pa_pairs] constraint row
    // type encoding: pidx >= 0 means PA pair, pidx < 0 means AD₂ pair
    int64_t n_pa_pairs;

    // AA' pairs (for AA' contribution to ww blocks)
    device_unique_ptr<int32_t> d_aa_pair_ptr;  // [n_asm_outputs + 1]
    device_unique_ptr<int2>    d_aa_pair_ij;   // [n_aa_pairs] CSR indices into A
    int64_t n_aa_pairs;

    // Per-output-element diagonal flags:
    //   diag_type = 0: no diagonal correction
    //   diag_type = 1: +ε (xx diagonal)
    //   diag_type = 2: +D₃[r] + ε (ww diagonal), with constraint r
    device_unique_ptr<int32_t> d_diag_type;  // [n_asm_outputs]
    device_unique_ptr<int32_t> d_diag_row;   // [n_asm_outputs] constraint row (for type 2)
    // AA' diagonal self-term: A[r,gamma]² for each gamma in row r
    device_unique_ptr<int32_t> d_aa_self_ptr;  // [n_asm_outputs + 1]
    device_unique_ptr<int32_t> d_aa_self_idx;  // [n_aa_self] CSR indices into A
    int64_t n_aa_self;

    // ================================================================
    // GPU mapping arrays for RHS formation / solution unpacking
    // ================================================================
    // x_to_block[alpha] = block index for x-variable alpha
    // x_local_idx[alpha] = local index within block for x-variable alpha
    // w_to_block[r] = block index for constraint r
    // w_local_idx[r] = local index within block for constraint r
    device_unique_ptr<int32_t> d_x_to_block;    // [n]
    device_unique_ptr<int32_t> d_x_local_idx;   // [n]
    device_unique_ptr<int32_t> d_w_to_block;    // [m]
    device_unique_ptr<int32_t> d_w_local_idx;   // [m]

    // A CSR structure on device (for SpMV in kernels)
    device_unique_ptr<int64_t> d_A_ro;  // [m+1]
    device_unique_ptr<int64_t> d_A_ci;  // [nnzA]
    device_unique_ptr<int64_t> d_P_ro;  // [n+1]
    device_unique_ptr<int64_t> d_P_ci;  // [nnzP]

    // CSC of A for A'*vec operations
    device_unique_ptr<int32_t> d_A_csc_colptr;  // [n+1]
    device_unique_ptr<int32_t> d_A_csc_rowidx;  // [nnzA]
    device_unique_ptr<int32_t> d_A_csc_to_csr;  // [nnzA] maps CSC position to CSR index

    DiffRiccatiData(
        int64_t n_, int64_t m_, int64_t batchSize_,
        const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP_,
        const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA_,
        const Cones& cones,
        const std::vector<int32_t>& x_block_sizes,  // forward Riccati block sizes
        const std::vector<int32_t>& perm = {},      // band->original column order ({} = identity)
        cudaStream_t stream = 0);

    ~DiffRiccatiData();

    DiffRiccatiData(const DiffRiccatiData&) = delete;
    DiffRiccatiData& operator=(const DiffRiccatiData&) = delete;

    void populate(const double* P_values, const double* A_values, cudaStream_t stream);
    bool assemble_and_factorize(const double* nonneg_H, cudaStream_t stream);
    void solve(double* rhs_sol, cudaStream_t stream);
    void solve2(double* rhs_sol1, double* rhs_sol2, cudaStream_t stream);

    [[nodiscard]] size_t memoryUsage() const noexcept;

private:
    void assign_constraints_to_blocks(
        const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA);
    void build_scatter_maps(
        const int64_t* P_ro, const int64_t* P_ci,
        const int64_t* A_ro, const int64_t* A_ci);
    void assemble_blocks(const double* nonneg_H, cudaStream_t stream);
};

} // namespace moreau
