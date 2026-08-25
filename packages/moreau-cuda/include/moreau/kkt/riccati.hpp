/**
 * @file riccati.hpp
 * @brief Batched block-tridiagonal (Riccati) KKT solver for CUDA
 *
 * When the Schur complement M = P + A'H^{-1}A is block-tridiagonal (e.g., MPC/MHE
 * problems), we exploit this structure via block-tridiagonal Cholesky factorization
 * (Riccati recursion). This is O(T*d^3) instead of O((T*d)^3), and maps naturally
 * to cuBLAS batched dense operations for massive GPU parallelism.
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <memory>
#include <vector>
#include <cstdint>
#include "moreau/cuda/utils.hpp"
#include "moreau/matrix/csr.hpp"
#include "moreau/vector/vector.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/kkt/kkt_solver.hpp"
#include "moreau/kkt/riccati_kernels.cuh"

namespace moreau {

enum class ScalingStrategy;

// Detect a block-tridiagonal column order for M = P + A'H^{-1}A and return the
// block sizes (empty if none). When `allow_permute` is true and the given order
// is not banded, a Reverse Cuthill–McKee reorder is attempted; if it exposes a
// banding, `*out_perm` receives the column order (new_to_old[bp] = original
// column) and the returned block sizes are in that permuted space. `*out_perm`
// is left empty for the identity (given) order.
std::vector<int32_t> detect_block_tridiagonal(
    int64_t n, int64_t m,
    const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
    const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
    const Cones& cones,
    bool allow_permute = false,
    std::vector<int32_t>* out_perm = nullptr);

struct RiccatiKKTData : public KKTSolver {
    int64_t n, m, batchSize;
    int64_t nblocks;
    int32_t max_block;

    // Block structure (host + device)
    std::vector<int32_t> h_block_sizes;
    std::vector<int32_t> h_block_offsets;
    std::vector<int32_t> h_col_to_block;

    device_unique_ptr<int32_t> d_block_sizes;
    device_unique_ptr<int32_t> d_block_offsets;
    device_unique_ptr<int32_t> d_col_to_block;

    // Column permutation that exposes the band: new_to_old[bp] = original column
    // at band position bp. has_perm_ == false means identity (column index ==
    // band position) — the device array is then null and kernels skip the
    // gather/scatter. Internally everything works in band space; only the
    // x-vector boundaries (RHS reads, solution writes) map through this order.
    bool has_perm_ = false;
    std::vector<int32_t> h_new_to_old;
    device_unique_ptr<int32_t> d_new_to_old;

    // Cumulative element offsets for D and L blocks
    std::vector<int64_t> h_D_offsets;
    std::vector<int64_t> h_L_offsets;
    device_unique_ptr<int64_t> d_D_offsets;
    device_unique_ptr<int64_t> d_L_offsets;

    int64_t D_total_elems;
    int64_t L_total_elems;

    // Dense block data [batchSize * total_elems]
    device_unique_ptr<double> d_D;
    device_unique_ptr<double> d_L;
    device_unique_ptr<double> d_S;     // Cholesky factors
    device_unique_ptr<double> d_work;  // workspace

    // h_inv [batchSize * m]
    device_unique_ptr<double> d_h_inv;

    // A values stored in CSR order [batchSize * nnzA]
    // P values stored in CSR order [batchSize * nnzP]
    device_unique_ptr<double> d_A_values;
    device_unique_ptr<double> d_P_values;
    int64_t nnzP, nnzA;

    // CSC structure of A (for column-oriented access in form_schur_rhs)
    device_unique_ptr<int64_t> d_A_colptr;   // [n+1]
    device_unique_ptr<int32_t> d_A_rowval;   // [nnzA]
    device_unique_ptr<int32_t> d_csc_to_csr; // [nnzA] maps CSC position -> CSR index

    // CSR view of A (for row-oriented access in recover_z and AHA assembly)
    device_unique_ptr<int32_t> d_a_csr_row_start;  // [m+1]
    device_unique_ptr<int32_t> d_a_csr_cols;        // [nnzA]
    device_unique_ptr<int32_t> d_a_csr_to_csc;      // [nnzA] CSR idx -> CSC idx

    // Fused AHA+P scatter map: precomputed gather for M = P + A'*H^{-1}*A assembly
    // Output elements enumerated as D[0..D_total) then L[D_total..D_total+L_total)
    int64_t n_aha_outputs;   // D_total_elems + L_total_elems
    int64_t n_aha_pairs;     // total number of (csr_i, csr_j, row) triples
    device_unique_ptr<int32_t> d_aha_pair_ptr;    // [n_outputs+1] CSR-style pointers
    device_unique_ptr<int2> d_aha_pair_ij;        // [n_pairs] packed (csr_i, csr_j)
    device_unique_ptr<int32_t> d_aha_pair_row;    // [n_pairs]
    device_unique_ptr<int32_t> d_aha_p_val_idx;   // [n_outputs] P CSR index or -1

    // Workspace vectors
    device_unique_ptr<double> d_schur_rhs;  // [batchSize * n]
    device_unique_ptr<double> d_lhsx;       // [batchSize * n]
    device_unique_ptr<double> d_lhsz;       // [batchSize * m]
    device_unique_ptr<double> d_rhsx;       // [batchSize * n]
    device_unique_ptr<double> d_rhsz;       // [batchSize * m]

    // Second workspace for multi-RHS solve2
    device_unique_ptr<double> d_lhsx2;      // [batchSize * n]
    device_unique_ptr<double> d_lhsz2;      // [batchSize * m]
    device_unique_ptr<double> d_rhsx2;      // [batchSize * n]
    device_unique_ptr<double> d_rhsz2;      // [batchSize * m]

    // cuBLAS batched pointer arrays [batchSize] each
    device_unique_ptr<double*> d_ptr_A;
    device_unique_ptr<double*> d_ptr_B;
    device_unique_ptr<double*> d_ptr_C;

    // Pre-computed pointer arrays for S, L, and work blocks [nblocks * batchSize]
    // Avoids per-step setup_ptrs kernel launches in the cuBLAS path
    device_unique_ptr<double*> d_S_block_ptrs;   // [nblocks][batchSize] -> S[i] for each batch
    device_unique_ptr<double*> d_L_block_ptrs;   // [(nblocks-1)][batchSize] -> L[i] for each batch
    device_unique_ptr<double*> d_work_block_ptrs; // [batchSize] -> work for each batch

    cublasHandle_t cublas_handle;
    cusolverDnHandle_t cusolver_handle;
    device_unique_ptr<double> d_cusolver_work;
    int cusolver_work_size;
    device_unique_ptr<int> d_cusolver_info;  // [batchSize]
    bool populated_ = false;
    double reg_eps_ = 1e-8;  // Cholesky regularization epsilon

    RiccatiKKTData(
        int64_t n_, int64_t m_, int64_t batchSize_,
        const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP_,
        const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA_,
        const Cones& cones,
        const std::vector<int32_t>& block_sizes,
        const std::vector<int32_t>& perm = {},
        cudaStream_t stream = 0);

    RiccatiKKTData(const RiccatiKKTData&) = delete;
    RiccatiKKTData& operator=(const RiccatiKKTData&) = delete;
    RiccatiKKTData(RiccatiKKTData&&) noexcept = default;
    RiccatiKKTData& operator=(RiccatiKKTData&&) noexcept = default;
    ~RiccatiKKTData() {
        if (cusolver_handle) cusolverDnDestroy(cusolver_handle);
    }

    void populate(CSR& P, CSR& A, cudaStream_t stream = 0) override;
    void update_H(const Cones& cones, const double* mu_data = nullptr, cudaStream_t stream = 0) override;

    bool update(
        Cones& cones,
        const BatchedVector& s, const BatchedVector& z, const BatchedVector& μ,
        ScalingStrategy scaling,
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        const BatchedVector& q, const BatchedVector& b,
        BatchedVector& workx, BatchedVector& const_rhs, BatchedVector& const_sol,
        BatchedVector& x2, BatchedVector& z2,
        cudaStream_t stream = 0) override;

    bool update(
        Cones& cones,
        const BatchedVector& s, const BatchedVector& z, const BatchedVector& μ,
        ScalingStrategy scaling,
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        cudaStream_t stream = 0) override;

    bool updateFactorOnly(
        Cones& cones,
        const BatchedVector& s, const BatchedVector& z, const BatchedVector& μ,
        ScalingStrategy scaling,
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        const BatchedVector& q, const BatchedVector& b,
        BatchedVector& workx, BatchedVector& const_rhs, BatchedVector& const_sol,
        BatchedVector& x2, BatchedVector& z2,
        cudaStream_t stream = 0) override;

    void solve(const double* rhs, double* sol, cudaStream_t stream = 0) override;
    void solve2(const double* rhs, double* sol, cudaStream_t stream = 0) override;

    /** Solve both the affine and constant RHS in a single pass over S/L.
     *  Must be called after updateFactorOnly() which defers the constant solve.
     *  affine_rhs/sol: interleaved [n+m] affine RHS/sol
     *  const_x/z: output buffers for constant solution [batchSize*n], [batchSize*m]
     */
    void solve_combined(
        const double* affine_rhs, double* affine_sol,
        double* const_x, double* const_z,
        cudaStream_t stream = 0) override;

    bool regularize_and_refactor(
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        cudaStream_t stream = 0) override;

    [[nodiscard]] size_t memoryUsage() const noexcept override;
    [[nodiscard]] KKTSolverType solverType() const noexcept override { return KKTSolverType::Riccati; }

    void setCublasHandle(cublasHandle_t handle) override { cublas_handle = handle; }

    // Made accessible for DiffKKT backward pass (needs to call with custom h_inv)
    bool assemble_and_factorize(cudaStream_t stream);

private:
    void solve_schur(
        const double* rx, const double* rz,
        double* sol_x, double* sol_z,
        cudaStream_t stream);

    // Helper: set up batch pointers for a specific block
    void setup_ptrs_for_block(
        double** d_ptrs, double* base, int64_t stride, int64_t offset,
        int64_t batchSize, cudaStream_t stream);
};

} // namespace moreau
