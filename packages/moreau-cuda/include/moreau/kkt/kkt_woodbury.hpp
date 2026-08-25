/**
 * @file kkt_woodbury.hpp
 * @brief Woodbury KKT solver for diagonal P + low-rank/sparse A problems
 *
 * Exploits the Woodbury matrix identity when:
 * - P is diagonal (n×n)
 * - A has k zero-cone rows and n_nonneg nonneg rows
 * - Nonneg rows classified as sparse (≤1 nnz) or dense (>1 nnz)
 * - Sparse nonneg: A_s^T H_s A_s is diagonal, folded into D
 * - Dense nonneg: folded into F_all alongside zero-cone rows
 *
 * Reduces the (n+m)×(n+m) KKT system to a k_total×k_total Schur complement:
 *   D = P + εI + A_s^T Γ_s A_s   (diagonal)
 *   F_all = [A_zero; A_d]         (k_total×n, k_total = k + k_d)
 *   S = C_all^{-1} + F_all^T D^{-1} F_all  (k_total×k_total, Cholesky factored)
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "moreau/cuda/utils.hpp"
#include "moreau/matrix/csr.hpp"
#include "moreau/vector/vector.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/kkt/kkt_solver.hpp"

namespace moreau {

// Forward declarations
enum class ScalingStrategy;

struct WoodburyKKTData : public KKTSolver {
    int64_t n_ = 0;         // primal dimension
    int64_t m_ = 0;         // dual dimension (k + n_nonneg)
    int64_t k_ = 0;         // number of zero cones (= rank of A_zero)
    int64_t n_nonneg_ = 0;  // number of nonneg cones
    int64_t batchSize_ = 0;

    // ========== Dense nonneg rows (Phase 2) ==========
    // Dense nonneg rows (>1 nnz) get folded into F_all, expanding the Schur complement
    int64_t k_d_ = 0;       // number of dense nonneg rows
    int64_t k_total_ = 0;   // k_ + k_d_ (Schur complement size)

    std::vector<int64_t> dense_nonneg_idx_;         // host (k_d_): maps dense index -> nonneg index
    device_unique_ptr<int64_t> d_dense_nonneg_idx_; // device (k_d_)

    // ========== Sparse nonneg row structure (set at construction) ==========
    // Each sparse nonneg row has ≤1 nonzero. A_s^T H_s A_s is diagonal.
    int64_t n_sparse_ = 0;  // number of nonneg rows with exactly 1 nonzero

    // Per sparse row: which nonneg index and which column
    std::vector<int64_t> sparse_nonneg_idx_;  // host (n_sparse_): nonneg row index
    std::vector<int64_t> sparse_col_;         // host (n_sparse_): column index

    // CSC of A_s: for each column j, col_offsets_[j]..col_offsets_[j+1]
    // index into col_sparse_rows_ giving the sparse row indices touching col j
    std::vector<int64_t> col_offsets_;        // host (n+1)
    std::vector<int64_t> col_sparse_rows_;    // host (n_sparse_)

    // Device copies
    device_unique_ptr<int64_t> d_sparse_nonneg_idx_;  // (n_sparse_)
    device_unique_ptr<int64_t> d_sparse_col_;         // (n_sparse_)
    device_unique_ptr<int64_t> d_col_offsets_;         // (n+1)
    device_unique_ptr<int64_t> d_col_sparse_rows_;     // (n_sparse_)

    // Empty nonneg rows (0 nonzeros): dlam = -h_inv * rhs_lam
    int64_t n_empty_ = 0;
    std::vector<int64_t> empty_nonneg_idx_;   // host (n_empty_)
    device_unique_ptr<int64_t> d_empty_nonneg_idx_;  // (n_empty_)

    // ========== Per-batch (precomputed at populate) ==========
    device_unique_ptr<double> F_all_;        // (B_F, n, k_total) row-major [A_zero^T; A_d^T]
                                             // B_F = 1 if shared, B if per-batch
    device_unique_ptr<double> P_diag_;       // (B, n) P diagonal per batch
    device_unique_ptr<double> A_s_vals_;     // (B_As, n_sparse_) A values for sparse nonneg rows
                                             // B_As = 1 if shared, B if per-batch
    bool f_all_shared_ = false;              // true if F_all is identical across batches
    long long f_all_stride_ = 0;             // cuBLAS stride for F_all: 0 if shared, n*k_total if per-batch
    bool a_s_shared_ = false;                // true if A_s_vals is identical across batches

    // ========== Direct-x nonneg cones (Phase 5: Woodbury+nonneg-direct-x) ==========
    // Per direct-x nonneg index, we add Hs_i = z_x/x[i] (NT scaling, slack
    // form after primal↔dual swap — see `update_xcones_nonneg_scaling_kernel`)
    // to the (1,1) block diagonal at column i. The per-iter Hs values live in
    // `cones.xcone_Hs[]`; the cone layout is collapsed into:
    //   xn_idx_[k] = primal x index for the k-th direct-x nonneg slot
    //   xn_hs_off_[k] = offset into cones.xcone_Hs for the k-th slot
    // disjointness across x_cones is enforced upstream (Cones::initialize).
    int64_t n_xcone_nonneg_ = 0;
    std::vector<int64_t> xn_idx_host_;       // host (n_xcone_nonneg_)
    std::vector<int64_t> xn_hs_off_host_;    // host (n_xcone_nonneg_)
    device_unique_ptr<int64_t> d_xn_idx_;    // device (n_xcone_nonneg_)
    device_unique_ptr<int64_t> d_xn_hs_off_; // device (n_xcone_nonneg_)
    device_unique_ptr<double> xn_hs_;        // (B, n_xcone_nonneg_) per-iter Hs gathered
    // Reverse lookup x_index -> xn slot (or -1 if x[j] is unconstrained by
    // any direct-x nonneg cone). Length n_, built once at construction.
    device_unique_ptr<int64_t> d_xn_x_to_slot_;

    // ========== Per-batch (updated each iteration) ==========
    device_unique_ptr<double> H_nonneg_;         // (B, n_nonneg) w^2 values (all nonneg rows)
    device_unique_ptr<double> d_tilde_inv_;      // (B, n) 1/D[j]
    device_unique_ptr<double> h_tilde_inv_;      // (B, n_nonneg) 1/(w^2 + eps) for all nonneg

    // Schur complement via F_all^T @ diag(d_tilde_inv) @ F_all
    device_unique_ptr<double> F_scaled_;         // (B, n, k_total) workspace: d_tilde_inv * F_all
    device_unique_ptr<double> S_f64_;            // (B, k_total, k_total) Schur complement

    // Regularization
    device_unique_ptr<double> eps_;              // (B) per-batch regularization
    device_unique_ptr<double> C_all_;            // (B, k_total) combined C diagonal
                                                 //   [0..k): 1/eps  (zero-cone)
                                                 //   [k..k_total): 1/(w_d^2+eps) (dense nonneg)

    // Solve workspace — allocated at 2x for fused solve2 (nrhs=2)
    device_unique_ptr<double> rhs_x_;        // (2B, n)
    device_unique_ptr<double> rhs_nu_;       // (2B, k_total) — combined [rhs_y; rhs_lam_d]
    device_unique_ptr<double> rhs_lam_;      // (2B, n_nonneg) — full nonneg RHS (for sparse recovery)
    device_unique_ptr<double> rhs_x_adj_;    // (2B, n) adjusted RHS
    device_unique_ptr<double> Dinv_rhs_;     // (2B, n)
    device_unique_ptr<double> schur_rhs_;    // (2B, k_total) Schur RHS
    device_unique_ptr<double> d_nu_;         // (2B, k_total) Schur solution
    device_unique_ptr<double> Fdnu_;         // (2B, n) F_all @ d_nu
    device_unique_ptr<double> dx_;           // (2B, n) primal step
    device_unique_ptr<double> dy_nu_;        // (2B, k_total) combined [dy; dlam_d]
    device_unique_ptr<double> dlam_;         // (2B, n_nonneg) nonneg dual step

    // cuSOLVER batched Cholesky pointer arrays (used when B > 1, dense Schur)
    device_unique_ptr<double*> S_f64_ptrs_;
    device_unique_ptr<double*> dnu_f64_ptrs_;     // points into d_nu_[0..B*k_total)
    device_unique_ptr<double*> dnu_f64_ptrs2_;    // points into d_nu_[B*k_total..2B*k_total)
    device_unique_ptr<int> cusolver_info_;

    // cuSOLVER non-batched workspace (used when B == 1, dense Schur)
    device_unique_ptr<double> potrf_workspace_;
    int potrf_workspace_size_ = 0;

    // KKT-level iterative refinement workspace
    device_unique_ptr<double> refine_kkt_out_;   // (2B, N) where N = n + m

    // Handles
    cublasHandle_t cublas_ = nullptr;
    cusolverDnHandle_t cusolver_ = nullptr;
    bool handles_owned_ = false;

    bool populated_ = false;

    // ========== Constructor / Destructor ==========

    WoodburyKKTData(
        int64_t n, int64_t m, int64_t batchSize,
        const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
        const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
        const Cones& cones,
        cudaStream_t stream = 0
    );

    ~WoodburyKKTData();

    // No copy
    WoodburyKKTData(const WoodburyKKTData&) = delete;
    WoodburyKKTData& operator=(const WoodburyKKTData&) = delete;

    // Move ok
    WoodburyKKTData(WoodburyKKTData&&) noexcept = default;
    WoodburyKKTData& operator=(WoodburyKKTData&&) noexcept = default;

    // ========== Interface (matches KKTData) ==========

    void populate(CSR& P, CSR& A, cudaStream_t stream = 0) override;

    void update_H(const Cones& cones, const double* mu_data = nullptr, cudaStream_t stream = 0) override;

    bool regularize_and_refactor(
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        cudaStream_t stream = 0
    ) override;

    void solve(const double* rhs, double* sol, cudaStream_t stream = 0) override;

    void solve2(const double* rhs, double* sol, cudaStream_t stream = 0) override;

    // Compound update methods
    bool update(
        Cones& cones,
        const BatchedVector& s,
        const BatchedVector& z,
        const BatchedVector& mu,
        ScalingStrategy scaling,
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        const BatchedVector& q,
        const BatchedVector& b,
        BatchedVector& workx,
        BatchedVector& const_rhs,
        BatchedVector& const_sol,
        BatchedVector& x2,
        BatchedVector& z2,
        cudaStream_t stream = 0
    ) override;

    bool update(
        Cones& cones,
        const BatchedVector& s,
        const BatchedVector& z,
        const BatchedVector& mu,
        ScalingStrategy scaling,
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        cudaStream_t stream = 0
    ) override;

    bool updateFactorOnly(
        Cones& cones,
        const BatchedVector& s,
        const BatchedVector& z,
        const BatchedVector& mu,
        ScalingStrategy scaling,
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        const BatchedVector& q,
        const BatchedVector& b,
        BatchedVector& workx,
        BatchedVector& const_rhs,
        BatchedVector& const_sol,
        BatchedVector& x2,
        BatchedVector& z2,
        cudaStream_t stream = 0
    ) override;

    [[nodiscard]] size_t memoryUsage() const noexcept override;
    [[nodiscard]] KKTSolverType solverType() const noexcept override { return KKTSolverType::Woodbury; }

    // ========== Detection ==========

    /**
     * Check if problem structure is compatible with Woodbury solver.
     * Requirements:
     * - Only Zero + Nonneg cones (diagonal H)
     * - P is diagonal (nnzP ≤ n, each row has at most one diagonal entry)
     * - k_total = k + k_d < n (where k_d = dense nonneg rows with >1 nnz)
     * - Sparse nonneg rows have ≤1 nonzero (column-orthogonal A_s)
     * - Every column is "covered" by P (nonzero diagonal) or a sparse nonneg
     *   row. Uncovered columns have D[j] ≈ ε, making D^{-1} ≈ 1/ε which
     *   destroys Schur complement conditioning.
     */
    static bool isCompatible(
        int64_t n, int64_t m,
        const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
        const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
        const Cones& cones
    );

private:
    void build_ptr_arrays();

    // Core Woodbury solve (no refinement) for nrhs=1 or nrhs=2.
    // Writes result into sol. rhs and sol are (eB, N) where eB = nrhs * B.
    void solve_core(const double* rhs, double* sol, int nrhs, cudaStream_t stream);

    // KKT matvec: out = KKT * sol, using Woodbury structure.
    // sol is (eB, N), out is (eB, N). Uses refine_kkt_out_ as workspace.
    void kkt_matvec(const double* sol, double* out, int nrhs, cudaStream_t stream);

    // Core solve implementation: calls solve_core, then 1 step of KKT-level refinement.
    void solve_impl(const double* rhs, double* sol, int nrhs, cudaStream_t stream);
};

} // namespace moreau
