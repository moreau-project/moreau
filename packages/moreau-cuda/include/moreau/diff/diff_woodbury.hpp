/**
 * @file diff_woodbury.hpp
 * @brief Woodbury-specialized backward pass via direct Schur complement solve
 *
 * Solves (J'J + eI)w = rhs_bar, then y_out = J*w.
 * Exploits diagonal P + low-rank A structure via block elimination:
 *   1. Eliminate y_s (diagonal D_s = I + H^2 + eI)
 *   2. Solve reduced (n+m+1) system via two-level Woodbury:
 *      - Inner Woodbury: M'_{xx} = D_x + GG' where G=[sqrt(Λ)*F_all, c1]
 *      - Outer Schur complement on [ν, τ] variables (k_ext × k_ext)
 *   3. Back-substitute for y_s
 *   4. J-matvec: y_out = J * w
 *
 * Key difference from original: tau is handled as part of the Schur complement
 * (not via separate Sherman-Morrison), and the inner Woodbury correctly accounts
 * for the off-diagonal A'ΛA + c1c1' structure of M'_{xx}.
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cstdint>
#include <memory>

#include "moreau/cuda/utils.hpp"
#include "moreau/cones/cones.hpp"

namespace moreau {

struct WoodburyKKTData;

struct DiffWoodbury {
    int64_t n_ = 0, m_ = 0, k_ = 0, k_total_ = 0;
    int64_t k_ext_ = 0;   // k_total + 1 (extra slot for tau)
    int64_t n_nonneg_ = 0, n_sparse_ = 0, n_empty_ = 0, k_d_ = 0;
    int64_t jdim_ = 0;    // n + 2m + 1
    int64_t batchSize_ = 0;

    static constexpr double eps_ = 1e-8;

    // ---- Borrowed from forward WoodburyKKTData (set by setup()) ----
    const double* F_all_ = nullptr;
    long long f_all_stride_ = 0;
    bool f_all_shared_ = false;
    const double* P_diag_ = nullptr;
    const double* A_s_vals_ = nullptr;
    const int64_t* sparse_nonneg_idx_ = nullptr;
    const int64_t* sparse_col_ = nullptr;
    const int64_t* col_offsets_ = nullptr;
    const int64_t* col_sparse_rows_ = nullptr;
    const int64_t* dense_nonneg_idx_ = nullptr;
    const int64_t* empty_nonneg_idx_ = nullptr;

    // ---- Per-backward-call data (set by setup()) ----
    const double* H_nonneg_ = nullptr;  // (B, n_nonneg)
    const double* q_ = nullptr;         // (B, n)
    const double* b_ = nullptr;         // (B, m)
    // The operator-assembly + base-solve path sees ZEROED c1/c2/c3 (set to
    // zero_c*_ below) so it factors the base operator B = J_nogap'J_nogap + εI
    // (J without the gap row). The gap row's rank-1 g = [c1; c2; 0; c3] is then
    // applied as ONE global Woodbury correction (solveAdjoint), avoiding the
    // inconsistent per-block split that broke sparse-P gradients.
    const double* c1_ = nullptr;        // (B, n)  — zero during assembly/base solve
    const double* c2_ = nullptr;        // (B, m)  — zero during assembly/base solve
    const double* c3_ = nullptr;        // (B, 1)  — zero during assembly/base solve

    // ---- Real gap coefficients (used by J_matvec + the global gap Woodbury) ----
    const double* c1_real_ = nullptr;   // (B, n)
    const double* c2_real_ = nullptr;   // (B, m)
    const double* c3_real_ = nullptr;   // (B, 1)
    device_unique_ptr<double> zero_c1_; // (B, n) all-zero
    device_unique_ptr<double> zero_c2_; // (B, m) all-zero
    device_unique_ptr<double> zero_c3_; // (B, 1) all-zero

    // ---- Global gap-row Woodbury workspace ----
    device_unique_ptr<double> wg_x_, wg_z_, wg_s_, wg_tau_;  // cached wg = B^{-1} g
    device_unique_ptr<double> g_rhs_;                        // [c1; c2; 0; c3] (B, jdim)
    device_unique_ptr<double> gap_g_w0_;   // (B) g'·(base solve)
    device_unique_ptr<double> gap_g_wg_;   // (B) g'·wg
    device_unique_ptr<double> gap_coeff_;  // (B) (g'·rhs solve)/(1 + g'·wg)

    // ---- Conditioning-check workspace ----
    // After the gap-Woodbury solve we form M·w = J'(J·w) + εw (needs Jt_matvec,
    // since the chain only gives J·w) and report a relative residual. The normal
    // equations M=J'J+εI square the conditioning, so on near-singular instances
    // the residual is large — backward() then falls back to the κ-accurate cuDSS.
    device_unique_ptr<double> wacc_jdim_;  // (B, jdim) running w during refinement
    device_unique_ptr<double> jw_;         // (B, jdim) J·w / residual scratch
    device_unique_ptr<double> Mv_;         // (B, jdim) M·w
    device_unique_ptr<double> mres_;       // (B, jdim) residual rhs − M·w
    device_unique_ptr<double> resid_norm_; // (B) relative ‖M·w − rhs‖ after the solve

    // ---- s-elimination diagonals (B, m) ----
    device_unique_ptr<double> g_;       // 1/(1+h^2+eps)
    device_unique_ptr<double> Lambda_;  // (h^2+eps)/(1+h^2+eps)
    device_unique_ptr<double> gh1_;     // g*(1+h)

    // ---- Reduced system diagonals ----
    device_unique_ptr<double> D_x_;        // (B, n)
    device_unique_ptr<double> D_x_inv_;    // (B, n)
    device_unique_ptr<double> D_z_;        // (B, n_nonneg) — diagonal of M'[λ,λ] (without c2c2')
    device_unique_ptr<double> D_z_inv_;    // (B, n_nonneg) — 1/D_z (diagonal-only inverse)
    device_unique_ptr<double> rho_;        // (B) — 1 + c2_λ'D_z_inv c2_λ (Sherman-Morrison denom)

    // ---- tau coupling vectors (computed by tau_elim_diag kernel) ----
    device_unique_ptr<double> v_x_;     // (B, n) — M'_{xτ}
    device_unique_ptr<double> v_z_;     // (B, m) — M'_{zτ} (all rows, used for v_nu gather)
    device_unique_ptr<double> sigma_;   // (B)    — M'_{ττ}

    // ---- F_ext: coupling x → [ν, τ] (B, n, k_ext) ----
    // F_ext[:,i] = (P+1-gh1[i])*A[i,:] + c1*c2[i]  for i < k_total
    // F_ext[:,k_total] = v_x  (tau coupling)
    device_unique_ptr<double> F_ext_;
    long long f_ext_stride_ = 0;

    // ---- Inner Woodbury: M'_{xx} = D_x + G*G' ----
    device_unique_ptr<double> G_;           // (B, n, k_ext): [sqrt(Λ)*F_all, c1]
    device_unique_ptr<double> G_scaled_;    // (B, n, k_ext): D_x_inv * G (workspace)
    device_unique_ptr<double> C_inner_;     // (B, k_ext, k_ext): I + G'D^{-1}G (LDL^T)

    // ---- Outer Schur complement ----
    device_unique_ptr<double> B_full_;      // (B, k_ext, k_ext): full [M'νν, vν; vν', σ]
    device_unique_ptr<double> S_schur_;     // (B, k_ext, k_ext): T = B - F'K_hat (LDL^T)
    device_unique_ptr<double> S_schur_orig_; // (B, k_ext, k_ext): unfactored copy for refinement

    // ---- Precomputed M'_{xx}^{-1} * F_ext = K_hat (B, n, k_ext) ----
    device_unique_ptr<double> K_hat_;

    // ---- v_nu: tau coupling at zero/dense positions + sigma (B, k_ext) ----
    device_unique_ptr<double> v_nu_;

    // ---- Workspace for GEMM intermediates ----
    device_unique_ptr<double> temp_kk_;     // (B, k_ext, k_ext)
    device_unique_ptr<double> mx_scratch_;  // (B, n) — scratch for apply_Mx_inv

    // ---- Reduced RHS ----
    device_unique_ptr<double> rt_x_;    // (B, n)
    device_unique_ptr<double> rt_z_;    // (B, m)
    device_unique_ptr<double> rt_tau_;  // (B)

    // ---- Woodbury solve workspace ----
    device_unique_ptr<double> rhs_x_;      // (B, n)
    device_unique_ptr<double> rhs_nu_;     // (B, k_ext)
    device_unique_ptr<double> rhs_lam_;    // (B, n_nonneg)
    device_unique_ptr<double> rhs_x_adj_;  // (B, n)
    device_unique_ptr<double> u_x_;        // (B, n) — M'_{xx}^{-1} * rhs_x_adj
    device_unique_ptr<double> d_nu_;       // (B, k_ext)
    device_unique_ptr<double> schur_resid_; // (B, k_ext) — refinement residual

    // ---- Solution vectors ----
    device_unique_ptr<double> y_x_;     // (B, n)
    device_unique_ptr<double> y_z_;     // (B, m)
    device_unique_ptr<double> y_s_;     // (B, m)
    device_unique_ptr<double> y_tau_;   // (B)
    device_unique_ptr<double> dlam_nonneg_; // (B, n_nonneg) temp for nonneg y_z recovery

    // ---- GEMV workspace ----
    device_unique_ptr<double> gemv_k_;  // (B, k_ext)
    device_unique_ptr<double> gemv_m_;  // (B, m)

    cublasHandle_t cublas_ = nullptr;       // borrowed, not owned
    cusolverDnHandle_t cusolver_ = nullptr; // borrowed, not owned

    DiffWoodbury(int64_t n, int64_t m, int64_t batchSize,
                 const Cones& cones,
                 cublasHandle_t cublas, cusolverDnHandle_t cusolver,
                 cudaStream_t stream = 0);
    ~DiffWoodbury() = default;

    DiffWoodbury(const DiffWoodbury&) = delete;
    DiffWoodbury& operator=(const DiffWoodbury&) = delete;

    void setup(const WoodburyKKTData& wb,
               const double* H_nonneg,
               const double* q, const double* b,
               const double* c1, const double* c2, const double* c3,
               cudaStream_t stream);

    void solveAdjoint(const double* rhs_bar, double* y_out, cudaStream_t stream);

    // Per-batch relative ‖M·w − rhs‖ from the last solveAdjoint. Large values
    // flag instances where the normal-equation Woodbury chain lost accuracy
    // (κ² conditioning); callers fall back to the κ-accurate cuDSS path.
    [[nodiscard]] const double* residualNorms() const noexcept { return resid_norm_.get(); }

    [[nodiscard]] size_t memoryUsage() const noexcept;

private:
    void precompute(cudaStream_t stream);
    void eliminate_s_rhs(const double* rhs_bar, cudaStream_t stream);
    void solve_reduced(cudaStream_t stream);
    void back_substitute(const double* rhs_bar, cudaStream_t stream);
    // Solve B w = rhs_bar (base operator, gap excluded); result in y_x_/y_z_/y_s_/y_tau_.
    void base_solve(const double* rhs_bar, cudaStream_t stream);
    // Apply M^{-1} = (B + gg')^{-1} to rhs_bar via the cached gap Woodbury;
    // result in y_x_/y_z_/y_s_/y_tau_. (precompute() + the wg cache must be set.)
    void apply_Minv(const double* rhs_bar, cudaStream_t stream);
    // out (B,jdim, variable layout) = J'·u, u in row layout (B,jdim).
    void Jt_matvec(const double* u, double* out, cudaStream_t stream);
    void J_matvec(double* y_out, cudaStream_t stream);
    void build_F_ext(cudaStream_t stream);
    void build_G(cudaStream_t stream);
    void build_inner_woodbury(cudaStream_t stream);
    void build_B_full(cudaStream_t stream);
    void build_and_factor_outer_schur(cudaStream_t stream);
    void apply_Mx_inv(const double* rhs, double* out, cudaStream_t stream);
};

} // namespace moreau
