/**
 * @file woodbury_kernels.cuh
 * @brief CUDA kernel declarations for the Woodbury KKT solver backend
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace moreau {

enum class DiffWbLdltVariant : int {
    Auto = 0,
    Scalar = 1,
    Warp = 2,
    Block = 3,
};

void set_diff_wb_ldlt_variant(DiffWbLdltVariant variant);
DiffWbLdltVariant get_diff_wb_ldlt_variant();

// dx recovery: dx = d_tilde_inv * (rhs_x_adj - Fdnu)
void launch_woodbury_dx_recovery(
    double* dx, const double* d_tilde_inv,
    const double* rhs_x_adj, const double* Fdnu,
    int64_t count, cudaStream_t stream = 0);

// Elementwise multiply: out = a * b
void launch_woodbury_elementwise_mul(
    double* out, const double* a, const double* b,
    int64_t count, cudaStream_t stream = 0);

// Compute H_nonneg = w^2 from nonneg cone w values
void launch_woodbury_compute_H_nonneg(
    double* H_nonneg, const double* nonneg_w,
    int64_t count, cudaStream_t stream = 0);

// Compute C_diag = 1/eps for zero-cone regularization diagonal
void launch_woodbury_compute_C_diag(
    double* C_diag, const double* eps,
    int64_t batch, int64_t k, cudaStream_t stream = 0);

// Compute eps = eps_c + eps_p * max_diag (per batch)
void launch_woodbury_compute_eps(
    double* eps, double eps_c, double eps_p,
    int64_t batch, cudaStream_t stream = 0);

// F_scaled[b,i,j] = d_tilde_inv[b,i] * F[f_idx] — scale F by per-batch diagonal
// f_shared: if true, F is (n,k) shared; if false, F is (B,n,k) per-batch
void launch_woodbury_scale_F(
    double* F_scaled, const double* F, const double* d_tilde_inv,
    int64_t batch, int64_t n, int64_t k, bool f_shared, cudaStream_t stream = 0);

// Set S = eps*I (batched): S[b,i,j] = (i==j) ? eps[b] : 0
void launch_woodbury_set_identity_eps(
    double* S, const double* eps, int64_t batch, int64_t k, cudaStream_t stream = 0);

// Pack solution: interleave [dx; dy; dlam] into sol(n+m)
// dy is read with stride k (contiguous)
void launch_woodbury_pack_sol(
    double* sol, const double* dx, const double* dy, const double* dlam,
    int64_t batch, int64_t n, int64_t k, int64_t n_nonneg, cudaStream_t stream = 0);

// Pack solution with different stride for dy_nu: reads k entries with stride k_total
// sol[b*N + n + i] = dy_nu[b * k_total + i] for i < k
void launch_woodbury_pack_sol_kt(
    double* sol, const double* dx, const double* dy_nu, const double* dlam,
    int64_t batch, int64_t n, int64_t k, int64_t k_total, int64_t n_nonneg,
    cudaStream_t stream = 0);

// Unpack RHS: split rhs(n+m) into rhs_x(n), rhs_y(k), rhs_lam(n_nonneg)
// rhs_y is written with stride k (contiguous)
void launch_woodbury_unpack_rhs(
    const double* rhs, double* rhs_x, double* rhs_y, double* rhs_lam,
    int64_t batch, int64_t n, int64_t k, int64_t n_nonneg, cudaStream_t stream = 0);

// Unpack RHS with different stride for rhs_nu: writes k entries with stride k_total
// rhs_nu[b * k_total + i] = rhs[b * N + n + i] for i < k
void launch_woodbury_unpack_rhs_kt(
    const double* rhs, double* rhs_x, double* rhs_nu, double* rhs_lam,
    int64_t batch, int64_t n, int64_t k, int64_t k_total, int64_t n_nonneg,
    cudaStream_t stream = 0);

// dy recovery: dy = d_nu - C_diag * rhs_y (per-batch C_diag broadcast for eB)
// C_diag is (B, k), d_nu and rhs_y are (eB, k)
void launch_woodbury_dy_recovery(
    double* dy, const double* d_nu, const double* C_diag,
    const double* rhs_y, int64_t batch, int64_t k, cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Generalized sparse nonneg kernels (Phase 1)
// ---------------------------------------------------------------------------

// Compute d_tilde_inv and h_tilde_inv using CSC structure of A_s.
// D[b,j] = P_diag[b,j] + eps[b] + sum_{sparse rows r touching col j} a_r^2 / (w_r^2 + eps)
// d_tilde_inv[b,j] = 1/D[b,j]
// h_tilde_inv[b,i] = 1/(H_nonneg[b,i] + eps[b]) for ALL nonneg rows
void launch_woodbury_d_tilde_inv_sparse(
    double* d_tilde_inv,            // (B, n) output
    double* h_tilde_inv,            // (B, n_nonneg) output
    const double* P_diag,           // (B, n)
    const double* H_nonneg,         // (B, n_nonneg)
    const double* A_s_vals,         // (B, n_sparse)
    const double* eps,              // (B)
    const int64_t* sparse_nonneg_idx, // (n_sparse) -> nonneg index
    const int64_t* col_offsets,     // (n+1) CSC of A_s
    const int64_t* col_sparse_rows, // (n_sparse) sparse row indices per column
    const double* xn_hs,            // (B, n_xn) direct-x nonneg Hs (or nullptr)
    const int64_t* xn_x_to_slot,    // (n) lookup x->xn slot or -1 (or nullptr)
    int64_t n_xn,
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// Gather direct-x nonneg Hs values from cones.xcone_Hs into a contiguous
// per-batch buffer indexed by the cone-flat slot. Used by Woodbury to feed
// the (1,1)-block diagonal contribution into d_tilde_inv each iteration.
void launch_woodbury_gather_xcone_nonneg_hs(
    double* xn_hs,                  // (B, n_xn) output
    const double* xcone_Hs,         // (B, totalXConeHsEntries)
    const int64_t* xn_hs_off,       // (n_xn) per-slot offset into xcone_Hs
    int64_t totalXConeHsEntries,
    int64_t batch, int64_t n_xn,
    cudaStream_t stream = 0);

// RHS adjustment with sparse nonneg rows:
// rhs_x_adj[b,j] = rhs_x[b,j] + sum_{sparse r touching col j} a_r * h_inv[nonneg_r] * rhs_lam[nonneg_r]
void launch_woodbury_rhs_adj_sparse(
    double* rhs_x_adj,             // (B, n) output
    const double* rhs_x,           // (B, n)
    const double* h_tilde_inv,     // (B, n_nonneg)
    const double* rhs_lam,         // (B, n_nonneg)
    const double* A_s_vals,        // (B, n_sparse)
    const int64_t* sparse_nonneg_idx, // (n_sparse) -> nonneg index
    const int64_t* col_offsets,    // (n+1)
    const int64_t* col_sparse_rows, // (n_sparse)
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// dlam recovery for sparse nonneg rows:
// dlam[b, nonneg_idx] = h_tilde_inv[b, nonneg_idx] * (a_r * dx[b, col_r] - rhs_lam[b, nonneg_idx])
void launch_woodbury_dlam_sparse(
    double* dlam,                   // (B, n_nonneg) — write at nonneg indices
    const double* h_tilde_inv,      // (B, n_nonneg)
    const double* dx,               // (B, n)
    const double* rhs_lam,          // (B, n_nonneg)
    const double* A_s_vals,         // (B, n_sparse)
    const int64_t* sparse_col,     // (n_sparse) column index per sparse row
    const int64_t* sparse_nonneg_idx, // (n_sparse) -> nonneg index
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// dlam for empty nonneg rows (0 nonzeros):
// dlam[b, nonneg_idx] = -h_tilde_inv[b, nonneg_idx] * rhs_lam[b, nonneg_idx]
void launch_woodbury_dlam_empty(
    double* dlam,                   // (B, n_nonneg) — write at nonneg indices
    const double* h_tilde_inv,      // (B, n_nonneg)
    const double* rhs_lam,          // (B, n_nonneg)
    const int64_t* empty_nonneg_idx, // (n_empty) -> nonneg index
    int64_t batch, int64_t n_nonneg, int64_t n_empty,
    cudaStream_t stream = 0);

// Max diag with sparse nonneg contributions
void launch_woodbury_max_diag_sparse(
    double* max_diag,               // (B) output
    const double* P_diag,           // (B, n)
    const double* H_nonneg,         // (B, n_nonneg)
    const double* A_s_vals,         // (B, n_sparse)
    const int64_t* sparse_nonneg_idx, // (n_sparse) -> nonneg index
    const int64_t* col_offsets,     // (n+1)
    const int64_t* col_sparse_rows, // (n_sparse)
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// Dense nonneg kernels (Phase 2)
// ---------------------------------------------------------------------------

// Compute C_all diagonal: combined zero-cone + dense nonneg regularization
// C_all[b, i] = 1/eps[b]                                   for i < k
// C_all[b, k+d] = 1/(H_nonneg[b, dense_nonneg_idx[d]] + eps[b])  for d < k_d
void launch_woodbury_compute_C_all(
    double* C_all,                    // (B, k_total) output
    const double* eps,                // (B)
    const double* H_nonneg,           // (B, n_nonneg)
    const int64_t* dense_nonneg_idx,  // (k_d) -> nonneg index
    int64_t batch, int64_t k, int64_t k_d, int64_t k_total, int64_t n_nonneg,
    cudaStream_t stream = 0);

// Add C_all^{-1} to S diagonal (replaces set_identity_eps for k_total case):
// S[b, i, i] += eps[b]                                      for i < k
// S[b, k+d, k+d] += H_nonneg[b, dense_nonneg_idx[d]] + eps[b]  for d < k_d
void launch_woodbury_set_C_all_inv_diagonal(
    double* S,                        // (B, k_total, k_total) in-place
    const double* eps,                // (B)
    const double* H_nonneg,           // (B, n_nonneg)
    const int64_t* dense_nonneg_idx,  // (k_d) -> nonneg index
    int64_t batch, int64_t k, int64_t k_d, int64_t k_total, int64_t n_nonneg,
    cudaStream_t stream = 0);

// Gather dense nonneg RHS into combined Schur RHS:
// rhs_nu[b, k+d] = rhs_lam[b, dense_nonneg_idx[d]]
void launch_woodbury_gather_dense_rhs(
    double* rhs_nu,                   // (eB, k_total) — write at [k..k_total]
    const double* rhs_lam,            // (eB, n_nonneg)
    const int64_t* dense_nonneg_idx,  // (k_d) -> nonneg index
    int64_t eB, int64_t k, int64_t k_total, int64_t n_nonneg,
    cudaStream_t stream = 0);

// Scatter dense nonneg duals from Schur solution into dlam:
// dlam[b, dense_nonneg_idx[d]] = dy_nu[b, k+d]
void launch_woodbury_scatter_dlam_dense(
    double* dlam,                     // (eB, n_nonneg) — write at dense positions
    const double* dy_nu,              // (eB, k_total) — combined [dy; dlam_d]
    int64_t eB, int64_t k, int64_t k_total, int64_t n_nonneg,
    const int64_t* dense_nonneg_idx,  // (k_d) -> nonneg index
    int64_t k_d,
    cudaStream_t stream = 0);

// ---------------------------------------------------------------------------
// KKT matvec kernels (for iterative refinement)
// ---------------------------------------------------------------------------

// out_x = (P_diag + eps) * dx, elementwise (B, n)
void launch_woodbury_kkt_matvec_x(
    double* out_x,                    // (B, n) output
    const double* dx,                 // (B, n)
    const double* P_diag,             // (B, n)
    const double* eps,                // (B) per-batch regularization
    int64_t batch, int64_t n,
    cudaStream_t stream = 0);

// Scatter-accumulate A_s^T * dlam_s into out_x using CSC structure:
// out_x[b, col_j] += sum_{sparse r in col j} a_r * dlam[b, nonneg_r]
void launch_woodbury_kkt_As_T_dlam(
    double* out_x,                    // (B, n) in-place accumulate
    const double* dlam,               // (B, n_nonneg)
    const double* A_s_vals,           // (B, n_sparse)
    const int64_t* sparse_nonneg_idx, // (n_sparse) -> nonneg index
    const int64_t* col_offsets,       // (n+1) CSC of A_s
    const int64_t* col_sparse_rows,   // (n_sparse)
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// out_lam_s[r] = a_r * dx[col_r] - (w_r^2 + eps) * dlam_r for sparse nonneg rows
void launch_woodbury_kkt_matvec_sparse_z(
    double* out,                      // (B, n_nonneg) — write at sparse nonneg indices
    const double* dx,                 // (B, n)
    const double* dlam,               // (B, n_nonneg)
    const double* H_nonneg,           // (B, n_nonneg)
    const double* eps,                // (B)
    const double* A_s_vals,           // (B, n_sparse)
    const int64_t* sparse_col,        // (n_sparse) column index
    const int64_t* sparse_nonneg_idx, // (n_sparse) -> nonneg index
    int64_t batch, int64_t n, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// out_lam_e[r] = -(w_r^2 + eps) * dlam_e for empty nonneg rows
void launch_woodbury_kkt_matvec_empty_z(
    double* out,                      // (B, n_nonneg) — write at empty nonneg indices
    const double* dlam,               // (B, n_nonneg)
    const double* H_nonneg,           // (B, n_nonneg)
    const double* eps,                // (B)
    const int64_t* empty_nonneg_idx,  // (n_empty) -> nonneg index
    int64_t batch, int64_t n_nonneg, int64_t n_empty,
    cudaStream_t stream = 0);

// Subtract C_all_inv * nu from out_nu (for KKT matvec nu component):
// out[b,i] -= (1/C_all[b,i]) * nu[b,i]
// For i < k:  1/C_all = eps[b]
// For i >= k: 1/C_all = H_nonneg[b, dense_nonneg_idx[i-k]] + eps[b]
// eB may exceed B (nrhs > 1); data arrays (eps, H_nonneg, dense_nonneg_idx) use b % B.
void launch_woodbury_kkt_matvec_nu_sub(
    double* out,                      // (eB, k_total) in-place subtract
    const double* nu,                 // (eB, k_total) the [dy; dlam_d] vector
    const double* eps,                // (B) per-batch regularization
    const double* H_nonneg,           // (B, n_nonneg)
    const int64_t* dense_nonneg_idx,  // (k_d) -> nonneg index (may be null if k_d==0)
    int64_t batch, int64_t eB,
    int64_t k, int64_t k_d, int64_t k_total, int64_t n_nonneg,
    cudaStream_t stream = 0);

// Compute residual: r = rhs - y
void launch_woodbury_refine_residual(
    double* r, const double* rhs, const double* y,
    int64_t count, cudaStream_t stream = 0);

// x += delta
void launch_woodbury_axpy(
    double* x, const double* delta,
    int64_t count, cudaStream_t stream = 0);

// ---- Fused single-kernel Woodbury solve (B=1 only) ----
// Performs the entire solve_core in a SINGLE kernel launch.
// Eliminates cuBLAS GEMV overhead (~5μs per call × 6 calls = 30μs saved per iteration).
// F_all is (n, k_total) row-major. S_chol is pre-factored (k_total, k_total) lower Cholesky.
// nrhs = 1 or 2. sol/rhs are (nrhs, N) where N = n + m.
void launch_woodbury_fused_solve(
    double* sol, const double* rhs,
    const double* F_all, const double* d_tilde_inv,
    const double* h_tilde_inv, const double* C_all,
    const double* S_chol,  // pre-factored Cholesky of Schur complement
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    const int64_t* dense_nonneg_idx, const int64_t* empty_nonneg_idx,
    int64_t n, int64_t m, int64_t k, int64_t k_d, int64_t k_total,
    int64_t n_nonneg, int64_t n_sparse, int64_t n_empty,
    int nrhs, cudaStream_t stream = 0);

// ============================================================
// Backward pass kernels (DiffWoodbury direct Schur complement)
// ============================================================

// Elementwise inverse: out[i] = 1.0 / in[i]
void launch_woodbury_elementwise_inv(
    double* out, const double* in, int64_t N, cudaStream_t stream = 0);

// Compute s-elimination diagonals from H values.
// h_i = 1.0 for zero cones (i < k), H_nonneg[i-k] for nonneg cones
// g[i] = 1/(1+h²+eps), Lambda[i] = 1-g[i], gh1[i] = g[i]*(1+h)
void launch_diff_wb_s_elim_diagonals(
    double* g, double* Lambda, double* gh1,
    const double* H_nonneg, double eps,
    int64_t batch, int64_t m, int64_t k, int64_t n_nonneg,
    cudaStream_t stream = 0);

// D_x with full cross-term correction from sparse z-elimination
void launch_diff_wb_compute_D_x(
    double* D_x,
    const double* P_diag, const double* Lambda,
    const double* A_s_vals, double eps,
    const double* H_nonneg, const double* c1, const double* c2,
    const double* D_z_inv, const double* rho,
    const int64_t* sparse_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// D_z[i] = a_i^2 + 2+eps - (1+h)^2/(1+h^2+eps) + c2_nonneg[i]^2
void launch_diff_wb_compute_D_z(
    double* D_z, double* D_z_inv,
    const double* H_nonneg, const double* c2,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx,
    double eps,
    int64_t batch, int64_t m, int64_t k, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// Build F_ext = [F_all; c1'] with k_ext = k_total + 1
void launch_diff_wb_build_F_ext(
    double* F_ext, const double* F_all, const double* c1,
    int64_t B_F, int64_t n, int64_t k_total, int64_t k_ext,
    cudaStream_t stream = 0);

// Build F_ext with backward cross-coupling
void launch_diff_wb_build_F_ext_backward(
    double* F_ext, const double* F_all, long long f_all_stride,
    const double* P_diag, const double* gh1, const double* c1, const double* c2,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext, cudaStream_t stream = 0);

// Compute tau-elimination diagonal quantities (v_x, v_z, sigma)
void launch_diff_wb_tau_elim_diag(
    double* v_x, double* v_z, double* sigma,
    const double* P_diag, const double* q, const double* b,
    const double* c1, const double* c2, const double* c3,
    const double* Lambda, const double* g, const double* gh1,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    double eps,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// RHS correction after s-elimination (sparse parts + diagonal)
void launch_diff_wb_s_elim_rhs(
    const double* rhs_bar, double* rt_x, double* rt_z, double* rt_tau,
    const double* g, const double* gh1, const double* b,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// Recover y_s from back-substitution (diagonal + sparse parts)
void launch_diff_wb_recover_y_s(
    double* y_s,
    const double* rhs_bar, const double* y_x,
    const double* y_z, const double* y_tau,
    const double* g, const double* gh1, const double* b,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// J-matvec diagonal/sparse parts: y_out = J * w
void launch_diff_wb_J_matvec_diag(
    double* y_out,
    const double* w_x, const double* w_z, const double* w_s,
    const double* w_tau,
    const double* P_diag, const double* H_nonneg,
    const double* q, const double* b,
    const double* c1, const double* c2, const double* c3,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// Gather from m-vector at zero/dense positions into k_total-vector with optional weight
void launch_diff_wb_gather_weighted(
    double* out, const double* in, const double* weight,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total,
    cudaStream_t stream = 0);

// Scatter-add from k_total-vector into m-vector at zero/dense positions with optional weight
void launch_diff_wb_scatter_add_weighted(
    double* out, const double* in, const double* weight,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total,
    cudaStream_t stream = 0);

// Backward RHS adjustment for sparse nonneg elimination.
// rhs_x_adj[j] = rhs_x[j] + sum_{sparse r in col j} cross_xz[j,r] * D_z_inv[r] * rhs_lam[r]
// where cross_xz[j,r] = P[j]*a + a*G[r] + c1[j]*c2[k+r]
// G[r] = (h^2-h+eps)/(1+h^2+eps)
void launch_diff_wb_rhs_adj_sparse(
    double* rhs_x_adj, const double* rhs_x,
    const double* D_z_inv, const double* rhs_lam,
    const double* P_diag, const double* A_s_vals,
    const double* H_nonneg, const double* c1, const double* c2,
    double eps,
    const int64_t* sparse_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    const int64_t* sparse_col,
    int64_t batch, int64_t n, int64_t m, int64_t k,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// Backward dlam recovery for sparse nonneg:
// y_z[ni] = D_z_inv[ni] * (cross_xz_T[ni] * y_x + S_xz_correction - rhs_lam[ni])
// where the full formula accounts for the cross-term
void launch_diff_wb_dlam_sparse(
    double* dlam, const double* D_z_inv,
    const double* y_x, const double* y_nu, const double* y_tau,
    const double* rhs_lam,
    const double* P_diag, const double* A_s_vals,
    const double* H_nonneg, const double* c1, const double* c2,
    const double* v_z,
    const double* F_all, long long f_all_stride,
    const int64_t* dense_nonneg_idx,
    double eps,
    const int64_t* sparse_col, const int64_t* sparse_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// Unpack rhat_z (B,m) into rhs_nu (B,k_ext), rhs_lam (B,n_nonneg)
// rhs_nu[0:k] = rhat_z[0:k], rhs_nu[k:k_total] = rhat_z[k+dense_idx],
// rhs_nu[k_total] = 0, rhs_lam[i] = rhat_z[k+i]
void launch_diff_wb_unpack_z(
    double* rhs_nu, double* rhs_lam,
    const double* rhat_z,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total, int64_t k_ext,
    int64_t n_nonneg,
    cudaStream_t stream = 0);

// Scatter zero-cone y_z from gemv_k into y_z (B,m)
// y_z[b*m + i] = gemv_k[b*k_ext + i] for i < k
void launch_diff_wb_scatter_y_z_zero(
    double* y_z, const double* gemv_k,
    int64_t batch, int64_t m, int64_t k, int64_t k_ext,
    cudaStream_t stream = 0);

// Strided axpy: dst[b*dst_stride + dst_offset + i] += src[b*src_stride + i] for i < count
void launch_diff_wb_strided_axpy(
    double* dst, const double* src,
    int64_t batch, int64_t count, int64_t dst_stride, int64_t dst_offset, int64_t src_stride,
    cudaStream_t stream = 0);

// Extract from strided source, multiply element-wise, store contiguous:
// out[b*m + i] = a[b*m + i] * src[b*src_stride + src_offset + i]
void launch_diff_wb_extract_mul(
    double* out, const double* a, const double* src,
    int64_t batch, int64_t m, int64_t src_stride, int64_t src_offset,
    cudaStream_t stream = 0);

// Pack dlam_nonneg (B, n_nonneg stride) into y_z (B, m stride) at offset k
void launch_diff_wb_pack_dlam_to_yz(
    double* y_z, const double* src,
    int64_t batch, int64_t m, int64_t k, int64_t n_nonneg,
    cudaStream_t stream = 0);

// Fill buffer with constant value
void launch_woodbury_fill_constant(
    double* out, double val,
    int64_t count, cudaStream_t stream = 0);

// ---- New kernels for corrected backward pass (inner Woodbury + full Schur) ----

// Build G = [sqrt(Λ_zd)*F_all, c1] for inner Woodbury
void launch_diff_wb_build_G(
    double* G, const double* F_all, long long f_all_stride,
    const double* Lambda, const double* c1,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext, cudaStream_t stream = 0);

// Add diagonal + c2*c2' + tau coupling to B_full (after F_all'F_all GEMM)
void launch_diff_wb_build_B_corrections(
    double* B, const double* v_nu, const double* sigma,
    const double* H_nonneg, const double* c2,
    const int64_t* dense_nonneg_idx, double eps,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total, int64_t k_ext,
    cudaStream_t stream = 0);

// Add identity to diagonal of k_ext × k_ext matrix
void launch_diff_wb_add_identity(
    double* A, int64_t batch, int64_t k_ext, cudaStream_t stream = 0);

// Add scaled identity (eps * I) to diagonal of k_ext × k_ext matrix
void launch_diff_wb_add_identity_scaled(
    double* A, double scale, int64_t batch, int64_t k_ext, cudaStream_t stream = 0);

// LDL^T factorization of k_ext × k_ext matrices (one per batch).
// Overwrites S in-place: L stored in strict lower triangle, D on diagonal.
void launch_diff_wb_ldlt_factor(
    double* S, int64_t batch, int64_t k_ext, cudaStream_t stream = 0);
void launch_diff_wb_ldlt_factor_scalar(
    double* S, int64_t batch, int64_t k_ext, cudaStream_t stream = 0);
void launch_diff_wb_ldlt_factor_warp(
    double* S, int64_t batch, int64_t k_ext, cudaStream_t stream = 0);
void launch_diff_wb_ldlt_factor_block(
    double* S, int64_t batch, int64_t k_ext, cudaStream_t stream = 0);

// LDL^T solve: x = (LDL^T)^{-1} b.  S holds the factored L/D from ldlt_factor.
// b is overwritten with x. One k_ext-vector per batch.
void launch_diff_wb_ldlt_solve(
    const double* S, double* b, int64_t batch, int64_t k_ext, cudaStream_t stream = 0);
void launch_diff_wb_ldlt_solve_scalar(
    const double* S, double* b, int64_t batch, int64_t k_ext, cudaStream_t stream = 0);
void launch_diff_wb_ldlt_solve_warp(
    const double* S, double* b, int64_t batch, int64_t k_ext, cudaStream_t stream = 0);
void launch_diff_wb_ldlt_solve_block(
    const double* S, double* b, int64_t batch, int64_t k_ext, cudaStream_t stream = 0);

// LDL^T solve with multiple RHS: X = (LDL^T)^{-1} B where B is k_ext × nrhs.
// B is overwritten with X. Column-major with ldb = k_ext.
void launch_diff_wb_ldlt_solve_mrhs(
    const double* S, double* B, int64_t batch, int64_t k_ext, int64_t nrhs,
    cudaStream_t stream = 0);
void launch_diff_wb_ldlt_solve_mrhs_scalar(
    const double* S, double* B, int64_t batch, int64_t k_ext, int64_t nrhs,
    cudaStream_t stream = 0);
void launch_diff_wb_ldlt_solve_mrhs_warp(
    const double* S, double* B, int64_t batch, int64_t k_ext, int64_t nrhs,
    cudaStream_t stream = 0);
void launch_diff_wb_ldlt_solve_mrhs_block(
    const double* S, double* B, int64_t batch, int64_t k_ext, int64_t nrhs,
    cudaStream_t stream = 0);

// Gather v_nu from v_z at zero/dense positions, append sigma
void launch_diff_wb_gather_v_nu(
    double* v_nu, const double* v_z, const double* sigma,
    const int64_t* dense_nonneg_idx,
    int64_t batch, int64_t m, int64_t k, int64_t k_d, int64_t k_total, int64_t k_ext,
    cudaStream_t stream = 0);

// Compute Sherman-Morrison denominator: rho = 1 + sum_{i∈λ}(c2[k+i]^2 / D_z[i])
void launch_diff_wb_compute_rho(
    double* rho, const double* D_z_inv, const double* c2,
    const int64_t* dense_nonneg_idx, int64_t k_d,
    int64_t batch, int64_t m, int64_t k,
    int64_t n_nonneg,
    cudaStream_t stream = 0);

// Apply sparse nonneg z-elimination corrections to B_full, F_ext, v_x, v_nu, sigma.
// When sparse nonneg variables (λ) are eliminated, they create fill-in between
// ν (zero/dense) variables and τ that must be subtracted from the Schur complement blocks.
void launch_diff_wb_sparse_z_elim_corrections(
    double* B_full, double* F_ext, double* v_x, double* v_nu, double* sigma,
    const double* F_all, long long f_all_stride,
    const double* D_z_inv, const double* rho, const double* v_z,
    const double* P_diag, const double* gh1, const double* c1, const double* c2,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    const int64_t* dense_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// Apply λ-elimination corrections to RHS ν and τ components.
// rhs_nu[l] -= Σ_i M'[ν_l, λ_i] * D_z_inv[i] * rhs_lam[i]
// rhs_nu[k_total] -= Σ_i v_z[k+ni] * D_z_inv[i] * rhs_lam[i]
void launch_diff_wb_rhs_lambda_elim(
    double* rhs_nu,
    const double* rhs_lam, const double* D_z_inv,
    const double* v_z,
    const double* F_all, long long f_all_stride,
    const double* c2,
    const double* A_s_vals,
    const int64_t* sparse_nonneg_idx, const int64_t* sparse_col,
    const int64_t* dense_nonneg_idx,
    const int64_t* col_offsets, const int64_t* col_sparse_rows,
    int64_t batch, int64_t n, int64_t m, int64_t k, int64_t k_d,
    int64_t k_total, int64_t k_ext,
    int64_t n_nonneg, int64_t n_sparse,
    cudaStream_t stream = 0);

// Set last column of F_ext to v_x
void launch_diff_wb_set_F_ext_last_col(
    double* F_ext, const double* v_x,
    int64_t batch, int64_t n, int64_t k_ext, int64_t k_total,
    cudaStream_t stream = 0);

// Elementwise vector subtraction: out = a - b
void launch_diff_wb_vec_sub(
    double* out, const double* a, const double* b,
    int64_t count, cudaStream_t stream = 0);

// Copy one scalar per batch between strided arrays:
// dst[b * dst_stride + dst_offset] = src[b * src_stride + src_offset]
void launch_diff_wb_strided_scalar_copy(
    double* dst, int64_t dst_stride, int64_t dst_offset,
    const double* src, int64_t src_stride, int64_t src_offset,
    int64_t batch, cudaStream_t stream = 0);

} // namespace moreau
