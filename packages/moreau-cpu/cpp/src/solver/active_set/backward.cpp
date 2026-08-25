/**
 * @file backward.cpp
 * @brief CPU backward pass for active-set QP solver
 *
 * Two modes:
 * - Exact: hard active-set KKT adjoint (discontinuous at constraint transitions)
 * - Smoothed: barrier-smoothed KKT adjoint (C^∞, smooth through transitions)
 *
 * Smoothed mode uses H_i = z_i^2 / (z_i^2 + μ) as a soft indicator of
 * constraint activity, matching the IPM's smoothed differentiation approach.
 */

#include "moreau/solver/active_set/backward.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#include <algorithm>

namespace moreau {

// ============================================================================
// Packed upper-triangular helpers
// ============================================================================

static inline int packed_row_start(int i, int n) {
    return i * n - (i * (i - 1)) / 2;
}

static void packed_upper_tri_matvec(const double* Rinv, const double* x,
                                     double* y, int n) {
    std::memset(y, 0, n * sizeof(double));
    for (int i = 0; i < n; i++) {
        int rs = packed_row_start(i, n);
        for (int j = i; j < n; j++) {
            y[i] += Rinv[rs + (j - i)] * x[j];
        }
    }
}

static void packed_upper_tri_transpose_matvec(const double* Rinv, const double* x,
                                               double* y, int n) {
    std::memset(y, 0, n * sizeof(double));
    for (int i = 0; i < n; i++) {
        int rs = packed_row_start(i, n);
        for (int j = i; j < n; j++) {
            y[j] += Rinv[rs + (j - i)] * x[i];
        }
    }
}

static void apply_Hinv(const double* Rinv, const double* RinvD,
                        const double* x, double* y, double* temp, int n) {
    if (RinvD) {
        for (int i = 0; i < n; i++) y[i] = RinvD[i] * RinvD[i] * x[i];
    } else {
        packed_upper_tri_transpose_matvec(Rinv, x, temp, n);
        packed_upper_tri_matvec(Rinv, temp, y, n);
    }
}

static void apply_Hinv(const ActiveSetBackwardState& state,
                       const double* x, double* y, double* temp, int n) {
    const double* rinv = state.use_rinv_diag ? nullptr : state.rinv.data();
    const double* rinv_diag = state.use_rinv_diag ? state.rinv_diag.data() : nullptr;
    apply_Hinv(rinv, rinv_diag, x, y, temp, n);
}

static bool dense_cholesky(double* A, int n) {
    for (int j = 0; j < n; j++) {
        double sum = A[j * n + j];
        for (int k = 0; k < j; k++) sum -= A[j * n + k] * A[j * n + k];
        if (sum <= 0.0) return false;
        A[j * n + j] = std::sqrt(sum);
        for (int i = j + 1; i < n; i++) {
            double s = A[i * n + j];
            for (int k = 0; k < j; k++) s -= A[i * n + k] * A[j * n + k];
            A[i * n + j] = s / A[j * n + j];
        }
    }
    return true;
}

static void solve_lower(const double* L, double* x, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i; j++) x[i] -= L[i * n + j] * x[j];
        x[i] /= L[i * n + i];
    }
}

static void solve_upper_from_lower(const double* L, double* x, int n) {
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) x[i] -= L[j * n + i] * x[j];
        x[i] /= L[i * n + i];
    }
}

// ============================================================================
// Smoothed backward pass
// ============================================================================
//
// Differentiates through the smoothed KKT conditions (central path):
//   H x + q + A'z = 0
//   Ax + s = b
//   s_i * z_i = μ   (complementarity on central path)
//
// Three-variable adjoint (x, z, s independent). Eliminating the
// complementarity adjoint u3 = Z^{-1}(ds_bar - u2) yields:
//   [H        A'         ] [v1]   [dx_bar                       ]
//   [A   -diag(S/Z)      ] [v3] = [dz_bar - diag(S/Z) ds_bar   ]
//
// Schur complement (note -(-(S/Z)) = +S/Z):
//   (A H^{-1} A' + diag(S/Z)) v3 = A v1_init - dz_bar + diag(S/Z) ds_bar
//   v1 = v1_init - H^{-1} A' v3
//
// Note: ds_bar enters ONLY through the Schur RHS term (S/Z)*ds_bar.
// Do NOT also apply the two-variable s=b-Ax corrections (dx_bar_eff,
// db+=ds_bar, dA-=ds_bar*x') — that would double-count ds_bar.
//
// S/Z acts as regularization. For inactive constraints (s≫0, z≈0):
//   s_μ/z_μ = s/(μ/s) = s²/μ → large → v3[i] → 0 (constraint ignored, correct)
// For active constraints (s≈0, z≫0):
//   s_μ/z_μ = (μ/z)/z = μ/z² → small → v3[i] set by KKT (correct)
// The transition is smooth.

static void backward_smoothed(
    const ActiveSetBackwardState& state,
    ActiveSetBackwardWorkspace& bw,
    const double* H, const double* A,
    const double* dx_bar, const double* dz_bar, const double* ds_bar,
    const double* x_sol, const double* z_sol, const double* s_sol,
    int n, int m,
    int64_t numZeroCones,
    double smoothing_mu,
    double* dH_out, double* dq_out, double* dA_out, double* db_out
) {
    double* z_mu = bw.z_mu.data();
    double* h = bw.h_weights.data();  // reuse as s_mu
    double* v1 = bw.v1.data();
    double* v3 = bw.v2.data();  // adjoint dual
    double* dx_bar_eff = bw.dx_bar_eff.data();
    double* B = bw.B.data();      // A * Rinv [m × n]
    double* Schur = bw.Schur.data();
    double* temp = bw.temp.data();
    double* correction = bw.correction.data();
    double* hinv_temp = bw.Hinv_temp.data();
    double* rhs = bw.dlam_bar.data();
    double* s_mu = bw.h_weights.data();  // alias

    std::memset(dH_out, 0, n * n * sizeof(double));
    std::memset(dq_out, 0, n * sizeof(double));
    std::memset(dA_out, 0, m * n * sizeof(double));
    std::memset(db_out, 0, m * sizeof(double));

    // Compute central-path iterates (z_μ, s_μ) satisfying |s_μ| * |z_μ| = μ
    // z_mu preserves sign for correct adjoint (dA uses z_mu in A'*z derivative)
    for (int64_t i = 0; i < numZeroCones; i++) {
        // Equalities: z free, s=0. z_mu keeps sign of z_sol.
        double az = std::max(std::abs(z_sol[i]), std::sqrt(smoothing_mu));
        double sign = (z_sol[i] >= 0) ? 1.0 : -1.0;
        z_mu[i] = sign * az;
        s_mu[i] = smoothing_mu / az;  // s_mu >= 0 (for Schur regularization)
    }
    for (int64_t i = numZeroCones; i < m; i++) {
        double si = std::max(s_sol[i], 0.0);
        double zi = std::max(z_sol[i], 0.0);
        if (si > 1e-12) {
            s_mu[i] = si;
            z_mu[i] = smoothing_mu / si;
        } else if (zi > 1e-12) {
            z_mu[i] = zi;
            s_mu[i] = smoothing_mu / zi;
        } else {
            // Both near zero: symmetric point on central path
            s_mu[i] = std::sqrt(smoothing_mu);
            z_mu[i] = std::sqrt(smoothing_mu);
        }
    }

    // v1_init = H^{-1} * dx_bar
    // (ds_bar enters only via the Schur RHS, not through dx_bar)
    std::memcpy(dx_bar_eff, dx_bar, n * sizeof(double));
    apply_Hinv(state, dx_bar_eff, v1, hinv_temp, n);

    // B = A * Rinv [m × n] (unweighted)
    if (state.use_rinv_diag) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                B[i * n + j] = A[i * n + j] * state.rinv_diag[j];
    } else {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k <= j; k++) {
                    int rs = packed_row_start(k, n);
                    sum += A[i * n + k] * state.rinv[rs + (j - k)];
                }
                B[i * n + j] = sum;
            }
    }

    // Schur = A H^{-1} A' + diag(s_μ/z_μ) = B*B' + diag(s_μ/z_μ)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            for (int k = 0; k < n; k++) dot += B[i * n + k] * B[j * n + k];
            Schur[i * m + j] = dot;
            Schur[j * m + i] = dot;
        }
        Schur[i * m + i] += s_mu[i] / std::abs(z_mu[i]);
    }

    // RHS = A * v1_init - dz_bar + diag(s_μ/z_μ) * ds_bar
    for (int i = 0; i < m; i++) {
        double Av1 = 0.0;
        for (int j = 0; j < n; j++) Av1 += A[i * n + j] * v1[j];
        rhs[i] = Av1 - dz_bar[i] + (s_mu[i] / std::abs(z_mu[i])) * ds_bar[i];
    }
    std::memcpy(v3, rhs, m * sizeof(double));

    // Solve Schur * v3 = rhs
    if (!dense_cholesky(Schur, m)) {
        throw std::runtime_error(
            "Cholesky factorization failed in smoothed backward pass (m=" +
            std::to_string(m) + "). The Schur complement is not positive definite. "
            "Try increasing diff_smoothing_mu or using diff_method='exact'.");
    } else {
        solve_lower(Schur, v3, m);
        solve_upper_from_lower(Schur, v3, m);
    }

    // v1 = v1_init - H^{-1} * A' * v3
    std::memset(temp, 0, n * sizeof(double));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            temp[j] += A[i * n + j] * v3[i];
    apply_Hinv(state, temp, correction, hinv_temp, n);
    for (int i = 0; i < n; i++) v1[i] -= correction[i];

    // Extract gradients
    for (int i = 0; i < n; i++) dq_out[i] = -v1[i];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dH_out[i * n + j] = -v1[i] * x_sol[j];

    // dA[i,:] = -v3[i] * x' - z_mu[i] * v1'
    // db[i] = v3[i]
    // (ds_bar is fully handled through the Schur RHS; no additional corrections)
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            dA_out[i * n + j] = -v3[i] * x_sol[j] - z_mu[i] * v1[j];
        }
        db_out[i] = v3[i];
    }
}

// ============================================================================
// Exact backward pass (original implementation)
// ============================================================================

static void backward_exact(
    const ActiveSetBackwardState& state,
    ActiveSetBackwardWorkspace& bw,
    const double* H, const double* A,
    const double* dx_bar, const double* dz_bar, const double* ds_bar,
    const double* x_sol,
    int n, int m,
    int64_t numZeroCones,
    double* dH_out, double* dq_out, double* dA_out, double* db_out
) {
    const int n_active = state.n_active;
    const double eps_reg = 1e-12;

    double* dlam_bar_S = bw.dlam_bar.data();
    double* v1 = bw.v1.data();
    double* v2 = bw.v2.data();
    double* dx_bar_eff = bw.dx_bar_eff.data();
    double* A_S = bw.A_W.data();
    double* B_buf = bw.B.data();
    double* Schur = bw.Schur.data();
    double* temp = bw.temp.data();
    double* correction = bw.correction.data();
    double* hinv_temp = bw.Hinv_temp.data();

    std::memset(dH_out, 0, n * n * sizeof(double));
    std::memset(dq_out, 0, n * sizeof(double));
    std::memset(dA_out, 0, m * n * sizeof(double));
    std::memset(db_out, 0, m * sizeof(double));

    // Map dz_bar → dlam_bar_S
    for (int k = 0; k < n_active; k++) {
        int idx = state.ws[k];
        dlam_bar_S[k] = (state.sense[idx] & DAQP_LOWER) ? -dz_bar[idx] : dz_bar[idx];
    }

    // Effective dx_bar
    for (int i = 0; i < n; i++) dx_bar_eff[i] = dx_bar[i];
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) dx_bar_eff[j] -= A[i * n + j] * ds_bar[i];
    }

    // v1_init = H^{-1} * dx_bar_eff
    apply_Hinv(state, dx_bar_eff, v1, hinv_temp, n);

    if (n_active == 0) {
        for (int i = 0; i < n; i++) dq_out[i] = -v1[i];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                dH_out[i * n + j] = -v1[i] * x_sol[j];
        for (int i = 0; i < m; i++) db_out[i] = ds_bar[i];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                dA_out[i * n + j] += -ds_bar[i] * x_sol[j];
        return;
    }

    // Build A_S [n_active × n]
    for (int k = 0; k < n_active; k++)
        std::memcpy(A_S + k * n, A + state.ws[k] * n, n * sizeof(double));

    // B = A_S * Rinv
    if (state.use_rinv_diag) {
        for (int k = 0; k < n_active; k++)
            for (int j = 0; j < n; j++)
                B_buf[k * n + j] = A_S[k * n + j] * state.rinv_diag[j];
    } else {
        for (int k = 0; k < n_active; k++)
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int i = 0; i <= j; i++) {
                    int rs = packed_row_start(i, n);
                    sum += A_S[k * n + i] * state.rinv[rs + (j - i)];
                }
                B_buf[k * n + j] = sum;
            }
    }

    // Schur = B * B^T + εI
    for (int i = 0; i < n_active; i++) {
        for (int j = 0; j <= i; j++) {
            double dot = 0.0;
            for (int k = 0; k < n; k++) dot += B_buf[i * n + k] * B_buf[j * n + k];
            Schur[i * n_active + j] = dot;
            Schur[j * n_active + i] = dot;
        }
        Schur[i * n_active + i] += eps_reg;
    }

    // RHS = A_S * v1_init - dlam_bar_S
    for (int k = 0; k < n_active; k++) {
        double Av1 = 0.0;
        for (int j = 0; j < n; j++) Av1 += A_S[k * n + j] * v1[j];
        v2[k] = Av1 - dlam_bar_S[k];
    }

    if (!dense_cholesky(Schur, n_active)) {
        throw std::runtime_error(
            "Cholesky factorization failed in exact backward pass (n_active=" +
            std::to_string(n_active) + "). The Schur complement is not positive definite. "
            "Try diff_method='smoothed' which is more numerically robust.");
    } else {
        solve_lower(Schur, v2, n_active);
        solve_upper_from_lower(Schur, v2, n_active);
    }

    // v1 -= H^{-1} * A_S^T * v2
    std::memset(temp, 0, n * sizeof(double));
    for (int k = 0; k < n_active; k++)
        for (int j = 0; j < n; j++)
            temp[j] += A_S[k * n + j] * v2[k];
    apply_Hinv(state, temp, correction, hinv_temp, n);
    for (int i = 0; i < n; i++) v1[i] -= correction[i];

    // Extract gradients
    for (int i = 0; i < n; i++) dq_out[i] = -v1[i];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            dH_out[i * n + j] = -v1[i] * x_sol[j];

    for (int k = 0; k < n_active; k++) {
        int idx = state.ws[k];
        double lambda_k = state.lam_star[k];
        for (int j = 0; j < n; j++)
            dA_out[idx * n + j] = -v2[k] * x_sol[j] - lambda_k * v1[j];
        db_out[idx] = (state.sense[idx] & DAQP_LOWER) ? -v2[k] : v2[k];
    }

    for (int i = 0; i < m; i++) db_out[i] += ds_bar[i];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            dA_out[i * n + j] += -ds_bar[i] * x_sol[j];
}

// ============================================================================
// Dispatch
// ============================================================================

ActiveSetBackwardState save_backward_state(const DaqpWorkspace& work, int n, int m) {
    ActiveSetBackwardState state;
    const int packed = daqp_arsum(n);
    state.use_rinv_diag = work.RinvD != nullptr;
    state.rinv.resize(packed, 0.0);
    state.rinv_diag.resize(n, 0.0);
    if (state.use_rinv_diag) {
        std::copy(work.RinvD, work.RinvD + n, state.rinv_diag.begin());
    } else {
        std::copy(work.Rinv, work.Rinv + packed, state.rinv.begin());
    }
    state.n_active = work.n_active;
    state.ws.resize(m, 0);
    state.sense.resize(m, 0);
    state.lam_star.resize(m, 0.0);
    for (int i = 0; i < m; i++) {
        state.sense[i] = work.sense[i];
    }
    for (int i = 0; i < work.n_active; i++) {
        state.ws[i] = work.WS[i];
        state.lam_star[i] = work.lam_star[i];
    }
    return state;
}

void active_set_backward_single(
    const ActiveSetBackwardState& state,
    ActiveSetBackwardWorkspace& bw,
    const double* H, const double* A,
    const double* q, const double* b,
    const double* dx_bar, const double* dz_bar, const double* ds_bar,
    const double* x_sol, const double* z_sol, const double* s_sol,
    int n, int m,
    int64_t numZeroCones,
    ActiveSetDiffMethod diff_method,
    double smoothing_mu,
    double* dH_out, double* dq_out, double* dA_out, double* db_out
) {
    if (diff_method == ActiveSetDiffMethod::Smoothed) {
        backward_smoothed(state, bw, H, A, dx_bar, dz_bar, ds_bar, x_sol, z_sol, s_sol,
                          n, m, numZeroCones, smoothing_mu,
                          dH_out, dq_out, dA_out, db_out);
    } else {
        backward_exact(state, bw, H, A, dx_bar, dz_bar, ds_bar, x_sol,
                       n, m, numZeroCones,
                       dH_out, dq_out, dA_out, db_out);
    }
}

} // namespace moreau
