/**
 * @file backward.hpp
 * @brief CPU backward pass for active-set QP solver
 *
 * KKT implicit differentiation for active-set-solved QPs.
 * Supports exact (hard active-set) and smoothed (barrier-based) modes.
 * All temporary memory pre-allocated to avoid heap allocations in hot loop.
 */

#pragma once

#include "moreau/solver/active_set/types.hpp"
#include <cstdint>
#include <vector>

namespace moreau {

/**
 * @brief Explicit forward state needed by the active-set backward pass.
 *
 * This is the workspace-derived subset of the forward solve state that the
 * backward formulas read. It is intentionally independent of solver ownership
 * so it can be cached per forward call and restored later without replaying
 * solve().
 */
struct ActiveSetBackwardState {
    std::vector<double> rinv;      // packed upper-triangular [n*(n+1)/2] if !use_rinv_diag
    std::vector<double> rinv_diag; // diagonal representation [n] if use_rinv_diag
    std::vector<int32_t> ws;       // active-set indices [m], first n_active entries used
    std::vector<int32_t> sense;    // constraint sense flags [m]
    std::vector<double> lam_star;  // active multipliers [m], first n_active entries used
    int32_t n_active = 0;
    bool use_rinv_diag = false;
};

/**
 * @brief Pre-allocated workspace for backward pass (one per thread)
 */
struct ActiveSetBackwardWorkspace {
    std::vector<double> dlam_bar;     // [m] (full m for smoothed, n_active for exact)
    std::vector<double> v1;           // [n]
    std::vector<double> v2;           // [m]
    std::vector<double> dx_bar_eff;   // [n]
    std::vector<double> A_W;          // [m*n] weighted A rows (A_S for exact, D_h*A for smoothed)
    std::vector<double> B;            // [m*n] A_W * Rinv
    std::vector<double> Schur;        // [m*m]
    std::vector<double> temp;         // [n]
    std::vector<double> correction;   // [n]
    std::vector<double> Hinv_temp;    // [n]
    std::vector<double> h_weights;    // [m] smoothed cone weights
    std::vector<double> z_mu;         // [m] central-path dual (smoothed mode)

    void allocate(int n, int m) {
        dlam_bar.resize(m);
        v1.resize(n);
        v2.resize(m);
        dx_bar_eff.resize(n);
        A_W.resize(m * n);
        B.resize(m * n);
        Schur.resize(m * m);
        temp.resize(n);
        correction.resize(n);
        Hinv_temp.resize(n);
        h_weights.resize(m);
        z_mu.resize(m);
    }
};

/**
 * @brief Compute backward pass for a single QP problem
 *
 * @param z_sol Dual solution for this batch element [m]
 * @param s_sol Slack solution for this batch element [m]
 * @param diff_method Exact or Smoothed differentiation
 * @param smoothing_mu Smoothing parameter μ (only used in Smoothed mode)
 */
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
);

/**
 * @brief Capture the workspace-derived forward state required by backward.
 */
ActiveSetBackwardState save_backward_state(const DaqpWorkspace& work, int n, int m);

} // namespace moreau
