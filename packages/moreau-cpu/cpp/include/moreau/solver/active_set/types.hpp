/**
 * @file types.hpp
 * @brief Data types for active-set QP solver
 *
 * Derived from DAQP (https://github.com/darnstrom/daqp),
 * Copyright (c) 2022 Daniel Arnström, licensed under the MIT License.
 * See the repository NOTICE file for the full license text.
 *
 * Pure CPU, no CUDA dependencies.
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <limits>
#include <vector>

namespace moreau {

// ============================================================================
// Constants
// ============================================================================

constexpr int DAQP_EMPTY_IND = -1;
constexpr int DAQP_UNCONSTRAINED_OPTIMAL = -2;
constexpr double DAQP_INF = 1e30;

// Exit flags
constexpr int DAQP_EXIT_OPTIMAL = 1;
constexpr int DAQP_EXIT_INFEASIBLE = -1;
constexpr int DAQP_EXIT_CYCLE = -2;
constexpr int DAQP_EXIT_UNBOUNDED = -3;
constexpr int DAQP_EXIT_ITERLIMIT = -4;
constexpr int DAQP_EXIT_NONCONVEX = -5;
constexpr int DAQP_EXIT_OVERDETERMINED_INITIAL = -6;
constexpr int DAQP_EXIT_TIMELIMIT = -7;

// Constraint sense masks
constexpr int DAQP_ACTIVE = 1;
constexpr int DAQP_LOWER = 2;
constexpr int DAQP_IMMUTABLE = 4;

// Macros as inline functions
inline int daqp_arsum(int x) { return (x * (x + 1)) / 2; }
inline int daqp_r_offset(int x, int y) { return ((2 * y - x - 1) * x) / 2; }

// ============================================================================
// Settings
// ============================================================================

// Differentiation method for active-set backward pass
enum class ActiveSetDiffMethod {
    Exact = 0,    // Hard active-set KKT (discontinuous at transitions)
    Smoothed = 1, // Barrier-smoothed KKT (C^∞, smooth through transitions)
};

struct ActiveSetSettings {
    double primal_tol = 1e-6;
    double dual_tol = 1e-12;
    double zero_tol = 1e-11;
    double pivot_tol = 1e-6;
    double progress_tol = 1e-14;
    double sing_tol = 3.7e-11;
    double refactor_tol = 1e-9;
    double eps_prox = 1e-6;
    double fval_bound = 1e30;
    int iter_limit = 10000;
    double time_limit = 1e30;  // seconds (propagated from Settings.timeLimit)
    int cycle_tol = 10;

    // Differentiation settings
    ActiveSetDiffMethod diff_method = ActiveSetDiffMethod::Exact;
    double diff_smoothing_mu = 1e-4;  // Smoothing parameter (larger = smoother)
};

// ============================================================================
// Workspace (single problem, all CPU)
// ============================================================================

struct DaqpWorkspace {
    DaqpWorkspace() = default;
    DaqpWorkspace(const DaqpWorkspace&) = delete;
    DaqpWorkspace& operator=(const DaqpWorkspace&) = delete;
    DaqpWorkspace(DaqpWorkspace&&) = delete;
    DaqpWorkspace& operator=(DaqpWorkspace&&) = delete;

    int n = 0;   // Number of primal variables
    int m = 0;   // Total number of constraints (no simple bounds: ms=0)

    // LDP data (owned by vectors below, pointers alias into them)
    double* M = nullptr;       // [m, n] transformed constraints (row-major)
    double* Rinv = nullptr;    // [n*(n+1)/2] upper-triangular packed
    double* RinvD = nullptr;   // [n] diagonal case (points into same storage as Rinv)
    double* v = nullptr;       // [n]
    double* dupper = nullptr;  // [m]
    double* dlower = nullptr;  // [m]
    int* sense = nullptr;      // [m]
    double* scaling = nullptr; // [m]

    // Iterates
    double* x = nullptr;       // [n] final primal solution
    double* u = nullptr;       // [n] stores M'*lam_star
    double* lam = nullptr;     // [m] dual iterate
    double* lam_star = nullptr;// [m] current constrained stationary point

    // LDL factors (M_active * M_active' = L * D * L')
    double* L = nullptr;       // [m*(m+1)/2] packed lower triangular
    double* D = nullptr;       // [m] diagonal
    double* xldl = nullptr;    // [m] forward sub intermediate
    double* zldl = nullptr;    // [m] scaled intermediate

    int* WS = nullptr;         // [m+1] working set indices
    int n_active = 0;
    int sing_ind = DAQP_EMPTY_IND;
    int reuse_ind = 0;

    double fval = 0.0;
    double soft_slack = 0.0;
    int iterations = 0;

    // Semi-proximal support
    int* prox_mask = nullptr;  // [n]
    int n_prox = 0;

    ActiveSetSettings settings;

    // Optional per-iteration callback for verbose output
    // Args: (iter, n_active, fval, added_constraint, removed_constraint)
    using IterCallback = void(*)(int iter, int n_active, double fval,
                                  int added, int removed, void* user_data);
    IterCallback iter_callback = nullptr;
    void* iter_callback_data = nullptr;

    // ========================================================================
    // Owning storage — pointers above alias into these
    // ========================================================================
    std::vector<double> M_buf;
    std::vector<double> Rinv_buf;
    std::vector<double> v_buf;
    std::vector<double> dupper_buf;
    std::vector<double> dlower_buf;
    std::vector<int> sense_buf;
    std::vector<double> scaling_buf;
    std::vector<double> x_buf;
    std::vector<double> u_buf;
    std::vector<double> lam_buf;
    std::vector<double> lam_star_buf;
    std::vector<double> L_buf;
    std::vector<double> D_buf;
    std::vector<double> xldl_buf;
    std::vector<double> zldl_buf;
    std::vector<int> WS_buf;
    std::vector<int> prox_mask_buf;

    /**
     * @brief Allocate all workspace buffers for given dimensions
     */
    void allocate(int n_, int m_) {
        n = n_;
        m = m_;

        M_buf.resize(m * n, 0.0);
        Rinv_buf.resize(daqp_arsum(n), 0.0);  // packed upper-triangular
        v_buf.resize(n, 0.0);
        dupper_buf.resize(m, 0.0);
        dlower_buf.resize(m, 0.0);
        sense_buf.resize(m, 0);
        scaling_buf.resize(m, 1.0);
        x_buf.resize(n, 0.0);
        u_buf.resize(n, 0.0);
        lam_buf.resize(m, 0.0);
        lam_star_buf.resize(m, 0.0);
        L_buf.resize(daqp_arsum(m), 0.0);
        D_buf.resize(m, 0.0);
        xldl_buf.resize(m, 0.0);
        zldl_buf.resize(m, 0.0);
        WS_buf.resize(m + 1, 0);
        prox_mask_buf.resize(n, 0);

        // Set pointers
        M = M_buf.data();
        Rinv = Rinv_buf.data();
        RinvD = nullptr;
        v = v_buf.data();
        dupper = dupper_buf.data();
        dlower = dlower_buf.data();
        sense = sense_buf.data();
        scaling = scaling_buf.data();
        x = x_buf.data();
        u = u_buf.data();
        lam = lam_buf.data();
        lam_star = lam_star_buf.data();
        L = L_buf.data();
        D = D_buf.data();
        xldl = xldl_buf.data();
        zldl = zldl_buf.data();
        WS = WS_buf.data();
        prox_mask = prox_mask_buf.data();
    }

    /**
     * @brief Reset iteration state (not problem data)
     */
    void reset() {
        sing_ind = DAQP_EMPTY_IND;
        n_active = 0;
        reuse_ind = 0;
        fval = 0.0;
        soft_slack = 0.0;
        iterations = 0;
        std::fill(lam_buf.begin(), lam_buf.end(), 0.0);
        std::fill(lam_star_buf.begin(), lam_star_buf.end(), 0.0);
        std::fill(u_buf.begin(), u_buf.end(), 0.0);
    }
};

} // namespace moreau
