/**
 * @file c_api_active_set.cpp
 * @brief C API implementation for the active-set QP solver
 */

#include "moreau/active_set_c_api.h"
#include "moreau/solver/active_set/solver.hpp"
#include "moreau/cones/cones.hpp"
#include <string>
#include <new>

// ============================================================================
// Thread-local error reporting
// ============================================================================

static thread_local std::string g_last_error;

static void clear_last_error() {
    g_last_error.clear();
}

static void set_last_error(const std::string& msg) {
    g_last_error = msg;
}

extern "C" {

const char* moreau_as_last_error(void) {
    if (g_last_error.empty()) return nullptr;
    return g_last_error.c_str();
}

// ============================================================================
// Settings defaults
// ============================================================================

void moreau_as_settings_default(moreau_as_settings_t* settings) {
    if (!settings) return;
    settings->primal_tol = 1e-6;
    settings->dual_tol = 1e-12;
    settings->zero_tol = 1e-11;
    settings->pivot_tol = 1e-6;
    settings->progress_tol = 1e-14;
    settings->fval_bound = 1e30;
    settings->iter_limit = 10000;
    settings->time_limit = 1e30;
    settings->cycle_tol = 10;
    settings->diff_method = MOREAU_AS_DIFF_EXACT;
    settings->diff_smoothing_mu = 1e-4;
}

// ============================================================================
// Convert C settings to C++ settings
// ============================================================================

static moreau::ActiveSetSettings to_cpp_settings(const moreau_as_settings_t* s) {
    moreau::ActiveSetSettings cpp;
    if (!s) return cpp;
    cpp.primal_tol = s->primal_tol;
    cpp.dual_tol = s->dual_tol;
    cpp.zero_tol = s->zero_tol;
    cpp.pivot_tol = s->pivot_tol;
    cpp.progress_tol = s->progress_tol;
    cpp.fval_bound = s->fval_bound;
    cpp.iter_limit = s->iter_limit;
    cpp.time_limit = s->time_limit;
    cpp.cycle_tol = s->cycle_tol;
    cpp.diff_method = (s->diff_method == MOREAU_AS_DIFF_SMOOTHED)
                      ? moreau::ActiveSetDiffMethod::Smoothed
                      : moreau::ActiveSetDiffMethod::Exact;
    cpp.diff_smoothing_mu = s->diff_smoothing_mu;
    return cpp;
}

// ============================================================================
// Opaque handle
// ============================================================================

struct moreau_as_solver_s {
    moreau::ActiveSetSolver solver;

    moreau_as_solver_s(
        int64_t n, int64_t m, int64_t batchSize,
        const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
        const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
        const moreau::Cones& cones,
        const moreau::ActiveSetSettings& settings,
        bool enable_grad, bool verbose
    ) : solver(n, m, batchSize, P_ro, P_ci, nnzP, A_ro, A_ci, nnzA,
               cones, settings, enable_grad, verbose) {}
};

// ============================================================================
// Lifecycle
// ============================================================================

moreau_as_error_t moreau_as_create(
    moreau_as_solver_t** solver_out,
    int64_t n, int64_t m, int64_t batch_size,
    const int64_t* P_row_offsets, const int64_t* P_col_indices, int64_t nnz_P,
    const int64_t* A_row_offsets, const int64_t* A_col_indices, int64_t nnz_A,
    int64_t num_zero_cones, int64_t num_nonneg_cones,
    const moreau_as_settings_t* settings,
    int enable_grad, int verbose
) {
    clear_last_error();
    if (!solver_out) {
        set_last_error("solver_out is NULL");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }
    *solver_out = nullptr;

    try {
        moreau::Cones cones;
        cones.numZeroCones = num_zero_cones;
        cones.numNonnegCones = num_nonneg_cones;

        moreau::ActiveSetSettings cpp_settings = to_cpp_settings(settings);

        *solver_out = new moreau_as_solver_s(
            n, m, batch_size,
            P_row_offsets, P_col_indices, nnz_P,
            A_row_offsets, A_col_indices, nnz_A,
            cones, cpp_settings,
            enable_grad != 0, verbose != 0
        );
        return MOREAU_AS_OK;
    } catch (const std::invalid_argument& e) {
        set_last_error(e.what());
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    } catch (const std::bad_alloc& e) {
        set_last_error(std::string("Out of memory: ") + e.what());
        return MOREAU_AS_ERROR_INTERNAL;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return MOREAU_AS_ERROR_INTERNAL;
    }
}

moreau_as_error_t moreau_as_setup(
    moreau_as_solver_t* solver,
    const double* P_values, const double* A_values,
    int shared
) {
    clear_last_error();
    if (!solver) {
        set_last_error("solver is NULL");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }
    try {
        solver->solver.setup(P_values, A_values, shared != 0);
        return MOREAU_AS_OK;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return MOREAU_AS_ERROR_INTERNAL;
    }
}

moreau_as_error_t moreau_as_solve(
    moreau_as_solver_t* solver,
    const double* q, const double* b
) {
    clear_last_error();
    if (!solver) {
        set_last_error("solver is NULL");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }
    try {
        solver->solver.solve(q, b);
        return MOREAU_AS_OK;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return MOREAU_AS_ERROR_INTERNAL;
    }
}

moreau_as_error_t moreau_as_solve_warm(
    moreau_as_solver_t* solver,
    const double* q, const double* b,
    const double* warm_x, const double* warm_z, const double* warm_s
) {
    clear_last_error();
    if (!solver) {
        set_last_error("solver is NULL");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }
    try {
        solver->solver.solve_warm_start(q, b, warm_x, warm_z, warm_s);
        return MOREAU_AS_OK;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return MOREAU_AS_ERROR_INTERNAL;
    }
}

moreau_as_error_t moreau_as_backward(
    moreau_as_solver_t* solver,
    const double* dx, const double* dz, const double* ds
) {
    clear_last_error();
    if (!solver) {
        set_last_error("solver is NULL");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }
    try {
        solver->solver.backward(dx, dz, ds);
        return MOREAU_AS_OK;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return MOREAU_AS_ERROR_INTERNAL;
    }
}

moreau_as_error_t moreau_as_backward_with_data(
    moreau_as_solver_t* solver,
    const double* dx, const double* dz, const double* ds,
    const double* P_values, const double* A_values, int shared,
    const double* q, const double* b,
    const double* x, const double* z, const double* s,
    const double* state_rinv,
    const double* state_rinv_diag,
    const int32_t* state_use_rinv_diag,
    const int32_t* state_n_active,
    const int32_t* state_ws,
    const int32_t* state_sense,
    const double* state_lam_star
) {
    clear_last_error();
    if (!solver) {
        set_last_error("solver is NULL");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }
    try {
        solver->solver.backward_with_data(
            dx, dz, ds,
            P_values, A_values, shared != 0,
            q, b, x, z, s,
            state_rinv, state_rinv_diag,
            state_use_rinv_diag, state_n_active,
            state_ws, state_sense, state_lam_star
        );
        return MOREAU_AS_OK;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return MOREAU_AS_ERROR_INTERNAL;
    }
}

void moreau_as_destroy(moreau_as_solver_t* solver) {
    delete solver;
}

// ============================================================================
// Solution access
// ============================================================================

moreau_as_error_t moreau_as_get_solution(
    const moreau_as_solver_t* solver,
    int64_t batch_idx,
    moreau_as_solution_t* sol
) {
    clear_last_error();
    if (!solver || !sol) {
        set_last_error("solver or sol is NULL");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }
    const auto& s = solver->solver;
    int64_t bs = s.batchSize();
    if (batch_idx < 0 || batch_idx >= bs) {
        set_last_error("batch_idx out of range");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }

    int64_t n = s.n();
    int64_t m = s.m();
    sol->x = s.x_sol.data() + batch_idx * n;
    sol->z = s.z_sol.data() + batch_idx * m;
    sol->s = s.s_sol.data() + batch_idx * m;
    sol->status = s.status_vec[batch_idx];
    sol->obj_val = s.obj_val[batch_idx];
    sol->iterations = s.iters[batch_idx];
    return MOREAU_AS_OK;
}

moreau_as_error_t moreau_as_get_solve_time(
    const moreau_as_solver_t* solver,
    double* time_out
) {
    clear_last_error();
    if (!solver || !time_out) {
        set_last_error("solver or time_out is NULL");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }
    *time_out = solver->solver.solve_time;
    return MOREAU_AS_OK;
}

// ============================================================================
// Backward results access
// ============================================================================

moreau_as_error_t moreau_as_get_backward(
    const moreau_as_solver_t* solver,
    int64_t batch_idx,
    moreau_as_backward_t* grad
) {
    clear_last_error();
    if (!solver || !grad) {
        set_last_error("solver or grad is NULL");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }
    const auto& s = solver->solver;
    int64_t bs = s.batchSize();
    if (batch_idx < 0 || batch_idx >= bs) {
        set_last_error("batch_idx out of range");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }

    int64_t n = s.n();
    int64_t m = s.m();
    int64_t nnzP = s.nnzP();
    int64_t nnzA = s.nnzA();
    grad->dP_values = s.dP_values.data() + batch_idx * nnzP;
    grad->dA_values = s.dA_values.data() + batch_idx * nnzA;
    grad->dq = s.dq.data() + batch_idx * n;
    grad->db = s.db.data() + batch_idx * m;
    return MOREAU_AS_OK;
}

moreau_as_error_t moreau_as_get_backward_state(
    const moreau_as_solver_t* solver,
    int64_t batch_idx,
    moreau_as_backward_state_t* state
) {
    clear_last_error();
    if (!solver || !state) {
        set_last_error("solver or state is NULL");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }
    const auto& s = solver->solver;
    int64_t bs = s.batchSize();
    if (batch_idx < 0 || batch_idx >= bs) {
        set_last_error("batch_idx out of range");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }

    const auto& backward_state = s.backward_state(batch_idx);
    state->rinv = backward_state.rinv.empty() ? nullptr : backward_state.rinv.data();
    state->rinv_diag = backward_state.rinv_diag.empty() ? nullptr : backward_state.rinv_diag.data();
    state->ws = backward_state.ws.empty() ? nullptr : backward_state.ws.data();
    state->sense = backward_state.sense.empty() ? nullptr : backward_state.sense.data();
    state->lam_star = backward_state.lam_star.empty() ? nullptr : backward_state.lam_star.data();
    state->n_active = backward_state.n_active;
    state->use_rinv_diag = backward_state.use_rinv_diag ? 1 : 0;
    return MOREAU_AS_OK;
}

// ============================================================================
// Dimensions
// ============================================================================

moreau_as_error_t moreau_as_get_dims(
    const moreau_as_solver_t* solver,
    int64_t* n_out, int64_t* m_out,
    int64_t* batch_size_out,
    int64_t* nnz_P_out, int64_t* nnz_A_out
) {
    clear_last_error();
    if (!solver) {
        set_last_error("solver is NULL");
        return MOREAU_AS_ERROR_INVALID_ARGUMENT;
    }
    const auto& s = solver->solver;
    if (n_out) *n_out = s.n();
    if (m_out) *m_out = s.m();
    if (batch_size_out) *batch_size_out = s.batchSize();
    if (nnz_P_out) *nnz_P_out = s.nnzP();
    if (nnz_A_out) *nnz_A_out = s.nnzA();
    return MOREAU_AS_OK;
}

} // extern "C"
