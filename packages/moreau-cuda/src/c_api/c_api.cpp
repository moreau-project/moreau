/**
 * @file c_api.cpp
 * @brief C API implementation for Moreau CUDA solver
 *
 * Implements the functions declared in include/moreau.h for the CUDA backend.
 * All value pointers (P_values, A_values, q, b, solution x/z/s) are device pointers.
 * Structure pointers (row_offsets, col_indices) are host pointers at create() time.
 */

// moreau.h is at the repo root include/ directory, added via target_include_directories
#include "moreau.h"

#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <stdexcept>

#include "moreau/solver/solver.hpp"
#include "moreau/diff/diff.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/vector/vector.hpp"
#include "moreau/solution/solution.hpp"
#include "moreau/solver/status.hpp"

// ---------------------------------------------------------------------------
// CUDA error checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do { \
    cudaError_t _err = (call); \
    if (_err != cudaSuccess) { \
        set_last_error(std::string(#call " failed: ") + cudaGetErrorString(_err)); \
        return MOREAU_ERROR_CUDA; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Thread-local error message
// ---------------------------------------------------------------------------
static thread_local std::string g_last_error;

static void set_last_error(const std::string& msg) {
    g_last_error = msg;
}

static void clear_last_error() {
    g_last_error.clear();
}

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------
struct moreau_solver_s {
    std::unique_ptr<moreau::CompiledSolver> solver;
    std::unique_ptr<moreau::DiffState> diff_state;
    moreau::Settings settings;
    int64_t n;
    int64_t m;
    int64_t nnz_P;
    int64_t nnz_A;
    int64_t batch_size;
    bool enable_grad;
    bool is_setup;
    bool has_solution;
};

// ---------------------------------------------------------------------------
// Status conversion
// ---------------------------------------------------------------------------
static moreau_status_t convert_status(moreau::SolverStatus status) {
    switch (status) {
        case moreau::SolverStatus::Unsolved:              return MOREAU_STATUS_UNSOLVED;
        case moreau::SolverStatus::Solved:                return MOREAU_STATUS_SOLVED;
        case moreau::SolverStatus::PrimalInfeasible:      return MOREAU_STATUS_PRIMAL_INFEASIBLE;
        case moreau::SolverStatus::DualInfeasible:        return MOREAU_STATUS_DUAL_INFEASIBLE;
        case moreau::SolverStatus::AlmostSolved:          return MOREAU_STATUS_ALMOST_SOLVED;
        case moreau::SolverStatus::AlmostPrimalInfeasible: return MOREAU_STATUS_ALMOST_PRIMAL_INFEASIBLE;
        case moreau::SolverStatus::AlmostDualInfeasible:  return MOREAU_STATUS_ALMOST_DUAL_INFEASIBLE;
        case moreau::SolverStatus::MaxIterations:         return MOREAU_STATUS_MAX_ITERATIONS;
        case moreau::SolverStatus::MaxTime:               return MOREAU_STATUS_MAX_TIME;
        case moreau::SolverStatus::NumericalError:        return MOREAU_STATUS_NUMERICAL_ERROR;
        case moreau::SolverStatus::InsufficientProgress:  return MOREAU_STATUS_INSUFFICIENT_PROGRESS;
        case moreau::SolverStatus::CallbackTerminated:    return MOREAU_STATUS_CALLBACK_TERMINATED;
        default:                                          return MOREAU_STATUS_UNSOLVED;
    }
}

// ---------------------------------------------------------------------------
// Cones validation
//
// Defense-in-depth validation of every field the user supplies through
// `moreau_cones_t`. Without this, a negative
// `soc_dim` would be cast to a huge size_t and downstream allocators
// abort the process; an alpha outside (0, 1) silently corrupts power-cone
// projection. The Python wrapper validates these at a higher level, but
// raw C/Julia callers reach this code directly.
//
// Returns an empty error string on success; on first failure, returns a
// human-readable message describing the offending field.
// ---------------------------------------------------------------------------
static std::string validate_cones(const moreau_cones_t* c) {
    if (c->num_zero_cones < 0)
        return "num_zero_cones must be >= 0, got " + std::to_string(c->num_zero_cones);
    if (c->num_nonneg_cones < 0)
        return "num_nonneg_cones must be >= 0, got " + std::to_string(c->num_nonneg_cones);
    if (c->num_soc_cones < 0)
        return "num_soc_cones must be >= 0, got " + std::to_string(c->num_soc_cones);
    if (c->num_exp_cones < 0)
        return "num_exp_cones must be >= 0, got " + std::to_string(c->num_exp_cones);
    if (c->num_power_cones < 0)
        return "num_power_cones must be >= 0, got " + std::to_string(c->num_power_cones);

    if (c->num_soc_cones > 0) {
        if (c->soc_dims == nullptr) {
            return "soc_dims is NULL but num_soc_cones > 0";
        }
        for (int64_t i = 0; i < c->num_soc_cones; ++i) {
            if (c->soc_dims[i] < 2) {
                return "soc_dims[" + std::to_string(i) + "] = " +
                       std::to_string(c->soc_dims[i]) +
                       " must be >= 2 (SOC requires dim >= 2)";
            }
        }
    }

    if (c->num_power_cones > 0) {
        if (c->power_alphas == nullptr) {
            return "power_alphas is NULL but num_power_cones > 0";
        }
        for (int64_t i = 0; i < c->num_power_cones; ++i) {
            double a = c->power_alphas[i];
            if (!std::isfinite(a) || a <= 0.0 || a >= 1.0) {
                return "power_alphas[" + std::to_string(i) + "] = " +
                       std::to_string(a) +
                       " must be finite and in (0, 1)";
            }
        }
    }

    return std::string();
}

// ---------------------------------------------------------------------------
// Settings/Cones conversion (assumes validate_cones() already passed)
// ---------------------------------------------------------------------------
static moreau::Cones convert_cones(const moreau_cones_t* c) {
    moreau::Cones cones;
    cones.numZeroCones = c->num_zero_cones;
    cones.numNonnegCones = c->num_nonneg_cones;
    cones.numSocCones = c->num_soc_cones;
    if (c->num_soc_cones > 0 && c->soc_dims != nullptr) {
        cones.socConeDims.assign(c->soc_dims, c->soc_dims + c->num_soc_cones);
    }
    cones.numExpCones = c->num_exp_cones;
    cones.numPowerCones = c->num_power_cones;
    if (c->num_power_cones > 0 && c->power_alphas != nullptr) {
        cones.powerAlphas.assign(c->power_alphas, c->power_alphas + c->num_power_cones);
    }
    return cones;
}

static moreau::Settings convert_settings(const moreau_settings_t* s) {
    moreau::Settings settings;
    settings.maxIter = s->max_iter;
    settings.timeLimit = s->time_limit;
    settings.verbose = (s->verbose != 0);

    auto& ipm = settings.ipm;
    ipm.tolGapAbs = s->ipm.tol_gap_abs;
    ipm.tolGapRel = s->ipm.tol_gap_rel;
    ipm.tolFeas = s->ipm.tol_feas;
    ipm.tolInfeasAbs = s->ipm.tol_infeas_abs;
    ipm.tolInfeasRel = s->ipm.tol_infeas_rel;
    ipm.tolKtRatio = s->ipm.tol_ktratio;
    ipm.reducedTolGapAbs = s->ipm.reduced_tol_gap_abs;
    ipm.reducedTolGapRel = s->ipm.reduced_tol_gap_rel;
    ipm.reducedTolFeas = s->ipm.reduced_tol_feas;
    ipm.reducedTolInfeasAbs = s->ipm.reduced_tol_infeas_abs;
    ipm.reducedTolInfeasRel = s->ipm.reduced_tol_infeas_rel;
    ipm.reducedTolKtRatio = s->ipm.reduced_tol_ktratio;

    ipm.equilibrationSettings.enable = (s->ipm.equilibrate_enable != 0);
    ipm.equilibrationSettings.max_iter = s->ipm.equilibrate_max_iter;
    ipm.equilibrationSettings.scale_min = s->ipm.equilibrate_min_scaling;
    ipm.equilibrationSettings.scale_max = s->ipm.equilibrate_max_scaling;

    ipm.maxStepFraction = s->ipm.max_step_fraction;
    ipm.linesearchBacktrackStep = s->ipm.linesearch_backtrack_step;
    ipm.minSwitchStepLength = s->ipm.min_switch_step_length;
    ipm.minTerminateStepLength = s->ipm.min_terminate_step_length;

    switch (s->ipm.direct_solve_method) {
        case MOREAU_DIRECT_SOLVE_CUDSS:
            ipm.kktSolverType = moreau::KKTSolverType::CuDSS;
            break;
        default:
            ipm.kktSolverType = moreau::KKTSolverType::Auto;
            break;
    }

    ipm.staticRegularizationEnable = (s->ipm.static_regularization_enable != 0);
    ipm.staticRegularizationConstant = s->ipm.static_regularization_constant;
    ipm.staticRegularizationProportional = s->ipm.static_regularization_proportional;
    ipm.dynamicRegularizationEnable = (s->ipm.dynamic_regularization_enable != 0);
    ipm.dynamicRegularizationEps = s->ipm.dynamic_regularization_eps;
    ipm.dynamicRegularizationDelta = s->ipm.dynamic_regularization_delta;

    return settings;
}

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------
extern "C"
const char* moreau_version(void) {
#ifdef MOREAU_VERSION
    static const char version[] = MOREAU_VERSION;
#else
    static const char version[] = "unknown";
#endif
    return version;
}

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------
extern "C"
const char* moreau_last_error(void) {
    if (g_last_error.empty()) {
        return nullptr;
    }
    return g_last_error.c_str();
}

// ---------------------------------------------------------------------------
// Settings defaults
// ---------------------------------------------------------------------------
extern "C"
void moreau_settings_default(moreau_settings_t* s) {
    if (!s) return;

    s->batch_size = 1;
    s->max_iter = 200;
    s->time_limit = INFINITY;
    s->verbose = 1;
    s->enable_grad = 0;

    auto& ipm = s->ipm;
    ipm.tol_gap_abs = 1e-8;
    ipm.tol_gap_rel = 1e-8;
    ipm.tol_feas = 1e-8;
    ipm.tol_infeas_abs = 1e-8;
    ipm.tol_infeas_rel = 1e-8;
    ipm.tol_ktratio = 1e-6;
    ipm.reduced_tol_gap_abs = 5e-5;
    ipm.reduced_tol_gap_rel = 5e-5;
    ipm.reduced_tol_feas = 1e-4;
    ipm.reduced_tol_infeas_abs = 5e-12;
    ipm.reduced_tol_infeas_rel = 5e-5;
    ipm.reduced_tol_ktratio = 1e-4;
    ipm.equilibrate_enable = 1;
    ipm.equilibrate_max_iter = 10;
    ipm.equilibrate_min_scaling = 1e-4;
    ipm.equilibrate_max_scaling = 1e4;
    ipm.max_step_fraction = 0.99;
    ipm.linesearch_backtrack_step = 0.8;
    ipm.min_switch_step_length = 0.1;
    ipm.min_terminate_step_length = 1e-4;
    ipm.direct_solve_method = MOREAU_DIRECT_SOLVE_AUTO;
    ipm.static_regularization_enable = 1;
    ipm.static_regularization_constant = 1e-8;
    ipm.static_regularization_proportional =
        std::numeric_limits<double>::epsilon() * std::numeric_limits<double>::epsilon();
    ipm.dynamic_regularization_enable = 1;
    ipm.dynamic_regularization_eps = 1e-13;
    ipm.dynamic_regularization_delta = 2e-7;
}

// ---------------------------------------------------------------------------
// Create
// ---------------------------------------------------------------------------
extern "C"
moreau_error_t moreau_solver_create(
    moreau_solver_t** solver_out,
    int64_t n,
    int64_t m,
    const int64_t* P_row_offsets,
    const int64_t* P_col_indices,
    int64_t nnz_P,
    const int64_t* A_row_offsets,
    const int64_t* A_col_indices,
    int64_t nnz_A,
    const moreau_cones_t* cones,
    const moreau_settings_t* settings
) {
    clear_last_error();

    if (!solver_out) {
        set_last_error("solver_out is NULL");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }
    *solver_out = nullptr;

    if (n < 0 || m < 0 || nnz_P < 0 || nnz_A < 0) {
        set_last_error("Negative dimension");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }
    if (!P_row_offsets || !A_row_offsets || !cones) {
        set_last_error("NULL pointer for required argument");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }
    if (nnz_P > 0 && !P_col_indices) {
        set_last_error("P_col_indices is NULL but nnz_P > 0");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }
    if (nnz_A > 0 && !A_col_indices) {
        set_last_error("A_col_indices is NULL but nnz_A > 0");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }

    // Defense-in-depth: validate every cone field that comes from user memory.
    // Malformed cones (negative SOC dim, alpha outside (0,1), etc.) previously
    // reached cone projection code and aborted the process. The Python wrapper
    // validates these too, but raw C/Julia callers bypass that.
    {
        std::string cone_err = validate_cones(cones);
        if (!cone_err.empty()) {
            set_last_error("Invalid cones argument: " + cone_err);
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }
    }

    try {
        // Default settings if not provided
        moreau_settings_t default_s;
        moreau_settings_default(&default_s);
        const moreau_settings_t* s = settings ? settings : &default_s;

        int64_t batch_size = (s->batch_size < 1) ? 1 : s->batch_size;
        bool enable_grad = (s->enable_grad != 0);

        moreau::Cones moreau_cones = convert_cones(cones);
        moreau::Settings moreau_settings = convert_settings(s);

        auto handle = new moreau_solver_s();
        handle->n = n;
        handle->m = m;
        handle->nnz_P = nnz_P;
        handle->nnz_A = nnz_A;
        handle->batch_size = batch_size;
        handle->enable_grad = enable_grad;
        handle->is_setup = false;
        handle->has_solution = false;
        handle->settings = moreau_settings;

        handle->solver = std::make_unique<moreau::CompiledSolver>(
            n, m, batch_size,
            P_row_offsets, P_col_indices, nnz_P,
            A_row_offsets, A_col_indices, nnz_A,
            moreau_cones,
            moreau_settings
        );

        // Pre-initialize DiffKKT if gradients needed
        if (enable_grad) {
            moreau::init_diff_kkt(*handle->solver);
            handle->diff_state = std::make_unique<moreau::DiffState>(
                n, m, batch_size, nnz_P, nnz_A
            );
        }

        *solver_out = handle;
        return MOREAU_OK;
    } catch (const std::exception& e) {
        set_last_error(std::string("Failed to create solver: ") + e.what());
        return MOREAU_ERROR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------
extern "C"
moreau_error_t moreau_solver_setup(
    moreau_solver_t* solver,
    const double* P_values,
    int64_t P_count,
    const double* A_values,
    int64_t A_count
) {
    clear_last_error();

    if (!solver) {
        set_last_error("solver is NULL");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }

    try {
        // CUDA backend expects device pointers of size batch_size * nnz
        // If shared (P_count == nnz_P), we need to broadcast to batch_size * nnz_P
        int64_t nnz_P = solver->nnz_P;
        int64_t nnz_A = solver->nnz_A;
        int64_t bs = solver->batch_size;
        bool p_shared = (P_count == nnz_P);
        bool a_shared = (A_count == nnz_A);

        if (!p_shared && P_count != bs * nnz_P) {
            set_last_error("P_count must be nnz_P or batch_size * nnz_P");
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }
        if (!a_shared && A_count != bs * nnz_A) {
            set_last_error("A_count must be nnz_A or batch_size * nnz_A");
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        // For shared values, broadcast to batch_size copies on device
        double* d_P = nullptr;
        double* d_A = nullptr;
        bool p_temp = false;
        bool a_temp = false;

        if (p_shared && bs > 1 && nnz_P > 0) {
            cudaError_t err = cudaMalloc(&d_P, bs * nnz_P * sizeof(double));
            if (err != cudaSuccess) {
                set_last_error(std::string("cudaMalloc failed for P broadcast: ") + cudaGetErrorString(err));
                return MOREAU_ERROR_CUDA;
            }
            for (int64_t i = 0; i < bs; i++) {
                err = cudaMemcpy(d_P + i * nnz_P, P_values, nnz_P * sizeof(double), cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    cudaFree(d_P);
                    set_last_error(std::string("cudaMemcpy failed for P broadcast: ") + cudaGetErrorString(err));
                    return MOREAU_ERROR_CUDA;
                }
            }
            p_temp = true;
        } else {
            d_P = const_cast<double*>(P_values);
        }

        if (a_shared && bs > 1 && nnz_A > 0) {
            cudaError_t err = cudaMalloc(&d_A, bs * nnz_A * sizeof(double));
            if (err != cudaSuccess) {
                if (p_temp) cudaFree(d_P);
                set_last_error(std::string("cudaMalloc failed for A broadcast: ") + cudaGetErrorString(err));
                return MOREAU_ERROR_CUDA;
            }
            for (int64_t i = 0; i < bs; i++) {
                err = cudaMemcpy(d_A + i * nnz_A, A_values, nnz_A * sizeof(double), cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess) {
                    if (p_temp) cudaFree(d_P);
                    cudaFree(d_A);
                    set_last_error(std::string("cudaMemcpy failed for A broadcast: ") + cudaGetErrorString(err));
                    return MOREAU_ERROR_CUDA;
                }
            }
            a_temp = true;
        } else {
            d_A = const_cast<double*>(A_values);
        }

        try {
            solver->solver->setup(d_P, d_A);
        } catch (...) {
            if (p_temp) cudaFree(d_P);
            if (a_temp) cudaFree(d_A);
            throw;
        }

        if (p_temp) cudaFree(d_P);
        if (a_temp) cudaFree(d_A);

        solver->is_setup = true;
        solver->has_solution = false;
        return MOREAU_OK;
    } catch (const std::exception& e) {
        set_last_error(std::string("setup() failed: ") + e.what());
        return MOREAU_ERROR_CUDA;
    }
}

// ---------------------------------------------------------------------------
// Solve
// ---------------------------------------------------------------------------
extern "C"
moreau_error_t moreau_solver_solve(
    moreau_solver_t* solver,
    const double* q,
    const double* b
) {
    clear_last_error();

    if (!solver) {
        set_last_error("solver is NULL");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }
    if (!solver->is_setup) {
        set_last_error("solve() called before setup()");
        return MOREAU_ERROR_NOT_SETUP;
    }
    if ((!q && solver->n > 0) || (!b && solver->m > 0)) {
        set_last_error("q or b is NULL");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }

    try {
        solver->solver->solve(q, b);

        // Cache solution for backward if grad enabled
        if (solver->enable_grad && solver->diff_state) {
            moreau::cache_solution_for_backward(*solver->diff_state, *solver->solver);
        }

        solver->has_solution = true;
        return MOREAU_OK;
    } catch (const std::exception& e) {
        set_last_error(std::string("solve() failed: ") + e.what());
        return MOREAU_ERROR_NUMERICAL;
    }
}

// ---------------------------------------------------------------------------
// Solve with warm start
// ---------------------------------------------------------------------------
extern "C"
moreau_error_t moreau_solver_solve_warm(
    moreau_solver_t* solver,
    const double* q,
    const double* b,
    const double* warm_x,
    const double* warm_z,
    const double* warm_s
) {
    clear_last_error();

    if (!solver) {
        set_last_error("solver is NULL");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }
    if (!solver->is_setup) {
        set_last_error("solve() called before setup()");
        return MOREAU_ERROR_NOT_SETUP;
    }
    if ((!q && solver->n > 0) || (!b && solver->m > 0)
        || (!warm_x && solver->n > 0)
        || (!warm_z && solver->m > 0)
        || (!warm_s && solver->m > 0)) {
        set_last_error("NULL pointer for required argument");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }

    try {
        solver->solver->solve(q, b, warm_x, warm_z, warm_s);

        if (solver->enable_grad && solver->diff_state) {
            moreau::cache_solution_for_backward(*solver->diff_state, *solver->solver);
        }

        solver->has_solution = true;
        return MOREAU_OK;
    } catch (const std::exception& e) {
        set_last_error(std::string("solve_warm() failed: ") + e.what());
        return MOREAU_ERROR_NUMERICAL;
    }
}

// ---------------------------------------------------------------------------
// Destroy
// ---------------------------------------------------------------------------
extern "C"
void moreau_solver_destroy(moreau_solver_t* solver) {
    delete solver;
}

// ---------------------------------------------------------------------------
// Get solution (returns device pointers)
// ---------------------------------------------------------------------------
extern "C"
moreau_error_t moreau_solver_get_solution(
    const moreau_solver_t* solver,
    int64_t batch_idx,
    moreau_solution_t* sol
) {
    clear_last_error();

    if (!solver || !sol) {
        set_last_error("NULL pointer");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }
    if (!solver->has_solution) {
        set_last_error("No solution available (call solve() first)");
        return MOREAU_ERROR_NOT_SETUP;
    }
    if (batch_idx < 0 || batch_idx >= solver->batch_size) {
        set_last_error("batch_idx out of range");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }

    try {
        int64_t n = solver->n;
        int64_t m = solver->m;
        const auto& solution = solver->solver->solution;

        // Pointers into device memory
        sol->x = solution.x.data() + batch_idx * n;
        sol->z = solution.z.data() + batch_idx * m;
        sol->s = solution.s.data() + batch_idx * m;

        // Use host-side status from Info (synced after solve)
        if (batch_idx < static_cast<int64_t>(solver->solver->info.status.size())) {
            sol->status = convert_status(solver->solver->info.status[batch_idx]);
        } else {
            sol->status = MOREAU_STATUS_UNSOLVED;
        }

        double obj_val = 0;
        CUDA_CHECK(cudaMemcpy(&obj_val, solution.obj_val.data() + batch_idx,
                   sizeof(double), cudaMemcpyDeviceToHost));
        sol->obj_val = obj_val;

        double obj_val_dual = 0;
        CUDA_CHECK(cudaMemcpy(&obj_val_dual, solution.obj_val_dual.data() + batch_idx,
                   sizeof(double), cudaMemcpyDeviceToHost));
        sol->obj_val_dual = obj_val_dual;

        sol->solve_time = solution.solve_time;
        sol->iterations = static_cast<int32_t>(solution.iterations);

        double r_prim = 0, r_dual = 0;
        CUDA_CHECK(cudaMemcpy(&r_prim, solution.r_prim.data() + batch_idx,
                   sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&r_dual, solution.r_dual.data() + batch_idx,
                   sizeof(double), cudaMemcpyDeviceToHost));
        sol->r_prim = r_prim;
        sol->r_dual = r_dual;

        return MOREAU_OK;
    } catch (const std::exception& e) {
        set_last_error(std::string("get_solution() failed: ") + e.what());
        return MOREAU_ERROR_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// Copy solution (device to host)
// ---------------------------------------------------------------------------
extern "C"
moreau_error_t moreau_solver_copy_solution(
    const moreau_solver_t* solver,
    int64_t batch_idx,
    double* x_out,
    double* z_out,
    double* s_out
) {
    clear_last_error();

    if (!solver) {
        set_last_error("solver is NULL");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }
    if (!solver->has_solution) {
        set_last_error("No solution available");
        return MOREAU_ERROR_NOT_SETUP;
    }
    if (batch_idx < 0 || batch_idx >= solver->batch_size) {
        set_last_error("batch_idx out of range");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }

    try {
        int64_t n = solver->n;
        int64_t m = solver->m;
        const auto& solution = solver->solver->solution;

        if (x_out) {
            CUDA_CHECK(cudaMemcpy(x_out, solution.x.data() + batch_idx * n,
                       n * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (z_out) {
            CUDA_CHECK(cudaMemcpy(z_out, solution.z.data() + batch_idx * m,
                       m * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (s_out) {
            CUDA_CHECK(cudaMemcpy(s_out, solution.s.data() + batch_idx * m,
                       m * sizeof(double), cudaMemcpyDeviceToHost));
        }

        return MOREAU_OK;
    } catch (const std::exception& e) {
        set_last_error(std::string("copy_solution() failed: ") + e.what());
        return MOREAU_ERROR_CUDA;
    }
}

// ---------------------------------------------------------------------------
// Get status
// ---------------------------------------------------------------------------
extern "C"
moreau_status_t moreau_solver_get_status(
    const moreau_solver_t* solver,
    int64_t batch_idx
) {
    if (!solver || !solver->has_solution) {
        return MOREAU_STATUS_UNSOLVED;
    }
    if (batch_idx < 0 || batch_idx >= solver->batch_size) {
        return MOREAU_STATUS_UNSOLVED;
    }

    // Use Info's host-side status vector (synced after solve)
    if (batch_idx < static_cast<int64_t>(solver->solver->info.status.size())) {
        return convert_status(solver->solver->info.status[batch_idx]);
    }

    return MOREAU_STATUS_UNSOLVED;
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------
extern "C"
moreau_error_t moreau_solver_backward(
    moreau_solver_t* solver,
    const double* dx,
    const double* dz,
    const double* ds,
    double* dP_out,
    double* dA_out,
    double* dq_out,
    double* db_out
) {
    clear_last_error();

    if (!solver) {
        set_last_error("solver is NULL");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }
    if (!solver->enable_grad || !solver->diff_state) {
        set_last_error("backward() requires enable_grad=1 in settings");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }
    if (!solver->has_solution) {
        set_last_error("backward() requires solve() to be called first");
        return MOREAU_ERROR_NOT_SETUP;
    }
    if (!dx) {
        set_last_error("dx is NULL");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }

    try {
        int64_t n = solver->n;
        int64_t m = solver->m;
        int64_t bs = solver->batch_size;

        // Create BatchedVector views for input device pointers
        moreau::BatchedVector dx_bar(const_cast<double*>(dx), n, bs);

        // Wrap device pointer or create zero-filled temporary
        auto make_grad_vec = [&](const double* ptr) {
            if (ptr) {
                return std::make_unique<moreau::BatchedVector>(const_cast<double*>(ptr), m, bs);
            }
            auto v = std::make_unique<moreau::BatchedVector>(m, bs);
            v->setToConstant(0.0);
            return v;
        };

        auto dz_vec = make_grad_vec(dz);
        auto ds_vec = make_grad_vec(ds);

        moreau::backward(*solver->diff_state, dx_bar, *dz_vec, *ds_vec, *solver->solver);

        // Copy results to output buffers (device-to-device)
        if (dP_out && solver->nnz_P > 0) {
            CUDA_CHECK(cudaMemcpy(dP_out, solver->diff_state->dP_values.data(),
                       bs * solver->nnz_P * sizeof(double), cudaMemcpyDeviceToDevice));
        }
        if (dA_out && solver->nnz_A > 0) {
            CUDA_CHECK(cudaMemcpy(dA_out, solver->diff_state->dA_values.data(),
                       bs * solver->nnz_A * sizeof(double), cudaMemcpyDeviceToDevice));
        }
        if (dq_out) {
            CUDA_CHECK(cudaMemcpy(dq_out, solver->diff_state->dq.data(),
                       bs * n * sizeof(double), cudaMemcpyDeviceToDevice));
        }
        if (db_out) {
            CUDA_CHECK(cudaMemcpy(db_out, solver->diff_state->db.data(),
                       bs * m * sizeof(double), cudaMemcpyDeviceToDevice));
        }

        return MOREAU_OK;
    } catch (const std::exception& e) {
        set_last_error(std::string("backward() failed: ") + e.what());
        return MOREAU_ERROR_NUMERICAL;
    }
}

// ---------------------------------------------------------------------------
// Get dims
// ---------------------------------------------------------------------------
extern "C"
moreau_error_t moreau_solver_get_dims(
    const moreau_solver_t* solver,
    int64_t* n_out,
    int64_t* m_out,
    int64_t* batch_size_out,
    int64_t* nnz_P_out,
    int64_t* nnz_A_out
) {
    clear_last_error();

    if (!solver) {
        set_last_error("solver is NULL");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }

    if (n_out) *n_out = solver->n;
    if (m_out) *m_out = solver->m;
    if (batch_size_out) *batch_size_out = solver->batch_size;
    if (nnz_P_out) *nnz_P_out = solver->nnz_P;
    if (nnz_A_out) *nnz_A_out = solver->nnz_A;

    return MOREAU_OK;
}

// ---------------------------------------------------------------------------
// Memory usage
// ---------------------------------------------------------------------------
extern "C"
moreau_error_t moreau_solver_memory_usage(
    const moreau_solver_t* solver,
    size_t* bytes_out
) {
    clear_last_error();

    if (!solver || !bytes_out) {
        set_last_error("NULL pointer");
        return MOREAU_ERROR_INVALID_ARGUMENT;
    }

    try {
        *bytes_out = solver->solver->memoryUsage();
        return MOREAU_OK;
    } catch (const std::exception& e) {
        set_last_error(std::string("memoryUsage() failed: ") + e.what());
        return MOREAU_ERROR_INTERNAL;
    }
}
