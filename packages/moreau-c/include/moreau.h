/**
 * @file moreau.h
 * @brief Moreau C API — Batched differentiable convex conic solver
 *
 * Unified C interface for both CPU (Rust/QDLDL/faer) and CUDA (cuDSS) backends.
 * Link against libmoreau_cpu and/or libmoreau_cuda depending on your target.
 *
 * Three-step API:
 *   1. moreau_solver_create()  — allocate solver with problem structure
 *   2. moreau_solver_setup()   — set P and A matrix values
 *   3. moreau_solver_solve()   — solve with q, b vectors
 *   4. moreau_solver_destroy() — free resources
 *
 * All data is double precision (float64). Matrices are CSR format.
 * CPU backend operates on host pointers. CUDA backend operates on device pointers
 * (for matrix values, q, b, and solution output).
 */

#ifndef MOREAU_H
#define MOREAU_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Version                                                                   */
/* ========================================================================= */

/**
 * @brief Return the library version string (e.g. "0.1.5").
 *
 * The returned pointer is valid for the lifetime of the library.
 */
const char* moreau_version(void);

/* ========================================================================= */
/* Error handling                                                            */
/* ========================================================================= */

/** Error codes returned by all API functions. */
typedef enum {
    MOREAU_OK = 0,
    MOREAU_ERROR_INVALID_ARGUMENT = 1,
    MOREAU_ERROR_NOT_SETUP = 2,       /**< solve() called before setup() */
    MOREAU_ERROR_NUMERICAL = 4,
    MOREAU_ERROR_OUT_OF_MEMORY = 5,
    MOREAU_ERROR_CUDA = 6,            /**< CUDA runtime error (CUDA backend only) */
    MOREAU_ERROR_INTERNAL = 99,
} moreau_error_t;

/**
 * @brief Return a human-readable description of the last error.
 *
 * Thread-local. Valid until the next moreau_* call on the same thread.
 * Returns NULL if no error has occurred.
 */
const char* moreau_last_error(void);

/* ========================================================================= */
/* Solver status                                                             */
/* ========================================================================= */

/** Termination status (matches both CPU and CUDA backends). */
typedef enum {
    MOREAU_STATUS_UNSOLVED = 0,
    MOREAU_STATUS_SOLVED,
    MOREAU_STATUS_PRIMAL_INFEASIBLE,
    MOREAU_STATUS_DUAL_INFEASIBLE,
    MOREAU_STATUS_ALMOST_SOLVED,
    MOREAU_STATUS_ALMOST_PRIMAL_INFEASIBLE,
    MOREAU_STATUS_ALMOST_DUAL_INFEASIBLE,
    MOREAU_STATUS_MAX_ITERATIONS,
    MOREAU_STATUS_MAX_TIME,
    MOREAU_STATUS_NUMERICAL_ERROR,
    MOREAU_STATUS_INSUFFICIENT_PROGRESS,
    MOREAU_STATUS_CALLBACK_TERMINATED,
} moreau_status_t;

/* ========================================================================= */
/* Direct solve method                                                       */
/* ========================================================================= */

/** KKT linear system solver selection. */
typedef enum {
    MOREAU_DIRECT_SOLVE_AUTO = 0,  /**< Backend picks best available */
    MOREAU_DIRECT_SOLVE_QDLDL,     /**< QDLDL (CPU only) */
    MOREAU_DIRECT_SOLVE_FAER,      /**< faer sparse LDL (CPU only) */
    MOREAU_DIRECT_SOLVE_CUDSS,     /**< cuDSS (CUDA only) */
} moreau_direct_solve_method_t;

/* ========================================================================= */
/* Cone specification                                                        */
/* ========================================================================= */

/**
 * @brief Cone structure for the constraint s in K.
 *
 * Ordering in the constraint vector s is:
 *   [zero (num_zero_cones)] [nonneg (num_nonneg_cones)]
 *   [SOC_1 (soc_dims[0])] ... [SOC_k (soc_dims[k-1])]
 *   [exp_1 (3)] ... [exp_j (3)]
 *   [pow_1 (3)] ... [pow_p (3)]
 */
typedef struct {
    int64_t num_zero_cones;    /**< Number of zero-cone (equality) constraints */
    int64_t num_nonneg_cones;  /**< Number of nonneg-cone (inequality) constraints */
    int64_t num_soc_cones;     /**< Number of second-order cones */
    const int64_t* soc_dims;   /**< Dimension of each SOC cone, length num_soc_cones.
                                     Each dimension must be >= 2. NULL if num_soc_cones == 0.
                                     Copied at create time; caller may free after. */
    int64_t num_exp_cones;     /**< Number of exponential cones (each dim 3) */
    int64_t num_power_cones;   /**< Number of power cones (each dim 3) */
    const double* power_alphas; /**< Power cone alphas, length num_power_cones.
                                     Each alpha must be in (0, 1). NULL if num_power_cones == 0.
                                     Copied at create time; caller may free after. */
} moreau_cones_t;

/* ========================================================================= */
/* Settings                                                                  */
/* ========================================================================= */

/**
 * @brief IPM-specific tolerance and algorithm settings.
 *
 * Initialize with moreau_ipm_settings_default() to get safe defaults.
 */
typedef struct {
    /* Convergence tolerances */
    double tol_gap_abs;        /**< Absolute duality gap tolerance (default 1e-8) */
    double tol_gap_rel;        /**< Relative duality gap tolerance (default 1e-8) */
    double tol_feas;           /**< Primal/dual feasibility tolerance (default 1e-8) */
    double tol_infeas_abs;     /**< Absolute infeasibility tolerance (default 1e-8) */
    double tol_infeas_rel;     /**< Relative infeasibility tolerance (default 1e-8) */
    double tol_ktratio;        /**< kappa/tau tolerance (default 1e-6) */

    /* Reduced-accuracy tolerances (for AlmostSolved status) */
    double reduced_tol_gap_abs;
    double reduced_tol_gap_rel;
    double reduced_tol_feas;
    double reduced_tol_infeas_abs;
    double reduced_tol_infeas_rel;
    double reduced_tol_ktratio;

    /* Equilibration */
    int    equilibrate_enable;   /**< Enable data equilibration (default 1) */
    int    equilibrate_max_iter; /**< Max equilibration iterations (default 10) */
    double equilibrate_min_scaling; /**< Min equilibration scale (default 1e-4) */
    double equilibrate_max_scaling; /**< Max equilibration scale (default 1e+4) */

    /* Step size control */
    double max_step_fraction;        /**< Max interior point step (default 0.99) */
    double linesearch_backtrack_step; /**< Backtracking factor (default 0.8) */
    double min_switch_step_length;   /**< Min step for PrimalDual scaling (default 0.1) */
    double min_terminate_step_length; /**< Min step for termination (default 1e-4) */

    /* KKT solver */
    moreau_direct_solve_method_t direct_solve_method;

    /* Static regularization */
    int    static_regularization_enable;
    double static_regularization_constant;
    double static_regularization_proportional;

    /* Dynamic regularization */
    int    dynamic_regularization_enable;
    double dynamic_regularization_eps;
    double dynamic_regularization_delta;
} moreau_ipm_settings_t;

/**
 * @brief Top-level solver settings.
 *
 * Initialize with moreau_settings_default() to get safe defaults.
 */
typedef struct {
    int64_t  batch_size;     /**< Number of problems to solve in parallel (default 1) */
    uint32_t max_iter;       /**< Maximum IPM iterations (default 200) */
    double   time_limit;     /**< Maximum solve time in seconds (default: no limit) */
    int      verbose;        /**< Print iteration log (default 1). Causes CPU/GPU sync on CUDA. */
    int      enable_grad;    /**< Enable backward-pass gradient computation (default 0) */

    moreau_ipm_settings_t ipm;  /**< IPM-specific settings */
} moreau_settings_t;

/**
 * @brief Fill settings with safe defaults.
 */
void moreau_settings_default(moreau_settings_t* settings);

/* ========================================================================= */
/* Opaque solver handle                                                      */
/* ========================================================================= */

/** Opaque solver handle. */
typedef struct moreau_solver_s moreau_solver_t;

/* ========================================================================= */
/* Solver lifecycle                                                          */
/* ========================================================================= */

/**
 * @brief Create a solver for the given problem structure.
 *
 * Allocates internal data structures for the sparsity pattern and cones.
 * Does NOT copy matrix values — call moreau_solver_setup() for that.
 *
 * @param[out] solver_out  Receives the new solver handle on success.
 * @param n                Number of primal variables.
 * @param m                Number of constraints.
 * @param P_row_offsets    CSR row offsets for P (length n+1).
 * @param P_col_indices    CSR column indices for P (length nnz_P).
 * @param nnz_P            Number of nonzeros in P.
 *                         P must be full symmetric (both upper and lower triangles).
 * @param A_row_offsets    CSR row offsets for A (length m+1).
 * @param A_col_indices    CSR column indices for A (length nnz_A).
 * @param nnz_A            Number of nonzeros in A.
 * @param cones            Cone specification. Copied; caller retains ownership.
 * @param settings         Solver settings. NULL for defaults.
 * @return MOREAU_OK on success.
 */
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
);

/**
 * @brief Set matrix values (Step 2).
 *
 * Can be called multiple times to update values for the same structure.
 *
 * For batch_size == 1, P_values has length nnz_P, A_values has length nnz_A.
 * For batch_size > 1, values are either:
 *   - length nnz (shared across all batches), or
 *   - length batch_size * nnz (row-major: batch 0, batch 1, ...).
 *
 * CPU backend: host pointers. CUDA backend: device pointers.
 *
 * @par P symmetry contract
 * P_values must satisfy P[i,j] == P[j,i] for every (i,j) in the sparsity
 * pattern. This is the caller's responsibility — the solver does NOT check
 * it. Asymmetric values produce undefined behavior: silent non-convergence
 * on the CUDA backend, lower-triangle drop on the CPU backend.
 * (Python wrappers validate this; raw C/C++/Julia callers do not.)
 *
 * @param solver     Solver handle.
 * @param P_values   P matrix values. Must be symmetric (see contract above).
 * @param P_count    Number of doubles in P_values (nnz_P or batch_size * nnz_P).
 * @param A_values   A matrix values.
 * @param A_count    Number of doubles in A_values (nnz_A or batch_size * nnz_A).
 * @return MOREAU_OK on success.
 */
moreau_error_t moreau_solver_setup(
    moreau_solver_t* solver,
    const double* P_values,
    int64_t P_count,
    const double* A_values,
    int64_t A_count
);

/**
 * @brief Solve the optimization problem (Step 3).
 *
 * Requires setup() to have been called first.
 *
 * For batch_size == 1, q has length n, b has length m.
 * For batch_size > 1, arrays are row-major: batch_size * n and batch_size * m.
 *
 * CPU backend: host pointers. CUDA backend: device pointers.
 *
 * @param solver  Solver handle.
 * @param q       Linear cost vector(s).
 * @param b       Constraint RHS vector(s).
 * @return MOREAU_OK on success (even if individual batches are infeasible — check status).
 */
moreau_error_t moreau_solver_solve(
    moreau_solver_t* solver,
    const double* q,
    const double* b
);

/**
 * @brief Solve with warm start.
 *
 * Same as moreau_solver_solve() but uses provided (x, z, s) as initial point.
 *
 * @param solver   Solver handle.
 * @param q        Linear cost vector(s), length [batch_size *] n.
 * @param b        Constraint RHS vector(s), length [batch_size *] m.
 * @param warm_x   Warm start primal variables, length [batch_size *] n.
 * @param warm_z   Warm start dual variables, length [batch_size *] m.
 * @param warm_s   Warm start slack variables, length [batch_size *] m.
 * @return MOREAU_OK on success.
 */
moreau_error_t moreau_solver_solve_warm(
    moreau_solver_t* solver,
    const double* q,
    const double* b,
    const double* warm_x,
    const double* warm_z,
    const double* warm_s
);

/**
 * @brief Destroy solver and free all resources.
 *
 * After this call, the handle is invalid. Passing NULL is a no-op.
 */
void moreau_solver_destroy(moreau_solver_t* solver);

/* ========================================================================= */
/* Solution access                                                           */
/* ========================================================================= */

/**
 * @brief Solution data for a single problem in the batch.
 *
 * Pointers are valid until the next solve() or destroy() call on the same solver.
 * CPU backend: host pointers. CUDA backend: device pointers.
 */
typedef struct {
    const double* x;          /**< Primal solution, length n */
    const double* z;          /**< Dual solution, length m */
    const double* s;          /**< Slack variables, length m */
    moreau_status_t status;   /**< Termination status */
    double obj_val;           /**< Primal objective value */
    double obj_val_dual;      /**< Dual objective value */
    double solve_time;        /**< Solve time in seconds */
    int32_t iterations;       /**< Number of IPM iterations */
    double r_prim;            /**< Primal residual */
    double r_dual;            /**< Dual residual */
} moreau_solution_t;

/**
 * @brief Get the solution for a specific batch index.
 *
 * @param solver     Solver handle.
 * @param batch_idx  Batch index (0 to batch_size-1).
 * @param[out] sol   Filled with solution data. Pointers are borrowed from solver.
 * @return MOREAU_OK on success.
 */
moreau_error_t moreau_solver_get_solution(
    const moreau_solver_t* solver,
    int64_t batch_idx,
    moreau_solution_t* sol
);

/**
 * @brief Copy solution vectors to caller-provided buffers.
 *
 * Unlike moreau_solver_get_solution() which returns internal pointers,
 * this copies data into user-owned buffers. For the CUDA backend, this
 * performs a device-to-host copy.
 *
 * @param solver     Solver handle.
 * @param batch_idx  Batch index (0 to batch_size-1).
 * @param x_out      Output buffer for primal solution (length n), or NULL to skip.
 * @param z_out      Output buffer for dual solution (length m), or NULL to skip.
 * @param s_out      Output buffer for slack variables (length m), or NULL to skip.
 * @return MOREAU_OK on success.
 */
moreau_error_t moreau_solver_copy_solution(
    const moreau_solver_t* solver,
    int64_t batch_idx,
    double* x_out,
    double* z_out,
    double* s_out
);

/**
 * @brief Get the solver status for a specific batch index.
 *
 * Lighter-weight than moreau_solver_get_solution() when you only need the status.
 */
moreau_status_t moreau_solver_get_status(
    const moreau_solver_t* solver,
    int64_t batch_idx
);

/* ========================================================================= */
/* Backward differentiation                                                  */
/* ========================================================================= */

/**
 * @brief Compute backward pass (gradients of solution w.r.t. problem data).
 *
 * Requires enable_grad=1 in settings.
 *
 * Given gradients of a loss w.r.t. (x, z, s), computes gradients w.r.t.
 * (P_values, A_values, q, b).
 *
 * CPU backend: host pointers. CUDA backend: device pointers.
 *
 * @param solver       Solver handle.
 * @param dx           Gradient of loss w.r.t. x, length [batch_size *] n.
 * @param dz           Gradient of loss w.r.t. z, length [batch_size *] m. May be NULL.
 * @param ds           Gradient of loss w.r.t. s, length [batch_size *] m. May be NULL.
 * @param dP_out       Output: gradient w.r.t. P values, length [batch_size *] nnz_P. May be NULL.
 * @param dA_out       Output: gradient w.r.t. A values, length [batch_size *] nnz_A. May be NULL.
 * @param dq_out       Output: gradient w.r.t. q, length [batch_size *] n. May be NULL.
 * @param db_out       Output: gradient w.r.t. b, length [batch_size *] m. May be NULL.
 * @return MOREAU_OK on success.
 */
moreau_error_t moreau_solver_backward(
    moreau_solver_t* solver,
    const double* dx,
    const double* dz,
    const double* ds,
    double* dP_out,
    double* dA_out,
    double* dq_out,
    double* db_out
);

/* ========================================================================= */
/* Solver information                                                        */
/* ========================================================================= */

/**
 * @brief Query problem dimensions.
 */
moreau_error_t moreau_solver_get_dims(
    const moreau_solver_t* solver,
    int64_t* n_out,
    int64_t* m_out,
    int64_t* batch_size_out,
    int64_t* nnz_P_out,
    int64_t* nnz_A_out
);

/**
 * @brief Query solver memory usage in bytes.
 */
moreau_error_t moreau_solver_memory_usage(
    const moreau_solver_t* solver,
    size_t* bytes_out
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MOREAU_H */
