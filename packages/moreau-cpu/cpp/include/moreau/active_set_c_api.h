/**
 * @file active_set_c_api.h
 * @brief C API for the active-set QP solver
 *
 * Four-step API:
 *   1. moreau_as_create()  — allocate solver with problem structure
 *   2. moreau_as_setup()   — set P and A matrix values
 *   3. moreau_as_solve()   — solve with q, b vectors
 *   4. moreau_as_destroy() — free resources
 *
 * All data is double precision (float64). Matrices are CSR format.
 * All pointers are host pointers.
 */

#ifndef MOREAU_ACTIVE_SET_C_API_H
#define MOREAU_ACTIVE_SET_C_API_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================= */
/* Error handling                                                            */
/* ========================================================================= */

typedef enum {
    MOREAU_AS_OK = 0,
    MOREAU_AS_ERROR_INVALID_ARGUMENT = 1,
    MOREAU_AS_ERROR_NOT_SETUP = 2,
    MOREAU_AS_ERROR_NOT_SOLVED = 3,
    MOREAU_AS_ERROR_GRAD_DISABLED = 4,
    MOREAU_AS_ERROR_INTERNAL = 99,
} moreau_as_error_t;

/**
 * @brief Return a human-readable description of the last error.
 *
 * Thread-local. Valid until the next moreau_as_* call on the same thread.
 * Returns NULL if no error has occurred.
 */
const char* moreau_as_last_error(void);

/* ========================================================================= */
/* Differentiation method                                                    */
/* ========================================================================= */

typedef enum {
    MOREAU_AS_DIFF_EXACT = 0,
    MOREAU_AS_DIFF_SMOOTHED = 1,
} moreau_as_diff_method_t;

/* ========================================================================= */
/* Settings                                                                  */
/* ========================================================================= */

typedef struct {
    double primal_tol;       /* default 1e-6 */
    double dual_tol;         /* default 1e-12 */
    double zero_tol;         /* default 1e-11 */
    double pivot_tol;        /* default 1e-6 */
    double progress_tol;     /* default 1e-14 */
    double fval_bound;       /* default 1e30 */
    int    iter_limit;       /* default 10000 */
    double time_limit;       /* default 1e30 */
    int    cycle_tol;        /* default 10 */

    moreau_as_diff_method_t diff_method;  /* default EXACT */
    double diff_smoothing_mu;             /* default 1e-4 */
} moreau_as_settings_t;

/**
 * @brief Fill settings with safe defaults.
 */
void moreau_as_settings_default(moreau_as_settings_t* settings);

/* ========================================================================= */
/* Opaque solver handle                                                      */
/* ========================================================================= */

typedef struct moreau_as_solver_s moreau_as_solver_t;

/* ========================================================================= */
/* Solver lifecycle                                                          */
/* ========================================================================= */

/**
 * @brief Create an active-set solver for the given problem structure.
 *
 * Only zero + nonneg cones are supported (pure QP).
 *
 * @param[out] solver_out   Receives the new solver handle on success.
 * @param n                 Number of primal variables.
 * @param m                 Number of constraints.
 * @param batch_size        Number of problems to solve in parallel.
 * @param P_row_offsets     CSR row offsets for P (length n+1).
 * @param P_col_indices     CSR column indices for P (length nnz_P).
 * @param nnz_P             Number of nonzeros in P (full symmetric).
 * @param A_row_offsets     CSR row offsets for A (length m+1).
 * @param A_col_indices     CSR column indices for A (length nnz_A).
 * @param nnz_A             Number of nonzeros in A.
 * @param num_zero_cones    Number of zero (equality) constraints.
 * @param num_nonneg_cones  Number of nonneg (inequality) constraints.
 * @param settings          Solver settings. NULL for defaults.
 * @param enable_grad       Enable backward-pass gradient computation.
 * @param verbose           Print iteration log.
 * @return MOREAU_AS_OK on success.
 */
moreau_as_error_t moreau_as_create(
    moreau_as_solver_t** solver_out,
    int64_t n,
    int64_t m,
    int64_t batch_size,
    const int64_t* P_row_offsets,
    const int64_t* P_col_indices,
    int64_t nnz_P,
    const int64_t* A_row_offsets,
    const int64_t* A_col_indices,
    int64_t nnz_A,
    int64_t num_zero_cones,
    int64_t num_nonneg_cones,
    const moreau_as_settings_t* settings,
    int enable_grad,
    int verbose
);

/**
 * @brief Set P and A matrix values (Step 2).
 *
 * @param solver    Solver handle.
 * @param P_values  P matrix values, length nnz_P (shared) or batch_size * nnz_P.
 * @param A_values  A matrix values, length nnz_A (shared) or batch_size * nnz_A.
 * @param shared    1 if values are shared across batch, 0 if per-batch.
 * @return MOREAU_AS_OK on success.
 */
moreau_as_error_t moreau_as_setup(
    moreau_as_solver_t* solver,
    const double* P_values,
    const double* A_values,
    int shared
);

/**
 * @brief Solve the QP (Step 3).
 *
 * @param solver  Solver handle.
 * @param q       Linear cost, length batch_size * n.
 * @param b       Constraint RHS, length batch_size * m.
 * @return MOREAU_AS_OK on success.
 */
moreau_as_error_t moreau_as_solve(
    moreau_as_solver_t* solver,
    const double* q,
    const double* b
);

/**
 * @brief Solve with warm start.
 *
 * @param solver   Solver handle.
 * @param q        Linear cost, length batch_size * n.
 * @param b        Constraint RHS, length batch_size * m.
 * @param warm_x   Warm start primal, length batch_size * n.
 * @param warm_z   Warm start dual, length batch_size * m.
 * @param warm_s   Warm start slack, length batch_size * m.
 * @return MOREAU_AS_OK on success.
 */
moreau_as_error_t moreau_as_solve_warm(
    moreau_as_solver_t* solver,
    const double* q,
    const double* b,
    const double* warm_x,
    const double* warm_z,
    const double* warm_s
);

/**
 * @brief Compute backward pass.
 *
 * Requires enable_grad=1 at creation and a prior solve() call.
 *
 * @param solver  Solver handle.
 * @param dx      Gradient w.r.t. x, length batch_size * n.
 * @param dz      Gradient w.r.t. z, length batch_size * m. May be NULL.
 * @param ds      Gradient w.r.t. s, length batch_size * m. May be NULL.
 * @return MOREAU_AS_OK on success.
 */
moreau_as_error_t moreau_as_backward(
    moreau_as_solver_t* solver,
    const double* dx,
    const double* dz,
    const double* ds
);

/**
 * @brief Compute backward pass from explicit problem data and saved solution.
 *
 * Unlike moreau_as_backward(), this does not require a prior solve() call on
 * the solver handle. The provided forward inputs and iterates are used to
 * reconstruct the backward state inside the backend.
 *
 * @param solver     Solver handle.
 * @param dx         Gradient w.r.t. x, length batch_size * n.
 * @param dz         Gradient w.r.t. z, length batch_size * m.
 * @param ds         Gradient w.r.t. s, length batch_size * m.
 * @param P_values   P values, length nnz_P if shared, else batch_size * nnz_P.
 * @param A_values   A values, length nnz_A if shared, else batch_size * nnz_A.
 * @param shared     Whether P_values and A_values are shared across the batch.
 * @param q          Linear cost, length batch_size * n.
 * @param b          Constraint RHS, length batch_size * m.
 * @param x          Saved primal solution, length batch_size * n.
 * @param z          Saved dual solution, length batch_size * m.
 * @param s          Saved slack solution, length batch_size * m.
 * @return MOREAU_AS_OK on success.
 */
moreau_as_error_t moreau_as_backward_with_data(
    moreau_as_solver_t* solver,
    const double* dx,
    const double* dz,
    const double* ds,
    const double* P_values,
    const double* A_values,
    int shared,
    const double* q,
    const double* b,
    const double* x,
    const double* z,
    const double* s,
    const double* state_rinv,
    const double* state_rinv_diag,
    const int32_t* state_use_rinv_diag,
    const int32_t* state_n_active,
    const int32_t* state_ws,
    const int32_t* state_sense,
    const double* state_lam_star
);

/**
 * @brief Destroy solver and free all resources.
 *
 * After this call, the handle is invalid. Passing NULL is a no-op.
 */
void moreau_as_destroy(moreau_as_solver_t* solver);

/* ========================================================================= */
/* Solution access                                                           */
/* ========================================================================= */

/**
 * @brief Solution data for a single problem in the batch.
 */
typedef struct {
    const double* x;          /**< Primal solution, length n */
    const double* z;          /**< Dual solution, length m */
    const double* s;          /**< Slack variables, length m */
    int32_t status;           /**< DAQP exit flag (1 = optimal) */
    double obj_val;           /**< Primal objective value */
    int32_t iterations;       /**< Number of active-set iterations */
} moreau_as_solution_t;

/**
 * @brief Get the solution for a specific batch index.
 *
 * Pointers are borrowed from the solver — valid until next solve() or destroy().
 */
moreau_as_error_t moreau_as_get_solution(
    const moreau_as_solver_t* solver,
    int64_t batch_idx,
    moreau_as_solution_t* sol
);

/**
 * @brief Get the total solve time in seconds (all batches).
 */
moreau_as_error_t moreau_as_get_solve_time(
    const moreau_as_solver_t* solver,
    double* time_out
);

/* ========================================================================= */
/* Backward results access                                                   */
/* ========================================================================= */

/**
 * @brief Backward gradient data for a single problem in the batch.
 */
typedef struct {
    const double* dP_values;  /**< Gradient w.r.t. P values, length nnz_P */
    const double* dA_values;  /**< Gradient w.r.t. A values, length nnz_A */
    const double* dq;         /**< Gradient w.r.t. q, length n */
    const double* db;         /**< Gradient w.r.t. b, length m */
} moreau_as_backward_t;

typedef struct {
    const double* rinv;       /**< Packed upper-triangular inverse factor, length n*(n+1)/2 */
    const double* rinv_diag;  /**< Diagonal inverse factor, length n */
    const int32_t* ws;        /**< Active-set indices, length m */
    const int32_t* sense;     /**< Constraint sense flags, length m */
    const double* lam_star;   /**< Active multipliers, length m */
    int32_t n_active;         /**< Number of active constraints */
    int32_t use_rinv_diag;    /**< 1 if rinv_diag is active, else rinv */
} moreau_as_backward_state_t;

/**
 * @brief Get backward results for a specific batch index.
 *
 * Pointers are borrowed — valid until next backward() or destroy().
 */
moreau_as_error_t moreau_as_get_backward(
    const moreau_as_solver_t* solver,
    int64_t batch_idx,
    moreau_as_backward_t* grad
);

moreau_as_error_t moreau_as_get_backward_state(
    const moreau_as_solver_t* solver,
    int64_t batch_idx,
    moreau_as_backward_state_t* state
);

/* ========================================================================= */
/* Dimensions                                                                */
/* ========================================================================= */

moreau_as_error_t moreau_as_get_dims(
    const moreau_as_solver_t* solver,
    int64_t* n_out,
    int64_t* m_out,
    int64_t* batch_size_out,
    int64_t* nnz_P_out,
    int64_t* nnz_A_out
);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MOREAU_ACTIVE_SET_C_API_H */
