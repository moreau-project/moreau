/**
 * @file core.hpp
 * @brief Core active-set algorithm functions
 *
 * Derived from DAQP (https://github.com/darnstrom/daqp),
 * Copyright (c) 2022 Daniel Arnström, licensed under the MIT License.
 * See the repository NOTICE file for the full license text.
 *
 * Pure CPU, no dependencies.
 */

#pragma once

#include "moreau/solver/active_set/types.hpp"

namespace moreau {

// ============================================================================
// Main solver entry point
// ============================================================================

/**
 * @brief Run the dual active-set iteration loop (LDP form)
 * @return Exit flag (DAQP_EXIT_OPTIMAL, DAQP_EXIT_INFEASIBLE, etc.)
 */
int daqp_ldp(DaqpWorkspace* work);

/**
 * @brief Convert LDP solution back to QP solution: x = Rinv*(u-v)
 */
void ldp2qp_solution(DaqpWorkspace* work);

// ============================================================================
// Iteration subroutines
// ============================================================================

/**
 * @brief Compute the constrained stationary point (CSP)
 *
 * Solves (L*D*L') * lam_star = -d for the active constraints,
 * reusing forward substitution work where possible.
 */
void daqp_compute_CSP(DaqpWorkspace* work);

/**
 * @brief Compute singular direction when LDL is singular
 */
void daqp_compute_singular_direction(DaqpWorkspace* work);

/**
 * @brief Check dual feasibility and remove blocking constraint if needed
 * @return 1 if a constraint was removed, 0 if dual feasible (or infeasible)
 */
int daqp_remove_blocking(DaqpWorkspace* work);

/**
 * @brief Compute primal u = -M'*lam_star and objective fval = ||u||^2
 */
void daqp_compute_primal_and_fval(DaqpWorkspace* work);

/**
 * @brief Find most violated inactive constraint and add it
 * @return 1 if a constraint was added, 0 if primal feasible
 */
int daqp_add_infeasible(DaqpWorkspace* work);

/**
 * @brief Remove a constraint from the active set
 */
void daqp_remove_constraint(DaqpWorkspace* work, int rm_ind);

/**
 * @brief Add a constraint to the active set
 */
void daqp_add_constraint(DaqpWorkspace* work, int add_ind, double lam);

/**
 * @brief Pivot the last two active constraints for numerical stability
 */
void daqp_pivot_last(DaqpWorkspace* work);

/**
 * @brief Activate all constraints marked active in sense[]
 * @return 1 on success, negative on error
 */
int daqp_activate_constraints(DaqpWorkspace* work);

/**
 * @brief One step of iterative refinement for active constraints
 */
void daqp_refine_active(DaqpWorkspace* work);

} // namespace moreau
