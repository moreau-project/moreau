/**
 * @file factorization.hpp
 * @brief LDL factorization with rank-1 updates for active-set solver
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

/**
 * @brief Dot product of two vectors
 */
double daqp_dot(const double* v1, const double* v2, int n);

/**
 * @brief Rank-1 update of LDL factorization when adding a constraint
 *
 * Adds the constraint with index add_ind to the active set.
 * Updates L and D in-place.
 */
void daqp_update_LDL_add(DaqpWorkspace* work, int add_ind);

/**
 * @brief Rank-1 downdate of LDL factorization when removing a constraint
 *
 * Removes the constraint at position rm_ind in the working set.
 * Uses Algorithm C1 from Gill 1974.
 */
void daqp_update_LDL_remove(DaqpWorkspace* work, int rm_ind);

} // namespace moreau
