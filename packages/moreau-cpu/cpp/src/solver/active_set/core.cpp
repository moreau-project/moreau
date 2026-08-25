/**
 * @file core.cpp
 * @brief Core active-set algorithm
 *
 * Derived from DAQP (https://github.com/darnstrom/daqp),
 * Copyright (c) 2022 Daniel Arnström, licensed under the MIT License.
 * See the repository NOTICE file for the full license text.
 *
 * 
 * Stripped: soft weights, binary constraints, BnB, AVI, hierarchy, time limit.
 * Simplified: ms=0 (no simple bounds, all constraints are general).
 */

#include "moreau/solver/active_set/core.hpp"
#include "moreau/solver/active_set/factorization.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>

namespace moreau {

// ============================================================================
// Constraint management
// ============================================================================

void daqp_remove_constraint(DaqpWorkspace* work, const int rm_ind) {
    // Update data structures
    work->sense[work->WS[rm_ind]] &= ~DAQP_ACTIVE;  // SET_INACTIVE
    daqp_update_LDL_remove(work, rm_ind);
    work->n_active--;

    for (int i = rm_ind; i < work->n_active; i++) {
        work->WS[i] = work->WS[i + 1];
        work->lam[i] = work->lam[i + 1];
    }

    // Can only reuse work less than the ind that was removed
    if (rm_ind < work->reuse_ind)
        work->reuse_ind = rm_ind;

    // Check if the removal led to singularity
    if (work->n_active > 0 && work->D[work->n_active - 1] < work->settings.sing_tol) {
        work->sing_ind = work->n_active - 1;
        work->D[work->n_active - 1] = 0;
    } else {
        daqp_pivot_last(work);
    }
}

void daqp_add_constraint(DaqpWorkspace* work, const int add_ind, double lam) {
    work->sense[add_ind] |= DAQP_ACTIVE;
    daqp_update_LDL_add(work, add_ind);
    work->WS[work->n_active] = add_ind;
    work->lam[work->n_active] = lam;
    work->n_active++;

    daqp_pivot_last(work);
}

void daqp_pivot_last(DaqpWorkspace* work) {
    const int rm_ind = work->n_active - 2;
    if (work->n_active > 1 &&
            work->D[rm_ind] < work->settings.pivot_tol &&
            work->D[rm_ind] < work->D[work->n_active - 1]) {
        const int ind_old = work->WS[rm_ind];
        const double lam_old = work->lam[rm_ind];
        daqp_remove_constraint(work, rm_ind);  // may recursively call pivot_last

        if (work->sing_ind != DAQP_EMPTY_IND) return;

        daqp_add_constraint(work, ind_old, lam_old);
    }
}

int daqp_activate_constraints(DaqpWorkspace* work) {
    for (int i = 0; i < work->m; i++) {
        if (work->sense[i] & DAQP_ACTIVE) {
            if (work->sense[i] & DAQP_LOWER)
                daqp_add_constraint(work, i, -1.0);
            else
                daqp_add_constraint(work, i, 1.0);
        }
        if (work->sing_ind != DAQP_EMPTY_IND) {
            int exitflag = 1;
            for (; i < work->m; i++) {
                if (work->sense[i] & DAQP_ACTIVE) {
                    if (work->sense[i] & DAQP_IMMUTABLE)
                        exitflag = DAQP_EXIT_OVERDETERMINED_INITIAL;
                    else
                        work->sense[i] &= ~DAQP_ACTIVE;
                }
            }
            work->n_active--;
            work->sing_ind = DAQP_EMPTY_IND;
            return exitflag;
        }
    }
    return 1;
}

// ============================================================================
// CSP and direction computation
// ============================================================================

void daqp_compute_CSP(DaqpWorkspace* work) {
    int i, j, disp;
    double sum;

    // Forward substitution (xi <-- L\d)
    for (i = work->reuse_ind, disp = daqp_arsum(work->reuse_ind); i < work->n_active; i++) {
        // Setup RHS
        if (work->sense[work->WS[i]] & DAQP_LOWER)
            sum = -work->dlower[work->WS[i]];
        else
            sum = -work->dupper[work->WS[i]];

        for (j = 0; j < i; j++)
            sum -= work->L[disp++] * work->xldl[j];
        disp++;  // Skip 1 in L
        work->xldl[i] = sum;
    }

    // Scale with D (zi = xi/di)
    for (i = work->reuse_ind; i < work->n_active; i++)
        work->zldl[i] = work->xldl[i] / work->D[i];

    // Backward substitution (lam_star <-- L'\z)
    int start_disp = daqp_arsum(work->n_active) - 1;
    for (i = work->n_active - 1; i >= 0; i--) {
        sum = work->zldl[i];
        disp = start_disp--;
        for (j = work->n_active - 1; j > i; j--) {
            sum -= work->lam_star[j] * work->L[disp];
            disp -= j;
        }
        work->lam_star[i] = sum;
    }
    work->reuse_ind = work->n_active;
}

void daqp_compute_singular_direction(DaqpWorkspace* work) {
    int i, j, disp;
    const int offset_L = daqp_arsum(work->sing_ind);
    int start_disp = offset_L - 1;

    // Backward substitution (p_tilde <-- L'\(-l))
    for (i = work->sing_ind - 1; i >= 0; i--) {
        work->lam_star[i] = -work->L[offset_L + i];
        disp = start_disp--;
        for (j = work->sing_ind - 1; j > i; j--) {
            work->lam_star[i] -= work->lam_star[j] * work->L[disp];
            disp -= j;
        }
    }
    work->lam_star[work->sing_ind] = 1;

    // Flip to ensure descent direction
    if (work->sense[work->WS[work->sing_ind]] & DAQP_LOWER)
        for (i = 0; i <= work->sing_ind; i++)
            work->lam_star[i] = -work->lam_star[i];
}

// ============================================================================
// Feasibility checks
// ============================================================================

int daqp_remove_blocking(DaqpWorkspace* work) {
    int i, rm_ind = DAQP_EMPTY_IND;
    double alpha = DAQP_INF;
    double alpha_cand;
    const double dual_tol = work->settings.dual_tol;

    for (i = 0; i < work->n_active; i++) {
        if (work->sense[work->WS[i]] & DAQP_IMMUTABLE) continue;

        if (work->sense[work->WS[i]] & DAQP_LOWER) {
            if (work->lam_star[i] < dual_tol) continue;  // lam <= 0 for lower -> dual feasible
        } else if (work->lam_star[i] > -dual_tol) continue;  // lam >= 0 for upper -> dual feasible

        if (work->sing_ind == DAQP_EMPTY_IND)
            alpha_cand = -work->lam[i] / (work->lam_star[i] - work->lam[i]);
        else
            alpha_cand = -work->lam[i] / work->lam_star[i];

        if (alpha_cand < alpha) {
            alpha = alpha_cand;
            rm_ind = i;
        }
    }

    if (rm_ind == DAQP_EMPTY_IND) return 0;  // Dual feasible or primal infeasible

    // Update lambda
    if (work->sing_ind == DAQP_EMPTY_IND)
        for (i = 0; i < work->n_active; i++)
            work->lam[i] += alpha * (work->lam_star[i] - work->lam[i]);
    else
        for (i = 0; i < work->n_active; i++)
            work->lam[i] += alpha * work->lam_star[i];

    // Remove the constraint
    work->sing_ind = DAQP_EMPTY_IND;
    daqp_remove_constraint(work, rm_ind);
    return 1;
}

void daqp_compute_primal_and_fval(DaqpWorkspace* work) {
    int i, j, disp, id;
    double fval = 0.0;

    // Reset u
    for (j = 0; j < work->n; j++)
        work->u[j] = 0;
    work->soft_slack = 0;

    // u <-- -M' * lam_star
    for (i = 0; i < work->n_active; i++) {
        id = work->WS[i];
        for (j = 0, disp = work->n * id; j < work->n; j++)
            work->u[j] -= work->M[disp++] * work->lam_star[i];
    }

    // fval = ||u||^2
    for (j = 0; j < work->n; j++)
        fval += work->u[j] * work->u[j];
    work->fval = fval;
}

int daqp_add_infeasible(DaqpWorkspace* work) {
    int j, disp;
    const double ep = -work->settings.primal_tol;
    double min_val = 0.0;
    double bound;
    double Mu, min_cand;
    int isupper = 0, add_ind = DAQP_EMPTY_IND;

    // General two-sided constraints (all constraints, since ms=0)
    for (j = 0, disp = 0; j < work->m; j++) {
        // Never activate immutable or already active constraints
        if (work->sense[j] & (DAQP_ACTIVE | DAQP_IMMUTABLE)) {
            disp += work->n;
            continue;
        }
        Mu = daqp_dot(work->M + disp, work->u, work->n);
        disp += work->n;
        bound = (work->scaling != nullptr) ? ep * work->scaling[j] : ep;

        min_cand = work->dupper[j] - Mu;
        if (min_cand < min_val && min_cand < bound) {
            add_ind = j;
            isupper = 1;
            min_val = min_cand;
        } else {
            min_cand = Mu - work->dlower[j];
            if (min_cand < min_val && min_cand < bound) {
                add_ind = j;
                isupper = 0;
                min_val = min_cand;
            }
        }
    }

    // No constraint is infeasible
    if (add_ind == DAQP_EMPTY_IND) return 0;

    // Set bound direction
    if (isupper)
        work->sense[add_ind] &= ~DAQP_LOWER;  // SET_UPPER
    else
        work->sense[add_ind] |= DAQP_LOWER;    // SET_LOWER

    // Swap lam and lam_star
    double* swp_ptr = work->lam;
    work->lam = work->lam_star;
    work->lam_star = swp_ptr;

    // Add the constraint
    daqp_add_constraint(work, add_ind, isupper ? 1.0 : -1.0);
    return 1;
}

// ============================================================================
// Iterative refinement
// ============================================================================

void daqp_refine_active(DaqpWorkspace* work) {
    int i, j, disp, id;
    double sum, Mu, d;

    // Compute -r[i] = -(M_i*u - d_i) and store in xldl[i]
    for (i = 0; i < work->n_active; i++) {
        id = work->WS[i];
        Mu = 0;
        for (j = 0, disp = work->n * id; j < work->n; j++)
            Mu += work->M[disp++] * work->u[j];
        d = (work->sense[id] & DAQP_LOWER) ? work->dlower[id] : work->dupper[id];
        work->xldl[i] = Mu - d;
    }

    // Forward substitution L * y = xldl
    for (i = 0, disp = 0; i < work->n_active; i++) {
        sum = work->xldl[i];
        for (j = 0; j < i; j++)
            sum -= work->L[disp++] * work->xldl[j];
        disp++;
        work->xldl[i] = sum;
    }

    // Scale by D^{-1}
    for (i = 0; i < work->n_active; i++)
        work->zldl[i] = work->xldl[i] / work->D[i];

    // Backward substitution L' * delta_lam = zldl -> stored in xldl
    {
        int start_disp = daqp_arsum(work->n_active) - 1;
        for (i = work->n_active - 1; i >= 0; i--) {
            sum = work->zldl[i];
            disp = start_disp--;
            for (j = work->n_active - 1; j > i; j--) {
                sum -= work->xldl[j] * work->L[disp];
                disp -= j;
            }
            work->xldl[i] = sum;
        }
    }

    // Update lam_star += delta_lam
    for (i = 0; i < work->n_active; i++)
        work->lam_star[i] += work->xldl[i];

    // Update u -= M'*delta_lam and recompute fval
    for (i = 0; i < work->n_active; i++) {
        double dlam = work->xldl[i];
        id = work->WS[i];
        for (j = 0, disp = work->n * id; j < work->n; j++)
            work->u[j] -= work->M[disp++] * dlam;
    }

    // Recompute fval
    double fval = work->soft_slack;
    for (j = 0; j < work->n; j++)
        fval += work->u[j] * work->u[j];
    work->fval = fval;
}

// ============================================================================
// Main iteration loop
// ============================================================================

int daqp_ldp(DaqpWorkspace* work) {
    int exitflag = DAQP_EXIT_ITERLIMIT;
    int tried_repair = 0, cycle_counter = 0;
    double best_fval = -1;
    const double fval_bound = 2 * work->settings.fval_bound;
    int prev_n_active = work->n_active;

    const double time_limit = work->settings.time_limit;
    const bool check_time = (time_limit < 1e29);
    auto t_start = check_time ? std::chrono::steady_clock::now()
                              : std::chrono::steady_clock::time_point{};

    int iter;
    for (iter = 1; iter < work->settings.iter_limit; ++iter) {
        // Periodic time-limit check (every 64 iters to avoid clock overhead)
        if (check_time && (iter & 63) == 0) {
            auto elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_start).count();
            if (elapsed >= time_limit) {
                exitflag = DAQP_EXIT_TIMELIMIT;
                break;
            }
        }
        if (work->sing_ind == DAQP_EMPTY_IND) {
            daqp_compute_CSP(work);

            // Check dual feasibility
            if (!daqp_remove_blocking(work)) {
                daqp_compute_primal_and_fval(work);

                // Infeasibility check
                if (work->fval > fval_bound) {
                    exitflag = DAQP_EXIT_INFEASIBLE;
                    break;
                }

                // Try to add most violated constraint
                if (!daqp_add_infeasible(work)) {
                    // All KKT conditions satisfied -> optimum

                    double min_D = work->D[0];
                    for (int i = 1; i < work->n_active; i++)
                        if (work->D[i] < min_D) min_D = work->D[i];

                    // Refactorize if ill-conditioned
                    if (work->n_active > 2 && tried_repair != 1 &&
                            min_D < work->settings.refactor_tol) {
                        tried_repair = 1;
                        for (int i = 0; i < work->n_active; i++) {
                            if (work->lam[i] >= 0)
                                work->sense[work->WS[i]] &= ~DAQP_LOWER;
                            else
                                work->sense[work->WS[i]] |= DAQP_LOWER;
                        }
                        work->sing_ind = DAQP_EMPTY_IND;
                        work->n_active = 0;
                        work->reuse_ind = 0;
                        daqp_activate_constraints(work);
                        continue;
                    }

                    // Iterative refinement if near-singular
                    if (work->n_active > 0 && min_D < work->settings.pivot_tol)
                        daqp_refine_active(work);

                    exitflag = DAQP_EXIT_OPTIMAL;
                    break;
                }

                // Cycle guard
                if (work->fval - best_fval < work->settings.progress_tol) {
                    if (cycle_counter++ > work->settings.cycle_tol) {
                        if (tried_repair == 1) {
                            exitflag = DAQP_EXIT_CYCLE;
                            break;
                        } else {
                            tried_repair = 1;
                            work->sing_ind = DAQP_EMPTY_IND;
                            work->n_active = 0;
                            work->reuse_ind = 0;
                            daqp_activate_constraints(work);
                            cycle_counter = 0;
                            best_fval = -1;
                        }
                    }
                } else {
                    best_fval = work->fval;
                    cycle_counter = 0;
                }
            }
        } else {
            // Singular case
            daqp_compute_singular_direction(work);
            if (!daqp_remove_blocking(work)) {
                exitflag = DAQP_EXIT_INFEASIBLE;
                break;
            }
        }

        // Per-iteration callback
        if (work->iter_callback) {
            int delta = work->n_active - prev_n_active;
            int added = delta > 0 ? delta : 0;
            int removed = delta < 0 ? -delta : 0;
            work->iter_callback(iter, work->n_active, work->fval,
                                added, removed, work->iter_callback_data);
            prev_n_active = work->n_active;
        }
    }

    work->iterations = iter;
    return exitflag;
}

void ldp2qp_solution(DaqpWorkspace* work) {
    int i, j, disp;

    // x* = Rinv*(u-v)
    if (work->v != nullptr)
        for (i = 0; i < work->n; i++) work->x[i] = work->u[i] - work->v[i];
    else
        for (i = 0; i < work->n; i++) work->x[i] = work->u[i];

    if (work->Rinv != nullptr) {
        // Upper-triangular packed multiply
        for (i = 0, disp = 0; i < work->n; i++) {
            work->x[i] *= work->Rinv[disp++];
            for (j = i + 1; j < work->n; j++)
                work->x[i] += work->Rinv[disp++] * work->x[j];
        }
    } else if (work->RinvD != nullptr) {
        for (i = 0; i < work->n; i++)
            work->x[i] *= work->RinvD[i];
    }

    // Scale dual variables back
    if (work->scaling != nullptr) {
        for (i = 0; i < work->n_active; i++)
            work->lam_star[i] *= work->scaling[work->WS[i]];
    }
}

} // namespace moreau
