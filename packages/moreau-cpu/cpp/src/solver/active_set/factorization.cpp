/**
 * @file factorization.cpp
 * @brief LDL factorization with rank-1 updates
 *
 * Derived from DAQP (https://github.com/darnstrom/daqp),
 * Copyright (c) 2022 Daniel Arnström, licensed under the MIT License.
 * See the repository NOTICE file for the full license text.
 *
 * 
 * Stripped: soft weights, binary constraints, AVI.
 */

#include "moreau/solver/active_set/factorization.hpp"

namespace moreau {

double daqp_dot(const double* v1, const double* v2, const int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += v1[i] * v2[i];
    return sum;
}

void daqp_update_LDL_add(DaqpWorkspace* work, const int add_ind) {
    work->sing_ind = DAQP_EMPTY_IND;
    int i, j, disp;
    const int new_L_start = daqp_arsum(work->n_active);
    double sum;
    double* Mi;

    // All constraints are general (ms=0)
    Mi = work->M + work->n * add_ind;

    // di <-- Mi' Mi (always 1 if normalized)
    sum = 0.0;
    for (i = 0; i < work->n; i++)
        sum += Mi[i] * Mi[i];
    work->D[work->n_active] = sum;

    if (work->n_active == 0) return;

    // store l <-- Mk' * Mi for each active constraint k
    for (i = 0; i < work->n_active; i++) {
        const int id = work->WS[i];
        double* Mk = work->M + work->n * id;
        work->L[new_L_start + i] = daqp_dot(Mk, Mi, work->n);
    }

    // Forward substitution: l <-- L \ (Mk*m)
    for (i = 0, disp = 0; i < work->n_active; i++) {
        sum = work->L[new_L_start + i];
        for (j = 0; j < i; j++)
            sum -= work->L[disp++] * work->L[new_L_start + j];
        work->L[new_L_start + i] = sum;
        disp++; // Skip diagonal (which is 1)
    }

    // Scale: l_i <-- l_i/d_i and update d_new -= l'Dl
    sum = work->D[work->n_active];
    double tmp;
    for (i = 0, disp = new_L_start; i < work->n_active; i++, disp++) {
        tmp = work->L[disp];
        work->L[disp] /= work->D[i];
        sum -= tmp * work->L[disp];
    }
    work->D[work->n_active] = sum;

    // Check for singularity
    if (work->D[work->n_active] < work->settings.sing_tol ||
            work->n_active >= work->n) {
        work->sing_ind = work->n_active;
        work->D[work->n_active] = 0;
    }
}

void daqp_update_LDL_remove(DaqpWorkspace* work, const int rm_ind) {
    if (work->n_active == rm_ind + 1)
        return;

    int i, j, r, old_disp, new_disp, w_count;
    const int n_update = work->n_active - rm_ind - 1;
    double* w = &work->zldl[rm_ind]; // zldl will be obsolete => reuse to avoid allocation

    // Extract parts to keep/update in L & D
    new_disp = daqp_arsum(rm_ind);
    old_disp = new_disp + (rm_ind + 1);
    w_count = 0;

    // Remove column rm_ind: copy row i into i-1
    for (i = rm_ind + 1; i < work->n_active; old_disp++, new_disp++, i++) {
        for (j = 0; j < i; j++) {
            if (j != rm_ind)
                work->L[new_disp++] = work->L[old_disp++];
            else
                w[w_count++] = work->L[old_disp++];
        }
    }

    // Algorithm C1 in Gill 1974 for low-rank update of LDL
    double p, beta, dbar, alpha = work->D[rm_ind];
    old_disp = daqp_arsum(rm_ind) + rm_ind;
    for (j = 0, i = rm_ind + 1; j < n_update; j++, i++) {
        p = w[j];
        dbar = work->D[i] + alpha * p * p;
        work->D[i - 1] = dbar;

        beta = p * alpha / dbar;
        alpha = work->D[i] * alpha / dbar;

        old_disp += i;
        for (r = j + 1, new_disp = old_disp + j; r < n_update; r++) {
            w[r] -= p * work->L[new_disp];
            work->L[new_disp] += beta * w[r];
            new_disp += rm_ind + r + 1;
        }
    }
}

} // namespace moreau
