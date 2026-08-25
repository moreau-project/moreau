/**
 * @file transform.cpp
 * @brief QP-to-LDP transformation
 *
 * Derived from DAQP (https://github.com/darnstrom/daqp),
 * Copyright (c) 2022 Daniel Arnström, licensed under the MIT License.
 * See the repository NOTICE file for the full license text.
 *
 * 
 * Simplified: ms=0 (no simple bounds), no AVI, no hierarchy.
 */

#include "moreau/solver/active_set/transform.hpp"
#include "moreau/solver/active_set/core.hpp"
#include "moreau/solver/active_set/factorization.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace moreau {

void csr_to_dense(double* dense, int64_t rows, int64_t cols,
                  const int64_t* row_offsets, const int64_t* col_indices,
                  const double* values, bool symmetric) {
    std::memset(dense, 0, rows * cols * sizeof(double));
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t k = row_offsets[i]; k < row_offsets[i + 1]; k++) {
            int64_t j = col_indices[k];
            dense[i * cols + j] = values[k];
            if (symmetric && i != j) {
                dense[j * cols + i] = values[k];
            }
        }
    }
}

int daqp_update_Rinv(DaqpWorkspace* work, const double* H) {
    const int n = work->n;
    const double zero_tol = work->settings.zero_tol;
    const double eps = work->settings.eps_prox;

    // Reset proximal mask
    if (work->prox_mask != nullptr) {
        for (int i = 0; i < n; i++) work->prox_mask[i] = 0;
    }
    work->n_prox = 0;

    // Check if diagonal
    bool is_diagonal = true;
    for (int i = 0; i < n && is_diagonal; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && (H[i * n + j] > zero_tol || H[i * n + j] < -zero_tol)) {
                is_diagonal = false;
                break;
            }
        }
    }

    // Diagonal case
    if (is_diagonal) {
        if (work->Rinv != nullptr) {
            work->RinvD = work->Rinv;
            work->Rinv = nullptr;
        }
        for (int i = 0; i < n; i++) {
            double Hi = H[i * n + i];
            if (Hi <= zero_tol) {
                if (work->prox_mask != nullptr) work->prox_mask[i] = 1;
                work->n_prox++;
                Hi += eps;
            }
            if (Hi <= zero_tol) return DAQP_EXIT_NONCONVEX;
            work->RinvD[i] = 1.0 / std::sqrt(Hi);
        }
        return 1;
    }

    // General case: ensure Rinv points to allocated data
    if (work->RinvD != nullptr) {
        work->Rinv = work->RinvD;
        work->RinvD = nullptr;
    }

    // Pack H into upper-triangular Rinv (symmetrize)
    int disp = 0;
    for (int i = 0; i < n; i++) {
        work->Rinv[disp++] = H[i * n + i];
        for (int j = i + 1; j < n; j++)
            work->Rinv[disp++] = 0.5 * (H[i * n + j] + H[j * n + i]);
    }

    // Cholesky factorization (packed upper-triangular)
    int disp2;
    for (int i = 0, disp = 0; i < n; disp += n - i, i++) {
        double diag_i = work->Rinv[disp];
        for (int k = 0, disp2 = i; k < i; k++, disp2 += n - k)
            diag_i -= work->Rinv[disp2] * work->Rinv[disp2];
        if (diag_i <= zero_tol) {
            if (work->prox_mask != nullptr) work->prox_mask[i] = 1;
            work->n_prox++;
            diag_i += eps;
        }
        if (diag_i <= zero_tol) return DAQP_EXIT_NONCONVEX;
        diag_i = 1.0 / std::sqrt(diag_i);
        for (int j = 1; j < n - i; j++) {
            for (int k = 0, disp2 = i; k < i; k++, disp2 += n - k)
                work->Rinv[disp + j] -= work->Rinv[disp2] * work->Rinv[disp2 + j];
            work->Rinv[disp + j] *= diag_i;
        }
        work->Rinv[disp] = diag_i;
    }

    // R -> Rinv (in-place inversion of upper-triangular)
    for (int k = 0, disp = 0; k < n; k++) {
        disp2 = disp;
        work->Rinv[disp] = work->Rinv[disp2++];
        for (int j = k + 1; j < n; j++) work->Rinv[disp2++] *= -work->Rinv[disp];
        disp++;
        for (int i = k + 1; i < n; i++, disp++) {
            work->Rinv[disp] *= work->Rinv[disp2++];
            for (int j = 1; j < n - i; j++)
                work->Rinv[disp + j] -= work->Rinv[disp2++] * work->Rinv[disp];
        }
    }
    return 1;
}

int daqp_update_M(DaqpWorkspace* work, const double* A) {
    const int n = work->n;
    const int m = work->m;

    // Zero M before accumulation — required because the packed Rinv path
    // uses += to accumulate off-diagonal contributions.
    std::memset(work->M, 0, m * n * sizeof(double));

    if (work->Rinv != nullptr) {
        // M = A * Rinv (using packed upper-triangular Rinv)
        for (int k = 0, disp2 = n * m - 1; k < m; k++, disp2 -= n) {
            int disp = daqp_arsum(n);
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < j; i++)
                    work->M[disp2 - i] += work->Rinv[--disp] * A[disp2 - j];
                work->M[disp2 - j] = work->Rinv[--disp] * A[disp2 - j];
            }
        }
    } else if (work->RinvD != nullptr) {
        // Diagonal case: M[i,j] = A[i,j] * RinvD[j]
        for (int k = 0, disp = 0; k < m; k++) {
            for (int i = 0; i < n; i++, disp++)
                work->M[disp] = A[disp] * work->RinvD[i];
        }
    } else {
        // Identity case: M = A
        std::memcpy(work->M, A, m * n * sizeof(double));
    }

    // Reset workspace state
    work->sing_ind = DAQP_EMPTY_IND;
    work->n_active = 0;
    work->reuse_ind = 0;

    return daqp_normalize_M(work);
}

int daqp_normalize_M(DaqpWorkspace* work) {
    const int n = work->n;
    const double zero_tol = work->settings.zero_tol;

    for (int i = 0, disp = 0; i < work->m; i++) {
        double scaling_i = 0;
        for (int j = 0; j < n; disp++, j++)
            scaling_i += work->M[disp] * work->M[disp];

        if (scaling_i < zero_tol) {
            // Zero-row constraint: A[i,:]*Rinv ≈ 0, so the constraint
            // degenerates to 0*u = d.  Check if bounds are feasible
            // before discarding.  At this point dupper/dlower still hold
            // the original (unscaled, pre-daqp_update_d) bounds from
            // buildBounds.
            double du = work->dupper[i];
            double dl = work->dlower[i];
            if (du < -work->settings.primal_tol ||
                dl >  work->settings.primal_tol) {
                // 0 <= du is violated (upper) or 0 >= dl is violated (lower)
                return DAQP_EXIT_INFEASIBLE;
            }
            work->sense[i] = DAQP_IMMUTABLE;
            continue;
        }

        scaling_i = 1.0 / std::sqrt(scaling_i);
        work->scaling[i] = scaling_i;
        for (int j = 0, d = disp - n; j < n; j++, d++)
            work->M[d] *= scaling_i;
    }
    return 0;
}

void daqp_update_v(const double* f, DaqpWorkspace* work) {
    const int n = work->n;
    if (work->v == nullptr || f == nullptr) return;

    if (work->Rinv == nullptr) {
        if (work->RinvD != nullptr)
            for (int i = 0; i < n; i++) work->v[i] = f[i] * work->RinvD[i];
        else
            for (int i = 0; i < n; i++) work->v[i] = f[i];
        return;
    }

    // v = Rinv' * f (using packed upper-triangular Rinv)
    std::memset(work->v, 0, n * sizeof(double));
    int disp = daqp_arsum(n);
    for (int j = n - 1; j >= 0; j--) {
        for (int i = n - 1; i > j; i--)
            work->v[i] += work->Rinv[--disp] * f[j];
        work->v[j] = work->Rinv[--disp] * f[j];
    }
}

void daqp_update_d(DaqpWorkspace* work, const double* bupper, const double* blower) {
    const int n = work->n;
    work->reuse_ind = 0;

    // Apply scaling to bounds
    if (work->scaling != nullptr) {
        for (int i = 0; i < work->m; i++) {
            work->dupper[i] = bupper[i] * work->scaling[i];
            work->dlower[i] = blower[i] * work->scaling[i];
        }
    } else {
        for (int i = 0; i < work->m; i++) {
            work->dupper[i] = bupper[i];
            work->dlower[i] = blower[i];
        }
    }

    if (work->v == nullptr) return;

    // d += M*v
    for (int i = 0, disp = 0; i < work->m; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++)
            sum += work->M[disp++] * work->v[j];
        work->dupper[i] += sum;
        work->dlower[i] += sum;
    }
}

int daqp_check_bounds(DaqpWorkspace* work, const double* bupper, const double* blower) {
    int do_activate = 0;
    for (int i = 0; i < work->m; i++) {
        if (work->sense[i] & DAQP_IMMUTABLE) continue;
        double diff = bupper[i] - blower[i];
        if (diff < -work->settings.primal_tol) {
            return DAQP_EXIT_INFEASIBLE;
        } else if (diff < work->settings.zero_tol) {
            // Equality constraint
            work->sense[i] |= (DAQP_ACTIVE | DAQP_IMMUTABLE);
            do_activate = 1;
        }
    }
    return do_activate;
}

int daqp_setup_ldp(DaqpWorkspace* work,
                   const double* H, const double* f,
                   const double* A,
                   const double* bupper, const double* blower) {
    int error_flag;

    // 1. Check bounds for equalities and trivial infeasibility
    error_flag = daqp_check_bounds(work, bupper, blower);
    if (error_flag < 0) return error_flag;
    bool do_activate = (error_flag == 1);

    // 2. Cholesky of H → Rinv
    error_flag = daqp_update_Rinv(work, H);
    if (error_flag < 0) return error_flag;

    // 3. v = Rinv' * f
    daqp_update_v(f, work);

    // 4. M = A * Rinv + normalize
    error_flag = daqp_update_M(work, A);
    if (error_flag < 0) return error_flag;

    // 5. d = b + M*v (with scaling)
    daqp_update_d(work, bupper, blower);

    // 6. Activate equality constraints
    if (do_activate) {
        work->sing_ind = DAQP_EMPTY_IND;
        work->n_active = 0;
        work->reuse_ind = 0;
        error_flag = daqp_activate_constraints(work);
        if (error_flag < 0) return error_flag;
    }

    return 0;
}

void dense_to_csr_values(const double* dense, int64_t rows, int64_t cols,
                         const int64_t* row_offsets, const int64_t* col_indices,
                         double* values, bool symmetric) {
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t k = row_offsets[i]; k < row_offsets[i + 1]; k++) {
            int64_t j = col_indices[k];
            if (symmetric && i != j) {
                // dH = -v1 * x' is an asymmetric outer product. Since P is
                // full symmetric (P[i,j] == P[j,i]), changing one CSR entry
                // affects both, so the gradient is dH[i,j] + dH[j,i].
                values[k] = dense[i * cols + j] + dense[j * cols + i];
            } else {
                values[k] = dense[i * cols + j];
            }
        }
    }
}

} // namespace moreau
