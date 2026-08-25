/**
 * @file transform.hpp
 * @brief QP-to-LDP transformation and moreau format conversion
 *
 * Converts moreau's conic form (P, q, A, b, cones) to the
 * internal LDP form (Rinv, M, v, d), and maps solutions back.
 */

#pragma once

#include "moreau/solver/active_set/types.hpp"
#include <cstdint>
#include <vector>

namespace moreau {

/**
 * @brief Compute Rinv from dense H (Cholesky + inversion)
 * @return 1 on success, negative exit flag on error
 */
int daqp_update_Rinv(DaqpWorkspace* work, const double* H);

/**
 * @brief Compute M = A * Rinv (constraint transformation)
 * @return 0 on success, negative on error
 */
int daqp_update_M(DaqpWorkspace* work, const double* A);

/**
 * @brief Compute v = Rinv' * f (linear term transformation)
 */
void daqp_update_v(const double* f, DaqpWorkspace* work);

/**
 * @brief Compute d = b + M*v (bound transformation)
 */
void daqp_update_d(DaqpWorkspace* work, const double* bupper, const double* blower);

/**
 * @brief Normalize M rows to unit length, store scaling factors
 * @return 0 on success, negative on error
 */
int daqp_normalize_M(DaqpWorkspace* work);

/**
 * @brief Check for equality constraints (bupper ≈ blower) and mark them
 * @return 1 if equalities found (need activation), 0 otherwise, negative on error
 */
int daqp_check_bounds(DaqpWorkspace* work, const double* bupper, const double* blower);

/**
 * @brief Full QP-to-LDP setup: Rinv, M, v, d, normalize, check bounds
 *
 * Given dense H [n×n row-major], f [n], A [m×n row-major], bupper [m], blower [m]:
 * 1. Cholesky of H → Rinv
 * 2. M = A * Rinv
 * 3. Normalize M
 * 4. v = Rinv' * f
 * 5. d = b + M*v (with scaling)
 * 6. Check bounds for equalities
 * 7. Activate equality constraints
 *
 * @return 0 on success, negative on error
 */
int daqp_setup_ldp(DaqpWorkspace* work,
                   const double* H, const double* f,
                   const double* A,
                   const double* bupper, const double* blower);

/**
 * @brief Convert sparse CSR matrix to dense row-major format
 *
 * @param dense Output dense matrix [rows × cols], row-major
 * @param rows Number of rows
 * @param cols Number of columns
 * @param row_offsets CSR row offsets [rows+1]
 * @param col_indices CSR column indices [nnz]
 * @param values CSR values [nnz]
 * @param symmetric If true, mirror upper triangle to lower (for P matrix)
 */
void csr_to_dense(double* dense, int64_t rows, int64_t cols,
                  const int64_t* row_offsets, const int64_t* col_indices,
                  const double* values, bool symmetric = false);

/**
 * @brief Extract sparse CSR values from a dense row-major matrix
 *
 * Inverse of csr_to_dense: reads values at CSR-specified positions.
 *
 * @param dense Input dense matrix [rows × cols], row-major
 * @param rows Number of rows
 * @param cols Number of columns
 * @param row_offsets CSR row offsets [rows+1]
 * @param col_indices CSR column indices [nnz]
 * @param values Output CSR values [nnz]
 * @param symmetric If true, for off-diagonal entries (i,j), sum dense[i][j] + dense[j][i]
 *                  to account for P being stored as full symmetric
 */
void dense_to_csr_values(const double* dense, int64_t rows, int64_t cols,
                         const int64_t* row_offsets, const int64_t* col_indices,
                         double* values, bool symmetric = false);

} // namespace moreau
