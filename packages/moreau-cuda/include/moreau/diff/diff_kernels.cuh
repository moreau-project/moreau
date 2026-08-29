/**
 * @file kernels.cuh
 * @brief CUDA kernel declarations for differentiation
 *
 * Contains all CUDA kernel declarations for forward and backward
 * differentiation of conic optimization problems.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace moreau {

// ============================================================================
// Cone Projection Kernels
// ============================================================================

/**
 * @brief Project onto dual of zero cone (identity)
 */
void project_zero_cone_dual(
    double* pi_u,
    const double* u,
    int64_t numZeroCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
);

/**
 * @brief Project onto dual of nonnegative cone: pi_u = max(u, 0)
 */
void project_nonneg_cone_dual(
    double* pi_u,
    const double* u,
    int64_t offset,
    int64_t numNonnegCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
);

/**
 * @brief Project onto dual of SOC (second-order cone, variable dimension)
 */
void project_soc_cone_dual(
    double* pi_u,
    const double* u,
    int64_t offset,
    int64_t numSocCones,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    int64_t totalSocDim,
    int64_t batchSize,
    int64_t m,
    const int64_t* d_soc_sz_offsets,
    cudaStream_t stream
);


/**
 * @brief Project onto dual of exponential cone
 */
void project_exp_cone_dual(
    double* pi_u,
    const double* u,
    int64_t offset,
    int64_t numExpCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
);

/**
 * @brief Project onto dual of power cone
 */
void project_power_cone_dual(
    double* pi_u,
    const double* u,
    const double* alphas,
    int64_t offset,
    int64_t numPowerCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
);

/**
 * @brief Project onto dual of generalized power cone
 */
void project_genpow_cone_dual(
    double* pi_u,
    const double* u,
    const double* alphas,
    const int64_t* d_dim1s,
    const int64_t* d_dim2s,
    const int64_t* d_offsets,
    const int64_t* d_alpha_offsets,
    int64_t offset,
    int64_t numGenPowerCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream,
    double* work_vec = nullptr,
    int64_t totalGenPowerDim = 0,
    const int64_t* d_sz_offsets = nullptr
);

// ============================================================================
// Cone Derivative Kernels
// ============================================================================

/**
 * @brief Compute derivative of nonnegative cone projection
 *
 * DΠ_K*(u) = diag(u > 0)
 */
void compute_nonneg_derivative(
    double* H_diag,
    const double* u,
    int64_t offset,
    int64_t numNonnegCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
);

/**
 * @brief Compute derivative of SOC projection (variable dimension)
 *
 * For each SOC cone of dim d: H is a d x d symmetric matrix stored as
 * upper triangle (d*(d+1)/2 elements). Total storage = totalSocHsEntries.
 */
void compute_soc_derivative(
    double* H,
    const double* u,
    int64_t offset,
    int64_t numSocCones,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_Hs_offsets,
    int64_t totalSocDim,
    int64_t totalSocHsEntries,
    int64_t batchSize,
    int64_t m,
    const int64_t* d_soc_sz_offsets,
    cudaStream_t stream
);

/**
 * @brief Compute derivative of SOC projection with sparse expansion (dim > 4)
 *
 * Splits dense (dim<=4) and sparse (dim>4) cones:
 * - Dense: upper triangle in H buffer (same as compute_soc_derivative)
 * - Sparse: diagonal + rank-2 decomposition (diag, v1, v2, c1, c2)
 */
void compute_soc_derivative_sparse(
    double* H,                        // Dense SOC Hs (dim<=4 only)
    const double* u,
    int64_t offset,
    int64_t numSocCones,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_Hs_dense_offsets, // dense-only prefix sum
    const int64_t* d_soc_sparse_indices,
    int64_t totalDenseSocHsEntries,
    double* sparse_diag,              // [batch][totalSparseSocDim]
    double* sparse_v1,                // [batch][totalSparseSocDim]
    double* sparse_v2,                // [batch][totalSparseSocDim]
    double* sparse_c1,                // [batch][numSparseSoc]
    double* sparse_c2,                // [batch][numSparseSoc]
    const int64_t* d_soc_sparse_offsets,
    int64_t totalSparseSocDim,
    int64_t numSparseSoc,
    int64_t batchSize,
    int64_t m,
    const int64_t* d_soc_sz_offsets,
    cudaStream_t stream,
    // Number of SOC cones with dim > SOC_PARALLEL_THRESHOLD; these are
    // processed by a block-per-cone kernel launched alongside the small path.
    int64_t numLargeSoc = 0
);

/**
 * @brief Compute derivative of exponential cone projection
 *
 * H is a 3x3 matrix stored as upper triangle (6 elements)
 */
void compute_exp_derivative(
    double* H,
    const double* u,
    int64_t offset,
    int64_t numExpCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
);

/**
 * @brief Compute derivative of power cone projection
 *
 * H is a 3x3 matrix stored as upper triangle (6 elements)
 */
void compute_power_derivative(
    double* H,
    const double* u,
    const double* alphas,
    int64_t offset,
    int64_t numPowerCones,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
);

/**
 * @brief Compute derivative of generalized power cone projection (sparse decomposition)
 *
 * Outputs diagonal + rank-3 decomposition instead of dense dim×dim matrix.
 * H = diag + left1*right1^T + left2*right2^T + c3*left3*left3^T
 * GenPowerCone is NOT self-dual, so terms 1-2 are asymmetric.
 */
void compute_genpow_derivative_sparse(
    double* sparse_diag,              // [batch][totalGenpowDim]
    double* sparse_left1,             // [batch][totalGenpowDim]
    double* sparse_right1,            // [batch][totalGenpowDim]
    double* sparse_left2,             // [batch][totalGenpowDim]
    double* sparse_right2,            // [batch][totalGenpowDim]
    double* sparse_left3,             // [batch][totalGenpowDim]
    double* sparse_c3,                // [batch][numGenPowerCones]
    const double* u,
    const double* alphas,
    const int64_t* d_dim1s,
    const int64_t* d_dim2s,
    const int64_t* d_offsets,
    const int64_t* d_alpha_offsets,
    int64_t offset,
    int64_t numGenPowerCones,
    int64_t totalGenpowDim,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream,
    double* work_vec = nullptr,
    double* work_dim1 = nullptr,
    int64_t totalGenPowerAlphas = 0,
    const int64_t* d_sz_offsets = nullptr
);

// ============================================================================
// Backward Pass Kernels
// ============================================================================

/**
 * @brief Compute u = z - s (batched)
 */
void compute_u_from_z_s(
    double* u,
    const double* z,
    const double* s,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Build adjoint RHS for HSDE system
 *
 * Transforms upstream gradients (dx_bar, dz_bar, ds_bar) into the
 * RHS for the augmented adjoint linear system.
 *
 * rhs[0:n]       = dx_bar
 * rhs[n:n+m]     = dz_bar + ds_bar  (w_bar)
 * rhs[n+m:n+2m]  = -ds_bar          (du_bar)
 * rhs[n+2m]      = -dx_bar'x - dz_bar'z - ds_bar's  (dt_bar)
 */
// Direct-x variant: also takes `xn` (total direct-x dim) and an upstream
// `dz_x_bar` (in equilibrated frame). When `dz_x_bar` is nullptr the
// direct-x slot of the RHS is filled with zeros (preserves the slack-only
// convention). Otherwise dz_x flows positively to BOTH the du_x slot
// and the primal-x slot at the corresponding `xcone_indices[k]`, and
// adds `-sum(dz_x_bar * z_x_eq)` to the τ row.
//
// `xcone_indices` is a [xn] flat list of primal indices J across all
// direct-x cones (concat in spec order). `z_x_eq` is the equilibrated
// direct-x dual at the converged solution.
//
// τ slot is at index n + 2m + xn.
void build_adjoint_rhs_hsde_with_xcones(
    double* rhs,
    const double* dx_bar,
    const double* dz_bar,
    const double* ds_bar,
    const double* dz_x_bar,
    const double* x,
    const double* z,
    const double* s,
    const double* z_x_eq,
    const int64_t* xcone_indices,
    int64_t n,
    int64_t m,
    int64_t xn,
    int64_t batchSize,
    cudaStream_t stream = 0
);

// Convert the IPM-internal direct-x dual `z_x_int` to user (original)
// frame, applying both τ-normalization and equilibration scaling:
//   z_x_user[b, k] = z_x_int[b, k] * d[J[k]] / (τ_raw[b] * c_scale[b]).
// Mirrors CPU `Variables::unscale` for z_x.
void unscale_z_x(
    double* z_x_user,
    const double* z_x_int,
    const double* dinv,
    const double* c_scale,
    const double* tau_raw,
    const int64_t* xcone_indices,
    int64_t n,
    int64_t total_xn,
    int64_t batchSize,
    cudaStream_t stream = 0
);

// Equilibrate user-frame `z_x_user` (the direct-x cone duals returned by
// `Solution`) to the equilibrated τ=1 frame stored in `DiffState.z_x`:
//   z_x_eq[b, k] = z_x_user[b, k] * c_scale[b] / d[J[k]]
// (Inverse of the unscale `z_x_user = z_x_eq * d[J] / c`.)
void equilibrate_z_x(
    double* z_x_eq,
    const double* z_x_user,
    const double* dinv,
    const double* c_scale,
    const int64_t* xcone_indices,
    int64_t n,
    int64_t xn,
    int64_t batchSize,
    cudaStream_t stream = 0
);

// Equilibrate user-frame `dz_x_bar` to the equilibrated frame used in
// IFT-direct backward:
//   dz_x_eq[b, k] = dz_x_bar[b, k] * d[J[k]] / c_scale[b]
// where d = 1 / dinv (per-batch equilibration scaling). Mirrors the CPU
// chain rule on `z_x_orig = z_x_eq * d[J] / c`.
void equilibrate_dz_x(
    double* dz_x_eq,
    const double* dz_x_bar,
    const double* dinv,
    const double* c_scale,
    const int64_t* xcone_indices,
    int64_t n,
    int64_t xn,
    int64_t batchSize,
    cudaStream_t stream = 0
);

void build_adjoint_rhs_hsde(
    double* rhs,
    const double* dx_bar,
    const double* dz_bar,
    const double* ds_bar,
    const double* x,
    const double* z,
    const double* s,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Build adjoint RHS for QP equality system
 *
 * rhs[0:n]   = dx_bar
 * rhs[n:n+m] = dz_bar
 */
void build_adjoint_rhs_qp_eq(
    double* rhs,
    const double* dx_bar,
    const double* dz_bar,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Compute HSDE coefficients c1, c2, c3
 *
 * c1[i] = -(2/τ) * (Px)[i] - q[i]
 * c2[i] = -b[i]
 * c3 = x'Px / τ²
 */
void compute_hsde_coefficients(
    double* c1,
    double* c2,
    double* c3,
    const double* Px,
    const double* x,
    const double* q,
    const double* b,
    const double* tau,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Extract gradients from adjoint solution (HSDE)
 *
 * dq_bar = -τ * λ₁ + λ₄ * x
 * db_bar = τ * λ₂ + λ₄ * z
 *
 * Note: Uses z (not Π_K*(u)) per diffqcp formula. The sign convention
 * is + because our λ₄ = -w₃ in diffqcp notation.
 */
// Direct-x variant: τ slot moves to index n + 2m + xn.
void extract_gradients_hsde_with_xcones(
    double* dq,
    double* db,
    const double* lambda,
    const double* x,
    const double* z,
    const double* tau,
    int64_t n,
    int64_t m,
    int64_t xn,
    int64_t batchSize,
    cudaStream_t stream = 0
);

void extract_gradients_hsde(
    double* dq,
    double* db,
    const double* lambda,  // Full solution [n + 2m + 1]
    const double* x,
    const double* z,       // Dual variable z (not pi_u)
    const double* tau,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Extract gradients from adjoint solution (QP equality)
 *
 * dq_bar = -λ_x
 * db_bar = λ_z
 */
void extract_gradients_qp_eq(
    double* dq,
    double* db,
    const double* lambda,  // Full solution [n + m]
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Extract gradients from equilibrated adjoint solution (QP equality)
 *
 * Combines extraction and equilibration transform in one kernel:
 * dq = -c * D * λ_x_eq = -c * λ_x_eq / dinv
 * db = E * λ_z_eq = λ_z_eq / einv
 */
void extract_gradients_qp_eq_with_equilibration(
    double* dq,
    double* db,
    const double* lambda,  // Full solution [n + m] in equilibrated space
    const double* dinv,
    const double* einv,
    const double* c,       // c_scale (scalar per batch)
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Transform dP gradient from equilibrated to original space
 *
 * dP[i,j] = dP_eq[i,j] * dinv[i] * dinv[j] / c
 */
void transform_dP_from_equilibrated(
    double* dP_values,     // In-place modification
    const int64_t* P_row_offsets,
    const int64_t* P_col_indices,
    const double* dinv,
    const double* c,
    int64_t n,
    int64_t nnzP,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Transform dA gradient from equilibrated to original space
 *
 * dA[i,j] = dA_eq[i,j] * einv[i] * dinv[j]
 */
void transform_dA_from_equilibrated(
    double* dA_values,     // In-place modification
    const int64_t* A_row_offsets,
    const int64_t* A_col_indices,
    const double* dinv,
    const double* einv,
    int64_t n,
    int64_t m,
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Compute P matrix gradient dP at sparse positions
 *
 * For HSDE: dP[i,j] = -τ * λ₁[i] * x[j] - τ * λ₁[j] * x[i]
 *                    + λ₄ * x[i] * x[j]  (symmetrized for diagonal)
 *
 * @param dP_values Output gradient values [batchSize][nnzP]
 * @param P_row_offsets CSR row offsets
 * @param P_col_indices CSR column indices
 * @param lambda Adjoint solution
 * @param x Primal solution
 * @param tau Homogenization scalar
 * @param n Number of variables
 * @param nnzP Number of nonzeros in P
 * @param batchSize Batch size
 * @param stream CUDA stream
 */
void compute_dP_gradient_hsde(
    double* dP_values,
    const int64_t* P_row_offsets,
    const int64_t* P_col_indices,
    const double* lambda,
    const double* x,
    const double* tau,
    int64_t n,
    int64_t m,
    int64_t nnzP,
    int64_t batchSize,
    cudaStream_t stream
);

// Direct-x variant: τ slot of `lambda` is at index n + 2m + xn.
void compute_dP_gradient_hsde_with_xcones(
    double* dP_values,
    const int64_t* P_row_offsets,
    const int64_t* P_col_indices,
    const double* lambda,
    const double* x,
    const double* tau,
    int64_t n,
    int64_t m,
    int64_t xn,
    int64_t nnzP,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Compute P matrix gradient dP for QP equality
 *
 * dP[i,j] = -λ_x[i] * x[j] (symmetrized)
 *
 * @param lambda_stride Stride between batches in lambda_x (typically n+m for sol layout)
 */
void compute_dP_gradient_qp_eq(
    double* dP_values,
    const int64_t* P_row_offsets,
    const int64_t* P_col_indices,
    const double* lambda_x,
    const double* x,
    int64_t n,
    int64_t lambda_stride,
    int64_t nnzP,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Compute A matrix gradient dA at sparse positions (HSDE)
 *
 * dA[i,j] = -λ₁[j] * z[i] - λ₂[i] * x[j]
 */
void compute_dA_gradient_hsde(
    double* dA_values,
    const int64_t* A_row_offsets,
    const int64_t* A_col_indices,
    const double* lambda,
    const double* x,
    const double* z,
    const double* tau,
    int64_t n,
    int64_t m,
    int64_t lambda_stride,  // stride between batches in lambda (n + 2*m + 1 for non-bounds, n + 2*m_aug + 1 for bounds)
    int64_t z_stride,       // stride between batches in z (m for non-bounds, m_aug for bounds)
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Compute A matrix gradient dA for QP equality
 *
 * dA[i,j] = -λ_z[i] * x[j] - λ_x[j] * z[i]
 *
 * @param sol Solution vector with layout [lam_x (n), lam_z (m)] per batch
 * @param sol_stride Stride between batches in sol (typically n+m)
 */
void compute_dA_gradient_qp_eq(
    double* dA_values,
    const int64_t* A_row_offsets,
    const int64_t* A_col_indices,
    const double* sol,
    const double* x,
    const double* z,
    int64_t n,
    int64_t m,
    int64_t sol_stride,
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream
);

// ============================================================================
// Equilibration Kernels for Differentiation
// ============================================================================

/**
 * @brief Transform upstream gradients from original to equilibrated space
 *
 * dx_eq = D * dx
 * dz_eq = E / c * dz
 * ds_eq = E^{-1} * ds
 */
void transform_upstream_grads_to_equilibrated(
    double* dx_eq,
    double* dz_eq,
    double* ds_eq,
    const double* dx,
    const double* dz,
    const double* ds,
    const double* d,
    const double* e,
    const double* einv,
    const double* c,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Transform output gradients from equilibrated to original space
 *
 * dP += c * D * dP_eq * D
 * dq += c * D * dq_eq
 * dA += E * dA_eq * D
 * db += E * db_eq
 */
void transform_output_grads_to_original(
    double* dP_values,
    double* dq,
    double* dA_values,
    double* db,
    const double* dP_eq,
    const double* dq_eq,
    const double* dA_eq,
    const double* db_eq,
    const int64_t* P_row_offsets,
    const int64_t* P_col_indices,
    const int64_t* A_row_offsets,
    const int64_t* A_col_indices,
    const double* d,
    const double* e,
    const double* c,
    int64_t n,
    int64_t m,
    int64_t nnzP,
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream
);

// ============================================================================
// Forward Pass Kernels
// ============================================================================

/**
 * @brief Build forward RHS for QP equality system
 *
 * The forward QP system solves:
 * [P  A'] [dx]   [-(dq + dP*x + dA'*z)]
 * [A  0 ] [dz] = [db - dA*x          ]
 */
void build_forward_rhs_qp_eq(
    double* rhs,
    const double* dP_x,      // dP * x [batchSize][n]
    const double* dAt_z,     // dA' * z [batchSize][n]
    const double* dA_x,      // dA * x [batchSize][m]
    const double* dq,        // [batchSize][n]
    const double* db,        // [batchSize][m]
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Compute x' * dP * x for each batch
 *
 * Uses x and dP*x (already computed) to compute the quadratic form.
 * Result is a scalar per batch.
 */
void compute_x_dP_x(
    double* result,          // [batchSize]
    const double* dP_x,      // dP * x [batchSize][n]
    const double* x,         // [batchSize][n]
    int64_t n,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Build forward RHS for HSDE system
 *
 * The forward HSDE system solves J * [dz_x; w; du; dt]' = [r1; r2; 0; r3]
 *
 * r1 = -(dP*x + dA'*z + dq*τ)
 * r2 = -(dA*x - db*τ)
 * r3 = x'*dP*x/τ + dq'*x + db'*z  (uses z to match backward pass)
 */
void build_forward_rhs_hsde(
    double* rhs,
    const double* dP_x,      // dP * x [batchSize][n]
    const double* dAt_z,     // dA' * z [batchSize][n]
    const double* dA_x,      // dA * x [batchSize][m]
    const double* dq,        // [batchSize][n]
    const double* db,        // [batchSize][m]
    const double* x,         // [batchSize][n]
    const double* z,         // [batchSize][m] - use z (not pi_u) to match backward
    const double* tau,       // [batchSize][1]
    const double* x_dP_x,    // x' * dP * x [batchSize][1]
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Recover solution derivatives from augmented system solution
 *
 * dx = dz_x - dt * x
 * dz = w - dt * z
 * ds = w - du - dt * s
 */
void recover_solution_derivatives(
    double* dx,
    double* dz,
    double* ds,
    const double* sol,  // Full solution [n + 2m + 1]
    const double* x,
    const double* z,
    const double* s,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

// ============================================================================
// Augmented KKT System Kernels
// ============================================================================

/**
 * @brief Update augmented KKT matrix for differentiation
 *
 * The differentiation KKT has structure:
 * [I   J ]   [y  ]   [0      ]
 * [J' -εI] * [sol] = [rhs_bar]
 *
 * where J is the Jacobian of the KKT conditions.
 * This kernel updates J with current H (cone derivative) blocks.
 */
// Note: update_diff_kkt_H_block is now handled internally by DiffKKT::updateJ
// via populate_H_blocks_kernel (not externally callable)

// ============================================================================
// Fused Projection + Derivative Kernels
// ============================================================================

/**
 * @brief Fused: compute u = z - s, cone projection, and derivative in one pass per cone type
 *
 * For exp/power cones, this avoids redundant Newton iterations by sharing
 * the primal projection result between the dual projection and derivative.
 * Replaces: compute_u_from_z_s + compute_cone_projection + compute_cone_derivative
 */
void fused_cone_projection_and_derivative(
    double* u,
    double* pi_u,
    double* nonneg_H,
    double* soc_H,
    const int64_t* d_soc_Hs_offsets,
    int64_t totalDenseSocHsEntries,
    double* soc_sparse_diag,
    double* soc_sparse_v1,
    double* soc_sparse_v2,
    double* soc_sparse_c1,
    double* soc_sparse_c2,
    const int64_t* d_soc_sparse_indices,
    const int64_t* d_soc_sparse_offsets,
    int64_t totalSparseSocDim,
    int64_t numSparseSoc,
    double* exp_H,
    double* power_H,
    const double* z,
    const double* s,
    const double* powerAlphas,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_sz_offsets,
    int64_t totalSocDim,
    int64_t totalPsdSvecDim,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Direct-x SOC Jacobian (IFT-direct backward, rank-2 sparse)
 *
 * For direct-x SOC cones with dim > 4, computes the rank-2
 * decomposition of `DΠ_K*(u_x)` where `u_x = z_x − x[J]`. Mirrors the
 * slack-side `compute_soc_derivative_sparse` formulas (interior /
 * polar / boundary) using direct-x indexing.
 */
void compute_xcone_soc_rank2(
    double* sparse_diag,
    double* sparse_v1,
    double* sparse_v2,
    double* sparse_c1,
    double* sparse_c2,
    const double* x_eq,
    const double* z_x_eq,
    const int64_t* xcone_indices,
    const int64_t* xcone_numel_offsets,
    const int64_t* d_xsoc_sparse_dim_offsets,
    const int64_t* d_xsoc_sparse_to_xc,
    const int64_t* d_xsoc_sparse_dims,
    int64_t numSparseXSoc,
    int64_t totalSparseXSocDim,
    int64_t n,
    int64_t totalXConeNumel,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Direct-x GenPowerCone Jacobian (IFT-direct backward)
 *
 * Computes dense H_x = DΠ_K*(u_x) for direct-x GenPower cones, where
 * u_x_k = z_x_k − x[J_xc[k]]. Internally runs the same Newton iteration
 * + boundary detection as the slack kernel, builds the rank-3
 * decomposition into the supplied stripe buffers, and expands it to a
 * dense `dim*dim` row-major block in `xcone_genpow_H` at the per-cone
 * `xcone_h_off[xc]` offset.
 *
 * `work_vec` and `work_dim1` are scratch arrays of size
 * `totalXGenPowerDim` and `7 * totalXGenPowerAlphas` per batch.
 * The CPU reference also uses dense expansion; sparse rank-3 expansion
 * for direct-x is a future optimization.
 */
void compute_xcone_genpow_H(
    double* xcone_genpow_H,
    double* rank3_diag,
    double* rank3_left1,
    double* rank3_right1,
    double* rank3_left2,
    double* rank3_right2,
    double* rank3_left3,
    double* rank3_c3,
    const double* x_eq,
    const double* z_x_eq,
    const int64_t* xcone_kinds,
    const int64_t* xcone_indices,
    const int64_t* xcone_numel_offsets,
    const int64_t* xcone_h_off,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const int64_t* d_xcone_genpow_dim_offsets,
    const double*  d_xcone_genpow_alphas,
    double* work_vec,
    double* work_dim1,
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t n,
    int64_t totalXConeNumel,
    int64_t totalXGenPowKkt,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Asymmetric direct-x cone Jacobian (IFT-direct backward)
 *
 * Computes dense H_x = DΠ_K*(u_x) for direct-x cones with kind ∈
 * {Exp, Power}, where u_x_k = z_x_k − x[J_xc[k]]. The kernel walks all
 * direct-x cones and skips kinds that are not Exp/Power. Outputs:
 *   - xcone_exp_H[batchSize][totalXExpKkt]    row-major 3*3 per cone
 *   - xcone_pow_H[batchSize][totalXPowKkt]    row-major 3*3 per cone
 * Per-cone offsets come from `xcone_h_off[xc]`.
 */
void compute_xcone_asymm_H(
    double* xcone_exp_H,
    double* xcone_pow_H,
    const double* x_eq,
    const double* z_x_eq,
    const int64_t* xcone_kinds,
    const int64_t* xcone_indices,
    const int64_t* xcone_numel_offsets,
    const int64_t* xcone_h_off,
    const int64_t* d_xcone_pow_idx,
    const double*  d_xcone_pow_alpha,
    int64_t numXCones,
    int64_t n,
    int64_t totalXConeNumel,
    int64_t totalXExpKkt,
    int64_t totalXPowKkt,
    int64_t batchSize,
    cudaStream_t stream
);

/**
 * @brief Fused: compute u = z - s and cone projection (no derivative)
 *
 * For the forward path where only the projection is needed.
 * Replaces: compute_u_from_z_s + compute_cone_projection
 */
void fused_u_and_cone_projection(
    double* u,
    double* pi_u,
    const double* z,
    const double* s,
    const double* powerAlphas,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_sz_offsets,
    int64_t totalSocDim,
    int64_t totalPsdSvecDim,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
);

// ============================================================================
// PSD Cone Projection and Derivative
// ============================================================================

// Forward declarations
struct Cones;
struct ConeDerivatives;

/**
 * @brief Project onto dual PSD cone (self-dual: eigendecomp, clamp eigenvalues >= 0)
 *
 * For 1×1 cones: scalar max(0, u).
 * For general dims: cuSOLVER eigendecomp → clamp eigenvalues → reconstruct.
 * Caches eigenvalues/eigenvectors in derivs for reuse by compute_psd_derivative.
 */
void project_psd_cone_dual(
    double* pi_u,
    const double* u,
    int64_t offset,
    const Cones& cones,
    ConeDerivatives& derivs,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
);

/**
 * @brief Compute PSD cone derivative DΠ_K*(u) using Ω matrix
 *
 * PSD is self-dual, derivative is symmetric upper triangle.
 * Reuses cached eigendecomp from project_psd_cone_dual.
 * For 1×1: scalar derivative (0 or 1).
 * For general: Ω[i,j] quotient formula + basis vector sweep.
 */
void compute_psd_derivative(
    double* psd_H,
    const double* u,
    int64_t offset,
    const Cones& cones,
    ConeDerivatives& derivs,
    int64_t batchSize,
    int64_t m,
    cudaStream_t stream
);

/**
 * @brief Compute PSD direct-x cone derivative DΠ_K(u_x) for IFT-direct backward.
 *
 * Mirrors `compute_psd_derivative` but operates on direct-x PSD cones whose
 * input is `u_x = z_x − x[J]` (gathered svec across the cone's primal index
 * set), and writes a dense `svec_dim × svec_dim` row-major Jacobian per
 * cone into `xcone_psd_H`. Eigendecomp is computed fresh per backward call
 * (not cached from forward) and uses `derivs.d_psd_cusolver_work` sized for
 * the largest direct-x PSD cone in the constructor.
 */
void compute_xcone_psd_derivative(
    double* xcone_psd_H,
    const double* x_eq,
    const double* z_x_eq,
    const Cones& cones,
    ConeDerivatives& derivs,
    int64_t totalXConeNumel,
    int64_t totalXPsdKkt,
    int64_t n,
    int64_t batchSize,
    cudaStream_t stream
);

} // namespace moreau
