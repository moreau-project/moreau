/**
 * @file smoothing_kernels.cu
 * @brief CUDA kernels for cone smoothing (warm start projection)
 *
 * Implements per-cone smoothing to project warm start (z, s) pairs onto
 * the mu-central path. Follows the algorithm from arXiv:2512.00693.
 *
 * For each cone type, solves: z + mu * grad_f*(z) = work
 * where work = z - s (preserved), and grad_f* is the dual barrier gradient.
 */

#include "moreau/cones/cones.hpp"
#include "moreau/cones/cone_kernels.cuh"
#include "moreau/cones/common.cuh"
#include "moreau/profiling/profiler.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace moreau {

using namespace cones;

// ============================================================================
// Device helper: 3x3 Cholesky factor + solve (unrolled, no pivoting)
// ============================================================================

/**
 * @brief Unrolled 3x3 Cholesky factorization
 *
 * Input A is packed upper triangle: [A00, A01, A02, A11, A12, A22]
 * Output L is packed lower triangle stored in same layout:
 *   L[0]=L00, L[1]=L10, L[2]=L20, L[3]=L11, L[4]=L21, L[5]=L22
 *
 * @return true if factorization succeeded (all pivots positive)
 */
__device__ __forceinline__ bool cholesky_3x3_factor(double* L, const double* A) {
    // A[0]=A00, A[1]=A01, A[2]=A02, A[3]=A11, A[4]=A12, A[5]=A22
    double t = A[0];
    if (t <= 0.0) return false;
    L[0] = sqrt(t);           // L00

    L[1] = A[1] / L[0];       // L10 = A01 / L00

    t = A[3] - L[1] * L[1];   // A11 - L10^2
    if (t <= 0.0) return false;
    L[3] = sqrt(t);            // L11

    L[2] = A[2] / L[0];       // L20 = A02 / L00
    L[4] = (A[4] - L[1] * L[2]) / L[3]; // L21 = (A12 - L10*L20) / L11

    t = A[5] - L[2] * L[2] - L[4] * L[4]; // A22 - L20^2 - L21^2
    if (t <= 0.0) return false;
    L[5] = sqrt(t);            // L22

    return true;
}

/**
 * @brief Solve L L^T x = b given Cholesky factor L
 *
 * L packed as: [L00, L10, L20, L11, L21, L22]
 */
__device__ __forceinline__ void cholesky_3x3_solve(double* x, const double* L, const double* b) {
    // Forward substitution: L c = b
    double c0 = b[0] / L[0];
    double c1 = (b[1] - L[1] * c0) / L[3];
    double c2 = (b[2] - L[2] * c0 - L[4] * c1) / L[5];

    // Backward substitution: L^T x = c
    x[2] = c2 / L[5];
    x[1] = (c1 - L[4] * x[2]) / L[3];
    x[0] = (c0 - L[1] * x[1] - L[2] * x[2]) / L[0];
}

// ============================================================================
// Device helper: Exp cone dual gradient and Hessian
// ============================================================================

/**
 * @brief Compute dual barrier gradient and Hessian for exponential cone
 *
 * Dual exp cone: z3 >= -z1*e^(z2/z1 - 1), z3 > 0, z1 < 0
 * grad and H are output arrays (grad[3], H[6] packed upper triangle)
 */
__device__ __forceinline__ void exp_dual_grad_H(double* grad, double* H, const double* z) {
    double l = logsafe(-z[2] / z[0]);
    double r = -z[0] * l - z[0] + z[1];
    double c2 = 1.0 / r;

    grad[0] = c2 * l - 1.0 / z[0];
    grad[1] = -c2;
    grad[2] = (c2 * z[0] - 1.0) / z[2];

    H[0] = (r * r - z[0] * r + l * l * z[0] * z[0]) / (r * z[0] * z[0] * r);
    H[1] = -l / (r * r);
    H[2] = (z[1] - z[0]) / (r * r * z[2]);
    H[3] = 1.0 / (r * r);
    H[4] = -z[0] / (r * r * z[2]);
    H[5] = (r * r - z[0] * r + z[0] * z[0]) / (r * r * z[2] * z[2]);
}

/**
 * @brief Check if z is in the interior of the dual exponential cone
 */
__device__ __forceinline__ bool exp_is_dual_feasible(const double* z) {
    if (z[2] > 0.0 && z[0] < 0.0) {
        double res = z[1] - z[0] - z[0] * logsafe(-z[2] / z[0]);
        return res > 0.0;
    }
    return false;
}

// ============================================================================
// Device helper: Power cone dual gradient and Hessian
// ============================================================================

/**
 * @brief Compute dual barrier gradient and Hessian for power cone
 *
 * Dual power cone: (z1/alpha)^{2*alpha} * (z2/(1-alpha))^{2*(1-alpha)} >= z3^2
 * with z1 > 0, z2 > 0
 */
__device__ __forceinline__ void power_dual_grad_H(double* grad, double* H, const double* z, double alpha) {
    double two = 2.0;
    double four = 4.0;

    double phi = pow(z[0] / alpha, two * alpha) * pow(z[1] / (1.0 - alpha), two - two * alpha);
    // Avoid catastrophic cancellation: phi - z[2]² = (√φ - |z₂|)(√φ + |z₂|)
    double sqrt_phi = sqrt(phi);
    double abs_z2 = fabs(z[2]);
    double psi = (sqrt_phi - abs_z2) * (sqrt_phi + abs_z2);

    double g_psi[3];
    g_psi[0] = two * alpha * phi / (z[0] * psi);
    g_psi[1] = two * (1.0 - alpha) * phi / (z[1] * psi);
    g_psi[2] = -two * z[2] / psi;

    H[0] = g_psi[0] * g_psi[0] - two * alpha * (two * alpha - 1.0) * phi / (z[0] * z[0] * psi)
        + (1.0 - alpha) / (z[0] * z[0]);
    H[1] = g_psi[0] * g_psi[1] - four * alpha * (1.0 - alpha) * phi / (z[0] * z[1] * psi);
    H[2] = g_psi[0] * g_psi[2];
    H[3] = g_psi[1] * g_psi[1]
        - two * (1.0 - alpha) * (1.0 - two * alpha) * phi / (z[1] * z[1] * psi)
        + alpha / (z[1] * z[1]);
    H[4] = g_psi[1] * g_psi[2];
    H[5] = g_psi[2] * g_psi[2] + two / psi;

    grad[0] = -two * alpha * phi / (z[0] * psi) - (1.0 - alpha) / z[0];
    grad[1] = -two * (1.0 - alpha) * phi / (z[1] * psi) - alpha / z[1];
    grad[2] = two * z[2] / psi;
}

/**
 * @brief Check if z is in the interior of the dual power cone
 */
__device__ __forceinline__ bool power_is_dual_feasible(const double* z, double alpha) {
    if (z[0] > 0.0 && z[1] > 0.0) {
        double two = 2.0;
        double res = exp(
            (alpha * two) * logsafe(z[0] / alpha)
            + ((1.0 - alpha) * two) * logsafe(z[1] / (1.0 - alpha))
        ) - z[2] * z[2];
        return res > 0.0;
    }
    return false;
}

// ============================================================================
// Fused smoothing kernel for all cone types
// ============================================================================

/**
 * @brief Compute dual barrier gradient and Hessian for generalized power cone
 *
 * Updates grad[dim] and d1[dim1], d2_out, p[dim], q[dim1], r[dim2] for the
 * GenPowerCone decomposition H = D + pp' - qq' - rr'
 */
__device__ __forceinline__ void genpow_dual_grad_H(
    double* grad, double* d1, double& d2_out,
    double* p, double* q, double* r,
    const double* z, int64_t dim1, int64_t dim2,
    const double* alphas
) {
    // phi = ∏(z_i/αi)^{2αi}  — use log-sum-exp for numerical stability
    double log_phi = 0.0;
    for (int64_t i = 0; i < dim1; i++) {
        log_phi += 2.0 * alphas[i] * log(z[i] / alphas[i]);
    }
    double phi = exp(log_phi);

    double norm2w = 0.0;
    for (int64_t i = 0; i < dim2; i++) {
        norm2w += z[dim1 + i] * z[dim1 + i];
    }

    double zeta = phi - norm2w;
    // Caller must ensure z is dual-feasible (zeta > 0) before calling

    // tau[i] = 2*αi/zi, gradient
    for (int64_t i = 0; i < dim1; i++) {
        double tau_i = 2.0 * alphas[i] / z[i];
        q[i] = tau_i;  // temporarily store tau in q
        grad[i] = -tau_i * phi / zeta - (1.0 - alphas[i]) / z[i];
    }
    for (int64_t i = 0; i < dim2; i++) {
        grad[dim1 + i] = 2.0 * z[dim1 + i] / zeta;
    }

    // Hessian decomposition (with guarded denominators)
    constexpr double eps = 1e-300;
    double p0 = sqrt(phi * (phi + norm2w) / 2.0);
    double p1 = (p0 > eps) ? -2.0 * phi / p0 : 0.0;
    double q0 = sqrt(zeta * phi / 2.0);
    double phi_plus_norm2w = phi + norm2w;
    double r1 = (phi_plus_norm2w > eps) ? 2.0 * sqrt(zeta / phi_plus_norm2w) : 0.0;

    for (int64_t i = 0; i < dim1; i++) {
        double tau_i = q[i];
        d1[i] = tau_i * phi / (zeta * z[i]) + (1.0 - alphas[i]) / (z[i] * z[i]);
    }
    d2_out = 2.0 / zeta;

    // p vector
    for (int64_t i = 0; i < dim1; i++) {
        p[i] = (p0 / zeta) * q[i];  // q still has tau
    }
    for (int64_t i = 0; i < dim2; i++) {
        p[dim1 + i] = (p1 / zeta) * z[dim1 + i];
    }

    // q = (q0/zeta)*tau
    for (int64_t i = 0; i < dim1; i++) {
        q[i] *= (q0 / zeta);
    }

    // r
    for (int64_t i = 0; i < dim2; i++) {
        r[i] = (r1 / zeta) * z[dim1 + i];
    }
}

/**
 * @brief Check if z is in the interior of the dual generalized power cone
 */
__device__ __forceinline__ bool genpow_is_dual_feasible_smooth(
    const double* z, int64_t dim1, int64_t dim2, const double* alphas
) {
    for (int64_t i = 0; i < dim1; i++) {
        if (z[i] <= 0.0) return false;
    }
    double log_prod = 0.0;
    for (int64_t i = 0; i < dim1; i++) {
        log_prod += 2.0 * alphas[i] * log(z[i] / alphas[i]);
    }
    double norm2w = 0.0;
    for (int64_t i = 0; i < dim2; i++) {
        norm2w += z[dim1 + i] * z[dim1 + i];
    }
    return exp(log_prod) - norm2w > 0.0;
}

__global__ void smoothing_all_cones_kernel(
    double* __restrict__ z,             // [batchSize * m_total] — z variables (modified in-place)
    const double* __restrict__ work,    // [batchSize * m_total] — work = z_orig - s_orig (preserved)
    const double* __restrict__ mu,      // [batchSize] — per-batch warmness parameter
    const double* __restrict__ d_powerAlphas,  // [numPowerCones] — power cone alpha params
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m_total,
    int64_t batchSize,
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    int64_t totalSocDim,
    const int64_t* __restrict__ d_soc_sz_offsets,
    // GenPowerCone params
    int64_t numGenPowerCones,
    const double* __restrict__ d_genPowerAlphas,
    const int64_t* __restrict__ d_genPowerDim1s,
    const int64_t* __restrict__ d_genPowerDim2s,
    const int64_t* __restrict__ d_genPowerOffsets,
    const int64_t* __restrict__ d_genPowerAlphaOffsets,
    const int64_t* __restrict__ d_genPowerSzOffsets,
    int64_t totalGenPowerDim,
    // GenPowerCone workspace (replaces per-thread stack arrays)
    double* __restrict__ smooth_zlocal,     // [batchSize][totalGenPowerDim]
    double* __restrict__ smooth_wlocal,     // [batchSize][totalGenPowerDim]
    double* __restrict__ smooth_res,        // [batchSize][totalGenPowerDim]
    double* __restrict__ smooth_delta,      // [batchSize][totalGenPowerDim]
    double* __restrict__ smooth_hmat,       // [batchSize][totalGenPowerDimSq]
    double* __restrict__ smooth_lmat,       // [batchSize][totalGenPowerDimSq]
    const int64_t* __restrict__ d_genPowerDimSqOffsets,  // prefix-sum of dim^2
    int64_t totalGenPowerDimSq,
    // Reused forward-solver buffers (safe: smoothing and scaling don't overlap)
    double* __restrict__ genpow_grad_buf,   // [batchSize][totalGenPowerDim]
    double* __restrict__ genpow_d1_buf,     // [batchSize][totalGenPowerAlphas]
    double* __restrict__ genpow_p_buf,      // [batchSize][totalGenPowerDim]
    double* __restrict__ genpow_q_buf,      // [batchSize][totalGenPowerAlphas]
    double* __restrict__ genpow_r_buf,      // [batchSize][totalGenPowerDim2]
    int64_t totalGenPowerAlphas,
    int64_t totalGenPowerDim2
) {
    int64_t batch = blockIdx.x;
    int64_t stride = blockDim.x;

    if (batch >= batchSize) return;

    double mu_val = mu[batch];

    // ========== ZERO CONES ==========
    // z_new = work (so that s_new = z_new - work = 0)
    if (numZeroCones > 0) {
        int64_t base = batch * m_total;
        for (int64_t i = threadIdx.x; i < numZeroCones; i += stride) {
            z[base + i] = work[base + i];
        }
    }

    // ========== NONNEGATIVE CONES ==========
    // z_new = (w + sqrt(w^2 + 4*mu)) / 2
    if (numNonnegCones > 0) {
        int64_t offset = numZeroCones;
        int64_t base = batch * m_total + offset;
        double four_mu = 4.0 * mu_val;

        for (int64_t i = threadIdx.x; i < numNonnegCones; i += stride) {
            double w = work[base + i];
            z[base + i] = (w + sqrt(w * w + four_mu)) / 2.0;
        }
    }

    // ========== SECOND-ORDER CONES (variable dim) ==========
    // Analytic formula from socone.rs:316-351
    if (numSocCones > 0) {
        int64_t offset = numZeroCones + numNonnegCones;

        for (int64_t cone_idx = threadIdx.x; cone_idx < numSocCones; cone_idx += stride) {
            int64_t dim = d_soc_dims[cone_idx];
            int64_t sz_off = d_soc_sz_offsets[cone_idx];
            int64_t base = batch * m_total + offset + sz_off;

            double w0 = work[base];
            double w0sq = w0 * w0;

            // ||w[1:]||^2
            double w1sq = 0.0;
            for (int64_t i = 1; i < dim; i++) {
                w1sq += work[base + i] * work[base + i];
            }

            // Avoid catastrophic cancellation: w0² - ||w1||² = (w0 - ||w1||)(w0 + ||w1||)
            double w1norm = sqrt(w1sq);
            double res = (w0 - w1norm) * (w0 + w1norm);

            double c = res / mu_val;
            double inner = c * c + 4.0 * (2.0 * (w0sq + w1sq) / mu_val + 4.0);
            double gamma = (c + sqrt(inner)) / 2.0;
            double gamma_sq_minus_4 = gamma * gamma - 4.0;

            // Choose root based on sign of w[0]
            double rho;
            if (w0 > 0.0) {
                rho = (gamma + sqrt(fmax(gamma_sq_minus_4, 0.0))) / 2.0;
            } else {
                rho = (gamma - sqrt(fmax(gamma_sq_minus_4, 0.0))) / 2.0;
            }

            // Compute z[0]
            double rho_minus_1 = rho - 1.0;
            if (fabs(rho_minus_1) < DEVICE_EPSILON) {
                z[base] = sqrt(mu_val + w1sq / 4.0);
            } else {
                z[base] = rho / rho_minus_1 * w0;
            }

            // Compute z[1:]
            double d = rho / (rho + 1.0);
            for (int64_t i = 1; i < dim; i++) {
                z[base + i] = d * work[base + i];
            }
        }
    }

    // ========== EXPONENTIAL CONES (dim 3) ==========
    // Newton iteration to solve: z + mu * grad_f*(z) = work
    if (numExpCones > 0) {
        int64_t offset = numZeroCones + numNonnegCones + totalSocDim;
        const double tol = DEVICE_SQRT_EPSILON;
        const double two_minus_sqrt3 = 2.0 - 1.7320508075688772;  // 2 - sqrt(3)
        const double min_val = 1e-6;

        for (int64_t cone_idx = threadIdx.x; cone_idx < numExpCones; cone_idx += stride) {
            int64_t base = batch * m_total + offset + cone_idx * 3;

            double z_local[3] = {z[base], z[base + 1], z[base + 2]};
            double w_local[3] = {work[base], work[base + 1], work[base + 2]};

            // Check if current z is a good starting point
            bool needs_fallback = !exp_is_dual_feasible(z_local)
                || fabs(z_local[0]) < min_val
                || fabs(z_local[2]) < min_val;

            if (needs_fallback) {
                z_local[0] = -1.051383945322714;
                z_local[1] = 0.556409619469370;
                z_local[2] = 1.258967884768947;
            }

            double grad[3], H[6];

            for (int iter = 0; iter < 100; iter++) {
                exp_dual_grad_H(grad, H, z_local);

                // Residual: res = mu * grad + (z - work)
                double res[3];
                res[0] = mu_val * grad[0] + z_local[0] - w_local[0];
                res[1] = mu_val * grad[1] + z_local[1] - w_local[1];
                res[2] = mu_val * grad[2] + z_local[2] - w_local[2];

                double res_norm = sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2]);
                if (res_norm < tol) break;

                // Form matrix: mat = mu * H_dual + I
                double mat[6];
                mat[0] = mu_val * H[0] + 1.0;
                mat[1] = mu_val * H[1];
                mat[2] = mu_val * H[2];
                mat[3] = mu_val * H[3] + 1.0;
                mat[4] = mu_val * H[4];
                mat[5] = mu_val * H[5] + 1.0;

                // Cholesky factorize with regularization
                double L[6];
                double max_elem = fmax(fmax(fabs(mat[0]), fabs(mat[3])), fabs(mat[5]));
                double regularizer = DEVICE_EPSILON * max_elem;
                bool factor_ok = false;

                for (int reg_iter = 0; reg_iter < 10; reg_iter++) {
                    if (cholesky_3x3_factor(L, mat)) {
                        factor_ok = true;
                        break;
                    }
                    mat[0] += regularizer;
                    mat[3] += regularizer;
                    mat[5] += regularizer;
                }

                if (!factor_ok) break;

                // Newton step: solve mat * delta = res
                double delta[3];
                cholesky_3x3_solve(delta, L, res);

                // Newton decrement
                double lambda_sq = res[0] * delta[0] + res[1] * delta[1] + res[2] * delta[2];
                double lambda = (lambda_sq > 0.0) ? sqrt(lambda_sq) : 0.0;

                // Damped update
                double damping = (lambda < two_minus_sqrt3) ? 1.0 : 1.0 / (1.0 + lambda);

                z_local[0] -= damping * delta[0];
                z_local[1] -= damping * delta[1];
                z_local[2] -= damping * delta[2];
            }

            z[base]     = z_local[0];
            z[base + 1] = z_local[1];
            z[base + 2] = z_local[2];
        }
    }

    // ========== POWER CONES (dim 3) ==========
    // Newton iteration to solve: z + mu * grad_f*(z) = work
    if (numPowerCones > 0) {
        int64_t offset = numZeroCones + numNonnegCones + totalSocDim + numExpCones * 3;
        const double tol = DEVICE_SQRT_EPSILON;
        const double two_minus_sqrt3 = 2.0 - 1.7320508075688772;
        const double min_val = 1e-6;

        for (int64_t cone_idx = threadIdx.x; cone_idx < numPowerCones; cone_idx += stride) {
            int64_t base = batch * m_total + offset + cone_idx * 3;
            double alpha = d_powerAlphas[cone_idx];

            double z_local[3] = {z[base], z[base + 1], z[base + 2]};
            double w_local[3] = {work[base], work[base + 1], work[base + 2]};

            // Check if current z is a good starting point
            bool needs_fallback = !power_is_dual_feasible(z_local, alpha)
                || fabs(z_local[0]) < min_val
                || fabs(z_local[1]) < min_val;

            if (needs_fallback) {
                z_local[0] = sqrt(1.0 + alpha);
                z_local[1] = sqrt(2.0 - alpha);
                z_local[2] = 0.0;
            }

            double grad[3], H[6];

            for (int iter = 0; iter < 100; iter++) {
                power_dual_grad_H(grad, H, z_local, alpha);

                // Residual: res = mu * grad + (z - work)
                double res[3];
                res[0] = mu_val * grad[0] + z_local[0] - w_local[0];
                res[1] = mu_val * grad[1] + z_local[1] - w_local[1];
                res[2] = mu_val * grad[2] + z_local[2] - w_local[2];

                double res_norm = sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2]);
                if (res_norm < tol) break;

                // Form matrix: mat = mu * H_dual + I
                double mat[6];
                mat[0] = mu_val * H[0] + 1.0;
                mat[1] = mu_val * H[1];
                mat[2] = mu_val * H[2];
                mat[3] = mu_val * H[3] + 1.0;
                mat[4] = mu_val * H[4];
                mat[5] = mu_val * H[5] + 1.0;

                // Cholesky factorize with regularization
                double L[6];
                double max_elem = fmax(fmax(fabs(mat[0]), fabs(mat[3])), fabs(mat[5]));
                double regularizer = DEVICE_EPSILON * max_elem;
                bool factor_ok = false;

                for (int reg_iter = 0; reg_iter < 10; reg_iter++) {
                    if (cholesky_3x3_factor(L, mat)) {
                        factor_ok = true;
                        break;
                    }
                    mat[0] += regularizer;
                    mat[3] += regularizer;
                    mat[5] += regularizer;
                }

                if (!factor_ok) break;

                // Newton step: solve mat * delta = res
                double delta[3];
                cholesky_3x3_solve(delta, L, res);

                // Newton decrement
                double lambda_sq = res[0] * delta[0] + res[1] * delta[1] + res[2] * delta[2];
                double lambda = (lambda_sq > 0.0) ? sqrt(lambda_sq) : 0.0;

                // Damped update
                double damping = (lambda < two_minus_sqrt3) ? 1.0 : 1.0 / (1.0 + lambda);

                z_local[0] -= damping * delta[0];
                z_local[1] -= damping * delta[1];
                z_local[2] -= damping * delta[2];
            }

            z[base]     = z_local[0];
            z[base + 1] = z_local[1];
            z[base + 2] = z_local[2];
        }
    }

    // ========== GENERALIZED POWER CONES (variable dim) ==========
    // Newton iteration to solve: z + mu * grad_f*(z) = work
    // Uses variable-dim dense Cholesky with preallocated workspace buffers
    if (numGenPowerCones > 0) {
        int64_t offset = numZeroCones + numNonnegCones + totalSocDim + numExpCones * 3 + numPowerCones * 3;
        const double tol = DEVICE_SQRT_EPSILON;
        const double two_minus_sqrt3 = 2.0 - 1.7320508075688772;
        const double min_val = 1e-6;

        for (int64_t cone_idx = threadIdx.x; cone_idx < numGenPowerCones; cone_idx += stride) {
            int64_t dim1 = d_genPowerDim1s[cone_idx];
            int64_t dim2 = d_genPowerDim2s[cone_idx];
            int64_t dim = dim1 + dim2;
            int64_t gp_off = d_genPowerOffsets[cone_idx];
            int64_t sz_off = d_genPowerSzOffsets[cone_idx];
            int64_t alpha_off = d_genPowerAlphaOffsets[cone_idx];
            int64_t dimsq_off = d_genPowerDimSqOffsets[cone_idx];
            int64_t base = batch * m_total + offset + sz_off;
            const double* cone_alphas = &d_genPowerAlphas[alpha_off];

            // Workspace pointers into preallocated buffers
            double* z_local = smooth_zlocal + batch * totalGenPowerDim + gp_off;
            double* w_local = smooth_wlocal + batch * totalGenPowerDim + gp_off;
            double* grad    = genpow_grad_buf + batch * totalGenPowerDim + gp_off;
            double* d1_arr  = genpow_d1_buf + batch * totalGenPowerAlphas + alpha_off;
            double* p_arr   = genpow_p_buf + batch * totalGenPowerDim + gp_off;
            double* q_arr   = genpow_q_buf + batch * totalGenPowerAlphas + alpha_off;
            // r_arr needs dim2-offset: compute from prefix sums
            int64_t dim2_off = gp_off - alpha_off;  // totalDim - totalAlphas up to this cone = totalDim2 up to this cone
            double* r_arr   = genpow_r_buf + batch * totalGenPowerDim2 + dim2_off;
            double* res     = smooth_res + batch * totalGenPowerDim + gp_off;
            double* delta   = smooth_delta + batch * totalGenPowerDim + gp_off;
            double* hmat    = smooth_hmat + batch * totalGenPowerDimSq + dimsq_off;
            double* lmat    = smooth_lmat + batch * totalGenPowerDimSq + dimsq_off;
            double d2_val;

            // Load z and work
            for (int64_t i = 0; i < dim; i++) {
                z_local[i] = z[base + i];
                w_local[i] = work[base + i];
            }

            // Check if current z is a good starting point
            bool needs_fallback = !genpow_is_dual_feasible_smooth(z_local, dim1, dim2, cone_alphas);
            if (!needs_fallback) {
                for (int64_t i = 0; i < dim1; i++) {
                    if (fabs(z_local[i]) < min_val) { needs_fallback = true; break; }
                }
            }

            if (needs_fallback) {
                for (int64_t i = 0; i < dim1; i++) {
                    z_local[i] = sqrt(1.0 + cone_alphas[i]);
                }
                for (int64_t i = 0; i < dim2; i++) {
                    z_local[dim1 + i] = 0.0;
                }
            }

            for (int iter = 0; iter < 100; iter++) {
                // Check feasibility before computing gradient/Hessian.
                // If Newton step moved z outside the cone, roll back to
                // the last feasible iterate (saved in res buffer).
                bool iter_feasible = genpow_is_dual_feasible_smooth(z_local, dim1, dim2, cone_alphas);
                if (!iter_feasible) {
                    if (iter > 0) {
                        for (int64_t i = 0; i < dim; i++) z_local[i] = res[i];
                    }
                    break;
                }
                bool any_nonpos = false;
                for (int64_t i = 0; i < dim1; i++) {
                    if (z_local[i] <= 0.0) { any_nonpos = true; break; }
                }
                if (any_nonpos) {
                    if (iter > 0) {
                        for (int64_t i = 0; i < dim; i++) z_local[i] = res[i];
                    }
                    break;
                }

                // Compute gradient and Hessian decomposition
                genpow_dual_grad_H(grad, d1_arr, d2_val, p_arr, q_arr, r_arr,
                                   z_local, dim1, dim2, cone_alphas);

                // Residual: res = mu * grad + (z - work)
                double res_norm_sq = 0.0;
                for (int64_t i = 0; i < dim; i++) {
                    res[i] = mu_val * grad[i] + z_local[i] - w_local[i];
                    res_norm_sq += res[i] * res[i];
                }
                if (sqrt(res_norm_sq) < tol) break;

                // Form Hessian matrix: H = mu*(D + pp' - qq' - rr') + I
                for (int64_t i = 0; i < dim; i++) {
                    for (int64_t j = 0; j < dim; j++) {
                        double val = 0.0;
                        if (i == j) {
                            val += (i < dim1) ? (mu_val * d1_arr[i] + 1.0) : (mu_val * d2_val + 1.0);
                        }
                        val += mu_val * p_arr[i] * p_arr[j];
                        if (i < dim1 && j < dim1) val -= mu_val * q_arr[i] * q_arr[j];
                        if (i >= dim1 && j >= dim1) val -= mu_val * r_arr[i - dim1] * r_arr[j - dim1];
                        hmat[i * dim + j] = val;
                    }
                }

                // Dense Cholesky with regularization
                double max_elem = 0.0;
                for (int64_t i = 0; i < dim; i++) {
                    max_elem = fmax(max_elem, fabs(hmat[i * dim + i]));
                }
                double regularizer = DEVICE_EPSILON * max_elem;
                bool factor_ok = false;

                for (int reg_iter = 0; reg_iter < 10; reg_iter++) {
                    // Copy hmat to lmat for factorization
                    for (int64_t i = 0; i < dim * dim; i++) lmat[i] = hmat[i];

                    bool ok = true;
                    for (int64_t j = 0; j < dim; j++) {
                        double sum = lmat[j * dim + j];
                        for (int64_t k = 0; k < j; k++) {
                            sum -= lmat[j * dim + k] * lmat[j * dim + k];
                        }
                        if (sum <= 0.0) { ok = false; break; }
                        double ljj = sqrt(sum);
                        lmat[j * dim + j] = ljj;
                        for (int64_t i = j + 1; i < dim; i++) {
                            double s = lmat[i * dim + j];
                            for (int64_t k = 0; k < j; k++) {
                                s -= lmat[i * dim + k] * lmat[j * dim + k];
                            }
                            lmat[i * dim + j] = s / ljj;
                        }
                    }

                    if (ok) {
                        // Forward substitution: L*y = res
                        for (int64_t i = 0; i < dim; i++) {
                            double s = res[i];
                            for (int64_t j = 0; j < i; j++) {
                                s -= lmat[i * dim + j] * delta[j];
                            }
                            delta[i] = s / lmat[i * dim + i];
                        }
                        // Backward substitution: L'*delta = y
                        for (int64_t i = dim - 1; i >= 0; i--) {
                            double s = delta[i];
                            for (int64_t j = i + 1; j < dim; j++) {
                                s -= lmat[j * dim + i] * delta[j];
                            }
                            delta[i] = s / lmat[i * dim + i];
                        }
                        factor_ok = true;
                        break;
                    }
                    // Add regularization
                    for (int64_t i = 0; i < dim; i++) {
                        hmat[i * dim + i] += regularizer;
                    }
                }

                if (!factor_ok) break;

                // Newton decrement
                double lambda_sq = 0.0;
                for (int64_t i = 0; i < dim; i++) {
                    lambda_sq += res[i] * delta[i];
                }
                double lambda = (lambda_sq > 0.0) ? sqrt(lambda_sq) : 0.0;

                // Save current feasible z for rollback (reuse res buffer,
                // which is no longer needed after lambda_sq computation)
                for (int64_t i = 0; i < dim; i++) res[i] = z_local[i];

                // Damped update
                double damping = (lambda < two_minus_sqrt3) ? 1.0 : 1.0 / (1.0 + lambda);
                for (int64_t i = 0; i < dim; i++) {
                    z_local[i] -= damping * delta[i];
                }
            }

            // Write back
            for (int64_t i = 0; i < dim; i++) {
                z[base + i] = z_local[i];
            }
        }
    }
}

// ============================================================================
// Host wrapper
// ============================================================================

void smoothing_all_cones(
    double* z,
    const double* work,
    const double* mu,
    const double* d_powerAlphas,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m_total,
    int64_t batchSize,
    cudaStream_t stream,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    int64_t totalSocDim,
    const int64_t* d_soc_sz_offsets,
    // GenPowerCone params
    int64_t numGenPowerCones,
    const double* d_genPowerAlphas,
    const int64_t* d_genPowerDim1s,
    const int64_t* d_genPowerDim2s,
    const int64_t* d_genPowerOffsets,
    const int64_t* d_genPowerAlphaOffsets,
    const int64_t* d_genPowerSzOffsets,
    int64_t totalGenPowerDim,
    // GenPowerCone workspace
    double* smooth_zlocal,
    double* smooth_wlocal,
    double* smooth_res,
    double* smooth_delta,
    double* smooth_hmat,
    double* smooth_lmat,
    const int64_t* d_genPowerDimSqOffsets,
    int64_t totalGenPowerDimSq,
    double* genpow_grad_buf,
    double* genpow_d1_buf,
    double* genpow_p_buf,
    double* genpow_q_buf,
    double* genpow_r_buf,
    int64_t totalGenPowerAlphas,
    int64_t totalGenPowerDim2
) {
    int64_t maxCones = numNonnegCones;
    maxCones = (numSocCones > maxCones) ? numSocCones : maxCones;
    maxCones = (numExpCones > maxCones) ? numExpCones : maxCones;
    maxCones = (numPowerCones > maxCones) ? numPowerCones : maxCones;
    maxCones = (numGenPowerCones > maxCones) ? numGenPowerCones : maxCones;
    maxCones = (numZeroCones > maxCones) ? numZeroCones : maxCones;

    if (maxCones == 0) return;

    int threadsPerBlock = (maxCones < 256) ? ((maxCones + 31) / 32 * 32) : 256;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    MOREAU_KERNEL_LAUNCH(smoothing_all_cones_kernel, batchSize, threadsPerBlock, 0, stream,
        z, work, mu, d_powerAlphas,
        numZeroCones, numNonnegCones, numSocCones, numExpCones, numPowerCones,
        m_total, batchSize,
        d_soc_dims, d_soc_offsets, totalSocDim, d_soc_sz_offsets,
        numGenPowerCones, d_genPowerAlphas,
        d_genPowerDim1s, d_genPowerDim2s,
        d_genPowerOffsets, d_genPowerAlphaOffsets, d_genPowerSzOffsets, totalGenPowerDim,
        smooth_zlocal, smooth_wlocal, smooth_res, smooth_delta,
        smooth_hmat, smooth_lmat,
        d_genPowerDimSqOffsets, totalGenPowerDimSq,
        genpow_grad_buf, genpow_d1_buf, genpow_p_buf, genpow_q_buf, genpow_r_buf,
        totalGenPowerAlphas, totalGenPowerDim2
    );
}

} // namespace moreau
