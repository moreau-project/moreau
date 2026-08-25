/**
 * @file composite_kernels.cu
 * @brief CUDA kernels that operate on all cone types together
 *
 * Includes fused scaling kernel, margins computation, and shifts.
 */

#include "moreau/cones/cones.hpp"
#include "moreau/cones/cone_kernels.cuh"
#include "moreau/cones/common.cuh"
#include "moreau/profiling/profiler.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace moreau {

using namespace cones;

// Shared device functions (gradient_primal_exp, gradient_primal_power)
// are defined in common.cuh

// ============================================================================
// Fused Cone Scaling Kernel (all cone types in one kernel)
// ============================================================================

__global__ void update_scaling_all_cones_kernel(
    // Nonneg outputs
    double* __restrict__ nonneg_w,
    double* __restrict__ nonneg_lambda,
    // SOC outputs
    // Only cones with dim <= SOC_PARALLEL_THRESHOLD are processed here;
    // larger cones are handled by update_large_soc_scaling_kernel launched
    // separately from the host. Sorted-ascending dim guarantees small
    // cones occupy [0, numSmallSoc) and large cones [numSmallSoc, numSocCones).
    double* soc_u,
    double* soc_v,
    double* soc_d,
    double* soc_w,
    double* soc_eta,
    double* soc_lambda,
    double* soc_Hs,
    // Exp outputs
    double* exp_grad,
    double* exp_H_dual,
    double* exp_Hs,
    double* exp_z,
    // Power outputs
    double* power_grad,
    double* power_H_dual,
    double* power_Hs,
    double* power_z,
    // GenPowerCone outputs
    double* genpow_grad,
    double* genpow_z_out,
    double* genpow_Hs,
    double* genpow_p,
    double* genpow_q,
    double* genpow_r,
    double* genpow_d1,
    double* genpow_d2,
    // Inputs
    const double* s,
    const double* z,
    const double* mu,
    const double* d_powerAlphas,
    // GenPowerCone inputs
    const double* d_genPowerAlphas,
    const int64_t* d_genPowerDim1s,
    const int64_t* d_genPowerDim2s,
    const int64_t* d_genPowerOffsets,
    const int64_t* d_genPowerAlphaOffsets,
    const int64_t* d_genPowerHsOffsets,
    const int64_t* d_genPowerSzOffsets,
    int64_t numGenPowerCones,
    int64_t numSmallGenPow,
    int64_t totalGenPowerDim,
    int64_t totalGenPowerAlphas,
    int64_t totalGenPowerGradEntries,
    int64_t totalGenPowerHsEntries,
    // Cone structure
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numSmallSoc,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m_total,
    int64_t batchSize,
    int scaling_strategy,
    // Variable-dim SOC params
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_Hs_offsets,
    int64_t totalSocDim,
    int64_t totalSocHsEntries,
    // Scaling success flag (set to 0 on degenerate SOC state)
    int32_t* scaling_success,
    // Sorted SOC s/z offsets
    const int64_t* d_soc_sz_offsets
) {
    int64_t batch = blockIdx.x;
    int64_t stride = blockDim.x;

    // Fill one SOC block of soc_Hs with identity-equivalent values when degenerate.
    // Dense (dim<=4): upper triangle of I; Sparse (dim>4): diagonal = η²·diag(d, 1, ..., 1)
    // For sparse cones, d_val is the d parameter (e.g. 0.5 for identity scaling).
    // The expansion columns add η²*(uu' - vv'), so soc_Hs stores only the diagonal part.
    auto write_identity_soc_Hs = [&](int64_t hs_base, double eta_sq, int64_t dim, double d_val) {
        if (dim > 4) {
            // Sparse: diagonal = η²·diag(d, 1, ..., 1)
            soc_Hs[hs_base] = eta_sq * d_val;
            for (int64_t i = 1; i < dim; i++) {
                soc_Hs[hs_base + i] = eta_sq;
            }
        } else {
            // Dense: upper triangle
            int64_t k = 0;
            for (int64_t i = 0; i < dim; i++) {
                for (int64_t j = i; j < dim; j++) {
                    soc_Hs[hs_base + k] = (i == j) ? eta_sq : 0.0;
                    k++;
                }
            }
        }
    };

    if (batch >= batchSize) return;

    // ========== NONNEGATIVE CONES ==========
    if (numNonnegCones > 0) {
        int64_t offset_s = numZeroCones;
        int64_t offset_z = numZeroCones;

        for (int64_t idx = threadIdx.x; idx < numNonnegCones; idx += stride) {
            int64_t s_idx = batch * m_total + offset_s + idx;
            int64_t z_idx = batch * m_total + offset_z + idx;
            int64_t out_idx = batch * numNonnegCones + idx;

            double si = s[s_idx];
            double zi = z[z_idx];

            nonneg_lambda[out_idx] = sqrt(si * zi);
            nonneg_w[out_idx] = sqrt(si / zi);
        }
    }

    // ========== SOC CONES (with fused populate_soc_Hs) ==========
    // Only small cones (dim <= SOC_PARALLEL_THRESHOLD) are processed here;
    // large cones are handled by update_large_soc_scaling launched from
    // the host. Cones are sorted ascending in Cones::initialize(), so
    // small cones are the contiguous prefix [0, numSmallSoc).
    if (numSmallSoc > 0) {
        int64_t offset_soc = numZeroCones + numNonnegCones;

        for (int64_t cone_idx = threadIdx.x; cone_idx < numSmallSoc; cone_idx += stride) {
            int64_t dim = d_soc_dims[cone_idx];
            int64_t soc_off = d_soc_offsets[cone_idx];     // for internal arrays
            int64_t sz_off = d_soc_sz_offsets[cone_idx];   // for s/z vectors
            int64_t hs_off = d_soc_Hs_offsets[cone_idx];

            int64_t s_base = batch * m_total + offset_soc + sz_off;
            int64_t z_base = batch * m_total + offset_soc + sz_off;

            // Compute SOC residual for s and z
            double s0 = s[s_base];
            double z0 = z[z_base];
            double s_tail_sq = 0.0, z_tail_sq = 0.0;
            for (int64_t i = 1; i < dim; i++) {
                s_tail_sq += s[s_base + i] * s[s_base + i];
                z_tail_sq += z[z_base + i] * z[z_base + i];
            }
            double s_tail_norm = sqrt(s_tail_sq);
            double z_tail_norm = sqrt(z_tail_sq);
            double s_res = (s0 - s_tail_norm) * (s0 + s_tail_norm);
            double z_res = (z0 - z_tail_norm) * (z0 + z_tail_norm);
            double sscale = (s_res > 0.0) ? sqrt(s_res) : 0.0;
            double zscale = (z_res > 0.0) ? sqrt(z_res) : 0.0;

            int64_t out_base_uvw = batch * totalSocDim + soc_off;
            int64_t out_base_d = batch * numSocCones + cone_idx;
            int64_t Hs_base = batch * totalSocHsEntries + hs_off;

            if (zscale == 0.0 || sscale == 0.0) {
                // s or z is not interior — signal failure (matches CPU Clarabel behavior)
                scaling_success[batch] = 0;
                // Still set safe identity-like values so parallel threads don't read garbage
                for (int64_t i = 1; i < dim; i++) {
                    soc_w[out_base_uvw + i] = 0.0;
                }
                soc_w[out_base_uvw] = 1.0;

                soc_eta[out_base_d] = 1.0;

                soc_lambda[out_base_uvw] = 1.0;
                for (int64_t i = 1; i < dim; i++) {
                    soc_lambda[out_base_uvw + i] = 0.0;
                }

                if (dim > 4) {
                    soc_u[out_base_uvw] = 0.7071067811865476;
                    for (int64_t i = 1; i < dim; i++) soc_u[out_base_uvw + i] = 0.0;
                    for (int64_t i = 0; i < dim; i++) soc_v[out_base_uvw + i] = 0.0;
                    soc_d[out_base_d] = 0.5;
                } else {
                    for (int64_t i = 0; i < dim; i++) {
                        soc_u[out_base_uvw + i] = 0.0;
                        soc_v[out_base_uvw + i] = 0.0;
                    }
                    soc_d[out_base_d] = 0.0;
                }

                write_identity_soc_Hs(Hs_base, 1.0, dim, 0.5);
                continue;
            }

            double eta = sqrt(sscale / zscale);

            // Construct w: w = s/sscale + J(z/zscale)
            double w0 = s0 / sscale + z0 / zscale;
            double w_tail_sq = 0.0;
            for (int64_t i = 1; i < dim; i++) {
                double wi = s[s_base + i] / sscale - z[z_base + i] / zscale;
                soc_w[out_base_uvw + i] = wi;  // temp store
                w_tail_sq += wi * wi;
            }
            double w_tail_norm = sqrt(w_tail_sq);
            double w_res = (w0 - w_tail_norm) * (w0 + w_tail_norm);
            double wscale = (w_res > 0.0) ? sqrt(w_res) : 0.0;
            if (wscale == 0.0) {
                // w is not interior — signal failure (matches CPU Clarabel behavior)
                scaling_success[batch] = 0;
                // Still set safe identity-like values so parallel threads don't read garbage
                soc_w[out_base_uvw] = 1.0;
                for (int64_t i = 1; i < dim; i++) {
                    soc_w[out_base_uvw + i] = 0.0;
                }

                soc_eta[out_base_d] = eta;

                soc_lambda[out_base_uvw] = sqrt(sscale * zscale);
                for (int64_t i = 1; i < dim; i++) {
                    soc_lambda[out_base_uvw + i] = 0.0;
                }

                if (dim > 4) {
                    soc_u[out_base_uvw] = 0.7071067811865476;
                    for (int64_t i = 1; i < dim; i++) soc_u[out_base_uvw + i] = 0.0;
                    for (int64_t i = 0; i < dim; i++) soc_v[out_base_uvw + i] = 0.0;
                    soc_d[out_base_d] = 0.5;
                } else {
                    for (int64_t i = 0; i < dim; i++) {
                        soc_u[out_base_uvw + i] = 0.0;
                        soc_v[out_base_uvw + i] = 0.0;
                    }
                    soc_d[out_base_d] = 0.0;
                }

                double eta_sq = eta * eta;
                write_identity_soc_Hs(Hs_base, eta_sq, dim, 0.5);
                continue;
            }

            // Normalize w
            w0 /= wscale;
            for (int64_t i = 1; i < dim; i++) {
                soc_w[out_base_uvw + i] /= wscale;
            }

            // Force normalization
            double w1sq = w_tail_sq / (wscale * wscale);
            w0 = sqrt(1.0 + w1sq);

            soc_w[out_base_uvw] = w0;
            soc_eta[out_base_d] = eta;

            double two = 2.0;
            double half = 0.5;
            double gamma = wscale * half;

            double a = (gamma + z0 / zscale) / sscale;
            double b = (gamma + s0 / sscale) / zscale;
            double denom = s0 / sscale + z0 / zscale + two * gamma;
            double lambda_scale = sqrt(sscale * zscale);

            soc_lambda[out_base_uvw] = gamma * lambda_scale;
            for (int64_t i = 1; i < dim; i++) {
                double li = (a * s[s_base + i] + b * z[z_base + i]) / denom;
                soc_lambda[out_base_uvw + i] = li * lambda_scale;
            }

            double eta_sq = eta * eta;

            if (dim > 4) {
                // Sparse expansion: compute u, v, d from w
                double alpha = two * w0;
                double wsq = w0 * w0 + w1sq;
                double wsqinv = 1.0 / wsq;
                double d_val = 0.5 * wsqinv;

                double u0 = sqrt(wsq - d_val);
                double u1_scale = alpha / u0;
                double v1_scale = sqrt(2.0 * (2.0 + wsqinv) / (2.0 * wsq - wsqinv));

                soc_u[out_base_uvw] = u0;
                for (int64_t i = 1; i < dim; i++) {
                    soc_u[out_base_uvw + i] = u1_scale * soc_w[out_base_uvw + i];
                }
                soc_v[out_base_uvw] = 0.0;
                for (int64_t i = 1; i < dim; i++) {
                    soc_v[out_base_uvw + i] = v1_scale * soc_w[out_base_uvw + i];
                }
                soc_d[out_base_d] = d_val;

                // Sparse Hs: diagonal only = η² * diag(d, 1, 1, ..., 1)
                soc_Hs[Hs_base] = eta_sq * d_val;
                for (int64_t i = 1; i < dim; i++) {
                    soc_Hs[Hs_base + i] = eta_sq;
                }
            } else {
                // Dense form: u, v, d not used
                for (int64_t i = 0; i < dim; i++) soc_u[out_base_uvw + i] = 0.0;
                for (int64_t i = 0; i < dim; i++) soc_v[out_base_uvw + i] = 0.0;
                soc_d[out_base_d] = 0.0;

                // Dense Hs: Hs = η² * (2*ww^T - J)
                constexpr double sqrt2 = cones::DEVICE_SQRT2;
                int64_t k = 0;
                for (int64_t i = 0; i < dim; i++) {
                    double wi = soc_w[out_base_uvw + i];
                    for (int64_t j = i; j < dim; j++) {
                        double wj = soc_w[out_base_uvw + j];
                        double val = eta_sq * two * wi * wj;
                        if (i == j) {
                            if (i == 0) {
                                // Avoid cancellation: η²(2w₀² - 1) = η²(√2·w₀ - 1)(√2·w₀ + 1)
                                val = eta_sq * (sqrt2 * wi - 1.0) * (sqrt2 * wi + 1.0);
                            } else {
                                val += eta_sq;  // -J diagonal: -(-1) = +1 for i>0
                            }
                        }
                        soc_Hs[Hs_base + k] = val;
                        k++;
                    }
                }
            }
        }
    }

    // ========== EXPONENTIAL CONES ==========
    if (numExpCones > 0) {
        int64_t offset_s = numZeroCones + numNonnegCones + totalSocDim;
        int64_t offset_z = numZeroCones + numNonnegCones + totalSocDim;

        for (int64_t cone_idx = threadIdx.x; cone_idx < numExpCones; cone_idx += stride) {
            int64_t s_base = batch * m_total + offset_s + cone_idx * 3;
            int64_t z_base = batch * m_total + offset_z + cone_idx * 3;

            double s_local[3] = {s[s_base], s[s_base + 1], s[s_base + 2]};
            double z_local[3] = {z[z_base], z[z_base + 1], z[z_base + 2]};

            double l = logsafe(-z_local[2] / z_local[0]);
            double r = -z_local[0] * l - z_local[0] + z_local[1];
            double c2 = 1.0 / r;

            double grad[3];
            grad[0] = c2 * l - 1.0 / z_local[0];
            grad[1] = -c2;
            grad[2] = (c2 * z_local[0] - 1.0) / z_local[2];

            double H[6];
            H[0] = (r * r - z_local[0] * r + l * l * z_local[0] * z_local[0]) / (r * z_local[0] * z_local[0] * r);
            H[1] = -l / (r * r);
            H[2] = (z_local[1] - z_local[0]) / (r * r * z_local[2]);
            H[3] = 1.0 / (r * r);
            H[4] = -z_local[0] / (r * r * z_local[2]);
            H[5] = (r * r - z_local[0] * r + z_local[0] * z_local[0]) / (r * r * z_local[2] * z_local[2]);

            double Hs[6];
            double three = 3.0;

            double zt[3];
            gradient_primal_exp(zt, s_local);
            double* st = grad;

            double dot_sz = dot3(s_local, z_local);
            double mu_local = dot_sz / three;
            double mu_t = dot3(st, zt) / three;

            double delta_s[3], delta_z[3];
            for (int i = 0; i < 3; i++) {
                delta_s[i] = s_local[i] + mu_local * st[i];
                delta_z[i] = z_local[i] + mu_local * zt[i];
            }
            double dot_delta_sz = dot3(delta_s, delta_z);

            double de1 = mu_local * mu_t - 1.0;
            double de2 = quad_form3(H, zt) - three * mu_t * mu_t;

            bool use_primal_dual = (fabs(de1) > DEVICE_SQRT_EPSILON) &&
                                   (fabs(de2) > DEVICE_EPSILON) &&
                                   (dot_sz > 0.0) &&
                                   (dot_delta_sz > 0.0);

            if (use_primal_dual) {
                double tmp[3];
                matvec3_sym(tmp, H, zt);
                for (int i = 0; i < 3; i++) {
                    tmp[i] = mu_t * st[i] - tmp[i];
                }

                for (int i = 0; i < 3; i++) {
                    for (int j = i; j < 3; j++) {
                        int idx = sym_idx(i, j);
                        Hs[idx] = H[idx] - st[i] * st[j] / three - tmp[i] * tmp[j] / de2;
                    }
                }
                double t = mu_local * norm_fro3(Hs);

                double axis_z[3];
                axis_z[0] = z_local[1] * zt[2] - z_local[2] * zt[1];
                axis_z[1] = z_local[2] * zt[0] - z_local[0] * zt[2];
                axis_z[2] = z_local[0] * zt[1] - z_local[1] * zt[0];
                normalize3(axis_z);

                for (int i = 0; i < 3; i++) {
                    for (int j = i; j < 3; j++) {
                        int idx = sym_idx(i, j);
                        Hs[idx] = s_local[i] * s_local[j] / dot_sz +
                                  delta_s[i] * delta_s[j] / dot_delta_sz +
                                  t * axis_z[i] * axis_z[j];
                    }
                }
            } else {
                for (int i = 0; i < 6; i++) {
                    Hs[i] = mu_local * H[i];
                }
            }

            int64_t out_base_grad = batch * (numExpCones * 3) + cone_idx * 3;
            int64_t out_base_H = batch * (numExpCones * 6) + cone_idx * 6;
            int64_t out_base_z = batch * (numExpCones * 3) + cone_idx * 3;

            exp_grad[out_base_grad] = grad[0];
            exp_grad[out_base_grad + 1] = grad[1];
            exp_grad[out_base_grad + 2] = grad[2];

            for (int i = 0; i < 6; i++) {
                exp_H_dual[out_base_H + i] = H[i];
                exp_Hs[out_base_H + i] = Hs[i];
            }

            exp_z[out_base_z] = z_local[0];
            exp_z[out_base_z + 1] = z_local[1];
            exp_z[out_base_z + 2] = z_local[2];
        }
    }

    // ========== POWER CONES ==========
    if (numPowerCones > 0) {
        int64_t offset_s = numZeroCones + numNonnegCones + totalSocDim + numExpCones * 3;
        int64_t offset_z = numZeroCones + numNonnegCones + totalSocDim + numExpCones * 3;
        double mu_val = mu[batch];

        for (int64_t cone_idx = threadIdx.x; cone_idx < numPowerCones; cone_idx += stride) {
            int64_t z_base = batch * m_total + offset_z + cone_idx * 3;

            double z_local[3] = {z[z_base], z[z_base + 1], z[z_base + 2]};
            double alpha = d_powerAlphas[cone_idx];

            double two = 2.0;
            double four = 4.0;

            double phi = pow(z_local[0] / alpha, two * alpha) * pow(z_local[1] / (1.0 - alpha), two - two * alpha);
            // Avoid catastrophic cancellation: phi - z[2]² = (√φ - |z₂|)(√φ + |z₂|)
            double sqrt_phi = sqrt(phi);
            double abs_z2 = fabs(z_local[2]);
            double psi = (sqrt_phi - abs_z2) * (sqrt_phi + abs_z2);

            double g_psi[3];
            g_psi[0] = two * alpha * phi / (z_local[0] * psi);
            g_psi[1] = two * (1.0 - alpha) * phi / (z_local[1] * psi);
            g_psi[2] = -two * z_local[2] / psi;

            double H[6];
            H[0] = g_psi[0] * g_psi[0] - two * alpha * (two * alpha - 1.0) * phi / (z_local[0] * z_local[0] * psi)
                + (1.0 - alpha) / (z_local[0] * z_local[0]);
            H[1] = g_psi[0] * g_psi[1] - four * alpha * (1.0 - alpha) * phi / (z_local[0] * z_local[1] * psi);
            H[2] = g_psi[0] * g_psi[2];
            H[3] = g_psi[1] * g_psi[1]
                - two * (1.0 - alpha) * (1.0 - two * alpha) * phi / (z_local[1] * z_local[1] * psi)
                + alpha / (z_local[1] * z_local[1]);
            H[4] = g_psi[1] * g_psi[2];
            H[5] = g_psi[2] * g_psi[2] + two / psi;

            double grad[3];
            grad[0] = -two * alpha * phi / (z_local[0] * psi) - (1.0 - alpha) / z_local[0];
            grad[1] = -two * (1.0 - alpha) * phi / (z_local[1] * psi) - alpha / z_local[1];
            grad[2] = two * z_local[2] / psi;

            double Hs[6];

            if (scaling_strategy == 1) {
                for (int i = 0; i < 6; i++) {
                    Hs[i] = mu_val * H[i];
                }
            } else {
                int64_t s_base = batch * m_total + offset_s + cone_idx * 3;
                double s_local[3] = {s[s_base], s[s_base + 1], s[s_base + 2]};

                double zt[3];
                gradient_primal_power(zt, s_local, alpha);
                double* st = grad;

                double three = 3.0;
                double dot_sz = dot3(s_local, z_local);
                double mu_local = dot_sz / three;
                double mu_t = dot3(st, zt) / three;

                double delta_s[3], delta_z[3];
                for (int i = 0; i < 3; i++) {
                    delta_s[i] = s_local[i] + mu_local * st[i];
                    delta_z[i] = z_local[i] + mu_local * zt[i];
                }
                double dot_delta_sz = dot3(delta_s, delta_z);

                double de1 = mu_local * mu_t - 1.0;
                double de2 = quad_form3(H, zt) - three * mu_t * mu_t;

                bool use_primal_dual = (fabs(de1) > DEVICE_SQRT_EPSILON) &&
                                       (fabs(de2) > DEVICE_EPSILON) &&
                                       (dot_sz > 0.0) &&
                                       (dot_delta_sz > 0.0);

                if (use_primal_dual) {
                    double tmp[3];
                    matvec3_sym(tmp, H, zt);
                    for (int i = 0; i < 3; i++) {
                        tmp[i] = mu_t * st[i] - tmp[i];
                    }

                    double Hs_temp[6];
                    for (int i = 0; i < 6; i++) {
                        Hs_temp[i] = H[i];
                    }
                    for (int i = 0; i < 3; i++) {
                        for (int j = i; j < 3; j++) {
                            int idx = sym_idx(i, j);
                            Hs_temp[idx] -= st[i] * st[j] / three + tmp[i] * tmp[j] / de2;
                        }
                    }

                    double norm_fro_sq = Hs_temp[0] * Hs_temp[0] + Hs_temp[3] * Hs_temp[3] + Hs_temp[5] * Hs_temp[5] +
                                         2.0 * (Hs_temp[1] * Hs_temp[1] + Hs_temp[2] * Hs_temp[2] + Hs_temp[4] * Hs_temp[4]);
                    double t = mu_local * sqrt(norm_fro_sq);

                    double axis_z[3];
                    axis_z[0] = z_local[1] * zt[2] - z_local[2] * zt[1];
                    axis_z[1] = z_local[2] * zt[0] - z_local[0] * zt[2];
                    axis_z[2] = z_local[0] * zt[1] - z_local[1] * zt[0];
                    double norm_axis = sqrt(dot3(axis_z, axis_z));
                    if (norm_axis > 1e-16) {
                        for (int i = 0; i < 3; i++) axis_z[i] /= norm_axis;
                    }

                    for (int i = 0; i < 3; i++) {
                        for (int j = i; j < 3; j++) {
                            int idx = sym_idx(i, j);
                            Hs[idx] = s_local[i] * s_local[j] / dot_sz
                                    + delta_s[i] * delta_s[j] / dot_delta_sz
                                    + t * axis_z[i] * axis_z[j];
                        }
                    }
                } else {
                    for (int i = 0; i < 6; i++) {
                        Hs[i] = mu_val * H[i];
                    }
                }
            }

            int64_t out_base_grad = batch * (numPowerCones * 3) + cone_idx * 3;
            int64_t out_base_H = batch * (numPowerCones * 6) + cone_idx * 6;
            int64_t out_base_z = batch * (numPowerCones * 3) + cone_idx * 3;

            power_grad[out_base_grad] = grad[0];
            power_grad[out_base_grad + 1] = grad[1];
            power_grad[out_base_grad + 2] = grad[2];

            for (int i = 0; i < 6; i++) {
                power_H_dual[out_base_H + i] = H[i];
                power_Hs[out_base_H + i] = Hs[i];
            }

            power_z[out_base_z] = z_local[0];
            power_z[out_base_z + 1] = z_local[1];
            power_z[out_base_z + 2] = z_local[2];
        }
    }

    // ========== GENERALIZED POWER CONES ==========
    // Dual-only scaling: compute gradient and Hessian of dual barrier, store Hs.
    // Only small cones (dim <= GENPOW_PARALLEL_THRESHOLD) are processed here;
    // large cones are handled by update_large_genpow_scaling launched from
    // the host. Cones are sorted ascending in Cones::initialize(), so small
    // cones are the contiguous prefix [0, numSmallGenPow).
    if (numSmallGenPow > 0) {
        int64_t offset_z = numZeroCones + numNonnegCones + totalSocDim + numExpCones * 3 + numPowerCones * 3;
        double mu_val = mu[batch];

        for (int64_t cone_idx = threadIdx.x; cone_idx < numSmallGenPow; cone_idx += stride) {
            int64_t dim1 = d_genPowerDim1s[cone_idx];
            int64_t dim2 = d_genPowerDim2s[cone_idx];
            int64_t dim = dim1 + dim2;
            int64_t gp_off = d_genPowerOffsets[cone_idx];
            int64_t alpha_off = d_genPowerAlphaOffsets[cone_idx];
            int64_t hs_off = d_genPowerHsOffsets[cone_idx];
            int64_t sz_off = d_genPowerSzOffsets[cone_idx];
            int64_t z_base = batch * m_total + offset_z + sz_off;
            int64_t grad_base = batch * totalGenPowerDim + gp_off;
            int64_t z_out_base = grad_base;  // same layout
            int64_t Hs_base = batch * totalGenPowerHsEntries + hs_off;

            // p/q/r/d1 use same offsets as grad for the per-cone data
            // p is dim-length, q is dim1-length, r is dim2-length, d1 is dim1-length
            int64_t p_base = grad_base;  // dim entries
            int64_t q_base = batch * totalGenPowerAlphas + alpha_off;  // dim1 entries

            // Check dual feasibility (z_i > 0) before computing log
            // to avoid NaN from log of non-positive values
            bool dual_infeasible = false;
            for (int64_t i = 0; i < dim1; i++) {
                if (z[z_base + i] <= 0.0) { dual_infeasible = true; break; }
            }
            if (dual_infeasible) {
                if (scaling_success) scaling_success[batch] = 0;
                continue;
            }

            // Compute phi = ∏(z_i/αi)^{2αi} — use log-sum-exp for numerical stability
            double log_phi = 0.0;
            for (int64_t i = 0; i < dim1; i++) {
                double zi = z[z_base + i];
                double ai = d_genPowerAlphas[alpha_off + i];
                log_phi += 2.0 * ai * log(zi / ai);
            }
            double phi = exp(log_phi);

            // Compute ||w||²
            double norm2w = 0.0;
            for (int64_t i = 0; i < dim2; i++) {
                double wi = z[z_base + dim1 + i];
                norm2w += wi * wi;
            }

            double zeta = phi - norm2w;
            if (zeta <= 0.0) {
                if (scaling_success) scaling_success[batch] = 0;
                continue;
            }

            // Compute gradient
            // tau[i] = 2*αi/zi (reuse q storage temporarily)
            // grad[i<dim1] = -tau[i]*phi/zeta - (1-αi)/zi
            // grad[i>=dim1] = 2*zi/zeta
            for (int64_t i = 0; i < dim1; i++) {
                double zi = z[z_base + i];
                double ai = d_genPowerAlphas[alpha_off + i];
                double tau_i = 2.0 * ai / zi;
                genpow_grad[grad_base + i] = -tau_i * phi / zeta - (1.0 - ai) / zi;
                // Store tau temporarily for Hessian computation
                genpow_q[q_base + i] = tau_i;
            }
            for (int64_t i = 0; i < dim2; i++) {
                genpow_grad[grad_base + dim1 + i] = 2.0 * z[z_base + dim1 + i] / zeta;
            }

            // Store z copy
            for (int64_t i = 0; i < dim; i++) {
                genpow_z_out[z_out_base + i] = z[z_base + i];
            }

            // Compute Hessian decomposition: H = D + pp' - qq' - rr'
            constexpr double eps = 1e-300;
            double p0 = sqrt(phi * (phi + norm2w) / 2.0);
            double p1 = (p0 > eps) ? -2.0 * phi / p0 : 0.0;
            double q0 = sqrt(zeta * phi / 2.0);
            double phi_plus_norm2w = phi + norm2w;
            double r1 = (phi_plus_norm2w > eps) ? 2.0 * sqrt(zeta / phi_plus_norm2w) : 0.0;

            // d1[i] = tau[i]*phi/(zeta*z[i]) + (1-αi)/(z[i]*z[i])
            int64_t d1_base = batch * totalGenPowerAlphas + alpha_off;
            for (int64_t i = 0; i < dim1; i++) {
                double zi = z[z_base + i];
                double ai = d_genPowerAlphas[alpha_off + i];
                double tau_i = genpow_q[q_base + i];  // tau stored in q
                genpow_d1[d1_base + i] = tau_i * phi / (zeta * zi) + (1.0 - ai) / (zi * zi);
            }
            // d2 = 2/zeta (scalar, store once per cone)
            double d2_val = 2.0 / zeta;
            genpow_d2[batch * numGenPowerCones + cone_idx] = d2_val;

            // p[..dim1] = (p0/zeta)*tau, p[dim1..] = (p1/zeta)*z[dim1..]
            for (int64_t i = 0; i < dim1; i++) {
                double tau_i = genpow_q[q_base + i];
                genpow_p[p_base + i] = (p0 / zeta) * tau_i;
            }
            for (int64_t i = 0; i < dim2; i++) {
                genpow_p[p_base + dim1 + i] = (p1 / zeta) * z[z_base + dim1 + i];
            }

            // q = (q0/zeta)*tau (q was tau, now scale)
            for (int64_t i = 0; i < dim1; i++) {
                genpow_q[q_base + i] *= (q0 / zeta);
            }

            // r[..dim2] = (r1/zeta)*z[dim1..]
            // r is BatchedVector(totalGenPowerDim - totalGenPowerAlphas, batchSize)
            // gp_off - alpha_off = sum of dim2_j for j<cone_idx
            int64_t r_base_cone = batch * (totalGenPowerDim - totalGenPowerAlphas) + (gp_off - alpha_off);
            for (int64_t i = 0; i < dim2; i++) {
                genpow_r[r_base_cone + i] = (r1 / zeta) * z[z_base + dim1 + i];
            }

            // Build Hs = μ*(D + pp' - qq' - rr')
            if (dim > 4) {
                // Sparse: store D diagonal only (expansion columns carry rank-3 terms pp'-qq'-rr')
                for (int64_t i = 0; i < dim; i++) {
                    double d_val_i = (i < dim1) ? genpow_d1[d1_base + i] : d2_val;
                    genpow_Hs[Hs_base + i] = mu_val * d_val_i;
                }
            } else {
                // Dense: full upper triangle
                int64_t hs_idx = 0;
                for (int64_t i = 0; i < dim; i++) {
                    for (int64_t j = i; j < dim; j++) {
                        double val = 0.0;

                        // Diagonal part D
                        if (i == j) {
                            if (i < dim1) {
                                val += genpow_d1[d1_base + i];
                            } else {
                                val += d2_val;
                            }
                        }

                        // + pp'
                        val += genpow_p[p_base + i] * genpow_p[p_base + j];

                        // - qq' (only when both i,j < dim1)
                        if (i < dim1 && j < dim1) {
                            val -= genpow_q[q_base + i] * genpow_q[q_base + j];
                        }

                        // - rr' (only when both i,j >= dim1)
                        if (i >= dim1 && j >= dim1) {
                            val -= genpow_r[r_base_cone + (i - dim1)] * genpow_r[r_base_cone + (j - dim1)];
                        }

                        genpow_Hs[Hs_base + hs_idx] = mu_val * val;
                        hs_idx++;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Block-per-cone GenPowerCone scaling for high-dimensional cones
// ============================================================================
//
// For GenPowerCones with dim = dim1 + dim2 > GENPOW_PARALLEL_THRESHOLD,
// the composite kernel's one-thread-per-cone loop becomes a serial reduction
// over dim1 (log_phi) and dim2 (||w||²). This kernel assigns one block per
// (batch, large_cone) and uses shared-memory reductions so all block threads
// cooperate on the norm computations. Per-entry writes (grad, p, q, r, d1,
// Hs diagonal) are also parallelized across the block.
//
// Large cones always take the sparse (dim > 4) path for Hs layout.
// Sorted-ascending order in Cones::initialize() means large cones occupy
// the contiguous suffix [numSmallGenPow, numGenPowerCones).
//
// Early exits mirror the composite kernel: dual-infeasible (any z_i <= 0
// for i<dim1) or zeta <= 0 set scaling_success[batch]=0 and leave outputs
// unchanged. All threads must branch together, so these predicates are
// broadcast via a shared flag.

__global__ void update_large_genpow_scaling_kernel(
    double* __restrict__ genpow_grad,
    double* __restrict__ genpow_z_out,
    double* __restrict__ genpow_Hs,
    double* __restrict__ genpow_p,
    double* __restrict__ genpow_q,
    double* __restrict__ genpow_r,
    double* __restrict__ genpow_d1,
    double* __restrict__ genpow_d2,
    const double* __restrict__ z,
    const double* __restrict__ mu,
    const double* __restrict__ d_genPowerAlphas,
    const int64_t* __restrict__ d_genPowerDim1s,
    const int64_t* __restrict__ d_genPowerDim2s,
    const int64_t* __restrict__ d_genPowerOffsets,
    const int64_t* __restrict__ d_genPowerAlphaOffsets,
    const int64_t* __restrict__ d_genPowerHsOffsets,
    const int64_t* __restrict__ d_genPowerSzOffsets,
    int64_t offset_z_cones,
    int64_t m_total,
    int64_t numGenPowerCones,
    int64_t numSmallGenPow,
    int64_t numLargeGenPow,
    int64_t batchSize,
    int64_t totalGenPowerDim,
    int64_t totalGenPowerAlphas,
    int64_t totalGenPowerHsEntries,
    int32_t* __restrict__ scaling_success
) {
    int64_t large_idx = blockIdx.x;
    int64_t batch = blockIdx.y;
    if (large_idx >= numLargeGenPow || batch >= batchSize) return;

    const int tid = threadIdx.x;
    const int blockDimX = blockDim.x;

    int64_t cone_idx = numSmallGenPow + large_idx;
    int64_t dim1 = d_genPowerDim1s[cone_idx];
    int64_t dim2 = d_genPowerDim2s[cone_idx];
    int64_t dim = dim1 + dim2;
    int64_t gp_off = d_genPowerOffsets[cone_idx];
    int64_t alpha_off = d_genPowerAlphaOffsets[cone_idx];
    int64_t hs_off = d_genPowerHsOffsets[cone_idx];
    int64_t sz_off = d_genPowerSzOffsets[cone_idx];

    int64_t z_base = batch * m_total + offset_z_cones + sz_off;
    int64_t grad_base = batch * totalGenPowerDim + gp_off;
    int64_t Hs_base = batch * totalGenPowerHsEntries + hs_off;
    int64_t p_base = grad_base;
    int64_t q_base = batch * totalGenPowerAlphas + alpha_off;
    int64_t d1_base = q_base;
    int64_t r_base = batch * (totalGenPowerDim - totalGenPowerAlphas) + (gp_off - alpha_off);

    extern __shared__ double smem[];
    double* scratch = smem;

    // ---- Step 1: reduce log_phi and dual-feasibility flag in one pass ----
    // For threads processing z_i <= 0, log(z_i/ai) is undefined; contribute 0
    // to log_phi and 1 to the infeasibility count. The sum-as-OR works because
    // we only need to distinguish "any bad" from "none bad".
    double my_log_phi = 0.0;
    double my_infeasible = 0.0;
    for (int64_t i = tid; i < dim1; i += blockDimX) {
        double zi = z[z_base + i];
        double ai = d_genPowerAlphas[alpha_off + i];
        if (zi > 0.0) {
            my_log_phi += 2.0 * ai * log(zi / ai);
        } else {
            my_infeasible += 1.0;
        }
    }
    double log_phi = cones::block_sum_reduce(my_log_phi, scratch, tid);
    double infeasible_count = cones::block_sum_reduce(my_infeasible, scratch, tid);

    if (infeasible_count > 0.0) {
        if (tid == 0 && scaling_success) scaling_success[batch] = 0;
        return;
    }

    // ---- Step 2: reduce ||w||² ----
    double my_norm2w = 0.0;
    for (int64_t i = tid; i < dim2; i += blockDimX) {
        double wi = z[z_base + dim1 + i];
        my_norm2w += wi * wi;
    }
    double norm2w = cones::block_sum_reduce(my_norm2w, scratch, tid);

    double phi = exp(log_phi);
    double zeta = phi - norm2w;
    if (zeta <= 0.0) {
        if (tid == 0 && scaling_success) scaling_success[batch] = 0;
        return;
    }

    // ---- Step 3: scalar coefficients derived by every thread ----
    constexpr double eps = 1e-300;
    double p0 = sqrt(phi * (phi + norm2w) / 2.0);
    double p1 = (p0 > eps) ? -2.0 * phi / p0 : 0.0;
    double q0 = sqrt(zeta * phi / 2.0);
    double phi_plus_norm2w = phi + norm2w;
    double r1 = (phi_plus_norm2w > eps) ? 2.0 * sqrt(zeta / phi_plus_norm2w) : 0.0;
    double d2_val = 2.0 / zeta;
    double mu_val = mu[batch];

    if (tid == 0) {
        genpow_d2[batch * numGenPowerCones + cone_idx] = d2_val;
    }

    // ---- Step 4: parallel writes for the dim1 block ----
    // Each thread processes distinct i values, so reads of z[i]/alphas[i]
    // never race with writes to grad/d1/p/q/Hs (all indexed by the same i).
    double inv_zeta = 1.0 / zeta;
    double p0_over_zeta = p0 * inv_zeta;
    double q0_over_zeta = q0 * inv_zeta;
    double phi_over_zeta = phi * inv_zeta;

    for (int64_t i = tid; i < dim1; i += blockDimX) {
        double zi = z[z_base + i];
        double ai = d_genPowerAlphas[alpha_off + i];
        double tau = 2.0 * ai / zi;
        double inv_zi = 1.0 / zi;
        double d1_val = tau * phi_over_zeta * inv_zi + (1.0 - ai) * inv_zi * inv_zi;

        genpow_grad[grad_base + i] = -tau * phi_over_zeta - (1.0 - ai) * inv_zi;
        genpow_z_out[grad_base + i] = zi;
        genpow_d1[d1_base + i] = d1_val;
        genpow_p[p_base + i] = p0_over_zeta * tau;
        genpow_q[q_base + i] = q0_over_zeta * tau;
        genpow_Hs[Hs_base + i] = mu_val * d1_val;  // sparse diagonal
    }

    // ---- Step 5: parallel writes for the dim2 block ----
    double p1_over_zeta = p1 * inv_zeta;
    double r1_over_zeta = r1 * inv_zeta;
    double two_over_zeta = 2.0 * inv_zeta;
    double mu_d2 = mu_val * d2_val;

    for (int64_t i = tid; i < dim2; i += blockDimX) {
        double wi = z[z_base + dim1 + i];
        genpow_grad[grad_base + dim1 + i] = two_over_zeta * wi;
        genpow_z_out[grad_base + dim1 + i] = wi;
        genpow_p[p_base + dim1 + i] = p1_over_zeta * wi;
        genpow_r[r_base + i] = r1_over_zeta * wi;
        genpow_Hs[Hs_base + dim1 + i] = mu_d2;  // sparse diagonal
    }
}

void update_large_genpow_scaling(
    double* genpow_grad,
    double* genpow_z_out,
    double* genpow_Hs,
    double* genpow_p,
    double* genpow_q,
    double* genpow_r,
    double* genpow_d1,
    double* genpow_d2,
    const double* z,
    const double* mu,
    const double* d_genPowerAlphas,
    const int64_t* d_genPowerDim1s,
    const int64_t* d_genPowerDim2s,
    const int64_t* d_genPowerOffsets,
    const int64_t* d_genPowerAlphaOffsets,
    const int64_t* d_genPowerHsOffsets,
    const int64_t* d_genPowerSzOffsets,
    int64_t offset_z_cones,
    int64_t m_total,
    int64_t numGenPowerCones,
    int64_t numSmallGenPow,
    int64_t numLargeGenPow,
    int64_t batchSize,
    int64_t totalGenPowerDim,
    int64_t totalGenPowerAlphas,
    int64_t totalGenPowerHsEntries,
    int32_t* scaling_success,
    cudaStream_t stream
) {
    if (numLargeGenPow == 0 || batchSize == 0) return;

    const int block_size = cones::GENPOW_PARALLEL_BLOCK_SIZE;
    dim3 grid(static_cast<unsigned int>(numLargeGenPow),
              static_cast<unsigned int>(batchSize));
    dim3 block(block_size);
    size_t smem_bytes = sizeof(double) * block_size;

    MOREAU_KERNEL_LAUNCH(update_large_genpow_scaling_kernel, grid, block, smem_bytes, stream,
        genpow_grad, genpow_z_out, genpow_Hs,
        genpow_p, genpow_q, genpow_r, genpow_d1, genpow_d2,
        z, mu,
        d_genPowerAlphas, d_genPowerDim1s, d_genPowerDim2s,
        d_genPowerOffsets, d_genPowerAlphaOffsets,
        d_genPowerHsOffsets, d_genPowerSzOffsets,
        offset_z_cones, m_total,
        numGenPowerCones, numSmallGenPow, numLargeGenPow, batchSize,
        totalGenPowerDim, totalGenPowerAlphas, totalGenPowerHsEntries,
        scaling_success
    );
}

bool update_scaling_all_cones(
    double* nonneg_w,
    double* nonneg_lambda,
    double* soc_u,
    double* soc_v,
    double* soc_d,
    double* soc_w,
    double* soc_eta,
    double* soc_lambda,
    double* soc_Hs,
    double* exp_grad,
    double* exp_H_dual,
    double* exp_Hs,
    double* exp_z,
    double* power_grad,
    double* power_H_dual,
    double* power_Hs,
    double* power_z,
    const double* s,
    const double* z,
    const double* mu,
    const double* d_powerAlphas,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m_total,
    int64_t batchSize,
    int scaling_strategy,
    cudaStream_t stream,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_Hs_offsets,
    int64_t totalSocDim,
    int64_t totalSocHsEntries,
    int32_t* d_scaling_success,
    const int64_t* d_soc_sz_offsets,
    volatile int32_t* h_scaling_success_pinned,
    // GenPowerCone params
    double* genpow_grad,
    double* genpow_z_out,
    double* genpow_Hs,
    double* genpow_p,
    double* genpow_q,
    double* genpow_r,
    double* genpow_d1,
    double* genpow_d2,
    const double* d_genPowerAlphas,
    const int64_t* d_genPowerDim1s,
    const int64_t* d_genPowerDim2s,
    const int64_t* d_genPowerOffsets,
    const int64_t* d_genPowerAlphaOffsets,
    const int64_t* d_genPowerHsOffsets,
    const int64_t* d_genPowerSzOffsets,
    int64_t numGenPowerCones,
    int64_t totalGenPowerDim,
    int64_t totalGenPowerAlphas,
    int64_t totalGenPowerGradEntries,
    int64_t totalGenPowerHsEntries,
    int64_t numLargeSoc,
    int64_t numLargeGenPow
) {
    int64_t numSmallSoc = numSocCones - numLargeSoc;
    int64_t numSmallGenPow = numGenPowerCones - numLargeGenPow;

    int64_t maxCones = numNonnegCones;
    maxCones = (numSmallSoc > maxCones) ? numSmallSoc : maxCones;
    maxCones = (numExpCones > maxCones) ? numExpCones : maxCones;
    maxCones = (numPowerCones > maxCones) ? numPowerCones : maxCones;
    maxCones = (numSmallGenPow > maxCones) ? numSmallGenPow : maxCones;

    // Initialize per-batch scaling success flags to 1 (success);
    // kernel sets to 0 on degenerate SOC/GenPowerCone state for that batch.
    // d_scaling_success is a device-mapped pointer into h_scaling_success_pinned,
    // so writing to the host pointer is visible to the GPU (and vice versa).
    if ((numSocCones > 0 || numGenPowerCones > 0) && h_scaling_success_pinned) {
        std::fill_n(h_scaling_success_pinned, batchSize, 1);
    }

    if (maxCones > 0) {
        int threadsPerBlock = (maxCones < 256) ? ((maxCones + 31) / 32 * 32) : 256;
        if (threadsPerBlock < 32) threadsPerBlock = 32;

        MOREAU_KERNEL_LAUNCH(update_scaling_all_cones_kernel, batchSize, threadsPerBlock, 0, stream,
            nonneg_w, nonneg_lambda,
            soc_u, soc_v, soc_d, soc_w, soc_eta, soc_lambda, soc_Hs,
            exp_grad, exp_H_dual, exp_Hs, exp_z,
            power_grad, power_H_dual, power_Hs, power_z,
            genpow_grad, genpow_z_out, genpow_Hs, genpow_p, genpow_q, genpow_r, genpow_d1, genpow_d2,
            s, z, mu, d_powerAlphas,
            d_genPowerAlphas, d_genPowerDim1s, d_genPowerDim2s,
            d_genPowerOffsets, d_genPowerAlphaOffsets, d_genPowerHsOffsets, d_genPowerSzOffsets,
            numGenPowerCones, numSmallGenPow, totalGenPowerDim, totalGenPowerAlphas,
            totalGenPowerGradEntries, totalGenPowerHsEntries,
            numZeroCones, numNonnegCones, numSocCones, numSmallSoc,
            numExpCones, numPowerCones,
            m_total, batchSize, scaling_strategy,
            d_soc_dims, d_soc_offsets, d_soc_Hs_offsets, totalSocDim, totalSocHsEntries,
            d_scaling_success, d_soc_sz_offsets
        );
    }

    // Large SOC cones (dim > SOC_PARALLEL_THRESHOLD) use a block-per-cone
    // kernel that parallelizes the intra-cone reductions. Both kernels
    // share the s/z inputs (read-only) and write to disjoint per-cone slices
    // of the output arrays, so no explicit sync is required between them.
    if (numLargeSoc > 0) {
        int64_t offset_soc = numZeroCones + numNonnegCones;
        update_large_soc_scaling(
            soc_u, soc_v, soc_d, soc_w, soc_eta, soc_lambda, soc_Hs,
            s, z, offset_soc, m_total,
            numSocCones, numSmallSoc, numLargeSoc, batchSize,
            d_scaling_success,
            d_soc_dims, d_soc_offsets, d_soc_Hs_offsets, d_soc_sz_offsets,
            totalSocDim, totalSocHsEntries,
            stream
        );
    }

    // Large GenPowerCones (dim > GENPOW_PARALLEL_THRESHOLD) use a
    // block-per-cone kernel. Disjoint output slices, shared read-only z/mu
    // inputs — safe to launch concurrently with the composite kernel.
    if (numLargeGenPow > 0) {
        int64_t offset_z_cones = numZeroCones + numNonnegCones + totalSocDim
                               + numExpCones * 3 + numPowerCones * 3;
        update_large_genpow_scaling(
            genpow_grad, genpow_z_out, genpow_Hs,
            genpow_p, genpow_q, genpow_r, genpow_d1, genpow_d2,
            z, mu,
            d_genPowerAlphas, d_genPowerDim1s, d_genPowerDim2s,
            d_genPowerOffsets, d_genPowerAlphaOffsets,
            d_genPowerHsOffsets, d_genPowerSzOffsets,
            offset_z_cones, m_total,
            numGenPowerCones, numSmallGenPow, numLargeGenPow, batchSize,
            totalGenPowerDim, totalGenPowerAlphas, totalGenPowerHsEntries,
            d_scaling_success, stream
        );
    }

    // No sync here — the caller reads h_scaling_success_pinned later,
    // after the kernel has naturally completed.
    return true;
}

// ============================================================================
// Margins Computation
// ============================================================================

// MarginResults is defined in cones.hpp

__global__ void compute_margins_impl(
    const double* __restrict__ z,
    MarginResults* batch_results,
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
    const int64_t* __restrict__ d_soc_sz_offsets
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    extern __shared__ double shared_mem[];
    double* s_min = shared_mem;
    double* s_pos = shared_mem + blockDim.x;

    double local_min = 1e300;
    double local_pos = 0.0;

    int64_t offset = batch * m_total;
    offset += numZeroCones;  // Skip zero cones

    // Nonnegative cones
    int64_t nonneg_start = offset;
    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        double val = z[nonneg_start + i];
        local_min = fmin(local_min, val);
        local_pos += fmax(val, 0.0);
    }
    offset += numNonnegCones;

    // SOC cones (use d_soc_sz_offsets for s/z access)
    int64_t soc_start = offset;
    for (int64_t i = threadIdx.x; i < numSocCones; i += blockDim.x) {
        int64_t dim = d_soc_dims[i];
        int64_t sz_off = d_soc_sz_offsets[i];
        double t = z[soc_start + sz_off];
        double tail_sq = 0.0;
        for (int64_t j = 1; j < dim; j++) {
            tail_sq += z[soc_start + sz_off + j] * z[soc_start + sz_off + j];
        }
        double norm_x = sqrt(tail_sq);
        double margin = t - norm_x;
        local_min = fmin(local_min, margin);
        local_pos += fmax(margin, 0.0);
    }
    offset += totalSocDim;

    // Exp cones (conservative margin)
    int64_t exp_start = offset;
    for (int64_t i = threadIdx.x; i < numExpCones; i += blockDim.x) {
        double v1 = z[exp_start + i * 3 + 1];
        double v2 = z[exp_start + i * 3 + 2];
        double margin = fmin(v1, v2);
        local_min = fmin(local_min, margin);
        local_pos += fmax(margin, 0.0);
    }
    offset += numExpCones * 3;

    // Power cones (conservative margin)
    int64_t power_start = offset;
    for (int64_t i = threadIdx.x; i < numPowerCones; i += blockDim.x) {
        double v0 = z[power_start + i * 3];
        double v1 = z[power_start + i * 3 + 1];
        double margin = fmin(v0, v1);
        local_min = fmin(local_min, margin);
        local_pos += fmax(margin, 0.0);
    }
    // Note: GenPowerCone margins are unreachable (asymmetric cones use unit_initialization)
    // but if we add them, conservative margin = min of p components (z[0..dim1])

    s_min[threadIdx.x] = local_min;
    s_pos[threadIdx.x] = local_pos;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_min[threadIdx.x] = fmin(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
            s_pos[threadIdx.x] += s_pos[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        batch_results[batch].min_margin = s_min[0];
        batch_results[batch].pos_margin = s_pos[0];
    }
}


// ============================================================================
// Per-batch Margins Computation (no cross-batch reduction)
// ============================================================================

__global__ void copy_margin_results_kernel(
    double* __restrict__ d_min_margin_out,
    double* __restrict__ d_pos_margin_out,
    const MarginResults* d_batch_results,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    d_min_margin_out[batch] = d_batch_results[batch].min_margin;
    d_pos_margin_out[batch] = d_batch_results[batch].pos_margin;
}

void compute_margins_batched_kernel(
    const double* z,
    double* d_min_margin_out,
    double* d_pos_margin_out,
    MarginResults* d_batch_results,
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
    const int64_t* d_soc_sz_offsets
) {
    if (m_total == 0 || batchSize == 0) {
        // Set output to default values (large min margin means no constraint)
        double large_val = 1e300;
        for (int64_t b = 0; b < batchSize; b++) {
            cudaMemcpyAsync(&d_min_margin_out[b], &large_val, sizeof(double), cudaMemcpyHostToDevice, stream);
        }
        cudaMemsetAsync(d_pos_margin_out, 0, sizeof(double) * batchSize, stream);
        return;
    }

    int threadsPerBlock = 256;
    size_t sharedMemSize = 2 * threadsPerBlock * sizeof(double);

    // Step 1: Compute per-batch margins (same kernel as before)
    MOREAU_KERNEL_LAUNCH(compute_margins_impl, batchSize, threadsPerBlock, sharedMemSize, stream,
        z, d_batch_results,
        numZeroCones, numNonnegCones, numSocCones, numExpCones, numPowerCones,
        m_total, batchSize,
        d_soc_dims, d_soc_offsets, totalSocDim, d_soc_sz_offsets
    );

    // Step 2: Copy per-batch results to output arrays (no cross-batch reduction!)
    int copyThreads = 256;
    int copyBlocks = (batchSize + copyThreads - 1) / copyThreads;
    MOREAU_KERNEL_LAUNCH(copy_margin_results_kernel, copyBlocks, copyThreads, 0, stream,
        d_min_margin_out, d_pos_margin_out, d_batch_results, batchSize
    );
}

// ============================================================================
// Per-batch Scaled Unit Shift
// ============================================================================

__global__ void scaled_unit_shift_batched_impl(
    double* __restrict__ z,
    const double* __restrict__ alpha,  // Per-batch alpha values [batchSize]
    bool is_primal_cone,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m_total,
    int64_t batchSize,
    const int64_t* __restrict__ d_soc_offsets,
    int64_t totalSocDim,
    const int64_t* __restrict__ d_soc_sz_offsets
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double alpha_batch = alpha[batch];  // Read per-batch alpha
    int64_t offset = batch * m_total;

    // Zero cone: set to zero for primal
    if (is_primal_cone && numZeroCones > 0) {
        for (int64_t i = threadIdx.x; i < numZeroCones; i += blockDim.x) {
            z[offset + i] = 0.0;
        }
    }
    offset += numZeroCones;

    // Nonnegative cone: z += alpha
    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        z[offset + i] += alpha_batch;
    }
    offset += numNonnegCones;

    // SOC cones: add alpha to t component (use d_soc_sz_offsets for s/z access)
    for (int64_t i = threadIdx.x; i < numSocCones; i += blockDim.x) {
        z[offset + d_soc_sz_offsets[i]] += alpha_batch;
    }
    offset += totalSocDim;

    // Exp cones: add alpha to y and z components
    for (int64_t i = threadIdx.x; i < numExpCones; i += blockDim.x) {
        z[offset + i * 3 + 1] += alpha_batch;
        z[offset + i * 3 + 2] += alpha_batch;
    }
    offset += numExpCones * 3;

    // Power cones: add alpha to x and y components
    for (int64_t i = threadIdx.x; i < numPowerCones; i += blockDim.x) {
        z[offset + i * 3] += alpha_batch;
        z[offset + i * 3 + 1] += alpha_batch;
    }
}

void scaled_unit_shift_batched_kernel(
    double* z,
    const double* alpha,
    bool is_primal_cone,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m_total,
    int64_t batchSize,
    cudaStream_t stream,
    const int64_t* d_soc_offsets,
    int64_t totalSocDim,
    const int64_t* d_soc_sz_offsets
) {
    if (m_total == 0 || batchSize == 0) return;

    int64_t maxCones = max(max(max(numZeroCones, numNonnegCones),
                               max(numSocCones, numExpCones)),
                          numPowerCones);

    int threadsPerBlock = (maxCones < 256) ? ((maxCones + 31) / 32 * 32) : 256;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    MOREAU_KERNEL_LAUNCH(scaled_unit_shift_batched_impl, batchSize, threadsPerBlock, 0, stream,
        z, alpha, is_primal_cone,
        numZeroCones, numNonnegCones, numSocCones, numExpCones, numPowerCones,
        m_total, batchSize,
        d_soc_offsets, totalSocDim, d_soc_sz_offsets
    );
}

// ============================================================================
// Fused margins + alpha + shift (3→1 kernel per vector)
// ============================================================================
__global__ void fused_margins_and_shift_impl(
    double* __restrict__ z,
    bool is_primal_cone,
    double degree,
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
    const int64_t* __restrict__ d_soc_sz_offsets)
{
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    extern __shared__ double shared_mem[];
    double* s_min = shared_mem;
    double* s_pos = shared_mem + blockDim.x;

    // ---- Phase 1: compute margins (matches compute_margins_impl) ----
    double local_min = 1e300;
    double local_pos = 0.0;

    int64_t offset = batch * m_total;
    offset += numZeroCones;  // skip zero cones

    int64_t nonneg_start = offset;
    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        double val = z[nonneg_start + i];
        local_min = fmin(local_min, val);
        local_pos += fmax(val, 0.0);
    }
    offset += numNonnegCones;

    // SOC cones: variable dimension, use d_soc_sz_offsets for s/z access
    int64_t soc_start = offset;
    for (int64_t i = threadIdx.x; i < numSocCones; i += blockDim.x) {
        int64_t dim = d_soc_dims[i];
        int64_t sz_off = d_soc_sz_offsets[i];
        double t = z[soc_start + sz_off];
        double tail_sq = 0.0;
        for (int64_t j = 1; j < dim; j++) {
            tail_sq += z[soc_start + sz_off + j] * z[soc_start + sz_off + j];
        }
        double norm_x = sqrt(tail_sq);
        double margin = t - norm_x;
        local_min = fmin(local_min, margin);
        local_pos += fmax(margin, 0.0);
    }
    offset += totalSocDim;

    int64_t exp_start = offset;
    for (int64_t i = threadIdx.x; i < numExpCones; i += blockDim.x) {
        double v1 = z[exp_start + i * 3 + 1];
        double v2 = z[exp_start + i * 3 + 2];
        double margin = fmin(v1, v2);
        local_min = fmin(local_min, margin);
        local_pos += fmax(margin, 0.0);
    }
    offset += numExpCones * 3;

    int64_t power_start = offset;
    for (int64_t i = threadIdx.x; i < numPowerCones; i += blockDim.x) {
        double v0 = z[power_start + i * 3];
        double v1 = z[power_start + i * 3 + 1];
        double margin = fmin(v0, v1);
        local_min = fmin(local_min, margin);
        local_pos += fmax(margin, 0.0);
    }

    s_min[threadIdx.x] = local_min;
    s_pos[threadIdx.x] = local_pos;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < static_cast<unsigned>(stride)) {
            s_min[threadIdx.x] = fmin(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
            s_pos[threadIdx.x] += s_pos[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // ---- Phase 2: thread 0 computes alpha, broadcasts via shared mem ----
    if (threadIdx.x == 0) {
        double min_m = s_min[0];
        double pos_m = s_pos[0];
        double target = fmax(1.0, (pos_m * 0.1) / degree);
        double alpha;
        if (min_m <= 0.0) {
            alpha = -min_m + target;
        } else if (min_m < target) {
            alpha = target - min_m;
        } else {
            alpha = 0.0;
        }
        s_min[0] = alpha; // reuse shared mem to broadcast
    }
    __syncthreads();
    double alpha_batch = s_min[0];

    // ---- Phase 3: apply shift (matches scaled_unit_shift_batched_impl) ----
    offset = batch * m_total;

    // Zero cone: set to zero for primal
    if (is_primal_cone && numZeroCones > 0) {
        for (int64_t i = threadIdx.x; i < numZeroCones; i += blockDim.x) {
            z[offset + i] = 0.0;
        }
    }
    offset += numZeroCones;

    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        z[offset + i] += alpha_batch;
    }
    offset += numNonnegCones;

    // SOC cones: add alpha to t component via d_soc_sz_offsets
    for (int64_t i = threadIdx.x; i < numSocCones; i += blockDim.x) {
        z[offset + d_soc_sz_offsets[i]] += alpha_batch;
    }
    offset += totalSocDim;

    for (int64_t i = threadIdx.x; i < numExpCones; i += blockDim.x) {
        z[offset + i * 3 + 1] += alpha_batch;
        z[offset + i * 3 + 2] += alpha_batch;
    }
    offset += numExpCones * 3;

    for (int64_t i = threadIdx.x; i < numPowerCones; i += blockDim.x) {
        z[offset + i * 3] += alpha_batch;
        z[offset + i * 3 + 1] += alpha_batch;
    }
}

void fused_margins_and_shift_kernel(
    double* z,
    bool is_primal_cone,
    double degree,
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
    const int64_t* d_soc_sz_offsets)
{
    if (m_total == 0 || batchSize == 0) return;

    int threadsPerBlock = 256;
    size_t sharedMemSize = 2 * threadsPerBlock * sizeof(double);

    MOREAU_KERNEL_LAUNCH(fused_margins_and_shift_impl, batchSize, threadsPerBlock, sharedMemSize, stream,
        z, is_primal_cone, degree,
        numZeroCones, numNonnegCones, numSocCones, numExpCones, numPowerCones,
        m_total, batchSize,
        d_soc_dims, d_soc_offsets, totalSocDim, d_soc_sz_offsets);
}

} // namespace moreau
