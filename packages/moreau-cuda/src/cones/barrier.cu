/**
 * @file barrier.cu
 * @brief Barrier function implementation for cones
 */

#include "moreau/cones/cones.hpp"
#include "moreau/cones/common.cuh"
#include "moreau/profiling/profiler.hpp"
#include <cmath>
#include <limits>
#include <vector>

namespace moreau {

// Use canonical helpers from common.cuh (all live in moreau::cones).
using cones::logsafe;
using cones::wright_omega;
using cones::gradient_primal_exp;
using cones::gradient_primal_power;
using cones::newton_raphson_powcone;

// Exponential cone barrier functions
// Primal cone: s[2] >= s[1] * exp(s[0]/s[1]), s[1],s[2] > 0
// Dual cone: z[0] < 0, z[2] > 0, -z[0]*exp(z[1]/z[0]) >= z[2]

__device__ inline double exp_barrier_primal(const double* s) {
    // Primal barrier:
    // f(s) = ⟨s,g(s)⟩ - f*(-g(s))
    //      = -2*log(s[1]) - log(s[2]) - log((1-ω)^2/ω) - 3,
    // where ω = wright_omega(1 - s[0]/s[1] - log(s[1]/s[2]))
    // Note: barω = (ω-1)^2/ω

    if (s[1] <= 0.0 || s[2] <= 0.0) {
        return INFINITY;
    }

    double z_arg = 1.0 - s[0] / s[1] - logsafe(s[1] / s[2]);
    double omega = wright_omega(z_arg);

    if (!isfinite(omega) || omega <= 0.0) {
        return INFINITY;
    }

    // Transform omega to barω = (ω-1)^2/ω
    double bar_omega = (omega - 1.0) * (omega - 1.0) / omega;

    return -logsafe(bar_omega) - 2.0 * logsafe(s[1]) - logsafe(s[2]) - 3.0;
}

__device__ inline double exp_barrier_dual(const double* z) {
    // Dual barrier:
    // f*(z) = -log(z[1] - z[0] - z[0]*log(z[2]/-z[0])) - log(-z[0]) - log(z[2])

    if (z[0] >= 0.0 || z[2] <= 0.0) {
        return INFINITY;
    }

    double l = logsafe(-z[2] / z[0]);
    double arg = z[1] - z[0] - z[0] * l;

    if (arg <= 0.0) {
        return INFINITY;
    }

    return -logsafe(-z[2] * z[0]) - logsafe(arg);
}

// ============================================================================
// FUSED barrier kernel - combines all cone types into single launch
// One block per batch element; threads within a block handle cones.
// ============================================================================
__global__ void compute_barriers_all_cones_kernel(
    double* __restrict__ barrier_out,          // [batchSize] output
    const double* __restrict__ z,
    const double* __restrict__ s,
    const double* __restrict__ dz,
    const double* __restrict__ ds,
    double alpha,
    int64_t m_total,              // total constraint dimension per batch
    // Cone structure offsets (in constraint space)
    int64_t offset_zero,
    int64_t offset_nonneg,
    int64_t offset_soc,
    int64_t offset_exp,
    int64_t offset_power,
    // Cone counts
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    // Power cone alphas
    const double* __restrict__ d_powerAlphas,
    // Variable-dim SOC params
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_offsets,
    const int64_t* __restrict__ d_soc_sz_offsets,
    // GenPowerCone params
    int64_t numGenPowerCones,
    int64_t offset_genpow,
    const double* __restrict__ d_genPowerAlphas,
    const int64_t* __restrict__ d_genPowerDim1s,
    const int64_t* __restrict__ d_genPowerDim2s,
    const int64_t* __restrict__ d_genPowerOffsets,
    const int64_t* __restrict__ d_genPowerAlphaOffsets,
    const int64_t* __restrict__ d_genPowerSzOffsets
) {
    int64_t batch = blockIdx.x;
    int64_t batch_off = batch * m_total;
    int64_t stride = blockDim.x;

    // ========== NONNEGATIVE CONES ==========
    // barrier = -sum(log(s_i * z_i))
    for (int64_t i = threadIdx.x; i < numNonnegCones; i += stride) {
        int64_t idx = batch_off + offset_nonneg + i;
        double si = s[idx] + alpha * ds[idx];
        double zi = z[idx] + alpha * dz[idx];
        double val = -logsafe(si * zi);
        atomicAdd(&barrier_out[batch], val);
    }
    __syncthreads();

    // ========== SOC CONES (variable dim) ==========
    // barrier = -0.5 * log(residual_s * residual_z)
    // where residual = s[0]^2 - ||s[1:]||^2
    for (int64_t cone_idx = threadIdx.x; cone_idx < numSocCones; cone_idx += stride) {
        int64_t dim = d_soc_dims[cone_idx];
        int64_t sz_off = d_soc_sz_offsets[cone_idx];
        int64_t base = batch_off + offset_soc + sz_off;

        // Compute residuals: x[0]^2 - ||x[1:]||^2 for s+alpha*ds and z+alpha*dz
        // Use (x0 - ||x1:||) * (x0 + ||x1:||) to avoid catastrophic cancellation
        double cur_s0 = s[base] + alpha * ds[base];
        double cur_z0 = z[base] + alpha * dz[base];
        double s_tail_sq = 0.0, z_tail_sq = 0.0;
        for (int64_t i = 1; i < dim; i++) {
            double si = s[base + i] + alpha * ds[base + i];
            double zi = z[base + i] + alpha * dz[base + i];
            s_tail_sq += si * si;
            z_tail_sq += zi * zi;
        }
        double s_tail_norm = sqrt(s_tail_sq);
        double z_tail_norm = sqrt(z_tail_sq);
        double res_s = (cur_s0 - s_tail_norm) * (cur_s0 + s_tail_norm);
        double res_z = (cur_z0 - z_tail_norm) * (cur_z0 + z_tail_norm);

        double barrier_val;
        if (res_s > 0.0 && res_z > 0.0) {
            barrier_val = -0.5 * logsafe(res_s * res_z);
        } else {
            barrier_val = INFINITY;
        }
        atomicAdd(&barrier_out[batch], barrier_val);
    }
    __syncthreads();

    // ========== EXPONENTIAL CONES ==========
    for (int64_t cone_idx = threadIdx.x; cone_idx < numExpCones; cone_idx += stride) {
        int64_t base = batch_off + offset_exp + cone_idx * 3;

        double cur_s[3], cur_z[3];
        for (int i = 0; i < 3; i++) {
            cur_s[i] = s[base + i] + alpha * ds[base + i];
            cur_z[i] = z[base + i] + alpha * dz[base + i];
        }

        double barrier_val = 0.0;
        barrier_val += exp_barrier_primal(cur_s);
        barrier_val += exp_barrier_dual(cur_z);

        atomicAdd(&barrier_out[batch], barrier_val);
    }
    __syncthreads();

    // ========== POWER CONES ==========
    for (int64_t cone_idx = threadIdx.x; cone_idx < numPowerCones; cone_idx += stride) {
        int64_t base = batch_off + offset_power + cone_idx * 3;
        double cone_alpha = d_powerAlphas[cone_idx];

        double cur_s[3], cur_z[3];
        for (int i = 0; i < 3; i++) {
            cur_s[i] = s[base + i] + alpha * ds[base + i];
            cur_z[i] = z[base + i] + alpha * dz[base + i];
        }

        double barrier_val = 0.0;

        // Basic primal barrier approximation
        if (cur_s[0] > 0.0 && cur_s[1] > 0.0) {
            barrier_val -= cone_alpha * logsafe(cur_s[0]);
            barrier_val -= (1.0 - cone_alpha) * logsafe(cur_s[1]);
        } else {
            barrier_val = INFINITY;
        }

        // Basic dual barrier approximation
        if (cur_z[0] > 0.0 && cur_z[1] > 0.0) {
            barrier_val -= cone_alpha * logsafe(cur_z[0]);
            barrier_val -= (1.0 - cone_alpha) * logsafe(cur_z[1]);
        } else {
            barrier_val = INFINITY;
        }

        atomicAdd(&barrier_out[batch], barrier_val);
    }

    // ========== GENPOWERCONE ==========
    // GenPowerCone: K = {(p,w) : prod(p_i^alpha_i) >= ||w||, p_i >= 0}
    // Dual cone: K* = {(z,u) : prod((z_i/alpha_i)^alpha_i) >= ||u||, z_i >= 0}
    //
    // Dual barrier: f*(z) = -log(zeta) - sum((1-alpha_i)*log(z_i))
    //   where zeta = prod((z_i/alpha_i)^{2*alpha_i}) - ||u||^2
    //
    // Primal barrier: f(s) = -f*(-grad_primal(s)) - degree
    //   where grad_primal requires Newton-Raphson to find g1
    //   degree = dim1 + 1
    for (int64_t cone_idx = threadIdx.x; cone_idx < numGenPowerCones; cone_idx += stride) {
        int64_t dim1 = d_genPowerDim1s[cone_idx];
        int64_t dim2 = d_genPowerDim2s[cone_idx];
        int64_t sz_off = d_genPowerSzOffsets[cone_idx];
        int64_t alpha_off = d_genPowerAlphaOffsets[cone_idx];
        int64_t base = batch_off + offset_genpow + sz_off;

        double barrier_val = 0.0;

        // --- Dual barrier: f*(z) = -log(zeta) - sum((1-ai)*log(zi)) ---
        {
            bool feasible = true;
            double log_phi = 0.0;
            for (int64_t i = 0; i < dim1; ++i) {
                double zi = z[base + i] + alpha * dz[base + i];
                double ai = d_genPowerAlphas[alpha_off + i];
                if (zi <= 0.0) { feasible = false; break; }
                log_phi += 2.0 * ai * logsafe(zi / ai);
            }
            if (feasible) {
                double norm2w = 0.0;
                for (int64_t i = dim1; i < dim1 + dim2; ++i) {
                    double zi = z[base + i] + alpha * dz[base + i];
                    norm2w += zi * zi;
                }
                double phi = exp(log_phi);
                double zeta = phi - norm2w;
                if (zeta > 0.0) {
                    barrier_val -= logsafe(zeta);
                    for (int64_t i = 0; i < dim1; ++i) {
                        double zi = z[base + i] + alpha * dz[base + i];
                        double ai = d_genPowerAlphas[alpha_off + i];
                        barrier_val -= (1.0 - ai) * logsafe(zi);
                    }
                } else {
                    barrier_val = INFINITY;
                }
            } else {
                barrier_val = INFINITY;
            }
        }

        // --- Primal barrier: f(s) = -f*(-g(s)) - degree ---
        // where g(s) = gradient_primal(s) requires Newton-Raphson
        if (isfinite(barrier_val)) {
            bool feasible = true;

            // Compute phi = prod(si^{2*ai}) and check feasibility
            double log_phi_s = 0.0;
            for (int64_t i = 0; i < dim1; ++i) {
                double si = s[base + i] + alpha * ds[base + i];
                if (si <= 0.0) { feasible = false; break; }
                double ai = d_genPowerAlphas[alpha_off + i];
                log_phi_s += 2.0 * ai * logsafe(si);
            }

            if (feasible) {
                double phi_s = exp(log_phi_s);

                // Compute norm_r = ||w||
                double norm2w_s = 0.0;
                for (int64_t i = dim1; i < dim1 + dim2; ++i) {
                    double si = s[base + i] + alpha * ds[base + i];
                    norm2w_s += si * si;
                }
                double norm_r = sqrt(norm2w_s);

                // Compute psi = 1 / sum(ai^2)
                double sum_ai_sq = 0.0;
                for (int64_t i = 0; i < dim1; ++i) {
                    double ai = d_genPowerAlphas[alpha_off + i];
                    sum_ai_sq += ai * ai;
                }
                double psi = 1.0 / sum_ai_sq;

                if (norm_r > 1e-15) {
                    // Newton-Raphson to find g1
                    // f(x) = -log(2x/nr + x^2) + sum(2*ai*(log(x*nr + (1+ai)/ai) - log(pi)))
                    double nr = norm_r;
                    double denom = phi_s - nr * nr;
                    double x;
                    if (denom > 1e-30) {
                        x = -1.0 / nr
                            + (psi * nr + sqrt((phi_s / (nr * nr) + psi * psi - 1.0) * phi_s))
                              / denom;
                    } else {
                        x = 1.0 / nr;  // fallback
                    }
                    if (x <= 0.0) x = 1e-10;

                    // Newton-Raphson (one-sided): matches CPU newton_raphson_onesided
                    for (int iter = 0; iter < 100; ++iter) {
                        // f(x)
                        double f0 = -logsafe(2.0 * x / nr + x * x);
                        for (int64_t i = 0; i < dim1; ++i) {
                            double ai = d_genPowerAlphas[alpha_off + i];
                            double si = s[base + i] + alpha * ds[base + i];
                            f0 += 2.0 * ai * (logsafe(x * nr + (1.0 + ai) / ai) - logsafe(si));
                        }
                        // f'(x)
                        double dfdx = -(2.0 * x + 2.0 / nr) / (x * x + 2.0 * x / nr);
                        for (int64_t i = 0; i < dim1; ++i) {
                            double ai = d_genPowerAlphas[alpha_off + i];
                            dfdx += 2.0 * ai * nr / (nr * x + (1.0 + ai) / ai);
                        }
                        double dx = -f0 / dfdx;
                        constexpr double eps = 2.220446049250313e-16;  // f64 epsilon
                        constexpr double sqrt_eps = 1.4901161193847656e-8;
                        if (dx < eps || fabs(dx / x) < sqrt_eps || fabs(dfdx) < eps) break;
                        x += dx;
                    }

                    double g1 = x;

                    // Compute g = -gradient_primal(s), i.e. the negated gradient
                    // g_p[i] = (1 + ai + ai*g1*norm_r) / si  (negated from gradient)
                    // g_w[i] = -(g1 / norm_r) * w_i           (negated from gradient)
                    // Then evaluate barrier_dual(g) where g = -gradient_primal

                    // For barrier_dual(g), need:
                    //   log_phi_g = sum(2*ai*log(g_p[i]/ai))
                    //   norm2_gw  = sum(g_w[i]^2) = g1^2 * norm2w_s / norm_r^2 = g1^2
                    //   zeta_g    = exp(log_phi_g) - norm2_gw

                    double log_phi_g = 0.0;
                    for (int64_t i = 0; i < dim1; ++i) {
                        double ai = d_genPowerAlphas[alpha_off + i];
                        double si = s[base + i] + alpha * ds[base + i];
                        double gpi = (1.0 + ai + ai * g1 * nr) / si;
                        log_phi_g += 2.0 * ai * logsafe(gpi / ai);
                    }
                    double phi_g = exp(log_phi_g);
                    double norm2_gw = g1 * g1;  // ||g_w||^2 = (g1/nr)^2 * nr^2 = g1^2
                    double zeta_g = phi_g - norm2_gw;

                    if (zeta_g > 0.0) {
                        // f*(g) = -log(zeta_g) - sum((1-ai)*log(gpi))
                        double dual_bar = -logsafe(zeta_g);
                        for (int64_t i = 0; i < dim1; ++i) {
                            double ai = d_genPowerAlphas[alpha_off + i];
                            double si = s[base + i] + alpha * ds[base + i];
                            double gpi = (1.0 + ai + ai * g1 * nr) / si;
                            dual_bar -= (1.0 - ai) * logsafe(gpi);
                        }
                        // Primal barrier = -dual_bar - degree
                        double degree = (double)(dim1 + 1);
                        barrier_val += -dual_bar - degree;
                    } else {
                        barrier_val = INFINITY;
                    }
                } else {
                    // norm_r ~ 0: gradient simplifies (no Newton needed)
                    // g_p[i] = (1+ai)/si, g_w = 0
                    double log_phi_g = 0.0;
                    for (int64_t i = 0; i < dim1; ++i) {
                        double ai = d_genPowerAlphas[alpha_off + i];
                        double si = s[base + i] + alpha * ds[base + i];
                        double gpi = (1.0 + ai) / si;
                        log_phi_g += 2.0 * ai * logsafe(gpi / ai);
                    }
                    double phi_g = exp(log_phi_g);
                    // norm2_gw = 0
                    if (phi_g > 0.0) {
                        double dual_bar = -logsafe(phi_g);
                        for (int64_t i = 0; i < dim1; ++i) {
                            double ai = d_genPowerAlphas[alpha_off + i];
                            double si = s[base + i] + alpha * ds[base + i];
                            double gpi = (1.0 + ai) / si;
                            dual_bar -= (1.0 - ai) * logsafe(gpi);
                        }
                        double degree = (double)(dim1 + 1);
                        barrier_val += -dual_bar - degree;
                    } else {
                        barrier_val = INFINITY;
                    }
                }
            } else {
                barrier_val = INFINITY;
            }
        }

        atomicAdd(&barrier_out[batch], barrier_val);
    }
}

double Cones::computeBarrier(
    const BatchedVector& z,
    const BatchedVector& s,
    const BatchedVector& dz,
    const BatchedVector& ds,
    double alpha,
    cudaStream_t stream
) {
    // Check if any cones need barrier computation
    if (numNonnegCones == 0 && numSocCones == 0 && numExpCones == 0 && numPowerCones == 0 && numGenPowerCones == 0) {
        return 0.0;
    }

    int64_t bs = z.batchSize();
    int64_t m_total = z.n();

    // d_barrier_work is pre-allocated in Cones::initialize() with batchSize elements
    cudaMemsetAsync(d_barrier_work, 0, sizeof(double) * bs, stream);

    // Compute cone offsets in constraint space
    int64_t offset_zero = 0;
    int64_t offset_nonneg = offset_zero + numZeroCones;
    int64_t offset_soc = offset_nonneg + numNonnegCones;
    int64_t offset_exp = offset_soc + totalSocDim;
    int64_t offset_power = offset_exp + numExpCones * 3;
    int64_t offset_genpow = offset_power + numPowerCones * 3;

    // Determine thread count based on max cones
    int64_t maxCones = numNonnegCones;
    maxCones = (numSocCones > maxCones) ? numSocCones : maxCones;
    maxCones = (numExpCones > maxCones) ? numExpCones : maxCones;
    maxCones = (numPowerCones > maxCones) ? numPowerCones : maxCones;
    maxCones = (numGenPowerCones > maxCones) ? numGenPowerCones : maxCones;

    int threadsPerBlock = (maxCones < 256) ? ((maxCones + 31) / 32 * 32) : 256;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    // One block per batch element
    MOREAU_KERNEL_LAUNCH(compute_barriers_all_cones_kernel, bs, threadsPerBlock, 0, stream,
        d_barrier_work,
        z.data(),
        s.data(),
        dz.data(),
        ds.data(),
        alpha,
        m_total,
        offset_zero,
        offset_nonneg,
        offset_soc,
        offset_exp,
        offset_power,
        numNonnegCones,
        numSocCones,
        numExpCones,
        numPowerCones,
        d_powerAlphas,
        d_soc_dims,
        d_soc_offsets,
        d_soc_sz_offsets,
        numGenPowerCones,
        offset_genpow,
        d_genPowerAlphas,
        d_genPowerDim1s,
        d_genPowerDim2s,
        d_genPowerOffsets,
        d_genPowerAlphaOffsets,
        d_genPowerSzOffsets
    );

    // Copy per-batch results to pinned host memory and sync
    cudaMemcpyAsync(h_barrier_pinned, d_barrier_work, sizeof(double) * bs, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    double total = 0.0;
    for (int64_t i = 0; i < bs; i++) {
        total += h_barrier_pinned[i];
    }
    return total;
}

} // namespace moreau
