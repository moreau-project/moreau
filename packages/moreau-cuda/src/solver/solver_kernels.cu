/**
 * @file kernels.cu
 * @brief CUDA kernel implementations for solver operations
 */

#include "moreau/solver/solver_kernels.cuh"
#include "moreau/profiling/profiler.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>
#include <cstdio>

namespace moreau {

// Machine epsilon for double precision
constexpr double DOUBLE_EPSILON = DBL_EPSILON;  // ~2.22e-16

// Maximum threads per block for norm reduction kernels
// This must match the launch configuration (256 threads)
constexpr int NORM_REDUCTION_THREADS = 256;

/**
 * @brief Compute scaled L2 norm: ||v .* scale||_2
 *
 * Uses warp-level shuffle reduction for small blocks (blockDim.x == 32)
 * or shared memory reduction for larger blocks.
 */
__device__ double norm_scaled_device(
    const double* v,
    const double* scale,
    int64_t n,
    int64_t batch_offset
) {
    double local_sum = 0.0;
    int tid = threadIdx.x;

    // Accumulate squared scaled values
    for (int64_t i = tid; i < n; i += blockDim.x) {
        double val = v[batch_offset + i] * scale[batch_offset + i];
        local_sum += val * val;
    }

    if (blockDim.x == 32) {
        // Warp-level reduction via shuffle (no shared memory needed)
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        return (tid == 0) ? sqrt(local_sum) : 0.0;
    } else {
        __shared__ double shared_sum[NORM_REDUCTION_THREADS];
        shared_sum[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_sum[tid] += shared_sum[tid + stride];
            }
            __syncthreads();
        }
        return (tid == 0) ? sqrt(shared_sum[0]) : 0.0;
    }
}

/**
 * @brief Compute scaled infinity norm: max(|v .* scale|)
 *
 * Uses warp-level shuffle reduction for small blocks (blockDim.x == 32)
 * or shared memory reduction for larger blocks.
 */
__device__ double norm_inf_scaled_device(
    const double* v,
    const double* scale,
    int64_t n,
    int64_t batch_offset
) {
    double local_max = 0.0;
    int tid = threadIdx.x;

    // Find max of scaled absolute values
    for (int64_t i = tid; i < n; i += blockDim.x) {
        double val = fabs(v[batch_offset + i] * scale[batch_offset + i]);
        local_max = fmax(local_max, val);
    }

    if (blockDim.x == 32) {
        // Warp-level reduction via shuffle (no shared memory needed)
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmax(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
        }
        return (tid == 0) ? local_max : 0.0;
    } else {
        __shared__ double shared_max[NORM_REDUCTION_THREADS];
        shared_max[tid] = local_max;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_max[tid] = fmax(shared_max[tid], shared_max[tid + stride]);
            }
            __syncthreads();
        }
        return (tid == 0) ? shared_max[0] : 0.0;
    }
}

/**
 * @brief Compute ‖z_x .* d[J]‖_2 where J is gathered through d_xcone_indices.
 *
 * Mirror of `norm_scaled_device` for the direct-x dual: each entry of `z_x`
 * pairs with `x[idx]`, so the natural unscaling factor is `d[idx]`, not `e`.
 * Returns 0 when totalXConeNumel == 0; result is meaningful only on tid 0.
 */
__device__ double norm_zx_scaled_d_device(
    const double* z_x,                  // [batch * totalXConeNumel + k]
    const double* d,                    // [batch * n + idx]
    const int64_t* xcone_indices,       // length totalXConeNumel
    int64_t totalXConeNumel,
    int64_t batch_offset_zx,
    int64_t batch_offset_n
) {
    double local_sum = 0.0;
    int tid = threadIdx.x;

    for (int64_t k = tid; k < totalXConeNumel; k += blockDim.x) {
        int64_t idx = xcone_indices[k];
        double val = z_x[batch_offset_zx + k] * d[batch_offset_n + idx];
        local_sum += val * val;
    }

    if (blockDim.x == 32) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        return (tid == 0) ? sqrt(local_sum) : 0.0;
    } else {
        __shared__ double shared_sum_zx[NORM_REDUCTION_THREADS];
        shared_sum_zx[tid] = local_sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_sum_zx[tid] += shared_sum_zx[tid + stride];
            }
            __syncthreads();
        }
        return (tid == 0) ? sqrt(shared_sum_zx[0]) : 0.0;
    }
}

/**
 * @brief Update solver information metrics kernel
 *
 * One block per batch, computes all convergence metrics
 */
__global__ void update_info_kernel_impl(
    double* __restrict__ cost_primal,
    double* __restrict__ cost_dual,
    double* __restrict__ res_primal,
    double* __restrict__ res_dual,
    double* __restrict__ res_primal_inf,
    double* __restrict__ res_dual_inf,
    double* __restrict__ gap_abs,
    double* __restrict__ gap_rel,
    double* __restrict__ ktratio,
    const double* __restrict__ rx,
    const double* __restrict__ rz,
    const double* __restrict__ rtau,
    const double* __restrict__ dot_qx,
    const double* __restrict__ dot_bz,
    const double* __restrict__ tau,
    const double* __restrict__ kappa,
    const double* __restrict__ dot_xPx,
    const double* __restrict__ rx_inf,
    const double* __restrict__ rz_inf,
    const double* __restrict__ Px,
    const double* __restrict__ x,
    const double* __restrict__ z,
    const double* __restrict__ s,
    const double* __restrict__ z_x,
    const int64_t* __restrict__ d_xcone_indices,
    int64_t totalXConeNumel,
    const double* __restrict__ normb_cached,
    const double* __restrict__ normq_cached,
    const double* __restrict__ d,
    const double* __restrict__ dinv,
    const double* __restrict__ e,
    const double* __restrict__ einv,
    const double* __restrict__ c,
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    int tid = threadIdx.x;

    // Get batch-specific values
    double tau_val = tau[batch];
    double tau_inv = 1.0 / tau_val;
    double kappa_val = kappa[batch];
    double c_val = c[batch];
    double c_inv = 1.0 / c_val;

    int64_t batch_offset_n = batch * n;
    int64_t batch_offset_m = batch * m;

    // Compute τ-scaled costs
    double xPx_tauinvsq_over2 = dot_xPx[batch] * tau_inv * tau_inv / 2.0;
    double cost_p = (dot_qx[batch] * tau_inv + xPx_tauinvsq_over2) * c_inv;
    double cost_d = (-dot_bz[batch] * tau_inv - xPx_tauinvsq_over2) * c_inv;

    // Compute scaled norms using parallel reduction
    double normx = norm_scaled_device(x, d, n, batch_offset_n);
    double normz = norm_scaled_device(z, e, m, batch_offset_m) * c_inv;
    double norms = norm_scaled_device(s, einv, m, batch_offset_m);

    // Use cached norms for b and q (computed once at setup)
    double normb = normb_cached[batch];
    double normq = normq_cached[batch];

    // Infeasibility residuals
    double rx_inf_norm = norm_scaled_device(rx_inf, dinv, n, batch_offset_n) * c_inv;
    double rz_inf_norm = norm_scaled_device(rz_inf, einv, m, batch_offset_m);
    double Px_norm = norm_scaled_device(Px, dinv, n, batch_offset_n);

    // Direct-x dual `z_x` contributes to the primal-infeasibility certificate
    // `‖A^T z − Σ_J E_J^T z_x‖ → 0` and pairs with `x[J]` (so it unscales by
    // `d[J]`, not `e`). Without this term in the relative-residual denominator,
    // a certificate with small `‖z‖` but large `‖z_x‖` can fail the relative
    // test even when the absolute residual is well below tolerance.
    double norm_zx_d_cinv = 0.0;
    if (totalXConeNumel > 0 && z_x != nullptr && d_xcone_indices != nullptr) {
        int64_t batch_offset_zx = batch * totalXConeNumel;
        norm_zx_d_cinv = norm_zx_scaled_d_device(
            z_x, d, d_xcone_indices, totalXConeNumel,
            batch_offset_zx, batch_offset_n) * c_inv;
    }

    double res_p_inf = rx_inf_norm / fmax(1.0, normz + norm_zx_d_cinv);
    double res_d_inf = fmax(
        Px_norm / fmax(1.0, normx),
        rz_inf_norm / fmax(1.0, normx + norms)
    );

    // Unscale by τ (but NOT normb and normq which are cached equilibrated values)
    normx *= tau_inv;
    normz *= tau_inv;
    norms *= tau_inv;

    // Relative residuals
    double rz_norm = norm_scaled_device(rz, einv, m, batch_offset_m) * tau_inv;
    double rx_norm = norm_scaled_device(rx, dinv, n, batch_offset_n) * tau_inv * c_inv;

    double res_p = rz_norm / fmax(1.0, normb + normx + norms);
    double res_d = rx_norm / fmax(1.0, normq + normx + normz);

    // Gaps
    double gap_a = fabs(cost_p - cost_d);
    double gap_r = gap_a / fmax(1.0, fmin(fabs(cost_p), fabs(cost_d)));

    // κ/τ ratio
    double kt_ratio = kappa_val * tau_inv;

    // Thread 0 writes results
    if (tid == 0) {
        cost_primal[batch] = cost_p;
        cost_dual[batch] = cost_d;
        res_primal[batch] = res_p;
        res_dual[batch] = res_d;
        res_primal_inf[batch] = res_p_inf;
        res_dual_inf[batch] = res_d_inf;
        gap_abs[batch] = gap_a;
        gap_rel[batch] = gap_r;
        ktratio[batch] = kt_ratio;
    }
}

void update_info_kernel(
    double* cost_primal,
    double* cost_dual,
    double* res_primal,
    double* res_dual,
    double* res_primal_inf,
    double* res_dual_inf,
    double* gap_abs,
    double* gap_rel,
    double* ktratio,
    const double* rx,
    const double* rz,
    const double* rtau,
    const double* dot_qx,
    const double* dot_bz,
    const double* tau,
    const double* kappa,
    const double* dot_xPx,
    const double* rx_inf,
    const double* rz_inf,
    const double* Px,
    const double* x,
    const double* z,
    const double* s,
    const double* z_x,
    const int64_t* d_xcone_indices,
    int64_t totalXConeNumel,
    const double* normb_cached,
    const double* normq_cached,
    const double* d,
    const double* dinv,
    const double* e,
    const double* einv,
    const double* c,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    // Use 32 threads (warp shuffle reduction) for small problems,
    // 256 threads (shared memory reduction) for larger ones.
    // A single warp can stride over max(n, m, totalXConeNumel) elements efficiently.
    int64_t max_dim = (n > m) ? n : m;
    if (totalXConeNumel > max_dim) max_dim = totalXConeNumel;
    int threadsPerBlock = (max_dim <= 512) ? 32 : 256;
    int numBlocks = batchSize;

    MOREAU_KERNEL_LAUNCH(update_info_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        cost_primal, cost_dual, res_primal, res_dual,
        res_primal_inf, res_dual_inf, gap_abs, gap_rel, ktratio,
        rx, rz, rtau, dot_qx, dot_bz, tau, kappa, dot_xPx,
        rx_inf, rz_inf, Px, x, z, s,
        z_x, d_xcone_indices, totalXConeNumel,
        normb_cached, normq_cached,
        d, dinv, e, einv, c,
        n, m, batchSize);
}

/**
 * @brief Kernel to compute centering parameter: sigma = (1 - α)³
 */
__global__ void calc_centering_parameter_impl(
    double* __restrict__ sigma,
    const double* __restrict__ alpha,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batchSize) return;

    double one_minus_alpha = 1.0 - alpha[batch];
    sigma[batch] = one_minus_alpha * one_minus_alpha * one_minus_alpha;
}

void calc_centering_parameter_kernel(
    double* sigma,
    const double* alpha,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(calc_centering_parameter_impl, numBlocks, threadsPerBlock, 0, stream,
        sigma, alpha, batchSize);
}

__global__ void calc_mehrotra_correction_impl(
    double* __restrict__ m,
    const double* __restrict__ alpha,
    int64_t iter,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batchSize) return;

    // m = (iter > 1) ? 1.0 : alpha
    m[batch] = (iter > 1) ? 1.0 : alpha[batch];
}

void calc_mehrotra_correction_kernel(
    double* m,
    const double* alpha,
    int64_t iter,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(calc_mehrotra_correction_impl, numBlocks, threadsPerBlock, 0, stream,
        m, alpha, iter, batchSize);
}

// ============================================================================
// FUSED solver parameters kernel - computes sigma and m in one launch (2→1)
// ============================================================================
__global__ void calc_solver_parameters_impl(
    double* __restrict__ sigma,
    double* __restrict__ m_out,
    const double* __restrict__ alpha,
    int64_t iter,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch >= batchSize) return;

    double alpha_val = alpha[batch];

    // Centering parameter: sigma = (1 - α)³
    double one_minus_alpha = 1.0 - alpha_val;
    sigma[batch] = one_minus_alpha * one_minus_alpha * one_minus_alpha;

    // Mehrotra correction: m = (iter > 1) ? 1.0 : alpha
    m_out[batch] = (iter > 1) ? 1.0 : alpha_val;
}

void calc_solver_parameters_kernel(
    double* sigma,
    double* m_out,
    const double* alpha,
    int64_t iter,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(calc_solver_parameters_impl, numBlocks, threadsPerBlock, 0, stream,
        sigma, m_out, alpha, iter, batchSize);
}

__global__ void compute_cached_norms_kernel_impl(
    double* __restrict__ normb,
    double* __restrict__ normq,
    const double* __restrict__ b,
    const double* __restrict__ q,
    const double* __restrict__ einv,
    const double* __restrict__ dinv,
    const double* __restrict__ c,
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    int tid = threadIdx.x;
    double cinv = 1.0 / c[batch];
    int64_t batch_offset_n = batch * n;
    int64_t batch_offset_m = batch * m;

    // Compute ||b||_inf with equilibration scaling (using parallel reduction)
    double nb = norm_inf_scaled_device(b, einv, m, batch_offset_m);
    __syncthreads();  // Barrier before reusing shared memory in norm_inf_scaled_device

    // Compute ||q||_inf with equilibration scaling (using parallel reduction)
    double nq = norm_inf_scaled_device(q, dinv, n, batch_offset_n) * cinv;

    // Thread 0 writes the results
    if (tid == 0) {
        normb[batch] = nb;
        normq[batch] = nq;
    }
}

void compute_cached_norms_kernel(
    double* normb,
    double* normq,
    const double* b,
    const double* q,
    const double* einv,
    const double* dinv,
    const double* c,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    // Use 32 threads (warp shuffle reduction) for small problems
    int64_t max_dim = (n > m) ? n : m;
    int threadsPerBlock = (max_dim <= 512) ? 32 : 256;
    int numBlocks = batchSize;

    MOREAU_KERNEL_LAUNCH(compute_cached_norms_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        normb, normq, b, q, einv, dinv, c, n, m, batchSize);
}

/**
 * @brief Kernel to project vectors onto nonnegative orthant: v = max(v, eps)
 */
__global__ void project_nonneg_impl(
    double* __restrict__ v,
    double eps,
    int64_t n,
    int64_t batchSize
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_elements = n * batchSize;

    if (idx >= total_elements) return;

    v[idx] = fmax(v[idx], eps);
}

void project_nonneg_kernel(
    double* v,
    double eps,
    int64_t n,
    int64_t batchSize,
    cudaStream_t stream
) {
    int64_t total_elements = n * batchSize;
    int threadsPerBlock = 256;
    int numBlocks = (total_elements + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(project_nonneg_impl, numBlocks, threadsPerBlock, 0, stream,
        v, eps, n, batchSize);
}

// Device function to check convergence for a single batch
__device__ int32_t check_convergence_single_batch(
    double gap_abs,
    double gap_rel,
    double res_primal,
    double res_dual,
    double ktratio,
    double res_primal_inf,
    double res_dual_inf,
    double dot_qx,
    double dot_bz,
    double tolGapAbs,
    double tolGapRel,
    double tolFeas,
    double tolInfeasAbs,
    double tolInfeasRel,
    double tolKtRatio,
    double reducedTolGapAbs,
    double reducedTolGapRel,
    double reducedTolFeas,
    double reducedTolInfeasAbs,
    double reducedTolInfeasRel,
    double reducedTolKtRatio
) {
    // Check for optimality (primary tolerances)
    bool gap_converged = (gap_abs < tolGapAbs) || (gap_rel < tolGapRel);
    bool feasible = (res_primal < tolFeas) && (res_dual < tolFeas);
    bool interior = (ktratio <= 1.0);

    if (gap_converged && feasible && interior) {
        return 1; // Solved
    }

    // Check for primal infeasibility (primary tolerances)
    double ktratio_threshold = 1000.0 / tolKtRatio;
    if (ktratio > ktratio_threshold && dot_bz < -tolInfeasAbs &&
        res_primal_inf < -tolInfeasRel * dot_bz) {
        return 2; // PrimalInfeasible
    }

    // Check for dual infeasibility (primary tolerances)
    if (ktratio > ktratio_threshold && dot_qx < -tolInfeasAbs &&
        res_dual_inf < -tolInfeasRel * dot_qx) {
        return 3; // DualInfeasible
    }

    // Note: Reduced tolerance checks (AlmostSolved, etc.) are only done in post_process,
    // not during the main solve loop, to match Clarabel.rs behavior.
    return 0; // Unsolved
}

__global__ void check_termination_impl(
    int32_t* __restrict__ status,
    int32_t* __restrict__ any_done,
    int32_t* __restrict__ iterations_per_batch,
    const double* __restrict__ gap_abs,
    const double* __restrict__ gap_rel,
    const double* __restrict__ res_primal,
    const double* __restrict__ res_dual,
    const double* __restrict__ ktratio,
    const double* __restrict__ res_primal_inf,
    const double* __restrict__ res_dual_inf,
    const double* __restrict__ dot_qx,
    const double* __restrict__ dot_bz,
    const double* __restrict__ prev_res_primal,
    const double* __restrict__ prev_res_dual,
    const double* __restrict__ prev_gap_abs,
    const double* __restrict__ prev_gap_rel,
    double tolGapAbs,
    double tolGapRel,
    double tolFeas,
    double tolInfeasAbs,
    double tolInfeasRel,
    double tolKtRatio,
    double reducedTolGapAbs,
    double reducedTolGapRel,
    double reducedTolFeas,
    double reducedTolInfeasAbs,
    double reducedTolInfeasRel,
    double reducedTolKtRatio,
    int64_t iter,
    int64_t maxIter,
    double solve_time_seconds,
    double timeLimit,
    int64_t batchSize
) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    // Check time limit first - applies to all batches
    if (solve_time_seconds > timeLimit) {
        if (status[b] == 0) { // Only update if Unsolved
            status[b] = 8; // MaxTime
            iterations_per_batch[b] = static_cast<int32_t>(iter);
        }
        atomicAdd(any_done, 1);  // Count how many batches are done
        return;
    }

    // Skip if already terminated
    if (status[b] != 0) {
        atomicAdd(any_done, 1);  // Count how many batches are done
        return;
    }

    // Check for NaN in key metrics - indicates numerical breakdown
    if (isnan(res_primal[b]) || isnan(res_dual[b]) ||
        isnan(gap_abs[b]) || isnan(gap_rel[b])) {
        status[b] = 9; // NumericalError
        iterations_per_batch[b] = static_cast<int32_t>(iter);
        atomicAdd(any_done, 1);
        return;
    }

    // 1. Check convergence
    int32_t conv_status = check_convergence_single_batch(
        gap_abs[b], gap_rel[b], res_primal[b], res_dual[b],
        ktratio[b], res_primal_inf[b], res_dual_inf[b],
        dot_qx[b], dot_bz[b],
        tolGapAbs, tolGapRel, tolFeas, tolInfeasAbs, tolInfeasRel, tolKtRatio,
        reducedTolGapAbs, reducedTolGapRel, reducedTolFeas,
        reducedTolInfeasAbs, reducedTolInfeasRel, reducedTolKtRatio
    );

    if (conv_status != 0) {
        status[b] = conv_status;
        iterations_per_batch[b] = static_cast<int32_t>(iter);
        atomicAdd(any_done, 1);  // Count how many batches are done
        return;
    }

    // 2. Check for insufficient progress (only if iter > 1)
    if (iter > 1) {
        bool residuals_worse = (res_dual[b] > prev_res_dual[b]) ||
                               (res_primal[b] > prev_res_primal[b]);

        if (residuals_worse) {
            // Poor progress at high tolerance - if we were almost converged
            if (ktratio[b] < DOUBLE_EPSILON * 100.0 &&
                (prev_gap_abs[b] < tolGapAbs || prev_gap_rel[b] < tolGapRel)) {
                status[b] = 10; // InsufficientProgress
                iterations_per_batch[b] = static_cast<int32_t>(iter);
                atomicAdd(any_done, 1);  // Count how many batches are done
                return;
            }

            // Going backwards - stop if residuals diverge significantly
            if (ktratio[b] < 1.0) {
                if ((res_dual[b] > tolFeas * 100.0 && res_dual[b] > prev_res_dual[b] * 100.0) ||
                    (res_primal[b] > tolFeas * 100.0 && res_primal[b] > prev_res_primal[b] * 100.0)) {
                    status[b] = 10; // InsufficientProgress
                    iterations_per_batch[b] = static_cast<int32_t>(iter);
                    atomicAdd(any_done, 1);  // Count how many batches are done
                    return;
                }
            }
        }

        // Gap regression — analogous to residual regression, catches the case
        // where a single bad step collapses τ and sends gap_abs to +∞ while
        // residuals drift downward. Guarded on prev_gap_abs < reducedTol so
        // this fires only on the "near-optimum then blew up" pathology, not
        // on rough early-iterate steps where gap naturally oscillates.
        if (ktratio[b] < 1.0 &&
            prev_gap_abs[b] < reducedTolGapAbs &&
            gap_abs[b] > reducedTolGapAbs * 100.0 &&
            gap_abs[b] > prev_gap_abs[b] * 100.0) {
            status[b] = 10; // InsufficientProgress
            iterations_per_batch[b] = static_cast<int32_t>(iter);
            atomicAdd(any_done, 1);
            return;
        }

        // In-loop AlmostSolved on stagnation: iterate has been within reduced
        // tolerances for two consecutive iters, hasn't reached tight tolerance,
        // and hasn't moved meaningfully. Prevents the solver from running to
        // MaxIter (and being ruined by a late bad step) when KKT-solve noise
        // pins gap_rel slightly above the tight tolerance. The "prev also in
        // reduced" guard avoids firing on trivial problems that converge
        // straight through the reduced zone to tight in a single iteration.
        bool reduced_converged =
            ((gap_abs[b] < reducedTolGapAbs) || (gap_rel[b] < reducedTolGapRel)) &&
            (res_primal[b] < reducedTolFeas) &&
            (res_dual[b] < reducedTolFeas) &&
            (ktratio[b] <= reducedTolKtRatio);
        bool prev_was_reduced =
            (prev_gap_abs[b] < reducedTolGapAbs) || (prev_gap_rel[b] < reducedTolGapRel);
        bool not_in_tight =
            (gap_abs[b] >= tolGapAbs) && (gap_rel[b] >= tolGapRel);
        if (reduced_converged && prev_was_reduced && not_in_tight) {
            double denom = fmax(fabs(prev_gap_abs[b]), 1e-300);
            double gap_rel_change = fabs(gap_abs[b] - prev_gap_abs[b]) / denom;
            // Threshold 1e-4: distinguishes "noise-floor frozen" from
            // "slow but real" convergence near the cone boundary.
            // - cuDSS noise on PSD pins gap changes to ~1e-8 per iter
            //   (4 orders below this threshold) — fires correctly.
            // - GenPowerCone lacks its Mehrotra higher-order correction
            //   (see genpowcone.rs combined_ds_shift), giving linear
            //   ~0.5% per-iter convergence near the boundary (3 orders
            //   above this threshold) — does not fire, letting the
            //   solver run to MaxIter and refine the iterate for the
            //   ill-conditioned backward pass.
            if (gap_rel_change < 1e-4) {
                status[b] = 4; // AlmostSolved
                iterations_per_batch[b] = static_cast<int32_t>(iter);
                atomicAdd(any_done, 1);
                return;
            }
        }
    }

    // 3. Check iteration limit
    if (iter >= maxIter) {
        status[b] = 7; // MaxIterations
        iterations_per_batch[b] = static_cast<int32_t>(iter);
        atomicAdd(any_done, 1);  // Count how many batches are done
    }
}

/**
 * @brief GPU kernel for backtracking line search with inline cone membership
 *
 * Each thread handles one batch and performs backtracking independently.
 * Checks both barrier function AND cone feasibility:
 * - Scalar barrier: central_coef * log(mu) - log(τ) - log(κ)
 * - Nonneg cone: s_i >= 0 for all nonneg elements
 * - SOC cone: s[0] >= ||s[1:]||_2 (s[0] >= sqrt(s[1]^2 + s[2]^2))
 * - Exp/power cones: checked via dual gradient feasibility
 *
 * Cone layout in s/z: [zero(numZero), nonneg(numNonneg), soc(variable dims), exp(numExp*3), power(numPower*3)]
 */
__global__ void backtrack_line_search_impl(
    double* __restrict__ alpha,
    const double* __restrict__ alpha_init,
    double* __restrict__ barrier_work,
    const double* __restrict__ s,
    const double* __restrict__ z,
    const double* __restrict__ tau,
    const double* __restrict__ kappa,
    const double* __restrict__ ds,
    const double* __restrict__ dz,
    const double* __restrict__ dtau,
    const double* __restrict__ dkappa,
    double* __restrict__ sz_dot_work,
    double backtrack_factor,
    int64_t degree,
    int64_t m,
    int64_t batchSize,
    int max_iters,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    const int64_t* __restrict__ socConeDims
) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batchSize) return;

    double alpha_val = alpha_init[batch_idx];
    double central_coef = static_cast<double>(degree + 1);

    // Backtracking loop
    for (int iter = 0; iter < max_iters; iter++) {
        // Compute current τ and κ at this step length
        double cur_tau = tau[batch_idx] + alpha_val * dtau[batch_idx];
        double cur_kappa = kappa[batch_idx] + alpha_val * dkappa[batch_idx];

        bool feasible = (cur_tau > 0.0 && cur_kappa > 0.0);

        // Compute s'z dot product and check cone membership inline
        double sz_sum = 0.0;
        int64_t base = batch_idx * m;
        int64_t offset = 0;

        // Skip zero cones (no feasibility constraints on s)
        for (int64_t i = 0; i < numZeroCones && feasible; i++) {
            double cur_s_i = s[base + offset] + alpha_val * ds[base + offset];
            double cur_z_i = z[base + offset] + alpha_val * dz[base + offset];
            sz_sum += cur_s_i * cur_z_i;
            offset++;
        }

        // Nonneg cones: s_i >= 0
        for (int64_t i = 0; i < numNonnegCones && feasible; i++) {
            double cur_s_i = s[base + offset] + alpha_val * ds[base + offset];
            double cur_z_i = z[base + offset] + alpha_val * dz[base + offset];
            sz_sum += cur_s_i * cur_z_i;
            if (cur_s_i < 0.0 || cur_z_i < 0.0) feasible = false;
            offset++;
        }

        // SOC cones (variable dim): s[0] >= ||s[1:]|| for both s and z
        for (int64_t c = 0; c < numSocCones && feasible; c++) {
            int64_t dim = socConeDims[c];

            double cs0 = s[base + offset] + alpha_val * ds[base + offset];
            double cz0 = z[base + offset] + alpha_val * dz[base + offset];
            sz_sum += cs0 * cz0;

            double s_tail_sq = 0.0;
            double z_tail_sq = 0.0;
            for (int64_t j = 1; j < dim; j++) {
                double cs_j = s[base + offset + j] + alpha_val * ds[base + offset + j];
                double cz_j = z[base + offset + j] + alpha_val * dz[base + offset + j];
                sz_sum += cs_j * cz_j;
                s_tail_sq += cs_j * cs_j;
                z_tail_sq += cz_j * cz_j;
            }

            if (cs0 < sqrt(s_tail_sq)) feasible = false;
            if (cz0 < sqrt(z_tail_sq)) feasible = false;
            offset += dim;
        }

        // Remaining elements (exp, power cones - just accumulate s'z, skip detailed check)
        for (int64_t i = offset; i < m; i++) {
            double cur_s_i = s[base + i] + alpha_val * ds[base + i];
            double cur_z_i = z[base + i] + alpha_val * dz[base + i];
            sz_sum += cur_s_i * cur_z_i;
        }

        if (!feasible) {
            alpha_val *= backtrack_factor;
            continue;
        }

        // Compute mu = (s'z + τ*κ) / (degree + 1)
        double mu = (sz_sum + cur_tau * cur_kappa) / central_coef;

        // Compute barrier: central_coef * log(mu) - log(τ) - log(κ)
        double barrier_val;
        if (mu > 0.0) {
            barrier_val = central_coef * log(mu) - log(cur_tau) - log(cur_kappa);
        } else {
            barrier_val = HUGE_VAL;
        }

        if (barrier_val < 1.0) {
            break;
        } else {
            alpha_val *= backtrack_factor;
        }
    }

    // Store final alpha for this batch
    alpha[batch_idx] = alpha_val;
}

void backtrack_line_search_kernel(
    double* alpha,
    const double* alpha_init,
    double* barrier_work,
    const double* s,
    const double* z,
    const double* tau,
    const double* kappa,
    const double* ds,
    const double* dz,
    const double* dtau,
    const double* dkappa,
    double* sz_dot,
    double backtrack_factor,
    int64_t degree,
    int64_t m,
    int64_t batchSize,
    int max_iters,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    const int64_t* socConeDims,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(backtrack_line_search_impl, numBlocks, threadsPerBlock, 0, stream,
        alpha, alpha_init, barrier_work, s, z, tau, kappa,
        ds, dz, dtau, dkappa, sz_dot, backtrack_factor,
        degree, m, batchSize, max_iters,
        numZeroCones, numNonnegCones, numSocCones, socConeDims);
}

/**
 * @brief Kernel 1: One thread per batch, determine which batches need saving.
 *
 * Checks status and atomically claims the solution_saved flag. Writes the
 * result to should_save[b], which is stable for Kernel 2 to read without
 * cross-block races.
 */
__global__ void mark_batches_to_save_impl(
    int32_t* __restrict__ should_save,
    int32_t* __restrict__ solution_saved,
    const int32_t* __restrict__ status,
    int64_t batchSize
) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    should_save[b] = 0;
    if (status[b] != 0 && solution_saved[b] == 0) {
        if (atomicCAS(&solution_saved[b], 0, 1) == 0) {
            should_save[b] = 1;
        }
    }
}

/**
 * @brief Kernel 2: Copy elements for batches marked by Kernel 1.
 *
 * All threads read should_save[b] which was fully written by the completed
 * Kernel 1, so there is no cross-block race.
 */
__global__ void copy_terminated_solutions_impl(
    const double* __restrict__ x_src,
    const double* __restrict__ s_src,
    const double* __restrict__ z_src,
    const double* __restrict__ tau_src,
    const double* __restrict__ kappa_src,
    const double* __restrict__ cost_primal_src,
    const double* __restrict__ cost_dual_src,
    double* __restrict__ x_dst,
    double* __restrict__ s_dst,
    double* __restrict__ z_dst,
    double* __restrict__ tau_dst,
    double* __restrict__ kappa_dst,
    double* __restrict__ cost_primal_dst,
    double* __restrict__ cost_dual_dst,
    const int32_t* __restrict__ should_save,
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t max_dim = max(n, m);
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize * max_dim) return;

    int64_t b = idx / max_dim;
    int64_t elem = idx % max_dim;

    if (!should_save[b]) return;

    if (elem < n) {
        x_dst[b * n + elem] = x_src[b * n + elem];
    }
    if (elem < m) {
        s_dst[b * m + elem] = s_src[b * m + elem];
        z_dst[b * m + elem] = z_src[b * m + elem];
    }
    if (elem == 0) {
        tau_dst[b] = tau_src[b];
        kappa_dst[b] = kappa_src[b];
        cost_primal_dst[b] = cost_primal_src[b];
        cost_dual_dst[b] = cost_dual_src[b];
    }
}

void save_terminated_solutions_kernel(
    const double* x_src,
    const double* s_src,
    const double* z_src,
    const double* tau_src,
    const double* kappa_src,
    const double* cost_primal_src,
    const double* cost_dual_src,
    double* x_dst,
    double* s_dst,
    double* z_dst,
    double* tau_dst,
    double* kappa_dst,
    double* cost_primal_dst,
    double* cost_dual_dst,
    int32_t* solution_saved,
    int32_t* should_save,
    const int32_t* status,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    // Kernel 1: one thread per batch, determine which to save
    {
        int threadsPerBlock = 256;
        int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
        MOREAU_KERNEL_LAUNCH(mark_batches_to_save_impl, numBlocks, threadsPerBlock, 0, stream,
            should_save, solution_saved, status, batchSize);
    }

    // Kernel 2: copy elements for marked batches (implicitly serialized on same stream)
    {
        int64_t total_elements = batchSize * std::max(n, m);
        int threadsPerBlock = 256;
        int numBlocks = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
        MOREAU_KERNEL_LAUNCH(copy_terminated_solutions_impl, numBlocks, threadsPerBlock, 0, stream,
            x_src, s_src, z_src,
            tau_src, kappa_src,
            cost_primal_src, cost_dual_src,
            x_dst, s_dst, z_dst,
            tau_dst, kappa_dst,
            cost_primal_dst, cost_dual_dst,
            should_save,
            n, m, batchSize);
    }
}

// Restore working variables for terminated batches from saved solution.
// This prevents converged batches from polluting UBATCH KKT factorization.
__global__ void restore_terminated_variables_impl(
    double* __restrict__ x_work,
    double* __restrict__ s_work,
    double* __restrict__ z_work,
    double* __restrict__ tau_work,
    double* __restrict__ kappa_work,
    const double* __restrict__ x_saved,
    const double* __restrict__ s_saved,
    const double* __restrict__ z_saved,
    const double* __restrict__ tau_saved,
    const double* __restrict__ kappa_saved,
    const int32_t* __restrict__ solution_saved,
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t max_dim = max(n, m);
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize * max_dim) return;

    int64_t b = idx / max_dim;
    int64_t elem = idx % max_dim;

    if (!solution_saved[b]) return;

    if (elem < n) {
        x_work[b * n + elem] = x_saved[b * n + elem];
    }
    if (elem < m) {
        s_work[b * m + elem] = s_saved[b * m + elem];
        z_work[b * m + elem] = z_saved[b * m + elem];
    }
    if (elem == 0) {
        tau_work[b] = tau_saved[b];
        kappa_work[b] = kappa_saved[b];
    }
}

void restore_terminated_variables_kernel(
    double* x_work,
    double* s_work,
    double* z_work,
    double* tau_work,
    double* kappa_work,
    const double* x_saved,
    const double* s_saved,
    const double* z_saved,
    const double* tau_saved,
    const double* kappa_saved,
    const int32_t* solution_saved,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    int64_t total_elements = batchSize * std::max(n, m);
    int threadsPerBlock = 256;
    int numBlocks = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    MOREAU_KERNEL_LAUNCH(restore_terminated_variables_impl, numBlocks, threadsPerBlock, 0, stream,
        x_work, s_work, z_work, tau_work, kappa_work,
        x_saved, s_saved, z_saved, tau_saved, kappa_saved,
        solution_saved, n, m, batchSize);
}

// Best-iterate snapshot: mark which batches are in the reduced-tolerance zone
// AND have a strictly tighter worst-ratio score than the snapshot already
// held. Score = max(res_primal/tolFeas, res_dual/tolFeas, gap_abs/tolGapAbs);
// lower is closer to tight convergence. See the CPU mirror in
// info.rs::try_save_best_iterate for the rationale.
__global__ void mark_best_iterate_impl(
    int32_t* __restrict__ should_save,
    int32_t* __restrict__ solution_saved,
    int32_t* __restrict__ d_best_saved,
    const int32_t* __restrict__ status,
    const double* __restrict__ gap_abs,
    const double* __restrict__ gap_rel,
    const double* __restrict__ res_primal,
    const double* __restrict__ res_dual,
    const double* __restrict__ ktratio,
    const double* __restrict__ best_res_primal,
    const double* __restrict__ best_res_dual,
    const double* __restrict__ best_gap_abs,
    double tolGapAbs,
    double tolFeas,
    double reducedTolGapAbs,
    double reducedTolGapRel,
    double reducedTolFeas,
    double reducedTolKtRatio,
    int64_t batchSize
) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    should_save[b] = 0;
    if (status[b] != 0) return;

    bool gap_converged = (gap_abs[b] < reducedTolGapAbs) || (gap_rel[b] < reducedTolGapRel);
    bool feasible = (res_primal[b] < reducedTolFeas) && (res_dual[b] < reducedTolFeas);
    bool interior = (ktratio[b] <= reducedTolKtRatio);
    if (!(gap_converged && feasible && interior)) return;

    double pf = fmax(tolFeas, DBL_EPSILON);
    double gf = fmax(tolGapAbs, DBL_EPSILON);
    double cur_score = fmax(fmax(res_primal[b] / pf, res_dual[b] / pf), gap_abs[b] / gf);
    if (d_best_saved[b]) {
        double prev_score = fmax(
            fmax(best_res_primal[b] / pf, best_res_dual[b] / pf),
            best_gap_abs[b] / gf
        );
        if (cur_score >= prev_score) return;
    }

    // solution_saved==1 blocks the downstream save_terminated kernel from
    // clobbering with a bad terminal iterate; d_best_saved records that the
    // snapshot is an in-zone one so the post-loop restore knows to promote
    // status to AlmostSolved.
    should_save[b] = 1;
    solution_saved[b] = 1;
    d_best_saved[b] = 1;
}

// Best-iterate snapshot: copy variables and metrics for marked batches.
__global__ void copy_best_iterate_impl(
    const double* __restrict__ x_src,
    const double* __restrict__ s_src,
    const double* __restrict__ z_src,
    const double* __restrict__ tau_src,
    const double* __restrict__ kappa_src,
    const double* __restrict__ cost_primal_src,
    const double* __restrict__ cost_dual_src,
    const double* __restrict__ res_primal_src,
    const double* __restrict__ res_dual_src,
    const double* __restrict__ gap_abs_src,
    const double* __restrict__ gap_rel_src,
    const double* __restrict__ ktratio_src,
    double* __restrict__ x_dst,
    double* __restrict__ s_dst,
    double* __restrict__ z_dst,
    double* __restrict__ tau_dst,
    double* __restrict__ kappa_dst,
    double* __restrict__ cost_primal_dst,
    double* __restrict__ cost_dual_dst,
    double* __restrict__ best_res_primal,
    double* __restrict__ best_res_dual,
    double* __restrict__ best_gap_abs,
    double* __restrict__ best_gap_rel,
    double* __restrict__ best_ktratio,
    const int32_t* __restrict__ should_save,
    int64_t n,
    int64_t m,
    int64_t batchSize
) {
    int64_t max_dim = max(n, m);
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize * max_dim) return;

    int64_t b = idx / max_dim;
    int64_t elem = idx % max_dim;

    if (!should_save[b]) return;

    if (elem < n) {
        x_dst[b * n + elem] = x_src[b * n + elem];
    }
    if (elem < m) {
        s_dst[b * m + elem] = s_src[b * m + elem];
        z_dst[b * m + elem] = z_src[b * m + elem];
    }
    if (elem == 0) {
        tau_dst[b] = tau_src[b];
        kappa_dst[b] = kappa_src[b];
        cost_primal_dst[b] = cost_primal_src[b];
        cost_dual_dst[b] = cost_dual_src[b];
        best_res_primal[b] = res_primal_src[b];
        best_res_dual[b] = res_dual_src[b];
        best_gap_abs[b] = gap_abs_src[b];
        best_gap_rel[b] = gap_rel_src[b];
        best_ktratio[b] = ktratio_src[b];
    }
}

void save_best_iterate_kernel(
    const double* x_src,
    const double* s_src,
    const double* z_src,
    const double* tau_src,
    const double* kappa_src,
    const double* cost_primal_src,
    const double* cost_dual_src,
    const double* res_primal_src,
    const double* res_dual_src,
    const double* gap_abs_src,
    const double* gap_rel_src,
    const double* ktratio_src,
    double* x_dst,
    double* s_dst,
    double* z_dst,
    double* tau_dst,
    double* kappa_dst,
    double* cost_primal_dst,
    double* cost_dual_dst,
    double* best_res_primal,
    double* best_res_dual,
    double* best_gap_abs,
    double* best_gap_rel,
    double* best_ktratio,
    int32_t* solution_saved,
    int32_t* should_save,
    int32_t* d_best_saved,
    const int32_t* status,
    double tolGapAbs,
    double tolFeas,
    double reducedTolGapAbs,
    double reducedTolGapRel,
    double reducedTolFeas,
    double reducedTolKtRatio,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream
) {
    // `should_save` is the scratch already owned by Solution::should_save. It's
    // reused later in the same iteration by save_terminated_solutions — safe
    // because both pipelines run sequentially on the same CUDA stream.
    int threadsPerBlock = 256;
    int numBlocksMark = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    MOREAU_KERNEL_LAUNCH(mark_best_iterate_impl, numBlocksMark, threadsPerBlock, 0, stream,
        should_save, solution_saved, d_best_saved,
        status,
        gap_abs_src, gap_rel_src, res_primal_src, res_dual_src, ktratio_src,
        best_res_primal, best_res_dual, best_gap_abs,
        tolGapAbs, tolFeas,
        reducedTolGapAbs, reducedTolGapRel, reducedTolFeas, reducedTolKtRatio,
        batchSize);

    int64_t total_elements = batchSize * std::max(n, m);
    int numBlocksCopy = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
    MOREAU_KERNEL_LAUNCH(copy_best_iterate_impl, numBlocksCopy, threadsPerBlock, 0, stream,
        x_src, s_src, z_src, tau_src, kappa_src,
        cost_primal_src, cost_dual_src,
        res_primal_src, res_dual_src,
        gap_abs_src, gap_rel_src, ktratio_src,
        x_dst, s_dst, z_dst, tau_dst, kappa_dst,
        cost_primal_dst, cost_dual_dst,
        best_res_primal, best_res_dual,
        best_gap_abs, best_gap_rel, best_ktratio,
        should_save,
        n, m, batchSize);
}

// Post-loop restore: promote status to AlmostSolved for batches with a
// best-iterate snapshot that landed on a non-convergent terminal status,
// and overwrite info metrics with the snapshot so downstream post_process
// and the solution status display reflect the restored iterate.
__global__ void restore_best_iterate_impl(
    double* __restrict__ res_primal,
    double* __restrict__ res_dual,
    double* __restrict__ gap_abs,
    double* __restrict__ gap_rel,
    double* __restrict__ ktratio,
    double* __restrict__ cost_primal,
    double* __restrict__ cost_dual,
    const double* __restrict__ best_res_primal,
    const double* __restrict__ best_res_dual,
    const double* __restrict__ best_gap_abs,
    const double* __restrict__ best_gap_rel,
    const double* __restrict__ best_ktratio,
    const double* __restrict__ best_cost_primal,
    const double* __restrict__ best_cost_dual,
    int32_t* __restrict__ status,
    const int32_t* __restrict__ d_best_saved,
    int64_t batchSize
) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;
    if (!d_best_saved[b]) return;

    int32_t s = status[b];
    // Promote only on non-convergent terminations: InsufficientProgress(10),
    // NumericalError(9), MaxIterations(7), MaxTime(8). Leave Solved/Almost*
    // and Primal/Dual(in)feasible alone — the solver's chosen terminal
    // status already reflects the right iterate.
    bool promote = (s == 7 || s == 8 || s == 9 || s == 10);
    if (!promote) return;

    res_primal[b] = best_res_primal[b];
    res_dual[b] = best_res_dual[b];
    gap_abs[b] = best_gap_abs[b];
    gap_rel[b] = best_gap_rel[b];
    ktratio[b] = best_ktratio[b];
    cost_primal[b] = best_cost_primal[b];
    cost_dual[b] = best_cost_dual[b];
    status[b] = 4; // AlmostSolved
}

void restore_best_iterate_kernel(
    double* res_primal,
    double* res_dual,
    double* gap_abs,
    double* gap_rel,
    double* ktratio,
    double* cost_primal,
    double* cost_dual,
    const double* best_res_primal,
    const double* best_res_dual,
    const double* best_gap_abs,
    const double* best_gap_rel,
    const double* best_ktratio,
    const double* best_cost_primal,
    const double* best_cost_dual,
    int32_t* status,
    const int32_t* d_best_saved,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    MOREAU_KERNEL_LAUNCH(restore_best_iterate_impl, numBlocks, threadsPerBlock, 0, stream,
        res_primal, res_dual, gap_abs, gap_rel, ktratio, cost_primal, cost_dual,
        best_res_primal, best_res_dual, best_gap_abs, best_gap_rel, best_ktratio,
        best_cost_primal, best_cost_dual,
        status, d_best_saved, batchSize);
}

/**
 * @brief Pack interleaved kernel implementation
 *
 * Each thread handles one element in the output array.
 * Determines which batch, whether it's x or z, and which element within x or z.
 */
__global__ void pack_interleaved_impl(
    double* __restrict__ out,
    const double* __restrict__ x,
    const double* __restrict__ z,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batchSize
) {
    int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = n + m + p;  // Size per batch in output
    int64_t total_size = N * batchSize;

    if (global_idx >= total_size) return;

    // Determine which batch and position within batch
    int64_t batch_idx = global_idx / N;
    int64_t local_idx = global_idx % N;

    if (local_idx < n) {
        // This is an x element
        out[global_idx] = x[batch_idx * n + local_idx];
    } else if (local_idx < n + m) {
        // This is a z element
        int64_t z_idx = local_idx - n;
        out[global_idx] = z[batch_idx * m + z_idx];
    } else {
        // Expansion entry: zero-fill
        out[global_idx] = 0.0;
    }
}

void pack_interleaved_kernel(
    double* out,
    const double* x,
    const double* z,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batchSize,
    cudaStream_t stream
) {
    int64_t total_size = (n + m + p) * batchSize;
    if (total_size == 0) return;
    int threadsPerBlock = 256;
    int numBlocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(pack_interleaved_impl, numBlocks, threadsPerBlock, 0, stream,
        out, x, z, n, m, p, batchSize);
}

// ============================================================================
// Fused waxpby + pack interleaved: x_src copied, z_part = a*z_a + b*z_b
// Eliminates 1 kernel per KKT solve (waxpby into workz then pack)
// ============================================================================
__global__ void waxpby_and_pack_interleaved_impl(
    double* __restrict__ out,
    const double* __restrict__ x_src,
    const double* __restrict__ z_a,
    const double* __restrict__ z_b,
    double a,
    double b,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batchSize
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    const int64_t N = n + m + p;
    if (batch >= batchSize || idx >= N) return;

    int64_t out_idx = batch * N + idx;
    if (idx < n) {
        out[out_idx] = x_src[batch * n + idx];
    } else if (idx < n + m) {
        int64_t z_idx = batch * m + (idx - n);
        out[out_idx] = a * z_a[z_idx] + b * z_b[z_idx];
    } else {
        out[out_idx] = 0.0;
    }
}

void waxpby_and_pack_interleaved_kernel(
    double* out,
    const double* x_src,
    const double* z_a,
    const double* z_b,
    double a,
    double b,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batchSize,
    cudaStream_t stream
) {
    const int64_t N = n + m + p;
    if (N * batchSize == 0) return;
    dim3 block(256, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(waxpby_and_pack_interleaved_impl, grid, block, 0, stream,
        out, x_src, z_a, z_b, a, b, n, m, p, batchSize);
}

/**
 * @brief Unpack interleaved kernel implementation
 *
 * Each thread handles one element in the input array.
 * Determines which batch, whether it's x or z, and where to write it.
 */
__global__ void unpack_interleaved_impl(
    double* __restrict__ x,
    double* __restrict__ z,
    const double* __restrict__ in,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batchSize
) {
    int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t N = n + m + p;  // Size per batch in input
    int64_t total_size = N * batchSize;

    if (global_idx >= total_size) return;

    // Determine which batch and position within batch
    int64_t batch_idx = global_idx / N;
    int64_t local_idx = global_idx % N;

    if (local_idx < n) {
        // This is an x element
        x[batch_idx * n + local_idx] = in[global_idx];
    } else if (local_idx < n + m) {
        // This is a z element
        int64_t z_idx = local_idx - n;
        z[batch_idx * m + z_idx] = in[global_idx];
    }
    // else: expansion entry, discard
}

void unpack_interleaved_kernel(
    double* x,
    double* z,
    const double* in,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batchSize,
    cudaStream_t stream
) {
    int64_t total_size = (n + m + p) * batchSize;
    if (total_size == 0) return;
    int threadsPerBlock = 256;
    int numBlocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(unpack_interleaved_impl, numBlocks, threadsPerBlock, 0, stream,
        x, z, in, n, m, p, batchSize);
}

void check_termination_async_kernel(
    int32_t* status,
    int32_t* d_any_done,
    int32_t* h_any_done,  // Pinned host memory for polling (may be same as d_any_done for mapped memory)
    int32_t* d_iterations_per_batch,
    const double* gap_abs,
    const double* gap_rel,
    const double* res_primal,
    const double* res_dual,
    const double* ktratio,
    const double* res_primal_inf,
    const double* res_dual_inf,
    const double* dot_qx,
    const double* dot_bz,
    const double* prev_res_primal,
    const double* prev_res_dual,
    const double* prev_gap_abs,
    const double* prev_gap_rel,
    double tolGapAbs,
    double tolGapRel,
    double tolFeas,
    double tolInfeasAbs,
    double tolInfeasRel,
    double tolKtRatio,
    double reducedTolGapAbs,
    double reducedTolGapRel,
    double reducedTolFeas,
    double reducedTolInfeasAbs,
    double reducedTolInfeasRel,
    double reducedTolKtRatio,
    int64_t iter,
    int64_t maxIter,
    double solve_time_seconds,
    double timeLimit,
    int64_t batchSize,
    cudaStream_t stream
) {
    // Reset d_any_done to 0 (async, no sync)
    cudaMemsetAsync(d_any_done, 0, sizeof(int32_t), stream);

    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    // Launch termination check kernel (same implementation as check_termination_impl)
    MOREAU_KERNEL_LAUNCH(check_termination_impl, numBlocks, threadsPerBlock, 0, stream,
        status, d_any_done, d_iterations_per_batch, gap_abs, gap_rel, res_primal, res_dual, ktratio,
        res_primal_inf, res_dual_inf, dot_qx, dot_bz,
        prev_res_primal, prev_res_dual, prev_gap_abs, prev_gap_rel,
        tolGapAbs, tolGapRel, tolFeas, tolInfeasAbs, tolInfeasRel, tolKtRatio,
        reducedTolGapAbs, reducedTolGapRel, reducedTolFeas,
        reducedTolInfeasAbs, reducedTolInfeasRel, reducedTolKtRatio,
        iter, maxIter, solve_time_seconds, timeLimit, batchSize);

    // NO SYNC HERE - caller can poll h_any_done directly (pinned/mapped memory)
}

// Set all batch statuses to a given value
__global__ void set_all_status_impl(
    int32_t* __restrict__ status,
    int32_t value,
    int64_t batchSize
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize) return;
    status[i] = value;
}

void set_all_status_kernel(
    int32_t* status,
    int32_t value,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (batchSize == 0) return;
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    MOREAU_KERNEL_LAUNCH(set_all_status_impl, numBlocks, threadsPerBlock, 0, stream,
        status, value, batchSize);
}

// Set status only for batches whose solution has NOT been saved yet.
// Already-converged batches (solution_saved[b] == 1) keep their status.
__global__ void set_unsaved_status_impl(
    int32_t* __restrict__ status,
    const int32_t* __restrict__ solution_saved,
    int32_t value,
    int64_t batchSize
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize) return;
    if (solution_saved[i] == 0) {
        status[i] = value;
    }
}

void set_unsaved_status_kernel(
    int32_t* status,
    const int32_t* solution_saved,
    int32_t value,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (batchSize == 0) return;
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    MOREAU_KERNEL_LAUNCH(set_unsaved_status_impl, numBlocks, threadsPerBlock, 0, stream,
        status, solution_saved, value, batchSize);
}

// Set status for batches where scaling failed AND solution not yet saved.
__global__ void set_unsaved_status_where_failed_impl(
    int32_t* __restrict__ status,
    const int32_t* __restrict__ solution_saved,
    int32_t value,
    int64_t batchSize,
    const volatile int32_t* __restrict__ scaling_success
) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize) return;
    if (solution_saved[i] == 0 && scaling_success[i] == 0) {
        status[i] = value;
    }
}

void set_unsaved_status_kernel(
    int32_t* status,
    const int32_t* solution_saved,
    int32_t value,
    int64_t batchSize,
    const volatile int32_t* scaling_success,
    cudaStream_t stream
) {
    if (batchSize == 0) return;
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    MOREAU_KERNEL_LAUNCH(set_unsaved_status_where_failed_impl, numBlocks, threadsPerBlock, 0, stream,
        status, solution_saved, value, batchSize, scaling_success);
}

__global__ void zero_alpha_for_terminated_impl(
    double* __restrict__ alpha,
    const int32_t* __restrict__ status,
    int64_t batchSize
) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;
    if (status[b] != 0) {
        alpha[b] = 0.0;
    }
}

void zero_alpha_for_terminated_kernel(
    double* alpha,
    const int32_t* status,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (batchSize == 0) return;
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    MOREAU_KERNEL_LAUNCH(zero_alpha_for_terminated_impl, numBlocks, threadsPerBlock, 0, stream,
        alpha, status, batchSize);
}

// Per-batch PrimalDual→Dual fallback. Mirror of CPU
// `core/solver.rs::strategy_checkpoint_small_step`: when α<min_switch_step_length
// and the batch is in PrimalDual, switch to Dual and zero α (skip step).
// 1 = PrimalDual, 0 = Dual.
__global__ void switch_scaling_on_small_step_impl(
    double* __restrict__ alpha,
    int8_t* pd_enabled_per_batch,
    int64_t batchSize,
    double min_switch_step_length
) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;
    if (pd_enabled_per_batch[b] != 1) return;
    if (alpha[b] >= min_switch_step_length) return;
    pd_enabled_per_batch[b] = 0;
    alpha[b] = 0.0;
}

void switch_scaling_on_small_step_kernel(
    double* alpha,
    int8_t* pd_enabled_per_batch,
    int64_t batchSize,
    double min_switch_step_length,
    cudaStream_t stream
) {
    if (batchSize == 0 || pd_enabled_per_batch == nullptr) return;
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    MOREAU_KERNEL_LAUNCH(switch_scaling_on_small_step_impl, numBlocks, threadsPerBlock, 0, stream,
        alpha, pd_enabled_per_batch, batchSize, min_switch_step_length);
}

// CPU analog: see `strategy_checkpoint_small_step` in
// packages/moreau-cpu/src/solver/core/solver.rs. When cone backtracking forces
// α below `minTerminateStepLength` the Newton direction has nothing left to
// contribute, so terminating immediately matches CPU semantics. We only mark
// still-active batches; converged batches keep their existing status.
__global__ void small_step_terminate_impl(
    const double* __restrict__ alpha,
    int32_t* __restrict__ status,
    int32_t* __restrict__ iterations_per_batch,
    double minTerminateStepLength,
    int64_t iter,
    int64_t batchSize
) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;
    if (status[b] != 0) return;  // already terminated
    if (alpha[b] <= minTerminateStepLength) {
        status[b] = 10;  // InsufficientProgress (promoted to AlmostSolved by post_process)
        iterations_per_batch[b] = static_cast<int32_t>(iter);
    }
}

void small_step_terminate_kernel(
    const double* alpha,
    int32_t* status,
    int32_t* iterations_per_batch,
    double minTerminateStepLength,
    int64_t iter,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (batchSize == 0) return;
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    MOREAU_KERNEL_LAUNCH(small_step_terminate_impl, numBlocks, threadsPerBlock, 0, stream,
        alpha, status, iterations_per_batch, minTerminateStepLength, iter, batchSize);
}

__global__ void init_pd_enabled_per_batch_impl(int8_t* arr, int64_t batchSize, int8_t value) {
    int64_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;
    arr[b] = value;
}

void init_pd_enabled_per_batch_kernel(
    int8_t* pd_enabled_per_batch,
    int64_t batchSize,
    int8_t value,
    cudaStream_t stream
) {
    if (batchSize == 0 || pd_enabled_per_batch == nullptr) return;
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    MOREAU_KERNEL_LAUNCH(init_pd_enabled_per_batch_impl, numBlocks, threadsPerBlock, 0, stream,
        pd_enabled_per_batch, batchSize, value);
}

/**
 * @brief GPU kernel for computing per-batch shift alpha for cone initialization
 *
 * Each thread handles one batch. Computes:
 *   target = max(1.0, pos_margin * 0.1 / degree)
 *   if min_margin <= 0: alpha = -min_margin + target
 *   elif min_margin < target: alpha = target - min_margin
 *   else: alpha = 0
 */
__global__ void compute_init_shift_alpha_impl(
    double* __restrict__ alpha,
    const double* __restrict__ min_margin,
    const double* __restrict__ pos_margin,
    double degree,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    double min_m = min_margin[batch];
    double pos_m = pos_margin[batch];

    // Compute target: max(1.0, pos_margin * 0.1 / degree)
    double target = fmax(1.0, (pos_m * 0.1) / degree);

    // Compute alpha based on margin condition
    double alpha_val;
    if (min_m <= 0.0) {
        // Outside cone: shift to bring inside, then to target
        alpha_val = -min_m + target;
    } else if (min_m < target) {
        // Inside but small margin: shift to target
        alpha_val = target - min_m;
    } else {
        // Good margin: no shift needed (except for zero cone which is handled separately)
        alpha_val = 0.0;
    }

    alpha[batch] = alpha_val;
}

void compute_init_shift_alpha_kernel(
    double* alpha,
    const double* min_margin,
    const double* pos_margin,
    double degree,
    int64_t batchSize,
    cudaStream_t stream
) {
    if (batchSize == 0) return;

    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(compute_init_shift_alpha_impl, numBlocks, threadsPerBlock, 0, stream,
        alpha, min_margin, pos_margin, degree, batchSize);
}

// ============================================================================
// Fused tau numerator base: ξ=x/τ, tau_num = (rτ - rκ/τ) + q'x1 + b'z1  (3 → 1 kernel)
// ============================================================================
__global__ void fused_tau_numerator_base_kernel_impl(
    double* __restrict__ xi_out,          // [n * batchSize] — output ξ = x/τ
    double* __restrict__ tau_num,         // [batchSize] — output (atomicAdd)
    const double* __restrict__ x_vec,     // [n * batchSize] — variables.x
    const double* __restrict__ tau_vec,   // [batchSize] — variables.τ
    const double* __restrict__ rtau,      // [batchSize] — rhs.τ
    const double* __restrict__ rkappa,    // [batchSize] — rhs.κ
    const double* __restrict__ q,         // [n * batchSize]
    const double* __restrict__ x1,        // [n * batchSize]
    const double* __restrict__ b_vec,     // [m * batchSize]
    const double* __restrict__ z1,        // [m * batchSize]
    int64_t n,
    int64_t m,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    if (batch >= batchSize) return;

    double tau_val = tau_vec[batch];

    // Phase 1: compute ξ and accumulate partial dot products
    double local_qx1 = 0.0;
    for (int64_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        int64_t gi = batch * n + i;
        xi_out[gi] = x_vec[gi] / tau_val;
        local_qx1 += q[gi] * x1[gi];
    }

    double local_bz1 = 0.0;
    for (int64_t i = idx; i < m; i += blockDim.x * gridDim.x) {
        int64_t gi = batch * m + i;
        local_bz1 += b_vec[gi] * z1[gi];
    }

    // Phase 2: warp-shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_qx1 += __shfl_down_sync(0xffffffff, local_qx1, offset);
        local_bz1 += __shfl_down_sync(0xffffffff, local_bz1, offset);
    }

    // Shared memory reduction across warps
    __shared__ double sh_qx1[8];  // max 256/32 = 8 warps
    __shared__ double sh_bz1[8];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (lane == 0) {
        sh_qx1[warp_id] = local_qx1;
        sh_bz1[warp_id] = local_bz1;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        double val_qx1 = (lane < num_warps) ? sh_qx1[lane] : 0.0;
        double val_bz1 = (lane < num_warps) ? sh_bz1[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val_qx1 += __shfl_down_sync(0xffffffff, val_qx1, offset);
            val_bz1 += __shfl_down_sync(0xffffffff, val_bz1, offset);
        }
        if (lane == 0) {
            double base = (blockIdx.x == 0) ? (rtau[batch] - rkappa[batch] / tau_val) : 0.0;
            atomicAdd(&tau_num[batch], base + val_qx1 + val_bz1);
        }
    }
}

void fused_tau_numerator_base_kernel(
    double* xi_out,
    double* tau_num,
    const double* x_vec,
    const double* tau_vec,
    const double* rtau,
    const double* rkappa,
    const double* q,
    const double* x1,
    const double* b_vec,
    const double* z1,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream)
{
    if (batchSize == 0) return;
    cudaMemsetAsync(tau_num, 0, sizeof(double) * batchSize, stream);
    int64_t max_dim = (n > m) ? n : m;
    if (max_dim == 0) max_dim = 1;
    dim3 blk(256, 1, 1);
    dim3 grd((max_dim + blk.x - 1) / blk.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(fused_tau_numerator_base_kernel_impl, grd, blk, 0, stream,
        xi_out, tau_num, x_vec, tau_vec, rtau, rkappa,
        q, x1, b_vec, z1, n, m, batchSize);
}

// ============================================================================
// Fused tau denominator base: tau_den = κ/τ + (-q'x2) + (-b'z2)  (2 → 1 kernel)
// ============================================================================
__global__ void fused_tau_denominator_base_kernel_impl(
    double* __restrict__ tau_den,         // [batchSize] — output (atomicAdd)
    const double* __restrict__ kappa,     // [batchSize] — variables.κ
    const double* __restrict__ tau_vec,   // [batchSize] — variables.τ
    const double* __restrict__ q,         // [n * batchSize]
    const double* __restrict__ x2,        // [n * batchSize]
    const double* __restrict__ b_vec,     // [m * batchSize]
    const double* __restrict__ z2,        // [m * batchSize]
    int64_t n,
    int64_t m,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    if (batch >= batchSize) return;

    double local_qx2 = 0.0;
    for (int64_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        int64_t gi = batch * n + i;
        local_qx2 += q[gi] * x2[gi];
    }

    double local_bz2 = 0.0;
    for (int64_t i = idx; i < m; i += blockDim.x * gridDim.x) {
        int64_t gi = batch * m + i;
        local_bz2 += b_vec[gi] * z2[gi];
    }

    // Warp-shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_qx2 += __shfl_down_sync(0xffffffff, local_qx2, offset);
        local_bz2 += __shfl_down_sync(0xffffffff, local_bz2, offset);
    }

    // Shared memory reduction across warps
    __shared__ double sh_qx2[8];
    __shared__ double sh_bz2[8];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (lane == 0) {
        sh_qx2[warp_id] = local_qx2;
        sh_bz2[warp_id] = local_bz2;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        double val_qx2 = (lane < num_warps) ? sh_qx2[lane] : 0.0;
        double val_bz2 = (lane < num_warps) ? sh_bz2[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val_qx2 += __shfl_down_sync(0xffffffff, val_qx2, offset);
            val_bz2 += __shfl_down_sync(0xffffffff, val_bz2, offset);
        }
        if (lane == 0) {
            double base = (blockIdx.x == 0) ? (kappa[batch] / tau_vec[batch]) : 0.0;
            atomicAdd(&tau_den[batch], base - val_qx2 - val_bz2);
        }
    }
}

void fused_tau_denominator_base_kernel(
    double* tau_den,
    const double* kappa,
    const double* tau_vec,
    const double* q,
    const double* x2,
    const double* b_vec,
    const double* z2,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream)
{
    if (batchSize == 0) return;
    cudaMemsetAsync(tau_den, 0, sizeof(double) * batchSize, stream);
    int64_t max_dim = (n > m) ? n : m;
    if (max_dim == 0) max_dim = 1;
    dim3 blk(256, 1, 1);
    dim3 grd((max_dim + blk.x - 1) / blk.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(fused_tau_denominator_base_kernel_impl, grd, blk, 0, stream,
        tau_den, kappa, tau_vec, q, x2, b_vec, z2, n, m, batchSize);
}

// ============================================================================
// Fused double quad form: tau_den += (ξ-x2)'P(ξ-x2) - x2'Px2  (4 → 1 kernel)
// Eliminates ξ_minus_x2 temporary vector.
// ============================================================================
__global__ void fused_double_quad_form_kernel_impl(
    double* __restrict__ tau_den,         // [batchSize] — accumulate into (atomicAdd)
    const int64_t* __restrict__ P_rowOffsets,
    const int64_t* __restrict__ P_colIndices,
    const double* __restrict__ P_values,
    const double* __restrict__ xi_vec,    // [n * batchSize] — ξ
    const double* __restrict__ x2_vec,    // [n * batchSize] — x2
    int64_t n,
    int64_t nnzP,
    int64_t batchSize)
{
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    if (batch >= batchSize) return;

    // Threads with row >= n contribute 0 (no early return — must participate in reduction)
    double contribution = 0.0;

    if (row < n) {
        int64_t batch_val = batch * nnzP;
        int64_t batch_vec = batch * n;

        int64_t rs = P_rowOffsets[row];
        int64_t re = P_rowOffsets[row + 1];

        double Py_diff = 0.0;  // P*(ξ-x2) for this row
        double Py_x2 = 0.0;    // P*x2 for this row
        for (int64_t j = rs; j < re; ++j) {
            int64_t col = P_colIndices[j];
            double pval = P_values[batch_val + j];
            double xi_col = xi_vec[batch_vec + col];
            double x2_col = x2_vec[batch_vec + col];
            Py_diff += pval * (xi_col - x2_col);
            Py_x2 += pval * x2_col;
        }

        double xi_row = xi_vec[batch_vec + row];
        double x2_row = x2_vec[batch_vec + row];
        contribution = (xi_row - x2_row) * Py_diff - x2_row * Py_x2;
    }

    // Warp-shuffle reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        contribution += __shfl_down_sync(0xffffffff, contribution, offset);
    }

    // Shared memory reduction across warps
    __shared__ double sh[8];
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;

    if (lane == 0) {
        sh[warp_id] = contribution;
    }
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + 31) / 32;
        double val = (lane < num_warps) ? sh[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) {
            atomicAdd(&tau_den[batch], val);
        }
    }
}

void fused_double_quad_form_kernel(
    double* tau_den,
    const int64_t* P_rowOffsets,
    const int64_t* P_colIndices,
    const double* P_values,
    const double* xi_vec,
    const double* x2_vec,
    int64_t n,
    int64_t nnzP,
    int64_t batchSize,
    cudaStream_t stream)
{
    if (n == 0 || batchSize == 0) return;
    dim3 blk(256, 1, 1);
    dim3 grd((n + blk.x - 1) / blk.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(fused_double_quad_form_kernel_impl, grd, blk, 0, stream,
        tau_den, P_rowOffsets, P_colIndices, P_values,
        xi_vec, x2_vec, n, nnzP, batchSize);
}

// ============================================================================
// Sanitize inf in b for nonneg cone rows
// ============================================================================

static __global__ void sanitize_inf_b_kernel_impl(
    double* __restrict__ b,
    double* __restrict__ A_values,
    const int64_t* __restrict__ A_rowOffsets,
    int* __restrict__ infeasible_flags,
    int64_t zero_row_start,
    int64_t zero_row_end,
    int64_t nonneg_row_start,
    int64_t nonneg_row_end,
    int64_t m,
    int64_t nnzA,
    int64_t batchSize)
{
    int64_t num_rows = (zero_row_end - zero_row_start) + (nonneg_row_end - nonneg_row_start);
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batchSize * num_rows;
    if (idx >= total) return;

    int64_t batch = idx / num_rows;
    int64_t local_row = idx % num_rows;

    int64_t num_zero = zero_row_end - zero_row_start;
    double* b_batch = b + batch * m;

    if (local_row < num_zero) {
        // Zero cone row: any inf is infeasible
        int64_t row = zero_row_start + local_row;
        if (isinf(b_batch[row])) {
            infeasible_flags[batch] = 1;
        }
    } else {
        // Nonneg cone row
        int64_t row = nonneg_row_start + (local_row - num_zero);
        double bi = b_batch[row];
        if (isinf(bi)) {
            if (bi > 0.0) {
                // +inf: vacuous, zero A row and set b=1 (not 0, because
                // b=0 forces s=0 and the IPM barrier -log(s) diverges)
                b_batch[row] = 1.0;
                double* A_batch = A_values + batch * nnzA;
                int64_t row_start = A_rowOffsets[row];
                int64_t row_end = A_rowOffsets[row + 1];
                for (int64_t k = row_start; k < row_end; k++) {
                    A_batch[k] = 0.0;
                }
            } else {
                // -inf: infeasible
                infeasible_flags[batch] = 1;
            }
        }
    }
}

void sanitize_inf_b_kernel(
    double* b,
    double* A_values,
    const int64_t* A_rowOffsets,
    int* infeasible_flags,
    int64_t zero_row_start,
    int64_t zero_row_end,
    int64_t nonneg_row_start,
    int64_t nonneg_row_end,
    int64_t m,
    int64_t nnzA,
    int64_t batchSize,
    cudaStream_t stream)
{
    int64_t num_rows = (zero_row_end - zero_row_start) + (nonneg_row_end - nonneg_row_start);
    if (num_rows == 0 || batchSize == 0) return;

    // Clear infeasible flags
    cudaMemsetAsync(infeasible_flags, 0, sizeof(int) * batchSize, stream);

    int64_t total = batchSize * num_rows;
    dim3 blk(256);
    dim3 grd((total + blk.x - 1) / blk.x);
    MOREAU_KERNEL_LAUNCH(sanitize_inf_b_kernel_impl, grd, blk, 0, stream,
        b, A_values, A_rowOffsets, infeasible_flags,
        zero_row_start, zero_row_end,
        nonneg_row_start, nonneg_row_end, m, nnzA, batchSize);
}

// ============================================================================
// YOLO mode: NaN-gated snapshot kernels
// ============================================================================

/**
 * @brief Kernel 1: Check variables for NaN, per batch independently.
 *
 * Sets has_nan[b] = 1 (via atomicMax) if any element of x, s, z, τ, or κ
 * for batch b is NaN. Must be preceded by cudaMemsetAsync(has_nan, 0, ...).
 */
__global__ void yolo_check_nan_impl(
    int32_t* __restrict__ has_nan,
    const double* __restrict__ x, const double* __restrict__ s, const double* __restrict__ z,
    const double* __restrict__ tau, const double* __restrict__ kappa,
    const double* __restrict__ z_x,            // direct-x dual; nullptr when total_xn == 0
    int64_t n, int64_t m, int64_t total_xn, int64_t batchSize
) {
    // One block per batch, threads cover elements within that batch
    int64_t b = blockIdx.x;
    if (b >= batchSize) return;

    // x + s + z + tau + kappa + z_x
    int64_t per_batch = n + m + m + 2 + total_xn;
    for (int64_t i = threadIdx.x; i < per_batch; i += blockDim.x) {
        double val;
        if (i < n) {
            val = x[b * n + i];
        } else if (i < n + m) {
            val = s[b * m + (i - n)];
        } else if (i < n + m + m) {
            val = z[b * m + (i - n - m)];
        } else if (i == n + m + m) {
            val = tau[b];
        } else if (i == n + m + m + 1) {
            val = kappa[b];
        } else {
            val = z_x[b * total_xn + (i - n - m - m - 2)];
        }

        if (isnan(val)) {
            atomicMax(&has_nan[b], 1);
            return;  // early exit — one NaN is enough
        }
    }
}

/**
 * @brief Kernel 2: Per-batch conditional copy to snapshot buffers.
 *
 * For each batch, copies only if has_nan[b] == 0 (i.e., all variables valid).
 * Same stream ordering guarantees has_nan was written by kernel 1.
 */
__global__ void yolo_conditional_copy_impl(
    const int32_t* __restrict__ has_nan,
    const double* __restrict__ x_src, const double* __restrict__ s_src, const double* __restrict__ z_src,
    const double* __restrict__ tau_src, const double* __restrict__ kappa_src,
    const double* __restrict__ z_x_src,        // direct-x dual; nullptr when total_xn == 0
    double* __restrict__ x_dst, double* __restrict__ s_dst, double* __restrict__ z_dst,
    double* __restrict__ tau_dst, double* __restrict__ kappa_dst,
    double* __restrict__ z_x_dst,              // nullptr when total_xn == 0
    int64_t n, int64_t m, int64_t total_xn, int64_t batchSize
) {
    int64_t b = blockIdx.x;
    if (b >= batchSize) return;

    // Skip this batch if it has NaN
    if (has_nan[b]) return;

    int64_t max_dim = (n > m) ? n : m;
    if (total_xn > max_dim) max_dim = total_xn;
    for (int64_t i = threadIdx.x; i < max_dim; i += blockDim.x) {
        if (i < n) {
            x_dst[b * n + i] = x_src[b * n + i];
        }
        if (i < m) {
            s_dst[b * m + i] = s_src[b * m + i];
            z_dst[b * m + i] = z_src[b * m + i];
        }
        if (i < total_xn) {
            z_x_dst[b * total_xn + i] = z_x_src[b * total_xn + i];
        }
        if (i == 0) {
            tau_dst[b] = tau_src[b];
            kappa_dst[b] = kappa_src[b];
        }
    }
}

void yolo_snapshot_if_valid_kernel(
    const double* x, const double* s, const double* z,
    const double* tau, const double* kappa,
    const double* z_x,
    double* x_dst, double* s_dst, double* z_dst,
    double* tau_dst, double* kappa_dst,
    double* z_x_dst,
    int32_t* d_has_nan,
    int64_t n, int64_t m, int64_t total_xn, int64_t batchSize,
    cudaStream_t stream
) {
    // Step 1: Reset per-batch NaN flags
    cudaMemsetAsync(d_has_nan, 0, sizeof(int32_t) * batchSize, stream);

    int threadsPerBlock = 256;

    // Step 2: Check for NaN per batch (one block per batch). z_x is folded
    // into the per-batch sweep so a NaN in the direct-x dual blocks the
    // snapshot exactly as a NaN in x/s/z would.
    MOREAU_KERNEL_LAUNCH(yolo_check_nan_impl, batchSize, threadsPerBlock, 0, stream,
        d_has_nan, x, s, z, tau, kappa, z_x, n, m, total_xn, batchSize);

    // Step 3: Conditionally copy per batch (one block per batch)
    MOREAU_KERNEL_LAUNCH(yolo_conditional_copy_impl, batchSize, threadsPerBlock, 0, stream,
        d_has_nan,
        x, s, z, tau, kappa, z_x,
        x_dst, s_dst, z_dst, tau_dst, kappa_dst, z_x_dst,
        n, m, total_xn, batchSize);
}

} // namespace moreau
