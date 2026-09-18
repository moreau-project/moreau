/**
 * @file warm_start_kernels.cu
 * @brief CUDA kernels for warm start utility operations
 *
 * Provides the warmness-mu computation kernel that combines
 * primal/dual residuals and gap measures into a single per-batch mu value.
 */

#include "moreau/vector/vector.hpp"
#include "moreau/profiling/profiler.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace moreau {

// Keep smoothing away from a numerically singular cone boundary. This is an
// interior perturbation floor, not a convergence tolerance.
constexpr double WARM_START_MU_FLOOR = 1e-6;

__global__ void copy_direct_duals_masked_kernel(
    double* dst, const double* src, const int32_t* mask, int64_t xn, int64_t batch_size
) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < xn * batch_size && mask[i / xn]) dst[i] = src[i];
}

void copy_direct_duals_masked(double* dst, const double* src, const int32_t* mask,
                             int64_t xn, int64_t batch_size, cudaStream_t stream) {
    if (xn == 0 || batch_size == 0) return;
    const int64_t blocks = (xn * batch_size + 255) / 256;
    MOREAU_KERNEL_LAUNCH(copy_direct_duals_masked_kernel, blocks, 256, 0, stream,
                        dst, src, mask, xn, batch_size);
}

__global__ void interiorize_direct_warm_start_kernel(
    double* x, double* z_x, const double* mu,
    const double* res_primal, const double* res_dual,
    const double* gap_abs, const double* gap_rel,
    double tol_feas, double tol_gap_abs, double tol_gap_rel,
    const int64_t* kinds, const int64_t* dims, const int64_t* offsets,
    const int64_t* indices, const int64_t* pow_idx, const double* pow_alpha,
    const int64_t* gp_idx, const int64_t* gp_dim1,
    const int64_t* gp_alpha_offsets, const double* gp_alphas,
    int64_t n, int64_t xn, bool has_slack_cones
) {
    const int64_t batch = blockIdx.x;
    const int64_t cone = blockIdx.y;
    const int64_t kind = kinds[cone];
    // Preserve an already converged point for iteration zero. A boundary point
    // below the smoothing floor still needs a shift if it misses the tolerances.
    const bool optimal = mu[batch] <= 1.0
        && res_primal[batch] < tol_feas && res_dual[batch] < tol_feas
        && (gap_abs[batch] < tol_gap_abs || gap_rel[batch] < tol_gap_rel);
    if (!has_slack_cones && optimal) return;
    const int64_t offset = offsets[cone];
    for (int64_t j = threadIdx.x; j < dims[cone]; j += blockDim.x) {
        double unit = 0.0;
        if (kind == 0) {
            unit = 1.0;
        } else if (kind == 1) {
            unit = j == 0 ? 1.0 : 0.0;
        } else if (kind == 2) {
            for (int64_t col = 0, diag = 0; diag < dims[cone]; ++col, diag += col + 1)
                if (j == diag) unit = 1.0;
        } else if (kind == 3) {
            unit = j == 0 ? -1.051383945322714
                 : j == 1 ? 0.556409619469370 : 1.258967884768947;
        } else if (kind == 4 && j < 2) {
            const double alpha = pow_alpha[pow_idx[cone]];
            unit = sqrt(j == 0 ? 1.0 + alpha : 2.0 - alpha);
        } else if (kind == 5) {
            const int64_t gp = gp_idx[cone];
            if (j < gp_dim1[gp])
                unit = sqrt(1.0 + gp_alphas[gp_alpha_offsets[gp] + j]);
        }
        const double shift = mu[batch] * unit;
        x[batch * n + indices[offset + j]] += shift;
        z_x[batch * xn + offset + j] += shift;
    }
}

void interiorize_direct_warm_start(
    double* x, double* z_x, const double* mu,
    const double* res_primal, const double* res_dual,
    const double* gap_abs, const double* gap_rel,
    double tol_feas, double tol_gap_abs, double tol_gap_rel,
    const int64_t* kinds, const int64_t* dims, const int64_t* offsets,
    const int64_t* indices, const int64_t* pow_idx, const double* pow_alpha,
    const int64_t* gp_idx, const int64_t* gp_dim1,
    const int64_t* gp_alpha_offsets, const double* gp_alphas,
    int64_t n, int64_t xn, int64_t num_cones, int64_t batch_size, bool has_slack_cones,
    cudaStream_t stream
) {
    if (num_cones == 0 || batch_size == 0) return;
    const dim3 grid(batch_size, num_cones);
    MOREAU_KERNEL_LAUNCH(interiorize_direct_warm_start_kernel, grid, 128, 0, stream,
        x, z_x, mu, res_primal, res_dual, gap_abs, gap_rel,
        tol_feas, tol_gap_abs, tol_gap_rel,
        kinds, dims, offsets, indices, pow_idx, pow_alpha,
        gp_idx, gp_dim1, gp_alpha_offsets, gp_alphas, n, xn, has_slack_cones);
}

// ============================================================================
// Warmness mu computation kernel
// ============================================================================

/**
 * @brief Compute warmness parameter mu from residuals and gap
 *
 * mu[i] = max(res_primal[i], res_dual[i], min(gap_abs[i], gap_rel[i]))
 * with a positive floor to keep smoothing away from a singular boundary.
 *
 * One thread per batch problem.
 */
__global__ void compute_warmness_mu_kernel(
    double* __restrict__ mu_out,              // [batchSize] — output per-batch mu
    const double* __restrict__ res_primal,    // [batchSize]
    const double* __restrict__ res_dual,      // [batchSize]
    const double* __restrict__ gap_abs,       // [batchSize]
    const double* __restrict__ gap_rel,       // [batchSize]
    int64_t batchSize
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    double rp = res_primal[idx];
    double rd = res_dual[idx];
    double ga = gap_abs[idx];
    double gr = gap_rel[idx];

    // mu = max(res_primal, res_dual, min(gap_abs, gap_rel))
    double gap_min = fmin(ga, gr);
    double mu = fmax(fmax(rp, rd), gap_min);

    mu = fmax(mu, WARM_START_MU_FLOOR);

    mu_out[idx] = mu;
}

/**
 * @brief Host wrapper for warmness mu computation
 */
void compute_warmness_mu(
    double* mu_out,
    const double* res_primal,
    const double* res_dual,
    const double* gap_abs,
    const double* gap_rel,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int blocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(compute_warmness_mu_kernel, blocks, threadsPerBlock, 0, stream,
        mu_out, res_primal, res_dual, gap_abs, gap_rel, batchSize
    );
}

} // namespace moreau
