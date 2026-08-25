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

// ============================================================================
// Warmness mu computation kernel
// ============================================================================

/**
 * @brief Compute warmness parameter mu from residuals and gap
 *
 * mu[i] = max(res_primal[i], res_dual[i], min(gap_abs[i], gap_rel[i]))
 * with a floor of 1e-6 to ensure mu is never too small.
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

    // Floor at 1e-6
    mu = fmax(mu, 1e-6);

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
