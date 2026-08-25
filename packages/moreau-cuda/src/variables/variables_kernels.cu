/**
 * @file kernels.cu
 * @brief CUDA kernel implementations for variable operations
 */

#include "moreau/variables/variables_kernels.cuh"
#include "moreau/cuda/utils.cuh"
#include "moreau/profiling/profiler.hpp"

namespace moreau {

__global__ void calc_mu_kernel_impl(
    double* __restrict__ mu,
    const double* __restrict__ dot_sz,
    const double* __restrict__ tau,
    const double* __restrict__ kappa,
    int64_t degree,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    double denom = static_cast<double>(degree + 1);
    mu[batch] = (dot_sz[batch] + tau[batch] * kappa[batch]) / denom;
}

void calc_mu_kernel(
    double* mu,
    const double* dot_sz,
    const double* tau,
    const double* kappa,
    int64_t degree,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(calc_mu_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        mu, dot_sz, tau, kappa, degree, batchSize
    );
}

__global__ void multiply_vectors_kernel_impl(
    double* __restrict__ result,
    const double* __restrict__ a,
    const double* __restrict__ b,
    int64_t batchSize
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    result[idx] = a[idx] * b[idx];
}

void multiply_vectors_kernel(
    double* result,
    const double* a,
    const double* b,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(multiply_vectors_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        result, a, b, batchSize
    );
}

__global__ void calc_step_length_scalar_kernel_impl(
    double* __restrict__ alpha,
    const double* __restrict__ current,
    const double* __restrict__ step,
    int64_t batchSize
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    if (step[idx] < 0.0) {
        double candidate = -current[idx] / step[idx];
        atomicMinDouble(&alpha[idx], candidate);
    }
}

void calc_step_length_scalar_kernel(
    double* alpha,
    const double* current,
    const double* step,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(calc_step_length_scalar_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        alpha, current, step, batchSize
    );
}

// Fused kernel: initializes alpha, alpha_z, alpha_s to 1.0 and computes step length constraints
// for both tau and kappa in a single kernel launch.
// Also initializes alpha_z and alpha_s to alpha (eliminating 2 D2D memcpys in Cones::step_length).
__global__ void calc_step_length_tau_kappa_init_kernel_impl(
    double* __restrict__ alpha,
    const double* __restrict__ tau,
    const double* __restrict__ step_tau,
    const double* __restrict__ kappa,
    const double* __restrict__ step_kappa,
    double* __restrict__ alpha_z,
    double* __restrict__ alpha_s,
    int64_t batchSize
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    // Start with max step length of 1.0
    double result = 1.0;

    // Constraint from tau: if step_tau < 0, then alpha <= -tau / step_tau
    if (step_tau[idx] < 0.0) {
        double candidate = -tau[idx] / step_tau[idx];
        result = fmin(result, candidate);
    }

    // Constraint from kappa: if step_kappa < 0, then alpha <= -kappa / step_kappa
    if (step_kappa[idx] < 0.0) {
        double candidate = -kappa[idx] / step_kappa[idx];
        result = fmin(result, candidate);
    }

    alpha[idx] = result;
    // Initialize alpha_z and alpha_s to same value (eliminates D2D memcpys)
    if (alpha_z) alpha_z[idx] = result;
    if (alpha_s) alpha_s[idx] = result;
}

void calc_step_length_tau_kappa_init_kernel(
    double* alpha,
    const double* tau,
    const double* step_tau,
    const double* kappa,
    const double* step_kappa,
    int64_t batchSize,
    cudaStream_t stream,
    double* alpha_z,
    double* alpha_s
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(calc_step_length_tau_kappa_init_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        alpha, tau, step_tau, kappa, step_kappa, alpha_z, alpha_s, batchSize
    );
}

__global__ void elementwise_min_kernel_impl(
    double* __restrict__ result,
    const double* __restrict__ a,
    const double* __restrict__ b,
    int64_t batchSize
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    result[idx] = fmin(a[idx], b[idx]);
}

void elementwise_min_kernel(
    double* result,
    const double* a,
    const double* b,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(elementwise_min_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        result, a, b, batchSize
    );
}

__global__ void scale_by_scalar_kernel_impl(
    double* __restrict__ data,
    double scale,
    int64_t batchSize
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    data[idx] *= scale;
}

void scale_by_scalar_kernel(
    double* data,
    double scale,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(scale_by_scalar_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        data, scale, batchSize
    );
}

__global__ void compute_kappa_rhs_kernel_impl(
    double* __restrict__ kappa_rhs,
    const double* __restrict__ sigma,
    const double* __restrict__ mu,
    const double* __restrict__ m,
    const double* __restrict__ step_tau,
    const double* __restrict__ step_kappa,
    const double* __restrict__ var_tau,
    const double* __restrict__ var_kappa,
    int64_t batchSize
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;

    // Formula: kappa_rhs = -sigma*mu + m * step_τ * step_κ + var_τ * var_κ
    double sigma_mu = sigma[idx] * mu[idx];
    double mehrotra_correction = m[idx] * step_tau[idx] * step_kappa[idx];
    double current_product = var_tau[idx] * var_kappa[idx];

    kappa_rhs[idx] = -sigma_mu + mehrotra_correction + current_product;
}

void compute_kappa_rhs_kernel(
    double* kappa_rhs,
    const double* sigma,
    const double* mu,
    const double* m,
    const double* step_tau,
    const double* step_kappa,
    const double* var_tau,
    const double* var_kappa,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(compute_kappa_rhs_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        kappa_rhs, sigma, mu, m, step_tau, step_kappa, var_tau, var_kappa, batchSize
    );
}

__global__ void scale_vectors_by_batched_scalars_kernel_impl(
    double* __restrict__ output,
    const double* __restrict__ input,
    const double* __restrict__ scalars,
    int64_t n,
    int64_t batchSize
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total_size = n * batchSize;

    if (idx >= total_size) return;

    // Determine which batch this element belongs to
    int64_t batch = idx / n;

    // Scale by the corresponding batch scalar
    output[idx] = scalars[batch] * input[idx];
}

void scale_vectors_by_batched_scalars_kernel(
    double* output,
    const double* input,
    const double* scalars,
    int64_t n,
    int64_t batchSize,
    cudaStream_t stream
) {
    // Early return for zero-size vectors to avoid CUDA error from numBlocks=0
    if (n == 0) return;

    int64_t total_size = n * batchSize;
    int threadsPerBlock = 256;
    int numBlocks = (total_size + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(scale_vectors_by_batched_scalars_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        output, input, scalars, n, batchSize
    );
}

__global__ void update_tau_kappa_kernel_impl(
    double* __restrict__ var_tau,
    double* __restrict__ var_kappa,
    const double* __restrict__ step_tau,
    const double* __restrict__ step_kappa,
    const double* __restrict__ alpha,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    // var_tau += alpha * step_tau
    var_tau[batch] += alpha[batch] * step_tau[batch];

    // var_kappa += alpha * step_kappa
    var_kappa[batch] += alpha[batch] * step_kappa[batch];
}

void update_tau_kappa_kernel(
    double* var_tau,
    double* var_kappa,
    const double* step_tau,
    const double* step_kappa,
    const double* alpha,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(update_tau_kappa_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        var_tau, var_kappa, step_tau, step_kappa, alpha, batchSize
    );
}

__global__ void compute_delta_kappa_kernel_impl(
    double* __restrict__ lhs_kappa,
    const double* __restrict__ rhs_kappa,
    const double* __restrict__ var_kappa,
    const double* __restrict__ lhs_tau,
    const double* __restrict__ var_tau,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch >= batchSize) return;

    // lhs_kappa = -(rhs_kappa + var_kappa * lhs_tau) / var_tau
    lhs_kappa[batch] = -(rhs_kappa[batch] + var_kappa[batch] * lhs_tau[batch]) / var_tau[batch];
}

void compute_delta_kappa_kernel(
    double* lhs_kappa,
    const double* rhs_kappa,
    const double* var_kappa,
    const double* lhs_tau,
    const double* var_tau,
    int64_t batchSize,
    cudaStream_t stream
) {
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;

    MOREAU_KERNEL_LAUNCH(compute_delta_kappa_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        lhs_kappa, rhs_kappa, var_kappa, lhs_tau, var_tau, batchSize
    );
}

// ============================================================================
// Fused step length finalize: alpha = min(alpha_z, alpha_s) * scale  (2→1)
// ============================================================================
__global__ void fused_step_length_finalize_kernel_impl(
    double* __restrict__ alpha,
    const double* __restrict__ alpha_z,
    const double* __restrict__ alpha_s,
    double scale,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize) return;
    alpha[idx] = fmin(alpha_z[idx], alpha_s[idx]) * scale;
}

void fused_step_length_finalize_kernel(
    double* alpha,
    const double* alpha_z,
    const double* alpha_s,
    double scale,
    int64_t batchSize,
    cudaStream_t stream)
{
    int threadsPerBlock = 256;
    int numBlocks = (batchSize + threadsPerBlock - 1) / threadsPerBlock;
    MOREAU_KERNEL_LAUNCH(fused_step_length_finalize_kernel_impl, numBlocks, threadsPerBlock, 0, stream,
        alpha, alpha_z, alpha_s, scale, batchSize);
}

// ============================================================================
// Fused scale residuals: one_minus_sigma=1-σ, x=(1-σ)*rx, τ=(1-σ)*rτ  (3→1)
// ============================================================================
__global__ void fused_scale_residuals_kernel_impl(
    double* __restrict__ one_minus_sigma,
    double* __restrict__ x_out,
    double* __restrict__ tau_out,
    const double* __restrict__ sigma,
    const double* __restrict__ rx,
    const double* __restrict__ rtau,
    int64_t n,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    if (batch >= batchSize) return;

    double oms = 1.0 - sigma[batch];

    // Thread 0 stores one_minus_sigma for this batch
    if (idx == 0) {
        one_minus_sigma[batch] = oms;
        tau_out[batch] = oms * rtau[batch];
    }

    if (idx < n) {
        x_out[batch * n + idx] = oms * rx[batch * n + idx];
    }
}

void fused_scale_residuals_kernel(
    double* one_minus_sigma,
    double* x_out,
    double* tau_out,
    const double* sigma,
    const double* rx,
    const double* rtau,
    int64_t n,
    int64_t batchSize,
    cudaStream_t stream)
{
    int64_t grid_x = (n > 0) ? n : 1;
    dim3 blk(256, 1, 1);
    dim3 grd((grid_x + blk.x - 1) / blk.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(fused_scale_residuals_kernel_impl, grd, blk, 0, stream,
        one_minus_sigma, x_out, tau_out, sigma, rx, rtau, n, batchSize);
}

// ============================================================================
// Fused scalar prep: kappa_rhs, scaled_affine_z=m*aff.z, sigma_mu=σ*μ  (3→1)
// ============================================================================
__global__ void fused_scalar_prep_kernel_impl(
    double* __restrict__ kappa_rhs,
    double* __restrict__ scaled_affine_z,
    double* __restrict__ sigma_mu,
    const double* __restrict__ sigma_in,
    const double* __restrict__ mu,
    const double* __restrict__ m_val,
    const double* __restrict__ affine_tau,
    const double* __restrict__ affine_kappa,
    const double* __restrict__ var_tau,
    const double* __restrict__ var_kappa,
    const double* __restrict__ affine_z,
    int64_t m,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    if (batch >= batchSize) return;

    if (idx == 0) {
        // Compute kappa_rhs = -σ*μ + m * step_τ * step_κ + var_τ * var_κ
        double sm = sigma_in[batch] * mu[batch];
        double mehrotra = m_val[batch] * affine_tau[batch] * affine_kappa[batch];
        double current = var_tau[batch] * var_kappa[batch];
        kappa_rhs[batch] = -sm + mehrotra + current;

        // Compute sigma_mu = σ * μ
        sigma_mu[batch] = sm;
    }

    if (idx < m) {
        // scaled_affine_z = m * affine_z
        scaled_affine_z[batch * m + idx] = m_val[batch] * affine_z[batch * m + idx];
    }
}

void fused_scalar_prep_kernel(
    double* kappa_rhs,
    double* scaled_affine_z,
    double* sigma_mu,
    const double* sigma,
    const double* mu,
    const double* m_val,
    const double* affine_tau,
    const double* affine_kappa,
    const double* var_tau,
    const double* var_kappa,
    const double* affine_z,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream)
{
    int64_t grid_x = (m > 0) ? m : 1;
    dim3 blk(256, 1, 1);
    dim3 grd((grid_x + blk.x - 1) / blk.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(fused_scalar_prep_kernel_impl, grd, blk, 0, stream,
        kappa_rhs, scaled_affine_z, sigma_mu,
        sigma, mu, m_val, affine_tau, affine_kappa,
        var_tau, var_kappa, affine_z, m, batchSize);
}

// ============================================================================
// Fused affine step RHS: x=rx, z=rz, τ=rτ, κ=var_τ*var_κ  (3 memcpys + 1 kernel → 1 kernel)
// ============================================================================
__global__ void fused_affine_step_rhs_kernel_impl(
    double* __restrict__ x_out,
    double* __restrict__ z_out,
    double* __restrict__ tau_out,
    double* __restrict__ kappa_out,
    const double* __restrict__ rx,
    const double* __restrict__ rz,
    const double* __restrict__ rtau,
    const double* __restrict__ var_tau,
    const double* __restrict__ var_kappa,
    int64_t n,
    int64_t m,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    if (batch >= batchSize) return;

    // Copy x = rx
    if (idx < n) {
        x_out[batch * n + idx] = rx[batch * n + idx];
    }

    // Copy z = rz
    if (idx < m) {
        z_out[batch * m + idx] = rz[batch * m + idx];
    }

    // Scalars: τ = rτ, κ = var_τ * var_κ (one thread per batch)
    if (idx == 0) {
        tau_out[batch] = rtau[batch];
        kappa_out[batch] = var_tau[batch] * var_kappa[batch];
    }
}

void fused_affine_step_rhs_kernel(
    double* x_out,
    double* z_out,
    double* tau_out,
    double* kappa_out,
    const double* rx,
    const double* rz,
    const double* rtau,
    const double* var_tau,
    const double* var_kappa,
    int64_t n,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream)
{
    int64_t max_dim = (n > m) ? n : m;
    if (max_dim == 0) max_dim = 1;  // at least 1 for scalar ops
    dim3 blk(256, 1, 1);
    dim3 grd((max_dim + blk.x - 1) / blk.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(fused_affine_step_rhs_kernel_impl, grd, blk, 0, stream,
        x_out, z_out, tau_out, kappa_out,
        rx, rz, rtau, var_tau, var_kappa,
        n, m, batchSize);
}

// ============================================================================
// Fused axpby + scale: s += z; z = oms * rz  (2 → 1 kernel)
// ============================================================================
__global__ void fused_axpby_and_scale_kernel_impl(
    double* __restrict__ s,
    double* __restrict__ z,
    const double* __restrict__ rz,
    const double* __restrict__ oms,
    int64_t m,
    int64_t batchSize)
{
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t batch = blockIdx.y;
    if (batch >= batchSize || idx >= m) return;

    int64_t gi = batch * m + idx;
    double z_val = z[gi];
    s[gi] += z_val;
    z[gi] = oms[batch] * rz[batch * m + idx];
}

void fused_axpby_and_scale_kernel(
    double* s,
    double* z,
    const double* rz,
    const double* oms,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream)
{
    if (m == 0) return;
    dim3 blk(256, 1, 1);
    dim3 grd((m + blk.x - 1) / blk.x, batchSize, 1);
    MOREAU_KERNEL_LAUNCH(fused_axpby_and_scale_kernel_impl, grd, blk, 0, stream,
        s, z, rz, oms, m, batchSize);
}

} // namespace moreau
