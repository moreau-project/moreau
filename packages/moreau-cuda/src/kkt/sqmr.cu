/**
 * @file sqmr.cu
 * @brief Generic batched preconditioned SQMR Krylov core.
 *
 * Workspace + QMR update kernels + batched dot. The caller drives the iteration
 * (matvec + preconditioner). Ported from `ptn/sqmr-refinement`, stripped of the
 * cuDSS-specific KKT refinement driver and symmetric-CSR SpMV.
 */

#include "moreau/kkt/sqmr.cuh"
#include <cmath>
#include <new>
#include <stdexcept>

namespace moreau {

// ============================================================================
// Workspace
// ============================================================================

SqmrWorkspace::SqmrWorkspace(int64_t N, int64_t batchSize)
    : N_(N), batchSize_(batchSize)
{
    const size_t vec_size = static_cast<size_t>(N) * static_cast<size_t>(batchSize);
    const size_t scalar_size = static_cast<size_t>(batchSize);
    // 6 vectors + 5 scalar buffers + 2 QMR state scalars
    const size_t total = 6 * vec_size + 7 * scalar_size;

    double* ptr = nullptr;
    auto e = cudaMalloc(&ptr, sizeof(double) * total);
    if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));

    cudaMemset(ptr, 0, sizeof(double) * total);
    memory_.reset(ptr);

    size_t offset = 0;
    r_            = ptr + offset; offset += vec_size;
    z_            = ptr + offset; offset += vec_size;
    p_            = ptr + offset; offset += vec_size;
    w_            = ptr + offset; offset += vec_size;
    d_            = ptr + offset; offset += vec_size;
    precond_rhs_  = ptr + offset; offset += vec_size;
    sigma_buf_    = ptr + offset; offset += scalar_size;
    alpha_buf_    = ptr + offset; offset += scalar_size;
    rnorm_sq_buf_ = ptr + offset; offset += scalar_size;
    rho_buf_      = ptr + offset; offset += scalar_size;
    rho_new_buf_  = ptr + offset; offset += scalar_size;
    scalars_      = ptr + offset; offset += 2 * scalar_size;
}

// ============================================================================
// Batched standard dot product (cuBLAS)
// ============================================================================

void dot_batched_cublas(
    cublasHandle_t handle,
    int64_t N, int64_t batchSize,
    const double* x, const double* y,
    double* result,
    cudaStream_t stream)
{
    if (N == 0) {
        cudaMemsetAsync(result, 0, sizeof(double) * batchSize, stream);
        return;
    }

    cublasSetStream(handle, stream);

    const double alpha = 1.0;
    const double beta = 0.0;

    // C = xᵀy per batch: M=1, N=1, K=vecLen; column vectors with stride N.
    cublasDgemmStridedBatched(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        1, 1, static_cast<int>(N),
        &alpha,
        x, static_cast<int>(N), N,
        y, static_cast<int>(N), N,
        &beta,
        result, 1, 1,
        static_cast<int>(batchSize));
}

// ============================================================================
// SQMR residual update: r -= α*w where α = ρ/σ
// ============================================================================

__global__ void sqmr_r_update_kernel(
    int64_t N, int64_t batchSize,
    double* __restrict__ r,
    const double* __restrict__ w,
    const double* __restrict__ rho_buf,
    const double* __restrict__ sigma_buf,
    double* __restrict__ alpha_buf)
{
    const int64_t b = blockIdx.y;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (b >= batchSize) return;

    const double sigma = sigma_buf[b];
    const double alpha = (fabs(sigma) > 1e-300) ? rho_buf[b] / sigma : 0.0;

    if (i < N) {
        r[b * N + i] -= alpha * w[b * N + i];
    }
    if (i == 0) {
        alpha_buf[b] = alpha;
    }
}

void sqmr_r_update(
    int64_t N, int64_t batchSize,
    double* r, const double* w,
    const double* rho_buf, const double* sigma_buf,
    double* alpha_buf,
    cudaStream_t stream)
{
    if (N == 0) return;
    const int blockSize = 256;
    dim3 grid(static_cast<unsigned>((N + blockSize - 1) / blockSize),
              static_cast<unsigned>(batchSize));
    sqmr_r_update_kernel<<<grid, blockSize, 0, stream>>>(
        N, batchSize, r, w, rho_buf, sigma_buf, alpha_buf);
}

// ============================================================================
// SQMR QMR smoothing + direction update + prepare next iteration
// ============================================================================

constexpr int SCALAR_TAU        = 0;
constexpr int SCALAR_THETA_PREV = 1;

__global__ void sqmr_update_kernel(
    int64_t N, int64_t batchSize,
    double* __restrict__ x,
    double* __restrict__ d,
    double* __restrict__ p,
    const double* __restrict__ z,
    const double* __restrict__ alpha_buf,
    const double* __restrict__ rnorm_sq_buf,
    double* __restrict__ rho_buf,
    const double* __restrict__ rho_new_buf,
    double* __restrict__ scalars)
{
    const int64_t b = blockIdx.y;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (b >= batchSize) return;

    const double tau = scalars[SCALAR_TAU * batchSize + b];
    const double theta_prev = scalars[SCALAR_THETA_PREV * batchSize + b];
    const double alpha = alpha_buf[b];
    const double r_norm = sqrt(rnorm_sq_buf[b]);
    const double rho_val = rho_buf[b];
    const double rho_new = rho_new_buf[b];

    // QMR smoothing
    const double theta = (tau > 1e-300) ? r_norm / tau : 0.0;
    const double c = 1.0 / sqrt(1.0 + theta * theta);
    const double tau_new = tau * theta * c;
    const double eta = c * c * alpha;
    const double tpc_sq = (theta_prev * c) * (theta_prev * c);

    const double beta = rho_val != 0.0 ? rho_new / rho_val : 0.0;

    if (i < N) {
        const int64_t idx = b * N + i;
        const double p_i = p[idx];

        const double d_new = eta * p_i + tpc_sq * d[idx];
        x[idx] += d_new;
        d[idx] = d_new;

        p[idx] = z[idx] + beta * p_i;
    }

    if (i == 0) {
        scalars[SCALAR_TAU * batchSize + b] = tau_new;
        scalars[SCALAR_THETA_PREV * batchSize + b] = theta;
        rho_buf[b] = rho_new;
    }
}

void sqmr_update(
    int64_t N, int64_t batchSize,
    double* x, double* d, double* p, const double* z,
    const double* alpha_buf, const double* rnorm_sq_buf,
    double* rho_buf, const double* rho_new_buf,
    double* scalars,
    cudaStream_t stream)
{
    if (N == 0) return;
    const int blockSize = 256;
    dim3 grid(static_cast<unsigned>((N + blockSize - 1) / blockSize),
              static_cast<unsigned>(batchSize));
    sqmr_update_kernel<<<grid, blockSize, 0, stream>>>(
        N, batchSize, x, d, p, z, alpha_buf, rnorm_sq_buf,
        rho_buf, rho_new_buf, scalars);
}

// ============================================================================
// Raw waxpby: z = a*x + b*y
// ============================================================================

__global__ void waxpby_raw_kernel(
    double* __restrict__ z,
    double a, const double* __restrict__ x,
    double b, const double* __restrict__ y,
    int64_t total_size)
{
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < total_size) {
        z[i] = a * x[i] + b * y[i];
    }
}

void waxpby_raw(
    double* z,
    double a, const double* x,
    double b, const double* y,
    int64_t total_size,
    cudaStream_t stream)
{
    if (total_size == 0) return;
    const int blockSize = 256;
    const int grid = static_cast<int>((total_size + blockSize - 1) / blockSize);
    waxpby_raw_kernel<<<grid, blockSize, 0, stream>>>(z, a, x, b, y, total_size);
}

// ============================================================================
// Scalar initialization
// ============================================================================

__global__ void sqmr_init_scalars_kernel(
    int64_t batchSize,
    const double* __restrict__ rnorm_sq_buf,
    double* __restrict__ scalars)
{
    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= batchSize) return;

    scalars[0 * batchSize + b] = sqrt(rnorm_sq_buf[b]);  // tau = ‖r‖₂
    scalars[1 * batchSize + b] = 0.0;                     // theta_prev = 0
}

void sqmr_init_scalars(
    int64_t batchSize,
    const double* rnorm_sq_buf,
    double* scalars,
    cudaStream_t stream)
{
    if (batchSize == 0) return;
    const int blockSize = 256;
    const int grid = static_cast<int>((batchSize + blockSize - 1) / blockSize);
    sqmr_init_scalars_kernel<<<grid, blockSize, 0, stream>>>(
        batchSize, rnorm_sq_buf, scalars);
}

} // namespace moreau
