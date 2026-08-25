/**
 * @file kernels.cuh
 * @brief CUDA kernel declarations for variable operations
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace moreau {

/**
 * @brief Compute complementarity measure μ = (dot_sz + τ*κ) / (degree + 1)
 *
 * @param mu Output: complementarity measure [batchSize]
 * @param dot_sz s'z inner product [batchSize]
 * @param tau τ values [batchSize]
 * @param kappa κ values [batchSize]
 * @param degree Total number of cone constraints
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void calc_mu_kernel(
    double* mu,
    const double* dot_sz,
    const double* tau,
    const double* kappa,
    int64_t degree,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Element-wise multiplication of two vectors: result = a * b
 *
 * @param result Output: result[i] = a[i] * b[i] [batchSize]
 * @param a Input vector a [batchSize]
 * @param b Input vector b [batchSize]
 * @param batchSize Number of elements
 * @param stream CUDA stream
 */
void multiply_vectors_kernel(
    double* result,
    const double* a,
    const double* b,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Calculate step length for scalar variable
 *
 * Updates alpha: if step < 0, then alpha = min(alpha, -current / step)
 *
 * @param alpha Input/Output: step length [batchSize]
 * @param current Current scalar value [batchSize]
 * @param step Step direction [batchSize]
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void calc_step_length_scalar_kernel(
    double* alpha,
    const double* current,
    const double* step,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Fused kernel: initialize alpha to 1.0 and compute step length for tau and kappa
 *
 * Computes: alpha = min(1.0, -tau/step_tau if step_tau<0, -kappa/step_kappa if step_kappa<0)
 * This replaces 3 kernel launches: setToConstant(1.0) + calc_step_length_scalar(tau) + calc_step_length_scalar(kappa)
 *
 * @param alpha Output: step length [batchSize], initialized and constrained
 * @param tau Current tau values [batchSize]
 * @param step_tau Step tau values [batchSize]
 * @param kappa Current kappa values [batchSize]
 * @param step_kappa Step kappa values [batchSize]
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void calc_step_length_tau_kappa_init_kernel(
    double* alpha,
    const double* tau,
    const double* step_tau,
    const double* kappa,
    const double* step_kappa,
    int64_t batchSize,
    cudaStream_t stream = 0,
    double* alpha_z = nullptr,
    double* alpha_s = nullptr
);

/**
 * @brief Element-wise minimum: result = min(a, b)
 *
 * @param result Output: result[i] = min(a[i], b[i]) [batchSize]
 * @param a Input vector a [batchSize]
 * @param b Input vector b [batchSize]
 * @param batchSize Number of elements
 * @param stream CUDA stream
 */
void elementwise_min_kernel(
    double* result,
    const double* a,
    const double* b,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Scale each element by a scalar: data[i] = scale * data[i]
 *
 * @param data Input/Output vector [batchSize]
 * @param scale Scalar multiplier
 * @param batchSize Number of elements
 * @param stream CUDA stream
 */
void scale_by_scalar_kernel(
    double* data,
    double scale,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Compute kappa RHS with Mehrotra correction: κ = -σ*μ + m * step_τ * step_κ + var_τ * var_κ
 *
 * Formula: kappa_rhs[i] = -sigma[i] * mu[i] + m[i] * step_tau[i] * step_kappa[i] + var_tau[i] * var_kappa[i]
 *
 * @param kappa_rhs Output: kappa RHS values [batchSize]
 * @param sigma Centering parameter [batchSize]
 * @param mu Complementarity measure [batchSize]
 * @param m Mehrotra correction factor [batchSize]
 * @param step_tau Affine step tau values [batchSize]
 * @param step_kappa Affine step kappa values [batchSize]
 * @param var_tau Current tau values [batchSize]
 * @param var_kappa Current kappa values [batchSize]
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
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
    cudaStream_t stream = 0
);

/**
 * @brief Scale vectors by batched scalar values: output[batch*n + i] = scalars[batch] * input[batch*n + i]
 *
 * Each batch is scaled by its corresponding scalar value.
 *
 * @param output Output: scaled vectors [n * batchSize]
 * @param input Input vectors [n * batchSize]
 * @param scalars Scalar values per batch [batchSize]
 * @param n Vector dimension
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void scale_vectors_by_batched_scalars_kernel(
    double* output,
    const double* input,
    const double* scalars,
    int64_t n,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Update tau and kappa: var_tau += alpha * step_tau, var_kappa += alpha * step_kappa
 *
 * Performs batched update: var[batch] = var[batch] + alpha[batch] * step[batch]
 *
 * @param var_tau Output: current tau values [batchSize]
 * @param var_kappa Output: current kappa values [batchSize]
 * @param step_tau Input: step tau values [batchSize]
 * @param step_kappa Input: step kappa values [batchSize]
 * @param alpha Input: step length [batchSize]
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void update_tau_kappa_kernel(
    double* var_tau,
    double* var_kappa,
    const double* step_tau,
    const double* step_kappa,
    const double* alpha,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Compute Δκ = -(rhs_κ + var_κ * lhs_τ) / var_τ
 *
 * Formula: lhs_kappa[i] = -(rhs_kappa[i] + var_kappa[i] * lhs_tau[i]) / var_tau[i]
 *
 * @param lhs_kappa Output: Δκ values [batchSize]
 * @param rhs_kappa Input: RHS κ values [batchSize]
 * @param var_kappa Input: current κ values [batchSize]
 * @param lhs_tau Input: Δτ values [batchSize]
 * @param var_tau Input: current τ values [batchSize]
 * @param batchSize Number of batches
 * @param stream CUDA stream
 */
void compute_delta_kappa_kernel(
    double* lhs_kappa,
    const double* rhs_kappa,
    const double* var_kappa,
    const double* lhs_tau,
    const double* var_tau,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Fused: alpha = min(alpha_z, alpha_s) * scale  (2→1 kernel)
 */
void fused_step_length_finalize_kernel(
    double* alpha,
    const double* alpha_z,
    const double* alpha_s,
    double scale,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Fused: compute (1-σ), scale rx by (1-σ), scale rτ by (1-σ) (3→1 kernel)
 */
void fused_scale_residuals_kernel(
    double* one_minus_sigma,
    double* x_out,
    double* tau_out,
    const double* sigma,
    const double* rx,
    const double* rtau,
    int64_t n,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Fused: compute kappa_rhs, scaled_affine_z=m*aff.z, sigma_mu=σ*μ (3→1 kernel)
 */
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
    cudaStream_t stream = 0
);

/**
 * @brief Fused affine step RHS: x=rx, z=rz, τ=rτ, κ=var_τ*var_κ (3 memcpys + 1 kernel → 1 kernel)
 */
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
    cudaStream_t stream = 0
);

/**
 * @brief Fused axpby + scale: s += z; z = oms * rz (2 → 1 kernel)
 */
void fused_axpby_and_scale_kernel(
    double* s,
    double* z,
    const double* rz,
    const double* oms,
    int64_t m,
    int64_t batchSize,
    cudaStream_t stream = 0
);

} // namespace moreau
