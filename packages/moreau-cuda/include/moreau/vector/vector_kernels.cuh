// vector_kernels.cuh
#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdint>
#include <vector>

namespace moreau {

// Forward declaration
struct BatchedVector;

void setConstantDevice(
    double* data,
    double value,
    int64_t size,
    cudaStream_t stream = 0
);

/**
 * @brief Batch set constant - sets multiple arrays to the same value in one kernel launch
 * @param ptrs Vector of device pointers to arrays
 * @param sizes Vector of sizes for each array
 * @param value The constant value to set
 * @param stream CUDA stream
 */
void batchSetConstant(
    const std::vector<double*>& ptrs,
    const std::vector<int64_t>& sizes,
    double value,
    cudaStream_t stream = 0
);

/**
 * @brief Compute reciprocal: out = 1.0 / in
 * Replaces setToConstant(1.0) + elementwise_div pattern
 */
void reciprocal(
    double* out,
    const double* in,
    int64_t size,
    cudaStream_t stream = 0
);

void scale_by_scalar(
    BatchedVector& d_array,
    double scale,
    cudaStream_t stream = 0
);

void compute_mean(
    const BatchedVector& d_array,
    BatchedVector& d_mean_result,
    cudaStream_t stream = 0
);

void compute_inf_norm(
    const BatchedVector& d_array,
    BatchedVector& d_norm_result,
    cudaStream_t stream = 0
);

// y = a*x + b*y (batched AXPBY)
void axpby(
    BatchedVector& y,
    double a,
    const BatchedVector& x,
    double b,
    cudaStream_t stream = 0
);

// z = a*x + b*y (batched WAXPBY - write into z)
void waxpby(
    BatchedVector& z,
    double a,
    const BatchedVector& x,
    double b,
    const BatchedVector& y,
    cudaStream_t stream = 0
);

// Dot product: result[i] = sum_j(x[j,i] * y[j,i]) for each batch i
// result is a scalar per batch (batchSize scalars)
void dot_batched(
    BatchedVector& result,
    const BatchedVector& x,
    const BatchedVector& y,
    cudaStream_t stream = 0
);

// Quadratic form: result[i] = x[:,i]^T * P * y[:,i] for each batch i
// P is symmetric CSR matrix (upper triangle only)
// result is a scalar per batch (batchSize scalars)
// temp, buffer, and buffer_size are pre-allocated workspaces to avoid allocation in hot path
void quad_form_batched(
    BatchedVector& result,
    const void* P_csr,  // CSR matrix handle
    const BatchedVector& x,
    const BatchedVector& y,
    BatchedVector& temp,  // workspace for P*y (size n)
    void*& buffer,  // cuSPARSE SpMV buffer (will be allocated if needed)
    size_t& buffer_size,  // size of buffer
    void*& aligned_buffer_y,  // aligned buffer for y (allocated if batchSize>1 && n is odd)
    void*& aligned_buffer_temp,  // aligned buffer for temp (allocated if batchSize>1 && n is odd)
    size_t& aligned_buffer_size,  // size of aligned buffers
    cusparseHandle_t cusparse_handle,  // cuSPARSE handle from Solver
    cudaStream_t stream = 0
);

// Quadratic form for symmetric matrices: result[i] = x[:,i]^T * P * y[:,i] for each batch i
// P is symmetric CSR matrix stored as FULL MATRIX (both upper and lower triangle entries)
// Uses regular SpMV kernel to compute P*y, then dot product with x
// result is a scalar per batch (batchSize scalars)
// temp is a pre-allocated workspace vector (size n per batch)
void quad_form_symmetric_batched(
    BatchedVector& result,
    const void* P_csr,  // CSR matrix handle (full symmetric matrix)
    const BatchedVector& x,
    const BatchedVector& y,
    BatchedVector& temp,  // workspace for P*y (size n)
    cudaStream_t stream = 0
);

// Element-wise division: z = x / y (batched)
void elementwise_div(
    BatchedVector& z,
    const BatchedVector& x,
    const BatchedVector& y,
    cudaStream_t stream = 0
);

// Safe element-wise division: z = x / max(|y|, eps) * sign(y)
// Prevents NaN from near-zero denominators
void safe_elementwise_div(
    BatchedVector& z,
    const BatchedVector& x,
    const BatchedVector& y,
    double eps,
    cudaStream_t stream = 0
);

// Element-wise multiplication: z = x * y (batched)
void elementwise_mul(
    BatchedVector& z,
    const BatchedVector& x,
    const BatchedVector& y,
    cudaStream_t stream = 0
);

// Multiply vector by scalar per batch: y[i] = x[i] * scalar_per_batch[batch_idx]
void mul_per_batch(
    BatchedVector& y,
    const BatchedVector& x,
    const BatchedVector& scalars,
    cudaStream_t stream = 0
);

// Scalar addition to vector: y[i] = x[i] + scalar_per_batch[batch_idx]
void add_scalar_per_batch(
    BatchedVector& y,
    const BatchedVector& x,
    const BatchedVector& scalars,
    cudaStream_t stream = 0
);

// Multiply vector by scalar per batch: y[i] = scalar_per_batch[batch_idx] * x[i]
void scale_per_batch(
    BatchedVector& y,
    const BatchedVector& scalars,
    const BatchedVector& x,
    cudaStream_t stream = 0
);

// Divide vector by scalar per batch: y[i] = x[i] / scalar_per_batch[batch_idx]
void div_per_batch(
    BatchedVector& y,
    const BatchedVector& x,
    const BatchedVector& scalars,
    cudaStream_t stream = 0
);

// Batched AXPY: y += alpha[batch] * x for each batch
// y[batch*n + i] += alpha[batch] * x[batch*n + i]
void axpy_batched(
    BatchedVector& y,
    const BatchedVector& alpha,
    const BatchedVector& x,
    cudaStream_t stream = 0
);

// Element-wise minimum with scalar per batch: y[i] = min(x[i], scalar_per_batch[batch_idx])
void min_scalar_per_batch(
    BatchedVector& y,
    const BatchedVector& x,
    const BatchedVector& scalars,
    cudaStream_t stream = 0
);

// Scalar minus vector: y = scalar - x (batched)
// Computes y[i] = scalar - x[i] for each element in each batch
void scalar_minus_vector(
    BatchedVector& y,
    double scalar,
    const BatchedVector& x,
    cudaStream_t stream = 0
);

// Compute adaptive centering target: mu_step[b] = min(mu[b] * factor, target)
void compute_mu_step(
    BatchedVector& mu_step,
    const BatchedVector& mu,
    double factor,
    double target,
    cudaStream_t stream = 0
);

// Sanitize alpha: replace NaN/negative with 0.0
void sanitize_alpha(
    BatchedVector& alpha,
    cudaStream_t stream = 0
);

// Zero the step direction for batches that contain NaN.
// Prevents NaN propagation through add_step (since 0*NaN = NaN in IEEE 754).
void zero_nan_step(
    BatchedVector& step_x,
    BatchedVector& step_z,
    BatchedVector& step_s,
    int64_t n,
    int64_t m,
    cudaStream_t stream = 0
);

// ============================================================================
// FUSED KERNELS - reduce kernel launch overhead
// ============================================================================

/**
 * @brief Fused dot product + accumulate: dest += alpha * (x · y)
 *
 * Replaces the common pattern:
 *   dot_batched(temp, x, y, stream);
 *   axpby(dest, alpha, temp, 1.0, stream);
 *
 * Saves one kernel launch and one intermediate storage.
 *
 * @param dest Output scalar per batch (accumulated into)
 * @param alpha Scalar multiplier for dot product
 * @param x First input vector
 * @param y Second input vector
 * @param stream CUDA stream
 */
void dot_add(
    BatchedVector& dest,
    double alpha,
    const BatchedVector& x,
    const BatchedVector& y,
    cudaStream_t stream = 0
);

/**
 * @brief Fused z = x1 + tau * x2 where tau is per-batch scalar
 *
 * Replaces the common pattern:
 *   waxpby(z, 1.0, x1, 0.0, x1, stream);  // z = x1 (copy)
 *   scale_per_batch(temp, tau, x2, stream);  // temp = tau * x2
 *   axpby(z, 1.0, temp, 1.0, stream);  // z += temp
 *
 * Saves two kernel launches.
 *
 * @param z Output vector
 * @param x1 First input vector
 * @param tau Per-batch scalar multiplier (batchSize scalars)
 * @param x2 Second input vector
 * @param stream CUDA stream
 */
void axpby_scaled(
    BatchedVector& z,
    const BatchedVector& x1,
    const BatchedVector& tau,
    const BatchedVector& x2,
    cudaStream_t stream = 0
);

/**
 * @brief Fused 2-dot: dest += alpha0*(x0·y0) + alpha1*(x1·y1)
 *
 * @param dest Output scalar per batch (accumulated into)
 * @param alpha0, alpha1 Scalar multipliers
 * @param x0, y0 First vector pair
 * @param x1, y1 Second vector pair
 * @param stream CUDA stream
 */
void multi_dot_add_2(
    BatchedVector& dest,
    double alpha0, const BatchedVector& x0, const BatchedVector& y0,
    double alpha1, const BatchedVector& x1, const BatchedVector& y1,
    cudaStream_t stream = 0
);

/**
 * @brief Fused 4-dot: dest += alpha0*(x0·y0) + alpha1*(x1·y1) + alpha2*(x2·y2) + alpha3*(x3·y3)
 *
 * @param dest Output scalar per batch (accumulated into)
 * @param alpha0..3 Scalar multipliers
 * @param x0..3, y0..3 Vector pairs
 * @param stream CUDA stream
 */
void multi_dot_add_4(
    BatchedVector& dest,
    double alpha0, const BatchedVector& x0, const BatchedVector& y0,
    double alpha1, const BatchedVector& x1, const BatchedVector& y1,
    double alpha2, const BatchedVector& x2, const BatchedVector& y2,
    double alpha3, const BatchedVector& x3, const BatchedVector& y3,
    cudaStream_t stream = 0
);

/**
 * @brief Fused waxpby + scale: z = a*x + tau*y where tau is per-batch scalar
 *
 * Replaces the common pattern:
 *   waxpby(workz, 1.0, const_term, -1.0, rhs, stream);
 *   scale_per_batch(workz, tau, workz, stream);
 *
 * @param z Output vector
 * @param a Scalar multiplier for x
 * @param x First input vector
 * @param tau Per-batch scalar multiplier for y (batchSize scalars)
 * @param y Second input vector
 * @param stream CUDA stream
 */
void waxpby_scaled(
    BatchedVector& z,
    double a,
    const BatchedVector& x,
    const BatchedVector& tau,
    const BatchedVector& y,
    cudaStream_t stream = 0
);

/**
 * @brief Fused triple subtraction: z0=x0-y0, z1=x1-y1, z2=x2-y2
 *
 * Replaces the common pattern:
 *   waxpby(z0, 1.0, x0, -1.0, y0, stream);
 *   waxpby(z1, 1.0, x1, -1.0, y1, stream);
 *   waxpby(z2, 1.0, x2, -1.0, y2, stream);
 *
 * Saves two kernel launches. Vectors can have different sizes.
 */
void triple_sub(
    BatchedVector& z0, const BatchedVector& x0, const BatchedVector& y0,
    BatchedVector& z1, const BatchedVector& x1, const BatchedVector& y1,
    BatchedVector& z2, const BatchedVector& x2, const BatchedVector& y2,
    cudaStream_t stream = 0
);

/**
 * @brief Fused dest += a*x + b*y for per-batch scalars
 *
 * Replaces:
 *   axpby(dest, a, x, 1.0, stream);
 *   axpby(dest, b, y, 1.0, stream);
 *
 * Saves one kernel launch.
 */
void axpby2_scalar(
    BatchedVector& dest,
    double a, const BatchedVector& x,
    double b, const BatchedVector& y,
    cudaStream_t stream = 0
);

/**
 * @brief Fused step update: x+=alpha*dx, s+=alpha*ds, z+=alpha*dz, tau+=alpha*dtau, kappa+=alpha*dkappa
 *
 * Replaces 3x axpy_batched + update_tau_kappa_kernel (4 launches -> 1).
 */
void apply_step(
    BatchedVector& x, const BatchedVector& dx,
    BatchedVector& s, const BatchedVector& ds,
    BatchedVector& z, const BatchedVector& dz,
    BatchedVector& tau, const BatchedVector& dtau,
    BatchedVector& kappa, const BatchedVector& dkappa,
    const BatchedVector& alpha,
    cudaStream_t stream = 0
);

/**
 * @brief z_x += α · dz_x for a direct-x cone dual vector (per-batch α).
 *
 * Companion to apply_step; called only when totalXConeNumel > 0 so the
 * existing slack-only path is unaffected.
 */
void apply_step_z_x(
    BatchedVector& z_x, const BatchedVector& dz_x,
    const BatchedVector& alpha,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0
);

/**
 * @brief Fused dest = a - b / c for per-batch scalars
 *
 * Replaces:
 *   memcpy(dest, a);
 *   elementwise_div(temp, b, c);
 *   axpby(dest, -1.0, temp, 1.0);
 *
 * Saves two kernel launches + one memcpy.
 */
void sub_div_scalar(
    BatchedVector& dest,
    const BatchedVector& a,
    const BatchedVector& b,
    const BatchedVector& c,
    cudaStream_t stream = 0
);

/**
 * @brief Fused warm-start scaling: x*=dinv, z*=einv*c, s*=e  (4→1 kernel)
 *
 * Replaces:
 *   elementwise_mul(x, x, dinv)
 *   elementwise_mul(z, z, einv)
 *   mul_per_batch(z, z, c)
 *   elementwise_mul(s, s, e)
 */
void fused_warm_start_scaling(
    BatchedVector& x,
    BatchedVector& z,
    BatchedVector& s,
    const BatchedVector& dinv,
    const BatchedVector& einv,
    const BatchedVector& e,
    const BatchedVector& c,
    cudaStream_t stream = 0
);

/**
 * @brief Build per-batch normalization scale: τ for optimal batches, κ for infeasible batches
 *
 * Used in solution post-processing to match CPU solver behavior where
 * infeasibility certificates are normalized by κ instead of τ.
 */
void build_normalization_scale(
    BatchedVector& scale,
    const BatchedVector& tau,
    const BatchedVector& kappa,
    const int32_t* status,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Set objective values to NaN for infeasible batches
 *
 * For infeasible problems, the objective values are meaningless.
 * Matches CPU solver behavior of setting obj_val/obj_val_dual to NaN.
 */
void set_infeasible_obj_nan(
    BatchedVector& obj_val,
    BatchedVector& obj_val_dual,
    const int32_t* status,
    int64_t batchSize,
    cudaStream_t stream = 0
);

} // namespace moreau
