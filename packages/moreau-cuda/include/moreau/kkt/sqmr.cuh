/**
 * @file sqmr.cuh
 * @brief Generic batched preconditioned SQMR (Symmetric Quasi-Minimum Residual)
 *
 * Problem-agnostic Krylov core: workspace + the QMR update kernels + a batched
 * dot product. The caller supplies the matvec (symmetric, possibly indefinite)
 * and the preconditioner apply; SQMR handles indefiniteness via QMR smoothing
 * and uses the standard (unsigned) inner product, so the preconditioner may be
 * symmetric indefinite. Ported from the `ptn/sqmr-refinement` branch, stripped
 * of the cuDSS-specific KKT refinement driver.
 *
 * Reference: Freund & Nachtigal, "A new Krylov-subspace method for symmetric
 * indefinite linear systems" (1994).
 *
 * STATUS: currently UNWIRED (no caller). Kept available for robustifying an
 * indefinite inner solve. NOTE: this is plain SQMR without look-ahead, so it can
 * break down (σ = pᵀMp ≈ 0) on genuinely indefinite operators; add look-ahead or
 * an early-stop best-iterate guard before relying on it.
 */

#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "moreau/cuda/utils.hpp"

namespace moreau {

/**
 * @brief Workspace for batched preconditioned SQMR iterations.
 *
 * Preallocates all device vectors needed for the conjugate-direction
 * iteration and QMR smoothing, avoiding per-solve allocations.
 */
struct SqmrWorkspace {
    device_unique_ptr<double> memory_;

    // Vectors (each N*batchSize): r, z, p, w, d, precond_rhs
    double* r_;
    double* z_;
    double* p_;
    double* w_;
    double* d_;
    double* precond_rhs_;

    // Per-batch scalar buffers
    double* sigma_buf_;     // σ = pᵀw [batchSize]
    double* alpha_buf_;     // α = ρ/σ [batchSize]
    double* rnorm_sq_buf_;  // ‖r‖² [batchSize]
    double* rho_buf_;       // ρ = rᵀz [batchSize]
    double* rho_new_buf_;   // ρ_new [batchSize]

    // Per-batch QMR state: [tau, theta_prev] packed as 2*batchSize
    double* scalars_;

    int64_t N_;
    int64_t batchSize_;

    SqmrWorkspace(int64_t N, int64_t batchSize);

    SqmrWorkspace(const SqmrWorkspace&) = delete;
    SqmrWorkspace& operator=(const SqmrWorkspace&) = delete;
};

// Batched standard dot product: result[b] = x[:,b]ᵀ y[:,b] (batchSize scalars).
void dot_batched_cublas(
    cublasHandle_t handle,
    int64_t N, int64_t batchSize,
    const double* x, const double* y,
    double* result,
    cudaStream_t stream = 0);

// SQMR residual update: r -= (ρ/σ)*w, store α = ρ/σ per batch.
void sqmr_r_update(
    int64_t N, int64_t batchSize,
    double* r, const double* w,
    const double* rho_buf, const double* sigma_buf,
    double* alpha_buf,
    cudaStream_t stream = 0);

// SQMR QMR smoothing + direction update + prepare next iteration:
//   d = η*p + (θ_prev*c)²*d;  x += d;  p = z + β*p;  advance (τ, θ_prev, ρ).
void sqmr_update(
    int64_t N, int64_t batchSize,
    double* x, double* d, double* p, const double* z,
    const double* alpha_buf, const double* rnorm_sq_buf,
    double* rho_buf, const double* rho_new_buf,
    double* scalars,
    cudaStream_t stream = 0);

// Initialize SQMR scalar state: τ = √(rnorm_sq), θ_prev = 0.
void sqmr_init_scalars(
    int64_t batchSize,
    const double* rnorm_sq_buf,
    double* scalars,
    cudaStream_t stream = 0);

// Batched waxpby on raw pointers: z = a*x + b*y over total_size elements.
void waxpby_raw(
    double* z,
    double a, const double* x,
    double b, const double* y,
    int64_t total_size,
    cudaStream_t stream = 0);

} // namespace moreau
