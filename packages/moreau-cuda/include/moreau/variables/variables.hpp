/**
 * @file variables.hpp
 * @brief Solver variables for interior-point method
 *
 * This module defines the variable structure for the primal-dual
 * interior-point solver with homogenization.
 */

#pragma once

#include "moreau/vector/vector.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>

namespace moreau {

// Forward declarations
struct Residuals;
struct Cones;
enum class ScalingStrategy;

/**
 * @brief Variables for standard-form conic solver
 *
 * Contains primal variables (x), slack variables (s), dual variables (z),
 * and homogenization scalars (τ, κ) for the interior-point method.
 */
struct Variables {
    // Primal, slack, and dual variables
    BatchedVector x;     // Scaled primal variables [batchSize][n]
    BatchedVector s;     // Slack variables [batchSize][m]
    BatchedVector z;     // Scaled dual variables [batchSize][m]

    // Direct-x cone dual variables (stacked across all x-cones).
    // Length = totalXConeNumel. For runs with no direct-x cones, this
    // stays at placeholder size (1, batchSize) and all x-cone
    // operations are no-ops. CPU counterpart: DefaultVariables.z_x.
    BatchedVector z_x;   // [batchSize][totalXConeNumel]

    // Homogenization scalars (size 1 per batch)
    BatchedVector τ;     // Homogenization scalar τ [batchSize][1]
    BatchedVector κ;     // Homogenization scalar κ [batchSize][1]

    // Work vectors for combined_step_rhs
    BatchedVector one_minus_sigma;  // workspace for (1-σ) (1)
    BatchedVector scaled_affine_z;  // workspace for scaled affine z (m)
    BatchedVector sigma_mu_vec;     // workspace for σμ (1)

    // Work vectors for barrier function
    BatchedVector cur_s;            // workspace for s + α*ds (m)
    BatchedVector cur_z;            // workspace for z + α*dz (m)
    BatchedVector sz_dot;           // workspace for s'z dot product (1)

    /**
     * @brief Construct variables for given problem dimensions.
     *
     * @param n Number of primal variables
     * @param m Number of constraints
     * @param batchSize Number of problems to solve in parallel
     * @param totalXConeNumel Sum of direct-x cone dim sizes; 0 when
     *                       cones.dir_cones is empty (placeholder z_x).
     */
    Variables(int64_t n, int64_t m, int64_t batchSize,
              int64_t totalXConeNumel = 0)
        : x(n, batchSize),
          s(m, batchSize),
          z(m, batchSize),
          z_x(totalXConeNumel > 0 ? totalXConeNumel : 1, batchSize),
          τ(1, batchSize),
          κ(1, batchSize),
          one_minus_sigma(1, batchSize),
          scaled_affine_z(m, batchSize),
          sigma_mu_vec(1, batchSize),
          cur_s(m, batchSize),
          cur_z(m, batchSize),
          sz_dot(1, batchSize),
          total_xcone_numel_(totalXConeNumel)
    {}

    /// Total direct-x cone dimension this Variables was sized for.
    /// Returns 0 when no x-cones are present (z_x is placeholder).
    [[nodiscard]] int64_t totalXConeNumel() const noexcept {
        return total_xcone_numel_;
    }

    // No copying (large device buffers)
    Variables(const Variables&) = delete;
    Variables& operator=(const Variables&) = delete;

    // Moves ok
    Variables(Variables&&) noexcept = default;
    Variables& operator=(Variables&&) noexcept = default;

    ~Variables() = default;

    /**
     * @brief Calculate total memory usage in bytes
     */
    [[nodiscard]] size_t memoryUsage() const noexcept {
        return x.memoryUsage() + s.memoryUsage() + z.memoryUsage() +
               z_x.memoryUsage() +
               τ.memoryUsage() + κ.memoryUsage() +
               one_minus_sigma.memoryUsage() + scaled_affine_z.memoryUsage() +
               sigma_mu_vec.memoryUsage() +
               cur_s.memoryUsage() + cur_z.memoryUsage() +
               sz_dot.memoryUsage();
    }

    /**
     * @brief Initialize variables to default starting values
     * @param stream CUDA stream for async operations
     */
    void initialize(cudaStream_t stream = 0) {
        x.setToConstant(0.0, stream);
        s.setToConstant(1.0, stream);  // Start with slack in interior
        z.setToConstant(0.0, stream);
        // Direct-x dual: start at 0 when present (symmetric-cone
        // default from the CPU path). Placeholder copies are harmless.
        if (total_xcone_numel_ > 0) z_x.setToConstant(0.0, stream);
        τ.setToConstant(1.0, stream);  // Start with τ = 1
        κ.setToConstant(1.0, stream);  // Start with κ = 1
    }

    /**
     * @brief Calculate complementarity measure μ
     *
     * Computes μ = (s'z + τ*κ) / (degree + 1)
     * where degree is the total number of cone constraints.
     *
     * @param mu Output: complementarity measure [batchSize]
     * @param residuals Residuals containing dot_sz
     * @param cones Cone structure for computing degree
     * @param stream CUDA stream for async operations
     */
    void calc_mu(BatchedVector& mu, const Residuals& residuals, const Cones& cones,
                 cudaStream_t stream = 0);

    /**
     * @brief Update cone scaling matrices
     *
     * Updates the scaling for each cone type based on current s, z, and μ values.
     *
     * @param cones Cone structure to update
     * @param μ Complementarity measure [batchSize]
     * @param scaling Scaling strategy (PrimalDual or Dual)
     * @param stream CUDA stream for async operations
     * @return true if scaling succeeded, false if scaling failed
     */
    /// Scale cones and (when `kkt_solver` is non-null) fuse the direct-x
    /// Hs / expansion scatter directly into KKT.values — eliminating the
    /// per-iter `refresh_xcone_hs` round trip. Pass nullptr to use the
    /// classic two-step path (scaling, then external refresh).
    /// `pd_enabled_per_batch`: optional device array (length=batchSize, int8).
    /// 1 = PrimalDual scaling for that batch, 0 = Dual. When non-null, the
    /// direct-x GenPower PD-axis kernel honors per-batch state. When null,
    /// `scaling` applies uniformly to all batches (legacy behaviour).
    bool scale_cones(Cones& cones, const BatchedVector& μ, ScalingStrategy scaling,
                     cudaStream_t stream = 0,
                     class KKTSolver* kkt_solver = nullptr,
                     const int8_t* pd_enabled_per_batch = nullptr);

    /**
     * @brief Compute affine step RHS
     *
     * Computes the right-hand side for the affine step direction:
     * - x = rx (primal residual)
     * - z = rz (dual residual)
     * - s = affine_ds (cone-dependent)
     * - τ = rτ (gap residual)
     * - κ = τ * κ (from current variables)
     *
     * @param residuals Current residuals
     * @param variables Current variables
     * @param cones Cone structure
     * @param stream CUDA stream for async operations
     */
    void affine_step_rhs(const Residuals& residuals, const Variables& variables, Cones& cones, cudaStream_t stream = 0);

    /**
     * @brief Compute combined step RHS
     *
     * Computes the right-hand side for the combined (corrector) step using
     * Mehrotra's predictor-corrector method:
     * - x = (1-σ) * rx
     * - z = (1-σ) * rz
     * - s = combined_ds_shift (from affine step and centering)
     * - τ = (1-σ) * rτ
     * - κ = -σμ + m * Δτ_aff * Δκ_aff + τ * κ
     *
     * @param residuals Current residuals
     * @param variables Current variables
     * @param cones Cone structure
     * @param affine_step Affine step direction (for Mehrotra correction)
     * @param sigma Centering parameter [batchSize][1] (device pointer)
     * @param mu Complementarity measure [batchSize][1] (device pointer)
     * @param m Mehrotra correction factor [batchSize] (1.0 normally, alpha_aff on first iter)
     * @param stream CUDA stream for async operations
     */
    void combined_step_rhs(
        const Residuals& residuals,
        const Variables& variables,
        Cones& cones,
        Variables& affine_step,
        const BatchedVector& sigma,
        const BatchedVector& mu,
        const BatchedVector& m,
        cudaStream_t stream = 0);

    /**
     * @brief Update variables with step: variables += alpha * step
     *
     * Updates all variable components: x, s, z, τ, κ
     *
     * @param step Step direction
     * @param alpha Step length [batchSize]
     * @param stream CUDA stream for async operations
     */
    void add_step(const Variables& step, const BatchedVector& alpha, cudaStream_t stream = 0);

    /**
     * @brief Calculate maximum step length to stay within cones
     *
     * Computes the maximum α such that:
     * - current + α * step stays within cones
     * - τ + α * step.τ > 0
     * - κ + α * step.κ > 0
     *
     * For combined steps, applies max_step_fraction to reduce step size.
     *
     * @param step Step direction
     * @param cones Cone structure
     * @param max_step_fraction Maximum step fraction (typically 0.99)
     * @param is_combined_step True for combined step, false for affine step
     * @param alpha Output: maximum step length [batchSize][1]
     * @param work1 Work vector [batchSize][1]
     * @param work2 Work vector [batchSize][1]
     * @param stream CUDA stream for async operations
     */
    void calc_step_length(
        const Variables& step,
        Cones& cones,
        double max_step_fraction,
        bool is_combined_step,
        BatchedVector& alpha,
        BatchedVector& work1,
        BatchedVector& work2,
        double backtrack_step = 0.8,
        double min_step_length = 1e-4,
        cudaStream_t stream = 0);

    /**
     * @brief Compute barrier function at variables + α * step
     *
     * Computes the barrier function:
     * barrier = (degree+1)*log(μ) - log(τ) - log(κ) + cone_barriers
     * where μ = (s'z + τ*κ) / (degree + 1)
     *
     * @param step Step direction
     * @param alpha Step length
     * @param cones Cone structure
     * @param stream CUDA stream for async operations
     * @return Barrier value
     */
    double barrier(
        const Variables& step,
        double alpha,
        Cones& cones,
        cudaStream_t stream = 0);

private:
    int64_t total_xcone_numel_ = 0;
};

} // namespace moreau
