/**
 * @file smoothed_kernels.cuh
 * @brief Smoothed cone derivative kernels for smoothed differentiation
 *
 * Computes H = (I + μ·∇²φ*(z))^{-1} for each cone type.
 * Supports zero, nonneg, and SOC cones.
 */

#pragma once

#include <cuda_runtime.h>
#include "moreau/vector/vector.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/diff/diff.hpp"

namespace moreau {

/**
 * @brief Compute smoothed cone derivatives for all cones
 *
 * For zero cones: H = I (handled by updateJ, not here)
 * For nonneg cones: H[i] = z[i]² / (z[i]² + μ)
 * For SOC cones: two-step Sherman-Morrison (dense dim<=4, sparse dim>4)
 *
 * @param z Dual variables [batchSize][m]
 * @param s Slack variables [batchSize][m]
 * @param mu Smoothing parameter per batch [batchSize][1]
 * @param derivs Output cone derivatives (pre-allocated)
 * @param cones Cone structure
 * @param stream CUDA stream
 */
void compute_smoothed_cone_derivative(
    const BatchedVector& z,
    const BatchedVector& s,
    const BatchedVector& mu,
    ConeDerivatives& derivs,
    const Cones& cones,
    cudaStream_t stream = 0
);

} // namespace moreau
