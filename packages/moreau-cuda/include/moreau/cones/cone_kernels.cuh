/**
 * @file cone_kernels.cuh
 * @brief CUDA kernel declarations for cone operations
 *
 * This header defines GPU kernels for projection onto convex cones
 * and computation of cone-specific scaling matrices. Per-kind scaling
 * goes through `update_scaling_all_cones` (fused) plus
 * `update_large_soc_scaling` for the dim > SOC_PARALLEL_THRESHOLD suffix.
 */

#pragma once

#include <cuda_runtime.h>

namespace moreau {

/**
 * @brief Update scaling for high-dimensional SOC cones (block-per-cone)
 *
 * Complements update_scaling_all_cones for SOC cones with
 * dim > SOC_PARALLEL_THRESHOLD. Launches one block per (batch, large_cone)
 * and uses a shared-memory block reduction to compute the tail norms in
 * parallel across threads, rather than having a single thread serially loop
 * over dim entries. All tail writes (w, u, v, λ, Hs diagonal) are also
 * parallelized across the block.
 *
 * Cones are sorted by ascending dim in Cones::initialize(), so the large
 * cones always occupy the contiguous suffix [numSmallSoc, numSocCones).
 * scaling_success[batch] may be set to 0 on degenerate s/z/w (matches the
 * composite kernel's behavior).
 */
void update_large_soc_scaling(
    double* soc_u,
    double* soc_v,
    double* soc_d,
    double* soc_w,
    double* soc_eta,
    double* soc_lambda,
    double* soc_Hs,
    const double* s,
    const double* z,
    int64_t offset_soc,
    int64_t m_total,
    int64_t numSocCones,
    int64_t numSmallSoc,
    int64_t numLargeSoc,
    int64_t batchSize,
    int32_t* scaling_success,
    const int64_t* d_soc_dims,
    const int64_t* d_soc_offsets,
    const int64_t* d_soc_Hs_offsets,
    const int64_t* d_soc_sz_offsets,
    int64_t totalSocDim,
    int64_t totalSocHsEntries,
    cudaStream_t stream = 0
);

/**
 * @brief Fused scaling update for all cone types in one kernel
 *
 * Combines nonneg, SOC, exp, and power cone scaling into a single kernel launch,
 * including the populate_soc_Hs operation for SOC cones.
 * This reduces kernel launch overhead significantly.
 *
 * @return true if all scaling succeeded, false if any SOC cone hit a degenerate state
 */
bool update_scaling_all_cones(
    // Nonneg outputs
    double* nonneg_w,
    double* nonneg_lambda,
    // SOC outputs
    double* soc_u,
    double* soc_v,
    double* soc_d,
    double* soc_w,
    double* soc_eta,
    double* soc_lambda,
    double* soc_Hs,
    // Exp outputs
    double* exp_grad,
    double* exp_H_dual,
    double* exp_Hs,
    double* exp_z,
    // Power outputs
    double* power_grad,
    double* power_H_dual,
    double* power_Hs,
    double* power_z,
    // Inputs
    const double* s,
    const double* z,
    const double* mu,
    const double* d_powerAlphas,
    // Cone structure
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m_total,
    int64_t batchSize,
    int scaling_strategy,
    cudaStream_t stream = 0,
    const int64_t* d_soc_dims = nullptr,
    const int64_t* d_soc_offsets = nullptr,
    const int64_t* d_soc_Hs_offsets = nullptr,
    int64_t totalSocDim = 0,
    int64_t totalSocHsEntries = 0,
    int32_t* d_scaling_success = nullptr,
    const int64_t* d_soc_sz_offsets = nullptr,
    volatile int32_t* h_scaling_success_pinned = nullptr,
    // GenPowerCone outputs
    double* genpow_grad = nullptr,
    double* genpow_z_out = nullptr,
    double* genpow_Hs = nullptr,
    double* genpow_p = nullptr,
    double* genpow_q = nullptr,
    double* genpow_r = nullptr,
    double* genpow_d1 = nullptr,
    double* genpow_d2 = nullptr,
    // GenPowerCone inputs
    const double* d_genPowerAlphas = nullptr,
    const int64_t* d_genPowerDim1s = nullptr,
    const int64_t* d_genPowerDim2s = nullptr,
    const int64_t* d_genPowerOffsets = nullptr,
    const int64_t* d_genPowerAlphaOffsets = nullptr,
    const int64_t* d_genPowerHsOffsets = nullptr,
    const int64_t* d_genPowerSzOffsets = nullptr,
    int64_t numGenPowerCones = 0,
    int64_t totalGenPowerDim = 0,
    int64_t totalGenPowerAlphas = 0,
    int64_t totalGenPowerGradEntries = 0,
    int64_t totalGenPowerHsEntries = 0,
    // Number of SOC cones with dim > SOC_PARALLEL_THRESHOLD; these are
    // handled by a separate block-per-cone kernel launched from within
    // update_scaling_all_cones.
    int64_t numLargeSoc = 0,
    // Same idea for GenPowerCones with total dim > GENPOW_PARALLEL_THRESHOLD.
    int64_t numLargeGenPow = 0
);

/**
 * @brief Update scaling for high-dimensional GenPowerCones (block-per-cone).
 *
 * Complements update_scaling_all_cones for GenPowerCones with total dim
 * > GENPOW_PARALLEL_THRESHOLD. Launches one block per (batch, large_cone)
 * and uses shared-memory block reductions for log_phi (over dim1) and
 * ||w||² (over dim2). Tail writes for grad/p/q/r/d1/z_out/Hs-diagonal are
 * parallelized across the block.
 *
 * Early exits (dual-infeasible z_i<=0, or zeta<=0) set scaling_success[batch]
 * to 0 and leave output arrays untouched, matching the composite kernel.
 */
void update_large_genpow_scaling(
    double* genpow_grad,
    double* genpow_z_out,
    double* genpow_Hs,
    double* genpow_p,
    double* genpow_q,
    double* genpow_r,
    double* genpow_d1,
    double* genpow_d2,
    const double* z,
    const double* mu,
    const double* d_genPowerAlphas,
    const int64_t* d_genPowerDim1s,
    const int64_t* d_genPowerDim2s,
    const int64_t* d_genPowerOffsets,
    const int64_t* d_genPowerAlphaOffsets,
    const int64_t* d_genPowerHsOffsets,
    const int64_t* d_genPowerSzOffsets,
    int64_t offset_z_cones,
    int64_t m_total,
    int64_t numGenPowerCones,
    int64_t numSmallGenPow,
    int64_t numLargeGenPow,
    int64_t batchSize,
    int64_t totalGenPowerDim,
    int64_t totalGenPowerAlphas,
    int64_t totalGenPowerHsEntries,
    int32_t* scaling_success,
    cudaStream_t stream = 0
);

// Forward declaration for MarginResults (defined in cones.hpp)
struct MarginResults;

/**
 * @brief Compute margins per-batch (no cross-batch reduction)
 *
 * Unlike compute_margins_kernel, this version returns per-batch margins
 * without reducing across batches. This ensures batch isolation for
 * correct gradient computation.
 *
 * @param z Variable vector to check [batchSize * m_total]
 * @param d_min_margin_out Output: per-batch minimum margin [batchSize]
 * @param d_pos_margin_out Output: per-batch positive margin sum [batchSize]
 * @param d_batch_results Pre-allocated buffer for intermediate results [batchSize]
 * @param numZeroCones Number of zero cone elements
 * @param numNonnegCones Number of nonnegative cone elements
 * @param numSocCones Number of SOC cones (each size 3)
 * @param numExpCones Number of exp cones (each size 3)
 * @param numPowerCones Number of power cones (each size 3)
 * @param m_total Total constraints per batch
 * @param batchSize Number of problems
 * @param stream CUDA stream
 */
void compute_margins_batched_kernel(
    const double* z,
    double* d_min_margin_out,
    double* d_pos_margin_out,
    MarginResults* d_batch_results,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m_total,
    int64_t batchSize,
    cudaStream_t stream,
    const int64_t* d_soc_dims = nullptr,
    const int64_t* d_soc_offsets = nullptr,
    int64_t totalSocDim = 0,
    const int64_t* d_soc_sz_offsets = nullptr
);

/**
 * @brief Apply per-batch scaled unit shift to move variables into cone interior
 *
 * Like scaled_unit_shift_kernel but with per-batch alpha values.
 * This ensures batch isolation for correct gradient computation.
 *
 * @param z Variable vector to modify [batchSize * m_total]
 * @param alpha Per-batch shift amounts [batchSize]
 * @param is_primal_cone True for primal cone (s), false for dual cone (z)
 * @param numZeroCones Number of zero cone elements
 * @param numNonnegCones Number of nonnegative cone elements
 * @param numSocCones Number of SOC cones (each size 3)
 * @param numExpCones Number of exp cones (each size 3)
 * @param numPowerCones Number of power cones (each size 3)
 * @param m_total Total constraints per batch
 * @param batchSize Number of problems
 * @param stream CUDA stream
 */
void scaled_unit_shift_batched_kernel(
    double* z,
    const double* alpha,
    bool is_primal_cone,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m_total,
    int64_t batchSize,
    cudaStream_t stream,
    const int64_t* d_soc_offsets = nullptr,
    int64_t totalSocDim = 0,
    const int64_t* d_soc_sz_offsets = nullptr
);

/**
 * @brief Fused margins + alpha + shift (3→1 kernel per vector)
 *
 * Combines compute_margins_batched + compute_init_shift_alpha + scaled_unit_shift_batched
 * into a single kernel launch.
 *
 * @param z Vector to shift into cone interior [batchSize * m_total]
 * @param is_primal_cone Whether this is the primal cone (zero cone zeroed if true)
 * @param degree Total cone degree for alpha computation
 * @param numZeroCones, numNonnegCones, numSocCones, numExpCones, numPowerCones Cone counts
 * @param m_total Total constraints
 * @param batchSize Number of problems
 * @param stream CUDA stream
 */
void fused_margins_and_shift_kernel(
    double* z,
    bool is_primal_cone,
    double degree,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m_total,
    int64_t batchSize,
    cudaStream_t stream,
    const int64_t* d_soc_dims = nullptr,
    const int64_t* d_soc_offsets = nullptr,
    int64_t totalSocDim = 0,
    const int64_t* d_soc_sz_offsets = nullptr
);

/**
 * @brief Fused smoothing for all cone types (warm start projection)
 *
 * Projects z onto the mu-central path using cone-specific smoothing formulas.
 * After smoothing, s is recovered as s = z - work (where work = z_orig - s_orig).
 *
 * @param z Variable to smooth (modified in-place) [batchSize * m_total]
 * @param work Preserved quantity work = z - s [batchSize * m_total]
 * @param mu Per-batch warmness parameter [batchSize]
 * @param d_powerAlphas Power cone alpha parameters [numPowerCones]
 * @param numZeroCones Number of zero cone elements
 * @param numNonnegCones Number of nonnegative cone elements
 * @param numSocCones Number of SOC cones (each size 3)
 * @param numExpCones Number of exp cones (each size 3)
 * @param numPowerCones Number of power cones (each size 3)
 * @param m_total Total constraints per batch
 * @param batchSize Number of problems
 * @param stream CUDA stream
 */
void smoothing_all_cones(
    double* z,
    const double* work,
    const double* mu,
    const double* d_powerAlphas,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numSocCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t m_total,
    int64_t batchSize,
    cudaStream_t stream = 0,
    const int64_t* d_soc_dims = nullptr,
    const int64_t* d_soc_offsets = nullptr,
    int64_t totalSocDim = 0,
    const int64_t* d_soc_sz_offsets = nullptr,
    // GenPowerCone params
    int64_t numGenPowerCones = 0,
    const double* d_genPowerAlphas = nullptr,
    const int64_t* d_genPowerDim1s = nullptr,
    const int64_t* d_genPowerDim2s = nullptr,
    const int64_t* d_genPowerOffsets = nullptr,
    const int64_t* d_genPowerAlphaOffsets = nullptr,
    const int64_t* d_genPowerSzOffsets = nullptr,
    int64_t totalGenPowerDim = 0,
    // GenPowerCone workspace
    double* smooth_zlocal = nullptr,
    double* smooth_wlocal = nullptr,
    double* smooth_res = nullptr,
    double* smooth_delta = nullptr,
    double* smooth_hmat = nullptr,
    double* smooth_lmat = nullptr,
    const int64_t* d_genPowerDimSqOffsets = nullptr,
    int64_t totalGenPowerDimSq = 0,
    double* genpow_grad_buf = nullptr,
    double* genpow_d1_buf = nullptr,
    double* genpow_p_buf = nullptr,
    double* genpow_q_buf = nullptr,
    double* genpow_r_buf = nullptr,
    int64_t totalGenPowerAlphas = 0,
    int64_t totalGenPowerDim2 = 0
);

} // namespace moreau
