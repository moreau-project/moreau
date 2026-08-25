/**
 * @file psd_kernels.cuh
 * @brief CUDA kernel declarations for PSD (SDP) cone operations
 *
 * PSD cones use cuSOLVER for dense linear algebra (Cholesky, SVD, eigendecomp)
 * and custom kernels for svec packing/unpacking and symmetric Kronecker products.
 * These cannot be fused into the existing composite kernel due to cuSOLVER
 * library calls — they use separate kernel launches.
 */

#pragma once

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cstdint>

namespace moreau {

// Forward declaration
struct Cones;

/**
 * @brief Update PSD cone scaling: compute R, Rinv, lambda, Hs
 *
 * For each PSD cone per batch:
 * 1. svec_to_mat(s) → S, svec_to_mat(z) → Z
 * 2. Cholesky: S = L1·L1^T, Z = L2·L2^T
 * 3. SVD: L2^T·L1 = U·Σ·V^T → λ = σ
 * 4. R = L1·V·Λ^{-1/2}, Rinv = Λ^{-1/2}·U^T·L2^T
 * 5. Hs = skron(R·R^T)
 *
 * Reports per-batch numerical failure (non-finite Hs from a degenerate s/z) by
 * writing d_scaling_success[b] = 0 for the affected batch indices, mirroring the
 * per-batch flag the composite SOC/GenPow kernel sets. d_scaling_success must be a
 * length-batchSize array initialized to 1; pass nullptr to skip the check.
 */
void update_psd_scaling(
    Cones& cones,
    const double* s,
    const double* z,
    int64_t m_total,
    int32_t* d_scaling_success = nullptr,
    cudaStream_t stream = 0
);

/**
 * @brief Direct-x PSD NT scaling. See `update_psd_scaling` for the math;
 * this variant operates on direct-x PSD cones with the primal↔dual swap.
 *
 * @param x        Primal x [batchSize * n_primal]
 * @param z_x      Direct-x dual, flat over all xcones [batchSize * totalXConeNumel]
 * @param n_primal Length of x per batch element
 */
bool update_xcones_psd_scaling(
    Cones& cones,
    const double* x,
    const double* z_x,
    int64_t n_primal,
    cudaStream_t stream = 0
);

/**
 * @brief Direct-x PSD `affine_ds`: step_rhs.z_x[J] := svec(diag(λ²)).
 *
 * Mirrors the slack `psd_affine_ds_kernel` but writes into the direct-x
 * dual step buffer at `d_xcone_psd_in_full_offsets[psd_idx]`.
 */
void fill_step_rhs_zx_psd_affine(
    Cones& cones,
    double* step_rhs_z_x,             // [batchSize * totalXConeNumel]
    cudaStream_t stream = 0
);

/**
 * @brief Direct-x PSD combined-step shift contribution.
 *
 * Adds to step_rhs.z_x[J]:
 *   shift = svec((R·step_aff_x_mat·R^T) ∘_λ (Rinv·m·step_aff_z_x_mat·Rinv^T))
 *           − σμ·svec(I)
 * where ∘_λ is the symmetric Jordan product around λ. Mirrors slack
 * `psd_combined_ds_shift` with primal↔dual swap (R applies to step_x
 * gathered from the primal, Rinv to step_z_x flat).
 */
void add_combined_ds_shift_psd(
    Cones& cones,
    double* step_rhs_z_x,
    const double* step_aff_x,         // [batchSize * n_primal]
    const double* step_aff_z_x,       // [batchSize * totalXConeNumel]
    const double* sigma_mu,           // [batchSize]
    const double* mehrotra_m,         // [batchSize]
    int64_t n_primal,
    cudaStream_t stream = 0
);

/**
 * @brief Direct-x PSD step length reduction.
 *
 * For each PSD x-cone, compute the PSD-cone step length on (x[J], step.x[J])
 * (primal side → α_z) and (z_x, step.z_x) (dual side → α_s) via the standard
 * Mehrotra eigendecomp. Atomic-min reduces into the per-batch `alpha_s` /
 * `alpha_z` running minimums. Mirrors slack `psd_step_length`.
 */
void xcone_step_length_psd_reduce(
    Cones& cones,
    double* alpha_s,                  // [batchSize]
    double* alpha_z,                  // [batchSize]
    const double* var_x,              // [batchSize * n_primal]
    const double* var_z_x,            // [batchSize * totalXConeNumel]
    const double* step_x,             // [batchSize * n_primal]
    const double* step_z_x,           // [batchSize * totalXConeNumel]
    int64_t n_primal,
    double max_step,
    cudaStream_t stream = 0
);

/**
 * @brief Direct-x PSD combined-step KKT-RHS correction.
 *
 * Mirrors slack `psd_ds_from_dz_offset` and the SOC variant
 * `subtract_xcone_combined_from_workx_soc`. For each PSD x-cone:
 *   1. mat_D = svec_to_mat(rhs.z_x[J] flat)
 *   2. mat_D[i,j] *= 2 / (λ[i] + λ[j])     (λ-inv-circ)
 *   3. mat_out = R · mat_D · R^T
 *   4. workx[xc.indices[i]] -= mat_to_svec(mat_out)[i]   (scatter)
 */
void subtract_xcone_combined_from_workx_psd(
    Cones& cones,
    double* workx,                    // [batchSize * n_primal]
    const double* rhs_z_x,            // [batchSize * totalXConeNumel]
    int64_t n_primal,
    cudaStream_t stream = 0
);

/**
 * @brief Direct-x PSD Δz_x recovery (affine step).
 *
 *   dz_x[J flat] = -Hs · gather(lhs.x[J]) - var_z_x[J flat]
 *
 * where Hs is the dense svec_dim × svec_dim Hessian in xcone_Hs upper-tri
 * format. Mirrors `recover_dz_x_affine_soc` for PSD.
 */
void recover_dz_x_affine_psd(
    Cones& cones,
    double* dz_x,                     // [batchSize * totalXConeNumel]
    const double* lhs_x,              // [batchSize * n_primal] (Δx)
    const double* var_z_x,            // [batchSize * totalXConeNumel]
    int64_t n_primal,
    cudaStream_t stream = 0
);

/**
 * @brief Direct-x PSD Δz_x recovery (combined step).
 *
 *   dz_x[J flat] = -Hs · gather(lhs.x[J]) - c_J_combined
 *
 * where c_J_combined = svec(R · λ_inv_circ(mat(rhs.z_x[J])) · R^T) — same
 * formula as `subtract_xcone_combined_from_workx_psd` but emitted as svec
 * into the dual step buffer instead of scattered into workx.
 */
void recover_dz_x_combined_psd(
    Cones& cones,
    double* dz_x,                     // [batchSize * totalXConeNumel]
    const double* lhs_x,              // [batchSize * n_primal] (Δx)
    const double* rhs_z_x,            // [batchSize * totalXConeNumel]
    int64_t n_primal,
    cudaStream_t stream = 0
);

/**
 * @brief Shift each PSD x-cone's `x[J]` slice into the cone interior.
 *
 * After the initial KKT solve, `x[J]` may not be in the PSD cone (the seeded
 * Hs = I doesn't enforce the constraint). For each PSD x-cone:
 *   1. Gather `x[J]` into matrix form
 *   2. Eigendecompose to find min eigenvalue λ_min
 *   3. If λ_min ≤ 0: add (-λ_min + target)·I to the matrix
 *      Else if λ_min < target: add (target - λ_min)·I
 *   where target = max(1, pos_margin·0.1 / degree). The shift adds a
 *   constant α to each diagonal svec entry of `x[J]`.
 *
 * Mirrors CPU `_shift_single_cone_to_interior` for direct-x PSD cones.
 */
void shift_x_into_psd_interior(
    Cones& cones,
    double* x,                        // [batchSize * n_primal]
    int64_t n_primal,
    double total_degree,
    cudaStream_t stream = 0
);

/**
 * @brief Per-batch finite check over a [batchSize][batch_stride] array.
 * For any non-finite element in batch b, sets d_flags[b] to 0 (via atomicMin).
 * d_flags must be a length-batchSize array initialized to 1.
 */
void check_finite_kernel(const double* data, int64_t n, int64_t batch_stride,
                         int32_t* d_flags, cudaStream_t stream = 0);

/**
 * @brief Compute margins (min eigenvalue) for PSD cone variables
 *
 * PSD cone margin = min eigenvalue of mat(z) for each cone.
 * Uses cuSOLVER eigendecomposition.
 *
 * @param z Variables to check [batchSize * m_total]
 * @param psd_offset Offset of PSD cones in s/z vector
 * @param min_margin Output: minimum margin (atomicMin across all batches)
 * @param pos_margin Output: sum of positive margins
 */
void compute_psd_margins(
    const Cones& cones,
    const double* z,
    int64_t psd_offset,
    int64_t m_total,
    double* d_min_margin,
    double* d_pos_margin,
    cudaStream_t stream = 0
);

/**
 * @brief Apply scaled unit shift for PSD cones
 *
 * Shifts z by alpha * I (identity in svec form):
 * z[triangular_index(k)] += alpha for k = 0..n-1
 */
void psd_scaled_unit_shift(
    double* z,
    double alpha,
    int64_t psd_offset,
    int64_t m_total,
    const Cones& cones,
    cudaStream_t stream = 0
);

/**
 * @brief PSD cone step length computation
 *
 * Eigendecomp of scaled direction, step = -1/min(neg eigenvalues)
 */
void psd_step_length(
    Cones& cones,
    const double* dz,
    const double* ds,
    const double* z,
    const double* s,
    double* alpha_z,
    double* alpha_s,
    int64_t psd_offset,
    int64_t m_total,
    cudaStream_t stream = 0
);

/**
 * @brief PSD cone barrier: -log det(S) - log det(Z)
 */
void psd_barrier(
    const Cones& cones,
    const double* z,
    const double* dz,
    double alpha,
    int64_t psd_offset,
    int64_t m_total,
    double* barrier_out,
    cudaStream_t stream = 0
);

/**
 * @brief PSD cone smoothing for warm start
 *
 * Eigendecomp z = QΛQ^T, smooth: λ_i → (w_i + √(w_i²+4μ))/2, reconstruct
 */
void psd_smoothing(
    Cones& cones,
    double* z,
    const double* work,
    const double* mu,
    int64_t psd_offset,
    int64_t m_total,
    cudaStream_t stream = 0
);

// ============================================================================
// Kernel declarations (called from cones.cu)
// ============================================================================

/// Initialize PSD cone variables to svec(I)
__global__ void initPsdConesKernel(
    double* s, double* z, int64_t psd_offset,
    const int64_t* d_dims, const int64_t* d_sz_offsets,
    int64_t numPsdCones, int64_t m_total
);

/// Compute affine ds for PSD cones: ds = svec(diag(λ²))
__global__ void psd_affine_ds_kernel(
    double* ds, const double* psd_lambda,
    int64_t psd_offset,
    const int64_t* d_dims, const int64_t* d_sz_offsets,
    const int64_t* d_mat_offsets,
    int64_t numPsdCones, int64_t totalPsdSvecDim,
    int64_t totalPsdMatDim, int64_t m_total
);

/// Compute Δs_from_Δz_offset for PSD cones: out = W^T(λ \ ds)
void psd_ds_from_dz_offset(
    Cones& cones,
    double* out, const double* ds,
    int64_t psd_offset, int64_t m_total,
    cudaStream_t stream = 0
);

/// Multiply by PSD Hessian: y = Hs * x
__global__ void psd_mul_Hs_kernel(
    double* y, const double* x, const double* psd_Hs,
    int64_t psd_offset,
    const int64_t* d_dims, const int64_t* d_sz_offsets,
    const int64_t* d_Hs_offsets,
    int64_t numPsdCones, int64_t totalPsdHsEntries,
    int64_t m_total
);

/// Combined DS shift for PSD cones: shift = W^{-T}(step_s) ∘ W(step_z) - σμ·e
void psd_combined_ds_shift(
    Cones& cones,
    double* shift, double* step_z, double* step_s,
    const double* sigma_mu,
    int64_t psd_offset, int64_t m_total,
    cudaStream_t stream = 0
);

} // namespace moreau
