/**
 * @file kkt_kernels.cuh
 * @brief CUDA kernel declarations for KKT system operations
 *
 * This header defines GPU kernels for assembling and manipulating
 * KKT system matrices, including regularization and scaling.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace moreau {

// Forward declaration
struct CSR;

/**
 * @brief Populate destination CSR matrix values from source using index mapping
 * @param src Source CSR matrix
 * @param dst Destination CSR matrix
 * @param idx_map Device pointer to index mapping (src index -> dst index)
 */
void populate_values_via_map(
    CSR& src,
    CSR& dst,
    int64_t* idx_map,
    cudaStream_t stream = 0
);

/**
 * @brief Pack RHS for uniform batched KKT solve: [x_part; b_part] per batch
 */
void pack_const_rhs(
    const double* x_part,
    const double* b_part,
    double* rhs,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size,
    cudaStream_t stream = 0
);

/**
 * @brief Fused: negate q and pack with b into const RHS (eliminates separate axpby)
 *
 * Computes rhs = [-q; b] per batch in interleaved layout, replacing
 * the separate axpby(workx, -1, q, 0) + pack_const_rhs(workx, b, rhs) calls.
 */
void negate_and_pack_const_rhs_kernel(
    const double* q,
    const double* b,
    double* rhs,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size,
    cudaStream_t stream = 0
);

/**
 * @brief Unpack KKT solution in uniform batch layout into x2/z2 buffers
 */
void unpack_const_sol(
    const double* sol,
    double* x2,
    double* z2,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size,
    cudaStream_t stream = 0
);

/**
 * @brief Pack two RHS vectors into 2-column format for batched multi-RHS solve
 *
 * Layout: [batch0_col0(n+m), batch0_col1(n+m), batch1_col0(n+m), batch1_col1(n+m), ...]
 *
 * @param x_part0 First RHS x part (device) [n * batch_size]
 * @param z_part0 First RHS z part (device) [m * batch_size]
 * @param x_part1 Second RHS x part (device) [n * batch_size]
 * @param z_part1 Second RHS z part (device) [m * batch_size]
 * @param rhs2 Output: packed 2-RHS matrix (device) [2*(n+m) * batch_size]
 * @param n Primal dimension
 * @param m Dual dimension
 * @param batch_size Number of batches
 */
void pack_rhs2(
    const double* x_part0,
    const double* z_part0,
    const double* x_part1,
    const double* z_part1,
    double* rhs2,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size,
    cudaStream_t stream = 0
);

/**
 * @brief Unpack 2-column solution into separate x/z buffers
 *
 * @param sol2 Input: packed 2-column solution (device) [2*(n+m+p) * batch_size]
 * @param x0 Output: first solution x part (device) [n * batch_size]
 * @param z0 Output: first solution z part (device) [m * batch_size]
 * @param x1 Output: second solution x part (device) [n * batch_size]
 * @param z1 Output: second solution z part (device) [m * batch_size]
 * @param n Primal dimension
 * @param m Dual dimension
 * @param p Expansion dimension (0 if no sparse SOC)
 * @param batch_size Number of batches
 */
void unpack_sol2(
    const double* sol2,
    double* x0,
    double* z0,
    double* x1,
    double* z1,
    int64_t n,
    int64_t m,
    int64_t p,
    int64_t batch_size,
    cudaStream_t stream = 0
);

/**
 * @brief Pointers + sizes consumed by `update_kkt_H_block`.
 *
 * Bundles ~50 raw arguments so the call site reads as field
 * initialisation rather than a long positional argument list. All
 * pointers are device memory; all dimensions are int64 counts.
 *
 * Members default to nullptr/0 — cone families that aren't present
 * (e.g. no SOC cones) leave their pointers null and dim counts zero,
 * and the kernel's grid-stride loops short-circuit on zero counts.
 */
struct KKTHBlockArgs {
    // KKT target.
    double* kkt_values = nullptr;
    int64_t nnzKKT = 0;
    int64_t batchSize = 0;

    // Zero + Nonneg slack cones.
    const int64_t* H_diagIdx = nullptr;
    const double* nonneg_w = nullptr;
    int64_t numZeroCones = 0;
    int64_t numNonnegCones = 0;

    // Exp slack cones (3x3 blocks).
    const int64_t* H_exp_idx = nullptr;
    const double* exp_Hs = nullptr;
    int64_t numExpCones = 0;

    // SOC slack cones (dense upper-tri for dim<=4; diagonal for dim>4).
    const int64_t* H_soc_idx = nullptr;
    const double* soc_Hs = nullptr;
    int64_t numSocCones = 0;
    int64_t totalSocHsEntries = 0;

    // Power slack cones (3x3 blocks).
    const int64_t* H_power_idx = nullptr;
    const double* power_Hs = nullptr;
    int64_t numPowerCones = 0;

    // Sparse SOC expansion (for dim > 4 cones).
    const int64_t* H_soc_u_idx = nullptr;
    const int64_t* H_soc_v_idx = nullptr;
    const int64_t* H_soc_exp_diag_idx = nullptr;
    const double* soc_u = nullptr;
    const double* soc_v = nullptr;
    const double* soc_eta = nullptr;
    const int64_t* d_soc_dims = nullptr;
    const int64_t* d_soc_offsets = nullptr;
    int64_t totalSocDim = 0;
    int64_t numSparseSoc = 0;
    const int64_t* d_soc_sparse_offsets = nullptr;
    const int64_t* d_soc_sparse_indices = nullptr;

    // GenPowerCone (dense: upper-tri Hs; sparse: diagonal + expansion).
    const int64_t* H_genpow_idx = nullptr;
    const int64_t* H_genpow_q_idx = nullptr;
    const int64_t* H_genpow_r_idx = nullptr;
    const int64_t* H_genpow_p_idx = nullptr;
    const int64_t* H_genpow_exp_diag_idx = nullptr;
    const double* genpow_Hs = nullptr;
    const double* genpow_q = nullptr;
    const double* genpow_r = nullptr;
    const double* genpow_p = nullptr;
    const double* genpow_mu = nullptr;
    int64_t numGenPowerCones = 0;
    int64_t numSparseGenPow = 0;
    int64_t totalGenPowerHsEntries = 0;
    const int64_t* d_genPowerDim1s = nullptr;
    const int64_t* d_genPowerDim2s = nullptr;
    const int64_t* d_genPowerOffsets = nullptr;
    const int64_t* d_genPowerAlphaOffsets = nullptr;
    const int64_t* d_genPowerSparseOffsets = nullptr;
    const int64_t* d_genPowerSparseIndices = nullptr;
    int64_t totalGenPowerDim = 0;
    int64_t totalGenPowerAlphas = 0;

    // Rank-9 PD-scaling expansion: 6 PD-axis off-diagonal column index
    // arrays (each totalSparseGenPowDim entries, aligned with
    // H_genpow_p_idx) plus PD state: pd_axes[6 * totalGenPowerDim],
    // pd_coefs/pd_signs/pd_active [6 * numGenPowerCones]. When inactive
    // the kernel writes a tiny ε to PD-axis off-diags and a +1 sentinel;
    // when active it writes -sqrt(|coef|)·axis and sign×(1/coef) (with
    // equilibration when w = |coef|·||axis||² > 1e12).
    const int64_t* H_genpow_pd_axis_idx[6] = {nullptr, nullptr, nullptr,
                                              nullptr, nullptr, nullptr};
    const double* genpow_pd_axes = nullptr;
    const double* genpow_pd_coefs = nullptr;
    const double* genpow_pd_signs = nullptr;
    const double* genpow_pd_active = nullptr;
};

/**
 * @brief Update H block of KKT matrix with cone scaling values.
 *
 * Copies Hs values from cone scaling arrays into the appropriate locations
 * in the KKT matrix. Handles zero cones, nonnegative cones, variable-dim
 * SOC blocks (dense + sparse expansion), 3x3 exp/power blocks, and
 * GenPower cones (dense + sparse + PD-scaling expansion).
 */
void update_kkt_H_block(const KKTHBlockArgs& args, cudaStream_t stream = 0);

/**
 * @brief Update NT scaling for direct-x nonneg cones.
 *
 * For each nonneg x-cone c with indices J of length k, reads x[J] and
 * the cone's dual z_x (flat, offset = xcone_numel_offsets[c]) and
 * writes:
 *   xcone_w[off + p]      = sqrt(x[J[p]] / z_x[off + p])
 *   xcone_lambda[off + p] = sqrt(x[J[p]] * z_x[off + p])
 *   xcone_Hs[hs_off + p]  = x[J[p]] / z_x[off + p]     (diagonal slot)
 *
 * Cones with a non-nonneg kind are skipped (handled by separate
 * kernels in later commits). Caller supplies the per-batch strides
 * `n` and `totalXConeNumel` as well as the total Hs buffer stride so
 * the kernel can index into flat device arrays correctly.
 */
/**
 * @brief Update NT scaling + pack Hs for direct-x SOC cones. Gathers
 * x[J[p]] into a local register set per cone, applies the standard SOC
 * NT scaling (same math as slack SOC via `update_soc_scaling_impl`),
 * and writes:
 *   - `xcone_w[numel_off + p]`      W vector
 *   - `xcone_lambda[numel_off + p]` λ = W·z
 *   - `xcone_eta[cone]`             η scaling factor
 *   - `xcone_Hs[hs_off + e]`        dense: column-major upper-tri of
 *                                   η²(2ww' − J) for dim ≤ 4;
 *                                   sparse: DIAGONAL η²·[d, 1, 1, ..., 1]
 *                                   for dim > 4.
 *   - `xcone_u`, `xcone_v`, `xcone_d`  rank-2 sparse expansion (dim > 4
 *                                      only) in cone-internal order;
 *                                      scattered to KKT u/v columns by
 *                                      `refresh_xcone_hs`.
 *
 * One block per (batch, cone) with `SOC_PARALLEL_BLOCK_SIZE` threads
 * cooperating via `cones::block_sum_reduce` for tail-norm reductions —
 * dim has no upper bound. Cones with kind ≠ SOC are skipped (nonneg
 * handled by update_xcones_nonneg_scaling).
 */
void update_xcones_soc_scaling(
    const double* x,                        // [batchSize * n]
    const double* z_x,                      // [batchSize * totalXConeNumel]
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_sparse_indices,  // [numXCones], ≥0 iff dim>4
    const int64_t* d_xcone_sparse_offsets,  // [numXCones+1]
    const int64_t* d_xcone_cone_pos_for_sorted, // [totalXConeNumel]
    double* xcone_w,                        // [batchSize * totalXConeNumel]
    double* xcone_lambda,
    double* xcone_eta,                      // [batchSize * numXCones]
    double* xcone_Hs,                       // [batchSize * totalXConeHsEntries]
    double* xcone_u,                        // [batchSize * totalSparseXSocDim]
    double* xcone_v,                        // [batchSize * totalSparseXSocDim]
    double* xcone_d,                        // [batchSize * numSparseXSoc]
    // Fused scatter: when non-null, Hs / expansion values are written
    // directly into KKT.values, making the per-iter `refresh_xcone_hs`
    // pass a no-op. Pass nullptrs for the classic two-step path.
    double* kkt_values,
    const double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    const int64_t* H_xcone_u_idx,
    const int64_t* H_xcone_v_idx,
    const int64_t* H_xcone_exp_diag_idx,
    int64_t nnzKKT,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t totalSparseXSocDim,
    int64_t numSparseXSoc,
    cudaStream_t stream = 0
);

void update_xcones_nonneg_scaling(
    const double* x,                        // [batchSize * n]
    const double* z_x,                      // [batchSize * totalXConeNumel]
    const int64_t* d_xcone_kinds,           // [numXCones]
    const int64_t* d_xcone_dims,            // [numXCones]
    const int64_t* d_xcone_numel_offsets,   // [numXCones + 1]
    const int64_t* d_xcone_hs_offsets,      // [numXCones + 1]
    const int64_t* d_xcone_indices,         // [totalXConeNumel]
    double* xcone_w,                        // [batchSize * totalXConeNumel]
    double* xcone_lambda,                   // [batchSize * totalXConeNumel]
    double* xcone_Hs,                       // [batchSize * totalXConeHsEntries]
    // Fused scatter: nullable; see SOC variant above.
    double* kkt_values,
    const double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    int64_t nnzKKT,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0
);

/**
 * @brief Direct-x nonneg step-math kernels.
 *
 * These mirror the CPU machinery in packages/moreau-cpu:
 * - `direct_x_affine_ds` (fill): step_rhs.z_x[k] = x[J[k]] · z_x[k]  (λ²)
 * - `direct_x_combined_ds_shift` (add): step_rhs.z_x[k] += step_aff.x[J[k]] · step_aff.z_x[k] - σμ
 * - affine RHS correction: workx[J[k]] -= variables.z_x[k]
 * - combined RHS correction: workx[J[k]] -= step_rhs.z_x[k] / variables.x[J[k]]
 * - affine Δz_x recovery: Δz_x[k] = -xcone_Hs[k] · Δx[J[k]] - variables.z_x[k]
 * - combined Δz_x recovery: Δz_x[k] = -xcone_Hs[k] · Δx[J[k]] - step_rhs.z_x[k] / variables.x[J[k]]
 *
 * All are one-block-per-batch, thread-stride over direct-x entries.
 * Indices in `d_xcone_indices` are disjoint across x-cones so no atomics.
 */
void fill_step_rhs_zx_nonneg_affine(
    double* step_rhs_z_x,
    const double* x,
    const double* var_z_x,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_kind_per_entry,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void add_combined_ds_shift_nonneg(
    double* step_rhs_z_x,
    const double* step_aff_x,
    const double* step_aff_z_x,
    const double* sigma_mu,
    const double* mehrotra_m,   // per-batch m scaling applied to step_aff.z_x
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_kind_per_entry,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void subtract_xcone_affine_from_workx(
    double* workx,
    const double* var_z_x,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void subtract_xcone_combined_from_workx(
    double* workx,
    const double* rhs_z_x,
    const double* var_x,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_kind_per_entry,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void recover_dz_x_affine_nonneg(
    double* dz_x,
    const double* lhs_x,
    const double* var_z_x,
    const double* xcone_Hs,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_kind_per_entry,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0);

void recover_dz_x_combined_nonneg(
    double* dz_x,
    const double* lhs_x,
    const double* rhs_z_x,
    const double* var_x,
    const double* xcone_Hs,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_kind_per_entry,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0);

/// Margin-based shift of primal x[J] into the direct-x nonneg cone
/// interior, mirroring CPU `_shift_single_cone_to_interior`:
///   1. Per cone, compute min_margin = min_p x[J[p]] and
///      pos_margin = Σ_p max(x[J[p]], 0).
///   2. target = max(1, 0.1 * pos_margin / dim).
///   3. If min_margin ≤ 0: shift all x[J[p]] by (-min_margin + target).
///      Else if min_margin < target: shift by (target - min_margin).
///      Else: no shift.
///
/// Block-cooperative per (batch, cone) — `SOC_PARALLEL_BLOCK_SIZE` threads
/// reduce min / positive-sum via `cones::block_{min,sum}_reduce`.
void shift_x_into_nonneg_interior(
    double* x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    cudaStream_t stream = 0);

/// Margin-based SOC-interior shift for direct-x SOC (dense, dim ≤ 4).
/// Mirror of CPU `_shift_single_cone_to_interior` wrapped through SOC
/// margins. No-op for nonneg cones.
void shift_x_into_soc_interior(
    double* x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    cudaStream_t stream = 0);

/// Initialize x[J] for direct-x Exp cones (kind==3) to the ExpCone
/// self-conjugate unit-init point (-1.051..., 0.556..., 1.259...).
/// Mirrors CPU direct_x_unit_initialization for ExponentialCone.
void init_xcone_x_exp(
    double* x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    cudaStream_t stream = 0);

/// Cone-interior default for z_x: 1.0 per entry for nonneg, e_0 for SOC.
/// Runs before the x[J] shift at IPM start.
void init_xcone_z_x(
    double* z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    int64_t batchSize, int64_t numXCones, int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void xcone_step_length_nonneg_reduce(
    double* alpha_s,
    double* alpha_z,
    const double* var_x,
    const double* var_z_x,
    const double* step_x,
    const double* step_z_x,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_kind_per_entry,
    int64_t batchSize, int64_t n, int64_t totalXConeNumel,
    double max_step,
    cudaStream_t stream = 0);

/**
 * @brief Direct-x SOC step-math kernels.
 *
 * These mirror the CPU SOC path wrapped through the direct-x swap
 * (primal slot ↔ slack slot). All are one-block-per-(batch, cone) with
 * `SOC_PARALLEL_BLOCK_SIZE` threads cooperating via shared-memory
 * reductions — dim is unbounded (matches scaling).
 *
 * - soc affine_ds:     step_rhs.z_x[off..off+dim] = λ∘λ (Jordan square)
 * - soc combined_ds_shift: step_rhs.z_x += (W·Δx) ∘ (W⁻¹·m·Δz) − σμ·e
 * - soc subtract_combined: workx[J] -= Δs_from_Δz_offset(rhs.z_x, x[J])
 * - soc recover_affine:  Δz_x = -Hs·Δx[J] - variables.z_x
 * - soc recover_combined: Δz_x = -Hs·Δx[J] - Δs_from_Δz_offset(rhs.z_x, x[J])
 * - soc step_length:    quadratic min-ratio along SOC boundary
 */
void fill_step_rhs_zx_soc_affine(
    double* step_rhs_z_x,
    const double* xcone_lambda,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    int64_t batchSize, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void add_combined_ds_shift_soc(
    double* step_rhs_z_x,
    const double* step_aff_x,
    const double* step_aff_z_x,
    const double* sigma_mu,
    const double* mehrotra_m,
    const double* xcone_w,
    const double* xcone_eta,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void subtract_xcone_combined_from_workx_soc(
    double* workx,
    const double* rhs_z_x,
    const double* var_x,
    const double* xcone_w,
    const double* xcone_lambda,
    const double* xcone_eta,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void recover_dz_x_affine_soc(
    double* dz_x,
    const double* lhs_x,
    const double* var_z_x,
    const double* xcone_w,
    const double* xcone_eta,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void recover_dz_x_combined_soc(
    double* dz_x,
    const double* lhs_x,
    const double* rhs_z_x,
    const double* var_x,
    const double* xcone_w,
    const double* xcone_lambda,
    const double* xcone_eta,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void xcone_step_length_soc_reduce(
    double* alpha_s,
    double* alpha_z,
    const double* var_x,
    const double* var_z_x,
    const double* step_x,
    const double* step_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    double max_step,
    cudaStream_t stream = 0);

/**
 * @brief Direct-x ExpCone step-math kernels (Phase 2: asymmetric cones).
 *
 * These mirror the CPU `direct_x_*` trait methods for ExponentialCone. Unlike
 * symmetric cones (SOC, nonneg), there is no primal↔dual NT swap — the primal
 * barrier Hessian is used directly for the (1,1) block augmentation.
 *
 *   - exp scaling:   xcone_Hs[cone] = mu * H_primal(x),
 *                    xcone_grad_primal[cone] = ∇F_primal(x)
 *   - exp affine_ds: step_rhs.z_x[cone] = var_z_x[cone]  (identity copy)
 *   - exp combined_ds_shift: step_rhs.z_x[cone] += σμ * ∇F_primal(x)
 *   - exp step_length: backtrack on is_primal_feasible(x + α·dx)
 *                      and is_dual_feasible(z + α·dz)
 *   - exp subtract_combined: workx[J] -= rhs_z_x  (identity offset, no /x)
 *   - exp recover_affine:    dz_x = -Hs·dx[J] - var_z_x
 *   - exp recover_combined:  dz_x = -Hs·dx[J] - rhs_z_x
 */
void update_xcones_exp_scaling(
    const double* x,
    const double* z_x,
    const double* mu,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    double* xcone_Hs,
    double* xcone_grad_primal,
    double* kkt_values,
    const double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    int64_t nnzKKT,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0);

void fill_step_rhs_zx_exp_affine(
    double* step_rhs_z_x,
    const double* var_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    int64_t batchSize, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void add_combined_ds_shift_exp(
    double* step_rhs_z_x,
    const double* sigma_mu,
    const double* mehrotra_m,
    const double* mu,
    const double* var_x,
    const double* step_aff_x,
    const double* step_aff_z_x,
    const double* xcone_Hs,
    const double* xcone_grad_primal,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0);

void xcone_step_length_exp_reduce(
    double* alpha_s,
    double* alpha_z,
    const double* var_x,
    const double* var_z_x,
    const double* step_x,
    const double* step_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    double max_step,
    double min_step,
    double backtrack,
    cudaStream_t stream = 0);

void subtract_xcone_combined_from_workx_exp(
    double* workx,
    const double* rhs_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void recover_dz_x_affine_exp(
    double* dz_x,
    const double* lhs_x,
    const double* var_z_x,
    const double* xcone_Hs,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0);

void recover_dz_x_combined_exp(
    double* dz_x,
    const double* lhs_x,
    const double* rhs_z_x,
    const double* xcone_Hs,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0);

// ============================================================================
// Direct-x PowerCone kernel declarations (Phase 3)
// ============================================================================
//
// Mirror of the ExpCone direct-x API (kind == 3), but for kind == 4 (Power).
// The primal barrier F(s) = -log(phi - s[2]²) - (1-α)·log(s[0]) - α·log(s[1])
// where phi = s[0]^(2α) · s[1]^(2(1-α)). α is per-cone and passed via
// d_xcone_pow_idx / d_xcone_pow_alpha arrays.

/**
 * @brief Initialize x[J] for Power x-cones to the unit-init point
 *        (sqrt(1+α), sqrt(2-α), 0). Filter by kind == 4.
 */
void init_xcone_x_pow(
    double* x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_pow_idx,
    const double*  d_xcone_pow_alpha,
    int64_t batchSize, int64_t n, int64_t numXCones,
    cudaStream_t stream = 0);

/**
 * @brief Overwrite z_x for Power x-cones with the correct unit-init point
 *        (sqrt(1+α), sqrt(2-α), 0). Must be called after init_xcone_z_x
 *        when numXPowerCones > 0, to correct the wrong Exp point.
 *        Filter by kind == 4.
 */
void init_xcone_z_x_pow(
    double* z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_pow_idx,
    const double*  d_xcone_pow_alpha,
    int64_t batchSize, int64_t numXCones, int64_t totalXConeNumel,
    cudaStream_t stream = 0);

/**
 * @brief Compute Hs = mu·∇²F_primal(x) and grad_primal = ∇F_primal(x) for
 *        each Power x-cone. Filter by kind == 4. Accepts optional fused KKT
 *        scatter (kkt_values, xcone_px_baseline, H_xcone_hs_idx all nullable).
 */
void update_xcones_pow_scaling(
    const double* x,
    const double* z_x,
    const double* mu,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_pow_idx,
    const double*  d_xcone_pow_alpha,
    double* xcone_Hs,
    double* xcone_grad_primal,
    double* kkt_values,
    const double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    int64_t nnzKKT,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0);

void fill_step_rhs_zx_pow_affine(
    double* step_rhs_z_x,
    const double* var_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    int64_t batchSize, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void add_combined_ds_shift_pow(
    double* step_rhs_z_x,
    const double* sigma_mu,
    const double* mehrotra_m,
    const double* mu,
    const double* var_x,
    const double* step_aff_x,
    const double* step_aff_z_x,
    const double* xcone_Hs,
    const double* xcone_grad_primal,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_pow_idx,
    const double*  d_xcone_pow_alpha,
    int64_t batchSize, int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0);

void xcone_step_length_pow_reduce(
    double* alpha_s,
    double* alpha_z,
    const double* var_x,
    const double* var_z_x,
    const double* step_x,
    const double* step_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_pow_idx,
    const double*  d_xcone_pow_alpha,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    double max_step,
    double min_step,
    double backtrack,
    cudaStream_t stream = 0);

void subtract_xcone_combined_from_workx_pow(
    double* workx,
    const double* rhs_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void recover_dz_x_affine_pow(
    double* dz_x,
    const double* lhs_x,
    const double* var_z_x,
    const double* xcone_Hs,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0);

void recover_dz_x_combined_pow(
    double* dz_x,
    const double* lhs_x,
    const double* rhs_z_x,
    const double* xcone_Hs,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0);

// ============================================================================
// Direct-x GenPowerCone kernel declarations (Phase 4)
// ============================================================================
//
// Variable-dim GenPow: K = {(p,w) : ∏ p_i^αi ≥ ‖w‖₂, p_i ≥ 0}
// dim1 = len(alphas), dim2 = len(w), dim = dim1 + dim2.
// Primal barrier Hessian: Hs = μ·(D + p·p' - q·q' - r·r') (rank-3 sparse).
// Dense (dim≤4): full upper-tri Hs. Sparse (dim>4): diagonal + 3 expansion cols.
// Direct-x Dsigns = [+1,+1,-1] (opposite of slack's [-1,-1,+1]).

/**
 * @brief Initialize x[J] for GenPower x-cones to the unit-init point.
 *        x[i] = sqrt(1 + alpha_i) for i < dim1, x[i] = 0 for i >= dim1.
 *        Filter by kind == XConeKind::GenPower.
 */
void init_xcone_x_genpow(
    double* x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const double*  d_xcone_genpow_alphas,
    int64_t batchSize, int64_t n, int64_t numXCones,
    cudaStream_t stream = 0);

/**
 * @brief Overwrite z_x for GenPower x-cones with the unit-init point.
 *        Same formula as init_xcone_x_genpow. Filter by kind == GenPower.
 */
void init_xcone_z_x_genpow(
    double* z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const double*  d_xcone_genpow_alphas,
    int64_t batchSize, int64_t numXCones, int64_t totalXConeNumel,
    cudaStream_t stream = 0);

/**
 * @brief Compute Hs = μ·(D + p·p' - q·q' - r·r') for each GenPow x-cone.
 *        Dense (dim≤4): writes full upper-tri to xcone_Hs.
 *        Sparse (dim>4): writes diagonal to xcone_Hs, p/q/r vectors, and
 *        scatters into KKT expansion columns.
 *        Filter by kind == GenPower.
 */
void update_xcones_genpow_scaling(
    const double* x,
    const double* z_x,
    const double* mu,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const int64_t* d_xcone_genpow_dim_offsets,
    const int64_t* d_xcone_genpow_sparse_idx,
    const int64_t* d_xcone_genpow_sparse_offsets,
    const int64_t* d_xcone_genpow_sparse_q_offsets,
    const int64_t* d_xcone_genpow_sparse_r_offsets,
    const double*  d_xcone_genpow_alphas,
    double* xcone_Hs,
    double* xcone_grad_primal,
    double* xcone_genpow_p,
    double* xcone_genpow_q,
    double* xcone_genpow_r,
    double* xcone_genpow_d1,
    double* xcone_genpow_d2,
    double* kkt_values,
    const double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    const int64_t* H_xcone_genpow_q_idx,
    const int64_t* H_xcone_genpow_r_idx,
    const int64_t* H_xcone_genpow_p_idx,
    const int64_t* H_xcone_genpow_pd_axis_idx_0,
    const int64_t* H_xcone_genpow_pd_axis_idx_1,
    const int64_t* H_xcone_genpow_pd_axis_idx_2,
    const int64_t* H_xcone_genpow_pd_axis_idx_3,
    const int64_t* H_xcone_genpow_pd_axis_idx_4,
    const int64_t* H_xcone_genpow_pd_axis_idx_5,
    const int64_t* H_xcone_genpow_exp_diag_idx,  // points into xcone exp_diag array
    // Optional rank-6 PD-scaling state (nullable → inactive defaults).
    const double* xgenpow_pd_axes,    // (B, 6 * totalXGenPowerDim)
    const double* xgenpow_pd_coefs,   // (B, 6 * numXGenPowerCones)
    const double* xgenpow_pd_signs,   // (B, 6 * numXGenPowerCones)
    const double* xgenpow_pd_active,  // (B, numXGenPowerCones)
    int64_t nnzKKT,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    int64_t numSparseXGenPow,
    int64_t totalSparseXGenPowDim,
    cudaStream_t stream = 0);

void fill_step_rhs_zx_genpow_affine(
    double* step_rhs_z_x,
    const double* var_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    int64_t batchSize, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void add_combined_ds_shift_genpow(
    double* step_rhs_z_x,
    const double* sigma_mu,
    const double* xcone_grad_primal,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    int64_t batchSize, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

/**
 * @brief 3rd-order Mehrotra η correction for direct-x GenPow:
 *        step_rhs_z_x[i] -= (μ/2) · D³F(x)[step_x, step_x, ·] · cap_scale
 * with K=7 ∞-norm cap against ‖step_z‖_∞. Mirror of CPU
 * `GenPowerCone::higher_correction_primal_direct`.
 */
void subtract_eta_primal_genpow(
    double* step_rhs_z_x,
    const double* var_x,
    const double* step_x,
    const double* step_z,
    const double* mu_per_batch,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const double* d_xcone_genpow_alphas,
    int64_t batchSize,
    int64_t n,
    int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void xcone_step_length_genpow_reduce(
    double* alpha_s,
    double* alpha_z,
    const double* var_x,
    const double* var_z_x,
    const double* step_x,
    const double* step_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const double*  d_xcone_genpow_alphas,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    double max_step,
    double min_step,
    double backtrack,
    cudaStream_t stream = 0);

void subtract_xcone_combined_from_workx_genpow(
    double* workx,
    const double* rhs_z_x,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_indices,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t totalXConeNumel,
    cudaStream_t stream = 0);

void recover_dz_x_affine_genpow(
    double* dz_x,
    const double* lhs_x,
    const double* var_z_x,
    const double* xcone_Hs,
    const double* xcone_genpow_p,
    const double* xcone_genpow_q,
    const double* xcone_genpow_r,
    const double* xgenpow_pd_axes,
    const double* xgenpow_pd_coefs,
    const double* xgenpow_pd_signs,
    const double* xgenpow_pd_active,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const int64_t* d_xcone_genpow_dim_offsets,
    const int64_t* d_xcone_genpow_sparse_idx,
    const int64_t* d_xcone_genpow_sparse_offsets,
    double* genpow_recover_workspace,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    int64_t numSparseXGenPow,
    cudaStream_t stream = 0);

void recover_dz_x_combined_genpow(
    double* dz_x,
    const double* lhs_x,
    const double* rhs_z_x,
    const double* xcone_Hs,
    const double* xcone_genpow_p,
    const double* xcone_genpow_q,
    const double* xcone_genpow_r,
    const double* xgenpow_pd_axes,
    const double* xgenpow_pd_coefs,
    const double* xgenpow_pd_signs,
    const double* xgenpow_pd_active,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* d_xcone_hs_offsets,
    const int64_t* d_xcone_indices,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_alpha_offsets,
    const int64_t* d_xcone_genpow_dim_offsets,
    const int64_t* d_xcone_genpow_sparse_idx,
    const int64_t* d_xcone_genpow_sparse_offsets,
    double* genpow_recover_workspace,
    int64_t batchSize, int64_t n, int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXConeNumel,
    int64_t totalXConeHsEntries,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    int64_t numSparseXGenPow,
    cudaStream_t stream = 0);

/**
 * @brief Refresh KKT expansion columns for sparse GenPow x-cones.
 *        Scatters q, r, p column values and exp-diag entries into KKT.
 *        Called from refresh_xcone_hs when numSparseXGenPow > 0.
 */
void refresh_xcone_genpow_expansion(
    double* kkt_values,
    const double* xcone_genpow_p,
    const double* xcone_genpow_q,
    const double* xcone_genpow_r,
    const int64_t* d_xcone_genpow_sparse_offsets,
    const int64_t* d_xcone_genpow_sparse_alpha_offsets,
    const int64_t* d_xcone_genpow_sparse_q_offsets,
    const int64_t* d_xcone_genpow_sparse_r_offsets,
    const int64_t* d_xcone_genpow_sparse_to_gidx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_dim_offsets,
    const int64_t* H_xcone_genpow_q_idx,
    const int64_t* H_xcone_genpow_r_idx,
    const int64_t* H_xcone_genpow_p_idx,
    const int64_t* H_xcone_genpow_exp_diag_idx,
    int64_t batchSize, int64_t nnzKKT,
    int64_t numSparseXGenPow,
    int64_t totalXGenPowerDim,
    int64_t totalXGenPowerAlphas,
    int64_t totalXGenPowerDim2,
    cudaStream_t stream = 0);

/**
 * @brief Apply rank-9 PD axes (6 off-diag cols + 6 diag entries per cone)
 *        to the KKT block of sparse GenPow x-cones. Writes active values
 *        when xgenpow_pd_active[c]=1 (computed by compute_xgenpow_pd_axes);
 *        writes inactive sentinels (tiny ε + sign·1) otherwise so cuDSS
 *        sees a structurally non-singular block.
 *
 *        Called from refresh_xcone_hs after refresh_xcone_genpow_expansion.
 */
void apply_xgenpow_pd_to_kkt(
    double* kkt_values,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_dim1s,
    const int64_t* d_xcone_genpow_dim2s,
    const int64_t* d_xcone_genpow_dim_offsets,
    const int64_t* d_xcone_genpow_sparse_idx,
    const int64_t* d_xcone_genpow_sparse_offsets,
    const double* xgenpow_pd_axes,
    const double* xgenpow_pd_coefs,
    const double* xgenpow_pd_signs,
    const double* xgenpow_pd_active,
    const int64_t* H_xcone_genpow_pd_axis_idx_0,
    const int64_t* H_xcone_genpow_pd_axis_idx_1,
    const int64_t* H_xcone_genpow_pd_axis_idx_2,
    const int64_t* H_xcone_genpow_pd_axis_idx_3,
    const int64_t* H_xcone_genpow_pd_axis_idx_4,
    const int64_t* H_xcone_genpow_pd_axis_idx_5,
    const int64_t* H_xcone_genpow_exp_diag_idx,
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t totalXGenPowerDim,
    int64_t nnzKKT,
    int64_t batchSize,
    cudaStream_t stream = 0);

/**
 * @brief Refresh the static `dsigns` array's direct-x GenPow PD-axis slots
 *        to match the runtime sign of the sentinel `apply_xgenpow_pd_to_kkt`
 *        writes to each diagonal.
 *
 * The sentinel's *sign* is always `-pd_signs[k]` (both active and inactive
 * branches), so dsigns[col_for_axis_k] = `(pd_signs > 0) ? -1 : +1`.
 *
 * `pd_signs` is per-batch but `dsigns` is single-batch — refresh reads
 * batch 0. Batches with diverging cone-active state will see a mild
 * regularization sign mismatch that cuDSS pivoting absorbs.
 *
 * @param pd_axis_base_col First direct-x GenPow PD-axis column position:
 *        n + m + 2·numSparseSoc + 9·numSparseGenPow + 2·numSparseXSoc.
 */
void refresh_xgenpow_pd_dsigns(
    int8_t* dsigns,
    const double* pd_signs,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_genpow_idx,
    const int64_t* d_xcone_genpow_sparse_idx,
    int64_t numXCones,
    int64_t numXGenPowerCones,
    int64_t pd_axis_base_col,
    cudaStream_t stream = 0);

/**
 * @brief Slack GenPow analogue of refresh_xgenpow_pd_dsigns.
 *
 * Refreshes dsigns for the slack GenPow PD-axis diagonal slots. Slack
 * sentinel = +sign[k] (versus -sign[k] for direct-x), so dsigns is +1 when
 * pd_signs[k] > 0 and -1 otherwise — opposite of the direct-x mapping.
 * Single-batch refresh from batch 0 (see refresh_xgenpow_pd_dsigns for the
 * caveat about cross-batch sign divergence).
 *
 * @param pd_axis_base_col First slack GenPow PD-axis column position:
 *        n + m + 2·numSparseSoc.
 */
void refresh_genpow_pd_dsigns(
    int8_t* dsigns,
    const double* pd_signs,
    const int64_t* d_genPowerSparseIndices,
    int64_t numGenPowerCones,
    int64_t pd_axis_base_col,
    cudaStream_t stream = 0);

/**
 * @brief Snapshot KKT.values at direct-x cone Hs slots into a baseline buffer.
 *
 * One-shot init after populate(): records P's contribution at each x-cone
 * Hs slot so subsequent refresh_xcone_hs calls can add the running Hs
 * update on top without stomping the static P value.
 *
 *   px_baseline[b][k] = kkt_values[b][H_xcone_hs_idx[k]]
 */
void snapshot_kkt_at_xcone_slots(
    const double* kkt_values,
    double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0
);

/**
 * @brief Refresh direct-x cone contributions in the KKT (1,1) block.
 *
 * For each x-cone Hs entry k in [0, totalXConeHsEntries), writes:
 *   kkt_values[H_xcone_hs_idx[k]] = xcone_px_baseline[k] + xcone_Hs[k]
 * for every batch. The baseline captures P's contribution at that KKT
 * slot (or 0 for structural-zero positions added by Pext), so calling
 * this per IPM iteration keeps `KKT(i,j) = P(i,j) + Σ_J H_J(i,j)` at
 * every (i,j) that belongs to some direct-x cone footprint.
 *
 * Mirrors the CPU `refresh_hx_blocks` path.
 *
 * @param kkt_values        KKT matrix values (device) [batchSize * nnzKKT]
 * @param xcone_Hs          Direct-x Hs entries (device) [batchSize * totalXConeHsEntries]
 * @param xcone_px_baseline Per-slot P contribution (device) [batchSize * totalXConeHsEntries]
 * @param H_xcone_hs_idx    Slot map (device) [totalXConeHsEntries]
 * @param batchSize         Number of problems
 * @param nnzKKT            KKT nnz
 * @param totalXConeHsEntries  Number of x-cone Hs entries
 */
void refresh_xcone_hs(
    double* kkt_values,
    const double* xcone_Hs,
    const double* xcone_px_baseline,
    const int64_t* H_xcone_hs_idx,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0
);

/**
 * @brief Seed xcone_Hs with the identity-scaling Hs per cone. Used in
 * default_start BEFORE the first refresh_xcone_hs to prime the KKT (1,1)
 * slots with a well-conditioned initial scaling.
 *
 * Per cone:
 *   - Nonneg: diagonal, all ones.
 *   - Dense SOC (dim ≤ 4): column-major upper-tri packed identity (1s
 *     only at diagonal positions; zeros off-diagonal).
 *   - Rank-2 sparse SOC (dim > 4): diagonal `[0.5, 1, 1, ..., 1]`
 *     (η=1, d=0.5 at identity NT scaling).
 *
 * CRITICAL: the generic `xcone_Hs.setToConstant(1.0)` is WRONG for
 * dense SOC (makes Hs rank-1 all-ones, blowing up initial μ by ~200×).
 */
void seed_xcone_Hs_identity(
    double* xcone_Hs,
    const int64_t* d_xcone_kinds,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_hs_offsets,
    int64_t batchSize,
    int64_t numXCones,
    int64_t totalXConeHsEntries,
    cudaStream_t stream = 0
);

/**
 * @brief Scatter direct-x sparse SOC rank-2 expansion into KKT (u/v
 * columns + 2-diag expansion). Per-iter counterpart to the `xcone_Hs`
 * diagonal refresh for dim > 4 direct-x SOC cones. Signs follow direct-x
 * convention (+Hs contribution): `+η²·u`, `+η²·v`, `[+η², -η²]` diag.
 *
 * `d_xcone_cone_pos_for_sorted` permutes cone-internal entries (the
 * order xcone_u/v are written by the scaling kernel) into sorted-row
 * order (the order H_xcone_u_idx / H_xcone_v_idx expect). This matches
 * CPU `DirectXSparseMap.cone_pos_for_sorted`.
 */
void refresh_xcone_sparse_expansion(
    double* kkt_values,
    const double* xcone_u,
    const double* xcone_v,
    const double* xcone_eta,
    const int64_t* d_xcone_dims,
    const int64_t* d_xcone_sparse_indices,
    const int64_t* d_xcone_sparse_offsets,
    const int64_t* d_xcone_cone_pos_for_sorted,
    const int64_t* d_xcone_numel_offsets,
    const int64_t* H_xcone_u_idx,
    const int64_t* H_xcone_v_idx,
    const int64_t* H_xcone_exp_diag_idx,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t numXCones,
    int64_t totalSparseXSocDim,
    int64_t numSparseXSoc,
    cudaStream_t stream = 0
);

/**
 * @brief Update PSD Hessian block of KKT matrix
 *
 * Scatters PSD Hs values (upper triangle of svec_dim×svec_dim per cone)
 * into the KKT matrix at pre-computed index positions.
 */
void update_kkt_psd_H_block(
    double* kkt_values,
    const int64_t* H_psd_idx,
    const double* psd_Hs,
    const int64_t* d_psd_Hs_offsets,
    int64_t totalPsdHsEntries,
    int64_t numPsdCones,
    int64_t batchSize,
    int64_t nnzKKT,
    cudaStream_t stream = 0
);

/**
 * @brief Backup diagonal values from KKT matrix
 * @param kkt_values KKT matrix values (device) [batchSize * nnzKKT]
 * @param work_diag Workspace for diagonal backup (device) [batchSize * (n+m)]
 * @param diag_full Indices of diagonal entries (device) [n+m]
 * @param batchSize Number of problems
 * @param nnzKKT Number of nonzeros in KKT
 * @param N Dimension of KKT (n+m)
 */
void backup_diagonal(
    const double* kkt_values,
    double* work_diag,
    const int64_t* diag_full,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream = 0
);

/**
 * @brief Fused backup and regularize diagonal in a single kernel
 * @param kkt_values KKT matrix values (device) [batchSize * nnzKKT]
 * @param work_diag Workspace for diagonal backup (device) [batchSize * (n+m)]
 * @param diag_full Indices of diagonal entries (device) [n+m]
 * @param dsigns Signs for regularization (device) [n+m]: +1 or -1
 * @param eps Regularization parameters per batch (device) [batchSize]
 * @param batchSize Number of problems
 * @param nnzKKT Number of nonzeros in KKT
 * @param N Dimension of KKT (n+m)
 */
void backup_and_regularize_diagonal(
    double* kkt_values,
    double* work_diag,
    const int64_t* diag_full,
    const int8_t* dsigns,
    const double* eps,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream = 0
);

/**
 * @brief Apply regularization to KKT diagonal
 * @param kkt_values KKT matrix values (device) [batchSize * nnzKKT]
 * @param diag_full Indices of diagonal entries (device) [n+m]
 * @param dsigns Signs for regularization (device) [n+m]: +1 or -1
 * @param eps Regularization parameters per batch (device) [batchSize]
 * @param batchSize Number of problems
 * @param nnzKKT Number of nonzeros in KKT
 * @param N Dimension of KKT (n+m)
 */
void regularize_diagonal(
    double* kkt_values,
    const int64_t* diag_full,
    const int8_t* dsigns,
    const double* eps,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream = 0
);

/**
 * @brief Restore diagonal values to KKT matrix
 * @param kkt_values KKT matrix values (device) [batchSize * nnzKKT]
 * @param work_diag Workspace with backed up diagonal (device) [batchSize * (n+m)]
 * @param diag_full Indices of diagonal entries (device) [n+m]
 * @param batchSize Number of problems
 * @param nnzKKT Number of nonzeros in KKT
 * @param N Dimension of KKT (n+m)
 */
void restore_diagonal(
    double* kkt_values,
    const double* work_diag,
    const int64_t* diag_full,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream = 0
);

/**
 * @brief Compute infinity norm of diagonal (max abs value) per batch
 * @param kkt_values KKT matrix values (device) [batchSize * nnzKKT]
 * @param diag_full Indices of diagonal entries (device) [n+m]
 * @param result Output: max diagonal value per batch (device) [batchSize]
 * @param batchSize Number of batches
 * @param nnzKKT Number of nonzeros in KKT
 * @param N Dimension of KKT (n+m)
 */
void diagonal_inf_norm(
    const double* kkt_values,
    const int64_t* diag_full,
    double* result,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream = 0
);

/**
 * @brief Compute regularizer per batch: eps[i] = eps_c + eps_p * max_diag[i]
 * @param eps_values Input/Output: max diagonal values, overwritten with eps values (device) [batchSize]
 * @param eps_c Constant regularization term
 * @param eps_p Proportional regularization term
 * @param batchSize Number of batches
 */
void compute_regularizer_per_batch(
    double* eps_values,
    double eps_c,
    double eps_p,
    int64_t batchSize,
    cudaStream_t stream = 0
);

/**
 * @brief Fused backup + inf-norm + regularize diagonal (4→1 kernel)
 *
 * Single kernel that:
 * 1. Backs up diagonal values to work_diag
 * 2. Computes inf-norm of diagonal via shared memory reduction
 * 3. Computes eps = eps_c + eps_p * max_diag
 * 4. Applies regularization: diag += sign * eps
 *
 * @param kkt_values KKT matrix values (device) [batchSize * nnzKKT]
 * @param work_diag Workspace for diagonal backup (device) [batchSize * N]
 * @param diag_full Indices of diagonal entries (device) [N]
 * @param dsigns Signs for regularization (device) [N]: +1 or -1
 * @param eps_out Output: regularization eps per batch (device) [batchSize]
 * @param eps_c Constant regularization term
 * @param eps_p Proportional regularization term
 * @param batchSize Number of problems
 * @param nnzKKT Number of nonzeros in KKT
 * @param N Dimension of KKT (n+m)
 */
void fused_backup_infnorm_regularize(
    double* kkt_values,
    double* work_diag,
    const int64_t* diag_full,
    const int8_t* dsigns,
    double* eps_out,
    double eps_c,
    double eps_p,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream = 0
);

/**
 * @brief Set KKT diagonal values to 1.0 for warmup factorization
 *
 * Used to initialize KKT matrix before warmup factorization to pre-allocate cuDSS workspace.
 *
 * @param kkt_values KKT matrix values (device) [batchSize * nnzKKT]
 * @param diag_full Indices of diagonal entries (device) [N]
 * @param batchSize Number of problems
 * @param nnzKKT Number of nonzeros in KKT
 * @param N Dimension of KKT (n+m)
 */
void set_kkt_diagonal_to_ones(
    double* kkt_values,
    const int64_t* diag_full,
    int64_t batchSize,
    int64_t nnzKKT,
    int64_t N,
    cudaStream_t stream = 0
);

// ========== Full matrix expansion for Streams strategy ==========

/**
 * @brief Expand upper-triangle CSR values to full symmetric CSR
 *
 * Given an upper-triangle CSR matrix, copies values to a full symmetric CSR.
 * The full CSR must have been pre-constructed with the correct structure.
 *
 * @param upper_values Upper-triangle CSR values (device) [batchSize * nnz_upper]
 * @param full_values Output: full CSR values (device) [batchSize * nnz_full]
 * @param upper_to_full_map Mapping from upper index to full index (device) [nnz_upper]
 * @param upper_to_transpose_map Mapping from upper index to transpose position in full (device) [nnz_upper]
 *                               -1 for diagonal entries (no transpose needed)
 * @param nnz_upper Number of nonzeros in upper-triangle matrix
 * @param nnz_full Number of nonzeros in full matrix
 * @param batchSize Number of problems
 */
void expand_upper_to_full(
    const double* upper_values,
    double* full_values,
    const int64_t* upper_to_full_map,
    const int64_t* upper_to_transpose_map,
    int64_t nnz_upper,
    int64_t nnz_full,
    int64_t batchSize,
    cudaStream_t stream = 0
);

} // namespace moreau
