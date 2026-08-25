/**
 * @file diff_kkt.hpp
 * @brief KKT system for differentiation
 *
 * The differentiation KKT has a different structure from the forward KKT.
 * For HSDE formulation, the augmented system is:
 *
 * K = [I   J ]
 *     [J' -εI]
 *
 * where J is the Jacobian of the KKT conditions:
 *
 * J = [P      A'     0     q   ]   (n rows)
 *     [A      I     -I    -b   ]   (m rows)
 *     [0      I     -H     0   ]   (m rows)
 *     [c1     c2     0     c3  ]   (1 row)
 *
 * Total J dimension: (n + 2m + 1) × (n + 2m + 1)
 * Augmented K dimension: 2*(n + 2m + 1) × 2*(n + 2m + 1)
 *
 * For adjoint solve: K * [y; lam] = [0; rhs_bar]
 * We extract lam (the second half).
 *
 * === Riccati backward pass (zero+nonneg cones only) ===
 *
 * When the forward solver used Riccati (block-tridiagonal structure), the
 * backward pass can exploit the same structure. The HSDE Jacobian J is:
 *
 *   J*lam = -rhs_bar
 *
 * For zero+nonneg cones, H is diagonal with h[i] ∈ {0,1}. We eliminate
 * λ_w and λ_u from J to get a regularized Schur complement system:
 *
 *   M_back = P + εI + A' * diag(h_inv_back) * A
 *
 * where h_inv_back[i] = 1/ε for active constraints (h[i]=1, including all
 * zero cones) and h_inv_back[i] = 0 for inactive nonneg (h[i]=0).
 *
 * This has the SAME block-tridiagonal structure as the forward KKT and
 * can be factored/solved using the existing Riccati recursion.
 *
 * The τ variable is handled via bordering (2 Riccati solves).
 */

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudss.h>

#include "moreau/kkt/cudss_compat.hpp"
#include <cstdint>
#include <memory>
#include <vector>

#include "moreau/cuda/utils.hpp"
#include "moreau/matrix/csr.hpp"
#include "moreau/vector/vector.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"

namespace moreau {

// Forward declarations
struct RiccatiKKTData;
struct DiffRiccatiData;

/**
 * @brief KKT system for HSDE differentiation
 *
 * This class manages the construction and solution of the augmented
 * KKT system used in backward differentiation for HSDE formulation.
 *
 * Two solver backends:
 * - cuDSS: General purpose, works for all cone types
 * - Riccati: Specialized for block-tridiagonal structure (zero+nonneg cones)
 */
struct DiffKKT {
    CSR KKT;                           // Augmented KKT matrix (upper triangle, cuDSS only)
    int64_t n = 0;                     // Number of primal variables
    int64_t m = 0;                     // Number of constraints
    int64_t jdim = 0;                  // J dimension = n + 2m + 1 + 2*numSparseSoc
    int64_t augdim = 0;                // Augmented dimension = 2 * jdim
    int64_t batchSize = 0;
    KKTSolverType solverType_ = KKTSolverType::CuDSS;

    // === cuDSS-specific data ===

    // Index maps for populating J blocks
    device_unique_ptr<int64_t> P_idx_;      // P block indices in KKT
    device_unique_ptr<int64_t> P_val_idx_;  // Which P value to use (for upper-tri only)
    device_unique_ptr<int64_t> A_idx_;      // A block indices in KKT
    device_unique_ptr<int64_t> A_val_idx_;  // Which A value to use
    device_unique_ptr<int64_t> At_idx_;     // A' block indices in KKT
    device_unique_ptr<int64_t> At_val_idx_; // Which A value to use for each A' position
    device_unique_ptr<int64_t> H_diag_idx_; // H diagonal indices
    device_unique_ptr<int64_t> H_soc_idx_;  // SOC H block indices (dim*dim per cone, variable dim)
    device_unique_ptr<int64_t> H_exp_idx_;  // Exp H block indices
    device_unique_ptr<int64_t> H_power_idx_;// Power H block indices
    device_unique_ptr<int64_t> H_psd_idx_;  // PSD H block indices (svec_dim*svec_dim per cone)
    device_unique_ptr<int64_t> d_psd_Hs_offsets_;   // [numPsdCones+1] prefix sum of upper-tri entries
    device_unique_ptr<int64_t> d_psd_kkt_offsets_;  // [numPsdCones+1] prefix sum of svec_dim*svec_dim
    device_unique_ptr<int64_t> d_psd_svec_dims_;    // [numPsdCones] svec dimension per cone
    int64_t numPsdCones_ = 0;
    int64_t totalPsdHsEntries_ = 0;        // Total upper-tri entries for all PSD cones
    int64_t totalPsdKktEntries_ = 0;       // Total svec_dim*svec_dim entries for all PSD cones

    device_unique_ptr<int64_t> q_idx_;      // q column indices
    device_unique_ptr<int64_t> b_idx_;      // b column indices
    device_unique_ptr<int64_t> c1_idx_;     // c1 row indices
    device_unique_ptr<int64_t> c2_idx_;     // c2 row indices
    device_unique_ptr<int64_t> c3_idx_;     // c3 position

    // SOC variable-dim info for H block population
    device_unique_ptr<int64_t> d_soc_dims_;        // [numSocCones] per-cone dim
    device_unique_ptr<int64_t> d_soc_Hs_offsets_;  // [numSocCones+1] prefix sum of upper-tri entries (dense-only)
    device_unique_ptr<int64_t> d_soc_kkt_offsets_;  // [numSocCones+1] prefix sum of dim*dim KKT entries (dense-only)
    int64_t numSocCones_ = 0;
    int64_t totalSocHsEntries_ = 0;       // Dense-only Hs total
    int64_t totalSocKktEntries_ = 0;      // Dense-only dim*dim total

    // Sparse SOC expansion (dim > 4)
    // H-block row entries for sparse cones: 3 per row (diag + v1 col + v2 col)
    device_unique_ptr<int64_t> H_soc_sparse_diag_idx_;  // [totalSparseSocDim] diag KKT idx
    device_unique_ptr<int64_t> H_soc_v1_col_idx_;       // [totalSparseSocDim] v1 expansion col KKT idx
    device_unique_ptr<int64_t> H_soc_v2_col_idx_;       // [totalSparseSocDim] v2 expansion col KKT idx
    // Expansion column entries (-c*v[i] in H-block rows, reversed direction)
    // These are the same positions as v1_col/v2_col, so not stored separately.
    // Expansion row entries at du-columns:
    device_unique_ptr<int64_t> H_soc_exp_v1_du_idx_;    // [totalSparseSocDim] v1 exp row -> du col KKT idx
    device_unique_ptr<int64_t> H_soc_exp_v2_du_idx_;    // [totalSparseSocDim] v2 exp row -> du col KKT idx
    // Expansion row diagonal entries:
    device_unique_ptr<int64_t> H_soc_exp_diag_idx_;     // [2*numSparseSoc] expansion row diag KKT idx
    // Sparse SOC offset array for kernel dispatch
    device_unique_ptr<int64_t> d_soc_sparse_offsets_;   // [numSocCones+1] prefix sum of dim for sparse cones
    device_unique_ptr<int64_t> d_soc_sparse_indices_;   // [numSocCones] sparse cone idx (-1 if dense)
    int64_t numSparseSoc_ = 0;
    int64_t totalSparseSocDim_ = 0;
    int64_t base_jdim_ = 0;   // n + 2m + 1 (without expansion vars)

    // GenPowerCone variable-dim info for H block population (sparse: diagonal + rank-3)
    device_unique_ptr<int64_t> d_genpow_dims_;          // [numGenPowerCones] total dim per cone
    int64_t numGenPowerCones_ = 0;

    // Sparse GenPowerCone expansion (3 expansion columns per cone)
    // H-block row entries: 4 per row (diag + 3 expansion col entries)
    device_unique_ptr<int64_t> H_genpow_sparse_diag_idx_;  // [totalGenpowDim] diag KKT idx
    device_unique_ptr<int64_t> H_genpow_v1_col_idx_;       // [totalGenpowDim] v1 expansion col KKT idx
    device_unique_ptr<int64_t> H_genpow_v2_col_idx_;       // [totalGenpowDim] v2 expansion col KKT idx
    device_unique_ptr<int64_t> H_genpow_v3_col_idx_;       // [totalGenpowDim] v3 expansion col KKT idx
    // Expansion row entries at du-columns
    device_unique_ptr<int64_t> H_genpow_exp_v1_du_idx_;    // [totalGenpowDim] v1 exp row -> du col KKT idx
    device_unique_ptr<int64_t> H_genpow_exp_v2_du_idx_;    // [totalGenpowDim] v2 exp row -> du col KKT idx
    device_unique_ptr<int64_t> H_genpow_exp_v3_du_idx_;    // [totalGenpowDim] v3 exp row -> du col KKT idx
    // Expansion row diagonal entries
    device_unique_ptr<int64_t> H_genpow_exp_diag_idx_;     // [3*numGenPowerCones] expansion row diag KKT idx
    // Offset arrays for kernel dispatch
    device_unique_ptr<int64_t> d_genpow_sparse_offsets_;   // [numGenPowerCones+1] prefix sum of dim per cone
    int64_t totalGenpowDim_ = 0;

    // Sizes for kernel dispatches
    int64_t nnzP_ = 0;
    int64_t nnzA_ = 0;
    int64_t nnzAt_ = 0;

    // Identity block diagonal indices (for top-left I and bottom-right -εI)
    device_unique_ptr<int64_t> I_diag_idx_;
    device_unique_ptr<int64_t> negI_diag_idx_;

    // Identity blocks within J:
    // J[n:n+m, n:n+m] = I
    // J[n:n+m, n+m:n+2m] = -I
    // J[n+m:n+2m, n:n+m] = I
    device_unique_ptr<int64_t> J_I_row1_idx_;     // I in row block 1
    device_unique_ptr<int64_t> J_negI_row1_idx_;  // -I in row block 1
    device_unique_ptr<int64_t> J_I_row2_idx_;     // I in row block 2

    // ========================================================================
    // Direct-x cone augmentation (IFT-direct backward).
    //
    // Direct-x cones extend the HSDE Jacobian J with `xn` extra rows + a
    // matching `du_x` column block. The block layout is:
    //   J cols (after augmentation): [x | w | du_slack | du_x | τ | (slack expansion)]
    //   J rows (after augmentation): [stat | slack feas | slack cone | direct-x cone | hsde | (slack expansion)]
    // The direct-x cone row k (within cone xc, with primal index J_xc[k])
    // encodes  +x[J_xc[k]] − (I − H_x)·du_x = 0,  contributing:
    //   - +1 at the x column J_xc[k]
    //   - +(δ_{kl} − H_x[k,l]) entries in the du_x columns of cone xc
    // Mirror entries appear in the stationarity rows: for each direct-x
    // index, row J_xc[k] gains −H_x[k,l] entries in du_x cols of cone xc.
    //
    // Storage: nonneg direct-x is diagonal (1 H entry per index), so each
    // index contributes 3 KKT entries (E_J at x col, du_x diagonal, mirror
    // in stationarity row). SOC / PSD direct-x H is dense per cone.
    // ========================================================================

    // Number of direct-x cones and total numel (Σ_xc |J_xc|).
    int64_t numXCones_ = 0;
    int64_t totalXConeNumel_ = 0;
    int64_t totalXNonneg_ = 0;        // total dim across nonneg direct-x cones
    int64_t totalXSocDim_ = 0;        // total dim across SOC direct-x cones (dense path)
    int64_t totalXPsdSvecDim_ = 0;    // total svec dim across PSD direct-x cones
    int64_t totalXExpKkt_ = 0;        // 9 * (count of Exp direct-x cones)
    int64_t totalXPowKkt_ = 0;        // 9 * (count of Power direct-x cones)
    int64_t totalXGenPowKkt_ = 0;     // sum of dim*dim across GenPow direct-x cones (legacy dense)
    int64_t numXGenPowerCones_ = 0;   // count of GenPower direct-x cones (rank-3 sparse path)
    int64_t totalXGenPowDim_ = 0;     // sum of (dim1+dim2) across GenPow direct-x cones

    // Per-cone metadata, kept device-side for kernel dispatch.
    device_unique_ptr<int64_t> d_xcone_kinds_;       // [numXCones_] 0=nonneg, 1=SOC, 2=PSD
    device_unique_ptr<int64_t> d_xcone_dims_;        // [numXCones_] |J_xc|
    device_unique_ptr<int64_t> d_xcone_numel_offsets_; // [numXCones_+1] prefix sum of dims
    device_unique_ptr<int64_t> d_xcone_indices_;     // [totalXConeNumel_] flat J indices
    device_unique_ptr<int64_t> d_xcone_psd_k_;       // [numXCones_] svec k, 0 if not PSD

    // KKT index maps for direct-x:
    //   xcone_E_x_idx_      [totalXConeNumel_] : where the +1 (E_J) entry
    //                       lives in the direct-x row × x col J_xc[k].
    //   xcone_stat_idx_     [totalXConeNumel_ * max_dim_per_cone] flat —
    //                       too irregular; use per-cone packed layouts.
    // For the simplest case (nonneg direct-x is diagonal H_x), each index
    // has exactly one stationarity-row entry and one du_x-row diagonal
    // entry. Pack them:
    //   xcone_nonneg_stat_idx_  [totalXNonneg_]: stat row → du_x col diag entry
    //   xcone_nonneg_du_idx_    [totalXNonneg_]: direct-x row → du_x col diag entry
    //   xcone_nonneg_E_idx_     [totalXNonneg_]: direct-x row → x col J[k] entry
    // For SOC/PSD direct-x H_x is dense per cone; index maps store the
    // |J_xc|² entries that represent the dense block (column-major within
    // cone), one set per cone bucket.
    // Flat index maps + per-entry metadata for direct-x value writes:
    //   xcone_E_idx_   [totalXConeNumel_]    KKT slot for +1 at row J_xc[k]
    //   xcone_du_idx_  [Σ dim²]              KKT slot for du_x col l × direct-x row k
    //   xcone_stat_idx_[Σ dim²]              KKT slot for du_x col l × stat row J_xc[k]
    // Plus packed per-entry meta `(xc_idx, k, [l])` so the kernel can
    // look up `H_x[k, l]` from the cone-kind-specific storage.
    device_unique_ptr<int64_t> xcone_E_idx_;
    device_unique_ptr<int64_t> xcone_du_idx_;
    device_unique_ptr<int64_t> xcone_stat_idx_;
    int64_t numXConeE_   = 0;  // = totalXConeNumel_
    int64_t numXConeDu_  = 0;  // = Σ dim²
    int64_t numXConeStat_ = 0; // = Σ dim²
    // Stat meta `l` (the du_x col local index) — packed parallel to
    // `xcone_stat_idx_`. Stored separately because the other meta slots
    // were already aliased.
    device_unique_ptr<int64_t> xcone_stat_meta_l_;
    device_unique_ptr<int64_t> xcone_nonneg_stat_idx_;
    device_unique_ptr<int64_t> xcone_nonneg_du_idx_;
    device_unique_ptr<int64_t> xcone_nonneg_E_idx_;

    // Dense SOC direct-x: per cone, dim*dim entries each in stat-rows and
    // du_x-rows. Plus dim entries for the E_J +1's.
    device_unique_ptr<int64_t> xcone_soc_stat_idx_;   // [Σ dim²]
    device_unique_ptr<int64_t> xcone_soc_du_idx_;     // [Σ dim²]
    device_unique_ptr<int64_t> xcone_soc_E_idx_;      // [Σ dim] one per index
    device_unique_ptr<int64_t> d_xcone_soc_dim_offsets_;  // [num_x_soc+1] prefix of dim
    device_unique_ptr<int64_t> d_xcone_soc_kkt_offsets_;  // [num_x_soc+1] prefix of dim²
    int64_t num_x_soc_ = 0;
    int64_t total_x_soc_kkt_ = 0;

    // Dense PSD direct-x: per cone, svec_dim*svec_dim entries each in stat
    // rows and du_x-rows.
    device_unique_ptr<int64_t> xcone_psd_stat_idx_;   // [Σ svec_dim²]
    device_unique_ptr<int64_t> xcone_psd_du_idx_;     // [Σ svec_dim²]
    device_unique_ptr<int64_t> xcone_psd_E_idx_;      // [Σ svec_dim] one per index
    device_unique_ptr<int64_t> d_xcone_psd_svec_offsets_;  // [num_x_psd+1] prefix of svec_dim
    device_unique_ptr<int64_t> d_xcone_psd_kkt_offsets_;   // [num_x_psd+1] prefix of svec_dim²
    int64_t num_x_psd_ = 0;
    int64_t total_x_psd_kkt_ = 0;

    // Direct-x GenPow rank-3 sparse expansion KKT slots (mirrors slack
    // `H_genpow_*_idx_`). Each direct-x GenPow cone of dim d emits
    // 4d + 3d + 3 entries instead of d² dense.
    //
    // For each row k in 0..d of cone xc (totalXGenPowDim entries each):
    //   - 1 du-col diag entry at stat row J_xc[k]: H_xcone_genpow_du_stat_diag_idx_
    //   - 1 du-col diag entry at direct-x row xc_row_base+k: H_xcone_genpow_du_dx_diag_idx_
    //   - 3 expansion-col entries at stat row J_xc[k]: H_xcone_genpow_v{1,2,3}_col_stat_idx_
    //   - 3 expansion-col entries at direct-x row xc_row_base+k: H_xcone_genpow_v{1,2,3}_col_dx_idx_
    // For each direct-x expansion row r (3 per cone, totalXGenPowDim entries each):
    //   - d du-col entries: H_xcone_genpow_exp_v{1,2,3}_du_idx_
    // Plus 3*numXGenPowerCones expansion-col self-diag entries.
    device_unique_ptr<int64_t> H_xcone_genpow_du_stat_diag_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_du_dx_diag_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_v1_col_stat_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_v2_col_stat_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_v3_col_stat_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_v1_col_dx_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_v2_col_dx_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_v3_col_dx_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_exp_v1_du_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_exp_v2_du_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_exp_v3_du_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_exp_diag_idx_;
    device_unique_ptr<int64_t> d_xcone_genpow_dim_offsets_;   // [numXGenPow+1] prefix of dim per cone

    // Direct-x SOC rank-2 sparse expansion KKT slots (mirrors slack
    // `H_soc_*_idx_`). Only direct-x SOC cones with dim > 4 use the
    // sparse path; smaller cones stay on the dense `xcone_soc_H` path.
    int64_t numSparseXSoc_ = 0;          // count of direct-x SOC dim > 4
    int64_t totalSparseXSocDim_ = 0;     // sum of dim across sparse direct-x SOC cones
    device_unique_ptr<int64_t> H_xcone_soc_du_stat_diag_idx_; // [totalSparseXSocDim] stat-row du-col diag
    device_unique_ptr<int64_t> H_xcone_soc_du_dx_diag_idx_;   // [totalSparseXSocDim] dx-row du-col diag
    device_unique_ptr<int64_t> H_xcone_soc_v1_col_stat_idx_;  // [totalSparseXSocDim]
    device_unique_ptr<int64_t> H_xcone_soc_v2_col_stat_idx_;  // [totalSparseXSocDim]
    device_unique_ptr<int64_t> H_xcone_soc_v1_col_dx_idx_;    // [totalSparseXSocDim]
    device_unique_ptr<int64_t> H_xcone_soc_v2_col_dx_idx_;    // [totalSparseXSocDim]
    device_unique_ptr<int64_t> H_xcone_soc_exp_v1_du_idx_;    // [totalSparseXSocDim]
    device_unique_ptr<int64_t> H_xcone_soc_exp_v2_du_idx_;    // [totalSparseXSocDim]
    device_unique_ptr<int64_t> H_xcone_soc_exp_diag_idx_;     // [2*numSparseXSoc]
    device_unique_ptr<int64_t> d_xcone_soc_sparse_dim_offsets_; // [numSparseXSoc+1] prefix of dim
    device_unique_ptr<int64_t> d_xcone_soc_sparse_to_xc_;       // [numSparseXSoc] -> position in cones.x_cones
    device_unique_ptr<int64_t> d_xcone_soc_sparse_dims_;        // [numSparseXSoc] dim per cone

    // cuDSS structures
    cudssHandle_t cudss_handle_ = nullptr;
    cudssConfig_t cudss_config_ = nullptr;
    cudssData_t cudss_data_ = nullptr;
    cudssMatrix_t kkt_matrix_ = nullptr;
    cudssMatrix_t rhs_matrix_ = nullptr;
    cudssMatrix_t sol_matrix_ = nullptr;
    bool cudss_initialized_ = false;

    // Work vectors (cuDSS path)
    BatchedVector work_rhs_;   // [augdim]
    BatchedVector work_sol_;   // [augdim]

    // === Riccati-specific data ===
    std::unique_ptr<DiffRiccatiData> diff_riccati_;  // J'J backward pass data

    // Riccati backward work vectors (in combined (x,w) space, size n+m)
    BatchedVector riccati_rhs_;     // [n+m] RHS in combined space
    BatchedVector riccati_bvec_;    // [n+m] bordering vector
    BatchedVector riccati_sol0_;    // [n+m] solution for main RHS
    BatchedVector riccati_sol1_;    // [n+m] solution for bordering
    BatchedVector riccati_scalars_; // [2] per-batch scalars (lam_tau, beta_sm)

    // Cached HSDE data for Riccati backward (needed across updateJ/solveAdjoint)
    const double* cached_q_ = nullptr;
    const double* cached_b_ = nullptr;
    const double* cached_c1_ = nullptr;
    const double* cached_c2_ = nullptr;
    const double* cached_c3_ = nullptr;
    const double* cached_nonneg_H_ = nullptr;
    int64_t numZeroCones_ = 0;
    int64_t numNonnegCones_ = 0;

    /**
     * @brief Construct DiffKKT with problem structure
     *
     * @param n_ Number of primal variables
     * @param m_ Number of constraints
     * @param batch_ Batch size
     * @param P_ro P row offsets (CSR)
     * @param P_ci P column indices (CSR)
     * @param nnzP Number of nonzeros in P
     * @param A_ro A row offsets (CSR)
     * @param A_ci A column indices (CSR)
     * @param nnzA Number of nonzeros in A
     * @param cones Cone structure
     * @param stream CUDA stream
     */
    DiffKKT(int64_t n_, int64_t m_, int64_t batch_,
            const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
            const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
            const Cones& cones,
            KKTSolverType type = KKTSolverType::CuDSS,
            cudaStream_t stream = 0);

    /**
     * @brief Update J blocks with current values
     */
    void updateJ(
        const double* P_values,         // P values [batchSize][nnzP_orig]
        int64_t nnzP_orig,              // Original nnz in P (may include lower triangle)
        const double* A_values,         // A values [batchSize][nnzA_orig]
        int64_t nnzA_orig,              // Original nnz in A
        const double* q,                // q vector [batchSize][n]
        const double* b,                // b vector [batchSize][m]
        const double* c1,               // c1 = -(2/τ)Px - q [batchSize][n]
        const double* c2,               // c2 = -b [batchSize][m]
        const double* c3,               // c3 = x'Px/τ² [batchSize][1]
        const double* nonneg_H,         // Nonneg DΠ diagonals
        const double* soc_H,            // Dense SOC DΠ blocks (dim<=4 only)
        const double* soc_sparse_diag,  // Sparse SOC diagonal [batchSize][totalSparseSocDim]
        const double* soc_sparse_v1,    // Sparse SOC v1 [batchSize][totalSparseSocDim]
        const double* soc_sparse_v2,    // Sparse SOC v2 [batchSize][totalSparseSocDim]
        const double* soc_sparse_c1,    // Sparse SOC c1 [batchSize][numSparseSoc]
        const double* soc_sparse_c2,    // Sparse SOC c2 [batchSize][numSparseSoc]
        const double* exp_H,            // Exp DΠ blocks
        const double* power_H,          // Power DΠ blocks
        const double* psd_H,            // PSD DΠ blocks (upper-tri svec_dim*svec_dim per cone)
        const double* genpow_sparse_diag,  // GenPowerCone diagonal [batchSize][totalGenpowDim]
        const double* genpow_sparse_left1, // GenPowerCone left1 (a) [batchSize][totalGenpowDim]
        const double* genpow_sparse_right1,// GenPowerCone right1 (b) [batchSize][totalGenpowDim]
        const double* genpow_sparse_left2, // GenPowerCone left2 (e) [batchSize][totalGenpowDim]
        const double* genpow_sparse_right2,// GenPowerCone right2 (f) [batchSize][totalGenpowDim]
        const double* genpow_sparse_left3, // GenPowerCone left3 (g) [batchSize][totalGenpowDim]
        const double* genpow_sparse_c3,    // GenPowerCone c_ww [batchSize][numGenPowerCones]
        // Direct-x cone projection Jacobians (IFT-direct path).
        // Pass nullptr (or arbitrary) when `cones.x_cones` is empty.
        const double* xcone_nonneg_H,      // [batchSize][totalXNonneg_]   diagonal
        const double* xcone_soc_H,         // [batchSize][total_x_soc_kkt_] dense per cone
        const double* xcone_psd_H,         // [batchSize][total_x_psd_kkt_] dense per cone
        const double* xcone_exp_H,         // [batchSize][totalXExpKkt_]    dense 3*3 per cone
        const double* xcone_pow_H,         // [batchSize][totalXPowKkt_]    dense 3*3 per cone
        const double* xcone_genpow_H,      // [batchSize][totalXGenPowKkt_] dense dim*dim per cone (legacy)
        // Direct-x GenPow rank-3 stripes (used by sparse expansion path).
        const double* xcone_genpow_rank3_diag,
        const double* xcone_genpow_rank3_left1,
        const double* xcone_genpow_rank3_right1,
        const double* xcone_genpow_rank3_left2,
        const double* xcone_genpow_rank3_right2,
        const double* xcone_genpow_rank3_left3,
        const double* xcone_genpow_rank3_c3,
        // Direct-x SOC rank-2 stripes (used for SOC dim > 4).
        const double* xcone_soc_rank2_diag,
        const double* xcone_soc_rank2_v1,
        const double* xcone_soc_rank2_v2,
        const double* xcone_soc_rank2_c1,
        const double* xcone_soc_rank2_c2,
        const Cones& cones,
        cudaStream_t stream = 0
    );

    /**
     * @brief Factorize the augmented KKT matrix
     */
    void factor(cudaStream_t stream = 0);

    /**
     * @brief Solve the adjoint system
     *
     * Solves K * [y; lam] = [0; rhs_bar] and returns y
     */
    void solveAdjoint(
        const double* rhs_bar,
        double* lam,
        cudaStream_t stream = 0
    );

    /**
     * @brief Solve the forward system
     *
     * Solves K * [y; sol] = [rhs; 0] and returns sol
     */
    void solveForward(
        const double* rhs,
        double* sol,
        cudaStream_t stream = 0
    );

    // Non-copyable, non-movable (always used through unique_ptr)
    DiffKKT(const DiffKKT&) = delete;
    DiffKKT& operator=(const DiffKKT&) = delete;
    DiffKKT(DiffKKT&&) = delete;
    DiffKKT& operator=(DiffKKT&&) = delete;

    ~DiffKKT();

    [[nodiscard]] size_t memoryUsage() const noexcept;

    // Accessors for direct-x SOC rank-2 sparse path.
    int64_t getNumSparseXSoc() const noexcept { return numSparseXSoc_; }
    int64_t getTotalSparseXSocDim() const noexcept { return totalSparseXSocDim_; }
    const int64_t* getXSocSparseDimOffsets() const noexcept { return d_xcone_soc_sparse_dim_offsets_.get(); }
    const int64_t* getXSocSparseToXc() const noexcept { return d_xcone_soc_sparse_to_xc_.get(); }
    const int64_t* getXSocSparseDims() const noexcept { return d_xcone_soc_sparse_dims_.get(); }

private:
    void initialize_cudss(cudaStream_t stream);
    void cleanup_cudss();

    // Riccati backward pass methods
    void initialize_riccati(
        int64_t n_, int64_t m_, int64_t batch_,
        const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
        const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
        const Cones& cones,
        cudaStream_t stream);
    void updateJ_riccati(
        const double* P_values, int64_t nnzP_orig,
        const double* A_values, int64_t nnzA_orig,
        const double* nonneg_H, const Cones& cones,
        cudaStream_t stream);
    void factor_riccati(cudaStream_t stream);
    void solveAdjoint_riccati(
        const double* rhs_bar, double* lam, cudaStream_t stream);
};

} // namespace moreau
