/**
 * @file diff.hpp
 * @brief GPU-based differentiation for conic optimization
 *
 * This module provides batched forward and backward (adjoint) differentiation
 * for conic optimization problems solved by the moreau CUDA solver.
 *
 * The differentiation is based on implicit differentiation of the KKT conditions
 * using the Homogeneous Self-Dual Embedding (HSDE) formulation.
 *
 * Mathematical Background:
 * ========================
 *
 * The conic problem is:
 *   minimize    (1/2)x'Px + q'x
 *   subject to  Ax + s = b
 *               x ∈ K1,  s ∈ K2
 * K2 constrains the slack s; K1 constrains x directly (direct-x cones).
 *
 * At optimality, the KKT conditions hold. By differentiating these conditions
 * with respect to problem parameters, we can compute:
 *
 * 1. Forward differentiation: Given perturbations (dP, dq, dA, db), compute
 *    how the solution (dx, dz, ds) changes.
 *
 * 2. Backward (adjoint) differentiation: Given upstream gradients (dx_bar,
 *    dz_bar, ds_bar), compute gradients w.r.t. problem data (dP, dq, dA, db).
 *
 * Two formulations are supported:
 * - QP Equality: For problems with only zero cones (equality constraints)
 * - HSDE General: For problems with general cones (nonneg, SOC, exp, power)
 *
 * Implementation Notes:
 * ====================
 * - All operations are batched (one problem per batch item)
 * - No host-device transfers in the differentiation loop
 * - Memory is pre-allocated at DiffState construction
 * - Uses cuDSS for sparse linear solves (shared with forward solver)
 */

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include "moreau/vector/vector.hpp"
#include "moreau/matrix/csr.hpp"
#include "moreau/cones/cones.hpp"

namespace moreau {

// Forward declarations
struct CompiledSolver;
struct KKTData;

/**
 * @brief Cone derivative block representation
 *
 * Represents the derivative DΠ_K*(u) of the dual cone projection at point u.
 * Different cone types have different derivative structures:
 * - Zero cone: 0 (no derivative)
 * - Nonnegative: Diagonal (0 or 1 depending on sign of u)
 * - SOC: Dense dim x dim matrix (variable dimension, stored as upper triangle)
 * - Exp/Power: Dense 3x3 matrix
 */
enum class ConeDerivType {
    Zero,      // H = 0 (zero cone)
    Diagonal,  // H is diagonal (nonnegative cone)
    DenseNxN,  // H is dense NxN (SOC cones, variable dimension)
    Dense3x3   // H is dense 3x3 (exp, power cones)
};

/**
 * @brief Cached state for efficient gradient computation
 *
 * Stores the solution and intermediate data from the forward solve
 * that is needed for backward differentiation. This avoids recomputing
 * during the backward pass.
 *
 * Memory layout (all batched):
 * - x, z, s: Solution variables [batchSize][n/m]
 * - tau: Homogenization scalar [batchSize][1]
 * - d, e: Equilibration scaling vectors
 * - c: Equilibration cost scaling
 * - P_values, A_values: Equilibrated matrix values
 * - q, b: Equilibrated vectors
 */
struct DiffState {
    int64_t n;          // Number of primal variables
    int64_t m;          // Number of constraints
    int64_t batchSize;  // Batch size

    // Cached solution (equilibrated space)
    BatchedVector x;      // [batchSize][n]
    BatchedVector z;      // [batchSize][m]
    BatchedVector s;      // [batchSize][m]
    BatchedVector tau;    // [batchSize][1]

    // Cached u = z - s for cone derivative
    BatchedVector u;      // [batchSize][m]

    // Cached cone projection Π_K*(u)
    BatchedVector pi_u;   // [batchSize][m]

    // Cached direct-x dual `z_x` in the solver's equilibrated frame.
    // length = total direct-x numel (Σ |J_xc|), or 1 when there are no
    // direct-x cones (placeholder allocation).
    BatchedVector z_x;    // [batchSize][totalXConeNumel or 1]
    int64_t totalXConeNumel = 0;

    // Cached equilibration factors
    BatchedVector d;      // [batchSize][n] - column scaling
    BatchedVector dinv;   // [batchSize][n] - inverse column scaling
    BatchedVector e;      // [batchSize][m] - row scaling
    BatchedVector einv;   // [batchSize][m] - inverse row scaling
    BatchedVector c_scale; // [batchSize][1] - cost scaling

    // Work vectors for differentiation solve
    BatchedVector work_n;     // [batchSize][n]
    BatchedVector work_m;     // [batchSize][m]
    BatchedVector work_m2;    // [batchSize][m]
    BatchedVector rhs;        // [batchSize][n + 2m + 1] (for HSDE)
    BatchedVector sol;        // [batchSize][n + 2m + 1] (for HSDE)

    // Work vectors for HSDE coefficients
    BatchedVector c1;         // [batchSize][n]
    BatchedVector c2;         // [batchSize][m]
    BatchedVector c3;         // [batchSize][1]

    // Smoothed differentiation: cached central-path iterate
    BatchedVector smoothing_x;   // [batchSize][n]
    BatchedVector smoothing_z;   // [batchSize][m]
    BatchedVector smoothing_s;   // [batchSize][m]
    BatchedVector smoothing_mu;  // [batchSize][1]
    bool smoothing_cached = false;

    // Output gradient storage (CSR-ordered values matching P/A sparsity)
    BatchedVector dP_values;  // [batchSize][nnzP]
    BatchedVector dq;         // [batchSize][n]
    BatchedVector dA_values;  // [batchSize][nnzA]
    BatchedVector db;         // [batchSize][m]

    /**
     * @brief Construct DiffState with pre-allocated memory
     */
    DiffState(
        int64_t n_, int64_t m_, int64_t batchSize_,
        int64_t nnzP, int64_t nnzA,
        cudaStream_t stream = 0,
        int64_t totalXConeNumel_ = 0
    );

    /// Resize rhs/sol/z_x to accommodate `xn` direct-x slots if needed.
    /// Idempotent: a no-op when the buffers are already large enough.
    void resize_for_xcones(int64_t xn, cudaStream_t stream = 0);

    // No copy
    DiffState(const DiffState&) = delete;
    DiffState& operator=(const DiffState&) = delete;

    // Move ok
    DiffState(DiffState&&) noexcept = default;
    DiffState& operator=(DiffState&&) noexcept = default;

    /**
     * @brief Calculate memory usage in bytes
     */
    [[nodiscard]] size_t memoryUsage() const noexcept;
};

/**
 * @brief Cone derivative data for differentiation
 *
 * Stores the derivative matrices DΠ_K*(u) for all cones.
 * Uses the same packed upper-triangle storage as the KKT H block.
 */
struct ConeDerivatives {
    int64_t batchSize;

    // Nonnegative cone derivatives (diagonal)
    // H[i] = 1 if u[i] > 0, 0 otherwise
    BatchedVector nonneg_H;  // [batchSize][numNonnegCones]

    // Dense SOC cone derivatives (dim <= 4 only)
    // Upper triangle = dim*(dim+1)/2 elements per cone
    // Total size = totalDenseSocHsEntries (sum over dim<=4 cones only)
    BatchedVector soc_H;     // [batchSize][totalDenseSocHsEntries]

    // Sparse SOC cone derivatives (dim > 4): diagonal + rank-2
    // H = diag(d) + c1 * v1 * v1^T + c2 * v2 * v2^T
    BatchedVector soc_sparse_diag;  // [batchSize][totalSparseSocDim]
    BatchedVector soc_sparse_v1;    // [batchSize][totalSparseSocDim]
    BatchedVector soc_sparse_v2;    // [batchSize][totalSparseSocDim]
    BatchedVector soc_sparse_c1;    // [batchSize][numSparseSoc]
    BatchedVector soc_sparse_c2;    // [batchSize][numSparseSoc]
    int64_t totalDenseSocHsEntries = 0;
    int64_t totalSparseSocDim = 0;

    // Dense-only Hs offset array for dense kernel dispatch
    // Prefix sum of dim*(dim+1)/2 for dim<=4 cones only (0 for sparse cones)
    int64_t* d_dense_soc_Hs_offsets = nullptr;

    // Exp cone derivatives (dense 3x3, full matrix = 9 elements)
    // Exp cone is NOT self-dual, so derivative is NOT symmetric
    BatchedVector exp_H;     // [batchSize][numExpCones * 9]

    // Power cone derivatives (dense 3x3, full matrix = 9 elements)
    // Power cone is NOT self-dual, so derivative is NOT symmetric
    BatchedVector power_H;   // [batchSize][numPowerCones * 9]

    // PSD cone derivatives (dense svec_dim×svec_dim, upper triangle)
    // PSD cone IS self-dual, so derivative IS symmetric
    BatchedVector psd_H;     // [batchSize][totalPsdHsEntries]
    int64_t totalPsdHsEntries = 0;

    // PSD eigendecomposition cache (shared between projection and derivative)
    // Projection computes eigendecomp, derivative reuses it.
    BatchedVector psd_eigvals;   // [batchSize][totalPsdMatDim] - eigenvalues
    BatchedVector psd_eigvecs;   // [batchSize][totalPsdMatSqDim] - eigenvectors (column-major)
    BatchedVector psd_omega;     // [batchSize][totalPsdMatSqDim] - Ω matrix workspace
    BatchedVector psd_work_mat;  // [batchSize][totalPsdMatSqDim] - temp matrix workspace
    BatchedVector psd_work_mat2; // [batchSize][totalPsdMatSqDim] - temp matrix workspace
    BatchedVector psd_work_svec; // [batchSize][totalPsdSvecDim] - temp svec workspace
    int64_t totalPsdMatDim = 0;
    int64_t totalPsdMatSqDim = 0;
    int64_t totalPsdSvecDim = 0;

    // Pre-allocated cuSOLVER workspace for eigendecomp (sized for largest PSD cone)
    double* d_psd_cusolver_work = nullptr;
    int d_psd_cusolver_work_size = 0;
    int* d_psd_info = nullptr;  // cuSOLVER info output

    // ===== Direct-x cone derivative storage (IFT-direct backward) =====
    // Same per-cone-kind layout as the slack arrays above, but indexed
    // by direct-x cone position. `xcone_nonneg_H` is the diagonal H
    // value per direct-x nonneg index (length = total direct-x nonneg
    // dim). `xcone_soc_H` and `xcone_psd_H` store dense `dim×dim`
    // (row-major) blocks per direct-x SOC / PSD cone, indexed by the
    // `h_off` offsets the DiffKKT computes.
    BatchedVector xcone_nonneg_H;
    BatchedVector xcone_soc_H;
    BatchedVector xcone_psd_H;
    // Asymmetric direct-x cones: same dense per-cone layout as slack
    // (Exp/Power: 3*3 row-major; GenPow: variable dim*dim row-major).
    // The CPU dispatch goes through `derivative_cone_sparse(u_x, slack_kind, dual=true)`
    // — the asymmetric primal/dual swap matters only for the FORWARD scaling.
    BatchedVector xcone_exp_H;
    BatchedVector xcone_pow_H;
    BatchedVector xcone_genpow_H;
    int64_t totalXNonneg = 0;
    int64_t totalXSocKkt = 0;
    int64_t totalXPsdKkt = 0;
    int64_t totalXExpKkt = 0;
    int64_t totalXPowKkt = 0;
    int64_t totalXGenPowKkt = 0;

    // GenPowerCone derivatives: sparse decomposition (diagonal + rank-3)
    // H = diag + left1*right1^T + left2*right2^T + c3*left3*left3^T
    BatchedVector genpow_sparse_diag;    // [batchSize][totalGenpowDim]
    BatchedVector genpow_sparse_left1;   // [batchSize][totalGenpowDim]
    BatchedVector genpow_sparse_right1;  // [batchSize][totalGenpowDim]
    BatchedVector genpow_sparse_left2;   // [batchSize][totalGenpowDim]
    BatchedVector genpow_sparse_right2;  // [batchSize][totalGenpowDim]
    BatchedVector genpow_sparse_left3;   // [batchSize][totalGenpowDim]
    BatchedVector genpow_sparse_c3;      // [batchSize][numGenPowerCones]
    int64_t totalGenpowDim = 0;

    // GenPowerCone workspace for projection and derivative kernels
    // Replaces per-thread stack arrays to support arbitrary dimensions.
    BatchedVector genpow_diff_work_vec;   // [batchSize][totalGenPowerDim] for neg_v/xi scratch
    BatchedVector genpow_diff_work_dim1;  // [batchSize][7 * totalGenPowerAlphas] for 7 dim1-sized arrays

    // Same shape as `genpow_diff_work_*` above but sized for the
    // direct-x GenPow cones (`Cones.totalXGenPowerDim` /
    // `totalXGenPowerAlphas`). Used by `compute_xcone_genpow_H` to host
    // the Newton iterate xi, the per-pi-star intermediates, and the
    // rank-3 decomposition vectors before expansion to dense H_x.
    BatchedVector xcone_genpow_diff_work_vec;
    BatchedVector xcone_genpow_diff_work_dim1;
    // Rank-3 decomposition scratch for direct-x GenPow backward. The
    // slack-side `genpow_sparse_*` arrays serve the same role; we keep
    // the direct-x copies separate so a problem with both slack and
    // direct-x GenPow cones does not have its scratch buffers overlap.
    BatchedVector xcone_genpow_rank3_diag;   // [batch][totalXGenPowerDim]
    BatchedVector xcone_genpow_rank3_left1;  // [batch][totalXGenPowerDim]
    BatchedVector xcone_genpow_rank3_right1; // [batch][totalXGenPowerDim]
    BatchedVector xcone_genpow_rank3_left2;  // [batch][totalXGenPowerDim]
    BatchedVector xcone_genpow_rank3_right2; // [batch][totalXGenPowerDim]
    BatchedVector xcone_genpow_rank3_left3;  // [batch][totalXGenPowerDim]
    BatchedVector xcone_genpow_rank3_c3;     // [batch][numXGenPowerCones]

    // Direct-x SOC rank-2 stripes (used by sparse expansion path,
    // dim > 4). Mirrors slack `soc_sparse_*` family.
    BatchedVector xcone_soc_rank2_diag;  // [batch][totalSparseXSocDim]
    BatchedVector xcone_soc_rank2_v1;    // [batch][totalSparseXSocDim]
    BatchedVector xcone_soc_rank2_v2;    // [batch][totalSparseXSocDim]
    BatchedVector xcone_soc_rank2_c1;    // [batch][numSparseXSoc]
    BatchedVector xcone_soc_rank2_c2;    // [batch][numSparseXSoc]
    int64_t totalSparseXSocDim = 0;
    int64_t numSparseXSoc = 0;

    ConeDerivatives(const Cones& cones, int64_t batchSize_, cudaStream_t stream = 0);

    ~ConeDerivatives() {
        if (d_dense_soc_Hs_offsets) {
            cudaFree(d_dense_soc_Hs_offsets);
            d_dense_soc_Hs_offsets = nullptr;
        }
        if (d_psd_cusolver_work) {
            cudaFree(d_psd_cusolver_work);
            d_psd_cusolver_work = nullptr;
        }
        if (d_psd_info) {
            cudaFree(d_psd_info);
            d_psd_info = nullptr;
        }
    }

    // No copy
    ConeDerivatives(const ConeDerivatives&) = delete;
    ConeDerivatives& operator=(const ConeDerivatives&) = delete;

    // Move: transfer ownership of d_dense_soc_Hs_offsets
    ConeDerivatives(ConeDerivatives&& other) noexcept
        : batchSize(other.batchSize),
          nonneg_H(std::move(other.nonneg_H)),
          soc_H(std::move(other.soc_H)),
          soc_sparse_diag(std::move(other.soc_sparse_diag)),
          soc_sparse_v1(std::move(other.soc_sparse_v1)),
          soc_sparse_v2(std::move(other.soc_sparse_v2)),
          soc_sparse_c1(std::move(other.soc_sparse_c1)),
          soc_sparse_c2(std::move(other.soc_sparse_c2)),
          totalDenseSocHsEntries(other.totalDenseSocHsEntries),
          totalSparseSocDim(other.totalSparseSocDim),
          d_dense_soc_Hs_offsets(other.d_dense_soc_Hs_offsets),
          exp_H(std::move(other.exp_H)),
          power_H(std::move(other.power_H)),
          psd_H(std::move(other.psd_H)),
          totalPsdHsEntries(other.totalPsdHsEntries),
          psd_eigvals(std::move(other.psd_eigvals)),
          psd_eigvecs(std::move(other.psd_eigvecs)),
          psd_omega(std::move(other.psd_omega)),
          psd_work_mat(std::move(other.psd_work_mat)),
          psd_work_mat2(std::move(other.psd_work_mat2)),
          psd_work_svec(std::move(other.psd_work_svec)),
          totalPsdMatDim(other.totalPsdMatDim),
          totalPsdMatSqDim(other.totalPsdMatSqDim),
          totalPsdSvecDim(other.totalPsdSvecDim),
          d_psd_cusolver_work(other.d_psd_cusolver_work),
          d_psd_cusolver_work_size(other.d_psd_cusolver_work_size),
          d_psd_info(other.d_psd_info),
          genpow_sparse_diag(std::move(other.genpow_sparse_diag)),
          genpow_sparse_left1(std::move(other.genpow_sparse_left1)),
          genpow_sparse_right1(std::move(other.genpow_sparse_right1)),
          genpow_sparse_left2(std::move(other.genpow_sparse_left2)),
          genpow_sparse_right2(std::move(other.genpow_sparse_right2)),
          genpow_sparse_left3(std::move(other.genpow_sparse_left3)),
          genpow_sparse_c3(std::move(other.genpow_sparse_c3)),
          totalGenpowDim(other.totalGenpowDim),
          genpow_diff_work_vec(std::move(other.genpow_diff_work_vec)),
          genpow_diff_work_dim1(std::move(other.genpow_diff_work_dim1)),
          xcone_genpow_diff_work_vec(std::move(other.xcone_genpow_diff_work_vec)),
          xcone_genpow_diff_work_dim1(std::move(other.xcone_genpow_diff_work_dim1)),
          xcone_genpow_rank3_diag(std::move(other.xcone_genpow_rank3_diag)),
          xcone_genpow_rank3_left1(std::move(other.xcone_genpow_rank3_left1)),
          xcone_genpow_rank3_right1(std::move(other.xcone_genpow_rank3_right1)),
          xcone_genpow_rank3_left2(std::move(other.xcone_genpow_rank3_left2)),
          xcone_genpow_rank3_right2(std::move(other.xcone_genpow_rank3_right2)),
          xcone_genpow_rank3_left3(std::move(other.xcone_genpow_rank3_left3)),
          xcone_genpow_rank3_c3(std::move(other.xcone_genpow_rank3_c3)),
          xcone_soc_rank2_diag(std::move(other.xcone_soc_rank2_diag)),
          xcone_soc_rank2_v1(std::move(other.xcone_soc_rank2_v1)),
          xcone_soc_rank2_v2(std::move(other.xcone_soc_rank2_v2)),
          xcone_soc_rank2_c1(std::move(other.xcone_soc_rank2_c1)),
          xcone_soc_rank2_c2(std::move(other.xcone_soc_rank2_c2)),
          totalSparseXSocDim(other.totalSparseXSocDim),
          numSparseXSoc(other.numSparseXSoc),
          xcone_nonneg_H(std::move(other.xcone_nonneg_H)),
          xcone_soc_H(std::move(other.xcone_soc_H)),
          xcone_psd_H(std::move(other.xcone_psd_H)),
          xcone_exp_H(std::move(other.xcone_exp_H)),
          xcone_pow_H(std::move(other.xcone_pow_H)),
          xcone_genpow_H(std::move(other.xcone_genpow_H)),
          totalXNonneg(other.totalXNonneg),
          totalXSocKkt(other.totalXSocKkt),
          totalXPsdKkt(other.totalXPsdKkt),
          totalXExpKkt(other.totalXExpKkt),
          totalXPowKkt(other.totalXPowKkt),
          totalXGenPowKkt(other.totalXGenPowKkt)
    {
        other.d_dense_soc_Hs_offsets = nullptr;
        other.d_psd_cusolver_work = nullptr;
        other.d_psd_info = nullptr;
    }
    ConeDerivatives& operator=(ConeDerivatives&& other) noexcept {
        if (this != &other) {
            if (d_dense_soc_Hs_offsets) cudaFree(d_dense_soc_Hs_offsets);
            if (d_psd_cusolver_work) cudaFree(d_psd_cusolver_work);
            if (d_psd_info) cudaFree(d_psd_info);
            // Move all members
            batchSize = other.batchSize;
            nonneg_H = std::move(other.nonneg_H);
            soc_H = std::move(other.soc_H);
            soc_sparse_diag = std::move(other.soc_sparse_diag);
            soc_sparse_v1 = std::move(other.soc_sparse_v1);
            soc_sparse_v2 = std::move(other.soc_sparse_v2);
            soc_sparse_c1 = std::move(other.soc_sparse_c1);
            soc_sparse_c2 = std::move(other.soc_sparse_c2);
            totalDenseSocHsEntries = other.totalDenseSocHsEntries;
            totalSparseSocDim = other.totalSparseSocDim;
            d_dense_soc_Hs_offsets = other.d_dense_soc_Hs_offsets;
            other.d_dense_soc_Hs_offsets = nullptr;
            exp_H = std::move(other.exp_H);
            power_H = std::move(other.power_H);
            psd_H = std::move(other.psd_H);
            totalPsdHsEntries = other.totalPsdHsEntries;
            psd_eigvals = std::move(other.psd_eigvals);
            psd_eigvecs = std::move(other.psd_eigvecs);
            psd_omega = std::move(other.psd_omega);
            psd_work_mat = std::move(other.psd_work_mat);
            psd_work_mat2 = std::move(other.psd_work_mat2);
            psd_work_svec = std::move(other.psd_work_svec);
            totalPsdMatDim = other.totalPsdMatDim;
            totalPsdMatSqDim = other.totalPsdMatSqDim;
            totalPsdSvecDim = other.totalPsdSvecDim;
            d_psd_cusolver_work = other.d_psd_cusolver_work;
            d_psd_cusolver_work_size = other.d_psd_cusolver_work_size;
            d_psd_info = other.d_psd_info;
            other.d_psd_cusolver_work = nullptr;
            other.d_psd_info = nullptr;
            genpow_sparse_diag = std::move(other.genpow_sparse_diag);
            genpow_sparse_left1 = std::move(other.genpow_sparse_left1);
            genpow_sparse_right1 = std::move(other.genpow_sparse_right1);
            genpow_sparse_left2 = std::move(other.genpow_sparse_left2);
            genpow_sparse_right2 = std::move(other.genpow_sparse_right2);
            genpow_sparse_left3 = std::move(other.genpow_sparse_left3);
            genpow_sparse_c3 = std::move(other.genpow_sparse_c3);
            totalGenpowDim = other.totalGenpowDim;
            genpow_diff_work_vec = std::move(other.genpow_diff_work_vec);
            genpow_diff_work_dim1 = std::move(other.genpow_diff_work_dim1);
            xcone_genpow_diff_work_vec = std::move(other.xcone_genpow_diff_work_vec);
            xcone_genpow_diff_work_dim1 = std::move(other.xcone_genpow_diff_work_dim1);
            xcone_genpow_rank3_diag = std::move(other.xcone_genpow_rank3_diag);
            xcone_genpow_rank3_left1 = std::move(other.xcone_genpow_rank3_left1);
            xcone_genpow_rank3_right1 = std::move(other.xcone_genpow_rank3_right1);
            xcone_genpow_rank3_left2 = std::move(other.xcone_genpow_rank3_left2);
            xcone_genpow_rank3_right2 = std::move(other.xcone_genpow_rank3_right2);
            xcone_genpow_rank3_left3 = std::move(other.xcone_genpow_rank3_left3);
            xcone_genpow_rank3_c3 = std::move(other.xcone_genpow_rank3_c3);
            xcone_soc_rank2_diag = std::move(other.xcone_soc_rank2_diag);
            xcone_soc_rank2_v1 = std::move(other.xcone_soc_rank2_v1);
            xcone_soc_rank2_v2 = std::move(other.xcone_soc_rank2_v2);
            xcone_soc_rank2_c1 = std::move(other.xcone_soc_rank2_c1);
            xcone_soc_rank2_c2 = std::move(other.xcone_soc_rank2_c2);
            totalSparseXSocDim = other.totalSparseXSocDim;
            numSparseXSoc = other.numSparseXSoc;
            xcone_nonneg_H = std::move(other.xcone_nonneg_H);
            xcone_soc_H = std::move(other.xcone_soc_H);
            xcone_psd_H = std::move(other.xcone_psd_H);
            xcone_exp_H = std::move(other.xcone_exp_H);
            xcone_pow_H = std::move(other.xcone_pow_H);
            xcone_genpow_H = std::move(other.xcone_genpow_H);
            totalXNonneg = other.totalXNonneg;
            totalXSocKkt = other.totalXSocKkt;
            totalXPsdKkt = other.totalXPsdKkt;
            totalXExpKkt = other.totalXExpKkt;
            totalXPowKkt = other.totalXPowKkt;
            totalXGenPowKkt = other.totalXGenPowKkt;
        }
        return *this;
    }
};

// ============================================================================
// Differentiation Functions
// ============================================================================

/**
 * @brief Compute cone projection Π_K*(u)
 *
 * Projects u onto the dual cone K*.
 * - Zero cone: Π_K*(u) = u (identity)
 * - Nonneg cone: Π_K*(u) = max(u, 0)
 * - SOC: Π_K*(u) = standard SOC projection
 * - Exp: Π_K*(u) = exponential cone dual projection
 * - Power: Π_K*(u) = power cone dual projection
 *
 * @param u Input vector [batchSize][m]
 * @param pi_u Output projection [batchSize][m]
 * @param cones Cone structure
 * @param stream CUDA stream
 */
void compute_cone_projection(
    const BatchedVector& u,
    BatchedVector& pi_u,
    const Cones& cones,
    cudaStream_t stream = 0,
    double* genpow_work_vec = nullptr,
    int64_t totalGenPowerDim = 0
);

/**
 * @brief Compute cone derivative DΠ_K*(u)
 *
 * Computes the Jacobian of the dual cone projection at point u.
 * - Zero cone: DΠ = I (identity)
 * - Nonneg cone: DΠ = diag(u > 0)
 * - SOC: DΠ = dense dim×dim matrix (variable dimension per cone)
 * - Exp: DΠ = dense 3x3 matrix
 * - Power: DΠ = dense 3x3 matrix
 *
 * @param u Input vector [batchSize][m]
 * @param derivs Output derivatives (pre-allocated)
 * @param cones Cone structure
 * @param stream CUDA stream
 */
void compute_cone_derivative(
    const BatchedVector& u,
    ConeDerivatives& derivs,
    const Cones& cones,
    cudaStream_t stream = 0,
    double* genpow_work_vec = nullptr,
    int64_t totalGenPowerDim = 0,
    double* genpow_work_dim1 = nullptr,
    int64_t totalGenPowerAlphas = 0
);

/**
 * @brief Cache solution state for backward pass
 *
 * Saves the current solution and equilibration state for use in backward().
 * Should be called after solve() completes successfully.
 *
 * @param state DiffState to populate
 * @param solver Solver with completed solve
 * @param stream CUDA stream
 */
void cache_solution_for_backward(
    DiffState& state,
    const CompiledSolver& solver,
    cudaStream_t stream = 0
);

/**
 * @brief Backward (adjoint) differentiation
 *
 * Given upstream gradients (dx_bar, dz_bar, ds_bar) w.r.t. the solution,
 * computes gradients w.r.t. the problem data (dP, dq, dA, db).
 *
 * This is the key operation for training neural networks that include
 * conic optimization as a layer.
 *
 * @param state Cached solution state from forward solve
 * @param dx_bar Upstream gradient w.r.t. x [batchSize][n]
 * @param dz_bar Upstream gradient w.r.t. z [batchSize][m]
 * @param ds_bar Upstream gradient w.r.t. s [batchSize][m]
 * @param solver Solver with KKT and data
 * @param stream CUDA stream
 *
 * After call, state.dP_values, state.dq, state.dA_values, state.db
 * contain the computed gradients.
 */
void backward(
    DiffState& state,
    const BatchedVector& dx_bar,
    const BatchedVector& dz_bar,
    const BatchedVector& ds_bar,
    CompiledSolver& solver,
    cudaStream_t stream = 0
);

/**
 * Same as `backward(...)` above, plus an optional upstream gradient on
 * the direct-x cone duals `z_x`. `dz_x_bar` (in user/original frame)
 * has shape `[batchSize, total_xn]`. Pass `nullptr` for slack-only or
 * to skip dz_x backprop (equivalent to the simpler signature).
 */
void backward_with_dz_x(
    DiffState& state,
    const BatchedVector& dx_bar,
    const BatchedVector& dz_bar,
    const BatchedVector& ds_bar,
    const BatchedVector* dz_x_bar,  // nullable
    CompiledSolver& solver,
    cudaStream_t stream = 0
);

/**
 * @brief Forward differentiation
 *
 * Given perturbations to problem data (dP, dq, dA, db), computes
 * the corresponding perturbations to the solution (dx, dz, ds).
 *
 * This is the forward pass of implicit differentiation, computing how
 * the solution changes when problem data is perturbed.
 *
 * @param state Cached solution state from forward solve
 * @param dP_values Perturbation to P values [batchSize][nnzP]
 * @param dq Perturbation to q [batchSize][n]
 * @param dA_values Perturbation to A values [batchSize][nnzA]
 * @param db Perturbation to b [batchSize][m]
 * @param dx Output: solution derivative [batchSize][n]
 * @param dz Output: solution derivative [batchSize][m]
 * @param ds Output: solution derivative [batchSize][m]
 * @param solver Solver with KKT and data (may be modified to cache DiffKKT)
 * @param stream CUDA stream
 */
void forward(
    DiffState& state,
    const BatchedVector& dP_values,
    const BatchedVector& dq,
    const BatchedVector& dA_values,
    const BatchedVector& db,
    BatchedVector& dx,
    BatchedVector& dz,
    BatchedVector& ds,
    CompiledSolver& solver,
    cudaStream_t stream = 0
);

/**
 * @brief Eagerly initialize the DiffKKT for backward pass
 *
 * This function pre-computes the KKT system structure used in
 * backward differentiation. Called during solver construction when
 * enable_grad=true to avoid lazy initialization overhead on first backward.
 *
 * @param solver Solver to initialize diff_kkt on
 * @param stream CUDA stream
 */
void init_diff_kkt(
    CompiledSolver& solver,
    cudaStream_t stream = 0
);

/**
 * @brief Preallocated workspace for backward differentiation
 *
 * Holds all temporary vectors needed during backward() to avoid
 * runtime cudaMalloc calls. Created lazily on first backward() call
 * and reused for subsequent calls.
 */
struct BackwardWorkspace {
    int64_t n;
    int64_t m;
    int64_t batchSize;

    // HSDE path work vectors (equilibrated space - used for cone derivatives)
    BatchedVector x_eq;       // [batchSize][n]
    BatchedVector z_eq;       // [batchSize][m]
    BatchedVector s_eq;       // [batchSize][m]
    BatchedVector u_eq;       // [batchSize][m]
    BatchedVector pi_u_eq;    // [batchSize][m]
    BatchedVector Px_eq;      // [batchSize][n]

    // Equilibrated upstream gradient vectors
    BatchedVector dx_eq;      // [batchSize][n]
    BatchedVector dz_eq;      // [batchSize][m]
    BatchedVector ds_eq;      // [batchSize][m]

    // Smoothed diff: mu computed from equilibrated smoothing iterate
    BatchedVector smoothing_mu_comp;  // [batchSize][1]

    /**
     * @brief Construct workspace with preallocated memory
     *
     * @param n_ Number of primal variables
     * @param m_ Number of constraints
     * @param batchSize_ Batch size
     * @param stream CUDA stream for async allocation
     */
    BackwardWorkspace(int64_t n_, int64_t m_, int64_t batchSize_, cudaStream_t stream = 0);

    // No copy
    BackwardWorkspace(const BackwardWorkspace&) = delete;
    BackwardWorkspace& operator=(const BackwardWorkspace&) = delete;

    // Move ok
    BackwardWorkspace(BackwardWorkspace&&) noexcept = default;
    BackwardWorkspace& operator=(BackwardWorkspace&&) noexcept = default;
};

} // namespace moreau
