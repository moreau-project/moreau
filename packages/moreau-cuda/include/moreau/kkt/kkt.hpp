/**
 * @file kkt.hpp
 * @brief KKT system construction and factorization
 *
 * This module handles the construction, factorization, and solution
 * of the Karush-Kuhn-Tucker (KKT) linear systems that arise in
 * interior-point methods for conic optimization.
 */

#pragma once

#include <cuda_runtime.h>
#include <cudss.h>

#include "moreau/kkt/cudss_compat.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iomanip>
#include "moreau/cuda/utils.hpp"
#include "moreau/matrix/csr.hpp"        // CSR (always-owning)
#include "moreau/vector/vector.hpp"     // BatchedVector
#include "moreau/cones/cones.hpp"   // ConeStructure
#include "moreau/kkt/kkt_kernels.cuh"   // CUDA kernels
#include "moreau/settings/settings.hpp" // CuDSSStrategy
#include "moreau/kkt/kkt_solver.hpp"    // KKTSolver base class

namespace moreau {
// Forward declaration
enum class ScalingStrategy;
}

/*
The KKT system is the quasidefinite (symmetric) matrix:
KKT = [P + eI    A^T   ]
      [   A   -(H + eI)]

We just need to store its upper triangular part

H is block-diagonal:
* for zero cones, H is diagonal
* for nonneg cones, H is diagonal
* for soc cones, H is dim x dim (variable per cone)
* for pow/exp cones, H is 3x3

* for zero cone, H is zero
* for nonneg cones, H is diag(w * w), where lambda = sqrt(s * z), and w = sqrt(s / z)
* for socones, for dense form, we return H = \eta^2 (2*ww^T - J), where J = diag(1,-I).
* for pow, it's a bit more complicated, but not much given its 3x3
* for exp, it's a bit more complicated, but not much given its 3x3

So for example:
* numZeroCones=1, numNonnegCones=1, numSocCones=1, numExpCones=1, numPowCones=1:

[
1 0 0 0 0 0 0 0 0 0 0
0 1 0 0 0 0 0 0 0 0 0
0 0 1 1 1 0 0 0 0 0 0
0 0 1 1 1 0 0 0 0 0 0
0 0 1 1 1 0 0 0 0 0 0
0 0 0 0 0 1 1 1 0 0 0
0 0 0 0 0 1 1 1 0 0 0
0 0 0 0 0 1 1 1 0 0 0
0 0 0 0 0 0 0 0 1 1 1
0 0 0 0 0 0 0 0 1 1 1
0 0 0 0 0 0 0 0 1 1 1
*/

namespace moreau {
struct KKTData : public KKTSolver {
    CSR KKT;                               // (n+m+p)×(n+m+p), upper triangle only; values sized: batchSize*nnz
    device_unique_ptr<int64_t> P_diagIdx_;         // device, length n: indices in KKT.values for diag(P+εI)
    device_unique_ptr<int64_t> H_diagIdx_;         // device, length m+p: indices in KKT.values for diag(-(H+εI)) + expansion diag
    device_unique_ptr<int64_t> Pnnz_idx_;
    device_unique_ptr<int64_t> Annz_idx_;

    // Indices for dense cone blocks - stored as packed upper triangle
    // SOC dense (dim<=4): variable dim per cone, upper triangle entries
    // SOC sparse (dim>4): diagonal-only entries (dim per cone)
    // Exp/Power: 6 indices per cone (3x3 upper triangle)
    device_unique_ptr<int64_t> H_soc_idx_;         // device, length totalSocHsEntries (dense=tri, sparse=diag)
    device_unique_ptr<int64_t> H_exp_idx_;         // device, length numExpCones * 6
    device_unique_ptr<int64_t> H_power_idx_;       // device, length numPowerCones * 6
    device_unique_ptr<int64_t> H_psd_idx_;         // device, length totalPsdHsEntries
    device_unique_ptr<int64_t> H_genpow_idx_;      // device, length totalGenPowerHsEntries (dense: upper-tri, sparse: diag)

    // Sparse SOC expansion column indices (for dim > 4 cones)
    device_unique_ptr<int64_t> H_soc_u_idx_;       // device, length totalSparseSocDim: u column entries
    device_unique_ptr<int64_t> H_soc_v_idx_;       // device, length totalSparseSocDim: v column entries
    device_unique_ptr<int64_t> H_soc_exp_diag_idx_; // device, length 2*numSparseSoc: expansion diagonal entries

    // GenPowerCone sparse expansion column indices (sparse cones only, dim > 4)
    device_unique_ptr<int64_t> H_genpow_q_idx_;    // device, length totalSparseGenPowAlphas: q column entries
    device_unique_ptr<int64_t> H_genpow_r_idx_;    // device, length totalSparseGenPowDim2: r column entries
    device_unique_ptr<int64_t> H_genpow_p_idx_;    // device, length totalSparseGenPowDim: p column entries
    device_unique_ptr<int64_t> H_genpow_exp_diag_idx_; // device, length 9*numSparseGenPow: expansion diagonal entries (q/r/p + 6 PD axes)
    // 6 PD-axis off-diagonal column index arrays. Each array has
    // `totalSparseGenPowDim` entries (one per cone-block row), aligned with
    // H_genpow_p_idx_'s offset structure.
    device_unique_ptr<int64_t> H_genpow_pd_axis_idx_[6];

    // Direct-x cone KKT index map. Mirrors the
    // CPU `LDLDataMap.Hxblocks` — a flat array of KKT.values slots, one
    // per Hs entry. Layout depends on per-cone kind:
    //   - Nonneg: k diagonal entries per cone (kind = Nonneg)
    //   - SOC dense (dim <= 4): k*(k+1)/2 upper-triangle entries per
    //     cone, column-major over sorted_indices
    //   - SOC sparse (dim > 4): k diagonal entries per cone; off-diagonal
    //     coupling lives in H_xcone_u/v_idx_ + H_xcone_exp_diag_idx_
    //     below (rank-2 expansion columns, mirror of slack SOC).
    // Length: totalXConeHsEntries. Nullptr when x_cones is empty.
    device_unique_ptr<int64_t> H_xcone_hs_idx_;

    // Direct-x sparse SOC expansion column indices.
    // Mirror of H_soc_u_idx_ / H_soc_v_idx_ / H_soc_exp_diag_idx_ but
    // keyed on x-cone sparse order rather than slack SOC sparse order.
    //
    // H_xcone_v_idx_[sparse_xsoc_offset[c] + p] — v-column KKT slot for
    //   row sorted_J[p] of the p-th sparse x-cone.
    // H_xcone_u_idx_[...]                       — u-column KKT slot (col = v_col + 1).
    // H_xcone_exp_diag_idx_[2*c .. 2*c+1]       — [v-diag, u-diag] per sparse x-cone.
    // Lengths: totalSparseXSocDim, totalSparseXSocDim, 2*numSparseXSoc.
    device_unique_ptr<int64_t> H_xcone_u_idx_;
    device_unique_ptr<int64_t> H_xcone_v_idx_;
    device_unique_ptr<int64_t> H_xcone_exp_diag_idx_;

    // Direct-x sparse GenPow expansion column indices (dim > 4).
    // Rank-9 expansion: Hs = μ*(D + p·p' - q·q' - r·r') + Σ_k sign_k·coef_k·a_k·a_k'.
    // Dsigns for xcone = [+1, +1, -1] (q,r,p, opposite of slack's [-1,-1,+1])
    // followed by 6 PD-axis dsigns (dynamic, ±1 per axis depending on pd_signs).
    //
    // H_xcone_genpow_q_idx_[genpow_dim1_offset[c] + i]  — q-col slot for row i < dim1
    // H_xcone_genpow_r_idx_[genpow_dim2_offset[c] + j] — r-col slot for row dim1+j
    // H_xcone_genpow_p_idx_[genpow_dim_offset[c] + i]  — p-col slot for all rows
    // H_xcone_genpow_pd_axis_idx_[k][genpow_dim_offset[c] + i] — k-th PD axis slot for row i
    // H_xcone_genpow_exp_diag_idx_[9*c..9*c+8]         — [q,r,p, then 6 PD] per cone
    // Lengths: totalSparseXGenPowAlphas, totalSparseXGenPowDim2,
    //          totalSparseXGenPowDim, totalSparseXGenPowDim each (PD axes),
    //          9*numSparseXGenPow.
    device_unique_ptr<int64_t> H_xcone_genpow_q_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_r_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_p_idx_;
    device_unique_ptr<int64_t> H_xcone_genpow_pd_axis_idx_[6];
    device_unique_ptr<int64_t> H_xcone_genpow_exp_diag_idx_;

    // P-contribution baseline for direct-x cone Hs slots. After populate()
    // writes P values into KKT.values (and εI regularization is applied),
    // init_xcone_px_baseline() snapshots the per-slot value here so the
    // per-iteration refresh_xcone_hs() can keep P static and only add the
    // running Hs update. Allocated lazily when numXCones > 0.
    std::unique_ptr<BatchedVector> xcone_px_baseline_;

    int64_t n = 0, m = 0, p = 0, batchSize = 0;   // p = 2 * numSparseSoc + 9 * numSparseGenPow + 2 * numSparseXSoc + 9 * numSparseXGenPow (extra expansion rows/cols)

    // ========== UBatch strategy: cuDSS structures (single handle for entire batch) ==========
    cudssHandle_t cudss_handle_ = nullptr;
    cudssConfig_t cudss_config_ = nullptr;
    cudssData_t cudss_data_ = nullptr;
    cudssMatrix_t kkt_matrix_ = nullptr;
    cudssMatrix_t rhs_matrix_ = nullptr;
    cudssMatrix_t sol_matrix_ = nullptr;
    // Multi-RHS matrices for batched solves (2 RHS at once)
    cudssMatrix_t rhs2_matrix_ = nullptr;
    cudssMatrix_t sol2_matrix_ = nullptr;
    bool cudss_initialized_ = false;
    int ir_n_steps_ = 0;  // Iterative refinement steps (affects diagonal restore behavior)
    int64_t maxLuNnz_ = -1;  // cuDSS max LU fill-in (-1 = use cuDSS default of 100*nnz)
    bool cudss_pivot_enable_ = false;
    int cudss_ir_steps_ = 0;
    double dynamic_reg_eps_ = 0.0;  // analog of qdldl's regularize_eps (CPU). 0 disables.

    bool populated_ = false;                            // Debug flag: true after populate() called

    // Regularization support
    device_unique_ptr<int8_t> dsigns_;        // device, length n+m+p: expected signs of D in LDL^T (+1 for P, -1 for H, [-1,+1] per sparse SOC)
    device_unique_ptr<int64_t> diag_full_;    // device, length n+m+p: indices of diagonal entries in KKT
    device_unique_ptr<double> work_diag_;     // device, length n+m+p: workspace for diagonal backup/restore
    double diagonal_regularizer_ = 0.0;       // Last regularizer value used

    // Host-based constructor: P/A structure is provided on CPU
    KKTData(int64_t n_, int64_t m_, int64_t batch_,
            const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
            const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
            const Cones& cones,
            int cudss_ir_steps = 0,
            bool cudss_pivot_enable = false,
            cudaStream_t stream = 0,
            int64_t maxLuNnz = -1,
            double dynamic_reg_eps = 0.0)
        : n(n_), m(m_),
          // p = 2 expansion cols/rows per slack sparse SOC cone (dim>4)
          //   + 9 per slack sparse GenPowerCone (dim>4) (3 expansion + 6 PD axes)
          //   + 2 per direct-x sparse SOC cone (dim>4)
          //   + 9 per direct-x sparse GenPow cone (dim>4) (3 expansion + 6 PD axes)
          p(2 * cones.numSparseSoc + 9 * cones.numSparseGenPow +
            [&]{
                int64_t n_sparse_xsoc = 0;
                int64_t n_sparse_xgenpow = 0;
                for (const auto& xc : cones.x_cones) {
                    if (xc.kind == XConeKind::SOC &&
                        xc.indices.size() > 4) ++n_sparse_xsoc;
                    else if (xc.kind == XConeKind::GenPower &&
                             xc.indices.size() > 4) ++n_sparse_xgenpow;
                }
                return 2 * n_sparse_xsoc + 9 * n_sparse_xgenpow;
            }()),
          batchSize(batch_),
          maxLuNnz_(maxLuNnz),
          cudss_pivot_enable_(cudss_pivot_enable),
          cudss_ir_steps_(cudss_ir_steps),
          dynamic_reg_eps_(dynamic_reg_eps)
    {
        if (n < 0 || m < 0 || batch_ <= 0)
            throw std::invalid_argument("KKTData: invalid dimensions");
        if (!P_ro || (nnzP && !P_ci) || !A_ro || (nnzA && !A_ci))
            throw std::invalid_argument("KKTData: null P/A inputs");

        const int64_t N = n + m + p;

        // Build A^T placement cheaply: for each col i, list all (row, csr_idx) pairs with A(row,i) ≠ 0.
        std::vector<std::vector<std::pair<int64_t, int64_t>>> A_col_entries(static_cast<size_t>(n));
        for (int64_t r = 0; r < m; ++r) {
            for (int64_t pp = A_ro[r]; pp < A_ro[r + 1]; ++pp) {
                const int64_t j = A_ci[pp];      // column in A
                if (j >= 0 && j < n) A_col_entries[(size_t)j].push_back({r, pp});
            }
        }

        // First pass: construct columns incrementally and row offsets.
        std::vector<int64_t> rowOff; rowOff.resize((size_t)N + 1, 0);
        std::vector<int64_t> colIdx; colIdx.reserve((size_t)(nnzP + nnzA + n + m + p)); // rough guess

        std::vector<int64_t> Pdiag((size_t)n, -1);
        std::vector<int64_t> Hdiag((size_t)(m + p), -1);  // m constraint diags + p expansion diags
        std::vector<int64_t> Pnnz_idx((size_t)nnzP, -1);
        std::vector<int64_t> Annz_idx((size_t)nnzA, -1);

        // Direct-x cone Pext contributions, indexed per row of the (1,1)
        // block. For each dense SOC x-cone with sorted indices J, row J[p]
        // contributes upper-triangle cols {J[p], J[p+1], ..., J[k-1]}. For
        // nonneg x-cones, row i ∈ J contributes only col i (the diagonal,
        // which overlaps with P's diag or the εI fallback). Sparse SOC
        // x-cones (dim > 4) add diagonal-only rows here; their off-diagonal
        // coupling is carried by the rank-2 expansion cols appended below.
        std::vector<std::vector<int64_t>> xcone_extra_cols((size_t)n);
        for (const auto& xc : cones.x_cones) {
            if (xc.kind == XConeKind::Nonneg) {
                for (int64_t idx : xc.indices) {
                    if (idx < 0 || idx >= n) {
                        throw std::runtime_error(
                            "KKTData: direct-x cone index " + std::to_string(idx) +
                            " out of range [0, " + std::to_string(n) + ")");
                    }
                    xcone_extra_cols[(size_t)idx].push_back(idx);
                }
            } else if (xc.kind == XConeKind::PSD) {
                // PSD direct-x: always dense svec_dim × svec_dim upper-tri
                // footprint over sorted indices J (regardless of size).
                const int64_t svec_dim = static_cast<int64_t>(xc.indices.size());
                std::vector<int64_t> sorted_J = xc.indices;
                std::sort(sorted_J.begin(), sorted_J.end());
                for (int64_t p = 0; p < svec_dim; ++p) {
                    int64_t row = sorted_J[(size_t)p];
                    if (row < 0 || row >= n) {
                        throw std::runtime_error(
                            "KKTData: direct-x PSD cone index " +
                            std::to_string(row) + " out of range [0, " +
                            std::to_string(n) + ")");
                    }
                    for (int64_t q = p; q < svec_dim; ++q) {
                        xcone_extra_cols[(size_t)row].push_back(
                            sorted_J[(size_t)q]);
                    }
                }
            } else if (xc.kind == XConeKind::SOC) {
                const int64_t k = static_cast<int64_t>(xc.indices.size());
                std::vector<int64_t> sorted_J = xc.indices;
                std::sort(sorted_J.begin(), sorted_J.end());
                if (k > 4) {
                    // Sparse SOC: diagonal only. Off-diagonal coupling is
                    // carried by the rank-2 expansion columns (u, v)
                    // emitted after A^T in each row — see the second
                    // inner loop below.
                    for (int64_t p = 0; p < k; ++p) {
                        int64_t row = sorted_J[(size_t)p];
                        if (row < 0 || row >= n) {
                            throw std::runtime_error(
                                "KKTData: direct-x SOC cone index " +
                                std::to_string(row) + " out of range [0, " +
                                std::to_string(n) + ")");
                        }
                        xcone_extra_cols[(size_t)row].push_back(row);
                    }
                } else {
                    // Dense SOC: full upper triangle footprint.
                    for (int64_t p = 0; p < k; ++p) {
                        int64_t row = sorted_J[(size_t)p];
                        if (row < 0 || row >= n) {
                            throw std::runtime_error(
                                "KKTData: direct-x SOC cone index " +
                                std::to_string(row) + " out of range [0, " +
                                std::to_string(n) + ")");
                        }
                        for (int64_t q = p; q < k; ++q) {
                            xcone_extra_cols[(size_t)row].push_back(
                                sorted_J[(size_t)q]);
                        }
                    }
                }
            } else if (xc.kind == XConeKind::Exp ||
                       xc.kind == XConeKind::Power) {
                // Asymmetric Exp / Power direct-x: 3-D dense Hs (the
                // primal-barrier Hessian μ·∇²F_primal(x)), full upper-tri
                // footprint over sorted indices J — same shape as a 3-D
                // dense SOC but driven by the closed-form 3-LB primal
                // barrier (see CPU expcone.rs / powcone.rs).
                const int64_t k = static_cast<int64_t>(xc.indices.size());
                if (k != 3) {
                    throw std::runtime_error(
                        "KKTData: ExponentialXCone / PowerXCone must have "
                        "exactly 3 indices");
                }
                std::vector<int64_t> sorted_J = xc.indices;
                std::sort(sorted_J.begin(), sorted_J.end());
                for (int64_t p = 0; p < k; ++p) {
                    int64_t row = sorted_J[(size_t)p];
                    if (row < 0 || row >= n) {
                        throw std::runtime_error(
                            "KKTData: direct-x Exp/Power cone index " +
                            std::to_string(row) + " out of range [0, " +
                            std::to_string(n) + ")");
                    }
                    for (int64_t q = p; q < k; ++q) {
                        xcone_extra_cols[(size_t)row].push_back(
                            sorted_J[(size_t)q]);
                    }
                }
            } else if (xc.kind == XConeKind::GenPower) {
                // GenPower direct-x: dim = dim1 + dim2.
                // Dense (dim <= 4): full upper-triangle footprint.
                // Sparse (dim > 4): diagonal-only; off-diagonal coupling
                // carried by rank-3 expansion columns (q, r, p).
                const int64_t k = static_cast<int64_t>(xc.indices.size());
                std::vector<int64_t> sorted_J = xc.indices;
                std::sort(sorted_J.begin(), sorted_J.end());
                if (k > 4) {
                    // Sparse: diagonal entries only in (1,1) block.
                    for (int64_t p = 0; p < k; ++p) {
                        int64_t row = sorted_J[(size_t)p];
                        if (row < 0 || row >= n) {
                            throw std::runtime_error(
                                "KKTData: direct-x GenPow cone index " +
                                std::to_string(row) + " out of range [0, " +
                                std::to_string(n) + ")");
                        }
                        xcone_extra_cols[(size_t)row].push_back(row);
                    }
                } else {
                    // Dense: full upper triangle.
                    for (int64_t p = 0; p < k; ++p) {
                        int64_t row = sorted_J[(size_t)p];
                        if (row < 0 || row >= n) {
                            throw std::runtime_error(
                                "KKTData: direct-x GenPow cone index " +
                                std::to_string(row) + " out of range [0, " +
                                std::to_string(n) + ")");
                        }
                        for (int64_t q = p; q < k; ++q) {
                            xcone_extra_cols[(size_t)row].push_back(
                                sorted_J[(size_t)q]);
                        }
                    }
                }
            } else {
                throw std::runtime_error(
                    "KKTData: unknown XConeKind in cones.x_cones");
            }
        }
        // Sort + dedupe each row's extras so we can merge against P's
        // row (which CSR already keeps sorted by col).
        for (auto& row_extras : xcone_extra_cols) {
            std::sort(row_extras.begin(), row_extras.end());
            row_extras.erase(std::unique(row_extras.begin(), row_extras.end()),
                             row_extras.end());
        }

        // Direct-x sparse SOC: precompute per-row
        // expansion-column footprint. After A^T in each (1,1) row that
        // belongs to a sparse x-cone, we append 2 entries (v col, then
        // u col) so CSR row order stays: [(1,1) cols, A^T cols, exp cols].
        //
        // Expansion cols for sparse x-cones live AFTER all slack
        // expansion cols (slack SOC + slack GenPow), at columns
        //   n + m + p_slack + 2 * sparse_xcone_idx  (v)
        //   n + m + p_slack + 2 * sparse_xcone_idx + 1  (u)
        // Each slack sparse GenPow cone consumes 9 expansion cols (q/r/p
        // + 6 PD axes) — see the `p` initializer at line 195 and the
        // `exp_col += 9` at line 936. Using `3 * numSparseGenPow` here
        // (the rank-3 q/r/p alone) would place direct-x sparse SOC/GenPow
        // expansion cols 6·numSparseGenPow positions too early, overlapping
        // the last 6 PD-axis cols of slack GenPow.
        const int64_t p_slack =
            2 * cones.numSparseSoc + 9 * cones.numSparseGenPow;
        std::vector<int64_t> xcone_sparse_idx_of(cones.x_cones.size(),
                                                 int64_t{-1});
        int64_t num_sparse_xsoc = 0;
        std::vector<int64_t> sparse_xsoc_offset;   // prefix over sparse x-cones of dim
        std::vector<int64_t> sparse_xsoc_dim;      // dim per sparse x-cone
        for (size_t c = 0; c < cones.x_cones.size(); ++c) {
            const auto& xc = cones.x_cones[c];
            if (xc.kind == XConeKind::SOC && xc.indices.size() > 4) {
                xcone_sparse_idx_of[c] = num_sparse_xsoc++;
                sparse_xsoc_offset.push_back(
                    sparse_xsoc_offset.empty()
                        ? int64_t{0}
                        : sparse_xsoc_offset.back() + sparse_xsoc_dim.back());
                sparse_xsoc_dim.push_back(
                    static_cast<int64_t>(xc.indices.size()));
            }
        }

        // Per-row metadata: for each row i, if some sparse x-cone owns
        // it, remember (v_col, flat_idx) where flat_idx is this row's
        // slot in the H_xcone_u/v_indices flat arrays. -1 = no sparse
        // x-cone on this row. Indices across x-cones must be disjoint
        // (API contract), so at most one entry per row.
        std::vector<int64_t> xsoc_row_v_col((size_t)n, int64_t{-1});
        std::vector<int64_t> xsoc_row_flat_idx((size_t)n, int64_t{-1});
        for (size_t c = 0; c < cones.x_cones.size(); ++c) {
            if (xcone_sparse_idx_of[c] < 0) continue;
            const auto& xc = cones.x_cones[c];
            int64_t sparse_idx = xcone_sparse_idx_of[c];
            int64_t base_flat = sparse_xsoc_offset[(size_t)sparse_idx];
            int64_t v_col = n + m + p_slack + 2 * sparse_idx;

            // Sort the cone's indices to get sorted_pos.
            std::vector<int64_t> sorted_J = xc.indices;
            std::sort(sorted_J.begin(), sorted_J.end());
            for (int64_t p = 0; p < (int64_t)sorted_J.size(); ++p) {
                int64_t row = sorted_J[(size_t)p];
                if (xsoc_row_v_col[(size_t)row] != -1) {
                    throw std::runtime_error(
                        "KKTData: direct-x cone indices overlap at row " +
                        std::to_string(row) +
                        "; indices across x_cones must be disjoint.");
                }
                xsoc_row_v_col[(size_t)row] = v_col;
                xsoc_row_flat_idx[(size_t)row] = base_flat + p;
            }
        }
        // Flat index arrays for sparse x-cone u/v columns (populated in
        // the (1,1) loop below and uploaded after Phase 2 rows finalize).
        const int64_t total_sparse_xsoc_dim =
            sparse_xsoc_offset.empty()
                ? int64_t{0}
                : sparse_xsoc_offset.back() + sparse_xsoc_dim.back();
        std::vector<int64_t> H_xcone_u_indices((size_t)total_sparse_xsoc_dim, -1);
        std::vector<int64_t> H_xcone_v_indices((size_t)total_sparse_xsoc_dim, -1);

        // Direct-x sparse GenPow (Phase 4 + rank-9 PD): precompute per-row metadata.
        // 9 expansion cols per sparse cone live after x-SOC expansion cols, at:
        //   q_col + 0  (q col)
        //   q_col + 1  (r col)
        //   q_col + 2  (p col)
        //   q_col + 3..8  (6 PD axes)
        // where q_col = p_slack + 2*num_sparse_xsoc + 9*sparse_xgenpow_idx.
        const int64_t p_slack_xsoc = p_slack + 2 * num_sparse_xsoc;

        // GenPow sparse x-cone metadata
        std::vector<int64_t> xcone_genpow_sparse_idx_of(cones.x_cones.size(), int64_t{-1});
        int64_t num_sparse_xgenpow = 0;
        // Per-sparse-xcone: dim1, dim2, dim offset into q_indices and r_indices
        std::vector<int64_t> sparse_xgenpow_dim1;  // dim1 per sparse x-GenPow cone
        std::vector<int64_t> sparse_xgenpow_dim2;  // dim2 per sparse x-GenPow cone
        std::vector<int64_t> sparse_xgenpow_q_offset;  // prefix over sparse x-GenPow cones (dim1)
        std::vector<int64_t> sparse_xgenpow_r_offset;  // prefix over sparse x-GenPow cones (dim2)
        std::vector<int64_t> sparse_xgenpow_p_offset;  // prefix over sparse x-GenPow cones (dim)
        for (size_t c = 0; c < cones.x_cones.size(); ++c) {
            const auto& xc = cones.x_cones[c];
            if (xc.kind == XConeKind::GenPower && xc.indices.size() > 4) {
                xcone_genpow_sparse_idx_of[c] = num_sparse_xgenpow++;
                const int64_t d1 = static_cast<int64_t>(xc.gen_power_alphas.size());
                const int64_t d  = static_cast<int64_t>(xc.indices.size());
                const int64_t d2 = d - d1;
                int64_t q_off = sparse_xgenpow_q_offset.empty()
                    ? int64_t{0}
                    : sparse_xgenpow_q_offset.back() + sparse_xgenpow_dim1.back();
                int64_t r_off = sparse_xgenpow_r_offset.empty()
                    ? int64_t{0}
                    : sparse_xgenpow_r_offset.back() + sparse_xgenpow_dim2.back();
                int64_t p_off = sparse_xgenpow_p_offset.empty()
                    ? int64_t{0}
                    : sparse_xgenpow_p_offset.back() +
                      (sparse_xgenpow_dim1.back() + sparse_xgenpow_dim2.back());
                sparse_xgenpow_q_offset.push_back(q_off);
                sparse_xgenpow_r_offset.push_back(r_off);
                sparse_xgenpow_p_offset.push_back(p_off);
                sparse_xgenpow_dim1.push_back(d1);
                sparse_xgenpow_dim2.push_back(d2);
            }
        }
        // Total entry counts for xcone GenPow expansion arrays
        const int64_t total_sparse_xgenpow_dim1 = sparse_xgenpow_q_offset.empty()
            ? int64_t{0}
            : sparse_xgenpow_q_offset.back() + sparse_xgenpow_dim1.back();
        const int64_t total_sparse_xgenpow_dim2 = sparse_xgenpow_r_offset.empty()
            ? int64_t{0}
            : sparse_xgenpow_r_offset.back() + sparse_xgenpow_dim2.back();
        const int64_t total_sparse_xgenpow_dim = sparse_xgenpow_p_offset.empty()
            ? int64_t{0}
            : sparse_xgenpow_p_offset.back() +
              (sparse_xgenpow_dim1.back() + sparse_xgenpow_dim2.back());

        // Per-row metadata: q_col (expansion col), row_type (0=dim1,1=dim2), flat_idx
        // For each row in the (1,1) block that belongs to a sparse x-GenPow cone:
        //   q_col_for_row[i] = expansion column for q (or -1)
        //   genpow_row_type[i] = 0 if i < dim1 part (q-col gets entry), 1 if i >= dim1 (r-col gets entry)
        //   genpow_p_flat[i] = flat index in H_xcone_genpow_p_indices
        //   genpow_qr_flat[i] = flat index in H_xcone_genpow_q/r_indices (-1 if not q or r)
        struct XGenPowRowInfo { int64_t q_col; int64_t row_type; int64_t p_flat; int64_t qr_flat; };
        std::vector<XGenPowRowInfo> xgenpow_row_info((size_t)n, {-1, -1, -1, -1});
        for (size_t c = 0; c < cones.x_cones.size(); ++c) {
            if (xcone_genpow_sparse_idx_of[c] < 0) continue;
            const auto& xc = cones.x_cones[c];
            int64_t sparse_idx = xcone_genpow_sparse_idx_of[c];
            int64_t d1 = sparse_xgenpow_dim1[(size_t)sparse_idx];
            int64_t q_col = n + m + p_slack_xsoc + 9 * sparse_idx;  // q expansion col

            // Sort (row, cone_pos) pairs so we can walk rows in CSR-ascending
            // order while remembering each entry's original cone-internal
            // position. The cone-internal position determines whether the
            // row belongs to the p-part (cone_pos < dim1) or the w-part
            // (cone_pos >= dim1), and is the same index the scaling kernel
            // (kernels.cu update_xcones_genpow_scaling_kernel) uses when
            // writing q/r/p values. Sorting by row alone — as the previous
            // code did — silently mis-classified interleaved orderings.
            std::vector<std::pair<int64_t, int64_t>> sorted_pairs;
            sorted_pairs.reserve(xc.indices.size());
            for (int64_t i = 0; i < (int64_t)xc.indices.size(); ++i) {
                sorted_pairs.emplace_back(xc.indices[(size_t)i], i);
            }
            std::sort(sorted_pairs.begin(), sorted_pairs.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });
            for (const auto& kv : sorted_pairs) {
                int64_t row = kv.first;
                int64_t cone_pos = kv.second;
                if (xgenpow_row_info[(size_t)row].q_col != -1 ||
                    xsoc_row_v_col[(size_t)row] != -1) {
                    throw std::runtime_error(
                        "KKTData: direct-x cone indices overlap at row " +
                        std::to_string(row) + "; indices across x_cones must be disjoint.");
                }
                int64_t row_type = (cone_pos < d1) ? 0 : 1;
                int64_t p_flat = sparse_xgenpow_p_offset[(size_t)sparse_idx] + cone_pos;
                int64_t qr_flat = (row_type == 0)
                    ? sparse_xgenpow_q_offset[(size_t)sparse_idx] + cone_pos
                    : sparse_xgenpow_r_offset[(size_t)sparse_idx] + (cone_pos - d1);
                xgenpow_row_info[(size_t)row] = {q_col, row_type, p_flat, qr_flat};
            }
        }
        // Flat index arrays for xcone GenPow expansion columns (rank-3 q/r/p +
        // rank-6 PD axes; PD-axis arrays are dim-aligned, one slot per cone row).
        std::vector<int64_t> H_xcone_genpow_q_indices((size_t)total_sparse_xgenpow_dim1, -1);
        std::vector<int64_t> H_xcone_genpow_r_indices((size_t)total_sparse_xgenpow_dim2, -1);
        std::vector<int64_t> H_xcone_genpow_p_indices((size_t)total_sparse_xgenpow_dim, -1);
        std::vector<int64_t> H_xcone_genpow_pd_axis_indices[6];
        for (int kk = 0; kk < 6; ++kk) {
            H_xcone_genpow_pd_axis_indices[kk].assign(
                (size_t)total_sparse_xgenpow_dim, -1);
        }

        int64_t nnz = 0;
        rowOff[0] = 0;

        // Top-left block: Upper(Pext) with ensured diagonal; Top-right block: A^T
        // Pext = P ∪ direct-x footprint (upper tri), with structural zeros
        // at direct-x-only positions. When x_cones is empty, the merge
        // collapses to the original P-only loop and the path is unchanged.
        for (int64_t i = 0; i < n; ++i) {
            bool hasDiag = false;

            // Skip P lower-tri entries (mark Pnnz_idx = -1).
            int64_t pp = P_ro[i];
            const int64_t pp_end = P_ro[i + 1];
            while (pp < pp_end && P_ci[pp] < i) {
                Pnnz_idx[(size_t)pp] = -1;
                pp++;
            }

            // Merge P's row i (upper tri, sorted) with xcone_extra_cols[i]
            // (sorted + deduped). Cols appearing in both sources get a
            // single KKT slot shared by the P entry (Pnnz_idx) and the
            // x-cone Hs second-pass lookup.
            const auto& extras = xcone_extra_cols[(size_t)i];
            size_t e_cur = 0;
            while (pp < pp_end || e_cur < extras.size()) {
                const int64_t p_col = (pp < pp_end)
                    ? P_ci[pp]
                    : std::numeric_limits<int64_t>::max();
                const int64_t e_col = (e_cur < extras.size())
                    ? extras[e_cur]
                    : std::numeric_limits<int64_t>::max();
                int64_t emit_col;
                bool from_p = false;

                if (p_col == e_col) {
                    emit_col = p_col;
                    from_p = true;
                    ++pp;
                    ++e_cur;
                } else if (p_col < e_col) {
                    emit_col = p_col;
                    from_p = true;
                    ++pp;
                } else {
                    emit_col = e_col;
                    ++e_cur;
                }

                if (emit_col == i) { hasDiag = true; Pdiag[(size_t)i] = nnz; }
                colIdx.push_back(emit_col);
                if (from_p) Pnnz_idx[(size_t)(pp - 1)] = nnz;
                ++nnz;
            }

            if (!hasDiag) {                     // add εI slot if missing
                Pdiag[(size_t)i] = nnz;
                colIdx.push_back(i);
                ++nnz;
            }

            // A^T: for each A(r,i)≠0, add (i, n+r) (always upper since n+r ≥ i)
            for (const auto& [r, pp_a] : A_col_entries[(size_t)i]) {
                colIdx.push_back(n + r);
                Annz_idx[pp_a] = nnz;  // Map CSR index pp_a of A to KKT position nnz
                ++nnz;
            }

            // Direct-x sparse SOC expansion col entries. For
            // rows belonging to a sparse x-cone, append v col then u col
            // after all A^T entries. v_col < u_col ensures CSR col order
            // within the row remains ascending.
            if (xsoc_row_v_col[(size_t)i] != -1) {
                int64_t v_col = xsoc_row_v_col[(size_t)i];
                int64_t flat = xsoc_row_flat_idx[(size_t)i];
                // Direct-x SOC convention is flipped vs slack: dsigns at
                // lines 1558-1559 are +1 (v) / -1 (u), and the scaling
                // kernel writes diag `[+η², -η²]` (kernels.cu:2613-2615
                // and CPU directldlkktsolver.rs:289).
                // v column entry (diag = +η²)
                H_xcone_v_indices[(size_t)flat] = nnz;
                colIdx.push_back(v_col);
                ++nnz;
                // u column entry (diag = -η²)
                H_xcone_u_indices[(size_t)flat] = nnz;
                colIdx.push_back(v_col + 1);
                ++nnz;
            }

            // Direct-x sparse GenPow expansion col entries (rank-9 PD). For
            // rows belonging to a sparse x-GenPow cone, append:
            //   if dim1 part: q col (at q_col), p col (at q_col+2)
            //   if dim2 part: r col (at q_col+1), p col (at q_col+2)
            //   then 6 PD-axis cols (at q_col+3 .. q_col+8) for all rows.
            // Cols are always ascending: q_col < q_col+1 < ... < q_col+8.
            if (xgenpow_row_info[(size_t)i].q_col != -1) {
                const auto& gpi = xgenpow_row_info[(size_t)i];
                int64_t q_col = gpi.q_col;
                if (gpi.row_type == 0) {
                    // dim1 part: q and p entries
                    H_xcone_genpow_q_indices[(size_t)gpi.qr_flat] = nnz;
                    colIdx.push_back(q_col);   // q col
                    ++nnz;
                } else {
                    // dim2 part: r and p entries
                    H_xcone_genpow_r_indices[(size_t)gpi.qr_flat] = nnz;
                    colIdx.push_back(q_col + 1);  // r col
                    ++nnz;
                }
                // p col (all rows)
                H_xcone_genpow_p_indices[(size_t)gpi.p_flat] = nnz;
                colIdx.push_back(q_col + 2);  // p col
                ++nnz;
                // 6 PD-axis cols (all rows)
                for (int kk = 0; kk < 6; ++kk) {
                    H_xcone_genpow_pd_axis_indices[kk][(size_t)gpi.p_flat] = nnz;
                    colIdx.push_back(q_col + 3 + kk);
                    ++nnz;
                }
            }

            rowOff[(size_t)i + 1] = nnz;
        }

        // Bottom-right block: -(H+εI)
        // Zero & nonneg cones → 1×1 diagonal; SOC/EXP/POW → 3×3 upper pattern.
        int64_t row = n;
        int64_t hidx = 0;

        const int64_t num_diag_cones = cones.numZeroCones + cones.numNonnegCones;
        for (int64_t k = 0; k < num_diag_cones; ++k) {
            // row n+k
            colIdx.push_back(row);              // diagonal
            Hdiag[(size_t)hidx++] = nnz;
            ++nnz;
            rowOff[(size_t)row + 1] = nnz;
            ++row;
        }

        // Resolve SOC dims: use original (unsorted) dims for KKT construction
        // because KKT rows must match the A matrix constraint order.
        // socConeDimsOriginal is set during initialize() before sorting.
        const auto& socDimsForKKT = cones.socConeDimsOriginal.empty() ? cones.socConeDims : cones.socConeDimsOriginal;
        const bool socDimsAvailable = !socDimsForKKT.empty();
        int64_t socHsReserve = cones.totalSocHsEntries;
        if (!socDimsAvailable && cones.numSocCones > 0) {
            socHsReserve = cones.numSocCones * 6;  // dim=3 -> 6 entries each
        }

        // Store indices for cone blocks separately for each cone type
        // SOC: dense=upper-tri, sparse=diagonal-only + expansion columns
        std::vector<int64_t> H_soc_indices;
        std::vector<int64_t> H_soc_u_indices;   // u column entries for sparse SOC
        std::vector<int64_t> H_soc_v_indices;   // v column entries for sparse SOC
        std::vector<int64_t> H_soc_exp_diag_indices; // expansion diagonal entries
        std::vector<int64_t> H_exp_indices;
        std::vector<int64_t> H_power_indices;
        std::vector<int64_t> H_genpow_indices;   // diagonal entries for GenPowerCone
        std::vector<int64_t> H_genpow_q_indices;  // q column entries (dim1 per cone)
        std::vector<int64_t> H_genpow_r_indices;  // r column entries (dim2 per cone)
        std::vector<int64_t> H_genpow_p_indices;  // p column entries (dim per cone)
        std::vector<int64_t> H_genpow_exp_diag_indices; // 9 expansion diag entries per sparse cone (q/r/p + 6 PD axes)
        // 6 PD-axis off-diagonal column index arrays. Each axis has `dim`
        // entries per sparse cone (one per cone-block row), aligned with
        // genpow_p_indices' offset structure.
        std::vector<int64_t> H_genpow_pd_axis_indices[6];
        H_soc_indices.reserve((size_t)socHsReserve);
        H_exp_indices.reserve((size_t)(cones.numExpCones * 6));
        H_power_indices.reserve((size_t)(cones.numPowerCones * 6));

        // Helper to add a dim x dim upper-triangular block and store its indices
        auto add_dense_block = [&](int64_t dim, std::vector<int64_t>& indices_out) {
            const int64_t r0 = row;
            for (int64_t i = 0; i < dim; ++i) {
                for (int64_t j = i; j < dim; ++j) {
                    indices_out.push_back(nnz);
                    colIdx.push_back(r0 + j);
                    if (i == j) Hdiag[(size_t)hidx++] = nnz;
                    ++nnz;
                }
                rowOff[(size_t)r0 + i + 1] = nnz;
            }
            row += dim;
        };

        // Helper to add a dense upper-triangle block in svec (column-major) order.
        // Used for PSD cones where Hs is stored in svec order: (0,0),(0,1),(1,1),(0,2),(1,2),(2,2),...
        auto add_svec_dense_block = [&](int64_t dim, std::vector<int64_t>& indices_out) {
            const int64_t r0 = row;
            // svec order: for j=0..dim-1: for i=0..j
            // But CSR needs entries sorted by column within each row.
            // We need to build the CSR structure correctly: for each row i,
            // add columns j=i..dim-1 (same as row-major). BUT the indices_out
            // must be ordered to match svec storage.
            //
            // Solution: build a mapping from svec index → KKT nnz position.
            // First, create entries in CSR row order (for correct KKT structure),
            // then reorder indices_out to match svec order.

            std::vector<int64_t> row_major_indices;
            int64_t block_start = nnz;
            for (int64_t i = 0; i < dim; ++i) {
                for (int64_t j = i; j < dim; ++j) {
                    row_major_indices.push_back(nnz);
                    colIdx.push_back(r0 + j);
                    if (i == j) Hdiag[(size_t)hidx++] = nnz;
                    ++nnz;
                }
                rowOff[(size_t)r0 + i + 1] = nnz;
            }

            // Now reorder: map svec index → row-major index
            // svec: (0,0),(0,1),(1,1),(0,2),(1,2),(2,2),...
            // row-major: (0,0),(0,1),(0,2),...,(1,1),(1,2),...,(2,2),...
            int64_t svec_idx = 0;
            for (int64_t j = 0; j < dim; ++j) {
                for (int64_t i = 0; i <= j; ++i) {
                    // Find the row-major position for (i,j)
                    // Row i starts at row_major position: i*dim - i*(i-1)/2 - i = i*dim - i*(i+1)/2
                    int64_t rm_pos = 0;
                    for (int64_t ii = 0; ii < i; ii++) rm_pos += (dim - ii);
                    rm_pos += (j - i);
                    indices_out.push_back(row_major_indices[rm_pos]);
                    svec_idx++;
                }
            }

            row += dim;
        };

        // Helper to add a diagonal-only block (for sparse SOC dim > 4)
        auto add_diag_block = [&](int64_t dim, std::vector<int64_t>& indices_out) {
            const int64_t r0 = row;
            for (int64_t i = 0; i < dim; ++i) {
                indices_out.push_back(nnz);
                colIdx.push_back(r0 + i);
                Hdiag[(size_t)hidx++] = nnz;
                ++nnz;
                // Leave rowOff incomplete here - expansion columns fill remaining entries
            }
            // Don't advance row yet - expansion columns add to these rows
        };


        // Track the next expansion column
        int64_t exp_col = n + m;  // expansion rows start after (n+m) rows

        // Phase 1: Add all cone block rows (n..n+m-1) in order
        // Expansion column entries are placed at columns exp_col, exp_col+1
        // but expansion ROWS are deferred to Phase 2

        // Track which expansion columns belong to which sparse cone
        struct PendingExpansionRow {
            int64_t exp_col;  // column index for this expansion
            int64_t num_cols; // 2 for SOC, 3 for GenPowerCone
        };
        std::vector<PendingExpansionRow> pendingExpRows;

        // Add cone blocks in Clarabel/CVXPY order:
        // Zero, Nonnegative, SOC, PSD, Exponential, Power, Generalized Power.
        for (int64_t i = 0; i < cones.numSocCones; ++i) {
            int64_t dim = socDimsAvailable ? socDimsForKKT[i] : 3;
            if (dim > 4) {
                // Sparse SOC: diagonal block + expansion column entries per row
                const int64_t r0 = row;

                // Add diagonal + v column + u column entries per cone row
                // Ordering matches Rust: v first (col = exp_col), u second (col = exp_col+1)
                // This ensures correct Schur complement signs:
                //   v col has expansion diag -η² → contribution = +η²vv^T
                //   u col has expansion diag +η² → contribution = -η²uu^T
                for (int64_t ii = 0; ii < dim; ++ii) {
                    // Diagonal entry
                    H_soc_indices.push_back(nnz);
                    colIdx.push_back(r0 + ii);
                    Hdiag[(size_t)hidx++] = nnz;
                    ++nnz;

                    // v column entry (column = exp_col)
                    H_soc_v_indices.push_back(nnz);
                    colIdx.push_back(exp_col);
                    ++nnz;

                    // u column entry (column = exp_col + 1)
                    H_soc_u_indices.push_back(nnz);
                    colIdx.push_back(exp_col + 1);
                    ++nnz;

                    rowOff[(size_t)r0 + ii + 1] = nnz;
                }

                // Defer expansion rows to Phase 2
                pendingExpRows.push_back({exp_col, 2});

                row += dim;
                exp_col += 2;
            } else {
                // Dense SOC: full upper triangle block
                add_dense_block(dim, H_soc_indices);
            }
        }

        // Add PSD cone blocks (svec_dim × svec_dim dense upper triangle)
        // Use original (unsorted) dims for KKT construction
        const auto& psdDimsForKKT = cones.psdConeDimsOriginal.empty() ? cones.psdConeDims : cones.psdConeDimsOriginal;
        std::vector<int64_t> H_psd_indices;
        if (!psdDimsForKKT.empty()) {
            int64_t psd_hs_reserve = 0;
            for (auto d : psdDimsForKKT) {
                int64_t svec_dim = d * (d + 1) / 2;
                psd_hs_reserve += svec_dim * (svec_dim + 1) / 2;
            }
            H_psd_indices.reserve((size_t)psd_hs_reserve);
        }
        for (int64_t i = 0; i < cones.numPsdCones; ++i) {
            int64_t mat_dim = psdDimsForKKT.empty() ? 1 : psdDimsForKKT[i];
            int64_t svec_dim = mat_dim * (mat_dim + 1) / 2;
            add_svec_dense_block(svec_dim, H_psd_indices);
        }

        // Add exp cone blocks (always 3x3)
        for (int64_t i = 0; i < cones.numExpCones; ++i) {
            add_dense_block(3, H_exp_indices);
        }

        // Add power cone blocks (always 3x3)
        for (int64_t i = 0; i < cones.numPowerCones; ++i) {
            add_dense_block(3, H_power_indices);
        }

        // Add GenPowerCone blocks: dense (dim<=4) or sparse expansion (dim>4)
        // Dense: full upper-triangle block (like SOC dim<=4)
        // Sparse: diagonal + q/r/p expansion columns, Hs = μ*(D + pp' - qq' - rr')
        // Use original (unsorted) dims for KKT construction since KKT rows
        // must match A matrix constraint order. Reorder index arrays below.
        const auto& gpDim1sForKKT = cones.genPowerDim1sOriginal.empty() ? cones.genPowerDim1s : cones.genPowerDim1sOriginal;
        const auto& gpDim2sForKKT = cones.genPowerDim2sOriginal.empty() ? cones.genPowerDim2s : cones.genPowerDim2sOriginal;
        for (int64_t i = 0; i < cones.numGenPowerCones; ++i) {
            int64_t dim1 = gpDim1sForKKT[i];
            int64_t dim2 = gpDim2sForKKT[i];
            int64_t dim = dim1 + dim2;

            if (dim > 4) {
                // Sparse GenPowerCone: diagonal + expansion column entries per row
                const int64_t r0 = row;

                for (int64_t ii = 0; ii < dim; ++ii) {
                    // Diagonal entry
                    H_genpow_indices.push_back(nnz);
                    colIdx.push_back(r0 + ii);
                    Hdiag[(size_t)hidx++] = nnz;
                    ++nnz;

                    // q column entry (only for ii < dim1)
                    if (ii < dim1) {
                        H_genpow_q_indices.push_back(nnz);
                        colIdx.push_back(exp_col);
                        ++nnz;
                    }

                    // r column entry (only for ii >= dim1)
                    if (ii >= dim1) {
                        H_genpow_r_indices.push_back(nnz);
                        colIdx.push_back(exp_col + 1);
                        ++nnz;
                    }

                    // p column entry (all rows)
                    H_genpow_p_indices.push_back(nnz);
                    colIdx.push_back(exp_col + 2);
                    ++nnz;

                    // 6 PD-axis off-diagonal column entries (all rows).
                    // When PD scaling is inactive these slots get a tiny
                    // structural ε plus a +1 sentinel so the rank-1 block
                    // is well-conditioned; when active they hold the
                    // actual axis scaled by ±sqrt(|coef|).
                    for (int k = 0; k < 6; ++k) {
                        H_genpow_pd_axis_indices[k].push_back(nnz);
                        colIdx.push_back(exp_col + 3 + k);
                        ++nnz;
                    }

                    rowOff[(size_t)r0 + ii + 1] = nnz;
                }

                // Defer expansion rows to Phase 2 — 9 rows per sparse cone:
                // q, r, p (rank-3) + 6 PD axes.
                pendingExpRows.push_back({exp_col, 9});

                row += dim;
                exp_col += 9;
            } else {
                // Dense GenPowerCone: full upper triangle block
                add_dense_block(dim, H_genpow_indices);
            }
        }

        // Sanity check cone rows
        if (row != n + m) throw std::runtime_error("KKTData: H block rows mismatch (constraint rows)");

        // Phase 2: Add expansion rows (n+m..n+m+p-1)
        // These must come after ALL cone rows to maintain CSR ordering
        for (const auto& pending : pendingExpRows) {
            int64_t ec = pending.exp_col;

            if (pending.num_cols == 2) {
                // SOC sparse expansion: 2 rows (v diagonal, u diagonal)
                // Row ec: diagonal entry only (v column diagonal = -η²)
                H_soc_exp_diag_indices.push_back(nnz);
                colIdx.push_back(ec);
                Hdiag[(size_t)hidx++] = nnz;
                ++nnz;
                rowOff[(size_t)ec + 1] = nnz;

                // Row ec+1: diagonal entry only (u column diagonal = +η²)
                H_soc_exp_diag_indices.push_back(nnz);
                colIdx.push_back(ec + 1);
                Hdiag[(size_t)hidx++] = nnz;
                ++nnz;
                rowOff[(size_t)ec + 2] = nnz;
            } else {
                // GenPowerCone sparse expansion: 9 rows (q, r, p, then 6 PD).
                // First 3 are dual rank-3 (dsigns -1, -1, +1). Last 6 are PD
                // axes — dsigns are dynamic per iteration (depend on
                // `pd_signs`); the static dsigns here is +1 by default and
                // gets overwritten by `update_dsigns_from_pd_signs_kernel`
                // when PD scaling is active for that cone.
                for (int k = 0; k < 9; ++k) {
                    H_genpow_exp_diag_indices.push_back(nnz);
                    colIdx.push_back(ec + k);
                    Hdiag[(size_t)hidx++] = nnz;
                    ++nnz;
                    rowOff[(size_t)ec + k + 1] = nnz;
                }
            }
        }

        // Phase 2b: Direct-x sparse SOC expansion diag rows.
        // Appended after slack pendingExpRows so CSR row order is:
        // (1,1)[0..n] < slack[n..n+m] < slack_exp[n+m..n+m+p_slack] <
        // xsoc_exp[n+m+p_slack..n+m+p_slack+2*num_sparse_xsoc].
        std::vector<int64_t> H_xcone_exp_diag_indices;
        H_xcone_exp_diag_indices.reserve((size_t)(2 * num_sparse_xsoc + 9 * num_sparse_xgenpow));
        for (int64_t sparse_idx = 0; sparse_idx < num_sparse_xsoc; ++sparse_idx) {
            int64_t v_col = n + m + p_slack + 2 * sparse_idx;
            // Row v_col: v diagonal (= +η², direct-x sign convention)
            H_xcone_exp_diag_indices.push_back(nnz);
            colIdx.push_back(v_col);
            Hdiag[(size_t)hidx++] = nnz;
            ++nnz;
            rowOff[(size_t)v_col + 1] = nnz;
            // Row v_col+1: u diagonal (= -η², direct-x sign convention)
            H_xcone_exp_diag_indices.push_back(nnz);
            colIdx.push_back(v_col + 1);
            Hdiag[(size_t)hidx++] = nnz;
            ++nnz;
            rowOff[(size_t)v_col + 2] = nnz;
            exp_col += 2;
        }

        // Phase 2c: Direct-x sparse GenPow expansion diag rows (rank-9 PD).
        // First 3 rows: q,r,p with dsigns [+1, +1, -1] (opposite of slack's
        // [-1,-1,+1]; direct-x contributes +Hs to (1,1), so rank-3 signs flip).
        //   Row q_col:   q diag (+1) → contribution -μ·qq'
        //   Row q_col+1: r diag (+1) → contribution -μ·rr'
        //   Row q_col+2: p diag (-1) → contribution +μ·pp'
        // Next 6 rows: PD-axis diags with dsigns dynamic (rewritten per
        // iteration by the scaling kernel). Default value here is +1, matching
        // the inactive sentinel that the kernel writes when pd_active=0.
        for (int64_t sparse_idx = 0; sparse_idx < num_sparse_xgenpow; ++sparse_idx) {
            int64_t q_col = n + m + p_slack_xsoc + 9 * sparse_idx;
            for (int kk = 0; kk < 9; ++kk) {
                H_xcone_exp_diag_indices.push_back(nnz);
                colIdx.push_back(q_col + kk);
                Hdiag[(size_t)hidx++] = nnz;
                ++nnz;
                rowOff[(size_t)q_col + kk + 1] = nnz;
            }
            exp_col += 9;
        }

        if (exp_col != N) throw std::runtime_error("KKTData: expansion column count mismatch");
        const int64_t nnzKKT = nnz;

        // Allocate device CSR and upload indices
        CSR kkt(N, N, nnzKKT, batch_);
        kkt.indicesCpuToGpu(rowOff.data(), colIdx.data(), stream);
        cudaStreamSynchronize(stream);

        // Upload diagonal index maps to device
        int64_t* dP = nullptr; int64_t* dH = nullptr;
        if (n > 0) {
            auto e = cudaMalloc(&dP, sizeof(int64_t) * (size_t)n);
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dP, Pdiag.data(), sizeof(int64_t) * (size_t)n, cudaMemcpyHostToDevice, stream);
        }
        if (m + p > 0) {
            auto e = cudaMalloc(&dH, sizeof(int64_t) * (size_t)(m + p));
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH, Hdiag.data(), sizeof(int64_t) * (size_t)(m + p), cudaMemcpyHostToDevice, stream);
        }
        cudaStreamSynchronize(stream);

        int64_t* dPnnz = nullptr; int64_t* dAnnz = nullptr;
        if (n > 0) {
            auto e = cudaMalloc(&dPnnz, sizeof(int64_t) * (size_t)nnzP);
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dPnnz, Pnnz_idx.data(), sizeof(int64_t) * (size_t)nnzP, cudaMemcpyHostToDevice, stream);
        }
        if (m > 0) {
            auto e = cudaMalloc(&dAnnz, sizeof(int64_t) * (size_t)nnzA);
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dAnnz, Annz_idx.data(), sizeof(int64_t) * (size_t)nnzA, cudaMemcpyHostToDevice, stream);
        }
        cudaStreamSynchronize(stream);

        // Reorder H_soc_indices from original cone order to sorted cone order
        // so that soc_Hs[sorted_idx] maps to the correct KKT position.
        if (!cones.socSortPerm.empty() && !H_soc_indices.empty()) {
            const auto& perm = cones.socSortPerm;

            // Compute Hs entry counts per cone in original order
            auto hsEntriesForDim = [](int64_t dim) -> int64_t {
                return (dim > 4) ? dim : (dim * (dim + 1) / 2);
            };

            // Build prefix sums for original and sorted orders
            std::vector<int64_t> orig_hs_off(cones.numSocCones + 1, 0);
            std::vector<int64_t> sorted_hs_off(cones.numSocCones + 1, 0);
            for (int64_t i = 0; i < cones.numSocCones; ++i) {
                orig_hs_off[i + 1] = orig_hs_off[i] + hsEntriesForDim(cones.socConeDimsOriginal[i]);
                sorted_hs_off[i + 1] = sorted_hs_off[i] + hsEntriesForDim(cones.socConeDims[i]);
            }

            // Permute: sorted cone j was originally perm[j]
            std::vector<int64_t> reordered(H_soc_indices.size());
            for (int64_t j = 0; j < cones.numSocCones; ++j) {
                int64_t orig_cone = perm[j];
                int64_t orig_start = orig_hs_off[orig_cone];
                int64_t sorted_start = sorted_hs_off[j];
                int64_t count = hsEntriesForDim(cones.socConeDims[j]);
                for (int64_t k = 0; k < count; ++k) {
                    reordered[sorted_start + k] = H_soc_indices[orig_start + k];
                }
            }
            H_soc_indices = reordered;

            // Similarly reorder sparse SOC expansion indices (u, v, exp_diag)
            // These are per-sparse-cone, indexed by sparse cone order
            if (!H_soc_u_indices.empty()) {
                // Build prefix sums of dim for sparse cones only, in original and sorted order
                std::vector<int64_t> orig_sparse_off(cones.numSocCones + 1, 0);
                std::vector<int64_t> sorted_sparse_off(cones.numSocCones + 1, 0);
                for (int64_t i = 0; i < cones.numSocCones; ++i) {
                    int64_t orig_dim = cones.socConeDimsOriginal[i];
                    orig_sparse_off[i + 1] = orig_sparse_off[i] + ((orig_dim > 4) ? orig_dim : 0);
                    int64_t sorted_dim = cones.socConeDims[i];
                    sorted_sparse_off[i + 1] = sorted_sparse_off[i] + ((sorted_dim > 4) ? sorted_dim : 0);
                }

                // Permute u indices
                std::vector<int64_t> reordered_u(H_soc_u_indices.size());
                std::vector<int64_t> reordered_v(H_soc_v_indices.size());
                for (int64_t j = 0; j < cones.numSocCones; ++j) {
                    if (cones.socConeDims[j] <= 4) continue;
                    int64_t orig_cone = perm[j];
                    int64_t orig_start = orig_sparse_off[orig_cone];
                    int64_t sorted_start = sorted_sparse_off[j];
                    int64_t dim = cones.socConeDims[j];
                    for (int64_t k = 0; k < dim; ++k) {
                        reordered_u[sorted_start + k] = H_soc_u_indices[orig_start + k];
                        reordered_v[sorted_start + k] = H_soc_v_indices[orig_start + k];
                    }
                }
                H_soc_u_indices = reordered_u;
                H_soc_v_indices = reordered_v;

                // Permute exp_diag indices (2 per sparse cone)
                if (!H_soc_exp_diag_indices.empty()) {
                    // Build sparse cone count prefix sums
                    std::vector<int64_t> orig_sparse_cnt(cones.numSocCones + 1, 0);
                    std::vector<int64_t> sorted_sparse_cnt(cones.numSocCones + 1, 0);
                    for (int64_t i = 0; i < cones.numSocCones; ++i) {
                        orig_sparse_cnt[i + 1] = orig_sparse_cnt[i] + ((cones.socConeDimsOriginal[i] > 4) ? 1 : 0);
                        sorted_sparse_cnt[i + 1] = sorted_sparse_cnt[i] + ((cones.socConeDims[i] > 4) ? 1 : 0);
                    }
                    std::vector<int64_t> reordered_diag(H_soc_exp_diag_indices.size());
                    for (int64_t j = 0; j < cones.numSocCones; ++j) {
                        if (cones.socConeDims[j] <= 4) continue;
                        int64_t orig_cone = perm[j];
                        int64_t orig_idx = orig_sparse_cnt[orig_cone] * 2;
                        int64_t sorted_idx = sorted_sparse_cnt[j] * 2;
                        reordered_diag[sorted_idx] = H_soc_exp_diag_indices[orig_idx];
                        reordered_diag[sorted_idx + 1] = H_soc_exp_diag_indices[orig_idx + 1];
                    }
                    H_soc_exp_diag_indices = reordered_diag;
                }
            }
        }

        // Reorder GenPowerCone H indices from original cone order to sorted cone order
        // so that sorted cone j's kernel writes go to the correct KKT positions.
        // H_genpow_indices has mixed content: dense cones → dim*(dim+1)/2 entries, sparse → dim entries
        // q/r/p/exp_diag indices only have entries from sparse cones (dim > 4)
        if (!cones.genPowerSortPerm.empty() && !H_genpow_indices.empty()) {
            const auto& perm = cones.genPowerSortPerm;  // perm[sorted_idx] = original_idx

            // Build prefix sums for H_genpow_indices (mixed: dense=tri, sparse=diag)
            // and for q/r/p (sparse-only)
            std::vector<int64_t> orig_hs_off(cones.numGenPowerCones + 1, 0);
            std::vector<int64_t> sorted_hs_off(cones.numGenPowerCones + 1, 0);
            // Sparse-only prefix sums for q/r/p
            std::vector<int64_t> orig_q_off(cones.numGenPowerCones + 1, 0);
            std::vector<int64_t> sorted_q_off(cones.numGenPowerCones + 1, 0);
            std::vector<int64_t> orig_r_off(cones.numGenPowerCones + 1, 0);
            std::vector<int64_t> sorted_r_off(cones.numGenPowerCones + 1, 0);
            std::vector<int64_t> orig_p_off(cones.numGenPowerCones + 1, 0);
            std::vector<int64_t> sorted_p_off(cones.numGenPowerCones + 1, 0);
            // Sparse cone count prefix sums for exp_diag (3 entries per sparse cone)
            std::vector<int64_t> orig_sparse_cnt(cones.numGenPowerCones + 1, 0);
            std::vector<int64_t> sorted_sparse_cnt(cones.numGenPowerCones + 1, 0);

            for (int64_t i = 0; i < cones.numGenPowerCones; ++i) {
                int64_t orig_dim1 = gpDim1sForKKT[i];
                int64_t orig_dim2 = gpDim2sForKKT[i];
                int64_t orig_dim = orig_dim1 + orig_dim2;
                int64_t orig_hs = (orig_dim <= 4) ? orig_dim * (orig_dim + 1) / 2 : orig_dim;
                orig_hs_off[i + 1] = orig_hs_off[i] + orig_hs;
                bool orig_sparse = (orig_dim > 4);
                if (orig_sparse) {
                    orig_q_off[i + 1] = orig_q_off[i] + orig_dim1;
                    orig_r_off[i + 1] = orig_r_off[i] + orig_dim2;
                    orig_p_off[i + 1] = orig_p_off[i] + orig_dim;
                    orig_sparse_cnt[i + 1] = orig_sparse_cnt[i] + 1;
                } else {
                    orig_q_off[i + 1] = orig_q_off[i];
                    orig_r_off[i + 1] = orig_r_off[i];
                    orig_p_off[i + 1] = orig_p_off[i];
                    orig_sparse_cnt[i + 1] = orig_sparse_cnt[i];
                }

                int64_t sorted_dim1 = cones.genPowerDim1s[i];
                int64_t sorted_dim2 = cones.genPowerDim2s[i];
                int64_t sorted_dim = sorted_dim1 + sorted_dim2;
                int64_t sorted_hs = (sorted_dim <= 4) ? sorted_dim * (sorted_dim + 1) / 2 : sorted_dim;
                sorted_hs_off[i + 1] = sorted_hs_off[i] + sorted_hs;
                bool sorted_sparse = (sorted_dim > 4);
                if (sorted_sparse) {
                    sorted_q_off[i + 1] = sorted_q_off[i] + sorted_dim1;
                    sorted_r_off[i + 1] = sorted_r_off[i] + sorted_dim2;
                    sorted_p_off[i + 1] = sorted_p_off[i] + sorted_dim;
                    sorted_sparse_cnt[i + 1] = sorted_sparse_cnt[i] + 1;
                } else {
                    sorted_q_off[i + 1] = sorted_q_off[i];
                    sorted_r_off[i + 1] = sorted_r_off[i];
                    sorted_p_off[i + 1] = sorted_p_off[i];
                    sorted_sparse_cnt[i + 1] = sorted_sparse_cnt[i];
                }
            }

            // Permute H_genpow_indices (mixed dense/sparse entries)
            {
                std::vector<int64_t> reordered(H_genpow_indices.size());
                for (int64_t j = 0; j < cones.numGenPowerCones; ++j) {
                    int64_t orig_cone = perm[j];
                    int64_t count = sorted_hs_off[j + 1] - sorted_hs_off[j];
                    for (int64_t k = 0; k < count; ++k)
                        reordered[sorted_hs_off[j] + k] = H_genpow_indices[orig_hs_off[orig_cone] + k];
                }
                H_genpow_indices = reordered;
            }
            // Permute q/r/p (sparse cones only)
            if (!H_genpow_q_indices.empty()) {
                std::vector<int64_t> reordered(H_genpow_q_indices.size());
                for (int64_t j = 0; j < cones.numGenPowerCones; ++j) {
                    int64_t orig_cone = perm[j];
                    int64_t sorted_dim = cones.genPowerDim1s[j] + cones.genPowerDim2s[j];
                    if (sorted_dim <= 4) continue;
                    int64_t count = cones.genPowerDim1s[j];
                    for (int64_t k = 0; k < count; ++k)
                        reordered[sorted_q_off[j] + k] = H_genpow_q_indices[orig_q_off[orig_cone] + k];
                }
                H_genpow_q_indices = reordered;
            }
            if (!H_genpow_r_indices.empty()) {
                std::vector<int64_t> reordered(H_genpow_r_indices.size());
                for (int64_t j = 0; j < cones.numGenPowerCones; ++j) {
                    int64_t orig_cone = perm[j];
                    int64_t sorted_dim = cones.genPowerDim1s[j] + cones.genPowerDim2s[j];
                    if (sorted_dim <= 4) continue;
                    int64_t count = cones.genPowerDim2s[j];
                    for (int64_t k = 0; k < count; ++k)
                        reordered[sorted_r_off[j] + k] = H_genpow_r_indices[orig_r_off[orig_cone] + k];
                }
                H_genpow_r_indices = reordered;
            }
            if (!H_genpow_p_indices.empty()) {
                std::vector<int64_t> reordered(H_genpow_p_indices.size());
                for (int64_t j = 0; j < cones.numGenPowerCones; ++j) {
                    int64_t orig_cone = perm[j];
                    int64_t sorted_dim = cones.genPowerDim1s[j] + cones.genPowerDim2s[j];
                    if (sorted_dim <= 4) continue;
                    int64_t count = cones.genPowerDim1s[j] + cones.genPowerDim2s[j];
                    for (int64_t k = 0; k < count; ++k)
                        reordered[sorted_p_off[j] + k] = H_genpow_p_indices[orig_p_off[orig_cone] + k];
                }
                H_genpow_p_indices = reordered;
            }
            if (!H_genpow_exp_diag_indices.empty()) {
                std::vector<int64_t> reordered(H_genpow_exp_diag_indices.size());
                for (int64_t j = 0; j < cones.numGenPowerCones; ++j) {
                    int64_t sorted_dim = cones.genPowerDim1s[j] + cones.genPowerDim2s[j];
                    if (sorted_dim <= 4) continue;
                    int64_t orig_cone = perm[j];
                    int64_t orig_idx = orig_sparse_cnt[orig_cone] * 9;
                    int64_t sorted_idx = sorted_sparse_cnt[j] * 9;
                    for (int kk = 0; kk < 9; ++kk) {
                        reordered[sorted_idx + kk] = H_genpow_exp_diag_indices[orig_idx + kk];
                    }
                }
                H_genpow_exp_diag_indices = reordered;
            }
            // Permute the 6 PD-axis off-diagonal column index arrays.
            // Each array has the same per-cone offset structure as
            // H_genpow_p_indices (one entry per cone-block row).
            for (int kk = 0; kk < 6; ++kk) {
                if (H_genpow_pd_axis_indices[kk].empty()) continue;
                std::vector<int64_t> reordered(H_genpow_pd_axis_indices[kk].size());
                for (int64_t j = 0; j < cones.numGenPowerCones; ++j) {
                    int64_t orig_cone = perm[j];
                    int64_t sorted_dim = cones.genPowerDim1s[j] + cones.genPowerDim2s[j];
                    if (sorted_dim <= 4) continue;
                    int64_t count = cones.genPowerDim1s[j] + cones.genPowerDim2s[j];
                    for (int64_t k = 0; k < count; ++k)
                        reordered[sorted_p_off[j] + k] = H_genpow_pd_axis_indices[kk][orig_p_off[orig_cone] + k];
                }
                H_genpow_pd_axis_indices[kk] = reordered;
            }
        }

        // Upload cone block indices
        int64_t* dH_soc = nullptr; int64_t* dH_exp = nullptr; int64_t* dH_power = nullptr;
        int64_t* dH_soc_u = nullptr; int64_t* dH_soc_v = nullptr; int64_t* dH_soc_exp_diag = nullptr;

        if (!H_soc_indices.empty()) {
            auto e = cudaMalloc(&dH_soc, sizeof(int64_t) * H_soc_indices.size());
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_soc, H_soc_indices.data(), sizeof(int64_t) * H_soc_indices.size(), cudaMemcpyHostToDevice, stream);
        }

        if (!H_soc_u_indices.empty()) {
            auto e = cudaMalloc(&dH_soc_u, sizeof(int64_t) * H_soc_u_indices.size());
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_soc_u, H_soc_u_indices.data(), sizeof(int64_t) * H_soc_u_indices.size(), cudaMemcpyHostToDevice, stream);
        }

        if (!H_soc_v_indices.empty()) {
            auto e = cudaMalloc(&dH_soc_v, sizeof(int64_t) * H_soc_v_indices.size());
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_soc_v, H_soc_v_indices.data(), sizeof(int64_t) * H_soc_v_indices.size(), cudaMemcpyHostToDevice, stream);
        }

        if (!H_soc_exp_diag_indices.empty()) {
            auto e = cudaMalloc(&dH_soc_exp_diag, sizeof(int64_t) * H_soc_exp_diag_indices.size());
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_soc_exp_diag, H_soc_exp_diag_indices.data(), sizeof(int64_t) * H_soc_exp_diag_indices.size(), cudaMemcpyHostToDevice, stream);
        }

        // Direct-x sparse SOC expansion indices: upload u/v
        // per-row slots and per-cone exp-diag slots.
        int64_t* dH_xcone_u = nullptr;
        int64_t* dH_xcone_v = nullptr;
        int64_t* dH_xcone_exp_diag = nullptr;
        if (!H_xcone_u_indices.empty()) {
            auto e = cudaMalloc(&dH_xcone_u,
                                sizeof(int64_t) * H_xcone_u_indices.size());
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_xcone_u, H_xcone_u_indices.data(),
                            sizeof(int64_t) * H_xcone_u_indices.size(),
                            cudaMemcpyHostToDevice, stream);
        }
        if (!H_xcone_v_indices.empty()) {
            auto e = cudaMalloc(&dH_xcone_v,
                                sizeof(int64_t) * H_xcone_v_indices.size());
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_xcone_v, H_xcone_v_indices.data(),
                            sizeof(int64_t) * H_xcone_v_indices.size(),
                            cudaMemcpyHostToDevice, stream);
        }
        if (!H_xcone_exp_diag_indices.empty()) {
            auto e = cudaMalloc(&dH_xcone_exp_diag,
                                sizeof(int64_t) * H_xcone_exp_diag_indices.size());
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_xcone_exp_diag,
                            H_xcone_exp_diag_indices.data(),
                            sizeof(int64_t) * H_xcone_exp_diag_indices.size(),
                            cudaMemcpyHostToDevice, stream);
        }

        // Direct-x sparse GenPow expansion indices (Phase 4): upload q/r/p
        // per-row slots and exp-diag slots (stored in H_xcone_exp_diag_indices,
        // appended after x-SOC entries).
        int64_t* dH_xcone_genpow_q = nullptr;
        int64_t* dH_xcone_genpow_r = nullptr;
        int64_t* dH_xcone_genpow_p = nullptr;
        if (!H_xcone_genpow_q_indices.empty()) {
            auto e = cudaMalloc(&dH_xcone_genpow_q,
                                sizeof(int64_t) * H_xcone_genpow_q_indices.size());
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_xcone_genpow_q, H_xcone_genpow_q_indices.data(),
                            sizeof(int64_t) * H_xcone_genpow_q_indices.size(),
                            cudaMemcpyHostToDevice, stream);
        }
        if (!H_xcone_genpow_r_indices.empty()) {
            auto e = cudaMalloc(&dH_xcone_genpow_r,
                                sizeof(int64_t) * H_xcone_genpow_r_indices.size());
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_xcone_genpow_r, H_xcone_genpow_r_indices.data(),
                            sizeof(int64_t) * H_xcone_genpow_r_indices.size(),
                            cudaMemcpyHostToDevice, stream);
        }
        if (!H_xcone_genpow_p_indices.empty()) {
            auto e = cudaMalloc(&dH_xcone_genpow_p,
                                sizeof(int64_t) * H_xcone_genpow_p_indices.size());
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_xcone_genpow_p, H_xcone_genpow_p_indices.data(),
                            sizeof(int64_t) * H_xcone_genpow_p_indices.size(),
                            cudaMemcpyHostToDevice, stream);
        }

        // Direct-x sparse GenPow PD-axis off-diag column index arrays (6).
        // One slot per cone-block row, aligned with H_xcone_genpow_p_indices.
        int64_t* dH_xcone_genpow_pd_axis[6] = {nullptr, nullptr, nullptr,
                                               nullptr, nullptr, nullptr};
        for (int kk = 0; kk < 6; ++kk) {
            if (H_xcone_genpow_pd_axis_indices[kk].empty()) continue;
            auto e = cudaMalloc(
                &dH_xcone_genpow_pd_axis[kk],
                sizeof(int64_t) * H_xcone_genpow_pd_axis_indices[kk].size());
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(
                dH_xcone_genpow_pd_axis[kk],
                H_xcone_genpow_pd_axis_indices[kk].data(),
                sizeof(int64_t) * H_xcone_genpow_pd_axis_indices[kk].size(),
                cudaMemcpyHostToDevice, stream);
        }

        if (cones.numExpCones > 0) {
            auto e = cudaMalloc(&dH_exp, sizeof(int64_t) * (size_t)(cones.numExpCones * 6));
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_exp, H_exp_indices.data(), sizeof(int64_t) * (size_t)(cones.numExpCones * 6), cudaMemcpyHostToDevice, stream);
        }

        if (cones.numPowerCones > 0) {
            auto e = cudaMalloc(&dH_power, sizeof(int64_t) * (size_t)(cones.numPowerCones * 6));
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_power, H_power_indices.data(), sizeof(int64_t) * (size_t)(cones.numPowerCones * 6), cudaMemcpyHostToDevice, stream);
        }

        // Reorder PSD indices from original order to sorted order (like SOC)
        int64_t* dH_psd = nullptr;
        if (!H_psd_indices.empty() && !cones.psdSortPerm.empty()) {
            const auto& perm = cones.psdSortPerm;
            auto psdHsForDim = [](int64_t dim) -> int64_t {
                int64_t svec_dim = dim * (dim + 1) / 2;
                return svec_dim * (svec_dim + 1) / 2;
            };
            std::vector<int64_t> orig_hs_off(cones.numPsdCones + 1, 0);
            std::vector<int64_t> sorted_hs_off(cones.numPsdCones + 1, 0);
            for (int64_t i = 0; i < cones.numPsdCones; ++i) {
                orig_hs_off[i + 1] = orig_hs_off[i] + psdHsForDim(cones.psdConeDimsOriginal[i]);
                sorted_hs_off[i + 1] = sorted_hs_off[i] + psdHsForDim(cones.psdConeDims[i]);
            }
            std::vector<int64_t> reordered_psd(H_psd_indices.size());
            for (int64_t j = 0; j < cones.numPsdCones; ++j) {
                int64_t orig_cone = perm[j];
                int64_t orig_start = orig_hs_off[orig_cone];
                int64_t sorted_start = sorted_hs_off[j];
                int64_t count = psdHsForDim(cones.psdConeDims[j]);
                for (int64_t k = 0; k < count; ++k) {
                    reordered_psd[sorted_start + k] = H_psd_indices[orig_start + k];
                }
            }
            H_psd_indices = reordered_psd;
        }
        if (!H_psd_indices.empty()) {
            auto e = cudaMalloc(&dH_psd, sizeof(int64_t) * H_psd_indices.size());
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_psd, H_psd_indices.data(), sizeof(int64_t) * H_psd_indices.size(), cudaMemcpyHostToDevice, stream);
        }

        int64_t* dH_genpow = nullptr;
        int64_t* dH_genpow_q = nullptr;
        int64_t* dH_genpow_r = nullptr;
        int64_t* dH_genpow_p = nullptr;
        int64_t* dH_genpow_exp_diag = nullptr;
        int64_t* dH_genpow_pd_axis[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

        if (!H_genpow_indices.empty()) {
            auto e = cudaMalloc(&dH_genpow, sizeof(int64_t) * H_genpow_indices.size());
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_genpow, H_genpow_indices.data(), sizeof(int64_t) * H_genpow_indices.size(), cudaMemcpyHostToDevice, stream);
        }

        if (!H_genpow_q_indices.empty()) {
            auto e = cudaMalloc(&dH_genpow_q, sizeof(int64_t) * H_genpow_q_indices.size());
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_genpow_q, H_genpow_q_indices.data(), sizeof(int64_t) * H_genpow_q_indices.size(), cudaMemcpyHostToDevice, stream);
        }

        if (!H_genpow_r_indices.empty()) {
            auto e = cudaMalloc(&dH_genpow_r, sizeof(int64_t) * H_genpow_r_indices.size());
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_genpow_r, H_genpow_r_indices.data(), sizeof(int64_t) * H_genpow_r_indices.size(), cudaMemcpyHostToDevice, stream);
        }

        if (!H_genpow_p_indices.empty()) {
            auto e = cudaMalloc(&dH_genpow_p, sizeof(int64_t) * H_genpow_p_indices.size());
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_genpow_p, H_genpow_p_indices.data(), sizeof(int64_t) * H_genpow_p_indices.size(), cudaMemcpyHostToDevice, stream);
        }

        if (!H_genpow_exp_diag_indices.empty()) {
            auto e = cudaMalloc(&dH_genpow_exp_diag, sizeof(int64_t) * H_genpow_exp_diag_indices.size());
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_genpow_exp_diag, H_genpow_exp_diag_indices.data(), sizeof(int64_t) * H_genpow_exp_diag_indices.size(), cudaMemcpyHostToDevice, stream);
        }

        for (int kk = 0; kk < 6; ++kk) {
            if (H_genpow_pd_axis_indices[kk].empty()) continue;
            auto e = cudaMalloc(&dH_genpow_pd_axis[kk], sizeof(int64_t) * H_genpow_pd_axis_indices[kk].size());
            if (e == cudaErrorMemoryAllocation) throw std::bad_alloc{};
            if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
            cudaMemcpyAsync(dH_genpow_pd_axis[kk], H_genpow_pd_axis_indices[kk].data(),
                            sizeof(int64_t) * H_genpow_pd_axis_indices[kk].size(),
                            cudaMemcpyHostToDevice, stream);
        }

        cudaStreamSynchronize(stream);

        // Initialize regularization data structures
        // Build diag_full: indices of all diagonal entries in KKT (CSR upper triangle)
        // For CSR upper triangle, diagonal is the FIRST entry in each row
        std::vector<int64_t> diag_full_host(N);
        for (int64_t i = 0; i < N; ++i) {
            diag_full_host[i] = rowOff[i];  // First entry in each row
        }

        // Build dsigns: +1 for P block (first n), -1 for H block (next m),
        // [-1,+1] per sparse SOC, [-1,-1,+1] per GenPowerCone
        std::vector<int8_t> dsigns_host(N);
        for (int64_t i = 0; i < n; ++i) dsigns_host[i] = 1;
        for (int64_t i = n; i < n + m; ++i) dsigns_host[i] = -1;
        // Expansion rows (matching Rust convention)
        {
            int64_t exp_idx = n + m;
            // SOC sparse expansion: [-1, +1] per cone
            for (int64_t i = 0; i < cones.numSocCones; ++i) {
                int64_t dim = socDimsAvailable ? socDimsForKKT[i] : 3;
                if (dim > 4) {
                    dsigns_host[exp_idx] = -1;     // v column (diag = -η²)
                    dsigns_host[exp_idx + 1] = 1;  // u column (diag = +η²)
                    exp_idx += 2;
                }
            }
            // GenPowerCone expansion: [-1, -1, +1] (q, r, p) followed by
            // 6 PD-axis slots. PD-axis dsigns are dynamic per iteration —
            // refreshed by `refresh_genpow_pd_dsigns` from runtime pd_signs
            // after every scaling update (called in update_H below). The
            // +1 default here is just the post-construction state when PD
            // scaling is inactive (matches the +sign·1 sentinel).
            for (int64_t i = 0; i < cones.numGenPowerCones; ++i) {
                int64_t dim = gpDim1sForKKT[i] + gpDim2sForKKT[i];
                if (dim > 4) {
                    dsigns_host[exp_idx] = -1;     // q column
                    dsigns_host[exp_idx + 1] = -1; // r column
                    dsigns_host[exp_idx + 2] = 1;  // p column
                    for (int kk = 0; kk < 6; ++kk) {
                        dsigns_host[exp_idx + 3 + kk] = 1;  // PD-axis (dynamic)
                    }
                    exp_idx += 9;
                }
            }
            // Direct-x sparse SOC expansion: [+1, -1] per sparse x-cone.
            // OPPOSITE of slack's [-1, +1] because direct-x contributes
            // `+Hs` to the (1,1) block (slack contributes `-Hs`). Paired
            // with the sign-flipped u/v column values in
            // refresh_xcone_sparse_expansion, the Schur elimination still
            // yields the rank-2 Hs contribution with the direct-x sign.
            for (int64_t s = 0; s < num_sparse_xsoc; ++s) {
                dsigns_host[exp_idx] = 1;      // v col (diag = +η²)
                dsigns_host[exp_idx + 1] = -1; // u col (diag = -η²)
                exp_idx += 2;
            }
            // Direct-x sparse GenPow expansion: [+1, +1, -1] for q,r,p, then
            // 6 PD-axis dsigns. q/r dsigns +1 flip the contribution sign
            // giving -μ·qq', -μ·rr'; p dsign -1 gives +μ·pp'.
            //
            // PD-axis dsigns default to -1, matching the inactive sentinel
            // written by `apply_xgenpow_pd_to_kkt` (sent_sign = -sign[k] =
            // -1 for default sign[k] = +1; see genpow_pd_kernels.cu:864).
            // The PD-axis dsigns are refreshed per iteration by
            // `refresh_xgenpow_pd_dsigns` (called from `refresh_xcone_hs`)
            // to track sign flips in pd_signs.
            for (int64_t s = 0; s < num_sparse_xgenpow; ++s) {
                dsigns_host[exp_idx]     =  1;  // q col diag → -μ·qq'
                dsigns_host[exp_idx + 1] =  1;  // r col diag → -μ·rr'
                dsigns_host[exp_idx + 2] = -1;  // p col diag → +μ·pp'
                for (int kk = 0; kk < 6; ++kk) {
                    dsigns_host[exp_idx + 3 + kk] = -1;
                }
                exp_idx += 9;
            }
        }

        // Direct-x cone KKT index map. Flat array of
        // KKT.values slots, one per Hs entry. Layout per cone:
        //   - Nonneg: k diagonal entries (KKT slot = Pdiag[J[p]])
        //   - SOC dense (dim <= 4): k*(k+1)/2 column-major upper-tri
        //     entries over sorted_indices. Each entry (r_logical,
        //     c_logical) with r_logical <= c_logical maps to KKT slot at
        //     row sorted_J[r_logical], col sorted_J[c_logical], resolved
        //     by scanning rowOff/colIdx for that (row, col) pair.
        //
        // Read cones.x_cones directly (authoritative source) — cones.numXCones
        // is only populated by Cones::initialize(), which may not have run
        // when KKTData is constructed in isolation (e.g. unit tests).
        int64_t* d_H_xcone_hs = nullptr;
        if (!cones.x_cones.empty()) {
            // Count Hs entries per cone in cones.x_cones order.
            int64_t total_hs = 0;
            for (const auto& xc : cones.x_cones) {
                const int64_t k = static_cast<int64_t>(xc.indices.size());
                if (xc.kind == XConeKind::Nonneg) {
                    total_hs += k;
                } else if (xc.kind == XConeKind::SOC) {
                    // Dense: full upper tri; Sparse: diagonal only
                    // (off-diagonals carried by u/v expansion cols).
                    total_hs += (k > 4) ? k : k * (k + 1) / 2;
                } else if (xc.kind == XConeKind::Exp ||
                           xc.kind == XConeKind::Power) {
                    // 3D dense asymmetric cones: 6 packed upper-tri entries.
                    total_hs += k * (k + 1) / 2;
                } else if (xc.kind == XConeKind::GenPower) {
                    // Dense (k<=4): full upper-tri; Sparse (k>4): diagonal only.
                    total_hs += (k > 4) ? k : k * (k + 1) / 2;
                } else if (xc.kind == XConeKind::PSD) {
                    // svec_dim = k = indices.size(); Hs is svec_dim × svec_dim
                    // upper-tri (always dense, no sparse expansion path).
                    total_hs += k * (k + 1) / 2;
                }
            }

            // Helper: find KKT.values slot for (row, col) by scanning the
            // emitted colIdx in that row. O(row-density); fine for k ≤ 4.
            auto find_slot = [&](int64_t row, int64_t col) -> int64_t {
                for (int64_t t = rowOff[(size_t)row]; t < rowOff[(size_t)row + 1]; ++t) {
                    if (colIdx[(size_t)t] == col) return t;
                }
                throw std::runtime_error(
                    "KKTData: direct-x (row=" + std::to_string(row) +
                    ", col=" + std::to_string(col) + ") missing from KKT");
            };

            std::vector<int64_t> xcone_hs_host;
            xcone_hs_host.reserve((size_t)total_hs);
            for (const auto& xc : cones.x_cones) {
                const int64_t k = static_cast<int64_t>(xc.indices.size());
                if (xc.kind == XConeKind::Nonneg) {
                    for (int64_t idx : xc.indices) {
                        xcone_hs_host.push_back(Pdiag[(size_t)idx]);
                    }
                } else if (xc.kind == XConeKind::SOC && k > 4) {
                    // Sparse SOC: k diagonal entries on Pdiag. The scaling
                    // kernel writes a head-specific value `η²·d` at
                    // cone-internal position 0 and `η²` at positions 1..k-1
                    // (kernels.cu:878-892). So the diagonal slot at
                    // cone-internal i must point at Pdiag[xc.indices[i]] —
                    // mapping cone-internal head/tail through the user's
                    // possibly-unsorted index list.
                    for (int64_t i = 0; i < k; ++i) {
                        xcone_hs_host.push_back(
                            Pdiag[(size_t)xc.indices[(size_t)i]]);
                    }
                } else if (xc.kind == XConeKind::SOC) {  // SOC dense (k <= 4)
                    // Cone-internal column-major upper-tri walk: matches the
                    // scaling kernel's `for cc, rr <= cc` over cone-internal
                    // (rr, cc) and the (1,0)/(0,0) head/tail structure of
                    // SOC's `η²·(2ww' − J)`. Each (r_logical, c_logical) maps
                    // to KKT cell (min(idx[r], idx[c]), max(idx[r], idx[c]))
                    // — H is symmetric so the upper-tri slot is the same
                    // regardless of which logical index is row vs col. This
                    // matches PSD direct-x's pattern at the next branch and
                    // lifts the implicit "indices must be ascending"
                    // restriction.
                    for (int64_t c_logical = 0; c_logical < k; ++c_logical) {
                        for (int64_t r_logical = 0; r_logical <= c_logical; ++r_logical) {
                            int64_t a = xc.indices[(size_t)r_logical];
                            int64_t b = xc.indices[(size_t)c_logical];
                            int64_t row = a < b ? a : b;
                            int64_t col = a < b ? b : a;
                            xcone_hs_host.push_back(find_slot(row, col));
                        }
                    }
                } else if (xc.kind == XConeKind::Exp ||
                           xc.kind == XConeKind::Power) {
                    // 3D dense asymmetric cones: cone-internal column-major
                    // upper-tri walk (same pattern as PSD/SOC dense above).
                    // x[0]/x[1]/x[2] are the cone-internal positions; the
                    // KKT cell for each (r_logical, c_logical) pair is at
                    // (min, max) of the user's index pair.
                    for (int64_t c_logical = 0; c_logical < k; ++c_logical) {
                        for (int64_t r_logical = 0; r_logical <= c_logical; ++r_logical) {
                            int64_t a = xc.indices[(size_t)r_logical];
                            int64_t b = xc.indices[(size_t)c_logical];
                            int64_t row = a < b ? a : b;
                            int64_t col = a < b ? b : a;
                            xcone_hs_host.push_back(find_slot(row, col));
                        }
                    }
                } else if (xc.kind == XConeKind::GenPower && k > 4) {
                    // Sparse GenPow: k diagonal entries on Pdiag. The
                    // scaling kernel writes d1[i] (depends on αᵢ) for
                    // i ∈ [0, dim1) and d2 (shared) for i ∈ [dim1, dim)
                    // — both indexed by **cone-internal** position. The
                    // matching diagonal slot is therefore at
                    // Pdiag[xc.indices[i]], not Pdiag[sorted_J[i]].
                    for (int64_t i = 0; i < k; ++i) {
                        xcone_hs_host.push_back(
                            Pdiag[(size_t)xc.indices[(size_t)i]]);
                    }
                } else if (xc.kind == XConeKind::GenPower) {  // dense k <= 4
                    // Dense GenPow: cone-internal column-major upper-tri
                    // walk (same pattern as SOC/Exp/Power dense above).
                    // The cone's (p, w) layout — p at cone positions
                    // [0, dim1) and w at [dim1, dim) — is preserved by the
                    // scaling kernel, so logical-position iteration matches
                    // its writes.
                    for (int64_t c_logical = 0; c_logical < k; ++c_logical) {
                        for (int64_t r_logical = 0; r_logical <= c_logical; ++r_logical) {
                            int64_t a = xc.indices[(size_t)r_logical];
                            int64_t b = xc.indices[(size_t)c_logical];
                            int64_t row = a < b ? a : b;
                            int64_t col = a < b ? b : a;
                            xcone_hs_host.push_back(find_slot(row, col));
                        }
                    }
                } else if (xc.kind == XConeKind::PSD) {
                    // PSD direct-x: skron_kernel writes the dense svec×svec
                    // upper-tri Hessian in cone-local svec order, where
                    // flat index `pos = c_logical*(c_logical+1)/2 + r_logical`
                    // (with r_logical ≤ c_logical) maps to Hessian entry
                    // (svec_pos[r_logical], svec_pos[c_logical]) in
                    // problem space — i.e., (xc.indices[r_logical],
                    // xc.indices[c_logical]). Since xc.indices may be in
                    // any order (the user defines svec layout), use
                    // (min, max) to find the upper-triangle CSR slot.
                    for (int64_t c_logical = 0; c_logical < k; ++c_logical) {
                        for (int64_t r_logical = 0; r_logical <= c_logical; ++r_logical) {
                            int64_t a = xc.indices[(size_t)r_logical];
                            int64_t b = xc.indices[(size_t)c_logical];
                            int64_t row = a < b ? a : b;
                            int64_t col = a < b ? b : a;
                            xcone_hs_host.push_back(find_slot(row, col));
                        }
                    }
                }
            }

            if (total_hs > 0) {
                auto e_xc = cudaMalloc(&d_H_xcone_hs,
                                       sizeof(int64_t) * (size_t)total_hs);
                if (e_xc != cudaSuccess)
                    throw std::runtime_error(cudaGetErrorString(e_xc));
                cudaMemcpyAsync(d_H_xcone_hs, xcone_hs_host.data(),
                                sizeof(int64_t) * (size_t)total_hs,
                                cudaMemcpyHostToDevice, stream);
            }
        }

        // Upload to device
        int8_t* d_dsigns = nullptr;
        int64_t* d_diag_full = nullptr;
        double* d_work_diag = nullptr;

        auto e = cudaMalloc(&d_dsigns, sizeof(int8_t) * N);
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
        cudaMemcpyAsync(d_dsigns, dsigns_host.data(), sizeof(int8_t) * N, cudaMemcpyHostToDevice, stream);

        e = cudaMalloc(&d_diag_full, sizeof(int64_t) * N);
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
        cudaMemcpyAsync(d_diag_full, diag_full_host.data(), sizeof(int64_t) * N, cudaMemcpyHostToDevice, stream);

        e = cudaMalloc(&d_work_diag, sizeof(double) * N * batchSize);
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));

        cudaStreamSynchronize(stream);

        // Finish RAII state
        KKT = std::move(kkt);
        P_diagIdx_.reset(dP);
        H_diagIdx_.reset(dH);
        Annz_idx_.reset(dAnnz);
        Pnnz_idx_.reset(dPnnz);
        H_soc_idx_.reset(dH_soc);
        H_exp_idx_.reset(dH_exp);
        H_power_idx_.reset(dH_power);
        H_genpow_idx_.reset(dH_genpow);
        H_genpow_q_idx_.reset(dH_genpow_q);
        H_genpow_r_idx_.reset(dH_genpow_r);
        H_genpow_p_idx_.reset(dH_genpow_p);
        H_genpow_exp_diag_idx_.reset(dH_genpow_exp_diag);
        for (int kk = 0; kk < 6; ++kk) {
            H_genpow_pd_axis_idx_[kk].reset(dH_genpow_pd_axis[kk]);
        }
        H_soc_u_idx_.reset(dH_soc_u);
        H_soc_v_idx_.reset(dH_soc_v);
        H_soc_exp_diag_idx_.reset(dH_soc_exp_diag);
        H_xcone_hs_idx_.reset(d_H_xcone_hs);
        H_xcone_u_idx_.reset(dH_xcone_u);
        H_xcone_v_idx_.reset(dH_xcone_v);
        H_xcone_exp_diag_idx_.reset(dH_xcone_exp_diag);
        H_xcone_genpow_q_idx_.reset(dH_xcone_genpow_q);
        H_xcone_genpow_r_idx_.reset(dH_xcone_genpow_r);
        H_xcone_genpow_p_idx_.reset(dH_xcone_genpow_p);
        for (int kk = 0; kk < 6; ++kk) {
            H_xcone_genpow_pd_axis_idx_[kk].reset(dH_xcone_genpow_pd_axis[kk]);
        }
        H_psd_idx_.reset(dH_psd);
        dsigns_.reset(d_dsigns);
        diag_full_.reset(d_diag_full);
        work_diag_.reset(d_work_diag);

        // Initialize work vector for per-batch diagonal norms
        max_diag_per_batch_ = std::make_unique<BatchedVector>(1, batchSize);

        // Allocate the direct-x cone P-baseline buffer when any x-cone is
        // present. Actual P values are snapshotted into this buffer by
        // init_xcone_px_baseline() after populate() runs.
        if (!cones.x_cones.empty()) {
            int64_t total_hs = 0;
            for (const auto& xc : cones.x_cones) {
                const int64_t k = static_cast<int64_t>(xc.indices.size());
                if (xc.kind == XConeKind::Nonneg) {
                    total_hs += k;
                } else if (xc.kind == XConeKind::SOC) {
                    total_hs += (k > 4) ? k : k * (k + 1) / 2;
                } else if (xc.kind == XConeKind::Exp ||
                           xc.kind == XConeKind::Power) {
                    total_hs += k * (k + 1) / 2;
                } else if (xc.kind == XConeKind::GenPower) {
                    total_hs += (k > 4) ? k : k * (k + 1) / 2;
                } else if (xc.kind == XConeKind::PSD) {
                    // svec_dim × svec_dim upper-tri (always dense).
                    total_hs += k * (k + 1) / 2;
                }
            }
            if (total_hs > 0) {
                xcone_px_baseline_ =
                    std::make_unique<BatchedVector>(total_hs, batchSize);
                xcone_px_baseline_->setToConstant(0.0, stream);
            }
        }

        // Initialize cuDSS for factorization
        initialize_cudss(stream, CuDSSStrategy::Auto, maxLuNnz_);
    }

    /// Snapshot KKT.values at every direct-x cone Hs slot into the
    /// px_baseline buffer. Call once after populate() + regularization;
    /// subsequent IPM iterations add Hs on top via refresh_xcone_hs().
    void init_xcone_px_baseline(const Cones& cones,
                                cudaStream_t stream = 0) override
    {
        if (!xcone_px_baseline_ || !H_xcone_hs_idx_) return;
        const int64_t total_hs = xcone_px_baseline_->n();
        snapshot_kkt_at_xcone_slots(
            KKT.values(), xcone_px_baseline_->data(),
            H_xcone_hs_idx_.get(),
            batchSize, KKT.nnz(), total_hs, stream);
    }

    /// Raw pointer pack the fused scale+scatter path uses to write x-cone
    /// Hs / expansion values directly into KKT.values, avoiding the extra
    /// global round-trip that `refresh_xcone_hs` otherwise pays.
    XConeScatterTargets xcone_scatter_targets() override {
        XConeScatterTargets t;
        t.kkt_values             = KKT.values();
        t.xcone_px_baseline      = xcone_px_baseline_ ? xcone_px_baseline_->data() : nullptr;
        t.H_xcone_hs_idx         = H_xcone_hs_idx_.get();
        t.H_xcone_u_idx          = H_xcone_u_idx_.get();
        t.H_xcone_v_idx          = H_xcone_v_idx_.get();
        t.H_xcone_exp_diag_idx   = H_xcone_exp_diag_idx_.get();
        t.H_xcone_genpow_q_idx   = H_xcone_genpow_q_idx_.get();
        t.H_xcone_genpow_r_idx   = H_xcone_genpow_r_idx_.get();
        t.H_xcone_genpow_p_idx   = H_xcone_genpow_p_idx_.get();
        for (int kk = 0; kk < 6; ++kk) {
            t.H_xcone_genpow_pd_axis_idx[kk] =
                H_xcone_genpow_pd_axis_idx_[kk].get();
        }
        // exp_diag entries: [2*numSparseXSoc SOC entries, 3*numSparseXGenPow GenPow entries]
        // The offset into H_xcone_exp_diag_idx_ where GenPow entries start equals
        // 2 * numSparseXSoc, but we don't track that count here directly.
        // We pass this as 0 for now; the kernel receives the pointer to the
        // start of all xcone exp diag entries and uses the offset to find GenPow.
        // The caller (kernels.cu) needs to know the SOC count separately.
        // We store it inline: offset = 2*numSparseXSoc (from Cones).
        t.xcone_genpow_exp_diag_offset = 0;  // filled in by caller via cones.numSparseXSoc
        t.nnzKKT                 = KKT.nnz();
        return t;
    }

    /// Refresh the KKT (1,1) block contributions coming from x-cone Hs.
    /// Must be called after each x-cone scaling update and before the
    /// KKT linear solve. For rank-2 sparse SOC x-cones (dim > 4) this
    /// also scatters the u/v expansion columns and diag writeback.
    ///
    /// NOTE: With the fused scale+scatter path enabled this becomes a
    /// no-op — `update_xcones_*_scaling` writes directly to KKT.values.
    /// The method is retained for compatibility with any callers that
    /// run scaling via the classic (non-fused) pass.
    void refresh_xcone_hs(const Cones& cones,
                          cudaStream_t stream = 0) override
    {
        if (!xcone_px_baseline_ || !H_xcone_hs_idx_) return;
        const int64_t total_hs = xcone_px_baseline_->n();
        ::moreau::refresh_xcone_hs(
            KKT.values(),
            cones.xcone_Hs.data(),
            xcone_px_baseline_->data(),
            H_xcone_hs_idx_.get(),
            batchSize, KKT.nnz(), total_hs, stream);

        if (cones.numSparseXSoc > 0 && H_xcone_u_idx_ && H_xcone_v_idx_) {
            ::moreau::refresh_xcone_sparse_expansion(
                KKT.values(),
                cones.xcone_u.data(),
                cones.xcone_v.data(),
                cones.xcone_eta.data(),
                cones.d_xcone_dims,
                cones.d_xcone_sparse_indices,
                cones.d_xcone_sparse_offsets,
                cones.d_xcone_cone_pos_for_sorted,
                cones.d_xcone_numel_offsets,
                H_xcone_u_idx_.get(),
                H_xcone_v_idx_.get(),
                H_xcone_exp_diag_idx_.get(),
                batchSize, KKT.nnz(),
                cones.numXCones,
                cones.totalSparseXSocDim,
                cones.numSparseXSoc,
                stream);
        }
        if (cones.numSparseXGenPow > 0 &&
            H_xcone_genpow_q_idx_ && H_xcone_genpow_r_idx_ && H_xcone_genpow_p_idx_) {
            // The xcone exp_diag array has SOC entries first (2*numSparseXSoc),
            // then GenPow entries (9*numSparseXGenPow): q,r,p + 6 PD axes per cone.
            const int64_t* genpow_exp_diag_ptr =
                H_xcone_exp_diag_idx_
                    ? H_xcone_exp_diag_idx_.get() + 2 * cones.numSparseXSoc
                    : nullptr;
            ::moreau::refresh_xcone_genpow_expansion(
                KKT.values(),
                cones.xcone_genpow_p.data(),
                cones.xcone_genpow_q.data(),
                cones.xcone_genpow_r.data(),
                cones.d_xcone_genpow_sparse_offsets,
                cones.d_xcone_genpow_sparse_alpha_offsets,
                cones.d_xcone_genpow_sparse_q_offsets,
                cones.d_xcone_genpow_sparse_r_offsets,
                cones.d_xcone_genpow_sparse_to_gidx,
                cones.d_xcone_genpow_dim1s,
                cones.d_xcone_genpow_dim2s,
                cones.d_xcone_genpow_dim_offsets,
                H_xcone_genpow_q_idx_.get(),
                H_xcone_genpow_r_idx_.get(),
                H_xcone_genpow_p_idx_.get(),
                genpow_exp_diag_ptr,
                batchSize, KKT.nnz(),
                cones.numSparseXGenPow,
                cones.totalXGenPowerDim,
                cones.totalXGenPowerAlphas,
                cones.totalXGenPowerDim2,
                stream);

            // Rank-9 PD axis scatter: writes the 6 PD off-diag columns and
            // 6 PD diagonals per sparse genpow cone. Uses active values when
            // pd_active=1 (computed in compute_xgenpow_pd_axes); writes
            // inactive sentinels (tiny ε + sign·1) otherwise so cuDSS sees a
            // structurally non-singular block.
            ::moreau::apply_xgenpow_pd_to_kkt(
                KKT.values(),
                cones.d_xcone_kinds, cones.d_xcone_dims,
                cones.d_xcone_genpow_idx,
                cones.d_xcone_genpow_dim1s,
                cones.d_xcone_genpow_dim2s,
                cones.d_xcone_genpow_dim_offsets,
                cones.d_xcone_genpow_sparse_idx,
                cones.d_xcone_genpow_sparse_offsets,
                cones.xcone_genpow_pd_axes.data(),
                cones.xcone_genpow_pd_coefs.data(),
                cones.xcone_genpow_pd_signs.data(),
                cones.xcone_genpow_pd_active.data(),
                H_xcone_genpow_pd_axis_idx_[0].get(),
                H_xcone_genpow_pd_axis_idx_[1].get(),
                H_xcone_genpow_pd_axis_idx_[2].get(),
                H_xcone_genpow_pd_axis_idx_[3].get(),
                H_xcone_genpow_pd_axis_idx_[4].get(),
                H_xcone_genpow_pd_axis_idx_[5].get(),
                genpow_exp_diag_ptr,
                cones.numXCones,
                cones.numXGenPowerCones,
                cones.totalXGenPowerDim,
                KKT.nnz(),
                batchSize,
                stream);

            // Refresh the dsigns slots for direct-x GenPow PD axes
            // from `pd_signs` (batch 0) so static regularization adds
            // ε with the correct sign. Without this, when `pd_signs[k]`
            // flips to -1 the sentinel written above is `+1` but
            // dsigns still says `-1`, and the regularizer subtracts ε
            // from a positive diagonal — cuDSS pivoting absorbs the
            // mismatch but the regularization becomes ineffective at
            // the affected slots.
            //
            // pd_axis_base_col = first direct-x GenPow PD-axis-0 col
            //   = n + m + p_slack + 2·num_sparse_xsoc, where
            //     p_slack = 2·numSparseSoc + 9·numSparseGenPow
            //     num_sparse_xsoc = cones.numSparseXSoc
            if (dsigns_) {
                int64_t pd_axis_base_col =
                    n + m
                    + 2 * cones.numSparseSoc
                    + 9 * cones.numSparseGenPow
                    + 2 * cones.numSparseXSoc;
                ::moreau::refresh_xgenpow_pd_dsigns(
                    dsigns_.get(),
                    cones.xcone_genpow_pd_signs.data(),
                    cones.d_xcone_kinds,
                    cones.d_xcone_genpow_idx,
                    cones.d_xcone_genpow_sparse_idx,
                    cones.numXCones,
                    cones.numXGenPowerCones,
                    pd_axis_base_col,
                    stream);
            }
        }
    }

    // accessors
    [[nodiscard]] int64_t* PDiagIndex() noexcept { return P_diagIdx_.get(); }
    [[nodiscard]] int64_t* HDiagIndex() noexcept { return H_diagIdx_.get(); }
    [[nodiscard]] int64_t* PNNZIndex() noexcept { return Pnnz_idx_.get(); }
    [[nodiscard]] int64_t* ANNZIndex() noexcept { return Annz_idx_.get(); }
    [[nodiscard]] int64_t* HSOCIndex() noexcept { return H_soc_idx_.get(); }
    [[nodiscard]] int64_t* HExpIndex() noexcept { return H_exp_idx_.get(); }
    [[nodiscard]] int64_t* HPowerIndex() noexcept { return H_power_idx_.get(); }
    [[nodiscard]] int64_t* HGenPowIndex() noexcept { return H_genpow_idx_.get(); }
    [[nodiscard]] int64_t* HGenPowQIndex() noexcept { return H_genpow_q_idx_.get(); }
    [[nodiscard]] int64_t* HGenPowRIndex() noexcept { return H_genpow_r_idx_.get(); }
    [[nodiscard]] int64_t* HGenPowPIndex() noexcept { return H_genpow_p_idx_.get(); }
    [[nodiscard]] int64_t* HGenPowExpDiagIndex() noexcept { return H_genpow_exp_diag_idx_.get(); }
    [[nodiscard]] int64_t* HSOCUIndex() noexcept { return H_soc_u_idx_.get(); }
    [[nodiscard]] int64_t* HSOCVIndex() noexcept { return H_soc_v_idx_.get(); }
    [[nodiscard]] int64_t* HSOCExpDiagIndex() noexcept { return H_soc_exp_diag_idx_.get(); }
    [[nodiscard]] int64_t* HXConeHsIndex() noexcept { return H_xcone_hs_idx_.get(); }
    [[nodiscard]] int64_t* HXConeUIndex() noexcept { return H_xcone_u_idx_.get(); }
    [[nodiscard]] int64_t* HXConeVIndex() noexcept { return H_xcone_v_idx_.get(); }
    [[nodiscard]] int64_t* HXConeExpDiagIndex() noexcept { return H_xcone_exp_diag_idx_.get(); }
    [[nodiscard]] int64_t* HXConeGenPowQIndex() noexcept { return H_xcone_genpow_q_idx_.get(); }
    [[nodiscard]] int64_t* HXConeGenPowRIndex() noexcept { return H_xcone_genpow_r_idx_.get(); }
    [[nodiscard]] int64_t* HXConeGenPowPIndex() noexcept { return H_xcone_genpow_p_idx_.get(); }
    [[nodiscard]] int64_t* HPsdIndex() noexcept { return H_psd_idx_.get(); }
    [[nodiscard]] const CSR& matrix() const noexcept { return KKT; }
    [[nodiscard]] CSR& matrix() noexcept { return KKT; }

    /**
     * @brief Populate KKT matrix with P and A values
     *
     * Copies values from sparse matrices P and A into the appropriate positions
     * in the KKT matrix structure. This must be called:
     * - After construction (sets initial P and A values)
     * - Before the first call to update() (which assumes P and A are populated)
     *
     * @pre The KKT matrix structure must be initialized (via constructor)
     * @pre P and A must have the same sparsity pattern as provided to constructor
     * @post KKT matrix contains P values in top-left block and A^T in top-right
     *
     * @param P Objective Hessian matrix (n x n, upper triangle, batched values)
     * @param A Constraint matrix (m x n, batched values)
     * @param stream CUDA stream for async execution
     */
    void populate(
        CSR& P,
        CSR& A,
        cudaStream_t stream = 0
    ) override {
        // Direct-x cones add structural-zero slots to KKT (off-diagonals
        // of the J×J submatrix that aren't in P). `populate_values_via_map`
        // only writes the P/A nnz slots; the direct-x extras keep whatever
        // was written there in the previous IPM iter — for example, the
        // final Hs values from a prior solve(). On the next solve,
        // Zero all KKT.values before populating — including the first call.
        // Structural-zero slots (off-diagonal Exp/Power Hs entries, expansion
        // col entries) must be zero so that `init_xcone_px_baseline` snapshots
        // zero for those entries. Without this, the first solve would snapshot
        // garbage from cudaMalloc, corrupting every subsequent refresh_xcone_hs.
        cudaMemsetAsync(KKT.values(), 0,
                        sizeof(double) * KKT.nnz() * batchSize, stream);
        // Only populate P if it has nonzeros (support for LP where P is empty)
        if (P.nnz() > 0) {
            populate_values_via_map(P, KKT, Pnnz_idx_.get(), stream);
        }
        populate_values_via_map(A, KKT, Annz_idx_.get(), stream);
        populated_ = true;
    }

    // Update H block with cone scaling values
    void update_H(
        const Cones& cones,
        const double* mu_data = nullptr,
        cudaStream_t stream = 0
    ) override {
        KKTHBlockArgs args;
        args.kkt_values = KKT.values();
        args.nnzKKT = KKT.nnz();
        args.batchSize = batchSize;

        args.H_diagIdx = H_diagIdx_.get();
        args.nonneg_w = cones.nonneg_w.data();
        args.numZeroCones = cones.numZeroCones;
        args.numNonnegCones = cones.numNonnegCones;

        args.H_exp_idx = H_exp_idx_.get();
        args.exp_Hs = cones.exp_Hs.data();
        args.numExpCones = cones.numExpCones;

        args.H_soc_idx = H_soc_idx_.get();
        args.soc_Hs = cones.soc_Hs.data();
        args.numSocCones = cones.numSocCones;
        args.totalSocHsEntries = cones.totalSocHsEntries;

        args.H_power_idx = H_power_idx_.get();
        args.power_Hs = cones.power_Hs.data();
        args.numPowerCones = cones.numPowerCones;

        // Sparse SOC expansion
        args.H_soc_u_idx = H_soc_u_idx_.get();
        args.H_soc_v_idx = H_soc_v_idx_.get();
        args.H_soc_exp_diag_idx = H_soc_exp_diag_idx_.get();
        args.soc_u = cones.soc_u.data();
        args.soc_v = cones.soc_v.data();
        args.soc_eta = cones.soc_eta.data();
        args.d_soc_dims = cones.d_soc_dims;
        args.d_soc_offsets = cones.d_soc_offsets;
        args.totalSocDim = cones.totalSocDim;
        args.numSparseSoc = cones.numSparseSoc;
        args.d_soc_sparse_offsets = cones.d_soc_sparse_offsets;
        args.d_soc_sparse_indices = cones.d_soc_sparse_indices;

        // GenPowerCone (dense + sparse)
        args.H_genpow_idx = H_genpow_idx_.get();
        args.H_genpow_q_idx = H_genpow_q_idx_.get();
        args.H_genpow_r_idx = H_genpow_r_idx_.get();
        args.H_genpow_p_idx = H_genpow_p_idx_.get();
        args.H_genpow_exp_diag_idx = H_genpow_exp_diag_idx_.get();
        args.genpow_Hs = cones.genpow_Hs.data();
        args.genpow_q = cones.genpow_q.data();
        args.genpow_r = cones.genpow_r.data();
        args.genpow_p = cones.genpow_p.data();
        args.genpow_mu = mu_data;
        args.numGenPowerCones = cones.numGenPowerCones;
        args.numSparseGenPow = cones.numSparseGenPow;
        args.totalGenPowerHsEntries = cones.totalGenPowerHsEntries;
        args.d_genPowerDim1s = cones.d_genPowerDim1s;
        args.d_genPowerDim2s = cones.d_genPowerDim2s;
        args.d_genPowerOffsets = cones.d_genPowerOffsets;
        args.d_genPowerAlphaOffsets = cones.d_genPowerAlphaOffsets;
        args.d_genPowerSparseOffsets = cones.d_genPowerSparseOffsets;
        args.d_genPowerSparseIndices = cones.d_genPowerSparseIndices;
        args.totalGenPowerDim = cones.totalGenPowerDim;
        args.totalGenPowerAlphas = cones.totalGenPowerAlphas;

        // Rank-9 PD-scaling arrays + state
        for (int k = 0; k < 6; ++k) {
            args.H_genpow_pd_axis_idx[k] = H_genpow_pd_axis_idx_[k].get();
        }
        args.genpow_pd_axes = cones.genpow_pd_axes.data();
        args.genpow_pd_coefs = cones.genpow_pd_coefs.data();
        args.genpow_pd_signs = cones.genpow_pd_signs.data();
        args.genpow_pd_active = cones.genpow_pd_active.data();

        update_kkt_H_block(args, stream);

        // Refresh slack GenPow PD-axis dsigns from the runtime pd_signs.
        // Mirrors the direct-x refresh below in refresh_xcone_hs; the
        // expansion-col comment near `dsigns_host` claims an update kernel
        // runs each iter, but only this site (and refresh_xcone_hs) actually
        // does the work.
        if (dsigns_ && cones.numSparseGenPow > 0) {
            int64_t slack_pd_axis_base_col = n + m + 2 * cones.numSparseSoc;
            ::moreau::refresh_genpow_pd_dsigns(
                dsigns_.get(),
                cones.genpow_pd_signs.data(),
                cones.d_genPowerSparseIndices,
                cones.numGenPowerCones,
                slack_pd_axis_base_col,
                stream);
        }

        // PSD cones: scatter psd_Hs into KKT at H_psd_idx_ positions
        if (cones.numPsdCones > 0 && H_psd_idx_) {
            update_kkt_psd_H_block(
                KKT.values(),
                H_psd_idx_.get(),
                cones.psd_Hs.data(),
                cones.d_psd_Hs_offsets,
                cones.totalPsdHsEntries,
                cones.numPsdCones,
                batchSize,
                KKT.nnz(),
                stream
            );
        }
    }

    /**
     * @brief Update cone scaling and KKT H block
     *
     * This function updates the cone scaling (computing w, λ, Hs for all cone types)
     * and then updates the KKT matrix H block with the new scaling values,
     * and refactorizes the KKT system.
     *
     * @param cones Cone data structure (will be modified with new scaling)
     * @param s Primal slack variables
     * @param z Dual variables
     * @param μ Barrier parameter
     * @param static_regularization_enable Whether to apply static regularization
     * @param static_regularization_constant Constant regularization term
     * @param static_regularization_proportional Proportional regularization term
     * @param stream CUDA stream
     * @return true if scaling update and factorization succeeded, false otherwise
     */
    bool update(
        Cones& cones,
        const BatchedVector& s,
        const BatchedVector& z,
        const BatchedVector& μ,
        ScalingStrategy scaling,
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        const BatchedVector& q,
        const BatchedVector& b,
        BatchedVector& workx,
        BatchedVector& const_rhs,
        BatchedVector& const_sol,
        BatchedVector& x2,
        BatchedVector& z2,
        cudaStream_t stream = 0
    ) override;

    // Legacy version without constant RHS solve (for backwards compatibility)
    bool update(
        Cones& cones,
        const BatchedVector& s,
        const BatchedVector& z,
        const BatchedVector& μ,
        ScalingStrategy scaling,
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        cudaStream_t stream = 0
    ) override {
        // Precondition: populate() must be called before update()
        assert(populated_ && "KKTData::update() called before populate()");

        // Update cone scaling (this also populates soc_Hs internally)
        bool success = cones.update_scaling(s, z, μ, scaling, stream);
        if (!success) return false;

        // Update KKT H block with new cone scaling values
        update_H(cones, μ.data(), stream);

        // Regularize and refactorize the KKT system
        success = regularize_and_refactor(
            static_regularization_enable,
            static_regularization_constant,
            static_regularization_proportional,
            stream
        );

        return success;
    }

    /**
     * @brief Update KKT system (factor + solve constant RHS)
     *
     * Factorizes the KKT system and solves the constant RHS [-q; b],
     * outputting the solution into x2, z2.
     */
    bool updateFactorOnly(
        Cones& cones,
        const BatchedVector& s,
        const BatchedVector& z,
        const BatchedVector& μ,
        ScalingStrategy scaling,
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        const BatchedVector& q,
        const BatchedVector& b,
        BatchedVector& workx,
        BatchedVector& const_rhs,
        BatchedVector& const_sol,
        BatchedVector& x2,
        BatchedVector& z2,
        cudaStream_t stream = 0
    ) override;

    // Factorize the KKT matrix (without regularization)
    void factor(cudaStream_t stream = 0) {
        if (!cudss_initialized_) {
            throw std::runtime_error("cuDSS not initialized");
        }

        // Ensure cuDSS uses the caller's stream (XLA FFI may pass a non-default stream)
        cudssSetStream(cudss_handle_, stream);

        // UBatch strategy: factorize using SYMMETRIC + UPPER (no full matrix expansion)
        cudssStatus_t status = cudssExecute(
            cudss_handle_, CUDSS_PHASE_FACTORIZATION,
            cudss_config_, cudss_data_,
            kkt_matrix_, sol_matrix_, rhs_matrix_
        );

        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cuDSS factorization failed with status: " + std::to_string(status));
        }
    }

    // Regularize and refactor the KKT matrix
    // static_regularization_enable: whether to apply regularization
    // static_regularization_constant: constant term ε_c
    // static_regularization_proportional: proportional term ε_p (multiplies max diagonal)
    bool regularize_and_refactor(
        bool static_regularization_enable,
        double static_regularization_constant,
        double static_regularization_proportional,
        cudaStream_t stream = 0
    ) override;


    // Solve the KKT system: sol = KKT^{-1} * rhs
    // rhs and sol are device pointers to batched vectors [batchSize * (n+m)]
    // NOTE: rhs and sol MUST be different buffers when IR (iterative refinement)
    // is enabled, because IR reads the original RHS during residual computation.
    void solve(const double* rhs, double* sol, cudaStream_t stream = 0) override {
        if (!cudss_initialized_) {
            throw std::runtime_error("cuDSS not initialized");
        }

        // Ensure cuDSS uses the caller's stream (XLA FFI may pass a non-default stream)
        cudssSetStream(cudss_handle_, stream);

        // UBatch strategy: solve all batches at once
        cudssMatrixSetValues(rhs_matrix_, const_cast<double*>(rhs));
        cudssMatrixSetValues(sol_matrix_, sol);

        cudssStatus_t status = cudssExecute(
            cudss_handle_, CUDSS_PHASE_SOLVE,
            cudss_config_, cudss_data_,
            kkt_matrix_, sol_matrix_, rhs_matrix_
        );

        if (status != CUDSS_STATUS_SUCCESS) {
            std::cerr << "cuDSS SOLVE failed with status: " << status << std::endl;
            std::cerr << "  batchSize: " << batchSize << std::endl;
            std::cerr << "  N: " << n + m + p << std::endl;
            throw std::runtime_error("cuDSS solve failed with status: " + std::to_string(status));
        }

#ifdef MOREAU_DEBUG
        // Compute linear system residual: ||KKT*sol - rhs||_inf for each batch
        // This is the single most important diagnostic for cuDSS accuracy
        if (debug_solve_residual_) {
            computeAndPrintSolveResidual(rhs, sol, stream);
        }
#endif
    }

#ifdef MOREAU_DEBUG
    // Enable/disable per-solve residual computation
    bool debug_solve_residual_ = false;
    int debug_solve_counter_ = 0;
    int debug_factor_counter_ = 0;

    void setDebugSolveResidual(bool enable) { debug_solve_residual_ = enable; debug_solve_counter_ = 0; }

    // Compute ||KKT_full * sol - rhs||_inf using host-side SpMV
    // KKT is stored as upper triangle; we compute full symmetric product
    void computeAndPrintSolveResidual(const double* d_rhs, const double* d_sol, cudaStream_t stream) {
        const int64_t N = n + m + p;
        const int64_t nnzKKT = KKT.nnz();

        cudaStreamSynchronize(stream);

        // Copy everything to host
        std::vector<double> h_rhs(N * batchSize), h_sol(N * batchSize), h_vals(nnzKKT * batchSize);
        std::vector<int64_t> h_ro(N + 1), h_ci(nnzKKT);

        cudaMemcpy(h_rhs.data(), d_rhs, sizeof(double) * N * batchSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_sol.data(), d_sol, sizeof(double) * N * batchSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_vals.data(), KKT.values(), sizeof(double) * nnzKKT * batchSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ro.data(), KKT.rowOffsets(), sizeof(int64_t) * (N + 1), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ci.data(), KKT.colIndices(), sizeof(int64_t) * nnzKKT, cudaMemcpyDeviceToHost);

        debug_solve_counter_++;

        for (int64_t b = 0; b < batchSize; ++b) {
            const double* vals = h_vals.data() + b * nnzKKT;
            const double* x = h_sol.data() + b * N;
            const double* r = h_rhs.data() + b * N;

            // Compute y = KKT_full * x (symmetric: upper stored)
            std::vector<double> y(N, 0.0);
            for (int64_t i = 0; i < N; ++i) {
                for (int64_t jj = h_ro[i]; jj < h_ro[i + 1]; ++jj) {
                    int64_t j = h_ci[jj];
                    double v = vals[jj];
                    y[i] += v * x[j];
                    if (j != i) {
                        y[j] += v * x[i];  // symmetric: lower triangle
                    }
                }
            }

            // Compute residual = y - rhs
            double max_res = 0.0, max_rhs = 0.0, max_sol = 0.0;
            int64_t max_res_idx = 0;
            for (int64_t i = 0; i < N; ++i) {
                double res = std::abs(y[i] - r[i]);
                if (res > max_res) { max_res = res; max_res_idx = i; }
                if (std::abs(r[i]) > max_rhs) max_rhs = std::abs(r[i]);
                if (std::abs(x[i]) > max_sol) max_sol = std::abs(x[i]);
            }
            double rel_res = max_rhs > 0 ? max_res / max_rhs : max_res;

            std::cout << std::scientific << std::setprecision(4);
            std::cout << "[KKT RESIDUAL #" << debug_solve_counter_ << "] batch=" << b
                      << " ||KKT*sol-rhs||_inf=" << max_res
                      << " rel=" << rel_res
                      << " at row " << max_res_idx;
            // Identify which block the worst row belongs to
            if (max_res_idx < n) std::cout << " (P block)";
            else if (max_res_idx < n + m) std::cout << " (H block, cone row " << (max_res_idx - n) << ")";
            else std::cout << " (expansion row " << (max_res_idx - n - m) << ")";
            std::cout << " ||rhs||=" << max_rhs << " ||sol||=" << max_sol << "\n";

            // Print per-block residuals
            double max_P_res = 0, max_H_res = 0, max_exp_res = 0;
            for (int64_t i = 0; i < n; ++i) max_P_res = std::max(max_P_res, std::abs(y[i] - r[i]));
            for (int64_t i = n; i < n + m; ++i) max_H_res = std::max(max_H_res, std::abs(y[i] - r[i]));
            for (int64_t i = n + m; i < N; ++i) max_exp_res = std::max(max_exp_res, std::abs(y[i] - r[i]));
            std::cout << "  P-block res=" << max_P_res
                      << "  H-block res=" << max_H_res
                      << "  expansion res=" << max_exp_res << "\n";

            // For catastrophic residuals (rel > 0.01), print RHS and solution
            if (rel_res > 0.01) {
                std::cout << "  *** CATASTROPHIC RESIDUAL ***\n";
                std::cout << "  RHS: [";
                for (int64_t i = 0; i < N; ++i) {
                    std::cout << std::setprecision(6) << r[i];
                    if (i < N-1) std::cout << ", ";
                }
                std::cout << "]\n";
                std::cout << "  SOL: [";
                for (int64_t i = 0; i < N; ++i) {
                    std::cout << std::setprecision(6) << x[i];
                    if (i < N-1) std::cout << ", ";
                }
                std::cout << "]\n";
                std::cout << "  KKT*SOL: [";
                for (int64_t i = 0; i < N; ++i) {
                    std::cout << std::setprecision(6) << y[i];
                    if (i < N-1) std::cout << ", ";
                }
                std::cout << "]\n";
                std::cout << "  RESIDUAL per row: [";
                for (int64_t i = 0; i < N; ++i) {
                    std::cout << std::setprecision(4) << (y[i] - r[i]);
                    if (i < N-1) std::cout << ", ";
                }
                std::cout << "]\n";
            }
        }
    }
#endif

    // Solve the KKT system with 2 RHS at once: [sol0, sol1] = KKT^{-1} * [rhs0, rhs1]
    // rhs and sol are device pointers to batched matrices [batchSize * (n+m+p) * 2]
    // Layout: [all_col0s, all_col1s] = [batch0_col0, batch1_col0, ..., batch0_col1, batch1_col1, ...]
    // This matches pack_rhs2() output: col0s for all batches first, then col1s
    void solve2(const double* rhs, double* sol, cudaStream_t stream = 0) override {
        if (!cudss_initialized_) {
            throw std::runtime_error("cuDSS not initialized");
        }

        const int64_t N = n + m + p;

        // Solve twice: once for col0, once for col1
        // This matches how Schur/Dense/SchurSparse solve2 works
        solve(rhs, sol, stream);
        solve(rhs + N * batchSize, sol + N * batchSize, stream);
    }


    // no copy
    KKTData(const KKTData&) = delete;
    KKTData& operator=(const KKTData&) = delete;
    // moves ok
    KKTData(KKTData&&) noexcept = default;
    KKTData& operator=(KKTData&&) noexcept = default;

    ~KKTData() {
        cleanup_cudss();
    }

    [[nodiscard]] KKTSolverType solverType() const noexcept override { return KKTSolverType::CuDSS; }

    /**
     * @brief Calculate total memory usage in bytes
     */
    [[nodiscard]] size_t memoryUsage() const noexcept override {
        return KKT.memoryUsage() +
               sizeof(int64_t) * n +    // P_diagIdx_
               sizeof(int64_t) * m +    // H_diagIdx_
               sizeof(int64_t) * KKT.nnz() +  // Pnnz_idx_ (approx)
               sizeof(int64_t) * KKT.nnz();   // Annz_idx_ (approx)
    }

private:
    // Work vector for per-batch diagonal norms
    mutable std::unique_ptr<BatchedVector> max_diag_per_batch_;

    void initialize_cudss(cudaStream_t stream, CuDSSStrategy strategy = CuDSSStrategy::Auto, int64_t maxLuNnz = -1) {
        if (cudss_initialized_) return;

        const int64_t N = n + m + p;

        // UBatch strategy: cuDSS uniform batching with SYMMETRIC + UPPER
        // Benchmarks show SYMMETRIC + UPPER is ~20% faster than GENERAL + FULL

        // Create cuDSS handle
        cudssStatus_t status = cudssCreate(&cudss_handle_);
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssCreate failed with status: " + std::to_string(status));
        }

        status = cudssSetStream(cudss_handle_, stream);
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssSetStream failed with status: " + std::to_string(status));
        }

        // Create cuDSS configuration
        status = cudssConfigCreate(&cudss_config_);
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssConfigCreate failed with status: " + std::to_string(status));
        }

        // Set batch size for uniform batching
        int batch_count = static_cast<int>(batchSize);
        status = cudssConfigSet(cudss_config_, CUDSS_CONFIG_UBATCH_SIZE, &batch_count, sizeof(batch_count));
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssConfigSet UBATCH_SIZE failed with status: " + std::to_string(status));
        }

        // Process all matrices in batch at once (not one at a time)
        int ubatch_index = -1;
        status = cudssConfigSet(cudss_config_, CUDSS_CONFIG_UBATCH_INDEX, &ubatch_index, sizeof(ubatch_index));
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssConfigSet UBATCH_INDEX failed with status: " + std::to_string(status));
        }

        // NOTE: cuDSS 0.8 solves are not bitwise deterministic (0.7 was);
        // CUDSS_CONFIG_DETERMINISTIC_MODE is NOT_SUPPORTED with uniform
        // batching, so repeated solves may differ by ~1 ulp on >= 0.8.

        // Pivoting: NONE for quasi-definite systems (default), on when enabled
        int pivot_type = cudss_pivot_enable_ ? MOREAU_CUDSS_PIVOT_ON : CUDSS_PIVOT_NONE;
        status = cudssConfigSet(cudss_config_, CUDSS_CONFIG_PIVOT_TYPE, &pivot_type, sizeof(pivot_type));
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssConfigSet PIVOT_TYPE failed with status: " + std::to_string(status));
        }

        // Iterative refinement for improved KKT solve accuracy
        ir_n_steps_ = cudss_ir_steps_;
        status = cudssConfigSet(cudss_config_, CUDSS_CONFIG_IR_N_STEPS, &ir_n_steps_, sizeof(ir_n_steps_));
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssConfigSet IR_N_STEPS failed with status: " + std::to_string(status));
        }

        // Dynamic regularization: any pivot smaller than `dynamic_reg_eps_`
        // is perturbed up to that value (sign-preserving). Matches CPU
        // qdldl's `regularize_eps` and stabilizes the KKT solve when cone
        // Hessians shrink toward 0 (asymmetric cones near boundary).
        if (dynamic_reg_eps_ > 0.0) {
            status = cudssConfigSet(cudss_config_, CUDSS_CONFIG_PIVOT_EPSILON,
                                    &dynamic_reg_eps_, sizeof(dynamic_reg_eps_));
            if (status != CUDSS_STATUS_SUCCESS) {
                throw std::runtime_error("cudssConfigSet PIVOT_EPSILON failed with status: " + std::to_string(status));
            }
        }

        // Max LU fill-in limit: cuDSS default is 100*nnz which can OOM for dense SDP.
        // User can set a lower value via IPMSettings.maxLuNnz.
        if (maxLuNnz >= 0) {
            status = cudssConfigSet(cudss_config_, CUDSS_CONFIG_MAX_LU_NNZ, &maxLuNnz, sizeof(maxLuNnz));
            if (status != CUDSS_STATUS_SUCCESS) {
                throw std::runtime_error("cudssConfigSet MAX_LU_NNZ failed with status: " + std::to_string(status));
            }
        }

        // Create cuDSS data object
        status = cudssDataCreate(cudss_handle_, &cudss_data_);
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssDataCreate failed with status: " + std::to_string(status));
        }

        // Create cuDSS matrix with SYMMETRIC + UPPER (fastest for UBATCH mode)
        // Benchmarks: SYMMETRIC+UPPER = 74.97ms vs GENERAL+FULL = 93.29ms (~20% faster)
        status = cudssMatrixCreateCsr(
            &kkt_matrix_, N, N, KKT.nnz(),
            KKT.rowOffsets(), nullptr, KKT.colIndices(), KKT.values(),
            MOREAU_CUDSS_CSR_I64_F64,
            CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO
        );
        if (status != CUDSS_STATUS_SUCCESS) {
            std::cerr << "cudssMatrixCreateCsr failed with status: " << status << "\n";
            throw std::runtime_error("cudssMatrixCreateCsr failed with status: " + std::to_string(status));
        }

        // Create dense RHS matrix (N x 1 per batch)
        // Note: We'll set the actual data pointer later in solve()
        status = cudssMatrixCreateDn(
            &rhs_matrix_, N, 1, N,
            nullptr, MOREAU_CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR
        );
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssMatrixCreateDn (rhs) failed with status: " + std::to_string(status));
        }

        // Create dense solution matrix (N x 1 per batch)
        status = cudssMatrixCreateDn(
            &sol_matrix_, N, 1, N,
            nullptr, MOREAU_CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR
        );
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssMatrixCreateDn (sol) failed with status: " + std::to_string(status));
        }

        // Create 2-RHS matrices for batched solves (constant + step in one call)
        // Layout: [batch0_rhs0, batch0_rhs1, batch1_rhs0, batch1_rhs1, ...]
        // Each RHS is N elements, so total is N * 2 * batchSize
        status = cudssMatrixCreateDn(
            &rhs2_matrix_, N, 2, N,  // N rows, 2 columns (RHS), leading dim N
            nullptr, MOREAU_CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR
        );
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssMatrixCreateDn (rhs2) failed with status: " + std::to_string(status));
        }

        status = cudssMatrixCreateDn(
            &sol2_matrix_, N, 2, N,
            nullptr, MOREAU_CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR
        );
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssMatrixCreateDn (sol2) failed with status: " + std::to_string(status));
        }

        // Perform symbolic analysis
        status = cudssExecute(
            cudss_handle_, CUDSS_PHASE_ANALYSIS,
            cudss_config_, cudss_data_,
            kkt_matrix_, sol_matrix_, rhs_matrix_
        );
        if (status != CUDSS_STATUS_SUCCESS) {
            throw std::runtime_error("cudssExecute ANALYSIS failed with status: " + std::to_string(status));
        }

        // Warmup factorization and solve to pre-allocate cuDSS internal workspace
        // Only do this for batched solvers (batchSize > 1) where the overhead is amortized
        // For single problems, the warmup cost isn't worth it
        if (batchSize > 1) {
            // Set KKT diagonal to 1.0 for identity-like matrix (will be overwritten by populate())
            set_kkt_diagonal_to_ones(
                KKT.values(), diag_full_.get(), batchSize, KKT.nnz(), N);
            cudaDeviceSynchronize();

            status = cudssExecute(
                cudss_handle_, CUDSS_PHASE_FACTORIZATION,
                cudss_config_, cudss_data_,
                kkt_matrix_, sol_matrix_, rhs_matrix_
            );
            if (status != CUDSS_STATUS_SUCCESS) {
                std::cerr << "cuDSS FACTORIZATION (warmup) failed with status: " << status << std::endl;
                std::cerr << "  batchSize: " << batchSize << std::endl;
                std::cerr << "  N: " << N << std::endl;
                std::cerr << "  nnz: " << KKT.nnz() << std::endl;
                throw std::runtime_error("cudssExecute FACTORIZATION (warmup) failed with status: " + std::to_string(status));
            }

            // Warmup solve to pre-allocate cuDSS solve workspace
            double* warmup_rhs = nullptr;
            double* warmup_sol = nullptr;
            cudaMalloc(&warmup_rhs, sizeof(double) * N * batchSize);
            cudaMalloc(&warmup_sol, sizeof(double) * N * batchSize);
            cudaMemset(warmup_rhs, 0, sizeof(double) * N * batchSize);

            cudssMatrixSetValues(rhs_matrix_, warmup_rhs);
            cudssMatrixSetValues(sol_matrix_, warmup_sol);

            status = cudssExecute(
                cudss_handle_, CUDSS_PHASE_SOLVE,
                cudss_config_, cudss_data_,
                kkt_matrix_, sol_matrix_, rhs_matrix_
            );
            if (status != CUDSS_STATUS_SUCCESS) {
                std::cerr << "cuDSS SOLVE (warmup) failed with status: " << status << std::endl;
                throw std::runtime_error("cudssExecute SOLVE (warmup) failed with status: " + std::to_string(status));
            }

            cudaFree(warmup_rhs);
            cudaFree(warmup_sol);

            // Reset rhs/sol matrices to nullptr so they don't dangle after
            // freeing warmup buffers. The first factorization passes these
            // handles to cudssExecute; dangling pointers cause corruption.
            cudssMatrixSetValues(rhs_matrix_, nullptr);
            cudssMatrixSetValues(sol_matrix_, nullptr);

            // Clear values (will be populated later by populate())
            cudaMemset(KKT.values(), 0, sizeof(double) * KKT.nnz() * batchSize);
        }

        cudss_initialized_ = true;
    }

    void cleanup_cudss() {
        if (!cudss_initialized_) return;

        // Ensure GPU operations complete before destroying handles
        cudaDeviceSynchronize();

        if (sol2_matrix_) cudssMatrixDestroy(sol2_matrix_);
        if (rhs2_matrix_) cudssMatrixDestroy(rhs2_matrix_);
        if (sol_matrix_) cudssMatrixDestroy(sol_matrix_);
        if (rhs_matrix_) cudssMatrixDestroy(rhs_matrix_);
        if (kkt_matrix_) cudssMatrixDestroy(kkt_matrix_);
        if (cudss_data_) cudssDataDestroy(cudss_handle_, cudss_data_);
        if (cudss_config_) cudssConfigDestroy(cudss_config_);
        if (cudss_handle_) cudssDestroy(cudss_handle_);

        sol_matrix_ = nullptr;
        rhs_matrix_ = nullptr;
        kkt_matrix_ = nullptr;
        cudss_data_ = nullptr;
        cudss_config_ = nullptr;
        cudss_handle_ = nullptr;
        cudss_initialized_ = false;
    }
};

// Implementation of regularize_and_refactor (out-of-line to avoid header bloat)
inline bool KKTData::regularize_and_refactor(
    bool static_regularization_enable,
    double static_regularization_constant,
    double static_regularization_proportional,
    cudaStream_t stream
) {
    const int64_t N = n + m + p;

    if (static_regularization_enable) {
        // Fused: backup diagonal + inf-norm + compute eps + regularize (4→1 kernel)
        fused_backup_infnorm_regularize(
            KKT.values(), work_diag_.get(), diag_full_.get(), dsigns_.get(),
            max_diag_per_batch_->data(),
            static_regularization_constant,
            static_regularization_proportional,
            batchSize, KKT.nnz(), N, stream
        );
    } else {
        // When regularization is off, zero max_diag so eps = machine epsilon
        cudaMemsetAsync(max_diag_per_batch_->data(), 0, sizeof(double) * batchSize, stream);
    }

    // 4. Refactor with cuDSS (UBatch: factorize all batches at once)
    // Ensure cuDSS uses the caller's stream (XLA FFI may pass a non-default stream)
    cudssSetStream(cudss_handle_, stream);

    cudssStatus_t status = cudssExecute(
        cudss_handle_, CUDSS_PHASE_FACTORIZATION,
        cudss_config_, cudss_data_,
        kkt_matrix_, sol_matrix_, rhs_matrix_
    );

    bool is_success = (status == CUDSS_STATUS_SUCCESS);

#ifdef MOREAU_DEBUG
    debug_factor_counter_++;
    if (debug_solve_residual_ && is_success) {
        // Print KKT diagonal range and regularizer after factorization
        cudaStreamSynchronize(stream);
        std::vector<double> h_vals(KKT.nnz());
        std::vector<int64_t> h_diag(N);
        cudaMemcpy(h_vals.data(), KKT.values(), sizeof(double) * KKT.nnz(), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_diag.data(), diag_full_.get(), sizeof(int64_t) * N, cudaMemcpyDeviceToHost);

        double min_diag = 1e300, max_diag = -1e300;
        double min_P_diag = 1e300, max_P_diag = -1e300;
        double min_H_diag = 1e300, max_H_diag = -1e300;
        double min_exp_diag = 1e300, max_exp_diag = -1e300;
        for (int64_t i = 0; i < N; ++i) {
            double d = h_vals[h_diag[i]];
            if (d < min_diag) min_diag = d;
            if (d > max_diag) max_diag = d;
            if (i < n) { if (d < min_P_diag) min_P_diag = d; if (d > max_P_diag) max_P_diag = d; }
            else if (i < n + m) { if (d < min_H_diag) min_H_diag = d; if (d > max_H_diag) max_H_diag = d; }
            else { if (d < min_exp_diag) min_exp_diag = d; if (d > max_exp_diag) max_exp_diag = d; }
        }
        std::cerr << std::scientific << std::setprecision(10);
        std::cerr << "\n=== GPU KKT iter " << debug_factor_counter_ << " ===\n";
        std::cerr << "[KKT DIAG] (with regularization) range=[" << min_diag << ", " << max_diag << "]\n";
        std::cerr << "  P diag: [" << min_P_diag << ", " << max_P_diag << "]\n";
        std::cerr << "  H diag: [" << min_H_diag << ", " << max_H_diag << "]\n";
        if (p > 0) std::cerr << "  Exp diag: [" << min_exp_diag << ", " << max_exp_diag << "]\n";

        // Full KKT dense dump (to stderr for comparison with CPU)
        {
            std::cerr << "GPU KKT dense (" << N << "x" << N << "), n=" << n
                      << ", m=" << m << ", p=" << p << ":\n";
            // Copy CSR to host
            std::vector<int64_t> h_ro(N + 1), h_ci(KKT.nnz());
            KKT.indicesGpuToCpu(h_ro.data(), h_ci.data());
            // Reconstruct full symmetric dense from upper-triangle CSR
            std::vector<double> dense(N * N, 0.0);
            for (int64_t i = 0; i < N; i++) {
                for (int64_t idx = h_ro[i]; idx < h_ro[i + 1]; idx++) {
                    int64_t j = h_ci[idx];
                    dense[i * N + j] = h_vals[idx];
                    dense[j * N + i] = h_vals[idx];
                }
            }
            for (int64_t i = 0; i < N; i++) {
                std::cerr << "  [";
                for (int64_t j = 0; j < N; j++) {
                    if (j > 0) std::cerr << ", ";
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%+.16e", dense[i * N + j]);
                    std::cerr << buf;
                }
                std::cerr << "]\n";
            }
            std::cerr << "GPU KKT diagonal:\n";
            for (int64_t i = 0; i < N; i++) {
                const char* block = (i < n) ? "P" : (i < n + m) ? "H" : "E";
                char buf[32];
                snprintf(buf, sizeof(buf), "%+.16e", dense[i * N + i]);
                std::cerr << "  [" << i << "] (" << block << ") " << buf << "\n";
            }
        }
    }
#endif

    // Restore diagonal after factorization. cuDSS has captured the regularized
    // values into its internal L/D factors. Note: with IR_N_STEPS > 0, cuDSS
    // references the original matrix for residual computation during solve,
    // so IR refines against the un-regularized system (using regularized factors
    // as a preconditioner).
    if (static_regularization_enable) {
        restore_diagonal(
            KKT.values(), work_diag_.get(), diag_full_.get(),
            batchSize, KKT.nnz(), N, stream
        );
    }

    return is_success;
}

} // namespace moreau
