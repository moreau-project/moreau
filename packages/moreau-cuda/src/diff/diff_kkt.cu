/**
 * @file diff_kkt.cu
 * @brief Implementation of DiffKKT for HSDE differentiation
 */

#include "moreau/diff/diff_kkt.hpp"
#include "moreau/diff/diff_riccati.hpp"
#include "moreau/kkt/riccati.hpp"
#include "moreau/profiling/profiler.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>

namespace moreau {

static constexpr double REG_DIFF = 1e-8;  // Regularization for diff KKT

/// Convert cuDSS status code to a human-readable string
static const char* cudss_status_string(cudssStatus_t status) {
    switch (status) {
        case CUDSS_STATUS_SUCCESS:          return "SUCCESS";
        case CUDSS_STATUS_NOT_INITIALIZED:  return "NOT_INITIALIZED";
        case CUDSS_STATUS_ALLOC_FAILED:     return "ALLOC_FAILED (out of GPU memory)";
        case CUDSS_STATUS_INVALID_VALUE:    return "INVALID_VALUE";
        case CUDSS_STATUS_NOT_SUPPORTED:    return "NOT_SUPPORTED";
        case CUDSS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUDSS_STATUS_INTERNAL_ERROR:   return "INTERNAL_ERROR";
        default:                            return "UNKNOWN";
    }
}

/// Report available GPU memory (for diagnostics)
static std::string gpu_memory_info() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (err != cudaSuccess) {
        return "  GPU memory: unavailable";
    }
    double free_mb = free_bytes / (1024.0 * 1024.0);
    double total_mb = total_bytes / (1024.0 * 1024.0);
    double used_mb = total_mb - free_mb;
    return "  GPU memory: " + std::to_string(static_cast<int>(free_mb)) + " MB free / "
         + std::to_string(static_cast<int>(total_mb)) + " MB total ("
         + std::to_string(static_cast<int>(used_mb)) + " MB used)";
}

// Compute total direct-x numel (sum of |J_xc|) from cones.x_cones.
// Defined as a free function so it can also be used by the size constants
// below in the member-initialiser list.
static int64_t compute_xn(const moreau::Cones& cones) {
    int64_t xn = 0;
    for (const auto& xc : cones.x_cones) {
        xn += static_cast<int64_t>(xc.indices.size());
    }
    return xn;
}

DiffKKT::DiffKKT(
    int64_t n_, int64_t m_, int64_t batch_,
    const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
    const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
    const Cones& cones,
    KKTSolverType type,
    cudaStream_t stream
) : n(n_), m(m_), batchSize(batch_),
    jdim(n_ + 2*m_ + compute_xn(cones) + 1 + 2 * cones.numSparseSoc + 3 * cones.numGenPowerCones + 3 * cones.numXGenPowerCones + 2 * cones.numSparseXSoc),
    augdim(2 * (n_ + 2*m_ + compute_xn(cones) + 1 + 2 * cones.numSparseSoc + 3 * cones.numGenPowerCones + 3 * cones.numXGenPowerCones + 2 * cones.numSparseXSoc)),
    solverType_(type),
    work_rhs_(2 * (n_ + 2*m_ + compute_xn(cones) + 1 + 2 * cones.numSparseSoc + 3 * cones.numGenPowerCones + 3 * cones.numXGenPowerCones + 2 * cones.numSparseXSoc), batch_),
    work_sol_(2 * (n_ + 2*m_ + compute_xn(cones) + 1 + 2 * cones.numSparseSoc + 3 * cones.numGenPowerCones + 3 * cones.numXGenPowerCones + 2 * cones.numSparseXSoc), batch_),
    numSparseSoc_(cones.numSparseSoc),
    base_jdim_(n_ + 2*m_ + compute_xn(cones) + 1),
    riccati_rhs_(n_ + m_, batch_),
    riccati_bvec_(n_ + m_, batch_),
    riccati_sol0_(n_ + m_, batch_),
    riccati_sol1_(n_ + m_, batch_),
    riccati_scalars_(2, batch_)
{
    // Direct-x cone metadata cached on `this` for use throughout the
    // sparsity build and `updateJ`.
    numXCones_ = static_cast<int64_t>(cones.x_cones.size());
    totalXConeNumel_ = compute_xn(cones);
    int64_t xn_ctor = totalXConeNumel_;

    // Host scratch built during sparsity construction. Each entry stores
    // a KKT nnz position; meta carries `(xc_idx, local_row, local_col)`
    // so the upload pass can group them by cone for kernel dispatch.
    struct XConeMeta { int64_t xc_idx; int64_t k; int64_t l; };
    std::vector<int64_t>   xcone_E_idx_host_local;
    std::vector<XConeMeta> xcone_E_meta_host;
    std::vector<int64_t>   xcone_du_idx_host_local;
    std::vector<XConeMeta> xcone_du_meta_host;
    // Stationarity-row mirror: row J_xc[k] gains a -H_x[k,l] entry per
    // du_x col l in cone xc. Captured per-x-row when iterating x columns.
    std::vector<int64_t>   xcone_stat_idx_host;
    std::vector<XConeMeta> xcone_stat_meta_host;
    if (type == KKTSolverType::Riccati) {
        initialize_riccati(n_, m_, batch_, P_ro, P_ci, nnzP,
                          A_ro, A_ci, nnzA, cones, stream);
        return;
    }

    // Build the sparsity pattern for the augmented KKT matrix
    // K = [I   J ]
    //     [J' -εI]
    //
    // We store upper triangle in CSR format
    // Row i (0 <= i < jdim): columns i..jdim-1 from I, then columns jdim..augdim-1 from J row i
    // Row i (jdim <= i < augdim): column i from -εI (diagonal only, since J' is in upper part)

    std::vector<int64_t> rowOff(augdim + 1, 0);
    std::vector<int64_t> colIdx;

    // Track index positions for updating values later
    std::vector<int64_t> I_diag_host(jdim, -1);
    std::vector<int64_t> negI_diag_host(jdim, -1);
    std::vector<int64_t> P_idx_host;
    std::vector<int64_t> P_val_idx_host;   // Which P value index to use for each P position
    std::vector<int64_t> A_idx_host;
    std::vector<int64_t> A_val_idx_host;   // Which A value index to use for each A position
    std::vector<int64_t> At_idx_host;
    std::vector<int64_t> At_val_idx_host;  // Which A value index to use for each A' position
    std::vector<int64_t> H_diag_idx_host;
    std::vector<int64_t> H_soc_idx_host;
    std::vector<int64_t> H_exp_idx_host;
    std::vector<int64_t> H_power_idx_host;
    std::vector<int64_t> H_psd_idx_host;
    // Sparse GenPowerCone expansion host index arrays
    std::vector<int64_t> H_xcone_soc_du_stat_diag_idx_host;
    std::vector<int64_t> H_xcone_soc_du_dx_diag_idx_host;
    std::vector<int64_t> H_xcone_soc_v1_col_stat_idx_host;
    std::vector<int64_t> H_xcone_soc_v2_col_stat_idx_host;
    std::vector<int64_t> H_xcone_soc_v1_col_dx_idx_host;
    std::vector<int64_t> H_xcone_soc_v2_col_dx_idx_host;
    std::vector<int64_t> H_xcone_soc_exp_v1_du_idx_host;
    std::vector<int64_t> H_xcone_soc_exp_v2_du_idx_host;
    std::vector<int64_t> H_xcone_soc_exp_diag_idx_host;
    std::vector<int64_t> H_xcone_genpow_du_stat_diag_idx_host;
    std::vector<int64_t> H_xcone_genpow_du_dx_diag_idx_host;
    std::vector<int64_t> H_xcone_genpow_v1_col_stat_idx_host;
    std::vector<int64_t> H_xcone_genpow_v2_col_stat_idx_host;
    std::vector<int64_t> H_xcone_genpow_v3_col_stat_idx_host;
    std::vector<int64_t> H_xcone_genpow_v1_col_dx_idx_host;
    std::vector<int64_t> H_xcone_genpow_v2_col_dx_idx_host;
    std::vector<int64_t> H_xcone_genpow_v3_col_dx_idx_host;
    std::vector<int64_t> H_xcone_genpow_exp_v1_du_idx_host;
    std::vector<int64_t> H_xcone_genpow_exp_v2_du_idx_host;
    std::vector<int64_t> H_xcone_genpow_exp_v3_du_idx_host;
    std::vector<int64_t> H_xcone_genpow_exp_diag_idx_host;
    std::vector<int64_t> H_genpow_sparse_diag_idx_host;
    std::vector<int64_t> H_genpow_v1_col_idx_host;
    std::vector<int64_t> H_genpow_v2_col_idx_host;
    std::vector<int64_t> H_genpow_v3_col_idx_host;
    std::vector<int64_t> H_genpow_exp_v1_du_idx_host;
    std::vector<int64_t> H_genpow_exp_v2_du_idx_host;
    std::vector<int64_t> H_genpow_exp_v3_du_idx_host;
    std::vector<int64_t> H_genpow_exp_diag_idx_host;

    std::vector<int64_t> q_idx_host(n, -1);
    std::vector<int64_t> b_idx_host(m, -1);
    std::vector<int64_t> c1_idx_host(n, -1);
    std::vector<int64_t> c2_idx_host(m, -1);
    int64_t c3_idx_val = -1;

    // Identity blocks within J
    std::vector<int64_t> J_I_row1_idx_host(m, -1);     // J[n:n+m, n:n+m] = I
    std::vector<int64_t> J_negI_row1_idx_host(m, -1);  // J[n:n+m, n+m:n+2m] = -I
    std::vector<int64_t> J_I_row2_idx_host(m, -1);     // J[n+m:n+2m, n:n+m] = I

    // Sparse SOC expansion host index arrays
    std::vector<int64_t> H_soc_sparse_diag_idx_host;
    std::vector<int64_t> H_soc_v1_col_idx_host;
    std::vector<int64_t> H_soc_v2_col_idx_host;
    std::vector<int64_t> H_soc_exp_v1_du_idx_host;
    std::vector<int64_t> H_soc_exp_v2_du_idx_host;
    std::vector<int64_t> H_soc_exp_diag_idx_host;

    // Resolve SOC dims: use original (unsorted) dims for KKT construction
    // because KKT rows must match the A matrix constraint order.
    // Derivative values arrive in sorted order and are remapped via index reordering below.
    const auto& socDimsForKKT = cones.socConeDimsOriginal.empty()
        ? cones.socConeDims : cones.socConeDimsOriginal;

    // Compute expansion column mapping
    // For each sparse SOC cone (in original order), track expansion column info
    struct SocExpansionInfo {
        int64_t cone_idx;       // index in socDimsForKKT (original order)
        int64_t cone_offset;    // offset of this cone in the u vector (within SOC cones)
        int64_t soc_dim;        // dimension of this cone
        int64_t exp_col_base;   // expansion column base in J (base_jdim + 2*sparse_idx)
    };
    std::vector<SocExpansionInfo> soc_expansion_info;
    {
        int64_t cone_offset = 0;
        int64_t sparse_idx = 0;
        for (int64_t k = 0; k < cones.numSocCones; ++k) {
            int64_t d = socDimsForKKT[k];
            if (d > 4) {
                soc_expansion_info.push_back({
                    k, cone_offset, d,
                    base_jdim_ + 2 * sparse_idx
                });
                ++sparse_idx;
            }
            cone_offset += d;
        }
    }

    // Compute GenPowerCone expansion column mapping
    struct GenpowExpansionInfo {
        int64_t cone_idx;       // index in genPowerCones
        int64_t cone_offset;    // offset of this cone in the u vector (within genpow cones)
        int64_t genpow_dim;     // total dimension of this cone
        int64_t exp_col_base;   // expansion column base in J
    };
    std::vector<GenpowExpansionInfo> genpow_expansion_info;
    {
        int64_t cone_offset = 0;
        int64_t soc_exp_count = 2 * cones.numSparseSoc;  // SOC expansion vars come first
        for (int64_t k = 0; k < cones.numGenPowerCones; ++k) {
            int64_t d = cones.genPowerDim1s[k] + cones.genPowerDim2s[k];
            genpow_expansion_info.push_back({
                k, cone_offset, d,
                base_jdim_ + soc_exp_count + 3 * k
            });
            cone_offset += d;
        }
    }

    // Direct-x GenPow expansion info. Direct-x expansion vars come AFTER
    // all slack expansion vars. `xc_du_offset` is the cone's start offset
    // within the [n+2m, n+2m+xn) du_x block. `cone_x_idx` is the cone's
    // position in `cones.x_cones` (used to look up indices for stat-row
    // emission).
    struct XGenpowExpansionInfo {
        int64_t cone_x_idx;     // index in cones.x_cones
        int64_t xc_du_offset;   // offset within direct-x du_x block
        int64_t genpow_dim;     // total dimension of this cone
        int64_t exp_col_base;   // expansion column base in J
    };
    std::vector<XGenpowExpansionInfo> xgenpow_expansion_info;
    numXGenPowerCones_ = cones.numXGenPowerCones;
    int64_t total_xgenpow_dim = 0;
    {
        int64_t xc_du_offset = 0;
        int64_t slack_exp_count = 2 * cones.numSparseSoc + 3 * cones.numGenPowerCones;
        int64_t xgp_idx = 0;
        for (int64_t c = 0; c < numXCones_; ++c) {
            const auto& xc = cones.x_cones[c];
            int64_t d = static_cast<int64_t>(xc.indices.size());
            if (xc.kind == XConeKind::GenPower) {
                xgenpow_expansion_info.push_back({
                    c, xc_du_offset, d,
                    base_jdim_ + slack_exp_count + 3 * xgp_idx
                });
                total_xgenpow_dim += d;
                ++xgp_idx;
            }
            xc_du_offset += d;
        }
    }
    totalXGenPowDim_ = total_xgenpow_dim;

    // Direct-x SOC sparse expansion info (only cones with dim > 4 use
    // the rank-2 path; smaller stay on dense `xcone_soc_H`). Direct-x
    // SOC expansion vars come AFTER all GenPow expansion vars.
    struct XSocSparseInfo {
        int64_t cone_x_idx;
        int64_t xc_du_offset;
        int64_t soc_dim;
        int64_t exp_col_base;
        int64_t sparse_dim_off;  // offset into the per-row sparse stripe arrays
    };
    std::vector<XSocSparseInfo> xsoc_sparse_info;
    int64_t total_xsoc_sparse_dim = 0;
    {
        int64_t xc_du_offset = 0;
        int64_t pre_xsoc_exp_count =
            2 * cones.numSparseSoc + 3 * cones.numGenPowerCones
            + 3 * cones.numXGenPowerCones;
        int64_t xsoc_idx = 0;
        for (int64_t c = 0; c < numXCones_; ++c) {
            const auto& xc = cones.x_cones[c];
            int64_t d = static_cast<int64_t>(xc.indices.size());
            if (xc.kind == XConeKind::SOC && d > 4) {
                xsoc_sparse_info.push_back({
                    c, xc_du_offset, d,
                    base_jdim_ + pre_xsoc_exp_count + 2 * xsoc_idx,
                    total_xsoc_sparse_dim
                });
                total_xsoc_sparse_dim += d;
                ++xsoc_idx;
            }
            xc_du_offset += d;
        }
    }
    numSparseXSoc_ = static_cast<int64_t>(xsoc_sparse_info.size());
    totalSparseXSocDim_ = total_xsoc_sparse_dim;

    // Sparse SOC indexing info (in original order for KKT row structure)
    std::vector<int64_t> soc_sparse_offsets_host(cones.numSocCones + 1, 0);
    std::vector<int64_t> soc_sparse_indices_host(cones.numSocCones, -1);
    totalSparseSocDim_ = 0;
    {
        int64_t sparse_dim_acc = 0;
        int64_t sparse_idx = 0;
        for (int64_t i = 0; i < cones.numSocCones; ++i) {
            int64_t d = socDimsForKKT[i];
            if (d > 4) {
                soc_sparse_indices_host[i] = sparse_idx++;
                sparse_dim_acc += d;
            }
            soc_sparse_offsets_host[i + 1] = sparse_dim_acc;
        }
        totalSparseSocDim_ = sparse_dim_acc;
    }

    int64_t nnz = 0;

    // Build A column entries for A' placement
    std::vector<std::vector<std::pair<int64_t, int64_t>>> A_col_entries(n);
    for (int64_t r = 0; r < m; ++r) {
        for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p) {
            int64_t j = A_ci[p];
            if (j >= 0 && j < n) A_col_entries[j].push_back({r, p});
        }
    }

    // Build P column entries for symmetric P expansion
    // P is stored as upper triangle in CSR. P_col_entries[j] contains pairs (row, p)
    // where p is the index in P's value array for P[row, j] with row <= j
    std::vector<std::vector<std::pair<int64_t, int64_t>>> P_col_entries(n);
    for (int64_t r = 0; r < n; ++r) {
        for (int64_t p = P_ro[r]; p < P_ro[r + 1]; ++p) {
            int64_t j = P_ci[p];
            if (j >= 0 && j < n) P_col_entries[j].push_back({r, p});
        }
    }

    // First jdim rows: [I | J]
    for (int64_t row = 0; row < jdim; ++row) {
        // Upper triangle of I: just diagonal
        I_diag_host[row] = nnz;
        colIdx.push_back(row);
        ++nnz;

        // J[row, :] goes into columns jdim..augdim-1
        // J structure depends on which block of rows we're in

        if (row < n) {
            // J row 0..n-1: [P_row | A'_row | 0 | q]

            // P block: we need the FULL row of P (not just upper triangle)
            // P[row, j] for j >= row: stored in P's row `row`
            // P[row, j] for j < row: equals P[j, row], stored in P's row `j`

            // Collect all P entries for this row
            std::vector<std::pair<int64_t, int64_t>> p_row_entries;  // (col, val_idx)

            // From P's row `row` (upper triangle entries where j >= row)
            for (int64_t p = P_ro[row]; p < P_ro[row + 1]; ++p) {
                int64_t j = P_ci[p];
                if (j >= row) {
                    p_row_entries.push_back({j, p});
                }
            }

            // From P's columns < row: P[row, j] = P[j, row] stored in row j of P
            // These are in P_col_entries[row] for columns j < row
            for (const auto& [r, p] : P_col_entries[row]) {
                if (r < row) {  // Only add lower triangle (j < row, here r is the row in P which equals j)
                    p_row_entries.push_back({r, p});  // P[row, r] = P[r, row]
                }
            }

            // Sort by column index (for CSR format)
            std::sort(p_row_entries.begin(), p_row_entries.end());

            // Add to pattern
            for (const auto& [j, p] : p_row_entries) {
                int64_t kkt_col = jdim + j;
                P_idx_host.push_back(nnz);
                P_val_idx_host.push_back(p);  // Track which P value to use
                colIdx.push_back(kkt_col);
                ++nnz;
            }

            // A' block: A'[row, w] = A[w, row] for w in 0..m-1
            // Columns n..n+m-1 of J, i.e., jdim+n..jdim+n+m-1 of K
            for (const auto& [w, p] : A_col_entries[row]) {
                int64_t kkt_col = jdim + n + w;
                At_idx_host.push_back(nnz);
                At_val_idx_host.push_back(p);  // Track which A value to use
                colIdx.push_back(kkt_col);
                ++nnz;
            }

            // Direct-x stationarity-row entries: if `row` is one of the
            // gathered indices J_xc[k] for some direct-x cone xc, then
            // F_1[row] picks up `−H_x[k, l]·du_x_l` for every l in cone xc.
            // The mirror entries live in J's stationarity row × du_x col
            // (col n+2m + xc_off + l), which in K is at col jdim + n + 2m
            // + xc_off + l. Since du_x cols come before τ, these entries
            // are sorted before the τ col entry below.
            for (int64_t xc_idx = 0; xc_idx < numXCones_; ++xc_idx) {
                const auto& xc = cones.x_cones[xc_idx];
                int64_t dim_xc = static_cast<int64_t>(xc.indices.size());
                int64_t xc_off = 0;
                for (int64_t i = 0; i < xc_idx; ++i) {
                    xc_off += static_cast<int64_t>(cones.x_cones[i].indices.size());
                }
                for (int64_t k = 0; k < dim_xc; ++k) {
                    if (xc.indices[k] == row) {
                        if (xc.kind == XConeKind::GenPower) {
                            int64_t xgp_exp_col_base = -1;
                            for (const auto& info : xgenpow_expansion_info) {
                                if (info.cone_x_idx == xc_idx) {
                                    xgp_exp_col_base = info.exp_col_base;
                                    break;
                                }
                            }
                            H_xcone_genpow_du_stat_diag_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + n + 2 * m + xc_off + k);
                            ++nnz;
                            H_xcone_genpow_v1_col_stat_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + xgp_exp_col_base);
                            ++nnz;
                            H_xcone_genpow_v2_col_stat_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + xgp_exp_col_base + 1);
                            ++nnz;
                            H_xcone_genpow_v3_col_stat_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + xgp_exp_col_base + 2);
                            ++nnz;
                        } else if (xc.kind == XConeKind::SOC && dim_xc > 4) {
                            // Direct-x SOC rank-2 sparse path.
                            int64_t xsoc_exp_col_base = -1;
                            for (const auto& info : xsoc_sparse_info) {
                                if (info.cone_x_idx == xc_idx) {
                                    xsoc_exp_col_base = info.exp_col_base;
                                    break;
                                }
                            }
                            H_xcone_soc_du_stat_diag_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + n + 2 * m + xc_off + k);
                            ++nnz;
                            H_xcone_soc_v1_col_stat_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + xsoc_exp_col_base);
                            ++nnz;
                            H_xcone_soc_v2_col_stat_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + xsoc_exp_col_base + 1);
                            ++nnz;
                        } else {
                            // Dense path (Nonneg/SOC dim≤4/PSD/Exp/Power):
                            // emit full dim_xc mirror entries.
                            for (int64_t l = 0; l < dim_xc; ++l) {
                                xcone_stat_idx_host.push_back(nnz);
                                xcone_stat_meta_host.push_back({xc_idx, k, l});
                                colIdx.push_back(jdim + n + 2 * m + xc_off + l);
                                ++nnz;
                            }
                        }
                        break;  // each x index belongs to at most one cone
                    }
                }
            }

            // q column (tau column = column n+2m+xn of J = column jdim + n + 2*m + xn of K)
            q_idx_host[row] = nnz;
            colIdx.push_back(jdim + n + 2 * m + xn_ctor);
            ++nnz;

        } else if (row < n + m) {
            // J row n..n+m-1: [A_row | I | -I | -b]
            int64_t w_idx = row - n;  // 0..m-1

            // A block (columns 0..n-1 of J)
            for (int64_t p = A_ro[w_idx]; p < A_ro[w_idx + 1]; ++p) {
                int64_t j = A_ci[p];
                int64_t kkt_col = jdim + j;
                A_idx_host.push_back(nnz);
                A_val_idx_host.push_back(p);  // Track which A value to use
                colIdx.push_back(kkt_col);
                ++nnz;
            }

            // I diagonal at (w_idx, w_idx) in J block (1, 1)
            // Column n + w_idx of J = jdim + n + w_idx of K
            J_I_row1_idx_host[w_idx] = nnz;  // Track for initialization
            colIdx.push_back(jdim + n + w_idx);
            ++nnz;

            // -I diagonal at (w_idx, w_idx) in J block (1, 2)
            // Column n + m + w_idx of J = jdim + n + m + w_idx of K
            J_negI_row1_idx_host[w_idx] = nnz;  // Track for initialization
            colIdx.push_back(jdim + n + m + w_idx);
            ++nnz;

            // -b column (column n+2m of J = column jdim + n + 2*m of K)
            b_idx_host[w_idx] = nnz;
            colIdx.push_back(jdim + n + 2 * m + xn_ctor);
            ++nnz;

        } else if (row < n + 2*m) {
            // J row n+m..n+2m-1: [0 | I | -H | 0]
            int64_t u_idx = row - n - m;  // 0..m-1

            // I at (u_idx, u_idx) in J block (2, 1)
            // Column n + u_idx of J = jdim + n + u_idx of K
            J_I_row2_idx_host[u_idx] = nnz;  // Track for initialization
            colIdx.push_back(jdim + n + u_idx);
            ++nnz;

            // -H block (columns n+m..n+2m-1 of J)
            // The structure depends on which cone u_idx belongs to
            int64_t offset = 0;
            bool found = false;

            // Zero cones (H = 0, diagonal structure)
            if (u_idx < cones.numZeroCones) {
                // Zero cone: H = 0, add diagonal position for consistency
                H_diag_idx_host.push_back(nnz);
                colIdx.push_back(jdim + n + m + u_idx);
                ++nnz;
                found = true;
            }
            offset = cones.numZeroCones;

            // Nonnegative cones (diagonal)
            if (!found && u_idx < offset + cones.numNonnegCones) {
                H_diag_idx_host.push_back(nnz);
                colIdx.push_back(jdim + n + m + u_idx);
                ++nnz;
                found = true;
            }
            offset += cones.numNonnegCones;

            // SOC cones (variable dim)
            // Dense (dim<=4): full dim*dim entries in H block
            // Sparse (dim>4): diagonal + 2 expansion column entries per row
            if (!found) {
                for (int64_t k = 0; k < cones.numSocCones && !found; ++k) {
                    int64_t soc_dim = socDimsForKKT[k];
                    if (u_idx >= offset && u_idx < offset + soc_dim) {
                        if (soc_dim <= 4) {
                            // Dense: full dim*dim entries
                            for (int64_t c = 0; c < soc_dim; ++c) {
                                H_soc_idx_host.push_back(nnz);
                                colIdx.push_back(jdim + n + m + offset + c);
                                ++nnz;
                            }
                        } else {
                            // Sparse: diagonal entry + v1 expansion col + v2 expansion col
                            // Find this cone's expansion info
                            int64_t exp_col_base = -1;
                            for (const auto& info : soc_expansion_info) {
                                if (info.cone_idx == k) {
                                    exp_col_base = info.exp_col_base;
                                    break;
                                }
                            }

                            // Diagonal entry at column offset + local_row
                            H_soc_sparse_diag_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + n + m + offset + (u_idx - offset));
                            ++nnz;

                            // v1 expansion column entry
                            H_soc_v1_col_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + exp_col_base);
                            ++nnz;

                            // v2 expansion column entry
                            H_soc_v2_col_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + exp_col_base + 1);
                            ++nnz;
                        }
                        found = true;
                    }
                    offset += soc_dim;
                }
            }

            // PSD cones (variable dim, dense svec_dim x svec_dim)
            // PSD derivative is symmetric, stored as upper triangle in psd_H.
            // KKT needs full svec_dim * svec_dim entries per cone.
            if (!found) {
                const auto& psdDimsForKKT = cones.psdConeDimsOriginal.empty()
                    ? cones.psdConeDims : cones.psdConeDimsOriginal;
                for (int64_t k = 0; k < cones.numPsdCones && !found; ++k) {
                    int64_t mat_dim = psdDimsForKKT[k];
                    int64_t svec_dim = mat_dim * (mat_dim + 1) / 2;
                    if (u_idx >= offset && u_idx < offset + svec_dim) {
                        // Full row of svec_dim entries
                        for (int64_t c = 0; c < svec_dim; ++c) {
                            H_psd_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + n + m + offset + c);
                            ++nnz;
                        }
                        found = true;
                    }
                    offset += svec_dim;
                }
            }

            // Exp cones (3x3 dense)
            if (!found) {
                for (int64_t k = 0; k < cones.numExpCones && !found; ++k) {
                    if (u_idx >= offset && u_idx < offset + 3) {
                        // Add ALL entries for this row (full row)
                        for (int64_t c = 0; c < 3; ++c) {
                            H_exp_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + n + m + offset + c);
                            ++nnz;
                        }
                        found = true;
                    }
                    offset += 3;
                }
            }

            // Power cones (3x3 dense)
            if (!found) {
                for (int64_t k = 0; k < cones.numPowerCones && !found; ++k) {
                    if (u_idx >= offset && u_idx < offset + 3) {
                        // Add ALL entries for this row (full row)
                        for (int64_t c = 0; c < 3; ++c) {
                            H_power_idx_host.push_back(nnz);
                            colIdx.push_back(jdim + n + m + offset + c);
                            ++nnz;
                        }
                        found = true;
                    }
                    offset += 3;
                }
            }

            // GenPowerCones (variable dim, sparse: diagonal + 3 expansion columns)
            if (!found) {
                for (int64_t k = 0; k < cones.numGenPowerCones && !found; ++k) {
                    int64_t gp_dim = cones.genPowerDim1s[k] + cones.genPowerDim2s[k];
                    if (u_idx >= offset && u_idx < offset + gp_dim) {
                        // Find this cone's expansion info
                        int64_t exp_col_base = genpow_expansion_info[k].exp_col_base;

                        // Diagonal entry
                        H_genpow_sparse_diag_idx_host.push_back(nnz);
                        colIdx.push_back(jdim + n + m + offset + (u_idx - offset));
                        ++nnz;

                        // v1 expansion column entry
                        H_genpow_v1_col_idx_host.push_back(nnz);
                        colIdx.push_back(jdim + exp_col_base);
                        ++nnz;

                        // v2 expansion column entry
                        H_genpow_v2_col_idx_host.push_back(nnz);
                        colIdx.push_back(jdim + exp_col_base + 1);
                        ++nnz;

                        // v3 expansion column entry
                        H_genpow_v3_col_idx_host.push_back(nnz);
                        colIdx.push_back(jdim + exp_col_base + 2);
                        ++nnz;

                        found = true;
                    }
                    offset += gp_dim;
                }
            }

        } else if (row >= n + 2*m && row < n + 2*m + xn_ctor) {
            // Direct-x cone projection row.
            // F_xc_k = +x[J_xc[k]] − (I − H_x)·du_x = 0
            //   ⇒ J entry at x col J_xc[k]: +1
            //   ⇒ J entries at du_x cols of cone xc: −δ_{kl} + H_x[k,l]
            // (Stationarity-row mirror entries `−H_x[k,l]` are added in the
            //  stat-row branch above when iterating x cols.)
            int64_t xrow = row - (n + 2 * m);  // 0..xn_ctor-1
            // Locate the cone and within-cone index.
            int64_t xc_idx = -1;
            int64_t k_in_cone = 0;
            {
                int64_t off = 0;
                for (int64_t i = 0; i < numXCones_; ++i) {
                    int64_t d = static_cast<int64_t>(cones.x_cones[i].indices.size());
                    if (xrow >= off && xrow < off + d) {
                        xc_idx = i;
                        k_in_cone = xrow - off;
                        break;
                    }
                    off += d;
                }
            }
            const auto& xc = cones.x_cones[xc_idx];
            int64_t dim_xc = static_cast<int64_t>(xc.indices.size());
            // E_J entry: +1 at K col jdim + J_xc[k_in_cone].
            int64_t Jxk = xc.indices[k_in_cone];
            xcone_E_idx_host_local.push_back(nnz);
            xcone_E_meta_host.push_back({xc_idx, k_in_cone, Jxk});
            colIdx.push_back(jdim + Jxk);
            ++nnz;
            int64_t xc_du_base = n + 2 * m + (xrow - k_in_cone);
            if (xc.kind == XConeKind::GenPower) {
                int64_t xgp_exp_col_base = -1;
                for (const auto& info : xgenpow_expansion_info) {
                    if (info.cone_x_idx == xc_idx) {
                        xgp_exp_col_base = info.exp_col_base;
                        break;
                    }
                }
                H_xcone_genpow_du_dx_diag_idx_host.push_back(nnz);
                colIdx.push_back(jdim + xc_du_base + k_in_cone);
                ++nnz;
                H_xcone_genpow_v1_col_dx_idx_host.push_back(nnz);
                colIdx.push_back(jdim + xgp_exp_col_base);
                ++nnz;
                H_xcone_genpow_v2_col_dx_idx_host.push_back(nnz);
                colIdx.push_back(jdim + xgp_exp_col_base + 1);
                ++nnz;
                H_xcone_genpow_v3_col_dx_idx_host.push_back(nnz);
                colIdx.push_back(jdim + xgp_exp_col_base + 2);
                ++nnz;
            } else if (xc.kind == XConeKind::SOC && dim_xc > 4) {
                int64_t xsoc_exp_col_base = -1;
                for (const auto& info : xsoc_sparse_info) {
                    if (info.cone_x_idx == xc_idx) {
                        xsoc_exp_col_base = info.exp_col_base;
                        break;
                    }
                }
                H_xcone_soc_du_dx_diag_idx_host.push_back(nnz);
                colIdx.push_back(jdim + xc_du_base + k_in_cone);
                ++nnz;
                H_xcone_soc_v1_col_dx_idx_host.push_back(nnz);
                colIdx.push_back(jdim + xsoc_exp_col_base);
                ++nnz;
                H_xcone_soc_v2_col_dx_idx_host.push_back(nnz);
                colIdx.push_back(jdim + xsoc_exp_col_base + 1);
                ++nnz;
            } else {
                // Dense path: emit dim_xc du_x col entries.
                for (int64_t l = 0; l < dim_xc; ++l) {
                    xcone_du_idx_host_local.push_back(nnz);
                    xcone_du_meta_host.push_back({xc_idx, k_in_cone, l});
                    colIdx.push_back(jdim + xc_du_base + l);
                    ++nnz;
                }
            }
        } else if (row == n + 2*m + xn_ctor) {
            // J row n+2m+xn (tau row): [c1 | c2 | 0 | 0 | c3 | 0...0]

            // c1 (columns 0..n-1 of J)
            for (int64_t i = 0; i < n; ++i) {
                c1_idx_host[i] = nnz;
                colIdx.push_back(jdim + i);
                ++nnz;
            }

            // c2 (columns n..n+m-1 of J)
            for (int64_t i = 0; i < m; ++i) {
                c2_idx_host[i] = nnz;
                colIdx.push_back(jdim + n + i);
                ++nnz;
            }

            // c3 (column n+2m of J, which is column jdim + n + 2*m of K)
            // Note: this is NOT augdim-1 anymore due to expansion columns
            c3_idx_val = nnz;
            colIdx.push_back(jdim + n + 2 * m + xn_ctor);
            ++nnz;
        } else {
            // Expansion rows: row = n + 2*m + xn + 1 + exp_idx
            // First 2*numSparseSoc rows: SOC expansion (2 per sparse SOC cone)
            // Next 3*numGenPowerCones rows: GenPowerCone expansion (3 per cone)
            int64_t exp_row_idx = row - (n + 2*m + xn_ctor + 1);

            if (exp_row_idx < 2 * static_cast<int64_t>(numSparseSoc_)) {
                // SOC expansion row
                int64_t sparse_cone = exp_row_idx / 2;
                bool is_v1 = (exp_row_idx % 2 == 0);

                const auto& info = soc_expansion_info[sparse_cone];
                int64_t cone_offset_in_u = cones.numZeroCones + cones.numNonnegCones + info.cone_offset;

                // du-column entries: -v[i] at columns (n+m + cone_offset_in_u + i) of J
                for (int64_t i = 0; i < info.soc_dim; ++i) {
                    if (is_v1) {
                        H_soc_exp_v1_du_idx_host.push_back(nnz);
                    } else {
                        H_soc_exp_v2_du_idx_host.push_back(nnz);
                    }
                    colIdx.push_back(jdim + n + m + cone_offset_in_u + i);
                    ++nnz;
                }

                // Diagonal entry at expansion column (= this row's column in J)
                H_soc_exp_diag_idx_host.push_back(nnz);
                colIdx.push_back(jdim + row);
                ++nnz;
            } else if (exp_row_idx < 2 * static_cast<int64_t>(numSparseSoc_)
                       + 3 * cones.numGenPowerCones) {
                // GenPowerCone slack expansion row
                int64_t gp_exp_idx = exp_row_idx - 2 * static_cast<int64_t>(numSparseSoc_);
                int64_t gp_cone = gp_exp_idx / 3;
                int64_t local_exp = gp_exp_idx % 3;  // 0=v1(right1), 1=v2(right2), 2=v3(left3)

                const auto& info = genpow_expansion_info[gp_cone];
                // u offset for this cone within the H block:
                // zero + nonneg + soc + psd + exp + power + genpow_offset
                int64_t cone_offset_in_u = cones.numZeroCones + cones.numNonnegCones
                    + cones.totalSocDim + cones.totalPsdSvecDim
                    + cones.numExpCones * 3 + cones.numPowerCones * 3
                    + info.cone_offset;

                // du-column entries: -right_k[i] at columns (n+m + cone_offset_in_u + i) of J
                for (int64_t i = 0; i < info.genpow_dim; ++i) {
                    if (local_exp == 0) {
                        H_genpow_exp_v1_du_idx_host.push_back(nnz);
                    } else if (local_exp == 1) {
                        H_genpow_exp_v2_du_idx_host.push_back(nnz);
                    } else {
                        H_genpow_exp_v3_du_idx_host.push_back(nnz);
                    }
                    colIdx.push_back(jdim + n + m + cone_offset_in_u + i);
                    ++nnz;
                }

                // Diagonal entry at expansion column (= this row's column in J)
                H_genpow_exp_diag_idx_host.push_back(nnz);
                colIdx.push_back(jdim + row);
                ++nnz;
            } else if (exp_row_idx < 2 * static_cast<int64_t>(numSparseSoc_)
                       + 3 * cones.numGenPowerCones
                       + 3 * cones.numXGenPowerCones) {
                // Direct-x GenPow expansion row. Layout: 3 rows per
                // direct-x GenPow cone, after all slack expansion rows.
                int64_t xgp_exp_idx = exp_row_idx
                    - 2 * static_cast<int64_t>(numSparseSoc_)
                    - 3 * cones.numGenPowerCones;
                int64_t xgp_cone = xgp_exp_idx / 3;
                int64_t local_exp = xgp_exp_idx % 3;
                const auto& info = xgenpow_expansion_info[xgp_cone];

                for (int64_t i = 0; i < info.genpow_dim; ++i) {
                    if (local_exp == 0) {
                        H_xcone_genpow_exp_v1_du_idx_host.push_back(nnz);
                    } else if (local_exp == 1) {
                        H_xcone_genpow_exp_v2_du_idx_host.push_back(nnz);
                    } else {
                        H_xcone_genpow_exp_v3_du_idx_host.push_back(nnz);
                    }
                    colIdx.push_back(jdim + n + 2 * m + info.xc_du_offset + i);
                    ++nnz;
                }
                H_xcone_genpow_exp_diag_idx_host.push_back(nnz);
                colIdx.push_back(jdim + row);
                ++nnz;
            } else {
                // Direct-x SOC sparse expansion row (rank-2). 2 rows per
                // direct-x sparse SOC cone, after all GenPow expansion.
                int64_t xsoc_exp_idx = exp_row_idx
                    - 2 * static_cast<int64_t>(numSparseSoc_)
                    - 3 * cones.numGenPowerCones
                    - 3 * cones.numXGenPowerCones;
                int64_t xsoc_cone = xsoc_exp_idx / 2;
                int64_t local_exp = xsoc_exp_idx % 2;
                const auto& info = xsoc_sparse_info[xsoc_cone];

                for (int64_t i = 0; i < info.soc_dim; ++i) {
                    if (local_exp == 0) {
                        H_xcone_soc_exp_v1_du_idx_host.push_back(nnz);
                    } else {
                        H_xcone_soc_exp_v2_du_idx_host.push_back(nnz);
                    }
                    colIdx.push_back(jdim + n + 2 * m + info.xc_du_offset + i);
                    ++nnz;
                }
                H_xcone_soc_exp_diag_idx_host.push_back(nnz);
                colIdx.push_back(jdim + row);
                ++nnz;
            }
        }

        rowOff[row + 1] = nnz;
    }

    // Last jdim rows (jdim..augdim-1): [J' | -εI]
    // Since we store upper triangle and J' entries have row >= jdim, col < jdim,
    // the J' block is in the lower triangle, so we only store -εI diagonal
    for (int64_t row = jdim; row < augdim; ++row) {
        negI_diag_host[row - jdim] = nnz;
        colIdx.push_back(row);
        ++nnz;
        rowOff[row + 1] = nnz;
    }

    // Create CSR and upload
    int64_t nnzKKT = nnz;
    CSR kkt(augdim, augdim, nnzKKT, batchSize);
    kkt.indicesCpuToGpu(rowOff.data(), colIdx.data(), stream);

    // Upload index maps to device
    auto upload_indices = [&stream](const std::vector<int64_t>& host, device_unique_ptr<int64_t>& dev) {
        if (host.empty()) return;
        int64_t* ptr = nullptr;
        auto e = cudaMalloc(&ptr, sizeof(int64_t) * host.size());
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
        CUDA_THROW(cudaMemcpyAsync(ptr, host.data(), sizeof(int64_t) * host.size(), cudaMemcpyHostToDevice, stream));
        dev.reset(ptr);
    };

    // Store sizes for kernel dispatches
    nnzP_ = P_idx_host.size();
    nnzA_ = A_idx_host.size();
    nnzAt_ = At_idx_host.size();

    // Compute SOC offsets and reorder H index maps before uploading.
    // The KKT pattern above is built in original cone order, but derivative
    // data arrives in sorted order.  We reorder the index maps so that
    // populate_H_blocks_kernel can iterate in sorted order and write to the
    // correct (original-order) KKT positions.
    numSocCones_ = cones.numSocCones;
    std::vector<int64_t> soc_Hs_offsets;
    std::vector<int64_t> soc_kkt_offsets;
    if (cones.numSocCones > 0) {
        // Compute dense-only Hs offsets in SORTED order (indexes into sorted derivative arrays)
        soc_Hs_offsets.assign(cones.numSocCones + 1, 0);
        int64_t hs_acc = 0;
        for (int64_t k = 0; k < cones.numSocCones; ++k) {
            int64_t d = cones.socConeDims[k];  // sorted
            soc_Hs_offsets[k] = hs_acc;
            if (d <= 4) {
                hs_acc += d * (d + 1) / 2;
            }
        }
        soc_Hs_offsets[cones.numSocCones] = hs_acc;
        totalSocHsEntries_ = hs_acc;

        // Compute dense-only KKT offsets in SORTED order
        soc_kkt_offsets.assign(cones.numSocCones + 1, 0);
        int64_t kkt_acc = 0;
        for (int64_t k = 0; k < cones.numSocCones; ++k) {
            int64_t d = cones.socConeDims[k];  // sorted
            soc_kkt_offsets[k] = kkt_acc;
            if (d <= 4) {
                kkt_acc += d * d;
            }
        }
        soc_kkt_offsets[cones.numSocCones] = kkt_acc;
        totalSocKktEntries_ = kkt_acc;

        // Reorder H_soc index maps from original cone order to sorted cone order
        // so that sorted derivative slot j maps to original cone perm[j]'s KKT positions.
        if (!cones.socSortPerm.empty()) {
            const auto& perm = cones.socSortPerm;

            // Reorder dense SOC H indices (dim*dim per dense cone)
            if (!H_soc_idx_host.empty()) {
                std::vector<int64_t> orig_kkt_off(cones.numSocCones + 1, 0);
                for (int64_t i = 0; i < cones.numSocCones; ++i) {
                    int64_t d = socDimsForKKT[i];  // original
                    orig_kkt_off[i + 1] = orig_kkt_off[i] + ((d <= 4) ? d * d : 0);
                }

                std::vector<int64_t> reordered(H_soc_idx_host.size());
                for (int64_t j = 0; j < cones.numSocCones; ++j) {
                    int64_t d = cones.socConeDims[j];  // sorted
                    if (d > 4) continue;
                    int64_t orig_cone = perm[j];
                    int64_t orig_start = orig_kkt_off[orig_cone];
                    int64_t sorted_start = soc_kkt_offsets[j];
                    int64_t count = d * d;
                    for (int64_t k = 0; k < count; ++k) {
                        reordered[sorted_start + k] = H_soc_idx_host[orig_start + k];
                    }
                }
                H_soc_idx_host = reordered;
            }

            // Reorder sparse SOC indices
            if (!H_soc_sparse_diag_idx_host.empty()) {
                std::vector<int64_t> orig_sparse_off(cones.numSocCones + 1, 0);
                std::vector<int64_t> sorted_sparse_off(cones.numSocCones + 1, 0);
                for (int64_t i = 0; i < cones.numSocCones; ++i) {
                    int64_t od = socDimsForKKT[i];
                    orig_sparse_off[i + 1] = orig_sparse_off[i] + ((od > 4) ? od : 0);
                    int64_t sd = cones.socConeDims[i];
                    sorted_sparse_off[i + 1] = sorted_sparse_off[i] + ((sd > 4) ? sd : 0);
                }

                std::vector<int64_t> reord_diag(H_soc_sparse_diag_idx_host.size());
                std::vector<int64_t> reord_v1(H_soc_v1_col_idx_host.size());
                std::vector<int64_t> reord_v2(H_soc_v2_col_idx_host.size());
                std::vector<int64_t> reord_exp_v1(H_soc_exp_v1_du_idx_host.size());
                std::vector<int64_t> reord_exp_v2(H_soc_exp_v2_du_idx_host.size());
                for (int64_t j = 0; j < cones.numSocCones; ++j) {
                    if (cones.socConeDims[j] <= 4) continue;
                    int64_t orig_cone = perm[j];
                    int64_t orig_start = orig_sparse_off[orig_cone];
                    int64_t sorted_start = sorted_sparse_off[j];
                    int64_t dim = cones.socConeDims[j];
                    for (int64_t k = 0; k < dim; ++k) {
                        reord_diag[sorted_start + k] = H_soc_sparse_diag_idx_host[orig_start + k];
                        reord_v1[sorted_start + k] = H_soc_v1_col_idx_host[orig_start + k];
                        reord_v2[sorted_start + k] = H_soc_v2_col_idx_host[orig_start + k];
                        reord_exp_v1[sorted_start + k] = H_soc_exp_v1_du_idx_host[orig_start + k];
                        reord_exp_v2[sorted_start + k] = H_soc_exp_v2_du_idx_host[orig_start + k];
                    }
                }
                H_soc_sparse_diag_idx_host = reord_diag;
                H_soc_v1_col_idx_host = reord_v1;
                H_soc_v2_col_idx_host = reord_v2;
                H_soc_exp_v1_du_idx_host = reord_exp_v1;
                H_soc_exp_v2_du_idx_host = reord_exp_v2;

                // Reorder expansion row diagonal indices (2 per sparse cone)
                if (!H_soc_exp_diag_idx_host.empty()) {
                    std::vector<int64_t> orig_sparse_cnt(cones.numSocCones + 1, 0);
                    std::vector<int64_t> sorted_sparse_cnt(cones.numSocCones + 1, 0);
                    for (int64_t i = 0; i < cones.numSocCones; ++i) {
                        orig_sparse_cnt[i + 1] = orig_sparse_cnt[i] + ((socDimsForKKT[i] > 4) ? 1 : 0);
                        sorted_sparse_cnt[i + 1] = sorted_sparse_cnt[i] + ((cones.socConeDims[i] > 4) ? 1 : 0);
                    }
                    std::vector<int64_t> reord_exp_diag(H_soc_exp_diag_idx_host.size());
                    for (int64_t j = 0; j < cones.numSocCones; ++j) {
                        if (cones.socConeDims[j] <= 4) continue;
                        int64_t orig_cone = perm[j];
                        int64_t orig_idx = orig_sparse_cnt[orig_cone] * 2;
                        int64_t sorted_idx = sorted_sparse_cnt[j] * 2;
                        reord_exp_diag[sorted_idx] = H_soc_exp_diag_idx_host[orig_idx];
                        reord_exp_diag[sorted_idx + 1] = H_soc_exp_diag_idx_host[orig_idx + 1];
                    }
                    H_soc_exp_diag_idx_host = reord_exp_diag;
                }
            }

            // Rebuild sparse SOC offset/index arrays in sorted order for kernel dispatch
            soc_sparse_offsets_host.assign(cones.numSocCones + 1, 0);
            soc_sparse_indices_host.assign(cones.numSocCones, -1);
            {
                int64_t sparse_dim_acc = 0;
                int64_t sparse_idx = 0;
                for (int64_t i = 0; i < cones.numSocCones; ++i) {
                    int64_t d = cones.socConeDims[i];  // sorted
                    if (d > 4) {
                        soc_sparse_indices_host[i] = sparse_idx++;
                        sparse_dim_acc += d;
                    }
                    soc_sparse_offsets_host[i + 1] = sparse_dim_acc;
                }
            }
        }
    }

    // Upload all index maps to device
    upload_indices(I_diag_host, I_diag_idx_);
    upload_indices(negI_diag_host, negI_diag_idx_);
    upload_indices(P_idx_host, P_idx_);
    upload_indices(P_val_idx_host, P_val_idx_);
    upload_indices(A_idx_host, A_idx_);
    upload_indices(A_val_idx_host, A_val_idx_);
    upload_indices(At_idx_host, At_idx_);
    upload_indices(At_val_idx_host, At_val_idx_);
    upload_indices(H_diag_idx_host, H_diag_idx_);
    upload_indices(H_soc_idx_host, H_soc_idx_);
    upload_indices(H_exp_idx_host, H_exp_idx_);
    upload_indices(H_power_idx_host, H_power_idx_);
    upload_indices(q_idx_host, q_idx_);
    upload_indices(b_idx_host, b_idx_);
    upload_indices(c1_idx_host, c1_idx_);
    upload_indices(c2_idx_host, c2_idx_);
    upload_indices(J_I_row1_idx_host, J_I_row1_idx_);
    upload_indices(J_negI_row1_idx_host, J_negI_row1_idx_);
    upload_indices(J_I_row2_idx_host, J_I_row2_idx_);

    // Direct-x KKT index maps. The sparsity construction collects three
    // groups, all in cone-iteration / row-iteration order; we also build
    // per-entry kind/k/l metadata so the value-write kernel can look up
    // H_x[k, l] for the right cone kind without further bookkeeping.
    if (numXCones_ > 0) {
        upload_indices(xcone_E_idx_host_local, xcone_E_idx_);
        upload_indices(xcone_du_idx_host_local, xcone_du_idx_);
        upload_indices(xcone_stat_idx_host, xcone_stat_idx_);
        numXConeE_   = static_cast<int64_t>(xcone_E_idx_host_local.size());
        numXConeDu_  = static_cast<int64_t>(xcone_du_idx_host_local.size());
        numXConeStat_ = static_cast<int64_t>(xcone_stat_idx_host.size());
        // Reuse xcone_psd_* slots as generic stat/du index storage when
        // the value-write kernel lands. Per-entry meta gets uploaded as
        // packed int64_t arrays of (k, l, kind, h_off) so the kernel can
        // gather H_x without a per-cone dispatch table. Kind enum: 0
        // nonneg, 1 SOC, 2 PSD, 3 Exp, 4 Power, 5 GenPow.
        std::vector<int64_t> xcone_kinds(numXCones_);
        std::vector<int64_t> xcone_dims(numXCones_);
        std::vector<int64_t> xcone_psd_k(numXCones_, 0);
        std::vector<int64_t> xcone_numel_offsets(numXCones_ + 1, 0);
        // h_off[xc] = starting offset into the kind-specific H_x storage
        // for cone xc. Each kind has its own dense buffer:
        //   nonneg : diagonal, length totalXNonneg_
        //   SOC    : dense dim*dim per cone
        //   PSD    : dense dim*dim per cone (full square)
        //   Exp    : dense 3*3 per cone (totalXExpKkt_ = 9 * numXExp)
        //   Power  : dense 3*3 per cone (totalXPowKkt_ = 9 * numXPow)
        //   GenPow : dense dim*dim per cone (variable dim)
        std::vector<int64_t> xcone_h_off(numXCones_, 0);
        int64_t nn_acc = 0, soc_acc = 0, psd_acc = 0;
        int64_t exp_acc = 0, pow_acc = 0, gp_acc = 0;
        for (int64_t i = 0; i < numXCones_; ++i) {
            const auto& xc = cones.x_cones[i];
            xcone_dims[i] = static_cast<int64_t>(xc.indices.size());
            xcone_numel_offsets[i + 1] = xcone_numel_offsets[i] + xcone_dims[i];
            switch (xc.kind) {
                case XConeKind::Nonneg:
                    xcone_kinds[i] = 0;
                    xcone_h_off[i] = nn_acc;
                    nn_acc += xcone_dims[i];
                    break;
                case XConeKind::SOC:
                    xcone_kinds[i] = 1;
                    xcone_h_off[i] = soc_acc;
                    soc_acc += xcone_dims[i] * xcone_dims[i];
                    break;
                case XConeKind::PSD:
                    xcone_kinds[i] = 2;
                    xcone_psd_k[i] = xc.psd_k;
                    xcone_h_off[i] = psd_acc;
                    psd_acc += xcone_dims[i] * xcone_dims[i];
                    break;
                case XConeKind::Exp:
                    xcone_kinds[i] = 3;
                    xcone_h_off[i] = exp_acc;
                    exp_acc += 9;  // 3*3
                    break;
                case XConeKind::Power:
                    xcone_kinds[i] = 4;
                    xcone_h_off[i] = pow_acc;
                    pow_acc += 9;  // 3*3
                    break;
                case XConeKind::GenPower:
                    xcone_kinds[i] = 5;
                    xcone_h_off[i] = gp_acc;
                    gp_acc += xcone_dims[i] * xcone_dims[i];
                    break;
            }
        }
        totalXNonneg_ = nn_acc;
        totalXExpKkt_ = exp_acc;
        totalXPowKkt_ = pow_acc;
        totalXGenPowKkt_ = gp_acc;
        // Upload metadata.
        std::vector<int64_t> xcone_indices_flat;
        xcone_indices_flat.reserve(totalXConeNumel_);
        for (const auto& xc : cones.x_cones) {
            for (int64_t v : xc.indices) xcone_indices_flat.push_back(v);
        }
        upload_indices(xcone_kinds, d_xcone_kinds_);
        upload_indices(xcone_dims, d_xcone_dims_);
        upload_indices(xcone_numel_offsets, d_xcone_numel_offsets_);
        upload_indices(xcone_indices_flat, d_xcone_indices_);
        upload_indices(xcone_psd_k, d_xcone_psd_k_);
        // Also upload the per-entry meta tuples (xc_idx, k, l) by
        // flattening into separate int64_t arrays.
        std::vector<int64_t> E_meta_xc(xcone_E_meta_host.size());
        std::vector<int64_t> E_meta_k (xcone_E_meta_host.size());
        for (size_t i = 0; i < xcone_E_meta_host.size(); ++i) {
            E_meta_xc[i] = xcone_E_meta_host[i].xc_idx;
            E_meta_k[i]  = xcone_E_meta_host[i].k;
        }
        std::vector<int64_t> du_meta_xc(xcone_du_meta_host.size());
        std::vector<int64_t> du_meta_k (xcone_du_meta_host.size());
        std::vector<int64_t> du_meta_l (xcone_du_meta_host.size());
        for (size_t i = 0; i < xcone_du_meta_host.size(); ++i) {
            du_meta_xc[i] = xcone_du_meta_host[i].xc_idx;
            du_meta_k[i]  = xcone_du_meta_host[i].k;
            du_meta_l[i]  = xcone_du_meta_host[i].l;
        }
        std::vector<int64_t> stat_meta_xc(xcone_stat_meta_host.size());
        std::vector<int64_t> stat_meta_k (xcone_stat_meta_host.size());
        std::vector<int64_t> stat_meta_l (xcone_stat_meta_host.size());
        for (size_t i = 0; i < xcone_stat_meta_host.size(); ++i) {
            stat_meta_xc[i] = xcone_stat_meta_host[i].xc_idx;
            stat_meta_k[i]  = xcone_stat_meta_host[i].k;
            stat_meta_l[i]  = xcone_stat_meta_host[i].l;
        }
        // Reuse xcone_nonneg_*_idx_ slots to also stash these meta tables;
        // dedicated members would inflate the header, but we only ever
        // need them inside `populate_xcone_H_blocks_kernel`. Pack into a
        // single int64 array of (E_idx, E_xc, E_k | du_idx, du_xc, du_k,
        // du_l | stat_idx, stat_xc, stat_k, stat_l) and stash on the
        // device.
        upload_indices(E_meta_xc,    xcone_nonneg_E_idx_);
        upload_indices(E_meta_k,     xcone_nonneg_du_idx_);
        upload_indices(du_meta_xc,   xcone_soc_E_idx_);
        upload_indices(du_meta_k,    xcone_soc_du_idx_);
        upload_indices(du_meta_l,    xcone_soc_stat_idx_);
        upload_indices(stat_meta_xc, xcone_psd_E_idx_);
        upload_indices(stat_meta_k,  xcone_psd_du_idx_);
        upload_indices(stat_meta_l,  xcone_stat_meta_l_);
        upload_indices(xcone_h_off,  d_xcone_psd_svec_offsets_);
        // Also stash totals for SOC and PSD direct-x KKT-block sizing.
        total_x_soc_kkt_ = soc_acc;
        total_x_psd_kkt_ = psd_acc;
    }

    // Store sizes for kernel dispatches
    nnzP_ = P_idx_host.size();
    nnzA_ = A_idx_host.size();
    nnzAt_ = At_idx_host.size();

    // Store SOC variable-dim info for populate_H_blocks_kernel
    // Now only dense cones (dim<=4) use the dense Hs/KKT arrays
    numSocCones_ = cones.numSocCones;
    if (cones.numSocCones > 0) {
        upload_indices(cones.socConeDims, d_soc_dims_);
        upload_indices(soc_Hs_offsets, d_soc_Hs_offsets_);
        upload_indices(soc_kkt_offsets, d_soc_kkt_offsets_);
        if (numSparseSoc_ > 0) {
            upload_indices(soc_sparse_offsets_host, d_soc_sparse_offsets_);
            upload_indices(soc_sparse_indices_host, d_soc_sparse_indices_);
            upload_indices(H_soc_sparse_diag_idx_host, H_soc_sparse_diag_idx_);
            upload_indices(H_soc_v1_col_idx_host, H_soc_v1_col_idx_);
            upload_indices(H_soc_v2_col_idx_host, H_soc_v2_col_idx_);
            upload_indices(H_soc_exp_v1_du_idx_host, H_soc_exp_v1_du_idx_);
            upload_indices(H_soc_exp_v2_du_idx_host, H_soc_exp_v2_du_idx_);
            upload_indices(H_soc_exp_diag_idx_host, H_soc_exp_diag_idx_);
        }
    }

    // Store PSD cone variable-dim info for populate_H_blocks_kernel
    numPsdCones_ = cones.numPsdCones;
    if (cones.numPsdCones > 0) {
        // Compute Hs offsets (upper-tri of svec_dim x svec_dim) and KKT offsets (full svec_dim x svec_dim)
        // in SORTED order (derivative data arrives in sorted order)
        auto psdHsForDim = [](int64_t dim) -> int64_t {
            int64_t svec_dim = dim * (dim + 1) / 2;
            return svec_dim * (svec_dim + 1) / 2;
        };
        auto psdKktForDim = [](int64_t dim) -> int64_t {
            int64_t svec_dim = dim * (dim + 1) / 2;
            return svec_dim * svec_dim;
        };

        std::vector<int64_t> psd_Hs_offsets(cones.numPsdCones + 1, 0);
        std::vector<int64_t> psd_kkt_offsets(cones.numPsdCones + 1, 0);
        int64_t hs_acc = 0, kkt_acc = 0;
        for (int64_t k = 0; k < cones.numPsdCones; ++k) {
            psd_Hs_offsets[k] = hs_acc;
            psd_kkt_offsets[k] = kkt_acc;
            hs_acc += psdHsForDim(cones.psdConeDims[k]);
            kkt_acc += psdKktForDim(cones.psdConeDims[k]);
        }
        psd_Hs_offsets[cones.numPsdCones] = hs_acc;
        psd_kkt_offsets[cones.numPsdCones] = kkt_acc;
        totalPsdHsEntries_ = hs_acc;
        totalPsdKktEntries_ = kkt_acc;

        // Reorder PSD indices from original cone order to sorted cone order
        if (!cones.psdSortPerm.empty() && !H_psd_idx_host.empty()) {
            const auto& perm = cones.psdSortPerm;
            const auto& psdDimsOrig = cones.psdConeDimsOriginal;

            std::vector<int64_t> orig_kkt_off(cones.numPsdCones + 1, 0);
            for (int64_t i = 0; i < cones.numPsdCones; ++i) {
                orig_kkt_off[i + 1] = orig_kkt_off[i] + psdKktForDim(psdDimsOrig[i]);
            }

            std::vector<int64_t> reordered(H_psd_idx_host.size());
            for (int64_t j = 0; j < cones.numPsdCones; ++j) {
                int64_t orig_cone = perm[j];
                int64_t orig_start = orig_kkt_off[orig_cone];
                int64_t sorted_start = psd_kkt_offsets[j];
                int64_t count = psdKktForDim(cones.psdConeDims[j]);
                for (int64_t k = 0; k < count; ++k) {
                    reordered[sorted_start + k] = H_psd_idx_host[orig_start + k];
                }
            }
            H_psd_idx_host = reordered;
        }

        // Build svec_dims array for kernel dispatch
        std::vector<int64_t> psd_svec_dims(cones.numPsdCones);
        for (int64_t k = 0; k < cones.numPsdCones; ++k) {
            psd_svec_dims[k] = cones.psdConeDims[k] * (cones.psdConeDims[k] + 1) / 2;
        }

        upload_indices(H_psd_idx_host, H_psd_idx_);
        upload_indices(psd_Hs_offsets, d_psd_Hs_offsets_);
        upload_indices(psd_kkt_offsets, d_psd_kkt_offsets_);
        upload_indices(psd_svec_dims, d_psd_svec_dims_);
    }

    // Store GenPowerCone variable-dim info for populate_H_blocks_kernel (sparse)
    numGenPowerCones_ = cones.numGenPowerCones;
    if (cones.numGenPowerCones > 0) {
        // Build dims array and sparse offsets (prefix sum of dim per cone)
        std::vector<int64_t> genpow_dims(cones.numGenPowerCones);
        std::vector<int64_t> genpow_sparse_offsets(cones.numGenPowerCones + 1, 0);
        int64_t dim_acc = 0;
        for (int64_t k = 0; k < cones.numGenPowerCones; ++k) {
            int64_t d = cones.genPowerDim1s[k] + cones.genPowerDim2s[k];
            genpow_dims[k] = d;
            genpow_sparse_offsets[k] = dim_acc;
            dim_acc += d;
        }
        genpow_sparse_offsets[cones.numGenPowerCones] = dim_acc;
        totalGenpowDim_ = dim_acc;

        upload_indices(genpow_dims, d_genpow_dims_);
        upload_indices(genpow_sparse_offsets, d_genpow_sparse_offsets_);

        // Upload sparse GenPowerCone index arrays
        upload_indices(H_genpow_sparse_diag_idx_host, H_genpow_sparse_diag_idx_);
        upload_indices(H_genpow_v1_col_idx_host, H_genpow_v1_col_idx_);
        upload_indices(H_genpow_v2_col_idx_host, H_genpow_v2_col_idx_);
        upload_indices(H_genpow_v3_col_idx_host, H_genpow_v3_col_idx_);
        upload_indices(H_genpow_exp_v1_du_idx_host, H_genpow_exp_v1_du_idx_);
        upload_indices(H_genpow_exp_v2_du_idx_host, H_genpow_exp_v2_du_idx_);
        upload_indices(H_genpow_exp_v3_du_idx_host, H_genpow_exp_v3_du_idx_);
        upload_indices(H_genpow_exp_diag_idx_host, H_genpow_exp_diag_idx_);
    }

    // Upload direct-x SOC rank-2 expansion index arrays.
    if (numSparseXSoc_ > 0) {
        upload_indices(H_xcone_soc_du_stat_diag_idx_host, H_xcone_soc_du_stat_diag_idx_);
        upload_indices(H_xcone_soc_du_dx_diag_idx_host,   H_xcone_soc_du_dx_diag_idx_);
        upload_indices(H_xcone_soc_v1_col_stat_idx_host,  H_xcone_soc_v1_col_stat_idx_);
        upload_indices(H_xcone_soc_v2_col_stat_idx_host,  H_xcone_soc_v2_col_stat_idx_);
        upload_indices(H_xcone_soc_v1_col_dx_idx_host,    H_xcone_soc_v1_col_dx_idx_);
        upload_indices(H_xcone_soc_v2_col_dx_idx_host,    H_xcone_soc_v2_col_dx_idx_);
        upload_indices(H_xcone_soc_exp_v1_du_idx_host,    H_xcone_soc_exp_v1_du_idx_);
        upload_indices(H_xcone_soc_exp_v2_du_idx_host,    H_xcone_soc_exp_v2_du_idx_);
        upload_indices(H_xcone_soc_exp_diag_idx_host,     H_xcone_soc_exp_diag_idx_);
        std::vector<int64_t> dim_offsets(numSparseXSoc_ + 1, 0);
        std::vector<int64_t> sparse_to_xc(numSparseXSoc_, 0);
        std::vector<int64_t> sparse_dims(numSparseXSoc_, 0);
        for (int64_t s = 0; s < numSparseXSoc_; ++s) {
            const auto& info = xsoc_sparse_info[s];
            sparse_to_xc[s] = info.cone_x_idx;
            sparse_dims[s]  = info.soc_dim;
            dim_offsets[s + 1] = dim_offsets[s] + info.soc_dim;
        }
        upload_indices(dim_offsets, d_xcone_soc_sparse_dim_offsets_);
        upload_indices(sparse_to_xc, d_xcone_soc_sparse_to_xc_);
        upload_indices(sparse_dims, d_xcone_soc_sparse_dims_);
    }

    // Upload direct-x GenPow rank-3 expansion index arrays.
    if (cones.numXGenPowerCones > 0) {
        upload_indices(H_xcone_genpow_du_stat_diag_idx_host, H_xcone_genpow_du_stat_diag_idx_);
        upload_indices(H_xcone_genpow_du_dx_diag_idx_host,   H_xcone_genpow_du_dx_diag_idx_);
        upload_indices(H_xcone_genpow_v1_col_stat_idx_host,  H_xcone_genpow_v1_col_stat_idx_);
        upload_indices(H_xcone_genpow_v2_col_stat_idx_host,  H_xcone_genpow_v2_col_stat_idx_);
        upload_indices(H_xcone_genpow_v3_col_stat_idx_host,  H_xcone_genpow_v3_col_stat_idx_);
        upload_indices(H_xcone_genpow_v1_col_dx_idx_host,    H_xcone_genpow_v1_col_dx_idx_);
        upload_indices(H_xcone_genpow_v2_col_dx_idx_host,    H_xcone_genpow_v2_col_dx_idx_);
        upload_indices(H_xcone_genpow_v3_col_dx_idx_host,    H_xcone_genpow_v3_col_dx_idx_);
        upload_indices(H_xcone_genpow_exp_v1_du_idx_host,    H_xcone_genpow_exp_v1_du_idx_);
        upload_indices(H_xcone_genpow_exp_v2_du_idx_host,    H_xcone_genpow_exp_v2_du_idx_);
        upload_indices(H_xcone_genpow_exp_v3_du_idx_host,    H_xcone_genpow_exp_v3_du_idx_);
        upload_indices(H_xcone_genpow_exp_diag_idx_host,     H_xcone_genpow_exp_diag_idx_);
        // Per-cone dim offset table for kernel dispatch.
        std::vector<int64_t> dim_offsets(cones.numXGenPowerCones + 1, 0);
        for (int64_t k = 0; k < cones.numXGenPowerCones; ++k) {
            dim_offsets[k + 1] = dim_offsets[k] + xgenpow_expansion_info[k].genpow_dim;
        }
        upload_indices(dim_offsets, d_xcone_genpow_dim_offsets_);
    }

    // Store c3 index (single value)
    if (c3_idx_val >= 0) {
        int64_t* ptr = nullptr;
        auto e = cudaMalloc(&ptr, sizeof(int64_t));
        if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
        CUDA_THROW(cudaMemcpyAsync(ptr, &c3_idx_val, sizeof(int64_t), cudaMemcpyHostToDevice, stream));
        c3_idx_.reset(ptr);
    }

    cudaStreamSynchronize(stream);

    KKT = std::move(kkt);

    // Initialize cuDSS solver
    initialize_cudss(stream);
}

DiffKKT::~DiffKKT() {
    if (solverType_ == KKTSolverType::Riccati) {
        // diff_riccati_ is a unique_ptr, cleaned up automatically
    } else {
        cleanup_cudss();
    }
}

void DiffKKT::initialize_cudss(cudaStream_t stream) {
    if (cudss_initialized_) return;

    // Create cuDSS handle
    cudssStatus_t status = cudssCreate(&cudss_handle_);
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("DiffKKT cudssCreate failed: cuDSS ")
            + cudss_status_string(status));
    }

    cudssSetStream(cudss_handle_, stream);

    // Create config
    status = cudssConfigCreate(&cudss_config_);
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("DiffKKT cudssConfigCreate failed: cuDSS ")
            + cudss_status_string(status));
    }

    // Set batch size
    int batch_count = static_cast<int>(batchSize);
    status = cudssConfigSet(cudss_config_, CUDSS_CONFIG_UBATCH_SIZE, &batch_count, sizeof(batch_count));
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("DiffKKT cudssConfigSet UBATCH_SIZE failed: cuDSS ")
            + cudss_status_string(status));
    }

    // Process all batches at once (not one at a time)
    // Without this, cuDSS defaults to processing only batch 0, causing gradient pollution
    int ubatch_index = -1;
    status = cudssConfigSet(cudss_config_, CUDSS_CONFIG_UBATCH_INDEX, &ubatch_index, sizeof(ubatch_index));
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("DiffKKT cudssConfigSet UBATCH_INDEX failed: cuDSS ")
            + cudss_status_string(status));
    }

    // Disable pivoting for quasi-definite systems
    // The HSDE augmented system [I J; J' -εI] is quasi-definite (matching forward KKT behavior)
    int pivot_type = CUDSS_PIVOT_NONE;
    status = cudssConfigSet(cudss_config_, CUDSS_CONFIG_PIVOT_TYPE, &pivot_type, sizeof(pivot_type));
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("DiffKKT cudssConfigSet PIVOT_TYPE failed: cuDSS ")
            + cudss_status_string(status));
    }

    // Iterative refinement for improved solve accuracy (matching forward KKT)
    int ir_n_steps = 2;
    status = cudssConfigSet(cudss_config_, CUDSS_CONFIG_IR_N_STEPS, &ir_n_steps, sizeof(ir_n_steps));
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("DiffKKT cudssConfigSet IR_N_STEPS failed: cuDSS ")
            + cudss_status_string(status));
    }

    // Create data object
    status = cudssDataCreate(cudss_handle_, &cudss_data_);
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("DiffKKT cudssDataCreate failed: cuDSS ")
            + cudss_status_string(status));
    }

    // Create matrix objects
    status = cudssMatrixCreateCsr(
        &kkt_matrix_, augdim, augdim, KKT.nnz(),
        KKT.rowOffsets(), nullptr, KKT.colIndices(), KKT.values(),
        MOREAU_CUDSS_CSR_I64_F64,
        CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, CUDSS_BASE_ZERO
    );
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("DiffKKT cudssMatrixCreateCsr failed: cuDSS ")
            + cudss_status_string(status));
    }

    status = cudssMatrixCreateDn(
        &rhs_matrix_, augdim, 1, augdim,
        nullptr, MOREAU_CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR
    );
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("DiffKKT cudssMatrixCreateDn (rhs) failed: cuDSS ")
            + cudss_status_string(status));
    }

    status = cudssMatrixCreateDn(
        &sol_matrix_, augdim, 1, augdim,
        nullptr, MOREAU_CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR
    );
    if (status != CUDSS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("DiffKKT cudssMatrixCreateDn (sol) failed: cuDSS ")
            + cudss_status_string(status));
    }

    // Symbolic analysis
    status = cudssExecute(
        cudss_handle_, CUDSS_PHASE_ANALYSIS,
        cudss_config_, cudss_data_,
        kkt_matrix_, sol_matrix_, rhs_matrix_
    );
    if (status != CUDSS_STATUS_SUCCESS) {
        std::cerr << "DiffKKT symbolic analysis failed (cuDSS status: "
                  << cudss_status_string(status) << ")" << std::endl;
        std::cerr << "  KKT dim: " << augdim << "x" << augdim
                  << ", nnz: " << KKT.nnz()
                  << ", batch: " << batchSize << std::endl;
        std::cerr << gpu_memory_info() << std::endl;
        throw std::runtime_error(
            std::string("DiffKKT symbolic analysis failed: cuDSS ")
            + cudss_status_string(status)
            + " (KKT " + std::to_string(augdim) + "x" + std::to_string(augdim)
            + ", nnz=" + std::to_string(KKT.nnz())
            + ", batch=" + std::to_string(batchSize) + ")");
    }

    cudss_initialized_ = true;
}

void DiffKKT::cleanup_cudss() {
    if (!cudss_initialized_) return;

    cudaDeviceSynchronize();

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

// CUDA kernel to initialize DiffKKT values
__global__ void init_diff_kkt_values_kernel(
    double* __restrict__ values,
    const int64_t* __restrict__ I_diag_idx,
    const int64_t* __restrict__ negI_diag_idx,
    const int64_t* __restrict__ J_I_row1_idx,
    const int64_t* __restrict__ J_negI_row1_idx,
    const int64_t* __restrict__ J_I_row2_idx,
    int64_t jdim,
    int64_t m,
    int64_t nnzKKT,
    int64_t batchSize,
    double reg
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double* vals = values + batch * nnzKKT;

    // Initialize I diagonal with 1.0 (top-left block of augmented K)
    for (int64_t i = threadIdx.x; i < jdim; i += blockDim.x) {
        vals[I_diag_idx[i]] = 1.0;
    }

    // Initialize -εI diagonal with -reg (bottom-right block of augmented K)
    for (int64_t i = threadIdx.x; i < jdim; i += blockDim.x) {
        vals[negI_diag_idx[i]] = -reg;
    }

    // Initialize I block within J: J[n:n+m, n:n+m] = I
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        vals[J_I_row1_idx[i]] = 1.0;
    }

    // Initialize -I block within J: J[n:n+m, n+m:n+2m] = -I
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        vals[J_negI_row1_idx[i]] = -1.0;
    }

    // Initialize I block within J: J[n+m:n+2m, n:n+m] = I
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        vals[J_I_row2_idx[i]] = 1.0;
    }
}

// CUDA kernel to populate J blocks
__global__ void populate_J_blocks_kernel(
    double* __restrict__ kkt_values,
    const int64_t* __restrict__ P_idx, const int64_t* __restrict__ P_val_idx, const double* __restrict__ P_values, int64_t nnzP, int64_t nnzP_orig,
    const int64_t* __restrict__ A_idx, const int64_t* __restrict__ A_val_idx, const double* __restrict__ A_values, int64_t nnzA, int64_t nnzA_orig,
    const int64_t* __restrict__ At_idx, const int64_t* __restrict__ At_val_idx, int64_t nnzAt,
    const int64_t* __restrict__ q_idx, const double* __restrict__ q, int64_t n,
    const int64_t* __restrict__ b_idx, const double* __restrict__ b, int64_t m,
    const int64_t* __restrict__ c1_idx, const double* __restrict__ c1,
    const int64_t* __restrict__ c2_idx, const double* __restrict__ c2,
    const int64_t* __restrict__ c3_idx, const double* __restrict__ c3,
    int64_t nnzKKT,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double* vals = kkt_values + batch * nnzKKT;
    const double* P_batch = P_values + batch * nnzP_orig;
    const double* A_batch = A_values + batch * nnzA_orig;

    // P values - use P_val_idx to index into original P values
    for (int64_t i = threadIdx.x; i < nnzP; i += blockDim.x) {
        vals[P_idx[i]] = P_batch[P_val_idx[i]];
    }

    // A values - use A_val_idx to index into original A values
    for (int64_t i = threadIdx.x; i < nnzA; i += blockDim.x) {
        vals[A_idx[i]] = A_batch[A_val_idx[i]];
    }

    // A' values - use At_val_idx to index into A values
    for (int64_t i = threadIdx.x; i < nnzAt; i += blockDim.x) {
        vals[At_idx[i]] = A_batch[At_val_idx[i]];
    }

    // q values
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        vals[q_idx[i]] = q[batch * n + i];
    }

    // b values (negated)
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        vals[b_idx[i]] = -b[batch * m + i];
    }

    // c1 values
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        vals[c1_idx[i]] = c1[batch * n + i];
    }

    // c2 values
    for (int64_t i = threadIdx.x; i < m; i += blockDim.x) {
        vals[c2_idx[i]] = c2[batch * m + i];
    }

    // c3 value
    if (threadIdx.x == 0 && c3_idx != nullptr) {
        vals[c3_idx[0]] = c3[batch];
    }
}

// CUDA kernel to populate H (cone derivative) blocks
__global__ void populate_H_blocks_kernel(
    double* __restrict__ kkt_values,
    const int64_t* __restrict__ H_diag_idx,
    const double* __restrict__ nonneg_H,
    // Dense SOC (dim<=4)
    const int64_t* __restrict__ H_soc_idx,
    const double* __restrict__ soc_H,
    const int64_t* __restrict__ d_soc_dims,
    const int64_t* __restrict__ d_soc_Hs_offsets,
    const int64_t* __restrict__ d_soc_kkt_offsets,
    int64_t numSocCones,
    int64_t totalSocHsEntries,
    int64_t totalSocKktEntries,
    // Sparse SOC (dim>4)
    const int64_t* __restrict__ H_soc_sparse_diag_idx,
    const int64_t* __restrict__ H_soc_v1_col_idx,
    const int64_t* __restrict__ H_soc_v2_col_idx,
    const int64_t* __restrict__ H_soc_exp_v1_du_idx,
    const int64_t* __restrict__ H_soc_exp_v2_du_idx,
    const int64_t* __restrict__ H_soc_exp_diag_idx,
    const double* __restrict__ soc_sparse_diag,
    const double* __restrict__ soc_sparse_v1,
    const double* __restrict__ soc_sparse_v2,
    const double* __restrict__ soc_sparse_c1,
    const double* __restrict__ soc_sparse_c2,
    const int64_t* __restrict__ d_soc_sparse_offsets,
    const int64_t* __restrict__ d_soc_sparse_indices,
    int64_t totalSparseSocDim,
    int64_t numSparseSoc,
    // Exp/Power
    const int64_t* __restrict__ H_exp_idx,
    const double* __restrict__ exp_H,
    const int64_t* __restrict__ H_power_idx,
    const double* __restrict__ power_H,
    // PSD cones (dense svec_dim x svec_dim, symmetric)
    const int64_t* __restrict__ H_psd_idx,
    const double* __restrict__ psd_H,
    const int64_t* __restrict__ d_psd_Hs_offsets,
    const int64_t* __restrict__ d_psd_kkt_offsets,
    const int64_t* __restrict__ d_psd_svec_dims,
    int64_t numPsdCones,
    int64_t totalPsdHsEntries,
    int64_t totalPsdKktEntries,
    // Sparse GenPowerCone (diagonal + rank-3)
    const int64_t* __restrict__ H_genpow_sparse_diag_idx,
    const int64_t* __restrict__ H_genpow_v1_col_idx,
    const int64_t* __restrict__ H_genpow_v2_col_idx,
    const int64_t* __restrict__ H_genpow_v3_col_idx,
    const int64_t* __restrict__ H_genpow_exp_v1_du_idx,
    const int64_t* __restrict__ H_genpow_exp_v2_du_idx,
    const int64_t* __restrict__ H_genpow_exp_v3_du_idx,
    const int64_t* __restrict__ H_genpow_exp_diag_idx,
    const double* __restrict__ genpow_sparse_diag,
    const double* __restrict__ genpow_sparse_left1,
    const double* __restrict__ genpow_sparse_right1,
    const double* __restrict__ genpow_sparse_left2,
    const double* __restrict__ genpow_sparse_right2,
    const double* __restrict__ genpow_sparse_left3,
    const double* __restrict__ genpow_sparse_c3,
    const int64_t* __restrict__ d_genpow_sparse_offsets,
    int64_t numGenPowerCones,
    int64_t totalGenpowDim,
    int64_t numZeroCones,
    int64_t numNonnegCones,
    int64_t numExpCones,
    int64_t numPowerCones,
    int64_t nnzKKT,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double* vals = kkt_values + batch * nnzKKT;

    // Zero cones: -H = -I for zero cone
    int64_t h_idx = 0;
    for (int64_t i = threadIdx.x; i < numZeroCones; i += blockDim.x) {
        vals[H_diag_idx[h_idx + i]] = -1.0;
    }
    h_idx += numZeroCones;

    // Nonnegative cones: H diagonal
    for (int64_t i = threadIdx.x; i < numNonnegCones; i += blockDim.x) {
        vals[H_diag_idx[h_idx + i]] = -nonneg_H[batch * numNonnegCones + i];
    }
    h_idx += numNonnegCones;

    // Dense SOC cones (dim<=4): variable-dim dense blocks, symmetric (self-dual cone)
    // Uses dense-only offsets — sparse cones have offset 0 and are skipped by totalSocKktEntries
    for (int64_t tid = threadIdx.x; tid < totalSocKktEntries; tid += blockDim.x) {
        int64_t lo_k = 0, hi_k = numSocCones;
        while (lo_k < hi_k) {
            int64_t mid = (lo_k + hi_k) / 2;
            if (d_soc_kkt_offsets[mid + 1] <= tid) {
                lo_k = mid + 1;
            } else {
                hi_k = mid;
            }
        }
        int64_t k = lo_k;
        int64_t dim = d_soc_dims[k];
        int64_t local_idx = tid - d_soc_kkt_offsets[k];
        int64_t r = local_idx / dim;
        int64_t c = local_idx % dim;

        int64_t lo_rc = (r <= c) ? r : c;
        int64_t hi_rc = (r <= c) ? c : r;
        int64_t tri_idx = lo_rc * (2 * dim - lo_rc + 1) / 2 + (hi_rc - lo_rc);

        int64_t hs_off = d_soc_Hs_offsets[k];
        int64_t soc_h_base = batch * totalSocHsEntries + hs_off;
        vals[H_soc_idx[tid]] = -soc_H[soc_h_base + tri_idx];
    }

    // Sparse SOC cones (dim>4): populate using rank-2 decomposition
    // H-block rows: diagonal + expansion column entries
    // Expansion rows: -v entries at du columns + diagonal 1
    if (numSparseSoc > 0) {
        // 1. Diagonal entries in H-block rows: -diag[i]
        for (int64_t tid = threadIdx.x; tid < totalSparseSocDim; tid += blockDim.x) {
            vals[H_soc_sparse_diag_idx[tid]] = -soc_sparse_diag[batch * totalSparseSocDim + tid];
        }

        // 2. v1 expansion column entries in H-block rows: -c1 * v1[i]
        for (int64_t tid = threadIdx.x; tid < totalSparseSocDim; tid += blockDim.x) {
            // Find which sparse cone this entry belongs to (binary search in sparse_offsets)
            int64_t lo_k = 0, hi_k = numSocCones;
            while (lo_k < hi_k) {
                int64_t mid = (lo_k + hi_k) / 2;
                if (d_soc_sparse_offsets[mid + 1] <= tid) {
                    lo_k = mid + 1;
                } else {
                    hi_k = mid;
                }
            }
            int64_t sparse_cone = d_soc_sparse_indices[lo_k];
            double c1_val = soc_sparse_c1[batch * numSparseSoc + sparse_cone];
            vals[H_soc_v1_col_idx[tid]] = -c1_val * soc_sparse_v1[batch * totalSparseSocDim + tid];
        }

        // 3. v2 expansion column entries in H-block rows: -c2 * v2[i]
        for (int64_t tid = threadIdx.x; tid < totalSparseSocDim; tid += blockDim.x) {
            int64_t lo_k = 0, hi_k = numSocCones;
            while (lo_k < hi_k) {
                int64_t mid = (lo_k + hi_k) / 2;
                if (d_soc_sparse_offsets[mid + 1] <= tid) {
                    lo_k = mid + 1;
                } else {
                    hi_k = mid;
                }
            }
            int64_t sparse_cone = d_soc_sparse_indices[lo_k];
            double c2_val = soc_sparse_c2[batch * numSparseSoc + sparse_cone];
            vals[H_soc_v2_col_idx[tid]] = -c2_val * soc_sparse_v2[batch * totalSparseSocDim + tid];
        }

        // 4. Expansion row du-column entries: -v1[i] and -v2[i]
        for (int64_t tid = threadIdx.x; tid < totalSparseSocDim; tid += blockDim.x) {
            vals[H_soc_exp_v1_du_idx[tid]] = -soc_sparse_v1[batch * totalSparseSocDim + tid];
            vals[H_soc_exp_v2_du_idx[tid]] = -soc_sparse_v2[batch * totalSparseSocDim + tid];
        }

        // 5. Expansion row diagonal entries: 1.0
        for (int64_t tid = threadIdx.x; tid < 2 * numSparseSoc; tid += blockDim.x) {
            vals[H_soc_exp_diag_idx[tid]] = 1.0;
        }
    }

    // Exp cones: 3x3 dense blocks (9 entries per cone), NOT symmetric
    for (int64_t tid = threadIdx.x; tid < numExpCones * 9; tid += blockDim.x) {
        int64_t exp_offset = batch * numExpCones * 9 + tid;
        vals[H_exp_idx[tid]] = -exp_H[exp_offset];
    }

    // Power cones: 3x3 dense blocks (9 entries per cone), NOT symmetric
    for (int64_t tid = threadIdx.x; tid < numPowerCones * 9; tid += blockDim.x) {
        int64_t pow_offset = batch * numPowerCones * 9 + tid;
        vals[H_power_idx[tid]] = -power_H[pow_offset];
    }

    // PSD cones: dense svec_dim x svec_dim blocks, symmetric (self-dual cone)
    // psd_H stores upper triangle in svec order; KKT needs full matrix.
    // Map full (r,c) to upper-tri index for reading (same approach as dense SOC).
    if (numPsdCones > 0 && totalPsdKktEntries > 0) {
        for (int64_t tid = threadIdx.x; tid < totalPsdKktEntries; tid += blockDim.x) {
            // Binary search for which PSD cone this tid belongs to
            int64_t lo_k = 0, hi_k = numPsdCones;
            while (lo_k < hi_k) {
                int64_t mid = (lo_k + hi_k) / 2;
                if (d_psd_kkt_offsets[mid + 1] <= tid) {
                    lo_k = mid + 1;
                } else {
                    hi_k = mid;
                }
            }
            int64_t k = lo_k;
            int64_t svec_dim = d_psd_svec_dims[k];
            int64_t local_idx = tid - d_psd_kkt_offsets[k];
            int64_t r = local_idx / svec_dim;
            int64_t c = local_idx % svec_dim;

            // Map (r,c) to column-major upper-tri index: col*(col+1)/2 + row
            // (scatter_jacobian_col_kernel stores H[k*(k+1)/2 + i] for col k, row i)
            int64_t lo_rc = (r <= c) ? r : c;
            int64_t hi_rc = (r <= c) ? c : r;
            int64_t tri_idx = hi_rc * (hi_rc + 1) / 2 + lo_rc;

            int64_t hs_off = d_psd_Hs_offsets[k];
            int64_t psd_h_base = batch * totalPsdHsEntries + hs_off;
            vals[H_psd_idx[tid]] = -psd_H[psd_h_base + tri_idx];
        }
    }

    // GenPowerCone: sparse decomposition (diagonal + rank-3)
    // H = diag + left1*right1^T + left2*right2^T + c3*left3*left3^T
    if (numGenPowerCones > 0) {
        // 1. Diagonal entries in H-block rows: -diag[i]
        for (int64_t tid = threadIdx.x; tid < totalGenpowDim; tid += blockDim.x) {
            vals[H_genpow_sparse_diag_idx[tid]] = -genpow_sparse_diag[batch * totalGenpowDim + tid];
        }

        // 2. v1 expansion column entries in H-block rows: -left1[i]
        for (int64_t tid = threadIdx.x; tid < totalGenpowDim; tid += blockDim.x) {
            vals[H_genpow_v1_col_idx[tid]] = -genpow_sparse_left1[batch * totalGenpowDim + tid];
        }

        // 3. v2 expansion column entries in H-block rows: -left2[i]
        for (int64_t tid = threadIdx.x; tid < totalGenpowDim; tid += blockDim.x) {
            vals[H_genpow_v2_col_idx[tid]] = -genpow_sparse_left2[batch * totalGenpowDim + tid];
        }

        // 4. v3 expansion column entries in H-block rows: -c3*left3[i]
        for (int64_t tid = threadIdx.x; tid < totalGenpowDim; tid += blockDim.x) {
            // Find which cone this entry belongs to (binary search in sparse offsets)
            int64_t lo_k = 0, hi_k = numGenPowerCones;
            while (lo_k < hi_k) {
                int64_t mid = (lo_k + hi_k) / 2;
                if (d_genpow_sparse_offsets[mid + 1] <= tid) {
                    lo_k = mid + 1;
                } else {
                    hi_k = mid;
                }
            }
            double c3_val = genpow_sparse_c3[batch * numGenPowerCones + lo_k];
            vals[H_genpow_v3_col_idx[tid]] = -c3_val * genpow_sparse_left3[batch * totalGenpowDim + tid];
        }

        // 5. Expansion row du-column entries: -right1[i], -right2[i], -left3[i]
        for (int64_t tid = threadIdx.x; tid < totalGenpowDim; tid += blockDim.x) {
            vals[H_genpow_exp_v1_du_idx[tid]] = -genpow_sparse_right1[batch * totalGenpowDim + tid];
            vals[H_genpow_exp_v2_du_idx[tid]] = -genpow_sparse_right2[batch * totalGenpowDim + tid];
            vals[H_genpow_exp_v3_du_idx[tid]] = -genpow_sparse_left3[batch * totalGenpowDim + tid];
        }

        // 6. Expansion row diagonal entries: 1.0
        for (int64_t tid = threadIdx.x; tid < 3 * numGenPowerCones; tid += blockDim.x) {
            vals[H_genpow_exp_diag_idx[tid]] = 1.0;
        }
    }
}

// Direct-x GenPow rank-3 sparse value-write kernel.
//
// Reads the rank-3 decomposition stripes computed by
// `compute_xcone_genpow_H_kernel` and writes the corresponding values
// into the KKT slots reserved during sparsity construction. Mirrors the
// slack-side `populate_H_blocks_kernel` GenPow branch but indexed for
// direct-x rows (stat + direct-x rows for the rank-1 contributions).
//
// For each row i in [0, totalXGenPowDim) (one per row across all
// direct-x GenPow cones, prefixed by cone via d_dim_offsets):
//   - du-col diag at stat row J_xc[k]:        -diag[i]
//   - du-col diag at direct-x row xc_row+k:    1 - diag[i]
//   - exp-col 1 at stat row + dx row:         -left1[i]    (c1 coeff = 1)
//   - exp-col 2 at stat row + dx row:         -left2[i]    (c2 coeff = 1)
//   - exp-col 3 at stat row + dx row:         -c3[cone] * left3[i]
//   - exp-row 1 du-col entry:                 -right1[i]
//   - exp-row 2 du-col entry:                 -right2[i]
//   - exp-row 3 du-col entry:                 -left3[i]  (right3 ≡ left3)
// Plus self-diag = 1.0 at the 3 expansion-row diagonals per cone.
//
// One block per batch; threads stride over totalXGenPowDim entries +
// 3*numXGenPowerCones expansion-diag entries. The c3 lookup needs the
// cone index for entry i, found via binary search on d_dim_offsets.
__global__ void populate_xcone_genpow_rank3_kernel(
    double* __restrict__ kkt_values,
    const int64_t* __restrict__ H_xcone_genpow_du_stat_diag_idx,
    const int64_t* __restrict__ H_xcone_genpow_du_dx_diag_idx,
    const int64_t* __restrict__ H_xcone_genpow_v1_col_stat_idx,
    const int64_t* __restrict__ H_xcone_genpow_v2_col_stat_idx,
    const int64_t* __restrict__ H_xcone_genpow_v3_col_stat_idx,
    const int64_t* __restrict__ H_xcone_genpow_v1_col_dx_idx,
    const int64_t* __restrict__ H_xcone_genpow_v2_col_dx_idx,
    const int64_t* __restrict__ H_xcone_genpow_v3_col_dx_idx,
    const int64_t* __restrict__ H_xcone_genpow_exp_v1_du_idx,
    const int64_t* __restrict__ H_xcone_genpow_exp_v2_du_idx,
    const int64_t* __restrict__ H_xcone_genpow_exp_v3_du_idx,
    const int64_t* __restrict__ H_xcone_genpow_exp_diag_idx,
    const int64_t* __restrict__ d_xcone_genpow_dim_offsets,   // [numXGenPowerCones+1]
    const double* __restrict__ rank3_diag,
    const double* __restrict__ rank3_left1,
    const double* __restrict__ rank3_right1,
    const double* __restrict__ rank3_left2,
    const double* __restrict__ rank3_right2,
    const double* __restrict__ rank3_left3,
    const double* __restrict__ rank3_c3,
    int64_t numXGenPowerCones,
    int64_t totalXGenPowDim,
    int64_t nnzKKT,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;
    double* vals = kkt_values + batch * nnzKKT;

    const double* diag_b   = rank3_diag   + batch * totalXGenPowDim;
    const double* l1_b     = rank3_left1  + batch * totalXGenPowDim;
    const double* r1_b     = rank3_right1 + batch * totalXGenPowDim;
    const double* l2_b     = rank3_left2  + batch * totalXGenPowDim;
    const double* r2_b     = rank3_right2 + batch * totalXGenPowDim;
    const double* l3_b     = rank3_left3  + batch * totalXGenPowDim;
    const double* c3_b     = rank3_c3     + batch * numXGenPowerCones;

    for (int64_t i = threadIdx.x; i < totalXGenPowDim; i += blockDim.x) {
        // Find cone index via binary search on dim_offsets.
        int64_t lo = 0, hi = numXGenPowerCones;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if (d_xcone_genpow_dim_offsets[mid + 1] <= i) lo = mid + 1; else hi = mid;
        }
        int64_t cone = lo;
        double c3v = c3_b[cone];

        double d  = diag_b[i];
        double l1 = l1_b[i];
        double r1 = r1_b[i];
        double l2 = l2_b[i];
        double r2 = r2_b[i];
        double l3 = l3_b[i];

        vals[H_xcone_genpow_du_stat_diag_idx[i]] = -d;
        vals[H_xcone_genpow_du_dx_diag_idx[i]]   = 1.0 - d;
        vals[H_xcone_genpow_v1_col_stat_idx[i]]  = -l1;
        vals[H_xcone_genpow_v1_col_dx_idx[i]]    = -l1;
        vals[H_xcone_genpow_v2_col_stat_idx[i]]  = -l2;
        vals[H_xcone_genpow_v2_col_dx_idx[i]]    = -l2;
        vals[H_xcone_genpow_v3_col_stat_idx[i]]  = -c3v * l3;
        vals[H_xcone_genpow_v3_col_dx_idx[i]]    = -c3v * l3;
        vals[H_xcone_genpow_exp_v1_du_idx[i]]    = -r1;
        vals[H_xcone_genpow_exp_v2_du_idx[i]]    = -r2;
        vals[H_xcone_genpow_exp_v3_du_idx[i]]    = -l3;
    }
    // Expansion-row self-diagonals: 1.0 at 3*numXGenPowerCones slots.
    for (int64_t i = threadIdx.x; i < 3 * numXGenPowerCones; i += blockDim.x) {
        vals[H_xcone_genpow_exp_diag_idx[i]] = 1.0;
    }
}

// Direct-x SOC rank-2 sparse value-write kernel. Same shape as
// `populate_xcone_genpow_rank3_kernel` but with 2 ranks instead of 3.
__global__ void populate_xcone_soc_rank2_kernel(
    double* __restrict__ kkt_values,
    const int64_t* __restrict__ H_du_stat_diag_idx,
    const int64_t* __restrict__ H_du_dx_diag_idx,
    const int64_t* __restrict__ H_v1_col_stat_idx,
    const int64_t* __restrict__ H_v2_col_stat_idx,
    const int64_t* __restrict__ H_v1_col_dx_idx,
    const int64_t* __restrict__ H_v2_col_dx_idx,
    const int64_t* __restrict__ H_exp_v1_du_idx,
    const int64_t* __restrict__ H_exp_v2_du_idx,
    const int64_t* __restrict__ H_exp_diag_idx,
    const int64_t* __restrict__ d_dim_offsets,        // [numSparseXSoc+1]
    const double* __restrict__ sparse_diag,
    const double* __restrict__ sparse_v1,
    const double* __restrict__ sparse_v2,
    const double* __restrict__ sparse_c1,
    const double* __restrict__ sparse_c2,
    int64_t numSparseXSoc,
    int64_t totalSparseXSocDim,
    int64_t nnzKKT,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;
    double* vals = kkt_values + batch * nnzKKT;
    const double* diag_b = sparse_diag + batch * totalSparseXSocDim;
    const double* v1_b   = sparse_v1   + batch * totalSparseXSocDim;
    const double* v2_b   = sparse_v2   + batch * totalSparseXSocDim;
    const double* c1_b   = sparse_c1   + batch * numSparseXSoc;
    const double* c2_b   = sparse_c2   + batch * numSparseXSoc;

    for (int64_t i = threadIdx.x; i < totalSparseXSocDim; i += blockDim.x) {
        int64_t lo = 0, hi = numSparseXSoc;
        while (lo < hi) {
            int64_t mid = (lo + hi) / 2;
            if (d_dim_offsets[mid + 1] <= i) lo = mid + 1; else hi = mid;
        }
        int64_t cone = lo;
        double c1v = c1_b[cone];
        double c2v = c2_b[cone];
        double d  = diag_b[i];
        double v1 = v1_b[i];
        double v2 = v2_b[i];
        vals[H_du_stat_diag_idx[i]] = -d;
        vals[H_du_dx_diag_idx[i]]   = 1.0 - d;
        vals[H_v1_col_stat_idx[i]]  = -c1v * v1;
        vals[H_v1_col_dx_idx[i]]    = -c1v * v1;
        vals[H_v2_col_stat_idx[i]]  = -c2v * v2;
        vals[H_v2_col_dx_idx[i]]    = -c2v * v2;
        vals[H_exp_v1_du_idx[i]]    = -v1;
        vals[H_exp_v2_du_idx[i]]    = -v2;
    }
    for (int64_t i = threadIdx.x; i < 2 * numSparseXSoc; i += blockDim.x) {
        vals[H_exp_diag_idx[i]] = 1.0;
    }
}

// Direct-x cone value-write kernel.
//
// Writes the +1 / `−δ_{kl} + H_x[k, l]` / `−H_x[k, l]` entries into the
// KKT matrix slots reserved during sparsity construction. Each entry
// carries metadata (xc_idx, k, [l]); the kernel uses xcone_kinds to
// dispatch to the right H_x storage layout.
//
// One block per batch.
__global__ void populate_xcone_H_blocks_kernel(
    double* __restrict__ kkt_values,
    // E entries
    const int64_t* __restrict__ xcone_E_idx,
    const int64_t* __restrict__ E_meta_xc,
    const int64_t* __restrict__ E_meta_k,
    int64_t numE,
    // du entries (direct-x row × du_x col)
    const int64_t* __restrict__ xcone_du_idx,
    const int64_t* __restrict__ du_meta_xc,
    const int64_t* __restrict__ du_meta_k,
    const int64_t* __restrict__ du_meta_l,
    int64_t numDu,
    // stat entries (stationarity row × du_x col)
    const int64_t* __restrict__ xcone_stat_idx,
    const int64_t* __restrict__ stat_meta_xc,
    const int64_t* __restrict__ stat_meta_k,
    const int64_t* __restrict__ stat_meta_l,
    int64_t numStat,
    // Per-cone metadata
    const int64_t* __restrict__ xcone_kinds,
    const int64_t* __restrict__ xcone_dims,
    const int64_t* __restrict__ xcone_h_off,
    const int64_t* __restrict__ xcone_psd_k,
    // H_x storage per cone kind
    const double* __restrict__ xcone_nonneg_H,
    int64_t totalXNonneg,
    const double* __restrict__ xcone_soc_H,
    int64_t totalXSocKkt,
    const double* __restrict__ xcone_psd_H,
    int64_t totalXPsdKkt,
    const double* __restrict__ xcone_exp_H,
    int64_t totalXExpKkt,
    const double* __restrict__ xcone_pow_H,
    int64_t totalXPowKkt,
    const double* __restrict__ xcone_genpow_H,
    int64_t totalXGenPowKkt,
    int64_t nnzKKT,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double* vals = kkt_values + batch * nnzKKT;

    // E entries: +1.
    for (int64_t i = threadIdx.x; i < numE; i += blockDim.x) {
        vals[xcone_E_idx[i]] = 1.0;
    }

    // Helper to fetch H_x[k, l] for cone xc. Each kind has a dedicated
    // dense storage buffer. Asymmetric direct-x (Exp/Power/GenPow) reuses
    // the slack-form projection Jacobian formulas — the asymmetric primal/
    // dual swap matters only for the FORWARD scaling, not for the backward
    // projection Jacobian. See CPU `get_cone_derivative_sparse_xcones`.
    #define H_X_LOOKUP(xc, k, l)                                                   \
        ([&]() -> double {                                                         \
            int64_t _kind = xcone_kinds[(xc)];                                     \
            int64_t _dim  = xcone_dims[(xc)];                                      \
            int64_t _off  = xcone_h_off[(xc)];                                     \
            if (_kind == 0) {                                                      \
                return ((k) != (l)) ? 0.0                                          \
                    : xcone_nonneg_H[batch * totalXNonneg + _off + (k)];           \
            } else if (_kind == 1) {                                               \
                return xcone_soc_H[batch * totalXSocKkt + _off + (k) * _dim + (l)];\
            } else if (_kind == 2) {                                               \
                return xcone_psd_H[batch * totalXPsdKkt + _off + (k) * _dim + (l)];\
            } else if (_kind == 3) {                                               \
                return xcone_exp_H[batch * totalXExpKkt + _off + (k) * 3 + (l)];   \
            } else if (_kind == 4) {                                               \
                return xcone_pow_H[batch * totalXPowKkt + _off + (k) * 3 + (l)];   \
            } else {                                                               \
                return xcone_genpow_H[batch * totalXGenPowKkt + _off + (k) * _dim + (l)];\
            }                                                                      \
        }())

    // du entries: δ_{kl} − H_x[k, l] at the du_x col l × direct-x row k slot
    // (i.e. the (I − H_x) contribution to the F_4 row). Matches the CPU
    // IFT-direct sign convention in `build_hsde_augmented_system_sparse_full`
    // (kkt.rs). The opposite sign would flip the off-diagonal coupling in
    // the reduced stationarity for non-diagonal H_x (SOC, PSD).
    for (int64_t i = threadIdx.x; i < numDu; i += blockDim.x) {
        int64_t xc = du_meta_xc[i];
        int64_t k  = du_meta_k[i];
        int64_t l  = du_meta_l[i];
        double h   = H_X_LOOKUP(xc, k, l);
        double v   = (k == l) ? (1.0 - h) : (-h);
        vals[xcone_du_idx[i]] = v;
    }

    // stat entries: −H_x[k, l] at the du_x col l × stationarity row J_xc[k] slot.
    for (int64_t i = threadIdx.x; i < numStat; i += blockDim.x) {
        int64_t xc = stat_meta_xc[i];
        int64_t k  = stat_meta_k[i];
        int64_t l  = stat_meta_l[i];
        double h   = H_X_LOOKUP(xc, k, l);
        vals[xcone_stat_idx[i]] = -h;
    }
    #undef H_X_LOOKUP
}

void DiffKKT::updateJ(
    const double* P_values,
    int64_t nnzP_orig,
    const double* A_values,
    int64_t nnzA_orig,
    const double* q,
    const double* b,
    const double* c1,
    const double* c2,
    const double* c3,
    const double* nonneg_H,
    const double* soc_H,
    const double* soc_sparse_diag,
    const double* soc_sparse_v1,
    const double* soc_sparse_v2,
    const double* soc_sparse_c1,
    const double* soc_sparse_c2,
    const double* exp_H,
    const double* power_H,
    const double* psd_H,
    const double* genpow_sparse_diag,
    const double* genpow_sparse_left1,
    const double* genpow_sparse_right1,
    const double* genpow_sparse_left2,
    const double* genpow_sparse_right2,
    const double* genpow_sparse_left3,
    const double* genpow_sparse_c3,
    const double* xcone_nonneg_H,
    const double* xcone_soc_H,
    const double* xcone_psd_H,
    const double* xcone_exp_H,
    const double* xcone_pow_H,
    const double* xcone_genpow_H,
    const double* xcone_genpow_rank3_diag,
    const double* xcone_genpow_rank3_left1,
    const double* xcone_genpow_rank3_right1,
    const double* xcone_genpow_rank3_left2,
    const double* xcone_genpow_rank3_right2,
    const double* xcone_genpow_rank3_left3,
    const double* xcone_genpow_rank3_c3,
    const double* xcone_soc_rank2_diag,
    const double* xcone_soc_rank2_v1,
    const double* xcone_soc_rank2_v2,
    const double* xcone_soc_rank2_c1,
    const double* xcone_soc_rank2_c2,
    const Cones& cones,
    cudaStream_t stream
) {
    if (solverType_ == KKTSolverType::Riccati) {
        // Cache HSDE data needed for bordering in solveAdjoint
        cached_q_ = q;
        cached_b_ = b;
        cached_c1_ = c1;
        cached_c2_ = c2;
        cached_c3_ = c3;
        cached_nonneg_H_ = nonneg_H;
        updateJ_riccati(P_values, nnzP_orig, A_values, nnzA_orig,
                        nonneg_H, cones, stream);
        return;
    }
    // First initialize I and -εI diagonals and identity blocks within J
    MOREAU_KERNEL_LAUNCH(init_diff_kkt_values_kernel, batchSize, 256, 0, stream,
        KKT.values(),
        I_diag_idx_.get(),
        negI_diag_idx_.get(),
        J_I_row1_idx_.get(),
        J_negI_row1_idx_.get(),
        J_I_row2_idx_.get(),
        jdim,
        m,
        KKT.nnz(),
        batchSize,
        REG_DIFF
    );

    // Populate P, A, A', q, b, c1, c2, c3 values
    MOREAU_KERNEL_LAUNCH(populate_J_blocks_kernel, batchSize, 256, 0, stream,
        KKT.values(),
        P_idx_.get(), P_val_idx_.get(), P_values, nnzP_, nnzP_orig,
        A_idx_.get(), A_val_idx_.get(), A_values, nnzA_, nnzA_orig,
        At_idx_.get(), At_val_idx_.get(), nnzAt_,
        q_idx_.get(), q, n,
        b_idx_.get(), b, m,
        c1_idx_.get(), c1,
        c2_idx_.get(), c2,
        c3_idx_.get(), c3,
        KKT.nnz(),
        batchSize
    );

    // Populate H blocks (cone derivatives)
    MOREAU_KERNEL_LAUNCH(populate_H_blocks_kernel, batchSize, 256, 0, stream,
        KKT.values(),
        H_diag_idx_.get(),
        nonneg_H,
        // Dense SOC
        H_soc_idx_.get(),
        soc_H,
        d_soc_dims_.get(),
        d_soc_Hs_offsets_.get(),
        d_soc_kkt_offsets_.get(),
        numSocCones_,
        totalSocHsEntries_,
        totalSocKktEntries_,
        // Sparse SOC
        H_soc_sparse_diag_idx_.get(),
        H_soc_v1_col_idx_.get(),
        H_soc_v2_col_idx_.get(),
        H_soc_exp_v1_du_idx_.get(),
        H_soc_exp_v2_du_idx_.get(),
        H_soc_exp_diag_idx_.get(),
        soc_sparse_diag,
        soc_sparse_v1,
        soc_sparse_v2,
        soc_sparse_c1,
        soc_sparse_c2,
        d_soc_sparse_offsets_.get(),
        d_soc_sparse_indices_.get(),
        totalSparseSocDim_,
        numSparseSoc_,
        // Exp/Power
        H_exp_idx_.get(),
        exp_H,
        H_power_idx_.get(),
        power_H,
        // PSD cones
        H_psd_idx_.get(),
        psd_H,
        d_psd_Hs_offsets_.get(),
        d_psd_kkt_offsets_.get(),
        d_psd_svec_dims_.get(),
        numPsdCones_,
        totalPsdHsEntries_,
        totalPsdKktEntries_,
        // Sparse GenPowerCone
        H_genpow_sparse_diag_idx_.get(),
        H_genpow_v1_col_idx_.get(),
        H_genpow_v2_col_idx_.get(),
        H_genpow_v3_col_idx_.get(),
        H_genpow_exp_v1_du_idx_.get(),
        H_genpow_exp_v2_du_idx_.get(),
        H_genpow_exp_v3_du_idx_.get(),
        H_genpow_exp_diag_idx_.get(),
        genpow_sparse_diag,
        genpow_sparse_left1,
        genpow_sparse_right1,
        genpow_sparse_left2,
        genpow_sparse_right2,
        genpow_sparse_left3,
        genpow_sparse_c3,
        d_genpow_sparse_offsets_.get(),
        numGenPowerCones_,
        totalGenpowDim_,
        cones.numZeroCones,
        cones.numNonnegCones,
        cones.numExpCones,
        cones.numPowerCones,
        KKT.nnz(),
        batchSize
    );

    // Populate direct-x cone H blocks if any direct-x cones are present.
    // The caller passes direct-x H values via the new parameters of
    // updateJ; this kernel writes them at the index slots reserved
    // during sparsity construction.
    if (numXCones_ > 0 &&
        xcone_E_idx_.get() != nullptr &&
        xcone_du_idx_.get() != nullptr &&
        xcone_stat_idx_.get() != nullptr) {
        MOREAU_KERNEL_LAUNCH(populate_xcone_H_blocks_kernel, batchSize, 256, 0, stream,
            KKT.values(),
            xcone_E_idx_.get(),
            xcone_nonneg_E_idx_.get(),  // E_meta_xc
            xcone_nonneg_du_idx_.get(), // E_meta_k
            numXConeE_,
            xcone_du_idx_.get(),
            xcone_soc_E_idx_.get(),     // du_meta_xc
            xcone_soc_du_idx_.get(),    // du_meta_k
            xcone_soc_stat_idx_.get(),  // du_meta_l
            numXConeDu_,
            xcone_stat_idx_.get(),
            xcone_psd_E_idx_.get(),     // stat_meta_xc
            xcone_psd_du_idx_.get(),    // stat_meta_k
            xcone_stat_meta_l_.get(),   // stat_meta_l
            numXConeStat_,
            d_xcone_kinds_.get(),
            d_xcone_dims_.get(),
            d_xcone_psd_svec_offsets_.get(),  // h_off (per cone)
            d_xcone_psd_k_.get(),
            xcone_nonneg_H,
            totalXNonneg_,
            xcone_soc_H,
            total_x_soc_kkt_,
            xcone_psd_H,
            total_x_psd_kkt_,
            xcone_exp_H,
            totalXExpKkt_,
            xcone_pow_H,
            totalXPowKkt_,
            xcone_genpow_H,
            totalXGenPowKkt_,
            KKT.nnz(),
            batchSize
        );
    }

    // Direct-x SOC rank-2 sparse expansion: write rank-2 stripe values.
    if (numSparseXSoc_ > 0 &&
        H_xcone_soc_du_stat_diag_idx_.get() != nullptr &&
        xcone_soc_rank2_diag != nullptr) {
        MOREAU_KERNEL_LAUNCH(populate_xcone_soc_rank2_kernel, batchSize, 256, 0, stream,
            KKT.values(),
            H_xcone_soc_du_stat_diag_idx_.get(),
            H_xcone_soc_du_dx_diag_idx_.get(),
            H_xcone_soc_v1_col_stat_idx_.get(),
            H_xcone_soc_v2_col_stat_idx_.get(),
            H_xcone_soc_v1_col_dx_idx_.get(),
            H_xcone_soc_v2_col_dx_idx_.get(),
            H_xcone_soc_exp_v1_du_idx_.get(),
            H_xcone_soc_exp_v2_du_idx_.get(),
            H_xcone_soc_exp_diag_idx_.get(),
            d_xcone_soc_sparse_dim_offsets_.get(),
            xcone_soc_rank2_diag,
            xcone_soc_rank2_v1,
            xcone_soc_rank2_v2,
            xcone_soc_rank2_c1,
            xcone_soc_rank2_c2,
            numSparseXSoc_,
            totalSparseXSocDim_,
            KKT.nnz(),
            batchSize
        );
    }

    // Direct-x GenPow rank-3 sparse expansion: write rank-3 stripe values
    // into the dedicated KKT slots reserved during sparsity construction.
    if (numXGenPowerCones_ > 0 &&
        H_xcone_genpow_du_stat_diag_idx_.get() != nullptr &&
        xcone_genpow_rank3_diag != nullptr) {
        MOREAU_KERNEL_LAUNCH(populate_xcone_genpow_rank3_kernel, batchSize, 256, 0, stream,
            KKT.values(),
            H_xcone_genpow_du_stat_diag_idx_.get(),
            H_xcone_genpow_du_dx_diag_idx_.get(),
            H_xcone_genpow_v1_col_stat_idx_.get(),
            H_xcone_genpow_v2_col_stat_idx_.get(),
            H_xcone_genpow_v3_col_stat_idx_.get(),
            H_xcone_genpow_v1_col_dx_idx_.get(),
            H_xcone_genpow_v2_col_dx_idx_.get(),
            H_xcone_genpow_v3_col_dx_idx_.get(),
            H_xcone_genpow_exp_v1_du_idx_.get(),
            H_xcone_genpow_exp_v2_du_idx_.get(),
            H_xcone_genpow_exp_v3_du_idx_.get(),
            H_xcone_genpow_exp_diag_idx_.get(),
            d_xcone_genpow_dim_offsets_.get(),
            xcone_genpow_rank3_diag,
            xcone_genpow_rank3_left1,
            xcone_genpow_rank3_right1,
            xcone_genpow_rank3_left2,
            xcone_genpow_rank3_right2,
            xcone_genpow_rank3_left3,
            xcone_genpow_rank3_c3,
            numXGenPowerCones_,
            totalXGenPowDim_,
            KKT.nnz(),
            batchSize
        );
    }
}

void DiffKKT::factor(cudaStream_t stream) {
    if (solverType_ == KKTSolverType::Riccati) {
        factor_riccati(stream);
        return;
    }

    if (!cudss_initialized_) {
        throw std::runtime_error("cuDSS not initialized");
    }

    cudssStatus_t status = cudssExecute(
        cudss_handle_, CUDSS_PHASE_FACTORIZATION,
        cudss_config_, cudss_data_,
        kkt_matrix_, sol_matrix_, rhs_matrix_
    );

    if (status != CUDSS_STATUS_SUCCESS) {
        std::cerr << "DiffKKT factorization failed (cuDSS status: "
                  << cudss_status_string(status) << ")" << std::endl;
        std::cerr << "  KKT dim: " << augdim << "x" << augdim
                  << ", nnz: " << KKT.nnz()
                  << ", batch: " << batchSize << std::endl;
        std::cerr << gpu_memory_info() << std::endl;

        std::string msg = "DiffKKT factorization failed: cuDSS "
            + std::string(cudss_status_string(status))
            + " (KKT " + std::to_string(augdim) + "x" + std::to_string(augdim)
            + ", nnz=" + std::to_string(KKT.nnz())
            + ", batch=" + std::to_string(batchSize) + ")";
        if (status == CUDSS_STATUS_ALLOC_FAILED) {
            msg += ". Insufficient GPU memory for backward pass. "
                   "Try reducing batch_size or freeing GPU memory.";
        }
        throw std::runtime_error(msg);
    }
}

// Kernel to build augmented RHS [0; rhs_bar]
// rhs_bar is base_jdim-sized; we pad expansion vars with zeros
__global__ void build_aug_rhs_kernel(
    double* __restrict__ aug_rhs,
    const double* __restrict__ rhs_bar,
    int64_t base_jdim,
    int64_t jdim,
    int64_t augdim,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double* out = aug_rhs + batch * augdim;
    const double* in = rhs_bar + batch * base_jdim;

    // First jdim entries are 0
    for (int64_t i = threadIdx.x; i < jdim; i += blockDim.x) {
        out[i] = 0.0;
    }

    // Second half: first base_jdim entries from rhs_bar, rest zero (expansion vars)
    for (int64_t i = threadIdx.x; i < jdim; i += blockDim.x) {
        out[jdim + i] = (i < base_jdim) ? in[i] : 0.0;
    }
}

// Kernel to extract lam from augmented solution
// Only copies base_jdim entries (discards expansion var values)
__global__ void extract_lam_kernel(
    double* __restrict__ lam,
    const double* __restrict__ aug_sol,
    int64_t base_jdim,
    int64_t augdim,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    const double* sol = aug_sol + batch * augdim;
    double* out = lam + batch * base_jdim;

    // Extract first base_jdim entries (skip expansion variable values)
    for (int64_t i = threadIdx.x; i < base_jdim; i += blockDim.x) {
        out[i] = sol[i];
    }
}

void DiffKKT::solveAdjoint(
    const double* rhs_bar,
    double* lam,
    cudaStream_t stream
) {
    if (solverType_ == KKTSolverType::Riccati) {
        solveAdjoint_riccati(rhs_bar, lam, stream);
        return;
    }


    // Build augmented RHS [0; rhs_bar] with zero padding for expansion vars
    MOREAU_KERNEL_LAUNCH(build_aug_rhs_kernel, batchSize, 256, 0, stream,
        work_rhs_.data(),
        rhs_bar,
        base_jdim_, jdim, augdim, batchSize
    );

    {
        cudssMatrixSetValues(rhs_matrix_, work_rhs_.data());
        cudssMatrixSetValues(sol_matrix_, work_sol_.data());

        cudssStatus_t status = cudssExecute(
            cudss_handle_, CUDSS_PHASE_SOLVE,
            cudss_config_, cudss_data_,
            kkt_matrix_, sol_matrix_, rhs_matrix_
        );

        if (status != CUDSS_STATUS_SUCCESS) {
            std::cerr << "DiffKKT adjoint solve failed (cuDSS status: "
                      << cudss_status_string(status) << ")" << std::endl;
            std::cerr << "  KKT dim: " << augdim << "x" << augdim
                      << ", batch: " << batchSize << std::endl;
            std::cerr << gpu_memory_info() << std::endl;

            std::string msg = "DiffKKT adjoint solve failed: cuDSS "
                + std::string(cudss_status_string(status))
                + " (KKT " + std::to_string(augdim) + "x" + std::to_string(augdim)
                + ", batch=" + std::to_string(batchSize) + ")";
            if (status == CUDSS_STATUS_ALLOC_FAILED) {
                msg += ". Insufficient GPU memory. "
                       "Try reducing batch_size or freeing GPU memory.";
            }
            throw std::runtime_error(msg);
        }
    }

    // Extract lam from first half (only base_jdim entries, skip expansion vars)
    MOREAU_KERNEL_LAUNCH(extract_lam_kernel, batchSize, 256, 0, stream,
        lam,
        work_sol_.data(),
        base_jdim_, augdim, batchSize
    );
}

// Kernel to build forward augmented RHS [rhs; 0]
// rhs is base_jdim-sized; we pad expansion vars with zeros
__global__ void build_forward_aug_rhs_kernel(
    double* __restrict__ aug_rhs,
    const double* __restrict__ rhs,
    int64_t base_jdim,
    int64_t jdim,
    int64_t augdim,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    double* out = aug_rhs + batch * augdim;
    const double* in = rhs + batch * base_jdim;

    // First half: first base_jdim entries from rhs, rest zero (expansion vars)
    for (int64_t i = threadIdx.x; i < jdim; i += blockDim.x) {
        out[i] = (i < base_jdim) ? in[i] : 0.0;
    }

    // Last jdim entries are 0
    for (int64_t i = threadIdx.x; i < jdim; i += blockDim.x) {
        out[jdim + i] = 0.0;
    }
}

// Kernel to extract sol from augmented solution (second half for forward)
// Only copies base_jdim entries (discards expansion var values)
__global__ void extract_sol_kernel(
    double* __restrict__ sol,
    const double* __restrict__ aug_sol,
    int64_t base_jdim,
    int64_t jdim,
    int64_t augdim,
    int64_t batchSize
) {
    int64_t batch = blockIdx.x;
    if (batch >= batchSize) return;

    const double* full = aug_sol + batch * augdim;
    double* out = sol + batch * base_jdim;

    // For forward differentiation, we want the second half of the solution
    // Only copy base_jdim entries (skip expansion variable values)
    for (int64_t i = threadIdx.x; i < base_jdim; i += blockDim.x) {
        out[i] = full[jdim + i];
    }
}

void DiffKKT::solveForward(
    const double* rhs,
    double* sol,
    cudaStream_t stream
) {
    // Build augmented RHS [rhs; 0] with zero padding for expansion vars
    MOREAU_KERNEL_LAUNCH(build_forward_aug_rhs_kernel, batchSize, 256, 0, stream,
        work_rhs_.data(),
        rhs,
        base_jdim_, jdim, augdim, batchSize
    );

    // cuDSS solve
    cudssMatrixSetValues(rhs_matrix_, work_rhs_.data());
    cudssMatrixSetValues(sol_matrix_, work_sol_.data());

    cudssStatus_t status = cudssExecute(
        cudss_handle_, CUDSS_PHASE_SOLVE,
        cudss_config_, cudss_data_,
        kkt_matrix_, sol_matrix_, rhs_matrix_
    );

    if (status != CUDSS_STATUS_SUCCESS) {
        std::cerr << "DiffKKT forward solve failed (cuDSS status: "
                  << cudss_status_string(status) << ")" << std::endl;
        std::cerr << "  KKT dim: " << augdim << "x" << augdim
                  << ", batch: " << batchSize << std::endl;
        std::cerr << gpu_memory_info() << std::endl;

        std::string msg = "DiffKKT forward solve failed: cuDSS "
            + std::string(cudss_status_string(status))
            + " (KKT " + std::to_string(augdim) + "x" + std::to_string(augdim)
            + ", batch=" + std::to_string(batchSize) + ")";
        if (status == CUDSS_STATUS_ALLOC_FAILED) {
            msg += ". Insufficient GPU memory. "
                   "Try reducing batch_size or freeing GPU memory.";
        }
        throw std::runtime_error(msg);
    }

    // Extract sol from second half (only base_jdim entries)
    MOREAU_KERNEL_LAUNCH(extract_sol_kernel, batchSize, 256, 0, stream,
        sol,
        work_sol_.data(),
        base_jdim_, jdim, augdim, batchSize
    );
}

size_t DiffKKT::memoryUsage() const noexcept {
    size_t total = work_rhs_.memoryUsage() + work_sol_.memoryUsage() +
                   riccati_rhs_.memoryUsage() + riccati_bvec_.memoryUsage() +
                   riccati_sol0_.memoryUsage() + riccati_sol1_.memoryUsage();
    if (solverType_ == KKTSolverType::Riccati && diff_riccati_) {
        total += diff_riccati_->memoryUsage();
    } else {
        total += KKT.memoryUsage();
    }
    return total;
}

// ============================================================================
// Riccati backward pass implementation
// ============================================================================

// Solves (J̃'J̃ + εI)*lam = -rhs_bar using DiffRiccatiData.
// See diff_riccati.hpp for the mathematical derivation.

// ============================================================================
// GPU kernels for Riccati backward solve pipeline
// ============================================================================
// Form rhs0 (main) and rhs1 (bordering vector) in block order on GPU.
// One thread per (batch, variable) — variables are either x or w type.
// rhs0_x[α] = -r_x[α] - Σ_r A'[α,r]*D_u⁻¹[r]*r_u[r]  (in block order)
// rhs0_w[r] = -r_w[r] - (1+h[r])*D_u⁻¹[r]*r_u[r]       (in block order)
// rhs1_x[α] = (Pq)[α] - Σ_r A'[α,r]*D₁[r]*b[r]
// rhs1_w[r] = (Aq)[r] - b[r] + (1+h[r])*D_u⁻¹[r]*b[r]

// Kernel for x-part of rhs0 and rhs1
__global__ void backward_form_rhs_x_kernel(
    double* __restrict__ rhs0,       // [batch * nxw] in block order
    double* __restrict__ rhs1,       // [batch * nxw] in block order
    const double* __restrict__ rhs_bar, // [batch * jdim]
    const double* __restrict__ P_values,
    const double* __restrict__ A_values,
    const double* __restrict__ q,
    const double* __restrict__ b,
    const double* __restrict__ nonneg_H,
    const double* __restrict__ c1,
    const double* __restrict__ c3,
    const int64_t* __restrict__ P_ro,
    const int64_t* __restrict__ P_ci,
    const int32_t* __restrict__ A_csc_colptr,
    const int32_t* __restrict__ A_csc_rowidx,
    const int32_t* __restrict__ A_csc_to_csr,
    const int32_t* __restrict__ x_to_block,
    const int32_t* __restrict__ x_local_idx,
    const int32_t* __restrict__ block_offsets,
    int64_t n, int64_t m, int64_t nxw, int64_t jdim,
    int64_t nnzP, int64_t nnzA,
    int64_t numZeroCones, int64_t numNonnegCones,
    double eps, int64_t batchSize)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (tid >= batchSize * n) return;
    int64_t bat = tid / n;
    int64_t alpha = tid % n;

    auto get_h = [&](int64_t r) -> double {
        if (r < numZeroCones) return 1.0;
        if (r < numZeroCones + numNonnegCones)
            return nonneg_H[bat * numNonnegCones + (r - numZeroCones)];
        return 1.0;
    };

    int32_t bi = x_to_block[alpha];
    int32_t li = x_local_idx[alpha];
    int64_t out_idx = bat * nxw + block_offsets[bi] + li;
    int64_t a_base = bat * nnzA;
    int64_t p_base = bat * nnzP;

    // rhs0: f_x[α] = -r_x[α] - Σ_r A[r,α]*D_u⁻¹[r]*r_u[r]
    double f = -rhs_bar[bat * jdim + alpha];
    int32_t col_start = A_csc_colptr[alpha];
    int32_t col_end = A_csc_colptr[alpha + 1];
    for (int32_t p = col_start; p < col_end; ++p) {
        int32_t r = A_csc_rowidx[p];
        double h = get_h(r);
        double du_inv = 1.0 / (1.0 + h + eps);
        f -= A_values[a_base + A_csc_to_csr[p]] * du_inv * rhs_bar[bat * jdim + n + m + r];
    }
    rhs0[out_idx] = f;

    // rhs1: bvec_x[α] = (Pq)[α] - Σ_r A[r,α]*D₁[r]*b[r]
    double bv = 0.0;
    // Pq contribution
    for (int64_t pp = P_ro[alpha]; pp < P_ro[alpha + 1]; ++pp)
        bv += P_values[p_base + pp] * q[bat * n + P_ci[pp]];
    // -A'*diag(D₁)*b contribution
    for (int32_t p = col_start; p < col_end; ++p) {
        int32_t r = A_csc_rowidx[p];
        double h = get_h(r);
        double d1 = (h + eps) / (1.0 + h + eps);
        bv -= A_values[a_base + A_csc_to_csr[p]] * d1 * b[bat * m + r];
    }
    // gap (tau) row cross term: J'J[x,tau] includes c1*c3. Zero when x=0
    // (i.e. q=b=0), which is why q=0 tests never exercised it.
    bv += c1[bat * n + alpha] * c3[bat];
    rhs1[out_idx] = bv;
}

// Kernel for w-part of rhs0 and rhs1
__global__ void backward_form_rhs_w_kernel(
    double* __restrict__ rhs0,
    double* __restrict__ rhs1,
    const double* __restrict__ rhs_bar,
    const double* __restrict__ A_values,
    const double* __restrict__ q,
    const double* __restrict__ b,
    const double* __restrict__ nonneg_H,
    const double* __restrict__ c2,
    const double* __restrict__ c3,
    const int64_t* __restrict__ A_ro,
    const int64_t* __restrict__ A_ci,
    const int32_t* __restrict__ w_to_block,
    const int32_t* __restrict__ w_local_idx,
    const int32_t* __restrict__ block_offsets,
    int64_t n, int64_t m, int64_t nxw, int64_t jdim,
    int64_t nnzA,
    int64_t numZeroCones, int64_t numNonnegCones,
    double eps, int64_t batchSize)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (tid >= batchSize * m) return;
    int64_t bat = tid / m;
    int64_t r = tid % m;

    auto get_h = [&](int64_t ri) -> double {
        if (ri < numZeroCones) return 1.0;
        if (ri < numZeroCones + numNonnegCones)
            return nonneg_H[bat * numNonnegCones + (ri - numZeroCones)];
        return 1.0;
    };

    double h = get_h(r);
    double du_inv = 1.0 / (1.0 + h + eps);
    int32_t bi = w_to_block[r];
    int32_t li = w_local_idx[r];
    int64_t out_idx = bat * nxw + block_offsets[bi] + li;
    int64_t a_base = bat * nnzA;

    // rhs0: f_w[r] = -r_w[r] - (1+h)*D_u⁻¹*r_u[r]
    rhs0[out_idx] = -rhs_bar[bat * jdim + n + r]
                    - (1.0 + h) * du_inv * rhs_bar[bat * jdim + n + m + r];

    // rhs1: bvec_w[r] = (Aq)[r] - b[r] + (1+h)*D_u⁻¹*b[r]
    double aq = 0.0;
    for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p)
        aq += A_values[a_base + p] * q[bat * n + A_ci[p]];
    // gap (tau) row cross term: J'J[w,tau] includes c2*c3 (zero when q=b=0).
    rhs1[out_idx] = aq + c2[bat * m + r] * c3[bat]
                    - b[bat * m + r] + (1.0 + h) * du_inv * b[bat * m + r];
}

// Kernel for c_vec in block order (τ-row of J projected to (x,w) space)
__global__ void backward_form_cvec_kernel(
    double* __restrict__ cvec,       // [batch * nxw] in block order
    const double* __restrict__ c1,   // [batch * n]
    const double* __restrict__ c2,   // [batch * m]
    const int32_t* __restrict__ x_to_block,
    const int32_t* __restrict__ x_local_idx,
    const int32_t* __restrict__ w_to_block,
    const int32_t* __restrict__ w_local_idx,
    const int32_t* __restrict__ block_offsets,
    int64_t n, int64_t m, int64_t nxw,
    int64_t batchSize)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t total = batchSize * (n + m);
    if (tid >= total) return;
    int64_t bat = tid / (n + m);
    int64_t var = tid % (n + m);

    double* out = cvec + bat * nxw;
    if (var < n) {
        int32_t bi = x_to_block[var];
        int32_t li = x_local_idx[var];
        out[block_offsets[bi] + li] = c1[bat * n + var];
    } else {
        int64_t r = var - n;
        int32_t bi = w_to_block[r];
        int32_t li = w_local_idx[r];
        out[block_offsets[bi] + li] = c2[bat * m + r];
    }
}

// Kernel 1: Sherman-Morrison + bordering. 256 threads per batch for reductions.
// Outputs per-batch scalars: scalars[bat*2] = lam_tau, scalars[bat*2+1] = beta_sm
__global__ void backward_sm_bordering_kernel(
    double* __restrict__ scalars,    // [batch * 2] output: lam_tau, beta_sm
    const double* __restrict__ sol0,  // [batch * nxw] M⁻¹*rhs0 (block order)
    const double* __restrict__ sol1,  // [batch * nxw] M⁻¹*rhs1 (block order)
    const double* __restrict__ solc,  // [batch * nxw] M⁻¹*cvec (block order)
    const double* __restrict__ bvec,  // [batch * nxw] bordering vector (block order)
    const double* __restrict__ cvec,  // [batch * nxw] cvec (block order)
    const double* __restrict__ rhs_bar, // [batch * jdim]
    const double* __restrict__ q,
    const double* __restrict__ b_vec,
    const double* __restrict__ c3,
    int64_t n, int64_t m, int64_t nxw, int64_t jdim,
    double eps, int64_t batchSize)
{
    int64_t bat = blockIdx.x;
    if (bat >= batchSize) return;

    const double* s0 = sol0 + bat * nxw;
    const double* s1 = sol1 + bat * nxw;
    const double* sc = solc + bat * nxw;
    const double* cv = cvec + bat * nxw;
    const double* r1 = bvec + bat * nxw;

    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Phase 1: SM dot products (c_dot_vc, c_dot_v0, c_dot_v1)
    double pvc = 0, pv0 = 0, pv1 = 0;
    for (int64_t i = tid; i < nxw; i += nthreads) {
        double c = cv[i], v0 = s0[i], vc = sc[i];
        pvc += c * vc;
        pv0 += c * v0;
        pv1 += c * s1[i];
    }
    sdata[tid] = pvc;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    double c_dot_vc = sdata[0];

    sdata[tid] = pv0;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    double c_dot_v0 = sdata[0];

    sdata[tid] = pv1;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    double c_dot_v1 = sdata[0];

    double sigma = 1.0 + c_dot_vc;
    double alpha0 = c_dot_v0 / sigma;
    double alpha1 = c_dot_v1 / sigma;

    // Phase 2: d_ττ = q'q + b'b + c3² + ε
    const double* q_b = q + bat * n;
    const double* b_b = b_vec + bat * m;
    double pdt = 0;
    for (int64_t i = tid; i < n; i += nthreads) pdt += q_b[i] * q_b[i];
    for (int64_t i = tid; i < m; i += nthreads) pdt += b_b[i] * b_b[i];
    sdata[tid] = pdt;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    double d_tt = sdata[0] + c3[bat] * c3[bat] + eps;

    // Phase 3: bordering dot products (b_dot_w0, b_dot_w1)
    double pw0 = 0, pw1 = 0;
    for (int64_t i = tid; i < nxw; i += nthreads) {
        double w0 = s0[i] - alpha0 * sc[i];
        double w1 = s1[i] - alpha1 * sc[i];
        pw0 += r1[i] * w0;
        pw1 += r1[i] * w1;
    }
    sdata[tid] = pw0;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    double b_dot_w0 = sdata[0];

    sdata[tid] = pw1;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    double b_dot_w1 = sdata[0];

    if (tid == 0) {
        double denom_tau = d_tt - b_dot_w1;
        double lam_tau = 0.0;
        if (denom_tau > 1e-15 || denom_tau < -1e-15)
            lam_tau = (-rhs_bar[bat * jdim + jdim - 1] - b_dot_w0) / denom_tau;
        double beta_sm = alpha0 - lam_tau * alpha1;
        scalars[bat * 2] = lam_tau;
        scalars[bat * 2 + 1] = beta_sm;
    }
}

// Kernel 2: Unpack lam_x, lam_w from block order to global. One thread per (batch, var).
__global__ void backward_unpack_lam_kernel(
    double* __restrict__ lam_out,    // [batch * jdim]
    const double* __restrict__ sol0,
    const double* __restrict__ sol1,
    const double* __restrict__ solc,
    const double* __restrict__ scalars,
    const int32_t* __restrict__ x_to_block,
    const int32_t* __restrict__ x_local_idx,
    const int32_t* __restrict__ w_to_block,
    const int32_t* __restrict__ w_local_idx,
    const int32_t* __restrict__ block_offsets,
    int64_t n, int64_t m, int64_t nxw, int64_t jdim,
    int64_t batchSize)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (tid >= batchSize * (n + m)) return;

    int64_t bat = tid / (n + m);
    int64_t var = tid % (n + m);

    double lam_tau = scalars[bat * 2];
    double beta_sm = scalars[bat * 2 + 1];
    const double* s0 = sol0 + bat * nxw;
    const double* s1 = sol1 + bat * nxw;
    const double* sc = solc + bat * nxw;

    if (var < n) {
        int32_t bi = x_to_block[var];
        int32_t li = x_local_idx[var];
        int64_t idx = block_offsets[bi] + li;
        lam_out[bat * jdim + var] = s0[idx] - lam_tau * s1[idx] - beta_sm * sc[idx];
    } else {
        int64_t r = var - n;
        int32_t bi = w_to_block[r];
        int32_t li = w_local_idx[r];
        int64_t idx = block_offsets[bi] + li;
        lam_out[bat * jdim + n + r] = s0[idx] - lam_tau * s1[idx] - beta_sm * sc[idx];
    }
}

// Kernel 3: Recover lam_u and set lam_tau. One thread per (batch, constraint).
__global__ void backward_recover_lam_u_kernel(
    double* __restrict__ lam,
    const double* __restrict__ scalars,
    const double* __restrict__ rhs_bar,
    const double* __restrict__ A_values,
    const double* __restrict__ b_vec,
    const double* __restrict__ nonneg_H,
    const int64_t* __restrict__ A_ro,
    const int64_t* __restrict__ A_ci,
    int64_t n, int64_t m, int64_t jdim,
    int64_t nnzA,
    int64_t numZeroCones, int64_t numNonnegCones,
    double eps, int64_t batchSize)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (tid >= batchSize * m) return;

    int64_t bat = tid / m;
    int64_t r = tid % m;

    double lam_tau = scalars[bat * 2];
    const double* A_val = A_values + bat * nnzA;
    double* lam_b = lam + bat * jdim;

    double ax = 0.0;
    for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p)
        ax += A_val[p] * lam_b[A_ci[p]];

    double h;
    if (r < numZeroCones) h = 1.0;
    else if (r < numZeroCones + numNonnegCones)
        h = nonneg_H[bat * numNonnegCones + (r - numZeroCones)];
    else h = 1.0;

    double muu = 1.0 + h * h + eps;
    lam_b[n + m + r] = (-rhs_bar[bat * jdim + n + m + r] + ax
                        + (1.0 + h) * lam_b[n + r]
                        - b_vec[bat * m + r] * lam_tau) / muu;

    if (r == 0) lam_b[jdim - 1] = lam_tau;
}

// Kernel 4: Compute y = -J*lam. Reads lam from lam_buf, writes y to y_out.
// Two separate buffers so no in-place conflict.
// One thread per (batch, output_index), output_index ∈ [0, jdim).
__global__ void backward_compute_y_kernel(
    double* __restrict__ y_out,
    const double* __restrict__ lam_buf,
    const double* __restrict__ P_values,
    const double* __restrict__ A_values,
    const double* __restrict__ q,
    const double* __restrict__ b_vec,
    const double* __restrict__ c1,
    const double* __restrict__ c2,
    const double* __restrict__ c3,
    const double* __restrict__ nonneg_H,
    const int64_t* __restrict__ P_ro,
    const int64_t* __restrict__ P_ci,
    const int64_t* __restrict__ A_ro,
    const int64_t* __restrict__ A_ci,
    const int32_t* __restrict__ A_csc_colptr,
    const int32_t* __restrict__ A_csc_rowidx,
    const int32_t* __restrict__ A_csc_to_csr,
    int64_t n, int64_t m, int64_t jdim,
    int64_t nnzP, int64_t nnzA,
    int64_t numZeroCones, int64_t numNonnegCones,
    int64_t batchSize)
{
    int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (tid >= batchSize * jdim) return;

    int64_t bat = tid / jdim;
    int64_t k = tid % jdim;

    const double* lam = lam_buf + bat * jdim;
    const double* A_val = A_values + bat * nnzA;
    const double* P_val = P_values + bat * nnzP;
    double lam_tau = lam[jdim - 1];

    if (k < n) {
        // y_x[α] = -(P*lam_x + A'*lam_w + q*lam_τ)
        int64_t alpha = k;
        double val = 0.0;
        for (int64_t p = P_ro[alpha]; p < P_ro[alpha + 1]; ++p)
            val += P_val[p] * lam[P_ci[p]];
        int32_t col_start = A_csc_colptr[alpha];
        int32_t col_end = A_csc_colptr[alpha + 1];
        for (int32_t p = col_start; p < col_end; ++p)
            val += A_val[A_csc_to_csr[p]] * lam[n + A_csc_rowidx[p]];
        val += q[bat * n + alpha] * lam_tau;
        y_out[bat * jdim + k] = -val;

    } else if (k < n + m) {
        // y_w[r] = -(A*lam_x + lam_w - lam_u - b*lam_τ)
        int64_t r = k - n;
        double ax = 0.0;
        for (int64_t p = A_ro[r]; p < A_ro[r + 1]; ++p)
            ax += A_val[p] * lam[A_ci[p]];
        y_out[bat * jdim + k] = -(ax + lam[n + r] - lam[n + m + r]
                                   - b_vec[bat * m + r] * lam_tau);

    } else if (k < n + 2 * m) {
        // y_u[i] = -(lam_w - H*lam_u)
        int64_t i = k - n - m;
        double h;
        if (i < numZeroCones) h = 1.0;
        else if (i < numZeroCones + numNonnegCones)
            h = nonneg_H[bat * numNonnegCones + (i - numZeroCones)];
        else h = 1.0;
        y_out[bat * jdim + k] = -(lam[n + i] - h * lam[n + m + i]);

    } else {
        // y_τ = -(c1'*lam_x + c2'*lam_w + c3*lam_τ)
        double val = 0.0;
        for (int64_t i = 0; i < n; ++i) val += c1[bat * n + i] * lam[i];
        for (int64_t i = 0; i < m; ++i) val += c2[bat * m + i] * lam[n + i];
        val += c3[bat] * lam_tau;
        y_out[bat * jdim + k] = -val;
    }
}


void DiffKKT::initialize_riccati(
    int64_t n_, int64_t m_, int64_t batch_,
    const int64_t* P_ro, const int64_t* P_ci, int64_t nnzP,
    const int64_t* A_ro, const int64_t* A_ci, int64_t nnzA,
    const Cones& cones,
    cudaStream_t stream)
{
    // Detect block-tridiagonal structure (same as forward), allowing the same
    // bandwidth-reducing (RCM) column reorder. detect_block_tridiagonal is a
    // pure function of the sparsity, so this reproduces exactly the perm the
    // forward factory chose; DiffRiccatiData bakes it into its (x,w) block maps.
    std::vector<int32_t> perm;
    auto block_sizes = detect_block_tridiagonal(
        n_, m_, P_ro, P_ci, nnzP, A_ro, A_ci, nnzA, cones,
        /*allow_permute=*/true, &perm);

    if (block_sizes.empty()) {
        throw std::runtime_error(
            "DiffKKT: Riccati requested but block-tridiagonal structure not detected. "
            "This should not happen since the forward solver already validated it.");
    }

    // Create DiffRiccatiData which operates in (x,w) combined space
    diff_riccati_ = std::make_unique<DiffRiccatiData>(
        n_, m_, batch_,
        P_ro, P_ci, nnzP,
        A_ro, A_ci, nnzA,
        cones, block_sizes, perm, stream);

    numZeroCones_ = cones.numZeroCones;
    numNonnegCones_ = cones.numNonnegCones;
}

void DiffKKT::updateJ_riccati(
    const double* P_values, int64_t /*nnzP_orig*/,
    const double* A_values, int64_t /*nnzA_orig*/,
    const double* nonneg_H, const Cones& /*cones*/,
    cudaStream_t stream)
{
    diff_riccati_->populate(P_values, A_values, stream);
}

void DiffKKT::factor_riccati(cudaStream_t stream) {
    bool success = diff_riccati_->assemble_and_factorize(cached_nonneg_H_, stream);
    if (!success) {
        throw std::runtime_error(
            "DiffKKT Riccati factorization failed (J'J approach). "
            "The block-tridiagonal Cholesky factorization is not positive definite.");
    }
}

void DiffKKT::solveAdjoint_riccati(
    const double* rhs_bar, double* lam, cudaStream_t stream)
{
    int64_t nxw = n + m;
    double eps = diff_riccati_->reg_eps_;
    int threads = 256;

    // Buffer allocation:
    //   d_lhs  → rhs0, then sol0, then solc
    //   d_lhs2 → rhs1, then sol1 (preserved)
    //   riccati_rhs_  → cvec (preserved)
    //   riccati_bvec_ → bvec (copy of rhs1 before solve)
    //   riccati_sol0_ → copy of sol0 (preserved after solc overwrites d_lhs)
    //   riccati_sol1_ → temp lam buffer for y=-J*lam
    //   riccati_scalars_ → [2] per batch: lam_tau, beta_sm

    double* d_rhs0 = diff_riccati_->d_lhs.get();
    double* d_rhs1 = diff_riccati_->d_lhs2.get();
    double* d_cvec = riccati_rhs_.data();
    double* d_bvec = riccati_bvec_.data();
    double* d_sol0 = riccati_sol0_.data();
    double* d_lam_tmp = riccati_sol1_.data();  // [batch * nxw] reused for [batch * jdim]
    double* d_scalars = riccati_scalars_.data();

    // Step 1: Form rhs0 and rhs1 in block order
    {
        int64_t total_x = batchSize * n;
        MOREAU_KERNEL_LAUNCH(backward_form_rhs_x_kernel, (int)((total_x + threads - 1) / threads), threads, 0, stream,
            d_rhs0, d_rhs1, rhs_bar,
            diff_riccati_->d_P_values.get(), diff_riccati_->d_A_values.get(),
            cached_q_, cached_b_, cached_nonneg_H_, cached_c1_, cached_c3_,
            diff_riccati_->d_P_ro.get(), diff_riccati_->d_P_ci.get(),
            diff_riccati_->d_A_csc_colptr.get(),
            diff_riccati_->d_A_csc_rowidx.get(),
            diff_riccati_->d_A_csc_to_csr.get(),
            diff_riccati_->d_x_to_block.get(),
            diff_riccati_->d_x_local_idx.get(),
            diff_riccati_->d_block_offsets.get(),
            n, m, nxw, jdim,
            diff_riccati_->nnzP, diff_riccati_->nnzA,
            numZeroCones_, numNonnegCones_,
            eps, batchSize);

        int64_t total_w = batchSize * m;
        MOREAU_KERNEL_LAUNCH(backward_form_rhs_w_kernel, (int)((total_w + threads - 1) / threads), threads, 0, stream,
            d_rhs0, d_rhs1, rhs_bar,
            diff_riccati_->d_A_values.get(),
            cached_q_, cached_b_, cached_nonneg_H_, cached_c2_, cached_c3_,
            diff_riccati_->d_A_ro.get(), diff_riccati_->d_A_ci.get(),
            diff_riccati_->d_w_to_block.get(),
            diff_riccati_->d_w_local_idx.get(),
            diff_riccati_->d_block_offsets.get(),
            n, m, nxw, jdim, diff_riccati_->nnzA,
            numZeroCones_, numNonnegCones_,
            eps, batchSize);
    }

    // Step 2: Form c_vec in block order
    {
        int64_t total_cv = batchSize * (n + m);
        MOREAU_KERNEL_LAUNCH(backward_form_cvec_kernel, (int)((total_cv + threads - 1) / threads), threads, 0, stream,
            d_cvec, cached_c1_, cached_c2_,
            diff_riccati_->d_x_to_block.get(), diff_riccati_->d_x_local_idx.get(),
            diff_riccati_->d_w_to_block.get(), diff_riccati_->d_w_local_idx.get(),
            diff_riccati_->d_block_offsets.get(),
            n, m, nxw, batchSize);
    }

    // Step 3: Save rhs1 (bordering vector) BEFORE solve2 overwrites it
    CUDA_THROW(cudaMemcpyAsync(d_bvec, d_rhs1,
                   sizeof(double) * batchSize * nxw, cudaMemcpyDeviceToDevice, stream));

    // Step 4: Solve 2 RHS simultaneously
    diff_riccati_->solve2(d_rhs0, d_rhs1, stream);
    // d_rhs0 = sol0, d_rhs1 = sol1

    // Step 5: Save sol0, copy cvec to d_lhs, solve 3rd RHS
    CUDA_THROW(cudaMemcpyAsync(d_sol0, d_rhs0,
                   sizeof(double) * batchSize * nxw, cudaMemcpyDeviceToDevice, stream));
    CUDA_THROW(cudaMemcpyAsync(d_rhs0, d_cvec,
                   sizeof(double) * batchSize * nxw, cudaMemcpyDeviceToDevice, stream));
    diff_riccati_->solve(d_rhs0, stream);
    // d_rhs0 = solc

    // Step 6: Sherman-Morrison + bordering (256 threads per batch, parallel reductions)
    int smem = threads * sizeof(double);
    MOREAU_KERNEL_LAUNCH(backward_sm_bordering_kernel, batchSize, threads, smem, stream,
        d_scalars, d_sol0, d_rhs1, d_rhs0, d_bvec, d_cvec,
        rhs_bar, cached_q_, cached_b_, cached_c3_,
        n, m, nxw, jdim, eps, batchSize);

    // Step 7: Unpack lam from block order to global order
    // Use `lam` (output) as temp buffer for lam values.
    // We'll overwrite with y=-J*lam in step 9.
    {
        int64_t total = batchSize * (n + m);
        MOREAU_KERNEL_LAUNCH(backward_unpack_lam_kernel, (int)((total + threads - 1) / threads), threads, 0, stream,
            lam, d_sol0, d_rhs1, d_rhs0, d_scalars,
            diff_riccati_->d_x_to_block.get(), diff_riccati_->d_x_local_idx.get(),
            diff_riccati_->d_w_to_block.get(), diff_riccati_->d_w_local_idx.get(),
            diff_riccati_->d_block_offsets.get(),
            n, m, nxw, jdim, batchSize);
    }

    // Step 8: Recover lam_u
    {
        int64_t total = batchSize * m;
        MOREAU_KERNEL_LAUNCH(backward_recover_lam_u_kernel, (int)((total + threads - 1) / threads), threads, 0, stream,
            lam, d_scalars, rhs_bar,
            diff_riccati_->d_A_values.get(), cached_b_, cached_nonneg_H_,
            diff_riccati_->d_A_ro.get(), diff_riccati_->d_A_ci.get(),
            n, m, jdim, diff_riccati_->nnzA,
            numZeroCones_, numNonnegCones_,
            eps, batchSize);
    }

    // Step 9: compute y = -J*lam; copy lam to temp to avoid read-write hazard.
    CUDA_THROW(cudaMemcpyAsync(work_rhs_.data(), lam,
                   sizeof(double) * batchSize * jdim, cudaMemcpyDeviceToDevice, stream));

    {
        int64_t total = batchSize * jdim;
        MOREAU_KERNEL_LAUNCH(backward_compute_y_kernel, (int)((total + threads - 1) / threads), threads, 0, stream,
            lam,                    // output: y
            work_rhs_.data(),       // input: lam (copy)
            diff_riccati_->d_P_values.get(), diff_riccati_->d_A_values.get(),
            cached_q_, cached_b_,
            cached_c1_, cached_c2_, cached_c3_,
            cached_nonneg_H_,
            diff_riccati_->d_P_ro.get(), diff_riccati_->d_P_ci.get(),
            diff_riccati_->d_A_ro.get(), diff_riccati_->d_A_ci.get(),
            diff_riccati_->d_A_csc_colptr.get(),
            diff_riccati_->d_A_csc_rowidx.get(),
            diff_riccati_->d_A_csc_to_csr.get(),
            n, m, jdim,
            diff_riccati_->nnzP, diff_riccati_->nnzA,
            numZeroCones_, numNonnegCones_,
            batchSize);
    }
}

} // namespace moreau
