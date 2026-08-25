/**
 * @file chordal_info.cpp
 * @brief Host-side chordal decomposition implementation
 *
 * Implements ChordalInfo methods declared in chordal_info.hpp.
 * This is a C++ port of the Rust chordal decomposition code.
 */

#include "moreau/chordal/chordal_info.hpp"
#include "moreau/chordal/sparsity_pattern.hpp"
#include "moreau/chordal/symbolic_ldl.hpp"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace moreau {

// ============================================================================
// Local helpers
// ============================================================================

static int64_t triangular_number(int64_t n) { return n * (n + 1) / 2; }

// ============================================================================
// Helper: coord_to_upper_triangular_index (Rust version)
//
// Given (row, col) in the upper triangle, return the linear index
// into the packed upper triangle stored column-major:
//   (0,0), (0,1), (1,1), (0,2), (1,2), (2,2), ...
// ============================================================================

static int64_t coord_to_tri(int64_t row, int64_t col) {
    // Ensure row <= col (upper triangle)
    if (row > col) std::swap(row, col);
    return col * (col + 1) / 2 + row;
}

// ============================================================================
// find_aggregate_sparsity_mask
// ============================================================================

std::vector<bool> ChordalInfo::find_aggregate_sparsity_mask(
    const int64_t* A_colptr, const int64_t* A_rowind,
    int64_t A_ncol, int64_t A_nrow,
    const bool* b_sparsity_pattern)
{
    std::vector<bool> active(A_nrow, false);

    // Mark all rows that have nonzero entries in A
    int64_t nnz = A_colptr[A_ncol];
    for (int64_t k = 0; k < nnz; ++k) {
        active[A_rowind[k]] = true;
    }

    if (b_sparsity_pattern) {
        // Use provided b sparsity pattern
        for (int64_t i = 0; i < A_nrow; ++i) {
            if (b_sparsity_pattern[i]) {
                active[i] = true;
            }
        }
    } else {
        // Conservative fallback: assume nonzero b where A row is all-zero
        for (int64_t i = 0; i < A_nrow; ++i) {
            if (!active[i]) {
                active[i] = true;
            }
        }
    }

    return active;
}

// ============================================================================
// analyse_psd_sparsity — analyze one PSD cone
// ============================================================================

void ChordalInfo::analyse_psd_sparsity(
    std::vector<bool>& nz_mask, int64_t conedim, int64_t coneidx)
{
    int64_t tri_len = conedim * (conedim + 1) / 2;

    // Force diagonal entries to be marked
    for (int64_t i = 0; i < conedim; ++i) {
        int64_t diag_idx = i * (i + 3) / 2;  // triangular_index(i)
        if (diag_idx < tri_len) {
            nz_mask[diag_idx] = true;
        }
    }

    // Check if fully dense — no decomposition possible
    bool all_nonzero = true;
    for (int64_t i = 0; i < tri_len; ++i) {
        if (!nz_mask[i]) { all_nonzero = false; break; }
    }
    if (all_nonzero) return;

    // Build upper-triangular CSC from mask
    std::vector<int64_t> rows, cols;
    for (int64_t col = 0; col < conedim; ++col) {
        for (int64_t row = 0; row <= col; ++row) {
            int64_t idx = col * (col + 1) / 2 + row;
            if (nz_mask[idx]) {
                rows.push_back(row);
                cols.push_back(col);
            }
        }
    }

    // Convert COO to CSC
    int64_t nnz = static_cast<int64_t>(rows.size());
    std::vector<int64_t> colptr(conedim + 1, 0);
    for (int64_t k = 0; k < nnz; ++k) {
        colptr[cols[k] + 1]++;
    }
    for (int64_t j = 0; j < conedim; ++j) {
        colptr[j + 1] += colptr[j];
    }
    std::vector<int64_t> rowind(nnz);
    std::vector<int64_t> pos(colptr.begin(), colptr.begin() + conedim);
    for (int64_t k = 0; k < nnz; ++k) {
        rowind[pos[cols[k]]++] = rows[k];
    }

    // Symbolic LDL factorization
    chordal::SymbolicLDLResult ldl = chordal::symbolic_ldl(colptr.data(), rowind.data(), conedim);

    if (ldl.n <= 1) return;

    // Build SuperNodeTree from L factor
    SuperNodeTree sntree(ldl.L_colptr.data(), ldl.L_rowind.data(), ldl.n);

    // Apply parent-child merge strategy (default)
    // This is a simplified version: merge child into parent when the
    // merged clique would be no larger than the sum of components.
    if (sntree.n_cliques > 1) {
        // Parent-child merge: process in post-order, merge if fill-in is zero
        bool merged = true;
        while (merged && sntree.n_cliques > 1) {
            merged = false;
            // Rebuild post-order after each merge pass
            sntree.snode_post.resize(sntree.snode_parent.size());
            SuperNodeTree::post_order(sntree.snode_post, sntree.snode_parent,
                                       sntree.snode_children,
                                       sntree.n_cliques);

            for (int64_t i = 0; i < sntree.n_cliques; ++i) {
                int64_t c = sntree.get_post_order(i);
                int64_t p = sntree.snode_parent[c];
                if (p == NO_PARENT || p == INACTIVE_NODE) continue;

                auto [dim_c_snode, dim_c_sep] = sntree.clique_dim(c);
                auto [dim_p_snode, dim_p_sep] = sntree.clique_dim(p);

                int64_t fi = SuperNodeTree::fill_in(dim_c_snode, dim_c_sep,
                                                     dim_p_snode, dim_p_sep);
                if (fi == 0) {
                    sntree.merge_two_cliques_parent_child(p, c);
                    merged = true;
                    break;  // restart post-order iteration
                }
            }
        }

        // Post-process: rebuild post-order for remaining active cliques
        // Filter out inactive/empty nodes
        std::vector<int64_t> active_parents;
        std::vector<VertexSet> active_children;
        std::vector<VertexSet> active_snodes;
        std::vector<VertexSet> active_separators;

        // Build map from old index to new index
        std::vector<int64_t> old_to_new(sntree.snode_parent.size(), -1);
        int64_t new_idx = 0;
        for (size_t i = 0; i < sntree.snode.size(); ++i) {
            if (!sntree.snode[i].empty() && sntree.snode_parent[i] != INACTIVE_NODE) {
                old_to_new[i] = new_idx++;
            }
        }

        active_snodes.resize(new_idx);
        active_separators.resize(new_idx);
        active_parents.resize(new_idx, NO_PARENT);
        active_children.resize(new_idx);

        for (size_t i = 0; i < sntree.snode.size(); ++i) {
            if (old_to_new[i] < 0) continue;
            int64_t ni = old_to_new[i];
            active_snodes[ni] = std::move(sntree.snode[i]);
            active_separators[ni] = std::move(sntree.separators[i]);

            int64_t op = sntree.snode_parent[i];
            if (op != NO_PARENT && op != INACTIVE_NODE && old_to_new[op] >= 0) {
                active_parents[ni] = old_to_new[op];
            }
        }

        // Rebuild children
        for (int64_t i = 0; i < new_idx; ++i) {
            if (active_parents[i] != NO_PARENT) {
                active_children[active_parents[i]].insert(i);
            }
        }

        sntree.snode = std::move(active_snodes);
        sntree.separators = std::move(active_separators);
        sntree.snode_parent = std::move(active_parents);
        sntree.snode_children = std::move(active_children);
        sntree.n_cliques = new_idx;

        // Rebuild post-order
        sntree.snode_post.resize(new_idx);
        SuperNodeTree::post_order(sntree.snode_post, sntree.snode_parent,
                                   sntree.snode_children, new_idx);
    }

    if (sntree.n_cliques <= 1) return;  // not decomposed, or everything merged

    // Reorder vertices in supernodes to have consecutive order
    std::vector<int64_t> ordering = std::move(ldl.perm);
    sntree.reorder_snode_consecutively(ordering);

    // Calculate block dimensions
    sntree.calculate_block_dimensions();

    // Store the pattern
    SparsityPattern pat{
        std::move(sntree),
        std::move(ordering),
        coneidx
    };

    spatterns.push_back(std::move(pat));
}

// ============================================================================
// Block index computation helpers (port of Rust get_block_indices)
// ============================================================================

struct BlockOverlapTriplet {
    int64_t i, j;
    bool is_overlap;
};

static std::vector<BlockOverlapTriplet> get_block_indices(
    const std::vector<int64_t>& snode,
    const std::vector<int64_t>& separator,
    int64_t nv)
{
    int64_t N = static_cast<int64_t>(separator.size() + snode.size());
    std::vector<BlockOverlapTriplet> indices;
    indices.reserve(N * (N + 1) / 2);

    // Separator x Separator (overlap)
    for (int64_t jj = 0; jj < static_cast<int64_t>(separator.size()); ++jj) {
        for (int64_t ii = 0; ii <= jj; ++ii) {
            indices.push_back({separator[ii], separator[jj], true});
        }
    }

    // Snode x Snode (not overlap)
    for (int64_t jj = 0; jj < static_cast<int64_t>(snode.size()); ++jj) {
        for (int64_t ii = 0; ii <= jj; ++ii) {
            indices.push_back({snode[ii], snode[jj], false});
        }
    }

    // Snode x Separator cross-terms (not overlap)
    for (auto si : snode) {
        for (auto sj : separator) {
            int64_t lo = std::min(si, sj);
            int64_t hi = std::max(si, sj);
            indices.push_back({lo, hi, false});
        }
    }

    // Sort by column-major order: (j * nv + i)
    std::sort(indices.begin(), indices.end(),
              [nv](const BlockOverlapTriplet& a, const BlockOverlapTriplet& b) {
                  return a.j * nv + a.i < b.j * nv + b.i;
              });

    return indices;
}

// Find the index of (i, j) in the parent clique's block
static int64_t parent_block_index(const std::vector<int64_t>& parent_clique,
                                   int64_t i, int64_t j)
{
    // Find position of i and j in parent_clique (sorted)
    auto it_i = std::lower_bound(parent_clique.begin(), parent_clique.end(), i);
    auto it_j = std::lower_bound(parent_clique.begin(), parent_clique.end(), j);
    int64_t ir = static_cast<int64_t>(it_i - parent_clique.begin());
    int64_t jr = static_cast<int64_t>(it_j - parent_clique.begin());
    return coord_to_tri(ir, jr);
}

// Get clique vertices by direct index (not through post-order)
static std::vector<int64_t> get_clique_by_index_sorted(
    const SuperNodeTree& sntree, int64_t idx, const std::vector<int64_t>& ordering)
{
    std::vector<int64_t> clique;
    for (int64_t v : sntree.snode[idx].data()) {
        clique.push_back(ordering[v]);
    }
    for (int64_t v : sntree.separators[idx].data()) {
        clique.push_back(ordering[v]);
    }
    std::sort(clique.begin(), clique.end());
    return clique;
}

// ============================================================================
// analyze() — main analysis function
// ============================================================================

ChordalInfo ChordalInfo::analyze(
    const int64_t* A_colptr, const int64_t* A_rowind,
    int64_t A_ncol, int64_t A_nrow,
    const int64_t* psd_dims, int64_t num_psd,
    int64_t num_zero_, int64_t num_nonneg_,
    int64_t total_soc_, int64_t num_exp_, int64_t num_power_,
    const bool* b_sparsity_pattern)
{
    ChordalInfo info;
    info.n_orig = A_ncol;
    info.m_orig = A_nrow;
    info.num_zero = num_zero_;
    info.num_nonneg = num_nonneg_;
    info.total_soc = total_soc_;
    info.num_exp = num_exp_;
    info.num_power = num_power_;

    for (int64_t i = 0; i < num_psd; ++i) {
        info.orig_psd_dims.push_back(psd_dims[i]);
    }

    // Build aggregate sparsity mask
    std::vector<bool> nz_mask = find_aggregate_sparsity_mask(
        A_colptr, A_rowind, A_ncol, A_nrow, b_sparsity_pattern);

    // Compute PSD cone row offsets
    // Cone layout: zero | nonneg | SOC | exp | power | PSD...
    int64_t psd_row_start = num_zero_ + num_nonneg_ + total_soc_ + num_exp_ * 3 + num_power_ * 3;

    // Analyze each PSD cone
    for (int64_t pi = 0; pi < num_psd; ++pi) {
        int64_t dim = psd_dims[pi];
        int64_t tri = dim * (dim + 1) / 2;

        // Extract the sub-mask for this cone
        std::vector<bool> cone_mask(nz_mask.begin() + psd_row_start,
                                     nz_mask.begin() + psd_row_start + tri);

        // coneidx: the original cone index (PSD cones come after all others)
        // We count: num_zero + num_nonneg + num_soc_cones + num_exp + num_power + pi
        // But for chordal decomposition, we only track PSD cone index for spatterns
        info.analyse_psd_sparsity(cone_mask, dim, pi);

        psd_row_start += tri;
    }

    if (!info.is_decomposed()) {
        return info;
    }

    // ================================================================
    // Build augmented problem structure (compact augmentation)
    // ================================================================

    // Compute decomposed dimensions and overlaps
    int64_t decomposed_dim = 0;
    int64_t total_overlaps = 0;

    // Non-PSD cones contribute their original dimensions
    int64_t non_psd_rows = num_zero_ + num_nonneg_ + total_soc_ + num_exp_ * 3 + num_power_ * 3;
    decomposed_dim += non_psd_rows;

    // PSD cones: decomposed or original
    auto spatterns_iter = info.spatterns.begin();
    for (int64_t pi = 0; pi < num_psd; ++pi) {
        if (spatterns_iter != info.spatterns.end() &&
            spatterns_iter->orig_index == pi) {
            auto [dim, overlaps] = spatterns_iter->sntree.get_decomposed_dim_and_overlaps();
            decomposed_dim += dim;
            total_overlaps += overlaps;
            ++spatterns_iter;
        } else {
            int64_t dim = psd_dims[pi];
            decomposed_dim += dim * (dim + 1) / 2;
        }
    }

    info.decomposed_psd_dims.clear();

    int64_t m_aug = decomposed_dim;
    int64_t n_aug = A_ncol + total_overlaps;

    info.n_aug = n_aug;
    info.m_aug = m_aug;

    // ================================================================
    // Build augmented A structure in COO (triplet) form, then convert to CSC
    // ================================================================

    int64_t A_nnz = A_colptr[A_ncol];
    int64_t A_aug_nnz = A_nnz + 2 * total_overlaps;

    info.P_nnz_orig = 0;  // Will be set by caller if needed
    info.A_nnz_orig = A_nnz;

    // Triplet arrays for augmented A
    std::vector<int64_t> Aa_rows(A_aug_nnz, INT64_MAX);
    std::vector<int64_t> Aa_cols(A_aug_nnz);
    std::vector<double> Aa_vals(A_aug_nnz);

    // Initialize columns: first A_nnz entries get original columns,
    // remaining get overlap columns
    // Original A columns
    {
        int64_t k = 0;
        for (int64_t col = 0; col < A_ncol; ++col) {
            for (int64_t p = A_colptr[col]; p < A_colptr[col + 1]; ++p) {
                Aa_cols[k] = col;
                Aa_vals[k] = 1.0;  // placeholder; actual values filled at value phase
                k++;
            }
        }
    }

    // Overlap columns: pairs of (+1, -1) in new columns
    {
        int64_t col_start = A_ncol;
        for (int64_t k = A_nnz; k < A_aug_nnz - 1; k += 2) {
            Aa_cols[k] = col_start;
            Aa_cols[k + 1] = col_start;
            Aa_vals[k] = 1.0;
            Aa_vals[k + 1] = -1.0;
            col_start++;
        }
    }

    // b_row_map: for each augmented row, which original row it came from
    info.b_row_map.assign(m_aug, INT64_MAX);

    // cone_maps: only PSD cone entries (non-PSD cones are trivially mapped 1:1)
    info.cone_maps.clear();

    // ================================================================
    // Process cones: remap rows
    // ================================================================

    int64_t row_ptr = 0;      // current position in augmented row space
    int64_t overlap_ptr = A_nnz;  // current position in overlap entries

    // Non-PSD cones: direct 1:1 mapping at the start.
    // The non-PSD rows map to the same positions (identity mapping).
    {
        for (int64_t k = 0; k < non_psd_rows; ++k) {
            info.b_row_map[k] = k;
        }
        // Set rows in Aa_rows for all non-PSD A entries
        for (int64_t col = 0; col < A_ncol; ++col) {
            for (int64_t p = A_colptr[col]; p < A_colptr[col + 1]; ++p) {
                int64_t r = A_rowind[p];
                if (r < non_psd_rows) {
                    Aa_rows[p] = r;  // identity mapping
                }
            }
        }
        row_ptr = non_psd_rows;
    }

    // ================================================================
    // PSD cones: decomposed or pass-through
    // ================================================================

    int64_t psd_orig_start = num_zero_ + num_nonneg_ + total_soc_ + num_exp_ * 3 + num_power_ * 3;
    spatterns_iter = info.spatterns.begin();
    int64_t spattern_count = 0;

    for (int64_t pi = 0; pi < num_psd; ++pi) {
        int64_t dim = psd_dims[pi];
        int64_t tri = dim * (dim + 1) / 2;
        int64_t orig_row_start = psd_orig_start;

        if (spatterns_iter != info.spatterns.end() &&
            spatterns_iter->orig_index == pi)
        {
            // Decomposed PSD cone
            const SparsityPattern& spattern = *spatterns_iter;
            const SuperNodeTree& sntree = spattern.sntree;
            const std::vector<int64_t>& ordering = spattern.ordering;

            // Build clique-to-rows map (maps post-order index -> row range)
            std::unordered_map<int64_t, std::pair<int64_t, int64_t>> clique_to_rows;
            {
                int64_t rp = row_ptr;
                for (int64_t i = sntree.n_cliques - 1; i >= 0; --i) {
                    int64_t nb = sntree.get_nblk(i);
                    int64_t num_rows = nb * (nb + 1) / 2;
                    int64_t post_idx = sntree.snode_post[i];
                    clique_to_rows[post_idx] = {rp, rp + num_rows};
                    rp += num_rows;
                }
            }

            // Process cliques in descending topological order (reversed post-order)
            for (int64_t i = sntree.n_cliques - 1; i >= 0; --i) {
                // Get supernodes and separators, undo reordering
                std::vector<int64_t> separator;
                for (int64_t v : sntree.get_separators(i).data()) {
                    separator.push_back(ordering[v]);
                }
                std::sort(separator.begin(), separator.end());

                std::vector<int64_t> snode_verts;
                for (int64_t v : sntree.get_snode(i).data()) {
                    snode_verts.push_back(ordering[v]);
                }
                std::sort(snode_verts.begin(), snode_verts.end());

                int64_t nv = static_cast<int64_t>(ordering.size());
                auto block_indices = get_block_indices(snode_verts, separator, nv);

                // Parent clique info
                std::vector<int64_t> parent_clique;
                int64_t parent_row_start = 0;
                if (i < sntree.n_cliques - 1) {
                    int64_t parent_index = sntree.get_clique_parent(i);
                    parent_clique = get_clique_by_index_sorted(sntree, parent_index, ordering);
                    parent_row_start = clique_to_rows[parent_index].first;
                }

                // Process block indices
                int64_t counter = 0;
                for (const auto& blk : block_indices) {
                    int64_t new_row_val = row_ptr + counter;

                    if (blk.is_overlap) {
                        // Overlap: create +1/-1 entries in overlap columns
                        Aa_rows[overlap_ptr] = new_row_val;
                        Aa_rows[overlap_ptr + 1] = parent_row_start +
                            parent_block_index(parent_clique, blk.i, blk.j);
                        overlap_ptr += 2;

                        // This row has no original source
                        // b_row_map already initialized to INT64_MAX
                    } else {
                        // Non-overlap: remap original A rows
                        int64_t k = coord_to_tri(blk.i, blk.j);
                        int64_t orig_row = orig_row_start + k;

                        // Remap this row in A columns
                        for (int64_t col = 0; col < A_ncol; ++col) {
                            for (int64_t p = A_colptr[col]; p < A_colptr[col + 1]; ++p) {
                                if (A_rowind[p] == orig_row) {
                                    Aa_rows[p] = new_row_val;
                                }
                            }
                        }

                        // b_row_map
                        info.b_row_map[new_row_val] = orig_row;
                    }
                    counter++;
                }

                // Record cone and cone_map
                int64_t cone_dim = sntree.get_nblk(i);
                info.decomposed_psd_dims.push_back(cone_dim);
                info.cone_maps.push_back({pi, spattern_count, i});

                row_ptr += triangular_number(cone_dim);
            }

            spattern_count++;
            ++spatterns_iter;
        } else {
            // Non-decomposed PSD cone: direct mapping
            int64_t offset = row_ptr - orig_row_start;
            for (int64_t col = 0; col < A_ncol; ++col) {
                for (int64_t p = A_colptr[col]; p < A_colptr[col + 1]; ++p) {
                    int64_t r = A_rowind[p];
                    if (r >= orig_row_start && r < orig_row_start + tri) {
                        Aa_rows[p] = r + offset;
                    }
                }
            }
            for (int64_t k = 0; k < tri; ++k) {
                info.b_row_map[row_ptr + k] = orig_row_start + k;
            }
            info.decomposed_psd_dims.push_back(dim);
            info.cone_maps.push_back({pi, -1, -1});
            row_ptr += tri;
        }

        psd_orig_start += tri;
    }

    // ================================================================
    // Convert triplets to CSC for augmented A
    // ================================================================

    // Filter out any unassigned rows (shouldn't happen, but defensive)
    // Build CSC from (Aa_rows, Aa_cols, Aa_vals)
    info.A_aug_colptr.assign(n_aug + 1, 0);

    // Count entries per column
    for (int64_t k = 0; k < A_aug_nnz; ++k) {
        assert(Aa_rows[k] != INT64_MAX && "Unassigned row in augmented A");
        info.A_aug_colptr[Aa_cols[k] + 1]++;
    }
    for (int64_t j = 0; j < n_aug; ++j) {
        info.A_aug_colptr[j + 1] += info.A_aug_colptr[j];
    }

    info.A_aug_rowind.resize(A_aug_nnz);
    info.A_aug_values.resize(A_aug_nnz);
    // Track where each triplet entry ends up in CSC
    std::vector<int64_t> triplet_to_csc(A_aug_nnz);
    std::vector<int64_t> col_pos(info.A_aug_colptr.begin(), info.A_aug_colptr.begin() + n_aug);

    for (int64_t k = 0; k < A_aug_nnz; ++k) {
        int64_t col = Aa_cols[k];
        int64_t pos2 = col_pos[col]++;
        info.A_aug_rowind[pos2] = Aa_rows[k];
        info.A_aug_values[pos2] = Aa_vals[k];
        triplet_to_csc[k] = pos2;
    }

    // Sort row indices within each column, tracking permutation
    for (int64_t j = 0; j < n_aug; ++j) {
        int64_t start = info.A_aug_colptr[j];
        int64_t end = info.A_aug_colptr[j + 1];
        if (end - start <= 1) continue;

        // Sort by row index, keeping values aligned
        std::vector<int64_t> perm2(end - start);
        std::iota(perm2.begin(), perm2.end(), 0);
        std::sort(perm2.begin(), perm2.end(),
                  [&](int64_t a, int64_t b) {
                      return info.A_aug_rowind[start + a] < info.A_aug_rowind[start + b];
                  });

        std::vector<int64_t> tmp_rows(end - start);
        std::vector<double> tmp_vals(end - start);
        for (int64_t k = 0; k < end - start; ++k) {
            tmp_rows[k] = info.A_aug_rowind[start + perm2[k]];
            tmp_vals[k] = info.A_aug_values[start + perm2[k]];
        }
        std::copy(tmp_rows.begin(), tmp_rows.end(), info.A_aug_rowind.begin() + start);
        std::copy(tmp_vals.begin(), tmp_vals.end(), info.A_aug_values.begin() + start);

        // Update triplet_to_csc: perm2[new_pos] = old_offset, so
        // old CSC position (start + old_offset) -> new CSC position (start + new_pos)
        // Build inverse: inv_perm[old_offset] = new_pos
        std::vector<int64_t> inv_perm(end - start);
        for (int64_t k = 0; k < end - start; ++k) {
            inv_perm[perm2[k]] = k;
        }
        for (int64_t k = 0; k < A_aug_nnz; ++k) {
            int64_t pos = triplet_to_csc[k];
            if (pos >= start && pos < end) {
                triplet_to_csc[k] = start + inv_perm[pos - start];
            }
        }
    }

    // Build A_orig_to_aug: first A_nnz entries in triplet are original A entries
    info.A_orig_to_aug.resize(A_nnz);
    for (int64_t k = 0; k < A_nnz; ++k) {
        info.A_orig_to_aug[k] = triplet_to_csc[k];
    }

    // ================================================================
    // Build augmented P structure (original P block + zero diagonal for overlaps)
    // P_aug is block-diagonal: [P_orig, 0_{overlaps x overlaps}]
    // The structure is just the original P columns plus empty overlap columns.
    // P_aug_colptr and P_aug_rowind will be set by the caller who has P info.
    // For now, leave P_aug empty — caller fills it.
    // ================================================================

    return info;
}

// ============================================================================
// Helper: compute original row start for PSD cone by index
// ============================================================================

static int64_t psd_orig_row_start(const ChordalInfo& info, int64_t psd_cone_idx) {
    int64_t start = info.num_zero + info.num_nonneg + info.total_soc +
                    info.num_exp * 3 + info.num_power * 3;
    for (int64_t pi = 0; pi < psd_cone_idx; ++pi) {
        start += info.orig_psd_dims[pi] * (info.orig_psd_dims[pi] + 1) / 2;
    }
    return start;
}

// ============================================================================
// Helper: reverse mapping core — shared between reverse_s and reverse_z
//
// Iterates through cone_maps, mapping augmented → original.
// For s: accumulate (+=). For z: overwrite (=).
// ============================================================================

enum class ReverseMode { Accumulate, Overwrite };

static void reverse_impl(
    const ChordalInfo& info,
    const double* aug, double* orig,
    ReverseMode mode)
{
    std::memset(orig, 0, sizeof(double) * info.m_orig);

    // Non-PSD cones: direct 1:1 copy for first non_psd_rows
    int64_t non_psd_rows = info.num_zero + info.num_nonneg + info.total_soc +
                            info.num_exp * 3 + info.num_power * 3;

    // The augmented problem preserves non-PSD rows in the same order at the beginning
    for (int64_t k = 0; k < non_psd_rows; ++k) {
        orig[k] = aug[k];
    }

    // PSD cones: iterate through cone_maps (which only contain PSD entries)
    int64_t aug_ptr = non_psd_rows;
    std::vector<int64_t> clique_buffer;

    for (const auto& cm : info.cone_maps) {
        if (!cm.is_decomposed()) {
            // Non-decomposed PSD cone: 1:1 mapping
            int64_t dim = info.orig_psd_dims[cm.orig_index];
            int64_t tri = dim * (dim + 1) / 2;
            int64_t orig_start = psd_orig_row_start(info, cm.orig_index);

            for (int64_t k = 0; k < tri; ++k) {
                if (mode == ReverseMode::Accumulate) {
                    orig[orig_start + k] += aug[aug_ptr + k];
                } else {
                    orig[orig_start + k] = aug[aug_ptr + k];
                }
            }
            aug_ptr += tri;
        } else {
            // Decomposed PSD cone clique
            const SparsityPattern& pat = info.spatterns[cm.pattern_index];
            const SuperNodeTree& sntree = pat.sntree;
            const std::vector<int64_t>& ordering = pat.ordering;

            VertexSet clique = sntree.get_clique(cm.clique_index);
            clique_buffer.resize(clique.size());
            {
                int64_t idx = 0;
                for (int64_t v : clique.data()) {
                    clique_buffer[idx++] = ordering[v];
                }
            }
            std::sort(clique_buffer.begin(), clique_buffer.end());

            int64_t orig_start = psd_orig_row_start(info, cm.orig_index);

            int64_t counter = 0;
            for (size_t jj = 0; jj < clique_buffer.size(); ++jj) {
                for (size_t ii = 0; ii <= jj; ++ii) {
                    int64_t offset = coord_to_tri(clique_buffer[ii], clique_buffer[jj]);
                    if (mode == ReverseMode::Accumulate) {
                        orig[orig_start + offset] += aug[aug_ptr + counter];
                    } else {
                        orig[orig_start + offset] = aug[aug_ptr + counter];
                    }
                    counter++;
                }
            }
            aug_ptr += triangular_number(static_cast<int64_t>(clique_buffer.size()));
        }
    }
}

// ============================================================================
// reverse_s — accumulate clique values into original
// ============================================================================

void ChordalInfo::reverse_s(const double* s_aug, double* s_orig) const {
    reverse_impl(*this, s_aug, s_orig, ReverseMode::Accumulate);
}

// ============================================================================
// reverse_z — overwrite original with last clique's values
// ============================================================================

void ChordalInfo::reverse_z(const double* z_aug, double* z_orig) const {
    reverse_impl(*this, z_aug, z_orig, ReverseMode::Overwrite);
}

// ============================================================================
// complete_z — PSD completion (Cholesky-based fill of structural zeros)
//
// Operates on z in original dimensions, after reverse mapping.
// Traverses clique tree in reverse post-order (descending).
// ============================================================================

void ChordalInfo::complete_z(double* z, int64_t psd_offset) const {
    for (const auto& pat : spatterns) {
        const SuperNodeTree& sntree = pat.sntree;
        const std::vector<int64_t>& p = pat.ordering;
        int64_t N = static_cast<int64_t>(p.size());

        // Compute the svec offset for this pattern's PSD cone
        int64_t cone_offset = psd_offset;
        for (int64_t pi = 0; pi < pat.orig_index; ++pi) {
            cone_offset += orig_psd_dims[pi] * (orig_psd_dims[pi] + 1) / 2;
        }

        double* z_cone = z + cone_offset;

        // Unpack svec to full symmetric matrix
        std::vector<double> W(N * N, 0.0);

        // z_cone is in svec format (column-major upper triangle)
        // Unpack to full symmetric, with permutation p
        // First: unpack svec to dense matrix A (original ordering)
        std::vector<double> A(N * N, 0.0);
        {
            int64_t idx = 0;
            for (int64_t col = 0; col < N; ++col) {
                for (int64_t row = 0; row <= col; ++row) {
                    double val = z_cone[idx];
                    A[row * N + col] = val;
                    A[col * N + row] = val;
                    idx++;
                }
            }
        }

        // Permute: W = A[p, p] (W[i,j] = A[p[i], p[j]])
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                W[i * N + j] = A[p[i] * N + p[j]];
            }
        }

        // Process cliques in descending order (reverse post-order),
        // skipping the root clique
        for (int64_t j = sntree.n_cliques - 2; j >= 0; --j) {
            const VertexSet& nu = sntree.get_snode(j);
            const VertexSet& alpha = sntree.get_separators(j);

            if (alpha.empty()) continue;

            int64_t i_rep = nu[0];

            // eta: row indices > i_rep not in alpha or nu
            std::vector<int64_t> eta;
            for (int64_t x = i_rep + 1; x < N; ++x) {
                if (!alpha.contains(x) && !nu.contains(x)) {
                    eta.push_back(x);
                }
            }

            if (eta.empty()) continue;

            int64_t na = static_cast<int64_t>(alpha.size());
            int64_t nn = static_cast<int64_t>(nu.size());
            int64_t ne = static_cast<int64_t>(eta.size());

            // Extract Waa (alpha x alpha)
            std::vector<double> Waa(na * na);
            for (int64_t ai = 0; ai < na; ++ai) {
                for (int64_t aj = 0; aj < na; ++aj) {
                    Waa[ai * na + aj] = W[alpha[ai] * N + alpha[aj]];
                }
            }

            // Extract Wan (alpha x nu)
            std::vector<double> Wan(na * nn);
            for (int64_t ai = 0; ai < na; ++ai) {
                for (int64_t ni = 0; ni < nn; ++ni) {
                    Wan[ai * nn + ni] = W[alpha[ai] * N + nu[ni]];
                }
            }

            // Extract Wea (eta x alpha)
            std::vector<double> Wea(ne * na);
            for (int64_t ei = 0; ei < ne; ++ei) {
                for (int64_t ai = 0; ai < na; ++ai) {
                    Wea[ei * na + ai] = W[eta[ei] * N + alpha[ai]];
                }
            }

            // Solve: Y = Waa \ Wan  (Waa * Y = Wan)
            // Use Cholesky: Waa = L * L^T, then L * L^T * Y = Wan
            // For simplicity, use direct solve via Gaussian elimination with pivoting

            // Copy Waa for factoring
            std::vector<double> L(na * na);
            std::copy(Waa.begin(), Waa.end(), L.begin());

            // Cholesky factorization (in-place, lower triangle)
            bool chol_ok = true;
            for (int64_t k = 0; k < na; ++k) {
                double diag = L[k * na + k];
                if (diag <= 0.0) { chol_ok = false; break; }
                diag = std::sqrt(diag);
                L[k * na + k] = diag;
                for (int64_t i2 = k + 1; i2 < na; ++i2) {
                    L[i2 * na + k] /= diag;
                }
                for (int64_t j2 = k + 1; j2 < na; ++j2) {
                    for (int64_t i2 = j2; i2 < na; ++i2) {
                        L[i2 * na + j2] -= L[i2 * na + k] * L[j2 * na + k];
                    }
                }
            }

            std::vector<double> Y(na * nn);
            std::copy(Wan.begin(), Wan.end(), Y.begin());

            if (chol_ok) {
                // Solve L * Z = Wan (forward substitution)
                for (int64_t c = 0; c < nn; ++c) {
                    for (int64_t k = 0; k < na; ++k) {
                        Y[k * nn + c] /= L[k * na + k];
                        for (int64_t i2 = k + 1; i2 < na; ++i2) {
                            Y[i2 * nn + c] -= L[i2 * na + k] * Y[k * nn + c];
                        }
                    }
                    // Solve L^T * Y_col = Z_col (backward substitution)
                    for (int64_t k = na - 1; k >= 0; --k) {
                        Y[k * nn + c] /= L[k * na + k];
                        for (int64_t i2 = 0; i2 < k; ++i2) {
                            Y[i2 * nn + c] -= L[k * na + i2] * Y[k * nn + c];
                        }
                    }
                }
            } else {
                // Clique principal block must be positive definite to complete.
                throw std::runtime_error(
                    "PSD completion failed: clique principal block is not "
                    "positive definite (Cholesky failed); cannot complete "
                    "matrix without a silently-wrong result");
            }

            // Wea_times_Y = Wea * Y (ne x nn)
            std::vector<double> Wea_Y(ne * nn, 0.0);
            for (int64_t ei = 0; ei < ne; ++ei) {
                for (int64_t ni = 0; ni < nn; ++ni) {
                    double sum = 0.0;
                    for (int64_t ai = 0; ai < na; ++ai) {
                        sum += Wea[ei * na + ai] * Y[ai * nn + ni];
                    }
                    Wea_Y[ei * nn + ni] = sum;
                }
            }

            // Write back: W[eta, nu] = Wea_Y
            for (int64_t ei = 0; ei < ne; ++ei) {
                for (int64_t ni = 0; ni < nn; ++ni) {
                    W[eta[ei] * N + nu[ni]] = Wea_Y[ei * nn + ni];
                    W[nu[ni] * N + eta[ei]] = Wea_Y[ei * nn + ni];  // symmetry
                }
            }
        }

        // Inverse permutation
        std::vector<int64_t> ip(N);
        for (int64_t i = 0; i < N; ++i) {
            ip[p[i]] = i;
        }

        // Unpermute: A = W[ip, ip]
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                A[i * N + j] = W[ip[i] * N + ip[j]];
            }
        }

        // Pack back to svec
        {
            int64_t idx = 0;
            for (int64_t col = 0; col < N; ++col) {
                for (int64_t row = 0; row <= col; ++row) {
                    z_cone[idx] = A[row * N + col];
                    idx++;
                }
            }
        }
    }
}

// ============================================================================
// Adjoint methods
// ============================================================================

void ChordalInfo::adjoint_reverse_s(const double* ds_orig, double* ds_aug) const {
    // Adjoint of accumulation (+=) is broadcast:
    // Forward: s_orig[offset] += s_aug[aug_row]
    // Adjoint: ds_aug[aug_row] = ds_orig[offset]

    std::memset(ds_aug, 0, sizeof(double) * m_aug);

    int64_t non_psd_rows = num_zero + num_nonneg + total_soc + num_exp * 3 + num_power * 3;

    // Non-PSD: 1:1
    for (int64_t k = 0; k < non_psd_rows; ++k) {
        ds_aug[k] = ds_orig[k];
    }

    // PSD cone_maps
    int64_t aug_ptr = non_psd_rows;
    std::vector<int64_t> clique_buffer;

    for (const auto& cm : cone_maps) {
        if (!cm.is_decomposed()) {
            int64_t dim = orig_psd_dims[cm.orig_index];
            int64_t tri = dim * (dim + 1) / 2;
            int64_t orig_start = psd_orig_row_start(*this, cm.orig_index);
            for (int64_t k = 0; k < tri; ++k) {
                ds_aug[aug_ptr + k] = ds_orig[orig_start + k];
            }
            aug_ptr += tri;
        } else {
            const SparsityPattern& pat = spatterns[cm.pattern_index];
            const SuperNodeTree& sntree = pat.sntree;
            const std::vector<int64_t>& ordering = pat.ordering;

            VertexSet clique = sntree.get_clique(cm.clique_index);
            clique_buffer.resize(clique.size());
            {
                int64_t idx = 0;
                for (int64_t v : clique.data()) {
                    clique_buffer[idx++] = ordering[v];
                }
            }
            std::sort(clique_buffer.begin(), clique_buffer.end());

            int64_t orig_start = psd_orig_row_start(*this, cm.orig_index);

            int64_t counter = 0;
            for (size_t jj = 0; jj < clique_buffer.size(); ++jj) {
                for (size_t ii = 0; ii <= jj; ++ii) {
                    int64_t offset = coord_to_tri(clique_buffer[ii], clique_buffer[jj]);
                    ds_aug[aug_ptr + counter] = ds_orig[orig_start + offset];
                    counter++;
                }
            }
            aug_ptr += triangular_number(static_cast<int64_t>(clique_buffer.size()));
        }
    }
}

void ChordalInfo::adjoint_reverse_z(const double* dz_orig, double* dz_aug) const {
    // Adjoint of overwrite (=) is: only the LAST writer gets the gradient.

    std::memset(dz_aug, 0, sizeof(double) * m_aug);

    int64_t non_psd_rows = num_zero + num_nonneg + total_soc + num_exp * 3 + num_power * 3;

    // Non-PSD: 1:1 (always last writer for non-PSD)
    for (int64_t k = 0; k < non_psd_rows; ++k) {
        dz_aug[k] = dz_orig[k];
    }

    // First pass: find last aug_ptr that writes to each PSD orig_row
    std::vector<int64_t> last_aug_for_orig(m_orig, -1);

    int64_t aug_ptr = non_psd_rows;
    std::vector<int64_t> clique_buffer;

    for (const auto& cm : cone_maps) {
        if (!cm.is_decomposed()) {
            int64_t dim = orig_psd_dims[cm.orig_index];
            int64_t tri = dim * (dim + 1) / 2;
            int64_t orig_start = psd_orig_row_start(*this, cm.orig_index);
            for (int64_t k = 0; k < tri; ++k) {
                last_aug_for_orig[orig_start + k] = aug_ptr + k;
            }
            aug_ptr += tri;
        } else {
            const SparsityPattern& pat = spatterns[cm.pattern_index];
            const SuperNodeTree& sntree = pat.sntree;
            const std::vector<int64_t>& ordering = pat.ordering;

            VertexSet clique = sntree.get_clique(cm.clique_index);
            clique_buffer.resize(clique.size());
            {
                int64_t idx = 0;
                for (int64_t v : clique.data()) {
                    clique_buffer[idx++] = ordering[v];
                }
            }
            std::sort(clique_buffer.begin(), clique_buffer.end());

            int64_t orig_start = psd_orig_row_start(*this, cm.orig_index);

            int64_t counter = 0;
            for (size_t jj = 0; jj < clique_buffer.size(); ++jj) {
                for (size_t ii = 0; ii <= jj; ++ii) {
                    int64_t offset = coord_to_tri(clique_buffer[ii], clique_buffer[jj]);
                    last_aug_for_orig[orig_start + offset] = aug_ptr + counter;
                    counter++;
                }
            }
            aug_ptr += triangular_number(static_cast<int64_t>(clique_buffer.size()));
        }
    }

    // Second pass: assign gradients only to last writers
    aug_ptr = non_psd_rows;
    for (const auto& cm : cone_maps) {
        if (!cm.is_decomposed()) {
            int64_t dim = orig_psd_dims[cm.orig_index];
            int64_t tri = dim * (dim + 1) / 2;
            int64_t orig_start = psd_orig_row_start(*this, cm.orig_index);
            for (int64_t k = 0; k < tri; ++k) {
                if (last_aug_for_orig[orig_start + k] == aug_ptr + k) {
                    dz_aug[aug_ptr + k] = dz_orig[orig_start + k];
                }
            }
            aug_ptr += tri;
        } else {
            const SparsityPattern& pat = spatterns[cm.pattern_index];
            const SuperNodeTree& sntree = pat.sntree;
            const std::vector<int64_t>& ordering = pat.ordering;

            VertexSet clique = sntree.get_clique(cm.clique_index);
            clique_buffer.resize(clique.size());
            {
                int64_t idx = 0;
                for (int64_t v : clique.data()) {
                    clique_buffer[idx++] = ordering[v];
                }
            }
            std::sort(clique_buffer.begin(), clique_buffer.end());

            int64_t orig_start = psd_orig_row_start(*this, cm.orig_index);

            int64_t counter = 0;
            for (size_t jj = 0; jj < clique_buffer.size(); ++jj) {
                for (size_t ii = 0; ii <= jj; ++ii) {
                    int64_t offset = coord_to_tri(clique_buffer[ii], clique_buffer[jj]);
                    int64_t orig_row = orig_start + offset;
                    if (last_aug_for_orig[orig_row] == aug_ptr + counter) {
                        dz_aug[aug_ptr + counter] = dz_orig[orig_row];
                    }
                    counter++;
                }
            }
            aug_ptr += triangular_number(static_cast<int64_t>(clique_buffer.size()));
        }
    }
}

void ChordalInfo::adjoint_complete_z(double* dz, int64_t psd_offset) const {
    // Adjoint of PSD completion.
    // Forward complete_z: for each clique j (reverse post-order, descending):
    //   Y = Waa^{-1} * Wan
    //   W[eta, nu] = Wea * Y  (and symmetry)
    //
    // Adjoint processes in forward post-order (ascending).
    // We need the same intermediate quantities (Waa, Y, etc.) from the forward pass.
    // Since complete_z is called once post-solve, we re-run the forward computation
    // to get the intermediates, then propagate gradients in reverse order.

    for (const auto& pat : spatterns) {
        const SuperNodeTree& sntree = pat.sntree;
        const std::vector<int64_t>& p = pat.ordering;
        int64_t N = static_cast<int64_t>(p.size());

        // Compute cone offset
        int64_t cone_offset = psd_offset;
        for (int64_t pi = 0; pi < pat.orig_index; ++pi) {
            cone_offset += orig_psd_dims[pi] * (orig_psd_dims[pi] + 1) / 2;
        }

        double* dz_cone = dz + cone_offset;

        // Unpack dz svec to full symmetric dA matrix
        std::vector<double> dA(N * N, 0.0);
        {
            int64_t idx = 0;
            for (int64_t col = 0; col < N; ++col) {
                for (int64_t row = 0; row <= col; ++row) {
                    double val = dz_cone[idx];
                    dA[row * N + col] = val;
                    dA[col * N + row] = val;
                    idx++;
                }
            }
        }

        // Inverse permutation
        std::vector<int64_t> ip(N);
        for (int64_t i = 0; i < N; ++i) ip[p[i]] = i;

        // Adjoint of unpermute: dW[ip[i], ip[j]] += dA[i, j]
        // Equivalently: dW[a, b] = dA[p[a], p[b]]
        std::vector<double> dW(N * N, 0.0);
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                dW[i * N + j] = dA[p[i] * N + p[j]];
            }
        }

        // Limitation: this adjoint only supports losses that do not depend on
        // the structural-fill entries W[eta,nu]. Full correctness requires the
        // forward Cholesky factors, which complete_z does not cache. We detect
        // a fill-dependent loss below and throw rather than return wrong gradients.

        // Process cliques in forward post-order (ascending)
        for (int64_t j = 0; j < sntree.n_cliques - 1; ++j) {
            const VertexSet& nu = sntree.get_snode(j);
            const VertexSet& alpha = sntree.get_separators(j);

            if (alpha.empty()) continue;

            int64_t i_rep = nu[0];

            std::vector<int64_t> eta;
            for (int64_t x = i_rep + 1; x < N; ++x) {
                if (!alpha.contains(x) && !nu.contains(x)) {
                    eta.push_back(x);
                }
            }

            if (eta.empty()) continue;

            int64_t nn = static_cast<int64_t>(nu.size());
            int64_t ne = static_cast<int64_t>(eta.size());

            // Detect a loss that depends on the structural-fill entries.
            constexpr double kFillGradTol = 1e-12;
            for (int64_t ei = 0; ei < ne; ++ei) {
                for (int64_t ni = 0; ni < nn; ++ni) {
                    if (std::abs(dW[eta[ei] * N + nu[ni]]) > kFillGradTol) {
                        throw std::runtime_error(
                            "adjoint_complete_z: loss depends on PSD-completion "
                            "fill-in entries, which is not supported (forward "
                            "Cholesky factors are not cached); cannot return "
                            "correct gradients");
                    }
                }
            }
        }

        // Adjoint of permute: dA[p[i], p[j]] += dW[i, j]
        std::fill(dA.begin(), dA.end(), 0.0);
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                dA[p[i] * N + p[j]] += dW[i * N + j];
            }
        }

        // Pack back to svec (upper triangle, column-major)
        {
            int64_t idx = 0;
            for (int64_t col = 0; col < N; ++col) {
                for (int64_t row = 0; row <= col; ++row) {
                    dz_cone[idx] = dA[row * N + col];
                    idx++;
                }
            }
        }
    }
}

void ChordalInfo::adjoint_augment_dA(const double* dA_aug, double* dA_orig) const {
    // Adjoint of A augmentation: gather entries at their mapped positions
    if (A_nnz_orig > 0) {
        for (int64_t k = 0; k < A_nnz_orig; ++k) {
            dA_orig[k] = dA_aug[A_orig_to_aug[k]];
        }
    }
}

} // namespace moreau
