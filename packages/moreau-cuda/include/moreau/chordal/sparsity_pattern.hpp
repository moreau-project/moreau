#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "symbolic_ldl.hpp"

// Forward declaration: SuperNodeTree is defined in supernode_tree.hpp
// (being created in parallel). Include it when available.
// #include "supernode_tree.hpp"

namespace moreau {
namespace chordal {

// Holds clique and sparsity data for a single PSD cone constraint
// that has been analyzed for chordal decomposition.
//
// This mirrors the Rust SparsityPattern struct in sparsity_pattern.rs.
// It wraps a SuperNodeTree (built from the L factor), the fill-reducing
// ordering, and the index of the original cone.
struct SparsityPattern {
    // The supernodal elimination tree built from the L factor.
    // Forward-declared; will be populated once supernode_tree.hpp is available.
    // For now we store the raw L factor data needed to construct it.
    struct LFactorData {
        std::vector<int64_t> colptr;   // CSC column pointers
        std::vector<int64_t> rowind;   // CSC row indices
        int64_t n;                     // dimension
    };

    LFactorData L_data;                // L factor for deferred SuperNodeTree construction
    std::vector<int64_t> ordering;     // fill-reducing permutation (maps reordered -> original)
    int64_t orig_index;                // index of the original cone being decomposed
    std::string merge_method;          // merge strategy used

    SparsityPattern() : orig_index(-1) {}

    // Construct a SparsityPattern from an nz_mask (boolean sparsity pattern
    // of a PSD cone's triangle), its dimension, original cone index, and
    // merge method string.
    //
    // This performs:
    //   1. Build upper-triangular CSC from the nz_mask
    //   2. Symbolic LDL factorization (AMD + elimination tree + L pattern)
    //   3. Store results for later SuperNodeTree construction
    //
    // nz_mask: boolean array of length conedim*(conedim+1)/2 indicating
    //          which entries of the upper-triangular vectorized matrix are nonzero.
    //          Indexed in column-major upper-triangular order:
    //          (0,0), (0,1), (1,1), (0,2), (1,2), (2,2), ...
    // conedim: the matrix dimension of the PSD cone
    // cone_index: original index of this cone in the problem's cone list
    // merge: merge method string ("none", "parent_child", "clique_graph")
    static SparsityPattern from_nz_mask(
        const bool* nz_mask,
        int64_t conedim,
        int64_t cone_index,
        const std::string& merge
    ) {
        int64_t tri_len = conedim * (conedim + 1) / 2;

        // Force diagonal entries to be marked (required for LDL)
        // The diagonal entry (i,i) is at triangular_index(i) = i*(i+1)/2 + i = i*(i+3)/2
        // In column-major upper triangular: entry for (row, col) with row<=col
        // is at col*(col+1)/2 + row.
        // So diagonal (i,i) is at i*(i+1)/2 + i.
        std::vector<bool> mask(nz_mask, nz_mask + tri_len);
        for (int64_t i = 0; i < conedim; ++i) {
            int64_t diag_idx = i * (i + 1) / 2 + i;
            mask[diag_idx] = true;
        }

        // Check if fully dense -- no decomposition possible
        bool all_nonzero = true;
        for (int64_t i = 0; i < tri_len; ++i) {
            if (!mask[i]) { all_nonzero = false; break; }
        }

        SparsityPattern pat;
        pat.orig_index = cone_index;
        pat.merge_method = merge;

        if (all_nonzero) {
            // Dense pattern: no decomposition possible.
            // Return with empty L_data to signal this.
            pat.L_data.n = 0;
            return pat;
        }

        // Build upper-triangular CSC from mask.
        // The mask is stored in column-major upper triangular order:
        // column col contains rows 0..col, starting at offset col*(col+1)/2.
        std::vector<int64_t> rows, cols;
        for (int64_t col = 0; col < conedim; ++col) {
            for (int64_t row = 0; row <= col; ++row) {
                int64_t idx = col * (col + 1) / 2 + row;
                if (mask[idx]) {
                    rows.push_back(row);
                    cols.push_back(col);
                }
            }
        }

        // Convert COO to CSC (already column-sorted)
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
        SymbolicLDLResult ldl = symbolic_ldl(colptr.data(), rowind.data(), conedim);

        pat.ordering = std::move(ldl.perm);
        pat.L_data.colptr = std::move(ldl.L_colptr);
        pat.L_data.rowind = std::move(ldl.L_rowind);
        pat.L_data.n = ldl.n;

        return pat;
    }

    // Returns true if this pattern represents a valid decomposition
    // (i.e., the cone was sparse enough to analyze)
    bool is_valid() const {
        return L_data.n > 0;
    }

    // Accessors for the L factor data (needed for SuperNodeTree construction)
    int64_t L_ncols() const { return L_data.n; }

    const std::vector<int64_t>& L_colptr() const { return L_data.colptr; }
    const std::vector<int64_t>& L_rowind() const { return L_data.rowind; }

    // Number of subdiagonal nonzeros in column j of L
    int64_t L_col_nnz(int64_t j) const {
        return L_data.colptr[j + 1] - L_data.colptr[j];
    }

    // Row indices in column j of L (subdiagonal entries)
    // Returns pointer to first row index and count
    const int64_t* L_col_rows(int64_t j) const {
        return L_data.rowind.data() + L_data.colptr[j];
    }

    // --------------------------------------------------------------------------
    // L-factor graph queries used by SuperNodeTree construction
    // --------------------------------------------------------------------------

    // Find parent of vertex v in the elimination tree implied by L.
    // parent(v) = min row index > v in column v of L.
    // Returns -1 if v is a root (last column or no subdiagonal entries).
    int64_t parent_from_L(int64_t v) const {
        if (v == L_data.n - 1) return -1;
        int64_t start = L_data.colptr[v];
        int64_t end = L_data.colptr[v + 1];
        if (start == end) return -1;
        // The first entry in column v should be the smallest row > v
        // (L is lower triangular, subdiagonal only)
        return L_data.rowind[start];
    }

    // Get all higher-order neighbors of vertex v from L.
    // These are all row indices in column v of L.
    std::vector<int64_t> higher_neighbors(int64_t v) const {
        int64_t start = L_data.colptr[v];
        int64_t end = L_data.colptr[v + 1];
        return std::vector<int64_t>(
            L_data.rowind.begin() + start,
            L_data.rowind.begin() + end
        );
    }

    // Degree of vertex v in L (number of subdiagonal entries in column v)
    int64_t higher_degree(int64_t v) const {
        return L_data.colptr[v + 1] - L_data.colptr[v];
    }
};

}  // namespace chordal
}  // namespace moreau
