#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace moreau {
namespace chordal {

// Result of symbolic LDL factorization.
// Contains only the sparsity pattern of L (no numerical values)
// and the fill-reducing permutation.
struct SymbolicLDLResult {
    std::vector<int64_t> L_colptr;  // CSC column pointers for L (size n+1)
    std::vector<int64_t> L_rowind;  // CSC row indices for L (subdiagonal only)
    std::vector<int64_t> perm;      // fill-reducing permutation
    int64_t n;                      // matrix dimension
};

namespace detail {

static constexpr int64_t UNKNOWN = -1;

// --------------------------------------------------------------------------
// AMD (Approximate Minimum Degree) ordering
// --------------------------------------------------------------------------

// Builds a full (symmetric) adjacency list from upper-triangular CSC input.
// The input is the upper triangle of a symmetric matrix in CSC format.
// Returns adjacency list as (adj_ptr, adj_list) where adj_ptr[i]..adj_ptr[i+1]
// gives the neighbors of vertex i (excluding self-loops).
inline void build_adjacency(
    const int64_t* colptr,
    const int64_t* rowind,
    int64_t n,
    std::vector<int64_t>& adj_ptr,
    std::vector<int64_t>& adj_list
) {
    // Count degree for each vertex (from upper triangle, both directions)
    std::vector<int64_t> degree(n, 0);
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t p = colptr[j]; p < colptr[j + 1]; ++p) {
            int64_t i = rowind[p];
            if (i != j) {  // skip diagonal
                degree[i]++;
                degree[j]++;
            }
        }
    }

    // Build adj_ptr as cumulative sum
    adj_ptr.resize(n + 1);
    adj_ptr[0] = 0;
    for (int64_t i = 0; i < n; ++i) {
        adj_ptr[i + 1] = adj_ptr[i] + degree[i];
    }

    // Fill adjacency list
    adj_list.resize(adj_ptr[n]);
    std::vector<int64_t> pos(adj_ptr.begin(), adj_ptr.begin() + n);

    for (int64_t j = 0; j < n; ++j) {
        for (int64_t p = colptr[j]; p < colptr[j + 1]; ++p) {
            int64_t i = rowind[p];
            if (i != j) {
                adj_list[pos[i]++] = j;
                adj_list[pos[j]++] = i;
            }
        }
    }
}

// Simple minimum-degree ordering.
// For each step, eliminates the node with minimum degree in the
// remaining graph, updating the degrees of its neighbors.
// Uses an external-degree approximation (not exact AMD, but sufficient
// for sparsity detection in chordal decomposition).
inline void amd_ordering(
    const int64_t* colptr,
    const int64_t* rowind,
    int64_t n,
    std::vector<int64_t>& perm,
    std::vector<int64_t>& iperm
) {
    perm.resize(n);
    iperm.resize(n);

    if (n == 0) return;

    // Build full adjacency structure
    std::vector<int64_t> adj_ptr, adj_list;
    build_adjacency(colptr, rowind, n, adj_ptr, adj_list);

    // Track current degree of each node and whether it's been eliminated
    std::vector<int64_t> degree(n);
    for (int64_t i = 0; i < n; ++i) {
        degree[i] = adj_ptr[i + 1] - adj_ptr[i];
    }
    std::vector<bool> eliminated(n, false);

    // For each node, maintain a dynamic neighbor set.
    // We use sorted vectors for simplicity since these matrices are small
    // (PSD cone dimensions, not the full problem).
    std::vector<std::vector<int64_t>> neighbors(n);
    for (int64_t i = 0; i < n; ++i) {
        neighbors[i].reserve(degree[i]);
        for (int64_t p = adj_ptr[i]; p < adj_ptr[i + 1]; ++p) {
            neighbors[i].push_back(adj_list[p]);
        }
        std::sort(neighbors[i].begin(), neighbors[i].end());
    }

    for (int64_t step = 0; step < n; ++step) {
        // Find non-eliminated node with minimum degree
        int64_t min_deg = n + 1;
        int64_t pivot = -1;
        for (int64_t i = 0; i < n; ++i) {
            if (!eliminated[i] && degree[i] < min_deg) {
                min_deg = degree[i];
                pivot = i;
            }
        }
        assert(pivot >= 0);

        perm[step] = pivot;
        iperm[pivot] = step;
        eliminated[pivot] = true;

        // Get active neighbors of pivot
        std::vector<int64_t> active_nbrs;
        active_nbrs.reserve(neighbors[pivot].size());
        for (int64_t nb : neighbors[pivot]) {
            if (!eliminated[nb]) {
                active_nbrs.push_back(nb);
            }
        }

        // Add edges between all pairs of active neighbors (fill-in).
        // This is the key step: when we eliminate a node, all its
        // neighbors become connected (clique formation).
        for (size_t a = 0; a < active_nbrs.size(); ++a) {
            int64_t u = active_nbrs[a];
            // Remove pivot from u's neighbor list
            auto& nu = neighbors[u];
            nu.erase(std::remove(nu.begin(), nu.end(), pivot), nu.end());

            for (size_t b = a + 1; b < active_nbrs.size(); ++b) {
                int64_t v = active_nbrs[b];
                // Add edge u-v if not already present
                if (!std::binary_search(nu.begin(), nu.end(), v)) {
                    // Insert maintaining sorted order
                    nu.insert(std::lower_bound(nu.begin(), nu.end(), v), v);
                    auto& nv = neighbors[v];
                    nv.insert(std::lower_bound(nv.begin(), nv.end(), u), u);
                }
            }
            degree[u] = static_cast<int64_t>(nu.size());
        }

        // Active neighbors handled above; eliminated nodes need no update.
        degree[pivot] = 0;
        neighbors[pivot].clear();
    }
}

// --------------------------------------------------------------------------
// Permute symmetric matrix (upper triangular CSC -> upper triangular CSC)
// --------------------------------------------------------------------------

// Given upper-triangular CSC matrix A and inverse permutation iperm,
// compute P = iperm' * A * iperm in upper triangular CSC form.
inline void permute_symmetric(
    const int64_t* Ac,    // colptr, size n+1
    const int64_t* Ar,    // rowind
    int64_t n,
    const int64_t* iperm, // inverse permutation
    std::vector<int64_t>& Pc,  // output colptr
    std::vector<int64_t>& Pr   // output rowind
) {
    int64_t nnz = Ac[n];
    Pc.resize(n + 1, 0);
    Pr.resize(nnz);

    // Count entries per column of permuted matrix
    std::vector<int64_t> count(n, 0);
    for (int64_t colA = 0; colA < n; ++colA) {
        int64_t colP = iperm[colA];
        for (int64_t p = Ac[colA]; p < Ac[colA + 1]; ++p) {
            int64_t rowA = Ar[p];
            if (rowA <= colA) {  // upper triangular
                int64_t rowP = iperm[rowA];
                int64_t col_idx = std::max(rowP, colP);
                count[col_idx]++;
            }
        }
    }

    // Cumulative sum for colptr
    Pc[0] = 0;
    for (int64_t i = 0; i < n; ++i) {
        Pc[i + 1] = Pc[i] + count[i];
    }

    // Fill row indices
    std::vector<int64_t> pos(Pc.begin(), Pc.begin() + n);
    for (int64_t colA = 0; colA < n; ++colA) {
        int64_t colP = iperm[colA];
        for (int64_t p = Ac[colA]; p < Ac[colA + 1]; ++p) {
            int64_t rowA = Ar[p];
            if (rowA <= colA) {
                int64_t rowP = iperm[rowA];
                int64_t col_idx = std::max(colP, rowP);
                int64_t row_idx = std::min(colP, rowP);
                Pr[pos[col_idx]++] = row_idx;
            }
        }
    }

    // Sort row indices within each column (needed for etree computation)
    for (int64_t j = 0; j < n; ++j) {
        std::sort(Pr.begin() + Pc[j], Pr.begin() + Pc[j + 1]);
    }
}

// --------------------------------------------------------------------------
// Elimination tree computation
// --------------------------------------------------------------------------

// Compute the elimination tree and column counts for L.
// etree[i] = parent of i in elimination tree (UNKNOWN if root)
// Lnz[i] = number of nonzeros in column i of L (subdiagonal)
//
// Input: upper triangular CSC matrix (colptr Ap, rowind Ai, dimension n)
// This is the QDLDL etree algorithm.
inline void compute_etree(
    int64_t n,
    const int64_t* Ap,
    const int64_t* Ai,
    std::vector<int64_t>& etree,
    std::vector<int64_t>& Lnz
) {
    etree.assign(n, UNKNOWN);
    Lnz.assign(n, 0);
    std::vector<int64_t> work(n);

    for (int64_t j = 0; j < n; ++j) {
        work[j] = j;
        for (int64_t p = Ap[j]; p < Ap[j + 1]; ++p) {
            int64_t i = Ai[p];
            // Walk up the tree from i until we reach j or a node
            // already marked as visited for column j
            while (work[i] != j) {
                if (etree[i] == UNKNOWN) {
                    etree[i] = j;
                }
                Lnz[i]++;
                work[i] = j;
                i = etree[i];
            }
        }
    }
}

// --------------------------------------------------------------------------
// Symbolic factorization: compute the sparsity pattern of L
// --------------------------------------------------------------------------

// Given the elimination tree and column nonzero counts, compute the
// CSC structure of L (column pointers and row indices, no values).
//
// This mirrors the QDLDL _factor_inner logic but only tracks which
// rows appear in each column (symbolic/logical factorization).
inline void symbolic_factor(
    int64_t n,
    const int64_t* Ap,     // upper triangular input colptr
    const int64_t* Ai,     // upper triangular input rowind
    const int64_t* etree,  // elimination tree
    const int64_t* Lnz,    // column counts for L
    std::vector<int64_t>& Lp,  // output L colptr
    std::vector<int64_t>& Li   // output L rowind
) {
    // Compute L column pointers from column counts
    Lp.resize(n + 1);
    Lp[0] = 0;
    for (int64_t j = 0; j < n; ++j) {
        Lp[j + 1] = Lp[j] + Lnz[j];
    }

    int64_t nnzL = Lp[n];
    Li.resize(nnzL);

    // Working arrays
    std::vector<bool> y_markers(n, false);
    std::vector<int64_t> y_idx(n);
    std::vector<int64_t> elim_buffer(n);
    std::vector<int64_t> next_colspace(Lp.begin(), Lp.begin() + n);

    for (int64_t k = 1; k < n; ++k) {
        int64_t nnz_y = 0;

        // Determine nonzero pattern in row k of L
        for (int64_t p = Ap[k]; p < Ap[k + 1]; ++p) {
            int64_t bidx = Ai[p];
            if (bidx == k) continue;  // diagonal entry, skip

            if (!y_markers[bidx]) {
                y_markers[bidx] = true;
                elim_buffer[0] = bidx;
                int64_t nnz_e = 1;

                int64_t next_idx = etree[bidx];
                while (next_idx != UNKNOWN && next_idx < k) {
                    if (y_markers[next_idx]) break;
                    y_markers[next_idx] = true;
                    elim_buffer[nnz_e] = next_idx;
                    next_idx = etree[next_idx];
                    nnz_e++;
                }

                // Put elimination list in reverse order
                while (nnz_e > 0) {
                    nnz_e--;
                    y_idx[nnz_y++] = elim_buffer[nnz_e];
                }
            }
        }

        // Place row indices in L
        for (int64_t i = nnz_y - 1; i >= 0; --i) {
            int64_t cidx = y_idx[i];
            int64_t pos = next_colspace[cidx];
            Li[pos] = k;
            next_colspace[cidx]++;
            y_markers[cidx] = false;
        }
    }
}

// --------------------------------------------------------------------------
// Connect graph: ensure L represents a connected adjacency
// --------------------------------------------------------------------------

// If L has disconnected blocks (a column j with no subdiagonal entries
// and j < n-1), add an entry at (j+1, j) to connect the blocks.
// This matches the Rust connect_graph function.
inline void connect_graph(
    int64_t n,
    std::vector<int64_t>& Lp,
    std::vector<int64_t>& Li
) {
    // First pass: identify columns that need connection
    std::vector<int64_t> cols_to_connect;
    for (int64_t j = 0; j < n - 1; ++j) {
        bool connected = false;
        for (int64_t p = Lp[j]; p < Lp[j + 1]; ++p) {
            if (Li[p] > j) {
                connected = true;
                break;
            }
        }
        if (!connected) {
            cols_to_connect.push_back(j);
        }
    }

    if (cols_to_connect.empty()) return;

    // Build new L with extra entries
    int64_t old_nnz = Lp[n];
    int64_t extra = static_cast<int64_t>(cols_to_connect.size());
    std::vector<int64_t> new_Lp(n + 1);
    std::vector<int64_t> new_Li(old_nnz + extra);

    // Track which columns need an extra entry
    std::vector<bool> needs_connect(n, false);
    for (int64_t j : cols_to_connect) {
        needs_connect[j] = true;
    }

    new_Lp[0] = 0;
    int64_t write_pos = 0;
    for (int64_t j = 0; j < n; ++j) {
        // Copy existing entries for this column
        for (int64_t p = Lp[j]; p < Lp[j + 1]; ++p) {
            new_Li[write_pos++] = Li[p];
        }
        // Add connecting entry if needed
        if (needs_connect[j]) {
            new_Li[write_pos++] = j + 1;
        }
        new_Lp[j + 1] = write_pos;
    }

    Lp = std::move(new_Lp);
    Li = std::move(new_Li);
}

}  // namespace detail

// --------------------------------------------------------------------------
// Main entry point: symbolic LDL factorization
// --------------------------------------------------------------------------

// Perform symbolic LDL factorization of a symmetric matrix given in upper
// triangular CSC format. Returns the sparsity pattern of L and a
// fill-reducing (AMD) permutation.
//
// Parameters:
//   colptr  - CSC column pointers (size n+1)
//   rowind  - CSC row indices (upper triangular, rows <= col)
//   n       - matrix dimension
//
// The returned L is lower triangular with only subdiagonal entries stored
// (no diagonal, since L has implicit unit diagonal in LDL^T).
inline SymbolicLDLResult symbolic_ldl(
    const int64_t* colptr,
    const int64_t* rowind,
    int64_t n
) {
    SymbolicLDLResult result;
    result.n = n;

    if (n == 0) {
        result.L_colptr = {0};
        return result;
    }

    if (n == 1) {
        result.L_colptr = {0, 0};
        result.perm = {0};
        return result;
    }

    // Step 1: Compute AMD fill-reducing ordering
    std::vector<int64_t> iperm;
    detail::amd_ordering(colptr, rowind, n, result.perm, iperm);

    // Step 2: Permute matrix to upper triangular form under new ordering
    std::vector<int64_t> perm_colptr, perm_rowind;
    detail::permute_symmetric(colptr, rowind, n, iperm.data(),
                              perm_colptr, perm_rowind);

    // Step 3: Compute elimination tree and column counts
    std::vector<int64_t> etree, Lnz;
    detail::compute_etree(n, perm_colptr.data(), perm_rowind.data(), etree, Lnz);

    // Step 4: Symbolic factorization to get L pattern
    detail::symbolic_factor(n, perm_colptr.data(), perm_rowind.data(),
                            etree.data(), Lnz.data(),
                            result.L_colptr, result.L_rowind);

    // Step 5: Connect disconnected blocks in L
    detail::connect_graph(n, result.L_colptr, result.L_rowind);

    return result;
}

}  // namespace chordal
}  // namespace moreau
