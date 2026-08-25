#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace moreau {
namespace chordal {

// ---------------------------------------------------------------------------
// Forward declaration: SuperNodeTree is defined in supernode_tree.hpp.
// The merge strategies only need a pointer/reference to it.
// ---------------------------------------------------------------------------
struct SuperNodeTree;

// ---------------------------------------------------------------------------
// Constants matching the Rust implementation
// ---------------------------------------------------------------------------
static constexpr int64_t NO_PARENT = std::numeric_limits<int64_t>::max();
static constexpr int64_t INACTIVE_NODE = std::numeric_limits<int64_t>::max() - 1;

// ---------------------------------------------------------------------------
// VertexSet — ordered set preserving insertion order.
// We use std::vector<int64_t> with uniqueness checks for a faithful port of
// the Rust IndexSet<usize>. Where set operations are performance-critical we
// also maintain an auxiliary hash set for O(1) lookups.
// ---------------------------------------------------------------------------
class VertexSet {
public:
    VertexSet() = default;

    void insert(int64_t v) {
        if (lookup_.insert(v).second) {
            order_.push_back(v);
        }
    }

    void shift_remove(int64_t v) {
        if (lookup_.erase(v)) {
            order_.erase(std::remove(order_.begin(), order_.end(), v), order_.end());
        }
    }

    bool contains(int64_t v) const { return lookup_.count(v) != 0; }

    size_t len() const { return order_.size(); }
    bool is_empty() const { return order_.empty(); }

    void clear() {
        order_.clear();
        lookup_.clear();
    }

    void sort() { std::sort(order_.begin(), order_.end()); }

    // Iteration
    using iterator = std::vector<int64_t>::iterator;
    using const_iterator = std::vector<int64_t>::const_iterator;
    iterator begin() { return order_.begin(); }
    iterator end() { return order_.end(); }
    const_iterator begin() const { return order_.begin(); }
    const_iterator end() const { return order_.end(); }

    // Check if this set is a subset of `other`
    bool is_subset(const VertexSet& other) const {
        for (int64_t v : order_) {
            if (!other.contains(v)) return false;
        }
        return true;
    }

    // Elements in `this ∩ other`
    VertexSet intersection(const VertexSet& other) const {
        VertexSet result;
        for (int64_t v : order_) {
            if (other.contains(v)) result.insert(v);
        }
        return result;
    }

    bool operator==(const VertexSet& other) const {
        if (len() != other.len()) return false;
        for (int64_t v : order_) {
            if (!other.contains(v)) return false;
        }
        return true;
    }
    bool operator!=(const VertexSet& other) const { return !(*this == other); }

    // Extend from another container
    template <typename Iter>
    void extend(Iter first, Iter last) {
        for (; first != last; ++first) insert(*first);
    }
    void extend(const VertexSet& other) { extend(other.begin(), other.end()); }

    const std::vector<int64_t>& data() const { return order_; }

private:
    std::vector<int64_t> order_;
    std::unordered_set<int64_t> lookup_;
};

// ---------------------------------------------------------------------------
// SimpleCsc — minimal CSC sparse matrix with integer values
// ---------------------------------------------------------------------------
struct SimpleCsc {
    std::vector<int64_t> colptr;   // length ncol+1
    std::vector<int64_t> rowind;   // length nnz
    std::vector<int64_t> values;   // length nnz
    int64_t nrow = 0;
    int64_t ncol = 0;

    SimpleCsc() = default;

    SimpleCsc(int64_t nrow, int64_t ncol)
        : colptr(static_cast<size_t>(ncol + 1), 0), nrow(nrow), ncol(ncol) {}

    // Build from COO triplets. Duplicates at the same (row, col) are overwritten
    // (last value wins). Assumes row > col (lower triangular).
    static SimpleCsc from_triplets(int64_t nrow, int64_t ncol,
                                   const std::vector<int64_t>& rows,
                                   const std::vector<int64_t>& cols,
                                   const std::vector<int64_t>& vals) {
        SimpleCsc m;
        m.nrow = nrow;
        m.ncol = ncol;

        // Deduplicate: keep last value for each (row, col)
        struct Entry { int64_t row, col, val; };
        std::vector<Entry> entries;
        entries.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            entries.push_back({rows[i], cols[i], vals[i]});
        }

        // Sort by (col, row)
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            return a.col < b.col || (a.col == b.col && a.row < b.row);
        });

        // Deduplicate (keep last)
        std::vector<Entry> unique;
        unique.reserve(entries.size());
        for (size_t i = 0; i < entries.size(); ++i) {
            if (!unique.empty() && unique.back().row == entries[i].row &&
                unique.back().col == entries[i].col) {
                unique.back().val = entries[i].val;
            } else {
                unique.push_back(entries[i]);
            }
        }

        m.colptr.assign(static_cast<size_t>(ncol + 1), 0);
        m.rowind.resize(unique.size());
        m.values.resize(unique.size());

        for (size_t i = 0; i < unique.size(); ++i) {
            m.colptr[static_cast<size_t>(unique[i].col + 1)]++;
        }
        for (int64_t c = 0; c < ncol; ++c) {
            m.colptr[static_cast<size_t>(c + 1)] += m.colptr[static_cast<size_t>(c)];
        }

        // Fill row indices and values
        std::vector<int64_t> cursor(m.colptr.begin(), m.colptr.end());
        for (size_t i = 0; i < unique.size(); ++i) {
            int64_t col = unique[i].col;
            int64_t pos = cursor[static_cast<size_t>(col)]++;
            m.rowind[static_cast<size_t>(pos)] = unique[i].row;
            m.values[static_cast<size_t>(pos)] = unique[i].val;
        }

        return m;
    }

    int64_t nnz() const { return static_cast<int64_t>(rowind.size()); }

    // Get value at (row, col), returns std::nullopt if not present
    std::optional<int64_t> get_entry(int64_t row, int64_t col) const {
        if (col < 0 || col >= ncol) return std::nullopt;
        int64_t start = colptr[static_cast<size_t>(col)];
        int64_t end = colptr[static_cast<size_t>(col + 1)];
        for (int64_t i = start; i < end; ++i) {
            if (rowind[static_cast<size_t>(i)] == row) {
                return values[static_cast<size_t>(i)];
            }
        }
        return std::nullopt;
    }

    // Set value at (row, col). If the entry exists, update it. If not, insert.
    void set_entry(int64_t row, int64_t col, int64_t val) {
        if (col < 0 || col >= ncol) return;
        int64_t start = colptr[static_cast<size_t>(col)];
        int64_t end = colptr[static_cast<size_t>(col + 1)];
        for (int64_t i = start; i < end; ++i) {
            if (rowind[static_cast<size_t>(i)] == row) {
                values[static_cast<size_t>(i)] = val;
                return;
            }
        }
        // Insert new entry: we need to shift everything
        auto pos = static_cast<size_t>(end); // insert at end of this column
        rowind.insert(rowind.begin() + static_cast<ptrdiff_t>(pos), row);
        values.insert(values.begin() + static_cast<ptrdiff_t>(pos), val);
        for (int64_t c = col + 1; c <= ncol; ++c) {
            colptr[static_cast<size_t>(c)]++;
        }
    }

    // Remove all entries with value == 0
    void dropzeros() {
        std::vector<int64_t> new_rowind;
        std::vector<int64_t> new_values;
        new_rowind.reserve(rowind.size());
        new_values.reserve(values.size());

        std::vector<int64_t> new_colptr(static_cast<size_t>(ncol + 1), 0);

        for (int64_t c = 0; c < ncol; ++c) {
            new_colptr[static_cast<size_t>(c)] = static_cast<int64_t>(new_rowind.size());
            int64_t start = colptr[static_cast<size_t>(c)];
            int64_t end = colptr[static_cast<size_t>(c + 1)];
            for (int64_t i = start; i < end; ++i) {
                if (values[static_cast<size_t>(i)] != 0) {
                    new_rowind.push_back(rowind[static_cast<size_t>(i)]);
                    new_values.push_back(values[static_cast<size_t>(i)]);
                }
            }
        }
        new_colptr[static_cast<size_t>(ncol)] = static_cast<int64_t>(new_rowind.size());

        colptr = std::move(new_colptr);
        rowind = std::move(new_rowind);
        values = std::move(new_values);
    }

    // Convert linear nzval index to (row, col) coordinate
    std::pair<int64_t, int64_t> index_to_coord(int64_t idx) const {
        int64_t row = rowind[static_cast<size_t>(idx)];
        int64_t col = 0;
        for (int64_t c = 0; c < ncol; ++c) {
            if (idx >= colptr[static_cast<size_t>(c)] && idx < colptr[static_cast<size_t>(c + 1)]) {
                col = c;
                break;
            }
        }
        return {row, col};
    }

    // Find all nonzero entries as (rows, cols, vals)
    void findnz(std::vector<int64_t>& rows, std::vector<int64_t>& cols,
                 std::vector<int64_t>& vals) const {
        rows.clear(); cols.clear(); vals.clear();
        rows.reserve(rowind.size());
        cols.reserve(rowind.size());
        vals.reserve(rowind.size());
        for (int64_t c = 0; c < ncol; ++c) {
            int64_t start = colptr[static_cast<size_t>(c)];
            int64_t end = colptr[static_cast<size_t>(c + 1)];
            for (int64_t i = start; i < end; ++i) {
                rows.push_back(rowind[static_cast<size_t>(i)]);
                cols.push_back(c);
                vals.push_back(values[static_cast<size_t>(i)]);
            }
        }
    }
};

// ---------------------------------------------------------------------------
// SuperNodeTree definition — the merge strategies operate on this structure.
// This must match the fields used by the Rust SuperNodeTree.
// ---------------------------------------------------------------------------
struct SuperNodeTree {
    std::vector<VertexSet> snode;           // vertices of supernodes (cliques)
    std::vector<int64_t> snode_post;        // post-order of supernodal elimination tree
    std::vector<int64_t> snode_parent;      // parent of each supernode
    std::vector<VertexSet> snode_children;  // children of each supernode
    std::vector<int64_t> post;              // post-ordering of vertices in elim tree
    std::vector<VertexSet> separators;      // vertices of clique separators
    std::vector<int64_t> nblk;              // block sizes (populated after merge)
    bool nblk_valid = false;                // whether nblk has been populated
    int64_t n_cliques = 0;                  // number of nonempty cliques
};

// ---------------------------------------------------------------------------
// DisjointSetUnion — union-find with path compression and union by rank
// ---------------------------------------------------------------------------
class DisjointSetUnion {
public:
    explicit DisjointSetUnion(int64_t n)
        : parents_(static_cast<size_t>(n)), ranks_(static_cast<size_t>(n), 0) {
        std::iota(parents_.begin(), parents_.end(), int64_t(0));
    }

    void unite(int64_t x, int64_t y) {
        int64_t r = root(x);
        int64_t s = root(y);
        if (r == s) return;
        if (ranks_[static_cast<size_t>(r)] > ranks_[static_cast<size_t>(s)]) {
            parents_[static_cast<size_t>(s)] = r;
        } else if (ranks_[static_cast<size_t>(r)] < ranks_[static_cast<size_t>(s)]) {
            parents_[static_cast<size_t>(r)] = s;
        } else {
            parents_[static_cast<size_t>(r)] = s;
            ranks_[static_cast<size_t>(s)]++;
        }
    }

    bool in_same_set(int64_t x, int64_t y) {
        return root(x) == root(y);
    }

private:
    int64_t root(int64_t x) {
        auto ux = static_cast<size_t>(x);
        while (parents_[ux] != x) {
            parents_[ux] = parents_[static_cast<size_t>(parents_[ux])]; // path compression
            x = parents_[ux];
            ux = static_cast<size_t>(x);
        }
        return x;
    }

    std::vector<int64_t> parents_;
    std::vector<int64_t> ranks_;
};

// ---------------------------------------------------------------------------
// Edge weight method enum
// ---------------------------------------------------------------------------
enum class EdgeWeightMethod {
    Cubic = 1,
};

// ---------------------------------------------------------------------------
// MergeStrategy — abstract base class (maps to the Rust trait)
// ---------------------------------------------------------------------------
class MergeStrategy {
public:
    virtual ~MergeStrategy() = default;

    // Default merge loop: initialise, traverse, evaluate, merge, update, post-process
    virtual void merge_cliques(SuperNodeTree& t) {
        initialise(t);

        while (!is_done()) {
            auto cand = traverse(t);
            if (!cand.has_value()) break;

            bool do_merge = evaluate(t, *cand);
            if (do_merge) {
                merge_two_cliques(t, *cand);
            }

            update_strategy(t, *cand, do_merge);

            if (t.n_cliques == 1) break;
        }
        post_process_merge(t);
    }

    virtual void initialise(SuperNodeTree& t) = 0;
    virtual bool is_done() const = 0;
    virtual std::optional<std::pair<int64_t, int64_t>> traverse(SuperNodeTree& t) = 0;
    virtual bool evaluate(SuperNodeTree& t, std::pair<int64_t, int64_t> cand) = 0;
    virtual void merge_two_cliques(SuperNodeTree& t, std::pair<int64_t, int64_t> cand) = 0;
    virtual void update_strategy(SuperNodeTree& t, std::pair<int64_t, int64_t> cand, bool do_merge) = 0;
    virtual void post_process_merge(SuperNodeTree& t) = 0;
};

// ---------------------------------------------------------------------------
// NoMergeStrategy — trivial strategy that does nothing
// ---------------------------------------------------------------------------
class NoMergeStrategy : public MergeStrategy {
public:
    void initialise(SuperNodeTree& /*t*/) override {}
    bool is_done() const override { return true; }

    std::optional<std::pair<int64_t, int64_t>> traverse(SuperNodeTree& /*t*/) override {
        // Should never be called
        assert(false && "NoMergeStrategy::traverse should never be called");
        return std::nullopt;
    }

    bool evaluate(SuperNodeTree& /*t*/, std::pair<int64_t, int64_t> /*cand*/) override {
        assert(false && "NoMergeStrategy::evaluate should never be called");
        return false;
    }

    void merge_two_cliques(SuperNodeTree& /*t*/, std::pair<int64_t, int64_t> /*cand*/) override {
        assert(false && "NoMergeStrategy::merge_two_cliques should never be called");
    }

    void update_strategy(SuperNodeTree& /*t*/, std::pair<int64_t, int64_t> /*cand*/,
                          bool /*do_merge*/) override {
        assert(false && "NoMergeStrategy::update_strategy should never be called");
    }

    void post_process_merge(SuperNodeTree& /*t*/) override {}
};

// ---------------------------------------------------------------------------
// Internal helper utilities (in anonymous namespace to keep them file-local)
// ---------------------------------------------------------------------------
namespace detail {

// Union sets[c1] = sets[c1] ∪ sets[c2], handling the fact that c1 and c2
// index the same vector.
inline void set_union_into_indexed(std::vector<VertexSet>& sets, int64_t c1, int64_t c2) {
    if (c1 == c2) return;
    // Collect c2 elements first (avoid aliasing issues)
    std::vector<int64_t> elems(sets[static_cast<size_t>(c2)].begin(),
                                sets[static_cast<size_t>(c2)].end());
    for (int64_t v : elems) {
        sets[static_cast<size_t>(c1)].insert(v);
    }
}

// Return the number of elements in s1 ∩ s2.
inline int64_t intersect_dim(const VertexSet& s1, const VertexSet& s2) {
    const VertexSet& sa = (s1.len() < s2.len()) ? s1 : s2;
    const VertexSet& sb = (s1.len() < s2.len()) ? s2 : s1;

    int64_t dim = 0;
    for (int64_t e : sa) {
        if (sb.contains(e)) dim++;
    }
    return dim;
}

// Return the size of s1 ∪ s2 (assuming elements are unique within each set).
inline int64_t union_dim(const VertexSet& s1, const VertexSet& s2) {
    return static_cast<int64_t>(s1.len()) + static_cast<int64_t>(s2.len()) - intersect_dim(s1, s2);
}

// Edge metric for merge decision. Currently only cubic.
inline int64_t edge_metric(const VertexSet& c_a, const VertexSet& c_b, EdgeWeightMethod method) {
    int64_t n1 = static_cast<int64_t>(c_a.len());
    int64_t n2 = static_cast<int64_t>(c_b.len());
    int64_t nm = union_dim(c_a, c_b);

    switch (method) {
        case EdgeWeightMethod::Cubic:
            return n1 * n1 * n1 + n2 * n2 * n2 - nm * nm * nm;
    }
    return 0; // unreachable
}

// Check if s1 ∩ s2 == s3.
inline bool inter_equal(const VertexSet& s1, const VertexSet& s2, const VertexSet& s3) {
    int64_t dim = 0;
    int64_t len_s1 = static_cast<int64_t>(s1.len());
    int64_t len_s2 = static_cast<int64_t>(s2.len());
    int64_t len_s3 = static_cast<int64_t>(s3.len());

    int64_t max_intersect = len_s1 + len_s2;
    if (max_intersect < len_s3) return false;

    const VertexSet& sa = (len_s1 < len_s2) ? s1 : s2;
    const VertexSet& sb = (len_s1 < len_s2) ? s2 : s1;

    for (int64_t e : sa) {
        if (sb.contains(e)) {
            dim++;
            if (dim > len_s3) return false;
            if (!s3.contains(e)) return false;
        }
        max_intersect--;
        if (max_intersect < len_s3) return false;
    }
    return dim == len_s3;
}

// DFS on a hash-table graph
inline void dfs_hashtable(VertexSet& component, int64_t v,
                           std::unordered_map<int64_t, bool>& visited,
                           const std::unordered_map<int64_t, std::vector<int64_t>>& H) {
    visited[v] = true;
    component.insert(v);
    auto it = H.find(v);
    if (it != H.end()) {
        for (int64_t n : it->second) {
            if (!visited[n]) {
                dfs_hashtable(component, n, visited, H);
            }
        }
    }
}

// Build the separator graph H for a given separator and subset of cliques.
// Returns a hash-table adjacency list.
inline std::unordered_map<int64_t, std::vector<int64_t>> separator_graph(
    const std::vector<int64_t>& clique_ind,
    const VertexSet& separator,
    const std::vector<VertexSet>& snd) {

    std::unordered_map<int64_t, std::vector<int64_t>> H;
    size_t nindex = clique_ind.size();

    for (size_t i = 0; i < nindex; ++i) {
        for (size_t j = i + 1; j < nindex; ++j) {
            int64_t ca = clique_ind[i];
            int64_t cb = clique_ind[j];
            // If intersection of snd[ca] and snd[cb] is NOT equal to separator,
            // then they are connected in the separator graph.
            if (!inter_equal(snd[static_cast<size_t>(ca)], snd[static_cast<size_t>(cb)], separator)) {
                H[ca].push_back(cb);
                H[cb].push_back(ca);
            }
        }
    }

    // Add unconnected cliques (ensure all clique_ind entries are in H)
    for (int64_t v : clique_ind) {
        if (H.find(v) == H.end()) {
            H[v] = {};
        }
    }
    return H;
}

// Find connected components in the separator graph H.
inline std::vector<VertexSet> find_components(
    const std::unordered_map<int64_t, std::vector<int64_t>>& H,
    const std::vector<int64_t>& clique_ind) {

    std::unordered_map<int64_t, bool> visited;
    for (int64_t v : clique_ind) {
        visited[v] = false;
    }

    std::vector<VertexSet> components;
    for (int64_t v : clique_ind) {
        if (!visited[v]) {
            VertexSet component;
            dfs_hashtable(component, v, visited, H);
            components.push_back(std::move(component));
        }
    }
    return components;
}

// Check whether a pair of cliques are in different connected components.
inline bool is_unconnected(std::pair<int64_t, int64_t> pair,
                            const std::vector<VertexSet>& components) {
    for (const auto& comp : components) {
        if (comp.contains(pair.first)) {
            return !comp.contains(pair.second);
        }
    }
    return true; // should not reach here
}

// Compute the reduced clique graph (Habib-Stacho algorithm).
// Returns (rows, cols) of edges in the reduced clique graph.
inline void compute_reduced_clique_graph(
    std::vector<VertexSet>& separators,
    const std::vector<VertexSet>& snode,
    std::vector<int64_t>& rows,
    std::vector<int64_t>& cols) {

    rows.clear();
    cols.clear();

    // Sort separators by decreasing cardinality
    std::sort(separators.begin(), separators.end(),
              [](const VertexSet& a, const VertexSet& b) {
                  return a.len() > b.len();
              });

    for (const auto& separator : separators) {
        // Find cliques that contain the separator
        std::vector<int64_t> clique_indices;
        for (size_t i = 0; i < snode.size(); ++i) {
            if (separator.is_subset(snode[i])) {
                clique_indices.push_back(static_cast<int64_t>(i));
            }
        }

        // Compute the separator graph
        auto H = separator_graph(clique_indices, separator, snode);

        // Find connected components
        auto components = find_components(H, clique_indices);

        // For each pair of cliques that contain the separator, add an edge
        // if they are in different connected components
        size_t nc = clique_indices.size();
        for (size_t i = 0; i < nc; ++i) {
            for (size_t j = i + 1; j < nc; ++j) {
                auto pair = std::make_pair(clique_indices[i], clique_indices[j]);
                if (is_unconnected(pair, components)) {
                    rows.push_back(std::max(pair.first, pair.second));
                    cols.push_back(std::min(pair.first, pair.second));
                }
            }
        }
    }
}

// Compute edge weights for all edges specified by (rows, cols).
inline std::vector<int64_t> compute_weights(
    const std::vector<int64_t>& rows,
    const std::vector<int64_t>& cols,
    const std::vector<VertexSet>& snode,
    EdgeWeightMethod method) {

    std::vector<int64_t> weights(rows.size(), 0);
    for (size_t k = 0; k < rows.size(); ++k) {
        weights[k] = edge_metric(snode[static_cast<size_t>(rows[k])],
                                  snode[static_cast<size_t>(cols[k])], method);
    }
    return weights;
}

// Compute adjacency table from CSC edge matrix.
inline std::unordered_map<int64_t, VertexSet> compute_adjacency_table(
    const SimpleCsc& edges, int64_t num_vertices) {

    std::unordered_map<int64_t, VertexSet> table;
    for (int64_t i = 0; i < num_vertices; ++i) {
        table[i] = VertexSet();
    }

    for (int64_t col = 0; col < num_vertices; ++col) {
        int64_t start = edges.colptr[static_cast<size_t>(col)];
        int64_t end = edges.colptr[static_cast<size_t>(col + 1)];
        for (int64_t j = start; j < end; ++j) {
            int64_t row = edges.rowind[static_cast<size_t>(j)];
            table[row].insert(col);
            table[col].insert(row);
        }
    }
    return table;
}

// Check whether an edge is permissible for merging.
// An edge (c1, c2) is permissible if for every common neighbor N,
// c1 ∩ N == c2 ∩ N, or if no common neighbors exist.
inline bool is_permissible(
    std::pair<int64_t, int64_t> edge,
    const std::unordered_map<int64_t, VertexSet>& adjacency_table,
    const std::vector<VertexSet>& snode) {

    int64_t c1 = edge.first;
    int64_t c2 = edge.second;

    const VertexSet& adj1 = adjacency_table.at(c1);
    const VertexSet& adj2 = adjacency_table.at(c2);

    // Find common neighbors
    VertexSet common = adj1.intersection(adj2);

    for (int64_t neighbor : common) {
        VertexSet int1 = snode[static_cast<size_t>(c1)].intersection(
            snode[static_cast<size_t>(neighbor)]);
        VertexSet int2 = snode[static_cast<size_t>(c2)].intersection(
            snode[static_cast<size_t>(neighbor)]);
        if (int1 != int2) {
            return false;
        }
    }
    return true;
}

// Find the (row, col) of the element with the maximum value in the CSC matrix.
inline std::pair<int64_t, int64_t> max_elem(const SimpleCsc& A) {
    if (A.values.empty()) return {0, 0};

    size_t max_idx = 0;
    int64_t max_val = A.values[0];
    for (size_t i = 1; i < A.values.size(); ++i) {
        if (A.values[i] > max_val) {
            max_val = A.values[i];
            max_idx = i;
        }
    }
    return A.index_to_coord(static_cast<int64_t>(max_idx));
}

// Return a permutation that sorts values in descending order.
inline std::vector<int64_t> sortperm_rev(const std::vector<int64_t>& values) {
    std::vector<int64_t> p(values.size());
    std::iota(p.begin(), p.end(), int64_t(0));
    std::sort(p.begin(), p.end(), [&values](int64_t a, int64_t b) {
        return values[static_cast<size_t>(a)] > values[static_cast<size_t>(b)];
    });
    return p;
}

// Given a linear nzval index, return the (row, col) edge from the CSC matrix.
inline std::pair<int64_t, int64_t> edge_from_index(const SimpleCsc& A, int64_t idx) {
    return A.index_to_coord(idx);
}

// Replace edge values with clique intersection cardinalities.
inline void clique_intersections(SimpleCsc& E, const std::vector<VertexSet>& snd) {
    for (int64_t col = 0; col < E.ncol; ++col) {
        int64_t start = E.colptr[static_cast<size_t>(col)];
        int64_t end = E.colptr[static_cast<size_t>(col + 1)];
        for (int64_t j = start; j < end; ++j) {
            int64_t row = E.rowind[static_cast<size_t>(j)];
            E.values[static_cast<size_t>(j)] = intersect_dim(
                snd[static_cast<size_t>(row)], snd[static_cast<size_t>(col)]);
        }
    }
}

// Kruskal's algorithm for maximum-weight spanning tree.
// Marks MST edges with value -1 in E.
inline void kruskal(SimpleCsc& E, int64_t num_cliques) {
    int64_t num_initial_cliques = E.ncol;
    DisjointSetUnion dsu(num_initial_cliques);

    std::vector<int64_t> I0, J0, V0;
    E.findnz(I0, J0, V0);

    // Sort by decreasing weight
    auto p = sortperm_rev(V0);

    // Apply permutation to I0, J0
    std::vector<int64_t> I(p.size()), J(p.size());
    for (size_t k = 0; k < p.size(); ++k) {
        I[k] = I0[static_cast<size_t>(p[k])];
        J[k] = J0[static_cast<size_t>(p[k])];
    }

    int64_t num_edges_found = 0;

    for (size_t k = 0; k < I.size(); ++k) {
        int64_t row = I[k];
        int64_t col = J[k];
        if (!dsu.in_same_set(row, col)) {
            dsu.unite(row, col);
            // Mark MST edge with -1
            E.values[static_cast<size_t>(p[k])] = -1;
            num_edges_found++;
            if (num_edges_found >= num_cliques - 1) break;
        }
    }
}

// Find all neighbors of clique c in the edge matrix E.
// Neighbors are in row c (columns < c) and column c (rows > c).
inline std::vector<int64_t> find_neighbors(const SimpleCsc& edges, int64_t c) {
    std::vector<int64_t> neighbors;

    // Find nonzero columns in row c (i.e., for col < c, check if (c, col) exists)
    if (c > 0) {
        for (int64_t col = 0; col < c; ++col) {
            auto val = edges.get_entry(c, col);
            if (val.has_value() && *val != 0) {
                neighbors.push_back(col);
            }
        }
    }

    // Find nonzero rows in column c (rows > c are stored in column c)
    if (c < edges.ncol - 1) {
        int64_t start = edges.colptr[static_cast<size_t>(c)];
        int64_t end = edges.colptr[static_cast<size_t>(c + 1)];
        for (int64_t i = start; i < end; ++i) {
            neighbors.push_back(edges.rowind[static_cast<size_t>(i)]);
        }
    }

    return neighbors;
}

// Assign children along the MST rooted at c using iterative DFS.
inline void assign_children(std::vector<int64_t>& snode_parent,
                             std::vector<VertexSet>& snode_children,
                             int64_t c, const SimpleCsc& edges) {
    std::vector<int64_t> stack = {c};

    while (!stack.empty()) {
        int64_t cur = stack.back();
        stack.pop_back();

        auto neighbors = find_neighbors(edges, cur);
        for (int64_t n : neighbors) {
            int64_t row = std::max(cur, n);
            int64_t col = std::min(cur, n);
            auto val = edges.get_entry(row, col);
            bool is_mst_edge = val.has_value() && *val == -1;
            bool is_not_parent = snode_parent[static_cast<size_t>(cur)] != n;

            if (is_mst_edge && is_not_parent) {
                snode_parent[static_cast<size_t>(n)] = cur;
                snode_children[static_cast<size_t>(cur)].insert(n);
                stack.push_back(n);
            }
        }
    }
}

// Determine parent structure from the MST. Root is the clique containing
// the vertex with the highest order.
inline void determine_parent_cliques(
    std::vector<int64_t>& snode_parent,
    std::vector<VertexSet>& snode_children,
    const std::vector<VertexSet>& cliques,
    const std::vector<int64_t>& post,
    const SimpleCsc& E) {

    // Vertex with highest order
    int64_t v = post.back();
    int64_t root = 0;

    // Find clique that contains that vertex
    for (size_t k = 0; k < cliques.size(); ++k) {
        if (cliques[k].contains(v)) {
            snode_parent[k] = NO_PARENT;
            root = static_cast<int64_t>(k);
            break;
        }
    }

    // Assign children via DFS along the MST
    assign_children(snode_parent, snode_children, root, E);
}

// Compute a post-ordering for the supernodes.
// This is a DFS-based topological ordering.
inline void post_order(std::vector<int64_t>& post_out,
                        const std::vector<int64_t>& parent,
                        std::vector<VertexSet>& children,
                        int64_t nc) {
    int64_t n = static_cast<int64_t>(parent.size());
    std::vector<int64_t> order(static_cast<size_t>(n), nc + 1);

    // Find root (node with NO_PARENT)
    int64_t root = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (parent[static_cast<size_t>(i)] == NO_PARENT) {
            root = i;
            break;
        }
    }
    assert(root >= 0 && "SuperNodeTree must have a root");

    std::vector<int64_t> stack = {root};

    // Initialise post to identity
    post_out.resize(static_cast<size_t>(n));
    std::iota(post_out.begin(), post_out.end(), int64_t(0));

    int64_t idx = nc;

    while (!stack.empty()) {
        int64_t v = stack.back();
        stack.pop_back();

        order[static_cast<size_t>(v)] = idx;
        idx--;

        // Sort children for consistency with the Rust/Julia implementation
        children[static_cast<size_t>(v)].sort();
        for (int64_t ch : children[static_cast<size_t>(v)]) {
            stack.push_back(ch);
        }
    }

    // Sort post by order
    std::sort(post_out.begin(), post_out.end(), [&order](int64_t x, int64_t y) {
        return order[static_cast<size_t>(x)] < order[static_cast<size_t>(y)];
    });

    // Truncate if nc < n
    if (nc != n) {
        post_out.resize(static_cast<size_t>(nc));
    }
}

// Split clique sets back into supernodes and separators by traversing
// the tree in ascending topological order.
inline void split_cliques(
    std::vector<VertexSet>& snode,
    std::vector<VertexSet>& separators,
    const std::vector<int64_t>& snode_parent,
    const std::vector<int64_t>& snode_post,
    int64_t num_cliques) {

    // Traverse in ascending topological order (skip root which is last)
    for (int64_t j = 0; j < num_cliques - 1; ++j) {
        int64_t c_ind = snode_post[static_cast<size_t>(j)];
        int64_t p_ind = snode_parent[static_cast<size_t>(c_ind)];

        // separator = intersection of clique with parent
        separators[static_cast<size_t>(c_ind)] = snode[static_cast<size_t>(c_ind)].intersection(
            snode[static_cast<size_t>(p_ind)]);

        // snode = clique minus separator
        VertexSet tmp;
        for (int64_t s : snode[static_cast<size_t>(c_ind)]) {
            if (!separators[static_cast<size_t>(c_ind)].contains(s)) {
                tmp.insert(s);
            }
        }
        snode[static_cast<size_t>(c_ind)] = std::move(tmp);
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// ParentChildMergeStrategy
// ---------------------------------------------------------------------------
class ParentChildMergeStrategy : public MergeStrategy {
public:
    explicit ParentChildMergeStrategy(int64_t t_fill = 8, int64_t t_size = 8)
        : stop_(false), clique_index_(0), t_fill_(t_fill), t_size_(t_size) {}

    void initialise(SuperNodeTree& t) override {
        // Start with node that has second-highest order
        clique_index_ = static_cast<int64_t>(t.snode.size()) - 2;
    }

    bool is_done() const override { return stop_; }

    std::optional<std::pair<int64_t, int64_t>> traverse(SuperNodeTree& t) override {
        int64_t c = t.snode_post[static_cast<size_t>(clique_index_)];
        return std::make_pair(t.snode_parent[static_cast<size_t>(c)], c);
    }

    bool evaluate(SuperNodeTree& t, std::pair<int64_t, int64_t> cand) override {
        if (stop_) return false;

        int64_t parent = cand.first;
        int64_t child = cand.second;

        int64_t dim_parent_snode = static_cast<int64_t>(t.snode[static_cast<size_t>(parent)].len());
        int64_t dim_parent_sep = static_cast<int64_t>(t.separators[static_cast<size_t>(parent)].len());
        int64_t dim_clique_snode = static_cast<int64_t>(t.snode[static_cast<size_t>(child)].len());
        int64_t dim_clique_sep = static_cast<int64_t>(t.separators[static_cast<size_t>(child)].len());

        int64_t dim_parent = dim_parent_snode + dim_parent_sep;
        int64_t dim_clique = dim_clique_snode + dim_clique_sep;
        int64_t fill = (dim_parent - dim_clique_sep) * (dim_clique - dim_clique_sep);
        int64_t max_snode = std::max(dim_clique_snode, dim_parent_snode);

        return fill <= t_fill_ || max_snode <= t_size_;
    }

    void merge_two_cliques(SuperNodeTree& t, std::pair<int64_t, int64_t> cand) override {
        // Determine which is parent and which is child
        int64_t p, ch;
        if (t.snode_children[static_cast<size_t>(cand.first)].contains(cand.second)) {
            p = cand.first;
            ch = cand.second;
        } else {
            p = cand.second;
            ch = cand.first;
        }

        // Merge child's vertex sets into parent's
        detail::set_union_into_indexed(t.snode, p, ch);
        t.snode[static_cast<size_t>(ch)].clear();
        t.separators[static_cast<size_t>(ch)].clear();

        // Update parent structure: grandchildren point to p
        for (int64_t grandch : t.snode_children[static_cast<size_t>(ch)]) {
            t.snode_parent[static_cast<size_t>(grandch)] = p;
        }
        t.snode_parent[static_cast<size_t>(ch)] = INACTIVE_NODE;

        // Update children structure
        t.snode_children[static_cast<size_t>(p)].shift_remove(ch);
        detail::set_union_into_indexed(t.snode_children, p, ch);
        t.snode_children[static_cast<size_t>(ch)].clear();

        t.n_cliques--;
    }

    void update_strategy(SuperNodeTree& /*t*/, std::pair<int64_t, int64_t> /*cand*/,
                          bool /*do_merge*/) override {
        if (clique_index_ == 0) {
            stop_ = true;
        } else {
            clique_index_--;
        }
    }

    void post_process_merge(SuperNodeTree& t) override {
        detail::post_order(t.snode_post, t.snode_parent, t.snode_children, t.n_cliques);
    }

private:
    bool stop_;
    int64_t clique_index_;
    int64_t t_fill_;
    int64_t t_size_;
};

// ---------------------------------------------------------------------------
// CliqueGraphMergeStrategy — full clique graph merging (Garstka et al. 2019)
// ---------------------------------------------------------------------------
class CliqueGraphMergeStrategy : public MergeStrategy {
public:
    explicit CliqueGraphMergeStrategy(EdgeWeightMethod ew = EdgeWeightMethod::Cubic)
        : stop_(false), edge_weight_(ew) {}

    void initialise(SuperNodeTree& t) override {
        // Merge separators into supernodes to get full cliques.
        // We give up the tree structure; after merging, a new clique tree
        // will be computed in post_process_merge.
        for (size_t i = 0; i < t.snode.size(); ++i) {
            for (int64_t s : t.separators[i]) {
                t.snode[i].insert(s);
            }
        }

        // Clear parent/children (we operate on a graph, not a tree)
        for (size_t i = 0; i < t.snode_parent.size(); ++i) {
            t.snode_parent[i] = INACTIVE_NODE;
            t.snode_children[i].clear();
        }

        // Compute the reduced clique graph edges
        std::vector<int64_t> rows, cols;
        detail::compute_reduced_clique_graph(t.separators, t.snode, rows, cols);

        // Compute edge weights
        auto weights = detail::compute_weights(rows, cols, t.snode, edge_weight_);

        // Build the CSC edge matrix
        edges_ = SimpleCsc::from_triplets(t.n_cliques, t.n_cliques, rows, cols, weights);

        // Build adjacency table
        adjacency_table_ = detail::compute_adjacency_table(edges_, t.n_cliques);

        stop_ = false;
    }

    bool is_done() const override { return stop_; }

    std::optional<std::pair<int64_t, int64_t>> traverse(SuperNodeTree& t) override {
        // Find edge with highest weight
        auto edge = detail::max_elem(edges_);

        if (detail::is_permissible(edge, adjacency_table_, t.snode)) {
            return edge;
        }

        // Sort weights in descending order and try edges
        auto p = detail::sortperm_rev(edges_.values);

        for (size_t k = 1; k < p.size(); ++k) {
            auto candidate = detail::edge_from_index(edges_, p[k]);
            if (detail::is_permissible(candidate, adjacency_table_, t.snode)) {
                return candidate;
            }
        }

        return std::nullopt;
    }

    bool evaluate(SuperNodeTree& /*t*/, std::pair<int64_t, int64_t> cand) override {
        int64_t c1 = cand.first;
        int64_t c2 = cand.second;

        auto val = edges_.get_entry(c1, c2);
        bool do_merge = val.has_value() && *val >= 0;

        if (!do_merge) {
            stop_ = true;
        }
        return do_merge;
    }

    void merge_two_cliques(SuperNodeTree& t, std::pair<int64_t, int64_t> cand) override {
        int64_t c1 = cand.first;
        int64_t c2 = cand.second;

        // Merge clique c2 into c1
        detail::set_union_into_indexed(t.snode, c1, c2);
        t.snode[static_cast<size_t>(c2)].clear();

        t.n_cliques--;
    }

    void update_strategy(SuperNodeTree& t, std::pair<int64_t, int64_t> cand,
                          bool do_merge) override {
        if (!do_merge) return;

        int64_t c1_ind = cand.first;
        int64_t c_removed = cand.second;
        int64_t n = edges_.ncol;

        const VertexSet& c1 = t.snode[static_cast<size_t>(c1_ind)];
        const VertexSet& neighbors = adjacency_table_[c1_ind];

        // Neighbors exclusive to the removed clique (not already neighbors of c1)
        VertexSet new_neighbors;
        for (int64_t e : adjacency_table_[c_removed]) {
            if (!neighbors.contains(e) && e != c1_ind) {
                new_neighbors.insert(e);
            }
        }

        // Recalculate edge values of all of c1's neighbors
        for (int64_t n_ind : neighbors) {
            if (n_ind != c_removed) {
                const VertexSet& neighbor = t.snode[static_cast<size_t>(n_ind)];
                int64_t row = std::max(c1_ind, n_ind);
                int64_t col = std::min(c1_ind, n_ind);
                int64_t val = detail::edge_metric(c1, neighbor, edge_weight_);
                edges_.set_entry(row, col, val);
            }
        }

        // Point edges exclusive to removed clique to surviving clique
        for (int64_t n_ind : new_neighbors) {
            const VertexSet& neighbor = t.snode[static_cast<size_t>(n_ind)];
            int64_t row = std::max(c1_ind, n_ind);
            int64_t col = std::min(c1_ind, n_ind);
            int64_t val = detail::edge_metric(c1, neighbor, edge_weight_);
            edges_.set_entry(row, col, val);
        }

        // Zero out all edges involving the removed clique
        for (int64_t row = c_removed + 1; row < n; ++row) {
            edges_.set_entry(row, c_removed, 0);
        }
        for (int64_t col = 0; col < c_removed; ++col) {
            edges_.set_entry(c_removed, col, 0);
        }
        edges_.dropzeros();

        // Update adjacency table: add new_neighbors to c1's neighbor list
        for (int64_t nn : new_neighbors) {
            adjacency_table_[c1_ind].insert(nn);
            adjacency_table_[nn].insert(c1_ind);
        }

        // Remove the removed clique from the adjacency table
        adjacency_table_.erase(c_removed);
        for (auto& [key, set] : adjacency_table_) {
            set.shift_remove(c_removed);
        }
    }

    void post_process_merge(SuperNodeTree& t) override {
        // Number the non-empty supernodes
        t.snode_post.clear();
        for (size_t i = 0; i < t.snode.size(); ++i) {
            if (!t.snode[i].is_empty()) {
                t.snode_post.push_back(static_cast<int64_t>(i));
            }
        }

        t.snode_parent.assign(t.snode.size(), INACTIVE_NODE);

        // Recompute a clique tree from the clique graph
        if (t.n_cliques > 1) {
            clique_tree_from_graph(t);
        }

        // Sort supernodes and separators
        for (auto& s : t.snode) s.sort();
        for (auto& s : t.separators) s.sort();
    }

private:
    void clique_tree_from_graph(SuperNodeTree& t) {
        // Replace edge values with clique intersection cardinalities
        detail::clique_intersections(edges_, t.snode);

        // Maximum-weight spanning tree via Kruskal
        detail::kruskal(edges_, t.n_cliques);

        // Determine parent structure from the MST
        detail::determine_parent_cliques(
            t.snode_parent, t.snode_children, t.snode, t.post, edges_);

        // Recompute post-ordering (may shrink to n_cliques)
        detail::post_order(t.snode_post, t.snode_parent, t.snode_children, t.n_cliques);

        // Clear separators; they will be rebuilt in split_cliques
        for (auto& sep : t.separators) sep.clear();

        // Split clique sets back into supernodes and separators
        detail::split_cliques(t.snode, t.separators, t.snode_parent, t.snode_post, t.n_cliques);
    }

    bool stop_;
    SimpleCsc edges_;
    std::unordered_map<int64_t, VertexSet> adjacency_table_;
    EdgeWeightMethod edge_weight_;
};

} // namespace chordal
} // namespace moreau
