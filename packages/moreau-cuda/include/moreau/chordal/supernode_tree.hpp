#ifndef MOREAU_CHORDAL_SUPERNODE_TREE_HPP
#define MOREAU_CHORDAL_SUPERNODE_TREE_HPP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stack>
#include <vector>

namespace moreau {

// Sentinel value marking root nodes (no parent).
static constexpr int64_t NO_PARENT = INT64_MAX;

// Sentinel value marking inactive (merged) nodes.
static constexpr int64_t INACTIVE_NODE = INT64_MAX - 1;

// ---------------------------------------------------------------------------
// VertexSet -- ordered set of unique int64_t values backed by std::vector.
//
// Maintains sorted order so that iteration is deterministic and set operations
// (union, intersection, subset, etc.) can be done in linear time via merges.
// ---------------------------------------------------------------------------
class VertexSet {
public:
    VertexSet() = default;

    explicit VertexSet(size_t capacity) { data_.reserve(capacity); }

    // Insert v if not already present, maintaining sorted order.
    void insert(int64_t v) {
        auto it = std::lower_bound(data_.begin(), data_.end(), v);
        if (it == data_.end() || *it != v) {
            data_.insert(it, v);
        }
    }

    // Remove v if present, maintaining sorted order.
    void remove(int64_t v) {
        auto it = std::lower_bound(data_.begin(), data_.end(), v);
        if (it != data_.end() && *it == v) {
            data_.erase(it);
        }
    }

    bool contains(int64_t v) const {
        return std::binary_search(data_.begin(), data_.end(), v);
    }

    bool is_subset_of(const VertexSet& other) const {
        // Every element in *this must be in other.
        size_t j = 0;
        for (int64_t v : data_) {
            while (j < other.data_.size() && other.data_[j] < v) ++j;
            if (j >= other.data_.size() || other.data_[j] != v) return false;
            ++j;
        }
        return true;
    }

    void clear() { data_.clear(); }
    bool empty() const { return data_.empty(); }
    size_t size() const { return data_.size(); }

    // Sort the set (should already be sorted, but provided for explicitness
    // after bulk operations).
    void sort() { std::sort(data_.begin(), data_.end()); }

    // Extend from a range [first, last) of int64_t, then re-sort/unique.
    template <typename It>
    void extend(It first, It last) {
        for (auto it = first; it != last; ++it) {
            insert(*it);
        }
    }

    void extend(const VertexSet& other) {
        extend(other.begin(), other.end());
    }

    // Compute union of *this and other into a new VertexSet.
    VertexSet set_union(const VertexSet& other) const {
        VertexSet out;
        out.data_.resize(data_.size() + other.data_.size());
        auto end = std::set_union(
            data_.begin(), data_.end(),
            other.data_.begin(), other.data_.end(),
            out.data_.begin());
        out.data_.erase(end, out.data_.end());
        return out;
    }

    // Compute intersection with other into a new VertexSet.
    VertexSet intersection(const VertexSet& other) const {
        VertexSet out;
        out.data_.resize(std::min(data_.size(), other.data_.size()));
        auto end = std::set_intersection(
            data_.begin(), data_.end(),
            other.data_.begin(), other.data_.end(),
            out.data_.begin());
        out.data_.erase(end, out.data_.end());
        return out;
    }

    // Compute difference (*this \ other) into a new VertexSet.
    VertexSet difference(const VertexSet& other) const {
        VertexSet out;
        out.data_.reserve(data_.size());
        std::set_difference(
            data_.begin(), data_.end(),
            other.data_.begin(), other.data_.end(),
            std::back_inserter(out.data_));
        return out;
    }

    bool operator==(const VertexSet& other) const { return data_ == other.data_; }
    bool operator!=(const VertexSet& other) const { return data_ != other.data_; }

    // Iterators
    using iterator = std::vector<int64_t>::iterator;
    using const_iterator = std::vector<int64_t>::const_iterator;
    iterator begin() { return data_.begin(); }
    iterator end() { return data_.end(); }
    const_iterator begin() const { return data_.begin(); }
    const_iterator end() const { return data_.end(); }

    int64_t operator[](size_t i) const { return data_[i]; }
    int64_t& operator[](size_t i) { return data_[i]; }

    // Direct access for low-level algorithms.
    const std::vector<int64_t>& data() const { return data_; }

private:
    std::vector<int64_t> data_;
};

// ---------------------------------------------------------------------------
// SuperNodeTree -- analysis of the sparsity pattern of an LDL factor matrix L.
//
// Accepts L in CSC format (colptr, rowind arrays, nrow/ncol dimensions).
// Values of L are not needed -- only the sparsity pattern matters.
// ---------------------------------------------------------------------------
class SuperNodeTree {
public:
    // Vertices of supernodes (also called residuals).
    std::vector<VertexSet> snode;
    // Post-order of supernodal elimination tree.
    std::vector<int64_t> snode_post;
    // Parent of each supernode.
    std::vector<int64_t> snode_parent;
    // Children of each supernode.
    std::vector<VertexSet> snode_children;
    // Post-ordering of the vertices in the elimination tree.
    std::vector<int64_t> post;
    // Vertices of clique separators.
    std::vector<VertexSet> separators;
    // Block dimensions per clique (post-merge). Empty until calculate_block_dimensions().
    std::optional<std::vector<int64_t>> nblk;
    // Number of nonempty supernodes / cliques.
    int64_t n_cliques;

    // -----------------------------------------------------------------------
    // Construct from an LDL factor L given in CSC format.
    //   colptr:  column pointers, length ncol+1
    //   rowind:  row indices (only the *below-diagonal* entries per column;
    //            colptr[j]..colptr[j+1] gives the row indices > j that are
    //            nonzero in column j of L -- the unit diagonal is excluded)
    //   nrow:    number of rows (== number of columns for square L)
    // -----------------------------------------------------------------------
    SuperNodeTree(const int64_t* colptr,
                  const int64_t* rowind,
                  int64_t nrow)
    {
        // -- elimination tree parent structure from L --
        std::vector<int64_t> parent = parent_from_L(colptr, rowind, nrow);

        // -- children & post-order --
        std::vector<VertexSet> children = children_from_parent(parent);
        post.resize(static_cast<size_t>(nrow));
        post_order(post, parent, children, nrow);

        // -- higher degree --
        std::vector<int64_t> degree = higher_degree(colptr, nrow);

        // -- supernodes (Pothen-Sun) --
        find_supernodes(parent, post, degree, snode, snode_parent);

        // -- supernode children & post-order --
        snode_children = children_from_parent(snode_parent);
        snode_post.resize(snode_parent.size());
        post_order(snode_post, snode_parent, snode_children,
                   static_cast<int64_t>(snode_parent.size()));

        // -- separators --
        separators = find_separators(colptr, rowind, nrow, snode);

        n_cliques = static_cast<int64_t>(snode.size());
    }

    // -----------------------------------------------------------------------
    // Accessors (indexed through post-order)
    // -----------------------------------------------------------------------

    int64_t get_post_order(int64_t i) const {
        return snode_post[static_cast<size_t>(i)];
    }

    const VertexSet& get_snode(int64_t i) const {
        return snode[static_cast<size_t>(snode_post[static_cast<size_t>(i)])];
    }

    const VertexSet& get_separators(int64_t i) const {
        return separators[static_cast<size_t>(snode_post[static_cast<size_t>(i)])];
    }

    int64_t get_clique_parent(int64_t clique_index) const {
        return snode_parent[static_cast<size_t>(
            snode_post[static_cast<size_t>(clique_index)])];
    }

    int64_t get_nblk(int64_t i) const {
        assert(nblk.has_value());
        return nblk.value()[static_cast<size_t>(i)];
    }

    int64_t get_overlap(int64_t i) const {
        return static_cast<int64_t>(
            separators[static_cast<size_t>(
                snode_post[static_cast<size_t>(i)])]
                .size());
    }

    VertexSet get_clique(int64_t i) const {
        int64_t c = snode_post[static_cast<size_t>(i)];
        const VertexSet& s1 = snode[static_cast<size_t>(c)];
        const VertexSet& s2 = separators[static_cast<size_t>(c)];
        return s1.set_union(s2);
    }

    // Returns (decomposed_dim, overlaps).
    std::pair<int64_t, int64_t> get_decomposed_dim_and_overlaps() const {
        int64_t dim = 0;
        int64_t overlaps = 0;
        for (int64_t i = 0; i < n_cliques; ++i) {
            int64_t nb = get_nblk(i);
            dim += triangular_number(nb);
            int64_t ov = get_overlap(i);
            overlaps += triangular_number(ov);
        }
        return {dim, overlaps};
    }

    // -----------------------------------------------------------------------
    // reorder_snode_consecutively
    //
    // Reorders vertices in each supernode (and separator) to have consecutive
    // numbering. Also modifies `ordering` to map the new vertices back to
    // their original positions.
    // -----------------------------------------------------------------------
    void reorder_snode_consecutively(std::vector<int64_t>& ordering) {
        size_t n = post.size();
        std::vector<int64_t> p(n, 0);

        int64_t k = 0;
        for (size_t idx = 0; idx < snode_post.size(); ++idx) {
            int64_t si = snode_post[idx];
            VertexSet& sn = snode[static_cast<size_t>(si)];
            int64_t sn_len = static_cast<int64_t>(sn.size());

            // Copy current vertices into p[k..k+sn_len), then sort.
            for (int64_t j = 0; j < sn_len; ++j) {
                p[static_cast<size_t>(k + j)] = sn[static_cast<size_t>(j)];
            }
            std::sort(p.begin() + k, p.begin() + k + sn_len);

            // Reassign snode to consecutive values k..k+sn_len.
            sn.clear();
            for (int64_t j = k; j < k + sn_len; ++j) {
                sn.insert(j);
            }
            k += sn_len;
        }

        // Compute inverse permutation.
        std::vector<int64_t> p_inv = invperm(p);

        // Permute separators using p_inv.
        for (VertexSet& sp : separators) {
            size_t sp_len = sp.size();
            assert(p.size() >= sp_len);

            // Use p[0..sp_len) as scratch.
            for (size_t i = 0; i < sp_len; ++i) {
                p[i] = p_inv[static_cast<size_t>(sp[static_cast<int64_t>(i)])];
            }
            sp.clear();
            for (size_t i = 0; i < sp_len; ++i) {
                sp.insert(p[i]);
            }
        }

        // Inverse-permute ordering: ordering[p_inv[i]] = tmp[i].
        std::vector<int64_t> tmp = ordering;
        ipermute(ordering, tmp, p_inv);
    }

    // -----------------------------------------------------------------------
    // calculate_block_dimensions
    //
    // Populates nblk with the block size (snode + separator) for each clique,
    // in post-order.
    // -----------------------------------------------------------------------
    void calculate_block_dimensions() {
        std::vector<int64_t> blk(static_cast<size_t>(n_cliques));
        for (int64_t i = 0; i < n_cliques; ++i) {
            int64_t c = snode_post[static_cast<size_t>(i)];
            blk[static_cast<size_t>(i)] =
                static_cast<int64_t>(separators[static_cast<size_t>(c)].size()
                                     + snode[static_cast<size_t>(c)].size());
        }
        nblk = std::move(blk);
    }

    // -----------------------------------------------------------------------
    // Merge helpers (used by merge strategies operating on the tree)
    // -----------------------------------------------------------------------

    // Merge supernode c2 into c1 in the parent-child merge pattern:
    //   - Union snode[p] |= snode[ch], clear ch.
    //   - Clear separator[ch].
    //   - Re-parent grandchildren of ch to p.
    //   - Update snode_children.
    //   - Decrement n_cliques.
    void merge_two_cliques_parent_child(int64_t parent_idx, int64_t child_idx) {
        size_t p = static_cast<size_t>(parent_idx);
        size_t ch = static_cast<size_t>(child_idx);

        // Merge child's vertices into parent's vertex set.
        snode[p].extend(snode[ch]);
        snode[ch].clear();
        separators[ch].clear();

        // Re-parent grandchildren.
        for (int64_t grandch : snode_children[ch]) {
            snode_parent[static_cast<size_t>(grandch)] = parent_idx;
        }
        snode_parent[ch] = INACTIVE_NODE;

        // Update children: remove ch from parent's children, add ch's children to parent.
        snode_children[p].remove(child_idx);
        snode_children[p].extend(snode_children[ch]);
        snode_children[ch].clear();

        --n_cliques;
    }

    // Merge clique c2 into c1 for clique-graph merge pattern (no separator update).
    void merge_two_cliques_graph(int64_t c1, int64_t c2) {
        size_t i1 = static_cast<size_t>(c1);
        size_t i2 = static_cast<size_t>(c2);

        snode[i1].extend(snode[i2]);
        snode[i2].clear();

        --n_cliques;
    }

    // Determine which of two cliques is the parent and which is the child.
    // Returns (parent, child).
    std::pair<int64_t, int64_t> determine_parent(int64_t c1, int64_t c2) const {
        if (snode_children[static_cast<size_t>(c1)].contains(c2)) {
            return {c1, c2};
        }
        return {c2, c1};
    }

    // Clique dimensions (snode_size, separator_size) for a raw index (not post-ordered).
    std::pair<int64_t, int64_t> clique_dim(int64_t i) const {
        size_t idx = static_cast<size_t>(i);
        return {static_cast<int64_t>(snode[idx].size()),
                static_cast<int64_t>(separators[idx].size())};
    }

    // Compute fill-in from merging two cliques with given dimensions.
    static int64_t fill_in(int64_t dim_clique_snode, int64_t dim_clique_sep,
                           int64_t dim_parent_snode, int64_t dim_parent_sep) {
        int64_t dim_parent = dim_parent_snode + dim_parent_sep;
        int64_t dim_clique = dim_clique_snode + dim_clique_sep;
        return (dim_parent - dim_clique_sep) * (dim_clique - dim_clique_sep);
    }

    // -----------------------------------------------------------------------
    // Static post_order (exposed for use after merge post-processing).
    // -----------------------------------------------------------------------
    static void post_order(std::vector<int64_t>& post_out,
                           const std::vector<int64_t>& parent,
                           std::vector<VertexSet>& children,
                           int64_t nc)
    {
        int64_t n = static_cast<int64_t>(parent.size());

        // Find root (node whose parent == NO_PARENT).
        int64_t root = -1;
        for (int64_t i = 0; i < n; ++i) {
            if (parent[static_cast<size_t>(i)] == NO_PARENT) {
                root = i;
                break;
            }
        }
        assert(root >= 0 && "Elimination tree must have a root");

        std::vector<int64_t> order(static_cast<size_t>(n), nc + 1);

        std::vector<int64_t> stack;
        stack.reserve(static_cast<size_t>(n));
        stack.push_back(root);

        post_out.resize(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            post_out[static_cast<size_t>(i)] = i;
        }

        int64_t idx = nc;
        while (!stack.empty()) {
            int64_t v = stack.back();
            stack.pop_back();
            order[static_cast<size_t>(v)] = idx;
            --idx;

            // Sort children so they are pushed in consistent order.
            children[static_cast<size_t>(v)].sort();
            for (int64_t ch : children[static_cast<size_t>(v)]) {
                stack.push_back(ch);
            }
        }

        // Sort post_out by the order values.
        std::sort(post_out.begin(), post_out.end(),
                  [&order](int64_t a, int64_t b) {
                      return order[static_cast<size_t>(a)]
                           < order[static_cast<size_t>(b)];
                  });

        // Truncate to nc if nc < n.
        if (nc < n) {
            post_out.resize(static_cast<size_t>(nc));
        }
    }

private:
    // -----------------------------------------------------------------------
    // Internal: parent_from_L
    // parent[j] = first row index > j in column j, or NO_PARENT if none.
    // -----------------------------------------------------------------------
    static std::vector<int64_t> parent_from_L(const int64_t* colptr,
                                               const int64_t* rowind,
                                               int64_t nrow)
    {
        std::vector<int64_t> parent(static_cast<size_t>(nrow), NO_PARENT);
        for (int64_t j = 0; j < nrow; ++j) {
            if (j == nrow - 1) {
                parent[static_cast<size_t>(j)] = NO_PARENT;
            } else {
                // First entry in column j is the parent.
                parent[static_cast<size_t>(j)] = rowind[colptr[j]];
            }
        }
        return parent;
    }

    // -----------------------------------------------------------------------
    // Internal: children_from_parent
    // -----------------------------------------------------------------------
    static std::vector<VertexSet> children_from_parent(
        const std::vector<int64_t>& parent)
    {
        size_t n = parent.size();
        std::vector<VertexSet> children(n);
        for (size_t i = 0; i < n; ++i) {
            int64_t pi = parent[i];
            if (pi != NO_PARENT) {
                children[static_cast<size_t>(pi)].insert(static_cast<int64_t>(i));
            }
        }
        return children;
    }

    // -----------------------------------------------------------------------
    // Internal: higher_degree
    // degree[v] = number of below-diagonal nonzeros in column v of L.
    // -----------------------------------------------------------------------
    static std::vector<int64_t> higher_degree(const int64_t* colptr,
                                               int64_t nrow)
    {
        std::vector<int64_t> degree(static_cast<size_t>(nrow), 0);
        for (int64_t v = 0; v < nrow - 1; ++v) {
            degree[static_cast<size_t>(v)] = colptr[v + 1] - colptr[v];
        }
        return degree;
    }

    // -----------------------------------------------------------------------
    // Internal: Pothen-Sun algorithm for supernode detection.
    //
    // Populates snode_out and snode_parent_out.
    // -----------------------------------------------------------------------
    static void find_supernodes(
        const std::vector<int64_t>& parent,
        const std::vector<int64_t>& post_in,
        const std::vector<int64_t>& degree,
        std::vector<VertexSet>& snode_out,
        std::vector<int64_t>& snode_parent_out)
    {
        int64_t n = static_cast<int64_t>(parent.size());

        // snode_index[v] < 0: v is a representative vertex.
        // snode_index[v] >= 0: v belongs to supernode snode_index[v].
        std::vector<int64_t> snode_index(static_cast<size_t>(n), -1);
        std::vector<int64_t> snode_par(static_cast<size_t>(n), NO_PARENT);
        std::vector<VertexSet> children(static_cast<size_t>(n));

        // Find root.
        int64_t root_index = -1;
        for (int64_t i = 0; i < n; ++i) {
            if (parent[static_cast<size_t>(i)] == NO_PARENT) {
                root_index = i;
                break;
            }
        }

        // Process vertices in post-order.
        for (int64_t pi = 0; pi < static_cast<int64_t>(post_in.size()); ++pi) {
            int64_t v = post_in[static_cast<size_t>(pi)];
            size_t vi = static_cast<size_t>(v);

            if (parent[vi] == NO_PARENT) {
                children[static_cast<size_t>(root_index)].insert(v);
            } else {
                children[static_cast<size_t>(parent[vi])].insert(v);
            }

            if (parent[vi] != NO_PARENT) {
                size_t pv = static_cast<size_t>(parent[vi]);

                if (degree[vi] - 1 == degree[pv] && snode_index[pv] == -1) {
                    // Case A: v is a representative vertex.
                    if (snode_index[vi] < 0) {
                        snode_index[pv] = v;
                        snode_index[vi] -= 1;
                    }
                    // Case B: v is not a representative vertex.
                    else {
                        snode_index[pv] = snode_index[vi];
                        size_t tmp = static_cast<size_t>(snode_index[vi]);
                        snode_index[tmp] -= 1;
                    }
                } else if (snode_index[vi] < 0) {
                    snode_par[vi] = v;
                } else {
                    size_t sv = static_cast<size_t>(snode_index[vi]);
                    snode_par[sv] = snode_index[vi];
                }
            }

            // k: representative vertex of the supernode that v belongs to.
            int64_t k = (snode_index[vi] < 0) ? v : snode_index[vi];

            // Loop over v's children.
            const VertexSet& v_children = children[vi];
            for (int64_t w : v_children) {
                size_t wi = static_cast<size_t>(w);
                int64_t l = (snode_index[wi] < 0) ? w : snode_index[wi];
                if (l != k) {
                    snode_par[static_cast<size_t>(l)] = k;
                }
            }
        }

        // Collect representative vertices.
        std::vector<int64_t> repr_vertex;
        repr_vertex.reserve(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            if (snode_index[static_cast<size_t>(i)] < 0) {
                repr_vertex.push_back(i);
            }
        }

        // Collect parents of representatives.
        std::vector<int64_t> repr_parent(repr_vertex.size());
        for (size_t i = 0; i < repr_vertex.size(); ++i) {
            repr_parent[i] = snode_par[static_cast<size_t>(repr_vertex[i])];
        }

        // Build output snode_parent indexed by representative position.
        snode_parent_out.assign(repr_vertex.size(), NO_PARENT);
        for (size_t i = 0; i < repr_parent.size(); ++i) {
            // Find repr_parent[i] in repr_vertex.
            auto it = std::find(repr_vertex.begin(), repr_vertex.end(), repr_parent[i]);
            if (it != repr_vertex.end()) {
                snode_parent_out[i] = static_cast<int64_t>(
                    std::distance(repr_vertex.begin(), it));
            } else {
                snode_parent_out[i] = NO_PARENT;
            }
        }

        // Build snode sets: each representative collects its members.
        std::vector<VertexSet> snodes(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            size_t ii = static_cast<size_t>(i);
            if (snode_index[ii] < 0) {
                snodes[ii].insert(i);
            } else {
                snodes[static_cast<size_t>(snode_index[ii])].insert(i);
            }
        }

        // Retain only non-empty snodes.
        snode_out.clear();
        snode_out.reserve(repr_vertex.size());
        for (auto& s : snodes) {
            if (!s.empty()) {
                snode_out.push_back(std::move(s));
            }
        }
    }

    // -----------------------------------------------------------------------
    // Internal: find_separators
    //
    // For each supernode, the separator is the set of higher-order neighbors
    // of the representative vertex (min vertex in the supernode) that are NOT
    // in the supernode itself.
    // -----------------------------------------------------------------------
    static std::vector<VertexSet> find_separators(
        const int64_t* colptr,
        const int64_t* rowind,
        int64_t /*nrow*/,
        const std::vector<VertexSet>& snodes)
    {
        std::vector<VertexSet> seps(snodes.size());

        for (size_t si = 0; si < snodes.size(); ++si) {
            const VertexSet& sn = snodes[si];
            // Representative vertex = min element.
            int64_t vrep = sn[0]; // Already sorted, so [0] is min.

            // Higher-order neighbors: rowind[colptr[vrep] .. colptr[vrep+1]).
            int64_t start = colptr[vrep];
            int64_t end = colptr[vrep + 1];

            for (int64_t j = start; j < end; ++j) {
                int64_t neighbor = rowind[j];
                if (!sn.contains(neighbor)) {
                    seps[si].insert(neighbor);
                }
            }
        }
        return seps;
    }

    // -----------------------------------------------------------------------
    // Permutation utilities
    // -----------------------------------------------------------------------

    // Inverse permutation: result[p[i]] = i.
    static std::vector<int64_t> invperm(const std::vector<int64_t>& p) {
        std::vector<int64_t> b(p.size(), 0);
        for (size_t i = 0; i < p.size(); ++i) {
            size_t j = static_cast<size_t>(p[i]);
            assert(j < p.size());
            b[j] = static_cast<int64_t>(i);
        }
        return b;
    }

    // Inverse permute: out[pinv[i]] = src[i].
    static void ipermute(std::vector<int64_t>& out,
                         const std::vector<int64_t>& src,
                         const std::vector<int64_t>& pinv)
    {
        assert(out.size() == src.size());
        assert(out.size() == pinv.size());
        for (size_t i = 0; i < src.size(); ++i) {
            out[static_cast<size_t>(pinv[i])] = src[i];
        }
    }

    // Triangular number: k*(k+1)/2.
    static int64_t triangular_number(int64_t k) {
        return k * (k + 1) / 2;
    }
};

} // namespace moreau

#endif // MOREAU_CHORDAL_SUPERNODE_TREE_HPP
