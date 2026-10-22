/**
 * @file test_psd_chordal.cpp
 * @brief Tests for chordal decomposition infrastructure
 *
 * Tests the standalone ChordalInfo API (sparsity detection, augmentation,
 * reverse mapping) without solver integration. Validates that:
 * - Dense PSD cones are NOT decomposed (no-op)
 * - Block-diagonal PSD cones ARE decomposed into smaller cliques
 * - Augmentation structure is correct (dimensions, nnz)
 * - Reverse mapping correctly reconstructs original-dimension solution
 * - Each clique merge method gives the CPU backend's cliques
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

#include "moreau/chordal/chordal_info.hpp"
#include "moreau/chordal/symbolic_ldl.hpp"
#include "moreau/chordal/sparsity_pattern.hpp"

using namespace moreau;
using namespace moreau::chordal;

// ============================================================================
// Helper: build CSC identity matrix
// ============================================================================
void make_identity_csc(int64_t n, std::vector<int64_t>& colptr, std::vector<int64_t>& rowind) {
    colptr.resize(n + 1);
    rowind.resize(n);
    std::iota(colptr.begin(), colptr.end(), 0);
    std::iota(rowind.begin(), rowind.end(), 0);
}

// Helper: build CSC from dense row-major matrix
void dense_to_csc(const std::vector<double>& dense, int64_t nrow, int64_t ncol,
                  std::vector<int64_t>& colptr, std::vector<int64_t>& rowind) {
    colptr.resize(ncol + 1, 0);
    rowind.clear();
    for (int64_t j = 0; j < ncol; j++) {
        colptr[j] = rowind.size();
        for (int64_t i = 0; i < nrow; i++) {
            if (dense[i * ncol + j] != 0.0) {
                rowind.push_back(i);
            }
        }
    }
    colptr[ncol] = rowind.size();
}

// ============================================================================
// Tests
// ============================================================================

class ChordalTest : public ::testing::Test {};

// Dense PSD cone (fully connected) should NOT be decomposed
TEST_F(ChordalTest, DensePsdNoDecomposition) {
    // 3×3 PSD cone, A = identity (6×6), all rows active → dense → no decomposition
    int64_t n = 6, m = 6;  // svec dim for PSD(3) = 6
    std::vector<int64_t> A_colptr, A_rowind;
    make_identity_csc(n, A_colptr, A_rowind);

    std::vector<int64_t> psd_dims = {3};

    auto info = ChordalInfo::analyze(
        A_colptr.data(), A_rowind.data(), n, m,
        psd_dims.data(), 1,
        0, 0, 0, 0, 0,  // no other cones
        "clique_graph"
    );

    EXPECT_FALSE(info.is_decomposed()) << "Dense PSD should not be decomposed";
}

// Block-diagonal PSD: 4×4 matrix with 2×2 blocks should decompose into two PSD(2)
TEST_F(ChordalTest, BlockDiagonalDecomposition) {
    // 4×4 PSD matrix: [[a b 0 0], [b c 0 0], [0 0 d e], [0 0 e f]]
    // svec: [a, √2*b, c, 0, 0, d, 0, 0, √2*e, f]  — 10 entries
    // But svec for 4×4 has dim = 4*5/2 = 10 entries
    // Active pattern: positions for (0,0), (0,1), (1,1), (2,2), (2,3), (3,3)
    // = svec indices 0, 1, 2, 5, 8, 9
    // Inactive: indices 3, 4, 6, 7

    int64_t psd_dim = 4;
    int64_t svec_dim = psd_dim * (psd_dim + 1) / 2;  // 10
    int64_t n = 2;  // 2 variables controlling block diagonals
    int64_t m = svec_dim;

    // Build A that only touches the diagonal block positions
    // x0 → svec[0] (block 1 diagonal), x1 → svec[5] (block 2 diagonal)
    // In CSC: col 0 has row 0, col 1 has row 5
    std::vector<int64_t> A_colptr = {0, 1, 2};
    std::vector<int64_t> A_rowind = {0, 5};

    std::vector<int64_t> psd_dims = {psd_dim};

    // b is all zero — provide exact sparsity pattern to enable decomposition
    std::vector<bool> b_pattern(m, false);
    std::unique_ptr<bool[]> b_mask(new bool[m]);
    for (int64_t i = 0; i < m; ++i) b_mask[i] = false;

    auto info = ChordalInfo::analyze(
        A_colptr.data(), A_rowind.data(), n, m,
        psd_dims.data(), 1,
        0, 0, 0, 0, 0,
        "clique_graph",
        b_mask.get()
    );

    std::cout << "Block-diag decomposed: " << info.is_decomposed() << std::endl;
    if (info.is_decomposed()) {
        std::cout << "  Spatterns: " << info.spatterns.size() << std::endl;
        std::cout << "  Decomposed PSD dims:";
        for (auto d : info.decomposed_psd_dims) std::cout << " " << d;
        std::cout << std::endl;
        std::cout << "  n_aug=" << info.n_aug << " m_aug=" << info.m_aug << std::endl;
    }

    // Should decompose (has sparsity)
    EXPECT_TRUE(info.is_decomposed()) << "Block-diagonal PSD should decompose";

    // The decomposed PSD dims should be smaller than original
    if (info.is_decomposed()) {
        for (auto d : info.decomposed_psd_dims) {
            EXPECT_LT(d, psd_dim) << "Decomposed cones should be smaller than original";
        }
    }
}

TEST_F(ChordalTest, MixedConeOrderKeepsPsdBeforeExpAndPower) {
    // zero(1) | nonneg(1) | SOC(3) | PSD(4)(10) | EXP(3) | POW3D(3)
    const int64_t prefix_rows = 5;
    const int64_t psd_rows = 10;
    const int64_t suffix_rows = 6;
    const int64_t m = prefix_rows + psd_rows + suffix_rows;
    const int64_t n = 2;

    // Touch two disconnected diagonal blocks in the PSD cone so it decomposes.
    std::vector<int64_t> A_colptr = {0, 1, 2};
    std::vector<int64_t> A_rowind = {prefix_rows, prefix_rows + 5};
    std::vector<int64_t> psd_dims = {4};
    std::unique_ptr<bool[]> b_mask(new bool[m]);
    for (int64_t i = 0; i < m; ++i) b_mask[i] = false;

    auto info = ChordalInfo::analyze(
        A_colptr.data(), A_rowind.data(), n, m,
        psd_dims.data(), 1,
        1, 1, 3, 1, 1,
        "clique_graph",
        b_mask.get()
    );
    ASSERT_TRUE(info.is_decomposed());

    int64_t augmented_psd_rows = 0;
    for (int64_t dim : info.decomposed_psd_dims) {
        augmented_psd_rows += dim * (dim + 1) / 2;
    }
    const int64_t augmented_suffix_start = prefix_rows + augmented_psd_rows;

    // Prefix and suffix rows are copied around the decomposed PSD block.
    for (int64_t i = 0; i < prefix_rows; ++i) {
        EXPECT_EQ(info.b_row_map[i], i);
    }
    for (int64_t i = 0; i < suffix_rows; ++i) {
        EXPECT_EQ(info.b_row_map[augmented_suffix_start + i], prefix_rows + psd_rows + i);
    }

    std::vector<double> augmented(info.m_aug, 0.0);
    std::vector<double> original(m, 0.0);
    for (int64_t i = 0; i < prefix_rows; ++i) augmented[i] = 10.0 + i;
    for (int64_t i = 0; i < suffix_rows; ++i) {
        augmented[augmented_suffix_start + i] = 20.0 + i;
    }
    info.reverse_s(augmented.data(), original.data());
    for (int64_t i = 0; i < prefix_rows; ++i) EXPECT_EQ(original[i], 10.0 + i);
    for (int64_t i = 0; i < suffix_rows; ++i) {
        EXPECT_EQ(original[prefix_rows + psd_rows + i], 20.0 + i);
    }
}

// Non-PSD cones should pass through unchanged
TEST_F(ChordalTest, NonPsdConesUnchanged) {
    // 2 nonneg + PSD(3) = m = 2 + 6 = 8
    int64_t n = 3, m = 8;
    // A: identity-like, all rows active
    std::vector<int64_t> A_colptr = {0, 3, 5, 8};
    std::vector<int64_t> A_rowind = {0, 1, 2, 3, 4, 5, 6, 7};

    std::vector<int64_t> psd_dims = {3};

    auto info = ChordalInfo::analyze(
        A_colptr.data(), A_rowind.data(), n, m,
        psd_dims.data(), 1,
        0, 2, 0, 0, 0,  // 2 nonneg cones
        "clique_graph"
    );

    // Dense PSD with all rows active → no decomposition
    EXPECT_FALSE(info.is_decomposed());
}

// ============================================================================
// Symbolic LDL tests
// ============================================================================

class SymbolicLDLTest : public ::testing::Test {};

// Test AMD + symbolic factorization on a small matrix
TEST_F(SymbolicLDLTest, SmallMatrix) {
    // 3×3 identity (trivial case)
    // CSC upper triangle: colptr = [0,1,2,3], rowind = [0,1,2]
    std::vector<int64_t> colptr = {0, 1, 2, 3};
    std::vector<int64_t> rowind = {0, 1, 2};

    auto result = symbolic_ldl(colptr.data(), rowind.data(), 3);

    EXPECT_EQ(result.n, 3);
    EXPECT_EQ(result.perm.size(), 3u);
    // L_colptr.back() = total nnz in L (including diagonal or off-diagonal)
    // For identity, L may be sparse — just check it's valid
    EXPECT_GT(result.L_colptr.size(), 0u);
}

// Test on arrow/star pattern (creates fill-in)
TEST_F(SymbolicLDLTest, ArrowPattern) {
    // 4×4 arrow matrix: row 0 connected to all others
    // Upper triangle CSC:
    // col 0: rows 0
    // col 1: rows 0, 1
    // col 2: rows 0, 2
    // col 3: rows 0, 3
    std::vector<int64_t> colptr = {0, 1, 3, 5, 7};
    std::vector<int64_t> rowind = {0, 0, 1, 0, 2, 0, 3};

    auto result = symbolic_ldl(colptr.data(), rowind.data(), 4);

    EXPECT_EQ(result.n, 4);
    EXPECT_EQ(result.perm.size(), 4u);
    // AMD should push the dense node (0) to the end
    // L should exist and be valid
    EXPECT_GT(result.L_colptr.back(), 0);
}

// ============================================================================
// SuperNodeTree tests
// ============================================================================

class SuperNodeTreeTest : public ::testing::Test {};

TEST_F(SuperNodeTreeTest, ConstructFromLDL) {
    // Build a simple L factor pattern: 3×3 chain graph
    // L (lower triangle, CSC):
    // col 0: rows 0, 1
    // col 1: rows 1, 2
    // col 2: row 2
    std::vector<int64_t> colptr = {0, 2, 4, 5};
    std::vector<int64_t> rowind = {0, 1, 1, 2, 2};

    SuperNodeTree tree(colptr.data(), rowind.data(), 3);

    EXPECT_GT(tree.n_cliques, 0);
    std::cout << "Chain graph: n_cliques=" << tree.n_cliques << std::endl;
}

// ============================================================================
// Clique merging
// ============================================================================

// Strictly lower CSC pattern of the Cholesky factor of an n×n banded matrix
// with the given bandwidth, in natural order (no fill).
void banded_L(int64_t n, int64_t bandwidth,
              std::vector<int64_t>& colptr, std::vector<int64_t>& rowind) {
    colptr.assign(1, 0);
    rowind.clear();
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = j + 1; i <= std::min(j + bandwidth, n - 1); ++i) rowind.push_back(i);
        colptr.push_back(rowind.size());
    }
}

// The cliques (supernode plus separator) of a merged tree, each sorted, in
// lexicographic order.
std::vector<std::vector<int64_t>> cliques_of(const SuperNodeTree& tree) {
    std::vector<std::vector<int64_t>> out;
    for (int64_t i = 0; i < tree.n_cliques; ++i) {
        int64_t c = tree.get_post_order(i);
        std::vector<int64_t> clique(tree.snode[c].begin(), tree.snode[c].end());
        clique.insert(clique.end(), tree.separators[c].begin(), tree.separators[c].end());
        std::sort(clique.begin(), clique.end());
        out.push_back(clique);
    }
    std::sort(out.begin(), out.end());
    return out;
}

// Analyze a PSD(dim) cone whose only nonzeros are the diagonal and first
// superdiagonal, as in #83's example.
ChordalInfo analyze_tridiagonal_psd(int64_t dim, const std::string& merge_method) {
    // svec is the column-major upper triangle: (i, j), i <= j, is j(j+1)/2 + i
    std::vector<int64_t> rows;
    for (int64_t j = 0; j < dim; ++j) {
        if (j > 0) rows.push_back(j * (j + 1) / 2 + j - 1);
        rows.push_back(j * (j + 1) / 2 + j);
    }
    int64_t n = rows.size(), m = dim * (dim + 1) / 2;
    std::vector<int64_t> A_colptr(n + 1);
    std::iota(A_colptr.begin(), A_colptr.end(), 0);
    std::unique_ptr<bool[]> b_mask(new bool[m]());
    std::vector<int64_t> psd_dims = {dim};
    return ChordalInfo::analyze(A_colptr.data(), rows.data(), n, m, psd_dims.data(), 1,
                                0, 0, 0, 0, 0, merge_method, b_mask.get());
}

class MergeMethodTest : public ::testing::Test {};

// #83: on a tridiagonal PSD(8) the CPU gives 7 cones of size 2 with
// clique_graph and none, and merges everything with parent_child.
TEST_F(MergeMethodTest, TridiagonalPsd8) {
    for (const std::string method : {"clique_graph", "none"}) {
        auto info = analyze_tridiagonal_psd(8, method);
        ASSERT_TRUE(info.is_decomposed()) << method;
        EXPECT_EQ(info.decomposed_psd_dims, std::vector<int64_t>(7, 2)) << method;
    }
    auto info = analyze_tridiagonal_psd(8, "parent_child");
    EXPECT_FALSE(info.is_decomposed());
}

TEST_F(MergeMethodTest, UnknownMethodThrows) {
    EXPECT_THROW(analyze_tridiagonal_psd(8, "parent-child"), std::invalid_argument);
}

// Bandwidth 4 on 7 vertices gives the cliques {0..4}, {1..5}, {2..6} in a
// path. Both clique-graph edges weigh 34, and the CPU (Rust's max_by_key
// takes the last maximum) merges the last two.
TEST_F(MergeMethodTest, CliqueGraphTieBreak) {
    std::vector<int64_t> colptr, rowind;
    banded_L(7, 4, colptr, rowind);
    using Cliques = std::vector<std::vector<int64_t>>;

    SuperNodeTree none(colptr.data(), rowind.data(), 7);
    merge_cliques(none, "none");
    EXPECT_EQ(cliques_of(none), (Cliques{{0, 1, 2, 3, 4}, {1, 2, 3, 4, 5}, {2, 3, 4, 5, 6}}));

    SuperNodeTree clique_graph(colptr.data(), rowind.data(), 7);
    merge_cliques(clique_graph, "clique_graph");
    EXPECT_EQ(cliques_of(clique_graph), (Cliques{{0, 1, 2, 3, 4}, {1, 2, 3, 4, 5, 6}}));

    SuperNodeTree parent_child(colptr.data(), rowind.data(), 7);
    merge_cliques(parent_child, "parent_child");
    EXPECT_EQ(cliques_of(parent_child), (Cliques{{0, 1, 2, 3, 4, 5, 6}}));
}
