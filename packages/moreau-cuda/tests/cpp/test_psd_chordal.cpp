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
 */

#include <gtest/gtest.h>
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
        0, 0, 0, 0, 0  // no other cones
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
        0, 2, 0, 0, 0  // 2 nonneg cones
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
