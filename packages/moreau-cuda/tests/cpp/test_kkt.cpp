// test_kkt.cpp
#include <gtest/gtest.h>
#include "moreau/kkt/kkt.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/matrix/csr.hpp"
#include <vector>
#include <algorithm>
#include <set>
#include <iostream>

using namespace moreau;

class KKTConstructionTest : public ::testing::Test {
protected:
    void SetUp() override { batchSize = 8; }
    void TearDown() override {}

    // P = I_n
    void createDiagonalP(int n, std::vector<int64_t>& rowOffsets,
                         std::vector<int64_t>& colIndices) {
        rowOffsets.resize(n + 1);
        colIndices.resize(n);
        for (int i = 0; i <= n; i++) rowOffsets[i] = i;
        for (int i = 0; i < n; i++)  colIndices[i] = i;
    }

    // A = I_m
    void createIdentityA(int m, std::vector<int64_t>& rowOffsets,
                         std::vector<int64_t>& colIndices) {
        rowOffsets.resize(m + 1);
        colIndices.resize(m);
        for (int i = 0; i <= m; i++) rowOffsets[i] = i;
        for (int i = 0; i < m; i++)  colIndices[i] = i;
    }

    // P with missing diagonals in `missingDiags`
    void createSparseP(int n, std::vector<int64_t>& rowOffsets,
                       std::vector<int64_t>& colIndices,
                       const std::set<int>& missingDiags) {
        rowOffsets.resize(n + 1);
        colIndices.clear();
        rowOffsets[0] = 0;
        for (int i = 0; i < n; i++) {
            if (!missingDiags.count(i)) colIndices.push_back(i);
            rowOffsets[i + 1] = static_cast<int64_t>(colIndices.size());
        }
    }

    // Download KKT upper-tri structure to dense 0/1 matrix
    std::vector<std::vector<int>> downloadKKTStructure(const CSR& K) {
        int dim = K.n();
        std::vector<int64_t> rowOffsets(dim + 1);
        std::vector<int64_t> colIndices(K.nnz());

        cudaMemcpy(rowOffsets.data(), K.rowOffsets(), (dim + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(colIndices.data(), K.colIndices(), K.nnz() * sizeof(int64_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        std::vector<std::vector<int>> dense(dim, std::vector<int>(dim, 0));
        for (int i = 0; i < dim; i++) {
            for (int64_t p = rowOffsets[i]; p < rowOffsets[i + 1]; ++p) {
                int j = static_cast<int>(colIndices[p]);
                if (j >= i) {
                    dense[i][j] = 1;
                } else {
                    // should never happen (we store upper triangle only)
                    EXPECT_GE(j, i) << "Lower triangular entry found at (" << i << "," << j << ")";
                }
            }
        }
        return dense;
    }

    void printStructure(const std::vector<std::vector<int>>& dense) {
        std::cout << "\nMatrix structure (upper triangular, "
                  << dense.size() << "x" << dense.size() << "):\n";
        for (size_t i = 0; i < dense.size(); i++) {
            for (size_t j = 0; j < dense[i].size(); j++) std::cout << dense[i][j] << " ";
            std::cout << "\n";
        }
    }

    // Verify block structure
    void verifyKKTBlockStructure(const CSR& K, int n, int m,
                                 const Cones& cones, bool print = false) {
        auto dense = downloadKKTStructure(K);
        if (print) printStructure(dense);

        ASSERT_EQ(static_cast<int>(dense.size()), n + m) << "Matrix dimension mismatch";

        // P block diagonals (0..n-1, 0..n-1)
        for (int i = 0; i < n; i++) {
            EXPECT_EQ(dense[i][i], 1) << "P block missing diagonal at (" << i << "," << i << ")";
        }

        // A^T block: identity A ⇒ (i, n+i) present for i < m
        for (int i = 0; i < std::min(n, m); i++) {
            EXPECT_EQ(dense[i][n + i], 1) << "A^T missing at (" << i << "," << (n + i) << ")";
        }

        // No lower in top-left
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++)
                EXPECT_EQ(dense[i][j], 0) << "Lower entry in P block at (" << i << "," << j << ")";

        // H block rows begin at n
        int row = n;

        // Zero cones: diagonal only
        for (int i = 0; i < cones.numZeroCones; i++, row++) {
            EXPECT_EQ(dense[row][row], 1) << "Zero cone missing diag at row " << row;
            for (int j = row + 1; j < n + m; j++)
                EXPECT_EQ(dense[row][j], 0) << "Zero cone off-diag at (" << row << "," << j << ")";
        }

        // Nonneg cones: diagonal only
        for (int i = 0; i < cones.numNonnegCones; i++, row++) {
            EXPECT_EQ(dense[row][row], 1) << "Nonneg cone missing diag at row " << row;
            for (int j = row + 1; j < n + m; j++)
                EXPECT_EQ(dense[row][j], 0) << "Nonneg cone off-diag at (" << row << "," << j << ")";
        }

        // SOC cones: 3x3 upper
        for (int b = 0; b < cones.numSocCones; b++) {
            verify3x3BlockUpperTri(dense, row, "SOC", n, m);
            row += 3;
        }
        for (int b = 0; b < cones.numExpCones; b++) {
            verify3x3BlockUpperTri(dense, row, "Exp", n, m);
            row += 3;
        }
        for (int b = 0; b < cones.numPowerCones; b++) {
            verify3x3BlockUpperTri(dense, row, "Power", n, m);
            row += 3;
        }

        EXPECT_EQ(row, n + m) << "Did not process all cone rows";
    }

    void verify3x3BlockUpperTri(const std::vector<std::vector<int>>& dense,
                                int startRow, const std::string& coneType, int n, int m) {
        // Row 0
        EXPECT_EQ(dense[startRow + 0][startRow + 0], 1)
            << coneType << " cone missing (0,0) at " << (startRow + 0);
        EXPECT_EQ(dense[startRow + 0][startRow + 1], 1)
            << coneType << " cone missing (0,1) at (" << (startRow + 0) << "," << (startRow + 1) << ")";
        EXPECT_EQ(dense[startRow + 0][startRow + 2], 1)
            << coneType << " cone missing (0,2) at (" << (startRow + 0) << "," << (startRow + 2) << ")";
        // Row 1
        EXPECT_EQ(dense[startRow + 1][startRow + 1], 1)
            << coneType << " cone missing (1,1) at " << (startRow + 1);
        EXPECT_EQ(dense[startRow + 1][startRow + 2], 1)
            << coneType << " cone missing (1,2) at (" << (startRow + 1) << "," << (startRow + 2) << ")";
        // Row 2
        EXPECT_EQ(dense[startRow + 2][startRow + 2], 1)
            << coneType << " cone missing (2,2) at " << (startRow + 2);

        // No entries outside block to the right
        for (int i = 0; i < 3; i++)
            for (int j = startRow + 3; j < n + m; j++)
                EXPECT_EQ(dense[startRow + i][j], 0)
                    << coneType << " cone has entry outside 3x3 at ("
                    << (startRow + i) << "," << j << ")";
    }

    void verifyKKTStructure(const CSR& K, int64_t n, int64_t m,
                            int64_t expectedNnz, int expectedBatchSize = -1) {
        EXPECT_EQ(K.n(), n + m);
        EXPECT_EQ(K.m(), n + m);
        EXPECT_EQ(K.nnz(), expectedNnz);
        if (expectedBatchSize == -1) expectedBatchSize = batchSize;
        EXPECT_EQ(K.batchSize(), expectedBatchSize);
    }

    int batchSize;
};

// ---- Tests ----

TEST_F(KKTConstructionTest, AllZeroConesWithStructure) {
    int n = 5, m = 3;

    std::vector<int64_t> P_ro, P_ci;
    createDiagonalP(n, P_ro, P_ci);

    std::vector<int64_t> A_ro, A_ci;
    createIdentityA(m, A_ro, A_ci);

    Cones cones{};
    cones.numZeroCones = 3;

    KKTData kkt(n, m, batchSize,
                P_ro.data(), P_ci.data(), (int64_t)P_ci.size(),
                A_ro.data(), A_ci.data(), (int64_t)A_ci.size(),
                cones);

    verifyKKTStructure(kkt.KKT, n, m, /*expectedNnz=*/11);
    verifyKKTBlockStructure(kkt.KKT, n, m, cones);
}

TEST_F(KKTConstructionTest, AllNonnegConesWithStructure) {
    int n = 4, m = 4;

    std::vector<int64_t> P_ro, P_ci;
    createDiagonalP(n, P_ro, P_ci);

    std::vector<int64_t> A_ro, A_ci;
    createIdentityA(m, A_ro, A_ci);

    Cones cones{};
    cones.numNonnegCones = 4;

    KKTData kkt(n, m, batchSize,
                P_ro.data(), P_ci.data(), (int64_t)P_ci.size(),
                A_ro.data(), A_ci.data(), (int64_t)A_ci.size(),
                cones);

    verifyKKTStructure(kkt.KKT, n, m, /*expectedNnz=*/12);
    verifyKKTBlockStructure(kkt.KKT, n, m, cones);
}

TEST_F(KKTConstructionTest, SingleSocConeWithStructure) {
    int n = 2, m = 3;  // one 3D SOC

    std::vector<int64_t> P_ro, P_ci;
    createDiagonalP(n, P_ro, P_ci);

    // A should be m×n (3×2), not m×m. Create first n rows of identity.
    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci;
    for (int i = 0; i <= m; i++) {
        A_ro[i] = (i < n) ? i : n;  // First n rows have 1 entry each, rest have 0
    }
    for (int i = 0; i < n; i++) {
        A_ci.push_back(i);  // Diagonal entries only for first n rows
    }

    Cones cones{};
    cones.socConeDims = {3};
    cones.numSocCones = 1;

    KKTData kkt(n, m, batchSize,
                P_ro.data(), P_ci.data(), (int64_t)P_ci.size(),
                A_ro.data(), A_ci.data(), (int64_t)A_ci.size(),
                cones);

    verifyKKTStructure(kkt.KKT, n, m, /*expectedNnz=*/10);
    verifyKKTBlockStructure(kkt.KKT, n, m, cones);
}

TEST_F(KKTConstructionTest, MixedConesExactExample) {
    // Example from comment
    int n = 2;
    int m = 11; // 1 zero + 1 nonneg + 3 SOC + 3 EXP + 3 POW

    // P = I_2
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    // A is 11x2 with first two rows identity
    std::vector<int64_t> A_ro = {0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
    std::vector<int64_t> A_ci = {0, 1};

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 1;
    cones.socConeDims = {3};
    cones.numSocCones = 1;
    cones.numExpCones = 1;
    cones.numPowerCones = 1;

    KKTData kkt(n, m, batchSize,
                P_ro.data(), P_ci.data(), (int64_t)P_ci.size(),
                A_ro.data(), A_ci.data(), (int64_t)A_ci.size(),
                cones);

    // 2 (P diag) + 2 (A^T, because first two A rows are identity) + 1 (zero cone diag)
    // + 1 (nonneg diag) + 3+2+1 for each of 3×3 blocks (total 3*(3+2+1)=18) => 24
    verifyKKTStructure(kkt.KKT, n, m, /*expectedNnz=*/24);

    auto dense = downloadKKTStructure(kkt.KKT);

    // Expected (upper) 13x13
    std::vector<std::vector<int>> expected = {
        {1,0,1,0,0,0,0,0,0,0,0,0,0},
        {0,1,0,1,0,0,0,0,0,0,0,0,0},
        {0,0,1,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,1,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,1,1,1,0,0,0,0,0,0},
        {0,0,0,0,0,1,1,0,0,0,0,0,0},
        {0,0,0,0,0,0,1,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,1,1,1,0,0,0},
        {0,0,0,0,0,0,0,0,1,1,0,0,0},
        {0,0,0,0,0,0,0,0,0,1,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,1,1,1},
        {0,0,0,0,0,0,0,0,0,0,0,1,1},
        {0,0,0,0,0,0,0,0,0,0,0,0,1},
    };

    ASSERT_EQ(dense.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        for (size_t j = 0; j < expected[i].size(); ++j)
            EXPECT_EQ(dense[i][j], expected[i][j]) << "Mismatch at (" << i << "," << j << ")";

    verifyKKTBlockStructure(kkt.KKT, n, m, cones, true);
}

TEST_F(KKTConstructionTest, PMissingDiagonalsWithStructure) {
    int n = 5, m = 2;

    std::vector<int64_t> P_ro, P_ci;
    std::set<int> missingDiags = {1, 3};
    createSparseP(n, P_ro, P_ci, missingDiags);

    std::vector<int64_t> A_ro, A_ci;
    createIdentityA(m, A_ro, A_ci);

    Cones cones{};
    cones.numZeroCones = 2;

    KKTData kkt(n, m, batchSize,
                P_ro.data(), P_ci.data(), (int64_t)P_ci.size(),
                A_ro.data(), A_ci.data(), (int64_t)A_ci.size(),
                cones);

    verifyKKTStructure(kkt.KKT, n, m, /*expectedNnz=*/9);
    verifyKKTBlockStructure(kkt.KKT, n, m, cones);

    // Ensure added diagonals exist
    auto dense = downloadKKTStructure(kkt.KKT);
    EXPECT_EQ(dense[1][1], 1) << "Missing diagonal at (1,1) not added";
    EXPECT_EQ(dense[3][3], 1) << "Missing diagonal at (3,3) not added";
}

TEST_F(KKTConstructionTest, PopulateValuesFromP) {
    int n = 4, m = 3;

    // Create non-trivial tridiagonal P matrix (upper triangle only)
    // P = [1  2  0  0]
    //     [-  3  4  0]
    //     [-  -  5  6]
    //     [-  -  -  7]
    std::vector<int64_t> P_ro = {0, 2, 4, 6, 7};  // row offsets
    std::vector<int64_t> P_ci = {0, 1,    // row 0: cols 0,1
                                  1, 2,    // row 1: cols 1,2
                                  2, 3,    // row 2: cols 2,3
                                  3};      // row 3: col 3
    int64_t nnzP = 7;

    // Create non-trivial sparse A matrix (3×4)
    // A = [1  0  2  0]
    //     [0  3  0  4]
    //     [5  0  0  6]
    std::vector<int64_t> A_ro = {0, 2, 4, 6};  // row offsets
    std::vector<int64_t> A_ci = {0, 2,    // row 0: cols 0,2
                                  1, 3,    // row 1: cols 1,3
                                  0, 3};   // row 2: cols 0,3
    int64_t nnzA = 6;

    Cones cones{};
    cones.numNonnegCones = 3;  // m=3 nonneg cones

    // Build the KKT structure
    KKTData kkt(n, m, batchSize,
                P_ro.data(), P_ci.data(), nnzP,
                A_ro.data(), A_ci.data(), nnzA,
                cones);

    // Create a P matrix on device with test values
    CSR P(n, n, nnzP, batchSize);
    P.indicesCpuToGpu(P_ro.data(), P_ci.data());

    // Fill P with distinct values per batch and nonzero
    // Pattern: batch b, nonzero i -> value = (b+1)*10 + (i+1)
    std::vector<double> P_values(batchSize * nnzP);
    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < nnzP; ++i) {
            P_values[b * nnzP + i] = (b + 1) * 10.0 + (i + 1);
        }
    }
    P.valuesCpuToGpu(P_values.data());

    // Create an A matrix on device with test values
    CSR A(m, n, nnzA, batchSize);
    A.indicesCpuToGpu(A_ro.data(), A_ci.data());

    // Fill A with distinct values per batch and nonzero
    // Pattern: batch b, nonzero i -> value = (b+1)*100 + (i+1)
    std::vector<double> A_values(batchSize * nnzA);
    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < nnzA; ++i) {
            A_values[b * nnzA + i] = (b + 1) * 100.0 + (i + 1);
        }
    }
    A.valuesCpuToGpu(A_values.data());
    cudaDeviceSynchronize();

    // Populate KKT from P and A
    kkt.populate(P, A);
    cudaDeviceSynchronize();

    // Download KKT values and verify
    std::vector<double> KKT_values(batchSize * kkt.KKT.nnz());
    kkt.KKT.valuesGpuToCpu(KKT_values.data());
    cudaDeviceSynchronize();

    // Download P index map to verify mapping
    std::vector<int64_t> P_idx_map(nnzP);
    cudaMemcpy(P_idx_map.data(), kkt.PNNZIndex(), nnzP * sizeof(int64_t), cudaMemcpyDeviceToHost);

    // Download A index map to verify mapping
    std::vector<int64_t> A_idx_map(nnzA);
    cudaMemcpy(A_idx_map.data(), kkt.ANNZIndex(), nnzA * sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Verify that P values were copied to correct locations in KKT
    std::cout << "\nVerifying P matrix population (nnzP=" << nnzP << "):\n";
    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < nnzP; ++i) {
            int64_t kkt_idx = P_idx_map[i];
            ASSERT_GE(kkt_idx, 0) << "Invalid P index map at i=" << i;
            ASSERT_LT(kkt_idx, kkt.KKT.nnz()) << "P index map out of bounds";

            double expected = (b + 1) * 10.0 + (i + 1);
            double actual = KKT_values[b * kkt.KKT.nnz() + kkt_idx];

            EXPECT_DOUBLE_EQ(actual, expected)
                << "Batch " << b << ", P nonzero " << i
                << " not copied correctly to KKT[" << kkt_idx << "]";
        }
    }

    // Verify that A values were copied to correct locations in KKT
    std::cout << "Verifying A matrix population (nnzA=" << nnzA << "):\n";
    for (int b = 0; b < batchSize; ++b) {
        for (int i = 0; i < nnzA; ++i) {
            int64_t kkt_idx = A_idx_map[i];
            ASSERT_GE(kkt_idx, 0) << "Invalid A index map at i=" << i;
            ASSERT_LT(kkt_idx, kkt.KKT.nnz()) << "A index map out of bounds";

            double expected = (b + 1) * 100.0 + (i + 1);
            double actual = KKT_values[b * kkt.KKT.nnz() + kkt_idx];

            EXPECT_DOUBLE_EQ(actual, expected)
                << "Batch " << b << ", A nonzero " << i
                << " not copied correctly to KKT[" << kkt_idx << "]";
        }
    }
    std::cout << "All values verified successfully!\n";
}

// Main
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
