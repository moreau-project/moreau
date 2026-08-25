// test_batch_scaling_bug.cpp
// Test for the 100x scaling bug in batched solutions

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>

using namespace moreau;

class BatchScalingBugTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetLastError();
    }
    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// Build a Sudoku-like constraint matrix
void buildSudokuConstraints(
    int gridSize,  // e.g., 9 for 9x9 Sudoku
    std::vector<int64_t>& P_ro,
    std::vector<int64_t>& P_ci,
    std::vector<int64_t>& A_ro,
    std::vector<int64_t>& A_ci,
    int64_t& n,
    int64_t& m_eq,
    int64_t& m_total,
    int64_t& nnzP,
    int64_t& nnzA
) {
    n = gridSize * gridSize * gridSize;  // 729 for 9x9
    int boxSize = (int)sqrt(gridSize);   // 3 for 9x9

    // Count constraints
    // Cell constraints: gridSize^2 (81)
    // Row constraints: gridSize^2 (81)
    // Column constraints: gridSize^2 (81)
    // Box constraints: gridSize^2 (81)
    m_eq = 4 * gridSize * gridSize;  // 324 for 9x9

    // Bounds: lower and upper for each variable
    m_total = m_eq + 2 * n;  // 324 + 2*729 = 1782

    // P is diagonal
    nnzP = n;
    P_ro.resize(n + 1);
    P_ci.resize(n);
    for (int64_t i = 0; i <= n; i++) P_ro[i] = i;
    for (int64_t i = 0; i < n; i++) P_ci[i] = i;

    // A: equality constraints + bound constraints
    // Each equality constraint has gridSize nonzeros
    // Each bound constraint has 1 nonzero
    nnzA = m_eq * gridSize + 2 * n;

    A_ro.resize(m_total + 1);
    A_ci.resize(nnzA);

    auto var_idx = [gridSize](int i, int j, int k) {
        return i * gridSize * gridSize + j * gridSize + k;
    };

    int64_t row = 0;
    int64_t nnz_idx = 0;

    // Cell constraints: for each cell (i,j), sum over digits k
    for (int i = 0; i < gridSize; i++) {
        for (int j = 0; j < gridSize; j++) {
            A_ro[row] = nnz_idx;
            for (int k = 0; k < gridSize; k++) {
                A_ci[nnz_idx++] = var_idx(i, j, k);
            }
            row++;
        }
    }

    // Row constraints: for each row i and digit k, sum over columns j
    for (int i = 0; i < gridSize; i++) {
        for (int k = 0; k < gridSize; k++) {
            A_ro[row] = nnz_idx;
            for (int j = 0; j < gridSize; j++) {
                A_ci[nnz_idx++] = var_idx(i, j, k);
            }
            row++;
        }
    }

    // Column constraints: for each column j and digit k, sum over rows i
    for (int j = 0; j < gridSize; j++) {
        for (int k = 0; k < gridSize; k++) {
            A_ro[row] = nnz_idx;
            for (int i = 0; i < gridSize; i++) {
                A_ci[nnz_idx++] = var_idx(i, j, k);
            }
            row++;
        }
    }

    // Box constraints
    for (int box_i = 0; box_i < boxSize; box_i++) {
        for (int box_j = 0; box_j < boxSize; box_j++) {
            for (int k = 0; k < gridSize; k++) {
                A_ro[row] = nnz_idx;
                for (int di = 0; di < boxSize; di++) {
                    for (int dj = 0; dj < boxSize; dj++) {
                        int i = box_i * boxSize + di;
                        int j = box_j * boxSize + dj;
                        A_ci[nnz_idx++] = var_idx(i, j, k);
                    }
                }
                row++;
            }
        }
    }

    // Lower bounds: -x <= -eps  (row has one entry: -1)
    for (int64_t i = 0; i < n; i++) {
        A_ro[row] = nnz_idx;
        A_ci[nnz_idx++] = i;
        row++;
    }

    // Upper bounds: x <= 1-eps  (row has one entry: 1)
    for (int64_t i = 0; i < n; i++) {
        A_ro[row] = nnz_idx;
        A_ci[nnz_idx++] = i;
        row++;
    }

    A_ro[row] = nnz_idx;  // final row offset
}

TEST_F(BatchScalingBugTest, SimplexComparison) {
    // Use smaller grid for faster testing
    int gridSize = 4;  // 4x4 Sudoku (easier)

    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    int64_t n, m_eq, m, nnzP, nnzA;
    buildSudokuConstraints(gridSize, P_ro, P_ci, A_ro, A_ci, n, m_eq, m, nnzP, nnzA);

    std::cout << "Problem: n=" << n << ", m_eq=" << m_eq << ", m=" << m << std::endl;

    double eps = 0.05;
    double P_diag = 0.01;

    // Prepare data for batch=1 and batch=2
    for (int batchSize : {1, 2}) {
        std::cout << "\n=== Testing batchSize=" << batchSize << " ===" << std::endl;

        std::vector<double> P_val(nnzP * batchSize);
        std::vector<double> A_val(nnzA * batchSize);
        std::vector<double> q(n * batchSize);
        std::vector<double> b(m * batchSize);

        std::mt19937 rng(42);
        std::uniform_real_distribution<double> q_dist(-0.1, 0.1);

        for (int batch = 0; batch < batchSize; batch++) {
            // P = P_diag * I
            for (int64_t i = 0; i < nnzP; i++) {
                P_val[batch * nnzP + i] = P_diag;
            }

            // A values: equality rows have 1.0, lower bounds have -1.0, upper bounds have 1.0
            int64_t val_idx = 0;
            // Equality constraints: all 1.0
            for (int64_t i = 0; i < m_eq * gridSize; i++) {
                A_val[batch * nnzA + val_idx++] = 1.0;
            }
            // Lower bounds: -1.0
            for (int64_t i = 0; i < n; i++) {
                A_val[batch * nnzA + val_idx++] = -1.0;
            }
            // Upper bounds: 1.0
            for (int64_t i = 0; i < n; i++) {
                A_val[batch * nnzA + val_idx++] = 1.0;
            }

            // q: random small values
            for (int64_t i = 0; i < n; i++) {
                q[batch * n + i] = q_dist(rng);
            }

            // b: equality RHS = 1, bounds
            for (int64_t i = 0; i < m_eq; i++) {
                b[batch * m + i] = 1.0;
            }
            for (int64_t i = 0; i < n; i++) {
                b[batch * m + m_eq + i] = -eps;  // -x <= -eps
            }
            for (int64_t i = 0; i < n; i++) {
                b[batch * m + m_eq + n + i] = 1.0 - eps;  // x <= 1-eps
            }

            // Add clue as soft bias: tighten bounds for clue digit toward 1,
            // but leave a feasible gap (no zero-width boxes that kill IPM conditioning)
            {
                int64_t cell = batch;  // different cell per batch but deterministic
                int digit = 0;
                int i = cell / gridSize;
                int j = cell % gridSize;

                std::cout << "    Batch " << batch << " clue: cell=" << cell << " (i=" << i << ",j=" << j << "), digit=" << digit << std::endl;

                for (int k = 0; k < gridSize; k++) {
                    int64_t var = i * gridSize * gridSize + j * gridSize + k;
                    if (k == digit) {
                        b[batch * m + m_eq + var] = -0.8;           // x >= 0.8
                        b[batch * m + m_eq + n + var] = 1.0 - eps;  // x <= 1-eps
                    } else {
                        b[batch * m + m_eq + var] = -eps;           // x >= eps
                        b[batch * m + m_eq + n + var] = 0.2;        // x <= 0.2
                    }
                }
            }
        }

        Cones cones{};
        cones.numZeroCones = m_eq;
        cones.numNonnegCones = 2 * n;

        Settings settings;
        settings.verbose = false;

        CompiledSolver solver(n, m, batchSize,
                      P_ro.data(), P_ci.data(), nnzP,
                      A_ro.data(), A_ci.data(), nnzA,
                      cones, settings);

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
        cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
        cudaMalloc(&d_q, sizeof(double) * n * batchSize);
        cudaMalloc(&d_b, sizeof(double) * m * batchSize);

        cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

        solver.solveAll(d_P, d_A, d_q, d_b);

        std::vector<double> x_sol(n * batchSize);
        std::vector<int32_t> status(batchSize);
        cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

        // Debug: print some b values for each batch
        std::cout << "  b values (eq[0-2], lb[0-2], ub[0-2]) per batch:" << std::endl;
        for (int batch = 0; batch < batchSize; batch++) {
            std::cout << "    Batch " << batch << ": eq=";
            for (int i = 0; i < 3; i++) std::cout << b[batch * m + i] << " ";
            std::cout << " lb=";
            for (int i = 0; i < 3; i++) std::cout << b[batch * m + m_eq + i] << " ";
            std::cout << " ub=";
            for (int i = 0; i < 3; i++) std::cout << b[batch * m + m_eq + n + i] << " ";
            std::cout << std::endl;
        }

        // Check q values
        std::cout << "  q values [0-5] per batch:" << std::endl;
        for (int batch = 0; batch < batchSize; batch++) {
            std::cout << "    Batch " << batch << ": ";
            for (int i = 0; i < 6; i++) std::cout << q[batch * n + i] << " ";
            std::cout << std::endl;
        }

        for (int batch = 0; batch < batchSize; batch++) {
            double sum = 0.0, x_min = 1e10, x_max = -1e10;
            for (int64_t i = 0; i < n; i++) {
                double x = x_sol[batch * n + i];
                sum += x;
                x_min = std::min(x_min, x);
                x_max = std::max(x_max, x);
            }
            std::cout << "  Batch " << batch << ": status=" << status[batch]
                      << ", sum=" << sum << ", x_min=" << x_min << ", x_max=" << x_max << std::endl;

            // For 4x4 Sudoku with gridSize cells, expect sum ≈ gridSize^2 (16 cells, each summing to 1)
            double expected_sum = gridSize * gridSize;
            EXPECT_NEAR(sum, expected_sum, expected_sum * 0.1)
                << "Batch " << batch << " sum differs from expected by more than 10%";
        }

        cudaFree(d_P);
        cudaFree(d_A);
        cudaFree(d_q);
        cudaFree(d_b);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
