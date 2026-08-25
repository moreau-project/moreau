// test_woodbury.cpp
// Tests for the Woodbury/Phi-GEMM KKT backend on portfolio-type problems
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/kkt/kkt_woodbury.hpp"
#include "moreau/kkt/woodbury_kernels.cuh"
#include <algorithm>
#include <chrono>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <limits>
#include <numeric>

using namespace moreau;

class WoodburyTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Helper: build a portfolio problem with diagonal P and low-rank factor model.
//
//   min  (1/2) x^T diag(D) x + q^T x
//   s.t. F^T x = b_eq   (k equality constraints, zero cones)
//        x >= 0          (n nonneg cones)
//
// CSR structure:
//   A = [ F^T ]    (k x n)   — zero cones (equality)
//       [ -I  ]    (n x n)   — nonneg cones (x >= 0)
//
// P = diag(D)  (n x n diagonal, upper-triangular CSR)
struct PortfolioProblem {
    int64_t n, k, m;
    int64_t nnzP, nnzA;
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double> P_values;
    std::vector<int64_t> A_ro, A_ci;
    std::vector<double> A_values;
    std::vector<double> q_data;
    std::vector<double> b_data;
    Cones cones;
};

PortfolioProblem buildPortfolio(int64_t n, int64_t k, unsigned seed = 42) {
    PortfolioProblem prob;
    prob.n = n;
    prob.k = k;
    prob.m = k + n;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.1, 2.0);
    std::normal_distribution<double> ndist(0.0, 1.0);

    // P = diag(D), D[i] > 0
    prob.nnzP = n;
    prob.P_ro.resize(n + 1);
    prob.P_ci.resize(n);
    prob.P_values.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        prob.P_ro[i] = i;
        prob.P_ci[i] = i;
        prob.P_values[i] = dist(rng);
    }
    prob.P_ro[n] = n;

    // Generate F (n x k) — factor loadings
    std::vector<double> F(n * k);
    for (int64_t i = 0; i < n * k; ++i)
        F[i] = ndist(rng);

    // A = [F^T; -I]
    // F^T is k x n (each row has n entries)
    // -I  is n x n (each row has 1 entry)
    prob.nnzA = k * n + n;
    prob.A_ro.resize(prob.m + 1);
    prob.A_ci.resize(prob.nnzA);
    prob.A_values.resize(prob.nnzA);

    int64_t nnz_idx = 0;
    // First k rows: F^T (dense rows)
    for (int64_t i = 0; i < k; ++i) {
        prob.A_ro[i] = nnz_idx;
        for (int64_t j = 0; j < n; ++j) {
            prob.A_ci[nnz_idx] = j;
            prob.A_values[nnz_idx] = F[j * k + i]; // F^T[i,j] = F[j,i]
            nnz_idx++;
        }
    }
    // Next n rows: -I
    for (int64_t i = 0; i < n; ++i) {
        prob.A_ro[k + i] = nnz_idx;
        prob.A_ci[nnz_idx] = i;
        prob.A_values[nnz_idx] = -1.0;
        nnz_idx++;
    }
    prob.A_ro[prob.m] = nnz_idx;

    // q = -expected_returns (negative for max return)
    prob.q_data.resize(n);
    for (int64_t i = 0; i < n; ++i)
        prob.q_data[i] = -dist(rng) * 0.1;

    // b = [b_eq; 0]
    // b_eq[0] = 1.0 (budget constraint: sum x = 1)
    // remaining b_eq entries = 0 (other factor constraints)
    // b_nonneg = 0 (x >= 0 → -x + s = 0, s >= 0)
    prob.b_data.resize(prob.m, 0.0);
    prob.b_data[0] = 1.0;

    prob.cones = {};
    prob.cones.numZeroCones = k;
    prob.cones.numNonnegCones = n;

    return prob;
}

// Solve a portfolio problem with both CuDSS and Woodbury, compare results
TEST_F(WoodburyTest, PortfolioSolveMatchesCuDSS) {
    const int64_t n = 20;
    const int64_t k = 3;
    const int64_t batchSize = 1;

    auto prob = buildPortfolio(n, k);

    // Solve with CuDSS
    Settings settings_cudss;
    settings_cudss.ipm.kktSolverType = KKTSolverType::CuDSS;
    settings_cudss.maxIter = 100;
    settings_cudss.verbose = false;
    settings_cudss.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver_cudss(
        prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings_cudss
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * prob.n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * prob.m * batchSize);
    cudaMemcpy(d_P, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q_data.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b_data.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

    solver_cudss.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_cudss(prob.n), s_cudss(prob.m);
    std::vector<int32_t> status_cudss(batchSize);
    cudaMemcpy(x_cudss.data(), solver_cudss.solution.x.data(),
               sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_cudss.data(), solver_cudss.solution.s.data(),
               sizeof(double) * prob.m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_cudss.data(), solver_cudss.solution.status.get(),
               sizeof(int32_t), cudaMemcpyDeviceToHost);

    double obj_cudss;
    cudaMemcpy(&obj_cudss, solver_cudss.solution.obj_val.data(),
               sizeof(double), cudaMemcpyDeviceToHost);

    ASSERT_EQ(status_cudss[0], 1) << "CuDSS solver did not converge (status="
                                   << status_cudss[0] << ")";

    // Solve with Woodbury
    Settings settings_wb;
    settings_wb.ipm.kktSolverType = KKTSolverType::Woodbury;
    settings_wb.maxIter = 100;
    settings_wb.verbose = false;
    settings_wb.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver_wb(
        prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings_wb
    );

    solver_wb.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_wb(prob.n), s_wb(prob.m);
    std::vector<int32_t> status_wb(batchSize);
    cudaMemcpy(x_wb.data(), solver_wb.solution.x.data(),
               sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_wb.data(), solver_wb.solution.s.data(),
               sizeof(double) * prob.m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_wb.data(), solver_wb.solution.status.get(),
               sizeof(int32_t), cudaMemcpyDeviceToHost);

    double obj_wb;
    cudaMemcpy(&obj_wb, solver_wb.solution.obj_val.data(),
               sizeof(double), cudaMemcpyDeviceToHost);

    ASSERT_EQ(status_wb[0], 1) << "Woodbury solver did not converge (status="
                                << status_wb[0] << ")";

    // Compare objectives
    double obj_tol = 1e-4;
    EXPECT_NEAR(obj_wb, obj_cudss, std::max(obj_tol, std::abs(obj_cudss) * 1e-4))
        << "Objective mismatch: CuDSS=" << obj_cudss << " Woodbury=" << obj_wb;

    // Compare x solutions
    for (int64_t i = 0; i < prob.n; ++i) {
        EXPECT_NEAR(x_wb[i], x_cudss[i], 1e-4)
            << "x[" << i << "] mismatch: CuDSS=" << x_cudss[i]
            << " Woodbury=" << x_wb[i];
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Woodbury is opt-in only — verify Auto mode does not select it
TEST_F(WoodburyTest, AutoModeDoesNotSelectWoodbury) {
    auto prob = buildPortfolio(10, 2);

    Settings settings;
    settings.ipm.kktSolverType = KKTSolverType::Auto;
    settings.verbose = false;

    CompiledSolver solver(
        prob.n, prob.m, 1,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings
    );

    EXPECT_NE(solver.kkt->solverType(), KKTSolverType::Woodbury)
        << "Woodbury should not be auto-selected";
}

// Test batched portfolio solve with Woodbury
TEST_F(WoodburyTest, BatchedPortfolioSolve) {
    const int64_t n = 15;
    const int64_t k = 3;
    const int64_t batchSize = 4;

    auto prob = buildPortfolio(n, k);

    Settings settings;
    settings.ipm.kktSolverType = KKTSolverType::Woodbury;
    settings.maxIter = 100;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings
    );

    // Create batched data: same structure, different q vectors
    std::vector<double> P_batch(prob.nnzP * batchSize);
    std::vector<double> A_batch(prob.nnzA * batchSize);
    std::vector<double> q_batch(prob.n * batchSize);
    std::vector<double> b_batch(prob.m * batchSize);

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(0.1, 2.0);

    for (int64_t bi = 0; bi < batchSize; ++bi) {
        // Same P and A across batch
        std::copy(prob.P_values.begin(), prob.P_values.end(),
                  P_batch.begin() + bi * prob.nnzP);
        std::copy(prob.A_values.begin(), prob.A_values.end(),
                  A_batch.begin() + bi * prob.nnzA);
        std::copy(prob.b_data.begin(), prob.b_data.end(),
                  b_batch.begin() + bi * prob.m);
        // Different q per batch
        for (int64_t i = 0; i < prob.n; ++i)
            q_batch[bi * prob.n + i] = -dist(rng) * 0.1;
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * prob.n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * prob.m * batchSize);
    cudaMemcpy(d_P, P_batch.data(), sizeof(double) * prob.nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_batch.data(), sizeof(double) * prob.nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_batch.data(), sizeof(double) * prob.n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_batch.data(), sizeof(double) * prob.m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(batchSize);
    cudaMemcpy(status.data(), solver.solution.status.get(),
               sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    for (int64_t bi = 0; bi < batchSize; ++bi) {
        EXPECT_EQ(status[bi], 1) << "Batch " << bi << " did not converge (status="
                                  << status[bi] << ")";
    }

    // Check that solutions are non-negative and sum approximately to 1
    std::vector<double> x_sol(prob.n * batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(),
               sizeof(double) * prob.n * batchSize, cudaMemcpyDeviceToHost);

    for (int64_t bi = 0; bi < batchSize; ++bi) {
        double sum = 0.0;
        for (int64_t i = 0; i < prob.n; ++i) {
            double xi = x_sol[bi * prob.n + i];
            EXPECT_GE(xi, -1e-6) << "Batch " << bi << " x[" << i << "] = " << xi << " < 0";
            sum += xi;
        }
        // Budget constraint: first zero-cone row is sum = 1
        // But we have k zero-cone constraints and only b[0] = 1
        // The sum is constrained by F^T x = b_eq, which is more complex.
        // Just check feasibility: all x >= 0 and solver converged.
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test that isCompatible returns false for non-portfolio structures
TEST_F(WoodburyTest, DetectionRejectsNonDiagonalP) {
    // Dense P (not diagonal) → Woodbury should not be selected
    int64_t n = 3, m = 4;
    std::vector<int64_t> P_ro = {0, 2, 4, 5};
    std::vector<int64_t> P_ci = {0, 1, 1, 2, 2};
    int64_t nnzP = 5;

    // A = [F^T(1xn); -I(nxn)]
    // F^T row: 3 entries, -I rows: 1 each → nnzA = 3 + 3 = 6
    std::vector<int64_t> A_ro = {0, 3, 4, 5, 6};
    std::vector<int64_t> A_ci = {0, 1, 2, 0, 1, 2};
    int64_t nnzA = 6;

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 3;

    EXPECT_FALSE(WoodburyKKTData::isCompatible(n, m, P_ro.data(), P_ci.data(), nnzP,
                                                A_ro.data(), A_ci.data(), nnzA, cones))
        << "Should reject non-diagonal P";
}

TEST_F(WoodburyTest, DetectionRejectsSOCCones) {
    // Diagonal P but with SOC cones → not supported
    int64_t n = 5;
    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci(n);
    for (int64_t i = 0; i < n; ++i) { P_ro[i] = i; P_ci[i] = i; }
    P_ro[n] = n;

    // Dummy A (doesn't matter, rejected at cone check)
    int64_t m = 6;
    std::vector<int64_t> A_ro(m + 1, 0);
    std::vector<int64_t> A_ci;
    int64_t nnzA = 0;

    Cones cones{};
    cones.numZeroCones = 2;
    cones.numNonnegCones = 3;
    cones.numSocCones = 1;

    EXPECT_FALSE(WoodburyKKTData::isCompatible(n, m, P_ro.data(), P_ci.data(), n,
                                                A_ro.data(), A_ci.data(), nnzA, cones))
        << "Should reject problems with SOC cones";
}

TEST_F(WoodburyTest, DetectionAcceptsPortfolio) {
    auto prob = buildPortfolio(20, 3);
    EXPECT_TRUE(WoodburyKKTData::isCompatible(
        prob.n, prob.m, prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA, prob.cones))
        << "Should accept portfolio-shaped problem";
}

// Test that Woodbury accepts box-constrained problems (x >= lb, x <= ub)
TEST_F(WoodburyTest, DetectionAcceptsBoxConstraints) {
    int64_t n = 5, k = 2;
    int64_t n_nonneg = 2 * n;  // lower + upper bounds
    int64_t m = k + n_nonneg;

    // Diagonal P
    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci(n);
    for (int64_t i = 0; i < n; ++i) { P_ro[i] = i; P_ci[i] = i; }
    P_ro[n] = n;

    // A = [F^T(k×n); -I(n×n); I(n×n)]
    int64_t nnzA = k * n + 2 * n;
    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci(nnzA);

    int64_t idx = 0;
    // Zero-cone rows (dense F^T)
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) A_ci[idx++] = j;
    }
    // Lower bound rows: -I
    for (int64_t i = 0; i < n; ++i) {
        A_ro[k + i] = idx;
        A_ci[idx++] = i;
    }
    // Upper bound rows: I
    for (int64_t i = 0; i < n; ++i) {
        A_ro[k + n + i] = idx;
        A_ci[idx++] = i;
    }
    A_ro[m] = idx;

    Cones cones{};
    cones.numZeroCones = k;
    cones.numNonnegCones = n_nonneg;

    EXPECT_TRUE(WoodburyKKTData::isCompatible(n, m, P_ro.data(), P_ci.data(), n,
                                               A_ro.data(), A_ci.data(), nnzA, cones))
        << "Should accept box-constrained problem (each nonneg row has 1 nnz)";
}

// Test partial bounds (some vars bounded, some not)
TEST_F(WoodburyTest, DetectionAcceptsPartialBounds) {
    int64_t n = 5, k = 2;
    int64_t n_nonneg = 3;  // only 3 of 5 vars have lower bounds
    int64_t m = k + n_nonneg;

    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci(n);
    for (int64_t i = 0; i < n; ++i) { P_ro[i] = i; P_ci[i] = i; }
    P_ro[n] = n;

    int64_t nnzA = k * n + n_nonneg;
    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci(nnzA);

    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) A_ci[idx++] = j;
    }
    // Bounds on vars 0, 2, 4 only
    int64_t bound_vars[] = {0, 2, 4};
    for (int64_t i = 0; i < n_nonneg; ++i) {
        A_ro[k + i] = idx;
        A_ci[idx++] = bound_vars[i];
    }
    A_ro[m] = idx;

    Cones cones{};
    cones.numZeroCones = k;
    cones.numNonnegCones = n_nonneg;

    EXPECT_TRUE(WoodburyKKTData::isCompatible(n, m, P_ro.data(), P_ci.data(), n,
                                               A_ro.data(), A_ci.data(), nnzA, cones))
        << "Should accept partial bounds (n_nonneg < n)";
}

// Test rejection when columns are uncovered (P[j]=0 and no sparse nonneg touches j)
TEST_F(WoodburyTest, DetectionRejectsUncoveredColumns) {
    // HVAC-like: sparse P (some zero diagonals) + partial bounds
    // Columns 1,2 have P=0 and no nonneg bounds → uncovered → rejected
    int64_t n = 6, k = 2;
    int64_t n_nonneg = 3;  // bounds on cols 0, 3, 5 only
    int64_t m = k + n_nonneg;

    // P has entries only at cols 0, 4, 5 (like HVAC: Ti, dqh, ddqh)
    std::vector<int64_t> P_ro = {0, 1, 1, 1, 1, 2, 3};  // n+1 entries
    std::vector<int64_t> P_ci = {0, 4, 5};
    int64_t nnzP = 3;

    int64_t nnzA = k * n + n_nonneg;
    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci(nnzA);
    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) A_ci[idx++] = j;
    }
    // Bounds on cols 0, 3, 5 — cols 1, 2 are uncovered
    int64_t bound_vars[] = {0, 3, 5};
    for (int64_t i = 0; i < n_nonneg; ++i) {
        A_ro[k + i] = idx;
        A_ci[idx++] = bound_vars[i];
    }
    A_ro[m] = idx;

    Cones cones{};
    cones.numZeroCones = k;
    cones.numNonnegCones = n_nonneg;

    EXPECT_FALSE(WoodburyKKTData::isCompatible(n, m, P_ro.data(), P_ci.data(), nnzP,
                                                A_ro.data(), A_ci.data(), nnzA, cones))
        << "Should reject: cols 1,2 have P=0 and no nonneg bounds (uncovered)";
}

// Test acceptance when all columns are covered by bounds (even with sparse P)
TEST_F(WoodburyTest, DetectionAcceptsSparseP_AllColumnsCovered) {
    int64_t n = 6, k = 2;
    int64_t n_nonneg = 6;  // bounds on all columns
    int64_t m = k + n_nonneg;

    // P has entries only at cols 0, 4, 5
    std::vector<int64_t> P_ro = {0, 1, 1, 1, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 4, 5};
    int64_t nnzP = 3;

    int64_t nnzA = k * n + n_nonneg;
    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci(nnzA);
    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) A_ci[idx++] = j;
    }
    for (int64_t i = 0; i < n_nonneg; ++i) {
        A_ro[k + i] = idx;
        A_ci[idx++] = i;
    }
    A_ro[m] = idx;

    Cones cones{};
    cones.numZeroCones = k;
    cones.numNonnegCones = n_nonneg;

    EXPECT_TRUE(WoodburyKKTData::isCompatible(n, m, P_ro.data(), P_ci.data(), nnzP,
                                               A_ro.data(), A_ci.data(), nnzA, cones))
        << "Should accept: all cols covered by P or nonneg bounds";
}

// Test rejection of dense nonneg rows (>1 nnz per row)
TEST_F(WoodburyTest, DetectionAcceptsDenseNonnegRows) {
    // Phase 2: dense nonneg rows (>1 nnz) are accepted and folded into F_all
    int64_t n = 5, k = 2;
    int64_t n_nonneg = 2;
    int64_t m = k + n_nonneg;

    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci(n);
    for (int64_t i = 0; i < n; ++i) { P_ro[i] = i; P_ci[i] = i; }
    P_ro[n] = n;

    // One nonneg row with 2 nonzeros (dense), one with 1 nonzero (sparse)
    int64_t nnzA = k * n + 3;
    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci(nnzA);

    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) A_ci[idx++] = j;
    }
    // First nonneg row: 2 nonzeros (dense → goes into F_all)
    A_ro[k] = idx;
    A_ci[idx++] = 0;
    A_ci[idx++] = 1;
    // Second nonneg row: 1 nonzero (sparse)
    A_ro[k + 1] = idx;
    A_ci[idx++] = 2;
    A_ro[m] = idx;

    Cones cones{};
    cones.numZeroCones = k;
    cones.numNonnegCones = n_nonneg;

    // k_total = k + k_d = 2 + 1 = 3, n = 5, so k_total < n → compatible
    EXPECT_TRUE(WoodburyKKTData::isCompatible(n, m, P_ro.data(), P_ci.data(), n,
                                               A_ro.data(), A_ci.data(), nnzA, cones))
        << "Should accept dense nonneg rows when k_total < n";
}

TEST_F(WoodburyTest, DetectionRejectsLargeSchur) {
    // When k_total >= n, Woodbury should be rejected
    int64_t n = 4, k = 2;
    int64_t k_d = 3;  // 3 dense nonneg rows → k_total = 5 >= n = 4
    int64_t n_nonneg = k_d;
    int64_t m = k + n_nonneg;

    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci(n);
    for (int64_t i = 0; i < n; ++i) { P_ro[i] = i; P_ci[i] = i; }
    P_ro[n] = n;

    // All nonneg rows dense (2 nnz each)
    int64_t nnzA = k * n + k_d * 2;
    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci(nnzA);

    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) A_ci[idx++] = j;
    }
    for (int64_t d = 0; d < k_d; ++d) {
        A_ro[k + d] = idx;
        A_ci[idx++] = 0;
        A_ci[idx++] = 1;
    }
    A_ro[m] = idx;

    Cones cones{};
    cones.numZeroCones = k;
    cones.numNonnegCones = n_nonneg;

    // k_total = 2 + 3 = 5 >= n = 4 → reject
    EXPECT_FALSE(WoodburyKKTData::isCompatible(n, m, P_ro.data(), P_ci.data(), n,
                                                A_ro.data(), A_ci.data(), nnzA, cones))
        << "Should reject when k_total >= n";
}

// Helper: build a box-constrained QP
//   min  (1/2) x^T diag(D) x + q^T x
//   s.t. F^T x = b_eq       (k zero cones)
//        x >= lb             (n nonneg cones: -x + s = -lb)
//        -x >= -ub           (n nonneg cones:  x + s =  ub)
struct BoxQPProblem {
    int64_t n, k, m;
    int64_t nnzP, nnzA;
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double> P_values;
    std::vector<int64_t> A_ro, A_ci;
    std::vector<double> A_values;
    std::vector<double> q_data;
    std::vector<double> b_data;
    Cones cones;
};

BoxQPProblem buildBoxQP(int64_t n, int64_t k, unsigned seed = 42) {
    BoxQPProblem prob;
    prob.n = n;
    prob.k = k;
    int64_t n_nonneg = 2 * n;  // lower + upper bounds
    prob.m = k + n_nonneg;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.1, 2.0);
    std::normal_distribution<double> ndist(0.0, 1.0);

    // Diagonal P
    prob.nnzP = n;
    prob.P_ro.resize(n + 1);
    prob.P_ci.resize(n);
    prob.P_values.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        prob.P_ro[i] = i;
        prob.P_ci[i] = i;
        prob.P_values[i] = dist(rng);
    }
    prob.P_ro[n] = n;

    // F (n x k)
    std::vector<double> F(n * k);
    for (int64_t i = 0; i < n * k; ++i) F[i] = ndist(rng);

    // A = [F^T; -I; I]
    prob.nnzA = k * n + 2 * n;
    prob.A_ro.resize(prob.m + 1);
    prob.A_ci.resize(prob.nnzA);
    prob.A_values.resize(prob.nnzA);

    int64_t idx = 0;
    // Zero-cone rows: F^T (k x n, dense)
    for (int64_t i = 0; i < k; ++i) {
        prob.A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) {
            prob.A_ci[idx] = j;
            prob.A_values[idx] = F[j * k + i];
            idx++;
        }
    }
    // Lower bound rows: -I (x >= lb → -x + s = -lb)
    for (int64_t i = 0; i < n; ++i) {
        prob.A_ro[k + i] = idx;
        prob.A_ci[idx] = i;
        prob.A_values[idx] = -1.0;
        idx++;
    }
    // Upper bound rows: I (-x >= -ub → x + s = ub)
    for (int64_t i = 0; i < n; ++i) {
        prob.A_ro[k + n + i] = idx;
        prob.A_ci[idx] = i;
        prob.A_values[idx] = 1.0;
        idx++;
    }
    prob.A_ro[prob.m] = idx;

    // q
    prob.q_data.resize(n);
    for (int64_t i = 0; i < n; ++i)
        prob.q_data[i] = ndist(rng) * 0.5;

    // b = [b_eq; -lb; ub]
    prob.b_data.resize(prob.m, 0.0);
    prob.b_data[0] = 1.0;  // budget
    for (int64_t i = 0; i < n; ++i) {
        prob.b_data[k + i] = 0.0;      // lb = 0 → -lb = 0
        prob.b_data[k + n + i] = 0.5;  // ub = 0.5
    }

    prob.cones = {};
    prob.cones.numZeroCones = k;
    prob.cones.numNonnegCones = n_nonneg;

    return prob;
}

// Test box-constrained QP solve: Woodbury vs CuDSS
TEST_F(WoodburyTest, BoxQPSolveMatchesCuDSS) {
    const int64_t n = 15;
    const int64_t k = 3;
    const int64_t batchSize = 1;

    auto prob = buildBoxQP(n, k);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP);
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA);
    cudaMalloc(&d_q, sizeof(double) * prob.n);
    cudaMalloc(&d_b, sizeof(double) * prob.m);
    cudaMemcpy(d_P, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q_data.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b_data.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

    // Solve with CuDSS
    Settings settings_cudss;
    settings_cudss.ipm.kktSolverType = KKTSolverType::CuDSS;
    settings_cudss.maxIter = 100;
    settings_cudss.verbose = false;
    settings_cudss.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver_cudss(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings_cudss);
    solver_cudss.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_cudss(prob.n);
    std::vector<int32_t> status_cudss(1);
    double obj_cudss;
    cudaMemcpy(x_cudss.data(), solver_cudss.solution.x.data(), sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_cudss.data(), solver_cudss.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_cudss, solver_cudss.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(status_cudss[0], 1) << "CuDSS did not converge";

    // Solve with Woodbury
    Settings settings_wb;
    settings_wb.ipm.kktSolverType = KKTSolverType::Woodbury;
    settings_wb.maxIter = 100;
    settings_wb.verbose = false;
    settings_wb.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver_wb(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings_wb);
    solver_wb.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_wb(prob.n);
    std::vector<int32_t> status_wb(1);
    double obj_wb;
    cudaMemcpy(x_wb.data(), solver_wb.solution.x.data(), sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_wb.data(), solver_wb.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_wb, solver_wb.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(status_wb[0], 1) << "Woodbury did not converge (status=" << status_wb[0] << ")";

    EXPECT_NEAR(obj_wb, obj_cudss, std::max(1e-4, std::abs(obj_cudss) * 1e-4))
        << "Objective mismatch";

    for (int64_t i = 0; i < prob.n; ++i) {
        EXPECT_NEAR(x_wb[i], x_cudss[i], 1e-4) << "x[" << i << "] mismatch";
    }

    // Verify box constraints
    for (int64_t i = 0; i < prob.n; ++i) {
        EXPECT_GE(x_wb[i], -1e-6) << "Lower bound violated at x[" << i << "]";
        EXPECT_LE(x_wb[i], 0.5 + 1e-6) << "Upper bound violated at x[" << i << "]";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test partial bounds solve: only some variables bounded
TEST_F(WoodburyTest, PartialBoundsSolveMatchesCuDSS) {
    const int64_t n = 10;
    const int64_t k = 2;
    const int64_t batchSize = 1;

    // Only 5 of 10 vars have lower bounds
    int64_t n_nonneg = 5;
    int64_t m = k + n_nonneg;

    std::mt19937 rng(77);
    std::uniform_real_distribution<double> dist(0.1, 2.0);
    std::normal_distribution<double> ndist(0.0, 1.0);

    // Diagonal P
    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci(n);
    std::vector<double> P_vals(n);
    for (int64_t i = 0; i < n; ++i) {
        P_ro[i] = i; P_ci[i] = i; P_vals[i] = dist(rng);
    }
    P_ro[n] = n;

    // F (n x k)
    std::vector<double> F(n * k);
    for (auto& v : F) v = ndist(rng);

    // A = [F^T(k×n); -I partial(5×n)]
    // Bounded vars: 0, 2, 4, 6, 8
    int64_t bound_vars[] = {0, 2, 4, 6, 8};
    int64_t nnzA = k * n + n_nonneg;
    std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
    std::vector<double> A_vals(nnzA);

    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) {
            A_ci[idx] = j;
            A_vals[idx] = F[j * k + i];
            idx++;
        }
    }
    for (int64_t i = 0; i < n_nonneg; ++i) {
        A_ro[k + i] = idx;
        A_ci[idx] = bound_vars[i];
        A_vals[idx] = -1.0;
        idx++;
    }
    A_ro[m] = idx;

    std::vector<double> q_data(n), b_data(m, 0.0);
    for (auto& v : q_data) v = ndist(rng) * 0.5;
    b_data[0] = 1.0;

    Cones cones{};
    cones.numZeroCones = k;
    cones.numNonnegCones = n_nonneg;

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * n);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMemcpy(d_P, P_vals.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_vals.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // CuDSS
    Settings s_c;
    s_c.ipm.kktSolverType = KKTSolverType::CuDSS;
    s_c.maxIter = 100; s_c.verbose = false;
    s_c.ipm.equilibrationSettings.enable = true;
    CompiledSolver sc(n, m, batchSize, P_ro.data(), P_ci.data(), n,
                      A_ro.data(), A_ci.data(), nnzA, cones, s_c);
    sc.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_c(n);
    std::vector<int32_t> st_c(1);
    double obj_c;
    cudaMemcpy(x_c.data(), sc.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(st_c.data(), sc.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_c, sc.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(st_c[0], 1) << "CuDSS did not converge";

    // Woodbury
    Settings s_w;
    s_w.ipm.kktSolverType = KKTSolverType::Woodbury;
    s_w.maxIter = 100; s_w.verbose = false;
    s_w.ipm.equilibrationSettings.enable = true;
    CompiledSolver sw(n, m, batchSize, P_ro.data(), P_ci.data(), n,
                      A_ro.data(), A_ci.data(), nnzA, cones, s_w);
    sw.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_w(n);
    std::vector<int32_t> st_w(1);
    double obj_w;
    cudaMemcpy(x_w.data(), sw.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(st_w.data(), sw.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_w, sw.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(st_w[0], 1) << "Woodbury did not converge (status=" << st_w[0] << ")";

    EXPECT_NEAR(obj_w, obj_c, std::max(1e-4, std::abs(obj_c) * 1e-4));
    for (int64_t i = 0; i < n; ++i)
        EXPECT_NEAR(x_w[i], x_c[i], 1e-4) << "x[" << i << "] mismatch";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// Stress test: portfolio across multiple random seeds
TEST_F(WoodburyTest, PortfolioMultiSeedCorrectness) {
    const int64_t batchSize = 1;
    struct Config { int64_t n; int64_t k; };
    std::vector<Config> configs = {{10, 2}, {20, 3}, {50, 5}, {100, 10}};

    for (auto& cfg : configs) {
        for (unsigned seed = 1; seed <= 5; ++seed) {
            auto prob = buildPortfolio(cfg.n, cfg.k, seed);

            double *d_P, *d_A, *d_q, *d_b;
            cudaMalloc(&d_P, sizeof(double) * prob.nnzP);
            cudaMalloc(&d_A, sizeof(double) * prob.nnzA);
            cudaMalloc(&d_q, sizeof(double) * prob.n);
            cudaMalloc(&d_b, sizeof(double) * prob.m);
            cudaMemcpy(d_P, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
            cudaMemcpy(d_A, prob.A_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
            cudaMemcpy(d_q, prob.q_data.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, prob.b_data.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

            // CuDSS
            Settings s_c;
            s_c.ipm.kktSolverType = KKTSolverType::CuDSS;
            s_c.maxIter = 100; s_c.verbose = false;
            s_c.ipm.equilibrationSettings.enable = true;
            CompiledSolver sc(prob.n, prob.m, batchSize,
                prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
                prob.cones, s_c);
            sc.solveAll(d_P, d_A, d_q, d_b);

            std::vector<double> x_c(prob.n);
            std::vector<int32_t> st_c(1);
            double obj_c;
            cudaMemcpy(x_c.data(), sc.solution.x.data(), sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
            cudaMemcpy(st_c.data(), sc.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&obj_c, sc.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
            ASSERT_EQ(st_c[0], 1) << "CuDSS n=" << cfg.n << " k=" << cfg.k << " seed=" << seed;

            // Woodbury
            Settings s_w;
            s_w.ipm.kktSolverType = KKTSolverType::Woodbury;
            s_w.maxIter = 100; s_w.verbose = false;
            s_w.ipm.equilibrationSettings.enable = true;
            CompiledSolver sw(prob.n, prob.m, batchSize,
                prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
                prob.cones, s_w);
            sw.solveAll(d_P, d_A, d_q, d_b);

            std::vector<double> x_w(prob.n);
            std::vector<int32_t> st_w(1);
            double obj_w;
            cudaMemcpy(x_w.data(), sw.solution.x.data(), sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
            cudaMemcpy(st_w.data(), sw.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&obj_w, sw.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
            ASSERT_EQ(st_w[0], 1) << "Woodbury n=" << cfg.n << " k=" << cfg.k << " seed=" << seed
                                   << " status=" << st_w[0];

            EXPECT_NEAR(obj_w, obj_c, std::max(1e-4, std::abs(obj_c) * 1e-4))
                << "Obj mismatch n=" << cfg.n << " k=" << cfg.k << " seed=" << seed;

            for (int64_t i = 0; i < prob.n; ++i) {
                EXPECT_NEAR(x_w[i], x_c[i], 1e-4)
                    << "x[" << i << "] n=" << cfg.n << " k=" << cfg.k << " seed=" << seed;
            }

            cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
        }
    }
}

// Box QP multi-seed correctness
TEST_F(WoodburyTest, BoxQPMultiSeedCorrectness) {
    const int64_t batchSize = 1;
    struct Config { int64_t n; int64_t k; };
    std::vector<Config> configs = {{10, 2}, {20, 3}, {50, 5}};

    for (auto& cfg : configs) {
        for (unsigned seed = 1; seed <= 5; ++seed) {
            auto prob = buildBoxQP(cfg.n, cfg.k, seed);

            double *d_P, *d_A, *d_q, *d_b;
            cudaMalloc(&d_P, sizeof(double) * prob.nnzP);
            cudaMalloc(&d_A, sizeof(double) * prob.nnzA);
            cudaMalloc(&d_q, sizeof(double) * prob.n);
            cudaMalloc(&d_b, sizeof(double) * prob.m);
            cudaMemcpy(d_P, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
            cudaMemcpy(d_A, prob.A_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
            cudaMemcpy(d_q, prob.q_data.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, prob.b_data.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

            // CuDSS
            Settings s_c;
            s_c.ipm.kktSolverType = KKTSolverType::CuDSS;
            s_c.maxIter = 100; s_c.verbose = false;
            s_c.ipm.equilibrationSettings.enable = true;
            CompiledSolver sc(prob.n, prob.m, batchSize,
                prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
                prob.cones, s_c);
            sc.solveAll(d_P, d_A, d_q, d_b);

            std::vector<double> x_c(prob.n);
            std::vector<int32_t> st_c(1);
            double obj_c;
            cudaMemcpy(x_c.data(), sc.solution.x.data(), sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
            cudaMemcpy(st_c.data(), sc.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&obj_c, sc.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
            ASSERT_EQ(st_c[0], 1) << "CuDSS n=" << cfg.n << " k=" << cfg.k << " seed=" << seed;

            // Woodbury
            Settings s_w;
            s_w.ipm.kktSolverType = KKTSolverType::Woodbury;
            s_w.maxIter = 100; s_w.verbose = false;
            s_w.ipm.equilibrationSettings.enable = true;
            CompiledSolver sw(prob.n, prob.m, batchSize,
                prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
                prob.cones, s_w);
            sw.solveAll(d_P, d_A, d_q, d_b);

            std::vector<double> x_w(prob.n);
            std::vector<int32_t> st_w(1);
            double obj_w;
            cudaMemcpy(x_w.data(), sw.solution.x.data(), sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
            cudaMemcpy(st_w.data(), sw.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&obj_w, sw.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
            ASSERT_EQ(st_w[0], 1) << "Woodbury BoxQP n=" << cfg.n << " k=" << cfg.k
                                   << " seed=" << seed << " status=" << st_w[0];

            EXPECT_NEAR(obj_w, obj_c, std::max(1e-4, std::abs(obj_c) * 1e-4))
                << "Obj mismatch BoxQP n=" << cfg.n << " k=" << cfg.k << " seed=" << seed;

            for (int64_t i = 0; i < prob.n; ++i) {
                EXPECT_NEAR(x_w[i], x_c[i], 1e-4)
                    << "x[" << i << "] BoxQP n=" << cfg.n << " k=" << cfg.k << " seed=" << seed;
                EXPECT_GE(x_w[i], -1e-6) << "lb violated";
                EXPECT_LE(x_w[i], 0.5 + 1e-6) << "ub violated";
            }

            cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
        }
    }
}

// Batched box QP solve
TEST_F(WoodburyTest, BatchedBoxQPSolve) {
    const int64_t n = 20, k = 3, batchSize = 8;
    auto prob = buildBoxQP(n, k);

    std::vector<double> P_b(prob.nnzP * batchSize);
    std::vector<double> A_b(prob.nnzA * batchSize);
    std::vector<double> q_b(prob.n * batchSize);
    std::vector<double> b_b(prob.m * batchSize);

    std::mt19937 rng(999);
    std::normal_distribution<double> nd(0.0, 0.5);

    for (int64_t bi = 0; bi < batchSize; ++bi) {
        std::copy(prob.P_values.begin(), prob.P_values.end(), P_b.begin() + bi * prob.nnzP);
        std::copy(prob.A_values.begin(), prob.A_values.end(), A_b.begin() + bi * prob.nnzA);
        std::copy(prob.b_data.begin(), prob.b_data.end(), b_b.begin() + bi * prob.m);
        for (int64_t i = 0; i < prob.n; ++i)
            q_b[bi * prob.n + i] = nd(rng);
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * prob.n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * prob.m * batchSize);
    cudaMemcpy(d_P, P_b.data(), sizeof(double) * prob.nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_b.data(), sizeof(double) * prob.nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_b.data(), sizeof(double) * prob.n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_b.data(), sizeof(double) * prob.m * batchSize, cudaMemcpyHostToDevice);

    Settings s_w;
    s_w.ipm.kktSolverType = KKTSolverType::Woodbury;
    s_w.maxIter = 100; s_w.verbose = false;
    s_w.ipm.equilibrationSettings.enable = true;
    CompiledSolver sw(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, s_w);
    sw.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(batchSize);
    std::vector<double> x_sol(prob.n * batchSize);
    cudaMemcpy(status.data(), sw.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(x_sol.data(), sw.solution.x.data(), sizeof(double) * prob.n * batchSize, cudaMemcpyDeviceToHost);

    for (int64_t bi = 0; bi < batchSize; ++bi) {
        EXPECT_EQ(status[bi], 1) << "Batch " << bi << " status=" << status[bi];
        for (int64_t i = 0; i < prob.n; ++i) {
            double xi = x_sol[bi * prob.n + i];
            EXPECT_GE(xi, -1e-6) << "Batch " << bi << " lb violated at x[" << i << "]=" << xi;
            EXPECT_LE(xi, 0.5 + 1e-6) << "Batch " << bi << " ub violated at x[" << i << "]=" << xi;
        }
    }

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// Scaled bounds: A_s has non-unit coefficients
// min (1/2) x^T D x + q^T x  s.t. F^T x = b, 2x >= 0, 3x <= 1.5
TEST_F(WoodburyTest, ScaledBoundsMatchesCuDSS) {
    const int64_t n = 15, k = 3, batchSize = 1;
    int64_t n_nonneg = 2 * n;
    int64_t m = k + n_nonneg;

    std::mt19937 rng(55);
    std::uniform_real_distribution<double> ud(0.1, 2.0);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::vector<int64_t> P_ro(n + 1), P_ci(n);
    std::vector<double> P_vals(n);
    for (int64_t i = 0; i < n; ++i) { P_ro[i] = i; P_ci[i] = i; P_vals[i] = ud(rng); }
    P_ro[n] = n;

    std::vector<double> F(n * k);
    for (auto& v : F) v = nd(rng);

    // A = [F^T; -2I; 3I]  (scaled bounds: -2x + s = 0 → x >= 0,  3x + s = 4.5 → x <= 1.5)
    int64_t nnzA = k * n + 2 * n;
    std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
    std::vector<double> A_vals(nnzA);
    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) {
            A_ci[idx] = j; A_vals[idx] = F[j * k + i]; idx++;
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        A_ro[k + i] = idx;
        A_ci[idx] = i; A_vals[idx] = -2.0; idx++;  // -2x + s = 0, s >= 0 → x >= 0
    }
    for (int64_t i = 0; i < n; ++i) {
        A_ro[k + n + i] = idx;
        A_ci[idx] = i; A_vals[idx] = 3.0; idx++;  // 3x + s = 4.5, s >= 0 → x <= 1.5
    }
    A_ro[m] = idx;

    // Loose equality: first row of F^T, easy to satisfy with wide bounds
    std::vector<double> q_data(n), b_data(m, 0.0);
    for (auto& v : q_data) v = nd(rng) * 0.5;
    // Set b_eq to something feasible: sum of F row * 0.5 (midpoint of bounds)
    for (int64_t i = 0; i < k; ++i) {
        double sum = 0;
        for (int64_t j = 0; j < n; ++j) sum += F[j * k + i] * 0.5;
        b_data[i] = sum;
    }
    // b_nonneg: lower bounds → 0, upper bounds → 4.5
    for (int64_t i = 0; i < n; ++i) b_data[k + n + i] = 4.5;

    Cones cones{};
    cones.numZeroCones = k;
    cones.numNonnegCones = n_nonneg;

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * n);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMemcpy(d_P, P_vals.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_vals.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // CuDSS
    Settings s_c;
    s_c.ipm.kktSolverType = KKTSolverType::CuDSS;
    s_c.maxIter = 100; s_c.verbose = false;
    s_c.ipm.equilibrationSettings.enable = true;
    CompiledSolver sc(n, m, batchSize, P_ro.data(), P_ci.data(), n,
                      A_ro.data(), A_ci.data(), nnzA, cones, s_c);
    sc.solveAll(d_P, d_A, d_q, d_b);
    std::vector<double> x_c(n);
    std::vector<int32_t> st_c(1);
    double obj_c;
    cudaMemcpy(x_c.data(), sc.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(st_c.data(), sc.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_c, sc.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(st_c[0], 1) << "CuDSS did not converge";

    // Woodbury
    Settings s_w;
    s_w.ipm.kktSolverType = KKTSolverType::Woodbury;
    s_w.maxIter = 100; s_w.verbose = false;
    s_w.ipm.equilibrationSettings.enable = true;
    CompiledSolver sw(n, m, batchSize, P_ro.data(), P_ci.data(), n,
                      A_ro.data(), A_ci.data(), nnzA, cones, s_w);
    sw.solveAll(d_P, d_A, d_q, d_b);
    std::vector<double> x_w(n);
    std::vector<int32_t> st_w(1);
    double obj_w;
    cudaMemcpy(x_w.data(), sw.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(st_w.data(), sw.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_w, sw.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(st_w[0], 1) << "Woodbury did not converge (status=" << st_w[0] << ")";

    EXPECT_NEAR(obj_w, obj_c, std::max(1e-4, std::abs(obj_c) * 1e-4));
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(x_w[i], x_c[i], 1e-4) << "x[" << i << "]";
        EXPECT_GE(x_w[i], -1e-6) << "lb violated";
        EXPECT_LE(x_w[i], 1.5 + 1e-6) << "ub violated";
    }

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// Empty nonneg rows: nonneg rows with 0 nonzeros (pure slack)
TEST_F(WoodburyTest, EmptyNonnegRowsSolve) {
    // Problem: min (1/2) x^T D x + q^T x  s.t. F^T x = b, plus 3 "empty" nonneg constraints
    // The empty rows correspond to s >= 0 with no x coupling (A row is all zeros)
    // This is unusual but should not crash — the solver should handle gracefully
    const int64_t n = 10, k = 2, batchSize = 1;
    int64_t n_nonneg = n + 3;  // n bounded + 3 empty
    int64_t m = k + n_nonneg;

    std::mt19937 rng(88);
    std::uniform_real_distribution<double> ud(0.1, 2.0);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::vector<int64_t> P_ro(n + 1), P_ci(n);
    std::vector<double> P_vals(n);
    for (int64_t i = 0; i < n; ++i) { P_ro[i] = i; P_ci[i] = i; P_vals[i] = ud(rng); }
    P_ro[n] = n;

    std::vector<double> F(n * k);
    for (auto& v : F) v = nd(rng);

    // A = [F^T(k×n); -I(n×n); 0(3×n)]  — last 3 rows are empty
    int64_t nnzA = k * n + n;
    std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
    std::vector<double> A_vals(nnzA);
    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) {
            A_ci[idx] = j; A_vals[idx] = F[j * k + i]; idx++;
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        A_ro[k + i] = idx;
        A_ci[idx] = i; A_vals[idx] = -1.0; idx++;
    }
    // 3 empty rows
    for (int64_t i = 0; i < 3; ++i) {
        A_ro[k + n + i] = idx;
    }
    A_ro[m] = idx;

    std::vector<double> q_data(n), b_data(m, 0.0);
    for (auto& v : q_data) v = -ud(rng) * 0.1;
    b_data[0] = 1.0;

    Cones cones{};
    cones.numZeroCones = k;
    cones.numNonnegCones = n_nonneg;

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * n);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMemcpy(d_P, P_vals.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_vals.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // CuDSS
    Settings s_c;
    s_c.ipm.kktSolverType = KKTSolverType::CuDSS;
    s_c.maxIter = 100; s_c.verbose = false;
    s_c.ipm.equilibrationSettings.enable = true;
    CompiledSolver sc(n, m, batchSize, P_ro.data(), P_ci.data(), n,
                      A_ro.data(), A_ci.data(), nnzA, cones, s_c);
    sc.solveAll(d_P, d_A, d_q, d_b);
    std::vector<double> x_c(n);
    std::vector<int32_t> st_c(1);
    double obj_c;
    cudaMemcpy(x_c.data(), sc.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(st_c.data(), sc.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_c, sc.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(st_c[0], 1) << "CuDSS did not converge";

    // Woodbury
    Settings s_w;
    s_w.ipm.kktSolverType = KKTSolverType::Woodbury;
    s_w.maxIter = 100; s_w.verbose = false;
    s_w.ipm.equilibrationSettings.enable = true;
    CompiledSolver sw(n, m, batchSize, P_ro.data(), P_ci.data(), n,
                      A_ro.data(), A_ci.data(), nnzA, cones, s_w);
    sw.solveAll(d_P, d_A, d_q, d_b);
    std::vector<double> x_w(n);
    std::vector<int32_t> st_w(1);
    double obj_w;
    cudaMemcpy(x_w.data(), sw.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(st_w.data(), sw.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_w, sw.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(st_w[0], 1) << "Woodbury did not converge (status=" << st_w[0] << ")";

    EXPECT_NEAR(obj_w, obj_c, std::max(1e-4, std::abs(obj_c) * 1e-4));
    for (int64_t i = 0; i < n; ++i)
        EXPECT_NEAR(x_w[i], x_c[i], 1e-4) << "x[" << i << "]";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// ===========================================================================
// Phase 2: Dense nonneg rows (inequality constraints)
// ===========================================================================

// Helper: build a problem with dense inequality constraints
// min (1/2) x^T diag(D) x + q^T x
// s.t. F x = b_eq       (k zero cones)
//      G x >= h          (k_d dense nonneg cones: -G x + s = -h)
struct DenseInequalityProblem {
    int64_t n, k, k_d, m, nnzP, nnzA;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_values, A_values;
    std::vector<double> q, b;
    Cones cones;
};

DenseInequalityProblem buildDenseInequality(int64_t n, int64_t k, int64_t k_d, unsigned seed = 42) {
    DenseInequalityProblem p;
    p.n = n; p.k = k; p.k_d = k_d;
    int64_t n_nonneg = k_d;
    p.m = k + n_nonneg;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ud(0.1, 2.0);
    std::normal_distribution<double> nd(0.0, 1.0);

    // Diagonal P
    p.nnzP = n;
    p.P_ro.resize(n + 1);
    p.P_ci.resize(n);
    p.P_values.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        p.P_ro[i] = i; p.P_ci[i] = i;
        p.P_values[i] = ud(rng);
    }
    p.P_ro[n] = n;

    // F (k x n) dense factor loadings
    std::vector<double> F(n * k);
    for (auto& v : F) v = nd(rng);

    // G (k_d x n) dense inequality matrix
    std::vector<double> G(n * k_d);
    for (auto& v : G) v = nd(rng) * 0.5;

    // Build A = [F; -G]
    p.nnzA = k * n + k_d * n;
    p.A_ro.resize(p.m + 1);
    p.A_ci.resize(p.nnzA);
    p.A_values.resize(p.nnzA);

    int64_t idx = 0;
    // Zero-cone rows (F)
    for (int64_t i = 0; i < k; ++i) {
        p.A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) {
            p.A_ci[idx] = j;
            p.A_values[idx] = F[j * k + i];
            idx++;
        }
    }
    // Nonneg rows (-G, so constraint is Gx >= h ↔ -Gx + s = -h, s >= 0)
    for (int64_t d = 0; d < k_d; ++d) {
        p.A_ro[k + d] = idx;
        for (int64_t j = 0; j < n; ++j) {
            p.A_ci[idx] = j;
            p.A_values[idx] = -G[j * k_d + d];
            idx++;
        }
    }
    p.A_ro[p.m] = idx;

    // q and b
    p.q.resize(n);
    for (auto& v : p.q) v = nd(rng);

    // b_eq = F * x_mid where x_mid is random feasible point
    std::vector<double> x_mid(n);
    for (auto& v : x_mid) v = ud(rng) * 0.5;

    p.b.resize(p.m);
    // b_eq = F * x_mid
    for (int64_t i = 0; i < k; ++i) {
        double sum = 0.0;
        for (int64_t j = 0; j < n; ++j) sum += F[j * k + i] * x_mid[j];
        p.b[i] = sum;
    }
    // b_nonneg = -G * x_mid + slack (so Gx_mid >= h is feasible)
    for (int64_t d = 0; d < k_d; ++d) {
        double sum = 0.0;
        for (int64_t j = 0; j < n; ++j) sum += -G[j * k_d + d] * x_mid[j];
        p.b[k + d] = sum - 0.5;  // slack to ensure feasibility
    }

    p.cones = {};
    p.cones.numZeroCones = k;
    p.cones.numNonnegCones = n_nonneg;
    return p;
}

// Helper: build a mixed problem with dense + sparse nonneg rows
// min (1/2) x^T diag(D) x + q^T x
// s.t. F x = b_eq       (k zero cones)
//      G x >= h          (k_d dense nonneg cones)
//      x >= 0            (n sparse nonneg cones)
struct MixedProblem {
    int64_t n, k, k_d, m, nnzP, nnzA;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_values, A_values;
    std::vector<double> q, b;
    Cones cones;
};

MixedProblem buildMixedProblem(int64_t n, int64_t k, int64_t k_d, unsigned seed = 42) {
    MixedProblem p;
    p.n = n; p.k = k; p.k_d = k_d;
    int64_t n_nonneg = k_d + n;  // k_d dense + n sparse
    p.m = k + n_nonneg;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ud(0.1, 2.0);
    std::normal_distribution<double> nd(0.0, 1.0);

    // Diagonal P
    p.nnzP = n;
    p.P_ro.resize(n + 1);
    p.P_ci.resize(n);
    p.P_values.resize(n);
    for (int64_t i = 0; i < n; ++i) {
        p.P_ro[i] = i; p.P_ci[i] = i;
        p.P_values[i] = ud(rng);
    }
    p.P_ro[n] = n;

    std::vector<double> F(n * k);
    for (auto& v : F) v = nd(rng);

    std::vector<double> G(n * k_d);
    for (auto& v : G) v = nd(rng) * 0.5;

    // A = [F; -G; -I]
    p.nnzA = k * n + k_d * n + n;
    p.A_ro.resize(p.m + 1);
    p.A_ci.resize(p.nnzA);
    p.A_values.resize(p.nnzA);

    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        p.A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) {
            p.A_ci[idx] = j;
            p.A_values[idx] = F[j * k + i];
            idx++;
        }
    }
    // Dense nonneg rows (-G)
    for (int64_t d = 0; d < k_d; ++d) {
        p.A_ro[k + d] = idx;
        for (int64_t j = 0; j < n; ++j) {
            p.A_ci[idx] = j;
            p.A_values[idx] = -G[j * k_d + d];
            idx++;
        }
    }
    // Sparse nonneg rows (-I for x >= 0)
    for (int64_t i = 0; i < n; ++i) {
        p.A_ro[k + k_d + i] = idx;
        p.A_ci[idx] = i;
        p.A_values[idx] = -1.0;
        idx++;
    }
    p.A_ro[p.m] = idx;

    // q and b — feasible at x_mid > 0
    p.q.resize(n);
    for (auto& v : p.q) v = nd(rng);

    std::vector<double> x_mid(n);
    for (auto& v : x_mid) v = ud(rng);

    p.b.resize(p.m);
    for (int64_t i = 0; i < k; ++i) {
        double sum = 0.0;
        for (int64_t j = 0; j < n; ++j) sum += F[j * k + i] * x_mid[j];
        p.b[i] = sum;
    }
    for (int64_t d = 0; d < k_d; ++d) {
        double sum = 0.0;
        for (int64_t j = 0; j < n; ++j) sum += -G[j * k_d + d] * x_mid[j];
        p.b[k + d] = sum - 0.5;
    }
    for (int64_t i = 0; i < n; ++i) {
        p.b[k + k_d + i] = -x_mid[i] + 0.0;  // -I * x_mid + s = 0 → s = x_mid > 0
    }

    p.cones = {};
    p.cones.numZeroCones = k;
    p.cones.numNonnegCones = n_nonneg;
    return p;
}

// Dense inequality: A = [F; -G], all nonneg rows are dense
TEST_F(WoodburyTest, DenseInequalitySolveMatchesCuDSS) {
    int64_t n = 20, k = 5, k_d = 3;
    auto p = buildDenseInequality(n, k, k_d, 42);
    int64_t batchSize = 1;

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * p.nnzP);
    cudaMalloc(&d_A, sizeof(double) * p.nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * p.m);
    cudaMemcpy(d_P, p.P_values.data(), sizeof(double) * p.nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, p.A_values.data(), sizeof(double) * p.nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, p.q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, p.b.data(), sizeof(double) * p.m, cudaMemcpyHostToDevice);

    // CuDSS
    Settings s_c;
    s_c.ipm.kktSolverType = KKTSolverType::CuDSS;
    s_c.maxIter = 100; s_c.verbose = false;
    CompiledSolver sc(n, p.m, batchSize, p.P_ro.data(), p.P_ci.data(), p.nnzP,
                      p.A_ro.data(), p.A_ci.data(), p.nnzA, p.cones, s_c);
    sc.solveAll(d_P, d_A, d_q, d_b);
    std::vector<double> x_c(n);
    std::vector<int32_t> st_c(1);
    double obj_c;
    cudaMemcpy(x_c.data(), sc.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(st_c.data(), sc.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_c, sc.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(st_c[0], 1) << "CuDSS did not converge (status=" << st_c[0] << ")";

    // Woodbury
    Settings s_w;
    s_w.ipm.kktSolverType = KKTSolverType::Woodbury;
    s_w.maxIter = 100; s_w.verbose = false;
    CompiledSolver sw(n, p.m, batchSize, p.P_ro.data(), p.P_ci.data(), p.nnzP,
                      p.A_ro.data(), p.A_ci.data(), p.nnzA, p.cones, s_w);
    sw.solveAll(d_P, d_A, d_q, d_b);
    std::vector<double> x_w(n);
    std::vector<int32_t> st_w(1);
    double obj_w;
    cudaMemcpy(x_w.data(), sw.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(st_w.data(), sw.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_w, sw.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(st_w[0], 1) << "Woodbury did not converge (status=" << st_w[0] << ")";

    EXPECT_NEAR(obj_w, obj_c, std::max(1e-4, std::abs(obj_c) * 1e-4));
    for (int64_t i = 0; i < n; ++i)
        EXPECT_NEAR(x_w[i], x_c[i], 1e-4) << "x[" << i << "]";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// Mixed: dense nonneg (G x >= h) + sparse nonneg (x >= 0)
TEST_F(WoodburyTest, MixedDenseSparseMatchesCuDSS) {
    int64_t n = 20, k = 5, k_d = 3;
    auto p = buildMixedProblem(n, k, k_d, 42);
    int64_t batchSize = 1;

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * p.nnzP);
    cudaMalloc(&d_A, sizeof(double) * p.nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * p.m);
    cudaMemcpy(d_P, p.P_values.data(), sizeof(double) * p.nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, p.A_values.data(), sizeof(double) * p.nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, p.q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, p.b.data(), sizeof(double) * p.m, cudaMemcpyHostToDevice);

    // CuDSS
    Settings s_c;
    s_c.ipm.kktSolverType = KKTSolverType::CuDSS;
    s_c.maxIter = 100; s_c.verbose = false;
    CompiledSolver sc(n, p.m, batchSize, p.P_ro.data(), p.P_ci.data(), p.nnzP,
                      p.A_ro.data(), p.A_ci.data(), p.nnzA, p.cones, s_c);
    sc.solveAll(d_P, d_A, d_q, d_b);
    std::vector<double> x_c(n);
    std::vector<int32_t> st_c(1);
    double obj_c;
    cudaMemcpy(x_c.data(), sc.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(st_c.data(), sc.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_c, sc.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(st_c[0], 1) << "CuDSS did not converge (status=" << st_c[0] << ")";

    // Woodbury
    Settings s_w;
    s_w.ipm.kktSolverType = KKTSolverType::Woodbury;
    s_w.maxIter = 100; s_w.verbose = false;
    CompiledSolver sw(n, p.m, batchSize, p.P_ro.data(), p.P_ci.data(), p.nnzP,
                      p.A_ro.data(), p.A_ci.data(), p.nnzA, p.cones, s_w);
    sw.solveAll(d_P, d_A, d_q, d_b);
    std::vector<double> x_w(n);
    std::vector<int32_t> st_w(1);
    double obj_w;
    cudaMemcpy(x_w.data(), sw.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(st_w.data(), sw.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&obj_w, sw.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    ASSERT_EQ(st_w[0], 1) << "Woodbury did not converge (status=" << st_w[0] << ")";

    EXPECT_NEAR(obj_w, obj_c, std::max(1e-4, std::abs(obj_c) * 1e-4));
    for (int64_t i = 0; i < n; ++i)
        EXPECT_NEAR(x_w[i], x_c[i], 1e-4) << "x[" << i << "]";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// Multi-seed stress test for dense inequality problems
TEST_F(WoodburyTest, DenseInequalityMultiSeedCorrectness) {
    struct Config { int64_t n, k, k_d; };
    Config configs[] = {
        {10, 3, 2},
        {20, 5, 3},
        {50, 10, 5},
    };
    unsigned seeds[] = {42, 123, 777, 2024, 9999};

    for (auto& cfg : configs) {
        for (auto seed : seeds) {
            auto p = buildDenseInequality(cfg.n, cfg.k, cfg.k_d, seed);
            int64_t batchSize = 1;

            double *d_P, *d_A, *d_q, *d_b;
            cudaMalloc(&d_P, sizeof(double) * p.nnzP);
            cudaMalloc(&d_A, sizeof(double) * p.nnzA);
            cudaMalloc(&d_q, sizeof(double) * cfg.n);
            cudaMalloc(&d_b, sizeof(double) * p.m);
            cudaMemcpy(d_P, p.P_values.data(), sizeof(double) * p.nnzP, cudaMemcpyHostToDevice);
            cudaMemcpy(d_A, p.A_values.data(), sizeof(double) * p.nnzA, cudaMemcpyHostToDevice);
            cudaMemcpy(d_q, p.q.data(), sizeof(double) * cfg.n, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, p.b.data(), sizeof(double) * p.m, cudaMemcpyHostToDevice);

            Settings s_c;
            s_c.ipm.kktSolverType = KKTSolverType::CuDSS;
            s_c.maxIter = 100; s_c.verbose = false;
            CompiledSolver sc(cfg.n, p.m, batchSize, p.P_ro.data(), p.P_ci.data(), p.nnzP,
                              p.A_ro.data(), p.A_ci.data(), p.nnzA, p.cones, s_c);
            sc.solveAll(d_P, d_A, d_q, d_b);
            std::vector<int32_t> st_c(1);
            double obj_c;
            cudaMemcpy(st_c.data(), sc.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&obj_c, sc.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
            if (st_c[0] != 1) { cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b); continue; }

            Settings s_w;
            s_w.ipm.kktSolverType = KKTSolverType::Woodbury;
            s_w.maxIter = 100; s_w.verbose = false;
            CompiledSolver sw(cfg.n, p.m, batchSize, p.P_ro.data(), p.P_ci.data(), p.nnzP,
                              p.A_ro.data(), p.A_ci.data(), p.nnzA, p.cones, s_w);
            sw.solveAll(d_P, d_A, d_q, d_b);
            std::vector<int32_t> st_w(1);
            double obj_w;
            cudaMemcpy(st_w.data(), sw.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&obj_w, sw.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);

            EXPECT_EQ(st_w[0], 1) << "n=" << cfg.n << " k=" << cfg.k
                                  << " k_d=" << cfg.k_d << " seed=" << seed
                                  << " Woodbury status=" << st_w[0];

            if (st_w[0] == 1) {
                EXPECT_NEAR(obj_w, obj_c, std::max(1e-4, std::abs(obj_c) * 1e-3))
                    << "n=" << cfg.n << " k=" << cfg.k << " k_d=" << cfg.k_d << " seed=" << seed;
            }

            cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
        }
    }
}

// Batched dense inequality
TEST_F(WoodburyTest, BatchedDenseInequalitySolve) {
    int64_t n = 15, k = 4, k_d = 3;
    auto p = buildDenseInequality(n, k, k_d, 42);
    int64_t batchSize = 8;

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * p.nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * p.nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * p.m * batchSize);
    for (int64_t b = 0; b < batchSize; ++b) {
        cudaMemcpy(d_P + b * p.nnzP, p.P_values.data(), sizeof(double) * p.nnzP, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A + b * p.nnzA, p.A_values.data(), sizeof(double) * p.nnzA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q + b * n, p.q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b + b * p.m, p.b.data(), sizeof(double) * p.m, cudaMemcpyHostToDevice);
    }

    // Verify CuDSS can solve all batches first
    Settings s_c;
    s_c.ipm.kktSolverType = KKTSolverType::CuDSS;
    s_c.maxIter = 100; s_c.verbose = false;
    CompiledSolver sc(n, p.m, batchSize, p.P_ro.data(), p.P_ci.data(), p.nnzP,
                      p.A_ro.data(), p.A_ci.data(), p.nnzA, p.cones, s_c);
    sc.solveAll(d_P, d_A, d_q, d_b);
    std::vector<int32_t> st_c(batchSize);
    cudaMemcpy(st_c.data(), sc.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    for (int64_t b = 0; b < batchSize; ++b) {
        ASSERT_EQ(st_c[b], 1) << "CuDSS batch " << b << " did not converge (status=" << st_c[b] << ")";
    }

    Settings s_w;
    s_w.ipm.kktSolverType = KKTSolverType::Woodbury;
    s_w.maxIter = 100; s_w.verbose = false;
    CompiledSolver sw(n, p.m, batchSize, p.P_ro.data(), p.P_ci.data(), p.nnzP,
                      p.A_ro.data(), p.A_ci.data(), p.nnzA, p.cones, s_w);
    sw.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> st(batchSize);
    cudaMemcpy(st.data(), sw.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    for (int64_t b = 0; b < batchSize; ++b) {
        EXPECT_EQ(st[b], 1) << "Batch " << b << " did not converge (status=" << st[b] << ")";
    }

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// Mixed dense + sparse: A = [F; -G; -I]
TEST_F(WoodburyTest, MixedDenseSparseMultiSeedCorrectness) {
    struct Config { int64_t n, k, k_d; };
    Config configs[] = {
        {10, 3, 2},
        {20, 5, 3},
    };
    unsigned seeds[] = {42, 123, 777};

    for (auto& cfg : configs) {
        for (auto seed : seeds) {
            auto p = buildMixedProblem(cfg.n, cfg.k, cfg.k_d, seed);
            int64_t batchSize = 1;

            double *d_P, *d_A, *d_q, *d_b;
            cudaMalloc(&d_P, sizeof(double) * p.nnzP);
            cudaMalloc(&d_A, sizeof(double) * p.nnzA);
            cudaMalloc(&d_q, sizeof(double) * cfg.n);
            cudaMalloc(&d_b, sizeof(double) * p.m);
            cudaMemcpy(d_P, p.P_values.data(), sizeof(double) * p.nnzP, cudaMemcpyHostToDevice);
            cudaMemcpy(d_A, p.A_values.data(), sizeof(double) * p.nnzA, cudaMemcpyHostToDevice);
            cudaMemcpy(d_q, p.q.data(), sizeof(double) * cfg.n, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, p.b.data(), sizeof(double) * p.m, cudaMemcpyHostToDevice);

            Settings s_c;
            s_c.ipm.kktSolverType = KKTSolverType::CuDSS;
            s_c.maxIter = 100; s_c.verbose = false;
            CompiledSolver sc(cfg.n, p.m, batchSize, p.P_ro.data(), p.P_ci.data(), p.nnzP,
                              p.A_ro.data(), p.A_ci.data(), p.nnzA, p.cones, s_c);
            sc.solveAll(d_P, d_A, d_q, d_b);
            std::vector<int32_t> st_c(1);
            double obj_c;
            cudaMemcpy(st_c.data(), sc.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&obj_c, sc.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
            if (st_c[0] != 1) { cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b); continue; }

            Settings s_w;
            s_w.ipm.kktSolverType = KKTSolverType::Woodbury;
            s_w.maxIter = 100; s_w.verbose = false;
            CompiledSolver sw(cfg.n, p.m, batchSize, p.P_ro.data(), p.P_ci.data(), p.nnzP,
                              p.A_ro.data(), p.A_ci.data(), p.nnzA, p.cones, s_w);
            sw.solveAll(d_P, d_A, d_q, d_b);
            std::vector<int32_t> st_w(1);
            double obj_w;
            cudaMemcpy(st_w.data(), sw.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(&obj_w, sw.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);

            EXPECT_EQ(st_w[0], 1) << "n=" << cfg.n << " k=" << cfg.k
                                  << " k_d=" << cfg.k_d << " seed=" << seed
                                  << " Woodbury status=" << st_w[0];

            if (st_w[0] == 1) {
                EXPECT_NEAR(obj_w, obj_c, std::max(1e-4, std::abs(obj_c) * 1e-3))
                    << "n=" << cfg.n << " k=" << cfg.k << " k_d=" << cfg.k_d << " seed=" << seed;
            }

            cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
        }
    }
}

// MPC/banded tests removed — block-tridiagonal problems are handled by
// the Riccati solver (tried before Woodbury in KKTSolver dispatch).
// The Woodbury solver only handles dense Schur complement problems.

// ============================================================================
// Backward pass (gradient) tests for DiffWoodbury
// ============================================================================

#include "moreau/diff/diff.hpp"

// Compare gradients from Woodbury backward vs cuDSS backward on the same problem.
// Uses finite differences as ground truth.
TEST_F(WoodburyTest, BackwardGradientsMatchFiniteDiff) {
    const int64_t n = 10;
    const int64_t k = 2;
    const int64_t batchSize = 1;
    const double eps_fd = 1e-5;

    auto prob = buildPortfolio(n, k, 123);

    // Helper: solve and get objective value
    auto solve_obj = [&](const std::vector<double>& q_vec,
                         const std::vector<double>& b_vec) -> double {
        Settings s;
        s.ipm.kktSolverType = KKTSolverType::Woodbury;
        s.verbose = false;

        CompiledSolver solver(
            prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, s);

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * prob.nnzP);
        cudaMalloc(&d_A, sizeof(double) * prob.nnzA);
        cudaMalloc(&d_q, sizeof(double) * n);
        cudaMalloc(&d_b, sizeof(double) * prob.m);
        cudaMemcpy(d_P, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, prob.A_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q_vec.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_vec.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        double obj;
        cudaMemcpy(&obj, solver.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
        return obj;
    };

    // Solve with Woodbury + grad enabled
    Settings s_wb;
    s_wb.ipm.kktSolverType = KKTSolverType::Woodbury;
    s_wb.verbose = false;
    s_wb.enableGrad = true;

    CompiledSolver solver_wb(
        prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, s_wb);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP);
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * prob.m);
    cudaMemcpy(d_P, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q_data.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b_data.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);
    solver_wb.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    // Check solver status
    int32_t status;
    cudaMemcpy(&status, solver_wb.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(status, static_cast<int32_t>(SolverStatus::Solved)) << "Woodbury solve failed";
    ASSERT_TRUE(solver_wb.kkt->isWoodbury()) << "Expected Woodbury solver to be active";

    // Backward pass: dx_bar = 1 (sum of x), dz_bar = ds_bar = 0
    // This computes d(obj + sum(x)) / d(q, b, P, A) where obj depends on q,b,P,A
    BatchedVector dx_bar(n, batchSize);
    BatchedVector dz_bar(prob.m, batchSize);
    BatchedVector ds_bar(prob.m, batchSize);

    std::vector<double> dx_ones(n, 1.0);
    cudaMemcpy(dx_bar.data(), dx_ones.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    dz_bar.setToConstant(0.0);
    ds_bar.setToConstant(0.0);

    backward(*solver_wb.diff_state_, dx_bar, dz_bar, ds_bar, solver_wb, 0);
    cudaDeviceSynchronize();

    // Read gradients
    std::vector<double> dq_wb(n), db_wb(prob.m);
    cudaMemcpy(dq_wb.data(), solver_wb.diff_state_->dq.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(db_wb.data(), solver_wb.diff_state_->db.data(), sizeof(double) * prob.m, cudaMemcpyDeviceToHost);

    // Compute x from solution
    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver_wb.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    double base_loss = 0.0;
    for (int i = 0; i < n; ++i) base_loss += x_sol[i];

    // Finite difference check on dq
    double max_dq_err = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        auto q_plus = prob.q_data;
        auto q_minus = prob.q_data;
        q_plus[i] += eps_fd;
        q_minus[i] -= eps_fd;

        // Solve with perturbed q, get x, compute sum(x)
        auto get_loss = [&](const std::vector<double>& q_vec) -> double {
            Settings s;
            s.ipm.kktSolverType = KKTSolverType::Woodbury;
            s.verbose = false;

            CompiledSolver sol(prob.n, prob.m, batchSize,
                prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
                prob.cones, s);

            double *dp, *da, *dq2, *db2;
            cudaMalloc(&dp, sizeof(double) * prob.nnzP);
            cudaMalloc(&da, sizeof(double) * prob.nnzA);
            cudaMalloc(&dq2, sizeof(double) * n);
            cudaMalloc(&db2, sizeof(double) * prob.m);
            cudaMemcpy(dp, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
            cudaMemcpy(da, prob.A_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
            cudaMemcpy(dq2, q_vec.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(db2, prob.b_data.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);
            sol.solveAll(dp, da, dq2, db2);
            cudaDeviceSynchronize();
            std::vector<double> x(n);
            cudaMemcpy(x.data(), sol.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
            cudaFree(dp); cudaFree(da); cudaFree(dq2); cudaFree(db2);
            double loss = 0.0;
            for (int j = 0; j < n; ++j) loss += x[j];
            return loss;
        };

        double fd = (get_loss(q_plus) - get_loss(q_minus)) / (2.0 * eps_fd);
        double err = std::abs(dq_wb[i] - fd);
        max_dq_err = std::max(max_dq_err, err);
    }

    std::cout << "Max dq error vs finite diff: " << max_dq_err << std::endl;
    EXPECT_LT(max_dq_err, 1e-3) << "dq gradient doesn't match finite differences";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// Batched backward should be consistent across identical batch elements.
//
// Exact cross-backend gradient agreement is already covered in test_diff.cpp.
// Here we validate the batched Woodbury path directly without depending on
// forward-solve details that can legitimately differ across KKT backends.
TEST_F(WoodburyTest, BackwardBatchedConsistent) {
    const int64_t n = 20;
    const int64_t k = 3;
    const int64_t batchSize = 2;

    auto prob = buildPortfolio(n, k, 777);

    // Build batched data (same problem replicated)
    std::vector<double> P_bat, A_bat, q_bat, b_bat;
    for (int64_t b = 0; b < batchSize; ++b) {
        P_bat.insert(P_bat.end(), prob.P_values.begin(), prob.P_values.end());
        A_bat.insert(A_bat.end(), prob.A_values.begin(), prob.A_values.end());
        q_bat.insert(q_bat.end(), prob.q_data.begin(), prob.q_data.end());
        b_bat.insert(b_bat.end(), prob.b_data.begin(), prob.b_data.end());
    }

    auto run_backward = [&]() {
        Settings s;
        s.ipm.kktSolverType = KKTSolverType::Woodbury;
        s.verbose = false;
        s.enableGrad = true;

        CompiledSolver solver(
            prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, s);

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * prob.nnzP * batchSize);
        cudaMalloc(&d_A, sizeof(double) * prob.nnzA * batchSize);
        cudaMalloc(&d_q, sizeof(double) * n * batchSize);
        cudaMalloc(&d_b, sizeof(double) * prob.m * batchSize);
        cudaMemcpy(d_P, P_bat.data(), sizeof(double) * P_bat.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_bat.data(), sizeof(double) * A_bat.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q_bat.data(), sizeof(double) * q_bat.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_bat.data(), sizeof(double) * b_bat.size(), cudaMemcpyHostToDevice);
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        // Backward with dx_bar = 1
        BatchedVector dx_bar(n, batchSize);
        BatchedVector dz_bar(prob.m, batchSize);
        BatchedVector ds_bar(prob.m, batchSize);
        std::vector<double> dx_ones(n * batchSize, 1.0);
        cudaMemcpy(dx_bar.data(), dx_ones.data(), sizeof(double) * dx_ones.size(), cudaMemcpyHostToDevice);
        dz_bar.setToConstant(0.0);
        ds_bar.setToConstant(0.0);

        backward(*solver.diff_state_, dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        std::vector<double> dq(n * batchSize), db(prob.m * batchSize);
        std::vector<double> dP(prob.nnzP * batchSize), dA(prob.nnzA * batchSize);
        cudaMemcpy(dq.data(), solver.diff_state_->dq.data(), sizeof(double) * dq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(db.data(), solver.diff_state_->db.data(), sizeof(double) * db.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(dP.data(), solver.diff_state_->dP_values.data(), sizeof(double) * dP.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(dA.data(), solver.diff_state_->dA_values.data(), sizeof(double) * dA.size(), cudaMemcpyDeviceToHost);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
        return std::make_tuple(dq, db, dP, dA);
    };

    auto [dq_wb, db_wb, dP_wb, dA_wb] = run_backward();

    auto max_batch_diff = [&](const std::vector<double>& values, int64_t stride) {
        double mx = 0.0;
        for (int64_t i = 0; i < stride; ++i)
            mx = std::max(mx, std::abs(values[i] - values[stride + i]));
        return mx;
    };

    double dq_diff = max_batch_diff(dq_wb, n);
    double db_diff = max_batch_diff(db_wb, prob.m);
    double dP_diff = max_batch_diff(dP_wb, prob.nnzP);
    double dA_diff = max_batch_diff(dA_wb, prob.nnzA);

    std::cout << "Woodbury batched backward consistency:" << std::endl;
    std::cout << "  max |dq[0] - dq[1]| = " << dq_diff << std::endl;
    std::cout << "  max |db[0] - db[1]| = " << db_diff << std::endl;
    std::cout << "  max |dP[0] - dP[1]| = " << dP_diff << std::endl;
    std::cout << "  max |dA[0] - dA[1]| = " << dA_diff << std::endl;

    EXPECT_LT(dq_diff, 1e-7) << "dq gradients differ across identical batch elements";
    EXPECT_LT(db_diff, 1e-7) << "db gradients differ across identical batch elements";
    EXPECT_LT(dP_diff, 1e-7) << "dP gradients differ across identical batch elements";
    EXPECT_LT(dA_diff, 1e-7) << "dA gradients differ across identical batch elements";
}

// ============================================================================
// Regression tests for PR review fixes
// ============================================================================

// Requesting Woodbury on an incompatible problem should throw
TEST_F(WoodburyTest, ExplicitWoodburyThrowsOnIncompatible) {
    // Build a problem with SOC cones (incompatible with Woodbury)
    int64_t n = 5, m = 5;
    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci(n);
    for (int64_t i = 0; i < n; ++i) { P_ro[i] = i; P_ci[i] = i; }
    P_ro[n] = n;

    // Simple A: identity
    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci(m);
    for (int64_t i = 0; i < m; ++i) { A_ro[i] = i; A_ci[i] = i; }
    A_ro[m] = m;

    Cones cones{};
    cones.numZeroCones = 2;
    cones.numNonnegCones = 0;
    cones.numSocCones = 1;
    cones.socConeDims = {3};

    Settings settings;
    settings.ipm.kktSolverType = KKTSolverType::Woodbury;

    EXPECT_THROW(
        CompiledSolver(n, m, 1,
            P_ro.data(), P_ci.data(), n,
            A_ro.data(), A_ci.data(), m,
            cones, settings),
        std::runtime_error
    ) << "Should throw when Woodbury is requested on incompatible problem";
}

// Explicit Woodbury request reports correct solver type
TEST_F(WoodburyTest, ExplicitWoodburyReportsCorrectType) {
    auto prob = buildPortfolio(10, 2);

    Settings settings;
    settings.ipm.kktSolverType = KKTSolverType::Woodbury;
    settings.verbose = false;

    CompiledSolver solver(
        prob.n, prob.m, 1,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings
    );

    EXPECT_EQ(solver.kkt->solverType(), KKTSolverType::Woodbury);
    EXPECT_TRUE(solver.kkt->isWoodbury());
    EXPECT_FALSE(solver.kkt->isRiccati());
}

// ============================================================================
// Benchmark: Woodbury vs cuDSS (forward + backward)
// ============================================================================

namespace {

double minMs(const std::vector<double>& samples) {
    if (samples.empty()) return std::numeric_limits<double>::quiet_NaN();
    return *std::min_element(samples.begin(), samples.end());
}

bool ldltVariantSupported(DiffWbLdltVariant variant, int64_t k_ext) {
    switch (variant) {
        case DiffWbLdltVariant::Auto:
        case DiffWbLdltVariant::Scalar:
            return true;
        case DiffWbLdltVariant::Warp:
            return k_ext <= 32;
        case DiffWbLdltVariant::Block:
            return k_ext <= 64;
    }
    return false;
}

void printBenchCell(double value_ms) {
    if (std::isnan(value_ms)) {
        std::cout << std::setw(11) << "n/a";
    } else {
        std::cout << std::setw(11) << std::fixed << std::setprecision(2) << value_ms;
    }
}

} // namespace

TEST_F(WoodburyTest, DISABLED_BenchmarkFullBackwardPipeline) {
    struct Config {
        int64_t n, k, batchSize;
        std::string label;
    };

    std::vector<Config> configs = {
        {50,    3,    1, "n=50    k=3   B=1   "},
        {50,    3,   64, "n=50    k=3   B=64  "},
        {200,  10,    1, "n=200   k=10  B=1   "},
        {200,  10,   64, "n=200   k=10  B=64  "},
        {500,  20,    1, "n=500   k=20  B=1   "},
        {500,  20,   64, "n=500   k=20  B=64  "},
        {1000, 30,    1, "n=1000  k=30  B=1   "},
        {1000, 30,   64, "n=1000  k=30  B=64  "},
        {2000, 50,    1, "n=2000  k=50  B=1   "},
    };

    const int WARMUP_ITER = 2;

    std::cout << "\n=== Full Woodbury Backward Pipeline Benchmark ===" << std::endl;
    std::cout << "Times are minimum milliseconds for solveAll + backward with"
              << " a size-dependent timed iteration count." << std::endl;
    std::cout << std::string(120, '-') << std::endl;
    std::cout << std::left
              << std::setw(24) << "Config"
              << std::setw(7) << "k_ext"
              << std::setw(11) << "cuDSS"
              << std::setw(11) << "auto"
              << std::setw(11) << "scalar"
              << std::setw(11) << "warp"
              << std::setw(11) << "block"
              << std::endl;
    std::cout << std::string(120, '-') << std::endl;

    for (auto& cfg : configs) {
        auto prob = buildPortfolio(cfg.n, cfg.k, 42);
        const int64_t k_ext = prob.k + 1;
        const int timed_iters = (cfg.n * cfg.batchSize >= 64000) ? 3 : 7;

        // Build batched data
        std::vector<double> P_bat, A_bat, q_bat, b_bat;
        for (int64_t b = 0; b < cfg.batchSize; ++b) {
            P_bat.insert(P_bat.end(), prob.P_values.begin(), prob.P_values.end());
            A_bat.insert(A_bat.end(), prob.A_values.begin(), prob.A_values.end());
            auto q_copy = prob.q_data;
            for (auto& v : q_copy) v *= (1.0 + 0.01 * b);
            q_bat.insert(q_bat.end(), q_copy.begin(), q_copy.end());
            b_bat.insert(b_bat.end(), prob.b_data.begin(), prob.b_data.end());
        }

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * prob.nnzP * cfg.batchSize);
        cudaMalloc(&d_A, sizeof(double) * prob.nnzA * cfg.batchSize);
        cudaMalloc(&d_q, sizeof(double) * prob.n * cfg.batchSize);
        cudaMalloc(&d_b, sizeof(double) * prob.m * cfg.batchSize);
        cudaMemcpy(d_P, P_bat.data(), sizeof(double) * P_bat.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_bat.data(), sizeof(double) * A_bat.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q_bat.data(), sizeof(double) * q_bat.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_bat.data(), sizeof(double) * b_bat.size(), cudaMemcpyHostToDevice);

        auto bench_pipeline = [&](KKTSolverType kkt_type, DiffWbLdltVariant variant) -> double {
            if (kkt_type == KKTSolverType::Woodbury && !ldltVariantSupported(variant, k_ext))
                return std::numeric_limits<double>::quiet_NaN();

            set_diff_wb_ldlt_variant(
                kkt_type == KKTSolverType::Woodbury ? variant : DiffWbLdltVariant::Auto);

            Settings s;
            s.verbose = false;
            s.maxIter = 100;
            s.enableGrad = true;
            s.ipm.kktSolverType = kkt_type;

            CompiledSolver solver(
                prob.n, prob.m, cfg.batchSize,
                prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
                prob.cones, s);

            BatchedVector dx_bar(prob.n, cfg.batchSize);
            BatchedVector dz_bar(prob.m, cfg.batchSize);
            BatchedVector ds_bar(prob.m, cfg.batchSize);
            std::vector<double> ones(prob.n * cfg.batchSize, 1.0);
            cudaMemcpy(dx_bar.data(), ones.data(), sizeof(double) * ones.size(), cudaMemcpyHostToDevice);
            dz_bar.setToConstant(0.0);
            ds_bar.setToConstant(0.0);

            for (int iter = 0; iter < WARMUP_ITER; ++iter) {
                solver.solveAll(d_P, d_A, d_q, d_b);
                backward(*solver.diff_state_, dx_bar, dz_bar, ds_bar, solver, 0);
            }
            cudaDeviceSynchronize();

            std::vector<double> samples;
            samples.reserve(timed_iters);
            for (int iter = 0; iter < timed_iters; ++iter) {
                auto t0 = std::chrono::high_resolution_clock::now();
                solver.solveAll(d_P, d_A, d_q, d_b);
                backward(*solver.diff_state_, dx_bar, dz_bar, ds_bar, solver, 0);
                cudaDeviceSynchronize();
                auto t1 = std::chrono::high_resolution_clock::now();
                samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }

            set_diff_wb_ldlt_variant(DiffWbLdltVariant::Auto);
            return minMs(samples);
        };

        const double cudss = bench_pipeline(KKTSolverType::CuDSS, DiffWbLdltVariant::Auto);
        const double wb_auto = bench_pipeline(KKTSolverType::Woodbury, DiffWbLdltVariant::Auto);
        const double wb_scalar = bench_pipeline(KKTSolverType::Woodbury, DiffWbLdltVariant::Scalar);
        const double wb_warp = bench_pipeline(KKTSolverType::Woodbury, DiffWbLdltVariant::Warp);
        const double wb_block = bench_pipeline(KKTSolverType::Woodbury, DiffWbLdltVariant::Block);

        std::cout << std::left << std::setw(24) << cfg.label
                  << std::setw(7) << k_ext;
        printBenchCell(cudss);
        printBenchCell(wb_auto);
        printBenchCell(wb_scalar);
        printBenchCell(wb_warp);
        printBenchCell(wb_block);
        std::cout << std::endl;

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    set_diff_wb_ldlt_variant(DiffWbLdltVariant::Auto);
    std::cout << std::string(120, '-') << std::endl;
    std::cout << "Variant coverage: warp <= 32, block <= 64, scalar always available, auto dispatches by size."
              << std::endl;
}

// ============================================================================
// Adversarial + fuzz suite ported from closed PR #103 (feat/woodbury-d-clamping).
// These never landed when that PR was closed; current main had zero Woodbury
// fuzz/adversarial coverage. Backward fuzz (FuzzBackwardSparseP) is ported
// separately against the current diff API.
// ============================================================================

// ============================================================================
// Adversarial tests: try to break Woodbury with ill-conditioned problems
// ============================================================================

// Helper: check KKT conditions for the conic QP:
//   min (1/2)x'Px + q'x  s.t.  Ax + s = b, s in K
// KKT conditions:
//   primal feasibility: Ax + s - b = 0
//   dual feasibility:   Px + q + A'z = 0
//   complementarity:    s'z = 0  (per cone)
//   cone membership:    s in K, z in K*
// Returns max residual. For zero cones s_i = 0, for nonneg cones s_i >= 0, z_i >= 0, s_i*z_i = 0.
struct KKTResiduals {
    double primal_feas;  // max |Ax + s - b|
    double dual_feas;    // max |Px + q + A'z|
    double complementarity;  // max |s_i * z_i| over nonneg cones
    double cone_viol;    // max violation of cone membership (s >= 0, z >= 0 for nonneg)
};

KKTResiduals checkKKT(
    int64_t n, int64_t m,
    const std::vector<int64_t>& P_ro, const std::vector<int64_t>& P_ci,
    const std::vector<double>& P_vals,
    const std::vector<int64_t>& A_ro, const std::vector<int64_t>& A_ci,
    const std::vector<double>& A_vals,
    const std::vector<double>& q, const std::vector<double>& b,
    const Cones& cones,
    const std::vector<double>& x, const std::vector<double>& z, const std::vector<double>& s)
{
    KKTResiduals res{0, 0, 0, 0};

    // Primal feasibility: Ax + s - b
    // A is CSR (m x n)
    for (int64_t i = 0; i < m; ++i) {
        double ax_i = 0;
        for (int64_t idx = A_ro[i]; idx < A_ro[i + 1]; ++idx)
            ax_i += A_vals[idx] * x[A_ci[idx]];
        double r = std::abs(ax_i + s[i] - b[i]);
        res.primal_feas = std::max(res.primal_feas, r);
    }

    // Dual feasibility: Px + q + A'z
    // P is CSR (n x n), A' computed by iterating CSR rows of A
    std::vector<double> dual_res(n);
    // Px
    for (int64_t i = 0; i < n; ++i) {
        double px_i = 0;
        for (int64_t idx = P_ro[i]; idx < P_ro[i + 1]; ++idx) {
            int64_t j = P_ci[idx];
            px_i += P_vals[idx] * x[j];
            if (j != i) // symmetric: add transpose contribution
                dual_res[j] += P_vals[idx] * x[i];
        }
        dual_res[i] += px_i + q[i];
    }
    // A'z
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t idx = A_ro[i]; idx < A_ro[i + 1]; ++idx)
            dual_res[A_ci[idx]] += A_vals[idx] * z[i];
    }
    for (int64_t i = 0; i < n; ++i)
        res.dual_feas = std::max(res.dual_feas, std::abs(dual_res[i]));

    // Complementarity and cone membership for nonneg cones
    int64_t row = cones.numZeroCones;  // skip zero cones
    for (int64_t i = 0; i < cones.numNonnegCones; ++i, ++row) {
        res.complementarity = std::max(res.complementarity, std::abs(s[row] * z[row]));
        res.cone_viol = std::max(res.cone_viol, std::max(-s[row], -z[row]));
    }
    // Zero cone complementarity: s should be 0 (already in primal_feas effectively)

    return res;
}

// Helper: solve with both CuDSS and Woodbury, return (status_cudss, status_wb,
// obj_cudss, obj_wb, x_cudss, x_wb, z, s).
struct DualSolveResult {
    int32_t status_cudss, status_wb;
    double obj_cudss, obj_wb;
    std::vector<double> x_cudss, x_wb;
    std::vector<double> z_cudss, z_wb;
    std::vector<double> s_cudss, s_wb;
};

DualSolveResult solveWithBoth(
    int64_t n, int64_t m, int64_t batchSize,
    const std::vector<int64_t>& P_ro, const std::vector<int64_t>& P_ci, int64_t nnzP,
    const std::vector<int64_t>& A_ro, const std::vector<int64_t>& A_ci, int64_t nnzA,
    const std::vector<double>& P_values, const std::vector<double>& A_values,
    const std::vector<double>& q_data, const std::vector<double>& b_data,
    const Cones& cones, bool verbose = false)
{
    DualSolveResult r;
    r.x_cudss.resize(n);
    r.x_wb.resize(n);
    r.z_cudss.resize(m);
    r.z_wb.resize(m);
    r.s_cudss.resize(m);
    r.s_wb.resize(m);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // CuDSS
    Settings s_c;
    s_c.ipm.kktSolverType = KKTSolverType::CuDSS;
    s_c.maxIter = 200; s_c.verbose = verbose;
    CompiledSolver sc(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                      A_ro.data(), A_ci.data(), nnzA, cones, s_c);
    sc.solveAll(d_P, d_A, d_q, d_b);
    cudaMemcpy(r.x_cudss.data(), sc.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.z_cudss.data(), sc.solution.z.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.s_cudss.data(), sc.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(&r.status_cudss, sc.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&r.obj_cudss, sc.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);

    // Woodbury
    Settings s_w;
    s_w.ipm.kktSolverType = KKTSolverType::Woodbury;
    s_w.maxIter = 200; s_w.verbose = verbose;
    CompiledSolver sw(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                      A_ro.data(), A_ci.data(), nnzA, cones, s_w);
    // Prove Woodbury actually ran: an explicit Woodbury request constructs the
    // Woodbury backend or throws — it never silently falls back to cuDSS
    // (kkt_solver.cpp). Guards against a gate change masking a failure as a pass.
    EXPECT_EQ(sw.kkt->solverType(), KKTSolverType::Woodbury)
        << "explicit Woodbury did not select the Woodbury backend";
    sw.solveAll(d_P, d_A, d_q, d_b);
    cudaMemcpy(r.x_wb.data(), sw.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.z_wb.data(), sw.solution.z.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.s_wb.data(), sw.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(&r.status_wb, sw.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&r.obj_wb, sw.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    return r;
}

// Test 1: Near-zero P diagonal entries.
// Some P[i] are 1e-14, making D[j] ≈ eps. This should poison d_tilde_inv.
TEST_F(WoodburyTest, AdversarialTinyPDiagonal) {
    const int64_t n = 20, k = 3;
    const int64_t m = k + n;  // k zero + n nonneg (sparse bounds)

    std::mt19937 rng(99);
    std::normal_distribution<double> nd(0.0, 1.0);

    // Diagonal P: first 5 entries are tiny, rest are normal
    std::vector<int64_t> P_ro(n + 1), P_ci(n);
    std::vector<double> P_vals(n);
    for (int64_t i = 0; i < n; ++i) {
        P_ro[i] = i; P_ci[i] = i;
        P_vals[i] = (i < 5) ? 1e-14 : 1.0;
    }
    P_ro[n] = n;

    // A = [F^T (k x n); -I (n x n)]
    std::vector<double> F(n * k);
    for (auto& v : F) v = nd(rng);

    int64_t nnzA = k * n + n;
    std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
    std::vector<double> A_vals(nnzA);
    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) {
            A_ci[idx] = j; A_vals[idx] = F[j * k + i]; idx++;
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        A_ro[k + i] = idx;
        A_ci[idx] = i; A_vals[idx] = -1.0; idx++;
    }
    A_ro[m] = idx;

    // q, b for a feasible problem
    std::vector<double> q(n), b(m);
    for (auto& v : q) v = nd(rng) * 0.1;
    // b_eq = F * x_mid, x_mid = 0.5 (all feasible for x >= 0)
    for (int64_t i = 0; i < k; ++i) {
        double sum = 0;
        for (int64_t j = 0; j < n; ++j) sum += F[j * k + i] * 0.5;
        b[i] = sum;
    }
    for (int64_t i = 0; i < n; ++i) b[k + i] = -0.5;  // -I * 0.5 + s = -0.5, s = 0 (tight)
    // Relax bounds to ensure strict feasibility
    for (int64_t i = 0; i < n; ++i) b[k + i] = -1.0;

    Cones cones{}; cones.numZeroCones = k; cones.numNonnegCones = n;

    auto r = solveWithBoth(n, m, 1, P_ro, P_ci, n, A_ro, A_ci, nnzA,
                           P_vals, A_vals, q, b, cones);

    // CuDSS should solve fine
    ASSERT_EQ(r.status_cudss, 1) << "CuDSS did not converge";
    // Woodbury must also converge and match
    ASSERT_EQ(r.status_wb, 1)
        << "Woodbury failed with tiny P diagonal (status=" << r.status_wb << ")";
    EXPECT_NEAR(r.obj_wb, r.obj_cudss, std::max(1e-4, std::abs(r.obj_cudss) * 1e-3))
        << "Objective mismatch with tiny P diagonal";
}

// Test 2: Nearly linearly dependent dense nonneg rows.
// Two dense inequality rows are nearly parallel → ill-conditioned Schur complement.
TEST_F(WoodburyTest, AdversarialParallelDenseRows) {
    const int64_t n = 20, k = 3, k_d = 2;
    const int64_t n_nonneg = k_d;  // only dense nonneg (no sparse bounds)
    const int64_t m = k + n_nonneg;

    std::mt19937 rng(77);
    std::normal_distribution<double> nd(0.0, 1.0);

    // Diagonal P (healthy)
    std::vector<int64_t> P_ro(n + 1), P_ci(n);
    std::vector<double> P_vals(n);
    for (int64_t i = 0; i < n; ++i) {
        P_ro[i] = i; P_ci[i] = i; P_vals[i] = 1.0;
    }
    P_ro[n] = n;

    // F (k x n) for zero cones
    std::vector<double> F(n * k);
    for (auto& v : F) v = nd(rng);

    // G (k_d x n) for dense nonneg — make row 1 nearly parallel to row 0
    std::vector<double> G(n * k_d, 0.0);
    for (int64_t j = 0; j < n; ++j) G[j * k_d + 0] = nd(rng);
    for (int64_t j = 0; j < n; ++j) G[j * k_d + 1] = G[j * k_d + 0] + 1e-12 * nd(rng);

    // A = [F; -G] — all rows are dense
    int64_t nnzA = k * n + k_d * n;
    std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
    std::vector<double> A_vals(nnzA);
    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) {
            A_ci[idx] = j; A_vals[idx] = F[j * k + i]; idx++;
        }
    }
    for (int64_t d = 0; d < k_d; ++d) {
        A_ro[k + d] = idx;
        for (int64_t j = 0; j < n; ++j) {
            A_ci[idx] = j; A_vals[idx] = -G[j * k_d + d]; idx++;
        }
    }
    A_ro[m] = idx;

    // Feasible q, b
    std::vector<double> q(n), b(m);
    for (auto& v : q) v = nd(rng) * 0.1;
    std::vector<double> x_mid(n, 0.5);
    for (int64_t i = 0; i < k; ++i) {
        double sum = 0;
        for (int64_t j = 0; j < n; ++j) sum += F[j * k + i] * x_mid[j];
        b[i] = sum;
    }
    for (int64_t d = 0; d < k_d; ++d) {
        double sum = 0;
        for (int64_t j = 0; j < n; ++j) sum += -G[j * k_d + d] * x_mid[j];
        b[k + d] = sum - 1.0;  // slack
    }

    Cones cones{}; cones.numZeroCones = k; cones.numNonnegCones = n_nonneg;

    // Verify Woodbury accepts this
    ASSERT_TRUE(WoodburyKKTData::isCompatible(
        n, m, P_ro.data(), P_ci.data(), n, A_ro.data(), A_ci.data(), nnzA, cones))
        << "Expected Woodbury to accept this problem";

    auto r = solveWithBoth(n, m, 1, P_ro, P_ci, n, A_ro, A_ci, nnzA,
                           P_vals, A_vals, q, b, cones);

    ASSERT_EQ(r.status_cudss, 1) << "CuDSS did not converge";
    ASSERT_EQ(r.status_wb, 1)
        << "Woodbury failed with nearly parallel dense rows (status=" << r.status_wb << ")";
    EXPECT_NEAR(r.obj_wb, r.obj_cudss, std::max(1e-4, std::abs(r.obj_cudss) * 1e-3))
        << "Objective mismatch with parallel dense rows";
}

// Test 3: k_total = n - 1 (Schur complement nearly as large as the system).
// Woodbury formula subtracts two large nearly-equal matrices → cancellation.
TEST_F(WoodburyTest, AdversarialSchurNearlyFull) {
    // n = 10, k = 9 → k_total = 9 = n - 1
    const int64_t n = 10, k = 9;
    const int64_t m = k + n;  // 9 zero + 10 nonneg (sparse bounds x >= 0)

    std::mt19937 rng(55);
    std::normal_distribution<double> nd(0.0, 1.0);

    // Diagonal P
    std::vector<int64_t> P_ro(n + 1), P_ci(n);
    std::vector<double> P_vals(n);
    for (int64_t i = 0; i < n; ++i) {
        P_ro[i] = i; P_ci[i] = i; P_vals[i] = 1.0;
    }
    P_ro[n] = n;

    // F (k x n) random dense
    std::vector<double> F(n * k);
    for (auto& v : F) v = nd(rng);

    // A = [F^T (k x n); -I (n x n)]
    int64_t nnzA = k * n + n;
    std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
    std::vector<double> A_vals(nnzA);
    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) {
            A_ci[idx] = j; A_vals[idx] = F[j * k + i]; idx++;
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        A_ro[k + i] = idx;
        A_ci[idx] = i; A_vals[idx] = -1.0; idx++;
    }
    A_ro[m] = idx;

    // Feasible q, b
    // x_mid = 0.5 is the feasible point.
    // Zero cones (equality): F^T x = b_eq → b_eq = F^T * x_mid
    // Nonneg cones: -x_i + s_i = b_i, s_i >= 0 → need b_i > -x_mid_i
    // Set b[k+i] = -x_mid - slack so s = x_mid + b[k+i] = -slack... wait:
    // s_i = b_i - (-1)*x_i = b_i + x_i, so s_i >= 0 requires b_i >= -x_i
    // At x_mid = 0.5: b_i = -0.5 + 1.0 = 0.5 gives s = 1.0 > 0
    std::vector<double> q(n), b(m);
    for (auto& v : q) v = nd(rng) * 0.1;
    for (int64_t i = 0; i < k; ++i) {
        double sum = 0;
        for (int64_t j = 0; j < n; ++j) sum += F[j * k + i] * 0.5;
        b[i] = sum;
    }
    for (int64_t i = 0; i < n; ++i) b[k + i] = -0.5 + 1.0;  // s_i = b_i + x_i = 1.5 at x_mid

    Cones cones{}; cones.numZeroCones = k; cones.numNonnegCones = n;

    ASSERT_TRUE(WoodburyKKTData::isCompatible(
        n, m, P_ro.data(), P_ci.data(), n, A_ro.data(), A_ci.data(), nnzA, cones));

    auto r = solveWithBoth(n, m, 1, P_ro, P_ci, n, A_ro, A_ci, nnzA,
                           P_vals, A_vals, q, b, cones);

    ASSERT_EQ(r.status_wb, 1)
        << "Woodbury failed with k_total=n-1 (status=" << r.status_wb << ")";
    if (r.status_cudss == 1) {
        EXPECT_NEAR(r.obj_wb, r.obj_cudss, std::max(1e-4, std::abs(r.obj_cudss) * 1e-3))
            << "Objective mismatch with nearly-full Schur complement";
    } else {
        auto kkt = checkKKT(n, m, P_ro, P_ci, P_vals, A_ro, A_ci, A_vals, q, b, cones,
                            r.x_wb, r.z_wb, r.s_wb);
        EXPECT_LT(kkt.primal_feas, 1e-6) << "KKT primal feasibility violation";
        EXPECT_LT(kkt.dual_feas, 1e-6) << "KKT dual feasibility violation";
        EXPECT_LT(kkt.complementarity, 1e-6) << "KKT complementarity violation";
        EXPECT_LE(kkt.cone_viol, 1e-6) << "KKT cone membership violation";
    }
}

// Test 4: Sparse nonneg rows with tiny A values.
// Columns are "covered" by sparse nonneg rows, but A[i] ≈ 1e-14 →
// contribution a²·w² ≈ 0 → D[j] ≈ eps → d_tilde_inv huge.
TEST_F(WoodburyTest, AdversarialTinySparseA) {
    const int64_t n = 20, k = 3;
    const int64_t m = k + n;

    std::mt19937 rng(33);
    std::normal_distribution<double> nd(0.0, 1.0);

    // Sparse P: half the columns have large P, half have tiny P (1e-6)
    // All columns have entries so the problem remains bounded
    std::vector<int64_t> P_ro(n + 1), P_ci(n);
    std::vector<double> P_vals(n);
    for (int64_t i = 0; i < n; ++i) {
        P_ro[i] = i;
        P_ci[i] = i;
        P_vals[i] = (i % 2 == 0) ? 1.0 : 1e-6;
    }
    P_ro[n] = n;
    int64_t nnzP = n;

    // A = [F^T (k x n); diag(a) (n x n)]
    // For odd columns (not covered by P), a[i] = 1e-6 → small but not degenerate
    // (1e-14 would make the problem dual infeasible since those columns have no P entry)
    std::vector<double> F(n * k);
    for (auto& v : F) v = nd(rng);

    int64_t nnzA = k * n + n;
    std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
    std::vector<double> A_vals(nnzA);
    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) {
            A_ci[idx] = j; A_vals[idx] = F[j * k + i]; idx++;
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        A_ro[k + i] = idx;
        A_ci[idx] = i;
        // Odd columns: small A values to stress Woodbury numerics
        A_vals[idx] = (i % 2 == 1) ? -1e-6 : -1.0;
        idx++;
    }
    A_ro[m] = idx;

    // Feasible q, b at x_mid = 0.5
    // Nonneg row i: a_i * x_i + s_i = b_i, s_i >= 0
    // At x_mid: s_i = b_i - a_i * 0.5, need s_i > 0
    // Set b[k+i] = a_i * 0.5 + 1.0 so s_i = 1.0 at x_mid
    std::vector<double> q(n), b(m);
    for (auto& v : q) v = nd(rng) * 0.1;
    for (int64_t i = 0; i < k; ++i) {
        double sum = 0;
        for (int64_t j = 0; j < n; ++j) sum += F[j * k + i] * 0.5;
        b[i] = sum;
    }
    for (int64_t i = 0; i < n; ++i) {
        double a_i = (i % 2 == 1) ? -1e-6 : -1.0;
        b[k + i] = a_i * 0.5 + 1.0;  // s_i = 1.0 at x_mid
    }

    Cones cones{}; cones.numZeroCones = k; cones.numNonnegCones = n;

    // Should pass detection (columns are structurally covered)
    ASSERT_TRUE(WoodburyKKTData::isCompatible(
        n, m, P_ro.data(), P_ci.data(), nnzP, A_ro.data(), A_ci.data(), nnzA, cones));

    auto r = solveWithBoth(n, m, 1, P_ro, P_ci, nnzP, A_ro, A_ci, nnzA,
                           P_vals, A_vals, q, b, cones);

    ASSERT_EQ(r.status_wb, 1)
        << "Woodbury failed with tiny sparse A values (status=" << r.status_wb << ")";
    if (r.status_cudss == 1) {
        EXPECT_NEAR(r.obj_wb, r.obj_cudss, std::max(1e-4, std::abs(r.obj_cudss) * 1e-3))
            << "Objective mismatch with tiny sparse A values";
    } else {
        auto kkt = checkKKT(n, m, P_ro, P_ci, P_vals, A_ro, A_ci, A_vals, q, b, cones,
                            r.x_wb, r.z_wb, r.s_wb);
        EXPECT_LT(kkt.primal_feas, 1e-6) << "KKT primal feasibility violation";
        EXPECT_LT(kkt.dual_feas, 1e-6) << "KKT dual feasibility violation";
        EXPECT_LT(kkt.complementarity, 1e-6) << "KKT complementarity violation";
        EXPECT_LE(kkt.cone_viol, 1e-6) << "KKT cone membership violation";
    }
}

// Test 5: Wildly different P diagonal scales (triggers extreme equilibration).
// P[0..4] = 1e8, P[5..19] = 1e-8 → equilibration creates skewed D.
TEST_F(WoodburyTest, AdversarialExtremeScaling) {
    const int64_t n = 20, k = 3;
    const int64_t m = k + n;

    std::mt19937 rng(44);
    std::normal_distribution<double> nd(0.0, 1.0);

    std::vector<int64_t> P_ro(n + 1), P_ci(n);
    std::vector<double> P_vals(n);
    for (int64_t i = 0; i < n; ++i) {
        P_ro[i] = i; P_ci[i] = i;
        P_vals[i] = (i < 5) ? 1e8 : 1e-8;
    }
    P_ro[n] = n;

    std::vector<double> F(n * k);
    for (auto& v : F) v = nd(rng);

    int64_t nnzA = k * n + n;
    std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
    std::vector<double> A_vals(nnzA);
    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) {
            A_ci[idx] = j; A_vals[idx] = F[j * k + i]; idx++;
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        A_ro[k + i] = idx;
        A_ci[idx] = i; A_vals[idx] = -1.0; idx++;
    }
    A_ro[m] = idx;

    std::vector<double> q(n), b(m);
    for (auto& v : q) v = nd(rng) * 0.1;
    for (int64_t i = 0; i < k; ++i) {
        double sum = 0;
        for (int64_t j = 0; j < n; ++j) sum += F[j * k + i] * 0.5;
        b[i] = sum;
    }
    for (int64_t i = 0; i < n; ++i) b[k + i] = -1.0;

    Cones cones{}; cones.numZeroCones = k; cones.numNonnegCones = n;

    auto r = solveWithBoth(n, m, 1, P_ro, P_ci, n, A_ro, A_ci, nnzA,
                           P_vals, A_vals, q, b, cones);

    ASSERT_EQ(r.status_cudss, 1) << "CuDSS did not converge";
    ASSERT_EQ(r.status_wb, 1)
        << "Woodbury failed with extreme P scaling (status=" << r.status_wb << ")";
    EXPECT_NEAR(r.obj_wb, r.obj_cudss, std::max(1e-3, std::abs(r.obj_cudss) * 1e-2))
        << "Objective mismatch with extreme scaling";
}

// Test 6: Dense nonneg row nearly parallel to a zero-cone row.
// This makes columns of F_all nearly dependent across the zero/dense boundary.
TEST_F(WoodburyTest, AdversarialDenseParallelToEquality) {
    const int64_t n = 15, k = 2, k_d = 1;
    const int64_t n_nonneg = k_d + n;  // 1 dense + n sparse
    const int64_t m = k + n_nonneg;

    std::mt19937 rng(66);
    std::normal_distribution<double> nd(0.0, 1.0);

    // Diagonal P
    std::vector<int64_t> P_ro(n + 1), P_ci(n);
    std::vector<double> P_vals(n);
    for (int64_t i = 0; i < n; ++i) {
        P_ro[i] = i; P_ci[i] = i; P_vals[i] = 1.0;
    }
    P_ro[n] = n;

    // Zero-cone row 0: random dense
    std::vector<double> F0(n), F1(n);
    for (int64_t j = 0; j < n; ++j) F0[j] = nd(rng);
    for (int64_t j = 0; j < n; ++j) F1[j] = nd(rng);

    // Dense nonneg row: nearly identical to F0
    std::vector<double> G0(n);
    for (int64_t j = 0; j < n; ++j) G0[j] = F0[j] + 1e-13 * nd(rng);

    // A = [F0; F1; -G0; -I]
    int64_t nnzA = k * n + k_d * n + n;
    std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
    std::vector<double> A_vals(nnzA);
    int64_t idx = 0;
    // Zero rows
    A_ro[0] = idx;
    for (int64_t j = 0; j < n; ++j) { A_ci[idx] = j; A_vals[idx] = F0[j]; idx++; }
    A_ro[1] = idx;
    for (int64_t j = 0; j < n; ++j) { A_ci[idx] = j; A_vals[idx] = F1[j]; idx++; }
    // Dense nonneg row
    A_ro[2] = idx;
    for (int64_t j = 0; j < n; ++j) { A_ci[idx] = j; A_vals[idx] = -G0[j]; idx++; }
    // Sparse nonneg rows (-I)
    for (int64_t i = 0; i < n; ++i) {
        A_ro[k + k_d + i] = idx;
        A_ci[idx] = i; A_vals[idx] = -1.0; idx++;
    }
    A_ro[m] = idx;

    // Feasible q, b
    std::vector<double> x_mid(n, 0.5);
    std::vector<double> q(n), b(m);
    for (auto& v : q) v = nd(rng) * 0.1;
    // b_eq
    b[0] = 0; b[1] = 0;
    for (int64_t j = 0; j < n; ++j) { b[0] += F0[j] * x_mid[j]; b[1] += F1[j] * x_mid[j]; }
    // b for dense nonneg: -G0 * x + s = b, s >= 0
    // At x_mid: s = b - (-G0' * x_mid) = b + G0' * x_mid
    // Set b so s = 1.0 at x_mid: b = -G0' * x_mid + 1.0
    b[2] = 0;
    for (int64_t j = 0; j < n; ++j) b[2] += -G0[j] * x_mid[j];
    b[2] += 1.0;  // s = 1.0 at x_mid
    // b for sparse nonneg: -x_i + s_i = b_i, s_i = b_i + x_i
    // Set b so s = 1.0 at x_mid: b_i = 1.0 - x_mid = 0.5
    for (int64_t i = 0; i < n; ++i) b[k + k_d + i] = -0.5 + 1.0;

    Cones cones{};
    cones.numZeroCones = k;
    cones.numNonnegCones = n_nonneg;

    ASSERT_TRUE(WoodburyKKTData::isCompatible(
        n, m, P_ro.data(), P_ci.data(), n, A_ro.data(), A_ci.data(), nnzA, cones));

    auto r = solveWithBoth(n, m, 1, P_ro, P_ci, n, A_ro, A_ci, nnzA,
                           P_vals, A_vals, q, b, cones);

    ASSERT_EQ(r.status_wb, 1)
        << "Woodbury failed with dense row parallel to equality (status=" << r.status_wb << ")";
    if (r.status_cudss == 1) {
        EXPECT_NEAR(r.obj_wb, r.obj_cudss, std::max(1e-4, std::abs(r.obj_cudss) * 1e-3))
            << "Objective mismatch with dense row parallel to equality";
    } else {
        auto kkt = checkKKT(n, m, P_ro, P_ci, P_vals, A_ro, A_ci, A_vals, q, b, cones,
                            r.x_wb, r.z_wb, r.s_wb);
        EXPECT_LT(kkt.primal_feas, 1e-6) << "KKT primal feasibility violation";
        EXPECT_LT(kkt.dual_feas, 1e-6) << "KKT dual feasibility violation";
        EXPECT_LT(kkt.complementarity, 1e-6) << "KKT complementarity violation";
        EXPECT_LE(kkt.cone_viol, 1e-6) << "KKT cone membership violation";
    }
}

// Test 7: Zero P (pure LP) — all coverage comes from sparse nonneg rows.
// D = eps + A_s^T Gamma A_s, eps ≈ 1e-13 → D dominated by cone scaling.
TEST_F(WoodburyTest, AdversarialZeroP) {
    const int64_t n = 15, k = 3;
    const int64_t m = k + n;

    std::mt19937 rng(88);
    std::normal_distribution<double> nd(0.0, 1.0);

    // P = 0 (but structurally diagonal for coverage check to pass)
    // Actually — P must have entries to cover columns. With P=0, coverage
    // comes from the -I nonneg rows. P can be empty (nnzP=0).
    // But isCompatible requires coverage: P or sparse nonneg.
    // Sparse nonneg rows (-I) cover all columns, so P can be zero.
    std::vector<int64_t> P_ro(n + 1, 0), P_ci;
    std::vector<double> P_vals;
    int64_t nnzP = 0;

    std::vector<double> F(n * k);
    for (auto& v : F) v = nd(rng);

    int64_t nnzA = k * n + n;
    std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
    std::vector<double> A_vals(nnzA);
    int64_t idx = 0;
    for (int64_t i = 0; i < k; ++i) {
        A_ro[i] = idx;
        for (int64_t j = 0; j < n; ++j) {
            A_ci[idx] = j; A_vals[idx] = F[j * k + i]; idx++;
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        A_ro[k + i] = idx;
        A_ci[idx] = i; A_vals[idx] = -1.0; idx++;
    }
    A_ro[m] = idx;

    std::vector<double> q(n), b(m);
    for (auto& v : q) v = nd(rng);
    for (int64_t i = 0; i < k; ++i) {
        double sum = 0;
        for (int64_t j = 0; j < n; ++j) sum += F[j * k + i] * 0.5;
        b[i] = sum;
    }
    for (int64_t i = 0; i < n; ++i) b[k + i] = -1.0;

    Cones cones{}; cones.numZeroCones = k; cones.numNonnegCones = n;

    ASSERT_TRUE(WoodburyKKTData::isCompatible(
        n, m, P_ro.data(), P_ci.data(), nnzP, A_ro.data(), A_ci.data(), nnzA, cones));

    auto r = solveWithBoth(n, m, 1, P_ro, P_ci, nnzP, A_ro, A_ci, nnzA,
                           P_vals, A_vals, q, b, cones);

    ASSERT_EQ(r.status_cudss, 1) << "CuDSS did not converge";
    ASSERT_EQ(r.status_wb, 1)
        << "Woodbury failed on pure LP with zero P (status=" << r.status_wb << ")";
    EXPECT_NEAR(r.obj_wb, r.obj_cudss, std::max(1e-4, std::abs(r.obj_cudss) * 1e-3))
        << "Objective mismatch on pure LP";
}

// ============================================================================
// Fuzz tester: random Woodbury-compatible problems
// ============================================================================

// Generate a random Woodbury-compatible problem that is guaranteed feasible and bounded.
struct FuzzProblem {
    int64_t n, m, k, k_d, n_nonneg;
    int64_t nnzP, nnzA;
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double> P_vals;
    std::vector<int64_t> A_ro, A_ci;
    std::vector<double> A_vals;
    std::vector<double> q, b;
    Cones cones;
};

FuzzProblem generateFuzzProblem(std::mt19937& rng) {
    FuzzProblem prob;

    // Sample dimensions
    std::vector<int64_t> n_choices = {5, 8, 10, 15, 20, 30, 50};
    prob.n = n_choices[std::uniform_int_distribution<int>(0, n_choices.size() - 1)(rng)];
    int64_t n = prob.n;

    // k = number of zero cones (equality rows with dense entries)
    int64_t max_k = std::min(n / 2, (int64_t)10);
    prob.k = std::uniform_int_distribution<int64_t>(0, max_k)(rng);
    int64_t k = prob.k;

    // k_d = number of dense nonneg rows
    int64_t max_kd = std::min(n / 3, (int64_t)5);
    max_kd = std::max((int64_t)0, std::min(max_kd, n - k - 2));
    prob.k_d = (max_kd > 0) ? std::uniform_int_distribution<int64_t>(0, max_kd)(rng) : 0;
    int64_t k_d = prob.k_d;

    // Decide which columns have P entries.
    // Columns without P model variables like x in "min r'Dr s.t. r = Fx, x >= 0".
    // They need box constraints (lower + upper bound) to guarantee boundedness.
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    double p_frac = unit(rng);  // fraction of columns with P > 0
    // 10% chance of P=0 on all columns (pure LP with box constraints)
    if (unit(rng) < 0.1) p_frac = 0.0;

    std::vector<bool> has_P(n, false);
    int64_t n_P = static_cast<int64_t>(std::ceil(p_frac * n));
    std::vector<int64_t> col_perm(n);
    std::iota(col_perm.begin(), col_perm.end(), 0);
    std::shuffle(col_perm.begin(), col_perm.end(), rng);
    for (int64_t i = 0; i < n_P; ++i) has_P[col_perm[i]] = true;

    // Sparse nonneg rows.
    // Columns with P: get one sparse nonneg row (lower bound), like x >= 0.
    // Columns without P: get two sparse nonneg rows (lower + upper bound)
    //   to guarantee boundedness without relying on indirect coupling.
    // Also add optional extra random bounds.
    std::vector<std::pair<int64_t, double>> sparse_nonneg_entries;  // (col, coefficient)
    for (int64_t j = 0; j < n; ++j) {
        // Lower bound: -x_j + s = b, s >= 0 => x_j >= -b (i.e. x_j <= b when coeff is +1)
        sparse_nonneg_entries.push_back({j, -1.0});
        if (!has_P[j]) {
            // Upper bound: +x_j + s = b, s >= 0 => x_j <= b
            sparse_nonneg_entries.push_back({j, 1.0});
        }
    }
    // Add extra random bounds (0 to n additional)
    int64_t n_extra = std::uniform_int_distribution<int64_t>(0, n)(rng);
    for (int64_t i = 0; i < n_extra; ++i) {
        int64_t col = std::uniform_int_distribution<int64_t>(0, n - 1)(rng);
        double sign = (unit(rng) < 0.5) ? -1.0 : 1.0;
        sparse_nonneg_entries.push_back({col, sign});
    }
    // Sort by column for CSR
    std::sort(sparse_nonneg_entries.begin(), sparse_nonneg_entries.end());

    int64_t n_sparse = sparse_nonneg_entries.size();
    prob.n_nonneg = k_d + n_sparse;
    prob.m = k + prob.n_nonneg;
    int64_t m = prob.m;

    // Build P (sparse diagonal CSR — only columns with has_P)
    std::uniform_real_distribution<double> log_scale(-4.0, 4.0);
    prob.P_ro.resize(n + 1);
    int64_t pidx = 0;
    for (int64_t i = 0; i < n; ++i) {
        prob.P_ro[i] = pidx;
        if (has_P[i]) {
            prob.P_ci.push_back(i);
            prob.P_vals.push_back(std::pow(10.0, log_scale(rng)));
            pidx++;
        }
    }
    prob.P_ro[n] = pidx;
    prob.nnzP = pidx;

    // Build A
    std::normal_distribution<double> nd(0.0, 1.0);
    // nnzA = k * n (dense zero rows) + k_d * n (dense nonneg rows) + n_sparse (sparse nonneg)
    prob.nnzA = k * n + k_d * n + n_sparse;
    prob.A_ro.resize(m + 1);
    prob.A_ci.resize(prob.nnzA);
    prob.A_vals.resize(prob.nnzA);

    int64_t aidx = 0;
    // Zero-cone rows (dense)
    for (int64_t i = 0; i < k; ++i) {
        prob.A_ro[i] = aidx;
        for (int64_t j = 0; j < n; ++j) {
            prob.A_ci[aidx] = j;
            prob.A_vals[aidx] = nd(rng);
            aidx++;
        }
    }
    // Dense nonneg rows
    for (int64_t i = 0; i < k_d; ++i) {
        prob.A_ro[k + i] = aidx;
        for (int64_t j = 0; j < n; ++j) {
            prob.A_ci[aidx] = j;
            prob.A_vals[aidx] = -std::abs(nd(rng));  // negative so -A*x + s = b → s = b + A*x
            aidx++;
        }
    }
    // Sparse nonneg rows (one entry each)
    for (int64_t i = 0; i < n_sparse; ++i) {
        prob.A_ro[k + k_d + i] = aidx;
        prob.A_ci[aidx] = sparse_nonneg_entries[i].first;
        prob.A_vals[aidx] = sparse_nonneg_entries[i].second;
        aidx++;
    }
    prob.A_ro[m] = aidx;

    // Cones
    prob.cones = Cones{};
    prob.cones.numZeroCones = k;
    prob.cones.numNonnegCones = prob.n_nonneg;

    // Generate feasible q, b
    // x_mid: strictly positive interior point
    std::vector<double> x_mid(n);
    for (auto& v : x_mid) v = std::uniform_real_distribution<double>(0.1, 1.0)(rng);

    prob.q.resize(n);
    for (auto& v : prob.q) v = nd(rng);

    prob.b.resize(m);
    for (int64_t i = 0; i < m; ++i) {
        double Ax_i = 0;
        for (int64_t idx = prob.A_ro[i]; idx < prob.A_ro[i + 1]; ++idx)
            Ax_i += prob.A_vals[idx] * x_mid[prob.A_ci[idx]];
        if (i < k) {
            // Zero cone: equality, s = 0
            prob.b[i] = Ax_i;
        } else {
            // Nonneg cone: s = b - Ax >= 0, set slack in [0.5, 2.0]
            double slack = std::uniform_real_distribution<double>(0.5, 2.0)(rng);
            prob.b[i] = Ax_i + slack;
        }
    }

    return prob;
}

TEST_F(WoodburyTest, FuzzRandomProblems) {
    const int NUM_INSTANCES = 200;
    const unsigned BASE_SEED = 54321;

    int both_solved = 0, wb_only = 0, cudss_only = 0, both_failed = 0, skipped = 0;

    for (int trial = 0; trial < NUM_INSTANCES; ++trial) {
        std::mt19937 rng(BASE_SEED + trial);
        auto prob = generateFuzzProblem(rng);

        // Check Woodbury compatibility
        bool compat = WoodburyKKTData::isCompatible(
            prob.n, prob.m,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones);
        if (!compat) {
            skipped++;
            continue;
        }

        auto r = solveWithBoth(prob.n, prob.m, 1,
                               prob.P_ro, prob.P_ci, prob.nnzP,
                               prob.A_ro, prob.A_ci, prob.nnzA,
                               prob.P_vals, prob.A_vals, prob.q, prob.b,
                               prob.cones);

        bool wb_ok = (r.status_wb == 1 || r.status_wb == 4);   // Solved or AlmostSolved
        bool cd_ok = (r.status_cudss == 1 || r.status_cudss == 4);

        if (wb_ok && cd_ok) {
            both_solved++;
            EXPECT_NEAR(r.obj_wb, r.obj_cudss,
                        std::max(1e-4, std::abs(r.obj_cudss) * 1e-3))
                << "Trial " << trial << " (n=" << prob.n << " k=" << prob.k
                << " k_d=" << prob.k_d << " seed=" << (BASE_SEED + trial)
                << "): objective mismatch";
        } else if (wb_ok && !cd_ok) {
            wb_only++;
            // Validate Woodbury via KKT
            auto kkt = checkKKT(prob.n, prob.m,
                                prob.P_ro, prob.P_ci, prob.P_vals,
                                prob.A_ro, prob.A_ci, prob.A_vals,
                                prob.q, prob.b, prob.cones,
                                r.x_wb, r.z_wb, r.s_wb);
            EXPECT_LT(kkt.primal_feas, 1e-6)
                << "Trial " << trial << " KKT primal feas";
            EXPECT_LT(kkt.dual_feas, 1e-6)
                << "Trial " << trial << " KKT dual feas";
            EXPECT_LT(kkt.complementarity, 1e-6)
                << "Trial " << trial << " KKT complementarity";
            EXPECT_LE(kkt.cone_viol, 1e-6)
                << "Trial " << trial << " KKT cone viol";
        } else if (!wb_ok && cd_ok) {
            cudss_only++;
            ADD_FAILURE() << "Trial " << trial << " (n=" << prob.n
                << " k=" << prob.k << " k_d=" << prob.k_d
                << " seed=" << (BASE_SEED + trial)
                << "): Woodbury status=" << r.status_wb
                << " but CuDSS solved — regression";
        } else {
            both_failed++;
            // Both failed on a problem we constructed to be feasible — unexpected
            // Log but don't hard-fail (numerical edge case)
            std::cout << "  [trial " << trial << "] both failed: wb=" << r.status_wb
                      << " cudss=" << r.status_cudss
                      << " (n=" << prob.n << " k=" << prob.k << " k_d=" << prob.k_d
                      << " seed=" << (BASE_SEED + trial) << ")" << std::endl;
        }
    }

    std::cout << "\nFuzz results (" << NUM_INSTANCES << " trials, "
              << skipped << " skipped incompatible):\n"
              << "  Both solved:  " << both_solved << "\n"
              << "  WB only:      " << wb_only << "\n"
              << "  CuDSS only:   " << cudss_only << "\n"
              << "  Both failed:  " << both_failed << std::endl;

    // Hard requirement: Woodbury must never fail when CuDSS succeeds
    EXPECT_EQ(cudss_only, 0) << "Woodbury failed on problems CuDSS solved";
    // Most problems should be solved by at least one solver
    EXPECT_LT(both_failed, NUM_INSTANCES / 10)
        << "Too many problems unsolvable by both solvers";
}

TEST_F(WoodburyTest, FuzzBatchedSparseP) {
    // Use the same seeds as FuzzRandomProblems (known feasible+bounded)
    // and batch 4 problems together with different q vectors.
    const int NUM_INSTANCES = 50;
    const unsigned BASE_SEED = 54321;  // same as FuzzRandomProblems
    const int64_t batchSize = 4;

    int both_solved = 0, wb_only = 0, cudss_only = 0, both_failed = 0, skipped = 0;

    for (int trial = 0; trial < NUM_INSTANCES; ++trial) {
        std::mt19937 rng(BASE_SEED + trial);
        auto prob = generateFuzzProblem(rng);

        bool compat = WoodburyKKTData::isCompatible(
            prob.n, prob.m,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones);
        if (!compat) { skipped++; continue; }

        // Batch 4 copies with small perturbations to q.
        // Feasibility (determined by A, b, cones) is unchanged.
        // Boundedness is guaranteed by P + box constraints.
        std::vector<double> q_bat, b_bat;
        for (int64_t b = 0; b < batchSize; ++b) {
            for (int64_t i = 0; i < prob.n; ++i)
                q_bat.push_back(prob.q[i] + 0.01 * b);  // additive perturbation
            b_bat.insert(b_bat.end(), prob.b.begin(), prob.b.end());
        }

        // Replicate P and A for all batch elements
        std::vector<double> P_bat, A_bat;
        for (int64_t b = 0; b < batchSize; ++b) {
            P_bat.insert(P_bat.end(), prob.P_vals.begin(), prob.P_vals.end());
            A_bat.insert(A_bat.end(), prob.A_vals.begin(), prob.A_vals.end());
        }

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * prob.nnzP * batchSize);
        cudaMalloc(&d_A, sizeof(double) * prob.nnzA * batchSize);
        cudaMalloc(&d_q, sizeof(double) * prob.n * batchSize);
        cudaMalloc(&d_b, sizeof(double) * prob.m * batchSize);
        cudaMemcpy(d_P, P_bat.data(), sizeof(double) * P_bat.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_bat.data(), sizeof(double) * A_bat.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q_bat.data(), sizeof(double) * q_bat.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_bat.data(), sizeof(double) * b_bat.size(), cudaMemcpyHostToDevice);

        // Solve with both backends
        auto solve_one = [&](KKTSolverType kkt) {
            Settings s;
            s.ipm.kktSolverType = kkt;
            s.maxIter = 200; s.verbose = false;
            CompiledSolver solver(prob.n, prob.m, batchSize,
                                  prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                                  prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
                                  prob.cones, s);
            solver.solveAll(d_P, d_A, d_q, d_b);
            std::vector<int32_t> statuses(batchSize);
            std::vector<double> objs(batchSize);
            cudaMemcpy(statuses.data(), solver.solution.status.get(),
                       sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
            cudaMemcpy(objs.data(), solver.solution.obj_val.data(),
                       sizeof(double) * batchSize, cudaMemcpyDeviceToHost);
            return std::make_pair(statuses, objs);
        };

        auto [st_cd, obj_cd] = solve_one(KKTSolverType::CuDSS);
        auto [st_wb, obj_wb] = solve_one(KKTSolverType::Woodbury);

        bool all_cd_ok = true, all_wb_ok = true;
        for (int64_t b = 0; b < batchSize; ++b) {
            if (st_cd[b] != 1 && st_cd[b] != 4) all_cd_ok = false;
            if (st_wb[b] != 1 && st_wb[b] != 4) all_wb_ok = false;
        }

        if (all_wb_ok && all_cd_ok) {
            both_solved++;
            for (int64_t b = 0; b < batchSize; ++b) {
                EXPECT_NEAR(obj_wb[b], obj_cd[b],
                            std::max(1e-4, std::abs(obj_cd[b]) * 1e-3))
                    << "Trial " << trial << " batch " << b << " objective mismatch";
            }
        } else if (all_wb_ok && !all_cd_ok) {
            wb_only++;
        } else if (!all_wb_ok && all_cd_ok) {
            cudss_only++;
            ADD_FAILURE() << "Trial " << trial << " (n=" << prob.n
                << " k=" << prob.k << " k_d=" << prob.k_d
                << "): Woodbury failed in batched mode but CuDSS solved";
        } else {
            both_failed++;
        }

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    std::cout << "\nBatched fuzz results (" << NUM_INSTANCES << " trials, "
              << skipped << " skipped):\n"
              << "  Both solved: " << both_solved << "  WB only: " << wb_only
              << "  CuDSS only: " << cudss_only << "  Both failed: " << both_failed << std::endl;

    EXPECT_EQ(cudss_only, 0) << "Woodbury failed on batched problems CuDSS solved";
    EXPECT_EQ(both_failed, 0) << "Both solvers failed on feasible+bounded problem";
}

TEST_F(WoodburyTest, FuzzLargeN) {
    // Test n > 1024 to exercise the multi-column-per-thread path in the clamped kernel.
    // Use n=2048 with sparse P (50% columns have P entry).
    const int NUM_INSTANCES = 10;
    const unsigned BASE_SEED = 88888;

    int both_solved = 0, cudss_only = 0, skipped = 0;

    for (int trial = 0; trial < NUM_INSTANCES; ++trial) {
        std::mt19937 rng(BASE_SEED + trial);

        // Fixed large dimensions
        const int64_t n = 2048;
        const int64_t k = 3;   // few equality constraints
        const int64_t k_d = 0; // no dense nonneg (keep it simple)

        std::uniform_real_distribution<double> unit(0.0, 1.0);
        std::normal_distribution<double> nd(0.0, 1.0);
        std::uniform_real_distribution<double> log_scale(-4.0, 4.0);

        // P: 50% of columns have entries
        std::vector<bool> has_P(n, false);
        for (int64_t j = 0; j < n; ++j) has_P[j] = (unit(rng) < 0.5);

        // Sparse nonneg: lower bound on every column, upper bound on uncovered columns
        std::vector<std::pair<int64_t, double>> sparse_entries;
        for (int64_t j = 0; j < n; ++j) {
            sparse_entries.push_back({j, -1.0});
            if (!has_P[j]) sparse_entries.push_back({j, 1.0});
        }
        std::sort(sparse_entries.begin(), sparse_entries.end());
        int64_t n_sparse = sparse_entries.size();
        int64_t n_nonneg = k_d + n_sparse;
        int64_t m = k + n_nonneg;

        // Build P
        std::vector<int64_t> P_ro(n + 1);
        std::vector<int64_t> P_ci;
        std::vector<double> P_vals;
        int64_t pidx = 0;
        for (int64_t i = 0; i < n; ++i) {
            P_ro[i] = pidx;
            if (has_P[i]) {
                P_ci.push_back(i);
                P_vals.push_back(std::pow(10.0, log_scale(rng)));
                pidx++;
            }
        }
        P_ro[n] = pidx;
        int64_t nnzP = pidx;

        // Build A
        int64_t nnzA = k * n + n_sparse;
        std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
        std::vector<double> A_vals(nnzA);
        int64_t aidx = 0;
        for (int64_t i = 0; i < k; ++i) {
            A_ro[i] = aidx;
            for (int64_t j = 0; j < n; ++j) {
                A_ci[aidx] = j; A_vals[aidx] = nd(rng); aidx++;
            }
        }
        for (int64_t i = 0; i < n_sparse; ++i) {
            A_ro[k + i] = aidx;
            A_ci[aidx] = sparse_entries[i].first;
            A_vals[aidx] = sparse_entries[i].second;
            aidx++;
        }
        A_ro[m] = aidx;

        Cones cones{}; cones.numZeroCones = k; cones.numNonnegCones = n_nonneg;

        if (!WoodburyKKTData::isCompatible(
                n, m, P_ro.data(), P_ci.data(), nnzP,
                A_ro.data(), A_ci.data(), nnzA, cones)) {
            skipped++; continue;
        }

        // Feasible q, b
        std::vector<double> x_mid(n);
        for (auto& v : x_mid) v = std::uniform_real_distribution<double>(0.1, 1.0)(rng);
        std::vector<double> q(n), b(m);
        for (auto& v : q) v = nd(rng);
        for (int64_t i = 0; i < m; ++i) {
            double Ax_i = 0;
            for (int64_t idx = A_ro[i]; idx < A_ro[i + 1]; ++idx)
                Ax_i += A_vals[idx] * x_mid[A_ci[idx]];
            b[i] = (i < k) ? Ax_i : Ax_i + std::uniform_real_distribution<double>(0.5, 2.0)(rng);
        }

        auto r = solveWithBoth(n, m, 1, P_ro, P_ci, nnzP, A_ro, A_ci, nnzA,
                               P_vals, A_vals, q, b, cones);

        bool wb_ok = (r.status_wb == 1 || r.status_wb == 4);
        bool cd_ok = (r.status_cudss == 1 || r.status_cudss == 4);

        if (wb_ok && cd_ok) {
            both_solved++;
            EXPECT_NEAR(r.obj_wb, r.obj_cudss,
                        std::max(1e-4, std::abs(r.obj_cudss) * 1e-3))
                << "Trial " << trial << " n=2048 objective mismatch";
        } else if (!wb_ok && cd_ok) {
            cudss_only++;
            ADD_FAILURE() << "Trial " << trial
                << " n=2048: Woodbury status=" << r.status_wb << " but CuDSS solved";
        }
    }

    std::cout << "\nLarge-n fuzz results (" << NUM_INSTANCES << " trials, "
              << skipped << " skipped):\n"
              << "  Both solved: " << both_solved
              << "  CuDSS only: " << cudss_only << std::endl;

    EXPECT_EQ(cudss_only, 0);
}

// Diagnostic (DISABLED): localize the sparse-P backward bug to a single block.
//
// Both DiffWoodbury and DiffKKT compute sol = J (J'J + εI)^{-1} rhs from the
// SAME state.rhs. DiffKKT solves the augmented system K=[[I,J],[J',-εI]] with
// RHS [0; rhs]; its work_sol_ SECOND half is the augmented w = -(J'J+εI)^{-1}rhs,
// which maps block-for-block onto DiffWoodbury's internal (y_x, y_z, y_s, y_tau):
//   y_x  == -w[0:n], y_z == -w[n:n+m], y_s == -w[n+m:n+2m], y_tau == -w[n+2m].
// The first half is lam == DiffWoodbury's y_out. So this isolates the reduced
// solve (steps 1-3, the w blocks) from the final J-matvec (step 4, y_out).
// Run: --gtest_also_run_disabled_tests --gtest_filter=*DiagnoseBackwardBlocks*
TEST_F(WoodburyTest, DISABLED_DiagnoseBackwardBlocks) {
    auto D2H = [](std::vector<double>& dst, const double* src, int64_t len) {
        cudaMemcpy(dst.data(), src, sizeof(double) * len, cudaMemcpyDeviceToHost);
    };
    auto blkdiff = [](const std::vector<double>& a, const std::vector<double>& w,
                      int64_t off, int64_t len) {
        double mx = 0.0;
        for (int64_t i = 0; i < len; ++i)
            mx = std::max(mx, std::abs(a[i] - (-w[off + i])));  // y == -w
        return mx;
    };
    auto vmaxdiff = [](const std::vector<double>& a, const std::vector<double>& b) {
        double mx = 0.0; for (size_t i = 0; i < a.size(); ++i) mx = std::max(mx, std::abs(a[i]-b[i])); return mx;
    };

    // Run the block comparison on one problem (raw arrays so both FuzzProblem
    // and PortfolioProblem can feed it). Prints per-block (y vs -w_cd) diffs.
    auto diagnose = [&](const std::string& label,
                        int64_t n, int64_t m,
                        const std::vector<int64_t>& P_ro, const std::vector<int64_t>& P_ci, int64_t nnzP,
                        const std::vector<int64_t>& A_ro, const std::vector<int64_t>& A_ci, int64_t nnzA,
                        const std::vector<double>& P_vals, const std::vector<double>& A_vals,
                        const std::vector<double>& q, const std::vector<double>& b,
                        const Cones& cones) {
        const int64_t jdim = n + 2 * m + 1;
        if (!WoodburyKKTData::isCompatible(n, m, P_ro.data(), P_ci.data(), nnzP,
                                           A_ro.data(), A_ci.data(), nnzA, cones)) {
            std::cout << label << ": skipped (incompatible)\n"; return;
        }
        auto setup_dev = [&](double*& dP, double*& dA, double*& dq, double*& db) {
            cudaMalloc(&dP, sizeof(double) * nnzP); cudaMalloc(&dA, sizeof(double) * nnzA);
            cudaMalloc(&dq, sizeof(double) * n);    cudaMalloc(&db, sizeof(double) * m);
            cudaMemcpy(dP, P_vals.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
            cudaMemcpy(dA, A_vals.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
            cudaMemcpy(dq, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(db, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
        };
        auto make_settings = [&](KKTSolverType t) {
            Settings s; s.ipm.kktSolverType = t; s.verbose = false; s.enableGrad = true; return s;
        };
        auto do_backward = [&](CompiledSolver& solver, double* dP, double* dA, double* dq, double* db) {
            solver.solveAll(dP, dA, dq, db);
            cudaDeviceSynchronize();
            int32_t status;
            cudaMemcpy(&status, solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
            if (status != 1 && status != 4) return false;
            BatchedVector dx_bar(n, 1), dz_bar(m, 1), ds_bar(m, 1);
            std::vector<double> ones(n, 1.0);
            cudaMemcpy(dx_bar.data(), ones.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
            dz_bar.setToConstant(0.0); ds_bar.setToConstant(0.0);
            backward(*solver.diff_state_, dx_bar, dz_bar, ds_bar, solver, 0);
            cudaDeviceSynchronize();
            return true;
        };

        std::vector<double> rhs_wb(jdim), yout_wb(jdim), yx(n), yz(m), ys(m), ytau(1);
        // Equilibrated problem data (to rebuild the true dense J host-side).
        std::vector<double> Pdiag(n, 0.0), Aden(m * n, 0.0), Hfull(m, 0.0);
        std::vector<double> qv(n), bv(m), c1v(n), c2v(m), c3v(1);
        // DiffWoodbury's assembled reduced operator pieces.
        int64_t kext = 0;
        std::vector<double> Dxv, Gmat, Fext, Bfull, Dzv, rhov(1), sigv(1), vzv, vnuv;
        const int64_t k = cones.numZeroCones, nnn = cones.numNonnegCones;
        {
            CompiledSolver solver(n, m, 1, P_ro.data(), P_ci.data(), nnzP,
                A_ro.data(), A_ci.data(), nnzA, cones, make_settings(KKTSolverType::Woodbury));
            double *dP, *dA, *dq, *db; setup_dev(dP, dA, dq, db);
            bool ok = do_backward(solver, dP, dA, dq, db);
            if (!ok) { std::cout << label << ": WB solve failed\n";
                       cudaFree(dP);cudaFree(dA);cudaFree(dq);cudaFree(db); return; }
            D2H(rhs_wb, solver.diff_state_->rhs.data(), jdim);
            D2H(yout_wb, solver.diff_state_->sol.data(), jdim);
            D2H(yx, solver.diff_woodbury_->y_x_.get(), n);
            D2H(yz, solver.diff_woodbury_->y_z_.get(), m);
            D2H(ys, solver.diff_woodbury_->y_s_.get(), m);
            D2H(ytau, solver.diff_woodbury_->y_tau_.get(), 1);

            // --- Capture DiffWoodbury's assembled reduced operator pieces ---
            kext = solver.diff_woodbury_->k_ext_;
            Dxv.resize(n); Gmat.resize(n*kext); Fext.resize(n*kext); Bfull.resize(kext*kext);
            D2H(Dxv, solver.diff_woodbury_->D_x_.get(), n);
            D2H(Gmat, solver.diff_woodbury_->G_.get(), n*kext);
            D2H(Fext, solver.diff_woodbury_->F_ext_.get(), n*kext);
            D2H(Bfull, solver.diff_woodbury_->B_full_.get(), kext*kext);
            Dzv.resize(nnn); vzv.resize(m); vnuv.resize(kext);
            if (nnn>0) { D2H(Dzv, solver.diff_woodbury_->D_z_.get(), nnn);
                         D2H(rhov, solver.diff_woodbury_->rho_.get(), 1); }
            D2H(vzv, solver.diff_woodbury_->v_z_.get(), m);
            D2H(vnuv, solver.diff_woodbury_->v_nu_.get(), kext);
            D2H(sigv, solver.diff_woodbury_->sigma_.get(), 1);

            // --- Capture equilibrated P (diagonal), A (dense), q, b, c1/c2/c3, H ---
            const CSR& Peq = solver.data.P; const CSR& Aeq = solver.data.A;
            std::vector<int64_t> pro(n + 1), pci(Peq.nnz()), aro(m + 1), aci(Aeq.nnz());
            std::vector<double> pval(Peq.nnz()), aval(Aeq.nnz());
            cudaMemcpy(pro.data(), Peq.rowOffsets(), sizeof(int64_t)*(n+1), cudaMemcpyDeviceToHost);
            if (Peq.nnz()) { cudaMemcpy(pci.data(), Peq.colIndices(), sizeof(int64_t)*Peq.nnz(), cudaMemcpyDeviceToHost);
                             cudaMemcpy(pval.data(), Peq.values(), sizeof(double)*Peq.nnz(), cudaMemcpyDeviceToHost); }
            cudaMemcpy(aro.data(), Aeq.rowOffsets(), sizeof(int64_t)*(m+1), cudaMemcpyDeviceToHost);
            cudaMemcpy(aci.data(), Aeq.colIndices(), sizeof(int64_t)*Aeq.nnz(), cudaMemcpyDeviceToHost);
            cudaMemcpy(aval.data(), Aeq.values(), sizeof(double)*Aeq.nnz(), cudaMemcpyDeviceToHost);
            for (int64_t r = 0; r < n; ++r)
                for (int64_t p = pro[r]; p < pro[r+1]; ++p) if (pci[p] == r) Pdiag[r] = pval[p];
            for (int64_t r = 0; r < m; ++r)
                for (int64_t p = aro[r]; p < aro[r+1]; ++p) Aden[r*n + aci[p]] = aval[p];
            D2H(qv, solver.data.q.data(), n);
            D2H(bv, solver.data.b.data(), m);
            D2H(c1v, solver.diff_state_->c1.data(), n);
            D2H(c2v, solver.diff_state_->c2.data(), m);
            D2H(c3v, solver.diff_state_->c3.data(), 1);
            // Recover the full m-vector H from DiffWoodbury's PERSISTENT s-elim
            // diagonals: gh1 = g*(1+h), g = 1/(1+h²+ε) ⇒ h = gh1/g − 1.
            // (H_nonneg_ borrows derivs, which is freed when backward returns.)
            std::vector<double> gbuf(m), gh1buf(m);
            D2H(gbuf, solver.diff_woodbury_->g_.get(), m);
            D2H(gh1buf, solver.diff_woodbury_->gh1_.get(), m);
            for (int64_t u = 0; u < m; ++u) Hfull[u] = gh1buf[u] / gbuf[u] - 1.0;
            cudaFree(dP);cudaFree(dA);cudaFree(dq);cudaFree(db);
        }

        std::vector<double> rhs_cd(jdim), lam_cd(jdim), w_cd;
        {
            CompiledSolver solver(n, m, 1, P_ro.data(), P_ci.data(), nnzP,
                A_ro.data(), A_ci.data(), nnzA, cones, make_settings(KKTSolverType::CuDSS));
            double *dP, *dA, *dq, *db; setup_dev(dP, dA, dq, db);
            bool ok = do_backward(solver, dP, dA, dq, db);
            if (!ok) { std::cout << label << ": CD solve failed\n";
                       cudaFree(dP);cudaFree(dA);cudaFree(dq);cudaFree(db); return; }
            const int64_t jdim_cd = solver.diff_kkt_->jdim;
            std::vector<double> work_sol(2 * jdim_cd);
            D2H(work_sol, solver.diff_kkt_->work_sol_.data(), 2 * jdim_cd);
            w_cd.assign(work_sol.begin() + jdim_cd, work_sol.begin() + jdim_cd + jdim);
            D2H(lam_cd, solver.diff_state_->sol.data(), jdim);
            D2H(rhs_cd, solver.diff_state_->rhs.data(), jdim);
            cudaFree(dP);cudaFree(dA);cudaFree(dq);cudaFree(db);
        }

        std::cout << std::scientific << std::setprecision(3)
            << label << " (n=" << n << " m=" << m << ")\n"
            << "  rhs match (sanity)       : " << vmaxdiff(rhs_wb, rhs_cd) << "\n"
            << "  y_out vs lam (end-to-end): " << vmaxdiff(yout_wb, lam_cd) << "\n"
            << "  --- w blocks (y vs -w_cd) ---\n"
            << "  y_x  : " << blkdiff(yx,   w_cd, 0,       n) << "\n"
            << "  y_z  : " << blkdiff(yz,   w_cd, n,       m) << "\n"
            << "  y_s  : " << blkdiff(ys,   w_cd, n + m,   m) << "\n"
            << "  y_tau: " << blkdiff(ytau, w_cd, n + 2*m, 1) << "\n";

        // ---- Problem characterization (what makes an instance near-singular) ----
        auto vmax = [](const std::vector<double>& v){ double mx=0; for(double x:v) mx=std::max(mx,std::abs(x)); return mx; };
        int64_t pcols = 0; double pmin = 1e300, pmax = 0;
        for (int64_t j=0;j<n;++j) if (Pdiag[j]!=0.0){ pcols++; pmin=std::min(pmin,std::abs(Pdiag[j])); pmax=std::max(pmax,std::abs(Pdiag[j])); }
        int64_t kd_c=0, n_sparse_c=0; std::vector<int> nsc(n,0);
        for (int64_t r=k;r<m;++r){ int nz=0,col=-1; for(int64_t j=0;j<n;++j) if(Aden[r*n+j]!=0.0){nz++;col=j;}
                                   if(nz>1) kd_c++; else if(nz==1){ n_sparse_c++; nsc[col]++; } }
        int64_t multibound=0, uncovered=0;
        for (int64_t j=0;j<n;++j){ if(nsc[j]>1) multibound++; if(Pdiag[j]==0.0 && nsc[j]==0) uncovered++; }
        double wtrue = vmax(w_cd), rmax = vmax(rhs_wb);
        std::cout << "  PROBLEM: zero=" << k << " dense_nn=" << kd_c << " sparse_nn=" << n_sparse_c
                  << " multibound_cols=" << multibound << " uncovered_cols=" << uncovered << "\n"
                  << "    P: " << pcols << "/" << n << " cols nonzero, |P_eq| in ["
                  << (pcols?pmin:0.0) << ", " << pmax << "]\n"
                  << "    ||w_true||=" << wtrue << " ||rhs||=" << rmax
                  << "  ~1/lambda_min = " << wtrue/(rmax+1e-300) << "\n"
                  << "    gap: c3=" << c3v[0] << " max|c1|=" << vmax(c1v) << " max|c2|=" << vmax(c2v) << "\n";

        // ---- Host-side TRUE dense J / M = J'J + εI (noise-free reference) ----
        // J columns: [x(n) | z1(m) | z2(m) | tau(1)]; rows: [stat(n) | prim(m) |
        // comp(m) | gap(1)]. Mirrors diff_kkt.cu's J structure for zero+nonneg.
        const double EPS = 1e-8;
        std::vector<double> J(jdim * jdim, 0.0);
        auto Jat = [&](int64_t r, int64_t c) -> double& { return J[r * jdim + c]; };
        const int64_t Cx = 0, Cz1 = n, Cz2 = n + m, Ctau = n + 2*m;
        const int64_t Rstat = 0, Rprim = n, Rcomp = n + m, Rgap = n + 2*m;
        for (int64_t i = 0; i < n; ++i) {            // stationarity rows
            Jat(Rstat+i, Cx+i)  = Pdiag[i];          // P (diagonal)
            for (int64_t w = 0; w < m; ++w) Jat(Rstat+i, Cz1+w) = Aden[w*n + i];  // A'
            Jat(Rstat+i, Ctau) = qv[i];              // q
        }
        for (int64_t w = 0; w < m; ++w) {            // primal-feas rows
            for (int64_t j = 0; j < n; ++j) Jat(Rprim+w, Cx+j) = Aden[w*n + j];   // A
            Jat(Rprim+w, Cz1+w) = 1.0;               // I
            Jat(Rprim+w, Cz2+w) = -1.0;              // -I
            Jat(Rprim+w, Ctau)  = -bv[w];            // -b
        }
        for (int64_t u = 0; u < m; ++u) {            // complementarity rows
            Jat(Rcomp+u, Cz1+u) = 1.0;               // I
            Jat(Rcomp+u, Cz2+u) = -Hfull[u];         // -H
        }
        for (int64_t i = 0; i < n; ++i) Jat(Rgap, Cx+i)  = c1v[i];   // gap row
        for (int64_t w = 0; w < m; ++w) Jat(Rgap, Cz1+w) = c2v[w];
        Jat(Rgap, Ctau) = c3v[0];

        std::vector<double> M(jdim * jdim, 0.0);     // M = J'J + εI
        for (int64_t i = 0; i < jdim; ++i)
            for (int64_t j = 0; j < jdim; ++j) {
                double s = 0.0;
                for (int64_t r = 0; r < jdim; ++r) s += J[r*jdim+i] * J[r*jdim+j];
                M[i*jdim+j] = s + (i == j ? EPS : 0.0);
            }

        // w_wb concatenated [yx; yz; ys; ytau].
        std::vector<double> w_wb(jdim);
        for (int64_t i=0;i<n;++i) w_wb[Cx+i]=yx[i];
        for (int64_t i=0;i<m;++i) w_wb[Cz1+i]=yz[i];
        for (int64_t i=0;i<m;++i) w_wb[Cz2+i]=ys[i];
        w_wb[Ctau]=ytau[0];

        // Cholesky solve M w_true = rhs_wb (M is SPD).
        std::vector<double> L(jdim*jdim, 0.0), w_true(rhs_wb);
        for (int64_t i=0;i<jdim;++i)
            for (int64_t j=0;j<=i;++j) {
                double s = M[i*jdim+j];
                for (int64_t kk=0;kk<j;++kk) s -= L[i*jdim+kk]*L[j*jdim+kk];
                if (i==j) L[i*jdim+j] = std::sqrt(s);
                else      L[i*jdim+j] = s / L[j*jdim+j];
            }
        for (int64_t i=0;i<jdim;++i){ double s=w_true[i]; for(int64_t kk=0;kk<i;++kk) s-=L[i*jdim+kk]*w_true[kk]; w_true[i]=s/L[i*jdim+i]; }
        for (int64_t i=jdim-1;i>=0;--i){ double s=w_true[i]; for(int64_t kk=i+1;kk<jdim;++kk) s-=L[kk*jdim+i]*w_true[kk]; w_true[i]=s/L[i*jdim+i]; }

        // residual = M w_wb - rhs_wb (is DiffWoodbury solving THIS operator?)
        std::vector<double> resid(jdim);
        double res_inf = 0.0, wtrue_diff = 0.0;
        for (int64_t i=0;i<jdim;++i){
            double Mw=0.0; for(int64_t j=0;j<jdim;++j) Mw += M[i*jdim+j]*w_wb[j];
            resid[i] = Mw - rhs_wb[i];
            res_inf = std::max(res_inf, std::abs(resid[i]));
            wtrue_diff = std::max(wtrue_diff, std::abs(w_true[i] - w_wb[i]));
        }
        // Localize residual: per-block maxima + per-x-column (P=0?, #sparse rows).
        auto blkmax = [&](int64_t off, int64_t len){ double mx=0; for(int64_t i=0;i<len;++i) mx=std::max(mx,std::abs(resid[off+i])); return mx; };
        // count sparse nonneg rows (row>=k, exactly one nonzero) touching each col
        std::vector<int> nsparse_col(n, 0);
        for (int64_t r=k;r<m;++r){ int nz=0,col=-1; for(int64_t j=0;j<n;++j) if(Aden[r*n+j]!=0.0){nz++;col=j;} if(nz==1) nsparse_col[col]++; }
        std::cout << "  resid by block: x=" << blkmax(0,n) << " z1=" << blkmax(n,m)
                  << " z2=" << blkmax(n+m,m) << " tau=" << blkmax(n+2*m,1) << "\n";
        std::cout << "  x-col residuals (j: Pdiag, #sparse, |r_x|):\n";
        for (int64_t j=0;j<n;++j)
            std::cout << "    " << j << ": P=" << Pdiag[j] << " ns=" << nsparse_col[j]
                      << " |r|=" << std::abs(resid[j]) << "\n";

        // ---- Reduced-operator localization (only when k_d==0: ν = zero cones) ----
        // Build the TRUE reduced operator R_true over (x, ν=zero-cones, tau) by
        // Schur-eliminating z2 (all m) and sparse-λ (all nonneg z1) from M.
        // Compare to DiffWoodbury's assembled R_wb = [[diag(Dx)+GG', Fext],[Fext', Bfull]].
        // NOTE: since the gap-row Woodbury fix, the captured Dx/G/Fext/Bfull reflect
        // the BASE operator B (gap row excluded), so R_wb differs from R_true by the
        // gap rank-1 gg' by construction — the authoritative check is the solve
        // residual ‖M·w_wb − rhs‖ below, which uses the gap-corrected solution.
        // Split nonneg rows into dense (ν, kept) and sparse (λ, eliminated), in
        // m-index order (matches the F_all / B_full ν column order: zero cones
        // then dense nonneg).
        std::vector<int64_t> dense_rows, sparse_rows;  // m-indices
        for (int64_t r=k;r<m;++r){ int nz=0; for(int64_t j=0;j<n;++j) if(Aden[r*n+j]!=0.0) nz++;
                                   (nz>1 ? dense_rows : sparse_rows).push_back(r); }
        const int64_t k_d = (int64_t)dense_rows.size();
        if (k + k_d == kext - 1) {
            const int64_t rd = n + kext;          // reduced dimension
            // The captured Dx/G/Fext/Bfull assemble the BASE operator B = M − gg'
            // (gap row excluded). Compare against the reduced B, not reduced M.
            std::vector<double> g(jdim, 0.0);
            for (int64_t i=0;i<n;++i) g[i] = c1v[i];
            for (int64_t i=0;i<m;++i) g[n+i] = c2v[i];   // z2 block stays 0
            g[n+2*m] = c3v[0];
            std::vector<double> Bmat(jdim*jdim);
            for (int64_t i=0;i<jdim;++i) for (int64_t j=0;j<jdim;++j) Bmat[i*jdim+j] = M[i*jdim+j] - g[i]*g[j];
            // K: x 0..n-1, ν = zero cones then dense nonneg rows, tau n+2m.
            std::vector<int64_t> K, E;
            for (int64_t j=0;j<n;++j) K.push_back(j);
            for (int64_t i=0;i<k;++i) K.push_back(n + i);
            for (int64_t r : dense_rows) K.push_back(n + r);
            K.push_back(n + 2*m);
            for (int64_t i=0;i<m;++i) E.push_back(n + m + i);   // z2
            for (int64_t r : sparse_rows) E.push_back(n + r);   // sparse λ
            const int64_t ne = (int64_t)E.size();
            auto Mat = [&](const std::vector<int64_t>& R, const std::vector<int64_t>& C, int64_t i, int64_t j){ return Bmat[R[i]*jdim + C[j]]; };
            // Mee^{-1} Mek via Cholesky on Mee (SPD principal submatrix).
            std::vector<double> Lee(ne*ne, 0.0);
            for (int64_t i=0;i<ne;++i) for (int64_t j=0;j<=i;++j){ double s=Mat(E,E,i,j); for(int64_t t=0;t<j;++t) s-=Lee[i*ne+t]*Lee[j*ne+t]; Lee[i*ne+j]=(i==j)?std::sqrt(s):s/Lee[j*ne+j]; }
            // X = Mee^{-1} Mek  (ne x rd)
            std::vector<double> X(ne*rd);
            for (int64_t c=0;c<rd;++c){
                std::vector<double> y(ne);
                for (int64_t i=0;i<ne;++i){ double s=Mat(E,K,i,c); for(int64_t t=0;t<i;++t) s-=Lee[i*ne+t]*y[t]; y[i]=s/Lee[i*ne+i]; }
                for (int64_t i=ne-1;i>=0;--i){ double s=y[i]; for(int64_t t=i+1;t<ne;++t) s-=Lee[t*ne+i]*X[t*rd+c]; X[i*rd+c]=s/Lee[i*ne+i]; }
            }
            std::vector<double> Rtrue(rd*rd);
            for (int64_t i=0;i<rd;++i) for (int64_t j=0;j<rd;++j){ double s=Mat(K,K,i,j); for(int64_t t=0;t<ne;++t) s-=Mat(K,E,i,t)*X[t*rd+j]; Rtrue[i*rd+j]=s; }
            // R_wb: [[diag(Dx)+GG' , Fext],[Fext', Bfull]]
            std::vector<double> Rwb(rd*rd, 0.0);
            for (int64_t i=0;i<n;++i) for (int64_t j=0;j<n;++j){ double s=(i==j?Dxv[i]:0.0); for(int64_t t=0;t<kext;++t) s+=Gmat[i*kext+t]*Gmat[j*kext+t]; Rwb[i*rd+j]=s; }
            for (int64_t i=0;i<n;++i) for (int64_t t=0;t<kext;++t){ Rwb[i*rd+(n+t)] = Fext[i*kext+t]; Rwb[(n+t)*rd+i] = Fext[i*kext+t]; }
            for (int64_t a=0;a<kext;++a) for (int64_t bb=0;bb<kext;++bb) Rwb[(n+a)*rd+(n+bb)] = Bfull[a*kext+bb];
            auto bmax = [&](int64_t r0,int64_t r1,int64_t c0,int64_t c1){ double mx=0; for(int64_t i=r0;i<r1;++i) for(int64_t j=c0;j<c1;++j) mx=std::max(mx,std::abs(Rwb[i*rd+j]-Rtrue[i*rd+j])); return mx; };
            std::cout << "  --- base operator R_wb (assembly) vs R_base (true B) (rd=" << rd << ") ---\n"
                      << "    xx   : " << bmax(0,n,0,n)
                      << "  x-nu/tau: " << bmax(0,n,n,rd)
                      << "  nu/tau block: " << bmax(n,rd,n,rd) << "\n";
            std::cout << std::fixed << std::setprecision(5);
            std::cout << "  c2(m): "; for (double v: c2v) std::cout<<v<<" "; std::cout<<" c3="<<c3v[0]<<"\n";
            std::cout << "  Hfull(m): "; for (double v: Hfull) std::cout<<v<<" "; std::cout<<"\n";
            std::cout << "  D_z: "; for (double v: Dzv) std::cout<<v<<" "; std::cout<<" rho="<<rhov[0]<<" sigma="<<sigv[0]<<"\n";
            std::cout << "  v_z(m): "; for (double v: vzv) std::cout<<v<<" "; std::cout<<"\n";
            std::cout << "  v_nu(kext): "; for (double v: vnuv) std::cout<<v<<" "; std::cout<<"\n";
            for (int64_t i=0;i<rd;++i){ std::cout << "    Rwb["<<i<<"]:";
                for (int64_t j=0;j<rd;++j) std::cout << " " << Rwb[i*rd+j];
                std::cout << "   Rtrue:";
                for (int64_t j=0;j<rd;++j) std::cout << " " << Rtrue[i*rd+j];
                std::cout << "\n"; }
            std::cout << std::scientific << std::setprecision(3);
        }
        // y_out_true = J w_true; vs DiffWoodbury y_out and vs cuDSS lam
        std::vector<double> yout_true(jdim, 0.0);
        for (int64_t r=0;r<jdim;++r){ double s=0; for(int64_t c=0;c<jdim;++c) s+=J[r*jdim+c]*w_true[c]; yout_true[r]=s; }
        std::cout
            << "  --- host dense reference ---\n"
            << "  ||M*w_wb - rhs||_inf (solve resid) : " << res_inf << "\n"
            << "  ||w_true - w_wb||_inf              : " << wtrue_diff << "\n"
            << "  ||yout_true - yout_wb||_inf        : " << vmaxdiff(yout_true, yout_wb) << "\n"
            << "  ||yout_true - lam_cd||_inf (J check): " << vmaxdiff(yout_true, lam_cd) << "\n";
    };

    // CONTROL: well-conditioned portfolio (full diagonal P, sparse -I nonneg
    // rows). This passes BackwardGradientsMatchFiniteDiff, so its blocks MUST
    // match — validates the oracle/sign/mapping before trusting the fuzz cases.
    {
        auto p = buildPortfolio(10, 2, 123);
        diagnose("CONTROL portfolio", p.n, p.m, p.P_ro, p.P_ci, p.nnzP,
                 p.A_ro, p.A_ci, p.nnzA, p.P_values, p.A_values, p.q_data, p.b_data, p.cones);
    }

    // DECISIVE CONTROLS: isolate "multi-sparse-row-per-column λ coupling" from
    // "small P magnitude". Both use tiny P=1e-3 (so D_x≈P² can't mask the bug).
    //   C1: ONE sparse nonneg row per column (no λ-λ coupling) → expect CORRECT.
    //   C2: TWO sparse rows on each column (lower+upper) → expect WRONG.
    // If C1 is clean and C2 is broken, the bug is the omitted a_i·a_i' coupling.
    {
        std::vector<int64_t> Pro{0,1,2}, Pci{0,1};
        std::vector<double>  Pv{1e-3,1e-3};
        // C1: zero cone x0+x1=1 ; then x0<=1, x1<=1  (one sparse row/col)
        Cones cn1; cn1.numZeroCones = 1; cn1.numNonnegCones = 2;
        std::vector<int64_t> Aro1{0,2,3,4}, Aci1{0,1, 0, 1};
        std::vector<double>  Av1{1.0,1.0, 1.0, 1.0}, q1{-0.5,-0.5}, b1{1.0, 1.0,1.0};
        diagnose("C1 small-P 1/col", 2, 3, Pro, Pci, 2, Aro1, Aci1, 4, Pv, Av1, q1, b1, cn1);

        // C2: zero cone x0+x1=1 ; then x0<=1,x0>=-1,x1<=1,x1>=-1 (two rows/col)
        Cones cn2; cn2.numZeroCones = 1; cn2.numNonnegCones = 4;
        std::vector<int64_t> Aro2{0,2,3,4,5,6}, Aci2{0,1, 0, 0, 1, 1};
        std::vector<double>  Av2{1.0,1.0, 1.0,-1.0, 1.0,-1.0}, q2{-0.5,-0.5}, b2{1.0, 1.0,1.0, 1.0,1.0};
        diagnose("C2 small-P 2/col", 2, 5, Pro, Pci, 2, Aro2, Aci2, 6, Pv, Av2, q2, b2, cn2);

        // Same structures with LARGE P=1.0 — factor P-magnitude from sparse-count.
        std::vector<double> PvL{1.0,1.0};
        diagnose("C1 large-P 1/col", 2, 3, Pro, Pci, 2, Aro1, Aci1, 4, PvL, Av1, q1, b1, cn1);
        diagnose("C2 large-P 2/col", 2, 5, Pro, Pci, 2, Aro2, Aci2, 6, PvL, Av2, q2, b2, cn2);
    }

    // FUZZ: sparse-P cases that fail the backward gradient check.
    const unsigned BASE_SEED = 99999;
    for (int trial : {2, 0, 1, 5, 6, 7, 8}) {
        std::mt19937 rng(BASE_SEED + trial);
        auto prob = generateFuzzProblem(rng);
        diagnose("trial " + std::to_string(trial), prob.n, prob.m,
                 prob.P_ro, prob.P_ci, prob.nnzP, prob.A_ro, prob.A_ci, prob.nnzA,
                 prob.P_vals, prob.A_vals, prob.q, prob.b, prob.cones);
    }
}

// Backward sparse-P fuzz (ported from PR #103). Fills the gap that the current
// WoodburyDiffTest suite leaves: those cover only well-conditioned P in [0.1,2.0].
// This drives the sparse/zero-P backward path, the class that historically broke
// DiffWoodbury's backward. Compares WB gradients to cuDSS; the WB backward now
// (a) builds the correct operator (gap-row global Woodbury + per-column SM for
// same-column nonneg bounds) and (b) falls back to cuDSS on the rare near-singular
// instances where the κ² normal-equation chain loses accuracy. Instances whose two
// forward solves disagree (degenerate primal) are skipped — the gradient compare
// is ill-posed there, not a backward bug.
TEST_F(WoodburyTest, FuzzBackwardSparseP) {
    // Test backward pass on sparse-P problems (the same class that broke forward).
    // Compare Woodbury backward gradients to CuDSS backward gradients.
    const int NUM_INSTANCES = 20;
    const unsigned BASE_SEED = 99999;

    int pass = 0, fail = 0, skipped = 0;

    for (int trial = 0; trial < NUM_INSTANCES; ++trial) {
        std::mt19937 rng(BASE_SEED + trial);
        auto prob = generateFuzzProblem(rng);

        bool compat = WoodburyKKTData::isCompatible(
            prob.n, prob.m,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones);
        if (!compat) { skipped++; continue; }

        auto run_backward = [&](KKTSolverType kkt_type) {
            Settings s;
            s.ipm.kktSolverType = kkt_type;
            s.verbose = false;
            s.enableGrad = true;

            CompiledSolver solver(
                prob.n, prob.m, 1,
                prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
                prob.cones, s);

            double *d_P, *d_A, *d_q, *d_b;
            cudaMalloc(&d_P, sizeof(double) * prob.nnzP);
            cudaMalloc(&d_A, sizeof(double) * prob.nnzA);
            cudaMalloc(&d_q, sizeof(double) * prob.n);
            cudaMalloc(&d_b, sizeof(double) * prob.m);
            cudaMemcpy(d_P, prob.P_vals.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
            cudaMemcpy(d_A, prob.A_vals.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
            cudaMemcpy(d_q, prob.q.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, prob.b.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);
            solver.solveAll(d_P, d_A, d_q, d_b);
            cudaDeviceSynchronize();

            // Check solve status
            int32_t status;
            cudaMemcpy(&status, solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
            if (status != 1 && status != 4) {
                cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
                return std::make_tuple(std::vector<double>(), std::vector<double>(),
                                       std::vector<double>(), std::vector<double>(),
                                       std::vector<double>(), false);
            }

            // Forward solution (to distinguish a backward bug from primal
            // non-uniqueness on these LP-like sparse-P problems).
            std::vector<double> x_sol(prob.n);
            cudaMemcpy(x_sol.data(), solver.solution.x.data(),
                       sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

            // Backward with dx_bar = 1
            BatchedVector dx_bar(prob.n, 1);
            BatchedVector dz_bar(prob.m, 1);
            BatchedVector ds_bar(prob.m, 1);
            std::vector<double> dx_ones(prob.n, 1.0);
            cudaMemcpy(dx_bar.data(), dx_ones.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
            dz_bar.setToConstant(0.0);
            ds_bar.setToConstant(0.0);

            backward(*solver.diff_state_, dx_bar, dz_bar, ds_bar, solver, 0);
            cudaDeviceSynchronize();

            std::vector<double> dq(prob.n), db(prob.m);
            std::vector<double> dP(prob.nnzP), dA(prob.nnzA);
            cudaMemcpy(dq.data(), solver.diff_state_->dq.data(), sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
            cudaMemcpy(db.data(), solver.diff_state_->db.data(), sizeof(double) * prob.m, cudaMemcpyDeviceToHost);
            cudaMemcpy(dP.data(), solver.diff_state_->dP_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyDeviceToHost);
            cudaMemcpy(dA.data(), solver.diff_state_->dA_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyDeviceToHost);

            cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
            return std::make_tuple(dq, db, dP, dA, x_sol, true);
        };

        auto [dq_wb, db_wb, dP_wb, dA_wb, x_wb, wb_ok] = run_backward(KKTSolverType::Woodbury);
        auto [dq_cd, db_cd, dP_cd, dA_cd, x_cd, cd_ok] = run_backward(KKTSolverType::CuDSS);

        if (!wb_ok || !cd_ok) { skipped++; continue; }

        auto max_diff = [](const std::vector<double>& a, const std::vector<double>& b) {
            double mx = 0.0;
            for (size_t i = 0; i < a.size(); ++i)
                mx = std::max(mx, std::abs(a[i] - b[i]));
            return mx;
        };
        // Relative gradient mismatch: ‖a−b‖_inf / (‖b‖_inf + 1). These gradients
        // span orders of magnitude (P from 1e-4 to 1e4), so a flat absolute
        // tolerance is wrong; a near-singular instance can carry a large-magnitude
        // gradient whose two solvers still agree to high relative accuracy.
        auto rel_diff = [&](const std::vector<double>& a, const std::vector<double>& b) {
            double nb = 0.0; for (double v : b) nb = std::max(nb, std::abs(v));
            return max_diff(a, b) / (nb + 1.0);
        };

        double dq_diff = rel_diff(dq_wb, dq_cd);
        double db_diff = rel_diff(db_wb, db_cd);
        double dP_diff = rel_diff(dP_wb, dP_cd);
        double dA_diff = rel_diff(dA_wb, dA_cd);

        double max_all = std::max({dq_diff, db_diff, dP_diff, dA_diff});
        if (max_all < 1e-3) {
            pass++;
        } else {
            fail++;
            ADD_FAILURE() << "Trial " << trial << " (n=" << prob.n
                << " k=" << prob.k << " k_d=" << prob.k_d
                << " seed=" << (BASE_SEED + trial)
                << "): backward gradient mismatch: dq=" << dq_diff
                << " db=" << db_diff << " dP=" << dP_diff << " dA=" << dA_diff
                << " | forward x_diff=" << max_diff(x_wb, x_cd);
        }
    }

    std::cout << "\nBackward sparse-P fuzz results (" << NUM_INSTANCES << " trials, "
              << skipped << " skipped):\n"
              << "  Pass: " << pass << "  Fail: " << fail << std::endl;

    EXPECT_EQ(fail, 0);
}
