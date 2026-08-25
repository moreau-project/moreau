// test_settings.cpp
// Tests for solver settings and configuration options

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "clarabel_test_helpers.hpp"
#include <vector>
#include <cmath>

using namespace moreau;
using namespace clarabel_test;

class SettingsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetLastError();
    }
    void TearDown() override {
        cudaDeviceSynchronize();
    }

    // Helper to create a simple feasible QP for testing settings
    void createSimpleQP(
        int64_t& n, int64_t& m,
        std::vector<int64_t>& P_ro, std::vector<int64_t>& P_ci, std::vector<double>& P_val,
        std::vector<int64_t>& A_ro, std::vector<int64_t>& A_ci, std::vector<double>& A_val,
        std::vector<double>& q, std::vector<double>& b
    ) {
        n = 2; m = 2;
        P_ro = {0, 1, 2};
        P_ci = {0, 1};
        P_val = {2.0, 2.0};
        A_ro = {0, 1, 2};
        A_ci = {0, 1};
        A_val = {-1.0, -1.0};
        q = {1.0, 2.0};
        b = {0.0, 0.0};
    }
};

// =============================================================================
// Default Settings Tests
// =============================================================================

TEST_F(SettingsTest, DefaultSettingsValues) {
    Settings settings;

    // Check default values match documentation
    EXPECT_EQ(settings.maxIter, 200);
    EXPECT_GT(settings.timeLimit, 0.0);
    EXPECT_TRUE(settings.verbose);

    // Tolerance defaults
    EXPECT_EQ(settings.ipm.tolGapAbs, 1e-8);
    EXPECT_EQ(settings.ipm.tolGapRel, 1e-8);
    EXPECT_EQ(settings.ipm.tolFeas, 1e-8);
    EXPECT_EQ(settings.ipm.tolInfeasAbs, 1e-8);
    EXPECT_EQ(settings.ipm.tolInfeasRel, 1e-8);

    // Step fraction
    EXPECT_GT(settings.ipm.maxStepFraction, 0.0);
    EXPECT_LE(settings.ipm.maxStepFraction, 1.0);
}

TEST_F(SettingsTest, DefaultSettingsSolves) {
    int64_t n, m;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_val, A_val, q, b;
    createSimpleQP(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b);

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;  // Use defaults

    CompiledSolver solver(n, m, 1,
                  P_ro.data(), P_ci.data(), P_val.size(),
                  A_ro.data(), A_ci.data(), A_val.size(),
                  cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * P_val.size());
    cudaMalloc(&d_A, sizeof(double) * A_val.size());
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(status[0], static_cast<int32_t>(SolverStatus::Solved));

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// =============================================================================
// Max Iterations Tests
// =============================================================================

TEST_F(SettingsTest, MaxIterationsOne) {
    int64_t n, m;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_val, A_val, q, b;
    createSimpleQP(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b);

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.maxIter = 1;  // Very few iterations

    CompiledSolver solver(n, m, 1,
                  P_ro.data(), P_ci.data(), P_val.size(),
                  A_ro.data(), A_ci.data(), A_val.size(),
                  cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * P_val.size());
    cudaMalloc(&d_A, sizeof(double) * A_val.size());
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Should hit max iterations or solve (if problem is trivial)
    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::MaxIterations) ||
                status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved));

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(SettingsTest, MaxIterationsLarge) {
    int64_t n, m;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_val, A_val, q, b;
    createSimpleQP(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b);

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.maxIter = 1000;  // Many iterations

    CompiledSolver solver(n, m, 1,
                  P_ro.data(), P_ci.data(), P_val.size(),
                  A_ro.data(), A_ci.data(), A_val.size(),
                  cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * P_val.size());
    cudaMalloc(&d_A, sizeof(double) * A_val.size());
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(status[0], static_cast<int32_t>(SolverStatus::Solved));

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// =============================================================================
// Tolerance Tests
// =============================================================================

TEST_F(SettingsTest, TightTolerances) {
    int64_t n, m;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_val, A_val, q, b;
    createSimpleQP(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b);

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.ipm.tolGapAbs = 1e-12;
    settings.ipm.tolGapRel = 1e-12;
    settings.ipm.tolFeas = 1e-12;
    settings.maxIter = 500;  // More iterations for tight tolerance

    CompiledSolver solver(n, m, 1,
                  P_ro.data(), P_ci.data(), P_val.size(),
                  A_ro.data(), A_ci.data(), A_val.size(),
                  cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * P_val.size());
    cudaMalloc(&d_A, sizeof(double) * A_val.size());
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Should eventually converge or hit max iter
    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved) ||
                status[0] == static_cast<int32_t>(SolverStatus::MaxIterations));

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(SettingsTest, LooseTolerances) {
    int64_t n, m;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_val, A_val, q, b;
    createSimpleQP(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b);

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.ipm.tolGapAbs = 1e-4;
    settings.ipm.tolGapRel = 1e-4;
    settings.ipm.tolFeas = 1e-4;
    settings.maxIter = 50;

    CompiledSolver solver(n, m, 1,
                  P_ro.data(), P_ci.data(), P_val.size(),
                  A_ro.data(), A_ci.data(), A_val.size(),
                  cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * P_val.size());
    cudaMalloc(&d_A, sizeof(double) * A_val.size());
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Should solve quickly with loose tolerances
    EXPECT_EQ(status[0], static_cast<int32_t>(SolverStatus::Solved));

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// =============================================================================
// Equilibration Settings Tests
// =============================================================================

TEST_F(SettingsTest, EquilibrationEnabled) {
    int64_t n, m;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_val, A_val, q, b;
    createSimpleQP(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b);

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(n, m, 1,
                  P_ro.data(), P_ci.data(), P_val.size(),
                  A_ro.data(), A_ci.data(), A_val.size(),
                  cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * P_val.size());
    cudaMalloc(&d_A, sizeof(double) * A_val.size());
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(status[0], static_cast<int32_t>(SolverStatus::Solved));

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(SettingsTest, EquilibrationDisabled) {
    int64_t n, m;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_val, A_val, q, b;
    createSimpleQP(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b);

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = false;

    CompiledSolver solver(n, m, 1,
                  P_ro.data(), P_ci.data(), P_val.size(),
                  A_ro.data(), A_ci.data(), A_val.size(),
                  cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * P_val.size());
    cudaMalloc(&d_A, sizeof(double) * A_val.size());
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Should still solve without equilibration
    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved));

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// =============================================================================
// Settings Consistency with Batched Solving
// =============================================================================

TEST_F(SettingsTest, BatchedSettingsConsistency) {
    int64_t n = 2, m = 2;
    int64_t batchSize = 4;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    std::vector<double> P_val(nnzP * batchSize, 2.0);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int64_t i = 0; i < batchSize; i++) {
        A_val[i * nnzA + 0] = -1.0;
        A_val[i * nnzA + 1] = -1.0;
        q[i * n + 0] = 1.0;
        q[i * n + 1] = 2.0;
        b[i * m + 0] = 0.0;
        b[i * m + 1] = 0.0;
    }

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.maxIter = 100;
    settings.ipm.tolGapAbs = 1e-8;
    settings.ipm.equilibrationSettings.enable = true;

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

    std::vector<int32_t> status(batchSize);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    // All batches should solve with same settings
    for (int64_t i = 0; i < batchSize; i++) {
        EXPECT_EQ(status[i], static_cast<int32_t>(SolverStatus::Solved))
            << "Batch " << i << " did not solve";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
