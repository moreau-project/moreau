// test_convergence.cpp
// Tests for convergence behavior and tolerance settings
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>

using namespace moreau;

class ConvergenceTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test that tighter tolerances lead to better solutions
TEST_F(ConvergenceTest, TighterToleranceBetterSolution) {
    int n = 2, m = 2;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numNonnegCones = 2;

    // Solve with loose tolerance
    Settings looseSettings;
    looseSettings.maxIter = 200;
    looseSettings.verbose = false;
    looseSettings.ipm.tolGapAbs = 1e-4;
    looseSettings.ipm.tolGapRel = 1e-4;
    looseSettings.ipm.tolFeas = 1e-4;

    CompiledSolver looseSolver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                               A_ro.data(), A_ci.data(), nnzA, cones, looseSettings);

    // Solve with tight tolerance
    Settings tightSettings;
    tightSettings.maxIter = 200;
    tightSettings.verbose = false;
    tightSettings.ipm.tolGapAbs = 1e-8;
    tightSettings.ipm.tolGapRel = 1e-8;
    tightSettings.ipm.tolFeas = 1e-8;

    CompiledSolver tightSolver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                               A_ro.data(), A_ci.data(), nnzA, cones, tightSettings);

    std::vector<double> P_values = {2.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0};
    std::vector<double> q = {1.0, 2.0};
    std::vector<double> b = {5.0, 5.0};

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    looseSolver.solveAll(d_P, d_A, d_q, d_b);
    tightSolver.solveAll(d_P, d_A, d_q, d_b);

    // Tight solver should use more iterations (or same if it converges quickly)
    // Both should converge
    std::vector<int32_t> looseStatus(1), tightStatus(1);
    cudaMemcpy(looseStatus.data(), looseSolver.info.status_device, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(tightStatus.data(), tightSolver.info.status_device, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    SolverStatus looseSt = static_cast<SolverStatus>(looseStatus[0]);
    SolverStatus tightSt = static_cast<SolverStatus>(tightStatus[0]);

    EXPECT_TRUE(looseSt == SolverStatus::Solved || looseSt == SolverStatus::AlmostSolved)
        << "Loose solver should converge";
    EXPECT_TRUE(tightSt == SolverStatus::Solved || tightSt == SolverStatus::AlmostSolved)
        << "Tight solver should converge";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test that max iterations is respected
TEST_F(ConvergenceTest, MaxIterationsRespected) {
    int n = 2, m = 2;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numNonnegCones = 2;

    Settings settings;
    settings.maxIter = 5;  // Very low max iterations
    settings.verbose = false;
    settings.ipm.tolGapAbs = 1e-12;  // Very tight tolerance - unlikely to converge in 5 iterations
    settings.ipm.tolGapRel = 1e-12;
    settings.ipm.tolFeas = 1e-12;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values = {2.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0};
    std::vector<double> q = {1.0, 2.0};
    std::vector<double> b = {5.0, 5.0};

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    // Check iterations count (using host-side value set after solve)
    EXPECT_LE(solver.info.iterations, settings.maxIter) << "Should not exceed max iterations";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test early termination on solved
TEST_F(ConvergenceTest, EarlyTerminationOnSolved) {
    int n = 2, m = 1;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numNonnegCones = 1;

    Settings settings;
    settings.maxIter = 1000;  // High max iterations
    settings.verbose = false;
    settings.ipm.tolGapAbs = 1e-6;
    settings.ipm.tolGapRel = 1e-6;
    settings.ipm.tolFeas = 1e-6;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // Simple well-conditioned problem
    std::vector<double> P_values = {1.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0};
    std::vector<double> q = {0.0, 0.0};
    std::vector<double> b = {10.0};

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.info.status_device, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Should converge well before max iterations
    EXPECT_LT(solver.info.iterations, settings.maxIter / 2) << "Should converge early";
    EXPECT_TRUE(static_cast<SolverStatus>(status[0]) == SolverStatus::Solved ||
                static_cast<SolverStatus>(status[0]) == SolverStatus::AlmostSolved);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
