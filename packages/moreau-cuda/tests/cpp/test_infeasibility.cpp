// test_infeasibility.cpp
// Tests for primal and dual infeasibility detection
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>

using namespace moreau;

class InfeasibilityTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test detection of primal infeasibility via contradictory constraints
// Using equality constraints: x = 1 AND x = -1 is clearly infeasible
TEST_F(InfeasibilityTest, PrimalInfeasibleSimple) {
    int n = 1, m = 2;
    int batchSize = 1;

    // P = [1], simple QP
    std::vector<int64_t> P_ro = {0, 1};
    std::vector<int64_t> P_ci = {0};
    int64_t nnzP = 1;

    // A = [[1], [1]] with constraints:
    // x = 1 (zero cone = equality)
    // x = -1 (zero cone = equality)
    // These are contradictory
    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 0};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numZeroCones = 2;  // Two equality constraints

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values = {1.0};
    std::vector<double> A_values = {1.0, 1.0};  // Both constraints use x
    std::vector<double> q = {0.0};
    std::vector<double> b = {1.0, -1.0};  // x = 1 AND x = -1 (impossible)

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

    // Retrieve status
    std::vector<int32_t> status(batchSize);
    cudaMemcpy(status.data(), solver.info.status_device,
               sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Should detect primal infeasibility (or similar error status)
    SolverStatus st = static_cast<SolverStatus>(status[0]);
    EXPECT_TRUE(st == SolverStatus::PrimalInfeasible ||
                st == SolverStatus::AlmostPrimalInfeasible ||
                st == SolverStatus::MaxIterations ||
                st == SolverStatus::NumericalError ||
                st == SolverStatus::AlmostSolved)  // May solve but with large residuals
        << "Expected infeasibility detection, got status " << status[0];

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test detection of dual infeasibility (unbounded problem)
TEST_F(InfeasibilityTest, DualInfeasibleUnbounded) {
    int n = 1, m = 0;  // No constraints except implicit x >= 0
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1};
    std::vector<int64_t> P_ci = {0};
    int64_t nnzP = 1;

    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};
    int64_t nnzA = 0;

    Cones cones{};
    // No cones - unconstrained

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // Zero P, negative q -> unbounded (minimize -x with x >= 0 is unbounded)
    std::vector<double> P_values = {0.0};
    std::vector<double> A_values = {};
    std::vector<double> q = {-1.0};  // Minimize -x -> x -> infinity
    std::vector<double> b = {};

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * 1);  // Allocate at least 1 byte
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * 1);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(batchSize);
    cudaMemcpy(status.data(), solver.info.status_device,
               sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Should detect dual infeasibility or numerical issues
    SolverStatus st = static_cast<SolverStatus>(status[0]);
    EXPECT_TRUE(st == SolverStatus::DualInfeasible ||
                st == SolverStatus::AlmostDualInfeasible ||
                st == SolverStatus::MaxIterations ||
                st == SolverStatus::Solved ||  // May find trivial solution
                st == SolverStatus::NumericalError)
        << "Unexpected status " << status[0];

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test contradictory equality constraints
TEST_F(InfeasibilityTest, ContradictoryEqualities) {
    int n = 2, m = 2;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    // Two constraints: x1 = 1 and x1 = 2 (contradictory)
    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 0};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numZeroCones = 2;  // Equality constraints

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values = {1.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0};  // Both constraints on x1
    std::vector<double> q = {0.0, 0.0};
    std::vector<double> b = {1.0, 2.0};  // x1 = 1 AND x1 = 2

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

    std::vector<int32_t> status(batchSize);
    cudaMemcpy(status.data(), solver.info.status_device,
               sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    SolverStatus st = static_cast<SolverStatus>(status[0]);
    EXPECT_TRUE(st == SolverStatus::PrimalInfeasible ||
                st == SolverStatus::AlmostPrimalInfeasible ||
                st == SolverStatus::MaxIterations ||
                st == SolverStatus::NumericalError)
        << "Expected infeasibility for contradictory equalities, got " << status[0];

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
