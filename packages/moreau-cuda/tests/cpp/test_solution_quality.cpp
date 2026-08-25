// test_solution_quality.cpp
// Tests verifying solution quality via KKT residuals and constraint satisfaction
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>

using namespace moreau;

class SolutionQualityTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test that residuals are small at optimal solution
TEST_F(SolutionQualityTest, ResidualSmallAtOptimum) {
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
    settings.maxIter = 100;
    settings.verbose = false;
    settings.ipm.tolFeas = 1e-8;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values = {1.0, 1.0};
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

    std::vector<double> resPrimal(1), resDual(1);
    solver.info.res_primal.gpuToCpu(resPrimal.data());
    solver.info.res_dual.gpuToCpu(resDual.data());
    cudaDeviceSynchronize();

    EXPECT_LT(resPrimal[0], 1e-6) << "Primal residual should be small";
    EXPECT_LT(resDual[0], 1e-6) << "Dual residual should be small";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test constraint satisfaction for inequality constraints
TEST_F(SolutionQualityTest, InequalityConstraintSatisfied) {
    int n = 2, m = 2;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    // A = [[1, 0], [0, 1]] - constraints x1 <= b1, x2 <= b2
    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numNonnegCones = 2;

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // min x1 + x2  s.t. x1 <= 3, x2 <= 4
    std::vector<double> P_values = {0.01, 0.01};  // Small regularization
    std::vector<double> A_values = {1.0, 1.0};
    std::vector<double> q = {1.0, 1.0};
    std::vector<double> b = {3.0, 4.0};

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

    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    // Check constraints: Ax <= b (with small tolerance)
    EXPECT_LE(x[0], b[0] + 1e-4) << "x1 <= " << b[0] << " should be satisfied";
    EXPECT_LE(x[1], b[1] + 1e-4) << "x2 <= " << b[1] << " should be satisfied";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test constraint satisfaction for equality constraints
TEST_F(SolutionQualityTest, EqualityConstraintSatisfied) {
    int n = 3, m = 2;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;

    // A = [[1, 1, 0], [0, 1, 1]]
    std::vector<int64_t> A_ro = {0, 2, 4};
    std::vector<int64_t> A_ci = {0, 1, 1, 2};
    int64_t nnzA = 4;

    Cones cones{};
    cones.numZeroCones = 2;  // Both are equality constraints

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // min 0.5*(x1^2 + x2^2 + x3^2)  s.t. x1 + x2 = 3, x2 + x3 = 4
    std::vector<double> P_values = {1.0, 1.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> q = {0.0, 0.0, 0.0};
    std::vector<double> b = {3.0, 4.0};

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

    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    // Check equality constraints: Ax = b
    double constraint1 = x[0] + x[1];
    double constraint2 = x[1] + x[2];

    EXPECT_NEAR(constraint1, b[0], 1e-4) << "x1 + x2 = " << b[0] << " should be satisfied";
    EXPECT_NEAR(constraint2, b[1], 1e-4) << "x2 + x3 = " << b[1] << " should be satisfied";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test that slack variables are non-negative
TEST_F(SolutionQualityTest, SlackNonNegative) {
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
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values = {1.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0};
    std::vector<double> q = {-1.0, -1.0};  // Push x toward positive
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

    std::vector<double> s(m);
    solver.solution.s.gpuToCpu(s.data(), 0);
    cudaDeviceSynchronize();

    // Slack variables should be non-negative for nonneg cones
    for (int i = 0; i < m; i++) {
        EXPECT_GE(s[i], -1e-6) << "Slack s[" << i << "] should be non-negative";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
