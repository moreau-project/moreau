// test_cone_varieties.cpp
// Tests for various cone type combinations
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>

using namespace moreau;

class ConeVarietiesTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test with only zero cones (equality constraints)
TEST_F(ConeVarietiesTest, OnlyZeroCones) {
    int n = 2, m = 1;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numZeroCones = 1;  // Equality constraint: Ax = b

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // min 0.5*(x1^2 + x2^2) + x1 + x2  s.t. x1 + x2 = 4
    std::vector<double> P_values = {1.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0};
    std::vector<double> q = {1.0, 1.0};
    std::vector<double> b = {4.0};

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

    // Optimal: x1 = x2 = 1.5 (symmetric solution with equality)
    EXPECT_NEAR(x[0] + x[1], 4.0, 1e-4) << "Equality constraint should be satisfied";
    EXPECT_NEAR(x[0], x[1], 1e-4) << "Symmetric problem should have symmetric solution";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test with only nonnegative cones (inequality constraints)
TEST_F(ConeVarietiesTest, OnlyNonnegCones) {
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

    // min 0.5*(x1^2 + x2^2)  s.t. x1 <= 1, x2 <= 1
    std::vector<double> P_values = {1.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0};
    std::vector<double> q = {0.0, 0.0};
    std::vector<double> b = {1.0, 1.0};

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

    // Optimal: x1 = x2 = 0 (minimize quadratic with no linear term)
    EXPECT_NEAR(x[0], 0.0, 1e-4);
    EXPECT_NEAR(x[1], 0.0, 1e-4);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test with mixed zero and nonnegative cones
TEST_F(ConeVarietiesTest, MixedZeroAndNonneg) {
    int n = 2, m = 2;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2, 3};
    std::vector<int64_t> A_ci = {0, 1, 0};
    int64_t nnzA = 3;

    Cones cones{};
    cones.numZeroCones = 1;     // First constraint: equality
    cones.numNonnegCones = 1;   // Second constraint: inequality

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // min 0.5*(x1^2 + x2^2)  s.t. x1 + x2 = 2, x1 <= 1
    std::vector<double> P_values = {1.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0, 1.0};
    std::vector<double> q = {0.0, 0.0};
    std::vector<double> b = {2.0, 1.0};

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

    // Optimal: x1 = 1, x2 = 1 (equality constraint binds, inequality at boundary)
    EXPECT_NEAR(x[0] + x[1], 2.0, 1e-4) << "Equality should be satisfied";
    EXPECT_LE(x[0], 1.0 + 1e-4) << "Inequality should be satisfied";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test with no constraints (just P and q)
TEST_F(ConeVarietiesTest, NoCones) {
    int n = 2, m = 0;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};
    int64_t nnzA = 0;

    Cones cones{};  // No cones

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // min 0.5*(x1^2 + x2^2) + x1 + 2*x2  (unconstrained)
    std::vector<double> P_values = {1.0, 1.0};
    std::vector<double> A_values = {};
    std::vector<double> q = {1.0, 2.0};
    std::vector<double> b = {};

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * 1);  // Allocate at least 1
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * 1);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    // Optimal: x1 = -1, x2 = -2 (solve P*x + q = 0)
    EXPECT_NEAR(x[0], -1.0, 1e-4);
    EXPECT_NEAR(x[1], -2.0, 1e-4);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
