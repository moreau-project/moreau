// test_eq_constrained.cpp
// Tests for equality-only constraints (ZeroCone)
// Equivalent to packages/moreau-cpu/tests/rust/basic_eq_constrained.rs

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>
#include <limits>

using namespace moreau;

class EqConstrainedTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test feasible equality-constrained problem
// min 0.5 * ||x||^2  s.t. A*x = b
// where A = [0 1 1; 0 1 -1], b = [2; 0]
// Solution: x = [0, 1, 1]
TEST_F(EqConstrainedTest, FeasibleProblem) {
    const int64_t n = 3;
    const int64_t m = 2;
    const int64_t batchSize = 1;

    // P = I (identity matrix)
    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;
    std::vector<double> P_values = {1.0, 1.0, 1.0};

    // q = 0
    std::vector<double> q(n, 0.0);

    // A = [0 1 1; 0 1 -1] in CSR format
    std::vector<int64_t> A_ro = {0, 2, 4};
    std::vector<int64_t> A_ci = {1, 2, 1, 2};
    int64_t nnzA = 4;
    std::vector<double> A_values = {1.0, 1.0, 1.0, -1.0};

    // b = [2, 0]
    std::vector<double> b = {2.0, 0.0};

    // Unbounded
    std::vector<double> l(n, -std::numeric_limits<double>::infinity());
    std::vector<double> u(n, std::numeric_limits<double>::infinity());

    // Cones: all equality constraints (ZeroCone)
    Cones cones;
    cones.numZeroCones = m;
    cones.numNonnegCones = 0;

    Settings settings;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize,
                          P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA,
                          cones, settings);

    // Allocate device memory
    double *d_P, *d_A, *d_q, *d_b, *d_l, *d_u;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMalloc(&d_l, sizeof(double) * n);
    cudaMalloc(&d_u, sizeof(double) * n);

    // Copy to device
    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P, d_A, d_q, d_b);

    // Get solution
    std::vector<double> x(n);
    cudaMemcpy(x.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    // Check solution: x = [0, 1, 1]
    EXPECT_NEAR(x[0], 0.0, 1e-5);
    EXPECT_NEAR(x[1], 1.0, 1e-5);
    EXPECT_NEAR(x[2], 1.0, 1e-5);

    // Cleanup
    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
    cudaFree(d_l);
    cudaFree(d_u);
}

// Test mixed equality and inequality constraints
TEST_F(EqConstrainedTest, MixedEqualityInequality) {
    // Problem: min 0.5*(x1^2 + x2^2) + x1 + x2
    // s.t. x1 + x2 = 1 (equality)
    //      x1 >= 0, x2 >= 0 (inequality)

    const int64_t n = 2;
    const int64_t m = 3; // 1 equality + 2 inequality
    const int64_t batchSize = 1;

    // P = I
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<double> P_values = {1.0, 1.0};

    // q = [1, 1]
    std::vector<double> q = {1.0, 1.0};

    // A = [1 1; -1 0; 0 -1] (equality, then x1 >= 0, x2 >= 0 as -x <= 0)
    std::vector<int64_t> A_ro = {0, 2, 3, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    int64_t nnzA = 4;
    std::vector<double> A_values = {1.0, 1.0, -1.0, -1.0};

    // b = [1, 0, 0]
    std::vector<double> b = {1.0, 0.0, 0.0};

    // Unbounded
    std::vector<double> l(n, -std::numeric_limits<double>::infinity());
    std::vector<double> u(n, std::numeric_limits<double>::infinity());

    // Cones: 1 equality, 2 inequality
    Cones cones;
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    Settings settings;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize,
                          P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA,
                          cones, settings);

    // Allocate device memory
    double *d_P, *d_A, *d_q, *d_b, *d_l, *d_u;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMalloc(&d_l, sizeof(double) * n);
    cudaMalloc(&d_u, sizeof(double) * n);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x(n);
    cudaMemcpy(x.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    // Check solution satisfies equality constraint
    EXPECT_NEAR(x[0] + x[1], 1.0, 1e-5);

    // Check non-negativity
    EXPECT_GE(x[0], -1e-5);
    EXPECT_GE(x[1], -1e-5);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
    cudaFree(d_l);
    cudaFree(d_u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
