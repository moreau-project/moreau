// test_unconstrained.cpp
// Tests for unconstrained QP problems (m=0)
// Equivalent to packages/moreau-cpu/tests/rust/basic_unconstrained.rs

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>
#include <limits>

using namespace moreau;

class UnconstrainedTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test unconstrained feasible problem
// min 0.5 * x'Px + q'x  with P = I, q = [1, 2, -3]
// Solution: x = -q = [-1, -2, 3]
TEST_F(UnconstrainedTest, Feasible) {
    const int64_t n = 3;
    const int64_t m = 1;  // Need at least m=1 for GPU solver
    const int64_t batchSize = 1;

    // P = I (identity)
    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;
    std::vector<double> P_values = {1.0, 1.0, 1.0};

    // q = [1, 2, -3]
    std::vector<double> q = {1.0, 2.0, -3.0};

    // Dummy zero cone constraint (0 = 0)
    std::vector<int64_t> A_ro = {0, 0};
    std::vector<int64_t> A_ci = {};
    int64_t nnzA = 0;
    std::vector<double> A_values = {};
    std::vector<double> b = {0.0};

    // Unbounded
    std::vector<double> l(n, -std::numeric_limits<double>::infinity());
    std::vector<double> u(n, std::numeric_limits<double>::infinity());

    Cones cones;
    cones.numZeroCones = 1;

    Settings settings;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize,
                          P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA,
                          cones, settings);

    double *d_P, *d_A, *d_q, *d_b, *d_l, *d_u;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * std::max(nnzA, int64_t(1)));
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMalloc(&d_l, sizeof(double) * n);
    cudaMalloc(&d_u, sizeof(double) * n);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x(n);
    cudaMemcpy(x.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    // Solution: x = -q = [-1, -2, 3]
    EXPECT_NEAR(x[0], -1.0, 1e-5);
    EXPECT_NEAR(x[1], -2.0, 1e-5);
    EXPECT_NEAR(x[2], 3.0, 1e-5);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
    cudaFree(d_l);
    cudaFree(d_u);
}

// Test unconstrained with non-identity P
// min 0.5 * x'Px + q'x with P = diag(2, 4, 6), q = [2, 4, 6]
// Solution: Px = -q => x = [-1, -1, -1]
TEST_F(UnconstrainedTest, NonIdentityP) {
    const int64_t n = 3;
    const int64_t m = 1;
    const int64_t batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;
    std::vector<double> P_values = {2.0, 4.0, 6.0};

    std::vector<double> q = {2.0, 4.0, 6.0};

    std::vector<int64_t> A_ro = {0, 0};
    std::vector<int64_t> A_ci = {};
    int64_t nnzA = 0;
    std::vector<double> A_values = {};
    std::vector<double> b = {0.0};

    std::vector<double> l(n, -std::numeric_limits<double>::infinity());
    std::vector<double> u(n, std::numeric_limits<double>::infinity());

    Cones cones;
    cones.numZeroCones = 1;

    Settings settings;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize,
                          P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA,
                          cones, settings);

    double *d_P, *d_A, *d_q, *d_b, *d_l, *d_u;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * std::max(nnzA, int64_t(1)));
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMalloc(&d_l, sizeof(double) * n);
    cudaMalloc(&d_u, sizeof(double) * n);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u.data(), sizeof(double) * n, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x(n);
    cudaMemcpy(x.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    // Solution: x = -P^{-1}q = [-1, -1, -1]
    EXPECT_NEAR(x[0], -1.0, 1e-5);
    EXPECT_NEAR(x[1], -1.0, 1e-5);
    EXPECT_NEAR(x[2], -1.0, 1e-5);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
    cudaFree(d_l);
    cudaFree(d_u);
}

// Test unconstrained batched
TEST_F(UnconstrainedTest, Batched) {
    const int64_t n = 2;
    const int64_t m = 1;
    const int64_t batchSize = 3;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 0};
    std::vector<int64_t> A_ci = {};
    int64_t nnzA = 0;

    Cones cones;
    cones.numZeroCones = 1;

    Settings settings;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize,
                          P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA,
                          cones, settings);

    // P = I for all problems
    std::vector<double> P_batch(nnzP * batchSize);
    for (int64_t i = 0; i < batchSize; i++) {
        P_batch[i * nnzP + 0] = 1.0;
        P_batch[i * nnzP + 1] = 1.0;
    }

    std::vector<double> A_batch(1);  // dummy

    // Different q for each problem
    std::vector<double> q_batch = {
        1.0, 2.0,    // q1 -> x = [-1, -2]
        -3.0, 4.0,   // q2 -> x = [3, -4]
        0.5, -0.5,   // q3 -> x = [-0.5, 0.5]
    };

    std::vector<double> b_batch(m * batchSize, 0.0);
    std::vector<double> l_batch(n * batchSize, -std::numeric_limits<double>::infinity());
    std::vector<double> u_batch(n * batchSize, std::numeric_limits<double>::infinity());

    double *d_P, *d_A, *d_q, *d_b, *d_l, *d_u;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * std::max(int64_t(1), nnzA * batchSize));
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);
    cudaMalloc(&d_l, sizeof(double) * n * batchSize);
    cudaMalloc(&d_u, sizeof(double) * n * batchSize);

    cudaMemcpy(d_P, P_batch.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_batch.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_batch.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_l, l_batch.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u_batch.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_batch(n * batchSize);
    cudaMemcpy(x_batch.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);

    // Check solutions
    EXPECT_NEAR(x_batch[0 * n + 0], -1.0, 1e-4);
    EXPECT_NEAR(x_batch[0 * n + 1], -2.0, 1e-4);
    EXPECT_NEAR(x_batch[1 * n + 0], 3.0, 1e-4);
    EXPECT_NEAR(x_batch[1 * n + 1], -4.0, 1e-4);
    EXPECT_NEAR(x_batch[2 * n + 0], -0.5, 1e-4);
    EXPECT_NEAR(x_batch[2 * n + 1], 0.5, 1e-4);

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
