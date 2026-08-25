// test_random_problems.cpp
// Tests with randomly generated problems to verify robustness
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>
#include <random>

using namespace moreau;

class RandomProblemsTest : public ::testing::Test {
protected:
    std::mt19937 rng;

    void SetUp() override {
        rng.seed(42);  // Fixed seed for reproducibility
    }
    void TearDown() override {}

    // Generate random positive definite diagonal P
    std::vector<double> randomDiagonalP(int n, double minVal = 0.1, double maxVal = 10.0) {
        std::uniform_real_distribution<double> dist(minVal, maxVal);
        std::vector<double> P(n);
        for (int i = 0; i < n; i++) {
            P[i] = dist(rng);
        }
        return P;
    }

    // Generate random vector
    std::vector<double> randomVector(int n, double minVal = -10.0, double maxVal = 10.0) {
        std::uniform_real_distribution<double> dist(minVal, maxVal);
        std::vector<double> v(n);
        for (int i = 0; i < n; i++) {
            v[i] = dist(rng);
        }
        return v;
    }
};

// Test random diagonal QP with inequality constraints
TEST_F(RandomProblemsTest, RandomDiagonalQP) {
    int n = 5, m = 3;
    int batchSize = 1;

    // Diagonal P
    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci(n);
    for (int i = 0; i <= n; i++) P_ro[i] = i;
    for (int i = 0; i < n; i++) P_ci[i] = i;
    int64_t nnzP = n;

    // Diagonal A (each constraint affects one variable)
    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci(m);
    for (int i = 0; i <= m; i++) A_ro[i] = i;
    for (int i = 0; i < m; i++) A_ci[i] = i;
    int64_t nnzA = m;

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    auto P_values = randomDiagonalP(n);
    auto A_values = randomVector(m, 0.1, 5.0);  // Positive A values
    auto q = randomVector(n);
    auto b = randomVector(m, 1.0, 20.0);  // Positive b for feasibility

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

    // Solution should be finite
    for (int i = 0; i < n; i++) {
        EXPECT_FALSE(std::isnan(x[i])) << "x[" << i << "] is NaN";
        EXPECT_FALSE(std::isinf(x[i])) << "x[" << i << "] is Inf";
    }

    // Constraints should be satisfied (Ax <= b)
    for (int i = 0; i < m; i++) {
        EXPECT_LE(A_values[i] * x[i], b[i] + 1e-4)
            << "Constraint " << i << " violated: " << A_values[i] * x[i] << " > " << b[i];
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test multiple random problems in a batch
TEST_F(RandomProblemsTest, BatchOfRandomProblems) {
    int n = 3, m = 2;
    int batchSize = 5;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;

    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // Generate different random data for each batch
    std::vector<double> P_values(nnzP * batchSize);
    std::vector<double> A_values(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int batch = 0; batch < batchSize; batch++) {
        auto P_batch = randomDiagonalP(nnzP);
        auto A_batch = randomVector(nnzA, 0.5, 2.0);
        auto q_batch = randomVector(n);
        auto b_batch = randomVector(m, 5.0, 15.0);

        for (int i = 0; i < nnzP; i++) P_values[batch * nnzP + i] = P_batch[i];
        for (int i = 0; i < nnzA; i++) A_values[batch * nnzA + i] = A_batch[i];
        for (int i = 0; i < n; i++) q[batch * n + i] = q_batch[i];
        for (int i = 0; i < m; i++) b[batch * m + i] = b_batch[i];
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x(n * batchSize);
    std::vector<int32_t> status(batchSize);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaMemcpy(status.data(), solver.info.status_device, sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // All batches should produce finite solutions
    for (int batch = 0; batch < batchSize; batch++) {
        for (int i = 0; i < n; i++) {
            EXPECT_FALSE(std::isnan(x[batch * n + i]))
                << "Batch " << batch << ", x[" << i << "] is NaN";
            EXPECT_FALSE(std::isinf(x[batch * n + i]))
                << "Batch " << batch << ", x[" << i << "] is Inf";
        }

        // Status should be valid
        SolverStatus st = static_cast<SolverStatus>(status[batch]);
        EXPECT_TRUE(st == SolverStatus::Solved ||
                    st == SolverStatus::AlmostSolved ||
                    st == SolverStatus::MaxIterations)
            << "Batch " << batch << " has unexpected status " << status[batch];
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test with random equality constraints
TEST_F(RandomProblemsTest, RandomEqualityConstraints) {
    int n = 4, m = 2;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2, 3, 4};
    std::vector<int64_t> P_ci = {0, 1, 2, 3};
    int64_t nnzP = 4;

    // Each equality constraint uses two variables
    std::vector<int64_t> A_ro = {0, 2, 4};
    std::vector<int64_t> A_ci = {0, 1, 2, 3};
    int64_t nnzA = 4;

    Cones cones{};
    cones.numZeroCones = m;  // Equality constraints

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    auto P_values = randomDiagonalP(nnzP);
    auto A_values = randomVector(nnzA, 0.5, 2.0);
    auto q = randomVector(n);
    auto b = randomVector(m, -5.0, 5.0);

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
    double c1 = A_values[0] * x[0] + A_values[1] * x[1];
    double c2 = A_values[2] * x[2] + A_values[3] * x[3];

    EXPECT_NEAR(c1, b[0], 1e-4) << "First equality constraint violated";
    EXPECT_NEAR(c2, b[1], 1e-4) << "Second equality constraint violated";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
