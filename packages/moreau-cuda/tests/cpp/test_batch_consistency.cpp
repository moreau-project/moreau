// test_batch_consistency.cpp
// Tests that batched problems produce consistent results with single problems
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>

using namespace moreau;

class BatchConsistencyTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper to check if two doubles are close
    bool isClose(double a, double b, double rtol = 1e-5, double atol = 1e-8) {
        return std::abs(a - b) <= atol + rtol * std::abs(b);
    }
};

// Test that identical problems in a batch produce identical results
TEST_F(BatchConsistencyTest, IdenticalProblemsInBatch) {
    int n = 2, m = 2;
    int batchSize = 4;

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

    // Same problem data for all batches
    std::vector<double> P_values(nnzP * batchSize);
    std::vector<double> A_values(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int batch = 0; batch < batchSize; batch++) {
        P_values[batch * nnzP + 0] = 2.0;
        P_values[batch * nnzP + 1] = 1.0;
        A_values[batch * nnzA + 0] = 1.0;
        A_values[batch * nnzA + 1] = 1.0;
        q[batch * n + 0] = 1.0;
        q[batch * n + 1] = 2.0;
        b[batch * m + 0] = 5.0;
        b[batch * m + 1] = 5.0;
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
    std::vector<double> z(m * batchSize);
    solver.solution.x.gpuToCpu(x.data(), 0);
    solver.solution.z.gpuToCpu(z.data(), 0);
    cudaDeviceSynchronize();

    // All batches should have the same solution (within tolerance)
    for (int batch = 1; batch < batchSize; batch++) {
        for (int i = 0; i < n; i++) {
            EXPECT_TRUE(isClose(x[batch * n + i], x[i], 1e-4, 1e-6))
                << "x mismatch at batch " << batch << ", index " << i
                << ": " << x[batch * n + i] << " vs " << x[i];
        }
        for (int i = 0; i < m; i++) {
            EXPECT_TRUE(isClose(z[batch * m + i], z[i], 1e-4, 1e-6))
                << "z mismatch at batch " << batch << ", index " << i
                << ": " << z[batch * m + i] << " vs " << z[i];
        }
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test that different problems in batch produce different results
TEST_F(BatchConsistencyTest, DifferentProblemsInBatch) {
    int n = 2, m = 1;
    int batchSize = 3;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numNonnegCones = 1;

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // Different q vectors for each batch
    std::vector<double> P_values(nnzP * batchSize);
    std::vector<double> A_values(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b_vec(m * batchSize);

    for (int batch = 0; batch < batchSize; batch++) {
        P_values[batch * nnzP + 0] = 1.0;
        P_values[batch * nnzP + 1] = 1.0;
        A_values[batch * nnzA + 0] = 1.0;
        A_values[batch * nnzA + 1] = 1.0;
        // Different linear costs
        q[batch * n + 0] = 1.0 + batch;
        q[batch * n + 1] = 2.0 - batch * 0.5;
        b_vec[batch * m + 0] = 10.0;
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_vec.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x(n * batchSize);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    // Solutions should be different (at least some)
    bool anyDifferent = false;
    for (int batch = 1; batch < batchSize; batch++) {
        for (int i = 0; i < n; i++) {
            if (!isClose(x[batch * n + i], x[i], 1e-2, 1e-4)) {
                anyDifferent = true;
                break;
            }
        }
        if (anyDifferent) break;
    }
    EXPECT_TRUE(anyDifferent) << "Expected different solutions for different q vectors";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test batch size of 1 works correctly
TEST_F(BatchConsistencyTest, SingleBatch) {
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
    std::vector<double> q = {1.0, 1.0};
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

    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    // Should produce a valid solution
    for (int i = 0; i < n; i++) {
        EXPECT_FALSE(std::isnan(x[i])) << "x[" << i << "] is NaN";
        EXPECT_FALSE(std::isinf(x[i])) << "x[" << i << "] is Inf";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
