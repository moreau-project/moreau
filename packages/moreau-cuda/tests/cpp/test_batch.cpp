// test_batch.cpp
// Comprehensive tests for batched solving functionality

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "clarabel_test_helpers.hpp"
#include <vector>
#include <random>
#include <cmath>

using namespace moreau;
using namespace clarabel_test;

class BatchTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetLastError();
    }
    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// =============================================================================
// Basic Batch Size Tests
// =============================================================================

TEST_F(BatchTest, SingleBatch) {
    // Simple QP with batch size 1
    int64_t n = 2, m = 2;
    int64_t batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    std::vector<double> P_val = {2.0, 2.0};

    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    std::vector<double> A_val = {-1.0, -1.0};

    std::vector<double> q = {1.0, 1.0};
    std::vector<double> b = {0.0, 0.0};

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize,
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

    std::vector<double> x_sol(n);
    std::vector<int32_t> status(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);

    EXPECT_EQ(status[0], static_cast<int32_t>(SolverStatus::Solved));
    // x should be at boundary: x = [0, 0]
    EXPECT_NEAR(x_sol[0], 0.0, 1e-6);
    EXPECT_NEAR(x_sol[1], 0.0, 1e-6);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(BatchTest, SmallBatch) {
    // Batch size 4
    int64_t n = 2, m = 2;
    int64_t batchSize = 4;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    // Same problem for all batches
    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int64_t i = 0; i < batchSize; i++) {
        P_val[i * nnzP + 0] = 2.0;
        P_val[i * nnzP + 1] = 2.0;
        A_val[i * nnzA + 0] = 1.0;
        A_val[i * nnzA + 1] = 1.0;
        q[i * n + 0] = -1.0;
        q[i * n + 1] = -2.0;
        b[i * m + 0] = 1.0;
        b[i * m + 1] = 1.0;
    }

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.verbose = false;

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

    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    // All batches should converge to same solution
    for (int64_t i = 0; i < batchSize; i++) {
        EXPECT_EQ(status[i], static_cast<int32_t>(SolverStatus::Solved))
            << "Batch " << i << " did not solve";
    }

    // Check consistency across batches
    for (int64_t i = 1; i < batchSize; i++) {
        EXPECT_NEAR(x_sol[i * n + 0], x_sol[0], 1e-8)
            << "Batch " << i << " x[0] differs from batch 0";
        EXPECT_NEAR(x_sol[i * n + 1], x_sol[1], 1e-8)
            << "Batch " << i << " x[1] differs from batch 0";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(BatchTest, LargeBatch) {
    // Batch size 64
    int64_t n = 3, m = 3;
    int64_t batchSize = 64;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;

    std::vector<int64_t> A_ro = {0, 1, 2, 3};
    std::vector<int64_t> A_ci = {0, 1, 2};
    int64_t nnzA = 3;

    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int64_t i = 0; i < batchSize; i++) {
        P_val[i * nnzP + 0] = 2.0;
        P_val[i * nnzP + 1] = 2.0;
        P_val[i * nnzP + 2] = 2.0;
        A_val[i * nnzA + 0] = -1.0;
        A_val[i * nnzA + 1] = -1.0;
        A_val[i * nnzA + 2] = -1.0;
        q[i * n + 0] = 1.0;
        q[i * n + 1] = 1.0;
        q[i * n + 2] = 1.0;
        b[i * m + 0] = 0.0;
        b[i * m + 1] = 0.0;
        b[i * m + 2] = 0.0;
    }

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.verbose = false;

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

    int solved_count = 0;
    for (int64_t i = 0; i < batchSize; i++) {
        if (status[i] == static_cast<int32_t>(SolverStatus::Solved) ||
            status[i] == static_cast<int32_t>(SolverStatus::AlmostSolved)) {
            solved_count++;
        }
    }
    EXPECT_EQ(solved_count, batchSize) << "Not all batches solved";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(BatchTest, VeryLargeBatch) {
    // Batch size 128
    int64_t n = 2, m = 1;
    int64_t batchSize = 128;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int64_t i = 0; i < batchSize; i++) {
        P_val[i * nnzP + 0] = 2.0;
        P_val[i * nnzP + 1] = 2.0;
        A_val[i * nnzA + 0] = 1.0;
        A_val[i * nnzA + 1] = 1.0;
        q[i * n + 0] = 0.0;
        q[i * n + 1] = 0.0;
        b[i * m + 0] = 1.0;
    }

    Cones cones{};
    cones.numZeroCones = m;

    Settings settings;
    settings.verbose = false;

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

    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    // All should solve to x = [0.5, 0.5]
    for (int64_t i = 0; i < batchSize; i++) {
        EXPECT_TRUE(status[i] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status[i] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "Batch " << i << " status: " << status[i];
        EXPECT_NEAR(x_sol[i * n + 0], 0.5, 1e-6) << "Batch " << i;
        EXPECT_NEAR(x_sol[i * n + 1], 0.5, 1e-6) << "Batch " << i;
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// =============================================================================
// Different Problems in Batch
// =============================================================================

TEST_F(BatchTest, DifferentQVectors) {
    // Same structure, different q vectors
    int64_t n = 2, m = 2;
    int64_t batchSize = 4;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    // Different q for each batch
    std::vector<std::vector<double>> q_vals = {
        {-1.0, -1.0},
        {-2.0, -1.0},
        {-1.0, -2.0},
        {-2.0, -2.0}
    };

    for (int64_t i = 0; i < batchSize; i++) {
        P_val[i * nnzP + 0] = 2.0;
        P_val[i * nnzP + 1] = 2.0;
        A_val[i * nnzA + 0] = 1.0;
        A_val[i * nnzA + 1] = 1.0;
        q[i * n + 0] = q_vals[i][0];
        q[i * n + 1] = q_vals[i][1];
        b[i * m + 0] = 1.0;
        b[i * m + 1] = 1.0;
    }

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.verbose = false;

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

    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    // All should solve
    for (int64_t i = 0; i < batchSize; i++) {
        EXPECT_TRUE(status[i] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status[i] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "Batch " << i << " did not solve";
    }

    // Solutions should be different
    bool all_same = true;
    for (int64_t i = 1; i < batchSize; i++) {
        if (std::abs(x_sol[i * n + 0] - x_sol[0]) > 1e-6 ||
            std::abs(x_sol[i * n + 1] - x_sol[1]) > 1e-6) {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same) << "Different q should give different solutions";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(BatchTest, DifferentBVectors) {
    // Same structure, different b vectors
    int64_t n = 2, m = 1;
    int64_t batchSize = 4;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    // Different b for each batch
    double b_vals[] = {1.0, 2.0, 3.0, 4.0};

    for (int64_t i = 0; i < batchSize; i++) {
        P_val[i * nnzP + 0] = 2.0;
        P_val[i * nnzP + 1] = 2.0;
        A_val[i * nnzA + 0] = 1.0;
        A_val[i * nnzA + 1] = 1.0;
        q[i * n + 0] = 0.0;
        q[i * n + 1] = 0.0;
        b[i * m + 0] = b_vals[i];
    }

    Cones cones{};
    cones.numZeroCones = m;

    Settings settings;
    settings.verbose = false;

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

    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    // All should solve with x[0] + x[1] = b[i]
    for (int64_t i = 0; i < batchSize; i++) {
        EXPECT_TRUE(status[i] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status[i] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "Batch " << i << " did not solve";

        double sum = x_sol[i * n + 0] + x_sol[i * n + 1];
        EXPECT_NEAR(sum, b_vals[i], 1e-6)
            << "Batch " << i << " constraint violated";

        // Optimal: x[0] = x[1] = b/2
        EXPECT_NEAR(x_sol[i * n + 0], b_vals[i] / 2.0, 1e-6);
        EXPECT_NEAR(x_sol[i * n + 1], b_vals[i] / 2.0, 1e-6);
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(BatchTest, DifferentPMatrices) {
    // Different P matrices for each batch
    int64_t n = 2, m = 1;
    int64_t batchSize = 3;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    // Different P diagonals: [1,1], [2,2], [1,4]
    std::vector<std::vector<double>> P_diags = {
        {1.0, 1.0},
        {2.0, 2.0},
        {1.0, 4.0}
    };

    for (int64_t i = 0; i < batchSize; i++) {
        P_val[i * nnzP + 0] = P_diags[i][0];
        P_val[i * nnzP + 1] = P_diags[i][1];
        A_val[i * nnzA + 0] = 1.0;
        A_val[i * nnzA + 1] = 1.0;
        q[i * n + 0] = 0.0;
        q[i * n + 1] = 0.0;
        b[i * m + 0] = 1.0;
    }

    Cones cones{};
    cones.numZeroCones = m;

    Settings settings;
    settings.verbose = false;

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

    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    // Batch 0: P=[1,1], x=[0.5, 0.5]
    // Batch 1: P=[2,2], x=[0.5, 0.5]
    // Batch 2: P=[1,4], x=[0.8, 0.2] (optimal when P1*x1 = P2*x2)
    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(x_sol[0 * n + 0], 0.5, 1e-5);
    EXPECT_NEAR(x_sol[0 * n + 1], 0.5, 1e-5);

    EXPECT_TRUE(status[1] == static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(x_sol[1 * n + 0], 0.5, 1e-5);
    EXPECT_NEAR(x_sol[1 * n + 1], 0.5, 1e-5);

    EXPECT_TRUE(status[2] == static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(x_sol[2 * n + 0], 0.8, 1e-5);
    EXPECT_NEAR(x_sol[2 * n + 1], 0.2, 1e-5);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// =============================================================================
// Odd Dimension Batch Tests (Memory Alignment)
// =============================================================================

TEST_F(BatchTest, OddDimensionN3) {
    // n=3 (odd) exposes memory alignment issues
    int64_t n = 3, m = 1;
    int64_t batchSize = 4;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;

    std::vector<int64_t> A_ro = {0, 3};
    std::vector<int64_t> A_ci = {0, 1, 2};
    int64_t nnzA = 3;

    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int64_t i = 0; i < batchSize; i++) {
        P_val[i * nnzP + 0] = 2.0;
        P_val[i * nnzP + 1] = 2.0;
        P_val[i * nnzP + 2] = 2.0;
        A_val[i * nnzA + 0] = 1.0;
        A_val[i * nnzA + 1] = 1.0;
        A_val[i * nnzA + 2] = 1.0;
        q[i * n + 0] = 0.0;
        q[i * n + 1] = 0.0;
        q[i * n + 2] = 0.0;
        b[i * m + 0] = 3.0;
    }

    Cones cones{};
    cones.numZeroCones = m;

    Settings settings;
    settings.verbose = false;

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

    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    // All batches should get x = [1, 1, 1]
    for (int64_t i = 0; i < batchSize; i++) {
        EXPECT_TRUE(status[i] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status[i] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "Batch " << i << " did not solve";
        EXPECT_NEAR(x_sol[i * n + 0], 1.0, 1e-6) << "Batch " << i << " x[0]";
        EXPECT_NEAR(x_sol[i * n + 1], 1.0, 1e-6) << "Batch " << i << " x[1]";
        EXPECT_NEAR(x_sol[i * n + 2], 1.0, 1e-6) << "Batch " << i << " x[2]";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(BatchTest, OddDimensionN5) {
    // n=5 (odd)
    int64_t n = 5, m = 1;
    int64_t batchSize = 3;

    std::vector<int64_t> P_ro = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> P_ci = {0, 1, 2, 3, 4};
    int64_t nnzP = 5;

    std::vector<int64_t> A_ro = {0, 5};
    std::vector<int64_t> A_ci = {0, 1, 2, 3, 4};
    int64_t nnzA = 5;

    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int64_t i = 0; i < batchSize; i++) {
        for (int64_t j = 0; j < nnzP; j++) {
            P_val[i * nnzP + j] = 2.0;
        }
        for (int64_t j = 0; j < nnzA; j++) {
            A_val[i * nnzA + j] = 1.0;
        }
        for (int64_t j = 0; j < n; j++) {
            q[i * n + j] = 0.0;
        }
        b[i * m + 0] = 5.0;
    }

    Cones cones{};
    cones.numZeroCones = m;

    Settings settings;
    settings.verbose = false;

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

    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    // All batches should get x = [1, 1, 1, 1, 1]
    for (int64_t i = 0; i < batchSize; i++) {
        EXPECT_TRUE(status[i] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status[i] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "Batch " << i << " did not solve";
        for (int64_t j = 0; j < n; j++) {
            EXPECT_NEAR(x_sol[i * n + j], 1.0, 1e-6)
                << "Batch " << i << " x[" << j << "]";
        }
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// =============================================================================
// Batch with Mixed Cones
// =============================================================================

TEST_F(BatchTest, BatchWithSOC) {
    // Batch with second-order cone (dimension 3)
    int64_t n = 3, m = 3;
    int64_t batchSize = 4;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;

    // Identity A (SOC: t >= ||u||)
    std::vector<int64_t> A_ro = {0, 1, 2, 3};
    std::vector<int64_t> A_ci = {0, 1, 2};
    int64_t nnzA = 3;

    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int64_t i = 0; i < batchSize; i++) {
        P_val[i * nnzP + 0] = 0.0;
        P_val[i * nnzP + 1] = 0.0;
        P_val[i * nnzP + 2] = 0.0;
        A_val[i * nnzA + 0] = -1.0;
        A_val[i * nnzA + 1] = -1.0;
        A_val[i * nnzA + 2] = -1.0;
        // min t s.t. t >= ||u||, u = [1, 0]
        q[i * n + 0] = 1.0;  // t
        q[i * n + 1] = 0.0;  // u1
        q[i * n + 2] = 0.0;  // u2
        b[i * m + 0] = 0.0;
        b[i * m + 1] = -1.0;  // u1 = 1
        b[i * m + 2] = 0.0;   // u2 = 0
    }

    Cones cones{};
    cones.socConeDims = {3};
    cones.numSocCones = 1;  // One 3D SOC

    Settings settings;
    settings.verbose = false;

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

    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    // Solution: min t s.t. t >= ||(u1-1, u2)||
    // Optimal: u1=1, u2=0, t=0 (s = [0, 0, 0] on SOC boundary)
    for (int64_t i = 0; i < batchSize; i++) {
        EXPECT_TRUE(status[i] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status[i] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "Batch " << i << " did not solve";
        EXPECT_NEAR(x_sol[i * n + 0], 0.0, 1e-4) << "Batch " << i << " t";
        EXPECT_NEAR(x_sol[i * n + 1], 1.0, 1e-4) << "Batch " << i << " u1";
        EXPECT_NEAR(x_sol[i * n + 2], 0.0, 1e-4) << "Batch " << i << " u2";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(BatchTest, BatchMixedZeroNonneg) {
    // Mixed zero (equality) and nonneg (inequality) cones
    int64_t n = 2, m = 3;
    int64_t batchSize = 4;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    // A = [[1, 1], [-1, 0], [0, -1]]
    std::vector<int64_t> A_ro = {0, 2, 3, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    int64_t nnzA = 4;

    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int64_t i = 0; i < batchSize; i++) {
        P_val[i * nnzP + 0] = 2.0;
        P_val[i * nnzP + 1] = 2.0;
        A_val[i * nnzA + 0] = 1.0;
        A_val[i * nnzA + 1] = 1.0;
        A_val[i * nnzA + 2] = -1.0;
        A_val[i * nnzA + 3] = -1.0;
        q[i * n + 0] = 0.0;
        q[i * n + 1] = 0.0;
        b[i * m + 0] = 1.0;  // x0 + x1 = 1
        b[i * m + 1] = 0.0;  // x0 >= 0
        b[i * m + 2] = 0.0;  // x1 >= 0
    }

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    Settings settings;
    settings.verbose = false;

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

    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    // Solution: x = [0.5, 0.5]
    for (int64_t i = 0; i < batchSize; i++) {
        EXPECT_TRUE(status[i] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status[i] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "Batch " << i << " did not solve";
        EXPECT_NEAR(x_sol[i * n + 0], 0.5, 1e-6) << "Batch " << i;
        EXPECT_NEAR(x_sol[i * n + 1], 0.5, 1e-6) << "Batch " << i;
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// =============================================================================
// Batch Objective Value Tests
// =============================================================================

TEST_F(BatchTest, ObjectiveValueConsistency) {
    // Verify objective values are computed correctly for batch
    int64_t n = 2, m = 1;
    int64_t batchSize = 4;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int64_t i = 0; i < batchSize; i++) {
        P_val[i * nnzP + 0] = 2.0;
        P_val[i * nnzP + 1] = 2.0;
        A_val[i * nnzA + 0] = 1.0;
        A_val[i * nnzA + 1] = 1.0;
        q[i * n + 0] = 0.0;
        q[i * n + 1] = 0.0;
        b[i * m + 0] = static_cast<double>(i + 1);  // b = 1, 2, 3, 4
    }

    Cones cones{};
    cones.numZeroCones = m;

    Settings settings;
    settings.verbose = false;

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

    std::vector<double> obj_primal(batchSize);
    solver.info.cost_primal.gpuToCpu(obj_primal.data());

    // x = [b/2, b/2], obj = (1/2) * 2 * (b/2)^2 * 2 = b^2/2
    // b=1: obj=0.5, b=2: obj=2, b=3: obj=4.5, b=4: obj=8
    double expected_obj[] = {0.5, 2.0, 4.5, 8.0};

    for (int64_t i = 0; i < batchSize; i++) {
        EXPECT_NEAR(obj_primal[i], expected_obj[i], 1e-6)
            << "Batch " << i << " objective mismatch";
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
