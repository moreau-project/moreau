// test_batch_b_vectors.cpp
// Test for batch bug when b vectors differ across batch elements

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <set>

using namespace moreau;

class BatchBVectorsTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetLastError();
    }
    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// Build a simplex problem: min q'x + 0.5*x'Px s.t. sum(x) = 1, x >= eps
// Returns: n (variables), m (constraints), nnzP, nnzA, and fills vectors
void buildSimplexProblem(
    int n,
    std::vector<int64_t>& P_ro,
    std::vector<int64_t>& P_ci,
    std::vector<int64_t>& A_ro,
    std::vector<int64_t>& A_ci,
    int64_t& m,
    int64_t& nnzP,
    int64_t& nnzA
) {
    // P = diagonal (identity scaled)
    P_ro.resize(n + 1);
    P_ci.resize(n);
    for (int i = 0; i <= n; i++) P_ro[i] = i;
    for (int i = 0; i < n; i++) P_ci[i] = i;
    nnzP = n;

    // A: first row is equality (sum = 1), remaining are -I (for x >= eps)
    m = 1 + n;
    nnzA = n + n;  // n entries for equality row, n entries for -I diagonal

    A_ro.resize(m + 1);
    A_ci.resize(nnzA);

    // Row 0: equality constraint (sum(x) = 1)
    A_ro[0] = 0;
    for (int i = 0; i < n; i++) {
        A_ci[i] = i;
    }
    A_ro[1] = n;

    // Rows 1..n: -x_i <= -eps (i.e., x_i >= eps)
    for (int i = 0; i < n; i++) {
        A_ci[n + i] = i;
        A_ro[2 + i] = n + i + 1;
    }
}

// Test with small n where the bug doesn't appear
TEST_F(BatchBVectorsTest, SmallProblemDifferentB) {
    int64_t n = 10;
    int64_t batchSize = 4;

    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    int64_t m, nnzP, nnzA;
    buildSimplexProblem(n, P_ro, P_ci, A_ro, A_ci, m, nnzP, nnzA);

    double eps = 1e-4;

    // Build batched data
    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    for (int64_t batch = 0; batch < batchSize; batch++) {
        // P = 0.01 * I
        for (int64_t i = 0; i < nnzP; i++) {
            P_val[batch * nnzP + i] = 0.01;
        }

        // A: first n entries are 1.0 (equality), next n are -1.0 (bounds)
        for (int64_t i = 0; i < n; i++) {
            A_val[batch * nnzA + i] = 1.0;
        }
        for (int64_t i = 0; i < n; i++) {
            A_val[batch * nnzA + n + i] = -1.0;
        }

        // q: zeros
        for (int64_t i = 0; i < n; i++) {
            q[batch * n + i] = 0.0;
        }

        // b: DIFFERENT per batch
        // b[0] = 1 (equality: sum = 1)
        // b[1..n] = -eps (bounds: x >= eps)
        b[batch * m + 0] = 1.0;
        for (int64_t i = 1; i <= n; i++) {
            b[batch * m + i] = -eps;
        }

        // Modify bounds for some variables in each batch differently
        // Batch 0: default
        // Batch 1: x[0] >= 0.3
        // Batch 2: x[0] >= 0.5, x[1] >= 0.3
        // Batch 3: x[0] >= 0.2, x[1] >= 0.2, x[2] >= 0.2
        if (batch == 1) {
            b[batch * m + 1] = -0.3;  // x[0] >= 0.3
        } else if (batch == 2) {
            b[batch * m + 1] = -0.5;  // x[0] >= 0.5
            b[batch * m + 2] = -0.3;  // x[1] >= 0.3
        } else if (batch == 3) {
            b[batch * m + 1] = -0.2;  // x[0] >= 0.2
            b[batch * m + 2] = -0.2;  // x[1] >= 0.2
            b[batch * m + 3] = -0.2;  // x[2] >= 0.2
        }
    }

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = n;

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

    // Check all batches solve and satisfy sum(x) = 1
    for (int64_t batch = 0; batch < batchSize; batch++) {
        EXPECT_TRUE(status[batch] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status[batch] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "Batch " << batch << " did not solve, status: " << status[batch];

        double sum = 0.0;
        for (int64_t i = 0; i < n; i++) {
            sum += x_sol[batch * n + i];
        }
        EXPECT_NEAR(sum, 1.0, 1e-4)
            << "Batch " << batch << " sum(x) = " << sum << " != 1.0";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test with larger n where the bug appears (n >= 100)
TEST_F(BatchBVectorsTest, LargeProblemDifferentB) {
    int64_t n = 200;  // Bug appears around n >= 100
    int64_t batchSize = 8;

    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    int64_t m, nnzP, nnzA;
    buildSimplexProblem(n, P_ro, P_ci, A_ro, A_ci, m, nnzP, nnzA);

    double eps = 1e-4;

    // Build batched data
    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    std::mt19937 rng(42);

    for (int64_t batch = 0; batch < batchSize; batch++) {
        // P = 0.01 * I
        for (int64_t i = 0; i < nnzP; i++) {
            P_val[batch * nnzP + i] = 0.01;
        }

        // A: first n entries are 1.0 (equality), next n are -1.0 (bounds)
        for (int64_t i = 0; i < n; i++) {
            A_val[batch * nnzA + i] = 1.0;
        }
        for (int64_t i = 0; i < n; i++) {
            A_val[batch * nnzA + n + i] = -1.0;
        }

        // q: small random values
        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        for (int64_t i = 0; i < n; i++) {
            q[batch * n + i] = dist(rng);
        }

        // b: start with default
        b[batch * m + 0] = 1.0;  // equality: sum = 1
        for (int64_t i = 1; i <= n; i++) {
            b[batch * m + i] = -eps;  // x >= eps
        }

        // DIFFERENT bounds per batch: randomly increase some lower bounds
        // Keep total lower bounds reasonable (sum of bounds < 1.0 for feasibility)
        int numConstrained = 3;  // fixed small number to ensure feasibility
        std::uniform_int_distribution<int64_t> idx_dist(0, n - 1);
        std::uniform_real_distribution<double> bound_dist(0.01, 0.05);  // smaller bounds

        std::set<int64_t> constrained_vars;
        while (constrained_vars.size() < static_cast<size_t>(numConstrained)) {
            constrained_vars.insert(idx_dist(rng));
        }
        for (int64_t var_idx : constrained_vars) {
            double min_val = bound_dist(rng);
            b[batch * m + 1 + var_idx] = -min_val;  // x[var_idx] >= min_val
        }
    }

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = n;

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

    int failures = 0;
    for (int64_t batch = 0; batch < batchSize; batch++) {
        double sum = 0.0;
        for (int64_t i = 0; i < n; i++) {
            sum += x_sol[batch * n + i];
        }

        bool solved = (status[batch] == static_cast<int32_t>(SolverStatus::Solved) ||
                       status[batch] == static_cast<int32_t>(SolverStatus::AlmostSolved));
        bool sum_ok = std::abs(sum - 1.0) < 0.1;

        if (!solved || !sum_ok) {
            failures++;
            std::cout << "Batch " << batch << ": status=" << status[batch]
                      << ", sum(x)=" << sum << std::endl;
        }

        EXPECT_TRUE(solved) << "Batch " << batch << " did not solve";
        EXPECT_NEAR(sum, 1.0, 0.1) << "Batch " << batch << " sum(x) = " << sum;
    }

    EXPECT_EQ(failures, 0) << failures << " out of " << batchSize << " batches failed";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test individual solves vs batched solve to isolate the bug
TEST_F(BatchBVectorsTest, CompareSingleVsBatched) {
    int64_t n = 150;
    int64_t batchSize = 4;

    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    int64_t m, nnzP, nnzA;
    buildSimplexProblem(n, P_ro, P_ci, A_ro, A_ci, m, nnzP, nnzA);

    double eps = 1e-4;

    // Build data for a single problem template
    std::vector<double> P_val_single(nnzP);
    std::vector<double> A_val_single(nnzA);

    for (int64_t i = 0; i < nnzP; i++) P_val_single[i] = 0.01;
    for (int64_t i = 0; i < n; i++) A_val_single[i] = 1.0;
    for (int64_t i = 0; i < n; i++) A_val_single[n + i] = -1.0;

    // Different b vectors for each problem
    std::vector<std::vector<double>> b_per_problem(batchSize);
    std::vector<std::vector<double>> q_per_problem(batchSize);

    std::mt19937 rng(123);
    std::uniform_real_distribution<double> bound_dist(0.01, 0.15);

    for (int64_t batch = 0; batch < batchSize; batch++) {
        b_per_problem[batch].resize(m);
        q_per_problem[batch].resize(n);

        b_per_problem[batch][0] = 1.0;
        for (int64_t i = 1; i <= n; i++) {
            b_per_problem[batch][i] = -eps;
        }

        // Vary bounds
        for (int i = 0; i < 10; i++) {
            int64_t idx = rng() % n;
            b_per_problem[batch][1 + idx] = -bound_dist(rng);
        }

        for (int64_t i = 0; i < n; i++) {
            q_per_problem[batch][i] = 0.0;
        }
    }

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = n;

    Settings settings;
    settings.verbose = false;

    // Solve individually
    std::vector<double> x_single(n * batchSize);

    for (int64_t batch = 0; batch < batchSize; batch++) {
        CompiledSolver solver_single(n, m, 1,
                      P_ro.data(), P_ci.data(), nnzP,
                      A_ro.data(), A_ci.data(), nnzA,
                      cones, settings);

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * nnzP);
        cudaMalloc(&d_A, sizeof(double) * nnzA);
        cudaMalloc(&d_q, sizeof(double) * n);
        cudaMalloc(&d_b, sizeof(double) * m);

        cudaMemcpy(d_P, P_val_single.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_val_single.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q_per_problem[batch].data(), sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_per_problem[batch].data(), sizeof(double) * m, cudaMemcpyHostToDevice);

        solver_single.solveAll(d_P, d_A, d_q, d_b);

        cudaMemcpy(&x_single[batch * n], solver_single.solution.x.data(),
                   sizeof(double) * n, cudaMemcpyDeviceToHost);

        double sum = 0.0;
        for (int64_t i = 0; i < n; i++) sum += x_single[batch * n + i];
        std::cout << "Single solve batch " << batch << ": sum(x) = " << sum << std::endl;

        cudaFree(d_P);
        cudaFree(d_A);
        cudaFree(d_q);
        cudaFree(d_b);
    }

    // Solve batched
    std::vector<double> P_val_batch(nnzP * batchSize);
    std::vector<double> A_val_batch(nnzA * batchSize);
    std::vector<double> q_batch(n * batchSize);
    std::vector<double> b_batch(m * batchSize);

    for (int64_t batch = 0; batch < batchSize; batch++) {
        for (int64_t i = 0; i < nnzP; i++) P_val_batch[batch * nnzP + i] = P_val_single[i];
        for (int64_t i = 0; i < nnzA; i++) A_val_batch[batch * nnzA + i] = A_val_single[i];
        for (int64_t i = 0; i < n; i++) q_batch[batch * n + i] = q_per_problem[batch][i];
        for (int64_t i = 0; i < m; i++) b_batch[batch * m + i] = b_per_problem[batch][i];
    }

    CompiledSolver solver_batch(n, m, batchSize,
                  P_ro.data(), P_ci.data(), nnzP,
                  A_ro.data(), A_ci.data(), nnzA,
                  cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P, P_val_batch.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val_batch.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_batch.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_batch.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    solver_batch.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_batch(n * batchSize);
    cudaMemcpy(x_batch.data(), solver_batch.solution.x.data(),
               sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);

    // Compare
    for (int64_t batch = 0; batch < batchSize; batch++) {
        double sum_single = 0.0, sum_batch = 0.0;
        for (int64_t i = 0; i < n; i++) {
            sum_single += x_single[batch * n + i];
            sum_batch += x_batch[batch * n + i];
        }

        std::cout << "Batch " << batch << ": single sum=" << sum_single
                  << ", batched sum=" << sum_batch << std::endl;

        EXPECT_NEAR(sum_batch, 1.0, 0.1)
            << "Batched solve batch " << batch << " failed: sum=" << sum_batch;
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test with equilibration disabled to confirm it's the cause
TEST_F(BatchBVectorsTest, LargeProblemNoEquilibration) {
    int64_t n = 200;
    int64_t batchSize = 8;

    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    int64_t m, nnzP, nnzA;
    buildSimplexProblem(n, P_ro, P_ci, A_ro, A_ci, m, nnzP, nnzA);

    double eps = 1e-4;

    std::vector<double> P_val(nnzP * batchSize);
    std::vector<double> A_val(nnzA * batchSize);
    std::vector<double> q(n * batchSize);
    std::vector<double> b(m * batchSize);

    std::mt19937 rng(42);

    for (int64_t batch = 0; batch < batchSize; batch++) {
        for (int64_t i = 0; i < nnzP; i++) P_val[batch * nnzP + i] = 0.01;
        for (int64_t i = 0; i < n; i++) A_val[batch * nnzA + i] = 1.0;
        for (int64_t i = 0; i < n; i++) A_val[batch * nnzA + n + i] = -1.0;

        std::uniform_real_distribution<double> dist(-0.1, 0.1);
        for (int64_t i = 0; i < n; i++) q[batch * n + i] = dist(rng);

        b[batch * m + 0] = 1.0;
        for (int64_t i = 1; i <= n; i++) b[batch * m + i] = -eps;

        int numConstrained = 3;  // fixed small number
        std::uniform_int_distribution<int64_t> idx_dist(0, n - 1);
        std::uniform_real_distribution<double> bound_dist(0.01, 0.05);

        std::set<int64_t> constrained_vars;
        while (constrained_vars.size() < static_cast<size_t>(numConstrained)) {
            constrained_vars.insert(idx_dist(rng));
        }
        for (int64_t var_idx : constrained_vars) {
            double min_val = bound_dist(rng);
            b[batch * m + 1 + var_idx] = -min_val;
        }
    }

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = n;

    Settings settings;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = false;  // DISABLE EQUILIBRATION

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

    int failures = 0;
    for (int64_t batch = 0; batch < batchSize; batch++) {
        double sum = 0.0;
        for (int64_t i = 0; i < n; i++) sum += x_sol[batch * n + i];

        bool solved = (status[batch] == static_cast<int32_t>(SolverStatus::Solved) ||
                       status[batch] == static_cast<int32_t>(SolverStatus::AlmostSolved));
        bool sum_ok = std::abs(sum - 1.0) < 0.1;

        if (!solved || !sum_ok) {
            failures++;
            std::cout << "[NoEquil] Batch " << batch << ": status=" << status[batch]
                      << ", sum(x)=" << sum << std::endl;
        }

        EXPECT_TRUE(solved) << "Batch " << batch << " did not solve (no equilibration)";
        EXPECT_NEAR(sum, 1.0, 0.1) << "Batch " << batch << " sum(x) = " << sum << " (no equilibration)";
    }

    std::cout << "[NoEquil] Failures: " << failures << "/" << batchSize << std::endl;
    EXPECT_EQ(failures, 0) << "With equilibration disabled, no failures expected";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
