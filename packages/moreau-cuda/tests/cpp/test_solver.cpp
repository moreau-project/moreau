// test_solver.cpp
#include <gtest/gtest.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/diff/diff.hpp"
#include <vector>
#include <cmath>

using namespace moreau;

class SolverTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SolverTest, ConstructorBasic) {
    int n = 3, m = 2;
    int batchSize = 2;

    // Simple diagonal P matrix
    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;

    // Simple identity A matrix
    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numNonnegCones = 2;

    // Construct solver
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones
    );

    EXPECT_EQ(solver.data.n, n);
    EXPECT_EQ(solver.data.m, m);
    EXPECT_EQ(solver.data.batchSize, batchSize);
}

TEST_F(SolverTest, SolveWithDevicePointers) {
    int n = 3, m = 2;
    int batchSize = 2;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;

    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numNonnegCones = 2;

    Settings settings;
    settings.maxIter = 5;
    settings.verbose = false;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory for problem data
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    // Initialize with test data on host
    std::vector<double> P_values(nnzP * batchSize, 1.0);
    std::vector<double> A_values(nnzA * batchSize, 1.0);
    std::vector<double> q_data(n * batchSize, 1.0);
    std::vector<double> b_data(m * batchSize, 2.0);

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Solver should either converge or hit max iterations
    EXPECT_LE(solver.solution.iterations, settings.maxIter);
    EXPECT_GT(solver.solution.solve_time, 0.0);
    EXPECT_NE(solver.solution.x.data(), nullptr);
    EXPECT_NE(solver.solution.z.data(), nullptr);
    EXPECT_NE(solver.solution.s.data(), nullptr);

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(SolverTest, MemoryUsage) {
    int n = 10, m = 5;
    int batchSize = 4;

    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci(n);
    for (int i = 0; i <= n; i++) P_ro[i] = i;
    for (int i = 0; i < n; i++) P_ci[i] = i;

    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci(m);
    for (int i = 0; i <= m; i++) A_ro[i] = i;
    for (int i = 0; i < m; i++) A_ci[i] = i;

    Cones cones{};
    cones.numNonnegCones = 5;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), n,
        A_ro.data(), A_ci.data(), m,
        cones
    );

    size_t mem = solver.memoryUsage();
    EXPECT_GT(mem, 0);

    std::cout << "Solver memory usage: " << mem << " bytes\n";
}

// Test backward pass for HSDE path (mixed zero + nonneg cones) with numerical gradient verification
TEST_F(SolverTest, BackwardHSDENumericalGradientDb) {
    int n = 2, m = 3;
    int batchSize = 1;

    // P = I (identity), stored in CSR as diagonal
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    // A = [[1, 1], [1, 0], [0, 1]] in CSR
    std::vector<int64_t> A_ro = {0, 2, 3, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    int64_t nnzA = 4;

    // Mixed cones: 1 zero (equality) + 2 nonneg (inequalities) - forces HSDE path
    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Problem data
    std::vector<double> P_values = {1.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> q = {2.0, 1.0};
    std::vector<double> b = {1.0, 2.0, 2.0};
    // Allocate device memory
    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // Solve (auto-caches solution for backward)
    solver.solveAll(d_P, d_A, d_q, d_b);
    solver.info.sync_status_to_host(0);
    EXPECT_TRUE(solver.info.status[0] == moreau::SolverStatus::Solved ||
                solver.info.status[0] == moreau::SolverStatus::AlmostSolved);

    // Get solution
    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    // Backward pass with dx_bar = [1, 1]
    BatchedVector dx_bar(n, batchSize);
    BatchedVector dz_bar(m, batchSize);
    BatchedVector ds_bar(m, batchSize);

    std::vector<double> dx_bar_data = {1.0, 1.0};
    std::vector<double> dz_bar_data(m, 0.0);
    std::vector<double> ds_bar_data(m, 0.0);
    cudaMemcpy(dx_bar.data(), dx_bar_data.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dz_bar.data(), dz_bar_data.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(ds_bar.data(), ds_bar_data.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);

    // Get analytical gradients
    std::vector<double> analytical_db(m);
    cudaMemcpy(analytical_db.data(), solver.diff_state()->db.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    // Compute numerical gradient for db via finite differences
    double eps = 1e-5;
    std::vector<double> numerical_db(m);

    for (int i = 0; i < m; i++) {
        // Perturb b[i] positively
        std::vector<double> b_plus = b;
        b_plus[i] += eps;

        CompiledSolver solver_plus(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                                   A_ro.data(), A_ci.data(), nnzA, cones, settings);
        double *d_b_plus;
        cudaMalloc(&d_b_plus, sizeof(double) * m);
        cudaMemcpy(d_b_plus, b_plus.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
        solver_plus.solveAll(d_P, d_A, d_q, d_b_plus);

        std::vector<double> x_plus(n);
        cudaMemcpy(x_plus.data(), solver_plus.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
        double loss_plus = x_plus[0] + x_plus[1];

        // Perturb b[i] negatively
        std::vector<double> b_minus = b;
        b_minus[i] -= eps;

        CompiledSolver solver_minus(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                                    A_ro.data(), A_ci.data(), nnzA, cones, settings);
        double *d_b_minus;
        cudaMalloc(&d_b_minus, sizeof(double) * m);
        cudaMemcpy(d_b_minus, b_minus.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
        solver_minus.solveAll(d_P, d_A, d_q, d_b_minus);

        std::vector<double> x_minus(n);
        cudaMemcpy(x_minus.data(), solver_minus.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
        double loss_minus = x_minus[0] + x_minus[1];

        numerical_db[i] = (loss_plus - loss_minus) / (2.0 * eps);

        cudaFree(d_b_plus);
        cudaFree(d_b_minus);
    }

    // Compare analytical vs numerical gradients
    double tol = 1e-4;
    for (int i = 0; i < m; i++) {
        double diff = std::abs(analytical_db[i] - numerical_db[i]);
        EXPECT_LT(diff, tol) << "db[" << i << "] mismatch: analytical=" << analytical_db[i]
                             << ", numerical=" << numerical_db[i] << ", diff=" << diff;
    }

    // Cleanup
    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test backward pass for dq gradient in HSDE path
TEST_F(SolverTest, BackwardHSDENumericalGradientDq) {
    int n = 2, m = 3;
    int batchSize = 1;

    // P = I (identity), stored in CSR as diagonal (same as db test)
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    // A = [[1, 1], [1, 0], [0, 1]] in CSR
    std::vector<int64_t> A_ro = {0, 2, 3, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    int64_t nnzA = 4;

    // Mixed cones: 1 zero (equality) + 2 nonneg (inequalities) - forces HSDE path
    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Use same problem data as db test (which we know converges)
    std::vector<double> P_values = {1.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0, 1.0, 1.0};
    std::vector<double> q = {2.0, 1.0};
    std::vector<double> b = {1.0, 2.0, 2.0};

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // Solve (auto-caches solution for backward)
    solver.solveAll(d_P, d_A, d_q, d_b);
    solver.info.sync_status_to_host(0);
    EXPECT_TRUE(solver.info.status[0] == moreau::SolverStatus::Solved ||
                solver.info.status[0] == moreau::SolverStatus::AlmostSolved);

    // Backward with dx_bar = [1, 1]
    BatchedVector dx_bar(n, batchSize);
    BatchedVector dz_bar(m, batchSize);
    BatchedVector ds_bar(m, batchSize);

    std::vector<double> dx_bar_data = {1.0, 1.0};
    std::vector<double> dz_bar_data(m, 0.0);
    std::vector<double> ds_bar_data(m, 0.0);
    cudaMemcpy(dx_bar.data(), dx_bar_data.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dz_bar.data(), dz_bar_data.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(ds_bar.data(), ds_bar_data.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);

    std::vector<double> analytical_dq(n);
    cudaMemcpy(analytical_dq.data(), solver.diff_state()->dq.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    // Numerical gradient for dq
    double eps = 1e-5;
    std::vector<double> numerical_dq(n);

    for (int i = 0; i < n; i++) {
        std::vector<double> q_plus = q;
        q_plus[i] += eps;

        CompiledSolver solver_plus(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                                   A_ro.data(), A_ci.data(), nnzA, cones, settings);
        double *d_q_plus;
        cudaMalloc(&d_q_plus, sizeof(double) * n);
        cudaMemcpy(d_q_plus, q_plus.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
        solver_plus.solveAll(d_P, d_A, d_q_plus, d_b);

        std::vector<double> x_plus(n);
        cudaMemcpy(x_plus.data(), solver_plus.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
        double loss_plus = x_plus[0] + x_plus[1];

        std::vector<double> q_minus = q;
        q_minus[i] -= eps;

        CompiledSolver solver_minus(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                                    A_ro.data(), A_ci.data(), nnzA, cones, settings);
        double *d_q_minus;
        cudaMalloc(&d_q_minus, sizeof(double) * n);
        cudaMemcpy(d_q_minus, q_minus.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
        solver_minus.solveAll(d_P, d_A, d_q_minus, d_b);

        std::vector<double> x_minus(n);
        cudaMemcpy(x_minus.data(), solver_minus.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
        double loss_minus = x_minus[0] + x_minus[1];

        numerical_dq[i] = (loss_plus - loss_minus) / (2.0 * eps);

        cudaFree(d_q_plus);
        cudaFree(d_q_minus);
    }

    double tol = 1e-4;
    for (int i = 0; i < n; i++) {
        double diff = std::abs(analytical_dq[i] - numerical_dq[i]);
        EXPECT_LT(diff, tol) << "dq[" << i << "] mismatch: analytical=" << analytical_dq[i]
                             << ", numerical=" << numerical_dq[i] << ", diff=" << diff;
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
