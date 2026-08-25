// test_simple_qp.cpp
#include <gtest/gtest.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include <vector>
#include <iostream>
#include <iomanip>

using namespace moreau;

class SimpleQPTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SimpleQPTest, QPWithEqualityAndInequality) {
    // This QP has an equality constraint and inequality constraints:
    //
    //     minimize    (1/2) xᵗ P x + qᵗ x
    //     subject to  x₁ + x₂ = 1  (equality)
    //                 x₁ + s₁ = 2, s₁ >= 0  (i.e., x₁ <= 2)
    //                 x₂ + s₂ = 2, s₂ >= 0  (i.e., x₂ <= 2)
    //
    // This is feasible: x = [0, 1] satisfies all constraints.

    int n = 2;  // 2 variables
    int m = 3;  // 3 constraints
    int batchSize = 1;

    // P is 2x2 identity
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<double> P_values = {1.0, 1.0};

    // q = [2.0, 1.0]
    std::vector<double> q_data = {2.0, 1.0};

    // A matrix:
    // Row 0: x₁ + x₂ = 1   (equality, zero cone)
    // Row 1: x₁ + s₁ = 2   (inequality, s₁ >= 0)
    // Row 2: x₂ + s₂ = 2   (inequality, s₂ >= 0)
    std::vector<int64_t> A_ro = {0, 2, 3, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    int64_t nnzA = 4;
    std::vector<double> A_values = {1.0, 1.0, 1.0, 1.0};

    // b vector
    std::vector<double> b_data = {1.0, 2.0, 2.0};

    // Cone structure: 1 zero (equality), 2 nonneg (inequalities)
    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 10;  // Run enough iterations to converge
    settings.verbose = true;

    // Create solver
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

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Copy solution to host
    std::vector<double> x_sol(n * batchSize);
    std::vector<double> s_sol(m * batchSize);
    std::vector<double> z_sol(m * batchSize);
    std::vector<int32_t> status_sol(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Print solution
    std::cout << "\n=== MOREAU SOLUTION ===" << std::endl;
    std::cout << "x = [";
    for (int i = 0; i < n; i++) {
        std::cout << x_sol[i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "s = [";
    for (int i = 0; i < m; i++) {
        std::cout << s_sol[i];
        if (i < m - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "z = [";
    for (int i = 0; i < m; i++) {
        std::cout << z_sol[i];
        if (i < m - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Get tau and kappa from variables
    std::vector<double> tau_host(batchSize);
    std::vector<double> kappa_host(batchSize);
    solver.variables.τ.gpuToCpu(tau_host.data());
    solver.variables.κ.gpuToCpu(kappa_host.data());
    std::cout << "τ = " << tau_host[0] << std::endl;
    std::cout << "κ = " << kappa_host[0] << std::endl;

    // Get objective value
    std::vector<double> cost_primal(batchSize);
    solver.info.cost_primal.gpuToCpu(cost_primal.data());
    std::cout << "Optimal objective: " << cost_primal[0] << std::endl;
    std::cout << "Status: " << status_sol[0] << " (1=Solved, 2=AlmostSolved)" << std::endl;
    std::cout << "===========================\n" << std::endl;

    // Check that solver converged
    EXPECT_TRUE(status_sol[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status_sol[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Expected Solved (1) or AlmostSolved (2), got " << status_sol[0];

    // Expected solution from Clarabel (Rust):
    // x = [2.4308818538642494e-10, 0.9999999997569121]
    // s = [0.0, 1.9999999997569118, 1.0000000002430876]
    // z = [-2.0000000002479426, 3.2555674870730805e-11, 5.187299172411473e-10]
    // τ = 1.0
    // κ = 7.496023490295008e-11

    std::vector<double> expected_x = {2.4308818538642494e-10, 0.9999999997569121};
    std::vector<double> expected_s = {0.0, 1.9999999997569118, 1.0000000002430876};
    std::vector<double> expected_z = {-2.0000000002479426, 3.2555674870730805e-11, 5.187299172411473e-10};

    // Check x solution (primal variables)
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x_sol[i], expected_x[i], 1e-6) << "x[" << i << "] mismatch";
    }

    // Check s solution (slack variables)
    for (int i = 0; i < m; i++) {
        double tol = (std::abs(expected_s[i]) < 1e-6) ? 1e-6 : 1e-6;
        EXPECT_NEAR(s_sol[i], expected_s[i], tol) << "s[" << i << "] mismatch";
    }

    // Check z solution (dual variables)
    for (int i = 0; i < m; i++) {
        double tol = (std::abs(expected_z[i]) < 1e-6) ? 1e-6 : 1e-6;
        EXPECT_NEAR(z_sol[i], expected_z[i], tol) << "z[" << i << "] mismatch";
    }

    // Check tau and kappa
    // Note: tau may not be exactly 1.0 if we stop early; the important thing is that
    // the unscaled solution (x/tau, s/tau, z/tau) is correct
    EXPECT_GT(tau_host[0], 0.1) << "tau should be positive (feasible problem)";
    EXPECT_LT(kappa_host[0], 1e-6) << "kappa should be near zero (feasible problem)";

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
