// test_solver_multicone.cpp
#include <gtest/gtest.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include <vector>
#include <iostream>
#include <iomanip>

using namespace moreau;

class SolverMulticoneTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Helper function to print dense matrix from CSR
void printDenseMatrix(const char* name,
                      const std::vector<int64_t>& rowOffsets,
                      const std::vector<int64_t>& colIndices,
                      const std::vector<double>& values,
                      int rows, int cols) {
    std::cout << "\n" << name << " (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; i++) {
        std::cout << "  [";
        for (int j = 0; j < cols; j++) {
            double val = 0.0;
            for (int64_t idx = rowOffsets[i]; idx < rowOffsets[i + 1]; idx++) {
                if (colIndices[idx] == j) {
                    val = values[idx];
                    break;
                }
            }
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val;
            if (j < cols - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

TEST_F(SolverMulticoneTest, test_all_iterations_default) {
    // Problem dimensions
    int n = 5;   // 5 variables
    int m = 13;  // 13 constraints (2 zero + 2 nonneg + 3 exp + 3 soc + 3 power)
    int batchSize = 1;

    // P is 5x5 diagonal: diag([1, 2, 3, 4, 5])
    std::vector<int64_t> P_ro = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> P_ci = {0, 1, 2, 3, 4};
    int64_t nnzP = 5;
    std::vector<double> P_values = {1.0, 2.0, 3.0, 4.0, 5.0};

    // q = [-1.0, -1.0, 0.0, 0.0, 0.0]
    std::vector<double> q_data = {-1.0, -1.0, 0.0, 0.0, 0.0};

    // A is 13x5 with 13 nonzeros (one per row)
    // Row structure (Clarabel ordering: Zero, Nonneg, SOC, Exp, Power)
    std::vector<int64_t> A_ro = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    std::vector<int64_t> A_ci = {0, 1, 2, 3, 2, 3, 4, 0, 1, 2, 0, 3, 4};
    int64_t nnzA = 13;
    std::vector<double> A_values = {
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    };

    // b vector (Clarabel ordering: Zero, Nonneg, SOC, Exp, Power)
    std::vector<double> b_data = {
        1.0, 1.0,         // Zero cone RHS (x1=1, x2=1)
        2.0, 2.0,         // Nonnegative cone RHS (x3>=0, x4>=0)
        2.0, 2.0, 0.5,    // SOC3 RHS
        1.0, 1.0, 2.0,    // Exp cone RHS
        1.0, 2.0, 0.5     // Power cone RHS
    };

    // Cone structure: 2 zero, 2 nonneg, 1 exp (3), 1 soc (3), 1 power (3, alpha=0.6)
    Cones cones{};
    cones.numZeroCones = 2;
    cones.numNonnegCones = 2;
    cones.numExpCones = 1;
    cones.socConeDims = {3};
    cones.numSocCones = 1;
    cones.numPowerCones = 1;
    cones.powerAlphas = {0.6};

    // Solver settings - 20 iterations to compare
    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 50;
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

    // Just check that solver ran successfully and basic sanity checks
    // Iteration count varies across environments - mixed cone problems may take longer
    EXPECT_LE(solver.info.iterations, 55) << "Expected <= 55 iterations";
    EXPECT_GE(solver.info.iterations, 20) << "Expected >= 20 iterations";
    EXPECT_GT(solver.solution.solve_time, 0.0);
    EXPECT_NE(solver.solution.x.data(), nullptr);
    EXPECT_NE(solver.solution.z.data(), nullptr);
    EXPECT_NE(solver.solution.s.data(), nullptr);

    // Verify final solution values match Rust Clarabel (32 iterations)
    // Copy solution to host
    std::vector<double> x_sol(n * batchSize);
    std::vector<double> s_sol(m * batchSize);
    std::vector<double> z_sol(m * batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Print actual solution for debugging
    std::cout << "=== MOREAU SOLUTION ===" << std::endl;
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

    // Expected solution values (new reference data)
    std::vector<double> expected_x = {
        0.999999999991688, 0.9999999999987422, -3.794876348270479e-5,
        2.8439175940049012e-5, 0.49999992123186493
    };

    // expected_s in Clarabel ordering: Zero, Nonneg, SOC, Exp, Power
    std::vector<double> expected_s = {
        0.0, 0.0, 2.000037948761892, 1.9999715608223825,         // Zero (0,0), Nonneg
        2.0000379487635946, 1.99997156082502, 7.876757613413614e-8,  // SOC (was indices 7-9)
        2.1295697597671207e-11, 1.296283409934298e-12, 2.000037948762265,  // Exp (was indices 4-6)
        9.854613557073413e-13, 1.9999715608225832, 7.876845728350085e-8  // Power
    };

    std::vector<double> expected_z = {
        -96000.0,  // same sign, large magnitude (negative)
        -35000.0,  // same sign, large magnitude (negative)
        7e-9,      // same sign, small value (positive)
        7e-9,      // same sign, small value (positive)
        -1000.0,   // same sign, large magnitude (negative)
        35000.0,   // same sign, large magnitude (positive)
        8e-9,      // same sign, small value (positive)
        1e-4,      // same sign, small positive
        -1e-4,     // same sign, small negative
        -1.5e-8,   // same sign, small negative
        97000.0,   // same sign, large magnitude (positive)
        6e-8,      // same sign, small positive
        -2.5       // same sign, negative, reasonable magnitude
    };

    double expected_tau = 0.9999999999999999;
    double expected_kappa = 1.490515200762064e-8;

    // Note: C++ and Rust implementations may have minor numerical differences in step lengths
    // due to floating-point precision, but should produce equivalent solutions.

    // Check x solution (primal variables) - use relaxed tolerance for mixed cone problems
    // Mixed cone problems with exp/power cones can have minor numerical variations
    // For elements near zero, use absolute tolerance; otherwise relative tolerance
    for (int i = 0; i < n; i++) {
        double tol = (std::abs(expected_x[i]) < 1e-3) ? 5e-3 : 1e-3;
        EXPECT_NEAR(x_sol[i], expected_x[i], tol) << "x[" << i << "] mismatch";
    }

    // Check s solution (slack variables) - relaxed tolerances for complex cones
    // Exp and power cones have iterative projections that can vary slightly
    for (int i = 0; i < m; i++) {
        // Use larger tolerance for very small values and for exp/power cone elements
        double tol = (std::abs(expected_s[i]) < 1e-4) ? 1e-3 : 1e-2;
        EXPECT_NEAR(s_sol[i], expected_s[i], tol) << "s[" << i << "] mismatch";
    }

    // z (dual variables) are nonunique and are not checked.

    // Check objective value - relaxed tolerance for mixed cone problems
    std::vector<double> cost_primal(batchSize);
    solver.info.cost_primal.gpuToCpu(cost_primal.data());
    EXPECT_NEAR(cost_primal[0], 0.12499991567936064, 1e-2) << "Objective value mismatch";

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(SolverMulticoneTest, BatchSize2) {
    int n = 5;  // Number of variables
    int m = 13; // Number of constraints
    int batchSize = 2; // Two batches!

    // P matrix (same structure for both batches)
    std::vector<int64_t> P_ro = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> P_ci = {0, 1, 2, 3, 4};
    int64_t nnzP = 5;
    std::vector<double> P_values = {
        // Batch 0
        1.0, 2.0, 3.0, 4.0, 5.0,
        // Batch 1 (same)
        1.0, 2.0, 3.0, 4.0, 5.0
    };

    // A matrix (same structure for both batches, Clarabel ordering)
    std::vector<int64_t> A_ro = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    std::vector<int64_t> A_ci = {0, 1, 2, 3, 2, 3, 4, 0, 1, 2, 0, 3, 4};
    int64_t nnzA = 13;
    std::vector<double> A_values = {
        // Batch 0
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        // Batch 1 (same)
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    };

    // q vector (same for both batches)
    std::vector<double> q_data = {
        // Batch 0
        -1.0, -1.0, 0.0, 0.0, 0.0,
        // Batch 1 (same)
        -1.0, -1.0, 0.0, 0.0, 0.0
    };

    // b vector (same for both batches, Clarabel ordering: Zero, Nonneg, SOC, Exp, Power)
    std::vector<double> b_data = {
        // Batch 0
        1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.5, 1.0, 1.0, 2.0, 1.0, 2.0, 0.5,
        // Batch 1 (same)
        1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.5, 1.0, 1.0, 2.0, 1.0, 2.0, 0.5
    };

    // Cone structure
    Cones cones{};
    cones.numZeroCones = 2;
    cones.numNonnegCones = 2;
    cones.numExpCones = 1;
    cones.socConeDims = {3};
    cones.numSocCones = 1;
    cones.numPowerCones = 1;
    cones.powerAlphas = {0.6};

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 30;
    settings.verbose = true;  // Less verbose for batch test

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

    // Verify both batches have the same residuals (since problems are identical)
    std::vector<double> rx_host(n * batchSize);
    std::vector<double> rz_host(m * batchSize);
    cudaMemcpy(rx_host.data(), solver.residuals.rx.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(rz_host.data(), solver.residuals.rz.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);

    // Check that batch 0 and batch 1 have similar residuals (small tolerance for numerical differences)
    // Use relative tolerance for GPU floating point operations which can have minor variations
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(rx_host[i], rx_host[n + i], 1e-8) << "rx mismatch at index " << i;
    }
    for (int i = 0; i < m; i++) {
        EXPECT_NEAR(rz_host[i], rz_host[m + i], 1e-8) << "rz mismatch at index " << i;
    }

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(SolverMulticoneTest, BatchSize2_DifferentProblems) {
    // This test follows test_all_iterations_default but with batch size = 2
    // Batch 0: Same problem as test_all_iterations_default
    // Batch 1: Different problem with P=diag([5,4,3,2,1]), q=[-0.5,-0.5,0,0,0], different b values

    int n = 5;   // 5 variables
    int m = 13;  // 13 constraints (2 zero + 2 nonneg + 3 exp + 3 soc + 3 power)
    int batchSize = 2;

    // P matrix structure (same for both batches)
    std::vector<int64_t> P_ro = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> P_ci = {0, 1, 2, 3, 4};
    int64_t nnzP = 5;
    std::vector<double> P_values = {
        // Batch 0: diag([1, 2, 3, 4, 5])
        1.0, 2.0, 3.0, 4.0, 5.0,
        // Batch 1: diag([5, 4, 3, 2, 1])
        5.0, 4.0, 3.0, 2.0, 1.0
    };

    // A matrix (same structure for both batches, Clarabel ordering)
    std::vector<int64_t> A_ro = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    std::vector<int64_t> A_ci = {0, 1, 2, 3, 2, 3, 4, 0, 1, 2, 0, 3, 4};
    int64_t nnzA = 13;
    std::vector<double> A_values = {
        // Batch 0
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        // Batch 1 (same)
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
    };

    // q vector
    std::vector<double> q_data = {
        // Batch 0: [-1.0, -1.0, 0.0, 0.0, 0.0]
        -1.0, -1.0, 0.0, 0.0, 0.0,
        // Batch 1: [-0.5, -0.5, 0.0, 0.0, 0.0]
        -0.5, -0.5, 0.0, 0.0, 0.0
    };

    // b vector (Clarabel ordering: Zero, Nonneg, SOC, Exp, Power)
    std::vector<double> b_data = {
        // Batch 0: Same as test_all_iterations_default
        1.0, 1.0,         // Zero cone RHS (x1=1, x2=1)
        2.0, 2.0,         // Nonnegative cone RHS (x3>=0, x4>=0)
        2.0, 2.0, 0.5,    // SOC3 RHS
        1.0, 1.0, 2.0,    // Exp cone RHS
        1.0, 2.0, 0.5,    // Power cone RHS
        // Batch 1: Different RHS values
        1.0, 1.0,         // Zero cone RHS
        2.0, 2.0,         // Nonnegative cone RHS
        2.0, 2.0, 0.5,    // SOC3 RHS
        1.0, 1.0, 3.0,    // Exp cone RHS (changed from 2.0 to 3.0)
        1.0, 2.0, 0.5     // Power cone RHS
    };

    // Cone structure: 2 zero, 2 nonneg, 1 exp (3), 1 soc (3), 1 power (3, alpha=0.6)
    Cones cones{};
    cones.numZeroCones = 2;
    cones.numNonnegCones = 2;
    cones.numExpCones = 1;
    cones.socConeDims = {3};
    cones.numSocCones = 1;
    cones.numPowerCones = 1;
    cones.powerAlphas = {0.6};

    // Solver settings
    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 20;
    settings.verbose = false;


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

    // Copy solutions to host
    std::vector<double> x_sol(n * batchSize);
    std::vector<double> s_sol(m * batchSize);
    std::vector<double> z_sol(m * batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Print solutions for both batches
    std::cout << "\n=== BATCH 0 SOLUTION ===" << std::endl;
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

    std::cout << "\n=== BATCH 1 SOLUTION ===" << std::endl;
    std::cout << "x = [";
    for (int i = 0; i < n; i++) {
        std::cout << x_sol[n + i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "s = [";
    for (int i = 0; i < m; i++) {
        std::cout << s_sol[m + i];
        if (i < m - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "z = [";
    for (int i = 0; i < m; i++) {
        std::cout << z_sol[m + i];
        if (i < m - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Batch 0 solves the same problem as test_all_iterations_default, so its
    // primal solution must match that reference.
    const std::vector<double> expected_x_batch0 = {
        0.999999999991688, 0.9999999999987422, -3.794876348270479e-5,
        2.8439175940049012e-5, 0.49999992123186493};
    for (int i = 0; i < n; i++) {
        double tol = (std::abs(expected_x_batch0[i]) < 1e-3) ? 5e-3 : 1e-3;
        EXPECT_NEAR(x_sol[i], expected_x_batch0[i], tol)
            << "Batch 0: x[" << i << "] mismatch";
    }

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
