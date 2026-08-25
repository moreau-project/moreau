// test_empty_A_matrix.cpp
// Tests that verify the solver properly handles empty A matrices
//
// Summary of behavior after fixes:
// =================================
// The solver now supports empty A matrices naturally through the interior-point method:
//
// 1. m=0 (unconstrained optimization):
//    - Problem: minimize (1/2)x'Px + q'x
//    - Works: CSR matrix allows m=0, SpMV kernels handle nnz=0
//
// 2. m>0, nnzA=0 (constraints with empty A):
//    - Constraints become: 0·x + s = b → s = b
//    - Problem: minimize (1/2)x'Px + q'x subject to s=b, s∈K
//    - Works: Solver handles this naturally, KKT system becomes block diagonal
//
// 3. Both P and A empty (LP with fixed constraints):
//    - Problem: minimize q'x subject to s=b, s∈K
//    - Works: Both matrices can be empty
//
// Implementation fixes applied:
// - SpMV kernels: Early return when nnz=0 (prevents invalid grid dimensions)
// - CSR constructor: Allow m=0 (changed m<=0 to m<0 validation)
// - allocateAndCopyRowOfToGPU: Handle empty vectors (return nullptr)
//
#include <gtest/gtest.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include <vector>
#include <iostream>
#include <algorithm>

using namespace moreau;

class EmptyAMatrixTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test case 1: m = 0 (no constraints at all)
// This represents an unconstrained optimization problem
TEST_F(EmptyAMatrixTest, ZeroConstraints) {
    // Problem: minimize (1/2)x'Px + q'x with no constraints
    // Analytical solution: x* = -P^{-1}q = -[1 0; 0 0.5] * [-1; -1] = [1, 0.5]
    int n = 2;   // 2 variables
    int m = 0;   // 0 constraints
    int batchSize = 1;

    // P is 2x2 diagonal: diag([1, 2])
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<double> P_values = {1.0, 2.0};

    // q = [-1.0, -1.0]
    std::vector<double> q_data = {-1.0, -1.0};

    // A is 0x2 (no rows)
    std::vector<int64_t> A_ro = {0};  // Only one entry for m+1 = 0+1 = 1
    std::vector<int64_t> A_ci = {};   // No column indices
    int64_t nnzA = 0;                 // No nonzeros
    std::vector<double> A_values = {};

    // b vector (empty)
    std::vector<double> b_data = {};

    // No cones (all zero)
    Cones cones{};
    cones.numZeroCones = 0;
    cones.numNonnegCones = 0;

    // Solver settings
    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    // Create solver - should now succeed with fixed CSR constructor
    std::cout << "\n=== Testing m=0 (unconstrained optimization) ===\n";
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * std::max<int64_t>(nnzA, 1) * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * std::max<int64_t>(m, 1) * batchSize);

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Get solution
    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::cout << "Solution: x = [" << x_sol[0] << ", " << x_sol[1] << "]\n";
    std::cout << "Expected: x = [1.0, 0.5]\n";

    // IPM solves to default tolerance (1e-8)
    EXPECT_NEAR(x_sol[0], 1.0, 1e-8);
    EXPECT_NEAR(x_sol[1], 0.5, 1e-8);

    // Verify the solution satisfies optimality: Px + q = 0
    // P = diag([1, 2]), so Px + q = [x[0] - 1, 2*x[1] - 1]
    double grad_0 = 1.0 * x_sol[0] - 1.0;
    double grad_1 = 2.0 * x_sol[1] - 1.0;
    EXPECT_NEAR(grad_0, 0.0, 1e-8) << "Gradient component 0 should be zero";
    EXPECT_NEAR(grad_1, 0.0, 1e-8) << "Gradient component 1 should be zero";

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test case 2: m > 0 but nnzA = 0 (constraints exist but A matrix is empty)
// This is mathematically invalid - we have constraints Ax + s = b but A has no entries
TEST_F(EmptyAMatrixTest, ConstraintsWithEmptyA) {
    // Problem dimensions
    int n = 2;   // 2 variables
    int m = 3;   // 3 constraints, but A will be empty
    int batchSize = 1;

    // P is 2x2 diagonal: diag([1, 2])
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<double> P_values = {1.0, 2.0};

    // q = [-1.0, -1.0]
    std::vector<double> q_data = {-1.0, -1.0};

    // A is 3x2 but completely empty (all zeros)
    std::vector<int64_t> A_ro = {0, 0, 0, 0};  // m+1 = 4 entries, all pointing to index 0
    std::vector<int64_t> A_ci = {};            // No column indices
    int64_t nnzA = 0;                          // No nonzeros
    std::vector<double> A_values = {};

    // b vector
    std::vector<double> b_data = {1.0, 1.0, 1.0};

    // Cone structure: 3 nonneg cones
    Cones cones{};
    cones.numZeroCones = 0;
    cones.numNonnegCones = 3;

    // Solver settings
    Settings settings;
    settings.maxIter = 10;
    settings.verbose = false;  // Less verbose to see errors more clearly

    // Construction succeeds, but try to solve
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * std::max<int64_t>(nnzA, 1) * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    // This demonstrates the error: solve() will fail with empty A
    std::cout << "\n=== Attempting to solve with empty A matrix (m>0, nnzA=0) ===\n";
    ASSERT_NO_THROW(solver.solveAll(d_P_values, d_A_values, d_q, d_b));

    // s should match b because constraints reduce to s = b
    std::vector<double> s_host(m);
    cudaMemcpy(s_host.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < m; ++i) {
        EXPECT_NEAR(s_host[i], b_data[i], 1e-6);
    }

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test case 3: Attempt to solve with empty A (if construction doesn't fail)
// This tests the solve() method behavior with empty A
TEST_F(EmptyAMatrixTest, SolveWithEmptyA) {
    // Similar to ConstraintsWithEmptyA, but we also try to solve
    int n = 2;
    int m = 2;
    int batchSize = 1;

    // P is 2x2 diagonal
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<double> P_values = {1.0, 2.0};

    // q vector
    std::vector<double> q_data = {-1.0, -1.0};

    // A is empty but claims to be 2x2
    std::vector<int64_t> A_ro = {0, 0, 0};  // m+1 = 3 entries
    std::vector<int64_t> A_ci = {};
    int64_t nnzA = 0;
    std::vector<double> A_values = {};

    // b vector
    std::vector<double> b_data = {1.0, 1.0};

    // Cone structure
    Cones cones{};
    cones.numNonnegCones = 2;

    Settings settings;
    settings.maxIter = 10;
    settings.verbose = false;  // Less verbose for cleaner test output

    // Try to create and solve
    bool construction_succeeded = false;
    CompiledSolver* solver_ptr = nullptr;

    try {
        solver_ptr = new CompiledSolver(
            n, m, batchSize,
            P_ro.data(), P_ci.data(), nnzP,
            A_ro.data(), A_ci.data(), nnzA,
            cones,
            settings
        );
        construction_succeeded = true;

        // If construction succeeded, try to solve
        // Allocate device memory
        double *d_P_values, *d_A_values, *d_q, *d_b;
        cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
        cudaMalloc(&d_A_values, sizeof(double) * std::max<int64_t>(nnzA, 1) * batchSize);
        cudaMalloc(&d_q, sizeof(double) * n * batchSize);
        cudaMalloc(&d_b, sizeof(double) * m * batchSize);

        // Copy to device
        cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

        // Attempt to solve - should succeed even with empty A
        ASSERT_NO_THROW({
            solver_ptr->solveAll(d_P_values, d_A_values, d_q, d_b);
        }) << "Solver should handle empty A matrix without throwing";

        std::vector<double> s_host(m);
        cudaMemcpy(s_host.data(), solver_ptr->solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (int i = 0; i < m; ++i) {
            EXPECT_NEAR(s_host[i], b_data[i], 1e-6);
        }

        // Clean up
        cudaFree(d_P_values);
        cudaFree(d_A_values);
        cudaFree(d_q);
        cudaFree(d_b);

        delete solver_ptr;
    } catch (const std::exception& e) {
        std::cout << "Construction or solve threw unexpectedly: " << e.what() << std::endl;
        FAIL() << "Solver should support empty A without throwing";
        if (solver_ptr) delete solver_ptr;
    }
}

// Test case 4: LP (empty P) with empty A, m=2 nonneg cones.
//
// Problem:  min -x_0 - x_1  s.t.  0·x + s = b = [1, 1],  s ∈ R+²
// s is determined by feasibility (s = [1, 1]), x is free → objective → -∞.
// So the problem is UNBOUNDED / dual-infeasible, and the correct answer
// is SolverStatus::DualInfeasible, not a feasible point with s = b.
//
// Previously DISABLED with the wrong expectation (s ≈ b, which is what a
// feasible solution would satisfy but this problem has no feasible optimum).
// Both CPU and CUDA correctly classify this as DualInfeasible.
TEST_F(EmptyAMatrixTest, EmptyPAndEmptyA) {
    int n = 2;
    int m = 2;
    int batchSize = 1;

    // P is empty (LP formulation)
    std::vector<int64_t> P_ro = {0, 0, 0};  // n+1 = 3 entries, all zeros
    std::vector<int64_t> P_ci = {};
    int64_t nnzP = 0;
    std::vector<double> P_values = {};

    // q vector
    std::vector<double> q_data = {-1.0, -1.0};

    // A is also empty
    std::vector<int64_t> A_ro = {0, 0, 0};  // m+1 = 3 entries
    std::vector<int64_t> A_ci = {};
    int64_t nnzA = 0;
    std::vector<double> A_values = {};

    // b vector
    std::vector<double> b_data = {1.0, 1.0};

    // Cone structure
    Cones cones{};
    cones.numNonnegCones = 2;

    Settings settings;
    settings.maxIter = 10;
    settings.verbose = false;

    // Construction succeeds and solving should also succeed
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * std::max<int64_t>(nnzP, 1) * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * std::max<int64_t>(nnzA, 1) * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    // Copy to device
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    ASSERT_NO_THROW(solver.solveAll(d_P_values, d_A_values, d_q, d_b));

    std::vector<int32_t> st(1);
    cudaMemcpy(st.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    EXPECT_EQ(static_cast<SolverStatus>(st[0]), SolverStatus::DualInfeasible);

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test case 5: Single zero cone with empty A
// Tests edge case with just zero cones
TEST_F(EmptyAMatrixTest, ZeroConesWithEmptyA) {
    int n = 2;
    int m = 2;
    int batchSize = 1;

    // P is 2x2 identity
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<double> P_values = {1.0, 1.0};

    // q vector
    std::vector<double> q_data = {0.0, 0.0};

    // A is empty
    std::vector<int64_t> A_ro = {0, 0, 0};
    std::vector<int64_t> A_ci = {};
    int64_t nnzA = 0;
    std::vector<double> A_values = {};

    // b vector
    std::vector<double> b_data = {0.0, 0.0};

    // Cone structure: 2 zero cones
    Cones cones{};
    cones.numZeroCones = 2;

    Settings settings;
    settings.maxIter = 10;
    settings.verbose = false;

    // Construction succeeds and solving should also succeed
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * std::max<int64_t>(nnzA, 1) * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    // This demonstrates the error: zero cones with empty A
    std::cout << "\n=== Attempting to solve with zero cones and empty A ===\n";
    ASSERT_NO_THROW(solver.solveAll(d_P_values, d_A_values, d_q, d_b));

    std::vector<double> s_host(m);
    cudaMemcpy(s_host.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (int i = 0; i < m; ++i) {
        EXPECT_NEAR(s_host[i], b_data[i], 1e-6);
    }

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test case 6: m=0 with P empty and q=0 (zero objective)
// This should return Solved (any x is optimal, infinite solutions).
// Previously DISABLED with a note about "returns Unknown status" — that was
// caused by reading from the now-removed CompiledSolver::status field which
// held doubles but received int32 bytes. Fixed by removing the field and
// routing all status reads through solver.solution.status (int32_t).
TEST_F(EmptyAMatrixTest, ZeroObjectiveUnconstrained) {
    int n = 2;
    int m = 0;
    int batchSize = 1;

    // P is empty (LP formulation)
    std::vector<int64_t> P_ro = {0, 0, 0};  // n+1 = 3 entries, all zeros
    std::vector<int64_t> P_ci = {};
    int64_t nnzP = 0;
    std::vector<double> P_values = {};

    // q vector is ZERO
    std::vector<double> q_data = {0.0, 0.0};

    // A is empty (no constraints)
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};
    int64_t nnzA = 0;
    std::vector<double> A_values = {};

    // b vector (empty)
    std::vector<double> b_data = {};

    // No cones
    Cones cones{};
    cones.numZeroCones = 0;
    cones.numNonnegCones = 0;

    // Solver settings
    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    std::cout << "\n=== Testing m=0, P=0, q=0 (zero objective, unconstrained) ===\n";
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * std::max<int64_t>(nnzP, 1) * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * std::max<int64_t>(nnzA, 1) * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * std::max<int64_t>(m, 1) * batchSize);

    // Copy to device
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Get solution and status
    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    std::vector<int32_t> status_host(batchSize);
    cudaMemcpy(status_host.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::cout << "Solution: x = [" << x_sol[0] << ", " << x_sol[1] << "]\n";
    std::cout << "Status: " << status_host[0]
              << " (1=Solved, 21=DualInfeasible)\n";

    // Should return Solved status (any x works, we return x=0)
    EXPECT_EQ(static_cast<SolverStatus>(status_host[0]), SolverStatus::Solved);

    // x should be 0 (our chosen solution from infinite possibilities)
    EXPECT_NEAR(x_sol[0], 0.0, 1e-10);
    EXPECT_NEAR(x_sol[1], 0.0, 1e-10);

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test case 7: m=0 with batching (batchSize > 1)
// Verifies that cuDSS uniform batching works correctly
TEST_F(EmptyAMatrixTest, UnconstrainedBatched) {
    int n = 2;
    int m = 0;
    int batchSize = 4;  // Test with multiple problems

    // P is 2x2 diagonal: diag([1, 2])
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<double> P_values = {1.0, 2.0};

    // Different q vectors for each batch
    // Batch 0: q = [-1, -1] → x = [1, 0.5]
    // Batch 1: q = [-2, -2] → x = [2, 1.0]
    // Batch 2: q = [-0.5, -1] → x = [0.5, 0.5]
    // Batch 3: q = [0, 0] → x = [0, 0]
    std::vector<double> q_data = {
        -1.0, -1.0,   // Batch 0
        -2.0, -2.0,   // Batch 1
        -0.5, -1.0,   // Batch 2
         0.0,  0.0    // Batch 3
    };

    // Expected solutions
    std::vector<double> expected_x = {
        1.0, 0.5,     // Batch 0
        2.0, 1.0,     // Batch 1
        0.5, 0.5,     // Batch 2
        0.0, 0.0      // Batch 3
    };

    // A is empty (no constraints)
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};
    int64_t nnzA = 0;
    std::vector<double> A_values = {};
    std::vector<double> b_data = {};

    // No cones
    Cones cones{};
    cones.numZeroCones = 0;
    cones.numNonnegCones = 0;

    // Solver settings
    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    std::cout << "\n=== Testing m=0 with batching (batchSize=" << batchSize << ") ===\n";
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * std::max<int64_t>(nnzA, 1) * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * std::max<int64_t>(m, 1) * batchSize);

    // Copy P values (replicated for each batch)
    std::vector<double> P_values_batched(nnzP * batchSize);
    for (int b = 0; b < batchSize; ++b) {
        for (int64_t i = 0; i < nnzP; ++i) {
            P_values_batched[b * nnzP + i] = P_values[i];
        }
    }
    cudaMemcpy(d_P_values, P_values_batched.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Get solutions
    std::vector<double> x_sol(n * batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Verify all batch solutions
    for (int b = 0; b < batchSize; ++b) {
        std::cout << "Batch " << b << ": x = [" << x_sol[b * n + 0] << ", " << x_sol[b * n + 1] << "], "
                  << "expected = [" << expected_x[b * n + 0] << ", " << expected_x[b * n + 1] << "]\n";

        EXPECT_NEAR(x_sol[b * n + 0], expected_x[b * n + 0], 1e-8)
            << "Batch " << b << " x[0] mismatch";
        EXPECT_NEAR(x_sol[b * n + 1], expected_x[b * n + 1], 1e-8)
            << "Batch " << b << " x[1] mismatch";
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
