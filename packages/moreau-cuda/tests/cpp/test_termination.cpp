// test_termination.cpp
// Tests for solver termination conditions
// Each test expects EXACTLY ONE status result.
//
// These tests verify the termination logic matches Clarabel.rs.
// Note: Some termination conditions (InsufficientProgress, NumericalError) are
// difficult to test deterministically because GPU and CPU numerical differences
// cause different convergence paths.
//
// Verified against Clarabel.rs and Moreau:
//   Solved:           Both return Solved
//   PrimalInfeasible: Both return PrimalInfeasible
//   DualInfeasible:   Both return DualInfeasible
//   MaxIterations:    Both return MaxIterations (with tight tols, max_iter=3)
//   MaxTime:          Both return MaxTime (with timeLimit=1e-10)
//
// SolverStatus enum values:
//   0  Unsolved, 1  Solved, 2  PrimalInfeasible, 3  DualInfeasible
//   4  AlmostSolved, 5  AlmostPrimalInfeasible, 6  AlmostDualInfeasible
//   7  MaxIterations, 8  MaxTime, 9  NumericalError
//   10 InsufficientProgress

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>
#include <iostream>

using namespace moreau;

// Helper class for termination tests
class TerminationTestHelper {
public:
    int64_t n, m, batchSize;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_val, A_val, q, b;
    Cones cones;
    Settings settings;

    std::vector<double> x_sol, s_sol, z_sol;
    std::vector<int32_t> status_sol;
    std::vector<double> obj_primal, obj_dual;
    int32_t iterations;

    TerminationTestHelper(int64_t n_, int64_t m_, int64_t batch_ = 1)
        : n(n_), m(m_), batchSize(batch_), iterations(0) {
        settings.verbose = false;
        settings.maxIter = 200;
        settings.ipm.equilibrationSettings.enable = true;
    }

    void solve() {
        int64_t nnzP = P_val.size() / batchSize;
        int64_t nnzA = A_val.size() / batchSize;

        CompiledSolver solver(n, m, batchSize,
                      P_ro.data(), P_ci.data(), nnzP,
                      A_ro.data(), A_ci.data(), nnzA,
                      cones, settings);

        double *d_P_values, *d_A_values, *d_q, *d_b;
        cudaMalloc(&d_P_values, sizeof(double) * std::max((int64_t)1, nnzP * batchSize));
        cudaMalloc(&d_A_values, sizeof(double) * std::max((int64_t)1, nnzA * batchSize));
        cudaMalloc(&d_q, sizeof(double) * n * batchSize);
        cudaMalloc(&d_b, sizeof(double) * m * batchSize);

        if (nnzP > 0) cudaMemcpy(d_P_values, P_val.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
        if (nnzA > 0) cudaMemcpy(d_A_values, A_val.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

        solver.solveAll(d_P_values, d_A_values, d_q, d_b);

        x_sol.resize(n * batchSize);
        s_sol.resize(m * batchSize);
        z_sol.resize(m * batchSize);
        status_sol.resize(batchSize);
        obj_primal.resize(batchSize);
        obj_dual.resize(batchSize);

        cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
        solver.info.cost_primal.gpuToCpu(obj_primal.data());
        solver.info.cost_dual.gpuToCpu(obj_dual.data());
        iterations = solver.info.iterations;

        cudaDeviceSynchronize();
        cudaFree(d_P_values);
        cudaFree(d_A_values);
        cudaFree(d_q);
        cudaFree(d_b);
    }

    SolverStatus getStatus(int batch = 0) const {
        return static_cast<SolverStatus>(status_sol[batch]);
    }
};

// =============================================================================
// Test 1: Solved
// Clarabel.rs: Solved, x=[-1.0], obj=-0.5
// Moreau:      Solved, x=[-1.0], obj=-0.5
// =============================================================================

TEST(TerminationTest, Solved) {
    cudaGetLastError();
    TerminationTestHelper h(1, 1);

    h.P_ro = {0, 1};
    h.P_ci = {0};
    h.P_val = {1.0};

    h.A_ro = {0, 1};
    h.A_ci = {0};
    h.A_val = {1.0};

    h.q = {1.0};
    h.b = {10.0};

    h.cones.numNonnegCones = 1;
    h.solve();

    EXPECT_EQ(h.getStatus(0), SolverStatus::Solved);
    EXPECT_NEAR(h.x_sol[0], -1.0, 1e-4);
    EXPECT_NEAR(h.obj_primal[0], -0.5, 1e-4);
}

// =============================================================================
// Test 2: PrimalInfeasible
// Clarabel.rs: PrimalInfeasible
// Moreau:      PrimalInfeasible
// =============================================================================

TEST(TerminationTest, PrimalInfeasible) {
    cudaGetLastError();
    TerminationTestHelper h(1, 2);

    h.P_ro = {0, 1};
    h.P_ci = {0};
    h.P_val = {1.0};

    h.A_ro = {0, 1, 2};
    h.A_ci = {0, 0};
    h.A_val = {-1.0, 1.0};

    h.q = {0.0};
    h.b = {-2.0, 1.0};  // x >= 2 AND x <= 1 (impossible)

    h.cones.numNonnegCones = 2;
    h.solve();

    EXPECT_EQ(h.getStatus(0), SolverStatus::PrimalInfeasible);
}

// =============================================================================
// Test 3: DualInfeasible
// Clarabel.rs: DualInfeasible
// Moreau:      DualInfeasible
// =============================================================================

TEST(TerminationTest, DualInfeasible) {
    cudaGetLastError();
    TerminationTestHelper h(1, 1);

    h.P_ro = {0, 0};
    h.P_ci = {};
    h.P_val = {};

    h.A_ro = {0, 1};
    h.A_ci = {0};
    h.A_val = {-1.0};

    h.q = {-1.0};  // min -x s.t. x >= 0 (unbounded)
    h.b = {0.0};

    h.cones.numNonnegCones = 1;
    h.solve();

    EXPECT_EQ(h.getStatus(0), SolverStatus::DualInfeasible);
}

// =============================================================================
// Test 4: MaxIterations
// Clarabel.rs: MaxIterations (iterations=3)
// Moreau:      MaxIterations (iterations=3)
// =============================================================================

TEST(TerminationTest, MaxIterations) {
    cudaGetLastError();
    TerminationTestHelper h(2, 4);

    h.P_ro = {0, 2, 4};
    h.P_ci = {0, 1, 0, 1};
    h.P_val = {4.0, 1.0, 1.0, 2.0};

    h.A_ro = {0, 2, 3, 4, 6};
    h.A_ci = {0, 1, 0, 1, 0, 1};
    h.A_val = {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0};

    h.q = {1.0, 1.0};
    h.b = {-1.0, 0.0, 0.0, 1.0};

    h.cones.numNonnegCones = 4;

    h.settings.maxIter = 3;
    h.settings.ipm.tolGapAbs = 1e-15;
    h.settings.ipm.tolGapRel = 1e-15;
    h.settings.ipm.tolFeas = 1e-15;

    h.solve();

    EXPECT_EQ(h.getStatus(0), SolverStatus::MaxIterations);
}

// =============================================================================
// Test 5: MaxTime
// Clarabel.rs: MaxTime
// Moreau:      MaxTime
// =============================================================================

TEST(TerminationTest, MaxTime) {
    cudaGetLastError();
    TerminationTestHelper h(2, 4);

    h.P_ro = {0, 2, 4};
    h.P_ci = {0, 1, 0, 1};
    h.P_val = {4.0, 1.0, 1.0, 2.0};

    h.A_ro = {0, 2, 3, 4, 6};
    h.A_ci = {0, 1, 0, 1, 0, 1};
    h.A_val = {-1.0, -1.0, -1.0, -1.0, 1.0, 1.0};

    h.q = {1.0, 1.0};
    h.b = {-1.0, 0.0, 0.0, 1.0};

    h.cones.numNonnegCones = 4;

    h.settings.timeLimit = 1e-10;  // Essentially zero

    h.solve();

    EXPECT_EQ(h.getStatus(0), SolverStatus::MaxTime);
}

// =============================================================================
// Test 6: Batched Termination (Different status per batch)
// Tests that solve-one-all correctly handles batches with different outcomes
// =============================================================================

TEST(TerminationTest, BatchedDifferentStatus) {
    cudaGetLastError();
    TerminationTestHelper h(1, 2, 2);  // batch size = 2

    h.P_ro = {0, 1};
    h.P_ci = {0};
    h.P_val = {1.0, 1.0};

    h.A_ro = {0, 1, 2};
    h.A_ci = {0, 0};
    h.A_val = {-1.0, 1.0, -1.0, 1.0};

    h.q = {0.0, 0.0};
    h.b = {0.0, 2.0,    // Batch 0: x >= 0, x <= 2 (feasible)
           -2.0, 1.0};  // Batch 1: x >= 2, x <= 1 (infeasible)

    h.cones.numNonnegCones = 2;
    h.solve();

    EXPECT_EQ(h.getStatus(0), SolverStatus::Solved);
    EXPECT_EQ(h.getStatus(1), SolverStatus::PrimalInfeasible);
}

// =============================================================================
// Test 7: Status Enum Values Match Clarabel.rs
// =============================================================================

TEST(TerminationTest, StatusEnumValues) {
    EXPECT_EQ(static_cast<int>(SolverStatus::Unsolved), 0);
    EXPECT_EQ(static_cast<int>(SolverStatus::Solved), 1);
    EXPECT_EQ(static_cast<int>(SolverStatus::PrimalInfeasible), 2);
    EXPECT_EQ(static_cast<int>(SolverStatus::DualInfeasible), 3);
    EXPECT_EQ(static_cast<int>(SolverStatus::AlmostSolved), 4);
    EXPECT_EQ(static_cast<int>(SolverStatus::AlmostPrimalInfeasible), 5);
    EXPECT_EQ(static_cast<int>(SolverStatus::AlmostDualInfeasible), 6);
    EXPECT_EQ(static_cast<int>(SolverStatus::MaxIterations), 7);
    EXPECT_EQ(static_cast<int>(SolverStatus::MaxTime), 8);
    EXPECT_EQ(static_cast<int>(SolverStatus::NumericalError), 9);
    EXPECT_EQ(static_cast<int>(SolverStatus::InsufficientProgress), 10);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
