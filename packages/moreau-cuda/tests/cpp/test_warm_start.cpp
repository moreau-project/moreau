// test_warm_start.cpp
// Tests for warm starting functionality in the GPU solver.
//
// Coverage:
// 1. Basic warm start reduces iterations (FewerIterations)
// 2. Warm start with perturbed problem data (PerturbedProblem)
// 3. Warm start with SOC cones (SOCCone)
// 4. Bad warm start still converges (BadPointStillConverges)
// 5. Warm start with batched problems (Batched)
// 6. Warm start solution matches cold start (verified in FewerIterations, PerturbedProblem, SOCCone)

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "clarabel_test_helpers.hpp"
#include <vector>
#include <cmath>

using namespace moreau;
using namespace clarabel_test;

class WarmStartTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetLastError();
    }
    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// =============================================================================
// Helpers
// =============================================================================

struct SimpleQP {
    int64_t n = 2, m = 5;
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<int64_t> A_ro = {0, 2, 3, 4, 5, 6};
    std::vector<int64_t> A_ci = {0, 1, 0, 1, 0, 1};
    int64_t nnzA = 6;
    Cones cones{};

    SimpleQP() {
        cones.numZeroCones = 1;
        cones.numNonnegCones = 4;
    }
};

struct SOCProblem {
    int64_t n = 3, m = 3;
    std::vector<int64_t> P_ro = {0, 0, 0, 0};
    std::vector<int64_t> P_ci = {};
    int64_t nnzP = 0;
    std::vector<int64_t> A_ro = {0, 1, 2, 3};
    std::vector<int64_t> A_ci = {0, 1, 2};
    int64_t nnzA = 3;
    Cones cones{};

    SOCProblem() {
        cones.socConeDims = {3};
    }
};

struct SOCVarLenProblem {
    int64_t n = 5, m = 5;
    std::vector<int64_t> P_ro = {0, 0, 0, 0, 0, 0};
    std::vector<int64_t> P_ci = {};
    int64_t nnzP = 0;
    std::vector<int64_t> A_ro = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> A_ci = {0, 1, 2, 3, 4};
    int64_t nnzA = 5;
    Cones cones{};

    SOCVarLenProblem() {
        cones.socConeDims = {5};
        cones.numSocCones = 1;
    }
};

// Helper to allocate + copy to GPU
static double* toDevice(const std::vector<double>& host) {
    double* d_ptr = nullptr;
    if (host.empty()) return d_ptr;
    cudaMalloc(&d_ptr, sizeof(double) * host.size());
    cudaMemcpy(d_ptr, host.data(), sizeof(double) * host.size(), cudaMemcpyHostToDevice);
    return d_ptr;
}

// Helper to copy from GPU to host
static std::vector<double> toHost(const double* d_ptr, size_t count) {
    std::vector<double> host(count);
    cudaMemcpy(host.data(), d_ptr, sizeof(double) * count, cudaMemcpyDeviceToHost);
    return host;
}

// =============================================================================
// Test: Warm start from cold solution converges in fewer iterations
// =============================================================================

TEST_F(WarmStartTest, FewerIterations) {
    SimpleQP prob;
    int64_t batchSize = 1;

    // P_val, A_val, q, b for batch=1
    std::vector<double> P_val = {2.0, 2.0};
    std::vector<double> A_val = {1.0, 1.0, -1.0, -1.0, 1.0, 1.0};
    std::vector<double> q = {1.0, -1.0};
    std::vector<double> b = {1.0, 0.0, 0.0, 1.0, 1.0};

    Settings settings;
    settings.verbose = false;

    // --- Cold solve ---
    CompiledSolver solver_cold(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_P = toDevice(P_val);
    double* d_A = toDevice(A_val);
    double* d_q = toDevice(q);
    double* d_b = toDevice(b);

    solver_cold.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status_cold(batchSize);
    cudaMemcpy(status_cold.data(), solver_cold.solution.status.get(),
               sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    ASSERT_EQ(status_cold[0], static_cast<int32_t>(SolverStatus::Solved));

    int32_t iters_cold = solver_cold.info.iterations;

    // Retrieve cold solution (x, z, s)
    auto x_cold = toHost(solver_cold.solution.x.data(), prob.n);
    auto z_cold = toHost(solver_cold.solution.z.data(), prob.m);
    auto s_cold = toHost(solver_cold.solution.s.data(), prob.m);

    // --- Warm solve ---
    CompiledSolver solver_warm(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    // setup then solve with warm start
    double* d_P2 = toDevice(P_val);
    double* d_A2 = toDevice(A_val);
    solver_warm.setup(d_P2, d_A2);

    double* d_warm_x = toDevice(x_cold);
    double* d_warm_z = toDevice(z_cold);
    double* d_warm_s = toDevice(s_cold);

    solver_warm.solve(d_q, d_b, d_warm_x, d_warm_z, d_warm_s);

    std::vector<int32_t> status_warm(batchSize);
    cudaMemcpy(status_warm.data(), solver_warm.solution.status.get(),
               sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    ASSERT_EQ(status_warm[0], static_cast<int32_t>(SolverStatus::Solved));

    int32_t iters_warm = solver_warm.info.iterations;

    EXPECT_LT(iters_warm, iters_cold)
        << "Warm start (" << iters_warm << " iters) should be faster than cold ("
        << iters_cold << " iters)";

    // Solutions should match
    auto x_warm = toHost(solver_warm.solution.x.data(), prob.n);
    for (int64_t i = 0; i < prob.n; i++) {
        EXPECT_NEAR(x_warm[i], x_cold[i], 1e-4) << "x[" << i << "]";
    }

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    cudaFree(d_P2); cudaFree(d_A2);
    cudaFree(d_warm_x); cudaFree(d_warm_z); cudaFree(d_warm_s);
}

// =============================================================================
// Test: Warm start from perturbed problem
// =============================================================================

TEST_F(WarmStartTest, PerturbedProblem) {
    SimpleQP prob;
    int64_t batchSize = 1;

    std::vector<double> P_val = {2.0, 2.0};
    std::vector<double> A_val = {1.0, 1.0, -1.0, -1.0, 1.0, 1.0};
    std::vector<double> q1 = {1.0, -1.0};
    std::vector<double> q2 = {1.1, -0.9};  // small perturbation
    std::vector<double> b = {1.0, 0.0, 0.0, 1.0, 1.0};

    Settings settings;
    settings.verbose = false;

    // Solve P1 cold
    CompiledSolver solver1(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_P = toDevice(P_val);
    double* d_A = toDevice(A_val);
    double* d_q1 = toDevice(q1);
    double* d_q2 = toDevice(q2);
    double* d_b = toDevice(b);

    solver1.solveAll(d_P, d_A, d_q1, d_b);

    std::vector<int32_t> status1(1);
    cudaMemcpy(status1.data(), solver1.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(status1[0], static_cast<int32_t>(SolverStatus::Solved));

    auto x1 = toHost(solver1.solution.x.data(), prob.n);
    auto z1 = toHost(solver1.solution.z.data(), prob.m);
    auto s1 = toHost(solver1.solution.s.data(), prob.m);

    // Solve P2 cold
    CompiledSolver solver2_cold(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_P3 = toDevice(P_val);
    double* d_A3 = toDevice(A_val);
    solver2_cold.solveAll(d_P3, d_A3, d_q2, d_b);

    std::vector<int32_t> status2c(1);
    cudaMemcpy(status2c.data(), solver2_cold.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(status2c[0], static_cast<int32_t>(SolverStatus::Solved));

    auto x2_cold = toHost(solver2_cold.solution.x.data(), prob.n);

    // Solve P2 warm from P1
    CompiledSolver solver2_warm(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_P4 = toDevice(P_val);
    double* d_A4 = toDevice(A_val);
    solver2_warm.setup(d_P4, d_A4);

    double* d_warm_x = toDevice(x1);
    double* d_warm_z = toDevice(z1);
    double* d_warm_s = toDevice(s1);

    solver2_warm.solve(d_q2, d_b, d_warm_x, d_warm_z, d_warm_s);

    std::vector<int32_t> status2w(1);
    cudaMemcpy(status2w.data(), solver2_warm.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(status2w[0], static_cast<int32_t>(SolverStatus::Solved));

    auto x2_warm = toHost(solver2_warm.solution.x.data(), prob.n);

    // Solutions should match
    for (int64_t i = 0; i < prob.n; i++) {
        EXPECT_NEAR(x2_warm[i], x2_cold[i], 1e-4) << "x[" << i << "]";
    }

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q1); cudaFree(d_q2); cudaFree(d_b);
    cudaFree(d_P3); cudaFree(d_A3); cudaFree(d_P4); cudaFree(d_A4);
    cudaFree(d_warm_x); cudaFree(d_warm_z); cudaFree(d_warm_s);
}

// =============================================================================
// Test: Warm start with SOC cone
// =============================================================================

TEST_F(WarmStartTest, SOCCone) {
    SOCProblem prob;
    int64_t batchSize = 1;

    std::vector<double> P_val = {};
    std::vector<double> A_val = {-1.0, -1.0, -1.0};
    std::vector<double> q = {1.0, 0.5, 0.5};
    std::vector<double> b = {0.0, 0.0, 0.0};

    Settings settings;
    settings.verbose = false;

    // Cold solve
    CompiledSolver solver_cold(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_P = nullptr;  // no P values
    double* d_A = toDevice(A_val);
    double* d_q = toDevice(q);
    double* d_b = toDevice(b);

    solver_cold.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status_cold(1);
    cudaMemcpy(status_cold.data(), solver_cold.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(status_cold[0], static_cast<int32_t>(SolverStatus::Solved));

    auto x_cold = toHost(solver_cold.solution.x.data(), prob.n);
    auto z_cold = toHost(solver_cold.solution.z.data(), prob.m);
    auto s_cold = toHost(solver_cold.solution.s.data(), prob.m);

    // Warm solve
    CompiledSolver solver_warm(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_A2 = toDevice(A_val);
    solver_warm.setup(d_P, d_A2);

    double* d_warm_x = toDevice(x_cold);
    double* d_warm_z = toDevice(z_cold);
    double* d_warm_s = toDevice(s_cold);

    solver_warm.solve(d_q, d_b, d_warm_x, d_warm_z, d_warm_s);

    std::vector<int32_t> status_warm(1);
    cudaMemcpy(status_warm.data(), solver_warm.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(status_warm[0], static_cast<int32_t>(SolverStatus::Solved));

    auto x_warm = toHost(solver_warm.solution.x.data(), prob.n);
    for (int64_t i = 0; i < prob.n; i++) {
        EXPECT_NEAR(x_warm[i], x_cold[i], 1e-6) << "x[" << i << "]";
    }

    cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    cudaFree(d_A2);
    cudaFree(d_warm_x); cudaFree(d_warm_z); cudaFree(d_warm_s);
}

// =============================================================================
// Test: Warm start with variable-length SOC cone (dim=5)
// =============================================================================

TEST_F(WarmStartTest, SOCConeVarLen) {
    SOCVarLenProblem prob;
    int64_t batchSize = 1;

    std::vector<double> P_val = {};
    std::vector<double> A_val = {-1.0, -1.0, -1.0, -1.0, -1.0};
    std::vector<double> q = {1.0, 0.5, 0.5, 0.3, 0.3};
    std::vector<double> b = {0.0, 0.0, 0.0, 0.0, 0.0};

    Settings settings;
    settings.verbose = false;

    // Cold solve
    CompiledSolver solver_cold(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_P = nullptr;  // no P values
    double* d_A = toDevice(A_val);
    double* d_q = toDevice(q);
    double* d_b = toDevice(b);

    solver_cold.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status_cold(1);
    cudaMemcpy(status_cold.data(), solver_cold.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(status_cold[0], static_cast<int32_t>(SolverStatus::Solved));

    auto x_cold = toHost(solver_cold.solution.x.data(), prob.n);
    auto z_cold = toHost(solver_cold.solution.z.data(), prob.m);
    auto s_cold = toHost(solver_cold.solution.s.data(), prob.m);

    // Warm solve
    CompiledSolver solver_warm(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_A2 = toDevice(A_val);
    solver_warm.setup(d_P, d_A2);

    double* d_warm_x = toDevice(x_cold);
    double* d_warm_z = toDevice(z_cold);
    double* d_warm_s = toDevice(s_cold);

    solver_warm.solve(d_q, d_b, d_warm_x, d_warm_z, d_warm_s);

    std::vector<int32_t> status_warm(1);
    cudaMemcpy(status_warm.data(), solver_warm.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(status_warm[0], static_cast<int32_t>(SolverStatus::Solved));

    auto x_warm = toHost(solver_warm.solution.x.data(), prob.n);
    for (int64_t i = 0; i < prob.n; i++) {
        EXPECT_NEAR(x_warm[i], x_cold[i], 1e-6) << "x[" << i << "]";
    }

    cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    cudaFree(d_A2);
    cudaFree(d_warm_x); cudaFree(d_warm_z); cudaFree(d_warm_s);
}

// =============================================================================
// Test: Bad warm start still converges
// =============================================================================

TEST_F(WarmStartTest, BadPointStillConverges) {
    SimpleQP prob;
    int64_t batchSize = 1;

    std::vector<double> P_val = {2.0, 2.0};
    std::vector<double> A_val = {1.0, 1.0, -1.0, -1.0, 1.0, 1.0};
    std::vector<double> q = {1.0, -1.0};
    std::vector<double> b_vec = {1.0, 0.0, 0.0, 1.0, 1.0};

    Settings settings;
    settings.verbose = false;

    // Cold solve for reference
    CompiledSolver solver_ref(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_P = toDevice(P_val);
    double* d_A = toDevice(A_val);
    double* d_q = toDevice(q);
    double* d_b = toDevice(b_vec);

    solver_ref.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status_ref(1);
    cudaMemcpy(status_ref.data(), solver_ref.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    ASSERT_EQ(status_ref[0], static_cast<int32_t>(SolverStatus::Solved));

    auto x_ref = toHost(solver_ref.solution.x.data(), prob.n);

    // Bad warm start point
    std::vector<double> warm_x = {0.5, 0.5};
    std::vector<double> warm_z(prob.m, 1.0);
    std::vector<double> warm_s(prob.m, 1.0);

    CompiledSolver solver_warm(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_P2 = toDevice(P_val);
    double* d_A2 = toDevice(A_val);
    solver_warm.setup(d_P2, d_A2);

    double* d_warm_x = toDevice(warm_x);
    double* d_warm_z = toDevice(warm_z);
    double* d_warm_s = toDevice(warm_s);

    solver_warm.solve(d_q, d_b, d_warm_x, d_warm_z, d_warm_s);

    std::vector<int32_t> status_warm(1);
    cudaMemcpy(status_warm.data(), solver_warm.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    EXPECT_TRUE(
        status_warm[0] == static_cast<int32_t>(SolverStatus::Solved) ||
        status_warm[0] == static_cast<int32_t>(SolverStatus::AlmostSolved)
    ) << "Bad warm start should still converge, got status " << status_warm[0];

    auto x_warm = toHost(solver_warm.solution.x.data(), prob.n);
    for (int64_t i = 0; i < prob.n; i++) {
        EXPECT_NEAR(x_warm[i], x_ref[i], 1e-5) << "x[" << i << "]";
    }

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    cudaFree(d_P2); cudaFree(d_A2);
    cudaFree(d_warm_x); cudaFree(d_warm_z); cudaFree(d_warm_s);
}

// =============================================================================
// Test: Batched warm start
// =============================================================================

TEST_F(WarmStartTest, Batched) {
    SimpleQP prob;
    int64_t batchSize = 2;

    // Batch data: each problem has its own P/A/q/b
    std::vector<double> P_val(prob.nnzP * batchSize);
    std::vector<double> A_val(prob.nnzA * batchSize);
    std::vector<double> q(prob.n * batchSize);
    std::vector<double> b(prob.m * batchSize);

    // Problem 0: q = [1, -1], Problem 1: q = [0.5, -0.5]
    std::vector<std::vector<double>> q_vals = {{1.0, -1.0}, {0.5, -0.5}};
    std::vector<double> b_single = {1.0, 0.0, 0.0, 1.0, 1.0};
    std::vector<double> P_single = {2.0, 2.0};
    std::vector<double> A_single = {1.0, 1.0, -1.0, -1.0, 1.0, 1.0};

    for (int64_t i = 0; i < batchSize; i++) {
        std::copy(P_single.begin(), P_single.end(), P_val.begin() + i * prob.nnzP);
        std::copy(A_single.begin(), A_single.end(), A_val.begin() + i * prob.nnzA);
        std::copy(q_vals[i].begin(), q_vals[i].end(), q.begin() + i * prob.n);
        std::copy(b_single.begin(), b_single.end(), b.begin() + i * prob.m);
    }

    Settings settings;
    settings.verbose = false;

    // --- Cold solve ---
    CompiledSolver solver_cold(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_P = toDevice(P_val);
    double* d_A = toDevice(A_val);
    double* d_q = toDevice(q);
    double* d_b = toDevice(b);

    solver_cold.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status_cold(batchSize);
    cudaMemcpy(status_cold.data(), solver_cold.solution.status.get(),
               sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < batchSize; i++) {
        ASSERT_EQ(status_cold[i], static_cast<int32_t>(SolverStatus::Solved))
            << "Cold batch " << i << " did not solve";
    }

    auto x_cold = toHost(solver_cold.solution.x.data(), prob.n * batchSize);
    auto z_cold = toHost(solver_cold.solution.z.data(), prob.m * batchSize);
    auto s_cold = toHost(solver_cold.solution.s.data(), prob.m * batchSize);

    // --- Warm solve ---
    CompiledSolver solver_warm(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_P2 = toDevice(P_val);
    double* d_A2 = toDevice(A_val);
    solver_warm.setup(d_P2, d_A2);

    double* d_warm_x = toDevice(x_cold);
    double* d_warm_z = toDevice(z_cold);
    double* d_warm_s = toDevice(s_cold);

    solver_warm.solve(d_q, d_b, d_warm_x, d_warm_z, d_warm_s);

    std::vector<int32_t> status_warm(batchSize);
    cudaMemcpy(status_warm.data(), solver_warm.solution.status.get(),
               sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < batchSize; i++) {
        ASSERT_EQ(status_warm[i], static_cast<int32_t>(SolverStatus::Solved))
            << "Warm batch " << i << " did not solve";
    }

    auto x_warm = toHost(solver_warm.solution.x.data(), prob.n * batchSize);
    for (int64_t i = 0; i < batchSize; i++) {
        for (int64_t j = 0; j < prob.n; j++) {
            EXPECT_NEAR(x_warm[i * prob.n + j], x_cold[i * prob.n + j], 1e-4)
                << "Batch " << i << " x[" << j << "]";
        }
    }

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    cudaFree(d_P2); cudaFree(d_A2);
    cudaFree(d_warm_x); cudaFree(d_warm_z); cudaFree(d_warm_s);
}

// =============================================================================
// Test: NaN warm start triggers NumericalError (not MaxIterations)
// =============================================================================

TEST_F(WarmStartTest, NaNWarmStartDetected) {
    SimpleQP prob;
    int64_t batchSize = 1;

    std::vector<double> P_val = {2.0, 2.0};
    std::vector<double> A_val = {1.0, 1.0, -1.0, -1.0, 1.0, 1.0};
    std::vector<double> q = {1.0, -1.0};
    std::vector<double> b = {1.0, 0.0, 0.0, 1.0, 1.0};

    Settings settings;
    settings.verbose = false;
    settings.maxIter = 200;

    CompiledSolver solver(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_P = toDevice(P_val);
    double* d_A = toDevice(A_val);
    double* d_q = toDevice(q);
    double* d_b = toDevice(b);
    solver.setup(d_P, d_A);

    // Warm start with NaN values
    double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> warm_x = {nan, nan};
    std::vector<double> warm_z = {1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<double> warm_s = {1.0, 1.0, 1.0, 1.0, 1.0};

    double* d_warm_x = toDevice(warm_x);
    double* d_warm_z = toDevice(warm_z);
    double* d_warm_s = toDevice(warm_s);

    solver.solve(d_q, d_b, d_warm_x, d_warm_z, d_warm_s);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Should detect NaN quickly and return NumericalError, NOT MaxIterations
    EXPECT_EQ(status[0], static_cast<int32_t>(SolverStatus::NumericalError))
        << "NaN warm start should produce NumericalError, got status " << status[0];

    // Should terminate quickly (not run all 200 iterations)
    EXPECT_LT(solver.info.iterations, 10)
        << "NaN should be detected within a few iterations, not " << solver.info.iterations;

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    cudaFree(d_warm_x); cudaFree(d_warm_z); cudaFree(d_warm_s);
}

// =============================================================================
// Test: NaN in one batch element doesn't affect other batches
// =============================================================================

TEST_F(WarmStartTest, NaNWarmStartBatchedIsolation) {
    SimpleQP prob;
    int64_t batchSize = 2;

    std::vector<double> P_val(prob.nnzP * batchSize);
    std::vector<double> A_val(prob.nnzA * batchSize);
    std::vector<double> q(prob.n * batchSize);
    std::vector<double> b(prob.m * batchSize);

    std::vector<double> P_single = {2.0, 2.0};
    std::vector<double> A_single = {1.0, 1.0, -1.0, -1.0, 1.0, 1.0};
    std::vector<double> q_single = {1.0, -1.0};
    std::vector<double> b_single = {1.0, 0.0, 0.0, 1.0, 1.0};

    for (int64_t i = 0; i < batchSize; i++) {
        std::copy(P_single.begin(), P_single.end(), P_val.begin() + i * prob.nnzP);
        std::copy(A_single.begin(), A_single.end(), A_val.begin() + i * prob.nnzA);
        std::copy(q_single.begin(), q_single.end(), q.begin() + i * prob.n);
        std::copy(b_single.begin(), b_single.end(), b.begin() + i * prob.m);
    }

    Settings settings;
    settings.verbose = false;

    CompiledSolver solver(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);

    double* d_P = toDevice(P_val);
    double* d_A = toDevice(A_val);
    double* d_q = toDevice(q);
    double* d_b = toDevice(b);
    solver.setup(d_P, d_A);

    // Batch 0: NaN warm start, Batch 1: valid warm start
    double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> warm_x = {nan, nan,   0.5, 0.5};
    std::vector<double> warm_z(prob.m * batchSize, 1.0);
    std::vector<double> warm_s(prob.m * batchSize, 1.0);

    double* d_warm_x = toDevice(warm_x);
    double* d_warm_z = toDevice(warm_z);
    double* d_warm_s = toDevice(warm_s);

    solver.solve(d_q, d_b, d_warm_x, d_warm_z, d_warm_s);

    std::vector<int32_t> status(batchSize);
    cudaMemcpy(status.data(), solver.solution.status.get(),
               sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    // Batch 0 should get NumericalError
    EXPECT_EQ(status[0], static_cast<int32_t>(SolverStatus::NumericalError))
        << "Batch 0 (NaN warm start) should get NumericalError, got " << status[0];

    // Batch 1 should solve normally
    EXPECT_TRUE(
        status[1] == static_cast<int32_t>(SolverStatus::Solved) ||
        status[1] == static_cast<int32_t>(SolverStatus::AlmostSolved)
    ) << "Batch 1 (valid warm start) should solve, got status " << status[1];

    // Batch 1 solution should not contain NaN
    auto x_sol = toHost(solver.solution.x.data(), prob.n * batchSize);
    EXPECT_FALSE(std::isnan(x_sol[prob.n + 0])) << "Batch 1 x[0] is NaN";
    EXPECT_FALSE(std::isnan(x_sol[prob.n + 1])) << "Batch 1 x[1] is NaN";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    cudaFree(d_warm_x); cudaFree(d_warm_z); cudaFree(d_warm_s);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
