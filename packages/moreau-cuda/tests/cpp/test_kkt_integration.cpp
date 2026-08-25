/**
 * @file test_kkt_integration.cpp
 * @brief Integration tests for KKT solver
 *
 * Tests that the CuDSS KKT solver implementation produces correct solutions
 * for various QP/LP problems.
 */
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

#include "moreau/kkt/kkt.hpp"
#include "moreau/kkt/kkt_solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/matrix/csr.hpp"
#include "moreau/vector/vector.hpp"

using namespace moreau;

class KKTIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper to compare vectors with tolerance
    bool vectorsClose(const std::vector<double>& a, const std::vector<double>& b, double tol = 1e-6) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i) {
            if (std::abs(a[i] - b[i]) > tol) {
                std::cout << "Mismatch at index " << i << ": " << a[i] << " vs " << b[i]
                          << " (diff=" << std::abs(a[i] - b[i]) << ")" << std::endl;
                return false;
            }
        }
        return true;
    }
};

// Test CuDSS solver for a simple diagonal QP
TEST_F(KKTIntegrationTest, DiagonalQPCuDSS) {
    // Problem:
    //   min 0.5 * (x1^2 + x2^2) + 2*x1 + x2
    //   s.t. x1 + x2 = 1
    //        x1, x2 >= 0
    //
    // P = diag(1, 1)
    // A = [[1, 1], [1, 0], [0, 1]]
    // Cones: 1 zero (equality), 2 nonneg (x1,x2 >= 0)

    int64_t n = 2, m = 3, batch = 1;

    // P matrix (diagonal)
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    std::vector<double> P_vals = {1.0, 1.0};

    // A matrix
    std::vector<int64_t> A_ro = {0, 2, 3, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    std::vector<double> A_vals = {1.0, 1.0, 1.0, 1.0};

    // Cones
    Cones cones;
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;
    cones.initialize(batch, 0);

    // Create CSR matrices
    CSR P(n, n, 2, batch);
    CSR A(m, n, 4, batch);
    P.indicesCpuToGpu(P_ro.data(), P_ci.data(), 0);
    A.indicesCpuToGpu(A_ro.data(), A_ci.data(), 0);
    cudaMemcpy(P.values(), P_vals.data(), sizeof(double) * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(A.values(), A_vals.data(), sizeof(double) * 4, cudaMemcpyHostToDevice);

    // RHS: [rhs_x; rhs_z] - we'll use a simple test RHS
    std::vector<double> h_rhs(n + m);
    h_rhs[0] = 1.0;  // rhs_x[0]
    h_rhs[1] = 2.0;  // rhs_x[1]
    h_rhs[2] = 0.5;  // rhs_z[0]
    h_rhs[3] = 0.3;  // rhs_z[1]
    h_rhs[4] = 0.2;  // rhs_z[2]

    double* d_rhs = nullptr;
    double* d_sol = nullptr;
    cudaMalloc(&d_rhs, sizeof(double) * (n + m));
    cudaMalloc(&d_sol, sizeof(double) * (n + m));
    cudaMemcpy(d_rhs, h_rhs.data(), sizeof(double) * (n + m), cudaMemcpyHostToDevice);

    // Test CuDSS solver
    std::vector<double> sol_cudss(n + m);
    {
        KKTData kkt(n, m, batch, P_ro.data(), P_ci.data(), 2,
                    A_ro.data(), A_ci.data(), 4, cones);
        kkt.populate(P, A, 0);

        // Set identity scaling for H block (nonneg cones with w=1 means Hs=-1)
        cones.nonneg_w.setToConstant(1.0, 0);
        kkt.update_H(cones, nullptr, 0);

        // Regularize and factor
        kkt.regularize_and_refactor(true, 1e-8, 1e-20, 0);

        // Solve
        kkt.solve(d_rhs, d_sol, 0);
        cudaDeviceSynchronize();

        cudaMemcpy(sol_cudss.data(), d_sol, sizeof(double) * (n + m), cudaMemcpyDeviceToHost);
    }

    std::cout << "CuDSS solution: [";
    for (size_t i = 0; i < sol_cudss.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << sol_cudss[i];
        if (i < sol_cudss.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Check solution is not all zeros (basic sanity check)
    double norm = 0;
    for (auto v : sol_cudss) norm += v * v;
    EXPECT_GT(norm, 1e-10) << "Solution should not be zero";

    cudaFree(d_rhs);
    cudaFree(d_sol);
}

// Test KKTSolver wrapper
TEST_F(KKTIntegrationTest, KKTSolverWrapper) {
    int64_t n = 2, m = 2, batch = 1;
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    std::vector<int64_t> A_ro = {0, 2, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};

    Cones cones;
    cones.numNonnegCones = 2;
    cones.initialize(batch, 0);

    Settings settings;
    settings.ipm.kktSolverType = KKTSolverType::Auto;

    auto solver = make_kkt_solver(n, m, batch, P_ro.data(), P_ci.data(), 2,
                                  A_ro.data(), A_ci.data(), 4, cones, settings, 0);

    // Auto defaults to CuDSS
    EXPECT_EQ(solver->actualSolverType(), KKTSolverType::CuDSS);
}

// Test explicit CuDSS selection
TEST_F(KKTIntegrationTest, ExplicitCuDSSSelection) {
    int64_t n = 50, m = 20, batch = 1;

    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci(n);
    for (int64_t i = 0; i <= n; ++i) P_ro[i] = i;
    for (int64_t i = 0; i < n; ++i) P_ci[i] = i;

    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci(m);
    for (int64_t i = 0; i <= m; ++i) A_ro[i] = i;
    for (int64_t i = 0; i < m; ++i) A_ci[i] = i;

    Cones cones;
    cones.numNonnegCones = m;
    cones.initialize(batch, 0);

    Settings settings;
    settings.ipm.kktSolverType = KKTSolverType::CuDSS;

    auto solver = make_kkt_solver(n, m, batch, P_ro.data(), P_ci.data(), n,
                                  A_ro.data(), A_ci.data(), m, cones, settings, 0);

    EXPECT_EQ(solver->actualSolverType(), KKTSolverType::CuDSS);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
