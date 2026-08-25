// test_cvxpy_ported.cpp
// Tests ported from failing CVXPY Moreau tests
// Ground truth values verified with CLARABEL solver

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <limits>

using namespace moreau;

// Helper class (same as test_clarabel_ported.cpp)
class ClarabelTestHelper {
public:
    int64_t n, m, batchSize;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_values, A_values, q_data, b_data;
    Cones cones;
    Settings settings;

    std::vector<double> x_sol, s_sol, z_sol;
    std::vector<int32_t> status_sol;
    std::vector<double> obj_primal, obj_dual;

    ClarabelTestHelper(int64_t n_, int64_t m_, int64_t batch_ = 1)
        : n(n_), m(m_), batchSize(batch_) {
        settings.verbose = false;
        settings.maxIter = 200;
        settings.ipm.equilibrationSettings.enable = true;
    }

    void solve() {
        int64_t nnzP = P_values.size() / batchSize;
        int64_t nnzA = A_values.size() / batchSize;

        CompiledSolver solver(
            n, m, batchSize,
            P_ro.data(), P_ci.data(), nnzP,
            A_ro.data(), A_ci.data(), nnzA,
            cones,
            settings
        );

        double *d_P_values, *d_A_values, *d_q, *d_b;
        cudaMalloc(&d_P_values, sizeof(double) * std::max<size_t>(1, nnzP * batchSize));
        cudaMalloc(&d_A_values, sizeof(double) * std::max<size_t>(1, nnzA * batchSize));
        cudaMalloc(&d_q, sizeof(double) * n * batchSize);
        cudaMalloc(&d_b, sizeof(double) * m * batchSize);

        if (nnzP > 0) {
            cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
        }
        if (nnzA > 0) {
            cudaMemcpy(d_A_values, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

        // Solve (convenience method - infinite bounds)
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

        cudaDeviceSynchronize();

        cudaFree(d_P_values);
        cudaFree(d_A_values);
        cudaFree(d_q);
        cudaFree(d_b);
    }

    bool isSolved(int batch = 0) const {
        return status_sol[batch] == static_cast<int32_t>(SolverStatus::Solved) ||
               status_sol[batch] == static_cast<int32_t>(SolverStatus::AlmostSolved);
    }

    bool isPrimalInfeasible(int batch = 0) const {
        return status_sol[batch] == static_cast<int32_t>(SolverStatus::PrimalInfeasible);
    }

    bool isDualInfeasible(int batch = 0) const {
        return status_sol[batch] == static_cast<int32_t>(SolverStatus::DualInfeasible);
    }

    bool isMaxIterations(int batch = 0) const {
        return status_sol[batch] == static_cast<int32_t>(SolverStatus::MaxIterations);
    }
};

class CVXPYPortedTest : public ::testing::Test {
protected:
    void SetUp() override { cudaGetLastError(); }
    void TearDown() override { cudaDeviceSynchronize(); }
};


TEST_F(CVXPYPortedTest, Expcone1) {
    // Test: expcone_1
    // n=3, m=8
    // Cones: zero=0, nonneg=5, no soc, exp=1, no power
    // Expected objective: 0.23534820367582565

    ClarabelTestHelper h(3, 8);

    h.A_ro = {0, 3, 6, 7, 8, 9, 10, 11, 12};
    h.A_ci = {0, 1, 2, 0, 1, 2, 0, 1, 2, 2, 1, 0};
    h.A_values = {1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

    h.P_ro = {0, 0, 0, 0};  // Empty P matrix - all row offsets should be 0
    h.P_ci = {};
    h.P_values = {};

    h.q_data = {3, 2, 1};
    h.b_data = {1, -0.1, 0, 0, 0, 0, 0, 0};

    h.cones.numZeroCones = 0;
    h.cones.numNonnegCones = 5;
    h.cones.numSocCones = 0;
    h.cones.numExpCones = 1;
    h.cones.numPowerCones = 0;

    h.settings.maxIter = 500;
    h.solve();

    EXPECT_TRUE(h.isSolved()) << "Expected Solved, got status " << h.status_sol[0];

    double expected_obj = 0.23534820367582565;
    EXPECT_NEAR(h.obj_primal[0], expected_obj, 1e-3) << "Objective mismatch";
}

TEST_F(CVXPYPortedTest, Socp1) {
    // Test: socp_1
    // n=5, m=8
    // Cones: zero=0, nonneg=2, soc dims=[3, 3], no exp, no power
    // Expected objective: -13.548638886940894

    ClarabelTestHelper h(5, 8);

    h.A_ro = {0, 3, 4, 5, 6, 7, 8, 9, 10};
    h.A_ci = {0, 1, 2, 4, 3, 0, 1, 4, 3, 2};
    h.A_values = {-1, -1, -3, 1, -1, -1, -1, -1, -1, -1};

    h.P_ro = {0, 0, 0, 0, 0, 0};  // Empty P matrix - all row offsets should be 0
    h.P_ci = {};
    h.P_values = {};

    h.q_data = {3, 2, 1, 0, 0};
    h.b_data = {-1, 5, 0, 0, 0, 0, 0, 0};

    h.cones.numZeroCones = 0;
    h.cones.numNonnegCones = 2;
    h.cones.socConeDims = {3, 3};
    h.cones.numSocCones = 2;
    h.cones.numExpCones = 0;
    h.cones.numPowerCones = 0;

    h.settings.maxIter = 500;
    h.solve();

    EXPECT_TRUE(h.isSolved()) << "Expected Solved, got status " << h.status_sol[0];

    double expected_obj = -13.548638886940894;
    EXPECT_NEAR(h.obj_primal[0], expected_obj, 1e-3) << "Objective mismatch";
}

TEST_F(CVXPYPortedTest, Pcp2) {
    // Test: pcp_2
    // n=5, m=7
    // Cones: zero=1, nonneg=0, no soc, no exp, power alphas=[0.2, 0.4]
    // Expected objective: -1.8073406312209352

    ClarabelTestHelper h(5, 7);

    h.A_ro = {0, 3, 4, 5, 6, 7, 7, 8};
    h.A_ci = {2, 3, 4, 2, 3, 0, 4, 1};
    h.A_values = {1, 1, 0.5, -1, -1, -1, -1, -1};

    h.P_ro = {0, 0, 0, 0, 0, 0};  // Empty P matrix - all row offsets should be 0
    h.P_ci = {};
    h.P_values = {};

    h.q_data = {-1, -1, 1, 0, 0};
    h.b_data = {2, 0, 0, 0, 0, 1, 0};

    h.cones.numZeroCones = 1;
    h.cones.numNonnegCones = 0;
    h.cones.numSocCones = 0;
    h.cones.numExpCones = 0;
    h.cones.numPowerCones = 2;
    h.cones.powerAlphas = {0.2, 0.4};

    h.settings.maxIter = 500;
    h.solve();

    EXPECT_TRUE(h.isSolved()) << "Expected Solved, got status " << h.status_sol[0];

    double expected_obj = -1.8073406312209352;
    EXPECT_NEAR(h.obj_primal[0], expected_obj, 1e-3) << "Objective mismatch";
}

TEST_F(CVXPYPortedTest, ExpSoc1) {
    // Test: exp_soc_1
    // n=9, m=16
    // Cones: zero=0, nonneg=1, soc dims=[3, 3], exp=3, no power
    // Expected objective: 4.075119715968798

    ClarabelTestHelper h(9, 16);

    h.A_ro = {0, 2, 3, 6, 8, 9, 10, 11, 12, 12, 13, 14, 14, 15, 16, 16, 17};
    h.A_ci = {0, 8, 4, 5, 6, 7, 6, 7, 8, 4, 7, 1, 5, 2, 6, 3, 7};
    h.A_values = {-1, 1, -1, -1.352774926, -1.323206075, -2.380292492, -0.6550768528, -0.04638007515, -1, -1, -1.738406322, -1, -1, -1, -1, -1, -1};

    h.P_ro = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    h.P_ci = {};
    h.P_values = {};

    h.q_data = {1, -0.75, -0.75, -0.75, 0, 0, 0, 0, 0};
    h.b_data = {0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0};

    h.cones.numZeroCones = 0;
    h.cones.numNonnegCones = 1;
    h.cones.socConeDims = {3, 3};
    h.cones.numSocCones = 2;
    h.cones.numExpCones = 3;
    h.cones.numPowerCones = 0;

    h.settings.maxIter = 500;
    h.solve();

    EXPECT_TRUE(h.isSolved()) << "Expected Solved, got status " << h.status_sol[0];

    double expected_obj = 4.075119715968798;
    EXPECT_NEAR(h.obj_primal[0], expected_obj, 1e-3) << "Objective mismatch";
}

TEST_F(CVXPYPortedTest, Pcp1) {
    // Test: pcp_1
    // n=7, m=12
    // Cones: zero=0, nonneg=3, no soc, no exp, power alphas=[0.5, 0.5, 0.5]
    // Expected objective: -13.54863882775228

    ClarabelTestHelper h(7, 12);

    h.A_ro = {0, 4, 7, 8, 8, 9, 10, 10, 11, 12, 12, 13, 14};
    h.A_ci = {3, 4, 5, 6, 0, 1, 2, 6, 3, 0, 4, 1, 5, 2};
    h.A_values = {1, 1, 1, -1, -1, -1, -3, 1, -1, -1, -1, -1, -1, -1};

    h.P_ro = {0, 0, 0, 0, 0, 0, 0, 0};  // Empty P matrix - all row offsets should be 0
    h.P_ci = {};
    h.P_values = {};

    h.q_data = {3, 2, 1, 0, 0, 0, 0};
    h.b_data = {0, -1, 25, 1, 0, 0, 1, 0, 0, 1, 0, 0};

    h.cones.numZeroCones = 0;
    h.cones.numNonnegCones = 3;
    h.cones.numSocCones = 0;
    h.cones.numExpCones = 0;
    h.cones.numPowerCones = 3;
    h.cones.powerAlphas = {0.5, 0.5, 0.5};

    h.settings.maxIter = 500;
    h.solve();

    EXPECT_TRUE(h.isSolved()) << "Expected Solved, got status " << h.status_sol[0];

    double expected_obj = -13.54863882775228;
    EXPECT_NEAR(h.obj_primal[0], expected_obj, 1e-3) << "Objective mismatch";
}

TEST_F(CVXPYPortedTest, Socp2) {
    // Test: socp_2
    // n=3, m=8
    // Cones: zero=0, nonneg=5, soc dims=[3], no exp, no power
    // Expected objective: -8.99999999674191

    ClarabelTestHelper h(3, 8);

    h.A_ro = {0, 2, 3, 4, 5, 6, 6, 8, 9};
    h.A_ci = {0, 1, 2, 2, 0, 1, 0, 1, 2};
    h.A_values = {2, 1, -1, 1, -1, -1, -1, -2, -1};

    h.P_ro = {0, 0, 0, 0};  // Empty P matrix - all row offsets should be 0
    h.P_ci = {};
    h.P_values = {};

    h.q_data = {-4, -5, 0};
    h.b_data = {3, 0, 0, 0, 0, 3, 0, 0};

    h.cones.numZeroCones = 0;
    h.cones.numNonnegCones = 5;
    h.cones.socConeDims = {3};
    h.cones.numSocCones = 1;
    h.cones.numExpCones = 0;
    h.cones.numPowerCones = 0;

    h.settings.maxIter = 500;
    h.solve();

    EXPECT_TRUE(h.isSolved()) << "Expected Solved, got status " << h.status_sol[0];

    double expected_obj = -8.99999999674191;
    EXPECT_NEAR(h.obj_primal[0], expected_obj, 1e-3) << "Objective mismatch";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
