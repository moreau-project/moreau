// test_objective_values.cpp
// Tests verifying objective value computation and optimality
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <cmath>

using namespace moreau;

class ObjectiveValuesTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Compute objective value: 0.5 * x'Px + q'x
    double computeObjective(const std::vector<double>& P_diag,
                            const std::vector<double>& q,
                            const std::vector<double>& x) {
        double obj = 0.0;
        for (size_t i = 0; i < x.size(); i++) {
            obj += 0.5 * P_diag[i] * x[i] * x[i] + q[i] * x[i];
        }
        return obj;
    }
};

// Test that computed objective matches manual calculation
TEST_F(ObjectiveValuesTest, ObjectiveMatchesManualCalculation) {
    int n = 2, m = 2;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numNonnegCones = 2;

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values = {2.0, 3.0};
    std::vector<double> A_values = {1.0, 1.0};
    std::vector<double> q = {1.0, -1.0};
    std::vector<double> b = {5.0, 5.0};

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    double manualObj = computeObjective(P_values, q, x);

    // Get solver's computed objective
    std::vector<double> objPrimal(1);
    solver.info.cost_primal.gpuToCpu(objPrimal.data());
    cudaDeviceSynchronize();

    EXPECT_NEAR(objPrimal[0], manualObj, 1e-4)
        << "Solver objective should match manual calculation";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test that optimal objective is correct for known problem
TEST_F(ObjectiveValuesTest, KnownOptimalObjective) {
    int n = 2, m = 1;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;

    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;

    Cones cones{};
    cones.numZeroCones = 1;  // Equality constraint

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    // min 0.5*(x1^2 + x2^2)  s.t. x1 + x2 = 2
    // Optimal: x1 = x2 = 1, objective = 0.5*(1 + 1) = 1
    std::vector<double> P_values = {1.0, 1.0};
    std::vector<double> A_values = {1.0, 1.0};
    std::vector<double> q = {0.0, 0.0};
    std::vector<double> b = {2.0};

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> objPrimal(1);
    solver.info.cost_primal.gpuToCpu(objPrimal.data());
    cudaDeviceSynchronize();

    EXPECT_NEAR(objPrimal[0], 1.0, 1e-4) << "Optimal objective should be 1.0";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Test that primal and dual objectives are close at optimality
TEST_F(ObjectiveValuesTest, PrimalDualGap) {
    int n = 3, m = 2;
    int batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;

    std::vector<int64_t> A_ro = {0, 2, 3};
    std::vector<int64_t> A_ci = {0, 1, 2};
    int64_t nnzA = 3;

    Cones cones{};
    cones.numNonnegCones = 2;

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;
    settings.ipm.tolGapAbs = 1e-8;
    settings.ipm.tolGapRel = 1e-8;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), nnzP,
                          A_ro.data(), A_ci.data(), nnzA, cones, settings);

    std::vector<double> P_values = {1.0, 2.0, 3.0};
    std::vector<double> A_values = {1.0, 1.0, 1.0};
    std::vector<double> q = {1.0, 2.0, 3.0};
    std::vector<double> b = {10.0, 10.0};

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> objPrimal(1), objDual(1), gapAbs(1);
    solver.info.cost_primal.gpuToCpu(objPrimal.data());
    solver.info.cost_dual.gpuToCpu(objDual.data());
    solver.info.gap_abs.gpuToCpu(gapAbs.data());
    cudaDeviceSynchronize();

    // Duality gap should be small at optimality
    EXPECT_LT(gapAbs[0], 1e-5) << "Duality gap should be small";
    EXPECT_NEAR(objPrimal[0], objDual[0], 1e-4)
        << "Primal and dual objectives should be close";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
