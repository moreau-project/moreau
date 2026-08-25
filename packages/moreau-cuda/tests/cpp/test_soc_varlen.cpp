/**
 * @file test_soc_varlen.cpp
 * @brief Variable-dimension Second-Order Cone (SOC) problem tests with known solutions
 *
 * Tests SOC cones with dimensions other than 3 (dim=2, 5, 10, mixed dims)
 * and combinations with nonnegative cones.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/solver.hpp"

using namespace moreau;

namespace {

/**
 * @brief Sample a point strictly inside the SOC of given dimension.
 *
 * SOC(dim) = { (t, u1, ..., u_{dim-1}) : t >= ||u|| }
 * Returns a point with t = ||u|| + margin, so it's strictly interior.
 */
std::vector<double> sampleSocInterior(std::mt19937_64& rng, int64_t dim, double margin = 0.1) {
    std::normal_distribution<double> normal(0.0, 1.0);
    std::vector<double> point(dim);
    double norm_sq = 0.0;
    for (int64_t i = 1; i < dim; i++) {
        point[i] = normal(rng);
        norm_sq += point[i] * point[i];
    }
    point[0] = std::sqrt(norm_sq) + std::abs(margin);
    return point;
}

/**
 * @brief Check if a point lies inside the SOC of given dimension.
 *
 * SOC(dim) = { (t, u) : t >= ||u|| }
 */
bool isInSoc(const double* data, int64_t dim, double tol = 1e-8) {
    double norm_sq = 0.0;
    for (int64_t i = 1; i < dim; i++) {
        norm_sq += data[i] * data[i];
    }
    return data[0] >= std::sqrt(norm_sq) - tol;
}

/**
 * @brief Check if a vector lies in a product of variable-dimension SOC cones.
 */
bool isInSocProductVarLen(const std::vector<double>& s,
                          const std::vector<int64_t>& socDims,
                          int64_t offset,
                          double tol = 1e-8) {
    int64_t pos = offset;
    for (int64_t k = 0; k < static_cast<int64_t>(socDims.size()); k++) {
        int64_t dim = socDims[k];
        if (!isInSoc(s.data() + pos, dim, tol)) return false;
        pos += dim;
    }
    return true;
}

} // namespace

class SOCVarLenTest : public ::testing::Test {
protected:
    void SetUp() override { cudaGetLastError(); }
    void TearDown() override { cudaDeviceSynchronize(); }
};

/**
 * @brief Test 1: Single SOC with dim=2 (the smallest valid SOC).
 *
 * SOC(2) = { (t, u) : t >= |u| }
 *
 * Problem: min 0.5||x||^2 + q'x  s.t. Ax + s = b, s in SOC(2)
 *   n=4, m=2, 1 SOC cone of dim 2.
 */
TEST_F(SOCVarLenTest, Dim2SOCConverges) {
    const int64_t n = 4;
    const int64_t m = 2;
    const int64_t batchSize = 1;
    const uint64_t seed = 100;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    // x_star: random primal optimum
    std::vector<double> x_star(n);
    for (int64_t j = 0; j < n; j++) x_star[j] = normal(rng);

    // q = -x_star so unconstrained minimizer is x_star
    std::vector<double> q(n);
    for (int64_t j = 0; j < n; j++) q[j] = -x_star[j];

    // A: random dense m x n
    std::vector<std::vector<double>> A_dense(m, std::vector<double>(n));
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            A_dense[i][j] = normal(rng);

    // s_star: strictly inside SOC(2)
    std::vector<double> s_star = sampleSocInterior(rng, 2, 0.1);

    // b = A * x_star + s_star
    std::vector<double> b(m, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) b[i] += A_dense[i][j] * x_star[j];
        b[i] += s_star[i];
    }

    // P = I (diagonal CSR)
    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;
    for (int64_t i = 0; i < n; i++) {
        P_ro[i] = i;
        P_ci.push_back(i);
        P_val.push_back(1.0);
    }
    P_ro[n] = n;

    // A in CSR (dense, all entries)
    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;
    A_ro.push_back(0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            A_ci.push_back(j);
            A_val.push_back(A_dense[i][j]);
        }
        A_ro.push_back(static_cast<int64_t>(A_val.size()));
    }

    int64_t nnzP = static_cast<int64_t>(P_val.size());
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    Cones cones{};
    cones.socConeDims = {2};
    cones.numSocCones = 1;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    std::vector<double> s_sol(m);
    std::vector<int32_t> status(1);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Check status
    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Status = " << status[0];

    // Check x close to x_star
    double max_x_err = 0.0;
    for (int64_t i = 0; i < n; i++)
        max_x_err = std::max(max_x_err, std::abs(x_sol[i] - x_star[i]));
    EXPECT_LT(max_x_err, 1e-4) << "max |x_sol - x_star| = " << max_x_err;

    // Check s in SOC(2)
    EXPECT_TRUE(isInSoc(s_sol.data(), 2, 1e-6)) << "s not in SOC(2)";

    // Check primal residual ||Ax + s - b||
    double max_res = 0.0;
    for (int64_t i = 0; i < m; i++) {
        double r = -b[i] + s_sol[i];
        for (int64_t j = 0; j < n; j++) r += A_dense[i][j] * x_sol[j];
        max_res = std::max(max_res, std::abs(r));
    }
    EXPECT_LT(max_res, 1e-5) << "primal residual = " << max_res;

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * @brief Test 2: Single SOC with dim=5.
 *
 * SOC(5) = { (t, u1, u2, u3, u4) : t >= sqrt(u1^2+u2^2+u3^2+u4^2) }
 *
 * n=6, m=5, 1 SOC cone of dim 5.
 */
TEST_F(SOCVarLenTest, Dim5SOCConverges) {
    const int64_t n = 6;
    const int64_t m = 5;
    const int64_t batchSize = 1;
    const uint64_t seed = 200;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> x_star(n);
    for (int64_t j = 0; j < n; j++) x_star[j] = normal(rng);

    std::vector<double> q(n);
    for (int64_t j = 0; j < n; j++) q[j] = -x_star[j];

    std::vector<std::vector<double>> A_dense(m, std::vector<double>(n));
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            A_dense[i][j] = normal(rng);

    std::vector<double> s_star = sampleSocInterior(rng, 5, 0.15);

    std::vector<double> b(m, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) b[i] += A_dense[i][j] * x_star[j];
        b[i] += s_star[i];
    }

    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;
    for (int64_t i = 0; i < n; i++) {
        P_ro[i] = i;
        P_ci.push_back(i);
        P_val.push_back(1.0);
    }
    P_ro[n] = n;

    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;
    A_ro.push_back(0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            A_ci.push_back(j);
            A_val.push_back(A_dense[i][j]);
        }
        A_ro.push_back(static_cast<int64_t>(A_val.size()));
    }

    int64_t nnzP = static_cast<int64_t>(P_val.size());
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    Cones cones{};
    cones.socConeDims = {5};
    cones.numSocCones = 1;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    std::vector<double> s_sol(m);
    std::vector<int32_t> status(1);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Status = " << status[0];

    double max_x_err = 0.0;
    for (int64_t i = 0; i < n; i++)
        max_x_err = std::max(max_x_err, std::abs(x_sol[i] - x_star[i]));
    EXPECT_LT(max_x_err, 1e-4) << "max |x_sol - x_star| = " << max_x_err;

    EXPECT_TRUE(isInSoc(s_sol.data(), 5, 1e-6)) << "s not in SOC(5)";

    double max_res = 0.0;
    for (int64_t i = 0; i < m; i++) {
        double r = -b[i] + s_sol[i];
        for (int64_t j = 0; j < n; j++) r += A_dense[i][j] * x_sol[j];
        max_res = std::max(max_res, std::abs(r));
    }
    EXPECT_LT(max_res, 1e-5) << "primal residual = " << max_res;

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * @brief Test 3: Single SOC with dim=10 (a "sparse" SOC since dim > 4).
 *
 * SOC(10) = { (t, u1, ..., u9) : t >= ||u|| }
 *
 * n=8, m=10, 1 SOC cone of dim 10.
 */
TEST_F(SOCVarLenTest, Dim10SOCConverges) {
    const int64_t n = 8;
    const int64_t m = 10;
    const int64_t batchSize = 1;
    const uint64_t seed = 300;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> x_star(n);
    for (int64_t j = 0; j < n; j++) x_star[j] = normal(rng);

    std::vector<double> q(n);
    for (int64_t j = 0; j < n; j++) q[j] = -x_star[j];

    std::vector<std::vector<double>> A_dense(m, std::vector<double>(n));
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            A_dense[i][j] = normal(rng);

    std::vector<double> s_star = sampleSocInterior(rng, 10, 0.2);

    std::vector<double> b(m, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) b[i] += A_dense[i][j] * x_star[j];
        b[i] += s_star[i];
    }

    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;
    for (int64_t i = 0; i < n; i++) {
        P_ro[i] = i;
        P_ci.push_back(i);
        P_val.push_back(1.0);
    }
    P_ro[n] = n;

    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;
    A_ro.push_back(0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            A_ci.push_back(j);
            A_val.push_back(A_dense[i][j]);
        }
        A_ro.push_back(static_cast<int64_t>(A_val.size()));
    }

    int64_t nnzP = static_cast<int64_t>(P_val.size());
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    Cones cones{};
    cones.socConeDims = {10};
    cones.numSocCones = 1;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    std::vector<double> s_sol(m);
    std::vector<int32_t> status(1);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Status = " << status[0];

    double max_x_err = 0.0;
    for (int64_t i = 0; i < n; i++)
        max_x_err = std::max(max_x_err, std::abs(x_sol[i] - x_star[i]));
    EXPECT_LT(max_x_err, 1e-4) << "max |x_sol - x_star| = " << max_x_err;

    EXPECT_TRUE(isInSoc(s_sol.data(), 10, 1e-6)) << "s not in SOC(10)";

    double max_res = 0.0;
    for (int64_t i = 0; i < m; i++) {
        double r = -b[i] + s_sol[i];
        for (int64_t j = 0; j < n; j++) r += A_dense[i][j] * x_sol[j];
        max_res = std::max(max_res, std::abs(r));
    }
    EXPECT_LT(max_res, 1e-5) << "primal residual = " << max_res;

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * @brief Test 4: Multiple SOCs with mixed dimensions.
 *
 * 4 SOC cones with dims {2, 3, 5, 4}.  Total SOC dim = 14.
 * n=10, m=14.
 *
 * This exercises both "dense" SOC cones (dim <= 4: dims 2, 3, 4)
 * and "sparse" SOC cones (dim > 4: dim 5).
 */
TEST_F(SOCVarLenTest, MixedDimSOCConverges) {
    const int64_t n = 10;
    const std::vector<int64_t> socDims = {2, 3, 5, 4};
    const int64_t numSocCones = 4;
    int64_t m = 0;
    for (auto d : socDims) m += d;  // m = 14
    const int64_t batchSize = 1;
    const uint64_t seed = 400;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> x_star(n);
    for (int64_t j = 0; j < n; j++) x_star[j] = normal(rng);

    std::vector<double> q(n);
    for (int64_t j = 0; j < n; j++) q[j] = -x_star[j];

    std::vector<std::vector<double>> A_dense(m, std::vector<double>(n));
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            A_dense[i][j] = normal(rng);

    // Build s_star by sampling interior of each SOC cone (large margin keeps constraints slack)
    std::vector<double> s_star(m);
    int64_t offset = 0;
    for (int64_t k = 0; k < numSocCones; k++) {
        auto block = sampleSocInterior(rng, socDims[k], 1.0);
        for (int64_t i = 0; i < socDims[k]; i++) {
            s_star[offset + i] = block[i];
        }
        offset += socDims[k];
    }

    std::vector<double> b(m, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) b[i] += A_dense[i][j] * x_star[j];
        b[i] += s_star[i];
    }

    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;
    for (int64_t i = 0; i < n; i++) {
        P_ro[i] = i;
        P_ci.push_back(i);
        P_val.push_back(1.0);
    }
    P_ro[n] = n;

    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;
    A_ro.push_back(0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            A_ci.push_back(j);
            A_val.push_back(A_dense[i][j]);
        }
        A_ro.push_back(static_cast<int64_t>(A_val.size()));
    }

    int64_t nnzP = static_cast<int64_t>(P_val.size());
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    Cones cones{};
    cones.socConeDims = socDims;
    cones.numSocCones = numSocCones;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    std::vector<double> s_sol(m);
    std::vector<int32_t> status(1);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Status = " << status[0];

    double max_x_err = 0.0;
    for (int64_t i = 0; i < n; i++)
        max_x_err = std::max(max_x_err, std::abs(x_sol[i] - x_star[i]));
    EXPECT_LT(max_x_err, 1e-4) << "max |x_sol - x_star| = " << max_x_err;

    // Check each SOC cone individually
    EXPECT_TRUE(isInSocProductVarLen(s_sol, socDims, 0, 1e-6))
        << "s not in product of variable-dim SOC cones";

    double max_res = 0.0;
    for (int64_t i = 0; i < m; i++) {
        double r = -b[i] + s_sol[i];
        for (int64_t j = 0; j < n; j++) r += A_dense[i][j] * x_sol[j];
        max_res = std::max(max_res, std::abs(r));
    }
    EXPECT_LT(max_res, 1e-5) << "primal residual = " << max_res;

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * @brief Test 5: Mixed SOC dims plus nonnegative cones.
 *
 * Cone product: 3 nonneg cones, then SOC(2) and SOC(5).
 * n=8, m=10 (3 nonneg + 2 + 5).
 *
 * Cone ordering in Moreau: zero, nonneg, SOC, exp, power.
 * So the first 3 constraint rows are nonneg, then 2 rows for SOC(2), then 5 for SOC(5).
 */
TEST_F(SOCVarLenTest, MixedDimWithNonneg) {
    const int64_t n = 8;
    const int64_t numNonneg = 3;
    const std::vector<int64_t> socDims = {2, 5};
    const int64_t numSocCones = 2;
    int64_t totalSocDim = 0;
    for (auto d : socDims) totalSocDim += d;  // 7
    const int64_t m = numNonneg + totalSocDim;  // 10
    const int64_t batchSize = 1;
    const uint64_t seed = 500;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> x_star(n);
    for (int64_t j = 0; j < n; j++) x_star[j] = normal(rng);

    std::vector<double> q(n);
    for (int64_t j = 0; j < n; j++) q[j] = -x_star[j];

    std::vector<std::vector<double>> A_dense(m, std::vector<double>(n));
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            A_dense[i][j] = normal(rng);

    // Build s_star: nonneg part (positive values) + SOC parts (interior)
    std::vector<double> s_star(m);

    // Nonneg cones: s_i > 0
    std::uniform_real_distribution<double> unif(0.5, 2.0);
    for (int64_t i = 0; i < numNonneg; i++) {
        s_star[i] = unif(rng);
    }

    // SOC cones: interior points (large margin keeps constraints slack)
    int64_t soc_offset = numNonneg;
    for (int64_t k = 0; k < numSocCones; k++) {
        auto block = sampleSocInterior(rng, socDims[k], 1.0);
        for (int64_t i = 0; i < socDims[k]; i++) {
            s_star[soc_offset + i] = block[i];
        }
        soc_offset += socDims[k];
    }

    std::vector<double> b(m, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) b[i] += A_dense[i][j] * x_star[j];
        b[i] += s_star[i];
    }

    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;
    for (int64_t i = 0; i < n; i++) {
        P_ro[i] = i;
        P_ci.push_back(i);
        P_val.push_back(1.0);
    }
    P_ro[n] = n;

    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;
    A_ro.push_back(0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            A_ci.push_back(j);
            A_val.push_back(A_dense[i][j]);
        }
        A_ro.push_back(static_cast<int64_t>(A_val.size()));
    }

    int64_t nnzP = static_cast<int64_t>(P_val.size());
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    Cones cones{};
    cones.numNonnegCones = numNonneg;
    cones.socConeDims = socDims;
    cones.numSocCones = numSocCones;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    std::vector<double> s_sol(m);
    std::vector<int32_t> status(1);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Status = " << status[0];

    double max_x_err = 0.0;
    for (int64_t i = 0; i < n; i++)
        max_x_err = std::max(max_x_err, std::abs(x_sol[i] - x_star[i]));
    EXPECT_LT(max_x_err, 1e-4) << "max |x_sol - x_star| = " << max_x_err;

    // Check nonneg cone feasibility: s[0..numNonneg-1] >= 0
    for (int64_t i = 0; i < numNonneg; i++) {
        EXPECT_GE(s_sol[i], -1e-6) << "s[" << i << "] = " << s_sol[i] << " violates nonneg";
    }

    // Check SOC cone feasibility (starts after nonneg)
    EXPECT_TRUE(isInSocProductVarLen(s_sol, socDims, numNonneg, 1e-6))
        << "s not in product of variable-dim SOC cones";

    double max_res = 0.0;
    for (int64_t i = 0; i < m; i++) {
        double r = -b[i] + s_sol[i];
        for (int64_t j = 0; j < n; j++) r += A_dense[i][j] * x_sol[j];
        max_res = std::max(max_res, std::abs(r));
    }
    EXPECT_LT(max_res, 1e-5) << "primal residual = " << max_res;

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * @brief Test 6: Setup + solve workflow for variable-length SOC cones with updates.
 *
 * Uses one-time setup of P/A with cones {2, 5}, then two solves with
 * updated q and b to exercise the three-step API path for reused symbolic structure.
 */
TEST_F(SOCVarLenTest, SetupSolveThenUpdateQb) {
    const int64_t n = 4;
    const int64_t m = 7;
    const int64_t batchSize = 1;
    const std::vector<int64_t> socDims = {2, 5};
    const int64_t numSocCones = 2;
    const uint64_t seed = 700;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    // Shared A matrix for both solves (dense rows with 2 nonzeros each, keep deterministic)
    std::vector<std::vector<double>> A_dense(m, std::vector<double>(n));
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            A_dense[i][j] = normal(rng);
        }
    }

    // Convert dense A to compact CSR with 2 entries per row.
    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;
    A_ro.reserve(static_cast<size_t>(m + 1));
    A_ro.push_back(0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < 2; j++) {
            int64_t col = j % n;
            A_ci.push_back(col);
            A_val.push_back(A_dense[i][col]);
        }
        A_ro.push_back(static_cast<int64_t>(A_val.size()));
    }

    // P = I in upper-triangular CSR format
    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;
    P_ro[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        P_ci.push_back(i);
        P_val.push_back(1.0);
        P_ro[i + 1] = i + 1;
    }
    int64_t nnzP = static_cast<int64_t>(P_val.size());
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    // Build two target solutions with different optima.
    std::vector<double> x_star1(n);
    for (int64_t j = 0; j < n; j++) {
        x_star1[j] = normal(rng);
    }
    std::vector<double> s_star1(m);
    auto block2a = sampleSocInterior(rng, socDims[0], 1.0);
    auto block5a = sampleSocInterior(rng, socDims[1], 1.0);
    for (int64_t i = 0; i < socDims[0]; i++) s_star1[i] = block2a[i];
    for (int64_t i = 0; i < socDims[1]; i++) s_star1[socDims[0] + i] = block5a[i];

    std::vector<double> q1(n);
    for (int64_t j = 0; j < n; j++) q1[j] = -x_star1[j];

    std::vector<double> x_star2 = x_star1;
    x_star2[0] += 0.75;
    x_star2[2] -= 0.50;
    std::vector<double> s_star2(m);
    auto block2b = sampleSocInterior(rng, socDims[0], 1.0);
    auto block5b = sampleSocInterior(rng, socDims[1], 1.0);
    for (int64_t i = 0; i < socDims[0]; i++) s_star2[i] = block2b[i];
    for (int64_t i = 0; i < socDims[1]; i++) s_star2[socDims[0] + i] = block5b[i];

    std::vector<double> q2(n);
    for (int64_t j = 0; j < n; j++) q2[j] = -x_star2[j];

    std::vector<double> b1(m, 0.0), b2(m, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < 2; j++) b1[i] += A_dense[i][j] * x_star1[j];
        for (int64_t j = 0; j < 2; j++) b2[i] += A_dense[i][j] * x_star2[j];
        b1[i] += s_star1[i];
        b2[i] += s_star2[i];
    }

    Cones cones{};
    cones.socConeDims = socDims;
    cones.numSocCones = numSocCones;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);

    solver.setup(d_P, d_A);

    cudaMemcpy(d_q, q1.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b1.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    solver.solve(d_q, d_b);

    std::vector<double> x1(n);
    std::vector<double> s1(m);
    std::vector<int32_t> status(1);
    cudaMemcpy(x1.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s1.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);

    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "first solve status = " << status[0];

    double first_x_err = 0.0;
    for (int64_t i = 0; i < n; i++) {
        first_x_err = std::max(first_x_err, std::abs(x1[i] - x_star1[i]));
    }
    EXPECT_LT(first_x_err, 1e-4);

    // Update q/b while keeping same structure and P/A (three-step API update path).
    solver.setup(d_P, d_A);
    cudaMemcpy(d_q, q2.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b2.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    solver.solve(d_q, d_b);

    std::vector<double> x2(n);
    std::vector<double> s2(m);
    cudaMemcpy(x2.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s2.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);

    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "second solve status = " << status[0];

    double second_x_err = 0.0;
    for (int64_t i = 0; i < n; i++) {
        second_x_err = std::max(second_x_err, std::abs(x2[i] - x_star2[i]));
    }
    EXPECT_LT(second_x_err, 1e-4);

    bool changed = false;
    for (int64_t i = 0; i < n; i++) {
        if (std::abs(x1[i] - x2[i]) > 1e-6) changed = true;
    }
    EXPECT_TRUE(changed) << "setup->solve update path did not change the solution";

    double first_res = 0.0;
    double second_res = 0.0;
    for (int64_t i = 0; i < m; i++) {
        double r1 = -b1[i] + s1[i];
        double r2 = -b2[i] + s2[i];
        for (int64_t j = 0; j < 2; j++) {
            r1 += A_dense[i][j] * x1[j];
            r2 += A_dense[i][j] * x2[j];
        }
        first_res = std::max(first_res, std::abs(r1));
        second_res = std::max(second_res, std::abs(r2));
    }
    EXPECT_LT(first_res, 1e-5);
    EXPECT_LT(second_res, 1e-5);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * @brief Test 7: Single SOC with dim=100 (large sparse SOC).
 *
 * SOC(100) = { (t, u1, ..., u99) : t >= ||u|| }
 *
 * Tests that large-dimension SOC cones work correctly with sparse Hs
 * representation and that all kernel loops handle large dims properly.
 *
 * n=50, m=100, 1 SOC cone of dim 100.
 */
TEST_F(SOCVarLenTest, Dim100SOCConverges) {
    const int64_t socDim = 100;
    const int64_t n = 50;
    const int64_t m = socDim;
    const int64_t batchSize = 1;
    const uint64_t seed = 800;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> x_star(n);
    for (int64_t j = 0; j < n; j++) x_star[j] = normal(rng);

    std::vector<double> q(n);
    for (int64_t j = 0; j < n; j++) q[j] = -x_star[j];

    std::vector<std::vector<double>> A_dense(m, std::vector<double>(n));
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            A_dense[i][j] = normal(rng);

    std::vector<double> s_star = sampleSocInterior(rng, socDim, 0.3);

    std::vector<double> b(m, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) b[i] += A_dense[i][j] * x_star[j];
        b[i] += s_star[i];
    }

    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;
    for (int64_t i = 0; i < n; i++) {
        P_ro[i] = i;
        P_ci.push_back(i);
        P_val.push_back(1.0);
    }
    P_ro[n] = n;

    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;
    A_ro.push_back(0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            A_ci.push_back(j);
            A_val.push_back(A_dense[i][j]);
        }
        A_ro.push_back(static_cast<int64_t>(A_val.size()));
    }

    int64_t nnzP = static_cast<int64_t>(P_val.size());
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    Cones cones{};
    cones.socConeDims = {socDim};
    cones.numSocCones = 1;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    std::vector<double> s_sol(m);
    std::vector<int32_t> status(1);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Status = " << status[0];

    double max_x_err = 0.0;
    for (int64_t i = 0; i < n; i++)
        max_x_err = std::max(max_x_err, std::abs(x_sol[i] - x_star[i]));
    EXPECT_LT(max_x_err, 1e-4) << "max |x_sol - x_star| = " << max_x_err;

    EXPECT_TRUE(isInSoc(s_sol.data(), socDim, 1e-6)) << "s not in SOC(100)";

    double max_res = 0.0;
    for (int64_t i = 0; i < m; i++) {
        double r = -b[i] + s_sol[i];
        for (int64_t j = 0; j < n; j++) r += A_dense[i][j] * x_sol[j];
        max_res = std::max(max_res, std::abs(r));
    }
    EXPECT_LT(max_res, 1e-5) << "primal residual = " << max_res;

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * @brief Test 8: Mixed dense/sparse SOC cones with nonneg + large SOC.
 *
 * Cone product: 5 nonneg, SOC(2), SOC(4), SOC(50).
 * Total m = 5 + 2 + 4 + 50 = 61.
 *
 * Exercises dense SOC (dim=2,4) and large sparse SOC (dim=50) together
 * with nonneg cones.
 */
TEST_F(SOCVarLenTest, MixedDenseSparseLargeWithNonneg) {
    const int64_t n = 40;
    const int64_t numNonneg = 5;
    const std::vector<int64_t> socDims = {2, 4, 50};
    const int64_t numSocCones = 3;
    int64_t totalSocDim = 0;
    for (auto d : socDims) totalSocDim += d;
    const int64_t m = numNonneg + totalSocDim;
    const int64_t batchSize = 1;
    const uint64_t seed = 900;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> x_star(n);
    for (int64_t j = 0; j < n; j++) x_star[j] = normal(rng);

    std::vector<double> q(n);
    for (int64_t j = 0; j < n; j++) q[j] = -x_star[j];

    std::vector<std::vector<double>> A_dense(m, std::vector<double>(n));
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            A_dense[i][j] = normal(rng);

    std::vector<double> s_star(m);
    std::uniform_real_distribution<double> unif(0.5, 2.0);
    for (int64_t i = 0; i < numNonneg; i++) {
        s_star[i] = unif(rng);
    }
    int64_t soc_offset = numNonneg;
    for (int64_t k = 0; k < numSocCones; k++) {
        auto block = sampleSocInterior(rng, socDims[k], 1.0);
        for (int64_t i = 0; i < socDims[k]; i++) {
            s_star[soc_offset + i] = block[i];
        }
        soc_offset += socDims[k];
    }

    std::vector<double> b(m, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) b[i] += A_dense[i][j] * x_star[j];
        b[i] += s_star[i];
    }

    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;
    for (int64_t i = 0; i < n; i++) {
        P_ro[i] = i;
        P_ci.push_back(i);
        P_val.push_back(1.0);
    }
    P_ro[n] = n;

    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;
    A_ro.push_back(0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            A_ci.push_back(j);
            A_val.push_back(A_dense[i][j]);
        }
        A_ro.push_back(static_cast<int64_t>(A_val.size()));
    }

    int64_t nnzP = static_cast<int64_t>(P_val.size());
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    Cones cones{};
    cones.numNonnegCones = numNonneg;
    cones.socConeDims = socDims;
    cones.numSocCones = numSocCones;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    std::vector<double> s_sol(m);
    std::vector<int32_t> status(1);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Status = " << status[0];

    double max_x_err = 0.0;
    for (int64_t i = 0; i < n; i++)
        max_x_err = std::max(max_x_err, std::abs(x_sol[i] - x_star[i]));
    EXPECT_LT(max_x_err, 1e-4) << "max |x_sol - x_star| = " << max_x_err;

    for (int64_t i = 0; i < numNonneg; i++) {
        EXPECT_GE(s_sol[i], -1e-6) << "s[" << i << "] = " << s_sol[i] << " violates nonneg";
    }

    EXPECT_TRUE(isInSocProductVarLen(s_sol, socDims, numNonneg, 1e-6))
        << "s not in product of variable-dim SOC cones";

    double max_res = 0.0;
    for (int64_t i = 0; i < m; i++) {
        double r = -b[i] + s_sol[i];
        for (int64_t j = 0; j < n; j++) r += A_dense[i][j] * x_sol[j];
        max_res = std::max(max_res, std::abs(r));
    }
    EXPECT_LT(max_res, 1e-5) << "primal residual = " << max_res;

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * @brief Test 9: Verify sparse SOC expansion Schur complement correctness.
 *
 * For a SOC cone with dim > 4 (sparse expansion), the scaling matrix is:
 *   H = eta^2 * (diag(d, 1, ..., 1) + uu^T - vv^T)
 *
 * In the KKT system this is represented via two expansion columns/rows.
 * The Schur complement of eliminating expansion variables should recover H.
 *
 * This test constructs a SOC(10) problem, solves it, then reads back the
 * cone scaling data and verifies the Schur complement identity.
 */
TEST_F(SOCVarLenTest, KKTSchurComplementVerification) {
    const int64_t dim = 10;
    const int64_t n = dim;
    const int64_t m = dim;
    const std::vector<int64_t> socDims = {dim};
    const int64_t numSocCones = 1;
    const int64_t batchSize = 1;

    // P = I (diagonal CSR)
    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci(n);
    std::vector<double> P_val(n, 1.0);
    for (int64_t i = 0; i <= n; i++) P_ro[i] = i;
    for (int64_t i = 0; i < n; i++) P_ci[i] = i;

    // A = I (diagonal CSR)
    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci(m);
    std::vector<double> A_val(m, 1.0);
    for (int64_t i = 0; i <= m; i++) A_ro[i] = i;
    for (int64_t i = 0; i < m; i++) A_ci[i] = i;

    int64_t nnzP = static_cast<int64_t>(P_val.size());
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    // q: push solution toward SOC interior
    std::vector<double> q(n, 0.0);
    q[0] = -2.0;  // large s[0] component
    for (int64_t i = 1; i < n; i++) q[i] = 0.1;

    // b: point in SOC interior
    std::vector<double> b(m, 0.0);
    b[0] = 3.0;
    for (int64_t i = 1; i < m; i++) b[i] = 0.1;

    Cones cones{};
    cones.socConeDims = socDims;
    cones.numSocCones = numSocCones;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = true;
    settings.ipm.kktSolverType = KKTSolverType::CuDSS;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::cout << "[SchurComplement] status = " << status[0] << std::endl;

    // Read back cone scaling data from solver.data.cones
    auto& solverCones = solver.data.cones;

    std::vector<double> h_w(dim), h_u(dim), h_v(dim);
    std::vector<double> h_eta(numSocCones), h_d(numSocCones);

    cudaMemcpy(h_w.data(), solverCones.soc_w.data(), dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_u.data(), solverCones.soc_u.data(), dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v.data(), solverCones.soc_v.data(), dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_eta.data(), solverCones.soc_eta.data(), numSocCones * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d.data(), solverCones.soc_d.data(), numSocCones * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double eta = h_eta[0];
    double d_val = h_d[0];
    double eta_sq = eta * eta;

    std::cout << "[SchurComplement] eta = " << eta << ", d = " << d_val << std::endl;
    std::cout << "[SchurComplement] w[0] = " << h_w[0] << std::endl;
    std::cout << "[SchurComplement] u[0] = " << h_u[0] << ", u[1] = " << h_u[1] << std::endl;
    std::cout << "[SchurComplement] v[0] = " << h_v[0] << ", v[1] = " << h_v[1] << std::endl;

    // Verify: H_sparse = eta^2 * (diag(d, 1, ..., 1) + uu^T - vv^T)
    // H_dense  = eta^2 * (2*ww^T - J),  J = diag(1, -1, ..., -1)
    // They should be equal.
    double max_diff = 0.0;
    for (int64_t i = 0; i < dim; i++) {
        for (int64_t j = 0; j < dim; j++) {
            // Sparse form
            double diag_ij = (i == j) ? ((i == 0) ? d_val : 1.0) : 0.0;
            double H_sparse = eta_sq * (diag_ij + h_u[i] * h_u[j] - h_v[i] * h_v[j]);

            // Dense form
            double J_ij = (i == j) ? ((i == 0) ? 1.0 : -1.0) : 0.0;
            double H_dense = eta_sq * (2.0 * h_w[i] * h_w[j] - J_ij);

            double diff = std::abs(H_sparse - H_dense);
            if (diff > max_diff) max_diff = diff;

            double scale = std::max(std::abs(H_sparse), std::abs(H_dense));
            double rel_diff = (scale > 1e-15) ? diff / scale : diff;
            if (rel_diff > max_diff) max_diff = rel_diff;

            if (rel_diff > 1e-10) {
                std::cout << "[SchurComplement] MISMATCH H[" << i << "][" << j
                          << "]: sparse=" << H_sparse << " dense=" << H_dense
                          << " diff=" << diff << " rel=" << rel_diff << std::endl;
            }
        }
    }
    std::cout << "[SchurComplement] max relative |H_sparse - H_dense| = " << max_diff << std::endl;
    EXPECT_LT(max_diff, 1e-6) << "Sparse and dense SOC scaling matrices disagree";

    // Also verify w is a valid SOC point: w[0] = sqrt(1 + ||w[1:]||^2)
    double w_tail_sq = 0.0;
    for (int64_t i = 1; i < dim; i++) w_tail_sq += h_w[i] * h_w[i];
    double w0_expected = sqrt(1.0 + w_tail_sq);
    EXPECT_NEAR(h_w[0], w0_expected, 1e-10) << "w[0] not normalized";

    // Verify u, v, d consistency with w (from CPU formulas in socone.rs)
    double w0 = h_w[0];
    double w1sq = w_tail_sq;
    double wsq = w0 * w0 + w1sq;
    double wsqinv = 1.0 / wsq;
    double d_expected = 0.5 * wsqinv;
    EXPECT_NEAR(d_val, d_expected, 1e-10) << "d doesn't match w";

    double u0_expected = sqrt(wsq - d_expected);
    EXPECT_NEAR(h_u[0], u0_expected, 1e-10) << "u[0] doesn't match w";

    double u_alpha = 2.0 * w0;
    double u1_scale = u_alpha / u0_expected;
    for (int64_t i = 1; i < dim; i++) {
        double ui_expected = u1_scale * h_w[i];
        EXPECT_NEAR(h_u[i], ui_expected, 1e-10) << "u[" << i << "] doesn't match w";
    }

    EXPECT_NEAR(h_v[0], 0.0, 1e-10) << "v[0] should be 0";
    double v1_scale = sqrt(2.0 * (2.0 + wsqinv) / (2.0 * wsq - wsqinv));
    for (int64_t i = 1; i < dim; i++) {
        double vi_expected = v1_scale * h_w[i];
        EXPECT_NEAR(h_v[i], vi_expected, 1e-10) << "v[" << i << "] doesn't match w";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * @brief Test 10: Large sparse SOC dim=20 convergence test.
 *
 * Minimal reproducer for the convergence failure with large sparse SOC cones.
 * SOC(20) with cuDSS solver.
 */
TEST_F(SOCVarLenTest, LargeSparseSocDim20Converges) {
    const int64_t dim = 20;
    const int64_t n = dim;
    const int64_t m = dim;
    const std::vector<int64_t> socDims = {dim};
    const int64_t numSocCones = 1;
    const int64_t batchSize = 1;

    // P = I (diagonal CSR)
    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci(n);
    std::vector<double> P_val(n, 1.0);
    for (int64_t i = 0; i <= n; i++) P_ro[i] = i;
    for (int64_t i = 0; i < n; i++) P_ci[i] = i;

    // A = I (diagonal CSR)
    std::vector<int64_t> A_ro(m + 1);
    std::vector<int64_t> A_ci(m);
    std::vector<double> A_val(m, 1.0);
    for (int64_t i = 0; i <= m; i++) A_ro[i] = i;
    for (int64_t i = 0; i < m; i++) A_ci[i] = i;

    int64_t nnzP = static_cast<int64_t>(P_val.size());
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    // q: push solution toward SOC interior
    std::vector<double> q(n, 0.0);
    q[0] = -2.0;
    for (int64_t i = 1; i < n; i++) q[i] = 0.1;

    // b: point in SOC interior
    std::vector<double> b(m, 0.0);
    b[0] = 3.0;
    for (int64_t i = 1; i < m; i++) b[i] = 0.1;

    Cones cones{};
    cones.socConeDims = socDims;
    cones.numSocCones = numSocCones;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = true;
    settings.ipm.kktSolverType = KKTSolverType::CuDSS;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::cout << "[SOC20] status = " << status[0] << std::endl;

    // Read back solution
    std::vector<double> x_sol(n), s_sol(m);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Check if solver converged
    bool converged = (status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                      status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved));

    if (!converged) {
        std::cout << "[SOC20] FAILED TO CONVERGE - dumping diagnostics:" << std::endl;
        std::cout << "[SOC20] x = [";
        for (int64_t i = 0; i < n; i++) std::cout << x_sol[i] << (i < n-1 ? ", " : "");
        std::cout << "]" << std::endl;
        std::cout << "[SOC20] s = [";
        for (int64_t i = 0; i < m; i++) std::cout << s_sol[i] << (i < m-1 ? ", " : "");
        std::cout << "]" << std::endl;

        // Read cone scaling
        auto& solverCones = solver.data.cones;
        std::vector<double> h_w(dim), h_u(dim), h_v(dim);
        std::vector<double> h_eta(1), h_d(1);
        cudaMemcpy(h_w.data(), solverCones.soc_w.data(), dim * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_u.data(), solverCones.soc_u.data(), dim * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_v.data(), solverCones.soc_v.data(), dim * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_eta.data(), solverCones.soc_eta.data(), sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_d.data(), solverCones.soc_d.data(), sizeof(double), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        std::cout << "[SOC20] eta = " << h_eta[0] << ", d = " << h_d[0] << std::endl;
        std::cout << "[SOC20] w[0:3] = " << h_w[0] << ", " << h_w[1] << ", " << h_w[2] << std::endl;
        std::cout << "[SOC20] u[0:3] = " << h_u[0] << ", " << h_u[1] << ", " << h_u[2] << std::endl;
        std::cout << "[SOC20] v[0:3] = " << h_v[0] << ", " << h_v[1] << ", " << h_v[2] << std::endl;
    }

    EXPECT_TRUE(converged) << "SOC(20) solver did not converge, status=" << status[0];

    if (converged) {
        // Verify s is in SOC
        EXPECT_TRUE(isInSocProductVarLen(s_sol, socDims, 0, 1e-6))
            << "s not in SOC";

        // Check primal residual
        double max_res = 0.0;
        for (int64_t i = 0; i < m; i++) {
            double r = x_sol[i] + s_sol[i] - b[i];  // A=I
            max_res = std::max(max_res, std::abs(r));
        }
        EXPECT_LT(max_res, 1e-5) << "primal residual = " << max_res;
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * @brief Test 11: Verify identity scaling consistency for sparse SOC cones.
 *
 * After set_identity_scaling, the soc_Hs diagonal for sparse cones should
 * be [d, 1, 1, ..., 1] (with d=0.5), NOT [1, 1, ..., 1].
 * Combined with the expansion columns (u, v, eta), the total effective
 * Hs must equal the identity matrix.
 */
TEST_F(SOCVarLenTest, SparseIdentityScalingConsistency) {
    const int64_t dim = 10;
    const int64_t n = dim;
    const int64_t m = dim;
    const std::vector<int64_t> socDims = {dim};
    const int64_t numSocCones = 1;
    const int64_t batchSize = 1;

    std::vector<int64_t> P_ro(n + 1), P_ci(n);
    std::vector<double> P_val(n, 1.0);
    for (int64_t i = 0; i <= n; i++) P_ro[i] = i;
    for (int64_t i = 0; i < n; i++) P_ci[i] = i;

    std::vector<int64_t> A_ro(m + 1), A_ci(m);
    std::vector<double> A_val(m, 1.0);
    for (int64_t i = 0; i <= m; i++) A_ro[i] = i;
    for (int64_t i = 0; i < m; i++) A_ci[i] = i;

    int64_t nnzP = n, nnzA = m;

    Cones cones{};
    cones.socConeDims = socDims;
    cones.numSocCones = numSocCones;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.kktSolverType = KKTSolverType::CuDSS;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings
    );

    // After construction, set_identity_scaling is called during default_start.
    // We can inspect the cone data directly after calling it.
    auto& c = solver.data.cones;
    c.set_identity_scaling();

    // Read back soc_Hs, u, v, d, eta
    int64_t hsEntries = c.totalSocHsEntries;
    std::vector<double> h_Hs(hsEntries);
    std::vector<double> h_u(dim), h_v(dim);
    std::vector<double> h_eta(1), h_d(1);

    cudaMemcpy(h_Hs.data(), c.soc_Hs.data(), hsEntries * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_u.data(), c.soc_u.data(), dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v.data(), c.soc_v.data(), dim * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_eta.data(), c.soc_eta.data(), sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_d.data(), c.soc_d.data(), sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double eta = h_eta[0];
    double d_val = h_d[0];
    double eta_sq = eta * eta;

    // For identity scaling: eta=1, d=0.5, u=(1/sqrt(2), 0, ...), v=0
    EXPECT_NEAR(eta, 1.0, 1e-12);
    EXPECT_NEAR(d_val, 0.5, 1e-12);
    EXPECT_NEAR(h_u[0], 1.0 / std::sqrt(2.0), 1e-12);
    for (int64_t i = 1; i < dim; i++) {
        EXPECT_NEAR(h_u[i], 0.0, 1e-12) << "u[" << i << "] should be 0";
    }
    for (int64_t i = 0; i < dim; i++) {
        EXPECT_NEAR(h_v[i], 0.0, 1e-12) << "v[" << i << "] should be 0";
    }

    // soc_Hs diagonal should be [0.5, 1, 1, ..., 1] (NOT [1, 1, ..., 1])
    // Because soc_Hs stores eta^2 * diag(d, 1, ..., 1), and the expansion adds eta^2*(uu' - vv')
    EXPECT_NEAR(h_Hs[0], 0.5, 1e-12) << "soc_Hs[0] should be d=0.5, not 1.0";
    for (int64_t i = 1; i < dim; i++) {
        EXPECT_NEAR(h_Hs[i], 1.0, 1e-12) << "soc_Hs[" << i << "] should be 1.0";
    }

    // Verify the effective Hs = diag(soc_Hs) + eta^2*(uu' - vv') = I
    for (int64_t i = 0; i < dim; i++) {
        for (int64_t j = 0; j < dim; j++) {
            double diag_ij = (i == j) ? h_Hs[i] : 0.0;
            double H_eff = diag_ij + eta_sq * (h_u[i] * h_u[j] - h_v[i] * h_v[j]);
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(H_eff, expected, 1e-12)
                << "Effective Hs[" << i << "][" << j << "] = " << H_eff
                << ", expected identity = " << expected;
        }
    }
}

/**
 * @brief Exercise the block-per-cone SOC scaling path.
 *
 * Two large cones (dim=256, dim=128) mixed with two small cones (dim=3, dim=4)
 * and a few nonneg constraints. The small cones stay in the composite
 * thread-per-cone kernel; the large cones route through the new parallel
 * block-per-cone kernel. This simultaneously tests:
 *   - correct partitioning (numLargeSoc count, small/large kernel split),
 *   - grid.x = numLargeSoc multi-block launch in the large kernel,
 *   - sorted-ascending layout → sparse offsets match across kernels.
 */
TEST_F(SOCVarLenTest, BlockPerConeLargeAndSmallMixed) {
    const int64_t numNonneg = 3;
    const std::vector<int64_t> socDims = {3, 4, 128, 256};
    const int64_t n = 60;
    int64_t totalSocDim = 0;
    for (auto d : socDims) totalSocDim += d;
    const int64_t m = numNonneg + totalSocDim;
    const int64_t batchSize = 1;
    const uint64_t seed = 2025;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> x_star(n);
    for (int64_t j = 0; j < n; j++) x_star[j] = normal(rng);

    std::vector<double> q(n);
    for (int64_t j = 0; j < n; j++) q[j] = -x_star[j];

    std::vector<std::vector<double>> A_dense(m, std::vector<double>(n));
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            A_dense[i][j] = normal(rng);

    std::vector<double> s_star(m, 0.0);
    for (int64_t i = 0; i < numNonneg; i++) s_star[i] = 0.3;
    int64_t offset = numNonneg;
    for (auto dim : socDims) {
        auto block = sampleSocInterior(rng, dim, 0.5);
        for (int64_t i = 0; i < dim; i++) s_star[offset + i] = block[i];
        offset += dim;
    }

    std::vector<double> b(m, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) b[i] += A_dense[i][j] * x_star[j];
        b[i] += s_star[i];
    }

    std::vector<int64_t> P_ro(n + 1);
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;
    for (int64_t i = 0; i < n; i++) {
        P_ro[i] = i;
        P_ci.push_back(i);
        P_val.push_back(1.0);
    }
    P_ro[n] = n;

    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;
    A_ro.push_back(0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            A_ci.push_back(j);
            A_val.push_back(A_dense[i][j]);
        }
        A_ro.push_back(static_cast<int64_t>(A_val.size()));
    }

    int64_t nnzP = static_cast<int64_t>(P_val.size());
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    Cones cones{};
    cones.socConeDims = socDims;
    cones.numSocCones = static_cast<int64_t>(socDims.size());
    cones.numNonnegCones = numNonneg;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings
    );

    // Verify the partition was computed correctly: dim=3,4 small; dim=128,256 large.
    EXPECT_EQ(solver.data.cones.numLargeSoc, 2);
    EXPECT_EQ(solver.data.cones.numSocCones, 4);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    std::vector<double> s_sol(m);
    std::vector<int32_t> status(1);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Status = " << status[0];

    EXPECT_TRUE(isInSocProductVarLen(s_sol, socDims, numNonneg, 1e-5))
        << "s not in product of SOC cones";
    for (int64_t i = 0; i < numNonneg; i++) {
        EXPECT_GE(s_sol[i], -1e-6) << "nonneg s[" << i << "] = " << s_sol[i];
    }

    double max_res = 0.0;
    for (int64_t i = 0; i < m; i++) {
        double r = -b[i] + s_sol[i];
        for (int64_t j = 0; j < n; j++) r += A_dense[i][j] * x_sol[j];
        max_res = std::max(max_res, std::abs(r));
    }
    EXPECT_LT(max_res, 1e-5) << "primal residual = " << max_res;

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
