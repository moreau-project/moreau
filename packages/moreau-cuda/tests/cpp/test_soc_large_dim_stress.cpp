/**
 * @file test_soc_large_dim_stress.cpp
 * @brief Stress tests for large-dimension SOC cones that expose cuDSS limitations.
 *
 * These tests construct random feasible QPs with large SOC cones (dim > 4,
 * triggering sparse expansion in the KKT system). They use dense random P,
 * ill-conditioned A matrices, and near-boundary SOC slack to create hard
 * instances that cuDSS struggles with.
 *
 * Tests are DISABLED_ because they expose known cuDSS limitations.
 * Run with --gtest_also_run_disabled_tests to execute them.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/solver.hpp"

using namespace moreau;

namespace {

struct StressProblem {
    int64_t n, m;
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double> P_val;
    std::vector<int64_t> A_ro, A_ci;
    std::vector<double> A_val;
    std::vector<double> q, b;
    Cones cones;
};

/**
 * @brief Build a hard random QP designed to stress cuDSS sparse SOC expansion.
 *
 * Key ingredients:
 * 1. Dense random P = L*L' + eps*I
 * 2. Ill-conditioned A — rows scaled by powers of 10
 * 3. Near-boundary SOC slack
 * 4. Random q not aligned with feasible solution
 */
StressProblem buildHardProblem(
    const std::vector<int64_t>& socDims,
    int64_t numNonneg,
    int64_t n,
    uint64_t seed,
    double soc_margin = 0.001,
    double cond_range = 4.0)
{
    StressProblem prob;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_real_distribution<double> uniform01(0.0, 1.0);

    int64_t totalSocDim = 0;
    for (auto d : socDims) totalSocDim += d;
    int64_t m = totalSocDim + numNonneg;

    prob.n = n;
    prob.m = m;

    // P: dense symmetric PD = L*L' + 0.1*I
    std::vector<double> P_dense(n * n, 0.0);
    std::vector<double> L(n * n, 0.0);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j <= i; j++) {
            L[i * n + j] = normal(rng) * 0.3;
        }
        L[i * n + i] += 1.0;
    }
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j <= i; j++) {
            double val = 0.0;
            for (int64_t k = 0; k <= std::min(i, j); k++) {
                val += L[i * n + k] * L[j * n + k];
            }
            P_dense[i * n + j] = val;
            P_dense[j * n + i] = val;
        }
    }
    for (int64_t i = 0; i < n; i++) P_dense[i * n + i] += 0.1;

    // P CSR (full symmetric)
    prob.P_ro.resize(n + 1);
    prob.P_ci.resize(n * n);
    prob.P_val.resize(n * n);
    for (int64_t i = 0; i <= n; i++) prob.P_ro[i] = i * n;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            prob.P_ci[i * n + j] = j;
            prob.P_val[i * n + j] = P_dense[i * n + j];
        }
    }

    // A: random dense m x n with ill-conditioned row scaling
    std::vector<double> A_dense(m * n);
    for (int64_t i = 0; i < m; i++) {
        double log_scale = (uniform01(rng) - 0.5) * cond_range;
        double scale = std::pow(10.0, log_scale);
        for (int64_t j = 0; j < n; j++) {
            A_dense[i * n + j] = normal(rng) * scale;
        }
    }

    // x_star: random
    std::vector<double> x_star(n);
    for (int64_t j = 0; j < n; j++) x_star[j] = normal(rng);

    // s_star: near SOC boundary
    std::vector<double> s_star(m, 0.0);
    int64_t offset = 0;
    for (auto dim : socDims) {
        double norm_sq = 0.0;
        for (int64_t i = 1; i < dim; i++) {
            s_star[offset + i] = normal(rng);
            norm_sq += s_star[offset + i] * s_star[offset + i];
        }
        s_star[offset] = std::sqrt(norm_sq) + soc_margin;
        offset += dim;
    }
    for (int64_t i = 0; i < numNonneg; i++) {
        s_star[offset + i] = soc_margin;
    }

    // q: random
    prob.q.resize(n);
    for (int64_t j = 0; j < n; j++) prob.q[j] = normal(rng) * 2.0;

    // b = A * x_star + s_star
    prob.b.resize(m, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            prob.b[i] += A_dense[i * n + j] * x_star[j];
        }
        prob.b[i] += s_star[i];
    }

    // A CSR
    prob.A_ro.resize(m + 1);
    prob.A_ci.resize(m * n);
    prob.A_val.resize(m * n);
    for (int64_t i = 0; i <= m; i++) prob.A_ro[i] = i * n;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            prob.A_ci[i * n + j] = j;
            prob.A_val[i * n + j] = A_dense[i * n + j];
        }
    }

    prob.cones = Cones{};
    prob.cones.socConeDims = socDims;
    prob.cones.numSocCones = static_cast<int64_t>(socDims.size());
    prob.cones.numNonnegCones = numNonneg;

    return prob;
}

std::pair<int32_t, std::string> solveAndReport(const StressProblem& prob, const std::string& label) {
    int64_t nnzP = static_cast<int64_t>(prob.P_val.size());
    int64_t nnzA = static_cast<int64_t>(prob.A_val.size());

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    Cones cones = prob.cones;

    CompiledSolver solver(
        prob.n, prob.m, /*batchSize=*/1,
        prob.P_ro.data(), prob.P_ci.data(), nnzP,
        prob.A_ro.data(), prob.A_ci.data(), nnzA,
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * prob.n);
    cudaMalloc(&d_b, sizeof(double) * prob.m);

    cudaMemcpy(d_P, prob.P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);

    auto st = static_cast<SolverStatus>(status[0]);
    std::string name;
    switch (st) {
        case SolverStatus::Solved: name = "Solved"; break;
        case SolverStatus::AlmostSolved: name = "AlmostSolved"; break;
        case SolverStatus::MaxIterations: name = "MaxIterations"; break;
        case SolverStatus::InsufficientProgress: name = "InsufficientProgress"; break;
        case SolverStatus::NumericalError: name = "NumericalError"; break;
        case SolverStatus::PrimalInfeasible: name = "PrimalInfeasible"; break;
        case SolverStatus::DualInfeasible: name = "DualInfeasible"; break;
        default: name = "Unknown(" + std::to_string(status[0]) + ")"; break;
    }
    std::cout << "  [" << label << "] n=" << prob.n << " m=" << prob.m
              << " status=" << name << std::endl;

    return {status[0], name};
}

inline bool isSolved(int32_t status) {
    return status == static_cast<int32_t>(SolverStatus::Solved) ||
           status == static_cast<int32_t>(SolverStatus::AlmostSolved);
}

int runSeedSweep(
    const std::vector<int64_t>& socDims,
    int64_t numNonneg,
    int64_t n,
    uint64_t seedStart,
    int nSeeds,
    const std::string& prefix,
    double soc_margin = 0.001,
    double cond_range = 4.0)
{
    int failures = 0;
    for (uint64_t seed = seedStart; seed < seedStart + static_cast<uint64_t>(nSeeds); seed++) {
        auto prob = buildHardProblem(socDims, numNonneg, n, seed, soc_margin, cond_range);
        auto [status, name] = solveAndReport(prob, prefix + std::to_string(seed));
        if (!isSolved(status)) failures++;
    }
    return failures;
}

} // namespace

class SOCLargeDimStressTest : public ::testing::Test {
protected:
    void SetUp() override { cudaGetLastError(); }
    void TearDown() override { cudaDeviceSynchronize(); }
};

/**
 * Case 1: Single SOC dim=50, square A (n=m=50).
 * Dense P, ill-conditioned A, near-boundary SOC slack.
 * The 50x50 KKT system with 2 expansion rows is small enough that
 * cuDSS's lack of pivoting causes InsufficientProgress.
 */
TEST_F(SOCLargeDimStressTest, DISABLED_SingleDim50_Square) {
    constexpr int N = 20;
    int f = runSeedSweep({50}, 0, 50, 100, N, "dim50_s");
    EXPECT_EQ(f, 0) << f << "/" << N << " failed — cuDSS with dim=50 SOC, square A";
}

/**
 * Case 2: Five SOC cones [10]*5, square A (n=m=50).
 * Same n=50 as Case 1 but split into 5 cones → 10 expansion rows.
 * Tests whether more expansion rows (per KKT dimension) makes things worse.
 */
TEST_F(SOCLargeDimStressTest, DISABLED_FiveDim10_Square) {
    constexpr int N = 20;
    int f = runSeedSweep({10, 10, 10, 10, 10}, 0, 50, 200, N, "5x10_s");
    EXPECT_EQ(f, 0) << f << "/" << N << " failed — cuDSS with 5x dim=10 SOC, square A";
}

/**
 * Case 3: Single SOC dim=30, square A (n=m=30).
 * Even smaller problem where expansion rows are a larger fraction of total KKT.
 */
TEST_F(SOCLargeDimStressTest, DISABLED_SingleDim30_Square) {
    constexpr int N = 20;
    int f = runSeedSweep({30}, 0, 30, 300, N, "dim30_s");
    EXPECT_EQ(f, 0) << f << "/" << N << " failed — cuDSS with dim=30 SOC, square A";
}

/**
 * Case 4: Single SOC dim=20, square A (n=m=20).
 * Even smaller problem where expansion rows are a huge fraction of KKT.
 */
TEST_F(SOCLargeDimStressTest, DISABLED_SingleDim20_Square) {
    constexpr int N = 20;
    int f = runSeedSweep({20}, 0, 20, 400, N, "dim20_s");
    EXPECT_EQ(f, 0) << f << "/" << N << " failed — cuDSS with dim=20 SOC, square A";
}

/**
 * Case 5: Single SOC dim=40 + 5 nonneg, square A (n=m=45).
 * Large SOC cone combined with nonneg, near the n=50 sweet spot.
 */
TEST_F(SOCLargeDimStressTest, DISABLED_SingleDim40_Nonneg_Square) {
    constexpr int N = 20;
    int f = runSeedSweep({40}, 5, 45, 500, N, "dim40nn_s");
    EXPECT_EQ(f, 0) << f << "/" << N << " failed — cuDSS with dim=40 SOC + nonneg, square A";
}
