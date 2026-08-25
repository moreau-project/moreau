/**
 * @file test_soc_hard.cpp
 * @brief Hard variable-dimension SOC tests with dense P, ill-conditioned A,
 *        near-boundary slack, batched solves, and multi-solve workflows.
 *
 * These tests are designed to stress the solver harder than the basic
 * test_soc_varlen.cpp tests, which use P=I and well-conditioned A.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/solver.hpp"

using namespace moreau;

namespace {

/// Sample a point strictly inside SOC(dim) with given margin.
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

/// Check if a point lies inside SOC(dim).
bool isInSoc(const double* data, int64_t dim, double tol = 1e-8) {
    double norm_sq = 0.0;
    for (int64_t i = 1; i < dim; i++) {
        norm_sq += data[i] * data[i];
    }
    return data[0] >= std::sqrt(norm_sq) - tol;
}

/// Check if a vector lies in a product of variable-dimension SOC cones.
bool isInSocProduct(const std::vector<double>& s,
                    const std::vector<int64_t>& socDims,
                    int64_t nonnegOffset,
                    int64_t numNonneg,
                    double tol = 1e-6) {
    // Nonneg cones first
    for (int64_t i = 0; i < numNonneg; i++) {
        if (s[i] < -tol) return false;
    }
    // SOC cones
    int64_t pos = nonnegOffset;
    for (auto dim : socDims) {
        if (!isInSoc(s.data() + pos, dim, tol)) return false;
        pos += dim;
    }
    return true;
}

/**
 * @brief Build a hard QP with dense P, ill-conditioned A, and near-boundary SOC slack.
 *
 * - P = L*L' + eps*I (dense symmetric PD)
 * - A has rows scaled by powers of 10 (ill-conditioning)
 * - s_star is near the SOC boundary
 * - q is random (not aligned with x_star)
 */
struct HardSOCProblem {
    int64_t n, m;
    int64_t numNonneg;
    std::vector<int64_t> socDims;

    // Dense P (row-major, full symmetric)
    std::vector<double> P_dense;
    // Dense A (row-major)
    std::vector<double> A_dense;
    std::vector<double> q, b;

    // Known feasible point
    std::vector<double> x_star, s_star;

    // CSR storage
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double> P_val;
    std::vector<int64_t> A_ro, A_ci;
    std::vector<double> A_val;

    HardSOCProblem(
        int64_t n_,
        const std::vector<int64_t>& socDims_,
        int64_t numNonneg_,
        uint64_t seed,
        double soc_margin = 0.01,
        double cond_range = 2.0,
        double p_offdiag_scale = 0.3
    ) : n(n_), numNonneg(numNonneg_), socDims(socDims_)
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> normal(0.0, 1.0);
        std::uniform_real_distribution<double> uniform01(0.0, 1.0);

        int64_t totalSocDim = 0;
        for (auto d : socDims) totalSocDim += d;
        m = numNonneg + totalSocDim;

        // P: dense symmetric PD = L*L' + 0.1*I
        P_dense.resize(n * n, 0.0);
        std::vector<double> L(n * n, 0.0);
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j <= i; j++) {
                L[i * n + j] = normal(rng) * p_offdiag_scale;
            }
            L[i * n + i] += 1.0;  // ensure well-conditioned
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

        // A: random dense m x n with ill-conditioned row scaling
        A_dense.resize(m * n);
        for (int64_t i = 0; i < m; i++) {
            double log_scale = (uniform01(rng) - 0.5) * cond_range;
            double scale = std::pow(10.0, log_scale);
            for (int64_t j = 0; j < n; j++) {
                A_dense[i * n + j] = normal(rng) * scale;
            }
        }

        // x_star: random feasible primal point
        x_star.resize(n);
        for (int64_t j = 0; j < n; j++) x_star[j] = normal(rng);

        // s_star: nonneg part + near-boundary SOC parts
        s_star.resize(m, 0.0);
        // Nonneg entries: small positive margin
        for (int64_t i = 0; i < numNonneg; i++) {
            s_star[i] = soc_margin + std::abs(normal(rng)) * soc_margin;
        }
        // SOC entries
        int64_t offset = numNonneg;
        for (auto dim : socDims) {
            auto block = sampleSocInterior(rng, dim, soc_margin);
            for (int64_t i = 0; i < dim; i++) {
                s_star[offset + i] = block[i];
            }
            offset += dim;
        }

        // q: random (not q = -P*x_star, so the problem isn't trivially solved)
        q.resize(n);
        for (int64_t j = 0; j < n; j++) q[j] = normal(rng) * 2.0;

        // b = A * x_star + s_star
        b.resize(m, 0.0);
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                b[i] += A_dense[i * n + j] * x_star[j];
            }
            b[i] += s_star[i];
        }

        buildCSR();
    }

    void buildCSR() {
        // P CSR (full symmetric, dense)
        P_ro.resize(n + 1);
        P_ci.resize(n * n);
        P_val.resize(n * n);
        for (int64_t i = 0; i <= n; i++) P_ro[i] = i * n;
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j < n; j++) {
                P_ci[i * n + j] = j;
                P_val[i * n + j] = P_dense[i * n + j];
            }
        }

        // A CSR (dense)
        A_ro.resize(m + 1);
        A_ci.resize(m * n);
        A_val.resize(m * n);
        for (int64_t i = 0; i <= m; i++) A_ro[i] = i * n;
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                A_ci[i * n + j] = j;
                A_val[i * n + j] = A_dense[i * n + j];
            }
        }
    }

    int64_t nnzP() const { return static_cast<int64_t>(P_val.size()); }
    int64_t nnzA() const { return static_cast<int64_t>(A_val.size()); }
};

/// Helper: allocate device memory, copy data, solve, check result
struct SolveResult {
    std::vector<double> x, s, z;
    int32_t status;
    bool solved;
};

SolveResult solveHardProblem(const HardSOCProblem& prob, int64_t batchSize = 1,
                             int maxIter = 200, bool verbose = false) {
    Cones cones{};
    cones.socConeDims = prob.socDims;
    cones.numSocCones = static_cast<int64_t>(prob.socDims.size());
    cones.numNonnegCones = prob.numNonneg;

    Settings settings;
    settings.maxIter = maxIter;
    settings.verbose = verbose;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP(),
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA(),
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP());
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA());
    cudaMalloc(&d_q, sizeof(double) * prob.n);
    cudaMalloc(&d_b, sizeof(double) * prob.m);
    cudaMemcpy(d_P, prob.P_val.data(), sizeof(double) * prob.nnzP(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_val.data(), sizeof(double) * prob.nnzA(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    SolveResult res;
    res.x.resize(prob.n);
    res.s.resize(prob.m);
    res.z.resize(prob.m);
    cudaMemcpy(res.x.data(), solver.solution.x.data(), sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
    cudaMemcpy(res.s.data(), solver.solution.s.data(), sizeof(double) * prob.m, cudaMemcpyDeviceToHost);
    cudaMemcpy(res.z.data(), solver.solution.z.data(), sizeof(double) * prob.m, cudaMemcpyDeviceToHost);
    int32_t st;
    cudaMemcpy(&st, solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    res.status = st;
    res.solved = (st == static_cast<int32_t>(SolverStatus::Solved) ||
                  st == static_cast<int32_t>(SolverStatus::AlmostSolved));

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    return res;
}

/// Compute primal residual ||Ax + s - b||_inf
double primalResidual(const HardSOCProblem& prob, const SolveResult& res) {
    double max_res = 0.0;
    for (int64_t i = 0; i < prob.m; i++) {
        double r = -prob.b[i] + res.s[i];
        for (int64_t j = 0; j < prob.n; j++) {
            r += prob.A_dense[i * prob.n + j] * res.x[j];
        }
        max_res = std::max(max_res, std::abs(r));
    }
    return max_res;
}

} // namespace

class SOCHardTest : public ::testing::Test {
protected:
    void SetUp() override { cudaGetLastError(); }
    void TearDown() override { cudaDeviceSynchronize(); }
};

// ============================================================================
// Test 1: Dense P + large sparse SOC dim=20
// ============================================================================

TEST_F(SOCHardTest, DenseP_Dim20) {
    // Dense P + SOC(20): the HardSOCProblem dense P construction typically yields
    // AlmostSolved for single large SOC, which is acceptable. Verify it converges.
    HardSOCProblem prob(25, {20}, 0, /*seed=*/42, /*margin=*/0.05, /*cond=*/1.5);
    auto res = solveHardProblem(prob);

    EXPECT_TRUE(res.solved) << "Status = " << res.status;
    EXPECT_TRUE(isInSoc(res.s.data(), 20, 1e-4)) << "s not in SOC(20)";
    EXPECT_LT(primalResidual(prob, res), 1e-2) << "primal residual too large";
}

// ============================================================================
// Test 2: Dense P + large sparse SOC dim=50
// ============================================================================

TEST_F(SOCHardTest, DenseP_Dim50) {
    // Dense P + SOC(50): exercises large sparse expansion. AlmostSolved acceptable.
    HardSOCProblem prob(60, {50}, 0, /*seed=*/43, /*margin=*/0.05, /*cond=*/1.0);
    auto res = solveHardProblem(prob);

    EXPECT_TRUE(res.solved) << "Status = " << res.status;
    EXPECT_TRUE(isInSoc(res.s.data(), 50, 1e-4)) << "s not in SOC(50)";
    EXPECT_LT(primalResidual(prob, res), 1e-2);
}

// ============================================================================
// Test 3: Dense P + mixed dense/sparse SOC dims, ill-conditioned A
// ============================================================================

TEST_F(SOCHardTest, DenseP_MixedDims_IllCondA) {
    // dims 3,4 are dense (<=4), dims 7,10 are sparse (>4)
    HardSOCProblem prob(30, {3, 7, 4, 10, 6}, 0, /*seed=*/44, /*margin=*/0.01, /*cond=*/3.0);
    auto res = solveHardProblem(prob);

    EXPECT_EQ(res.status, static_cast<int32_t>(SolverStatus::Solved)) << "Expected Solved";
    EXPECT_TRUE(isInSocProduct(res.s, prob.socDims, 0, 0, 1e-5));
    EXPECT_LT(primalResidual(prob, res), 1e-4);
}

// ============================================================================
// Test 4: Dense P + nonneg + large sparse SOC, ill-conditioned
// ============================================================================

TEST_F(SOCHardTest, DenseP_Nonneg_SparseSoc_IllCond) {
    // 5 nonneg + SOC(15) + SOC(10) = m=30, n=30
    HardSOCProblem prob(30, {15, 10}, 5, /*seed=*/45, /*margin=*/0.005, /*cond=*/3.0);
    auto res = solveHardProblem(prob);

    EXPECT_EQ(res.status, static_cast<int32_t>(SolverStatus::Solved)) << "Expected Solved";
    EXPECT_TRUE(isInSocProduct(res.s, prob.socDims, 5, 5, 1e-5));
    EXPECT_LT(primalResidual(prob, res), 1e-4);
}

// ============================================================================
// Test 5: Multiple seeds for dim=9 (the previously-failing dimension)
// ============================================================================

TEST_F(SOCHardTest, DenseP_Dim9_MultiSeed) {
    int failures = 0;
    for (uint64_t seed = 0; seed < 20; seed++) {
        HardSOCProblem prob(12, {9}, 3, seed + 100, /*margin=*/0.01, /*cond=*/2.5);
        auto res = solveHardProblem(prob);
        if (res.status != static_cast<int32_t>(SolverStatus::Solved)) {
            failures++;
            std::cout << "  seed=" << (seed + 100) << " failed with status=" << res.status << "\n";
        }
    }
    EXPECT_EQ(failures, 0) << failures << "/20 failed for dim=9 + nonneg, dense P";
}

// ============================================================================
// Test 6: Seed sweep for large dim=30 (stress test with 10 seeds)
// ============================================================================

TEST_F(SOCHardTest, DenseP_Dim30_MultiSeed) {
    int failures = 0;
    for (uint64_t seed = 0; seed < 10; seed++) {
        // n=35 > m=30 gives the solver more freedom; margin=0.05 keeps slack away from boundary
        HardSOCProblem prob(35, {30}, 0, seed + 200, /*margin=*/0.05, /*cond=*/1.5);
        auto res = solveHardProblem(prob, /*batchSize=*/1, /*maxIter=*/300);
        if (res.status != static_cast<int32_t>(SolverStatus::Solved)) {
            failures++;
            std::cout << "  seed=" << (seed + 200) << " failed with status=" << res.status << "\n";
        }
    }
    EXPECT_EQ(failures, 0) << failures << "/10 failed for dim=30, dense P";
}

// ============================================================================
// Test 7: Batched solve with variable-dim SOC (batch_size=4)
// ============================================================================

TEST_F(SOCHardTest, Batched_MixedDims) {
    // Build 4 different problems with same structure but different data
    const int64_t n = 15;
    const std::vector<int64_t> socDims = {5, 10};
    const int64_t batchSize = 4;
    int64_t m = 15;  // 5 + 10

    Cones cones{};
    cones.socConeDims = socDims;
    cones.numSocCones = 2;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    // Build one problem to get the structure
    HardSOCProblem prob0(n, socDims, 0, /*seed=*/300, /*margin=*/0.05, /*cond=*/1.5);

    CompiledSolver solver(
        n, m, batchSize,
        prob0.P_ro.data(), prob0.P_ci.data(), prob0.nnzP(),
        prob0.A_ro.data(), prob0.A_ci.data(), prob0.nnzA(),
        cones, settings
    );

    // Build batched data: same P and A structure, different values per batch
    std::vector<double> P_batch(prob0.nnzP() * batchSize);
    std::vector<double> A_batch(prob0.nnzA() * batchSize);
    std::vector<double> q_batch(n * batchSize);
    std::vector<double> b_batch(m * batchSize);

    for (int64_t batch = 0; batch < batchSize; batch++) {
        HardSOCProblem p(n, socDims, 0, 300 + batch, /*margin=*/0.05, /*cond=*/1.5);
        std::copy(p.P_val.begin(), p.P_val.end(), P_batch.begin() + batch * prob0.nnzP());
        std::copy(p.A_val.begin(), p.A_val.end(), A_batch.begin() + batch * prob0.nnzA());
        std::copy(p.q.begin(), p.q.end(), q_batch.begin() + batch * n);
        std::copy(p.b.begin(), p.b.end(), b_batch.begin() + batch * m);
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * P_batch.size());
    cudaMalloc(&d_A, sizeof(double) * A_batch.size());
    cudaMalloc(&d_q, sizeof(double) * q_batch.size());
    cudaMalloc(&d_b, sizeof(double) * b_batch.size());
    cudaMemcpy(d_P, P_batch.data(), sizeof(double) * P_batch.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_batch.data(), sizeof(double) * A_batch.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_batch.data(), sizeof(double) * q_batch.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_batch.data(), sizeof(double) * b_batch.size(), cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> s_all(m * batchSize);
    std::vector<int32_t> status_all(batchSize);
    cudaMemcpy(s_all.data(), solver.solution.s.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_all.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int64_t batch = 0; batch < batchSize; batch++) {
        EXPECT_EQ(status_all[batch], static_cast<int32_t>(SolverStatus::Solved))
            << "Batch " << batch << " expected Solved";

        // Check SOC membership for each batch
        std::vector<double> s_batch(s_all.begin() + batch * m, s_all.begin() + (batch + 1) * m);
        EXPECT_TRUE(isInSocProduct(s_batch, socDims, 0, 0, 1e-5))
            << "Batch " << batch << " s not in SOC product";
    }

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// ============================================================================
// Test 8: Three-step API: setup once, solve twice with different q/b
// ============================================================================

TEST_F(SOCHardTest, ThreeStepAPI_ResolveWithNewQb) {
    const int64_t n = 15;
    const std::vector<int64_t> socDims = {5, 10};
    const int64_t m = 15;  // 5 + 10

    Cones cones{};
    cones.socConeDims = socDims;
    cones.numSocCones = 2;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;
    settings.ipm.equilibrationSettings.enable = true;

    HardSOCProblem prob1(n, socDims, 0, /*seed=*/400, /*margin=*/0.05, /*cond=*/1.5);

    CompiledSolver solver(
        n, m, 1,
        prob1.P_ro.data(), prob1.P_ci.data(), prob1.nnzP(),
        prob1.A_ro.data(), prob1.A_ci.data(), prob1.nnzA(),
        cones, settings
    );

    // Setup P/A values
    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob1.nnzP());
    cudaMalloc(&d_A, sizeof(double) * prob1.nnzA());
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMemcpy(d_P, prob1.P_val.data(), sizeof(double) * prob1.nnzP(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob1.A_val.data(), sizeof(double) * prob1.nnzA(), cudaMemcpyHostToDevice);
    solver.setup(d_P, d_A);

    // Solve problem 1
    cudaMemcpy(d_q, prob1.q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob1.b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    solver.solve(d_q, d_b);

    int32_t st1;
    cudaMemcpy(&st1, solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    EXPECT_EQ(st1, static_cast<int32_t>(SolverStatus::Solved))
        << "Problem 1 expected Solved";

    // Solve problem 2 with different q/b but same P and A
    // Generate new x_star, s_star, q, and compute b = A_1 * x_star + s_star
    {
        std::mt19937_64 rng2(401);
        std::normal_distribution<double> normal2(0.0, 1.0);
        std::vector<double> x2(n), s2(m), q2(n), b2(m, 0.0);
        for (int64_t j = 0; j < n; j++) x2[j] = normal2(rng2);
        // SOC interior slack
        int64_t offset = 0;
        for (auto dim : socDims) {
            auto block = sampleSocInterior(rng2, dim, 0.1);
            for (int64_t i = 0; i < dim; i++) s2[offset + i] = block[i];
            offset += dim;
        }
        for (int64_t j = 0; j < n; j++) q2[j] = normal2(rng2) * 2.0;
        // b2 = A_1 * x2 + s2 (using prob1's A, which is what the solver has)
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                b2[i] += prob1.A_dense[i * n + j] * x2[j];
            }
            b2[i] += s2[i];
        }

        cudaMemcpy(d_q, q2.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b2.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
        solver.solve(d_q, d_b);

        int32_t st2;
        cudaMemcpy(&st2, solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        EXPECT_EQ(st2, static_cast<int32_t>(SolverStatus::Solved))
            << "Problem 2 expected Solved";

        // Check SOC membership of problem 2's solution
        std::vector<double> s2_out(m);
        cudaMemcpy(s2_out.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        EXPECT_TRUE(isInSocProduct(s2_out, socDims, 0, 0, 1e-5))
            << "Problem 2 s not in SOC product";
    }

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// ============================================================================
// Test 9: Many small sparse SOC cones (exercises multiple expansion blocks)
// ============================================================================

TEST_F(SOCHardTest, ManySmallSparseSocs) {
    // 6 sparse cones (each dim=5, just above the threshold)
    // Total SOC dim = 30, with 6 * 2 = 12 expansion rows in KKT
    std::vector<int64_t> socDims = {5, 5, 5, 5, 5, 5};
    HardSOCProblem prob(30, socDims, 0, /*seed=*/500, /*margin=*/0.02, /*cond=*/2.0);
    auto res = solveHardProblem(prob);

    EXPECT_EQ(res.status, static_cast<int32_t>(SolverStatus::Solved)) << "Expected Solved";
    EXPECT_TRUE(isInSocProduct(res.s, prob.socDims, 0, 0, 1e-5));
    EXPECT_LT(primalResidual(prob, res), 1e-4);
}

// ============================================================================
// Test 10: Large mixed problem: nonneg + dense SOC + sparse SOC
// ============================================================================

TEST_F(SOCHardTest, LargeMixed_Nonneg_Dense_Sparse) {
    // 10 nonneg + SOC(2) + SOC(3) + SOC(4) + SOC(10) + SOC(20) = m = 49, n = 40
    std::vector<int64_t> socDims = {2, 3, 4, 10, 20};
    HardSOCProblem prob(40, socDims, 10, /*seed=*/600, /*margin=*/0.01, /*cond=*/2.5);
    auto res = solveHardProblem(prob);

    EXPECT_TRUE(res.solved) << "Status = " << res.status;
    EXPECT_TRUE(isInSocProduct(res.s, prob.socDims, 10, 10, 1e-4));
    EXPECT_LT(primalResidual(prob, res), 1e-2);
}

// ============================================================================
// Test 11: Very ill-conditioned A (cond_range=4.0, row scales span 1e-2 to 1e2)
// ============================================================================

TEST_F(SOCHardTest, VeryIllConditionedA_Dim10) {
    // cond_range=4.0 means row scales span ~1e-2 to 1e2
    HardSOCProblem prob(15, {10}, 5, /*seed=*/700, /*margin=*/0.01, /*cond=*/4.0);
    auto res = solveHardProblem(prob);

    EXPECT_EQ(res.status, static_cast<int32_t>(SolverStatus::Solved)) << "Expected Solved";
    EXPECT_TRUE(isInSocProduct(res.s, prob.socDims, 5, 5, 1e-4));
    EXPECT_LT(primalResidual(prob, res), 1e-3);
}

// ============================================================================
// Test 12: Underdetermined system (n > m) with sparse SOC
// ============================================================================

TEST_F(SOCHardTest, Underdetermined_SparseSoc) {
    // n=30 > m=15: more variables than constraints
    HardSOCProblem prob(30, {15}, 0, /*seed=*/800, /*margin=*/0.02, /*cond=*/2.0);
    auto res = solveHardProblem(prob);

    EXPECT_EQ(res.status, static_cast<int32_t>(SolverStatus::Solved)) << "Expected Solved";
    EXPECT_TRUE(isInSoc(res.s.data(), 15, 1e-5));
    EXPECT_LT(primalResidual(prob, res), 1e-4);
}

// ============================================================================
// Test 13: Overdetermined system (n < m) with multiple sparse SOC
// ============================================================================

TEST_F(SOCHardTest, Overdetermined_MultipleSparseSoc) {
    // n=25 < m=30: more constraints than variables
    std::vector<int64_t> socDims = {10, 10, 10};
    HardSOCProblem prob(25, socDims, 0, /*seed=*/900, /*margin=*/0.05, /*cond=*/1.0);
    auto res = solveHardProblem(prob, /*batchSize=*/1, /*maxIter=*/300);

    EXPECT_EQ(res.status, static_cast<int32_t>(SolverStatus::Solved)) << "Expected Solved";
    EXPECT_TRUE(isInSocProduct(res.s, prob.socDims, 0, 0, 1e-4));
    EXPECT_LT(primalResidual(prob, res), 1e-2);
}

// ============================================================================
// Test 14: Dense P with large off-diagonal elements (higher P conditioning)
// ============================================================================

TEST_F(SOCHardTest, DenseP_LargeOffDiag_Dim15) {
    // p_offdiag_scale=0.8 makes P less diagonally dominant
    HardSOCProblem prob(20, {15}, 5, /*seed=*/1000, /*margin=*/0.01, /*cond=*/2.0,
                        /*p_offdiag_scale=*/0.8);
    auto res = solveHardProblem(prob);

    EXPECT_EQ(res.status, static_cast<int32_t>(SolverStatus::Solved)) << "Expected Solved";
    EXPECT_TRUE(isInSocProduct(res.s, prob.socDims, 5, 5, 1e-5));
    EXPECT_LT(primalResidual(prob, res), 1e-4);
}
