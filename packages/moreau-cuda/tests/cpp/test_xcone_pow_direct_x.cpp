// test_xcone_pow_direct_x.cpp
//
// End-to-end direct-x PowerCone solves on CUDA, cross-checked against the
// CPU reference in packages/moreau-cpu/tests/xcone_pow_equivalence.rs.
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <limits>

#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/solver.hpp"
#include "moreau/solver/status.hpp"

using namespace moreau;

namespace {
template <typename T>
T* cuda_upload(const std::vector<T>& host) {
    T* d = nullptr;
    if (host.empty()) return nullptr;
    cudaMalloc(&d, sizeof(T) * host.size());
    cudaMemcpy(d, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice);
    return d;
}

SolverStatus get_status(CompiledSolver& solver, int batch = 0) {
    int32_t s = 0;
    cudaMemcpy(&s, solver.info.status_device + batch, sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return static_cast<SolverStatus>(s);
}
}  // namespace

// -----------------------------------------------------------------------
// SimpleProblemConverges
//
// min 0.5 ||x - target||^2  s.t.  x ∈ PowerCone(α=0.4)
// with target = (2, 3, 1) (inside the cone: 2^0.4 * 3^0.6 ≈ 2.55 > 1).
// P = I, q = (-2, -3, -1).
// Mirrors CPU test_pow_direct_x_solves_simple_problem.
// Expected: Solved or AlmostSolved; x[0], x[1] > 0.
// -----------------------------------------------------------------------
TEST(XConePowDirectXTest, SimpleProblemConverges) {
    constexpr int batchSize = 1;
    constexpr int n = 3, m = 0;
    constexpr double alpha = 0.4;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    cones.dir_cones.push_back(SupportedXConeT{XConeKind::Power, {0, 1, 2}, 0, alpha});

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), 3,
                          A_ro.data(), A_ci.data(), 0, cones, settings);

    std::vector<double> P_values = {1.0, 1.0, 1.0};
    std::vector<double> q = {-2.0, -3.0, -1.0};

    double* d_P = cuda_upload(P_values);
    double* d_q = cuda_upload(q);
    double *d_A = nullptr, *d_b = nullptr;
    cudaMalloc(&d_A, sizeof(double));
    cudaMalloc(&d_b, sizeof(double));

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    SolverStatus st = get_status(solver, 0);
    EXPECT_TRUE(st == SolverStatus::Solved || st == SolverStatus::AlmostSolved)
        << "Expected Solved or AlmostSolved, got " << static_cast<int>(st);

    // Check x satisfies power cone constraints: x[0], x[1] > 0 and
    // x[0]^α * x[1]^(1-α) >= |x[2]|.
    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    EXPECT_GT(x[0], -1e-6) << "x[0] must be non-negative in PowerCone";
    EXPECT_GT(x[1], -1e-6) << "x[1] must be non-negative in PowerCone";
    if (x[0] > 1e-8 && x[1] > 1e-8) {
        double lhs = std::pow(x[0], alpha) * std::pow(x[1], 1.0 - alpha);
        EXPECT_GT(lhs + 1e-3, std::abs(x[2]))
            << "PowerCone primal: x[0]^α * x[1]^(1-α) = " << lhs
            << ", |x[2]| = " << std::abs(x[2]);
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// -----------------------------------------------------------------------
// StackedConesConverge
//
// K=10 stacked PowerCone direct-x cones on (3k, 3k+1, 3k+2) for k=0..9.
// n=30, P=I, q[3k]=-2, q[3k+1]=-3, q[3k+2]=-1, all α=0.4.
// Mirrors CPU test_pow_direct_x_stacked_solves (with smaller K).
// Expected: Solved or AlmostSolved.
// -----------------------------------------------------------------------
TEST(XConePowDirectXTest, StackedConesConverge) {
    constexpr int K = 10;
    constexpr int n = 3 * K, m = 0;
    constexpr int batchSize = 1;
    constexpr double alpha = 0.4;

    // P = I
    std::vector<int64_t> P_ro(n + 1);
    for (int i = 0; i <= n; ++i) P_ro[i] = i;
    std::vector<int64_t> P_ci(n);
    for (int i = 0; i < n; ++i) P_ci[i] = i;
    std::vector<double> P_values(n, 1.0);

    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    // q[3k]=-2, q[3k+1]=-3, q[3k+2]=-1
    std::vector<double> q(n);
    for (int k = 0; k < K; ++k) {
        q[3*k + 0] = -2.0;
        q[3*k + 1] = -3.0;
        q[3*k + 2] = -1.0;
    }

    Cones cones{};
    for (int k = 0; k < K; ++k) {
        cones.dir_cones.push_back(SupportedXConeT{
            XConeKind::Power,
            {3*k, 3*k+1, 3*k+2},
            0,      // psd_k (unused for Power)
            alpha   // power_alpha
        });
    }

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), n,
                          A_ro.data(), A_ci.data(), 0, cones, settings);

    double* d_P = cuda_upload(P_values);
    double* d_q = cuda_upload(q);
    double *d_A = nullptr, *d_b = nullptr;
    cudaMalloc(&d_A, sizeof(double));
    cudaMalloc(&d_b, sizeof(double));

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    SolverStatus st = get_status(solver, 0);
    EXPECT_TRUE(st == SolverStatus::Solved || st == SolverStatus::AlmostSolved)
        << "K=" << K << " stacked PowerCone direct-x: expected Solved/AlmostSolved, got "
        << static_cast<int>(st);

    // Spot-check that all cones are primal-feasible.
    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    for (int k = 0; k < K; ++k) {
        double x0 = x[3*k+0], x1 = x[3*k+1], x2 = x[3*k+2];
        EXPECT_GT(x0, -1e-6) << "cone " << k << ": x[0] must be non-negative";
        EXPECT_GT(x1, -1e-6) << "cone " << k << ": x[1] must be non-negative";
        if (x0 > 1e-8 && x1 > 1e-8) {
            double lhs = std::pow(x0, alpha) * std::pow(x1, 1.0 - alpha);
            EXPECT_GT(lhs + 1e-3, std::abs(x2))
                << "cone " << k << ": PowerCone primal violated";
        }
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// -----------------------------------------------------------------------
// BatchedSimpleProblem
//
// Same as SimpleProblemConverges but with batchSize=4 to exercise
// batched paths.
// -----------------------------------------------------------------------
TEST(XConePowDirectXTest, BatchedSimpleProblem) {
    constexpr int batchSize = 4;
    constexpr int n = 3, m = 0;
    constexpr double alpha = 0.4;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    cones.dir_cones.push_back(SupportedXConeT{XConeKind::Power, {0, 1, 2}, 0, alpha});

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), 3,
                          A_ro.data(), A_ci.data(), 0, cones, settings);

    // Per-batch P=I; per-batch q identical.
    std::vector<double> P_values(3 * batchSize);
    for (int b = 0; b < batchSize; ++b) {
        P_values[b*3 + 0] = 1.0;
        P_values[b*3 + 1] = 1.0;
        P_values[b*3 + 2] = 1.0;
    }
    std::vector<double> q(n * batchSize);
    for (int b = 0; b < batchSize; ++b) {
        q[b*n + 0] = -2.0;
        q[b*n + 1] = -3.0;
        q[b*n + 2] = -1.0;
    }

    double* d_P = cuda_upload(P_values);
    double* d_q = cuda_upload(q);
    double *d_A = nullptr, *d_b = nullptr;
    cudaMalloc(&d_A, sizeof(double));
    cudaMalloc(&d_b, sizeof(double));

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    for (int b = 0; b < batchSize; ++b) {
        SolverStatus st = get_status(solver, b);
        EXPECT_TRUE(st == SolverStatus::Solved || st == SolverStatus::AlmostSolved)
            << "batch " << b << ": expected Solved/AlmostSolved, got "
            << static_cast<int>(st);
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
