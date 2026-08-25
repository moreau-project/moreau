// test_xcone_exp_direct_x.cpp
//
// End-to-end direct-x ExpCone solves on CUDA, cross-checked against the
// CPU reference in packages/moreau-cpu/tests/xcone_exp_equivalence.rs.
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
// min 0.5 ||x - target||^2  s.t.  x ∈ ExpCone
// with target = (0, 1, 5). P = I, q = (0, -1, -5).
// Mirrors CPU test_exp_direct_x_solves_simple_problem.
// Expected: Solved or AlmostSolved; x lies in the exponential cone.
// -----------------------------------------------------------------------
TEST(XConeExpDirectXTest, SimpleProblemConverges) {
    constexpr int batchSize = 1;
    constexpr int n = 3, m = 0;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    cones.x_cones.push_back(SupportedXConeT{XConeKind::Exp, {0, 1, 2}});

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), 3,
                          A_ro.data(), A_ci.data(), 0, cones, settings);

    std::vector<double> P_values = {1.0, 1.0, 1.0};
    std::vector<double> q = {0.0, -1.0, -5.0};

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

    // Check x lies in the exponential cone: x1 > 0, x2 > 0,
    // x1 * log(x2/x1) - x0 >= 0.
    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    EXPECT_GT(x[1], 0.0) << "x[1] must be positive in ExpCone";
    EXPECT_GT(x[2], 0.0) << "x[2] must be positive in ExpCone";
    if (x[1] > 1e-8 && x[2] > 1e-8) {
        double u = x[1] * std::log(x[2] / x[1]) - x[0];
        EXPECT_GT(u, -1e-4) << "ExpCone primal: x1*log(x2/x1)-x0 must be >= 0";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// -----------------------------------------------------------------------
// StackedConesConverge
//
// K=10 stacked ExpCone direct-x cones on (3k, 3k+1, 3k+2) for k=0..9.
// n=30, P=I, q[3k]=-0, q[3k+1]=-1, q[3k+2]=-5.
// Mirrors CPU test_exp_direct_x_stacked_solves (with smaller K).
// Expected: Solved or AlmostSolved.
// -----------------------------------------------------------------------
TEST(XConeExpDirectXTest, StackedConesConverge) {
    constexpr int K = 10;
    constexpr int n = 3 * K, m = 0;
    constexpr int batchSize = 1;

    // P = I (diagonal CSR: offsets 0..n, cols 0..n-1, values all 1.0)
    std::vector<int64_t> P_ro(n + 1);
    for (int i = 0; i <= n; ++i) P_ro[i] = i;
    std::vector<int64_t> P_ci(n);
    for (int i = 0; i < n; ++i) P_ci[i] = i;
    std::vector<double> P_values(n, 1.0);

    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    // q[3k] = 0, q[3k+1] = -1, q[3k+2] = -5
    std::vector<double> q(n);
    for (int k = 0; k < K; ++k) {
        q[3*k + 0] = 0.0;
        q[3*k + 1] = -1.0;
        q[3*k + 2] = -5.0;
    }

    Cones cones{};
    for (int k = 0; k < K; ++k) {
        cones.x_cones.push_back(SupportedXConeT{
            XConeKind::Exp,
            {3*k, 3*k+1, 3*k+2}
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
        << "K=" << K << " stacked ExpCone direct-x: expected Solved/AlmostSolved, got "
        << static_cast<int>(st);

    // Spot-check that all cones are primal-feasible.
    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    for (int k = 0; k < K; ++k) {
        double x0 = x[3*k+0], x1 = x[3*k+1], x2 = x[3*k+2];
        EXPECT_GT(x1, 0.0) << "cone " << k << ": x[1] must be positive";
        EXPECT_GT(x2, 0.0) << "cone " << k << ": x[2] must be positive";
        if (x1 > 1e-8 && x2 > 1e-8) {
            double u = x1 * std::log(x2 / x1) - x0;
            EXPECT_GT(u, -1e-3) << "cone " << k << ": ExpCone primal violated";
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
TEST(XConeExpDirectXTest, BatchedSimpleProblem) {
    constexpr int batchSize = 4;
    constexpr int n = 3, m = 0;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    cones.x_cones.push_back(SupportedXConeT{XConeKind::Exp, {0, 1, 2}});

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
        q[b*n + 0] = 0.0;
        q[b*n + 1] = -1.0;
        q[b*n + 2] = -5.0;
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
