// test_direct_cone_genpow.cpp
//
// End-to-end direct-x GenPowerCone solves on CUDA.
// GenPowerCone(α, dim2) = {(p,w) : ∏ p_i^αi ≥ ||w||₂, p_i ≥ 0}
// where α_i > 0, Σα_i = 1.
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <numeric>

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

bool is_in_genpow(const std::vector<double>& x, int64_t offset,
                  const std::vector<double>& alphas, int64_t dim2,
                  double tol = 1e-4) {
    int64_t dim1 = static_cast<int64_t>(alphas.size());
    for (int64_t i = 0; i < dim1; ++i)
        if (x[offset + i] < -tol) return false;
    double prod = 1.0;
    for (int64_t i = 0; i < dim1; ++i)
        prod *= std::pow(std::max(x[offset + i], 0.0), alphas[i]);
    double w_norm = 0.0;
    for (int64_t j = 0; j < dim2; ++j)
        w_norm += x[offset + dim1 + j] * x[offset + dim1 + j];
    w_norm = std::sqrt(w_norm);
    return prod + tol >= w_norm;
}
}  // namespace

// -----------------------------------------------------------------------
// SimpleProblemConverges
//
// min 0.5 ||x - target||^2  s.t.  x ∈ GenPowerCone(α=[0.4, 0.6], dim2=1)
// with target = (2, 3, 1) (inside: 2^0.4 * 3^0.6 ≈ 2.55 > 1).
// P = I (3x3), q = (-2, -3, -1), n=3, m=0.
// Expected: Solved or AlmostSolved; x lies in the cone.
// -----------------------------------------------------------------------
TEST(XConeGenPowDirectXTest, SimpleProblemConverges) {
    constexpr int batchSize = 1;
    constexpr int n = 3, m = 0;
    const std::vector<double> alphas = {0.4, 0.6};
    constexpr int64_t dim2 = 1;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    SupportedXConeT xc;
    xc.kind = XConeKind::GenPower;
    xc.indices = {0, 1, 2};
    xc.gen_power_alphas = alphas;
    xc.gen_power_dim2 = dim2;
    cones.dir_cones.push_back(xc);

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

    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    EXPECT_TRUE(is_in_genpow(x, 0, alphas, dim2))
        << "Solution does not lie in GenPowerCone";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// -----------------------------------------------------------------------
// HigherDimConverges
//
// Tests a 5D GenPowerCone (dim>4, sparse path):
// GenPowerCone(α=[0.2, 0.2, 0.2, 0.4], dim2=1)
// min 0.5 ||x - target||^2, target = (1, 1, 1, 2, 0.5).
// P = I (5x5), q = (-1, -1, -1, -2, -0.5), n=5, m=0.
// Expected: Solved or AlmostSolved; x lies in the cone.
// -----------------------------------------------------------------------
TEST(XConeGenPowDirectXTest, HigherDimConverges) {
    constexpr int batchSize = 1;
    constexpr int n = 5, m = 0;
    const std::vector<double> alphas = {0.2, 0.2, 0.2, 0.4};
    constexpr int64_t dim2 = 1;

    std::vector<int64_t> P_ro(n + 1);
    for (int i = 0; i <= n; ++i) P_ro[i] = i;
    std::vector<int64_t> P_ci(n);
    for (int i = 0; i < n; ++i) P_ci[i] = i;

    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    SupportedXConeT xc;
    xc.kind = XConeKind::GenPower;
    xc.indices = {0, 1, 2, 3, 4};
    xc.gen_power_alphas = alphas;
    xc.gen_power_dim2 = dim2;
    cones.dir_cones.push_back(xc);

    Settings settings;
    settings.maxIter = 300;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), n,
                          A_ro.data(), A_ci.data(), 0, cones, settings);

    std::vector<double> P_values(n, 1.0);
    std::vector<double> q = {-1.0, -1.0, -1.0, -2.0, -0.5};

    double* d_P = cuda_upload(P_values);
    double* d_q = cuda_upload(q);
    double *d_A = nullptr, *d_b = nullptr;
    cudaMalloc(&d_A, sizeof(double));
    cudaMalloc(&d_b, sizeof(double));

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    SolverStatus st = get_status(solver, 0);
    EXPECT_TRUE(st == SolverStatus::Solved || st == SolverStatus::AlmostSolved)
        << "Expected Solved or AlmostSolved (5D sparse path), got "
        << static_cast<int>(st);

    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    EXPECT_TRUE(is_in_genpow(x, 0, alphas, dim2))
        << "5D solution does not lie in GenPowerCone";

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// -----------------------------------------------------------------------
// StackedConesConverge
//
// K=5 stacked GenPowerCone(α=[0.4, 0.6], dim2=1) cones, n=15, m=0.
// P = I, q[3k]=−2, q[3k+1]=−3, q[3k+2]=−1.
// Expected: Solved or AlmostSolved; each cone's projection is feasible.
// -----------------------------------------------------------------------
TEST(XConeGenPowDirectXTest, StackedConesConverge) {
    constexpr int K = 5;
    constexpr int n = 3 * K, m = 0;
    constexpr int batchSize = 1;
    const std::vector<double> alphas = {0.4, 0.6};
    constexpr int64_t dim2 = 1;

    std::vector<int64_t> P_ro(n + 1);
    for (int i = 0; i <= n; ++i) P_ro[i] = i;
    std::vector<int64_t> P_ci(n);
    for (int i = 0; i < n; ++i) P_ci[i] = i;

    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    std::vector<double> q(n);
    for (int k = 0; k < K; ++k) {
        q[3*k + 0] = -2.0;
        q[3*k + 1] = -3.0;
        q[3*k + 2] = -1.0;
    }

    Cones cones{};
    for (int k = 0; k < K; ++k) {
        SupportedXConeT xc;
        xc.kind = XConeKind::GenPower;
        xc.indices = {3*k, 3*k+1, 3*k+2};
        xc.gen_power_alphas = alphas;
        xc.gen_power_dim2 = dim2;
        cones.dir_cones.push_back(xc);
    }

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), n,
                          A_ro.data(), A_ci.data(), 0, cones, settings);

    std::vector<double> P_values(n, 1.0);

    double* d_P = cuda_upload(P_values);
    double* d_q = cuda_upload(q);
    double *d_A = nullptr, *d_b = nullptr;
    cudaMalloc(&d_A, sizeof(double));
    cudaMalloc(&d_b, sizeof(double));

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    SolverStatus st = get_status(solver, 0);
    EXPECT_TRUE(st == SolverStatus::Solved || st == SolverStatus::AlmostSolved)
        << "K=" << K << " stacked GenPow cones: expected Solved/AlmostSolved, got "
        << static_cast<int>(st);

    std::vector<double> x(n);
    solver.solution.x.gpuToCpu(x.data(), 0);
    cudaDeviceSynchronize();

    for (int k = 0; k < K; ++k) {
        EXPECT_TRUE(is_in_genpow(x, 3*k, alphas, dim2))
            << "cone " << k << " primal infeasible";
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Regression: zero cone + direct-x GenPower must NOT auto-select Riccati.
TEST(XConeGenPowDirectXTest, ZeroPlusDirectXGenPowAutoSelectsCuDSS) {
    constexpr int batchSize = 1;
    constexpr int dim1 = 8, dim2 = 4;
    constexpr int n = dim1 + dim2;
    constexpr int m = dim1;
    const std::vector<double> alphas(dim1, 1.0 / dim1);

    std::vector<int64_t> P_ro(n + 1);
    for (int i = 0; i <= n; ++i) P_ro[i] = i;
    std::vector<int64_t> P_ci(n);
    for (int i = 0; i < n; ++i) P_ci[i] = i;

    std::vector<int64_t> A_ro(m + 1);
    for (int i = 0; i <= m; ++i) A_ro[i] = i;
    std::vector<int64_t> A_ci(m);
    for (int i = 0; i < m; ++i) A_ci[i] = i;
    std::vector<double> A_val(m, 1.0);

    Cones cones{};
    cones.numZeroCones = m;
    SupportedXConeT xc;
    xc.kind = XConeKind::GenPower;
    xc.indices.resize(n);
    std::iota(xc.indices.begin(), xc.indices.end(), 0);
    xc.gen_power_alphas = alphas;
    xc.gen_power_dim2 = dim2;
    cones.dir_cones.push_back(xc);

    Settings settings;
    settings.maxIter = 50;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), n,
                          A_ro.data(), A_ci.data(), m, cones, settings);

    std::vector<double> P_values(n, 2.0);
    std::vector<double> q(n, 0.0);
    for (int i = dim1; i < n; ++i) q[i] = -1.0;
    std::vector<double> b(m, 1.0);

    double* d_P = cuda_upload(P_values);
    double* d_A = cuda_upload(A_val);
    double* d_q = cuda_upload(q);
    double* d_b = cuda_upload(b);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    SolverStatus st = get_status(solver, 0);
    EXPECT_TRUE(st == SolverStatus::Solved || st == SolverStatus::AlmostSolved)
        << "auto must NOT pick Riccati for direct-x genpow; got " << static_cast<int>(st);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
