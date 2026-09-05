// test_direct_cone_infeasibility.cpp
//
// HSDE infeasibility / unboundedness detection on the CUDA path with
// direct-x (xcone) constraints. Mirrors packages/moreau-cpu tests
// direct_cone_infeasibility.rs. The bug being pinned is that the
// primal-infeasibility certificate `‖A^T z − Σ_J E_J^T z_x‖ → 0` was
// missing the direct-x term, so direct-x infeasible problems used to fall
// through to NumericalError or MaxIterations.
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

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

// `min 0 s.t. x = -1, x ≥ 0` is primal-infeasible.
// Direct-x form: zero-cone equality `x = -1` plus direct-x `x ∈ R+`.
TEST(XConeInfeasibilityTest, NonnegDirectXPrimalInfeasible) {
    constexpr int batchSize = 1;
    constexpr int n = 1, m = 1;

    std::vector<int64_t> P_ro = {0, 0};       // P all zero
    std::vector<int64_t> P_ci = {};
    std::vector<int64_t> A_ro = {0, 1};       // A = [[1]]
    std::vector<int64_t> A_ci = {0};

    Cones cones{};
    cones.numZeroCones = 1;                    // single equality row
    cones.dir_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {0}});

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), 0,
                          A_ro.data(), A_ci.data(), 1, cones, settings);

    std::vector<double> P_values = {};
    std::vector<double> A_values = {1.0};
    std::vector<double> q = {0.0};
    std::vector<double> b = {-1.0};

    double* d_P = cuda_upload(P_values);
    double* d_A = cuda_upload(A_values);
    double* d_q = cuda_upload(q);
    double* d_b = cuda_upload(b);

    solver.solveAll(d_P, d_A, d_q, d_b);

    SolverStatus st = get_status(solver);
    EXPECT_TRUE(st == SolverStatus::PrimalInfeasible ||
                st == SolverStatus::AlmostPrimalInfeasible)
        << "expected PrimalInfeasible (direct-x nonneg), got "
        << static_cast<int>(st);

    if (d_P) cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// `min -x s.t. x ≥ 0` is unbounded ⇒ DualInfeasible.
TEST(XConeInfeasibilityTest, NonnegDirectXDualInfeasible) {
    constexpr int batchSize = 1;
    constexpr int n = 1, m = 0;

    std::vector<int64_t> P_ro = {0, 0};
    std::vector<int64_t> P_ci = {};
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    cones.dir_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {0}});

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), 0,
                          A_ro.data(), A_ci.data(), 0, cones, settings);

    std::vector<double> q = {-1.0};

    double *d_P = nullptr, *d_A = nullptr, *d_b = nullptr;
    cudaMalloc(&d_P, sizeof(double));
    cudaMalloc(&d_A, sizeof(double));
    cudaMalloc(&d_b, sizeof(double));
    double* d_q = cuda_upload(q);

    solver.solveAll(d_P, d_A, d_q, d_b);

    SolverStatus st = get_status(solver);
    EXPECT_TRUE(st == SolverStatus::DualInfeasible ||
                st == SolverStatus::AlmostDualInfeasible)
        << "expected DualInfeasible (direct-x nonneg unbounded), got "
        << static_cast<int>(st);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Direct-x SOC primal infeasible: `(t, v) ∈ SOC_3` requires `t ≥ ‖v‖`,
// so forcing `t = -1` via a zero-cone equality is infeasible.
TEST(XConeInfeasibilityTest, SOCDirectXPrimalInfeasible) {
    constexpr int batchSize = 1;
    constexpr int n = 3, m = 1;

    std::vector<int64_t> P_ro = {0, 0, 0, 0};   // P all zero
    std::vector<int64_t> P_ci = {};
    // A = [[1, 0, 0]] (single equality on x[0])
    std::vector<int64_t> A_ro = {0, 1};
    std::vector<int64_t> A_ci = {0};

    Cones cones{};
    cones.numZeroCones = 1;
    cones.dir_cones.push_back(SupportedXConeT{XConeKind::SOC, {0, 1, 2}});

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), 0,
                          A_ro.data(), A_ci.data(), 1, cones, settings);

    std::vector<double> A_values = {1.0};
    std::vector<double> q = {0.0, 0.0, 0.0};
    std::vector<double> b = {-1.0};

    double *d_P = nullptr;
    cudaMalloc(&d_P, sizeof(double));
    double* d_A = cuda_upload(A_values);
    double* d_q = cuda_upload(q);
    double* d_b = cuda_upload(b);

    solver.solveAll(d_P, d_A, d_q, d_b);

    SolverStatus st = get_status(solver);
    EXPECT_TRUE(st == SolverStatus::PrimalInfeasible ||
                st == SolverStatus::AlmostPrimalInfeasible)
        << "expected PrimalInfeasible (direct-x SOC), got "
        << static_cast<int>(st);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Direct-x SOC dual infeasible: `min -t s.t. (t, v) ∈ SOC_3` is unbounded.
TEST(XConeInfeasibilityTest, SOCDirectXDualInfeasible) {
    constexpr int batchSize = 1;
    constexpr int n = 3, m = 0;

    std::vector<int64_t> P_ro = {0, 0, 0, 0};
    std::vector<int64_t> P_ci = {};
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci = {};

    Cones cones{};
    cones.dir_cones.push_back(SupportedXConeT{XConeKind::SOC, {0, 1, 2}});

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), 0,
                          A_ro.data(), A_ci.data(), 0, cones, settings);

    std::vector<double> q = {-1.0, 0.0, 0.0};

    double *d_P = nullptr, *d_A = nullptr, *d_b = nullptr;
    cudaMalloc(&d_P, sizeof(double));
    cudaMalloc(&d_A, sizeof(double));
    cudaMalloc(&d_b, sizeof(double));
    double* d_q = cuda_upload(q);

    solver.solveAll(d_P, d_A, d_q, d_b);

    SolverStatus st = get_status(solver);
    EXPECT_TRUE(st == SolverStatus::DualInfeasible ||
                st == SolverStatus::AlmostDualInfeasible)
        << "expected DualInfeasible (direct-x SOC unbounded), got "
        << static_cast<int>(st);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Direct-x PSD primal infeasible: `svec(X)[0] = -1` forces the (1,1)
// diagonal of a 2×2 PSD matrix to be negative.
TEST(XConeInfeasibilityTest, PSDDirectXPrimalInfeasible) {
    constexpr int batchSize = 1;
    constexpr int n = 3, m = 1;

    std::vector<int64_t> P_ro = {0, 0, 0, 0};
    std::vector<int64_t> P_ci = {};
    // A = [[1, 0, 0]] (forces svec(X)[0] = -1)
    std::vector<int64_t> A_ro = {0, 1};
    std::vector<int64_t> A_ci = {0};

    Cones cones{};
    cones.numZeroCones = 1;
    cones.dir_cones.push_back(
        SupportedXConeT{XConeKind::PSD, {0, 1, 2}, /*psd_k=*/2});

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), 0,
                          A_ro.data(), A_ci.data(), 1, cones, settings);

    std::vector<double> A_values = {1.0};
    std::vector<double> q = {0.0, 0.0, 0.0};
    std::vector<double> b = {-1.0};

    double *d_P = nullptr;
    cudaMalloc(&d_P, sizeof(double));
    double* d_A = cuda_upload(A_values);
    double* d_q = cuda_upload(q);
    double* d_b = cuda_upload(b);

    solver.solveAll(d_P, d_A, d_q, d_b);

    SolverStatus st = get_status(solver);
    EXPECT_TRUE(st == SolverStatus::PrimalInfeasible ||
                st == SolverStatus::AlmostPrimalInfeasible)
        << "expected PrimalInfeasible (direct-x PSD), got "
        << static_cast<int>(st);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Mixed: direct-x nonneg + slack equality `a^T x = -2 (a > 0)` is infeasible.
TEST(XConeInfeasibilityTest, NonnegDirectXMixedPrimalInfeasible) {
    constexpr int batchSize = 1;
    constexpr int n = 2, m = 1;

    std::vector<int64_t> P_ro = {0, 0, 0};     // P all zero
    std::vector<int64_t> P_ci = {};
    // A = [[1, 1]]
    std::vector<int64_t> A_ro = {0, 2};
    std::vector<int64_t> A_ci = {0, 1};

    Cones cones{};
    cones.numZeroCones = 1;
    cones.dir_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {0, 1}});

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(n, m, batchSize, P_ro.data(), P_ci.data(), 0,
                          A_ro.data(), A_ci.data(), 2, cones, settings);

    std::vector<double> A_values = {1.0, 1.0};
    std::vector<double> q = {1.0, 1.0};
    std::vector<double> b = {-2.0};

    double *d_P = nullptr;
    cudaMalloc(&d_P, sizeof(double));
    double* d_A = cuda_upload(A_values);
    double* d_q = cuda_upload(q);
    double* d_b = cuda_upload(b);

    solver.solveAll(d_P, d_A, d_q, d_b);

    SolverStatus st = get_status(solver);
    EXPECT_TRUE(st == SolverStatus::PrimalInfeasible ||
                st == SolverStatus::AlmostPrimalInfeasible)
        << "expected PrimalInfeasible (mixed direct-x nonneg + slack eq), got "
        << static_cast<int>(st);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
