// test_direct_cone_asymm_backward.cpp
//
// CUDA backward parity for asymmetric direct-x cones (Exp, Power) vs
// the slack-form backward on the same problem. The asymmetric primal/
// dual swap matters only for the FORWARD scaling; the backward
// projection Jacobian is the same function as for the slack form, so
// the gradient outputs (dq, dP) must match.
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#include "moreau/cones/cones.hpp"
#include "moreau/diff/diff.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/solver.hpp"

using namespace moreau;

namespace {
template <typename T>
T* cuda_upload(const std::vector<T>& host) {
    T* d = nullptr;
    if (host.empty()) {
        cudaMalloc(&d, sizeof(T));
        return d;
    }
    cudaMalloc(&d, sizeof(T) * host.size());
    cudaMemcpy(d, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice);
    return d;
}

void cuda_set(BatchedVector& v, const std::vector<double>& host, int64_t batch_dim) {
    cudaMemcpy(v.data(), host.data(),
               sizeof(double) * batch_dim, cudaMemcpyHostToDevice);
}

std::vector<double> cuda_get(const BatchedVector& v) {
    std::vector<double> host(v.n() * v.batchSize());
    cudaMemcpy(host.data(), v.data(),
               sizeof(double) * host.size(), cudaMemcpyDeviceToHost);
    return host;
}
}  // namespace

// min 0.5 ||x - target||^2 with target = (0, 1, 5), x ∈ ExpCone (slack
// or direct-x). The cone is active at the optimum, so the backward
// projection Jacobian is in the boundary regime (Newton + 4×4 invert
// path), exercising the new compute_xcone_asymm_H kernel.
TEST(XConeAsymmBackwardCuda, ExpDirectXMatchesSlack) {
    constexpr int batchSize = 1;
    constexpr int n = 3;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<double>  P_val = {1.0, 1.0, 1.0};
    std::vector<double>  q     = {0.0, -1.0, -5.0};
    std::vector<double>  dx    = {1.0, 1.0, 1.0};

    std::vector<double> dq_slack, dP_slack;
    {
        constexpr int m = n;
        std::vector<int64_t> A_ro = {0, 1, 2, 3};
        std::vector<int64_t> A_ci = {0, 1, 2};
        std::vector<double>  A_val = {-1.0, -1.0, -1.0};
        std::vector<double>  b     = {0.0, 0.0, 0.0};

        Cones cones{};
        cones.numExpCones = 1;

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), P_val.size(),
                              A_ro.data(), A_ci.data(), A_val.size(),
                              cones, settings);

        double* d_P = cuda_upload(P_val);
        double* d_A = cuda_upload(A_val);
        double* d_q = cuda_upload(q);
        double* d_b = cuda_upload(b);
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(n, batchSize), dz_bar(m, batchSize), ds_bar(m, batchSize);
        cuda_set(dx_bar, dx, n);
        std::vector<double> zeros_m(m, 0.0);
        cuda_set(dz_bar, zeros_m, m);
        cuda_set(ds_bar, zeros_m, m);

        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        dq_slack = cuda_get(solver.diff_state()->dq);
        dP_slack = cuda_get(solver.diff_state()->dP_values);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    std::vector<double> dq_direct, dP_direct;
    {
        constexpr int m = 0;
        std::vector<int64_t> A_ro = {0};
        std::vector<int64_t> A_ci = {};

        Cones cones{};
        cones.dir_cones.push_back(SupportedXConeT{XConeKind::Exp, {0, 1, 2}});

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), P_val.size(),
                              A_ro.data(), A_ci.data(), 0,
                              cones, settings);

        double* d_P = cuda_upload(P_val);
        double* d_A = nullptr; cudaMalloc(&d_A, sizeof(double));
        double* d_q = cuda_upload(q);
        double* d_b = nullptr; cudaMalloc(&d_b, sizeof(double));
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(n, batchSize);
        BatchedVector dz_bar(1, batchSize);
        BatchedVector ds_bar(1, batchSize);
        cuda_set(dx_bar, dx, n);
        std::vector<double> zero1 = {0.0};
        cuda_set(dz_bar, zero1, 1);
        cuda_set(ds_bar, zero1, 1);

        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        dq_direct = cuda_get(solver.diff_state()->dq);
        dP_direct = cuda_get(solver.diff_state()->dP_values);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    ASSERT_EQ(dq_slack.size(), dq_direct.size());
    for (size_t i = 0; i < dq_slack.size(); ++i) {
        EXPECT_NEAR(dq_slack[i], dq_direct[i], 1e-3)
            << "dq[" << i << "] disagree";
    }

    ASSERT_EQ(dP_slack.size(), dP_direct.size());
    for (size_t k = 0; k < dP_slack.size(); ++k) {
        EXPECT_NEAR(dP_slack[k], dP_direct[k], 1e-3)
            << "dP[" << k << "] disagree";
    }
}

// min 0.5 ||x - target||^2 with target = (2, 2, 0.5), x ∈ PowerCone(α=0.4).
// PowerCone is {(x_0, x_1, x_2) : x_0^α x_1^(1−α) ≥ |x_2|, x_0,x_1 ≥ 0}.
// target lies strictly in the cone interior, so the projection Jacobian
// is in the easy interior regime (block = I, dual H = 0). This isolates
// the direct-x dispatch + storage from boundary-Newton numerical noise.
TEST(XConeAsymmBackwardCuda, PowerDirectXMatchesSlack) {
    constexpr int batchSize = 1;
    constexpr int n = 3;
    const double alpha = 0.4;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<double>  P_val = {1.0, 1.0, 1.0};
    // 2^0.4 * 2^0.6 = 2 ≥ 0.5 → target is interior; constraint inactive.
    std::vector<double>  q     = {-2.0, -2.0, -0.5};
    std::vector<double>  dx    = {1.0, 1.0, 1.0};

    std::vector<double> dq_slack, dP_slack;
    {
        constexpr int m = n;
        std::vector<int64_t> A_ro = {0, 1, 2, 3};
        std::vector<int64_t> A_ci = {0, 1, 2};
        std::vector<double>  A_val = {-1.0, -1.0, -1.0};
        std::vector<double>  b     = {0.0, 0.0, 0.0};

        Cones cones{};
        cones.numPowerCones = 1;
        cones.powerAlphas = {alpha};

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), P_val.size(),
                              A_ro.data(), A_ci.data(), A_val.size(),
                              cones, settings);

        double* d_P = cuda_upload(P_val);
        double* d_A = cuda_upload(A_val);
        double* d_q = cuda_upload(q);
        double* d_b = cuda_upload(b);
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(n, batchSize), dz_bar(m, batchSize), ds_bar(m, batchSize);
        cuda_set(dx_bar, dx, n);
        std::vector<double> zeros_m(m, 0.0);
        cuda_set(dz_bar, zeros_m, m);
        cuda_set(ds_bar, zeros_m, m);

        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        dq_slack = cuda_get(solver.diff_state()->dq);
        dP_slack = cuda_get(solver.diff_state()->dP_values);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    std::vector<double> dq_direct, dP_direct;
    {
        constexpr int m = 0;
        std::vector<int64_t> A_ro = {0};
        std::vector<int64_t> A_ci = {};

        Cones cones{};
        SupportedXConeT pow_xc{};
        pow_xc.kind = XConeKind::Power;
        pow_xc.indices = {0, 1, 2};
        pow_xc.power_alpha = alpha;
        cones.dir_cones.push_back(std::move(pow_xc));

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), P_val.size(),
                              A_ro.data(), A_ci.data(), 0,
                              cones, settings);

        double* d_P = cuda_upload(P_val);
        double* d_A = nullptr; cudaMalloc(&d_A, sizeof(double));
        double* d_q = cuda_upload(q);
        double* d_b = nullptr; cudaMalloc(&d_b, sizeof(double));
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(n, batchSize);
        BatchedVector dz_bar(1, batchSize);
        BatchedVector ds_bar(1, batchSize);
        cuda_set(dx_bar, dx, n);
        std::vector<double> zero1 = {0.0};
        cuda_set(dz_bar, zero1, 1);
        cuda_set(ds_bar, zero1, 1);

        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        dq_direct = cuda_get(solver.diff_state()->dq);
        dP_direct = cuda_get(solver.diff_state()->dP_values);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    ASSERT_EQ(dq_slack.size(), dq_direct.size());
    for (size_t i = 0; i < dq_slack.size(); ++i) {
        EXPECT_NEAR(dq_slack[i], dq_direct[i], 1e-3)
            << "dq[" << i << "] disagree";
    }

    ASSERT_EQ(dP_slack.size(), dP_direct.size());
    for (size_t k = 0; k < dP_slack.size(); ++k) {
        EXPECT_NEAR(dP_slack[k], dP_direct[k], 1e-3)
            << "dP[" << k << "] disagree";
    }
}

// High-dim SOC direct-x backward parity (dim=20) — exercises the
// rank-2 sparse expansion path on CUDA. Mirrors the CPU
// `test_backward_direct_x_soc_high_dim_matches_slack` regression.
TEST(XConeAsymmBackwardCuda, SOCDirectXMatchesSlackHighDim) {
    constexpr int batchSize = 1;
    constexpr int n = 20;

    std::vector<int64_t> P_ro(n + 1);
    for (int i = 0; i <= n; ++i) P_ro[i] = i;
    std::vector<int64_t> P_ci(n);
    for (int i = 0; i < n; ++i) P_ci[i] = i;
    std::vector<double> P_val(n, 1.0);
    std::vector<double> q(n, 0.0);
    q[0] = -2.0;  // pull x[0] toward -2 → SOC binding.
    std::vector<double> dx(n, 1.0);

    std::vector<double> dq_slack, dP_slack;
    {
        constexpr int m = n;
        std::vector<int64_t> A_ro(n + 1);
        for (int i = 0; i <= n; ++i) A_ro[i] = i;
        std::vector<int64_t> A_ci(n);
        for (int i = 0; i < n; ++i) A_ci[i] = i;
        std::vector<double> A_val(n, -1.0);
        std::vector<double> b(n, 0.0);

        Cones cones{};
        cones.numSocCones = 1;
        cones.socConeDims = {static_cast<int64_t>(n)};

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), P_val.size(),
                              A_ro.data(), A_ci.data(), A_val.size(),
                              cones, settings);
        double* d_P = cuda_upload(P_val);
        double* d_A = cuda_upload(A_val);
        double* d_q = cuda_upload(q);
        double* d_b = cuda_upload(b);
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(n, batchSize), dz_bar(m, batchSize), ds_bar(m, batchSize);
        cuda_set(dx_bar, dx, n);
        std::vector<double> zeros_m(m, 0.0);
        cuda_set(dz_bar, zeros_m, m);
        cuda_set(ds_bar, zeros_m, m);
        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();
        dq_slack = cuda_get(solver.diff_state()->dq);
        dP_slack = cuda_get(solver.diff_state()->dP_values);
        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    std::vector<double> dq_direct, dP_direct;
    {
        constexpr int m = 0;
        std::vector<int64_t> A_ro = {0};
        std::vector<int64_t> A_ci = {};

        Cones cones{};
        SupportedXConeT xc;
        xc.kind = XConeKind::SOC;
        xc.indices.assign(n, 0);
        for (int i = 0; i < n; ++i) xc.indices[i] = i;
        cones.dir_cones.push_back(std::move(xc));

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), P_val.size(),
                              A_ro.data(), A_ci.data(), 0,
                              cones, settings);
        double* d_P = cuda_upload(P_val);
        double* d_A = nullptr; cudaMalloc(&d_A, sizeof(double));
        double* d_q = cuda_upload(q);
        double* d_b = nullptr; cudaMalloc(&d_b, sizeof(double));
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(n, batchSize);
        BatchedVector dz_bar(1, batchSize);
        BatchedVector ds_bar(1, batchSize);
        cuda_set(dx_bar, dx, n);
        std::vector<double> zero1 = {0.0};
        cuda_set(dz_bar, zero1, 1);
        cuda_set(ds_bar, zero1, 1);
        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();
        dq_direct = cuda_get(solver.diff_state()->dq);
        dP_direct = cuda_get(solver.diff_state()->dP_values);
        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    ASSERT_EQ(dq_slack.size(), dq_direct.size());
    for (size_t i = 0; i < dq_slack.size(); ++i) {
        EXPECT_NEAR(dq_slack[i], dq_direct[i], 1e-3)
            << "dq[" << i << "] disagree";
    }
    ASSERT_EQ(dP_slack.size(), dP_direct.size());
    for (size_t k = 0; k < dP_slack.size(); ++k) {
        EXPECT_NEAR(dP_slack[k], dP_direct[k], 1e-3)
            << "dP[" << k << "] disagree";
    }
}

// GenPowerCone direct-x backward parity vs slack form, dim2=1 case
// (c3 term folded into the diagonal). The 3D GenPow with α=(0.5, 0.5)
// reduces to the standard SOC pattern but via the GenPower codepath.
// Target lies on the boundary so the Newton solve runs.
TEST(XConeAsymmBackwardCuda, GenPowerDirectXMatchesSlack3D) {
    constexpr int batchSize = 1;
    constexpr int n = 3;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<double>  P_val = {1.0, 1.0, 1.0};
    // Target (1, 1, 2): geometric mean of (1,1) is 1; |w| = 2 > 1
    // → cone violated; constraint binding.
    std::vector<double>  q     = {-1.0, -1.0, -2.0};
    std::vector<double>  dx    = {1.0, 1.0, 1.0};

    std::vector<double> dq_slack, dP_slack;
    {
        constexpr int m = n;
        std::vector<int64_t> A_ro = {0, 1, 2, 3};
        std::vector<int64_t> A_ci = {0, 1, 2};
        std::vector<double>  A_val = {-1.0, -1.0, -1.0};
        std::vector<double>  b     = {0.0, 0.0, 0.0};

        Cones cones{};
        cones.numGenPowerCones = 1;
        cones.genPowerAlphas = {0.5, 0.5};
        cones.genPowerDim1s = {2};
        cones.genPowerDim2s = {1};

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), P_val.size(),
                              A_ro.data(), A_ci.data(), A_val.size(),
                              cones, settings);

        double* d_P = cuda_upload(P_val);
        double* d_A = cuda_upload(A_val);
        double* d_q = cuda_upload(q);
        double* d_b = cuda_upload(b);
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(n, batchSize), dz_bar(m, batchSize), ds_bar(m, batchSize);
        cuda_set(dx_bar, dx, n);
        std::vector<double> zeros_m(m, 0.0);
        cuda_set(dz_bar, zeros_m, m);
        cuda_set(ds_bar, zeros_m, m);

        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        dq_slack = cuda_get(solver.diff_state()->dq);
        dP_slack = cuda_get(solver.diff_state()->dP_values);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    std::vector<double> dq_direct, dP_direct;
    {
        constexpr int m = 0;
        std::vector<int64_t> A_ro = {0};
        std::vector<int64_t> A_ci = {};

        Cones cones{};
        SupportedXConeT gp_xc{};
        gp_xc.kind = XConeKind::GenPower;
        gp_xc.indices = {0, 1, 2};
        gp_xc.gen_power_alphas = {0.5, 0.5};
        gp_xc.gen_power_dim2 = 1;
        cones.dir_cones.push_back(std::move(gp_xc));

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), P_val.size(),
                              A_ro.data(), A_ci.data(), 0,
                              cones, settings);

        double* d_P = cuda_upload(P_val);
        double* d_A = nullptr; cudaMalloc(&d_A, sizeof(double));
        double* d_q = cuda_upload(q);
        double* d_b = nullptr; cudaMalloc(&d_b, sizeof(double));
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(n, batchSize);
        BatchedVector dz_bar(1, batchSize);
        BatchedVector ds_bar(1, batchSize);
        cuda_set(dx_bar, dx, n);
        std::vector<double> zero1 = {0.0};
        cuda_set(dz_bar, zero1, 1);
        cuda_set(ds_bar, zero1, 1);

        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        dq_direct = cuda_get(solver.diff_state()->dq);
        dP_direct = cuda_get(solver.diff_state()->dP_values);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    ASSERT_EQ(dq_slack.size(), dq_direct.size());
    for (size_t i = 0; i < dq_slack.size(); ++i) {
        EXPECT_NEAR(dq_slack[i], dq_direct[i], 1e-2)
            << "dq[" << i << "] disagree";
    }

    ASSERT_EQ(dP_slack.size(), dP_direct.size());
    for (size_t k = 0; k < dP_slack.size(); ++k) {
        EXPECT_NEAR(dP_slack[k], dP_direct[k], 1e-2)
            << "dP[" << k << "] disagree";
    }
}

// GenPowerCone direct-x backward parity vs slack form, 4D interior case
// (cone constraint inactive). Exercises the rank-3 c3 term since dim2=2.
TEST(XConeAsymmBackwardCuda, GenPowerDirectXMatchesSlack) {
    constexpr int batchSize = 1;
    constexpr int n = 4;  // dim1=2, dim2=2

    std::vector<int64_t> P_ro = {0, 1, 2, 3, 4};
    std::vector<int64_t> P_ci = {0, 1, 2, 3};
    std::vector<double>  P_val = {1.0, 1.0, 1.0, 1.0};
    // Target (2, 2, 0.5, 0.5): for α=(0.5,0.5), p=(2,2) gives geometric
    // mean = 2; ||w||=√(0.25+0.25)≈0.707 < 2 → cone interior.
    std::vector<double>  q     = {-2.0, -2.0, -0.5, -0.5};
    std::vector<double>  dx    = {1.0, 1.0, 1.0, 1.0};

    std::vector<double> dq_slack, dP_slack;
    {
        constexpr int m = n;
        std::vector<int64_t> A_ro = {0, 1, 2, 3, 4};
        std::vector<int64_t> A_ci = {0, 1, 2, 3};
        std::vector<double>  A_val = {-1.0, -1.0, -1.0, -1.0};
        std::vector<double>  b     = {0.0, 0.0, 0.0, 0.0};

        Cones cones{};
        cones.numGenPowerCones = 1;
        cones.genPowerAlphas = {0.5, 0.5};
        cones.genPowerDim1s = {2};
        cones.genPowerDim2s = {2};

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), P_val.size(),
                              A_ro.data(), A_ci.data(), A_val.size(),
                              cones, settings);

        double* d_P = cuda_upload(P_val);
        double* d_A = cuda_upload(A_val);
        double* d_q = cuda_upload(q);
        double* d_b = cuda_upload(b);
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(n, batchSize), dz_bar(m, batchSize), ds_bar(m, batchSize);
        cuda_set(dx_bar, dx, n);
        std::vector<double> zeros_m(m, 0.0);
        cuda_set(dz_bar, zeros_m, m);
        cuda_set(ds_bar, zeros_m, m);

        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        dq_slack = cuda_get(solver.diff_state()->dq);
        dP_slack = cuda_get(solver.diff_state()->dP_values);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    std::vector<double> dq_direct, dP_direct;
    {
        constexpr int m = 0;
        std::vector<int64_t> A_ro = {0};
        std::vector<int64_t> A_ci = {};

        Cones cones{};
        SupportedXConeT gp_xc{};
        gp_xc.kind = XConeKind::GenPower;
        gp_xc.indices = {0, 1, 2, 3};
        gp_xc.gen_power_alphas = {0.5, 0.5};
        gp_xc.gen_power_dim2 = 2;
        cones.dir_cones.push_back(std::move(gp_xc));

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), P_val.size(),
                              A_ro.data(), A_ci.data(), 0,
                              cones, settings);

        double* d_P = cuda_upload(P_val);
        double* d_A = nullptr; cudaMalloc(&d_A, sizeof(double));
        double* d_q = cuda_upload(q);
        double* d_b = nullptr; cudaMalloc(&d_b, sizeof(double));
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(n, batchSize);
        BatchedVector dz_bar(1, batchSize);
        BatchedVector ds_bar(1, batchSize);
        cuda_set(dx_bar, dx, n);
        std::vector<double> zero1 = {0.0};
        cuda_set(dz_bar, zero1, 1);
        cuda_set(ds_bar, zero1, 1);

        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        dq_direct = cuda_get(solver.diff_state()->dq);
        dP_direct = cuda_get(solver.diff_state()->dP_values);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    ASSERT_EQ(dq_slack.size(), dq_direct.size());
    for (size_t i = 0; i < dq_slack.size(); ++i) {
        EXPECT_NEAR(dq_slack[i], dq_direct[i], 1e-3)
            << "dq[" << i << "] disagree";
    }

    ASSERT_EQ(dP_slack.size(), dP_direct.size());
    for (size_t k = 0; k < dP_slack.size(); ++k) {
        EXPECT_NEAR(dP_slack[k], dP_direct[k], 1e-3)
            << "dP[" << k << "] disagree";
    }
}
