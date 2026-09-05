// test_xcone_backward.cpp
//
// Native CUDA direct-x backward (IFT-direct path). Mirrors the CPU
// xcone_nonneg_equivalence::test_backward_direct_x_matches_slack: solve
// the same QP via slack form vs. direct-x form on CUDA, run backward
// with the same upstream `dx`, and assert dP/dq agree.
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

// Small QP: min 0.5 x'Px + q'x  s.t. x >= 0 (direct-x or slack).
// P = 2I, q = (-1, 2, -0.5). Slack-form unconstrained opt is (0.5, -1, 0.25),
// clipped to (0.5, 0, 0.25) by the nonneg cone.
TEST(XConeBackwardCuda, NonnegDirectXMatchesSlack) {
    constexpr int batchSize = 1;
    constexpr int n = 3;

    // P = 2I (diagonal).
    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<double>  P_val = {2.0, 2.0, 2.0};
    std::vector<double>  q     = {-1.0, 2.0, -0.5};
    std::vector<double>  dx    = {1.0, 1.0, 1.0};

    // ----- Slack form: A = -I, b = 0, single nonneg cone of dim n -----
    std::vector<double> dq_slack, dP_slack;
    {
        constexpr int m = n;
        std::vector<int64_t> A_ro = {0, 1, 2, 3};
        std::vector<int64_t> A_ci = {0, 1, 2};
        std::vector<double>  A_val = {-1.0, -1.0, -1.0};
        std::vector<double>  b     = {0.0, 0.0, 0.0};

        Cones cones{};
        cones.numNonnegCones = n;

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;

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

    // ----- Direct-x form: empty A, single nonneg direct-x on all of x -----
    std::vector<double> dq_direct, dP_direct;
    {
        constexpr int m = 0;
        std::vector<int64_t> A_ro = {0};
        std::vector<int64_t> A_ci = {};
        std::vector<double>  A_val;  // empty
        std::vector<double>  b;      // empty

        Cones cones{};
        cones.dir_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {0, 1, 2}});

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;

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
        BatchedVector dz_bar(1, batchSize);  // m=0 → placeholder
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

    // dq must match (same n components).
    ASSERT_EQ(dq_slack.size(), dq_direct.size());
    for (size_t i = 0; i < dq_slack.size(); ++i) {
        EXPECT_NEAR(dq_slack[i], dq_direct[i], 1e-4)
            << "dq[" << i << "] disagree";
    }

    // dP must match.
    ASSERT_EQ(dP_slack.size(), dP_direct.size());
    for (size_t k = 0; k < dP_slack.size(); ++k) {
        EXPECT_NEAR(dP_slack[k], dP_direct[k], 1e-4)
            << "dP[" << k << "] disagree";
    }
}

// SOC direct-x backward parity: solve the same SOC-constrained QP via
// slack form vs direct-x form on CUDA, run backward with the same
// upstream `dx`, and assert dP/dq agree. Constraint is active on the
// SOC boundary so the cone-projection Jacobian is the boundary case
// (not pure identity / zero). Equilibration is disabled to keep the
// reference math straightforward; the IFT-direct path is invariant
// under uniform per-cone equilibration anyway.
TEST(XConeBackwardCuda, SOCDirectXMatchesSlack) {
    constexpr int batchSize = 1;
    constexpr int n = 3;
    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<double>  P_val = {1.0, 1.0, 1.0};
    std::vector<double>  q     = {1.0, 2.0, 0.0};      // pushes x[0] negative
    std::vector<double>  dx    = {1.0, 1.0, 1.0};

    // ----- Slack form: A = -I, b = 0, single SOC of dim 3 -----
    std::vector<double> dq_slack, dP_slack;
    {
        constexpr int m = n;
        std::vector<int64_t> A_ro = {0, 1, 2, 3};
        std::vector<int64_t> A_ci = {0, 1, 2};
        std::vector<double>  A_val = {-1.0, -1.0, -1.0};
        std::vector<double>  b     = {0.0, 0.0, 0.0};

        Cones cones{};
        cones.numSocCones = 1;
        cones.socConeDims = {3};

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

    // ----- Direct-x form: empty A, single SOC direct-x on all of x -----
    std::vector<double> dq_direct, dP_direct;
    {
        constexpr int m = 0;
        std::vector<int64_t> A_ro = {0};
        std::vector<int64_t> A_ci = {};

        Cones cones{};
        cones.dir_cones.push_back(SupportedXConeT{XConeKind::SOC, {0, 1, 2}});

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
        EXPECT_NEAR(dq_slack[i], dq_direct[i], 1e-4)
            << "dq[" << i << "] disagree";
    }
    ASSERT_EQ(dP_slack.size(), dP_direct.size());
    for (size_t k = 0; k < dP_slack.size(); ++k) {
        EXPECT_NEAR(dP_slack[k], dP_direct[k], 1e-4)
            << "dP[" << k << "] disagree";
    }
}

// PSD direct-x backward parity: 2x2 PSD cone constraint via slack form
// vs direct-x form on CUDA. Equilibration disabled. n=3 primal vars
// correspond to svec(X) = (X[0,0], √2·X[0,1], X[1,1]). Solution lies on
// the PSD boundary when q points outside the cone.
TEST(XConeBackwardCuda, PSDDirectXMatchesSlack) {
    constexpr int batchSize = 1;
    constexpr int n = 3;
    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<double>  P_val = {1.0, 1.0, 1.0};
    // q pulls X off the PSD cone interior so the constraint binds.
    std::vector<double>  q     = {-0.5, 1.0, -0.5};
    std::vector<double>  dx    = {1.0, 1.0, 1.0};

    // ----- Slack form: A = -I, b = 0, single PSD cone (psd_k=2) -----
    std::vector<double> dq_slack, dP_slack;
    {
        constexpr int m = n;
        std::vector<int64_t> A_ro = {0, 1, 2, 3};
        std::vector<int64_t> A_ci = {0, 1, 2};
        std::vector<double>  A_val = {-1.0, -1.0, -1.0};
        std::vector<double>  b     = {0.0, 0.0, 0.0};

        Cones cones{};
        cones.numPsdCones = 1;
        cones.psdConeDims = {2};
        cones.psdConeDimsOriginal = {2};
        cones.psdSortPerm = {0};

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

    // ----- Direct-x form: empty A, single PSD direct-x on all of x -----
    std::vector<double> dq_direct, dP_direct;
    {
        constexpr int m = 0;
        std::vector<int64_t> A_ro = {0};
        std::vector<int64_t> A_ci = {};

        Cones cones{};
        SupportedXConeT xc{XConeKind::PSD, {0, 1, 2}};
        xc.psd_k = 2;
        cones.dir_cones.push_back(xc);

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
        EXPECT_NEAR(dq_slack[i], dq_direct[i], 1e-4)
            << "dq[" << i << "] disagree";
    }
    ASSERT_EQ(dP_slack.size(), dP_direct.size());
    for (size_t k = 0; k < dP_slack.size(); ++k) {
        EXPECT_NEAR(dP_slack[k], dP_direct[k], 1e-4)
            << "dP[" << k << "] disagree";
    }
}

// Batched direct-x backward: solve two distinct nonneg-direct-x QPs in a
// single batched CompiledSolver call, then run backward and assert each
// per-batch gradient matches its corresponding single-problem reference.
TEST(XConeBackwardCuda, NonnegDirectXBatchedMatchesSingle) {
    constexpr int batchSize = 2;
    constexpr int n = 3;
    constexpr int m = 0;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<int64_t> A_ro = {0};
    // Two distinct problems sharing P structure but with different q values.
    // q1 leaves the cone interior; q2 pushes one component negative so the
    // cone binds asymmetrically — exercises the per-batch H_x dependence.
    std::vector<double> P_val_b = {2.0, 2.0, 2.0,  2.0, 2.0, 2.0};
    std::vector<double> q_b     = {-1.0, -1.0, -1.0,  -0.5, -0.5,  1.0};
    std::vector<double> dx_b    = {1.0, 1.0, 1.0,  1.0, 1.0, 1.0};

    auto run_direct_x = [&](int batch_dim, const std::vector<double>& P_val,
                            const std::vector<double>& q,
                            const std::vector<double>& dx)
        -> std::pair<std::vector<double>, std::vector<double>>
    {
        Cones cones{};
        cones.dir_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {0, 1, 2}});

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batch_dim,
                              P_ro.data(), P_ci.data(), 3,
                              A_ro.data(), nullptr, 0,
                              cones, settings);

        double* d_P = cuda_upload(P_val);
        double* d_A = nullptr; cudaMalloc(&d_A, sizeof(double));
        double* d_q = cuda_upload(q);
        double* d_b = nullptr; cudaMalloc(&d_b, sizeof(double));
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(n, batch_dim);
        BatchedVector dz_bar(1, batch_dim);
        BatchedVector ds_bar(1, batch_dim);
        cuda_set(dx_bar, dx, n * batch_dim);
        std::vector<double> zeros(batch_dim, 0.0);
        cuda_set(dz_bar, zeros, batch_dim);
        cuda_set(ds_bar, zeros, batch_dim);

        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        auto dq = cuda_get(solver.diff_state()->dq);
        auto dP = cuda_get(solver.diff_state()->dP_values);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
        return {dq, dP};
    };

    auto batched = run_direct_x(batchSize, P_val_b, q_b, dx_b);
    auto& dq_batched = batched.first;
    auto& dP_batched = batched.second;

    for (int b = 0; b < batchSize; ++b) {
        std::vector<double> P1(P_val_b.begin() + b * 3, P_val_b.begin() + (b + 1) * 3);
        std::vector<double> q1(q_b.begin() + b * n, q_b.begin() + (b + 1) * n);
        std::vector<double> dx1(dx_b.begin() + b * n, dx_b.begin() + (b + 1) * n);
        auto single = run_direct_x(1, P1, q1, dx1);
        const auto& dq1 = single.first;
        const auto& dP1 = single.second;

        for (int i = 0; i < n; ++i) {
            EXPECT_NEAR(dq_batched[b * n + i], dq1[i], 1e-4)
                << "batch " << b << " dq[" << i << "] disagree";
        }
        for (size_t k = 0; k < dP1.size(); ++k) {
            EXPECT_NEAR(dP_batched[b * 3 + k], dP1[k], 1e-4)
                << "batch " << b << " dP[" << k << "] disagree";
        }
    }
}

// dz_x backward: upstream gradient on the direct-x dual must produce
// gradients on (q) that match finite differences of `z_x_orig`. The
// boundary-active nonneg case has H_x = I, so d(z_x_internal)/d(q) = 1
// and we expect d(z_x_orig[k]) / d(q[k]) = 1 along the active dimension.
TEST(XConeBackwardCuda, NonnegDirectXDzXMatchesFiniteDifference) {
    constexpr int batchSize = 1;
    constexpr int n = 3;
    constexpr int m = 0;
    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<int64_t> A_ro = {0};
    std::vector<double>  P_val = {1.0, 1.0, 1.0};
    // Active boundary on x[2] (z_x[2] > 0 at convergence).
    std::vector<double>  q     = {-0.5, -0.5, 1.0};

    auto solve_zx = [&](const std::vector<double>& qv)
        -> std::vector<double>
    {
        Cones cones{};
        cones.dir_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {0, 1, 2}});

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), 3,
                              A_ro.data(), nullptr, 0,
                              cones, settings);

        double* d_P = cuda_upload(P_val);
        double* d_A = nullptr; cudaMalloc(&d_A, sizeof(double));
        double* d_q = cuda_upload(qv);
        double* d_b = nullptr; cudaMalloc(&d_b, sizeof(double));
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        // Read z_x out of variables (post τ-divide is in cache_solution_for_backward)
        // — easier path: re-pull from solver.diff_state()->z_x after a no-op backward.
        BatchedVector dx_bar(n, batchSize), dz_bar(1, batchSize), ds_bar(1, batchSize);
        std::vector<double> zero_n(n, 0.0);
        cuda_set(dx_bar, zero_n, n);
        std::vector<double> zero1 = {0.0};
        cuda_set(dz_bar, zero1, 1);
        cuda_set(ds_bar, zero1, 1);
        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        // z_x snapshotted in equilibrated frame (τ=1, no equilibration).
        // cuda_get on diff_state.z_x gives the equilibrated values.
        auto z_x = cuda_get(solver.diff_state()->z_x);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
        return z_x;
    };

    auto solve_with_dz_x = [&](int j_dz_x) -> std::vector<double>
    {
        Cones cones{};
        cones.dir_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {0, 1, 2}});

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.equilibrationSettings.enable = false;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), 3,
                              A_ro.data(), nullptr, 0,
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
        BatchedVector dz_x_bar(n, batchSize);
        std::vector<double> zero_n(n, 0.0);
        cuda_set(dx_bar, zero_n, n);
        std::vector<double> zero1 = {0.0};
        cuda_set(dz_bar, zero1, 1);
        cuda_set(ds_bar, zero1, 1);
        std::vector<double> ej(n, 0.0);
        ej[j_dz_x] = 1.0;
        cuda_set(dz_x_bar, ej, n);

        backward_with_dz_x(*solver.diff_state(),
                           dx_bar, dz_bar, ds_bar, &dz_x_bar, solver, 0);
        cudaDeviceSynchronize();

        auto dq = cuda_get(solver.diff_state()->dq);
        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
        return dq;
    };

    int j = 2;
    auto analytic = solve_with_dz_x(j);

    double eps = 1e-6;
    std::vector<double> fd(n, 0.0);
    for (int k = 0; k < n; ++k) {
        std::vector<double> qp = q; qp[k] += eps;
        std::vector<double> qm = q; qm[k] -= eps;
        auto z_p = solve_zx(qp);
        auto z_m = solve_zx(qm);
        fd[k] = (z_p[j] - z_m[j]) / (2.0 * eps);
    }

    for (int k = 0; k < n; ++k) {
        EXPECT_NEAR(analytic[k], fd[k], 1e-3)
            << "dz_x dq[" << k << "] analytic=" << analytic[k]
            << " fd=" << fd[k];
    }
}

// Woodbury forward + direct-x backward. The Woodbury KKT solver is opt-in
// (kktSolverType=Woodbury) and accepts direct-x nonneg cones in the forward
// solve, but DiffWoodbury has no x-cone adjoint path. The backward must
// therefore route through the general DiffKKT/cuDSS adjoint while the
// forward still uses Woodbury. This test pins that composition: the
// Woodbury-forward gradient must match the all-cuDSS gradient.
//
// Problem (Woodbury-compatible: diagonal P, one zero cone, k_total<n):
//   n=3, m=1.  P = I.  A = [1 0 0], b = 0.5  →  x[0] = 0.5.
//   direct-x nonneg on {1, 2}.  q = (0, -1, -2)  →  x ≈ (0.5, 1, 2),
//   cone inactive.
TEST(XConeBackwardCuda, WoodburyForwardDirectXBackwardMatchesCuDSS) {
    constexpr int batchSize = 1;
    constexpr int n = 3, m = 1;

    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    std::vector<double>  P_val = {1.0, 1.0, 1.0};
    std::vector<int64_t> A_ro = {0, 1};
    std::vector<int64_t> A_ci = {0};
    std::vector<double>  A_val = {1.0};
    std::vector<double>  b   = {0.5};
    std::vector<double>  q   = {0.0, -1.0, -2.0};
    std::vector<double>  dx  = {1.0, 1.0, 1.0};

    auto solve_and_backward = [&](KKTSolverType kkt_type,
                                  std::vector<double>& dq_out,
                                  std::vector<double>& dP_out,
                                  bool& is_woodbury) {
        Cones cones{};
        cones.numZeroCones = 1;
        cones.dir_cones.push_back(SupportedXConeT{XConeKind::Nonneg, {1, 2}});

        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.kktSolverType = kkt_type;

        CompiledSolver solver(n, m, batchSize,
                              P_ro.data(), P_ci.data(), P_val.size(),
                              A_ro.data(), A_ci.data(), A_val.size(),
                              cones, settings);
        is_woodbury = solver.kkt->isWoodbury();

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

        dq_out = cuda_get(solver.diff_state()->dq);
        dP_out = cuda_get(solver.diff_state()->dP_values);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    };

    std::vector<double> dq_wb, dP_wb, dq_cudss, dP_cudss;
    bool wb_selected = false, cudss_selected = false;
    solve_and_backward(KKTSolverType::Woodbury, dq_wb, dP_wb, wb_selected);
    solve_and_backward(KKTSolverType::CuDSS, dq_cudss, dP_cudss, cudss_selected);

    // The Woodbury request must actually have produced a Woodbury forward
    // solver — otherwise the test isn't exercising the composition.
    EXPECT_TRUE(wb_selected) << "kktSolverType=Woodbury did not select Woodbury";
    EXPECT_FALSE(cudss_selected) << "kktSolverType=CuDSS selected Woodbury";

    ASSERT_EQ(dq_wb.size(), dq_cudss.size());
    for (size_t i = 0; i < dq_wb.size(); ++i) {
        EXPECT_NEAR(dq_wb[i], dq_cudss[i], 1e-4) << "dq[" << i << "] disagree";
    }
    ASSERT_EQ(dP_wb.size(), dP_cudss.size());
    for (size_t k = 0; k < dP_wb.size(); ++k) {
        EXPECT_NEAR(dP_wb[k], dP_cudss[k], 1e-4) << "dP[" << k << "] disagree";
    }
}
