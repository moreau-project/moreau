// Benchmark: CUDA direct-x backward at high dim. Times `solveAll +
// backward` for slack vs direct-x on GenPower and SOC at varying dim.
// Single-cone problems are used so dim scales the per-cone H-block
// directly. The current dense direct-x backward path emits dim*dim
// entries per cone in the augmented KKT, so factor + solveAdjoint
// time scales superlinearly with dim. Once the rank-2 / rank-3 sparse
// expansion lands on CUDA, this bench is the regression gate.
//
// Usage:
//   ./bench_xcone_backward
//
// Backward timing methodology:
//   1. solveAll(d_P, d_A, d_q, d_b)
//   2. backward(state, dx_bar, dz_bar, ds_bar, solver, stream)
//   3. cudaDeviceSynchronize, record total ms (forward + backward)
//   We also record forward-only as a reference; subtraction gives
//   pure-backward time.

#include "moreau/cones/cones.hpp"
#include "moreau/diff/diff.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/solver.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <numeric>

using namespace moreau;

template <typename T>
T* upload_replicated(const std::vector<T>& host, int B) {
    T* d = nullptr;
    if (host.empty()) { cudaMalloc(&d, sizeof(T)); return d; }
    size_t per = host.size();
    cudaMalloc(&d, sizeof(T) * per * B);
    for (int b = 0; b < B; ++b) {
        cudaMemcpy(d + b * per, host.data(), sizeof(T) * per, cudaMemcpyHostToDevice);
    }
    return d;
}

Settings make_settings() {
    Settings s;
    s.maxIter = 200;
    s.verbose = false;
    s.enableGrad = true;
    s.ipm.equilibrationSettings.enable = false;
    return s;
}

// Single GenPower cone at dim = dim1 + dim2.
struct GenPowProblem {
    int n;
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double> P_vals, q;
    std::vector<int64_t> A_slack_ro, A_slack_ci;
    std::vector<double> A_slack_vals, b_slack;
    std::vector<int64_t> A_dx_ro, A_dx_ci;
    std::vector<double> A_dx_vals;
    Cones cones_slack, cones_dx;
    GenPowProblem(int dim1, int dim2) {
        n = dim1 + dim2;
        for (int i = 0; i <= n; ++i) P_ro.push_back(i);
        for (int i = 0; i < n; ++i) P_ci.push_back(i);
        P_vals.assign(n, 1.0);
        std::vector<double> alphas(dim1, 1.0 / dim1);
        q.assign(n, 0.0);
        // Target inside cone: p_i = 2.0 (geo mean ≈ 2), w_j = 0.1.
        for (int i = 0; i < dim1; ++i) q[i] = -2.0;
        for (int j = 0; j < dim2; ++j) q[dim1 + j] = -0.1;
        // Slack
        A_slack_ro.assign(n + 1, 0);
        for (int i = 0; i <= n; ++i) A_slack_ro[i] = i;
        A_slack_ci.assign(n, 0);
        for (int i = 0; i < n; ++i) A_slack_ci[i] = i;
        A_slack_vals.assign(n, -1.0);
        b_slack.assign(n, 0.0);
        cones_slack.numGenPowerCones = 1;
        cones_slack.genPowerAlphas = alphas;
        cones_slack.genPowerDim1s = {static_cast<int64_t>(dim1)};
        cones_slack.genPowerDim2s = {static_cast<int64_t>(dim2)};
        // Direct-x: empty A
        A_dx_ro = {0};
        A_dx_ci = {};
        A_dx_vals = {};
        SupportedXConeT xc;
        xc.kind = XConeKind::GenPower;
        xc.indices.assign(n, 0);
        for (int i = 0; i < n; ++i) xc.indices[i] = i;
        xc.gen_power_alphas = alphas;
        xc.gen_power_dim2 = static_cast<int64_t>(dim2);
        cones_dx.dir_cones.push_back(std::move(xc));
    }
};

// Single SOC cone at dim = n.
struct SocProblem {
    int n;
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double> P_vals, q;
    std::vector<int64_t> A_slack_ro, A_slack_ci;
    std::vector<double> A_slack_vals, b_slack;
    std::vector<int64_t> A_dx_ro, A_dx_ci;
    std::vector<double> A_dx_vals;
    Cones cones_slack, cones_dx;
    SocProblem(int dim) {
        n = dim;
        for (int i = 0; i <= n; ++i) P_ro.push_back(i);
        for (int i = 0; i < n; ++i) P_ci.push_back(i);
        P_vals.assign(n, 1.0);
        q.assign(n, 0.0);
        q[0] = -2.0;  // pull x[0] toward -2 → SOC binding.
        A_slack_ro.assign(n + 1, 0);
        for (int i = 0; i <= n; ++i) A_slack_ro[i] = i;
        A_slack_ci.assign(n, 0);
        for (int i = 0; i < n; ++i) A_slack_ci[i] = i;
        A_slack_vals.assign(n, -1.0);
        b_slack.assign(n, 0.0);
        cones_slack.numSocCones = 1;
        cones_slack.socConeDims = {static_cast<int64_t>(dim)};
        A_dx_ro = {0};
        A_dx_ci = {};
        A_dx_vals = {};
        SupportedXConeT xc;
        xc.kind = XConeKind::SOC;
        xc.indices.assign(n, 0);
        for (int i = 0; i < n; ++i) xc.indices[i] = i;
        cones_dx.dir_cones.push_back(std::move(xc));
    }
};

template <typename Problem>
void bench_problem(const std::string& label, Problem& prob, int batchSize, int iters) {
    int64_t nnzP = static_cast<int64_t>(prob.P_vals.size());
    int64_t nnzA_slack = static_cast<int64_t>(prob.A_slack_vals.size());
    int64_t m_slack = static_cast<int64_t>(prob.b_slack.size());

    auto bench_path = [&](bool slack) {
        int m = slack ? static_cast<int>(m_slack) : 0;
        const auto& cones = slack ? prob.cones_slack : prob.cones_dx;
        const auto& A_ro = slack ? prob.A_slack_ro : prob.A_dx_ro;
        const auto& A_ci = slack ? prob.A_slack_ci : prob.A_dx_ci;
        const auto& A_vals = slack ? prob.A_slack_vals : prob.A_dx_vals;
        int64_t nnzA = static_cast<int64_t>(A_vals.size());

        CompiledSolver solver(
            prob.n, m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), nnzP,
            A_ro.data(), A_ci.data(), nnzA,
            cones, make_settings());

        double* d_P = upload_replicated(prob.P_vals, batchSize);
        double* d_q = upload_replicated(prob.q, batchSize);
        double* d_A;
        double* d_b;
        if (slack) {
            d_A = upload_replicated(prob.A_slack_vals, batchSize);
            d_b = upload_replicated(prob.b_slack, batchSize);
        } else {
            std::vector<double> dummy(1, 0.0);
            d_A = upload_replicated(dummy, batchSize);
            cudaMalloc(&d_b, sizeof(double) * batchSize);
        }

        BatchedVector dx_bar(prob.n, batchSize);
        BatchedVector dz_bar(m > 0 ? m : 1, batchSize);
        BatchedVector ds_bar(m > 0 ? m : 1, batchSize);
        std::vector<double> dx_h(prob.n * batchSize, 1.0);
        std::vector<double> z_h((m > 0 ? m : 1) * batchSize, 0.0);
        cudaMemcpy(dx_bar.data(), dx_h.data(), sizeof(double) * dx_h.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(dz_bar.data(), z_h.data(), sizeof(double) * z_h.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(ds_bar.data(), z_h.data(), sizeof(double) * z_h.size(), cudaMemcpyHostToDevice);

        // Warmup
        solver.solveAll(d_P, d_A, d_q, d_b);
        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        std::vector<double> total_ms;
        std::vector<double> forward_ms;
        for (int i = 0; i < iters; ++i) {
            cudaEvent_t s0, s1, s2;
            cudaEventCreate(&s0);
            cudaEventCreate(&s1);
            cudaEventCreate(&s2);
            cudaEventRecord(s0);
            solver.solveAll(d_P, d_A, d_q, d_b);
            cudaEventRecord(s1);
            backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
            cudaEventRecord(s2);
            cudaEventSynchronize(s2);
            float fwd = 0, total = 0;
            cudaEventElapsedTime(&fwd, s0, s1);
            cudaEventElapsedTime(&total, s0, s2);
            forward_ms.push_back(fwd);
            total_ms.push_back(total);
            cudaEventDestroy(s0);
            cudaEventDestroy(s1);
            cudaEventDestroy(s2);
        }
        std::sort(total_ms.begin(), total_ms.end());
        std::sort(forward_ms.begin(), forward_ms.end());
        double t_med = total_ms[total_ms.size() / 2];
        double f_med = forward_ms[forward_ms.size() / 2];
        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
        return std::make_pair(f_med, t_med);
    };

    auto s = bench_path(true);
    auto d = bench_path(false);
    double s_bwd = s.second - s.first;
    double d_bwd = d.second - d.first;
    double speedup = (d_bwd > 0.0) ? s_bwd / d_bwd : 0.0;

    std::cout << "  " << std::left << std::setw(20) << label
              << " B=" << std::setw(3) << batchSize
              << "  slack(fwd/bwd) = " << std::fixed << std::setprecision(2)
              << std::setw(7) << s.first << "/" << std::setw(7) << s_bwd << "ms"
              << "  dx(fwd/bwd) = "
              << std::setw(7) << d.first << "/" << std::setw(7) << d_bwd << "ms"
              << "  bwd-speedup=" << std::setprecision(2) << speedup << "x\n";
}

int main() {
    const int ITERS = 5;

    std::cout << "=== Direct-x backward CUDA bench ===\n";
    std::cout << "Forward + backward timed via cudaEvent. bwd = total - fwd.\n";
    std::cout << "Negative speedup means direct-x slower than slack at this size.\n\n";

    std::cout << "GenPower direct-x (single cone, varying dim, dim1=dim2=dim/2):\n";
    for (int dim : {8, 16, 32, 64}) {
        int dim1 = dim / 2;
        int dim2 = dim - dim1;
        for (int B : {1, 16, 64}) {
            GenPowProblem prob(dim1, dim2);
            bench_problem("GenPow dim=" + std::to_string(dim), prob, B, ITERS);
        }
    }

    std::cout << "\nSOC direct-x (single cone, varying dim):\n";
    for (int dim : {8, 16, 32, 64, 128}) {
        for (int B : {1, 16, 64}) {
            SocProblem prob(dim);
            bench_problem("SOC dim=" + std::to_string(dim), prob, B, ITERS);
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
