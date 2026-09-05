/**
 * @brief Benchmark: CUDA slack vs CUDA direct-x for asymmetric cones (Exp, Power, GenPow).
 *
 * Same QP `min 0.5||x - target||^2` solved two ways:
 *   slack:    A = -I, b = 0, s in K  (n extra rows, standard cone constraint)
 *   direct-x: dir_cones with Exp/Power/GenPower  (cone augmentation in (1,1) block)
 *
 * Usage:
 *   ./bench_xcone_asymmetric
 *
 * Output: timing table comparing slack vs direct-x for K=1,10,100,1000 stacked cones.
 */

#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <numeric>

using namespace moreau;

// ─── Utility ────────────────────────────────────────────────────────────────

template <typename T>
T* cuda_upload(const std::vector<T>& host) {
    T* d = nullptr;
    if (host.empty()) {
        cudaMalloc(&d, sizeof(T));  // at least 1 byte to avoid null issues
        return d;
    }
    cudaMalloc(&d, sizeof(T) * host.size());
    cudaMemcpy(d, host.data(), sizeof(T) * host.size(), cudaMemcpyHostToDevice);
    return d;
}

// Upload `batchSize` identical copies of `host` (for batched bench — same problem
// replicated across batch dim).
template <typename T>
T* cuda_upload_replicated(const std::vector<T>& host, int batchSize) {
    T* d = nullptr;
    if (host.empty()) {
        cudaMalloc(&d, sizeof(T));
        return d;
    }
    size_t per = host.size();
    cudaMalloc(&d, sizeof(T) * per * batchSize);
    for (int b = 0; b < batchSize; ++b) {
        cudaMemcpy(d + b * per, host.data(), sizeof(T) * per,
                   cudaMemcpyHostToDevice);
    }
    return d;
}

// Build diagonal P = I as CSR (full symmetric, diagonal only).
void make_diag_P(int n,
                 std::vector<int64_t>& ro, std::vector<int64_t>& ci,
                 std::vector<double>& vals) {
    ro.resize(n + 1);
    ci.resize(n);
    vals.resize(n, 1.0);
    for (int i = 0; i <= n; ++i) ro[i] = i;
    for (int i = 0; i < n; ++i) ci[i] = i;
}

// Build neg-eye A = -I (m x n, m == n) as CSR.
void make_neg_eye(int n,
                  std::vector<int64_t>& ro, std::vector<int64_t>& ci,
                  std::vector<double>& vals) {
    ro.resize(n + 1);
    ci.resize(n);
    vals.resize(n, -1.0);
    for (int i = 0; i <= n; ++i) ro[i] = i;
    for (int i = 0; i < n; ++i) ci[i] = i;
}

// Empty A (0 x n).
void make_empty_A(int n,
                  std::vector<int64_t>& ro, std::vector<int64_t>& ci,
                  std::vector<double>& vals) {
    ro = {0};
    ci.clear();
    vals.clear();
    (void)n;
}

Settings make_settings() {
    Settings s;
    s.maxIter = 200;
    s.verbose = false;
    // Note: CPU bench disables equilibration; CUDA does NOT, because on CUDA
    // equilibration helps Exp converge (15 vs 20 iters). Different optimal
    // settings between backends — IPM trajectory differs because of how the
    // initial scaling kicks off.
    return s;
}

struct SolveResult {
    double time_ms;
    int32_t iterations;
    int32_t status;
};

// Time a single-batch solve using cudaEvent for accurate GPU timing.
SolveResult time_solve(CompiledSolver& solver,
                       double* d_P, double* d_A, double* d_q, double* d_b) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    int32_t iters = solver.info.iterations;
    int32_t st = 0;
    cudaMemcpy(&st, solver.info.status_device, sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    return {static_cast<double>(elapsed_ms), iters, st};
}

const char* status_name(int32_t st) {
    switch (st) {
        case 1: return "Solved";
        case 4: return "AlmostSolved";
        case 7: return "MaxIter";
        default: return "Other";
    }
}

// ─── Problem constructors ────────────────────────────────────────────────────

struct ExpProblem {
    int n;
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double>  P_vals, q;
    // Slack form
    std::vector<int64_t> A_slack_ro, A_slack_ci;
    std::vector<double>  A_slack_vals, b_slack;
    Cones cones_slack;
    // Direct-x form
    std::vector<int64_t> A_dx_ro, A_dx_ci;
    std::vector<double>  A_dx_vals;
    Cones cones_dx;

    explicit ExpProblem(int K) {
        n = 3 * K;
        make_diag_P(n, P_ro, P_ci, P_vals);

        q.resize(n);
        for (int k = 0; k < K; ++k) {
            q[3*k + 0] = 0.0;
            q[3*k + 1] = -1.0;
            q[3*k + 2] = -5.0;
        }

        // Slack: A = -I, b = 0, K exp cones
        make_neg_eye(n, A_slack_ro, A_slack_ci, A_slack_vals);
        b_slack.assign(n, 0.0);
        cones_slack = Cones{};
        cones_slack.numExpCones = K;

        // Direct-x: no A rows, K x-cones
        make_empty_A(n, A_dx_ro, A_dx_ci, A_dx_vals);
        cones_dx = Cones{};
        for (int k = 0; k < K; ++k) {
            SupportedXConeT xc;
            xc.kind = XConeKind::Exp;
            xc.indices = {3*k, 3*k+1, 3*k+2};
            cones_dx.dir_cones.push_back(xc);
        }
    }
};

struct PowProblem {
    int n;
    double alpha;
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double>  P_vals, q;
    std::vector<int64_t> A_slack_ro, A_slack_ci;
    std::vector<double>  A_slack_vals, b_slack;
    Cones cones_slack;
    std::vector<int64_t> A_dx_ro, A_dx_ci;
    std::vector<double>  A_dx_vals;
    Cones cones_dx;

    PowProblem(int K, double alpha_) : alpha(alpha_) {
        n = 3 * K;
        make_diag_P(n, P_ro, P_ci, P_vals);

        q.resize(n);
        for (int k = 0; k < K; ++k) {
            q[3*k + 0] = -2.0;
            q[3*k + 1] = -3.0;
            q[3*k + 2] = -1.0;
        }

        make_neg_eye(n, A_slack_ro, A_slack_ci, A_slack_vals);
        b_slack.assign(n, 0.0);
        cones_slack = Cones{};
        cones_slack.numPowerCones = K;
        cones_slack.powerAlphas.assign(K, alpha);

        make_empty_A(n, A_dx_ro, A_dx_ci, A_dx_vals);
        cones_dx = Cones{};
        for (int k = 0; k < K; ++k) {
            SupportedXConeT xc;
            xc.kind = XConeKind::Power;
            xc.power_alpha = alpha;
            xc.indices = {3*k, 3*k+1, 3*k+2};
            cones_dx.dir_cones.push_back(xc);
        }
    }
};

struct GenPowProblem {
    int n;
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double>  P_vals, q;
    std::vector<int64_t> A_slack_ro, A_slack_ci;
    std::vector<double>  A_slack_vals, b_slack;
    Cones cones_slack;
    std::vector<int64_t> A_dx_ro, A_dx_ci;
    std::vector<double>  A_dx_vals;
    Cones cones_dx;

    // Single GenPowCone with uniform alphas
    GenPowProblem(int dim1, int dim2) {
        n = dim1 + dim2;
        make_diag_P(n, P_ro, P_ci, P_vals);

        std::vector<double> alphas(dim1, 1.0 / dim1);

        q.resize(n);
        for (int i = 0; i < dim1; ++i) q[i] = -2.0;
        for (int j = 0; j < dim2; ++j) q[dim1 + j] = -0.5;

        make_neg_eye(n, A_slack_ro, A_slack_ci, A_slack_vals);
        b_slack.assign(n, 0.0);
        cones_slack = Cones{};
        cones_slack.numGenPowerCones = 1;
        cones_slack.genPowerAlphas = alphas;
        cones_slack.genPowerDim1s = {static_cast<int64_t>(dim1)};
        cones_slack.genPowerDim2s = {static_cast<int64_t>(dim2)};

        make_empty_A(n, A_dx_ro, A_dx_ci, A_dx_vals);
        cones_dx = Cones{};
        SupportedXConeT xc;
        xc.kind = XConeKind::GenPower;
        for (int i = 0; i < n; ++i) xc.indices.push_back(i);
        xc.gen_power_alphas = alphas;
        xc.gen_power_dim2 = static_cast<int64_t>(dim2);
        cones_dx.dir_cones.push_back(xc);
    }
};

// Stacked 3D GenPow (same shape as Pow, different code path)
struct GenPow3DProblem {
    int n;
    double alpha;
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double>  P_vals, q;
    std::vector<int64_t> A_slack_ro, A_slack_ci;
    std::vector<double>  A_slack_vals, b_slack;
    Cones cones_slack;
    std::vector<int64_t> A_dx_ro, A_dx_ci;
    std::vector<double>  A_dx_vals;
    Cones cones_dx;

    GenPow3DProblem(int K, double alpha_) : alpha(alpha_) {
        n = 3 * K;
        make_diag_P(n, P_ro, P_ci, P_vals);

        std::vector<double> alphas = {alpha, 1.0 - alpha};
        int dim2 = 1;

        q.resize(n);
        for (int k = 0; k < K; ++k) {
            q[3*k + 0] = -2.0;
            q[3*k + 1] = -3.0;
            q[3*k + 2] = -1.0;
        }

        make_neg_eye(n, A_slack_ro, A_slack_ci, A_slack_vals);
        b_slack.assign(n, 0.0);
        cones_slack = Cones{};
        cones_slack.numGenPowerCones = K;
        for (int k = 0; k < K; ++k) {
            for (double a : alphas) cones_slack.genPowerAlphas.push_back(a);
            cones_slack.genPowerDim1s.push_back(2);
            cones_slack.genPowerDim2s.push_back(dim2);
        }

        make_empty_A(n, A_dx_ro, A_dx_ci, A_dx_vals);
        cones_dx = Cones{};
        for (int k = 0; k < K; ++k) {
            SupportedXConeT xc;
            xc.kind = XConeKind::GenPower;
            xc.indices = {3*k, 3*k+1, 3*k+2};
            xc.gen_power_alphas = alphas;
            xc.gen_power_dim2 = static_cast<int64_t>(dim2);
            cones_dx.dir_cones.push_back(xc);
        }
    }
};

// ─── Bench harness ──────────────────────────────────────────────────────────

struct BenchResult {
    double slack_ms;
    double dx_ms;
    int32_t slack_iters;
    int32_t dx_iters;
    int32_t slack_status;
    int32_t dx_status;
};

template <typename Problem>
BenchResult bench_problem(Problem& prob, int iters, int batchSize = 1) {
    int64_t nnzP = static_cast<int64_t>(prob.P_vals.size());
    int64_t nnzA_slack = static_cast<int64_t>(prob.A_slack_vals.size());
    int64_t m_slack = static_cast<int64_t>(prob.b_slack.size());

    // ---- Slack form ----
    CompiledSolver solver_slack(
        prob.n, static_cast<int>(m_slack), batchSize,
        prob.P_ro.data(), prob.P_ci.data(), nnzP,
        prob.A_slack_ro.data(), prob.A_slack_ci.data(), nnzA_slack,
        prob.cones_slack, make_settings());

    double* d_P_s   = cuda_upload_replicated(prob.P_vals,       batchSize);
    double* d_A_s   = cuda_upload_replicated(prob.A_slack_vals, batchSize);
    double* d_q_s   = cuda_upload_replicated(prob.q,            batchSize);
    double* d_b_s   = cuda_upload_replicated(prob.b_slack,      batchSize);

    // Warmup
    time_solve(solver_slack, d_P_s, d_A_s, d_q_s, d_b_s);

    std::vector<double> slack_times;
    slack_times.reserve(iters);
    SolveResult last_s{};
    for (int i = 0; i < iters; ++i) {
        last_s = time_solve(solver_slack, d_P_s, d_A_s, d_q_s, d_b_s);
        slack_times.push_back(last_s.time_ms);
    }
    std::sort(slack_times.begin(), slack_times.end());
    double s_med = slack_times[slack_times.size() / 2];

    cudaFree(d_P_s); cudaFree(d_A_s); cudaFree(d_q_s); cudaFree(d_b_s);

    // ---- Direct-x form ----
    int64_t nnzA_dx = static_cast<int64_t>(prob.A_dx_vals.size());
    // A_dx_ro has exactly 1 element ({0}) for empty A
    CompiledSolver solver_dx(
        prob.n, 0, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), nnzP,
        prob.A_dx_ro.data(), prob.A_dx_ci.data(), nnzA_dx,
        prob.cones_dx, make_settings());

    double* d_P_d   = cuda_upload_replicated(prob.P_vals, batchSize);
    // For empty A, just allocate 1 double per batch to avoid null ptr issues
    std::vector<double> dummy_A(1, 0.0);
    double* d_A_d   = cuda_upload_replicated(dummy_A, batchSize);
    double* d_q_d   = cuda_upload_replicated(prob.q, batchSize);
    double* d_b_d;
    cudaMalloc(&d_b_d, sizeof(double) * batchSize);  // empty but non-null

    // Warmup
    time_solve(solver_dx, d_P_d, d_A_d, d_q_d, d_b_d);

    std::vector<double> dx_times;
    dx_times.reserve(iters);
    SolveResult last_d{};
    for (int i = 0; i < iters; ++i) {
        last_d = time_solve(solver_dx, d_P_d, d_A_d, d_q_d, d_b_d);
        dx_times.push_back(last_d.time_ms);
    }
    std::sort(dx_times.begin(), dx_times.end());
    double d_med = dx_times[dx_times.size() / 2];

    cudaFree(d_P_d); cudaFree(d_A_d); cudaFree(d_q_d); cudaFree(d_b_d);

    return {s_med, d_med, last_s.iterations, last_d.iterations,
            last_s.status, last_d.status};
}

void print_row(const std::string& label, int n, int B, const BenchResult& r) {
    double speedup = (r.dx_ms > 0.0) ? r.slack_ms / r.dx_ms : 0.0;
    std::cout << "  " << std::left << std::setw(18) << label
              << " n=" << std::setw(5) << n
              << " B=" << std::setw(4) << B
              << " slack=" << std::fixed << std::setprecision(2)
              << std::setw(8) << r.slack_ms << "ms"
              << "(it=" << r.slack_iters << "," << status_name(r.slack_status) << ")"
              << "  dx=" << std::setw(8) << r.dx_ms << "ms"
              << "(it=" << r.dx_iters << "," << status_name(r.dx_status) << ")"
              << "  speedup=" << std::setprecision(2) << speedup << "x"
              << "\n";
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    const double POW_ALPHA = 0.4;
    const int ITERS = 5;

    std::cout << "=== Asymmetric direct-x vs slack benchmark (CUDA) ===\n";
    std::cout << "Problem: min 0.5||x - target||^2  s.t.  x in K\n";
    std::cout << "         slack form: A=-I, b=0, s in K\n";
    std::cout << "         direct-x:   dir_cones with Exp/Power/GenPower\n";
    std::cout << "         (each batch entry = same problem replicated)\n\n";

    const std::vector<int> batch_sizes = {1, 16, 64, 256};

    // ---- ExpCone ----
    std::cout << "ExpCone (3D, K stacked):\n";
    for (int K : {10, 100, 1000}) {
        for (int B : batch_sizes) {
            ExpProblem prob(K);
            auto r = bench_problem(prob, ITERS, B);
            print_row("Exp K=" + std::to_string(K), 3 * K, B, r);
        }
    }

    // ---- PowerCone ----
    std::cout << "\nPowerCone (3D, alpha=0.4, K stacked):\n";
    for (int K : {10, 100, 1000}) {
        for (int B : batch_sizes) {
            PowProblem prob(K, POW_ALPHA);
            auto r = bench_problem(prob, ITERS, B);
            print_row("Pow K=" + std::to_string(K), 3 * K, B, r);
        }
    }

    // ---- GenPowerCone (single variable-dim cone) ----
    // Note: CUDA kernel stack size limits safe dim to ~100 for single-cone form.
    std::cout << "\nGenPowerCone (uniform alpha, single cone, varying dim):\n";
    for (auto [dim1, dim2] : std::vector<std::pair<int,int>>{{2,2},{4,8},{8,16}}) {
        for (int B : batch_sizes) {
            GenPowProblem prob(dim1, dim2);
            auto r = bench_problem(prob, ITERS, B);
            print_row("GenPow " + std::to_string(dim1) + "/" + std::to_string(dim2),
                      dim1 + dim2, B, r);
        }
    }

    // ---- 3D GenPowerCone stacked (same shape as Pow, different code path) ----
    std::cout << "\n3D GenPowerCone stacked (mathematically same shape as Pow, different code path):\n";
    for (int K : {10, 100, 1000}) {
        for (int B : batch_sizes) {
            GenPow3DProblem prob(K, POW_ALPHA);
            auto r = bench_problem(prob, ITERS, B);
            print_row("GenPow3D K=" + std::to_string(K), 3 * K, B, r);
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}
