// Focused micro-bench: Exp K=1000 B=256 only, repeated 10 times.
// Measures wall time of solveAll() to characterize the η correction's
// per-iter cost.

#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/solver.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>

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

int main() {
    constexpr int K = 1000;
    constexpr int B = 256;
    constexpr int N_RUNS = 10;
    constexpr int n = 3 * K;

    std::vector<int64_t> P_ro(n + 1), P_ci(n);
    std::vector<double> P_vals(n, 1.0);
    for (int i = 0; i <= n; ++i) P_ro[i] = i;
    for (int i = 0; i < n; ++i) P_ci[i] = i;

    std::vector<double> q(n);
    for (int k = 0; k < K; ++k) {
        q[3*k + 0] = 0.0;
        q[3*k + 1] = -1.0;
        q[3*k + 2] = -5.0;
    }

    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci;
    Cones cones;
    for (int k = 0; k < K; ++k) {
        SupportedXConeT xc;
        xc.kind = XConeKind::Exp;
        xc.indices = {3*k, 3*k+1, 3*k+2};
        cones.x_cones.push_back(xc);
    }

    Settings settings;
    settings.maxIter = 100;
    settings.verbose = false;

    CompiledSolver solver(n, 0, B, P_ro.data(), P_ci.data(), n,
                          A_ro.data(), A_ci.data(), 0,
                          cones, settings);

    double* d_P = upload_replicated(P_vals, B);
    double* d_q = upload_replicated(q, B);
    std::vector<double> dummy_A(1, 0.0);
    double* d_A = upload_replicated(dummy_A, B);
    double* d_b;
    cudaMalloc(&d_b, sizeof(double) * B);

    // Warmup
    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    std::vector<double> times;
    times.reserve(N_RUNS);
    for (int i = 0; i < N_RUNS; ++i) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Exp K=" << K << " B=" << B << ", " << N_RUNS << " runs (sorted):";
    for (double t : times) std::cout << "  " << t;
    std::cout << "\n";
    std::cout << "  median = " << times[N_RUNS / 2] << " ms\n";
    std::cout << "  min    = " << times[0] << " ms\n";
    std::cout << "  max    = " << times[N_RUNS - 1] << " ms\n";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    return 0;
}
