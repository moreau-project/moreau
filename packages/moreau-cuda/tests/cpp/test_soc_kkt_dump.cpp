/**
 * @file test_soc_kkt_dump.cpp
 * @brief KKT matrix dump test for comparing CPU vs GPU.
 *
 * Identical problem to Rust test: packages/moreau-cpu/tests/soc_kkt_dump.rs
 *
 * Run CUDA with MOREAU_DEBUG=ON cmake:
 *   cmake .. -DCMAKE_CUDA_ARCHITECTURES=120 -DMOREAU_DEBUG=ON
 *   make -j4 test_soc_kkt_dump && ./test_soc_kkt_dump
 *
 * Run Rust:
 *   MOREAU_DEBUG=1 cargo test --test soc_kkt_dump -- --nocapture
 *
 * Compare the KKT dense matrices, RHS, and solutions.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/solver.hpp"
#include "moreau/kkt/kkt_solver.hpp"

using namespace moreau;

namespace {

/// Dump dense KKT matrix from upper-triangle CSR on device
void dumpKKTDense(const CSR& kkt_csr, int64_t n, int64_t m, int64_t p, int64_t batch_idx = 0) {
    int64_t N = kkt_csr.rows();
    int64_t nnz = kkt_csr.nnz();

    std::vector<int64_t> h_ro(N + 1), h_ci(nnz);
    std::vector<double> h_val(nnz);

    kkt_csr.indicesGpuToCpu(h_ro.data(), h_ci.data());
    cudaMemcpy(h_val.data(), kkt_csr.values() + batch_idx * nnz,
               sizeof(double) * nnz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Reconstruct full symmetric dense from upper-triangle CSR
    std::vector<double> dense(N * N, 0.0);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t idx = h_ro[i]; idx < h_ro[i + 1]; idx++) {
            int64_t j = h_ci[idx];
            dense[i * N + j] = h_val[idx];
            dense[j * N + i] = h_val[idx];  // symmetric
        }
    }

    std::cerr << "GPU KKT dense (" << N << "x" << N << "), n=" << n
              << ", m=" << m << ", p=" << p << ":\n";
    for (int64_t i = 0; i < N; i++) {
        std::cerr << "  [";
        for (int64_t j = 0; j < N; j++) {
            if (j > 0) std::cerr << ", ";
            char buf[32];
            snprintf(buf, sizeof(buf), "%+.10e", dense[i * N + j]);
            std::cerr << buf;
        }
        std::cerr << "]\n";
    }
    std::cerr << "GPU KKT diagonal:\n";
    for (int64_t i = 0; i < N; i++) {
        const char* block = (i < n) ? "P" : (i < n + m) ? "H" : "E";
        char buf[32];
        snprintf(buf, sizeof(buf), "%+.10e", dense[i * N + i]);
        std::cerr << "  [" << i << "] (" << block << ") " << buf << "\n";
    }
}

} // namespace

class SOCKKTDumpTest : public ::testing::Test {
protected:
    void SetUp() override { cudaGetLastError(); }
    void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_F(SOCKKTDumpTest, Dim5_DumpKKT) {
    // Same problem as Rust test: P=2I, A=-I, SOC(5)
    int64_t n = 5, m = 5;

    // P = 2*I (diagonal CSR)
    std::vector<int64_t> P_ro = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> P_ci = {0, 1, 2, 3, 4};
    std::vector<double> P_val = {2.0, 2.0, 2.0, 2.0, 2.0};

    // A = -I (diagonal CSR)
    std::vector<int64_t> A_ro = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> A_ci = {0, 1, 2, 3, 4};
    std::vector<double> A_val = {-1.0, -1.0, -1.0, -1.0, -1.0};

    std::vector<double> q = {-3.0, -0.5, -0.3, -0.2, -0.1};
    std::vector<double> b = {0.0, 0.0, 0.0, 0.0, 0.0};

    Cones cones{};
    cones.socConeDims = {5};
    cones.numSocCones = 1;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = true;

    CompiledSolver solver(
        n, m, 1,
        P_ro.data(), P_ci.data(), static_cast<int64_t>(P_val.size()),
        A_ro.data(), A_ci.data(), static_cast<int64_t>(A_val.size()),
        cones, settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * P_val.size());
    cudaMalloc(&d_A, sizeof(double) * A_val.size());
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMemcpy(d_P, P_val.data(), sizeof(double) * P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // Enable per-iteration KKT dump (via MOREAU_DEBUG compile flag)
#ifdef MOREAU_DEBUG
    dynamic_cast<moreau::KKTData&>(*solver.kkt).setDebugSolveResidual(true);
#endif

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    // Also dump final KKT for comparison
    int64_t p = 2 * cones.numSocCones; // sparse expansion for dim=5 > 4
    std::cerr << "\n=== KKT DUMP (after solve, final) ===\n";
    dumpKKTDense(dynamic_cast<moreau::KKTData&>(*solver.kkt).KKT, n, m, p);

    // Print solution
    std::vector<double> x_sol(n), s_sol(m), z_sol(m);
    int32_t status;
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(&status, solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::cerr << "\nGPU Solution:\n";
    std::cerr << "  status = " << status << "\n";
    std::cerr << "  x = [";
    for (int i = 0; i < n; i++) {
        if (i > 0) std::cerr << ", ";
        std::cerr << x_sol[i];
    }
    std::cerr << "]\n";
    std::cerr << "  s = [";
    for (int i = 0; i < m; i++) {
        if (i > 0) std::cerr << ", ";
        std::cerr << s_sol[i];
    }
    std::cerr << "]\n";

    EXPECT_TRUE(status == static_cast<int32_t>(SolverStatus::Solved) ||
                status == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Status = " << status;

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}
