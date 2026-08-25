/**
 * @file test_soc_cudss_debug.cpp
 * @brief Deterministic SOC problem that CPU solves but CUDA fails.
 *
 * Problem: SOC dim=9, n=m=9, dense P=L*L'+0.1*I, ill-conditioned dense A.
 * CPU (faer) solves in 14 iterations with pres ~1e-16.
 * CUDA (cuDSS) gets InsufficientProgress after 13 iterations, pres degrades to ~1e-4.
 *
 * The same hardcoded data is in the Rust test:
 *   packages/moreau-cpu/tests/soc_cudss_debug.rs
 *
 * Build: cmake .. -DCMAKE_CUDA_ARCHITECTURES=120 -DMOREAU_DEBUG=ON
 * Run:   ./test_soc_cudss_debug --gtest_also_run_disabled_tests
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/solver.hpp"

using namespace moreau;

namespace {

// Hardcoded problem data: SOC dim=9, n=m=9
// Generated from: dense P=L*L'+0.1*I, ill-conditioned A, near-boundary s_star
// using std::mt19937_64 seed=1 with the buildHardProblem() construction.
// CPU (faer): Solved in 14 iter
// CUDA (cuDSS): InsufficientProgress after 13 iter

constexpr int64_t DIM = 9;

// P: 9x9 dense symmetric PD (row-major, full symmetric)
// clang-format off
const double P_DATA[81] = {
    0.8813684360892713, -0.0104482831051364, -0.0660172698450806, 0.5139145383631920, -0.1719162764265312, -0.1663217569421306, -0.1077196616316239, -0.6337370330256225, -0.1045334934248386,
    -0.0104482831051364, 1.5546892996210937, -0.2868123763292688, 0.3552867808248623, 0.2463681049267817, -0.6290043492333443, -0.2617468433150026, 0.2631797659538586, 0.0074785154594253,
    -0.0660172698450806, -0.2868123763292688, 1.1299615857865741, -0.0803738336997715, -0.4834730349295611, 0.0820516000911139, -0.0377469979649632, 0.1191651987504730, 0.0344898505263154,
    0.5139145383631920, 0.3552867808248623, -0.0803738336997715, 1.0805146597217179, -0.1787456338861447, -0.0494756609839209, 0.0878229396134882, -0.6287378712764367, -0.0660042882429381,
    -0.1719162764265312, 0.2463681049267817, -0.4834730349295611, -0.1787456338861447, 2.0333217176625835, -0.4099303827535680, 0.4950163861905103, 0.0761777560021472, 0.2672830679032708,
    -0.1663217569421306, -0.6290043492333443, 0.0820516000911139, -0.0494756609839209, -0.4099303827535680, 0.7542148233228343, 0.0680450473258006, 0.0388834083016216, -0.0106697596843358,
    -0.1077196616316239, -0.2617468433150026, -0.0377469979649632, 0.0878229396134882, 0.4950163861905103, 0.0680450473258006, 2.3822851559739484, -0.2948616869697505, 0.0729012045032322,
    -0.6337370330256225, 0.2631797659538586, 0.1191651987504730, -0.6287378712764367, 0.0761777560021472, 0.0388834083016216, -0.2948616869697505, 2.2424393770211410, 0.5900228008748000,
    -0.1045334934248386, 0.0074785154594253, 0.0344898505263154, -0.0660042882429381, 0.2672830679032708, -0.0106697596843358, 0.0729012045032322, 0.5900228008748000, 2.6262115978440099,
};

// A: 9x9 dense (row-major), ill-conditioned rows (scales from 0.01 to 100)
const double A_DATA[81] = {
    0.0042915990266934, -0.0149789690784752, -0.0023738802217544, 0.0025084725067375, -0.0016405009772905, 0.0086506187228476, 0.0015598476884384, 0.0093684041954812, 0.0088584109229293,
    -55.0362508435247122, 246.0171288932346840, 46.1486747278358749, 180.2473702362682104, 125.9233548073076463, 169.5642013396453365, 99.5453630566473890, 84.8492872426583489, 190.6767174325800909,
    -8.0286203749668097, 3.8064165984500202, 8.7267819803647377, -0.2282524012818710, -14.5674321107803664, 11.0584018771880874, 9.0117025885701345, -8.4610421892528311, -5.3960630022683489,
    6.4147495291768664, -0.3596408019570574, 3.9742287581290570, 8.5746965776425839, 10.7170585445202633, 3.9288416329541342, -4.9877018508890503, 7.8957910609474169, -8.4850908806045116,
    -12.0648815582483078, 7.9943454997194170, 13.0381131771062080, 5.6569163620365419, -7.5438261708235421, -0.3471238885941678, -16.4912853697369464, 9.1761766550125525, 9.2194552840640043,
    0.1124683931826382, -0.1019644826808370, -0.1148676522773351, 0.0792100888949301, -0.0008032940689585, 0.2544476033095708, 0.0052965511972236, -0.2957911261052132, -0.3969608433063294,
    -0.1395910694964428, -0.8488564920985243, -1.0667468912270737, -1.1260651481621107, -0.2696435582585219, 0.4097072674830249, 0.3568876635183209, 2.0803531720723636, -0.2796116139788041,
    0.0040802329350315, 0.0098727502211793, -0.0101003921842419, 0.0172838924164012, 0.0002270449254411, -0.0077748160180778, -0.0191513595281752, 0.0163225533746440, 0.0195653061372624,
    -0.0450283107191711, 0.1242498912488013, -0.0468212643469924, -0.0219707932885815, -0.0018708045690442, -0.1418875168020497, -0.0370507924662932, -0.0289737552085187, -0.0144759118179412,
};

const double Q_DATA[9] = {
    1.4603936906103834, -0.6618816801638917, 5.3503060477264466, -1.2531686712828847,
    -1.2592185128487174, 2.1484989088119559, -0.0772294135914003, -0.4666364868170140,
    0.8459788682991369,
};

const double B_DATA[9] = {
    3.7844021014264690, -250.0760239727437124, 39.1605482164058998, -28.3022598876882334,
    -0.6280184107880145, -1.3541396827091523, -4.3942311628569692, 2.2261790625341638,
    0.2373638610394039,
};
// clang-format on

/// Helper: dump a dense matrix from upper-triangular CSR to stdout
void dumpKKTDense(const CSR& kkt_csr, int64_t batch_idx = 0) {
    int64_t N = kkt_csr.rows();
    int64_t nnz = kkt_csr.nnz();

    std::vector<int64_t> ro(N + 1), ci(nnz);
    std::vector<double> vals(nnz);
    kkt_csr.indicesGpuToCpu(ro.data(), ci.data());
    cudaMemcpy(vals.data(), kkt_csr.values() + batch_idx * nnz,
               sizeof(double) * nnz, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Build full symmetric dense matrix
    std::vector<double> dense(N * N, 0.0);
    for (int64_t i = 0; i < N; i++) {
        for (int64_t jj = ro[i]; jj < ro[i + 1]; jj++) {
            int64_t j = ci[jj];
            dense[i * N + j] = vals[jj];
            if (j != i) dense[j * N + i] = vals[jj];  // symmetric fill
        }
    }

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "\nKKT dense (" << N << "x" << N << "):\n";
    for (int64_t i = 0; i < N; i++) {
        for (int64_t j = 0; j < N; j++) {
            double v = dense[i * N + j];
            if (std::abs(v) < 1e-15)
                std::cout << std::setw(12) << ".";
            else
                std::cout << std::setw(12) << v;
        }
        std::cout << "\n";
    }

    // Print diagonal with block labels
    std::cout << "\nKKT diagonal:\n";
    for (int64_t i = 0; i < N; i++) {
        std::cout << "  [" << std::setw(2) << i << "] = " << std::setprecision(8)
                  << dense[i * N + i];
        if (i < DIM) std::cout << "  (P block)";
        else if (i < 2 * DIM) std::cout << "  (H/cone block, row " << (i - DIM) << ")";
        else std::cout << "  (expansion row " << (i - 2 * DIM) << ")";
        std::cout << "\n";
    }

    // Print condition number estimate via diagonal ratio
    double min_abs_diag = 1e300, max_abs_diag = 0;
    for (int64_t i = 0; i < N; i++) {
        double ad = std::abs(dense[i * N + i]);
        if (ad > 0) {
            if (ad < min_abs_diag) min_abs_diag = ad;
            if (ad > max_abs_diag) max_abs_diag = ad;
        }
    }
    std::cout << "\nDiagonal condition: max/min = " << std::setprecision(4)
              << max_abs_diag / min_abs_diag << "\n";

    // Print expansion rows detail
    if (N > 2 * DIM) {
        std::cout << "\nExpansion rows (last " << (N - 2 * DIM) << "):\n";
        for (int64_t i = 2 * DIM; i < N; i++) {
            std::cout << "  Row " << i << ": ";
            for (int64_t jj = ro[i]; jj < ro[i + 1]; jj++)
                std::cout << "(" << ci[jj] << ":" << std::setprecision(6) << vals[jj] << ") ";
            std::cout << "\n";
        }
    }

    // Print u/v column entries (columns n+m and n+m+1)
    if (N > 2 * DIM) {
        std::cout << "\nExpansion column entries (v-col=" << 2*DIM << ", u-col=" << 2*DIM+1 << "):\n";
        for (int64_t i = DIM; i < 2 * DIM; i++) {
            std::cout << "  row " << i << ": v=" << std::setprecision(6) << dense[i * N + 2*DIM]
                      << "  u=" << dense[i * N + 2*DIM + 1] << "\n";
        }
    }
}

/// Helper: manually compute KKT * x and check residual against rhs
/// KKT is stored as upper-triangle CSR; we compute full symmetric product
void checkKKTResidual(const CSR& kkt_csr, const double* d_rhs, const double* d_sol,
                      const char* label, int64_t batch_idx = 0) {
    int64_t N = kkt_csr.rows();
    int64_t nnz = kkt_csr.nnz();

    std::vector<int64_t> ro(N + 1), ci(nnz);
    std::vector<double> vals(nnz), rhs(N), sol(N);
    kkt_csr.indicesGpuToCpu(ro.data(), ci.data());
    cudaMemcpy(vals.data(), kkt_csr.values() + batch_idx * nnz,
               sizeof(double) * nnz, cudaMemcpyDeviceToHost);
    cudaMemcpy(rhs.data(), d_rhs + batch_idx * N, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(sol.data(), d_sol + batch_idx * N, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // y = KKT_full * sol
    std::vector<double> y(N, 0.0);
    for (int64_t i = 0; i < N; ++i) {
        for (int64_t jj = ro[i]; jj < ro[i + 1]; ++jj) {
            int64_t j = ci[jj];
            y[i] += vals[jj] * sol[j];
            if (j != i) y[j] += vals[jj] * sol[i];
        }
    }

    // Compute residual
    double max_res = 0, max_rhs = 0;
    std::vector<double> res(N);
    for (int64_t i = 0; i < N; ++i) {
        res[i] = y[i] - rhs[i];
        max_res = std::max(max_res, std::abs(res[i]));
        max_rhs = std::max(max_rhs, std::abs(rhs[i]));
    }

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "[" << label << "] ||KKT*sol-rhs||_inf = " << max_res
              << "  relative = " << (max_rhs > 0 ? max_res / max_rhs : max_res) << "\n";

    // Per-row residuals (show worst 5)
    std::vector<std::pair<double, int64_t>> sorted_res(N);
    for (int64_t i = 0; i < N; ++i) sorted_res[i] = {std::abs(res[i]), i};
    std::sort(sorted_res.rbegin(), sorted_res.rend());

    std::cout << "  Worst residual rows:\n";
    for (int k = 0; k < std::min((int64_t)5, N); ++k) {
        auto [r, idx] = sorted_res[k];
        std::cout << "    row " << idx << ": res=" << r
                  << "  rhs=" << rhs[idx] << "  y=" << y[idx];
        if (idx < DIM) std::cout << " (P)";
        else if (idx < 2*DIM) std::cout << " (H, cone " << (idx-DIM) << ")";
        else std::cout << " (exp " << (idx-2*DIM) << ")";
        std::cout << "\n";
    }

    // Print solution vector
    std::cout << "  Solution: [";
    for (int64_t i = 0; i < N; ++i) {
        std::cout << std::setprecision(6) << sol[i];
        if (i < N-1) std::cout << ", ";
    }
    std::cout << "]\n";
}

} // namespace

/**
 * Previously DISABLED because CUDA's cuDSS factorization produced inaccurate
 * KKT solve directions for this ill-conditioned SOC(dim=9) problem (sparse
 * expansion path) where CPU's faer solver succeeded. The bug appears to have
 * been fixed since — this test now reliably lands in Solved/AlmostSolved on
 * T500 (sm_75). Re-enabled so it guards against regression.
 *
 * Compile with -DMOREAU_DEBUG=ON for per-iteration residual tracking.
 */
TEST(SOCCuDSSDebug, Dim9_CpuSolves_CudaFails) {
    const int64_t n = DIM, m = DIM;
    const int64_t nnzP = n * n, nnzA = m * n;

    // Build dense CSR for P (full symmetric)
    std::vector<int64_t> P_ro(n + 1), P_ci(nnzP);
    for (int64_t i = 0; i <= n; i++) P_ro[i] = i * n;
    for (int64_t i = 0; i < nnzP; i++) P_ci[i] = i % n;

    // Build dense CSR for A
    std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
    for (int64_t i = 0; i <= m; i++) A_ro[i] = i * n;
    for (int64_t i = 0; i < nnzA; i++) A_ci[i] = i % n;

    Cones cones{};
    cones.socConeDims = {DIM};
    cones.numSocCones = 1;

    Settings settings;
    settings.maxIter = 200;
    settings.verbose = true;

    CompiledSolver solver(n, m, 1,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings);

#ifdef MOREAU_DEBUG
    // Enable per-solve residual computation
    dynamic_cast<moreau::KKTData&>(*solver.kkt).setDebugSolveResidual(true);
#endif

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMemcpy(d_P, P_DATA, sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_DATA, sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, Q_DATA, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B_DATA, sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    // Dump final KKT matrix
    std::cout << "\n====== FINAL KKT MATRIX ======\n";
    dumpKKTDense(dynamic_cast<moreau::KKTData&>(*solver.kkt).KKT);

    // Get result
    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << "\nFinal status: " << status[0] << "\n";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);

    // This is expected to fail on CUDA — that's the point of the test
    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "CUDA status=" << status[0]
        << " (CPU solves this with faer — this is a cuDSS limitation)";
}

/**
 * Standalone KKT residual test: construct a simple KKT system manually,
 * factor+solve with cuDSS, and check the residual directly.
 * This isolates the cuDSS accuracy problem from the full solver.
 */
TEST(SOCCuDSSDebug, DISABLED_StandaloneKKTResidual) {
    const int64_t n = DIM, m = DIM;
    const int64_t nnzP = n * n, nnzA = m * n;

    // Build dense CSR for P and A
    std::vector<int64_t> P_ro(n + 1), P_ci(nnzP);
    for (int64_t i = 0; i <= n; i++) P_ro[i] = i * n;
    for (int64_t i = 0; i < nnzP; i++) P_ci[i] = i % n;

    std::vector<int64_t> A_ro(m + 1), A_ci(nnzA);
    for (int64_t i = 0; i <= m; i++) A_ro[i] = i * n;
    for (int64_t i = 0; i < nnzA; i++) A_ci[i] = i % n;

    Cones cones{};
    cones.socConeDims = {DIM};
    cones.numSocCones = 1;

    Settings settings;
    settings.maxIter = 1;  // Only 1 iteration to capture first KKT solve
    settings.verbose = true;

    CompiledSolver solver(n, m, 1,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings);

#ifdef MOREAU_DEBUG
    dynamic_cast<moreau::KKTData&>(*solver.kkt).setDebugSolveResidual(true);
#endif

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMemcpy(d_P, P_DATA, sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_DATA, sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, Q_DATA, sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, B_DATA, sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::cout << "\n====== KKT AFTER 1 ITERATION ======\n";
    dumpKKTDense(dynamic_cast<moreau::KKTData&>(*solver.kkt).KKT);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}
