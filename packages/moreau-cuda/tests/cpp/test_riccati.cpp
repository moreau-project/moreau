/**
 * @file test_riccati.cpp
 * @brief Tests for batched block-tridiagonal (Riccati) KKT solver
 *
 * Constructs MPC-structured QPs and verifies:
 * 1. Block-tridiagonal structure detection
 * 2. Riccati solver produces correct results (vs cuDSS reference)
 * 3. Benchmark: Riccati vs cuDSS performance
 */

#include <gtest/gtest.h>
#include "moreau/solver/solver.hpp"
#include "moreau/kkt/riccati.hpp"
#include <vector>
#include <cmath>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <algorithm>
#include <utility>
#include <array>

using namespace moreau;

// ============================================================================
// Helper: build MPC-structured QP
// ============================================================================

struct MPCProblem {
    int64_t n, m;
    int64_t nnzP, nnzA;
    std::vector<int64_t> P_ro, P_ci;
    std::vector<int64_t> A_ro, A_ci;
    std::vector<double> P_values;
    std::vector<double> A_values;
    std::vector<double> q_values;
    std::vector<double> b_values;
    Cones cones;
};

/**
 * Build an MPC QP: T stages, nx state dim, nu control dim
 *
 * Variables: [x0, u0, x1, u1, ..., x_{T-1}, u_{T-1}, x_T]
 * Total n = T*(nx+nu) + nx
 *
 * Cost: sum_t (x_t'Qx_t + u_t'Ru_t) + x_T'Qf x_T
 * P = blockdiag(Q, R, Q, R, ..., Qf)  (diagonal, upper triangle only)
 *
 * Constraints:
 * - Dynamics: x_{t+1} = A_dyn*x_t + B_dyn*u_t  ->  [A_dyn, B_dyn, -I] * [x_t; u_t; x_{t+1}] = 0
 *   These are T*nx zero cone constraints
 * - Bounds: u_t >= lb, u_t <= ub  ->  u_t >= lb, -u_t >= -ub
 *   These are 2*T*nu nonneg cone constraints
 *
 * Total m = T*nx (dynamics, zero cones) + 2*T*nu (bounds, nonneg cones)
 */
MPCProblem buildMPCProblem(int T, int nx, int nu, int batchSize = 1, bool equality_only = false,
                           double R_scale = 0.01, double Qf_scale = 10.0) {
    MPCProblem prob;

    int n_total = T * (nx + nu) + nx;
    int m_dynamics = T * nx;
    int m_bounds = equality_only ? 0 : 2 * T * nu;
    int m_total = m_dynamics + m_bounds;

    prob.n = n_total;
    prob.m = m_total;

    // P: diagonal only (upper triangle)
    // Variables layout: x0(nx), u0(nu), x1(nx), u1(nu), ..., xT(nx)
    prob.P_ro.resize(n_total + 1);
    prob.P_ci.resize(n_total);
    prob.P_values.resize(n_total);

    for (int i = 0; i <= n_total; ++i) prob.P_ro[i] = i;
    for (int i = 0; i < n_total; ++i) prob.P_ci[i] = i;

    // Fill P values: Q for states, R for controls
    for (int t = 0; t < T; ++t) {
        int x_offset = t * (nx + nu);
        int u_offset = x_offset + nx;
        for (int i = 0; i < nx; ++i) prob.P_values[x_offset + i] = 1.0; // Q = I
        for (int i = 0; i < nu; ++i) prob.P_values[u_offset + i] = R_scale; // R = R_scale*I
    }
    // Terminal cost
    int xT_offset = T * (nx + nu);
    for (int i = 0; i < nx; ++i) prob.P_values[xT_offset + i] = Qf_scale; // Qf = Qf_scale*I

    prob.nnzP = n_total;

    // A: dynamics + bounds
    // Row layout: [dynamics(T*nx); bounds(2*T*nu)]
    //
    // Dynamics row for constraint t, state component k:
    //   A_dyn[k,:]*x_t + B_dyn[k,:]*u_t - x_{t+1}[k] = 0
    //   Nonzeros in columns: x_t[0..nx-1], u_t[0..nu-1], x_{t+1}[k]
    //   That's nx + nu + 1 nonzeros per row
    //
    // Bounds:
    //   u_t[j] >= lb  ->  row with +1 at u_t[j]  (nonneg cone)
    //   -u_t[j] >= -ub  ->  row with -1 at u_t[j]  (nonneg cone)

    int nnz_dynamics = m_dynamics * (nx + nu + 1);
    int nnz_bounds = m_bounds;  // 1 nonzero per bound constraint
    int nnzA_total = nnz_dynamics + nnz_bounds;

    prob.A_ro.resize(m_total + 1);
    prob.A_ci.resize(nnzA_total);
    prob.A_values.resize(nnzA_total);

    int row = 0;
    int aidx = 0;
    prob.A_ro[0] = 0;

    // Dynamics constraints (zero cones)
    for (int t = 0; t < T; ++t) {
        int x_offset = t * (nx + nu);
        int u_offset = x_offset + nx;
        int x_next_offset = (t + 1) * (nx + nu);
        if (t == T - 1) x_next_offset = T * (nx + nu);

        for (int k = 0; k < nx; ++k) {
            // A_dyn[k,:] * x_t  (simple: A_dyn = 0.9*I + 0.1*shift)
            for (int j = 0; j < nx; ++j) {
                prob.A_ci[aidx] = x_offset + j;
                double val = (j == k) ? 0.9 : ((j == (k + 1) % nx) ? 0.1 : 0.0);
                prob.A_values[aidx] = val;
                if (val != 0.0) ++aidx;
                else {
                    // Only include nonzero entries
                    // Actually for simplicity, let's use a dense A_dyn first
                    // but that means nnz per row = nx + nu + 1 which includes zeros.
                    // Let's simplify: A_dyn = I, B_dyn = I (truncated)
                }
            }
            // B_dyn[k,:] * u_t
            for (int j = 0; j < nu; ++j) {
                prob.A_ci[aidx] = u_offset + j;
                prob.A_values[aidx] = (j == k && k < nu) ? 1.0 : 0.0;
                ++aidx;
            }
            // -I * x_{t+1}[k]
            prob.A_ci[aidx] = x_next_offset + k;
            prob.A_values[aidx] = -1.0;
            ++aidx;

            prob.A_ro[row + 1] = aidx;
            ++row;
        }
    }

    // Wait, this includes zero entries. Let me redo with sparse construction.
    // Reset and do it properly.

    prob.A_ro.clear();
    prob.A_ci.clear();
    prob.A_values.clear();

    prob.A_ro.push_back(0);
    row = 0;

    // Dynamics: A_dyn = I (nx x nx), B_dyn = I_trunc (nx x nu)
    // x_{t+1} = x_t + u_t (for components 0..min(nx,nu)-1)
    // x_{t+1}[k] = x_t[k] + (k < nu ? u_t[k] : 0) for k=0..nx-1
    for (int t = 0; t < T; ++t) {
        int x_off = t * (nx + nu);
        int u_off = x_off + nx;
        int x_next_off;
        if (t < T - 1) x_next_off = (t + 1) * (nx + nu);
        else x_next_off = T * (nx + nu);

        for (int k = 0; k < nx; ++k) {
            // A_dyn: x_t[k] coefficient = 1.0
            prob.A_ci.push_back(x_off + k);
            prob.A_values.push_back(1.0);

            // B_dyn: u_t[k] coefficient = 1.0 if k < nu
            if (k < nu) {
                prob.A_ci.push_back(u_off + k);
                prob.A_values.push_back(1.0);
            }

            // -I: x_{t+1}[k] coefficient = -1.0
            prob.A_ci.push_back(x_next_off + k);
            prob.A_values.push_back(-1.0);

            prob.A_ro.push_back((int64_t)prob.A_ci.size());
        }
    }

    // Bounds: u_t >= -1, u_t <= 1
    if (!equality_only) {
        for (int t = 0; t < T; ++t) {
            int u_off = t * (nx + nu) + nx;
            for (int j = 0; j < nu; ++j) {
                prob.A_ci.push_back(u_off + j);
                prob.A_values.push_back(1.0);
                prob.A_ro.push_back((int64_t)prob.A_ci.size());
            }
            for (int j = 0; j < nu; ++j) {
                prob.A_ci.push_back(u_off + j);
                prob.A_values.push_back(-1.0);
                prob.A_ro.push_back((int64_t)prob.A_ci.size());
            }
        }
    }

    prob.nnzA = (int64_t)prob.A_ci.size();

    // Cones: zero (dynamics) + nonneg (bounds)
    prob.cones = Cones{};
    prob.cones.numZeroCones = m_dynamics;
    prob.cones.numNonnegCones = m_bounds;
    prob.cones.initialize(batchSize, 0);

    // q and b vectors
    prob.q_values.resize(n_total, 0.0);
    prob.b_values.resize(m_total, 0.0);
    // b for bounds: b_lower = 1 (representing lb=-1), b_upper = 1 (representing ub=1)
    for (int i = m_dynamics; i < m_total; ++i) {
        prob.b_values[i] = 1.0;
    }

    return prob;
}

// ============================================================================
// Tests
// ============================================================================

TEST(RiccatiTest, DetectBlockTridiagonal) {
    auto prob = buildMPCProblem(10, 4, 2);

    auto blocks = detect_block_tridiagonal(
        prob.n, prob.m,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones);

    ASSERT_FALSE(blocks.empty()) << "Should detect block-tridiagonal structure for MPC QP";

    // Print detected blocks
    int total = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
        total += blocks[i];
    }
    EXPECT_EQ(total, (int)prob.n);

    std::cout << "Detected " << blocks.size() << " blocks: ";
    for (size_t i = 0; i < std::min(blocks.size(), (size_t)10); ++i)
        std::cout << blocks[i] << " ";
    if (blocks.size() > 10) std::cout << "...";
    std::cout << std::endl;
}

// A dense row of A (e.g. a portfolio sum-to-one or factor-model row) must bail
// to cuDSS without materializing the O(nnz_r^2) A'A clique that would otherwise
// blow up host memory. Here the single all-ones equality row couples every
// column, so detection must return empty cheaply.
TEST(RiccatiTest, DenseRowBailsToCuDSS) {
    const int64_t n = 2000;

    // P = I (diagonal, upper triangle).
    std::vector<int64_t> P_ro(n + 1), P_ci(n);
    std::vector<int64_t> A_ro, A_ci;
    for (int64_t i = 0; i < n; ++i) { P_ro[i] = i; P_ci[i] = i; }
    P_ro[n] = n;

    // Row 0: dense all-ones equality row (zero cone). Rows 1..n: -e_i (nonneg).
    A_ro.push_back(0);
    for (int64_t j = 0; j < n; ++j) A_ci.push_back(j);
    A_ro.push_back((int64_t)A_ci.size());
    for (int64_t i = 0; i < n; ++i) {
        A_ci.push_back(i);
        A_ro.push_back((int64_t)A_ci.size());
    }

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = n;

    auto blocks = detect_block_tridiagonal(
        n, (int64_t)A_ro.size() - 1,
        P_ro.data(), P_ci.data(), n,
        A_ro.data(), A_ci.data(), (int64_t)A_ci.size(),
        cones);

    EXPECT_TRUE(blocks.empty()) << "Dense row must not be detected as block-tridiagonal";
}

TEST(RiccatiTest, AssemblyCheck) {
    // Small problem to manually verify block assembly
    int T = 3, nx = 2, nu = 1;
    int batchSize = 1;
    auto prob = buildMPCProblem(T, nx, nu, batchSize);

    auto blocks = detect_block_tridiagonal(
        prob.n, prob.m,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones);

    std::cout << "n=" << prob.n << " m=" << prob.m
              << " nnzP=" << prob.nnzP << " nnzA=" << prob.nnzA << std::endl;
    ASSERT_FALSE(blocks.empty());

    std::cout << "Blocks (" << blocks.size() << "): ";
    for (auto b : blocks) std::cout << b << " ";
    std::cout << std::endl;

    // Print P structure
    std::cout << "P (CSR upper):" << std::endl;
    for (int64_t i = 0; i < prob.n; ++i) {
        for (int64_t p = prob.P_ro[i]; p < prob.P_ro[i+1]; ++p) {
            std::cout << "  P[" << i << "," << prob.P_ci[p] << "] = " << prob.P_values[p] << std::endl;
        }
    }

    // Print A structure (first few rows)
    std::cout << "A (CSR):" << std::endl;
    for (int64_t r = 0; r < std::min(prob.m, (int64_t)10); ++r) {
        std::cout << "  row " << r << ": ";
        for (int64_t p = prob.A_ro[r]; p < prob.A_ro[r+1]; ++p) {
            std::cout << "(" << prob.A_ci[p] << "," << prob.A_values[p] << ") ";
        }
        std::cout << std::endl;
    }

    // Try to construct the Riccati solver and run factorization
    Settings settings;
    settings.verbose = true;
    settings.maxIter = 5;
    settings.ipm.kktSolverType = KKTSolverType::Riccati;

    try {
        CompiledSolver solver(
            prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, settings);

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * prob.nnzP);
        cudaMalloc(&d_A, sizeof(double) * prob.nnzA);
        cudaMalloc(&d_q, sizeof(double) * prob.n);
        cudaMalloc(&d_b, sizeof(double) * prob.m);
        cudaMemcpy(d_P, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, prob.A_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, prob.q_values.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, prob.b_values.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

        solver.solveAll(d_P, d_A, d_q, d_b);
        std::cout << "Riccati solved successfully! iter=" << solver.solution.iterations << std::endl;

        // Verify solve status
        int32_t h_status;
        cudaMemcpy(&h_status, solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
        EXPECT_TRUE(h_status == (int32_t)SolverStatus::Solved ||
                    h_status == (int32_t)SolverStatus::AlmostSolved)
            << "Expected Solved or AlmostSolved, got status " << h_status;

        // Verify solution is finite
        std::vector<double> x(prob.n);
        cudaMemcpy(x.data(), solver.solution.x.data(),
                   sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
        for (int64_t i = 0; i < prob.n; ++i) {
            EXPECT_TRUE(std::isfinite(x[i]))
                << "x[" << i << "] = " << x[i] << " is not finite";
        }

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    } catch (const std::exception& e) {
        FAIL() << "Riccati assembly/solve threw exception: " << e.what();
    }
}

TEST(RiccatiTest, SolveCorrectness) {
    int T = 10, nx = 4, nu = 2;
    int batchSize = 1;
    auto prob = buildMPCProblem(T, nx, nu, batchSize);

    // Solve with cuDSS (reference)
    Settings settings_cudss;
    settings_cudss.verbose = false;
    settings_cudss.maxIter = 100;
    settings_cudss.ipm.kktSolverType = KKTSolverType::CuDSS;

    CompiledSolver solver_cudss(
        prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings_cudss);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * prob.n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * prob.m * batchSize);

    cudaMemcpy(d_P, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q_values.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b_values.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

    solver_cudss.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_cudss(prob.n * batchSize);
    cudaMemcpy(x_cudss.data(), solver_cudss.solution.x.data(),
               sizeof(double) * prob.n * batchSize, cudaMemcpyDeviceToHost);

    std::cout << "cuDSS: iter=" << solver_cudss.solution.iterations << std::endl;

    // Solve with Riccati
    Settings settings_riccati;
    settings_riccati.verbose = false;
    settings_riccati.maxIter = 100;
    settings_riccati.ipm.kktSolverType = KKTSolverType::Riccati;

    CompiledSolver solver_riccati(
        prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings_riccati);

    solver_riccati.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_riccati(prob.n * batchSize);
    cudaMemcpy(x_riccati.data(), solver_riccati.solution.x.data(),
               sizeof(double) * prob.n * batchSize, cudaMemcpyDeviceToHost);

    std::cout << "Riccati: iter=" << solver_riccati.solution.iterations << std::endl;

    // Compare solutions
    double max_diff = 0.0;
    for (size_t i = 0; i < x_cudss.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(x_cudss[i] - x_riccati[i]));
    }
    std::cout << "Max x diff: " << max_diff << std::endl;

    // They should agree to reasonable tolerance
    EXPECT_LT(max_diff, 1e-4) << "Riccati and cuDSS solutions should match";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// ============================================================================
// Permutation recovery: a scrambled MPC variable order is not block-tridiagonal
// as given, but an internal RCM reorder recovers the band so Riccati still
// applies. These exercise the permuted detection + the x-vector gather/scatter.
// ============================================================================

// Apply a column permutation to an MPC QP. `old_to_new[i]` is the new index of
// original variable i. P is permuted symmetrically, A's columns are permuted,
// and q is permuted; rows, b, and cones are unchanged. CSR columns are
// re-sorted within each row. The solution maps as x_new[old_to_new[i]] = x[i].
// `p` is taken by value and modified in place (Cones is move-only).
static MPCProblem permuteMPCProblem(MPCProblem p, const std::vector<int>& old_to_new) {
    int n = (int)p.n, m = (int)p.m;

    // q: q_new[old_to_new[i]] = q[i]
    std::vector<double> q_new(n, 0.0);
    for (int i = 0; i < n; ++i) q_new[old_to_new[i]] = p.q_values[i];
    p.q_values = std::move(q_new);

    // P: symmetric row + column permute.
    std::vector<std::vector<std::pair<int, double>>> prows(n);
    for (int i = 0; i < n; ++i)
        for (int64_t pp = p.P_ro[i]; pp < p.P_ro[i + 1]; ++pp)
            prows[old_to_new[i]].push_back({old_to_new[(int)p.P_ci[pp]], p.P_values[pp]});
    std::vector<int64_t> P_ro(n + 1, 0), P_ci;
    std::vector<double> P_vals;
    for (int i = 0; i < n; ++i) {
        std::sort(prows[i].begin(), prows[i].end());
        P_ro[i + 1] = P_ro[i] + (int64_t)prows[i].size();
    }
    for (int i = 0; i < n; ++i)
        for (auto& e : prows[i]) { P_ci.push_back(e.first); P_vals.push_back(e.second); }
    p.P_ro = std::move(P_ro); p.P_ci = std::move(P_ci); p.P_values = std::move(P_vals);

    // A: column permute (rows unchanged), re-sort columns within each row.
    std::vector<int64_t> A_ci(p.A_ci.size());
    std::vector<double> A_vals(p.A_values.size());
    for (int r = 0; r < m; ++r) {
        std::vector<std::pair<int, double>> entries;
        for (int64_t pp = p.A_ro[r]; pp < p.A_ro[r + 1]; ++pp)
            entries.push_back({old_to_new[(int)p.A_ci[pp]], p.A_values[pp]});
        std::sort(entries.begin(), entries.end());
        int64_t pos = p.A_ro[r];
        for (auto& e : entries) { A_ci[pos] = e.first; A_vals[pos] = e.second; ++pos; }
    }
    p.A_ci = std::move(A_ci); p.A_values = std::move(A_vals);

    return p;
}

// A deterministic shuffle (valid permutation for any n) that destroys the MPC
// locality, so the scrambled problem is not banded in its given column order.
static std::vector<int> scrambleMap(int n, uint32_t seed) {
    std::vector<int> map(n);
    std::iota(map.begin(), map.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(map.begin(), map.end(), rng);
    return map;
}

TEST(RiccatiTest, DetectBlockTridiagonalPermuted) {
    // n > 2*128 so a scrambled dynamics row spans beyond the fixed-order
    // bandwidth bound — exercising the permutation path's relaxed pre-filter.
    int T = 52, nx = 3, nu = 2;
    auto base = buildMPCProblem(T, nx, nu);
    ASSERT_GT(base.n, 2 * 128);

    auto map = scrambleMap((int)base.n, /*seed=*/12345u);
    auto prob = permuteMPCProblem(std::move(base), map);

    // "smem-usable" = a non-empty decomposition whose blocks fit the Riccati
    // shared-memory bound (anything larger is rejected and routed to cuDSS).
    auto fits = [](const std::vector<int32_t>& b) {
        return !b.empty() &&
               *std::max_element(b.begin(), b.end()) <= kRiccatiMaxSmemBlock;
    };

    // Given (scrambled) order: no usable block-tridiagonal decomposition (the
    // scattered coupling forces one oversized block).
    auto blocks_fixed = detect_block_tridiagonal(
        prob.n, prob.m,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, /*allow_permute=*/false, nullptr);
    EXPECT_FALSE(fits(blocks_fixed))
        << "Scrambled MPC must not be Riccati-usable in the given order";

    // With an RCM reorder, a usable banded block structure is recovered.
    std::vector<int32_t> perm;
    auto blocks_perm = detect_block_tridiagonal(
        prob.n, prob.m,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, /*allow_permute=*/true, &perm);

    ASSERT_TRUE(fits(blocks_perm)) << "RCM should recover a smem-fitting block structure";
    ASSERT_EQ((int64_t)perm.size(), prob.n) << "A non-identity column order must be returned";

    int total = 0;
    for (auto b : blocks_perm) total += b;
    EXPECT_EQ(total, (int)prob.n);

    // perm must be a valid permutation of [0, n).
    std::vector<char> seen(prob.n, 0);
    for (int32_t v : perm) {
        ASSERT_GE(v, 0); ASSERT_LT(v, (int32_t)prob.n);
        EXPECT_FALSE(seen[v]) << "perm has a repeated index";
        seen[v] = 1;
    }

    int32_t max_blk = *std::max_element(blocks_perm.begin(), blocks_perm.end());
    std::cout << "Recovered " << blocks_perm.size() << " blocks, max block "
              << max_blk << std::endl;
}

TEST(RiccatiTest, SolveCorrectnessPermuted) {
    // n > 2*128 so the scrambled order forces an oversized block (not
    // Riccati-usable), making the internal RCM reorder the only way the
    // explicit-Riccati solve below can succeed.
    int T = 52, nx = 3, nu = 2;
    int batchSize = 1;
    auto base = buildMPCProblem(T, nx, nu, batchSize);
    ASSERT_GT(base.n, 2 * 128);
    auto map = scrambleMap((int)base.n, /*seed=*/777u);
    auto prob = permuteMPCProblem(std::move(base), map);

    // The scrambled problem is not block-tridiagonal as given, so the Riccati
    // path below can only succeed via the solver's internal RCM reorder.
    auto blocks_fixed = detect_block_tridiagonal(
        prob.n, prob.m,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, /*allow_permute=*/false, nullptr);
    bool fixed_usable = !blocks_fixed.empty() &&
        *std::max_element(blocks_fixed.begin(), blocks_fixed.end()) <= kRiccatiMaxSmemBlock;
    ASSERT_FALSE(fixed_usable) << "scrambled MPC should not be Riccati-usable as given";

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * prob.n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * prob.m * batchSize);
    cudaMemcpy(d_P, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q_values.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b_values.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

    // cuDSS reference (order-agnostic).
    Settings settings_cudss;
    settings_cudss.verbose = false;
    settings_cudss.maxIter = 100;
    settings_cudss.ipm.kktSolverType = KKTSolverType::CuDSS;
    CompiledSolver solver_cudss(
        prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings_cudss);
    solver_cudss.solveAll(d_P, d_A, d_q, d_b);
    std::vector<double> x_cudss(prob.n * batchSize);
    cudaMemcpy(x_cudss.data(), solver_cudss.solution.x.data(),
               sizeof(double) * prob.n * batchSize, cudaMemcpyDeviceToHost);

    // Explicit Riccati: grad is off, so the factory recovers the band via RCM.
    // (If the reorder did not fire, the explicit request would throw.)
    Settings settings_riccati;
    settings_riccati.verbose = false;
    settings_riccati.maxIter = 100;
    settings_riccati.ipm.kktSolverType = KKTSolverType::Riccati;
    CompiledSolver solver_riccati(
        prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings_riccati);
    solver_riccati.solveAll(d_P, d_A, d_q, d_b);
    std::vector<double> x_riccati(prob.n * batchSize);
    cudaMemcpy(x_riccati.data(), solver_riccati.solution.x.data(),
               sizeof(double) * prob.n * batchSize, cudaMemcpyDeviceToHost);

    double max_diff = 0.0;
    for (size_t i = 0; i < x_cudss.size(); ++i)
        max_diff = std::max(max_diff, std::abs(x_cudss[i] - x_riccati[i]));
    std::cout << "Permuted MPC: cuDSS iter=" << solver_cudss.solution.iterations
              << " Riccati iter=" << solver_riccati.solution.iterations
              << " max x diff=" << max_diff << std::endl;

    EXPECT_LT(max_diff, 1e-4)
        << "RCM-reordered Riccati must match cuDSS on a scrambled MPC";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

TEST(RiccatiTest, SolveCorrectnessBatched) {
    int T = 10, nx = 4, nu = 2;
    int batchSize = 8;
    auto prob = buildMPCProblem(T, nx, nu, batchSize);

    // Replicate problem data for batch
    std::vector<double> P_batch(prob.nnzP * batchSize);
    std::vector<double> A_batch(prob.nnzA * batchSize);
    std::vector<double> q_batch(prob.n * batchSize);
    std::vector<double> b_batch(prob.m * batchSize);
    for (int b = 0; b < batchSize; ++b) {
        std::copy(prob.P_values.begin(), prob.P_values.end(), P_batch.begin() + b * prob.nnzP);
        std::copy(prob.A_values.begin(), prob.A_values.end(), A_batch.begin() + b * prob.nnzA);
        std::copy(prob.q_values.begin(), prob.q_values.end(), q_batch.begin() + b * prob.n);
        std::copy(prob.b_values.begin(), prob.b_values.end(), b_batch.begin() + b * prob.m);
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * prob.n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * prob.m * batchSize);

    cudaMemcpy(d_P, P_batch.data(), sizeof(double) * prob.nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_batch.data(), sizeof(double) * prob.nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_batch.data(), sizeof(double) * prob.n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_batch.data(), sizeof(double) * prob.m * batchSize, cudaMemcpyHostToDevice);

    // cuDSS reference
    Settings settings_cudss;
    settings_cudss.verbose = false;
    settings_cudss.maxIter = 100;
    settings_cudss.ipm.kktSolverType = KKTSolverType::CuDSS;

    CompiledSolver solver_cudss(
        prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings_cudss);

    solver_cudss.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_cudss(prob.n * batchSize);
    cudaMemcpy(x_cudss.data(), solver_cudss.solution.x.data(),
               sizeof(double) * prob.n * batchSize, cudaMemcpyDeviceToHost);

    // Riccati
    Settings settings_riccati;
    settings_riccati.verbose = false;
    settings_riccati.maxIter = 100;
    settings_riccati.ipm.kktSolverType = KKTSolverType::Riccati;

    CompiledSolver solver_riccati(
        prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings_riccati);

    solver_riccati.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_riccati(prob.n * batchSize);
    cudaMemcpy(x_riccati.data(), solver_riccati.solution.x.data(),
               sizeof(double) * prob.n * batchSize, cudaMemcpyDeviceToHost);

    std::cout << "cuDSS iter=" << solver_cudss.solution.iterations
              << ", Riccati iter=" << solver_riccati.solution.iterations << std::endl;

    // Compare all batch elements
    double max_diff = 0.0;
    for (size_t i = 0; i < x_cudss.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(x_cudss[i] - x_riccati[i]));
    }
    std::cout << "Max x diff (B=" << batchSize << "): " << max_diff << std::endl;

    EXPECT_LT(max_diff, 1e-4) << "Riccati and cuDSS batch solutions should match";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

TEST(RiccatiTest, DISABLED_Benchmark) {
    struct Config {
        int T, nx, nu, batchSize;
        bool equality_only;
        std::string label;
    };

    std::vector<Config> configs = {
        // smem path (block size <= 48)
        {50, 12,  6,   1,  true,  "T=50  nx=12  nu=6  B=1    eq"},
        {50, 12,  6,  64,  true,  "T=50  nx=12  nu=6  B=64   eq"},
        {50, 12,  6, 512,  true,  "T=50  nx=12  nu=6  B=512  eq"},
        {50, 12,  6,2048,  true,  "T=50  nx=12  nu=6  B=2048 eq"},
        {50, 12,  4,   1,  false, "T=50  nx=12  nu=4  B=1"},
        {50, 12,  4, 512,  false, "T=50  nx=12  nu=4  B=512"},
        {20,  8,  4, 512,  false, "T=20  nx=8   nu=4  B=512"},
    };

    std::cout << "\n=== Riccati vs cuDSS Benchmark ===" << std::endl;
    std::cout << std::string(90, '-') << std::endl;
    printf("%-30s %8s %8s %10s %10s %8s\n",
           "Config", "n", "m", "cuDSS(ms)", "Ricc(ms)", "Speedup");
    std::cout << std::string(90, '-') << std::endl;

    for (auto& cfg : configs) {
        auto prob = buildMPCProblem(cfg.T, cfg.nx, cfg.nu, cfg.batchSize, cfg.equality_only);

        // Replicate problem data for batch
        std::vector<double> P_batch(prob.nnzP * cfg.batchSize);
        std::vector<double> A_batch(prob.nnzA * cfg.batchSize);
        std::vector<double> q_batch(prob.n * cfg.batchSize);
        std::vector<double> b_batch(prob.m * cfg.batchSize);
        for (int b = 0; b < cfg.batchSize; ++b) {
            std::copy(prob.P_values.begin(), prob.P_values.end(), P_batch.begin() + b * prob.nnzP);
            std::copy(prob.A_values.begin(), prob.A_values.end(), A_batch.begin() + b * prob.nnzA);
            std::copy(prob.q_values.begin(), prob.q_values.end(), q_batch.begin() + b * prob.n);
            std::copy(prob.b_values.begin(), prob.b_values.end(), b_batch.begin() + b * prob.m);
        }

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * prob.nnzP * cfg.batchSize);
        cudaMalloc(&d_A, sizeof(double) * prob.nnzA * cfg.batchSize);
        cudaMalloc(&d_q, sizeof(double) * prob.n * cfg.batchSize);
        cudaMalloc(&d_b, sizeof(double) * prob.m * cfg.batchSize);

        cudaMemcpy(d_P, P_batch.data(), sizeof(double) * prob.nnzP * cfg.batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_batch.data(), sizeof(double) * prob.nnzA * cfg.batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q_batch.data(), sizeof(double) * prob.n * cfg.batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_batch.data(), sizeof(double) * prob.m * cfg.batchSize, cudaMemcpyHostToDevice);

        const int N_ITER = 5;

        // cuDSS solve
        double cudss_time = -1.0;
        {
            Settings s; s.verbose = false; s.maxIter = 100;
            s.ipm.kktSolverType = KKTSolverType::CuDSS;
            CompiledSolver solver(prob.n, prob.m, cfg.batchSize,
                prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
                prob.cones, s);
            // Warmup
            solver.solveAll(d_P, d_A, d_q, d_b);
            cudaDeviceSynchronize();

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < N_ITER; ++iter)
                solver.solveAll(d_P, d_A, d_q, d_b);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            cudss_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / N_ITER;
        }

        // Riccati solve
        double riccati_time = -1.0;
        {
            Settings s; s.verbose = false; s.maxIter = 100;
            s.ipm.kktSolverType = KKTSolverType::Riccati;
            CompiledSolver solver(prob.n, prob.m, cfg.batchSize,
                prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
                prob.cones, s);
            // Warmup
            solver.solveAll(d_P, d_A, d_q, d_b);
            cudaDeviceSynchronize();

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < N_ITER; ++iter)
                solver.solveAll(d_P, d_A, d_q, d_b);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            riccati_time = std::chrono::duration<double, std::milli>(t1 - t0).count() / N_ITER;
        }

        double speedup = (riccati_time > 0) ? cudss_time / riccati_time : 0;
        printf("%-30s %8lld %8lld %10.2f %10.2f %7.1fx\n",
               cfg.label.c_str(), (long long)prob.n, (long long)prob.m,
               cudss_time, riccati_time, speedup);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    std::cout << std::string(90, '-') << std::endl;
}

// ============================================================================
// Riccati Backward Pass Tests
// ============================================================================

#include "moreau/diff/diff.hpp"

TEST(RiccatiBackwardTest, DISABLED_Benchmark) {
    struct Config {
        int T, nx, nu, batchSize;
        bool equality_only;
        std::string label;
    };

    std::vector<Config> configs = {
        // smem path (block size <= 48)
        {50, 12,  6,   1,  true,  "T=50  nx=12  nu=6  B=1    eq"},
        {50, 12,  6,  64,  true,  "T=50  nx=12  nu=6  B=64   eq"},
        {50, 12,  6, 512,  true,  "T=50  nx=12  nu=6  B=512  eq"},
        {50, 12,  6,   1,  false, "T=50  nx=12  nu=6  B=1"},
        {50, 12,  6,  64,  false, "T=50  nx=12  nu=6  B=64"},
        {50, 12,  6, 512,  false, "T=50  nx=12  nu=6  B=512"},
        {20,  4,  2,   1,  true,  "T=20  nx=4   nu=2  B=1    eq"},
        {20,  4,  2, 512,  true,  "T=20  nx=4   nu=2  B=512  eq"},
    };

    std::cout << "\n=== Backward: Riccati vs cuDSS Benchmark ===" << std::endl;
    std::cout << std::string(100, '-') << std::endl;
    printf("%-35s %8s %8s %10s %10s %8s\n",
           "Config", "n", "m", "cuDSS(ms)", "Ricc(ms)", "Speedup");
    std::cout << std::string(100, '-') << std::endl;

    for (auto& cfg : configs) {
        auto prob = buildMPCProblem(cfg.T, cfg.nx, cfg.nu, cfg.batchSize, cfg.equality_only);

        std::vector<double> P_batch(prob.nnzP * cfg.batchSize);
        std::vector<double> A_batch(prob.nnzA * cfg.batchSize);
        std::vector<double> q_batch(prob.n * cfg.batchSize);
        std::vector<double> b_batch(prob.m * cfg.batchSize);
        for (int b = 0; b < cfg.batchSize; ++b) {
            std::copy(prob.P_values.begin(), prob.P_values.end(), P_batch.begin() + b * prob.nnzP);
            std::copy(prob.A_values.begin(), prob.A_values.end(), A_batch.begin() + b * prob.nnzA);
            std::copy(prob.q_values.begin(), prob.q_values.end(), q_batch.begin() + b * prob.n);
            std::copy(prob.b_values.begin(), prob.b_values.end(), b_batch.begin() + b * prob.m);
        }

        double *d_P, *d_A, *d_q, *d_b;
        cudaMalloc(&d_P, sizeof(double) * prob.nnzP * cfg.batchSize);
        cudaMalloc(&d_A, sizeof(double) * prob.nnzA * cfg.batchSize);
        cudaMalloc(&d_q, sizeof(double) * prob.n * cfg.batchSize);
        cudaMalloc(&d_b, sizeof(double) * prob.m * cfg.batchSize);

        cudaMemcpy(d_P, P_batch.data(), sizeof(double) * prob.nnzP * cfg.batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_A, A_batch.data(), sizeof(double) * prob.nnzA * cfg.batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q, q_batch.data(), sizeof(double) * prob.n * cfg.batchSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b_batch.data(), sizeof(double) * prob.m * cfg.batchSize, cudaMemcpyHostToDevice);

        const int N_ITER = 5;

        auto bench_backward = [&](KKTSolverType type) -> double {
            Settings s; s.verbose = false; s.maxIter = 100;
            s.enableGrad = true;
            s.ipm.kktSolverType = type;
            CompiledSolver solver(prob.n, prob.m, cfg.batchSize,
                prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
                prob.cones, s);

            solver.solveAll(d_P, d_A, d_q, d_b);
            cudaDeviceSynchronize();

            BatchedVector dx_bar(prob.n, cfg.batchSize);
            BatchedVector dz_bar(prob.m, cfg.batchSize);
            BatchedVector ds_bar(prob.m, cfg.batchSize);
            std::vector<double> dx_data(prob.n * cfg.batchSize, 1.0);
            cudaMemcpy(dx_bar.data(), dx_data.data(), sizeof(double) * dx_data.size(), cudaMemcpyHostToDevice);
            dz_bar.setToConstant(0.0);
            ds_bar.setToConstant(0.0);

            // Warmup
            backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
            cudaDeviceSynchronize();

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int iter = 0; iter < N_ITER; ++iter)
                backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count() / N_ITER;
        };

        double cudss_time = bench_backward(KKTSolverType::CuDSS);
        double riccati_time = bench_backward(KKTSolverType::Riccati);

        double speedup = (riccati_time > 0) ? cudss_time / riccati_time : 0;
        printf("%-35s %8lld %8lld %10.2f %10.2f %7.1fx\n",
               cfg.label.c_str(), (long long)prob.n, (long long)prob.m,
               cudss_time, riccati_time, speedup);

        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    }

    std::cout << std::string(100, '-') << std::endl;
}

/**
 * @brief Test Riccati backward pass against cuDSS backward pass
 */
TEST(RiccatiBackwardTest, GradientsMatchCuDSS) {
    int T = 5, nx = 4, nu = 2;
    int batchSize = 2;
    auto prob = buildMPCProblem(T, nx, nu, batchSize);

    auto upload = [&](const std::vector<double>& h, int reps = 1) {
        std::vector<double> batched;
        for (int r = 0; r < reps; ++r)
            batched.insert(batched.end(), h.begin(), h.end());
        double* d; cudaMalloc(&d, sizeof(double) * batched.size());
        cudaMemcpy(d, batched.data(), sizeof(double) * batched.size(), cudaMemcpyHostToDevice);
        return d;
    };

    double* d_P = upload(prob.P_values, batchSize);
    double* d_A = upload(prob.A_values, batchSize);
    double* d_q = upload(prob.q_values, batchSize);
    double* d_b = upload(prob.b_values, batchSize);

    auto run_backward = [&](KKTSolverType type,
                            std::vector<double>& out_dq, std::vector<double>& out_db,
                            std::vector<double>& out_dP, std::vector<double>& out_dA) {
        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.kktSolverType = type;

        CompiledSolver solver(prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, settings);

        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(prob.n, batchSize);
        BatchedVector dz_bar(prob.m, batchSize);
        BatchedVector ds_bar(prob.m, batchSize);
        std::vector<double> dx_data(prob.n * batchSize, 1.0);
        cudaMemcpy(dx_bar.data(), dx_data.data(), sizeof(double) * dx_data.size(), cudaMemcpyHostToDevice);
        dz_bar.setToConstant(0.0);
        ds_bar.setToConstant(0.0);

        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        out_dq.resize(prob.n * batchSize);
        out_db.resize(prob.m * batchSize);
        out_dP.resize(prob.nnzP * batchSize);
        out_dA.resize(prob.nnzA * batchSize);
        cudaMemcpy(out_dq.data(), solver.diff_state()->dq.data(),
                  sizeof(double) * out_dq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_db.data(), solver.diff_state()->db.data(),
                  sizeof(double) * out_db.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_dP.data(), solver.diff_state()->dP_values.data(),
                  sizeof(double) * out_dP.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_dA.data(), solver.diff_state()->dA_values.data(),
                  sizeof(double) * out_dA.size(), cudaMemcpyDeviceToHost);
    };

    std::vector<double> cudss_dq, cudss_db, cudss_dP, cudss_dA;
    std::vector<double> riccati_dq, riccati_db, riccati_dP, riccati_dA;
    run_backward(KKTSolverType::CuDSS, cudss_dq, cudss_db, cudss_dP, cudss_dA);
    run_backward(KKTSolverType::Riccati, riccati_dq, riccati_db, riccati_dP, riccati_dA);

    auto max_rel_err = [](const std::vector<double>& ref, const std::vector<double>& test) {
        double mx = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            double err = std::abs(ref[i] - test[i]);
            double scale = std::max(1.0, std::abs(ref[i]));
            mx = std::max(mx, err / scale);
        }
        return mx;
    };

    double dq_err = max_rel_err(cudss_dq, riccati_dq);
    double db_err = max_rel_err(cudss_db, riccati_db);
    double dP_err = max_rel_err(cudss_dP, riccati_dP);
    double dA_err = max_rel_err(cudss_dA, riccati_dA);

    std::cout << "Riccati vs cuDSS backward:" << std::endl;
    std::cout << "  max relative dq error: " << dq_err << std::endl;
    std::cout << "  max relative db error: " << db_err << std::endl;
    std::cout << "  max relative dP error: " << dP_err << std::endl;
    std::cout << "  max relative dA error: " << dA_err << std::endl;

    EXPECT_LT(dq_err, 1e-3);
    EXPECT_LT(db_err, 1e-3);
    EXPECT_LT(dP_err, 1e-3);
    EXPECT_LT(dA_err, 1e-3);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// Max relative gradient errors (dq, db, dP, dA) between the Riccati and cuDSS
// backward passes on one problem. Seed dL/dx = ones, dL/dz = dL/ds = 0.
struct BackwardGradErrors { double dq, db, dP, dA; };

static BackwardGradErrors riccati_vs_cudss_backward(const MPCProblem& p, int batchSize) {
    auto upload = [&](const std::vector<double>& h) {
        std::vector<double> batched;
        for (int r = 0; r < batchSize; ++r) batched.insert(batched.end(), h.begin(), h.end());
        double* d; cudaMalloc(&d, sizeof(double) * batched.size());
        cudaMemcpy(d, batched.data(), sizeof(double) * batched.size(), cudaMemcpyHostToDevice);
        return d;
    };
    double* d_P = upload(p.P_values);
    double* d_A = upload(p.A_values);
    double* d_q = upload(p.q_values);
    double* d_b = upload(p.b_values);

    auto run = [&](KKTSolverType type, std::vector<double>& dq, std::vector<double>& db,
                   std::vector<double>& dP, std::vector<double>& dA) {
        Settings s; s.verbose = false; s.enableGrad = true; s.ipm.kktSolverType = type;
        CompiledSolver solver(p.n, p.m, batchSize,
            p.P_ro.data(), p.P_ci.data(), p.nnzP,
            p.A_ro.data(), p.A_ci.data(), p.nnzA, p.cones, s);
        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(p.n, batchSize), dz_bar(p.m, batchSize), ds_bar(p.m, batchSize);
        std::vector<double> ones(p.n * batchSize, 1.0);
        cudaMemcpy(dx_bar.data(), ones.data(), sizeof(double) * ones.size(), cudaMemcpyHostToDevice);
        dz_bar.setToConstant(0.0); ds_bar.setToConstant(0.0);
        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        dq.resize(p.n * batchSize); db.resize(p.m * batchSize);
        dP.resize(p.nnzP * batchSize); dA.resize(p.nnzA * batchSize);
        cudaMemcpy(dq.data(), solver.diff_state()->dq.data(), sizeof(double) * dq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(db.data(), solver.diff_state()->db.data(), sizeof(double) * db.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(dP.data(), solver.diff_state()->dP_values.data(), sizeof(double) * dP.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(dA.data(), solver.diff_state()->dA_values.data(), sizeof(double) * dA.size(), cudaMemcpyDeviceToHost);
    };

    std::vector<double> cdq, cdb, cdP, cdA, rdq, rdb, rdP, rdA;
    run(KKTSolverType::CuDSS, cdq, cdb, cdP, cdA);
    run(KKTSolverType::Riccati, rdq, rdb, rdP, rdA);

    auto err = [](const std::vector<double>& a, const std::vector<double>& b) {
        double mx = 0;
        for (size_t i = 0; i < a.size(); ++i)
            mx = std::max(mx, std::abs(a[i] - b[i]) / std::max(1.0, std::abs(a[i])));
        return mx;
    };
    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    return { err(cdq, rdq), err(cdb, rdb), err(cdP, rdP), err(cdA, rdA) };
}

// Backward through a *scrambled* MPC: the forward recovers the band via RCM and
// the backward (DiffRiccatiData) mirrors the same reorder. Correctness is
// measured *relative to the natural-order baseline*: the cuDSS-vs-Riccati
// gradient gap is set by forward convergence at this horizon (size, not order),
// so the permuted problem must agree no worse than the same unscrambled problem.
TEST(RiccatiBackwardTest, GradientsMatchCuDSS_Permuted) {
    int T = 52, nx = 3, nu = 2;  // n = 263 > kRiccatiMaxSmemBlock: scrambled order needs the reorder
    int batchSize = 2;
    MPCProblem natural = buildMPCProblem(T, nx, nu, batchSize);
    MPCProblem to_scramble = buildMPCProblem(T, nx, nu, batchSize);
    ASSERT_GT(natural.n, kRiccatiMaxSmemBlock);
    auto map = scrambleMap((int)to_scramble.n, /*seed=*/2024u);
    MPCProblem scrambled = permuteMPCProblem(std::move(to_scramble), map);

    // The scrambled order is not Riccati-usable as given, so the backward only
    // works via the internal RCM reorder of DiffRiccatiData.
    auto bf = detect_block_tridiagonal(
        scrambled.n, scrambled.m, scrambled.P_ro.data(), scrambled.P_ci.data(), scrambled.nnzP,
        scrambled.A_ro.data(), scrambled.A_ci.data(), scrambled.nnzA, scrambled.cones,
        /*allow_permute=*/false, nullptr);
    bool fixed_usable = !bf.empty() &&
        *std::max_element(bf.begin(), bf.end()) <= kRiccatiMaxSmemBlock;
    ASSERT_FALSE(fixed_usable) << "scrambled MPC should not be Riccati-usable as given";

    BackwardGradErrors nat = riccati_vs_cudss_backward(natural, batchSize);
    BackwardGradErrors perm = riccati_vs_cudss_backward(scrambled, batchSize);

    std::cout << "backward Riccati-vs-cuDSS  natural: dq=" << nat.dq << " db=" << nat.db
              << "  permuted: dq=" << perm.dq << " db=" << perm.db
              << " dP=" << perm.dP << " dA=" << perm.dA << std::endl;

    // Permutation must not degrade gradient agreement beyond the natural-order
    // baseline (a small slack absorbs reorder-dependent rounding).
    EXPECT_LT(perm.dq, std::max(1e-4, 5.0 * nat.dq));
    EXPECT_LT(perm.db, std::max(1e-4, 5.0 * nat.db));
    EXPECT_LT(perm.dP, std::max(1e-4, 5.0 * nat.dP));
    EXPECT_LT(perm.dA, std::max(1e-4, 5.0 * nat.dA));
}

// Regression for the gap-row cross-term bug: with nonzero q the solution (and the
// HSDE gap row) is non-trivial, so the Riccati backward must include the [c1;c2]*c3
// term in its tau-bordering. Before the fix this was wrong by 100-1000x; every
// other backward test uses q=0 (=> x=0 => c1=c2=c3=0) and never exercised it.
// Equality-only keeps the solution map smooth so cuDSS is an exact reference.
TEST(RiccatiBackwardTest, GradientsMatchCuDSS_NonzeroQ) {
    int T = 10, nx = 4, nu = 2, batchSize = 2;
    auto prob = buildMPCProblem(T, nx, nu, batchSize, /*equality_only=*/true);
    for (int64_t i = 0; i < prob.n; ++i)
        prob.q_values[i] = 0.1 * (1.0 + (double)(i % 7));

    auto e = riccati_vs_cudss_backward(prob, batchSize);
    std::cout << "nonzero-q backward Riccati-vs-cuDSS: dq=" << e.dq << " db=" << e.db
              << " dP=" << e.dP << " dA=" << e.dA << std::endl;
    EXPECT_LT(e.dq, 1e-3);
    EXPECT_LT(e.db, 1e-3);
    EXPECT_LT(e.dP, 1e-3);
    EXPECT_LT(e.dA, 1e-3);
}

// Full {cone type} x {q} x {b} matrix for the Riccati backward vs cuDSS (which is
// FD-validated). The gap-row cross-term bug only showed up for nonzero q; the
// nonneg-cone path additionally exercises the D1/D_u (get_h) weights that the
// equality-only cases skip. Covers every combo so no corner is assumed correct.
TEST(RiccatiBackwardTest, GradientsMatchCuDSS_QBCombos) {
    const int T = 8, nx = 4, nu = 2, batchSize = 2;

    auto add_q = [](MPCProblem& p) {
        for (int64_t i = 0; i < p.n; ++i) p.q_values[i] = 0.1 * (1.0 + (double)(i % 7));
    };
    auto add_b = [](MPCProblem& p) {
        // Perturb only the existing entries; keep dynamics (zero-cone) rows feasible
        // by using small values so the QP stays solvable across both backends.
        for (int64_t i = 0; i < p.m; ++i) p.b_values[i] += 0.05 * (1.0 + (double)(i % 5));
    };

    for (bool equality : {true, false}) {
        for (bool nz_q : {false, true}) {
            for (bool nz_b : {false, true}) {
                auto prob = buildMPCProblem(T, nx, nu, batchSize, equality);
                if (nz_q) add_q(prob);
                if (nz_b) add_b(prob);

                auto e = riccati_vs_cudss_backward(prob, batchSize);
                std::cout << "  combo[" << (equality ? "eq" : "ineq")
                          << " q=" << nz_q << " b=" << nz_b << "]"
                          << " dq=" << e.dq << " db=" << e.db
                          << " dP=" << e.dP << " dA=" << e.dA << std::endl;

                std::string tag = std::string(equality ? "eq" : "ineq")
                                + " q=" + std::to_string(nz_q) + " b=" + std::to_string(nz_b);
                EXPECT_LT(e.dq, 1e-3) << tag;
                EXPECT_LT(e.db, 1e-3) << tag;
                EXPECT_LT(e.dP, 1e-3) << tag;
                EXPECT_LT(e.dA, 1e-3) << tag;
            }
        }
    }
}

/**
 * @brief Finite-difference validation of Riccati backward dq gradients
 */
TEST(RiccatiBackwardTest, FiniteDifference_dq) {
    int T = 3, nx = 2, nu = 1;
    int batchSize = 1;
    auto prob = buildMPCProblem(T, nx, nu, batchSize);

    auto upload = [](const std::vector<double>& h) {
        double* d; cudaMalloc(&d, sizeof(double) * h.size());
        cudaMemcpy(d, h.data(), sizeof(double) * h.size(), cudaMemcpyHostToDevice);
        return d;
    };

    double* d_P = upload(prob.P_values);
    double* d_A = upload(prob.A_values);
    double* d_q; cudaMalloc(&d_q, sizeof(double) * prob.n);
    cudaMemcpy(d_q, prob.q_values.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    double* d_b = upload(prob.b_values);

    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;
    settings.ipm.kktSolverType = KKTSolverType::Riccati;

    // Reference solve
    CompiledSolver solver(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);
    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    std::vector<double> x_ref(prob.n);
    cudaMemcpy(x_ref.data(), solver.solution.x.data(),
              sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

    // Backward: dx_bar = e_0
    BatchedVector dx_bar(prob.n, batchSize);
    BatchedVector dz_bar(prob.m, batchSize);
    BatchedVector ds_bar(prob.m, batchSize);
    dx_bar.setToConstant(0.0);
    dz_bar.setToConstant(0.0);
    ds_bar.setToConstant(0.0);
    std::vector<double> dx_data(prob.n, 0.0);
    dx_data[0] = 1.0;
    cudaMemcpy(dx_bar.data(), dx_data.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);

    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
    cudaDeviceSynchronize();

    std::vector<double> analytic_dq(prob.n);
    cudaMemcpy(analytic_dq.data(), solver.diff_state()->dq.data(),
              sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

    // Finite differences
    double h = 1e-6;
    double max_err = 0;
    for (int64_t i = 0; i < prob.n; ++i) {
        std::vector<double> q_pert = prob.q_values;
        q_pert[i] += h;
        cudaMemcpy(d_q, q_pert.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);

        CompiledSolver solver_pert(prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, settings);
        solver_pert.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        std::vector<double> x_pert(prob.n);
        cudaMemcpy(x_pert.data(), solver_pert.solution.x.data(),
                  sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

        double fd_grad = (x_pert[0] - x_ref[0]) / h;
        double err = std::abs(fd_grad - analytic_dq[i]);
        double scale = std::max(1.0, std::abs(fd_grad));
        max_err = std::max(max_err, err / scale);
    }

    std::cout << "Riccati backward FD validation (dq): max relative error = " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-3);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

/**
 * @brief Finite-difference validation of db gradients (equality-only)
 */
TEST(RiccatiBackwardTest, FiniteDifference_db) {
    int T = 3, nx = 2, nu = 1;
    int batchSize = 1;
    auto prob = buildMPCProblem(T, nx, nu, batchSize, /*equality_only=*/true);

    auto upload = [](const std::vector<double>& h) {
        double* d; cudaMalloc(&d, sizeof(double) * h.size());
        cudaMemcpy(d, h.data(), sizeof(double) * h.size(), cudaMemcpyHostToDevice);
        return d;
    };

    double* d_P = upload(prob.P_values);
    double* d_A = upload(prob.A_values);
    double* d_q = upload(prob.q_values);
    double* d_b; cudaMalloc(&d_b, sizeof(double) * prob.m);
    cudaMemcpy(d_b, prob.b_values.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;
    settings.ipm.kktSolverType = KKTSolverType::Riccati;

    // Reference solve
    CompiledSolver solver(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);
    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    std::vector<double> x_ref(prob.n);
    cudaMemcpy(x_ref.data(), solver.solution.x.data(),
              sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

    // Backward: dx_bar = ones
    BatchedVector dx_bar(prob.n, batchSize);
    BatchedVector dz_bar(prob.m, batchSize);
    BatchedVector ds_bar(prob.m, batchSize);
    std::vector<double> dx_data(prob.n, 1.0);
    cudaMemcpy(dx_bar.data(), dx_data.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    dz_bar.setToConstant(0.0);
    ds_bar.setToConstant(0.0);

    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
    cudaDeviceSynchronize();

    std::vector<double> analytic_db(prob.m);
    cudaMemcpy(analytic_db.data(), solver.diff_state()->db.data(),
              sizeof(double) * prob.m, cudaMemcpyDeviceToHost);

    // Finite differences: d(sum(x))/db[i]
    double h = 1e-6;
    double max_err = 0;
    for (int64_t i = 0; i < prob.m; ++i) {
        std::vector<double> b_pert = prob.b_values;
        b_pert[i] += h;
        cudaMemcpy(d_b, b_pert.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

        CompiledSolver solver_pert(prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, settings);
        solver_pert.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        std::vector<double> x_pert(prob.n);
        cudaMemcpy(x_pert.data(), solver_pert.solution.x.data(),
                  sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

        double sum_ref = 0, sum_pert = 0;
        for (int j = 0; j < prob.n; ++j) { sum_ref += x_ref[j]; sum_pert += x_pert[j]; }
        double fd_grad = (sum_pert - sum_ref) / h;
        double err = std::abs(fd_grad - analytic_db[i]);
        double scale = std::max(1.0, std::abs(fd_grad));
        max_err = std::max(max_err, err / scale);
    }

    std::cout << "Riccati backward FD validation (db): max relative error = " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-3);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// Asserts the Riccati (and cuDSS/Woodbury) backward `db` against central-
// difference ground truth on smooth equality-only MPCs with a non-trivial
// (nonzero-q) solution, across a horizon sweep, a cost-conditioning sweep, and a
// b-dominated case. This is the test the gap-row cross-term bug would have failed
// (it was wrong by 100-17000x vs FD); the earlier print-only version asserted
// nothing and so caught nothing.
TEST(RiccatiBackwardTest, LongHorizonBackwardFD) {
    auto set_nonzero_q = [](MPCProblem& p) {
        for (int64_t i = 0; i < p.n; ++i) p.q_values[i] = 0.1 * (1.0 + (double)(i % 7));
    };

    // Returns {cuDSS-vs-FD, Riccati-vs-FD, cuDSS-vs-Riccati} max relative db error.
    auto errors = [](const MPCProblem& prob, double tol, double h) {
        const int batchSize = 1;
        auto up = [&](const std::vector<double>& v) {
            double* d; cudaMalloc(&d, sizeof(double) * v.size());
            cudaMemcpy(d, v.data(), sizeof(double) * v.size(), cudaMemcpyHostToDevice);
            return d;
        };
        double* dP = up(prob.P_values); double* dA = up(prob.A_values); double* dq = up(prob.q_values);
        double* db; cudaMalloc(&db, sizeof(double) * prob.m);

        auto sumx = [&](const std::vector<double>& bvec) {
            cudaMemcpy(db, bvec.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);
            Settings s; s.verbose = false; s.maxIter = 400;
            s.ipm.kktSolverType = KKTSolverType::CuDSS;
            s.ipm.tolGapAbs = tol; s.ipm.tolGapRel = tol; s.ipm.tolFeas = tol;
            CompiledSolver sol(prob.n, prob.m, batchSize,
                prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                prob.A_ro.data(), prob.A_ci.data(), prob.nnzA, prob.cones, s);
            sol.solveAll(dP, dA, dq, db); cudaDeviceSynchronize();
            std::vector<double> x(prob.n);
            cudaMemcpy(x.data(), sol.solution.x.data(), sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
            double r = 0; for (double v : x) r += v; return r;
        };
        auto analytic = [&](KKTSolverType type) {
            cudaMemcpy(db, prob.b_values.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);
            Settings s; s.verbose = false; s.maxIter = 400; s.enableGrad = true; s.ipm.kktSolverType = type;
            s.ipm.tolGapAbs = tol; s.ipm.tolGapRel = tol; s.ipm.tolFeas = tol;
            CompiledSolver sol(prob.n, prob.m, batchSize,
                prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
                prob.A_ro.data(), prob.A_ci.data(), prob.nnzA, prob.cones, s);
            sol.solveAll(dP, dA, dq, db); cudaDeviceSynchronize();
            BatchedVector dxb(prob.n, batchSize), dzb(prob.m, batchSize), dsb(prob.m, batchSize);
            std::vector<double> ones(prob.n, 1.0);
            cudaMemcpy(dxb.data(), ones.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
            dzb.setToConstant(0.0); dsb.setToConstant(0.0);
            backward(*sol.diff_state(), dxb, dzb, dsb, sol, 0); cudaDeviceSynchronize();
            std::vector<double> g(prob.m);
            cudaMemcpy(g.data(), sol.diff_state()->db.data(), sizeof(double) * prob.m, cudaMemcpyDeviceToHost);
            return g;
        };

        std::vector<int64_t> idx;
        for (int k = 0; k < 12; ++k) idx.push_back((int64_t)(((double)k + 0.5) / 12.0 * prob.m));
        std::vector<double> fd(idx.size());
        for (size_t k = 0; k < idx.size(); ++k) {
            auto bp = prob.b_values; bp[idx[k]] += h; double Lp = sumx(bp);
            auto bm = prob.b_values; bm[idx[k]] -= h; double Lm = sumx(bm);
            fd[k] = (Lp - Lm) / (2 * h);
        }
        auto cud = analytic(KKTSolverType::CuDSS);
        auto ric = analytic(KKTSolverType::Riccati);
        std::vector<double> wood; bool wood_ok = true;
        try { wood = analytic(KKTSolverType::Woodbury); }
        catch (const std::exception&) { wood_ok = false; }
        double e_cf = 0, e_rf = 0, e_wf = 0;
        for (size_t k = 0; k < idx.size(); ++k) {
            double sc = std::max(1.0, std::abs(fd[k]));
            e_cf = std::max(e_cf, std::abs(cud[idx[k]] - fd[k]) / sc);
            e_rf = std::max(e_rf, std::abs(ric[idx[k]] - fd[k]) / sc);
            if (wood_ok) e_wf = std::max(e_wf, std::abs(wood[idx[k]] - fd[k]) / sc);
        }
        cudaFree(dP); cudaFree(dA); cudaFree(dq); cudaFree(db);
        return std::array<double, 3>{e_cf, e_rf, wood_ok ? e_wf : -1.0};
    };

    // All three backends agree with central-difference FD to <=1.2e-3 across
    // every case below; the FD step error (truncation + IPM tolerance) is itself
    // ~1e-3 at the longest horizons, so a flat 5e-3 bound sits safely above the
    // FD noise yet ~100x below the gap-row bug (0.66-17000x), which it would have
    // caught at every single case.
    const double FD_TOL = 5e-3;
    auto assert_against_fd = [&](const std::array<double, 3>& e, const std::string& tag) {
        std::cout << "  " << tag << "  cuDSS-vs-FD=" << e[0]
                  << "  Riccati-vs-FD=" << e[1] << "  Woodbury-vs-FD=" << e[2] << "\n";
        EXPECT_LT(e[0], FD_TOL) << tag << " (cuDSS)";
        EXPECT_LT(e[1], FD_TOL) << tag << " (Riccati)";
        if (e[2] >= 0.0) EXPECT_LT(e[2], FD_TOL) << tag << " (Woodbury)";
    };

    std::cout << "\n[horizon sweep, default cond R=0.01 Qf=10]\n";
    for (int T : {5, 20, 40}) {
        auto prob = buildMPCProblem(T, 4, 2, 1, /*equality_only=*/true);
        set_nonzero_q(prob);
        assert_against_fd(errors(prob, 1e-10, 1e-4), "T=" + std::to_string(T));
    }

    std::cout << "[conditioning sweep, fixed T=40]\n";
    struct Cond { double R, Qf; };
    for (auto c : {Cond{1.0, 1.0}, Cond{0.1, 10.0}, Cond{0.01, 10.0}, Cond{0.001, 100.0}}) {
        auto prob = buildMPCProblem(40, 4, 2, 1, /*equality_only=*/true, c.R, c.Qf);
        set_nonzero_q(prob);
        assert_against_fd(errors(prob, 1e-10, 1e-4),
                          "R=" + std::to_string(c.R) + " Qf=" + std::to_string(c.Qf));
    }

    std::cout << "[dominant-b check, T=20] (nonzero, b-dominated d_tt)\n";
    {
        auto prob = buildMPCProblem(20, 4, 2, 1, /*equality_only=*/true);
        set_nonzero_q(prob);
        for (int64_t i = 0; i < prob.m; ++i) prob.b_values[i] = 3.0 * (1.0 + (double)(i % 5));
        assert_against_fd(errors(prob, 1e-10, 1e-4), "dominant-b");
    }
}

/**
 * @brief Finite-difference validation of dP and dA gradients
 */
TEST(RiccatiBackwardTest, FiniteDifference_dP_dA) {
    int T = 3, nx = 2, nu = 1;
    int batchSize = 1;
    // Use mixed cones so we get nontrivial bounds, and set nonzero q for nontrivial x
    auto prob = buildMPCProblem(T, nx, nu, batchSize, /*equality_only=*/false);
    // Set nonzero q to get x ≠ 0 (needed for dP, dA to be nonzero)
    for (int i = 0; i < prob.n; ++i)
        prob.q_values[i] = 0.1 * (i + 1);

    auto upload = [](const std::vector<double>& h) {
        double* d; cudaMalloc(&d, sizeof(double) * h.size());
        cudaMemcpy(d, h.data(), sizeof(double) * h.size(), cudaMemcpyHostToDevice);
        return d;
    };

    double* d_P; cudaMalloc(&d_P, sizeof(double) * prob.nnzP);
    cudaMemcpy(d_P, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
    double* d_A; cudaMalloc(&d_A, sizeof(double) * prob.nnzA);
    cudaMemcpy(d_A, prob.A_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
    double* d_q = upload(prob.q_values);
    double* d_b = upload(prob.b_values);

    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;
    settings.ipm.kktSolverType = KKTSolverType::Riccati;

    // Reference solve
    CompiledSolver solver(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);
    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    std::vector<double> x_ref(prob.n);
    cudaMemcpy(x_ref.data(), solver.solution.x.data(),
              sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
    double obj_ref = 0;
    for (int i = 0; i < prob.n; ++i) obj_ref += x_ref[i];

    // Backward: dx_bar = ones (so objective = sum(x))
    BatchedVector dx_bar(prob.n, batchSize);
    BatchedVector dz_bar(prob.m, batchSize);
    BatchedVector ds_bar(prob.m, batchSize);
    std::vector<double> dx_data(prob.n, 1.0);
    cudaMemcpy(dx_bar.data(), dx_data.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    dz_bar.setToConstant(0.0);
    ds_bar.setToConstant(0.0);

    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
    cudaDeviceSynchronize();

    std::vector<double> analytic_dP(prob.nnzP), analytic_dA(prob.nnzA);
    cudaMemcpy(analytic_dP.data(), solver.diff_state()->dP_values.data(),
              sizeof(double) * prob.nnzP, cudaMemcpyDeviceToHost);
    cudaMemcpy(analytic_dA.data(), solver.diff_state()->dA_values.data(),
              sizeof(double) * prob.nnzA, cudaMemcpyDeviceToHost);

    double h = 1e-6;

    // FD for dP
    double max_dP_err = 0;
    for (int64_t i = 0; i < prob.nnzP; ++i) {
        std::vector<double> P_pert = prob.P_values;
        P_pert[i] += h;
        // P must be symmetric — find the symmetric partner and perturb it too
        // For diagonal P (MPC), P[i,i] has no partner. For off-diagonal, need both.
        // Since buildMPCProblem uses diagonal P, each entry is self-symmetric.
        cudaMemcpy(d_P, P_pert.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);

        CompiledSolver solver_pert(prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, settings);
        solver_pert.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        std::vector<double> x_pert(prob.n);
        cudaMemcpy(x_pert.data(), solver_pert.solution.x.data(),
                  sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

        double obj_pert = 0;
        for (int j = 0; j < prob.n; ++j) obj_pert += x_pert[j];
        double fd = (obj_pert - obj_ref) / h;
        double err = std::abs(fd - analytic_dP[i]) / std::max(1.0, std::abs(fd));
        max_dP_err = std::max(max_dP_err, err);
    }
    // Restore P
    cudaMemcpy(d_P, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);

    // FD for dA
    double max_dA_err = 0;
    for (int64_t i = 0; i < prob.nnzA; ++i) {
        std::vector<double> A_pert = prob.A_values;
        A_pert[i] += h;
        cudaMemcpy(d_A, A_pert.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);

        CompiledSolver solver_pert(prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, settings);
        solver_pert.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        std::vector<double> x_pert(prob.n);
        cudaMemcpy(x_pert.data(), solver_pert.solution.x.data(),
                  sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

        double obj_pert = 0;
        for (int j = 0; j < prob.n; ++j) obj_pert += x_pert[j];
        double fd = (obj_pert - obj_ref) / h;
        double err = std::abs(fd - analytic_dA[i]) / std::max(1.0, std::abs(fd));
        max_dA_err = std::max(max_dA_err, err);
    }

    std::cout << "Riccati backward FD (dP): " << max_dP_err
              << ", (dA): " << max_dA_err << std::endl;
    // dP FD has larger error due to equilibration + quadratic x dependence
    EXPECT_LT(max_dP_err, 5e-2);
    EXPECT_LT(max_dA_err, 1e-3);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

/**
 * @brief Finite-difference validation with mixed cones (zero + nonneg)
 */
TEST(RiccatiBackwardTest, FiniteDifferenceMixed) {
    int T = 3, nx = 2, nu = 1;
    int batchSize = 1;
    auto prob = buildMPCProblem(T, nx, nu, batchSize, /*equality_only=*/false);

    auto upload = [](const std::vector<double>& h) {
        double* d; cudaMalloc(&d, sizeof(double) * h.size());
        cudaMemcpy(d, h.data(), sizeof(double) * h.size(), cudaMemcpyHostToDevice);
        return d;
    };

    double* d_P = upload(prob.P_values);
    double* d_A = upload(prob.A_values);
    double* d_q; cudaMalloc(&d_q, sizeof(double) * prob.n);
    cudaMemcpy(d_q, prob.q_values.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    double* d_b; cudaMalloc(&d_b, sizeof(double) * prob.m);
    cudaMemcpy(d_b, prob.b_values.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;
    settings.ipm.kktSolverType = KKTSolverType::Riccati;

    // Reference solve
    CompiledSolver solver(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, settings);
    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    std::vector<double> x_ref(prob.n);
    cudaMemcpy(x_ref.data(), solver.solution.x.data(),
              sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

    // Backward: dx_bar = e_0
    BatchedVector dx_bar(prob.n, batchSize);
    BatchedVector dz_bar(prob.m, batchSize);
    BatchedVector ds_bar(prob.m, batchSize);
    dx_bar.setToConstant(0.0);
    dz_bar.setToConstant(0.0);
    ds_bar.setToConstant(0.0);
    std::vector<double> dx_data(prob.n, 0.0);
    dx_data[0] = 1.0;
    cudaMemcpy(dx_bar.data(), dx_data.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);

    backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
    cudaDeviceSynchronize();

    std::vector<double> analytic_dq(prob.n), analytic_db(prob.m);
    cudaMemcpy(analytic_dq.data(), solver.diff_state()->dq.data(),
              sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
    cudaMemcpy(analytic_db.data(), solver.diff_state()->db.data(),
              sizeof(double) * prob.m, cudaMemcpyDeviceToHost);

    // FD for dq
    double h = 1e-6;
    double max_dq_err = 0;
    for (int64_t i = 0; i < prob.n; ++i) {
        std::vector<double> q_pert = prob.q_values;
        q_pert[i] += h;
        cudaMemcpy(d_q, q_pert.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);

        CompiledSolver solver_pert(prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, settings);
        solver_pert.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        std::vector<double> x_pert(prob.n);
        cudaMemcpy(x_pert.data(), solver_pert.solution.x.data(),
                  sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

        double fd = (x_pert[0] - x_ref[0]) / h;
        double err = std::abs(fd - analytic_dq[i]) / std::max(1.0, std::abs(fd));
        max_dq_err = std::max(max_dq_err, err);
    }
    // Restore q
    cudaMemcpy(d_q, prob.q_values.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);

    // FD for db
    double max_db_err = 0;
    for (int64_t i = 0; i < prob.m; ++i) {
        std::vector<double> b_pert = prob.b_values;
        b_pert[i] += h;
        cudaMemcpy(d_b, b_pert.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

        CompiledSolver solver_pert(prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, settings);
        solver_pert.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        std::vector<double> x_pert(prob.n);
        cudaMemcpy(x_pert.data(), solver_pert.solution.x.data(),
                  sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

        double fd = (x_pert[0] - x_ref[0]) / h;
        double err = std::abs(fd - analytic_db[i]) / std::max(1.0, std::abs(fd));
        max_db_err = std::max(max_db_err, err);
    }

    std::cout << "Mixed cones FD (dq): " << max_dq_err << ", (db): " << max_db_err << std::endl;
    EXPECT_LT(max_dq_err, 1e-3);
    EXPECT_LT(max_db_err, 1e-3);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

/**
 * @brief Riccati vs cuDSS at larger problem size and batch
 */
TEST(RiccatiBackwardTest, GradientsMatchCuDSS_LargeBatch) {
    int T = 10, nx = 4, nu = 2;
    int batchSize = 32;
    auto prob = buildMPCProblem(T, nx, nu, batchSize);

    auto upload = [&](const std::vector<double>& h, int reps = 1) {
        std::vector<double> batched;
        for (int r = 0; r < reps; ++r)
            batched.insert(batched.end(), h.begin(), h.end());
        double* d; cudaMalloc(&d, sizeof(double) * batched.size());
        cudaMemcpy(d, batched.data(), sizeof(double) * batched.size(), cudaMemcpyHostToDevice);
        return d;
    };

    double* d_P = upload(prob.P_values, batchSize);
    double* d_A = upload(prob.A_values, batchSize);
    double* d_q = upload(prob.q_values, batchSize);
    double* d_b = upload(prob.b_values, batchSize);

    auto run_backward = [&](KKTSolverType type,
                            std::vector<double>& out_dq, std::vector<double>& out_db,
                            std::vector<double>& out_dP, std::vector<double>& out_dA) {
        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.kktSolverType = type;

        CompiledSolver solver(prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, settings);

        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(prob.n, batchSize);
        BatchedVector dz_bar(prob.m, batchSize);
        BatchedVector ds_bar(prob.m, batchSize);
        std::vector<double> dx_data(prob.n * batchSize, 1.0);
        cudaMemcpy(dx_bar.data(), dx_data.data(), sizeof(double) * dx_data.size(), cudaMemcpyHostToDevice);
        dz_bar.setToConstant(0.0);
        ds_bar.setToConstant(0.0);

        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        out_dq.resize(prob.n * batchSize);
        out_db.resize(prob.m * batchSize);
        out_dP.resize(prob.nnzP * batchSize);
        out_dA.resize(prob.nnzA * batchSize);
        cudaMemcpy(out_dq.data(), solver.diff_state()->dq.data(), sizeof(double) * out_dq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_db.data(), solver.diff_state()->db.data(), sizeof(double) * out_db.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_dP.data(), solver.diff_state()->dP_values.data(), sizeof(double) * out_dP.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_dA.data(), solver.diff_state()->dA_values.data(), sizeof(double) * out_dA.size(), cudaMemcpyDeviceToHost);
    };

    auto max_rel_err = [](const std::vector<double>& ref, const std::vector<double>& test) {
        double mx = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            double err = std::abs(ref[i] - test[i]);
            mx = std::max(mx, err / std::max(1.0, std::abs(ref[i])));
        }
        return mx;
    };

    std::vector<double> cudss_dq, cudss_db, cudss_dP, cudss_dA;
    std::vector<double> riccati_dq, riccati_db, riccati_dP, riccati_dA;
    run_backward(KKTSolverType::CuDSS, cudss_dq, cudss_db, cudss_dP, cudss_dA);
    run_backward(KKTSolverType::Riccati, riccati_dq, riccati_db, riccati_dP, riccati_dA);

    double dq_err = max_rel_err(cudss_dq, riccati_dq);
    double db_err = max_rel_err(cudss_db, riccati_db);
    double dP_err = max_rel_err(cudss_dP, riccati_dP);
    double dA_err = max_rel_err(cudss_dA, riccati_dA);

    std::cout << "Large batch (T=10, B=32): dq=" << dq_err
              << " db=" << db_err << " dP=" << dP_err << " dA=" << dA_err << std::endl;

    EXPECT_LT(dq_err, 1e-3);
    EXPECT_LT(db_err, 1e-3);
    EXPECT_LT(dP_err, 1e-3);
    EXPECT_LT(dA_err, 1e-3);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

/**
 * @brief Riccati vs cuDSS for equality-only at larger T
 */
TEST(RiccatiBackwardTest, GradientsMatchCuDSS_EqualityLargeT) {
    int T = 20, nx = 4, nu = 2;
    int batchSize = 4;
    auto prob = buildMPCProblem(T, nx, nu, batchSize, /*equality_only=*/true);

    auto upload = [&](const std::vector<double>& h, int reps = 1) {
        std::vector<double> batched;
        for (int r = 0; r < reps; ++r)
            batched.insert(batched.end(), h.begin(), h.end());
        double* d; cudaMalloc(&d, sizeof(double) * batched.size());
        cudaMemcpy(d, batched.data(), sizeof(double) * batched.size(), cudaMemcpyHostToDevice);
        return d;
    };

    double* d_P = upload(prob.P_values, batchSize);
    double* d_A = upload(prob.A_values, batchSize);
    double* d_q = upload(prob.q_values, batchSize);
    double* d_b = upload(prob.b_values, batchSize);

    auto run_backward = [&](KKTSolverType type,
                            std::vector<double>& out_dq, std::vector<double>& out_db,
                            std::vector<double>& out_dP, std::vector<double>& out_dA) {
        Settings settings;
        settings.verbose = false;
        settings.enableGrad = true;
        settings.ipm.kktSolverType = type;

        CompiledSolver solver(prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, settings);

        solver.solveAll(d_P, d_A, d_q, d_b);
        cudaDeviceSynchronize();

        BatchedVector dx_bar(prob.n, batchSize);
        BatchedVector dz_bar(prob.m, batchSize);
        BatchedVector ds_bar(prob.m, batchSize);
        std::vector<double> dx_data(prob.n * batchSize, 1.0);
        cudaMemcpy(dx_bar.data(), dx_data.data(), sizeof(double) * dx_data.size(), cudaMemcpyHostToDevice);
        dz_bar.setToConstant(0.0);
        ds_bar.setToConstant(0.0);

        backward(*solver.diff_state(), dx_bar, dz_bar, ds_bar, solver, 0);
        cudaDeviceSynchronize();

        out_dq.resize(prob.n * batchSize);
        out_db.resize(prob.m * batchSize);
        out_dP.resize(prob.nnzP * batchSize);
        out_dA.resize(prob.nnzA * batchSize);
        cudaMemcpy(out_dq.data(), solver.diff_state()->dq.data(), sizeof(double) * out_dq.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_db.data(), solver.diff_state()->db.data(), sizeof(double) * out_db.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_dP.data(), solver.diff_state()->dP_values.data(), sizeof(double) * out_dP.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(out_dA.data(), solver.diff_state()->dA_values.data(), sizeof(double) * out_dA.size(), cudaMemcpyDeviceToHost);
    };

    auto max_rel_err = [](const std::vector<double>& ref, const std::vector<double>& test) {
        double mx = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
            double err = std::abs(ref[i] - test[i]);
            mx = std::max(mx, err / std::max(1.0, std::abs(ref[i])));
        }
        return mx;
    };

    std::vector<double> cudss_dq, cudss_db, cudss_dP, cudss_dA;
    std::vector<double> riccati_dq, riccati_db, riccati_dP, riccati_dA;
    run_backward(KKTSolverType::CuDSS, cudss_dq, cudss_db, cudss_dP, cudss_dA);
    run_backward(KKTSolverType::Riccati, riccati_dq, riccati_db, riccati_dP, riccati_dA);

    double dq_err = max_rel_err(cudss_dq, riccati_dq);
    double db_err = max_rel_err(cudss_db, riccati_db);
    double dP_err = max_rel_err(cudss_dP, riccati_dP);
    double dA_err = max_rel_err(cudss_dA, riccati_dA);

    std::cout << "Equality T=20 B=4: dq=" << dq_err
              << " db=" << db_err << " dP=" << dP_err << " dA=" << dA_err << std::endl;

    EXPECT_LT(dq_err, 1e-3);
    EXPECT_LT(db_err, 1e-3);
    EXPECT_LT(dP_err, 1e-3);
    EXPECT_LT(dA_err, 1e-3);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

/**
 * @brief Test that auto-dispatch falls back to cuDSS for large blocks
 */
TEST(RiccatiTest, AutoDispatchFallsToCuDSSForLargeBlocks) {
    // nx=50, nu=30 → block size = 80 >> smem limit
    int T = 3, nx = 50, nu = 30;
    int batchSize = 2;
    auto prob = buildMPCProblem(T, nx, nu, batchSize);

    Settings s; s.verbose = false;
    s.ipm.kktSolverType = KKTSolverType::Auto;
    CompiledSolver solver(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, s);

    // Auto should pick cuDSS since blocks exceed smem limit
    EXPECT_EQ(solver.kkt->actualSolverType(), KKTSolverType::CuDSS);
}

/**
 * @brief Test that explicit Riccati throws for oversized blocks
 */
TEST(RiccatiTest, ExplicitRiccatiThrowsForLargeBlocks) {
    // nx=50, nu=30 → block size = 80 >> smem limit
    int T = 3, nx = 50, nu = 30;
    int batchSize = 2;
    auto prob = buildMPCProblem(T, nx, nu, batchSize);

    Settings s; s.verbose = false;
    s.ipm.kktSolverType = KKTSolverType::Riccati;
    EXPECT_THROW(
        CompiledSolver(prob.n, prob.m, batchSize,
            prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
            prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
            prob.cones, s),
        std::runtime_error);
}

// ============================================================================
// Minimal horizon edge case tests
// ============================================================================

struct MinimalHorizonParams {
    int T, nx, nu;
};

class RiccatiMinimalHorizon : public ::testing::TestWithParam<MinimalHorizonParams> {};

TEST_P(RiccatiMinimalHorizon, RiccatiMatchesCuDSS) {
    auto params = GetParam();
    int batchSize = 1;
    auto prob = buildMPCProblem(params.T, params.nx, params.nu, batchSize);

    auto blocks = detect_block_tridiagonal(
        prob.n, prob.m,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones);

    if (blocks.empty()) {
        GTEST_SKIP() << "No block-tridiagonal structure detected for T="
                     << params.T << " nx=" << params.nx << " nu=" << params.nu;
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP);
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA);
    cudaMalloc(&d_q, sizeof(double) * prob.n);
    cudaMalloc(&d_b, sizeof(double) * prob.m);
    cudaMemcpy(d_P, prob.P_values.data(), sizeof(double) * prob.nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_values.data(), sizeof(double) * prob.nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q_values.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b_values.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

    Settings s_cudss;
    s_cudss.verbose = false;
    s_cudss.ipm.kktSolverType = KKTSolverType::CuDSS;
    CompiledSolver solver_cudss(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, s_cudss);
    solver_cudss.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_cudss(prob.n);
    cudaMemcpy(x_cudss.data(), solver_cudss.solution.x.data(),
               sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

    Settings s_riccati;
    s_riccati.verbose = false;
    s_riccati.ipm.kktSolverType = KKTSolverType::Riccati;
    CompiledSolver solver_riccati(prob.n, prob.m, batchSize,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP,
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA,
        prob.cones, s_riccati);
    solver_riccati.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_riccati(prob.n);
    cudaMemcpy(x_riccati.data(), solver_riccati.solution.x.data(),
               sizeof(double) * prob.n, cudaMemcpyDeviceToHost);

    int32_t status_cudss, status_riccati;
    cudaMemcpy(&status_cudss, solver_cudss.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&status_riccati, solver_riccati.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    EXPECT_TRUE(status_cudss == (int32_t)SolverStatus::Solved ||
                status_cudss == (int32_t)SolverStatus::AlmostSolved);
    EXPECT_TRUE(status_riccati == (int32_t)SolverStatus::Solved ||
                status_riccati == (int32_t)SolverStatus::AlmostSolved);

    double max_diff = 0.0;
    for (size_t i = 0; i < x_cudss.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(x_cudss[i] - x_riccati[i]));
    }
    std::cout << "T=" << params.T << " nx=" << params.nx << " nu=" << params.nu
              << " max diff: " << max_diff << std::endl;
    EXPECT_LT(max_diff, 1e-4);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

INSTANTIATE_TEST_SUITE_P(
    MinimalHorizons, RiccatiMinimalHorizon,
    ::testing::Values(
        MinimalHorizonParams{1, 4, 2},
        MinimalHorizonParams{2, 2, 1},
        MinimalHorizonParams{2, 4, 2}
    ));
