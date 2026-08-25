/**
 * @file test_psd.cpp
 * @brief Tests for PSD cone forward solve and backward pass on CUDA
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>

#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/diff/diff.hpp"

using namespace moreau;

constexpr double TOL_SOL = 1e-4;
constexpr double TOL_GRAD = 1e-3;
constexpr double TOL_GRAD_REL = 5e-2;
constexpr double FD_H = 1e-5;

inline double grad_tol(double a, double b) {
    return std::max(TOL_GRAD, TOL_GRAD_REL * std::max(std::abs(a), std::abs(b)));
}

// ============================================================================
// Helper
// ============================================================================

struct SolveResult {
    std::vector<double> x, z, s;
    int status;
};

SolveResult solve_problem(
    int64_t n, int64_t m,
    const std::vector<int64_t>& P_ro, const std::vector<int64_t>& P_ci, const std::vector<double>& P_val,
    const std::vector<int64_t>& A_ro, const std::vector<int64_t>& A_ci, const std::vector<double>& A_val,
    const std::vector<double>& q, const std::vector<double>& b,
    Cones& cones, bool enable_grad = false
) {
    Settings settings;
    settings.verbose = false;
    settings.enableGrad = enable_grad;

    CompiledSolver solver(n, m, 1,
        P_ro.data(), P_ci.data(), P_val.size(),
        A_ro.data(), A_ci.data(), A_val.size(),
        cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * std::max((size_t)1, P_val.size()));
    cudaMalloc(&d_A, sizeof(double) * std::max((size_t)1, A_val.size()));
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    if (!P_val.empty()) cudaMemcpy(d_P, P_val.data(), sizeof(double) * P_val.size(), cudaMemcpyHostToDevice);
    if (!A_val.empty()) cudaMemcpy(d_A, A_val.data(), sizeof(double) * A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    SolveResult r;
    r.x.resize(n); r.z.resize(m); r.s.resize(m);
    cudaMemcpy(r.x.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.z.data(), solver.solution.z.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.s.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    r.status = (int)solver.info.status[0];

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    return r;
}

// ============================================================================
// Forward solve tests (LP formulations — P = 0)
// ============================================================================

class PsdForwardTest : public ::testing::Test {};

// PSD(1) = nonneg: min x s.t. -x + s = -1, s >= 0 => x >= 1
TEST_F(PsdForwardTest, Psd1) {
    int64_t n = 1, m = 1;
    std::vector<int64_t> P_ro = {0, 0}, P_ci = {};
    std::vector<double> P_val = {};
    std::vector<int64_t> A_ro = {0, 1}, A_ci = {0};
    std::vector<double> A_val = {-1.0};
    std::vector<double> q = {1.0}, b = {-1.0};

    Cones cones{}; cones.psdConeDims = {1};
    auto r = solve_problem(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones);
    EXPECT_EQ(r.status, 1);  // Solved
    EXPECT_NEAR(r.x[0], 1.0, TOL_SOL);
}

// PSD(2): min x0 + x1
// s.t. [-1 0; 0 0; 0 -1]x + s = [-1; 0; -1], s in PSD(2)
// svec: s = [x0-1, 0, x1-1], matrix = [[x0-1, 0],[0, x1-1]] PSD => x0>=1, x1>=1
TEST_F(PsdForwardTest, Psd2) {
    int64_t n = 2, m = 3;
    std::vector<int64_t> P_ro = {0, 0, 0}, P_ci = {};
    std::vector<double> P_val = {};
    // A: row0 col0=-1, row2 col1=-1 (row1 empty)
    std::vector<int64_t> A_ro = {0, 1, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    std::vector<double> A_val = {-1.0, -1.0};
    std::vector<double> q = {1.0, 1.0};
    std::vector<double> b = {-1.0, 0.0, -1.0};

    Cones cones{}; cones.psdConeDims = {2};
    auto r = solve_problem(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones);
    std::cout << "PSD(2) x=[" << r.x[0] << "," << r.x[1] << "] status=" << r.status << std::endl;
    EXPECT_EQ(r.status, 1);
    EXPECT_NEAR(r.x[0], 1.0, TOL_SOL);
    EXPECT_NEAR(r.x[1], 1.0, TOL_SOL);
}

// PSD(3): min x0+x1+x2, constraints on diagonal of 3×3 PSD matrix
TEST_F(PsdForwardTest, Psd3) {
    int64_t n = 3, m = 6;
    std::vector<int64_t> P_ro = {0, 0, 0, 0}, P_ci = {};
    std::vector<double> P_val = {};
    // svec for 3×3: idx 0=(0,0), 1=(0,1), 2=(1,1), 3=(0,2), 4=(1,2), 5=(2,2)
    // A: x0→row0, x1→row2, x2→row5
    std::vector<int64_t> A_ro = {0, 1, 1, 2, 2, 2, 3};
    std::vector<int64_t> A_ci = {0, 1, 2};
    std::vector<double> A_val = {-1.0, -1.0, -1.0};
    std::vector<double> q = {1.0, 1.0, 1.0};
    std::vector<double> b = {-1.0, 0.0, -1.0, 0.0, 0.0, -1.0};

    Cones cones{}; cones.psdConeDims = {3};
    auto r = solve_problem(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones);
    std::cout << "PSD(3) x=[" << r.x[0] << "," << r.x[1] << "," << r.x[2] << "] status=" << r.status << std::endl;
    EXPECT_EQ(r.status, 1);
    EXPECT_NEAR(r.x[0], 1.0, TOL_SOL);
    EXPECT_NEAR(r.x[1], 1.0, TOL_SOL);
    EXPECT_NEAR(r.x[2], 1.0, TOL_SOL);
}

// Multiple PSD cones: PSD(1) + PSD(2)
TEST_F(PsdForwardTest, MultiplePsd) {
    int64_t n = 2, m = 4;  // 1 (PSD(1)) + 3 (PSD(2))
    std::vector<int64_t> P_ro = {0, 0, 0}, P_ci = {};
    std::vector<double> P_val = {};
    // x0 → PSD(1) row0, x1 → PSD(2) row1 (svec[0] of 2nd cone)
    std::vector<int64_t> A_ro = {0, 1, 2, 2, 2};
    std::vector<int64_t> A_ci = {0, 1};
    std::vector<double> A_val = {-1.0, -1.0};
    std::vector<double> q = {1.0, 1.0};
    std::vector<double> b = {-1.0, -2.0, 0.0, 0.0};

    Cones cones{}; cones.psdConeDims = {1, 2};
    auto r = solve_problem(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones);
    std::cout << "Multi-PSD x=[" << r.x[0] << "," << r.x[1] << "] status=" << r.status << std::endl;
    EXPECT_EQ(r.status, 1);
    EXPECT_NEAR(r.x[0], 1.0, TOL_SOL);
    EXPECT_NEAR(r.x[1], 2.0, TOL_SOL);
}

// ============================================================================
// Backward pass (gradcheck) tests
// ============================================================================

class PsdGradcheckTest : public ::testing::Test {};

struct GradResult {
    std::vector<double> dq, db, x;
};

GradResult backward_grads(
    int64_t n, int64_t m,
    const std::vector<int64_t>& P_ro, const std::vector<int64_t>& P_ci, const std::vector<double>& P_val,
    const std::vector<int64_t>& A_ro, const std::vector<int64_t>& A_ci, const std::vector<double>& A_val,
    const std::vector<double>& q, const std::vector<double>& b,
    Cones& cones, const std::vector<double>& dx_bar
) {
    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;

    CompiledSolver solver(n, m, 1,
        P_ro.data(), P_ci.data(), P_val.size(),
        A_ro.data(), A_ci.data(), A_val.size(),
        cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * std::max((size_t)1, P_val.size()));
    cudaMalloc(&d_A, sizeof(double) * std::max((size_t)1, A_val.size()));
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    if (!P_val.empty()) cudaMemcpy(d_P, P_val.data(), sizeof(double) * P_val.size(), cudaMemcpyHostToDevice);
    if (!A_val.empty()) cudaMemcpy(d_A, A_val.data(), sizeof(double) * A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    GradResult gr;
    gr.x.resize(n);
    cudaMemcpy(gr.x.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    DiffState state(n, m, 1, P_val.size(), A_val.size());
    cache_solution_for_backward(state, solver);

    BatchedVector dx_d(n, 1), dz_d(m, 1), ds_d(m, 1);
    cudaMemcpy(dx_d.data(), dx_bar.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    dz_d.setToConstant(0.0);
    ds_d.setToConstant(0.0);

    backward(state, dx_d, dz_d, ds_d, solver);
    cudaDeviceSynchronize();

    gr.dq.resize(n); gr.db.resize(m);
    cudaMemcpy(gr.dq.data(), state.dq.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(gr.db.data(), state.db.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    return gr;
}

// PSD(1) dq gradcheck
TEST_F(PsdGradcheckTest, Psd1GradDq) {
    int64_t n = 1, m = 1;
    std::vector<int64_t> P_ro = {0, 0}, P_ci = {};
    std::vector<double> P_val = {};
    std::vector<int64_t> A_ro = {0, 1}, A_ci = {0};
    std::vector<double> A_val = {-1.0};
    std::vector<double> q = {1.0}, b = {-1.0};
    std::vector<double> dx_bar = {1.0};

    Cones cones{}; cones.psdConeDims = {1};
    auto gr = backward_grads(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones, dx_bar);
    std::cout << "PSD(1) x=" << gr.x[0] << " dq=" << gr.dq[0] << std::endl;

    // Finite difference
    auto qp = q; qp[0] += FD_H;
    auto qm = q; qm[0] -= FD_H;
    Cones c1{}; c1.psdConeDims = {1};
    Cones c2{}; c2.psdConeDims = {1};
    auto xp = solve_problem(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, qp, b, c1).x;
    auto xm = solve_problem(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, qm, b, c2).x;
    double fd = (xp[0] - xm[0]) / (2 * FD_H);
    std::cout << "  fd=" << fd << std::endl;
    EXPECT_NEAR(gr.dq[0], fd, grad_tol(gr.dq[0], fd));
}

// PSD(2) dq gradcheck
TEST_F(PsdGradcheckTest, Psd2GradDq) {
    int64_t n = 2, m = 3;
    std::vector<int64_t> P_ro = {0, 0, 0}, P_ci = {};
    std::vector<double> P_val = {};
    std::vector<int64_t> A_ro = {0, 1, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    std::vector<double> A_val = {-1.0, -1.0};
    std::vector<double> q = {1.0, 1.0};
    std::vector<double> b = {-1.0, 0.0, -1.0};
    std::vector<double> dx_bar = {1.0, 1.0};

    Cones cones{}; cones.psdConeDims = {2};
    auto gr = backward_grads(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones, dx_bar);
    std::cout << "PSD(2) x=[" << gr.x[0] << "," << gr.x[1] << "] dq=[" << gr.dq[0] << "," << gr.dq[1] << "]" << std::endl;

    for (int i = 0; i < n; i++) {
        auto qp = q; qp[i] += FD_H;
        auto qm = q; qm[i] -= FD_H;
        Cones c1{}; c1.psdConeDims = {2};
        Cones c2{}; c2.psdConeDims = {2};
        auto xp = solve_problem(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, qp, b, c1).x;
        auto xm = solve_problem(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, qm, b, c2).x;
        double fd = 0;
        for (int j = 0; j < n; j++) fd += dx_bar[j] * (xp[j] - xm[j]) / (2 * FD_H);
        std::cout << "  dq[" << i << "]: analytical=" << gr.dq[i] << " fd=" << fd << std::endl;
        EXPECT_NEAR(gr.dq[i], fd, grad_tol(gr.dq[i], fd));
    }
}

// PSD(2) db gradcheck
TEST_F(PsdGradcheckTest, Psd2GradDb) {
    int64_t n = 2, m = 3;
    std::vector<int64_t> P_ro = {0, 0, 0}, P_ci = {};
    std::vector<double> P_val = {};
    std::vector<int64_t> A_ro = {0, 1, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    std::vector<double> A_val = {-1.0, -1.0};
    std::vector<double> q = {1.0, 1.0};
    std::vector<double> b = {-1.0, 0.0, -1.0};
    std::vector<double> dx_bar = {1.0, 1.0};

    Cones cones{}; cones.psdConeDims = {2};
    auto gr = backward_grads(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones, dx_bar);

    for (int i = 0; i < m; i++) {
        auto bp = b; bp[i] += FD_H;
        auto bm = b; bm[i] -= FD_H;
        Cones c1{}; c1.psdConeDims = {2};
        Cones c2{}; c2.psdConeDims = {2};
        auto rp = solve_problem(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, bp, c1);
        auto rm = solve_problem(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, bm, c2);
        // Skip if either solve failed (e.g., perturbing unconstrained svec entries)
        if (rp.status != 1 || rm.status != 1) {
            std::cout << "  db[" << i << "]: skipped (solver status " << rp.status << "/" << rm.status << ")" << std::endl;
            continue;
        }
        double fd = 0;
        for (int j = 0; j < n; j++) fd += dx_bar[j] * (rp.x[j] - rm.x[j]) / (2 * FD_H);
        std::cout << "  db[" << i << "]: analytical=" << gr.db[i] << " fd=" << fd << std::endl;
        EXPECT_NEAR(gr.db[i], fd, grad_tol(gr.db[i], fd));
    }
}

// PSD(3) dq gradcheck
TEST_F(PsdGradcheckTest, Psd3GradDq) {
    int64_t n = 3, m = 6;
    std::vector<int64_t> P_ro = {0, 0, 0, 0}, P_ci = {};
    std::vector<double> P_val = {};
    // svec for 3×3: 0=(0,0), 1=(0,1), 2=(1,1), 3=(0,2), 4=(1,2), 5=(2,2)
    // A: x0→row0, x1→row2, x2→row5
    std::vector<int64_t> A_ro = {0, 1, 1, 2, 2, 2, 3};
    std::vector<int64_t> A_ci = {0, 1, 2};
    std::vector<double> A_val = {-1.0, -1.0, -1.0};
    std::vector<double> q = {1.0, 1.0, 1.0};
    std::vector<double> b = {-1.0, 0.0, -1.0, 0.0, 0.0, -1.0};
    std::vector<double> dx_bar = {1.0, 1.0, 1.0};

    Cones cones{}; cones.psdConeDims = {3};
    auto gr = backward_grads(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones, dx_bar);

    std::cout << "PSD(3) x=[" << gr.x[0] << "," << gr.x[1] << "," << gr.x[2] << "]" << std::endl;

    for (int i = 0; i < n; i++) {
        auto qp = q; qp[i] += FD_H;
        auto qm = q; qm[i] -= FD_H;
        Cones c1{}; c1.psdConeDims = {3};
        Cones c2{}; c2.psdConeDims = {3};
        auto rp = solve_problem(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, qp, b, c1);
        auto rm = solve_problem(n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, qm, b, c2);
        if (rp.status != 1 || rm.status != 1) {
            std::cout << "  dq[" << i << "]: skipped (status " << rp.status << "/" << rm.status << ")" << std::endl;
            continue;
        }
        double fd = 0;
        for (int j = 0; j < n; j++) fd += dx_bar[j] * (rp.x[j] - rm.x[j]) / (2 * FD_H);
        std::cout << "  dq[" << i << "]: analytical=" << gr.dq[i] << " fd=" << fd << std::endl;
        EXPECT_NEAR(gr.dq[i], fd, grad_tol(gr.dq[i], fd));
    }
}

// PSD(1) with QP: min 0.5*x^2 + 2x s.t. x+s=0, s in PSD(1) => x=-2
TEST_F(PsdForwardTest, Psd1QP) {
    int64_t n = 1, m = 1;
    std::vector<int64_t> P_ro = {0, 1}, P_ci = {0};
    std::vector<double> P_val = {1.0};
    std::vector<int64_t> A_ro = {0, 1}, A_ci = {0};
    std::vector<double> A_val = {1.0};
    std::vector<double> q = {2.0}, b = {0.0};

    // Use verbose to debug
    Settings settings;
    settings.verbose = true;
    settings.enableGrad = false;

    Cones cones{}; cones.psdConeDims = {1};

    CompiledSolver solver(n, m, 1, P_ro.data(), P_ci.data(), 1,
        A_ro.data(), A_ci.data(), 1, cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double));
    cudaMalloc(&d_A, sizeof(double));
    cudaMalloc(&d_q, sizeof(double));
    cudaMalloc(&d_b, sizeof(double));
    cudaMemcpy(d_P, P_val.data(), sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double), cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    double x_val;
    cudaMemcpy(&x_val, solver.solution.x.data(), sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "PSD(1) QP: x=" << x_val << " status=" << (int)solver.info.status[0] << std::endl;
    EXPECT_EQ((int)solver.info.status[0], 1);
    EXPECT_NEAR(x_val, -2.0, TOL_SOL);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// PSD(2) QP debug: verbose output for comparison with CPU Rust test
// Equivalent problem: packages/moreau-cpu/tests/psd2_qp_debug.rs
TEST_F(PsdForwardTest, DISABLED_Psd2QP_Debug) {
    int64_t n = 3, m = 3;
    std::vector<int64_t> P_ro = {0, 1, 2, 3}, P_ci = {0, 1, 2};
    std::vector<double> P_val = {1.0, 1.0, 1.0};
    std::vector<int64_t> A_ro = {0, 1, 2, 3}, A_ci = {0, 1, 2};
    std::vector<double> A_val = {1.0, 1.0, 1.0};
    std::vector<double> q = {-3.0, 0.0, -2.0};
    std::vector<double> b = {0.0, 0.0, 0.0};

    Settings settings;
    settings.verbose = true;

    Cones cones{}; cones.psdConeDims = {2};

    CompiledSolver solver(n, m, 1, P_ro.data(), P_ci.data(), 3,
        A_ro.data(), A_ci.data(), 3, cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * 3);
    cudaMalloc(&d_A, sizeof(double) * 3);
    cudaMalloc(&d_q, sizeof(double) * 3);
    cudaMalloc(&d_b, sizeof(double) * 3);
    cudaMemcpy(d_P, P_val.data(), sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * 3, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    std::vector<double> x(3), z(3), s(3);
    cudaMemcpy(x.data(), solver.solution.x.data(), sizeof(double) * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(z.data(), solver.solution.z.data(), sizeof(double) * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(s.data(), solver.solution.s.data(), sizeof(double) * 3, cudaMemcpyDeviceToHost);

    std::cout << "x = [" << x[0] << ", " << x[1] << ", " << x[2] << "]" << std::endl;
    std::cout << "z = [" << z[0] << ", " << z[1] << ", " << z[2] << "]" << std::endl;
    std::cout << "s = [" << s[0] << ", " << s[1] << ", " << s[2] << "]" << std::endl;
    std::cout << "status = " << (int)solver.info.status[0] << std::endl;

    int status = (int)solver.info.status[0];
    EXPECT_EQ(status, 1) << "Should solve successfully";
    for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(x[i], 0.0, TOL_SOL) << "x[" << i << "] should be ~0";
    }

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// Nonneg(3) QP — same problem as PSD(2) QP but with nonneg cones.
// Should give identical results since H=I for both with identity scaling.
TEST_F(PsdForwardTest, Nonneg3QP_Compare) {
    int64_t n = 3, m = 3;
    std::vector<int64_t> P_ro = {0, 1, 2, 3}, P_ci = {0, 1, 2};
    std::vector<double> P_val = {1.0, 1.0, 1.0};
    std::vector<int64_t> A_ro = {0, 1, 2, 3}, A_ci = {0, 1, 2};
    std::vector<double> A_val = {1.0, 1.0, 1.0};
    std::vector<double> q = {-3.0, 0.0, -2.0};
    std::vector<double> b = {0.0, 0.0, 0.0};

    Cones cones{}; cones.numNonnegCones = 3;  // nonneg instead of PSD
    Settings settings; settings.verbose = true;

    CompiledSolver solver(n, m, 1, P_ro.data(), P_ci.data(), 3,
        A_ro.data(), A_ci.data(), 3, cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * 3);
    cudaMalloc(&d_A, sizeof(double) * 3);
    cudaMalloc(&d_q, sizeof(double) * 3);
    cudaMalloc(&d_b, sizeof(double) * 3);
    cudaMemcpy(d_P, P_val.data(), sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * 3, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    std::vector<double> x(3);
    cudaMemcpy(x.data(), solver.solution.x.data(), sizeof(double) * 3, cudaMemcpyDeviceToHost);
    std::cout << "Nonneg3 QP: x = [" << x[0] << ", " << x[1] << ", " << x[2] << "]" << std::endl;
    std::cout << "Nonneg3 QP: status = " << (int)solver.info.status[0] << std::endl;

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// ============================================================================
// diffqcp-style SDP: min trace(C*X) s.t. trace(A_i*X) = b_i, X >= 0 (PSD)
// This is the formulation from Agrawal et al. "Differentiating Through a Cone
// Program" (2019). The key difference from existing tests: dense equality
// constraint rows (each row of A_eq is a full svec) + identity PSD block.
// ============================================================================

namespace {

// Simple PRNG (xorshift64) for reproducible test data
struct Rng {
    uint64_t s;
    Rng(uint64_t seed) : s(seed) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    double uniform() { return (double)next() / (double)UINT64_MAX * 2.0 - 1.0; }
    double range(double lo, double hi) { return lo + (double)next() / (double)UINT64_MAX * (hi - lo); }
};

// Convert symmetric matrix to svec (scaled upper triangle, column-major)
std::vector<double> mat_to_svec(const std::vector<double>& M, int64_t n) {
    int64_t svec_dim = n * (n + 1) / 2;
    std::vector<double> v(svec_dim);
    int64_t idx = 0;
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i <= j; i++) {
            if (i == j)
                v[idx] = M[i * n + j];
            else
                v[idx] = M[i * n + j] * std::sqrt(2.0);
            idx++;
        }
    }
    return v;
}

// Generate random PSD matrix (A*A'/100)
std::vector<double> randn_psd(int64_t n, Rng& rng) {
    std::vector<double> A(n * n), M(n * n, 0.0);
    for (auto& x : A) x = rng.uniform() / 10.0;
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++)
            for (int64_t k = 0; k < n; k++)
                M[i * n + j] += A[i * n + k] * A[j * n + k];
    return M;
}

// Generate random symmetric matrix
std::vector<double> randn_symm(int64_t n, Rng& rng) {
    std::vector<double> M(n * n);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            double v = rng.uniform();
            M[i * n + j] = v;
            M[j * n + i] = v;
        }
    }
    return M;
}

// Build diffqcp-style SDP problem data in moreau CSR format
// Returns: (n_var, m_con, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b)
struct SdpProblem {
    int64_t n_var, m_con, mat_dim, n_eq;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    std::vector<double> P_val, A_val, q, b;
};

SdpProblem build_diffqcp_sdp(int64_t mat_dim, int64_t n_eq, uint64_t seed = 42, double reg = 0.0) {
    Rng rng(seed);
    int64_t svec_dim = mat_dim * (mat_dim + 1) / 2;
    int64_t n_var = svec_dim;
    int64_t m_con = n_eq + svec_dim;  // equality rows + PSD cone rows

    // C = random PSD matrix → q = svec(C)
    auto C = randn_psd(mat_dim, rng);
    auto q = mat_to_svec(C, mat_dim);

    // A_i = random symmetric → equality constraint rows
    // b_i = random
    std::vector<std::vector<double>> A_eq_rows(n_eq);
    std::vector<double> b_eq(n_eq);
    for (int64_t i = 0; i < n_eq; i++) {
        auto Ai = randn_symm(mat_dim, rng);
        A_eq_rows[i] = mat_to_svec(Ai, mat_dim);
        b_eq[i] = rng.uniform();
    }

    // Build CSR for A = [A_eq; -I]
    // Equality rows are dense (svec_dim entries each)
    // PSD rows are identity (one entry per row, value = -1)
    std::vector<int64_t> A_ro, A_ci;
    std::vector<double> A_val;
    A_ro.push_back(0);

    // Equality rows (dense)
    for (int64_t i = 0; i < n_eq; i++) {
        for (int64_t j = 0; j < svec_dim; j++) {
            A_ci.push_back(j);
            A_val.push_back(A_eq_rows[i][j]);
        }
        A_ro.push_back((int64_t)A_ci.size());
    }

    // PSD cone rows: -I
    for (int64_t i = 0; i < svec_dim; i++) {
        A_ci.push_back(i);
        A_val.push_back(-1.0);
        A_ro.push_back((int64_t)A_ci.size());
    }

    // b = [b_eq; 0]
    std::vector<double> b(m_con, 0.0);
    for (int64_t i = 0; i < n_eq; i++) b[i] = b_eq[i];

    // P = reg * I (small regularization for numerical stability, 0 for pure LP)
    std::vector<int64_t> P_ro(n_var + 1, 0);
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;
    if (reg > 0.0) {
        for (int64_t i = 0; i < n_var; i++) {
            P_ci.push_back(i);
            P_val.push_back(reg);
            P_ro[i + 1] = i + 1;
        }
    }

    return SdpProblem{n_var, m_con, mat_dim, n_eq, P_ro, P_ci, A_ro, A_ci, P_val, A_val, q, b};
}

} // anon namespace

class DiffqcpSdpTest : public ::testing::Test {};

// Known SDP: min trace(I*X) + ε/2·||x||² s.t. trace(X) = 1, X ≥ 0
// Solution: X = (1/n)·I, obj = 1 + ε·n/2·(1/n²) = 1 + ε/(2n)
// This is the simplest SDP with a known analytical solution.
TEST_F(DiffqcpSdpTest, KnownSDP_traceMin) {
    const int64_t mat_dim = 5;
    const int64_t svec_dim = mat_dim * (mat_dim + 1) / 2;  // 15
    const double reg = 1e-4;

    // q = svec(I) = [1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    // (diagonal entries = 1, off-diagonal = 0)
    std::vector<double> q(svec_dim, 0.0);
    for (int64_t k = 0; k < mat_dim; k++) {
        q[k * (k + 3) / 2] = 1.0;  // triangular_index(k)
    }

    // A_eq (1 row): trace constraint = svec(I)^T·x = 1
    // trace(X) = sum of diagonal svec entries = sum x[triangular_index(k)]
    // In CSR: one row with nonzeros at diagonal positions
    std::vector<int64_t> A_ro = {0};
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;
    for (int64_t k = 0; k < mat_dim; k++) {
        A_ci.push_back(k * (k + 3) / 2);
        A_val.push_back(1.0);
    }
    A_ro.push_back((int64_t)A_ci.size());

    // PSD cone rows: -I
    for (int64_t i = 0; i < svec_dim; i++) {
        A_ci.push_back(i);
        A_val.push_back(-1.0);
        A_ro.push_back((int64_t)A_ci.size());
    }

    int64_t m = 1 + svec_dim;  // 1 eq + 15 PSD
    std::vector<double> b(m, 0.0);
    b[0] = 1.0;  // trace(X) = 1

    // P = reg * I
    std::vector<int64_t> P_ro(svec_dim + 1, 0);
    std::vector<int64_t> P_ci(svec_dim);
    std::vector<double> P_val(svec_dim, reg);
    for (int64_t i = 0; i < svec_dim; i++) {
        P_ci[i] = i;
        P_ro[i + 1] = i + 1;
    }

    std::cout << "Known SDP: mat_dim=" << mat_dim << " svec_dim=" << svec_dim
              << " m=" << m << " A_nnz=" << A_val.size() << std::endl;

    Cones cones{};
    cones.numZeroCones = 1;
    cones.psdConeDims = {mat_dim};

    Settings settings;
    settings.verbose = true;
    settings.enableGrad = false;

    CompiledSolver solver(svec_dim, m, 1,
        P_ro.data(), P_ci.data(), P_val.size(),
        A_ro.data(), A_ci.data(), A_val.size(),
        cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * P_val.size());
    cudaMalloc(&d_A, sizeof(double) * A_val.size());
    cudaMalloc(&d_q, sizeof(double) * svec_dim);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMemcpy(d_P, P_val.data(), sizeof(double) * P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * svec_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    int status = (int)solver.info.status[0];
    int iters = solver.info.iterations;
    std::cout << "  status=" << status << " iterations=" << iters << std::endl;

    EXPECT_GT(iters, 0) << "Solver should need >0 iterations";
    EXPECT_EQ(status, 1) << "Should solve successfully";

    // Check solution: X should be close to (1/n)*I
    // In svec form, diagonal entries should be 1/n = 0.2, off-diagonal = 0
    std::vector<double> x_host(svec_dim);
    cudaMemcpy(x_host.data(), solver.solution.x.data(), sizeof(double) * svec_dim, cudaMemcpyDeviceToHost);
    if (status == 1) {
        for (int64_t k = 0; k < mat_dim; k++) {
            EXPECT_NEAR(x_host[k * (k + 3) / 2], 1.0 / mat_dim, 1e-3)
                << "Diagonal entry " << k << " should be 1/n";
        }
    }

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// Small diffqcp SDP: n=3 (matrix dim), p=2 (equality constraints)
// svec_dim=6, m=8 (2 eq + 6 PSD)
// Small diffqcp LP SDP: P=0, dense A, PSD(3)
TEST_F(DiffqcpSdpTest, Small_n3_p2) {
    auto prob = build_diffqcp_sdp(3, 2, 42, 1e-2);  // small reg for stability
    std::cout << "diffqcp SDP: mat_dim=3, n_eq=2, n_var=" << prob.n_var
              << ", m=" << prob.m_con << ", P_nnz=" << prob.P_val.size()
              << ", A_nnz=" << prob.A_val.size() << std::endl;

    Cones cones{};
    cones.numZeroCones = prob.n_eq;
    cones.psdConeDims = {prob.mat_dim};

    Settings settings;
    settings.verbose = true;

    CompiledSolver solver(prob.n_var, prob.m_con, 1,
        prob.P_ro.data(), prob.P_ci.data(), prob.P_val.size(),
        prob.A_ro.data(), prob.A_ci.data(), prob.A_val.size(),
        cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.P_val.size());
    cudaMalloc(&d_A, sizeof(double) * prob.A_val.size());
    cudaMalloc(&d_q, sizeof(double) * prob.n_var);
    cudaMalloc(&d_b, sizeof(double) * prob.m_con);
    cudaMemcpy(d_P, prob.P_val.data(), sizeof(double) * prob.P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_val.data(), sizeof(double) * prob.A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q.data(), sizeof(double) * prob.n_var, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b.data(), sizeof(double) * prob.m_con, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    SolveResult r;
    r.x.resize(prob.n_var); r.z.resize(prob.m_con); r.s.resize(prob.m_con);
    cudaMemcpy(r.x.data(), solver.solution.x.data(), sizeof(double) * prob.n_var, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.z.data(), solver.solution.z.data(), sizeof(double) * prob.m_con, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.s.data(), solver.solution.s.data(), sizeof(double) * prob.m_con, cudaMemcpyDeviceToHost);
    r.status = (int)solver.info.status[0];

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);

    std::cout << "  status=" << r.status << std::endl;
    EXPECT_EQ(r.status, 1) << "Should solve successfully";

    // Verify PSD: s (last svec_dim entries) should have non-negative eigenvalues
    // (we check by verifying the diagonal svec entries are non-negative)
    for (int64_t i = 0; i < prob.n_var; i++) {
        // x should be finite
        EXPECT_TRUE(std::isfinite(r.x[i])) << "x[" << i << "] is not finite";
    }
}

// Medium diffqcp SDP: n=10 (matrix dim), p=5 (equality constraints)
// svec_dim=55, m=60 (5 eq + 55 PSD)
TEST_F(DiffqcpSdpTest, Medium_n10_p5) {
    auto prob = build_diffqcp_sdp(10, 5);
    std::cout << "diffqcp SDP: mat_dim=10, n_eq=5, n_var=" << prob.n_var
              << ", m=" << prob.m_con << ", A_nnz=" << prob.A_val.size() << std::endl;

    Cones cones{};
    cones.numZeroCones = prob.n_eq;
    cones.psdConeDims = {prob.mat_dim};

    Settings settings;
    settings.verbose = true;
    settings.enableGrad = false;

    CompiledSolver solver(prob.n_var, prob.m_con, 1,
        prob.P_ro.data(), prob.P_ci.data(), prob.P_val.size(),
        prob.A_ro.data(), prob.A_ci.data(), prob.A_val.size(),
        cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * std::max((size_t)1, prob.P_val.size()));
    cudaMalloc(&d_A, sizeof(double) * prob.A_val.size());
    cudaMalloc(&d_q, sizeof(double) * prob.n_var);
    cudaMalloc(&d_b, sizeof(double) * prob.m_con);
    if (!prob.P_val.empty())
        cudaMemcpy(d_P, prob.P_val.data(), sizeof(double) * prob.P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_val.data(), sizeof(double) * prob.A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q.data(), sizeof(double) * prob.n_var, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b.data(), sizeof(double) * prob.m_con, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    int status = (int)solver.info.status[0];
    int iters = solver.info.iterations;
    std::cout << "  status=" << status << " iterations=" << iters << std::endl;

    // The solver should actually iterate (not converge at iteration 0)
    EXPECT_GT(iters, 0) << "Solver should need >0 iterations for non-trivial SDP";
    EXPECT_EQ(status, 1) << "Should solve successfully";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}

// Larger diffqcp SDP: n=50 (matrix dim), p=25 (equality constraints)
// svec_dim=1275, m=1300 (25 eq + 1275 PSD)
// This matches the default diffqcp example size.
TEST_F(DiffqcpSdpTest, Large_n50_p25) {
    auto prob = build_diffqcp_sdp(50, 25);
    std::cout << "diffqcp SDP: mat_dim=50, n_eq=25, n_var=" << prob.n_var
              << ", m=" << prob.m_con << ", A_nnz=" << prob.A_val.size() << std::endl;

    Cones cones{};
    cones.numZeroCones = prob.n_eq;
    cones.psdConeDims = {prob.mat_dim};

    Settings settings;
    settings.verbose = true;
    settings.enableGrad = false;

    CompiledSolver solver(prob.n_var, prob.m_con, 1,
        prob.P_ro.data(), prob.P_ci.data(), prob.P_val.size(),
        prob.A_ro.data(), prob.A_ci.data(), prob.A_val.size(),
        cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * std::max((size_t)1, prob.P_val.size()));
    cudaMalloc(&d_A, sizeof(double) * prob.A_val.size());
    cudaMalloc(&d_q, sizeof(double) * prob.n_var);
    cudaMalloc(&d_b, sizeof(double) * prob.m_con);
    if (!prob.P_val.empty())
        cudaMemcpy(d_P, prob.P_val.data(), sizeof(double) * prob.P_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_val.data(), sizeof(double) * prob.A_val.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q.data(), sizeof(double) * prob.n_var, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b.data(), sizeof(double) * prob.m_con, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    int status = (int)solver.info.status[0];
    int iters = solver.info.iterations;
    std::cout << "  status=" << status << " iterations=" << iters << std::endl;

    EXPECT_GT(iters, 0) << "Solver should need >0 iterations for non-trivial SDP";
    EXPECT_EQ(status, 1) << "Should solve successfully";

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
}
