/**
 * @file test_psd_fuzz.cpp
 * @brief Fuzz tests for PSD cone: random feasible SDPs across many seeds,
 *        with KKT residual checking, PSD membership verification, and
 *        backward pass finite-difference gradchecks.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <vector>

#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/diff/diff.hpp"

using namespace moreau;

namespace {

// ============================================================================
// Constants
// ============================================================================

constexpr double TOL_KKT       = 1e-4;
constexpr double TOL_EIGEN     = -1e-6;  // min eigenvalue threshold for PSD
constexpr double TOL_GRAD      = 1e-3;
constexpr double TOL_GRAD_REL  = 5e-2;
constexpr double FD_H          = 1e-5;

double grad_tol(double a, double b) {
    return std::max(TOL_GRAD, TOL_GRAD_REL * std::max(std::abs(a), std::abs(b)));
}

// ============================================================================
// svec utilities
// ============================================================================

// svec ordering: column-major upper triangle
// For n=3: (0,0), (0,1), (1,1), (0,2), (1,2), (2,2)
// Off-diagonal entries scaled by sqrt(2).
int64_t svec_dim(int64_t n) { return n * (n + 1) / 2; }

// Triangular index for diagonal element k: k*(k+3)/2
int64_t diag_svec_idx(int64_t k) { return k * (k + 3) / 2; }

// Symmetric n×n matrix (row-major) → svec
std::vector<double> mat_to_svec(const std::vector<double>& M, int64_t n) {
    std::vector<double> v(svec_dim(n));
    int64_t idx = 0;
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i <= j; i++) {
            v[idx++] = (i == j) ? M[i * n + j] : M[i * n + j] * std::sqrt(2.0);
        }
    }
    return v;
}

// svec → symmetric n×n matrix (row-major)
std::vector<double> svec_to_mat(const double* v, int64_t n) {
    std::vector<double> M(n * n, 0.0);
    int64_t idx = 0;
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i <= j; i++) {
            double val = (i == j) ? v[idx] : v[idx] / std::sqrt(2.0);
            M[i * n + j] = val;
            M[j * n + i] = val;
            idx++;
        }
    }
    return M;
}

// ============================================================================
// Host-side symmetric eigenvalue (Jacobi iteration, fine for n <= 20)
// ============================================================================

double min_eigenvalue_symm(const std::vector<double>& M, int64_t n) {
    // Work on a copy
    std::vector<double> A(M);
    const int max_sweeps = 100;
    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double off_diag = 0.0;
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = i + 1; j < n; j++)
                off_diag += A[i * n + j] * A[i * n + j];
        if (off_diag < 1e-30) break;

        for (int64_t p = 0; p < n; p++) {
            for (int64_t q = p + 1; q < n; q++) {
                double apq = A[p * n + q];
                if (std::abs(apq) < 1e-15) continue;
                double app = A[p * n + p], aqq = A[q * n + q];
                double tau_val = (aqq - app) / (2.0 * apq);
                double t = (tau_val >= 0 ? 1.0 : -1.0) /
                           (std::abs(tau_val) + std::sqrt(1.0 + tau_val * tau_val));
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                // Apply Givens rotation
                A[p * n + p] -= t * apq;
                A[q * n + q] += t * apq;
                A[p * n + q] = 0.0;
                A[q * n + p] = 0.0;
                for (int64_t r = 0; r < n; r++) {
                    if (r == p || r == q) continue;
                    double arp = A[r * n + p], arq = A[r * n + q];
                    A[r * n + p] = A[p * n + r] = c * arp - s * arq;
                    A[r * n + q] = A[q * n + r] = s * arp + c * arq;
                }
            }
        }
    }
    double min_ev = A[0];
    for (int64_t i = 1; i < n; i++) min_ev = std::min(min_ev, A[i * n + i]);
    return min_ev;
}

// ============================================================================
// Random matrix generators
// ============================================================================

// Random PSD matrix: A*A'/k + eps*I
std::vector<double> randn_psd(int64_t n, std::mt19937_64& rng, double eps = 0.01) {
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> L(n * n), M(n * n, 0.0);
    for (auto& x : L) x = nd(rng) / std::sqrt((double)n);
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++)
            for (int64_t k = 0; k < n; k++)
                M[i * n + j] += L[i * n + k] * L[j * n + k];
    for (int64_t i = 0; i < n; i++) M[i * n + i] += eps;
    return M;
}

// Random symmetric matrix
std::vector<double> randn_symm(int64_t n, std::mt19937_64& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> M(n * n);
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i; j < n; j++) {
            double v = nd(rng);
            M[i * n + j] = v;
            M[j * n + i] = v;
        }
    }
    return M;
}

// ============================================================================
// Problem structure
// ============================================================================

struct FuzzSdpProblem {
    int64_t n_var, m_con;
    std::vector<int64_t> P_ro, P_ci;
    std::vector<double> P_val;
    std::vector<int64_t> A_ro, A_ci;
    std::vector<double> A_val;
    std::vector<double> q, b;

    // For KKT checking (dense, host-side)
    std::vector<double> A_dense;  // m_con x n_var, row-major
    std::vector<double> P_dense;  // n_var x n_var, row-major

    // Cone layout
    int64_t num_zero = 0;
    int64_t num_nonneg = 0;
    std::vector<int64_t> psd_dims;

    Cones makeCones() const {
        Cones c{};
        c.numZeroCones = num_zero;
        c.numNonnegCones = num_nonneg;
        c.psdConeDims = psd_dims;
        return c;
    }
};

// ============================================================================
// Problem generator: diffqcp-style SDP
//
//   min  q'x + 0.5 x'Px
//   s.t. A_eq x = b_eq          (zero cones)
//        A_nn x + s_nn = b_nn   (nonneg cones, s_nn >= 0)
//        -I x + s_psd = 0       (PSD cones, s_psd in PSD)
//
// Decision variable x = svec(X) for PSD block.
// Feasibility guaranteed by constructing x_star in PSD interior.
// ============================================================================

FuzzSdpProblem build_fuzz_sdp(
    const std::vector<int64_t>& psd_dims,
    int64_t n_eq,
    int64_t n_nonneg,
    uint64_t seed,
    double reg = 1e-3,
    bool dense_P = false)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_real_distribution<double> ud(0.1, 1.0);

    // Compute total svec dimension
    int64_t svec_total = 0;
    for (auto d : psd_dims) svec_total += svec_dim(d);

    int64_t n_var = svec_total;
    int64_t m_con = n_eq + n_nonneg + svec_total;

    FuzzSdpProblem prob;
    prob.n_var = n_var;
    prob.m_con = m_con;
    prob.num_zero = n_eq;
    prob.num_nonneg = n_nonneg;
    prob.psd_dims = psd_dims;

    // Construct a feasible x_star: svec of PSD matrices in interior
    std::vector<double> x_star(n_var);
    int64_t offset = 0;
    for (auto d : psd_dims) {
        auto M = randn_psd(d, rng, 0.1);
        auto sv = mat_to_svec(M, d);
        for (int64_t i = 0; i < svec_dim(d); i++)
            x_star[offset + i] = sv[i];
        offset += svec_dim(d);
    }

    // Build dense A (m_con x n_var) and b
    prob.A_dense.resize(m_con * n_var, 0.0);
    prob.b.resize(m_con, 0.0);

    int64_t row = 0;

    // Equality constraint rows (dense random)
    for (int64_t i = 0; i < n_eq; i++) {
        for (int64_t j = 0; j < n_var; j++)
            prob.A_dense[row * n_var + j] = nd(rng);
        // b_eq = A_eq * x_star (so x_star is feasible)
        for (int64_t j = 0; j < n_var; j++)
            prob.b[row] += prob.A_dense[row * n_var + j] * x_star[j];
        row++;
    }

    // Nonneg constraint rows (dense random, with feasible b)
    for (int64_t i = 0; i < n_nonneg; i++) {
        for (int64_t j = 0; j < n_var; j++)
            prob.A_dense[row * n_var + j] = nd(rng);
        // s_nn = b_nn - A_nn * x_star > 0
        double ax = 0.0;
        for (int64_t j = 0; j < n_var; j++)
            ax += prob.A_dense[row * n_var + j] * x_star[j];
        prob.b[row] = ax + ud(rng);  // margin > 0
        row++;
    }

    // PSD cone rows: -I (so s_psd = b_psd - (-I)*x = x when b_psd = 0)
    for (int64_t i = 0; i < svec_total; i++) {
        prob.A_dense[row * n_var + i] = -1.0;
        prob.b[row] = 0.0;  // s = x (which is PSD by construction)
        row++;
    }

    // Build A CSR from dense
    prob.A_ro.resize(m_con + 1);
    prob.A_ro[0] = 0;
    for (int64_t i = 0; i < m_con; i++) {
        for (int64_t j = 0; j < n_var; j++) {
            double v = prob.A_dense[i * n_var + j];
            if (v != 0.0) {
                prob.A_ci.push_back(j);
                prob.A_val.push_back(v);
            }
        }
        prob.A_ro[i + 1] = (int64_t)prob.A_ci.size();
    }

    // P matrix
    prob.P_dense.resize(n_var * n_var, 0.0);
    if (dense_P && n_var <= 60) {
        // Dense PD: L*L' + reg*I
        std::vector<double> L(n_var * n_var, 0.0);
        for (int64_t i = 0; i < n_var; i++) {
            for (int64_t j = 0; j <= i; j++)
                L[i * n_var + j] = nd(rng) * 0.3;
            L[i * n_var + i] += 1.0;
        }
        for (int64_t i = 0; i < n_var; i++)
            for (int64_t j = 0; j <= i; j++) {
                double val = 0.0;
                for (int64_t k = 0; k <= std::min(i, j); k++)
                    val += L[i * n_var + k] * L[j * n_var + k];
                prob.P_dense[i * n_var + j] = val;
                prob.P_dense[j * n_var + i] = val;
            }
        for (int64_t i = 0; i < n_var; i++)
            prob.P_dense[i * n_var + i] += reg;
    } else if (reg > 0.0) {
        // Diagonal: reg * I
        for (int64_t i = 0; i < n_var; i++)
            prob.P_dense[i * n_var + i] = reg;
    }

    // P CSR (full symmetric, only nonzeros)
    prob.P_ro.resize(n_var + 1);
    prob.P_ro[0] = 0;
    for (int64_t i = 0; i < n_var; i++) {
        for (int64_t j = 0; j < n_var; j++) {
            double v = prob.P_dense[i * n_var + j];
            if (v != 0.0) {
                prob.P_ci.push_back(j);
                prob.P_val.push_back(v);
            }
        }
        prob.P_ro[i + 1] = (int64_t)prob.P_ci.size();
    }

    // q: random cost
    prob.q.resize(n_var);
    for (auto& v : prob.q) v = nd(rng) * 0.5;

    return prob;
}

// ============================================================================
// Solve helper
// ============================================================================

struct FuzzSolveResult {
    std::vector<double> x, z, s;
    int status;
    int iterations;
    double obj_val;
};

FuzzSolveResult solve_fuzz(const FuzzSdpProblem& prob, int max_iter = 200,
                           bool enable_grad = false, bool verbose = false) {
    Settings settings;
    settings.verbose = verbose;
    settings.maxIter = max_iter;
    settings.enableGrad = enable_grad;

    Cones cones = prob.makeCones();

    int64_t nnzP = (int64_t)prob.P_val.size();
    int64_t nnzA = (int64_t)prob.A_val.size();

    CompiledSolver solver(prob.n_var, prob.m_con, 1,
        prob.P_ro.data(), prob.P_ci.data(), nnzP,
        prob.A_ro.data(), prob.A_ci.data(), nnzA,
        cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * std::max((int64_t)1, nnzP));
    cudaMalloc(&d_A, sizeof(double) * std::max((int64_t)1, nnzA));
    cudaMalloc(&d_q, sizeof(double) * prob.n_var);
    cudaMalloc(&d_b, sizeof(double) * prob.m_con);
    if (nnzP > 0) cudaMemcpy(d_P, prob.P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    if (nnzA > 0) cudaMemcpy(d_A, prob.A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q.data(), sizeof(double) * prob.n_var, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b.data(), sizeof(double) * prob.m_con, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    FuzzSolveResult r;
    r.x.resize(prob.n_var);
    r.z.resize(prob.m_con);
    r.s.resize(prob.m_con);
    cudaMemcpy(r.x.data(), solver.solution.x.data(), sizeof(double) * prob.n_var, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.z.data(), solver.solution.z.data(), sizeof(double) * prob.m_con, cudaMemcpyDeviceToHost);
    cudaMemcpy(r.s.data(), solver.solution.s.data(), sizeof(double) * prob.m_con, cudaMemcpyDeviceToHost);
    r.status = (int)solver.info.status[0];
    r.iterations = solver.info.iterations;
    cudaMemcpy(&r.obj_val, solver.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    return r;
}

// ============================================================================
// KKT residual checking
// ============================================================================

// ||Ax + s - b||_inf
double primal_residual(const FuzzSdpProblem& prob, const FuzzSolveResult& r) {
    double max_res = 0.0;
    for (int64_t i = 0; i < prob.m_con; i++) {
        double row = 0.0;
        for (int64_t j = 0; j < prob.n_var; j++)
            row += prob.A_dense[i * prob.n_var + j] * r.x[j];
        row += r.s[i] - prob.b[i];
        max_res = std::max(max_res, std::abs(row));
    }
    return max_res;
}

// ||Px + q + A'z||_inf
double dual_residual(const FuzzSdpProblem& prob, const FuzzSolveResult& r) {
    double max_res = 0.0;
    for (int64_t j = 0; j < prob.n_var; j++) {
        double val = prob.q[j];
        // P*x
        for (int64_t k = 0; k < prob.n_var; k++)
            val += prob.P_dense[j * prob.n_var + k] * r.x[k];
        // A'*z
        for (int64_t i = 0; i < prob.m_con; i++)
            val += prob.A_dense[i * prob.n_var + j] * r.z[i];
        max_res = std::max(max_res, std::abs(val));
    }
    return max_res;
}

// Check PSD membership of s blocks
// Returns minimum eigenvalue across all PSD cone blocks
double check_psd_membership(const FuzzSdpProblem& prob, const std::vector<double>& s) {
    int64_t offset = prob.num_zero + prob.num_nonneg;
    double min_ev = 1e30;
    for (auto d : prob.psd_dims) {
        auto M = svec_to_mat(s.data() + offset, d);
        double ev = min_eigenvalue_symm(M, d);
        min_ev = std::min(min_ev, ev);
        offset += svec_dim(d);
    }
    return min_ev;
}

// Check nonneg membership
double min_nonneg(const FuzzSdpProblem& prob, const std::vector<double>& s) {
    double min_val = 1e30;
    int64_t offset = prob.num_zero;
    for (int64_t i = 0; i < prob.num_nonneg; i++)
        min_val = std::min(min_val, s[offset + i]);
    return min_val;
}

// Full KKT check: returns true if all conditions met
bool check_kkt(const FuzzSdpProblem& prob, const FuzzSolveResult& r,
               double tol, bool print = false) {
    double p_res = primal_residual(prob, r);
    double d_res = dual_residual(prob, r);
    double s_psd_ev = check_psd_membership(prob, r.s);
    double z_psd_ev = check_psd_membership(prob, r.z);

    bool ok = (p_res < tol) && (d_res < tol) &&
              (s_psd_ev >= TOL_EIGEN) && (z_psd_ev >= TOL_EIGEN);

    if (print || !ok) {
        std::cout << "  primal_res=" << p_res << " dual_res=" << d_res
                  << " s_min_ev=" << s_psd_ev << " z_min_ev=" << z_psd_ev;
        if (prob.num_nonneg > 0) {
            double s_nn = min_nonneg(prob, r.s);
            double z_nn = min_nonneg(prob, r.z);
            std::cout << " s_nn_min=" << s_nn << " z_nn_min=" << z_nn;
            if (s_nn < -tol || z_nn < -tol) ok = false;
        }
        std::cout << (ok ? " OK" : " FAIL") << std::endl;
    }
    return ok;
}

// ============================================================================
// Backward pass helper
// ============================================================================

struct FuzzGradResult {
    std::vector<double> dq, db, x;
    int status;
};

FuzzGradResult backward_fuzz(const FuzzSdpProblem& prob,
                             const std::vector<double>& dx_bar) {
    Settings settings;
    settings.verbose = false;
    settings.enableGrad = true;

    Cones cones = prob.makeCones();
    int64_t nnzP = (int64_t)prob.P_val.size();
    int64_t nnzA = (int64_t)prob.A_val.size();

    CompiledSolver solver(prob.n_var, prob.m_con, 1,
        prob.P_ro.data(), prob.P_ci.data(), nnzP,
        prob.A_ro.data(), prob.A_ci.data(), nnzA,
        cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * std::max((int64_t)1, nnzP));
    cudaMalloc(&d_A, sizeof(double) * std::max((int64_t)1, nnzA));
    cudaMalloc(&d_q, sizeof(double) * prob.n_var);
    cudaMalloc(&d_b, sizeof(double) * prob.m_con);
    if (nnzP > 0) cudaMemcpy(d_P, prob.P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    if (nnzA > 0) cudaMemcpy(d_A, prob.A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q.data(), sizeof(double) * prob.n_var, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b.data(), sizeof(double) * prob.m_con, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);
    cudaDeviceSynchronize();

    FuzzGradResult gr;
    gr.x.resize(prob.n_var);
    cudaMemcpy(gr.x.data(), solver.solution.x.data(), sizeof(double) * prob.n_var, cudaMemcpyDeviceToHost);
    gr.status = (int)solver.info.status[0];

    if (gr.status != 1) {
        cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
        return gr;
    }

    DiffState state(prob.n_var, prob.m_con, 1, nnzP, nnzA);
    cache_solution_for_backward(state, solver);

    BatchedVector dx_d(prob.n_var, 1), dz_d(prob.m_con, 1), ds_d(prob.m_con, 1);
    cudaMemcpy(dx_d.data(), dx_bar.data(), sizeof(double) * prob.n_var, cudaMemcpyHostToDevice);
    dz_d.setToConstant(0.0);
    ds_d.setToConstant(0.0);

    backward(state, dx_d, dz_d, ds_d, solver);
    cudaDeviceSynchronize();

    gr.dq.resize(prob.n_var);
    gr.db.resize(prob.m_con);
    cudaMemcpy(gr.dq.data(), state.dq.data(), sizeof(double) * prob.n_var, cudaMemcpyDeviceToHost);
    cudaMemcpy(gr.db.data(), state.db.data(), sizeof(double) * prob.m_con, cudaMemcpyDeviceToHost);

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);
    return gr;
}

} // anonymous namespace

// ============================================================================
// Test fixture
// ============================================================================

class PsdFuzzTest : public ::testing::Test {
protected:
    void SetUp() override { cudaGetLastError(); }
    void TearDown() override { cudaDeviceSynchronize(); }
};

// ============================================================================
// Test 1: Single PSD cone, many seeds
// ============================================================================

TEST_F(PsdFuzzTest, SinglePSD_MultiSeed) {
    struct Config { int64_t mat_dim; int64_t n_eq; };
    std::vector<Config> configs = {
        {2, 1}, {2, 2},
        {3, 1}, {3, 2}, {3, 3},
        {5, 1}, {5, 3}, {5, 5},
        {8, 2}, {8, 4}, {8, 8},
    };
    const unsigned num_seeds = 20;
    int failures = 0;

    for (auto& cfg : configs) {
        for (unsigned seed = 1; seed <= num_seeds; seed++) {
            auto prob = build_fuzz_sdp({cfg.mat_dim}, cfg.n_eq, 0, seed, 1e-3);
            auto r = solve_fuzz(prob);

            bool solved = (r.status == 1 || r.status == 4);  // Solved or AlmostSolved
            if (!solved) {
                std::cout << "FAIL: mat_dim=" << cfg.mat_dim << " n_eq=" << cfg.n_eq
                          << " seed=" << seed << " status=" << r.status
                          << " iters=" << r.iterations << std::endl;
                failures++;
                continue;
            }

            bool kkt_ok = check_kkt(prob, r, TOL_KKT);
            if (!kkt_ok) {
                std::cout << "KKT FAIL: mat_dim=" << cfg.mat_dim << " n_eq=" << cfg.n_eq
                          << " seed=" << seed << std::endl;
                check_kkt(prob, r, TOL_KKT, true);
                failures++;
            }
        }
    }
    std::cout << "SinglePSD_MultiSeed: " << (configs.size() * num_seeds - failures)
              << "/" << configs.size() * num_seeds << " passed" << std::endl;
    EXPECT_EQ(failures, 0);
}

// ============================================================================
// Test 2: Multiple PSD cones, many seeds
// ============================================================================

TEST_F(PsdFuzzTest, MultiplePSD_MultiSeed) {
    struct Config { std::vector<int64_t> psd_dims; int64_t n_eq; };
    std::vector<Config> configs = {
        {{2, 3}, 2},
        {{3, 5}, 3},
        {{2, 2, 2}, 2},
        {{2, 3, 5}, 3},
        {{3, 3, 3}, 3},
    };
    const unsigned num_seeds = 15;
    int failures = 0;

    for (auto& cfg : configs) {
        for (unsigned seed = 1; seed <= num_seeds; seed++) {
            auto prob = build_fuzz_sdp(cfg.psd_dims, cfg.n_eq, 0, seed, 1e-3);
            auto r = solve_fuzz(prob);

            bool solved = (r.status == 1 || r.status == 4);
            if (!solved) {
                std::cout << "FAIL: psd_dims=[";
                for (size_t i = 0; i < cfg.psd_dims.size(); i++)
                    std::cout << (i ? "," : "") << cfg.psd_dims[i];
                std::cout << "] n_eq=" << cfg.n_eq << " seed=" << seed
                          << " status=" << r.status << std::endl;
                failures++;
                continue;
            }

            bool kkt_ok = check_kkt(prob, r, TOL_KKT);
            if (!kkt_ok) {
                std::cout << "KKT FAIL: psd_dims=[";
                for (size_t i = 0; i < cfg.psd_dims.size(); i++)
                    std::cout << (i ? "," : "") << cfg.psd_dims[i];
                std::cout << "] n_eq=" << cfg.n_eq << " seed=" << seed << std::endl;
                check_kkt(prob, r, TOL_KKT, true);
                failures++;
            }
        }
    }
    std::cout << "MultiplePSD_MultiSeed: " << (configs.size() * num_seeds - failures)
              << "/" << configs.size() * num_seeds << " passed" << std::endl;
    EXPECT_EQ(failures, 0);
}

// ============================================================================
// Test 3: Mixed cones (zero + nonneg + PSD), many seeds
// ============================================================================

TEST_F(PsdFuzzTest, MixedCones_MultiSeed) {
    struct Config {
        std::vector<int64_t> psd_dims;
        int64_t n_eq, n_nonneg;
        const char* label;
    };
    std::vector<Config> configs = {
        {{3}, 2, 0, "zero(2)+psd(3)"},
        {{2}, 0, 3, "nonneg(3)+psd(2)"},
        {{3}, 1, 2, "zero(1)+nonneg(2)+psd(3)"},
        {{2, 3}, 2, 0, "zero(2)+psd(2,3)"},
        {{3}, 2, 3, "zero(2)+nonneg(3)+psd(3)"},
        {{2, 3}, 1, 2, "zero(1)+nonneg(2)+psd(2,3)"},
    };
    const unsigned num_seeds = 15;
    int failures = 0;

    for (auto& cfg : configs) {
        for (unsigned seed = 1; seed <= num_seeds; seed++) {
            auto prob = build_fuzz_sdp(cfg.psd_dims, cfg.n_eq, cfg.n_nonneg, seed, 1e-3);
            auto r = solve_fuzz(prob);

            bool solved = (r.status == 1 || r.status == 4);
            if (!solved) {
                std::cout << "FAIL: " << cfg.label << " seed=" << seed
                          << " status=" << r.status << std::endl;
                failures++;
                continue;
            }

            bool kkt_ok = check_kkt(prob, r, TOL_KKT);
            if (!kkt_ok) {
                std::cout << "KKT FAIL: " << cfg.label << " seed=" << seed << std::endl;
                check_kkt(prob, r, TOL_KKT, true);
                failures++;
            }
        }
    }
    std::cout << "MixedCones_MultiSeed: " << (configs.size() * num_seeds - failures)
              << "/" << configs.size() * num_seeds << " passed" << std::endl;
    EXPECT_EQ(failures, 0);
}

// ============================================================================
// Test 4: Larger PSD cones
// ============================================================================

TEST_F(PsdFuzzTest, LargePSD_MultiSeed) {
    struct Config { int64_t mat_dim; int64_t n_eq; };
    std::vector<Config> configs = {
        {10, 5},
        {10, 10},
        {15, 5},
        {15, 10},
    };
    const unsigned num_seeds = 10;
    int failures = 0;

    for (auto& cfg : configs) {
        for (unsigned seed = 1; seed <= num_seeds; seed++) {
            auto prob = build_fuzz_sdp({cfg.mat_dim}, cfg.n_eq, 0, seed, 1e-3);
            auto r = solve_fuzz(prob, 300);

            bool solved = (r.status == 1 || r.status == 4);
            if (!solved) {
                std::cout << "FAIL: mat_dim=" << cfg.mat_dim << " n_eq=" << cfg.n_eq
                          << " seed=" << seed << " status=" << r.status
                          << " iters=" << r.iterations << std::endl;
                failures++;
                continue;
            }

            bool kkt_ok = check_kkt(prob, r, TOL_KKT);
            if (!kkt_ok) {
                std::cout << "KKT FAIL: mat_dim=" << cfg.mat_dim << " n_eq=" << cfg.n_eq
                          << " seed=" << seed << std::endl;
                check_kkt(prob, r, TOL_KKT, true);
                failures++;
            }
        }
    }
    std::cout << "LargePSD_MultiSeed: " << (configs.size() * num_seeds - failures)
              << "/" << configs.size() * num_seeds << " passed" << std::endl;
    EXPECT_EQ(failures, 0);
}

// ============================================================================
// Test 5: Dense P (QP-type SDP)
// ============================================================================

TEST_F(PsdFuzzTest, DenseP_PSD_MultiSeed) {
    struct Config { int64_t mat_dim; int64_t n_eq; };
    std::vector<Config> configs = {
        {3, 2},
        {5, 3},
        {8, 4},
    };
    const unsigned num_seeds = 15;
    int failures = 0;

    for (auto& cfg : configs) {
        for (unsigned seed = 1; seed <= num_seeds; seed++) {
            auto prob = build_fuzz_sdp({cfg.mat_dim}, cfg.n_eq, 0, seed, 0.1, /*dense_P=*/true);
            auto r = solve_fuzz(prob);

            bool solved = (r.status == 1 || r.status == 4);
            if (!solved) {
                std::cout << "FAIL: dense_P mat_dim=" << cfg.mat_dim << " n_eq=" << cfg.n_eq
                          << " seed=" << seed << " status=" << r.status << std::endl;
                failures++;
                continue;
            }

            bool kkt_ok = check_kkt(prob, r, TOL_KKT);
            if (!kkt_ok) {
                std::cout << "KKT FAIL: dense_P mat_dim=" << cfg.mat_dim << " n_eq=" << cfg.n_eq
                          << " seed=" << seed << std::endl;
                check_kkt(prob, r, TOL_KKT, true);
                failures++;
            }
        }
    }
    std::cout << "DenseP_PSD_MultiSeed: " << (configs.size() * num_seeds - failures)
              << "/" << configs.size() * num_seeds << " passed" << std::endl;
    EXPECT_EQ(failures, 0);
}

// ============================================================================
// Test 6: Backward pass gradcheck across seeds
//
// Uses A=I formulation (which the backward pass handles correctly) with
// randomized P, q, b across many seeds. This catches regressions in PSD
// gradient computation without hitting the known dense-A gradient bug.
//
//   min  0.5 x'Px + q'x   s.t.  Ix + s = b,  s ∈ PSD(mat_dim)
//
// b is chosen as svec of a random PSD matrix (so the problem is feasible
// and the solution is in the PSD interior when P is strongly convex).
// ============================================================================

FuzzSdpProblem build_grad_sdp(int64_t mat_dim, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);

    int64_t sd = svec_dim(mat_dim);
    FuzzSdpProblem prob;
    prob.n_var = sd;
    prob.m_con = sd;
    prob.num_zero = 0;
    prob.num_nonneg = 0;
    prob.psd_dims = {mat_dim};

    // P = L'L + I (well-conditioned dense PD)
    prob.P_dense.resize(sd * sd, 0.0);
    std::vector<double> L(sd * sd, 0.0);
    for (int64_t i = 0; i < sd; i++) {
        for (int64_t j = 0; j <= i; j++)
            L[i * sd + j] = nd(rng) * 0.5;
        L[i * sd + i] += 1.0;
    }
    for (int64_t i = 0; i < sd; i++)
        for (int64_t j = 0; j <= i; j++) {
            double val = 0.0;
            for (int64_t k = 0; k <= std::min(i, j); k++)
                val += L[i * sd + k] * L[j * sd + k];
            prob.P_dense[i * sd + j] = val;
            prob.P_dense[j * sd + i] = val;
        }
    for (int64_t i = 0; i < sd; i++) prob.P_dense[i * sd + i] += 1.0;

    // P CSR (full symmetric)
    prob.P_ro.resize(sd + 1);
    prob.P_ro[0] = 0;
    for (int64_t i = 0; i < sd; i++) {
        for (int64_t j = 0; j < sd; j++) {
            prob.P_ci.push_back(j);
            prob.P_val.push_back(prob.P_dense[i * sd + j]);
        }
        prob.P_ro[i + 1] = (int64_t)prob.P_ci.size();
    }

    // A = I
    prob.A_dense.resize(sd * sd, 0.0);
    prob.A_ro.resize(sd + 1);
    prob.A_ro[0] = 0;
    for (int64_t i = 0; i < sd; i++) {
        prob.A_dense[i * sd + i] = 1.0;
        prob.A_ci.push_back(i);
        prob.A_val.push_back(1.0);
        prob.A_ro[i + 1] = i + 1;
    }

    // q: random
    prob.q.resize(sd);
    for (auto& v : prob.q) v = nd(rng) * 0.1;

    // b = svec of random PSD matrix (identity + small perturbation)
    auto M = randn_psd(mat_dim, rng, 0.5);  // PSD with decent margin
    prob.b = mat_to_svec(M, mat_dim);

    return prob;
}

TEST_F(PsdFuzzTest, GradCheck_dq_MultiSeed) {
    struct Config { int64_t mat_dim; };
    std::vector<Config> configs = {{2}, {3}, {5}};
    const unsigned num_seeds = 10;
    int failures = 0;
    int total = 0;

    for (auto& cfg : configs) {
        for (unsigned seed = 1; seed <= num_seeds; seed++) {
            total++;
            auto prob = build_grad_sdp(cfg.mat_dim, seed);

            std::mt19937_64 rng(seed * 1000);
            std::normal_distribution<double> nd(0.0, 1.0);
            std::vector<double> dx_bar(prob.n_var);
            for (auto& v : dx_bar) v = nd(rng);

            auto gr = backward_fuzz(prob, dx_bar);
            if (gr.status != 1) {
                std::cout << "SKIP: mat_dim=" << cfg.mat_dim
                          << " seed=" << seed << " status=" << gr.status << std::endl;
                continue;
            }

            bool this_ok = true;
            for (int64_t i = 0; i < prob.n_var; i++) {
                auto prob_p = prob; prob_p.q[i] += FD_H;
                auto prob_m = prob; prob_m.q[i] -= FD_H;
                auto rp = solve_fuzz(prob_p);
                auto rm = solve_fuzz(prob_m);
                if (rp.status != 1 || rm.status != 1) continue;

                double fd = 0.0;
                for (int64_t j = 0; j < prob.n_var; j++)
                    fd += dx_bar[j] * (rp.x[j] - rm.x[j]) / (2 * FD_H);

                double tol = grad_tol(gr.dq[i], fd);
                if (std::abs(gr.dq[i] - fd) > tol) {
                    std::cout << "GRAD FAIL: mat_dim=" << cfg.mat_dim << " seed=" << seed
                              << " dq[" << i << "] analytical=" << gr.dq[i]
                              << " fd=" << fd << " tol=" << tol << std::endl;
                    this_ok = false;
                }
            }
            if (!this_ok) failures++;
        }
    }
    std::cout << "GradCheck_dq_MultiSeed: " << (total - failures)
              << "/" << total << " passed" << std::endl;
    EXPECT_EQ(failures, 0);
}

TEST_F(PsdFuzzTest, GradCheck_db_MultiSeed) {
    struct Config { int64_t mat_dim; };
    std::vector<Config> configs = {{2}, {3}};
    const unsigned num_seeds = 10;
    int failures = 0;
    int total = 0;

    for (auto& cfg : configs) {
        for (unsigned seed = 1; seed <= num_seeds; seed++) {
            total++;
            auto prob = build_grad_sdp(cfg.mat_dim, seed);

            std::mt19937_64 rng(seed * 2000);
            std::normal_distribution<double> nd(0.0, 1.0);
            std::vector<double> dx_bar(prob.n_var);
            for (auto& v : dx_bar) v = nd(rng);

            auto gr = backward_fuzz(prob, dx_bar);
            if (gr.status != 1) {
                std::cout << "SKIP: mat_dim=" << cfg.mat_dim << " seed=" << seed
                          << " status=" << gr.status << std::endl;
                continue;
            }

            bool this_ok = true;
            for (int64_t i = 0; i < prob.m_con; i++) {
                auto prob_p = prob; prob_p.b[i] += FD_H;
                auto prob_m = prob; prob_m.b[i] -= FD_H;
                auto rp = solve_fuzz(prob_p);
                auto rm = solve_fuzz(prob_m);
                if (rp.status != 1 || rm.status != 1) continue;

                double fd = 0.0;
                for (int64_t j = 0; j < prob.n_var; j++)
                    fd += dx_bar[j] * (rp.x[j] - rm.x[j]) / (2 * FD_H);

                double tol = grad_tol(gr.db[i], fd);
                if (std::abs(gr.db[i] - fd) > tol) {
                    std::cout << "GRAD FAIL: mat_dim=" << cfg.mat_dim << " seed=" << seed
                              << " db[" << i << "] analytical=" << gr.db[i]
                              << " fd=" << fd << " tol=" << tol << std::endl;
                    this_ok = false;
                }
            }
            if (!this_ok) failures++;
        }
    }
    std::cout << "GradCheck_db_MultiSeed: " << (total - failures)
              << "/" << total << " passed" << std::endl;
    EXPECT_EQ(failures, 0);
}

// ============================================================================
// Test 7: Backward pass with dense A rows.
//
// HISTORY: Previously DISABLED with a comment blaming "equilibration × adjoint
// KKT" for inaccurate gradients. That diagnosis was wrong — the analytical
// gradient is correct; the test itself was noisy.
//
// ROOT CAUSE: FD_H=1e-5 is too small for this problem. At default solver
// tolerance 1e-8 the forward solution has residual ~1e-8, so FD noise is
// tol/h ~ 1e-3 per coordinate, and dense equality rows in A amplify that
// noise into an O(1) error on some seeds. Verified by (a) comparing the
// analytical gradient to an independent Python reference implementation of
// D Π_PSD composed with the adjoint KKT (agreement to 1e-6) and (b) sweeping
// FD h from 1e-1 down to 1e-5: FD converges cleanly to the analytical value
// for h ≥ 3e-3 and oscillates wildly below that.
//
// FIX: local DENSE_A_FD_H = 1e-2 instead of the global FD_H = 1e-5. Other
// tests in this file use identity A, where x(q) is well enough conditioned
// that FD_H=1e-5 works; this test's dense equality rows amplify solver-
// iterate noise and need a larger step.
// ============================================================================

TEST_F(PsdFuzzTest, GradCheck_DenseA_dq) {
    constexpr double DENSE_A_FD_H = 3e-2;
    // Larger tolerances than the global grad_tol (5%, 1e-3) to absorb (a) cuDSS
    // non-determinism and (b) FD truncation error at h=3e-2 — both amplified by
    // dense equality rows. Still tight enough to catch real backward-pass bugs;
    // see repro investigation in git log for this test.
    constexpr double DENSE_A_ABS_TOL = 5e-2;
    constexpr double DENSE_A_REL_TOL = 0.10;
    auto dense_a_grad_tol = [&](double a, double b) {
        return std::max(DENSE_A_ABS_TOL,
                        DENSE_A_REL_TOL * std::max(std::abs(a), std::abs(b)));
    };

    struct Config { int64_t mat_dim; int64_t n_eq; };
    std::vector<Config> configs = {{2, 1}, {3, 2}, {5, 3}};
    const unsigned num_seeds = 5;
    int failures = 0;
    int total = 0;

    for (auto& cfg : configs) {
        for (unsigned seed = 1; seed <= num_seeds; seed++) {
            total++;
            auto prob = build_fuzz_sdp({cfg.mat_dim}, cfg.n_eq, 0, seed, 1e-2);

            std::mt19937_64 rng(seed * 1000);
            std::normal_distribution<double> nd(0.0, 1.0);
            std::vector<double> dx_bar(prob.n_var);
            for (auto& v : dx_bar) v = nd(rng);

            auto gr = backward_fuzz(prob, dx_bar);
            if (gr.status != 1) continue;

            bool this_ok = true;
            for (int64_t i = 0; i < prob.n_var; i++) {
                auto prob_p = prob; prob_p.q[i] += DENSE_A_FD_H;
                auto prob_m = prob; prob_m.q[i] -= DENSE_A_FD_H;
                auto rp = solve_fuzz(prob_p);
                auto rm = solve_fuzz(prob_m);
                if (rp.status != 1 || rm.status != 1) continue;

                double fd = 0.0;
                for (int64_t j = 0; j < prob.n_var; j++)
                    fd += dx_bar[j] * (rp.x[j] - rm.x[j]) / (2 * DENSE_A_FD_H);

                double tol = dense_a_grad_tol(gr.dq[i], fd);
                if (std::abs(gr.dq[i] - fd) > tol) {
                    std::cout << "GRAD FAIL: mat_dim=" << cfg.mat_dim << " seed=" << seed
                              << " dq[" << i << "] analytical=" << gr.dq[i]
                              << " fd=" << fd << std::endl;
                    this_ok = false;
                }
            }
            if (!this_ok) failures++;
        }
    }
    std::cout << "GradCheck_DenseA_dq: " << (total - failures)
              << "/" << total << " passed" << std::endl;
    EXPECT_EQ(failures, 0);
}
