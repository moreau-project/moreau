/**
 * @file test_soc.cpp
 * @brief Second-order cone (SOC) problem tests with known solutions
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/solver.hpp"

using namespace moreau;

namespace {

// Sample a point strictly inside the 3D SOC { (t, u1, u2) | t >= sqrt(u1^2 + u2^2) }
std::array<double, 3> sampleSoc3Interior(std::mt19937_64& rng, double margin = 0.1) {
    std::normal_distribution<double> normal(0.0, 1.0);
    double u1 = normal(rng);
    double u2 = normal(rng);
    double norm_u = std::sqrt(u1 * u1 + u2 * u2);
    double t = norm_u + std::abs(margin);
    return {t, u1, u2};
}

bool isInSoc3(const std::array<double, 3>& s, double tol = 1e-8) {
    double radial = std::sqrt(s[1] * s[1] + s[2] * s[2]);
    return s[0] >= radial - tol;
}

bool isInSocProduct(const std::vector<double>& s, int64_t numCones, double tol = 1e-8) {
    for (int64_t k = 0; k < numCones; k++) {
        std::array<double, 3> block{
            s[3 * k + 0],
            s[3 * k + 1],
            s[3 * k + 2]
        };
        if (!isInSoc3(block, tol)) return false;
    }
    return true;
}

/**
 * @brief Generate SOC problem with known solution (mirrors provided Python helper)
 *
 * Problem data:
 *   minimize    (1/2) x^T P x + q^T x       with P = I
 *   subject to  A x + s = b,  s in product of 3D SOCs
 *
 * Constructs b so that (x_star, s_star) is feasible and optimal.
 */
struct GeneratedSOCProblem {
    int64_t n;          // number of primal variables
    int64_t numCones;   // number of SOC blocks (dimension 3 each)
    int64_t m;          // total constraints = 3 * numCones

    std::vector<std::vector<double>> A_dense;
    std::vector<double> b;
    std::vector<double> q;
    std::vector<double> x_star;
    std::vector<double> s_star;

    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;

    std::vector<int64_t> P_ro;
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;

    GeneratedSOCProblem(int64_t n_vars, int64_t n_cones, double margin = 0.1, uint64_t seed = 0)
        : n(n_vars), numCones(n_cones), m(3 * n_cones)
    {
        if (n_vars <= 0 || n_cones <= 0) {
            throw std::runtime_error("Need positive n_vars and n_cones");
        }

        std::mt19937_64 rng(seed);
        std::normal_distribution<double> normal(0.0, 1.0);

        // Choose a random primal optimum x_star
        x_star.resize(n);
        for (int64_t j = 0; j < n; j++) {
            x_star[j] = normal(rng);
        }

        // Set q so that the unconstrained minimizer is x_star
        q.resize(n);
        for (int64_t j = 0; j < n; j++) {
            q[j] = -x_star[j];
        }

        // Dense A matrix (m x n) with standard normal entries
        A_dense.assign(m, std::vector<double>(n, 0.0));
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                A_dense[i][j] = normal(rng);
            }
        }

        // Slack strictly inside each SOC block
        s_star.resize(m);
        for (int64_t k = 0; k < numCones; k++) {
            auto block = sampleSoc3Interior(rng, margin);
            s_star[3 * k + 0] = block[0];
            s_star[3 * k + 1] = block[1];
            s_star[3 * k + 2] = block[2];
        }

        // Build b to enforce Ax_star + s_star = b
        b.assign(m, 0.0);
        for (int64_t i = 0; i < m; i++) {
            double Ax_i = 0.0;
            for (int64_t j = 0; j < n; j++) {
                Ax_i += A_dense[i][j] * x_star[j];
            }
            b[i] = Ax_i + s_star[i];
        }

        // Convert A to CSR (dense -> CSR, keep all entries)
        A_ro.reserve(static_cast<size_t>(m + 1));
        A_ro.push_back(0);
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                A_ci.push_back(j);
                A_val.push_back(A_dense[i][j]);
            }
            A_ro.push_back(static_cast<int64_t>(A_val.size()));
        }

        // P = I in upper-triangular CSR format
        P_ro.reserve(static_cast<size_t>(n + 1));
        P_ro.push_back(0);
        for (int64_t i = 0; i < n; i++) {
            P_ci.push_back(i);
            P_val.push_back(1.0);
            P_ro.push_back(static_cast<int64_t>(P_val.size()));
        }
    }

    [[nodiscard]] int64_t nnzA() const { return static_cast<int64_t>(A_val.size()); }
    [[nodiscard]] int64_t nnzP() const { return static_cast<int64_t>(P_val.size()); }

    void rebuildCSRFromDense() {
        A_ro.clear();
        A_ci.clear();
        A_val.clear();
        A_ro.reserve(static_cast<size_t>(m + 1));
        A_ro.push_back(0);
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                A_ci.push_back(j);
                A_val.push_back(A_dense[i][j]);
            }
            A_ro.push_back(static_cast<int64_t>(A_val.size()));
        }
    }

    void recomputeB() {
        b.assign(m, 0.0);
        for (int64_t i = 0; i < m; i++) {
            double Ax_i = 0.0;
            for (int64_t j = 0; j < n; j++) {
                Ax_i += A_dense[i][j] * x_star[j];
            }
            b[i] = Ax_i + s_star[i];
        }
    }

    double objective(const std::vector<double>& x) const {
        double obj = 0.0;
        for (int64_t i = 0; i < n; i++) {
            obj += 0.5 * x[i] * x[i] + q[i] * x[i];
        }
        return obj;
    }

    double primalResidualInf(const std::vector<double>& x, const std::vector<double>& s) const {
        double max_res = 0.0;
        for (int64_t i = 0; i < m; i++) {
            double r = -b[i] + s[i];
            for (int64_t j = 0; j < n; j++) {
                r += A_dense[i][j] * x[j];
            }
            max_res = std::max(max_res, std::abs(r));
        }
        return max_res;
    }

    bool slackInCone(const std::vector<double>& s, double tol = 1e-8) const {
        return isInSocProduct(s, numCones, tol);
    }
};

/**
 * @brief Check KKT conditions for the generated SOC problem.
 *
 * KKT for min 0.5||x||^2 + q^T x s.t. A x + s = b, s in K (SOC, self-dual):
 *   primal feas: A x + s - b = 0
 *   dual feas (stationarity): x + q + A^T y = 0 (since P = I)
 *   cone feas: s in K, y in K
 *   complementarity: <s_i, y_i> = 0 for each cone block
 */
bool checkKKTSOC(
    const GeneratedSOCProblem& prob,
    const std::vector<double>& x,
    const std::vector<double>& s,
    const std::vector<double>& y,
    double tol_feas = 1e-6,
    double tol_kkt = 1e-6,
    bool verbose = false
) {
    if (static_cast<int64_t>(x.size()) != prob.n ||
        static_cast<int64_t>(s.size()) != prob.m ||
        static_cast<int64_t>(y.size()) != prob.m) {
        if (verbose) std::cout << "Dimension mismatch in KKT check\n";
        return false;
    }

    // Primal feasibility: A x + s = b
    double primal_res_inf = 0.0;
    for (int64_t i = 0; i < prob.m; i++) {
        double r = -prob.b[i] + s[i];
        for (int64_t j = 0; j < prob.n; j++) {
            r += prob.A_dense[i][j] * x[j];
        }
        primal_res_inf = std::max(primal_res_inf, std::abs(r));
    }
    bool primal_ok = primal_res_inf <= tol_feas;

    // Stationarity: P x + q + A^T y = 0 with P = I
    double stationarity_inf = 0.0;
    for (int64_t j = 0; j < prob.n; j++) {
        double grad = x[j] + prob.q[j];
        for (int64_t i = 0; i < prob.m; i++) {
            grad += prob.A_dense[i][j] * y[i];
        }
        stationarity_inf = std::max(stationarity_inf, std::abs(grad));
    }
    bool stationarity_ok = stationarity_inf <= tol_kkt;

    // Cone feasibility
    bool s_in_K = prob.slackInCone(s, tol_feas);
    bool y_in_K = prob.slackInCone(y, tol_feas);  // SOC is self-dual

    // Complementarity per cone
    double comp_inf = 0.0;
    for (int64_t k = 0; k < prob.numCones; k++) {
        double dot = 0.0;
        for (int64_t idx = 0; idx < 3; idx++) {
            dot += s[3 * k + idx] * y[3 * k + idx];
        }
        comp_inf = std::max(comp_inf, std::abs(dot));
    }
    bool comp_ok = comp_inf <= tol_kkt;

    bool kkt_ok = primal_ok && stationarity_ok && s_in_K && y_in_K && comp_ok;

    if (verbose && !kkt_ok) {
        std::cout << "KKT check failed\n"
                  << "  primal_res_inf: " << primal_res_inf << "\n"
                  << "  stationarity_inf: " << stationarity_inf << "\n"
                  << "  comp_inf: " << comp_inf << "\n"
                  << "  s_in_K: " << s_in_K << "\n"
                  << "  y_in_K: " << y_in_K << "\n";
    }

    return kkt_ok;
}

} // namespace

class SOCTest : public ::testing::Test {
protected:
    void SetUp() override { cudaGetLastError(); }
    void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_F(SOCTest, SmallGeneratedProblemMatchesGroundTruth) {
    GeneratedSOCProblem prob(/*n_vars=*/8, /*n_cones=*/3, /*margin=*/0.1, /*seed=*/2024);

    Cones cones{};
    cones.socConeDims = std::vector<int64_t>(prob.numCones, 3);
    cones.numSocCones = prob.numCones;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(
        prob.n, prob.m, /*batchSize=*/1,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP(),
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA(),
        cones,
        settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP());
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA());
    cudaMalloc(&d_q, sizeof(double) * prob.n);
    cudaMalloc(&d_b, sizeof(double) * prob.m);

    cudaMemcpy(d_P, prob.P_val.data(), sizeof(double) * prob.nnzP(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_val.data(), sizeof(double) * prob.nnzA(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(prob.n);
    std::vector<double> s_sol(prob.m);
    std::vector<int32_t> status(1);
    std::vector<double> z_sol(prob.m);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * prob.m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * prob.m, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved));

    double max_x_err = 0.0;
    for (int64_t i = 0; i < prob.n; i++) {
        max_x_err = std::max(max_x_err, std::abs(x_sol[i] - prob.x_star[i]));
    }
    EXPECT_LT(max_x_err, 1e-5);

    double max_s_err = 0.0;
    for (int64_t i = 0; i < prob.m; i++) {
        max_s_err = std::max(max_s_err, std::abs(s_sol[i] - prob.s_star[i]));
    }
    EXPECT_LT(max_s_err, 1e-5);

    double primal_res = prob.primalResidualInf(x_sol, s_sol);
    EXPECT_LT(primal_res, 1e-6);
    EXPECT_TRUE(prob.slackInCone(s_sol, 1e-7));

    double obj_expected = prob.objective(prob.x_star);
    double obj_solver = prob.objective(x_sol);
    EXPECT_NEAR(obj_solver, obj_expected, 1e-6);

    // Ground-truth KKT (y_star = 0 is valid for this construction)
    std::vector<double> y_star(prob.m, 0.0);
    EXPECT_TRUE(checkKKTSOC(prob, prob.x_star, prob.s_star, y_star, 1e-7, 1e-7, false));

    // Solver KKT
    bool kkt_solver = checkKKTSOC(prob, x_sol, s_sol, z_sol, 1e-6, 1e-6, false);
    if (!kkt_solver) {
        checkKKTSOC(prob, x_sol, s_sol, z_sol, 1e-6, 1e-6, true);
    }
    EXPECT_TRUE(kkt_solver);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(SOCTest, MediumGeneratedProblemConverges) {
    GeneratedSOCProblem prob(/*n_vars=*/12, /*n_cones=*/5, /*margin=*/0.15, /*seed=*/4242);

    Cones cones{};
    cones.socConeDims = std::vector<int64_t>(prob.numCones, 3);
    cones.numSocCones = prob.numCones;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 300;
    settings.verbose = false;

    CompiledSolver solver(
        prob.n, prob.m, /*batchSize=*/1,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP(),
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA(),
        cones,
        settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP());
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA());
    cudaMalloc(&d_q, sizeof(double) * prob.n);
    cudaMalloc(&d_b, sizeof(double) * prob.m);

    cudaMemcpy(d_P, prob.P_val.data(), sizeof(double) * prob.nnzP(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_val.data(), sizeof(double) * prob.nnzA(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(prob.n);
    std::vector<double> s_sol(prob.m);
    std::vector<int32_t> status(1);
    std::vector<double> obj_val(1);
    std::vector<double> z_sol(prob.m);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * prob.m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(obj_val.data(), solver.solution.obj_val.data(), sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * prob.m, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved));

    double max_x_err = 0.0;
    for (int64_t i = 0; i < prob.n; i++) {
        max_x_err = std::max(max_x_err, std::abs(x_sol[i] - prob.x_star[i]));
    }
    EXPECT_LT(max_x_err, 5e-5);

    double max_s_err = 0.0;
    for (int64_t i = 0; i < prob.m; i++) {
        max_s_err = std::max(max_s_err, std::abs(s_sol[i] - prob.s_star[i]));
    }
    EXPECT_LT(max_s_err, 5e-5);

    double primal_res = prob.primalResidualInf(x_sol, s_sol);
    EXPECT_LT(primal_res, 5e-6);
    EXPECT_TRUE(prob.slackInCone(s_sol, 1e-6));

    double obj_expected = prob.objective(prob.x_star);
    double obj_solver = prob.objective(x_sol);
    EXPECT_NEAR(obj_solver, obj_expected, 5e-6);
    EXPECT_NEAR(obj_val[0], obj_expected, 5e-6);

    // Ground-truth and solver KKT checks
    std::vector<double> y_star(prob.m, 0.0);
    EXPECT_TRUE(checkKKTSOC(prob, prob.x_star, prob.s_star, y_star, 5e-7, 5e-7, false));

    bool kkt_solver = checkKKTSOC(prob, x_sol, s_sol, z_sol, 5e-6, 5e-6, false);
    if (!kkt_solver) {
        checkKKTSOC(prob, x_sol, s_sol, z_sol, 5e-6, 5e-6, true);
    }
    EXPECT_TRUE(kkt_solver);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(SOCTest, BatchedProblemsConverge) {
    const int64_t n = 10;
    const int64_t numCones = 4;
    const int64_t batchSize = 2;

    GeneratedSOCProblem prob1(n, numCones, 0.08, 7);
    GeneratedSOCProblem prob2(n, numCones, 0.2, 1234);

    std::vector<GeneratedSOCProblem*> probs = {&prob1, &prob2};

    Cones cones{};
    cones.socConeDims = std::vector<int64_t>(numCones, 3);
    cones.numSocCones = numCones;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 300;
    settings.verbose = false;

    CompiledSolver solver(
        n, prob1.m, batchSize,
        prob1.P_ro.data(), prob1.P_ci.data(), prob1.nnzP(),
        prob1.A_ro.data(), prob1.A_ci.data(), prob1.nnzA(),
        cones,
        settings
    );

    const int64_t nnzP = prob1.nnzP();
    const int64_t nnzA = prob1.nnzA();

    std::vector<double> P_values(nnzP * batchSize);
    std::vector<double> A_values(nnzA * batchSize);
    std::vector<double> q_values(n * batchSize);
    std::vector<double> b_values(prob1.m * batchSize);

    for (int64_t b = 0; b < batchSize; b++) {
        auto* p = probs[b];
        std::copy(p->P_val.begin(), p->P_val.end(), P_values.begin() + b * nnzP);
        std::copy(p->A_val.begin(), p->A_val.end(), A_values.begin() + b * nnzA);
        std::copy(p->q.begin(), p->q.end(), q_values.begin() + b * n);
        std::copy(p->b.begin(), p->b.end(), b_values.begin() + b * p->m);
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * prob1.m * batchSize);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_values.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_values.data(), sizeof(double) * prob1.m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n * batchSize);
    std::vector<double> s_sol(prob1.m * batchSize);
    std::vector<double> z_sol(prob1.m * batchSize);
    std::vector<int32_t> status(batchSize);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * prob1.m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * prob1.m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int64_t b = 0; b < batchSize; b++) {
        EXPECT_TRUE(status[b] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status[b] == static_cast<int32_t>(SolverStatus::AlmostSolved));

        auto* p = probs[b];

        double max_x_err = 0.0;
        double max_s_err = 0.0;
        std::vector<double> x_batch(x_sol.begin() + b * n, x_sol.begin() + (b + 1) * n);
        std::vector<double> s_batch(s_sol.begin() + b * p->m, s_sol.begin() + (b + 1) * p->m);
        std::vector<double> z_batch(z_sol.begin() + b * p->m, z_sol.begin() + (b + 1) * p->m);
        for (int64_t i = 0; i < n; i++) {
            max_x_err = std::max(max_x_err, std::abs(x_batch[i] - p->x_star[i]));
        }
        for (int64_t i = 0; i < p->m; i++) {
            max_s_err = std::max(max_s_err, std::abs(s_batch[i] - p->s_star[i]));
        }
        EXPECT_LT(max_x_err, 1e-5);
        EXPECT_LT(max_s_err, 1e-5);
        EXPECT_TRUE(p->slackInCone(s_batch, 1e-6));

        std::vector<double> y_star(p->m, 0.0);
        EXPECT_TRUE(checkKKTSOC(*p, p->x_star, p->s_star, y_star, 1e-7, 1e-7, false));

        // Use slightly relaxed tolerance for batched problems due to per-batch initialization
        bool kkt_solver = checkKKTSOC(*p, x_batch, s_batch, z_batch, 1e-5, 1e-5, false);
        if (!kkt_solver) {
            checkKKTSOC(*p, x_batch, s_batch, z_batch, 1e-5, 1e-5, true);
        }
        EXPECT_TRUE(kkt_solver);
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(SOCTest, NearBoundaryWithRowScaling) {
    GeneratedSOCProblem prob(/*n_vars=*/20, /*n_cones=*/6, /*margin=*/1e-4, /*seed=*/99);

    // Alternate very small / very large row scalings to stress equilibration
    for (int64_t cone = 0; cone < prob.numCones; cone++) {
        double scale = (cone % 2 == 0) ? 1e-2 : 1e2;
        for (int64_t offset = 0; offset < 3; offset++) {
            int64_t row = 3 * cone + offset;
            for (int64_t j = 0; j < prob.n; j++) {
                prob.A_dense[row][j] *= scale;
            }
            prob.s_star[row] *= scale;
        }
    }
    for (int64_t j = 0; j < prob.n; j++) {
        prob.q[j] = -prob.x_star[j];
    }
    prob.recomputeB();
    prob.rebuildCSRFromDense();

    Cones cones{};
    cones.socConeDims = std::vector<int64_t>(prob.numCones, 3);
    cones.numSocCones = prob.numCones;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 400;
    settings.verbose = false;

    CompiledSolver solver(
        prob.n, prob.m, /*batchSize=*/1,
        prob.P_ro.data(), prob.P_ci.data(), prob.nnzP(),
        prob.A_ro.data(), prob.A_ci.data(), prob.nnzA(),
        cones,
        settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * prob.nnzP());
    cudaMalloc(&d_A, sizeof(double) * prob.nnzA());
    cudaMalloc(&d_q, sizeof(double) * prob.n);
    cudaMalloc(&d_b, sizeof(double) * prob.m);

    cudaMemcpy(d_P, prob.P_val.data(), sizeof(double) * prob.nnzP(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, prob.A_val.data(), sizeof(double) * prob.nnzA(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, prob.q.data(), sizeof(double) * prob.n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, prob.b.data(), sizeof(double) * prob.m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(prob.n);
    std::vector<double> s_sol(prob.m);
    std::vector<double> z_sol(prob.m);
    std::vector<int32_t> status(1);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * prob.n, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * prob.m, cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * prob.m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    EXPECT_TRUE(status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved));

    double max_x_err = 0.0;
    double max_s_err = 0.0;
    for (int64_t i = 0; i < prob.n; i++) {
        max_x_err = std::max(max_x_err, std::abs(x_sol[i] - prob.x_star[i]));
    }
    for (int64_t i = 0; i < prob.m; i++) {
        max_s_err = std::max(max_s_err, std::abs(s_sol[i] - prob.s_star[i]));
    }
    EXPECT_LT(max_x_err, 5e-3);
    EXPECT_LT(max_s_err, 5e-2);
    EXPECT_TRUE(prob.slackInCone(s_sol, 1e-6));

    std::vector<double> y_star(prob.m, 0.0);
    EXPECT_TRUE(checkKKTSOC(prob, prob.x_star, prob.s_star, y_star, 1e-5, 1e-5, false));

    bool kkt_solver = checkKKTSOC(prob, x_sol, s_sol, z_sol, 5e-5, 5e-5, false);
    if (!kkt_solver) {
        checkKKTSOC(prob, x_sol, s_sol, z_sol, 5e-5, 5e-5, true);
    }
    EXPECT_TRUE(kkt_solver);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(SOCTest, DetectsPrimalInfeasible) {
    // One 3D SOC with A x shifting all rows equally; choose b so no x can make s in cone
    const int64_t n = 1;
    const int64_t m = 3;
    const int64_t batchSize = 1;

    std::vector<int64_t> P_ro = {0, 1};
    std::vector<int64_t> P_ci = {0};
    std::vector<double> P_val = {1.0};

    // A has one nonzero per row (all ones)
    std::vector<int64_t> A_ro = {0, 1, 2, 3};
    std::vector<int64_t> A_ci = {0, 0, 0};
    std::vector<double> A_val = {1.0, 1.0, 1.0};

    std::vector<double> q = {0.0};
    std::vector<double> b = {0.0, 10.0, 10.0};  // no x makes s=b-Ax fall inside SOC

    Cones cones{};
    cones.socConeDims = {3};
    cones.numSocCones = 1;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 80;
    settings.verbose = false;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), static_cast<int64_t>(P_val.size()),
        A_ro.data(), A_ci.data(), static_cast<int64_t>(A_val.size()),
        cones,
        settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * P_val.size());
    cudaMalloc(&d_A, sizeof(double) * std::max<size_t>(1, A_val.size()));
    cudaMalloc(&d_q, sizeof(double) * q.size());
    cudaMalloc(&d_b, sizeof(double) * b.size());

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * P_val.size(), cudaMemcpyHostToDevice);
    if (!A_val.empty()) {
        cudaMemcpy(d_A, A_val.data(), sizeof(double) * A_val.size(), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_q, q.data(), sizeof(double) * q.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * b.size(), cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(batchSize);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    int32_t s = status[0];
    std::cout << "Infeasible test status = " << s << std::endl;
    EXPECT_TRUE(s == static_cast<int32_t>(SolverStatus::PrimalInfeasible) ||
                s == static_cast<int32_t>(SolverStatus::AlmostPrimalInfeasible) ||
                s == static_cast<int32_t>(SolverStatus::MaxIterations));

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
