/**
 * @file test_exp.cpp
 * @brief Exponential cone problem tests with known solutions
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

// Sample a point strictly inside the 3D exponential cone.
std::array<double, 3> sampleExpconeInterior(std::mt19937_64& rng, double margin = 0.1) {
    std::normal_distribution<double> normal(0.0, 1.0);
    double y = std::exp(normal(rng) * 0.5) + std::abs(margin);
    double x = normal(rng) * 0.5;
    double z = y * std::exp(x / y) + std::abs(margin);
    return {x, y, z};
}

bool isInExpcone(const std::array<double, 3>& v, double tol = 1e-8) {
    double x = v[0];
    double y = v[1];
    double z = v[2];

    if (y > tol) {
        return z >= y * std::exp(x / y) - tol;
    }
    if (std::abs(y) <= tol) {
        return (x <= tol) && (z >= -tol);
    }
    return false;
}

bool isInDualExpcone(const std::array<double, 3>& v, double tol = 1e-8) {
    double u = v[0];
    double v_mid = v[1];
    double w = v[2];

    if ((w < -tol) && (u > tol)) {
        return std::exp(1.0) * u + w * std::exp(v_mid / w) >= -tol;
    }
    if (std::abs(w) <= tol && u >= -tol) {
        return std::abs(v_mid) <= 1.0 / tol;
    }
    return false;
}

bool checkExpconeProductMembership(
    const std::vector<double>& z,
    int64_t numCones,
    bool primal = true,
    double tol = 1e-8
) {
    for (int64_t k = 0; k < numCones; k++) {
        std::array<double, 3> block{
            z[3 * k + 0],
            z[3 * k + 1],
            z[3 * k + 2]
        };
        if (primal) {
            if (!isInExpcone(block, tol)) return false;
        } else {
            if (!isInDualExpcone(block, tol)) return false;
        }
    }
    return true;
}

std::vector<double> expconeComplementarity(
    const std::vector<double>& s,
    const std::vector<double>& y,
    int64_t numCones
) {
    std::vector<double> comps(static_cast<size_t>(numCones), 0.0);
    for (int64_t k = 0; k < numCones; k++) {
        double dot = 0.0;
        for (int idx = 0; idx < 3; idx++) {
            dot += s[3 * k + idx] * y[3 * k + idx];
        }
        comps[static_cast<size_t>(k)] = dot;
    }
    return comps;
}

/**
 * @brief Generate exponential cone problem with known solution (mirrors Python helper)
 *
 * Problem data:
 *   minimize    (1/2) x^T P x + q^T x       with P = I
 *   subject to  A x + s = b,  s in product of 3D exponential cones
 *
 * Constructs b so that (x_star, s_star) is feasible and optimal.
 */
struct GeneratedExpconeProblem {
    int64_t n;
    int64_t numCones;
    int64_t m;

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

    GeneratedExpconeProblem(int64_t n_vars, int64_t n_cones, double margin = 0.1, uint64_t seed = 0)
        : n(n_vars), numCones(n_cones), m(3 * n_cones)
    {
        if (n_vars <= 0 || n_cones <= 0) {
            throw std::runtime_error("Need positive n_vars and n_cones");
        }

        std::mt19937_64 rng(seed);
        std::normal_distribution<double> normal(0.0, 1.0);

        x_star.resize(n);
        for (int64_t j = 0; j < n; j++) {
            x_star[j] = normal(rng);
        }

        q.resize(n);
        for (int64_t j = 0; j < n; j++) {
            q[j] = -x_star[j];
        }

        A_dense.assign(m, std::vector<double>(n, 0.0));
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                A_dense[i][j] = normal(rng);
            }
        }

        s_star.resize(m);
        for (int64_t k = 0; k < numCones; k++) {
            auto block = sampleExpconeInterior(rng, margin);
            s_star[3 * k + 0] = block[0];
            s_star[3 * k + 1] = block[1];
            s_star[3 * k + 2] = block[2];
        }

        b.assign(m, 0.0);
        for (int64_t i = 0; i < m; i++) {
            double Ax_i = 0.0;
            for (int64_t j = 0; j < n; j++) {
                Ax_i += A_dense[i][j] * x_star[j];
            }
            b[i] = Ax_i + s_star[i];
        }

        A_ro.reserve(static_cast<size_t>(m + 1));
        A_ro.push_back(0);
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                A_ci.push_back(j);
                A_val.push_back(A_dense[i][j]);
            }
            A_ro.push_back(static_cast<int64_t>(A_val.size()));
        }

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
        return checkExpconeProductMembership(s, numCones, true, tol);
    }

    bool dualInCone(const std::vector<double>& z, double tol = 1e-8) const {
        return checkExpconeProductMembership(z, numCones, false, tol);
    }
};

/**
 * @brief Check KKT conditions for the generated exponential cone problem.
 *
 * KKT for min 0.5||x||^2 + q^T x s.t. A x + s = b, s in K (exp cones):
 *   primal feas: A x + s - b = 0
 *   dual feas (stationarity): x + q + A^T y = 0 (since P = I)
 *   cone feas: s in K, y in K*
 *   complementarity: <s_i, y_i> = 0 for each cone block
 */
bool checkKKTExp(
    const GeneratedExpconeProblem& prob,
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

    double primal_res_inf = 0.0;
    for (int64_t i = 0; i < prob.m; i++) {
        double r = -prob.b[i] + s[i];
        for (int64_t j = 0; j < prob.n; j++) {
            r += prob.A_dense[i][j] * x[j];
        }
        primal_res_inf = std::max(primal_res_inf, std::abs(r));
    }
    bool primal_ok = primal_res_inf <= tol_feas;

    double stationarity_inf = 0.0;
    for (int64_t j = 0; j < prob.n; j++) {
        double grad = x[j] + prob.q[j];
        for (int64_t i = 0; i < prob.m; i++) {
            grad += prob.A_dense[i][j] * y[i];
        }
        stationarity_inf = std::max(stationarity_inf, std::abs(grad));
    }
    bool stationarity_ok = stationarity_inf <= tol_kkt;

    bool s_in_K = prob.slackInCone(s, tol_feas);
    bool y_in_K = prob.dualInCone(y, tol_feas);

    double comp_inf = 0.0;
    auto comps = expconeComplementarity(s, y, prob.numCones);
    for (double c : comps) {
        comp_inf = std::max(comp_inf, std::abs(c));
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

class ExpConeTest : public ::testing::Test {
protected:
    void SetUp() override { cudaGetLastError(); }
    void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_F(ExpConeTest, SmallGeneratedProblemMatchesGroundTruth) {
    GeneratedExpconeProblem prob(/*n_vars=*/6, /*n_cones=*/2, /*margin=*/0.2, /*seed=*/2025);

    Cones cones{};
    cones.numExpCones = prob.numCones;

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
    EXPECT_LT(max_x_err, 1e-4);
    EXPECT_LT(max_s_err, 1e-4);

    double primal_res = prob.primalResidualInf(x_sol, s_sol);
    EXPECT_LT(primal_res, 5e-5);
    EXPECT_TRUE(prob.slackInCone(s_sol, 1e-6));

    double obj_expected = prob.objective(prob.x_star);
    double obj_solver = prob.objective(x_sol);
    EXPECT_NEAR(obj_solver, obj_expected, 1e-4);

    std::vector<double> y_star(prob.m, 0.0);
    EXPECT_TRUE(checkKKTExp(prob, prob.x_star, prob.s_star, y_star, 1e-7, 1e-7, false));

    bool kkt_solver = checkKKTExp(prob, x_sol, s_sol, z_sol, 5e-5, 5e-5, false);
    if (!kkt_solver) {
        checkKKTExp(prob, x_sol, s_sol, z_sol, 5e-5, 5e-5, true);
    }
    EXPECT_TRUE(kkt_solver);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(ExpConeTest, BatchedProblemsConverge) {
    const int64_t n = 8;
    const int64_t numCones = 3;
    const int64_t batchSize = 2;

    GeneratedExpconeProblem prob1(n, numCones, 0.15, 7);
    GeneratedExpconeProblem prob2(n, numCones, 0.25, 1234);
    std::vector<GeneratedExpconeProblem*> probs = {&prob1, &prob2};

    Cones cones{};
    cones.numExpCones = numCones;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 350;
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
        std::vector<double> x_batch(x_sol.begin() + b * n, x_sol.begin() + (b + 1) * n);
        std::vector<double> s_batch(s_sol.begin() + b * p->m, s_sol.begin() + (b + 1) * p->m);
        std::vector<double> z_batch(z_sol.begin() + b * p->m, z_sol.begin() + (b + 1) * p->m);

        double max_x_err = 0.0;
        double max_s_err = 0.0;
        for (int64_t i = 0; i < n; i++) {
            max_x_err = std::max(max_x_err, std::abs(x_batch[i] - p->x_star[i]));
        }
        for (int64_t i = 0; i < p->m; i++) {
            max_s_err = std::max(max_s_err, std::abs(s_batch[i] - p->s_star[i]));
        }
        // Relaxed tolerances for batched exp cone (GPU numerical precision)
        EXPECT_LT(max_x_err, 1e-3);
        EXPECT_LT(max_s_err, 1e-3);
        EXPECT_TRUE(p->slackInCone(s_batch, 1e-6));

        std::vector<double> y_star(p->m, 0.0);
        EXPECT_TRUE(checkKKTExp(*p, p->x_star, p->s_star, y_star, 5e-7, 5e-7, false));

        bool kkt_solver = checkKKTExp(*p, x_batch, s_batch, z_batch, 1e-3, 1e-3, false);
        if (!kkt_solver) {
            checkKKTExp(*p, x_batch, s_batch, z_batch, 1e-4, 1e-4, true);
        }
        EXPECT_TRUE(kkt_solver);
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
