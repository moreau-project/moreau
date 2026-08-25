/**
 * @file test_power.cpp
 * @brief Power cone problem tests with known solutions
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
#include "moreau/solver/solver_kernels.cuh"
#include "moreau/solver/solver.hpp"

using namespace moreau;

namespace {

std::array<double, 3> samplePowerconeInterior(
    std::mt19937_64& rng,
    double alpha,
    double marginFactor = 0.5
) {
    std::uniform_real_distribution<double> unif(0.5, 1.5);
    double x = unif(rng);
    double y = unif(rng);
    double t = std::pow(x, alpha) * std::pow(y, 1.0 - alpha);
    std::uniform_real_distribution<double> zdist(-marginFactor * t, marginFactor * t);
    double z = zdist(rng);
    return {x, y, z};
}

bool isInPowercone(const std::array<double, 3>& v, double alpha, double tol = 1e-8) {
    double x = v[0];
    double y = v[1];
    double z = v[2];

    if (x < -tol || y < -tol) {
        return false;
    }

    double x_clipped = std::max(x, 0.0);
    double y_clipped = std::max(y, 0.0);
    double t = std::pow(x_clipped, alpha) * std::pow(y_clipped, 1.0 - alpha);
    return t + tol >= std::abs(z);
}

bool isInDualPowercone(const std::array<double, 3>& v, double alpha, double tol = 1e-8) {
    double u = v[0];
    double v_mid = v[1];
    double w = v[2];

    if (u < -tol || v_mid < -tol) {
        return false;
    }
    if (u <= tol && v_mid <= tol) {
        return std::abs(w) <= tol;
    }

    double u_clipped = std::max(u, tol);
    double v_clipped = std::max(v_mid, tol);
    double base = std::pow(u_clipped / alpha, alpha) *
                  std::pow(v_clipped / (1.0 - alpha), 1.0 - alpha);
    return base + tol >= std::abs(w);
}

bool checkPowerconeProductMembership(
    const std::vector<double>& z,
    const std::vector<double>& alphas,
    bool primal = true,
    double tol = 1e-8
) {
    if (alphas.empty()) return false;
    if (static_cast<int64_t>(z.size()) != static_cast<int64_t>(alphas.size()) * 3) return false;

    for (size_t k = 0; k < alphas.size(); k++) {
        std::array<double, 3> block{
            z[3 * k + 0],
            z[3 * k + 1],
            z[3 * k + 2]
        };
        if (primal) {
            if (!isInPowercone(block, alphas[k], tol)) return false;
        } else {
            if (!isInDualPowercone(block, alphas[k], tol)) return false;
        }
    }
    return true;
}

std::vector<double> powerconeComplementarity(
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
 * @brief Generate power cone problem with known solution (mirrors Python helper)
 *
 * Problem data:
 *   minimize    (1/2) x^T P x + q^T x       with P = I
 *   subject to  A x + s = b,  s in product of 3D power cones
 *
 * Constructs b so that (x_star, s_star) is feasible and optimal.
 */
struct GeneratedPowerconeProblem {
    int64_t n;
    int64_t numCones;
    int64_t m;
    std::vector<double> alphas;

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

    GeneratedPowerconeProblem(
        int64_t n_vars,
        const std::vector<double>& alphas_in,
        double margin = 0.4,
        uint64_t seed = 0
    ) : n(n_vars),
        numCones(static_cast<int64_t>(alphas_in.size())),
        m(3 * numCones),
        alphas(alphas_in)
    {
        if (n_vars <= 0 || numCones <= 0) {
            throw std::runtime_error("Need positive n_vars and at least one power cone");
        }
        for (double a : alphas) {
            if (!(a > 0.0 && a < 1.0)) {
                throw std::runtime_error("Power cone alpha must be in (0, 1)");
            }
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
            auto block = samplePowerconeInterior(rng, alphas[static_cast<size_t>(k)], margin);
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
        return checkPowerconeProductMembership(s, alphas, true, tol);
    }

    bool dualInCone(const std::vector<double>& z, double tol = 1e-8) const {
        return checkPowerconeProductMembership(z, alphas, false, tol);
    }
};

/**
 * @brief Check KKT conditions for the generated power cone problem.
 *
 * KKT for min 0.5||x||^2 + q^T x s.t. A x + s = b, s in K (power cones):
 *   primal feas: A x + s - b = 0
 *   dual feas (stationarity): x + q + A^T y = 0 (since P = I)
 *   cone feas: s in K, y in K*
 *   complementarity: <s_i, y_i> = 0 for each cone block
 */
bool checkKKTPowercone(
    const GeneratedPowerconeProblem& prob,
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
    auto comps = powerconeComplementarity(s, y, prob.numCones);
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

class PowerconeTest : public ::testing::Test {
protected:
    void SetUp() override { cudaGetLastError(); }
    void TearDown() override { cudaDeviceSynchronize(); }
};

TEST_F(PowerconeTest, SmallGeneratedProblemMatchesGroundTruth) {
    std::vector<double> alphas = {0.35, 0.6, 0.8};
    GeneratedPowerconeProblem prob(/*n_vars=*/10, alphas, /*margin=*/0.4, /*seed=*/2027);

    Cones cones{};
    cones.numPowerCones = prob.numCones;
    cones.powerAlphas = prob.alphas;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 280;
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
    EXPECT_LT(max_x_err, 1e-5);
    EXPECT_LT(max_s_err, 1e-5);

    double primal_res = prob.primalResidualInf(x_sol, s_sol);
    EXPECT_LT(primal_res, 1e-6);
    EXPECT_TRUE(prob.slackInCone(s_sol, 1e-7));

    double obj_expected = prob.objective(prob.x_star);
    double obj_solver = prob.objective(x_sol);
    EXPECT_NEAR(obj_solver, obj_expected, 1e-6);

    std::vector<double> y_star(prob.m, 0.0);
    EXPECT_TRUE(checkKKTPowercone(prob, prob.x_star, prob.s_star, y_star, 1e-7, 1e-7, false));

    bool kkt_solver = checkKKTPowercone(prob, x_sol, s_sol, z_sol, 1e-6, 1e-6, false);
    if (!kkt_solver) {
        checkKKTPowercone(prob, x_sol, s_sol, z_sol, 1e-6, 1e-6, true);
    }
    EXPECT_TRUE(kkt_solver);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(PowerconeTest, BatchedProblemsConverge) {
    const int64_t n = 12;
    const int64_t numCones = 4;
    const int64_t batchSize = 2;
    std::vector<double> alphas = {0.25, 0.45, 0.65, 0.85};

    GeneratedPowerconeProblem prob1(n, alphas, 0.3, 77);
    GeneratedPowerconeProblem prob2(n, alphas, 0.5, 1234);
    std::vector<GeneratedPowerconeProblem*> probs = {&prob1, &prob2};

    Cones cones{};
    cones.numPowerCones = numCones;
    cones.powerAlphas = alphas;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 360;
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
        auto* p = probs[static_cast<size_t>(b)];
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
        EXPECT_TRUE(status[static_cast<size_t>(b)] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status[static_cast<size_t>(b)] == static_cast<int32_t>(SolverStatus::AlmostSolved));

        auto* p = probs[static_cast<size_t>(b)];
        std::vector<double> x_batch(
            x_sol.begin() + b * n,
            x_sol.begin() + (b + 1) * n
        );
        std::vector<double> s_batch(
            s_sol.begin() + b * p->m,
            s_sol.begin() + (b + 1) * p->m
        );
        std::vector<double> z_batch(
            z_sol.begin() + b * p->m,
            z_sol.begin() + (b + 1) * p->m
        );

        double max_x_err = 0.0;
        double max_s_err = 0.0;
        for (int64_t i = 0; i < n; i++) {
            max_x_err = std::max(max_x_err, std::abs(x_batch[static_cast<size_t>(i)] - p->x_star[i]));
        }
        for (int64_t i = 0; i < p->m; i++) {
            max_s_err = std::max(max_s_err, std::abs(s_batch[static_cast<size_t>(i)] - p->s_star[i]));
        }
        EXPECT_LT(max_x_err, 2e-5);
        EXPECT_LT(max_s_err, 2e-5);
        EXPECT_TRUE(p->slackInCone(s_batch, 1e-6));
        EXPECT_TRUE(p->dualInCone(z_batch, 1e-6));

        std::vector<double> y_star(p->m, 0.0);
        EXPECT_TRUE(checkKKTPowercone(*p, p->x_star, p->s_star, y_star, 5e-7, 5e-7, false));

        bool kkt_solver = checkKKTPowercone(*p, x_batch, s_batch, z_batch, 5e-6, 5e-6, false);
        if (!kkt_solver) {
            checkKKTPowercone(*p, x_batch, s_batch, z_batch, 5e-6, 5e-6, true);
        }
        EXPECT_TRUE(kkt_solver);
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// Regression test for the small-step termination fix.
//
// When the IPM's combined-step length collapses to α ≈ 0 (cone backtracking
// exhausted because the Newton direction would exit the cone — observed
// when cuDSS noise on ill-conditioned matrices produces large/invalid
// directions near asymmetric-cone boundaries), the IPM previously had no
// in-loop termination check on CUDA and spun to MaxIterations. CPU has
// `strategy_checkpoint_small_step` doing exactly this.
//
// This unit test calls `small_step_terminate_kernel` directly with a
// mix of small/large α values across batches and verifies it marks only
// the small-α batches as InsufficientProgress (status code 10).
TEST(SmallStepTerminate, MarksSmallAlphaAsInsufficientProgress) {
    constexpr int64_t batchSize = 4;
    constexpr double minTerm = 1e-4;

    // Batch 0: α just above threshold -> NOT marked.
    // Batch 1: α equals threshold -> marked (kernel uses <=).
    // Batch 2: α well below threshold -> marked.
    // Batch 3: α already-terminated batch (Solved=1) -> status unchanged.
    std::vector<double> h_alpha = {0.5, minTerm, 1e-10, 0.42};
    std::vector<int32_t> h_status_in = {0, 0, 0,
                                        static_cast<int32_t>(SolverStatus::Solved)};
    std::vector<int32_t> h_iters_in = {0, 0, 0, 7};

    double* d_alpha = nullptr;
    int32_t* d_status = nullptr;
    int32_t* d_iters = nullptr;
    cudaMalloc(&d_alpha, sizeof(double) * batchSize);
    cudaMalloc(&d_status, sizeof(int32_t) * batchSize);
    cudaMalloc(&d_iters, sizeof(int32_t) * batchSize);
    cudaMemcpy(d_alpha, h_alpha.data(), sizeof(double) * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_status, h_status_in.data(), sizeof(int32_t) * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_iters, h_iters_in.data(), sizeof(int32_t) * batchSize, cudaMemcpyHostToDevice);

    small_step_terminate_kernel(d_alpha, d_status, d_iters, minTerm, /*iter=*/42, batchSize);
    cudaDeviceSynchronize();

    std::vector<int32_t> h_status_out(batchSize);
    std::vector<int32_t> h_iters_out(batchSize);
    cudaMemcpy(h_status_out.data(), d_status, sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_iters_out.data(), d_iters, sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);

    const int32_t IP = static_cast<int32_t>(SolverStatus::InsufficientProgress);
    EXPECT_EQ(h_status_out[0], 0)  << "Batch 0 (α > threshold) must stay Unsolved";
    EXPECT_EQ(h_status_out[1], IP) << "Batch 1 (α == threshold) must become InsufficientProgress";
    EXPECT_EQ(h_status_out[2], IP) << "Batch 2 (α << threshold) must become InsufficientProgress";
    EXPECT_EQ(h_status_out[3], static_cast<int32_t>(SolverStatus::Solved))
        << "Batch 3 (already Solved) must keep existing status";

    EXPECT_EQ(h_iters_out[0], 0)  << "Batch 0 iter unchanged";
    EXPECT_EQ(h_iters_out[1], 42) << "Batch 1 iter set to current iter";
    EXPECT_EQ(h_iters_out[2], 42) << "Batch 2 iter set to current iter";
    EXPECT_EQ(h_iters_out[3], 7)  << "Batch 3 iter unchanged";

    cudaFree(d_alpha);
    cudaFree(d_status);
    cudaFree(d_iters);
}

// Original (unhelpful) attempt at a problem-level regression test; kept as
// a smoke test that the boundary regime still converges (in some status)
// without exploding the iteration count.
TEST_F(PowerconeTest, BoundaryRegimeTerminatesBeforeMaxIter) {
    constexpr int64_t numStacks = 32;       // 32 stacked 3D power cones
    constexpr int64_t n_eq = 24;            // tight equality constraints
    constexpr int64_t n = 3 * numStacks;    // = 96
    constexpr int64_t m = n_eq + n;         // = 120 (slack form: [B; -I])
    constexpr double  alpha = 0.5;
    constexpr double  eps_P = 1e-6;         // boundary regime: P near singular

    std::vector<double> alphas(static_cast<size_t>(numStacks), alpha);

    Cones cones{};
    cones.numZeroCones = n_eq;
    cones.numPowerCones = numStacks;
    cones.powerAlphas = alphas;

    // P pattern: diagonal
    std::vector<int64_t> P_ro(static_cast<size_t>(n + 1));
    std::vector<int64_t> P_ci(static_cast<size_t>(n));
    for (int64_t i = 0; i <= n; ++i) P_ro[static_cast<size_t>(i)] = i;
    for (int64_t i = 0; i < n; ++i) P_ci[static_cast<size_t>(i)] = i;

    // A pattern: top block dense n_eq × n (zero cone), bottom block -I
    // (power slack).
    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    A_ro.reserve(static_cast<size_t>(m + 1));
    A_ro.push_back(0);
    int64_t nnz = 0;
    for (int64_t r = 0; r < n_eq; ++r) {
        for (int64_t j = 0; j < n; ++j) { A_ci.push_back(j); ++nnz; }
        A_ro.push_back(nnz);
    }
    for (int64_t r = 0; r < n; ++r) {
        A_ci.push_back(r); ++nnz;
        A_ro.push_back(nnz);
    }
    const int64_t nnzA = nnz;
    const int64_t nnzP = n;

    // Deterministic values. x0 = [1,1,0]^stack puts each cone at the
    // boundary face s_3=0 of its third coordinate — the worst-conditioned
    // point for the power-cone Hessian.
    std::mt19937_64 rng(137ULL * 100003ULL);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<double> target(static_cast<size_t>(n));
    for (int64_t j = 0; j < n; ++j) target[static_cast<size_t>(j)] = 0.5 * normal(rng);

    std::vector<double> B_vals(static_cast<size_t>(n_eq * n));
    const double scale = 1.0 / std::sqrt(static_cast<double>(n));
    for (int64_t i = 0; i < n_eq * n; ++i) B_vals[static_cast<size_t>(i)] = scale * normal(rng);

    std::vector<double> x0(static_cast<size_t>(n));
    for (int64_t k = 0; k < numStacks; ++k) {
        x0[static_cast<size_t>(3 * k + 0)] = 1.0;
        x0[static_cast<size_t>(3 * k + 1)] = 1.0;
        x0[static_cast<size_t>(3 * k + 2)] = 0.0;
    }

    std::vector<double> c_eq(static_cast<size_t>(n_eq), 0.0);
    for (int64_t r = 0; r < n_eq; ++r) {
        double acc = 0.0;
        for (int64_t j = 0; j < n; ++j) {
            acc += B_vals[static_cast<size_t>(r * n + j)] * x0[static_cast<size_t>(j)];
        }
        c_eq[static_cast<size_t>(r)] = acc;
    }

    std::vector<double> P_values(static_cast<size_t>(n), eps_P);
    std::vector<double> A_values(static_cast<size_t>(nnzA));
    std::copy(B_vals.begin(), B_vals.end(), A_values.begin());
    for (int64_t r = 0; r < n; ++r) A_values[static_cast<size_t>(n_eq * n + r)] = -1.0;

    std::vector<double> q_values = target;
    std::vector<double> b_values(static_cast<size_t>(m), 0.0);
    std::copy(c_eq.begin(), c_eq.end(), b_values.begin());

    Settings settings;
    settings.maxIter = 300;
    settings.verbose = false;

    CompiledSolver solver(
        n, m, /*batchSize=*/1,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP);
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_values.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_values.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<int32_t> status(1);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    const int32_t iters = solver.info.iterations;
    const int32_t s0 = status[0];

    // Must not hit MaxIter — the small-step kernel must terminate the loop
    // well before settings.maxIter. cuDSS nondeterminism means iter count
    // varies a couple counts run-to-run, but should stay well below 300.
    EXPECT_LT(iters, 60)
        << "Boundary-regime power-cone solve ran to " << iters
        << " iters; expected termination via small_step_terminate well below maxIter="
        << settings.maxIter << ". The IPM is failing to terminate when cone "
           "backtracking forces α toward 0.";

    EXPECT_TRUE(s0 == static_cast<int32_t>(SolverStatus::Solved) ||
                s0 == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Expected Solved or AlmostSolved, got status code " << s0;

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
