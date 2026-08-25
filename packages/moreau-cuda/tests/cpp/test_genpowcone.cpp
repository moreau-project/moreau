/**
 * @file test_genpowcone.cpp
 * @brief Generalized power cone problem tests with known solutions
 *
 * GenPowerCone(α, dim2) = {(p,w) : ∏ p_i^αi ≥ ||w||₂, p_i ≥ 0}
 * where α_i > 0, Σα_i = 1, p has dim1 = len(α) entries, w has dim2 entries.
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include "moreau/solver/solver.hpp"

using namespace moreau;

namespace {

/**
 * @brief Sample a point strictly inside the generalized power cone.
 *
 * Returns (p, w) where p has dim1 entries and w has dim2 entries,
 * satisfying ∏ p_i^(2αi) > ||w||₂² with margin.
 */
std::vector<double> sampleGenPowconeInterior(
    std::mt19937_64& rng,
    const std::vector<double>& alphas,
    int64_t dim2,
    double marginFactor = 0.5
) {
    int64_t dim1 = static_cast<int64_t>(alphas.size());
    std::vector<double> result(dim1 + dim2);

    std::uniform_real_distribution<double> unif(0.5, 2.0);

    // Generate p_i > 0
    for (int64_t i = 0; i < dim1; i++) {
        result[i] = unif(rng);
    }

    // Compute ∏ p_i^αi
    double prod = 1.0;
    for (int64_t i = 0; i < dim1; i++) {
        prod *= std::pow(result[i], alphas[i]);
    }

    // Generate w with ||w||₂ < prod * marginFactor
    double maxNorm = prod * marginFactor;
    std::normal_distribution<double> normal(0.0, 1.0);
    double wNormSq = 0.0;
    for (int64_t i = 0; i < dim2; i++) {
        result[dim1 + i] = normal(rng);
        wNormSq += result[dim1 + i] * result[dim1 + i];
    }
    if (wNormSq > 0.0 && dim2 > 0) {
        double scale = maxNorm / (std::sqrt(wNormSq) + 1e-12);
        for (int64_t i = 0; i < dim2; i++) {
            result[dim1 + i] *= scale;
        }
    }

    return result;
}

/**
 * @brief Check if point (p, w) is in GenPowerCone(α, dim2).
 * Primal: ∏ p_i^αi ≥ ||w||₂, p_i ≥ 0
 */
bool isInGenPowcone(
    const double* v,
    const std::vector<double>& alphas,
    int64_t dim2,
    double tol = 1e-8
) {
    int64_t dim1 = static_cast<int64_t>(alphas.size());

    for (int64_t i = 0; i < dim1; i++) {
        if (v[i] < -tol) return false;
    }

    double prod = 1.0;
    for (int64_t i = 0; i < dim1; i++) {
        prod *= std::pow(std::max(v[i], 0.0), alphas[i]);
    }

    double wNormSq = 0.0;
    for (int64_t i = 0; i < dim2; i++) {
        wNormSq += v[dim1 + i] * v[dim1 + i];
    }

    return prod + tol >= std::sqrt(wNormSq);
}

/**
 * @brief Check if point is in the dual of GenPowerCone.
 * Dual: ∏ (z_i/αi)^αi ≥ ||w||₂, z_i ≥ 0
 */
bool isInDualGenPowcone(
    const double* v,
    const std::vector<double>& alphas,
    int64_t dim2,
    double tol = 1e-8
) {
    int64_t dim1 = static_cast<int64_t>(alphas.size());

    for (int64_t i = 0; i < dim1; i++) {
        if (v[i] < -tol) return false;
    }

    double prod = 1.0;
    for (int64_t i = 0; i < dim1; i++) {
        double zi = std::max(v[i], 0.0);
        prod *= std::pow(zi / alphas[i], alphas[i]);
    }

    double wNormSq = 0.0;
    for (int64_t i = 0; i < dim2; i++) {
        wNormSq += v[dim1 + i] * v[dim1 + i];
    }

    return prod + tol >= std::sqrt(wNormSq);
}

/**
 * @brief Generated GenPowerCone problem with known solution.
 *
 * minimize    (1/2) x'Px + q'x   with P = I
 * subject to  Ax + s = b,  s ∈ K (product of GenPowerCones)
 */
struct GeneratedGenPowconeProblem {
    int64_t n;
    int64_t numCones;
    int64_t m;

    // Per-cone parameters
    std::vector<std::vector<double>> coneAlphas;   // alphas for each cone
    std::vector<int64_t> coneDim1s;                // dim1 per cone
    std::vector<int64_t> coneDim2s;                // dim2 per cone
    std::vector<int64_t> coneOffsets;              // prefix sum of (dim1+dim2)

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

    /**
     * @param n_vars Number of decision variables
     * @param alphas_list Vector of alpha vectors, one per cone
     * @param dim2s Vector of dim2 per cone
     * @param margin Margin factor for interior sampling
     * @param seed RNG seed
     */
    GeneratedGenPowconeProblem(
        int64_t n_vars,
        const std::vector<std::vector<double>>& alphas_list,
        const std::vector<int64_t>& dim2s,
        double margin = 0.4,
        uint64_t seed = 0
    ) : n(n_vars),
        numCones(static_cast<int64_t>(alphas_list.size())),
        coneAlphas(alphas_list),
        coneDim2s(dim2s)
    {
        // Compute dimensions
        coneDim1s.resize(numCones);
        coneOffsets.resize(numCones + 1, 0);
        m = 0;
        for (int64_t k = 0; k < numCones; k++) {
            coneDim1s[k] = static_cast<int64_t>(alphas_list[k].size());
            int64_t dim = coneDim1s[k] + dim2s[k];
            m += dim;
            coneOffsets[k + 1] = m;
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
            auto block = sampleGenPowconeInterior(rng, coneAlphas[k], coneDim2s[k], margin);
            int64_t off = coneOffsets[k];
            int64_t dim = coneDim1s[k] + coneDim2s[k];
            for (int64_t i = 0; i < dim; i++) {
                s_star[off + i] = block[i];
            }
        }

        b.assign(m, 0.0);
        for (int64_t i = 0; i < m; i++) {
            double Ax_i = 0.0;
            for (int64_t j = 0; j < n; j++) {
                Ax_i += A_dense[i][j] * x_star[j];
            }
            b[i] = Ax_i + s_star[i];
        }

        // CSR for A (dense — all entries)
        A_ro.reserve(m + 1);
        A_ro.push_back(0);
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                A_ci.push_back(j);
                A_val.push_back(A_dense[i][j]);
            }
            A_ro.push_back(static_cast<int64_t>(A_val.size()));
        }

        // P = I (diagonal)
        P_ro.reserve(n + 1);
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
        for (int64_t k = 0; k < numCones; k++) {
            if (!isInGenPowcone(
                s.data() + coneOffsets[k],
                coneAlphas[k], coneDim2s[k], tol)) {
                return false;
            }
        }
        return true;
    }

    bool dualInCone(const std::vector<double>& z, double tol = 1e-8) const {
        for (int64_t k = 0; k < numCones; k++) {
            if (!isInDualGenPowcone(
                z.data() + coneOffsets[k],
                coneAlphas[k], coneDim2s[k], tol)) {
                return false;
            }
        }
        return true;
    }

    /** Build flattened Cones struct for the solver. */
    Cones buildCones() const {
        Cones cones{};
        cones.numGenPowerCones = numCones;
        for (int64_t k = 0; k < numCones; k++) {
            for (double a : coneAlphas[k]) {
                cones.genPowerAlphas.push_back(a);
            }
            cones.genPowerDim1s.push_back(coneDim1s[k]);
            cones.genPowerDim2s.push_back(coneDim2s[k]);
        }
        return cones;
    }
};

/** Check KKT conditions for a GenPowerCone problem. */
bool checkKKTGenPowcone(
    const GeneratedGenPowconeProblem& prob,
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

    // Primal feasibility: Ax + s - b = 0
    double primal_res_inf = 0.0;
    for (int64_t i = 0; i < prob.m; i++) {
        double r = -prob.b[i] + s[i];
        for (int64_t j = 0; j < prob.n; j++) {
            r += prob.A_dense[i][j] * x[j];
        }
        primal_res_inf = std::max(primal_res_inf, std::abs(r));
    }
    bool primal_ok = primal_res_inf <= tol_feas;

    // Stationarity: x + q + A'y = 0
    double stationarity_inf = 0.0;
    for (int64_t j = 0; j < prob.n; j++) {
        double grad = x[j] + prob.q[j];
        for (int64_t i = 0; i < prob.m; i++) {
            grad += prob.A_dense[i][j] * y[i];
        }
        stationarity_inf = std::max(stationarity_inf, std::abs(grad));
    }
    bool stationarity_ok = stationarity_inf <= tol_kkt;

    // Cone membership
    bool s_in_K = prob.slackInCone(s, tol_feas);
    bool y_in_K = prob.dualInCone(y, tol_feas);

    // Complementarity per cone block
    double comp_inf = 0.0;
    for (int64_t k = 0; k < prob.numCones; k++) {
        double dot = 0.0;
        int64_t off = prob.coneOffsets[k];
        int64_t dim = prob.coneDim1s[k] + prob.coneDim2s[k];
        for (int64_t i = 0; i < dim; i++) {
            dot += s[off + i] * y[off + i];
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

/** Helper to run solver and extract host-side results. */
struct SolveResult {
    std::vector<double> x, s, z;
    std::vector<int32_t> status;
};

SolveResult solveGenPowconeProblem(
    const GeneratedGenPowconeProblem& prob,
    int64_t batchSize = 1,
    int64_t maxIter = 300
) {
    Cones cones = prob.buildCones();

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = maxIter;
    settings.verbose = false;

    CompiledSolver solver(
        prob.n, prob.m, batchSize,
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

    SolveResult res;
    res.x.resize(prob.n * batchSize);
    res.s.resize(prob.m * batchSize);
    res.z.resize(prob.m * batchSize);
    res.status.resize(batchSize);

    cudaMemcpy(res.x.data(), solver.solution.x.data(), sizeof(double) * prob.n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(res.s.data(), solver.solution.s.data(), sizeof(double) * prob.m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(res.z.data(), solver.solution.z.data(), sizeof(double) * prob.m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(res.status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);

    return res;
}

} // namespace

class GenPowconeTest : public ::testing::Test {
protected:
    void SetUp() override { cudaGetLastError(); }
    void TearDown() override { cudaDeviceSynchronize(); }
};

// --- Test: 2 alphas, dim2=1 (simplest case, similar to regular power cone) ---
TEST_F(GenPowconeTest, TwoAlphasDim2One) {
    std::vector<std::vector<double>> alphas = {{0.6, 0.4}};
    std::vector<int64_t> dim2s = {1};

    GeneratedGenPowconeProblem prob(8, alphas, dim2s, 0.4, 100);
    auto res = solveGenPowconeProblem(prob);

    EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "status=" << res.status[0];

    double max_x_err = 0.0;
    for (int64_t i = 0; i < prob.n; i++) {
        max_x_err = std::max(max_x_err, std::abs(res.x[i] - prob.x_star[i]));
    }
    EXPECT_LT(max_x_err, 1e-5);
    EXPECT_LT(prob.primalResidualInf(res.x, res.s), 1e-6);
    EXPECT_TRUE(prob.slackInCone(res.s, 1e-7));
    EXPECT_TRUE(checkKKTGenPowcone(prob, res.x, res.s, res.z, 1e-6, 1e-6, true));
}

// --- Test: 3 alphas, dim2=2 ---
TEST_F(GenPowconeTest, ThreeAlphasDim2Two) {
    std::vector<std::vector<double>> alphas = {{0.5, 0.3, 0.2}};
    std::vector<int64_t> dim2s = {2};

    GeneratedGenPowconeProblem prob(10, alphas, dim2s, 0.4, 200);
    auto res = solveGenPowconeProblem(prob);

    EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "status=" << res.status[0];

    double max_x_err = 0.0;
    for (int64_t i = 0; i < prob.n; i++) {
        max_x_err = std::max(max_x_err, std::abs(res.x[i] - prob.x_star[i]));
    }
    EXPECT_LT(max_x_err, 1e-5);
    EXPECT_LT(prob.primalResidualInf(res.x, res.s), 1e-6);
    EXPECT_TRUE(prob.slackInCone(res.s, 1e-7));
    EXPECT_TRUE(checkKKTGenPowcone(prob, res.x, res.s, res.z, 1e-6, 1e-6, true));
}

// --- Test: 4 alphas, dim2=3 (larger cone) ---
TEST_F(GenPowconeTest, FourAlphasDim2Three) {
    std::vector<std::vector<double>> alphas = {{0.4, 0.3, 0.2, 0.1}};
    std::vector<int64_t> dim2s = {3};

    GeneratedGenPowconeProblem prob(12, alphas, dim2s, 0.3, 300);
    auto res = solveGenPowconeProblem(prob, 1, 400);

    EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "status=" << res.status[0];

    double max_x_err = 0.0;
    for (int64_t i = 0; i < prob.n; i++) {
        max_x_err = std::max(max_x_err, std::abs(res.x[i] - prob.x_star[i]));
    }
    EXPECT_LT(max_x_err, 1e-4);
    EXPECT_LT(prob.primalResidualInf(res.x, res.s), 1e-5);
    EXPECT_TRUE(prob.slackInCone(res.s, 1e-6));
    EXPECT_TRUE(checkKKTGenPowcone(prob, res.x, res.s, res.z, 1e-5, 1e-5, true));
}

// --- Test: Multiple GenPowerCones ---
TEST_F(GenPowconeTest, MultipleGenPowerCones) {
    std::vector<std::vector<double>> alphas = {
        {0.6, 0.4},       // dim1=2, dim2=1 => total 3
        {0.5, 0.3, 0.2},  // dim1=3, dim2=2 => total 5
    };
    std::vector<int64_t> dim2s = {1, 2};

    GeneratedGenPowconeProblem prob(10, alphas, dim2s, 0.4, 400);
    auto res = solveGenPowconeProblem(prob, 1, 400);

    EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "status=" << res.status[0];

    EXPECT_LT(prob.primalResidualInf(res.x, res.s), 1e-5);
    EXPECT_TRUE(prob.slackInCone(res.s, 1e-6));
    EXPECT_TRUE(checkKKTGenPowcone(prob, res.x, res.s, res.z, 1e-5, 1e-5, true));
}

// --- Test: Multiple seeds (parametric robustness) ---
TEST_F(GenPowconeTest, MultipleSeeds) {
    std::vector<uint64_t> seeds = {1, 42, 100, 500, 999};

    for (uint64_t seed : seeds) {
        SCOPED_TRACE("seed=" + std::to_string(seed));

        std::vector<std::vector<double>> alphas = {{0.5, 0.3, 0.2}};
        std::vector<int64_t> dim2s = {2};

        GeneratedGenPowconeProblem prob(8, alphas, dim2s, 0.4, seed);
        auto res = solveGenPowconeProblem(prob);

        EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                    res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "seed=" << seed << " status=" << res.status[0];

        EXPECT_LT(prob.primalResidualInf(res.x, res.s), 1e-5);
        EXPECT_TRUE(prob.slackInCone(res.s, 1e-6));
    }
}

// --- Test: Batched solve ---
TEST_F(GenPowconeTest, BatchedSolve) {
    const int64_t batchSize = 2;

    std::vector<std::vector<double>> alphas = {{0.6, 0.4}};
    std::vector<int64_t> dim2s = {1};

    GeneratedGenPowconeProblem prob1(8, alphas, dim2s, 0.4, 77);
    GeneratedGenPowconeProblem prob2(8, alphas, dim2s, 0.5, 1234);

    Cones cones = prob1.buildCones();

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 300;
    settings.verbose = false;

    CompiledSolver solver(
        prob1.n, prob1.m, batchSize,
        prob1.P_ro.data(), prob1.P_ci.data(), prob1.nnzP(),
        prob1.A_ro.data(), prob1.A_ci.data(), prob1.nnzA(),
        cones,
        settings
    );

    const int64_t nnzP = prob1.nnzP();
    const int64_t nnzA = prob1.nnzA();

    std::vector<double> P_values(nnzP * batchSize);
    std::vector<double> A_values(nnzA * batchSize);
    std::vector<double> q_values(prob1.n * batchSize);
    std::vector<double> b_values(prob1.m * batchSize);

    std::vector<GeneratedGenPowconeProblem*> probs = {&prob1, &prob2};
    for (int64_t bi = 0; bi < batchSize; bi++) {
        auto* p = probs[bi];
        std::copy(p->P_val.begin(), p->P_val.end(), P_values.begin() + bi * nnzP);
        std::copy(p->A_val.begin(), p->A_val.end(), A_values.begin() + bi * nnzA);
        std::copy(p->q.begin(), p->q.end(), q_values.begin() + bi * p->n);
        std::copy(p->b.begin(), p->b.end(), b_values.begin() + bi * p->m);
    }

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * prob1.n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * prob1.m * batchSize);

    cudaMemcpy(d_P, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_values.data(), sizeof(double) * prob1.n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_values.data(), sizeof(double) * prob1.m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(prob1.n * batchSize);
    std::vector<double> s_sol(prob1.m * batchSize);
    std::vector<double> z_sol(prob1.m * batchSize);
    std::vector<int32_t> status(batchSize);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * prob1.n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * prob1.m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * prob1.m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int64_t bi = 0; bi < batchSize; bi++) {
        SCOPED_TRACE("batch=" + std::to_string(bi));
        auto* p = probs[bi];

        EXPECT_TRUE(status[bi] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status[bi] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "batch=" << bi << " status=" << status[bi];

        std::vector<double> x_batch(x_sol.begin() + bi * p->n, x_sol.begin() + (bi + 1) * p->n);
        std::vector<double> s_batch(s_sol.begin() + bi * p->m, s_sol.begin() + (bi + 1) * p->m);
        std::vector<double> z_batch(z_sol.begin() + bi * p->m, z_sol.begin() + (bi + 1) * p->m);

        EXPECT_LT(p->primalResidualInf(x_batch, s_batch), 2e-5);
        EXPECT_TRUE(p->slackInCone(s_batch, 1e-6));
        EXPECT_TRUE(p->dualInCone(z_batch, 1e-6));

        bool kkt_ok = checkKKTGenPowcone(*p, x_batch, s_batch, z_batch, 5e-6, 5e-6, false);
        if (!kkt_ok) {
            checkKKTGenPowcone(*p, x_batch, s_batch, z_batch, 5e-6, 5e-6, true);
        }
        EXPECT_TRUE(kkt_ok);
    }

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

// --- Test: Objective matches ground truth ---
TEST_F(GenPowconeTest, ObjectiveMatchesGroundTruth) {
    std::vector<std::vector<double>> alphas = {{0.5, 0.3, 0.2}};
    std::vector<int64_t> dim2s = {2};

    GeneratedGenPowconeProblem prob(10, alphas, dim2s, 0.4, 2027);
    auto res = solveGenPowconeProblem(prob);

    EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved));

    double obj_expected = prob.objective(prob.x_star);
    double obj_solver = prob.objective(res.x);
    EXPECT_NEAR(obj_solver, obj_expected, 1e-5);
}

// --- Test: High-dim cone (dim1=10, dim2=6, total=16) ---
TEST_F(GenPowconeTest, HighDim16) {
    // 10 alphas summing to 1, dim2=6 => total dim=16
    std::vector<double> alphas_vec(10);
    for (int i = 0; i < 10; i++) alphas_vec[i] = 0.1;
    std::vector<std::vector<double>> alphas = {alphas_vec};
    std::vector<int64_t> dim2s = {6};

    GeneratedGenPowconeProblem prob(20, alphas, dim2s, 0.3, 42);
    auto res = solveGenPowconeProblem(prob, 1, 500);

    EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "status=" << res.status[0];

    EXPECT_LT(prob.primalResidualInf(res.x, res.s), 1e-5);
    EXPECT_TRUE(prob.slackInCone(res.s, 1e-6));
    EXPECT_TRUE(checkKKTGenPowcone(prob, res.x, res.s, res.z, 1e-5, 1e-5, true));
}

// --- Test: dim1=20, dim2=12, total=32 ---
TEST_F(GenPowconeTest, Dim32) {
    // 20 alphas summing to 1, dim2=12 => total dim=32
    std::vector<double> alphas_vec(20);
    for (int i = 0; i < 20; i++) alphas_vec[i] = 1.0 / 20.0;
    std::vector<std::vector<double>> alphas = {alphas_vec};
    std::vector<int64_t> dim2s = {12};

    GeneratedGenPowconeProblem prob(40, alphas, dim2s, 0.3, 123);
    auto res = solveGenPowconeProblem(prob, 1, 500);

    EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "status=" << res.status[0];

    EXPECT_LT(prob.primalResidualInf(res.x, res.s), 1e-4);
    EXPECT_TRUE(prob.slackInCone(res.s, 1e-5));
    EXPECT_TRUE(checkKKTGenPowcone(prob, res.x, res.s, res.z, 1e-4, 1e-4, true));
}

// --- Test: Multiple seeds at dim=32 ---
TEST_F(GenPowconeTest, Dim32MultipleSeeds) {
    std::vector<uint64_t> seeds = {7, 77, 777};

    std::vector<double> alphas_vec(20);
    for (int i = 0; i < 20; i++) alphas_vec[i] = 1.0 / 20.0;

    for (uint64_t seed : seeds) {
        SCOPED_TRACE("seed=" + std::to_string(seed));

        std::vector<std::vector<double>> alphas = {alphas_vec};
        std::vector<int64_t> dim2s = {12};

        GeneratedGenPowconeProblem prob(40, alphas, dim2s, 0.3, seed);
        auto res = solveGenPowconeProblem(prob, 1, 500);

        EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                    res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "seed=" << seed << " status=" << res.status[0];

        EXPECT_LT(prob.primalResidualInf(res.x, res.s), 1e-4);
        EXPECT_TRUE(prob.slackInCone(res.s, 1e-5));
    }
}

// --- Test: Single alpha (degenerate, like SOC) ---
TEST_F(GenPowconeTest, SingleAlpha) {
    // Single alpha = 1.0, dim2 = 2 => K = {(p, w) : p ≥ ||w||₂, p ≥ 0}
    // This is equivalent to a second-order cone
    std::vector<std::vector<double>> alphas = {{1.0}};
    std::vector<int64_t> dim2s = {2};

    GeneratedGenPowconeProblem prob(6, alphas, dim2s, 0.4, 555);
    auto res = solveGenPowconeProblem(prob);

    EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "status=" << res.status[0];

    EXPECT_LT(prob.primalResidualInf(res.x, res.s), 1e-5);
    EXPECT_TRUE(prob.slackInCone(res.s, 1e-6));
}

// --- Test: Large cone dim=64 (dim1=40, dim2=24) ---
TEST_F(GenPowconeTest, Dim64) {
    std::vector<double> alphas_vec(40);
    for (int i = 0; i < 40; i++) alphas_vec[i] = 1.0 / 40.0;
    std::vector<std::vector<double>> alphas = {alphas_vec};
    std::vector<int64_t> dim2s = {24};

    GeneratedGenPowconeProblem prob(80, alphas, dim2s, 0.3, 64);
    auto res = solveGenPowconeProblem(prob, 1, 500);

    EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "status=" << res.status[0];

    EXPECT_LT(prob.primalResidualInf(res.x, res.s), 1e-4);
    EXPECT_TRUE(prob.slackInCone(res.s, 1e-5));
    EXPECT_TRUE(checkKKTGenPowcone(prob, res.x, res.s, res.z, 1e-4, 1e-4, true));
}

// --- Test: Large cone dim=128 (dim1=80, dim2=48) ---
TEST_F(GenPowconeTest, Dim128) {
    std::vector<double> alphas_vec(80);
    for (int i = 0; i < 80; i++) alphas_vec[i] = 1.0 / 80.0;
    std::vector<std::vector<double>> alphas = {alphas_vec};
    std::vector<int64_t> dim2s = {48};

    GeneratedGenPowconeProblem prob(160, alphas, dim2s, 0.3, 128);
    auto res = solveGenPowconeProblem(prob, 1, 500);

    EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "status=" << res.status[0];

    EXPECT_LT(prob.primalResidualInf(res.x, res.s), 1e-4);
    EXPECT_TRUE(prob.slackInCone(res.s, 1e-5));
    EXPECT_TRUE(checkKKTGenPowcone(prob, res.x, res.s, res.z, 1e-4, 1e-4, true));
}

// Exercises the block-per-cone GenPow kernel split: two large cones (dim=64,
// dim=128) alongside a small cone (dim=3). Small cones route through the
// composite thread-per-cone kernel; large cones use the parallel kernel.
// Asserts numLargeGenPow == 2 to confirm the partition.
TEST_F(GenPowconeTest, BlockPerConeLargeAndSmallMixed) {
    std::vector<double> alphas_small = {0.5, 0.5};
    std::vector<double> alphas_mid(40, 1.0 / 40.0);
    std::vector<double> alphas_big(80, 1.0 / 80.0);
    std::vector<std::vector<double>> alphas = {alphas_small, alphas_mid, alphas_big};
    std::vector<int64_t> dim2s = {1, 24, 48};  // total dims: 3, 64, 128

    GeneratedGenPowconeProblem prob(200, alphas, dim2s, 0.3, 42);

    // Sanity-check the partition via a freshly-initialized Cones (the helper
    // does not expose the solver-internal Cones).
    {
        Cones probe = prob.buildCones();
        probe.initialize(1, 0);
        EXPECT_EQ(probe.numGenPowerCones, 3);
        EXPECT_EQ(probe.numLargeGenPow, 2);
    }

    auto res = solveGenPowconeProblem(prob, 1, 500);
    EXPECT_TRUE(res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "status=" << res.status[0];
    EXPECT_LT(prob.primalResidualInf(res.x, res.s), 1e-4);
    EXPECT_TRUE(prob.slackInCone(res.s, 1e-5));
    EXPECT_TRUE(checkKKTGenPowcone(prob, res.x, res.s, res.z, 1e-4, 1e-4, true));
}

// --- Negative tests: invalid GenPowerCone metadata should throw ---

TEST_F(GenPowconeTest, RejectNegativeAlpha) {
    Cones cones{};
    cones.numGenPowerCones = 1;
    cones.genPowerDim1s = {2};
    cones.genPowerDim2s = {1};
    cones.genPowerAlphas = {-0.3, 1.3};  // negative alpha
    EXPECT_THROW(cones.initialize(1, 0), std::invalid_argument);
}

TEST_F(GenPowconeTest, RejectAlphaSumNotOne) {
    Cones cones{};
    cones.numGenPowerCones = 1;
    cones.genPowerDim1s = {2};
    cones.genPowerDim2s = {1};
    cones.genPowerAlphas = {0.3, 0.3};  // sum = 0.6, not 1
    EXPECT_THROW(cones.initialize(1, 0), std::invalid_argument);
}

TEST_F(GenPowconeTest, RejectDim2Zero) {
    Cones cones{};
    cones.numGenPowerCones = 1;
    cones.genPowerDim1s = {2};
    cones.genPowerDim2s = {0};  // invalid
    cones.genPowerAlphas = {0.5, 0.5};
    EXPECT_THROW(cones.initialize(1, 0), std::invalid_argument);
}

TEST_F(GenPowconeTest, RejectAlphasSizeMismatch) {
    Cones cones{};
    cones.numGenPowerCones = 1;
    cones.genPowerDim1s = {3};
    cones.genPowerDim2s = {1};
    cones.genPowerAlphas = {0.5, 0.5};  // 2 alphas but dim1=3
    EXPECT_THROW(cones.initialize(1, 0), std::invalid_argument);
}

// ============================================================================
// Fuzz test: random GenPowerCone configurations across many seeds
// ============================================================================

namespace {

/**
 * @brief Generate random alphas that sum to 1 with dim1 entries.
 * Uses Dirichlet-like sampling (exponential variates normalized).
 */
std::vector<double> randomAlphas(std::mt19937_64& rng, int64_t dim1) {
    std::exponential_distribution<double> exp_dist(1.0);
    std::vector<double> alphas(dim1);
    double sum = 0.0;
    for (int64_t i = 0; i < dim1; i++) {
        alphas[i] = exp_dist(rng);
        sum += alphas[i];
    }
    // Normalize to sum to 1, with a floor to avoid near-zero alphas
    for (int64_t i = 0; i < dim1; i++) {
        alphas[i] = std::max(alphas[i] / sum, 0.01);
    }
    // Re-normalize after floor
    sum = 0.0;
    for (auto a : alphas) sum += a;
    for (auto& a : alphas) a /= sum;
    return alphas;
}

struct FuzzConfig {
    std::vector<std::vector<double>> alphas_list;
    std::vector<int64_t> dim2s;
    int64_t n_vars;
    std::string label;
};

FuzzConfig randomFuzzConfig(std::mt19937_64& rng) {
    FuzzConfig cfg;
    std::uniform_int_distribution<int64_t> numConesDist(1, 4);
    std::uniform_int_distribution<int64_t> dim1Dist(2, 10);
    std::uniform_int_distribution<int64_t> dim2Dist(1, 8);

    int64_t numCones = numConesDist(rng);
    int64_t totalM = 0;
    for (int64_t k = 0; k < numCones; k++) {
        int64_t d1 = dim1Dist(rng);
        int64_t d2 = dim2Dist(rng);
        cfg.alphas_list.push_back(randomAlphas(rng, d1));
        cfg.dim2s.push_back(d2);
        totalM += d1 + d2;
    }
    // n_vars >= totalM to keep problems well-posed
    cfg.n_vars = std::max(totalM, static_cast<int64_t>(6)) + 4;

    // Build label
    cfg.label = std::to_string(numCones) + "cones[";
    for (int64_t k = 0; k < numCones; k++) {
        if (k > 0) cfg.label += ",";
        cfg.label += "(" + std::to_string(cfg.alphas_list[k].size())
                   + "+" + std::to_string(cfg.dim2s[k]) + ")";
    }
    cfg.label += "]n=" + std::to_string(cfg.n_vars);
    return cfg;
}

} // namespace

TEST_F(GenPowconeTest, FuzzRandomConfigs) {
    const int numProblems = 200;
    const int64_t maxIter = 500;
    const double tol_kkt = 1e-4;
    const double tol_feas = 1e-4;
    const double tol_cone = 1e-5;

    int passed = 0;
    int failed = 0;

    std::mt19937_64 metaRng(20260324);

    for (int trial = 0; trial < numProblems; trial++) {
        // Generate random config using meta-RNG
        auto cfg = randomFuzzConfig(metaRng);
        uint64_t problemSeed = metaRng();

        SCOPED_TRACE("trial=" + std::to_string(trial)
                   + " seed=" + std::to_string(problemSeed)
                   + " " + cfg.label);

        GeneratedGenPowconeProblem prob(
            cfg.n_vars, cfg.alphas_list, cfg.dim2s, 0.35, problemSeed);
        auto res = solveGenPowconeProblem(prob, 1, maxIter);

        bool solved = (res.status[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                       res.status[0] == static_cast<int32_t>(SolverStatus::AlmostSolved));

        if (!solved) {
            failed++;
            ADD_FAILURE() << "trial=" << trial
                          << " seed=" << problemSeed
                          << " " << cfg.label
                          << " status=" << res.status[0];
            continue;
        }

        std::vector<double> x_batch(res.x.begin(), res.x.begin() + prob.n);
        std::vector<double> s_batch(res.s.begin(), res.s.begin() + prob.m);
        std::vector<double> z_batch(res.z.begin(), res.z.begin() + prob.m);

        double pres = prob.primalResidualInf(x_batch, s_batch);
        bool cone_ok = prob.slackInCone(s_batch, tol_cone);
        bool kkt_ok = checkKKTGenPowcone(prob, x_batch, s_batch, z_batch,
                                          tol_feas, tol_kkt, false);

        if (pres > tol_feas || !cone_ok || !kkt_ok) {
            failed++;
            ADD_FAILURE() << "trial=" << trial
                          << " seed=" << problemSeed
                          << " " << cfg.label
                          << " pres=" << pres
                          << " cone_ok=" << cone_ok
                          << " kkt_ok=" << kkt_ok;
            if (!kkt_ok) {
                checkKKTGenPowcone(prob, x_batch, s_batch, z_batch,
                                   tol_feas, tol_kkt, true);
            }
        } else {
            passed++;
        }
    }

    std::cout << "GenPowcone fuzz: " << passed << "/" << numProblems
              << " passed, " << failed << " failed\n";
    EXPECT_EQ(failed, 0);
}

// --- Regression: Riccati auto-selection picked Riccati for genpow problems
// with at least one zero cone, leading to step-length collapse around iter
// 8–10 and a MaxIterations exit. The auto-selector now rejects any genpow
// (rank-9 sparse expansion) and falls through to cuDSS.
//
// Minimal repro: 1 zero cone (pinning x[0]=1) + GenPowerCone(α=[0.25]^4,
// dim2=8). On main pre-fix this ran with Riccati and stalled at MaxIter;
// post-fix it Solves in ~11 iters via cuDSS.
TEST_F(GenPowconeTest, ZeroPlusGenPowAutoSelectsCuDSS) {
    constexpr int64_t n = 12;
    constexpr int64_t m = 13;  // 1 zero + 12 genpow
    constexpr int64_t batchSize = 1;

    // P = 2·I (diagonal).
    std::vector<int64_t> P_ro(n + 1);
    for (int64_t i = 0; i <= n; ++i) P_ro[i] = i;
    std::vector<int64_t> P_ci(n);
    for (int64_t i = 0; i < n; ++i) P_ci[i] = i;
    std::vector<double> P_val(n, 2.0);
    std::vector<double> q(n, 0.0);
    for (int64_t i = 4; i < n; ++i) q[i] = -1.0;

    // A: 1 zero row pinning x[0]=1, then 12 genpow rows with A = -I.
    std::vector<int64_t> A_ro(m + 1);
    A_ro[0] = 0; A_ro[1] = 1;
    for (int64_t r = 1; r <= n; ++r) A_ro[r + 1] = A_ro[r] + 1;
    std::vector<int64_t> A_ci(m);
    A_ci[0] = 0;
    for (int64_t j = 0; j < n; ++j) A_ci[1 + j] = j;
    std::vector<double> A_val(m);
    A_val[0] = 1.0;
    for (int64_t j = 0; j < n; ++j) A_val[1 + j] = -1.0;

    std::vector<double> b(m, 0.0);
    b[0] = 1.0;

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numGenPowerCones = 1;
    cones.genPowerAlphas = {{0.25, 0.25, 0.25, 0.25}};
    cones.genPowerDim1s = {4};
    cones.genPowerDim2s = {8};

    Settings settings;
    settings.maxIter = 30;
    settings.verbose = false;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), n,
        A_ro.data(), A_ci.data(), m,
        cones, settings);

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * n);
    cudaMalloc(&d_A, sizeof(double) * m);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);
    cudaMemcpy(d_P, P_val.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    int32_t status;
    cudaMemcpy(&status, solver.solution.status.get(),
               sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(d_P); cudaFree(d_A); cudaFree(d_q); cudaFree(d_b);

    EXPECT_TRUE(status == static_cast<int32_t>(SolverStatus::Solved) ||
                status == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Expected Solved/AlmostSolved (auto must NOT pick Riccati for "
        << "genpow problems); got status=" << status;
}
