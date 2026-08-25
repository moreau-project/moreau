// test_qp.cpp
// Consolidated QP tests: simple QP, dense QP, and analytical solution tests
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <stdexcept>

using namespace moreau;

// Test fixture for simple QP tests
class SimpleQPTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test fixture for dense QP tests
class DenseQPTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test fixture for analytical solution tests
class AnalyticalSolutionTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    // Helper function to check solution against expected values
    void checkSolution(
        const std::vector<double>& computed,
        const std::vector<double>& expected,
        double tolerance,
        const std::string& name
    ) {
        ASSERT_EQ(computed.size(), expected.size())
            << name << " size mismatch";
        for (size_t i = 0; i < computed.size(); i++) {
            EXPECT_NEAR(computed[i], expected[i], tolerance)
                << name << "[" << i << "] mismatch";
        }
    }
};

/**
 * Check KKT conditions for:
 *   min 0.5 x^T P x + q^T x  s.t.  A x <= b, x >= 0
 * using the Moreau sign convention (x >= 0 is represented as -x <= 0),
 * so stationarity is: P x + q + A^T y - s = 0 with y >= 0, s >= 0.
 * Returns true if all conditions are satisfied within tolerance.
 */
bool checkQPKKT(
    const std::vector<std::vector<double>>& P,
    const std::vector<double>& q,
    const std::vector<std::vector<double>>& A,
    const std::vector<double>& b,
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& s,
    double tol = 1e-6,
    bool verbose = false
) {
    const int64_t m = A.size();
    const int64_t n = (m > 0) ? static_cast<int64_t>(A[0].size()) : static_cast<int64_t>(x.size());

    auto vecCloseToZero = [&](const std::vector<double>& v, const std::string& name) {
        for (size_t i = 0; i < v.size(); i++) {
            if (std::abs(v[i]) > tol) {
                if (verbose) std::cout << name << "[" << i << "] = " << v[i] << std::endl;
                return false;
            }
        }
        return true;
    };

    bool all_ok = true;

    // Primal feasibility: Ax <= b and x >= 0
    std::vector<double> Ax(m, 0.0);
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            Ax[i] += A[i][j] * x[j];
        }
        if (Ax[i] > b[i] + tol) {
            if (verbose) std::cout << "Primal infeasible at constraint " << i
                                   << ": Ax=" << Ax[i] << " > b=" << b[i] << std::endl;
            all_ok = false;
        }
    }
    for (int64_t j = 0; j < n; j++) {
        if (x[j] < -tol) {
            if (verbose) std::cout << "x[" << j << "] = " << x[j] << " < 0" << std::endl;
            all_ok = false;
        }
    }

    // Dual feasibility: y >= 0, s >= 0
    for (int64_t i = 0; i < m; i++) {
        if (y[i] < -tol) {
            if (verbose) std::cout << "y[" << i << "] = " << y[i] << " < 0" << std::endl;
            all_ok = false;
        }
    }
    for (int64_t j = 0; j < n; j++) {
        if (s[j] < -tol) {
            if (verbose) std::cout << "s[" << j << "] = " << s[j] << " < 0" << std::endl;
            all_ok = false;
        }
    }

    // Stationarity: P x + q + A^T y - s = 0   (note the minus sign)
    std::vector<double> stationarity(n, 0.0);
    for (int64_t i = 0; i < n; i++) {
        double Px_i = 0.0;
        for (int64_t j = 0; j < n; j++) {
            Px_i += P[i][j] * x[j];
        }
        stationarity[i] = Px_i + q[i] - s[i];
        for (int64_t k = 0; k < m; k++) {
            stationarity[i] += A[k][i] * y[k];
        }
    }
    all_ok = all_ok && vecCloseToZero(stationarity, "stationarity");

    // Complementary slackness: y_i * (b_i - Ax_i) = 0, s_j * x_j = 0
    for (int64_t i = 0; i < m; i++) {
        double slack = b[i] - Ax[i];
        double comp = y[i] * slack;
        if (std::abs(comp) > tol) {
            if (verbose) std::cout << "Comp slack violated at constraint " << i
                                   << ": y*slack = " << comp << std::endl;
            all_ok = false;
        }
    }
    for (int64_t j = 0; j < n; j++) {
        double comp = s[j] * x[j];
        if (std::abs(comp) > tol) {
            if (verbose) std::cout << "Comp slack violated at variable " << j
                                   << ": s*x = " << comp << std::endl;
            all_ok = false;
        }
    }

    return all_ok;
}

/**
 * Helper structure to generate a convex QP with a known KKT solution.
 * Mirrors the Python generator:
 *   minimize    0.5 x^T P x + q^T x
 *   subject to  A x <= b,  x >= 0
 */
struct GeneratedQP {
    int64_t m;
    int64_t n;

    std::vector<std::vector<double>> P_dense;
    std::vector<std::vector<double>> A_dense;
    std::vector<double> q;
    std::vector<double> b;

    std::vector<double> x_star;
    std::vector<double> y_star;
    std::vector<double> s_star;

    std::vector<int64_t> I_active;
    std::vector<int64_t> J_pos;
    std::vector<int64_t> J_zero;

    GeneratedQP(
        int64_t m_in,
        int64_t n_in,
        double density = 0.5,
        double active_frac = 0.5,
        double positive_frac = 0.5,
        double strong_convexity = 1e-2,
        unsigned seed = 42
    ) : m(m_in), n(n_in) {
        if (m <= 0 || n <= 0) {
            throw std::runtime_error("Need m > 0 and n > 0");
        }

        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uniform01(0.0, 1.0);
        std::uniform_real_distribution<double> pos_dist(0.5, 2.0);
        std::normal_distribution<double> normal_dist(0.0, 1.0);

        // 1. Choose x_star support
        int64_t k_pos = std::max<int64_t>(1, static_cast<int64_t>(positive_frac * n));
        if (k_pos > n) k_pos = n;

        std::vector<int64_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        J_pos.assign(indices.begin(), indices.begin() + k_pos);
        std::sort(J_pos.begin(), J_pos.end());
        for (int64_t j = 0; j < n; j++) {
            if (std::find(J_pos.begin(), J_pos.end(), j) == J_pos.end()) {
                J_zero.push_back(j);
            }
        }

        x_star.assign(n, 0.0);
        for (int64_t j : J_pos) {
            x_star[j] = pos_dist(rng);
        }

        // 2. Build A with desired density
        A_dense.assign(m, std::vector<double>(n, 0.0));
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                if (density >= 1.0 || uniform01(rng) < density) {
                    A_dense[i][j] = normal_dist(rng);
                }
            }
        }

        // 3. Decide active constraints
        int64_t k_active = std::max<int64_t>(1, static_cast<int64_t>(active_frac * m));
        if (k_active > m) k_active = m;
        std::vector<int64_t> constraint_indices(m);
        std::iota(constraint_indices.begin(), constraint_indices.end(), 0);
        std::shuffle(constraint_indices.begin(), constraint_indices.end(), rng);
        I_active.assign(constraint_indices.begin(), constraint_indices.begin() + k_active);
        std::sort(I_active.begin(), I_active.end());

        // 4. Build b = A x_star + r with r >= 0 and r_i = 0 for active constraints
        std::vector<double> r(m);
        for (int64_t i = 0; i < m; i++) {
            r[i] = pos_dist(rng);
        }
        for (int64_t i : I_active) {
            r[i] = 0.0;
        }

        b.assign(m, 0.0);
        for (int64_t i = 0; i < m; i++) {
            b[i] = r[i];
            for (int64_t j = 0; j < n; j++) {
                b[i] += A_dense[i][j] * x_star[j];
            }
        }

        // 5. Dual variables y_star (for Ax <= b)
        y_star.assign(m, 0.0);
        for (int64_t i : I_active) {
            y_star[i] = pos_dist(rng);
        }

        // 6. Dual variables s_star (for x >= 0)
        s_star.assign(n, 0.0);
        for (int64_t j : J_zero) {
            s_star[j] = pos_dist(rng);
        }

        // 7. Build PSD matrix P = M^T M + μI
        std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j < n; j++) {
                if (density >= 1.0 || uniform01(rng) < density) {
                    M[i][j] = normal_dist(rng);
                }
            }
        }

        P_dense.assign(n, std::vector<double>(n, 0.0));
        for (int64_t k = 0; k < n; k++) {
            for (int64_t i = 0; i < n; i++) {
                for (int64_t j = i; j < n; j++) {
                    P_dense[i][j] += M[k][i] * M[k][j];
                }
            }
        }
        for (int64_t i = 0; i < n; i++) {
            P_dense[i][i] += strong_convexity;
        }
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j < i; j++) {
                P_dense[i][j] = P_dense[j][i];
            }
        }

        // 8. q from stationarity: P x_star + q + A^T y_star - s_star = 0 (note sign)
        q.assign(n, 0.0);
        for (int64_t i = 0; i < n; i++) {
            double grad = 0.0;
            for (int64_t j = 0; j < n; j++) {
                grad += P_dense[i][j] * x_star[j];
            }
            for (int64_t k = 0; k < m; k++) {
                grad += A_dense[k][i] * y_star[k];
            }
            grad -= s_star[i];
            q[i] = -grad;
        }
    }

    /**
     * Convert A and bounds into Moreau format (Ax + s1 = b, -x + s2 = 0).
     */
    std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<double>, std::vector<double>, int64_t>
    toMoreauFormat(double drop_tol = 1e-14) const {
        int64_t total_m = m + n;

        std::vector<int64_t> A_ro;
        std::vector<int64_t> A_ci;
        std::vector<double> A_val;
        std::vector<double> b_combined;

        A_ro.push_back(0);
        int64_t nnz = 0;

        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                if (std::abs(A_dense[i][j]) > drop_tol) {
                    A_ci.push_back(j);
                    A_val.push_back(A_dense[i][j]);
                    nnz++;
                }
            }
            A_ro.push_back(nnz);
            b_combined.push_back(b[i]);
        }

        for (int64_t j = 0; j < n; j++) {
            A_ci.push_back(j);
            A_val.push_back(-1.0);
            nnz++;
            A_ro.push_back(nnz);
            b_combined.push_back(0.0);
        }

        return {A_ro, A_ci, A_val, b_combined, total_m};
    }

    /**
     * Convert symmetric dense P into CSR storing the full symmetric matrix.
     */
    std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<double>>
    PToCSRFull(double drop_tol = 1e-14) const {
        std::vector<int64_t> P_ro;
        std::vector<int64_t> P_ci;
        std::vector<double> P_val;

        P_ro.push_back(0);
        int64_t nnz = 0;
        for (int64_t i = 0; i < n; i++) {
            for (int64_t j = 0; j < n; j++) {
                if (std::abs(P_dense[i][j]) > drop_tol) {
                    P_ci.push_back(j);
                    P_val.push_back(P_dense[i][j]);
                    nnz++;
                }
            }
            P_ro.push_back(nnz);
        }
        return {P_ro, P_ci, P_val};
    }

    /**
     * Compute objective 0.5 x^T P x + q^T x.
     */
    double objective(const std::vector<double>& x) const {
        if (static_cast<int64_t>(x.size()) != n) {
            throw std::runtime_error("objective: dimension mismatch");
        }
        double obj = 0.0;
        for (int64_t i = 0; i < n; i++) {
            double Px_i = 0.0;
            for (int64_t j = 0; j < n; j++) {
                Px_i += P_dense[i][j] * x[j];
            }
            obj += 0.5 * x[i] * Px_i + q[i] * x[i];
        }
        return obj;
    }
};

class GeneratedQPTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetLastError();
    }
    void TearDown() override {
        cudaDeviceSynchronize();
    }
};

// ============================================================================
// Simple QP Tests
// ============================================================================

TEST_F(SimpleQPTest, QPWithEqualityAndInequality) {
    // This QP has an equality constraint and inequality constraints:
    //
    //     minimize    (1/2) xᵗ P x + qᵗ x
    //     subject to  x₁ + x₂ = 1  (equality)
    //                 x₁ + s₁ = 2, s₁ >= 0  (i.e., x₁ <= 2)
    //                 x₂ + s₂ = 2, s₂ >= 0  (i.e., x₂ <= 2)
    //
    // This is feasible: x = [0, 1] satisfies all constraints.

    int n = 2;  // 2 variables
    int m = 3;  // 3 constraints
    int batchSize = 1;

    // P is 2x2 identity
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<double> P_values = {1.0, 1.0};

    // q = [2.0, 1.0]
    std::vector<double> q_data = {2.0, 1.0};

    // A matrix:
    // Row 0: x₁ + x₂ = 1   (equality, zero cone)
    // Row 1: x₁ + s₁ = 2   (inequality, s₁ >= 0)
    // Row 2: x₂ + s₂ = 2   (inequality, s₂ >= 0)
    std::vector<int64_t> A_ro = {0, 2, 3, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    int64_t nnzA = 4;
    std::vector<double> A_values = {1.0, 1.0, 1.0, 1.0};

    // b vector
    std::vector<double> b_data = {1.0, 2.0, 2.0};

    // Cone structure: 1 zero (equality), 2 nonneg (inequalities)
    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 30;  // Run enough iterations to converge
    settings.verbose = true;

    // Create solver
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory for problem data
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Copy solution to host
    std::vector<double> x_sol(n * batchSize);
    std::vector<double> s_sol(m * batchSize);
    std::vector<double> z_sol(m * batchSize);
    std::vector<int32_t> status_sol(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Print solution
    std::cout << "\n=== MOREAU SOLUTION ===" << std::endl;
    std::cout << "x = [";
    for (int i = 0; i < n; i++) {
        std::cout << x_sol[i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "s = [";
    for (int i = 0; i < m; i++) {
        std::cout << s_sol[i];
        if (i < m - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "z = [";
    for (int i = 0; i < m; i++) {
        std::cout << z_sol[i];
        if (i < m - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Get tau and kappa from variables
    std::vector<double> tau_host(batchSize);
    std::vector<double> kappa_host(batchSize);
    solver.variables.τ.gpuToCpu(tau_host.data());
    solver.variables.κ.gpuToCpu(kappa_host.data());
    std::cout << "τ = " << tau_host[0] << std::endl;
    std::cout << "κ = " << kappa_host[0] << std::endl;

    // Get objective value
    std::vector<double> cost_primal(batchSize);
    solver.info.cost_primal.gpuToCpu(cost_primal.data());
    std::cout << "Optimal objective: " << cost_primal[0] << std::endl;
    std::cout << "Status: " << status_sol[0] << " (1=Solved, 2=AlmostSolved)" << std::endl;
    std::cout << "===========================\n" << std::endl;

    // Check that solver converged
    EXPECT_TRUE(status_sol[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status_sol[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Expected Solved (1) or AlmostSolved (2), got " << status_sol[0];

    // Expected solution from Clarabel (Rust):
    // x = [2.4308818538642494e-10, 0.9999999997569121]
    // s = [0.0, 1.9999999997569118, 1.0000000002430876]
    // z = [-2.0000000002479426, 3.2555674870730805e-11, 5.187299172411473e-10]
    // τ = 1.0
    // κ = 7.496023490295008e-11

    std::vector<double> expected_x = {2.4308818538642494e-10, 0.9999999997569121};
    std::vector<double> expected_s = {0.0, 1.9999999997569118, 1.0000000002430876};
    std::vector<double> expected_z = {-2.0000000002479426, 3.2555674870730805e-11, 5.187299172411473e-10};

    // Check x solution (primal variables)
    for (int i = 0; i < n; i++) {
        EXPECT_NEAR(x_sol[i], expected_x[i], 1e-6) << "x[" << i << "] mismatch";
    }

    // Check s solution (slack variables)
    for (int i = 0; i < m; i++) {
        double tol = (std::abs(expected_s[i]) < 1e-6) ? 1e-6 : 1e-6;
        EXPECT_NEAR(s_sol[i], expected_s[i], tol) << "s[" << i << "] mismatch";
    }

    // Check z solution (dual variables)
    for (int i = 0; i < m; i++) {
        double tol = (std::abs(expected_z[i]) < 1e-6) ? 1e-6 : 1e-6;
        EXPECT_NEAR(z_sol[i], expected_z[i], tol) << "z[" << i << "] mismatch";
    }

    // Check tau and kappa
    // Note: tau may not be exactly 1.0 if we stop early; the important thing is that
    // the unscaled solution (x/tau, s/tau, z/tau) is correct
    EXPECT_GT(tau_host[0], 0.1) << "tau should be positive (feasible problem)";
    EXPECT_LT(kappa_host[0], 1e-6) << "kappa should be near zero (feasible problem)";

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

// ============================================================================
// Dense QP Tests
// ============================================================================

TEST_F(DenseQPTest, DensePMatrixGeneralQP) {
    // This reproduces the Python general_qp test that fails
    //
    // Problem from Python with np.random.seed(42):
    //     minimize    (1/2) x'Px + q'x
    //     subject to  Ax <= b  (converted to Ax + s = b, s >= 0)
    //
    // P is a dense 5x5 positive definite matrix (not diagonal!)
    // This is the key difference from other tests

    int n = 5;  // 5 variables
    int m = 4;  // 4 inequality constraints
    int batchSize = 1;

    // Dense P matrix from Python: P = L.T @ L + I with seed(42)
    // where L = np.random.randn(5, 5) (full matrix from numpy)
    // P is stored in CSR format (upper triangle for symmetric matrix)
    // Full dense 5x5 symmetric matrix (P must be stored as full matrix):
    // [[ 3.9806,  0.0160, -0.0478,  0.1755,  0.5523],
    //  [ 0.0160,  4.8067,  0.6762,  1.1804,  3.2459],
    //  [-0.0478,  0.6762,  2.1703, -0.2183, -0.6332],
    //  [ 0.1755,  1.1804, -0.2183, 10.0551,  4.7469],
    //  [ 0.5523,  3.2459, -0.6332,  4.7469,  6.6155]]

    // CSR format for FULL 5x5 symmetric matrix (25 entries)
    std::vector<int64_t> P_ro = {0, 5, 10, 15, 20, 25};
    std::vector<int64_t> P_ci = {
        0, 1, 2, 3, 4,       // row 0
        0, 1, 2, 3, 4,       // row 1
        0, 1, 2, 3, 4,       // row 2
        0, 1, 2, 3, 4,       // row 3
        0, 1, 2, 3, 4        // row 4
    };
    std::vector<double> P_values = {
        // Row 0
        3.9805946,  0.01599104, -0.04782309,  0.17546989,  0.55226368,
        // Row 1 (symmetric: P[1,0]=P[0,1], etc.)
        0.01599104, 4.80673615,  0.67617578,  1.18044033,  3.24587224,
        // Row 2
       -0.04782309, 0.67617578,  2.17031369, -0.21833843, -0.63321793,
        // Row 3
        0.17546989, 1.18044033, -0.21833843, 10.05508255,  4.74692465,
        // Row 4
        0.55226368, 3.24587224, -0.63321793,  4.74692465,  6.61549503
    };
    int64_t nnzP = 25;

    // q vector from Python with seed(42) - generated AFTER L
    std::vector<double> q_data = {
        0.11092258970986608, -1.1509935774223028, 0.37569801834567196,
        -0.600638689918805, -0.2916937497932768
    };

    // A matrix (4x5 dense) from Python with seed(42) - generated AFTER q
    std::vector<double> A_dense = {
        -0.6017066122293969,  1.8522781845089378, -0.013497224737933921,
        -1.0577109289559004,  0.822544912103189,
        -1.2208436499710222,  0.2088635950047554, -1.9596701238797756,
        -1.3281860488984305,  0.19686123586912352,
         0.7384665799954104,  0.1713682811899705, -0.11564828238824053,
        -0.3011036955892888, -1.4785219903674274,
        -0.7198442083947086, -0.4606387709597875,  1.0571222262189157,
         0.3436182895684614, -1.763040155362734
    };

    // Convert A to CSR format (all 20 entries)
    std::vector<int64_t> A_ro = {0, 5, 10, 15, 20};
    std::vector<int64_t> A_ci = {
        0, 1, 2, 3, 4,       // row 0
        0, 1, 2, 3, 4,       // row 1
        0, 1, 2, 3, 4,       // row 2
        0, 1, 2, 3, 4        // row 3
    };
    std::vector<double> A_values = A_dense;
    int64_t nnzA = 20;

    // b vector from Python: random + 5.0 to ensure feasibility
    std::vector<double> b_data = {
        5.3240839693947954, 4.6149177195836835, 4.323077999694041, 5.611676288840868
    };

    // Cone structure: all 4 constraints are inequalities (nonneg)
    Cones cones{};
    cones.numNonnegCones = 4;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 100;
    settings.verbose = false;

    std::cout << "\n===============================================" << std::endl;
    std::cout << "DENSE QP TEST - Reproducing Python Failure" << std::endl;
    std::cout << "===============================================" << std::endl;
    std::cout << "Problem: n=" << n << ", m=" << m << ", nnz(P)=" << nnzP << ", nnz(A)=" << nnzA << std::endl;
    std::cout << "P is DENSE (25 entries), not diagonal!" << std::endl;
    std::cout << "\nExpected solution (from MOSEK):" << std::endl;
    std::cout << "  x = [0.00748897, 0.49920661, -0.4162938, 0.16012237, -0.35620865]" << std::endl;
    std::cout << "  obj = -0.361212759" << std::endl;
    std::cout << "===============================================\n" << std::endl;

    // Create solver
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory for problem data
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Copy solution to host
    std::vector<double> x_sol(n * batchSize);
    std::vector<double> s_sol(m * batchSize);
    std::vector<double> z_sol(m * batchSize);
    std::vector<int32_t> status_sol(batchSize);
    std::vector<double> obj_val(batchSize);

    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(obj_val.data(), solver.solution.obj_val.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Print solution
    std::cout << "\n=== MOREAU SOLUTION ===" << std::endl;
    std::cout << "Status: " << status_sol[0] << std::endl;
    std::cout << "Objective: " << std::scientific << std::setprecision(10) << obj_val[0] << std::endl;
    std::cout << "x = [";
    for (int i = 0; i < n; i++) {
        std::cout << std::fixed << std::setprecision(8) << x_sol[i];
        if (i < n - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "s = [";
    for (int i = 0; i < m; i++) {
        std::cout << std::scientific << std::setprecision(4) << s_sol[i];
        if (i < m - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Expected solution from MOSEK
    std::vector<double> x_expected = {0.00748897, 0.49920661, -0.4162938, 0.16012237, -0.35620865};
    double obj_expected = -0.361212759;

    // Compute error
    double x_error = 0.0;
    double x_norm = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = x_sol[i] - x_expected[i];
        x_error += diff * diff;
        x_norm += x_expected[i] * x_expected[i];
    }
    x_error = std::sqrt(x_error) / std::sqrt(x_norm);
    double obj_error = std::abs(obj_val[0] - obj_expected) / std::abs(obj_expected);

    std::cout << "\n=== COMPARISON WITH MOSEK ===" << std::endl;
    std::cout << "Solution error (relative): " << std::scientific << x_error << std::endl;
    std::cout << "Objective error (relative): " << std::scientific << obj_error << std::endl;

    if (x_error < 1e-4 && obj_error < 1e-4) {
        std::cout << "✓ GOOD AGREEMENT (errors < 1e-4)" << std::endl;
    } else {
        std::cout << "✗ POOR AGREEMENT (errors >= 1e-4)" << std::endl;
    }

    // Free device memory
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);

    // Test expectations
    EXPECT_EQ(status_sol[0], 1);  // SolverStatus::Solved
    EXPECT_LT(x_error, 1e-4);     // Solution should be accurate
    EXPECT_LT(obj_error, 1e-4);   // Objective should be accurate
}

// ============================================================================
// Analytical Solution Tests
// ============================================================================

TEST_F(AnalyticalSolutionTest, EqualityConstrainedQP) {
    // Equality-constrained quadratic program (minimum norm solution):
    //     minimize    (1/2) ||x||²
    //     subject to  x1 + x2 + x3 = 3
    //
    // Analytical solution via Lagrange multipliers:
    // ∇L = x + λ[1,1,1]ᵗ = 0  =>  x = -λ[1,1,1]ᵗ
    // Constraint: 3λ = -3  =>  λ = -1
    // Therefore: x* = [1, 1, 1]
    // Optimal objective: (1/2)(1 + 1 + 1) = 1.5

    std::cout << "\n=== EqualityConstrainedQP Test ===" << std::endl;

    int n = 3;  // 3 variables (ODD - exposes stride bug!)
    int m = 1;  // 1 equality constraint
    int batchSize = 2;  // Multiple batches to expose bug

    // P = I (identity)
    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;
    std::vector<double> P_values(nnzP * batchSize);
    for (int b = 0; b < batchSize; b++) {
        P_values[b * nnzP + 0] = 1.0;
        P_values[b * nnzP + 1] = 1.0;
        P_values[b * nnzP + 2] = 1.0;
    }

    // q = 0 for all batches
    std::vector<double> q_data(n * batchSize, 0.0);

    // A = [1, 1, 1] (one row for sum constraint)
    std::vector<int64_t> A_ro = {0, 3};
    std::vector<int64_t> A_ci = {0, 1, 2};
    int64_t nnzA = 3;
    std::vector<double> A_values(nnzA * batchSize);
    for (int b = 0; b < batchSize; b++) {
        A_values[b * nnzA + 0] = 1.0;
        A_values[b * nnzA + 1] = 1.0;
        A_values[b * nnzA + 2] = 1.0;
    }

    // b = [3] for all batches
    std::vector<double> b_data(m * batchSize);
    for (int b = 0; b < batchSize; b++) {
        b_data[b * m + 0] = 3.0;
    }

    // One zero cone (equality constraint)
    Cones cones{};
    cones.numZeroCones = 1;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 50;
    settings.verbose = false;

    // Create solver
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Copy solution to host
    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status_sol(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Get objective value
    std::vector<double> cost_primal(batchSize);
    solver.info.cost_primal.gpuToCpu(cost_primal.data());

    // Print solutions for both batches
    for (int b = 0; b < batchSize; b++) {
        std::cout << "Batch " << b << " solution: x = ["
                  << x_sol[b * n + 0] << ", "
                  << x_sol[b * n + 1] << ", "
                  << x_sol[b * n + 2] << "]" << std::endl;
        std::cout << "Batch " << b << " objective: " << cost_primal[b] << std::endl;
        std::cout << "Batch " << b << " status: " << status_sol[b] << std::endl;
    }

    // Check that solver converged for both batches
    for (int b = 0; b < batchSize; b++) {
        EXPECT_TRUE(status_sol[b] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status_sol[b] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "Batch " << b << ": Expected Solved (1) or AlmostSolved (4), got " << status_sol[b];
    }

    // Check analytical solution for both batches
    std::vector<double> expected_x = {1.0, 1.0, 1.0};
    for (int b = 0; b < batchSize; b++) {
        std::vector<double> batch_x(n);
        for (int i = 0; i < n; i++) {
            batch_x[i] = x_sol[b * n + i];
        }
        checkSolution(batch_x, expected_x, 1e-7, "Batch " + std::to_string(b) + " x");

        // Check objective value
        EXPECT_NEAR(cost_primal[b], 1.5, 1e-8) << "Batch " << b << " objective value mismatch";
    }

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(AnalyticalSolutionTest, TwoEqualityConstraints) {
    // Quadratic program with two equality constraints:
    //     minimize    (1/2) x'Px + q'x
    //     subject to  x1 + x2 = 2
    //                 x1 - x2 = 0
    //
    // Where P = diag([2, 2]) and q = [0, 0]
    //
    // From constraints: x1 = x2 and x1 + x2 = 2
    // => 2*x1 = 2 => x1 = 1, x2 = 1
    // Analytical solution: x* = [1, 1]
    // Optimal objective: (1/2)(2*1² + 2*1²) = 2

    std::cout << "\n=== TwoEqualityConstraints Test ===" << std::endl;

    int n = 2;  // 2 variables
    int m = 2;  // 2 equality constraints
    int batchSize = 1;

    // P = diag([2, 2])
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<double> P_values = {2.0, 2.0};

    // q = [0, 0]
    std::vector<double> q_data = {0.0, 0.0};

    // A = [[1, 1], [1, -1]]
    std::vector<int64_t> A_ro = {0, 2, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    int64_t nnzA = 4;
    std::vector<double> A_values = {1.0, 1.0, 1.0, -1.0};

    // b = [2, 0]
    std::vector<double> b_data = {2.0, 0.0};

    // Two zero cones (equality constraints)
    Cones cones{};
    cones.numZeroCones = 2;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 50;
    settings.verbose = false;

    // Create solver
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Copy solution to host
    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status_sol(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Get objective value
    std::vector<double> cost_primal(batchSize);
    solver.info.cost_primal.gpuToCpu(cost_primal.data());

    std::cout << "Solution: x = [" << x_sol[0] << ", " << x_sol[1] << "]" << std::endl;
    std::cout << "Objective: " << cost_primal[0] << std::endl;
    std::cout << "Status: " << status_sol[0] << std::endl;

    // Check that solver converged
    EXPECT_TRUE(status_sol[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status_sol[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Expected Solved (1) or AlmostSolved (4), got " << status_sol[0];

    // Check analytical solution
    std::vector<double> expected_x = {1.0, 1.0};
    checkSolution(x_sol, expected_x, 1e-7, "x");

    // Check objective value
    EXPECT_NEAR(cost_primal[0], 2.0, 1e-8) << "Objective value mismatch";

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(AnalyticalSolutionTest, SimpleNonnegativeConstraint) {
    // QP with nonnegative constraint:
    //     minimize    (1/2)(x² + y²) + 2x + 3y
    //     subject to  x >= 0, y >= 0
    //
    // Unconstrained optimum: x = -2, y = -3 (both negative)
    // With x, y >= 0: Active constraints at x = 0, y = 0
    // Analytical solution: x* = [0, 0]
    // Optimal objective: 0

    std::cout << "\n=== SimpleNonnegativeConstraint Test ===" << std::endl;

    int n = 2;  // 2 variables
    int m = 2;  // 2 nonnegative constraints
    int batchSize = 1;

    // P = I (identity)
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<double> P_values = {1.0, 1.0};

    // q = [2, 3]
    std::vector<double> q_data = {2.0, 3.0};

    // A matrix for x >= 0, y >= 0:
    // -x + s1 = 0  =>  s1 = x >= 0  (s1 >= 0 means x >= 0)
    // -y + s2 = 0  =>  s2 = y >= 0  (s2 >= 0 means y >= 0)
    std::vector<int64_t> A_ro = {0, 1, 2};
    std::vector<int64_t> A_ci = {0, 1};
    int64_t nnzA = 2;
    std::vector<double> A_values = {-1.0, -1.0};

    std::vector<double> b_data = {0.0, 0.0};

    // Two nonnegative cones
    Cones cones{};
    cones.numNonnegCones = 2;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 50;
    settings.verbose = false;

    // Create solver
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Copy solution to host
    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status_sol(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Get objective value
    std::vector<double> cost_primal(batchSize);
    solver.info.cost_primal.gpuToCpu(cost_primal.data());

    std::cout << "Solution: x = [" << x_sol[0] << ", " << x_sol[1] << "]" << std::endl;
    std::cout << "Objective: " << cost_primal[0] << std::endl;
    std::cout << "Status: " << status_sol[0] << std::endl;

    // Check that solver converged
    EXPECT_TRUE(status_sol[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status_sol[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Expected Solved (1) or AlmostSolved (4), got " << status_sol[0];

    // Check analytical solution
    std::vector<double> expected_x = {0.0, 0.0};
    checkSolution(x_sol, expected_x, 1e-6, "x");

    // Check objective value
    EXPECT_NEAR(cost_primal[0], 0.0, 1e-6) << "Objective value mismatch";

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(AnalyticalSolutionTest, MixedEqualityAndInequality) {
    // QP with mixed equality and inequality constraints:
    //     minimize    (1/2)(x² + y²)
    //     subject to  x + y = 2  (equality)
    //                 x >= 0     (inequality)
    //
    // From equality: y = 2 - x
    // Objective becomes: (1/2)(x² + (2-x)²) = (1/2)(2x² - 4x + 4)
    // Derivative: 2x - 2 = 0 => x = 1
    // Check constraint x >= 0: satisfied
    // Analytical solution: x* = [1, 1]
    // Optimal objective: (1/2)(1 + 1) = 1

    std::cout << "\n=== MixedEqualityAndInequality Test ===" << std::endl;

    int n = 2;  // 2 variables
    int m = 2;  // 1 equality + 1 inequality
    int batchSize = 1;

    // P = I (identity)
    std::vector<int64_t> P_ro = {0, 1, 2};
    std::vector<int64_t> P_ci = {0, 1};
    int64_t nnzP = 2;
    std::vector<double> P_values = {1.0, 1.0};

    // q = [0, 0]
    std::vector<double> q_data = {0.0, 0.0};

    // A matrix:
    // Row 0: x + y = 2     (equality, zero cone)
    // Row 1: -x + s = 0    (s = x >= 0, nonneg cone)
    std::vector<int64_t> A_ro = {0, 2, 3};
    std::vector<int64_t> A_ci = {0, 1, 0};
    int64_t nnzA = 3;
    std::vector<double> A_values = {1.0, 1.0, -1.0};

    std::vector<double> b_data = {2.0, 0.0};

    // 1 zero cone + 1 nonneg cone
    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = 1;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 50;
    settings.verbose = false;

    // Create solver
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Copy solution to host
    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status_sol(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Get objective value
    std::vector<double> cost_primal(batchSize);
    solver.info.cost_primal.gpuToCpu(cost_primal.data());

    std::cout << "Solution: x = [" << x_sol[0] << ", " << x_sol[1] << "]" << std::endl;
    std::cout << "Objective: " << cost_primal[0] << std::endl;
    std::cout << "Status: " << status_sol[0] << std::endl;

    // Check that solver converged
    EXPECT_TRUE(status_sol[0] == static_cast<int32_t>(SolverStatus::Solved) ||
                status_sol[0] == static_cast<int32_t>(SolverStatus::AlmostSolved))
        << "Expected Solved (1) or AlmostSolved (4), got " << status_sol[0];

    // Check analytical solution
    std::vector<double> expected_x = {1.0, 1.0};
    checkSolution(x_sol, expected_x, 1e-6, "x");

    // Check objective value
    EXPECT_NEAR(cost_primal[0], 1.0, 1e-7) << "Objective value mismatch";

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(AnalyticalSolutionTest, DiagonalQPWithLinearTerm) {
    // Diagonal QP with linear term:
    //     minimize    (1/2)(2x² + 3y² + 4z²) + x - 2y + 3z
    //     subject to  x + y + z = 1  (equality)
    //
    // This has a closed-form solution via Lagrange multipliers.
    // Analytical solution can be computed from:
    // [2 0 0] [x]   [1]   [1]
    // [0 3 0] [y] + [-2] + λ[1] = 0
    // [0 0 4] [z]   [3]   [1]
    //
    // => 2x + 1 + λ = 0
    // => 3y - 2 + λ = 0
    // => 4z + 3 + λ = 0
    // With x + y + z = 1
    //
    // From equations: x = -(1+λ)/2, y = (2-λ)/3, z = -(3+λ)/4
    // Substituting into constraint:
    // -(1+λ)/2 + (2-λ)/3 - (3+λ)/4 = 1
    // Multiply by 12: -6(1+λ) + 4(2-λ) - 3(3+λ) = 12
    // -6 - 6λ + 8 - 4λ - 9 - 3λ = 12
    // -7 - 13λ = 12
    // λ = -19/13
    //
    // x = -(1 - 19/13)/2 = -(-6/13)/2 = 3/13
    // y = (2 + 19/13)/3 = (45/13)/3 = 15/13
    // z = -(3 - 19/13)/4 = -(20/13)/4 = -5/13

    std::cout << "\n=== DiagonalQPWithLinearTerm Test ===" << std::endl;

    int n = 3;  // 3 variables (ODD - exposes stride bug!)
    int m = 1;  // 1 equality constraint
    int batchSize = 2;  // Multiple batches to expose bug

    // P = diag([2, 3, 4])
    std::vector<int64_t> P_ro = {0, 1, 2, 3};
    std::vector<int64_t> P_ci = {0, 1, 2};
    int64_t nnzP = 3;
    std::vector<double> P_values(nnzP * batchSize);
    for (int b = 0; b < batchSize; b++) {
        P_values[b * nnzP + 0] = 2.0;
        P_values[b * nnzP + 1] = 3.0;
        P_values[b * nnzP + 2] = 4.0;
    }

    // q = [1, -2, 3] for all batches
    std::vector<double> q_data(n * batchSize);
    for (int b = 0; b < batchSize; b++) {
        q_data[b * n + 0] = 1.0;
        q_data[b * n + 1] = -2.0;
        q_data[b * n + 2] = 3.0;
    }

    // A = [1, 1, 1]
    std::vector<int64_t> A_ro = {0, 3};
    std::vector<int64_t> A_ci = {0, 1, 2};
    int64_t nnzA = 3;
    std::vector<double> A_values(nnzA * batchSize);
    for (int b = 0; b < batchSize; b++) {
        A_values[b * nnzA + 0] = 1.0;
        A_values[b * nnzA + 1] = 1.0;
        A_values[b * nnzA + 2] = 1.0;
    }

    // b = [1] for all batches
    std::vector<double> b_data(m * batchSize);
    for (int b = 0; b < batchSize; b++) {
        b_data[b * m + 0] = 1.0;
    }

    // One zero cone
    Cones cones{};
    cones.numZeroCones = 1;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 50;
    settings.verbose = false;

    // Create solver
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    // Copy to device
    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Copy solution to host
    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status_sol(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Get objective value
    std::vector<double> cost_primal(batchSize);
    solver.info.cost_primal.gpuToCpu(cost_primal.data());

    // Calculate expected values
    double x_expected = 3.0 / 13.0;    // ≈ 0.230769
    double y_expected = 15.0 / 13.0;   // ≈ 1.153846
    double z_expected = -5.0 / 13.0;   // ≈ -0.384615
    double obj_expected = 0.5 * (2*x_expected*x_expected + 3*y_expected*y_expected + 4*z_expected*z_expected)
                        + x_expected - 2*y_expected + 3*z_expected;

    // Print solutions for both batches
    for (int b = 0; b < batchSize; b++) {
        std::cout << "Batch " << b << " solution: x = ["
                  << x_sol[b * n + 0] << ", "
                  << x_sol[b * n + 1] << ", "
                  << x_sol[b * n + 2] << "]" << std::endl;
        std::cout << "Batch " << b << " objective: " << cost_primal[b] << " (expected: " << obj_expected << ")" << std::endl;
        std::cout << "Batch " << b << " status: " << status_sol[b] << std::endl;
    }

    // Check that solver converged for both batches
    for (int b = 0; b < batchSize; b++) {
        EXPECT_TRUE(status_sol[b] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status_sol[b] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "Batch " << b << ": Expected Solved (1) or AlmostSolved (4), got " << status_sol[b];
    }

    // Check analytical solution for both batches
    std::vector<double> expected_x = {x_expected, y_expected, z_expected};
    for (int b = 0; b < batchSize; b++) {
        std::vector<double> batch_x(n);
        for (int i = 0; i < n; i++) {
            batch_x[i] = x_sol[b * n + i];
        }
        checkSolution(batch_x, expected_x, 1e-6, "Batch " + std::to_string(b) + " x");

        // Check objective value
        EXPECT_NEAR(cost_primal[b], obj_expected, 1e-6) << "Batch " << b << " objective value mismatch";
    }

    // Clean up
    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(AnalyticalSolutionTest, DirectStrideBugExposure) {
    // This test directly exposes the stride bug by:
    // 1. Using waxpby (writes with batch*n stride - WRONG)
    // 2. Using quad_form (reads with batch*n_padded stride - CORRECT)
    // When they disagree, batch 1 gets wrong data

    std::cout << "\n=== DirectStrideBugExposure Test ===" << std::endl;

    int n = 5;  // ODD number to trigger padding (n_padded = 6)
    int m = 1;
    int batchSize = 2;

    std::cout << "n = " << n << " (odd)" << std::endl;
    std::cout << "n_padded should be " << ((n + 1) & ~1) << " (even)" << std::endl;
    std::cout << "batchSize = " << batchSize << std::endl;
    std::cout << "\nMemory layout mismatch:" << std::endl;
    std::cout << "  Batch 0 offset with n:       0 * " << n << " = 0" << std::endl;
    std::cout << "  Batch 0 offset with n_padded: 0 * " << ((n+1)&~1) << " = 0" << std::endl;
    std::cout << "  Batch 1 offset with n:       1 * " << n << " = " << n << std::endl;
    std::cout << "  Batch 1 offset with n_padded: 1 * " << ((n+1)&~1) << " = " << ((n+1)&~1) << std::endl;
    std::cout << "  MISMATCH: " << (((n+1)&~1) - n) << " doubles difference!\n" << std::endl;

    // Problem: minimize (1/2)||x||² subject to sum(x) = 5
    // Analytical solution: x* = [1, 1, 1, 1, 1]

    std::vector<int64_t> P_ro = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> P_ci = {0, 1, 2, 3, 4};
    int64_t nnzP = 5;
    std::vector<double> P_values(nnzP * batchSize);
    for (int b = 0; b < batchSize; b++) {
        for (int i = 0; i < nnzP; i++) {
            P_values[b * nnzP + i] = 1.0;
        }
    }

    std::vector<double> q_data(n * batchSize, 0.0);

    std::vector<int64_t> A_ro = {0, 5};
    std::vector<int64_t> A_ci = {0, 1, 2, 3, 4};
    int64_t nnzA = 5;
    std::vector<double> A_values(nnzA * batchSize);
    for (int b = 0; b < batchSize; b++) {
        for (int i = 0; i < nnzA; i++) {
            A_values[b * nnzA + i] = 1.0;
        }
    }

    std::vector<double> b_data(m * batchSize);
    for (int b = 0; b < batchSize; b++) {
        b_data[b * m] = 5.0;
    }

    Cones cones{};
    cones.numZeroCones = 1;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 50;
    settings.verbose = false;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P_values, P_values.data(), sizeof(double) * nnzP * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_values.data(), sizeof(double) * nnzA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m * batchSize, cudaMemcpyHostToDevice);

    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    std::vector<double> x_sol(n * batchSize);
    std::vector<int32_t> status_sol(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::vector<double> cost_primal(batchSize);
    solver.info.cost_primal.gpuToCpu(cost_primal.data());

    // Print solutions
    std::cout << "\nResults:" << std::endl;
    for (int b = 0; b < batchSize; b++) {
        std::cout << "Batch " << b << " solution: x = [";
        for (int i = 0; i < n; i++) {
            std::cout << x_sol[b * n + i];
            if (i < n-1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Batch " << b << " objective: " << cost_primal[b] << " (expected: 2.5)" << std::endl;
        std::cout << "Batch " << b << " status: " << status_sol[b] << std::endl;
    }

    // Check convergence
    for (int b = 0; b < batchSize; b++) {
        EXPECT_TRUE(status_sol[b] == static_cast<int32_t>(SolverStatus::Solved) ||
                    status_sol[b] == static_cast<int32_t>(SolverStatus::AlmostSolved))
            << "Batch " << b << " did not converge";
    }

    // Expected solution for both batches: x = [1, 1, 1, 1, 1]
    std::vector<double> expected_x = {1.0, 1.0, 1.0, 1.0, 1.0};
    for (int b = 0; b < batchSize; b++) {
        std::vector<double> batch_x(n);
        for (int i = 0; i < n; i++) {
            batch_x[i] = x_sol[b * n + i];
        }

        std::cout << "\nBatch " << b << " checking correctness..." << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << "  x[" << i << "] = " << batch_x[i] << " (expected: " << expected_x[i] << ")" << std::endl;
        }

        checkSolution(batch_x, expected_x, 1e-6, "Batch " + std::to_string(b) + " x");
        EXPECT_NEAR(cost_primal[b], 2.5, 1e-6) << "Batch " << b << " objective mismatch";
    }

    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);
}

// ============================================================================
// Generated QP Tests (known KKT solutions)
// ============================================================================

// ============================================================================
// Large Dense P Matrix Tests (Bug Reproduction)
// ============================================================================

// This test reproduces the GPU dense P matrix bug seen in Python.
// Small dense P (5x5) works, but larger dense P (50x50) fails.
// The solver gets stuck with near-zero step sizes and increasing μ.
TEST_F(DenseQPTest, LargeDensePMatrixFactorModel) {
    // Factor model covariance: P = F @ F' + D
    // This creates a dense P matrix (portfolio optimization scenario)
    // n=50 to match the Python test that fails

    const int n = 50;
    const int m = n + 1;  // sum=1 constraint + n nonnegativity constraints
    const int batchSize = 1;
    const int num_factors = 10;

    std::cout << "\n===============================================" << std::endl;
    std::cout << "LARGE DENSE P TEST - Factor Model (n=" << n << ")" << std::endl;
    std::cout << "===============================================" << std::endl;

    // Generate factor model covariance with fixed seed (matches Python)
    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);

    // F is n x num_factors
    std::vector<std::vector<double>> F(n, std::vector<double>(num_factors));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < num_factors; j++) {
            F[i][j] = normal(rng) * 0.1;
        }
    }

    // D is diagonal idiosyncratic variance
    std::vector<double> d(n);
    for (int i = 0; i < n; i++) {
        d[i] = std::abs(normal(rng)) * 0.01 + 0.1;
    }

    // Q = F @ F' + D (full dense covariance matrix)
    std::vector<std::vector<double>> Q(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < num_factors; k++) {
                sum += F[i][k] * F[j][k];
            }
            Q[i][j] = sum;
        }
        Q[i][i] += d[i];  // Add diagonal
    }

    // Convert FULL symmetric Q to CSR format (not just upper triangle!)
    // This is required because the GPU residuals code does a standard SpMV assuming full matrix
    std::vector<int64_t> P_ro;
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;

    P_ro.push_back(0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {  // FULL matrix
            if (std::abs(Q[i][j]) > 1e-14) {
                P_ci.push_back(j);
                P_val.push_back(Q[i][j]);
            }
        }
        P_ro.push_back(static_cast<int64_t>(P_ci.size()));
    }
    int64_t nnzP = static_cast<int64_t>(P_val.size());

    std::cout << "P matrix: " << n << "x" << n << ", nnz=" << nnzP << " (FULL)" << std::endl;
    std::cout << "P density: " << (100.0 * nnzP / (n * n)) << "%" << std::endl;

    // q = 0 (minimize variance only)
    std::vector<double> q_data(n, 0.0);

    // A matrix: row 0 is sum=1, rows 1..n are -x_i <= 0 (nonnegativity)
    // Row 0: [1, 1, ..., 1] (sum constraint)
    // Row i (1..n): -e_i (nonnegativity as -x_i + s_i = 0, s_i >= 0)
    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;

    A_ro.push_back(0);

    // Row 0: sum = 1
    for (int j = 0; j < n; j++) {
        A_ci.push_back(j);
        A_val.push_back(1.0);
    }
    A_ro.push_back(static_cast<int64_t>(A_ci.size()));

    // Rows 1..n: -x_i <= 0
    for (int i = 0; i < n; i++) {
        A_ci.push_back(i);
        A_val.push_back(-1.0);
        A_ro.push_back(static_cast<int64_t>(A_ci.size()));
    }
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    // b = [1, 0, 0, ..., 0]
    std::vector<double> b_data(m, 0.0);
    b_data[0] = 1.0;

    // Cones: 1 zero (equality sum=1) + n nonneg (x >= 0)
    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = n;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 200;
    settings.verbose = true;  // Enable to see convergence issues

    std::cout << "Constraints: " << m << " (1 equality + " << n << " nonneg)" << std::endl;
    std::cout << "Expected: uniform portfolio x = [1/n, ..., 1/n]" << std::endl;
    std::cout << "===============================================\n" << std::endl;

    // Create solver
    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Allocate device memory
    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP * batchSize);
    cudaMalloc(&d_A_values, sizeof(double) * nnzA * batchSize);
    cudaMalloc(&d_q, sizeof(double) * n * batchSize);
    cudaMalloc(&d_b, sizeof(double) * m * batchSize);

    cudaMemcpy(d_P_values, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    // Copy solution
    std::vector<double> x_sol(n);
    std::vector<int32_t> status_sol(batchSize);
    std::vector<double> obj_val(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(obj_val.data(), solver.solution.obj_val.data(), sizeof(double) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Check solution
    std::cout << "\n=== SOLUTION ===" << std::endl;
    std::cout << "Status: " << status_sol[0] << " (1=Solved, 5=MaxIterations)" << std::endl;
    std::cout << "Objective: " << std::scientific << obj_val[0] << std::endl;

    // Check sum(x) = 1
    double sum_x = 0.0;
    for (int i = 0; i < n; i++) {
        sum_x += x_sol[i];
    }
    std::cout << "sum(x) = " << sum_x << " (should be 1.0)" << std::endl;

    // Check x >= 0
    double min_x = *std::min_element(x_sol.begin(), x_sol.end());
    std::cout << "min(x) = " << min_x << " (should be >= 0)" << std::endl;

    // Print first few elements
    std::cout << "x[0..4] = [";
    for (int i = 0; i < std::min(5, n); i++) {
        std::cout << std::fixed << std::setprecision(6) << x_sol[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << ", ...]" << std::endl;

    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);

    // Test expectations
    // NOTE: This test is expected to FAIL on GPU with the dense P bug
    // If it passes, the bug may have been fixed!
    EXPECT_EQ(status_sol[0], static_cast<int32_t>(SolverStatus::Solved))
        << "Solver should converge (currently fails with dense P on GPU)";
    EXPECT_NEAR(sum_x, 1.0, 1e-6) << "Sum constraint violated";
    EXPECT_GE(min_x, -1e-6) << "Nonnegativity constraint violated";
}

// Smaller dense P test to find the size threshold where failure begins
TEST_F(DenseQPTest, MediumDensePMatrixFactorModel) {
    // Same as above but with n=20 to find failure threshold
    const int n = 20;
    const int m = n + 1;
    const int batchSize = 1;
    const int num_factors = 5;

    std::cout << "\n===============================================" << std::endl;
    std::cout << "MEDIUM DENSE P TEST - Factor Model (n=" << n << ")" << std::endl;
    std::cout << "===============================================" << std::endl;

    std::mt19937 rng(42);
    std::normal_distribution<double> normal(0.0, 1.0);

    std::vector<std::vector<double>> F(n, std::vector<double>(num_factors));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < num_factors; j++) {
            F[i][j] = normal(rng) * 0.1;
        }
    }

    std::vector<double> d(n);
    for (int i = 0; i < n; i++) {
        d[i] = std::abs(normal(rng)) * 0.01 + 0.1;
    }

    std::vector<std::vector<double>> Q(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < num_factors; k++) {
                sum += F[i][k] * F[j][k];
            }
            Q[i][j] = sum;
        }
        Q[i][i] += d[i];
    }

    // Convert FULL symmetric Q to CSR format (required for GPU SpMV)
    std::vector<int64_t> P_ro;
    std::vector<int64_t> P_ci;
    std::vector<double> P_val;

    P_ro.push_back(0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {  // FULL matrix
            if (std::abs(Q[i][j]) > 1e-14) {
                P_ci.push_back(j);
                P_val.push_back(Q[i][j]);
            }
        }
        P_ro.push_back(static_cast<int64_t>(P_ci.size()));
    }
    int64_t nnzP = static_cast<int64_t>(P_val.size());

    std::cout << "P matrix: " << n << "x" << n << ", nnz=" << nnzP << " (FULL)" << std::endl;

    std::vector<double> q_data(n, 0.0);

    std::vector<int64_t> A_ro;
    std::vector<int64_t> A_ci;
    std::vector<double> A_val;

    A_ro.push_back(0);
    for (int j = 0; j < n; j++) {
        A_ci.push_back(j);
        A_val.push_back(1.0);
    }
    A_ro.push_back(static_cast<int64_t>(A_ci.size()));

    for (int i = 0; i < n; i++) {
        A_ci.push_back(i);
        A_val.push_back(-1.0);
        A_ro.push_back(static_cast<int64_t>(A_ci.size()));
    }
    int64_t nnzA = static_cast<int64_t>(A_val.size());

    std::vector<double> b_data(m, 0.0);
    b_data[0] = 1.0;

    Cones cones{};
    cones.numZeroCones = 1;
    cones.numNonnegCones = n;

    Settings settings;
    settings.ipm.equilibrationSettings.enable = true;
    settings.maxIter = 200;
    settings.verbose = false;

    CompiledSolver solver(
        n, m, batchSize,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    double *d_P_values, *d_A_values, *d_q, *d_b;
    cudaMalloc(&d_P_values, sizeof(double) * nnzP);
    cudaMalloc(&d_A_values, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P_values, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_values, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q_data.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_data.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P_values, d_A_values, d_q, d_b);

    std::vector<double> x_sol(n);
    std::vector<int32_t> status_sol(batchSize);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(status_sol.data(), solver.solution.status.get(), sizeof(int32_t) * batchSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    double sum_x = 0.0;
    for (int i = 0; i < n; i++) sum_x += x_sol[i];

    std::cout << "Status: " << status_sol[0] << ", sum(x) = " << sum_x << std::endl;

    cudaFree(d_P_values);
    cudaFree(d_A_values);
    cudaFree(d_q);
    cudaFree(d_b);

    EXPECT_EQ(status_sol[0], static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(sum_x, 1.0, 1e-6);
}

// ============================================================================
// Generated QP Tests (known KKT solutions)
// ============================================================================

TEST_F(GeneratedQPTest, GeneratedSmallDense) {
    // Dense problem with a constructed optimal solution
    GeneratedQP qp(12, 6, 1.0, 0.5, 0.6, 1e-2, 2025);

    auto [A_ro, A_ci, A_val, b_combined, total_m] = qp.toMoreauFormat();
    auto [P_ro, P_ci, P_val] = qp.PToCSRFull();

    const int64_t n = qp.n;
    const int64_t m = total_m;
    const int64_t batch = 1;
    const int64_t nnzA = static_cast<int64_t>(A_val.size());
    const int64_t nnzP = static_cast<int64_t>(P_val.size());

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.verbose = false;
    settings.maxIter = 300;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        n, m, batch,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Copy problem data to device
    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * std::max<int64_t>(1, nnzP));
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, qp.q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_combined.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    std::vector<double> z_sol(m);
    std::vector<int32_t> status(batch);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);

    EXPECT_EQ(status[0], 1) << "Solver should find optimal solution";

    // Map Moreau duals back to (y,s) for original problem
    std::vector<double> y_sol(qp.m);
    std::vector<double> s_dual(qp.n);
    for (int64_t i = 0; i < qp.m; i++) {
        y_sol[i] = z_sol[i];
    }
    for (int64_t j = 0; j < qp.n; j++) {
        s_dual[j] = z_sol[qp.m + j];
    }

    // Check KKT for solver output and for the generated ground truth
    bool kkt_solver = checkQPKKT(qp.P_dense, qp.q, qp.A_dense, qp.b, x_sol, y_sol, s_dual, 1e-5, false);
    if (!kkt_solver) {
        checkQPKKT(qp.P_dense, qp.q, qp.A_dense, qp.b, x_sol, y_sol, s_dual, 1e-5, true);
    }
    EXPECT_TRUE(kkt_solver);
    EXPECT_TRUE(checkQPKKT(qp.P_dense, qp.q, qp.A_dense, qp.b, qp.x_star, qp.y_star, qp.s_star, 1e-6, false));

    // Objective agreement
    double obj_solver = qp.objective(x_sol);
    double obj_expected = qp.objective(qp.x_star);
    EXPECT_NEAR(obj_solver, obj_expected, 1e-6);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

TEST_F(GeneratedQPTest, GeneratedMediumSparse) {
    // Moderately sized sparse problem
    GeneratedQP qp(40, 15, 0.25, 0.5, 0.5, 1e-2, 4242);

    auto [A_ro, A_ci, A_val, b_combined, total_m] = qp.toMoreauFormat();
    auto [P_ro, P_ci, P_val] = qp.PToCSRFull();

    const int64_t n = qp.n;
    const int64_t m = total_m;
    const int64_t batch = 1;
    const int64_t nnzA = static_cast<int64_t>(A_val.size());
    const int64_t nnzP = static_cast<int64_t>(P_val.size());

    Cones cones{};
    cones.numNonnegCones = m;

    Settings settings;
    settings.verbose = false;
    settings.maxIter = 400;
    settings.ipm.equilibrationSettings.enable = true;

    CompiledSolver solver(
        n, m, batch,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * std::max<int64_t>(1, nnzP));
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_P, P_val.data(), sizeof(double) * nnzP, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, qp.q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_combined.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    std::vector<double> z_sol(m);
    std::vector<int32_t> status(batch);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);

    EXPECT_EQ(status[0], 1) << "Solver should find optimal solution";

    std::vector<double> y_sol(qp.m);
    std::vector<double> s_dual(qp.n);
    for (int64_t i = 0; i < qp.m; i++) y_sol[i] = z_sol[i];
    for (int64_t j = 0; j < qp.n; j++) s_dual[j] = z_sol[qp.m + j];

    bool kkt_solver = checkQPKKT(qp.P_dense, qp.q, qp.A_dense, qp.b, x_sol, y_sol, s_dual, 1e-4, false);
    if (!kkt_solver) {
        checkQPKKT(qp.P_dense, qp.q, qp.A_dense, qp.b, x_sol, y_sol, s_dual, 1e-4, true);
    }
    EXPECT_TRUE(kkt_solver);
    EXPECT_TRUE(checkQPKKT(qp.P_dense, qp.q, qp.A_dense, qp.b, qp.x_star, qp.y_star, qp.s_star, 1e-6, false));

    double obj_solver = qp.objective(x_sol);
    double obj_expected = qp.objective(qp.x_star);
    EXPECT_NEAR(obj_solver, obj_expected, 1e-5);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
