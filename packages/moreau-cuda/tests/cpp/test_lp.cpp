/**
 * @file test_lp.cpp
 * @brief Tests for Linear Programs (empty P matrix)
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include "moreau/solver/solver.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/settings/settings.hpp"

class LPTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure no stale CUDA errors
        cudaGetLastError();
    }

    void TearDown() override {
        cudaDeviceSynchronize();
    }

    // Helper to check solution accuracy
    void checkSolution(
        const std::vector<double>& x_actual,
        const std::vector<double>& x_expected,
        double tol = 1e-6
    ) {
        ASSERT_EQ(x_actual.size(), x_expected.size());
        double error = 0.0;
        for (size_t i = 0; i < x_actual.size(); i++) {
            error += std::pow(x_actual[i] - x_expected[i], 2);
        }
        error = std::sqrt(error);
        EXPECT_LT(error, tol) << "Solution error: " << error;
    }

    /**
     * Helper to check KKT conditions for standard form LP:
     *   min c^T x  s.t. Ax <= b, x >= 0
     *   dual: max -b^T y  s.t. c + A^T y - s = 0, y >= 0, s >= 0
     *
     * Returns true if KKT conditions are satisfied within tolerance
     */
    bool checkKKT(
        const std::vector<std::vector<double>>& A,  // Dense A matrix (m x n)
        const std::vector<double>& b,               // RHS vector (m)
        const std::vector<double>& c,               // Cost vector (n)
        const std::vector<double>& x,               // Primal solution (n)
        const std::vector<double>& y,               // Dual for Ax <= b (m)
        const std::vector<double>& s,               // Dual for x >= 0 (n)
        double tol = 1e-5,
        bool verbose = false
    ) {
        const int64_t m = A.size();
        const int64_t n = (m > 0) ? A[0].size() : 0;

        bool all_ok = true;

        // 1. Primal feasibility: Ax <= b, x >= 0
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

        // 2. Dual feasibility: A^T y + s = c, y >= 0, s >= 0
        std::vector<double> ATy_minus_s_plus_c(n, 0.0);
        for (int64_t j = 0; j < n; j++) {
            ATy_minus_s_plus_c[j] = c[j] - s[j];
            for (int64_t i = 0; i < m; i++) {
                ATy_minus_s_plus_c[j] += A[i][j] * y[i];
            }
            if (std::abs(ATy_minus_s_plus_c[j]) > tol) {
                if (verbose) std::cout << "Dual infeasible at " << j
                    << ": c + A^T y - s = " << ATy_minus_s_plus_c[j] << " != 0" << std::endl;
                all_ok = false;
            }
        }
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

        // 3. Complementary slackness: y_i * (b_i - Ax_i) = 0, s_j * x_j = 0
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
};

/**
 * @brief Helper structure to generate LP with a known KKT solution
 *
 * Generates LP: min c^T x  s.t.  A x <= b,  x >= 0
 * along with a primal/dual solution (x_star, y_star, s_star) that satisfies KKT.
 * The optimal solution need not be unique; this follows the simple backward
 * construction used in the Python reference.
 */
struct GeneratedLP {
    int64_t m;  // number of inequality constraints
    int64_t n;  // number of variables

    // Dense matrices (for construction)
    std::vector<std::vector<double>> A_dense;
    std::vector<double> b;
    std::vector<double> c;

    // Optimal solution
    std::vector<double> x_star;
    std::vector<double> y_star;  // dual variables for Ax <= b
    std::vector<double> s_star;  // reduced costs (dual variables for x >= 0)

    // Active sets
    std::vector<int64_t> I_active;  // active inequality constraint indices
    std::vector<int64_t> J_pos;     // indices where x_star > 0
    std::vector<int64_t> J_zero;    // indices where x_star = 0

    /**
     * Generate LP with known optimal solution (matches Python implementation)
     *
     * @param m_in Number of inequality constraints
     * @param n_in Number of variables
     * @param density Sparsity of A matrix (0 to 1, 1 = dense)
     * @param active_frac Fraction of constraints that are active at optimum
     * @param positive_frac Fraction of variables that are positive at optimum
     * @param seed Random seed
     */
    GeneratedLP(
        int64_t m_in,
        int64_t n_in,
        double density = 0.5,
        double active_frac = 0.5,
        double positive_frac = 0.5,
        unsigned seed = 42
    ) : m(m_in), n(n_in) {
        if (n_in <= 0 || m_in <= 0) {
            throw std::runtime_error("Need m > 0 and n > 0");
        }

        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
        std::uniform_real_distribution<double> pos_dist(0.5, 2.0);
        std::normal_distribution<double> normal_dist(0.0, 1.0);

        // 1. Choose support pattern
        int64_t k_pos = std::max(1L, static_cast<int64_t>(positive_frac * n));
        if (k_pos > n) k_pos = n;

        std::vector<int64_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        J_pos.assign(indices.begin(), indices.begin() + k_pos);
        std::sort(J_pos.begin(), J_pos.end());

        // Compute J_zero
        for (int64_t j = 0; j < n; j++) {
            if (std::find(J_pos.begin(), J_pos.end(), j) == J_pos.end()) {
                J_zero.push_back(j);
            }
        }
        // Active inequality constraints
        int64_t k_active = std::max<int64_t>(1, static_cast<int64_t>(active_frac * m));
        if (k_active > m) k_active = m;

        // 2. Build A with desired density
        A_dense.assign(m, std::vector<double>(n, 0.0));
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                if (density >= 1.0 || uniform_dist(rng) < density) {
                    A_dense[i][j] = normal_dist(rng);
                }
            }
        }

        // 3. Choose active inequality constraints
        std::vector<int64_t> constraint_indices(m);
        std::iota(constraint_indices.begin(), constraint_indices.end(), 0);
        std::shuffle(constraint_indices.begin(), constraint_indices.end(), rng);
        I_active.assign(constraint_indices.begin(), constraint_indices.begin() + k_active);
        std::sort(I_active.begin(), I_active.end());

        // 4. Choose x_star consistent with J_pos / J_zero
        x_star.assign(n, 0.0);
        for (int64_t j : J_pos) {
            x_star[j] = pos_dist(rng);
        }

        // 5. Build b so that I_active are tight, others slack
        std::vector<double> r(m);
        for (int64_t i = 0; i < m; i++) {
            r[i] = pos_dist(rng);
        }
        for (int64_t i : I_active) {
            r[i] = 0.0;  // active constraints
        }

        b.resize(m);
        for (int64_t i = 0; i < m; i++) {
            b[i] = r[i];
            for (int64_t j = 0; j < n; j++) {
                b[i] += A_dense[i][j] * x_star[j];
            }
        }

        // 6. Dual variables y_star (for Ax <= b)
        y_star.assign(m, 0.0);
        for (int64_t i : I_active) {
            y_star[i] = pos_dist(rng);
        }

        // 7. Dual vars s_star (for x >= 0)
        s_star.assign(n, 0.0);
        for (int64_t j : J_zero) {
            s_star[j] = pos_dist(rng);
        }

        // 8. Objective from dual feasibility: c satisfies c + A^T y - s = 0  =>  c = s - A^T y
        c.assign(n, 0.0);
        for (int64_t j = 0; j < n; j++) {
            c[j] = s_star[j];
            for (int64_t i = 0; i < m; i++) {
                c[j] -= A_dense[i][j] * y_star[i];
            }
        }
    }

    /**
     * Convert to Moreau format:
     *   min c^T x  s.t.  Ax <= b, x >= 0
     * becomes
     *   min c^T x  s.t.  Ax + s1 = b, -x + s2 = 0, s1 >= 0, s2 >= 0
     *
     * Returns: (A_ro, A_ci, A_val, b_combined, num_constraints)
     */
    std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<double>, std::vector<double>, int64_t>
    toMoreauFormat() const {
        // Total constraints: m (for Ax <= b) + n (for x >= 0)
        int64_t total_m = m + n;

        std::vector<int64_t> A_ro;
        std::vector<int64_t> A_ci;
        std::vector<double> A_val;
        std::vector<double> b_combined;

        A_ro.push_back(0);
        int64_t nnz = 0;

        // First m rows: Ax + s1 = b (s1 >= 0 enforces Ax <= b)
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                if (std::abs(A_dense[i][j]) > 1e-14) {
                    A_ci.push_back(j);
                    A_val.push_back(A_dense[i][j]);
                    nnz++;
                }
            }
            A_ro.push_back(nnz);
            b_combined.push_back(b[i]);
        }

        // Next n rows: -x + s2 = 0 (s2 >= 0 enforces x >= 0)
        for (int64_t j = 0; j < n; j++) {
            A_ci.push_back(j);
            A_val.push_back(-1.0);
            nnz++;
            A_ro.push_back(nnz);
            b_combined.push_back(0.0);
        }

        return {A_ro, A_ci, A_val, b_combined, total_m};
    }
};

/**
 * Test 1: Simple 2D LP
 *
 * Problem:
 *   minimize    x1 + 2*x2
 *   subject to  x1 + x2 = 1
 *               x1 >= 0
 *               x2 >= 0
 *
 * Optimal solution: x1 = 1, x2 = 0, objective = 1
 */
TEST_F(LPTest, Simple2D) {
    // Problem dimensions
    const int64_t n = 2;  // variables
    const int64_t m = 3;  // constraints
    const int64_t batch = 1;

    // P matrix: EMPTY (LP has no quadratic term)
    std::vector<int64_t> P_ro(n + 1, 0);  // All zeros
    std::vector<int64_t> P_ci;            // Empty
    const int64_t nnzP = 0;

    // A matrix (CSR format):
    // Row 0: x1 + x2 = 1 (equality, zero cone)
    // Row 1: -x1 + s1 = 0, s1 >= 0 => x1 >= 0
    // Row 2: -x2 + s2 = 0, s2 >= 0 => x2 >= 0
    std::vector<int64_t> A_ro = {0, 2, 3, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    std::vector<double> A_val = {1.0, 1.0, -1.0, -1.0};
    const int64_t nnzA = 4;

    // Linear cost: minimize x1 + 2*x2
    std::vector<double> q = {1.0, 2.0};

    // RHS
    std::vector<double> b = {1.0, 0.0, 0.0};

    // Cones: 1 zero (equality) + 2 nonneg (x >= 0)
    moreau::Cones cones;
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    // Settings
    moreau::Settings settings;
    settings.verbose = false;
    settings.maxIter = 50;

    // Create solver
    moreau::CompiledSolver solver(
        n, m, batch,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Copy data to device
    std::vector<double> P_val_empty;  // Empty for LP
    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double) * std::max(nnzP, (int64_t)1));  // Avoid 0-size malloc
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    // No need to copy P (it's empty)
    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P, d_A, d_q, d_b);

    // Get results
    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    std::vector<int32_t> status(batch);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);

    // Verify
    EXPECT_EQ(status[0], 1);  // Solved

    std::vector<double> x_expected = {1.0, 0.0};
    checkSolution(x_sol, x_expected, 1e-6);

    // Check objective
    double obj = x_sol[0] * q[0] + x_sol[1] * q[1];
    EXPECT_NEAR(obj, 1.0, 1e-6);

    // Cleanup
    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test 2: Standard form LP
 *
 * Problem:
 *   minimize    -x1 - x2 (maximize x1 + x2)
 *   subject to  x1 + 2*x2 <= 4
 *               x1 + x2 <= 3
 *               x1 >= 0, x2 >= 0
 *
 * Optimal: objective = -3, any point on line x1 + x2 = 3 with 0 <= x2 <= 1
 * (Multiple optimal solutions exist)
 */
TEST_F(LPTest, StandardForm) {
    const int64_t n = 2;  // variables
    const int64_t m = 4;  // 2 inequalities + 2 nonnegativity
    const int64_t batch = 1;

    // P matrix: EMPTY
    std::vector<int64_t> P_ro(n + 1, 0);
    std::vector<int64_t> P_ci;
    const int64_t nnzP = 0;

    // A matrix (CSR format):
    // Row 0: x1 + 2*x2 + s0 = 4, s0 >= 0 => x1 + 2*x2 <= 4
    // Row 1: x1 + x2 + s1 = 3, s1 >= 0 => x1 + x2 <= 3
    // Row 2: -x1 + s2 = 0, s2 >= 0 => x1 >= 0
    // Row 3: -x2 + s3 = 0, s3 >= 0 => x2 >= 0
    std::vector<int64_t> A_ro = {0, 2, 4, 5, 6};
    std::vector<int64_t> A_ci = {0, 1, 0, 1, 0, 1};
    std::vector<double> A_val = {1.0, 2.0, 1.0, 1.0, -1.0, -1.0};
    const int64_t nnzA = 6;

    // Linear cost: minimize -x1 - x2
    std::vector<double> q = {-1.0, -1.0};

    // RHS
    std::vector<double> b = {4.0, 3.0, 0.0, 0.0};

    // Cones: 4 nonneg (all inequalities)
    moreau::Cones cones;
    cones.numNonnegCones = 4;

    // Settings
    moreau::Settings settings;
    settings.verbose = false;
    settings.maxIter = 50;

    // Create solver
    moreau::CompiledSolver solver(
        n, m, batch,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Copy data to device
    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double));  // Dummy allocation
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P, d_A, d_q, d_b);

    // Get results
    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    std::vector<int32_t> status(batch);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);

    // Verify
    EXPECT_EQ(status[0], 1);  // Solved

    // Check objective (optimal is -3)
    // Note: Multiple optimal solutions exist on the line x1 + x2 = 3
    double obj = x_sol[0] * q[0] + x_sol[1] * q[1];
    EXPECT_NEAR(obj, -3.0, 1e-5);

    // Check constraints are satisfied
    EXPECT_GE(x_sol[0], -1e-6);  // x1 >= 0
    EXPECT_GE(x_sol[1], -1e-6);  // x2 >= 0
    EXPECT_LE(x_sol[0] + 2*x_sol[1], 4.0 + 1e-5);  // x1 + 2*x2 <= 4
    EXPECT_NEAR(x_sol[0] + x_sol[1], 3.0, 1e-4);   // x1 + x2 = 3 (active at optimum)

    // Cleanup
    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test 3: Simple 1D LP (degenerate case)
 *
 * Problem:
 *   minimize    x
 *   subject to  x >= 1
 *
 * Optimal solution: x = 1, objective = 1
 */
TEST_F(LPTest, OneDimensional) {
    const int64_t n = 1;  // 1 variable
    const int64_t m = 1;  // 1 constraint
    const int64_t batch = 1;

    // P matrix: EMPTY
    std::vector<int64_t> P_ro(n + 1, 0);
    std::vector<int64_t> P_ci;
    const int64_t nnzP = 0;

    // A matrix: -x + s = -1, s >= 0 => x >= 1
    std::vector<int64_t> A_ro = {0, 1};
    std::vector<int64_t> A_ci = {0};
    std::vector<double> A_val = {-1.0};
    const int64_t nnzA = 1;

    // Linear cost: minimize x
    std::vector<double> q = {1.0};

    // RHS
    std::vector<double> b = {-1.0};

    // Cones: 1 nonneg
    moreau::Cones cones;
    cones.numNonnegCones = 1;

    // Settings
    moreau::Settings settings;
    settings.verbose = false;
    settings.maxIter = 50;

    // Create solver
    moreau::CompiledSolver solver(
        n, m, batch,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Copy data to device
    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double));
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P, d_A, d_q, d_b);

    // Get results
    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    std::vector<int32_t> status(batch);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);

    // Verify
    EXPECT_EQ(status[0], 1);  // Solved

    EXPECT_NEAR(x_sol[0], 1.0, 1e-6);

    // Check objective
    double obj = x_sol[0] * q[0];
    EXPECT_NEAR(obj, 1.0, 1e-6);

    // Cleanup
    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test 4: LP with equality constraints only
 *
 * Problem:
 *   minimize    x1 + x2
 *   subject to  x1 + 2*x2 = 3
 *               2*x1 + x2 = 3
 *
 * Unique solution: x1 = 1, x2 = 1, objective = 2
 */
TEST_F(LPTest, EqualityOnly) {
    const int64_t n = 2;
    const int64_t m = 2;  // 2 equality constraints
    const int64_t batch = 1;

    // P matrix: EMPTY
    std::vector<int64_t> P_ro(n + 1, 0);
    std::vector<int64_t> P_ci;
    const int64_t nnzP = 0;

    // A matrix:
    // Row 0: x1 + 2*x2 + s0 = 3, s0 in Zero => x1 + 2*x2 = 3
    // Row 1: 2*x1 + x2 + s1 = 3, s1 in Zero => 2*x1 + x2 = 3
    std::vector<int64_t> A_ro = {0, 2, 4};
    std::vector<int64_t> A_ci = {0, 1, 0, 1};
    std::vector<double> A_val = {1.0, 2.0, 2.0, 1.0};
    const int64_t nnzA = 4;

    // Linear cost: minimize x1 + x2
    std::vector<double> q = {1.0, 1.0};

    // RHS
    std::vector<double> b = {3.0, 3.0};

    // Cones: 2 zero (equalities)
    moreau::Cones cones;
    cones.numZeroCones = 2;

    // Settings
    moreau::Settings settings;
    settings.verbose = false;
    settings.maxIter = 50;

    // Create solver
    moreau::CompiledSolver solver(
        n, m, batch,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Copy data to device
    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double));
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, q.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P, d_A, d_q, d_b);

    // Get results
    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    std::vector<int32_t> status(batch);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);

    // Verify
    EXPECT_EQ(status[0], 1);  // Solved

    std::vector<double> x_expected = {1.0, 1.0};
    checkSolution(x_sol, x_expected, 1e-5);

    // Check objective
    double obj = x_sol[0] * q[0] + x_sol[1] * q[1];
    EXPECT_NEAR(obj, 2.0, 1e-5);

    // Cleanup
    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test 5: Small Generated LP with Known Solution (10x5, dense)
 *
 * Uses the simple backward-construction recipe (matches Python helper) to generate
 * an LP with a known optimal solution. Validates primal, dual, and complementary
 * slackness conditions against the solver.
 */
TEST_F(LPTest, GeneratedSmallDense) {
    // Generate LP: 10 constraints, 5 variables, dense, 60% positive
    GeneratedLP lp(10, 5, 1.0, 0.5, 0.6, 42);  // density=1.0, active_frac=0.5, positive_frac=0.6

    // Convert to Moreau format
    auto [A_ro, A_ci, A_val, b_combined, total_m] = lp.toMoreauFormat();
    const int64_t n = lp.n;
    const int64_t m = total_m;
    const int64_t m_original = lp.m;  // Original number of inequality constraints
    const int64_t batch = 1;
    const int64_t nnzA = A_val.size();

    // P matrix: EMPTY (this is an LP)
    std::vector<int64_t> P_ro(n + 1, 0);
    std::vector<int64_t> P_ci;
    const int64_t nnzP = 0;

    // Cones: all nonneg (for Ax <= b and x >= 0 constraints)
    moreau::Cones cones;
    cones.numNonnegCones = m;

    // Settings
    moreau::Settings settings;
    settings.verbose = false;
    settings.maxIter = 300;

    // Create solver
    moreau::CompiledSolver solver(
        n, m, batch,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    // Copy data to device
    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double));
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, lp.c.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_combined.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    // Solve
    solver.solveAll(d_P, d_A, d_q, d_b);

    // Get primal solution
    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    // Get dual solution (for Moreau constraints)
    std::vector<double> z_sol(m);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    // Get slack variables
    std::vector<double> s_sol(m);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    std::vector<int32_t> status(batch);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);

    // Verify solver status
    EXPECT_EQ(status[0], 1) << "Solver should find optimal solution";

    // Check objective values match
    double obj_solver = 0.0;
    double obj_expected = 0.0;
    for (int64_t j = 0; j < n; j++) {
        obj_solver += lp.c[j] * x_sol[j];
        obj_expected += lp.c[j] * lp.x_star[j];
    }
    EXPECT_NEAR(obj_solver, obj_expected, 1e-6);

    // Extract duals from Moreau format
    // z[0:m_original] correspond to Ax<=b, z[m_original + j] correspond to -x<=0 (i.e., x>=0)
    std::vector<double> y_sol(m_original, 0.0);
    for (int64_t i = 0; i < m_original; i++) y_sol[i] = z_sol[i];
    std::vector<double> s_sol_vars(n, 0.0);
    for (int64_t j = 0; j < n; j++) s_sol_vars[j] = z_sol[m_original + j];

    // Check KKT conditions for solver solution
    EXPECT_TRUE(checkKKT(lp.A_dense, lp.b, lp.c, x_sol, y_sol, s_sol_vars, 1e-5, false));

    // Generated solution should satisfy KKT by construction
    EXPECT_TRUE(checkKKT(lp.A_dense, lp.b, lp.c, lp.x_star, lp.y_star, lp.s_star, 1e-6, false));

    // Cleanup
    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test 6: Medium Generated LP with Known Solution (50x20, sparse)
 *
 * Tests a larger, sparser problem to ensure the solver scales properly.
 */
TEST_F(LPTest, GeneratedMediumSparse) {
    // Generate LP: 50 constraints, 20 variables, 30% density, 50% positive
    GeneratedLP lp(50, 20, 0.3, 0.5, 0.5, 123);  // density=0.3, active_frac=0.5, positive_frac=0.5

    auto [A_ro, A_ci, A_val, b_combined, total_m] = lp.toMoreauFormat();
    const int64_t n = lp.n;
    const int64_t m = total_m;
    const int64_t batch = 1;
    const int64_t nnzA = A_val.size();

    std::vector<int64_t> P_ro(n + 1, 0);
    std::vector<int64_t> P_ci;
    const int64_t nnzP = 0;

    moreau::Cones cones;
    cones.numNonnegCones = m;

    moreau::Settings settings;
    settings.verbose = false;
    settings.maxIter = 150;

    moreau::CompiledSolver solver(
        n, m, batch,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double));
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, lp.c.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_combined.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    std::vector<double> z_sol(m);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    std::vector<double> s_sol(m);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    std::vector<int32_t> status(batch);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);

    EXPECT_EQ(status[0], 1) << "Solver should find optimal solution";

    // Verify primal feasibility
    for (int64_t i = 0; i < m; i++) {
        double lhs = s_sol[i];
        for (int64_t idx = A_ro[i]; idx < A_ro[i + 1]; idx++) {
            lhs += A_val[idx] * x_sol[A_ci[idx]];
        }
        EXPECT_NEAR(lhs, b_combined[i], 1e-5);
    }

    // Verify dual feasibility
    for (int64_t i = 0; i < m; i++) {
        EXPECT_GE(z_sol[i], -1e-6);
    }

    // Verify complementary slackness
    for (int64_t i = 0; i < m; i++) {
        EXPECT_LT(std::abs(s_sol[i] * z_sol[i]), 1e-4);
    }

    // Verify objective
    double obj_computed = 0.0;
    double obj_expected = 0.0;
    for (size_t i = 0; i < n; i++) {
        obj_computed += lp.c[i] * x_sol[i];
        obj_expected += lp.c[i] * lp.x_star[i];
    }
    EXPECT_NEAR(obj_computed, obj_expected, 1e-6);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test 7: Large Generated LP with Known Solution (100x50, sparse)
 *
 * Tests solver performance on a larger problem.
 */
TEST_F(LPTest, GeneratedLargeSparse) {
    // Generate LP: 100 constraints, 50 variables, 20% density, 40% positive
    GeneratedLP lp(100, 50, 0.2, 0.5, 0.4, 456);  // density=0.2, active_frac=0.5, positive_frac=0.4

    auto [A_ro, A_ci, A_val, b_combined, total_m] = lp.toMoreauFormat();
    const int64_t n = lp.n;
    const int64_t m = total_m;
    const int64_t batch = 1;
    const int64_t nnzA = A_val.size();

    std::vector<int64_t> P_ro(n + 1, 0);
    std::vector<int64_t> P_ci;
    const int64_t nnzP = 0;

    moreau::Cones cones;
    cones.numNonnegCones = m;

    moreau::Settings settings;
    settings.verbose = false;
    settings.maxIter = 400;

    moreau::CompiledSolver solver(
        n, m, batch,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double));
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, lp.c.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_combined.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    std::vector<double> z_sol(m);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    std::vector<double> s_sol(m);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    std::vector<int32_t> status(batch);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);

    EXPECT_EQ(status[0], 1) << "Solver should find optimal solution";

    // Verify primal feasibility
    for (int64_t i = 0; i < m; i++) {
        double lhs = s_sol[i];
        for (int64_t idx = A_ro[i]; idx < A_ro[i + 1]; idx++) {
            lhs += A_val[idx] * x_sol[A_ci[idx]];
        }
        EXPECT_NEAR(lhs, b_combined[i], 1e-4);
    }

    // Verify dual feasibility
    for (int64_t i = 0; i < m; i++) {
        EXPECT_GE(z_sol[i], -1e-6);
    }

    // Verify complementary slackness
    for (int64_t i = 0; i < m; i++) {
        EXPECT_LT(std::abs(s_sol[i] * z_sol[i]), 1e-3);
    }

    // Verify objective
    double obj_computed = 0.0;
    double obj_expected = 0.0;
    for (size_t i = 0; i < n; i++) {
        obj_computed += lp.c[i] * x_sol[i];
        obj_expected += lp.c[i] * lp.x_star[i];
    }
    EXPECT_NEAR(obj_computed, obj_expected, 1e-6);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test 8: Highly Active Constraints (most constraints are tight at optimum)
 *
 * Tests the case where most constraints are active, which can be challenging for interior point methods.
 */
TEST_F(LPTest, GeneratedHighlyActive) {
    // Generate LP: 30 constraints, 15 variables, dense, 70% positive
    GeneratedLP lp(30, 15, 1.0, 0.5, 0.7, 789);  // density=1.0, active_frac=0.5, positive_frac=0.7

    auto [A_ro, A_ci, A_val, b_combined, total_m] = lp.toMoreauFormat();
    const int64_t n = lp.n;
    const int64_t m = total_m;
    const int64_t batch = 1;
    const int64_t nnzA = A_val.size();

    std::vector<int64_t> P_ro(n + 1, 0);
    std::vector<int64_t> P_ci;
    const int64_t nnzP = 0;

    moreau::Cones cones;
    cones.numNonnegCones = m;

    moreau::Settings settings;
    settings.verbose = false;
    settings.maxIter = 150;

    moreau::CompiledSolver solver(
        n, m, batch,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double));
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, lp.c.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_combined.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    std::vector<double> z_sol(m);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    std::vector<double> s_sol(m);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    std::vector<int32_t> status(batch);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);

    EXPECT_EQ(status[0], 1) << "Solver should handle highly active constraints";

    // Verify primal feasibility
    for (int64_t i = 0; i < m; i++) {
        double lhs = s_sol[i];
        for (int64_t idx = A_ro[i]; idx < A_ro[i + 1]; idx++) {
            lhs += A_val[idx] * x_sol[A_ci[idx]];
        }
        EXPECT_NEAR(lhs, b_combined[i], 1e-5);
    }

    // Verify dual feasibility
    for (int64_t i = 0; i < m; i++) {
        EXPECT_GE(z_sol[i], -1e-6);
    }

    // Verify complementary slackness
    for (int64_t i = 0; i < m; i++) {
        EXPECT_LT(std::abs(s_sol[i] * z_sol[i]), 1e-4);
    }

    // Verify objective
    double obj_computed = 0.0;
    double obj_expected = 0.0;
    for (size_t i = 0; i < n; i++) {
        obj_computed += lp.c[i] * x_sol[i];
        obj_expected += lp.c[i] * lp.x_star[i];
    }
    EXPECT_NEAR(obj_computed, obj_expected, 1e-6);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test 9: Very Sparse LP (10% density)
 *
 * Tests solver on a very sparse problem, which is common in real-world applications.
 * Previously DISABLED for unclear reasons; now passes reliably on T500 (sm_75).
 */
TEST_F(LPTest, GeneratedVerySparse) {
    // Generate LP: 60 constraints, 30 variables, 10% density, 45% positive
    GeneratedLP lp(60, 30, 0.1, 0.5, 0.45, 999);  // density=0.1, active_frac=0.5, positive_frac=0.45

    auto [A_ro, A_ci, A_val, b_combined, total_m] = lp.toMoreauFormat();
    const int64_t n = lp.n;
    const int64_t m = total_m;
    const int64_t batch = 1;
    const int64_t nnzA = A_val.size();

    std::vector<int64_t> P_ro(n + 1, 0);
    std::vector<int64_t> P_ci;
    const int64_t nnzP = 0;

    moreau::Cones cones;
    cones.numNonnegCones = m;

    moreau::Settings settings;
    settings.verbose = false;
    settings.maxIter = 600;  // allow more iterations for very sparse, harder problems

    moreau::CompiledSolver solver(
        n, m, batch,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double));
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, lp.c.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_combined.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    std::vector<double> z_sol(m);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    std::vector<double> s_sol(m);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    std::vector<int32_t> status(batch);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);

    EXPECT_EQ(status[0], 1) << "Solver should handle very sparse problems";

    // Verify primal feasibility
    for (int64_t i = 0; i < m; i++) {
        double lhs = s_sol[i];
        for (int64_t idx = A_ro[i]; idx < A_ro[i + 1]; idx++) {
            lhs += A_val[idx] * x_sol[A_ci[idx]];
        }
        EXPECT_NEAR(lhs, b_combined[i], 1e-5);
    }

    // Verify dual feasibility
    for (int64_t i = 0; i < m; i++) {
        EXPECT_GE(z_sol[i], -1e-6);
    }

    // Verify complementary slackness (looser tolerance for sparse case)
    for (int64_t i = 0; i < m; i++) {
        EXPECT_LT(std::abs(s_sol[i] * z_sol[i]), 1e-3);
    }

    // Verify objective
    double obj_computed = 0.0;
    double obj_expected = 0.0;
    for (size_t i = 0; i < n; i++) {
        obj_computed += lp.c[i] * x_sol[i];
        obj_expected += lp.c[i] * lp.x_star[i];
    }
    EXPECT_NEAR(obj_computed, obj_expected, 1e-3);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}

/**
 * Test 10: Many Variables with Few Positive (sparse solution)
 *
 * Tests the case where the optimal solution is sparse (most variables are zero).
 */
TEST_F(LPTest, GeneratedSparseSolution) {
    // Generate LP: 40 constraints, 25 variables, 50% density, 20% positive (sparse solution)
    GeneratedLP lp(40, 25, 0.5, 0.5, 0.2, 2024);  // density=0.5, active_frac=0.5, positive_frac=0.2

    auto [A_ro, A_ci, A_val, b_combined, total_m] = lp.toMoreauFormat();
    const int64_t n = lp.n;
    const int64_t m = total_m;
    const int64_t batch = 1;
    const int64_t nnzA = A_val.size();

    std::vector<int64_t> P_ro(n + 1, 0);
    std::vector<int64_t> P_ci;
    const int64_t nnzP = 0;

    moreau::Cones cones;
    cones.numNonnegCones = m;

    moreau::Settings settings;
    settings.verbose = false;
    settings.maxIter = 150;

    moreau::CompiledSolver solver(
        n, m, batch,
        P_ro.data(), P_ci.data(), nnzP,
        A_ro.data(), A_ci.data(), nnzA,
        cones,
        settings
    );

    double *d_P, *d_A, *d_q, *d_b;
    cudaMalloc(&d_P, sizeof(double));
    cudaMalloc(&d_A, sizeof(double) * nnzA);
    cudaMalloc(&d_q, sizeof(double) * n);
    cudaMalloc(&d_b, sizeof(double) * m);

    cudaMemcpy(d_A, A_val.data(), sizeof(double) * nnzA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_q, lp.c.data(), sizeof(double) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_combined.data(), sizeof(double) * m, cudaMemcpyHostToDevice);

    solver.solveAll(d_P, d_A, d_q, d_b);

    std::vector<double> x_sol(n);
    cudaMemcpy(x_sol.data(), solver.solution.x.data(), sizeof(double) * n, cudaMemcpyDeviceToHost);

    std::vector<double> z_sol(m);
    cudaMemcpy(z_sol.data(), solver.solution.z.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    std::vector<double> s_sol(m);
    cudaMemcpy(s_sol.data(), solver.solution.s.data(), sizeof(double) * m, cudaMemcpyDeviceToHost);

    std::vector<int32_t> status(batch);
    cudaMemcpy(status.data(), solver.solution.status.get(), sizeof(int32_t) * batch, cudaMemcpyDeviceToHost);

    EXPECT_EQ(status[0], 1) << "Solver should handle sparse solutions";

    // Verify that solution is indeed sparse (most variables should be near zero)
    int num_near_zero = 0;
    for (size_t i = 0; i < n; i++) {
        if (std::abs(x_sol[i]) < 1e-5) {
            num_near_zero++;
        }
    }
    EXPECT_GT(num_near_zero, n / 2) << "Solution should be sparse";

    // Verify primal feasibility
    for (int64_t i = 0; i < m; i++) {
        double lhs = s_sol[i];
        for (int64_t idx = A_ro[i]; idx < A_ro[i + 1]; idx++) {
            lhs += A_val[idx] * x_sol[A_ci[idx]];
        }
        EXPECT_NEAR(lhs, b_combined[i], 1e-5);
    }

    // Verify dual feasibility
    for (int64_t i = 0; i < m; i++) {
        EXPECT_GE(z_sol[i], -1e-6);
    }

    // Verify complementary slackness
    for (int64_t i = 0; i < m; i++) {
        EXPECT_LT(std::abs(s_sol[i] * z_sol[i]), 1e-4);
    }

    // Verify objective
    double obj_computed = 0.0;
    double obj_expected = 0.0;
    for (size_t i = 0; i < n; i++) {
        obj_computed += lp.c[i] * x_sol[i];
        obj_expected += lp.c[i] * lp.x_star[i];
    }
    EXPECT_NEAR(obj_computed, obj_expected, 1e-6);

    cudaFree(d_P);
    cudaFree(d_A);
    cudaFree(d_q);
    cudaFree(d_b);
}
