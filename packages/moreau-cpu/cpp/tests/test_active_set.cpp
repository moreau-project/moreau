/**
 * @file test_active_set.cpp
 * @brief Tests for the CPU active-set QP solver (DAQP algorithm)
 */
#include <gtest/gtest.h>
#include "moreau/solver/active_set/solver.hpp"
#include "moreau/solver/active_set/core.hpp"
#include "moreau/solver/active_set/transform.hpp"
#include "moreau/solver/active_set/factorization.hpp"
#include "moreau/solver/active_set/backward.hpp"
#include "moreau/cones/cones.hpp"
#include "moreau/solver/status.hpp"
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>
#include <random>
#include <numeric>

using namespace moreau;

class ActiveSetTest : public ::testing::Test {
protected:
    static constexpr double TOL = 1e-6;
};

// ============================================================================
// Unit tests for LDL factorization
// ============================================================================

TEST_F(ActiveSetTest, DotProduct) {
    double v1[] = {1.0, 2.0, 3.0};
    double v2[] = {4.0, 5.0, 6.0};
    EXPECT_NEAR(daqp_dot(v1, v2, 3), 32.0, 1e-12);
}

// ============================================================================
// Unit tests for QP→LDP transform
// ============================================================================

TEST_F(ActiveSetTest, CsrToDense) {
    int64_t ro[] = {0, 1, 2};
    int64_t ci[] = {0, 1};
    double vals[] = {1.0, 1.0};
    double dense[4] = {};

    csr_to_dense(dense, 2, 2, ro, ci, vals, false);
    EXPECT_EQ(dense[0], 1.0);
    EXPECT_EQ(dense[1], 0.0);
    EXPECT_EQ(dense[2], 0.0);
    EXPECT_EQ(dense[3], 1.0);
}

TEST_F(ActiveSetTest, CsrToDenseSymmetric) {
    int64_t ro[] = {0, 2, 3};
    int64_t ci[] = {0, 1, 1};
    double vals[] = {2.0, 1.0, 2.0};
    double dense[4] = {};

    csr_to_dense(dense, 2, 2, ro, ci, vals, true);
    EXPECT_EQ(dense[0], 2.0);
    EXPECT_EQ(dense[1], 1.0);
    EXPECT_EQ(dense[2], 1.0);
    EXPECT_EQ(dense[3], 2.0);
}

TEST_F(ActiveSetTest, DenseToCsrValues) {
    // Dense matrix [[2, 1], [1, 2]] → CSR values matching structure
    double dense[] = {2.0, 1.0, 1.0, 2.0};
    int64_t ro[] = {0, 2, 3};
    int64_t ci[] = {0, 1, 1};
    double vals[3] = {};

    dense_to_csr_values(dense, 2, 2, ro, ci, vals, false);
    EXPECT_EQ(vals[0], 2.0);
    EXPECT_EQ(vals[1], 1.0);
    EXPECT_EQ(vals[2], 2.0);
}

// ============================================================================
// End-to-end solver tests
// ============================================================================

TEST_F(ActiveSetTest, SimpleQP) {
    // minimize    0.5*(x1^2 + x2^2) + x1 + x2
    // subject to  x1 + x2 = 1, x1 >= 0, x2 >= 0
    const int64_t n = 2, m = 3, batch = 1;

    int64_t P_ro[] = {0, 1, 2};
    int64_t P_ci[] = {0, 1};
    int64_t A_ro[] = {0, 2, 3, 4};
    int64_t A_ci[] = {0, 1, 0, 1};

    Cones cones;
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 4, cones);

    double P_vals[] = {1.0, 1.0};
    double A_vals[] = {1.0, 1.0, -1.0, -1.0};
    solver.setup(P_vals, A_vals);

    double q[] = {1.0, 1.0};
    double b[] = {1.0, 0.0, 0.0};
    solver.solve(q, b);

    EXPECT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(solver.x_sol[0], 0.5, TOL);
    EXPECT_NEAR(solver.x_sol[1], 0.5, TOL);
    EXPECT_NEAR(solver.obj_val[0], 1.25, TOL);
    EXPECT_NEAR(solver.s_sol[0], 0.0, TOL);
    EXPECT_NEAR(solver.s_sol[1], 0.5, TOL);
    EXPECT_NEAR(solver.s_sol[2], 0.5, TOL);
}

TEST_F(ActiveSetTest, UnconstrainedQP) {
    const int64_t n = 2, m = 2, batch = 1;

    int64_t P_ro[] = {0, 1, 2};
    int64_t P_ci[] = {0, 1};
    int64_t A_ro[] = {0, 1, 2};
    int64_t A_ci[] = {0, 1};

    Cones cones;
    cones.numZeroCones = 0;
    cones.numNonnegCones = 2;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 2, cones);

    double P_vals[] = {1.0, 1.0};
    double A_vals[] = {1.0, 1.0};
    solver.setup(P_vals, A_vals);

    double q[] = {-1.0, -1.0};
    double b[] = {10.0, 10.0};
    solver.solve(q, b);

    EXPECT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(solver.x_sol[0], 1.0, TOL);
    EXPECT_NEAR(solver.x_sol[1], 1.0, TOL);
}

TEST_F(ActiveSetTest, ActiveBound) {
    const int64_t n = 1, m = 1, batch = 1;

    int64_t P_ro[] = {0, 1};
    int64_t P_ci[] = {0};
    int64_t A_ro[] = {0, 1};
    int64_t A_ci[] = {0};

    Cones cones;
    cones.numZeroCones = 0;
    cones.numNonnegCones = 1;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 1, A_ro, A_ci, 1, cones);

    double P_vals[] = {1.0};
    double A_vals[] = {1.0};
    solver.setup(P_vals, A_vals);

    double q[] = {-2.0};
    double b[] = {1.0};
    solver.solve(q, b);

    EXPECT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(solver.x_sol[0], 1.0, TOL);
    EXPECT_NEAR(solver.obj_val[0], 0.5 - 2.0, TOL);
}

TEST_F(ActiveSetTest, EqualityOnly) {
    const int64_t n = 2, m = 1, batch = 1;

    int64_t P_ro[] = {0, 1, 2};
    int64_t P_ci[] = {0, 1};
    int64_t A_ro[] = {0, 2};
    int64_t A_ci[] = {0, 1};

    Cones cones;
    cones.numZeroCones = 1;
    cones.numNonnegCones = 0;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 2, cones);

    double P_vals[] = {1.0, 1.0};
    double A_vals[] = {1.0, 1.0};
    solver.setup(P_vals, A_vals);

    double q[] = {0.0, 0.0};
    double b[] = {1.0};
    solver.solve(q, b);

    EXPECT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(solver.x_sol[0], 0.5, TOL);
    EXPECT_NEAR(solver.x_sol[1], 0.5, TOL);
}

TEST_F(ActiveSetTest, BatchSolve) {
    const int64_t n = 1, m = 1, batch = 3;

    int64_t P_ro[] = {0, 1};
    int64_t P_ci[] = {0};
    int64_t A_ro[] = {0, 1};
    int64_t A_ci[] = {0};

    Cones cones;
    cones.numZeroCones = 0;
    cones.numNonnegCones = 1;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 1, A_ro, A_ci, 1, cones);

    double P_vals[] = {1.0};
    double A_vals[] = {1.0};
    solver.setup(P_vals, A_vals);

    double q[] = {0.0, 0.0, 0.0};
    double b[] = {5.0, -1.0, 0.0};
    solver.solve(q, b);

    for (int i = 0; i < 3; i++) {
        EXPECT_EQ(solver.status_vec[i], static_cast<int32_t>(SolverStatus::Solved))
            << "Batch " << i << " not solved";
    }

    EXPECT_NEAR(solver.x_sol[0], 0.0, TOL);
    EXPECT_NEAR(solver.x_sol[1], -1.0, TOL);
    EXPECT_NEAR(solver.x_sol[2], 0.0, TOL);
}

TEST_F(ActiveSetTest, RejectSOC) {
    const int64_t n = 2, m = 3, batch = 1;
    int64_t P_ro[] = {0, 1, 2};
    int64_t P_ci[] = {0, 1};
    int64_t A_ro[] = {0, 2, 3, 4};
    int64_t A_ci[] = {0, 1, 0, 1};

    Cones cones;
    cones.numZeroCones = 0;
    cones.numNonnegCones = 0;
    cones.socConeDims = {3};

    EXPECT_THROW(
        ActiveSetSolver(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 4, cones),
        std::invalid_argument
    );
}

TEST_F(ActiveSetTest, DenseHessian) {
    const int64_t n = 2, m = 1, batch = 1;

    int64_t P_ro[] = {0, 2, 3};
    int64_t P_ci[] = {0, 1, 1};
    int64_t A_ro[] = {0, 2};
    int64_t A_ci[] = {0, 1};

    Cones cones;
    cones.numZeroCones = 0;
    cones.numNonnegCones = 1;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 3, A_ro, A_ci, 2, cones);

    double P_vals[] = {2.0, 1.0, 2.0};
    double A_vals[] = {1.0, 1.0};
    solver.setup(P_vals, A_vals);

    double q[] = {-1.0, -1.0};
    double b[] = {1.0};
    solver.solve(q, b);

    EXPECT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(solver.x_sol[0], 1.0/3.0, TOL);
    EXPECT_NEAR(solver.x_sol[1], 1.0/3.0, TOL);
}

// ============================================================================
// Backward pass tests
// ============================================================================

// Helper: create solver and solve for the ActiveBound problem (reused in backward tests)
struct ActiveBoundProblem {
    static constexpr int64_t n = 1, m = 1, batch = 1;
    int64_t P_ro[2] = {0, 1};
    int64_t P_ci[1] = {0};
    int64_t A_ro[2] = {0, 1};
    int64_t A_ci[1] = {0};
    double P_vals[1] = {1.0};
    double A_vals[1] = {1.0};
    double q[1] = {-2.0};
    double b[1] = {1.0};
};

TEST_F(ActiveSetTest, BackwardActiveBoundDq) {
    // min 0.5*x^2 - 2*x s.t. x <= 1 → x* = 1
    // Perturb q: if q changes by dq, how does x* change?
    // At active bound: x* = 1 regardless of q (bound is active),
    // so dx*/dq = 0. But the objective changes: d(obj)/dq = x* = 1
    //
    // Let's use upstream dx_bar = [1] and verify dq via finite differences.

    ActiveBoundProblem p;
    Cones cones;
    cones.numZeroCones = 0;
    cones.numNonnegCones = 1;

    ActiveSetSolver solver(p.n, p.m, p.batch,
        p.P_ro, p.P_ci, 1, p.A_ro, p.A_ci, 1, cones,
        ActiveSetSettings{}, /*enable_grad=*/true);

    solver.setup(p.P_vals, p.A_vals);
    solver.solve(p.q, p.b);

    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(solver.x_sol[0], 1.0, TOL);

    // Backward with dx_bar=[1], dz_bar=[0], ds_bar=[0]
    double dx_bar[] = {1.0};
    double dz_bar[] = {0.0};
    double ds_bar[] = {0.0};
    solver.backward(dx_bar, dz_bar, ds_bar);

    // Finite-difference verification for dq
    double eps = 1e-6;
    double q_pert[] = {p.q[0] + eps};

    ActiveSetSolver solver2(p.n, p.m, p.batch,
        p.P_ro, p.P_ci, 1, p.A_ro, p.A_ci, 1, cones);
    solver2.setup(p.P_vals, p.A_vals);
    solver2.solve(q_pert, p.b);

    double fd_dq = (solver2.x_sol[0] - solver.x_sol[0]) / eps;
    EXPECT_NEAR(solver.dq[0], fd_dq, 1e-4)
        << "dq: analytic=" << solver.dq[0] << " fd=" << fd_dq;
}

TEST_F(ActiveSetTest, BackwardActiveBoundDb) {
    // Same problem, but check db gradient
    ActiveBoundProblem p;
    Cones cones;
    cones.numZeroCones = 0;
    cones.numNonnegCones = 1;

    ActiveSetSolver solver(p.n, p.m, p.batch,
        p.P_ro, p.P_ci, 1, p.A_ro, p.A_ci, 1, cones,
        ActiveSetSettings{}, true);

    solver.setup(p.P_vals, p.A_vals);
    solver.solve(p.q, p.b);

    double dx_bar[] = {1.0};
    double dz_bar[] = {0.0};
    double ds_bar[] = {0.0};
    solver.backward(dx_bar, dz_bar, ds_bar);

    // FD for db
    double eps = 1e-6;
    double b_pert[] = {p.b[0] + eps};

    ActiveSetSolver solver2(p.n, p.m, p.batch,
        p.P_ro, p.P_ci, 1, p.A_ro, p.A_ci, 1, cones);
    solver2.setup(p.P_vals, p.A_vals);
    solver2.solve(p.q, b_pert);

    double fd_db = (solver2.x_sol[0] - solver.x_sol[0]) / eps;
    EXPECT_NEAR(solver.db[0], fd_db, 1e-4)
        << "db: analytic=" << solver.db[0] << " fd=" << fd_db;
}

TEST_F(ActiveSetTest, BackwardEqualityConstraint) {
    // min 0.5*(x1^2 + x2^2) s.t. x1 + x2 = 1
    // x* = [0.5, 0.5]
    const int64_t n = 2, m = 1, batch = 1;

    int64_t P_ro[] = {0, 1, 2};
    int64_t P_ci[] = {0, 1};
    int64_t A_ro[] = {0, 2};
    int64_t A_ci[] = {0, 1};

    Cones cones;
    cones.numZeroCones = 1;
    cones.numNonnegCones = 0;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 2, cones,
                           ActiveSetSettings{}, true);

    double P_vals[] = {1.0, 1.0};
    double A_vals[] = {1.0, 1.0};
    solver.setup(P_vals, A_vals);

    double q[] = {0.0, 0.0};
    double b[] = {1.0};
    solver.solve(q, b);

    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    // Backward with dx_bar = [1, 0]
    double dx_bar[] = {1.0, 0.0};
    double dz_bar[] = {0.0};
    double ds_bar[] = {0.0};
    solver.backward(dx_bar, dz_bar, ds_bar);

    // FD verification for dq
    double eps = 1e-6;
    for (int dim = 0; dim < 2; dim++) {
        double q_pert[2] = {q[0], q[1]};
        q_pert[dim] += eps;

        ActiveSetSolver solver2(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 2, cones);
        solver2.setup(P_vals, A_vals);
        solver2.solve(q_pert, b);

        // dx*/dq[dim] ≈ (x_pert - x) / eps → dq should match
        double fd = (solver2.x_sol[0] - solver.x_sol[0]) / eps;  // d(x1)/d(q[dim])
        EXPECT_NEAR(solver.dq[dim], fd, 1e-4)
            << "dq[" << dim << "]: analytic=" << solver.dq[dim] << " fd=" << fd;
    }

    // FD verification for db
    {
        double b_pert[] = {b[0] + eps};

        ActiveSetSolver solver2(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 2, cones);
        solver2.setup(P_vals, A_vals);
        solver2.solve(q, b_pert);

        double fd = (solver2.x_sol[0] - solver.x_sol[0]) / eps;
        EXPECT_NEAR(solver.db[0], fd, 1e-4)
            << "db: analytic=" << solver.db[0] << " fd=" << fd;
    }
}

TEST_F(ActiveSetTest, BackwardUnconstrainedQP) {
    // min 0.5*(x1^2 + x2^2) - x1 - x2 s.t. x1 <= 10, x2 <= 10
    // x* = [1, 1] (unconstrained optimum)
    const int64_t n = 2, m = 2, batch = 1;

    int64_t P_ro[] = {0, 1, 2};
    int64_t P_ci[] = {0, 1};
    int64_t A_ro[] = {0, 1, 2};
    int64_t A_ci[] = {0, 1};

    Cones cones;
    cones.numZeroCones = 0;
    cones.numNonnegCones = 2;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 2, cones,
                           ActiveSetSettings{}, true);

    double P_vals[] = {1.0, 1.0};
    double A_vals[] = {1.0, 1.0};
    solver.setup(P_vals, A_vals);

    double q[] = {-1.0, -1.0};
    double b[] = {10.0, 10.0};
    solver.solve(q, b);

    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(solver.x_sol[0], 1.0, TOL);

    // Backward with dx_bar = [1, 0]
    double dx_bar[] = {1.0, 0.0};
    double dz_bar[] = {0.0, 0.0};
    double ds_bar[] = {0.0, 0.0};
    solver.backward(dx_bar, dz_bar, ds_bar);

    // Unconstrained: x* = -H^{-1} q, so dx*/dq = -H^{-1}
    // With dx_bar = [1,0]: dq = -H^{-1}[0,:] = [-1, 0] (since H = I)
    EXPECT_NEAR(solver.dq[0], -1.0, 1e-4);
    EXPECT_NEAR(solver.dq[1], 0.0, 1e-4);

    // FD verification for dq
    double eps = 1e-6;
    for (int dim = 0; dim < 2; dim++) {
        double q_pert[2] = {q[0], q[1]};
        q_pert[dim] += eps;

        ActiveSetSolver solver2(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 2, cones);
        solver2.setup(P_vals, A_vals);
        solver2.solve(q_pert, b);

        double fd = (solver2.x_sol[0] - solver.x_sol[0]) / eps;
        EXPECT_NEAR(solver.dq[dim], fd, 1e-4)
            << "dq[" << dim << "]: analytic=" << solver.dq[dim] << " fd=" << fd;
    }
}

TEST_F(ActiveSetTest, BackwardDenseHessianFD) {
    // min 0.5 * x' [[2,1],[1,2]] x + [-1,-1]'x s.t. x1+x2 <= 1
    // x* = [1/3, 1/3] (unconstrained, constraint inactive)
    //
    // Use full symmetric CSR: perturbing entry (i,j) for i≠j must also
    // perturb the symmetric counterpart (j,i) to keep H symmetric.
    const int64_t n = 2, m = 1, batch = 1;

    int64_t P_ro[] = {0, 2, 4};
    int64_t P_ci[] = {0, 1, 0, 1};
    int64_t A_ro[] = {0, 2};
    int64_t A_ci[] = {0, 1};

    Cones cones;
    cones.numZeroCones = 0;
    cones.numNonnegCones = 1;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 4, A_ro, A_ci, 2, cones,
                           ActiveSetSettings{}, true);

    double P_vals[] = {2.0, 1.0, 1.0, 2.0};
    double A_vals[] = {1.0, 1.0};
    solver.setup(P_vals, A_vals);

    double q[] = {-1.0, -1.0};
    double b[] = {1.0};
    solver.solve(q, b);

    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    // Backward with dx_bar = [1, 0]
    double dx_bar[] = {1.0, 0.0};
    double dz_bar[] = {0.0};
    double ds_bar[] = {0.0};
    solver.backward(dx_bar, dz_bar, ds_bar);

    // FD for dP_values: perturb symmetric pairs together
    // CSR layout: [0]=(0,0), [1]=(0,1), [2]=(1,0), [3]=(1,1)
    double eps = 1e-6;

    // Test diagonal entries (no symmetric counterpart)
    for (int k : {0, 3}) {
        double P_pert[4] = {P_vals[0], P_vals[1], P_vals[2], P_vals[3]};
        P_pert[k] += eps;

        ActiveSetSolver solver2(n, m, batch, P_ro, P_ci, 4, A_ro, A_ci, 2, cones);
        solver2.setup(P_pert, A_vals);
        solver2.solve(q, b);

        double fd = (solver2.x_sol[0] - solver.x_sol[0]) / eps;
        EXPECT_NEAR(solver.dP_values[k], fd, 1e-3)
            << "dP[" << k << "]: analytic=" << solver.dP_values[k] << " fd=" << fd;
    }

    // Test off-diagonal: perturb both (0,1) and (1,0) together.
    // Since P is full symmetric, each CSR entry dP[k] already contains
    // the symmetrized gradient dH[i,j]+dH[j,i]. Both entries should
    // individually match the FD when perturbing both (i,j) and (j,i).
    {
        double P_pert[4] = {P_vals[0], P_vals[1] + eps, P_vals[2] + eps, P_vals[3]};

        ActiveSetSolver solver2(n, m, batch, P_ro, P_ci, 4, A_ro, A_ci, 2, cones);
        solver2.setup(P_pert, A_vals);
        solver2.solve(q, b);

        double fd = (solver2.x_sol[0] - solver.x_sol[0]) / eps;
        EXPECT_NEAR(solver.dP_values[1], fd, 1e-3)
            << "dP[1] off-diag: analytic=" << solver.dP_values[1] << " fd=" << fd;
        EXPECT_NEAR(solver.dP_values[2], fd, 1e-3)
            << "dP[2] off-diag: analytic=" << solver.dP_values[2] << " fd=" << fd;
        // Symmetry: dP[0,1] == dP[1,0]
        EXPECT_NEAR(solver.dP_values[1], solver.dP_values[2], 1e-12);
    }
}

TEST_F(ActiveSetTest, BackwardZeroUpstream) {
    // Zero upstream gradients should produce zero parameter gradients
    const int64_t n = 2, m = 1, batch = 1;

    int64_t P_ro[] = {0, 1, 2};
    int64_t P_ci[] = {0, 1};
    int64_t A_ro[] = {0, 2};
    int64_t A_ci[] = {0, 1};

    Cones cones;
    cones.numZeroCones = 1;
    cones.numNonnegCones = 0;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 2, cones,
                           ActiveSetSettings{}, true);

    double P_vals[] = {1.0, 1.0};
    double A_vals[] = {1.0, 1.0};
    solver.setup(P_vals, A_vals);

    double q[] = {0.0, 0.0};
    double b[] = {1.0};
    solver.solve(q, b);

    double dx_bar[] = {0.0, 0.0};
    double dz_bar[] = {0.0};
    double ds_bar[] = {0.0};
    solver.backward(dx_bar, dz_bar, ds_bar);

    for (int i = 0; i < 2; i++) {
        EXPECT_NEAR(solver.dq[i], 0.0, 1e-12) << "dq[" << i << "] should be zero";
    }
    for (int i = 0; i < 2; i++) {
        EXPECT_NEAR(solver.dP_values[i], 0.0, 1e-12) << "dP[" << i << "] should be zero";
    }
}

TEST_F(ActiveSetTest, BackwardDhGradientConsistency) {
    // For full symmetric CSR, the off-diagonal entries (i,j) and (j,i) are
    // independent CSR entries. The gradient dP[k] = dH[row][col] where
    // dH[i][j] = -v1[i]*x[j]. These are generally NOT equal for (i,j) vs (j,i),
    // but their sum dP[i,j]+dP[j,i] should match the FD when perturbing both.
    const int64_t n = 2, m = 1, batch = 1;

    int64_t P_ro[] = {0, 2, 4};
    int64_t P_ci[] = {0, 1, 0, 1};
    int64_t A_ro[] = {0, 2};
    int64_t A_ci[] = {0, 1};

    Cones cones;
    cones.numZeroCones = 1;
    cones.numNonnegCones = 0;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 4, A_ro, A_ci, 2, cones,
                           ActiveSetSettings{}, true);

    double P_vals[] = {2.0, 1.0, 1.0, 2.0};
    double A_vals[] = {1.0, 1.0};
    solver.setup(P_vals, A_vals);

    double q[] = {0.5, -0.5};
    double b[] = {1.0};
    solver.solve(q, b);

    double dx_bar[] = {1.0, 0.5};
    double dz_bar[] = {0.0};
    double ds_bar[] = {0.0};
    solver.backward(dx_bar, dz_bar, ds_bar);

    // Verify dP[0]+dP[3] (diagonal sum) via FD
    double eps = 1e-6;
    double P_pert[4] = {2.0 + eps, 1.0, 1.0, 2.0 + eps};
    ActiveSetSolver solver2(n, m, batch, P_ro, P_ci, 4, A_ro, A_ci, 2, cones);
    solver2.setup(P_pert, A_vals);
    solver2.solve(q, b);

    double fd = 0;
    for (int i = 0; i < n; i++) fd += dx_bar[i] * (solver2.x_sol[i] - solver.x_sol[i]) / eps;
    double analytic = solver.dP_values[0] + solver.dP_values[3];
    EXPECT_NEAR(analytic, fd, 1e-4)
        << "diagonal dP sum: analytic=" << analytic << " fd=" << fd;
}

TEST_F(ActiveSetTest, BackwardBatchConsistency) {
    // Verify batched backward produces same results as individual calls
    const int64_t n = 1, m = 1;

    int64_t P_ro[] = {0, 1};
    int64_t P_ci[] = {0};
    int64_t A_ro[] = {0, 1};
    int64_t A_ci[] = {0};

    Cones cones;
    cones.numZeroCones = 0;
    cones.numNonnegCones = 1;

    // Batch of 3
    ActiveSetSolver solver_batch(n, m, 3, P_ro, P_ci, 1, A_ro, A_ci, 1, cones,
                                  ActiveSetSettings{}, true);
    double P_vals[] = {1.0};
    double A_vals[] = {1.0};
    solver_batch.setup(P_vals, A_vals);

    double q_batch[] = {-2.0, -0.5, -3.0};
    double b_batch[] = {1.0, 1.0, 2.0};
    solver_batch.solve(q_batch, b_batch);

    double dx_batch[] = {1.0, 1.0, 1.0};
    double dz_batch[] = {0.0, 0.0, 0.0};
    double ds_batch[] = {0.0, 0.0, 0.0};
    solver_batch.backward(dx_batch, dz_batch, ds_batch);

    // Solve individually and compare
    for (int i = 0; i < 3; i++) {
        ActiveSetSolver solver_single(n, m, 1, P_ro, P_ci, 1, A_ro, A_ci, 1, cones,
                                       ActiveSetSettings{}, true);
        solver_single.setup(P_vals, A_vals);
        solver_single.solve(q_batch + i, b_batch + i);

        double dx_single[] = {1.0};
        double dz_single[] = {0.0};
        double ds_single[] = {0.0};
        solver_single.backward(dx_single, dz_single, ds_single);

        EXPECT_NEAR(solver_batch.dq[i], solver_single.dq[0], 1e-12)
            << "Batch " << i << " dq mismatch";
        EXPECT_NEAR(solver_batch.db[i], solver_single.db[0], 1e-12)
            << "Batch " << i << " db mismatch";
    }
}

// ============================================================================
// Warm starting tests
// ============================================================================

TEST_F(ActiveSetTest, WarmStartBasic) {
    // Solve, then warm-start with slightly perturbed RHS
    const int64_t n = 2, m = 3, batch = 1;

    int64_t P_ro[] = {0, 1, 2};
    int64_t P_ci[] = {0, 1};
    int64_t A_ro[] = {0, 2, 3, 4};
    int64_t A_ci[] = {0, 1, 0, 1};

    Cones cones;
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 4, cones);

    double P_vals[] = {1.0, 1.0};
    double A_vals[] = {1.0, 1.0, -1.0, -1.0};
    solver.setup(P_vals, A_vals);

    // First solve
    double q[] = {1.0, 1.0};
    double b[] = {1.0, 0.0, 0.0};
    solver.solve(q, b);
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    int cold_iters = solver.iters[0];

    // Slightly perturbed b
    double b2[] = {1.01, 0.0, 0.0};

    // Cold solve for comparison
    ActiveSetSolver solver_cold(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 4, cones);
    solver_cold.setup(P_vals, A_vals);
    solver_cold.solve(q, b2);
    int cold_iters2 = solver_cold.iters[0];

    // Warm-start solve
    ActiveSetSolver solver_warm(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 4, cones);
    solver_warm.setup(P_vals, A_vals);
    solver_warm.solve_warm_start(q, b2,
        solver.x_sol.data(), solver.z_sol.data(), solver.s_sol.data());

    EXPECT_EQ(solver_warm.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    // Solution should be close to perturbed solution
    EXPECT_NEAR(solver_warm.x_sol[0], solver_cold.x_sol[0], TOL);
    EXPECT_NEAR(solver_warm.x_sol[1], solver_cold.x_sol[1], TOL);

    // Warm start should use fewer or equal iterations
    std::cout << "  Warm start: " << solver_warm.iters[0]
              << " iters vs cold: " << cold_iters2 << " iters" << std::endl;
    // Don't strictly require fewer — the active set may differ
}

TEST_F(ActiveSetTest, WarmStartSameProblem) {
    // Warm-start with exact previous solution → should converge in very few iterations
    const int64_t n = 1, m = 1, batch = 1;

    int64_t P_ro[] = {0, 1};
    int64_t P_ci[] = {0};
    int64_t A_ro[] = {0, 1};
    int64_t A_ci[] = {0};

    Cones cones;
    cones.numZeroCones = 0;
    cones.numNonnegCones = 1;

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 1, A_ro, A_ci, 1, cones);

    double P_vals[] = {1.0};
    double A_vals[] = {1.0};
    solver.setup(P_vals, A_vals);

    double q[] = {-2.0};
    double b[] = {1.0};
    solver.solve(q, b);
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    // Warm-start with same problem
    ActiveSetSolver solver2(n, m, batch, P_ro, P_ci, 1, A_ro, A_ci, 1, cones);
    solver2.setup(P_vals, A_vals);
    solver2.solve_warm_start(q, b,
        solver.x_sol.data(), solver.z_sol.data(), solver.s_sol.data());

    EXPECT_EQ(solver2.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(solver2.x_sol[0], 1.0, TOL);

    // Should converge very quickly (0-2 iterations)
    std::cout << "  Warm start same problem: " << solver2.iters[0] << " iterations" << std::endl;
    EXPECT_LE(solver2.iters[0], 3);
}

// ============================================================================
// Stress tests
// ============================================================================

TEST_F(ActiveSetTest, RandomQP_n10_m20) {
    // Random QP with n=10 variables, m=20 constraints
    const int64_t n = 10, m = 20, batch = 1;
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Build full P = L*L' + I (guaranteed SPD)
    std::vector<double> L(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            L[i * n + j] = dist(rng);
        }
    }
    std::vector<double> P_dense(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = (i == j) ? 1.0 : 0.0;
            for (int k = 0; k <= std::min(i, j); k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            P_dense[i * n + j] = sum;
        }
    }

    // Build full A
    std::vector<double> A_dense(m * n);
    for (int i = 0; i < m * n; i++) A_dense[i] = dist(rng);

    // CSR: full dense matrices
    std::vector<int64_t> P_ro(n + 1), P_ci(n * n);
    std::vector<double> P_vals(n * n);
    int pk = 0;
    for (int i = 0; i < n; i++) {
        P_ro[i] = pk;
        for (int j = 0; j < n; j++) {
            P_ci[pk] = j;
            P_vals[pk] = P_dense[i * n + j];
            pk++;
        }
    }
    P_ro[n] = pk;

    std::vector<int64_t> A_ro(m + 1), A_ci(m * n);
    std::vector<double> A_vals(m * n);
    int ak = 0;
    for (int i = 0; i < m; i++) {
        A_ro[i] = ak;
        for (int j = 0; j < n; j++) {
            A_ci[ak] = j;
            A_vals[ak] = A_dense[i * n + j];
            ak++;
        }
    }
    A_ro[m] = ak;

    // 5 equalities, 15 inequalities
    Cones cones;
    cones.numZeroCones = 5;
    cones.numNonnegCones = 15;

    ActiveSetSolver solver(n, m, batch,
        P_ro.data(), P_ci.data(), pk,
        A_ro.data(), A_ci.data(), ak,
        cones, ActiveSetSettings{}, true);

    solver.setup(P_vals.data(), A_vals.data());

    std::vector<double> q(n), b(m);
    for (int i = 0; i < n; i++) q[i] = dist(rng);
    for (int i = 0; i < m; i++) b[i] = 2.0 + std::abs(dist(rng));  // feasible
    solver.solve(q.data(), b.data());

    EXPECT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved))
        << "Random QP should be solved";

    std::cout << "  Random QP n=" << n << " m=" << m
              << " iters=" << solver.iters[0] << std::endl;

    // Backward pass test: FD for dq
    double dx_bar_vec[10];
    for (int i = 0; i < n; i++) dx_bar_vec[i] = dist(rng);
    std::vector<double> dz_bar_vec(m, 0.0);
    std::vector<double> ds_bar_vec(m, 0.0);

    solver.backward(dx_bar_vec, dz_bar_vec.data(), ds_bar_vec.data());

    // Check dq with FD
    double eps = 1e-6;
    double max_fd_err = 0.0;
    for (int dim = 0; dim < n; dim++) {
        std::vector<double> q_pert = q;
        q_pert[dim] += eps;

        ActiveSetSolver solver2(n, m, batch,
            P_ro.data(), P_ci.data(), pk,
            A_ro.data(), A_ci.data(), ak,
            cones);
        solver2.setup(P_vals.data(), A_vals.data());
        solver2.solve(q_pert.data(), b.data());

        // directional derivative: sum_i dx_bar[i] * d(x_i)/d(q_dim)
        double fd = 0.0;
        for (int i = 0; i < n; i++) {
            fd += dx_bar_vec[i] * (solver2.x_sol[i] - solver.x_sol[i]) / eps;
        }
        double err = std::abs(solver.dq[dim] - fd);
        max_fd_err = std::max(max_fd_err, err);
    }
    std::cout << "  Backward max FD error (dq): " << max_fd_err << std::endl;
    EXPECT_LT(max_fd_err, 1e-3) << "Backward dq FD check failed";
}

// ============================================================================
// Helper: build a random QP with dense CSR structure
// ============================================================================

struct RandomQP {
    int64_t n, m;
    std::vector<double> P_dense, A_dense, P_vals, A_vals, q, b;
    std::vector<int64_t> P_ro, P_ci, A_ro, A_ci;
    int64_t nnzP, nnzA;
    Cones cones;

    RandomQP(int64_t n_, int64_t m_, int64_t numZero, int64_t numNonneg, int seed = 42)
        : n(n_), m(m_) {
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);

        // P = L*L' + I (SPD)
        std::vector<double> L(n * n, 0.0);
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j <= i; j++)
                L[i * n + j] = dist(rng);
        P_dense.resize(n * n, 0.0);
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n; j++) {
                double sum = (i == j) ? 1.0 : 0.0;
                for (int64_t k = 0; k <= std::min(i, j); k++)
                    sum += L[i * n + k] * L[j * n + k];
                P_dense[i * n + j] = sum;
            }

        // A random dense
        A_dense.resize(m * n);
        for (int64_t i = 0; i < m * n; i++) A_dense[i] = dist(rng);

        // Full CSR for P
        P_ro.resize(n + 1);
        P_ci.resize(n * n);
        P_vals.resize(n * n);
        int64_t pk = 0;
        for (int64_t i = 0; i < n; i++) {
            P_ro[i] = pk;
            for (int64_t j = 0; j < n; j++) {
                P_ci[pk] = j;
                P_vals[pk] = P_dense[i * n + j];
                pk++;
            }
        }
        P_ro[n] = pk;
        nnzP = pk;

        // Full CSR for A
        A_ro.resize(m + 1);
        A_ci.resize(m * n);
        A_vals.resize(m * n);
        int64_t ak = 0;
        for (int64_t i = 0; i < m; i++) {
            A_ro[i] = ak;
            for (int64_t j = 0; j < n; j++) {
                A_ci[ak] = j;
                A_vals[ak] = A_dense[i * n + j];
                ak++;
            }
        }
        A_ro[m] = ak;
        nnzA = ak;

        // q, b
        q.resize(n);
        b.resize(m);
        for (int64_t i = 0; i < n; i++) q[i] = dist(rng);
        for (int64_t i = 0; i < m; i++) b[i] = 2.0 + std::abs(dist(rng));

        cones.numZeroCones = numZero;
        cones.numNonnegCones = numNonneg;
    }
};

// ============================================================================
// Exact backward: dA FD test
// ============================================================================

TEST_F(ActiveSetTest, BackwardExactFD_dA) {
    RandomQP qp(10, 20, 5, 15, /*seed=*/42);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    ActiveSetSolver solver(qp.n, qp.m, 1,
        qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
        qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
        qp.cones, ActiveSetSettings{}, true);
    solver.setup(qp.P_vals.data(), qp.A_vals.data());
    solver.solve(qp.q.data(), qp.b.data());
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    std::vector<double> dx_bar(qp.n);
    for (int64_t i = 0; i < qp.n; i++) dx_bar[i] = dist(rng);
    std::vector<double> dz_bar(qp.m, 0.0), ds_bar(qp.m, 0.0);
    solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

    double eps = 1e-6;
    double max_err = 0.0;
    // Check a subset of A entries (every 5th to keep test fast)
    for (int64_t idx = 0; idx < qp.nnzA; idx += 5) {
        std::vector<double> A_pert = qp.A_vals;
        A_pert[idx] += eps;

        ActiveSetSolver solver2(qp.n, qp.m, 1,
            qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
            qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
            qp.cones);
        solver2.setup(qp.P_vals.data(), A_pert.data());
        solver2.solve(qp.q.data(), qp.b.data());

        double fd = 0.0;
        for (int64_t i = 0; i < qp.n; i++)
            fd += dx_bar[i] * (solver2.x_sol[i] - solver.x_sol[i]) / eps;
        double err = std::abs(solver.dA_values[idx] - fd);
        max_err = std::max(max_err, err);
    }
    std::cout << "  Backward exact max FD error (dA): " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-3) << "Backward dA FD check failed";
}

// ============================================================================
// Exact backward: dP per-entry FD test (verifies symmetric gradient fix)
// ============================================================================

TEST_F(ActiveSetTest, BackwardExactFD_dP) {
    RandomQP qp(10, 20, 5, 15, /*seed=*/42);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    ActiveSetSolver solver(qp.n, qp.m, 1,
        qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
        qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
        qp.cones, ActiveSetSettings{}, true);
    solver.setup(qp.P_vals.data(), qp.A_vals.data());
    solver.solve(qp.q.data(), qp.b.data());
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    std::vector<double> dx_bar(qp.n);
    for (int64_t i = 0; i < qp.n; i++) dx_bar[i] = dist(rng);
    std::vector<double> dz_bar(qp.m, 0.0), ds_bar(qp.m, 0.0);
    solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

    double eps = 1e-6;
    double max_err = 0.0;
    int64_t n = qp.n;

    // Check off-diagonal entries: perturb P[i,j] and P[j,i] together
    // Since P is full symmetric, each CSR entry should have the full gradient
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = i + 1; j < n && j < i + 3; j++) {  // check a few per row
            // Find CSR indices for (i,j) and (j,i)
            int64_t k_ij = -1, k_ji = -1;
            for (int64_t k = qp.P_ro[i]; k < qp.P_ro[i + 1]; k++)
                if (qp.P_ci[k] == j) { k_ij = k; break; }
            for (int64_t k = qp.P_ro[j]; k < qp.P_ro[j + 1]; k++)
                if (qp.P_ci[k] == i) { k_ji = k; break; }
            ASSERT_GE(k_ij, 0);
            ASSERT_GE(k_ji, 0);

            // Perturb both (i,j) and (j,i) by eps
            std::vector<double> P_pert = qp.P_vals;
            P_pert[k_ij] += eps;
            P_pert[k_ji] += eps;

            ActiveSetSolver solver2(qp.n, qp.m, 1,
                qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
                qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
                qp.cones);
            solver2.setup(P_pert.data(), qp.A_vals.data());
            solver2.solve(qp.q.data(), qp.b.data());

            double fd = 0.0;
            for (int64_t d = 0; d < n; d++)
                fd += dx_bar[d] * (solver2.x_sol[d] - solver.x_sol[d]) / eps;

            // Both dP[k_ij] and dP[k_ji] should equal the full directional deriv
            // (since we perturbed both by eps, the FD = dP[k_ij] = dP[k_ji])
            double err_ij = std::abs(solver.dP_values[k_ij] - fd);
            double err_ji = std::abs(solver.dP_values[k_ji] - fd);
            max_err = std::max(max_err, std::max(err_ij, err_ji));

            // Also verify symmetry: dP[i,j] == dP[j,i]
            EXPECT_NEAR(solver.dP_values[k_ij], solver.dP_values[k_ji], 1e-12)
                << "dP symmetry violated at (" << i << "," << j << ")";
        }
    }
    std::cout << "  Backward exact max FD error (dP off-diag): " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-3) << "Backward dP per-entry FD check failed";
}

// ============================================================================
// Smoothed backward: FD tests
// ============================================================================

TEST_F(ActiveSetTest, SmoothedBackwardFD_dq) {
    RandomQP qp(10, 20, 5, 15, /*seed=*/42);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    ActiveSetSettings as_settings;
    as_settings.diff_method = ActiveSetDiffMethod::Smoothed;
    as_settings.diff_smoothing_mu = 1e-6;

    ActiveSetSolver solver(qp.n, qp.m, 1,
        qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
        qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
        qp.cones, as_settings, true);
    solver.setup(qp.P_vals.data(), qp.A_vals.data());
    solver.solve(qp.q.data(), qp.b.data());
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    std::vector<double> dx_bar(qp.n);
    for (int64_t i = 0; i < qp.n; i++) dx_bar[i] = dist(rng);
    std::vector<double> dz_bar(qp.m, 0.0), ds_bar(qp.m, 0.0);
    solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

    // FD for dq: smoothed backward should approximate the smoothed map's Jacobian,
    // but FD probes the exact (non-smoothed) forward map. With small mu, the two
    // should be close.
    double eps = 1e-6;
    double max_err = 0.0;
    for (int64_t dim = 0; dim < qp.n; dim++) {
        std::vector<double> q_pert = qp.q;
        q_pert[dim] += eps;

        ActiveSetSolver solver2(qp.n, qp.m, 1,
            qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
            qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
            qp.cones);
        solver2.setup(qp.P_vals.data(), qp.A_vals.data());
        solver2.solve(q_pert.data(), qp.b.data());

        double fd = 0.0;
        for (int64_t i = 0; i < qp.n; i++)
            fd += dx_bar[i] * (solver2.x_sol[i] - solver.x_sol[i]) / eps;
        double err = std::abs(solver.dq[dim] - fd);
        max_err = std::max(max_err, err);
    }
    std::cout << "  Smoothed backward max FD error (dq): " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-3) << "Smoothed backward dq FD check failed";
}

TEST_F(ActiveSetTest, SmoothedBackwardFD_db) {
    RandomQP qp(10, 20, 5, 15, /*seed=*/42);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    ActiveSetSettings as_settings;
    as_settings.diff_method = ActiveSetDiffMethod::Smoothed;
    as_settings.diff_smoothing_mu = 1e-6;

    ActiveSetSolver solver(qp.n, qp.m, 1,
        qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
        qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
        qp.cones, as_settings, true);
    solver.setup(qp.P_vals.data(), qp.A_vals.data());
    solver.solve(qp.q.data(), qp.b.data());
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    std::vector<double> dx_bar(qp.n);
    for (int64_t i = 0; i < qp.n; i++) dx_bar[i] = dist(rng);
    std::vector<double> dz_bar(qp.m, 0.0), ds_bar(qp.m, 0.0);
    solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

    double eps = 1e-6;
    double max_err = 0.0;
    for (int64_t dim = 0; dim < qp.m; dim++) {
        std::vector<double> b_pert = qp.b;
        b_pert[dim] += eps;

        ActiveSetSolver solver2(qp.n, qp.m, 1,
            qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
            qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
            qp.cones);
        solver2.setup(qp.P_vals.data(), qp.A_vals.data());
        solver2.solve(qp.q.data(), b_pert.data());

        double fd = 0.0;
        for (int64_t i = 0; i < qp.n; i++)
            fd += dx_bar[i] * (solver2.x_sol[i] - solver.x_sol[i]) / eps;
        double err = std::abs(solver.db[dim] - fd);
        max_err = std::max(max_err, err);
    }
    std::cout << "  Smoothed backward max FD error (db): " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-3) << "Smoothed backward db FD check failed";
}

TEST_F(ActiveSetTest, SmoothedBackwardFD_dP) {
    RandomQP qp(10, 20, 5, 15, /*seed=*/42);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    ActiveSetSettings as_settings;
    as_settings.diff_method = ActiveSetDiffMethod::Smoothed;
    as_settings.diff_smoothing_mu = 1e-6;

    ActiveSetSolver solver(qp.n, qp.m, 1,
        qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
        qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
        qp.cones, as_settings, true);
    solver.setup(qp.P_vals.data(), qp.A_vals.data());
    solver.solve(qp.q.data(), qp.b.data());
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    std::vector<double> dx_bar(qp.n);
    for (int64_t i = 0; i < qp.n; i++) dx_bar[i] = dist(rng);
    std::vector<double> dz_bar(qp.m, 0.0), ds_bar(qp.m, 0.0);
    solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

    double eps = 1e-6;
    double max_err = 0.0;
    int64_t n = qp.n;

    // Check diagonal + a few off-diagonal P entries
    for (int64_t i = 0; i < n; i++) {
        // Diagonal: perturb P[i,i]
        int64_t k_ii = qp.P_ro[i] + i;  // dense CSR → entry i in row i
        std::vector<double> P_pert = qp.P_vals;
        P_pert[k_ii] += eps;

        ActiveSetSolver solver2(qp.n, qp.m, 1,
            qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
            qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
            qp.cones);
        solver2.setup(P_pert.data(), qp.A_vals.data());
        solver2.solve(qp.q.data(), qp.b.data());

        double fd = 0.0;
        for (int64_t d = 0; d < n; d++)
            fd += dx_bar[d] * (solver2.x_sol[d] - solver.x_sol[d]) / eps;
        double err = std::abs(solver.dP_values[k_ii] - fd);
        max_err = std::max(max_err, err);
    }
    std::cout << "  Smoothed backward max FD error (dP diag): " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-3) << "Smoothed backward dP FD check failed";
}

TEST_F(ActiveSetTest, SmoothedBackwardFD_dA) {
    RandomQP qp(10, 20, 5, 15, /*seed=*/42);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    ActiveSetSettings as_settings;
    as_settings.diff_method = ActiveSetDiffMethod::Smoothed;
    as_settings.diff_smoothing_mu = 1e-6;

    ActiveSetSolver solver(qp.n, qp.m, 1,
        qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
        qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
        qp.cones, as_settings, true);
    solver.setup(qp.P_vals.data(), qp.A_vals.data());
    solver.solve(qp.q.data(), qp.b.data());
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    std::vector<double> dx_bar(qp.n);
    for (int64_t i = 0; i < qp.n; i++) dx_bar[i] = dist(rng);
    std::vector<double> dz_bar(qp.m, 0.0), ds_bar(qp.m, 0.0);
    solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

    double eps = 1e-6;
    double max_err = 0.0;
    int64_t worst_idx = -1;
    for (int64_t idx = 0; idx < qp.nnzA; idx += 5) {
        std::vector<double> A_pert = qp.A_vals;
        A_pert[idx] += eps;

        ActiveSetSolver solver2(qp.n, qp.m, 1,
            qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
            qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
            qp.cones);
        solver2.setup(qp.P_vals.data(), A_pert.data());
        solver2.solve(qp.q.data(), qp.b.data());

        double fd = 0.0;
        for (int64_t i = 0; i < qp.n; i++)
            fd += dx_bar[i] * (solver2.x_sol[i] - solver.x_sol[i]) / eps;
        double err = std::abs(solver.dA_values[idx] - fd);
        if (err > max_err) { max_err = err; worst_idx = idx; }
    }
    std::cout << "  Smoothed backward max FD error (dA): " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-3) << "Smoothed backward dA FD check failed";
}

// ============================================================================
// Backward tests with nonzero dz_bar and ds_bar
// ============================================================================

// Helper: compute loss = dx_bar'*x + dz_bar'*z + ds_bar'*s
static double backward_loss(const ActiveSetSolver& s,
                             const double* dx_bar, const double* dz_bar,
                             const double* ds_bar, int64_t n, int64_t m) {
    double l = 0;
    for (int64_t i = 0; i < n; i++) l += dx_bar[i] * s.x_sol[i];
    for (int64_t i = 0; i < m; i++) l += dz_bar[i] * s.z_sol[i];
    for (int64_t i = 0; i < m; i++) l += ds_bar[i] * s.s_sol[i];
    return l;
}

TEST_F(ActiveSetTest, BackwardExactNonzeroDzDs_dq) {
    RandomQP qp(10, 20, 5, 15, /*seed=*/42);
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    ActiveSetSolver solver(qp.n, qp.m, 1,
        qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
        qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
        qp.cones, ActiveSetSettings{}, true);
    solver.setup(qp.P_vals.data(), qp.A_vals.data());
    solver.solve(qp.q.data(), qp.b.data());
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    std::vector<double> dx_bar(qp.n), dz_bar(qp.m), ds_bar(qp.m);
    for (auto& v : dx_bar) v = dist(rng);
    for (auto& v : dz_bar) v = dist(rng);
    for (auto& v : ds_bar) v = dist(rng);
    solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

    double eps = 1e-6;
    double max_err = 0;
    for (int64_t dim = 0; dim < qp.n; dim++) {
        std::vector<double> q_pert = qp.q;
        q_pert[dim] += eps;
        ActiveSetSolver s2(qp.n, qp.m, 1,
            qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
            qp.A_ro.data(), qp.A_ci.data(), qp.nnzA, qp.cones);
        s2.setup(qp.P_vals.data(), qp.A_vals.data());
        s2.solve(q_pert.data(), qp.b.data());
        double fd = (backward_loss(s2, dx_bar.data(), dz_bar.data(), ds_bar.data(), qp.n, qp.m)
                   - backward_loss(solver, dx_bar.data(), dz_bar.data(), ds_bar.data(), qp.n, qp.m)) / eps;
        max_err = std::max(max_err, std::abs(solver.dq[dim] - fd));
    }
    std::cout << "  Exact backward (nonzero dz/ds) max FD error (dq): " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-3);
}

TEST_F(ActiveSetTest, BackwardExactNonzeroDzDs_db) {
    RandomQP qp(10, 20, 5, 15, /*seed=*/42);
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    ActiveSetSolver solver(qp.n, qp.m, 1,
        qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
        qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
        qp.cones, ActiveSetSettings{}, true);
    solver.setup(qp.P_vals.data(), qp.A_vals.data());
    solver.solve(qp.q.data(), qp.b.data());
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    std::vector<double> dx_bar(qp.n), dz_bar(qp.m), ds_bar(qp.m);
    for (auto& v : dx_bar) v = dist(rng);
    for (auto& v : dz_bar) v = dist(rng);
    for (auto& v : ds_bar) v = dist(rng);
    solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

    double eps = 1e-6;
    double max_err = 0;
    for (int64_t dim = 0; dim < qp.m; dim++) {
        std::vector<double> b_pert = qp.b;
        b_pert[dim] += eps;
        ActiveSetSolver s2(qp.n, qp.m, 1,
            qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
            qp.A_ro.data(), qp.A_ci.data(), qp.nnzA, qp.cones);
        s2.setup(qp.P_vals.data(), qp.A_vals.data());
        s2.solve(qp.q.data(), b_pert.data());
        double fd = (backward_loss(s2, dx_bar.data(), dz_bar.data(), ds_bar.data(), qp.n, qp.m)
                   - backward_loss(solver, dx_bar.data(), dz_bar.data(), ds_bar.data(), qp.n, qp.m)) / eps;
        max_err = std::max(max_err, std::abs(solver.db[dim] - fd));
    }
    std::cout << "  Exact backward (nonzero dz/ds) max FD error (db): " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-3);
}

TEST_F(ActiveSetTest, SmoothedBackwardNonzeroDzDs_dq) {
    RandomQP qp(10, 20, 5, 15, /*seed=*/42);
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    ActiveSetSettings as_settings;
    as_settings.diff_method = ActiveSetDiffMethod::Smoothed;
    as_settings.diff_smoothing_mu = 1e-6;

    ActiveSetSolver solver(qp.n, qp.m, 1,
        qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
        qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
        qp.cones, as_settings, true);
    solver.setup(qp.P_vals.data(), qp.A_vals.data());
    solver.solve(qp.q.data(), qp.b.data());
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    std::vector<double> dx_bar(qp.n), dz_bar(qp.m), ds_bar(qp.m);
    for (auto& v : dx_bar) v = dist(rng);
    for (auto& v : dz_bar) v = dist(rng);
    for (auto& v : ds_bar) v = dist(rng);
    solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

    double eps = 1e-6;
    double max_err = 0;
    for (int64_t dim = 0; dim < qp.n; dim++) {
        std::vector<double> q_pert = qp.q;
        q_pert[dim] += eps;
        ActiveSetSolver s2(qp.n, qp.m, 1,
            qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
            qp.A_ro.data(), qp.A_ci.data(), qp.nnzA, qp.cones);
        s2.setup(qp.P_vals.data(), qp.A_vals.data());
        s2.solve(q_pert.data(), qp.b.data());
        double fd = (backward_loss(s2, dx_bar.data(), dz_bar.data(), ds_bar.data(), qp.n, qp.m)
                   - backward_loss(solver, dx_bar.data(), dz_bar.data(), ds_bar.data(), qp.n, qp.m)) / eps;
        max_err = std::max(max_err, std::abs(solver.dq[dim] - fd));
    }
    std::cout << "  Smoothed backward (nonzero dz/ds) max FD error (dq): " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-3);
}

TEST_F(ActiveSetTest, SmoothedBackwardNonzeroDzDs_db) {
    RandomQP qp(10, 20, 5, 15, /*seed=*/42);
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    ActiveSetSettings as_settings;
    as_settings.diff_method = ActiveSetDiffMethod::Smoothed;
    as_settings.diff_smoothing_mu = 1e-6;

    ActiveSetSolver solver(qp.n, qp.m, 1,
        qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
        qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
        qp.cones, as_settings, true);
    solver.setup(qp.P_vals.data(), qp.A_vals.data());
    solver.solve(qp.q.data(), qp.b.data());
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    std::vector<double> dx_bar(qp.n), dz_bar(qp.m), ds_bar(qp.m);
    for (auto& v : dx_bar) v = dist(rng);
    for (auto& v : dz_bar) v = dist(rng);
    for (auto& v : ds_bar) v = dist(rng);
    solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

    double eps = 1e-6;
    double max_err = 0;
    for (int64_t dim = 0; dim < qp.m; dim++) {
        std::vector<double> b_pert = qp.b;
        b_pert[dim] += eps;
        ActiveSetSolver s2(qp.n, qp.m, 1,
            qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
            qp.A_ro.data(), qp.A_ci.data(), qp.nnzA, qp.cones);
        s2.setup(qp.P_vals.data(), qp.A_vals.data());
        s2.solve(qp.q.data(), b_pert.data());
        double fd = (backward_loss(s2, dx_bar.data(), dz_bar.data(), ds_bar.data(), qp.n, qp.m)
                   - backward_loss(solver, dx_bar.data(), dz_bar.data(), ds_bar.data(), qp.n, qp.m)) / eps;
        max_err = std::max(max_err, std::abs(solver.db[dim] - fd));
    }
    std::cout << "  Smoothed backward (nonzero dz/ds) max FD error (db): " << max_err << std::endl;
    EXPECT_LT(max_err, 1e-3);
}

TEST_F(ActiveSetTest, SmoothedBackwardNonzeroDzDs_dA) {
    RandomQP qp(10, 20, 5, 15, /*seed=*/42);
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    ActiveSetSettings as_settings;
    as_settings.diff_method = ActiveSetDiffMethod::Smoothed;
    as_settings.diff_smoothing_mu = 1e-6;

    ActiveSetSolver solver(qp.n, qp.m, 1,
        qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
        qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
        qp.cones, as_settings, true);
    solver.setup(qp.P_vals.data(), qp.A_vals.data());
    solver.solve(qp.q.data(), qp.b.data());
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    std::vector<double> dx_bar(qp.n), dz_bar(qp.m), ds_bar(qp.m);
    for (auto& v : dx_bar) v = dist(rng);
    for (auto& v : dz_bar) v = dist(rng);
    for (auto& v : ds_bar) v = dist(rng);
    solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

    double eps = 1e-6;
    double max_err = 0;
    for (int64_t idx = 0; idx < qp.nnzA; idx += 5) {
        std::vector<double> A_pert = qp.A_vals;
        A_pert[idx] += eps;
        ActiveSetSolver s2(qp.n, qp.m, 1,
            qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
            qp.A_ro.data(), qp.A_ci.data(), qp.nnzA, qp.cones);
        s2.setup(qp.P_vals.data(), A_pert.data());
        s2.solve(qp.q.data(), qp.b.data());
        double fd = (backward_loss(s2, dx_bar.data(), dz_bar.data(), ds_bar.data(), qp.n, qp.m)
                   - backward_loss(solver, dx_bar.data(), dz_bar.data(), ds_bar.data(), qp.n, qp.m)) / eps;
        max_err = std::max(max_err, std::abs(solver.dA_values[idx] - fd));
    }
    std::cout << "  Smoothed backward (nonzero dz/ds) max FD error (dA): " << max_err << std::endl;
    // Slightly relaxed: smoothed analytic vs non-smoothed FD introduces O(μ) mismatch
    EXPECT_LT(max_err, 5e-3);
}

// ============================================================================
// Smoothing monotonicity: as μ increases, gradients should be smoother
// ============================================================================

TEST_F(ActiveSetTest, SmoothedMonotonicityInMu) {
    // Sweep q[0] across a constraint kink and measure gradient jump at each μ.
    // Larger μ should produce smaller jumps (smoother gradient).
    const int64_t n = 2, m = 3, batch = 1;

    int64_t P_ro[] = {0, 1, 2};
    int64_t P_ci[] = {0, 1};
    int64_t A_ro[] = {0, 2, 3, 4};
    int64_t A_ci[] = {0, 1, 0, 1};

    Cones cones;
    cones.numZeroCones = 1;
    cones.numNonnegCones = 2;

    double P_vals[] = {1.0, 1.0};
    double A_vals[] = {1.0, 1.0, -1.0, -1.0};

    double dx_bar[] = {1.0, 0.0};
    double dz_bar[] = {0.0, 0.0, 0.0};
    double ds_bar[] = {0.0, 0.0, 0.0};

    // Sweep q[0] near the kink where x1 hits the bound
    std::vector<double> mu_values = {1e-8, 1e-6, 1e-4};
    std::vector<double> max_jumps;

    for (double mu : mu_values) {
        ActiveSetSettings as_settings;
        as_settings.diff_method = ActiveSetDiffMethod::Smoothed;
        as_settings.diff_smoothing_mu = mu;

        std::vector<double> grads;
        int N = 101;
        for (int i = 0; i < N; i++) {
            double q0 = -2.0 + 4.0 * i / (N - 1);  // sweep from -2 to 2
            double q_arr[] = {q0, 0.0};
            double b[] = {1.0, 0.0, 0.0};

            ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 4, cones,
                                   as_settings, true);
            solver.setup(P_vals, A_vals);
            solver.solve(q_arr, b);
            if (solver.status_vec[0] != static_cast<int32_t>(SolverStatus::Solved))
                continue;
            solver.backward(dx_bar, dz_bar, ds_bar);
            grads.push_back(solver.dq[0]);
        }

        double max_jump = 0;
        for (size_t i = 1; i < grads.size(); i++)
            max_jump = std::max(max_jump, std::abs(grads[i] - grads[i-1]));
        max_jumps.push_back(max_jump);

        std::cout << "  μ=" << mu << " max_jump=" << max_jump << std::endl;
    }

    // Monotonicity: larger μ should produce smaller or equal max jumps
    for (size_t i = 1; i < max_jumps.size(); i++) {
        EXPECT_LE(max_jumps[i], max_jumps[i-1] + 1e-10)
            << "Smoothing monotonicity violated: μ=" << mu_values[i]
            << " jump=" << max_jumps[i] << " > μ=" << mu_values[i-1]
            << " jump=" << max_jumps[i-1];
    }
}

TEST_F(ActiveSetTest, SmoothedConvergesToExact) {
    // As μ → 0, smoothed gradients should converge to exact gradients
    RandomQP qp(10, 20, 5, 15, /*seed=*/42);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Compute exact gradients
    ActiveSetSolver exact_solver(qp.n, qp.m, 1,
        qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
        qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
        qp.cones, ActiveSetSettings{}, true);
    exact_solver.setup(qp.P_vals.data(), qp.A_vals.data());
    exact_solver.solve(qp.q.data(), qp.b.data());
    ASSERT_EQ(exact_solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    std::vector<double> dx_bar(qp.n), dz_bar(qp.m), ds_bar(qp.m);
    for (auto& v : dx_bar) v = dist(rng);
    for (auto& v : dz_bar) v = dist(rng);
    for (auto& v : ds_bar) v = dist(rng);
    exact_solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

    // Compute smoothed gradients at decreasing μ
    std::vector<double> mu_values = {1e-2, 1e-4, 1e-6, 1e-8};
    double prev_err = 1e10;

    for (double mu : mu_values) {
        ActiveSetSettings as_settings;
        as_settings.diff_method = ActiveSetDiffMethod::Smoothed;
        as_settings.diff_smoothing_mu = mu;

        ActiveSetSolver solver(qp.n, qp.m, 1,
            qp.P_ro.data(), qp.P_ci.data(), qp.nnzP,
            qp.A_ro.data(), qp.A_ci.data(), qp.nnzA,
            qp.cones, as_settings, true);
        solver.setup(qp.P_vals.data(), qp.A_vals.data());
        solver.solve(qp.q.data(), qp.b.data());
        solver.backward(dx_bar.data(), dz_bar.data(), ds_bar.data());

        double max_err = 0;
        for (int64_t i = 0; i < qp.n; i++)
            max_err = std::max(max_err, std::abs(solver.dq[i] - exact_solver.dq[i]));
        for (int64_t i = 0; i < qp.m; i++)
            max_err = std::max(max_err, std::abs(solver.db[i] - exact_solver.db[i]));

        std::cout << "  μ=" << mu << " max |smoothed - exact|=" << max_err << std::endl;

        EXPECT_LE(max_err, prev_err + 1e-10)
            << "Smoothed gradients should converge to exact as μ→0";
        prev_err = max_err;
    }

    // At μ=1e-8, should be very close to exact
    EXPECT_LT(prev_err, 1e-3) << "Smoothed gradients at μ=1e-8 should match exact";
}

// ============================================================================
// Throughput benchmarks
// ============================================================================

TEST_F(ActiveSetTest, BatchThroughput) {
    // Benchmark batch throughput at various sizes
    const int64_t n = 5, m = 10;
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Dense identity P, random A
    std::vector<int64_t> P_ro(n + 1), P_ci(n);
    std::vector<double> P_vals(n, 1.0);
    for (int i = 0; i < n; i++) { P_ro[i] = i; P_ci[i] = i; }
    P_ro[n] = n;

    std::vector<int64_t> A_ro(m + 1), A_ci(m * n);
    std::vector<double> A_vals(m * n);
    int ak = 0;
    for (int i = 0; i < m; i++) {
        A_ro[i] = ak;
        for (int j = 0; j < n; j++) {
            A_ci[ak] = j;
            A_vals[ak] = dist(rng);
            ak++;
        }
    }
    A_ro[m] = ak;

    Cones cones;
    cones.numZeroCones = 2;
    cones.numNonnegCones = 8;

    std::vector<int64_t> batch_sizes = {1, 4, 16, 64, 256, 1024};
    std::cout << "\n  Batch throughput (n=" << n << ", m=" << m << "):" << std::endl;

    for (int64_t batch : batch_sizes) {
        std::vector<double> q(batch * n), b(batch * m);
        for (int64_t i = 0; i < batch * n; i++) q[i] = dist(rng);
        for (int64_t i = 0; i < batch * m; i++) b[i] = 5.0 + std::abs(dist(rng));

        ActiveSetSolver solver(n, m, batch,
            P_ro.data(), P_ci.data(), n,
            A_ro.data(), A_ci.data(), ak,
            cones);
        solver.setup(P_vals.data(), A_vals.data());

        auto t0 = std::chrono::high_resolution_clock::now();
        solver.solve(q.data(), b.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        int solved = 0;
        for (int64_t i = 0; i < batch; i++) {
            if (solver.status_vec[i] == static_cast<int32_t>(SolverStatus::Solved))
                solved++;
        }

        std::cout << "    batch=" << batch
                  << " time=" << ms << "ms"
                  << " throughput=" << (batch / (ms / 1000.0)) << " solves/s"
                  << " solved=" << solved << "/" << batch
                  << std::endl;
    }
}

// ==========================================================================
// Regression: stale M buffer on repeated solve()
// ==========================================================================

TEST_F(ActiveSetTest, RepeatedSolveNoStaleM) {
    // Solve the same problem twice — results must match.
    // Before the fix, the += in daqp_update_M accumulated on stale M values
    // from the first solve, corrupting the second.
    const int64_t n = 2, m = 2, batch = 1;
    int64_t P_ro[] = {0, 2, 4};
    int64_t P_ci[] = {0, 1, 0, 1};
    double P_vals[] = {2.0, 0.5, 0.5, 2.0};  // off-diagonal to exercise += path
    int64_t A_ro[] = {0, 2, 4};
    int64_t A_ci[] = {0, 1, 0, 1};
    double A_vals[] = {1.0, 1.0, 1.0, -1.0};
    Cones cones{0, 2};
    double q[] = {-1.0, -2.0};
    double b[] = {1.0, 0.5};

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 4, A_ro, A_ci, 4, cones);
    solver.setup(P_vals, A_vals);

    solver.solve(q, b);
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    double x1_0 = solver.x_sol[0];
    double x1_1 = solver.x_sol[1];

    // Second solve with same data
    solver.solve(q, b);
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(solver.x_sol[0], x1_0, 1e-12)
        << "x[0] changed on repeated solve (stale M buffer)";
    EXPECT_NEAR(solver.x_sol[1], x1_1, 1e-12)
        << "x[1] changed on repeated solve (stale M buffer)";
}

TEST_F(ActiveSetTest, RepeatedSolveDifferentRHS) {
    // Solve two different problems on the same solver instance.
    const int64_t n = 2, m = 1, batch = 1;
    int64_t P_ro[] = {0, 2, 4};
    int64_t P_ci[] = {0, 1, 0, 1};
    double P_vals[] = {3.0, 1.0, 1.0, 3.0};
    int64_t A_ro[] = {0, 2};
    int64_t A_ci[] = {0, 1};
    double A_vals[] = {1.0, 1.0};
    Cones cones{0, 1};

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 4, A_ro, A_ci, 2, cones);
    solver.setup(P_vals, A_vals);

    double q1[] = {-1.0, -1.0};
    double b1[] = {1.0};
    solver.solve(q1, b1);
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    double x1_0 = solver.x_sol[0];

    double q2[] = {-2.0, 0.0};
    double b2[] = {0.5};
    solver.solve(q2, b2);
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));

    // Verify second solution is different (problem changed)
    // and re-solve first to confirm it still matches
    solver.solve(q1, b1);
    ASSERT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved));
    EXPECT_NEAR(solver.x_sol[0], x1_0, 1e-12)
        << "First problem result changed after solving a different problem";
}

// ==========================================================================
// Zero-row constraint infeasibility detection
// ==========================================================================

TEST_F(ActiveSetTest, ZeroRowConstraintInfeasibleNonneg) {
    // Constraint: 0*x + s = -1, s >= 0 → infeasible (s = -1 < 0)
    const int64_t n = 2, m = 2, batch = 1;
    int64_t P_ro[] = {0, 1, 2};
    int64_t P_ci[] = {0, 1};
    double P_vals[] = {1.0, 1.0};
    int64_t A_ro[] = {0, 2, 2};  // second row has 0 nonzeros
    int64_t A_ci[] = {0, 1};
    double A_vals[] = {1.0, 1.0};
    Cones cones{0, 2};  // both nonneg
    double q[] = {-1.0, -1.0};
    double b[] = {1.0, -1.0};  // b[1] = -1 → infeasible for zero row

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 2, cones);
    solver.setup(P_vals, A_vals);
    solver.solve(q, b);
    EXPECT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::PrimalInfeasible))
        << "Zero-row nonneg constraint with b<0 should be infeasible";
}

TEST_F(ActiveSetTest, ZeroRowConstraintInfeasibleEquality) {
    // Constraint: 0*x = 1 → infeasible
    const int64_t n = 2, m = 2, batch = 1;
    int64_t P_ro[] = {0, 1, 2};
    int64_t P_ci[] = {0, 1};
    double P_vals[] = {1.0, 1.0};
    int64_t A_ro[] = {0, 0, 2};  // first row has 0 nonzeros, second is normal
    int64_t A_ci[] = {0, 1};
    double A_vals[] = {1.0, 1.0};
    Cones cones{1, 1};  // 1 equality (zero cone), 1 nonneg
    double q[] = {-1.0, -1.0};
    double b[] = {1.0, 2.0};  // b[0] = 1 with zero A row → infeasible

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 2, cones);
    solver.setup(P_vals, A_vals);
    solver.solve(q, b);
    EXPECT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::PrimalInfeasible))
        << "Zero-row equality constraint with b!=0 should be infeasible";
}

TEST_F(ActiveSetTest, ZeroRowConstraintFeasible) {
    // Constraint: 0*x + s = 1, s >= 0 → feasible (s = 1 >= 0)
    const int64_t n = 2, m = 2, batch = 1;
    int64_t P_ro[] = {0, 1, 2};
    int64_t P_ci[] = {0, 1};
    double P_vals[] = {1.0, 1.0};
    int64_t A_ro[] = {0, 2, 2};  // second row has 0 nonzeros
    int64_t A_ci[] = {0, 1};
    double A_vals[] = {1.0, 1.0};
    Cones cones{0, 2};
    double q[] = {-1.0, -1.0};
    double b[] = {1.0, 1.0};  // b[1] = 1 >= 0, so zero-row is feasible (ignored)

    ActiveSetSolver solver(n, m, batch, P_ro, P_ci, 2, A_ro, A_ci, 2, cones);
    solver.setup(P_vals, A_vals);
    solver.solve(q, b);
    EXPECT_EQ(solver.status_vec[0], static_cast<int32_t>(SolverStatus::Solved))
        << "Zero-row nonneg constraint with b>=0 should be feasible";
}
