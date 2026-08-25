"""Tests for auto-detection routing on MPC-structured QPs.

CPU Riccati is intentionally disabled (faer outperforms it). These tests verify
that auto-detection correctly identifies block-tridiagonal structure and routes
to the appropriate sparse solver without breaking correctness. They do NOT test
the Riccati factorization itself.
"""

import moreau_cpu._cpu_solver as moreau
import numpy as np
import pytest
from scipy import sparse


def build_mpc_qp(horizon, nx, nu, dt=0.1):
    """Build an MPC-structured QP (double integrator) matching bench_riccati.rs."""
    n_vars = horizon * (nx + nu) + nx

    # P: diagonal (state cost + control cost + terminal cost)
    p_rows, p_cols, p_vals = [], [], []
    for t in range(horizon):
        x_off = t * (nx + nu)
        u_off = x_off + nx
        for i in range(nx):
            p_rows.append(x_off + i)
            p_cols.append(x_off + i)
            p_vals.append(1.0)
        for i in range(nu):
            p_rows.append(u_off + i)
            p_cols.append(u_off + i)
            p_vals.append(0.1)
    xt_off = horizon * (nx + nu)
    for i in range(nx):
        p_rows.append(xt_off + i)
        p_cols.append(xt_off + i)
        p_vals.append(10.0)

    P = sparse.csr_array((p_vals, (p_rows, p_cols)), shape=(n_vars, n_vars))
    P = sparse.triu(P).tocsr()
    q = np.zeros(n_vars)

    # Constraints: initial state + dynamics + control bounds
    n_eq_init = nx
    n_eq_dyn = horizon * nx
    n_ineq = horizon * nu * 2
    n_con = n_eq_init + n_eq_dyn + n_ineq

    a_rows, a_cols, a_vals = [], [], []
    b = np.zeros(n_con)
    half = nx // 2

    # Initial condition: x_0 = [1, 0, ...]
    row = 0
    for i in range(nx):
        a_rows.append(row)
        a_cols.append(i)
        a_vals.append(1.0)
        b[row] = 1.0 if i == 0 else 0.0
        row += 1

    # Dynamics
    for t in range(horizon):
        x_off = t * (nx + nu)
        u_off = x_off + nx
        x_next = x_off + nx + nu
        for i in range(half):
            # Position: x_{t+1,i} = x_{t,i} + dt*x_{t,half+i} + 0.5*dt^2*u_{t,i}
            a_rows.append(row)
            a_cols.append(x_off + i)
            a_vals.append(1.0)
            a_rows.append(row)
            a_cols.append(x_off + half + i)
            a_vals.append(dt)
            if i < nu:
                a_rows.append(row)
                a_cols.append(u_off + i)
                a_vals.append(0.5 * dt * dt)
            a_rows.append(row)
            a_cols.append(x_next + i)
            a_vals.append(-1.0)
            row += 1
            # Velocity: x_{t+1,half+i} = x_{t,half+i} + dt*u_{t,i}
            a_rows.append(row)
            a_cols.append(x_off + half + i)
            a_vals.append(1.0)
            if i < nu:
                a_rows.append(row)
                a_cols.append(u_off + i)
                a_vals.append(dt)
            a_rows.append(row)
            a_cols.append(x_next + half + i)
            a_vals.append(-1.0)
            row += 1

    # Control bounds: -1 <= u <= 1
    for t in range(horizon):
        u_off = t * (nx + nu) + nx
        for i in range(nu):
            a_rows.append(row)
            a_cols.append(u_off + i)
            a_vals.append(1.0)
            b[row] = 1.0
            row += 1
            a_rows.append(row)
            a_cols.append(u_off + i)
            a_vals.append(-1.0)
            b[row] = 1.0
            row += 1

    assert row == n_con
    A = sparse.csr_array((a_vals, (a_rows, a_cols)), shape=(n_con, n_vars))
    cones = [moreau.ZeroConeT(n_eq_init + n_eq_dyn), moreau.NonnegativeConeT(n_ineq)]

    return P, q, A, b, cones


def solve_with_method(P, q, A, b, cones, method):
    settings = moreau.DefaultSettings()
    settings.verbose = False
    settings.ipm.direct_solve_method = method
    solver = moreau.DefaultSolver(P, q, A, b, cones, settings)
    return solver.solve()


class TestRiccatiDetection:
    """Test that auto-detection routes MPC problems to Riccati."""

    def test_mpc_auto_matches_qdldl(self):
        """MPC problem solved with auto (Riccati) matches QDLDL solution."""
        P, q, A, b, cones = build_mpc_qp(horizon=20, nx=6, nu=3)

        sol_auto = solve_with_method(P, q, A, b, cones, "auto")
        sol_qdldl = solve_with_method(P, q, A, b, cones, "qdldl")

        assert sol_auto.status == moreau.SolverStatus.Solved
        assert sol_qdldl.status == moreau.SolverStatus.Solved
        np.testing.assert_allclose(sol_auto.x, sol_qdldl.x, atol=1e-6)
        np.testing.assert_allclose(sol_auto.obj_val, sol_qdldl.obj_val, atol=1e-6)

    def test_mpc_auto_matches_faer(self):
        """MPC problem solved with auto (Riccati) matches faer solution."""
        P, q, A, b, cones = build_mpc_qp(horizon=10, nx=4, nu=2)

        sol_auto = solve_with_method(P, q, A, b, cones, "auto")
        sol_faer = solve_with_method(P, q, A, b, cones, "faer")

        assert sol_auto.status == moreau.SolverStatus.Solved
        assert sol_faer.status == moreau.SolverStatus.Solved
        np.testing.assert_allclose(sol_auto.x, sol_faer.x, atol=1e-6)
        np.testing.assert_allclose(sol_auto.obj_val, sol_faer.obj_val, atol=1e-6)

    def test_non_mpc_falls_back_to_ldl(self):
        """Non-block-tridiagonal problem with auto still works (uses LDL)."""
        n = 10
        P = sparse.eye(n, format="csr")
        P = sparse.triu(P).tocsr()
        # Dense A: prevents block-tridiagonal detection
        A = sparse.random(n, n, density=0.5, format="csr")
        A = A + sparse.eye(n, format="csr")  # ensure nonsingular
        q = np.ones(n)
        b = np.ones(n) * 10.0
        cones = [moreau.NonnegativeConeT(n)]

        sol = solve_with_method(P, q, A, b, cones, "auto")
        assert sol.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)


class TestRiccatiSizes:
    """Test Riccati on different MPC problem sizes."""

    @pytest.mark.parametrize(
        "horizon,nx,nu",
        [
            (5, 4, 2),
            (10, 6, 3),
            (50, 4, 2),
            (20, 8, 4),
        ],
    )
    def test_mpc_sizes(self, horizon, nx, nu):
        P, q, A, b, cones = build_mpc_qp(horizon, nx, nu)

        sol_auto = solve_with_method(P, q, A, b, cones, "auto")
        sol_qdldl = solve_with_method(P, q, A, b, cones, "qdldl")

        assert sol_auto.status == moreau.SolverStatus.Solved
        np.testing.assert_allclose(sol_auto.x, sol_qdldl.x, atol=1e-6)


class TestMinimalHorizons:
    """Edge case tests: very small horizons and block sizes.

    Verify graceful fallback for minimal MPC problems where
    block-tridiagonal structure may be degenerate.
    """

    @pytest.mark.parametrize(
        "horizon,nx,nu",
        [
            (1, 2, 1),
            (1, 4, 2),
            (2, 2, 1),
            (2, 4, 2),
        ],
    )
    def test_minimal_horizon(self, horizon, nx, nu):
        P, q, A, b, cones = build_mpc_qp(horizon, nx, nu)

        sol_auto = solve_with_method(P, q, A, b, cones, "auto")
        sol_qdldl = solve_with_method(P, q, A, b, cones, "qdldl")

        assert sol_auto.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        np.testing.assert_allclose(sol_auto.x, sol_qdldl.x, atol=1e-6)
