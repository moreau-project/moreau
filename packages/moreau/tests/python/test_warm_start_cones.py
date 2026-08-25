"""Tests for warm starting with exponential and power cones.

Existing warm start tests only cover QP (zero+nonneg cones) and SOC.
This fills the gap for exp and power cones.
"""

import moreau
import pytest
import numpy as np


def _make_solver(prob, device, batch_size=1):
    """Create a CompiledSolver for the given problem."""
    settings = moreau.Settings(device=device, batch_size=batch_size)
    solver = moreau.CompiledSolver(
        n=prob["n"],
        m=prob["m"],
        P_row_offsets=prob["P_row_offsets"],
        P_col_indices=prob["P_col_indices"],
        A_row_offsets=prob["A_row_offsets"],
        A_col_indices=prob["A_col_indices"],
        cones=prob["cones"],
        settings=settings,
    )
    solver.setup(P_values=prob["P_values"], A_values=prob["A_values"])
    return solver


@pytest.fixture
def exp_problem():
    """Exponential cone problem.

    minimize  x0 + x1 + x2 + 0.5*(x0^2 + x1^2 + x2^2)
    subject to (x0, x1, x2) in K_exp  [x1*exp(x0/x1) <= x2, x1 > 0]
               x0 + x1 + x2 <= 3       (nonneg: -x0 - x1 - x2 + s = -3, s >= 0)
    """
    n = 3
    m = 4  # 3 for exp cone + 1 nonneg

    P_row_offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    P_col_indices = np.array([0, 1, 2], dtype=np.int64)
    P_values = np.array([1.0, 1.0, 1.0])

    # A maps variables to cone constraints:
    #   row 0-2: -I for exp cone (s = -Ax + b => s = x when b=0)
    #   row 3: -1 -1 -1 for nonneg (sum constraint)
    A_row_offsets = np.array([0, 2, 4, 6, 7], dtype=np.int64)
    A_col_indices = np.array([0, 0, 1, 1, 2, 2, 0], dtype=np.int64)

    # Wait, CSR: row 0 has entries at cols 0 and... Let me think more carefully.
    # Actually, we need A such that Ax + s = b with s in cone.
    # For exp cone: s in K_exp, where s = b - Ax.
    # We want s = x (variables are the cone elements), so A = -I, b = 0.
    # For nonneg: we want x0+x1+x2 <= 3, i.e. -(x0+x1+x2) + s = -3, s >= 0.

    # Row 0: col 0 -> -1.0
    # Row 1: col 1 -> -1.0
    # Row 2: col 2 -> -1.0
    # Row 3: col 0 -> -1.0, col 1 -> -1.0, col 2 -> -1.0
    A_row_offsets = np.array([0, 1, 2, 3, 6], dtype=np.int64)
    A_col_indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    A_values = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

    cones = moreau.Cones(num_exp_cones=1, num_nonneg_cones=1)

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "cones": cones,
        "q": np.array([1.0, 1.0, -1.0]),
        "b": np.array([0.0, 0.0, 0.0, -3.0]),
    }


@pytest.fixture
def power_problem():
    """Power cone problem.

    minimize  x0 + x1 + x2 + 0.5*(x0^2 + x1^2 + x2^2)
    subject to (x0, x1, x2) in K_pow(0.5)  [x0^0.5 * x1^0.5 >= |x2|, x0,x1 >= 0]
               x0 + x1 + x2 <= 5
    """
    n = 3
    m = 4  # 3 for power cone + 1 nonneg

    P_row_offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    P_col_indices = np.array([0, 1, 2], dtype=np.int64)
    P_values = np.array([1.0, 1.0, 1.0])

    # A = -I for power cone rows, then -[1,1,1] for nonneg
    A_row_offsets = np.array([0, 1, 2, 3, 6], dtype=np.int64)
    A_col_indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    A_values = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

    cones = moreau.Cones(power_alphas=[0.5], num_nonneg_cones=1)

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "cones": cones,
        "q": np.array([1.0, 1.0, 0.0]),
        "b": np.array([0.0, 0.0, 0.0, -5.0]),
    }


class TestWarmStartExpCone:
    """Test warm starting with exponential cone."""

    def test_warm_start_exp_fewer_iterations(self, exp_problem, device):
        prob = exp_problem
        q = prob["q"].reshape(1, -1)
        b = prob["b"].reshape(1, -1)

        # Cold solve
        solver_cold = _make_solver(prob, device)
        sol_cold = solver_cold.solve(qs=q, bs=b)
        info_cold = solver_cold.info
        assert info_cold.status[0] == moreau.SolverStatus.Solved
        iters_cold = (
            info_cold.iterations[0]
            if isinstance(info_cold.iterations, list)
            else info_cold.iterations
        )

        # Warm solve from cold solution
        solver_warm = _make_solver(prob, device)
        sol_warm = solver_warm.solve(
            qs=q,
            bs=b,
            warm_start=sol_cold.to_warm_start(),
        )
        info_warm = solver_warm.info
        assert info_warm.status[0] == moreau.SolverStatus.Solved
        iters_warm = (
            info_warm.iterations[0]
            if isinstance(info_warm.iterations, list)
            else info_warm.iterations
        )

        # Warm start should converge in fewer or equal iterations
        assert (
            iters_warm <= iters_cold
        ), f"Warm start ({iters_warm} iters) should be <= cold ({iters_cold} iters)"
        np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=5e-4)

    def test_warm_start_exp_perturbed(self, exp_problem, device):
        prob = exp_problem
        q1 = prob["q"].reshape(1, -1)
        q2 = (prob["q"] + np.array([0.1, -0.1, 0.05])).reshape(1, -1)
        b = prob["b"].reshape(1, -1)

        # Cold solve P1
        solver1 = _make_solver(prob, device)
        sol1 = solver1.solve(qs=q1, bs=b)
        assert solver1.info.status[0] == moreau.SolverStatus.Solved

        # Warm solve P2 from P1's solution
        solver2_warm = _make_solver(prob, device)
        sol2_warm = solver2_warm.solve(
            qs=q2,
            bs=b,
            warm_start=sol1.to_warm_start(),
        )
        assert solver2_warm.info.status[0] == moreau.SolverStatus.Solved

        # Cold solve P2 for reference
        solver2_cold = _make_solver(prob, device)
        sol2_cold = solver2_cold.solve(qs=q2, bs=b)
        assert solver2_cold.info.status[0] == moreau.SolverStatus.Solved

        np.testing.assert_allclose(sol2_warm.x, sol2_cold.x, atol=1e-4)


class TestWarmStartPowerCone:
    """Test warm starting with power cone."""

    def test_warm_start_power_fewer_iterations(self, power_problem, device):
        prob = power_problem
        q = prob["q"].reshape(1, -1)
        b = prob["b"].reshape(1, -1)

        # Cold solve
        solver_cold = _make_solver(prob, device)
        sol_cold = solver_cold.solve(qs=q, bs=b)
        info_cold = solver_cold.info
        assert info_cold.status[0] == moreau.SolverStatus.Solved
        iters_cold = (
            info_cold.iterations[0]
            if isinstance(info_cold.iterations, list)
            else info_cold.iterations
        )

        # Warm solve from cold solution
        solver_warm = _make_solver(prob, device)
        sol_warm = solver_warm.solve(
            qs=q,
            bs=b,
            warm_start=sol_cold.to_warm_start(),
        )
        info_warm = solver_warm.info
        assert info_warm.status[0] == moreau.SolverStatus.Solved
        iters_warm = (
            info_warm.iterations[0]
            if isinstance(info_warm.iterations, list)
            else info_warm.iterations
        )

        assert (
            iters_warm <= iters_cold
        ), f"Warm start ({iters_warm} iters) should be <= cold ({iters_cold} iters)"
        np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4)

    def test_warm_start_power_perturbed(self, power_problem, device):
        prob = power_problem
        q1 = prob["q"].reshape(1, -1)
        q2 = (prob["q"] + np.array([0.1, -0.1, 0.05])).reshape(1, -1)
        b = prob["b"].reshape(1, -1)

        solver1 = _make_solver(prob, device)
        sol1 = solver1.solve(qs=q1, bs=b)
        assert solver1.info.status[0] == moreau.SolverStatus.Solved

        solver2_warm = _make_solver(prob, device)
        sol2_warm = solver2_warm.solve(
            qs=q2,
            bs=b,
            warm_start=sol1.to_warm_start(),
        )
        assert solver2_warm.info.status[0] == moreau.SolverStatus.Solved

        solver2_cold = _make_solver(prob, device)
        sol2_cold = solver2_cold.solve(qs=q2, bs=b)
        assert solver2_cold.info.status[0] == moreau.SolverStatus.Solved

        np.testing.assert_allclose(sol2_warm.x, sol2_cold.x, atol=1e-4)

    def test_warm_start_power_different_alpha(self, device):
        """Test warm start with power cone alpha=0.3."""
        n, m = 3, 4
        P_row_offsets = np.array([0, 1, 2, 3], dtype=np.int64)
        P_col_indices = np.array([0, 1, 2], dtype=np.int64)
        P_values = np.array([1.0, 1.0, 1.0])

        A_row_offsets = np.array([0, 1, 2, 3, 6], dtype=np.int64)
        A_col_indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
        A_values = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

        cones = moreau.Cones(power_alphas=[0.3], num_nonneg_cones=1)

        prob = {
            "n": n,
            "m": m,
            "P_row_offsets": P_row_offsets,
            "P_col_indices": P_col_indices,
            "P_values": P_values,
            "A_row_offsets": A_row_offsets,
            "A_col_indices": A_col_indices,
            "A_values": A_values,
            "cones": cones,
        }

        q = np.array([[1.0, 1.0, 0.0]])
        b = np.array([[0.0, 0.0, 0.0, -5.0]])

        solver_cold = _make_solver(prob, device)
        sol_cold = solver_cold.solve(qs=q, bs=b)
        assert solver_cold.info.status[0] == moreau.SolverStatus.Solved

        solver_warm = _make_solver(prob, device)
        sol_warm = solver_warm.solve(
            qs=q,
            bs=b,
            warm_start=sol_cold.to_warm_start(),
        )
        assert solver_warm.info.status[0] == moreau.SolverStatus.Solved
        np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4)
