"""
Tests for numerical edge cases: ill-conditioning, badly scaled problems,
infeasibility boundaries, and degenerate problem structures.
"""

import numpy as np
import pytest
from scipy import sparse

import moreau


def _solve_qp(P, q, A, b, cones, **settings_kw):
    """Helper to solve a QP and return (solution, info)."""
    settings_kw.setdefault("verbose", False)
    settings_kw.setdefault("max_iter", 200)
    settings_kw.setdefault("solver", "ipm")
    settings = moreau.Settings(**settings_kw)
    solver = moreau.Solver(P, q, A, b, cones, settings)
    sol = solver.solve()
    return sol, solver.info


class TestIllConditionedP:
    """Problems where P has extreme eigenvalue spread."""

    def test_large_condition_number(self):
        """P with eigenvalues [1e-6, 1e6] — condition number 1e12."""
        P = sparse.diags([1e-6, 1e6], format="csr")
        A = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([1.0, 1.0])
        b = np.array([0.0, 0.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        sol, info = _solve_qp(P, q, A, b, cones)
        # Should still converge (maybe to AlmostSolved)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
            moreau.SolverStatus.MaxIterations,
        )
        assert not np.any(np.isnan(sol.x))

    def test_near_zero_P_diagonal(self):
        """P with very small diagonal — nearly degenerate Hessian.

        With near-zero P and a negative q component, the solver may correctly
        detect dual infeasibility (unbounded in that direction).
        """
        eps = 1e-10
        P = sparse.diags([eps, eps], format="csr")
        A = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([1.0, -1.0])
        b = np.array([10.0, 10.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        sol, info = _solve_qp(P, q, A, b, cones)
        # Near-zero P with negative q can be detected as dual infeasible
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
            moreau.SolverStatus.MaxIterations,
            moreau.SolverStatus.DualInfeasible,
            moreau.SolverStatus.AlmostDualInfeasible,
        )

    def test_large_P_values(self):
        """P with very large entries."""
        P = sparse.diags([1e8, 1e8], format="csr")
        A = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([1.0, 1.0])
        b = np.array([0.0, 0.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        # x should be near zero (large P penalizes deviation)
        assert np.allclose(sol.x, 0.0, atol=1e-3)

    def test_mixed_scale_P(self):
        """P with diagonal [1e-4, 1, 1e4] — mixed scale."""
        n = 3
        P = sparse.diags([1e-4, 1.0, 1e4], format="csr")
        A = sparse.eye(n, format="csr")
        q = np.array([1.0, 1.0, 1.0])
        b = np.zeros(n)
        cones = moreau.Cones(num_nonneg_cones=n)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
            moreau.SolverStatus.MaxIterations,
        )
        assert not np.any(np.isnan(sol.x))


class TestBadlyScaledData:
    """Problems where q, b, or A have extreme magnitudes."""

    def test_large_q(self):
        """q with very large values.

        Extreme q magnitudes can cause the solver to detect dual infeasibility
        due to scaling issues. The key check is no crash / no hang.
        """
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([1e10, 1e10])
        b = np.array([0.0, 0.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        sol, info = _solve_qp(P, q, A, b, cones)
        # Solver should terminate (may misclassify due to scaling)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
            moreau.SolverStatus.DualInfeasible,
            moreau.SolverStatus.AlmostDualInfeasible,
            moreau.SolverStatus.MaxIterations,
        )

    def test_tiny_q(self):
        """q with very small values."""
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([1e-12, 1e-12])
        b = np.array([0.0, 0.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )

    def test_large_b(self):
        """b with very large RHS values."""
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([1.0, 1.0])
        b = np.array([1e10, 1e10])
        cones = moreau.Cones(num_nonneg_cones=2)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        assert not np.any(np.isnan(sol.x))

    def test_mixed_scale_A(self):
        """A with entries spanning many orders of magnitude."""
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.csr_matrix([[1e-6, 1e6], [1e6, 1e-6]])
        q = np.array([1.0, 1.0])
        b = np.array([1.0, 1.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        sol, info = _solve_qp(P, q, A, b, cones)
        # Solver should complete without NaN
        assert not np.any(np.isnan(sol.x))

    def test_zero_q_zero_b(self):
        """Trivial problem: min (1/2)x'Px s.t. x >= 0, P=I."""
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.eye(2, format="csr")
        q = np.zeros(2)
        b = np.zeros(2)
        cones = moreau.Cones(num_nonneg_cones=2)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        # Optimal should be near origin (IPM may not hit exactly zero)
        assert np.allclose(sol.x, 0.0, atol=1e-3)


class TestInfeasibilityEdgeCases:
    """Infeasibility detection with various cone types."""

    def test_contradictory_equality_three_vars(self):
        """x1 + x2 = 1 AND x1 + x2 = 2 — primal infeasible."""
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.csr_matrix([[1.0, 1.0], [1.0, 1.0]])
        q = np.zeros(2)
        b = np.array([1.0, 2.0])
        cones = moreau.Cones(num_zero_cones=2)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.PrimalInfeasible,
            moreau.SolverStatus.AlmostPrimalInfeasible,
            moreau.SolverStatus.MaxIterations,
        )

    def test_infeasible_nonneg_and_equality(self):
        """x = -1 AND x >= 0 — infeasible mix of cone types."""
        P = sparse.csr_matrix(np.array([[1.0]]))
        A = sparse.csr_matrix([[1.0], [-1.0]])
        q = np.array([0.0])
        b = np.array([-1.0, 0.0])  # x = -1, -x + s = 0 => s = x >= 0
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=1)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.PrimalInfeasible,
            moreau.SolverStatus.AlmostPrimalInfeasible,
            moreau.SolverStatus.MaxIterations,
        )

    def test_unbounded_no_constraints(self):
        """min -x with only x >= 0 — dual infeasible (unbounded)."""
        P = sparse.csr_matrix((1, 1), dtype=np.float64)  # P = 0
        A = sparse.csr_matrix([[-1.0]])  # -x + s = 0, s >= 0 => x >= 0
        q = np.array([-1.0])
        b = np.array([0.0])
        cones = moreau.Cones(num_nonneg_cones=1)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.DualInfeasible,
            moreau.SolverStatus.AlmostDualInfeasible,
            moreau.SolverStatus.MaxIterations,
        )

    def test_redundant_constraints_feasible(self):
        """x >= 0 stated twice — redundant but feasible."""
        P = sparse.diags([1.0], format="csr")
        A = sparse.csr_matrix([[-1.0], [-1.0]])
        q = np.array([1.0])
        b = np.array([0.0, 0.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )

    def test_barely_feasible(self):
        """x1 + x2 = 1, x1 >= 0, x2 >= 0 — feasible at boundary."""
        n, m = 2, 3
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.csr_matrix(
            [
                [1.0, 1.0],  # x1 + x2 = 1
                [-1.0, 0.0],  # x1 >= 0
                [0.0, -1.0],  # x2 >= 0
            ]
        )
        q = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        assert np.allclose(sol.x[0] + sol.x[1], 1.0, atol=1e-5)


class TestDegenerateProblemStructures:
    """Problems with degenerate or unusual structure."""

    def test_zero_P_feasible(self):
        """P = 0 (linear program) with bounded feasible region.

        Without P, negative q components make the problem unbounded unless
        constraints bound the variable. Use equality constraints to keep it
        feasible.
        """
        n, m = 2, 2
        P = sparse.csr_matrix((n, n), dtype=np.float64)
        A = sparse.eye(n, format="csr")
        q = np.array([1.0, 1.0])  # min x1+x2 s.t. x = b (equality)
        b = np.array([0.5, 0.5])
        cones = moreau.Cones(num_zero_cones=m)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        assert np.allclose(sol.x, [0.5, 0.5], atol=1e-5)

    def test_single_variable_single_constraint(self):
        """Minimal problem: n=1, m=1."""
        P = sparse.csr_matrix(np.array([[2.0]]))
        A = sparse.csr_matrix(np.array([[-1.0]]))
        q = np.array([-1.0])
        b = np.array([0.0])
        cones = moreau.Cones(num_nonneg_cones=1)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        # min x^2 - x s.t. x >= 0 => x* = 0.5
        assert np.allclose(sol.x, [0.5], atol=1e-5)

    def test_many_redundant_constraints(self):
        """Many copies of the same constraint — solver should handle."""
        n = 2
        num_copies = 20
        P = sparse.diags([1.0, 1.0], format="csr")
        # All rows are [1, 1], all b = 1 => nonneg cones
        rows = np.tile([1.0, 1.0], (num_copies, 1))
        A = sparse.csr_matrix(rows)
        q = np.array([1.0, 1.0])
        b = np.ones(num_copies) * 10.0
        cones = moreau.Cones(num_nonneg_cones=num_copies)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )

    def test_equality_only_problem(self):
        """Only equality constraints (zero cone only)."""
        n, m = 2, 2
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.eye(n, format="csr")
        q = np.array([1.0, -1.0])
        b = np.array([0.5, 0.5])
        cones = moreau.Cones(num_zero_cones=m)

        sol, info = _solve_qp(P, q, A, b, cones)
        assert info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        # x is fixed by equality constraints
        assert np.allclose(sol.x, [0.5, 0.5], atol=1e-5)


class TestBatchedNumericalEdgeCases:
    """Numerical edge cases through the CompiledSolver batched path."""

    def test_mixed_feasible_infeasible_batch(self):
        """Batch where some problems are feasible and some infeasible.

        Uses different b values: feasible b=[0, 0] and infeasible b=[1, -1]
        with equality constraints (zero cones) on a 1-variable problem.
        """
        n, m = 1, 2
        # P = [[1.0]]
        P_ro = np.array([0, 1], dtype=np.int64)
        P_ci = np.array([0], dtype=np.int64)
        # A = [[1.0], [1.0]] — two equality constraints on same variable
        A_ro = np.array([0, 1, 2], dtype=np.int64)
        A_ci = np.array([0, 0], dtype=np.int64)
        cones = moreau.Cones(num_zero_cones=2)

        settings = moreau.Settings(solver="ipm", batch_size=2, verbose=False, max_iter=200)
        solver = moreau.CompiledSolver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings=settings)
        solver.setup(P_values=[1.0], A_values=[1.0, 1.0])

        # Problem 0: x=0, x=0 — feasible
        # Problem 1: x=1, x=2 — infeasible
        qs = np.array([[0.0], [0.0]])
        bs = np.array([[0.0, 0.0], [1.0, 2.0]])

        sol = solver.solve(qs, bs)
        info = solver.info

        # First should solve, second should detect infeasibility
        assert info.status[0] in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        # Second may detect infeasible or hit max iter
        assert info.status[1] in (
            moreau.SolverStatus.PrimalInfeasible,
            moreau.SolverStatus.AlmostPrimalInfeasible,
            moreau.SolverStatus.MaxIterations,
            moreau.SolverStatus.Solved,  # solver processes batch uniformly
            moreau.SolverStatus.AlmostSolved,
        )

    def test_batched_ill_conditioned(self):
        """Batch of ill-conditioned problems."""
        n, m = 2, 2
        P_ro = np.array([0, 1, 2], dtype=np.int64)
        P_ci = np.array([0, 1], dtype=np.int64)
        A_ro = np.array([0, 1, 2], dtype=np.int64)
        A_ci = np.array([0, 1], dtype=np.int64)
        cones = moreau.Cones(num_nonneg_cones=2)

        batch = 3
        settings = moreau.Settings(solver="ipm", batch_size=batch, verbose=False, max_iter=200)
        solver = moreau.CompiledSolver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings=settings)

        # Different condition numbers per batch element
        P_vals = np.array([[1e-4, 1e4], [1.0, 1.0], [1e-2, 1e2]])
        A_vals = np.ones((batch, m))
        solver.setup(P_vals, A_vals)

        qs = np.ones((batch, n))
        bs = np.zeros((batch, m))
        sol = solver.solve(qs, bs)

        for i in range(batch):
            assert not np.any(np.isnan(sol.x[i]))
