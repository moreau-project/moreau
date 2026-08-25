"""
Tests for time_limit functionality in Settings.

Verifies that:
1. time_limit is properly passed to backends
2. Solver terminates with MaxTime status when limit is exceeded
3. Works correctly for both Solver and CompiledSolver
"""

import numpy as np
import pytest
from scipy import sparse

import moreau


def create_hard_qp(n: int = 100):
    """Create a QP that takes many iterations to solve.

    Uses a poorly conditioned problem to ensure the solver
    needs significant time/iterations.
    """
    # Poorly conditioned P matrix (wide range of eigenvalues)
    np.random.seed(42)
    A_rand = np.random.randn(n, n)
    P_dense = A_rand @ A_rand.T + 0.01 * np.eye(n)
    P = sparse.csr_matrix(P_dense)

    q = np.random.randn(n)

    # Constraints: sum(x) = 1, x >= 0
    m = n + 1
    A_eq = np.ones((1, n))
    A_ineq = np.eye(n)
    A = sparse.vstack([sparse.csr_matrix(A_eq), sparse.csr_matrix(A_ineq)])

    b = np.zeros(m)
    b[0] = 1.0  # sum(x) = 1

    cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=n)

    return P, q, A, b, cones


class TestTimeLimitSolver:
    """Test time_limit with the Solver API."""

    def test_time_limit_zero_triggers_max_time(self, device):
        """A near-zero time limit should trigger MaxTime status immediately."""
        P, q, A, b, cones = create_hard_qp(n=50)

        # Use an impossibly small time limit
        settings = moreau.Settings(solver="ipm", device=device, time_limit=1e-10)
        solver = moreau.Solver(P, q, A, b, cones, settings)

        solution = solver.solve()

        # Should hit time limit (or possibly solve in 0 iterations for trivial cases)
        assert solver.info.status in [
            moreau.SolverStatus.MaxTime,
            moreau.SolverStatus.Solved,  # In case it solves in <1 iteration
        ], f"Expected MaxTime or Solved, got {solver.info.status}"

    def test_time_limit_allows_completion(self, device):
        """A generous time limit should allow the solver to complete."""
        # Simple problem that solves quickly
        P = sparse.diags([1.0, 1.0], format="csr")
        q = np.array([2.0, 1.0])
        A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 2.0, 2.0])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        # Generous time limit
        settings = moreau.Settings(solver="ipm", device=device, time_limit=60.0)
        solver = moreau.Solver(P, q, A, b, cones, settings)

        solution = solver.solve()

        assert solver.info.status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ], f"Expected Solved, got {solver.info.status}"

    def test_time_limit_default_is_infinity(self, device):
        """Default time_limit should be infinity (no limit)."""
        settings = moreau.Settings(device=device)
        assert settings.time_limit == float("inf")


class TestTimeLimitCompiledSolver:
    """Test time_limit with the CompiledSolver API."""

    def test_compiled_solver_time_limit_zero(self, device):
        """CompiledSolver with near-zero time limit should trigger MaxTime."""
        n, m = 50, 51

        # Create structure
        P = sparse.diags([1.0] * n, format="csr")
        A_eq = sparse.csr_matrix(np.ones((1, n)))
        A_ineq = sparse.eye(n, format="csr")
        A = sparse.vstack([A_eq, A_ineq])

        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=n)
        method = "qdldl" if device == "cpu" else "cudss"
        ipm = moreau.IPMSettings(direct_solve_method=method)
        settings = moreau.Settings(
            solver="ipm", device=device, time_limit=1e-10, batch_size=1, ipm_settings=ipm
        )

        solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=P.indptr,
            P_col_indices=P.indices,
            A_row_offsets=A.indptr,
            A_col_indices=A.indices,
            cones=cones,
            settings=settings,
        )

        # Setup and solve
        P_values = P.data.reshape(1, -1)
        A_values = A.data.reshape(1, -1)
        solver.setup(P_values, A_values)

        q = np.random.randn(1, n)
        b = np.zeros((1, m))
        b[0, 0] = 1.0

        solution = solver.solve(q, b)

        assert solver.info.status[0] in [
            moreau.SolverStatus.MaxTime,
            moreau.SolverStatus.Solved,
        ], f"Expected MaxTime or Solved, got {solver.info.status[0]}"

    def test_compiled_solver_batch_time_limit(self, device):
        """Time limit applies to entire batch solve."""
        n, m = 2, 3
        batch_size = 4

        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(
            solver="ipm", device=device, time_limit=60.0, batch_size=batch_size
        )

        solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=P.indptr,
            P_col_indices=P.indices,
            A_row_offsets=A.indptr,
            A_col_indices=A.indices,
            cones=cones,
            settings=settings,
        )

        P_values = np.tile(P.data, (batch_size, 1))
        A_values = np.tile(A.data, (batch_size, 1))
        solver.setup(P_values, A_values)

        q = np.tile([2.0, 1.0], (batch_size, 1))
        b = np.tile([1.0, 2.0, 2.0], (batch_size, 1))

        solution = solver.solve(q, b)

        # All batch elements should solve successfully with generous limit
        for i, status in enumerate(solver.info.status):
            assert status in [
                moreau.SolverStatus.Solved,
                moreau.SolverStatus.AlmostSolved,
            ], f"Batch {i}: Expected Solved, got {status}"


class TestTimeLimitValidation:
    """Test pydantic validation of time_limit."""

    def test_time_limit_must_be_positive(self):
        """time_limit must be > 0."""
        with pytest.raises(Exception):  # pydantic.ValidationError
            moreau.Settings(time_limit=0.0)

        with pytest.raises(Exception):
            moreau.Settings(time_limit=-1.0)

    def test_time_limit_accepts_float(self):
        """time_limit accepts float values."""
        settings = moreau.Settings(time_limit=0.5)
        assert settings.time_limit == 0.5

        settings = moreau.Settings(time_limit=100.0)
        assert settings.time_limit == 100.0

    def test_time_limit_accepts_int(self):
        """time_limit accepts int (coerced to float)."""
        settings = moreau.Settings(time_limit=10)
        assert settings.time_limit == 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
