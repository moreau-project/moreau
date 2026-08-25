"""
Tests for input validation edge cases: NaN/Inf rejection, invalid cone
parameters, API misuse sequences, and type/format errors.
"""

import numpy as np
import pytest
from pydantic import ValidationError
from scipy import sparse

import moreau

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_solver_args(n=2, m=2):
    """Return (P, q, A, b, cones) for a simple feasible QP."""
    P = sparse.diags(np.ones(n), format="csr")
    A = sparse.eye(m, n, format="csr") if m <= n else sparse.eye(m, n, format="csr")
    q = np.ones(n)
    b = np.zeros(m)
    cones = moreau.Cones(num_nonneg_cones=m)
    return P, q, A, b, cones


def _simple_compiled_solver(n=2, m=2, batch_size=1):
    """Create and return a ready CompiledSolver."""
    P_ro = np.arange(n + 1, dtype=np.int64)
    P_ci = np.arange(n, dtype=np.int64)
    A_ro = np.arange(m + 1, dtype=np.int64)
    A_ci = np.minimum(np.arange(m, dtype=np.int64), n - 1)
    cones = moreau.Cones(num_nonneg_cones=m)

    settings = moreau.Settings(
        device="cpu",
        batch_size=batch_size,
        verbose=False,
        max_iter=100,
        solver="ipm",
    )
    solver = moreau.CompiledSolver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings=settings)
    return solver


# ---------------------------------------------------------------------------
# NaN / Inf in Solver inputs
# ---------------------------------------------------------------------------


class TestSolverNaNInf:
    """NaN and Inf in Solver constructor inputs."""

    def test_nan_in_q(self):
        """q containing NaN should either raise or produce NumericalError."""
        P, _, A, b, cones = _simple_solver_args()
        q = np.array([1.0, np.nan])
        try:
            solver = moreau.Solver(
                P, q, A, b, cones, moreau.Settings(verbose=False, max_iter=50, solver="ipm")
            )
            sol = solver.solve()
            # If it doesn't raise, solution must flag the issue
            assert solver.info.status in (
                moreau.SolverStatus.NumericalError,
                moreau.SolverStatus.MaxIterations,
            ) or np.any(np.isnan(sol.x))
        except (ValueError, RuntimeError):
            pass  # Raising is acceptable

    def test_inf_in_q(self):
        """q containing Inf should either raise or produce NumericalError."""
        P, _, A, b, cones = _simple_solver_args()
        q = np.array([1.0, np.inf])
        try:
            solver = moreau.Solver(
                P, q, A, b, cones, moreau.Settings(verbose=False, max_iter=50, solver="ipm")
            )
            sol = solver.solve()
            assert (
                solver.info.status
                in (
                    moreau.SolverStatus.NumericalError,
                    moreau.SolverStatus.MaxIterations,
                )
                or np.any(np.isnan(sol.x))
                or np.any(np.isinf(sol.x))
            )
        except (ValueError, RuntimeError):
            pass

    def test_nan_in_b(self):
        """b containing NaN should either raise or produce NumericalError."""
        P, q, A, _, cones = _simple_solver_args()
        b = np.array([np.nan, 0.0])
        try:
            solver = moreau.Solver(
                P, q, A, b, cones, moreau.Settings(verbose=False, max_iter=50, solver="ipm")
            )
            sol = solver.solve()
            assert solver.info.status in (
                moreau.SolverStatus.NumericalError,
                moreau.SolverStatus.MaxIterations,
            ) or np.any(np.isnan(sol.x))
        except (ValueError, RuntimeError):
            pass

    def test_inf_in_b(self):
        """b containing +Inf in nonneg cone should solve (vacuous constraint)."""
        P, q, A, _, cones = _simple_solver_args()
        b = np.array([np.inf, 0.0])
        solver = moreau.Solver(P, q, A, b, cones, moreau.Settings(verbose=False, max_iter=50))
        sol = solver.solve()
        # +inf in nonneg b is sanitized: A row zeroed, b set to 1.
        # The vacuous constraint is removed, solver finds the optimum.
        assert solver.info.status == moreau.SolverStatus.Solved
        assert not np.any(np.isnan(sol.x))
        assert not np.any(np.isinf(sol.x))


# ---------------------------------------------------------------------------
# NaN / Inf in CompiledSolver inputs
# ---------------------------------------------------------------------------


class TestCompiledSolverNaNInf:
    """NaN and Inf through the CompiledSolver path."""

    def test_nan_in_P_values(self):
        """P_values containing NaN."""
        solver = _simple_compiled_solver(n=2, m=2)
        try:
            solver.setup(P_values=[np.nan, 1.0], A_values=[1.0, 1.0])
            sol = solver.solve(qs=[[1.0, 1.0]], bs=[[0.0, 0.0]])
            assert solver.info.status[0] in (
                moreau.SolverStatus.NumericalError,
                moreau.SolverStatus.MaxIterations,
            ) or np.any(np.isnan(sol.x))
        except (ValueError, RuntimeError):
            pass

    def test_nan_in_A_values(self):
        """A_values containing NaN."""
        solver = _simple_compiled_solver(n=2, m=2)
        try:
            solver.setup(P_values=[1.0, 1.0], A_values=[np.nan, 1.0])
            sol = solver.solve(qs=[[1.0, 1.0]], bs=[[0.0, 0.0]])
            assert solver.info.status[0] in (
                moreau.SolverStatus.NumericalError,
                moreau.SolverStatus.MaxIterations,
            ) or np.any(np.isnan(sol.x))
        except (ValueError, RuntimeError):
            pass

    def test_nan_in_qs(self):
        """qs containing NaN."""
        solver = _simple_compiled_solver(n=2, m=2)
        solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0])
        try:
            sol = solver.solve(qs=[[np.nan, 1.0]], bs=[[0.0, 0.0]])
            assert solver.info.status[0] in (
                moreau.SolverStatus.NumericalError,
                moreau.SolverStatus.MaxIterations,
            ) or np.any(np.isnan(sol.x))
        except (ValueError, RuntimeError):
            pass

    def test_inf_in_qs(self):
        """qs containing Inf."""
        solver = _simple_compiled_solver(n=2, m=2)
        solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0])
        try:
            sol = solver.solve(qs=[[np.inf, 1.0]], bs=[[0.0, 0.0]])
            assert (
                solver.info.status[0]
                in (
                    moreau.SolverStatus.NumericalError,
                    moreau.SolverStatus.MaxIterations,
                )
                or np.any(np.isnan(sol.x))
                or np.any(np.isinf(sol.x))
            )
        except (ValueError, RuntimeError):
            pass


# ---------------------------------------------------------------------------
# Invalid cone parameters
# ---------------------------------------------------------------------------


class TestConeValidation:
    """Cone construction validation."""

    def test_power_alpha_zero(self):
        """power_alphas = 0 should be rejected."""
        with pytest.raises((ValueError, ValidationError)):
            moreau.Cones(power_alphas=[0.0])

    def test_power_alpha_one(self):
        """power_alphas = 1 should be rejected."""
        with pytest.raises((ValueError, ValidationError)):
            moreau.Cones(power_alphas=[1.0])

    def test_power_alpha_negative(self):
        """power_alphas < 0 should be rejected."""
        with pytest.raises((ValueError, ValidationError)):
            moreau.Cones(power_alphas=[-0.5])

    def test_power_alpha_greater_than_one(self):
        """power_alphas > 1 should be rejected."""
        with pytest.raises((ValueError, ValidationError)):
            moreau.Cones(power_alphas=[1.5])

    def test_negative_num_zero_cones(self):
        """Negative cone count should be rejected."""
        with pytest.raises((ValueError, ValidationError)):
            moreau.Cones(num_zero_cones=-1)

    def test_negative_num_nonneg_cones(self):
        """Negative nonneg count should be rejected."""
        with pytest.raises((ValueError, ValidationError)):
            moreau.Cones(num_nonneg_cones=-1)

    def test_valid_power_alpha(self):
        """Valid power alpha should work."""
        cones = moreau.Cones(power_alphas=[0.5])
        assert cones.num_power_cones == 1
        assert cones.total_constraints() == 3

    def test_multiple_valid_power_alphas(self):
        """Multiple valid power alphas."""
        cones = moreau.Cones(power_alphas=[0.1, 0.5, 0.9])
        assert cones.num_power_cones == 3
        assert cones.total_constraints() == 9

    def test_cone_total_constraints(self):
        """Verify total_constraints() math."""
        cones = moreau.Cones(
            num_zero_cones=2,
            num_nonneg_cones=3,
            so_cone_dims=[3],
            num_exp_cones=1,
            power_alphas=[0.5],
        )
        # 2 + 3 + 3*1 + 3*1 + 3*1 = 14
        assert cones.total_constraints() == 14


# ---------------------------------------------------------------------------
# API misuse sequences
# ---------------------------------------------------------------------------


class TestAPIMisuse:
    """Calling the API in incorrect order or with wrong types."""

    def test_solve_without_setup(self):
        """Calling solve() before setup() on CompiledSolver."""
        solver = _simple_compiled_solver(n=2, m=2)
        # No setup() called — solve should raise
        with pytest.raises((RuntimeError, AttributeError, Exception)):
            solver.solve(qs=[[1.0, 1.0]], bs=[[0.0, 0.0]])

    def test_double_setup(self):
        """Calling setup() twice should work (overwrite values)."""
        solver = _simple_compiled_solver(n=2, m=2)
        solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0])
        # Second setup should not raise
        solver.setup(P_values=[2.0, 2.0], A_values=[2.0, 2.0])
        sol = solver.solve(qs=[[1.0, 1.0]], bs=[[0.0, 0.0]])
        assert not np.any(np.isnan(sol.x))

    def test_solve_twice(self):
        """Calling solve() twice should work (independent solves)."""
        solver = _simple_compiled_solver(n=2, m=2)
        solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0])

        sol1 = solver.solve(qs=[[1.0, 1.0]], bs=[[0.0, 0.0]])
        sol2 = solver.solve(qs=[[2.0, 2.0]], bs=[[0.0, 0.0]])

        assert not np.any(np.isnan(sol1.x))
        assert not np.any(np.isnan(sol2.x))

    def test_warm_start_wrong_type(self):
        """Passing wrong type for warm_start should raise TypeError."""
        solver = _simple_compiled_solver(n=2, m=2)
        solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0])

        with pytest.raises(TypeError, match="warm_start must be"):
            solver.solve(qs=[[1.0, 1.0]], bs=[[0.0, 0.0]], warm_start="not a warm start")

    def test_warm_start_wrong_dimensions(self):
        """WarmStart with wrong x dimensions should raise ValueError."""
        solver = _simple_compiled_solver(n=2, m=2)
        solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0])

        # Create a warm start with wrong dimensions
        ws = moreau.WarmStart(
            x=np.zeros(5),  # wrong: n=2 but giving 5
            z=np.zeros(2),
            s=np.zeros(2),
        )
        with pytest.raises((ValueError, RuntimeError)):
            solver.solve(qs=[[1.0, 1.0]], bs=[[0.0, 0.0]], warm_start=ws)


# ---------------------------------------------------------------------------
# Type / format edge cases
# ---------------------------------------------------------------------------


class TestTypeConversions:
    """Verify behavior with non-standard input types."""

    def test_int_arrays_accepted(self):
        """Integer arrays for q, b should be auto-converted to float64."""
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.eye(2, format="csr")
        q = np.array([1, 1], dtype=np.int64)  # integer
        b = np.array([0, 0], dtype=np.int64)
        cones = moreau.Cones(num_nonneg_cones=2)

        # Should not raise — auto-conversion to float64
        solver = moreau.Solver(P, q, A, b, cones, moreau.Settings(verbose=False))
        sol = solver.solve()
        assert sol.x.dtype == np.float64

    def test_float32_arrays_accepted(self):
        """float32 arrays should be auto-converted to float64."""
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.eye(2, format="csr")
        q = np.array([1.0, 1.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        cones = moreau.Cones(num_nonneg_cones=2)

        solver = moreau.Solver(P, q, A, b, cones, moreau.Settings(verbose=False))
        sol = solver.solve()
        assert sol.x.dtype == np.float64

    def test_list_inputs_accepted(self):
        """Plain Python lists for q, b should work."""
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.eye(2, format="csr")
        q = [1.0, 1.0]  # plain list
        b = [0.0, 0.0]
        cones = moreau.Cones(num_nonneg_cones=2)

        solver = moreau.Solver(P, q, A, b, cones, moreau.Settings(verbose=False))
        sol = solver.solve()
        assert not np.any(np.isnan(sol.x))

    def test_csc_matrix_accepted(self):
        """CSC format matrices should be auto-converted to CSR."""
        P = sparse.diags([1.0, 1.0], format="csc")  # CSC, not CSR
        A = sparse.eye(2, format="csc")
        q = np.array([1.0, 1.0])
        b = np.array([0.0, 0.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        # Should auto-convert CSC to CSR internally
        solver = moreau.Solver(P, q, A, b, cones, moreau.Settings(verbose=False))
        sol = solver.solve()
        assert not np.any(np.isnan(sol.x))


# ---------------------------------------------------------------------------
# Settings validation
# ---------------------------------------------------------------------------


class TestSettingsValidation:
    """Edge cases in Settings construction."""

    def test_zero_max_iter(self):
        """max_iter=0 should be accepted (returns immediately)."""
        P, q, A, b, cones = _simple_solver_args()
        try:
            solver = moreau.Solver(P, q, A, b, cones, moreau.Settings(verbose=False, max_iter=0))
            sol = solver.solve()
            # Should hit max iterations immediately
            assert solver.info.status in (
                moreau.SolverStatus.MaxIterations,
                moreau.SolverStatus.Solved,
                moreau.SolverStatus.AlmostSolved,
            )
        except (ValueError, ValidationError):
            pass  # Rejecting 0 is also acceptable

    def test_negative_max_iter_rejected(self):
        """max_iter < 0 should be rejected."""
        with pytest.raises((ValueError, ValidationError)):
            moreau.Settings(max_iter=-1)

    def test_invalid_device_string(self):
        """Invalid device name should raise at solver construction."""
        P, q, A, b, cones = _simple_solver_args()
        with pytest.raises((RuntimeError, ValueError)):
            moreau.Solver(P, q, A, b, cones, moreau.Settings(device="nonexistent_device"))


# ---------------------------------------------------------------------------
# P symmetry validation
# ---------------------------------------------------------------------------


class TestPSymmetryValidation:
    """P must be full symmetric (both upper and lower triangles)."""

    def test_upper_triangular_P_rejected(self):
        """P with only upper triangle should be rejected."""
        # Upper triangular only: [[1, 2], [0, 3]]
        P = sparse.csr_matrix(np.array([[1.0, 2.0], [0.0, 3.0]]))
        A = sparse.eye(2, format="csr")
        q = np.array([1.0, 1.0])
        b = np.array([0.0, 0.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"[Ss]ymmetr"):
            moreau.Solver(P, q, A, b, cones)

    def test_full_symmetric_P_accepted(self):
        """P with both triangles should be accepted."""
        P = sparse.csr_matrix(np.array([[1.0, 2.0], [2.0, 3.0]]))
        A = sparse.eye(2, format="csr")
        q = np.array([1.0, 1.0])
        b = np.array([0.0, 0.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        # Should not raise
        solver = moreau.Solver(P, q, A, b, cones, moreau.Settings(verbose=False))
        sol = solver.solve()
        assert not np.any(np.isnan(sol.x))
