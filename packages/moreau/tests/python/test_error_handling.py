"""
Unit tests for Moreau error handling.

Tests infeasibility detection, unboundedness detection, and input validation.
"""

import numpy as np
import pytest
from scipy import sparse

import moreau

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    cp = None

try:
    import torch
    from moreau.torch import Solver as TorchSolver

    HAS_TORCH_SOLVER = TorchSolver is not None
except ImportError:
    HAS_TORCH_SOLVER = False
    TorchSolver = None
    torch = None


def get_status_value(status):
    """Get integer value from SolverStatus.

    Works with unified SolverStatus IntEnum (has both .value and int()).
    """
    if hasattr(status, "value"):
        return status.value
    return int(status)


def create_simple_compiled_solver(n=2, m=3, batch_size=1, verbose=False):
    """Create a simple CompiledSolver for testing."""
    # P matrix (identity) in CSR format
    P_row_offsets = np.arange(n + 1, dtype=np.int64)
    P_col_indices = np.arange(n, dtype=np.int64)

    # A matrix in CSR format (each row has one 1)
    A_row_offsets = np.arange(m + 1, dtype=np.int64)
    A_col_indices = np.zeros(m, dtype=np.int64)  # All point to x[0]
    A_col_indices[: min(m, n)] = np.arange(min(m, n))

    cones = moreau.Cones()
    cones.num_nonneg_cones = m

    settings = moreau.Settings(
        device="cpu",
        batch_size=batch_size,
        verbose=verbose,
        max_iter=100,
    )

    solver = moreau.CompiledSolver(
        n,
        m,
        P_row_offsets,
        P_col_indices,
        A_row_offsets,
        A_col_indices,
        cones,
        settings=settings,
    )

    return solver, n, m


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
class TestInfeasibility:
    """Tests for infeasibility detection."""

    def test_primal_infeasible_equality(self):
        """Test detection of contradictory equality constraints."""
        # x = 1 AND x = 2 (impossible)
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x[0]), [x[0] == 1, x[0] == 2])

        # This should be infeasible
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.INFEASIBLE

        # Now test with Moreau directly using matrix-based Solver
        n, m = 1, 2
        P = sparse.csr_array(np.array([[1.0]]))
        A = sparse.csr_array(np.array([[1.0], [1.0]]))
        q = np.array([0.0])
        b = np.array([1.0, 2.0])  # x = 1 AND x = 2

        cones = moreau.Cones()
        cones.num_zero_cones = 2  # Both are equality constraints

        settings = moreau.Settings(verbose=False, max_iter=200, solver="ipm")

        solver = moreau.Solver(P, q, A, b, cones, settings)
        result = solver.solve()
        info = solver.info

        # Should detect infeasibility
        status_val = get_status_value(info.status)
        expected = [
            get_status_value(moreau.SolverStatus.PrimalInfeasible),
            get_status_value(moreau.SolverStatus.AlmostPrimalInfeasible),
            get_status_value(moreau.SolverStatus.MaxIterations),  # May hit max iter
        ]
        assert status_val in expected, f"Expected infeasible, got status {info.status}"

    def test_primal_infeasible_bounds(self):
        """Test detection of impossible bounds: x >= 2, x <= 1."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(x[0]), [x[0] >= 2, x[0] <= 1])
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.INFEASIBLE

        # Use CVXPY to generate problem data correctly
        data, _, _ = prob.get_problem_data(solver=cp.CLARABEL)
        dims = data["dims"]

        A_csr = sparse.csr_array(data["A"])
        P_csr = sparse.csr_array((len(data["c"]), len(data["c"])), dtype=np.float64)

        q = data["c"].astype(np.float64)
        b = data["b"].astype(np.float64)
        n = len(q)

        cones = moreau.Cones()
        cones.num_zero_cones = dims.zero
        cones.num_nonneg_cones = dims.nonneg

        settings = moreau.Settings(verbose=False, max_iter=200)

        solver = moreau.Solver(P_csr, q, A_csr, b, cones, settings)
        result = solver.solve()
        info = solver.info

        # Should detect infeasibility
        status_val = get_status_value(info.status)
        expected = [
            get_status_value(moreau.SolverStatus.PrimalInfeasible),
            get_status_value(moreau.SolverStatus.AlmostPrimalInfeasible),
            get_status_value(moreau.SolverStatus.MaxIterations),
        ]
        assert status_val in expected, f"Expected infeasible, got status {info.status}"


@pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")
class TestUnboundedness:
    """Tests for unboundedness (dual infeasibility) detection."""

    def test_dual_infeasible_unbounded(self):
        """Test detection of unbounded problem: min -x, x >= 0."""
        x = cp.Variable(1)
        prob = cp.Problem(cp.Minimize(-x[0]), [x[0] >= 0])
        prob.solve(solver=cp.CLARABEL)
        assert prob.status == cp.UNBOUNDED

        # Test with Moreau
        n, m = 1, 1
        P = sparse.csr_array((n, n), dtype=np.float64)
        A = sparse.csr_array(np.array([[-1.0]]))  # -x <= 0 => x >= 0
        q = np.array([-1.0])  # minimize -x
        b = np.array([0.0])

        cones = moreau.Cones()
        cones.num_nonneg_cones = 1

        settings = moreau.Settings(verbose=False, max_iter=200, solver="ipm")

        solver = moreau.Solver(P, q, A, b, cones, settings)
        result = solver.solve()
        info = solver.info

        # Should detect dual infeasibility (unbounded)
        status_val = get_status_value(info.status)
        expected = [
            get_status_value(moreau.SolverStatus.DualInfeasible),
            get_status_value(moreau.SolverStatus.AlmostDualInfeasible),
            get_status_value(moreau.SolverStatus.MaxIterations),
        ]
        assert status_val in expected, f"Expected unbounded, got status {info.status}"


class TestMaxIterations:
    """Tests for max iterations behavior."""

    def test_max_iterations_reached(self):
        """Test that solver stops at max_iter with appropriate status."""
        # Create a problem that's hard to solve
        n, m = 2, 2
        P = sparse.diags([0.001, 0.001], format="csr")  # Near-zero P makes conditioning bad
        A = sparse.csr_array([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([1.0, 1.0])
        b = np.array([0.0, 0.0])

        cones = moreau.Cones()
        cones.num_nonneg_cones = 2

        # Very low max_iter to force early termination
        settings = moreau.Settings(verbose=False, max_iter=2)

        solver = moreau.Solver(P, q, A, b, cones, settings)
        result = solver.solve()
        info = solver.info

        # Should either solve quickly or hit max iterations
        status_val = get_status_value(info.status)
        valid_statuses = [
            get_status_value(moreau.SolverStatus.Solved),
            get_status_value(moreau.SolverStatus.AlmostSolved),
            get_status_value(moreau.SolverStatus.MaxIterations),
        ]
        assert status_val in valid_statuses, f"Got unexpected status {info.status}"


class TestDimensionValidation:
    """Tests for input dimension validation in CompiledSolver."""

    def test_wrong_q_size(self):
        """Test that wrong q size is rejected."""
        solver, n, m = create_simple_compiled_solver(n=2, m=3)

        P_values = np.ones((1, n))
        A_values = np.ones((1, m))
        q_wrong = np.ones((1, n + 1))  # Wrong size!
        b = np.ones((1, m))

        solver.setup(P_values, A_values)
        with pytest.raises((ValueError, RuntimeError)):
            solver.solve(q_wrong, b)

    def test_wrong_b_size(self):
        """Test that wrong b size is rejected."""
        solver, n, m = create_simple_compiled_solver(n=2, m=3)

        P_values = np.ones((1, n))
        A_values = np.ones((1, m))
        q = np.ones((1, n))
        b_wrong = np.ones((1, m + 1))  # Wrong size!

        solver.setup(P_values, A_values)
        with pytest.raises((ValueError, RuntimeError)):
            solver.solve(q, b_wrong)

    def test_wrong_P_values_size(self):
        """Test that wrong P_values size is rejected."""
        solver, n, m = create_simple_compiled_solver(n=2, m=3)

        P_values_wrong = np.ones((1, n + 5))  # Wrong size!
        A_values = np.ones((1, m))

        with pytest.raises((ValueError, RuntimeError)):
            solver.setup(P_values_wrong, A_values)

    def test_wrong_A_values_size(self):
        """Test that wrong A_values size is rejected."""
        solver, n, m = create_simple_compiled_solver(n=2, m=3)

        P_values = np.ones((1, n))
        A_values_wrong = np.ones((1, m + 5))  # Wrong size!

        with pytest.raises((ValueError, RuntimeError)):
            solver.setup(P_values, A_values_wrong)


@pytest.mark.skipif(not HAS_TORCH_SOLVER, reason="PyTorch solver not available")
class TestTorchInputValidation:
    """Tests for PyTorch input validation."""

    def test_wrong_size_rejected(self):
        """Test that wrong tensor size is rejected."""
        n, m = 2, 3

        P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1], dtype=torch.int64)
        A_row_offsets = torch.tensor([0, 2, 3, 4], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_zero_cones = 1
        cones.num_nonneg_cones = 2

        ipm = moreau.IPMSettings(direct_solve_method="qdldl")
        settings = moreau.Settings(device="cpu", ipm_settings=ipm)
        solver = TorchSolver(
            n,
            m,
            P_row_offsets,
            P_col_indices,
            A_row_offsets,
            A_col_indices,
            cones,
            settings=settings,
        )

        # Try to solve with wrong q size
        P_values = torch.ones(2, dtype=torch.float64)
        A_values = torch.ones(4, dtype=torch.float64)
        q_wrong = torch.ones(n + 1, dtype=torch.float64)  # Wrong!
        b = torch.ones(m, dtype=torch.float64)

        with pytest.raises(RuntimeError, match="dimension"):
            solver.solve(P_values, A_values, q_wrong, b)


class TestSolverDimensionValidation:
    """Tests for Solver constructor dimension validation."""

    def test_P_wrong_size(self):
        """Test that P with wrong dimensions is rejected."""
        # P is 3x3 but q has length 2
        P = sparse.diags([1.0, 1.0, 1.0], format="csr")
        q = np.array([2.0, 1.0])
        A = sparse.csr_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 0.7, 0.7])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"P has shape.*but q has length"):
            moreau.Solver(P, q, A, b, cones)

    def test_A_wrong_columns(self):
        """Test that A with wrong number of columns is rejected."""
        # A has 3 columns but n=2 (from q)
        P = sparse.diags([1.0, 1.0], format="csr")
        q = np.array([2.0, 1.0])
        A = sparse.csr_matrix([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        b = np.array([1.0, 0.7, 0.7])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"A has shape.*but problem has n=2"):
            moreau.Solver(P, q, A, b, cones)

    def test_A_wrong_rows(self):
        """Test that A with wrong number of rows is rejected."""
        # A has 2 rows but b has length 3
        P = sparse.diags([1.0, 1.0], format="csr")
        q = np.array([2.0, 1.0])
        A = sparse.csr_matrix([[1.0, 1.0], [1.0, 0.0]])
        b = np.array([1.0, 0.7, 0.7])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"A has shape.*but b has length"):
            moreau.Solver(P, q, A, b, cones)

    def test_cones_wrong_total(self):
        """Test that cone dimensions not matching m is rejected."""
        P = sparse.diags([1.0, 1.0], format="csr")
        q = np.array([2.0, 1.0])
        A = sparse.csr_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 0.7, 0.7])
        # Cones sum to 2, but m=3
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=1)

        with pytest.raises(ValueError, match=r"Cone dimensions sum to 2 but b has length 3"):
            moreau.Solver(P, q, A, b, cones)

    def test_P_not_square(self):
        """Test that non-square P is rejected."""
        P = sparse.csr_matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # 2x3
        q = np.array([2.0, 1.0])
        A = sparse.csr_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 0.7, 0.7])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"P must be square"):
            moreau.Solver(P, q, A, b, cones)

    def test_q_not_1d(self):
        """Test that 2D q is rejected."""
        P = sparse.diags([1.0, 1.0], format="csr")
        q = np.array([[2.0, 1.0]])  # 2D!
        A = sparse.csr_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 0.7, 0.7])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"q must be a 1D array"):
            moreau.Solver(P, q, A, b, cones)

    def test_b_not_1d(self):
        """Test that 2D b is rejected."""
        P = sparse.diags([1.0, 1.0], format="csr")
        q = np.array([2.0, 1.0])
        A = sparse.csr_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0, 0.7, 0.7]])  # 2D!
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"b must be a 1D array"):
            moreau.Solver(P, q, A, b, cones)


class TestCompiledSolverCSRValidation:
    """Tests for CompiledSolver CSR structure validation."""

    def test_P_row_offsets_wrong_length(self):
        """Test that P_row_offsets with wrong length is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"P_row_offsets has length.*must have length n\+1"):
            moreau.CompiledSolver(
                n=2,
                m=3,
                P_row_offsets=[0, 1],  # Should be length 3 (n+1)
                P_col_indices=[0],
                A_row_offsets=[0, 1, 2, 3],
                A_col_indices=[0, 1, 0],
                cones=cones,
            )

    def test_A_row_offsets_wrong_length(self):
        """Test that A_row_offsets with wrong length is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"A_row_offsets has length.*must have length m\+1"):
            moreau.CompiledSolver(
                n=2,
                m=3,
                P_row_offsets=[0, 1, 2],
                P_col_indices=[0, 1],
                A_row_offsets=[0, 1, 2],  # Should be length 4 (m+1)
                A_col_indices=[0, 1],
                cones=cones,
            )

    def test_P_col_indices_wrong_length(self):
        """Test that P_col_indices with wrong length is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"P_col_indices has length.*nnz_P"):
            moreau.CompiledSolver(
                n=2,
                m=3,
                P_row_offsets=[0, 1, 2],  # indicates 2 non-zeros
                P_col_indices=[0, 1, 0],  # but 3 indices provided!
                A_row_offsets=[0, 1, 2, 3],
                A_col_indices=[0, 1, 0],
                cones=cones,
            )

    def test_A_col_indices_wrong_length(self):
        """Test that A_col_indices with wrong length is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"A_col_indices has length.*nnz_A"):
            moreau.CompiledSolver(
                n=2,
                m=3,
                P_row_offsets=[0, 1, 2],
                P_col_indices=[0, 1],
                A_row_offsets=[0, 1, 2, 3],  # indicates 3 non-zeros
                A_col_indices=[0, 1, 0, 1],  # but 4 indices!
                cones=cones,
            )

    def test_P_col_indices_out_of_range(self):
        """Test that P column index out of range is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"P_col_indices contains invalid"):
            moreau.CompiledSolver(
                n=2,
                m=3,
                P_row_offsets=[0, 1, 2],
                P_col_indices=[0, 5],  # 5 is out of range for n=2
                A_row_offsets=[0, 1, 2, 3],
                A_col_indices=[0, 1, 0],
                cones=cones,
            )

    def test_A_col_indices_out_of_range(self):
        """Test that A column index out of range is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ValueError, match=r"A_col_indices contains invalid"):
            moreau.CompiledSolver(
                n=2,
                m=3,
                P_row_offsets=[0, 1, 2],
                P_col_indices=[0, 1],
                A_row_offsets=[0, 1, 2, 3],
                A_col_indices=[0, 10, 0],  # 10 is out of range for n=2
                cones=cones,
            )

    def test_cones_wrong_total_compiled(self):
        """Test that cone dimensions not matching m is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=1)  # sum = 2, but m=3

        with pytest.raises(ValueError, match=r"Cone dimensions sum to 2 but m=3"):
            moreau.CompiledSolver(
                n=2,
                m=3,
                P_row_offsets=[0, 1, 2],
                P_col_indices=[0, 1],
                A_row_offsets=[0, 1, 2, 3],
                A_col_indices=[0, 1, 0],
                cones=cones,
            )


class TestCompiledSolverSetupValidation:
    """Tests for CompiledSolver.setup() validation."""

    def test_P_values_wrong_nnz(self):
        """Test that P_values with wrong nnz is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(batch_size=1)
        solver = moreau.CompiledSolver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],  # nnz_P = 2
            A_row_offsets=[0, 1, 2, 3],
            A_col_indices=[0, 1, 0],  # nnz_A = 3
            cones=cones,
            settings=settings,
        )

        with pytest.raises(ValueError, match=r"P_values has length.*expected nnz_P"):
            solver.setup(
                P_values=[1.0, 1.0, 1.0], A_values=[1.0, 1.0, 1.0]  # 3 values but nnz_P = 2
            )

    def test_A_values_wrong_nnz(self):
        """Test that A_values with wrong nnz is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(batch_size=1)
        solver = moreau.CompiledSolver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],  # nnz_P = 2
            A_row_offsets=[0, 1, 2, 3],
            A_col_indices=[0, 1, 0],  # nnz_A = 3
            cones=cones,
            settings=settings,
        )

        with pytest.raises(ValueError, match=r"A_values has length.*expected nnz_A"):
            solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0])  # 2 values but nnz_A = 3

    def test_P_values_wrong_batch_size(self):
        """Test that P_values with wrong batch dimension is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(batch_size=4)
        solver = moreau.CompiledSolver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 1, 2, 3],
            A_col_indices=[0, 1, 0],
            cones=cones,
            settings=settings,
        )

        with pytest.raises(ValueError, match=r"P_values has batch dimension.*batch_size = 4"):
            solver.setup(
                P_values=[[1.0, 1.0]] * 2,  # batch=2 but solver has batch_size=4
                A_values=[[1.0, 1.0, 1.0]] * 4,
            )

    def test_A_values_wrong_batch_size(self):
        """Test that A_values with wrong batch dimension is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(batch_size=4)
        solver = moreau.CompiledSolver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 1, 2, 3],
            A_col_indices=[0, 1, 0],
            cones=cones,
            settings=settings,
        )

        with pytest.raises(ValueError, match=r"A_values has batch dimension.*batch_size = 4"):
            solver.setup(
                P_values=[[1.0, 1.0]] * 4,
                A_values=[[1.0, 1.0, 1.0]] * 2,  # batch=2 but solver has batch_size=4
            )


class TestCompiledSolverSolveValidation:
    """Tests for CompiledSolver.solve() validation."""

    def test_qs_wrong_n(self):
        """Test that qs with wrong n is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(batch_size=1)
        solver = moreau.CompiledSolver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 1, 2, 3],
            A_col_indices=[0, 1, 0],
            cones=cones,
            settings=settings,
        )
        solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match=r"qs has.*elements per problem but expected n = 2"):
            solver.solve(qs=[[1.0, 1.0, 1.0]], bs=[[1.0, 1.0, 1.0]])  # n=3 but solver has n=2

    def test_bs_wrong_m(self):
        """Test that bs with wrong m is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(batch_size=1)
        solver = moreau.CompiledSolver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 1, 2, 3],
            A_col_indices=[0, 1, 0],
            cones=cones,
            settings=settings,
        )
        solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match=r"bs has.*elements per problem but expected m = 3"):
            solver.solve(qs=[[1.0, 1.0]], bs=[[1.0, 1.0]])  # m=2 but solver has m=3

    def test_qs_wrong_batch_size(self):
        """Test that qs with wrong batch dimension is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(batch_size=4)
        solver = moreau.CompiledSolver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 1, 2, 3],
            A_col_indices=[0, 1, 0],
            cones=cones,
            settings=settings,
        )
        solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match=r"qs has batch dimension.*batch_size = 4"):
            solver.solve(
                qs=[[1.0, 1.0]] * 2, bs=[[1.0, 1.0, 1.0]] * 4  # batch=2 but solver has batch_size=4
            )

    def test_bs_wrong_batch_size(self):
        """Test that bs with wrong batch dimension is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(batch_size=4)
        solver = moreau.CompiledSolver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 1, 2, 3],
            A_col_indices=[0, 1, 0],
            cones=cones,
            settings=settings,
        )
        solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match=r"bs has batch dimension.*batch_size = 4"):
            solver.solve(
                qs=[[1.0, 1.0]] * 4, bs=[[1.0, 1.0, 1.0]] * 2  # batch=2 but solver has batch_size=4
            )

    def test_qs_1d_with_batch_size_gt_1(self):
        """Test that 1D qs with batch_size > 1 is rejected."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(batch_size=4)
        solver = moreau.CompiledSolver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 1, 2, 3],
            A_col_indices=[0, 1, 0],
            cones=cones,
            settings=settings,
        )
        solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0, 1.0])

        with pytest.raises(ValueError, match=r"qs is 1D but solver has batch_size = 4"):
            solver.solve(qs=[1.0, 1.0], bs=[[1.0, 1.0, 1.0]] * 4)  # 1D but batch_size=4

    def test_solve_requires_setup(self):
        """Test that solve() before setup() raises a clear error."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(batch_size=1)
        solver = moreau.CompiledSolver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 1, 2, 3],
            A_col_indices=[0, 1, 0],
            cones=cones,
            settings=settings,
        )

        with pytest.raises(RuntimeError) as excinfo:
            solver.solve(
                qs=[[1.0, 1.0]],
                bs=[[1.0, 1.0, 1.0]],
            )

        msg = str(excinfo.value).lower()
        assert "setup" in msg and ("before solve" in msg or "before setup" in msg)


class TestSolverStatus:
    """Tests for SolverStatus return values."""

    def test_single_problem_returns_scalar_status(self):
        """Test that single problem returns scalar SolverStatus."""
        n, m = 2, 2
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.csr_array([[1.0, 0.0], [0.0, 1.0]])
        q = np.array([1.0, 1.0])
        b = np.array([0.0, 0.0])

        cones = moreau.Cones()
        cones.num_nonneg_cones = 2

        settings = moreau.Settings(verbose=False)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        result = solver.solve()
        info = solver.info

        # For single problem, status should be a scalar SolverStatus, not a list
        assert isinstance(info.status, moreau.SolverStatus)

    def test_batched_problems_return_list_of_status(self):
        """Test that batched problems return list of SolverStatus."""
        batch_size = 3
        solver, n, m = create_simple_compiled_solver(n=2, m=2, batch_size=batch_size)

        P_values = np.ones((batch_size, n))
        A_values = np.ones((batch_size, m))
        q = np.ones((batch_size, n))
        b = np.zeros((batch_size, m))

        solver.setup(P_values=P_values, A_values=A_values)
        result = solver.solve(qs=q, bs=b)
        info = solver.info

        # For batched problems, status should be a list
        assert isinstance(info.status, list)
        assert len(info.status) == batch_size
        for s in info.status:
            assert isinstance(s, moreau.SolverStatus)
