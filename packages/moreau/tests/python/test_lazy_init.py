"""
Unit tests for Solver and CompiledSolver initialization behavior.

Tests verify:
1. Solver is for single problems with matrix-based interface
2. CompiledSolver handles batched problems with pattern-based interface
3. PyTorch solver handles variable batch sizes
"""

import numpy as np
import pytest
from scipy import sparse

import moreau

try:
    import torch
    from moreau.torch import Solver as TorchSolver

    HAS_TORCH_SOLVER = TorchSolver is not None
except ImportError:
    HAS_TORCH_SOLVER = False
    TorchSolver = None
    torch = None


# ============================================================================
# Solver Tests (Single Problem - Matrix-based)
# ============================================================================


class TestSolverSingleProblem:
    """Tests for Solver (single problem, matrix-based interface)."""

    @pytest.fixture
    def simple_qp(self):
        """Create a simple QP problem with scipy sparse matrices."""
        n, m = 2, 3

        # P matrix (2x2 identity)
        P = sparse.diags([1.0, 1.0], format="csr")

        # A matrix
        A = sparse.csr_array(
            [
                [1.0, 1.0],  # equality
                [1.0, 0.0],  # inequality
                [0.0, 1.0],  # inequality
            ]
        )

        q = np.array([2.0, 1.0])
        b = np.array([1.0, 2.0, 2.0])

        # Cone structure
        cones = moreau.Cones()
        cones.num_zero_cones = 1
        cones.num_nonneg_cones = 2

        return {
            "n": n,
            "m": m,
            "P": P,
            "q": q,
            "A": A,
            "b": b,
            "cones": cones,
        }

    def test_solver_single_problem(self, simple_qp):
        """Test that Solver works for single problems."""
        p = simple_qp
        settings = moreau.Settings(verbose=False)

        # Create solver with matrix-based interface
        solver = moreau.Solver(
            p["P"],
            p["q"],
            p["A"],
            p["b"],
            p["cones"],
            settings,
        )

        result = solver.solve()
        info = solver.info

        # Output should be 1D for single problem
        assert result.x.shape == (p["n"],)
        assert info.status in [moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved]

    def test_solver_eager_init(self, simple_qp):
        """Test that Solver is eagerly initialized."""
        p = simple_qp
        settings = moreau.Settings(device="cpu", verbose=False)

        solver = moreau.Solver(
            p["P"],
            p["q"],
            p["A"],
            p["b"],
            p["cones"],
            settings,
        )

        # Should have basic properties set
        assert solver.n == p["n"]
        assert solver.m == p["m"]
        assert solver.device == "cpu"


# ============================================================================
# CompiledSolver Tests (Multiple Problems - Pattern-based)
# ============================================================================


class TestCompiledSolverMultipleProblems:
    """Tests for CompiledSolver (multiple problems, pattern-based interface)."""

    @pytest.fixture
    def problem_structure(self):
        """Create problem structure for CompiledSolver."""
        n, m = 2, 3

        # P matrix (2x2 identity) in CSR format
        P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
        P_col_indices = np.array([0, 1], dtype=np.int64)

        # A matrix in CSR format
        A_row_offsets = np.array([0, 2, 3, 4], dtype=np.int64)
        A_col_indices = np.array([0, 1, 0, 1], dtype=np.int64)

        # Cone structure
        cones = moreau.Cones()
        cones.num_zero_cones = 1
        cones.num_nonneg_cones = 2

        return {
            "n": n,
            "m": m,
            "P_row_offsets": P_row_offsets,
            "P_col_indices": P_col_indices,
            "A_row_offsets": A_row_offsets,
            "A_col_indices": A_col_indices,
            "cones": cones,
            "nnzP": 2,
            "nnzA": 4,
        }

    def test_compiled_solver_multiple_problems(self, problem_structure):
        """Test that CompiledSolver works for batched problems."""
        p = problem_structure
        batch_size = 4
        settings = moreau.Settings(verbose=False)

        # Create CompiledSolver with pattern-based interface
        solver = moreau.CompiledSolver(
            p["n"],
            p["m"],
            p["P_row_offsets"],
            p["P_col_indices"],
            p["A_row_offsets"],
            p["A_col_indices"],
            p["cones"],
            settings=moreau.Settings(verbose=False, batch_size=batch_size),
        )

        # 2D arrays (batched problem)
        P_values = np.ones((batch_size, p["nnzP"]))
        A_values = np.ones((batch_size, p["nnzA"]))
        q = np.ones((batch_size, p["n"])) * 2.0
        b = np.ones((batch_size, p["m"]))

        solver.setup(P_values=P_values, A_values=A_values)
        result = solver.solve(qs=q, bs=b)
        info = solver.info

        # Output should be 2D for batched problem
        assert result.x.shape == (batch_size, p["n"])

    def test_compiled_solver_dimensions(self, problem_structure):
        """Test CompiledSolver dimension reporting."""
        p = problem_structure
        batch_size = 4
        settings = moreau.Settings(verbose=False, batch_size=batch_size)

        solver = moreau.CompiledSolver(
            p["n"],
            p["m"],
            p["P_row_offsets"],
            p["P_col_indices"],
            p["A_row_offsets"],
            p["A_col_indices"],
            p["cones"],
            settings=settings,
        )

        assert solver.n == p["n"]
        assert solver.m == p["m"]


# ============================================================================
# PyTorch Tests (Dynamic Batch Sizes)
# ============================================================================


@pytest.mark.skipif(not HAS_TORCH_SOLVER, reason="PyTorch solver not available")
class TestTorchDynamicBatch:
    """Tests for PyTorch dynamic batch size support."""

    @pytest.fixture
    def problem_structure(self):
        """Create problem structure."""
        # Use the default device from moreau settings
        device = moreau.default_device()

        n, m = 2, 3

        # P matrix (2x2 identity) in CSR format
        P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1], dtype=torch.int64)

        # A matrix in CSR format
        A_row_offsets = torch.tensor([0, 2, 3, 4], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        # Cone structure
        cones = moreau.Cones()
        cones.num_zero_cones = 1
        cones.num_nonneg_cones = 2

        return {
            "n": n,
            "m": m,
            "P_row_offsets": P_row_offsets,
            "P_col_indices": P_col_indices,
            "A_row_offsets": A_row_offsets,
            "A_col_indices": A_col_indices,
            "cones": cones,
            "nnzP": 2,
            "nnzA": 4,
            "device": device,
        }

    def test_solver_dynamic_batch_sizes(self, problem_structure):
        """Test that PyTorch Solver handles variable batch sizes automatically."""
        p = problem_structure
        device = p["device"]
        settings = moreau.Settings(verbose=False)

        # Create solver (no batch_size needed - cvxpylayers-style)
        solver = TorchSolver(
            p["n"],
            p["m"],
            p["P_row_offsets"],
            p["P_col_indices"],
            p["A_row_offsets"],
            p["A_col_indices"],
            p["cones"],
            settings=settings,
        )

        # Solve with batch_size=4
        batch_size1 = 4
        P_values = torch.ones(batch_size1, p["nnzP"], dtype=torch.float64, device=device)
        A_values = torch.ones(batch_size1, p["nnzA"], dtype=torch.float64, device=device)
        q = torch.ones(batch_size1, p["n"], dtype=torch.float64, device=device) * 2.0
        b = torch.ones(batch_size1, p["m"], dtype=torch.float64, device=device)

        result = solver.solve(P_values, A_values, q, b)
        info = solver.info
        x = result.x
        assert x.shape == (batch_size1, p["n"])

        # Solve with different batch_size=8 - should work automatically
        batch_size2 = 8
        P_values2 = torch.ones(batch_size2, p["nnzP"], dtype=torch.float64, device=device)
        A_values2 = torch.ones(batch_size2, p["nnzA"], dtype=torch.float64, device=device)
        q2 = torch.ones(batch_size2, p["n"], dtype=torch.float64, device=device) * 2.0
        b2 = torch.ones(batch_size2, p["m"], dtype=torch.float64, device=device)

        result2 = solver.solve(P_values2, A_values2, q2, b2)
        info = solver.info
        x2 = result2.x
        assert x2.shape == (batch_size2, p["n"])

    def test_torch_solver_dynamic_batch_sizes(self, problem_structure):
        """Test that TorchSolver handles variable batch sizes automatically."""
        p = problem_structure
        device = p["device"]
        settings = moreau.Settings(verbose=False)

        # Create solver (no batch_size needed)
        solver = TorchSolver(
            p["n"],
            p["m"],
            p["P_row_offsets"],
            p["P_col_indices"],
            p["A_row_offsets"],
            p["A_col_indices"],
            p["cones"],
            settings=settings,
        )

        # Forward with batch_size=4
        batch_size1 = 4
        P_values = torch.ones(batch_size1, p["nnzP"], dtype=torch.float64, device=device)
        A_values = torch.ones(batch_size1, p["nnzA"], dtype=torch.float64, device=device)
        q = torch.ones(batch_size1, p["n"], dtype=torch.float64, device=device) * 2.0
        b = torch.ones(batch_size1, p["m"], dtype=torch.float64, device=device)

        solution = solver.solve(P_values, A_values, q, b)
        assert solution.x.shape == (batch_size1, p["n"])

        # Forward with different batch_size=8 - should work automatically
        batch_size2 = 8
        P_values2 = torch.ones(batch_size2, p["nnzP"], dtype=torch.float64, device=device)
        A_values2 = torch.ones(batch_size2, p["nnzA"], dtype=torch.float64, device=device)
        q2 = torch.ones(batch_size2, p["n"], dtype=torch.float64, device=device) * 2.0
        b2 = torch.ones(batch_size2, p["m"], dtype=torch.float64, device=device)

        solution2 = solver.solve(P_values2, A_values2, q2, b2)
        assert solution2.x.shape == (batch_size2, p["n"])

    def test_eager_init_with_batch_size(self, problem_structure):
        """Test that specifying batch_size at construction eagerly initializes."""
        p = problem_structure
        device = p["device"]

        # Create solver WITH batch_size (eager initialization)
        batch_size = 4
        settings = moreau.Settings(
            verbose=False,
            batch_size=batch_size,
        )
        solver = TorchSolver(
            p["n"],
            p["m"],
            p["P_row_offsets"],
            p["P_col_indices"],
            p["A_row_offsets"],
            p["A_col_indices"],
            p["cones"],
            settings=settings,
        )

        # Check already initialized
        assert solver.is_initialized
        assert solver.batch_size == batch_size

        # Solve with same batch size
        P_values = torch.ones(batch_size, p["nnzP"], dtype=torch.float64, device=device)
        A_values = torch.ones(batch_size, p["nnzA"], dtype=torch.float64, device=device)
        q = torch.ones(batch_size, p["n"], dtype=torch.float64, device=device) * 2.0
        b = torch.ones(batch_size, p["m"], dtype=torch.float64, device=device)

        result = solver.solve(P_values, A_values, q, b)
        info = solver.info
        x = result.x
        assert x.shape == (batch_size, p["n"])
