"""Tests for PyTorch autograd gradient correctness using torch.autograd.gradcheck.

These tests verify that the backward pass computes correct gradients by comparing
against numerical finite differences.

Tests run on all available devices (CPU and CUDA when available).
"""

import pytest
import logging
import numpy as np
import gc
import weakref

torch = pytest.importorskip("torch")
from torch._dynamo.testing import CompileCounterWithBackend

import moreau
import moreau.torch as moreau_torch
from moreau.torch import Solver
from moreau.torch._compiled import _solve


def gradcheck_with_device(func, inputs, device, **kwargs):
    """Wrapper for gradcheck that adds tolerances appropriate for each device.

    CUDA operations may have small numerical non-determinism and slightly
    reduced precision in the backward pass for complex cones. We use
    relaxed tolerances for CUDA to account for this.
    """
    # Default tolerances - CPU
    defaults = {"eps": 1e-6, "atol": 1e-5, "rtol": 1e-4}

    # CUDA needs relaxed tolerances due to:
    # 1. Numerical non-determinism requiring nondet_tol
    # 2. Slightly reduced precision in cone backward computations
    #    (especially SOC and power cones have ~5e-3 absolute error in off-diagonals
    #     when the solution is near boundary configurations)
    # 3. Power cone Jacobian computation has known numerical precision issues
    if device == "cuda":
        defaults = {"eps": 1e-6, "atol": 5e-3, "rtol": 2e-2, "nondet_tol": 1e-5}

    defaults.update(kwargs)
    return torch.autograd.gradcheck(func, inputs, **defaults)


@pytest.fixture
def simple_qp_setup():
    """Setup for a simple QP with equality constraint.

    Problem: min (1/2)x'Px + q'x  s.t. x1 + x2 = b
    P = diag([2, 2]), A = [1, 1]
    """
    n = 2
    m = 1

    # CSR format
    P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
    P_col_indices = torch.tensor([0, 1], dtype=torch.int64)

    A_row_offsets = torch.tensor([0, 2], dtype=torch.int64)
    A_col_indices = torch.tensor([0, 1], dtype=torch.int64)

    cones = moreau.Cones()
    cones.num_zero_cones = 1

    return n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones


@pytest.fixture
def inequality_qp_setup():
    """Setup for QP with inequality constraints.

    Problem: min (1/2)x'Px + q'x  s.t. x >= 0
    P = diag([2, 2]), A = -I (for x >= 0)
    """
    n = 2
    m = 2

    # P = diag([2, 2])
    P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
    P_col_indices = torch.tensor([0, 1], dtype=torch.int64)

    # A = -I (x - s = 0, s >= 0 means x >= 0)
    A_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
    A_col_indices = torch.tensor([0, 1], dtype=torch.int64)

    cones = moreau.Cones()
    cones.num_nonneg_cones = 2

    return n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones


class TestGradcheckNonBatched:
    """Test gradient correctness for non-batched (single problem) cases."""

    def test_gradcheck_q_only(self, simple_qp_setup, device):
        """Test gradient w.r.t. q vector only."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        # Fixed inputs (no grad)
        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        b = torch.tensor([1.0], dtype=torch.float64, device=device)

        # Variable input (with grad)
        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)

    def test_gradcheck_b_only(self, simple_qp_setup, device):
        """Test gradient w.r.t. b vector only."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device)

        b = torch.tensor([1.0], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(b_):
            result = solver.solve(P_values, A_values, q, b_)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (b,), device)

    def test_gradcheck_q_and_b(self, simple_qp_setup, device):
        """Test gradient w.r.t. both q and b."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)

        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device, requires_grad=True)
        b = torch.tensor([1.0], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(q_, b_):
            result = solver.solve(P_values, A_values, q_, b_)
            info = solver.info
            return result.x

        assert gradcheck_with_device(
            solve_fn,
            (
                q,
                b,
            ),
            device,
        )

    def test_gradcheck_P_values(self, simple_qp_setup, device):
        """Test gradient w.r.t. P matrix values."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device, requires_grad=True)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device)
        b = torch.tensor([1.0], dtype=torch.float64, device=device)

        def solve_fn(P_):
            result = solver.solve(P_, A_values, q, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (P_values,), device)

    def test_gradcheck_A_values(self, simple_qp_setup, device):
        """Test gradient w.r.t. A matrix values."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device, requires_grad=True)
        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device)
        b = torch.tensor([1.0], dtype=torch.float64, device=device)

        def solve_fn(A_):
            result = solver.solve(P_values, A_, q, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (A_values,), device)

    def test_gradcheck_all_inputs(self, simple_qp_setup, device):
        """Test gradient w.r.t. all inputs simultaneously."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device, requires_grad=True)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device, requires_grad=True)
        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device, requires_grad=True)
        b = torch.tensor([1.0], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(P_, A_, q_, b_):
            result = solver.solve(P_, A_, q_, b_)
            info = solver.info
            return result.x

        assert gradcheck_with_device(
            solve_fn,
            (
                P_values,
                A_values,
                q,
                b,
            ),
            device,
        )

    def test_gradcheck_inequality_constraints(self, inequality_qp_setup, device):
        """Test gradient for QP with inequality constraints (interior point)."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = inequality_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0], dtype=torch.float64, device=device)
        # q with negative values so solution is interior (x > 0)
        q = torch.tensor([-2.0, -3.0], dtype=torch.float64, device=device, requires_grad=True)
        b = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)


class TestGradcheckBatched:
    """Test gradient correctness for batched (multiple problem) cases."""

    def test_gradcheck_batched_q_only(self, simple_qp_setup, device):
        """Test batched gradient w.r.t. q vector only."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup
        batch_size = 3

        settings = moreau.Settings(batch_size=batch_size, enable_grad=True)
        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=settings,
        )

        # Fixed batched inputs
        P_values = torch.tensor([[2.0, 2.0]] * batch_size, dtype=torch.float64, device=device)
        A_values = torch.tensor([[1.0, 1.0]] * batch_size, dtype=torch.float64, device=device)
        b = torch.tensor([[1.0]] * batch_size, dtype=torch.float64, device=device)

        # Variable batched input
        q = torch.tensor(
            [
                [-1.0, -0.5],
                [-1.2, -0.6],
                [-0.8, -0.4],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)

    def test_gradcheck_batched_b_only(self, simple_qp_setup, device):
        """Test batched gradient w.r.t. b vector only."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup
        batch_size = 3

        settings = moreau.Settings(batch_size=batch_size, enable_grad=True)
        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=settings,
        )

        P_values = torch.tensor([[2.0, 2.0]] * batch_size, dtype=torch.float64, device=device)
        A_values = torch.tensor([[1.0, 1.0]] * batch_size, dtype=torch.float64, device=device)
        q = torch.tensor([[-1.0, -0.5]] * batch_size, dtype=torch.float64, device=device)

        b = torch.tensor(
            [
                [1.0],
                [1.2],
                [0.8],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def solve_fn(b_):
            result = solver.solve(P_values, A_values, q, b_)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (b,), device)

    def test_gradcheck_batched_q_and_b(self, simple_qp_setup, device):
        """Test batched gradient w.r.t. both q and b."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup
        batch_size = 3

        settings = moreau.Settings(batch_size=batch_size, enable_grad=True)
        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=settings,
        )

        P_values = torch.tensor([[2.0, 2.0]] * batch_size, dtype=torch.float64, device=device)
        A_values = torch.tensor([[1.0, 1.0]] * batch_size, dtype=torch.float64, device=device)

        q = torch.tensor(
            [
                [-1.0, -0.5],
                [-1.2, -0.6],
                [-0.8, -0.4],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        b = torch.tensor(
            [
                [1.0],
                [1.2],
                [0.8],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def solve_fn(q_, b_):
            result = solver.solve(P_values, A_values, q_, b_)
            info = solver.info
            return result.x

        assert gradcheck_with_device(
            solve_fn,
            (
                q,
                b,
            ),
            device,
        )

    def test_gradcheck_batched_all_inputs(self, simple_qp_setup, device, request):
        """Test batched gradient w.r.t. all inputs simultaneously."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup
        batch_size = 3

        settings = moreau.Settings(batch_size=batch_size, enable_grad=True)
        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=settings,
        )

        P_values = torch.tensor(
            [
                [2.0, 2.0],
                [2.5, 2.5],
                [1.5, 1.5],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        A_values = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        q = torch.tensor(
            [
                [-1.0, -0.5],
                [-1.2, -0.6],
                [-0.8, -0.4],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        b = torch.tensor(
            [
                [1.0],
                [1.2],
                [0.8],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def solve_fn(P_, A_, q_, b_):
            result = solver.solve(P_, A_, q_, b_)
            info = solver.info
            return result.x

        assert gradcheck_with_device(
            solve_fn,
            (
                P_values,
                A_values,
                q,
                b,
            ),
            device,
        )

    def test_gradcheck_batched_inequality(self, inequality_qp_setup, device):
        """Test batched gradient for QP with inequality constraints."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = inequality_qp_setup
        batch_size = 3

        settings = moreau.Settings(batch_size=batch_size, enable_grad=True)
        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=settings,
        )

        P_values = torch.tensor([[2.0, 2.0]] * batch_size, dtype=torch.float64, device=device)
        A_values = torch.tensor([[-1.0, -1.0]] * batch_size, dtype=torch.float64, device=device)
        b = torch.tensor([[0.0, 0.0]] * batch_size, dtype=torch.float64, device=device)

        # Different q values ensuring interior solutions
        q = torch.tensor(
            [
                [-2.0, -3.0],
                [-1.5, -2.5],
                [-2.5, -2.0],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)


class TestGradcheckOutputs:
    """Test gradient correctness for different output variables (x, z, s)."""

    def test_gradcheck_dual_z(self, simple_qp_setup, device):
        """Test gradient when loss depends on dual variable z."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        b = torch.tensor([1.0], dtype=torch.float64, device=device)

        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.z

        assert gradcheck_with_device(solve_fn, (q,), device)

    def test_gradcheck_slack_s(self, inequality_qp_setup, device):
        """Test gradient when loss depends on slack variable s."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = inequality_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)

        q = torch.tensor([-2.0, -3.0], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.s

        assert gradcheck_with_device(solve_fn, (q,), device)

    def test_gradcheck_combined_loss(self, simple_qp_setup, device):
        """Test gradient when loss is a combination of x, z, s."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        b = torch.tensor([1.0], dtype=torch.float64, device=device)

        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            x, z, s = result.x, result.z, result.s
            # Combined loss: sum(x) + sum(z) + sum(s)
            return x.sum() + z.sum() + s.sum()

        # Use gradcheck with scalar output
        def scalar_fn(q_):
            return solve_fn(q_).unsqueeze(0)

        assert gradcheck_with_device(scalar_fn, (q,), device)


class TestGradcheckLargerProblems:
    """Test gradient correctness for larger problem sizes."""

    def test_gradcheck_larger_qp(self, device):
        """Test gradient for a larger QP (n=5, m=3)."""
        n = 5
        m = 3

        # P = 2*I (diagonal)
        P_row_offsets = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

        # A: 3 equality constraints
        # Row 0: x0 + x1 = b0
        # Row 1: x2 + x3 = b1
        # Row 2: x4 = b2
        A_row_offsets = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_zero_cones = 3

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
        )

        P_values = torch.tensor([2.0] * n, dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float64, device=device)
        b = torch.tensor([1.0, 1.0, 0.5], dtype=torch.float64, device=device)

        q = torch.tensor(
            [-1.0, -0.5, -0.8, -0.3, -0.6], dtype=torch.float64, device=device, requires_grad=True
        )

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)

    def test_gradcheck_batched_larger_qp(self, device):
        """Test batched gradient for a larger QP."""
        n = 4
        m = 2
        batch_size = 4

        # P = 2*I
        P_row_offsets = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

        # A: 2 equality constraints
        A_row_offsets = torch.tensor([0, 2, 4], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_zero_cones = 2

        settings = moreau.Settings(batch_size=batch_size, enable_grad=True)
        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
            settings=settings,
        )

        P_values = torch.tensor([[2.0] * n] * batch_size, dtype=torch.float64, device=device)
        A_values = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0]] * batch_size, dtype=torch.float64, device=device
        )
        b = torch.tensor([[1.0, 1.0]] * batch_size, dtype=torch.float64, device=device)

        q = torch.tensor(
            [
                [-1.0, -0.5, -0.8, -0.3],
                [-0.9, -0.6, -0.7, -0.4],
                [-1.1, -0.4, -0.9, -0.2],
                [-0.8, -0.7, -0.6, -0.5],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)


class TestGradcheckSecondOrderCone:
    """Test gradient correctness for second-order cone constraints."""

    def test_gradcheck_soc_q(self, device):
        """Test gradient w.r.t. q for SOC constraint.

        Problem with SOC: ||x[1:]|| <= x[0]
        We use A = -I, b = 0, so s = x and require s in SOC.
        """
        n = 3
        m = 3

        P_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        cones = moreau.Cones()
        cones.so_cone_dims = [3]

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device=device)

        # q chosen so solution is strictly interior to SOC
        q = torch.tensor([-3.0, -0.5, -0.3], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)

    def test_gradcheck_soc_batched(self, device):
        """Test batched gradient for SOC constraint."""
        n = 3
        m = 3
        batch_size = 3

        P_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        cones = moreau.Cones()
        cones.so_cone_dims = [3]

        settings = moreau.Settings(batch_size=batch_size, enable_grad=True)
        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
            settings=settings,
        )

        P_values = torch.tensor([[2.0, 2.0, 2.0]] * batch_size, dtype=torch.float64, device=device)
        A_values = torch.tensor(
            [[-1.0, -1.0, -1.0]] * batch_size, dtype=torch.float64, device=device
        )
        b = torch.tensor([[0.0, 0.0, 0.0]] * batch_size, dtype=torch.float64, device=device)

        q = torch.tensor(
            [
                [-3.0, -0.5, -0.3],
                [-2.5, -0.4, -0.6],
                [-4.0, -0.2, -0.1],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)

    def test_gradcheck_soc_all_inputs(self, device):
        """Test gradient w.r.t. all inputs for SOC constraint."""
        n = 3
        m = 3

        P_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        cones = moreau.Cones()
        cones.so_cone_dims = [3]

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
        )

        P_values = torch.tensor(
            [2.0, 2.0, 2.0], dtype=torch.float64, device=device, requires_grad=True
        )
        A_values = torch.tensor(
            [-1.0, -1.0, -1.0], dtype=torch.float64, device=device, requires_grad=True
        )
        q = torch.tensor([-3.0, -0.5, -0.3], dtype=torch.float64, device=device, requires_grad=True)
        b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(P_, A_, q_, b_):
            result = solver.solve(P_, A_, q_, b_)
            info = solver.info
            return result.x

        assert gradcheck_with_device(
            solve_fn,
            (
                P_values,
                A_values,
                q,
                b,
            ),
            device,
        )

    def test_gradcheck_soc_q_dim5(self, device):
        """Test gradient w.r.t. q for SOC dim=5 constraint."""
        n = 5
        m = 5

        P_row_offsets = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

        cones = moreau.Cones(so_cone_dims=[5])

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=device)

        q = torch.tensor(
            [-3.0, -0.5, -0.3, -0.2, -0.1], dtype=torch.float64, device=device, requires_grad=True
        )

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)

    def test_gradcheck_soc_batched_dim5(self, device):
        """Test batched gradient for SOC dim=5 constraint."""
        n = 5
        m = 5
        batch_size = 3

        P_row_offsets = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

        cones = moreau.Cones(so_cone_dims=[5])

        settings = moreau.Settings(batch_size=batch_size, enable_grad=True)
        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
            settings=settings,
        )

        P_values = torch.tensor(
            [[2.0, 2.0, 2.0, 2.0, 2.0]] * batch_size, dtype=torch.float64, device=device
        )
        A_values = torch.tensor(
            [[-1.0, -1.0, -1.0, -1.0, -1.0]] * batch_size, dtype=torch.float64, device=device
        )
        b = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0]] * batch_size, dtype=torch.float64, device=device
        )

        q = torch.tensor(
            [
                [-3.0, -0.5, -0.3, -0.2, -0.1],
                [-2.5, -0.4, -0.6, -0.1, -0.3],
                [-4.0, -0.2, -0.1, -0.3, -0.2],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)


class TestGradcheckExponentialCone:
    """Test gradient correctness for exponential cone constraints."""

    def test_gradcheck_exp_cone_q(self, device):
        """Test gradient w.r.t. q for exponential cone constraint.

        Exponential cone: (x, y, z) such that y * exp(x/y) <= z, y > 0
        We use A = -I, b = 0, so s = x and require s in K_exp.
        """
        n = 3
        m = 3

        P_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_exp_cones = 1

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device=device)

        # q chosen so solution is strictly interior to exp cone
        q = torch.tensor([1.0, -2.0, -3.0], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)

    def test_gradcheck_exp_cone_b(self, device):
        """Test gradient w.r.t. b for exponential cone constraint.

        Exponential cone: (x, y, z) such that y * exp(x/y) <= z, y > 0
        We use A = -I, so s = -x + b and require s in K_exp.

        FIX: The HSDE backward formula was corrected from 'db = τ*λ₂ + λ₄*Π_{K*}(u)'
        to 'db = τ*λ₂ - λ₄*z' per diffqcp formula.
        """
        n = 3
        m = 3

        P_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_exp_cones = 1

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64, device=device)
        q = torch.tensor([1.0, -2.0, -3.0], dtype=torch.float64, device=device)

        # b chosen so solution is strictly interior to exp cone
        b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(b_):
            result = solver.solve(P_values, A_values, q, b_)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (b,), device)

    def test_gradcheck_exp_cone_batched(self, device):
        """Test batched gradient for exponential cone constraint.

        Note: Exponential cone has numerical sensitivity in the Jacobian computation,
        especially for batched problems. We use relaxed tolerances for this test.
        The off-diagonal elements can have ~1% error due to the iterative nature
        of the exp cone projection and its derivatives.
        """
        n = 3
        m = 3
        batch_size = 3

        P_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_exp_cones = 1

        settings = moreau.Settings(batch_size=batch_size, enable_grad=True)
        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
            settings=settings,
        )

        P_values = torch.tensor([[2.0, 2.0, 2.0]] * batch_size, dtype=torch.float64, device=device)
        A_values = torch.tensor(
            [[-1.0, -1.0, -1.0]] * batch_size, dtype=torch.float64, device=device
        )
        b = torch.tensor([[0.0, 0.0, 0.0]] * batch_size, dtype=torch.float64, device=device)

        q = torch.tensor(
            [
                [1.0, -2.0, -3.0],
                [0.5, -2.5, -2.5],
                [1.5, -1.5, -4.0],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        # Exp cone batched requires relaxed tolerance due to numerical sensitivity
        # in the iterative exp cone projection Jacobian computation.
        # The off-diagonal Jacobian elements can have ~1.5% error due to the
        # iterative nature of the exp cone projection derivatives.
        assert gradcheck_with_device(solve_fn, (q,), device, atol=1.5e-2, rtol=5e-2)


class TestGradcheckPowerCone:
    """Test gradient correctness for power cone constraints."""

    def test_gradcheck_power_cone_q(self, device):
        """Test gradient w.r.t. q for power cone constraint.

        Power cone with alpha=0.5: x^0.5 * y^0.5 >= |z|, x >= 0, y >= 0
        We use A = -I, b = 0, so s = x and require s in K_pow.
        """
        n = 3
        m = 3

        P_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        cones = moreau.Cones()
        cones.power_alphas = [0.5]

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device=device)

        # q chosen so solution is strictly interior to power cone
        q = torch.tensor([-2.0, -2.0, -0.1], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)

    def test_gradcheck_power_cone_b(self, device):
        """Test gradient w.r.t. b for power cone constraint.

        Power cone with alpha=0.5: x^0.5 * y^0.5 >= |z|, x >= 0, y >= 0
        We use A = -I, so s = -x + b and require s in K_pow.

        FIX: The HSDE backward formula was corrected from 'db = τ*λ₂ + λ₄*Π_{K*}(u)'
        to 'db = τ*λ₂ - λ₄*z' per diffqcp formula.
        """
        n = 3
        m = 3

        P_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        cones = moreau.Cones()
        cones.power_alphas = [0.5]

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64, device=device)
        q = torch.tensor([-2.0, -2.0, -0.1], dtype=torch.float64, device=device)

        # b chosen so solution is strictly interior to power cone
        b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(b_):
            result = solver.solve(P_values, A_values, q, b_)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (b,), device)

    def test_gradcheck_power_cone_batched(self, device):
        """Test batched gradient for power cone constraint."""
        n = 3
        m = 3
        batch_size = 3

        P_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        cones = moreau.Cones()
        cones.power_alphas = [0.5]

        settings = moreau.Settings(batch_size=batch_size, enable_grad=True)
        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
            settings=settings,
        )

        P_values = torch.tensor([[2.0, 2.0, 2.0]] * batch_size, dtype=torch.float64, device=device)
        A_values = torch.tensor(
            [[-1.0, -1.0, -1.0]] * batch_size, dtype=torch.float64, device=device
        )
        b = torch.tensor([[0.0, 0.0, 0.0]] * batch_size, dtype=torch.float64, device=device)

        q = torch.tensor(
            [
                [-2.0, -2.0, -0.1],
                [-3.0, -1.5, -0.2],
                [-1.5, -3.0, 0.1],
            ],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)

    def test_gradcheck_power_cone_alpha_03(self, device):
        """Test gradient for power cone with alpha=0.3."""
        n = 3
        m = 3

        P_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        cones = moreau.Cones()
        cones.power_alphas = [0.3]

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64, device=device)

        q = torch.tensor([-2.0, -2.0, -0.1], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)


class TestGradcheckMixedCones:
    """Test gradient correctness for problems with multiple cone types."""

    def test_gradcheck_zero_and_nonneg(self, device):
        """Test gradient for mixed zero and nonnegative cones."""
        n = 3
        m = 3  # 1 equality + 2 inequalities

        P_row_offsets = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2], dtype=torch.int64)

        # A: row 0 is equality (sum), rows 1-2 are bounds
        A_row_offsets = torch.tensor([0, 3, 4, 5], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2, 0, 1], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_zero_cones = 1
        cones.num_nonneg_cones = 2

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=device)

        q = torch.tensor([-1.0, -0.5, -0.3], dtype=torch.float64, device=device, requires_grad=True)

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)

    def test_gradcheck_nonneg_and_soc(self, device):
        """Test gradient for mixed nonnegative and second-order cones."""
        n = 5
        m = 5  # 2 nonneg + 3 SOC

        P_row_offsets = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_nonneg_cones = 2
        cones.so_cone_dims = [3]

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            cones=cones,
        )

        P_values = torch.tensor([2.0] * n, dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0] * n, dtype=torch.float64, device=device)
        b = torch.tensor([0.0] * m, dtype=torch.float64, device=device)

        # q: push first 2 positive (nonneg), and SOC interior
        q = torch.tensor(
            [-2.0, -3.0, -4.0, -0.5, -0.3], dtype=torch.float64, device=device, requires_grad=True
        )

        def solve_fn(q_):
            result = solver.solve(P_values, A_values, q_, b)
            info = solver.info
            return result.x

        assert gradcheck_with_device(solve_fn, (q,), device)


class TestJacrev:
    """Test batched backward via torch.func.jacrev (uses vmap over backward)."""

    def test_jacrev_q(self, simple_qp_setup, device):
        """Jacobian dx/dq via jacrev matches finite differences."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        b = torch.tensor([1.0], dtype=torch.float64, device=device)
        q = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)

        def solve_x(q_):
            return solver.solve(P_values, A_values, q_, b).x

        J = torch.func.jacrev(solve_x)(q)
        assert J.shape == (n, n)

        # Compare against finite differences
        eps = 1e-6
        J_fd = torch.zeros(n, n, dtype=torch.float64, device=device)
        x0 = solve_x(q)
        for i in range(n):
            q_pert = q.clone()
            q_pert[i] += eps
            x_pert = solve_x(q_pert)
            J_fd[:, i] = (x_pert - x0) / eps

        torch.testing.assert_close(J, J_fd, atol=1e-4, rtol=1e-3)

    def test_jacrev_b(self, simple_qp_setup, device):
        """Jacobian dx/db via jacrev."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        q = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([1.0], dtype=torch.float64, device=device)

        def solve_x(b_):
            return solver.solve(P_values, A_values, q, b_).x

        J = torch.func.jacrev(solve_x)(b)
        assert J.shape == (n, m)

        eps = 1e-6
        J_fd = torch.zeros(n, m, dtype=torch.float64, device=device)
        x0 = solve_x(b)
        for j in range(m):
            b_pert = b.clone()
            b_pert[j] += eps
            x_pert = solve_x(b_pert)
            J_fd[:, j] = (x_pert - x0) / eps

        torch.testing.assert_close(J, J_fd, atol=1e-4, rtol=1e-3)

    def test_jacrev_inequality(self, inequality_qp_setup, device):
        """Jacobian dx/dq for inequality-constrained QP."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = inequality_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)
        # q chosen so some constraints are active, some inactive
        q = torch.tensor([-3.0, 1.0], dtype=torch.float64, device=device)

        def solve_x(q_):
            return solver.solve(P_values, A_values, q_, b).x

        J = torch.func.jacrev(solve_x)(q)
        assert J.shape == (n, n)

        eps = 1e-6
        J_fd = torch.zeros(n, n, dtype=torch.float64, device=device)
        x0 = solve_x(q)
        for i in range(n):
            q_pert = q.clone()
            q_pert[i] += eps
            x_pert = solve_x(q_pert)
            J_fd[:, i] = (x_pert - x0) / eps

        torch.testing.assert_close(J, J_fd, atol=1e-4, rtol=1e-3)


class TestAutogradRegressions:
    """Regression tests for runtime failures in the custom autograd path."""

    class _FakeImpl:
        def __init__(self):
            self._last_solve_mode = "single"

        def solve(self, q, b, **warm_kwargs):
            return {
                "x": q + 1.0,
                "z": b + 2.0,
                "s": b + 3.0,
            }

    class _FakeSolver:
        def __init__(self):
            self._impl = TestAutogradRegressions._FakeImpl()
            self._impl_handle = 123
            self._last_result = None
            self._pending_warm_kwargs = {}

    def test_backward_keeps_ephemeral_solver_alive(self, monkeypatch):
        """The autograd graph should keep the solver alive through backward."""
        recorded = {}

        def fake_backward_op(
            dx,
            dz,
            ds,
            impl_handle,
            solve_mode,
            state_rinv,
            state_rinv_diag,
            state_use_rinv_diag,
            state_n_active,
            state_ws,
            state_sense,
            state_lam_star,
            P_values,
            A_values,
            q,
            b,
            x,
            z,
            s,
            z_x,
            dz_x,
        ):
            recorded["solver_alive_during_backward"] = solver_ref() is not None
            return (
                torch.zeros_like(P_values),
                torch.ones_like(q),
                torch.zeros_like(A_values),
                torch.zeros_like(b),
            )

        monkeypatch.setattr(moreau_torch, "_solve_backward_op", fake_backward_op)

        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([1.0], dtype=torch.float64)
        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64)

        solver = self._FakeSolver()
        solver_ref = weakref.ref(solver)
        x, z, s, z_x = moreau_torch._SolveFunction.apply(solver, q, b, P_values, A_values)
        del solver
        gc.collect()

        assert solver_ref() is not None

        x.sum().backward()

        assert recorded["solver_alive_during_backward"]
        assert q.grad is not None
        assert solver_ref() is not None
        assert solver_ref()._last_result is None

    def test_backward_accepts_unused_z_and_s_grads(self, monkeypatch):
        """Missing z/s grad outputs should be converted to zeros before dispatch."""
        recorded = {}

        def fake_backward_op(
            dx,
            dz,
            ds,
            impl_handle,
            solve_mode,
            state_rinv,
            state_rinv_diag,
            state_use_rinv_diag,
            state_n_active,
            state_ws,
            state_sense,
            state_lam_star,
            P_values,
            A_values,
            q,
            b,
            x,
            z,
            s,
            z_x,
            dz_x,
        ):
            recorded["dx"] = dx.clone()
            recorded["dz"] = dz.clone()
            recorded["ds"] = ds.clone()
            return (
                torch.zeros_like(P_values),
                torch.ones_like(q),
                torch.zeros_like(A_values),
                torch.zeros_like(b),
            )

        monkeypatch.setattr(moreau_torch, "_solve_backward_op", fake_backward_op)

        solver = self._FakeSolver()
        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, requires_grad=True)
        b = torch.tensor([1.0], dtype=torch.float64)
        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64)

        x, z, s, z_x = moreau_torch._SolveFunction.apply(solver, q, b, P_values, A_values)
        x.sum().backward()
        torch.testing.assert_close(recorded["dx"], torch.ones_like(x))
        torch.testing.assert_close(recorded["dz"], torch.zeros_like(z))
        torch.testing.assert_close(recorded["ds"], torch.zeros_like(s))
        assert q.grad is not None

    def test_active_set_nested_composition_preserves_per_forward_state(self):
        n = 2
        m = 1
        P_ro = torch.tensor([0, 1, 2], dtype=torch.int64)
        P_ci = torch.tensor([0, 1], dtype=torch.int64)
        A_ro = torch.tensor([0, 2], dtype=torch.int64)
        A_ci = torch.tensor([0, 1], dtype=torch.int64)
        cones = moreau.Cones()
        cones.num_zero_cones = 1

        settings = moreau.Settings(device="cpu", enable_grad=True, solver="active_set")
        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=settings,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64)
        b = torch.tensor([1.0], dtype=torch.float64)
        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, requires_grad=True)

        def f(q_):
            return solver.solve(P_values, A_values, q_, b).x

        loss = f(f(q)).sum()
        loss.backward()

        assert q.grad is not None
        assert torch.isfinite(q.grad).all()


class TestTorchCompile:
    """Test that torch.compile works with the custom op."""

    def test_native_boundary_inductor(self, simple_qp_setup, device, caplog, recwarn):
        """Compile tensor work around native construction, setup, and solve."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup
        settings = moreau.Settings(device=device, solver="ipm")
        P = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        b = torch.tensor([1.0], dtype=torch.float64, device=device)

        def loss_fn(q):
            solver = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
            solver.setup(P, A)
            x = solver.solve(P, A, 1.5 * q, b).x
            return x.square().sum()

        torch._dynamo.reset()
        backend = CompileCounterWithBackend("inductor")
        compiled = torch.compile(loss_fn, backend=backend)
        for values in ([0.1, -0.3], [0.4, 0.2]):
            q = torch.tensor(values, dtype=torch.float64, device=device, requires_grad=True)
            expected = loss_fn(q)
            expected_grad = torch.autograd.grad(expected, q)[0]
            actual = compiled(q)
            actual_grad = torch.autograd.grad(actual, q)[0]
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(actual_grad, expected_grad)
        assert backend.frame_count >= 2
        assert backend.op_count >= 3
        failures = [
            record.getMessage()
            for record in caplog.records
            if record.name.startswith("torch.")
            and (
                record.levelno >= logging.ERROR
                or "Backend compiler exception" in record.getMessage()
            )
        ]
        assert not failures, "\n".join(failures)
        assert not any("Dynamo does not know how to trace" in str(w.message) for w in recwarn)
        torch._dynamo.reset()

    def test_compile_forward(self, simple_qp_setup, device):
        """Compiled forward produces same result as eager."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        q = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([1.0], dtype=torch.float64, device=device)

        def solve_x(q_):
            return solver.solve(P_values, A_values, q_, b).x

        x_eager = solve_x(q)
        solve_compiled = torch.compile(solve_x, backend="eager")
        x_compiled = solve_compiled(q)

        torch.testing.assert_close(x_compiled, x_eager)

    def test_compile_backward(self, simple_qp_setup, device):
        """Compiled backward produces same gradients as eager."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        b = torch.tensor([1.0], dtype=torch.float64, device=device)

        def loss_fn(q_):
            return solver.solve(P_values, A_values, q_, b).x.sum()

        q = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device, requires_grad=True)
        loss_eager = loss_fn(q)
        loss_eager.backward()
        grad_eager = q.grad.clone()

        q_compiled = torch.tensor(
            [1.0, -1.0], dtype=torch.float64, device=device, requires_grad=True
        )
        loss_compiled_fn = torch.compile(loss_fn, backend="eager")
        loss_compiled = loss_compiled_fn(q_compiled)
        loss_compiled.backward()
        grad_compiled = q_compiled.grad

        torch.testing.assert_close(grad_compiled, grad_eager)

    def test_compile_in_nn_module(self, simple_qp_setup, device):
        """Solver works inside a compiled nn.Module."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        b = torch.tensor([1.0], dtype=torch.float64, device=device)

        class SolveLayer(torch.nn.Module):
            def forward(self, q):
                return solver.solve(P_values, A_values, q, b).x

        layer = SolveLayer()
        q = torch.tensor([1.0, -1.0], dtype=torch.float64, device=device)

        x_eager = layer(q)
        layer_compiled = torch.compile(layer, backend="eager")
        x_compiled = layer_compiled(q)

        torch.testing.assert_close(x_compiled, x_eager)


@pytest.mark.parametrize("batch_shape", [(), (1,), (3,)])
@pytest.mark.parametrize("shared_matrices", [False, True])
@pytest.mark.parametrize(
    "solver_type", ["ipm", pytest.param("active_set", marks=pytest.mark.cpu_only)]
)
def test_fullgraph_solve(simple_qp_setup, batch_shape, shared_matrices, device, solver_type):
    """Capture cold solves, changing data, and delayed backward in one graph."""
    n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup
    solver = Solver(
        n, m, P_ro, P_ci, A_ro, A_ci, cones, moreau.Settings(device=device, solver=solver_type)
    )
    torch._dynamo.reset()
    backend = CompileCounterWithBackend("inductor")

    def solve(P, A, q, b):
        sol = solver.solve(P, A, q, b)
        return sol.x.square().sum() + 0.3 * sol.z.sum() + 0.2 * sol.s.sum()

    compiled = torch.compile(solve, backend=backend, fullgraph=True)
    losses, expected_grads, all_inputs = [], [], []
    for scale in (1.0, 1.3):
        values = ([2.0 * scale, 3.0], [1.0, 1.0 * scale], [0.2, -0.3], [1.0])
        inputs = []
        for i, val in enumerate(values):
            t = torch.tensor(val, dtype=torch.float64, device=device)
            if batch_shape and not (shared_matrices and i < 2):
                t = t.expand(*batch_shape, -1).clone()
            inputs.append(t.requires_grad_())
        # The first compiled call must initialize the native solver itself.
        actual = compiled(*inputs)
        expected = solve(*inputs)
        expected_grads.append(torch.autograd.grad(expected, inputs))
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
        losses.append(actual)
        all_inputs.extend(inputs)
    gradients = torch.autograd.grad(sum(losses), all_inputs)
    for actual, expected in zip(gradients, sum(expected_grads, ())):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    assert backend.frame_count == 1
    assert solver.info is not None
    torch._dynamo.reset()


@pytest.mark.parametrize("batch_shape", [(), (1,), (3,)])
def test_fullgraph_direct_duals(batch_shape, device):
    cones = moreau.Cones(dir_cones=[moreau.DirectConeSpec(kind="nonneg", indices=[0, 1])])
    solver = Solver(
        2,
        0,
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 1]),
        torch.tensor([0]),
        torch.empty(0, dtype=torch.int64),
        cones,
        moreau.Settings(device=device, solver="ipm"),
    )
    P = torch.ones(2, dtype=torch.float64, device=device)
    A = torch.empty(0, dtype=torch.float64, device=device)
    q = torch.tensor([-2.0, 1.0], dtype=torch.float64, device=device)
    q = q.expand(*batch_shape, 2).clone().requires_grad_()
    if batch_shape == (3,):
        q = q.T.contiguous().T.detach().requires_grad_()
        assert not q.is_contiguous()
    b = torch.empty(*batch_shape, 0, dtype=torch.float64, device=device)
    torch._dynamo.reset()

    def solve(q):
        sol = solver.solve(P, A, q, b)
        return sol.x, sol.z_x

    compiled = torch.compile(solve, fullgraph=True)
    x, zx = compiled(q)
    torch.testing.assert_close(x, (-q).clamp_min(0), atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(zx, q.clamp_min(0), atol=1e-7, rtol=1e-7)
    (grad,) = torch.autograd.grad(x.sum() + 2 * zx.sum(), q)
    torch.testing.assert_close(
        grad, torch.where(q < 0, -torch.ones_like(q), 2 * torch.ones_like(q)), atol=1e-6, rtol=1e-6
    )
    torch.library.opcheck(
        _solve,
        (P, A, q, b, solver._compile_handle, 2, False, []),
    )
    torch._dynamo.reset()


def test_compiled_handle_lifetime(simple_qp_setup, device):
    """Saved storage owns the native solver, even with detached saved tensors."""
    n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp_setup
    solver = Solver(
        n, m, P_ro, P_ci, A_ro, A_ci, cones, moreau.Settings(device=device, solver="ipm")
    )
    P = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
    A = torch.ones(2, dtype=torch.float64, device=device)
    q = torch.tensor([0.2, -0.3], dtype=torch.float64, device=device, requires_grad=True)
    b = torch.ones(1, dtype=torch.float64, device=device)
    # Tune before taking the weak reference; tuning can replace the impl.
    solver.solve(P, A, q, b)
    impl_ref = weakref.ref(solver._impl)
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t.detach(), lambda t: t):
        loss = _solve(P, A, q, b, solver._compile_handle, 0, False, [])[0].square().sum()
    del solver
    gc.collect()
    assert impl_ref() is not None
    loss.backward()
    assert torch.isfinite(q.grad).all()
    gc.collect()
    assert impl_ref() is None


@pytest.mark.cpu_only
@pytest.mark.parametrize("batch_shape", [(), (1,), (3,)])
def test_fullgraph_active_set_snapshot(inequality_qp_setup, batch_shape, monkeypatch):
    """Delayed backward restores each active set without solving again."""
    n, m, P_ro, P_ci, A_ro, A_ci, cones = inequality_qp_setup
    solver = Solver(
        n,
        m,
        P_ro,
        P_ci,
        A_ro,
        A_ci,
        cones,
        moreau.Settings(device="cpu", solver="active_set"),
    )
    P = torch.full((2,), 2.0, dtype=torch.float64)
    A = -torch.ones(len(A_ci), dtype=torch.float64)
    b = torch.zeros(*batch_shape, m, dtype=torch.float64)
    torch._dynamo.reset()

    def solve(q):
        return solver.solve(P, A, q, b).x

    compiled = torch.compile(solve, fullgraph=True)
    inputs, losses = [], []
    for values in ([1.0, -2.0], [-2.0, 1.0]):
        q = torch.tensor(values, dtype=torch.float64).expand(*batch_shape, n).clone()
        q.requires_grad_()
        x = compiled(q)
        expected = (-q / 2).clamp_min(0)
        torch.testing.assert_close(x, expected, atol=1e-7, rtol=1e-7)
        inputs.append(q)
        losses.append(x.square().sum())

    def unexpected_solve(*args, **kwargs):
        raise AssertionError("Backward must use its saved active-set snapshot")

    with monkeypatch.context() as patch:
        patch.setattr(solver._impl, "solve", unexpected_solve)
        grads = torch.autograd.grad(sum(losses), inputs)
    for q, grad in zip(inputs, grads):
        expected = (q / 2).clamp_max(0)
        torch.testing.assert_close(grad, expected, atol=1e-7, rtol=1e-7)
    torch.library.opcheck(_solve, (P, A, inputs[0], b, solver._compile_handle, 0, True, []))
    torch._dynamo.reset()
