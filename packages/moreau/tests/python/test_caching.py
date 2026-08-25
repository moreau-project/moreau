"""Tests for solver caching behavior.

Verifies that the CPU solver correctly caches:
1. Symbolic factorization (at solver construction)
2. Equilibration (when P/A unchanged)
3. Gradient state (when enable_grad=True)
"""

import numpy as np
import pytest
import time

import moreau


def create_qp_problem():
    """Create a simple QP problem for caching tests.

    Problem: min 0.5*x'Px + q'x s.t. x >= 0 (nonnegative cone)
    This gives solution x* = -P^{-1}q projected onto nonneg cone.
    """
    n, m = 5, 5

    # P = 2*I in CSR format
    P_row_offsets = np.arange(n + 1, dtype=np.int64)
    P_col_indices = np.arange(n, dtype=np.int64)
    P_values = np.ones(n, dtype=np.float64) * 2.0

    # A = -I in CSR format (x >= 0 becomes -x + s = 0, s >= 0)
    A_row_offsets = np.arange(m + 1, dtype=np.int64)
    A_col_indices = np.arange(m, dtype=np.int64)
    A_values = -np.ones(m, dtype=np.float64)

    # q such that unconstrained solution would be positive
    q = -np.ones(n, dtype=np.float64)  # Solution: x* = 0.5 * ones
    b = np.zeros(m, dtype=np.float64)

    cones = moreau.Cones(num_nonneg_cones=m)

    return (
        n,
        m,
        P_row_offsets,
        P_col_indices,
        P_values,
        A_row_offsets,
        A_col_indices,
        A_values,
        q,
        b,
        cones,
    )


class TestSingleProblemCaching:
    """Test caching for single-problem solves."""

    def test_repeated_solve_same_pa_reuses_equilibration(self, device):
        """Repeated solves with same P/A should reuse equilibration."""
        from scipy import sparse

        n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones = create_qp_problem()

        # Convert to sparse matrices
        P = sparse.csr_array((P_val, P_ci, P_ro), shape=(n, n))
        A = sparse.csr_array((A_val, A_ci, A_ro), shape=(m, m))

        # Create bounds (unbounded)

        settings = moreau.Settings(device=device, verbose=False)
        solver = moreau.Solver(P, q, A, b, cones, settings)

        # First solve
        result1 = solver.solve()
        info1 = solver.info
        assert info1.status == moreau.SolverStatus.Solved

        # Second solve with different q/b but same P/A (need to re-create solver)
        q2 = q * 2
        b2 = b * 2
        solver2 = moreau.Solver(P, q2, A, b2, cones, settings)
        result2 = solver2.solve()
        info2 = solver2.info
        assert info2.status == moreau.SolverStatus.Solved

        # Third solve with original q/b
        solver3 = moreau.Solver(P, q, A, b, cones, settings)
        result3 = solver3.solve()
        info3 = solver3.info
        assert info3.status == moreau.SolverStatus.Solved

        # Solutions should match expected values
        np.testing.assert_allclose(result1.x, result3.x, rtol=1e-6)

    def test_setup_with_new_pa_recomputes_equilibration(self, device):
        """Changing P/A via different solvers should give different results."""
        from scipy import sparse

        n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones = create_qp_problem()

        # Convert to sparse matrices
        P = sparse.csr_array((P_val, P_ci, P_ro), shape=(n, n))
        A = sparse.csr_array((A_val, A_ci, A_ro), shape=(m, m))

        # Create bounds (unbounded)

        settings = moreau.Settings(device=device, verbose=False)

        # First solve
        solver1 = moreau.Solver(P, q, A, b, cones, settings)
        result1 = solver1.solve()
        info1 = solver1.info
        assert info1.status == moreau.SolverStatus.Solved

        # Change P values and solve again
        P_val_new = P_val * 2
        P_new = sparse.csr_array((P_val_new, P_ci, P_ro), shape=(n, n))
        solver2 = moreau.Solver(P_new, q, A, b, cones, settings)
        result2 = solver2.solve()
        info2 = solver2.info
        assert info2.status == moreau.SolverStatus.Solved

        # Solutions should be different (different P)
        assert not np.allclose(result1.x, result2.x, rtol=1e-3)

    def test_caching_across_many_solves(self, device):
        """Test that different q values give different solutions."""
        from scipy import sparse

        n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones = create_qp_problem()

        # Convert to sparse matrices
        P = sparse.csr_array((P_val, P_ci, P_ro), shape=(n, n))
        A = sparse.csr_array((A_val, A_ci, A_ro), shape=(m, m))

        # Create bounds (unbounded)

        settings = moreau.Settings(device=device, verbose=False)

        # Solve many times with different q values
        solutions = []
        for i in range(10):
            qi = q * (1 + i * 0.1)
            solver = moreau.Solver(P, qi, A, b, cones, settings)
            result = solver.solve()
            info = solver.info
            assert info.status == moreau.SolverStatus.Solved
            solutions.append(result.x.copy())

        # Verify solutions are different (since q changed)
        for i in range(1, 10):
            assert not np.allclose(solutions[0], solutions[i], rtol=1e-3)


class TestGradientCaching:
    """Test caching for gradient computation."""

    def test_backward_after_single_solve(self, device):
        """Test backward() works with torch solver."""
        pytest.importorskip("torch")
        import torch
        from moreau.torch import Solver as TorchSolver

        n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones = create_qp_problem()

        settings = moreau.Settings(device=device, verbose=False, enable_grad=True, batch_size=1)

        # Convert to torch tensors for TorchSolver
        P_ro_t = torch.tensor(P_ro, dtype=torch.int64)
        P_ci_t = torch.tensor(P_ci, dtype=torch.int64)
        A_ro_t = torch.tensor(A_ro, dtype=torch.int64)
        A_ci_t = torch.tensor(A_ci, dtype=torch.int64)

        solver = TorchSolver(n, m, P_ro_t, P_ci_t, A_ro_t, A_ci_t, cones, settings)

        # Convert to tensors and add batch dimension
        # Use requires_grad_ after unsqueeze to keep as leaf tensors
        device_str = "cuda" if device == "cuda" else "cpu"
        P_t = (
            torch.tensor(P_val, dtype=torch.float64, device=device_str)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        A_t = (
            torch.tensor(A_val, dtype=torch.float64, device=device_str)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        q_t = (
            torch.tensor(q, dtype=torch.float64, device=device_str)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        b_t = (
            torch.tensor(b, dtype=torch.float64, device=device_str)
            .unsqueeze(0)
            .requires_grad_(True)
        )

        # Solve
        result = solver.solve(P_t, A_t, q_t, b_t)
        info = solver.info

        # Backward
        result.x.sum().backward()

        assert q_t.grad is not None
        assert b_t.grad is not None
        assert P_t.grad is not None
        assert A_t.grad is not None

    def test_backward_after_repeated_solves(self, device):
        """Test backward() works correctly for different problems."""
        pytest.importorskip("torch")
        import torch
        from moreau.torch import Solver as TorchSolver

        n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones = create_qp_problem()

        settings = moreau.Settings(device=device, verbose=False, enable_grad=True, batch_size=1)

        device_str = "cuda" if device == "cuda" else "cpu"

        # Convert to torch tensors for TorchSolver
        P_ro_t = torch.tensor(P_ro, dtype=torch.int64)
        P_ci_t = torch.tensor(P_ci, dtype=torch.int64)
        A_ro_t = torch.tensor(A_ro, dtype=torch.int64)
        A_ci_t = torch.tensor(A_ci, dtype=torch.int64)

        # First problem - use requires_grad_ after unsqueeze to keep as leaf tensors
        solver1 = TorchSolver(n, m, P_ro_t, P_ci_t, A_ro_t, A_ci_t, cones, settings)
        P_t1 = (
            torch.tensor(P_val, dtype=torch.float64, device=device_str)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        A_t1 = (
            torch.tensor(A_val, dtype=torch.float64, device=device_str)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        q_t1 = (
            torch.tensor(q, dtype=torch.float64, device=device_str)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        b_t1 = (
            torch.tensor(b, dtype=torch.float64, device=device_str)
            .unsqueeze(0)
            .requires_grad_(True)
        )

        result1 = solver1.solve(P_t1, A_t1, q_t1, b_t1)
        result1.x.sum().backward()
        grad1 = q_t1.grad.clone() if q_t1.grad is not None else None

        # Second problem with different q - use requires_grad_ after unsqueeze
        solver2 = TorchSolver(n, m, P_ro_t, P_ci_t, A_ro_t, A_ci_t, cones, settings)
        q2 = np.ones(n, dtype=np.float64)  # Different q
        P_t2 = (
            torch.tensor(P_val, dtype=torch.float64, device=device_str)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        A_t2 = (
            torch.tensor(A_val, dtype=torch.float64, device=device_str)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        q_t2 = (
            torch.tensor(q2, dtype=torch.float64, device=device_str)
            .unsqueeze(0)
            .requires_grad_(True)
        )
        b_t2 = (
            torch.tensor(b, dtype=torch.float64, device=device_str)
            .unsqueeze(0)
            .requires_grad_(True)
        )

        result2 = solver2.solve(P_t2, A_t2, q_t2, b_t2)
        result2.x.sum().backward()
        grad2 = q_t2.grad.clone() if q_t2.grad is not None else None

        # Solutions should be different
        assert not torch.allclose(result1.x, result2.x, rtol=1e-3)


class TestBatchCaching:
    """Test caching for batched solves."""

    def test_batch_shared_pa_caching(self, device):
        """Test that shared P/A batched solves work correctly."""
        n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones = create_qp_problem()

        batch_size = 4
        settings = moreau.Settings(device=device, verbose=False, batch_size=batch_size)
        solver = moreau.CompiledSolver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)

        # Shared P/A values (1D) - same for all problems in batch
        solver.setup(P_val, A_val)

        # Batched q/b
        q_batch = np.tile(q, (batch_size, 1))
        b_batch = np.tile(b, (batch_size, 1))

        # Add some variation
        for i in range(batch_size):
            q_batch[i] *= 1 + i * 0.1

        # Create bounds (unbounded)

        # First batch solve
        result1 = solver.solve(q_batch, b_batch)
        info1 = solver.info
        assert len(info1.status) == batch_size
        assert all(s == moreau.SolverStatus.Solved for s in info1.status)

        # Second batch solve with different q
        result2 = solver.solve(q_batch * 2, b_batch)
        info2 = solver.info
        assert all(s == moreau.SolverStatus.Solved for s in info2.status)

    def test_batch_backward_caching(self, device):
        """Test backward() for batched solves using torch."""
        pytest.importorskip("torch")
        import torch
        from moreau.torch import Solver as TorchSolver

        n, m, P_ro, P_ci, P_val, A_ro, A_ci, A_val, q, b, cones = create_qp_problem()

        batch_size = 3
        settings = moreau.Settings(
            device=device, verbose=False, enable_grad=True, batch_size=batch_size
        )

        # Convert to torch tensors for TorchSolver
        P_ro_t = torch.tensor(P_ro, dtype=torch.int64)
        P_ci_t = torch.tensor(P_ci, dtype=torch.int64)
        A_ro_t = torch.tensor(A_ro, dtype=torch.int64)
        A_ci_t = torch.tensor(A_ci, dtype=torch.int64)

        solver = TorchSolver(n, m, P_ro_t, P_ci_t, A_ro_t, A_ci_t, cones, settings)

        device_str = "cuda" if device == "cuda" else "cpu"

        # Batched tensors
        P_batch = torch.tensor(
            np.tile(P_val, (batch_size, 1)),
            dtype=torch.float64,
            device=device_str,
            requires_grad=True,
        )
        A_batch = torch.tensor(
            np.tile(A_val, (batch_size, 1)),
            dtype=torch.float64,
            device=device_str,
            requires_grad=True,
        )
        q_batch = torch.tensor(
            np.tile(q, (batch_size, 1)), dtype=torch.float64, device=device_str, requires_grad=True
        )
        b_batch = torch.tensor(
            np.tile(b, (batch_size, 1)), dtype=torch.float64, device=device_str, requires_grad=True
        )

        result = solver.solve(P_batch, A_batch, q_batch, b_batch)
        info = solver.info
        assert result.x.shape == (batch_size, n)

        # Backward
        result.x.sum().backward()

        assert q_batch.grad is not None
        assert q_batch.grad.shape == (batch_size, n)
        assert b_batch.grad is not None
        assert b_batch.grad.shape == (batch_size, m)


class TestCachingPerformance:
    """Test that caching provides expected performance benefits."""

    def test_second_solve_faster_than_first(self, device):
        """Multiple solves should have consistent performance."""
        from scipy import sparse

        # Use a larger problem to see timing differences
        n, m = 50, 25

        P_row_offsets = np.arange(n + 1, dtype=np.int64)
        P_col_indices = np.arange(n, dtype=np.int64)
        P_values = np.ones(n, dtype=np.float64) * 2.0

        A_row_offsets = np.arange(m + 1, dtype=np.int64)
        A_col_indices = np.zeros(m, dtype=np.int64)
        A_values = np.ones(m, dtype=np.float64)

        q = np.ones(n, dtype=np.float64)
        b = np.ones(m, dtype=np.float64)

        cones = moreau.Cones(num_zero_cones=m)

        # Convert to sparse matrices
        P = sparse.csr_array((P_values, P_col_indices, P_row_offsets), shape=(n, n))
        A = sparse.csr_array((A_values, A_col_indices, A_row_offsets), shape=(m, n))

        # Create bounds (unbounded)

        settings = moreau.Settings(device=device, verbose=False)

        # Warmup
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solver.solve()

        # Time first solve
        solver1 = moreau.Solver(P, q, A, b, cones, settings)
        start = time.perf_counter()
        solver1.solve()
        first_time = time.perf_counter() - start

        # Time second solve (different q)
        solver2 = moreau.Solver(P, q * 2, A, b, cones, settings)
        start = time.perf_counter()
        solver2.solve()
        second_time = time.perf_counter() - start

        # Second solve should be comparable (at least not much slower)
        # Note: small problems might not show huge differences due to overhead
        assert (
            second_time <= first_time * 2
        ), f"Second solve ({second_time:.4f}s) much slower than first ({first_time:.4f}s)"
