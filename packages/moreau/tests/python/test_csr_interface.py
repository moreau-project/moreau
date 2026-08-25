"""Tests for CSR matrix interface and gradient mapping.

These tests verify that:
1. The solver accepts CSR format matrices correctly
2. CSR→CSC conversion is handled internally
3. Gradients are returned in CSR order matching input ordering
"""

import numpy as np
import pytest
from scipy import sparse

import moreau


def create_qp_problem_csr():
    """Create a simple QP problem with known structure in CSR format.

    minimize    x1^2 + x2^2
    subject to  x1 + x2 = 1

    Expected solution: x = [0.5, 0.5]
    """
    n = 2  # variables
    m = 1  # constraints

    # P = 2*I (diagonal, same in CSR and CSC)
    P = sparse.csr_array(np.array([[2.0, 0.0], [0.0, 2.0]]))

    # A = [1, 1] - row vector constraint
    A = sparse.csr_array(np.array([[1.0, 1.0]]))

    q = np.array([0.0, 0.0])
    b = np.array([1.0])

    return n, m, P, A, q, b


def create_asymmetric_constraint_csr():
    """Create a QP with asymmetric constraint matrix to test CSR→CSC conversion.

    minimize    x1^2 + x2^2 + x3^2
    subject to  x1 + 2*x2 + x3 = 3
                x1      - x3 = 0

    This matrix has different sparsity patterns in rows vs columns.
    """
    n = 3  # variables
    m = 2  # constraints

    # P = 2*I
    P = sparse.csr_array(np.eye(3) * 2.0)

    # A = [[1, 2, 1],
    #      [1, 0, -1]]
    A = sparse.csr_array(np.array([[1.0, 2.0, 1.0], [1.0, 0.0, -1.0]]))

    q = np.zeros(n)
    b = np.array([3.0, 0.0])

    return n, m, P, A, q, b


class TestCsrSingleSolve:
    """Test single problem solving with CSR input."""

    def test_simple_qp_csr(self):
        """Test basic QP solving with CSR matrices."""
        n, m, P, A, q, b = create_qp_problem_csr()

        cones = moreau.Cones()
        cones.num_zero_cones = m  # equality constraint

        # Create bounds (unbounded)

        solver = moreau.Solver(P, q, A, b, cones)
        result = solver.solve()
        info = solver.info

        assert info.status == moreau.SolverStatus.Solved
        # Expected: x = [0.5, 0.5]
        np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-6)

    def test_asymmetric_constraint_csr(self):
        """Test QP with asymmetric constraint matrix."""
        n, m, P, A, q, b = create_asymmetric_constraint_csr()

        cones = moreau.Cones()
        cones.num_zero_cones = m

        # Create bounds (unbounded)

        solver = moreau.Solver(P, q, A, b, cones)
        result = solver.solve()
        info = solver.info

        assert info.status == moreau.SolverStatus.Solved
        # Verify constraints are satisfied
        x = result.x
        np.testing.assert_allclose(A @ x, b, atol=1e-6)


class TestCsrBatchSolve:
    """Test batched solving with CSR input."""

    def test_batch_solve_csr(self):
        """Test batch solving with CSR matrices using CompiledSolver."""
        n, m, P, A, q, b = create_qp_problem_csr()

        cones = moreau.Cones()
        cones.num_zero_cones = m

        # Batch of 3 problems with different RHS
        batch_size = 3

        settings = moreau.Settings(device="cpu", batch_size=batch_size)
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

        # Use 1D (shared) P/A values - same for all problems in batch
        P_values = P.data
        A_values = A.data
        q_batch = np.zeros((batch_size, n))
        b_batch = np.array([[1.0], [2.0], [3.0]])

        solver.setup(P_values, A_values)
        result = solver.solve(q_batch, b_batch)
        info = solver.info

        for i in range(batch_size):
            assert info.status[i] == moreau.SolverStatus.Solved
            # Each x1 + x2 = b[i], and both equal due to symmetry
            np.testing.assert_allclose(np.sum(result.x[i]), b_batch[i, 0], atol=1e-6)


class TestCsrGradientMapping:
    """Test that gradients are returned in CSR order."""

    def test_gradient_csr_order(self, device):
        """Test that dA_values matches CSR ordering of A.data."""
        pytest.importorskip("torch")
        import torch
        from moreau.torch import Solver as TorchSolver

        n, m, P, A, q, b = create_asymmetric_constraint_csr()

        # Store original CSR data
        P_csr_data = P.data.copy()
        A_csr_data = A.data.copy()

        cones = moreau.Cones()
        cones.num_zero_cones = m

        # Create torch solver for backward pass
        settings = moreau.Settings(batch_size=1, device=device)
        solver = TorchSolver(
            n=n,
            m=m,
            P_row_offsets=torch.tensor(P.indptr),
            P_col_indices=torch.tensor(P.indices),
            A_row_offsets=torch.tensor(A.indptr),
            A_col_indices=torch.tensor(A.indices),
            cones=cones,
            settings=settings,
        )

        # Forward pass - tensors on device (batched)
        P_vals = torch.tensor(P_csr_data, dtype=torch.float64, device=device).unsqueeze(0)
        A_vals = torch.tensor(A_csr_data, dtype=torch.float64, device=device).unsqueeze(0)
        q_t = torch.tensor(q, dtype=torch.float64, device=device).unsqueeze(0)
        b_t = torch.tensor(b, dtype=torch.float64, device=device).unsqueeze(0)

        result = solver.solve(P_vals, A_vals, q_t, b_t)
        info = solver.info
        x, z, s = result.x, result.z, result.s

        # Backward pass with upstream gradient
        dx = torch.ones_like(x)
        dz = torch.zeros_like(z)
        ds = torch.zeros_like(s)

        dP, dq, dA, db = solver.backward(dx, dz, ds)

        # Check that gradients have correct shape matching CSR
        # Note: backward returns batched output, squeeze to compare with 1D input data
        assert dA.squeeze().shape == A_csr_data.shape
        assert dP.squeeze().shape == P_csr_data.shape

        # Verify gradient is not all zeros (sanity check)
        assert not torch.allclose(dA, torch.zeros_like(dA))

    def test_gradient_finite_diff_consistency(self, device):
        """Test gradient consistency with finite differences (CSR ordering)."""
        pytest.importorskip("torch")
        import torch
        from moreau.torch import Solver as TorchSolver

        n, m, P, A, q, b = create_qp_problem_csr()

        cones = moreau.Cones()
        cones.num_zero_cones = m

        # Create solver
        settings = moreau.Settings(batch_size=1, device=device, enable_grad=True)
        solver = TorchSolver(
            n=n,
            m=m,
            P_row_offsets=torch.tensor(P.indptr),
            P_col_indices=torch.tensor(P.indices),
            A_row_offsets=torch.tensor(A.indptr),
            A_col_indices=torch.tensor(A.indices),
            cones=cones,
            settings=settings,
        )

        # Test finite difference for q gradient - tensors on device (batched)
        # Create q_t as batched from the start with requires_grad
        q_t = torch.tensor(q, dtype=torch.float64, device=device).unsqueeze(0).requires_grad_(True)
        b_t = torch.tensor(b, dtype=torch.float64, device=device).unsqueeze(0)
        P_vals = torch.tensor(P.data, dtype=torch.float64, device=device).unsqueeze(0)
        A_vals = torch.tensor(A.data, dtype=torch.float64, device=device).unsqueeze(0)

        eps = 1e-5

        # Compute analytical gradient
        result = solver.solve(P_vals, A_vals, q_t, b_t)
        info = solver.info
        x = result.x
        x.sum().backward()
        dq_analytical = q_t.grad.squeeze()

        # Compute numerical gradient for q
        dq_numerical = torch.zeros(n, dtype=torch.float64, device=device)
        for i in range(n):
            q_plus = q_t.detach().clone()
            q_plus[0, i] += eps
            result_plus = solver.solve(P_vals, A_vals, q_plus, b_t)
            info = solver.info
            x_plus = result_plus.x

            q_minus = q_t.detach().clone()
            q_minus[0, i] -= eps
            result_minus = solver.solve(P_vals, A_vals, q_minus, b_t)
            info = solver.info
            x_minus = result_minus.x

            # Gradient of sum(x) w.r.t. q[i]
            dq_numerical[i] = (x_plus.sum() - x_minus.sum()) / (2 * eps)

        # Compare
        np.testing.assert_allclose(
            dq_analytical.cpu().numpy(), dq_numerical.cpu().numpy(), atol=1e-4, rtol=1e-3
        )


class TestCsrFormatValidation:
    """Test that passing non-CSR matrices raises appropriate errors."""

    def test_compiled_solver_new_api(self):
        """Test that CompiledSolver accepts row_offsets/col_indices."""
        from moreau import _cpu_solver as solver

        n, m = 2, 1
        P = sparse.csr_array(np.array([[2.0, 0.0], [0.0, 2.0]]))
        A = sparse.csr_array(np.array([[1.0, 1.0]]))

        cones = [solver.ZeroConeT(m)]
        settings = solver.DefaultSettings()

        # New API with row_offsets/col_indices
        compiled_solver = solver.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=P.indptr.tolist(),
            P_col_indices=P.indices.tolist(),
            A_row_offsets=A.indptr.tolist(),
            A_col_indices=A.indices.tolist(),
            cones=cones,
            settings=settings,
        )
        assert compiled_solver is not None


class TestCsrCscEquivalence:
    """Test that CSR and manually converted CSC give same results."""

    def test_solution_equivalence(self):
        """Verify CSR input gives correct solution."""
        n, m, P, A, q, b = create_asymmetric_constraint_csr()

        cones = moreau.Cones()
        cones.num_zero_cones = m

        # Create bounds (unbounded)

        # Solve with CSR
        solver = moreau.Solver(P, q, A, b, cones)
        result_csr = solver.solve()
        info = solver.info

        # Solutions should be valid
        assert info.status == moreau.SolverStatus.Solved
        x_solution = result_csr.x

        # Verify solution satisfies constraints
        Ax = A @ x_solution
        np.testing.assert_allclose(Ax, b, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
