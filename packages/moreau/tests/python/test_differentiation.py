"""
Unit tests for Moreau backward differentiation.

Tests verify:
- Backward differentiation: (dx, dz, ds) -> (dP, dA, dq, db)
- Finite difference validation
- PyTorch autograd integration

Tests run on both CPU and CUDA (when available).
"""

import numpy as np
import pytest
from scipy import sparse

# Check if PyTorch is available
try:
    import torch

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

# Check if moreau with Solver is available
try:
    import moreau
    from moreau.torch import Solver

    HAS_MOREAU_TORCH = Solver is not None
except ImportError:
    HAS_MOREAU_TORCH = False
    Solver = None

# Skip all tests if dependencies not available
pytestmark = [
    pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed"),
    pytest.mark.skipif(not HAS_MOREAU_TORCH, reason="moreau Solver not available"),
]

# Test tolerances
TOL_GRAD = 1e-3  # Tolerance for gradient checks vs finite differences
FINITE_DIFF_H = 1e-6  # Step size for finite differences


@pytest.fixture
def simple_qp_problem(device):
    """Create a simple QP problem for testing differentiation."""
    n = 2  # 2 variables
    m = 3  # 3 constraints

    # P matrix (2x2 identity) in CSR format
    P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
    P_col_indices = torch.tensor([0, 1], dtype=torch.int64)
    P_values = torch.tensor([[1.0, 1.0]], dtype=torch.float64, device=device)

    # Linear cost vector
    q = torch.tensor([[2.0, 1.0]], dtype=torch.float64, device=device)

    # A matrix in CSR format
    A_row_offsets = torch.tensor([0, 2, 3, 4], dtype=torch.int64)
    A_col_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    A_values = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float64, device=device)

    # RHS vector
    b = torch.tensor([[1.0, 2.0, 2.0]], dtype=torch.float64, device=device)

    # Cone structure
    cones = moreau.Cones()
    cones.num_zero_cones = 1
    cones.num_nonneg_cones = 2

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "q": q,
        "b": b,
        "cones": cones,
        "device": device,
    }


def solve_qp(P_values, A_values, q, b, solver):
    """Helper to solve QP and return solution."""
    result = solver.solve(q, b)
    info = solver.info
    return result.x, result.z, result.s


class TestBackwardDifferentiation:
    """Tests for backward differentiation pass."""

    def test_backward_basic(self, simple_qp_problem):
        """Test basic backward differentiation call."""
        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(
            verbose=False,
            batch_size=1,
            device=device,
        )

        solver = Solver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings,
        )

        # Solve first
        result = solver.solve(p["P_values"], p["A_values"], p["q"], p["b"])
        info = solver.info
        x, z, s = result.x, result.z, result.s

        # Create upstream gradients
        dx = torch.ones_like(x)
        dz = torch.zeros_like(z)
        ds = torch.zeros_like(s)

        # Call backward
        dP_values, dq, dA_values, db = solver.backward(dx, dz, ds)

        # Check output shapes
        assert dP_values.shape == p["P_values"].shape
        assert dq.shape == p["q"].shape
        assert dA_values.shape == p["A_values"].shape
        assert db.shape == p["b"].shape

        # Check outputs are on correct device
        assert dP_values.device.type == device
        assert dq.device.type == device
        assert dA_values.device.type == device
        assert db.device.type == device

        # Check outputs are finite
        assert torch.all(torch.isfinite(dP_values))
        assert torch.all(torch.isfinite(dq))
        assert torch.all(torch.isfinite(dA_values))
        assert torch.all(torch.isfinite(db))

    def test_backward_vs_finite_diff_q(self, simple_qp_problem):
        """Test backward gradient w.r.t. q against finite differences."""
        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(
            verbose=False,
            batch_size=1,
            device=device,
        )

        solver = Solver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings,
        )

        # Solve base problem
        result = solver.solve(p["P_values"], p["A_values"], p["q"], p["b"])
        info = solver.info
        x0, z0, s0 = result.x, result.z, result.s

        # Upstream gradient: gradient of x[0] w.r.t. solution
        dx = torch.zeros_like(x0)
        dx[0, 0] = 1.0
        dz = torch.zeros_like(z0)
        ds = torch.zeros_like(s0)

        # Backward pass
        dP_values, dq, dA_values, db = solver.backward(dx, dz, ds)

        # Compute finite difference gradient for q[0]
        h = FINITE_DIFF_H
        q_plus = p["q"].clone()
        q_plus[0, 0] += h
        result_plus = solver.solve(p["P_values"], p["A_values"], q_plus, p["b"])
        info = solver.info
        x_plus = result_plus.x

        q_minus = p["q"].clone()
        q_minus[0, 0] -= h
        result_minus = solver.solve(p["P_values"], p["A_values"], q_minus, p["b"])
        info = solver.info
        x_minus = result_minus.x

        fd_grad = (x_plus[0, 0] - x_minus[0, 0]) / (2 * h)

        # Compare: backward gives dq, which is dx^T * (dx/dq)
        # For scalar upstream dx[0]=1, dq[0] should equal dx[0]/dq[0]
        analytic_grad = dq[0, 0].item()

        print(
            f"[{device}] Backward dq[0]: analytic={analytic_grad:.6f}, finite_diff={fd_grad.item():.6f}"
        )
        assert (
            abs(analytic_grad - fd_grad.item()) < TOL_GRAD
        ), f"Backward gradient mismatch: analytic={analytic_grad}, fd={fd_grad.item()}"

    def test_backward_vs_finite_diff_b(self, simple_qp_problem):
        """Test backward gradient w.r.t. b against finite differences."""
        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(
            verbose=False,
            batch_size=1,
            device=device,
        )

        solver = Solver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings,
        )

        # Solve base problem
        result = solver.solve(p["P_values"], p["A_values"], p["q"], p["b"])
        info = solver.info
        x0, z0, s0 = result.x, result.z, result.s

        # Upstream gradient: gradient of x[0]
        dx = torch.zeros_like(x0)
        dx[0, 0] = 1.0
        dz = torch.zeros_like(z0)
        ds = torch.zeros_like(s0)

        # Backward pass
        dP_values, dq, dA_values, db = solver.backward(dx, dz, ds)

        # Compute finite difference gradient for b[0]
        h = FINITE_DIFF_H
        b_plus = p["b"].clone()
        b_plus[0, 0] += h
        result_plus = solver.solve(p["P_values"], p["A_values"], p["q"], b_plus)
        info = solver.info
        x_plus = result_plus.x

        b_minus = p["b"].clone()
        b_minus[0, 0] -= h
        result_minus = solver.solve(p["P_values"], p["A_values"], p["q"], b_minus)
        info = solver.info
        x_minus = result_minus.x

        fd_grad = (x_plus[0, 0] - x_minus[0, 0]) / (2 * h)
        analytic_grad = db[0, 0].item()

        print(
            f"[{device}] Backward db[0]: analytic={analytic_grad:.6f}, finite_diff={fd_grad.item():.6f}"
        )
        assert (
            abs(analytic_grad - fd_grad.item()) < TOL_GRAD
        ), f"Backward gradient mismatch: analytic={analytic_grad}, fd={fd_grad.item()}"


class TestPyTorchAutograd:
    """Tests for PyTorch autograd integration."""

    def test_autograd_backward(self, simple_qp_problem):
        """Test that PyTorch autograd backward works."""
        # MoreauSolveFunction imported at module level

        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(
            verbose=False,
            batch_size=1,
            device=device,
        )

        solver = Solver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings,
        )

        # Enable gradient tracking on inputs
        q = p["q"].clone().requires_grad_(True)
        b = p["b"].clone().requires_grad_(True)
        P_values = p["P_values"].clone().requires_grad_(True)
        A_values = p["A_values"].clone().requires_grad_(True)

        # Forward pass through autograd function
        result = solver.solve(P_values, A_values, q, b)
        info = solver.info
        x = result.x

        # Compute loss and backward
        loss = x.sum()
        loss.backward()

        # Check gradients exist and are finite
        assert q.grad is not None, "q.grad should exist"
        assert b.grad is not None, "b.grad should exist"
        assert P_values.grad is not None, "P_values.grad should exist"
        assert A_values.grad is not None, "A_values.grad should exist"

        assert torch.all(torch.isfinite(q.grad)), "q.grad should be finite"
        assert torch.all(torch.isfinite(b.grad)), "b.grad should be finite"

        print(f"[{device}] q.grad = {q.grad}")
        print(f"[{device}] b.grad = {b.grad}")

    def test_autograd_vs_manual_backward(self, simple_qp_problem):
        """Test that autograd backward matches manual backward call."""
        # MoreauSolveFunction imported at module level

        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(
            verbose=False,
            batch_size=1,
            device=device,
        )

        solver = Solver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings,
        )

        # Manual backward
        result_manual = solver.solve(p["P_values"], p["A_values"], p["q"], p["b"])
        info = solver.info
        x_manual, z_manual, s_manual = result_manual.x, result_manual.z, result_manual.s
        dx = torch.ones_like(x_manual)
        dz = torch.zeros_like(z_manual)
        ds = torch.zeros_like(s_manual)
        dP_manual, dq_manual, dA_manual, db_manual = solver.backward(dx, dz, ds)

        # Reset solver to get fresh state (will reinitialize on next solve)
        solver.reset()

        # Autograd backward
        q = p["q"].clone().requires_grad_(True)
        b = p["b"].clone().requires_grad_(True)
        P_values = p["P_values"].clone().requires_grad_(True)
        A_values = p["A_values"].clone().requires_grad_(True)

        result = solver.solve(P_values, A_values, q, b)
        info = solver.info
        x_auto = result.x
        loss = x_auto.sum()  # This gives upstream gradient of all ones for x
        loss.backward()

        # Compare gradients
        print(f"[{device}] Manual dq: {dq_manual}")
        print(f"[{device}] Autograd q.grad: {q.grad}")

        torch.testing.assert_close(q.grad, dq_manual, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(b.grad, db_manual, rtol=1e-5, atol=1e-5)


class TestTorchSolverDifferentiation:
    """Tests for Solver with differentiation."""

    def test_solver_backward(self, simple_qp_problem):
        """Test backward pass through Solver."""
        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(
            verbose=False,
            batch_size=1,
            device=device,
        )

        solver = Solver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings,
        )

        # Enable gradient tracking
        q = p["q"].clone().requires_grad_(True)

        # Forward
        solution = solver.solve(p["P_values"], p["A_values"], q, p["b"])

        # Backward
        loss = solution.x.sum()
        loss.backward()

        assert q.grad is not None
        assert torch.all(torch.isfinite(q.grad))
        print(f"[{device}] Solver q.grad = {q.grad}")


class TestBatchedDifferentiation:
    """Tests for batched differentiation."""

    def test_batched_backward(self, simple_qp_problem):
        """Test backward with batch_size > 1."""
        p = simple_qp_problem
        device = p["device"]
        batch_size = 4

        settings = moreau.Settings(
            verbose=False,
            batch_size=batch_size,
            device=device,
            enable_grad=True,
        )

        solver = Solver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings,
        )

        # Create batched inputs
        P_batch = p["P_values"].expand(batch_size, -1).contiguous()
        A_batch = p["A_values"].expand(batch_size, -1).contiguous()
        q_batch = p["q"].expand(batch_size, -1).contiguous()
        b_batch = p["b"].expand(batch_size, -1).contiguous()

        # Solve
        result = solver.solve(P_batch, A_batch, q_batch, b_batch)
        info = solver.info
        x, z, s = result.x, result.z, result.s

        # Backward with batched upstream gradients
        dx = torch.randn_like(x)
        dz = torch.randn_like(z)
        ds = torch.randn_like(s)

        dP, dq, dA, db = solver.backward(dx, dz, ds)

        # Check shapes
        assert dP.shape == (batch_size, solver.nnzP)
        assert dq.shape == (batch_size, p["n"])
        assert dA.shape == (batch_size, solver.nnzA)
        assert db.shape == (batch_size, p["m"])

        # Check all finite
        assert torch.all(torch.isfinite(dP))
        assert torch.all(torch.isfinite(dq))
        assert torch.all(torch.isfinite(dA))
        assert torch.all(torch.isfinite(db))


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "method,storage,interface",
    [
        (method, storage, interface)
        for method in ("auto", "active_set", "ipm")
        for storage in ("full", "full_unsorted")
        for interface in ("numpy", "torch")
    ]
    + [("active_set", storage, "torch") for storage in ("upper", "lower")],
)
def test_equality_qp_symmetric_P_gradient(method, storage, interface):
    """Full CSR shares an off-diagonal sensitivity; triangular CSR stores it once."""
    P = np.array([[3.0, 0.7], [0.7, 2.0]])
    q = np.array([-1.0, 0.3])
    A = sparse.csr_matrix([[1.0, 2.0]])
    b = np.array([1.4])
    dx = np.array([0.4, -0.8])
    cones = moreau.Cones(num_zero_cones=1)
    settings = moreau.Settings(
        device="cpu",
        solver=method,
        enable_grad=True,
        verbose=False,
        ipm_settings=moreau.IPMSettings(
            diff_method="exact", tol_feas=1e-11, tol_gap_abs=1e-11, tol_gap_rel=1e-11
        ),
    )
    matrix = sparse.csr_matrix(
        P if storage.startswith("full") else np.triu(P) if storage == "upper" else np.tril(P)
    )
    if storage == "full_unsorted":
        for start, end in zip(matrix.indptr[:-1], matrix.indptr[1:]):
            matrix.indices[start:end] = matrix.indices[start:end][::-1]
            matrix.data[start:end] = matrix.data[start:end][::-1]
    if interface == "numpy":
        solver = moreau.Solver(matrix, q, A, b, cones, settings)
        sol = solver.solve()
        x = sol.x
        gradients = {name: value.reshape(-1) for name, value in solver.backward(dx).items()}
        grad = gradients["dP_values"]
    else:
        solver = Solver(
            n=2,
            m=1,
            P_row_offsets=torch.tensor(matrix.indptr),
            P_col_indices=torch.tensor(matrix.indices),
            A_row_offsets=torch.tensor(A.indptr),
            A_col_indices=torch.tensor(A.indices),
            cones=cones,
            settings=settings,
        )
        values, av, qt, bt = [
            torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for v in (matrix.data, A.data, q, b)
        ]
        sol = solver.solve(values, av, qt, bt)
        (sol.x @ torch.tensor(dx)).backward()
        x = sol.x.detach().numpy()
        gradients = {
            name: tensor.grad.numpy()
            for name, tensor in zip(("dP_values", "dA_values", "dq", "db"), (values, av, qt, bt))
        }
        grad = gradients["dP_values"]
    assert solver._settings.solver == (
        moreau.SolverType.IPM if method == "ipm" else moreau.SolverType.ACTIVE_SET
    )

    def analytic_solution(H):
        K = np.block([[H, A.toarray().T], [A.toarray(), np.zeros((1, 1))]])
        return np.linalg.solve(K, np.r_[-q, b])[:2]

    np.testing.assert_allclose(x, analytic_solution(P), atol=1e-7)
    row = np.repeat(np.arange(2), np.diff(matrix.indptr))
    col = matrix.indices
    # Compare algorithms explicitly, including the saved-state autograd path.
    # Keep the independent analytic oracle below: agreement alone is insufficient.
    ipm_settings = settings.model_copy(deep=True)
    ipm_settings.solver = moreau.SolverType.IPM
    reference = moreau.Solver(sparse.csr_matrix(P), q, A, b, cones, ipm_settings)
    ref_sol = reference.solve()
    np.testing.assert_allclose(x, ref_sol.x, atol=1e-7)
    ref_grad = reference.backward(dx)
    for name in ("dP_values", "dq", "dA_values", "db"):
        expected = ref_grad[name].reshape(-1)
        if name == "dP_values":
            expected = expected.reshape(2, 2)[row, col]
            if not storage.startswith("full"):
                expected = expected * np.where(row == col, 1.0, 2.0)
        np.testing.assert_allclose(
            gradients[name],
            expected,
            atol=1e-7,
            rtol=1e-6,
            err_msg=f"{method} vs IPM: {name}",
        )
    for direction in (
        np.diag([1.0, 0.0]),
        np.diag([0.0, 1.0]),
        np.array([[0.0, 1.0], [1.0, 0.0]]),
    ):
        eps = 1e-5
        fd = (
            dx
            @ (analytic_solution(P + eps * direction) - analytic_solution(P - eps * direction))
            / (2 * eps)
        )
        assert abs(fd) > 1e-3
        np.testing.assert_allclose(grad @ direction[row, col], fd, atol=1e-7, rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
