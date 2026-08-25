"""
Unit tests for Moreau PyTorch integration.

Tests run on both CPU and CUDA (when available).
"""

import numpy as np
import pytest

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


@pytest.fixture
def simple_qp_problem(device):
    """Create a simple QP problem for testing."""
    n = 2  # 2 variables
    m = 3  # 3 constraints

    # P matrix (2x2 identity) in CSR format
    P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
    P_col_indices = torch.tensor([0, 1], dtype=torch.int64)
    # P_values with batch dimension [1, nnzP]
    P_values = torch.tensor([[1.0, 1.0]], dtype=torch.float64, device=device)

    # Linear cost vector with batch dimension [1, n]
    q = torch.tensor([[2.0, 1.0]], dtype=torch.float64, device=device)

    # A matrix in CSR format
    A_row_offsets = torch.tensor([0, 2, 3, 4], dtype=torch.int64)
    A_col_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
    # A_values with batch dimension [1, nnzA]
    A_values = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float64, device=device)

    # RHS vector with batch dimension [1, m]
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


class TestZeroCopyForward:
    """Tests for forward pass."""

    def test_basic_solve(self, simple_qp_problem):
        """Test basic solve with tensors."""
        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(device=device, max_iter=50, verbose=False)

        # Create solver with fixed batch_size
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

        result = solver.solve(p["P_values"], p["A_values"], p["q"], p["b"])
        info = solver.info
        x, z, s = result.x, result.z, result.s

        # Check outputs are on correct device
        assert x.device.type == device, f"x should be on {device}"
        assert z.device.type == device, f"z should be on {device}"
        assert s.device.type == device, f"s should be on {device}"
        # status is now a SolverStatus enum, not a tensor

        # Check dtypes
        assert x.dtype == torch.float64
        assert z.dtype == torch.float64
        assert s.dtype == torch.float64

        # Check shapes - outputs are always [batch, dim]
        assert x.shape == (1, p["n"])
        assert z.shape == (1, p["m"])
        assert s.shape == (1, p["m"])

        # Check solver converged (status is list of SolverStatus for batched)
        status = info.status[0] if isinstance(info.status, list) else info.status
        assert status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ], f"Expected Solved, got status {status}"

        # Check solution accuracy
        expected_x = torch.tensor(
            [[2.4308818538642494e-10, 0.9999999997569121]], device=device, dtype=torch.float64
        )
        torch.testing.assert_close(x, expected_x, rtol=1e-6, atol=1e-6)

    def test_output_stays_on_device(self, simple_qp_problem):
        """Verify that outputs stay on device without CPU roundtrip."""
        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(device=device, verbose=False)

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

        result = solver.solve(p["P_values"], p["A_values"], p["q"], p["b"])
        info = solver.info
        x = result.x

        # Perform operations on output without CPU copy
        x_squared = x**2
        x_sum = x.sum()

        # Should still be on same device
        assert x_squared.device.type == device
        assert x_sum.device.type == device

    def test_contiguous_input_handling(self, simple_qp_problem):
        """Test that non-contiguous inputs are handled correctly."""
        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(device=device, verbose=False)

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

        # Create non-contiguous tensor by expanding
        # p['q'] is already [1, n], expand to [2, n] then slice back
        q_expanded = p["q"].expand(2, -1)  # Shape: [2, 2], non-contiguous
        q_noncontig = q_expanded[0:1]  # Shape: [1, 2], may be non-contiguous

        # Should still work (binding makes contiguous internally)
        result = solver.solve(p["P_values"], p["A_values"], q_noncontig, p["b"])
        info = solver.info
        x = result.x

        assert x.device.type == device

    def test_batched_inputs(self, simple_qp_problem):
        """Test solve with batched (batch, dim) shaped inputs."""
        p = simple_qp_problem
        device = p["device"]

        # Create solver with fixed batch_size
        batch_size = 3
        settings = moreau.Settings(device=device, batch_size=batch_size, verbose=False)

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

        # Create batched inputs with shape [batch, dim]
        # p['q'] is already [1, n], expand to [batch_size, n]
        q_batched = p["q"].expand(batch_size, -1).contiguous()  # [3, 2]
        b_batched = p["b"].expand(batch_size, -1).contiguous()  # [3, 3]
        P_batched = p["P_values"].expand(batch_size, -1).contiguous()  # [3, nnzP]
        A_batched = p["A_values"].expand(batch_size, -1).contiguous()  # [3, nnzA]

        result = solver.solve(P_batched, A_batched, q_batched, b_batched)
        info = solver.info
        x, z, s = result.x, result.z, result.s

        # Check shapes
        assert x.shape == (batch_size, p["n"])
        assert z.shape == (batch_size, p["m"])
        assert s.shape == (batch_size, p["m"])

    def test_fixed_batch_size(self, simple_qp_problem):
        """Test that solver enforces fixed batch size from constructor."""
        p = simple_qp_problem
        device = p["device"]

        # Create solver with batch_size=2 for eager initialization
        settings = moreau.Settings(device=device, batch_size=2)
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

        # Verify batch_size accessor
        assert solver.batch_size == 2

        # Solve with correct batch_size=2
        # p['q'] is [1, n], expand to [2, n]
        q2 = p["q"].expand(2, -1).contiguous()
        b2 = p["b"].expand(2, -1).contiguous()
        P2 = p["P_values"].expand(2, -1).contiguous()
        A2 = p["A_values"].expand(2, -1).contiguous()
        result2 = solver.solve(P2, A2, q2, b2)
        info = solver.info
        x2 = result2.x
        assert x2.shape == (2, p["n"])


class TestSolverProperties:
    """Tests for solver properties and settings."""

    def test_dimensions(self, simple_qp_problem):
        """Test solver dimension reporting."""
        p = simple_qp_problem
        device = p["device"]

        # Create solver with explicit batch_size=1 for preinitialization
        settings = moreau.Settings(device=device, batch_size=1)
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

        assert solver.n == p["n"]
        assert solver.m == p["m"]
        assert solver.batch_size == 1
        assert solver.nnzP == len(p["P_col_indices"])
        assert solver.nnzA == len(p["A_col_indices"])

        dims = solver.get_dimensions()
        assert dims["n"] == p["n"]
        assert dims["m"] == p["m"]
        assert dims["batch_size"] == 1

    def test_settings(self):
        """Test Settings class."""
        settings = moreau.Settings()

        # Test default values (unified Settings has verbose=False by default)
        assert settings.max_iter == 200
        assert settings.verbose == False
        assert settings.ipm_settings.tol_gap_abs == 1e-8

        # Test modification - tolerances are in IPMSettings
        settings.max_iter = 100
        settings.verbose = False
        settings.ipm_settings.tol_gap_abs = 1e-6

        assert settings.max_iter == 100
        assert settings.verbose == False
        assert settings.ipm_settings.tol_gap_abs == 1e-6

    def test_cones(self):
        """Test Cones class."""
        cones = moreau.Cones()

        # Test default values
        assert cones.num_zero_cones == 0
        assert cones.num_nonneg_cones == 0

        # Test setting values
        cones.num_zero_cones = 2
        cones.num_nonneg_cones = 5
        cones.so_cone_dims = [3]

        assert cones.total_constraints() == 2 + 5 + 3  # 3 per SOC
        assert cones.degree() == 0 + 5 + 1  # 1 per SOC


class TestAutogradFunction:
    """Tests for autograd through solver.solve()."""

    def test_solve_with_autograd(self, simple_qp_problem):
        """Test forward pass with built-in autograd."""
        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(device=device, verbose=False)

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

        # Just call solve() - autograd is built-in
        result = solver.solve(p["P_values"], p["A_values"], p["q"], p["b"])
        info = solver.info
        x = result.x

        assert x.device.type == device
        # Output is always [batch, dim]
        assert x.shape == (1, p["n"])

    def test_backward_computes_gradients(self, simple_qp_problem):
        """Test that backward computes valid gradients."""
        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(device=device, verbose=False)

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

        # Enable grad tracking for q
        q = p["q"].clone().requires_grad_(True)

        # Just call solve() - autograd is built-in
        result = solver.solve(p["P_values"], p["A_values"], q, p["b"])
        info = solver.info
        x = result.x

        # Backward should work and compute gradients
        loss = x.sum()
        loss.backward()

        # q should have gradients
        assert q.grad is not None, "Expected gradients for q"
        assert q.grad.shape == q.shape, "Gradient shape should match q shape"
        # Gradient should be non-zero for this problem
        assert torch.abs(q.grad).sum() > 0, "Expected non-zero gradients"


class TestTorchSolverForward:
    """Tests for Solver forward pass."""

    def test_solver_forward(self, simple_qp_problem):
        """Test forward pass through Solver."""
        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(device=device, verbose=False)

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

        solution = solver.solve(p["P_values"], p["A_values"], p["q"], p["b"])

        assert solution.x.device.type == device
        # Output shape depends on input shape
        assert solution.x.shape[-1] == p["n"]


class TestInputValidation:
    """Tests for input validation."""

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_cross_device_solve_returns_to_input_device(self):
        """Test that results are returned on the input device, not the solver device."""
        n, m = 2, 3
        P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1], dtype=torch.int64)
        A_row_offsets = torch.tensor([0, 2, 3, 4], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_zero_cones = 1
        cones.num_nonneg_cones = 2

        # GPU data, CPU solver -> results should come back on GPU
        settings = moreau.Settings(device="cpu", verbose=False)
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

        P_values = torch.tensor([[1.0, 1.0]], dtype=torch.float64, device="cuda")
        A_values = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float64, device="cuda")
        q = torch.tensor([[2.0, 1.0]], dtype=torch.float64, device="cuda")
        b = torch.tensor([[1.0, 2.0, 2.0]], dtype=torch.float64, device="cuda")

        solution = solver.solve(P_values, A_values, q, b)
        assert solution.x.device.type == "cuda"
        assert solution.z.device.type == "cuda"
        assert solution.s.device.type == "cuda"

    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
    def test_mixed_device_inputs_rejected(self):
        """Test that mismatched input devices raise ValueError."""
        n, m = 2, 3
        P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1], dtype=torch.int64)
        A_row_offsets = torch.tensor([0, 2, 3, 4], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_zero_cones = 1
        cones.num_nonneg_cones = 2

        settings = moreau.Settings(device="cuda", verbose=False)
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

        # P on CPU, A on CUDA -> should error
        with pytest.raises(ValueError, match="must be on the same device"):
            solver.solve(
                torch.tensor([[1.0, 1.0]], dtype=torch.float64, device="cpu"),
                torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float64, device="cuda"),
                torch.tensor([[2.0, 1.0]], dtype=torch.float64, device="cpu"),
                torch.tensor([[1.0, 2.0, 2.0]], dtype=torch.float64, device="cpu"),
            )

    def test_wrong_dtype_rejected(self, simple_qp_problem):
        """Test that wrong dtype is rejected."""
        p = simple_qp_problem
        device = p["device"]

        settings = moreau.Settings(device=device, verbose=False)

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

        # float32 should be rejected
        P_float32 = p["P_values"].float()
        with pytest.raises(RuntimeError, match="float64"):
            solver.solve(P_float32, p["A_values"], p["q"], p["b"])

    def test_wrong_size_rejected(self, simple_qp_problem):
        """Test that wrong size is rejected."""
        p = simple_qp_problem
        device = p["device"]

        method = "qdldl" if device == "cpu" else "cudss"
        ipm = moreau.IPMSettings(direct_solve_method=method)
        settings = moreau.Settings(device=device, verbose=False, ipm_settings=ipm)

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

        # Wrong size q (should have n=2 elements, not 3)
        q_wrong = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64, device=device)
        with pytest.raises(RuntimeError, match="dimension.*must be"):
            solver.solve(p["P_values"], p["A_values"], q_wrong, p["b"])

    @pytest.mark.skipif(HAS_CUDA, reason="Test requires no CUDA backend")
    def test_cuda_unavailable_raises_import_error(self):
        """When device='cuda' is requested but unavailable, raise ImportError, not
        the inscrutable AssertionError from torch internals on .to('cuda')."""
        P_ro = torch.tensor([0, 1, 2], dtype=torch.int64)
        P_ci = torch.tensor([0, 1], dtype=torch.int64)
        A_ro = torch.tensor([0, 2, 3, 4], dtype=torch.int64)
        A_ci = torch.tensor([0, 1, 0, 1], dtype=torch.int64)
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        with pytest.raises(ImportError, match="not available"):
            Solver(
                n=2,
                m=3,
                P_row_offsets=P_ro,
                P_col_indices=P_ci,
                A_row_offsets=A_ro,
                A_col_indices=A_ci,
                cones=cones,
                settings=moreau.Settings(device="cuda", solver="ipm"),
            )


class TestGarbageCollection:
    """Tests for proper garbage collection of solvers."""

    def test_solver_gc_after_backward(self, device):
        """Verify solver can be garbage collected after backward completes.

        This tests that the reference cycle (ctx -> solver -> _last_result -> tensors)
        is properly broken so the solver can be freed.
        """
        import gc
        import weakref

        settings = moreau.Settings(device=device, verbose=False)
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        # Create solver
        solver = Solver(
            n=2,
            m=3,
            P_row_offsets=torch.tensor([0, 1, 2], dtype=torch.int64),
            P_col_indices=torch.tensor([0, 1], dtype=torch.int64),
            A_row_offsets=torch.tensor([0, 2, 3, 4], dtype=torch.int64),
            A_col_indices=torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            cones=cones,
            settings=settings,
        )

        # Keep weak ref to track GC
        solver_ref = weakref.ref(solver)

        # Setup and solve with grad
        P_values = torch.tensor([[1.0, 1.0]], dtype=torch.float64, device=device)
        A_values = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float64, device=device)

        q = torch.tensor([[1.0, 1.0]], dtype=torch.float64, device=device, requires_grad=True)
        b = torch.tensor([[1.0, 0.7, 0.7]], dtype=torch.float64, device=device)

        result = solver.solve(P_values, A_values, q, b)
        loss = result.x.sum()
        loss.backward()

        # Delete solver and all references to results, force GC
        del result, loss, solver
        gc.collect()

        # Verify solver was collected (weak ref returns None)
        assert solver_ref() is None, "Solver should be garbage collected after backward"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
