"""Tests for device selection in the unified moreau package.

Run with different devices:
    pytest packages/moreau/tests/python/ -v --device=cpu
    pytest packages/moreau/tests/python/ -v --device=cuda
    pytest packages/moreau/tests/python/ -v --device=auto
"""

import numpy as np
import pytest
import moreau


class TestDeviceAPI:
    """Test device API functions."""

    def test_available_devices(self):
        """Test available_devices() returns a list."""
        devices = moreau.available_devices()
        assert isinstance(devices, list)
        assert "cpu" in devices

    def test_device_available_cpu(self):
        """CPU should always be available."""
        assert moreau.device_available("cpu") is True

    def test_device_available_nonexistent(self):
        """Nonexistent devices should return False."""
        assert moreau.device_available("nonexistent") is False

    def test_default_device(self):
        """default_device() should return a string."""
        device = moreau.default_device()
        assert isinstance(device, str)
        assert device in moreau.available_devices()

    def test_set_default_device(self):
        """Test set_default_device() works."""
        original = moreau.get_default_device()
        try:
            moreau.set_default_device("cpu")
            assert moreau.default_device() == "cpu"
            assert moreau.get_default_device() == "cpu"
        finally:
            moreau.set_default_device(original)

    def test_set_default_device_invalid(self):
        """Test set_default_device() raises for invalid device."""
        with pytest.raises(ValueError):
            moreau.set_default_device("nonexistent")


class TestTorchSolverWithDevice:
    """Test PyTorch Solver respects device selection."""

    @pytest.fixture
    def torch_available(self):
        """Check if torch is available."""
        torch = pytest.importorskip("torch")
        return torch

    @pytest.fixture
    def simple_qp_torch(self, torch_available):
        """Simple QP problem data as torch tensors."""
        torch = torch_available
        n, m = 2, 1

        P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1], dtype=torch.int64)

        A_row_offsets = torch.tensor([0, 2], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_zero_cones = 1

        return {
            "n": n,
            "m": m,
            "P_row_offsets": P_row_offsets,
            "P_col_indices": P_col_indices,
            "A_row_offsets": A_row_offsets,
            "A_col_indices": A_col_indices,
            "cones": cones,
        }

    def test_torch_solver_uses_default_device(self, torch_available, simple_qp_torch, device):
        """Test torch solver uses the default device."""
        torch = torch_available
        from moreau.torch import Solver as TorchSolver

        p = simple_qp_torch
        settings = moreau.Settings()

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

        # Create tensors on the appropriate device
        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device)
        b = torch.tensor([1.0], dtype=torch.float64, device=device)

        result = solver.solve(P_values, A_values, q, b)
        info = solver.info

        # Check solution
        x_sum = result.x.sum().item()
        np.testing.assert_allclose(x_sum, 1.0, rtol=1e-4)

    def test_torch_solver_batched(self, torch_available, simple_qp_torch, device):
        """Test torch solver with batched inputs."""
        torch = torch_available
        from moreau.torch import Solver as TorchSolver

        p = simple_qp_torch
        batch_size = 4
        settings = moreau.Settings()

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

        # Create batched tensors
        P_values = (
            torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        A_values = (
            torch.tensor([1.0, 1.0], dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        q = (
            torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        b = (
            torch.tensor([1.0], dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        result = solver.solve(
            P_values.contiguous(), A_values.contiguous(), q.contiguous(), b.contiguous()
        )

        assert result.x.shape == (batch_size, p["n"])
