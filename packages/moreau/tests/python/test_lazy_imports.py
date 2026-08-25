"""
Tests for import behavior and device availability in moreau.

Verifies that:
1. Entry point discovery works correctly
2. Requesting device='cuda' gives clear errors when unavailable
3. device_available() and torch_available() work correctly
"""

import sys
import pytest


class TestLazyImports:
    """Test moreau import behavior."""

    def test_entry_point_discovery(self):
        """Verify that moreau_cuda registers via entry points when installed.

        With the entry point architecture, moreau_cuda is discovered and loaded
        when moreau is imported (if installed). This is the expected behavior.
        """
        # After importing moreau, if moreau_cuda is installed, it should be registered
        import moreau

        # Check that CUDA backend is registered via entry points
        from moreau._backend import _device_registry

        # If moreau_cuda is installed, it should have registered
        if "cuda" in _device_registry:
            assert _device_registry["cuda"]["solver"] is not None
            # The solver class should be from moreau_cuda
            assert "moreau_cuda" in str(_device_registry["cuda"]["solver"])

    def test_device_available_returns_bool(self):
        """Test that device_available() returns a boolean."""
        import moreau

        result = moreau.device_available("cuda")
        assert isinstance(result, bool)

        result = moreau.device_available("cpu")
        assert result is True  # CPU always available

    def test_cpu_device_explicit(self):
        """Test that device='cpu' works without CUDA."""
        pytest.importorskip("torch")
        import torch
        import moreau
        from moreau.torch import Solver

        # Setup minimal problem
        n, m = 2, 2
        P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1], dtype=torch.int64)
        A_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_nonneg_cones = 2

        # Create CPU solver explicitly
        settings = moreau.Settings(device="cpu")
        solver = Solver(
            n,
            m,
            P_row_offsets,
            P_col_indices,
            A_row_offsets,
            A_col_indices,
            cones,
            settings=settings,
        )

        assert solver.device == "cpu"

        # Should be able to solve
        P_values = torch.ones(2, dtype=torch.float64)
        A_values = torch.ones(2, dtype=torch.float64)
        q = torch.ones(2, dtype=torch.float64)
        b = torch.ones(2, dtype=torch.float64)

        result = solver.solve(P_values, A_values, q, b)
        info = solver.info
        x = result.x
        assert x.shape == (2,)

    def test_cuda_device_error_when_unavailable(self):
        """Test that device='cuda' gives clear error when CUDA not available."""
        pytest.importorskip("torch")
        # Skip if CUDA is actually available - we can't test the error case
        import moreau

        if moreau.device_available("cuda") and moreau.torch_available("cuda"):
            pytest.skip("CUDA is available, can't test error case")

        import torch
        from moreau.torch import Solver

        # Setup minimal problem
        n, m = 2, 2
        P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        P_col_indices = torch.tensor([0, 1], dtype=torch.int64)
        A_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
        A_col_indices = torch.tensor([0, 1], dtype=torch.int64)

        cones = moreau.Cones()
        cones.num_nonneg_cones = 2

        # Requesting CUDA should raise ImportError with helpful message
        settings = moreau.Settings(device="cuda")
        with pytest.raises(ImportError, match="PyTorch extension"):
            solver = Solver(
                n,
                m,
                P_row_offsets,
                P_col_indices,
                A_row_offsets,
                A_col_indices,
                cones,
                settings=settings,
            )


class TestTorchAvailableFunction:
    """Test torch_available() function behavior."""

    def test_torch_available_uses_registry(self):
        """Test that torch_available() uses the device registry."""
        import moreau
        import moreau._backend as backend

        # Result should be consistent
        result1 = moreau.torch_available("cuda")
        result2 = moreau.torch_available("cuda")

        assert result1 == result2

        # The result should match registry state
        if result1:
            assert "cuda" in backend._device_registry
            assert backend._device_registry["cuda"].get("torch_solver") is not None

    def test_torch_available_any_device(self):
        """Test that torch_available() with no argument checks any device."""
        import moreau

        result = moreau.torch_available()
        assert isinstance(result, bool)

    def test_torch_available_consistency(self):
        """Test that device_available and torch_available are consistent."""
        import moreau

        # If cuda device is not available, torch can't be available for it
        if not moreau.device_available("cuda"):
            assert moreau.torch_available("cuda") is False
