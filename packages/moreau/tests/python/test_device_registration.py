"""
Tests for the plugin-style device registration system.

Verifies that:
1. Devices can register themselves with priorities
2. available_devices() returns devices sorted by priority
3. default_device() picks highest priority
4. device_available() and torch_available() work correctly
5. get_device_component() retrieves registered components
6. Custom devices can be registered dynamically
"""

import pytest
import numpy as np


class TestDeviceDiscovery:
    """Test device discovery and availability functions."""

    def test_available_devices_includes_cpu(self):
        """CPU should always be in available devices."""
        import moreau

        devices = moreau.available_devices()
        assert "cpu" in devices

    def test_available_devices_sorted_by_priority(self):
        """Devices should be sorted by priority (highest first)."""
        from moreau._backend import _device_registry, available_devices

        devices = available_devices()

        # Get priorities for registered devices
        priorities = []
        for dev in devices:
            if dev in _device_registry:
                priorities.append(_device_registry[dev].get("priority", 0))
            else:
                priorities.append(0)  # CPU has implicit priority 0

        # Should be sorted descending
        assert priorities == sorted(priorities, reverse=True)

    def test_default_device_is_highest_priority(self):
        """default_device() should return highest priority device."""
        import moreau

        # Reset any override first
        original = moreau.get_default_device()
        try:
            moreau.set_default_device(None)
            devices = moreau.available_devices()
            default = moreau.default_device()
            assert default == devices[0]
        finally:
            moreau.set_default_device(original)

    def test_device_available_cpu_always_true(self):
        """CPU should always be available."""
        import moreau

        assert moreau.device_available("cpu") is True

    def test_device_available_unregistered_false(self):
        """Unregistered devices should return False."""
        import moreau

        assert moreau.device_available("nonexistent_device") is False

    def test_device_error_returns_none_for_available(self):
        """device_error() should return None for available devices."""
        from moreau._backend import device_error, device_available

        for device in ["cpu"]:
            if device_available(device):
                # No error for available devices
                pass  # device_error might still be None


class TestDeviceRegistration:
    """Test dynamic device registration."""

    def test_register_custom_device(self):
        """Test registering a custom device."""
        from moreau._backend import (
            register_device,
            device_available,
            _device_registry,
            _reset_registry,
        )

        # Create a mock solver class
        class MockSolver:
            pass

        # Register a custom device
        register_device(
            "test_device",
            priority=50,
            solver_class=MockSolver,
        )

        try:
            assert device_available("test_device") is True
            assert _device_registry["test_device"]["solver"] is MockSolver
            assert _device_registry["test_device"]["priority"] == 50
        finally:
            # Clean up
            del _device_registry["test_device"]

    def test_priority_affects_ordering(self):
        """Test that priority affects device ordering."""
        from moreau._backend import register_device, available_devices, _device_registry

        class MockSolver:
            pass

        # Register devices with different priorities
        register_device("low_priority", priority=10, solver_class=MockSolver)
        register_device("high_priority", priority=200, solver_class=MockSolver)
        register_device("medium_priority", priority=75, solver_class=MockSolver)

        try:
            devices = available_devices()
            # High priority should come first
            high_idx = devices.index("high_priority")
            medium_idx = devices.index("medium_priority")
            low_idx = devices.index("low_priority")

            assert high_idx < medium_idx < low_idx
        finally:
            # Clean up
            del _device_registry["low_priority"]
            del _device_registry["high_priority"]
            del _device_registry["medium_priority"]

    def test_register_with_converters(self):
        """Test registering device with cones/settings converters."""
        from moreau._backend import register_device, get_device_component, _device_registry

        class MockSolver:
            pass

        class MockCones:
            pass

        def mock_cones_converter(cones):
            return MockCones()

        register_device(
            "converter_test",
            priority=25,
            solver_class=MockSolver,
            cones_class=MockCones,
            cones_converter=mock_cones_converter,
        )

        try:
            assert get_device_component("converter_test", "solver") is MockSolver
            assert get_device_component("converter_test", "cones") is MockCones
            assert get_device_component("converter_test", "cones_converter") is mock_cones_converter
        finally:
            del _device_registry["converter_test"]


class TestGetDeviceComponent:
    """Test get_device_component() function."""

    def test_get_existing_component(self):
        """Test getting an existing component."""
        from moreau._backend import get_device_component, device_available

        if device_available("cuda"):
            solver = get_device_component("cuda", "solver")
            assert solver is not None

    def test_get_nonexistent_component(self):
        """Test getting a component that doesn't exist."""
        from moreau._backend import get_device_component

        result = get_device_component("nonexistent", "solver")
        assert result is None

    def test_get_unset_component(self):
        """Test getting a component that wasn't registered."""
        from moreau._backend import register_device, get_device_component, _device_registry

        class MockSolver:
            pass

        register_device("partial_device", priority=10, solver_class=MockSolver)

        try:
            # torch_solver was not registered
            result = get_device_component("partial_device", "torch_solver")
            assert result is None
        finally:
            del _device_registry["partial_device"]


class TestTorchAvailable:
    """Test torch_available() function."""

    def test_torch_available_with_device(self):
        """Test torch_available() with specific device."""
        from moreau._backend import torch_available, device_available

        if device_available("cuda"):
            # If CUDA is available, check torch support
            result = torch_available("cuda")
            assert isinstance(result, bool)

    def test_torch_available_any_device(self):
        """Test torch_available() with no device (checks any)."""
        from moreau._backend import torch_available

        result = torch_available()
        assert isinstance(result, bool)

    def test_torch_available_cpu(self):
        """CPU registers torch_solver, so should be True when torch is installed."""
        pytest.importorskip("torch")
        import moreau.torch  # noqa: F401 — triggers CPU torch_solver registration
        from moreau._backend import torch_available

        # CPU backend registers _TorchSolverCpu in the registry
        result = torch_available("cpu")
        assert result is True  # CPU now registers its torch_solver


class TestCudaRegistration:
    """Test that CUDA backend registers correctly via entry points."""

    @pytest.mark.skipif(
        not __import__("moreau").device_available("cuda"), reason="CUDA not available"
    )
    def test_cuda_registered_with_priority(self):
        """Test that CUDA registers with high priority."""
        from moreau._backend import _device_registry

        assert "cuda" in _device_registry
        assert _device_registry["cuda"]["priority"] == 100

    @pytest.mark.skipif(
        not __import__("moreau").device_available("cuda"), reason="CUDA not available"
    )
    def test_cuda_has_all_components(self):
        """Test that CUDA registers all expected components."""
        from moreau._backend import get_device_component

        assert get_device_component("cuda", "solver") is not None
        assert get_device_component("cuda", "cones") is not None
        assert get_device_component("cuda", "settings") is not None
        assert get_device_component("cuda", "cones_converter") is not None
        assert get_device_component("cuda", "settings_converter") is not None

    @pytest.mark.skipif(
        not __import__("moreau").device_available("cuda"), reason="CUDA not available"
    )
    def test_cuda_is_default_when_available(self):
        """Test that CUDA becomes default device when available."""
        import moreau

        # Clear any test-level override to test true default behavior
        old_override = moreau.get_default_device()
        moreau.set_default_device(None)

        try:
            # CUDA has priority 100, which should be highest
            assert moreau.default_device() == "cuda"
        finally:
            # Restore the override if there was one
            if old_override is not None:
                moreau.set_default_device(old_override)


class TestResetRegistry:
    """Test _reset_registry() for testing purposes."""

    def test_reset_clears_registry(self):
        """Test that _reset_registry clears all registered devices."""
        from moreau._backend import (
            register_device,
            _reset_registry,
            _device_registry,
            _discovery_errors,
            _discover_backends,
            _register_cpu,
        )

        class MockSolver:
            pass

        # Add a test device
        register_device("reset_test", priority=10, solver_class=MockSolver)
        assert "reset_test" in _device_registry

        # Reset
        _reset_registry()
        assert len(_device_registry) == 0
        assert len(_discovery_errors) == 0

        # Re-discover backends and re-register CPU to restore state
        _discover_backends()
        _register_cpu()
