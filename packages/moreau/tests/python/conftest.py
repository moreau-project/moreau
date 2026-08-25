"""
Pytest configuration for moreau unified package tests.

Usage:
    pytest packages/moreau/tests/python/ -v                  # Auto: runs on ALL available devices
    pytest packages/moreau/tests/python/ -v --device=cpu     # CPU only
    pytest packages/moreau/tests/python/ -v --device=cuda    # CUDA only
"""

import moreau
import pytest


def pytest_addoption(parser):
    """Add command-line options for device selection."""
    parser.addoption(
        "--device",
        action="store",
        default="auto",
        help="Device to use for tests: cpu, cuda, or auto (default: auto = all available)",
    )
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.slow (default: skip).",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA backend")
    config.addinivalue_line("markers", "cpu_only: mark test as CPU-only")
    config.addinivalue_line("markers", "torch: mark test as requiring PyTorch integration")
    config.addinivalue_line("markers", "jax: mark test as requiring JAX integration")
    config.addinivalue_line(
        "markers",
        "flaky: mark test as flaky when numerical noise can trigger intermittent failures",
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (skipped by default; pass --runslow to enable)"
    )


def pytest_generate_tests(metafunc):
    """Parametrize tests with device fixture across all available devices."""
    if "device" in metafunc.fixturenames:
        device_opt = metafunc.config.getoption("--device")

        if device_opt == "auto":
            # Run on all available devices
            devices = moreau.available_devices()
        else:
            # Run on specified device only
            devices = [device_opt]

        metafunc.parametrize("device", devices, indirect=True)


@pytest.fixture
def device(request):
    """Fixture that sets the default device and returns it."""
    device_name = request.param

    # Skip cpu_only tests when running on cuda device
    if device_name == "cuda" and request.node.get_closest_marker("cpu_only"):
        pytest.skip("CPU-only test skipped for CUDA device")

    moreau.set_default_device(device_name)
    yield device_name
    moreau.set_default_device(None)  # Reset after test


def pytest_collection_modifyitems(config, items):
    """Skip tests based on device availability and markers."""
    device_opt = config.getoption("--device")
    runslow = config.getoption("--runslow")

    for item in items:
        # Skip @slow tests unless --runslow is passed (fuzz / large parametrize).
        if item.get_closest_marker("slow") and not runslow:
            item.add_marker(pytest.mark.skip(reason="slow test skipped (pass --runslow to enable)"))

        # Skip CUDA tests if CUDA not available
        if item.get_closest_marker("cuda"):
            if not moreau.device_available("cuda"):
                item.add_marker(pytest.mark.skip(reason="CUDA not available"))

        # When testing specific device, skip tests for other devices
        if device_opt == "cpu":
            if item.get_closest_marker("cuda"):
                item.add_marker(pytest.mark.skip(reason="Skipping CUDA test (--device=cpu)"))
        elif device_opt == "cuda":
            if item.get_closest_marker("cpu_only"):
                item.add_marker(pytest.mark.skip(reason="Skipping CPU-only test (--device=cuda)"))


@pytest.fixture(autouse=True)
def skip_torch_tests_if_unavailable(request):
    """Auto-skip tests marked with @pytest.mark.torch if torch is not available."""
    if request.node.get_closest_marker("torch"):
        if not moreau.torch_available():
            pytest.skip("PyTorch integration not available")


@pytest.fixture(autouse=True)
def skip_jax_tests_if_unavailable(request):
    """Auto-skip tests marked with @pytest.mark.jax if jax is not available."""
    if request.node.get_closest_marker("jax"):
        if not moreau.jax_available():
            pytest.skip("JAX integration not available")


@pytest.fixture(autouse=True)
def force_device_selection(request):
    """Force device selection based on --device flag for ALL tests."""
    device_opt = request.config.getoption("--device")

    # If specific device requested, force it for all tests
    if device_opt != "auto":
        moreau.set_default_device(device_opt)
        yield
        moreau.set_default_device(None)
    else:
        yield
