# Import moreau first to avoid double-registration of types
# when pytest tries to import python/__init__.py as a module
import moreau
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "cuda: mark test as requiring CUDA backend")
    config.addinivalue_line("markers", "torch: mark test as requiring PyTorch integration")
    # Note: device forcing is done per-test via force_cpu_device fixture


def pytest_collection_modifyitems(config, items):
    """Skip CUDA/GPU tests - this is the CPU package.

    Only applies to tests within packages/moreau-cpu/.
    """
    skip_cuda = pytest.mark.skip(reason="Skipping CUDA/GPU test (moreau-cpu tests)")
    for item in items:
        # Only apply skip rules to tests in moreau-cpu package
        if "moreau-cpu" not in item.nodeid:
            continue

        # Skip tests with [cuda] parameter
        if "[cuda]" in item.nodeid:
            item.add_marker(skip_cuda)
        # Skip tests with 'cuda' or 'gpu' in the name (case insensitive)
        elif "cuda" in item.name.lower() or "gpu" in item.name.lower():
            item.add_marker(skip_cuda)
        # Skip backend consistency tests (they compare CPU vs CUDA)
        elif "consistency" in item.nodeid.lower():
            item.add_marker(skip_cuda)
        # Skip device priority tests (we override default device)
        elif "priority" in item.name.lower():
            item.add_marker(skip_cuda)
        # Skip torch input validation tests (test CUDA-specific behavior)
        elif "TorchInputValidation" in item.nodeid:
            item.add_marker(skip_cuda)
        # Skip tests marked with @pytest.mark.cuda
        elif item.get_closest_marker("cuda"):
            item.add_marker(skip_cuda)


@pytest.fixture(autouse=True)
def force_cpu_device(request):
    """Force CPU device for tests in moreau-cpu package only.

    This ensures CPU is used even when running alongside other test packages
    that may reset the default device.
    """
    # Only apply to tests within moreau-cpu package
    if "moreau-cpu" in request.node.nodeid:
        moreau.set_default_device("cpu")
        yield
        moreau.set_default_device(None)  # Reset after test
    else:
        yield


@pytest.fixture(autouse=True)
def skip_torch_tests_if_unavailable(request):
    """Auto-skip tests marked with @pytest.mark.torch if torch integration is not available."""
    marker = request.node.get_closest_marker("torch")
    if marker:
        if not moreau.torch_available():
            pytest.skip("PyTorch integration not available")
