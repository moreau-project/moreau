"""Backend discovery and registration for moreau.

This module provides a generic device registration system. Backend packages
register themselves via Python entry points:

    [project.entry-points."moreau.backends"]
    cuda = "moreau_cuda:_register"

When moreau is imported, it discovers all registered backends automatically.
The 'cpu' backend is always available as a built-in fallback.
"""

import time
from importlib.metadata import entry_points
from typing import Any, Callable, Dict, List, Optional

# Registry for device backends
# Structure: {'cuda': {'solver': ..., 'priority': 100, ...}}
_device_registry: Dict[str, dict] = {}

# Store any errors that occurred during backend discovery
_discovery_errors: Dict[str, str] = {}

# Override for default device (None = use priority-based selection)
_default_device_override: Optional[str] = None


def _discover_backends() -> None:
    """Discover and register backends via entry points."""
    try:
        eps = entry_points(group="moreau.backends")
    except TypeError:
        # Python 3.9 compatibility: entry_points() returns a dict
        eps = entry_points().get("moreau.backends", [])

    for ep in eps:
        try:
            register_func = ep.load()
            register_func()
        except Exception as e:
            # Store error for debugging but don't fail
            _discovery_errors[ep.name] = f"{type(e).__name__}: {e}"


def register_device(
    name: str,
    *,
    priority: int = 50,
    solver_class: Any,
    batch_solver_class: Any = None,
    cones_class: Any = None,
    settings_class: Any = None,
    torch_solver_class: Any = None,
    torch_batch_solver_class: Any = None,
    torch_layer_class: Any = None,
    torch_solve_function_class: Any = None,
    jax_solver_class: Any = None,
    cones_converter: Optional[Callable] = None,
    settings_converter: Optional[Callable] = None,
) -> None:
    """Register a device backend with moreau.

    Called by backend packages to register their components.

    Args:
        name: Device name (e.g., 'cuda', 'rocm', 'tpu')
        priority: Selection priority for device='auto' (higher = preferred).
                  Suggested values: cuda=100, rocm=90, tpu=80
        solver_class: The single-problem solver class for this device
        batch_solver_class: The batch solver class for this device (optional)
        cones_class: The internal cones type for this backend (optional)
        settings_class: The internal settings type for this backend (optional)
        torch_solver_class: PyTorch single-problem solver class (optional)
        torch_batch_solver_class: PyTorch batch solver class (optional)
        torch_layer_class: PyTorch nn.Module layer class (optional)
        torch_solve_function_class: PyTorch autograd.Function class (optional)
        jax_solver_class: JAX solver class (optional)
        cones_converter: Function to convert Cones to backend type (optional)
        settings_converter: Function to convert Settings to backend type (optional)
    """
    _device_registry[name] = {
        "priority": priority,
        "solver": solver_class,
        "batch_solver": batch_solver_class,
        "cones": cones_class,
        "settings": settings_class,
        "torch_solver": torch_solver_class,
        "torch_batch_solver": torch_batch_solver_class,
        "torch_layer": torch_layer_class,
        "torch_solve_function": torch_solve_function_class,
        "jax_solver": jax_solver_class,
        "cones_converter": cones_converter,
        "settings_converter": settings_converter,
    }


def _reset_registry() -> None:
    """Reset the registry (for testing only)."""
    _device_registry.clear()
    _discovery_errors.clear()


# === Public API ===


def available_devices() -> List[str]:
    """Return list of available device names, sorted by priority (highest first).

    Returns:
        List of device names. Always includes 'cpu' as fallback.
    """
    devices = list(_device_registry.keys())
    # Sort by priority descending
    devices.sort(key=lambda d: _device_registry[d].get("priority", 0), reverse=True)
    # CPU is always available as fallback
    if "cpu" not in devices:
        devices.append("cpu")
    return devices


def device_available(name: str) -> bool:
    """Check if a device is available.

    Args:
        name: Device name (e.g., 'cuda', 'cpu')

    Returns:
        True if the device is registered and available.
    """
    if name == "cpu":
        return True
    return name in _device_registry


def device_error(name: str) -> Optional[str]:
    """Return the error message if device discovery failed.

    Args:
        name: Device name

    Returns:
        Error message string, or None if no error.
    """
    return _discovery_errors.get(name)


def default_device() -> str:
    """Return the default device.

    If set_default_device() was called, returns that device.
    Otherwise returns the highest priority available device.

    Returns:
        Device name string.
    """
    if _default_device_override is not None:
        return _default_device_override
    devices = available_devices()
    return devices[0] if devices else "cpu"


def _choose_device(n: int, nnz_A: int, batch_size: int = 1) -> str:
    """Choose optimal device based on problem characteristics.

    Uses empirical thresholds to determine whether CPU or CUDA will be faster.
    CUDA has ~25ms fixed overhead from cuDSS initialization, so it only wins
    for larger problems where GPU parallelism overcomes this cost.

    If set_default_device() was called, returns that device instead of
    using the heuristics.

    Args:
        n: Number of primal variables
        nnz_A: Number of non-zeros in constraint matrix A
        batch_size: Number of problems to solve in parallel (default 1)

    Returns:
        Device name: 'cuda' if CUDA is available and problem is large enough,
        otherwise 'cpu'.

    Thresholds (empirically determined):
        - n < 500: Always CPU (problem too small)
        - n >= 750: Always CUDA (problem large enough)
        - nnz_A >= 50,000: CUDA wins (enough linear algebra work)
        - batch_size >= 2 and nnz_A >= 25,000: CUDA wins (batching helps)
    """
    # Respect user override
    if _default_device_override is not None:
        return _default_device_override

    # Check if CUDA is available
    cuda_available = device_available("cuda")
    if not cuda_available:
        return "cpu"

    # Small problems: CPU wins due to CUDA overhead
    if n < 500:
        return "cpu"

    # Large problems: CUDA wins
    if n >= 750:
        return "cuda"

    # Medium problems: depends on nnz_A and batch_size
    if nnz_A >= 50_000:
        return "cuda"

    if batch_size >= 2 and nnz_A >= 25_000:
        return "cuda"

    return "cpu"


def set_default_device(device: Optional[str]) -> None:
    """Set the default device for all solvers.

    This overrides the automatic priority-based device selection.
    Useful for testing or forcing a specific backend.

    Args:
        device: Device name ('cpu', 'cuda', etc.) or None to reset
                to automatic selection.

    Raises:
        ValueError: If the specified device is not available.

    Example:
        >>> import moreau
        >>> moreau.set_default_device('cpu')  # Force CPU
        >>> # All solvers will now use CPU by default
        >>> moreau.set_default_device(None)   # Reset to auto
    """
    global _default_device_override
    if device is not None and not device_available(device):
        raise ValueError(
            f"Device '{device}' is not available. " f"Available devices: {available_devices()}"
        )
    _default_device_override = device


def get_default_device() -> Optional[str]:
    """Get the current default device override.

    Returns:
        The device set by set_default_device(), or None if using
        automatic selection.
    """
    return _default_device_override


def get_device_component(device: str, component: str) -> Any:
    """Get a component from a registered device.

    Args:
        device: Device name (e.g., 'cuda')
        component: Component name. One of:
            - 'solver': Single-problem solver class
            - 'batch_solver': Batch solver class (three-step API)
            - 'cones': Internal cones type
            - 'settings': Internal settings type
            - 'torch_solver': PyTorch single-problem solver class
            - 'torch_batch_solver': PyTorch batch solver class
            - 'torch_layer': PyTorch nn.Module class
            - 'torch_solve_function': PyTorch autograd.Function
            - 'jax_solver': JAX solver class
            - 'cones_converter': Function to convert Cones
            - 'settings_converter': Function to convert Settings

    Returns:
        The component, or None if not available.
    """
    return _device_registry.get(device, {}).get(component)


def torch_available(device: str = None) -> bool:
    """Check if PyTorch integration is available for a device.

    Args:
        device: Device name. If None, checks if any device has torch support.

    Returns:
        True if PyTorch solver is available for the device.
    """
    if device is not None:
        return get_device_component(device, "torch_solver") is not None

    # Check if any registered device has torch support
    for dev in _device_registry:
        if _device_registry[dev].get("torch_solver") is not None:
            return True
    return False


def jax_available(device: str = None) -> bool:
    """Check if JAX integration is available for a device.

    Args:
        device: Device name. If None, checks if any device has JAX support.

    Returns:
        True if JAX solver is available for the device.
    """
    if device is not None:
        return get_device_component(device, "jax_solver") is not None

    # Check if any registered device has JAX support
    for dev in _device_registry:
        if _device_registry[dev].get("jax_solver") is not None:
            return True
    return False


def _settings_to_cpu(settings):
    """Convert Settings to CPU DefaultSettings."""
    from moreau_cpu._cpu import settings_to_cpu

    return settings_to_cpu(settings)


def _register_cpu() -> None:
    """Register built-in CPU backend.

    CPU is always available as a fallback with lowest priority.
    Note: CPU uses moreau_cpu.Solver (three-step pattern) for both solver_class
    and batch_solver_class since it handles both single and batch based on input shape.
    """
    import moreau_cpu
    from ._types import Cones

    register_device(
        "cpu",
        priority=0,  # Lowest priority (fallback)
        solver_class=moreau_cpu.Solver,  # Three-step pattern solver
        batch_solver_class=moreau_cpu.Solver,  # Same class handles batch
        cones_class=Cones,
        settings_class=moreau_cpu.DefaultSettings,
        torch_solver_class=None,
        torch_batch_solver_class=None,
        jax_solver_class=None,
        cones_converter=None,  # Solver handles conversion internally
        settings_converter=_settings_to_cpu,
    )
    # Register torch/jax CPU solvers if they've been imported already.
    # On initial load, these modules haven't been imported yet (circular import),
    # so they register themselves in their own __init__.py. But if _register_cpu()
    # is called again (e.g. after _reset_registry()), we need to re-register.
    try:
        from .torch._cpu_impl import _TorchSolverCpu

        _device_registry["cpu"]["torch_solver"] = _TorchSolverCpu
    except ImportError:
        pass

    try:
        from .jax._cpu_impl import JaxSolverCpu

        _device_registry["cpu"]["jax_solver"] = JaxSolverCpu
    except ImportError:
        pass


def _cuda_hardware_available() -> bool:
    """Check if CUDA hardware is present (without requiring moreau-cuda)."""
    # Try torch first (most common)
    try:
        import torch

        if torch.cuda.is_available():
            return True
    except ImportError:
        pass

    # Try cupy
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return True
    except (ImportError, Exception):
        pass

    # Try numba
    try:
        from numba import cuda

        if cuda.is_available():
            return True
    except (ImportError, Exception):
        pass

    return False


def _show_cuda_hint() -> None:
    """Show a hint if CUDA hardware is available but moreau-cuda isn't installed."""
    # Only show if:
    # 1. CUDA hardware is detected
    # 2. moreau-cuda is NOT registered (not installed)
    # 3. moreau-cuda is not currently being imported (circular import case)
    import sys

    if "moreau_cuda" in sys.modules:
        # moreau_cuda is already imported or being imported, skip hint
        return
    if "cuda" not in _device_registry and _cuda_hardware_available():
        import warnings

        warnings.warn(
            "CUDA GPU detected. Install moreau[cuda] for 10-100x faster solving: "
            "pip install moreau[cuda]",
            stacklevel=2,
        )


def solver_methods_for_device(device: str) -> List[str]:
    """Return candidate KKT solver methods for a given device.

    Args:
        device: Device name ('cuda', 'cpu', etc.)

    Returns:
        List of method name strings to benchmark during tune().
    """
    if device == "cuda":
        # Riccati is available via direct_solve_method='riccati' but not
        # benchmarked by auto-tune (it only works on block-tridiagonal
        # problems and Auto mode already selects it when appropriate).
        return ["cudss"]
    elif device == "cpu":
        return ["qdldl", "faer"]
    else:
        return ["qdldl"]


def _rank_method(device: str, n: int, m: int, nnz_A: int, batch_size: int) -> List[str]:
    """Return methods for a device, ordered by heuristic likelihood of winning.

    For CPU, faer wins on larger systems; QDLDL wins on small ones.

    Args:
        device: 'cuda', 'cpu', etc.
        n: Number of variables.
        m: Number of constraints.
        nnz_A: Number of non-zeros in A.
        batch_size: Batch size.
    """
    kkt_dim = n + m
    methods = solver_methods_for_device(device)
    if device == "cuda":
        return methods
    elif device == "cpu":
        # faer wins for larger systems; QDLDL for small
        if kkt_dim >= 500 or nnz_A >= 10_000:
            return [m for m in methods if m == "faer"] + [m for m in methods if m != "faer"]
        return [m for m in methods if m == "qdldl"] + [m for m in methods if m != "qdldl"]
    return methods


def auto_tune_candidates(
    device: str,
    method: str,
    n: int = 0,
    m: int = 0,
    nnz_P: int = 0,
    nnz_A: int = 0,
    batch_size: int = 1,
) -> List[tuple]:
    """Return (device, method) combos to benchmark, heuristic-best first.

    The first candidate runs without a time limit and sets the baseline.
    Subsequent candidates are time-limited against that baseline, so
    ordering matters for auto-tune speed.

    If device='auto', returns all available devices × their methods.
    If only method='auto', returns all methods for the explicit device.

    Args:
        device: Device setting ('auto' or explicit like 'cpu', 'cuda')
        method: direct_solve_method setting ('auto' or explicit)
        n: Number of variables (for heuristic ordering).
        m: Number of constraints (for heuristic ordering).
        nnz_P: Non-zeros in P (for heuristic ordering).
        nnz_A: Non-zeros in A (for heuristic ordering).
        batch_size: Batch size (for heuristic ordering).

    Returns:
        List of (device, method) tuples to benchmark, best guess first.
    """
    if device == "auto":
        # Order devices: heuristic-best first
        best_dev = _choose_device(n, nnz_A, batch_size)
        devices = available_devices()
        ordered_devs = [best_dev] + [d for d in devices if d != best_dev]
        combos = []
        for dev in ordered_devs:
            for meth in _rank_method(dev, n, m, nnz_A, batch_size):
                combos.append((dev, meth))
        return combos
    elif method == "auto":
        # Benchmark all methods on the explicit device, heuristic-best first
        return [(device, meth) for meth in _rank_method(device, n, m, nnz_A, batch_size)]
    else:
        # Both explicit — nothing to benchmark
        return []


def benchmark_candidates(
    candidates, build_trial_settings, build_trial_solver, solve_trial, time_limit, verbose=False
):
    """Benchmark auto-tune candidates and return (results, best_time, best_key).

    Shared by CompiledSolver, torch.Solver and jax.Solver. Iterates
    `candidates` (heuristic-best first), builds + times each trial solver,
    skips candidates whose construction blows the budget, then solves and
    records timing. The first candidate runs without a time limit and sets
    the baseline against which later challengers are limited.

    Args:
        candidates: List of (device, method) tuples to benchmark.
        build_trial_settings: fn(device, method, best_time) -> trial Settings.
        build_trial_solver: fn(trial_settings) -> trial solver (constructed
            and set up; this call is the timed construction phase).
        solve_trial: fn(trial_solver) -> info object with .solve_time,
            .iterations, .status.
        time_limit: Caller's configured time limit (passed to
            build_trial_settings for clamping).
        verbose: If True, print per-candidate progress.

    Returns:
        (results, best_time, best_key) where results maps 'device:method'
        keys to dicts with 'solve_time', 'iterations', 'status'.
    """
    from ._types import SolverStatus

    results = {}
    best_time = float("inf")
    best_key = None
    construction_limit = None

    if verbose:

        def _print(*a, **kw):
            print(*a, **kw, flush=True)

        _print(f"[moreau] auto-tune: benchmarking {len(candidates)} " f"candidate(s)...")
    else:
        _print = None

    for trial_device, trial_method in candidates:
        key = f"{trial_device}:{trial_method}"

        trial_settings = build_trial_settings(trial_device, trial_method, best_time)

        build_t0 = time.perf_counter()
        trial_solver = build_trial_solver(trial_settings)
        build_time = time.perf_counter() - build_t0

        # Set construction budget after first candidate
        if construction_limit is None:
            construction_limit = max(build_time * 5, 1.0)

        # Skip if construction took too long
        if build_time > construction_limit:
            results[key] = {
                "solve_time": float("inf"),
                "iterations": 0,
                "status": [SolverStatus.NumericalError],
            }
            if _print:
                _print(
                    f"[moreau] auto-tune:   {key:<16s} "
                    f"{build_time:.3f}s build  "
                    f"skipped (build > {construction_limit:.2f}s limit)"
                )
            continue

        info = solve_trial(trial_solver)
        results[key] = {
            "solve_time": info.solve_time,
            "iterations": info.iterations,
            "status": info.status,
        }

        _ok = (SolverStatus.Solved, SolverStatus.AlmostSolved)
        statuses = info.status if isinstance(info.status, list) else [info.status]
        is_ok = all(s in _ok for s in statuses)
        if is_ok and info.solve_time < best_time:
            best_time = info.solve_time
            best_key = key

        if _print:
            iters = info.iterations
            iters_str = str(max(iters)) if isinstance(iters, list) else str(iters)
            names = set(s.name if hasattr(s, "name") else str(s) for s in statuses)
            status_str = names.pop() if len(names) == 1 else "/".join(sorted(names))
            marker = " *" if key == best_key else ""
            _print(
                f"[moreau] auto-tune:   {key:<16s} "
                f"{build_time:.3f}s build  "
                f"{info.solve_time:.4f}s solve  "
                f"{iters_str} iters  "
                f"{status_str}{marker}"
            )

    return results, best_time, best_key


def needs_auto_tune(settings, already_tuned, original_device, original_method) -> bool:
    """Shared decision: should the frontend run auto-tune on this solve?

    Mirrored by torch.Solver and jax.Solver — they only differ in the
    impl they construct after the winner is picked.
    """
    if already_tuned:
        return False
    if not settings.auto_tune:
        return False
    from ._types import SolverType

    if settings.solver == SolverType.ACTIVE_SET:
        return False
    if original_device == "auto" and _default_device_override is not None:
        return False
    if original_device == "auto":
        return True
    if original_method == "auto":
        return True
    return False


def benchmark_and_select_winner(
    *,
    settings,
    cones,
    b_sparsity_pattern,
    n: int,
    m: int,
    P_ro,
    P_ci,
    A_ro,
    A_ci,
    P_np,
    A_np,
    q_np,
    b_np,
    original_device: str,
    original_method: str,
    current_device: str,
    component_key: str,
    component_cpu_fallback: bool,
):
    """Run the full auto-tune flow shared by torch + jax Solvers.

    1. Build candidates via auto_tune_candidates().
    2. Benchmark each via benchmark_candidates() (trials are plain
       moreau.CompiledSolver — no grad overhead).
    3. Pick the fastest winner that (a) succeeded and (b) has a
       backend component for `component_key` registered.
    4. Build a TuneResult.

    Returns:
        None if there are no candidates to benchmark (caller should
        just flag itself tuned).
        Otherwise (best_device, best_method, time_limit, results,
        tune_result) — caller is responsible for rebuilding its
        backend impl if (best_device, best_method) differs from the
        current config.
    """
    # Local imports avoid an import cycle at module load: _backend is
    # imported by _types, which is imported by moreau.__init__, which
    # then imports moreau.CompiledSolver back here.
    import moreau
    from ._types import SolverStatus, IPMSettings, TuneResult, _AUTO_TUNE_MARGIN

    batch_size = q_np.shape[0] if q_np.ndim == 2 else 1

    candidates = auto_tune_candidates(
        original_device,
        original_method,
        n=n,
        m=m,
        nnz_P=len(P_ci),
        nnz_A=len(A_ci),
        batch_size=batch_size,
    )
    if not candidates:
        return None

    verbose = settings.verbose

    def _build_trial_settings(trial_device, trial_method, best_time):
        trial = settings.model_copy(deep=True)
        trial.device = trial_device
        trial.verbose = False
        trial.enable_grad = False  # trial CompiledSolvers never need grad
        trial.batch_size = batch_size
        if trial.ipm_settings is None:
            trial.ipm_settings = IPMSettings()
        trial.ipm_settings.direct_solve_method = trial_method
        if best_time < float("inf"):
            trial.time_limit = min(best_time * _AUTO_TUNE_MARGIN, trial.time_limit)
        return trial

    def _build_trial_solver(trial_settings):
        trial = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=trial_settings,
            b_sparsity_pattern=b_sparsity_pattern,
        )
        trial.setup(P_np, A_np)
        return trial

    def _solve_trial(trial):
        trial.solve(q_np, b_np)
        return trial.info

    results, _, _ = benchmark_candidates(
        candidates,
        _build_trial_settings,
        _build_trial_solver,
        _solve_trial,
        settings.time_limit,
        verbose=verbose,
    )

    _ok = (SolverStatus.Solved, SolverStatus.AlmostSolved)
    sorted_keys = sorted(results, key=lambda k: results[k]["solve_time"])
    winner_key = None
    for key in sorted_keys:
        r_status = results[key]["status"]
        r_statuses = r_status if isinstance(r_status, list) else [r_status]
        if not all(s in _ok for s in r_statuses):
            continue
        dev = key.split(":", 1)[0]
        comp = get_device_component(dev, component_key)
        if comp is None and component_cpu_fallback and dev != "cpu":
            comp = get_device_component("cpu", component_key)
        if comp is not None:
            winner_key = key
            break

    if winner_key is None:
        winner_key = sorted_keys[0]
        best_device = current_device
        best_method = winner_key.split(":", 1)[1]
    else:
        best_device, best_method = winner_key.split(":", 1)

    best_time = results[winner_key]["solve_time"]
    tuned_limit = best_time * _AUTO_TUNE_MARGIN
    time_limit = max(tuned_limit, settings.time_limit)

    if verbose:
        print(
            f"[moreau] auto-tune: winner {best_device}:{best_method} " f"({best_time:.4f}s solve)",
            flush=True,
        )

    tune_result = TuneResult(
        device=best_device,
        method=best_method,
        time_limit=time_limit,
        results=results,
    )
    return best_device, best_method, time_limit, results, tune_result


# Public API (re-exported in moreau.__init__)
__all__ = [
    "available_devices",
    "device_available",
    "device_error",
    "default_device",
    "set_default_device",
    "get_default_device",
    "torch_available",
    "jax_available",
]

# Internal API (not re-exported, used by moreau internals and backend packages)
# - register_device: used by backend packages (moreau_cuda) to register devices
# - get_device_component: used by Solver/CompiledSolver and torch/jax modules


# Discover backends on import.
# Must run AFTER all helpers used by backend modules (auto_tune_candidates,
# benchmark_candidates, etc.) are defined — moreau_cuda.jax imports them at
# init time, and a partially-loaded _backend would surface as a circular
# ImportError that silently disables the CUDA JAX backend.
_discover_backends()

# Register CPU as built-in fallback
_register_cpu()

# Show hint about CUDA if hardware available but package not installed
_show_cuda_hint()
