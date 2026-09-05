"""
Moreau JAX integration with unified CPU/CUDA support.

Provides differentiable convex optimization for JAX with automatic
device selection between CPU and CUDA backends.

Example (full signature):
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from moreau.jax import Solver
    >>> import moreau
    >>>
    >>> cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
    >>> solver = Solver(n=2, m=3, P_row_offsets=..., P_col_indices=...,
    ...                 A_row_offsets=..., A_col_indices=..., cones=cones)
    >>>
    >>> solution = solver.solve(P_values, A_values, q, b)
    >>> print(solution.x, solver.info.status)

Example (two-step API):
    >>> solver = Solver(n=2, m=3, ...)
    >>> solver.setup(P_values, A_values)  # Set matrices once
    >>> solution = solver.solve(q, b)  # Solve with 2 args
    >>>
    >>> # User can vmap/jit as needed
    >>> batched = jax.vmap(solver.solve)
    >>> solutions = batched(q_batch, b_batch)

Example (differentiation):
    >>> def loss_fn(q):
    ...     return jnp.sum(solver.solve(q, b).x)
    >>> grad_q = jax.grad(loss_fn)(q)
"""

from typing import Optional, Sequence
import time
import warnings

from moreau._backend import (
    default_device,
    get_device_component,
    jax_available,
)
from moreau._types import (
    Cones,
    Settings,
    IPMSettings,
    WarmStart,
    BatchedWarmStart,
)
from ._types import JaxSolution, JaxSolveInfo
from moreau._backend import (
    benchmark_and_select_winner as _benchmark_and_select_winner,
    needs_auto_tune as _needs_auto_tune_shared,
)

# Register CPU JAX solver now that moreau._backend has finished loading.
# This import was impossible inside _backend._register_cpu() because
# _backend.py loads during moreau/__init__.py, creating a circular import.
from moreau._backend import _device_registry as _dr

if "cpu" in _dr and _dr["cpu"].get("jax_solver") is None:
    from ._cpu_impl import JaxSolverCpu

    _dr["cpu"]["jax_solver"] = JaxSolverCpu
del _dr


class Solver:
    """JAX solver with automatic device selection and gradient support.

    Provides a solve method compatible with jax.vmap, jax.grad, and jax.jit.
    Solve metadata (timing, status) is available via solver.info after solve().

    Supports two usage patterns:
    1. Full signature: solve(P_values, A_values, q, b)
    2. Two-step: setup(P_values, A_values) then solve(q, b)

    Args:
        n: Number of primal variables
        m: Number of constraints
        P_row_offsets: CSR row pointers for P matrix (array-like).
            P must be full symmetric (both upper and lower triangles).
        P_col_indices: CSR column indices for P matrix (array-like)
        A_row_offsets: CSR row pointers for A matrix (array-like)
        A_col_indices: CSR column indices for A matrix (array-like)
        cones: Cone specification (moreau.Cones object)
        settings: Optional solver settings (moreau.Settings object).
                  If device='auto' (default), uses CUDA if available, else CPU.
                  Note: jax.Solver always enables gradients (enable_grad=True)
                  internally regardless of this setting.
        jit: If True (default), JIT-compile the solve method for better performance.
        b_sparsity_pattern: Optional boolean mask of length m indicating which
                  entries of b are structurally nonzero. Enables more aggressive
                  chordal decomposition for PSD cones.

    Example (full signature):
        >>> solver = Solver(n=2, m=3, P_row_offsets=..., P_col_indices=...,
        ...                 A_row_offsets=..., A_col_indices=..., cones=cones)
        >>> solution = solver.solve(P_values, A_values, q, b)

    Example (two-step):
        >>> solver = Solver(n=2, m=3, ...)
        >>> solver.setup(P_values, A_values)  # Set matrices once
        >>> solution = solver.solve(q, b)  # Solve with 2 args
        >>>
        >>> # User can vmap/jit as needed
        >>> batched = jax.vmap(solver.solve)
        >>> solutions = batched(q_batch, b_batch)
    """

    def __init__(
        self,
        n: int,
        m: int,
        P_row_offsets,
        P_col_indices,
        A_row_offsets,
        A_col_indices,
        cones: Cones,
        settings: Optional[Settings] = None,
        jit: bool = True,
        b_sparsity_pattern: Optional[Sequence[bool]] = None,
    ):
        # Check for old CVXPY with SOC cones
        from moreau import _warn_cvxpy_soc_if_needed, _require_dir_cones_compatible

        _warn_cvxpy_soc_if_needed(cones)
        _require_dir_cones_compatible(cones, settings)

        # Direct cones thread through both CPU and CUDA JAX. CPU JAX uses
        # `pure_callback` to dispatch to the numpy CPU solver. CUDA JAX
        # extends the FFI with x_cone descriptors (kinds, indices,
        # alphas/dim2 for asymm), a `z_x` output, and a `dz_x` backward
        # input — see `jax_ffi.cu` MoreauSolveFwd / MoreauSolveBwd handler
        # signatures.

        if settings is None:
            settings = Settings()

        # Resolve device FIRST so active-set auto-select sees actual target
        device = settings.device
        if device == "auto":
            device = default_device()

        # Now resolve solver='auto' with the actual device known
        from moreau import _resolve_solver_type

        nnz_P = len(P_col_indices)
        settings, device = _resolve_solver_type(settings, n, m, cones, device, nnz_P=nnz_P)

        # Resolve the JAX class for the requested device. If unavailable, fall
        # back to CPU and update `device` so downstream consumers (self._device,
        # the direct_solve_method heuristic at the bottom of __init__, and the
        # settings copy below) see the actual backend, not the requested one.
        # Without this, e.g. device='cuda' with no CUDA backend kept device-
        # specific direct_solve_method='cudss' written into CPU settings, which
        # the CPU pybind layer rejects at solve time (#180).
        JaxSolverClass = get_device_component(device, "jax_solver")
        if JaxSolverClass is None and device != "cpu":
            cpu_class = get_device_component("cpu", "jax_solver")
            if cpu_class is not None:
                warnings.warn(
                    f"JAX extension for device='{device}' not available; "
                    f"falling back to CPU. Pass device='cpu' explicitly to silence this warning.",
                    stacklevel=2,
                )
                device = "cpu"
                # Clear cuda-only direct_solve_method (e.g. 'cudss') so the
                # CPU backend doesn't reject it.
                if settings.ipm_settings is not None:
                    method = (settings.ipm_settings.direct_solve_method or "").lower()
                    if method in ("cudss",):
                        settings = settings.model_copy(deep=True)
                        settings.ipm_settings.direct_solve_method = "auto"
                JaxSolverClass = cpu_class

        if JaxSolverClass is None:
            raise ImportError(
                f"JAX extension for device '{device}' not available. "
                f"Ensure JAX is installed and the appropriate moreau backend is built with JAX support. "
                f"Available devices with JAX support: {[d for d in ['cpu', 'cuda'] if jax_available(d)]}"
            )

        self._device = device
        self._n = n
        self._m = m
        self._settings = settings
        self._cones = cones
        self._P_row_offsets = P_row_offsets
        self._P_col_indices = P_col_indices
        self._A_row_offsets = A_row_offsets
        self._A_col_indices = A_col_indices
        self._jit = jit
        self._b_sparsity_pattern = (
            list(b_sparsity_pattern) if b_sparsity_pattern is not None else None
        )

        # Track construction time
        construction_start = time.perf_counter()

        # Create device-specific solver
        self._impl = JaxSolverClass(
            n,
            m,
            P_row_offsets,
            P_col_indices,
            A_row_offsets,
            A_col_indices,
            cones,
            settings,
            b_sparsity_pattern=self._b_sparsity_pattern,
        )

        self._construction_time = time.perf_counter() - construction_start

        # Get the raw solve function from impl
        self._solve_raw = self._impl.solve

        # JIT-compile for performance
        if jit:
            import jax

            self._solve_raw = jax.jit(self._solve_raw)

        # Remember original device/method for auto-tune detection.
        # Auto-tune (benchmarking) fires when device='auto' OR method='auto'.
        # When method='auto' and device is explicit, we resolve the method
        # heuristically for initial construction, but keep _original_method='auto'
        # so auto-tune benchmarks methods on first solve.
        self._original_device = settings.device
        ipm = settings.ipm_settings
        self._original_method = ipm.direct_solve_method if ipm else "auto"
        if self._original_method == "auto" and settings.device != "auto":
            from moreau._backend import _rank_method
            import numpy as _np

            nnz_A = len(_np.asarray(A_col_indices))
            ranked = _rank_method(device, n, m, nnz_A, settings.batch_size)
            initial_method = ranked[0] if ranked else "auto"
            if ipm:
                ipm.direct_solve_method = initial_method

        # Auto-tune state
        self._auto_tuned = False
        self._tune_result = None

        # Last solve info (populated after solve())
        self._info = None

        # Stored P/A data for two-step API
        self._P_values = None
        self._A_values = None

    def setup(self, P_values, A_values):
        """Set P and A matrix values for subsequent solve() calls.

        After calling setup(), solve() can be called with just (q, b).
        Gradients w.r.t. P/A are still computed when using the 2-arg solve().

        Args:
            P_values: P matrix values, shape (nnzP,)
            A_values: A matrix values, shape (nnzA,)
        """
        import jax.numpy as jnp

        self._P_values = jnp.asarray(P_values, dtype=jnp.float64)
        self._A_values = jnp.asarray(A_values, dtype=jnp.float64)

    def _needs_auto_tune(self) -> bool:
        return _needs_auto_tune_shared(
            self._settings,
            self._auto_tuned,
            self._original_device,
            self._original_method,
        )

    def _auto_tune(self, P_values, A_values, q, b):
        """Benchmark solver configs and lock in the fastest.

        Delegates the benchmark+winner-selection to the shared
        `_benchmark_and_select_winner` helper, then rebuilds this
        Solver's JAX impl with the winning device/method.
        """
        import warnings

        warnings.warn(
            "First solve is benchmarking solver configurations "
            "(auto_tune=True). Subsequent solves will be faster. "
            "Set auto_tune=False (default) to use heuristic selection "
            "without benchmarking.",
            stacklevel=3,
        )
        import numpy as np

        # Save original stored P/A (from setup()) so we don't overwrite
        # them when called from the 4-arg solve path.
        stored_P_values = self._P_values
        stored_A_values = self._A_values

        q_np = np.asarray(q, dtype=np.float64)
        b_np = np.asarray(b, dtype=np.float64)
        P_np = np.asarray(P_values, dtype=np.float64)
        A_np = np.asarray(A_values, dtype=np.float64)
        P_ro = np.asarray(self._P_row_offsets, dtype=np.int64)
        P_ci = np.asarray(self._P_col_indices, dtype=np.int64)
        A_ro = np.asarray(self._A_row_offsets, dtype=np.int64)
        A_ci = np.asarray(self._A_col_indices, dtype=np.int64)

        # Ensure 2D for CompiledSolver
        if q_np.ndim == 1:
            q_np = q_np.reshape(1, -1)
        if b_np.ndim == 1:
            b_np = b_np.reshape(1, -1)

        ipm = self._settings.ipm_settings
        current_method = ipm.direct_solve_method if ipm else "auto"

        out = _benchmark_and_select_winner(
            settings=self._settings,
            cones=self._cones,
            b_sparsity_pattern=self._b_sparsity_pattern,
            n=self._n,
            m=self._m,
            P_ro=P_ro,
            P_ci=P_ci,
            A_ro=A_ro,
            A_ci=A_ci,
            P_np=P_np,
            A_np=A_np,
            q_np=q_np,
            b_np=b_np,
            original_device=self._original_device,
            original_method=self._original_method,
            current_device=self._device,
            component_key="jax_solver",
            component_cpu_fallback=True,
        )
        if out is None:
            self._auto_tuned = True
            return

        best_device, best_method, time_limit, _results, tune_result = out
        self._tune_result = tune_result

        if (best_device, best_method) != (self._device, current_method):
            winning_settings = self._settings.model_copy(deep=True)
            winning_settings.device = best_device
            if winning_settings.ipm_settings is None:
                winning_settings.ipm_settings = IPMSettings()
            winning_settings.ipm_settings.direct_solve_method = best_method
            winning_settings.time_limit = time_limit
            self._settings = winning_settings
            self._device = best_device

            JaxSolverClass = get_device_component(best_device, "jax_solver")
            if JaxSolverClass is None and best_device != "cpu":
                JaxSolverClass = get_device_component("cpu", "jax_solver")

            self._impl = JaxSolverClass(
                self._n,
                self._m,
                self._P_row_offsets,
                self._P_col_indices,
                self._A_row_offsets,
                self._A_col_indices,
                self._cones,
                self._settings,
                b_sparsity_pattern=self._b_sparsity_pattern,
            )
            self._solve_raw = self._impl.solve
            if self._jit:
                import jax

                self._solve_raw = jax.jit(self._solve_raw)

            # Re-apply stored P/A data only if setup() was called
            if stored_P_values is not None:
                self._P_values = stored_P_values
                self._A_values = stored_A_values

        self._auto_tuned = True

    def solve(self, *args, warm_start=None) -> JaxSolution:
        """Solve the optimization problem.

        Two signatures supported:
        - solve(q, b): Uses P/A from setup()
        - solve(P_values, A_values, q, b): Full signature

        Args (2-arg form):
            q: Linear cost, shape (n,) or (batch, n)
            b: Constraint RHS, shape (m,) or (batch, m)

        Args (4-arg form):
            P_values: P matrix values, shape (nnzP,) or (batch, nnzP)
            A_values: A matrix values, shape (nnzA,) or (batch, nnzA)
            q, b: Same as 2-arg form.

        Keyword Args:
            warm_start: Optional WarmStart or BatchedWarmStart from a previous
                solve (e.g. ``solution.to_warm_start()``). Gradients do NOT
                flow through warm start values.

        Returns:
            JaxSolution: NamedTuple with x, z, s

        Note:
            Solver metadata (status, timing) is available via solver.info
            after calling solve().
        """
        if len(args) == 2:
            if self._P_values is None:
                raise RuntimeError("setup() must be called before solve(q, b)")
            q, b = args
            P_values, A_values = self._P_values, self._A_values
        elif len(args) == 4:
            P_values, A_values, q, b = args
        else:
            raise TypeError(f"solve() takes 2 or 4 arguments, got {len(args)}")

        # Auto-tune on first solve when device or method is 'auto'
        if self._needs_auto_tune():
            self._auto_tune(P_values, A_values, q, b)

        if warm_start is not None:
            if not isinstance(warm_start, (WarmStart, BatchedWarmStart)):
                raise TypeError(
                    f"warm_start must be a WarmStart or BatchedWarmStart, "
                    f"got {type(warm_start).__name__}"
                )

            # Use the dedicated warm-start solve path when available (FFI).
            # This passes warm arrays as proper JAX tensor arguments instead
            # of relying on a Python side-channel.
            import jax.numpy as jnp

            solve_warm_fn = self._impl.solve_warm
            if solve_warm_fn is not None:
                warm_x = jnp.asarray(warm_start.x, dtype=jnp.float64)
                warm_z = jnp.asarray(warm_start.z, dtype=jnp.float64)
                warm_s = jnp.asarray(warm_start.s, dtype=jnp.float64)
                # Direct dual: pass through if present, else zero-length
                # placeholder (the FFI handler ignores it when total_xn==0).
                total_xn = getattr(self._impl, "_total_xn", 0)
                if warm_start.z_x is not None:
                    warm_z_x = jnp.asarray(warm_start.z_x, dtype=jnp.float64)
                elif warm_x.ndim == 1:
                    warm_z_x = jnp.zeros((total_xn,), dtype=jnp.float64)
                else:
                    warm_z_x = jnp.zeros((warm_x.shape[0], total_xn), dtype=jnp.float64)
                solution, info = solve_warm_fn(
                    P_values, A_values, q, b, warm_x, warm_z, warm_s, warm_z_x
                )
            else:
                # Fallback: store on impl for the callback to pick up
                import numpy as np

                self._impl._pending_warm_start = {
                    "warm_x": np.asarray(warm_start.x, dtype=np.float64),
                    "warm_z": np.asarray(warm_start.z, dtype=np.float64),
                    "warm_s": np.asarray(warm_start.s, dtype=np.float64),
                    "warm_z_x": (
                        np.asarray(warm_start.z_x, dtype=np.float64)
                        if warm_start.z_x is not None
                        else None
                    ),
                }
                solution, info = self._solve_raw(P_values, A_values, q, b)
                self._impl._pending_warm_start = None
        else:
            solution, info = self._solve_raw(P_values, A_values, q, b)

        # Auto-retry cold if warm start may have caused the failure.
        # Helpers in _types.py keep no-retry status set + warning
        # text in sync with the unified Solver/CompiledSolver paths.
        if warm_start is not None:
            import jax.numpy as jnp
            from moreau._types import _normalize_status, _should_retry_cold, _warn_warm_retry

            status_enum = _normalize_status(int(jnp.asarray(info.status)))
            ipm = self._settings.ipm_settings
            if _should_retry_cold(status_enum, ipm):
                _warn_warm_retry([0], status_enum)
                solution, info = self._solve_raw(P_values, A_values, q, b)

        # Store info on the solver object (for non-vmap use)
        # Note: for vmap, info will be batched and this stores the last call
        self._info = info

        return solution

    @property
    def info(self) -> Optional[JaxSolveInfo]:
        """Metadata from the last solve() call.

        Returns None if solve() has not been called yet.

        Contains:
            - status: Solver status (as int, SolverStatus enum value)
            - obj_val: Objective value at solution
            - iterations: Number of IPM iterations
            - solve_time: Time in IPM iterations (seconds)
            - setup_time: Time setting matrix values (seconds)
            - construction_time: Time constructing solver (seconds)

        Note: For vmap calls, this returns info from the last single call,
        not the batched result. Access batched info via the returned tuple.
        """
        return self._info

    @property
    def device(self) -> str:
        """Return the active device name ('cpu' or 'cuda')."""
        return self._device

    @property
    def n(self) -> int:
        """Number of primal variables."""
        return self._n

    @property
    def m(self) -> int:
        """Number of constraints."""
        return self._m

    @property
    def construction_time(self) -> float:
        """Time spent constructing solver structure (seconds)."""
        return self._construction_time

    @property
    def tune_result(self):
        """Result from auto-tuning on the first solve() call.

        Returns None if auto-tune has not run yet (e.g. device and method
        were set explicitly, or solve() has not been called).
        """
        return self._tune_result


__all__ = ["Solver"]
