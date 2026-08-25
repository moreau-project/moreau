"""
Moreau PyTorch integration with unified CPU/CUDA support.

Provides differentiable convex optimization for PyTorch neural networks
with automatic device selection between CPU and CUDA backends.

Example:
    >>> import torch
    >>> from moreau.torch import Solver
    >>> import moreau
    >>>
    >>> # Setup problem structure
    >>> cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
    >>>
    >>> # Create solver (auto-selects CUDA if available)
    >>> solver = Solver(n=2, m=3, P_row_offsets=..., P_col_indices=...,
    ...                 A_row_offsets=..., A_col_indices=..., cones=cones)
    >>>
    >>> # Solve with autograd support (stateless — all data passed per call)
    >>> solution = solver.solve(P_values, A_values, q, b)
    >>> print(solution.x, solver.info.status)
    >>>
    >>> # Backpropagate through the solution
    >>> loss = solution.x.sum()
    >>> loss.backward()
"""

import warnings

import torch
from typing import Tuple, Optional, Sequence

# Use centralized backend for device availability
from moreau._backend import (
    torch_available,
    default_device,
    get_device_component,
)

from moreau._types import (
    Cones,
    Settings,
    IPMSettings,
    WarmStart,
    BatchedWarmStart,
    _normalize_status,
    _should_retry_cold,
    _warn_warm_retry,
)
from ._types import (
    TorchSolveInfo,
    TorchSolution,
    TorchBatchedSolveInfo,
    TorchBatchedSolution,
)
from moreau._backend import (
    benchmark_and_select_winner as _benchmark_and_select_winner,
    needs_auto_tune as _needs_auto_tune_shared,
)

# Register CPU torch solver now that moreau._backend has finished loading.
# This import was impossible inside _backend._register_cpu() because
# _backend.py loads during moreau/__init__.py, creating a circular import.
from moreau._backend import _device_registry as _dr

if "cpu" in _dr and _dr["cpu"].get("torch_solver") is None:
    from ._cpu_impl import _TorchSolverCpu

    _dr["cpu"]["torch_solver"] = _TorchSolverCpu
del _dr


def _validate_tensor_dtype(t: torch.Tensor, name: str) -> None:
    """Validate that a tensor is float64."""
    if t.dtype != torch.float64:
        raise RuntimeError(f"{name} must be float64, got {t.dtype}")


# Custom-op / autograd plumbing — see torch/_autograd.py.
# `_solve_backward_op` is re-exported so it remains patchable via
# `moreau.torch._solve_backward_op` (tests rely on this).
from ._autograd import _register_impl, _SolveFunction, _solve_backward_op


class Solver:
    """Unified PyTorch solver with automatic device selection and autograd support.

    Thread Safety:
        Solver instances are NOT thread-safe. Each thread should use its own
        Solver instance, or external synchronization must be used. Calling
        solve() concurrently from multiple threads on the same Solver instance
        will cause data races and undefined behavior.

    Args:
        n: Number of primal variables
        m: Number of constraints
        P_row_offsets: CSR row pointers for P matrix (torch.Tensor, CPU).
            P must be full symmetric (both upper and lower triangles).
        P_col_indices: CSR column indices for P matrix (torch.Tensor, CPU)
        A_row_offsets: CSR row pointers for A matrix (torch.Tensor, CPU)
        A_col_indices: CSR column indices for A matrix (torch.Tensor, CPU)
        cones: Cone specification (moreau.Cones object)
        settings: Optional solver settings (moreau.Settings object).
                  Contains device, batch_size, enable_grad and solver tolerances.
                  Note: enable_grad defaults to True for torch.Solver if not specified.
        b_sparsity_pattern: Optional boolean mask of length m indicating which
                  entries of b are structurally nonzero. Enables more aggressive
                  chordal decomposition for PSD cones.

    Example:
        >>> import torch
        >>> from moreau.torch import Solver
        >>> import moreau
        >>>
        >>> cones = moreau.Cones(num_nonneg_cones=2)
        >>> settings = moreau.Settings(device='cuda', batch_size=64)
        >>>
        >>> solver = Solver(
        ...     n=2, m=2,
        ...     P_row_offsets=torch.tensor([0, 1, 2]),
        ...     P_col_indices=torch.tensor([0, 1]),
        ...     A_row_offsets=torch.tensor([0, 1, 2]),
        ...     A_col_indices=torch.tensor([0, 1]),
        ...     cones=cones,
        ...     settings=settings,
        ... )
        >>>
        >>> solution = solver.solve(P_values, A_values, q, b)
        >>> print(solver.info.status, solver.info.obj_val)
    """

    def __init__(
        self,
        n: int,
        m: int,
        P_row_offsets: torch.Tensor,
        P_col_indices: torch.Tensor,
        A_row_offsets: torch.Tensor,
        A_col_indices: torch.Tensor,
        cones: Cones,
        settings: Optional[Settings] = None,
        b_sparsity_pattern: Optional[Sequence[bool]] = None,
    ):
        # Check for old CVXPY with SOC cones
        from moreau import _warn_cvxpy_soc_if_needed, _require_x_cones_compatible

        _warn_cvxpy_soc_if_needed(cones)
        _require_x_cones_compatible(cones, settings)

        # Use default settings if not provided, with enable_grad=True for torch
        # For torch.Solver, we always enable gradients since that's its purpose
        if settings is None:
            settings = Settings(enable_grad=True)
        elif not settings.enable_grad:
            # Force enable_grad for torch.Solver - that's its purpose
            # Settings is a pydantic model, so use model_copy
            settings = settings.model_copy(update={"enable_grad": True})

        # Extract device, batch_size, enable_grad from settings
        device = settings.device
        batch_size = settings.batch_size
        enable_grad = settings.enable_grad

        # Resolve 'auto' to best device with torch support FIRST,
        # so active-set auto-select can see the actual target device
        if device == "auto":
            device = "cpu"
            for dev in [default_device()]:
                if dev != "cpu" and torch_available(dev):
                    device = dev
                    break

        # Now resolve solver='auto' with the actual device known
        from moreau import _resolve_solver_type

        nnz_P = len(P_col_indices)
        settings, device = _resolve_solver_type(settings, n, m, cones, device, nnz_P=nnz_P)

        # Validate device availability before doing any tensor .to(device) calls,
        # which would otherwise raise inscrutable PyTorch internals errors (e.g.
        # "Torch not compiled with CUDA enabled") instead of the clean ImportError below.
        TorchSolverClass = get_device_component(device, "torch_solver")
        if TorchSolverClass is None:
            raise ImportError(
                f"PyTorch extension for device '{device}' not available. "
                f"Use device='cpu' or install the appropriate backend."
            )

        self._device = device
        self._n = n
        self._m = m
        self._settings = settings
        self._cones = cones
        self._batch_size = batch_size
        self._b_sparsity_pattern = (
            list(b_sparsity_pattern) if b_sparsity_pattern is not None else None
        )

        # Normalize structural indices to the solver's target device.
        # Users can pass tensors on any device (CPU or CUDA) and we move them.
        target = torch.device("cpu" if device == "cpu" else device)
        P_row_offsets = (
            P_row_offsets.to(target) if isinstance(P_row_offsets, torch.Tensor) else P_row_offsets
        )
        P_col_indices = (
            P_col_indices.to(target) if isinstance(P_col_indices, torch.Tensor) else P_col_indices
        )
        A_row_offsets = (
            A_row_offsets.to(target) if isinstance(A_row_offsets, torch.Tensor) else A_row_offsets
        )
        A_col_indices = (
            A_col_indices.to(target) if isinstance(A_col_indices, torch.Tensor) else A_col_indices
        )

        self._P_row_offsets = P_row_offsets
        self._P_col_indices = P_col_indices
        self._A_row_offsets = A_row_offsets
        self._A_col_indices = A_col_indices

        self._TorchSolverClass = TorchSolverClass

        # Track construction time
        import time

        construction_start = time.perf_counter()

        # Note: Don't convert cones or settings here - the device's torch_wrapper
        # expects Python dataclasses and does its own conversion.
        # Use unified interface: batch_size is optional keyword argument
        # Pass enable_grad to pre-compute gradient structures in constructor
        self._impl = TorchSolverClass(
            n,
            m,
            P_row_offsets,
            P_col_indices,
            A_row_offsets,
            A_col_indices,
            cones,
            settings,
            batch_size=batch_size,
            enable_grad=enable_grad,
            b_sparsity_pattern=self._b_sparsity_pattern,
        )

        # Setup gradients if requested - this pre-allocates Python-side buffers
        if enable_grad:
            self._impl.setup_grad(batch_size)

        self._construction_time = time.perf_counter() - construction_start
        self._setup_time = 0.0

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

            ranked = _rank_method(self._device, self._n, self._m, len(A_col_indices), batch_size)
            initial_method = ranked[0] if ranked else "auto"
            if ipm:
                ipm.direct_solve_method = initial_method

        # Auto-tune state
        self._auto_tuned = False
        self._tune_result = None

        # Last solve info (populated after solve())
        self._info = None

        # Register _impl in the backward op's registry so the vmap rule
        # can look it up from a tensor handle.
        self._impl_handle = _register_impl(self._impl)

    def setup(
        self,
        P_values: torch.Tensor,
        A_values: torch.Tensor,
    ):
        """Set P and A matrix values.

        Must be called before solve(). Can be called multiple times to update
        values for repeated solves with the same structure.

        Args:
            P_values: P matrix values, shape (batch, nnzP) or (nnzP,), float64
            A_values: A matrix values, shape (batch, nnzA) or (nnzA,), float64
        """
        import time

        _validate_tensor_dtype(P_values, "P_values")
        _validate_tensor_dtype(A_values, "A_values")

        # Validate all inputs are on the same device
        if P_values.device != A_values.device:
            raise ValueError(
                f"P_values (device={P_values.device}) and A_values "
                f"(device={A_values.device}) must be on the same device"
            )
        self._input_device = P_values.device

        # Move values to solver's device if needed
        target = torch.device("cpu" if self._device == "cpu" else self._device)
        if P_values.device != target:
            P_values = P_values.to(target)
        if A_values.device != target:
            A_values = A_values.to(target)

        self._P_values = P_values
        self._A_values = A_values
        setup_start = time.perf_counter()
        self._impl.setup(P_values, A_values)
        self._setup_time = time.perf_counter() - setup_start

    def _needs_auto_tune(self) -> bool:
        return _needs_auto_tune_shared(
            self._settings,
            self._auto_tuned,
            self._original_device,
            self._original_method,
        )

    def _auto_tune(self, q: torch.Tensor, b: torch.Tensor):
        """Benchmark solver configs and lock in the fastest.

        Delegates the benchmark+winner-selection to the shared
        `_benchmark_and_select_winner` helper, then rebuilds this
        Solver's torch impl with the winning device/method.
        """
        warnings.warn(
            "First solve is benchmarking solver configurations "
            "(auto_tune=True). Subsequent solves will be faster. "
            "Set auto_tune=False (default) to use heuristic selection "
            "without benchmarking.",
            stacklevel=3,
        )
        import numpy as np

        q_np = q.detach().cpu().numpy().astype(np.float64)
        b_np = b.detach().cpu().numpy().astype(np.float64)
        P_np = self._P_values.detach().cpu().numpy().astype(np.float64)
        A_np = self._A_values.detach().cpu().numpy().astype(np.float64)
        _cpu = lambda t: t.cpu() if hasattr(t, "cpu") else t
        P_ro = np.asarray(_cpu(self._P_row_offsets), dtype=np.int64)
        P_ci = np.asarray(_cpu(self._P_col_indices), dtype=np.int64)
        A_ro = np.asarray(_cpu(self._A_row_offsets), dtype=np.int64)
        A_ci = np.asarray(_cpu(self._A_col_indices), dtype=np.int64)

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
            component_key="torch_solver",
            component_cpu_fallback=False,
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

            TorchSolverClass = get_device_component(best_device, "torch_solver")
            if TorchSolverClass is not None:
                self._TorchSolverClass = TorchSolverClass

            # Move structural indices to the winning device
            target = torch.device("cpu" if best_device == "cpu" else best_device)
            self._P_row_offsets = (
                self._P_row_offsets.to(target)
                if isinstance(self._P_row_offsets, torch.Tensor)
                else self._P_row_offsets
            )
            self._P_col_indices = (
                self._P_col_indices.to(target)
                if isinstance(self._P_col_indices, torch.Tensor)
                else self._P_col_indices
            )
            self._A_row_offsets = (
                self._A_row_offsets.to(target)
                if isinstance(self._A_row_offsets, torch.Tensor)
                else self._A_row_offsets
            )
            self._A_col_indices = (
                self._A_col_indices.to(target)
                if isinstance(self._A_col_indices, torch.Tensor)
                else self._A_col_indices
            )

            self._impl = self._TorchSolverClass(
                self._n,
                self._m,
                self._P_row_offsets,
                self._P_col_indices,
                self._A_row_offsets,
                self._A_col_indices,
                self._cones,
                self._settings,
                batch_size=self._batch_size,
                enable_grad=self._settings.enable_grad,
                b_sparsity_pattern=self._b_sparsity_pattern,
            )
            self._impl_handle = _register_impl(self._impl)
            if self._settings.enable_grad:
                self._impl.setup_grad(self._batch_size)

            # Move stored tensors to the winning device
            target = "cpu" if best_device == "cpu" else best_device
            self._P_values = self._P_values.to(target)
            self._A_values = self._A_values.to(target)
            self._impl.setup(self._P_values, self._A_values)

        self._auto_tuned = True

    def solve(
        self,
        P_values: torch.Tensor,
        A_values: torch.Tensor,
        q: torch.Tensor,
        b: torch.Tensor,
        *,
        warm_start=None,
    ):
        """Solve the optimization problem.

        Stateless call — all problem data is passed as arguments.
        Safe for chained solves in autograd graphs (e.g. MPC rollouts).
        P/A setup is cached internally: if P and A haven't changed since
        the last call, the expensive setup step is skipped automatically.

        Args:
            P_values: P matrix values, shape (batch, nnzP) or (nnzP,), float64
            A_values: A matrix values, shape (batch, nnzA) or (nnzA,), float64
            q: Linear cost vector, shape (batch, n) or (n,)
            b: Constraint RHS, shape (batch, m) or (m,)
            warm_start: Optional WarmStart or BatchedWarmStart from a previous
                solve (e.g. ``solution.to_warm_start()``). Gradients do NOT
                flow through warm start values.

        Returns:
            Solution object:
            - For single problems: TorchSolution
            - For batched problems: TorchBatchedSolution

        Example:
            >>> solver = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones)
            >>> solution = solver.solve(P_values, A_values, q, b)
            >>> loss = solution.x.sum()
            >>> loss.backward()
        """
        self.setup(P_values, A_values)

        # Auto-tune on first solve when device or method is 'auto'
        if self._needs_auto_tune():
            self._auto_tune(q, b)

        # Validate q and b are on the same device as each other and as setup() inputs
        if q.device != b.device:
            raise ValueError(
                f"q (device={q.device}) and b (device={b.device}) " f"must be on the same device"
            )
        input_device = getattr(self, "_input_device", None)
        if input_device is not None and q.device != input_device:
            raise ValueError(
                f"solve() inputs (device={q.device}) must be on the same device "
                f"as setup() inputs (device={input_device})"
            )
        input_device = q.device

        # Move inputs to solver's device if needed (e.g. after auto-tune
        # changed device, or user passed tensors on a different device)
        target = torch.device("cpu" if self._device == "cpu" else self._device)
        if q.device != target:
            q = q.to(target)
        if b.device != target:
            b = b.to(target)

        # Convert warm_start to CUDA torch tensors (non-differentiable)
        if warm_start is not None:
            if not isinstance(warm_start, (WarmStart, BatchedWarmStart)):
                raise TypeError(
                    f"warm_start must be a WarmStart or BatchedWarmStart, "
                    f"got {type(warm_start).__name__}"
                )

            device = q.device
            warm_x = torch.tensor(warm_start.x, dtype=torch.float64, device=device)
            warm_z = torch.tensor(warm_start.z, dtype=torch.float64, device=device)
            warm_s = torch.tensor(warm_start.s, dtype=torch.float64, device=device)
            warm_z_x = None
            if getattr(warm_start, "z_x", None) is not None:
                warm_z_x = torch.tensor(warm_start.z_x, dtype=torch.float64, device=device)

            # Ensure 2D for batched solver
            if warm_x.dim() == 1:
                warm_x = warm_x.unsqueeze(0)
                warm_z = warm_z.unsqueeze(0)
                warm_s = warm_s.unsqueeze(0)
                if warm_z_x is not None:
                    warm_z_x = warm_z_x.unsqueeze(0)

            self._pending_warm_kwargs = {
                "warm_x": warm_x,
                "warm_z": warm_z,
                "warm_s": warm_s,
            }
            if warm_z_x is not None:
                self._pending_warm_kwargs["warm_z_x"] = warm_z_x
        else:
            self._pending_warm_kwargs = {}

        x, z, s, z_x = _SolveFunction.apply(self, q, b, self._P_values, self._A_values)

        # Clear pending warm start
        self._pending_warm_kwargs = {}

        # Get cached result info
        cached = self._last_result

        # Determine if batched based on output shape
        is_batched = x.dim() > 1

        # Auto-retry cold if warm start may have caused the failure.
        # Helpers in _types.py keep the no-retry set + warning text
        # in sync with the unified Solver/CompiledSolver paths.
        if warm_start is not None:
            ipm = self._settings.ipm_settings
            if is_batched:
                statuses = [_normalize_status(int(st)) for st in cached["status"]]
            else:
                statuses = _normalize_status(int(cached["status"]))
            failed_idx = _should_retry_cold(statuses, ipm)
            if failed_idx:
                _warn_warm_retry(failed_idx, statuses)
                self._pending_warm_kwargs = {}
                x, z, s, z_x = _SolveFunction.apply(self, q, b, self._P_values, self._A_values)
                cached = self._last_result

        # Move results back to the caller's input device
        if x.device != input_device:
            x = x.to(input_device)
            z = z.to(input_device)
            s = s.to(input_device)
            if z_x.numel() > 0:
                z_x = z_x.to(input_device)

        if is_batched:
            status_list = [_normalize_status(int(st)) for st in cached["status"]]

            solution = TorchBatchedSolution(
                x=x,
                z=z,
                s=s,
                z_x=z_x,
            )
            self._info = TorchBatchedSolveInfo(
                status=status_list,
                obj_val=cached["obj_val"],
                iterations=cached["iterations"],
                solve_time=cached["solve_time"],
                setup_time=self._setup_time,
                construction_time=self._construction_time,
            )
            return solution
        else:
            status = _normalize_status(int(cached["status"]))

            solution = TorchSolution(
                x=x,
                z=z,
                s=s,
                z_x=z_x,
            )
            self._info = TorchSolveInfo(
                status=status,
                obj_val=cached["obj_val"],
                iterations=cached["iterations"],
                solve_time=cached["solve_time"],
                setup_time=self._setup_time,
                construction_time=self._construction_time,
            )
            return solution

    def backward(
        self,
        dx: torch.Tensor,
        dz: Optional[torch.Tensor] = None,
        ds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gradients via implicit differentiation."""
        return self._impl.backward(dx, dz, ds)

    def setup_grad(self, batch_size: Optional[int] = None):
        """Setup for gradient computation (backward pass).

        Pre-allocates memory and preprocessing for backward(). Optional but
        recommended when calling backward() repeatedly.

        Args:
            batch_size: Optional batch size for pre-allocation.
        """
        self._impl.setup_grad(batch_size)

    @property
    def grad_initialized(self) -> bool:
        """Whether setup_grad() has been called."""
        return self._impl.grad_initialized

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
    def batch_size(self) -> int:
        """Current batch size."""
        return self._impl.batch_size

    @property
    def is_initialized(self) -> bool:
        """Whether solver has been initialized."""
        return self._impl.is_initialized

    @property
    def nnzP(self) -> int:
        """Number of non-zeros in P matrix."""
        return self._impl.nnzP

    @property
    def nnzA(self) -> int:
        """Number of non-zeros in A matrix."""
        return self._impl.nnzA

    @property
    def construction_time(self) -> float:
        """Time spent constructing solver structure (seconds)."""
        return self._construction_time

    def reset(self):
        """Reset solver state."""
        self._impl.reset()

    def get_dimensions(self) -> dict:
        """Return problem dimensions."""
        return self._impl.get_dimensions()

    @property
    def tune_result(self):
        """Result from auto-tuning on the first solve() call.

        Returns None if auto-tune has not run yet (e.g. device and method
        were set explicitly, or solve() has not been called).
        """
        return self._tune_result

    @property
    def info(self):
        """Metadata from the last solve() call.

        Returns None if solve() has not been called yet.

        Contains:
            - status: SolverStatus enum (or list for batched)
            - obj_val: Objective value tensor
            - iterations: Number of IPM iterations
            - solve_time: Time in IPM iterations (seconds)
            - setup_time: Time setting matrix values (seconds)
            - construction_time: Time constructing solver (seconds)
        """
        return self._info


__all__ = [
    "Solver",
]
