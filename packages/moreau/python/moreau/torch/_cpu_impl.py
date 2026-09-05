"""CPU implementation of TorchSolver API.

This module provides a unified CPU backend for the PyTorch solver interface,
automatically handling both single problems and batched problems:
- 1D inputs → single problem using DefaultSolver
- 2D inputs → batched problems using CompiledSolver (rayon parallelism)
- Automatic batch size management with warnings for frequent changes

Two-step API pattern:
1. Construct solver with structure (dimensions, sparsity patterns, cones)
2. solve(P_values, A_values, q, b) - Solve with all problem data
"""

import torch
import numpy as np
from typing import Tuple, Optional, Any
import warnings

# Import CPU solver extension
import moreau_cpu
from moreau_cpu import _cpu_solver
from moreau_cpu import cones_to_cpu, settings_to_cpu


class _TorchSolverCpu:
    """Unified CPU implementation of TorchSolver API.

    Automatically handles both single problems (1D inputs) and batched problems
    (2D inputs). Uses DefaultSolver for single problems and parallel CompiledSolver
    for batched problems.

    Two-step API pattern:
    1. Construct solver with structure (dimensions, sparsity patterns, cones)
    2. solve(P_values, A_values, q, b) - Solve with all problem data

    Args:
        n: Number of primal variables
        m: Number of constraints
        P_row_offsets: CSR row pointers for P matrix
        P_col_indices: CSR column indices for P matrix
        A_row_offsets: CSR row pointers for A matrix
        A_col_indices: CSR column indices for A matrix
        cones: Cone specification (moreau.Cones object)
        settings: Optional solver settings
        batch_size: Optional batch size hint. If provided, pre-allocates
                    CompiledSolver for this size. If actual batch size differs,
                    solver will adapt (with warning if changes are frequent).

    Example:
        >>> solver = _TorchSolverCpu(n, m, P_ro, P_ci, A_ro, A_ci, cones)
        >>>
        >>> # Single problem (1D inputs)
        >>> result = solver.solve(P_values, A_values, q, b)
        >>>
        >>> # Batched problems (2D inputs)
        >>> result = solver.solve(P_values, A_values, q_batch, b_batch)
    """

    # Warning threshold: warn if batch size changes more than this many times
    _BATCH_CHANGE_WARN_THRESHOLD = 5

    def __init__(
        self,
        n: int,
        m: int,
        P_row_offsets: torch.Tensor,
        P_col_indices: torch.Tensor,
        A_row_offsets: torch.Tensor,
        A_col_indices: torch.Tensor,
        cones: Any,
        settings: Optional[Any] = None,
        batch_size: Optional[int] = None,
        enable_grad: bool = False,
        b_sparsity_pattern=None,
    ):
        self._n = n
        self._m = m
        self._expected_batch_size = batch_size  # User-provided hint (or None)
        self._is_initialized = False
        self._grad_initialized = False
        self._enable_grad = enable_grad  # Whether to pre-compute gradient structures
        self._b_sparsity_pattern = (
            list(b_sparsity_pattern) if b_sparsity_pattern is not None else None
        )

        # Pre-allocated buffers for backward pass
        self._grad_dx_buffer = None
        self._grad_dz_buffer = None
        self._grad_ds_buffer = None

        # Convert tensors/lists/arrays to numpy
        def _to_numpy(x):
            if hasattr(x, "cpu"):
                return x.cpu().numpy().astype(np.int64)
            return np.asarray(x, dtype=np.int64)

        self._P_row_offsets = _to_numpy(P_row_offsets)
        self._P_col_indices = _to_numpy(P_col_indices)
        self._A_row_offsets = _to_numpy(A_row_offsets)
        self._A_col_indices = _to_numpy(A_col_indices)

        self._nnz_P = len(self._P_col_indices)
        self._nnz_A = len(self._A_col_indices)

        # Stored matrix values (set via setup)
        self._P_values: Optional[torch.Tensor] = None
        self._A_values: Optional[torch.Tensor] = None

        # Detect active-set solver request
        self._use_active_set = False
        self._original_settings = settings
        self._original_cones = cones  # Keep original for active-set (needs num_zero_cones etc.)
        if settings is not None:
            solver_type = getattr(settings, "solver", None)
            if solver_type is not None and "active_set" in str(solver_type).lower():
                self._use_active_set = True

        # Convert cones to CPU format
        if hasattr(cones, "num_zero_cones"):
            self._cones = cones_to_cpu(cones)
        else:
            # Assume already in CPU format
            self._cones = cones

        # Handle settings
        if settings is None:
            self._settings = _cpu_solver.DefaultSettings()
            # Default to verbose when no settings provided
            self._settings.verbose = True
        else:
            self._settings = settings_to_cpu(settings)

        # Solver instances (created on demand)
        self._default_solver = None  # For single problems, cached for backward()
        self._batch_solver = None  # For batched problems (fast, numpy-based)
        self._current_batch_size = None  # Current batch solver batch size

        # Track which solver was used for the last forward pass
        self._last_solve_mode = None  # 'single' or 'batch'

        # Batch size change tracking
        self._batch_size_change_count = 0
        self._warned_about_batch_changes = False

        self._is_initialized = True

    def _create_batch_solver(self, batch_size: int, b: Optional[np.ndarray] = None):
        """Create batch solver using moreau_cpu.Solver (fast, numpy-based).

        Uses the same underlying solver as JAX, which accepts numpy arrays
        directly without tolist() conversions.
        """
        if self._use_active_set:
            from moreau_cpu._cpu import ActiveSetSolver

            self._batch_solver = ActiveSetSolver(
                self._n,
                self._m,
                self._P_row_offsets,
                self._P_col_indices,
                self._A_row_offsets,
                self._A_col_indices,
                self._original_cones,
                self._original_settings,
                batch_size=batch_size,
                enable_grad=self._enable_grad,
            )
        else:
            b_sparsity_pattern = self._b_sparsity_pattern
            if (
                b_sparsity_pattern is None
                and b is not None
                and getattr(self._cones, "psd_dims", None)
            ):
                if b.ndim == 1:
                    b_sparsity_pattern = [abs(float(v)) > 0 for v in b]
                else:
                    b_sparsity_pattern = np.any(np.abs(b) > 0, axis=0).tolist()
            # Pass the original `Cones` object (with dir_cones intact) when
            # available — moreau_cpu.Solver dispatches direct routing
            # internally. cones_to_cpu strips dir_cones.
            cones_for_solver = (
                self._original_cones if hasattr(self._original_cones, "dir_cones") else self._cones
            )
            self._batch_solver = moreau_cpu.Solver(
                self._n,
                self._m,
                self._P_row_offsets,
                self._P_col_indices,
                self._A_row_offsets,
                self._A_col_indices,
                cones_for_solver,
                self._settings,
                batch_size=batch_size,
                enable_grad=self._enable_grad,
                b_sparsity_pattern=b_sparsity_pattern,
            )
        self._current_batch_size = batch_size

    def setup(
        self,
        P_values: torch.Tensor,
        A_values: torch.Tensor,
    ) -> None:
        """Set P and A matrix values.

        Must be called before solve(). Can be called multiple times to update
        values for repeated solves with the same structure.

        For batched problems, pass 2D tensors with shape (batch, nnz_P) and (batch, nnz_A).

        Args:
            P_values: P matrix values. Shape (nnz_P,) for single, (batch, nnz_P) for batched
            A_values: A matrix values. Shape (nnz_A,) for single, (batch, nnz_A) for batched

        Raises:
            ValueError: If value dimensions don't match sparsity pattern.
            RuntimeError: If tensors are not on CPU.

        Note:
            Tensors are automatically converted to float64 if needed.
        """
        # Validate and auto-convert tensors
        for name, tensor in [("P_values", P_values), ("A_values", A_values)]:
            if not tensor.device.type == "cpu":
                raise RuntimeError(
                    f"Tensor '{name}' is on device '{tensor.device}', but CPU solver requires CPU tensors. "
                    f"Move tensors to CPU with .cpu()."
                )
        # Auto-convert to float64 if needed (solver requires double precision)

        # Validate dimensions
        P_dim = P_values.shape[-1] if P_values.dim() > 0 else 0
        A_dim = A_values.shape[-1] if A_values.dim() > 0 else 0

        if P_dim != self._nnz_P:
            raise ValueError(f"P_values has {P_dim} elements, expected {self._nnz_P}")
        if A_dim != self._nnz_A:
            raise ValueError(f"A_values has {A_dim} elements, expected {self._nnz_A}")

        self._P_values = P_values.double()
        self._A_values = A_values.double()

    def _solve_single(
        self,
        P_values: np.ndarray,
        A_values: np.ndarray,
        q: np.ndarray,
        b: np.ndarray,
        warm_x: Optional[np.ndarray] = None,
        warm_z: Optional[np.ndarray] = None,
        warm_s: Optional[np.ndarray] = None,
    ) -> dict:
        """Solve a single problem using batch_solver with batch_size=1.

        Uses the fast numpy-based interface (moreau_cpu.Solver).
        """
        # Create batch_solver if needed
        if self._batch_solver is None or self._current_batch_size != 1:
            self._create_batch_solver(1, b)

        # Expand to 2D for batch solver
        P_batch = P_values.reshape(1, -1)
        A_batch = A_values.reshape(1, -1)
        q_batch = q.reshape(1, -1)
        b_batch = b.reshape(1, -1)

        # Setup matrices
        self._batch_solver.setup(P_batch, A_batch)

        # Build warm start kwargs
        warm_kwargs = {}
        if warm_x is not None:
            warm_kwargs["warm_x"] = warm_x.reshape(1, -1)
        if warm_z is not None:
            warm_kwargs["warm_z"] = warm_z.reshape(1, -1)
        if warm_s is not None:
            warm_kwargs["warm_s"] = warm_s.reshape(1, -1)

        # Solve with numpy arrays directly
        result = self._batch_solver.solve(q_batch, b_batch, **warm_kwargs)

        self._last_solve_mode = "batch"

        # Return dict matching numpy interface (1D tensors for single problem)
        x = torch.from_numpy(np.asarray(result["x"])[0])
        z = torch.from_numpy(np.asarray(result["z"])[0])
        s = torch.from_numpy(np.asarray(result["s"])[0])
        status = torch.tensor(int(result["status"][0]), dtype=torch.int32)
        obj_val = torch.tensor(result["obj_val"][0], dtype=torch.float64)
        obj_val_dual = torch.tensor(result["dual_obj_val"][0], dtype=torch.float64)
        iterations = torch.tensor(int(result["iterations"][0]), dtype=torch.int32)
        # r_prim and r_dual not returned by moreau_cpu.Solver
        r_prim = torch.tensor(0.0, dtype=torch.float64)
        r_dual = torch.tensor(0.0, dtype=torch.float64)
        solve_time = float(result["solve_time"])
        backward_state = result.get("_backward_state")

        # Direct dual (z_x): empty 1D tensor when no x-cones.
        z_x_arr = result.get("z_x")
        if z_x_arr is not None and getattr(z_x_arr, "size", 0):
            z_x = torch.from_numpy(np.asarray(z_x_arr)[0])
        else:
            z_x = torch.zeros(0, dtype=torch.float64)

        return {
            "x": x,
            "z": z,
            "s": s,
            "z_x": z_x,
            "status": status,
            "obj_val": obj_val,
            "obj_val_dual": obj_val_dual,
            "iterations": iterations,
            "r_prim": r_prim,
            "r_dual": r_dual,
            "solve_time": solve_time,
            "_backward_state": backward_state,
        }

    def _solve_batch(
        self,
        P_values: torch.Tensor,
        A_values: torch.Tensor,
        q: torch.Tensor,
        b: torch.Tensor,
        warm_x: Optional[np.ndarray] = None,
        warm_z: Optional[np.ndarray] = None,
        warm_s: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve batched problems with per-problem P/A using numpy interface."""
        batch_size = q.shape[0]

        # Check if we need to create/recreate batch_solver
        if self._batch_solver is None or self._current_batch_size != batch_size:
            # Only warn about batch size changes after the first solve
            if self._current_batch_size is not None:
                self._batch_size_change_count += 1

                if (
                    self._batch_size_change_count >= self._BATCH_CHANGE_WARN_THRESHOLD
                    and not self._warned_about_batch_changes
                ):
                    warnings.warn(
                        f"Batch size has changed {self._batch_size_change_count} times. "
                        f"Frequent batch size changes are slow because the batch solver must be recreated. "
                        f"Consider using a fixed batch_size or padding your batches to a consistent size.",
                        UserWarning,
                        stacklevel=4,
                    )
                    self._warned_about_batch_changes = True

            self._create_batch_solver(batch_size, b.numpy())

        # Convert to numpy (zero-copy for contiguous float64 tensors)
        P_np = P_values.numpy()
        A_np = A_values.numpy()
        q_np = q.numpy()
        b_np = b.numpy()

        # Setup matrices (per-problem P/A)
        self._batch_solver.setup(P_np, A_np)

        # Build warm start kwargs
        warm_kwargs = {}
        if warm_x is not None:
            warm_kwargs["warm_x"] = warm_x
        if warm_z is not None:
            warm_kwargs["warm_z"] = warm_z
        if warm_s is not None:
            warm_kwargs["warm_s"] = warm_s

        # Solve with numpy arrays directly
        result = self._batch_solver.solve(q_np, b_np, **warm_kwargs)

        # Convert results to torch tensors
        x = torch.from_numpy(np.asarray(result["x"]))
        z = torch.from_numpy(np.asarray(result["z"]))
        s = torch.from_numpy(np.asarray(result["s"]))
        status_vals = [int(st) for st in result["status"]]
        status = torch.tensor(status_vals, dtype=torch.int32)
        obj_val = torch.tensor(result["obj_val"], dtype=torch.float64)
        obj_val_dual = torch.tensor(result["dual_obj_val"], dtype=torch.float64)
        iterations = torch.tensor(result["iterations"], dtype=torch.int32)
        r_prim = torch.zeros(batch_size, dtype=torch.float64)
        r_dual = torch.zeros(batch_size, dtype=torch.float64)
        solve_time = float(result["solve_time"])
        backward_state = result.get("_backward_state")

        self._last_solve_mode = "batch"

        z_x_arr = result.get("z_x")
        if z_x_arr is not None and getattr(z_x_arr, "size", 0):
            z_x = torch.from_numpy(np.asarray(z_x_arr))
        else:
            z_x = torch.zeros((batch_size, 0), dtype=torch.float64)

        # Return dict matching numpy interface
        return {
            "x": x,
            "z": z,
            "s": s,
            "z_x": z_x,
            "status": status,
            "obj_val": obj_val,
            "obj_val_dual": obj_val_dual,
            "iterations": iterations,
            "r_prim": r_prim,
            "r_dual": r_dual,
            "solve_time": solve_time,
            "_backward_state": backward_state,
        }

    def _solve_batch_shared(
        self,
        P_values: torch.Tensor,
        A_values: torch.Tensor,
        q: torch.Tensor,
        b: torch.Tensor,
        warm_x: Optional[np.ndarray] = None,
        warm_z: Optional[np.ndarray] = None,
        warm_s: Optional[np.ndarray] = None,
    ) -> dict:
        """Solve batched problems with shared P/A using optimized numpy path.

        Uses moreau_cpu.Solver which accepts numpy arrays directly,
        avoiding expensive tolist() conversions.
        """
        batch_size = q.shape[0]

        # Create/recreate batch solver if needed
        need_new_solver = self._batch_solver is None or self._current_batch_size != batch_size
        if need_new_solver:
            if self._current_batch_size is not None:
                self._batch_size_change_count += 1
                if (
                    self._batch_size_change_count >= self._BATCH_CHANGE_WARN_THRESHOLD
                    and not self._warned_about_batch_changes
                ):
                    warnings.warn(
                        f"Batch size has changed {self._batch_size_change_count} times. "
                        f"Frequent batch size changes are slow because the batch solver must be recreated. "
                        f"Consider using a fixed batch_size or padding your batches to a consistent size.",
                        UserWarning,
                        stacklevel=4,
                    )
                    self._warned_about_batch_changes = True
            self._create_batch_solver(batch_size, b.numpy())
            self._cached_shared_P = None
            self._cached_shared_A = None

        # Convert to numpy (zero-copy for contiguous float64 tensors)
        P_np = P_values.numpy()
        A_np = A_values.numpy()
        q_np = q.numpy()
        b_np = b.numpy()

        # Check if P/A have changed - skip setup if unchanged
        P_unchanged = self._cached_shared_P is not None and torch.equal(
            self._cached_shared_P, P_values
        )
        A_unchanged = self._cached_shared_A is not None and torch.equal(
            self._cached_shared_A, A_values
        )

        if not (P_unchanged and A_unchanged):
            # Expand P/A to batch shape for setup
            P_batch = np.broadcast_to(P_np, (batch_size, P_np.shape[0])).copy()
            A_batch = np.broadcast_to(A_np, (batch_size, A_np.shape[0])).copy()
            self._batch_solver.setup(P_batch, A_batch)
            # Cache the values (detached to avoid holding onto computation graph)
            self._cached_shared_P = P_values.detach().clone()
            self._cached_shared_A = A_values.detach().clone()

        # Build warm start kwargs
        warm_kwargs = {}
        if warm_x is not None:
            warm_kwargs["warm_x"] = warm_x
        if warm_z is not None:
            warm_kwargs["warm_z"] = warm_z
        if warm_s is not None:
            warm_kwargs["warm_s"] = warm_s

        # Solve with numpy arrays directly (no tolist() needed!)
        result = self._batch_solver.solve(q_np, b_np, **warm_kwargs)

        # Convert results to torch tensors
        x = torch.from_numpy(np.asarray(result["x"]))
        z = torch.from_numpy(np.asarray(result["z"]))
        s = torch.from_numpy(np.asarray(result["s"]))
        # Convert status list to ints
        status_vals = [int(st) for st in result["status"]]
        status = torch.tensor(status_vals, dtype=torch.int32)
        obj_val = torch.tensor(result["obj_val"], dtype=torch.float64)
        obj_val_dual = torch.tensor(result["dual_obj_val"], dtype=torch.float64)
        iterations = torch.tensor(result["iterations"], dtype=torch.int32)
        # r_prim and r_dual are not returned by moreau_cpu.Solver - use zeros
        r_prim = torch.zeros(batch_size, dtype=torch.float64)
        r_dual = torch.zeros(batch_size, dtype=torch.float64)
        solve_time = float(result["solve_time"])
        backward_state = result.get("_backward_state")

        self._last_solve_mode = "batch"

        z_x_arr = result.get("z_x")
        if z_x_arr is not None and getattr(z_x_arr, "size", 0):
            z_x = torch.from_numpy(np.asarray(z_x_arr))
        else:
            z_x = torch.zeros((batch_size, 0), dtype=torch.float64)

        return {
            "x": x,
            "z": z,
            "s": s,
            "z_x": z_x,
            "status": status,
            "obj_val": obj_val,
            "obj_val_dual": obj_val_dual,
            "iterations": iterations,
            "r_prim": r_prim,
            "r_dual": r_dual,
            "solve_time": solve_time,
            "_backward_state": backward_state,
        }

    def solve(
        self,
        q: torch.Tensor,
        b: torch.Tensor,
        warm_x: Optional[torch.Tensor] = None,
        warm_z: Optional[torch.Tensor] = None,
        warm_s: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve the optimization problem.

        Requires setup() to be called first.

        Automatically detects single vs batched problems from input shape:
        - 1D inputs: Single problem, returns 1D results
        - 2D inputs: Batched problems, returns 2D results

        Args:
            q: Linear cost vector, shape (batch, n) or (n,)
            b: Constraint RHS, shape (batch, m) or (m,)
            warm_x: Optional warm start for primal variables
            warm_z: Optional warm start for dual variables
            warm_s: Optional warm start for slack variables

        Returns:
            Dict with keys: x, z, s, status, obj_val, etc. as CPU tensors

        Raises:
            RuntimeError: If setup() was not called first.
        """
        # Check that matrices have been set
        if self._P_values is None or self._A_values is None:
            raise RuntimeError("setup() must be called before solve()")

        # Validate tensors are on CPU (auto-convert to float64)
        for name, tensor in [("q", q), ("b", b)]:
            if not tensor.device.type == "cpu":
                raise RuntimeError(
                    f"Tensor '{name}' is on device '{tensor.device}', but CPU solver requires CPU tensors. "
                    f"Move tensors to CPU with .cpu() or use device='cuda'."
                )

        # Auto-convert to float64 (solver requires double precision)
        q = q.double()
        b = b.double()

        # Detect single vs batched from input shape
        is_single = q.dim() == 1

        # Validate dimensions
        q_dim = q.shape[-1] if q.dim() > 0 else 0
        b_dim = b.shape[-1] if b.dim() > 0 else 0

        if q_dim != self._n:
            raise RuntimeError(f"q dimension {q_dim} must be {self._n}")
        if b_dim != self._m:
            raise RuntimeError(f"b dimension {b_dim} must be {self._m}")

        # Convert warm start tensors to numpy once
        warm_np = {}
        if warm_x is not None:
            warm_np["warm_x"] = warm_x.cpu().double().numpy()
        if warm_z is not None:
            warm_np["warm_z"] = warm_z.cpu().double().numpy()
        if warm_s is not None:
            warm_np["warm_s"] = warm_s.cpu().double().numpy()

        # Get stored matrix values
        P_values = self._P_values
        A_values = self._A_values
        matrices_are_batched = P_values.dim() == 2

        # Single problem (1D inputs)
        if is_single:
            # Error if matrices are batched but q/b are single
            if matrices_are_batched:
                raise ValueError(
                    "Cannot use batched P/A matrices (2D) with single q/b vectors (1D). "
                    "Either use 1D P/A values or provide batched q/b."
                )
            P_np = P_values.numpy()
            A_np = A_values.numpy()
            q_np = q.numpy()
            b_np = b.numpy()

            return self._solve_single(P_np, A_np, q_np, b_np, **warm_np)

        # Batched problems (2D inputs)
        batch_size = q.shape[0]

        # Validate batch size consistency when matrices are batched
        if matrices_are_batched:
            matrix_batch_size = P_values.shape[0]
            if matrix_batch_size != batch_size:
                raise ValueError(
                    f"Batch size mismatch: P/A have batch size {matrix_batch_size}, "
                    f"but q/b have batch size {batch_size}. "
                    f"Use 1D P/A values for shared matrices across batch."
                )

        # Special case: batch_size=1 with 2D input
        if batch_size == 1:
            P_1d = P_values if P_values.dim() == 1 else P_values[0]
            A_1d = A_values if A_values.dim() == 1 else A_values[0]
            result = self._solve_single(
                P_1d.numpy(), A_1d.numpy(), q[0].numpy(), b[0].numpy(), **warm_np
            )
            # Convert to 2D output format
            return {
                "x": result["x"].unsqueeze(0),
                "z": result["z"].unsqueeze(0),
                "s": result["s"].unsqueeze(0),
                "status": result["status"].unsqueeze(0),
                "obj_val": result["obj_val"].unsqueeze(0),
                "obj_val_dual": result["obj_val_dual"].unsqueeze(0),
                "iterations": result["iterations"].unsqueeze(0),
                "r_prim": result["r_prim"].unsqueeze(0),
                "r_dual": result["r_dual"].unsqueeze(0),
                "solve_time": result["solve_time"],
            }

        # Use shared P/A optimization when matrices are 1D
        if P_values.dim() == 1:
            return self._solve_batch_shared(P_values, A_values, q, b, **warm_np)

        # Per-problem P/A (2D matrices)
        return self._solve_batch(P_values, A_values, q, b, **warm_np)

    def backward(
        self,
        dx: torch.Tensor,
        dz: Optional[torch.Tensor] = None,
        ds: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gradients via implicit differentiation.

        Args:
            dx: Gradient w.r.t. primal variables x (required)
            dz: Gradient w.r.t. dual variables z (optional, defaults to zeros)
            ds: Gradient w.r.t. slack variables s (optional, defaults to zeros)

        Returns:
            Tuple of (dP_values, dq, dA_values, db)
        """
        # Check if we have a cached solver for backward
        if self._last_solve_mode is None:
            raise RuntimeError("Must call solve() before backward()")

        # Validate dx is on CPU
        if not dx.device.type == "cpu":
            raise RuntimeError(
                f"Tensor 'dx' is on device '{dx.device}', but CPU solver requires CPU tensors. "
                f"Move tensors to CPU with .cpu() or use device='cuda'."
            )
        if dx.dtype != torch.float64:
            raise RuntimeError(
                f"Tensor 'dx' has dtype {dx.dtype}, but solver requires float64. "
                f"Convert with .double()."
            )

        # Use float64 CPU tensors
        dx = dx.double()

        # Detect single vs batched from input shape
        is_single = dx.dim() == 1

        # Handle optional dz and ds - default to zeros
        if dz is None:
            dz = (
                torch.zeros(self._m, dtype=torch.float64)
                if is_single
                else torch.zeros(dx.shape[0], self._m, dtype=torch.float64)
            )
        else:
            if not dz.device.type == "cpu":
                raise RuntimeError(f"Tensor 'dz' must be on CPU")
            dz = dz.double()

        if ds is None:
            ds = (
                torch.zeros(self._m, dtype=torch.float64)
                if is_single
                else torch.zeros(dx.shape[0], self._m, dtype=torch.float64)
            )
        else:
            if not ds.device.type == "cpu":
                raise RuntimeError(f"Tensor 'ds' must be on CPU")
            ds = ds.double()

        # Single problem: use DefaultSolver.backward_batch()
        # Use _last_solve_mode to determine which solver to use (not just existence of _default_solver)
        if self._last_solve_mode == "single" and self._default_solver is not None:
            # Extract 1D arrays for backward_batch
            if is_single:
                dx_np = dx.numpy()
                ds_np = ds.numpy()
                dz_np = dz.numpy()
            else:
                dx_np = dx[0].numpy()
                ds_np = ds[0].numpy()
                dz_np = dz[0].numpy()

            grad = self._default_solver.backward_batch(
                dx_np.tolist(), ds_np.tolist(), dz_np.tolist()
            )

            dP_values = torch.from_numpy(np.array(grad.dP_values))
            dA_values = torch.from_numpy(np.array(grad.dA_values))
            dq = torch.from_numpy(np.array(grad.dq))
            db = torch.from_numpy(np.array(grad.db))

            # If input was 2D (batch=1), output should also be 2D
            if not is_single:
                dP_values = dP_values.unsqueeze(0)
                dA_values = dA_values.unsqueeze(0)
                dq = dq.unsqueeze(0)
                db = db.unsqueeze(0)

            return dP_values, dq, dA_values, db

        # Batch mode: use batch_solver.backward() with numpy arrays
        if is_single:
            dx = dx.unsqueeze(0)
            dz = dz.unsqueeze(0)
            ds = ds.unsqueeze(0)

        batch_size = dx.shape[0]

        # Check solver is available
        if self._batch_solver is None:
            raise RuntimeError("No batch solver available for backward(). Call solve() first.")

        # Call backward with numpy arrays directly (no tolist() needed!)
        dx_np = dx.numpy()
        dz_np = dz.numpy()
        ds_np = ds.numpy()

        grad_result = self._batch_solver.backward(dx_np, dz_np, ds_np)

        # Convert results to torch tensors
        dP_values = torch.from_numpy(np.asarray(grad_result["dP_values"]))
        dA_values = torch.from_numpy(np.asarray(grad_result["dA_values"]))
        dq = torch.from_numpy(np.asarray(grad_result["dq"]))
        db = torch.from_numpy(np.asarray(grad_result["db"]))

        if is_single:
            dP_values = dP_values.squeeze(0)
            dA_values = dA_values.squeeze(0)
            dq = dq.squeeze(0)
            db = db.squeeze(0)

        return dP_values, dq, dA_values, db

    def backward_with_mode(
        self,
        dx: torch.Tensor,
        dz: Optional[torch.Tensor],
        ds: Optional[torch.Tensor],
        solve_mode: str,
        P_values: torch.Tensor,
        A_values: torch.Tensor,
        q: torch.Tensor,
        b: torch.Tensor,
        x: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
        state_rinv: Optional[torch.Tensor] = None,
        state_rinv_diag: Optional[torch.Tensor] = None,
        state_use_rinv_diag: Optional[torch.Tensor] = None,
        state_n_active: Optional[torch.Tensor] = None,
        state_ws: Optional[torch.Tensor] = None,
        state_sense: Optional[torch.Tensor] = None,
        state_lam_star: Optional[torch.Tensor] = None,
        z_x: Optional[torch.Tensor] = None,
        dz_x: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute gradients from explicitly provided problem data and solution.

        Uses backward_with_data which equilibrates internally and computes
        the adjoint KKT solve directly — no re-solve needed. This is safe
        for chained solves because each call uses only the provided data,
        not any mutable solver state.

        Args:
            dx: Gradient w.r.t. primal variables x
            dz: Gradient w.r.t. dual variables z (optional)
            ds: Gradient w.r.t. slack variables s (optional)
            solve_mode: 'single' or 'batch' - the mode used during forward
            P_values, A_values, q, b: Problem data from forward
            x, z, s: Solution from forward

        Returns:
            Tuple of (dP_values, dq, dA_values, db)
        """
        # Validate and default dz/ds
        dx = dx.double()
        is_single = dx.dim() == 1

        if dz is None:
            dz = (
                torch.zeros(self._m, dtype=torch.float64)
                if is_single
                else torch.zeros(dx.shape[0], self._m, dtype=torch.float64)
            )
        else:
            dz = dz.double()
        if ds is None:
            ds = (
                torch.zeros(self._m, dtype=torch.float64)
                if is_single
                else torch.zeros(dx.shape[0], self._m, dtype=torch.float64)
            )
        else:
            ds = ds.double()

        # Ensure all inputs are 2D (batch, dim) for the flat API
        if is_single:
            dx = dx.unsqueeze(0)
            dz = dz.unsqueeze(0)
            ds = ds.unsqueeze(0)
            P_values = P_values.unsqueeze(0) if P_values.dim() == 1 else P_values
            A_values = A_values.unsqueeze(0) if A_values.dim() == 1 else A_values
            q = q.unsqueeze(0) if q.dim() == 1 else q
            b = b.unsqueeze(0) if b.dim() == 1 else b
            x = x.unsqueeze(0) if x.dim() == 1 else x
            z = z.unsqueeze(0) if z.dim() == 1 else z
            s = s.unsqueeze(0) if s.dim() == 1 else s

        batch_size = dx.shape[0]

        # Ensure batch solver exists (it should from the forward pass)
        if self._batch_solver is None or self._current_batch_size != batch_size:
            if self._current_batch_size is not None and not getattr(
                self, "_warned_backward_batch_change", False
            ):
                warnings.warn(
                    f"backward_with_mode: batch size {batch_size} differs from "
                    f"solver batch size {self._current_batch_size}. Recreating "
                    f"CPU solver for the backward pass, which may degrade "
                    f"performance. This is expected when using torch.func.jacrev "
                    f"or vmap over backward. To avoid this overhead, construct "
                    f"the solver with batch_size={batch_size}.",
                    UserWarning,
                    stacklevel=3,
                )
                self._warned_backward_batch_change = True
            self._create_batch_solver(batch_size)

        # Use backward_with_data_flat if available (CompiledSolver / IPM),
        # otherwise fall back to re-solve (ActiveSetSolver).
        #
        # Additionally fall back to re-solve whenever the saved backward
        # state is missing — active-set's backward_with_data_flat requires
        # the factorization/active-set snapshot, and if the forward solve
        # was run with enable_grad=False or via an external caller that
        # doesn't propagate the state (e.g. cvxpylayers), the flat state
        # tensors are empty placeholders that would fail Rust-side dim
        # checks.
        active_set_state_missing = self._use_active_set and (
            state_ws is None or state_ws.numel() == 0
        )
        if not hasattr(self._batch_solver, "backward_with_data_flat") or active_set_state_missing:
            original_verbose = self._settings.verbose
            self._settings.verbose = False
            self.setup(P_values, A_values)
            self.solve(q, b)
            self._settings.verbose = original_verbose
            return self.backward(dx, dz, ds)

        # Expand shared P/A to batch size if needed
        if P_values.shape[0] == 1 and batch_size > 1:
            P_values = P_values.expand(batch_size, -1).contiguous()
        if A_values.shape[0] == 1 and batch_size > 1:
            A_values = A_values.expand(batch_size, -1).contiguous()

        if state_rinv is None:
            state_rinv = torch.empty(batch_size, 0, dtype=torch.float64)
        if state_rinv_diag is None:
            state_rinv_diag = torch.empty(batch_size, 0, dtype=torch.float64)
        if state_use_rinv_diag is None:
            state_use_rinv_diag = torch.zeros(batch_size, dtype=torch.int32)
        if state_n_active is None:
            state_n_active = torch.zeros(batch_size, dtype=torch.int32)
        if state_ws is None:
            state_ws = torch.empty(batch_size, 0, dtype=torch.int32)
        if state_sense is None:
            state_sense = torch.empty(batch_size, 0, dtype=torch.int32)
        if state_lam_star is None:
            state_lam_star = torch.empty(batch_size, 0, dtype=torch.float64)

        # Direct: pass z_x (saved direct dual) and dz_x (upstream
        # gradient on Solution.z_x) to backward_with_data_flat as kwargs.
        # Only forwarded when set; older Rust solvers / ActiveSet don't
        # accept the kwargs.
        xcone_kwargs = {}
        if z_x is not None:
            z_x_t = z_x.double()
            if is_single and z_x_t.dim() == 1:
                z_x_t = z_x_t.unsqueeze(0)
            if z_x_t.numel() > 0:
                xcone_kwargs["z_x_flat"] = z_x_t.contiguous().numpy().ravel()
        if dz_x is not None:
            dz_x_t = dz_x.double()
            if is_single and dz_x_t.dim() == 1:
                dz_x_t = dz_x_t.unsqueeze(0)
            if dz_x_t.numel() > 0:
                xcone_kwargs["dz_x_flat"] = dz_x_t.contiguous().numpy().ravel()

        if self._use_active_set:
            backward_state = self._batch_solver._solver._make_backward_state_from_flat(
                state_rinv.contiguous().double().numpy().ravel(),
                state_rinv_diag.contiguous().double().numpy().ravel(),
                state_use_rinv_diag.contiguous().to(dtype=torch.int32).numpy().ravel(),
                state_n_active.contiguous().to(dtype=torch.int32).numpy().ravel(),
                state_ws.contiguous().to(dtype=torch.int32).numpy().ravel(),
                state_sense.contiguous().to(dtype=torch.int32).numpy().ravel(),
                state_lam_star.contiguous().double().numpy().ravel(),
            )
            grad_result = self._batch_solver.backward_with_data_flat(
                dx.contiguous().numpy().ravel(),
                ds.contiguous().numpy().ravel(),
                dz.contiguous().numpy().ravel(),
                P_values.contiguous().double().numpy().ravel(),
                A_values.contiguous().double().numpy().ravel(),
                q.contiguous().double().numpy().ravel(),
                b.contiguous().double().numpy().ravel(),
                x.contiguous().double().numpy().ravel(),
                z.contiguous().double().numpy().ravel(),
                s.contiguous().double().numpy().ravel(),
                backward_state,
                batch_size,
                **xcone_kwargs,
            )
        else:
            # Compiled/IPM path restores its own explicit backward state.
            grad_result = self._batch_solver.backward_with_data_flat(
                dx.contiguous().numpy().ravel(),
                ds.contiguous().numpy().ravel(),
                dz.contiguous().numpy().ravel(),
                P_values.contiguous().double().numpy().ravel(),
                A_values.contiguous().double().numpy().ravel(),
                q.contiguous().double().numpy().ravel(),
                b.contiguous().double().numpy().ravel(),
                x.contiguous().double().numpy().ravel(),
                z.contiguous().double().numpy().ravel(),
                s.contiguous().double().numpy().ravel(),
                batch_size,
                **xcone_kwargs,
            )

        dP_values = torch.from_numpy(np.asarray(grad_result["dP_values"]))
        dq_out = torch.from_numpy(np.asarray(grad_result["dq"]))
        dA_values = torch.from_numpy(np.asarray(grad_result["dA_values"]))
        db_out = torch.from_numpy(np.asarray(grad_result["db"]))

        if is_single:
            dP_values = dP_values.squeeze(0) if dP_values.dim() > 1 else dP_values
            dq_out = dq_out.squeeze(0) if dq_out.dim() > 1 else dq_out
            dA_values = dA_values.squeeze(0) if dA_values.dim() > 1 else dA_values
            db_out = db_out.squeeze(0) if db_out.dim() > 1 else db_out
        else:
            # Rust may return flat 1D arrays for batch_size=1; reshape to match input shapes
            if dP_values.dim() == 1 and P_values.dim() == 2:
                dP_values = dP_values.reshape(P_values.shape)
            if dq_out.dim() == 1 and q.dim() == 2:
                dq_out = dq_out.reshape(q.shape)
            if dA_values.dim() == 1 and A_values.dim() == 2:
                dA_values = dA_values.reshape(A_values.shape)
            if db_out.dim() == 1 and b.dim() == 2:
                db_out = db_out.reshape(b.shape)

        return dP_values, dq_out, dA_values, db_out

    @property
    def n(self) -> int:
        return self._n

    @property
    def m(self) -> int:
        return self._m

    @property
    def batch_size(self) -> Optional[int]:
        """Current or expected batch size.

        Returns the current batch size if solve() has been called,
        otherwise returns the expected batch size hint from constructor.
        """
        return self._current_batch_size or self._expected_batch_size

    @property
    def batch_size_change_count(self) -> int:
        """Number of times batch size has changed."""
        return self._batch_size_change_count

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def nnzP(self) -> int:
        return self._nnz_P

    @property
    def nnzA(self) -> int:
        return self._nnz_A

    def reset(self):
        """Reset solver state (clears the cache)."""
        self._default_solver = None
        self._batch_solver = None
        self._current_batch_size = None
        self._cached_shared_P = None
        self._cached_shared_A = None
        # Don't reset batch_size_change_count - keep tracking across resets

    def setup_grad(self, batch_size: Optional[int] = None):
        """Setup for gradient computation (backward pass).

        Pre-allocates memory and enables gradient computation. Required before
        calling backward().

        Args:
            batch_size: Optional batch size for pre-allocation.
        """
        effective_batch = batch_size or self._expected_batch_size or 1

        # Enable gradient computation
        if not self._enable_grad:
            self._enable_grad = True

        # Pre-allocate numpy buffers for gradient computation
        # These avoid repeated allocation in backward()
        self._grad_dx_buffer = np.zeros((effective_batch, self._n), dtype=np.float64)
        self._grad_dz_buffer = np.zeros((effective_batch, self._m), dtype=np.float64)
        self._grad_ds_buffer = np.zeros((effective_batch, self._m), dtype=np.float64)

        self._grad_initialized = True

    @property
    def grad_initialized(self) -> bool:
        """Whether setup_grad() has been called."""
        return self._grad_initialized

    def get_dimensions(self) -> dict:
        return {
            "n": self._n,
            "m": self._m,
            "batch_size": self._current_batch_size or self._expected_batch_size or 0,
            "nnzP": self._nnz_P,
            "nnzA": self._nnz_A,
        }
