"""PyTorch integration for Moreau CUDA solver using tensor.data_ptr() for zero-copy."""

from typing import Dict, Optional, Tuple, Any
import warnings

import numpy as np
import torch

try:
    from ._moreau_cuda import Solver as _RawCudaSolver
    from ._moreau_cuda import Cones as _CudaCones
    from ._moreau_cuda import Settings as _CudaSettings
except ImportError:
    try:
        from _moreau_cuda import Solver as _RawCudaSolver
        from _moreau_cuda import Cones as _CudaCones
        from _moreau_cuda import Settings as _CudaSettings
    except ImportError as e:
        raise ImportError(
            "moreau_cuda C++ extension not found. Build with: pip install moreau-cuda"
        ) from e

from moreau._types import Cones, Settings

# For scipy sparse matrices in TorchSolver
try:
    from scipy import sparse as scipy_sparse
except ImportError:
    scipy_sparse = None


class _CudaSolveFunction(torch.autograd.Function):
    """Internal autograd Function for differentiable CUDA solve.

    This enables torch.autograd.gradcheck to work by properly connecting
    the computational graph.
    """

    @staticmethod
    def forward(ctx, solver, P_values, A_values, q, b):
        """Forward pass: solve the optimization problem."""
        # Ensure 2D for internal solver
        is_single_q = q.dim() == 1
        is_single_b = b.dim() == 1

        q_2d = q.unsqueeze(0) if is_single_q else q
        b_2d = b.unsqueeze(0) if is_single_b else b

        batch_size = q_2d.shape[0]

        # Handle P and A - ensure 2D and matching batch size
        P_2d = P_values.unsqueeze(0) if P_values.dim() == 1 else P_values
        A_2d = A_values.unsqueeze(0) if A_values.dim() == 1 else A_values

        # Broadcast shared matrices to batch size if needed
        if P_2d.shape[0] == 1 and batch_size > 1:
            P_2d = P_2d.expand(batch_size, -1).contiguous()
        if A_2d.shape[0] == 1 and batch_size > 1:
            A_2d = A_2d.expand(batch_size, -1).contiguous()

        device = q.device
        n, m = solver._n, solver._m

        # Allocate output tensors on GPU
        x = torch.empty((batch_size, n), dtype=torch.float64, device=device)
        z = torch.empty((batch_size, m), dtype=torch.float64, device=device)
        s = torch.empty((batch_size, m), dtype=torch.float64, device=device)
        status = torch.empty(batch_size, dtype=torch.int32, device=device)
        obj_val = torch.empty(batch_size, dtype=torch.float64, device=device)

        # Direct cone duals output (empty tensor when no direct cones).
        # Read from the precomputed `_total_x_dim` instead of the nanobind
        # `dir_cones` property — dynamo can't trace nb_method accessors.
        total_xn = solver._total_x_dim
        if total_xn > 0:
            z_x = torch.empty((batch_size, total_xn), dtype=torch.float64, device=device)
            z_x_out_ptr = z_x.data_ptr()
        else:
            z_x = torch.zeros((batch_size, 0), dtype=torch.float64, device=device)
            z_x_out_ptr = 0

        # Check for pending warm start (non-differentiable side data).
        # `pending_warm` is a 3-tuple (warm_x, warm_z, warm_s) when there are
        # no direct cones, or a 4-tuple (warm_x, warm_z, warm_s, warm_z_x)
        # when direct cones are present.
        pending_warm = getattr(solver, "_pending_warm_start", None)
        if pending_warm is not None:
            if len(pending_warm) == 4:
                warm_x, warm_z, warm_s, warm_z_x = pending_warm
                warm_z_x_ptr = (
                    warm_z_x.data_ptr() if warm_z_x is not None and warm_z_x.numel() > 0 else 0
                )
            else:
                warm_x, warm_z, warm_s = pending_warm
                warm_z_x_ptr = 0
            solver._impl.solve_warm_start_to_device_pointers(
                P_2d.data_ptr(),
                A_2d.data_ptr(),
                q_2d.data_ptr(),
                b_2d.data_ptr(),
                warm_x.data_ptr(),
                warm_z.data_ptr(),
                warm_s.data_ptr(),
                x.data_ptr(),
                z.data_ptr(),
                s.data_ptr(),
                status.data_ptr(),
                obj_val.data_ptr(),
                warm_z_x_ptr,
                z_x_out_ptr,
            )
        else:
            # Zero-copy solve using device pointers (with optional z_x out).
            solver._impl.solve_to_device_pointers(
                P_2d.data_ptr(),
                A_2d.data_ptr(),
                q_2d.data_ptr(),
                b_2d.data_ptr(),
                x.data_ptr(),
                z.data_ptr(),
                s.data_ptr(),
                status.data_ptr(),
                obj_val.data_ptr(),
                z_x_out_ptr,
            )

        # Save for backward - problem data and solution (incl. z_x).
        ctx.solver = solver
        ctx.is_single = is_single_q
        ctx.batch_size = batch_size
        ctx.save_for_backward(
            P_values,
            A_values,
            q,
            b,
            x.clone(),
            z.clone(),
            s.clone(),
            z_x.clone(),
        )

        return x, z, s, status, obj_val, z_x

    @staticmethod
    def backward(ctx, dx, dz, ds, dstatus, dobj, dz_x):
        """Backward pass: compute gradients via implicit differentiation."""
        P_values, A_values, q, b, x, z, s, z_x = ctx.saved_tensors
        solver = ctx.solver
        batch_size = ctx.batch_size
        device = dx.device

        # Ensure 2D for saved tensors
        P_2d = P_values if P_values.dim() == 2 else P_values.unsqueeze(0)
        A_2d = A_values if A_values.dim() == 2 else A_values.unsqueeze(0)
        q_2d = q if q.dim() == 2 else q.unsqueeze(0)
        b_2d = b if b.dim() == 2 else b.unsqueeze(0)
        x_2d = x if x.dim() == 2 else x.unsqueeze(0)
        z_2d = z if z.dim() == 2 else z.unsqueeze(0)
        s_2d = s if s.dim() == 2 else s.unsqueeze(0)

        # Direct dual: forward saved z_x; only meaningful when non-empty.
        # Bind the contiguous tensor to a name: `t.contiguous().data_ptr()`
        # on a non-contiguous `t` allocates a fresh copy whose only
        # reference is the temporary — CPython frees it the moment the
        # statement ends, leaving `z_x_ptr` dangling before the C++ call
        # below dereferences it (use-after-free of device memory).
        z_x_2d = z_x if z_x.dim() == 2 else z_x.unsqueeze(0)
        z_x_c = z_x_2d.contiguous()
        z_x_ptr = z_x_c.data_ptr() if z_x_c.numel() > 0 else 0

        # Restore full backward state: problem data, equilibration, and solution.
        # This reloads P/A/q/b, re-equilibrates, copies solution to DiffState,
        # and syncs equilibration factors — no re-solve needed.
        solver._impl.load_backward_state_from_device_pointers(
            P_2d.contiguous().data_ptr(),
            A_2d.contiguous().data_ptr(),
            q_2d.contiguous().data_ptr(),
            b_2d.contiguous().data_ptr(),
            x_2d.contiguous().data_ptr(),
            z_2d.contiguous().data_ptr(),
            s_2d.contiguous().data_ptr(),
            z_x_ptr,
        )

        # Ensure 2D
        dx_2d = dx if dx.dim() == 2 else dx.unsqueeze(0)
        dz_2d = dz if dz.dim() == 2 else dz.unsqueeze(0)
        ds_2d = ds if ds.dim() == 2 else ds.unsqueeze(0)

        # Direct upstream dz_x: pass through to the zero-copy backward
        # binding (added in #7); the C++ side dispatches via
        # `backward_with_dz_x`. dz_x_ptr=0 → skip the dz_x branch.
        dz_x_has_data = (
            dz_x is not None
            and dz_x.numel() > 0
            and not (dz_x.numel() == z_x_2d.numel() and torch.all(dz_x == 0))
        )

        # Allocate output gradients
        n, m = solver._n, solver._m
        nnzP, nnzA = solver._nnzP, solver._nnzA

        dP_values = torch.empty(batch_size, nnzP, dtype=torch.float64, device=device)
        dq = torch.empty(batch_size, n, dtype=torch.float64, device=device)
        dA_values = torch.empty(batch_size, nnzA, dtype=torch.float64, device=device)
        db = torch.empty(batch_size, m, dtype=torch.float64, device=device)

        if dz_x_has_data:
            dz_x_2d = dz_x if dz_x.dim() == 2 else dz_x.unsqueeze(0)
            # `dz_x` is an autograd upstream gradient and is frequently
            # non-contiguous (broadcast / sliced views), so `.contiguous()`
            # here usually does allocate a copy. Keep a reference alive
            # (`dz_x_c`) through the C++ call — otherwise dz_x_ptr dangles.
            dz_x_c = dz_x_2d.contiguous()
            dz_x_ptr = dz_x_c.data_ptr()
        else:
            dz_x_ptr = 0

        # Zero-copy device-pointer backward (with optional dz_x).
        solver._impl.backward_to_device_pointers(
            dx_2d.contiguous().data_ptr(),
            dz_2d.contiguous().data_ptr(),
            ds_2d.contiguous().data_ptr(),
            dP_values.data_ptr(),
            dq.data_ptr(),
            dA_values.data_ptr(),
            db.data_ptr(),
            dz_x_ptr,
        )

        # Handle dimensionality matching inputs
        if ctx.is_single:
            dP_values = dP_values.squeeze(0)
            dq = dq.squeeze(0)
            dA_values = dA_values.squeeze(0)
            db = db.squeeze(0)

        # Clear references to break reference cycle that prevents garbage collection
        # The autograd context holds a reference to solver, preventing it from being freed
        ctx.solver = None

        # Return gradients in same order as forward inputs:
        # solver (None), P_values, A_values, q, b
        return None, dP_values, dA_values, dq, db


def _cones_to_cuda(cones: Cones) -> _CudaCones:
    """Convert unified Cones to CUDA cones format."""
    cuda_cones = _CudaCones()
    cuda_cones.num_zero_cones = cones.num_zero_cones
    cuda_cones.num_nonneg_cones = cones.num_nonneg_cones
    cuda_cones.soc_cone_dims = list(cones.so_cone_dims)
    cuda_cones.num_exp_cones = cones.num_exp_cones
    cuda_cones.power_alphas = list(cones.power_alphas) if cones.power_alphas else []
    cuda_cones.num_power_cones = len(cuda_cones.power_alphas)
    if hasattr(cones, "psd_dims") and cones.psd_dims:
        cuda_cones.psd_cone_dims = list(cones.psd_dims)
    gen_params = getattr(cones, "gen_power_cone_params", [])
    if gen_params:
        all_alphas = []
        dim1s = []
        dim2s = []
        for alphas, dim2 in gen_params:
            all_alphas.extend(alphas)
            dim1s.append(len(alphas))
            dim2s.append(dim2)
        cuda_cones.gen_power_alphas = all_alphas
        cuda_cones.gen_power_dim1s = dim1s
        cuda_cones.gen_power_dim2s = dim2s
        cuda_cones.num_gen_power_cones = len(gen_params)
    else:
        cuda_cones.num_gen_power_cones = 0
    # Direct cones: thread through to CUDA backend so direct routing
    # fires. Each DirectConeSpec is mapped to the CUDA SupportedXConeT, threading
    # per-kind parameters (alpha for Power, alphas/dim2 for GenPower, psd_k
    # for PSD) so the CUDA backend sees the full descriptor.
    x_specs = getattr(cones, "dir_cones", None) or []
    if x_specs:
        from ._moreau_cuda import (
            SupportedXConeT as _CudaSupportedXConeT,
            XConeKind as _CudaXConeKind,
        )

        kind_map = {
            "nonneg": _CudaXConeKind.Nonneg,
            "soc": _CudaXConeKind.SOC,
            "exp": _CudaXConeKind.Exp,
            "power": _CudaXConeKind.Power,
            "gen_power": _CudaXConeKind.GenPower,
            "psd_triangle": _CudaXConeKind.PSD,
        }
        cuda_xs = []
        for spec in x_specs:
            kind_str = (getattr(spec, "kind", "") or "").lower()
            if kind_str not in kind_map:
                raise ValueError(f"Unknown DirectConeSpec.kind: {kind_str!r}")
            kwargs = dict(
                kind=kind_map[kind_str],
                indices=list(spec.indices),
            )
            if kind_str == "power":
                alpha = getattr(spec, "alpha", None)
                if alpha is None:
                    raise ValueError("DirectConeSpec(kind='power') requires alpha")
                kwargs["power_alpha"] = float(alpha)
            elif kind_str == "gen_power":
                alphas = getattr(spec, "alphas", None)
                dim2 = getattr(spec, "dim2", None)
                if alphas is None or dim2 is None:
                    raise ValueError("DirectConeSpec(kind='gen_power') requires alphas and dim2")
                kwargs["gen_power_alphas"] = [float(a) for a in alphas]
                kwargs["gen_power_dim2"] = int(dim2)
            elif kind_str == "psd_triangle":
                psd_k = getattr(spec, "psd_k", None)
                if psd_k is None:
                    raise ValueError("DirectConeSpec(kind='psd_triangle') requires psd_k")
                kwargs["psd_k"] = int(psd_k)
            cuda_xs.append(_CudaSupportedXConeT(**kwargs))
        cuda_cones.dir_cones = cuda_xs
    return cuda_cones


def _settings_to_cuda(settings: Settings) -> _CudaSettings:
    """Convert unified Settings to CUDA settings format.

    Delegates to the unified converter in `moreau_cuda` to avoid drift between
    the two paths — historically this function had its own (incomplete) copy
    of the conversion and silently dropped reduced_tol_*, max_lu_nnz, yolo,
    yolo_num_iters, and device_id.
    """
    from moreau._types import SolverType as _PySolverType

    if settings.solver == _PySolverType.ACTIVE_SET:
        raise ValueError(
            "solver='active_set' is not supported through the CUDA torch wrapper. "
            "Use device='cpu' or solver='ipm' for CUDA tensors."
        )

    # Ensure ipm_settings exists (may be None for active-set solver)
    if settings.ipm_settings is None:
        from moreau._types import IPMSettings

        settings = settings.model_copy(deep=True)
        settings.ipm_settings = IPMSettings()

    # Defer to the unified converter (single source of truth for the full
    # IPMSettings -> _CudaSettings mapping).
    from moreau_cuda import _settings_to_cuda as _unified_settings_to_cuda

    return _unified_settings_to_cuda(settings)


def _validate_cuda_tensor(t: torch.Tensor, name: str) -> None:
    """Validate that a tensor is on CUDA and float64."""
    if not t.is_cuda:
        raise RuntimeError(f"{name} must be a CUDA tensor, got device={t.device}")
    if t.dtype != torch.float64:
        raise RuntimeError(f"{name} must be float64, got {t.dtype}")


def _validate_tensor_size(t: torch.Tensor, name: str, expected_size: int, dim: int = -1) -> None:
    """Validate that tensor has expected size in given dimension."""
    actual_size = t.shape[dim]
    if actual_size != expected_size:
        raise RuntimeError(f"{name} dimension {dim} must be {expected_size}, got {actual_size}")


class TorchSolver:
    """Single-problem CUDA PyTorch solver matching the CPU TorchSolver API.

    Takes all problem data in the constructor, then call solve().

    Args:
        P: Quadratic cost matrix (scipy.sparse CSR, must be full symmetric)
        q: Linear cost tensor (n,), CUDA float64
        A: Constraint matrix (scipy.sparse CSR)
        b: Constraint RHS tensor (m,), CUDA float64
        cones: Cone specification
        settings: Optional solver settings
    """

    def __init__(
        self,
        P,  # scipy.sparse matrix
        q: torch.Tensor,
        A,  # scipy.sparse matrix
        b: torch.Tensor,
        cones: Cones,
        settings: Optional[Settings] = None,
    ):
        if scipy_sparse is None:
            raise ImportError("scipy is required for TorchSolver")

        # Validate torch tensors
        _validate_cuda_tensor(q, "q")
        _validate_cuda_tensor(b, "b")

        # Convert sparse matrices to CSR
        P_csr = scipy_sparse.csr_array(P, dtype=np.float64)
        A_csr = scipy_sparse.csr_array(A, dtype=np.float64)

        # Store problem dimensions
        self._n = P_csr.shape[0]
        self._m = A_csr.shape[0]
        self._device = q.device

        # Validate dimensions
        if q.shape[0] != self._n:
            raise ValueError(f"q has length {q.shape[0]}, expected {self._n}")
        if b.shape[0] != self._m:
            raise ValueError(f"b has length {b.shape[0]}, expected {self._m}")

        # Store problem data (keep as tensors for potential gradients)
        self._q = q.contiguous()
        self._b = b.contiguous()

        # Convert cones and settings
        if settings is None:
            settings = Settings()
        self._settings = settings
        self._cones = cones
        self._cuda_cones = _cones_to_cuda(cones)
        self._cuda_settings = _settings_to_cuda(settings)
        # Cache the direct cone total dimension so the autograd path
        # doesn't have to introspect nanobind properties at trace time —
        # `torch.compile`'s dynamo cannot inline `nb_method` accessors and
        # raises `AssertionError: expected FunctionType found nb_method`
        # on `getattr(self._cuda_cones, 'dir_cones', [])` calls inside
        # `_CudaSolveFunction.forward`. Pre-summing here keeps the hot
        # path numerical-only.
        self._total_x_dim = sum(
            len(xc.indices) for xc in getattr(self._cuda_cones, "dir_cones", [])
        )

        # Extract CSR structure and convert to GPU tensors
        self._P_row_offsets = np.asarray(P_csr.indptr, dtype=np.int64)
        self._P_col_indices = np.asarray(P_csr.indices, dtype=np.int64)
        self._P_values = torch.tensor(P_csr.data, dtype=torch.float64, device=self._device)
        self._A_row_offsets = np.asarray(A_csr.indptr, dtype=np.int64)
        self._A_col_indices = np.asarray(A_csr.indices, dtype=np.int64)
        self._A_values = torch.tensor(A_csr.data, dtype=torch.float64, device=self._device)

        self._nnzP = len(self._P_col_indices)
        self._nnzA = len(self._A_col_indices)

        # Create underlying CUDA solver with batch_size=1
        self._impl = _RawCudaSolver(
            self._n,
            self._m,
            1,  # batch_size=1
            self._P_row_offsets,
            self._P_col_indices,
            self._A_row_offsets,
            self._A_col_indices,
            self._cuda_cones,
            self._cuda_settings,
            False,  # enable_grad=False for single solver
        )

        # Store solution after solve
        self._solution: Optional[Dict[str, Any]] = None

    def solve(self) -> Dict[str, Any]:
        """Solve the optimization problem.

        Returns:
            Dict with keys: x, z, s, status, obj_val, solve_time
            All tensor outputs are CUDA float64.
        """
        device = self._device

        # Expand to 2D for CUDA solver (batch_size=1)
        P_2d = self._P_values.unsqueeze(0)
        A_2d = self._A_values.unsqueeze(0)
        q_2d = self._q.unsqueeze(0)
        b_2d = self._b.unsqueeze(0)

        # Allocate output tensors on GPU
        x = torch.empty((1, self._n), dtype=torch.float64, device=device)
        z = torch.empty((1, self._m), dtype=torch.float64, device=device)
        s = torch.empty((1, self._m), dtype=torch.float64, device=device)
        status = torch.empty(1, dtype=torch.int32, device=device)
        obj_val = torch.empty(1, dtype=torch.float64, device=device)

        # Zero-copy solve using device pointers
        self._impl.solve_to_device_pointers(
            P_2d.data_ptr(),
            A_2d.data_ptr(),
            q_2d.data_ptr(),
            b_2d.data_ptr(),
            x.data_ptr(),
            z.data_ptr(),
            s.data_ptr(),
            status.data_ptr(),
            obj_val.data_ptr(),
        )

        # Convert result to single-problem format
        self._solution = {
            "x": x.squeeze(0),
            "z": z.squeeze(0),
            "s": s.squeeze(0),
            "status": int(status[0].item()),
            "obj_val": obj_val[0].item(),
            "iterations": 0,  # Placeholder
            "solve_time": 0.0,  # Placeholder
        }
        return self._solution

    @property
    def n(self) -> int:
        """Number of primal variables."""
        return self._n

    @property
    def m(self) -> int:
        """Number of constraints."""
        return self._m

    @property
    def solution(self) -> Optional[Dict[str, Any]]:
        """Last solution, or None if solve() not called yet."""
        return self._solution


class TorchCompiledSolver:
    """CUDA PyTorch batch solver with three-step API and autograd support.

    Three-step API pattern:
    1. Construct solver with structure (dimensions, sparsity patterns, cones)
    2. setup(P_values, A_values) - Set matrix values
    3. solve(q, b) - Solve with linear cost and constraint RHS

    Args:
        n: Number of primal variables
        m: Number of constraints
        P_row_offsets: CSR row pointers for P matrix (torch.Tensor, can be CPU)
        P_col_indices: CSR column indices for P matrix (torch.Tensor, can be CPU)
        A_row_offsets: CSR row pointers for A matrix (torch.Tensor, can be CPU)
        A_col_indices: CSR column indices for A matrix (torch.Tensor, can be CPU)
        cones: Cone specification (moreau.Cones object from _types)
        settings: Optional solver settings (moreau.Settings object)
        batch_size: Optional batch size for eager initialization
        enable_grad: If True, enable gradient computation
    """

    _BATCH_CHANGE_WARN_THRESHOLD = 5

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
        *,  # Force remaining args to be keyword-only
        batch_size: Optional[int] = None,
        enable_grad: bool = False,
        b_sparsity_pattern=None,
    ):
        self._n = n
        self._m = m
        self._expected_batch_size = batch_size
        self._enable_grad = enable_grad
        self._b_sparsity_pattern = (
            list(b_sparsity_pattern) if b_sparsity_pattern is not None else None
        )

        # Convert structure to numpy for C++ solver
        self._P_row_offsets = P_row_offsets.cpu().numpy().astype("int64")
        self._P_col_indices = P_col_indices.cpu().numpy().astype("int64")
        self._A_row_offsets = A_row_offsets.cpu().numpy().astype("int64")
        self._A_col_indices = A_col_indices.cpu().numpy().astype("int64")

        self._nnzP = len(self._P_col_indices)
        self._nnzA = len(self._A_col_indices)

        # Convert cones and settings to CUDA format
        if settings is None:
            settings = Settings()
        self._settings = settings
        self._cones = cones
        self._cuda_cones = _cones_to_cuda(cones)
        self._cuda_settings = _settings_to_cuda(settings)
        # See `_CudaSolveFunction.forward` — cache total direct dim so
        # dynamo doesn't trace into nanobind property accessors.
        self._total_x_dim = sum(
            len(xc.indices) for xc in getattr(self._cuda_cones, "dir_cones", [])
        )

        # Stored matrix values (set via setup)
        self._P_values: Optional[torch.Tensor] = None
        self._A_values: Optional[torch.Tensor] = None

        # Solver instance (created on demand based on batch size)
        self._impl: Optional[_RawCudaSolver] = None
        self._current_batch_size: Optional[int] = None

        # Track last solve mode for backward
        self._last_solve_mode: Optional[str] = None

        # Batch size change tracking for warnings
        self._batch_size_change_count = 0
        self._warned_about_batch_changes = False

        # Gradient setup state
        self._grad_initialized = False

        # Eager initialization if batch_size is provided
        if batch_size is not None:
            self._create_solver(batch_size)

    def _create_solver(self, batch_size: int) -> None:
        """Create or recreate the raw CUDA solver for given batch size."""
        self._impl = _RawCudaSolver(
            self._n,
            self._m,
            batch_size,
            self._P_row_offsets,
            self._P_col_indices,
            self._A_row_offsets,
            self._A_col_indices,
            self._cuda_cones,
            self._cuda_settings,
            self._enable_grad,  # Pass enable_grad to C++ solver
            b_sparsity_pattern=self._b_sparsity_pattern,
        )
        self._current_batch_size = batch_size

    def setup(self, P_values: torch.Tensor, A_values: torch.Tensor) -> None:
        """Set P and A matrix values.

        Must be called before solve(). Can be called multiple times to update
        values for repeated solves with the same structure.

        Args:
            P_values: P matrix values, shape (nnzP,) or (batch, nnzP), CUDA float64
            A_values: A matrix values, shape (nnzA,) or (batch, nnzA), CUDA float64
        """
        _validate_cuda_tensor(P_values, "P_values")
        _validate_cuda_tensor(A_values, "A_values")

        # Validate sizes
        _validate_tensor_size(P_values, "P_values", self._nnzP, dim=-1)
        _validate_tensor_size(A_values, "A_values", self._nnzA, dim=-1)

        self._P_values = P_values.contiguous()
        self._A_values = A_values.contiguous()

    # The native solver consumes raw device pointers and mutates solver state;
    # Dynamo must treat this boundary as opaque rather than trace nanobind calls.
    @torch.compiler.disable
    def solve(
        self,
        q: torch.Tensor,
        b: torch.Tensor,
        warm_x: Optional[torch.Tensor] = None,
        warm_z: Optional[torch.Tensor] = None,
        warm_s: Optional[torch.Tensor] = None,
        warm_z_x: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Solve conic optimization problem(s).

        Requires setup() to be called first.

        Args:
            q: Linear cost, shape (n,) or (batch, n), CUDA float64
            b: Constraint RHS, shape (m,) or (batch, m), CUDA float64
            warm_x: Optional warm start primal, shape (batch, n), CUDA float64
            warm_z: Optional warm start dual, shape (batch, m), CUDA float64
            warm_s: Optional warm start slack, shape (batch, m), CUDA float64
            warm_z_x: Optional direct cone dual warm start,
                shape (batch, sum |J|), CUDA float64. Required to be non-None
                only when the problem has direct cones; ignored otherwise.

        Returns:
            Dict with keys: x, z, s, status, obj_val, iterations, solve_time
            All tensor outputs are CUDA float64 with same batch dimension as input.
        """
        if self._P_values is None or self._A_values is None:
            raise RuntimeError("setup() must be called before solve()")

        _validate_cuda_tensor(q, "q")
        _validate_cuda_tensor(b, "b")

        # Validate sizes
        _validate_tensor_size(q, "q", self._n, dim=-1)
        _validate_tensor_size(b, "b", self._m, dim=-1)

        # Detect single vs batched from input shape
        is_single = q.dim() == 1
        self._last_solve_mode = "single" if is_single else "batch"

        q = q.contiguous()
        b = b.contiguous()

        # Determine batch size for solver initialization
        batch_size = 1 if is_single else q.shape[0]

        # Create/recreate solver if needed
        if self._impl is None or self._current_batch_size != batch_size:
            if self._current_batch_size is not None:
                self._batch_size_change_count += 1
                if (
                    self._batch_size_change_count >= self._BATCH_CHANGE_WARN_THRESHOLD
                    and not self._warned_about_batch_changes
                ):
                    warnings.warn(
                        f"Batch size has changed {self._batch_size_change_count} times. "
                        f"Frequent batch size changes are slow because the CUDA solver must be recreated. "
                        f"Consider using a fixed batch size or padding batches to a consistent size.",
                        UserWarning,
                        stacklevel=2,
                    )
                    self._warned_about_batch_changes = True
            self._create_solver(batch_size)

        device = q.device

        # Use autograd function when enable_grad is True, grad is enabled, and any input requires grad.
        # The torch.is_grad_enabled() check is critical: when backward_with_mode() re-solves inside
        # torch.no_grad() to restore state, we must NOT enter the autograd path even if inputs
        # have requires_grad=True (e.g., saved tensors from ctx.save_for_backward preserve their
        # requires_grad flag). Without this check, non-leaf tensors would cause nested autograd
        # contexts that fail with CUDA errors.
        if (
            self._enable_grad
            and torch.is_grad_enabled()
            and (
                q.requires_grad
                or b.requires_grad
                or self._P_values.requires_grad
                or self._A_values.requires_grad
            )
        ):
            # Set pending warm start as non-differentiable side data.
            # Tensors enter the C++ layer via raw `.data_ptr()` reads, so
            # we must validate device + dtype and force contiguous storage
            # before stashing them — a non-contiguous tensor (e.g. a
            # sliced view) would otherwise pass `.data_ptr()` straight to
            # the C++ which then strides past valid memory.
            has_warm = warm_x is not None
            if has_warm:
                _validate_cuda_tensor(warm_x, "warm_x")
                _validate_cuda_tensor(warm_z, "warm_z")
                _validate_cuda_tensor(warm_s, "warm_s")
                _validate_tensor_size(warm_x, "warm_x", self._n, dim=-1)
                _validate_tensor_size(warm_z, "warm_z", self._m, dim=-1)
                _validate_tensor_size(warm_s, "warm_s", self._m, dim=-1)
                warm_x_c = warm_x.contiguous()
                warm_z_c = warm_z.contiguous()
                warm_s_c = warm_s.contiguous()
                if warm_z_x is not None:
                    _validate_cuda_tensor(warm_z_x, "warm_z_x")
                    # Size check is essential: warm_z_x reaches C++ as a raw
                    # `.data_ptr()` and the solver reads exactly
                    # `_total_x_dim` doubles per batch. An undersized tensor
                    # strides past valid memory (OOB device read); an
                    # oversized one silently ignores the tail.
                    _validate_tensor_size(warm_z_x, "warm_z_x", self._total_x_dim, dim=-1)
                    warm_z_x_c = warm_z_x.contiguous()
                    self._pending_warm_start = (warm_x_c, warm_z_c, warm_s_c, warm_z_x_c)
                else:
                    self._pending_warm_start = (warm_x_c, warm_z_c, warm_s_c)
            else:
                self._pending_warm_start = None

            # Use autograd function for gradient tracking
            try:
                x, z, s, status, obj_val, z_x = _CudaSolveFunction.apply(
                    self, self._P_values, self._A_values, q, b
                )
            finally:
                # Clear pending warm start even on exception
                self._pending_warm_start = None

            # Squeeze if single problem
            if is_single:
                x = x.squeeze(0)
                z = z.squeeze(0)
                s = s.squeeze(0)
                if z_x.numel() > 0:
                    z_x = z_x.squeeze(0)
                status_val = int(status[0].item())
                obj_val_scalar = obj_val[0].item()

                solve_info = self._impl.get_solve_info()
                return {
                    "x": x,
                    "z": z,
                    "s": s,
                    "z_x": z_x,
                    "status": status_val,
                    "obj_val": obj_val_scalar,
                    "iterations": solve_info["iterations"],
                    "solve_time": solve_info["solve_time"],
                    "setup_time": solve_info.get("setup_time", 0.0),
                    "construction_time": solve_info.get("construction_time", 0.0),
                }
            else:
                solve_info = self._impl.get_solve_info()
                return {
                    "x": x,
                    "z": z,
                    "s": s,
                    "z_x": z_x,
                    "status": status,
                    "obj_val": obj_val,
                    "iterations": solve_info["iterations"],
                    "solve_time": solve_info["solve_time"],
                    "setup_time": solve_info.get("setup_time", 0.0),
                    "construction_time": solve_info.get("construction_time", 0.0),
                }

        # Non-differentiable path (faster, no autograd overhead)
        # Expand to 2D if needed
        if is_single:
            q_2d = q.unsqueeze(0)
            b_2d = b.unsqueeze(0)
            P_2d = self._P_values.unsqueeze(0) if self._P_values.dim() == 1 else self._P_values
            A_2d = self._A_values.unsqueeze(0) if self._A_values.dim() == 1 else self._A_values
        else:
            q_2d = q
            b_2d = b

            # Handle matrices: either shared (1D) or per-batch (2D)
            if self._P_values.dim() == 1:
                # Shared matrices - broadcast to batch
                P_2d = self._P_values.unsqueeze(0).expand(batch_size, -1).contiguous()
                A_2d = self._A_values.unsqueeze(0).expand(batch_size, -1).contiguous()
            else:
                P_2d = self._P_values
                A_2d = self._A_values
                if P_2d.shape[0] != batch_size:
                    raise ValueError(
                        f"Batch size mismatch: P/A have batch size {P_2d.shape[0]}, "
                        f"but q/b have batch size {batch_size}."
                    )

        # Allocate output tensors on GPU
        x = torch.empty((batch_size, self._n), dtype=torch.float64, device=device)
        z = torch.empty((batch_size, self._m), dtype=torch.float64, device=device)
        s = torch.empty((batch_size, self._m), dtype=torch.float64, device=device)
        status = torch.empty(batch_size, dtype=torch.int32, device=device)
        obj_val = torch.empty(batch_size, dtype=torch.float64, device=device)

        # Direct output (empty when no direct cones). Use the
        # precomputed `_total_x_dim` so dynamo doesn't trace into the
        # nanobind `dir_cones` property (raises `AssertionError: expected
        # FunctionType found nb_method`).
        total_xn = self._total_x_dim
        if total_xn > 0:
            z_x_out = torch.empty((batch_size, total_xn), dtype=torch.float64, device=device)
            z_x_out_ptr = z_x_out.data_ptr()
        else:
            z_x_out = torch.zeros((batch_size, 0), dtype=torch.float64, device=device)
            z_x_out_ptr = 0

        # Zero-copy solve using device pointers.
        # Validate + contiguify warm tensors before reading `.data_ptr()`
        # — same reasoning as the autograd path above.
        has_warm = warm_x is not None
        if has_warm:
            _validate_cuda_tensor(warm_x, "warm_x")
            _validate_cuda_tensor(warm_z, "warm_z")
            _validate_cuda_tensor(warm_s, "warm_s")
            warm_x_c = warm_x.contiguous()
            warm_z_c = warm_z.contiguous()
            warm_s_c = warm_s.contiguous()
            if warm_z_x is not None and warm_z_x.numel() > 0:
                _validate_cuda_tensor(warm_z_x, "warm_z_x")
                warm_z_x_c = warm_z_x.contiguous()
                warm_z_x_ptr = warm_z_x_c.data_ptr()
            else:
                warm_z_x_ptr = 0
            self._impl.solve_warm_start_to_device_pointers(
                P_2d.data_ptr(),
                A_2d.data_ptr(),
                q_2d.data_ptr(),
                b_2d.data_ptr(),
                warm_x_c.data_ptr(),
                warm_z_c.data_ptr(),
                warm_s_c.data_ptr(),
                x.data_ptr(),
                z.data_ptr(),
                s.data_ptr(),
                status.data_ptr(),
                obj_val.data_ptr(),
                warm_z_x_ptr,
                z_x_out_ptr,
            )
        else:
            self._impl.solve_to_device_pointers(
                P_2d.data_ptr(),
                A_2d.data_ptr(),
                q_2d.data_ptr(),
                b_2d.data_ptr(),
                x.data_ptr(),
                z.data_ptr(),
                s.data_ptr(),
                status.data_ptr(),
                obj_val.data_ptr(),
                z_x_out_ptr,
            )

        # Get solve info (iterations, solve_time, setup_time, construction_time) from solver
        solve_info = self._impl.get_solve_info()
        iterations = solve_info["iterations"]
        solve_time = solve_info["solve_time"]
        setup_time = solve_info.get("setup_time", 0.0)
        construction_time = solve_info.get("construction_time", 0.0)

        # Squeeze if single problem
        if is_single:
            x = x.squeeze(0)
            z = z.squeeze(0)
            s = s.squeeze(0)
            zx_squeezed = z_x_out.squeeze(0) if z_x_out.numel() > 0 else z_x_out
            status_val = int(status[0].item())
            obj_val_scalar = obj_val[0].item()

            return {
                "x": x,
                "z": z,
                "s": s,
                "z_x": zx_squeezed,
                "status": status_val,
                "obj_val": obj_val_scalar,
                "iterations": iterations,
                "solve_time": solve_time,
                "setup_time": setup_time,
                "construction_time": construction_time,
            }
        else:
            return {
                "x": x,
                "z": z,
                "s": s,
                "z_x": z_x_out,
                "status": status,
                "obj_val": obj_val,
                "iterations": iterations,
                "solve_time": solve_time,
                "setup_time": setup_time,
                "construction_time": construction_time,
            }

    def setup_grad(self, batch_size: Optional[int] = None) -> None:
        """Setup for gradient computation (not yet implemented for CUDA)."""
        self._grad_initialized = True
        # Placeholder - actual gradient setup would happen here

    def backward(
        self,
        dx: torch.Tensor,
        dz: Optional[torch.Tensor] = None,
        ds: Optional[torch.Tensor] = None,
        dz_x: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute gradients via implicit differentiation.

        Args:
            dx: Upstream gradient w.r.t. x
            dz: Upstream gradient w.r.t. z (optional, defaults to zeros)
            ds: Upstream gradient w.r.t. s (optional, defaults to zeros)
            dz_x: Upstream gradient w.r.t. direct cone duals z_x.
                Pass None or empty for slack-only problems.

        Returns:
            Tuple of (dP_values, dq, dA_values, db) gradients.
        """
        if not self._enable_grad:
            raise RuntimeError("backward() requires enable_grad=True in settings")
        if self._impl is None:
            raise RuntimeError("solve() must be called before backward()")

        # Handle dimensions - ensure 2D
        is_single = dx.dim() == 1
        if is_single:
            dx = dx.unsqueeze(0)

        batch_size = dx.shape[0]
        device = dx.device

        # Default zeros for unspecified gradients
        if dz is None:
            dz = torch.zeros(batch_size, self._m, dtype=torch.float64, device=device)
        elif dz.dim() == 1:
            dz = dz.unsqueeze(0)

        if ds is None:
            ds = torch.zeros(batch_size, self._m, dtype=torch.float64, device=device)
        elif ds.dim() == 1:
            ds = ds.unsqueeze(0)

        # Ensure contiguous
        dx = dx.contiguous()
        dz = dz.contiguous()
        ds = ds.contiguous()

        # dz_x device-pointer plumbing — zero-copy when supplied.
        # `dz_x_c` must outlive the C++ call: a non-contiguous autograd
        # gradient makes `.contiguous()` allocate a copy whose only
        # reference is the temporary, freed the instant this statement
        # ends — leaving dz_x_ptr dangling (use-after-free).
        dz_x_ptr = 0
        if dz_x is not None and dz_x.numel() > 0:
            dz_x_2d = dz_x if dz_x.dim() == 2 else dz_x.unsqueeze(0)
            dz_x_c = dz_x_2d.contiguous()
            dz_x_ptr = dz_x_c.data_ptr()

        # Allocate output tensors
        dP_values = torch.empty(batch_size, self._nnzP, dtype=torch.float64, device=device)
        dq = torch.empty(batch_size, self._n, dtype=torch.float64, device=device)
        dA_values = torch.empty(batch_size, self._nnzA, dtype=torch.float64, device=device)
        db = torch.empty(batch_size, self._m, dtype=torch.float64, device=device)

        # Call C++ backward with device pointers
        self._impl.backward_to_device_pointers(
            dx.data_ptr(),
            dz.data_ptr(),
            ds.data_ptr(),
            dP_values.data_ptr(),
            dq.data_ptr(),
            dA_values.data_ptr(),
            db.data_ptr(),
            dz_x_ptr,
        )

        # Squeeze if single problem
        if is_single:
            dP_values = dP_values.squeeze(0)
            dq = dq.squeeze(0)
            dA_values = dA_values.squeeze(0)
            db = db.squeeze(0)

        return dP_values, dq, dA_values, db

    def backward_with_mode(
        self,
        dx: torch.Tensor,
        dz: torch.Tensor,
        ds: torch.Tensor,
        mode: str,
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
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute gradients with explicit saved tensors.

        Uses loadDataForBackward to reload and re-equilibrate problem data,
        then restores the saved solution and syncs equilibration factors to
        DiffState. No re-solve needed.

        Returns:
            Tuple of (dP_values, dq, dA_values, db) gradients.
        """
        # Determine batch size from upstream gradient
        batch_size = dx.shape[0] if dx.dim() == 2 else 1

        # Expand shared (1D) tensors to (batch, dim) for the C++ layer
        if P_values.dim() == 1 or (
            P_values.dim() == 2 and P_values.shape[0] == 1 and batch_size > 1
        ):
            P_values = P_values.reshape(1, -1).expand(batch_size, -1).contiguous()
        if A_values.dim() == 1 or (
            A_values.dim() == 2 and A_values.shape[0] == 1 and batch_size > 1
        ):
            A_values = A_values.reshape(1, -1).expand(batch_size, -1).contiguous()
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if b.dim() == 1:
            b = b.unsqueeze(0)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if s.dim() == 1:
            s = s.unsqueeze(0)

        # Ensure contiguous for data_ptr()
        P_values = P_values.contiguous()
        A_values = A_values.contiguous()
        q = q.contiguous()
        b = b.contiguous()
        x = x.contiguous()
        z = z.contiguous()
        s = s.contiguous()

        # Restore full backward state: problem data, equilibration, and solution.
        # This replaces the old re-solve approach ��� loadDataForBackward reloads
        # P/A/q/b and re-equilibrates, then the solution and equilibration
        # factors are synced to DiffState.
        # If batch_size differs from the solver's batch size (e.g. vmap
        # batching N upstream grads through a solver built for a different
        # size), create a temporary solver with the right batch size so the
        # backward runs fully parallel on the GPU.
        solver_batch = self._current_batch_size or 1
        if batch_size != solver_batch:
            if not getattr(self, "_warned_backward_batch_change", False):
                warnings.warn(
                    f"backward_with_mode: batch size {batch_size} differs from "
                    f"solver batch size {solver_batch}. Creating a temporary "
                    f"CUDA solver for the backward pass, which may degrade "
                    f"performance. This is expected when using torch.func.jacrev "
                    f"or vmap over backward. To avoid this overhead, construct "
                    f"the solver with batch_size={batch_size}.",
                    UserWarning,
                    stacklevel=3,
                )
                self._warned_backward_batch_change = True
            tmp = _RawCudaSolver(
                self._n,
                self._m,
                batch_size,
                self._P_row_offsets,
                self._P_col_indices,
                self._A_row_offsets,
                self._A_col_indices,
                self._cuda_cones,
                self._cuda_settings,
                True,  # enable_grad
            )
            tmp.load_backward_state_from_device_pointers(
                P_values.data_ptr(),
                A_values.data_ptr(),
                q.data_ptr(),
                b.data_ptr(),
                x.data_ptr(),
                z.data_ptr(),
                s.data_ptr(),
            )
            # Allocate output tensors and run backward
            device = dx.device
            dP_values = torch.empty(batch_size, self._nnzP, dtype=torch.float64, device=device)
            dq_out = torch.empty(batch_size, self._n, dtype=torch.float64, device=device)
            dA_values = torch.empty(batch_size, self._nnzA, dtype=torch.float64, device=device)
            db_out = torch.empty(batch_size, self._m, dtype=torch.float64, device=device)
            dx = dx.contiguous()
            if dz is None:
                dz = torch.zeros(batch_size, self._m, dtype=torch.float64, device=device)
            elif dz.dim() == 1:
                dz = dz.unsqueeze(0)
            if ds is None:
                ds = torch.zeros(batch_size, self._m, dtype=torch.float64, device=device)
            elif ds.dim() == 1:
                ds = ds.unsqueeze(0)
            dz = dz.contiguous()
            ds = ds.contiguous()
            dz_x_ptr_tmp = 0
            if dz_x is not None and dz_x.numel() > 0:
                dz_x_tmp_2d = dz_x if dz_x.dim() == 2 else dz_x.unsqueeze(0)
                # Keep the contiguous copy referenced through the C++ call
                # (see use-after-free note in backward()).
                dz_x_tmp_c = dz_x_tmp_2d.contiguous()
                dz_x_ptr_tmp = dz_x_tmp_c.data_ptr()
            tmp.backward_to_device_pointers(
                dx.data_ptr(),
                dz.data_ptr(),
                ds.data_ptr(),
                dP_values.data_ptr(),
                dq_out.data_ptr(),
                dA_values.data_ptr(),
                db_out.data_ptr(),
                dz_x_ptr_tmp,
            )
            return dP_values, dq_out, dA_values, db_out

        # z_x is also threaded into the backward state: when present, the
        # binding's `load_backward_state_from_device_pointers` 8th arg
        # restores the equilibrated direct dual into DiffState. (load
        # signature optional-zeroes when absent.)
        z_x_ptr = 0
        if z_x is not None and z_x.numel() > 0:
            z_x_2d = z_x if z_x.dim() == 2 else z_x.unsqueeze(0)
            # Keep the contiguous copy alive through the C++ call below
            # (see use-after-free note in backward()).
            z_x_state_c = z_x_2d.contiguous()
            z_x_ptr = z_x_state_c.data_ptr()
        self._impl.load_backward_state_from_device_pointers(
            P_values.data_ptr(),
            A_values.data_ptr(),
            q.data_ptr(),
            b.data_ptr(),
            x.data_ptr(),
            z.data_ptr(),
            s.data_ptr(),
            z_x_ptr,
        )
        return self.backward(dx, dz, ds, dz_x=dz_x)

    @property
    def n(self) -> int:
        return self._n

    @property
    def m(self) -> int:
        return self._m

    @property
    def batch_size(self) -> Optional[int]:
        """Current or expected batch size."""
        return self._current_batch_size or self._expected_batch_size

    @property
    def is_initialized(self) -> bool:
        """Whether the underlying solver has been created."""
        return self._impl is not None

    @property
    def nnzP(self) -> int:
        return self._nnzP

    @property
    def nnzA(self) -> int:
        return self._nnzA

    @property
    def grad_initialized(self) -> bool:
        return self._grad_initialized

    def reset(self) -> None:
        """Reset solver state."""
        self._impl = None
        self._current_batch_size = None
        self._P_values = None
        self._A_values = None

    def get_dimensions(self) -> Dict[str, int]:
        """Get problem dimensions."""
        return {
            "n": self._n,
            "m": self._m,
            "batch_size": self._current_batch_size or 0,
            "nnzP": self._nnzP,
            "nnzA": self._nnzA,
        }

    def __repr__(self) -> str:
        status = "initialized" if self.is_initialized else "uninitialized"
        batch_str = str(self.batch_size) if self.batch_size else "?"
        return f"TorchSolver(n={self._n}, m={self._m}, batch_size={batch_str}, {status})"
