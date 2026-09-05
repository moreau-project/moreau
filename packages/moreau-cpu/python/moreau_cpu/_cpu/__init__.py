"""CPU adapter layer: Provides CpuSolver wrapper for the Rust backend.

This module provides a unified solver that automatically handles both single
problems and batched problems:
- 1D inputs → single problem using DefaultSolver
- 2D inputs → batched problems using CompiledSolver (rayon parallelism)

Three-step API pattern:
1. Construct solver with structure (dimensions, sparsity patterns, cones)
   - Single: no work done
   - Batch: symbolic factorization done here
2. setup(P_values, A_values) - Set matrix values and equilibration
3. solve(q, b) - Solve with linear cost and constraint RHS
"""

import numpy as np
from scipy import sparse
from typing import Dict, Any, Optional, List
import time
import warnings
from enum import IntEnum

# Import from the Rust extension module
from moreau_cpu import _cpu_solver


class SolverStatus(IntEnum):
    """Solver status enum."""

    Unsolved = 0
    Solved = 1
    PrimalInfeasible = 2
    DualInfeasible = 3
    AlmostSolved = 4
    AlmostPrimalInfeasible = 5
    AlmostDualInfeasible = 6
    MaxIterations = 7
    MaxTime = 8
    NumericalError = 9
    InsufficientProgress = 10
    CallbackTerminated = 11


class Cones:
    """Cone specification for conic optimization problems.

    Attributes:
        num_zero_cones: Number of equality constraint dimensions (zero cone)
        num_nonneg_cones: Number of inequality constraint dimensions (nonnegative cone)
        so_cone_dims: List of second-order cone dimensions (each >= 2)
        num_exp_cones: Number of exponential cones (each is 3D)
        power_alphas: List of power cone alpha parameters (each cone is 3D)
        gen_power_cone_params: List of (alphas, dim2) tuples for generalized power cones.
            alphas is a list of positive floats summing to 1, dim2 >= 1.
        psd_dims: List of PSD cone matrix dimensions (each >= 1)
    """

    def __init__(
        self,
        num_zero_cones: int = 0,
        num_nonneg_cones: int = 0,
        so_cone_dims: List[int] = None,
        num_exp_cones: int = 0,
        power_alphas: List[float] = None,
        gen_power_cone_params: List[tuple] = None,
        psd_dims: List[int] = None,
        *,
        num_so_cones: int = 0,
    ):
        self.num_zero_cones = num_zero_cones
        self.num_nonneg_cones = num_nonneg_cones
        # so_cone_dims takes precedence; num_so_cones is backward compat
        if so_cone_dims is not None:
            self.so_cone_dims = list(so_cone_dims)
        elif num_so_cones > 0:
            self.so_cone_dims = [3] * num_so_cones
        else:
            self.so_cone_dims = []
        self.num_exp_cones = num_exp_cones
        self.power_alphas = power_alphas or []
        self.gen_power_cone_params = gen_power_cone_params or []
        self.psd_dims = list(psd_dims) if psd_dims is not None else []

    @property
    def num_so_cones(self) -> int:
        """Number of second-order cones."""
        return len(self.so_cone_dims)

    @property
    def num_psd_cones(self) -> int:
        """Number of PSD cones."""
        return len(self.psd_dims)


def _dir_cones_to_cpu(cones) -> List:
    """Extract direct cones from a `Cones`-like object and convert to the
    pyo3 `SupportedXConeT` variants.

    Returns an empty list if the input has no `dir_cones` attribute or its
    value is empty. `kind='psd_triangle'` requires the CPU backend to be
    built with the `sdp` feature (normally via `sdp-openblas,python`).
    """
    x_specs = getattr(cones, "dir_cones", None)
    if not x_specs:
        return []
    out = []
    for spec in x_specs:
        kind = getattr(spec, "kind", None)
        indices = list(getattr(spec, "indices", []))
        if kind == "nonneg":
            out.append(_cpu_solver.NonnegativeXConeT(indices))
        elif kind == "soc":
            out.append(_cpu_solver.SecondOrderXConeT(indices))
        elif kind == "exp":
            out.append(_cpu_solver.ExponentialXConeT(indices))
        elif kind == "power":
            alpha = getattr(spec, "alpha", None)
            if alpha is None:
                raise ValueError("DirectConeSpec(kind='power') requires alpha")
            out.append(_cpu_solver.PowerXConeT(indices, float(alpha)))
        elif kind == "gen_power":
            alphas = getattr(spec, "alphas", None)
            dim2 = getattr(spec, "dim2", None)
            if alphas is None or dim2 is None:
                raise ValueError("DirectConeSpec(kind='gen_power') requires alphas and dim2")
            out.append(
                _cpu_solver.GenPowerXConeT(
                    indices,
                    [float(a) for a in alphas],
                    int(dim2),
                )
            )
        elif kind == "psd_triangle":
            psd_k = getattr(spec, "psd_k", None)
            if psd_k is None:
                raise ValueError("DirectConeSpec(kind='psd_triangle') requires psd_k")
            xcone_t = getattr(_cpu_solver, "PSDTriangleXConeT", None)
            if xcone_t is None:
                raise NotImplementedError(
                    "PSD direct cones require the CPU backend to be built " "with the `sdp` feature"
                )
            out.append(xcone_t(indices, int(psd_k)))
        else:
            raise ValueError(f"Unknown DirectConeSpec.kind: {kind!r}")
    return out


def cones_to_cpu(cones) -> List:
    """Convert Cones dataclass to CPU solver SLACK cone list format.

    Direct cones (`cones.dir_cones`) are NOT included in the returned
    list — they go via a separate dir_cones argument on the CPU solver
    (see `_dir_cones_to_cpu`). Call both helpers to materialize the full
    cone set for CompiledSolver / DefaultSolver.
    """
    cpu_cones = []
    if cones.num_zero_cones > 0:
        cpu_cones.append(_cpu_solver.ZeroConeT(cones.num_zero_cones))
    if cones.num_nonneg_cones > 0:
        cpu_cones.append(_cpu_solver.NonnegativeConeT(cones.num_nonneg_cones))
    for dim in cones.so_cone_dims:
        cpu_cones.append(_cpu_solver.SecondOrderConeT(dim))
    # PSD cones must come before exp/power to match CVXPY/Clarabel cone ordering
    for dim in getattr(cones, "psd_dims", []):
        cpu_cones.append(_cpu_solver.PSDTriangleConeT(dim))
    for _ in range(cones.num_exp_cones):
        cpu_cones.append(_cpu_solver.ExponentialConeT())
    for alpha in cones.power_alphas:
        cpu_cones.append(_cpu_solver.PowerConeT(alpha))
    for alphas, dim2 in getattr(cones, "gen_power_cone_params", []):
        cpu_cones.append(_cpu_solver.GenPowerConeT(list(alphas), dim2))
    return cpu_cones


def settings_to_cpu(settings) -> "_cpu_solver.DefaultSettings":
    """Convert Settings object to CPU solver DefaultSettings."""
    if isinstance(settings, _cpu_solver.DefaultSettings):
        return settings

    cpu_settings = _cpu_solver.DefaultSettings()

    # Copy top-level settings
    for attr in ["max_iter", "time_limit", "verbose", "yolo", "yolo_num_iters"]:
        if hasattr(settings, attr):
            setattr(cpu_settings, attr, getattr(settings, attr))

    # Copy IPM settings to the nested ipm field
    # PyO3 returns a copy, so we get it, modify it, then reassign
    ipm_src = getattr(settings, "ipm_settings", None)
    if ipm_src is not None:
        ipm_dst = cpu_settings.ipm
        for attr in [
            "tol_gap_abs",
            "tol_gap_rel",
            "tol_feas",
            "tol_infeas_abs",
            "tol_infeas_rel",
            "tol_ktratio",
            "reduced_tol_gap_abs",
            "reduced_tol_gap_rel",
            "reduced_tol_feas",
            "reduced_tol_infeas_abs",
            "reduced_tol_infeas_rel",
            "reduced_tol_ktratio",
            "max_step_fraction",
            "equilibrate_enable",
            "direct_solve_method",
            "diff_method",
            "diff_smoothing_mu",
            "diff_smoothing_step_factor",
            "chordal_decomposition_enable",
            "chordal_decomposition_merge_method",
        ]:
            # Only copy if both source has the attribute AND dest can accept it
            if hasattr(ipm_src, attr) and hasattr(ipm_dst, attr):
                value = getattr(ipm_src, attr)
                try:
                    setattr(ipm_dst, attr, value)
                except (AttributeError, TypeError) as e:
                    # Log warning instead of silently ignoring
                    warnings.warn(
                        f"Failed to set IPM setting '{attr}' = {value!r}: {e}",
                        RuntimeWarning,
                        stacklevel=3,
                    )
        cpu_settings.ipm = ipm_dst

    return cpu_settings


# Expose internal solver classes with underscore prefix
# These are internal implementation details
_SolverInternal = _cpu_solver.DefaultSolver
_CompiledSolverInternal = _cpu_solver.CompiledSolver


class ActiveSetSolver:
    """CPU active-set QP solver with the same setup/solve/backward dict interface.

    Only supports zero + nonneg cones (pure QP). Wraps the C++ DAQP solver
    via the Rust FFI bindings.

    Same three-step API as Solver:
    1. Construct with structure
    2. setup(P_values, A_values)
    3. solve(q, b) → dict
    """

    def __init__(
        self,
        n: int,
        m: int,
        P_row_offsets,
        P_col_indices,
        A_row_offsets,
        A_col_indices,
        cones,
        settings=None,
        batch_size: Optional[int] = None,
        enable_grad: bool = False,
    ):
        self._n = n
        self._m = m
        self._P_row_offsets = np.asarray(P_row_offsets, dtype=np.int64)
        self._P_col_indices = np.asarray(P_col_indices, dtype=np.int64)
        self._A_row_offsets = np.asarray(A_row_offsets, dtype=np.int64)
        self._A_col_indices = np.asarray(A_col_indices, dtype=np.int64)
        self._nnz_P = len(P_col_indices)
        self._nnz_A = len(A_col_indices)
        self._enable_grad = enable_grad
        self._batch_size = batch_size or 1

        # Active-set CPU solver only supports zero + nonneg slack cones.
        # Direct cones (cones.dir_cones) and exotic slack cones (SOC, exp,
        # power, gen_power, psd_triangle) are not handled here: silently
        # dropping them would solve the wrong problem. The unified
        # `moreau.Solver` auto-resolver avoids routing such problems here
        # (see `_resolve_solver_type` in moreau/__init__.py), but reject
        # explicit `solver='active_set'` requests too.
        dir_cones = getattr(cones, "dir_cones", None) or []
        if dir_cones:
            raise ValueError(
                "ActiveSetSolver does not support direct cones "
                f"(`cones.dir_cones` has {len(dir_cones)} entries). Use "
                "`solver='ipm'` or `solver='auto'` for problems with dir_cones."
            )
        for attr, label in (
            ("num_exp_cones", "exponential"),
            ("power_alphas", "power"),
            ("so_cone_dims", "second-order"),
            ("gen_power_cone_params", "generalized power"),
            ("psd_dims", "PSD"),
        ):
            val = getattr(cones, attr, None)
            count = val if isinstance(val, int) else (len(val) if val else 0)
            if count:
                raise ValueError(
                    f"ActiveSetSolver does not support {label} cones "
                    f"({attr}={val!r}). Use `solver='ipm'` or "
                    f"`solver='auto'` for problems with non-LP/QP cones."
                )

        # Extract cone counts
        num_zero = getattr(cones, "num_zero_cones", 0)
        num_nonneg = getattr(cones, "num_nonneg_cones", 0)

        # Build active-set settings
        as_settings_kwargs = {}
        if settings is not None:
            as_src = getattr(settings, "active_set_settings", None)
            if as_src is not None:
                for attr in [
                    "primal_tol",
                    "dual_tol",
                    "zero_tol",
                    "pivot_tol",
                    "progress_tol",
                    "fval_bound",
                    "iter_limit",
                    "cycle_tol",
                    "diff_method",
                    "diff_smoothing_mu",
                ]:
                    if hasattr(as_src, attr):
                        as_settings_kwargs[attr] = getattr(as_src, attr)
            if hasattr(settings, "max_iter") and settings.max_iter is not None:
                # Only use top-level max_iter if the user didn't set iter_limit
                # in active_set_settings (active-set default is 10000, not 200)
                if "iter_limit" not in as_settings_kwargs or as_src is None:
                    as_settings_kwargs["iter_limit"] = int(settings.max_iter)
            if hasattr(settings, "time_limit") and settings.time_limit is not None:
                tl = settings.time_limit
                if tl != float("inf"):
                    as_settings_kwargs["time_limit"] = float(tl)

        verbose = getattr(settings, "verbose", False) if settings else False
        as_settings = _cpu_solver.ActiveSetSettings(**as_settings_kwargs)

        self._solver = _cpu_solver.ActiveSetSolver(
            n=n,
            m=m,
            batch_size=self._batch_size,
            P_row_offsets=self._P_row_offsets.tolist(),
            P_col_indices=self._P_col_indices.tolist(),
            nnz_P=self._nnz_P,
            A_row_offsets=self._A_row_offsets.tolist(),
            A_col_indices=self._A_col_indices.tolist(),
            nnz_A=self._nnz_A,
            num_zero_cones=num_zero,
            num_nonneg_cones=num_nonneg,
            settings=as_settings,
            enable_grad=enable_grad,
            verbose=verbose,
        )
        self._last_backward_state = None

    def setup(self, P_values, A_values):
        P_values = np.asarray(P_values, dtype=np.float64)
        A_values = np.asarray(A_values, dtype=np.float64)
        p_shared = P_values.ndim == 1
        a_shared = A_values.ndim == 1
        # C++ backend requires both shared or both per-batch;
        # if they differ, tile the shared one to per-batch
        if p_shared != a_shared:
            if p_shared:
                P_values = np.tile(P_values, (self._batch_size, 1))
            else:
                A_values = np.tile(A_values, (self._batch_size, 1))
            shared = False
        else:
            shared = p_shared
        P_flat = np.ascontiguousarray(P_values, dtype=np.float64).ravel()
        A_flat = np.ascontiguousarray(A_values, dtype=np.float64).ravel()
        self._solver.setup(P_flat, A_flat, shared)

    def solve(
        self,
        q,
        b,
        warm_x=None,
        warm_z=None,
        warm_s=None,
    ) -> Dict[str, Any]:
        q = np.asarray(q, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        is_single = q.ndim == 1
        if is_single:
            q = q.reshape(1, -1)
            b = b.reshape(1, -1)

        batch_size = q.shape[0]
        n, m = self._n, self._m

        q_flat = np.ascontiguousarray(q, dtype=np.float64).ravel()
        b_flat = np.ascontiguousarray(b, dtype=np.float64).ravel()

        if warm_x is not None:
            warm_x_flat = np.ascontiguousarray(warm_x, dtype=np.float64).ravel()
            warm_z_flat = np.ascontiguousarray(warm_z, dtype=np.float64).ravel()
            warm_s_flat = np.ascontiguousarray(warm_s, dtype=np.float64).ravel()
            raw = self._solver.solve(q_flat, b_flat, warm_x_flat, warm_z_flat, warm_s_flat)
        else:
            raw = self._solver.solve(q_flat, b_flat)

        x = np.array(raw["x"], dtype=np.float64).reshape(batch_size, n)
        z = np.array(raw["z"], dtype=np.float64).reshape(batch_size, m)
        s = np.array(raw["s"], dtype=np.float64).reshape(batch_size, m)
        statuses = [SolverStatus(st) for st in raw["status"]]
        obj_vals = list(raw["obj_val"])
        iters = list(raw["iterations"])
        backward_state = self._solver.get_backward_state() if self._enable_grad else None
        self._last_backward_state = backward_state

        result = {
            "x": x,
            "s": s,
            "z": z,
            "status": statuses,
            "obj_val": obj_vals,
            "dual_obj_val": obj_vals,
            "iterations": iters,
            "construction_time": raw.get("construction_time", 0.0),
            "setup_time": raw.get("setup_time", 0.0),
            "solve_time": raw.get("solve_time", 0.0),
            "_backward_state": backward_state,
        }

        if is_single:
            return {
                "x": result["x"].squeeze(0),
                "s": result["s"].squeeze(0),
                "z": result["z"].squeeze(0),
                "status": result["status"][0],
                "obj_val": result["obj_val"][0],
                "dual_obj_val": result["dual_obj_val"][0],
                "iterations": result["iterations"][0],
                "construction_time": result["construction_time"],
                "setup_time": result["setup_time"],
                "solve_time": result["solve_time"],
            }

        return result

    def backward(
        self,
        dx,
        dz=None,
        ds=None,
        dz_x=None,
    ) -> Dict[str, np.ndarray]:
        if not self._enable_grad:
            raise RuntimeError("backward() requires enable_grad=True")

        # Active-set CPU path doesn't support direct cones. The unified
        # `Solver.backward` wrapper passes `dz_x=dz_x` unconditionally
        # (default None) since direct problems exist on the IPM path,
        # so we accept the kwarg but raise on a non-trivial upstream.
        if dz_x is not None:
            dz_x_arr = np.asarray(dz_x, dtype=np.float64)
            if dz_x_arr.size > 0 and not np.all(dz_x_arr == 0.0):
                raise RuntimeError(
                    "ActiveSetSolver does not support direct cones; "
                    "use the IPM solver for problems with `dir_cones`"
                )

        dx = np.asarray(dx, dtype=np.float64)
        is_single = dx.ndim == 1
        if is_single:
            dx = dx.reshape(1, -1)
        batch_size = dx.shape[0]

        if dz is None:
            dz = np.zeros((batch_size, self._m), dtype=np.float64)
        else:
            dz = np.asarray(dz, dtype=np.float64)
            if dz.ndim == 1:
                dz = dz.reshape(1, -1)
        if ds is None:
            ds = np.zeros((batch_size, self._m), dtype=np.float64)
        else:
            ds = np.asarray(ds, dtype=np.float64)
            if ds.ndim == 1:
                ds = ds.reshape(1, -1)

        dx_flat = np.ascontiguousarray(dx, dtype=np.float64).ravel()
        dz_flat = np.ascontiguousarray(dz, dtype=np.float64).ravel()
        ds_flat = np.ascontiguousarray(ds, dtype=np.float64).ravel()

        raw = self._solver.backward(dx_flat, dz_flat, ds_flat)

        result = {
            "dP_values": np.array(raw["dP_values"], dtype=np.float64).reshape(
                batch_size, self._nnz_P
            ),
            "dA_values": np.array(raw["dA_values"], dtype=np.float64).reshape(
                batch_size, self._nnz_A
            ),
            "dq": np.array(raw["dq"], dtype=np.float64).reshape(batch_size, self._n),
            "db": np.array(raw["db"], dtype=np.float64).reshape(batch_size, self._m),
        }

        if is_single:
            return {k: v.squeeze(0) for k, v in result.items()}
        return result

    def backward_with_data_flat(
        self,
        dx_flat: np.ndarray,
        ds_flat: np.ndarray,
        dz_flat: np.ndarray,
        P_values_flat: np.ndarray,
        A_values_flat: np.ndarray,
        q_flat: np.ndarray,
        b_flat: np.ndarray,
        x_flat: np.ndarray,
        z_flat: np.ndarray,
        s_flat: np.ndarray,
        backward_state,
        batch_size: int,
    ) -> Dict[str, np.ndarray]:
        """Compute gradients from explicit problem data and saved solution."""
        if not self._enable_grad:
            raise RuntimeError("backward_with_data_flat() requires enable_grad=True")
        raw = self._solver.backward_with_data_flat(
            np.ascontiguousarray(dx_flat, dtype=np.float64).ravel(),
            np.ascontiguousarray(ds_flat, dtype=np.float64).ravel(),
            np.ascontiguousarray(dz_flat, dtype=np.float64).ravel(),
            np.ascontiguousarray(P_values_flat, dtype=np.float64).ravel(),
            np.ascontiguousarray(A_values_flat, dtype=np.float64).ravel(),
            np.ascontiguousarray(q_flat, dtype=np.float64).ravel(),
            np.ascontiguousarray(b_flat, dtype=np.float64).ravel(),
            np.ascontiguousarray(x_flat, dtype=np.float64).ravel(),
            np.ascontiguousarray(z_flat, dtype=np.float64).ravel(),
            np.ascontiguousarray(s_flat, dtype=np.float64).ravel(),
            backward_state,
            len(P_values_flat) == self._nnz_P and len(A_values_flat) == self._nnz_A,
        )

        result = {
            "dP_values": np.array(raw["dP_values"], dtype=np.float64).reshape(
                batch_size, self._nnz_P
            ),
            "dA_values": np.array(raw["dA_values"], dtype=np.float64).reshape(
                batch_size, self._nnz_A
            ),
            "dq": np.array(raw["dq"], dtype=np.float64).reshape(batch_size, self._n),
            "db": np.array(raw["db"], dtype=np.float64).reshape(batch_size, self._m),
        }
        return result


class Solver:
    """Unified CPU solver for single and batched conic optimization.

    Automatically handles both single problems (1D inputs) and batched problems
    (2D inputs).

    Architecture:
    - Single path: Uses CompiledSolver (batch_size=1)
    - Batch path: Uses CompiledSolver with parallel equilibration

    Three-step API pattern:

    Single problem:
    1. Constructor - no work done (lazy initialization)
    2. setup(P, A) - presolve + equilibration
    3. solve(q, b) - actual solve

    Batch problem:
    1. Constructor - symbolic factorization (if batch_size provided)
    2. setup(P, A) - equilibration
    3. solve(q, b) - actual solve

    Args:
        n: Number of primal variables
        m: Number of constraints
        P_row_offsets: CSR row pointers for P matrix (n+1 elements)
        P_col_indices: CSR column indices for P matrix
        A_row_offsets: CSR row pointers for A matrix (m+1 elements)
        A_col_indices: CSR column indices for A matrix
        cones: Cone specification (Cones object or similar)
        settings: Optional solver settings
        batch_size: Optional batch size hint. If provided, eagerly creates batch
                    solver with symbolic factorization at construction time.
        enable_grad: If True, pre-compute gradient structures for backward pass.
                     Required for backward() to work.

    Example:
        >>> solver = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones)
        >>> solver.setup(P_values, A_values)
        >>>
        >>> # Single problem (1D inputs)
        >>> result = solver.solve(q, b)
        >>>
        >>> # Batched problems (2D inputs)
        >>> result = solver.solve(q_batch, b_batch)
    """

    # Warning threshold for frequent batch size changes
    _BATCH_CHANGE_WARN_THRESHOLD = 5

    def __init__(
        self,
        n: int,
        m: int,
        P_row_offsets: np.ndarray,
        P_col_indices: np.ndarray,
        A_row_offsets: np.ndarray,
        A_col_indices: np.ndarray,
        cones,
        settings: Optional[Any] = None,
        batch_size: Optional[int] = None,
        enable_grad: bool = False,
        b_sparsity_pattern: Optional[list] = None,
    ):
        self._n = n
        self._m = m
        self._expected_batch_size = batch_size
        self._enable_grad = enable_grad
        self._b_sparsity_pattern = b_sparsity_pattern

        # Store CSR structure
        self._P_row_offsets = np.asarray(P_row_offsets, dtype=np.int64)
        self._P_col_indices = np.asarray(P_col_indices, dtype=np.int64)
        self._A_row_offsets = np.asarray(A_row_offsets, dtype=np.int64)
        self._A_col_indices = np.asarray(A_col_indices, dtype=np.int64)
        self._nnz_P = len(P_col_indices)
        self._nnz_A = len(A_col_indices)

        # Stored matrix values (set via setup)
        self._P_values: Optional[np.ndarray] = None
        self._A_values: Optional[np.ndarray] = None

        # Convert cones
        if hasattr(cones, "num_zero_cones"):
            self._cones = cones_to_cpu(cones)
            self._dir_cones = _dir_cones_to_cpu(cones)
        else:
            self._cones = cones
            self._dir_cones = []

        # Handle settings (for IPM path)
        if settings is None:
            self._settings = _cpu_solver.DefaultSettings()
            self._settings.verbose = True
        else:
            self._settings = settings_to_cpu(settings)

        # Solver instances
        # - _single_solver: created on demand at first single solve
        # - _compiled_solver: created eagerly if batch_size provided, else on demand
        self._single_solver = None
        self._compiled_solver = None
        self._current_batch_size = None

        # Cached P/A values for tracking when equilibration needs refresh
        self._cached_single_P = None
        self._cached_single_A = None
        self._cached_shared_P = None
        self._cached_shared_A = None

        # Batch size change tracking for warnings
        self._batch_size_change_count = 0
        self._warned_about_batch_changes = False

        # Gradient setup
        self._grad_initialized = enable_grad

        # Eagerly create solver if batch_size > 1, or batch_size == 1 with
        # b_sparsity_pattern (enables chordal decomposition at construction time)
        if batch_size is not None and batch_size > 1:
            self._create_compiled_solver()
            self._current_batch_size = batch_size
        elif batch_size is not None and batch_size == 1 and b_sparsity_pattern is not None:
            self._create_single_solver()
            self._current_batch_size = 1

    def _create_compiled_solver(self):
        """Create CompiledSolver.

        This performs symbolic factorization at construction time.
        """
        self._compiled_solver = _cpu_solver.CompiledSolver(
            n=self._n,
            m=self._m,
            P_row_offsets=self._P_row_offsets.tolist(),
            P_col_indices=self._P_col_indices.tolist(),
            A_row_offsets=self._A_row_offsets.tolist(),
            A_col_indices=self._A_col_indices.tolist(),
            cones=self._cones,
            settings=self._settings,
            enable_grad=self._enable_grad,
            dir_cones=self._dir_cones if self._dir_cones else None,
            b_sparsity_pattern=self._b_sparsity_pattern,
        )

    def setup(
        self,
        P_values: np.ndarray,
        A_values: np.ndarray,
    ) -> None:
        """Set P and A matrix values and precompute equilibration.

        Must be called before solve(). Can be called multiple times to update
        values for repeated solves with the same structure.

        Work done in setup():
        - Stores matrix values
        - Precomputes equilibration factors (for batch path if solver exists)

        For batched problems, pass 2D arrays with shape (batch, nnz_P) and (batch, nnz_A).

        Args:
            P_values: P matrix values. Shape (nnz_P,) for single, (batch, nnz_P) for batched
            A_values: A matrix values. Shape (nnz_A,) for single, (batch, nnz_A) for batched

        Raises:
            ValueError: If value dimensions don't match sparsity pattern.
            TypeError: If values are not float64.
        """
        P_values = np.asarray(P_values, dtype=np.float64)
        A_values = np.asarray(A_values, dtype=np.float64)

        # Validate dimensions
        P_dim = P_values.shape[-1] if P_values.ndim > 0 else 0
        A_dim = A_values.shape[-1] if A_values.ndim > 0 else 0

        if P_dim != self._nnz_P:
            raise ValueError(f"P_values has {P_dim} elements, expected {self._nnz_P}")
        if A_dim != self._nnz_A:
            raise ValueError(f"A_values has {A_dim} elements, expected {self._nnz_A}")

        self._P_values = P_values
        self._A_values = A_values

        # Clear cache to ensure solve() will re-setup with correct batch size
        self._cached_shared_P = None
        self._cached_shared_A = None

        # Only precompute equilibration for 2D (batched) values
        # For 1D (shared) values, defer to _solve_batch_shared which knows the batch size
        if self._compiled_solver is not None and P_values.ndim > 1:
            batch_size = P_values.shape[0]
            P_flat = np.ascontiguousarray(P_values, dtype=np.float64).ravel()
            A_flat = np.ascontiguousarray(A_values, dtype=np.float64).ravel()
            self._compiled_solver.setup_flat(P_flat, A_flat, batch_size)
            self._cached_shared_P = P_values.copy()
            self._cached_shared_A = A_values.copy()

    def _create_single_solver(self, b=None):
        """Create solver for single-problem path using CompiledSolver.

        Args:
            b: Optional b vector. If provided and no b_sparsity_pattern was set at
               construction, its nonzero pattern is used for chordal decomposition.
        """
        b_sparsity_pattern = self._b_sparsity_pattern
        if b_sparsity_pattern is None and b is not None:
            b_sparsity_pattern = [abs(float(v)) > 0 for v in b]
        self._single_solver = _cpu_solver.CompiledSolver(
            n=self._n,
            m=self._m,
            P_row_offsets=self._P_row_offsets.tolist(),
            P_col_indices=self._P_col_indices.tolist(),
            A_row_offsets=self._A_row_offsets.tolist(),
            A_col_indices=self._A_col_indices.tolist(),
            cones=self._cones,
            settings=self._settings,
            enable_grad=self._enable_grad,
            dir_cones=self._dir_cones if self._dir_cones else None,
            b_sparsity_pattern=b_sparsity_pattern,
        )
        self._cached_single_P = None
        self._cached_single_A = None

    def _solve_single(
        self,
        P_values: np.ndarray,
        A_values: np.ndarray,
        q: np.ndarray,
        b: np.ndarray,
    ) -> Dict[str, Any]:
        """Solve a single problem using cached CompiledSolver.

        Single problem path:
        - Solver construction: lazy, done on first solve (symbolic factorization)
        - Equilibration: done here if P/A changed (not in setup() for single path)
        - Solve: actual solve with q, b
        """
        # Create solver on first call, passing b for chordal sparsity analysis
        if self._single_solver is None:
            self._create_single_solver(b=b)

        # Check if P/A have changed - skip setup if unchanged
        P_unchanged = self._cached_single_P is not None and np.array_equal(
            self._cached_single_P, P_values
        )
        A_unchanged = self._cached_single_A is not None and np.array_equal(
            self._cached_single_A, A_values
        )

        start_time = time.perf_counter()

        if not (P_unchanged and A_unchanged):
            # P or A changed: use new setup() method
            self._single_solver.setup([P_values.tolist()], [A_values.tolist()])
            self._cached_single_P = P_values.copy()
            self._cached_single_A = A_values.copy()

        # Use new solve() method
        solutions = self._single_solver.solve(
            [q.tolist()],
            [b.tolist()],
        )
        solve_time = time.perf_counter() - start_time

        sol = solutions[0]
        return {
            "x": np.array(sol.x),
            "s": np.array(sol.s),
            "z": np.array(sol.z),
            "z_x": np.array(sol.z_x, dtype=np.float64),
            "status": SolverStatus(int(sol.status)),
            "obj_val": sol.obj_val,
            "dual_obj_val": sol.obj_val_dual,
            "iterations": sol.iterations,
            "construction_time": sol.construction_time,
            "setup_time": sol.setup_time,
            "solve_time": sol.solve_time,
        }

    def _solve_batch(
        self,
        P_values: np.ndarray,
        A_values: np.ndarray,
        q: np.ndarray,
        b: np.ndarray,
        warm_x: Optional[list] = None,
        warm_z: Optional[list] = None,
        warm_s: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Solve batched problems.

        Handles both shared P/A (1D) and per-problem P/A (2D) cases.

        Work done:
        - Creates batch solver if not exists (lazy init for when batch_size wasn't provided)
        - Precomputes equilibration only if P/A changed since setup()
        - Actual solve with q, b
        """
        batch_size = q.shape[0]

        # Create CompiledSolver if needed (lazy init when batch_size wasn't provided at construction)
        if self._compiled_solver is None:
            self._create_compiled_solver()
            # Solver just created, need equilibration
            self._cached_shared_P = None
            self._cached_shared_A = None

        # Check if P/A have changed since last equilibration
        P_unchanged = self._cached_shared_P is not None and np.array_equal(
            self._cached_shared_P, P_values
        )
        A_unchanged = self._cached_shared_A is not None and np.array_equal(
            self._cached_shared_A, A_values
        )

        if not (P_unchanged and A_unchanged):
            if P_values.ndim == 1:
                # Shared P/A: use setup_shared() with 1D values
                self._compiled_solver.setup_shared(P_values.tolist(), A_values.tolist(), batch_size)
            else:
                # Per-problem P/A: use flat setup via buffer protocol
                P_flat = np.ascontiguousarray(P_values, dtype=np.float64).ravel()
                A_flat = np.ascontiguousarray(A_values, dtype=np.float64).ravel()
                self._compiled_solver.setup_flat(P_flat, A_flat, batch_size)
            self._cached_shared_P = P_values.copy()
            self._cached_shared_A = A_values.copy()

        # Flat solve path: pass numpy arrays directly via buffer protocol
        q_flat = np.ascontiguousarray(q, dtype=np.float64).ravel()
        b_flat = np.ascontiguousarray(b, dtype=np.float64).ravel()

        warm_x_flat = None
        warm_z_flat = None
        warm_s_flat = None
        if warm_x is not None:
            warm_x_flat = np.ascontiguousarray(warm_x, dtype=np.float64).ravel()
        if warm_z is not None:
            warm_z_flat = np.ascontiguousarray(warm_z, dtype=np.float64).ravel()
        if warm_s is not None:
            warm_s_flat = np.ascontiguousarray(warm_s, dtype=np.float64).ravel()

        result = self._compiled_solver.solve_flat(
            q_flat,
            b_flat,
            batch_size,
            warm_x_flat=warm_x_flat,
            warm_z_flat=warm_z_flat,
            warm_s_flat=warm_s_flat,
        )

        # Reshape flat arrays to (batch, dim)
        n, m = self._n, self._m
        z_x_flat = np.array(result.z_x, dtype=np.float64)
        if z_x_flat.size:
            z_x = z_x_flat.reshape(batch_size, -1)
        else:
            z_x = np.zeros((batch_size, 0), dtype=np.float64)
        return {
            "x": np.array(result.x).reshape(batch_size, n),
            "s": np.array(result.s).reshape(batch_size, m),
            "z": np.array(result.z).reshape(batch_size, m),
            "z_x": z_x,
            "status": [SolverStatus(s) for s in result.status],
            "obj_val": result.obj_val,
            "dual_obj_val": result.obj_val_dual,
            "iterations": result.iterations,
            "construction_time": result.construction_time,
            "setup_time": result.setup_time,
            "solve_time": result.solve_time,
        }

    def solve(
        self,
        q: np.ndarray,
        b: np.ndarray,
        warm_x: Optional[list] = None,
        warm_z: Optional[list] = None,
        warm_s: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Solve conic optimization problem(s).

        Requires setup() to be called first.

        Automatically detects single vs batched problems from input shape:
        - 1D inputs: Single problem, returns 1D results
        - 2D inputs: Batched problems, returns 2D results

        Args:
            q: Linear cost. Shape (n,) for single, (batch, n) for batched
            b: Constraint RHS. Shape (m,) for single, (batch, m) for batched
            warm_x: Optional warm start primal variables (list of lists)
            warm_z: Optional warm start dual variables (list of lists)
            warm_s: Optional warm start slack variables (list of lists)

        Returns:
            Dict with keys: x, z, s, status, obj_val, iterations, solve_time
            - Single problem: scalars/1D arrays
            - Batched: lists/2D arrays

        Raises:
            RuntimeError: If setup() was not called first.
        """
        # Check that matrices have been set
        if self._P_values is None or self._A_values is None:
            raise RuntimeError("setup() must be called before solve()")

        q = np.asarray(q, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # Detect single vs batched from input shape
        is_single = q.ndim == 1

        has_warm = warm_x is not None

        # Validate dimensions
        q_dim = q.shape[-1] if q.ndim > 0 else 0
        b_dim = b.shape[-1] if b.ndim > 0 else 0

        if q_dim != self._n:
            raise ValueError(f"q has dimension {q_dim}, expected {self._n}")
        if b_dim != self._m:
            raise ValueError(f"b has dimension {b_dim}, expected {self._m}")

        # Get stored matrix values
        P_values = self._P_values
        A_values = self._A_values

        # Single problem (q/b are 1D) — use _solve_single only when
        # P_values is also 1D (no batch dimension) and no warm start.
        # When the unified wrapper tiles P/A to 2D (batch_size=1), _solve_single
        # can't handle the nested list conversion, so we route through batch.
        if is_single and not has_warm and P_values.ndim == 1:
            return self._solve_single(P_values, A_values, q, b)

        # If single, reshape to batch for the batch path
        if is_single:
            q = q.reshape(1, -1)
            b = b.reshape(1, -1)

        # Batched problems (q/b are 2D)
        batch_size = q.shape[0]

        # Batched solve (handles both shared and per-problem P/A)
        result = self._solve_batch(
            P_values, A_values, q, b, warm_x=warm_x, warm_z=warm_z, warm_s=warm_s
        )

        # If we reshuffled from single to batch, squeeze back
        if is_single:
            zx_arr = result.get("z_x")
            if zx_arr is not None and zx_arr.size:
                z_x_single = zx_arr.squeeze(0)
            else:
                z_x_single = np.zeros(0, dtype=np.float64)
            return {
                "x": result["x"].squeeze(0),
                "s": result["s"].squeeze(0),
                "z": result["z"].squeeze(0),
                "z_x": z_x_single,
                "status": result["status"][0],
                "obj_val": result["obj_val"][0],
                "dual_obj_val": result["dual_obj_val"][0],
                "iterations": result["iterations"][0],
                "construction_time": result["construction_time"],
                "setup_time": result["setup_time"],
                "solve_time": result["solve_time"],
            }

        return result

    def backward(
        self,
        dx: np.ndarray,
        dz: Optional[np.ndarray] = None,
        ds: Optional[np.ndarray] = None,
        dz_x: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute gradients via implicit differentiation.

        Uses cached state from the last solve() call.

        Args:
            dx: Upstream gradient w.r.t. x (required)
            dz: Upstream gradient w.r.t. z (optional, defaults to zeros)
            ds: Upstream gradient w.r.t. s (optional, defaults to zeros)
            dz_x: Upstream gradient w.r.t. direct dual z_x. Same shape
                as the `z_x` field returned by solve(); leave as None to
                skip (equivalent to all-zero dz_x).

        Returns:
            Dict with keys: dP_values, dq, dA_values, db

        """
        if not self._enable_grad:
            raise RuntimeError("backward() requires enable_grad=True")

        # Determine which solver has cache
        single_has_cache = self._single_solver is not None and self._single_solver.has_cache()
        batch_has_cache = self._compiled_solver is not None and self._compiled_solver.has_cache()

        if not single_has_cache and not batch_has_cache:
            raise RuntimeError("Must call solve() before backward()")

        dx = np.asarray(dx)
        if dx.dtype != np.float64:
            raise TypeError(f"dx has dtype {dx.dtype}, requires float64")

        is_single = dx.ndim == 1

        # Default dz/ds to zeros
        if dz is None:
            dz = (
                np.zeros(self._m, dtype=np.float64)
                if is_single
                else np.zeros((dx.shape[0], self._m), dtype=np.float64)
            )
        else:
            dz = np.asarray(dz)
            if dz.dtype != np.float64:
                raise TypeError(f"dz has dtype {dz.dtype}, requires float64")

        if ds is None:
            ds = (
                np.zeros(self._m, dtype=np.float64)
                if is_single
                else np.zeros((dx.shape[0], self._m), dtype=np.float64)
            )
        else:
            ds = np.asarray(ds)
            if ds.dtype != np.float64:
                raise TypeError(f"ds has dtype {ds.dtype}, requires float64")

        # Choose solver based on which has cache
        solver = self._single_solver if single_has_cache else self._compiled_solver

        # Reshape to batch format
        if is_single:
            dx = dx.reshape(1, -1)
            dz = dz.reshape(1, -1)
            ds = ds.reshape(1, -1)

        batch_size = dx.shape[0]
        cache_size = solver.cache_size()
        if batch_size != cache_size:
            raise ValueError(f"Batch size mismatch: got {batch_size}, expected {cache_size}")

        # Use flat backward path via buffer protocol
        dx_flat = np.ascontiguousarray(dx, dtype=np.float64).ravel()
        ds_flat = np.ascontiguousarray(ds, dtype=np.float64).ravel()
        dz_flat = np.ascontiguousarray(dz, dtype=np.float64).ravel()

        dz_x_flat = None
        if dz_x is not None:
            dz_x_arr = np.asarray(dz_x, dtype=np.float64)
            if dz_x_arr.dtype != np.float64:
                raise TypeError(f"dz_x has dtype {dz_x_arr.dtype}, requires float64")
            if is_single and dz_x_arr.ndim == 1:
                dz_x_arr = dz_x_arr.reshape(1, -1)
            if dz_x_arr.size:
                dz_x_flat = np.ascontiguousarray(dz_x_arr, dtype=np.float64).ravel()

        computed = solver.backward_flat(dx_flat, ds_flat, dz_flat, batch_size, dz_x_flat=dz_x_flat)

        n, m = self._n, self._m
        nnz_P = self._nnz_P
        nnz_A = self._nnz_A
        result = {
            "dP_values": np.array(computed.dP_values).reshape(batch_size, nnz_P),
            "dq": np.array(computed.dq).reshape(batch_size, n),
            "dA_values": np.array(computed.dA_values).reshape(batch_size, nnz_A),
            "db": np.array(computed.db).reshape(batch_size, m),
        }

        # Debug: expose smoothed iterates if available
        if hasattr(computed, "debug_smoothing_x") and len(computed.debug_smoothing_x) > 0:
            result["debug_smoothing_x"] = np.array(computed.debug_smoothing_x).reshape(
                batch_size, n
            )
            result["debug_smoothing_z"] = np.array(computed.debug_smoothing_z).reshape(
                batch_size, m
            )
            result["debug_smoothing_s"] = np.array(computed.debug_smoothing_s).reshape(
                batch_size, m
            )

        # Squeeze if input was 1D
        if is_single:
            result = {k: v[0] for k, v in result.items()}

        return result

    def backward_with_data_flat(
        self,
        dx_flat: np.ndarray,
        ds_flat: np.ndarray,
        dz_flat: np.ndarray,
        P_values_flat: np.ndarray,
        A_values_flat: np.ndarray,
        q_flat: np.ndarray,
        b_flat: np.ndarray,
        x_flat: np.ndarray,
        z_flat: np.ndarray,
        s_flat: np.ndarray,
        batch_size: int,
        z_x_flat: Optional[np.ndarray] = None,
        dz_x_flat: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute gradients from explicitly provided problem data and solution.

        Unlike backward(), this does not require a prior solve() call.
        Equilibrates internally and computes the adjoint KKT solve directly.

        All inputs are flat contiguous arrays: batch_size * per-problem-size.

        For direct cones, also supply `z_x_flat` (the saved direct
        cone duals from `solve()`) and optionally `dz_x_flat` (upstream
        gradient on `z_x`).

        Returns:
            Dict with keys: dP_values, dq, dA_values, db
        """
        if not self._enable_grad:
            raise RuntimeError("backward_with_data_flat() requires enable_grad=True")

        # Expand shared P/A to per-batch if needed
        nnz_P = self._nnz_P
        nnz_A = self._nnz_A
        if len(P_values_flat) == nnz_P and batch_size > 1:
            P_values_flat = np.tile(P_values_flat, batch_size)
        if len(A_values_flat) == nnz_A and batch_size > 1:
            A_values_flat = np.tile(A_values_flat, batch_size)

        # Get or create a CompiledSolver for this batch size
        solver = self._get_or_create_compiled_solver(batch_size)

        computed = solver.backward_with_data_flat(
            dx_flat,
            ds_flat,
            dz_flat,
            P_values_flat,
            A_values_flat,
            q_flat,
            b_flat,
            x_flat,
            z_flat,
            s_flat,
            batch_size,
            z_x_flat=z_x_flat,
            dz_x_flat=dz_x_flat,
        )

        n, m = self._n, self._m
        nnz_P = self._nnz_P
        nnz_A = self._nnz_A
        result = {
            "dP_values": np.array(computed.dP_values).reshape(batch_size, nnz_P),
            "dq": np.array(computed.dq).reshape(batch_size, n),
            "dA_values": np.array(computed.dA_values).reshape(batch_size, nnz_A),
            "db": np.array(computed.db).reshape(batch_size, m),
        }

        if batch_size == 1:
            result = {k: v[0] for k, v in result.items()}

        return result

    def _get_or_create_compiled_solver(self, batch_size: int):
        """Get or create a CompiledSolver for the given batch size."""
        if batch_size == 1 and self._single_solver is not None:
            return self._single_solver
        if self._compiled_solver is not None:
            return self._compiled_solver
        # Create and cache a CompiledSolver (enable_grad=True for backward_with_data)
        if not hasattr(self, "_bwd_compiled_solver"):
            self._bwd_compiled_solver = _cpu_solver.CompiledSolver(
                n=self._n,
                m=self._m,
                P_row_offsets=self._P_row_offsets.tolist(),
                P_col_indices=self._P_col_indices.tolist(),
                A_row_offsets=self._A_row_offsets.tolist(),
                A_col_indices=self._A_col_indices.tolist(),
                cones=self._cones,
                settings=self._settings,
                enable_grad=True,
            )
        return self._bwd_compiled_solver

    def setup_grad(self, batch_size: Optional[int] = None):
        """Setup for gradient computation (optional, for pre-allocation)."""
        effective_batch = batch_size or self._expected_batch_size or 1
        self._grad_dx_buffer = np.zeros((effective_batch, self._n), dtype=np.float64)
        self._grad_dz_buffer = np.zeros((effective_batch, self._m), dtype=np.float64)
        self._grad_ds_buffer = np.zeros((effective_batch, self._m), dtype=np.float64)
        self._grad_initialized = True

    @property
    def grad_initialized(self) -> bool:
        return self._grad_initialized

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


class DirectXSolverCpu:
    """Direct CPU solver (single or batched).

    Thin wrapper around `_cpu_solver.CompiledSolver` configured with
    `dir_cones`. The CompiledSolver does symbolic factorisation **once**
    at construction; `setup()` updates equilibration when matrix values
    change, and `solve()` reuses the precompiled structure for every
    call. This matches the slack-only CompiledSolver path and avoids
    the ~310 ms-per-call reconstruction of the previous
    `DefaultSolver.new_with_xcones` implementation.

    External API (`__init__`, `.setup()`, `.solve()`, `.backward()`)
    is unchanged.
    """

    def __init__(
        self,
        n: int,
        m: int,
        P_row_offsets,
        P_col_indices,
        A_row_offsets,
        A_col_indices,
        cones,
        settings=None,
        batch_size: Optional[int] = None,
        enable_grad: bool = False,
    ):
        self._batch_size = int(batch_size) if batch_size is not None else 1
        if self._batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._enable_grad = enable_grad
        self._n = n
        self._m = m
        self._P_row_offsets = np.asarray(P_row_offsets, dtype=np.int64)
        self._P_col_indices = np.asarray(P_col_indices, dtype=np.int64)
        self._A_row_offsets = np.asarray(A_row_offsets, dtype=np.int64)
        self._A_col_indices = np.asarray(A_col_indices, dtype=np.int64)
        self._nnz_P = len(P_col_indices)
        self._nnz_A = len(A_col_indices)

        # Separate slack cones and direct cones.
        if hasattr(cones, "num_zero_cones"):
            self._slack_cones = cones_to_cpu(cones)
            self._dir_cones = _dir_cones_to_cpu(cones)
        else:
            self._slack_cones = cones
            self._dir_cones = []

        self._settings = (
            _cpu_solver.DefaultSettings() if settings is None else settings_to_cpu(settings)
        )

        # Eagerly construct the underlying CompiledSolver (one-time
        # symbolic factorisation). Subsequent `setup()` / `solve()` calls
        # only re-equilibrate and re-factorise numerically.
        self._compiled_solver = _cpu_solver.CompiledSolver(
            n=self._n,
            m=self._m,
            P_row_offsets=self._P_row_offsets.tolist(),
            P_col_indices=self._P_col_indices.tolist(),
            A_row_offsets=self._A_row_offsets.tolist(),
            A_col_indices=self._A_col_indices.tolist(),
            cones=self._slack_cones,
            settings=self._settings,
            num_threads=1,
            enable_grad=enable_grad,
            dir_cones=self._dir_cones if self._dir_cones else None,
        )

        # Matrix values set via `.setup()`.
        self._P_values: Optional[np.ndarray] = None
        self._A_values: Optional[np.ndarray] = None

    def setup(self, P_values, A_values) -> None:
        """Set matrix values + run equilibration. Accepts either:
        - 1-D `(nnz,)` — shared across the batch (tiled internally for
          `batch_size > 1`)
        - 2-D `(batch_size, nnz)` — per-problem
        """
        P_values = np.asarray(P_values, dtype=np.float64)
        A_values = np.asarray(A_values, dtype=np.float64)
        for name, arr, nnz in (
            ("P_values", P_values, self._nnz_P),
            ("A_values", A_values, self._nnz_A),
        ):
            if arr.ndim == 1:
                if arr.shape[0] != nnz:
                    raise ValueError(f"{name} has {arr.shape[0]} elements, expected {nnz}")
            elif arr.ndim == 2:
                if arr.shape[0] != self._batch_size or arr.shape[1] != nnz:
                    raise ValueError(
                        f"{name} shape {arr.shape} must be "
                        f"({self._batch_size}, {nnz}) or ({nnz},)"
                    )
            else:
                raise ValueError(
                    f"{name} must be 1-D (nnz,) or 2-D (batch_size, nnz), " f"got shape {arr.shape}"
                )
        self._P_values = P_values
        self._A_values = A_values

        # Tile shared 1-D values to a flat (batch_size * nnz,) buffer so we
        # can call setup_flat uniformly.
        if P_values.ndim == 1:
            P_flat = np.broadcast_to(P_values, (self._batch_size, self._nnz_P)).reshape(-1).copy()
        else:
            P_flat = np.ascontiguousarray(P_values, dtype=np.float64).ravel()
        if A_values.ndim == 1:
            A_flat = np.broadcast_to(A_values, (self._batch_size, self._nnz_A)).reshape(-1).copy()
        else:
            A_flat = np.ascontiguousarray(A_values, dtype=np.float64).ravel()

        self._compiled_solver.setup_flat(P_flat, A_flat, self._batch_size)

    def solve(self, q, b, **kwargs) -> Dict[str, Any]:
        if self._P_values is None or self._A_values is None:
            raise RuntimeError("Must call setup(P, A) before solve()")

        # Warm-start vectors (each None, or shape (batch, dim)); flattened
        # to contiguous 1-D buffers as `solve_flat` expects. `warm_z_x` is
        # the direct cone dual; omitting it cold-inits the cone block.
        def _warm_flat(name):
            v = kwargs.get(name)
            if v is None:
                return None
            return np.ascontiguousarray(np.asarray(v, dtype=np.float64).reshape(-1))

        warm_x = _warm_flat("warm_x")
        warm_z = _warm_flat("warm_z")
        warm_s = _warm_flat("warm_s")
        warm_z_x = _warm_flat("warm_z_x")
        q_arr = np.asarray(q, dtype=np.float64)
        b_arr = np.asarray(b, dtype=np.float64)

        # Single-problem path: 1-D q, b.
        if self._batch_size == 1 and q_arr.ndim <= 1 and b_arr.ndim <= 1:
            q_1d = np.ascontiguousarray(q_arr.reshape(-1), dtype=np.float64)
            b_1d = np.ascontiguousarray(b_arr.reshape(-1), dtype=np.float64)
            if q_1d.size != self._n:
                raise ValueError(f"q has {q_1d.size} elements, expected {self._n}")
            if b_1d.size != self._m:
                raise ValueError(f"b has {b_1d.size} elements, expected {self._m}")

            start = time.perf_counter()
            sol = self._compiled_solver.solve_flat(q_1d, b_1d, 1, warm_x, warm_z, warm_s, warm_z_x)
            solve_time = time.perf_counter() - start

            return {
                "x": np.asarray(sol.x, dtype=np.float64),
                "s": np.asarray(sol.s, dtype=np.float64),
                "z": np.asarray(sol.z, dtype=np.float64),
                "z_x": np.asarray(sol.z_x, dtype=np.float64),
                "status": SolverStatus(int(sol.status[0])),
                "obj_val": float(sol.obj_val[0]),
                "dual_obj_val": float(sol.obj_val_dual[0]),
                "iterations": int(sol.iterations[0]),
                "construction_time": sol.construction_time,
                "setup_time": sol.setup_time,
                "solve_time": solve_time,
            }

        # Batched path: 2-D q/b with shape (batch_size, n/m).
        if q_arr.ndim != 2 or q_arr.shape != (self._batch_size, self._n):
            raise ValueError(f"q shape {q_arr.shape} must be ({self._batch_size}, {self._n})")
        if b_arr.ndim != 2 or b_arr.shape != (self._batch_size, self._m):
            raise ValueError(f"b shape {b_arr.shape} must be ({self._batch_size}, {self._m})")
        q_flat = np.ascontiguousarray(q_arr, dtype=np.float64).ravel()
        b_flat = np.ascontiguousarray(b_arr, dtype=np.float64).ravel()

        start = time.perf_counter()
        sol = self._compiled_solver.solve_flat(
            q_flat, b_flat, self._batch_size, warm_x, warm_z, warm_s, warm_z_x
        )
        solve_time = time.perf_counter() - start

        x_out = np.asarray(sol.x, dtype=np.float64).reshape(self._batch_size, self._n)
        s_out = np.asarray(sol.s, dtype=np.float64).reshape(self._batch_size, self._m)
        z_out = np.asarray(sol.z, dtype=np.float64).reshape(self._batch_size, self._m)
        z_x_flat = np.asarray(sol.z_x, dtype=np.float64)
        if z_x_flat.size:
            total_xn = z_x_flat.size // self._batch_size
            z_x_out = z_x_flat.reshape(self._batch_size, total_xn)
        else:
            z_x_out = np.zeros((self._batch_size, 0), dtype=np.float64)
        statuses = [SolverStatus(int(s)) for s in sol.status]
        obj_vals = np.asarray(sol.obj_val, dtype=np.float64)
        dual_obj_vals = np.asarray(sol.obj_val_dual, dtype=np.float64)
        iters = np.asarray(sol.iterations, dtype=np.int64)

        return {
            "x": x_out,
            "s": s_out,
            "z": z_out,
            "z_x": z_x_out,
            "status": statuses,
            "obj_val": obj_vals,
            "dual_obj_val": dual_obj_vals,
            "iterations": iters,
            "construction_time": sol.construction_time,
            "setup_time": sol.setup_time,
            "solve_time": solve_time,
        }

    def backward(self, dx, dz=None, ds=None, dz_x=None) -> Dict[str, Any]:
        """Adjoint differentiation via `CompiledSolver.backward_flat`
        (IFT path with cached state)."""
        if not self._enable_grad:
            raise RuntimeError("backward() requires enable_grad=True at construction")
        dx_flat = np.ascontiguousarray(np.asarray(dx, dtype=np.float64).reshape(-1))
        if dz is None:
            dz_flat = np.zeros(self._batch_size * self._m, dtype=np.float64)
        else:
            dz_flat = np.ascontiguousarray(np.asarray(dz, dtype=np.float64).reshape(-1))
        if ds is None:
            ds_flat = np.zeros(self._batch_size * self._m, dtype=np.float64)
        else:
            ds_flat = np.ascontiguousarray(np.asarray(ds, dtype=np.float64).reshape(-1))
        dz_x_flat = (
            np.ascontiguousarray(np.asarray(dz_x, dtype=np.float64).reshape(-1))
            if dz_x is not None
            else None
        )
        result = self._compiled_solver.backward_flat(
            dx_flat,
            ds_flat,
            dz_flat,
            self._batch_size,
            dz_x_flat,
        )
        return {
            "dP_values": np.asarray(result.dP_values, dtype=np.float64),
            "dq": np.asarray(result.dq, dtype=np.float64),
            "dA_values": np.asarray(result.dA_values, dtype=np.float64),
            "db": np.asarray(result.db, dtype=np.float64),
        }


__all__ = [
    "Solver",
    "DirectXSolverCpu",
    "Cones",
    "SolverStatus",
    "cones_to_cpu",
    "settings_to_cpu",
]
