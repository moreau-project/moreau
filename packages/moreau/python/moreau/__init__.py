"""Moreau: Conic optimization solver with CPU and optional accelerator backends.

Usage:
    pip install moreau           # CPU only
    pip install moreau[cuda]     # CPU + GPU

Example (Solver - single problem):
    import moreau
    import numpy as np
    from scipy import sparse

    # Problem data
    P = sparse.diags([1.0, 1.0], format='csr')
    q = np.array([2.0, 1.0])
    A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, 0.7, 0.7])

    cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

    # Create solver with all problem data
    solver = moreau.Solver(P, q, A, b, cones=cones)
    solution = solver.solve()
    print(solution.x, solver.info.status)

Example (CompiledSolver - multiple problems):
    import moreau
    import numpy as np

    # Define problem structure
    cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
    settings = moreau.Settings(device='cpu', batch_size=4)

    solver = moreau.CompiledSolver(
        n=2, m=3,
        P_row_offsets=[0, 1, 2], P_col_indices=[0, 1],
        A_row_offsets=[0, 2, 3, 4], A_col_indices=[0, 1, 0, 1],
        cones=cones, settings=settings
    )

    # Set matrix values for batch
    P_values = [[1.0, 1.0]] * 4
    A_values = [[1.0, 1.0, 1.0, 1.0]] * 4
    solver.setup(P_values, A_values)

    # Solve batch
    qs = [[2.0, 1.0]] * 4
    bs = [[1.0, 0.7, 0.7]] * 4
    solution = solver.solve(qs, bs)
    print(solution.x.shape, solver.info.status[0])
"""

__version__ = "0.4.0-beta.1"

# Import CPU backend types
import numpy as np
import time
import warnings
from scipy import sparse
from moreau_cpu import _cpu_solver

# Shared types
from ._types import (
    SolverType,
    SolverStatus,
    Cones,
    XConeSpec,
    Settings,
    IPMSettings,
    ActiveSetSettings,
    WarmStart,
    BatchedWarmStart,
    SolveInfo,
    Solution,
    BatchedSolveInfo,
    BatchedSolution,
    TuneResult,
    _normalize_status,
    _AUTO_TUNE_MARGIN,
    _should_retry_cold,
    _warn_warm_retry,
)
from typing import Optional, Sequence, Union

import numpy.typing as npt
from scipy.sparse import sparray, spmatrix

_PMatrixLike = Union[sparray, spmatrix, np.ndarray]


# Device registration and discovery (public API)
from ._backend import (
    available_devices,
    device_available,
    default_device,
    device_error,
    set_default_device,
    get_default_device,
    torch_available,
    jax_available,
)

# Internal imports (not part of public API, but needed by this module)
from ._backend import (
    get_device_component as _get_device_component,
    auto_tune_candidates as _auto_tune_candidates,
    benchmark_candidates as _benchmark_candidates,
)

# Validation helpers (private, re-exported here for backend modules)
from ._validation import (
    _to_csr,
    _validate_problem_dimensions,
    _validate_csr_structure,
    _validate_smoothed_diff_cones,
    _validate_setup_values,
    _validate_solve_inputs,
    _validate_P_symmetry,
    _validate_P_sparsity_pattern_symmetric,
    _validate_P_values_symmetry,
)

# Direct-x cone dispatch helpers (private). `_resolve_solver_type` and
# `_require_x_cones_compatible` are also imported by the torch/jax modules.
from ._dispatch import (
    _resolve_solver_type,
    _resolve_settings_and_device,
    _require_x_cones_compatible,
    _has_x_cones,
    _require_cpu_device_for_x_cones,
)


def _merge_failed_batch_field(warm_val, cold_val, failed_idx):
    """Merge a per-batch info field after a partial warm-start cold retry.

    For failed batch indices the cold value replaces the warm value; all
    other elements keep their warm value. ``obj_val`` and ``iterations`` are
    scalars when ``batch_size == 1`` and array-like otherwise, so both shapes
    are handled. The warm container is mutated in place when array-like.
    """
    if warm_val is None or cold_val is None:
        return warm_val
    if np.isscalar(warm_val) or np.ndim(warm_val) == 0:
        return cold_val if failed_idx else warm_val
    for i in failed_idx:
        warm_val[i] = cold_val[i]
    return warm_val


# Initialize BLAS/LAPACK
_cpu_solver.force_load_blas_lapack()

_cvxpy_soc_warning_issued = False


def _warn_cvxpy_soc_if_needed(cones):
    """Warn if CVXPY <= 1.8.1 is installed and the problem has SOC cones.

    CVXPY 1.8.2+ generates significantly better SOC formulations for Moreau.
    This warning is issued at most once per process.
    """
    global _cvxpy_soc_warning_issued
    if _cvxpy_soc_warning_issued:
        return
    if not cones.so_cone_dims:
        return
    import sys

    cvxpy = sys.modules.get("cvxpy")
    if cvxpy is None:
        return
    try:
        version_tuple = tuple(int(x) for x in cvxpy.__version__.split(".")[:3])
        if version_tuple <= (1, 8, 1):
            _cvxpy_soc_warning_issued = True
            warnings.warn(
                f"CVXPY {cvxpy.__version__} detected with SOC constraints. "
                "Upgrade to CVXPY >= 1.8.2 for significantly improved SOC "
                "performance with Moreau: pip install 'cvxpy>=1.8.2'",
                stacklevel=3,
            )
    except Exception:
        pass


class Solver:
    """Single-problem conic optimization solver.

    All problem data is provided in the constructor, then call solve().

    Problem Formulation:
        minimize    (1/2)x'Px + q'x
        subject to  Ax + s = b
                    x ∈ K1,  s ∈ K2

    K2 constrains the slack s; K1 constrains x directly (direct-x cones).

    Args:
        P: Quadratic objective matrix (scipy sparse or numpy array).
            Must be full symmetric (both upper and lower triangles).
        q: Linear objective vector
        A: Constraint matrix (scipy sparse or numpy array)
        b: Constraint RHS vector
        cones: Cone specification (moreau.Cones object)
        settings: Optional solver settings (moreau.Settings object)

    Example:
        >>> import moreau
        >>> from scipy import sparse
        >>> import numpy as np
        >>>
        >>> P = sparse.diags([1.0, 1.0], format='csr')
        >>> q = np.array([2.0, 1.0])
        >>> A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        >>> b = np.array([1.0, 0.7, 0.7])
        >>> cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        >>>
        >>> solver = moreau.Solver(P, q, A, b, cones)
        >>> solution = solver.solve()
        >>> print(solution.x, solver.info.status)
    """

    def __init__(
        self,
        P: _PMatrixLike,
        q: npt.ArrayLike,
        A: _PMatrixLike,
        b: npt.ArrayLike,
        cones: Cones,
        settings: Optional[Settings] = None,
    ) -> None:
        # Validate problem dimensions first (before any other processing)
        _validate_problem_dimensions(P, q, A, b, cones)

        # Direct-x cones dispatch:
        #   - CPU: bypass the CompiledSolver-based backend and go through
        #     `DirectXSolverCpu` (wraps `DefaultSolver.new_with_xcones`,
        #     includes backward via HSDE unfold).
        #   - CUDA: nonneg direct-x forward is handled natively by the
        #     CompiledSolver device path (C++ side threads x_cones through
        #     Cones/KKT/IPM). Fall through to the normal device init.
        if _has_x_cones(cones):
            _require_cpu_device_for_x_cones(settings, cones)
            dev = getattr(settings, "device", None) if settings is not None else None
            if dev in (None, "auto", "cpu"):
                self._init_with_x_cones(P, q, A, b, cones, settings)
                return
            # CUDA + nonneg x-cones: fall through to the standard device
            # init below. SOC x-cones on CUDA are blocked at
            # _require_cpu_device_for_x_cones (and at Cones::initialize
            # on the C++ side).

        # Validate smoothed diff is only used with LP cones (zero + nonneg)
        _validate_smoothed_diff_cones(settings, cones)

        # Check for old CVXPY with SOC cones
        _warn_cvxpy_soc_if_needed(cones)

        n = len(q)
        nnz_P = P.nnz if sparse.issparse(P) else np.count_nonzero(P)
        nnz_A = A.nnz if sparse.issparse(A) else np.count_nonzero(A)
        settings, device = _resolve_settings_and_device(
            settings,
            n=n,
            m=len(b),
            cones=cones,
            nnz_P=nnz_P,
            nnz_A=nnz_A,
            batch_size=1,
        )

        self._device = device
        self._settings = settings
        self._enable_grad = settings.enable_grad

        # Validate P matrix symmetry
        # Both backends require full symmetric P (not upper/lower triangle only)
        _validate_P_symmetry(P)

        # Convert matrices to CSR format
        P_row_offsets, P_col_indices, P_values = _to_csr(P)
        A_row_offsets, A_col_indices, A_values = _to_csr(A)

        # Store problem data
        q = np.asarray(q, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        self._n = len(q)
        self._m = len(b)
        self._nnz_P = len(P_values)
        self._nnz_A = len(A_values)

        # Store for solve
        self._P_values = P_values
        self._A_values = A_values
        self._q = q
        self._b = b
        self._cones = cones

        # Store CSR structure
        self._P_row_offsets = P_row_offsets
        self._P_col_indices = P_col_indices
        self._A_row_offsets = A_row_offsets
        self._A_col_indices = A_col_indices

        construction_start = time.perf_counter()
        self._init_device(device, cones, settings, settings.enable_grad)
        self._construction_time = time.perf_counter() - construction_start

        # Last solve info (populated after solve())
        self._info: Optional[SolveInfo] = None

    def _init_with_x_cones(self, P, q, A, b, cones, settings):
        """Dispatch path for problems with direct-x cones (CPU).

        Routes through `DirectXSolverCpu`, which now wraps
        `_cpu_solver.CompiledSolver` with `x_cones`. Symbolic
        factorisation runs once at construction; subsequent solves
        reuse the precompiled structure. Backward via `enable_grad=True`
        flows through `CompiledSolver.backward_flat`.
        """
        if settings is None:
            settings = Settings()
        # Store the same fields the normal __init__ would so the rest of
        # the Solver API (`.info`, `.solve()`) works unchanged.
        self._device = "cpu"
        self._settings = settings
        self._enable_grad = bool(getattr(settings, "enable_grad", False))
        _validate_P_symmetry(P)
        P_row_offsets, P_col_indices, P_values = _to_csr(P)
        A_row_offsets, A_col_indices, A_values = _to_csr(A)
        q = np.asarray(q, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        self._n = len(q)
        self._m = len(b)
        self._nnz_P = len(P_values)
        self._nnz_A = len(A_values)
        self._P_values = P_values
        self._A_values = A_values
        self._q = q
        self._b = b
        self._cones = cones
        self._P_row_offsets = P_row_offsets
        self._P_col_indices = P_col_indices
        self._A_row_offsets = A_row_offsets
        self._A_col_indices = A_col_indices
        self._info: Optional[SolveInfo] = None

        from moreau_cpu._cpu import DirectXSolverCpu

        construction_start = time.perf_counter()
        self._impl = DirectXSolverCpu(
            self._n,
            self._m,
            self._P_row_offsets,
            self._P_col_indices,
            self._A_row_offsets,
            self._A_col_indices,
            cones,
            settings,
            batch_size=1,
            enable_grad=self._enable_grad,
        )
        self._construction_time = time.perf_counter() - construction_start
        setup_start = time.perf_counter()
        self._impl.setup(self._P_values, self._A_values)
        self._setup_time = time.perf_counter() - setup_start
        # Device-specific storage expected by other Solver methods.
        self._device_cones = cones
        self._device_settings = settings

    def _init_device(self, device, cones, settings, enable_grad):
        """Initialize a registered device backend."""
        # Active-set solver: use dedicated class directly
        if settings is not None and settings.solver == SolverType.ACTIVE_SET:
            from moreau_cpu._cpu import ActiveSetSolver

            self._device_cones = cones
            self._device_settings = settings
            self._impl = ActiveSetSolver(
                self._n,
                self._m,
                self._P_row_offsets,
                self._P_col_indices,
                self._A_row_offsets,
                self._A_col_indices,
                cones,
                settings,
                batch_size=1,
                enable_grad=enable_grad,
            )
            setup_start = time.perf_counter()
            self._impl.setup(self._P_values, self._A_values)
            self._setup_time = time.perf_counter() - setup_start
            return

        DeviceSolver = _get_device_component(device, "solver")
        if DeviceSolver is None:
            raise RuntimeError(
                f"No solver registered for device '{device}'. "
                f"Available devices: {available_devices()}"
            )
        DeviceSettings = _get_device_component(device, "settings")
        cones_converter = _get_device_component(device, "cones_converter")
        settings_converter = _get_device_component(device, "settings_converter")

        # Convert Cones using registered converter
        if cones_converter is not None:
            device_cones = cones_converter(cones)
        else:
            device_cones = cones
        self._device_cones = device_cones

        # Convert Settings using registered converter
        if settings is None:
            device_settings = DeviceSettings() if DeviceSettings else None
            if device_settings is not None and hasattr(device_settings, "verbose"):
                device_settings.verbose = True
        elif settings_converter is not None:
            device_settings = settings_converter(settings)
        else:
            device_settings = settings
        self._device_settings = device_settings

        # Compute b_sparsity_pattern from b for chordal decomposition at construction
        b_sparsity_pattern = [abs(float(v)) > 0 for v in self._b]

        # Initialize unified solver (handles both single and batch internally)
        # We use batch_size=1 for single problem
        self._impl = DeviceSolver(
            self._n,
            self._m,
            self._P_row_offsets,
            self._P_col_indices,
            self._A_row_offsets,
            self._A_col_indices,
            device_cones,
            device_settings,
            batch_size=1,
            enable_grad=enable_grad,
            b_sparsity_pattern=b_sparsity_pattern,
        )

        # Setup with the matrix values
        setup_start = time.perf_counter()
        self._impl.setup(self._P_values, self._A_values)
        self._setup_time = time.perf_counter() - setup_start

    def solve(self, warm_start: Optional[WarmStart] = None) -> Solution:
        """Solve the optimization problem.

        Args:
            warm_start: Optional WarmStart from a previous solve
                (e.g. ``solution.to_warm_start()``). If the warm-started solve
                fails, it is automatically retried without warm start.

        Returns:
            Solution: primal/dual solution vectors (x, z, s)

        Note:
            Solver metadata (status, timing, iterations) is available via solver.info
            after calling solve().
        """
        kwargs = {}
        if warm_start is not None:
            if not isinstance(warm_start, WarmStart):
                raise TypeError(
                    f"warm_start must be a WarmStart, " f"got {type(warm_start).__name__}"
                )

            warm_x = np.asarray(warm_start.x, dtype=np.float64)
            warm_z = np.asarray(warm_start.z, dtype=np.float64)
            warm_s = np.asarray(warm_start.s, dtype=np.float64)

            if warm_x.shape != (self._n,):
                raise ValueError(f"warm_start.x shape {warm_x.shape}, expected ({self._n},)")
            if warm_z.shape != (self._m,):
                raise ValueError(f"warm_start.z shape {warm_z.shape}, expected ({self._m},)")
            if warm_s.shape != (self._m,):
                raise ValueError(f"warm_start.s shape {warm_s.shape}, expected ({self._m},)")

            # Backend uses batch_size=1 internally, reshape to (1, dim)
            kwargs["warm_x"] = warm_x.reshape(1, -1)
            kwargs["warm_z"] = warm_z.reshape(1, -1)
            kwargs["warm_s"] = warm_s.reshape(1, -1)

            # Direct-x dual: thread warm_z_x through when the WarmStart
            # carries it. Backends that haven't been updated to accept
            # the kwarg fall back via the TypeError handler below.
            zx = getattr(warm_start, "z_x", None)
            if zx is not None:
                zx_arr = np.asarray(zx, dtype=np.float64)
                if zx_arr.size > 0:
                    total_x_dim = sum(
                        len(getattr(xc, "indices", []))
                        for xc in (getattr(self._cones, "x_cones", None) or [])
                    )
                    if zx_arr.shape != (total_x_dim,):
                        raise ValueError(
                            f"warm_start.z_x shape {zx_arr.shape}, expected "
                            f"({total_x_dim},) (sum of x_cones[*].indices length)"
                        )
                    kwargs["warm_z_x"] = zx_arr.reshape(1, -1)

        try:
            result = self._impl.solve(self._q, self._b, **kwargs)
        except TypeError as e:
            if "warm_z_x" in str(e) and "warm_z_x" in kwargs:
                del kwargs["warm_z_x"]
                result = self._impl.solve(self._q, self._b, **kwargs)
            else:
                raise

        # Extract timing info from result, with fallbacks
        # Use self._construction_time if backend returns 0.0 (not tracked there)
        construction_time = result.get("construction_time", 0.0)
        if construction_time == 0.0:
            construction_time = self._construction_time
        setup_time = result.get("setup_time", 0.0)
        if setup_time == 0.0 and hasattr(self, "_setup_time"):
            setup_time = self._setup_time
        solve_time = result.get("solve_time", 0.0)

        status = _normalize_status(result["status"])

        solution = Solution(
            x=result["x"],
            z=result["z"],
            s=result["s"],
            z_x=result.get("z_x"),
        )

        self._info = SolveInfo(
            status=status,
            obj_val=float(result["obj_val"]),
            iterations=result["iterations"],
            solve_time=solve_time,
            setup_time=setup_time,
            construction_time=construction_time,
        )

        # Auto-retry cold if warm start may have caused the failure.
        # Helpers live in _types.py so all 4 solvers agree on the
        # status set + warning text.
        ipm = self._settings.ipm_settings
        if warm_start is not None and _should_retry_cold(status, ipm):
            _warn_warm_retry([0], status)
            result = self._impl.solve(self._q, self._b)
            construction_time = result.get("construction_time", 0.0)
            if construction_time == 0.0:
                construction_time = self._construction_time
            setup_time = result.get("setup_time", 0.0)
            if setup_time == 0.0 and hasattr(self, "_setup_time"):
                setup_time = self._setup_time
            solve_time = result.get("solve_time", 0.0)
            status = _normalize_status(result["status"])
            solution = Solution(
                x=result["x"],
                z=result["z"],
                s=result["s"],
                z_x=result.get("z_x"),
            )
            self._info = SolveInfo(
                status=status,
                obj_val=float(result["obj_val"]),
                iterations=result["iterations"],
                solve_time=solve_time,
                setup_time=setup_time,
                construction_time=construction_time,
            )

        return solution

    def backward(
        self,
        dx: npt.ArrayLike,
        dz: Optional[npt.ArrayLike] = None,
        ds: Optional[npt.ArrayLike] = None,
        dz_x: Optional[npt.ArrayLike] = None,
    ) -> dict:
        """Compute gradients via implicit differentiation using cached state.

        Pass `dz_x` (shape matching `Solution.z_x`) to backprop through the
        direct-x cone duals. Defaults to None (no upstream on `z_x`).
        Both CPU and CUDA backends support dz_x.

        Returns:
            dict with keys ``dq``, ``db``, ``dP_values``, ``dA_values``,
            each mapped to a numpy array of the corresponding shape.
        """
        return self._impl.backward(dx, dz, ds, dz_x=dz_x)

    @property
    def device(self) -> str:
        return self._device

    @property
    def n(self) -> int:
        return self._n

    @property
    def m(self) -> int:
        return self._m

    @property
    def construction_time(self) -> float:
        """Time spent constructing solver structure (seconds)."""
        return self._construction_time

    @property
    def info(self) -> Optional[SolveInfo]:
        """Metadata from the last solve() call.

        Returns None if solve() has not been called yet.

        Contains:
            - status: SolverStatus enum
            - obj_val: Objective value at solution
            - iterations: Number of IPM iterations
            - solve_time: Time in IPM iterations (seconds)
            - setup_time: Time setting matrix values (seconds)
            - construction_time: Time constructing solver (seconds)
        """
        return self._info


class CompiledSolver:
    """Compiled solver for multiple conic optimization problems with shared structure.

    Problem structure is defined at construction, matrix values via setup(),
    and solve parameters (q, b) via solve().

    Three-step API pattern:
    1. Construct with structure (dimensions, sparsity patterns, cones)
    2. setup(P_values, A_values) - Set matrix values for batch
    3. solve(qs, bs) - Solve with per-problem parameters

    Args:
        n: Number of primal variables
        m: Number of constraints
        P_row_offsets: CSR row pointers for P matrix (must be full symmetric)
        P_col_indices: CSR column indices for P matrix (must be full symmetric)
        A_row_offsets: CSR row pointers for A matrix
        A_col_indices: CSR column indices for A matrix
        cones: Cone specification (moreau.Cones object)
        settings: Solver settings (moreau.Settings object). Controls device, batch_size,
                  enable_grad, tolerances, etc. Defaults to Settings() if not provided.
        b_sparsity_pattern: Optional boolean mask of length m indicating which entries of b are
                    structurally nonzero. Enables chordal decomposition for PSD cones at
                    construction time. If None, a conservative fallback is used.

    Example:
        >>> import moreau
        >>> import numpy as np
        >>>
        >>> cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        >>> settings = moreau.Settings(batch_size=4)
        >>> solver = moreau.CompiledSolver(
        ...     n=2, m=3,
        ...     P_row_offsets=[0, 1, 2], P_col_indices=[0, 1],
        ...     A_row_offsets=[0, 2, 3, 4], A_col_indices=[0, 1, 0, 1],
        ...     cones=cones, settings=settings
        ... )
        >>>
        >>> P_values = [[1.0, 1.0]] * 4
        >>> A_values = [[1.0, 1.0, 1.0, 1.0]] * 4
        >>> solver.setup(P_values, A_values)
        >>>
        >>> qs = [[2.0, 1.0]] * 4
        >>> bs = [[1.0, 0.7, 0.7]] * 4
        >>> solution = solver.solve(qs, bs)
        >>> print(solution.x.shape, solver.info.status[0])
    """

    def __init__(
        self,
        n: int,
        m: int,
        P_row_offsets: npt.ArrayLike,
        P_col_indices: npt.ArrayLike,
        A_row_offsets: npt.ArrayLike,
        A_col_indices: npt.ArrayLike,
        cones: Cones,
        settings: Optional[Settings] = None,
        b_sparsity_pattern: Optional[Sequence[bool]] = None,
    ) -> None:

        # Validate CSR structure first (before any other processing)
        _validate_csr_structure(
            n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones
        )
        _require_x_cones_compatible(cones, settings)

        # Validate P sparsity pattern is symmetric (full symmetric required)
        # This catches structural issues early, before setup() is called
        _validate_P_sparsity_pattern_symmetric(n, P_row_offsets, P_col_indices)

        # Validate smoothed diff is only used with LP cones (zero + nonneg)
        _validate_smoothed_diff_cones(settings, cones)

        # Check for old CVXPY with SOC cones
        _warn_cvxpy_soc_if_needed(cones)

        # Settings.batch_size is the contract here, so peek it before
        # _resolve_settings_and_device returns the (possibly copied) settings.
        peek_batch_size = settings.batch_size if settings is not None else Settings().batch_size
        settings, device = _resolve_settings_and_device(
            settings,
            n=n,
            m=m,
            cones=cones,
            nnz_P=len(P_col_indices),
            nnz_A=len(A_col_indices),
            batch_size=peek_batch_size,
        )

        self._device = device
        self._n = n
        self._m = m
        self._batch_size = settings.batch_size
        self._enable_grad = settings.enable_grad
        self._nnz_P = len(P_col_indices)
        self._nnz_A = len(A_col_indices)

        # Store init params
        self._P_row_offsets = np.asarray(P_row_offsets, dtype=np.int64)
        self._P_col_indices = np.asarray(P_col_indices, dtype=np.int64)
        self._A_row_offsets = np.asarray(A_row_offsets, dtype=np.int64)
        self._A_col_indices = np.asarray(A_col_indices, dtype=np.int64)
        self._cones = cones
        self._settings = settings

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

            ranked = _rank_method(device, n, m, self._nnz_A, self._batch_size)
            initial_method = ranked[0] if ranked else "auto"
            if ipm:
                ipm.direct_solve_method = initial_method

        self._b_sparsity_pattern = (
            list(b_sparsity_pattern) if b_sparsity_pattern is not None else None
        )

        construction_start = time.perf_counter()
        self._init_device(device, cones, settings, self._batch_size, settings.enable_grad)
        self._construction_time = time.perf_counter() - construction_start

        # Auto-tune state
        self._auto_tuned = False
        self._tune_result: Optional[TuneResult] = None

        # Last solve info (populated after solve())
        self._info: Optional[BatchedSolveInfo] = None

    def _init_device(self, device, cones, settings, batch_size, enable_grad):
        """Initialize a registered device backend."""
        # Active-set solver: use dedicated class directly, no device registry
        if settings is not None and settings.solver == SolverType.ACTIVE_SET:
            from moreau_cpu._cpu import ActiveSetSolver

            self._device_cones = cones
            self._device_settings = settings
            self._impl = ActiveSetSolver(
                self._n,
                self._m,
                self._P_row_offsets,
                self._P_col_indices,
                self._A_row_offsets,
                self._A_col_indices,
                cones,
                settings,
                batch_size=batch_size,
                enable_grad=enable_grad,
            )
            return

        DeviceSolver = _get_device_component(device, "solver")
        if DeviceSolver is None:
            raise RuntimeError(
                f"No solver registered for device '{device}'. "
                f"Available devices: {available_devices()}"
            )
        DeviceSettings = _get_device_component(device, "settings")
        cones_converter = _get_device_component(device, "cones_converter")
        settings_converter = _get_device_component(device, "settings_converter")

        # Convert Cones using registered converter
        if cones_converter is not None:
            device_cones = cones_converter(cones)
        else:
            device_cones = cones
        self._device_cones = device_cones

        # Convert Settings using registered converter
        if settings is None:
            device_settings = DeviceSettings() if DeviceSettings else None
            if device_settings is not None and hasattr(device_settings, "verbose"):
                device_settings.verbose = True
        elif settings_converter is not None:
            device_settings = settings_converter(settings)
        else:
            device_settings = settings
        self._device_settings = device_settings

        # Initialize unified solver
        self._impl = DeviceSolver(
            self._n,
            self._m,
            self._P_row_offsets,
            self._P_col_indices,
            self._A_row_offsets,
            self._A_col_indices,
            device_cones,
            device_settings,
            batch_size=batch_size,
            enable_grad=enable_grad,
            b_sparsity_pattern=self._b_sparsity_pattern,
        )

    def setup(self, P_values: npt.ArrayLike, A_values: npt.ArrayLike) -> None:
        """Set P and A matrix values for the batch.

        Args:
            P_values: P matrix values. Shape (batch, nnz_P) for per-problem values,
                      or (nnz_P,) to share across all problems in the batch.
            A_values: A matrix values. Shape (batch, nnz_A) for per-problem values,
                      or (nnz_A,) to share across all problems in the batch.
        """
        # Validate dimensions before converting to avoid confusing errors
        _validate_setup_values(self._batch_size, self._nnz_P, self._nnz_A, P_values, A_values)

        P_values = np.asarray(P_values, dtype=np.float64)
        A_values = np.asarray(A_values, dtype=np.float64)

        # Don't tile 1D to 2D here - let the backend handle shared P/A
        # efficiently via setup_shared() which equilibrates once

        # Validate P matrix values symmetry (check first problem only)
        if P_values.size > 0:
            P_vals_0 = P_values[0] if P_values.ndim > 1 else P_values
            _validate_P_values_symmetry(self._n, self._P_row_offsets, self._P_col_indices, P_vals_0)

        # Cache broadcast values for auto-tune on first solve() and reconstruction
        self._cached_P_values = P_values
        self._cached_A_values = A_values

        setup_start = time.perf_counter()
        self._impl.setup(P_values, A_values)
        self._setup_time = time.perf_counter() - setup_start

    def _needs_auto_tune(self) -> bool:
        """Check if auto-tune benchmarking should run on this solve."""
        if self._auto_tuned:
            return False
        if not self._settings.auto_tune:
            return False
        # Active-set solver has no method variants to benchmark
        if self._settings.solver == SolverType.ACTIVE_SET:
            return False
        # If user called set_default_device(), respect their choice for device
        from moreau._backend import _default_device_override

        if self._original_device == "auto" and _default_device_override is not None:
            return False
        # Auto-tune when device='auto' OR method='auto'
        if self._original_device == "auto":
            return True
        if self._original_method == "auto":
            return True
        return False

    def _auto_tune(self, qs, bs):
        """Benchmark solver configs and lock in the fastest.

        1. Solve with current (heuristic-best) config — keep the result
        2. Benchmark remaining combos with time_limit = best_time * margin
        3. If a challenger wins, reconfigure self and re-solve
        """
        warnings.warn(
            "First solve is benchmarking solver configurations "
            "(auto_tune=True). Subsequent solves will be faster. "
            "Set auto_tune=False (default) to use heuristic selection "
            "without benchmarking.",
            stacklevel=3,
        )
        if not hasattr(self, "_cached_P_values") or self._cached_P_values is None:
            raise RuntimeError(
                "setup() must be called before solve(). "
                "Matrix values are needed to benchmark solvers."
            )

        # Get all candidates to benchmark, heuristic-best first
        candidates = _auto_tune_candidates(
            self._original_device,
            self._original_method,
            n=self._n,
            m=self._m,
            nnz_P=self._nnz_P,
            nnz_A=self._nnz_A,
            batch_size=self._batch_size,
        )

        if not candidates:
            self._auto_tuned = True
            return

        # Benchmark all candidates via the shared benchmark loop.
        verbose = self._settings.verbose

        def _build_trial_settings(trial_device, trial_method, best_time):
            trial_settings = self._settings.model_copy(deep=True)
            trial_settings.device = trial_device
            trial_settings.verbose = False
            trial_settings.auto_tune = False  # Prevent recursive auto-tune
            if trial_settings.ipm_settings is None:
                trial_settings.ipm_settings = IPMSettings()
            trial_settings.ipm_settings.direct_solve_method = trial_method
            # Time-limit challengers once we have a baseline
            if best_time < float("inf"):
                trial_settings.time_limit = min(
                    best_time * _AUTO_TUNE_MARGIN,
                    trial_settings.time_limit,
                )
            return trial_settings

        def _build_trial_solver(trial_settings):
            trial_solver = CompiledSolver(
                n=self._n,
                m=self._m,
                P_row_offsets=self._P_row_offsets,
                P_col_indices=self._P_col_indices,
                A_row_offsets=self._A_row_offsets,
                A_col_indices=self._A_col_indices,
                cones=self._cones,
                settings=trial_settings,
            )
            trial_solver.setup(self._cached_P_values, self._cached_A_values)
            return trial_solver

        def _solve_trial(trial_solver):
            trial_solver.solve(qs, bs)
            return trial_solver.info

        results, best_time, best_key = _benchmark_candidates(
            candidates,
            _build_trial_settings,
            _build_trial_solver,
            _solve_trial,
            self._settings.time_limit,
            verbose=verbose,
        )

        # Fall back to fastest candidate if none solved successfully
        if best_key is None and results:
            best_key = min(results, key=lambda k: results[k]["solve_time"])
            best_time = results[best_key]["solve_time"]

        if best_key is None:
            # All candidates failed — keep current config unchanged
            self._auto_tuned = True
            return

        best_device, best_method = best_key.split(":", 1)
        tuned_limit = best_time * _AUTO_TUNE_MARGIN
        time_limit = max(tuned_limit, self._settings.time_limit)

        if verbose:
            print(
                f"[moreau] auto-tune: winner {best_device}:{best_method} "
                f"({best_time:.4f}s solve)",
                flush=True,
            )

        self._tune_result = TuneResult(
            device=best_device,
            method=best_method,
            time_limit=time_limit,
            results=results,
        )

        # Reconfigure self with winning settings.
        # Build the new impl fully before mutating self so that a failure
        # leaves the solver in a usable state with the old configuration.
        winning_settings = self._settings.model_copy(deep=True)
        winning_settings.device = best_device
        if winning_settings.ipm_settings is None:
            winning_settings.ipm_settings = IPMSettings()
        winning_settings.ipm_settings.direct_solve_method = best_method
        winning_settings.time_limit = time_limit

        DeviceSolver = _get_device_component(best_device, "solver")
        cones_converter = _get_device_component(best_device, "cones_converter")
        settings_converter = _get_device_component(best_device, "settings_converter")

        new_device_cones = cones_converter(self._cones) if cones_converter else self._cones
        if settings_converter is not None:
            new_device_settings = settings_converter(winning_settings)
        else:
            new_device_settings = winning_settings

        new_impl = DeviceSolver(
            self._n,
            self._m,
            self._P_row_offsets,
            self._P_col_indices,
            self._A_row_offsets,
            self._A_col_indices,
            new_device_cones,
            new_device_settings,
            batch_size=self._batch_size,
            enable_grad=self._enable_grad,
            b_sparsity_pattern=self._b_sparsity_pattern,
        )
        new_impl.setup(self._cached_P_values, self._cached_A_values)

        # All succeeded — swap atomically
        self._settings = winning_settings
        self._device = best_device
        self._device_cones = new_device_cones
        self._device_settings = new_device_settings
        self._impl = new_impl
        self._auto_tuned = True

    def solve(
        self,
        qs: npt.ArrayLike,
        bs: npt.ArrayLike,
        warm_start: Optional[Union[WarmStart, BatchedWarmStart]] = None,
    ) -> BatchedSolution:
        """Solve a batch of problems with optional warm starting.

        On the first call, if device or method is 'auto', automatically
        benchmarks all candidate configurations and locks in the fastest
        for subsequent solves.

        Args:
            qs: List of q vectors (linear cost), one per problem
            bs: List of b vectors (constraint RHS), one per problem
            warm_start: Optional WarmStart or BatchedWarmStart from a previous
                solve (e.g. ``solution.to_warm_start()``). The solver applies
                central-path smoothing internally. If the warm-started solve
                fails, it is automatically retried without warm start.
                Control which statuses trigger retry via
                ``IPMSettings.warm_start_no_retry``.

        Returns:
            BatchedSolution: primal/dual solution arrays (batch, n/m)

        Note:
            Solver metadata (status, timing, iterations) is available via solver.info
            after calling solve().
        """
        # Validate dimensions before converting to avoid confusing errors
        _validate_solve_inputs(self._batch_size, self._n, self._m, qs, bs)

        qs = np.asarray(qs, dtype=np.float64)
        bs = np.asarray(bs, dtype=np.float64)

        # Auto-tune on first solve when device or method is 'auto'
        if self._needs_auto_tune():
            self._auto_tune(qs, bs)

        kwargs = {}
        if warm_start is not None:
            if not isinstance(warm_start, (WarmStart, BatchedWarmStart)):
                raise TypeError(
                    f"warm_start must be a WarmStart or BatchedWarmStart, "
                    f"got {type(warm_start).__name__}"
                )

            warm_x = np.asarray(warm_start.x, dtype=np.float64)
            warm_z = np.asarray(warm_start.z, dtype=np.float64)
            warm_s = np.asarray(warm_start.s, dtype=np.float64)

            # Handle 1D inputs (WarmStart for single problem)
            if warm_x.ndim == 1:
                warm_x = warm_x.reshape(1, -1)
            if warm_z.ndim == 1:
                warm_z = warm_z.reshape(1, -1)
            if warm_s.ndim == 1:
                warm_s = warm_s.reshape(1, -1)

            if warm_x.shape != (self._batch_size, self._n):
                raise ValueError(
                    f"warm_start.x shape {warm_x.shape}, expected ({self._batch_size}, {self._n})"
                )
            if warm_z.shape != (self._batch_size, self._m):
                raise ValueError(
                    f"warm_start.z shape {warm_z.shape}, expected ({self._batch_size}, {self._m})"
                )
            if warm_s.shape != (self._batch_size, self._m):
                raise ValueError(
                    f"warm_start.s shape {warm_s.shape}, expected ({self._batch_size}, {self._m})"
                )

            kwargs["warm_x"] = warm_x
            kwargs["warm_z"] = warm_z
            kwargs["warm_s"] = warm_s

            # Direct-x dual: thread warm_z_x through when the WarmStart
            # carries it and the backend accepts the kwarg. Backends that
            # haven't been updated silently fall back to the default
            # init via the C++ warmStart fallback (see solver.cpp
            # warm_z_x == nullptr branch).
            zx = getattr(warm_start, "z_x", None)
            if zx is not None:
                zx_arr = np.asarray(zx, dtype=np.float64)
                if zx_arr.size > 0:
                    if zx_arr.ndim == 1:
                        zx_arr = zx_arr.reshape(1, -1)
                    total_x_dim = sum(
                        len(getattr(xc, "indices", []))
                        for xc in (getattr(self._cones, "x_cones", None) or [])
                    )
                    if zx_arr.shape != (self._batch_size, total_x_dim):
                        raise ValueError(
                            f"warm_start.z_x shape {zx_arr.shape}, expected "
                            f"({self._batch_size}, {total_x_dim}) "
                            f"(batch_size × sum of x_cones[*].indices length)"
                        )
                    kwargs["warm_z_x"] = zx_arr

        try:
            result = self._impl.solve(qs, bs, **kwargs)
        except TypeError as e:
            # Backend doesn't accept warm_z_x — retry without it. The C++
            # `warmStart` fallback path initializes z_x to the default
            # unit-init point so the warm-start still works for the
            # other variables.
            if "warm_z_x" in str(e) and "warm_z_x" in kwargs:
                del kwargs["warm_z_x"]
                result = self._impl.solve(qs, bs, **kwargs)
            else:
                raise
        solution, status = self._process_result(result)

        # Auto-retry cold if warm start may have caused the failure.
        # When some elements converged warm-started and others didn't, the
        # cold re-solve is run on the whole batch (since the underlying
        # solver has fixed batch_size), but only the failed indices' results
        # are patched in. Successful warm-started solutions are preserved so
        # downstream backward passes use the iterate the user expected
        # (mixing cold results into successful elements would corrupt the
        # gradient on every otherwise-good batch element). #178
        ipm = self._settings.ipm_settings
        failed_idx = _should_retry_cold(status, ipm)
        if warm_start is not None and failed_idx:
            _warn_warm_retry(failed_idx, status)
            warm_info = self._info
            cold_result = self._impl.solve(qs, bs)
            cold_solution, cold_status = self._process_result(cold_result)
            cold_info = self._info
            for i in failed_idx:
                solution.x[i] = cold_solution.x[i]
                solution.z[i] = cold_solution.z[i]
                solution.s[i] = cold_solution.s[i]
                status[i] = cold_status[i]
            # Merge per-element info: keep warm values for successful
            # elements, take cold values for failed ones. obj_val and
            # iterations are scalars when batch_size == 1, arrays otherwise.
            warm_info.obj_val = _merge_failed_batch_field(
                warm_info.obj_val, cold_info.obj_val, failed_idx
            )
            warm_info.iterations = _merge_failed_batch_field(
                warm_info.iterations, cold_info.iterations, failed_idx
            )
            warm_info.status = status
            self._info = warm_info

        return solution

    def _process_result(self, result):
        """Convert raw backend result into BatchedSolution and status list."""
        construction_time = result.get("construction_time", 0.0)
        if construction_time == 0.0:
            construction_time = self._construction_time
        setup_time = result.get("setup_time", 0.0)
        if setup_time == 0.0 and hasattr(self, "_setup_time"):
            setup_time = self._setup_time
        solve_time = result.get("solve_time", 0.0)

        raw_status = result["status"]
        if np.isscalar(raw_status) or (hasattr(raw_status, "ndim") and raw_status.ndim == 0):
            status = [_normalize_status(int(raw_status))]
        else:
            status = [_normalize_status(s) for s in raw_status]

        solution = BatchedSolution(
            x=result["x"],
            z=result["z"],
            s=result["s"],
            z_x=result.get("z_x"),
        )

        self._info = BatchedSolveInfo(
            status=status,
            obj_val=result["obj_val"],
            iterations=result["iterations"],
            solve_time=solve_time,
            setup_time=setup_time,
            construction_time=construction_time,
        )

        return solution, status

    def backward(
        self,
        dx: npt.ArrayLike,
        dz: Optional[npt.ArrayLike] = None,
        ds: Optional[npt.ArrayLike] = None,
        dz_x: Optional[npt.ArrayLike] = None,
    ) -> dict:
        """Compute gradients via implicit differentiation using cached state.

        Pass `dz_x` (shape matching `Solution.z_x`) to backprop through the
        direct-x cone duals. Defaults to None (no upstream on `z_x`).
        Both CPU and CUDA backends support dz_x.

        Returns:
            dict with keys ``dq``, ``db``, ``dP_values``, ``dA_values``,
            each mapped to a numpy array of the corresponding batched shape.
        """
        return self._impl.backward(dx, dz, ds, dz_x=dz_x)

    @property
    def device(self) -> str:
        return self._device

    @property
    def n(self) -> int:
        return self._n

    @property
    def m(self) -> int:
        return self._m

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def construction_time(self) -> float:
        """Time spent constructing solver structure (seconds)."""
        return self._construction_time

    @property
    def tune_result(self) -> Optional[TuneResult]:
        """Result from auto-tuning on the first solve() call.

        Returns None if auto-tune has not run yet (e.g. device and method
        were set explicitly, or solve() has not been called).
        """
        return self._tune_result

    @property
    def info(self) -> Optional[BatchedSolveInfo]:
        """Metadata from the last solve() call.

        Returns None if solve() has not been called yet.

        Contains:
            - status: List of SolverStatus enums (one per problem)
            - obj_val: Objective values array, shape (batch_size,)
            - iterations: Number of IPM iterations
            - solve_time: Time in IPM iterations (seconds)
            - setup_time: Time setting matrix values (seconds)
            - construction_time: Time constructing solver (seconds)
        """
        return self._info


__all__ = [
    # High-level API
    "Solver",
    "CompiledSolver",
    "Cones",
    "XConeSpec",
    "Settings",
    "IPMSettings",
    "ActiveSetSettings",
    "SolverType",
    "SolverStatus",
    "WarmStart",
    "BatchedWarmStart",
    "SolveInfo",
    "Solution",
    "BatchedSolveInfo",
    "BatchedSolution",
    "TuneResult",
    # Device API
    "available_devices",
    "device_available",
    "device_error",
    "default_device",
    "set_default_device",
    "get_default_device",
    "torch_available",
    "jax_available",
    "__version__",
]
