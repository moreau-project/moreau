"""Moreau CUDA: GPU-accelerated conic optimization solver."""

__version__ = "0.3.3"

import ctypes
import glob
import os
import re
import numpy as np
import warnings
from typing import Optional, Dict, Any, Union
from scipy import sparse

from moreau._types import Cones, Settings, SolverStatus

_SONAME_RE = re.compile(r"(lib[a-zA-Z0-9_]+\.so\.\d+)")


def _preload_nvidia_libs():
    """Preload NVIDIA shared libraries from nvidia-* PyPI packages when the
    system dynamic linker can't already resolve them.

    nvidia-* PyPI packages install .so files into site-packages/nvidia/*/lib/,
    which is not on LD_LIBRARY_PATH. We dlopen them with RTLD_GLOBAL so the
    C extension's linker can resolve its symbols.

    If the system already provides a soname (e.g., a CUDA container with
    /usr/local/cuda/lib64 on LD_LIBRARY_PATH), we skip the PyPI copy: loading
    two copies of one soname via RTLD_GLOBAL leaves two versions of libraries
    like cuBLAS and cuSOLVER mapped in the process, and their per-process
    internal state ends up split between the copies, producing silent
    numerical failures.
    """
    try:
        import nvidia
    except ImportError:
        return

    nvidia_base = os.path.dirname(nvidia.__path__[0])
    lib_dirs = glob.glob(os.path.join(nvidia_base, "nvidia", "*", "lib"))
    if not lib_dirs:
        return

    # Dependency order: cublas before cublasLt/cusolver, cusolver before cudss.
    lib_priority = [
        "libcudart.so",
        "libnvJitLink.so",
        "libcublas.so",
        "libcublasLt.so",
        "libcusparse.so",
        "libcusolver.so",
        "libcudss.so",
    ]

    for lib_name in lib_priority:
        # One candidate file per soname found in the PyPI lib dirs.
        by_soname: Dict[str, str] = {}
        for lib_dir in lib_dirs:
            for so_file in sorted(glob.glob(os.path.join(lib_dir, lib_name + "*"))):
                if not os.path.isfile(so_file):
                    continue
                m = _SONAME_RE.match(os.path.basename(so_file))
                if m:
                    by_soname.setdefault(m.group(1), so_file)

        for soname, so_file in by_soname.items():
            try:
                ctypes.CDLL(soname)
                continue
            except OSError:
                pass
            try:
                ctypes.CDLL(so_file, mode=os.RTLD_GLOBAL)
            except OSError:
                pass


_preload_nvidia_libs()

try:
    from ._moreau_cuda import Solver as _RawCudaSolver
    from ._moreau_cuda import Cones as _CudaCones
    from ._moreau_cuda import Settings as _CudaSettings
    from ._moreau_cuda import KKTSolverType as _KKTSolverType
    from ._moreau_cuda import DiffMethod as _DiffMethod
    from ._moreau_cuda import XConeKind as _CudaXConeKind
    from ._moreau_cuda import SupportedXConeT as _CudaSupportedXConeT
    from ._moreau_cuda import device_count, get_device, set_device, get_device_name
except ImportError as e:
    raise ImportError(
        "Failed to load moreau_cuda extension. Possible causes:\n"
        "  1. CUDA Toolkit 12.0+ not installed\n"
        "  2. nvidia-cusolver-cu12 (or cu13) not installed\n"
        "  3. Incompatible GPU architecture\n"
        f"Original error: {e}"
    ) from e


# Map CPU-style direct_solve_method strings to CUDA KKTSolverType
_KKT_SOLVER_TYPE_MAP = {
    "auto": _KKTSolverType.Auto,
    "cudss": _KKTSolverType.CuDSS,
}
# Add Riccati if the C++ backend supports it
if hasattr(_KKTSolverType, "Riccati"):
    _KKT_SOLVER_TYPE_MAP["riccati"] = _KKTSolverType.Riccati
# Add Woodbury if the C++ backend supports it
if hasattr(_KKTSolverType, "Woodbury"):
    _KKT_SOLVER_TYPE_MAP["woodbury"] = _KKTSolverType.Woodbury

_DIFF_METHOD_MAP = {
    "auto": _DiffMethod.Auto,
    "exact": _DiffMethod.Exact,
    "smoothed": _DiffMethod.Smoothed,
}


_XCONE_KIND_MAP = {
    "nonneg": _CudaXConeKind.Nonneg,
    "soc": _CudaXConeKind.SOC,
    "psd_triangle": _CudaXConeKind.PSD,
    "exp": _CudaXConeKind.Exp,
    "power": _CudaXConeKind.Power,
    "gen_power": _CudaXConeKind.GenPower,
}


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
    # GenPowerCone: flatten alphas, extract dim1s and dim2s
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

    # Direct-x cones: translate each XConeSpec to a C++ SupportedXConeT.
    x_cones = list(getattr(cones, "x_cones", []) or [])
    cuda_x_cones = []
    for spec in x_cones:
        kind = spec.kind
        if kind not in _XCONE_KIND_MAP:
            raise NotImplementedError(
                f"CUDA direct-x cones do not yet support kind={kind!r}. "
                f"Supported kinds on CUDA: {sorted(_XCONE_KIND_MAP)}."
            )
        indices = [int(i) for i in spec.indices]

        if kind == "nonneg" or kind == "soc":
            cuda_x_cones.append(_CudaSupportedXConeT(_XCONE_KIND_MAP[kind], indices))
        elif kind == "psd_triangle":
            psd_k = getattr(spec, "psd_k", None)
            if psd_k is None:
                raise ValueError("XConeSpec(kind='psd_triangle') requires psd_k")
            cuda_x_cones.append(
                _CudaSupportedXConeT(_XCONE_KIND_MAP[kind], indices, psd_k=int(psd_k))
            )
        elif kind == "exp":
            if len(indices) != 3:
                raise ValueError(
                    f"XConeSpec(kind='exp') requires exactly 3 indices, got {len(indices)}"
                )
            cuda_x_cones.append(_CudaSupportedXConeT(_XCONE_KIND_MAP[kind], indices))
        elif kind == "power":
            alpha = getattr(spec, "alpha", None)
            if alpha is None:
                raise ValueError("XConeSpec(kind='power') requires alpha")
            alpha = float(alpha)
            if not (0.0 < alpha < 1.0):
                raise ValueError(f"XConeSpec(kind='power') alpha must be in (0, 1), got {alpha}")
            if len(indices) != 3:
                raise ValueError(
                    f"XConeSpec(kind='power') requires exactly 3 indices, got {len(indices)}"
                )
            cuda_x_cones.append(
                _CudaSupportedXConeT(_XCONE_KIND_MAP[kind], indices, power_alpha=alpha)
            )
        elif kind == "gen_power":
            alphas = getattr(spec, "alphas", None)
            dim2 = getattr(spec, "dim2", None)
            if alphas is None or dim2 is None:
                raise ValueError("XConeSpec(kind='gen_power') requires alphas and dim2")
            alphas = [float(a) for a in alphas]
            dim2 = int(dim2)
            if dim2 < 1:
                raise ValueError(f"XConeSpec(kind='gen_power') dim2 must be >= 1, got {dim2}")
            expected_len = len(alphas) + dim2
            if len(indices) != expected_len:
                raise ValueError(
                    f"XConeSpec(kind='gen_power'): len(indices)={len(indices)} "
                    f"must equal len(alphas)+dim2={expected_len}"
                )
            cuda_x_cones.append(
                _CudaSupportedXConeT(
                    _XCONE_KIND_MAP[kind], indices, gen_power_alphas=alphas, gen_power_dim2=dim2
                )
            )
    cuda_cones.x_cones = cuda_x_cones
    return cuda_cones


def _check_p_symmetric(n: int, P_row_offsets: np.ndarray, P_col_indices: np.ndarray) -> None:
    """Check that P sparsity pattern is full symmetric, not just one triangle.

    CUDA backend requires P to be stored as full symmetric matrix for SpMV.
    This detects upper-only or lower-only patterns and raises a clear error.
    """
    if len(P_col_indices) == 0:
        return  # Empty P is fine

    P_ro = np.asarray(P_row_offsets)
    P_ci = np.asarray(P_col_indices)

    # Check each row for off-diagonal entries
    has_upper = False  # entry above diagonal (col > row)
    has_lower = False  # entry below diagonal (col < row)

    for row in range(n):
        start, end = P_ro[row], P_ro[row + 1]
        for idx in range(start, end):
            col = P_ci[idx]
            if col > row:
                has_upper = True
            elif col < row:
                has_lower = True

            if has_upper and has_lower:
                return  # Both triangles present, OK

    # Only one triangle present
    if has_upper and not has_lower:
        raise ValueError(
            "CUDA requires P to be stored as full symmetric matrix, not upper triangular. "
            "Use: P_full = P + P.T - sparse.diags(P.diagonal())"
        )
    if has_lower and not has_upper:
        raise ValueError(
            "CUDA requires P to be stored as full symmetric matrix, not lower triangular. "
            "Use: P_full = P + P.T - sparse.diags(P.diagonal())"
        )


def _settings_to_cuda(settings: Settings) -> _CudaSettings:
    """Convert unified Settings to CUDA settings format."""
    if settings.ipm_settings is None:
        from moreau._types import IPMSettings

        settings = settings.model_copy(deep=True)
        settings.ipm_settings = IPMSettings()

    cuda_settings = _CudaSettings()
    cuda_settings.device_id = settings.device_id
    cuda_settings.max_iter = settings.max_iter
    cuda_settings.time_limit = settings.time_limit
    cuda_settings.verbose = settings.verbose

    # Copy IPM settings to the nested ipm field
    # NOTE: We must modify cuda_settings.ipm directly, not via a local variable,
    # because nanobind struct access creates copies. Each assignment below
    # modifies the original ipm struct in cuda_settings.
    cuda_settings.ipm.tol_gap_abs = settings.ipm_settings.tol_gap_abs
    cuda_settings.ipm.tol_gap_rel = settings.ipm_settings.tol_gap_rel
    cuda_settings.ipm.tol_feas = settings.ipm_settings.tol_feas
    cuda_settings.ipm.tol_infeas_abs = settings.ipm_settings.tol_infeas_abs
    cuda_settings.ipm.tol_infeas_rel = settings.ipm_settings.tol_infeas_rel
    cuda_settings.ipm.tol_ktratio = settings.ipm_settings.tol_ktratio
    cuda_settings.ipm.max_step_fraction = settings.ipm_settings.max_step_fraction
    cuda_settings.ipm.equilibration_enable = settings.ipm_settings.equilibrate_enable

    # Copy reduced tolerances for AlmostSolved convergence
    cuda_settings.ipm.reduced_tol_gap_abs = settings.ipm_settings.reduced_tol_gap_abs
    cuda_settings.ipm.reduced_tol_gap_rel = settings.ipm_settings.reduced_tol_gap_rel
    cuda_settings.ipm.reduced_tol_feas = settings.ipm_settings.reduced_tol_feas
    cuda_settings.ipm.reduced_tol_infeas_abs = settings.ipm_settings.reduced_tol_infeas_abs
    cuda_settings.ipm.reduced_tol_infeas_rel = settings.ipm_settings.reduced_tol_infeas_rel
    cuda_settings.ipm.reduced_tol_ktratio = settings.ipm_settings.reduced_tol_ktratio

    # Map direct_solve_method to kkt_solver_type
    method = settings.ipm_settings.direct_solve_method.lower()
    if method in _KKT_SOLVER_TYPE_MAP:
        cuda_settings.ipm.kkt_solver_type = _KKT_SOLVER_TYPE_MAP[method]
    else:
        # Default to Auto for unknown methods
        cuda_settings.ipm.kkt_solver_type = _KKTSolverType.Auto

    # Map diff settings
    diff_method = settings.ipm_settings.diff_method.lower()
    if diff_method in _DIFF_METHOD_MAP:
        cuda_settings.ipm.diff_method = _DIFF_METHOD_MAP[diff_method]
    else:
        cuda_settings.ipm.diff_method = _DiffMethod.Auto
    cuda_settings.ipm.diff_smoothing_mu = settings.ipm_settings.diff_smoothing_mu
    cuda_settings.ipm.diff_smoothing_step_factor = settings.ipm_settings.diff_smoothing_step_factor

    # cuDSS max LU fill-in limit
    cuda_settings.ipm.max_lu_nnz = settings.ipm_settings.max_lu_nnz

    # cuDSS-specific settings
    cuda_settings.ipm.cudss_ir_steps = settings.ipm_settings.cudss_ir_steps
    cuda_settings.ipm.cudss_pivot_enable = settings.ipm_settings.cudss_pivot_enable

    # Chordal decomposition merge strategy (must match CPU to keep CPU/CUDA
    # on the same PSD reformulation). #176
    cuda_settings.ipm.chordal_decomposition_merge_method = (
        settings.ipm_settings.chordal_decomposition_merge_method
    )

    # YOLO mode settings
    cuda_settings.yolo = settings.yolo
    cuda_settings.yolo_num_iters = settings.yolo_num_iters

    return cuda_settings


class Solver:
    """Single-problem CUDA solver matching the CPU Solver API.

    Takes all problem data in the constructor, then call solve().

    Args:
        P: Quadratic cost matrix (scipy.sparse CSR, must be full symmetric)
        q: Linear cost vector (n,)
        A: Constraint matrix (scipy.sparse CSR)
        b: Constraint RHS (m,)
        cones: Cone specification
        settings: Optional solver settings
    """

    def __init__(
        self,
        P: sparse.spmatrix,
        q: np.ndarray,
        A: sparse.spmatrix,
        b: np.ndarray,
        cones: Cones,
        settings: Optional[Settings] = None,
    ):
        # Convert to CSR if needed
        P_csr = sparse.csr_array(P, dtype=np.float64)
        A_csr = sparse.csr_array(A, dtype=np.float64)

        # Validate P is full symmetric (not triangular)
        _check_p_symmetric(P_csr.shape[0], P_csr.indptr, P_csr.indices)

        # Store problem dimensions
        self._n = P_csr.shape[0]
        self._m = A_csr.shape[0]

        # Store problem data
        self._q = np.asarray(q, dtype=np.float64).flatten()
        self._b = np.asarray(b, dtype=np.float64).flatten()

        # Validate dimensions
        if len(self._q) != self._n:
            raise ValueError(f"q has length {len(self._q)}, expected {self._n}")
        if len(self._b) != self._m:
            raise ValueError(f"b has length {len(self._b)}, expected {self._m}")

        self._cones = cones

        # Store matrix data
        self._P_csr = P_csr
        self._A_csr = A_csr

        # Continuous optimization - use CompiledSolver
        self._compiled_solver = CompiledSolver(
            n=self._n,
            m=self._m,
            P_row_offsets=P_csr.indptr,
            P_col_indices=P_csr.indices,
            A_row_offsets=A_csr.indptr,
            A_col_indices=A_csr.indices,
            cones=cones,
            settings=settings,
            batch_size=1,
            enable_grad=False,
        )

        # Setup matrix values
        self._compiled_solver.setup(P_csr.data, A_csr.data)

        # Store solution after solve
        self._solution: Optional[Dict[str, Any]] = None

    def solve(self) -> Dict[str, Any]:
        """Solve the optimization problem.

        Returns:
            Dict with keys: x, z, s, status, obj_val,
                           iterations, construction_time, setup_time, solve_time
        """
        self._solution = self._compiled_solver.solve(self._q, self._b)
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


class CompiledSolver:
    """CUDA batch solver with three-step API pattern.

    Three-step API pattern:
    1. Construct solver with structure (dimensions, sparsity patterns, cones)
    2. setup(P_values, A_values) - Set matrix values
    3. solve(q, b) - Solve with linear cost and constraint RHS

    P MUST be provided as a full symmetric matrix (both upper and lower
    triangle entries). The solver stores P fully for SpMV operations and
    extracts only the upper triangle for the KKT system.

    Args:
        n: Number of primal variables
        m: Number of constraints
        P_row_offsets: CSR row pointers for P matrix (n+1 elements)
        P_col_indices: CSR column indices for P matrix (can be full or upper-triangular)
        A_row_offsets: CSR row pointers for A matrix (m+1 elements)
        A_col_indices: CSR column indices for A matrix
        cones: Cone specification (CUDA _CudaCones)
        settings: Optional solver settings (CUDA _CudaSettings)
        batch_size: Optional batch size hint for pre-allocation.
        enable_grad: If True, enable gradient computation.
    """

    _BATCH_CHANGE_WARN_THRESHOLD = 5

    def __init__(
        self,
        n: int,
        m: int,
        P_row_offsets: np.ndarray,
        P_col_indices: np.ndarray,
        A_row_offsets: np.ndarray,
        A_col_indices: np.ndarray,
        cones: Union[Cones, _CudaCones],
        settings: Optional[Union[Settings, _CudaSettings]] = None,
        batch_size: Optional[int] = None,
        enable_grad: bool = False,
        b_sparsity_pattern: Optional[list] = None,
    ):
        self._n = n
        self._m = m
        self._expected_batch_size = batch_size
        self._enable_grad = enable_grad
        self._b_sparsity_pattern = b_sparsity_pattern

        # Store CSR structure as numpy arrays (full P matrix)
        self._P_row_offsets = np.asarray(P_row_offsets, dtype=np.int64)
        self._P_col_indices = np.asarray(P_col_indices, dtype=np.int64)

        # Validate P is full symmetric (not triangular)
        _check_p_symmetric(n, self._P_row_offsets, self._P_col_indices)
        self._A_row_offsets = np.asarray(A_row_offsets, dtype=np.int64)
        self._A_col_indices = np.asarray(A_col_indices, dtype=np.int64)
        self._nnz_P = len(P_col_indices)
        self._nnz_A = len(A_col_indices)

        # Convert cones if needed (accept both unified and CUDA types)
        if isinstance(cones, _CudaCones):
            self._cones = cones
        else:
            self._cones = _cones_to_cuda(cones)

        # Convert settings if needed (accept both unified and CUDA types)
        if settings is None:
            self._settings = _CudaSettings()
        elif isinstance(settings, _CudaSettings):
            self._settings = settings
        else:
            self._settings = _settings_to_cuda(settings)

        # Stored matrix values (set via setup)
        self._P_values: Optional[np.ndarray] = None
        self._A_values: Optional[np.ndarray] = None

        # Solver instance (created on demand based on batch size)
        self._impl: Optional[_RawCudaSolver] = None
        self._current_batch_size: Optional[int] = None

        # Batch size change tracking for warnings
        self._batch_size_change_count = 0
        self._warned_about_batch_changes = False

        # Eager initialization if batch_size is provided
        if batch_size is not None:
            self._create_solver(batch_size)

    def _create_solver(self, batch_size: int) -> None:
        """Create or recreate the raw CUDA solver for given batch size."""
        # device_id is now in settings, passed through to C++
        self._impl = _RawCudaSolver(
            self._n,
            self._m,
            batch_size,
            self._P_row_offsets,
            self._P_col_indices,
            self._A_row_offsets,
            self._A_col_indices,
            self._cones,
            self._settings,
            self._enable_grad,
            b_sparsity_pattern=self._b_sparsity_pattern,
        )
        self._current_batch_size = batch_size

    def setup(self, P_values: np.ndarray, A_values: np.ndarray) -> None:
        """Set P and A matrix values.

        Must be called before solve(). Can be called multiple times to update
        values for repeated solves with the same structure.

        P_values must be for a full symmetric matrix (matching the sparsity
        pattern provided at construction). The values should satisfy P[i,j] = P[j,i].

        For batched problems, pass 2D arrays with shape (batch, nnz_P) and (batch, nnz_A).
        For single problems, pass 1D arrays with shape (nnz_P,) and (nnz_A,).

        Args:
            P_values: P matrix values. Shape (nnz_P,) for single, (batch, nnz_P) for batched
            A_values: A matrix values. Shape (nnz_A,) for single, (batch, nnz_A) for batched
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

    def solve(
        self,
        q: np.ndarray,
        b: np.ndarray,
        warm_x: Optional[np.ndarray] = None,
        warm_z: Optional[np.ndarray] = None,
        warm_s: Optional[np.ndarray] = None,
        warm_z_x: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Solve conic optimization problem(s).

        Requires setup() to be called first.

        Automatically detects single vs batched problems from input shape:
        - 1D inputs: Single problem, returns 1D results
        - 2D inputs: Batched problems, returns 2D results

        Args:
            q: Linear cost. Shape (n,) for single, (batch, n) for batched
            b: Constraint RHS. Shape (m,) for single, (batch, m) for batched
            warm_x: Optional warm start primal variables. Shape (n,) or (batch, n)
            warm_z: Optional warm start dual variables. Shape (m,) or (batch, m)
            warm_s: Optional warm start slack variables. Shape (m,) or (batch, m)

        Returns:
            Dict with keys: x, z, s, status, obj_val, iterations, solve_time
        """
        if self._P_values is None or self._A_values is None:
            raise RuntimeError("setup() must be called before solve()")

        q = np.asarray(q, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        # Detect single vs batched from input shape
        is_single = q.ndim == 1

        # Validate dimensions
        q_dim = q.shape[-1] if q.ndim > 0 else 0
        b_dim = b.shape[-1] if b.ndim > 0 else 0

        if q_dim != self._n:
            raise ValueError(f"q has dimension {q_dim}, expected {self._n}")
        if b_dim != self._m:
            raise ValueError(f"b has dimension {b_dim}, expected {self._m}")

        # Determine batch size
        if is_single:
            batch_size = 1
            # Expand to 2D for CUDA solver
            q_2d = q.reshape(1, -1)
            b_2d = b.reshape(1, -1)
            P_2d = self._P_values.reshape(1, -1) if self._P_values.ndim == 1 else self._P_values
            A_2d = self._A_values.reshape(1, -1) if self._A_values.ndim == 1 else self._A_values
        else:
            batch_size = q.shape[0]
            q_2d = np.ascontiguousarray(q)
            b_2d = np.ascontiguousarray(b)

            # Handle matrices: either shared (1D) or per-batch (2D)
            if self._P_values.ndim == 1:
                # Shared matrices - broadcast to batch
                P_2d = np.tile(self._P_values.reshape(1, -1), (batch_size, 1))
                A_2d = np.tile(self._A_values.reshape(1, -1), (batch_size, 1))
            else:
                P_2d = np.ascontiguousarray(self._P_values)
                A_2d = np.ascontiguousarray(self._A_values)

                # Validate batch size consistency
                if P_2d.shape[0] != batch_size:
                    raise ValueError(
                        f"Batch size mismatch: P/A have batch size {P_2d.shape[0]}, "
                        f"but q/b have batch size {batch_size}."
                    )

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

        # Validate warm start: all or none
        warm_x_2d = warm_z_2d = warm_s_2d = warm_z_x_2d = None
        has_warm = warm_x is not None or warm_z is not None or warm_s is not None
        if has_warm:
            if warm_x is None or warm_z is None or warm_s is None:
                raise ValueError("warm_x, warm_z, warm_s must all be provided or all omitted")
            warm_x = np.asarray(warm_x, dtype=np.float64)
            warm_z = np.asarray(warm_z, dtype=np.float64)
            warm_s = np.asarray(warm_s, dtype=np.float64)
            if is_single:
                warm_x_2d = warm_x.reshape(1, -1) if warm_x.ndim == 1 else warm_x
                warm_z_2d = warm_z.reshape(1, -1) if warm_z.ndim == 1 else warm_z
                warm_s_2d = warm_s.reshape(1, -1) if warm_s.ndim == 1 else warm_s
            else:
                warm_x_2d = np.ascontiguousarray(warm_x)
                warm_z_2d = np.ascontiguousarray(warm_z)
                warm_s_2d = np.ascontiguousarray(warm_s)

            # Optional direct-x dual: only forward if non-empty.
            if warm_z_x is not None:
                warm_z_x_arr = np.asarray(warm_z_x, dtype=np.float64)
                if warm_z_x_arr.size > 0:
                    if is_single and warm_z_x_arr.ndim == 1:
                        warm_z_x_2d = warm_z_x_arr.reshape(1, -1)
                    else:
                        warm_z_x_2d = np.ascontiguousarray(warm_z_x_arr)

        # Call the raw CUDA solver
        if has_warm:
            if warm_z_x_2d is not None:
                result = self._impl.solve_warm_start(
                    P_2d, A_2d, q_2d, b_2d, warm_x_2d, warm_z_2d, warm_s_2d, warm_z_x_2d
                )
            else:
                result = self._impl.solve_warm_start(
                    P_2d, A_2d, q_2d, b_2d, warm_x_2d, warm_z_2d, warm_s_2d
                )
        else:
            result = self._impl.solve(P_2d, A_2d, q_2d, b_2d)

        # Cache solution for backward if gradients enabled.
        # `cache_solution_for_backward` already ran inside `_impl.solve` at
        # convergence (solver.cpp end-of-solve hook). Calling it again after
        # `refine_smoothing_iterate` would re-read `solver.variables.z_x`,
        # which `refineSmoothingIterate` mutates: the centering KKT solve
        # writes step_lhs.{x, z, s} but never recomputes step_lhs.z_x, so
        # `variables.add_step` adds whatever stale step_lhs.z_x runIPMLoop
        # last produced — corrupting state.z_x with bogus deltas. Only
        # populate state.smoothing_{x,z,s} here; let the at-convergence
        # cache from `_impl.solve` stand for state.z_x.
        if self._enable_grad:
            self._impl.refine_smoothing_iterate()

        # Convert result to match unified API
        # Direct-x cone duals (z_x): shape (batch, total_xn) when present,
        # (batch, 0) otherwise. Pass through to the unified Solution.
        z_x_2d = result.get("z_x")

        if is_single:
            # Squeeze results back to 1D
            if z_x_2d is not None and z_x_2d.size:
                z_x_1d = z_x_2d.squeeze(0)
            else:
                z_x_1d = np.zeros(0, dtype=np.float64)
            return {
                "x": result["x"].squeeze(0),
                "z": result["z"].squeeze(0),
                "s": result["s"].squeeze(0),
                "z_x": z_x_1d,
                "status": (
                    int(result["status"][0])
                    if hasattr(result["status"], "__len__")
                    else int(result["status"])
                ),
                "obj_val": (
                    result["obj_val"] if np.isscalar(result["obj_val"]) else result["obj_val"][0]
                ),
                "iterations": result["iterations"],
                "construction_time": result["construction_time"],
                "setup_time": result["setup_time"],
                "solve_time": result["solve_time"],
            }
        else:
            # Keep batched format
            return {
                "x": result["x"],
                "z": result["z"],
                "s": result["s"],
                "z_x": (
                    z_x_2d if z_x_2d is not None else np.zeros((batch_size, 0), dtype=np.float64)
                ),
                "status": result["status"],
                "obj_val": result["obj_val"],
                "iterations": result["iterations"],
                "construction_time": result["construction_time"],
                "setup_time": result["setup_time"],
                "solve_time": result["solve_time"],
            }

    def backward(
        self,
        dx: np.ndarray,
        dz: Optional[np.ndarray] = None,
        ds: Optional[np.ndarray] = None,
        dz_x: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute gradients via implicit differentiation.

        Must be called after solve(). The solution is automatically cached
        when enable_grad=True.

        Args:
            dx: Upstream gradient w.r.t. x. Shape (n,) for single, (batch, n) for batched
            dz: Upstream gradient w.r.t. z. Shape (m,) for single, (batch, m) for batched
                If None, defaults to zeros.
            ds: Upstream gradient w.r.t. s. Shape (m,) for single, (batch, m) for batched
                If None, defaults to zeros.
            dz_x: Upstream gradient w.r.t. direct-x cone duals z_x. Same
                shape as the `z_x` returned by solve(); leave None to skip.

        Returns:
            Dict with keys: dP_values, dq, dA_values, db
        """
        if not self._enable_grad:
            raise RuntimeError("backward() requires enable_grad=True in settings")
        if self._impl is None:
            raise RuntimeError("solve() must be called before backward()")

        dx = np.asarray(dx, dtype=np.float64)
        is_single = dx.ndim == 1
        batch_size = self._current_batch_size

        # Ensure 2D inputs
        dx_2d = dx.reshape(1, -1) if is_single else dx
        if dz is None:
            dz_2d = np.zeros((batch_size, self._m), dtype=np.float64)
        else:
            dz = np.asarray(dz, dtype=np.float64)
            dz_2d = dz.reshape(1, -1) if is_single else dz
        if ds is None:
            ds_2d = np.zeros((batch_size, self._m), dtype=np.float64)
        else:
            ds = np.asarray(ds, dtype=np.float64)
            ds_2d = ds.reshape(1, -1) if is_single else ds

        # Ensure contiguous
        dx_2d = np.ascontiguousarray(dx_2d)
        dz_2d = np.ascontiguousarray(dz_2d)
        ds_2d = np.ascontiguousarray(ds_2d)

        # Optional direct-x dual gradient.
        dz_x_2d = None
        if dz_x is not None:
            dz_x_arr = np.asarray(dz_x, dtype=np.float64)
            if is_single and dz_x_arr.ndim == 1:
                dz_x_arr = dz_x_arr.reshape(1, -1)
            if dz_x_arr.size:
                dz_x_2d = np.ascontiguousarray(dz_x_arr)

        # Call C++ backward
        if dz_x_2d is not None:
            result = self._impl.backward(dx_2d, dz_2d, ds_2d, dz_x_2d)
        else:
            result = self._impl.backward(dx_2d, dz_2d, ds_2d)

        # Squeeze for single problem
        if is_single:
            return {
                "dP_values": result["dP_values"].squeeze(0),
                "dq": result["dq"].squeeze(0),
                "dA_values": result["dA_values"].squeeze(0),
                "db": result["db"].squeeze(0),
            }
        else:
            return result

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
    def device_id(self) -> int:
        """CUDA device ID (-1 = use current device)."""
        return self._settings.device_id

    @property
    def grad_initialized(self) -> bool:
        return False  # Not yet implemented

    def get_dimensions(self) -> Dict[str, int]:
        """Get problem dimensions."""
        return {
            "n": self._n,
            "m": self._m,
            "nnzP": self._nnz_P,
            "nnzA": self._nnz_A,
        }

    def memory_usage(self) -> int:
        """Get GPU memory usage in bytes."""
        if self._impl is not None:
            return self._impl.memory_usage()
        return 0


# Expose raw CUDA solver for advanced users
CudaSolver = _RawCudaSolver

_torch_import_error = None
_TorchCompiledSolverCuda = None
try:
    from .torch_wrapper import TorchCompiledSolver as _TorchCompiledSolverCuda

    _torch_available = True
except ImportError as e:
    _torch_available = False
    _torch_import_error = str(e)

_jax_import_error = None
_JaxSolverCuda = None
try:
    from .jax import JaxSolverCuda as _JaxSolverCuda

    _jax_available = True
except ImportError as e:
    _jax_available = False
    _jax_import_error = str(e)


def torch_available() -> bool:
    return _torch_available


def torch_import_error() -> str:
    return _torch_import_error


def jax_available() -> bool:
    return _jax_available


def jax_import_error() -> str:
    return _jax_import_error


__all__ = [
    # Single-problem solver
    "Solver",
    # Batch solver (three-step API)
    "CompiledSolver",
    # Raw CUDA solver for advanced use
    "CudaSolver",
    # Type exports
    "Cones",
    "Settings",
    "SolverStatus",
    "_CudaCones",
    "_CudaSettings",
    "_cones_to_cuda",
    "_settings_to_cuda",
    "torch_available",
    "torch_import_error",
    "jax_available",
    "jax_import_error",
    "__version__",
    # Device management
    "device_count",
    "get_device",
    "set_device",
    "get_device_name",
]


def _register():
    """Register CUDA backend with unified moreau interface.

    Note: The unified moreau API (moreau.Solver, moreau.CompiledSolver) uses
    the three-step pattern solver (CompiledSolver here) for both single and batch.
    The single-problem Solver class is a convenience for direct moreau-cuda use.
    """
    from moreau._backend import register_device

    register_device(
        "cuda",
        priority=100,
        # Use CompiledSolver (three-step pattern) for unified API
        solver_class=CompiledSolver,
        batch_solver_class=CompiledSolver,
        cones_class=_CudaCones,
        settings_class=_CudaSettings,
        # TorchCompiledSolver for unified torch API
        torch_solver_class=_TorchCompiledSolverCuda,
        torch_batch_solver_class=_TorchCompiledSolverCuda,
        # JaxSolverCuda for unified JAX API (uses CUDA solver with pure_callback)
        jax_solver_class=_JaxSolverCuda,
        cones_converter=_cones_to_cuda,
        settings_converter=_settings_to_cuda,
    )
