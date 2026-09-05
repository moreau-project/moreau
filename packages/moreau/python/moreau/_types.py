"""Shared type definitions for moreau package.

This module defines the core data types used across CPU and GPU backends:
- SolverStatus: Unified status enum
- Cones: Cone specification
- Settings: Solver settings with sub-settings for different solver types

These are pure Python types that each backend converts internally.
"""

from dataclasses import dataclass, field
from enum import IntEnum, Enum
from typing import Dict, FrozenSet, List, Literal, Optional, Tuple, Union, Annotated

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class SolverType(str, Enum):
    """Solver algorithm type.

    - AUTO: Automatically select solver based on problem structure (default).
            Uses active-set for small QPs (n ≤ 500, m ≤ max(500, 2n), zero+nonneg only),
            IPM otherwise.
    - IPM: Interior-point method. High accuracy, moderate speed.
           Supports automatic differentiation. All cone types.
    - ACTIVE_SET: Dual active-set method. Very fast for small QPs.
                  CPU-only. Zero + nonneg cones only. Supports differentiation.
    """

    AUTO = "auto"
    IPM = "ipm"
    ACTIVE_SET = "active_set"


class SolverStatus(IntEnum):
    """Unified solver status enum supporting both CPU and GPU backends.

    This IntEnum provides a consistent API regardless of backend:
    - Supports direct comparison: status == SolverStatus.Solved
    - Supports int comparison: status == 1
    - Has .value and .name properties
    - Works with int(): int(status) == 1
    - str(status) returns the name (e.g. "Solved"), not the integer value
    """

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

    def __str__(self) -> str:
        return self.name


class DirectConeSpec(BaseModel):
    """Direct cone specification: constrain x[indices] ∈ K_{kind}.

    Unlike slack cones (constraints ``Ax + s = b`` with ``s ∈ K``), a direct
    cone constrains a subvector of the primal variable ``x`` directly. This
    avoids introducing a slack variable and can be significantly faster.

    Attributes:
        kind: Cone type. One of 'nonneg', 'soc', 'psd_triangle', 'exp', 'power'.
        indices: Distinct indices into ``x``, all in ``[0, n)``. Order matters
            for SOC (first entry is the scalar ``t``, remaining entries form the
            vector ``v``), for PSD (column-major ``svec`` ordering), for exp
            cone (3 indices in ``(u, v, w)`` order with the cone constraint
            ``v > 0`` and ``w >= v · exp(u/v)``), and for power cone (3 indices
            in ``(s1, s2, s3)`` order with ``s1^α · s2^(1-α) >= |s3|``,
            ``s1, s2 >= 0``). For nonneg, order is immaterial.
        psd_k: Side length of the PSD matrix (only for ``kind='psd_triangle'``).
            Requires ``len(indices) == psd_k * (psd_k + 1) // 2``.
        alpha: Power cone parameter (only for ``kind='power'``); must be in
            ``(0, 1)``.

    Example:
        >>> spec_soc = DirectConeSpec(kind='soc', indices=[3, 4, 5, 6])
        >>> spec_nn = DirectConeSpec(kind='nonneg', indices=[7])
        >>> spec_psd = DirectConeSpec(kind='psd_triangle', indices=list(range(6)), psd_k=3)
        >>> spec_exp = DirectConeSpec(kind='exp', indices=[0, 1, 2])
        >>> spec_pow = DirectConeSpec(kind='power', indices=[0, 1, 2], alpha=0.5)
    """

    model_config = ConfigDict(validate_assignment=True)

    kind: Literal["nonneg", "soc", "psd_triangle", "exp", "power", "gen_power"]
    indices: List[int]
    psd_k: Optional[int] = None
    alpha: Optional[float] = None
    alphas: Optional[List[float]] = None
    dim2: Optional[int] = None

    @field_validator("indices")
    @classmethod
    def validate_indices(cls, v: List[int]) -> List[int]:
        if len(v) == 0:
            raise ValueError("DirectConeSpec.indices must be non-empty")
        if len(set(v)) != len(v):
            raise ValueError(f"DirectConeSpec.indices contains duplicates: {v}")
        for idx in v:
            if idx < 0:
                raise ValueError(f"DirectConeSpec.indices must be non-negative, got {idx}")
        return v

    @model_validator(mode="after")
    def validate_kind_size(self):
        n = len(self.indices)
        if self.kind == "soc":
            if n < 2:
                raise ValueError(f"SOC x-cone requires >= 2 indices, got {n}")
            if self.psd_k is not None:
                raise ValueError("psd_k must be None for kind='soc'")
            if self.alpha is not None:
                raise ValueError("alpha must be None for kind='soc'")
            if self.alphas is not None or self.dim2 is not None:
                raise ValueError("alphas/dim2 must be None for kind='soc'")
        elif self.kind == "nonneg":
            if self.psd_k is not None:
                raise ValueError("psd_k must be None for kind='nonneg'")
            if self.alpha is not None:
                raise ValueError("alpha must be None for kind='nonneg'")
            if self.alphas is not None or self.dim2 is not None:
                raise ValueError("alphas/dim2 must be None for kind='nonneg'")
        elif self.kind == "psd_triangle":
            if self.psd_k is None or self.psd_k < 1:
                raise ValueError(f"psd_k must be >= 1 for kind='psd_triangle', got {self.psd_k}")
            expected = self.psd_k * (self.psd_k + 1) // 2
            if n != expected:
                raise ValueError(
                    f"PSD x-cone with psd_k={self.psd_k} requires " f"{expected} indices, got {n}"
                )
            if self.alpha is not None:
                raise ValueError("alpha must be None for kind='psd_triangle'")
            if self.alphas is not None or self.dim2 is not None:
                raise ValueError("alphas/dim2 must be None for kind='psd_triangle'")
        elif self.kind == "exp":
            if n != 3:
                raise ValueError(f"Exp x-cone requires exactly 3 indices, got {n}")
            if self.psd_k is not None:
                raise ValueError("psd_k must be None for kind='exp'")
            if self.alpha is not None:
                raise ValueError("alpha must be None for kind='exp'")
            if self.alphas is not None or self.dim2 is not None:
                raise ValueError("alphas/dim2 must be None for kind='exp'")
        elif self.kind == "power":
            if n != 3:
                raise ValueError(f"Power x-cone requires exactly 3 indices, got {n}")
            if self.psd_k is not None:
                raise ValueError("psd_k must be None for kind='power'")
            if self.alpha is None:
                raise ValueError("alpha must be set for kind='power'")
            if not (0.0 < self.alpha < 1.0):
                raise ValueError(f"alpha must be in (0, 1) for kind='power', got {self.alpha}")
        elif self.kind == "gen_power":
            if self.psd_k is not None:
                raise ValueError("psd_k must be None for kind='gen_power'")
            if self.alpha is not None:
                raise ValueError("alpha must be None for kind='gen_power' (use alphas)")
            if self.alphas is None or len(self.alphas) < 1:
                raise ValueError("alphas must be a non-empty list for kind='gen_power'")
            if self.dim2 is None or self.dim2 < 1:
                raise ValueError(f"dim2 must be >= 1 for kind='gen_power', got {self.dim2}")
            for i, a in enumerate(self.alphas):
                if a <= 0.0:
                    raise ValueError(f"alphas[{i}] must be > 0 for kind='gen_power', got {a}")
            asum = sum(self.alphas)
            if abs(asum - 1.0) > 1e-8 * len(self.alphas):
                raise ValueError(f"alphas must sum to 1 for kind='gen_power', got sum {asum}")
            expected = len(self.alphas) + self.dim2
            if n != expected:
                raise ValueError(
                    f"GenPower x-cone requires len(indices) == len(alphas) + dim2 "
                    f"= {len(self.alphas)} + {self.dim2} = {expected}, got {n}"
                )
            # Indices are cone-internal: positions [0, dim1) are the
            # p-part (paired with `alphas[i]`), positions [dim1, dim)
            # are the w-part. Both CPU `kkt_assembly.rs` and the CUDA
            # KKT layout (kkt.hpp) handle arbitrary user orderings
            # (interleaved p/w indices are allowed) via per-block
            # permutations.
        return self

    def numel(self) -> int:
        """Number of primal indices covered by this cone."""
        return len(self.indices)


class Cones(BaseModel):
    """Cone specification for conic optimization problems.

    Attributes:
        num_zero_cones: Number of equality constraint dimensions (zero cone)
        num_nonneg_cones: Number of inequality constraint dimensions (nonnegative cone)
        so_cone_dims: List of second-order cone dimensions (each >= 2)
        num_exp_cones: Number of exponential cones (each is 3D)
        power_alphas: List of power cone alpha parameters (each cone is 3D)
        gen_power_cone_params: List of generalized power cone parameters.
            Each element is (alphas, dim2) where alphas is a list of positive
            floats summing to 1 (length = dim1) and dim2 >= 1. Total cone
            dimension is dim1 + dim2.
        dir_cones: List of direct cone specifications constraining subvectors
            of ``x`` directly (see :class:`DirectConeSpec`). Indices across all
            entries must be pairwise disjoint. Direct cones are additive
            to the slack cones above; they do not consume rows of ``A`` or ``b``.

    Example:
        >>> cones = Cones(num_zero_cones=1, num_nonneg_cones=2, so_cone_dims=[3, 5])
        >>> # Backward compat: num_so_cones=2 creates [3, 3]
        >>> cones = Cones(num_zero_cones=1, num_nonneg_cones=2, num_so_cones=2)
        >>> # GenPowerCone: alphas=[0.3, 0.7], dim2=2 => total dim = 4
        >>> cones = Cones(gen_power_cone_params=[([0.3, 0.7], 2)])
        >>> # Direct nonneg: x[3] >= 0 without an extra slack row
        >>> cones = Cones(dir_cones=[DirectConeSpec(kind='nonneg', indices=[3])])
    """

    model_config = ConfigDict(validate_assignment=True)

    num_zero_cones: Annotated[int, Field(ge=0)] = 0
    num_nonneg_cones: Annotated[int, Field(ge=0)] = 0
    so_cone_dims: List[int] = Field(default_factory=list)
    num_exp_cones: Annotated[int, Field(ge=0)] = 0
    power_alphas: List[float] = Field(default_factory=list)
    gen_power_cone_params: List[Tuple[List[float], int]] = Field(default_factory=list)
    psd_dims: List[int] = Field(default_factory=list)
    dir_cones: List[DirectConeSpec] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def compat_num_so_cones(cls, data):
        """Backward compat: num_so_cones=N -> so_cone_dims=[3]*N."""
        if isinstance(data, dict) and "num_so_cones" in data and "so_cone_dims" not in data:
            n = data.pop("num_so_cones")
            if n < 0:
                raise ValueError(f"num_so_cones must be >= 0, got {n}")
            if n > 0:
                data["so_cone_dims"] = [3] * n
        return data

    @field_validator("so_cone_dims")
    @classmethod
    def validate_so_cone_dims(cls, v: List[int]) -> List[int]:
        """Validate SOC dimensions are >= 2."""
        for i, d in enumerate(v):
            if d < 2:
                raise ValueError(f"so_cone_dims[{i}] = {d}, must be >= 2")
        return v

    @field_validator("power_alphas")
    @classmethod
    def validate_power_alphas(cls, v: List[float]) -> List[float]:
        """Validate power cone alphas are in (0, 1)."""
        for i, alpha in enumerate(v):
            if not (0 < alpha < 1):
                raise ValueError(f"power_alphas[{i}] = {alpha} must be in (0, 1)")
        return v

    @field_validator("dir_cones")
    @classmethod
    def validate_dir_cones_disjoint(cls, v: List[DirectConeSpec]) -> List[DirectConeSpec]:
        """Ensure indices are pairwise disjoint across all direct cones."""
        seen: Dict[int, int] = {}
        for i, spec in enumerate(v):
            for idx in spec.indices:
                if idx in seen:
                    raise ValueError(
                        f"dir_cones index {idx} appears in both dir_cones[{seen[idx]}] "
                        f"and dir_cones[{i}]"
                    )
                seen[idx] = i
        return v

    @field_validator("gen_power_cone_params")
    @classmethod
    def validate_gen_power_cone_params(
        cls, v: List[Tuple[List[float], int]]
    ) -> List[Tuple[List[float], int]]:
        """Validate generalized power cone parameters."""
        for i, param in enumerate(v):
            if not isinstance(param, (list, tuple)) or len(param) != 2:
                raise ValueError(f"gen_power_cone_params[{i}] must be (alphas, dim2)")
            alphas, dim2 = param
            alphas = list(alphas)  # accept ndarray, tuple, etc.
            try:
                dim2 = int(dim2)
            except (TypeError, ValueError):
                raise ValueError(f"gen_power_cone_params[{i}]: dim2 must be convertible to int")
            if len(alphas) == 0:
                raise ValueError(f"gen_power_cone_params[{i}]: alphas must be a non-empty list")
            if dim2 < 1:
                raise ValueError(f"gen_power_cone_params[{i}]: dim2 must be an integer >= 1")
            for j, a in enumerate(alphas):
                if not (a > 0):
                    raise ValueError(f"gen_power_cone_params[{i}].alphas[{j}] = {a} must be > 0")
            alpha_sum = sum(alphas)
            if abs(alpha_sum - 1.0) > 1e-8 * len(alphas):
                raise ValueError(
                    f"gen_power_cone_params[{i}]: alphas must sum to 1, got {alpha_sum}"
                )
            v[i] = (alphas, dim2)
        return v

    @field_validator("psd_dims")
    @classmethod
    def validate_psd_dims(cls, v: List[int]) -> List[int]:
        """Validate PSD cone dimensions are >= 1."""
        for i, dim in enumerate(v):
            if dim < 1:
                raise ValueError(f"psd_dims[{i}] = {dim} must be >= 1")
        return v

    @property
    def num_so_cones(self) -> int:
        """Number of second-order cones."""
        return len(self.so_cone_dims)

    @property
    def num_power_cones(self) -> int:
        """Number of power cones."""
        return len(self.power_alphas)

    @property
    def num_gen_power_cones(self) -> int:
        """Number of generalized power cones."""
        return len(self.gen_power_cone_params)

    @property
    def num_psd_cones(self) -> int:
        """Number of PSD (SDP) cones."""
        return len(self.psd_dims)

    def total_constraints(self) -> int:
        """Total number of constraint rows across all cones."""
        total = self.num_zero_cones + self.num_nonneg_cones
        total += sum(self.so_cone_dims)
        total += 3 * self.num_exp_cones
        total += 3 * len(self.power_alphas)
        for alphas, dim2 in self.gen_power_cone_params:
            total += len(alphas) + dim2
        total += sum(d * (d + 1) // 2 for d in self.psd_dims)
        return total

    def degree(self) -> int:
        """Degree of the cone (barrier function degree)."""
        deg = self.num_nonneg_cones
        deg += len(self.so_cone_dims)
        deg += 3 * self.num_exp_cones
        deg += 3 * len(self.power_alphas)
        for alphas, _dim2 in self.gen_power_cone_params:
            # GenPowerCone degree = dim1 + 1
            deg += len(alphas) + 1
        deg += sum(self.psd_dims)
        for spec in self.dir_cones:
            if spec.kind == "nonneg":
                deg += len(spec.indices)
            elif spec.kind == "soc":
                deg += 1
            elif spec.kind == "psd_triangle":
                assert spec.psd_k is not None  # enforced by DirectConeSpec.validate_kind_size
                deg += spec.psd_k
            elif spec.kind == "exp":
                deg += 3
            elif spec.kind == "power":
                deg += 3
            elif spec.kind == "gen_power":
                assert spec.alphas is not None  # enforced by validate_kind_size
                deg += len(spec.alphas) + 1
        return deg

    def validate_dir_cone_indices(self, n: int) -> None:
        """Validate that all dir_cones indices lie in ``[0, n)``.

        Called by the Solver once ``n`` is known (Cones alone cannot check).
        """
        for i, spec in enumerate(self.dir_cones):
            for idx in spec.indices:
                if idx >= n:
                    raise ValueError(f"dir_cones[{i}].indices contains {idx} which is >= n={n}")


class IPMSettings(BaseModel):
    """Interior-point method specific settings.

    These settings control the behavior of the IPM solver algorithm.
    They are stored in Settings.ipm_settings when solver='ipm'.

    Attributes:
        Convergence tolerances:
        tol_gap_abs: Absolute duality gap tolerance (default: 1e-8)
        tol_gap_rel: Relative duality gap tolerance (default: 1e-8)
        tol_feas: Feasibility tolerance (default: 1e-8)
        tol_infeas_abs: Absolute infeasibility tolerance (default: 1e-8)
        tol_infeas_rel: Relative infeasibility tolerance (default: 1e-8)
        tol_ktratio: KT ratio tolerance for homogeneous self-dual (default: 1e-6)

        Algorithm control:
        max_step_fraction: Maximum step size fraction (default: 0.99)
        equilibrate_enable: Enable matrix equilibration (default: True)
        direct_solve_method: KKT solver method (default: 'auto')
            Options: 'auto', 'qdldl', 'faer-1t', 'faer-nt', 'faer', 'cudss', 'riccati', 'woodbury'
            Note: 'qdldl' is CPU-only; 'cudss'/'woodbury' are CUDA-only; 'riccati' requires
            block-tridiagonal structure (MPC/LQR problems); 'woodbury' requires diagonal P +
            low-rank A (portfolio-type problems).

        Reduced tolerances (for relaxed convergence):
        reduced_tol_gap_abs: Reduced absolute duality gap tolerance (default: 5e-5)
        reduced_tol_gap_rel: Reduced relative duality gap tolerance (default: 5e-5)
        reduced_tol_feas: Reduced feasibility tolerance (default: 1e-4)
        reduced_tol_infeas_abs: Reduced absolute infeasibility tolerance (default: 5e-12)
        reduced_tol_infeas_rel: Reduced relative infeasibility tolerance (default: 5e-5)
        reduced_tol_ktratio: Reduced KT ratio tolerance (default: 1e-4)

        Warm start retry control:
        warm_start_no_retry: Set of SolverStatus values that do NOT trigger a cold
            retry when warm starting. Default (None) uses {Solved, AlmostSolved,
            MaxIterations, CallbackTerminated}. All other statuses trigger an
            automatic cold retry with a warning.

    Example:
        >>> ipm = IPMSettings(tol_gap_abs=1e-6, tol_feas=1e-6)
        >>> settings = Settings(ipm_settings=ipm, device='cuda')
    """

    model_config = ConfigDict(validate_assignment=True)

    # Convergence tolerances
    tol_gap_abs: Annotated[float, Field(gt=0)] = 1e-8
    tol_gap_rel: Annotated[float, Field(gt=0)] = 1e-8
    tol_feas: Annotated[float, Field(gt=0)] = 1e-8
    tol_infeas_abs: Annotated[float, Field(gt=0)] = 1e-8
    tol_infeas_rel: Annotated[float, Field(gt=0)] = 1e-8
    tol_ktratio: Annotated[float, Field(gt=0)] = 1e-6

    # Algorithm control
    max_step_fraction: Annotated[float, Field(gt=0, le=1)] = 0.99
    equilibrate_enable: bool = True
    direct_solve_method: str = "auto"

    # cuDSS-specific settings (only apply when direct_solve_method='cudss')
    cudss_ir_steps: Annotated[int, Field(ge=0)] = 2
    cudss_pivot_enable: bool = False
    # Upper bound on LU fill-in NNZ for cuDSS factorization. Default -1
    # delegates to cuDSS's heuristic (100×nnz of KKT). Raise this for
    # dense PSD problems where the heuristic OOMs; set to a specific
    # number to cap memory at predictable cost.
    max_lu_nnz: int = -1

    # Reduced tolerances (for relaxed convergence)
    reduced_tol_gap_abs: Annotated[float, Field(gt=0)] = 5e-5
    reduced_tol_gap_rel: Annotated[float, Field(gt=0)] = 5e-5
    reduced_tol_feas: Annotated[float, Field(gt=0)] = 1e-4
    reduced_tol_infeas_abs: Annotated[float, Field(gt=0)] = 5e-12
    reduced_tol_infeas_rel: Annotated[float, Field(gt=0)] = 5e-5
    reduced_tol_ktratio: Annotated[float, Field(gt=0)] = 1e-4

    # Differentiation method
    diff_method: str = "auto"
    diff_smoothing_mu: Annotated[float, Field(gt=0)] = 1e-4
    diff_smoothing_step_factor: Annotated[float, Field(gt=1)] = 30.0

    # Warm start retry control: statuses that do NOT trigger a cold retry.
    # Default: {Solved, AlmostSolved, MaxIterations, CallbackTerminated}.
    # Set to None to use the default, or provide a custom frozenset of SolverStatus.
    warm_start_no_retry: Optional[FrozenSet[SolverStatus]] = None

    # Chordal decomposition for sparse PSD slack cones. Default True.
    # Disable when mixing slack PSD with direct cones (chordal augmentation
    # adds primal slack vars; direct indices are anchored at the original
    # x[J] partitions and the augmentation map doesn't yet thread them).
    chordal_decomposition_enable: bool = True

    # Chordal decomposition merge strategy for sparse PSD cones. Must match
    # across CPU and CUDA so the same problem produces the same decomposition
    # (and the same Solved iterate of the same reformulation) regardless of
    # backend. Accepted values: 'clique_graph' (default; Garstka et al. 2019),
    # 'parent_child', or 'none'. (#176)
    chordal_decomposition_merge_method: str = "clique_graph"

    @field_validator("direct_solve_method")
    @classmethod
    def validate_direct_solve_method(cls, v: str) -> str:
        """Validate KKT solver method."""
        valid = {"auto", "qdldl", "faer-1t", "faer-nt", "faer", "cudss", "riccati", "woodbury"}
        if v not in valid:
            raise ValueError(f"direct_solve_method must be one of {valid}, got '{v}'")
        return v

    @field_validator("diff_method")
    @classmethod
    def validate_diff_method(cls, v: str) -> str:
        """Validate differentiation method."""
        valid = {"auto", "smoothed", "exact"}
        if v not in valid:
            raise ValueError(f"diff_method must be one of {valid}, got '{v}'")
        return v

    @field_validator("chordal_decomposition_merge_method")
    @classmethod
    def validate_chordal_decomposition_merge_method(cls, v: str) -> str:
        """Validate chordal merge strategy. The same value must be honored
        by both CPU and CUDA backends to keep their decompositions identical."""
        valid = {"clique_graph", "parent_child", "none"}
        if v not in valid:
            raise ValueError(
                f"chordal_decomposition_merge_method must be one of {valid}, " f"got '{v}'"
            )
        return v


class ActiveSetSettings(BaseModel):
    """Active-set solver specific settings.

    These settings control the dual active-set QP solver.
    Only used when solver='active_set'. Supports differentiation
    (enable_grad=True) and warm starting. CPU-only. Zero + nonneg cones only.

    Attributes:
        primal_tol: Primal feasibility tolerance (default: 1e-6)
        dual_tol: Dual feasibility tolerance (default: 1e-12)
        zero_tol: Zero tolerance for numerical checks (default: 1e-11)
        pivot_tol: Pivot tolerance for LDL factorization (default: 1e-6)
        progress_tol: Progress tolerance for cycle detection (default: 1e-14)
        fval_bound: Objective bound for infeasibility detection (default: 1e30)
        iter_limit: Maximum iterations (default: 10000)
        cycle_tol: Cycle detection tolerance (default: 10)
        diff_method: Differentiation method - 'exact' (default) or 'smoothed'.
            'smoothed' uses barrier-based smoothing (h_i = z_i²/(z_i² + μ)) to
            produce C^∞ gradients through active-set transitions.
        diff_smoothing_mu: Smoothing parameter μ for 'smoothed' mode (default: 1e-4).
            Larger values produce smoother gradients at the cost of accuracy.
    """

    model_config = ConfigDict(validate_assignment=True)

    primal_tol: Annotated[float, Field(gt=0)] = 1e-6
    dual_tol: Annotated[float, Field(gt=0)] = 1e-12
    zero_tol: Annotated[float, Field(gt=0)] = 1e-11
    pivot_tol: Annotated[float, Field(gt=0)] = 1e-6
    progress_tol: Annotated[float, Field(gt=0)] = 1e-14
    fval_bound: float = 1e30
    iter_limit: Annotated[int, Field(ge=1)] = 10000
    cycle_tol: Annotated[int, Field(ge=1)] = 10
    diff_method: Literal["exact", "smoothed"] = "exact"
    diff_smoothing_mu: Annotated[float, Field(gt=0)] = 1e-4


class Settings(BaseModel):
    """Solver settings for CPU and CUDA backends.

    The Settings class provides a unified interface for configuring solver behavior.
    It uses a two-level structure:
    - Top-level settings control device, batching, and algorithm-agnostic options
    - Solver-specific settings (e.g., IPMSettings) control tolerances and algorithm details

    Attributes:
        solver: Solver algorithm - 'auto' (default), 'ipm', or 'active_set'.
                'auto' uses active-set for small QPs (n <= 500, zero+nonneg only),
                IPM otherwise. 'active_set' is CPU-only, zero+nonneg cones only.
        device: Device selection - 'auto' (default), 'cuda', or 'cpu'
        device_id: CUDA device ID to use (-1 = use current device, default: -1)
        batch_size: Number of problems in batch for CompiledSolver (default: 1)
        enable_grad: If True, pre-compute gradient structures for backward pass.
                     Required for backward() to work.
        auto_tune: If True, benchmark solver configurations on first solve() when
                   device='auto' or direct_solve_method='auto'. If False (default),
                   use heuristic selection without benchmarking.
        yolo: If True, run in YOLO mode: fixed iterations, no convergence check,
              no GPU-host sync. Incompatible with enable_grad=True (default: False).
        yolo_num_iters: Number of iterations to run in YOLO mode (default: 15)
        max_iter: Maximum iterations (default: 200)
        time_limit: Time limit in seconds (default: infinity)
        verbose: Enable verbose output (default: False)
        ipm_settings: IPM-specific settings including tolerances (auto-created if None)

    Example:
        >>> # Simple usage with defaults
        >>> settings = Settings(device='cuda', verbose=True)
        >>>
        >>> # With custom tolerances
        >>> ipm = IPMSettings(tol_gap_abs=1e-6, tol_feas=1e-6)
        >>> settings = Settings(ipm_settings=ipm, device='cuda')
    """

    model_config = ConfigDict(validate_assignment=True)

    # Solver selection
    solver: Union[SolverType, str] = SolverType.AUTO

    # Device and batching settings
    device: str = "auto"
    device_id: int = -1  # CUDA device ID (-1 = use current device)
    batch_size: Annotated[int, Field(ge=1)] = 1
    enable_grad: bool = False
    auto_tune: bool = False
    yolo: bool = False
    yolo_num_iters: Annotated[int, Field(ge=1)] = 15

    # Core settings
    max_iter: Annotated[int, Field(ge=1)] = 200
    time_limit: Annotated[float, Field(gt=0)] = float("inf")
    verbose: bool = False

    # Solver-specific settings (tolerances are in here)
    ipm_settings: Optional[IPMSettings] = None
    active_set_settings: Optional[ActiveSetSettings] = None

    @field_validator("solver", mode="before")
    @classmethod
    def normalize_solver(cls, v):
        """Normalize solver to SolverType enum."""
        if isinstance(v, str):
            return SolverType(v.lower())
        return v

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate device selection."""
        valid = {"auto", "cpu", "cuda"}
        if v not in valid:
            raise ValueError(f"device must be one of {valid}, got '{v}'")
        return v

    @model_validator(mode="after")
    def create_default_ipm_settings(self):
        """Create default IPM settings if needed."""
        if self.ipm_settings is None and self.solver in (SolverType.IPM, SolverType.AUTO):
            self.ipm_settings = IPMSettings()
        return self

    @model_validator(mode="after")
    def validate_yolo(self):
        """Validate YOLO mode settings."""
        if self.yolo:
            if self.enable_grad:
                raise ValueError(
                    "yolo=True is incompatible with enable_grad=True "
                    "(backward pass needs convergence data)"
                )
            object.__setattr__(self, "verbose", False)
        return self

    def to_device_settings(self, device: str):
        """Convert to device-specific settings.

        Args:
            device: Device name (e.g., 'cuda')

        Returns:
            The native settings type for the specified device.

        Raises:
            ImportError: If the device backend is not available.
        """
        from moreau._backend import get_device_component, device_available

        if not device_available(device):
            raise ImportError(f"Device '{device}' not available.")

        settings_converter = get_device_component(device, "settings_converter")
        if settings_converter is None:
            raise ImportError(f"Device '{device}' has no settings_converter registered.")
        return settings_converter(self)


def _normalize_status(status) -> SolverStatus:
    """Convert backend status to unified SolverStatus IntEnum.

    Handles numpy arrays, lists, backend enums, and raw integers.
    Returns scalar SolverStatus for scalar input, list for array input.
    """
    import numpy as np

    if isinstance(status, SolverStatus):
        return status
    if isinstance(status, np.integer):
        return SolverStatus(int(status))
    if isinstance(status, np.ndarray):
        return [SolverStatus(int(s)) for s in status]
    if isinstance(status, list):
        return [SolverStatus(int(s)) for s in status]
    if hasattr(status, "__iter__") and not isinstance(status, (int, str)):
        return [SolverStatus(int(s)) for s in status]
    return SolverStatus(int(status))


@dataclass
class WarmStart:
    """Warm start point for a single conic optimization problem.

    Contains primal, dual, and slack variables from a previous solve
    that can be used to accelerate convergence on a related problem.

    Attributes:
        x: Primal variables, shape (n,)
        z: Dual variables, shape (m,)
        s: Slack variables, shape (m,)
        z_x: Direct cone duals, shape (sum |J|,) or None for problems
            without direct cones.
    """

    x: "np.ndarray"
    z: "np.ndarray"
    s: "np.ndarray"
    z_x: "Optional[np.ndarray]" = None

    def __repr__(self) -> str:
        n = len(self.x) if hasattr(self.x, "__len__") else 0
        return f"WarmStart(n={n})"


@dataclass
class BatchedWarmStart:
    """Warm start point for batched conic optimization problems.

    Contains primal, dual, and slack variables from a previous batched solve
    that can be used to accelerate convergence on related problems.

    Attributes:
        x: Primal variables, shape (batch_size, n)
        z: Dual variables, shape (batch_size, m)
        s: Slack variables, shape (batch_size, m)
        z_x: Direct cone duals, shape (batch_size, sum |J|) or None.
    """

    x: "np.ndarray"
    z: "np.ndarray"
    s: "np.ndarray"
    z_x: "Optional[np.ndarray]" = None

    def __repr__(self) -> str:
        batch_size = self.x.shape[0] if hasattr(self.x, "shape") else len(self.x)
        return f"BatchedWarmStart(batch_size={batch_size})"

    def __len__(self) -> int:
        return self.x.shape[0] if hasattr(self.x, "shape") else len(self.x)

    def __getitem__(self, idx: int) -> WarmStart:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for batch size {len(self)}")
        zx_i = self.z_x[idx] if self.z_x is not None else None
        return WarmStart(x=self.x[idx], z=self.z[idx], s=self.s[idx], z_x=zx_i)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


@dataclass
class SolveInfo:
    """Solver metadata from a solve operation.

    Contains solver status, objective value, iteration count, and timing info.

    Attributes:
        status: Solver status (SolverStatus enum)
        obj_val: Objective value at solution
        iterations: Number of iterations taken
        solve_time: Time spent in IPM iterations (seconds)
        setup_time: Time spent setting matrix values (seconds)
        construction_time: Time spent constructing solver structure (seconds)

    Example:
        >>> solution = solver.solve(q, b)
        >>> info = solver.info
        >>> print(info.status)
        >>> print(f"Solved in {info.iterations} iterations, {info.solve_time:.3f}s")
    """

    status: SolverStatus
    obj_val: float
    iterations: int
    solve_time: float
    setup_time: float = 0.0
    construction_time: float = 0.0

    def __repr__(self) -> str:
        return (
            f"SolveInfo(status={self.status.name}, "
            f"obj_val={self.obj_val:.6g}, "
            f"iterations={self.iterations})"
        )


@dataclass
class Solution:
    """Solution vectors for a single conic optimization problem.

    Contains only the optimization variables (primal and dual solutions).
    Returned as the first element of the (Solution, SolveInfo) tuple from solve().

    Attributes:
        x: Primal solution vector, shape (n,)
        z: Dual variable (Lagrange multipliers), shape (m,)
        s: Slack variable, shape (m,)
        z_x: Direct cone duals, flat over `Cones.dir_cones` in spec order
            (length = sum of cone dimensions). Empty array when the
            problem has no direct cones.

    Example:
        >>> solution = solver.solve(q, b)
        >>> print(solution.x)  # Access primal solution
        >>> print(solver.info.status)  # Check solver status
    """

    x: "np.ndarray"
    z: "np.ndarray"
    s: "np.ndarray"
    z_x: "np.ndarray" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.z_x is None:
            import numpy as _np

            self.z_x = _np.zeros(0, dtype=_np.float64)

    def __repr__(self) -> str:
        n = len(self.x) if hasattr(self.x, "__len__") else 0
        return f"Solution(n={n})"

    def to_warm_start(self) -> WarmStart:
        """Create a WarmStart from this solution."""
        zx = self.z_x.copy() if getattr(self, "z_x", None) is not None else None
        return WarmStart(x=self.x.copy(), z=self.z.copy(), s=self.s.copy(), z_x=zx)


@dataclass
class BatchedSolveInfo:
    """Solver metadata from a batched solve operation.

    Contains per-problem status and objective values, plus shared timing info.

    Attributes:
        status: List of solver statuses, one per problem
        obj_val: Objective values, shape (batch_size,)
        iterations: Iterations taken (typically same for all in batch)
        solve_time: Time spent in IPM iterations (seconds)
        setup_time: Time spent setting matrix values (seconds)
        construction_time: Time spent constructing solver structure (seconds)

    Example:
        >>> solution = solver.solve(q_batch, b_batch)
        >>> info = solver.info
        >>> print(info.status[0])  # Check first problem's status
        >>> print(f"Batch solved in {info.solve_time:.3f}s")
    """

    status: List[SolverStatus]
    obj_val: List[float]
    iterations: List[int]
    solve_time: float
    setup_time: float = 0.0
    construction_time: float = 0.0

    def __repr__(self) -> str:
        batch_size = len(self.status)
        return (
            f"BatchedSolveInfo(batch_size={batch_size}, "
            f"iterations={self.iterations}, solve_time={self.solve_time:.6f}s)"
        )


@dataclass
class BatchedSolution:
    """Solution vectors for batched conic optimization problems.

    Contains only the optimization variables (primal and dual solutions).
    Returned on calls to CompiledSolver's solve method.

    Attributes:
        x: Primal solutions, shape (batch_size, n)
        z: Dual variables, shape (batch_size, m)
        s: Slack variables, shape (batch_size, m)
        z_x: Direct cone duals, shape (batch_size, total_xn). Empty
            (shape `(batch_size, 0)`) when there are no direct cones.

    Example:
        >>> solution = solver.solve(q_batch, b_batch)
        >>> print(solution.x.shape)  # (batch_size, n)
        >>> sol = solution[1]  # Get Solution for 2nd problem
        >>> print(sol.x)  # 1D array for that problem
    """

    x: "np.ndarray"
    z: "np.ndarray"
    s: "np.ndarray"
    z_x: "np.ndarray" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.z_x is None:
            import numpy as _np

            batch_size = self.x.shape[0] if hasattr(self.x, "shape") else len(self.x)
            self.z_x = _np.zeros((batch_size, 0), dtype=_np.float64)

    def __repr__(self) -> str:
        batch_size = self.x.shape[0] if hasattr(self.x, "shape") else len(self.x)
        return f"BatchedSolution(batch_size={batch_size})"

    def __len__(self) -> int:
        """Return the batch size."""
        return self.x.shape[0] if hasattr(self.x, "shape") else len(self.x)

    def __getitem__(self, idx: int) -> Solution:
        """Get Solution for a single problem in the batch.

        Args:
            idx: Index of the problem in the batch (0-based)

        Returns:
            Solution object for that problem

        Example:
            >>> solution = solver.solve(q_batch, b_batch)
            >>> sol = solution[1]  # Get 2nd problem's solution
            >>> print(sol.x)     # 1D array
        """
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for batch size {len(self)}")

        return Solution(
            x=self.x[idx],
            z=self.z[idx],
            s=self.s[idx],
            z_x=self.z_x[idx] if self.z_x.size else self.z_x,
        )

    def __iter__(self):
        """Iterate over solutions in the batch."""
        for i in range(len(self)):
            yield self[i]

    def to_warm_start(self) -> BatchedWarmStart:
        """Create a BatchedWarmStart from this solution."""
        zx = self.z_x.copy() if getattr(self, "z_x", None) is not None else None
        return BatchedWarmStart(x=self.x.copy(), z=self.z.copy(), s=self.s.copy(), z_x=zx)


# Torch solution/info types live in moreau.torch._types; Jax types in
# moreau.jax._types. Re-exported here for backward compatibility — some
# callers (including the moreau-cuda package) import them from moreau._types.
# Lazy so importing _types does not pull in moreau.torch / moreau.jax.
_TORCH_TYPE_NAMES = frozenset(
    {
        "TorchSolveInfo",
        "TorchSolution",
        "TorchBatchedSolveInfo",
        "TorchBatchedSolution",
    }
)
_JAX_TYPE_NAMES = frozenset({"JaxSolveInfo", "JaxSolution"})


def __getattr__(name):
    if name in _TORCH_TYPE_NAMES:
        from .torch import _types as _torch_types

        return getattr(_torch_types, name)
    if name in _JAX_TYPE_NAMES:
        from .jax import _types as _jax_types

        return getattr(_jax_types, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@dataclass
class TuneResult:
    """Result from CompiledSolver auto-tune on first solve() benchmarking.

    Attributes:
        device: Winning device name (e.g. 'cuda', 'cpu'). When auto-tune runs
            with device='auto', this reflects the best device found across all
            available devices. When an explicit device was set, this is that device.
        method: Winning KKT solver method name (e.g. 'qdldl', 'cudss', 'faer').
        time_limit: Time limit set for future solves (seconds), computed as
            best_time * margin.
        results: Per-method benchmark data. Keys are 'device:method' strings
            (e.g. 'cuda:qdldl') when cross-device tuning, or plain method names
            when single-device. Values are dicts with 'solve_time' (float),
            'iterations' (list[int]), and 'status' (list[SolverStatus]).
    """

    device: str
    method: str
    time_limit: float
    results: Dict[str, dict] = field(default_factory=dict)

    def __repr__(self) -> str:
        methods = ", ".join(f"{m}: {r['solve_time']:.4f}s" for m, r in self.results.items())
        return (
            f"TuneResult(device='{self.device}', method='{self.method}', "
            f"time_limit={self.time_limit:.4f}s, {{{methods}}})"
        )


# Statuses that do NOT trigger a cold retry when warm_start is provided.
# Shared across Solver, CompiledSolver, JAX Solver, and Torch Solver classes.
_WARM_START_NO_RETRY: FrozenSet[SolverStatus] = frozenset(
    {
        SolverStatus.Solved,
        SolverStatus.AlmostSolved,
        SolverStatus.MaxIterations,
        SolverStatus.CallbackTerminated,
    }
)

# Margin factor used during auto-tune: time-limit challengers after a baseline
# is established, and accept the winner only if it beats current by this factor.
_AUTO_TUNE_MARGIN: float = 1.5


def _resolve_no_retry_set(ipm_settings):
    """Resolve the warm-start no-retry status set from IPMSettings.

    Returns the user-provided `ipm.warm_start_no_retry` when set, else the
    default `_WARM_START_NO_RETRY` frozenset. Centralised so Solver,
    CompiledSolver, torch.Solver, and jax.Solver all agree on the same
    resolution.
    """
    if ipm_settings is not None and ipm_settings.warm_start_no_retry is not None:
        return ipm_settings.warm_start_no_retry
    return _WARM_START_NO_RETRY


def _should_retry_cold(statuses, ipm_settings):
    """Return the list of batch indices that need a cold retry.

    `statuses` is either a single `SolverStatus` (returns `[0]` or `[]`)
    or a list/array of `SolverStatus` (returns the indices with status
    not in the no-retry set). The IPM settings' `warm_start_no_retry`
    set takes precedence over the default; see `_resolve_no_retry_set`.

    Centralised so all four call sites (Solver, CompiledSolver,
    torch.Solver, jax.Solver) agree on which statuses trigger retry.
    """
    no_retry = _resolve_no_retry_set(ipm_settings)
    if isinstance(statuses, (list, tuple)):
        return [i for i, s in enumerate(statuses) if s not in no_retry]
    # Single-status path: collapse to a list-of-one indicator.
    return [0] if statuses not in no_retry else []


def _warn_warm_retry(failed_idx, statuses):
    """Emit the standard warm-start retry warning.

    Format differs slightly for single vs batched paths; centralised so
    the wording stays consistent across Solver and CompiledSolver.
    """
    import warnings as _warnings

    if isinstance(statuses, (list, tuple)):
        names = ", ".join(statuses[i].name for i in failed_idx)
        _warnings.warn(
            f"Warm-started solve failed at batch indices {list(failed_idx)} "
            f"({names}), retrying those elements without warm start.",
            stacklevel=3,
        )
    else:
        _warnings.warn(
            f"Warm-started solve failed ({statuses.name}), " f"retrying without warm start.",
            stacklevel=3,
        )


__all__ = [
    "SolverType",
    "SolverStatus",
    "Cones",
    "DirectConeSpec",
    "Settings",
    "IPMSettings",
    "ActiveSetSettings",
    "_normalize_status",
    "WarmStart",
    "BatchedWarmStart",
    "SolveInfo",
    "Solution",
    "BatchedSolveInfo",
    "BatchedSolution",
    # Re-exported lazily from moreau.torch._types / moreau.jax._types (see __getattr__).
    "TorchSolveInfo",
    "TorchSolution",
    "TorchBatchedSolveInfo",
    "TorchBatchedSolution",
    "JaxSolveInfo",
    "JaxSolution",
    "TuneResult",
    "_WARM_START_NO_RETRY",
    "_AUTO_TUNE_MARGIN",
    "_resolve_no_retry_set",
    "_should_retry_cold",
    "_warn_warm_retry",
]
