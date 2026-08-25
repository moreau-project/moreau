"""The JaxSolverCuda class and its global solver registry."""

from functools import partial
from typing import Optional
import atexit

import numpy as np

import jax
import jax.numpy as jnp

from moreau._types import Cones, Settings

from ._ffi import _get_ffi_lib, ffi_available
from ._lowering import _make_ffi_solve_fn, _make_ffi_solve_warm_fn

# =============================================================================
# Global Solver Registry (Python side - for fallback only)
# =============================================================================

_SOLVER_REGISTRY = {}


def _cleanup_solvers():
    """Clean up all solver instances before Python shutdown.

    This prevents "driver shutting down" errors by explicitly releasing
    CUDA resources before the CUDA driver starts its shutdown sequence.
    """
    global _SOLVER_REGISTRY
    if _SOLVER_REGISTRY is not None:
        for solver in list(_SOLVER_REGISTRY.values()):
            if hasattr(solver, "_cuda_solvers"):
                solver._cuda_solvers.clear()
        _SOLVER_REGISTRY.clear()


atexit.register(_cleanup_solvers)


class JaxSolverCuda:
    """CUDA JAX solver implementation using XLA FFI zero-copy.

    This class wraps the moreau-cuda solver to provide a JAX-compatible
    interface with vmap and gradient support. Uses XLA FFI for true zero-copy
    GPU tensor sharing between JAX and the CUDA solver.
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
        b_sparsity_pattern=None,
    ):
        import time

        start = time.perf_counter()

        self._n = n
        self._m = m
        self._cones = cones
        self._settings = settings if settings is not None else Settings()
        self._b_sparsity_pattern = (
            list(b_sparsity_pattern) if b_sparsity_pattern is not None else None
        )

        from moreau._types import SolverType as _PySolverType

        if self._settings.solver == _PySolverType.ACTIVE_SET:
            raise ValueError(
                "solver='active_set' is not supported through the CUDA JAX wrapper. "
                "Use device='cpu' or solver='ipm' for GPU JAX arrays."
            )

        # Convert to numpy arrays (these will be placed on GPU as needed)
        self._P_row_offsets = np.asarray(P_row_offsets, dtype=np.int64)
        self._P_col_indices = np.asarray(P_col_indices, dtype=np.int64)
        self._A_row_offsets = np.asarray(A_row_offsets, dtype=np.int64)
        self._A_col_indices = np.asarray(A_col_indices, dtype=np.int64)

        self._nnzP = len(self._P_col_indices)
        self._nnzA = len(self._A_col_indices)

        # Check if FFI is available
        self._ffi_lib = _get_ffi_lib()
        self._use_ffi = ffi_available()

        # Convert structure arrays to JAX arrays on GPU for FFI
        if self._use_ffi:
            cuda_device = jax.devices("cuda")[0]
            self._P_row_offsets_gpu = jax.device_put(
                jnp.array(self._P_row_offsets, dtype=jnp.int64),
                cuda_device,
            )
            self._P_col_indices_gpu = jax.device_put(
                jnp.array(self._P_col_indices, dtype=jnp.int64),
                cuda_device,
            )
            self._A_row_offsets_gpu = jax.device_put(
                jnp.array(self._A_row_offsets, dtype=jnp.int64),
                cuda_device,
            )
            self._A_col_indices_gpu = jax.device_put(
                jnp.array(self._A_col_indices, dtype=jnp.int64),
                cuda_device,
            )
            # SOC dims buffer (empty array if no SOC cones)
            soc_dims_list = list(cones.so_cone_dims) if cones.so_cone_dims else []
            self._soc_dims_gpu = jax.device_put(
                jnp.array(soc_dims_list, dtype=jnp.int64),
                cuda_device,
            )
            # PSD dims buffer (empty array if no PSD cones)
            psd_dims_list = (
                list(cones.psd_dims) if hasattr(cones, "psd_dims") and cones.psd_dims else []
            )
            self._psd_dims_gpu = jax.device_put(
                jnp.array(psd_dims_list, dtype=jnp.int64),
                cuda_device,
            )
            # Power cone alphas buffer
            power_alphas_list = list(cones.power_alphas) if cones.power_alphas else []
            self._power_alphas_gpu = jax.device_put(
                jnp.array(power_alphas_list, dtype=jnp.float64),
                cuda_device,
            )
            # GenPowerCone buffers
            gp_params = getattr(cones, "gen_power_cone_params", None) or []
            self._num_gen_power = len(gp_params)
            if gp_params:
                gp_alphas = []
                gp_dim1s = []
                gp_dim2s = []
                for alphas, dim2 in gp_params:
                    gp_alphas.extend(alphas)
                    gp_dim1s.append(len(alphas))
                    gp_dim2s.append(dim2)
                self._gen_power_alphas_gpu = jax.device_put(
                    jnp.array(gp_alphas, dtype=jnp.float64), cuda_device
                )
                self._gen_power_dim1s_gpu = jax.device_put(
                    jnp.array(gp_dim1s, dtype=jnp.int64), cuda_device
                )
                self._gen_power_dim2s_gpu = jax.device_put(
                    jnp.array(gp_dim2s, dtype=jnp.int64), cuda_device
                )
            else:
                self._gen_power_alphas_gpu = jax.device_put(
                    jnp.array([], dtype=jnp.float64), cuda_device
                )
                self._gen_power_dim1s_gpu = jax.device_put(
                    jnp.array([], dtype=jnp.int64), cuda_device
                )
                self._gen_power_dim2s_gpu = jax.device_put(
                    jnp.array([], dtype=jnp.int64), cuda_device
                )

            # Direct-x cone descriptors. We materialise the same flat
            # arrays the FFI handler reads (kinds + indices_offsets +
            # indices_flat + per-kind parameter slices). Keeps the FFI
            # signature stable across direct-x / slack-only problems —
            # zero-length buffers when no x_cones are present.
            x_specs = list(getattr(cones, "x_cones", None) or [])
            self._num_x_cones = len(x_specs)
            x_kinds, x_idx_off, x_idx_flat = [], [0], []
            x_pow_alphas, x_psd_ks = [], []
            x_gp_dim1s, x_gp_dim2s, x_gp_alphas_flat = [], [], []
            kind_map = {
                "nonneg": 0,
                "soc": 1,
                "psd_triangle": 2,
                "exp": 3,
                "power": 4,
                "gen_power": 5,
            }
            for spec in x_specs:
                kind_str = (getattr(spec, "kind", "") or "").lower()
                if kind_str not in kind_map:
                    raise ValueError(
                        f"Unknown XConeSpec.kind={kind_str!r}; "
                        "supported: nonneg, soc, psd_triangle, exp, power, gen_power"
                    )
                x_kinds.append(kind_map[kind_str])
                inds = list(spec.indices)
                x_idx_flat.extend(inds)
                x_idx_off.append(x_idx_off[-1] + len(inds))
                if kind_str == "power":
                    alpha = getattr(spec, "alpha", None)
                    if alpha is None:
                        raise ValueError("XConeSpec(kind='power') requires alpha")
                    x_pow_alphas.append(float(alpha))
                elif kind_str == "psd_triangle":
                    psd_k = getattr(spec, "psd_k", None)
                    if psd_k is None:
                        raise ValueError("XConeSpec(kind='psd_triangle') requires psd_k")
                    x_psd_ks.append(int(psd_k))
                elif kind_str == "gen_power":
                    alphas = list(getattr(spec, "alphas", []) or [])
                    dim2 = getattr(spec, "dim2", None)
                    if not alphas or dim2 is None:
                        raise ValueError("XConeSpec(kind='gen_power') requires alphas and dim2")
                    x_gp_dim1s.append(len(alphas))
                    x_gp_dim2s.append(int(dim2))
                    x_gp_alphas_flat.extend(alphas)
            self._total_xn = x_idx_off[-1] if x_idx_off else 0
            self._x_kinds_gpu = jax.device_put(jnp.array(x_kinds, dtype=jnp.int64), cuda_device)
            self._x_indices_offsets_gpu = jax.device_put(
                jnp.array(x_idx_off, dtype=jnp.int64), cuda_device
            )
            self._x_indices_flat_gpu = jax.device_put(
                jnp.array(x_idx_flat, dtype=jnp.int64), cuda_device
            )
            self._x_power_alphas_gpu = jax.device_put(
                jnp.array(x_pow_alphas, dtype=jnp.float64), cuda_device
            )
            self._x_psd_ks_gpu = jax.device_put(jnp.array(x_psd_ks, dtype=jnp.int64), cuda_device)
            self._x_gen_power_dim1s_gpu = jax.device_put(
                jnp.array(x_gp_dim1s, dtype=jnp.int64), cuda_device
            )
            self._x_gen_power_dim2s_gpu = jax.device_put(
                jnp.array(x_gp_dim2s, dtype=jnp.int64), cuda_device
            )
            self._x_gen_power_alphas_gpu = jax.device_put(
                jnp.array(x_gp_alphas_flat, dtype=jnp.float64), cuda_device
            )

        # For pure_callback fallback (always initialize for robustness)
        from .. import _cones_to_cuda, _settings_to_cuda, _CudaSettings

        self._cuda_cones = _cones_to_cuda(cones)
        self._cuda_settings = (
            _settings_to_cuda(self._settings) if self._settings else _CudaSettings()
        )
        self._cuda_solvers = {}

        # Register in Python registry (for fallback mode)
        self._solver_id = id(self)
        _SOLVER_REGISTRY[self._solver_id] = self

        self._pending_warm_start = None
        self._ffi_solve_warm_fn = None
        self._construction_time = time.perf_counter() - start

    def _get_or_create_cuda_solver(self, batch_size: int, enable_grad: bool = True):
        """Get or create a CUDA solver for fallback mode."""
        from .._moreau_cuda import Solver as _RawCudaSolver

        key = (batch_size, enable_grad)
        if key not in self._cuda_solvers:
            solver = _RawCudaSolver(
                self._n,
                self._m,
                batch_size,
                self._P_row_offsets,
                self._P_col_indices,
                self._A_row_offsets,
                self._A_col_indices,
                self._cuda_cones,
                self._cuda_settings,
                enable_grad,
                b_sparsity_pattern=self._b_sparsity_pattern,
            )
            self._cuda_solvers[key] = solver
        return self._cuda_solvers[key]

    def __del__(self):
        """Clean up registered solvers.

        Note: We explicitly clear CUDA solver references to avoid "driver shutting down"
        errors during Python interpreter shutdown. The order of cleanup matters.
        """
        # Clear CUDA solvers first (before registry cleanup)
        if hasattr(self, "_cuda_solvers"):
            self._cuda_solvers.clear()

        # Clear GPU arrays to release JAX references
        for attr in (
            "_P_row_offsets_gpu",
            "_P_col_indices_gpu",
            "_A_row_offsets_gpu",
            "_A_col_indices_gpu",
            "_soc_dims_gpu",
            "_psd_dims_gpu",
            "_power_alphas_gpu",
            "_gen_power_alphas_gpu",
            "_gen_power_dim1s_gpu",
            "_gen_power_dim2s_gpu",
        ):
            if hasattr(self, attr):
                try:
                    delattr(self, attr)
                except Exception:
                    pass

        # Remove from registry
        if hasattr(self, "_solver_id"):
            try:
                if _SOLVER_REGISTRY is not None and self._solver_id in _SOLVER_REGISTRY:
                    del _SOLVER_REGISTRY[self._solver_id]
            except (TypeError, KeyError, ReferenceError):
                pass  # Registry may be None during interpreter shutdown

    @property
    def solve(self):
        """Return the pure solve function.

        Uses XLA FFI for true zero-copy GPU tensor sharing when available,
        with a custom vmap rule to handle structure arrays correctly.
        Falls back to pure_callback when FFI is not available.

        Returns a function that returns (JaxSolution, JaxSolveInfo) tuple.
        """
        if self._use_ffi:
            # Create a solve function with static values captured in closures
            # This avoids tracer issues with custom_vmap
            if not hasattr(self, "_ffi_solve_fn"):
                self._ffi_solve_fn = _make_ffi_solve_fn(
                    self._n,
                    self._m,
                    self._cones.num_zero_cones,
                    self._cones.num_nonneg_cones,
                    getattr(self._cones, "num_exp_cones", 0),
                    self._cones.num_so_cones,
                    (
                        len(self._cones.psd_dims)
                        if hasattr(self._cones, "psd_dims") and self._cones.psd_dims
                        else 0
                    ),
                    self._cones.num_power_cones,
                    self._num_gen_power,
                    self._num_x_cones,
                    self._total_xn,
                    self._P_row_offsets_gpu,
                    self._P_col_indices_gpu,
                    self._A_row_offsets_gpu,
                    self._A_col_indices_gpu,
                    self._soc_dims_gpu,
                    self._psd_dims_gpu,
                    self._power_alphas_gpu,
                    self._gen_power_alphas_gpu,
                    self._gen_power_dim1s_gpu,
                    self._gen_power_dim2s_gpu,
                    self._x_kinds_gpu,
                    self._x_indices_offsets_gpu,
                    self._x_indices_flat_gpu,
                    self._x_power_alphas_gpu,
                    self._x_psd_ks_gpu,
                    self._x_gen_power_dim1s_gpu,
                    self._x_gen_power_dim2s_gpu,
                    self._x_gen_power_alphas_gpu,
                    self._construction_time,
                )
            return self._ffi_solve_fn
        else:
            from ._fallback import _solve_fallback_with_solution

            return partial(_solve_fallback_with_solution, self._solver_id, self._construction_time)

    @property
    def solve_warm(self):
        """Return a solve function that accepts warm start arrays.

        Uses XLA FFI warm-start handler for true zero-copy GPU tensor sharing.
        Falls back to pure_callback with solve_warm_start when FFI is not available.

        Returns a function:
            (P_data, A_data, q, b, warm_x, warm_z, warm_s) -> (JaxSolution, JaxSolveInfo)
        """
        if self._use_ffi:
            if self._ffi_solve_warm_fn is None:
                self._ffi_solve_warm_fn = _make_ffi_solve_warm_fn(
                    self._n,
                    self._m,
                    self._cones.num_zero_cones,
                    self._cones.num_nonneg_cones,
                    getattr(self._cones, "num_exp_cones", 0),
                    self._cones.num_so_cones,
                    (
                        len(self._cones.psd_dims)
                        if hasattr(self._cones, "psd_dims") and self._cones.psd_dims
                        else 0
                    ),
                    self._cones.num_power_cones,
                    self._num_gen_power,
                    self._num_x_cones,
                    self._total_xn,
                    self._P_row_offsets_gpu,
                    self._P_col_indices_gpu,
                    self._A_row_offsets_gpu,
                    self._A_col_indices_gpu,
                    self._soc_dims_gpu,
                    self._psd_dims_gpu,
                    self._power_alphas_gpu,
                    self._gen_power_alphas_gpu,
                    self._gen_power_dim1s_gpu,
                    self._gen_power_dim2s_gpu,
                    self._x_kinds_gpu,
                    self._x_indices_offsets_gpu,
                    self._x_indices_flat_gpu,
                    self._x_power_alphas_gpu,
                    self._x_psd_ks_gpu,
                    self._x_gen_power_dim1s_gpu,
                    self._x_gen_power_dim2s_gpu,
                    self._x_gen_power_alphas_gpu,
                    self._construction_time,
                )
            return self._ffi_solve_warm_fn
        else:
            from ._fallback import _solve_fallback_warm_with_solution

            return partial(
                _solve_fallback_warm_with_solution, self._solver_id, self._construction_time
            )
