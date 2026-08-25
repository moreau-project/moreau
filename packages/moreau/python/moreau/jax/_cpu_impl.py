"""CPU JAX implementation using pure_callback to wrap moreau-cpu solver.

This provides a JAX-compatible interface to the CPU solver with:
- jax.vmap support for batching
- jax.grad support via custom_vjp
- Works without GPU or CUDA

Requires JAX >= 0.6.0 for pure_callback vmap_method parameter.
"""

from functools import partial
from typing import Optional, Tuple, Any
import weakref
import numpy as np

import jax
import jax.numpy as jnp
from jax import custom_vjp

from moreau._types import Cones, Settings
from ._types import JaxSolution, JaxSolveInfo


def _ensure_x64_enabled():
    """Ensure JAX 64-bit mode is enabled (required for solver's numerical precision).

    This is called lazily when the solver is first used, rather than at import time,
    to avoid polluting global JAX config for users who import but don't use the JAX solver.
    """
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)


# Global registry for solver instances (keyed by solver_id).
# WeakValueDictionary so the registry doesn't keep wrappers alive after the
# user drops their reference — entries auto-evict on GC.
_SOLVER_REGISTRY: "weakref.WeakValueDictionary[int, JaxSolverCpu]" = weakref.WeakValueDictionary()
_SOLVER_ID_COUNTER = 0


def _get_next_solver_id() -> int:
    """Get the next unique solver ID."""
    global _SOLVER_ID_COUNTER
    _SOLVER_ID_COUNTER += 1
    return _SOLVER_ID_COUNTER


class JaxSolverCpu:
    """CPU JAX solver implementation using pure_callback.

    This class wraps the moreau-cpu solver to provide a JAX-compatible
    interface with vmap and gradient support.
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
        # Ensure JAX 64-bit mode is enabled (lazy, only when solver is created)
        _ensure_x64_enabled()

        import time

        start = time.perf_counter()

        self._n = n
        self._m = m
        self._cones = cones
        self._settings = settings if settings is not None else Settings()
        self._b_sparsity_pattern = (
            list(b_sparsity_pattern) if b_sparsity_pattern is not None else None
        )

        # Convert to numpy arrays
        self._P_row_offsets = np.asarray(P_row_offsets, dtype=np.int64)
        self._P_col_indices = np.asarray(P_col_indices, dtype=np.int64)
        self._A_row_offsets = np.asarray(A_row_offsets, dtype=np.int64)
        self._A_col_indices = np.asarray(A_col_indices, dtype=np.int64)

        self._nnzP = len(self._P_col_indices)
        self._nnzA = len(self._A_col_indices)

        # Direct-x cones: total dim across all x-cones, used to allocate the
        # z_x output buffer shape (and dz_x input shape) at the FFI boundary.
        # 0 for slack-only problems.
        self._total_x_dim = sum(
            len(getattr(xc, "indices", [])) for xc in (getattr(cones, "x_cones", None) or [])
        )

        # Register this solver in the global registry
        self._solver_id = _get_next_solver_id()
        _SOLVER_REGISTRY[self._solver_id] = self

        # Lazy initialization of underlying solver (created per batch size)
        self._cpu_solvers = {}  # batch_size -> moreau_cpu.Solver

        # Cache for P/A values to skip setup() when unchanged
        # Key: batch_size, Value: (P_data_bytes, A_data_bytes)
        self._cached_PA = {}

        self._pending_warm_start = None
        self._construction_time = time.perf_counter() - start

        # Last solve info (populated after solve())
        self._info = None

        # Warm start data (set by Solver.solve() before callback runs)
        self._pending_warm_start = None

    def _needs_setup(self, batch_size: int, P_data: np.ndarray, A_data: np.ndarray) -> bool:
        """Check if setup() needs to be called (P/A values changed)."""
        if batch_size not in self._cached_PA:
            return True
        cached_P, cached_A = self._cached_PA[batch_size]
        # Compare bytes for speed (avoids element-wise comparison)
        return P_data.tobytes() != cached_P or A_data.tobytes() != cached_A

    def _update_cache(self, batch_size: int, P_data: np.ndarray, A_data: np.ndarray):
        """Update the P/A cache after setup()."""
        self._cached_PA[batch_size] = (P_data.tobytes(), A_data.tobytes())

    def _get_or_create_solver(self, batch_size: int, b: Optional[np.ndarray] = None):
        """Get or create a CPU solver for the given batch size."""
        if batch_size not in self._cpu_solvers:
            # Check if active-set solver is requested
            solver_type = getattr(self._settings, "solver", None)
            use_active_set = solver_type is not None and "active_set" in str(solver_type).lower()
            if use_active_set:
                from moreau_cpu._cpu import ActiveSetSolver

                settings_with_grad = self._settings.model_copy(update={"enable_grad": True})
                solver = ActiveSetSolver(
                    self._n,
                    self._m,
                    self._P_row_offsets,
                    self._P_col_indices,
                    self._A_row_offsets,
                    self._A_col_indices,
                    self._cones,
                    settings_with_grad,
                    batch_size=batch_size,
                    enable_grad=True,
                )
            else:
                import moreau_cpu
                from moreau._backend import _settings_to_cpu

                settings_with_grad = self._settings.model_copy(update={"enable_grad": True})
                cpu_settings = _settings_to_cpu(settings_with_grad)
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
                solver = moreau_cpu.Solver(
                    self._n,
                    self._m,
                    self._P_row_offsets,
                    self._P_col_indices,
                    self._A_row_offsets,
                    self._A_col_indices,
                    self._cones,
                    cpu_settings,
                    batch_size=batch_size,
                    enable_grad=True,
                    b_sparsity_pattern=b_sparsity_pattern,
                )
            self._cpu_solvers[batch_size] = solver

        return self._cpu_solvers[batch_size]

    @property
    def solve(self):
        """Return the pure solve function.

        Attaches `self` as an attribute on the returned callable so that the
        WeakValueDictionary entry in `_SOLVER_REGISTRY` is kept alive for as
        long as the user holds the returned solve function. (Without this,
        the legacy `solver(...)` API — which returns only a bound partial —
        would let the wrapper be GC'd before any callbacks could run.)
        """
        sid = self._solver_id

        def solve_fn(P_data, A_data, q, b):
            return _solve_cpu(sid, P_data, A_data, q, b)

        solve_fn._wrapper = self  # noqa: keeps registry entry alive
        return solve_fn

    @property
    def solve_warm(self):
        """CPU backend uses _pending_warm_start side channel, no dedicated warm path."""
        return None


def _solve_cpu_callback(
    solver_id: int,
    P_data: np.ndarray,
    A_data: np.ndarray,
    q: np.ndarray,
    b: np.ndarray,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Callback function that runs on CPU (called from pure_callback).

    Returns 10 arrays: x, z, s, z_x, status, obj_val, iterations, solve_time,
    setup_time, construction_time. `z_x` has zero length on slack-only
    problems (total_x_dim = 0).
    """
    solver_wrapper = _SOLVER_REGISTRY[solver_id]

    # Convert to float64 (JAX may pass float32 by default)
    P_data = np.asarray(P_data, dtype=np.float64)
    A_data = np.asarray(A_data, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    # Determine batch size
    if q.ndim == 1:
        batch_size = 1
        # Expand to 2D
        P_data = P_data.reshape(1, -1)
        A_data = A_data.reshape(1, -1)
        q = q.reshape(1, -1)
        b = b.reshape(1, -1)
        squeeze_output = True
    else:
        batch_size = q.shape[0]
        squeeze_output = False

    # Get or create solver for this batch size
    cpu_solver = solver_wrapper._get_or_create_solver(batch_size, b)

    # Setup matrices only if P/A values changed
    if solver_wrapper._needs_setup(batch_size, P_data, A_data):
        cpu_solver.setup(P_data, A_data)
        solver_wrapper._update_cache(batch_size, P_data, A_data)

    # Check for pending warm start
    warm_kwargs = {}
    if solver_wrapper._pending_warm_start is not None:
        pending_warm = solver_wrapper._pending_warm_start
        warm_kwargs["warm_x"] = np.ascontiguousarray(pending_warm["warm_x"].reshape(batch_size, -1))
        warm_kwargs["warm_z"] = np.ascontiguousarray(pending_warm["warm_z"].reshape(batch_size, -1))
        warm_kwargs["warm_s"] = np.ascontiguousarray(pending_warm["warm_s"].reshape(batch_size, -1))

    # Solve
    result = cpu_solver.solve(q, b, **warm_kwargs)

    x = np.asarray(result["x"], dtype=np.float64)
    z = np.asarray(result["z"], dtype=np.float64)
    s = np.asarray(result["s"], dtype=np.float64)

    # Direct-x cone duals. CompiledSolver returns `z_x` only when x_cones
    # are present; otherwise we synthesise a zero-length buffer with the
    # right batch dimension. Active-set CPU solver is zero+nonneg-only and
    # does not include z_x in its result dict — for solver='auto' the
    # IPM path is selected when x_cones are present, so reaching this
    # branch with `z_x` absent should be impossible. Fall back to zeros
    # defensively.
    total_xn = solver_wrapper._total_x_dim
    if total_xn > 0:
        z_x_raw = result.get("z_x")
        if z_x_raw is None:
            z_x = np.zeros((batch_size, total_xn), dtype=np.float64)
        else:
            z_x = np.asarray(z_x_raw, dtype=np.float64)
            if z_x.ndim == 1:
                z_x = z_x.reshape(1, total_xn)
    else:
        z_x = np.zeros((batch_size, 0), dtype=np.float64)

    # Extract metadata (per-problem fields are lists, timing is scalar)
    # status/obj_val/iterations are lists of length batch_size
    # solve_time/setup_time/construction_time are scalars (total time)
    status_list = result.get("status", [0] * batch_size)
    obj_val_list = result.get("obj_val", [0.0] * batch_size)
    iterations_list = result.get("iterations", [0] * batch_size)

    # Convert status enum to int if needed
    status_vals = [int(s) if hasattr(s, "value") else int(s) for s in status_list]

    # Get timing values (scalar - total time for all problems)
    solve_time_val = float(result.get("solve_time", 0.0))
    setup_time_val = float(result.get("setup_time", 0.0))
    construction_time_val = float(solver_wrapper._construction_time)

    if squeeze_output:
        # Single problem: return scalar metadata
        x = x.squeeze(0)
        z = z.squeeze(0)
        s = s.squeeze(0)
        z_x = z_x.squeeze(0)
        status = np.asarray(status_vals[0], dtype=np.float64)
        obj_val = np.asarray(obj_val_list[0], dtype=np.float64)
        iterations = np.asarray(iterations_list[0], dtype=np.float64)
        # Timing: scalar for single problem
        solve_time = np.asarray(solve_time_val, dtype=np.float64)
        setup_time = np.asarray(setup_time_val, dtype=np.float64)
        construction_time = np.asarray(construction_time_val, dtype=np.float64)
    else:
        # Batched: return arrays (timing broadcast to batch size for vmap compat)
        status = np.asarray(status_vals, dtype=np.float64)
        obj_val = np.asarray(obj_val_list, dtype=np.float64)
        iterations = np.asarray(iterations_list, dtype=np.float64)
        solve_time = np.full(batch_size, solve_time_val, dtype=np.float64)
        setup_time = np.full(batch_size, setup_time_val, dtype=np.float64)
        construction_time = np.full(batch_size, construction_time_val, dtype=np.float64)

    return (x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time)


def _backward_cpu_callback(
    solver_id: int,
    dx: np.ndarray,
    dz: np.ndarray,
    ds: np.ndarray,
    dz_x: np.ndarray,
    P_data: np.ndarray,
    A_data: np.ndarray,
    q: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    s: np.ndarray,
    z_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Callback function for backward pass.

    Returns 4 arrays: dP, dA, dq, db
    """
    solver_wrapper = _SOLVER_REGISTRY[solver_id]

    # Convert to float64
    dx = np.asarray(dx, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    ds = np.asarray(ds, dtype=np.float64)
    dz_x = np.asarray(dz_x, dtype=np.float64)
    P_data = np.asarray(P_data, dtype=np.float64)
    A_data = np.asarray(A_data, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    s = np.asarray(s, dtype=np.float64)
    z_x = np.asarray(z_x, dtype=np.float64)

    # Determine batch size
    if dx.ndim == 1:
        batch_size = 1
        # Expand to 2D
        dx = dx.reshape(1, -1)
        dz = dz.reshape(1, -1)
        ds = ds.reshape(1, -1)
        dz_x = dz_x.reshape(1, -1)
        P_data = P_data.reshape(1, -1)
        A_data = A_data.reshape(1, -1)
        q = q.reshape(1, -1)
        b = b.reshape(1, -1)
        x = x.reshape(1, -1)
        z = z.reshape(1, -1)
        s = s.reshape(1, -1)
        z_x = z_x.reshape(1, -1)
        squeeze_output = True
    else:
        batch_size = dx.shape[0]
        squeeze_output = False

    # Get solver
    cpu_solver = solver_wrapper._get_or_create_solver(batch_size)

    # Use backward_with_data_flat if available (IPM CompiledSolver) — no re-solve.
    # Falls back to re-solve for active-set solver.
    has_dz_x = dz_x.size > 0
    if hasattr(cpu_solver, "backward_with_data_flat"):
        kw = {}
        if has_dz_x:
            kw["dz_x_flat"] = dz_x.ravel()
            kw["z_x_flat"] = z_x.ravel()
        if hasattr(cpu_solver, "_last_backward_state"):
            backward_state = getattr(cpu_solver, "_last_backward_state", None)
            if backward_state is None:
                raise RuntimeError(
                    "Active-set backward requires cached backward state from the forward solve"
                )
            grad_result = cpu_solver.backward_with_data_flat(
                dx.ravel(),
                ds.ravel(),
                dz.ravel(),
                P_data.ravel(),
                A_data.ravel(),
                q.ravel(),
                b.ravel(),
                x.ravel(),
                z.ravel(),
                s.ravel(),
                backward_state,
                batch_size,
                **kw,
            )
        else:
            grad_result = cpu_solver.backward_with_data_flat(
                dx.ravel(),
                ds.ravel(),
                dz.ravel(),
                P_data.ravel(),
                A_data.ravel(),
                q.ravel(),
                b.ravel(),
                x.ravel(),
                z.ravel(),
                s.ravel(),
                batch_size,
                **kw,
            )
        # backward_with_data_flat mutates the underlying compiled solver state
        # while bypassing setup(), so the next forward solve must not trust the
        # cached "P/A already loaded" marker. (From #125.)
        solver_wrapper._cached_PA.pop(batch_size, None)
    else:
        # Active-set fallback: setup + re-solve + backward
        if solver_wrapper._needs_setup(batch_size, P_data, A_data):
            cpu_solver.setup(P_data, A_data)
            solver_wrapper._update_cache(batch_size, P_data, A_data)
        cpu_solver.solve(q, b)
        kw = {}
        if has_dz_x:
            kw["dz_x"] = dz_x[0] if batch_size == 1 else dz_x
        if batch_size == 1:
            grad_result = cpu_solver.backward(dx[0], dz[0], ds[0], **kw)
        else:
            grad_result = cpu_solver.backward(dx, dz, ds, **kw)

    dP = np.asarray(grad_result["dP_values"], dtype=np.float64)
    dA = np.asarray(grad_result["dA_values"], dtype=np.float64)
    dq = np.asarray(grad_result["dq"], dtype=np.float64)
    db = np.asarray(grad_result["db"], dtype=np.float64)

    if squeeze_output:
        # truly unbatched (input was 1D) — squeeze to 1D
        dP = dP.squeeze(0) if dP.ndim > 1 else dP
        dA = dA.squeeze(0) if dA.ndim > 1 else dA
        dq = dq.squeeze(0) if dq.ndim > 1 else dq
        db = db.squeeze(0) if db.ndim > 1 else db
    elif batch_size == 1:
        # batch_size=1 via vmap — ensure 2D to match declared shapes
        dP = dP.reshape(1, -1) if dP.ndim == 1 else dP
        dA = dA.reshape(1, -1) if dA.ndim == 1 else dA
        dq = dq.reshape(1, -1) if dq.ndim == 1 else dq
        db = db.reshape(1, -1) if db.ndim == 1 else db

    return dP, dA, dq, db


@partial(custom_vjp, nondiff_argnums=(0,))
def _solve_cpu_raw(
    solver_id: int,
    P_data: jnp.ndarray,
    A_data: jnp.ndarray,
    q: jnp.ndarray,
    b: jnp.ndarray,
) -> Tuple[jnp.ndarray, ...]:
    """Solve conic QP on CPU. Returns (x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time)."""
    solver_wrapper = _SOLVER_REGISTRY[solver_id]
    n, m = solver_wrapper._n, solver_wrapper._m
    total_xn = solver_wrapper._total_x_dim

    # Determine output shapes
    scalar_shape = jax.ShapeDtypeStruct((), jnp.float64)
    if q.ndim == 1:
        x_shape = jax.ShapeDtypeStruct((n,), jnp.float64)
        z_shape = jax.ShapeDtypeStruct((m,), jnp.float64)
        s_shape = jax.ShapeDtypeStruct((m,), jnp.float64)
        z_x_shape = jax.ShapeDtypeStruct((total_xn,), jnp.float64)
        # Metadata: all scalars for single problem
        status_shape = scalar_shape
        obj_val_shape = scalar_shape
        iterations_shape = scalar_shape
        solve_time_shape = scalar_shape
        setup_time_shape = scalar_shape
        construction_time_shape = scalar_shape
    else:
        batch_size = q.shape[0]
        x_shape = jax.ShapeDtypeStruct((batch_size, n), jnp.float64)
        z_shape = jax.ShapeDtypeStruct((batch_size, m), jnp.float64)
        s_shape = jax.ShapeDtypeStruct((batch_size, m), jnp.float64)
        z_x_shape = jax.ShapeDtypeStruct((batch_size, total_xn), jnp.float64)
        # Metadata: per-problem arrays for batch (timing is broadcast to batch size for vmap compat)
        batch_scalar_shape = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        status_shape = batch_scalar_shape
        obj_val_shape = batch_scalar_shape
        iterations_shape = batch_scalar_shape
        solve_time_shape = batch_scalar_shape
        setup_time_shape = batch_scalar_shape
        construction_time_shape = batch_scalar_shape

    x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time = (
        jax.pure_callback(
            partial(_solve_cpu_callback, solver_id),
            (
                x_shape,
                z_shape,
                s_shape,
                z_x_shape,
                status_shape,
                obj_val_shape,
                iterations_shape,
                solve_time_shape,
                setup_time_shape,
                construction_time_shape,
            ),
            P_data,
            A_data,
            q,
            b,
            vmap_method="broadcast_all",
        )
    )

    return (x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time)


def _solve_cpu_fwd(
    solver_id: int,
    P_data: jnp.ndarray,
    A_data: jnp.ndarray,
    q: jnp.ndarray,
    b: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, ...], Any]:
    """Forward pass with saved residuals for backward."""
    x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time = (
        _solve_cpu_raw(solver_id, P_data, A_data, q, b)
    )

    # Save everything needed for backward (including z_x for direct-x dz_x).
    residuals = (P_data, A_data, q, b, x, z, s, z_x)
    return (
        x,
        z,
        s,
        z_x,
        status,
        obj_val,
        iterations,
        solve_time,
        setup_time,
        construction_time,
    ), residuals


def _solve_cpu_bwd(solver_id: int, residuals, g):
    """Backward pass via implicit differentiation."""
    P_data, A_data, q, b, x, z, s, z_x = residuals
    # g contains gradients for all 10 outputs.
    # x, z, s, z_x carry meaningful gradients into the implicit-diff KKT.
    (
        dx,
        dz,
        ds,
        dz_x,
        _dstatus,
        _dobj_val,
        _diterations,
        _dsolve_time,
        _dsetup_time,
        _dconstruction_time,
    ) = g

    solver_wrapper = _SOLVER_REGISTRY[solver_id]
    n, m = solver_wrapper._n, solver_wrapper._m
    nnzP, nnzA = solver_wrapper._nnzP, solver_wrapper._nnzA

    # Determine output shapes
    if dx.ndim == 1:
        dP_shape = jax.ShapeDtypeStruct((nnzP,), jnp.float64)
        dA_shape = jax.ShapeDtypeStruct((nnzA,), jnp.float64)
        dq_shape = jax.ShapeDtypeStruct((n,), jnp.float64)
        db_shape = jax.ShapeDtypeStruct((m,), jnp.float64)
    else:
        batch_size = dx.shape[0]
        dP_shape = jax.ShapeDtypeStruct((batch_size, nnzP), jnp.float64)
        dA_shape = jax.ShapeDtypeStruct((batch_size, nnzA), jnp.float64)
        dq_shape = jax.ShapeDtypeStruct((batch_size, n), jnp.float64)
        db_shape = jax.ShapeDtypeStruct((batch_size, m), jnp.float64)

    dP, dA, dq, db = jax.pure_callback(
        partial(_backward_cpu_callback, solver_id),
        (dP_shape, dA_shape, dq_shape, db_shape),
        dx,
        dz,
        ds,
        dz_x,
        P_data,
        A_data,
        q,
        b,
        x,
        z,
        s,
        z_x,
        vmap_method="broadcast_all",
    )

    # Return gradients in same order as forward inputs (excluding solver_id)
    return dP, dA, dq, db


_solve_cpu_raw.defvjp(_solve_cpu_fwd, _solve_cpu_bwd)


def _solve_cpu(
    solver_id: int,
    P_data: jnp.ndarray,
    A_data: jnp.ndarray,
    q: jnp.ndarray,
    b: jnp.ndarray,
) -> Tuple[JaxSolution, JaxSolveInfo]:
    """Solve conic QP on CPU. Returns (JaxSolution, JaxSolveInfo) tuple."""
    x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time = (
        _solve_cpu_raw(solver_id, P_data, A_data, q, b)
    )

    # Bundle into named tuples
    solution = JaxSolution(x=x, z=z, s=s, z_x=z_x)

    # Convert to appropriate types for JaxSolveInfo
    info = JaxSolveInfo(
        status=jnp.asarray(status, dtype=jnp.int32),
        obj_val=obj_val,
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        solve_time=solve_time,
        setup_time=setup_time,
        construction_time=construction_time,
    )

    return solution, info


__all__ = ["JaxSolverCpu"]
