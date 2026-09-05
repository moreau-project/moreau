"""pure_callback-based fallback solve path (when XLA FFI is unavailable)."""

from functools import partial
from typing import Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import custom_vjp

from moreau._types import JaxSolution, JaxSolveInfo

from ._solver import _SOLVER_REGISTRY

# =============================================================================
# Fallback: pure_callback-based solve (when FFI not available)
# =============================================================================


def _solve_fallback_callback(
    solver_id: int,
    P_data: np.ndarray,
    A_data: np.ndarray,
    q: np.ndarray,
    b: np.ndarray,
) -> Tuple[np.ndarray, ...]:
    """Callback function for pure_callback fallback.

    Returns 10 arrays: x, z, s, z_x, status, obj_val,
                      iterations, solve_time, setup_time, construction_time
    (z_x is zero-length for slack-only problems; carrying it as the 4th
    slot keeps the JaxSolution pytree consistent with the FFI path —
    `JaxSolution` is a 4-field NamedTuple `(x, z, s, z_x)` and constructing
    it with only 3 fields raises TypeError.)
    """
    solver_wrapper = _SOLVER_REGISTRY[solver_id]

    P_data = np.asarray(P_data, dtype=np.float64)
    A_data = np.asarray(A_data, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if q.ndim == 1:
        batch_size = 1
        P_data = P_data.reshape(1, -1)
        A_data = A_data.reshape(1, -1)
        q = q.reshape(1, -1)
        b = b.reshape(1, -1)
        squeeze = True
    else:
        batch_size = q.shape[0]
        squeeze = False

    P_data = np.ascontiguousarray(P_data)
    A_data = np.ascontiguousarray(A_data)
    q = np.ascontiguousarray(q)
    b = np.ascontiguousarray(b)

    cuda_solver = solver_wrapper._get_or_create_cuda_solver(batch_size, enable_grad=True)

    # Check for pending warm start. The underlying CUDA solver's
    # `solve_warm_start` signature is 7-arg (no direct warm); if a
    # `warm_z_x` was provided we pass it as a kwarg when the binding
    # accepts it, otherwise silently drop (matches the historical
    # behavior — direct warm in fallback is best-effort).
    if solver_wrapper._pending_warm_start is not None:
        pending_warm = solver_wrapper._pending_warm_start
        warm_x = np.ascontiguousarray(pending_warm["warm_x"].reshape(batch_size, -1))
        warm_z = np.ascontiguousarray(pending_warm["warm_z"].reshape(batch_size, -1))
        warm_s = np.ascontiguousarray(pending_warm["warm_s"].reshape(batch_size, -1))
        warm_z_x = pending_warm.get("warm_z_x")
        if warm_z_x is not None and warm_z_x.size > 0:
            try:
                warm_z_x_2d = np.ascontiguousarray(warm_z_x.reshape(batch_size, -1))
                result = cuda_solver.solve_warm_start(
                    P_data, A_data, q, b, warm_x, warm_z, warm_s, warm_z_x_2d
                )
            except TypeError:
                # Binding doesn't accept warm_z_x — fall through to slack-only warm.
                result = cuda_solver.solve_warm_start(P_data, A_data, q, b, warm_x, warm_z, warm_s)
        else:
            result = cuda_solver.solve_warm_start(P_data, A_data, q, b, warm_x, warm_z, warm_s)
    else:
        result = cuda_solver.solve(P_data, A_data, q, b)

    x = np.asarray(result["x"], dtype=np.float64)
    z = np.asarray(result["z"], dtype=np.float64)
    s = np.asarray(result["s"], dtype=np.float64)

    # Direct cone duals. The underlying CUDA solver returns z_x (shape
    # batch×total_xn) when dir_cones are present; otherwise we synthesise a
    # zero-length buffer with the right batch dimension. `JaxSolution`'s
    # 4-field shape requires z_x to be present even for slack-only solves.
    total_xn = solver_wrapper._total_xn
    z_x_raw = result.get("z_x")
    if z_x_raw is None:
        z_x = np.zeros((batch_size, total_xn), dtype=np.float64)
    else:
        z_x = np.asarray(z_x_raw, dtype=np.float64)
        if z_x.ndim == 1:
            z_x = z_x.reshape(batch_size, -1)

    # Extract metadata (broadcast to batch if needed)
    def _to_batch_array(val, default=0.0):
        arr = np.asarray(val if val is not None else default, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(batch_size, arr.item(), dtype=np.float64)
        return arr.reshape(batch_size)

    status = _to_batch_array(result.get("status", 0))
    obj_val = _to_batch_array(result.get("obj_val", 0.0))
    iterations = _to_batch_array(result.get("iterations", 0))
    solve_time = _to_batch_array(result.get("solve_time", 0.0))
    setup_time = _to_batch_array(result.get("setup_time", 0.0))
    construction_time = _to_batch_array(result.get("construction_time", 0.0))

    if squeeze:
        x = x.squeeze(0)
        z = z.squeeze(0)
        s = s.squeeze(0)
        z_x = z_x.squeeze(0)
        # Scalars become 0-d arrays
        status = status[0]
        obj_val = obj_val[0]
        iterations = iterations[0]
        solve_time = solve_time[0]
        setup_time = setup_time[0]
        construction_time = construction_time[0]

    return (x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time)


def _backward_fallback_callback(
    solver_id: int,
    dx: np.ndarray,
    dz: np.ndarray,
    ds: np.ndarray,
    P_data: np.ndarray,
    A_data: np.ndarray,
    q: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    z: np.ndarray,
    s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Callback function for backward pass fallback."""
    solver_wrapper = _SOLVER_REGISTRY[solver_id]

    dx = np.asarray(dx, dtype=np.float64)
    dz = np.asarray(dz, dtype=np.float64)
    ds = np.asarray(ds, dtype=np.float64)
    P_data = np.asarray(P_data, dtype=np.float64)
    A_data = np.asarray(A_data, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if dx.ndim == 1:
        batch_size = 1
        dx = dx.reshape(1, -1)
        dz = dz.reshape(1, -1)
        ds = ds.reshape(1, -1)
        P_data = P_data.reshape(1, -1)
        A_data = A_data.reshape(1, -1)
        q = q.reshape(1, -1)
        b = b.reshape(1, -1)
        squeeze = True
    else:
        batch_size = dx.shape[0]
        squeeze = False

    P_data = np.ascontiguousarray(P_data)
    A_data = np.ascontiguousarray(A_data)
    q = np.ascontiguousarray(q)
    b = np.ascontiguousarray(b)
    dx = np.ascontiguousarray(dx)
    dz = np.ascontiguousarray(dz)
    ds = np.ascontiguousarray(ds)

    cuda_solver = solver_wrapper._get_or_create_cuda_solver(batch_size, enable_grad=True)
    cuda_solver.solve(P_data, A_data, q, b)
    grad_result = cuda_solver.backward(dx, dz, ds)

    dP = np.asarray(grad_result["dP_values"], dtype=np.float64)
    dA = np.asarray(grad_result["dA_values"], dtype=np.float64)
    dq = np.asarray(grad_result["dq"], dtype=np.float64)
    db = np.asarray(grad_result["db"], dtype=np.float64)

    if squeeze:
        dP = dP.squeeze(0) if dP.ndim > 1 else dP
        dA = dA.squeeze(0) if dA.ndim > 1 else dA
        dq = dq.squeeze(0) if dq.ndim > 1 else dq
        db = db.squeeze(0) if db.ndim > 1 else db

    return dP, dA, dq, db


@partial(custom_vjp, nondiff_argnums=(0,))
def _solve_fallback(
    solver_id: int,
    P_data: jnp.ndarray,
    A_data: jnp.ndarray,
    q: jnp.ndarray,
    b: jnp.ndarray,
) -> Tuple[jnp.ndarray, ...]:
    """Solve conic QP using pure_callback fallback.

    Returns 10 arrays: x, z, s, z_x, status, obj_val,
                      iterations, solve_time, setup_time, construction_time
    """
    solver_wrapper = _SOLVER_REGISTRY[solver_id]
    n, m = solver_wrapper._n, solver_wrapper._m
    total_xn = solver_wrapper._total_xn

    if q.ndim == 1:
        x_shape = jax.ShapeDtypeStruct((n,), jnp.float64)
        z_shape = jax.ShapeDtypeStruct((m,), jnp.float64)
        s_shape = jax.ShapeDtypeStruct((m,), jnp.float64)
        z_x_shape = jax.ShapeDtypeStruct((total_xn,), jnp.float64)
        # Scalars for unbatched
        scalar_shape = jax.ShapeDtypeStruct((), jnp.float64)
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
        # Per-batch scalars
        status_shape = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        obj_val_shape = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        iterations_shape = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        solve_time_shape = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        setup_time_shape = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        construction_time_shape = jax.ShapeDtypeStruct((batch_size,), jnp.float64)

    x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time = (
        jax.pure_callback(
            partial(_solve_fallback_callback, solver_id),
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


def _solve_fallback_fwd(
    solver_id: int,
    P_data: jnp.ndarray,
    A_data: jnp.ndarray,
    q: jnp.ndarray,
    b: jnp.ndarray,
) -> Tuple[Tuple[jnp.ndarray, ...], tuple]:
    """Forward pass with saved residuals for backward."""
    x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time = (
        _solve_fallback(solver_id, P_data, A_data, q, b)
    )
    # Save inputs and x, z, s for backward (metadata not needed; z_x is the
    # direct dual but the fallback backward callback doesn't yet thread
    # it through, so omit from residuals — direct backward in fallback
    # mode is a known limitation.)
    residuals = (P_data, A_data, q, b, x, z, s)
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


def _solve_fallback_bwd(solver_id: int, residuals, g):
    """Backward pass via pure_callback.

    Receives 10 gradients but only uses first 3 (solution vectors).
    Metadata gradients (z_x grad, status, obj_val, iterations, times) are
    ignored — fallback mode does not yet support direct backward.
    """
    P_data, A_data, q, b, x, z, s = residuals
    # Unpack all 10 gradients - metadata + z_x grads will be ignored
    (
        dx,
        dz,
        ds,
        _dz_x,
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
        partial(_backward_fallback_callback, solver_id),
        (dP_shape, dA_shape, dq_shape, db_shape),
        dx,
        dz,
        ds,
        P_data,
        A_data,
        q,
        b,
        x,
        z,
        s,
        vmap_method="broadcast_all",
    )

    return dP, dA, dq, db


_solve_fallback.defvjp(_solve_fallback_fwd, _solve_fallback_bwd)


def _solve_fallback_with_solution(
    solver_id: int,
    wrapper_construction_time: float,
    P_data: jnp.ndarray,
    A_data: jnp.ndarray,
    q: jnp.ndarray,
    b: jnp.ndarray,
) -> Tuple[JaxSolution, JaxSolveInfo]:
    """Fallback solve that returns (JaxSolution, JaxSolveInfo) tuple."""
    (
        x,
        z,
        s,
        z_x,
        status,
        obj_val,
        iterations,
        solve_time,
        setup_time,
        _callback_construction_time,
    ) = _solve_fallback(solver_id, P_data, A_data, q, b)
    solution = JaxSolution(x=x, z=z, s=s, z_x=z_x)
    # Use wrapper construction time instead of callback-returned value (which is 0.0)
    info = JaxSolveInfo(
        status=status,
        obj_val=obj_val,
        iterations=iterations,
        solve_time=solve_time,
        setup_time=setup_time,
        construction_time=jnp.asarray(wrapper_construction_time, dtype=jnp.float64),
    )
    return solution, info


def _solve_fallback_warm_with_solution(
    solver_id: int,
    wrapper_construction_time: float,
    P_data: jnp.ndarray,
    A_data: jnp.ndarray,
    q: jnp.ndarray,
    b: jnp.ndarray,
    warm_x: jnp.ndarray,
    warm_z: jnp.ndarray,
    warm_s: jnp.ndarray,
    warm_z_x: Optional[jnp.ndarray] = None,
) -> Tuple[JaxSolution, JaxSolveInfo]:
    """Fallback warm-start solve via _pending_warm_start side channel.

    ``warm_z_x`` is accepted for FFI-path API parity but only stored when
    the underlying cuda_solver supports direct warm-start (otherwise it
    is silently dropped, since the fallback `solve_warm_start` binding
    only takes 7 arguments).
    """
    solver_wrapper = _SOLVER_REGISTRY[solver_id]
    pending = {
        "warm_x": np.asarray(warm_x, dtype=np.float64),
        "warm_z": np.asarray(warm_z, dtype=np.float64),
        "warm_s": np.asarray(warm_s, dtype=np.float64),
    }
    if warm_z_x is not None:
        pending["warm_z_x"] = np.asarray(warm_z_x, dtype=np.float64)
    solver_wrapper._pending_warm_start = pending
    try:
        (
            x,
            z,
            s,
            z_x,
            status,
            obj_val,
            iterations,
            solve_time,
            setup_time,
            _callback_construction_time,
        ) = _solve_fallback(solver_id, P_data, A_data, q, b)
    finally:
        solver_wrapper._pending_warm_start = None
    solution = JaxSolution(x=x, z=z, s=s, z_x=z_x)
    info = JaxSolveInfo(
        status=status,
        obj_val=obj_val,
        iterations=iterations,
        solve_time=solve_time,
        setup_time=setup_time,
        construction_time=jnp.asarray(wrapper_construction_time, dtype=jnp.float64),
    )
    return solution, info
