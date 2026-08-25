"""Forward and warm-start FFI lowering for the CUDA JAX solver.

These factory functions build the pure solve functions returned by
``JaxSolverCuda.solve`` / ``.solve_warm`` when XLA FFI is available.
"""

from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp
from jax import custom_vjp
from jax.custom_batching import custom_vmap

from moreau._types import JaxSolution, JaxSolveInfo

# =============================================================================
# FFI-based solve with custom vmap (zero-copy, handles batching correctly)
# =============================================================================


def _make_ffi_solve_fn(
    n: int,
    m: int,
    num_zero: int,
    num_nonneg: int,
    num_exp: int,
    num_soc: int,
    num_psd: int,
    num_power: int,
    num_gen_power: int,
    num_x_cones: int,
    total_xn: int,
    P_row_offsets_gpu: jnp.ndarray,
    P_col_indices_gpu: jnp.ndarray,
    A_row_offsets_gpu: jnp.ndarray,
    A_col_indices_gpu: jnp.ndarray,
    soc_dims_gpu: jnp.ndarray,
    psd_dims_gpu: jnp.ndarray,
    power_alphas_gpu: jnp.ndarray,
    gen_power_alphas_gpu: jnp.ndarray,
    gen_power_dim1s_gpu: jnp.ndarray,
    gen_power_dim2s_gpu: jnp.ndarray,
    x_kinds_gpu: jnp.ndarray,
    x_indices_offsets_gpu: jnp.ndarray,
    x_indices_flat_gpu: jnp.ndarray,
    x_power_alphas_gpu: jnp.ndarray,
    x_psd_ks_gpu: jnp.ndarray,
    x_gen_power_dim1s_gpu: jnp.ndarray,
    x_gen_power_dim2s_gpu: jnp.ndarray,
    x_gen_power_alphas_gpu: jnp.ndarray,
    wrapper_construction_time: float,
):
    """Factory function that creates FFI solve with static values captured in closures.

    This avoids the tracer issue with custom_vmap by capturing n, m, etc. as Python
    integers in closures rather than passing them as function arguments.

    The returned function:
    - Takes 1D unbatched inputs and returns 1D unbatched outputs
    - Works with jax.vmap for batching (via custom_vmap rule)
    - Works with jax.grad for gradients (via custom_vjp)
    - Composable: jax.grad(jax.vmap(...)) and jax.vmap(jax.grad(...)) both work
    """
    # Capture nnzP and nnzA as concrete values
    nnzP = P_col_indices_gpu.shape[0]
    nnzA = A_col_indices_gpu.shape[0]

    def _ffi_call_impl_batched(
        P_data: jnp.ndarray,
        A_data: jnp.ndarray,
        q: jnp.ndarray,
        b: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, ...]:
        """Raw FFI call - assumes data arrays are already properly shaped (batch, dim).

        Returns 10 arrays: x, z, s, z_x, status, obj_val, iterations,
                         solve_time, setup_time, construction_time
        """
        # Determine batch size from q shape (must be 2D: batch x n)
        batch_size = q.shape[0]

        # Define output types - n, m are captured from outer scope as concrete ints
        x_type = jax.ShapeDtypeStruct((batch_size, n), jnp.float64)
        z_type = jax.ShapeDtypeStruct((batch_size, m), jnp.float64)
        s_type = jax.ShapeDtypeStruct((batch_size, m), jnp.float64)
        z_x_type = jax.ShapeDtypeStruct((batch_size, total_xn), jnp.float64)
        # Metadata types (scalars per batch)
        status_type = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        obj_val_type = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        iterations_type = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        solve_time_type = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        setup_time_type = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        construction_time_type = jax.ShapeDtypeStruct((batch_size,), jnp.float64)

        # Call FFI - structure arrays are 1D, data arrays are 2D (batch, dim)
        ffi_fn = jax.ffi.ffi_call(
            "moreau_solve_fwd",
            (
                x_type,
                z_type,
                s_type,
                z_x_type,
                status_type,
                obj_val_type,
                iterations_type,
                solve_time_type,
                setup_time_type,
                construction_time_type,
            ),
        )
        x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time = (
            ffi_fn(
                P_row_offsets_gpu,
                P_col_indices_gpu,
                A_row_offsets_gpu,
                A_col_indices_gpu,
                soc_dims_gpu,
                psd_dims_gpu,
                power_alphas_gpu,
                gen_power_alphas_gpu,
                gen_power_dim1s_gpu,
                gen_power_dim2s_gpu,
                x_kinds_gpu,
                x_indices_offsets_gpu,
                x_indices_flat_gpu,
                x_power_alphas_gpu,
                x_psd_ks_gpu,
                x_gen_power_dim1s_gpu,
                x_gen_power_dim2s_gpu,
                x_gen_power_alphas_gpu,
                P_data,
                A_data,
                q,
                b,
                n=np.int64(n),
                m=np.int64(m),
                num_zero=np.int64(num_zero),
                num_nonneg=np.int64(num_nonneg),
                num_exp=np.int64(num_exp),
                num_soc=np.int64(num_soc),
                num_psd=np.int64(num_psd),
                num_power=np.int64(num_power),
                num_gen_power=np.int64(num_gen_power),
                num_x_cones=np.int64(num_x_cones),
            )
        )

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
        )

    def _ffi_call_bwd_raw(
        P_data: jnp.ndarray,
        A_data: jnp.ndarray,
        q: jnp.ndarray,
        b: jnp.ndarray,
        dx: jnp.ndarray,
        dz: jnp.ndarray,
        ds: jnp.ndarray,
        dz_x: jnp.ndarray,
        x: jnp.ndarray,
        z: jnp.ndarray,
        s: jnp.ndarray,
        z_x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Raw FFI backward call - assumes data arrays are 2D (batch, dim).

        Stateless: P_data, A_data, q, b are passed explicitly (saved from
        forward) so the backward uses the correct problem data even when
        a later solve has overwritten the cached solver state.

        Returns 4 arrays: dP, dA, dq, db
        """
        batch_size = dx.shape[0]

        # Define output types for gradients
        dP_type = jax.ShapeDtypeStruct((batch_size, nnzP), jnp.float64)
        dA_type = jax.ShapeDtypeStruct((batch_size, nnzA), jnp.float64)
        dq_type = jax.ShapeDtypeStruct((batch_size, n), jnp.float64)
        db_type = jax.ShapeDtypeStruct((batch_size, m), jnp.float64)

        # Call backward FFI - structure arrays are captured from closure (not vmapped)
        ffi_fn = jax.ffi.ffi_call(
            "moreau_solve_bwd",
            (dP_type, dA_type, dq_type, db_type),
        )
        dP, dA, dq_out, db_out = ffi_fn(
            P_row_offsets_gpu,
            P_col_indices_gpu,
            A_row_offsets_gpu,
            A_col_indices_gpu,
            soc_dims_gpu,
            psd_dims_gpu,
            power_alphas_gpu,
            gen_power_alphas_gpu,
            gen_power_dim1s_gpu,
            gen_power_dim2s_gpu,
            x_kinds_gpu,
            x_indices_offsets_gpu,
            x_indices_flat_gpu,
            x_power_alphas_gpu,
            x_psd_ks_gpu,
            x_gen_power_dim1s_gpu,
            x_gen_power_dim2s_gpu,
            x_gen_power_alphas_gpu,
            P_data,
            A_data,
            q,
            b,
            dx,
            dz,
            ds,
            dz_x,
            x,
            z,
            s,
            z_x,
            n=np.int64(n),
            m=np.int64(m),
            num_zero=np.int64(num_zero),
            num_nonneg=np.int64(num_nonneg),
            num_exp=np.int64(num_exp),
            num_soc=np.int64(num_soc),
            num_psd=np.int64(num_psd),
            num_power=np.int64(num_power),
            num_gen_power=np.int64(num_gen_power),
            num_x_cones=np.int64(num_x_cones),
        )

        return dP, dA, dq_out, db_out

    # Backward with custom_vmap to handle batching correctly
    # This is needed for grad(vmap(...)) case
    @custom_vmap
    def _ffi_call_bwd_impl_batched(
        P_data: jnp.ndarray,
        A_data: jnp.ndarray,
        q: jnp.ndarray,
        b: jnp.ndarray,
        dx: jnp.ndarray,
        dz: jnp.ndarray,
        ds: jnp.ndarray,
        dz_x: jnp.ndarray,
        x: jnp.ndarray,
        z: jnp.ndarray,
        s: jnp.ndarray,
        z_x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Backward FFI call with vmap support (unbatched version - adds batch dim)."""
        # Add batch dimension for FFI
        P_2d = P_data.reshape(1, -1)
        A_2d = A_data.reshape(1, -1)
        q_2d = q.reshape(1, -1)
        b_2d = b.reshape(1, -1)
        dx_2d = dx.reshape(1, -1)
        dz_2d = dz.reshape(1, -1)
        ds_2d = ds.reshape(1, -1)
        dz_x_2d = dz_x.reshape(1, -1)
        x_2d = x.reshape(1, -1)
        z_2d = z.reshape(1, -1)
        s_2d = s.reshape(1, -1)
        z_x_2d = z_x.reshape(1, -1)

        dP, dA, dq_out, db_out = _ffi_call_bwd_raw(
            P_2d, A_2d, q_2d, b_2d, dx_2d, dz_2d, ds_2d, dz_x_2d, x_2d, z_2d, s_2d, z_x_2d
        )

        # Remove batch dimension
        return dP.squeeze(0), dA.squeeze(0), dq_out.squeeze(0), db_out.squeeze(0)

    @_ffi_call_bwd_impl_batched.def_vmap
    def _bwd_vmap_rule(axis_size, in_batched, P_data, A_data, q, b, dx, dz, ds, dz_x, x, z, s, z_x):
        """Custom vmap rule for backward - data arrays are batched, structure arrays are not."""

        def ensure_batched(arr, is_batched):
            if is_batched:
                return arr
            else:
                return jnp.broadcast_to(arr, (axis_size,) + arr.shape)

        P_batched = ensure_batched(P_data, in_batched[0])
        A_batched = ensure_batched(A_data, in_batched[1])
        q_batched = ensure_batched(q, in_batched[2])
        b_batched = ensure_batched(b, in_batched[3])
        dx_batched = ensure_batched(dx, in_batched[4])
        dz_batched = ensure_batched(dz, in_batched[5])
        ds_batched = ensure_batched(ds, in_batched[6])
        dz_x_batched = ensure_batched(dz_x, in_batched[7])
        x_batched = ensure_batched(x, in_batched[8])
        z_batched = ensure_batched(z, in_batched[9])
        s_batched = ensure_batched(s, in_batched[10])
        z_x_batched = ensure_batched(z_x, in_batched[11])

        # Call raw FFI directly with batched data
        dP, dA, dq_out, db_out = _ffi_call_bwd_raw(
            P_batched,
            A_batched,
            q_batched,
            b_batched,
            dx_batched,
            dz_batched,
            ds_batched,
            dz_x_batched,
            x_batched,
            z_batched,
            s_batched,
            z_x_batched,
        )

        # All outputs are batched
        out_batched = (True, True, True, True)
        return (dP, dA, dq_out, db_out), out_batched

    # Batched solve (2D inputs/outputs) - raw FFI call
    def _solve_batched_raw(P_data, A_data, q, b):
        """Solve conic QP - batched version (2D inputs/outputs), raw FFI.

        Returns 9 arrays: x, z, s, status, obj_val,
                         iterations, solve_time, setup_time, construction_time
        """
        return _ffi_call_impl_batched(P_data, A_data, q, b)

    # Unbatched solve with custom_vjp for gradients
    # This takes 1D inputs and returns 1D outputs (10 outputs now)
    @custom_vjp
    def _solve_unbatched_with_grad(
        P_data: jnp.ndarray,
        A_data: jnp.ndarray,
        q: jnp.ndarray,
        b: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, ...]:
        """Solve conic QP - unbatched version with gradient support.

        Returns 10 arrays: x, z, s, z_x, status, obj_val,
                         iterations, solve_time, setup_time, construction_time
        """
        # Add batch dimension for FFI
        P_data_2d = P_data.reshape(1, -1)
        A_data_2d = A_data.reshape(1, -1)
        q_2d = q.reshape(1, -1)
        b_2d = b.reshape(1, -1)

        x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time = (
            _solve_batched_raw(P_data_2d, A_data_2d, q_2d, b_2d)
        )

        # Remove batch dimension (scalars become 0-d arrays)
        return (
            x.squeeze(0),
            z.squeeze(0),
            s.squeeze(0),
            z_x.squeeze(0),
            status.squeeze(0),
            obj_val.squeeze(0),
            iterations.squeeze(0),
            solve_time.squeeze(0),
            setup_time.squeeze(0),
            construction_time.squeeze(0),
        )

    def _solve_unbatched_fwd(P_data, A_data, q, b):
        """Forward pass with saved residuals for backward."""
        x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time = (
            _solve_unbatched_with_grad(P_data, A_data, q, b)
        )
        # Save x, z, s, z_x for backward (metadata not needed). Also save
        # P/A/q/b so the stateless backward can re-equilibrate.
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

    def _solve_unbatched_bwd(residuals, g):
        """Backward pass via XLA FFI (unbatched).

        Receives 10 gradients; uses the four solution-vector grads
        (dx, dz, ds, dz_x). Metadata grads are ignored.
        """
        P_data, A_data, q, b, x, z, s, z_x = residuals
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

        # Add batch dimension for backward FFI
        P_2d = P_data.reshape(1, -1)
        A_2d = A_data.reshape(1, -1)
        q_2d = q.reshape(1, -1)
        b_2d = b.reshape(1, -1)
        dx_2d = dx.reshape(1, -1)
        dz_2d = dz.reshape(1, -1)
        ds_2d = ds.reshape(1, -1)
        dz_x_2d = dz_x.reshape(1, -1)
        x_2d = x.reshape(1, -1)
        z_2d = z.reshape(1, -1)
        s_2d = s.reshape(1, -1)
        z_x_2d = z_x.reshape(1, -1)

        dP, dA, dq, db = _ffi_call_bwd_raw(
            P_2d, A_2d, q_2d, b_2d, dx_2d, dz_2d, ds_2d, dz_x_2d, x_2d, z_2d, s_2d, z_x_2d
        )

        # Remove batch dimension
        return dP.squeeze(0), dA.squeeze(0), dq.squeeze(0), db.squeeze(0)

    _solve_unbatched_with_grad.defvjp(_solve_unbatched_fwd, _solve_unbatched_bwd)

    # Main solve function - combines custom_vjp (outer) with custom_vmap (inner)
    # The pattern is: custom_vjp wraps custom_vmap
    # This ensures:
    # 1. Unbatched grad: custom_vjp handles it directly
    # 2. Unbatched forward: goes through custom_vmap base case
    # 3. Batched forward (vmap): custom_vmap rule handles it
    # 4. Batched grad (grad of vmap): custom_vjp handles it, uses batched FFI

    # Inner function with custom_vmap only (no vjp - that's handled by outer)
    @custom_vmap
    def _solve_vmap_only(
        P_data: jnp.ndarray,
        A_data: jnp.ndarray,
        q: jnp.ndarray,
        b: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, ...]:
        """Solve conic QP - unbatched forward pass (1D inputs/outputs).

        Returns 10 arrays: x, z, s, z_x, status, obj_val,
                         iterations, solve_time, setup_time, construction_time
        """
        # Add batch dimension for FFI
        P_data_2d = P_data.reshape(1, -1)
        A_data_2d = A_data.reshape(1, -1)
        q_2d = q.reshape(1, -1)
        b_2d = b.reshape(1, -1)

        x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time = (
            _solve_batched_raw(P_data_2d, A_data_2d, q_2d, b_2d)
        )

        # Remove batch dimension (scalars become 0-d arrays)
        return (
            x.squeeze(0),
            z.squeeze(0),
            s.squeeze(0),
            z_x.squeeze(0),
            status.squeeze(0),
            obj_val.squeeze(0),
            iterations.squeeze(0),
            solve_time.squeeze(0),
            setup_time.squeeze(0),
            construction_time.squeeze(0),
        )

    @_solve_vmap_only.def_vmap
    def _solve_vmap_rule(axis_size, in_batched, P_data, A_data, q, b):
        """Custom vmap rule: batch all data arrays together and call batched solver."""

        def ensure_batched(arr, is_batched):
            if is_batched:
                return arr
            else:
                return jnp.broadcast_to(arr, (axis_size,) + arr.shape)

        P_data_batched = ensure_batched(P_data, in_batched[0])
        A_data_batched = ensure_batched(A_data, in_batched[1])
        q_batched = ensure_batched(q, in_batched[2])
        b_batched = ensure_batched(b, in_batched[3])

        # Call batched FFI directly
        x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time = (
            _solve_batched_raw(P_data_batched, A_data_batched, q_batched, b_batched)
        )

        # All 10 outputs are batched
        out_batched = (True, True, True, True, True, True, True, True, True, True)
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
        ), out_batched

    # Outer function with custom_vjp for gradients
    @custom_vjp
    def _solve_with_grad(
        P_data: jnp.ndarray,
        A_data: jnp.ndarray,
        q: jnp.ndarray,
        b: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, ...]:
        """Solve conic QP with gradient support (unbatched inputs).

        Returns 10 arrays: x, z, s, z_x, status, obj_val,
                         iterations, solve_time, setup_time, construction_time
        """
        return _solve_vmap_only(P_data, A_data, q, b)

    def _solve_fwd(P_data, A_data, q, b):
        """Forward pass with saved residuals."""
        x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time = (
            _solve_with_grad(P_data, A_data, q, b)
        )
        # Save all problem data + solution (incl. z_x) for stateless
        # backward. This ensures chained solves (MPC rollouts) work
        # correctly — each backward uses its own saved data, not stale
        # solver state.
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

    def _solve_bwd(residuals, g):
        """Backward pass — stateless GPU backward via FFI.

        P_data, A_data, q, b are saved from the forward pass and passed
        to the FFI backward handler, which reloads and re-equilibrates
        the solver before computing gradients.  This ensures correctness
        for chained solves where a later forward pass overwrites state.
        """
        P_data, A_data, q, b, x, z, s, z_x = residuals
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

        # Check if we're in batched mode (2D arrays) or unbatched (1D arrays)
        if dx.ndim == 1:
            # Unbatched case - call the custom_vmap-enabled backward directly
            return _ffi_call_bwd_impl_batched(P_data, A_data, q, b, dx, dz, ds, dz_x, x, z, s, z_x)
        else:
            # Batched case - use vmap over the custom_vmap-enabled backward
            return jax.vmap(_ffi_call_bwd_impl_batched)(
                P_data, A_data, q, b, dx, dz, ds, dz_x, x, z, s, z_x
            )

    _solve_with_grad.defvjp(_solve_fwd, _solve_bwd)

    # Final wrapper to return (JaxSolution, JaxSolveInfo) tuple
    def _solve_with_solution(
        P_data: jnp.ndarray,
        A_data: jnp.ndarray,
        q: jnp.ndarray,
        b: jnp.ndarray,
    ) -> Tuple[JaxSolution, JaxSolveInfo]:
        """Solve conic QP and return (JaxSolution, JaxSolveInfo) tuple.

        This is the main entry point. Returns a tuple of NamedTuples
        which are pytree-compatible and work with jax.vmap/jax.grad.
        """
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
            _ffi_construction_time,
        ) = _solve_with_grad(P_data, A_data, q, b)
        solution = JaxSolution(x=x, z=z, s=s, z_x=z_x)
        # Use wrapper construction time instead of FFI-returned value (which is 0.0)
        info = JaxSolveInfo(
            status=status,
            obj_val=obj_val,
            iterations=iterations,
            solve_time=solve_time,
            setup_time=setup_time,
            construction_time=jnp.asarray(wrapper_construction_time, dtype=jnp.float64),
        )
        return solution, info

    return _solve_with_solution


# =============================================================================
# FFI-based warm-start solve
# =============================================================================


def _make_ffi_solve_warm_fn(
    n: int,
    m: int,
    num_zero: int,
    num_nonneg: int,
    num_exp: int,
    num_soc: int,
    num_psd: int,
    num_power: int,
    num_gen_power: int,
    num_x_cones: int,
    total_xn: int,
    P_row_offsets_gpu: jnp.ndarray,
    P_col_indices_gpu: jnp.ndarray,
    A_row_offsets_gpu: jnp.ndarray,
    A_col_indices_gpu: jnp.ndarray,
    soc_dims_gpu: jnp.ndarray,
    psd_dims_gpu: jnp.ndarray,
    power_alphas_gpu: jnp.ndarray,
    gen_power_alphas_gpu: jnp.ndarray,
    gen_power_dim1s_gpu: jnp.ndarray,
    gen_power_dim2s_gpu: jnp.ndarray,
    x_kinds_gpu: jnp.ndarray,
    x_indices_offsets_gpu: jnp.ndarray,
    x_indices_flat_gpu: jnp.ndarray,
    x_power_alphas_gpu: jnp.ndarray,
    x_psd_ks_gpu: jnp.ndarray,
    x_gen_power_dim1s_gpu: jnp.ndarray,
    x_gen_power_dim2s_gpu: jnp.ndarray,
    x_gen_power_alphas_gpu: jnp.ndarray,
    wrapper_construction_time: float,
):
    """Factory function for warm-start FFI solve.

    Returns a function:
        (P_data, A_data, q, b, warm_x, warm_z, warm_s, warm_z_x)
            -> (JaxSolution, JaxSolveInfo)

    Mirrors `_make_ffi_solve_fn` (the regular forward) — same x-cone GPU
    buffers and `num_x_cones`/`total_xn` plumbing — plus the four warm
    arrays (`warm_x`, `warm_z`, `warm_s`, `warm_z_x`). The C++ FFI handler
    `MoreauSolveFwdWarmImpl` expects the warm-`z_x` arg as a placeholder
    when `total_xn == 0`; the JaxSolver caller passes a zero-length
    buffer in that case.

    Backward gradients are zeros for the warm arrays (warm start is
    treated as a constant input).
    """
    nnzP = P_col_indices_gpu.shape[0]
    nnzA = A_col_indices_gpu.shape[0]

    def _ffi_call_warm_batched(
        P_data: jnp.ndarray,
        A_data: jnp.ndarray,
        q: jnp.ndarray,
        b: jnp.ndarray,
        warm_x: jnp.ndarray,
        warm_z: jnp.ndarray,
        warm_s: jnp.ndarray,
        warm_z_x: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, ...]:
        """Raw warm FFI call - assumes all arrays are 2D (batch, dim)."""
        batch_size = q.shape[0]

        x_type = jax.ShapeDtypeStruct((batch_size, n), jnp.float64)
        z_type = jax.ShapeDtypeStruct((batch_size, m), jnp.float64)
        s_type = jax.ShapeDtypeStruct((batch_size, m), jnp.float64)
        z_x_type = jax.ShapeDtypeStruct((batch_size, total_xn), jnp.float64)
        status_type = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        obj_val_type = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        iterations_type = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        solve_time_type = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        setup_time_type = jax.ShapeDtypeStruct((batch_size,), jnp.float64)
        construction_time_type = jax.ShapeDtypeStruct((batch_size,), jnp.float64)

        ffi_fn = jax.ffi.ffi_call(
            "moreau_solve_fwd_warm",
            (
                x_type,
                z_type,
                s_type,
                z_x_type,
                status_type,
                obj_val_type,
                iterations_type,
                solve_time_type,
                setup_time_type,
                construction_time_type,
            ),
        )
        return ffi_fn(
            P_row_offsets_gpu,
            P_col_indices_gpu,
            A_row_offsets_gpu,
            A_col_indices_gpu,
            soc_dims_gpu,
            psd_dims_gpu,
            power_alphas_gpu,
            gen_power_alphas_gpu,
            gen_power_dim1s_gpu,
            gen_power_dim2s_gpu,
            x_kinds_gpu,
            x_indices_offsets_gpu,
            x_indices_flat_gpu,
            x_power_alphas_gpu,
            x_psd_ks_gpu,
            x_gen_power_dim1s_gpu,
            x_gen_power_dim2s_gpu,
            x_gen_power_alphas_gpu,
            P_data,
            A_data,
            q,
            b,
            warm_x,
            warm_z,
            warm_s,
            warm_z_x,
            n=np.int64(n),
            m=np.int64(m),
            num_zero=np.int64(num_zero),
            num_nonneg=np.int64(num_nonneg),
            num_exp=np.int64(num_exp),
            num_soc=np.int64(num_soc),
            num_psd=np.int64(num_psd),
            num_power=np.int64(num_power),
            num_gen_power=np.int64(num_gen_power),
            num_x_cones=np.int64(num_x_cones),
        )

    # Backward FFI call for warm-start path (same FFI endpoint as regular backward)
    def _warm_bwd_raw(P_data, A_data, q, b, dx, dz, ds, dz_x, x, z, s, z_x):
        """Raw FFI backward call for warm-start — 2D (batch, dim) arrays."""
        batch_size = dx.shape[0]
        dP_type = jax.ShapeDtypeStruct((batch_size, nnzP), jnp.float64)
        dA_type = jax.ShapeDtypeStruct((batch_size, nnzA), jnp.float64)
        dq_type = jax.ShapeDtypeStruct((batch_size, n), jnp.float64)
        db_type = jax.ShapeDtypeStruct((batch_size, m), jnp.float64)

        ffi_fn = jax.ffi.ffi_call(
            "moreau_solve_bwd",
            (dP_type, dA_type, dq_type, db_type),
        )
        return ffi_fn(
            P_row_offsets_gpu,
            P_col_indices_gpu,
            A_row_offsets_gpu,
            A_col_indices_gpu,
            soc_dims_gpu,
            psd_dims_gpu,
            power_alphas_gpu,
            gen_power_alphas_gpu,
            gen_power_dim1s_gpu,
            gen_power_dim2s_gpu,
            x_kinds_gpu,
            x_indices_offsets_gpu,
            x_indices_flat_gpu,
            x_power_alphas_gpu,
            x_psd_ks_gpu,
            x_gen_power_dim1s_gpu,
            x_gen_power_dim2s_gpu,
            x_gen_power_alphas_gpu,
            P_data,
            A_data,
            q,
            b,
            dx,
            dz,
            ds,
            dz_x,
            x,
            z,
            s,
            z_x,
            n=np.int64(n),
            m=np.int64(m),
            num_zero=np.int64(num_zero),
            num_nonneg=np.int64(num_nonneg),
            num_exp=np.int64(num_exp),
            num_soc=np.int64(num_soc),
            num_psd=np.int64(num_psd),
            num_power=np.int64(num_power),
            num_gen_power=np.int64(num_gen_power),
            num_x_cones=np.int64(num_x_cones),
        )

    @custom_vmap
    def _warm_bwd_impl(P_data, A_data, q, b, dx, dz, ds, dz_x, x, z, s, z_x):
        """Backward with vmap support (unbatched version)."""
        result = _warm_bwd_raw(
            P_data.reshape(1, -1),
            A_data.reshape(1, -1),
            q.reshape(1, -1),
            b.reshape(1, -1),
            dx.reshape(1, -1),
            dz.reshape(1, -1),
            ds.reshape(1, -1),
            dz_x.reshape(1, -1),
            x.reshape(1, -1),
            z.reshape(1, -1),
            s.reshape(1, -1),
            z_x.reshape(1, -1),
        )
        return tuple(r.squeeze(0) for r in result)

    @_warm_bwd_impl.def_vmap
    def _warm_bwd_vmap_rule(
        axis_size, in_batched, P_data, A_data, q, b, dx, dz, ds, dz_x, x, z, s, z_x
    ):
        def ensure_batched(arr, is_batched):
            if is_batched:
                return arr
            return jnp.broadcast_to(arr, (axis_size,) + arr.shape)

        result = _warm_bwd_raw(
            ensure_batched(P_data, in_batched[0]),
            ensure_batched(A_data, in_batched[1]),
            ensure_batched(q, in_batched[2]),
            ensure_batched(b, in_batched[3]),
            ensure_batched(dx, in_batched[4]),
            ensure_batched(dz, in_batched[5]),
            ensure_batched(ds, in_batched[6]),
            ensure_batched(dz_x, in_batched[7]),
            ensure_batched(x, in_batched[8]),
            ensure_batched(z, in_batched[9]),
            ensure_batched(s, in_batched[10]),
            ensure_batched(z_x, in_batched[11]),
        )
        out_batched = (True, True, True, True)
        return result, out_batched

    # Inner function with custom_vmap
    @custom_vmap
    def _solve_warm_vmap_only(P_data, A_data, q, b, warm_x, warm_z, warm_s, warm_z_x):
        """Warm solve - unbatched forward (1D inputs/outputs)."""
        result = _ffi_call_warm_batched(
            P_data.reshape(1, -1),
            A_data.reshape(1, -1),
            q.reshape(1, -1),
            b.reshape(1, -1),
            warm_x.reshape(1, -1),
            warm_z.reshape(1, -1),
            warm_s.reshape(1, -1),
            warm_z_x.reshape(1, -1),
        )
        return tuple(r.squeeze(0) for r in result)

    @_solve_warm_vmap_only.def_vmap
    def _warm_vmap_rule(
        axis_size, in_batched, P_data, A_data, q, b, warm_x, warm_z, warm_s, warm_z_x
    ):
        """Custom vmap rule: batch all data arrays together."""

        def ensure_batched(arr, is_batched):
            if is_batched:
                return arr
            return jnp.broadcast_to(arr, (axis_size,) + arr.shape)

        result = _ffi_call_warm_batched(
            ensure_batched(P_data, in_batched[0]),
            ensure_batched(A_data, in_batched[1]),
            ensure_batched(q, in_batched[2]),
            ensure_batched(b, in_batched[3]),
            ensure_batched(warm_x, in_batched[4]),
            ensure_batched(warm_z, in_batched[5]),
            ensure_batched(warm_s, in_batched[6]),
            ensure_batched(warm_z_x, in_batched[7]),
        )
        out_batched = (True,) * 10
        return result, out_batched

    # Outer function with custom_vjp for gradients
    @custom_vjp
    def _solve_warm_with_grad(P_data, A_data, q, b, warm_x, warm_z, warm_s, warm_z_x):
        """Warm solve with gradient support (unbatched inputs)."""
        return _solve_warm_vmap_only(P_data, A_data, q, b, warm_x, warm_z, warm_s, warm_z_x)

    def _solve_warm_fwd(P_data, A_data, q, b, warm_x, warm_z, warm_s, warm_z_x):
        """Forward pass with saved residuals."""
        x, z, s, z_x, status, obj_val, iterations, solve_time, setup_time, construction_time = (
            _solve_warm_with_grad(P_data, A_data, q, b, warm_x, warm_z, warm_s, warm_z_x)
        )
        # Save all problem data + solution for stateless backward
        residuals = (P_data, A_data, q, b, x, z, s, z_x, warm_x, warm_z, warm_s, warm_z_x)
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

    def _solve_warm_bwd(residuals, g):
        """Backward pass — stateless GPU backward via FFI + zero grads for warm arrays."""
        P_data, A_data, q, b, x, z, s, z_x, warm_x, warm_z, warm_s, warm_z_x = residuals
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

        if dx.ndim == 1:
            dP, dA, dq, db = _warm_bwd_impl(P_data, A_data, q, b, dx, dz, ds, dz_x, x, z, s, z_x)
        else:
            dP, dA, dq, db = jax.vmap(_warm_bwd_impl)(
                P_data, A_data, q, b, dx, dz, ds, dz_x, x, z, s, z_x
            )

        return (
            dP,
            dA,
            dq,
            db,
            jnp.zeros_like(warm_x),
            jnp.zeros_like(warm_z),
            jnp.zeros_like(warm_s),
            jnp.zeros_like(warm_z_x),
        )

    _solve_warm_with_grad.defvjp(_solve_warm_fwd, _solve_warm_bwd)

    # Final wrapper returning (JaxSolution, JaxSolveInfo)
    def _solve_warm_with_solution(P_data, A_data, q, b, warm_x, warm_z, warm_s, warm_z_x):
        """Warm-start solve returning (JaxSolution, JaxSolveInfo)."""
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
            _ffi_construction_time,
        ) = _solve_warm_with_grad(P_data, A_data, q, b, warm_x, warm_z, warm_s, warm_z_x)
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

    return _solve_warm_with_solution
