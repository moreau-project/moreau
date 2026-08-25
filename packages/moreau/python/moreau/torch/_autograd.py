"""PyTorch custom-op and autograd plumbing for moreau (private).

Defines the ``moreau::solve_backward`` custom op (with fake + vmap rules)
and the ``_SolveFunction`` autograd Function that the torch ``Solver``
dispatches through. Keeping this here isolates the functorch-sensitive
machinery from the public Solver API.
"""

import torch

# ---------------------------------------------------------------------------
# Custom op: moreau::solve_backward
#
# The backward is a custom op so we can register a vmap rule, enabling
# torch.func.jacrev / torch.vmap / is_grads_batched=True to batch
# multiple upstream gradient vectors through a single forward solve.
#
# The forward uses torch.autograd.Function (see _SolveFunction below)
# because register_autograd's generated class doesn't define
# setup_context, which functorch transforms require.
# ---------------------------------------------------------------------------

# Maps integer handles → solver._impl instances for the backward custom op.
# Weakrefs so the registry doesn't prevent garbage collection.
import weakref

_IMPL_REGISTRY: dict[int, weakref.ref] = {}
_next_handle: int = 0


def _register_impl(impl) -> int:
    global _next_handle
    h = _next_handle
    _next_handle += 1
    _IMPL_REGISTRY[h] = weakref.ref(impl, lambda _: _IMPL_REGISTRY.pop(h, None))
    return h


def _get_impl(handle: int):
    ref = _IMPL_REGISTRY.get(handle)
    if ref is None:
        raise RuntimeError("Solver impl not found in registry")
    impl = ref()
    if impl is None:
        raise RuntimeError("Solver has been garbage collected")
    return impl


@torch.library.custom_op("moreau::solve_backward", mutates_args=())
def _solve_backward_op(
    dx: torch.Tensor,
    dz: torch.Tensor,
    ds: torch.Tensor,
    impl_handle: torch.Tensor,
    solve_mode: torch.Tensor,
    state_rinv: torch.Tensor,
    state_rinv_diag: torch.Tensor,
    state_use_rinv_diag: torch.Tensor,
    state_n_active: torch.Tensor,
    state_ws: torch.Tensor,
    state_sense: torch.Tensor,
    state_lam_star: torch.Tensor,
    P_values: torch.Tensor,
    A_values: torch.Tensor,
    q: torch.Tensor,
    b: torch.Tensor,
    x: torch.Tensor,
    z: torch.Tensor,
    s: torch.Tensor,
    z_x: torch.Tensor,
    dz_x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward pass: compute gradients via implicit differentiation.

    `z_x` (the saved direct-x dual) and `dz_x` (the upstream gradient on
    Solution.z_x) may be empty tensors when the solver has no direct-x
    cones; the impl handles either case.
    """
    impl = _get_impl(impl_handle.item())
    mode_str = "single" if solve_mode.item() == 0 else "batch"
    dP, dq, dA, db = impl.backward_with_mode(
        dx,
        dz,
        ds,
        mode_str,
        P_values,
        A_values,
        q,
        b,
        x,
        z,
        s,
        state_rinv,
        state_rinv_diag,
        state_use_rinv_diag,
        state_n_active,
        state_ws,
        state_sense,
        state_lam_star,
        z_x=z_x,
        dz_x=dz_x,
    )
    return dP, dq, dA, db


@_solve_backward_op.register_fake
def _solve_backward_op_fake(
    dx,
    dz,
    ds,
    impl_handle,
    solve_mode,
    state_rinv,
    state_rinv_diag,
    state_use_rinv_diag,
    state_n_active,
    state_ws,
    state_sense,
    state_lam_star,
    P_values,
    A_values,
    q,
    b,
    x,
    z,
    s,
    z_x,
    dz_x,
):
    return (
        torch.empty_like(P_values),
        torch.empty_like(q),
        torch.empty_like(A_values),
        torch.empty_like(b),
    )


@_solve_backward_op.register_vmap
def _solve_backward_op_vmap(
    info,
    in_dims,
    dx,
    dz,
    ds,
    impl_handle,
    solve_mode,
    state_rinv,
    state_rinv_diag,
    state_use_rinv_diag,
    state_n_active,
    state_ws,
    state_sense,
    state_lam_star,
    P_values,
    A_values,
    q,
    b,
    x,
    z,
    s,
    z_x,
    dz_x,
):
    (
        dx_bd,
        dz_bd,
        ds_bd,
        h_bd,
        m_bd,
        state_rinv_bd,
        state_rinv_diag_bd,
        state_use_rinv_diag_bd,
        state_n_active_bd,
        state_ws_bd,
        state_sense_bd,
        state_lam_star_bd,
        P_bd,
        A_bd,
        q_bd,
        b_bd,
        x_bd,
        z_bd,
        s_bd,
        z_x_bd,
        dz_x_bd,
    ) = in_dims
    N = info.batch_size

    # Expand all tensors to (N, ...), replicating if not batched by vmap.
    def _expand(t, bd):
        if bd is not None:
            return t.movedim(bd, 0)
        return t.unsqueeze(0).expand(N, *t.shape).contiguous()

    dx = _expand(dx, dx_bd)
    dz = _expand(dz, dz_bd)
    ds = _expand(ds, ds_bd)
    state_rinv = _expand(state_rinv, state_rinv_bd)
    state_rinv_diag = _expand(state_rinv_diag, state_rinv_diag_bd)
    state_use_rinv_diag = _expand(state_use_rinv_diag, state_use_rinv_diag_bd)
    state_n_active = _expand(state_n_active, state_n_active_bd)
    state_ws = _expand(state_ws, state_ws_bd)
    state_sense = _expand(state_sense, state_sense_bd)
    state_lam_star = _expand(state_lam_star, state_lam_star_bd)
    P_values = _expand(P_values, P_bd)
    A_values = _expand(A_values, A_bd)
    q = _expand(q, q_bd)
    b = _expand(b, b_bd)
    x = _expand(x, x_bd)
    z = _expand(z, z_bd)
    s = _expand(s, s_bd)
    # Direct-x state: empty tensors when no x-cones; expand only when non-empty.
    if z_x.numel() > 0:
        z_x = _expand(z_x, z_x_bd)
    if dz_x.numel() > 0:
        dz_x = _expand(dz_x, dz_x_bd)

    # All tensors are now (N, ...).  Call the raw implementation directly
    # (not through the op dispatch) to avoid re-entering the vmap handler.
    batch_mode = torch.tensor(1, dtype=torch.int64)
    dP, dq_out, dA, db_out = _solve_backward_op._backend_fns[None](
        dx,
        dz,
        ds,
        impl_handle,
        batch_mode,
        state_rinv,
        state_rinv_diag,
        state_use_rinv_diag,
        state_n_active,
        state_ws,
        state_sense,
        state_lam_star,
        P_values,
        A_values,
        q,
        b,
        x,
        z,
        s,
        z_x,
        dz_x,
    )

    # All outputs are batched on dim 0
    return (dP, dq_out, dA, db_out), (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Autograd integration: _SolveFunction
#
# Uses torch.autograd.Function with setup_context so that functorch
# transforms (jacrev, vmap, grad) work.  The backward dispatches to
# _solve_backward_op (custom op with register_vmap) for efficient
# batched backward.
#
# The solver is passed as a non-tensor arg to .apply(); only _impl needs
# to be in the registry (for the backward custom op's vmap rule).
# ---------------------------------------------------------------------------


class _SolveFunction(torch.autograd.Function):
    @staticmethod
    def forward(solver, q, b, P_values, A_values):
        warm_kwargs = getattr(solver, "_pending_warm_kwargs", {})
        with torch.no_grad():
            result = solver._impl.solve(q, b, **warm_kwargs)
        solver._last_result = result
        # `z_x` (direct-x dual) is exposed when the solver has direct-x
        # cones; otherwise an empty tensor (same dtype/device as q).
        z_x = result.get("z_x")
        if z_x is None or (hasattr(z_x, "numel") and z_x.numel() == 0):
            z_x = torch.empty(0, dtype=q.dtype, device=q.device)
        return result["x"], result["z"], result["s"], z_x

    @staticmethod
    def setup_context(ctx, inputs, output):
        solver, q, b, P_values, A_values = inputs
        x, z, s, z_x = output
        mode = 0 if solver._impl._last_solve_mode == "single" else 1
        cached = getattr(solver, "_last_result", {}) or {}
        backward_state = cached.get("_backward_state")
        device = q.device
        if backward_state is None:
            state_rinv = torch.empty(0, dtype=torch.float64, device=device)
            state_rinv_diag = torch.empty(0, dtype=torch.float64, device=device)
            state_use_rinv_diag = torch.empty(0, dtype=torch.int64, device=device)
            state_n_active = torch.empty(0, dtype=torch.int64, device=device)
            state_ws = torch.empty(0, dtype=torch.int64, device=device)
            state_sense = torch.empty(0, dtype=torch.int64, device=device)
            state_lam_star = torch.empty(0, dtype=torch.float64, device=device)
        else:
            flat_state = backward_state._to_flat_dict()
            state_rinv = torch.as_tensor(flat_state["rinv"], dtype=torch.float64, device=device)
            state_rinv_diag = torch.as_tensor(
                flat_state["rinv_diag"], dtype=torch.float64, device=device
            )
            state_use_rinv_diag = torch.as_tensor(
                flat_state["use_rinv_diag"], dtype=torch.int64, device=device
            )
            state_n_active = torch.as_tensor(
                flat_state["n_active"], dtype=torch.int64, device=device
            )
            state_ws = torch.as_tensor(flat_state["ws"], dtype=torch.int64, device=device)
            state_sense = torch.as_tensor(flat_state["sense"], dtype=torch.int64, device=device)
            state_lam_star = torch.as_tensor(
                flat_state["lam_star"], dtype=torch.float64, device=device
            )
        ctx.solver = solver
        ctx.save_for_backward(
            torch.tensor(solver._impl_handle, dtype=torch.int64),
            torch.tensor(mode, dtype=torch.int64),
            state_rinv,
            state_rinv_diag,
            state_use_rinv_diag,
            state_n_active,
            state_ws,
            state_sense,
            state_lam_star,
            P_values,
            A_values,
            q,
            b,
            x,
            z,
            s,
            z_x,
        )

    @staticmethod
    def backward(ctx, dx, dz, ds, dz_x):
        (
            impl_handle,
            solve_mode,
            state_rinv,
            state_rinv_diag,
            state_use_rinv_diag,
            state_n_active,
            state_ws,
            state_sense,
            state_lam_star,
            P_values,
            A_values,
            q,
            b,
            x,
            z,
            s,
            z_x,
        ) = ctx.saved_tensors

        if dx is None:
            dx = torch.zeros_like(x)
        if dz is None:
            dz = torch.zeros_like(z)
        if ds is None:
            ds = torch.zeros_like(s)
        if dz_x is None:
            dz_x = torch.zeros_like(z_x)

        # Resolve the op via the moreau.torch namespace so tests can
        # monkeypatch `moreau.torch._solve_backward_op`.
        from moreau import torch as _moreau_torch

        backward_op = getattr(_moreau_torch, "_solve_backward_op", _solve_backward_op)
        dP, dq, dA, db = backward_op(
            dx,
            dz,
            ds,
            impl_handle,
            solve_mode,
            state_rinv,
            state_rinv_diag,
            state_use_rinv_diag,
            state_n_active,
            state_ws,
            state_sense,
            state_lam_star,
            P_values,
            A_values,
            q,
            b,
            x,
            z,
            s,
            z_x,
            dz_x,
        )

        # Break reference cycle: solver -> _last_result -> tensors -> grad_fn -> ctx
        if hasattr(ctx.solver, "_last_result"):
            ctx.solver._last_result = None

        # Gradients in same order as inputs: solver, q, b, P_values, A_values
        return None, dq, db, dP, dA


__all__ = [
    "_IMPL_REGISTRY",
    "_register_impl",
    "_get_impl",
    "_solve_backward_op",
    "_SolveFunction",
]
