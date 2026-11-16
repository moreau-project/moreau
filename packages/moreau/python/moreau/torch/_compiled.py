"""
Copyright, the Moreau authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import weakref

import numpy as np
import torch

from ._autograd import _solve_backward_op, _status_tensor
from ._types import TorchBatchedSolution, TorchSolution

_SOLVERS = weakref.WeakValueDictionary()
_next_handle = 0


class _ImplHandle(np.ndarray):
    """Keep the native implementation alive through the saved tensor's storage.

    from_numpy retains this array until the last storage alias is released,
    including detach() aliases made by autograd's saved-tensor machinery.
    A weak registry alone loses solvers constructed locally during forward.
    """

    def __new__(cls, solver):
        handle = np.array(solver._impl_handle, dtype=np.int64).view(cls)
        handle.impl = solver._impl
        return handle


def _register_solver(solver):
    global _next_handle
    handle = _next_handle
    _next_handle += 1
    _SOLVERS[handle] = solver
    return handle


@torch.library.custom_op("moreau::solve", mutates_args=())
def _solve(
    P: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor,
    b: torch.Tensor,
    handle: int,
    direct_dual_size: int,
    active_set: bool,
    warm: list[torch.Tensor],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor],
]:
    # Only native execution touches the solver or its diagnostic metadata.
    # Backward saves explicit data and solutions, never the latest solve state.
    solver = _SOLVERS[handle]
    warm_start = None
    if warm:
        cls = TorchSolution if q.ndim == 1 else TorchBatchedSolution
        warm_start = cls(*warm)
    solution = solver._solve_eager(P, A, q, b, warm_start=warm_start)
    state = []
    if active_set:
        flat = solver._last_result["_backward_state"]._to_flat_dict()
        for name in ("rinv", "rinv_diag", "use_rinv_diag", "n_active", "ws", "sense", "lam_star"):
            dtype = torch.float64 if name in ("rinv", "rinv_diag", "lam_star") else torch.int64
            state.append(torch.tensor(flat[name], dtype=dtype, device=q.device))
    return (
        solution.x,
        solution.z,
        solution.s,
        solution.z_x,
        torch.from_numpy(_ImplHandle(solver)),
        _status_tensor(solver._last_result["status"]),
        state,
    )


@_solve.register_fake
def _solve_fake(P, A, q, b, handle, direct_dual_size, active_set, warm):
    zx_shape = (*q.shape[:-1], direct_dual_size) if direct_dual_size else (0,)
    state = []
    if active_set:
        batch = q.shape[0] if q.ndim > 1 else 1
        n, m = q.shape[-1], b.shape[-1]
        for i, size in enumerate((n * (n + 1) // 2, n, 1, 1, m, m, m)):
            dtype = torch.float64 if i in (0, 1, 6) else torch.int64
            state.append(torch.empty(batch * size, dtype=dtype, device=q.device))
    return (
        q.new_empty(q.shape),
        b.new_empty(b.shape),
        b.new_empty(b.shape),
        q.new_empty(zx_shape),
        torch.empty((), dtype=torch.int64, device="cpu"),
        torch.empty(q.shape[:-1], dtype=torch.int64, device="cpu"),
        state,
    )


def _setup_context(ctx, inputs, output):
    P, A, q, b, _handle, _direct_dual_size, _active_set, warm = inputs
    x, z, s, z_x, impl_handle, status, state = output
    ctx.save_for_backward(P, A, q, b, x, z, s, z_x, impl_handle, status, *state)
    ctx.mark_non_differentiable(impl_handle, status, *state)
    ctx.warm_count = len(warm)


def _backward(ctx, dx, dz, ds, dz_x, _handle_grad, _status_grad, _state_grads):
    P, A, q, b, x, z, s, z_x, impl_handle, status, *state = ctx.saved_tensors
    empty = q.new_empty(0)
    empty_int = torch.empty(0, dtype=torch.int64, device=q.device)
    if not state:
        state = [empty, empty, empty_int, empty_int, empty_int, empty_int, empty]
    mode = torch.tensor(int(q.ndim > 1), dtype=torch.int64)
    dP, dq, dA, db = _solve_backward_op(
        torch.zeros_like(x) if dx is None else dx,
        torch.zeros_like(z) if dz is None else dz,
        torch.zeros_like(s) if ds is None else ds,
        impl_handle,
        mode,
        status,
        *state,
        P,
        A,
        q,
        b,
        x,
        z,
        s,
        z_x,
        torch.zeros_like(z_x) if dz_x is None else dz_x,
    )
    return dP, dA, dq, db, None, None, None, [None] * ctx.warm_count


_solve.register_autograd(_backward, setup_context=_setup_context)


def _compiled_solve(solver, P, A, q, b, warm_start):
    warm = []
    if warm_start is not None:
        warm = [
            torch.as_tensor(t, dtype=q.dtype, device=q.device).detach()
            for t in (warm_start.x, warm_start.z, warm_start.s)
        ]
        zx = warm_start.z_x
        warm.append(
            q.new_empty(0)
            if zx is None
            else torch.as_tensor(zx, dtype=q.dtype, device=q.device).detach()
        )
    x, z, s, z_x, _, _status, _state = _solve(
        P,
        A,
        q,
        b,
        solver._compile_handle,
        solver._direct_dual_size,
        solver._compile_active_set,
        warm,
    )
    cls = TorchSolution if q.ndim == 1 else TorchBatchedSolution
    return cls(x, z, s, z_x)
