"""
Copyright, the CVXPY authors

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

from ._autograd import _solve_backward_op
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


@torch.library.custom_op("moreau::solve_cuda", mutates_args=())
def _solve_cuda(
    P: torch.Tensor,
    A: torch.Tensor,
    q: torch.Tensor,
    b: torch.Tensor,
    handle: int,
    direct_dual_size: int,
    warm: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Only native execution touches the solver or its diagnostic metadata.
    # Backward saves explicit data and solutions, never the latest solve state.
    solver = _SOLVERS[handle]
    warm_start = None
    if warm:
        cls = TorchSolution if q.ndim == 1 else TorchBatchedSolution
        warm_start = cls(*warm)
    solution = solver._solve_eager(P, A, q, b, warm_start=warm_start)
    return (
        solution.x,
        solution.z,
        solution.s,
        solution.z_x,
        torch.from_numpy(_ImplHandle(solver)),
    )


@_solve_cuda.register_fake
def _solve_cuda_fake(P, A, q, b, handle, direct_dual_size, warm):
    zx_shape = (*q.shape[:-1], direct_dual_size) if direct_dual_size else (0,)
    return (
        q.new_empty(q.shape),
        b.new_empty(b.shape),
        b.new_empty(b.shape),
        q.new_empty(zx_shape),
        torch.empty((), dtype=torch.int64, device="cpu"),
    )


def _setup_context(ctx, inputs, output):
    P, A, q, b, _handle, _direct_dual_size, warm = inputs
    x, z, s, z_x, impl_handle = output
    ctx.save_for_backward(P, A, q, b, x, z, s, z_x, impl_handle)
    ctx.mark_non_differentiable(impl_handle)
    ctx.warm_count = len(warm)


def _backward(ctx, dx, dz, ds, dz_x, _):
    P, A, q, b, x, z, s, z_x, impl_handle = ctx.saved_tensors
    empty = q.new_empty(0)
    empty_int = torch.empty(0, dtype=torch.int64, device=q.device)
    mode = torch.tensor(int(q.ndim > 1), dtype=torch.int64)
    dP, dq, dA, db = _solve_backward_op(
        torch.zeros_like(x) if dx is None else dx,
        torch.zeros_like(z) if dz is None else dz,
        torch.zeros_like(s) if ds is None else ds,
        impl_handle,
        mode,
        empty,
        empty,
        empty_int,
        empty_int,
        empty_int,
        empty_int,
        empty,
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
    return dP, dA, dq, db, None, None, [None] * ctx.warm_count


_solve_cuda.register_autograd(_backward, setup_context=_setup_context)


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
    x, z, s, z_x, _ = _solve_cuda(
        P, A, q, b, solver._compile_handle, solver._direct_dual_size, warm
    )
    cls = TorchSolution if q.ndim == 1 else TorchBatchedSolution
    return cls(x, z, s, z_x)
