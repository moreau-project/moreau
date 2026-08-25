"""PyTorch-specific solution and info dataclasses for moreau.

These mirror the numpy types in ``moreau._types`` but carry torch tensors.
They live here so the torch integration owns its own types; the package
root re-exports them from ``moreau._types`` for backward compatibility.
"""

from dataclasses import dataclass
from typing import List

from moreau._types import SolverStatus, WarmStart, BatchedWarmStart


@dataclass
class TorchSolveInfo:
    """Solver metadata from a PyTorch solve operation.

    Contains solver status, objective value, iteration count, and timing info.
    Returned as the second element of the (TorchSolution, TorchSolveInfo) tuple.

    Attributes:
        status: Solver status (SolverStatus enum)
        obj_val: Objective value tensor at solution
        iterations: Number of iterations tensor
        solve_time: Time spent in solve phase (seconds)
        setup_time: Time spent setting matrix values (seconds)
        construction_time: Time spent constructing solver structure (seconds)

    Example:
        >>> from moreau.torch import Solver
        >>> solution = solver.solve(P_values, A_values, q, b)
        >>> print(solver.info.status)
        >>> print(info.status)
        >>> print(f"Solved in {int(info.iterations)} iterations")
    """

    status: SolverStatus
    obj_val: "torch.Tensor"
    iterations: "torch.Tensor"
    solve_time: float
    setup_time: float = 0.0
    construction_time: float = 0.0

    def __repr__(self) -> str:
        return (
            f"TorchSolveInfo(status={self.status.name}, "
            f"obj_val={float(self.obj_val):.6g}, "
            f"iterations={int(self.iterations)})"
        )


@dataclass
class TorchSolution:
    """Solution vectors for a single PyTorch conic optimization problem.

    Contains only the optimization variables (primal and dual solutions).
    Returned as the first element of the (TorchSolution, TorchSolveInfo) tuple.

    Attributes:
        x: Primal solution tensor, shape (n,)
        z: Dual variable tensor (Lagrange multipliers), shape (m,)
        s: Slack variable tensor, shape (m,)

    Example:
        >>> from moreau.torch import Solver
        >>> solution = solver.solve(P_values, A_values, q, b)
        >>> print(solver.info.status)
        >>> print(solution.x)  # Access primal solution tensor
        >>> loss = solution.x.sum()
        >>> loss.backward()  # Gradients flow through solution
    """

    x: "torch.Tensor"
    z: "torch.Tensor"
    s: "torch.Tensor"
    z_x: "torch.Tensor" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.z_x is None:
            import torch as _torch

            self.z_x = _torch.zeros(0, dtype=_torch.float64)

    def __repr__(self) -> str:
        n = self.x.shape[-1] if hasattr(self.x, "shape") else 0
        return f"TorchSolution(n={n})"

    def to_warm_start(self) -> WarmStart:
        """Create a WarmStart from this solution (detaches and moves to CPU)."""
        import numpy as np

        zx_arr = None
        zx = getattr(self, "z_x", None)
        if zx is not None and zx.numel() > 0:
            zx_arr = zx.detach().cpu().numpy().astype(np.float64)
        return WarmStart(
            x=self.x.detach().cpu().numpy().astype(np.float64),
            z=self.z.detach().cpu().numpy().astype(np.float64),
            s=self.s.detach().cpu().numpy().astype(np.float64),
            z_x=zx_arr,
        )


@dataclass
class TorchBatchedSolveInfo:
    """Solver metadata from a batched PyTorch solve operation.

    Contains per-problem status and objective values, plus shared timing info.
    Returned as the second element of the (TorchBatchedSolution, TorchBatchedSolveInfo) tuple.

    Attributes:
        status: List of solver statuses, one per problem
        obj_val: Objective values tensor, shape (batch_size,)
        iterations: Iterations tensor
        solve_time: Time spent in solve phase (seconds)
        setup_time: Time spent setting matrix values (seconds)
        construction_time: Time spent constructing solver structure (seconds)

    Example:
        >>> from moreau.torch import Solver
        >>> solution = solver.solve(P_values, A_values, q_batch, b_batch)
        >>> print(info.status[0])  # Check first problem's status
        >>> print(f"Batch solved in {info.solve_time:.3f}s")
    """

    status: List[SolverStatus]
    obj_val: "torch.Tensor"
    iterations: "torch.Tensor"
    solve_time: float
    setup_time: float = 0.0
    construction_time: float = 0.0

    def __repr__(self) -> str:
        batch_size = len(self.status)
        # Handle both tensor and scalar iterations
        if hasattr(self.iterations, "__len__") and len(self.iterations) > 1:
            iters_str = f"[{', '.join(str(int(i)) for i in self.iterations)}]"
        else:
            iters_str = str(int(self.iterations))
        return (
            f"TorchBatchedSolveInfo(batch_size={batch_size}, "
            f"iterations={iters_str}, solve_time={self.solve_time:.6f}s)"
        )


@dataclass
class TorchBatchedSolution:
    """Solution vectors for batched PyTorch conic optimization problems.

    Contains only the optimization variables (primal and dual solutions).
    Returned as the first element of the (TorchBatchedSolution, TorchBatchedSolveInfo) tuple.

    Attributes:
        x: Primal solutions tensor, shape (batch_size, n)
        z: Dual variables tensor, shape (batch_size, m)
        s: Slack variables tensor, shape (batch_size, m)

    Example:
        >>> from moreau.torch import Solver
        >>> solution = solver.solve(P_values, A_values, q_batch, b_batch)
        >>> print(solution.x.shape)  # (batch_size, n)
        >>> sol = solution[1]  # Get TorchSolution for 2nd problem
        >>> print(sol.x)  # 1D tensor for that problem
    """

    x: "torch.Tensor"
    z: "torch.Tensor"
    s: "torch.Tensor"
    z_x: "torch.Tensor" = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.z_x is None:
            import torch as _torch

            batch_size = self.x.shape[0] if hasattr(self.x, "shape") else len(self.x)
            self.z_x = _torch.zeros((batch_size, 0), dtype=_torch.float64)

    def __repr__(self) -> str:
        batch_size = self.x.shape[0] if hasattr(self.x, "shape") else len(self.x)
        return f"TorchBatchedSolution(batch_size={batch_size})"

    def __len__(self) -> int:
        """Return the batch size."""
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> TorchSolution:
        """Get TorchSolution for a single problem in the batch.

        Args:
            idx: Index of the problem in the batch (0-based)

        Returns:
            TorchSolution object for that problem

        Example:
            >>> solution = solver.solve(P_values, A_values, q_batch, b_batch)
            >>> sol = solution[1]  # Get 2nd problem's solution
            >>> print(sol.x)     # 1D tensor
        """
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for batch size {len(self)}")

        return TorchSolution(
            x=self.x[idx],
            z=self.z[idx],
            s=self.s[idx],
            z_x=self.z_x[idx] if self.z_x.numel() else self.z_x,
        )

    def __iter__(self):
        """Iterate over solutions in the batch."""
        for i in range(len(self)):
            yield self[i]

    def to_warm_start(self) -> BatchedWarmStart:
        """Create a BatchedWarmStart from this solution (detaches and moves to CPU)."""
        import numpy as np

        zx_arr = None
        zx = getattr(self, "z_x", None)
        if zx is not None and zx.numel() > 0:
            zx_arr = zx.detach().cpu().numpy().astype(np.float64)
        return BatchedWarmStart(
            x=self.x.detach().cpu().numpy().astype(np.float64),
            z=self.z.detach().cpu().numpy().astype(np.float64),
            s=self.s.detach().cpu().numpy().astype(np.float64),
            z_x=zx_arr,
        )


__all__ = [
    "TorchSolveInfo",
    "TorchSolution",
    "TorchBatchedSolveInfo",
    "TorchBatchedSolution",
]
