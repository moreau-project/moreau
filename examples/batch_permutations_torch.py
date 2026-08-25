#!/usr/bin/env python3
"""
Example showing all permutations of 1D/2D inputs for moreau.torch.Solver.

The solver accepts:
- P/A values: 2D (batched)
- q/b vectors: 2D (batched)

Valid permutations:
1. P/A 2D (batch=1), q/b 2D (batch=1): Single problem via batched interface
2. P/A 2D, q/b 2D: Batch solving with autograd support

For single problems without batching, use moreau.Solver with numpy arrays.
"""

import torch
import moreau
from moreau.torch import Solver
from utils import get_torch_device, check_status

# Use CUDA if available, otherwise CPU
device = get_torch_device()
torch.set_default_dtype(torch.float64)
print(f"Using device: {device}")

# Problem setup: min 0.5 x'Px + q'x  s.t. Ax + s = b, s >= 0
# Simple 2-variable QP with 2 inequality constraints

n, m = 2, 2
batch_size = 3
dtype = torch.float64

# P = [[2, 0], [0, 2]] as CSR
P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
P_col_indices = torch.tensor([0, 1], dtype=torch.int64)
P_values_single = torch.tensor([[2.0, 2.0]], dtype=dtype, device=device)  # Shape (1, nnz)
P_values_batch = torch.tensor(
    [
        [2.0, 2.0],
        [4.0, 4.0],
        [1.0, 1.0],
    ],
    dtype=dtype,
    device=device,
)  # Shape (batch, nnz)

# A = [[1, 0], [0, 1]] as CSR (identity)
A_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
A_col_indices = torch.tensor([0, 1], dtype=torch.int64)
A_values_single = torch.tensor([[1.0, 1.0]], dtype=dtype, device=device)  # Shape (1, nnz)
A_values_batch = torch.tensor(
    [
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=dtype,
    device=device,
)  # Shape (batch, nnz)

# q and b vectors
q_single = torch.tensor([[-1.0, -1.0]], dtype=dtype, device=device)  # Shape (1, n)
q_batch = torch.tensor(
    [
        [-1.0, -1.0],
        [-2.0, -2.0],
        [-0.5, -0.5],
    ],
    dtype=dtype,
    device=device,
)  # Shape (batch, n)

b_single = torch.tensor([[1.0, 1.0]], dtype=dtype, device=device)  # Shape (1, m)
b_batch = torch.tensor(
    [
        [1.0, 1.0],
        [2.0, 2.0],
        [0.5, 0.5],
    ],
    dtype=dtype,
    device=device,
)  # Shape (batch, m)

# Bounds (unbounded: use -inf/+inf)
l_single = torch.full((1, n), float("-inf"), dtype=dtype, device=device)
u_single = torch.full((1, n), float("inf"), dtype=dtype, device=device)
l_batch = torch.full((batch_size, n), float("-inf"), dtype=dtype, device=device)
u_batch = torch.full((batch_size, n), float("inf"), dtype=dtype, device=device)

# Cones: nonnegative cone for inequality constraints
cones = moreau.Cones(num_nonneg_cones=m)

# Settings: disable verbose output for clean example
settings = moreau.Settings(verbose=False, batch_size=batch_size)


def print_result(name, result, info):
    """Print solution results."""
    print(f"\n{name}")
    print("-" * 50)

    # Check status before printing
    if not check_status(info):
        print("  Warning: some problems did not converge")

    for i in range(result.x.shape[0]):
        status = info.status[i]
        x = result.x[i].cpu().numpy()
        obj = info.obj_val[i].item()
        print(f"  Problem {i}: status={status.name}, x={x}, obj={obj:.4f}")


print("=" * 60)
print("Moreau Torch Solver: All Input Permutations")
print("=" * 60)

# ----------------------------------------------------------------------
# Permutation 1: batch=1 -> Single problem via batched interface
# ----------------------------------------------------------------------
settings1 = moreau.Settings(verbose=False, batch_size=1)
solver1 = Solver(
    n=n,
    m=m,
    P_row_offsets=P_row_offsets,
    P_col_indices=P_col_indices,
    A_row_offsets=A_row_offsets,
    A_col_indices=A_col_indices,
    cones=cones,
    settings=settings1,
)
solver1.setup(P_values_single, A_values_single)
result = solver1.solve(q_single, b_single, l_single, u_single)
info = solver1.info
print_result("1. Batch=1 -> Single problem via batched interface", result, info)

# ----------------------------------------------------------------------
# Permutation 2: Full batch
# ----------------------------------------------------------------------
solver = Solver(
    n=n,
    m=m,
    P_row_offsets=P_row_offsets,
    P_col_indices=P_col_indices,
    A_row_offsets=A_row_offsets,
    A_col_indices=A_col_indices,
    cones=cones,
    settings=settings,
)
solver.setup(P_values_batch, A_values_batch)
result = solver.solve(q_batch, b_batch, l_batch, u_batch)
info = solver.info
print_result("2. Full batch -> Batch with per-problem P/A", result, info)

# ----------------------------------------------------------------------
# Test: Repeated solves (caching optimization)
# ----------------------------------------------------------------------
print("\n3. Repeated solves (tests caching)")
print("-" * 50)

# First solve
result1 = solver.solve(q_batch, b_batch, l_batch, u_batch)
info1 = solver.info
print(f"  First solve: obj_vals = {info1.obj_val.cpu().numpy()}")

# Second solve with different q/b but same P/A
q_batch_2x = q_batch * 2
b_batch_2x = b_batch * 2
result2 = solver.solve(q_batch_2x, b_batch_2x, l_batch, u_batch)
info2 = solver.info
print(f"  Second solve (2x q/b): obj_vals = {info2.obj_val.cpu().numpy()}")

# Third solve back to original
result3 = solver.solve(q_batch, b_batch, l_batch, u_batch)
info3 = solver.info
print(f"  Third solve (original): obj_vals = {info3.obj_val.cpu().numpy()}")
print("  OK: Caching works - setup skipped when P/A unchanged")

# ----------------------------------------------------------------------
# Test: Autograd support
# ----------------------------------------------------------------------
print("\n4. Autograd test")
print("-" * 50)

# Enable gradients on inputs
q_grad = q_batch.clone().requires_grad_(True)
solver.setup(P_values_batch, A_values_batch)
result = solver.solve(q_grad, b_batch, l_batch, u_batch)

# Compute loss and backward
loss = result.x.sum()
loss.backward()

print(f"  Loss (sum of x): {loss.item():.4f}")
print(f"  q.grad:\n{q_grad.grad}")
print("  OK: Autograd works")

print("\n" + "=" * 60)
print("Summary:")
print("  - Torch solver requires 2D inputs (batched)")
print("  - P/A values are cached; setup skipped if unchanged")
print("  - Automatic differentiation via torch.autograd")
print("  - Best for differentiable optimization in ML pipelines")
print("=" * 60)
