#!/usr/bin/env python3
"""
Example showing all permutations of 1D/2D inputs for moreau.CompiledSolver.

The CompiledSolver accepts:
- P/A values: 1D (shared) or 2D (per-problem)
- q/b vectors: 2D (batched)

Valid permutations:
1. P/A 1D, q/b 2D (batch=1): Single problem via batched interface
2. P/A 1D, q/b 2D: Batch with shared P/A (optimized path)
3. P/A 2D, q/b 2D: Batch with per-problem P/A

For single problems without batching, use moreau.Solver instead.
"""

import numpy as np
import moreau
from utils import check_status

# Problem setup: min 0.5 x'Px + q'x  s.t. Ax + s = b, s >= 0
# Simple 2-variable QP with 2 inequality constraints

n, m = 2, 2
batch_size = 3

# P = [[2, 0], [0, 2]] as CSR
P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
P_col_indices = np.array([0, 1], dtype=np.int64)
P_values_single = np.array([2.0, 2.0])  # 1D: shared
P_values_batch = np.array(
    [
        [2.0, 2.0],
        [4.0, 4.0],
        [1.0, 1.0],
    ]
)  # 2D: per-problem

# A = [[1, 0], [0, 1]] as CSR (identity)
A_row_offsets = np.array([0, 1, 2], dtype=np.int64)
A_col_indices = np.array([0, 1], dtype=np.int64)
A_values_single = np.array([1.0, 1.0])  # 1D: shared
A_values_batch = np.array(
    [
        [1.0, 1.0],
        [1.0, 1.0],
        [1.0, 1.0],
    ]
)  # 2D: per-problem

# q and b vectors (batched - CompiledSolver requires 2D)
q_batch = np.array(
    [
        [-1.0, -1.0],
        [-2.0, -2.0],
        [-0.5, -0.5],
    ]
)  # 2D: batched

b_batch = np.array(
    [
        [1.0, 1.0],
        [2.0, 2.0],
        [0.5, 0.5],
    ]
)  # 2D: batched

# Cones: nonnegative cone for inequality constraints
cones = moreau.Cones(num_nonneg_cones=m)

# Settings: disable verbose output for clean example
settings = moreau.Settings(verbose=False, batch_size=batch_size)
settings_single = moreau.Settings(verbose=False, batch_size=1)


def print_result(name, result, info):
    """Print solution results."""
    print(f"\n{name}")
    print("-" * 50)

    # Check status before printing
    if not check_status(info):
        print("  Warning: some problems did not converge")

    # Handle both scalar and array/list obj_val (batch_size=1 may return scalar)
    obj_vals = info.obj_val
    if not hasattr(obj_vals, "__len__"):
        obj_vals = [obj_vals]

    for i, (status, x, obj) in enumerate(zip(info.status, result.x, obj_vals)):
        print(f"  Problem {i}: status={status.name}, x={x}, obj={obj:.4f}")


# Create solver once (structure is fixed)
solver = moreau.CompiledSolver(
    n=n,
    m=m,
    P_row_offsets=P_row_offsets,
    P_col_indices=P_col_indices,
    A_row_offsets=A_row_offsets,
    A_col_indices=A_col_indices,
    cones=cones,
    settings=settings,
)

print("=" * 60)
print("Moreau CompiledSolver: All Input Permutations")
print("=" * 60)

# ----------------------------------------------------------------------
# Permutation 1: P/A 1D, q/b 2D (batch=1) -> Single problem via batch
# ----------------------------------------------------------------------
q_single = q_batch[:1]  # Shape (1, n)
b_single = b_batch[:1]  # Shape (1, m)

# Need to recreate solver for batch=1
solver1 = moreau.CompiledSolver(
    n=n,
    m=m,
    P_row_offsets=P_row_offsets,
    P_col_indices=P_col_indices,
    A_row_offsets=A_row_offsets,
    A_col_indices=A_col_indices,
    cones=cones,
    settings=settings_single,
)
solver1.setup(P_values_single, A_values_single)
result = solver1.solve(q_single, b_single)
info = solver1.info
print_result("1. P/A 1D (shared), q/b 2D (batch=1) -> Single problem", result, info)

# ----------------------------------------------------------------------
# Permutation 2: P/A 1D, q/b 2D -> Batch with shared P/A (OPTIMIZED)
# ----------------------------------------------------------------------
solver.setup(P_values_single, A_values_single)
result = solver.solve(q_batch, b_batch)
info = solver.info
print_result(
    "2. P/A 1D (shared), q/b 2D (batch) -> Batch with shared P/A [OPTIMIZED]", result, info
)

# ----------------------------------------------------------------------
# Permutation 3: P/A 2D, q/b 2D -> Batch with per-problem P/A
# ----------------------------------------------------------------------
solver.setup(P_values_batch, A_values_batch)
result = solver.solve(q_batch, b_batch)
info = solver.info
print_result("3. P/A 2D (batch), q/b 2D (batch) -> Batch with per-problem P/A", result, info)

# ----------------------------------------------------------------------
# Test: Repeated solves with shared P/A (caching optimization)
# ----------------------------------------------------------------------
print("\n4. Repeated solves with shared P/A (tests caching)")
print("-" * 50)
solver.setup(P_values_single, A_values_single)

# First solve
result1 = solver.solve(q_batch, b_batch)
info1 = solver.info
print(f"  First solve: obj_vals = {info1.obj_val}")

# Second solve with different q/b but same P/A
q_batch_2x = q_batch * 2
b_batch_2x = b_batch * 2
result2 = solver.solve(q_batch_2x, b_batch_2x)
info2 = solver.info
print(f"  Second solve (2x q/b): obj_vals = {info2.obj_val}")

# Third solve back to original
result3 = solver.solve(q_batch, b_batch)
info3 = solver.info
print(f"  Third solve (original): obj_vals = {info3.obj_val}")
print("  OK: Caching works - setup skipped when P/A unchanged")

print("\n" + "=" * 60)
print("Summary:")
print("  - Permutation 2 uses optimized shared P/A path")
print("  - Equilibration computed once, shared across batch")
print("  - P/A values are cached; setup skipped if unchanged")
print("  - Best for parametric QPs where only q/b change")
print("=" * 60)
