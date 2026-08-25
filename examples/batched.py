#!/usr/bin/env python3
"""Example: Solving a batch of QPs with CompiledSolver.

This example shows how to solve a batch of quadratic programs
using moreau's CompiledSolver interface. When 2D arrays are passed,
the solver handles batched solving efficiently.
"""

import numpy as np
from scipy import sparse
import moreau
from utils import check_status

if __name__ == "__main__":
    print("=" * 60)
    print("Moreau Batched Example: Simple QP")
    print("=" * 60)

    print("\nProblem 1:")
    print("  min  x1^2 + x2^2 - x1 - x2")
    print("  s.t. x1 + x2 = 1")

    print("\nProblem 2:")
    print("  min  2x1^2 + x2^2 - 2x1 - 2x2")
    print("  s.t. x1 + x2 = 2")

    # Problem data (CSR format) - both problems share same sparsity pattern
    P1 = sparse.csr_array(np.array([[2.0, 0.0], [0.0, 2.0]]))
    P2 = sparse.csr_array(np.array([[4.0, 0.0], [0.0, 2.0]]))
    q1 = np.array([-1.0, -1.0])
    q2 = np.array([-2.0, -2.0])
    A1 = sparse.csr_array(np.array([[1.0, 1.0]]))
    A2 = sparse.csr_array(np.array([[1.0, 1.0]]))
    b1 = np.array([1.0])
    b2 = np.array([2.0])
    m, n = A1.shape
    batch_size = 2

    # Stack the problem data for the batch, shape: (batch_size, num_entries)
    P_values = np.vstack([P1.data, P2.data])  # (batch_size, nnz_P) = (2, 2)
    A_values = np.vstack([A1.data, A2.data])  # (batch_size, nnz_A) = (2, 2)
    qs = np.vstack([q1, q2])  # (batch_size, n) = (2, 2)
    bs = np.vstack([b1, b2])  # (batch_size, m) = (2, 1)

    # Cones: equality constraint (zero cone)
    cones = moreau.Cones(num_zero_cones=m)

    print(f"\n  n = {n}, m = {m}, batch_size = {batch_size}")

    settings = moreau.Settings(verbose=True, batch_size=batch_size)

    # Create CompiledSolver - all problems share the same sparsity pattern
    solver = moreau.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=P1.indptr,
        P_col_indices=P1.indices,
        A_row_offsets=A1.indptr,
        A_col_indices=A1.indices,
        cones=cones,
        settings=settings,
    )

    # Set matrices (batch of P and A values)
    solver.setup(P_values, A_values)

    # Solve batch
    print("\n--- Solving Batch ---")
    result = solver.solve(qs, bs)
    info = solver.info

    # Always check status before using solution
    if not check_status(info):
        print("Some problems failed - solutions may be unreliable")

    print("\nFirst problem solution:")
    print(f"  x = [{result.x[0][0]:.6f}, {result.x[0][1]:.6f}]")
    print(f"  z = [{result.z[0][0]:.6f}]")
    print(f"  s = [{result.s[0][0]:.6f}]")
    print(f"  obj = {info.obj_val[0]:.6f}")
    print(f"  status = {info.status[0].name}")

    print("\nSecond problem solution:")
    print(f"  x = [{result.x[1][0]:.6f}, {result.x[1][1]:.6f}]")
    print(f"  z = [{result.z[1][0]:.6f}]")
    print(f"  s = [{result.s[1][0]:.6f}]")
    print(f"  obj = {info.obj_val[1]:.6f}")
    print(f"  status = {info.status[1].name}")

    # Verify solutions
    print("\n--- Verification ---")
    for i, (qi, bi) in enumerate([(q1, b1), (q2, b2)]):
        x = result.x[i]
        print(f"Problem {i+1}:")
        print(f"  Constraint x1 + x2 = {bi[0]}: actual = {x[0] + x[1]:.6f}")

    print("\n" + "=" * 60)
    print("Batched QP example completed!")
    print("=" * 60)
