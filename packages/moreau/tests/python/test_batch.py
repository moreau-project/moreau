"""Tests for batch solving (parallel batch solving).

These tests verify:
1. CompiledSolver can handle 2D (batched) inputs
2. CompiledSolver solves single and multiple problems correctly
3. Results match expected solutions
"""

import moreau
import pytest
import numpy as np
from scipy import sparse


@pytest.fixture
def batch_qp_setup():
    """Setup for batch QP tests.

    Problem:
        minimize    x'Px + q'x
        subject to  Ax + s = b, s in K

    With K = {zero cone (equality), nonnegative cone (inequalities)}
    """
    # Problem dimensions
    n = 2  # number of variables
    m = 5  # number of constraints

    # Quadratic cost: minimize x'Px + q'x with P = 2*I
    # P in CSR format (diagonal)
    P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    P_col_indices = np.array([0, 1], dtype=np.int64)
    P_values = np.array([2.0, 2.0])

    # Constraints: Ax + s = b, s in K
    # Row 0: x + y = 1 (equality)
    # Row 1: -x <= 0 (x >= 0)
    # Row 2: -y <= 0 (y >= 0)
    # Row 3: x <= 1
    # Row 4: y <= 1
    A_row_offsets = np.array([0, 2, 3, 4, 5, 6], dtype=np.int64)
    A_col_indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    A_values = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])

    # Cones
    cones = moreau.Cones()
    cones.num_zero_cones = 1  # equality constraint
    cones.num_nonneg_cones = 4  # inequality constraints

    settings = moreau.Settings()
    settings.verbose = False

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "cones": cones,
        "settings": settings,
        "nnzP": len(P_col_indices),
        "nnzA": len(A_col_indices),
    }


def test_compiled_solver_creation_with_batch_hint(batch_qp_setup, device):
    """Test that CompiledSolver can be created with batch_size hint."""
    d = batch_qp_setup
    batch_size = 4

    settings = moreau.Settings(device=device, batch_size=batch_size, verbose=False)
    solver = moreau.CompiledSolver(
        n=d["n"],
        m=d["m"],
        P_row_offsets=d["P_row_offsets"],
        P_col_indices=d["P_col_indices"],
        A_row_offsets=d["A_row_offsets"],
        A_col_indices=d["A_col_indices"],
        cones=d["cones"],
        settings=settings,
    )

    assert solver is not None
    assert solver.n == d["n"]
    assert solver.m == d["m"]


def test_batch_solve_single_problem(batch_qp_setup, device):
    """Test solving a single problem using 2D inputs."""
    d = batch_qp_setup
    batch_size = 1
    n = d["n"]

    settings = moreau.Settings(device=device, batch_size=batch_size, verbose=False)
    solver = moreau.CompiledSolver(
        n=d["n"],
        m=d["m"],
        P_row_offsets=d["P_row_offsets"],
        P_col_indices=d["P_col_indices"],
        A_row_offsets=d["A_row_offsets"],
        A_col_indices=d["A_col_indices"],
        cones=d["cones"],
        settings=settings,
    )

    # Create a single problem in batched format: shape [1, nnz]
    P_batch = d["P_values"].reshape(1, -1)
    A_batch = d["A_values"].reshape(1, -1)
    q_batch = np.array([[0.0, 0.0]])  # Minimize just x'Px
    b_batch = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    solver.setup(P_batch, A_batch)
    result = solver.solve(q_batch, b_batch)
    info = solver.info

    assert result.x.shape == (batch_size, d["n"])
    # Solution should be approximately [0.5, 0.5] for minimize x^2 + y^2 s.t. x+y=1
    np.testing.assert_allclose(result.x[0], [0.5, 0.5], atol=1e-4)


def test_batch_solve_multiple_problems(batch_qp_setup, device):
    """Test solving multiple problems in parallel."""
    d = batch_qp_setup
    batch_size = 4
    n = d["n"]

    settings = moreau.Settings(device=device, batch_size=batch_size, verbose=False)
    solver = moreau.CompiledSolver(
        n=d["n"],
        m=d["m"],
        P_row_offsets=d["P_row_offsets"],
        P_col_indices=d["P_col_indices"],
        A_row_offsets=d["A_row_offsets"],
        A_col_indices=d["A_col_indices"],
        cones=d["cones"],
        settings=settings,
    )

    # Create batch of problems with different objectives
    P_batch = np.tile(d["P_values"], (batch_size, 1))
    A_batch = np.tile(d["A_values"], (batch_size, 1))
    q_batch = np.zeros((batch_size, d["n"]))
    b_batch = np.tile(np.array([1.0, 0.0, 0.0, 1.0, 1.0]), (batch_size, 1))

    # Vary the linear cost slightly across batch
    for i in range(batch_size):
        scale = 1.0 + i * 0.1
        q_batch[i] = np.array([-1.0 * scale, -0.5 * scale])

    solver.setup(P_batch, A_batch)
    result = solver.solve(q_batch, b_batch)
    info = solver.info

    assert result.x.shape == (batch_size, d["n"])
    for i in range(batch_size):
        assert not np.isnan(info.obj_val[i])
        # Each x should satisfy x1 + x2 = 1 (equality constraint)
        np.testing.assert_allclose(np.sum(result.x[i]), 1.0, atol=1e-4)


def test_compiled_solver_dimensions(device):
    """Test CompiledSolver reports correct dimensions."""
    n, m = 2, 3
    batch_size = 2

    # Create minimal CSR matrices
    P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    P_col_indices = np.array([0, 1], dtype=np.int64)

    A_row_offsets = np.array([0, 2, 3, 4], dtype=np.int64)
    A_col_indices = np.array([0, 1, 0, 1], dtype=np.int64)

    cones = moreau.Cones()
    cones.num_zero_cones = 1
    cones.num_nonneg_cones = 2

    settings = moreau.Settings(device=device, batch_size=batch_size, verbose=False)

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

    assert solver.n == n
    assert solver.m == m


def test_batch_vs_single_solve(batch_qp_setup, device):
    """Test that Solver and CompiledSolver produce same results."""
    d = batch_qp_setup
    n = d["n"]
    m = d["m"]

    # Same problem data
    q = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0, 1.0, 1.0])

    # Build sparse matrices for Solver
    P = sparse.csr_array((d["P_values"], d["P_col_indices"], d["P_row_offsets"]), shape=(n, n))
    A = sparse.csr_array((d["A_values"], d["A_col_indices"], d["A_row_offsets"]), shape=(m, n))

    # Solve with Solver (single problem API)
    settings = moreau.Settings(device=device, verbose=False)
    single_solver = moreau.Solver(P, q, A, b, d["cones"], settings)
    single_result = single_solver.solve()
    single_info = single_solver.info

    # Solve with CompiledSolver (batch of 1)
    batch_settings = moreau.Settings(device=device, batch_size=1, verbose=False)
    compiled_solver = moreau.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=d["P_row_offsets"],
        P_col_indices=d["P_col_indices"],
        A_row_offsets=d["A_row_offsets"],
        A_col_indices=d["A_col_indices"],
        cones=d["cones"],
        settings=batch_settings,
    )
    compiled_solver.setup(
        d["P_values"].reshape(1, -1),
        d["A_values"].reshape(1, -1),
    )
    batch_result = compiled_solver.solve(
        q.reshape(1, -1),
        b.reshape(1, -1),
    )
    batch_info = compiled_solver.info

    # Compare results
    np.testing.assert_allclose(single_result.x, batch_result.x[0], atol=1e-6)
    # obj_val may be scalar or array
    batch_obj = batch_info.obj_val if np.isscalar(batch_info.obj_val) else batch_info.obj_val[0]
    np.testing.assert_allclose(single_info.obj_val, batch_obj, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
