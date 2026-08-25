"""
Unit tests for Moreau Python interface
"""

import numpy as np
import pytest
from scipy import sparse

import moreau


def test_simple_qp():
    """Test a simple QP with equality and inequality constraints"""

    # Problem dimensions
    n = 2  # 2 variables
    m = 3  # 3 constraints

    # P matrix (2x2 identity)
    P = sparse.diags([1.0, 1.0], format="csr")

    # Linear cost vector
    q = np.array([2.0, 1.0])

    # A matrix (3x2)
    A = sparse.csr_array(
        [
            [1.0, 1.0],  # x1 + x2 = 1 (equality)
            [1.0, 0.0],  # x1 <= 2 (inequality)
            [0.0, 1.0],  # x2 <= 2 (inequality)
        ]
    )

    # RHS vector
    b = np.array([1.0, 2.0, 2.0])

    # Cone structure
    cones = moreau.Cones()
    cones.num_zero_cones = 1
    cones.num_nonneg_cones = 2

    # Solver settings
    settings = moreau.Settings()
    settings.max_iter = 50
    settings.verbose = False

    # Create solver with all problem data
    solver = moreau.Solver(P, q, A, b, cones, settings)

    # Solve
    result = solver.solve()
    info = solver.info

    # Check solver converged
    assert info.status in [
        moreau.SolverStatus.Solved,
        moreau.SolverStatus.AlmostSolved,
    ], f"Expected Solved or AlmostSolved, got {info.status}"

    # Expected solution from Clarabel reference solver
    expected_x = np.array([2.4308818538642494e-10, 0.9999999997569121])
    expected_s = np.array([0.0, 1.9999999997569118, 1.0000000002430876])

    # Check primal solution
    assert result.x.shape == (n,)
    np.testing.assert_allclose(result.x, expected_x, rtol=1e-6, atol=1e-6)

    # Check slack variables
    assert result.s.shape == (m,)
    np.testing.assert_allclose(result.s, expected_s, rtol=1e-6, atol=1e-6)

    # Verify constraints
    assert abs((result.x[0] + result.x[1]) - 1.0) < 1e-6  # x₁ + x₂ = 1
    assert result.x[0] <= 2.0 + 1e-6  # x₁ <= 2
    assert result.x[1] <= 2.0 + 1e-6  # x₂ <= 2


def test_solver_dimensions_with_batch_hint():
    """Test Solver dimension properties"""

    n, m = 5, 3

    # Create minimal sparse matrices
    P = sparse.diags([1.0] * n, format="csr")
    q = np.zeros(n)
    A = sparse.csr_array(np.zeros((m, n)))
    b = np.zeros(m)

    cones = moreau.Cones()
    cones.num_nonneg_cones = m

    settings = moreau.Settings(verbose=False)

    solver = moreau.Solver(P, q, A, b, cones, settings)

    assert solver.n == n
    assert solver.m == m


def test_cones():
    """Test cone structure creation"""

    cones = moreau.Cones()

    # Test default values
    assert cones.num_zero_cones == 0
    assert cones.num_nonneg_cones == 0
    assert cones.num_exp_cones == 0
    assert cones.num_so_cones == 0
    assert cones.num_power_cones == 0

    # Test setting values
    cones.num_zero_cones = 2
    cones.num_nonneg_cones = 5
    cones.so_cone_dims = [3]

    assert cones.num_zero_cones == 2
    assert cones.num_nonneg_cones == 5
    assert cones.num_so_cones == 1

    # Test total constraints
    # zero: 2, nonneg: 5, soc: 3 (3 per cone)
    assert cones.total_constraints() == 2 + 5 + 3

    # Test degree
    # zero: 0, nonneg: 5, soc: 1 (1 per cone)
    assert cones.degree() == 0 + 5 + 1


def test_settings():
    """Test settings creation and modification"""

    settings = moreau.Settings()

    # Test default values (unified Settings has verbose=False by default)
    assert settings.max_iter == 200
    assert settings.verbose == False
    assert settings.ipm_settings.tol_gap_abs == 1e-8

    # Test modification - tolerances are in IPMSettings
    settings.max_iter = 100
    settings.verbose = False
    settings.ipm_settings.tol_gap_abs = 1e-6

    assert settings.max_iter == 100
    assert settings.verbose == False
    assert settings.ipm_settings.tol_gap_abs == 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
