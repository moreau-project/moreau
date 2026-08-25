"""End-to-end infeasibility / unboundedness detection with direct-x cones.

The HSDE primal-infeasibility certificate `‖A^T z − Σ_J E_J^T z_x‖ → 0`
needs the direct-x dual `z_x` term; without it (the original behavior)
infeasible direct-x problems fell through to NumericalError or
MaxIterations on both CPU and CUDA. These tests pin the unified Python
API across both backends.
"""

import numpy as np
import pytest
from scipy import sparse

import moreau


def _settings(device, **kwargs):
    return moreau.Settings(device=device, verbose=False, **kwargs)


# ---------- nonneg direct-x ----------


def test_nonneg_direct_x_primal_infeasible(device):
    # min 0  s.t.  x = -1 (zero cone), x >= 0 (direct-x)
    P = sparse.csr_matrix((1, 1))
    q = np.array([0.0])
    A = sparse.csr_matrix([[1.0]])
    b = np.array([-1.0])
    cones = moreau.Cones(
        num_zero_cones=1,
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=[0])],
    )
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=_settings(device))
    solver.solve()
    assert solver.info.status in (
        moreau.SolverStatus.PrimalInfeasible,
        moreau.SolverStatus.AlmostPrimalInfeasible,
    ), f"expected PrimalInfeasible, got {solver.info.status}"


def test_nonneg_direct_x_dual_infeasible(device):
    # min -x  s.t.  x >= 0 (direct-x)  ->  unbounded
    P = sparse.csr_matrix((1, 1))
    q = np.array([-1.0])
    A = sparse.csr_matrix(np.zeros((0, 1)))
    b = np.array([])
    cones = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=[0])],
    )
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=_settings(device))
    solver.solve()
    assert solver.info.status in (
        moreau.SolverStatus.DualInfeasible,
        moreau.SolverStatus.AlmostDualInfeasible,
    ), f"expected DualInfeasible, got {solver.info.status}"


# ---------- SOC direct-x ----------


def test_soc_direct_x_primal_infeasible(device):
    # SOC_3 requires x[0] >= ||(x[1], x[2])||, so forcing x[0] = -1 is infeasible.
    n = 3
    P = sparse.csr_matrix((n, n))
    q = np.zeros(n)
    A = sparse.csr_matrix([[1.0, 0.0, 0.0]])
    b = np.array([-1.0])
    cones = moreau.Cones(
        num_zero_cones=1,
        x_cones=[moreau.XConeSpec(kind="soc", indices=[0, 1, 2])],
    )
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=_settings(device))
    solver.solve()
    assert solver.info.status in (
        moreau.SolverStatus.PrimalInfeasible,
        moreau.SolverStatus.AlmostPrimalInfeasible,
    ), f"expected PrimalInfeasible, got {solver.info.status}"


def test_soc_direct_x_dual_infeasible(device):
    # min -x[0]  s.t.  (x[0], x[1], x[2]) ∈ SOC_3  ->  unbounded
    n = 3
    P = sparse.csr_matrix((n, n))
    q = np.array([-1.0, 0.0, 0.0])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind="soc", indices=[0, 1, 2])],
    )
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=_settings(device))
    solver.solve()
    assert solver.info.status in (
        moreau.SolverStatus.DualInfeasible,
        moreau.SolverStatus.AlmostDualInfeasible,
    ), f"expected DualInfeasible, got {solver.info.status}"


# ---------- PSD direct-x ----------


def test_psd_direct_x_primal_infeasible(device):
    # 2×2 PSD matrix has nonneg diagonal, so forcing svec(X)[0] = -1 is infeasible.
    n = 3  # svec length for 2x2
    P = sparse.csr_matrix((n, n))
    q = np.zeros(n)
    A = sparse.csr_matrix([[1.0, 0.0, 0.0]])
    b = np.array([-1.0])
    cones = moreau.Cones(
        num_zero_cones=1,
        x_cones=[moreau.XConeSpec(kind="psd_triangle", indices=[0, 1, 2], psd_k=2)],
    )
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=_settings(device))
    solver.solve()
    assert solver.info.status in (
        moreau.SolverStatus.PrimalInfeasible,
        moreau.SolverStatus.AlmostPrimalInfeasible,
    ), f"expected PrimalInfeasible, got {solver.info.status}"


# ---------- Mixed slack + direct-x ----------


def test_mixed_direct_x_primal_infeasible(device):
    # x ∈ R+² (direct-x), slack equality x[0] + x[1] = -2  ->  infeasible
    P = sparse.csr_matrix((2, 2))
    q = np.array([1.0, 1.0])
    A = sparse.csr_matrix([[1.0, 1.0]])
    b = np.array([-2.0])
    cones = moreau.Cones(
        num_zero_cones=1,
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=[0, 1])],
    )
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=_settings(device))
    solver.solve()
    assert solver.info.status in (
        moreau.SolverStatus.PrimalInfeasible,
        moreau.SolverStatus.AlmostPrimalInfeasible,
    ), f"expected PrimalInfeasible, got {solver.info.status}"
