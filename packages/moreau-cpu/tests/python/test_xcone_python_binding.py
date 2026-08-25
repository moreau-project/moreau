"""Python pyo3 binding tests for direct-x cones.

These cover the wiring from Python (scipy CSR matrices + XConeT specs)
through `DefaultSolver.new_with_xcones` down to the Rust forward pass.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

import moreau_cpu._cpu_solver as cpu


def _direct_x_solve_nonneg(P, q, n):
    """Solve `min 0.5 x'Px + q'x s.t. x >= 0` via direct-x on all of x."""
    A = sparse.csr_matrix(np.zeros((0, n)))
    cones = []
    x_cones = [cpu.NonnegativeXConeT(list(range(n)))]

    settings = cpu.DefaultSettings()
    settings.ipm.presolve_enable = False
    solver = cpu.DefaultSolver.new_with_xcones(P, list(q), A, [], cones, x_cones, settings)
    solver.solve()
    return solver.get_solution()


def _slack_solve_nonneg(P, q, n):
    """Slack-form reference: `-x + s = 0, s ∈ R+^n`."""
    A = sparse.csr_matrix(-np.eye(n))
    cones = [cpu.NonnegativeConeT(n)]

    settings = cpu.DefaultSettings()
    settings.ipm.presolve_enable = False
    solver = cpu.DefaultSolver(P, list(q), A, [0.0] * n, cones, settings)
    solver.solve()
    return solver.get_solution()


def test_xcone_pyclass_repr():
    assert repr(cpu.NonnegativeXConeT([0, 2])) == "NonnegativeXConeT([0, 2])"
    assert repr(cpu.SecondOrderXConeT([1, 3, 5])) == "SecondOrderXConeT([1, 3, 5])"


def test_nonneg_direct_x_active_constraint():
    # min 0.5 x^2 + 2x s.t. x >= 0 -> x = 0
    P = sparse.csr_matrix([[1.0]])
    q = [2.0]
    sol = _direct_x_solve_nonneg(P, q, 1)
    assert str(sol.status).endswith("Solved")
    assert abs(sol.x[0]) < 1e-6


def test_nonneg_direct_x_matches_slack_3d():
    # 3-dim QP with P = 2I, q = [-1, 2, -0.5].
    P = sparse.csr_matrix(2 * np.eye(3))
    q = [-1.0, 2.0, -0.5]

    slack = _slack_solve_nonneg(P, q, 3)
    direct = _direct_x_solve_nonneg(P, q, 3)

    assert str(slack.status).endswith("Solved")
    assert str(direct.status).endswith("Solved")
    np.testing.assert_allclose(slack.x, direct.x, atol=1e-6)


def test_soc_direct_x_matches_slack():
    # min 0.5 x'x + [1, 2, 0]'x s.t. x ∈ SOC_3. Constraint is active.
    n = 3
    P = sparse.csr_matrix(np.eye(n))
    q = [1.0, 2.0, 0.0]

    # Slack form
    A_slack = sparse.csr_matrix(-np.eye(n))
    b_slack = [0.0] * n
    cones_slack = [cpu.SecondOrderConeT(n)]
    settings = cpu.DefaultSettings()
    settings.ipm.presolve_enable = False
    slack_solver = cpu.DefaultSolver(P, q, A_slack, b_slack, cones_slack, settings)
    slack_solver.solve()
    slack_sol = slack_solver.get_solution()

    # Direct-x form
    A_dx = sparse.csr_matrix(np.zeros((0, n)))
    cones_dx = []
    x_cones = [cpu.SecondOrderXConeT(list(range(n)))]
    settings2 = cpu.DefaultSettings()
    settings2.ipm.presolve_enable = False
    dx_solver = cpu.DefaultSolver.new_with_xcones(P, q, A_dx, [], cones_dx, x_cones, settings2)
    dx_solver.solve()
    dx_sol = dx_solver.get_solution()

    assert str(slack_sol.status).endswith("Solved")
    assert str(dx_sol.status).endswith("Solved")
    np.testing.assert_allclose(slack_sol.x, dx_sol.x, atol=1e-6)
