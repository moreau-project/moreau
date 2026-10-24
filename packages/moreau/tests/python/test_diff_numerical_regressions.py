"""Regressions for derivative fixes in the CPU KKT and exp/power cone code."""

import numpy as np
import pytest
from scipy import sparse

import moreau

clarabel = pytest.importorskip("clarabel")


def _cones(kind):
    """(moreau cones, clarabel cone) for "exp" or a power cone alpha."""
    if kind == "exp":
        return moreau.Cones(num_exp_cones=1), clarabel.ExponentialConeT()
    return moreau.Cones(power_alphas=[kind]), clarabel.PowerConeT(kind)


def _solve(P_diag, q, a, b, kind, enable_grad=False, **ipm):
    """min 1/2 x'diag(P)x + q'x  s.t.  a*x + s = b, s in K  (n = m = 3), on CPU."""
    solver = moreau.Solver(
        sparse.diags(P_diag, format="csr"),
        np.asarray(q, float),
        a * sparse.eye(3, format="csr"),
        np.asarray(b, float),
        _cones(kind)[0],
        moreau.Settings(
            device="cpu",
            solver="ipm",
            enable_grad=enable_grad,
            verbose=False,
            ipm_settings=moreau.IPMSettings(**ipm),
        ),
    )
    return solver, solver.solve()


def _clarabel(P_diag, q, a, b, kind):
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    result = clarabel.DefaultSolver(
        sparse.csc_matrix(sparse.diags(P_diag)),
        np.asarray(q, float),
        sparse.csc_matrix(a * sparse.eye(3)),
        np.asarray(b, float),
        [_cones(kind)[1]],
        settings,
    ).solve()
    assert str(result.status) == "Solved"
    return result


def _central_difference(f, eps):
    return np.array([(f(eps * e) - f(-eps * e)) / (2 * eps) for e in np.eye(3)])


def test_bug1_dead_code_build_qp_kkt_matrix():
    """
    Bug 1: Dead code in build_qp_kkt_matrix (kkt module)

    The function build_qp_kkt_matrix contained ~60 lines of dead code that was
    never executed because the function immediately called build_qp_kkt_dense_upper().

    This test verifies the fix (removing dead code) doesn't affect correctness.
    """
    n = 2
    m = 1

    # P = 2I
    P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    P_col_indices = np.array([0, 1], dtype=np.int64)
    P_values = np.array([2.0, 2.0])

    # A = [1, 1]
    A_row_offsets = np.array([0, 2], dtype=np.int64)
    A_col_indices = np.array([0, 1], dtype=np.int64)
    A_values = np.array([1.0, 1.0])

    q = np.array([0.0, 0.0])
    b = np.array([2.0])

    cones = moreau.Cones()
    cones.num_zero_cones = m

    settings = moreau.Settings()
    settings.verbose = False

    P = sparse.csr_array((P_values, P_col_indices, P_row_offsets), shape=(n, n))
    A = sparse.csr_array((A_values, A_col_indices, A_row_offsets), shape=(m, n))

    # Create bounds (unbounded)

    solver = moreau.Solver(P, q, A, b, cones, settings)
    result = solver.solve()
    info = solver.info

    # Solution should be x = [1, 1] (minimizer with constraint x1+x2=2)
    assert result is not None
    x = result.x
    assert abs(x[0] - 1.0) < 1e-4, f"x[0] = {x[0]}"
    assert abs(x[1] - 1.0) < 1e-4, f"x[1] = {x[1]}"


def test_bug2_singular_matrix_regularization():
    """Bug 2: invert_4x4 silently returned identity for a singular matrix.

    With P = 1e-6 I the optimum sits on the degenerate y = 0 face of the
    exponential cone with x ~ 1e6, where that path is reached. x[0] is only weakly
    determined, so compare the optimal value and x[2] = -q[2]/P against Clarabel;
    derivatives are covered by test_exp_power_projection_derivative.
    """
    P, q, b = [1e-6] * 3, [0.0, -1.0, 1.0], [0.0, 1.0, np.e]
    solver, sol = _solve(P, q, 1.0, b, "exp")
    assert solver.info.status == moreau.SolverStatus.Solved
    reference = _clarabel(P, q, 1.0, b, "exp")
    np.testing.assert_allclose(solver.info.obj_val, reference.obj_val, rtol=1e-7)
    np.testing.assert_allclose(sol.x[2], reference.x[2], rtol=1e-8)


# Problems from the original regressions for bugs 3-6. The cone is inactive at
# each optimum, so these check the forward solve; the projection tests below
# exercise the derivative paths those bugs lived in.
_ORIGINAL_CONE_CASES = {
    "bug3_exp_s_near_zero": ([1.0, 0.0, 0.0], [-1.0, 0.5, 1.0], "exp"),
    "bug4_exp_negative_s": ([0.0, 1.0, 0.0], [0.0, -0.5, 1.0], "exp"),
    "bug5_power_small_denominator": ([0.0, 0.0, 1.0], [1.0, 1.0, 0.9], 0.5),
    "bug6_power_z_zero": ([0.0, 0.0, 0.0], [1.0, 1.0, 0.0], 0.5),
    "bug6_power_z_tiny": ([0.0, 0.0, 0.0], [1.0, 1.0, 1e-15], 0.5),
}


@pytest.mark.parametrize("case", list(_ORIGINAL_CONE_CASES))
def test_original_cone_regressions_match_clarabel(case):
    q, b, kind = _ORIGINAL_CONE_CASES[case]
    solver, sol = _solve([1.0] * 3, q, 1.0, b, kind)
    assert solver.info.status == moreau.SolverStatus.Solved
    np.testing.assert_allclose(sol.x, _clarabel([1.0] * 3, q, 1.0, b, kind).x, atol=1e-6)


# Points c whose projection onto K makes the cone active: the general case, the
# exponential cone near and on its s = 0 face (bugs 3, 4), and the power cone with
# a small second component (bugs 5, 6).
_PROJECTION_CASES = {
    "exp_general_a": ("exp", (1.0, 1.0, 1.0)),
    "exp_general_b": ("exp", (2.0, 0.5, 0.3)),
    "exp_near_s_zero": ("exp", (0.3, -0.5, 0.4)),
    "exp_on_s_zero_face_a": ("exp", (-1.0, -0.05, 0.3)),
    "exp_on_s_zero_face_b": ("exp", (-0.4, -1.0, 0.8)),
    "power_general_a": (0.3, (0.2, 0.3, 1.0)),
    "power_general_b": (0.3, (-0.1, 0.5, 0.8)),
    "power_general_c": (0.3, (1.0, 0.1, 0.9)),
    "power_small_second": (0.3, (1.0, 0.01, 0.9)),
}


@pytest.mark.parametrize("case", list(_PROJECTION_CASES))
def test_exp_power_projection_derivative(case):
    """Backward through x = Proj_K(c) (an active exp/power cone) matches finite differences."""
    kind, c = _PROJECTION_CASES[case]
    tol = dict(diff_method="exact", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)

    def project(c, enable_grad=False):
        # min 1/2||x - c||^2  s.t.  -x + s = 0, s in K
        return _solve([1.0] * 3, -np.asarray(c, float), -1.0, np.zeros(3), kind, enable_grad, **tol)

    solver, sol = project(c, enable_grad=True)
    assert solver.info.status == moreau.SolverStatus.Solved
    assert np.linalg.norm(sol.z) > 1e-3, "cone must be active for this test to mean anything"
    # Clarabel at default tolerances is itself accurate to ~2e-5 on these points.
    reference = _clarabel([1.0] * 3, -np.asarray(c, float), -1.0, np.zeros(3), kind)
    np.testing.assert_allclose(sol.x, reference.x, atol=5e-5)

    # q = -c, so d(w.x)/dc = -dq. The forward solve is accurate to ~1e-6, so a 1e-3
    # step keeps FD noise near 1e-3; a wrong Jacobian (e.g. identity) is off by > 0.5.
    w = np.array([0.3, -1.1, 0.7])
    analytic = -solver.backward(w)["dq"].ravel()
    fd = _central_difference(lambda d: w @ project(np.add(c, d))[1].x, 1e-3)
    np.testing.assert_allclose(analytic, fd, atol=1e-2)


def test_inactive_exp_cone_gradient_matches_finite_differences():
    """With the exp cone inactive, dx/dq = -P^{-1}; backward must agree with FD."""
    P, b = [0.5] * 3, [0.0, 1.0, 3.0]
    solver, _ = _solve(P, np.zeros(3), 1.0, b, "exp", enable_grad=True)
    assert solver.info.status == moreau.SolverStatus.Solved
    analytic = solver.backward(np.ones(3))["dq"].ravel()
    fd = _central_difference(lambda d: _solve(P, d, 1.0, b, "exp")[1].x.sum(), 1e-4)
    np.testing.assert_allclose(analytic, fd, atol=1e-5)
    np.testing.assert_allclose(analytic, -1.0 / np.asarray(P), atol=1e-6)
