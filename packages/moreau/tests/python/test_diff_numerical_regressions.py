"""
Tests that expose 3 bugs found in packages/moreau/src/solver/implementations/default/diff.

Bug 1: Dead code in kkt.rs::build_qp_kkt_matrix (lines 438-492)
       The function constructs a sparse KKT matrix but then ignores it and
       calls build_qp_kkt_dense_upper() at line 496, making all prior work dead code.

Bug 2: Silent failure in cones.rs::invert_4x4 (lines 798-804)
       When the 4x4 matrix is singular during exponential cone Jacobian computation,
       the function silently returns identity instead of raising an error.
       This can cause completely wrong gradients without any warning.

Bug 3: Incorrect formula in cones.rs::derivative_exp_cone (line 666)
       When s == 0 in the exponential cone boundary case, the code uses
       r.abs() as a substitute for s, which is mathematically incorrect.
       The correct behavior should handle this edge case specially or use a
       proper limit.
"""

import numpy as np
import pytest
from scipy import sparse

clarabel = pytest.importorskip("clarabel")

# These regressions exercise CPU derivative code; pin the device so results do
# not depend on whether a GPU happens to be present.
_DEVICE = "cpu"


def _solve_three_var(P_diag, q, b, cones, **ipm):
    """min 1/2 x'diag(P)x + q'x  s.t.  x + s = b, s in K  (A = I, n = m = 3)."""
    import moreau

    solver = moreau.Solver(
        sparse.diags(P_diag, format="csr"),
        np.asarray(q, float),
        sparse.eye(3, format="csr"),
        np.asarray(b, float),
        cones,
        moreau.Settings(
            device=_DEVICE, solver="ipm", verbose=False, ipm_settings=moreau.IPMSettings(**ipm)
        ),
    )
    return solver, solver.solve()


def _clarabel_three_var(P_diag, q, b, cone):
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    result = clarabel.DefaultSolver(
        sparse.csc_matrix(sparse.diags(P_diag)),
        np.asarray(q, float),
        sparse.csc_matrix(sparse.eye(3)),
        np.asarray(b, float),
        [cone],
        settings,
    ).solve()
    assert str(result.status) == "Solved"
    return result


def test_bug1_dead_code_build_qp_kkt_matrix():
    """
    Bug 1: Dead code in build_qp_kkt_matrix (kkt module)

    The function build_qp_kkt_matrix contained ~60 lines of dead code that was
    never executed because the function immediately called build_qp_kkt_dense_upper().

    This test verifies the fix (removing dead code) doesn't affect correctness.
    """
    import moreau

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

    # Convert to sparse matrices
    from scipy import sparse

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
    """
    Bug 2: Silent failure in invert_4x4 (cones module)

    Previously, when the 4x4 matrix was singular, the code silently returned
    identity. With P = 1e-6 I the optimum sits on the degenerate y = 0 face of
    the exponential cone with x ~ 1e6, which is where that path is reached.

    The problem is badly conditioned: x[0] carries almost no weight in the
    objective and is only weakly determined, so the check compares the optimal
    value and the well-determined x[2] = -q[2]/P against Clarabel. Finite
    differences are not reliable at this scale, so derivative correctness is
    covered by test_exp_power_projection_derivative below.
    """
    import moreau

    P, q, b = [1e-6] * 3, [0.0, -1.0, 1.0], [0.0, 1.0, np.e]
    solver, sol = _solve_three_var(P, q, b, moreau.Cones(num_exp_cones=1))
    assert solver.info.status == moreau.SolverStatus.Solved
    reference = _clarabel_three_var(P, q, b, clarabel.ExponentialConeT())
    np.testing.assert_allclose(solver.info.obj_val, reference.obj_val, rtol=1e-7)
    np.testing.assert_allclose(sol.x[2], reference.x[2], rtol=1e-8)


# Problems from the original regressions for bugs 3-6. At each optimum the cone
# constraint is inactive (z = 0), so they check forward correctness only. The
# derivative paths those bugs lived in are exercised by the projection tests.
_ORIGINAL_CONE_CASES = {
    "bug3_exp_s_near_zero": ([1.0] * 3, [1.0, 0.0, 0.0], [-1.0, 0.5, 1.0], "exp"),
    "bug4_exp_negative_s": ([1.0] * 3, [0.0, 1.0, 0.0], [0.0, -0.5, 1.0], "exp"),
    "bug5_power_small_denominator": ([1.0] * 3, [0.0, 0.0, 1.0], [1.0, 1.0, 0.9], "power"),
    "bug6_power_z_zero": ([1.0] * 3, [0.0, 0.0, 0.0], [1.0, 1.0, 0.0], "power"),
    "bug6_power_z_tiny": ([1.0] * 3, [0.0, 0.0, 0.0], [1.0, 1.0, 1e-15], "power"),
}


@pytest.mark.parametrize("case", list(_ORIGINAL_CONE_CASES), ids=list(_ORIGINAL_CONE_CASES))
def test_original_cone_regressions_match_clarabel(case):
    import moreau

    P, q, b, kind = _ORIGINAL_CONE_CASES[case]
    if kind == "exp":
        cones, cone = moreau.Cones(num_exp_cones=1), clarabel.ExponentialConeT()
    else:
        cones, cone = moreau.Cones(power_alphas=[0.5]), clarabel.PowerConeT(0.5)
    solver, sol = _solve_three_var(P, q, b, cones)
    assert solver.info.status == moreau.SolverStatus.Solved
    np.testing.assert_allclose(sol.x, _clarabel_three_var(P, q, b, cone).x, atol=1e-6)


def _projection_solver(c, kind, enable_grad=False):
    """x = Proj_K(c):  min 1/2||x - c||^2  s.t.  -x + s = 0, s in K."""
    import moreau

    cones = moreau.Cones(num_exp_cones=1) if kind == "exp" else moreau.Cones(power_alphas=[0.3])
    tol = 1e-10
    solver = moreau.Solver(
        sparse.eye(3, format="csr"),
        -np.asarray(c, float),
        -sparse.eye(3, format="csr"),
        np.zeros(3),
        cones,
        moreau.Settings(
            device=_DEVICE,
            solver="ipm",
            enable_grad=enable_grad,
            verbose=False,
            ipm_settings=moreau.IPMSettings(
                diff_method="exact", tol_gap_abs=tol, tol_gap_rel=tol, tol_feas=tol
            ),
        ),
    )
    return solver, solver.solve()


# Points whose projection makes the cone constraint active, covering the general
# case, the exponential cone near and exactly on its s = 0 face (bugs 3 and 4),
# and the power cone including a small second component (bugs 5 and 6).
_PROJECTION_CASES = {
    "exp_general_a": ("exp", (1.0, 1.0, 1.0)),
    "exp_general_b": ("exp", (2.0, 0.5, 0.3)),
    "exp_near_s_zero": ("exp", (0.3, -0.5, 0.4)),
    "exp_on_s_zero_face_a": ("exp", (-1.0, -0.05, 0.3)),
    "exp_on_s_zero_face_b": ("exp", (-0.4, -1.0, 0.8)),
    "power_general_a": ("power", (0.2, 0.3, 1.0)),
    "power_general_b": ("power", (-0.1, 0.5, 0.8)),
    "power_general_c": ("power", (1.0, 0.1, 0.9)),
    "power_small_second": ("power", (1.0, 0.01, 0.9)),
}


@pytest.mark.parametrize("case", list(_PROJECTION_CASES), ids=list(_PROJECTION_CASES))
def test_exp_power_projection_derivative(case):
    """Backward through an active exp/power cone matches finite differences.

    The original regressions for the exp/power derivative fixes never called
    backward. Here the cone is active at the optimum, so the adjoint runs through
    the cone projection Jacobian, including the exponential cone's s = 0 face.
    """
    import moreau

    kind, c = _PROJECTION_CASES[case]
    solver, sol = _projection_solver(c, kind, enable_grad=True)
    assert solver.info.status == moreau.SolverStatus.Solved
    assert np.linalg.norm(sol.z) > 1e-3, "cone must be active for this test to mean anything"

    cone = clarabel.ExponentialConeT() if kind == "exp" else clarabel.PowerConeT(0.3)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    reference = clarabel.DefaultSolver(
        sparse.csc_matrix(sparse.eye(3)),
        -np.asarray(c, float),
        sparse.csc_matrix(-sparse.eye(3)),
        np.zeros(3),
        [cone],
        settings,
    ).solve()
    # Clarabel at default tolerances is itself accurate to ~2e-5 on these points.
    np.testing.assert_allclose(sol.x, reference.x, atol=5e-5)

    # q = -c, so d(w.x)/dc = -dq.
    w = np.array([0.3, -1.1, 0.7])
    analytic = -solver.backward(w)["dq"].ravel()
    # The forward solve is accurate to ~1e-6, so a 1e-3 step keeps finite-difference
    # noise near 1e-3. A wrong Jacobian (e.g. the identity) is off by more than 0.5.
    eps = 1e-3
    finite_difference = np.array(
        [
            (
                w @ _projection_solver(np.add(c, eps * e), kind)[1].x
                - w @ _projection_solver(np.subtract(c, eps * e), kind)[1].x
            )
            / (2 * eps)
            for e in np.eye(3)
        ]
    )
    np.testing.assert_allclose(analytic, finite_difference, atol=1e-2)


def test_inactive_exp_cone_gradient_matches_finite_differences():
    """With the exp cone inactive, dx/dq = -P^{-1}; backward must agree with FD."""
    import moreau

    P, b = [0.5] * 3, [0.0, 1.0, 3.0]
    cones = moreau.Cones(num_exp_cones=1)
    solver = moreau.Solver(
        sparse.diags(P, format="csr"),
        np.zeros(3),
        sparse.eye(3, format="csr"),
        np.asarray(b),
        cones,
        moreau.Settings(device=_DEVICE, solver="ipm", enable_grad=True, verbose=False),
    )
    solver.solve()
    assert solver.info.status == moreau.SolverStatus.Solved
    analytic = solver.backward(np.ones(3))["dq"].ravel()
    eps = 1e-4
    finite_difference = np.array(
        [
            (
                _solve_three_var(P, eps * e, b, cones)[1].x.sum()
                - _solve_three_var(P, -eps * e, b, cones)[1].x.sum()
            )
            / (2 * eps)
            for e in np.eye(3)
        ]
    )
    np.testing.assert_allclose(analytic, finite_difference, atol=1e-5)
    np.testing.assert_allclose(analytic, -1.0 / np.asarray(P), atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
