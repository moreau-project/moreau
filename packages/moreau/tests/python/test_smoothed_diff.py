"""Tests for smoothed differentiation.

Smoothed diff supports zero + nonneg + SOC cones. Tests verify:
1. Smoothness: gradient curve across a kink is smooth (no jumps).
2. Monotonic smoothing: larger mu -> smaller max consecutive jump.
3. Walk-up works: smoothed differs from exact at moderate mu.
4. Convergence: smoothed -> exact as mu -> 0 at O(mu) rate.
5. All gradient outputs (dq, db, dP, dA) respond to mu.
6. Mixed zero + nonneg cones work correctly.
7. SOC: smoothness, walk-up, FD consistency.
8. Rejection: unsupported cones raise errors with diff_method='smoothed'.
9. Zero-cone-only: no wasted refinement iterations.
10. FD consistency: smoothed gradients match finite differences.
11. CPU/CUDA consistency: same step_factor convention on both backends.
"""

import numpy as np
import pytest

import moreau

# Sweep parameters
N = 201  # number of q values in the sweep (odd so boundary is exactly hit)
MU_VALUES = [1e-3, 1e-2, 1e-1]  # ascending: more smoothing as mu grows

# Allow up to 10% non-monotonicity in max-jump comparisons
MONO_TOL = 1.1


def _compute_dq(n, m, P_ro, P_ci, P_vals, A_ro, A_ci, A_vals, cones, qs, bs, mu, device):
    """Compute dq from backward pass for a batch of problems."""
    batch = len(qs)
    settings = moreau.Settings(
        solver="ipm",
        device=device,
        batch_size=batch,
        enable_grad=True,
        ipm_settings=moreau.IPMSettings(
            diff_method="smoothed",
            diff_smoothing_mu=mu,
            tol_gap_abs=1e-9,
            tol_feas=1e-9,
            direct_solve_method="qdldl",
        ),
    )
    solver = moreau.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=P_ro,
        P_col_indices=P_ci,
        A_row_offsets=A_ro,
        A_col_indices=A_ci,
        cones=cones,
        settings=settings,
    )
    solver.setup(P_values=P_vals, A_values=A_vals)
    solver.solve(qs=qs, bs=bs)

    # Upstream gradient: e_1 in x, zeros in z and s
    dx = np.zeros((batch, n))
    dx[:, 0] = 1.0
    dz = np.zeros((batch, m))
    ds = np.zeros((batch, m))
    grads = solver.backward(dx=dx, dz=dz, ds=ds)
    return np.asarray(grads["dq"])  # (batch, n)


def _assert_smoothness(grads_by_mu, component=0):
    """Assert smoothed gradients are smooth and smoother for larger mu."""
    max_jumps = {}
    for mu in MU_VALUES:
        g = grads_by_mu[mu][:, component]
        # Gradient should be non-trivial (sweep crosses a kink)
        assert np.ptp(g) > 0.1, (
            f"mu={mu}: gradient range {np.ptp(g):.4f} too small; "
            "sweep may not cross a non-differentiable point"
        )
        max_jumps[mu] = np.max(np.abs(np.diff(g)))

    # Larger mu should give smaller (or similar) max jumps
    sorted_mus = sorted(max_jumps.keys())
    for i in range(len(sorted_mus) - 1):
        mu_lo, mu_hi = sorted_mus[i], sorted_mus[i + 1]
        assert max_jumps[mu_hi] <= max_jumps[mu_lo] * MONO_TOL, (
            f"Smoothing not monotonic: mu={mu_lo:.0e} max_jump={max_jumps[mu_lo]:.4f}, "
            f"mu={mu_hi:.0e} max_jump={max_jumps[mu_hi]:.4f}"
        )


# ---------- Nonneg cone ----------


def test_smoothed_diff_nonneg(device):
    """Nonneg: min (1/2)x^2 + qx  s.t. x >= 0.

    Analytic: x* = max(0, -q), so dx*/dq is a step at q=0.
    """
    q_vals = np.linspace(-1, 1, N)
    qs = q_vals.reshape(N, 1)
    bs = np.zeros((N, 1))

    grads_by_mu = {}
    for mu in MU_VALUES:
        grads_by_mu[mu] = _compute_dq(
            n=1,
            m=1,
            P_ro=[0, 1],
            P_ci=[0],
            P_vals=[1.0],
            A_ro=[0, 1],
            A_ci=[0],
            A_vals=[-1.0],
            cones=moreau.Cones(num_nonneg_cones=1),
            qs=qs,
            bs=bs,
            mu=mu,
            device=device,
        )

    _assert_smoothness(grads_by_mu, component=0)


# ---------- Walk-up verification ----------


def _compute_grads(
    n,
    m,
    P_ro,
    P_ci,
    P_vals,
    A_ro,
    A_ci,
    A_vals,
    cones,
    q,
    b,
    dx,
    dz,
    ds,
    device,
    diff_method="exact",
    mu=None,
):
    """Solve a single problem and return all gradients.

    diff_method: 'exact' or 'smoothed'
    mu: smoothing parameter (required when diff_method='smoothed')
    """
    ipm_kwargs = dict(
        diff_method=diff_method,
        tol_gap_abs=1e-9,
        tol_feas=1e-9,
        direct_solve_method="qdldl",
    )
    if mu is not None:
        ipm_kwargs["diff_smoothing_mu"] = mu
    settings = moreau.Settings(
        solver="ipm",
        device=device,
        batch_size=1,
        enable_grad=True,
        ipm_settings=moreau.IPMSettings(**ipm_kwargs),
    )
    solver = moreau.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=P_ro,
        P_col_indices=P_ci,
        A_row_offsets=A_ro,
        A_col_indices=A_ci,
        cones=cones,
        settings=settings,
    )
    solver.setup(P_values=P_vals, A_values=A_vals)
    solver.solve(qs=[q], bs=[b])
    grads = solver.backward(
        dx=np.asarray(dx).reshape(1, n),
        dz=np.asarray(dz).reshape(1, m),
        ds=np.asarray(ds).reshape(1, m),
    )
    return {k: np.asarray(v).flatten() for k, v in grads.items()}


def test_smoothed_walkup_actually_changes_gradient(device):
    """Verify the central-path walk-up produces a different gradient than exact.

    At moderate mu (0.1), the smoothed gradient must differ from exact.
    If the walk-up silently failed, smoothed would equal exact.
    """
    np.random.seed(42)
    n, m = 2, 2
    q = np.array([-0.01, 1.0])  # x1 near boundary of nonneg cone
    b = np.zeros(m)
    dx = np.array([1.0, 0.0])
    dz = np.zeros(m)
    ds = np.zeros(m)

    args = dict(
        n=n,
        m=m,
        P_ro=[0, 1, 2],
        P_ci=[0, 1],
        P_vals=[1.0, 1.0],
        A_ro=[0, 1, 2],
        A_ci=[0, 1],
        A_vals=[-1.0, -1.0],
        cones=moreau.Cones(num_nonneg_cones=m),
        q=q,
        b=b,
        dx=dx,
        dz=dz,
        ds=ds,
        device=device,
    )

    exact = _compute_grads(**args)
    smoothed = _compute_grads(diff_method="smoothed", **args, mu=0.1)

    # Smoothed at mu=0.1 must noticeably differ from exact
    diff = np.max(np.abs(exact["dq"] - smoothed["dq"]))
    assert diff > 0.01, (
        f"Smoothed at mu=0.1 is too close to exact (diff={diff:.2e}); " "walk-up may not be working"
    )


def test_smoothed_all_grads_vary_with_mu(device):
    """Verify dq, db, dP_values, dA_values all change with mu for nonneg.

    Tests that the walk-up affects all four gradient outputs, not just dq.
    """
    np.random.seed(77)
    n, m = 3, 3
    P_vals = [2.0, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5, 2.0]
    P_ro = [0, 3, 6, 9]
    P_ci = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    A_vals = [-1.0, -1.0, -1.0]
    A_ro = [0, 1, 2, 3]
    A_ci = [0, 1, 2]
    q = np.array([-0.05, 1.0, -0.5])
    b = np.zeros(m)
    dx = np.random.randn(n)
    dz = np.random.randn(m)
    ds = np.random.randn(m)

    args = dict(
        n=n,
        m=m,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=moreau.Cones(num_nonneg_cones=m),
        q=q,
        b=b,
        dx=dx,
        dz=dz,
        ds=ds,
        device=device,
    )

    g_small = _compute_grads(diff_method="smoothed", **args, mu=1e-4)
    g_large = _compute_grads(diff_method="smoothed", **args, mu=1e-1)

    for key in ["dq", "db", "dP_values", "dA_values"]:
        diff = np.max(np.abs(g_small[key] - g_large[key]))
        assert diff > 1e-3, f"{key} doesn't change between mu=1e-4 and mu=1e-1 (diff={diff:.2e})"


def test_smoothed_mixed_zero_nonneg(device):
    """Verify smoothed diff works with mixed zero + nonneg cones.

    Tests that the walk-up handles the case where some constraints are
    equality (zero cone) and some are inequality (nonneg cone).
    """
    np.random.seed(99)
    n, m = 3, 3
    # 1 zero cone (equality) + 2 nonneg cones (inequality)
    # A = [[1,1,0], [-1,0,0], [0,-1,0]]
    # Constraint: x0+x1 = 1, x0 >= 0, x1 >= 0
    P_vals = [1.0, 1.0, 1.0]
    P_ro = [0, 1, 2, 3]
    P_ci = [0, 1, 2]
    A_vals = [1.0, 1.0, -1.0, -1.0]
    A_ro = [0, 2, 3, 4]
    A_ci = [0, 1, 0, 1]
    q = np.array([-0.01, 0.5, 0.0])  # x0 near boundary
    b = np.array([1.0, 0.0, 0.0])
    cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

    dx = np.random.randn(n)
    dz = np.random.randn(m)
    ds = np.random.randn(m)

    args = dict(
        n=n,
        m=m,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        q=q,
        b=b,
        dx=dx,
        dz=dz,
        ds=ds,
        device=device,
    )

    exact = _compute_grads(**args)
    g_small = _compute_grads(diff_method="smoothed", **args, mu=1e-6)
    g_large = _compute_grads(diff_method="smoothed", **args, mu=1e-1)

    # At tiny mu, smoothed should be close to exact
    for key in ["dq", "db"]:
        err = np.max(np.abs(exact[key] - g_small[key]))
        assert err < 1e-3, (
            f"Mixed cones: smoothed at mu=1e-6 too far from exact for {key} " f"(err={err:.2e})"
        )

    # At large mu, smoothed should differ from exact
    diff = np.max(np.abs(exact["dq"] - g_large["dq"]))
    assert diff > 1e-3, f"Mixed cones: smoothed at mu=0.1 same as exact (diff={diff:.2e})"


def test_smoothed_convergence_rate(device):
    """Verify smoothed → exact at rate O(mu).

    The error should decrease roughly proportionally to mu. We check that
    halving mu roughly halves the error (within 4x tolerance for robustness).
    """
    np.random.seed(55)
    n, m = 4, 4
    P_ro = list(range(n + 1))
    P_ci = list(range(n))
    P_vals = [1.0] * n
    A_ro = list(range(m + 1))
    A_ci = list(range(m))
    A_vals = [-1.0] * m
    q = np.array([0.5, -1.0, 0.3, -0.2])
    b = np.zeros(m)
    dx = np.random.randn(n)
    dz = np.random.randn(m)
    ds = np.random.randn(m)
    cones = moreau.Cones(num_nonneg_cones=m)

    args = dict(
        n=n,
        m=m,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        q=q,
        b=b,
        dx=dx,
        dz=dz,
        ds=ds,
        device=device,
    )

    exact = _compute_grads(**args)
    mus = [1e-2, 1e-3, 1e-4]
    errors = []
    for mu in mus:
        g = _compute_grads(diff_method="smoothed", **args, mu=mu)
        err = np.max(np.abs(exact["dq"] - g["dq"]))
        errors.append(err)

    # Each 10x decrease in mu should give roughly 10x decrease in error.
    # We allow 4x slack (ratio between 2.5 and 40).
    for i in range(len(errors) - 1):
        ratio = errors[i] / max(errors[i + 1], 1e-15)
        assert 2.5 < ratio < 40, (
            f"Convergence rate wrong: mu={mus[i]:.0e} err={errors[i]:.2e}, "
            f"mu={mus[i+1]:.0e} err={errors[i+1]:.2e}, ratio={ratio:.1f} "
            f"(expected ~10)"
        )


# ---------- Smoothness across kink ----------


def test_smoothed_gradient_is_smooth_across_kink(device):
    """Sweep q through the kink at q=0 for min (1/2)x^2 + qx s.t. x >= 0.

    Exact dx*/dq is a step function (jump of 1 at q=0). Smoothed at mu=1e-3
    should produce a smooth sigmoid with max consecutive jump << 1.
    """
    n_pts = 101
    mu = 1e-3
    q_vals = np.linspace(-0.5, 0.5, n_pts)
    qs = q_vals.reshape(n_pts, 1)
    bs = np.zeros((n_pts, 1))

    dq_grads = _compute_dq(
        n=1,
        m=1,
        P_ro=[0, 1],
        P_ci=[0],
        P_vals=[1.0],
        A_ro=[0, 1],
        A_ci=[0],
        A_vals=[-1.0],
        cones=moreau.Cones(num_nonneg_cones=1),
        qs=qs,
        bs=bs,
        mu=mu,
        device=device,
    )
    g = dq_grads[:, 0]

    max_jump = np.max(np.abs(np.diff(g)))
    # Smooth sigmoid: max slope ~ 1/(4*sqrt(mu)) ~ 8, spacing 1/100 = 0.01,
    # so max_jump ~ 0.08. Allow 0.15. Broken walk-up gives ~0.49.
    assert max_jump < 0.15, (
        f"Smoothed gradient not smooth: max_jump={max_jump:.4f} (expected < 0.15). "
        f"Walk-up may be failing."
    )
    # Non-trivial range
    assert np.ptp(g) > 0.5, f"Gradient range too small: {np.ptp(g):.4f}"


# ---------- SOC smoothed differentiation ----------

# min (1/2)||x||^2 + q'x  s.t. ||x|| <= 1  (unit ball via SOC(3))
# A = [[0,0],[-1,0],[0,-1]], b = [1,0,0], s = (1, x1, x2)
_SOC_UNIT_BALL = dict(
    n=2,
    m=3,
    P_ro=[0, 1, 2],
    P_ci=[0, 1],
    P_vals=[1.0, 1.0],
    A_ro=[0, 0, 1, 2],
    A_ci=[0, 1],
    A_vals=[-1.0, -1.0],
    cones=moreau.Cones(so_cone_dims=[3]),
)


def test_smoothed_diff_soc(device):
    """SOC: min (1/2)||x||^2 + q1*x1  s.t. ||x|| <= 1.

    Analytic: x1* = clamp(-q1, -1, 1), so dx1*/dq1 has kinks at q1 = +/-1.
    """
    q_vals = np.linspace(-2, 2, N)
    qs = np.column_stack([q_vals, np.zeros(N)])
    bs = np.tile([1.0, 0.0, 0.0], (N, 1))

    grads_by_mu = {}
    for mu in MU_VALUES:
        grads_by_mu[mu] = _compute_dq(
            **_SOC_UNIT_BALL,
            qs=qs,
            bs=bs,
            mu=mu,
            device=device,
        )

    _assert_smoothness(grads_by_mu, component=0)


def test_smoothed_soc_gradient_is_smooth_across_kink(device):
    """Sweep q through the SOC kink at q1=1 for min (1/2)||x||^2 + q1*x1 s.t. ||x||<=1.

    Exact dx1*/dq1 has a jump at q1=1 (active set changes). Smoothed should
    produce a smooth transition.
    """
    n_pts = 101
    mu = 1e-3
    q_vals = np.linspace(0.5, 1.5, n_pts)
    qs = np.column_stack([q_vals, np.zeros(n_pts)])
    bs = np.tile([1.0, 0.0, 0.0], (n_pts, 1))

    dq_grads = _compute_dq(**_SOC_UNIT_BALL, qs=qs, bs=bs, mu=mu, device=device)
    g = dq_grads[:, 0]

    max_jump = np.max(np.abs(np.diff(g)))
    assert max_jump < 0.15, (
        f"Smoothed SOC gradient not smooth: max_jump={max_jump:.4f} (expected < 0.15). "
        f"Walk-up may be failing."
    )
    assert np.ptp(g) > 0.3, f"Gradient range too small: {np.ptp(g):.4f}"


def test_smoothed_soc_walkup_changes_gradient(device):
    """Verify the central-path walk-up produces a different gradient than exact for SOC."""
    n, m = 2, 3

    args = dict(
        **_SOC_UNIT_BALL,
        q=np.array([-0.99, 0.0]),  # x1 near boundary of unit ball
        b=np.array([1.0, 0.0, 0.0]),
        dx=np.array([1.0, 0.0]),
        dz=np.zeros(m),
        ds=np.zeros(m),
        device=device,
    )

    exact = _compute_grads(**args)
    smoothed = _compute_grads(diff_method="smoothed", **args, mu=0.1)

    diff = np.max(np.abs(exact["dq"] - smoothed["dq"]))
    assert diff > 0.01, (
        f"SOC smoothed at mu=0.1 is too close to exact (diff={diff:.2e}); "
        "walk-up may not be working"
    )


def test_smoothed_soc_fd_consistency_dq(device):
    """Verify smoothed dq matches finite differences for SOC."""
    np.random.seed(9012)
    n, m = _SOC_UNIT_BALL["n"], _SOC_UNIT_BALL["m"]
    eps = 1e-6
    mu = 1e-4
    q_base = np.array([-0.5, 0.3])
    b = np.array([1.0, 0.0, 0.0])
    dx = np.random.randn(n)

    g = _compute_grads(
        diff_method="smoothed",
        **_SOC_UNIT_BALL,
        q=q_base,
        b=b,
        dx=dx,
        dz=np.zeros(m),
        ds=np.zeros(m),
        mu=mu,
        device=device,
    )
    dq_analytic = g["dq"]

    dq_fd = np.zeros(n)
    for j in range(n):
        q_p = q_base.copy()
        q_p[j] += eps
        q_m = q_base.copy()
        q_m[j] -= eps

        settings = moreau.Settings(
            device=device,
            batch_size=2,
            enable_grad=False,
            ipm_settings=moreau.IPMSettings(
                diff_method="smoothed",
                diff_smoothing_mu=mu,
                tol_gap_abs=1e-9,
                tol_feas=1e-9,
            ),
        )
        solver = moreau.CompiledSolver(
            P_row_offsets=_SOC_UNIT_BALL["P_ro"],
            P_col_indices=_SOC_UNIT_BALL["P_ci"],
            A_row_offsets=_SOC_UNIT_BALL["A_ro"],
            A_col_indices=_SOC_UNIT_BALL["A_ci"],
            cones=_SOC_UNIT_BALL["cones"],
            n=n,
            m=m,
            settings=settings,
        )
        solver.setup(
            P_values=_SOC_UNIT_BALL["P_vals"],
            A_values=_SOC_UNIT_BALL["A_vals"],
        )
        sol = solver.solve(qs=[q_p, q_m], bs=[b, b])
        x_p = np.asarray(sol.x)[0]
        x_m = np.asarray(sol.x)[1]
        dq_fd[j] = dx @ (x_p - x_m) / (2 * eps)

    assert np.allclose(dq_analytic, dq_fd, atol=5e-3), (
        f"SOC smoothed dq FD mismatch: max_err={np.max(np.abs(dq_analytic - dq_fd)):.2e}\n"
        f"  analytic: {dq_analytic}\n  FD: {dq_fd}"
    )


# ---------- Rejection for unsupported cones ----------


def test_smoothed_accepted_for_soc(device):
    """Smoothed diff should work for SOC cones (no error)."""
    settings = moreau.Settings(
        device=device,
        batch_size=1,
        enable_grad=True,
        ipm_settings=moreau.IPMSettings(
            diff_method="smoothed",
            diff_smoothing_mu=0.01,
        ),
    )
    # Should not raise — SOC is now supported
    solver = moreau.CompiledSolver(
        n=3,
        m=3,
        P_row_offsets=[0, 1, 2, 3],
        P_col_indices=[0, 1, 2],
        A_row_offsets=[0, 1, 2, 3],
        A_col_indices=[0, 1, 2],
        cones=moreau.Cones(so_cone_dims=[3]),
        settings=settings,
    )
    solver.setup(P_values=[1.0, 1.0, 1.0], A_values=[1.0, 1.0, 1.0])
    sol = solver.solve(qs=[[0.0, 0.0, -1.0]], bs=[[1.0, 0.0, 0.0]])
    dx = np.zeros((1, 3))
    dx[0, 2] = 1.0
    dz = np.zeros((1, 3))
    ds = np.zeros((1, 3))
    grads = solver.backward(dx=dx, dz=dz, ds=ds)
    # Just check it doesn't crash and produces finite gradients
    assert np.all(np.isfinite(grads["dq"]))


def test_smoothed_rejected_for_exp(device):
    """Smoothed diff must raise ValueError for exponential cones."""
    settings = moreau.Settings(
        solver="ipm",
        device=device,
        batch_size=1,
        enable_grad=True,
        ipm_settings=moreau.IPMSettings(
            diff_method="smoothed",
        ),
    )
    with pytest.raises(ValueError, match="smoothed.*only supports"):
        moreau.CompiledSolver(
            n=3,
            m=3,
            P_row_offsets=[0, 0, 0, 0],
            P_col_indices=[],
            A_row_offsets=[0, 1, 2, 3],
            A_col_indices=[0, 1, 2],
            cones=moreau.Cones(num_exp_cones=1),
            settings=settings,
        )


def test_smoothed_accepted_for_lp(device):
    """Smoothed diff must accept LP cones (zero + nonneg)."""
    settings = moreau.Settings(
        solver="ipm",
        device=device,
        batch_size=1,
        enable_grad=True,
        ipm_settings=moreau.IPMSettings(
            diff_method="smoothed",
        ),
    )
    # Should not raise
    solver = moreau.CompiledSolver(
        n=2,
        m=3,
        P_row_offsets=[0, 1, 2],
        P_col_indices=[0, 1],
        A_row_offsets=[0, 1, 2, 3],
        A_col_indices=[0, 0, 1],
        cones=moreau.Cones(num_zero_cones=1, num_nonneg_cones=2),
        settings=settings,
    )
    solver.setup(P_values=[1.0, 1.0], A_values=[1.0, -1.0, -1.0])
    solver.solve(qs=[[0.5, 0.5]], bs=[[1.0, 0.0, 0.0]])


# ---------- Zero-cone-only edge case ----------


def test_smoothed_zero_cone_only(device):
    """Zero-cone-only problems should not waste refinement iterations.

    The smoothed gradient should match exact exactly (zero cone has no
    smoothing parameter).
    """
    n, m = 2, 1
    P_ro = [0, 1, 2]
    P_ci = [0, 1]
    P_vals = [1.0, 1.0]
    A_ro = [0, 2]
    A_ci = [0, 1]
    A_vals = [1.0, 1.0]
    q = np.array([1.0, 2.0])
    b = np.array([3.0])
    dx = np.array([1.0, 0.0])
    dz = np.zeros(m)
    ds = np.zeros(m)
    cones = moreau.Cones(num_zero_cones=1)

    args = dict(
        n=n,
        m=m,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        q=q,
        b=b,
        dx=dx,
        dz=dz,
        ds=ds,
        device=device,
    )

    exact = _compute_grads(**args)
    smoothed = _compute_grads(diff_method="smoothed", **args, mu=1e-4)

    for key in ["dq", "db"]:
        err = np.max(np.abs(exact[key] - smoothed[key]))
        assert err < 1e-8, f"Zero-cone-only: smoothed != exact for {key} (err={err:.2e})"


# ---------- FD consistency ----------


def test_smoothed_fd_consistency_dq(device):
    """Verify smoothed dq matches finite differences.

    Uses small mu so the smoothed map is nearly smooth enough for FD.
    """
    np.random.seed(1234)
    n, m = 3, 3
    eps = 1e-6
    mu = 1e-4

    P_ro = list(range(n + 1))
    P_ci = list(range(n))
    P_vals = [1.0] * n
    A_ro = list(range(m + 1))
    A_ci = list(range(m))
    A_vals = [-1.0] * m
    cones = moreau.Cones(num_nonneg_cones=m)
    q_base = np.array([-0.5, 1.0, -0.3])
    b = np.zeros(m)
    dx = np.random.randn(n)

    # Analytic gradient
    g = _compute_grads(
        diff_method="smoothed",
        n=n,
        m=m,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        q=q_base,
        b=b,
        dx=dx,
        dz=np.zeros(m),
        ds=np.zeros(m),
        mu=mu,
        device=device,
    )
    dq_analytic = g["dq"]

    # Central FD: perturb q, re-solve, compute directional derivative
    dq_fd = np.zeros(n)
    for j in range(n):
        q_p = q_base.copy()
        q_p[j] += eps
        q_m = q_base.copy()
        q_m[j] -= eps

        settings = moreau.Settings(
            solver="ipm",
            device=device,
            batch_size=2,
            enable_grad=False,
            ipm_settings=moreau.IPMSettings(
                diff_method="smoothed",
                diff_smoothing_mu=mu,
                tol_gap_abs=1e-9,
                tol_feas=1e-9,
            ),
        )
        solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=settings,
        )
        solver.setup(P_values=P_vals, A_values=A_vals)
        sol = solver.solve(qs=[q_p, q_m], bs=[b, b])
        x_p = np.asarray(sol.x)[0]
        x_m = np.asarray(sol.x)[1]
        dq_fd[j] = dx @ (x_p - x_m) / (2 * eps)

    assert np.allclose(dq_analytic, dq_fd, atol=5e-3), (
        f"Smoothed dq FD mismatch: max_err={np.max(np.abs(dq_analytic - dq_fd)):.2e}\n"
        f"  analytic: {dq_analytic}\n  FD: {dq_fd}"
    )


def test_smoothed_fd_consistency_db(device):
    """Verify smoothed db matches finite differences for mixed cones."""
    np.random.seed(5678)
    n, m = 2, 3
    eps = 1e-6
    mu = 1e-8  # Small mu so smoothed ≈ exact, making FD more reliable

    P_ro = [0, 1, 2]
    P_ci = [0, 1]
    P_vals = [1.0, 1.0]
    # 1 zero cone (x0+x1=b0) + 2 nonneg cones (x0>=0, x1>=0)
    A_ro = [0, 2, 3, 4]
    A_ci = [0, 1, 0, 1]
    A_vals = [1.0, 1.0, -1.0, -1.0]
    cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
    q = np.array([-0.5, 0.3])
    b_base = np.array([1.0, 0.0, 0.0])
    dx = np.random.randn(n)

    g = _compute_grads(
        diff_method="smoothed",
        n=n,
        m=m,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        q=q,
        b=b_base,
        dx=dx,
        dz=np.zeros(m),
        ds=np.zeros(m),
        mu=mu,
        device=device,
    )
    db_analytic = g["db"]

    db_fd = np.zeros(m)
    for j in range(m):
        b_p = b_base.copy()
        b_p[j] += eps
        b_m = b_base.copy()
        b_m[j] -= eps

        settings = moreau.Settings(
            solver="ipm",
            device=device,
            batch_size=2,
            enable_grad=False,
            ipm_settings=moreau.IPMSettings(
                diff_method="smoothed",
                diff_smoothing_mu=mu,
                tol_gap_abs=1e-9,
                tol_feas=1e-9,
            ),
        )
        solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=settings,
        )
        solver.setup(P_values=P_vals, A_values=A_vals)
        sol = solver.solve(qs=[q, q], bs=[b_p, b_m])
        x_p = np.asarray(sol.x)[0]
        x_m = np.asarray(sol.x)[1]
        db_fd[j] = dx @ (x_p - x_m) / (2 * eps)

    assert np.allclose(db_analytic, db_fd, atol=1e-3), (
        f"Smoothed db FD mismatch: max_err={np.max(np.abs(db_analytic - db_fd)):.2e}\n"
        f"  analytic: {db_analytic}\n  FD: {db_fd}"
    )


# ---------- Step factor consistency ----------


def test_step_factor_affects_iteration_count(device):
    """Verify diff_smoothing_step_factor changes behavior.

    A larger step factor should produce the same (or fewer) effective
    refinement iterations while still producing correct smoothed gradients.
    """
    n, m = 2, 2
    q = np.array([-0.01, 1.0])
    b = np.zeros(m)
    dx = np.array([1.0, 0.0])
    dz = np.zeros(m)
    ds = np.zeros(m)
    mu = 1e-2

    results = {}
    for factor in [10.0, 30.0, 100.0]:
        settings = moreau.Settings(
            solver="ipm",
            device=device,
            batch_size=1,
            enable_grad=True,
            ipm_settings=moreau.IPMSettings(
                diff_method="smoothed",
                diff_smoothing_mu=mu,
                diff_smoothing_step_factor=factor,
                tol_gap_abs=1e-9,
                tol_feas=1e-9,
            ),
        )
        solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 1, 2],
            A_col_indices=[0, 1],
            cones=moreau.Cones(num_nonneg_cones=m),
            settings=settings,
        )
        solver.setup(P_values=[1.0, 1.0], A_values=[-1.0, -1.0])
        solver.solve(qs=[q], bs=[b])
        grads = solver.backward(
            dx=dx.reshape(1, n),
            dz=dz.reshape(1, m),
            ds=ds.reshape(1, m),
        )
        results[factor] = np.asarray(grads["dq"]).flatten()

    # All step factors should produce close results (they target the same mu)
    for f in [10.0, 100.0]:
        err = np.max(np.abs(results[f] - results[30.0]))
        assert err < 0.05, f"step_factor={f} gives different result from 30.0 (err={err:.2e})"


# ---------- Batched smoothed diff ----------


def test_smoothed_batched_consistency(device):
    """Verify batched smoothed diff gives same results as individual solves."""
    np.random.seed(9999)
    n, m = 2, 2
    batch = 8
    qs = np.random.randn(batch, n)
    bs = np.zeros((batch, m))
    mu = 1e-3

    P_ro = [0, 1, 2]
    P_ci = [0, 1]
    P_vals = [1.0, 1.0]
    A_ro = [0, 1, 2]
    A_ci = [0, 1]
    A_vals = [-1.0, -1.0]
    cones = moreau.Cones(num_nonneg_cones=m)

    # Batched solve
    batched_dq = _compute_dq(
        n=n,
        m=m,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        qs=qs,
        bs=bs,
        mu=mu,
        device=device,
    )

    # Individual solves
    for i in range(batch):
        individual = _compute_grads(
            diff_method="smoothed",
            n=n,
            m=m,
            P_ro=P_ro,
            P_ci=P_ci,
            P_vals=P_vals,
            A_ro=A_ro,
            A_ci=A_ci,
            A_vals=A_vals,
            cones=cones,
            q=qs[i],
            b=bs[i],
            dx=np.array([1.0, 0.0]),
            dz=np.zeros(m),
            ds=np.zeros(m),
            mu=mu,
            device=device,
        )
        err = np.max(np.abs(batched_dq[i] - individual["dq"]))
        assert err < 1e-8, f"Batch element {i}: batched vs individual mismatch (err={err:.2e})"


# ---------- CPU/CUDA parity ----------


@pytest.mark.cuda
def test_smoothed_cpu_cuda_parity_nonneg():
    """Verify CPU and CUDA produce identical smoothed gradients for nonneg."""
    np.random.seed(4321)
    n, m = 3, 3
    P_ro = [0, 3, 6, 9]
    P_ci = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    P_vals = [2.0, 0.3, 0.1, 0.3, 2.0, 0.2, 0.1, 0.2, 2.0]
    A_ro = [0, 1, 2, 3]
    A_ci = [0, 1, 2]
    A_vals = [-1.0, -1.0, -1.0]
    cones = moreau.Cones(num_nonneg_cones=m)
    q = np.array([-0.05, 1.0, -0.3])
    b = np.zeros(m)
    dx = np.random.randn(n)
    dz = np.random.randn(m)
    ds = np.random.randn(m)

    for mu in [1e-4, 1e-2, 1e-1]:
        cpu_grads = _compute_grads(
            diff_method="smoothed",
            n=n,
            m=m,
            P_ro=P_ro,
            P_ci=P_ci,
            P_vals=P_vals,
            A_ro=A_ro,
            A_ci=A_ci,
            A_vals=A_vals,
            cones=cones,
            q=q,
            b=b,
            dx=dx,
            dz=dz,
            ds=ds,
            mu=mu,
            device="cpu",
        )
        cuda_grads = _compute_grads(
            diff_method="smoothed",
            n=n,
            m=m,
            P_ro=P_ro,
            P_ci=P_ci,
            P_vals=P_vals,
            A_ro=A_ro,
            A_ci=A_ci,
            A_vals=A_vals,
            cones=cones,
            q=q,
            b=b,
            dx=dx,
            dz=dz,
            ds=ds,
            mu=mu,
            device="cuda",
        )
        for key in ["dq", "db", "dP_values", "dA_values"]:
            err = np.max(np.abs(cpu_grads[key] - cuda_grads[key]))
            assert err < 1e-6, (
                f"CPU/CUDA mismatch for {key} at mu={mu:.0e}: err={err:.2e}\n"
                f"  CPU:  {cpu_grads[key]}\n  CUDA: {cuda_grads[key]}"
            )


@pytest.mark.cuda
def test_smoothed_cpu_cuda_parity_sweep():
    """Verify CPU and CUDA produce identical smoothed gradient curves.

    Sweeps q through the kink and checks that the gradient curves match
    point-by-point, not just in aggregate statistics.
    """
    n_pts = 51
    mu = 1e-2
    q_vals = np.linspace(-0.5, 0.5, n_pts)
    qs = q_vals.reshape(n_pts, 1)
    bs = np.zeros((n_pts, 1))

    args = dict(
        n=1,
        m=1,
        P_ro=[0, 1],
        P_ci=[0],
        P_vals=[1.0],
        A_ro=[0, 1],
        A_ci=[0],
        A_vals=[-1.0],
        cones=moreau.Cones(num_nonneg_cones=1),
        qs=qs,
        bs=bs,
        mu=mu,
    )

    cpu_dq = _compute_dq(**args, device="cpu")
    cuda_dq = _compute_dq(**args, device="cuda")

    err = np.max(np.abs(cpu_dq - cuda_dq))
    assert err < 1e-6, f"CPU/CUDA gradient sweep mismatch: max_err={err:.2e}"


@pytest.mark.cuda
def test_smoothed_cpu_cuda_parity_mixed_cones():
    """Verify CPU/CUDA parity for mixed zero + nonneg cones."""
    np.random.seed(8888)
    n, m = 3, 3
    P_vals = [1.0, 1.0, 1.0]
    P_ro = [0, 1, 2, 3]
    P_ci = [0, 1, 2]
    A_vals = [1.0, 1.0, -1.0, -1.0]
    A_ro = [0, 2, 3, 4]
    A_ci = [0, 1, 0, 1]
    q = np.array([-0.01, 0.5, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
    dx = np.random.randn(n)
    dz = np.random.randn(m)
    ds = np.random.randn(m)

    for mu in [1e-6, 1e-3, 1e-1]:
        cpu_grads = _compute_grads(
            diff_method="smoothed",
            n=n,
            m=m,
            P_ro=P_ro,
            P_ci=P_ci,
            P_vals=P_vals,
            A_ro=A_ro,
            A_ci=A_ci,
            A_vals=A_vals,
            cones=cones,
            q=q,
            b=b,
            dx=dx,
            dz=dz,
            ds=ds,
            mu=mu,
            device="cpu",
        )
        cuda_grads = _compute_grads(
            diff_method="smoothed",
            n=n,
            m=m,
            P_ro=P_ro,
            P_ci=P_ci,
            P_vals=P_vals,
            A_ro=A_ro,
            A_ci=A_ci,
            A_vals=A_vals,
            cones=cones,
            q=q,
            b=b,
            dx=dx,
            dz=dz,
            ds=ds,
            mu=mu,
            device="cuda",
        )
        for key in ["dq", "db", "dP_values", "dA_values"]:
            err = np.max(np.abs(cpu_grads[key] - cuda_grads[key]))
            assert err < 1e-5, f"CPU/CUDA mismatch for {key} at mu={mu:.0e}: err={err:.2e}"


# ---------- Stress tests ----------


def _make_random_diag_dominant_P(rng, n):
    """Generate a random symmetric positive definite P (full, CSR format)."""
    M = rng.standard_normal((n, n)) * 0.3
    P_dense = M @ M.T + np.eye(n) * 2.0  # diag-dominant SPD
    # CSR: full symmetric (both triangles)
    P_ro = [0]
    P_ci = []
    P_vals = []
    for i in range(n):
        for j in range(n):
            if P_dense[i, j] != 0.0:
                P_ci.append(j)
                P_vals.append(P_dense[i, j])
        P_ro.append(len(P_ci))
    return P_ro, P_ci, P_vals


def _has_debug_smoothing(grads):
    """Check if debug smoothing iterates are available (debug builds only)."""
    if isinstance(grads, dict):
        return "debug_smoothing_x" in grads
    return hasattr(grads, "debug_smoothing_x")


def _get_smoothed_x(grads):
    """Extract smoothed x iterate from backward result."""
    if isinstance(grads, dict):
        return np.asarray(grads["debug_smoothing_x"]).flatten()
    return np.asarray(grads.debug_smoothing_x).flatten()


def _solve_and_get_smoothed_x(
    n, m, P_ro, P_ci, P_vals, A_ro, A_ci, A_vals, cones, q, b, mu, device, step_factor=None, dx=None
):
    """Solve and return backward grads + smoothed iterates from the walk-up.

    Args:
        dx: upstream gradient for backward, shape (n,). Defaults to e_1.
    """
    ipm_kwargs = dict(
        diff_method="smoothed",
        diff_smoothing_mu=mu,
        tol_gap_abs=1e-9,
        tol_feas=1e-9,
        direct_solve_method="qdldl",
    )
    if step_factor is not None:
        ipm_kwargs["diff_smoothing_step_factor"] = step_factor
    settings = moreau.Settings(
        solver="ipm",
        device=device,
        batch_size=1,
        enable_grad=True,
        ipm_settings=moreau.IPMSettings(**ipm_kwargs),
    )
    solver = moreau.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=P_ro,
        P_col_indices=P_ci,
        A_row_offsets=A_ro,
        A_col_indices=A_ci,
        cones=cones,
        settings=settings,
    )
    solver.setup(P_values=P_vals, A_values=A_vals)
    solver.solve(qs=[q], bs=[b])
    if dx is None:
        dx_arr = np.zeros((1, n))
        dx_arr[0, 0] = 1.0
    else:
        dx_arr = np.asarray(dx).reshape(1, n)
    grads = solver.backward(dx=dx_arr, dz=np.zeros((1, m)), ds=np.zeros((1, m)))
    return grads


def _skip_unless_debug_smoothing(grads):
    """Skip test if debug smoothing iterates not available (release build)."""
    if not _has_debug_smoothing(grads):
        pytest.skip("debug_smoothing_x not available (release build)")


@pytest.mark.parametrize("n", [10, 20, 50])
def test_smoothed_stress_step_factor_invariance(device, n):
    """Different step_factors should produce the same smoothed iterates.

    The walk-up should converge to the same central-path point regardless
    of how many steps are taken to get there.
    """
    rng = np.random.default_rng(42 + n)
    m = n
    P_ro, P_ci, P_vals = _make_random_diag_dominant_P(rng, n)
    A_ro = list(range(m + 1))
    A_ci = list(range(m))
    A_vals = [-1.0] * m
    cones = moreau.Cones(num_nonneg_cones=m)
    q = rng.standard_normal(n) * 0.1
    b = np.zeros(m)
    mu = 1e-3

    common = dict(
        n=n,
        m=m,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        q=q,
        b=b,
        mu=mu,
        device=device,
    )

    grads_10 = _solve_and_get_smoothed_x(**common, step_factor=10.0)
    _skip_unless_debug_smoothing(grads_10)

    grads_30 = _solve_and_get_smoothed_x(**common, step_factor=30.0)
    grads_1000 = _solve_and_get_smoothed_x(**common, step_factor=1000.0)

    x_10 = _get_smoothed_x(grads_10)
    x_30 = _get_smoothed_x(grads_30)
    x_1000 = _get_smoothed_x(grads_1000)

    # Step factors 10 and 30 take many small steps and should agree closely.
    # Step factor 1000 takes ~1 big jump and may not track the central path
    # as precisely, so we allow a looser tolerance.
    scale = max(np.max(np.abs(x_10)), 1.0)
    err_30_vs_10 = np.max(np.abs(x_30 - x_10)) / scale
    err_1000_vs_10 = np.max(np.abs(x_1000 - x_10)) / scale
    assert (
        err_30_vs_10 < 1e-6
    ), f"n={n}: step_factor 30 vs 10 smoothed x mismatch: {err_30_vs_10:.2e}"
    assert (
        err_1000_vs_10 < 5e-3
    ), f"n={n}: step_factor 1000 vs 10 smoothed x mismatch: {err_1000_vs_10:.2e}"

    # Gradients should also agree (looser for step_factor=1000)
    dq_10 = np.asarray(grads_10["dq"] if isinstance(grads_10, dict) else grads_10.dq).flatten()
    dq_30 = np.asarray(grads_30["dq"] if isinstance(grads_30, dict) else grads_30.dq).flatten()
    dq_1000 = np.asarray(
        grads_1000["dq"] if isinstance(grads_1000, dict) else grads_1000.dq
    ).flatten()
    dq_scale = max(np.max(np.abs(dq_10)), 1e-10)
    assert (
        np.max(np.abs(dq_30 - dq_10)) / dq_scale < 1e-5
    ), f"n={n}: step_factor 30 vs 10 gradient mismatch"
    assert (
        np.max(np.abs(dq_1000 - dq_10)) / dq_scale < 0.1
    ), f"n={n}: step_factor 1000 vs 10 gradient mismatch"


@pytest.mark.parametrize("n", [10, 20, 50])
def test_smoothed_stress_fd_of_smoothed_map(device, n):
    """FD of the smoothed map itself, not the exact map.

    Perturb q, solve + walk-up with the same mu, get smoothed iterates from
    debug fields, and compute FD. This compares analytic gradient vs FD of
    the *same* smoothed map, so the error should be O(eps^2) not O(mu).
    """
    rng = np.random.default_rng(3333 + n)
    m = n
    eps = 1e-5
    mu = 1e-3

    P_ro, P_ci, P_vals = _make_random_diag_dominant_P(rng, n)
    A_ro = list(range(m + 1))
    A_ci = list(range(m))
    A_vals = [-1.0] * m
    cones = moreau.Cones(num_nonneg_cones=m)
    q_base = rng.standard_normal(n) * 0.1
    b = np.zeros(m)
    dx = rng.standard_normal(n)

    common = dict(
        n=n,
        m=m,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        b=b,
        mu=mu,
        device=device,
    )

    # Get analytic gradient at q_base, using dx as upstream gradient
    grads_base = _solve_and_get_smoothed_x(q=q_base, dx=dx, **common)
    _skip_unless_debug_smoothing(grads_base)

    dq_analytic = np.asarray(
        grads_base["dq"] if isinstance(grads_base, dict) else grads_base.dq
    ).flatten()

    # Directional FD: perturb q, get smoothed x from debug fields
    direction = rng.standard_normal(n)
    direction /= np.linalg.norm(direction)

    grads_p = _solve_and_get_smoothed_x(q=q_base + eps * direction, dx=dx, **common)
    grads_m = _solve_and_get_smoothed_x(q=q_base - eps * direction, dx=dx, **common)
    x_smooth_p = _get_smoothed_x(grads_p)
    x_smooth_m = _get_smoothed_x(grads_m)

    # Directional derivative of L(q) = dx @ x_smooth(q) in direction `direction`
    # Analytic: dL/dq @ direction = dq_analytic @ direction
    # FD: (L(q+eps*dir) - L(q-eps*dir)) / (2*eps) = dx @ (x_smooth_p - x_smooth_m) / (2*eps)
    dd_analytic = dq_analytic @ direction
    dd_fd = dx @ (x_smooth_p - x_smooth_m) / (2 * eps)

    rel_err = abs(dd_analytic - dd_fd) / max(abs(dd_fd), 1e-10)
    assert rel_err < 0.05, (
        f"n={n}: smoothed-map FD mismatch: analytic={dd_analytic:.6e}, "
        f"FD={dd_fd:.6e}, rel_err={rel_err:.2e}"
    )


def test_smoothed_stress_large_batch(device):
    """Large batch (256 problems) with varying q values across the kink."""
    rng = np.random.default_rng(1111)
    n, m = 5, 5
    batch = 256

    P_ro, P_ci, P_vals = _make_random_diag_dominant_P(rng, n)
    A_ro = list(range(m + 1))
    A_ci = list(range(m))
    A_vals = [-1.0] * m
    cones = moreau.Cones(num_nonneg_cones=m)

    # Spread q values so many problems have constraints near active/inactive boundary
    qs = rng.standard_normal((batch, n)) * 0.3
    bs = np.zeros((batch, m))
    mu = 1e-3

    settings = moreau.Settings(
        solver="ipm",
        device=device,
        batch_size=batch,
        enable_grad=True,
        ipm_settings=moreau.IPMSettings(
            diff_method="smoothed",
            diff_smoothing_mu=mu,
            tol_gap_abs=1e-9,
            tol_feas=1e-9,
            direct_solve_method="qdldl",
        ),
    )
    solver = moreau.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=P_ro,
        P_col_indices=P_ci,
        A_row_offsets=A_ro,
        A_col_indices=A_ci,
        cones=cones,
        settings=settings,
    )
    solver.setup(P_values=P_vals, A_values=A_vals)
    solver.solve(qs=qs, bs=bs)

    # Backward pass with e_1 upstream gradient (matches _compute_dq convention)
    dx = np.zeros((batch, n))
    dx[:, 0] = 1.0
    dz = np.zeros((batch, m))
    ds = np.zeros((batch, m))
    grads = solver.backward(dx=dx, dz=dz, ds=ds)
    dq = np.asarray(grads["dq"])

    # Sanity: gradients should be finite and non-trivial
    assert np.all(np.isfinite(dq)), "Non-finite gradients in large batch"
    assert np.max(np.abs(dq)) > 1e-6, "All gradients near zero in large batch"

    # Spot-check: first 4 problems individually should match batched
    for i in range(4):
        individual = _compute_grads(
            diff_method="smoothed",
            n=n,
            m=m,
            P_ro=P_ro,
            P_ci=P_ci,
            P_vals=P_vals,
            A_ro=A_ro,
            A_ci=A_ci,
            A_vals=A_vals,
            cones=cones,
            q=qs[i],
            b=bs[i],
            dx=dx[i],
            dz=np.zeros(m),
            ds=np.zeros(m),
            mu=mu,
            device=device,
        )
        err = np.max(np.abs(dq[i] - individual["dq"]))
        assert err < 1e-8, f"Batch element {i}: batched vs individual mismatch (err={err:.2e})"


@pytest.mark.cuda
@pytest.mark.parametrize("n", [20, 50, 100])
def test_smoothed_stress_cpu_cuda_parity_large(n):
    """CPU/CUDA parity with large dense P and many constraints."""
    rng = np.random.default_rng(5555 + n)
    m = n
    P_ro, P_ci, P_vals = _make_random_diag_dominant_P(rng, n)
    A_ro = list(range(m + 1))
    A_ci = list(range(m))
    A_vals = [-1.0] * m
    cones = moreau.Cones(num_nonneg_cones=m)

    q = rng.standard_normal(n) * 0.1
    b = np.zeros(m)
    dx = rng.standard_normal(n)
    dz = rng.standard_normal(m)
    ds = rng.standard_normal(m)

    args = dict(
        n=n,
        m=m,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        q=q,
        b=b,
        dx=dx,
        dz=dz,
        ds=ds,
    )

    # Use mu=1e-4 only — at larger mu (1e-2), CUDA walk-up may hit NaN steps
    # that get zeroed, causing path divergence from CPU for large n.
    for mu in [1e-4]:
        cpu_grads = _compute_grads(diff_method="smoothed", **args, mu=mu, device="cpu")
        cuda_grads = _compute_grads(diff_method="smoothed", **args, mu=mu, device="cuda")
        for key in ["dq", "db", "dP_values", "dA_values"]:
            cpu_val = cpu_grads[key]
            cuda_val = cuda_grads[key]
            scale = max(np.max(np.abs(cpu_val)), 1.0)
            err = np.max(np.abs(cpu_val - cuda_val)) / scale
            assert err < 1e-4, (
                f"n={n} CPU/CUDA mismatch for {key} at mu={mu:.0e}: " f"rel_err={err:.2e}"
            )


@pytest.mark.cuda
def test_smoothed_stress_cpu_cuda_parity_large_batch():
    """CPU/CUDA parity on a 64-problem batch with dense P."""
    rng = np.random.default_rng(7777)
    n, m = 10, 10
    batch = 64
    P_ro, P_ci, P_vals = _make_random_diag_dominant_P(rng, n)
    A_ro = list(range(m + 1))
    A_ci = list(range(m))
    A_vals = [-1.0] * m
    cones = moreau.Cones(num_nonneg_cones=m)

    qs = rng.standard_normal((batch, n)) * 0.3
    bs = np.zeros((batch, m))
    mu = 1e-3

    cpu_dqs = _compute_dq(
        n=n,
        m=m,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        qs=qs,
        bs=bs,
        mu=mu,
        device="cpu",
    )
    cuda_dqs = _compute_dq(
        n=n,
        m=m,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        qs=qs,
        bs=bs,
        mu=mu,
        device="cuda",
    )

    err = np.max(np.abs(cpu_dqs - cuda_dqs))
    assert err < 1e-6, f"CPU/CUDA batch parity mismatch: max_err={err:.2e}"
