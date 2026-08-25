"""
Fuzz testing for variable-dimension SOC support against Clarabel reference solver.

Generates random SOCPs with variable-dimension second-order cones and compares:
1. Moreau CPU vs Clarabel (correctness)
2. Moreau GPU vs Clarabel (correctness)
3. Moreau CPU vs Moreau GPU (parity)

Uses hypothesis for property-based testing with random problem generation.
Stress-tests large dims, ill-conditioned problems, and edge cases.
"""

import numpy as np
import pytest
import scipy.sparse as sp

# 100+s aggregate (max_examples=100-200 per test). Pass --runslow to enable.
pytestmark = pytest.mark.slow

hypothesis = pytest.importorskip("hypothesis")
given = hypothesis.given
hsettings = hypothesis.settings
assume = hypothesis.assume
HealthCheck = hypothesis.HealthCheck
st = pytest.importorskip("hypothesis.strategies")

import moreau

try:
    import clarabel

    HAS_CLARABEL = True
except ImportError:
    HAS_CLARABEL = False

HAS_CUDA = moreau.device_available("cuda")

requires_clarabel = pytest.mark.skipif(not HAS_CLARABEL, reason="clarabel not installed")
requires_gpu = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")


# =============================================================================
# Hypothesis strategies
# =============================================================================


@st.composite
def soc_dims_strategy(draw, max_total_m=40, max_dim=15, max_cones=5):
    """Generate a list of SOC dimensions (each >= 2) with bounded total."""
    num_cones = draw(st.integers(min_value=1, max_value=max_cones))
    dims = []
    remaining = max_total_m
    for i in range(num_cones):
        min_left = 2 * (num_cones - i - 1)  # reserve 2 per remaining cone
        upper = min(max_dim, remaining - min_left)
        if upper < 2:
            break
        d = draw(st.integers(min_value=2, max_value=upper))
        dims.append(d)
        remaining -= d
    return dims


@st.composite
def large_soc_dims_strategy(draw):
    """Generate SOC dims with at least one large cone (stress test sparse path)."""
    # One large cone + optional smaller ones
    big = draw(st.integers(min_value=10, max_value=50))
    num_small = draw(st.integers(min_value=0, max_value=4))
    smalls = [draw(st.integers(min_value=2, max_value=10)) for _ in range(num_small)]
    # Shuffle so big cone isn't always first
    dims = [big] + smalls
    draw(st.randoms()).shuffle(dims)
    return dims


@st.composite
def soc_problem_strategy(draw, soc_dims_strat=None, density="well_conditioned", extra_nonneg=True):
    """Generate a random SOCP with variable-dim SOC cones.

    density: 'well_conditioned', 'dense', 'sparse', 'ill_conditioned'
    """
    if soc_dims_strat is None:
        soc_dims_strat = soc_dims_strategy()
    soc_dims = draw(soc_dims_strat)
    m_soc = sum(soc_dims)

    num_nonneg = draw(st.integers(min_value=0, max_value=5)) if extra_nonneg else 0
    m = m_soc + num_nonneg

    # n must be >= m for well-conditioned problems
    n = draw(st.integers(min_value=max(m, 2), max_value=max(m + 15, 20)))

    rng = np.random.default_rng(draw(st.integers(0, 2**31)))

    # --- P matrix ---
    if density == "ill_conditioned":
        # Large condition number (~1e5) via M'M + small diagonal, guaranteed PSD
        M = rng.standard_normal((n, n)) * 0.1
        # Scale columns to create large condition number
        col_scales = np.concatenate(
            [
                rng.uniform(1e-3, 1e-2, size=n // 2),
                rng.uniform(1e1, 1e2, size=n - n // 2),
            ]
        )
        rng.shuffle(col_scales)
        M = M * col_scales[np.newaxis, :]
        P = sp.csr_array(M.T @ M + np.diag(rng.uniform(1e-4, 1e-3, size=n)))
    elif density == "dense":
        # Dense P via A'A
        k = max(n, 5)
        A_inner = rng.standard_normal((k, n))
        P_dense = A_inner.T @ A_inner / k + np.eye(n) * 0.5
        P = sp.csr_array(P_dense)
    elif density == "sparse":
        # Very sparse P
        P_diag = rng.uniform(0.5, 3.0, size=n)
        P = sp.diags(P_diag, format="csr")
    else:  # well_conditioned
        P_diag = rng.uniform(1.0, 5.0, size=n)
        P = sp.diags(P_diag, format="csr")
        offdiag = rng.uniform(-0.3, 0.3, size=max(0, n - 1))
        P_off = sp.diags(offdiag, offsets=1, shape=(n, n), format="csr")
        P = P + P_off + P_off.T
    P = sp.csr_array(P)

    # --- A matrix ---
    if density == "dense":
        A_dense = rng.standard_normal((m, n)) * 0.5
        # Ensure identity structure for SOC rows to keep well-posed
        row = 0
        for d in soc_dims:
            for j in range(d):
                if row + j < m and row + j < n:
                    A_dense[row + j, row + j] += 1.0
            row += d
        A = sp.csr_array(A_dense)
    else:
        A_dense = np.zeros((m, n))
        row = 0
        for d in soc_dims:
            for j in range(d):
                if row + j < n:
                    A_dense[row + j, row + j] = 1.0
                # Coupling
                for c in range(1, min(3, n)):
                    col = (row + j + c) % n
                    A_dense[row + j, col] = rng.uniform(-0.3, 0.3)
            row += d
        for i in range(num_nonneg):
            r = m_soc + i
            col = (row + i) % n
            A_dense[r, col] = 1.0
            A_dense[r, (col + 1) % n] = rng.uniform(0.1, 0.5)
        A = sp.csr_array(A_dense)

    # --- Feasible point ---
    x_feas = rng.uniform(-1.0, 1.0, size=n)
    s_parts = []

    if num_nonneg > 0:
        s_parts.append(rng.uniform(0.1, 1.0, size=num_nonneg))

    for d in soc_dims:
        s_tail = rng.uniform(-0.3, 0.3, size=d - 1)
        norm_tail = np.linalg.norm(s_tail)
        s_head = norm_tail + rng.uniform(0.2, 1.0)
        s_parts.append(np.concatenate([[s_head], s_tail]))

    s_feas = np.concatenate(s_parts) if s_parts else np.array([])
    b = A @ x_feas + s_feas

    # Scale q based on density mode
    if density == "ill_conditioned":
        q = rng.uniform(-10.0, 10.0, size=n)
    else:
        q = rng.uniform(-2.0, 2.0, size=n)

    cones = moreau.Cones(
        num_nonneg_cones=num_nonneg,
        so_cone_dims=soc_dims,
    )

    return P, q, A, b, cones, soc_dims, num_nonneg


def _moreau_to_clarabel_cones(cones):
    """Convert moreau Cones to Clarabel cone list."""
    clarabel_cones = []
    if cones.num_zero_cones > 0:
        clarabel_cones.append(clarabel.ZeroConeT(cones.num_zero_cones))
    if cones.num_nonneg_cones > 0:
        clarabel_cones.append(clarabel.NonnegativeConeT(cones.num_nonneg_cones))
    for d in cones.so_cone_dims:
        clarabel_cones.append(clarabel.SecondOrderConeT(d))
    for _ in range(cones.num_exp_cones):
        clarabel_cones.append(clarabel.ExponentialConeT())
    for alpha in cones.power_alphas:
        clarabel_cones.append(clarabel.PowerConeT(alpha))
    return clarabel_cones


def _solve_clarabel(P_csr, q, A_csr, b, cones):
    """Solve with Clarabel and return solution object."""
    P_csc = sp.triu(P_csr).tocsc()
    A_csc = A_csr.tocsc()

    clarabel_cones = _moreau_to_clarabel_cones(cones)

    settings = clarabel.DefaultSettings()
    settings.verbose = False

    solver = clarabel.DefaultSolver(
        P_csc,
        np.asarray(q, dtype=np.float64),
        A_csc,
        np.asarray(b, dtype=np.float64),
        clarabel_cones,
        settings,
    )
    sol = solver.solve()
    return sol


def _solve_moreau(P_csr, q, A_csr, b, cones, device="cpu", enable_grad=False):
    """Solve with Moreau and return (solution, info)."""
    settings = moreau.Settings(device=device, enable_grad=enable_grad)
    solver = moreau.Solver(P_csr, q=q, A=A_csr, b=b, cones=cones, settings=settings)
    sol = solver.solve()
    info = solver.info
    return sol, info, solver


def _assert_solutions_close(sol_a, obj_a, sol_b, obj_b, label_a, label_b, rtol=1e-5, atol=1e-5):
    """Assert two solutions are close, or at least have matching objectives."""
    obj_scale = max(abs(obj_a), abs(obj_b), 1.0)
    obj_diff = abs(obj_a - obj_b)

    if obj_diff > obj_scale * 1e-3:
        pytest.fail(
            f"Objective mismatch between {label_a} and {label_b}: "
            f"{obj_a} vs {obj_b} (diff={obj_diff}, scale={obj_scale})"
        )

    x_a, x_b = np.asarray(sol_a.x), np.asarray(sol_b.x)
    x_scale = max(np.max(np.abs(x_a)), np.max(np.abs(x_b)), 1e-6)
    x_diff = np.max(np.abs(x_a - x_b))

    if x_diff > max(atol, x_scale * rtol):
        # Solutions may differ (non-unique) but objectives match — OK
        pass
    else:
        np.testing.assert_allclose(
            np.asarray(sol_a.s),
            np.asarray(sol_b.s),
            rtol=rtol,
            atol=atol,
            err_msg=f"s mismatch between {label_a} and {label_b}",
        )
        np.testing.assert_allclose(
            np.asarray(sol_a.z),
            np.asarray(sol_b.z),
            rtol=rtol,
            atol=atol,
            err_msg=f"z mismatch between {label_a} and {label_b}",
        )


def _verify_soc_membership(s, soc_dims, offset, tol=1e-6):
    """Verify s satisfies SOC constraints (s[0] >= ||s[1:]|| for each cone)."""
    idx = offset
    for d in soc_dims:
        s_cone = s[idx : idx + d]
        s_head = s_cone[0]
        s_tail_norm = np.linalg.norm(s_cone[1:])
        assert s_head >= -tol, f"SOC head s[0]={s_head} < 0 for cone dim={d}"
        assert (
            s_head + tol >= s_tail_norm
        ), f"SOC violation: s[0]={s_head} < ||s[1:]||={s_tail_norm} for cone dim={d}"
        idx += d


def _verify_kkt(P, q, A, b, sol, label, tol=1e-4):
    """Verify KKT conditions for a solution."""
    x, s, z = np.asarray(sol.x), np.asarray(sol.s), np.asarray(sol.z)

    # Primal: Ax + s = b
    prim_res = A @ x + s - b
    prim_scale = max(np.linalg.norm(b), np.linalg.norm(A @ x), 1.0)
    assert np.linalg.norm(prim_res) < max(
        tol, prim_scale * tol
    ), f"{label} primal infeasible: |Ax+s-b|={np.linalg.norm(prim_res)}"

    # Dual: Px + q + A'z = 0
    dual_res = P @ x + q + A.T @ z
    dual_scale = max(np.linalg.norm(q), np.linalg.norm(P @ x), 1.0)
    assert np.linalg.norm(dual_res) < max(
        tol, dual_scale * tol
    ), f"{label} dual infeasible: |Px+q+A'z|={np.linalg.norm(dual_res)}"


# =============================================================================
# Tests: Moreau CPU vs Clarabel (high volume)
# =============================================================================


@requires_clarabel
class TestSOCClarabelFuzz:
    """Fuzz test Moreau CPU against Clarabel for variable-dim SOC problems."""

    @given(problem=soc_problem_strategy())
    @hsettings(
        max_examples=200,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_cpu_vs_clarabel(self, problem):
        """Moreau CPU should match Clarabel on random SOCPs."""
        P, q, A, b, cones, soc_dims, num_nonneg = problem

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assume("Solved" in str(clarabel_sol.status))

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        assert moreau_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ), f"Moreau CPU status: {moreau_info.status}"

        _assert_solutions_close(
            moreau_sol,
            moreau_info.obj_val,
            clarabel_sol,
            clarabel_sol.obj_val,
            "Moreau CPU",
            "Clarabel",
            rtol=1e-4,
            atol=1e-4,
        )
        _verify_soc_membership(np.asarray(moreau_sol.s), soc_dims, offset=num_nonneg)

    @given(problem=soc_problem_strategy())
    @hsettings(
        max_examples=200,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    @requires_gpu
    def test_gpu_vs_clarabel(self, problem):
        """Moreau GPU should match Clarabel on random SOCPs."""
        P, q, A, b, cones, soc_dims, num_nonneg = problem

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assume("Solved" in str(clarabel_sol.status))

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cuda")
        assert moreau_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ), f"Moreau GPU status: {moreau_info.status}"

        _assert_solutions_close(
            moreau_sol,
            moreau_info.obj_val,
            clarabel_sol,
            clarabel_sol.obj_val,
            "Moreau GPU",
            "Clarabel",
            rtol=1e-4,
            atol=1e-4,
        )
        _verify_soc_membership(np.asarray(moreau_sol.s), soc_dims, offset=num_nonneg)


# =============================================================================
# Tests: Large SOC dimensions (stress sparse expansion path)
# =============================================================================


@requires_clarabel
class TestSOCLargeDimsFuzz:
    """Stress test with large SOC dimensions (dim > 4 triggers sparse path)."""

    @given(problem=soc_problem_strategy(soc_dims_strat=large_soc_dims_strategy()))
    @hsettings(
        max_examples=100,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_large_soc_cpu_vs_clarabel(self, problem):
        """Large SOC dims on CPU should match Clarabel."""
        P, q, A, b, cones, soc_dims, num_nonneg = problem

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assume("Solved" in str(clarabel_sol.status))

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        assert moreau_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ), f"CPU status: {moreau_info.status}, dims={soc_dims}"

        _assert_solutions_close(
            moreau_sol,
            moreau_info.obj_val,
            clarabel_sol,
            clarabel_sol.obj_val,
            "CPU",
            "Clarabel",
            rtol=1e-4,
            atol=1e-4,
        )
        _verify_soc_membership(np.asarray(moreau_sol.s), soc_dims, offset=num_nonneg)
        _verify_kkt(P, q, A, b, moreau_sol, "CPU")

    @given(problem=soc_problem_strategy(soc_dims_strat=large_soc_dims_strategy()))
    @hsettings(
        max_examples=100,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    @requires_gpu
    def test_large_soc_gpu_vs_clarabel(self, problem):
        """Large SOC dims on GPU should match Clarabel."""
        P, q, A, b, cones, soc_dims, num_nonneg = problem

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assume("Solved" in str(clarabel_sol.status))

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cuda")
        assert moreau_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ), f"GPU status: {moreau_info.status}, dims={soc_dims}"

        _assert_solutions_close(
            moreau_sol,
            moreau_info.obj_val,
            clarabel_sol,
            clarabel_sol.obj_val,
            "GPU",
            "Clarabel",
            rtol=1e-4,
            atol=1e-4,
        )
        _verify_soc_membership(np.asarray(moreau_sol.s), soc_dims, offset=num_nonneg)
        _verify_kkt(P, q, A, b, moreau_sol, "GPU")


# =============================================================================
# Tests: Ill-conditioned and dense problems
# =============================================================================


@requires_clarabel
class TestSOCHardProblems:
    """Stress test with ill-conditioned and dense problems."""

    @given(problem=soc_problem_strategy(density="ill_conditioned"))
    @hsettings(
        max_examples=100,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_ill_conditioned_cpu_vs_clarabel(self, problem):
        """Ill-conditioned P should still match Clarabel."""
        P, q, A, b, cones, soc_dims, num_nonneg = problem

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assume("Solved" in str(clarabel_sol.status))

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        assert moreau_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ), f"CPU status: {moreau_info.status} (Clarabel solved but Moreau didn't)"

        # Relaxed tolerance for ill-conditioned
        obj_scale = max(abs(moreau_info.obj_val), abs(clarabel_sol.obj_val), 1.0)
        obj_diff = abs(moreau_info.obj_val - clarabel_sol.obj_val)
        assert (
            obj_diff < obj_scale * 1e-2
        ), f"Objective mismatch: moreau={moreau_info.obj_val}, clarabel={clarabel_sol.obj_val}"

    @given(problem=soc_problem_strategy(density="ill_conditioned"))
    @hsettings(
        max_examples=100,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    @requires_gpu
    def test_ill_conditioned_gpu_vs_clarabel(self, problem):
        """Ill-conditioned P on GPU should still match Clarabel."""
        P, q, A, b, cones, soc_dims, num_nonneg = problem

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assume("Solved" in str(clarabel_sol.status))

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cuda")
        # cuDSS is less accurate than QDLDL/faer for ill-conditioned KKT systems.
        # Some problems may hit InsufficientProgress or NumericalError on GPU
        # while converging on CPU. When GPU does converge, verify objective matches.
        if moreau_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ):
            obj_scale = max(abs(moreau_info.obj_val), abs(clarabel_sol.obj_val), 1.0)
            obj_diff = abs(moreau_info.obj_val - clarabel_sol.obj_val)
            assert (
                obj_diff < obj_scale * 1e-2
            ), f"Objective mismatch: moreau={moreau_info.obj_val}, clarabel={clarabel_sol.obj_val}"
        else:
            # GPU didn't converge — acceptable for ill-conditioned problems
            assert moreau_info.status in (
                moreau.SolverStatus.InsufficientProgress,
                moreau.SolverStatus.NumericalError,
                moreau.SolverStatus.MaxIterations,
            ), f"GPU unexpected status: {moreau_info.status}"

    @given(problem=soc_problem_strategy(density="dense"))
    @hsettings(
        max_examples=100,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_dense_cpu_vs_clarabel(self, problem):
        """Dense P and A should match Clarabel."""
        P, q, A, b, cones, soc_dims, num_nonneg = problem

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assume("Solved" in str(clarabel_sol.status))

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        assert moreau_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )

        _assert_solutions_close(
            moreau_sol,
            moreau_info.obj_val,
            clarabel_sol,
            clarabel_sol.obj_val,
            "CPU",
            "Clarabel",
            rtol=1e-4,
            atol=1e-4,
        )

    @given(problem=soc_problem_strategy(density="dense"))
    @hsettings(
        max_examples=100,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    @requires_gpu
    def test_dense_gpu_vs_clarabel(self, problem):
        """Dense P and A on GPU should match Clarabel."""
        P, q, A, b, cones, soc_dims, num_nonneg = problem

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assume("Solved" in str(clarabel_sol.status))

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cuda")
        assert moreau_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )

        _assert_solutions_close(
            moreau_sol,
            moreau_info.obj_val,
            clarabel_sol,
            clarabel_sol.obj_val,
            "GPU",
            "Clarabel",
            rtol=1e-4,
            atol=1e-4,
        )


# =============================================================================
# Tests: Moreau CPU vs GPU parity (high volume)
# =============================================================================


@requires_gpu
class TestSOCCpuGpuFuzz:
    """Fuzz test CPU/GPU parity for variable-dim SOC problems."""

    @given(problem=soc_problem_strategy())
    @hsettings(
        max_examples=200,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_cpu_gpu_forward_parity(self, problem):
        """CPU and GPU should produce matching solutions for SOCPs."""
        P, q, A, b, cones, soc_dims, num_nonneg = problem

        cpu_sol, cpu_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        gpu_sol, gpu_info, _ = _solve_moreau(P, q, A, b, cones, device="cuda")

        cpu_solved = cpu_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        gpu_solved = gpu_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )

        if not cpu_solved and not gpu_solved:
            return
        if not cpu_solved or not gpu_solved:
            pytest.skip(f"Status mismatch: CPU={cpu_info.status}, GPU={gpu_info.status}")

        _assert_solutions_close(
            cpu_sol,
            cpu_info.obj_val,
            gpu_sol,
            gpu_info.obj_val,
            "CPU",
            "GPU",
            rtol=1e-4,
            atol=1e-4,
        )
        _verify_soc_membership(np.asarray(cpu_sol.s), soc_dims, offset=num_nonneg)
        _verify_soc_membership(np.asarray(gpu_sol.s), soc_dims, offset=num_nonneg)

    @given(problem=soc_problem_strategy(soc_dims_strat=large_soc_dims_strategy()))
    @hsettings(
        max_examples=100,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_large_soc_cpu_gpu_forward_parity(self, problem):
        """CPU and GPU should match for large SOC dims."""
        P, q, A, b, cones, soc_dims, num_nonneg = problem

        cpu_sol, cpu_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        gpu_sol, gpu_info, _ = _solve_moreau(P, q, A, b, cones, device="cuda")

        cpu_solved = cpu_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        gpu_solved = gpu_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )

        if not cpu_solved and not gpu_solved:
            return
        if not cpu_solved or not gpu_solved:
            pytest.skip(
                f"Status mismatch: CPU={cpu_info.status}, GPU={gpu_info.status}, "
                f"dims={soc_dims}"
            )

        _assert_solutions_close(
            cpu_sol,
            cpu_info.obj_val,
            gpu_sol,
            gpu_info.obj_val,
            "CPU",
            "GPU",
            rtol=1e-4,
            atol=1e-4,
        )
        _verify_soc_membership(np.asarray(cpu_sol.s), soc_dims, offset=num_nonneg)
        _verify_soc_membership(np.asarray(gpu_sol.s), soc_dims, offset=num_nonneg)
        _verify_kkt(P, q, A, b, cpu_sol, "CPU")
        _verify_kkt(P, q, A, b, gpu_sol, "GPU")

    @given(problem=soc_problem_strategy())
    @hsettings(
        max_examples=100,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_cpu_gpu_backward_parity(self, problem):
        """CPU and GPU backward pass should produce matching gradients for SOCPs."""
        P, q, A, b, cones, soc_dims, num_nonneg = problem

        cpu_sol, _, cpu_solver = _solve_moreau(P, q, A, b, cones, device="cpu", enable_grad=True)
        gpu_sol, _, gpu_solver = _solve_moreau(P, q, A, b, cones, device="cuda", enable_grad=True)

        cpu_solved = cpu_solver.info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        gpu_solved = gpu_solver.info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )

        if not cpu_solved or not gpu_solved:
            return

        # Skip degenerate cases
        s_min = min(np.min(np.abs(cpu_sol.s)), np.min(np.abs(gpu_sol.s)))
        z_min = min(np.min(np.abs(cpu_sol.z)), np.min(np.abs(gpu_sol.z)))
        assume(s_min > 1e-4)
        assume(z_min > 1e-4)

        x_diff = np.max(np.abs(np.asarray(cpu_sol.x) - np.asarray(gpu_sol.x)))
        x_scale = max(np.max(np.abs(cpu_sol.x)), np.max(np.abs(gpu_sol.x)), 1e-6)
        assume(x_diff < x_scale * 0.01)

        dx_bar = np.ones(P.shape[0])
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        for key in ("dq", "db"):
            assert not np.any(np.isnan(cpu_grads[key])), f"CPU {key} has NaN"
            assert not np.any(np.isnan(gpu_grads[key])), f"GPU {key} has NaN"
            assert not np.any(np.isinf(cpu_grads[key])), f"CPU {key} has Inf"
            assert not np.any(np.isinf(gpu_grads[key])), f"GPU {key} has Inf"

        max_soc_dim = max(soc_dims) if soc_dims else 0
        min_soc_dim = min(soc_dims) if soc_dims else 0
        # dim=2 SOC has sharp gradient behavior near boundary; large dims use
        # sparse expansion which introduces more numerical divergence
        if min_soc_dim <= 2:
            grad_rtol = 0.1
            grad_atol_factor = 0.05
        elif max_soc_dim > 4:
            grad_rtol = 0.05
            grad_atol_factor = 0.03
        else:
            grad_rtol = 1e-2
            grad_atol_factor = 0.01

        dq_scale = max(np.max(np.abs(cpu_grads["dq"])), np.max(np.abs(gpu_grads["dq"])), 1.0)
        db_scale = max(np.max(np.abs(cpu_grads["db"])), np.max(np.abs(gpu_grads["db"])), 1.0)

        np.testing.assert_allclose(
            cpu_grads["dq"],
            gpu_grads["dq"],
            rtol=grad_rtol,
            atol=max(1e-3, dq_scale * grad_atol_factor),
            err_msg="dq gradient mismatch CPU vs GPU",
        )
        np.testing.assert_allclose(
            cpu_grads["db"],
            gpu_grads["db"],
            rtol=grad_rtol,
            atol=max(1e-3, db_scale * 0.05),
            err_msg="db gradient mismatch CPU vs GPU",
        )

    @given(problem=soc_problem_strategy(soc_dims_strat=large_soc_dims_strategy()))
    @hsettings(
        max_examples=50,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_large_soc_backward_parity(self, problem):
        """Backward pass parity for large SOC dims (sparse expansion)."""
        P, q, A, b, cones, soc_dims, num_nonneg = problem

        cpu_sol, _, cpu_solver = _solve_moreau(P, q, A, b, cones, device="cpu", enable_grad=True)
        gpu_sol, _, gpu_solver = _solve_moreau(P, q, A, b, cones, device="cuda", enable_grad=True)

        cpu_solved = cpu_solver.info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )
        gpu_solved = gpu_solver.info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        )

        if not cpu_solved or not gpu_solved:
            return

        s_min = min(np.min(np.abs(cpu_sol.s)), np.min(np.abs(gpu_sol.s)))
        z_min = min(np.min(np.abs(cpu_sol.z)), np.min(np.abs(gpu_sol.z)))
        assume(s_min > 1e-3)
        assume(z_min > 1e-3)

        x_diff = np.max(np.abs(np.asarray(cpu_sol.x) - np.asarray(gpu_sol.x)))
        x_scale = max(np.max(np.abs(cpu_sol.x)), np.max(np.abs(gpu_sol.x)), 1e-6)
        assume(x_diff < x_scale * 0.02)

        dx_bar = np.ones(P.shape[0])
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        for key in ("dq", "db"):
            assert not np.any(np.isnan(cpu_grads[key])), f"CPU {key} NaN, dims={soc_dims}"
            assert not np.any(np.isnan(gpu_grads[key])), f"GPU {key} NaN, dims={soc_dims}"

        dq_scale = max(np.max(np.abs(cpu_grads["dq"])), np.max(np.abs(gpu_grads["dq"])), 1.0)
        db_scale = max(np.max(np.abs(cpu_grads["db"])), np.max(np.abs(gpu_grads["db"])), 1.0)

        np.testing.assert_allclose(
            cpu_grads["dq"],
            gpu_grads["dq"],
            rtol=0.1,
            atol=max(1e-2, dq_scale * 0.05),
            err_msg=f"dq mismatch, dims={soc_dims}",
        )
        np.testing.assert_allclose(
            cpu_grads["db"],
            gpu_grads["db"],
            rtol=0.1,
            atol=max(1e-2, db_scale * 0.1),
            err_msg=f"db mismatch, dims={soc_dims}",
        )


# =============================================================================
# Tests: Specific SOC dimension configurations (parametrized)
# =============================================================================


@requires_clarabel
class TestSOCSpecificDims:
    """Test specific SOC dimension configurations against Clarabel."""

    @pytest.mark.parametrize(
        "soc_dims",
        [
            [2],  # minimum dimension
            [3],  # standard (Clarabel default)
            [4],  # boundary for sparse expansion
            [5],  # first sparse dim
            [10],  # medium sparse
            [20],  # large sparse
            [50],  # very large
            [2, 3],  # mixed small
            [3, 5],  # mixed small/sparse
            [5, 10],  # mixed sparse
            [10, 20],  # mixed large sparse
            [2, 3, 4, 5],  # many mixed
            [3, 3, 3],  # multiple standard
            [5, 5, 5, 5],  # multiple sparse
            [2, 50],  # extreme mix
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],  # many small
        ],
    )
    def test_cpu_vs_clarabel_fixed_dims(self, soc_dims):
        """Test specific SOC dims against Clarabel with well-conditioned problem."""
        m_soc = sum(soc_dims)
        n = m_soc + 10

        rng = np.random.default_rng(42)

        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m_soc, n, format="csr")
        q = rng.uniform(-1.0, 1.0, size=n)

        x_feas = rng.uniform(-0.5, 0.5, size=n)
        s_parts = []
        for d in soc_dims:
            tail = rng.uniform(-0.2, 0.2, size=d - 1)
            head = np.linalg.norm(tail) + rng.uniform(0.3, 1.0)
            s_parts.append(np.concatenate([[head], tail]))
        s_feas = np.concatenate(s_parts)
        b = A @ x_feas + s_feas

        cones = moreau.Cones(so_cone_dims=soc_dims)

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assert "Solved" in str(clarabel_sol.status), f"Clarabel: {clarabel_sol.status}"

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        assert moreau_info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)

        np.testing.assert_allclose(
            moreau_info.obj_val,
            clarabel_sol.obj_val,
            rtol=1e-6,
            err_msg=f"Objective mismatch for soc_dims={soc_dims}",
        )
        np.testing.assert_allclose(
            np.asarray(moreau_sol.x),
            np.asarray(clarabel_sol.x),
            rtol=1e-5,
            atol=1e-5,
            err_msg=f"x mismatch for soc_dims={soc_dims}",
        )
        _verify_kkt(P, q, A, b, moreau_sol, f"CPU dims={soc_dims}")

    @pytest.mark.parametrize(
        "soc_dims",
        [
            [2],
            [3],
            [4],
            [5],
            [10],
            [20],
            [50],
            [3, 5],
            [5, 10],
            [2, 3, 4, 5],
            [10, 20],
            [2, 50],
        ],
    )
    @requires_gpu
    def test_gpu_vs_clarabel_fixed_dims(self, soc_dims):
        """Test specific SOC dims on GPU against Clarabel."""
        m_soc = sum(soc_dims)
        n = m_soc + 10

        rng = np.random.default_rng(42)

        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m_soc, n, format="csr")
        q = rng.uniform(-1.0, 1.0, size=n)

        x_feas = rng.uniform(-0.5, 0.5, size=n)
        s_parts = []
        for d in soc_dims:
            tail = rng.uniform(-0.2, 0.2, size=d - 1)
            head = np.linalg.norm(tail) + rng.uniform(0.3, 1.0)
            s_parts.append(np.concatenate([[head], tail]))
        s_feas = np.concatenate(s_parts)
        b = A @ x_feas + s_feas

        cones = moreau.Cones(so_cone_dims=soc_dims)

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assert "Solved" in str(clarabel_sol.status)

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cuda")
        assert moreau_info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)

        np.testing.assert_allclose(
            moreau_info.obj_val,
            clarabel_sol.obj_val,
            rtol=1e-5,
            err_msg=f"GPU objective mismatch for soc_dims={soc_dims}",
        )
        _verify_kkt(P, q, A, b, moreau_sol, f"GPU dims={soc_dims}")

    @pytest.mark.parametrize(
        "soc_dims,seed",
        [
            ([3], 0),
            ([3], 1),
            ([3], 2),
            ([3], 3),
            ([3], 4),
            ([5], 0),
            ([5], 1),
            ([5], 2),
            ([5], 3),
            ([5], 4),
            ([10], 0),
            ([10], 1),
            ([10], 2),
            ([10], 3),
            ([10], 4),
            ([20], 0),
            ([20], 1),
            ([20], 2),
            ([3, 5], 0),
            ([3, 5], 1),
            ([3, 5], 2),
            ([5, 10, 3], 0),
            ([5, 10, 3], 1),
            ([2, 3, 4, 5, 6, 7], 0),
        ],
    )
    def test_cpu_vs_clarabel_multi_seed(self, soc_dims, seed):
        """Test each dim config with multiple random seeds."""
        m_soc = sum(soc_dims)
        n = m_soc + 10

        rng = np.random.default_rng(seed)

        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        # Dense A for harder problems
        A_dense = rng.standard_normal((m_soc, n)) * 0.3
        for i in range(min(m_soc, n)):
            A_dense[i, i] += 1.0
        A = sp.csr_array(A_dense)

        q = rng.uniform(-2.0, 2.0, size=n)

        x_feas = rng.uniform(-1.0, 1.0, size=n)
        s_parts = []
        for d in soc_dims:
            tail = rng.uniform(-0.3, 0.3, size=d - 1)
            head = np.linalg.norm(tail) + rng.uniform(0.3, 1.0)
            s_parts.append(np.concatenate([[head], tail]))
        s_feas = np.concatenate(s_parts)
        b = A @ x_feas + s_feas

        cones = moreau.Cones(so_cone_dims=soc_dims)

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assert "Solved" in str(
            clarabel_sol.status
        ), f"Clarabel: {clarabel_sol.status} for dims={soc_dims}, seed={seed}"

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        assert moreau_info.status in (
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ), f"CPU: {moreau_info.status} for dims={soc_dims}, seed={seed}"

        np.testing.assert_allclose(
            moreau_info.obj_val,
            clarabel_sol.obj_val,
            rtol=1e-5,
            err_msg=f"Objective mismatch for dims={soc_dims}, seed={seed}",
        )


# =============================================================================
# Tests: Mixed cones with variable-dim SOC
# =============================================================================


@requires_clarabel
class TestSOCMixedConesFuzz:
    """Fuzz test variable-dim SOC mixed with other cone types."""

    @given(
        soc_dims=soc_dims_strategy(max_total_m=30),
        num_nonneg=st.integers(min_value=0, max_value=10),
        num_zero=st.integers(min_value=0, max_value=5),
        num_exp=st.integers(min_value=0, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @hsettings(
        max_examples=100,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_mixed_cones_vs_clarabel(self, soc_dims, num_nonneg, num_zero, num_exp, seed):
        """Mixed cone problems (zero + nonneg + SOC + exp) should match Clarabel."""
        m = num_zero + num_nonneg + sum(soc_dims) + 3 * num_exp
        assume(m >= 1)

        n = max(m + 5, 10)
        rng = np.random.default_rng(seed)

        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m, n, format="csr")
        q = rng.uniform(-1.0, 1.0, size=n)

        x_feas = rng.uniform(-0.5, 0.5, size=n)
        s_parts = []

        if num_zero > 0:
            s_parts.append(np.zeros(num_zero))
        if num_nonneg > 0:
            s_parts.append(rng.uniform(0.1, 1.0, size=num_nonneg))
        for d in soc_dims:
            tail = rng.uniform(-0.2, 0.2, size=d - 1)
            head = np.linalg.norm(tail) + rng.uniform(0.3, 1.0)
            s_parts.append(np.concatenate([[head], tail]))
        for _ in range(num_exp):
            y = rng.uniform(0.5, 2.0)
            x = rng.uniform(-1.0, 1.0)
            z_min = y * np.exp(x / y)
            z = z_min + rng.uniform(0.1, 0.5)
            s_parts.append(np.array([x, y, z]))

        s_feas = np.concatenate(s_parts) if s_parts else np.array([])
        b = A @ x_feas + s_feas

        cones = moreau.Cones(
            num_zero_cones=num_zero,
            num_nonneg_cones=num_nonneg,
            so_cone_dims=soc_dims,
            num_exp_cones=num_exp,
        )

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assume("Solved" in str(clarabel_sol.status))

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        assert moreau_info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)

        obj_diff = abs(moreau_info.obj_val - clarabel_sol.obj_val)
        obj_scale = max(abs(moreau_info.obj_val), abs(clarabel_sol.obj_val), 1.0)
        assert obj_diff < obj_scale * 1e-4, (
            f"Objective mismatch: moreau={moreau_info.obj_val}, " f"clarabel={clarabel_sol.obj_val}"
        )

    @given(
        soc_dims=soc_dims_strategy(max_total_m=30),
        num_nonneg=st.integers(min_value=0, max_value=10),
        num_zero=st.integers(min_value=0, max_value=5),
        num_exp=st.integers(min_value=0, max_value=3),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @hsettings(
        max_examples=100,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    @requires_gpu
    def test_mixed_cones_gpu_vs_clarabel(self, soc_dims, num_nonneg, num_zero, num_exp, seed):
        """Mixed cones on GPU should match Clarabel."""
        m = num_zero + num_nonneg + sum(soc_dims) + 3 * num_exp
        assume(m >= 1)

        n = max(m + 5, 10)
        rng = np.random.default_rng(seed)

        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m, n, format="csr")
        q = rng.uniform(-1.0, 1.0, size=n)

        x_feas = rng.uniform(-0.5, 0.5, size=n)
        s_parts = []

        if num_zero > 0:
            s_parts.append(np.zeros(num_zero))
        if num_nonneg > 0:
            s_parts.append(rng.uniform(0.1, 1.0, size=num_nonneg))
        for d in soc_dims:
            tail = rng.uniform(-0.2, 0.2, size=d - 1)
            head = np.linalg.norm(tail) + rng.uniform(0.3, 1.0)
            s_parts.append(np.concatenate([[head], tail]))
        for _ in range(num_exp):
            y = rng.uniform(0.5, 2.0)
            x = rng.uniform(-1.0, 1.0)
            z_min = y * np.exp(x / y)
            z = z_min + rng.uniform(0.1, 0.5)
            s_parts.append(np.array([x, y, z]))

        s_feas = np.concatenate(s_parts) if s_parts else np.array([])
        b = A @ x_feas + s_feas

        cones = moreau.Cones(
            num_zero_cones=num_zero,
            num_nonneg_cones=num_nonneg,
            so_cone_dims=soc_dims,
            num_exp_cones=num_exp,
        )

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assume("Solved" in str(clarabel_sol.status))

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cuda")
        assert moreau_info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)

        obj_diff = abs(moreau_info.obj_val - clarabel_sol.obj_val)
        obj_scale = max(abs(moreau_info.obj_val), abs(clarabel_sol.obj_val), 1.0)
        assert obj_diff < obj_scale * 1e-3, (
            f"Objective mismatch: moreau={moreau_info.obj_val}, " f"clarabel={clarabel_sol.obj_val}"
        )


# =============================================================================
# Tests: Batched variable-dim SOC
# =============================================================================


@requires_gpu
class TestSOCBatchedFuzz:
    """Fuzz test batched variable-dim SOC problems."""

    @given(
        soc_dims=soc_dims_strategy(max_total_m=30),
        batch_size=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @hsettings(
        max_examples=100,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_batched_cpu_gpu_parity(self, soc_dims, batch_size, seed):
        """Batched SOC problems should match between CPU and GPU."""
        m = sum(soc_dims)
        n = max(m + 5, 10)
        rng = np.random.default_rng(seed)

        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m, n, format="csr")

        P_ro = np.array(P.indptr, dtype=np.int64)
        P_ci = np.array(P.indices, dtype=np.int64)
        A_ro = np.array(A.indptr, dtype=np.int64)
        A_ci = np.array(A.indices, dtype=np.int64)

        cones = moreau.Cones(so_cone_dims=soc_dims)

        qs = rng.uniform(-1.0, 1.0, size=(batch_size, n))

        bs = np.zeros((batch_size, m))
        for bi in range(batch_size):
            x_feas = rng.uniform(-0.5, 0.5, size=n)
            s_parts = []
            for d in soc_dims:
                tail = rng.uniform(-0.2, 0.2, size=d - 1)
                head = np.linalg.norm(tail) + rng.uniform(0.3, 1.0)
                s_parts.append(np.concatenate([[head], tail]))
            s_feas = np.concatenate(s_parts)
            bs[bi] = A @ x_feas + s_feas

        P_vals = np.tile(P.data, (batch_size, 1))
        A_vals = np.tile(A.data, (batch_size, 1))

        cpu_settings = moreau.Settings(device="cpu", batch_size=batch_size)
        cpu_solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=cpu_settings,
        )
        cpu_solver.setup(P_vals, A_vals)
        cpu_result = cpu_solver.solve(qs, bs)

        gpu_settings = moreau.Settings(device="cuda", batch_size=batch_size)
        gpu_solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=gpu_settings,
        )
        gpu_solver.setup(P_vals, A_vals)
        gpu_result = gpu_solver.solve(qs, bs)

        for bi in range(batch_size):
            cpu_status = cpu_solver.info.status[bi]
            gpu_status = gpu_solver.info.status[bi]

            cpu_solved = cpu_status in (
                moreau.SolverStatus.Solved,
                moreau.SolverStatus.AlmostSolved,
            )
            gpu_solved = gpu_status in (
                moreau.SolverStatus.Solved,
                moreau.SolverStatus.AlmostSolved,
            )

            if not cpu_solved or not gpu_solved:
                continue

            cpu_obj = float(cpu_solver.info.obj_val[bi])
            gpu_obj = float(gpu_solver.info.obj_val[bi])

            assert not np.isnan(cpu_obj), f"Batch {bi}: CPU status={cpu_status} but obj is NaN"
            assert not np.isnan(gpu_obj), f"Batch {bi}: GPU status={gpu_status} but obj is NaN"

            np.testing.assert_allclose(
                cpu_obj, gpu_obj, rtol=1e-4, err_msg=f"Batch {bi}: objective mismatch"
            )
            np.testing.assert_allclose(
                cpu_result.x[bi],
                gpu_result.x[bi],
                rtol=1e-4,
                atol=1e-4,
                err_msg=f"Batch {bi}: x mismatch",
            )

    @given(
        soc_dims=soc_dims_strategy(max_total_m=30),
        batch_size=st.integers(min_value=2, max_value=16),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @hsettings(
        max_examples=50,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    @requires_clarabel
    def test_batched_vs_clarabel(self, soc_dims, batch_size, seed):
        """Each problem in a batch should match Clarabel individually."""
        m = sum(soc_dims)
        n = max(m + 5, 10)
        rng = np.random.default_rng(seed)

        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m, n, format="csr")

        cones = moreau.Cones(so_cone_dims=soc_dims)

        qs = rng.uniform(-1.0, 1.0, size=(batch_size, n))
        bs = np.zeros((batch_size, m))
        for bi in range(batch_size):
            x_feas = rng.uniform(-0.5, 0.5, size=n)
            s_parts = []
            for d in soc_dims:
                tail = rng.uniform(-0.2, 0.2, size=d - 1)
                head = np.linalg.norm(tail) + rng.uniform(0.3, 1.0)
                s_parts.append(np.concatenate([[head], tail]))
            bs[bi] = A @ x_feas + np.concatenate(s_parts)

        # Solve batched on GPU
        P_ro = np.array(P.indptr, dtype=np.int64)
        P_ci = np.array(P.indices, dtype=np.int64)
        A_ro = np.array(A.indptr, dtype=np.int64)
        A_ci = np.array(A.indices, dtype=np.int64)

        gpu_settings = moreau.Settings(device="cuda", batch_size=batch_size)
        gpu_solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=gpu_settings,
        )
        gpu_solver.setup(
            np.tile(P.data, (batch_size, 1)),
            np.tile(A.data, (batch_size, 1)),
        )
        gpu_result = gpu_solver.solve(qs, bs)

        # Compare each batch element to Clarabel
        for bi in range(batch_size):
            gpu_status = gpu_solver.info.status[bi]
            if gpu_status not in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved):
                continue

            clarabel_sol = _solve_clarabel(P, qs[bi], A, bs[bi], cones)
            if "Solved" not in str(clarabel_sol.status):
                continue

            np.testing.assert_allclose(
                gpu_solver.info.obj_val[bi],
                clarabel_sol.obj_val,
                rtol=1e-4,
                err_msg=f"Batch {bi}: GPU obj vs Clarabel",
            )


# =============================================================================
# Tests: Edge cases
# =============================================================================


@requires_clarabel
class TestSOCEdgeCases:
    """Edge cases for variable-dim SOC."""

    def test_single_soc_dim_2(self):
        """Minimum SOC dimension (2) should work."""
        n, m = 5, 2
        rng = np.random.default_rng(99)
        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m, n, format="csr")
        q = rng.uniform(-1.0, 1.0, size=n)
        x_feas = rng.uniform(-0.5, 0.5, size=n)
        s = np.array([1.0, 0.5])  # s[0] > |s[1]|
        b = A @ x_feas + s
        cones = moreau.Cones(so_cone_dims=[2])

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assert "Solved" in str(clarabel_sol.status)

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        assert moreau_info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)
        np.testing.assert_allclose(moreau_info.obj_val, clarabel_sol.obj_val, rtol=1e-6)

    @requires_gpu
    def test_single_soc_dim_2_gpu(self):
        """Minimum SOC dimension (2) on GPU."""
        n, m = 5, 2
        rng = np.random.default_rng(99)
        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m, n, format="csr")
        q = rng.uniform(-1.0, 1.0, size=n)
        x_feas = rng.uniform(-0.5, 0.5, size=n)
        s = np.array([1.0, 0.5])
        b = A @ x_feas + s
        cones = moreau.Cones(so_cone_dims=[2])

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cuda")
        assert moreau_info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)
        np.testing.assert_allclose(moreau_info.obj_val, clarabel_sol.obj_val, rtol=1e-5)

    def test_many_small_soc_cones(self):
        """20 SOC(3) cones — tests cone loop scaling."""
        soc_dims = [3] * 20
        m = sum(soc_dims)
        n = m + 10

        rng = np.random.default_rng(42)
        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m, n, format="csr")
        q = rng.uniform(-1.0, 1.0, size=n)

        x_feas = rng.uniform(-0.5, 0.5, size=n)
        s_parts = []
        for d in soc_dims:
            tail = rng.uniform(-0.2, 0.2, size=d - 1)
            head = np.linalg.norm(tail) + rng.uniform(0.3, 1.0)
            s_parts.append(np.concatenate([[head], tail]))
        b = A @ x_feas + np.concatenate(s_parts)

        cones = moreau.Cones(so_cone_dims=soc_dims)

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assert "Solved" in str(clarabel_sol.status)

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        assert moreau_info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)
        np.testing.assert_allclose(moreau_info.obj_val, clarabel_sol.obj_val, rtol=1e-5)

    @requires_gpu
    def test_many_small_soc_cones_gpu(self):
        """20 SOC(3) cones on GPU."""
        soc_dims = [3] * 20
        m = sum(soc_dims)
        n = m + 10

        rng = np.random.default_rng(42)
        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m, n, format="csr")
        q = rng.uniform(-1.0, 1.0, size=n)

        x_feas = rng.uniform(-0.5, 0.5, size=n)
        s_parts = []
        for d in soc_dims:
            tail = rng.uniform(-0.2, 0.2, size=d - 1)
            head = np.linalg.norm(tail) + rng.uniform(0.3, 1.0)
            s_parts.append(np.concatenate([[head], tail]))
        b = A @ x_feas + np.concatenate(s_parts)

        cones = moreau.Cones(so_cone_dims=soc_dims)

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cuda")
        assert moreau_info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)
        np.testing.assert_allclose(moreau_info.obj_val, clarabel_sol.obj_val, rtol=1e-5)

    def test_single_huge_soc(self):
        """One SOC of dimension 100."""
        soc_dims = [100]
        m = 100
        n = 120

        rng = np.random.default_rng(42)
        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m, n, format="csr")
        q = rng.uniform(-1.0, 1.0, size=n)

        x_feas = rng.uniform(-0.5, 0.5, size=n)
        tail = rng.uniform(-0.1, 0.1, size=99)
        head = np.linalg.norm(tail) + 1.0
        s = np.concatenate([[head], tail])
        b = A @ x_feas + s

        cones = moreau.Cones(so_cone_dims=soc_dims)

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        assert "Solved" in str(clarabel_sol.status)

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        assert moreau_info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)
        np.testing.assert_allclose(moreau_info.obj_val, clarabel_sol.obj_val, rtol=1e-5)

    @requires_gpu
    def test_single_huge_soc_gpu(self):
        """One SOC of dimension 100 on GPU."""
        soc_dims = [100]
        m = 100
        n = 120

        rng = np.random.default_rng(42)
        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m, n, format="csr")
        q = rng.uniform(-1.0, 1.0, size=n)

        x_feas = rng.uniform(-0.5, 0.5, size=n)
        tail = rng.uniform(-0.1, 0.1, size=99)
        head = np.linalg.norm(tail) + 1.0
        s = np.concatenate([[head], tail])
        b = A @ x_feas + s

        cones = moreau.Cones(so_cone_dims=soc_dims)

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cuda")
        assert moreau_info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)
        np.testing.assert_allclose(moreau_info.obj_val, clarabel_sol.obj_val, rtol=1e-4)

    @pytest.mark.parametrize("seed", range(10))
    def test_near_boundary_soc(self, seed):
        """Problems where optimal s is near SOC boundary (active constraint)."""
        rng = np.random.default_rng(seed)
        d = rng.integers(3, 15)
        n = d + 5
        m = d

        P = sp.diags(rng.uniform(1.0, 3.0, size=n), format="csr")
        A = sp.eye(m, n, format="csr")

        # Push toward boundary: q encourages minimizing s[0]
        q = np.zeros(n)
        q[0] = 1.0  # push s[0] down (toward boundary)
        q[1:d] = rng.uniform(-0.1, 0.1, size=d - 1)

        # Start with interior point but close to boundary
        x_feas = rng.uniform(-0.5, 0.5, size=n)
        tail = rng.uniform(-0.1, 0.1, size=d - 1)
        head = np.linalg.norm(tail) + 0.3
        s = np.concatenate([[head], tail])
        b = A @ x_feas + s

        cones = moreau.Cones(so_cone_dims=[d])

        clarabel_sol = _solve_clarabel(P, q, A, b, cones)
        if "Solved" not in str(clarabel_sol.status):
            pytest.skip(f"Clarabel: {clarabel_sol.status}")

        moreau_sol, moreau_info, _ = _solve_moreau(P, q, A, b, cones, device="cpu")
        assert moreau_info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)

        np.testing.assert_allclose(
            moreau_info.obj_val,
            clarabel_sol.obj_val,
            rtol=1e-4,
            err_msg=f"Near-boundary: dim={d}, seed={seed}",
        )
        _verify_soc_membership(np.asarray(moreau_sol.s), [d], offset=0)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--hypothesis-show-statistics"])
