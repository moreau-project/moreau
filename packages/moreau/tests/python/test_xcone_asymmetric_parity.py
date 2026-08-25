"""CPU↔CUDA parity tests for asymmetric direct-x cones (Exp, Power, GenPower).

For each cone kind, solve the same problem on 'cpu' and 'cuda' (with the
**same** algorithm forced via ``solver='ipm'``) and assert that:
  - solver status agrees (both Solved or AlmostSolved)
  - primal solution x* agrees to within rtol=1e-4 / atol=1e-4
  - iteration count agrees exactly — both backends run the same IPM, so any
    drift indicates the CPU and CUDA code paths have diverged
    (η-correction tuning, KKT scatter, NT scaling, residual computation, …).

The ``solver='ipm'`` forcing is load-bearing: the default ``solver='auto'``
routes small QPs on CPU to the active-set solver, producing very different
iteration counts than CUDA's IPM. With IPM forced on both sides, the two
backends should agree iter-for-iter on well-conditioned problems.

All tests skip cleanly when CUDA is not available.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import moreau
from moreau._backend import device_available

PARITY_RTOL = 1e-4
PARITY_ATOL = 1e-4

COMMON_SETTINGS = dict(
    solver="ipm",
    ipm_settings=moreau.IPMSettings(
        presolve_enable=False,
        equilibrate_enable=False,
    ),
    verbose=False,
    max_iter=300,
)


@pytest.fixture(autouse=True)
def skip_no_cuda():
    if not device_available("cuda"):
        pytest.skip("CUDA backend not available")


def solve_on(device: str, P, q, A, b, cones, ipm_settings=None) -> tuple[np.ndarray, str, int]:
    """Solve a problem on the given device. Returns (x, status_name, iters)."""
    settings_kwargs = dict(COMMON_SETTINGS)
    if ipm_settings is not None:
        settings_kwargs = dict(settings_kwargs, ipm_settings=ipm_settings)
    settings = moreau.Settings(device=device, **settings_kwargs)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    sol = solver.solve()
    return sol.x, solver.info.status.name, int(solver.info.iterations)


def assert_parity(label, cpu, gpu, rtol=PARITY_RTOL, atol=PARITY_ATOL):
    x_cpu, st_cpu, it_cpu = cpu
    x_gpu, st_gpu, it_gpu = gpu
    assert st_cpu in ("Solved", "AlmostSolved"), f"{label} CPU status: {st_cpu}"
    assert st_gpu in ("Solved", "AlmostSolved"), f"{label} CUDA status: {st_gpu}"
    np.testing.assert_allclose(
        x_cpu,
        x_gpu,
        rtol=rtol,
        atol=atol,
        err_msg=f"{label} x* parity failed: cpu={x_cpu}, cuda={x_gpu}",
    )
    assert it_cpu == it_gpu, (
        f"{label} iter-count parity failed: cpu={it_cpu}, cuda={it_gpu}. "
        f"Both backends run the same IPM, so any drift indicates the CPU "
        f"and CUDA code paths have diverged (η correction, NT scaling, "
        f"KKT scatter, residual computation, …)."
    )


# ---------------------------------------------------------------------------
# ExpCone parity
# ---------------------------------------------------------------------------


def test_exp_direct_x_cpu_cuda_parity():
    """Exp direct-x: simple interior optimum."""
    n = 3
    P = sparse.eye(n, format="csr")
    q = np.array([0.0, -1.0, -5.0])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind="exp", indices=[0, 1, 2])],
    )

    cpu = solve_on("cpu", P, q, A, b, cones)
    gpu = solve_on("cuda", P, q, A, b, cones)
    assert_parity("Exp direct-x", cpu, gpu)


# ---------------------------------------------------------------------------
# PowerCone parity
# ---------------------------------------------------------------------------


def test_power_direct_x_cpu_cuda_parity():
    """Power direct-x: simple interior optimum."""
    n = 3
    alpha = 0.4
    P = sparse.eye(n, format="csr")
    q = np.array([-2.0, -3.0, -1.0])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind="power", indices=[0, 1, 2], alpha=alpha)],
    )

    cpu = solve_on("cpu", P, q, A, b, cones)
    gpu = solve_on("cuda", P, q, A, b, cones)
    assert_parity("Power direct-x", cpu, gpu)


def test_power_direct_x_boundary_active_parity():
    """Power direct-x: small P + objective on x[2] drives ψ→0."""
    n = 3
    alpha = 0.4
    P = sparse.eye(n, format="csr") * 1e-3
    q = np.array([-0.5, -0.5, 5.0])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind="power", indices=[0, 1, 2], alpha=alpha)],
    )

    cpu = solve_on("cpu", P, q, A, b, cones)
    gpu = solve_on("cuda", P, q, A, b, cones)
    assert_parity("Power direct-x boundary", cpu, gpu)


# ---------------------------------------------------------------------------
# GenPowerCone parity
# ---------------------------------------------------------------------------


def test_genpow_direct_x_cpu_cuda_parity():
    """GenPower direct-x: simple 3D interior optimum."""
    n = 3
    alphas = [0.4, 0.6]
    dim2 = 1
    P = sparse.eye(n, format="csr")
    q = np.array([-2.0, -3.0, -1.0])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        x_cones=[
            moreau.XConeSpec(
                kind="gen_power",
                indices=[0, 1, 2],
                alphas=alphas,
                dim2=dim2,
            )
        ],
    )

    cpu = solve_on("cpu", P, q, A, b, cones)
    gpu = solve_on("cuda", P, q, A, b, cones)
    assert_parity("GenPower direct-x", cpu, gpu)


def test_genpow_direct_x_stacked_parity():
    """GenPower direct-x: two stacked cones — exercises both K-cap pass-1/pass-2
    paths AND the dim-≤-4 dense Hs path, which carries the Mosek-Tunçel secant
    rank-2 terms enforcing `Hs·x = z_x`. Without those terms CUDA's dense Hs
    drifts (~7 iters: cpu=23 vs cuda=16) from the CPU iterate count."""
    n = 6
    P = sparse.eye(n, format="csr") * 0.01
    q = np.array([-0.5, -0.5, 5.0, -0.3, -0.3, 4.0])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        x_cones=[
            moreau.XConeSpec(kind="gen_power", indices=[0, 1, 2], alphas=[0.3, 0.7], dim2=1),
            moreau.XConeSpec(kind="gen_power", indices=[3, 4, 5], alphas=[0.4, 0.6], dim2=1),
        ]
    )

    cpu = solve_on("cpu", P, q, A, b, cones)
    gpu = solve_on("cuda", P, q, A, b, cones)
    assert_parity("GenPower direct-x stacked", cpu, gpu)


def test_genpow_direct_x_dim3_dense_path_parity():
    """GenPower direct-x at dim=3 (CUDA dense path, dim ≤ 4): the IPM
    iter count must match CPU iter-for-iter. Locks in the secant-fix
    commits (43068f11 + 146e0833) — pre-fix this case drifted by 5 iters
    (cpu=20 vs cuda=15) because CUDA's dense `update_xcones_genpow_scaling`
    branch built Hs as μ·(D + p·pᵀ − q·qᵀ − r·rᵀ) only, dropping CPU's
    `pd_scaling_nd_dense` secant rank-2 updates (z_x·z_xᵀ/⟨z_x,x⟩ +
    δz_x·δz_xᵀ/⟨δz_x,δx⟩) and the P_⊥ projection cleanup."""
    n = 3
    P = sparse.eye(n, format="csr") * 0.01
    q = np.array([-0.5, -0.5, 5.0])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        x_cones=[
            moreau.XConeSpec(kind="gen_power", indices=[0, 1, 2], alphas=[0.3, 0.7], dim2=1),
        ]
    )
    cpu = solve_on("cpu", P, q, A, b, cones)
    gpu = solve_on("cuda", P, q, A, b, cones)
    assert_parity("GenPower direct-x dim=3 dense", cpu, gpu)
