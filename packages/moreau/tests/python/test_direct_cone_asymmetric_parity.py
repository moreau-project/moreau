"""CPU↔CUDA parity tests for asymmetric direct-x cones (Exp, Power, GenPower).

For each cone kind, solve the same problem on 'cpu' and 'cuda' (with the
**same** algorithm forced via ``solver='ipm'``) and assert that:
  - solver status agrees (both Solved or AlmostSolved)
  - primal solution x* agrees to within rtol=1e-4 / atol=1e-4

CPU and CUDA use different KKT solvers and refinement stopping policies, so
matching iteration counts is not a correctness requirement. The dense GenPower
secant identities are tested directly by DenseScalingSatisfiesBothSecants in
moreau-cuda/tests/cpp/test_direct_cone_genpow.cpp.

The ``solver='ipm'`` forcing ensures both sides exercise the interior-point
implementation rather than selecting a different algorithm on CPU.

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


def solve_on(device: str, P, q, A, b, cones, ipm_settings=None) -> tuple[np.ndarray, str]:
    """Solve a problem on the given device. Returns (x, status_name)."""
    settings_kwargs = dict(COMMON_SETTINGS)
    if ipm_settings is not None:
        settings_kwargs = dict(settings_kwargs, ipm_settings=ipm_settings)
    settings = moreau.Settings(device=device, **settings_kwargs)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    sol = solver.solve()
    return sol.x, solver.info.status.name


def assert_parity(label, cpu, gpu, rtol=PARITY_RTOL, atol=PARITY_ATOL):
    x_cpu, st_cpu = cpu
    x_gpu, st_gpu = gpu
    assert st_cpu in ("Solved", "AlmostSolved"), f"{label} CPU status: {st_cpu}"
    assert st_gpu in ("Solved", "AlmostSolved"), f"{label} CUDA status: {st_gpu}"
    np.testing.assert_allclose(
        x_cpu,
        x_gpu,
        rtol=rtol,
        atol=atol,
        err_msg=f"{label} x* parity failed: cpu={x_cpu}, cuda={x_gpu}",
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
        dir_cones=[moreau.DirectConeSpec(kind="exp", indices=[0, 1, 2])],
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
        dir_cones=[moreau.DirectConeSpec(kind="power", indices=[0, 1, 2], alpha=alpha)],
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
        dir_cones=[moreau.DirectConeSpec(kind="power", indices=[0, 1, 2], alpha=alpha)],
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
        dir_cones=[
            moreau.DirectConeSpec(
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
    """Two stacked cones exercise the K-cap passes and dense Hs path."""
    n = 6
    P = sparse.eye(n, format="csr") * 0.01
    q = np.array([-0.5, -0.5, 5.0, -0.3, -0.3, 4.0])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[
            moreau.DirectConeSpec(kind="gen_power", indices=[0, 1, 2], alphas=[0.3, 0.7], dim2=1),
            moreau.DirectConeSpec(kind="gen_power", indices=[3, 4, 5], alphas=[0.4, 0.6], dim2=1),
        ]
    )

    cpu = solve_on("cpu", P, q, A, b, cones)
    gpu = solve_on("cuda", P, q, A, b, cones)
    assert_parity("GenPower direct-x stacked", cpu, gpu)


def test_genpow_direct_x_dim3_dense_path_parity():
    """Boundary-active solution parity through the dense (dim <= 4) Hs path."""
    n = 3
    P = sparse.eye(n, format="csr") * 0.01
    q = np.array([-0.5, -0.5, 5.0])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[
            moreau.DirectConeSpec(kind="gen_power", indices=[0, 1, 2], alphas=[0.3, 0.7], dim2=1),
        ]
    )
    cpu = solve_on("cpu", P, q, A, b, cones)
    gpu = solve_on("cuda", P, q, A, b, cones)
    assert_parity("GenPower direct-x dim=3 dense", cpu, gpu)
