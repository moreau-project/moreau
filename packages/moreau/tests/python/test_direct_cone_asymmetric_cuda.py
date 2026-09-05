"""Python-level integration tests for asymmetric direct-x cones (Exp, Power,
GenPower) on the CUDA backend.

Each test mirrors the corresponding C++ end-to-end test in
packages/moreau-cuda/tests/cpp/test_direct_cone_{exp,pow,genpow}.cpp
and cross-checks cone membership of the solution.

All tests skip cleanly when CUDA is not available.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import moreau
from moreau._backend import device_available


@pytest.fixture(autouse=True)
def skip_no_cuda():
    if not device_available("cuda"):
        pytest.skip("CUDA backend not available")


# ---------------------------------------------------------------------------
# Exp cone: min 0.5||x - (0,1,5)||^2  s.t. x ∈ ExpCone
# P = I, q = (0, -1, -5), no slack rows.
# ExpCone = {(x0, x1, x2) : x1 > 0, x0 <= x1 * exp(x2/x1)}.
# ---------------------------------------------------------------------------


def test_exp_direct_x_cuda_converges():
    """Exp direct-x on CUDA: solver converges and x lies in the exp cone."""
    n = 3
    P = sparse.eye(n, format="csr")
    q = np.array([0.0, -1.0, -5.0])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[moreau.DirectConeSpec(kind="exp", indices=[0, 1, 2])],
    )
    settings = moreau.Settings(device="cuda", verbose=False)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    sol = solver.solve()

    assert solver.info.status.name in (
        "Solved",
        "AlmostSolved",
    ), f"Unexpected status: {solver.info.status}"

    x = sol.x
    # ExpCone membership (Clarabel convention): x[1] > 0, x[2] > 0,
    # x[1] * log(x[2]/x[1]) >= x[0]  ⇔  x[2] >= x[1] * exp(x[0]/x[1]).
    assert x[1] > -1e-6, f"x[1]={x[1]} must be >= 0 for exp cone"
    assert x[2] > -1e-6, f"x[2]={x[2]} must be >= 0 for exp cone"
    if x[1] > 1e-10 and x[2] > 1e-10:
        assert x[2] >= x[1] * np.exp(x[0] / x[1]) - 1e-4, f"Exp cone violated: x={x}"


# ---------------------------------------------------------------------------
# Power cone: min 0.5||x - (2,3,1)||^2  s.t. x ∈ PowerCone(alpha=0.4)
# P = I, q = (-2, -3, -1), no slack rows.
# PowerCone(alpha) = {(x0, x1, x2) : |x2| <= x0^alpha * x1^(1-alpha), x0,x1>=0}.
# ---------------------------------------------------------------------------


def test_power_direct_x_cuda_converges():
    """Power direct-x on CUDA: solver converges and x lies in the power cone."""
    n = 3
    alpha = 0.4
    P = sparse.eye(n, format="csr")
    q = np.array([-2.0, -3.0, -1.0])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[moreau.DirectConeSpec(kind="power", indices=[0, 1, 2], alpha=alpha)],
    )
    settings = moreau.Settings(device="cuda", verbose=False)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    sol = solver.solve()

    assert solver.info.status.name in (
        "Solved",
        "AlmostSolved",
    ), f"Unexpected status: {solver.info.status}"

    x = sol.x
    # PowerCone: x[0] >= 0, x[1] >= 0, |x[2]| <= x[0]^alpha * x[1]^(1-alpha)
    assert x[0] >= -1e-6, f"x[0]={x[0]} must be >= 0 for power cone"
    assert x[1] >= -1e-6, f"x[1]={x[1]} must be >= 0 for power cone"
    p0 = max(x[0], 0.0)
    p1 = max(x[1], 0.0)
    rhs = p0**alpha * p1 ** (1 - alpha)
    assert (
        abs(x[2]) <= rhs + 1e-4
    ), f"Power cone violated: |x[2]|={abs(x[2]):.6f}, rhs={rhs:.6f}, x={x}"


# ---------------------------------------------------------------------------
# GenPower cone: min 0.5||x - (2,3,1)||^2  s.t. x ∈ GenPowerCone(α=[0.4,0.6], dim2=1)
# P = I (3x3), q = (-2, -3, -1), no slack rows.
# GenPowerCone = {(p,w) : prod(p_i^alpha_i) >= ||w||, p_i >= 0}.
# ---------------------------------------------------------------------------


def test_genpow_direct_x_cuda_converges():
    """GenPower direct-x on CUDA: solver converges and x lies in the gen-power cone."""
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
    settings = moreau.Settings(device="cuda", verbose=False)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    sol = solver.solve()

    assert solver.info.status.name in (
        "Solved",
        "AlmostSolved",
    ), f"Unexpected status: {solver.info.status}"

    x = sol.x
    dim1 = len(alphas)
    # GenPowerCone: p_i >= 0, prod(p_i^alpha_i) >= ||w||
    for i in range(dim1):
        assert x[i] >= -1e-6, f"x[{i}]={x[i]} must be >= 0 for gen-power cone"
    prod = float(np.prod([max(x[i], 0.0) ** alphas[i] for i in range(dim1)]))
    w_norm = float(np.linalg.norm(x[dim1 : dim1 + dim2]))
    assert (
        prod >= w_norm - 1e-4
    ), f"GenPower cone violated: prod={prod:.6f}, ||w||={w_norm:.6f}, x={x}"
