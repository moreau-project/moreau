"""
Copyright, the Moreau authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import warnings
from types import SimpleNamespace

import moreau
import numpy as np
import pytest
from scipy import sparse

from .cone_test_utils import CONE_CASES, planted_problem, slack_cones


@pytest.mark.parametrize("equilibrate", [False, True])
@pytest.mark.parametrize("equality_rows", [False, True])
@pytest.mark.parametrize("perturbation", ["exact", "dual", "primal", "gap"])
def test_boundary_warm_start_respects_tolerances(device, equilibrate, equality_rows, perturbation):
    # x* = (0, 1), z_x* = 1. Perturbations are below the 1e-6 smoothing
    # floor but above the requested tolerance, and must not strand the IPM.
    P = sparse.diags([2.0, 4.0], format="csr")
    q = np.array([1.0, -4.0])
    A = sparse.csr_matrix([[0.0, 1.0]]) if equality_rows else sparse.csr_matrix((0, 2))
    b = np.ones(A.shape[0])
    warm = moreau.WarmStart(
        x=np.array([0.0, 1.0]),
        z=np.zeros(len(b)),
        s=np.zeros(len(b)),
        z_x=np.ones(1),
    )
    if perturbation == "dual":
        warm.z_x[0] += 5e-7
    elif perturbation == "primal":
        warm.x[1] += 5e-7
    elif perturbation == "gap":
        warm.x[0] += 5e-7
        warm.z_x[0] += 1e-6  # Preserve stationarity while perturbing complementarity.
    solver = moreau.Solver(
        P,
        q,
        A,
        b,
        cones=moreau.Cones(
            num_zero_cones=len(b),
            dir_cones=[moreau.DirectConeSpec(kind="nonneg", indices=[0])],
        ),
        settings=moreau.Settings(
            device=device,
            solver="ipm",
            verbose=False,
            ipm_settings=moreau.IPMSettings(
                equilibrate_enable=equilibrate,
                presolve_enable=False,
                tol_feas=1e-10,
                tol_gap_abs=1e-10,
                tol_gap_rel=1e-10,
            ),
        ),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # A cold retry must not hide a failed warm solve.
        sol = solver.solve(warm_start=warm)
    assert solver.info.status.name == "Solved"
    assert (solver.info.iterations == 0) == (perturbation == "exact")
    np.testing.assert_allclose(sol.x, [0.0, 1.0], atol=2e-9, rtol=0)
    np.testing.assert_allclose(P @ sol.x + q + A.T @ sol.z, [sol.z_x[0], 0.0], atol=2e-9)
    assert abs(sol.x[0] * sol.z_x[0]) < 2e-9


@pytest.mark.parametrize(
    "cases", [(c,) for c in CONE_CASES] + [CONE_CASES], ids=[*CONE_CASES, "all"]
)
@pytest.mark.parametrize("equilibrate", [False, True])
def test_known_kkt_and_external_warm_start(device, cases, equilibrate):
    problem = planted_problem(cases)
    solver = problem.solver(device, equilibrate)
    sol = solver.solve()
    assert solver.info.status.name == "Solved"
    problem.check(sol)

    # A short budget rules out silently ignoring the supplied state.
    # Warnings fail here because Solver otherwise retries failed warm starts cold.
    warm_solver = problem.solver(device, equilibrate, max_iter=5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warm = warm_solver.solve(warm_start=problem.optimum)
    assert warm_solver.info.status.name in ("Solved", "AlmostSolved")
    assert warm_solver.info.iterations < solver.info.iterations
    problem.check(warm)


@pytest.mark.parametrize("equilibrate", [False, True])
def test_all_cones_match_slack_representation(device, equilibrate):
    problem = planted_problem()
    direct_solver = problem.solver(device, equilibrate)
    sol = direct_solver.solve()
    problem.check(sol)

    # Interleave old and newly introduced rows within each canonical cone family.
    blocks, rhs, specs, expected_z = [problem.A[:1]], [problem.b[:1]], [], [sol.z[:1]]
    offset, direct_offset = 1, 0
    for spec in problem.cones.dir_cones:
        dim = len(spec.indices)
        blocks.extend(
            [
                problem.A[offset : offset + dim],
                -sparse.eye(len(sol.x), format="csr")[spec.indices],
            ]
        )
        rhs.extend([problem.b[offset : offset + dim], np.zeros(dim)])
        specs.extend([spec, spec])
        expected_z.extend(
            [sol.z[offset : offset + dim], sol.z_x[direct_offset : direct_offset + dim]]
        )
        offset += dim
        direct_offset += dim
    slack_solver = moreau.Solver(
        problem.P,
        problem.q,
        sparse.vstack(blocks, format="csr"),
        np.concatenate(rhs),
        cones=slack_cones(specs, zero=1),
        settings=direct_solver._settings,
    )
    reference = slack_solver.solve()
    assert slack_solver.info.status.name == "Solved"
    np.testing.assert_allclose(reference.x, sol.x, atol=3e-4, rtol=3e-4)
    np.testing.assert_allclose(reference.z, np.concatenate(expected_z), atol=3e-4, rtol=3e-4)


@pytest.mark.parametrize("output", ["x", "s", "z", "z_x"])
def test_all_cones_directional_derivatives(device, output):
    problem = planted_problem()
    ipm = {"diff_method": "exact", "tol_gap_abs": 1e-11, "tol_gap_rel": 1e-11, "tol_feas": 1e-11}
    solver = problem.solver(device, enable_grad=True, ipm_options=ipm)
    sol = solver.solve()
    problem.check(sol)
    rng = np.random.default_rng(42)
    upstream = rng.normal(size=getattr(sol, output).shape)
    args = {"d" + output: upstream}
    args.setdefault("dx", np.zeros_like(sol.x))
    grad = solver.backward(**args)

    # Separate directions identify which data pullback failed; off-diagonal P
    # perturbations are symmetric, matching the public full-CSR convention.
    H = rng.normal(size=problem.P.shape)
    directions = {
        "P": 0.03 * (H + H.T),
        "A": rng.normal(scale=0.03, size=problem.A.shape),
        "q": rng.normal(size=problem.q.shape),
        "b": rng.normal(size=problem.b.shape),
    }
    # At smaller steps the boundary solve error can dominate the difference.
    # A 1e-2 step retains agreement with the analytic pullback on CPU and CUDA.
    eps = 1e-2
    for name, direction in directions.items():
        value = getattr(problem, name)
        losses = []
        for sign in (-1, 1):
            perturbation = sparse.csr_matrix(direction) if name in ("P", "A") else direction
            setattr(problem, name, value + sign * eps * perturbation)
            perturbed = problem.solver(device, ipm_options=ipm).solve()
            losses.append(upstream @ getattr(perturbed, output))
        setattr(problem, name, value)
        key = "d" + name + ("_values" if name in ("P", "A") else "")
        analytic = grad[key] @ direction.ravel() if name in ("P", "A") else grad[key] @ direction
        np.testing.assert_allclose(
            analytic,
            (losses[1] - losses[0]) / (2 * eps),
            atol=2e-3,
            rtol=3e-3,
            err_msg=f"{output} wrt {name}",
        )


def test_all_cones_batched_setup_and_warm_reuse(device):
    problem = planted_problem()
    solver = moreau.CompiledSolver(
        n=len(problem.q),
        m=len(problem.b),
        P_row_offsets=problem.P.indptr,
        P_col_indices=problem.P.indices,
        A_row_offsets=problem.A.indptr,
        A_col_indices=problem.A.indices,
        cones=problem.cones,
        settings=moreau.Settings(
            device=device,
            solver="ipm",
            batch_size=2,
            verbose=False,
            ipm_settings=moreau.IPMSettings(chordal_decomposition_enable=False),
        ),
    )
    previous = None
    for factors in ((0.7, 1.4), (1.8, 0.4)):
        problems = [planted_problem() for _ in factors]
        for p, factor in zip(problems, factors):
            p.P *= factor
            p.A /= factor
            p.q = -p.P @ p.optimum.x - p.A.T @ p.optimum.z
            indices = np.concatenate([c.indices for c in p.cones.dir_cones])
            p.q[indices] += p.optimum.z_x
            p.b = p.A @ p.optimum.x + p.optimum.s
        solver.setup(
            np.stack([p.P.data for p in problems]),
            np.stack([p.A.data for p in problems]),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sol = solver.solve(
                np.stack([p.q for p in problems]),
                np.stack([p.b for p in problems]),
                warm_start=previous,
            )
        for i, p in enumerate(problems):
            assert solver.info.status[i].name in ("Solved", "AlmostSolved")
            p.check(
                SimpleNamespace(**{name: getattr(sol, name)[i] for name in ("x", "s", "z", "z_x")})
            )
        previous = sol.to_warm_start()


@pytest.mark.parametrize("cone_kind", ["soc", "gen_power"])
@pytest.mark.parametrize("diff_method", ["auto", "exact"])
@pytest.mark.parametrize("equilibrate", [False, True])
def test_sparse_inactive_and_active_blocks(device, diff_method, equilibrate, cone_kind):
    """Sparse-only direct cones must populate E even without any dense H blocks."""
    if cone_kind == "gen_power":
        n = 8
        q = np.linspace(-1.7, 0.6, n)
        specs = [
            moreau.DirectConeSpec(
                kind="gen_power", indices=[0, 1, 2, 6], alphas=[0.2, 0.3, 0.5], dim2=1
            ),
            moreau.DirectConeSpec(
                kind="gen_power", indices=[3, 4, 5, 7], alphas=[0.3, 0.3, 0.4], dim2=1
            ),
        ]
    else:
        n = 12
        q = np.empty(n)
        q[::2] = [-2.0, 0.1, -0.1, 0.2, -0.2, 0.3]
        q[1::2] = [0.2, 0.4, -0.8, 0.6, 0.3, -0.5]
        specs = [
            moreau.DirectConeSpec(kind="soc", indices=list(range(start, n, 2))) for start in (0, 1)
        ]
    P = sparse.eye(n, format="csr")
    A = sparse.csr_matrix((0, n))
    b = np.zeros(0)
    cones = moreau.Cones(dir_cones=specs)
    settings = moreau.Settings(
        device=device,
        solver="ipm",
        enable_grad=True,
        verbose=False,
        ipm_settings=moreau.IPMSettings(
            diff_method=diff_method,
            equilibrate_enable=equilibrate,
            tol_gap_abs=1e-10,
            tol_gap_rel=1e-10,
            tol_feas=1e-10,
        ),
    )
    solver = moreau.Solver(P, q, A, b, cones, settings)
    sol = solver.solve()
    np.testing.assert_allclose(sol.z_x[: n // 2], 0, atol=1e-6)
    assert np.linalg.norm(sol.z_x[n // 2 :]) > 0.1
    # Check every Jacobian entry through both primal and direct-dual outputs.
    analytic = {
        out: np.stack(
            [
                solver.backward(
                    dx=np.eye(n)[i] if out == "x" else np.zeros(n),
                    dz_x=np.eye(n)[i] if out == "z_x" else None,
                )["dq"]
                for i in range(n)
            ]
        )
        for out in ("x", "z_x")
    }
    for j in range(n):
        eps = 1e-3
        plus = moreau.Solver(P, q + eps * np.eye(n)[j], A, b, cones, settings).solve()
        minus = moreau.Solver(P, q - eps * np.eye(n)[j], A, b, cones, settings).solve()
        for out in analytic:
            fd = (getattr(plus, out) - getattr(minus, out)) / (2 * eps)
            np.testing.assert_allclose(
                analytic[out][:, j], fd, atol=5e-4, rtol=5e-3, err_msg=f"{out} wrt q[{j}]"
            )
