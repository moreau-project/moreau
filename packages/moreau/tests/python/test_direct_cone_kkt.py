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
