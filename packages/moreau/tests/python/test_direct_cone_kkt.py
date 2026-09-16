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

import moreau
import numpy as np
import pytest
from scipy import sparse

from .cone_test_utils import planted_problem, slack_cones


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
