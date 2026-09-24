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

Pin the documented slack row order (docs/guide/basic-usage.md, "Constraint
Ordering"): zero, nonneg, SOC, PSD, exp, power, generalized power.
"""

import numpy as np
import pytest
from scipy import sparse

import moreau

# One block per cone, listed in the documented row order. Two cones of the
# multi-instance types check that same-type cones stay in list order too.
_BLOCKS = [
    ("zero", 2, {"num_zero_cones": 2}),
    ("nonneg", 3, {"num_nonneg_cones": 3}),
    ("soc3", 3, {"so_cone_dims": [3]}),
    ("soc4", 4, {"so_cone_dims": [4]}),
    ("psd2", 3, {"psd_dims": [2]}),
    ("psd3", 6, {"psd_dims": [3]}),
    ("exp_a", 3, {"num_exp_cones": 1}),
    ("exp_b", 3, {"num_exp_cones": 1}),
    ("power_03", 3, {"power_alphas": [0.3]}),
    ("power_06", 3, {"power_alphas": [0.6]}),
    ("gen_power", 5, {"gen_power_cone_params": [([0.2, 0.3, 0.5], 2)]}),
]


def _combined_cones():
    kwargs = {}
    for _, _, spec in _BLOCKS:
        for key, value in spec.items():
            if isinstance(value, list):
                kwargs[key] = kwargs.get(key, []) + value
            else:
                kwargs[key] = kwargs.get(key, 0) + value
    return moreau.Cones(**kwargs)


def _project(c, cones, device):
    """x = Proj_K(c):  min 1/2||x - c||^2  s.t.  -x + s = 0, s in K."""
    dim = len(c)
    tol = 1e-10
    solver = moreau.Solver(
        sparse.eye(dim, format="csr"),
        -np.asarray(c, float),
        -sparse.eye(dim, format="csr"),
        np.zeros(dim),
        cones,
        moreau.Settings(
            device=device,
            solver="ipm",
            verbose=False,
            ipm_settings=moreau.IPMSettings(
                tol_gap_abs=tol,
                tol_gap_rel=tol,
                tol_feas=tol,
                chordal_decomposition_enable=False,
            ),
        ),
    )
    sol = solver.solve()
    assert solver.info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)
    return sol.x


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_documented_row_order(device, seed):
    """Stacking rows in the documented order projects each block onto its own cone.

    The projection onto a product cone is the block-wise projection. Each block
    reference comes from a single-cone solve, where row order cannot matter.
    If the backend interpreted rows in any other order, blocks would be
    projected onto the wrong cone and the comparison would fail.
    """
    rng = np.random.default_rng(seed)
    dims = [dim for _, dim, _ in _BLOCKS]
    c = rng.normal(size=sum(dims))

    combined = _project(c, _combined_cones(), device)

    offset = 0
    for name, dim, spec in _BLOCKS:
        block = slice(offset, offset + dim)
        reference = _project(c[block], moreau.Cones(**spec), "cpu")
        np.testing.assert_allclose(combined[block], reference, atol=1e-5, err_msg=name)
        offset += dim
