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

Solver and CompiledSolver factor the CPU KKT system the same way (#50).
"""

import numpy as np
import pytest
import scipy.sparse as sp

import moreau
from moreau._backend import _rank_method

# LP-cone problem (1 nonneg, SOC(2), exp; n=7, m=6, P=0). CompiledSolver used to
# resolve 'auto' to QDLDL, which returns NumericalError at iteration 1 here.
N, M = 7, 6
A_INDPTR = [0, 4, 5, 7, 10, 14, 17]
A_INDICES = [0, 2, 3, 4, 3, 0, 6, 1, 4, 5, 0, 1, 2, 5, 0, 2, 5]
A_DATA = [
    -2.1636742943438168,
    -0.456492661923377,
    -12.729568434349678,
    -1.1654351578815638,
    -1336.3515092926057,
    1.0134427222111277,
    -651.72121822529,
    0.0196326053348175,
    -0.09681822425269146,
    -64.89869093242976,
    0.16521179558807483,
    0.02632437792959761,
    0.05383480507157506,
    -162.11530692777032,
    0.04787525148810976,
    0.0495469995931264,
    42.15196828765506,
]
Q = np.array(
    [
        -2.0230472049143406,
        0.02609628324814776,
        -0.0677988705390978,
        2515.6620159552212,
        -0.18574508385335897,
        -90.81525564355267,
        1226.8555857352953,
    ]
)
B = np.array(
    [
        -8.057044893523672,
        -411.01072784487013,
        514.2074041000508,
        25.831627350143258,
        65.44311381319277,
        -16.88628085334989,
    ]
)
CONES = moreau.Cones(num_nonneg_cones=1, so_cone_dims=[2], num_exp_cones=1)
OPTIMUM = -158.42504408


def _compiled(**ipm):
    solver = moreau.CompiledSolver(
        n=N,
        m=M,
        P_row_offsets=[0] * (N + 1),
        P_col_indices=[],
        A_row_offsets=A_INDPTR,
        A_col_indices=A_INDICES,
        cones=CONES,
        settings=moreau.Settings(
            solver="ipm", device="cpu", batch_size=1, ipm_settings=moreau.IPMSettings(**ipm)
        ),
    )
    solver.setup(np.zeros(0), np.array(A_DATA))
    return solver


def test_compiled_solver_leaves_cpu_auto_to_rust():
    assert _compiled()._settings.ipm_settings.direct_solve_method == "auto"


def test_compiled_solver_matches_solver():
    solver = moreau.Solver(
        sp.csr_matrix((N, N)),
        Q,
        sp.csr_matrix((A_DATA, A_INDICES, A_INDPTR), shape=(M, N)),
        B,
        CONES,
        moreau.Settings(solver="ipm", device="cpu"),
    )
    solver.solve()
    compiled = _compiled()
    compiled.solve(Q[None, :], B[None, :])
    assert solver.info.status == moreau.SolverStatus.Solved
    assert compiled.info.status[0] == moreau.SolverStatus.Solved
    assert solver.info.obj_val == pytest.approx(OPTIMUM, rel=1e-8)
    assert compiled.info.obj_val[0] == pytest.approx(OPTIMUM, rel=1e-8)


def test_explicit_qdldl_is_still_honoured():
    assert (
        _compiled(direct_solve_method="qdldl")._settings.ipm_settings.direct_solve_method == "qdldl"
    )


@pytest.mark.parametrize("n, m", [(5, 10), (100, 200), (5000, 10000)])
def test_cpu_ranking_puts_faer_first(n, m):
    assert _rank_method("cpu", n, m, nnz_A=3 * m, batch_size=1)[0] == "faer"
