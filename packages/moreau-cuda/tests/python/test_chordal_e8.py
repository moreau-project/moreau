"""CUDA solves of the E8 program with chordal decomposition (#82).

E8 (data/membership-e8.npz) is a 2997×297 conic program with 1101
nonnegative rows, 126 second-order cones and PSD cones of orders 6 to 24,
with optimal objective 7.607973. With chordal decomposition, CUDA stopped
with InsufficientProgress, short of the optimum, in about 80% of its solves,
while the CPU solves it. CUDA's solves differ from run to run (its
floating-point reductions aren't deterministic), so the test repeats the
solve. Fixed, about 1 solve in 250 still stalls, so the test allows one stall
in 10 solves, which happens with probability below 1e-5 at an 80% stall rate.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

import moreau

pytestmark = pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")

def test_e8_chordal_rarely_stalls():
    D = np.load(Path(__file__).parent / "data" / "membership-e8.npz")
    A = sp.csr_array(
        sp.csc_matrix((D["A_data"], D["A_indices"], D["A_indptr"]), shape=tuple(D["A_shape"]))
    )
    c = D["c"]
    cones = moreau.Cones(
        num_zero_cones=int(D["zero"]),
        num_nonneg_cones=int(D["nonneg"]),
        so_cone_dims=[int(k) for k in D["soc"]],
        psd_dims=[int(k) for k in D["psd"]],
    )
    statuses = []
    for _ in range(10):
        solver = moreau.Solver(
            sp.csr_array((c.size, c.size)), c, A, D["b"], cones, moreau.Settings(device="cuda")
        )
        solver.solve()
        statuses.append(solver.info.status)
    solved = (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)
    assert sum(status not in solved for status in statuses) <= 1, statuses
