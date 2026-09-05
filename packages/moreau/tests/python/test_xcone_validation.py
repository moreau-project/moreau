"""Validation tests for direct-x cones (DirectConeSpec + Cones.dir_cones).

Covers constructor validation, overlap rejection, OOB index detection,
and degree accounting. Forward/backward correctness lives in
test_xcone_forward.py and test_xcone_backward.py; integration features
(warm-start, active-set, PSD chordal, Woodbury, asymmetric kinds,
zerocopy) live in test_xcone_misc.py.
"""

import numpy as np
import pytest
from scipy import sparse

import moreau

# ---------- DirectConeSpec validation ----------


def test_xconespec_nonneg_valid():
    spec = moreau.DirectConeSpec(kind="nonneg", indices=[3])
    assert spec.numel() == 1
    assert spec.kind == "nonneg"
    assert spec.indices == [3]
    assert spec.psd_k is None


def test_xconespec_soc_valid():
    spec = moreau.DirectConeSpec(kind="soc", indices=[0, 1, 2, 3])
    assert spec.numel() == 4


def test_xconespec_psd_valid():
    # k=3 => vech size 6
    spec = moreau.DirectConeSpec(kind="psd_triangle", indices=list(range(6)), psd_k=3)
    assert spec.numel() == 6


def test_xconespec_rejects_empty_indices():
    with pytest.raises(ValueError, match="non-empty"):
        moreau.DirectConeSpec(kind="nonneg", indices=[])


def test_xconespec_rejects_duplicate_indices():
    with pytest.raises(ValueError, match="duplicates"):
        moreau.DirectConeSpec(kind="nonneg", indices=[2, 2])


def test_xconespec_rejects_negative_indices():
    with pytest.raises(ValueError, match="non-negative"):
        moreau.DirectConeSpec(kind="nonneg", indices=[-1])


def test_xconespec_soc_rejects_size_one():
    with pytest.raises(ValueError, match="SOC x-cone requires >= 2"):
        moreau.DirectConeSpec(kind="soc", indices=[0])


def test_xconespec_psd_rejects_wrong_length():
    # k=3 needs 6 indices; pass 5.
    with pytest.raises(ValueError, match="requires 6 indices"):
        moreau.DirectConeSpec(kind="psd_triangle", indices=[0, 1, 2, 3, 4], psd_k=3)


def test_xconespec_psd_requires_psd_k():
    with pytest.raises(ValueError, match="psd_k must be >= 1"):
        moreau.DirectConeSpec(kind="psd_triangle", indices=[0, 1, 2])


def test_xconespec_nonneg_rejects_psd_k():
    with pytest.raises(ValueError, match="psd_k must be None"):
        moreau.DirectConeSpec(kind="nonneg", indices=[0], psd_k=2)


# ---------- Cones.dir_cones validation ----------


def test_cones_dir_cones_disjoint_ok():
    cones = moreau.Cones(
        dir_cones=[
            moreau.DirectConeSpec(kind="nonneg", indices=[0, 1]),
            moreau.DirectConeSpec(kind="soc", indices=[2, 3, 4]),
        ],
    )
    assert len(cones.dir_cones) == 2


def test_cones_dir_cones_overlapping_rejected():
    with pytest.raises(ValueError, match="appears in both"):
        moreau.Cones(
            dir_cones=[
                moreau.DirectConeSpec(kind="nonneg", indices=[1, 2]),
                moreau.DirectConeSpec(kind="nonneg", indices=[2, 3]),
            ],
        )


def test_cones_validate_dir_cone_indices_oob():
    cones = moreau.Cones(
        dir_cones=[moreau.DirectConeSpec(kind="nonneg", indices=[5])],
    )
    cones.validate_dir_cone_indices(n=6)  # OK
    with pytest.raises(ValueError, match=r"contains 5 which is >= n=5"):
        cones.validate_dir_cone_indices(n=5)


def test_cones_degree_includes_dir_cones():
    base = moreau.Cones(num_nonneg_cones=2).degree()
    with_x = moreau.Cones(
        num_nonneg_cones=2,
        dir_cones=[
            moreau.DirectConeSpec(kind="nonneg", indices=[0, 1, 2]),  # +3
            moreau.DirectConeSpec(kind="soc", indices=[3, 4, 5, 6]),  # +1
        ],
    ).degree()
    assert with_x == base + 4


def test_cones_degree_counts_each_x_cone_kind():
    base = moreau.Cones().degree()

    def deg(spec):
        return moreau.Cones(dir_cones=[spec]).degree() - base

    assert deg(moreau.DirectConeSpec(kind="nonneg", indices=[0, 1, 2])) == 3
    assert deg(moreau.DirectConeSpec(kind="soc", indices=[0, 1, 2])) == 1
    assert deg(moreau.DirectConeSpec(kind="psd_triangle", indices=list(range(6)), psd_k=3)) == 3
    assert deg(moreau.DirectConeSpec(kind="exp", indices=[0, 1, 2])) == 3
    assert deg(moreau.DirectConeSpec(kind="power", indices=[0, 1, 2], alpha=0.5)) == 3
    # gen_power degree = len(alphas) + 1
    assert (
        deg(moreau.DirectConeSpec(kind="gen_power", indices=[0, 1, 2, 3], alphas=[0.5, 0.5], dim2=2))
        == 3
    )


def test_solver_out_of_bounds_index_raises_before_backend():
    P = sparse.diags([1.0, 1.0], format="csr")
    q = np.array([1.0, 1.0])
    A = sparse.csr_array([[1.0, 1.0]])
    b = np.array([1.0])
    # n=2, but index 5 is OOB.
    cones = moreau.Cones(
        num_zero_cones=1,
        dir_cones=[moreau.DirectConeSpec(kind="nonneg", indices=[5])],
    )
    with pytest.raises(ValueError, match=r"contains 5 which is >= n=2"):
        moreau.Solver(P, q, A, b, cones=cones)
