"""CPU↔CUDA parity tests for PSD (SDP) cones.

Closes a coverage gap from #133's review. The existing CUDA test suite
(test_psd.cpp, test_psd_fuzz.cpp, ...) checks KKT residual + backward
gradcheck via finite differences, but does not solve the same problem on
both backends and compare. Critical Rule #3 says CPU/GPU must agree to
tolerance, and the augmentation/reverse-mapping path (especially with
chordal decomposition) is exactly the kind of place where divergences
hide silently.

Each test solves a small PSD problem on both backends and asserts x and
obj_val agree.
"""

import numpy as np
import pytest

# CPU backend: lower-level CompiledSolver (the unified `moreau` Cones class
# does not yet expose `psd_dims` — that lands in #134, after which this test
# could move to the unified API).
moreau_cpu = pytest.importorskip(
    "moreau_cpu._cpu_solver", reason="moreau_cpu not built with sdp feature"
)

# CUDA backend: lower-level CompiledSolver. Skip if no GPU.
moreau_cuda = pytest.importorskip("moreau_cuda", reason="moreau_cuda not installed")
_CudaCones = moreau_cuda._CudaCones if hasattr(moreau_cuda, "_CudaCones") else None
pytestmark = pytest.mark.skipif(_CudaCones is None, reason="moreau_cuda._CudaCones not available")


# Tolerances. Bigger than the per-backend solver tolerance (1e-8) — CPU and
# CUDA take slightly different IPM paths (different KKT solver, different
# equilibration scaling rounding), so we expect agreement at the precision
# of the IPM termination criterion, not at machine epsilon.
TOL_X = 1e-5
TOL_OBJ = 1e-5


# ─── Helpers ────────────────────────────────────────────────────────────────


def _diag_csr(n):
    """CSR for an n×n diagonal matrix; returns (row_offsets, col_indices,
    values)."""
    return (np.arange(n + 1, dtype=np.int64), np.arange(n, dtype=np.int64), np.ones(n))


def _solve_cpu(n, m, P_ro, P_ci, A_ro, A_ci, P_vals, A_vals, q, b, psd_dims):
    cones = [moreau_cpu.PSDTriangleConeT(d) for d in psd_dims]
    settings = moreau_cpu.DefaultSettings()
    settings.verbose = False
    solver = moreau_cpu.CompiledSolver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings, 1, False)
    solver.setup([P_vals], [A_vals])
    solutions = solver.solve([q], [b])
    return solutions[0]


def _solve_cuda(n, m, P_ro, P_ci, A_ro, A_ci, P_vals, A_vals, q, b, psd_dims):
    cones = _CudaCones()
    cones.psd_cone_dims = list(psd_dims)
    solver = moreau_cuda.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=P_ro,
        P_col_indices=P_ci,
        A_row_offsets=A_ro,
        A_col_indices=A_ci,
        cones=cones,
        batch_size=1,
    )
    solver.setup(P_vals.reshape(1, -1), A_vals.reshape(1, -1))
    return solver.solve(q.reshape(1, -1), b.reshape(1, -1))


def _assert_parity(label, sol_cpu, sol_cuda):
    """sol_cpu is a moreau_cpu DefaultSolution; sol_cuda is whatever
    moreau_cuda.CompiledSolver.solve returns. Both expose .x and
    .status; CUDA result may be batched-shaped."""
    cpu_status_name = (
        sol_cpu.status.name if hasattr(sol_cpu.status, "name") else str(sol_cpu.status)
    )
    assert "Solved" in cpu_status_name, f"{label}: CPU did not solve ({cpu_status_name})"

    x_cpu = np.asarray(sol_cpu.x).ravel()
    x_cuda = np.asarray(sol_cuda.x).ravel()
    assert (
        x_cpu.shape == x_cuda.shape
    ), f"{label}: x shape {x_cpu.shape} (cpu) vs {x_cuda.shape} (cuda)"

    diff = np.max(np.abs(x_cpu - x_cuda))
    assert diff < TOL_X, (
        f"{label}: max |x_cpu - x_cuda| = {diff:.2e} (tol {TOL_X:.0e})\n"
        f"  cpu  x[:5] = {x_cpu[:5]}\n"
        f"  cuda x[:5] = {x_cuda[:5]}"
    )

    if hasattr(sol_cpu, "obj_val") and hasattr(sol_cuda, "obj_val"):
        obj_cpu = float(np.asarray(sol_cpu.obj_val).ravel()[0])
        obj_cuda = float(np.asarray(sol_cuda.obj_val).ravel()[0])
        assert abs(obj_cpu - obj_cuda) < TOL_OBJ, (
            f"{label}: |obj_cpu - obj_cuda| = " f"{abs(obj_cpu - obj_cuda):.2e} (tol {TOL_OBJ:.0e})"
        )


# ─── Tests ──────────────────────────────────────────────────────────────────


def test_parity_dense_psd3():
    """Dense PSD(3): no chordal decomposition path — exercises the basic
    spectral projection / scaling kernels."""
    n = m = 6
    P_ro, P_ci, P_vals = _diag_csr(n)
    A_ro, A_ci, A_vals = _diag_csr(m)
    b = np.array([3.0, 0.5 * np.sqrt(2), 3.0, 0.3 * np.sqrt(2), 0.4 * np.sqrt(2), 3.0])
    q = -b.copy()

    sol_cpu = _solve_cpu(n, m, P_ro, P_ci, A_ro, A_ci, P_vals, A_vals, q, b, [3])
    sol_cuda = _solve_cuda(n, m, P_ro, P_ci, A_ro, A_ci, P_vals, A_vals, q, b, [3])
    _assert_parity("dense_psd3", sol_cpu, sol_cuda)


def test_parity_dense_psd4():
    """Dense PSD(4): a slightly larger eigendecomposition."""
    n = m = 10
    P_ro, P_ci, P_vals = _diag_csr(n)
    A_ro, A_ci, A_vals = _diag_csr(m)
    b = np.zeros(m)
    for i in (0, 2, 5, 9):
        b[i] = 4.0
    for i in (1, 3, 4, 6, 7, 8):
        b[i] = 0.1 * np.sqrt(2)
    q = -0.5 * b

    sol_cpu = _solve_cpu(n, m, P_ro, P_ci, A_ro, A_ci, P_vals, A_vals, q, b, [4])
    sol_cuda = _solve_cuda(n, m, P_ro, P_ci, A_ro, A_ci, P_vals, A_vals, q, b, [4])
    _assert_parity("dense_psd4", sol_cpu, sol_cuda)


def test_parity_multi_psd():
    """Two separate PSD(3) cones. Tests the per-cone offsets / sort
    permutation; each cone is already at smallest decomposable form."""
    n = m = 12
    P_ro, P_ci, P_vals = _diag_csr(n)
    A_ro, A_ci, A_vals = _diag_csr(m)
    b1 = np.array([3.0, 0.1 * np.sqrt(2), 3.0, 0.05 * np.sqrt(2), 0.05 * np.sqrt(2), 3.0])
    b2 = np.array([2.5, 0.05 * np.sqrt(2), 2.5, 0.02 * np.sqrt(2), 0.02 * np.sqrt(2), 2.5])
    b = np.concatenate([b1, b2])
    q = -0.5 * b

    sol_cpu = _solve_cpu(n, m, P_ro, P_ci, A_ro, A_ci, P_vals, A_vals, q, b, [3, 3])
    sol_cuda = _solve_cuda(n, m, P_ro, P_ci, A_ro, A_ci, P_vals, A_vals, q, b, [3, 3])
    _assert_parity("multi_psd", sol_cpu, sol_cuda)


def test_parity_block_diag_psd6_chordal():
    """Block-diagonal PSD(6) = 2×PSD(3). Chordal decomposition fires on
    both backends — the case most likely to expose a divergence in the
    augmentation map or reverse-mapping path.

    We achieve block structure by giving A's cross-block svec rows zero
    columns AND zeroing b on those entries, which is what the chordal
    detection looks at.
    """
    mat_dim, block_dim = 6, 3
    n = m = mat_dim * (mat_dim + 1) // 2  # 21

    P_ro, P_ci, P_vals = _diag_csr(n)

    # A: identity rows on in-block svec entries, zero rows on cross-block
    in_block = []
    idx = 0
    for j in range(mat_dim):
        for i in range(j + 1):
            if i // block_dim == j // block_dim:
                in_block.append(idx)
            idx += 1
    A_ro = np.zeros(m + 1, dtype=np.int64)
    A_ci = np.zeros(len(in_block), dtype=np.int64)
    A_vals = np.ones(len(in_block))
    pos = 0
    for r in range(m):
        if r in in_block:
            A_ci[pos] = r
            pos += 1
            A_ro[r + 1] = A_ro[r] + 1
        else:
            A_ro[r + 1] = A_ro[r]

    # b: svec of a PD block-diagonal matrix. Cross-block entries stay 0.
    b = np.zeros(m)
    idx = 0
    for j in range(mat_dim):
        for i in range(j + 1):
            if i // block_dim == j // block_dim:
                if i == j:
                    b[idx] = 3.0
                else:
                    b[idx] = 0.1 * np.sqrt(2)
            idx += 1
    q = -0.5 * b

    sol_cpu = _solve_cpu(n, m, P_ro, P_ci, A_ro, A_ci, P_vals, A_vals, q, b, [mat_dim])
    sol_cuda = _solve_cuda(n, m, P_ro, P_ci, A_ro, A_ci, P_vals, A_vals, q, b, [mat_dim])
    _assert_parity("block_diag_psd6_chordal", sol_cpu, sol_cuda)


@pytest.mark.xfail(
    reason=(
        "#176: CUDA chordal merge currently hardcodes fill_in==0 parent-child "
        "(chordal_info.cpp), while CPU defaults to clique_graph (Garstka et "
        "al. 2019). The IPMSettings.chordal_decomposition_merge_method "
        "field is plumbed through to _CudaSettings but is NOT honored by "
        "CUDA — wiring requires an adapter between moreau::SuperNodeTree "
        "and moreau::chordal::SuperNodeTree (parallel implementations with "
        "incompatible VertexSet types). This test documents the divergence; "
        "remove the xfail when the adapter lands."
    ),
    strict=False,
)
def test_parity_banded_psd8_chordal_merge_strategy():
    """Banded PSD(8) (tridiagonal sparsity) — non-trivial chordal merge case.

    Regression for #176. For a banded pattern, CPU's clique_graph strategy
    and CUDA's hardcoded fill_in==0 parent-child produce different clique
    sets, so the two backends converge to different (Solved-status) iterates
    of fundamentally different reformulations.

    The existing block-diagonal parity test cannot catch this because every
    sensible merge collapses to the same outcome on perfectly separated
    blocks. A banded pattern (overlapping cliques along the diagonal) is the
    minimum nontrivial case.
    """
    mat_dim = 8
    n = m = mat_dim * (mat_dim + 1) // 2  # 36

    P_ro, P_ci, P_vals = _diag_csr(n)

    # Banded mask: keep diagonal entries (i,i) and immediate off-diagonals
    # (i,i+1). Both are active; the rest of the upper triangle is zero so
    # chordal detection sees a tridiagonal pattern that decomposes into
    # overlapping 2-cliques.
    banded = []
    idx = 0
    for j in range(mat_dim):
        for i in range(j + 1):
            if j - i <= 1:
                banded.append(idx)
            idx += 1

    A_ro = np.zeros(m + 1, dtype=np.int64)
    A_ci = np.zeros(len(banded), dtype=np.int64)
    A_vals = np.ones(len(banded))
    pos = 0
    for r in range(m):
        if r in banded:
            A_ci[pos] = r
            pos += 1
            A_ro[r + 1] = A_ro[r] + 1
        else:
            A_ro[r + 1] = A_ro[r]

    # b: svec of a PD tridiagonal matrix (strongly diagonal-dominant so the
    # projection has a well-defined unique solution). Off-band entries are 0.
    b = np.zeros(m)
    idx = 0
    for j in range(mat_dim):
        for i in range(j + 1):
            if i == j:
                b[idx] = 4.0
            elif j - i == 1:
                b[idx] = 0.2 * np.sqrt(2)
            idx += 1
    q = -0.5 * b

    sol_cpu = _solve_cpu(n, m, P_ro, P_ci, A_ro, A_ci, P_vals, A_vals, q, b, [mat_dim])
    sol_cuda = _solve_cuda(n, m, P_ro, P_ci, A_ro, A_ci, P_vals, A_vals, q, b, [mat_dim])
    _assert_parity("banded_psd8_chordal", sol_cpu, sol_cuda)
