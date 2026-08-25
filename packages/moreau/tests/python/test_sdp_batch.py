"""
Tests for PSD (SDP) cones in batch mode via CompiledSolver.

Covers:
- Forward solve: batch results match single-solver results
- Mixed cones (zero + PSD)
- Dense PSD cone (no exploitable sparsity = chordal is no-op)
- Sparse PSD cone large enough to trigger chordal decomposition
- Backward pass: finite-difference validation of dq, db, dP, dA gradients
"""

import numpy as np
import pytest
from scipy import sparse

import moreau

# PSD cones are now supported on both CPU and CUDA backends.

EPS_FD = 1e-5
TOL_FD = 1e-2


def _make_psd_svec(mat_dim, rng):
    """Generate a random PSD matrix and return its svec representation."""
    L = rng.standard_normal((mat_dim, mat_dim)) * 0.5
    M = L @ L.T + 0.1 * np.eye(mat_dim)
    svec = []
    for j in range(mat_dim):
        for i in range(j, mat_dim):
            if i == j:
                svec.append(M[i, j])
            else:
                svec.append(M[i, j] * np.sqrt(2))
    return np.array(svec)


def _make_sdp_batch_solver(
    n, m, P, A, cones, batch_size, enable_grad=False, tol=1e-9, b_sparsity_pattern=None
):
    settings = moreau.Settings(
        device="cpu", batch_size=batch_size, verbose=False, enable_grad=enable_grad
    )
    settings.ipm_settings.tol_gap_abs = tol
    settings.ipm_settings.tol_feas = tol
    solver = moreau.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=P.indptr.tolist(),
        P_col_indices=P.indices.tolist(),
        A_row_offsets=A.indptr.tolist(),
        A_col_indices=A.indices.tolist(),
        cones=cones,
        settings=settings,
        b_sparsity_pattern=b_sparsity_pattern,
    )
    solver.setup(P_values=P.data, A_values=A.data)
    return solver


def _single_solve(P, q, A, b, cones, tol=1e-9):
    settings = moreau.Settings(device="cpu", verbose=False)
    settings.ipm_settings.tol_gap_abs = tol
    settings.ipm_settings.tol_feas = tol
    single = moreau.Solver(P, q, A, b, cones, settings=settings)
    return single.solve()


class TestBatchSDPForward:
    """Batch forward solve with PSD cones."""

    def test_basic_batch_sdp(self):
        """Simple batched SDP: min ||x||^2 s.t. x + s = b, s in PSD(3)."""
        n, m, batch_size = 6, 6, 2
        P = sparse.eye(n, format="csr")
        A = sparse.eye(m, format="csr")
        cones = moreau.Cones(psd_dims=[3])

        solver = _make_sdp_batch_solver(n, m, P, A, cones, batch_size)

        qs = np.zeros((batch_size, n))
        bs = np.array(
            [
                [-3.0, 1.0, 4.0, 1.0, 2.0, 5.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            ]
        )

        solution = solver.solve(qs=qs, bs=bs)
        assert solution.x.shape == (batch_size, n)

        for i in range(batch_size):
            ref = _single_solve(P, qs[i], A, bs[i], cones)
            assert np.allclose(
                solution.x[i], ref.x, atol=1e-4
            ), f"Problem {i}: max diff = {np.max(np.abs(solution.x[i] - np.array(ref.x))):.2e}"

    def test_batch_sdp_parity_multi_problem(self):
        """Batch SDP with 4 different b vectors matches single-solver results."""
        rng = np.random.default_rng(42)
        n, m, batch_size = 6, 6, 4
        P = sparse.eye(n, format="csr")
        A = sparse.eye(m, format="csr")
        cones = moreau.Cones(psd_dims=[3])

        bs = np.array([_make_psd_svec(3, rng) for _ in range(batch_size)])
        qs = rng.standard_normal((batch_size, n)) * 0.1

        solver = _make_sdp_batch_solver(n, m, P, A, cones, batch_size)
        batch_sol = solver.solve(qs=qs, bs=bs)

        for i in range(batch_size):
            ref = _single_solve(P, qs[i], A, bs[i], cones)
            assert np.allclose(
                batch_sol.x[i], ref.x, atol=1e-4
            ), f"Problem {i}: max diff = {np.max(np.abs(batch_sol.x[i] - np.array(ref.x))):.2e}"

    def test_batch_sdp_mixed_cones(self):
        """Batch SDP with mixed cone types: zero + PSD(2)."""
        n, m, batch_size = 2, 4, 2
        P = sparse.eye(n, format="csr")
        A_dense = np.zeros((m, n))
        A_dense[0, 0] = 1.0
        A_dense[0, 1] = 1.0  # equality
        A_dense[1, 0] = 1.0
        A_dense[3, 1] = 1.0
        A = sparse.csr_matrix(A_dense)
        cones = moreau.Cones(num_zero_cones=1, psd_dims=[2])

        solver = _make_sdp_batch_solver(n, m, P, A, cones, batch_size)

        qs = np.zeros((batch_size, n))
        bs = np.array(
            [
                [1.0, 2.0, 0.0, 2.0],
                [2.0, 3.0, 0.0, 3.0],
            ]
        )

        solution = solver.solve(qs=qs, bs=bs)
        assert solution.x.shape == (batch_size, n)
        # Verify equality constraint
        for i in range(batch_size):
            assert abs(solution.x[i, 0] + solution.x[i, 1] - bs[i, 0]) < 1e-4

    def test_dense_psd_no_chordal(self):
        """Dense PSD(2) cone (no exploitable sparsity) — chordal should be a no-op."""
        n, m, batch_size = 3, 3, 2
        P = sparse.eye(n, format="csr")
        A = sparse.eye(m, format="csr")
        cones = moreau.Cones(psd_dims=[2])  # PSD(2) has 3 svec entries

        solver = _make_sdp_batch_solver(n, m, P, A, cones, batch_size)

        qs = np.zeros((batch_size, n))
        bs = np.array([[1.0, 0.0, 1.0], [2.0, 0.0, 2.0]])
        solution = solver.solve(qs=qs, bs=bs)
        assert solution.x.shape == (batch_size, n)

        for i in range(batch_size):
            ref = _single_solve(P, qs[i], A, bs[i], cones)
            assert np.allclose(solution.x[i], ref.x, atol=1e-4)

    def test_chordal_decomposition_batch(self):
        """Sparse PSD(6) cone with block-diagonal A triggers chordal decomposition."""
        mat_dim = 6
        svec_dim = mat_dim * (mat_dim + 1) // 2  # 21
        n, m, batch_size = svec_dim, svec_dim, 2

        # Build block-diagonal A: only touches entries within 3x3 diagonal blocks
        A_dense = np.eye(m)
        # Zero out cross-block entries
        idx = 0
        for j in range(mat_dim):
            for i in range(j, mat_dim):
                block_i = 0 if i < 3 else 1
                block_j = 0 if j < 3 else 1
                if block_i != block_j:
                    A_dense[idx, idx] = 0.0
                idx += 1
        A = sparse.csr_matrix(A_dense)
        A.eliminate_zeros()
        P = sparse.eye(n, format="csr")
        cones = moreau.Cones(psd_dims=[mat_dim])

        solver = _make_sdp_batch_solver(n, m, P, A, cones, batch_size)

        rng = np.random.default_rng(42)
        bs = np.array([_make_psd_svec(mat_dim, rng) for _ in range(batch_size)])
        # Zero out cross-block entries in b
        idx = 0
        for j in range(mat_dim):
            for i in range(j, mat_dim):
                block_i = 0 if i < 3 else 1
                block_j = 0 if j < 3 else 1
                if block_i != block_j:
                    bs[:, idx] = 0.0
                idx += 1
        qs = np.zeros((batch_size, n))

        solution = solver.solve(qs=qs, bs=bs)
        assert solution.x.shape == (batch_size, n)

        # Verify against single solver
        for i in range(batch_size):
            ref = _single_solve(P, qs[i], A, bs[i], cones)
            assert np.allclose(
                solution.x[i], ref.x, atol=1e-4
            ), f"Chordal batch vs single (problem {i}): max diff = {np.max(np.abs(solution.x[i] - np.array(ref.x))):.2e}"


def _make_inverse_matrix_problem(n_mat=3, seed=42):
    """Build a Schur complement SDP: find X s.t. [[A, I], [I, X]] >> 0.

    Uses the same variable-to-svec mapping that CVXPY produces.
    The PSD cone has block structure that triggers chordal decomposition
    when b_sparsity_pattern is provided, because b has nonzero entries in the
    A-block and identity-block positions (not just where A has nonzeros).

    Returns (n_vars, m, P, A, b, cones, b_sparsity_pattern).
    """
    rng = np.random.default_rng(seed)
    L = rng.standard_normal((n_mat, n_mat)) * 0.5
    A_mat = L @ L.T + 2.0 * np.eye(n_mat)

    mat_dim = 2 * n_mat
    svec_dim = mat_dim * (mat_dim + 1) // 2

    def svec_idx(i, j):
        if i < j:
            i, j = j, i
        return j * mat_dim - j * (j - 1) // 2 + (i - j)

    # CVXPY maps the n_mat*(n_mat+1)/2 free entries of symmetric X
    # into specific svec positions of the 2n x 2n block matrix.
    # The X block occupies rows/cols [n_mat..2*n_mat-1] of the block matrix.
    # Off-diagonal svec entries get coefficient -sqrt(2), diagonal get -1.
    n_vars = n_mat * (n_mat + 1) // 2
    rows, cols, vals = [], [], []
    x_idx = 0
    for col_x in range(n_mat):
        for row_x in range(col_x, n_mat):
            svec_row = svec_idx(n_mat + row_x, n_mat + col_x)
            rows.append(svec_row)
            cols.append(x_idx)
            if row_x == col_x:
                vals.append(-1.0)
            else:
                vals.append(-np.sqrt(2))
            x_idx += 1

    m = svec_dim
    A_sparse = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n_vars))
    P = sparse.csr_matrix((n_vars, n_vars))  # zero P (feasibility)

    # b contains the upper-left A_mat block and the off-diagonal identity block
    b = np.zeros(m)
    # Upper-left n_mat x n_mat block (A_mat)
    for col_b in range(n_mat):
        for row_b in range(col_b, n_mat):
            idx = svec_idx(row_b, col_b)
            if row_b == col_b:
                b[idx] = A_mat[row_b, col_b]
            else:
                b[idx] = A_mat[row_b, col_b] * np.sqrt(2)
    # Off-diagonal identity block at (n_mat+i, i)
    for i in range(n_mat):
        idx = svec_idx(n_mat + i, i)
        b[idx] = np.sqrt(2)

    cones = moreau.Cones(psd_dims=[mat_dim])
    b_sparsity_pattern = [abs(float(v)) > 0 for v in b]

    return n_vars, m, P, A_sparse, b, cones, b_sparsity_pattern


class TestChordalWithNonzeroB:
    """Chordal decomposition with nonzero b entries (inverse matrix / Schur complement)."""

    def test_inverse_matrix_single_solver(self):
        """moreau.Solver with Schur complement PSD triggers chordal decomp from b."""
        n_vars, m, P, A, b, cones, _ = _make_inverse_matrix_problem()
        solver = _single_solve(P, np.zeros(n_vars), A, b, cones)
        assert solver.x is not None

    def test_inverse_matrix_batch_with_b_sparsity_pattern(self):
        """CompiledSolver with b_sparsity_pattern enables chordal decomp at construction."""
        n_vars, m, P, A, b, cones, b_sparsity_pattern = _make_inverse_matrix_problem()
        batch_size = 2

        solver = _make_sdp_batch_solver(
            n_vars, m, P, A, cones, batch_size, b_sparsity_pattern=b_sparsity_pattern
        )
        qs = np.zeros((batch_size, n_vars))
        bs = np.tile(b, (batch_size, 1))
        batch_sol = solver.solve(qs=qs, bs=bs)
        assert batch_sol.x.shape == (batch_size, n_vars)

    def test_inverse_matrix_batch_matches_single(self):
        """Batch solve with chordal decomp matches single-solver results."""
        n_vars, m, P, A, b, cones, b_sparsity_pattern = _make_inverse_matrix_problem()
        batch_size = 2

        # Single solve
        ref = _single_solve(P, np.zeros(n_vars), A, b, cones)

        # Batch solve with b_sparsity_pattern
        solver = _make_sdp_batch_solver(
            n_vars, m, P, A, cones, batch_size, b_sparsity_pattern=b_sparsity_pattern
        )
        qs = np.zeros((batch_size, n_vars))
        bs = np.tile(b, (batch_size, 1))
        batch_sol = solver.solve(qs=qs, bs=bs)

        for i in range(batch_size):
            assert np.allclose(
                batch_sol.x[i], ref.x, atol=1e-4
            ), f"Problem {i}: max diff = {np.max(np.abs(batch_sol.x[i] - np.array(ref.x))):.2e}"

    def test_inverse_matrix_varied_b(self):
        """Batch with different A_mat (via b) all solve with chordal + convex objective."""
        batch_size = 4
        n_mat = 3
        mat_dim = 2 * n_mat

        n_vars, m, _, A, _, cones, b_sparsity_pattern = _make_inverse_matrix_problem(n_mat, seed=42)
        # Strongly convex objective makes the solution unique
        P = sparse.eye(n_vars, format="csr")

        def svec_idx(i, j):
            if i < j:
                i, j = j, i
            return j * mat_dim - j * (j - 1) // 2 + (i - j)

        # Generate different b vectors (different A_mat instances)
        bs = np.zeros((batch_size, m))
        for k in range(batch_size):
            rng_k = np.random.default_rng(100 + k)
            Lk = rng_k.standard_normal((n_mat, n_mat)) * 0.5
            Ak = Lk @ Lk.T + 2.0 * np.eye(n_mat)
            for col_b in range(n_mat):
                for row_b in range(col_b, n_mat):
                    idx = svec_idx(row_b, col_b)
                    bs[k, idx] = (
                        Ak[row_b, col_b] if row_b == col_b else Ak[row_b, col_b] * np.sqrt(2)
                    )
            for i in range(n_mat):
                idx = svec_idx(n_mat + i, i)
                bs[k, idx] = np.sqrt(2)

        solver = _make_sdp_batch_solver(
            n_vars, m, P, A, cones, batch_size, b_sparsity_pattern=b_sparsity_pattern
        )
        qs = np.zeros((batch_size, n_vars))
        batch_sol = solver.solve(qs=qs, bs=bs)

        # Verify each against single solver
        for i in range(batch_size):
            ref = _single_solve(P, qs[i], A, bs[i], cones)
            assert np.allclose(
                batch_sol.x[i], ref.x, atol=1e-3
            ), f"Problem {i}: max diff = {np.max(np.abs(batch_sol.x[i] - np.array(ref.x))):.2e}"


class TestBSparsityPatternEnablesDecomposition:
    """Verify that b_sparsity_pattern enables chordal decomposition that the
    conservative fallback would prevent.

    The conservative fallback (no b_sparsity_pattern) assumes all-zero-A rows
    have nonzero b, making the PSD variable appear denser and preventing
    decomposition. Providing the true (sparser) b pattern lets the solver
    decompose into smaller cliques.
    """

    @staticmethod
    def _make_block_diagonal_psd_problem(n_blocks=3, block_size=2):
        """Build a sparse PSD problem with block-diagonal structure.

        The PSD variable is (n_blocks * block_size) x (n_blocks * block_size),
        but only the diagonal blocks are touched by A. With b=0 everywhere,
        the true sparsity pattern is block-diagonal and chordally decomposable
        into n_blocks independent PSD(block_size) cones.

        Without b_sparsity_pattern, the conservative fallback marks all
        zero-A rows as having nonzero b, filling in the sparsity pattern
        and preventing decomposition.
        """
        mat_dim = n_blocks * block_size
        svec_dim = mat_dim * (mat_dim + 1) // 2

        def svec_idx(i, j):
            """Upper-triangular svec index for (i, j) with i >= j."""
            if i < j:
                i, j = j, i
            return j * mat_dim - j * (j - 1) // 2 + (i - j)

        # Decision variables: one per diagonal block entry
        # Each block has block_size*(block_size+1)/2 entries
        entries_per_block = block_size * (block_size + 1) // 2
        n_vars = n_blocks * entries_per_block

        rows, cols, vals = [], [], []
        var_idx = 0
        for blk in range(n_blocks):
            offset = blk * block_size
            for col_b in range(block_size):
                for row_b in range(col_b, block_size):
                    svec_row = svec_idx(offset + row_b, offset + col_b)
                    rows.append(svec_row)
                    cols.append(var_idx)
                    if row_b == col_b:
                        vals.append(-1.0)
                    else:
                        vals.append(-np.sqrt(2))
                    var_idx += 1

        m = svec_dim
        A_sparse = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n_vars))
        P = sparse.eye(n_vars, format="csr")  # Convex objective

        b = np.zeros(m)
        cones = moreau.Cones(psd_dims=[mat_dim])

        return n_vars, m, P, A_sparse, b, cones

    def test_with_true_b_sparsity_enables_decomposition(self):
        """b_sparsity_pattern=[False]*m (b=0) enables chordal decomposition
        of block-diagonal PSD structure."""
        n_vars, m, P, A, b, cones = self._make_block_diagonal_psd_problem(n_blocks=3, block_size=2)
        batch_size = 2

        # True pattern: b is all zero → no extra fill-in
        b_sparsity_pattern = [False] * m

        solver = _make_sdp_batch_solver(
            n_vars, m, P, A, cones, batch_size, b_sparsity_pattern=b_sparsity_pattern
        )

        rng = np.random.default_rng(123)
        qs = rng.standard_normal((batch_size, n_vars)) * 0.1
        bs = np.zeros((batch_size, m))
        batch_sol = solver.solve(qs=qs, bs=bs)

        # Verify against single solver
        for i in range(batch_size):
            ref = _single_solve(P, qs[i], A, bs[i], cones)
            assert np.allclose(
                batch_sol.x[i], ref.x, atol=1e-4
            ), f"Problem {i}: max diff = {np.max(np.abs(batch_sol.x[i] - np.array(ref.x))):.2e}"

    def test_without_b_sparsity_still_solves(self):
        """Conservative fallback (no b_sparsity_pattern) solves without error.
        Uses a strongly convex objective for unique solution."""
        n_vars, m, P, A, b, cones = self._make_block_diagonal_psd_problem(n_blocks=3, block_size=2)
        batch_size = 2

        # No b_sparsity_pattern → conservative fallback (no decomposition)
        solver = _make_sdp_batch_solver(n_vars, m, P, A, cones, batch_size, b_sparsity_pattern=None)

        rng = np.random.default_rng(123)
        qs = rng.standard_normal((batch_size, n_vars)) * 0.1
        bs = np.zeros((batch_size, m))
        batch_sol = solver.solve(qs=qs, bs=bs)

        # With b_sparsity_pattern → decomposition enabled
        solver_exact = _make_sdp_batch_solver(
            n_vars, m, P, A, cones, batch_size, b_sparsity_pattern=[False] * m
        )
        batch_sol_exact = solver_exact.solve(qs=qs, bs=bs)

        # Both paths should produce the same result (relaxed tolerance because
        # decomposed vs non-decomposed solve slightly different KKT systems)
        for i in range(batch_size):
            assert np.allclose(
                batch_sol.x[i], batch_sol_exact.x[i], atol=1e-3
            ), f"Problem {i}: max diff = {np.max(np.abs(batch_sol.x[i] - batch_sol_exact.x[i])):.2e}"

    def test_inverse_matrix_exact_b_pattern_matches_single(self):
        """Schur complement SDP: exact b pattern enables decomposition and
        matches single-solver results."""
        n_vars, m, P, A, b, cones, b_sparsity_pattern = _make_inverse_matrix_problem(n_mat=3)
        batch_size = 2
        qs = np.zeros((batch_size, n_vars))
        bs = np.tile(b, (batch_size, 1))

        # With exact b_sparsity_pattern → decomposition enabled
        solver_exact = _make_sdp_batch_solver(
            n_vars, m, P, A, cones, batch_size, b_sparsity_pattern=b_sparsity_pattern
        )
        sol_exact = solver_exact.solve(qs=qs, bs=bs)

        # Verify against single solver (which gets b directly and decomposes correctly)
        ref = _single_solve(P, qs[0], A, bs[0], cones)
        for i in range(batch_size):
            assert np.allclose(
                sol_exact.x[i], ref.x, atol=1e-4
            ), f"Exact pattern problem {i}: max diff = {np.max(np.abs(sol_exact.x[i] - np.array(ref.x))):.2e}"

    def test_larger_block_diagonal(self):
        """Larger block-diagonal PSD (4 blocks of size 3 = PSD(12)) with b=0.
        Conservative fallback fills the entire 12x12 matrix, preventing decomposition.
        True pattern reveals 4 independent PSD(3) blocks."""
        n_vars, m, P, A, b, cones = self._make_block_diagonal_psd_problem(n_blocks=4, block_size=3)
        batch_size = 2

        b_sparsity_pattern = [False] * m

        solver = _make_sdp_batch_solver(
            n_vars, m, P, A, cones, batch_size, b_sparsity_pattern=b_sparsity_pattern
        )

        rng = np.random.default_rng(456)
        qs = rng.standard_normal((batch_size, n_vars)) * 0.1
        bs = np.zeros((batch_size, m))
        batch_sol = solver.solve(qs=qs, bs=bs)

        for i in range(batch_size):
            ref = _single_solve(P, qs[i], A, bs[i], cones)
            assert np.allclose(
                batch_sol.x[i], ref.x, atol=1e-4
            ), f"Problem {i}: max diff = {np.max(np.abs(batch_sol.x[i] - np.array(ref.x))):.2e}"


class TestBatchSDPBackward:
    """Backward pass tests for batch SDP via CompiledSolver."""

    def _setup(self, batch_size, tol=1e-9):
        n, m = 6, 6
        P = sparse.eye(n, format="csr")
        A = sparse.eye(m, format="csr")
        cones = moreau.Cones(psd_dims=[3])
        solver = _make_sdp_batch_solver(n, m, P, A, cones, batch_size, enable_grad=True, tol=tol)
        return solver, n, m, P, A, cones

    def test_batch_sdp_backward_dq(self):
        """Finite-difference validation of dq gradient for batch SDP."""
        rng = np.random.default_rng(42)
        batch_size = 2
        solver, n, m, P, A, cones = self._setup(batch_size)

        q = rng.standard_normal(n) * 0.1
        b = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        qs = np.tile(q, (batch_size, 1))
        bs = np.tile(b, (batch_size, 1))

        solver.solve(qs=qs, bs=bs)

        rng2 = np.random.default_rng(100)
        dx_bar = rng2.standard_normal(n)
        dz_bar = rng2.standard_normal(m)
        ds_bar = np.zeros(m)

        grads = solver.backward(
            dx=[dx_bar] * batch_size,
            dz=[dz_bar] * batch_size,
            ds=[ds_bar] * batch_size,
        )

        dq_analytic = np.array(grads["dq"][0])
        dq_fd = np.zeros(n)
        for j in range(n):
            qs_p = qs.copy()
            qs_p[0, j] += EPS_FD
            qs_m = qs.copy()
            qs_m[0, j] -= EPS_FD
            sol_p = solver.solve(qs=qs_p, bs=bs)
            sol_m = solver.solve(qs=qs_m, bs=bs)
            dq_fd[j] = (
                dx_bar @ (sol_p.x[0] - sol_m.x[0])
                + dz_bar @ (sol_p.z[0] - sol_m.z[0])
                + ds_bar @ (sol_p.s[0] - sol_m.s[0])
            ) / (2 * EPS_FD)

        assert np.allclose(
            dq_analytic, dq_fd, atol=TOL_FD
        ), f"Batch SDP dq max diff: {np.max(np.abs(dq_analytic - dq_fd)):.2e}"

    def test_batch_sdp_backward_db(self):
        """Finite-difference validation of db gradient for batch SDP."""
        rng = np.random.default_rng(43)
        batch_size = 2
        solver, n, m, P, A, cones = self._setup(batch_size)

        q = rng.standard_normal(n) * 0.1
        b = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        qs = np.tile(q, (batch_size, 1))
        bs = np.tile(b, (batch_size, 1))

        solver.solve(qs=qs, bs=bs)

        rng2 = np.random.default_rng(101)
        dx_bar = rng2.standard_normal(n)
        dz_bar = rng2.standard_normal(m)
        ds_bar = np.zeros(m)

        grads = solver.backward(
            dx=[dx_bar] * batch_size,
            dz=[dz_bar] * batch_size,
            ds=[ds_bar] * batch_size,
        )

        db_analytic = np.array(grads["db"][0])
        db_fd = np.zeros(m)
        for j in range(m):
            bs_p = bs.copy()
            bs_p[0, j] += EPS_FD
            bs_m = bs.copy()
            bs_m[0, j] -= EPS_FD
            sol_p = solver.solve(qs=qs, bs=bs_p)
            sol_m = solver.solve(qs=qs, bs=bs_m)
            db_fd[j] = (
                dx_bar @ (sol_p.x[0] - sol_m.x[0])
                + dz_bar @ (sol_p.z[0] - sol_m.z[0])
                + ds_bar @ (sol_p.s[0] - sol_m.s[0])
            ) / (2 * EPS_FD)

        assert np.allclose(
            db_analytic, db_fd, atol=TOL_FD
        ), f"Batch SDP db max diff: {np.max(np.abs(db_analytic - db_fd)):.2e}"

    def test_batch_sdp_backward_dP(self):
        """Finite-difference validation of dP gradient for batch SDP."""
        rng = np.random.default_rng(44)
        batch_size = 1
        n, m = 6, 6
        P = sparse.eye(n, format="csr") * 2.0
        A = sparse.eye(m, format="csr")
        cones = moreau.Cones(psd_dims=[3])
        solver = _make_sdp_batch_solver(n, m, P, A, cones, batch_size, enable_grad=True)

        q = rng.standard_normal(n) * 0.1
        b = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
        qs = q.reshape(1, -1)
        bs = b.reshape(1, -1)

        solver.solve(qs=qs, bs=bs)

        rng2 = np.random.default_rng(102)
        dx_bar = rng2.standard_normal(n)
        dz_bar = rng2.standard_normal(m)
        ds_bar = np.zeros(m)

        grads = solver.backward(dx=[dx_bar], dz=[dz_bar], ds=[ds_bar])
        dP_analytic = np.array(grads["dP_values"][0])

        dP_fd = np.zeros_like(P.data)
        for k in range(len(P.data)):
            P_p = P.copy()
            P_p.data[k] += EPS_FD
            P_m = P.copy()
            P_m.data[k] -= EPS_FD
            solver.setup(P_values=P_p.data, A_values=A.data)
            sol_p = solver.solve(qs=qs, bs=bs)
            solver.setup(P_values=P_m.data, A_values=A.data)
            sol_m = solver.solve(qs=qs, bs=bs)
            dP_fd[k] = (
                dx_bar @ (sol_p.x[0] - sol_m.x[0])
                + dz_bar @ (sol_p.z[0] - sol_m.z[0])
                + ds_bar @ (sol_p.s[0] - sol_m.s[0])
            ) / (2 * EPS_FD)

        assert np.allclose(
            dP_analytic, dP_fd, atol=TOL_FD
        ), f"Batch SDP dP max diff: {np.max(np.abs(dP_analytic - dP_fd)):.2e}"
