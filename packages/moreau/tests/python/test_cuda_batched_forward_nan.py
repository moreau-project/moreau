"""Reproducer for CUDA batched forward solve producing NaN.

Individual CUDA solves and CPU batched solves all succeed for these problems,
but the CUDA batched solver produces NaN solutions for some batch elements.

The SOC test (test_batched_soc_degenerate_no_nan) is a regression test for
GitHub issue #115: uninitialized shared memory in fused_double_quad_form_kernel
caused NaN in tau_den, which propagated through the IPM to corrupt other batches.
"""

import numpy as np
import pytest

import moreau


def _make_random_diag_dominant_P(rng, n):
    diag = rng.uniform(1.0, 5.0, size=n)
    P_ro = list(range(n + 1))
    P_ci = list(range(n))
    P_vals = diag.tolist()
    return P_ro, P_ci, P_vals


@pytest.fixture
def problem_data():
    """256 nonneg-cone QPs that trigger NaN in CUDA batched solve."""
    rng = np.random.default_rng(1111)
    n, m = 5, 5
    batch = 256

    P_ro, P_ci, P_vals = _make_random_diag_dominant_P(rng, n)
    A_ro = list(range(m + 1))
    A_ci = list(range(m))
    A_vals = [-1.0] * m
    cones = moreau.Cones(num_nonneg_cones=m)

    qs = rng.standard_normal((batch, n)) * 0.3
    bs = np.zeros((batch, m))

    return dict(
        n=n,
        m=m,
        batch=batch,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        qs=qs,
        bs=bs,
    )


@pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
class TestCUDABatchedNaN:

    def test_batched_vs_cpu(self, problem_data):
        """CUDA batched solve should match CPU — no NaN."""
        d = problem_data
        ipm = moreau.IPMSettings(tol_gap_abs=1e-9, tol_feas=1e-9)

        # CPU batched (reference)
        cpu_settings = moreau.Settings(
            device="cpu",
            batch_size=d["batch"],
            ipm_settings=ipm,
        )
        cpu_solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_ro"],
            P_col_indices=d["P_ci"],
            A_row_offsets=d["A_ro"],
            A_col_indices=d["A_ci"],
            cones=d["cones"],
            settings=cpu_settings,
        )
        cpu_solver.setup(P_values=d["P_vals"], A_values=d["A_vals"])
        cpu_sol = cpu_solver.solve(qs=d["qs"], bs=d["bs"])
        x_cpu = np.asarray(cpu_sol.x)
        assert np.all(np.isfinite(x_cpu)), "CPU solve should not produce NaN"

        # CUDA batched
        cuda_settings = moreau.Settings(
            device="cuda",
            batch_size=d["batch"],
            ipm_settings=ipm,
        )
        cuda_solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_ro"],
            P_col_indices=d["P_ci"],
            A_row_offsets=d["A_ro"],
            A_col_indices=d["A_ci"],
            cones=d["cones"],
            settings=cuda_settings,
        )
        cuda_solver.setup(P_values=d["P_vals"], A_values=d["A_vals"])
        cuda_sol = cuda_solver.solve(qs=d["qs"], bs=d["bs"])
        x_cuda = np.asarray(cuda_sol.x)

        nan_mask = ~np.isfinite(x_cuda).all(axis=1)
        assert not nan_mask.any(), (
            f"CUDA batched solve produced NaN in {nan_mask.sum()}/{d['batch']} "
            f"rows: {np.where(nan_mask)[0].tolist()}"
        )

    def test_individual_solves_succeed(self, problem_data):
        """Each problem succeeds individually on CUDA — proves it's a batching bug."""
        d = problem_data
        ipm = moreau.IPMSettings(tol_gap_abs=1e-9, tol_feas=1e-9)

        # First find which rows fail in batched mode
        cuda_settings = moreau.Settings(
            device="cuda",
            batch_size=d["batch"],
            ipm_settings=ipm,
        )
        cuda_solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_ro"],
            P_col_indices=d["P_ci"],
            A_row_offsets=d["A_ro"],
            A_col_indices=d["A_ci"],
            cones=d["cones"],
            settings=cuda_settings,
        )
        cuda_solver.setup(P_values=d["P_vals"], A_values=d["A_vals"])
        cuda_sol = cuda_solver.solve(qs=d["qs"], bs=d["bs"])
        x_cuda = np.asarray(cuda_sol.x)
        nan_rows = np.where(~np.isfinite(x_cuda).all(axis=1))[0]

        if len(nan_rows) == 0:
            pytest.skip("No NaN rows in batched solve (bug not triggered)")

        # Solve each NaN row individually
        for r in nan_rows:
            single_settings = moreau.Settings(
                device="cuda",
                batch_size=1,
                ipm_settings=ipm,
            )
            single_solver = moreau.CompiledSolver(
                n=d["n"],
                m=d["m"],
                P_row_offsets=d["P_ro"],
                P_col_indices=d["P_ci"],
                A_row_offsets=d["A_ro"],
                A_col_indices=d["A_ci"],
                cones=d["cones"],
                settings=single_settings,
            )
            single_solver.setup(P_values=d["P_vals"], A_values=d["A_vals"])
            sol = single_solver.solve(
                qs=[d["qs"][r].tolist()],
                bs=[d["bs"][r].tolist()],
            )
            x = np.asarray(sol.x)
            assert np.all(
                np.isfinite(x)
            ), f"Row {r} also fails individually — not purely a batching bug"


@pytest.fixture
def soc_problem_data():
    """Batched SOC QPs with degenerate cones (optimal s at cone tip).

    Reproduces GitHub issue #115: n=43, m=57 with 7 SOC cones of dim 3.
    With n not a multiple of the warp size (32), the shared-memory reduction
    in fused_double_quad_form_kernel read uninitialized values, producing NaN.
    """
    rng = np.random.default_rng(42)
    n, m = 43, 57
    batch = 100

    # Diagonal P (like 2*I from sum_squares objective)
    P_ro = list(range(n + 1))
    P_ci = list(range(n))
    P_vals = [2.0] * n

    # Sparse A with random structure (nnz ~ 9 per row)
    A_rows, A_cols, A_data = [], [], []
    for i in range(m):
        nnz_row = rng.integers(3, min(n, 12))
        cols = sorted(rng.choice(n, size=nnz_row, replace=False))
        for c in cols:
            A_rows.append(i)
            A_cols.append(int(c))
            A_data.append(rng.standard_normal())
    # Build CSR
    A_ro = [0]
    cur = 0
    for i in range(m):
        count = sum(1 for r in A_rows if r == i)
        cur += count
        A_ro.append(cur)
    A_ci = A_cols
    A_vals = A_data

    # 15 zero + 21 nonneg + 7 SOC(3)
    cones = moreau.Cones(
        num_zero_cones=15,
        num_nonneg_cones=21,
        so_cone_dims=[3, 3, 3, 3, 3, 3, 3],
    )

    # Random q and b across batch (q ≈ 0, b has structure that makes some
    # SOC cones degenerate at the tip)
    qs = rng.standard_normal((batch, n)) * 0.01
    bs = rng.standard_normal((batch, m)) * 0.5

    return dict(
        n=n,
        m=m,
        batch=batch,
        P_ro=P_ro,
        P_ci=P_ci,
        P_vals=P_vals,
        A_ro=A_ro,
        A_ci=A_ci,
        A_vals=A_vals,
        cones=cones,
        qs=qs,
        bs=bs,
    )


@pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
class TestCUDABatchedSOCNaN:

    def test_batched_soc_degenerate_no_nan(self, soc_problem_data):
        """Batched SOC solve must not produce NaN (regression for #115).

        The root cause was uninitialized shared memory in
        fused_double_quad_form_kernel when n (43) is not a multiple of
        the block size (256). Inactive warps skipped __syncthreads__
        and left sh[] containing garbage that corrupted tau_den.
        """
        d = soc_problem_data
        settings = moreau.Settings(device="cuda", batch_size=d["batch"])
        solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_ro"],
            P_col_indices=d["P_ci"],
            A_row_offsets=d["A_ro"],
            A_col_indices=d["A_ci"],
            cones=d["cones"],
            settings=settings,
        )
        solver.setup(P_values=d["P_vals"], A_values=d["A_vals"])

        # Run multiple times — the bug was non-deterministic
        for trial in range(20):
            sol = solver.solve(qs=d["qs"], bs=d["bs"])
            x = np.asarray(sol.x)
            nan_mask = ~np.isfinite(x).all(axis=1)
            assert not nan_mask.any(), f"Trial {trial}: NaN in {nan_mask.sum()}/{d['batch']} rows"
