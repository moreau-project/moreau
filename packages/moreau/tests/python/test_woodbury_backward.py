"""
Test Woodbury backward pass on portfolio-type problems.

Compares Woodbury backward gradients against CuDSS backward gradients
on the same problem, and validates against finite differences.
"""

import numpy as np
import pytest
from scipy import sparse
import moreau


def _make_portfolio(n, k, seed=42):
    """Build a portfolio QP: min (1/2)x'Dx + q'x  s.t.  F'x = b_eq, x >= 0.

    Returns (P_csr, A_csr, q, b, cones) with diagonal P and [F'; -I] A.
    """
    rng = np.random.default_rng(seed)
    D = rng.uniform(0.1, 2.0, n)
    F = rng.standard_normal((n, k))
    q = -rng.uniform(0.0, 0.2, n)
    b_eq = np.zeros(k)
    b_eq[0] = 1.0  # budget

    P = sparse.diags(D, format="csr")
    A_top = sparse.csr_matrix(F.T)  # k × n (zero cones)
    A_bot = sparse.eye(
        n, format="csr"
    )  # n × n (nonneg cones, x >= 0: Ax+s=b → x+s=0 → A=-I, b=0... actually moreau convention is Ax+s=b with s≥0)
    # For x >= 0: use A = -I, b = 0, s = x ≥ 0
    A_bot = -sparse.eye(n, format="csr")
    A = sparse.vstack([A_top, A_bot], format="csr")

    m = k + n
    b = np.zeros(m)
    b[:k] = b_eq

    cones = moreau.Cones(num_zero_cones=k, num_nonneg_cones=n)
    return P, A, q, b, cones


def _solve_and_backward(
    P, A, q, b, cones, dx_bar, method, batch_size=1, diff_method="exact", diff_smoothing_mu=None
):
    """Solve and run backward pass, return dq."""
    n = P.shape[0]
    m = A.shape[0]

    ipm_kwargs = dict(direct_solve_method=method)
    if diff_method == "smoothed":
        ipm_kwargs["diff_method"] = "smoothed"
        ipm_kwargs["diff_smoothing_mu"] = diff_smoothing_mu or 1e-4
    # Tight tol so the forward iterate is identical across direct-solve
    # methods (Woodbury vs cuDSS) — at default 1e-8 the two backends can
    # land at points differing by ~1e-7 in `x`, and on the n=5000 k=50
    # challenge that propagates through the Schur complement to ~1e-2
    # absolute differences in `dq`. 1e-12 keeps them aligned to the
    # FD-comparable noise floor.
    ipm_kwargs["tol_gap_abs"] = 1e-12
    ipm_kwargs["tol_feas"] = 1e-12
    ipm = moreau.IPMSettings(**ipm_kwargs)
    settings = moreau.Settings(
        device="cuda",
        batch_size=batch_size,
        enable_grad=True,
        verbose=False,
        ipm_settings=ipm,
        max_iter=400,
    )

    solver = moreau.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=P.indptr.tolist(),
        P_col_indices=P.indices.tolist(),
        A_row_offsets=A.indptr.tolist(),
        A_col_indices=A.indices.tolist(),
        cones=cones,
        settings=settings,
    )

    # Shared P/A values across batch, per-batch q/b
    solver.setup(P_values=P.data, A_values=A.data)
    qs = np.tile(q, (batch_size, 1))
    bs = np.tile(b, (batch_size, 1))
    solution = solver.solve(qs=qs, bs=bs)

    dx_bar_batch = np.tile(dx_bar, (batch_size, 1))
    grads = solver.backward(dx=dx_bar_batch)
    dq = grads["dq"] if isinstance(grads, dict) else grads.dq
    return dq, solution


def _solve_only(P, A, q, b, cones, method):
    """Solve once and return x for finite-difference checks."""
    n = P.shape[0]
    m = A.shape[0]
    settings = moreau.Settings(
        device="cuda",
        batch_size=1,
        enable_grad=False,
        verbose=False,
        ipm_settings=moreau.IPMSettings(direct_solve_method=method),
    )
    solver = moreau.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=P.indptr.tolist(),
        P_col_indices=P.indices.tolist(),
        A_row_offsets=A.indptr.tolist(),
        A_col_indices=A.indices.tolist(),
        cones=cones,
        settings=settings,
    )
    solver.setup(P_values=P.data, A_values=A.data)
    solution = solver.solve(qs=[q], bs=[b])
    return np.asarray(solution.x).squeeze(0)


@pytest.fixture(
    params=[
        (10, 3),
        (20, 5),
        (50, 10),
    ]
)
def portfolio_problem(request):
    n, k = request.param
    rng = np.random.default_rng(123)
    P, A, q, b, cones = _make_portfolio(n, k)
    dx_bar = rng.standard_normal(n)
    return n, k, P, A, q, b, cones, dx_bar


@pytest.fixture(
    params=[
        # (n, k, condition_number, description)
        (30, 5, 1e4, "ill-conditioned P"),
        (20, 3, 1e6, "very ill-conditioned P"),
        (15, 5, 1e2, "k close to n/3"),
        (50, 2, 1e3, "high n/k ratio"),
        (1000, 30, 1e3, "large n=1000 k=30"),
        (5000, 50, 1e4, "large n=5000 k=50"),
    ]
)
def challenge_problem(request):
    """Portfolio problems with challenging conditioning."""
    n, k, cond, desc = request.param
    rng = np.random.default_rng(777)

    # P diagonal with large condition number
    D = np.logspace(-np.log10(cond) / 2, np.log10(cond) / 2, n)
    rng.shuffle(D)

    # Correlated factor loadings (near-collinear columns)
    base = rng.standard_normal((n, 1))
    F = base + 0.1 * rng.standard_normal((n, k))  # highly correlated

    q = -rng.uniform(0.01, 0.5, n)
    b_eq = np.zeros(k)
    b_eq[0] = 1.0
    # Add small RHS for other equality constraints to make problem non-trivial
    b_eq[1:] = rng.uniform(-0.1, 0.1, k - 1)

    P = sparse.diags(D, format="csr")
    A = sparse.vstack([sparse.csr_matrix(F.T), -sparse.eye(n, format="csr")], format="csr")
    m = k + n
    b = np.zeros(m)
    b[:k] = b_eq

    cones = moreau.Cones(num_zero_cones=k, num_nonneg_cones=n)
    dx_bar = rng.standard_normal(n)
    return n, k, P, A, q, b, cones, dx_bar, desc


@pytest.mark.cuda
class TestWoodburyBackwardChallenge:
    """Challenge problems with hard conditioning."""

    def test_exact_matches_cudss(self, challenge_problem):
        n, k, P, A, q, b, cones, dx_bar, desc = challenge_problem

        dq_wb, _ = _solve_and_backward(P, A, q, b, cones, dx_bar, "woodbury")
        dq_cudss, _ = _solve_and_backward(P, A, q, b, cones, dx_bar, "cudss")
        # Tolerance scales with k: Schur regularization is O(k * sqrt(eps_mach))
        tol = max(1e-5, k * 2e-6)
        np.testing.assert_allclose(
            dq_wb[0], dq_cudss[0], atol=tol, rtol=tol, err_msg=f"Challenge: {desc} (n={n}, k={k})"
        )

    @pytest.mark.flaky(reruns=3)
    def test_batched_consistent(self, challenge_problem):
        """Batched backward: all batch elements with identical data should match.

        Marked flaky because the forward solve (cuBLAS batched GEMMs) can produce
        slightly different rounding for batch>1, and for ill-conditioned Schur
        complements this amplifies into visible gradient differences.
        """
        n, k, P, A, q, b, cones, dx_bar, desc = challenge_problem
        batch_size = 3
        dq_wb, _ = _solve_and_backward(P, A, q, b, cones, dx_bar, "woodbury", batch_size=batch_size)
        tol = max(1e-5, k * 1e-6)
        for bi in range(1, batch_size):
            np.testing.assert_allclose(
                dq_wb[bi],
                dq_wb[0],
                atol=tol,
                rtol=tol,
                err_msg=f"Challenge batched: {desc} batch[{bi}] vs batch[0] (n={n}, k={k})",
            )


@pytest.mark.cuda
class TestWoodburyBackward:
    """Test Woodbury backward pass matches CuDSS."""

    def test_exact_matches_finite_difference(self):
        n, k = 8, 2
        P, A, q, b, cones = _make_portfolio(n, k, seed=321)
        rng = np.random.default_rng(654)
        dx_bar = rng.standard_normal(n)

        dq_wb, _ = _solve_and_backward(P, A, q, b, cones, dx_bar, "woodbury")

        eps = 1e-6
        fd = np.zeros(n)
        for j in range(n):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[j] += eps
            q_minus[j] -= eps
            x_plus = _solve_only(P, A, q_plus, b, cones, "woodbury")
            x_minus = _solve_only(P, A, q_minus, b, cones, "woodbury")
            fd[j] = dx_bar @ ((x_plus - x_minus) / (2.0 * eps))

        np.testing.assert_allclose(
            dq_wb[0],
            fd,
            atol=1e-6,
            rtol=1e-6,
            err_msg="Woodbury exact backward mismatch against finite differences",
        )

    def test_exact_matches_cudss(self, portfolio_problem):
        n, k, P, A, q, b, cones, dx_bar = portfolio_problem

        dq_wb, _ = _solve_and_backward(P, A, q, b, cones, dx_bar, "woodbury")
        dq_cudss, _ = _solve_and_backward(P, A, q, b, cones, dx_bar, "cudss")

        np.testing.assert_allclose(
            dq_wb[0],
            dq_cudss[0],
            atol=1e-7,
            rtol=1e-7,
            err_msg=f"Woodbury vs CuDSS exact diff mismatch (n={n}, k={k})",
        )

    def test_smoothed_matches_cudss(self, portfolio_problem):
        n, k, P, A, q, b, cones, dx_bar = portfolio_problem

        dq_wb, _ = _solve_and_backward(
            P, A, q, b, cones, dx_bar, "woodbury", diff_method="smoothed"
        )
        dq_cudss, _ = _solve_and_backward(
            P, A, q, b, cones, dx_bar, "cudss", diff_method="smoothed"
        )

        np.testing.assert_allclose(
            dq_wb[0],
            dq_cudss[0],
            atol=1e-6,
            rtol=1e-6,
            err_msg=f"Woodbury vs CuDSS smoothed diff mismatch (n={n}, k={k})",
        )

    def test_batched_matches_cudss(self, portfolio_problem):
        n, k, P, A, q, b, cones, dx_bar = portfolio_problem
        batch_size = 4

        dq_wb, _ = _solve_and_backward(P, A, q, b, cones, dx_bar, "woodbury", batch_size=batch_size)
        dq_cudss, _ = _solve_and_backward(P, A, q, b, cones, dx_bar, "cudss", batch_size=batch_size)

        for bi in range(batch_size):
            np.testing.assert_allclose(
                dq_wb[bi],
                dq_cudss[bi],
                atol=1e-7,
                rtol=1e-7,
                err_msg=f"Batched Woodbury vs CuDSS batch={bi} (n={n}, k={k})",
            )
