"""Regression test: active-set solver must not silently return infeasible iterates
on ill-scaled QPs.

Pre-fix: DAQP's QP→LDP transform suffers catastrophic cancellation when
|b·scale| << |M·v|, losing the original `b` in floating-point noise. DAQP
correctly reports OPTIMAL for the noise-driven transformed problem, and the
un-transformed x violates Ax ≤ b. Moreau used to report status=Solved.

Post-fix (Ruiz equilibration + post-solve cone-feasibility check):
either the returned iterate satisfies the nonneg cone (s >= 0), or the
status is downgraded to NumericalError — never `Solved` with s < 0.
"""

import numpy as np
import pytest
from scipy import sparse

moreau = pytest.importorskip("moreau")


def _build_ill_scaled_lp(seed: int = 0):
    """The original failing case: P=1e-6 I, q=1e6, A=1e6, b=O(1), nonneg cone.

    The unique feasible-optimal has s_i ≈ some positive slack; pre-fix the
    active-set solver returned Solved with min(s) ≈ -873.
    """
    rng = np.random.default_rng(seed)
    n, m = 5, 8
    L = rng.standard_normal((n, n)) * 0.3
    P = (L @ L.T + np.eye(n)) * 1e-6
    q = rng.standard_normal(n) * 1e6
    A = rng.standard_normal((m, n)) * 0.5 * 1e6
    x_star = rng.standard_normal(n) * 0.1
    s_star = rng.uniform(0.5, 1.5, m)
    b = (A @ x_star + s_star) * 1e-6
    return P, q, A, b, n, m


def test_active_set_ill_scaled_lp_never_silent_violation():
    P, q, A, b, n, m = _build_ill_scaled_lp(seed=0)
    cones = moreau.Cones(num_nonneg_cones=m)
    settings = moreau.Settings(solver="active_set", device="cpu", verbose=False)
    solver = moreau.Solver(
        sparse.csr_matrix(P), q, sparse.csr_matrix(A), b, cones, settings=settings
    )
    sol = solver.solve()
    status = solver.info.status

    primal_tol = 1e-6 * max(1.0, float(np.max(np.abs(b))))

    if status == moreau.SolverStatus.Solved:
        # If Solved, iterate MUST lie in the nonneg cone.
        assert float(np.min(sol.s)) >= -primal_tol, (
            f"status=Solved but min(s) = {float(np.min(sol.s))!r} violates the nonneg cone "
            f"beyond tol={primal_tol:.3e}; this is the regression the Ruiz + post-solve "
            f"feasibility check fixes."
        )
    else:
        # Otherwise the solver must loudly report an error — never silent bad answer.
        assert status == moreau.SolverStatus.NumericalError, (
            f"unexpected status {status!r} on ill-scaled LP; expected Solved with s >= 0 "
            f"or NumericalError."
        )


def test_active_set_ill_scaled_matches_ipm_when_solved():
    """When the active-set solver returns Solved, the objective should match IPM."""
    P, q, A, b, n, m = _build_ill_scaled_lp(seed=0)
    cones = moreau.Cones(num_nonneg_cones=m)

    # Reference from IPM
    ipm_settings = moreau.Settings(solver="ipm", device="cpu", verbose=False)
    ipm_solver = moreau.Solver(
        sparse.csr_matrix(P), q, sparse.csr_matrix(A), b, cones, settings=ipm_settings
    )
    ipm_sol = ipm_solver.solve()
    if ipm_solver.info.status != moreau.SolverStatus.Solved:
        pytest.skip("IPM did not solve cleanly; no reference to compare against")
    ipm_cost = 0.5 * ipm_sol.x @ P @ ipm_sol.x + q @ ipm_sol.x

    as_settings = moreau.Settings(solver="active_set", device="cpu", verbose=False)
    as_solver = moreau.Solver(
        sparse.csr_matrix(P), q, sparse.csr_matrix(A), b, cones, settings=as_settings
    )
    as_sol = as_solver.solve()
    if as_solver.info.status != moreau.SolverStatus.Solved:
        pytest.skip("active-set correctly reported a non-Solved status on this problem")
    as_cost = 0.5 * as_sol.x @ P @ as_sol.x + q @ as_sol.x

    # Objectives should agree to reasonable tolerance. Pre-fix the active-set cost
    # was ~13594 vs IPM's -0.2012 on this exact problem.
    rel_gap = abs(ipm_cost - as_cost) / max(1.0, abs(ipm_cost))
    assert rel_gap < 1e-3, (
        f"active-set cost={as_cost:.6e} diverges from IPM cost={ipm_cost:.6e} "
        f"(rel gap={rel_gap:.3e}) — likely the ill-scaling bug is back."
    )
