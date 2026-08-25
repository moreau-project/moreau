#!/usr/bin/env python3
"""Benchmark: active-set vs IPM for batched box QPs.

Reproduces the contact dynamics QP from DAIR exploration:
    minimize    0.5 x' P x + q' x
    subject to  x >= 0

where P is a dense SPD matrix (Delassus + regularization) and q is dense.

Usage:
    python scripts/bench_active_set_boxqp.py
    python scripts/bench_active_set_boxqp.py --nvar 50 --batch 128
"""

import argparse
import time
import numpy as np
import moreau


def make_box_qp_solver(nvar, batch_size, solver="auto"):
    """Create a reusable solver for box QPs: min 0.5 x'Px + q'x s.t. x >= 0.

    Args:
        nvar: Number of variables.
        batch_size: Number of problems to solve in parallel.
        solver: 'auto', 'active_set', or 'ipm'.

    Returns:
        A function solve(P_batch, q_batch) -> x that solves the batch of QPs.
    """
    P_ro = np.arange(nvar + 1, dtype=np.int64) * nvar
    P_ci = np.tile(np.arange(nvar, dtype=np.int64), nvar)
    A_ro = np.arange(nvar + 1, dtype=np.int64)
    A_ci = np.arange(nvar, dtype=np.int64)
    cones = moreau.Cones(num_nonneg_cones=nvar)
    settings = moreau.Settings(solver=solver, device="cuda", batch_size=batch_size)

    compiled = moreau.CompiledSolver(
        n=nvar,
        m=nvar,
        P_row_offsets=P_ro,
        P_col_indices=P_ci,
        A_row_offsets=A_ro,
        A_col_indices=A_ci,
        cones=cones,
        settings=settings,
    )

    A_vals = np.broadcast_to(-np.ones(nvar), (batch_size, nvar)).copy()
    bs = np.zeros((batch_size, nvar), dtype=np.float64)

    def solve(P_batch, q_batch):
        """Solve batch of box QPs.

        Args:
            P_batch: shape (batch, n, n), symmetric PD Hessians.
            q_batch: shape (batch, n), linear costs.

        Returns:
            x: shape (batch, n), optimal solutions.
        """
        P_flat = np.ascontiguousarray(P_batch.reshape(batch_size, -1), dtype=np.float64)
        q = np.ascontiguousarray(q_batch, dtype=np.float64)
        compiled.setup(P_flat, A_vals)
        return compiled.solve(q, bs).x

    return solve


def make_pd_batch(nvar, batch_size, rng):
    """Random SPD matrices simulating Delassus matrices."""
    P = np.empty((batch_size, nvar, nvar))
    for i in range(batch_size):
        A = rng.standard_normal((nvar, nvar))
        P[i] = A @ A.T + 1e-4 * np.eye(nvar)
    return P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvar", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    nvar, batch, n_trials = args.nvar, args.batch, args.trials
    rng = np.random.default_rng(42)

    # Create cached solvers
    solve_as = make_box_qp_solver(nvar, batch, solver="active_set")
    solve_ipm = make_box_qp_solver(nvar, batch, solver="ipm")

    # Warmup
    P0 = make_pd_batch(nvar, batch, rng)
    q0 = rng.standard_normal((batch, nvar))
    solve_as(P0, q0)
    solve_ipm(P0, q0)

    # Benchmark
    times = {"active_set": [], "ipm": []}
    solutions = {}
    for _ in range(n_trials):
        P = make_pd_batch(nvar, batch, rng)
        q = rng.standard_normal((batch, nvar))

        t0 = time.perf_counter()
        solutions["active_set"] = solve_as(P, q)
        times["active_set"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        solutions["ipm"] = solve_ipm(P, q)
        times["ipm"].append(time.perf_counter() - t0)

    print(f"Box QP benchmark: nvar={nvar}, batch={batch}, {n_trials} trials")
    print()
    print(f"{'':>16s}  {'median':>10s}  {'mean':>10s}  {'min':>10s}")
    print("-" * 52)
    for name in times:
        t = np.array(times[name]) * 1000
        print(f"{name:>16s}  {np.median(t):8.2f}ms  {np.mean(t):8.2f}ms  {np.min(t):8.2f}ms")

    t_as = np.median(times["active_set"])
    t_ipm = np.median(times["ipm"])
    print(f"\n  speedup: {t_ipm / t_as:.1f}x")

    diff = np.max(np.abs(solutions["active_set"] - solutions["ipm"]))
    print(f"  max |x_as - x_ipm|: {diff:.2e}")


if __name__ == "__main__":
    main()
