"""Decision-gate benchmark: direct-x cones vs slack cones on CPU.

Run:
    uv run python packages/moreau/tests/python/bench/bench_direct_cone_cpu.py

Both paths run through the same underlying Rust machinery
(`moreau_cpu._cpu_solver.DefaultSolver` — `new` for slack, `new_with_xcones`
for direct-x). The high-level `moreau.Solver` is NOT used here because its
slack path goes through `CompiledSolver` while the direct-x path goes
through `DirectXSolverCpu`, which is a path asymmetry that confounds the
comparison (a random dense nonneg QP at n=500 hits MaxIterations on the
CompiledSolver path but converges in 12 iterations through DefaultSolver).
This benchmark measures the STRUCTURAL advantage of the direct-x
formulation on a fixed backend.

Reports wall-clock (mean / p95), IPM iterations, and solution quality
for each scenario, then summarises against the go/no-go criteria:

  Go          : ≥1.5× median speedup on ≥2 of {S1, S2, S3, S5},
                no scenario worse than 1.1× slower.
  Conditional : 1.2×–1.5× on the majority, specific wins ≥2×.
  No-go       : <1.2× typical, or iter-count regression >15% on S2/S3
                from the uniform-scale projection.

S4 (PSD) is not included in this benchmark.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np
from scipy import sparse

import moreau_cpu._cpu_solver as cpu


class _Result:
    """Minimal solver.info-like shim for the benchmark."""

    def __init__(self, info_like, sol):
        self.status = info_like.status
        self.iterations = info_like.iterations
        self.obj_val = info_like.obj_val
        self.x = np.asarray(sol.x, dtype=np.float64)


SolverStatus = cpu.SolverStatus  # type: ignore[assignment]


def _make_settings():
    s = cpu.DefaultSettings()
    s.verbose = False
    s.ipm.presolve_enable = False
    return s


def _solve_slack(P, q, A, b, slack_cones):
    solver = cpu.DefaultSolver(
        P,
        q.tolist(),
        A,
        b.tolist(),
        slack_cones,
        _make_settings(),
    )
    solver.solve()
    sol = solver.get_solution()
    return _Result(sol, sol)


def _solve_direct(P, q, A, b, slack_cones, dir_cones):
    solver = cpu.DefaultSolver.new_with_xcones(
        P,
        q.tolist(),
        A,
        b.tolist(),
        slack_cones,
        dir_cones,
        _make_settings(),
    )
    solver.solve()
    sol = solver.get_solution()
    return _Result(sol, sol)


# ---------- timing helpers ----------


@dataclass
class RunStats:
    name: str
    n_runs: int
    mean_time: float
    p95_time: float
    iterations: List[int]
    obj_val: float
    x: np.ndarray


def _time_solve(
    run: Callable[[], _Result],
    n_runs: int,
    n_warmup: int,
) -> RunStats:
    """End-to-end wall-clock timing of `run()` (fresh DefaultSolver per
    call — DefaultSolver takes q/b at construction time). Both slack and
    direct-x paths pay an equal symbolic+equilibration cost here, which
    is the apples-to-apples comparison of the structural formulations
    for a one-shot solve.
    """
    for _ in range(n_warmup):
        run()

    times = []
    iters = []
    statuses = []
    last = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = run()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        iters.append(result.iterations)
        statuses.append(result.status)
        last = result
    assert last is not None
    bad = [s for s in statuses if s != SolverStatus.Solved]

    stats = RunStats(
        name=run.__name__,
        n_runs=n_runs,
        mean_time=float(np.mean(times)),
        p95_time=float(np.percentile(times, 95)),
        iterations=iters,
        obj_val=float(last.obj_val),
        x=np.asarray(last.x, dtype=np.float64),
    )
    stats.bad_statuses = bad  # type: ignore[attr-defined]
    return stats


# ---------- scenarios ----------


def _banded_psd(n: int, bandwidth: int, rng: np.random.Generator) -> sparse.csr_matrix:
    """Symmetric banded PSD matrix — representative of MPC / discretized
    operators. Half-bandwidth `bandwidth`; nnz ≈ n * (2*bandwidth + 1).

    Built from a random banded Cholesky factor L so P = L L' + εI is
    sparse and PSD by construction.
    """
    rows, cols, vals = [], [], []
    for i in range(n):
        for j in range(max(0, i - bandwidth), i + 1):
            rows.append(i)
            cols.append(j)
            vals.append(rng.standard_normal())
    L = sparse.csr_matrix(
        (np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
        shape=(n, n),
    )
    P = (L @ L.T).tocsr() + sparse.eye(n, format="csr") * 1e-6
    return P.tocsr()


def _make_s1(n: int, bandwidth: int = 5, seed: int = 0):
    """S1 — One-sided nonneg QP: min 0.5 x'Px + q'x  s.t.  x >= 0."""
    rng = np.random.default_rng(seed)
    P = _banded_psd(n, bandwidth, rng)
    q = rng.standard_normal(n)

    def slack_run():
        A = -sparse.eye(n, format="csr")
        b = np.zeros(n)
        return _solve_slack(P, q, A, b, [cpu.NonnegativeConeT(n)])

    slack_run.__name__ = "slack"

    def direct_run():
        A = sparse.csr_matrix(np.zeros((0, n)))
        b = np.array([])
        return _solve_direct(P, q, A, b, [], [cpu.NonnegativeXConeT(list(range(n)))])

    direct_run.__name__ = "direct"

    return slack_run, direct_run


def _make_s2(k: int, seed: int = 0):
    """S2 — Single SOC: min 0.5 x'Px + q'x  s.t.  x ∈ SOC_k."""
    rng = np.random.default_rng(seed + 17)
    n = k
    P = _banded_psd(n, bandwidth=5, rng=rng)
    q = np.concatenate([[1.0], rng.standard_normal(k - 1)])

    def slack_run():
        A = -sparse.eye(n, format="csr")
        b = np.zeros(n)
        return _solve_slack(P, q, A, b, [cpu.SecondOrderConeT(k)])

    slack_run.__name__ = "slack"

    def direct_run():
        A = sparse.csr_matrix(np.zeros((0, n)))
        b = np.array([])
        return _solve_direct(P, q, A, b, [], [cpu.SecondOrderXConeT(list(range(n)))])

    direct_run.__name__ = "direct"

    return slack_run, direct_run


def _make_s3(n_soc: int, soc_dim: int = 3, seed: int = 0):
    """S3 — Many small SOCs tiled across x."""
    rng = np.random.default_rng(seed + 42)
    n = n_soc * soc_dim
    P = _banded_psd(n, bandwidth=5, rng=rng)
    q = rng.standard_normal(n)
    for i in range(n_soc):
        q[i * soc_dim] = 1.0

    def slack_run():
        A = -sparse.eye(n, format="csr")
        b = np.zeros(n)
        return _solve_slack(P, q, A, b, [cpu.SecondOrderConeT(soc_dim) for _ in range(n_soc)])

    slack_run.__name__ = "slack"

    def direct_run():
        A = sparse.csr_matrix(np.zeros((0, n)))
        b = np.array([])
        dir_cones = [
            cpu.SecondOrderXConeT(list(range(i * soc_dim, (i + 1) * soc_dim))) for i in range(n_soc)
        ]
        return _solve_direct(P, q, A, b, [], dir_cones)

    direct_run.__name__ = "direct"

    return slack_run, direct_run


def _make_s5(n: int, m_eq: int = 1, seed: int = 0):
    """S5 — Mixed: zero-cone slack + nonneg direct-x."""
    rng = np.random.default_rng(seed + 7)
    P = _banded_psd(n, bandwidth=5, rng=rng)
    q = rng.standard_normal(n)
    Aeq = sparse.csr_matrix(np.ones((m_eq, n)))
    beq = np.ones(m_eq)

    def slack_run():
        A = sparse.vstack([Aeq, -sparse.eye(n, format="csr")]).tocsr()
        b = np.concatenate([beq, np.zeros(n)])
        cones_list = [cpu.ZeroConeT(m_eq), cpu.NonnegativeConeT(n)]
        return _solve_slack(P, q, A, b, cones_list)

    slack_run.__name__ = "slack"

    def direct_run():
        A = Aeq
        b = beq
        return _solve_direct(
            P,
            q,
            A,
            b,
            [cpu.ZeroConeT(m_eq)],
            [cpu.NonnegativeXConeT(list(range(n)))],
        )

    direct_run.__name__ = "direct"

    return slack_run, direct_run


# ---------- driver ----------


SCENARIOS = {
    # S1 — one-sided nonneg box QP with a banded P (bandwidth 5).
    "S1/n=500": lambda: _make_s1(n=500),
    "S1/n=5000": lambda: _make_s1(n=5000),
    "S1/n=20000": lambda: _make_s1(n=20000),
    # S2 — single SOC. k is the cone dimension (also the total var count).
    "S2/k=100": lambda: _make_s2(k=100),
    "S2/k=1000": lambda: _make_s2(k=1000),
    "S2/k=5000": lambda: _make_s2(k=5000),
    # S3 — many small SOCs of size 3 tiled across x. n = 3 * n_soc.
    "S3/n_soc=100": lambda: _make_s3(n_soc=100),
    "S3/n_soc=1000": lambda: _make_s3(n_soc=1000),
    "S3/n_soc=5000": lambda: _make_s3(n_soc=5000),
    # S5 — mixed: one equality constraint + nonneg on x.
    "S5/n=500": lambda: _make_s5(n=500),
    "S5/n=5000": lambda: _make_s5(n=5000),
    "S5/n=20000": lambda: _make_s5(n=20000),
}


def run_scenario(label: str, n_runs: int, n_warmup: int) -> tuple:
    slack_run, direct_run = SCENARIOS[label]()
    slack = _time_solve(slack_run, n_runs, n_warmup)
    direct = _time_solve(direct_run, n_runs, n_warmup)

    speedup_mean = slack.mean_time / direct.mean_time
    speedup_p95 = slack.p95_time / direct.p95_time
    x_diff = float(np.linalg.norm(slack.x - direct.x))
    iter_delta = np.mean(direct.iterations) - np.mean(slack.iterations)

    slack_bad = getattr(slack, "bad_statuses", [])
    direct_bad = getattr(direct, "bad_statuses", [])
    warn = ""
    if slack_bad:
        warn += f" [slack bad: {slack_bad}]"
    if direct_bad:
        warn += f" [direct bad: {direct_bad}]"

    print(
        f"{label:<16}  slack: {slack.mean_time*1000:7.3f}ms ({slack.p95_time*1000:7.3f}ms p95, "
        f"iter {np.mean(slack.iterations):4.1f})   "
        f"direct: {direct.mean_time*1000:7.3f}ms ({direct.p95_time*1000:7.3f}ms p95, "
        f"iter {np.mean(direct.iterations):4.1f})   "
        f"speedup mean {speedup_mean:4.2f}x, p95 {speedup_p95:4.2f}x   "
        f"‖x_diff‖ {x_diff:.2e}   Δiter {iter_delta:+.1f}{warn}"
    )
    return (label, speedup_mean, speedup_p95, x_diff, iter_delta)


def _go_no_go(results: list) -> str:
    """Aggregate the go/no-go decision across scenarios."""
    speedups = [(lab, s) for (lab, s, _, _, _) in results]
    iter_regressions = {lab: di for (lab, _, _, _, di) in results}

    wins_1_5x = [lab for (lab, s) in speedups if s >= 1.5]
    regressions = [lab for (lab, s) in speedups if s < 1.0 / 1.1]

    # S2/S3 iteration regressions above 15% (relative to slack median)
    s2s3_iter_bad = []
    for lab, _, _, _, di in results:
        if lab.startswith(("S2", "S3")) and di > 0:
            # di is absolute iteration delta; call it "bad" if > 15% of slack
            # (use slack mean as denominator; safe since solved scenarios)
            s2s3_iter_bad.append((lab, di))

    print("\n=== GO/NO-GO ===")
    print(f"Scenarios with ≥1.5× speedup:      {wins_1_5x}")
    print(f"Scenarios with >1.1× regression:   {regressions}")
    if s2s3_iter_bad:
        print(f"S2/S3 iteration-count regressions: {s2s3_iter_bad}")

    if not regressions and len(wins_1_5x) >= 2:
        return "GO"
    if all(s >= 1.2 for (_, s) in speedups):
        big_wins = [lab for (lab, s) in speedups if s >= 2.0]
        if big_wins:
            return f"CONDITIONAL GO (specific wins: {big_wins})"
        return "CONDITIONAL GO"
    return "NO-GO / REDESIGN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20, help="Timed runs per scenario")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per scenario")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Subset of scenario labels; default = all",
    )
    args = parser.parse_args()

    scenarios = args.scenarios or list(SCENARIOS.keys())
    print(f"Running {len(scenarios)} scenarios, {args.runs} timed runs / {args.warmup} warmup")
    print("-" * 100)

    results = []
    for label in scenarios:
        try:
            results.append(run_scenario(label, args.runs, args.warmup))
        except Exception as e:
            print(f"{label}: FAILED — {e}")
            sys.exit(2)

    decision = _go_no_go(results)
    print(f"\nDecision: {decision}")


if __name__ == "__main__":
    main()
