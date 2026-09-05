"""4-way direct-x vs slack PSD benchmark: {CPU, CUDA} × {slack, direct}.

Run:
    uv run python packages/moreau/tests/python/bench/bench_xcone_psd_cuda.py

Same problem instance runs through all four paths via `moreau.Solver`
with `device='cpu'` vs `device='cuda'` and Cones configured for slack
(non-empty `psd_dims`, A = -I) vs direct-x (non-empty `dir_cones`,
m = 0). Reports wall-clock (mean / p95 / min), iterations, and
pairwise speedups.

Mirrors `bench_xcone_cuda.py` for SOC, providing an apples-to-apples
slack-vs-direct comparison for PSD on CUDA.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
from scipy import sparse

import moreau


@dataclass
class RunStats:
    name: str
    n_runs: int
    mean_time: float
    p95_time: float
    min_time: float
    iterations: List[float]
    obj_val: float
    x: np.ndarray
    status: str
    failed: bool = False
    fail_reason: str = ""


def _banded_psd(n: int, bandwidth: int, rng: np.random.Generator) -> sparse.csr_matrix:
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
    return (L @ L.T).tocsr() + sparse.eye(n, format="csr") * 1e-6


def _solve(P, q, A, b, cones, settings):
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    sol = solver.solve()
    return sol, solver.info


def _time_solve(run, n_runs, n_warmup, name) -> RunStats:
    try:
        for _ in range(n_warmup):
            run()
    except Exception as e:
        return RunStats(
            name=name,
            n_runs=0,
            mean_time=float("nan"),
            p95_time=float("nan"),
            min_time=float("nan"),
            iterations=[],
            obj_val=float("nan"),
            x=np.zeros(0),
            status="",
            failed=True,
            fail_reason=str(e),
        )

    times: List[float] = []
    iters: List[float] = []
    statuses: List[str] = []
    last_sol = None
    last_obj = 0.0
    try:
        for _ in range(n_runs):
            t0 = time.perf_counter()
            sol, info = run()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            it_val = info.iterations
            iters.append(float(np.mean(np.asarray(it_val).ravel())))
            st_val = info.status
            st_arr = np.atleast_1d(np.asarray(st_val))
            statuses.append(str(st_arr[0]) if st_arr.size >= 1 else "")
            obj_val = info.obj_val
            last_obj = float(np.asarray(obj_val).ravel()[0]) if obj_val is not None else 0.0
            last_sol = sol
    except Exception as e:
        return RunStats(
            name=name,
            n_runs=len(times),
            mean_time=float(np.mean(times)) if times else float("nan"),
            p95_time=float(np.percentile(times, 95)) if times else float("nan"),
            min_time=float(np.min(times)) if times else float("nan"),
            iterations=iters,
            obj_val=last_obj,
            x=np.asarray(last_sol.x, dtype=np.float64) if last_sol else np.zeros(0),
            status=statuses[-1] if statuses else "",
            failed=True,
            fail_reason=str(e),
        )
    assert last_sol is not None
    return RunStats(
        name=name,
        n_runs=n_runs,
        mean_time=float(np.mean(times)),
        p95_time=float(np.percentile(times, 95)),
        min_time=float(np.min(times)),
        iterations=iters,
        obj_val=last_obj,
        x=np.asarray(last_sol.x, dtype=np.float64),
        status=statuses[-1],
    )


# ---------- problem factories ----------
# Both formulations solve the same problem:
#   min 0.5 x'Px + q'x  s.t.  x[J] ∈ K_PSD (svec layout, length k(k+1)/2)
# Slack form:  rows of A = -I[J], so s[i] = x[J[i]], s ∈ K_PSD.
# Direct-x:    no slack rows; x[J] ∈ K_PSD enforced as direct-x cone.


def _svec_dim(k: int) -> int:
    return k * (k + 1) // 2


def _make_single_psd(k: int, seed: int = 0) -> Dict[str, Callable[[], tuple]]:
    """Single PSD cone, k×k matrix, n = k(k+1)/2."""
    rng = np.random.default_rng(seed + 17)
    n = _svec_dim(k)
    P = _banded_psd(n, bandwidth=5, rng=rng)
    # q crafted so the unconstrained optimum is (-q) which is mostly
    # off-diagonal noise around an interior PSD matrix; the IPM will
    # iterate but the constraint may or may not bind. That's fine —
    # we're comparing time, not solutions.
    q_base = rng.standard_normal(n) * 0.5
    # Negate diagonal entries so unconstrained optimum has positive diag
    # (helps both paths converge to a similar interior point).
    for col in range(k):
        diag_pos = (col + 1) * (col + 2) // 2 - 1
        q_base[diag_pos] = -1.0
    q = q_base
    s_cpu = moreau.Settings(device="cpu", verbose=False)
    s_gpu = moreau.Settings(device="cuda", verbose=False)

    def _slack(settings):
        A = -sparse.eye(n, format="csr")
        b = np.zeros(n)
        cones = moreau.Cones(psd_dims=[k])
        return _solve(P, q, A, b, cones, settings)

    def _direct(settings):
        A = sparse.csr_matrix(np.zeros((0, n)))
        b = np.array([])
        cones = moreau.Cones(
            dir_cones=[
                moreau.DirectConeSpec(
                    kind="psd_triangle",
                    indices=list(range(n)),
                    psd_k=k,
                )
            ],
        )
        return _solve(P, q, A, b, cones, settings)

    return {
        "cpu-slack": lambda: _slack(s_cpu),
        "cpu-direct": lambda: _direct(s_cpu),
        "gpu-slack": lambda: _slack(s_gpu),
        "gpu-direct": lambda: _direct(s_gpu),
    }


def _make_tiled_psds(n_psd: int, k: int, seed: int = 0) -> Dict[str, Callable[[], tuple]]:
    """`n_psd` separate k×k PSD cones, each owning a disjoint svec block of x."""
    rng = np.random.default_rng(seed + 42)
    svec_block = _svec_dim(k)
    n = n_psd * svec_block
    P = _banded_psd(n, bandwidth=5, rng=rng)
    q_base = rng.standard_normal(n) * 0.5
    for c in range(n_psd):
        offset = c * svec_block
        for col in range(k):
            diag_pos = (col + 1) * (col + 2) // 2 - 1
            q_base[offset + diag_pos] = -1.0
    q = q_base
    s_cpu = moreau.Settings(device="cpu", verbose=False)
    s_gpu = moreau.Settings(device="cuda", verbose=False)

    def _slack(settings):
        A = -sparse.eye(n, format="csr")
        b = np.zeros(n)
        cones = moreau.Cones(psd_dims=[k] * n_psd)
        return _solve(P, q, A, b, cones, settings)

    def _direct(settings):
        A = sparse.csr_matrix(np.zeros((0, n)))
        b = np.array([])
        dir_cones = [
            moreau.DirectConeSpec(
                kind="psd_triangle",
                indices=list(range(c * svec_block, (c + 1) * svec_block)),
                psd_k=k,
            )
            for c in range(n_psd)
        ]
        return _solve(P, q, A, b, cones=moreau.Cones(dir_cones=dir_cones), settings=settings)

    return {
        "cpu-slack": lambda: _slack(s_cpu),
        "cpu-direct": lambda: _direct(s_cpu),
        "gpu-slack": lambda: _slack(s_gpu),
        "gpu-direct": lambda: _direct(s_gpu),
    }


SCENARIOS: Dict[str, Callable[[], Dict[str, Callable[[], tuple]]]] = {
    # Single-PSD k sweep: small (svec_dim 3), moderate (10, 28), large.
    "single-PSD/k=2": lambda: _make_single_psd(k=2),  # svec_dim = 3
    "single-PSD/k=3": lambda: _make_single_psd(k=3),  # svec_dim = 6
    "single-PSD/k=5": lambda: _make_single_psd(k=5),  # svec_dim = 15
    "single-PSD/k=8": lambda: _make_single_psd(k=8),  # svec_dim = 36
    "single-PSD/k=12": lambda: _make_single_psd(k=12),  # svec_dim = 78
    "single-PSD/k=20": lambda: _make_single_psd(k=20),  # svec_dim = 210
    "single-PSD/k=40": lambda: _make_single_psd(k=40),  # svec_dim = 820
    "single-PSD/k=60": lambda: _make_single_psd(k=60),  # svec_dim = 1830
    "single-PSD/k=80": lambda: _make_single_psd(k=80),  # svec_dim = 3240
    # Tiled many-small-PSDs: stresses kernel-launch overhead and KKT plumbing.
    "tiled-PSD/n_psd=10,k=3": lambda: _make_tiled_psds(n_psd=10, k=3),
    "tiled-PSD/n_psd=50,k=3": lambda: _make_tiled_psds(n_psd=50, k=3),
    "tiled-PSD/n_psd=100,k=3": lambda: _make_tiled_psds(n_psd=100, k=3),
    "tiled-PSD/n_psd=20,k=5": lambda: _make_tiled_psds(n_psd=20, k=5),
    "tiled-PSD/n_psd=50,k=5": lambda: _make_tiled_psds(n_psd=50, k=5),
}


# ---------- Batched scenarios ----------


def _compiledsolver_run(P, A, q_mat, b_mat, cones, settings):
    n = P.shape[0]
    m = A.shape[0]
    P_csr = P.tocsr()
    A_csr = A.tocsr()
    solver = moreau.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=P_csr.indptr.astype(np.int64),
        P_col_indices=P_csr.indices.astype(np.int64),
        A_row_offsets=A_csr.indptr.astype(np.int64),
        A_col_indices=A_csr.indices.astype(np.int64),
        cones=cones,
        settings=settings,
    )
    A_vals = A_csr.data if A_csr.nnz > 0 else np.array([])
    solver.setup(P_values=P_csr.data, A_values=A_vals)

    def run():
        sol = solver.solve(qs=q_mat, bs=b_mat)
        return sol, solver.info

    return run


def _make_batched_single_psd(
    k: int, batch_size: int, seed: int = 0
) -> Dict[str, Callable[[], tuple]]:
    rng = np.random.default_rng(seed + 17)
    n = _svec_dim(k)
    P = _banded_psd(n, bandwidth=5, rng=rng)
    q_base = rng.standard_normal(n) * 0.5
    for col in range(k):
        diag_pos = (col + 1) * (col + 2) // 2 - 1
        q_base[diag_pos] = -1.0
    q_mat = np.stack([q_base + 0.05 * rng.standard_normal(n) for _ in range(batch_size)])
    s_cpu = moreau.Settings(device="cpu", verbose=False, batch_size=batch_size)
    s_gpu = moreau.Settings(device="cuda", verbose=False, batch_size=batch_size)

    A_slack = -sparse.eye(n, format="csr")
    b_slack = np.zeros((batch_size, n))
    cones_slack = moreau.Cones(psd_dims=[k])

    A_direct = sparse.csr_matrix(np.zeros((0, n)))
    b_direct = np.zeros((batch_size, 0))
    cones_direct = moreau.Cones(
        dir_cones=[moreau.DirectConeSpec(kind="psd_triangle", indices=list(range(n)), psd_k=k)],
    )

    return {
        "cpu-slack": _compiledsolver_run(P, A_slack, q_mat, b_slack, cones_slack, s_cpu),
        "cpu-direct": _compiledsolver_run(P, A_direct, q_mat, b_direct, cones_direct, s_cpu),
        "gpu-slack": _compiledsolver_run(P, A_slack, q_mat, b_slack, cones_slack, s_gpu),
        "gpu-direct": _compiledsolver_run(P, A_direct, q_mat, b_direct, cones_direct, s_gpu),
    }


BATCHED_SCENARIOS: Dict[str, Callable[[], Dict[str, Callable[[], tuple]]]] = {
    "batched/single-PSD/k=5/B=64": lambda: _make_batched_single_psd(k=5, batch_size=64),
    "batched/single-PSD/k=5/B=256": lambda: _make_batched_single_psd(k=5, batch_size=256),
    "batched/single-PSD/k=10/B=64": lambda: _make_batched_single_psd(k=10, batch_size=64),
    "batched/single-PSD/k=20/B=32": lambda: _make_batched_single_psd(k=20, batch_size=32),
}

SCENARIOS.update(BATCHED_SCENARIOS)


def _fmt(stats: RunStats) -> str:
    if stats.failed:
        return f"FAIL ({stats.fail_reason[:50]})"
    return (
        f"{stats.mean_time*1000:7.2f}ms "
        f"(p95 {stats.p95_time*1000:6.2f}, min {stats.min_time*1000:6.2f}, "
        f"iter {np.mean(stats.iterations):4.1f}, {stats.status})"
    )


def _ratio(a: RunStats, b: RunStats) -> str:
    if a.failed or b.failed:
        return "  -  "
    if b.mean_time <= 0:
        return "  ∞  "
    return f"{a.mean_time/b.mean_time:4.2f}x"


def run_scenario(label, n_runs, n_warmup, paths):
    runs = SCENARIOS[label]()
    stats = {p: _time_solve(runs[p], n_runs, n_warmup, p) for p in paths}

    print(f"\n{label}")
    for p in paths:
        print(f"  {p:<12}: {_fmt(stats[p])}")

    if "cpu-slack" in stats and "cpu-direct" in stats:
        print(
            f"  cpu slack/direct:  {_ratio(stats['cpu-slack'], stats['cpu-direct'])}"
            f"  (>1 means direct is faster)"
        )
    if "gpu-slack" in stats and "gpu-direct" in stats:
        print(f"  gpu slack/direct:  {_ratio(stats['gpu-slack'], stats['gpu-direct'])}")
    if "cpu-slack" in stats and "gpu-slack" in stats:
        print(
            f"  slack  cpu/gpu:    {_ratio(stats['cpu-slack'],  stats['gpu-slack'])}"
            f"  (>1 means gpu is faster)"
        )
    if "cpu-direct" in stats and "gpu-direct" in stats:
        print(f"  direct cpu/gpu:    {_ratio(stats['cpu-direct'], stats['gpu-direct'])}")

    return (label, stats)


def _geomean(xs):
    xs = [x for x in xs if x > 0 and not np.isnan(x) and not np.isinf(x)]
    if not xs:
        return float("nan")
    return float(np.exp(np.mean(np.log(xs))))


def _summary(results):
    print("\n" + "=" * 40)
    print("Geomean pairwise speedups (mean time)")
    print("=" * 40)

    def _collect(num, den):
        out = []
        for _, st in results:
            if num in st and den in st and not st[num].failed and not st[den].failed:
                if st[den].mean_time > 0:
                    out.append(st[num].mean_time / st[den].mean_time)
        return out

    for label, pair in [
        ("cpu slack/direct", ("cpu-slack", "cpu-direct")),
        ("gpu slack/direct", ("gpu-slack", "gpu-direct")),
        ("slack  cpu/gpu", ("cpu-slack", "gpu-slack")),
        ("direct cpu/gpu", ("cpu-direct", "gpu-direct")),
    ]:
        ratios = _collect(*pair)
        if ratios:
            print(
                f"  {label}: geomean {_geomean(ratios):4.2f}x   "
                f"(min {min(ratios):4.2f}, max {max(ratios):4.2f}, n={len(ratios)})"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["cpu-slack", "cpu-direct", "gpu-slack", "gpu-direct"],
    )
    parser.add_argument("--scenarios", nargs="+", default=None)
    args = parser.parse_args()

    scenarios = args.scenarios or list(SCENARIOS.keys())
    print(f"Paths:     {args.paths}")
    print(f"Scenarios: {len(scenarios)}  ({args.runs} timed + {args.warmup} warmup per path)")

    results = []
    for label in scenarios:
        try:
            results.append(run_scenario(label, args.runs, args.warmup, args.paths))
        except Exception as e:
            print(f"{label}: FAILED (scenario factory) — {e}")
            sys.exit(2)

    _summary(results)


if __name__ == "__main__":
    main()
