"""4-way direct-x vs slack SOC benchmark: {CPU, CUDA} × {slack, direct}.

Run:
    uv run python packages/moreau/tests/python/bench/bench_xcone_cuda.py

Same problem instance runs through all four paths via `moreau.Solver`
with `device='cpu'` vs `device='cuda'` and Cones configured for slack
(non-empty `so_cone_dims`) vs direct-x (non-empty `x_cones`). Reports
wall-clock (mean / p95 / min), iterations, and pairwise speedups.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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
    iterations: List[int]
    obj_val: float
    x: np.ndarray
    status: str
    failed: bool = False
    fail_reason: str = ""


def _banded_psd(n: int, bandwidth: int, rng: np.random.Generator) -> sparse.csr_matrix:
    """Symmetric banded PSD via random banded-triangular Cholesky factor."""
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


def _time_solve(
    run: Callable[[], tuple],
    n_runs: int,
    n_warmup: int,
    name: str,
) -> RunStats:
    """Fresh Solver per call — matches single-problem usage."""
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
            # info.iterations / info.status may be scalar (single-problem)
            # or array (batched); coerce to a single summary value.
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
# Each factory returns a dict {path_name: run_fn} where all paths solve
# the SAME underlying problem instance.


def _make_single_soc(k: int, seed: int = 0) -> Dict[str, Callable[[], tuple]]:
    rng = np.random.default_rng(seed + 17)
    P = _banded_psd(k, bandwidth=5, rng=rng)
    q = np.concatenate([[1.0], rng.standard_normal(k - 1)])
    s_cpu = moreau.Settings(device="cpu", verbose=False)
    s_gpu = moreau.Settings(device="cuda", verbose=False)

    def _slack(settings):
        A = -sparse.eye(k, format="csr")
        b = np.zeros(k)
        cones = moreau.Cones(so_cone_dims=[k])
        return _solve(P, q, A, b, cones, settings)

    def _direct(settings):
        A = sparse.csr_matrix(np.zeros((0, k)))
        b = np.array([])
        cones = moreau.Cones(
            x_cones=[moreau.XConeSpec(kind="soc", indices=list(range(k)))],
        )
        return _solve(P, q, A, b, cones, settings)

    return {
        "cpu-slack": lambda: _slack(s_cpu),
        "cpu-direct": lambda: _direct(s_cpu),
        "gpu-slack": lambda: _slack(s_gpu),
        "gpu-direct": lambda: _direct(s_gpu),
    }


def _make_tiled_socs(n_soc: int, soc_dim: int, seed: int = 0) -> Dict[str, Callable[[], tuple]]:
    rng = np.random.default_rng(seed + 42)
    n = n_soc * soc_dim
    P = _banded_psd(n, bandwidth=5, rng=rng)
    q = rng.standard_normal(n)
    for i in range(n_soc):
        q[i * soc_dim] = 1.0
    s_cpu = moreau.Settings(device="cpu", verbose=False)
    s_gpu = moreau.Settings(device="cuda", verbose=False)

    def _slack(settings):
        A = -sparse.eye(n, format="csr")
        b = np.zeros(n)
        cones = moreau.Cones(so_cone_dims=[soc_dim] * n_soc)
        return _solve(P, q, A, b, cones, settings)

    def _direct(settings):
        A = sparse.csr_matrix(np.zeros((0, n)))
        b = np.array([])
        x_cones = [
            moreau.XConeSpec(kind="soc", indices=list(range(i * soc_dim, (i + 1) * soc_dim)))
            for i in range(n_soc)
        ]
        return _solve(P, q, A, b, cones=moreau.Cones(x_cones=x_cones), settings=settings)

    return {
        "cpu-slack": lambda: _slack(s_cpu),
        "cpu-direct": lambda: _direct(s_cpu),
        "gpu-slack": lambda: _slack(s_gpu),
        "gpu-direct": lambda: _direct(s_gpu),
    }


SCENARIOS = {
    # Single-SOC dim sweep: dense-Hs path (dim ≤ 4), rank-2 sparse path
    # (dim > 4), and large-dim regime where kernel throughput matters.
    "single-SOC/k=3": lambda: _make_single_soc(k=3),
    "single-SOC/k=4": lambda: _make_single_soc(k=4),
    "single-SOC/k=6": lambda: _make_single_soc(k=6),
    "single-SOC/k=10": lambda: _make_single_soc(k=10),
    "single-SOC/k=20": lambda: _make_single_soc(k=20),
    "single-SOC/k=40": lambda: _make_single_soc(k=40),
    "single-SOC/k=100": lambda: _make_single_soc(k=100),
    "single-SOC/k=500": lambda: _make_single_soc(k=500),
    "single-SOC/k=1000": lambda: _make_single_soc(k=1000),
    "single-SOC/k=5000": lambda: _make_single_soc(k=5000),
    "single-SOC/k=10000": lambda: _make_single_soc(k=10000),
    # Tiled many-small-SOCs: stresses per-cone kernel-launch overhead
    # and the KKT-assembly plumbing at higher cone counts.
    "tiled-SOC/n_soc=100,dim=3": lambda: _make_tiled_socs(n_soc=100, soc_dim=3),
    "tiled-SOC/n_soc=500,dim=3": lambda: _make_tiled_socs(n_soc=500, soc_dim=3),
    "tiled-SOC/n_soc=5000,dim=3": lambda: _make_tiled_socs(n_soc=5000, soc_dim=3),
    "tiled-SOC/n_soc=100,dim=6": lambda: _make_tiled_socs(n_soc=100, soc_dim=6),
    "tiled-SOC/n_soc=500,dim=6": lambda: _make_tiled_socs(n_soc=500, soc_dim=6),
    "tiled-SOC/n_soc=2000,dim=6": lambda: _make_tiled_socs(n_soc=2000, soc_dim=6),
    "tiled-SOC/n_soc=500,dim=10": lambda: _make_tiled_socs(n_soc=500, soc_dim=10),
    "tiled-SOC/n_soc=1000,dim=20": lambda: _make_tiled_socs(n_soc=1000, soc_dim=20),
}


# ---------- Batched scenarios (CompiledSolver on both CPU and GPU) ----------
# Now that batched direct-x is wired on both devices (CUDA natively via
# CompiledSolver; CPU via per-problem loop in DirectXSolverCpu).


def _compiledsolver_run(P, A, q_mat, b_mat, cones, settings):
    """Build a CompiledSolver callable that runs `batch_size` problems
    per call. Returns a no-arg callable for benchmarking.

    Note: construction + setup are done ONCE outside the callable so
    timing captures only the .solve() call (the amortized-solve regime).
    """
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


def _make_batched_single_soc(
    k: int, batch_size: int, seed: int = 0
) -> Dict[str, Callable[[], tuple]]:
    """Batched single-SOC: `batch_size` copies with varied q."""
    rng = np.random.default_rng(seed + 17)
    P = _banded_psd(k, bandwidth=5, rng=rng)
    q_base = np.concatenate([[1.0], rng.standard_normal(k - 1)])
    q_mat = np.stack([q_base + 0.05 * rng.standard_normal(k) for _ in range(batch_size)])
    s_cpu = moreau.Settings(device="cpu", verbose=False, batch_size=batch_size)
    s_gpu = moreau.Settings(device="cuda", verbose=False, batch_size=batch_size)

    A_slack = -sparse.eye(k, format="csr")
    b_slack = np.zeros((batch_size, k))
    cones_slack = moreau.Cones(so_cone_dims=[k])

    A_direct = sparse.csr_matrix(np.zeros((0, k)))
    b_direct = np.zeros((batch_size, 0))
    cones_direct = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind="soc", indices=list(range(k)))],
    )

    return {
        "cpu-slack": _compiledsolver_run(P, A_slack, q_mat, b_slack, cones_slack, s_cpu),
        "cpu-direct": _compiledsolver_run(P, A_direct, q_mat, b_direct, cones_direct, s_cpu),
        "gpu-slack": _compiledsolver_run(P, A_slack, q_mat, b_slack, cones_slack, s_gpu),
        "gpu-direct": _compiledsolver_run(P, A_direct, q_mat, b_direct, cones_direct, s_gpu),
    }


def _make_batched_tiled_socs(
    n_soc: int, soc_dim: int, batch_size: int, seed: int = 0
) -> Dict[str, Callable[[], tuple]]:
    """Batched tiled-SOC."""
    rng = np.random.default_rng(seed + 42)
    n = n_soc * soc_dim
    P = _banded_psd(n, bandwidth=5, rng=rng)
    q_base = rng.standard_normal(n)
    for i in range(n_soc):
        q_base[i * soc_dim] = 1.0
    q_mat = np.stack([q_base + 0.05 * rng.standard_normal(n) for _ in range(batch_size)])
    s_cpu = moreau.Settings(device="cpu", verbose=False, batch_size=batch_size)
    s_gpu = moreau.Settings(device="cuda", verbose=False, batch_size=batch_size)

    A_slack = -sparse.eye(n, format="csr")
    b_slack = np.zeros((batch_size, n))
    cones_slack = moreau.Cones(so_cone_dims=[soc_dim] * n_soc)

    A_direct = sparse.csr_matrix(np.zeros((0, n)))
    b_direct = np.zeros((batch_size, 0))
    cones_direct = moreau.Cones(
        x_cones=[
            moreau.XConeSpec(kind="soc", indices=list(range(i * soc_dim, (i + 1) * soc_dim)))
            for i in range(n_soc)
        ],
    )

    return {
        "cpu-slack": _compiledsolver_run(P, A_slack, q_mat, b_slack, cones_slack, s_cpu),
        "cpu-direct": _compiledsolver_run(P, A_direct, q_mat, b_direct, cones_direct, s_cpu),
        "gpu-slack": _compiledsolver_run(P, A_slack, q_mat, b_slack, cones_slack, s_gpu),
        "gpu-direct": _compiledsolver_run(P, A_direct, q_mat, b_direct, cones_direct, s_gpu),
    }


BATCHED_SCENARIOS = {
    "batched/single-SOC/k=100/B=64": lambda: _make_batched_single_soc(k=100, batch_size=64),
    "batched/single-SOC/k=100/B=256": lambda: _make_batched_single_soc(k=100, batch_size=256),
    "batched/single-SOC/k=500/B=64": lambda: _make_batched_single_soc(k=500, batch_size=64),
    "batched/single-SOC/k=1000/B=64": lambda: _make_batched_single_soc(k=1000, batch_size=64),
    "batched/single-SOC/k=5000/B=16": lambda: _make_batched_single_soc(k=5000, batch_size=16),
    "batched/tiled-SOC/n_soc=100,dim=6/B=64": lambda: _make_batched_tiled_socs(
        n_soc=100, soc_dim=6, batch_size=64
    ),
    "batched/tiled-SOC/n_soc=500,dim=6/B=64": lambda: _make_batched_tiled_socs(
        n_soc=500, soc_dim=6, batch_size=64
    ),
}

# Merge batched into the main scenarios registry — same 4-way comparison,
# just with the CompiledSolver path instead of fresh-Solver-per-call.
SCENARIOS.update(BATCHED_SCENARIOS)


def _fmt(stats: RunStats) -> str:
    if stats.failed:
        return f"FAIL ({stats.fail_reason[:30]})"
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


def run_scenario(label: str, n_runs: int, n_warmup: int, paths: List[str]):
    runs = SCENARIOS[label]()
    stats = {p: _time_solve(runs[p], n_runs, n_warmup, p) for p in paths}

    print(f"\n{label}")
    for p in paths:
        print(f"  {p:<12}: {_fmt(stats[p])}")

    # Pairwise ratios that matter:
    #  direct vs slack (on each device) — direct-x structural speedup
    #  gpu vs cpu      (on each formulation) — device speedup
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


def _geomean(xs: List[float]) -> float:
    xs = [x for x in xs if x > 0 and not np.isnan(x) and not np.isinf(x)]
    if not xs:
        return float("nan")
    return float(np.exp(np.mean(np.log(xs))))


def _summary(results: list) -> None:
    print("\n" + "=" * 40)
    print("Geomean pairwise speedups (mean time)")
    print("=" * 40)

    def _collect(num: str, den: str) -> List[float]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["cpu-slack", "cpu-direct", "gpu-slack", "gpu-direct"],
        help="Subset of paths to run (default: all 4)",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=None,
        help="Subset of scenario labels (default = all)",
    )
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
