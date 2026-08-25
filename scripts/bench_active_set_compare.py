#!/usr/bin/env python3
"""Benchmark: active-set vs IPM (CPU) vs IPM (cuDSS) — forward + backward on batched box QPs.

Usage:
    python scripts/bench_active_set_compare.py
    python scripts/bench_active_set_compare.py --nvar 50 --batch 128 --trials 30
"""

import argparse
import time
import numpy as np
import moreau


def make_solver(nvar, batch_size, solver, device, direct_solve_method=None, enable_grad=False):
    """Create a CompiledSolver for box QPs: min 0.5 x'Px + q'x  s.t. x >= 0."""
    P_ro = np.arange(nvar + 1, dtype=np.int64) * nvar
    P_ci = np.tile(np.arange(nvar, dtype=np.int64), nvar)
    A_ro = np.arange(nvar + 1, dtype=np.int64)
    A_ci = np.arange(nvar, dtype=np.int64)
    cones = moreau.Cones(num_nonneg_cones=nvar)

    ipm_settings = None
    if direct_solve_method is not None:
        ipm_settings = moreau.IPMSettings(direct_solve_method=direct_solve_method)

    settings = moreau.Settings(
        solver=solver,
        device=device,
        batch_size=batch_size,
        enable_grad=enable_grad,
        ipm_settings=ipm_settings,
    )

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
    return compiled


def make_pd_batch(nvar, batch_size, rng):
    """Random SPD matrices (Delassus-like)."""
    P = np.empty((batch_size, nvar, nvar))
    for i in range(batch_size):
        A = rng.standard_normal((nvar, nvar))
        P[i] = A @ A.T + 1e-4 * np.eye(nvar)
    return P


def run_forward(compiled, P_batch, q_batch, bs):
    batch_size = P_batch.shape[0]
    P_flat = np.ascontiguousarray(P_batch.reshape(batch_size, -1), dtype=np.float64)
    A_vals = np.broadcast_to(-np.ones(P_batch.shape[1]), (batch_size, P_batch.shape[1])).copy()
    compiled.setup(P_flat, A_vals)
    return compiled.solve(q_batch, bs)


def run_backward(compiled, n, batch_size):
    dx = np.ones((batch_size, n), dtype=np.float64)
    dz = np.zeros((batch_size, n), dtype=np.float64)
    ds = np.zeros((batch_size, n), dtype=np.float64)
    return compiled.backward(dx, dz, ds)


def bench(name, compiled, nvar, batch_size, n_trials, rng, enable_grad):
    bs = np.zeros((batch_size, nvar), dtype=np.float64)
    fwd_times = []
    bwd_times = []
    last_x = None

    for _ in range(n_trials):
        P = make_pd_batch(nvar, batch_size, rng)
        q = rng.standard_normal((batch_size, nvar))

        t0 = time.perf_counter()
        sol = run_forward(compiled, P, q, bs)
        fwd_times.append(time.perf_counter() - t0)
        last_x = sol.x.copy()

        if enable_grad:
            t0 = time.perf_counter()
            run_backward(compiled, nvar, batch_size)
            bwd_times.append(time.perf_counter() - t0)

    return {
        "name": name,
        "fwd": np.array(fwd_times) * 1000,
        "bwd": np.array(bwd_times) * 1000 if bwd_times else None,
        "x": last_x,
    }


def fmt_ms(arr):
    return f"{np.median(arr):8.2f}ms"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvar", type=int, default=30)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--no-backward", action="store_true")
    args = parser.parse_args()

    nvar, batch, n_trials = args.nvar, args.batch, args.trials
    enable_grad = not args.no_backward

    configs = [
        # active-set solver is CPU-only but lives in the CUDA package,
        # so it must be accessed via device="cuda"
        ("active_set", "active_set", "cuda", None),
        ("ipm_cpu", "ipm", "cpu", None),
        ("ipm_cudss", "ipm", "cuda", "cudss"),
    ]

    print(f"Box QP benchmark: n={nvar}, batch={batch}, trials={n_trials}, grad={enable_grad}")
    print()

    # Create solvers
    solvers = {}
    for label, solver, device, method in configs:
        try:
            solvers[label] = make_solver(nvar, batch, solver, device, method, enable_grad)
        except Exception as e:
            print(f"  skip {label}: {e}")

    # Warmup (1 trial each, shared seed so same data)
    for label, compiled in solvers.items():
        rng = np.random.default_rng(0)
        bench(label, compiled, nvar, batch, 1, rng, enable_grad)

    # Benchmark
    results = {}
    for label, compiled in solvers.items():
        rng = np.random.default_rng(42)
        results[label] = bench(label, compiled, nvar, batch, n_trials, rng, enable_grad)

    # Print forward times
    print(f"{'':>16s}  {'fwd median':>12s}  {'fwd mean':>12s}  {'fwd min':>12s}")
    print("-" * 60)
    for label in results:
        r = results[label]
        print(
            f"{label:>16s}  {fmt_ms(r['fwd'])}  {np.mean(r['fwd']):8.2f}ms  {np.min(r['fwd']):8.2f}ms"
        )

    if enable_grad:
        print()
        print(f"{'':>16s}  {'bwd median':>12s}  {'bwd mean':>12s}  {'bwd min':>12s}")
        print("-" * 60)
        for label in results:
            r = results[label]
            if r["bwd"] is not None:
                print(
                    f"{label:>16s}  {fmt_ms(r['bwd'])}  {np.mean(r['bwd']):8.2f}ms  {np.min(r['bwd']):8.2f}ms"
                )

        print()
        print(f"{'':>16s}  {'total median':>12s}")
        print("-" * 40)
        for label in results:
            r = results[label]
            if r["bwd"] is not None:
                total = r["fwd"] + r["bwd"]
                print(f"{label:>16s}  {np.median(total):8.2f}ms")

    # Speedups relative to ipm_cudss (if available)
    if "ipm_cudss" in results:
        ref_fwd = np.median(results["ipm_cudss"]["fwd"])
        print()
        print("Speedup vs ipm_cudss (forward):")
        for label in results:
            r = results[label]
            ratio = ref_fwd / np.median(r["fwd"])
            print(f"  {label:>16s}: {ratio:.2f}x")

    # Solution agreement
    labels = list(results.keys())
    if len(labels) >= 2:
        print()
        print("Solution agreement (max |x_i - x_j|):")
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b_label = labels[i], labels[j]
                diff = np.max(np.abs(results[a]["x"] - results[b_label]["x"]))
                print(f"  {a} vs {b_label}: {diff:.2e}")


if __name__ == "__main__":
    main()
