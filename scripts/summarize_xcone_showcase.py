"""Render showcase speedup tables from one or more bench JSON files.

    uv run python summarize_xcone_showcase.py bench1.json [bench2.json ...] [--cv 0.20]

Speedup = geomean over seeds of (slack_ms / other_ms); >1x = `other` faster.
All trials with ms>0 count — a slack run that stalls at MaxIterations still
consumes wall time, which is the source of the boundary blow-up wins.

Cells whose per-seed speedups vary by more than `--cv` (coefficient of
variation) are flagged with `~` and listed under "high variance" so they
can be oversampled with extra seeds.
"""

from __future__ import annotations
import argparse
import json
import math
import statistics
from collections import defaultdict

BATCHES = [1, 16, 64, 256]
SINGLE_KINDS = ["nonneg", "soc", "exp", "power", "psd", "gen_power"]
MIXED_KINDS = ["mixed_nn_psd", "mixed_socp", "mixed_exp_pow"]
REGIMES = ["interior", "boundary"]


def gm(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")


def load(paths):
    trials = []
    for p in paths:
        trials.extend(json.load(open(p))["trials"])
    return trials


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument(
        "--cv", type=float, default=0.20, help="flag cells whose per-seed speedup CV exceeds this"
    )
    args = ap.parse_args()
    trials = load(args.paths)

    # (kind, regime, device, batch, seed) -> {config: ms}
    ms = defaultdict(dict)
    solved = defaultdict(dict)
    for t in trials:
        k = (t["kind"], t["regime"], t["device"], t["batch_size"], t["seed"])
        ms[k][t["config"]] = t["ms"]
        solved[k][t["config"]] = (t["n_solved"], t["batch_size"])

    flagged = []  # (kind, regime, device, batch, cv, nseeds)

    def cell(kind, regime, dev, batch, numer="slack", denom="direct_x"):
        """Speedup of `denom` over `numer`: geomean of numer_ms/denom_ms."""
        ratios = []
        for k, m in ms.items():
            if k[:4] != (kind, regime, dev, batch):
                continue
            a, b = m.get(numer), m.get(denom)
            if a and b and a > 0 and b > 0:
                ratios.append(a / b)
        if not ratios:
            return "—"
        sp = gm(ratios)
        cv = statistics.pstdev(ratios) / statistics.mean(ratios) if len(ratios) > 1 else 0.0
        txt = f"{sp:.2f}×"
        if sp >= 1.5:
            txt = f"**{txt}**"
        if cv > args.cv and len(ratios) > 1:
            txt += "~"
            flagged.append((kind, regime, dev, batch, cv, len(ratios)))
        return txt

    def speedup_table(kinds, dev, denom, numer="slack"):
        print(f"| kind | regime | " + " | ".join(f"batch={b}" for b in BATCHES) + " |")
        print("|---|---|" + "---:|" * len(BATCHES))
        for kind in kinds:
            for regime in REGIMES:
                cells = [cell(kind, regime, dev, b, numer, denom) for b in BATCHES]
                print(f"| {kind} | {regime} | " + " | ".join(cells) + " |")

    for dev in ("cuda", "cpu"):
        print(f"\n### {dev.upper()} — direct-x speedup over slack\n")
        speedup_table(SINGLE_KINDS + MIXED_KINDS, dev, "direct_x")

    for dev in ("cuda", "cpu"):
        print(
            f"\n### {dev.upper()} — hybrid (block-0 direct-x, rest slack) speedup over all-slack\n"
        )
        speedup_table(MIXED_KINDS, dev, "hybrid")

    # Convergence at batch=256 — only rows where some config fell short.
    nseed = len({t["seed"] for t in trials})
    print(
        f"\n### Convergence at batch=256 (batch entries solved / total, "
        f"summed over {nseed} seeds)\n"
    )
    print("_Rows where every config solved 100% are omitted._\n")
    print("| kind | regime | device | slack | direct-x | hybrid |")
    print("|---|---|---|---|---|---|")
    for dev in ("cuda", "cpu"):
        for kind in SINGLE_KINDS + MIXED_KINDS:
            for regime in REGIMES:
                tot = defaultdict(lambda: [0, 0])
                for k, m in solved.items():
                    if k[:4] != (kind, regime, dev, 256):
                        continue
                    for cfg, (ns, bs) in m.items():
                        tot[cfg][0] += ns
                        tot[cfg][1] += bs
                if not tot:
                    continue
                if all(ns == bs for ns, bs in tot.values()):
                    continue  # everything solved — uninteresting

                def fmt(cfg):
                    return f"{tot[cfg][0]}/{tot[cfg][1]}" if cfg in tot else "—"

                print(
                    f"| {kind} | {regime} | {dev} | "
                    f"{fmt('slack')} | {fmt('direct_x')} | {fmt('hybrid')} |"
                )

    seeds = sorted({t["seed"] for t in trials})
    n_flag = len(set(flagged))
    print(
        f"\n_Geomean over {len(seeds)} seeds. `~` marks cells whose per-instance "
        f"speedup CV exceeds {args.cv:.0%} ({n_flag} of them) — the geomean is "
        f"stable but individual problems vary, typical of the boundary regime "
        f"where slack intermittently stalls._"
    )


if __name__ == "__main__":
    main()
