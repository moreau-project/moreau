"""Comprehensive x-cone showcase benchmark.

Compares slack-form vs direct-x form for each cone kind across:

- Nine kinds: six single-cone (NonNeg, SOC, Exp, Power, PSD, GenPower)
  and three mixed-cone (nonneg+PSD, nonneg+SOC, exp+power).
- Two regimes:
    interior  — `min ½‖x − target‖²  s.t.  Bx = c, x ∈ K`
    boundary  — `min c'x + ½ε‖x‖²    s.t.  Bx = c, x ∈ K`
                (linear objective + tiny regulariser; optimum on a face of K)
- Two devices: cpu, cuda
- Four batch sizes: 1, 16, 64, 256

Slack form embeds the cone as a slack: `Bx = c, −Ix + s = 0, s ∈ K`,
with m = (n_eq + |J|) constraints. Direct-x form declares the cone on
`x` directly via `Cones.dir_cones=[DirectConeSpec(...)]`, with m = n_eq. For
mixed-cone problems a third `hybrid` form declares the first cone block
direct-x and embeds the remaining blocks as slacks. All forms solve
mathematically equivalent problems and converge to the same optimum
(up to IPM tolerance).

Outputs JSON of all trials; render with `summarize_xcone_showcase.py`.

Usage:
    uv run python scripts/bench_xcone_showcase.py [--out PATH] \\
        [--seeds 7,41,137] [--kinds soc,exp,...] [--devices cpu,cuda]
"""

from __future__ import annotations
import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple, Sequence
import numpy as np
from scipy import sparse
import moreau

# ---------------------------------------------------------------------------
# Per-cone problem builders.
#
# Each builder returns a tuple (P, q, A_slack, b_slack, slack_cones,
# A_dirx, b_dirx, dirdir_cones) — the SAME problem in two formulations.
# Sparsity pattern is independent of seed/values; values come from rng(seed).
# ---------------------------------------------------------------------------


@dataclass
class ConeProblem:
    """One sparsity-pattern + per-seed values factory.

    `n` and `m_slack`/`m_dirx` are fixed (sparsity pattern shared across
    seeds in a batch). `gen_values(seed)` returns (P_values, A_slack_values,
    A_dirx_values, q, b_slack, b_dirx) for a single seed.
    """

    kind: str
    label: str
    n: int  # primal-variable dim
    cone_indices: List[int]
    # Common-shape sparse matrices (zeros at value positions; the actual
    # values are filled in per-seed by gen_values).
    P_row_offsets: np.ndarray
    P_col_indices: np.ndarray
    nnz_P: int
    A_slack_row_offsets: np.ndarray
    A_slack_col_indices: np.ndarray
    nnz_A_slack: int
    A_dirx_row_offsets: np.ndarray
    A_dirx_col_indices: np.ndarray
    nnz_A_dirx: int
    m_slack: int
    m_dirx: int
    slack_cones: moreau.Cones
    dirdir_cones: moreau.Cones
    target_or_cost_fn: callable  # rng → np.ndarray (shape (n,))
    interior_x0_fn: callable  # () → np.ndarray (shape (n,))
    cone_x_kind: str  # 'nonneg', 'soc', 'exp', 'power', 'psd', 'gen_power'
    regime: str  # 'interior' or 'boundary'
    eps: float  # regulariser for boundary regime; 0 for interior
    # Hybrid form (mixed-cone problems only): first block declared direct-x,
    # the remaining blocks embedded as slacks. None for single-cone problems.
    A_hybrid_row_offsets: Optional[np.ndarray] = None
    A_hybrid_col_indices: Optional[np.ndarray] = None
    nnz_A_hybrid: int = 0
    m_hybrid: int = 0
    hybrid_cones: Optional[moreau.Cones] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _csr_diag_pattern(n: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Diagonal P (eye(n) sparsity)."""
    return (np.arange(n + 1, dtype=np.int64), np.arange(n, dtype=np.int64), n)


def _stack_neg_I_below_B(
    B_rows: int, B_cols: int, B_indptr, B_indices, neg_I_dim: int
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Stack [B; −I] CSR pattern. Returns (row_offsets, col_indices, nnz).

    First `B_rows` rows: pattern of B.
    Next `neg_I_dim` rows: identity (one nonzero per row, at column i).
    """
    total_rows = B_rows + neg_I_dim
    nnz = len(B_indices) + neg_I_dim
    row_offsets = np.empty(total_rows + 1, dtype=np.int64)
    col_indices = np.empty(nnz, dtype=np.int64)
    # B portion
    row_offsets[: B_rows + 1] = B_indptr
    col_indices[: len(B_indices)] = B_indices
    # −I portion
    base = len(B_indices)
    for i in range(neg_I_dim):
        row_offsets[B_rows + 1 + i] = base + i + 1
        col_indices[base + i] = i
    return row_offsets, col_indices, nnz


def _build_eq_pattern(n_eq: int, n: int, seed: int = 0) -> Tuple[sparse.csr_matrix, callable]:
    """Random equality block B (dense Gaussian / sqrt(n)). Returns (B_pattern, gen_values_fn)."""
    rng = np.random.default_rng(seed)
    if n_eq == 0:
        empty = sparse.csr_matrix((0, n))
        return empty, lambda _seed: np.zeros(0)
    # Deterministic sparsity: dense (so all entries present).
    B_pat = sparse.csr_matrix(np.ones((n_eq, n)))

    def gen_B_vals(local_seed: int) -> np.ndarray:
        r = np.random.default_rng(local_seed + 1009)
        return r.standard_normal(n_eq * n) / np.sqrt(n)

    return B_pat, gen_B_vals


def _build_dirx_problem(
    kind: str,
    regime: str,
    n: int,
    n_eq: int,
    cone_indices: List[int],
    slack_cones: moreau.Cones,
    dirx_xcones,
    eps: float,
    seed_offset: int,
    target_fn: callable,
    x0_fn: callable,
) -> ConeProblem:
    """Common assembly: P=εI or I, A_slack = [B; −I], A_dirx = B."""
    P_ro, P_ci, nnz_P = _csr_diag_pattern(n)
    B, gen_B_vals = _build_eq_pattern(n_eq, n, seed=seed_offset)
    # Slack form: A_slack = [B; −I]
    A_slack_ro, A_slack_ci, nnz_A_slack = _stack_neg_I_below_B(
        n_eq, n, B.indptr, B.indices, neg_I_dim=n
    )
    m_slack = n_eq + n
    # Direct-x form: A_dirx = B
    if n_eq == 0:
        A_dirx_ro = np.zeros(1, dtype=np.int64)
        A_dirx_ci = np.zeros(0, dtype=np.int64)
        nnz_A_dirx = 0
    else:
        A_dirx_ro = np.asarray(B.indptr, dtype=np.int64)
        A_dirx_ci = np.asarray(B.indices, dtype=np.int64)
        nnz_A_dirx = B.nnz
    m_dirx = n_eq

    dirdir_cones = moreau.Cones(num_zero_cones=n_eq, dir_cones=dirx_xcones)

    def gen_values(seed: int):
        rng = np.random.default_rng(seed)
        target = target_fn(rng)
        if regime == "interior":
            P_vals = np.ones(n)
            q = -target
        else:  # boundary
            P_vals = np.full(n, eps)
            q = target  # interpret target as cost vector
        x0 = x0_fn()
        B_vals = gen_B_vals(seed)
        if n_eq > 0:
            B_csr = sparse.csr_matrix((B_vals, B.indices, B.indptr), shape=(n_eq, n))
            c_eq = B_csr @ x0
        else:
            c_eq = np.zeros(0)
        # Slack-form A values: B then −I (slack-side)
        A_slack_vals = np.concatenate([B_vals, -np.ones(n)]) if n_eq > 0 else -np.ones(n)
        b_slack = np.concatenate([c_eq, np.zeros(n)])
        # Direct-x form A and b
        A_dirx_vals = B_vals.copy() if n_eq > 0 else np.zeros(0)
        b_dirx = c_eq.copy()
        return P_vals, A_slack_vals, q, b_slack, A_dirx_vals, b_dirx

    return (
        ConeProblem(
            kind=kind,
            label="",
            n=n,
            cone_indices=cone_indices,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            nnz_P=nnz_P,
            A_slack_row_offsets=A_slack_ro,
            A_slack_col_indices=A_slack_ci,
            nnz_A_slack=nnz_A_slack,
            A_dirx_row_offsets=A_dirx_ro,
            A_dirx_col_indices=A_dirx_ci,
            nnz_A_dirx=nnz_A_dirx,
            m_slack=m_slack,
            m_dirx=m_dirx,
            slack_cones=slack_cones,
            dirdir_cones=dirdir_cones,
            target_or_cost_fn=target_fn,
            interior_x0_fn=x0_fn,
            cone_x_kind=kind,
            regime=regime,
            eps=eps,
        ),
        gen_values,
    )


# Per-cone-kind problem factories
def make_nonneg(dim: int, n_eq: int, regime: str, seed_offset: int) -> Tuple[ConeProblem, callable]:
    slack_cones = moreau.Cones(num_zero_cones=n_eq, num_nonneg_cones=dim)
    dirx_xc = [moreau.DirectConeSpec(kind="nonneg", indices=list(range(dim)))]

    def target_fn(rng):
        if regime == "interior":
            return rng.standard_normal(dim) * 0.5  # may have negatives → cone binds
        else:
            return rng.standard_normal(dim)

    def x0_fn():
        return np.ones(dim)

    eps = 1e-6 if regime == "boundary" else 0.0
    p, gv = _build_dirx_problem(
        "nonneg",
        regime,
        dim,
        n_eq,
        list(range(dim)),
        slack_cones,
        dirx_xc,
        eps,
        seed_offset,
        target_fn,
        x0_fn,
    )
    p.label = f"d={dim} eq={n_eq} {regime}"
    return p, gv


def make_soc(dim: int, n_eq: int, regime: str, seed_offset: int) -> Tuple[ConeProblem, callable]:
    slack_cones = moreau.Cones(num_zero_cones=n_eq, so_cone_dims=[dim])
    dirx_xc = [moreau.DirectConeSpec(kind="soc", indices=list(range(dim)))]

    def target_fn(rng):
        if regime == "interior":
            t = rng.standard_normal(dim)
            t[0] = 0.5 * np.linalg.norm(t[1:])  # ensure cone is binding
            return t
        else:
            c = rng.standard_normal(dim)
            c[0] = -abs(c[0]) - 0.5  # negative time-coord encourages binding
            return c

    def x0_fn():
        x = np.zeros(dim)
        x[0] = 2.0
        return x

    eps = 1e-6 if regime == "boundary" else 0.0
    p, gv = _build_dirx_problem(
        "soc",
        regime,
        dim,
        n_eq,
        list(range(dim)),
        slack_cones,
        dirx_xc,
        eps,
        seed_offset,
        target_fn,
        x0_fn,
    )
    p.label = f"d={dim} eq={n_eq} {regime}"
    return p, gv


def make_exp(
    num_stacks: int, n_eq: int, regime: str, seed_offset: int
) -> Tuple[ConeProblem, callable]:
    n = 3 * num_stacks
    slack_cones = moreau.Cones(num_zero_cones=n_eq, num_exp_cones=num_stacks)
    dirx_xc = [
        moreau.DirectConeSpec(kind="exp", indices=[3 * k, 3 * k + 1, 3 * k + 2])
        for k in range(num_stacks)
    ]

    def target_fn(rng):
        t = rng.standard_normal(n)
        if regime == "interior":
            t[0::3] = -0.5
        else:
            t[0::3] = -np.abs(t[0::3]) - 0.3
        return t

    def x0_fn():
        return np.tile(np.array([3.0, 1.0, 0.0]), num_stacks)

    eps = 1e-6 if regime == "boundary" else 0.0
    p, gv = _build_dirx_problem(
        "exp",
        regime,
        n,
        n_eq,
        list(range(n)),
        slack_cones,
        dirx_xc,
        eps,
        seed_offset,
        target_fn,
        x0_fn,
    )
    p.label = f"K={num_stacks} eq={n_eq} {regime}"
    return p, gv


def make_power(
    num_stacks: int, alpha: float, n_eq: int, regime: str, seed_offset: int
) -> Tuple[ConeProblem, callable]:
    n = 3 * num_stacks
    slack_cones = moreau.Cones(num_zero_cones=n_eq, power_alphas=[alpha] * num_stacks)
    dirx_xc = [
        moreau.DirectConeSpec(kind="power", indices=[3 * k, 3 * k + 1, 3 * k + 2], alpha=alpha)
        for k in range(num_stacks)
    ]

    def target_fn(rng):
        return rng.standard_normal(n) * 0.5

    def x0_fn():
        return np.tile(np.array([1.0, 1.0, 0.0]), num_stacks)

    eps = 1e-6 if regime == "boundary" else 0.0
    p, gv = _build_dirx_problem(
        "power",
        regime,
        n,
        n_eq,
        list(range(n)),
        slack_cones,
        dirx_xc,
        eps,
        seed_offset,
        target_fn,
        x0_fn,
    )
    p.label = f"K={num_stacks} α={alpha} eq={n_eq} {regime}"
    return p, gv


def make_psd(k: int, n_eq: int, regime: str, seed_offset: int) -> Tuple[ConeProblem, callable]:
    n = k * (k + 1) // 2
    slack_cones = moreau.Cones(num_zero_cones=n_eq, psd_dims=[k])
    dirx_xc = [moreau.DirectConeSpec(kind="psd_triangle", indices=list(range(n)), psd_k=k)]

    def target_fn(rng):
        return rng.standard_normal(n) * 0.5

    def x0_fn():
        x = np.zeros(n)
        for j in range(k):
            x[j * (j + 1) // 2 + j] = 1.0  # identity diagonal
        return x

    eps = 1e-6 if regime == "boundary" else 0.0
    p, gv = _build_dirx_problem(
        "psd",
        regime,
        n,
        n_eq,
        list(range(n)),
        slack_cones,
        dirx_xc,
        eps,
        seed_offset,
        target_fn,
        x0_fn,
    )
    p.label = f"k={k} eq={n_eq} {regime}"
    return p, gv


def make_gen_power(
    dim1: int, dim2: int, n_eq: int, regime: str, seed_offset: int
) -> Tuple[ConeProblem, callable]:
    n = dim1 + dim2
    alphas = [1.0 / dim1] * dim1  # uniform
    slack_cones = moreau.Cones(num_zero_cones=n_eq, gen_power_cone_params=[(alphas, dim2)])
    dirx_xc = [moreau.DirectConeSpec(kind="gen_power", indices=list(range(n)), alphas=alphas, dim2=dim2)]

    def target_fn(rng):
        return rng.standard_normal(n) * 0.5

    def x0_fn():
        x = np.zeros(n)
        x[:dim1] = 1.0
        return x

    eps = 1e-6 if regime == "boundary" else 0.0
    p, gv = _build_dirx_problem(
        "gen_power",
        regime,
        n,
        n_eq,
        list(range(n)),
        slack_cones,
        dirx_xc,
        eps,
        seed_offset,
        target_fn,
        x0_fn,
    )
    p.label = f"d1={dim1} d2={dim2} eq={n_eq} {regime}"
    return p, gv


def _merge_slack_cones(plist, num_zero: int, dir_cones=None) -> moreau.Cones:
    """Merge the slack-form cones of `plist` into one canonical Cones.

    `dir_cones`, if given, are attached as direct-x cones (used by the
    hybrid formulation where some blocks stay direct-x).
    """
    num_nn = sum(getattr(p.slack_cones, "num_nonneg_cones", 0) or 0 for p in plist)
    so_dims: List[int] = []
    num_exp = 0
    power_alphas: List[float] = []
    psd_dims: List[int] = []
    gpc_params = []
    for p in plist:
        sc = p.slack_cones
        so_dims.extend(getattr(sc, "so_cone_dims", None) or [])
        num_exp += getattr(sc, "num_exp_cones", 0) or 0
        power_alphas.extend(getattr(sc, "power_alphas", None) or [])
        psd_dims.extend(getattr(sc, "psd_dims", None) or [])
        gpc_params.extend(getattr(sc, "gen_power_cone_params", None) or [])
    kw = dict(num_zero_cones=num_zero)
    if num_nn:
        kw["num_nonneg_cones"] = num_nn
    if so_dims:
        kw["so_cone_dims"] = so_dims
    if num_exp:
        kw["num_exp_cones"] = num_exp
    if power_alphas:
        kw["power_alphas"] = power_alphas
    if psd_dims:
        kw["psd_dims"] = psd_dims
    if gpc_params:
        kw["gen_power_cone_params"] = gpc_params
    if dir_cones:
        kw["dir_cones"] = dir_cones
    return moreau.Cones(**kw)


def make_mixed(label: str, parts, regime: str, seed_offset: int) -> Tuple[ConeProblem, callable]:
    """Compose multiple single-cone builders into one heterogeneous-cone problem.

    `parts`: list of nullary callables, each returning (ConeProblem, gen_values).
    The resulting problem has variables x = [x_1; x_2; ...] with each x_i
    constrained by its block's cone, plus block-diagonal equality structure.

    Slack form: A_slack = [B_block_diag; −I_total], cones = [zero (sum eq);
    block_1 cone; block_2 cone; ...].
    Direct-x form: A_dirx = B_block_diag, dir_cones with shifted indices.
    """
    sub = [fn() for fn in parts]
    sub_p = [s[0] for s in sub]
    sub_gv = [s[1] for s in sub]

    n_offs = [0]
    for p in sub_p:
        n_offs.append(n_offs[-1] + p.n)
    n_total = n_offs[-1]
    n_eq_per = [p.m_dirx for p in sub_p]  # m_dirx == n_eq for these builders
    n_eq_total = sum(n_eq_per)

    # Equality block sparsity (pre-computed from each builder's A_dirx).
    # B_total is block-diagonal in cols, contiguous in rows.
    B_ro = np.empty(n_eq_total + 1, dtype=np.int64)
    B_ro[0] = 0
    write_idx = 1
    nnz_off = 0
    B_ci_list = []
    for i, p in enumerate(sub_p):
        for k in range(p.m_dirx):
            B_ro[write_idx] = nnz_off + int(p.A_dirx_row_offsets[k + 1])
            write_idx += 1
        if p.m_dirx > 0:
            B_ci_list.append(np.asarray(p.A_dirx_col_indices, dtype=np.int64) + n_offs[i])
            nnz_off += int(p.A_dirx_row_offsets[p.m_dirx])
    B_ci = np.concatenate(B_ci_list).astype(np.int64) if B_ci_list else np.zeros(0, dtype=np.int64)
    B_nnz = nnz_off

    # A_dirx = B_total
    A_dirx_ro, A_dirx_ci, nnz_A_dirx = B_ro, B_ci, B_nnz
    m_dirx = n_eq_total

    # A_slack = [B_total; −I_total], rows ordered: all eq, then slack rows
    # (one slack block per cone-bearing block, in the same block order).
    m_slack = n_eq_total + n_total
    nnz_A_slack = B_nnz + n_total
    A_slack_ro = np.empty(m_slack + 1, dtype=np.int64)
    A_slack_ro[: n_eq_total + 1] = B_ro
    base = B_nnz
    for r in range(n_total):
        A_slack_ro[n_eq_total + 1 + r] = base + r + 1
    A_slack_ci = np.empty(nnz_A_slack, dtype=np.int64)
    A_slack_ci[:B_nnz] = B_ci
    A_slack_ci[B_nnz:] = np.arange(n_total, dtype=np.int64)

    # P sparsity: diagonal over all variables.
    P_ro = np.arange(n_total + 1, dtype=np.int64)
    P_ci = np.arange(n_total, dtype=np.int64)
    nnz_P = n_total

    # Cones: merge slack_cones (zero cones first; then cone-list per block).
    slack_cones = _merge_slack_cones(sub_p, n_eq_total)

    # Direct-x cones: shift each block's dir_cones indices by n_offs[i].
    def _shift_xcones(plist_idx):
        out = []
        for i in plist_idx:
            for xc in getattr(sub_p[i].dirdir_cones, "dir_cones", None) or []:
                kw = dict(kind=xc.kind, indices=[int(idx) + n_offs[i] for idx in xc.indices])
                for attr in ("alpha", "alphas", "dim2", "psd_k"):
                    v = getattr(xc, attr, None)
                    if v is not None:
                        kw[attr] = v
                out.append(moreau.DirectConeSpec(**kw))
        return out

    dirdir_cones = moreau.Cones(num_zero_cones=n_eq_total, dir_cones=_shift_xcones(range(len(sub_p))))

    # Hybrid form: block 0 declared direct-x, blocks 1.. embedded as slacks.
    # A_hybrid = [B_total; −I over the slack blocks' columns only].
    n_slack_vars = n_total - n_offs[1]  # vars of blocks 1..K-1
    m_hybrid = n_eq_total + n_slack_vars
    nnz_A_hybrid = B_nnz + n_slack_vars
    A_hybrid_ro = np.empty(m_hybrid + 1, dtype=np.int64)
    A_hybrid_ro[: n_eq_total + 1] = B_ro
    for r in range(n_slack_vars):
        A_hybrid_ro[n_eq_total + 1 + r] = B_nnz + r + 1
    A_hybrid_ci = np.empty(nnz_A_hybrid, dtype=np.int64)
    A_hybrid_ci[:B_nnz] = B_ci
    A_hybrid_ci[B_nnz:] = n_offs[1] + np.arange(n_slack_vars, dtype=np.int64)
    hybrid_cones = _merge_slack_cones(sub_p[1:], n_eq_total, dir_cones=_shift_xcones([0]))

    def gen_values(seed: int):
        P_parts, A_slack_B_parts, A_dirx_parts = [], [], []
        q_parts, b_eq_parts = [], []
        for i, gv in enumerate(sub_gv):
            P_v, A_slack_v, q_i, b_slack_i, A_dirx_v, b_dirx_i = gv(seed * 7919 + 13 * i)
            P_parts.append(P_v)
            # The block's A_slack is [B_i; -I_i]. We split it: B_i_vals are the
            # first nnz_A_dirx entries, the trailing -I has values −1 (n_i).
            block_B_nnz = sub_p[i].nnz_A_dirx
            A_slack_B_parts.append(A_slack_v[:block_B_nnz])
            A_dirx_parts.append(A_dirx_v)
            q_parts.append(q_i)
            b_eq_parts.append(b_dirx_i)  # equality RHS only
        P_v = np.concatenate(P_parts) if P_parts else np.zeros(0)
        A_dirx_v = np.concatenate(A_dirx_parts) if A_dirx_parts else np.zeros(0)
        # A_slack values: concatenation of all blocks' B values, then (-1) * n_total.
        if A_slack_B_parts:
            A_slack_v = np.concatenate([np.concatenate(A_slack_B_parts), -np.ones(n_total)])
        else:
            A_slack_v = -np.ones(n_total)
        q = np.concatenate(q_parts) if q_parts else np.zeros(0)
        b_eq = np.concatenate(b_eq_parts) if b_eq_parts else np.zeros(0)
        b_dirx = b_eq.copy()
        b_slack = np.concatenate([b_eq, np.zeros(n_total)])
        # Hybrid: B values then −I over the slack blocks' columns only.
        A_hybrid_v = np.concatenate([A_dirx_v, -np.ones(n_slack_vars)])
        b_hybrid = np.concatenate([b_eq, np.zeros(n_slack_vars)])
        return (P_v, A_slack_v, q, b_slack, A_dirx_v, b_dirx, A_hybrid_v, b_hybrid)

    p_out = ConeProblem(
        kind="mixed",
        label=label,
        n=n_total,
        cone_indices=list(range(n_total)),
        P_row_offsets=P_ro,
        P_col_indices=P_ci,
        nnz_P=nnz_P,
        A_slack_row_offsets=A_slack_ro,
        A_slack_col_indices=A_slack_ci,
        nnz_A_slack=nnz_A_slack,
        A_dirx_row_offsets=A_dirx_ro,
        A_dirx_col_indices=A_dirx_ci,
        nnz_A_dirx=nnz_A_dirx,
        m_slack=m_slack,
        m_dirx=m_dirx,
        slack_cones=slack_cones,
        dirdir_cones=dirdir_cones,
        target_or_cost_fn=lambda rng: np.zeros(n_total),
        interior_x0_fn=lambda: np.zeros(n_total),
        cone_x_kind="mixed",
        regime=regime,
        eps=0.0,
        A_hybrid_row_offsets=A_hybrid_ro,
        A_hybrid_col_indices=A_hybrid_ci,
        nnz_A_hybrid=nnz_A_hybrid,
        m_hybrid=m_hybrid,
        hybrid_cones=hybrid_cones,
    )
    return p_out, gen_values


# Mixed-cone presets — each returns (ConeProblem, gen_values) for use in benches.
def make_mixed_nn_psd(
    nn_dim: int, k: int, n_eq_each: int, regime: str, seed_offset: int
) -> Tuple[ConeProblem, callable]:
    label = f"nn{nn_dim}+psd{k} eq2x{n_eq_each} {regime}"
    return make_mixed(
        label,
        [
            lambda: make_nonneg(dim=nn_dim, n_eq=n_eq_each, regime=regime, seed_offset=seed_offset),
            lambda: make_psd(k=k, n_eq=n_eq_each, regime=regime, seed_offset=seed_offset + 11),
        ],
        regime,
        seed_offset,
    )


def make_mixed_socp(
    soc_dim: int, nn_dim: int, n_eq_each: int, regime: str, seed_offset: int
) -> Tuple[ConeProblem, callable]:
    label = f"nn{nn_dim}+soc{soc_dim} eq2x{n_eq_each} {regime}"
    # Order: nonneg first, then SOC (matches moreau.Cones canonical layout).
    return make_mixed(
        label,
        [
            lambda: make_nonneg(
                dim=nn_dim, n_eq=n_eq_each, regime=regime, seed_offset=seed_offset + 11
            ),
            lambda: make_soc(dim=soc_dim, n_eq=n_eq_each, regime=regime, seed_offset=seed_offset),
        ],
        regime,
        seed_offset,
    )


def make_mixed_exp_power(
    num_exp_stacks: int, num_pow_stacks: int, n_eq_each: int, regime: str, seed_offset: int
) -> Tuple[ConeProblem, callable]:
    label = f"exp{num_exp_stacks}+pow{num_pow_stacks} eq2x{n_eq_each} {regime}"
    return make_mixed(
        label,
        [
            lambda: make_exp(
                num_stacks=num_exp_stacks, n_eq=n_eq_each, regime=regime, seed_offset=seed_offset
            ),
            lambda: make_power(
                num_stacks=num_pow_stacks,
                alpha=0.5,
                n_eq=n_eq_each,
                regime=regime,
                seed_offset=seed_offset + 11,
            ),
        ],
        regime,
        seed_offset,
    )


# ---------------------------------------------------------------------------
# Batched solve helpers
# ---------------------------------------------------------------------------
@dataclass
class Trial:
    kind: str
    regime: str  # 'interior' or 'boundary'
    label: str
    n: int  # primal dim
    batch_size: int
    device: str
    config: str  # 'slack' or 'direct_x'
    status_summary: str
    n_solved: int  # number of batch entries that converged (Solved/AlmostSolved)
    iterations_max: int
    ms: float  # total wall time for the batched solve (ms)
    obj_min: float  # min obj over batch (sanity)
    obj_max: float
    seed: int


def _solve_batched(
    p: ConeProblem,
    gen_values: callable,
    batch_size: int,
    mode: str,
    device: str,
    base_seed: int,
    reps: int = 2,
    max_iter: int = 300,
) -> Trial:
    """Solve a batch of size `batch_size` problems, all sharing the same
    sparsity pattern but with per-instance values from sequential seeds.
    Times the median of `reps` solves after one warmup.
    """
    nnz_P = p.nnz_P
    P_ro = p.P_row_offsets
    P_ci = p.P_col_indices
    if mode == "slack":
        nnz_A = p.nnz_A_slack
        m = p.m_slack
        A_ro = p.A_slack_row_offsets
        A_ci = p.A_slack_col_indices
        cones = p.slack_cones
    elif mode == "direct_x":
        nnz_A = p.nnz_A_dirx
        m = p.m_dirx
        A_ro = p.A_dirx_row_offsets
        A_ci = p.A_dirx_col_indices
        cones = p.dirdir_cones
    elif mode == "hybrid":
        if p.hybrid_cones is None:
            raise ValueError(f"hybrid mode unsupported for kind={p.kind}")
        nnz_A = p.nnz_A_hybrid
        m = p.m_hybrid
        A_ro = p.A_hybrid_row_offsets
        A_ci = p.A_hybrid_col_indices
        cones = p.hybrid_cones
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Generate per-instance values.
    P_values_2d = np.empty((batch_size, nnz_P), dtype=np.float64)
    A_values_2d = np.empty((batch_size, nnz_A), dtype=np.float64)
    qs = np.empty((batch_size, p.n), dtype=np.float64)
    bs = np.empty((batch_size, m), dtype=np.float64)
    for i in range(batch_size):
        seed_i = base_seed * 100003 + i
        vals = gen_values(seed_i)
        P_v, A_slack_v, q_i, b_slack_i, A_dirx_v, b_dirx_i = vals[:6]
        A_hybrid_v = vals[6] if len(vals) > 6 else None
        b_hybrid_v = vals[7] if len(vals) > 7 else None
        P_values_2d[i] = P_v
        qs[i] = q_i
        if mode == "slack":
            A_values_2d[i] = A_slack_v
            bs[i] = b_slack_i
        elif mode == "direct_x":
            A_values_2d[i] = A_dirx_v
            bs[i] = b_dirx_i
        else:
            A_values_2d[i] = A_hybrid_v
            bs[i] = b_hybrid_v

    settings = moreau.Settings(
        device=device, batch_size=batch_size, verbose=False, max_iter=max_iter, solver="ipm"
    )

    def run_one():
        solver = moreau.CompiledSolver(
            n=p.n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=settings,
        )
        solver.setup(P_values=P_values_2d, A_values=A_values_2d)
        t = time.perf_counter()
        sol = solver.solve(qs=qs, bs=bs)
        ms = (time.perf_counter() - t) * 1e3
        return ms, solver, sol

    # Warmup
    try:
        _ms_warm, _solver, _sol = run_one()
    except Exception as e:
        return Trial(
            kind=p.kind,
            regime=p.regime,
            label=p.label,
            n=p.n,
            batch_size=batch_size,
            device=device,
            config=mode,
            status_summary=f"Error:{type(e).__name__}",
            n_solved=0,
            iterations_max=-1,
            ms=-1.0,
            obj_min=float("nan"),
            obj_max=float("nan"),
            seed=base_seed,
        )
    times = []
    last_solver = last_sol = None
    for _ in range(reps):
        ms, last_solver, last_sol = run_one()
        times.append(ms)
    times.sort()
    median = times[len(times) // 2]

    # Aggregate batched info. Some solver paths return a single scalar
    # info (status, iterations, obj_val) when batch_size=1; others return
    # arrays. Normalise everything to a list.
    info = last_solver.info

    def _as_list(x):
        if x is None:
            return []
        if hasattr(x, "__len__") and not isinstance(x, str):
            return list(x)
        return [x]

    statuses = _as_list(getattr(info, "status", None))
    iters = _as_list(getattr(info, "iterations", None))
    objs = _as_list(getattr(info, "obj_val", None))
    solved_set = ("Solved", "AlmostSolved")
    n_solved = sum(1 for s in statuses if getattr(s, "name", str(s)) in solved_set)
    iter_max = max((int(x) for x in iters), default=-1)
    obj_min = float(min(objs)) if len(objs) else float("nan")
    obj_max = float(max(objs)) if len(objs) else float("nan")
    from collections import Counter

    name_counts = Counter(getattr(s, "name", str(s)) for s in statuses)
    if name_counts:
        most_common_name, _ = name_counts.most_common(1)[0]
        status_summary = most_common_name
    else:
        status_summary = "Unknown"

    return Trial(
        kind=p.kind,
        regime=p.regime,
        label=p.label,
        n=p.n,
        batch_size=batch_size,
        device=device,
        config=mode,
        status_summary=status_summary,
        n_solved=n_solved,
        iterations_max=iter_max,
        ms=median,
        obj_min=obj_min,
        obj_max=obj_max,
        seed=base_seed,
    )


# ---------------------------------------------------------------------------
# Bench grid
# ---------------------------------------------------------------------------
KIND_FACTORIES = [
    # (name, factory, args)
    ("nonneg", make_nonneg, dict(dim=256, n_eq=64)),
    ("soc", make_soc, dict(dim=128, n_eq=32)),
    ("exp", make_exp, dict(num_stacks=64, n_eq=48)),
    ("power", make_power, dict(num_stacks=64, alpha=0.5, n_eq=48)),
    ("psd", make_psd, dict(k=15, n_eq=15)),
    ("gen_power", make_gen_power, dict(dim1=32, dim2=16, n_eq=12)),
    ("mixed_nn_psd", make_mixed_nn_psd, dict(nn_dim=128, k=12, n_eq_each=24)),
    ("mixed_socp", make_mixed_socp, dict(soc_dim=96, nn_dim=96, n_eq_each=24)),
    (
        "mixed_exp_pow",
        make_mixed_exp_power,
        dict(num_exp_stacks=32, num_pow_stacks=32, n_eq_each=24),
    ),
]

REGIMES = ["interior", "boundary"]
BATCH_SIZES = [1, 16, 64, 256]


def modes_for(name: str) -> List[str]:
    """Configs to run for a kind. Hybrid only applies to mixed-cone kinds."""
    if name.startswith("mixed"):
        return ["slack", "direct_x", "hybrid"]
    return ["slack", "direct_x"]


def device_available(dev: str) -> bool:
    if dev == "cpu":
        return True
    try:
        cones = moreau.Cones(num_nonneg_cones=1)
        P = sparse.eye(1, format="csr")
        A = sparse.eye(1, format="csr")
        s = moreau.Settings(device=dev, verbose=False)
        moreau.Solver(P, np.zeros(1), A, np.zeros(1), cones, settings=s).solve()
        return True
    except Exception as e:
        print(f"  [skip] device={dev}: {type(e).__name__}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="bench_results/xcone_showcase.json")
    parser.add_argument("--devices", default="cpu,cuda")
    parser.add_argument("--seeds", default="7,41,137")
    parser.add_argument("--batch-sizes", default=",".join(str(b) for b in BATCH_SIZES))
    parser.add_argument(
        "--kinds", default="", help="comma-separated kind names to restrict to (default: all)"
    )
    args = parser.parse_args()

    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    devices = [d for d in devices if device_available(d)]
    seeds = [int(x) for x in args.seeds.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    kind_factories = KIND_FACTORIES
    if args.kinds.strip():
        want = {k.strip() for k in args.kinds.split(",") if k.strip()}
        kind_factories = [kf for kf in KIND_FACTORIES if kf[0] in want]
        missing = want - {kf[0] for kf in kind_factories}
        if missing:
            raise SystemExit(f"unknown --kinds: {sorted(missing)}")

    trials: List[Trial] = []
    grid = [
        (name, fac, fargs, regime, batch, mode, dev, seed)
        for (name, fac, fargs) in kind_factories
        for regime in REGIMES
        for batch in batch_sizes
        for mode in modes_for(name)
        for dev in devices
        for seed in seeds
    ]
    print(
        f"problems: {len(kind_factories)}  regimes: {len(REGIMES)}  batches: {batch_sizes}  "
        f"devices: {devices}  seeds: {seeds}"
    )
    print(f"total trials: {len(grid)}")

    last_factory_state = None
    p_cache = None
    gen_cache = None
    for idx, (name, fac, fargs, regime, batch, mode, dev, seed) in enumerate(grid):
        # Build problem (once per (kind, regime, seed_offset)).
        key = (name, regime, seed)
        if key != last_factory_state:
            p_cache, gen_cache = fac(regime=regime, seed_offset=seed, **fargs)
            p_cache.kind = name  # distinguish mixed presets in the table
            last_factory_state = key
        try:
            t = _solve_batched(p_cache, gen_cache, batch, mode, dev, seed)
        except Exception as e:
            t = Trial(
                kind=name,
                regime=regime,
                label=p_cache.label,
                n=p_cache.n,
                batch_size=batch,
                device=dev,
                config=mode,
                status_summary=f"Error:{type(e).__name__}",
                n_solved=0,
                iterations_max=-1,
                ms=-1.0,
                obj_min=float("nan"),
                obj_max=float("nan"),
                seed=seed,
            )
        trials.append(t)
        if idx % 8 == 0 or t.status_summary not in ("Solved", "AlmostSolved"):
            print(
                f"  [{idx+1:>4}/{len(grid)}] {name:<10} {regime:<8} batch={batch:>3} "
                f"{dev:<4} {mode:<8} {t.status_summary:<22} solved={t.n_solved:>3}/{batch} "
                f"iters≤{t.iterations_max:>3}  {t.ms:>8.1f} ms"
            )

    out_path = args.out
    import os

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"trials": [asdict(t) for t in trials]}, f, indent=2)
    print(f"\nwrote {out_path}: {len(trials)} trials")


if __name__ == "__main__":
    main()
