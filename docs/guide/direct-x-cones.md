# Direct-x Cones

Moreau's base conic form wraps every conic constraint in a slack
variable `s`:

$$
\begin{aligned}
\text{minimize} \quad & \tfrac{1}{2} x^\top P x + q^\top x \\
\text{subject to} \quad & A x + s = b,\quad s \in \mathcal{K}
\end{aligned}
$$

When a cone constrains a sub-vector of `x` directly — e.g. `x[J] ≥ 0`
or `x[J] ∈ SOC` — adding a slack variable inflates the problem with an
extra row of `A`, an extra slack, and an extra cone block. The
**direct-x** API lets you declare such constraints inline:

$$
\begin{aligned}
\text{minimize} \quad & \tfrac{1}{2} x^\top P x + q^\top x \\
\text{subject to} \quad & A x + s = b,\; s \in \mathcal{K}_{\text{slack}} \\
& x_J \in \mathcal{K}_J \quad \text{(for each declared partition } J \text{)}
\end{aligned}
$$

Slack and direct-x cones can be mixed freely. Direct-x typically solves
faster than the slack equivalent — single-SOC measured ~2.5× on CPU;
batched-CUDA wins are larger for cones with many active constraints.

## Python API

Declare direct-x cones via `XConeSpec` entries on `Cones.x_cones`:

```python
import numpy as np
from scipy import sparse
import moreau

# min 0.5 x'Px + q'x  s.t.  x[0] >= 0, x[1:4] in SOC_3
P = sparse.diags([1.0, 1.0, 1.0, 1.0], format='csr')
q = np.array([2.0, 1.0, 0.0, 0.0])
A = sparse.csr_matrix(np.zeros((0, 4)))
b = np.array([])

cones = moreau.Cones(
    x_cones=[
        moreau.XConeSpec(kind='nonneg', indices=[0]),
        moreau.XConeSpec(kind='soc',    indices=[1, 2, 3]),
    ],
)
solver = moreau.Solver(P, q, A, b, cones, moreau.Settings(device='cpu'))
sol = solver.solve()
print(sol.x)
```

`XConeSpec` fields:

- `kind`: one of `'nonneg'`, `'soc'`, `'psd_triangle'`, `'exp'`,
  `'power'`, `'gen_power'`.
- `indices`: list of distinct, non-negative indices into `x`. Must stay
  within `[0, n)`. Indices across all `x_cones` must be pairwise
  disjoint. Order matters for SOC (first entry is the scalar $t$,
  rest form the vector) and for PSD (column-major `svec`).
  For `gen_power`, the first `len(alphas)` entries are the $p_i$
  components and the remainder are the $w$ components.
- `psd_k`: required when `kind='psd_triangle'`; the side length of the
  PSD matrix. Requires `len(indices) == psd_k * (psd_k + 1) // 2`.
- `alpha`: required when `kind='power'`; the power parameter $\alpha \in (0, 1)$.
- `alphas`: required when `kind='gen_power'`; list of positive floats
  summing to 1. `len(alphas)` is $\text{dim}_1$.
- `dim2`: required when `kind='gen_power'`; the size of the $w$-block,
  $\text{dim}_2 \geq 1$. Requires `len(indices) == len(alphas) + dim2`.

## Supported configurations

| Cone type         | CPU forward | CPU backward | CUDA forward | CUDA backward | torch / JAX |
|-------------------|-------------|--------------|--------------|---------------|-------------|
| `nonneg`          | ✓           | ✓            | ✓            | ✓             | ✓           |
| `soc` (dim ≤ 4)   | ✓           | ✓            | ✓ (dense)    | ✓ (dense)     | ✓           |
| `soc` (dim > 4)   | ✓           | ✓            | ✓ (rank-2)   | ✓ (rank-2)    | ✓           |
| `psd_triangle`    | ✓           | ✓            | ✓            | ✓             | ✓           |
| `exp`             | ✓           | ✓            | ✓            | ✓             | ✓           |
| `power`           | ✓           | ✓            | ✓            | ✓             | ✓           |
| `gen_power`       | ✓           | ✓            | ✓            | ✓             | ✓           |

All cone kinds and surfaces above are wired end-to-end. Batched direct-x
(`CompiledSolver` with `batch_size > 1`) works on CPU and CUDA for every
listed kind.

## KKT solver compatibility

| Linear solver                        | direct-x support |
|--------------------------------------|------------------|
| `auto`, `qdldl`, `faer-1t`, `faer-nt`| ✓                |
| Pardiso MKL / Panua                  | ✓ (untested)     |
| `cudss`                              | ✓ (CUDA-only)    |
| `woodbury`                           | ✓ for nonneg direct-x only (CUDA-only) |
| `riccati`                            | ✗ — block-tridiag structure breaks |

`woodbury` exploits diagonal P + nonneg-only cone structure; with nonneg
direct-x cones we add their per-iteration diagonal Hs contribution into
the same Schur-complement diagonal that the slack nonneg path builds.
SOC, PSD, exp, power, and gen_power direct-x are not Woodbury-compatible
because they introduce non-diagonal terms in the (1,1) block.

`riccati` aren't valid `direct_solve_method` values on CPU; `cudss` and
`woodbury` are CUDA-only and rejected on CPU at settings validation.

## Equilibration caveat

Moreau uses Ruiz equilibration by default. For SOC and PSD direct-x
cones, the cone constraint isn't invariant under arbitrary per-entry
diagonal scaling, so Moreau replaces the per-entry Ruiz scalings on
each cone's indices with their geometric mean. This is automatic; no
configuration needed.

## Chordal decomposition and direct-x PSD

Chordal decomposition in Moreau analyses the sparsity of the rows of
`[A ; b]` restricted to each slack PSD cone's row range, and splits
decomposable cones into smaller blocks. **Direct-x PSD cones are not
chordally decomposed**: they live in the primal vector rather than in a
slack-row block, so the sparsity-driven analysis does not apply. If
your PSD matrix variable has known block-diagonal structure, declare
the blocks as separate direct-x cones with disjoint `indices`.

## What direct-x does for you

A direct-x cone constrains a subvector of `x` in place, without
introducing the extra slack variable, `A` row, and `b` entry that the
equivalent slack constraint would require. The smaller problem is
typically faster to solve and to differentiate. Direct-x duals are
returned alongside the usual primal/dual solution as `solution.z_x`
(see :doc:`../api/core`).
