# PSD (SDP) Cones

Moreau supports positive semidefinite (PSD) cone constraints for semidefinite
programming (SDP). PSD cones enable optimization over symmetric positive
semidefinite matrices, with applications in control (Lyapunov stability, MPC),
sum-of-squares programming, and covariance estimation.

:::{note}
PSD cones are supported on both **CPU and CUDA** backends. GPU support includes
forward solve, backward pass (differentiation), warm-start smoothing, batch
mode, and chordal decomposition for sparse PSD cones.
:::

## Quick Start

```python
import moreau
import numpy as np
from scipy import sparse

# 3x3 PSD cone => 6 svec variables (upper triangle in column-major order)
n = 6
m = 6

P = sparse.eye(n, format='csr')
A = sparse.eye(m, format='csr')
q = np.zeros(n)
b = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])  # svec of identity matrix

cones = moreau.Cones(psd_dims=[3])

solver = moreau.Solver(P, q, A, b, cones=cones)
solution = solver.solve()
```

---

## Svec Representation

PSD cone constraints operate on symmetric matrices stored in **svec (scaled
vectorization)** format. For a symmetric $n \times n$ matrix $X$, the svec
representation is a vector of length $n(n+1)/2$ containing the upper triangle
in column-major order, with off-diagonal entries scaled by $\sqrt{2}$:

$$
\text{svec}(X) = [X_{00},\; \sqrt{2}\,X_{01},\; X_{11},\; \sqrt{2}\,X_{02},\; \sqrt{2}\,X_{12},\; X_{22},\; \ldots]
$$

The $\sqrt{2}$ scaling preserves inner products: $\langle X, Y \rangle_F = \text{svec}(X)^\top \text{svec}(Y)$.

| Matrix dim | svec length | Example |
|-----------|-------------|---------|
| 1 | 1 | Scalar (equivalent to nonneg cone) |
| 2 | 3 | 2x2 symmetric |
| 3 | 6 | 3x3 symmetric |
| 4 | 10 | 4x4 symmetric |
| $n$ | $n(n+1)/2$ | General |

### Svec Indexing

The svec index for matrix entry $(i, j)$ with $i \le j$ is:

$$
\text{idx}(i, j) = \frac{j(j+1)}{2} + i
$$

```python
def svec_index(i, j):
    """Column-major upper triangle index."""
    if i > j:
        i, j = j, i
    return j * (j + 1) // 2 + i
```

---

## Cone Specification

Specify PSD cones via the `psd_dims` parameter, which takes a list of **matrix
dimensions** (not svec lengths):

```python
# Single 4x4 PSD cone (10 svec variables)
cones = moreau.Cones(psd_dims=[4])

# Two PSD cones: 3x3 and 2x2 (6 + 3 = 9 svec variables)
cones = moreau.Cones(psd_dims=[3, 2])

# Mixed: equality + inequality + PSD
cones = moreau.Cones(
    num_zero_cones=2,
    num_nonneg_cones=3,
    psd_dims=[4],
)
```

### Constraint Ordering

PSD cone rows come last in the constraint matrix, after all other cones:

1. Zero cone rows
2. Nonnegative cone rows
3. Second-order cone rows
4. Exponential cone rows
5. Power cone rows
6. **PSD cone rows** (svec length per cone)

```python
# 2 equality + 3 inequality + PSD(3) = 2 + 3 + 6 = 11 constraints
cones = moreau.Cones(num_zero_cones=2, num_nonneg_cones=3, psd_dims=[3])
assert cones.total_constraints() == 11
```

---

## Chordal Decomposition

For PSD cones with **sparse constraint structure**, Moreau automatically
applies chordal decomposition to split a large PSD cone into smaller ones.
This can dramatically reduce solve time.

Chordal decomposition fires when the constraint matrix $A$ has block-diagonal
or near-block-diagonal sparsity within the PSD cone rows. The solver detects
the sparsity pattern and decomposes the cone structure at construction time.

### When It Helps

Chordal decomposition is most effective when:
- The PSD matrix has known block-diagonal or banded structure
- The problem comes from an SOS (sum-of-squares) program with separable variables
- The underlying graph of the matrix is chordal or nearly chordal

### Performance

| PSD size | Dense | Chordal (block_dim=2) | Speedup |
|----------|-------|----------------------|---------|
| PSD(10) | 5ms | 12ms | 0.4x |
| PSD(20) | 20ms | 11ms | **1.7x** |
| PSD(40) | 153ms | 7ms | **20.7x** |

At larger sizes, chordal decomposition provides order-of-magnitude speedups.
The backward pass benefits even more (46x at PSD(40)) because the decomposed
KKT system is much smaller.

:::{note}
Chordal decomposition is fully automatic in both `Solver` and
`CompiledSolver`. No user configuration is needed — the solver detects
exploitable sparsity from the constraint matrix structure.

The clique merge strategy is controlled by
`IPMSettings.chordal_decomposition_merge_method` (default `'clique_graph'`;
also accepts `'parent_child'` and `'none'`).

CPU honors this setting today. CUDA currently hardcodes a fill_in==0
parent-child variant regardless of the setting — so a sparse PSD problem
may decompose differently on CPU than CUDA at present. The setting field
exists on `_CudaSettings` as forward-compatibility scaffolding while the
CUDA chordal path is wired through to honor it (tracked in #176).
:::

---

## Batched SDP (CompiledSolver)

PSD cones work with `CompiledSolver` for batched solving. The sparsity
pattern is fixed at construction; different `q` and `b` vectors (and
optionally different `P`/`A` values) are provided per batch element.

```python
import moreau
import numpy as np

mat_dim = 4
svec_dim = mat_dim * (mat_dim + 1) // 2  # 10
n = svec_dim
m = svec_dim
batch_size = 16

# P = I, A = I (CSR structure)
P_ro = list(range(n + 1))
P_ci = list(range(n))
A_ro = list(range(m + 1))
A_ci = list(range(m))

cones = moreau.Cones(psd_dims=[mat_dim])
settings = moreau.Settings(batch_size=batch_size, device='cpu')

solver = moreau.CompiledSolver(
    n=n, m=m,
    P_row_offsets=P_ro, P_col_indices=P_ci,
    A_row_offsets=A_ro, A_col_indices=A_ci,
    cones=cones, settings=settings,
)

# Setup shared P/A values
P_values = [1.0] * n
A_values = [1.0] * m
solver.setup(P_values=P_values, A_values=A_values)

# Solve batch with different b vectors
qs = [np.zeros(n) for _ in range(batch_size)]
bs = [np.random.randn(m) for _ in range(batch_size)]
solutions = solver.solve(qs=qs, bs=bs)
```

---

## Backward Pass (Differentiation)

PSD cones support implicit differentiation through the backward pass,
enabling gradient-based learning through SDP constraints.

```python
settings = moreau.Settings(enable_grad=True, device='cpu')
solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
solution = solver.solve()

# Compute gradients
grads = solver.backward(
    dx=np.ones(n),
    dz=np.zeros(m),
    ds=np.zeros(m),
)
# grads['dq'], grads['db'], grads['dP_values'], grads['dA_values']
```

### PyTorch Integration

```python
from moreau.torch import Solver

solver = Solver(
    n=n, m=m,
    P_row_offsets=P_ro, P_col_indices=P_ci,
    A_row_offsets=A_ro, A_col_indices=A_ci,
    cones=moreau.Cones(psd_dims=[3]),
)
solver.setup(P_values_tensor, A_values_tensor)
solution = solver.solve(q_tensor, b_tensor)
solution.x.sum().backward()  # gradients flow through PSD projection
```

---

## Application: Lyapunov Stability (MPC Terminal Constraints)

A common application is computing Lyapunov functions for MPC terminal
constraints. Given a discrete-time linear system $x_{k+1} = A_{\text{sys}} x_k$,
find a positive definite matrix $P$ such that:

$$
P \succeq \varepsilon I, \quad P - A_{\text{sys}}^\top P A_{\text{sys}} \succeq \varepsilon I
$$

This certifies that $V(x) = x^\top P x$ is a Lyapunov function, providing a
terminal cost and invariant set $\{x : V(x) \le 1\}$ for MPC.

In conic form, the decision variable is $\text{svec}(P)$ and the constraints
are two PSD cones (one for $P \succeq \varepsilon I$ and one for the Lyapunov
decrease condition).

For systems with **decoupled subsystems** ($A_{\text{sys}}$ block-diagonal),
the constraint matrices inherit block-diagonal sparsity, and chordal
decomposition splits each PSD cone into smaller blocks automatically.

The `sos_mpc` example in the repository demonstrates this formulation:

```bash
# Rust example: Lyapunov SDP for 3 decoupled 2D subsystems
cd packages/moreau-cpu
cargo run --release --features sdp-openblas --example sos_mpc -- 256
```

---

## Limitations

- **Double precision**: All computations use float64
- **Large dense SDPs**: For problems without exploitable sparsity, solve
  time scales as $O(n^3)$ in the matrix dimension. Consider reformulating
  with sparse structure when possible.
