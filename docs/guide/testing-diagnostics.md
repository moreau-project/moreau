# Testing & Diagnostics

Moreau provides utilities for generating test problems and verifying your installation.

## Diagnostic CLI

Verify your installation and backend availability:

```bash
python -m moreau check
```

This runs a comprehensive diagnostic that reports:
- Installed package versions (moreau, moreau-cpu, moreau-cuda)
- Available device backends and default device
- Optional dependencies (PyTorch, JAX, CuPy, SciPy, NumPy)
- CPU and CUDA solver tests with a simple QP
- PyTorch interface and autograd tests
- JAX interface test (solve a simple QP via `moreau.jax.Solver`)
- CuPy zero-copy test (pass CuPy device pointers straight to the CUDA solver —
  the same zero-copy path PyTorch/JAX use — with no host transfer)

Each interface test reports "not installed" when the optional dependency is
absent, so the diagnostic runs cleanly on any environment.

Example output:

```
============================================================
Moreau Solver Diagnostics
============================================================

Core Packages:
----------------------------------------
  ✓ moreau: 0.3.3
  ✓ moreau-cpu: 0.3.3
  ✓ moreau-cuda: 0.3.3

Device Backends:
----------------------------------------
  ✓ Available devices: cpu, cuda
  ✓ Default device: cuda

...
```

Use the `-v` flag for verbose output:

```bash
python -m moreau check -v
```

---

## Random Problem Generation

The `moreau.testing` module provides utilities for generating random feasible
conic programs, useful for benchmarking, testing, and development.

### RandomProblem

A container for a randomly generated cone program:

```python
from moreau.testing import RandomProblem

# RandomProblem fields:
# .P    - Quadratic objective matrix (n x n), CSR format, symmetric PSD
# .q    - Linear objective vector (n,)
# .A    - Constraint matrix (m x n), CSR format
# .b    - Constraint RHS (m,)
# .cones - Cone specification
# .x_feas - Known feasible primal point (n,)
# .s_feas - Known feasible slack point in int(K) (m,)
# .z_feas - Known feasible dual point in int(K*) (m,), if available
```

### random_cone_program

Generate a single random feasible cone program:

```python
from moreau.testing import random_cone_program
import moreau

# Generate a random QP with mixed cones
cones = moreau.Cones(
    num_zero_cones=2,
    num_nonneg_cones=5,
    so_cone_dims=[3, 5],
)
problem = random_cone_program(n=10, cones=cones, seed=42)

# Solve it
solver = moreau.Solver(problem.P, problem.q, problem.A, problem.b, problem.cones)
solution = solver.solve()
assert solver.info.status == moreau.SolverStatus.Solved
```

The problem is guaranteed feasible by construction: it samples a point in the
cone interior, generates a random constraint matrix, and sets `b = Ax + s` to
ensure feasibility.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | — | Number of variables |
| `cones` | `Cones` | — | Cone specification (determines m) |
| `density` | `float` | 0.3 | Sparsity of A and P (0 to 1) |
| `seed` | `int` | None | Random seed for reproducibility |

### random_cone_program_with_pattern

Generate a random problem with a specific CSR sparsity pattern:

```python
from moreau.testing import random_cone_program_with_pattern

# Reuse the pattern from an existing problem
problem2 = random_cone_program_with_pattern(
    n=10,
    cones=cones,
    P_row_offsets=problem.P.indptr,
    P_col_indices=problem.P.indices,
    A_row_offsets=problem.A.indptr,
    A_col_indices=problem.A.indices,
    seed=123,
)
```

### random_batch

Generate a batch of random problems with shared sparsity structure, suitable for
testing `CompiledSolver`:

```python
from moreau.testing import random_batch

cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=5)
first, problems = random_batch(
    n=10,
    cones=cones,
    batch_size=64,
    density=0.3,
    seed=42,
)

# Use the shared pattern for CompiledSolver
import numpy as np

settings = moreau.Settings(batch_size=64)
solver = moreau.CompiledSolver(
    n=10,
    m=cones.total_constraints(),
    P_row_offsets=first.P.indptr.tolist(),
    P_col_indices=first.P.indices.tolist(),
    A_row_offsets=first.A.indptr.tolist(),
    A_col_indices=first.A.indices.tolist(),
    cones=cones,
    settings=settings,
)

# Stack values for batched solve
P_values = np.array([p.P.data for p in problems])
A_values = np.array([p.A.data for p in problems])
qs = np.array([p.q for p in problems])
bs = np.array([p.b for p in problems])

solver.setup(P_values, A_values)
solution = solver.solve(qs, bs)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | — | Number of variables |
| `cones` | `Cones` | — | Cone specification |
| `batch_size` | `int` | — | Number of problems to generate |
| `density` | `float` | 0.3 | Sparsity of A and P |
| `seed` | `int` | None | Random seed |
| `shared_pattern` | `bool` | True | If True, all problems share P/A sparsity pattern |

**Returns:** `(first_problem, list_of_all_problems)` — the first problem can be
used to extract the shared CSR pattern.

### Cone Interior Sampling

Lower-level utilities for generating feasible points:

```python
from moreau.testing import sample_cone_interior, sample_dual_cone_interior
import numpy as np

cones = moreau.Cones(num_nonneg_cones=3, so_cone_dims=[3])
rng = np.random.default_rng(42)

s = sample_cone_interior(cones, rng)      # Point in int(K)
z = sample_dual_cone_interior(cones, rng)  # Point in int(K*)
```
