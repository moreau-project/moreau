# Quickstart

Solve your first convex optimization problem in 4 steps.

:::{admonition} Prerequisites
:class: tip

- Python 3.9+
- `pip install moreau`
- For gradients: PyTorch or JAX
:::

```{raw} html
<style>
.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 2rem;
    height: 2rem;
    background: var(--color-brand-primary);
    color: white;
    border-radius: 50%;
    font-weight: bold;
    margin-right: 0.75rem;
}
</style>
```

---

## <span class="step-number">1</span> Define the Problem

Create a convex conic optimization problem:

```python
import moreau
import numpy as np
from scipy import sparse

# Problem dimensions
n = 2  # variables
m = 3  # constraints

# Objective: (1/2)x'Px + q'x
P = sparse.diags([1.0, 1.0], format='csr')  # Quadratic term
q = np.array([2.0, 1.0])                     # Linear term

# Constraints: Ax + s = b, s in K
A = sparse.csr_array([
    [1.0, 1.0],  # x1 + x2 = 1 (equality via zero cone)
    [1.0, 0.0],  # x1 >= 0.7 (inequality via nonneg cone)
    [0.0, 1.0],  # x2 >= 0.7
])
b = np.array([1.0, 0.7, 0.7])
```

:::{tip}
**Sparse matrices** are recommended for performance. Use `scipy.sparse` CSR format.
:::

---

## <span class="step-number">2</span> Specify Cones

Define the cone structure for your constraints:

```python
# First constraint uses zero cone (equality)
# Next two use nonnegative cone (inequalities)
cones = moreau.Cones(
    num_zero_cones=1,     # 1 equality constraint
    num_nonneg_cones=2,   # 2 inequality constraints
)
```

:::{dropdown} Available Cones
| Cone | Description | Typical Use |
|------|-------------|-------------|
| `num_zero_cones` | Equality: $s = 0$ | Linear equalities |
| `num_nonneg_cones` | Inequality: $s \ge 0$ | Linear inequalities |
| `so_cone_dims` | Second-order (dim $\ge 2$) | Norm constraints |
| `num_exp_cones` | Exponential (dim 3) | Log/exp constraints |
| `power_alphas` | Power (dim 3) | Power function constraints |
| `gen_power_cone_params` | Generalized power (variable dim) | Geometric mean constraints |
:::

---

## <span class="step-number">3</span> Create Solver & Solve

Create a solver and solve the problem:

```python
# Create solver with problem data
solver = moreau.Solver(P, q, A, b, cones=cones)

# Solve
solution = solver.solve()

print(f"Optimal x: {solution.x}")
print(f"Status: {solver.info.status}")
print(f"Objective: {solver.info.obj_val}")
```

---

## <span class="step-number">4</span> Check Results

Access solution and solver information:

```python
# Primal solution
x = solution.x  # Optimal x values

# Dual variables
z = solution.z  # Dual variables
s = solution.s  # Slack variables

# Solver metadata
info = solver.info
print(f"Status: {info.status}")           # SolverStatus.Solved
print(f"Objective: {info.obj_val:.4f}")   # Optimal objective value
print(f"Iterations: {info.iterations}")   # IPM iterations
print(f"Solve time: {info.solve_time:.4f}s")
```

---

## Complete Example

Here's everything together:

```python
import moreau
import numpy as np
from scipy import sparse

# 1. Define problem
P = sparse.diags([1.0, 1.0], format='csr')
q = np.array([2.0, 1.0])
A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
b = np.array([1.0, 0.7, 0.7])

# 2. Specify cones
cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

# 3. Solve
solver = moreau.Solver(P, q, A, b, cones=cones)
solution = solver.solve()

# 4. Results
print(f"x = {solution.x}")
print(f"Status: {solver.info.status}")
```

---

## Batched Solving

For multiple problems with the same structure, use `CompiledSolver`:

```python
import moreau
import numpy as np

# Define structure (shared across batch)
cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
settings = moreau.Settings(batch_size=4)

solver = moreau.CompiledSolver(
    n=2, m=3,
    P_row_offsets=[0, 1, 2], P_col_indices=[0, 1],
    A_row_offsets=[0, 2, 3, 4], A_col_indices=[0, 1, 0, 1],
    cones=cones,
    settings=settings,
)

# Set matrix values (can be shared or per-problem)
solver.setup(
    P_values=[1.0, 1.0],  # Shared across batch
    A_values=[1.0, 1.0, 1.0, 1.0],
)

# Solve batch
qs = np.array([[2.0, 1.0]] * 4)
bs = np.array([[1.0, 0.7, 0.7]] * 4)
solution = solver.solve(qs, bs)

print(f"Batch solutions shape: {solution.x.shape}")  # (4, 2)
print(f"First status: {solver.info.status[0]}")
```

---

## Next Steps

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} PyTorch Integration
:link: api/torch
:link-type: doc

Use optimization layers in neural networks with autograd support.
:::

:::{grid-item-card} JAX Integration
:link: api/jax
:link-type: doc

Functional API compatible with vmap, jit, and grad.
:::

:::{grid-item-card} Batching Guide
:link: guide/batching
:link-type: doc

Solve thousands of problems in parallel for maximum throughput.
:::

:::{grid-item-card} Examples
:link: examples/index
:link-type: doc

Real-world applications: control, portfolio optimization, and more.
:::

::::
