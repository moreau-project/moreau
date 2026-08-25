# Basic Usage

This guide covers the fundamental usage patterns for Moreau.

## Single Problem Solving

For solving a single optimization problem, use the `Solver` class:

```python
import moreau
import numpy as np
from scipy import sparse

# Define problem data
P = sparse.diags([1.0, 2.0], format='csr')
q = np.array([1.0, 1.0])
A = sparse.csr_array([[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
b = np.array([1.0, 0.0, 0.0])

# Define cones
cones = moreau.Cones(
    num_zero_cones=1,    # First row: equality
    num_nonneg_cones=2,  # Remaining rows: inequalities
)

# Create solver and solve
solver = moreau.Solver(P, q, A, b, cones=cones)
solution = solver.solve()

# Access results
print(f"Optimal x: {solution.x}")
print(f"Dual variables: {solution.z}")
print(f"Slack: {solution.s}")
```

---

## Accessing Solver Information

After calling `solve()`, metadata is available via `solver.info`:

```python
solution = solver.solve()
info = solver.info

# Status
print(f"Status: {info.status}")  # SolverStatus.Solved

# Objective value
print(f"Objective: {info.obj_val}")

# Timing
print(f"Solve time: {info.solve_time:.4f}s")
print(f"Setup time: {info.setup_time:.4f}s")
print(f"Construction time: {info.construction_time:.4f}s")

# Iterations
print(f"IPM iterations: {info.iterations}")
```

---

## Solver Status

The solver returns a status indicating the solve outcome:

```python
from moreau import SolverStatus

status = solver.info.status

if status == SolverStatus.Solved:
    print("Optimal solution found")
elif status == SolverStatus.PrimalInfeasible:
    print("Problem is infeasible")
elif status == SolverStatus.DualInfeasible:
    print("Problem is unbounded")
elif status == SolverStatus.MaxIterations:
    print("Maximum iterations reached")
```

:::{dropdown} All Status Values
| Status | Value | Meaning |
|--------|-------|---------|
| `Unsolved` | 0 | Not yet solved |
| `Solved` | 1 | Optimal solution found |
| `PrimalInfeasible` | 2 | No feasible solution exists |
| `DualInfeasible` | 3 | Problem is unbounded |
| `AlmostSolved` | 4 | Near-optimal (relaxed tolerances met) |
| `AlmostPrimalInfeasible` | 5 | Likely infeasible (relaxed tolerances) |
| `AlmostDualInfeasible` | 6 | Likely unbounded (relaxed tolerances) |
| `MaxIterations` | 7 | Iteration limit reached |
| `MaxTime` | 8 | Time limit reached |
| `NumericalError` | 9 | Numerical issues encountered |
| `InsufficientProgress` | 10 | Solver stalled |
| `CallbackTerminated` | 11 | Terminated by user callback |

`SolverStatus` is an `IntEnum` — you can compare with integers: `status == 1` is equivalent to `status == SolverStatus.Solved`.
:::

---

## Matrix Requirements

### P Matrix (Quadratic Objective)

The P matrix must be:
- **Symmetric**: Both upper and lower triangles must be provided (full symmetric storage). The Python API validates symmetry and raises ``ValueError`` otherwise.
- **Positive semidefinite**: For convex problems
- **Sparse**: CSR format recommended

:::{tip}
If you have a matrix that may not be exactly symmetric (e.g. from numerical
construction), symmetrize it before passing to Moreau: ``P = (P + P.T) / 2``.
:::

```python
# Correct: Full symmetric matrix
P = sparse.csr_array([
    [2.0, 1.0],
    [1.0, 2.0],
])

# Incorrect: Upper triangle only (will raise error)
P_bad = sparse.csr_array([
    [2.0, 1.0],
    [0.0, 2.0],
])
```

### A Matrix (Constraints)

The constraint matrix A has shape `(m, n)` where:
- `m` = total number of cone constraints
- `n` = number of variables

```python
# m=3 constraints, n=2 variables
A = sparse.csr_array([
    [1.0, 1.0],   # Constraint 1
    [1.0, 0.0],   # Constraint 2
    [0.0, 1.0],   # Constraint 3
])
```

---

## Cone Specification

Cones define the constraint types. The order matters:

```python
cones = moreau.Cones(
    num_zero_cones=1,      # Equality constraints (first)
    num_nonneg_cones=3,    # Inequality constraints
    so_cone_dims=[3, 5],   # Second-order cones (dim 3 and dim 5)
    num_exp_cones=1,       # Exponential cones (dim 3)
    power_alphas=[0.5],    # Power cones (dim 3 each)
    gen_power_cone_params=[([0.3, 0.7], 2)],  # GenPowerCone: 2 alphas, dim2=2 (total dim=4)
    psd_dims=[4],          # PSD cones (4x4 matrix = 10 svec vars)
)

# Total constraints = 1 + 3 + (3+5) + 1*3 + 1*3 + 4 + 10 = 32
```

:::{note}
Each second-order cone has an arbitrary dimension >= 2, specified via
`so_cone_dims`. Each generalized power cone has variable dimension
determined by `(alphas, dim2)` — total dimension is `len(alphas) + dim2`.
:::

### Cone Definitions

**Second-order cone (SOC)** — arbitrary dimension $d \ge 2$:

$$\mathcal{K}_{\text{soc}}^d = \{s \in \mathbb{R}^d : \|s_{1:}\|_2 \le s_0 \}$$

**Exponential cone:**

$$\mathcal{K}_{\exp} = \mathrm{cl}\{(s_1, s_2, s_3) \in \mathbb{R}^3 : s_2 \exp(s_1 / s_2) \le s_3,\; s_2 > 0 \}$$

**Power cone** with parameter $\alpha \in (0, 1)$:

$$\mathcal{K}_{\text{pow}}(\alpha) = \{(s_1, s_2, s_3) \in \mathbb{R}^3 : s_1^{\alpha}\, s_2^{1-\alpha} \ge |s_3|,\; s_1 \ge 0,\; s_2 \ge 0 \}$$

**Generalized power cone** with parameters $\alpha_i > 0$, $\sum \alpha_i = 1$:

$$\mathcal{K}_{\text{gpow}}(\alpha, d_2) = \{(p, w) \in \mathbb{R}^{d_1 + d_2} : \prod_{i=1}^{d_1} p_i^{\alpha_i} \ge \|w\|_2,\; p_i \ge 0 \}$$

where $d_1 = |\alpha|$ is the number of power components and $d_2 \ge 1$ is the norm dimension. The total cone dimension is $d_1 + d_2$. Specified via `gen_power_cone_params=[(alphas, dim2), ...]`.

:::{note}
CUDA backward (gradient) accuracy for generalized power cones with total dimension $\ge 15$ is a known limitation. CPU backward works at all dimensions. For high-dimensional GenPowerCone gradients, use `device='cpu'` or verify with finite differences.
:::

**PSD cone** (positive semidefinite) — matrix dimension $d$, svec length $d(d+1)/2$:

$$\mathcal{K}_{\text{psd}}^d = \{s \in \mathbb{R}^{d(d+1)/2} : \text{mat}(s) \succeq 0 \}$$

where $\text{mat}(s)$ reconstructs the symmetric matrix from svec form (column-major upper triangle, off-diagonals scaled by $\sqrt{2}$). See [PSD Cones](psd-cones.md) for details.

### Constraint Ordering

Constraints in A and b must be ordered to match the cone specification:
1. Zero cone rows first
2. Nonnegative cone rows
3. Second-order cone rows (one block per cone, size = cone dimension)
4. Exponential cone rows (3 per cone)
5. Power cone rows (3 per cone)
6. Generalized power cone rows (one block per cone, size = `len(alphas) + dim2`)
7. PSD cone rows ($d(d+1)/2$ per cone)

---

## Settings

Customize solver behavior with `Settings`:

```python
settings = moreau.Settings(
    solver='auto',        # Solver algorithm ('auto', 'ipm', 'active_set')
    device='auto',        # 'auto', 'cpu', or 'cuda'
    device_id=-1,         # CUDA device ID (-1 = current device, ignored on CPU)
    batch_size=1,         # For CompiledSolver
    enable_grad=False,    # Enable gradient computation
    max_iter=200,         # Maximum iterations
    time_limit=float('inf'),  # Time limit in seconds
    verbose=False,        # Print solver progress
    ipm_settings=None,    # Fine-grained IPM control
    active_set_settings=None,  # Active-set solver control
)

solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
```

### IPM Settings

For fine-grained control over the interior-point method:

```python
ipm_settings = moreau.IPMSettings(
    tol_gap_abs=1e-8,    # Absolute gap tolerance
    tol_gap_rel=1e-8,    # Relative gap tolerance
    tol_feas=1e-8,       # Feasibility tolerance
)

settings = moreau.Settings(
    ipm_settings=ipm_settings,
)
```

See [Solver Settings](solver-settings.md) for the full list of IPM parameters.

---

## Implicit Differentiation

The `Solver` class supports implicit differentiation through `backward()`:

```python
settings = moreau.Settings(enable_grad=True)
solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
solution = solver.solve()

# Compute gradients of a scalar loss w.r.t. problem data
# dl_dx, dl_dz, dl_ds are the gradients of your loss w.r.t. x, z, s
dl_dP, dl_dq, dl_dA, dl_db = solver.backward(dl_dx, dl_dz, dl_ds)
```

For PyTorch and JAX, gradient computation is handled automatically by the
framework's autograd system — see the [PyTorch](../api/torch) and
[JAX](../api/jax) API docs.

---

## Error Handling

Moreau validates inputs and raises descriptive errors:

```python
try:
    solver = moreau.Solver(P, q, A, b, cones=cones)
    solution = solver.solve()
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Solver error: {e}")
```

Common validation errors:
- Dimension mismatches between P, q, A, b
- Non-symmetric P matrix
- Cone dimensions don't sum to constraint count
- Invalid CSR structure
