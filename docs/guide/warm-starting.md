# Warm Starting

Warm starting accelerates convergence when solving a sequence of related problems. By initializing the solver from a previous solution, the interior-point method starts closer to the optimum and typically converges in fewer iterations.

## When to Use Warm Starting

Warm starting is useful when you have:
- A sequence of problems with slowly varying parameters (e.g., MPC time steps)
- Sensitivity analysis where you perturb `q` or `b` slightly
- Iterative algorithms that solve similar subproblems repeatedly

```python
# Solve the initial problem
solver = moreau.CompiledSolver(n=n, m=m, ..., settings=settings)
solver.setup(P_values, A_values)
sol = solver.solve(qs=q_batch, bs=b_batch)

# Create warm start from previous solution
ws = sol.to_warm_start()

# Re-solve with perturbed parameters, warm starting from previous solution
sol2 = solver.solve(qs=q_perturbed, bs=b_perturbed, warm_start=ws)
```

---

## API

The `CompiledSolver.solve()` method accepts an optional `warm_start` argument:

```python
# Get warm start from a previous solution
ws = prev_solution.to_warm_start()  # BatchedSolution -> BatchedWarmStart

solution = solver.solve(
    qs=q_batch,          # (batch, n) linear cost vectors
    bs=b_batch,          # (batch, m) constraint RHS vectors
    warm_start=ws,       # BatchedWarmStart or WarmStart object
)
```

The `warm_start` parameter accepts a `WarmStart` (from `Solution.to_warm_start()`) or `BatchedWarmStart` (from `BatchedSolution.to_warm_start()`).

---

## Example: Perturbed QP

```python
import moreau
import numpy as np

# Problem structure
cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=4)
settings = moreau.Settings(batch_size=1)
solver = moreau.CompiledSolver(
    n=2, m=5,
    P_row_offsets=[0, 1, 2], P_col_indices=[0, 1],
    A_row_offsets=[0, 2, 3, 4, 5, 6],
    A_col_indices=[0, 1, 0, 1, 0, 1],
    cones=cones, settings=settings,
)
solver.setup(
    P_values=[2.0, 2.0],
    A_values=[1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
)

q = np.array([[1.0, -1.0]])
b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

# Cold solve
sol = solver.solve(qs=q, bs=b)
cold_iters = solver.info.iterations[0]

# Warm solve (same problem — converges faster)
ws = sol.to_warm_start()
sol_warm = solver.solve(qs=q, bs=b, warm_start=ws)
warm_iters = solver.info.iterations[0]

print(f"Cold: {cold_iters} iterations, Warm: {warm_iters} iterations")
```

---

## Batched Warm Starting

Warm start works with batched problems. Each problem in the batch gets its own warm start point:

```python
settings = moreau.Settings(batch_size=64)
solver = moreau.CompiledSolver(n=n, m=m, ..., settings=settings)
solver.setup(P_values, A_values)

# Solve first batch
sol = solver.solve(qs=q_batch, bs=b_batch)

# Warm start next batch from previous solutions
ws = sol.to_warm_start()  # BatchedWarmStart with shape (64, n/m)
sol2 = solver.solve(qs=q_batch_2, bs=b_batch_2, warm_start=ws)
```

---

## Shape Requirements

| Type | Field | Shape |
|------|-------|-------|
| `WarmStart` | `.x` | `(n,)` |
| `WarmStart` | `.z` | `(m,)` |
| `WarmStart` | `.s` | `(m,)` |
| `BatchedWarmStart` | `.x` | `(batch, n)` |
| `BatchedWarmStart` | `.z` | `(batch, m)` |
| `BatchedWarmStart` | `.s` | `(batch, m)` |

A `WarmStart` passed to a batched solver is automatically broadcast for `batch_size=1`.

---

## How It Works

When warm start vectors are provided, the solver:

1. **Scales** the warm start point into the equilibrated (internal) problem space
2. **Computes** a warmness ratio from the residuals and duality gap at the warm start point
3. **Applies central-path smoothing** to push the point onto the central path, ensuring the iterate is interior to the cones
4. **Starts the IPM** from this smoothed point instead of the default initialization

The warm start point should be in the original (unscaled) problem space — typically a previous `Solution`'s `x`, `z`, `s` arrays.

---

## Auto-Retry on Failure

If a warm-started solve produces a failure status (anything other than `Solved`, `AlmostSolved`, `MaxIterations`, or `CallbackTerminated`), the solver automatically retries without warm start and emits a warning.

You can customize which statuses are considered acceptable (i.e. do **not** trigger a cold retry) via `IPMSettings.warm_start_no_retry`:

```python
from moreau import IPMSettings, SolverStatus

# Also accept MaxTime without retrying cold
ipm = moreau.IPMSettings(
    warm_start_no_retry=frozenset({
        SolverStatus.Solved,
        SolverStatus.AlmostSolved,
        SolverStatus.MaxIterations,
        SolverStatus.MaxTime,
        SolverStatus.CallbackTerminated,
    })
)
```

---

## Single Problem Warm Starting

The `Solver` class also supports warm starting:

```python
import moreau
import numpy as np
from scipy import sparse

P = sparse.diags([1.0, 1.0], format='csr')
q = np.array([2.0, 1.0])
A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
b = np.array([1.0, 0.7, 0.7])
cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

solver = moreau.Solver(P, q, A, b, cones=cones)
solution = solver.solve()

# Warm start a new solver with slightly different q
ws = solution.to_warm_start()  # Solution -> WarmStart
solver2 = moreau.Solver(P, q + 0.1, A, b, cones=cones)
solution2 = solver2.solve(warm_start=ws)
```

---

## PyTorch and JAX Warm Starting

Both `moreau.torch.Solver` and `moreau.jax.Solver` support warm starting:

### PyTorch

```python
from moreau.torch import Solver
import torch

solver = Solver(n=2, m=3, ..., cones=cones)

solution = solver.solve(P_values, A_values, q, b)
ws = solution.to_warm_start()

# Re-solve with perturbed parameters
solution2 = solver.solve(P_values, A_values, q + 0.1, b, warm_start=ws)
```

### JAX

```python
from moreau.jax import Solver
import jax.numpy as jnp

solver = Solver(n=2, m=3, ..., cones=cones)
solver.setup(P_data, A_data)

solution = solver.solve(q, b)
ws = solution.to_warm_start()

# Re-solve with perturbed parameters
solution2 = solver.solve(q + 0.1, b, warm_start=ws)
```

---

## Notes

- **Bad warm starts are safe**: A poor warm start (e.g., random values) will still converge — it just won't reduce iterations
- **CPU and CUDA**: Warm starting is supported on both backends
- **Unchanged structure**: Warm starting only affects the initial point. The problem structure (sparsity pattern, cones) must remain the same as at construction time
