# JAX Integration

Moreau provides a first-class JAX integration that is compatible with `jax.grad`, `jax.vmap`, and `jax.jit`. It supports automatic device selection between CPU and CUDA backends and provides gradients via implicit differentiation.

## Quick Start

```python
import jax
import jax.numpy as jnp
from moreau.jax import Solver
import moreau

# 1. Define problem structure
cones = moreau.Cones(num_nonneg_cones=2)
solver = Solver(
    n=2, m=2,
    P_row_offsets=jnp.array([0, 1, 2]),
    P_col_indices=jnp.array([0, 1]),
    A_row_offsets=jnp.array([0, 1, 2]),
    A_col_indices=jnp.array([0, 1]),
    cones=cones,
)

# 2. Solve
P_data = jnp.array([1.0, 1.0])
A_data = jnp.array([1.0, 1.0])
q = jnp.array([1.0, 1.0])
b = jnp.array([0.5, 0.5])

solution = solver.solve(P_data, A_data, q, b)
print(solution.x)
```

---

## Two-Step API for Performance

If your problem matrices $P$ and $A$ are constant across many solves (but you still want to differentiate through them), use `setup()` to set them once. This avoids redundant preprocessing in subsequent `solve()` calls.

```python
# Set matrices once
solver.setup(P_data, A_data)

# Solve with only q and b
solution = solver.solve(q, b)

# Gradients w.r.t. P_data and A_data are still computed!
def loss_fn(P_vals):
    solver.setup(P_vals, A_data)
    return jnp.sum(solver.solve(q, b).x)

grad_P = jax.grad(loss_fn)(P_data)
```

---

## Differentiable Optimization

Moreau's JAX solver is fully differentiable. You can use `jax.grad` to compute the gradient of any scalar loss function with respect to the problem data ($P, A, q, b$).

```python
def loss_fn(q_val):
    solution = solver.solve(P_data, A_data, q_val, b)
    return jnp.sum(jnp.square(solution.x))

# Compute gradient w.r.t. q
q_grad = jax.grad(loss_fn)(q)
```

### Implicit Differentiation

Gradients are computed using the **implicit function theorem** on the KKT conditions of the optimization problem. This is much more memory-efficient than unrolling the solver's iterations and allows for differentiating through exact solutions.

---

## Batching with `jax.vmap`

The `Solver.solve` method is compatible with `jax.vmap`. You can batch over any combination of the input parameters.

```python
# Batch over q and b
batched_solve = jax.vmap(solver.solve, in_axes=(None, None, 0, 0))

q_batch = jax.random.normal(jax.random.PRNGKey(0), (64, 2))
b_batch = jnp.ones((64, 2))

solutions = batched_solve(P_data, A_data, q_batch, b_batch)
print(solutions.x.shape)  # (64, 2)
```

### Accessing Solve Info

After a (non-`vmap`) `solve()` call, solve metadata is available via
`solver.info` (status, objective value, iteration count, timings):

```python
solution = solver.solve(P_data, A_data, q, b)
print(int(solver.info.status))   # SolverStatus integer value
print(solver.info.solve_time)
```

When using `vmap`, `solver.info` only reflects the *last* internal call, so it
should not be relied on for per-problem batched metadata.

---

## JIT Compilation

The `Solver` class JIT-compiles its internal `solve` method by default during construction. This ensures maximum performance on the first call.

- To disable automatic JIT: `Solver(..., jit=False)`
- To JIT your own wrapper:
  ```python
  @jax.jit
  def my_custom_solve(q):
      return solver.solve(q, b).x
  ```

---

## Warm Starting in JAX

Warm starting is useful for iterative algorithms or MPC where the next problem is similar to the previous one.

```python
# 1. Initial solve
solution = solver.solve(P_data, A_data, q, b)

# 2. Get warm start point
ws = solution.to_warm_start()

# 3. Solve next problem with warm start
solution2 = solver.solve(P_data, A_data, q_new, b_new, warm_start=ws)
```

---

## Device Selection

Moreau's JAX integration automatically detects and uses the best available device.

- If CUDA is available and Moreau was built with CUDA support, it will use the GPU.
- Otherwise, it falls back to the CPU.

You can force a device via `Settings`:

```python
settings = moreau.Settings(device='cpu')
solver = Solver(..., settings=settings)
```

---

## Best Practices

1.  **Use `float64`**: Moreau performs all internal calculations in double precision. Ensure your JAX inputs are `jnp.float64` for best results.
    ```python
    jax.config.update("jax_enable_x64", True)
    ```
2.  **Pre-construct Solvers**: Avoid creating `Solver` objects inside JIT-compiled functions or loops. Construct them once and reuse them.
3.  **Two-Step API**: Use `setup(P, A)` whenever $P$ and $A$ are constant to skip redundant factorization steps.
4.  **Auto-Tune**: By default, Moreau uses a heuristic to select the KKT solver without benchmarking. Set `auto_tune=True` in `Settings` to benchmark different KKT solvers on the first solve and lock in the fastest. Alternatively, set `direct_solve_method` explicitly in `IPMSettings` to skip selection entirely.
