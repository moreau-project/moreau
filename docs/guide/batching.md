# Batched Solving

Moreau excels at solving many problems with shared structure. The `CompiledSolver` API is optimized for high-throughput batched workloads.

## When to Use Batching

Use `CompiledSolver` when you have:
- Multiple problems with the same sparsity pattern
- Need for parallel GPU execution
- Training loops where P/A structure is fixed

```python
# Batched: 1000 similar QPs
settings = moreau.Settings(batch_size=1000, device='cuda')
solver = moreau.CompiledSolver(n, m, ..., settings=settings)
solver.setup(P_values, A_values)
solutions = solver.solve(qs, bs)  # Solves all 1000 in parallel
```

---

## Three-Step API

`CompiledSolver` uses a three-step pattern:

### 1. Construct with Structure

Define the problem structure using CSR format:

```python
import moreau
import numpy as np

cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
settings = moreau.Settings(batch_size=64)

solver = moreau.CompiledSolver(
    n=2,  # Number of variables
    m=3,  # Number of constraints

    # P matrix structure (CSR)
    P_row_offsets=[0, 1, 2],   # Row pointers
    P_col_indices=[0, 1],      # Column indices

    # A matrix structure (CSR)
    A_row_offsets=[0, 2, 3, 4],
    A_col_indices=[0, 1, 0, 1],

    cones=cones,
    settings=settings,
)
```

### 2. Setup Matrix Values

Set the numerical values for P and A matrices:

```python
# Shared across batch (1D)
P_values = np.array([1.0, 1.0])
A_values = np.array([1.0, 1.0, 1.0, 1.0])
solver.setup(P_values, A_values)

# Or per-problem (2D): shape (batch_size, nnz)
P_values_batch = np.tile([1.0, 1.0], (64, 1))
A_values_batch = np.tile([1.0, 1.0, 1.0, 1.0], (64, 1))
solver.setup(P_values_batch, A_values_batch)
```

### 3. Solve with Parameters

Solve the batch with q and b vectors:

```python
# Shape: (batch_size, n) and (batch_size, m)
qs = np.random.randn(64, 2)
bs = np.ones((64, 3))

solution = solver.solve(qs, bs)

# Results have batch dimension
print(solution.x.shape)  # (64, 2)
print(solution.z.shape)  # (64, 3)
```

---

## Value Broadcasting

Both `setup()` and `solve()` support broadcasting:

### Shared Values (1D input)

```python
# Same P and A for all problems
solver.setup(
    P_values=[1.0, 1.0],      # Shape (nnz_P,) - shared
    A_values=[1.0, 1.0, 1.0, 1.0],  # Shape (nnz_A,) - shared
)
```

### Per-Problem Values (2D input)

```python
# Different P and A for each problem
solver.setup(
    P_values=P_batch,   # Shape (batch, nnz_P)
    A_values=A_batch,   # Shape (batch, nnz_A)
)
```

---

## Accessing Batch Results

Solution and info are batched:

```python
solution = solver.solve(qs, bs)
info = solver.info

# Solutions have batch dimension
x = solution.x  # Shape (batch, n)
z = solution.z  # Shape (batch, m)
s = solution.s  # Shape (batch, m)

# Info has list of statuses
statuses = info.status  # List of SolverStatus
obj_vals = info.obj_val  # Array of shape (batch,)

# Check specific problem
print(f"Problem 0: status={statuses[0]}, obj={obj_vals[0]}")
```

### Indexing Solutions

You can index into batched solutions:

```python
# Get solution for problem 5
sol_5 = solution[5]
print(sol_5.x)  # Shape (n,)
```

---

## Solver Properties

After construction, you can inspect solver metadata:

```python
solver.n                  # Number of variables
solver.m                  # Number of constraints
solver.device             # Active device ('cpu' or 'cuda')
solver.construction_time  # Time spent in constructor (seconds)
```

---

## Implicit Differentiation

`CompiledSolver` supports implicit differentiation through `backward()`:

```python
settings = moreau.Settings(batch_size=64, enable_grad=True)
solver = moreau.CompiledSolver(n=2, m=3, ..., settings=settings)
solver.setup(P_values, A_values)
solution = solver.solve(qs, bs)

# Compute gradients of a scalar loss w.r.t. problem data
# dl_dx: (batch, n), dl_dz: (batch, m), dl_ds: (batch, m)
dl_dP, dl_dq, dl_dA, dl_db = solver.backward(dl_dx, dl_dz, dl_ds)
```

For PyTorch and JAX, gradient computation is handled automatically by the
framework's autograd system — see the [PyTorch](../api/torch) and
[JAX](../api/jax) API docs.

---

## First-Solve Auto-Tune

By default (`auto_tune=False`), when `device='auto'` or
`direct_solve_method='auto'`, Moreau uses a heuristic to pick the best
configuration without benchmarking.

When `auto_tune=True` is set in `Settings`, the **first call to `solve()`**
benchmarks available configurations and locks in the fastest for all subsequent
solves. This makes the first solve ~2-4x slower than normal.

- `auto_tune=True` + `device='auto'`: benchmarks all device/method combinations
- `auto_tune=True` + explicit device + `method='auto'`: benchmarks methods for that device only

```python
settings = moreau.Settings(batch_size=1024, auto_tune=True)
solver = moreau.CompiledSolver(n, m, ..., settings=settings)

# First solve benchmarks (emits a UserWarning)
solution = solver.solve(qs, bs)  # slower (auto-tune)

# All subsequent solves use the locked-in winner
solution = solver.solve(qs, bs)  # normal speed
```

To skip all automatic selection, set both device and method explicitly:

```python
settings = moreau.Settings(
    device='cuda',
    batch_size=1024,
    ipm_settings=moreau.IPMSettings(direct_solve_method='cudss'),
)
```

### Persisting Auto-Tune Results

After the first solve, you can inspect the `TuneResult` to hard-code the winning
configuration in production and skip the first-solve delay:

```python
# Run auto-tune once
solution = solver.solve(qs, bs)
tune_result = solver.tune_result  # Available on CompiledSolver after first solve

if tune_result:
    print(f"Best device: {tune_result.device}")
    print(f"Best method: {tune_result.method}")

    # Use these in production settings to skip auto-tune
    settings = moreau.Settings(
        device=tune_result.device,
        ipm_settings=moreau.IPMSettings(
            direct_solve_method=tune_result.method,
        ),
    )
```

See [Device Selection](device-selection.md) for more details on auto-tune behavior.

---

## Re-solving with New Data

The compiled structure supports efficient re-solving:

```python
# Solve first batch
solution1 = solver.solve(qs_batch1, bs_batch1)

# Update values and solve again
solver.setup(new_P_values, new_A_values)
solution2 = solver.solve(qs_batch2, bs_batch2)
```

---

## GPU Performance Tips

For maximum GPU throughput:

1. **Use large batch sizes**: GPU parallelism scales with batch size
   ```python
   settings = moreau.Settings(batch_size=1024, device='cuda')
   ```

2. **Pre-allocate**: Reuse the same solver object
   ```python
   for batch in data_loader:
       solver.setup(batch.P, batch.A)
       solution = solver.solve(batch.q, batch.b)
   ```

3. **Avoid host transfers**: Keep data on GPU when using PyTorch/JAX
   ```python
   from moreau.torch import Solver
   # Tensors stay on GPU throughout
   solver.setup(P_values.cuda(), A_values.cuda())
   solution = solver.solve(q.cuda(), b.cuda())
   ```

---

## Memory Considerations

Batch size affects memory usage:

$$\text{Memory} \approx \text{batch\_size} \times (n + 2m) \times 8 \text{ bytes (float64)}$$

For large problems, balance batch size against available memory:

```python
# Estimate memory for batch
n, m, batch_size = 1000, 2000, 512
memory_mb = batch_size * (n + 2*m) * 8 / 1e6
print(f"Estimated memory: {memory_mb:.1f} MB")
```
