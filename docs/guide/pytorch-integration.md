# PyTorch Integration

Moreau provides a seamless integration with PyTorch, allowing you to use conic optimization as a differentiable layer in your neural networks. It supports full autograd and automatic device selection between CPU and CUDA.

## Quick Start

```python
import torch
from moreau.torch import Solver
import moreau

# 1. Define problem structure
cones = moreau.Cones(num_nonneg_cones=2)
solver = Solver(
    n=2, m=2,
    P_row_offsets=torch.tensor([0, 1, 2]),
    P_col_indices=torch.tensor([0, 1]),
    A_row_offsets=torch.tensor([0, 1, 2]),
    A_col_indices=torch.tensor([0, 1]),
    cones=cones,
)

# 2. Solve (stateless — all data passed per call)
P_values = torch.tensor([1.0, 1.0], dtype=torch.float64)
A_values = torch.tensor([1.0, 1.0], dtype=torch.float64)
q = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
b = torch.tensor([0.5, 0.5], dtype=torch.float64)

solution = solver.solve(P_values, A_values, q, b)

# 3. Backpropagate
loss = solution.x.sum()
loss.backward()

print(q.grad)  # dL/dq computed via implicit differentiation
```

---

## Differentiable Layers

You can wrap Moreau's `Solver` in a `torch.nn.Module` to create custom differentiable layers.

```python
import torch.nn as nn

class ConstrainedLayer(nn.Module):
    def __init__(self, n, m, P_ro, P_ci, A_ro, A_ci, cones):
        super().__init__()
        self.solver = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones)
        
        # Linear parameters that produce q
        self.linear = nn.Linear(10, n, dtype=torch.float64)
        
        # P and A are constant for this example
        self.register_buffer('P_vals', torch.ones(len(P_ci), dtype=torch.float64))
        self.register_buffer('A_vals', torch.ones(len(A_ci), dtype=torch.float64))
        self.register_buffer('b_vals', torch.ones(m, dtype=torch.float64))

    def forward(self, x):
        q = self.linear(x)
        # solve() is stateless — P/A setup is cached automatically
        solution = self.solver.solve(self.P_vals, self.A_vals, q, self.b_vals)
        return solution.x

# Usage
layer = ConstrainedLayer(...)
output = layer(input_data)
loss = (output - target).pow(2).sum()
loss.backward()
```

---

## Implicit Differentiation

Unlike some other libraries that differentiate through solver iterations (unrolling), Moreau uses **implicit differentiation**. This technique computes gradients based on the optimality (KKT) conditions of the problem.

- **Efficiency**: Requires constant memory regardless of the number of solver iterations.
- **Accuracy**: Differentiates through the exact solution, avoiding "gradient vanish/explosion" issues common in unrolled solvers.
- **Support**: Moreau supports differentiating with respect to ALL problem parameters ($P, A, q, b$).

---

## Batching and Parallelism

Moreau's PyTorch solver is optimized for high-throughput batching.

- **Fixed Batch Size**: Specify `batch_size` in `Settings` at construction to pre-allocate memory and structures for a specific batch size. This is recommended for production training loops.
- **Dynamic Batching**: If `batch_size` is not specified, Moreau handles dynamic batching automatically, but this may incur overhead for re-allocation.

```python
settings = moreau.Settings(batch_size=256, device='cuda')
solver = Solver(..., settings=settings)

# q has shape (256, n)
solution = solver.solve(P_values, A_values, q_batch, b_batch)
print(solution.x.shape)  # (256, n)
```

---

## GPU and Device Transfers

To minimize latency, keep your tensors on the same device as the solver.

```python
# Create solver on GPU
settings = moreau.Settings(device='cuda')
solver = Solver(..., settings=settings)

# Pass GPU tensors
P_gpu = torch.ones(nnzP, device='cuda', dtype=torch.float64)
q_gpu = torch.randn(batch, n, device='cuda', dtype=torch.float64, requires_grad=True)

solution = solver.solve(P_gpu, A_gpu, q_gpu, b_gpu)
# solution.x is already on GPU
```

Moreau handles moving inputs to the solver's device automatically if they are on a different device, but this will incur host-to-device transfer overhead.

---

## Warm Starting in PyTorch

Warm starting is supported and can significantly speed up convergence during training or MPC.

```python
# 1. Initial solve
solution = solver.solve(P_values, A_values, q, b)

# 2. Get warm start point
ws = solution.to_warm_start()  # Detaches and copies from GPU if needed

# 3. Solve next iteration with warm start
solution2 = solver.solve(P_values, A_values, q_new, b_new, warm_start=ws)
```

:::{note}
Gradients do **not** flow through the warm start values (`x, z, s`). They only flow through the problem data ($P, A, q, b$).
:::

---

## Best Practices

1.  **Use `float64`**: PyTorch defaults to `float32`, but Moreau requires `float64`. Ensure all tensors passed to the solver are `dtype=torch.float64`.
2.  **Thread Safety**: Moreau's `Solver` instances are **not thread-safe**. If you are using multi-threaded data loaders or training loops, ensure each thread has its own solver instance.
3.  **Stateless API**: `solve(P, A, q, b)` automatically caches the P/A setup — if P and A haven't changed since the last call, the expensive setup step is skipped. This makes it safe for chained solves in autograd graphs (e.g. MPC rollouts).
4.  **Auto-Tune**: By default, Moreau uses a heuristic to select the KKT solver without benchmarking. Set `auto_tune=True` in `Settings` to benchmark different KKT solvers on the first solve and lock in the fastest. Alternatively, set `direct_solve_method` explicitly in `IPMSettings` to skip selection entirely.
5.  **`setup_grad()`**: For repeated backward calls, call `solver.setup_grad(batch_size)` to pre-allocate gradient buffers and speed up the backward pass.
6.  **Memory Management**: For very large batches on GPU, monitor your VRAM usage. Moreau pre-allocates structures for the KKT system, which can be memory-intensive.
