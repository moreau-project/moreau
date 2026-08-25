# cvxpylayers Integration

[cvxpylayers](https://github.com/cvxpy/cvxpylayers) turns CVXPY problems into differentiable layers for PyTorch and JAX. Combined with Moreau as the solver backend, you get GPU-accelerated differentiable convex optimization with a high-level problem formulation.

:::{admonition} Requirements
:class: tip

Moreau support requires **cvxpy >= 1.8.2** and **cvxpylayers >= 1.0.4**:

```bash
pip install 'cvxpy>=1.8.2' 'cvxpylayers>=1.0.4'
```
:::

## How It Works

1. Define a **parametric** CVXPY problem using `cp.Parameter`
2. Wrap it in a `CvxpyLayer` with `solver="MOREAU"`
3. Call the layer with framework tensors — gradients flow through the solve

```
cp.Parameter values (PyTorch/JAX tensors)
    │
    ▼
┌─────────────┐
│ CvxpyLayer  │  Evaluates parameters → Solves with Moreau → Differentiates
└─────────────┘
    │
    ▼
cp.Variable solutions (with gradients)
```

---

## PyTorch

### Basic Usage

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# 1. Define parametric problem
n = 5
x = cp.Variable(n)
q = cp.Parameter(n)

objective = cp.Minimize(0.5 * cp.sum_squares(x) + q @ x)
constraints = [x >= 0, cp.sum(x) == 1]
problem = cp.Problem(objective, constraints)

# 2. Create differentiable layer
layer = CvxpyLayer(problem, parameters=[q], variables=[x], solver="MOREAU")

# 3. Solve with PyTorch tensors
q_val = torch.randn(n, dtype=torch.float64, requires_grad=True)
x_opt, = layer(q_val)

# 4. Backpropagate
loss = x_opt.sum()
loss.backward()
print(f"Gradient dL/dq: {q_val.grad}")
```

### Training Loop

Use a `CvxpyLayer` inside a standard PyTorch training loop to learn parameters of an optimization problem:

```python
import torch.nn as nn

# Parametric portfolio problem
n = 10
w = cp.Variable(n)
mu = cp.Parameter(n)       # Expected returns (learnable)
gamma = 1.0

Sigma = torch.eye(n, dtype=torch.float64) * 0.1

objective = cp.Minimize(-mu @ w + gamma * cp.quad_form(w, Sigma.numpy()))
constraints = [w >= 0, cp.sum(w) == 1]
problem = cp.Problem(objective, constraints)

portfolio_layer = CvxpyLayer(
    problem, parameters=[mu], variables=[w], solver="MOREAU"
)

# Learn expected returns to match target weights
target_weights = torch.softmax(torch.randn(n, dtype=torch.float64), dim=0)
mu_learned = torch.randn(n, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([mu_learned], lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    weights, = portfolio_layer(mu_learned)
    loss = torch.sum((weights - target_weights) ** 2)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```

### Multiple Parameters

You can differentiate through multiple `cp.Parameter` objects:

```python
n = 5
x = cp.Variable(n)
q = cp.Parameter(n)
b = cp.Parameter(1)

objective = cp.Minimize(0.5 * cp.sum_squares(x) + q @ x)
constraints = [x >= 0, cp.sum(x) <= b]
problem = cp.Problem(objective, constraints)

layer = CvxpyLayer(problem, parameters=[q, b], variables=[x], solver="MOREAU")

q_val = torch.randn(n, dtype=torch.float64, requires_grad=True)
b_val = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
x_opt, = layer(q_val, b_val)

x_opt.sum().backward()
print(f"dL/dq: {q_val.grad}")
print(f"dL/db: {b_val.grad}")
```

### Batched Forward Pass

cvxpylayers supports batched parameters — add a leading batch dimension:

```python
batch_size = 32
q_batch = torch.randn(batch_size, n, dtype=torch.float64, requires_grad=True)

# Each problem in the batch gets a different q
x_batch, = layer(q_batch)
print(x_batch.shape)  # (32, 5)

loss = x_batch.sum()
loss.backward()
print(q_batch.grad.shape)  # (32, 5)
```

---

## JAX

### Basic Usage

```python
import cvxpy as cp
import jax
import jax.numpy as jnp
from cvxpylayers.jax import CvxpyLayer

jax.config.update("jax_enable_x64", True)

# 1. Define parametric problem
n = 5
x = cp.Variable(n)
q = cp.Parameter(n)

objective = cp.Minimize(0.5 * cp.sum_squares(x) + q @ x)
constraints = [x >= 0, cp.sum(x) == 1]
problem = cp.Problem(objective, constraints)

# 2. Create differentiable layer
layer = CvxpyLayer(problem, parameters=[q], variables=[x], solver="MOREAU")

# 3. Solve and differentiate
q_val = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

def solve(q):
    x_opt, = layer(q)
    return x_opt.sum()

grad_fn = jax.grad(solve)
print(f"Gradient: {grad_fn(q_val)}")
```

### Batching with `jax.vmap`

```python
q_batch = jax.random.normal(jax.random.PRNGKey(0), (64, n))

# vmap over the parameter dimension
batched_solve = jax.vmap(lambda q: layer(q)[0])
x_batch = batched_solve(q_batch)
print(x_batch.shape)  # (64, 5)
```

### Combined `grad` and `vmap`

```python
def loss_fn(q):
    x_opt, = layer(q)
    return jnp.sum(x_opt ** 2)

# Gradient of each problem in the batch
batched_grad = jax.vmap(jax.grad(loss_fn))
grads = batched_grad(q_batch)
print(grads.shape)  # (64, 5)
```

---

## Solver Arguments

Pass solver-specific options through `solver_args` — either at construction time (applied to every call) or per-call (overrides defaults). Top-level {py:class}`~moreau.Settings` fields are passed directly; IPM-specific settings go in a nested `ipm_settings` dict:

```python
# At construction time (defaults for all forward calls)
layer = CvxpyLayer(
    problem, parameters=[q], variables=[x],
    solver="MOREAU",
    solver_args={
        "device": "cuda",
        "ipm_settings": {"tol_gap_abs": 1e-6, "tol_feas": 1e-6},
    },
)

# Per-call overrides (PyTorch)
x_opt, = layer(q_val, solver_args={"max_iter": 500, "verbose": True})

# Per-call overrides (JAX)
x_opt, = layer(q_val, solver_args={"max_iter": 500})

# IPM tolerances and KKT method via ipm_settings dict
x_opt, = layer(q_val, solver_args={
    "ipm_settings": {"tol_gap_abs": 1e-4, "direct_solve_method": "qdldl"},
})
```

---

## cvxpylayers vs Native Moreau API

Both cvxpylayers and Moreau's native PyTorch/JAX APIs support differentiable optimization. Choose based on your needs:

| | cvxpylayers + Moreau | Native `moreau.torch` / `moreau.jax` |
|---|---|---|
| **Problem formulation** | High-level CVXPY syntax | Manual CSR matrices and cones |
| **Reformulation** | Automatic (CVXPY handles it) | Manual |
| **Batching** | Supported (batch dim on parameters) | `CompiledSolver` with `batch_size` or `jax.vmap` |
| **GPU acceleration** | Yes (Moreau backend) | Yes (direct) |
| **Overhead** | DPP parameter evaluation on each forward pass | Minimal — structure compiled once |
| **Best for** | Prototyping, complex formulations | Production, training loops, maximum throughput |

:::{tip}
**Prototyping workflow**: Start with cvxpylayers for quick iteration, then port to the native API when you need maximum performance. The conic form that CVXPY generates can guide your manual formulation.
:::

---

## Troubleshooting

### `solver="MOREAU"` not recognized

Ensure you have the required versions:

```bash
pip install 'cvxpy>=1.8.2' 'cvxpylayers>=1.0.4'
```

Verify:
```python
import cvxpy as cp
assert hasattr(cp, 'MOREAU'), "MOREAU solver not available"
```

### `float32` errors

Moreau requires double precision. Ensure your tensors are `float64`:

```python
# PyTorch
q_val = torch.randn(n, dtype=torch.float64, requires_grad=True)

# JAX
jax.config.update("jax_enable_x64", True)
```

### Slow performance in training loops

cvxpylayers uses DPP to canonicalize the problem once at construction, but still evaluates parameter mappings on each forward pass. For training loops with fixed problem structure, the native [PyTorch](pytorch-integration.md) or [JAX](jax-integration.md) API avoids this overhead entirely.
