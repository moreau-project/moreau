# CVXPY Examples

Practical examples using Moreau as a CVXPY solver backend and with cvxpylayers for differentiable optimization.

:::{admonition} Guides
:class: seealso

For comprehensive documentation, see the [CVXPY Integration Guide](../guide/cvxpy-integration.md) and [cvxpylayers Integration Guide](../guide/cvxpylayers-integration.md).
:::

:::{admonition} Requirements
:class: tip

Moreau support requires **cvxpy >= 1.8.2** and **cvxpylayers >= 1.0.4**:

```bash
pip install 'cvxpy>=1.8.2' 'cvxpylayers>=1.0.4'
```
:::

## Using Moreau with CVXPY

Register Moreau as a CVXPY solver:

```python
import cvxpy as cp
import numpy as np

# Define problem in CVXPY
n = 10
x = cp.Variable(n)
P = np.eye(n)
q = np.random.randn(n)

objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x)
constraints = [x >= 0, cp.sum(x) == 1]
problem = cp.Problem(objective, constraints)

# Solve with Moreau
problem.solve(solver=cp.MOREAU)

print(f"Optimal value: {problem.value}")
print(f"Optimal x: {x.value}")
```

---

## cvxpylayers with PyTorch

Use cvxpylayers to create differentiable optimization layers:

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# Define parametric problem
n = 5
x = cp.Variable(n)
q = cp.Parameter(n)  # Learnable parameter

objective = cp.Minimize(0.5 * cp.sum_squares(x) + q @ x)
constraints = [x >= 0, cp.sum(x) == 1]
problem = cp.Problem(objective, constraints)

# Create differentiable layer
layer = CvxpyLayer(problem, parameters=[q], variables=[x], solver="MOREAU")

# Use in PyTorch
q_val = torch.randn(n, dtype=torch.float64, requires_grad=True)
x_opt, = layer(q_val)

# Backpropagate
loss = x_opt.sum()
loss.backward()
print(f"Gradient dL/dq: {q_val.grad}")
```

---

## cvxpylayers with JAX

Similarly for JAX:

```python
import cvxpy as cp
import jax
import jax.numpy as jnp
from cvxpylayers.jax import CvxpyLayer

# Define parametric problem
n = 5
x = cp.Variable(n)
q = cp.Parameter(n)

objective = cp.Minimize(0.5 * cp.sum_squares(x) + q @ x)
constraints = [x >= 0, cp.sum(x) == 1]
problem = cp.Problem(objective, constraints)

# Create differentiable layer
layer = CvxpyLayer(problem, parameters=[q], variables=[x], solver="MOREAU")

# Use with JAX
q_val = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

def solve(q):
    x_opt, = layer(q)
    return x_opt.sum()

# Compute gradient
grad_fn = jax.grad(solve)
print(f"Gradient: {grad_fn(q_val)}")
```

---

## Portfolio Optimization with cvxpylayers

A practical example: learning expected returns to match target portfolio weights.

```python
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

# Portfolio problem
n = 10  # Number of assets
w = cp.Variable(n)
mu = cp.Parameter(n)  # Expected returns (learnable)
gamma = 1.0  # Risk aversion

# Covariance matrix
Sigma = torch.eye(n, dtype=torch.float64) * 0.1

objective = cp.Minimize(-mu @ w + gamma * cp.quad_form(w, Sigma.numpy()))
constraints = [w >= 0, cp.sum(w) == 1]
problem = cp.Problem(objective, constraints)

# Create layer
portfolio_layer = CvxpyLayer(
    problem,
    parameters=[mu],
    variables=[w],
    solver="MOREAU"
)

# Target weights
target_weights = torch.softmax(torch.randn(n, dtype=torch.float64), dim=0)

# Learnable expected returns
mu_learned = torch.randn(n, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([mu_learned], lr=0.1)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()

    weights, = portfolio_layer(mu_learned)
    loss = torch.sum((weights - target_weights) ** 2)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")

print(f"Learned mu: {mu_learned.detach()}")
```

---

## When to Use CVXPY vs Native API

| Use Case | Recommendation |
|----------|----------------|
| Prototyping | CVXPY - easier problem formulation |
| Production with fixed structure | Native API - better performance |
| Differentiable optimization | cvxpylayers or native PyTorch/JAX |
| Batched solving | Native `CompiledSolver` - optimized for throughput |
| Complex cone constraints | CVXPY - automatic reformulation |

The native Moreau API provides better performance for production workloads, especially batched problems. CVXPY is excellent for prototyping and problems with complex structure that benefit from automatic reformulation.
