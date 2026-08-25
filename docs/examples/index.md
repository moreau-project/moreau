# Examples

Practical examples demonstrating Moreau for common optimization tasks.

```{toctree}
:maxdepth: 1

cvxpy
portfolio
mpc
```

## Quick Examples

### Simple QP

Minimize $\tfrac{1}{2}\|x\|^2 + c^\top x$ subject to $Ax \le b$:

```python
import moreau
import numpy as np
from scipy import sparse

# minimize (1/2)||x||^2 + c'x
# subject to Ax <= b

n, m = 10, 5
P = sparse.eye(n, format='csr')
q = np.random.randn(n)
A = sparse.random(m, n, density=0.3, format='csr')
b = np.ones(m)

cones = moreau.Cones(num_nonneg_cones=m)
solver = moreau.Solver(P, q, A, b, cones=cones)
solution = solver.solve()
print(f"Status: {solver.info.status}")
```

### Least Squares with Constraints

Minimize $\|Cx - d\|^2$ subject to $x \ge 0$:

```python
import moreau
import numpy as np
from scipy import sparse

# minimize ||Cx - d||^2
# subject to x >= 0

n, p = 20, 50
C = np.random.randn(p, n)
d = np.random.randn(p)

# Convert to QP: minimize x'(C'C)x - 2(C'd)'x
P = sparse.csr_array(C.T @ C)
P = (P + P.T) / 2  # Ensure symmetric
q = -2 * C.T @ d
A = sparse.eye(n, format='csr')  # x >= 0
b = np.zeros(n)

cones = moreau.Cones(num_nonneg_cones=n)
solver = moreau.Solver(P, q, A, b, cones=cones)
solution = solver.solve()
```

### PyTorch Training Loop

```python
import torch
from moreau.torch import Solver
import moreau

# Setup solver with CSR structure
cones = moreau.Cones(num_nonneg_cones=2)
settings = moreau.Settings(device='cuda', batch_size=32)
solver = Solver(
    n=2, m=2,
    P_row_offsets=torch.tensor([0, 1, 2]),
    P_col_indices=torch.tensor([0, 1]),
    A_row_offsets=torch.tensor([0, 1, 2]),
    A_col_indices=torch.tensor([0, 1]),
    cones=cones,
    settings=settings,
)

# Matrix values and target
P_values = torch.tensor([1.0, 1.0], dtype=torch.float64)
A_values = torch.tensor([1.0, 1.0], dtype=torch.float64)
b = torch.tensor([0.5, 0.5], dtype=torch.float64)
target = torch.tensor([0.3, 0.3], dtype=torch.float64)

# Learnable parameter
q_param = torch.randn(2, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([q_param], lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()

    solution = solver.solve(P_values, A_values, q_param, b)

    loss = (solution.x - target).pow(2).mean()
    loss.backward()
    optimizer.step()
```

### JAX with vmap

```python
import jax
import jax.numpy as jnp
from moreau.jax import Solver
import moreau

# Setup solver with CSR structure
cones = moreau.Cones(num_nonneg_cones=2)
solver = Solver(
    n=2, m=2,
    P_row_offsets=jnp.array([0, 1, 2]),
    P_col_indices=jnp.array([0, 1]),
    A_row_offsets=jnp.array([0, 1, 2]),
    A_col_indices=jnp.array([0, 1]),
    cones=cones,
)

# Matrix values
P_data = jnp.array([1.0, 1.0])
A_data = jnp.array([1.0, 1.0])
solver.setup(P_data, A_data)

# Batch solve with vmap
batched_solve = jax.vmap(solver.solve)

q_batch = jnp.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
b_batch = jnp.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

solutions = batched_solve(q_batch, b_batch)
print(solutions.x.shape)  # (4, 2)
```

## Application Areas

| Domain | Example Problem |
|--------|-----------------|
| **Finance** | Portfolio optimization, risk management |
| **Control** | Model predictive control, LQR |
| **Robotics** | Motion planning, inverse kinematics |
| **ML** | Constrained learning, optimal transport |
| **Signal Processing** | Sparse reconstruction, denoising |
