# Portfolio Optimization

This example demonstrates using Moreau for Markowitz portfolio optimization.

## Problem Formulation

The classic mean-variance portfolio optimization:

$$
\begin{aligned}
\text{maximize} \quad & \mu^\top w - \tfrac{\gamma}{2}\, w^\top \Sigma\, w \\
\text{subject to} \quad & \mathbf{1}^\top w = 1 \\
& w \ge 0
\end{aligned}
$$

Where:
- $w$ is the portfolio weights
- $\mu$ is expected returns
- $\Sigma$ is the covariance matrix
- $\gamma$ is the risk aversion parameter

Converting to standard form:

$$
\begin{aligned}
\text{minimize} \quad & \tfrac{\gamma}{2}\, w^\top \Sigma\, w - \mu^\top w \\
\text{subject to} \quad & \mathbf{1}^\top w = 1 \\
& w \ge 0
\end{aligned}
$$

## Implementation

```python
import moreau
import numpy as np
from scipy import sparse

def portfolio_optimize(mu, Sigma, gamma=1.0):
    """
    Solve mean-variance portfolio optimization.

    Args:
        mu: Expected returns, shape (n,)
        Sigma: Covariance matrix, shape (n, n)
        gamma: Risk aversion parameter

    Returns:
        Optimal weights, shape (n,)
    """
    n = len(mu)

    # Objective: (gamma/2) w'Sigma w - mu'w
    P = sparse.csr_array(gamma * Sigma)
    q = -mu

    # Constraints: sum(w) = 1, w >= 0
    A = sparse.vstack([
        sparse.csr_array(np.ones((1, n))),  # sum = 1
        sparse.eye(n, format='csr'),         # w >= 0
    ])
    b = np.concatenate([[1.0], np.zeros(n)])

    # Cone structure
    cones = moreau.Cones(
        num_zero_cones=1,      # equality: sum = 1
        num_nonneg_cones=n,    # inequalities: w >= 0
    )

    # Solve
    solver = moreau.Solver(P, q, A, b, cones=cones)
    solution = solver.solve()

    # Tip: On CUDA, use direct_solve_method='woodbury' for portfolio problems
    # with diagonal P + low-rank constraint structure for faster solves:
    # settings = moreau.Settings(device='cuda',
    #     ipm_settings=moreau.IPMSettings(direct_solve_method='woodbury'))

    if solver.info.status != moreau.SolverStatus.Solved:
        raise RuntimeError(f"Solver failed: {solver.info.status}")

    return solution.x

# Example usage
np.random.seed(42)
n = 10  # 10 assets

# Generate random market data
mu = np.random.randn(n) * 0.1  # Expected returns
A_factor = np.random.randn(n, 3)  # Factor loadings
Sigma = A_factor @ A_factor.T + 0.1 * np.eye(n)  # Covariance

# Optimize
weights = portfolio_optimize(mu, Sigma, gamma=2.0)
print(f"Optimal weights: {weights}")
print(f"Sum of weights: {weights.sum():.6f}")
print(f"Expected return: {mu @ weights:.4f}")
print(f"Portfolio variance: {weights @ Sigma @ weights:.4f}")
```

## Batched Portfolio Optimization

For multiple scenarios or hyperparameter sweeps:

```python
import moreau
import numpy as np
from scipy import sparse

def batch_portfolio_optimize(mu, Sigma, gammas):
    """
    Solve portfolio optimization for multiple risk aversion values.

    Args:
        mu: Expected returns, shape (n,)
        Sigma: Covariance matrix, shape (n, n)
        gammas: Array of risk aversion parameters, shape (batch,)

    Returns:
        Optimal weights, shape (batch, n)
    """
    n = len(mu)
    batch_size = len(gammas)

    # Build CSR structure
    P_csr = sparse.csr_array(Sigma)
    A_eq = sparse.csr_array(np.ones((1, n)))
    A_ineq = sparse.eye(n, format='csr')
    A_csr = sparse.vstack([A_eq, A_ineq])

    # Extract CSR components
    P_ro = P_csr.indptr.tolist()
    P_ci = P_csr.indices.tolist()
    A_ro = A_csr.indptr.tolist()
    A_ci = A_csr.indices.tolist()

    # Setup solver
    cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=n)
    settings = moreau.Settings(batch_size=batch_size)

    solver = moreau.CompiledSolver(
        n=n, m=1+n,
        P_row_offsets=P_ro, P_col_indices=P_ci,
        A_row_offsets=A_ro, A_col_indices=A_ci,
        cones=cones, settings=settings,
    )

    # P_values differ per gamma
    P_values_batch = np.array([gamma * P_csr.data for gamma in gammas])
    A_values = A_csr.data  # Shared

    solver.setup(P_values_batch, A_values)

    # q and b same for all (but batched format)
    qs = np.tile(-mu, (batch_size, 1))
    bs = np.tile(np.concatenate([[1.0], np.zeros(n)]), (batch_size, 1))

    solution = solver.solve(qs, bs)
    return solution.x

# Example
gammas = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
weights_batch = batch_portfolio_optimize(mu, Sigma, gammas)

for i, gamma in enumerate(gammas):
    w = weights_batch[i]
    ret = mu @ w
    var = w @ Sigma @ w
    print(f"gamma={gamma:.1f}: return={ret:.4f}, variance={var:.4f}")
```

## PyTorch: Learning Expected Returns

Use gradient-based learning to fit expected returns:

```python
import torch
from moreau.torch import Solver
import moreau

# Problem setup
n = 10
Sigma = torch.eye(n, dtype=torch.float64) * 0.1

# True optimal weights (target)
true_mu = torch.randn(n, dtype=torch.float64) * 0.1
true_weights = torch.softmax(true_mu / 0.1, dim=0)

# Setup solver structure
P_csr = torch.eye(n, dtype=torch.float64)  # Will scale by gamma
A_eq = torch.ones(1, n, dtype=torch.float64)
A_ineq = torch.eye(n, dtype=torch.float64)

# CSR indices
P_ro = torch.tensor([i for i in range(n+1)])
P_ci = torch.tensor([i for i in range(n)])
A_ro = torch.tensor([0] + [n + i for i in range(n + 1)])
A_ci = torch.tensor([i for i in range(n)] + [i for i in range(n)])

cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=n)
solver = Solver(n, 1+n, P_ro, P_ci, A_ro, A_ci, cones)

# Learnable expected returns
mu_learned = torch.randn(n, dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam([mu_learned], lr=0.1)

gamma = 2.0
P_values = gamma * torch.ones(n, dtype=torch.float64)
A_values = torch.ones(2*n, dtype=torch.float64)
b = torch.cat([torch.ones(1), torch.zeros(n)]).double()

# Training loop
for epoch in range(100):
    optimizer.zero_grad()

    q = -mu_learned
    solution = solver.solve(P_values, A_values, q, b)

    # Loss: match target weights
    loss = torch.sum((solution.x - true_weights) ** 2)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {loss.item():.6f}")
```
