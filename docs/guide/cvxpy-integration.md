# CVXPY Integration

Moreau can be used as a solver backend for [CVXPY](https://github.com/cvxpy/cvxpy), letting you formulate problems with CVXPY's high-level modeling language and solve them with Moreau's GPU-accelerated conic solver.

:::{admonition} Requirements
:class: tip

Moreau support requires **cvxpy >= 1.8.2** and **cvxpylayers >= 1.0.4**:

```bash
pip install 'cvxpy>=1.8.2' 'cvxpylayers>=1.0.4'
```
:::

## Quick Start

```python
import cvxpy as cp
import numpy as np

# Define problem using CVXPY's modeling language
x = cp.Variable(3)
objective = cp.Minimize(cp.sum_squares(x) + x[0])
constraints = [cp.sum(x) == 1, x >= 0]
problem = cp.Problem(objective, constraints)

# Solve with Moreau
problem.solve(solver=cp.MOREAU)

print(f"Status: {problem.status}")      # optimal
print(f"Optimal value: {problem.value}")
print(f"Optimal x: {x.value}")
```

CVXPY automatically converts your problem into the conic form that Moreau expects. You don't need to manually construct `P`, `A`, `b`, or specify cones.

---

## Supported Problem Types

Moreau supports any CVXPY problem that reduces to a conic program with the following cones:

| CVXPY Construct | Moreau Cone | Notes |
|-----------------|-------------|-------|
| `x == a` | Zero cone | Equality constraints |
| `x >= a`, `x <= a` | Nonnegative cone | Inequality constraints |
| `cp.SOC(t, x)` | Second-order cone | Arbitrary dimension >= 2 |
| `cp.norm(x) <= t` | Second-order cone | Via automatic reformulation |
| `cp.quad_form(x, P)` | Quadratic objective | Direct P matrix support |

:::{note}
Moreau supports arbitrary-dimension second-order cones natively. As of cvxpy 1.8.2, CVXPY passes variable-length SOC dimensions directly to Moreau without reformulation.
:::

---

## Linear Programs

```python
n = 5
c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
x = cp.Variable(n)

problem = cp.Problem(
    cp.Minimize(c @ x),
    [cp.sum(x) == 1, x >= 0]
)
problem.solve(solver=cp.MOREAU)
```

---

## Quadratic Programs

```python
n = 10
np.random.seed(42)
P = np.random.randn(n, n)
P = P.T @ P + np.eye(n)  # Positive definite
q = np.random.randn(n)

x = cp.Variable(n)
problem = cp.Problem(
    cp.Minimize(0.5 * cp.quad_form(x, P) + q @ x),
    [cp.sum(x) == 1, x >= 0]
)
problem.solve(solver=cp.MOREAU)
```

---

## Second-Order Cone Programs

```python
x = cp.Variable(3)
problem = cp.Problem(
    cp.Minimize(x[0]),
    [cp.SOC(x[0], x[1:]), x[1] == 1, x[2] == 0]
)
problem.solve(solver=cp.MOREAU)
# x[0] >= ||[x[1], x[2]]|| = 1
```

Norm constraints work too — CVXPY reformulates them into SOC form:

```python
n = 10
x = cp.Variable(n)
problem = cp.Problem(
    cp.Minimize(cp.norm(x)),
    [cp.sum(x) == n]
)
problem.solve(solver=cp.MOREAU)
```

---

## Mixed Constraints

Combine equality, inequality, and cone constraints freely:

```python
n = 5
returns = np.array([0.1, 0.2, 0.15, 0.12, 0.18])

w = cp.Variable(n)
problem = cp.Problem(
    cp.Minimize(cp.sum_squares(w)),
    [
        returns @ w >= 0.15,   # Return target
        cp.sum(w) == 1,        # Budget
        w >= 0,                # Long only
        cp.norm(w) <= 2,       # Risk limit
    ]
)
problem.solve(solver=cp.MOREAU)
```

---

## Solver Settings

Pass solver options as keyword arguments to `problem.solve()`. Top-level {py:class}`~moreau.Settings` fields (`device`, `max_iter`, `verbose`, `time_limit`) are passed directly; IPM-specific settings go in a nested `ipm_settings` dict:

```python
# Top-level settings (device, iteration limit, verbosity)
problem.solve(
    solver=cp.MOREAU,
    verbose=True,
    max_iter=500,
    device="cpu",       # 'auto' (default), 'cpu', or 'cuda'
    time_limit=10.0,    # seconds
)

# IPM tolerances and KKT solver method (via ipm_settings dict)
problem.solve(
    solver=cp.MOREAU,
    ipm_settings={
        "tol_gap_abs": 1e-6,
        "tol_feas": 1e-6,
        "tol_gap_rel": 1e-6,
        "direct_solve_method": "qdldl",  # 'auto', 'qdldl', 'faer', 'cudss', 'riccati', 'woodbury'
    },
)
```

---

## Verifying Against Other Solvers

CVXPY makes it easy to verify Moreau's solutions against other solvers:

```python
# Solve with CLARABEL
problem.solve(solver=cp.CLARABEL)
clarabel_obj = problem.value

# Solve with Moreau
problem.solve(solver=cp.MOREAU)
moreau_obj = problem.value

np.testing.assert_allclose(moreau_obj, clarabel_obj, rtol=1e-4)
```

---

## When to Use CVXPY vs Native API

| Use Case | Recommendation |
|----------|----------------|
| Prototyping | CVXPY — easier problem formulation |
| Production with fixed structure | Native API — better performance |
| Complex cone constraints | CVXPY — automatic reformulation |
| Batched solving | Native `CompiledSolver` — optimized for throughput |
| Differentiable optimization | cvxpylayers or native PyTorch/JAX API |

CVXPY adds some overhead from problem parsing and parameter evaluation compared to using Moreau directly. For production workloads — especially batched problems or training loops — the [native Moreau API](basic-usage.md) provides better performance.

:::{tip}
A common workflow is to prototype with CVXPY, then port to the native API for production. See [cvxpylayers Integration](cvxpylayers-integration.md) for differentiable optimization with CVXPY.
:::
