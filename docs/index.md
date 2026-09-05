# Moreau

```{raw} html
<p align="center" style="font-size: 1.4rem; color: var(--color-foreground-secondary); margin-bottom: 2rem;">
  <strong>GPU-native, batched, differentiable conic optimization</strong>
</p>
```

Moreau solves quadratic cone programs with a Rust CPU backend and a C++/CUDA GPU backend behind one Python API. Solve a single problem, a batch of thousands on the GPU, or differentiate through the solution for end-to-end learning — all with the same code.

---

## Frameworks

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} **NumPy**
:link: api/core
:link-type: doc
:class-card: sd-card-numpy

Just solve. Standard NumPy/SciPy interface for straightforward optimization without framework overhead.
:::

:::{grid-item-card} **PyTorch**
:link: api/torch
:link-type: doc
:class-card: sd-card-pytorch

Full autograd support with differentiable optimization. Embed optimization in your neural networks.
:::

:::{grid-item-card} **JAX**
:link: api/jax
:link-type: doc
:class-card: sd-card-jax

Compatible with `jax.grad`, `jax.vmap`, and `jax.jit`. Functional API for composable transformations.
:::

:::{grid-item-card} **CVXPY**
:link: examples/cvxpy
:link-type: doc
:class-card: sd-card-cvxpy

Use Moreau as a CVXPY solver backend, or combine with cvxpylayers for differentiable convex optimization.
:::

:::{grid-item-card} **Julia / JuMP**
:link: guide/jump-integration
:link-type: doc
:class-card: sd-card-julia

Full MathOptInterface wrapper for JuMP, plus a low-level Julia API with batching, CUDA, and ChainRules differentiation.
:::

::::

---

## Get Started in 30 Seconds

::::{grid} 1 1 2 2
:gutter: 4

:::{grid-item}
:columns: 12 12 7 7

```python
import moreau
import numpy as np
from scipy import sparse

# Define a simple QP
P = sparse.diags([1.0, 1.0], format='csr')
q = np.array([2.0, 1.0])
A = sparse.csr_array([
    [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]
])
b = np.array([1.0, 0.7, 0.7])
cones = moreau.Cones(
    num_zero_cones=1, num_nonneg_cones=2
)

# Solve
solver = moreau.Solver(P, q, A, b, cones=cones)
solution = solver.solve()
print(f"x = {solution.x}")
print(f"status = {solver.info.status}")
```
:::

:::{grid-item}
:columns: 12 12 5 5

**What's happening?**

1. Define a convex conic problem
2. Create a solver with problem data
3. Solve and get the optimal solution
4. Access solver metadata via `solver.info`

```{button-ref} installation
:color: primary
:expand:

Install
```

```{button-ref} quickstart
:color: secondary
:outline:
:expand:

Quickstart Guide
```

:::

::::

---

## Why Moreau?

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} GPU + CPU, one API
:class-card: sd-card-feature

A Rust CPU backend and a C++/CUDA GPU backend behind a single Python interface. Native CUDA keeps everything on-device — zero host-device transfers while solving.
:::

:::{grid-item-card} Batched on-device
:class-card: sd-card-feature

Solve thousands of problems that share a sparsity pattern in one `CompiledSolver` call, entirely on the GPU.
:::

:::{grid-item-card} Differentiable
:class-card: sd-card-feature

Native forward and backward (adjoint) passes for both LPs and QPs. Backpropagate through optimization layers in PyTorch and JAX.
:::

:::{grid-item-card} Many cones
:class-card: sd-card-feature

Zero, nonnegative, second-order, exponential, power, generalized-power, and PSD cones — plus direct conic constraints that skip the slack row.
:::

::::

---

## Used For

::::{grid} 2 2 4 4
:gutter: 2

:::{grid-item-card} Control
:text-align: center
:class-card: sd-card-usecase

MPC, trajectory optimization
:::

:::{grid-item-card} Finance
:text-align: center
:class-card: sd-card-usecase

Portfolio optimization
:::

:::{grid-item-card} ML
:text-align: center
:class-card: sd-card-usecase

Constrained learning
:::

:::{grid-item-card} Robotics
:text-align: center
:class-card: sd-card-usecase

Motion planning
:::

::::

```{button-ref} examples/index
:color: primary
:outline:
:expand:

Browse Examples
```

---

## Problem Formulation

Moreau solves convex conic programs of the form:

$$
\begin{aligned}
\text{minimize} \quad & \tfrac{1}{2} x^\top P x + q^\top x \\
\text{subject to} \quad & Ax + s = b \\
& x \in \mathcal{K}_1, \; s \in \mathcal{K}_2
\end{aligned}
$$

Where $\mathcal{K}_2$ constrains the slack $s$ and $\mathcal{K}_1$ constrains $x$
directly (direct conic constraints). The slack cone $\mathcal{K}_2$ is a product of:
- **Zero cone**: Equality constraints ($s = 0$)
- **Nonnegative cone**: Inequality constraints ($s \ge 0$)
- **Second-order cone**: Norm constraints ($\|s_{1:}\|_2 \le s_0$, arbitrary dimension $\ge 2$)
- **Exponential cone**: Log/exp constraints (dim 3)
- **Power cone**: Power function constraints (dim 3)
- **Generalized power cone**: Geometric mean constraints ($\prod p_i^{\alpha_i} \ge \|w\|_2$, variable dimension)
- **PSD cone**: Positive semidefinite matrix constraints ($\text{mat}(s) \succeq 0$) — see [PSD Cones guide](guide/psd-cones.md)

The cone $\mathcal{K}_1$ used in direct conic constraints admits the same building blocks, except it
takes the dual of the zero cone (the free cone) in place of the zero cone.

## Citing Moreau

If you use Moreau in your research, please cite it as:

```bibtex
@software{moreau2026,
  author  = {Barratt, Shane and Nobel, Parth and Diamond, Steven},
  title   = {Moreau: {GPU}-Native Differentiable Optimization},
  year    = {2026},
  url     = {https://moreau.so},
}
```

## References

Moreau's interior-point algorithm derives from Clarabel and its GPU variant
CuClarabel; the backward (adjoint) pass derives from diffcp and diffqcp; the
CPU active-set solver derives from DAQP.

1. P. J. Goulart and Y. Chen. *Clarabel: An interior-point solver for conic programs with quadratic objectives.* arXiv:2405.12762, 2024. <https://arxiv.org/abs/2405.12762>
2. Y. Chen, D. Tse, P. Nobel, P. Goulart, and S. Boyd. *CuClarabel: GPU Acceleration for a Conic Optimization Solver.* ACM Transactions on Mathematical Software (to appear). <https://doi.org/10.1145/3815420>
3. A. Agrawal, S. Barratt, S. Boyd, E. Busseti, and W. Moursi. *Differentiating through a Cone Program.* Journal of Applied and Numerical Optimization, 1(2):107–115, 2019.
4. Q. Healey, P. Nobel, and S. Boyd. *Differentiating Through a Quadratic Cone Program.* Optimization Letters (to appear). arXiv:2508.17522, 2025. <https://arxiv.org/abs/2508.17522>
5. D. Arnström, A. Bemporad, and D. Axehill. *A Dual Active-Set Solver for Embedded Quadratic Programming Using Recursive LDL^T Updates.* IEEE Transactions on Automatic Control, 67(8):4362–4369, 2022. <https://arxiv.org/abs/2103.16236>

```{toctree}
:maxdepth: 2
:hidden:

installation
quickstart
guide/index
api/index
examples/index
```
