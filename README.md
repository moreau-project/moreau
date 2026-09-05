# Moreau

**GPU-native, batched, differentiable conic optimization.**

[![PyPI version](https://img.shields.io/pypi/v/moreau.svg)](https://pypi.org/project/moreau/)
[![Python versions](https://img.shields.io/pypi/pyversions/moreau.svg)](https://pypi.org/project/moreau/)
[![Documentation](https://img.shields.io/badge/docs-moreau.so-blue.svg)](https://moreau.so)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Moreau solves quadratic cone programs

```
minimize     ½ xᵀP x + qᵀx
subject to   A x + s = b
             x ∈ K₁,   s ∈ K₂
```

where `s ∈ K₂` constrains the slack and `x ∈ K₁` is a **Direct Conic
Constraint**. Both `K₁` and `K₂` are products of nonnegative, second-order,
exponential, power, generalized-power, and PSD cones. The slack cone `K₂`
also admits the zero cone (equality constraints); `K₁` admits its dual,
the free cone, instead. A Rust CPU backend and a C++/CUDA GPU backend
sit behind one Python API — so you can solve a single problem, a batch of
thousands on the GPU, or differentiate through the solution for end-to-end
learning, all with the same code.

## Why Moreau

- **One API, CPU + GPU.** A Rust (CPU) and a C++/CUDA (GPU) solver behind a
  single Python interface. Pick the device or let Moreau choose.
- **Batched on-device.** Solve thousands of problems that share a sparsity
  pattern in a single `solve()` call, entirely on the GPU.
- **Differentiable.** Native forward *and* backward (adjoint) passes for both
  LPs and QPs — get gradients of the solution w.r.t. `P`, `q`, `A`, and `b`.
- **PyTorch & JAX native.** Drop into autograd (`loss.backward()`) or JAX
  (`jit` / `vmap` / `grad`); also works with CVXPY and cvxpylayers.
- **Many cones.** Zero, nonnegative, second-order, exponential, power,
  generalized-power, and PSD — plus *direct conic constraints* on `x`
  that skip the slack row.

## Installation

```bash
pip install moreau            # CPU backend
pip install moreau[cuda]      # + CUDA 12 GPU backend (requires Python 3.12+)
pip install moreau[cuda13]    # + CUDA 13 GPU backend (requires Python 3.12+)
```

Prebuilt wheels cover Linux x86_64, Linux aarch64 (Grace / Orin / Thor), and
macOS ARM64 for CPU, and Linux x86_64 / aarch64 for CUDA 12 and 13. To build
from source (Rust for CPU, C++/CUDA + nanobind for GPU), see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Quickstart

```python
import numpy as np
from scipy import sparse
import moreau

P = sparse.diags([1.0, 1.0], format="csr")
q = np.array([2.0, 1.0])
A = sparse.csr_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
b = np.array([1.0, 0.7, 0.7])
cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

solver = moreau.Solver(P, q, A, b, cones)
solution = solver.solve()

print(solution.x)          # primal solution
print(solver.info.status)  # SolverStatus.Solved
print(solver.info.obj_val) # objective value
```

> On the GPU backend, `P` must be a **full symmetric** matrix (both triangles),
> not just the upper triangle. The CPU backend accepts either.

## Differentiable optimization

Gradients flow through the solve in both PyTorch and JAX.

**PyTorch** — standard autograd:

```python
import torch
from moreau.torch import Solver
import moreau

cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
solver = Solver(n=2, m=3,
                P_row_offsets=torch.tensor([0, 1, 2]),
                P_col_indices=torch.tensor([0, 1]),
                A_row_offsets=torch.tensor([0, 2, 3, 4]),
                A_col_indices=torch.tensor([0, 1, 0, 1]),
                cones=cones)

P_values = torch.ones(2, dtype=torch.float64, requires_grad=True)
A_values = torch.ones(4, dtype=torch.float64, requires_grad=True)
q = torch.randn(8, 2, dtype=torch.float64, requires_grad=True)  # a batch of 8
b = torch.ones(8, 3, dtype=torch.float64, requires_grad=True)

result = solver.solve(P_values, A_values, q, b)
result.x.sum().backward()   # gradients w.r.t. P_values, A_values, q, b
```

**JAX** — `jit` / `vmap` / `grad`:

```python
import jax, jax.numpy as jnp
from moreau.jax import Solver
import moreau

cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
solver = Solver(n=2, m=3,
                P_row_offsets=[0, 1, 2], P_col_indices=[0, 1],
                A_row_offsets=[0, 2, 3, 4], A_col_indices=[0, 1, 0, 1],
                cones=cones)

P_data, A_data = jnp.array([1.0, 1.0]), jnp.array([1.0, 1.0, 1.0, 1.0])
q, b = jnp.array([2.0, 1.0]), jnp.array([1.0, 0.7, 0.7])

grad_q = jax.grad(lambda q: solver.solve(P_data, A_data, q, b).x.sum())(q)
```

## Batching

`CompiledSolver` reuses one sparsity pattern across a batch of problems and
solves them together on the GPU:

```python
import numpy as np
import moreau

cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
settings = moreau.Settings(device="cuda", batch_size=1024)

solver = moreau.CompiledSolver(
    n=2, m=3,
    P_row_offsets=[0, 1, 2], P_col_indices=[0, 1],
    A_row_offsets=[0, 2, 3, 4], A_col_indices=[0, 1, 0, 1],
    cones=cones, settings=settings,
)
solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0, 1.0, 1.0])  # shared

qs = np.random.randn(1024, 2)              # different objective per problem
bs = np.tile([1.0, 0.7, 0.7], (1024, 1))
result = solver.solve(qs, bs)              # result.x has shape (1024, 2)
```

## Performance

Benchmarks on real optimization problems (February 2026).
**GPU:** NVIDIA H100 80GB (Modal Cloud), CUDA 12.6, cuDSS 0.7.1.
**CPU:** AMD EPYC (Modal Cloud) for Moreau CPU; Intel Core Ultra 9 285H (local)
for Mosek/Clarabel. **Baselines:** Mosek 11, Clarabel, diffcp.

### Single solve

| Problem | Moreau CUDA (H100) | Mosek | Clarabel | vs Mosek | vs Clarabel |
|---------|-------------------|-------|----------|----------|-------------|
| **Multi-Period OPF** (53,829 vars, 127K constr.) | **250 ms** | 5,876 ms | 9,414 ms | **24×** | **38×** |
| **HVAC MPC** (123,000 vars, 181K constr.) | **3,769 ms** | 107,074 ms | 129,587 ms | **28×** | **34×** |
| **Multi-Period Portfolio** (45,150 vars, 72K constr.) | **191 ms** | 1,343 ms | 3,644 ms | **7×** | **19×** |

### Backward pass (differentiable optimization)

| Problem | Moreau CUDA | diffcp (LSQR, CPU) | Speedup |
|---------|-------------|---------------------|---------|
| **Multi-Period OPF** (53,829 vars) | **420 ms** | 4,934 ms | **12×** |

diffcp supports only LP backward via an LSQR adjoint; for QP it falls back to
finite-difference re-solves (~170 s/iter). Moreau provides native backward for
both LP and QP. Full batched results are in the
[documentation](https://moreau.so).

**Problems:** *Multi-Period OPF* — ERCOT grid (5,800 buses, 696 generators,
11,447 branches), 3 time periods with ramp coupling (LP). *HVAC MPC* —
1,000-zone building thermal control, 12-step horizon (QP). *Multi-Period
Portfolio* — 3,000 assets, 50 factors, 3 rebalancing periods (QP). Numbers are
from internal benchmarks on the hardware above; results vary with hardware,
drivers, and problem data.

## Documentation

Full guides and API reference: **[moreau.so](https://moreau.so)**.

- [Installation](docs/installation.md) · [Quickstart](docs/quickstart.md)
- [Basic usage](docs/guide/basic-usage.md) · [Batching](docs/guide/batching.md) · [Solver settings](docs/guide/solver-settings.md)
- [PyTorch](docs/guide/pytorch-integration.md) · [JAX](docs/guide/jax-integration.md) · [CVXPY](docs/guide/cvxpy-integration.md) · [cvxpylayers](docs/guide/cvxpylayers-integration.md)
- [Direct conic constraints](docs/guide/direct-cones.md) · [PSD cones](docs/guide/psd-cones.md) · [Warm starting](docs/guide/warm-starting.md)
- [Device selection](docs/guide/device-selection.md) · [Julia](docs/guide/julia-integration.md) / [JuMP](docs/guide/jump-integration.md)

## Citation

If you use Moreau in your research, please cite it:

```bibtex
@software{moreau2026,
  author       = {Barratt, Shane and Nobel, Parth and Diamond, Steven},
  title        = {Moreau: {GPU}-Native Differentiable Optimization},
  year         = {2026},
  url          = {https://moreau.so},
}
```

## References

Moreau's interior-point algorithm derives from Clarabel and its GPU variant
CuClarabel; the backward (adjoint) pass derives from diffcp and diffqcp; the
CPU active-set solver derives from DAQP.

1. P. J. Goulart and Y. Chen. *Clarabel: An interior-point solver for conic
   programs with quadratic objectives.* arXiv:2405.12762, 2024.
   <https://arxiv.org/abs/2405.12762>
2. Y. Chen, D. Tse, P. Nobel, P. Goulart, and S. Boyd. *CuClarabel: GPU
   Acceleration for a Conic Optimization Solver.* ACM Transactions on
   Mathematical Software (to appear). <https://doi.org/10.1145/3815420>
3. A. Agrawal, S. Barratt, S. Boyd, E. Busseti, and W. Moursi. *Differentiating
   through a Cone Program.* Journal of Applied and Numerical Optimization,
   1(2):107–115, 2019.
4. Q. Healey, P. Nobel, and S. Boyd. *Differentiating Through a Quadratic Cone
   Program.* Optimization Letters (to appear). arXiv:2508.17522, 2025.
   <https://arxiv.org/abs/2508.17522>
5. D. Arnström, A. Bemporad, and D. Axehill. *A Dual Active-Set Solver for
   Embedded Quadratic Programming Using Recursive LDL^T Updates.* IEEE
   Transactions on Automatic Control, 67(8):4362–4369, 2022.
   <https://arxiv.org/abs/2103.16236>

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) and our
[Code of Conduct](CODE_OF_CONDUCT.md). To report a security issue, see
[SECURITY.md](SECURITY.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for full terms
and upstream attribution (Clarabel.rs, diffqcp, DAQP).
