# Installation

Moreau is open source under the Apache 2.0 license and is distributed via
[PyPI](https://pypi.org/). Install with pip:

```bash
pip install moreau
```

---

## Quick Install

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} CPU Only
:class-card: sd-border-primary

```bash
pip install moreau
```

Pure Python interface with Rust-based CPU solver.
:::

:::{grid-item-card} GPU Support
:class-card: sd-border-secondary

```bash
pip install moreau[cuda]
```

CUDA 12 by default, or `moreau[cuda13]` for CUDA 13. Requires Python 3.12+.
:::

::::

---

## Choose Your Framework

::::{tab-set}

:::{tab-item} PyTorch
:sync: pytorch

If [PyTorch](https://pytorch.org/get-started/locally/) (>= 2.4) is installed, `moreau.torch` works automatically — no extra install step needed.

Provides `moreau.torch.Solver` for neural network integration with full autograd support.
:::

:::{tab-item} JAX
:sync: jax

If [JAX](https://docs.jax.dev/en/latest/installation.html) (>= 0.4.0) is installed, `moreau.jax` works automatically — no extra install step needed.

Provides `moreau.jax.Solver` compatible with `jax.grad`, `jax.vmap`, and `jax.jit`.
:::

::::

---

## Available Extras

| Extra | Installs | Notes |
|-------|----------|-------|
| `moreau[cuda]` | `moreau-cuda12` | CUDA 12 GPU backend (default) |
| `moreau[cuda12]` | `moreau-cuda12` | CUDA 12 GPU backend (explicit) |
| `moreau[cuda13]` | `moreau-cuda13` | CUDA 13 GPU backend |
| `moreau[test]` | `pytest`, `cvxpy` | Test dependencies |

---

## Supported Platforms

| OS | Arch | CPU | CUDA 12 | CUDA 13 |
|----|------|:---:|:-------:|:-------:|
| Linux | x86_64 | ✓ | ✓ | ✓ |
| Linux | aarch64 | ✓ | ✓ | ✓ |
| macOS | ARM64 | ✓ | — | — |

Linux wheels target `manylinux_2_28` (glibc ≥ 2.28). aarch64 CUDA wheels cover
Grace Hopper (sm_90) and Orin/Thor (sm_87); CUDA 13 aarch64 additionally covers
Blackwell (sm_120). If your device's compute capability is not listed, build
from source with `MOREAU_CUDA_ARCH=<capability>`.

---

## Verify Installation

Run the built-in diagnostic to check that everything is working:

```bash
python -m moreau check
```

This tests all installed backends (CPU, CUDA), framework integrations (PyTorch, JAX),
and autograd support in a single command.

For manual verification with each framework:

::::{tab-set}

:::{tab-item} Basic
:sync: basic

```python
import moreau
import numpy as np
from scipy import sparse

# Simple QP
P = sparse.diags([1.0, 1.0], format='csr')
q = np.array([1.0, 1.0])
A = sparse.csr_array([[1.0, 0.0], [0.0, 1.0]])
b = np.array([0.5, 0.5])

cones = moreau.Cones(num_nonneg_cones=2)
solver = moreau.Solver(P, q, A, b, cones=cones)
solution = solver.solve()

print(f"Solution: {solution.x}")
print(f"Status: {solver.info.status}")
print("Installation successful!")
```
:::

:::{tab-item} PyTorch
:sync: pytorch

```python
import torch
from moreau.torch import Solver
import moreau

cones = moreau.Cones(num_nonneg_cones=2)
solver = Solver(
    n=2, m=2,
    P_row_offsets=torch.tensor([0, 1, 2]),
    P_col_indices=torch.tensor([0, 1]),
    A_row_offsets=torch.tensor([0, 1, 2]),
    A_col_indices=torch.tensor([0, 1]),
    cones=cones,
)

P_values = torch.tensor([1.0, 1.0], dtype=torch.float64)
A_values = torch.tensor([1.0, 1.0], dtype=torch.float64)

q = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
b = torch.tensor([0.5, 0.5], dtype=torch.float64)
solution = solver.solve(P_values, A_values, q, b)

print(f"Solution: {solution.x}")
print(f"Device: {solver.device}")
print("PyTorch installation successful!")
```
:::

:::{tab-item} JAX
:sync: jax

```python
import jax.numpy as jnp
from moreau.jax import Solver
import moreau

cones = moreau.Cones(num_nonneg_cones=2)
solver = Solver(
    n=2, m=2,
    P_row_offsets=jnp.array([0, 1, 2]),
    P_col_indices=jnp.array([0, 1]),
    A_row_offsets=jnp.array([0, 1, 2]),
    A_col_indices=jnp.array([0, 1]),
    cones=cones,
)

P_data = jnp.array([1.0, 1.0])
A_data = jnp.array([1.0, 1.0])
q = jnp.array([1.0, 1.0])
b = jnp.array([0.5, 0.5])

solution = solver.solve(P_data, A_data, q, b)

print(f"Solution: {solution.x}")
print(f"Device: {solver.device}")
print("JAX installation successful!")
```
:::

::::

---

## GPU Acceleration

:::{admonition} CUDA Support
:class: tip

For NVIDIA GPUs, the CUDA backend provides significant speedups, especially for batched problems.
:::

**Check GPU availability:**

```python
import moreau

print(f"Available devices: {moreau.available_devices()}")
print(f"Default device: {moreau.default_device()}")
print(f"CUDA available: {moreau.device_available('cuda')}")
```

**Force GPU usage:**

```python
settings = moreau.Settings(device='cuda')
solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
```

---

## Dependencies

:::{dropdown} Core Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| Python | >= 3.9 | Runtime |
| NumPy | >= 1.19.0 | Array operations |
| Pydantic | >= 2.0.0 | Data validation |
| moreau-cpu | >= 0.3.3 | CPU solver backend |
:::

:::{dropdown} Optional Dependencies
| Package | Version | Purpose |
|---------|---------|---------|
| moreau-cuda12 | >= 0.3.3 | GPU solver backend (CUDA 12, Python >= 3.12) |
| moreau-cuda13 | >= 0.3.3 | GPU solver backend (CUDA 13, Python >= 3.12) |
| PyTorch | >= 2.4 | PyTorch integration |
| JAX | >= 0.4.0 | JAX integration |
| SciPy | >= 1.6.0 | Sparse matrix construction (used in examples) |
:::

---

## Troubleshooting

:::{dropdown} ImportError: moreau-cpu not found
The core CPU backend is required. Install it with:
```bash
pip install moreau-cpu
```
Or reinstall moreau:
```bash
pip install --force-reinstall moreau
```
:::

:::{dropdown} CUDA not detected
Ensure you have:
1. An NVIDIA GPU with CUDA support
2. CUDA drivers installed
3. The CUDA backend: `pip install moreau[cuda]`

Check CUDA availability:
```python
import moreau
print(moreau.device_available('cuda'))

# Diagnose the error
error = moreau.device_error('cuda')
if error:
    print(f"CUDA error: {error}")
```
:::

:::{dropdown} PyTorch/JAX import errors
Install the frameworks separately:

- **PyTorch**: See [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
- **JAX**: See [docs.jax.dev/installation](https://docs.jax.dev/en/latest/installation.html)

Once installed, `moreau.torch` and `moreau.jax` are available automatically.
:::
