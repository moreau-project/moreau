# Device Selection

Moreau automatically selects the best available device, or you can manually specify CPU or GPU.

## Automatic Selection

By default, Moreau uses heuristics to pick the best device based on problem size:

```python
import moreau

# Uses CUDA if available and problem is large enough, otherwise CPU
solver = moreau.Solver(P, q, A, b, cones=cones)
print(f"Using device: {solver.device}")
```

The automatic selection considers:
1. **CUDA availability** — falls back to CPU if CUDA is not installed
2. **Problem size** ($n$) — small problems ($n < 500$) always use CPU; large problems ($n \ge 750$) always use CUDA
3. **Constraint sparsity** ($\text{nnz}(A)$) — CUDA preferred when $\text{nnz}(A) \ge 50{,}000$
4. **Batch size** — CUDA preferred when batch size $\ge 2$ and $\text{nnz}(A) \ge 25{,}000$

### Heuristic Selection (Default)

By default (`auto_tune=False`), when `device='auto'` or `direct_solve_method='auto'`,
Moreau uses a heuristic to pick the best device and method **without benchmarking**.
This avoids any first-solve overhead:

```python
# Heuristic picks device + method instantly (no benchmarking)
settings = moreau.Settings(device='auto')  # auto_tune=False by default
solver = moreau.CompiledSolver(..., settings=settings)
solution = solver.solve(qs, bs)  # no first-solve penalty
```

The heuristic considers problem size, sparsity, and batch size to choose a
device/method combination:

- **CUDA**: cuDSS (general), Riccati (block-tridiagonal MPC/LQR), Woodbury (diagonal P + low-rank constraints)
- **CPU**: faer preferred when KKT dimension $\ge 500$ or $\text{nnz}(A) \ge 10{,}000$; QDLDL otherwise; Riccati for block-tridiagonal structure

### Opt-In Auto-Tune (Benchmarking)

For maximum performance, enable auto-tune to benchmark all candidate configurations
on the first solve and lock in the fastest:

```python
# Enable benchmarking on first solve
settings = moreau.Settings(device='auto', auto_tune=True)
solver = moreau.CompiledSolver(..., settings=settings)

# First solve benchmarks all combos (emits a UserWarning)
solution = solver.solve(qs, bs)  # ~2-4x slower

# Subsequent solves use the locked-in winner
solution = solver.solve(qs, bs)  # normal speed
```

Candidates are ordered by a heuristic so the likely-best runs first and sets a
tight time limit (1.5x the best time) for challengers. The heuristic ordering:

- **CUDA**: cuDSS (general), Riccati (block-tridiagonal), Woodbury (portfolio-type)
- **CPU**: faer tried first when KKT dimension $\ge 500$ or $\text{nnz}(A) \ge 10{,}000$; QDLDL otherwise; Riccati for block-tridiagonal

### Method Auto-Tune with Explicit Device

When the device is set explicitly (e.g. `device='cuda'`) but
`direct_solve_method='auto'` and `auto_tune=True`, the **first call to `solve()`**
benchmarks the available methods for that device and locks in the fastest:

```python
# Device explicit, method auto, benchmarking enabled
settings = moreau.Settings(
    device='cuda',
    auto_tune=True,
    ipm_settings=moreau.IPMSettings(direct_solve_method='auto'),
)
# First solve uses cudss on CUDA (only available method)
```

Without `auto_tune=True`, the heuristic picks a method without benchmarking.

### Interaction with `set_default_device()`

If `set_default_device()` has been called and `device='auto'`, the override is
used directly and auto-tune is skipped.

---

## Manual Device Selection

Override device selection via `Settings`:

```python
# Force CPU
settings = moreau.Settings(device='cpu')
solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)

# Force CUDA
settings = moreau.Settings(device='cuda')
solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
```

---

## Checking Availability

Query available devices:

```python
import moreau

# List all available devices (sorted by priority, highest first)
print(moreau.available_devices())  # ['cuda', 'cpu'] or ['cpu']

# Check specific device
print(moreau.device_available('cuda'))  # True/False

# Get default device
print(moreau.default_device())  # 'cuda' or 'cpu'

# Diagnose why a device is unavailable
error = moreau.device_error('cuda')
if error:
    print(f"CUDA unavailable: {error}")
```

---

## Global Default

Set a global default device:

```python
import moreau

# Set default for all new solvers
moreau.set_default_device('cuda')

# Now all solvers use CUDA by default
solver = moreau.Solver(P, q, A, b, cones=cones)  # Uses CUDA

# Can still override per-solver
settings = moreau.Settings(device='cpu')
cpu_solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)

# Reset to automatic selection
moreau.set_default_device(None)
```

---

## Framework Integration

### PyTorch

The PyTorch integration follows the same pattern:

```python
import torch
from moreau.torch import Solver
import moreau

cones = moreau.Cones(num_nonneg_cones=2)

# Auto-selects device
solver = Solver(
    n=2, m=2,
    P_row_offsets=torch.tensor([0, 1, 2]),
    P_col_indices=torch.tensor([0, 1]),
    A_row_offsets=torch.tensor([0, 1, 2]),
    A_col_indices=torch.tensor([0, 1]),
    cones=cones,
)
print(f"Device: {solver.device}")

# Force specific device
settings = moreau.Settings(device='cuda')
solver = Solver(
    n=2, m=2,
    P_row_offsets=torch.tensor([0, 1, 2]),
    P_col_indices=torch.tensor([0, 1]),
    A_row_offsets=torch.tensor([0, 1, 2]),
    A_col_indices=torch.tensor([0, 1]),
    cones=cones,
    settings=settings,
)
```

### JAX

Similarly for JAX:

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
print(f"Device: {solver.device}")
```

---

## Performance Guidance

The GPU's advantage comes from accelerating the sparse KKT factorization at each
interior-point iteration. Factorization cost depends on problem dimensions and
sparsity structure, but grows quickly with `n + m` — especially for dense or
high-fill-in problems. There is a crossover point below which the GPU's fixed
overhead (~25 ms from cuDSS initialization) dominates and CPU wins.

The numbers below were measured on an RTX 5090 Mobile. Datacenter GPUs (A100,
H100, B200) are roughly 2–4x faster, so their crossover points shift lower.

### CPU Performance

Moreau's CPU backend is not a fallback — it is a high-performance Rust
implementation that is often the best choice. The CPU solver uses optimized
sparse direct factorizations (QDLDL for small systems, faer for large ones with
automatic multi-threading) and has near-zero per-solve overhead, making it
especially efficient for batched workloads.

For problems with $n \le 500$, the CPU backend solves a 200-variable QP in under
2 ms and scales linearly across a batch: 128 such problems complete in ~184 ms
total (1.4 ms each), compared to 1,095 ms on GPU. This means a CPU-only
deployment can comfortably handle high-throughput workloads — hundreds of
small-to-medium solves per second — with no GPU infrastructure needed. Batching
on CPU is efficient because each solve
reuses the compiled problem structure with minimal dispatch overhead.


### Problem Size

Problem size ($n$) is the single strongest predictor:

| Variables ($n$) | Constraints ($m \approx 0.75n$) | CPU | GPU | Winner |
|:---:|:---:|---:|---:|:---:|
| 10–100 | 7–75 | 0.1–2 ms | 3–11 ms | **CPU** (5–40x) |
| 200 | 150 | 9 ms | 25 ms | **CPU** (2.8x) |
| 500 | 375 | 77 ms | 49 ms | **GPU** (1.6x) |
| 750 | 562 | 253 ms | 139 ms | **GPU** (1.8x) |
| 1,000 | 750 | 566 ms | 81 ms | **GPU** (7x) |
| 2,000 | 1,500 | 4,527 ms | 661 ms | **GPU** (6.9x) |

**Crossover: $n \approx 300$–$500$** on a laptop GPU; likely $n \approx 150$–$300$ on datacenter GPUs.

### Constraint Density

Denser constraint matrices create larger KKT systems, pulling the crossover
toward GPU. Measured at $n = 500$:

| A density | $\text{nnz}(A)$ | Winner |
|:---:|:---:|:---:|
| 0.1 % | 500 | **CPU** (1.7x) |
| 1 % | 1,900 | **CPU** (1.5x) |
| 5 % | 9,100 | **CPU** (1.3x) |
| 10 % | 17,900 | tie |
| 20 % | 33,900 | **GPU** (2x) |
| 50 % | 73,700 | **GPU** (3.1x) |

**Rule of thumb:** at $n = 500$, GPU starts winning around $\text{nnz}(A) \approx 15{,}000$–$18{,}000$.

### Constraint Ratio

More constraints per variable means a larger KKT system. At $n = 500$:

| $m / n$ | Winner |
|:---:|:---:|
| 0.25 | **CPU** (1.4x) |
| 0.5 | **CPU** (1.6x) |
| 1.0 | **GPU** (1.4x) |
| 2.0 | **GPU** (1.9x) |
| 4.0 | **GPU** (2.8x) |

### Sparsity Pattern

Structured (banded) sparsity favors CPU more than random sparsity because the
KKT factor is sparser and the CPU solver exploits this better:

| Pattern | $n = 500$ | $n = 1{,}000$ |
|:---:|:---:|:---:|
| Random | GPU (1.1x) | GPU (5.3x) |
| Banded | **CPU** (1.7x) | GPU (1.7x) |
| Diagonal P | tie | GPU (2.7x) |

### Problem Type

All cone types follow the same size-driven crossover. LPs show the strongest GPU
advantage at large sizes because the zero P matrix makes the KKT system cheaper
to factor, so more iterations happen per second on GPU:

| Type | $n = 200$ | $n = 500$ | $n = 1{,}000$ |
|:---:|:---:|:---:|:---:|
| QP | CPU (2.8x) | GPU (1.6x) | GPU (7x) |
| SOCP | CPU (1.5x) | GPU (1.04x) | GPU (3.5x) |
| ExpCone | CPU (4.3x) | tie | — |
| LP | CPU (1.4x) | GPU (2.1x) | GPU (11.4x) |

### Batching

Moreau solves each batch element sequentially within a single `solve()` call, so
batching scales total time linearly on both CPU and GPU. This means **batching
does not change the crossover** — the device that wins for a single problem also
wins per-problem in a batch:

| | $n = 200$, bs = 128 | $n = 500$, bs = 128 | $n = 750$, bs = 128 |
|:---:|:---:|:---:|:---:|
| CPU total | 184 ms | 2,289 ms | 8,473 ms |
| GPU total | 1,095 ms | 2,766 ms | 7,543 ms |
| CPU/prob | 1.4 ms | 17.9 ms | 66.2 ms |
| GPU/prob | 8.6 ms | 21.6 ms | 58.9 ms |
| Winner | **CPU** (6x) | **CPU** (1.2x) | **GPU** (1.1x) |

At small $n$ ($\le 200$), GPU overhead is amplified by batch size — the gap actually
widens. At $n = 500$, CPU wins the batched case even though GPU barely wins the
single-problem case, because GPU per-problem overhead accumulates.

### Quick Reference

| Scenario | Recommended Device |
|---|---|
| $n < 300$ | CPU — always, regardless of batch size |
| $300 \le n < 750$, sparse $A$ | CPU |
| $300 \le n < 750$, dense $A$ or high $m/n$ | CUDA |
| $n \ge 750$ | CUDA |
| Banded / structured sparsity, $n < 1{,}000$ | CPU |
| Quick prototyping, any size | CPU (no CUDA install needed) |

The `device='auto'` default uses a heuristic to pick the best configuration
without benchmarking. Set `auto_tune=True` to benchmark on the first solve
for maximum performance.
