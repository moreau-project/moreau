# Moreau - Batched Differentiable Convex Conic Solver

## Critical Rules

1. **No Silent Breakage**: Every change compiles (CPU Rust + GPU CUDA), passes tests, includes tests for new functionality
2. **GPU Performance**: Zero host↔device transfers or allocations inside iteration loops. All memory preallocated. No Python loops over batch dim.
3. **Numerical Fidelity**: Don't modify KKT/cone/diff math without instruction. CPU/GPU must agree to tolerance.
4. **No Tech Debt**: No hacks, no unresolved TODOs, no commented-out code. Unimplemented → `NotImplementedError`. No silent fallbacks.
5. **Documentation**: New APIs need docstrings + `docs/` updates in same PR. Always update `docs/` when features are added or changed.
6. **Prefer C++/Rust tests during active development**: Use C++ (CUDA) and Rust (CPU) tests for verifying correctness — they compile and run faster than the Python round-trip. Use Python tests for integration/API-level validation.
7. **Keep STATE.md, README.md, and docs/ current**: When adding or changing features, update `STATE.md` (project map and architecture), `README.md`, and `docs/` in the same PR. `STATE.md` is the source of truth for codebase layout — if you add files, move modules, or change architecture, reflect it there.
8. **Always run `git status` first**: Before starting work, run `git status` to know the current branch, uncommitted changes, and working state. Do not assume branch context.
9. **Never break the public API without a version bump and a deprecation**: follow the rules below.
10. **Never merge your own work**: follow the merging rules below.

## Versioning & Backward Compatibility

Moreau follows [Semantic Versioning 2.0.0](https://semver.org). Every package
(`moreau`, `moreau-cpu`, `moreau-cuda12/13`, `Moreau.jl`, the C API) shares one
version, changed only by `scripts/release.sh prepare` (see `RELEASE.md`).

**Public API**: everything a user can reach without a leading underscore:
names exported from `moreau`, `moreau.torch` and `moreau.jax`; `Settings`,
`IPMSettings` and `ActiveSetSettings` fields and their **defaults**;
`SolverStatus` values; solution/info fields, shapes and dtypes; `backward()`
keys; `include/moreau.h`; `Moreau.jl` and its MOI options. `_`-prefixed names
are private and may change in any release.

**Breaking change**: removing or renaming public API, changing a default
(including which solver or method `'auto'` picks), changing a return type,
shape or key, newly rejecting input that used to be accepted, or raising a
minimum supported Python, JAX, torch, CUDA or Julia version.

| Release | While 0.x (`0.MINOR.PATCH`) | From 1.0 (`MAJOR.MINOR.PATCH`) |
|---|---|---|
| PATCH | Bug fixes only. No public API or default changes. | Same |
| MINOR | New features **and** breaking changes | New backward-compatible features |
| MAJOR | — | Breaking changes |

Exception: a fix that stops returning a **wrong result** (for example, raising
on NaN input instead of returning `Solved`) may ship in a PATCH, even though
it rejects input that used to be accepted.

**Deprecation before removal.** A public API is removed only after at least one
released MINOR version in which using it emits a `DeprecationWarning` (Python)
or `Base.depwarn` (Julia). The warning names the replacement and the version
that removes it. Exception: an API that returns wrong results may raise
immediately in the next MINOR release.

**Every PR states its impact.** The PR description says one of: "No public API
change", "New feature", "Bug fix", or "**Breaking:** <what changed> —
<migration>". Release notes list every breaking change with its migration.
Breaking changes and deprecations get their own PR, not one bundled with
unrelated work.

## Merging

- All changes reach `main` through a pull request; never push to `main`
  directly or force-push it.
- A PR merges only when all of these hold:
  - it is not a draft;
  - every CI check is green, including the GPU job (CI DGX Spark);
  - it has at least one approving review from a maintainer other than the author;
  - no review thread is unresolved;
  - the branch is up to date with `main`.
- Claude never merges, approves, or enables auto-merge on a PR; a human does.
- Merge with a merge commit (the repository's convention). Before review, keep
  the branch as one or a few self-contained commits. After review starts, add
  commits rather than rewrite history; if a rebase is unavoidable, use
  `git push --force-with-lease` and say so on the PR.
- Merge a fix and its regression test in the same PR. Merge stacked or dependent
  PRs in order, and rerun CI on the later ones after each merge.

## Project Overview

Modified Clarabel with backward pass from diffqcp.

**References**: [Clarabel.rs](https://github.com/oxfordcontrol/Clarabel.rs) (CPU base), [diffqcp](https://github.com/cvxgrp/diffqcp) (backward pass), [DAQP](https://github.com/darnstrom/daqp) (CPU active-set solver)

## Architecture

```
packages/
├── moreau/           # Pure Python wrapper (unified API)
├── moreau-cpu/       # Rust solver (maturin build)
└── moreau-cuda/      # C++/CUDA solver (CMake build)
    ├── src/          # C++/CUDA source
    └── include/moreau/  # C++ headers
```

**Tests**: `packages/moreau/tests/python/` (unified), `packages/moreau-cpu/tests/python/` (CPU), `packages/moreau-cuda/tests/cpp/` (GPU C++)

## Build Commands

```bash
# CPU (Rust)
cd packages/moreau-cpu && maturin develop --release --features sdp-openblas,python && cargo test --features sdp-openblas

# GPU (C++/CUDA) — specify arch to speed up compilation (sm_120 = RTX 5090)
cd packages/moreau-cuda && mkdir -p build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=120 && make -j4 && ctest --output-on-failure

# GPU (Python wheel) — run from repo root
MOREAU_CUDA_ARCH=120 pip install -e packages/moreau-cuda --no-build-isolation

# C API shared libraries (for Julia, standalone C/C++ usage)
packages/moreau-c/build.sh cpu          # CPU only (requires Rust)
packages/moreau-c/build.sh cuda         # CUDA only (requires nvcc)
packages/moreau-c/build.sh all          # Both (skips CUDA if nvcc missing)
# Output: dist/c-api/{include/moreau.h, lib/libmoreau_cpu.so, lib/libmoreau_cuda.so}
# NOTE: CPU C API requires --features c-api (the build script handles this)

# Tests
pytest packages/moreau/tests/python/ -v
```

## Problem Formulation

```
minimize    (1/2)x'Px + q'x
subject to  Ax + s = b
            x ∈ K1,  s ∈ K2
```
K1, K2 = products of cones (zero, nonneg, SOC, PSD, exp, power, generalized power — the required slack row order).
K2 constrains the slack s; K1 constrains x directly (direct-x cones).

## Python API

### Single Problem (Solver)

```python
import moreau
from scipy import sparse
import numpy as np

P = sparse.diags([1.0, 1.0], format='csr')
A = sparse.csr_matrix([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

solver = moreau.Solver(P, q=np.array([2., 1.]), A=A, b=np.array([1., 0.7, 0.7]), cones=cones)
solution = solver.solve()  # solution.x, .z, .s
info = solver.info         # info.status, .obj_val, .iterations, .solve_time
```

### Batched Problems (CompiledSolver)

```python
# 1. Construct with structure (sparsity pattern, cones, settings with batch_size)
settings = moreau.Settings(batch_size=4)
solver = moreau.CompiledSolver(n=2, m=3, P_row_offsets=[0,1,2], P_col_indices=[0,1],
                               A_row_offsets=[0,2,3,4], A_col_indices=[0,1,0,1],
                               cones=cones, settings=settings)
# 2. Setup matrix values: (nnz,) shared or (batch, nnz) per-problem
solver.setup(P_values=[1.,1.], A_values=[1.,1.,1.,1.])  # shared across batch
# 3. Solve: shape (batch, n) or (batch, m)
solution = solver.solve(qs=[[2.,1.]]*4, bs=[[1.,0.7,0.7]]*4)
```

### PyTorch (with autograd)

```python
from moreau.torch import Solver
solver = Solver(n=2, m=3, P_row_offsets=P_ro, P_col_indices=P_ci,
                A_row_offsets=A_ro, A_col_indices=A_ci, cones=cones)
solution = solver.solve(P_values, A_values, q, b)  # stateless, P/A cached
solution.x.sum().backward()  # gradients flow through
```

### JAX (with jit/vmap/grad)

```python
from moreau.jax import Solver
solver = Solver(n=2, m=3, P_row_offsets=P_ro, P_col_indices=P_ci,
                A_row_offsets=A_ro, A_col_indices=A_ci, cones=cones)

# Option 1: Full signature
solution = solver.solve(P_data, A_data, q, b)

# Option 2: Two-step API (setup P/A once, solve with 2 args)
solver.setup(P_data, A_data)
solution = solver.solve(q, b)
jax.vmap(solver.solve)(q_batch, b_batch)  # batched
jax.grad(lambda q: solver.solve(q, b).x.sum())(q)  # gradients
```

### Device Selection & Auto-Tune

```python
moreau.Settings(device='auto')  # 'auto', 'cpu', 'cuda'
moreau.set_default_device('cuda')  # global override
```

When `device='auto'` or `direct_solve_method='auto'`, a heuristic picks the best
configuration without benchmarking. Set `auto_tune=True` to benchmark on first
`solve()` call and lock in the fastest.

- `auto_tune=False` (default): heuristic selection, no benchmarking
- `auto_tune=True` + `device='auto'`: benchmarks all device/method combinations
- `auto_tune=True` + explicit device + `method='auto'`: benchmarks methods for that device
- Both device and method explicit: no auto-tune regardless of `auto_tune`

### Warm Starting (Experimental)

```python
# Solve, then warm start a related problem
solution = solver.solve(qs=q1, bs=b1)
ws = solution.to_warm_start()            # BatchedSolution -> BatchedWarmStart
solution2 = solver.solve(qs=q2, bs=b2, warm_start=ws)
```

Auto-retries cold if warm-started solve fails (any status except Solved/AlmostSolved/MaxIterations/CallbackTerminated). Configurable via `IPMSettings(warm_start_no_retry=...)`.

### YOLO Mode (Fixed-Iteration, Zero-Sync)

```python
settings = moreau.Settings(
    device='cuda', batch_size=128,
    yolo=True, yolo_num_iters=10,
)
```

Runs exactly `yolo_num_iters` IPM iterations with no convergence checking and no
GPU-host sync. All batches return `MaxIterations` status. Automatically preserves
last non-NaN iterate (safe to overshoot). Incompatible with `enable_grad=True`.

## Key Types

- `Cones(num_zero_cones, num_nonneg_cones, so_cone_dims, num_exp_cones, power_alphas, gen_power_cone_params, psd_dims)` — `so_cone_dims` is a list of ints (each >= 2). Backward compat: `num_so_cones=N` creates `so_cone_dims=[3]*N`. `gen_power_cone_params` is a list of `(alphas, dim2)` tuples where alphas are positive floats summing to 1 and dim2 >= 1.
- `Settings(solver, device, batch_size, max_iter, verbose, enable_grad, yolo, yolo_num_iters, auto_tune, ipm_settings, active_set_settings)` — `solver`: `'auto'` (default, picks active-set for small QPs), `'ipm'`, or `'active_set'` (CPU, zero+nonneg cones only)
- `IPMSettings(tol_gap_abs, tol_feas, direct_solve_method, warm_start_no_retry, ...)` - tolerances default 1e-8. `direct_solve_method`: `'auto'` (default — uses Riccati for block-tridiagonal structure e.g. MPC/LQR, otherwise heuristic unless `auto_tune=True`), `'qdldl'` (CPU), `'faer'` (CPU), `'riccati'` (CPU/CUDA, block-tridiagonal only), `'cudss'` (CUDA), `'woodbury'` (CUDA, diagonal P + low-rank A, portfolio-type)
- `ActiveSetSettings(primal_tol, dual_tol, iter_limit, diff_method, diff_smoothing_mu, ...)` — `diff_method`: `'exact'` (default) or `'smoothed'`
- `SolverStatus`: Solved, PrimalInfeasible, DualInfeasible, AlmostSolved, MaxIterations, NumericalError, ...
- `Solution`: x, z, s (shapes: n or m); `to_warm_start()` -> `WarmStart`
- `BatchedSolution`: same fields with shape (batch, n/m), indexable: `solution[i]`; `to_warm_start()` -> `BatchedWarmStart`
- `WarmStart`: x, z, s arrays for warm starting a single problem
- `BatchedWarmStart`: x, z, s arrays (batch, dim), indexable/iterable

## Supported Cones

- **Zero cone**: equality constraints (any dimension)
- **Nonnegative cone**: inequality constraints (any dimension)
- **Second-order cone (SOC)**: arbitrary dimension >= 2 (specified per-cone via `so_cone_dims`)
- **Exponential cone**: 3D
- **Power cone**: 3D, alpha ∈ (0, 1)
- **Generalized power cone (GenPowerCone)**: variable dimension. K = {(p,w) : ∏ p_i^αi ≥ ||w||₂, p_i ≥ 0} where α_i > 0, Σα_i = 1. Specified via `gen_power_cone_params=[(alphas, dim2), ...]`.
- **PSD cone (SDP)**: any matrix dimension (`psd_dims=[n1, n2, ...]`); supported on CPU and CUDA in both `Solver` and `CompiledSolver` (batch) with backward pass. Chordal decomposition for sparse PSD is supported on both CPU and CUDA.

## Known Limitations

- Double precision (float64) only
- P matrix: must be full symmetric (both upper and lower triangles, validated at construction)

## License

Apache 2.0. See `LICENSE` at the repo root and `NOTICE` for upstream attribution
(Clarabel.rs, diffqcp, DAQP).

## Profiling

```bash
nsys profile -o profile ./test && nsys stats profile.nsys-rep
cmake .. -DMOREAU_ENABLE_PROFILING=ON && MOREAU_PROFILE_ENABLED=1 ./test
```
