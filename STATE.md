# Moreau Project State

Source of truth for codebase layout. Update in the same PR as any
architectural change (new file / module move / new feature).

## Packages

```
packages/
├── moreau/         # Pure Python — unified user-facing API
├── moreau-cpu/     # Rust — CPU solver (maturin build); IPM + active-set
├── moreau-cuda/    # C++/CUDA — GPU solver (CMake / scikit-build)
├── moreau-c/       # C ABI shared libraries (Julia, standalone C/C++)
└── moreau-julia/   # Julia MOI wrapper (Moreau.jl)
```

## packages/moreau (Python wrapper)

```
python/moreau/
├── __init__.py        # Solver, CompiledSolver (public API)
├── _backend.py        # Device + method resolution, auto-tune dispatch
├── _dispatch.py       # CPU/CUDA backend selection
├── _types.py          # Cones, Settings, IPMSettings, ActiveSetSettings,
│                      # Solution, BatchedSolution, WarmStart, BatchedWarmStart
├── _validation.py     # Input shape/dtype/sparsity validation
├── testing.py         # Public test helpers
├── torch/             # torch.Solver — PyTorch autograd + torch.compile
│   ├── __init__.py    # Solver API, eager/compiled dispatch
│   ├── _autograd.py   # Eager autograd and shared backward custom op
│   ├── _compiled.py   # Forward custom op, fake kernel, autograd registration
│   ├── _cpu_impl.py   # CPU backend adapter
│   └── _types.py      # Torch solution and warm-start containers
└── jax/               # jax.Solver — jit/vmap/grad-friendly (recently split out)

tests/python/          # Unified API integration tests
tests/python/bench/    # Decision-gate benchmarks (checked in)
```

The public direct cone API uses `DirectConeSpec` entries in `Cones.dir_cones`.
Both native solver bindings accept the `dir_cones` argument. See
[`docs/guide/direct-cones.md`](docs/guide/direct-cones.md) for usage.

PyTorch CPU and CUDA solves and backward passes use custom ops under
`torch.compile`, including with `fullgraph=True`. See the
[PyTorch guide](docs/guide/pytorch-integration.md#torchcompile).

## packages/moreau-cpu (Rust solver)

```
src/
├── lib.rs
├── c_api.rs                    # C ABI for moreau-c
├── active_set_ffi.rs           # FFI for active-set solver
├── python/                     # PyO3 bindings
├── algebra/                    # Linear algebra primitives
├── qdldl/                      # QDLDL direct sparse factorization
├── io/, timers/, utils/
└── solver/
    ├── api.rs                  # solver entry points
    ├── core/cones/             # Cone trait + impls (per-kind .rs files)
    └── implementations/
        ├── default/            # IPM solver
        │   ├── diff/           # Backward pass (recently split, 2026-04)
        │   ├── kkt/            # KKT assembly (recently split, 2026-04)
        │   └── kktsystem.rs    # KKT solve dispatch
        └── active_set/         # Small-QP active-set solver (DAQP-derived)
```

Cone modules: zero, nonneg, soc, exp, pow, genpow, psd-triangle, composite,
composite direct conic constraints. Symmetric and nonsymmetric helpers live in
`symmetric_common.rs` / `nonsymmetric_common.rs`.

Tests in `tests/` (top-level integration tests).

`default/compiled_solver.rs` shares Rayon pools among live compiled solvers
with the same thread count. The process-wide cache holds weak references;
solvers own the workers and release them when the last owner is dropped.
Its unit test checks sharing, and `tests/python/test_thread_pool_lifecycle.py`
checks constructing a fresh child solver after releasing parent solvers and
forking, with a timeout to catch hangs.

## packages/moreau-cuda (C++/CUDA solver)

```
src/
├── cones/        # Per-kind scaling/step kernels + composite kernels
├── kkt/          # KKT assembly, refresh, factorization dispatch;
│                 # also hosts direct-x cone kernels
├── diff/         # Backward-pass kernels (per-kind)
├── solver/       # IPM driver + warm-start
├── variables/    # Variable update kernels
├── residuals/    # Residual computation
├── equilibration/
├── chordal/      # Chordal decomposition for sparse PSD
└── vector/       # GPU vector primitives
```

Headers in `include/moreau/**` mirror the `src/` tree. Tests in
`tests/cpp/` (gtest binaries built with the main CMake config).

`diff/diff_kkt.cu` keeps cuDSS analysis, matrix/RHS assembly, factorization,
solves, refinement, and result extraction on an owned CUDA stream. This orders
shared workspace use across calls. Preallocated events connect it to the caller's
stream, including when JAX reuses a cached differentiation system on a different stream.
Backward vectors and CSR storage initialize on the caller's stream, including
lazy workspace allocation, Riccati workspace initialization, and direct-cone resizes.
`tests/cpp/test_diff.cpp` covers stream changes for single and batched gradients,
matrix updates, direct-cone dual cotangents, cold unconstrained solves, and
forward/adjoint solves with cached factors. Queued native solves and matrix
updates alternate nonblocking streams without synchronization between calls.
`tests/cpp/test_riccati.cpp` checks first-use backward initialization on a
nonblocking stream for equality and nonnegative MPC problems.

### Direct conic constraints and slack cones

Python specifies these with `DirectConeSpec` and `Cones.dir_cones`. Unknown
cone fields are rejected; generalized-power weights must be finite and positive.

Direct conic constraints constrain a sub-vector of x directly (`x[J] ∈ K`) instead
of introducing a slack `s = b - A·x`. Constraints on symmetric cones (nonneg,
SOC, PSD) use the slack-NT machinery with a primal↔dual swap; asymmetric
(exp, power, genpower) use explicit primal-barrier formulas. See the
block comment at the top of the direct-x section in
`packages/moreau-cuda/src/kkt/kkt_kernels.cu` and the matching CPU
trait methods in `packages/moreau-cpu/src/solver/core/cones/mod.rs`.

All direct-x SOC/nonneg/genpow scaling and step-math kernels are
block-cooperative (blockDim=`SOC_PARALLEL_BLOCK_SIZE`).

## packages/moreau-c (C ABI)

`build.sh` produces `dist/c-api/{include/moreau.h, lib/libmoreau_{cpu,cuda}.so}`.
CPU C API requires `--features c-api` on the moreau-cpu Cargo build.

## packages/moreau-julia (Moreau.jl)

Julia MOI wrapper. Links against `moreau-c`. Released under Apache 2.0.

## Test layout

| Where | What |
|---|---|
| `packages/moreau-cpu/tests/*.rs` | Rust integration tests (IPM, active-set, cones, backward) |
| `packages/moreau-cuda/tests/cpp/*.cpp` | GPU gtest binaries (per-kernel + end-to-end) |
| `packages/moreau/tests/python/*.py` | Unified-API integration tests |
| `packages/moreau/tests/python/bench/*.py` | Decision-gate benchmarks (checked-in artifacts) |

Critical Rule 6 in CLAUDE.md: prefer C++/Rust tests for fast inner-loop
verification; use Python tests for API/integration validation.

## Documentation

`docs/` contains the Sphinx user guides and API reference. The illustrated
[chordal decomposition guide](docs/guide/chordal-decomposition.md) covers
sparse PSD conversion, counterexamples, and CPU/CUDA execution paths.
Its checked-in SVG figures live in `docs/_static/figures/` and are generated
by `docs/_scripts/chordal_figures.py` (NumPy and Matplotlib).

## Build commands

See CLAUDE.md "Build Commands" section. Headline:

- CPU (Rust): `cd packages/moreau-cpu && maturin develop --release --features sdp-openblas,python && cargo test --features sdp-openblas`
- GPU (C++/CUDA): `cd packages/moreau-cuda/build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=<arch> && make -j4 && ctest --output-on-failure`
- GPU wheel: `MOREAU_CUDA_ARCH=<arch> pip install -e packages/moreau-cuda --no-build-isolation`
- Unified Python: `pytest packages/moreau/tests/python/ -v`

## License

Apache 2.0 (Apache License 2.0). See `LICENSE` and `NOTICE` for upstream
attribution (Clarabel.rs, diffqcp, DAQP).
