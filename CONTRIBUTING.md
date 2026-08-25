# Contributing to Moreau

Thanks for your interest in contributing! Moreau is a batched, differentiable
conic optimization solver with CPU (Rust) and GPU (C++/CUDA) backends and a
unified Python interface. Contributions of all kinds — bug reports, fixes,
tests, docs, and features — are welcome.

By contributing you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE).

## Repository layout

```
packages/
├── moreau/        # Unified Python interface (hatchling)
├── moreau-cpu/    # CPU backend, Rust (maturin)
└── moreau-cuda/   # GPU backend, C++/CUDA (scikit-build-core + nanobind)
```

User-facing documentation lives in `docs/` and at the project documentation
site.

## Building

Use [`uv`](https://docs.astral.sh/uv/) for Python tasks.

```bash
# CPU (Rust) — requires a Rust toolchain
cd packages/moreau-cpu
maturin develop --release --features sdp-openblas,python
cargo test --features sdp-openblas

# GPU (C++/CUDA) — requires nvcc + CMake (set arch to speed up compilation)
cd packages/moreau-cuda
mkdir -p build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=<your_sm_arch>
make -j4 && ctest --output-on-failure

# GPU Python wheel (from repo root)
MOREAU_CUDA_ARCH=<your_sm_arch> pip install -e packages/moreau-cuda --no-build-isolation
```

## Testing

Prefer the native test suites during development — they compile and run faster
than the Python round-trip:

- **Rust (CPU):** `cargo test` in `packages/moreau-cpu`
- **C++/CUDA (GPU):** `ctest` in `packages/moreau-cuda/build`
- **Python (integration/API):**

```bash
pytest packages/moreau/tests/python/ -v
pytest packages/moreau-cpu/tests/python/ -v
```

Every change should compile (CPU Rust + GPU CUDA where touched), pass existing
tests, and include tests for new functionality. CPU and GPU results must agree
to tolerance. Please don't weaken a test to make CI pass — fix the underlying
issue instead.

## Pull requests

- Branch off `main` and open a PR against `main`.
- Keep changes focused; update `docs/` and `STATE.md` in the same PR when you
  add or change features.
- New public APIs need docstrings and a `docs/` update.

## Documentation

See the [`docs/`](docs/) directory and the
[guides](docs/guide/index.md) for usage and integration details.
