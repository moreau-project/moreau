# WebAssembly

The Rust `moreau-cpu` package supports single-threaded browser execution on
`wasm32-unknown-unknown` with default features disabled. It provides the QDLDL
interior-point solver, forward and adjoint differentiation, and sequential
`CompiledSolver` batches. Native builds retain their parallel batch execution.

## Rust consumer

Depend on the CPU package as `moreau`, with `default-features = false`. For a
checkout containing the consumer beside Moreau:

```toml
[dependencies]
moreau = { package = "moreau-cpu", path = "../moreau/packages/moreau-cpu", default-features = false }
wasm-bindgen = "0.2"
```

For a git dependency, replace `path` with the Moreau repository URL and an
immutable `rev`. Moreau's Python release version does not imply that the Rust
package is published on crates.io.

A binding can call the Rust API directly:

```rust
use moreau::{algebra::CscMatrix, solver::*};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn bounded_quadratic(bound: f64) -> Result<f64, JsError> {
    if !bound.is_finite() {
        return Err(JsError::new("bound must be finite"));
    }
    let p = CscMatrix::identity(1);
    let a = CscMatrix::identity(1);
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    let mut solver = DefaultSolver::new(
        &p, &[-2.0], &a, &[bound], &[NonnegativeConeT(1)], settings,
    ).map_err(|e| JsError::new(&e.to_string()))?;
    solver.solve();
    if solver.solution.status != SolverStatus::Solved {
        return Err(JsError::new("solve did not converge"));
    }
    Ok(solver.solution.x[0])
}
```

Build the consumer with `wasm-pack build --target web`. Browser hosts own
JavaScript bindings, worker lifetimes, cancellation, and presentation. A
long-running synchronous solve belongs in a dedicated worker to keep the UI
responsive. No shared memory, cross-origin isolation, or native threads are
required by this configuration.

## Differentiation and batches

`DefaultSolver::forward_batch` accepts original-coordinate perturbations and
returns `(dx, dz, ds)` in original coordinates. `backward_batch` accepts upstream
solution gradients and returns gradients of the problem data. These methods
apply the solver's equilibration internally. The lower-level
`diff::differentiate` and `diff::differentiate_adjoint` functions operate on the
coordinates supplied by the caller.

`CompiledSolver` retains setup, solve, warm-start, and backward APIs in the
browser. Its thread-count argument accepts `0` (automatic) or `1`; both use one
execution thread. Larger values return `SolverError::BadInputData`. Batch size
is independent of thread count, and results preserve input order.

The browser build excludes native C++ active-set, faer-sparse, BLAS/LAPACK SDP,
PARDISO, Python, and C API features. Enabling these features reports an
unsupported-configuration build error. Other WASM targets, WebGPU, and browser
multithreading are not covered by this support configuration.

## Validation

```sh
cargo check --manifest-path packages/moreau-cpu/Cargo.toml \
  --target wasm32-unknown-unknown --no-default-features --locked
cargo test --manifest-path packages/moreau-cpu/Cargo.toml \
  --no-default-features --test browser --locked
```

For browser execution, install `wasm-bindgen-cli` at the `wasm-bindgen` version
recorded in the CPU package's lockfile, Firefox, and geckodriver. Put the tools
on `PATH`, then run:

```sh
CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER=wasm-bindgen-test-runner \
  cargo test --manifest-path packages/moreau-cpu/Cargo.toml \
  --target wasm32-unknown-unknown --no-default-features --test browser --locked
```

The test runner starts a local server and launches headless Firefox. Tests
execute solves, infeasibility detection, forward/adjoint identities, finite
differences, repeated batches, and thread-count validation.
