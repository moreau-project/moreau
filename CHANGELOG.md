# Changelog

All notable changes to Moreau are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Releases up to
0.4.1 are described in the
[GitHub releases](https://github.com/moreau-project/moreau/releases).

## [Unreleased]

### Added

- `torch.compile` support, including `fullgraph=True`, for CPU and CUDA solves
  and gradients (#25).
- A warning when an explicit `solver="active_set"` would store more than 1 GiB
  of dense `P` and `A` (#53).

### Changed

- **Breaking:** `Settings`, `IPMSettings` and `ActiveSetSettings` reject
  unknown fields with a pydantic `ValidationError` instead of ignoring them.
  Remove or correct misspelled settings (#70).
- **Breaking:** malformed CSR input and duplicate entries raise `ValueError`,
  and complex, string or object arrays raise `TypeError`, instead of being
  accepted. Call `.sum_duplicates()` on scipy matrices and pass real float
  arrays (#70).
- `solver="auto"` always picks the IPM. Active-set is used only with
  `solver="active_set"`, so small dense QPs may be slower by default (#97).
- On CPU, `CompiledSolver` with `direct_solve_method="auto"` lets the Rust core
  choose the linear solver, as `Solver` does, instead of picking QDLDL when
  `n + m < 500` (#71).

### Fixed

- NaN or Inf in `P`, `A` or `q`, or NaN in `b`, raises `ValueError` instead of
  returning `Solved` with a wrong or NaN solution. CUDA tensors are not
  scanned (#70).
- `Solver` and `CompiledSolver.setup()` copy their inputs, so editing the
  caller's arrays afterwards no longer changes the next solve (#70).
- CUDA backward passes could return zero or stale gradients when cached solves
  or newly allocated workspaces crossed CUDA streams (#112).
- CPU and CUDA HSDE derivatives could be biased in weakly constrained directions by
  the regularized differentiation solve. Apply defect correction to the
  unregularized equations in forward and adjoint differentiation, reusing
  the factorization in the cached adjoint and CUDA paths. Woodbury backward
  refinement also targets the unregularized normal equations.
- Exponential-cone differentiation could incorrectly classify points as polar
  on both CPU and CUDA because the dual membership test applied the factor
  `e` twice. Correct the inequality and the resulting projection Jacobians.
- CUDA chordal solves could stall when iterative refinement increased the KKT
  residual and the default start shifted PSD cones independently of other
  cones. Keep the refinement iterate with the smallest residual and use a
  common interior shift across cones for each problem (#110).
- CPU chordal decomposition filled `A`'s values in the wrong order, so it could
  solve a different problem and report `Solved` (#79).
- CUDA PSD solves could return NaN after shifting an all-zero PSD block, from a
  stale cuSOLVER eigendecomposition workspace (#94).
- CPU active-set crashed on problems with no constraints (#27).
- The CUDA JAX `solve_warm` required `warm_z_x`, which broke cvxpylayers' JAX
  warm starts; it now defaults to zeros (#92).
- The docs gave the wrong slack row order for PSD mixed with exponential or
  power cones (#55), and several examples were wrong or didn't run (#73).
- CUDA ordered PSD sparsity graphs with its own minimum-degree search instead of
  the CPU's AMD, so the two backends could split the same sparse PSD cone into
  different cliques. CUDA now uses SuiteSparse AMD, vendored (#108).
- CUDA equilibration of problems with `P = 0` (LPs, SOCPs, SDPs) started each
  Ruiz iteration from stale column norms, so it scaled `A` differently from the
  CPU (#107).
