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

- CUDA wheels build against cuDSS 1.0.0 and require `nvidia-cudss-cu12>=1.0,<1.1`
  (or `-cu13`), which adds NVRTC as a runtime dependency. Julia (`CUDSS_jll`)
  stays on cuDSS 0.7.1 until a 1.0 build is registered (#109).
- `solver="auto"` always picks the IPM. Active-set is used only with
  `solver="active_set"`, so small dense QPs may be slower by default (#97).
- On CPU, `CompiledSolver` with `direct_solve_method="auto"` lets the Rust core
  choose the linear solver, as `Solver` does, instead of picking QDLDL when
  `n + m < 500` (#71).

### Fixed

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
- CUDA equilibration of problems with `P = 0` (LPs, SOCPs, SDPs) started each
  Ruiz iteration from stale column norms, so it scaled `A` differently from the
  CPU (#107).
