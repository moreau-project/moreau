# Moreau.jl

Moreau.jl is a Julia interface to the
[Moreau](https://moreau.so) batched differentiable convex conic solver.

## Installation

```julia
import Pkg
Pkg.add(url="https://github.com/moreau-project/Moreau.jl")
```

The CPU library is installed automatically. CUDA 12 and CUDA 13 binaries are
lazy artifacts: they are downloaded only when a CUDA solve is requested, not
when Moreau.jl is installed or imported.

Moreau.jl releases are maintained independently from the native solver. After
a new native release, download its C-library archives and regenerate the
manifest before bumping the Julia package version:

```bash
julia scripts/generate_artifacts.jl v0.3.4 /path/to/release-assets
```

## Quick Start

```julia
using JuMP, Moreau
model = Model(Moreau.Optimizer)
set_silent(model)
@variable(model, x >= 0)
@variable(model, y >= 0)
@constraint(model, x + y == 1)
@objective(model, Min, x^2 + y^2)
optimize!(model)
value(x)  # ≈ 0.5
value(y)  # ≈ 0.5
```

## Documentation

- [JuMP integration](https://docs.moreau.so/guide/jump-integration.html) — MOI wrapper, supported cones, solver options, examples
- [Julia API](https://docs.moreau.so/guide/julia-integration.html) — `CompiledSolver`, batching, CUDA, gradients, ChainRules

## License

Apache 2.0. See `LICENSE` and `NOTICE` for upstream attribution
(Clarabel.rs, diffqcp, DAQP).
