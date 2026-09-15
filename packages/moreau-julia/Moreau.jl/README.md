# Moreau.jl

Moreau.jl is a Julia interface to the
[Moreau](https://moreau.so) batched differentiable convex conic solver.

## Installation

```julia
import Pkg
Pkg.add("Moreau")
```

The CPU library is provided by `Moreau_CPU_jll`, built from source through
Yggdrasil. Its build matrix includes 64-bit glibc and musl Linux (x86-64 and
ARM64), macOS (Intel and Apple Silicon), and x86-64 Windows.

Moreau.jl and the native solver use the same release version. Moreau.jl 0.4.0
requires native Moreau 0.4.0. The JLL compatibility bounds pin that version;
JLL build-number suffixes such as `+0` distinguish rebuilds of the same source.
The package name is `Moreau`.

### CUDA

Install and load the optional native CUDA package to enable the GPU backend:

```julia
Pkg.add("Moreau_CUDA_jll")
using Moreau, Moreau_CUDA_jll
```

Also load `CUDA` when passing `CuArray` inputs. The CUDA JLL selects a compatible
CUDA runtime through Julia's platform augmentation and CUDA.jl preferences. CPU
users do not depend on the CUDA JLL. The CUDA recipe targets Linux x86-64 and
ARM64 with CUDA 12 and 13, and pins cuDSS 0.7.1.

For development, `MOREAU_CPU_LIB` and `MOREAU_CUDA_LIB` can select local builds
of the same version. An explicit CUDA library override also requires its
matching CUDA runtime on the library search path.

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
