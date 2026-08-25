# Moreau.jl

Moreau.jl is a Julia interface to the
[Moreau](https://moreau.so) batched differentiable convex conic solver.

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

Apache 2.0. See the root `LICENSE` file and `NOTICE` for upstream
attribution (Clarabel.rs, diffqcp, DAQP).
