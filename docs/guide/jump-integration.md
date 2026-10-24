# JuMP Integration

Moreau provides a full [MathOptInterface](https://jump.dev/MathOptInterface.jl/stable/) (MOI) wrapper, so you can use Moreau as a solver backend in [JuMP](https://jump.dev/JuMP.jl/stable/) and any other MOI-compatible modeling tool.

For the low-level Julia API (batching, CUDA, differentiation), see the [Julia API guide](julia-integration.md).

## Installation

```julia
import Pkg
Pkg.add("Moreau")
```

The CPU shared library is supplied by the Yggdrasil-built `Moreau_CPU_jll`.
Moreau.jl and its native library use the same release version.

### CUDA Version Selection

The optional `Moreau_CUDA_jll` provides CUDA 12 and CUDA 13 builds for Linux.
Install and load it to enable GPU solves:

```julia
Pkg.add("Moreau_CUDA_jll")
using Moreau, Moreau_CUDA_jll
```

The JLL selects a compatible artifact using the runtime selected by CUDA.jl,
including its runtime-version preferences. CPU-only installations do not depend
on the CUDA JLL. Without the optional package, `device=:auto` uses the CPU.

---

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
println("x = ", value(x))  # ≈ 0.5
println("y = ", value(y))  # ≈ 0.5
```

---

## Supported Cones

| MOI Set | Description |
|---------|-------------|
| `MOI.Zeros` | Equality constraints ($s = 0$) |
| `MOI.Nonnegatives` | Inequality constraints ($s \ge 0$) |
| `MOI.SecondOrderCone` | $\lVert s_{1:} \rVert_2 \le s_0$ (arbitrary dimension $\ge 2$) |
| `MOI.ExponentialCone` | $y \exp(x/y) \le z,\; y > 0$ |
| `MOI.PowerCone{Float64}` | $x^a y^{1-a} \ge \lvert z \rvert,\; x,y \ge 0$ |

Supported objective types:

- `MOI.ScalarAffineFunction{Float64}` (linear)
- `MOI.ScalarQuadraticFunction{Float64}` (quadratic)

---

## Solver Options

Options can be set via JuMP's `optimizer_with_attributes` or directly through MOI:

```julia
# Via JuMP
model = Model(optimizer_with_attributes(
    Moreau.Optimizer,
    "max_iter" => 500,
    "verbose" => true,
))

# Via MOI
optimizer = Moreau.Optimizer()
MOI.set(optimizer, MOI.RawOptimizerAttribute("max_iter"), 500)
MOI.set(optimizer, MOI.RawOptimizerAttribute("verbose"), true)
```

Common options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_iter` | `Int` | 200 | Maximum IPM iterations |
| `verbose` | `Bool` | `false` | Print iteration log |
| `time_limit` | `Float64` | `Inf` | Maximum solve time (seconds) |
| `enable_grad` | `Bool` | `false` | Enable backward-pass gradient computation |
| `device` | `Symbol` | `:auto` | Device selection (`:cpu`, `:cuda`, or `:auto`) |

IPM tolerances can also be set as keyword arguments (e.g., `tol_gap_abs`, `tol_feas`). See the [Solver Settings](solver-settings.md) guide for the full list.

---

## Examples

### Linear Program

```julia
using JuMP, Moreau

model = Model(Moreau.Optimizer)
set_silent(model)

@variable(model, x[1:3] >= 0)
@constraint(model, x[1] + 2x[2] + 3x[3] <= 10)
@constraint(model, x[1] + x[2] >= 1)
@objective(model, Max, 5x[1] + 4x[2] + 3x[3])

optimize!(model)
println("Optimal value: ", objective_value(model))
println("x = ", value.(x))
```

### Second-Order Cone Program

```julia
using JuMP, Moreau

model = Model(Moreau.Optimizer)
set_silent(model)

@variable(model, t)
@variable(model, x[1:3])
@constraint(model, sum(x) == 1)
@constraint(model, [t; x] in SecondOrderCone())
@objective(model, Min, t)

optimize!(model)
println("Min norm: ", value(t))
println("x = ", value.(x))
```

### Quadratic Program

```julia
using JuMP, Moreau

model = Model(Moreau.Optimizer)
set_silent(model)

@variable(model, 0 <= x[1:2] <= 1)
@constraint(model, x[1] + x[2] == 1)
@objective(model, Min, 2x[1]^2 + x[2]^2 + x[1]*x[2] + x[1] + x[2])

optimize!(model)
println("x = ", value.(x))
```
