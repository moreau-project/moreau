# JuMP Integration

Moreau provides a full [MathOptInterface](https://jump.dev/MathOptInterface.jl/stable/) (MOI) wrapper, so you can use Moreau as a solver backend in [JuMP](https://jump.dev/JuMP.jl/stable/) and any other MOI-compatible modeling tool.

For the low-level Julia API (batching, CUDA, differentiation), see the [Julia API guide](julia-integration.md).

## Installation

```julia
import Pkg
Pkg.add(url="https://github.com/optimalintellect/Moreau.jl")
```

The package depends on a compiled C shared library (`libmoreau_cpu`). On first use, it will be downloaded automatically from the artifact server.

### CUDA Version Selection

Moreau is available as both a CUDA 12 and CUDA 13 build. By default, `Moreau_CUDA_jll` auto-detects the driver's maximum supported CUDA version via `nvidia-smi` and picks the best matching library (preferring CUDA 13 when the driver supports it, using only CUDA 12 on older drivers).

To override the auto-detection, set the `MOREAU_CUDA_VERSION` environment variable before loading the package:

```julia
# Force CUDA 12 even on a CUDA 13-capable driver
ENV["MOREAU_CUDA_VERSION"] = "12"
using Moreau
```

Valid values are `"12"`, `"13"`, `"12.2"`, `"13.0"`, etc. (only the major version matters).

If `nvidia-smi` is not available (e.g. in a container without it on PATH), the fallback is to try CUDA 13 first, then CUDA 12.

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
