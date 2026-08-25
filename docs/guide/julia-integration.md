# Julia API

Moreau.jl provides a low-level Julia API for direct access to the solver, bypassing JuMP/MOI. This is useful for batched solving, CUDA acceleration, warm starting, and differentiating through the solve.

For JuMP integration, see the [JuMP guide](jump-integration.md).

## One-Shot Solve

`moreau_solve` creates a solver, solves, and destroys it in one call. Use this for one-off problems.

```julia
using Moreau, SparseArrays

n, m = 2, 3
P = sparse([1, 2], [1, 2], [1.0, 1.0], n, n)
A = sparse([1, 1, 2, 3], [1, 2, 1, 2], [1.0, 1.0, 1.0, 1.0], m, n)
q = [2.0, 1.0]
b = [1.0, 0.7, 0.7]

sol = moreau_solve(
    n, m, P, A, q, b;
    zero_cones=1, nonneg_cones=2,
)
println("Status: ", sol.status)
println("x = ", sol.x)
println("Objective: ", sol.obj_val)
```

### Keyword Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `device` | `Symbol` | `:auto` | Device (`:cpu`, `:cuda`, or `:auto`) |
| `zero_cones` | `Int` | 0 | Number of zero-cone (equality) constraints |
| `nonneg_cones` | `Int` | 0 | Number of nonnegative-cone (inequality) constraints |
| `soc_dims` | `Vector{Int}` | `[]` | Dimension of each SOC cone (each $\ge 2$) |
| `exp_cones` | `Int` | 0 | Number of exponential cones |
| `power_alphas` | `Vector{Float64}` | `[]` | Power cone exponents (each in $(0, 1)$) |
| `warm_x`, `warm_z`, `warm_s` | `AbstractVector{Float64}` or `nothing` | `nothing` | Warm-start vectors |
| `verbose` | `Bool` | `false` | Print iteration log |
| `max_iter` | `Int` | 200 | Maximum IPM iterations |
| `time_limit` | `Float64` | `Inf` | Maximum solve time (seconds) |
| `enable_grad` | `Bool` | `false` | Enable gradient computation |

---

## CompiledSolver

For repeated solves with the same sparsity structure, `CompiledSolver` amortizes the setup cost. The sparsity pattern is fixed at construction; only numerical values change between solves.

```julia
using Moreau, SparseArrays

n, m = 2, 3
P = sparse([1, 2], [1, 2], [1.0, 1.0], n, n)
A = sparse([1, 1, 2, 3], [1, 2, 1, 2], [1.0, 1.0, 1.0, 1.0], m, n)

solver = CompiledSolver(n, m, P, A;
    zero_cones=1, nonneg_cones=2,
    device=:cpu,
)

# Set numerical values
setup!(solver, [1.0, 1.0], [1.0, 1.0, 1.0, 1.0])

# Solve
sol = solve!(solver, [2.0, 1.0], [1.0, 0.7, 0.7])
println("x = ", sol.x)

# Re-solve with different q, b (P/A unchanged)
sol2 = solve!(solver, [1.0, 2.0], [1.0, 0.7, 0.7])

# Warm start from previous solution
sol3 = solve!(solver, [1.5, 1.5], [1.0, 0.7, 0.7]; warm_start=sol2)

destroy!(solver)  # or let GC handle it
```

### Batched Solves

Solve multiple problems with the same structure in one call. Pass `batch_size` at construction, then use matrices `(dim, batch_size)` for `q`, `b`, and solutions.

```julia
batch_size = 4
solver = CompiledSolver(n, m, P, A;
    zero_cones=1, nonneg_cones=2,
    batch_size=batch_size,
)

# Shared P/A across batch (vector), or per-batch (matrix of size (nnz, batch_size))
setup!(solver, [1.0, 1.0], [1.0, 1.0, 1.0, 1.0])

# q and b are matrices: (n, batch_size) and (m, batch_size)
q_batch = randn(n, batch_size)
b_batch = [1.0 1.0 1.0 1.0; 0.7 0.7 0.7 0.7; 0.7 0.7 0.7 0.7]

bsol = solve!(solver, q_batch, b_batch)
println("Batch size: ", length(bsol))

# Index individual solutions
for i in eachindex(bsol)
    println("  Problem $i: status=$(bsol[i].status), x=$(bsol[i].x)")
end

# Warm start from previous batch
bsol2 = solve!(solver, q_batch, b_batch; warm_start=bsol)

destroy!(solver)
```

---

## CUDA Support

Moreau automatically selects the best device when `device=:auto` (default). You can also force a specific device:

```julia
# Force CUDA
solver = CompiledSolver(n, m, P, A;
    zero_cones=1, nonneg_cones=2,
    device=:cuda,
)

# Or use the one-shot API
sol = moreau_solve(n, m, P, A, q, b;
    zero_cones=1, nonneg_cones=2,
    device=:cuda,
)

# Set a global default
Moreau.set_default_device(:cuda)
```

When `CUDA.jl` is loaded, `CuVector{Float64}` inputs are passed to the solver without host-to-device copies, and solutions are returned as `CuVector{Float64}`:

```julia
using CUDA
q_gpu = CuVector([2.0, 1.0])
b_gpu = CuVector([1.0, 0.7, 0.7])
sol = moreau_solve(n, m, P, A, q_gpu, b_gpu;
    zero_cones=1, nonneg_cones=2, device=:cuda,
)
typeof(sol.x)  # CuVector{Float64}
```

Check availability:

```julia
Moreau.cuda_available()  # true if CUDA backend is loaded
```

For CUDA version selection (CUDA 12 vs 13), see the [installation section](jump-integration.md#cuda-version-selection).

---

## Gradients and Differentiation

### Explicit backward pass

Enable gradient computation and call `backward!` after a solve:

```julia
solver = CompiledSolver(n, m, P, A;
    zero_cones=1, nonneg_cones=2,
    enable_grad=true,
)
setup!(solver, P_values, A_values)
sol = solve!(solver, q, b)

# Compute gradients: dx is the upstream gradient of the loss w.r.t. x
grads = backward!(solver, dx)
# grads.dq, grads.db      — gradients w.r.t. q and b
# grads.dP_values, grads.dA_values  — gradients w.r.t. P and A values

# Optionally pass upstream gradients for z and s
grads = backward!(solver, dx; dz=dz, ds=ds)
```

### ChainRules (automatic differentiation)

When `ChainRulesCore` is loaded, Moreau defines `rrule` methods for AD frameworks (Zygote, Enzyme, etc.):

```julia
using ChainRulesCore

# solve! is differentiable w.r.t. q and b
sol, pullback = rrule(solve!, solver, q, b)
```

To differentiate w.r.t. `P_values` and `A_values` as well, use `setup_and_solve!`, which combines `setup!` and `solve!` into a single differentiable operation:

```julia
# setup_and_solve! is differentiable w.r.t. P_values, A_values, q, and b
sol, pullback = rrule(setup_and_solve!, solver, P_values, A_values, q, b)
```

Both single-problem (vector) and batched (matrix) variants are supported.

---

## Solution Types

### `Solution{V}`

Returned by `solve!` with vector inputs. `V` is the vector type (`Vector{Float64}` or `CuVector{Float64}`).

| Field | Type | Description |
|-------|------|-------------|
| `x` | `V` | Primal solution (length $n$) |
| `z` | `V` | Dual solution (length $m$) |
| `s` | `V` | Slack variables (length $m$) |
| `status` | `MoreauStatus` | Solver termination status |
| `obj_val` | `Float64` | Primal objective value |
| `obj_val_dual` | `Float64` | Dual objective value |
| `solve_time` | `Float64` | Solve time in seconds |
| `iterations` | `Int` | Number of IPM iterations |
| `r_prim` | `Float64` | Primal residual |
| `r_dual` | `Float64` | Dual residual |

### `BatchedSolution{M}`

Returned by `solve!` with matrix inputs. Columns correspond to batch elements.

| Field | Type | Description |
|-------|------|-------------|
| `x` | `M` | Primal solutions ($n \times$ batch_size) |
| `z` | `M` | Dual solutions ($m \times$ batch_size) |
| `s` | `M` | Slack variables ($m \times$ batch_size) |
| `status` | `Vector{MoreauStatus}` | Per-element status |
| `obj_val` | `Vector{Float64}` | Per-element primal objective |
| `obj_val_dual` | `Vector{Float64}` | Per-element dual objective |
| `solve_time` | `Float64` | Total solve time |
| `iterations` | `Int` | Maximum iterations across batch |
| `r_prim` | `Vector{Float64}` | Per-element primal residual |
| `r_dual` | `Vector{Float64}` | Per-element dual residual |

Index with `bsol[i]` to get a `Solution` for element `i`. Use `length(bsol)` and `eachindex(bsol)` for iteration.
