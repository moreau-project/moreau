module MoreauChainRulesExt

using Moreau
using ChainRulesCore

# ============================================================================
# Helper: extract tangent fields from a Solution Tangent
# ============================================================================

function _extract_tangent(Δ, field::Symbol, fallback_size)
    t = getproperty(Δ, field)
    if t isa AbstractArray
        return t
    else
        # ZeroTangent or NoTangent → zeros
        return zeros(Float64, fallback_size)
    end
end

# ============================================================================
# rrule for solve!(solver, q::Vector, b::Vector) → Solution
# Differentiable w.r.t. q and b only.
# ============================================================================

function ChainRulesCore.rrule(::typeof(solve!), solver::CompiledSolver,
                               q::AbstractVector{Float64}, b::AbstractVector{Float64};
                               kwargs...)
    sol = solve!(solver, q, b; kwargs...)

    function solve_pullback(Δ)
        dx = _extract_tangent(Δ, :x, solver.n)
        dz = _extract_tangent(Δ, :z, solver.m)
        ds = _extract_tangent(Δ, :s, solver.m)
        grads = backward!(solver, dx; dz=dz, ds=ds)
        return NoTangent(), NoTangent(), grads.dq, grads.db
    end

    return sol, solve_pullback
end

# ============================================================================
# rrule for solve!(solver, q::Matrix, b::Matrix) → BatchedSolution
# Differentiable w.r.t. q and b only.
# ============================================================================

function ChainRulesCore.rrule(::typeof(solve!), solver::CompiledSolver,
                               q::AbstractMatrix{Float64}, b::AbstractMatrix{Float64};
                               kwargs...)
    sol = solve!(solver, q, b; kwargs...)

    function solve_batch_pullback(Δ)
        dx = _extract_tangent(Δ, :x, (solver.n, solver.batch_size))
        dz = _extract_tangent(Δ, :z, (solver.m, solver.batch_size))
        ds = _extract_tangent(Δ, :s, (solver.m, solver.batch_size))
        grads = backward!(solver, dx; dz=dz, ds=ds)
        return NoTangent(), NoTangent(), grads.dq, grads.db
    end

    return sol, solve_batch_pullback
end

# ============================================================================
# rrule for setup_and_solve!(solver, P_values, A_values, q, b) → Solution
# Differentiable w.r.t. P_values, A_values, q, and b.
# ============================================================================

function ChainRulesCore.rrule(::typeof(setup_and_solve!), solver::CompiledSolver,
                               P_values::AbstractVector{Float64},
                               A_values::AbstractVector{Float64},
                               q::AbstractVector{Float64},
                               b::AbstractVector{Float64};
                               kwargs...)
    sol = setup_and_solve!(solver, P_values, A_values, q, b; kwargs...)

    function setup_and_solve_pullback(Δ)
        dx = _extract_tangent(Δ, :x, solver.n)
        dz = _extract_tangent(Δ, :z, solver.m)
        ds = _extract_tangent(Δ, :s, solver.m)
        grads = backward!(solver, dx; dz=dz, ds=ds)
        return NoTangent(), NoTangent(), grads.dP_values, grads.dA_values, grads.dq, grads.db
    end

    return sol, setup_and_solve_pullback
end

# ============================================================================
# rrule for batched setup_and_solve!
# ============================================================================

function ChainRulesCore.rrule(::typeof(setup_and_solve!), solver::CompiledSolver,
                               P_values::AbstractVector{Float64},
                               A_values::AbstractVector{Float64},
                               q::AbstractMatrix{Float64},
                               b::AbstractMatrix{Float64};
                               kwargs...)
    sol = setup_and_solve!(solver, P_values, A_values, q, b; kwargs...)

    function setup_and_solve_batch_pullback(Δ)
        dx = _extract_tangent(Δ, :x, (solver.n, solver.batch_size))
        dz = _extract_tangent(Δ, :z, (solver.m, solver.batch_size))
        ds = _extract_tangent(Δ, :s, (solver.m, solver.batch_size))
        grads = backward!(solver, dx; dz=dz, ds=ds)
        # P/A were shared (vector), but backward gives per-batch gradients (matrix).
        # Sum across batch dimension to get gradient for the shared values.
        dP = sum(grads.dP_values, dims=2) |> vec
        dA = sum(grads.dA_values, dims=2) |> vec
        return NoTangent(), NoTangent(), dP, dA, grads.dq, grads.db
    end

    return sol, setup_and_solve_batch_pullback
end

function ChainRulesCore.rrule(::typeof(setup_and_solve!), solver::CompiledSolver,
                               P_values::AbstractMatrix{Float64},
                               A_values::AbstractMatrix{Float64},
                               q::AbstractMatrix{Float64},
                               b::AbstractMatrix{Float64};
                               kwargs...)
    sol = setup_and_solve!(solver, P_values, A_values, q, b; kwargs...)

    function setup_and_solve_perbatch_pullback(Δ)
        dx = _extract_tangent(Δ, :x, (solver.n, solver.batch_size))
        dz = _extract_tangent(Δ, :z, (solver.m, solver.batch_size))
        ds = _extract_tangent(Δ, :s, (solver.m, solver.batch_size))
        grads = backward!(solver, dx; dz=dz, ds=ds)
        return NoTangent(), NoTangent(), grads.dP_values, grads.dA_values, grads.dq, grads.db
    end

    return sol, setup_and_solve_perbatch_pullback
end

end # module
