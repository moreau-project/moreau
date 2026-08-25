# ============================================================================
# MathOptInterface / JuMP integration
# Ported from SCS.jl (MIT License, SCS.jl contributors)
# ============================================================================

MOI.Utilities.@product_of_sets(
    _Cones,
    MOI.Zeros,
    MOI.Nonnegatives,
    MOI.SecondOrderCone,
    MOI.ExponentialCone,
    MOI.PowerCone{Float64},
)

struct _SetConstants{T}
    b::Vector{T}
    power_coefficients::Dict{Int,T}
    _SetConstants{T}() where {T} = new{T}(T[], Dict{Int,T}())
end

function Base.empty!(x::_SetConstants)
    empty!(x.b)
    empty!(x.power_coefficients)
    return x
end

Base.resize!(x::_SetConstants, n) = resize!(x.b, n)

function MOI.Utilities.load_constants(x::_SetConstants, offset, f)
    MOI.Utilities.load_constants(x.b, offset, f)
    return
end

function MOI.Utilities.function_constants(x::_SetConstants, i)
    return MOI.Utilities.function_constants(x.b, i)
end

function MOI.Utilities.load_constants(
    x::_SetConstants{T},
    offset,
    set::MOI.PowerCone{T},
) where {T}
    x.power_coefficients[offset+1] = set.exponent
    return
end

function MOI.Utilities.set_from_constants(x::_SetConstants, S, rows)
    return MOI.Utilities.set_from_constants(x.b, S, rows)
end

function MOI.Utilities.set_from_constants(
    x::_SetConstants{T},
    ::Type{MOI.PowerCone{T}},
    rows,
) where {T}
    @assert length(rows) == 3
    return MOI.PowerCone{T}(x.power_coefficients[first(rows)])
end

function MOI.Utilities.modify_constants(x::_SetConstants, rows, value)
    return MOI.Utilities.modify_constants(x.b, rows, value)
end

const OptimizerCache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    MOI.Utilities.MatrixOfConstraints{
        Float64,
        MOI.Utilities.MutableSparseMatrixCSC{
            Float64,
            Int,
            MOI.Utilities.ZeroBasedIndexing,
        },
        _SetConstants{Float64},
        _Cones{Float64},
    },
}

# ============================================================================
# MOISolution
# ============================================================================

mutable struct MOISolution
    primal::Vector{Float64}
    dual::Vector{Float64}
    slack::Vector{Float64}
    ret_val::MoreauStatus
    objective_value::Float64
    dual_objective_value::Float64
    objective_constant::Float64
    solve_time_sec::Float64
    iterations::Int
end

function MOISolution()
    return MOISolution(
        Float64[],
        Float64[],
        Float64[],
        MOREAU_STATUS_UNSOLVED,
        NaN,
        NaN,
        NaN,
        0.0,
        0,
    )
end

# ============================================================================
# Optimizer
# ============================================================================

mutable struct Optimizer <: MOI.AbstractOptimizer
    cones::Union{Nothing,_Cones{Float64}}
    sol::MOISolution
    silent::Bool
    options::Dict{Symbol,Any}

    function Optimizer(; kwargs...)
        optimizer = new(nothing, MOISolution(), false, Dict{Symbol,Any}())
        for (key, val) in kwargs
            MOI.set(optimizer, MOI.RawOptimizerAttribute(String(key)), val)
        end
        return optimizer
    end
end

function MOI.default_cache(::Optimizer, ::Type{Float64})
    return MOI.Utilities.UniversalFallback(OptimizerCache())
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Moreau"
MOI.get(::Optimizer, ::MOI.SolverVersion) = c_moreau_version()

MOI.is_empty(optimizer::Optimizer) = optimizer.cones === nothing

function MOI.empty!(optimizer::Optimizer)
    optimizer.cones = nothing
    optimizer.sol = MOISolution()
    return
end

###
### MOI.RawOptimizerAttribute
###

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    return optimizer.options[Symbol(param.name)] = value
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    return optimizer.options[Symbol(param.name)]
end

###
### MOI.Silent
###

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

###
### MOI.TimeLimitSec
###

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(optimizer::Optimizer, ::MOI.TimeLimitSec, time_limit::Real)
    optimizer.options[:time_limit] = convert(Float64, time_limit)
    return
end

function MOI.set(optimizer::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    delete!(optimizer.options, :time_limit)
    return
end

function MOI.get(optimizer::Optimizer, ::MOI.TimeLimitSec)
    value = get(optimizer.options, :time_limit, nothing)
    return value::Union{Float64,Nothing}
end

###
### MOI.AbsoluteGapTolerance / MOI.RelativeGapTolerance
###

MOI.supports(::Optimizer, ::MOI.AbsoluteGapTolerance) = true
MOI.supports(::Optimizer, ::MOI.RelativeGapTolerance) = true

function MOI.set(optimizer::Optimizer, ::MOI.AbsoluteGapTolerance, value::Real)
    optimizer.options[:tol_gap_abs] = convert(Float64, value)
    return
end

function MOI.set(optimizer::Optimizer, ::MOI.RelativeGapTolerance, value::Real)
    optimizer.options[:tol_gap_rel] = convert(Float64, value)
    return
end

function MOI.get(optimizer::Optimizer, ::MOI.AbsoluteGapTolerance)
    return get(optimizer.options, :tol_gap_abs, 1e-8)::Float64
end

function MOI.get(optimizer::Optimizer, ::MOI.RelativeGapTolerance)
    return get(optimizer.options, :tol_gap_rel, 1e-8)::Float64
end

###
### MOI.AbstractModelAttribute
###

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
        MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}},
    },
)
    return true
end

###
### MOI.supports_constraint
###

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{
        <:Union{
            MOI.Zeros,
            MOI.Nonnegatives,
            MOI.SecondOrderCone,
            MOI.ExponentialCone,
            MOI.PowerCone{Float64},
        },
    },
)
    return true
end

# ============================================================================
# Helpers
# ============================================================================

function _map_sets(f, sets, ::Type{S}) where {S}
    F = MOI.VectorAffineFunction{Float64}
    cis = MOI.get(sets, MOI.ListOfConstraintIndices{F,S}())
    return Int[f(MOI.get(sets, MOI.ConstraintSet(), ci)) for ci in cis]
end

# ============================================================================
# optimize! — two-level dispatch (SCS.jl pattern)
# ============================================================================

function MOI.optimize!(
    dest::Optimizer,
    src::MOI.Utilities.UniversalFallback{OptimizerCache},
)
    MOI.empty!(dest)
    index_map = MOI.Utilities.identity_index_map(src)
    Ab = src.model.constraints
    A = Ab.coefficients

    for (F, S) in keys(src.constraints)
        throw(MOI.UnsupportedConstraint{F,S}())
    end

    model_attributes = MOI.get(src, MOI.ListOfModelAttributesSet())
    max_sense = false
    obj_attr = nothing
    for attr in model_attributes
        if attr == MOI.ObjectiveSense()
            max_sense = MOI.get(src, attr) == MOI.MAX_SENSE
        elseif attr == MOI.Name()
            continue
        elseif attr isa MOI.ObjectiveFunction
            obj_attr = attr
        else
            throw(MOI.UnsupportedAttribute(attr))
        end
    end

    objective_constant = 0.0
    c = zeros(A.n)
    P = spzeros(A.n, A.n)

    if obj_attr == MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()
        obj = MOI.Utilities.canonical(MOI.get(src, obj_attr))
        objective_constant = MOI.constant(obj)
        for term in obj.terms
            c[term.variable.value] += (max_sense ? -1 : 1) * term.coefficient
        end
    elseif obj_attr == MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}()
        obj = MOI.Utilities.canonical(MOI.get(src, obj_attr))
        objective_constant = MOI.constant(obj)
        scale = max_sense ? -1 : 1
        for term in obj.affine_terms
            c[term.variable.value] += scale * term.coefficient
        end
        nnz_q = length(obj.quadratic_terms)
        I = zeros(Int, nnz_q)
        J = zeros(Int, nnz_q)
        V = zeros(Float64, nnz_q)
        for (k, qterm) in enumerate(obj.quadratic_terms)
            I[k] = qterm.variable_1.value
            J[k] = qterm.variable_2.value
            V[k] = scale * qterm.coefficient
        end
        if nnz_q > 0
            P_upper = sparse(I, J, V, A.n, A.n)
            # Symmetrize: MOI gives upper triangle, Moreau needs full symmetric
            P = P_upper + P_upper' - sparse(1:A.n, 1:A.n, diag(P_upper), A.n, A.n)
        end
    elseif obj_attr !== nothing
        throw(MOI.UnsupportedAttribute(obj_attr))
    end

    # Extract cone structure
    dest.cones = deepcopy(Ab.sets)
    zero_cones = MOI.Utilities.num_rows(Ab.sets, MOI.Zeros)
    nonneg_cones = MOI.Utilities.num_rows(Ab.sets, MOI.Nonnegatives)
    soc_dims = _map_sets(MOI.dimension, Ab, MOI.SecondOrderCone)
    exp_cones_count = div(Ab.sets.num_rows[4] - Ab.sets.num_rows[3], 3)
    power_alphas = Float64[]
    F = MOI.VectorAffineFunction{Float64}
    for ci in MOI.get(Ab, MOI.ListOfConstraintIndices{F,MOI.PowerCone{Float64}}())
        push!(power_alphas, MOI.get(Ab, MOI.ConstraintSet(), ci).exponent)
    end

    # Build A as Julia SparseMatrixCSC: negate A (SCS/Moreau convention),
    # convert zero-based → one-based indexing
    A_csc = SparseMatrixCSC(
        A.m, A.n,
        A.colptr .+ 1,
        A.rowval .+ 1,
        -A.nzval,
    )
    b = Ab.constants.b

    # Build solver options
    options = copy(dest.options)
    if dest.silent
        options[:verbose] = false
    end
    if !haskey(options, :verbose)
        options[:verbose] = false
    end

    # Extract device option (default :auto)
    device = pop!(options, :device, :auto)::Symbol

    sol = moreau_solve(
        A.n, A.m, P, A_csc, c, b;
        device=device,
        zero_cones=zero_cones, nonneg_cones=nonneg_cones,
        soc_dims=soc_dims, exp_cones=exp_cones_count,
        power_alphas=power_alphas,
        options...,
    )

    # Convert solution vectors to host for MOI (may be CuVector from CUDA backend)
    x_host = Vector{Float64}(sol.x)
    z_host = Vector{Float64}(sol.z)
    s_host = Vector{Float64}(sol.s)

    # Compute objective values, handling certificate cases
    sense_flip = max_sense ? -1 : 1
    if sol.status == MOREAU_STATUS_DUAL_INFEASIBLE && !isfinite(sol.obj_val)
        # Primal ray: c is in internal min-sense (already negated for MAX).
        # c'*x gives internal obj; flip to original sense.
        obj_val = sense_flip * dot(c, x_host)
    else
        obj_val = sense_flip * sol.obj_val
    end
    if sol.status == MOREAU_STATUS_PRIMAL_INFEASIBLE && !isfinite(sol.obj_val_dual)
        # Dual ray: Moreau's convention gives b'*z < 0 for infeasible.
        # MOI expects positive DualObjectiveValue for MIN_SENSE infeasible,
        # so negate, then apply sense_flip.
        obj_val_dual = sense_flip * (-dot(b, z_host))
    else
        obj_val_dual = sense_flip * sol.obj_val_dual
    end

    dest.sol = MOISolution(
        x_host,
        z_host,
        s_host,
        sol.status,
        obj_val,
        obj_val_dual,
        objective_constant,
        sol.solve_time,
        sol.iterations,
    )

    return index_map, false
end

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    cache = MOI.Utilities.UniversalFallback(OptimizerCache())
    index_map = MOI.copy_to(cache, src)
    MOI.optimize!(dest, cache)
    return index_map, false
end

# ============================================================================
# Result queries
# ============================================================================

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    s = optimizer.sol.ret_val
    if s == MOREAU_STATUS_UNSOLVED
        return MOI.OPTIMIZE_NOT_CALLED
    elseif s == MOREAU_STATUS_SOLVED
        return MOI.OPTIMAL
    elseif s == MOREAU_STATUS_PRIMAL_INFEASIBLE
        return MOI.INFEASIBLE
    elseif s == MOREAU_STATUS_DUAL_INFEASIBLE
        return MOI.DUAL_INFEASIBLE
    elseif s == MOREAU_STATUS_ALMOST_SOLVED
        return MOI.ALMOST_OPTIMAL
    elseif s == MOREAU_STATUS_ALMOST_PRIMAL_INFEASIBLE
        return MOI.ALMOST_INFEASIBLE
    elseif s == MOREAU_STATUS_ALMOST_DUAL_INFEASIBLE
        return MOI.ALMOST_DUAL_INFEASIBLE
    elseif s == MOREAU_STATUS_MAX_ITERATIONS
        return MOI.ITERATION_LIMIT
    elseif s == MOREAU_STATUS_MAX_TIME
        return MOI.TIME_LIMIT
    elseif s == MOREAU_STATUS_NUMERICAL_ERROR
        return MOI.NUMERICAL_ERROR
    elseif s == MOREAU_STATUS_INSUFFICIENT_PROGRESS
        return MOI.SLOW_PROGRESS
    elseif s == MOREAU_STATUS_CALLBACK_TERMINATED
        return MOI.INTERRUPTED
    else
        return MOI.OTHER_ERROR
    end
end

function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    return optimizer.sol.solve_time_sec
end

function MOI.get(optimizer::Optimizer, ::MOI.BarrierIterations)
    return Int64(optimizer.sol.iterations)
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return string(optimizer.sol.ret_val)
end

function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    value = optimizer.sol.objective_value
    if !MOI.Utilities.is_ray(MOI.get(optimizer, MOI.PrimalStatus()))
        value += optimizer.sol.objective_constant
    end
    return value
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    value = optimizer.sol.dual_objective_value
    if !MOI.Utilities.is_ray(MOI.get(optimizer, MOI.DualStatus()))
        value += optimizer.sol.objective_constant
    end
    return value
end

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    s = optimizer.sol.ret_val
    if s == MOREAU_STATUS_SOLVED
        return MOI.FEASIBLE_POINT
    elseif s == MOREAU_STATUS_ALMOST_SOLVED
        return MOI.NEARLY_FEASIBLE_POINT
    elseif s == MOREAU_STATUS_PRIMAL_INFEASIBLE
        return MOI.INFEASIBLE_POINT
    elseif s == MOREAU_STATUS_ALMOST_PRIMAL_INFEASIBLE
        return MOI.NEARLY_INFEASIBLE_POINT
    elseif s == MOREAU_STATUS_DUAL_INFEASIBLE
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif s == MOREAU_STATUS_ALMOST_DUAL_INFEASIBLE
        return MOI.NEARLY_REDUCTION_CERTIFICATE
    elseif s in (MOREAU_STATUS_MAX_ITERATIONS, MOREAU_STATUS_MAX_TIME,
                 MOREAU_STATUS_NUMERICAL_ERROR, MOREAU_STATUS_INSUFFICIENT_PROGRESS)
        return MOI.NEARLY_FEASIBLE_POINT
    else
        return MOI.NO_SOLUTION
    end
end

function MOI.get(optimizer::Optimizer, attr::MOI.DualStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    s = optimizer.sol.ret_val
    if s == MOREAU_STATUS_SOLVED
        return MOI.FEASIBLE_POINT
    elseif s == MOREAU_STATUS_ALMOST_SOLVED
        return MOI.NEARLY_FEASIBLE_POINT
    elseif s == MOREAU_STATUS_PRIMAL_INFEASIBLE
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif s == MOREAU_STATUS_ALMOST_PRIMAL_INFEASIBLE
        return MOI.NEARLY_REDUCTION_CERTIFICATE
    elseif s == MOREAU_STATUS_DUAL_INFEASIBLE
        return MOI.INFEASIBLE_POINT
    elseif s == MOREAU_STATUS_ALMOST_DUAL_INFEASIBLE
        return MOI.NEARLY_INFEASIBLE_POINT
    elseif s in (MOREAU_STATUS_MAX_ITERATIONS, MOREAU_STATUS_MAX_TIME,
                 MOREAU_STATUS_NUMERICAL_ERROR, MOREAU_STATUS_INSUFFICIENT_PROGRESS)
        return MOI.NEARLY_FEASIBLE_POINT
    else
        return MOI.NO_SOLUTION
    end
end

MOI.get(optimizer::Optimizer, ::MOI.ResultCount) = optimizer.sol.ret_val == MOREAU_STATUS_UNSOLVED ? 0 : 1

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.primal[vi.value]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.slack[MOI.Utilities.rows(optimizer.cones, ci)]
end

function MOI.get(
    optimizer::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{F,S},
) where {F,S}
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.sol.dual[MOI.Utilities.rows(optimizer.cones, ci)]
end
