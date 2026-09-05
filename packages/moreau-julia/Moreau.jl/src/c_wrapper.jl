# ============================================================================
# Enums
# ============================================================================

@enum MoreauError::Int32 begin
    MOREAU_OK = 0
    MOREAU_ERROR_INVALID_ARGUMENT = 1
    MOREAU_ERROR_NOT_SETUP = 2
    MOREAU_ERROR_NUMERICAL = 4
    MOREAU_ERROR_OUT_OF_MEMORY = 5
    MOREAU_ERROR_CUDA = 6
    MOREAU_ERROR_INTERNAL = 99
end

@enum MoreauStatus::Int32 begin
    MOREAU_STATUS_UNSOLVED = 0
    MOREAU_STATUS_SOLVED = 1
    MOREAU_STATUS_PRIMAL_INFEASIBLE = 2
    MOREAU_STATUS_DUAL_INFEASIBLE = 3
    MOREAU_STATUS_ALMOST_SOLVED = 4
    MOREAU_STATUS_ALMOST_PRIMAL_INFEASIBLE = 5
    MOREAU_STATUS_ALMOST_DUAL_INFEASIBLE = 6
    MOREAU_STATUS_MAX_ITERATIONS = 7
    MOREAU_STATUS_MAX_TIME = 8
    MOREAU_STATUS_NUMERICAL_ERROR = 9
    MOREAU_STATUS_INSUFFICIENT_PROGRESS = 10
    MOREAU_STATUS_CALLBACK_TERMINATED = 11
end

@enum MoreauDirectSolveMethod::Int32 begin
    MOREAU_DIRECT_SOLVE_AUTO = 0
    MOREAU_DIRECT_SOLVE_QDLDL = 1
    MOREAU_DIRECT_SOLVE_FAER = 2
    MOREAU_DIRECT_SOLVE_CUDSS = 3
end

# ============================================================================
# C-compatible structs (immutable so they are stored inline)
#
# IMPORTANT: These must be immutable structs so Julia stores them inline
# (value-type), matching C struct layout. Mutable structs are heap-allocated
# and stored as pointers when nested.
# ============================================================================

struct CMoreauIpmSettings
    tol_gap_abs::Float64
    tol_gap_rel::Float64
    tol_feas::Float64
    tol_infeas_abs::Float64
    tol_infeas_rel::Float64
    tol_ktratio::Float64
    reduced_tol_gap_abs::Float64
    reduced_tol_gap_rel::Float64
    reduced_tol_feas::Float64
    reduced_tol_infeas_abs::Float64
    reduced_tol_infeas_rel::Float64
    reduced_tol_ktratio::Float64
    equilibrate_enable::Int32
    equilibrate_max_iter::Int32
    equilibrate_min_scaling::Float64
    equilibrate_max_scaling::Float64
    max_step_fraction::Float64
    linesearch_backtrack_step::Float64
    min_switch_step_length::Float64
    min_terminate_step_length::Float64
    direct_solve_method::Int32
    static_regularization_enable::Int32
    static_regularization_constant::Float64
    static_regularization_proportional::Float64
    dynamic_regularization_enable::Int32
    dynamic_regularization_eps::Float64
    dynamic_regularization_delta::Float64
end

struct CMoreauSettings
    batch_size::Int64
    max_iter::UInt32
    time_limit::Float64
    verbose::Int32
    enable_grad::Int32
    ipm::CMoreauIpmSettings
end

struct CMoreauCones
    num_zero_cones::Int64
    num_nonneg_cones::Int64
    num_soc_cones::Int64
    soc_dims::Ptr{Int64}
    num_exp_cones::Int64
    num_power_cones::Int64
    power_alphas::Ptr{Float64}
end

struct CMoreauSolution
    x::Ptr{Float64}
    z::Ptr{Float64}
    s::Ptr{Float64}
    status::Int32
    obj_val::Float64
    obj_val_dual::Float64
    solve_time::Float64
    iterations::Int32
    r_prim::Float64
    r_dual::Float64
end

# ============================================================================
# Library handle + dlsym cache
# ============================================================================

const _sym_cache = Dict{Tuple{String,Symbol}, Ptr{Cvoid}}()

"""
    _cfunc(lib, name) -> Ptr{Cvoid}

Get a C function pointer by name from the given library path.
Results are cached to avoid repeated dlsym lookups.
"""
function _cfunc(lib::String, name::Symbol)
    key = (lib, name)
    ptr = get(_sym_cache, key, C_NULL)
    if ptr != C_NULL
        return ptr
    end
    handle = Libdl.dlopen(lib)
    ptr = Libdl.dlsym(handle, name)
    _sym_cache[key] = ptr
    return ptr
end

# ============================================================================
# ccall wrappers — parameterized by library path
#
# All functions accept a `lib` keyword (default: CPU library) so the same
# wrappers work for both CPU and CUDA backends. We use dlsym-based ccall
# since Julia's ccall requires compile-time constant library names.
# ============================================================================

function c_moreau_version(; lib::String=libmoreau)
    ptr = ccall(_cfunc(lib, :moreau_version), Cstring, ())
    return unsafe_string(ptr)
end

function c_moreau_last_error(; lib::String=libmoreau)
    ptr = ccall(_cfunc(lib, :moreau_last_error), Cstring, ())
    return ptr == C_NULL ? nothing : unsafe_string(ptr)
end

function c_moreau_settings_default(; lib::String=libmoreau)
    ref = Ref{CMoreauSettings}()
    ccall(_cfunc(lib, :moreau_settings_default), Cvoid, (Ptr{CMoreauSettings},), ref)
    return ref[]
end

function c_moreau_solver_create(
    n::Int64, m::Int64,
    P_row_offsets::Vector{Int64}, P_col_indices::Vector{Int64}, nnz_P::Int64,
    A_row_offsets::Vector{Int64}, A_col_indices::Vector{Int64}, nnz_A::Int64,
    cones::CMoreauCones,
    settings::CMoreauSettings;
    lib::String=libmoreau,
)
    solver_ptr = Ref{Ptr{Cvoid}}(C_NULL)
    cones_ref = Ref(cones)
    settings_ref = Ref(settings)
    err = ccall(
        _cfunc(lib, :moreau_solver_create), Int32,
        (Ptr{Ptr{Cvoid}}, Int64, Int64,
         Ptr{Int64}, Ptr{Int64}, Int64,
         Ptr{Int64}, Ptr{Int64}, Int64,
         Ptr{CMoreauCones}, Ptr{CMoreauSettings}),
        solver_ptr, n, m,
        P_row_offsets, P_col_indices, nnz_P,
        A_row_offsets, A_col_indices, nnz_A,
        cones_ref, settings_ref,
    )
    _check_error(MoreauError(err), "moreau_solver_create"; lib=lib)
    return solver_ptr[]
end

# --- Ptr{Cvoid} overloads for the one-shot moreau_solve path ---
# These are safe because the raw pointer has no finalizer and is used
# inside GC.@preserve blocks.

function c_moreau_solver_setup(solver::Ptr{Cvoid}, P_values::Vector{Float64}, A_values::Vector{Float64};
                               lib::String=libmoreau)
    err = ccall(
        _cfunc(lib, :moreau_solver_setup), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Int64, Ptr{Float64}, Int64),
        solver, P_values, length(P_values), A_values, length(A_values),
    )
    _check_error(MoreauError(err), "moreau_solver_setup"; lib=lib)
end

function c_moreau_solver_setup(solver::Ptr{Cvoid}, P_values::Ptr{Float64}, nnz_P::Int64,
                               A_values::Ptr{Float64}, nnz_A::Int64;
                               lib::String=libmoreau)
    err = ccall(
        _cfunc(lib, :moreau_solver_setup), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Int64, Ptr{Float64}, Int64),
        solver, P_values, nnz_P, A_values, nnz_A,
    )
    _check_error(MoreauError(err), "moreau_solver_setup"; lib=lib)
end

function c_moreau_solver_solve(solver::Ptr{Cvoid}, q::Vector{Float64}, b::Vector{Float64};
                               lib::String=libmoreau)
    err = ccall(
        _cfunc(lib, :moreau_solver_solve), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}),
        solver, q, b,
    )
    _check_error(MoreauError(err), "moreau_solver_solve"; lib=lib)
end

function c_moreau_solver_solve(solver::Ptr{Cvoid}, q::Ptr{Float64}, b::Ptr{Float64};
                               lib::String=libmoreau)
    err = ccall(
        _cfunc(lib, :moreau_solver_solve), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}),
        solver, q, b,
    )
    _check_error(MoreauError(err), "moreau_solver_solve"; lib=lib)
end

function c_moreau_solver_solve_warm(
    solver::Ptr{Cvoid},
    q::Vector{Float64}, b::Vector{Float64},
    warm_x::Vector{Float64}, warm_z::Vector{Float64}, warm_s::Vector{Float64};
    lib::String=libmoreau,
)
    err = ccall(
        _cfunc(lib, :moreau_solver_solve_warm), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        solver, q, b, warm_x, warm_z, warm_s,
    )
    _check_error(MoreauError(err), "moreau_solver_solve_warm"; lib=lib)
end

function c_moreau_solver_solve_warm(
    solver::Ptr{Cvoid},
    q::Ptr{Float64}, b::Ptr{Float64},
    warm_x::Ptr{Float64}, warm_z::Ptr{Float64}, warm_s::Ptr{Float64};
    lib::String=libmoreau,
)
    err = ccall(
        _cfunc(lib, :moreau_solver_solve_warm), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        solver, q, b, warm_x, warm_z, warm_s,
    )
    _check_error(MoreauError(err), "moreau_solver_solve_warm"; lib=lib)
end

function c_moreau_solver_get_solution(solver::Ptr{Cvoid}, batch_idx::Int64;
                                      lib::String=libmoreau)
    sol = Ref{CMoreauSolution}()
    err = ccall(
        _cfunc(lib, :moreau_solver_get_solution), Int32,
        (Ptr{Cvoid}, Int64, Ptr{CMoreauSolution}),
        solver, batch_idx, sol,
    )
    _check_error(MoreauError(err), "moreau_solver_get_solution"; lib=lib)
    return sol[]
end

function c_moreau_solver_copy_solution(solver::Ptr{Cvoid}, batch_idx::Int64, n::Int, m::Int;
                                       lib::String=libmoreau)
    x = Vector{Float64}(undef, n)
    z = Vector{Float64}(undef, m)
    s = Vector{Float64}(undef, m)
    err = ccall(
        _cfunc(lib, :moreau_solver_copy_solution), Int32,
        (Ptr{Cvoid}, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        solver, batch_idx, x, z, s,
    )
    _check_error(MoreauError(err), "moreau_solver_copy_solution"; lib=lib)
    return x, z, s
end

function c_moreau_solver_destroy(solver::Ptr{Cvoid}; lib::String=libmoreau)
    ccall(_cfunc(lib, :moreau_solver_destroy), Cvoid, (Ptr{Cvoid},), solver)
end

function c_moreau_solver_get_dims(solver::Ptr{Cvoid}; lib::String=libmoreau)
    n = Ref{Int64}(0)
    m = Ref{Int64}(0)
    batch_size = Ref{Int64}(0)
    nnz_P = Ref{Int64}(0)
    nnz_A = Ref{Int64}(0)
    err = ccall(
        _cfunc(lib, :moreau_solver_get_dims), Int32,
        (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}),
        solver, n, m, batch_size, nnz_P, nnz_A,
    )
    _check_error(MoreauError(err), "moreau_solver_get_dims"; lib=lib)
    return n[], m[], batch_size[], nnz_P[], nnz_A[]
end

# ============================================================================
# Error checking helper
# ============================================================================

function _check_error(err::MoreauError, func_name::String; lib::String=libmoreau)
    err == MOREAU_OK && return
    msg = c_moreau_last_error(; lib=lib)
    error_str = msg === nothing ? "unknown error" : msg
    throw(ErrorException("$func_name failed with $err: $error_str"))
end

# ============================================================================
# High-level Solution type
# ============================================================================

struct Solution{V<:AbstractVector{Float64}}
    x::V
    z::V
    s::V
    status::MoreauStatus
    obj_val::Float64
    obj_val_dual::Float64
    solve_time::Float64
    iterations::Int
    r_prim::Float64
    r_dual::Float64
end

"""
    BatchedSolution{M}

Solution for a batch of problems. Fields `x`, `z`, `s` are matrices where
column `i` is the solution for batch element `i`. Indexing `sol[i]` returns
a `Solution` for that element.
"""
struct BatchedSolution{M<:AbstractMatrix{Float64}}
    x::M                        # (n, batch_size)
    z::M                        # (m, batch_size)
    s::M                        # (m, batch_size)
    status::Vector{MoreauStatus}
    obj_val::Vector{Float64}
    obj_val_dual::Vector{Float64}
    solve_time::Float64
    iterations::Int
    r_prim::Vector{Float64}
    r_dual::Vector{Float64}
end

function Base.getindex(sol::BatchedSolution, i::Int)
    return Solution(
        sol.x[:, i], sol.z[:, i], sol.s[:, i],
        sol.status[i], sol.obj_val[i], sol.obj_val_dual[i],
        sol.solve_time, sol.iterations, sol.r_prim[i], sol.r_dual[i],
    )
end

Base.length(sol::BatchedSolution) = size(sol.x, 2)
Base.eachindex(sol::BatchedSolution) = 1:length(sol)

# ============================================================================
# CSC → CSR conversion helper
# ============================================================================

"""
    _csc_to_csr(M::SparseMatrixCSC) -> (row_offsets, col_indices, values)

Convert a Julia CSC sparse matrix to CSR format (0-indexed).
Uses the identity: CSR(M) = CSC(M').
"""
function _csc_to_csr(M::SparseMatrixCSC{Float64,Int})
    Mt = sparse(M')
    row_offsets = Vector{Int64}(Mt.colptr .- 1)
    col_indices = Vector{Int64}(Mt.rowval .- 1)
    values = Vector{Float64}(Mt.nzval)
    return row_offsets, col_indices, values
end

# ============================================================================
# Settings builder — creates CMoreauSettings with defaults + overrides
# ============================================================================

function _build_settings(;
    verbose::Bool=false,
    max_iter::Int=200,
    time_limit::Float64=Inf,
    enable_grad::Bool=false,
    lib::String=libmoreau,
    options...,
)
    # Get defaults from C
    defaults = c_moreau_settings_default(; lib=lib)
    ipm = defaults.ipm

    # Apply IPM option overrides
    ipm_fields = Dict{Symbol,Any}()
    for fname in fieldnames(CMoreauIpmSettings)
        ipm_fields[fname] = getfield(ipm, fname)
    end
    for (key, val) in options
        if haskey(ipm_fields, key)
            ftype = fieldtype(CMoreauIpmSettings, key)
            ipm_fields[key] = convert(ftype, val)
        end
    end

    new_ipm = CMoreauIpmSettings(
        (ipm_fields[fname] for fname in fieldnames(CMoreauIpmSettings))...
    )

    batch_sz = get(options, :batch_size, 1)
    return CMoreauSettings(
        Int64(batch_sz),
        UInt32(max_iter),
        time_limit,
        Int32(verbose ? 1 : 0),
        Int32(enable_grad ? 1 : 0),
        new_ipm,
    )
end

# ============================================================================
# Device selection
# ============================================================================

const _default_device = Ref{Union{Symbol,Nothing}}(nothing)

"""
    set_default_device(dev::Symbol)

Set the default device for `moreau_solve`. Valid values: `:cpu`, `:cuda`, `:auto`.
Pass `nothing` to reset to `:auto`.
"""
set_default_device(dev::Symbol) = (_default_device[] = dev)
set_default_device(::Nothing) = (_default_device[] = nothing)

"""
    _choose_device(n::Int, nnz_A::Int, batch_size::Int=1) -> Symbol

Heuristic device selection mirroring the Python `_choose_device()`.
"""
function _choose_device(n::Int, nnz_A::Int, batch_size::Int=1)
    # Check module-level override first
    if _default_device[] !== nothing
        dev = _default_device[]
        dev == :auto || return dev
    end
    # Small problems: CPU wins due to CUDA overhead (~25ms cuDSS init)
    n < 500 && return :cpu
    # Decide whether the problem benefits from CUDA before probing or
    # downloading the lazy CUDA artifact.
    use_cuda = n >= 750 || nnz_A >= 50_000 || (batch_size >= 2 && nnz_A >= 25_000)
    return use_cuda && cuda_available() ? :cuda : :cpu
end

function _get_lib(device::Symbol)
    if device == :cpu
        return libmoreau
    elseif device == :cuda
        if !_load_cuda_library(; allow_fallback=true)
            error("CUDA device requested but Moreau CUDA library is not available. " *
                  "Set MOREAU_CUDA_LIB or install the CUDA artifact.")
        end
        return libmoreau_cuda
    else
        error("Unknown device: $device. Use :cpu, :cuda, or :auto.")
    end
end

# ============================================================================
# High-level solve function
# ============================================================================

"""
    moreau_solve(n, m, P, A, q, b; kwargs...) -> Solution

Solve a conic QP using the Moreau solver.

# Problem formulation
    minimize    (1/2)x'Px + q'x
    subject to  Ax + s = b
                x ∈ K1,  s ∈ K2

K2 constrains the slack s; K1 constrains x directly (direct-x cones).

# Arguments
- `n::Int`: Number of primal variables
- `m::Int`: Number of constraints
- `P::SparseMatrixCSC{Float64}`: Quadratic cost matrix (must be full symmetric)
- `A::SparseMatrixCSC{Float64}`: Constraint matrix
- `q::Vector{Float64}`: Linear cost vector
- `b::Vector{Float64}`: Constraint RHS

# Keyword Arguments
- `device::Symbol=:auto`: Device to use (`:cpu`, `:cuda`, or `:auto`)
- `zero_cones::Int=0`: Number of zero-cone (equality) constraints
- `nonneg_cones::Int=0`: Number of nonneg-cone (inequality) constraints
- `soc_dims::Vector{Int}=Int[]`: Dimension of each SOC cone (each >= 2)
- `exp_cones::Int=0`: Number of exponential cones
- `power_alphas::Vector{Float64}=Float64[]`: Power cone alphas (each in (0,1))
- `warm_x`, `warm_z`, `warm_s`: Optional warm-start vectors
- `verbose::Bool=false`: Print iteration log
- `max_iter::Int=200`: Maximum IPM iterations
- `time_limit::Float64=Inf`: Maximum solve time in seconds
- `enable_grad::Bool=false`: Enable backward-pass gradient computation
- Additional IPM settings via keyword arguments matching field names
"""
function moreau_solve(
    n::Int, m::Int,
    P::SparseMatrixCSC{Float64,Int},
    A::SparseMatrixCSC{Float64,Int},
    q::AbstractVector{Float64}, b::AbstractVector{Float64};
    device::Symbol=:auto,
    zero_cones::Int=0, nonneg_cones::Int=0,
    soc_dims::Vector{Int}=Int[],
    exp_cones::Int=0, power_alphas::Vector{Float64}=Float64[],
    warm_x::Union{AbstractVector{Float64},Nothing}=nothing,
    warm_z::Union{AbstractVector{Float64},Nothing}=nothing,
    warm_s::Union{AbstractVector{Float64},Nothing}=nothing,
    verbose::Bool=false,
    max_iter::Int=200,
    time_limit::Float64=Inf,
    enable_grad::Bool=false,
    options...,
)
    # Validate vector types are consistent
    vecs = [("q", q), ("b", b)]
    warm_x !== nothing && push!(vecs, ("warm_x", warm_x))
    warm_z !== nothing && push!(vecs, ("warm_z", warm_z))
    warm_s !== nothing && push!(vecs, ("warm_s", warm_s))
    types = unique(typeof(v) for (_, v) in vecs)
    if length(types) > 1
        details = join(["$name::$(typeof(v))" for (name, v) in vecs], ", ")
        error("All vector arguments must have the same type, got: $details")
    end

    # Convert to CSR (0-indexed)
    P_ro, P_ci, P_vals = _csc_to_csr(P)
    A_ro, A_ci, A_vals = _csc_to_csr(A)

    nnz_P = Int64(length(P_vals))
    nnz_A = Int64(length(A_vals))

    # Resolve device
    resolved_device = device == :auto ? _choose_device(n, nnz_A) : device
    lib = _get_lib(resolved_device)

    # Build cones struct
    soc_dims_i64 = Vector{Int64}(soc_dims)
    cones = CMoreauCones(
        Int64(zero_cones),
        Int64(nonneg_cones),
        Int64(length(soc_dims)),
        length(soc_dims) > 0 ? pointer(soc_dims_i64) : C_NULL,
        Int64(exp_cones),
        Int64(length(power_alphas)),
        length(power_alphas) > 0 ? pointer(power_alphas) : C_NULL,
    )

    # Build settings
    settings = _build_settings(;
        verbose=verbose, max_iter=max_iter,
        time_limit=time_limit, enable_grad=enable_grad,
        lib=lib, options...,
    )

    if resolved_device == :cuda
        return _moreau_solve_cuda(
            n, m, P_ro, P_ci, P_vals, A_ro, A_ci, A_vals,
            nnz_P, nnz_A, q, b, cones, settings, lib,
            soc_dims_i64, power_alphas,
            warm_x, warm_z, warm_s,
        )
    else
        return _moreau_solve_cpu(
            n, m, P_ro, P_ci, P_vals, A_ro, A_ci, A_vals,
            nnz_P, nnz_A, q, b, cones, settings, lib,
            soc_dims_i64, power_alphas,
            warm_x, warm_z, warm_s,
        )
    end
end

function _moreau_solve_cpu(
    n, m, P_ro, P_ci, P_vals, A_ro, A_ci, A_vals,
    nnz_P, nnz_A, q, b, cones, settings, lib,
    soc_dims_i64, power_alphas,
    warm_x, warm_z, warm_s,
)
    solver = C_NULL
    try
        GC.@preserve P_ro P_ci P_vals A_ro A_ci A_vals q b soc_dims_i64 power_alphas begin
            solver = c_moreau_solver_create(
                Int64(n), Int64(m),
                P_ro, P_ci, nnz_P,
                A_ro, A_ci, nnz_A,
                cones, settings; lib=lib,
            )

            c_moreau_solver_setup(solver, P_vals, A_vals; lib=lib)

            if warm_x !== nothing && warm_z !== nothing && warm_s !== nothing
                c_moreau_solver_solve_warm(solver, q, b, warm_x, warm_z, warm_s; lib=lib)
            else
                c_moreau_solver_solve(solver, q, b; lib=lib)
            end

            x, z, s = c_moreau_solver_copy_solution(solver, Int64(0), n, m; lib=lib)
            sol_t = c_moreau_solver_get_solution(solver, Int64(0); lib=lib)

            return Solution(
                x, z, s,
                MoreauStatus(sol_t.status),
                sol_t.obj_val,
                sol_t.obj_val_dual,
                sol_t.solve_time,
                Int(sol_t.iterations),
                sol_t.r_prim,
                sol_t.r_dual,
            )
        end
    finally
        if solver != C_NULL
            c_moreau_solver_destroy(solver; lib=lib)
        end
    end
end

function _moreau_solve_cuda(
    n, m, P_ro, P_ci, P_vals, A_ro, A_ci, A_vals,
    nnz_P, nnz_A, q, b, cones, settings, lib,
    soc_dims_i64, power_alphas,
    warm_x, warm_z, warm_s,
)
    # Ensure CUDA runtime is loaded
    _init_cudart() || error("Failed to load CUDA runtime (libcudart). Is CUDA installed?")

    solver = C_NULL
    # Track device pointers for cleanup
    device_ptrs = Ptr{Float64}[]
    try
        GC.@preserve P_ro P_ci P_vals A_ro A_ci A_vals q b soc_dims_i64 power_alphas begin
            # create() always takes host pointers for structure
            solver = c_moreau_solver_create(
                Int64(n), Int64(m),
                P_ro, P_ci, nnz_P,
                A_ro, A_ci, nnz_A,
                cones, settings; lib=lib,
            )

            # setup() needs device pointers for values
            d_P_vals = _to_device(P_vals)
            push!(device_ptrs, d_P_vals)
            d_A_vals = _to_device(A_vals)
            push!(device_ptrs, d_A_vals)
            c_moreau_solver_setup(solver, d_P_vals, nnz_P, d_A_vals, nnz_A; lib=lib)

            # solve() needs device pointers for q, b
            d_q = _to_device(q)
            push!(device_ptrs, d_q)
            d_b = _to_device(b)
            push!(device_ptrs, d_b)

            if warm_x !== nothing && warm_z !== nothing && warm_s !== nothing
                d_wx = _to_device(warm_x)
                push!(device_ptrs, d_wx)
                d_wz = _to_device(warm_z)
                push!(device_ptrs, d_wz)
                d_ws = _to_device(warm_s)
                push!(device_ptrs, d_ws)
                c_moreau_solver_solve_warm(solver, d_q, d_b, d_wx, d_wz, d_ws; lib=lib)
            else
                c_moreau_solver_solve(solver, d_q, d_b; lib=lib)
            end

            # get_solution returns device pointers for x/z/s + scalar fields
            sol_t = c_moreau_solver_get_solution(solver, Int64(0); lib=lib)

            # Return same vector type as input q: CuVector in → CuVector out (D2D),
            # Vector in → Vector out (D2H)
            x = _from_device_like(sol_t.x, n, q)
            z = _from_device_like(sol_t.z, m, q)
            s = _from_device_like(sol_t.s, m, q)

            return Solution(
                x, z, s,
                MoreauStatus(sol_t.status),
                sol_t.obj_val,
                sol_t.obj_val_dual,
                sol_t.solve_time,
                Int(sol_t.iterations),
                sol_t.r_prim,
                sol_t.r_dual,
            )
        end
    finally
        # Free all device allocations
        for dptr in device_ptrs
            cuda_free(dptr)
        end
        if solver != C_NULL
            c_moreau_solver_destroy(solver; lib=lib)
        end
    end
end

# ============================================================================
# CompiledSolver — persistent solver handle for repeated solves
# ============================================================================

"""
    CompiledSolver

A persistent solver handle that amortizes create/setup cost across repeated solves.
The sparsity structure (P, A patterns) is fixed at construction; only the numerical
values can change between solves via `setup!`.

# Usage
```julia
solver = CompiledSolver(n, m, P, A;
    device=:cuda,
    zero_cones=1, nonneg_cones=2,
)
setup!(solver, P_values, A_values)
sol = solve!(solver, q, b)
# ... change q, b ...
sol2 = solve!(solver, q2, b2)
destroy!(solver)  # or let GC handle it
```
"""
mutable struct CompiledSolver
    handle::Ptr{Cvoid}
    lib::String
    device::Symbol
    n::Int
    m::Int
    batch_size::Int
    nnz_P::Int
    nnz_A::Int
    # GC roots: prevent collection of arrays whose pointers were passed to C
    _gc_roots::Vector{Any}

    function CompiledSolver(
        n::Int, m::Int,
        P::SparseMatrixCSC{Float64,Int},
        A::SparseMatrixCSC{Float64,Int};
        device::Symbol=:auto,
        batch_size::Int=1,
        zero_cones::Int=0, nonneg_cones::Int=0,
        soc_dims::Vector{Int}=Int[],
        exp_cones::Int=0, power_alphas::Vector{Float64}=Float64[],
        verbose::Bool=false,
        max_iter::Int=200,
        time_limit::Float64=Inf,
        enable_grad::Bool=false,
        options...,
    )
        P_ro, P_ci, _ = _csc_to_csr(P)
        A_ro, A_ci, _ = _csc_to_csr(A)
        nnz_P = Int64(nnz(P))
        nnz_A = Int64(nnz(A))

        resolved_device = device == :auto ? _choose_device(n, nnz_A) : device
        lib = _get_lib(resolved_device)

        soc_dims_i64 = Vector{Int64}(soc_dims)
        cones = CMoreauCones(
            Int64(zero_cones),
            Int64(nonneg_cones),
            Int64(length(soc_dims)),
            length(soc_dims) > 0 ? pointer(soc_dims_i64) : C_NULL,
            Int64(exp_cones),
            Int64(length(power_alphas)),
            length(power_alphas) > 0 ? pointer(power_alphas) : C_NULL,
        )

        settings = _build_settings(;
            verbose=verbose, max_iter=max_iter,
            time_limit=time_limit, enable_grad=enable_grad,
            lib=lib, batch_size=batch_size, options...,
        )

        handle = c_moreau_solver_create(
            Int64(n), Int64(m),
            P_ro, P_ci, nnz_P,
            A_ro, A_ci, nnz_A,
            cones, settings; lib=lib,
        )

        obj = new(handle, lib, resolved_device, n, m, batch_size, nnz_P, nnz_A,
                  Any[P_ro, P_ci, A_ro, A_ci, soc_dims_i64, power_alphas])
        finalizer(obj) do s
            if s.handle != C_NULL
                c_moreau_solver_destroy(s.handle; lib=s.lib)
                s.handle = C_NULL
            end
        end
        return obj
    end
end

# Allow passing CompiledSolver directly to ccall where Ptr{Cvoid} is expected.
# cconvert keeps the solver GC-rooted for the duration of the ccall,
# preventing the finalizer from freeing the handle mid-call.
Base.cconvert(::Type{Ptr{Cvoid}}, solver::CompiledSolver) = solver
Base.unsafe_convert(::Type{Ptr{Cvoid}}, solver::CompiledSolver) = solver.handle

# --- CompiledSolver ccall wrappers (must come after struct definition) ---

# Vector-based setup (CPU path)
function c_moreau_solver_setup(solver::CompiledSolver, P_values::Vector{Float64}, A_values::Vector{Float64})
    err = ccall(
        _cfunc(solver.lib, :moreau_solver_setup), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Int64, Ptr{Float64}, Int64),
        solver, P_values, length(P_values), A_values, length(A_values),
    )
    _check_error(MoreauError(err), "moreau_solver_setup"; lib=solver.lib)
end

# Pointer-based setup (CUDA path — accepts device pointers)
function c_moreau_solver_setup(solver::CompiledSolver, P_values::Ptr{Float64}, nnz_P::Int64,
                               A_values::Ptr{Float64}, nnz_A::Int64)
    err = ccall(
        _cfunc(solver.lib, :moreau_solver_setup), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Int64, Ptr{Float64}, Int64),
        solver, P_values, nnz_P, A_values, nnz_A,
    )
    _check_error(MoreauError(err), "moreau_solver_setup"; lib=solver.lib)
end

# Vector-based solve (CPU path)
function c_moreau_solver_solve(solver::CompiledSolver, q::Vector{Float64}, b::Vector{Float64})
    err = ccall(
        _cfunc(solver.lib, :moreau_solver_solve), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}),
        solver, q, b,
    )
    _check_error(MoreauError(err), "moreau_solver_solve"; lib=solver.lib)
end

# Pointer-based solve (CUDA path — accepts device pointers)
function c_moreau_solver_solve(solver::CompiledSolver, q::Ptr{Float64}, b::Ptr{Float64})
    err = ccall(
        _cfunc(solver.lib, :moreau_solver_solve), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}),
        solver, q, b,
    )
    _check_error(MoreauError(err), "moreau_solver_solve"; lib=solver.lib)
end

# Vector-based warm solve (CPU path)
function c_moreau_solver_solve_warm(
    solver::CompiledSolver,
    q::Vector{Float64}, b::Vector{Float64},
    warm_x::Vector{Float64}, warm_z::Vector{Float64}, warm_s::Vector{Float64},
)
    err = ccall(
        _cfunc(solver.lib, :moreau_solver_solve_warm), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        solver, q, b, warm_x, warm_z, warm_s,
    )
    _check_error(MoreauError(err), "moreau_solver_solve_warm"; lib=solver.lib)
end

# Pointer-based warm solve (CUDA path — accepts device pointers)
function c_moreau_solver_solve_warm(
    solver::CompiledSolver,
    q::Ptr{Float64}, b::Ptr{Float64},
    warm_x::Ptr{Float64}, warm_z::Ptr{Float64}, warm_s::Ptr{Float64},
)
    err = ccall(
        _cfunc(solver.lib, :moreau_solver_solve_warm), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        solver, q, b, warm_x, warm_z, warm_s,
    )
    _check_error(MoreauError(err), "moreau_solver_solve_warm"; lib=solver.lib)
end

function c_moreau_solver_get_solution(solver::CompiledSolver, batch_idx::Int64)
    sol = Ref{CMoreauSolution}()
    err = ccall(
        _cfunc(solver.lib, :moreau_solver_get_solution), Int32,
        (Ptr{Cvoid}, Int64, Ptr{CMoreauSolution}),
        solver, batch_idx, sol,
    )
    _check_error(MoreauError(err), "moreau_solver_get_solution"; lib=solver.lib)
    return sol[]
end

function c_moreau_solver_copy_solution(solver::CompiledSolver, batch_idx::Int64, n::Int, m::Int)
    x = Vector{Float64}(undef, n)
    z = Vector{Float64}(undef, m)
    s = Vector{Float64}(undef, m)
    err = ccall(
        _cfunc(solver.lib, :moreau_solver_copy_solution), Int32,
        (Ptr{Cvoid}, Int64, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        solver, batch_idx, x, z, s,
    )
    _check_error(MoreauError(err), "moreau_solver_copy_solution"; lib=solver.lib)
    return x, z, s
end

function c_moreau_solver_get_status(solver::CompiledSolver, batch_idx::Int64)
    status = ccall(
        _cfunc(solver.lib, :moreau_solver_get_status), Int32,
        (Ptr{Cvoid}, Int64),
        solver, batch_idx,
    )
    return MoreauStatus(status)
end

function c_moreau_solver_backward(
    solver::CompiledSolver,
    dx::Vector{Float64},
    dz::Union{Vector{Float64},Nothing},
    ds::Union{Vector{Float64},Nothing},
    nnz_P::Int, nnz_A::Int, n::Int, m::Int,
)
    dP_out = Vector{Float64}(undef, nnz_P)
    dA_out = Vector{Float64}(undef, nnz_A)
    dq_out = Vector{Float64}(undef, n)
    db_out = Vector{Float64}(undef, m)
    dz_ptr = dz === nothing ? Ptr{Float64}(C_NULL) : pointer(dz)
    ds_ptr = ds === nothing ? Ptr{Float64}(C_NULL) : pointer(ds)
    err = ccall(
        _cfunc(solver.lib, :moreau_solver_backward), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        solver, dx, dz_ptr, ds_ptr,
        dP_out, dA_out, dq_out, db_out,
    )
    _check_error(MoreauError(err), "moreau_solver_backward"; lib=solver.lib)
    return dP_out, dA_out, dq_out, db_out
end

function c_moreau_solver_get_dims(solver::CompiledSolver)
    n = Ref{Int64}(0)
    m = Ref{Int64}(0)
    batch_size = Ref{Int64}(0)
    nnz_P = Ref{Int64}(0)
    nnz_A = Ref{Int64}(0)
    err = ccall(
        _cfunc(solver.lib, :moreau_solver_get_dims), Int32,
        (Ptr{Cvoid}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}, Ptr{Int64}),
        solver, n, m, batch_size, nnz_P, nnz_A,
    )
    _check_error(MoreauError(err), "moreau_solver_get_dims"; lib=solver.lib)
    return n[], m[], batch_size[], nnz_P[], nnz_A[]
end

function c_moreau_solver_memory_usage(solver::CompiledSolver)
    bytes = Ref{Csize_t}(0)
    err = ccall(
        _cfunc(solver.lib, :moreau_solver_memory_usage), Int32,
        (Ptr{Cvoid}, Ptr{Csize_t}),
        solver, bytes,
    )
    _check_error(MoreauError(err), "moreau_solver_memory_usage"; lib=solver.lib)
    return bytes[]
end

# Pointer-based backward (CUDA path — all pointers are device pointers)
function c_moreau_solver_backward(
    solver::CompiledSolver,
    dx::Ptr{Float64}, dz::Ptr{Float64}, ds::Ptr{Float64},
    dP_out::Ptr{Float64}, dA_out::Ptr{Float64},
    dq_out::Ptr{Float64}, db_out::Ptr{Float64},
)
    err = ccall(
        _cfunc(solver.lib, :moreau_solver_backward), Int32,
        (Ptr{Cvoid}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
         Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}),
        solver, dx, dz, ds,
        dP_out, dA_out, dq_out, db_out,
    )
    _check_error(MoreauError(err), "moreau_solver_backward"; lib=solver.lib)
end

"""
    setup!(solver::CompiledSolver, P_values, A_values)

Set the numerical values of P and A. Must be called before the first `solve!`,
and can be called again to update values for subsequent solves.

Vectors of length `nnz_P`/`nnz_A` are shared across all batch elements.
For CUDA solvers, accepts either `Vector{Float64}` (auto-copied to device)
or `CuVector{Float64}` (zero-copy).
"""
function setup!(solver::CompiledSolver, P_values::AbstractVector{Float64}, A_values::AbstractVector{Float64})
    solver.handle == C_NULL && error("CompiledSolver has been destroyed")
    if solver.device == :cuda
        _init_cudart() || error("Failed to load CUDA runtime (libcudart)")
        d_P = _to_device(P_values)
        d_A = _to_device(A_values)
        try
            c_moreau_solver_setup(solver, d_P, Int64(length(P_values)),
                                  d_A, Int64(length(A_values)))
        finally
            _maybe_free_ptr(d_P, P_values)
            _maybe_free_ptr(d_A, A_values)
        end
    else
        c_moreau_solver_setup(solver, Vector{Float64}(P_values),
                              Vector{Float64}(A_values))
    end
    return solver
end

# Free device pointer only if we allocated it (i.e., input was a Vector, not CuVector)
_maybe_free_ptr(dptr::Ptr{Float64}, ::Vector{Float64}) = cuda_free(dptr)
_maybe_free_ptr(::Ptr{Float64}, ::AbstractVector{Float64}) = nothing  # CuVector: no-op

"""
    solve!(solver::CompiledSolver, q, b; warm_start=nothing) -> Solution

Solve with the current P/A values and the given q, b vectors.
Returns a `Solution{V}` where `V` matches the type of `q` and `b`.

For warm starting, pass `warm_start` as a `Solution` from a previous solve.
"""
function solve!(
    solver::CompiledSolver,
    q::AbstractVector{Float64},
    b::AbstractVector{Float64};
    warm_start::Union{Solution,Nothing}=nothing,
)
    solver.handle == C_NULL && error("CompiledSolver has been destroyed")
    typeof(q) == typeof(b) || error("q and b must have the same type, got $(typeof(q)) and $(typeof(b))")

    if solver.device == :cuda
        return _compiled_solve_cuda(solver, q, b, warm_start)
    else
        return _compiled_solve_cpu(solver, q, b, warm_start)
    end
end

function _compiled_solve_cpu(solver::CompiledSolver, q::AbstractVector{Float64},
                             b::AbstractVector{Float64}, warm_start)
    q_host = Vector{Float64}(q)
    b_host = Vector{Float64}(b)
    if warm_start !== nothing
        wx = Vector{Float64}(warm_start.x)
        wz = Vector{Float64}(warm_start.z)
        ws = Vector{Float64}(warm_start.s)
        c_moreau_solver_solve_warm(solver, q_host, b_host, wx, wz, ws)
    else
        c_moreau_solver_solve(solver, q_host, b_host)
    end

    x, z, s = c_moreau_solver_copy_solution(solver, Int64(0), solver.n, solver.m)
    sol_t = c_moreau_solver_get_solution(solver, Int64(0))
    return Solution(x, z, s, MoreauStatus(sol_t.status), sol_t.obj_val, sol_t.obj_val_dual,
                    sol_t.solve_time, Int(sol_t.iterations), sol_t.r_prim, sol_t.r_dual)
end

function _compiled_solve_cuda(solver::CompiledSolver, q::AbstractVector{Float64},
                              b::AbstractVector{Float64}, warm_start)
    _init_cudart() || error("Failed to load CUDA runtime (libcudart)")
    device_ptrs = Ptr{Float64}[]
    try
        d_q = _to_device(q)
        push!(device_ptrs, d_q)
        d_b = _to_device(b)
        push!(device_ptrs, d_b)

        if warm_start !== nothing
            d_wx = _to_device(warm_start.x)
            push!(device_ptrs, d_wx)
            d_wz = _to_device(warm_start.z)
            push!(device_ptrs, d_wz)
            d_ws = _to_device(warm_start.s)
            push!(device_ptrs, d_ws)
            c_moreau_solver_solve_warm(solver, d_q, d_b, d_wx, d_wz, d_ws)
        else
            c_moreau_solver_solve(solver, d_q, d_b)
        end

        sol_t = c_moreau_solver_get_solution(solver, Int64(0))
        x = _from_device_like(sol_t.x, solver.n, q)
        z = _from_device_like(sol_t.z, solver.m, q)
        s = _from_device_like(sol_t.s, solver.m, q)
        return Solution(x, z, s, MoreauStatus(sol_t.status), sol_t.obj_val, sol_t.obj_val_dual,
                        sol_t.solve_time, Int(sol_t.iterations), sol_t.r_prim, sol_t.r_dual)
    finally
        for dptr in device_ptrs
            _maybe_free_ptr(dptr, q)
        end
    end
end

"""
    setup!(solver::CompiledSolver, P_values::AbstractMatrix{Float64}, A_values::AbstractMatrix{Float64})

Set per-batch numerical values of P and A. Each column corresponds to one batch element.
`P_values` has size `(nnz_P, batch_size)`, `A_values` has size `(nnz_A, batch_size)`.
"""
function setup!(solver::CompiledSolver, P_values::AbstractMatrix{Float64}, A_values::AbstractMatrix{Float64})
    solver.handle == C_NULL && error("CompiledSolver has been destroyed")
    size(P_values, 2) == solver.batch_size || error("P_values has $(size(P_values, 2)) columns but batch_size is $(solver.batch_size)")
    size(A_values, 2) == solver.batch_size || error("A_values has $(size(A_values, 2)) columns but batch_size is $(solver.batch_size)")
    size(P_values, 1) == solver.nnz_P || error("P_values has $(size(P_values, 1)) rows but nnz_P is $(solver.nnz_P)")
    size(A_values, 1) == solver.nnz_A || error("A_values has $(size(A_values, 1)) rows but nnz_A is $(solver.nnz_A)")

    # Flatten to contiguous vector (column-major = batch-major for C API)
    P_flat = vec(Matrix{Float64}(P_values))
    A_flat = vec(Matrix{Float64}(A_values))

    if solver.device == :cuda
        _init_cudart() || error("Failed to load CUDA runtime (libcudart)")
        d_P = _to_device(P_flat)
        d_A = _to_device(A_flat)
        try
            c_moreau_solver_setup(solver, d_P, Int64(length(P_flat)),
                                  d_A, Int64(length(A_flat)))
        finally
            cuda_free(d_P)
            cuda_free(d_A)
        end
    else
        c_moreau_solver_setup(solver, P_flat, A_flat)
    end
    return solver
end

"""
    solve!(solver::CompiledSolver, q::AbstractMatrix{Float64}, b::AbstractMatrix{Float64}; ...) -> BatchedSolution

Solve a batch of problems. `q` has size `(n, batch_size)`, `b` has size `(m, batch_size)`.
Returns a `BatchedSolution` where column `i` is the solution for batch element `i`.
"""
function solve!(
    solver::CompiledSolver,
    q::AbstractMatrix{Float64},
    b::AbstractMatrix{Float64};
    warm_start::Union{BatchedSolution,Nothing}=nothing,
)
    solver.handle == C_NULL && error("CompiledSolver has been destroyed")
    size(q, 2) == solver.batch_size || error("q has $(size(q, 2)) columns but batch_size is $(solver.batch_size)")
    size(b, 2) == solver.batch_size || error("b has $(size(b, 2)) columns but batch_size is $(solver.batch_size)")
    size(q, 1) == solver.n || error("q has $(size(q, 1)) rows but n is $(solver.n)")
    size(b, 1) == solver.m || error("b has $(size(b, 1)) rows but m is $(solver.m)")

    if solver.device == :cuda
        return _compiled_solve_batch_cuda(solver, q, b, warm_start)
    else
        return _compiled_solve_batch_cpu(solver, q, b, warm_start)
    end
end

function _compiled_solve_batch_cpu(solver::CompiledSolver, q::AbstractMatrix{Float64},
                                    b::AbstractMatrix{Float64}, warm_start)
    q_flat = vec(Matrix{Float64}(q))
    b_flat = vec(Matrix{Float64}(b))
    if warm_start !== nothing
        wx = vec(Matrix{Float64}(warm_start.x))
        wz = vec(Matrix{Float64}(warm_start.z))
        ws = vec(Matrix{Float64}(warm_start.s))
        c_moreau_solver_solve_warm(solver, q_flat, b_flat, wx, wz, ws)
    else
        c_moreau_solver_solve(solver, q_flat, b_flat)
    end

    return _collect_batch_solution_cpu(solver)
end

function _compiled_solve_batch_cuda(solver::CompiledSolver, q::AbstractMatrix{Float64},
                                     b::AbstractMatrix{Float64}, warm_start)
    _init_cudart() || error("Failed to load CUDA runtime (libcudart)")
    q_flat = vec(Matrix{Float64}(q))
    b_flat = vec(Matrix{Float64}(b))
    device_ptrs = Ptr{Float64}[]
    try
        d_q = _to_device(q_flat)
        push!(device_ptrs, d_q)
        d_b = _to_device(b_flat)
        push!(device_ptrs, d_b)

        if warm_start !== nothing
            wx = vec(Matrix{Float64}(warm_start.x))
            wz = vec(Matrix{Float64}(warm_start.z))
            ws = vec(Matrix{Float64}(warm_start.s))
            d_wx = _to_device(wx)
            push!(device_ptrs, d_wx)
            d_wz = _to_device(wz)
            push!(device_ptrs, d_wz)
            d_ws = _to_device(ws)
            push!(device_ptrs, d_ws)
            c_moreau_solver_solve_warm(solver, d_q, d_b, d_wx, d_wz, d_ws)
        else
            c_moreau_solver_solve(solver, d_q, d_b)
        end

        return _collect_batch_solution_cpu(solver)  # always copy to host for now
    finally
        for dptr in device_ptrs
            cuda_free(dptr)
        end
    end
end

function _collect_batch_solution_cpu(solver::CompiledSolver)
    n, m, bs = solver.n, solver.m, solver.batch_size
    x_mat = Matrix{Float64}(undef, n, bs)
    z_mat = Matrix{Float64}(undef, m, bs)
    s_mat = Matrix{Float64}(undef, m, bs)
    statuses = Vector{MoreauStatus}(undef, bs)
    obj_vals = Vector{Float64}(undef, bs)
    obj_vals_dual = Vector{Float64}(undef, bs)
    r_prims = Vector{Float64}(undef, bs)
    r_duals = Vector{Float64}(undef, bs)
    solve_time = 0.0
    iterations = 0

    for i in 0:(bs-1)
        xi, zi, si = c_moreau_solver_copy_solution(solver, Int64(i), n, m)
        sol_t = c_moreau_solver_get_solution(solver, Int64(i))
        x_mat[:, i+1] = xi
        z_mat[:, i+1] = zi
        s_mat[:, i+1] = si
        statuses[i+1] = MoreauStatus(sol_t.status)
        obj_vals[i+1] = sol_t.obj_val
        obj_vals_dual[i+1] = sol_t.obj_val_dual
        r_prims[i+1] = sol_t.r_prim
        r_duals[i+1] = sol_t.r_dual
        solve_time = max(solve_time, sol_t.solve_time)
        iterations = max(iterations, Int(sol_t.iterations))
    end

    return BatchedSolution(x_mat, z_mat, s_mat, statuses, obj_vals, obj_vals_dual,
                           solve_time, iterations, r_prims, r_duals)
end

"""
    backward!(solver::CompiledSolver, dx; dz=nothing, ds=nothing) -> NamedTuple

Compute gradients of the solve w.r.t. problem data. Must be called after `solve!`
with `enable_grad=true`.

Returns `(dP_values, dA_values, dq, db)` as vectors (single) or matrices (batched).
"""
function backward!(
    solver::CompiledSolver,
    dx::AbstractVector{Float64};
    dz::Union{AbstractVector{Float64},Nothing}=nothing,
    ds::Union{AbstractVector{Float64},Nothing}=nothing,
)
    solver.handle == C_NULL && error("CompiledSolver has been destroyed")
    if solver.device == :cuda
        return _backward_cuda(solver, dx, dz, ds)
    else
        return _backward_cpu(solver, dx, dz, ds)
    end
end

function _backward_cpu(solver::CompiledSolver, dx, dz, ds)
    dP, dA, dq, db = c_moreau_solver_backward(
        solver, Vector{Float64}(dx),
        dz === nothing ? nothing : Vector{Float64}(dz),
        ds === nothing ? nothing : Vector{Float64}(ds),
        solver.nnz_P, solver.nnz_A, solver.n, solver.m,
    )
    return (dP_values=dP, dA_values=dA, dq=dq, db=db)
end

function _backward_cuda(solver::CompiledSolver, dx, dz, ds)
    _init_cudart() || error("Failed to load CUDA runtime (libcudart)")
    n, m, nnz_P, nnz_A = solver.n, solver.m, solver.nnz_P, solver.nnz_A
    device_ptrs = Ptr{Float64}[]
    try
        d_dx = _to_device(dx)
        push!(device_ptrs, d_dx)
        d_dz = dz === nothing ? Ptr{Float64}(C_NULL) : _to_device(dz)
        if dz !== nothing; push!(device_ptrs, d_dz); end
        d_ds = ds === nothing ? Ptr{Float64}(C_NULL) : _to_device(ds)
        if ds !== nothing; push!(device_ptrs, d_ds); end

        # Allocate device output buffers
        d_dP = Ptr{Float64}(cuda_malloc(nnz_P * sizeof(Float64)))
        push!(device_ptrs, d_dP)
        d_dA = Ptr{Float64}(cuda_malloc(nnz_A * sizeof(Float64)))
        push!(device_ptrs, d_dA)
        d_dq = Ptr{Float64}(cuda_malloc(n * sizeof(Float64)))
        push!(device_ptrs, d_dq)
        d_db = Ptr{Float64}(cuda_malloc(m * sizeof(Float64)))
        push!(device_ptrs, d_db)

        c_moreau_solver_backward(solver, d_dx, d_dz, d_ds, d_dP, d_dA, d_dq, d_db)

        # Copy results back to host
        dP = _from_device(d_dP, nnz_P)
        dA = _from_device(d_dA, nnz_A)
        dq = _from_device(d_dq, n)
        db = _from_device(d_db, m)
        return (dP_values=dP, dA_values=dA, dq=dq, db=db)
    finally
        for dptr in device_ptrs
            cuda_free(dptr)
        end
    end
end

function backward!(
    solver::CompiledSolver,
    dx::AbstractMatrix{Float64};
    dz::Union{AbstractMatrix{Float64},Nothing}=nothing,
    ds::Union{AbstractMatrix{Float64},Nothing}=nothing,
)
    solver.handle == C_NULL && error("CompiledSolver has been destroyed")
    bs = solver.batch_size
    size(dx, 2) == bs || error("dx has $(size(dx, 2)) columns but batch_size is $bs")

    if solver.device == :cuda
        return _backward_batch_cuda(solver, dx, dz, ds)
    else
        return _backward_batch_cpu(solver, dx, dz, ds)
    end
end

function _backward_batch_cpu(solver::CompiledSolver, dx, dz, ds)
    bs = solver.batch_size
    dx_flat = vec(Matrix{Float64}(dx))
    dz_flat = dz === nothing ? nothing : vec(Matrix{Float64}(dz))
    ds_flat = ds === nothing ? nothing : vec(Matrix{Float64}(ds))

    dP_flat, dA_flat, dq_flat, db_flat = c_moreau_solver_backward(
        solver, dx_flat, dz_flat, ds_flat,
        solver.nnz_P * bs, solver.nnz_A * bs, solver.n * bs, solver.m * bs,
    )

    # Reshape back to (dim, batch_size)
    dP = reshape(dP_flat, solver.nnz_P, bs)
    dA = reshape(dA_flat, solver.nnz_A, bs)
    dq = reshape(dq_flat, solver.n, bs)
    db = reshape(db_flat, solver.m, bs)
    return (dP_values=dP, dA_values=dA, dq=dq, db=db)
end

function _backward_batch_cuda(solver::CompiledSolver, dx, dz, ds)
    _init_cudart() || error("Failed to load CUDA runtime (libcudart)")
    bs = solver.batch_size
    n, m, nnz_P, nnz_A = solver.n, solver.m, solver.nnz_P, solver.nnz_A
    device_ptrs = Ptr{Float64}[]
    try
        dx_flat = vec(Matrix{Float64}(dx))
        d_dx = _to_device(dx_flat)
        push!(device_ptrs, d_dx)

        if dz !== nothing
            dz_flat = vec(Matrix{Float64}(dz))
            d_dz = _to_device(dz_flat)
            push!(device_ptrs, d_dz)
        else
            d_dz = Ptr{Float64}(C_NULL)
        end

        if ds !== nothing
            ds_flat = vec(Matrix{Float64}(ds))
            d_ds = _to_device(ds_flat)
            push!(device_ptrs, d_ds)
        else
            d_ds = Ptr{Float64}(C_NULL)
        end

        # Allocate device output buffers
        d_dP = Ptr{Float64}(cuda_malloc(nnz_P * bs * sizeof(Float64)))
        push!(device_ptrs, d_dP)
        d_dA = Ptr{Float64}(cuda_malloc(nnz_A * bs * sizeof(Float64)))
        push!(device_ptrs, d_dA)
        d_dq = Ptr{Float64}(cuda_malloc(n * bs * sizeof(Float64)))
        push!(device_ptrs, d_dq)
        d_db = Ptr{Float64}(cuda_malloc(m * bs * sizeof(Float64)))
        push!(device_ptrs, d_db)

        c_moreau_solver_backward(solver, d_dx, d_dz, d_ds, d_dP, d_dA, d_dq, d_db)

        # Copy results back to host and reshape
        dP = reshape(_from_device(d_dP, nnz_P * bs), nnz_P, bs)
        dA = reshape(_from_device(d_dA, nnz_A * bs), nnz_A, bs)
        dq = reshape(_from_device(d_dq, n * bs), n, bs)
        db = reshape(_from_device(d_db, m * bs), m, bs)
        return (dP_values=dP, dA_values=dA, dq=dq, db=db)
    finally
        for dptr in device_ptrs
            cuda_free(dptr)
        end
    end
end

"""
    setup_and_solve!(solver::CompiledSolver, P_values, A_values, q, b; ...) -> Solution/BatchedSolution

Convenience function that calls `setup!` then `solve!`. Useful as a single differentiable
operation when you need gradients w.r.t. P, A, q, and b simultaneously.
"""
function setup_and_solve!(
    solver::CompiledSolver,
    P_values::AbstractVector{Float64},
    A_values::AbstractVector{Float64},
    q::AbstractVector{Float64},
    b::AbstractVector{Float64};
    warm_start::Union{Solution,Nothing}=nothing,
)
    setup!(solver, P_values, A_values)
    return solve!(solver, q, b; warm_start=warm_start)
end

function setup_and_solve!(
    solver::CompiledSolver,
    P_values::AbstractVector{Float64},
    A_values::AbstractVector{Float64},
    q::AbstractMatrix{Float64},
    b::AbstractMatrix{Float64};
    warm_start::Union{BatchedSolution,Nothing}=nothing,
)
    setup!(solver, P_values, A_values)
    return solve!(solver, q, b; warm_start=warm_start)
end

function setup_and_solve!(
    solver::CompiledSolver,
    P_values::AbstractMatrix{Float64},
    A_values::AbstractMatrix{Float64},
    q::AbstractMatrix{Float64},
    b::AbstractMatrix{Float64};
    warm_start::Union{BatchedSolution,Nothing}=nothing,
)
    setup!(solver, P_values, A_values)
    return solve!(solver, q, b; warm_start=warm_start)
end

"""
    destroy!(solver::CompiledSolver)

Explicitly free the solver's C resources. After this call, `solve!` will error.
Also called automatically by the GC finalizer.
"""
function destroy!(solver::CompiledSolver)
    if solver.handle != C_NULL
        c_moreau_solver_destroy(solver.handle; lib=solver.lib)
        solver.handle = C_NULL
    end
    empty!(solver._gc_roots)
    return nothing
end
