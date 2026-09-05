const _ARTIFACTS_TOML = normpath(joinpath(@__DIR__, "..", "Artifacts.toml"))

const _libmoreau_handle = Ref{Ptr{Cvoid}}(C_NULL)
const _libmoreau_cuda_handle = Ref{Ptr{Cvoid}}(C_NULL)
const _cuda_available = Ref(false)

libmoreau::String = ""
libmoreau_cuda::String = ""

function _find_system_library(names::Vector{String})
    path = Libdl.find_library(names)
    return path === nothing ? "" : path
end

function _artifact_library_path(artifact_name::String, library_name::String)
    platform = Base.BinaryPlatforms.HostPlatform()
    hash = Artifacts.artifact_hash(
        artifact_name,
        _ARTIFACTS_TOML;
        platform=platform,
    )
    hash === nothing && return ""

    LazyArtifacts.ensure_artifact_installed(
        artifact_name,
        _ARTIFACTS_TOML;
        platform=platform,
    )
    root = Artifacts.artifact_path(hash)
    path = joinpath(root, "lib", library_name)
    return isfile(path) ? path : ""
end

function _cpu_library_name()
    return Sys.isapple() ? "libmoreau_cpu.dylib" : "libmoreau_cpu.so"
end

function _cuda_library_name()
    return Sys.isapple() ? "libmoreau_cuda.dylib" : "libmoreau_cuda.so"
end

function _init_cpu_library()
    path = get(ENV, "MOREAU_CPU_LIB", "")
    if isempty(path)
        path = _artifact_library_path("moreau_cpu", _cpu_library_name())
    end
    if isempty(path)
        path = _find_system_library([
            "moreau_cpu",
            "moreau",
            "libmoreau_cpu",
            "libmoreau",
        ])
    end
    if isempty(path)
        error(
            "Moreau CPU library not found. Set MOREAU_CPU_LIB, install a " *
            "supported Moreau artifact, or place libmoreau_cpu on the system " *
            "library path.",
        )
    end

    handle = Libdl.dlopen(path)
    global libmoreau = path
    _libmoreau_handle[] = handle
    return
end

function _parse_cuda_major(value::AbstractString)
    match_result = match(r"^(\d+)", strip(value))
    match_result === nothing && return nothing
    major = parse(Int, only(match_result.captures))
    return major in (12, 13) ? major : nothing
end

function _detect_cuda_major()
    try
        output = read(`nvidia-smi`, String)
        match_result = match(r"CUDA Version:\s*(\d+)", output)
        match_result === nothing && return nothing
        return parse(Int, only(match_result.captures))
    catch
        return nothing
    end
end

function _cuda_artifact_order(; allow_fallback::Bool=false)
    requested = get(ENV, "MOREAU_CUDA_VERSION", "")
    if !isempty(requested)
        major = _parse_cuda_major(requested)
        if major === nothing
            @warn "MOREAU_CUDA_VERSION=$requested is not a supported CUDA version"
            return ()
        end
        return ("moreau_cuda$major",)
    end

    detected = _detect_cuda_major()
    if detected !== nothing
        return detected >= 13 ? ("moreau_cuda13", "moreau_cuda12") : ("moreau_cuda12",)
    end
    return allow_fallback ? ("moreau_cuda13", "moreau_cuda12") : ()
end

function _try_cuda_library(path::String)
    isempty(path) && return false
    handle = Libdl.dlopen(path; throw_error=false)
    handle === nothing && return false

    global libmoreau_cuda = path
    _libmoreau_cuda_handle[] = handle
    _cuda_available[] = true
    return true
end

function _load_cuda_library(; allow_fallback::Bool=false)
    _cuda_available[] && return true

    path = get(ENV, "MOREAU_CUDA_LIB", "")
    _try_cuda_library(path) && return true

    path = _find_system_library(["moreau_cuda", "libmoreau_cuda"])
    _try_cuda_library(path) && return true

    for artifact_name in _cuda_artifact_order(; allow_fallback=allow_fallback)
        try
            path = _artifact_library_path(artifact_name, _cuda_library_name())
            _try_cuda_library(path) && return true
        catch exception
            @debug "Failed to load $artifact_name" exception=(exception, catch_backtrace())
        end
    end
    return false
end

function __init__()
    _init_cpu_library()
    return
end
