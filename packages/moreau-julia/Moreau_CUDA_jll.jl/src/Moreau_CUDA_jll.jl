module Moreau_CUDA_jll

using Artifacts
using LazyArtifacts: ensure_artifact_installed
using Libdl

const libmoreau_cuda_path = Ref{String}("")
const libmoreau_cuda_handle = Ref{Ptr{Cvoid}}(C_NULL)

# libmoreau_cuda is the symbol used in ccall — it must resolve to a library path.
# Empty string means CUDA library is not available.
libmoreau_cuda::String = ""

# Whether the CUDA library was successfully loaded
const _available = Ref{Bool}(false)

function __init__()
    # Priority 1: explicit env var (local development)
    path = get(ENV, "MOREAU_CUDA_LIB", "")

    # Priority 2: Julia Artifacts system (auto-downloads from Artifacts.toml)
    if isempty(path)
        artifacts_toml = joinpath(pkgdir(Moreau_CUDA_jll), "Artifacts.toml")
        if isfile(artifacts_toml)
            for artifact_name in _cuda_artifact_order()
                hash = artifact_hash(artifact_name, artifacts_toml;
                    platform=Base.BinaryPlatforms.HostPlatform())
                hash === nothing && continue
                try
                    ensure_artifact_installed(artifact_name, artifacts_toml;
                        platform=Base.BinaryPlatforms.HostPlatform())
                catch e
                    @debug "Failed to install $(artifact_name)" exception=(e, catch_backtrace())
                    continue
                end
                if artifact_exists(hash)
                    path = _find_lib_in_artifact(artifact_path(hash))
                    !isempty(path) && break
                end
            end
        end
    end

    # Priority 3: system library search
    if isempty(path)
        path = something(
            Libdl.find_library("moreau_cuda"),
            Libdl.find_library("libmoreau_cuda"),
            "",
        )
    end

    if isempty(path)
        @debug "Moreau CUDA library not found — CUDA backend unavailable. " *
               "Set MOREAU_CUDA_LIB or install the CUDA artifact."
        global libmoreau_cuda = ""
        _available[] = false
        return
    end

    try
        global libmoreau_cuda = path
        libmoreau_cuda_path[] = path
        libmoreau_cuda_handle[] = Libdl.dlopen(path)
        _available[] = true
    catch e
        @warn "Failed to load Moreau CUDA library" path exception=(e, catch_backtrace())
        global libmoreau_cuda = ""
        _available[] = false
    end
end

is_available() = _available[]

"""
    _cuda_artifact_order() -> Tuple{String, ...}

Return artifact names to try, in priority order:
1. `MOREAU_CUDA_VERSION` env var (e.g. "12" or "13") — use only that version
2. `nvidia-smi` — detect max supported CUDA version, prefer matching major
3. Fallback — try cuda13 first, then cuda12
"""
function _cuda_artifact_order()
    # Priority 1: explicit env var
    env_ver = get(ENV, "MOREAU_CUDA_VERSION", "")
    if !isempty(env_ver)
        major = _parse_cuda_major(env_ver)
        if major !== nothing
            @debug "MOREAU_CUDA_VERSION=$env_ver — using cuda$major"
            return ("moreau_cuda$major",)
        else
            @warn "MOREAU_CUDA_VERSION=$env_ver is not a valid CUDA version, ignoring"
        end
    end

    # Priority 2: detect from nvidia-smi
    detected = _detect_cuda_major()
    if detected !== nothing
        @debug "nvidia-smi reports CUDA $detected"
        if detected >= 13
            return ("moreau_cuda13", "moreau_cuda12")
        else
            return ("moreau_cuda12",)
        end
    end

    # Priority 3: fallback — prefer newer
    return ("moreau_cuda13", "moreau_cuda12")
end

function _parse_cuda_major(s::AbstractString)::Union{Int, Nothing}
    # Accept "12", "13", "12.2", "13.0", etc.
    m = match(r"^(\d+)", strip(s))
    m === nothing && return nothing
    major = parse(Int, m.captures[1])
    return major in (12, 13) ? major : nothing
end

function _detect_cuda_major()::Union{Int, Nothing}
    try
        output = read(`nvidia-smi`, String)
        # nvidia-smi header contains e.g. "CUDA Version: 13.0"
        m = match(r"CUDA Version:\s*(\d+)", output)
        m === nothing && return nothing
        return parse(Int, m.captures[1])
    catch
        return nothing
    end
end

function _find_lib_in_artifact(artifact_dir::String)
    lib_dir = joinpath(artifact_dir, "lib")
    if Sys.isapple()
        candidate = joinpath(lib_dir, "libmoreau_cuda.dylib")
    else
        candidate = joinpath(lib_dir, "libmoreau_cuda.so")
    end
    return isfile(candidate) ? candidate : ""
end

export libmoreau_cuda, is_available

end # module
