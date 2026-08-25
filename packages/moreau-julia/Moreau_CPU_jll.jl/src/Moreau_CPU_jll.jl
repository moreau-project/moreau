module Moreau_CPU_jll

using Artifacts
using LazyArtifacts: ensure_artifact_installed
using Libdl

const libmoreau_path = Ref{String}("")
const libmoreau_handle = Ref{Ptr{Cvoid}}(C_NULL)

# libmoreau is the symbol used in @ccall — it must resolve to a library path.
libmoreau::String = ""

function __init__()
    # Priority 1: explicit env var (local development)
    path = get(ENV, "MOREAU_CPU_LIB", "")

    # Priority 2: Julia Artifacts system (auto-downloads from Artifacts.toml)
    if isempty(path)
        artifacts_toml = joinpath(pkgdir(Moreau_CPU_jll), "Artifacts.toml")
        if isfile(artifacts_toml)
            hash = artifact_hash("moreau_cpu", artifacts_toml;
                platform=Base.BinaryPlatforms.HostPlatform())
            if hash !== nothing
                ensure_artifact_installed("moreau_cpu", artifacts_toml;
                    platform=Base.BinaryPlatforms.HostPlatform())
                if artifact_exists(hash)
                    path = _find_lib_in_artifact(artifact_path(hash))
                end
            end
        end
    end

    # Priority 3: system library search
    if isempty(path)
        path = something(
            Libdl.find_library("moreau_cpu"),
            Libdl.find_library("moreau"),
            Libdl.find_library("libmoreau_cpu"),
            Libdl.find_library("libmoreau"),
            "",
        )
    end

    if isempty(path)
        error(
            "Moreau CPU library not found. Either:\n" *
            "  1. Set MOREAU_CPU_LIB to the path of the libmoreau shared library, or\n" *
            "  2. Ensure Artifacts.toml has a valid moreau_cpu entry, or\n" *
            "  3. Place libmoreau_cpu on your system library path.",
        )
    end

    global libmoreau = path
    libmoreau_path[] = path
    libmoreau_handle[] = Libdl.dlopen(path)
end

function _find_lib_in_artifact(artifact_dir::String)
    lib_dir = joinpath(artifact_dir, "lib")
    if Sys.isapple()
        candidate = joinpath(lib_dir, "libmoreau_cpu.dylib")
    else
        candidate = joinpath(lib_dir, "libmoreau_cpu.so")
    end
    return isfile(candidate) ? candidate : ""
end

export libmoreau

end # module
