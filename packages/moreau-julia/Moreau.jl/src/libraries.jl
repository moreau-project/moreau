# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0

const _libmoreau_handle = Ref{Ptr{Cvoid}}(C_NULL)
const _libmoreau_cuda_handle = Ref{Ptr{Cvoid}}(C_NULL)
const _cuda_available = Ref(false)
const _cuda_jll_loader = Ref{Union{Nothing,Function}}(nothing)
libmoreau::String = ""
libmoreau_cuda::String = ""

function _check_library_version(handle, path)
    pointer = ccall(Libdl.dlsym(handle, :moreau_version), Cstring, ())
    version = VersionNumber(replace(unsafe_string(pointer), ".dev" => "-dev"))
    expected = pkgversion(@__MODULE__)
    version == expected || error(
        "Moreau.jl $expected requires native Moreau $expected; $path reports $version.",
    )
    return
end

function _init_cpu_library()
    path = get(ENV, "MOREAU_CPU_LIB", "")
    if isempty(path)
        Moreau_CPU_jll.is_available() || error("Moreau CPU JLL is unavailable on this platform")
        path = Moreau_CPU_jll.libmoreau
    end
    handle = Libdl.dlopen(path)
    try
        _check_library_version(handle, path)
    catch
        Libdl.dlclose(handle)
        rethrow()
    end
    global libmoreau = path
    _libmoreau_handle[] = handle
    return
end

function _try_cuda_library(path::String)
    isempty(path) && return false
    handle = Libdl.dlopen(path; throw_error=false)
    handle === nothing && return false
    try
        _check_library_version(handle, path)
    catch
        Libdl.dlclose(handle)
        rethrow()
    end
    global libmoreau_cuda = path
    _libmoreau_cuda_handle[] = handle
    _cuda_available[] = true
    return true
end

function _load_cuda_library()
    _cuda_available[] && return true
    path = get(ENV, "MOREAU_CUDA_LIB", "")
    if !isempty(path)
        return _try_cuda_library(path)
    end
    loader = _cuda_jll_loader[]
    return loader !== nothing && loader()
end

function __init__()
    _init_cpu_library()
    return
end
