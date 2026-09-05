# ============================================================================
# Lightweight CUDA runtime wrappers (no CUDA.jl dependency)
# ============================================================================

const _cudart_handle = Ref{Ptr{Cvoid}}(C_NULL)
const _cudart_loaded = Ref{Bool}(false)

# cudaMemcpyKind enum values
const cudaMemcpyHostToDevice = Cint(1)
const cudaMemcpyDeviceToHost = Cint(2)
const cudaMemcpyDeviceToDevice = Cint(3)

function _init_cudart()
    _cudart_loaded[] && return true

    # Try common libcudart names
    for name in ("libcudart", "cudart", "libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11")
        handle = Libdl.dlopen(name; throw_error=false)
        if handle !== nothing
            _cudart_handle[] = handle
            _cudart_loaded[] = true
            return true
        end
    end
    return false
end

function cuda_malloc(nbytes::Integer)
    ptr_ref = Ref{Ptr{Cvoid}}(C_NULL)
    err = ccall(
        Libdl.dlsym(_cudart_handle[], :cudaMalloc),
        Cint,
        (Ptr{Ptr{Cvoid}}, Csize_t),
        ptr_ref, Csize_t(nbytes),
    )
    err == 0 || error("cudaMalloc failed with error code $err")
    return ptr_ref[]
end

function cuda_memcpy_h2d(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, nbytes::Integer)
    err = ccall(
        Libdl.dlsym(_cudart_handle[], :cudaMemcpy),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint),
        dst, src, Csize_t(nbytes), cudaMemcpyHostToDevice,
    )
    err == 0 || error("cudaMemcpy H2D failed with error code $err")
end

function cuda_free(ptr::Ptr)
    ptr == C_NULL && return
    err = ccall(
        Libdl.dlsym(_cudart_handle[], :cudaFree),
        Cint,
        (Ptr{Cvoid},),
        Ptr{Cvoid}(ptr),
    )
    err == 0 || error("cudaFree failed with error code $err")
end

"""
    _to_device(v::AbstractVector{Float64}) -> Ptr{Float64}

Allocate device memory and copy host vector to device. Caller must `cuda_free` the result.
The CUDA extension overrides this for CuVector to return the device pointer directly (zero-copy).
"""
function _to_device(v::AbstractVector{Float64})
    return _to_device(Vector{Float64}(v))
end

function _to_device(v::Vector{Float64})
    nbytes = sizeof(v)
    if nbytes == 0
        return Ptr{Float64}(C_NULL)
    end
    dptr = cuda_malloc(nbytes)
    GC.@preserve v begin
        cuda_memcpy_h2d(dptr, Ptr{Cvoid}(pointer(v)), nbytes)
    end
    return Ptr{Float64}(dptr)
end

function cuda_memcpy_d2h(dst::Ptr{Cvoid}, src::Ptr{Cvoid}, nbytes::Integer)
    err = ccall(
        Libdl.dlsym(_cudart_handle[], :cudaMemcpy),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t, Cint),
        dst, src, Csize_t(nbytes), cudaMemcpyDeviceToHost,
    )
    err == 0 || error("cudaMemcpy D2H failed with error code $err")
end

"""
    _from_device(device_ptr::Ptr{Float64}, len::Int) -> Vector{Float64}

Copy `len` Float64 values from device pointer to a new host Vector.
"""
function _from_device(device_ptr::Ptr{Float64}, len::Int)
    v = Vector{Float64}(undef, len)
    if len > 0
        GC.@preserve v begin
            cuda_memcpy_d2h(Ptr{Cvoid}(pointer(v)), Ptr{Cvoid}(device_ptr), len * sizeof(Float64))
        end
    end
    return v
end

"""
    _from_device_like(device_ptr::Ptr{Float64}, len::Int, like::AbstractVector) -> AbstractVector{Float64}

Copy `len` Float64 values from a device pointer, returning the same vector type as `like`.
Fallback: always returns a host Vector (D2H copy).
When CUDA.jl is loaded, the extension overrides for CuVector to return a CuVector (D2D copy).
"""
_from_device_like(device_ptr::Ptr{Float64}, len::Int, ::AbstractVector) = _from_device(device_ptr, len)

"""
    cuda_available() -> Bool

Return `true` if the CUDA backend can be loaded. The CUDA artifact is installed
on demand only when a compatible driver or `MOREAU_CUDA_VERSION` is detected.
"""
function cuda_available()
    return _load_cuda_library()
end
