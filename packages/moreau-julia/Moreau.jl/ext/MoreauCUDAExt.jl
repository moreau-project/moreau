module MoreauCUDAExt

using Moreau
using CUDA

"""
    Moreau._to_device(v::CuVector{Float64}) -> Ptr{Float64}

Extract device pointer from a CuArray — no copy needed, data is already on GPU.
"""
function Moreau._to_device(v::CuVector{Float64})
    return Ptr{Float64}(pointer(v))
end

"""
    Moreau._from_device_like(device_ptr::Ptr{Float64}, len::Int, ::CuVector) -> CuVector{Float64}

Copy `len` Float64 values from a device pointer into a new CuVector (D2D copy).
The returned CuVector owns its own memory, independent of the source pointer.
"""
function Moreau._from_device_like(device_ptr::Ptr{Float64}, len::Int, ::CuVector)
    v = CuVector{Float64}(undef, len)
    if len > 0
        unsafe_copyto!(pointer(v), CUDA.CuPtr{Float64}(UInt(device_ptr)), len)
    end
    return v
end

"""
    Moreau._maybe_free(::CuVector) -> Nothing

No-op: CUDA.jl manages CuArray memory.
"""
function Moreau._maybe_free(::CuVector)
    return nothing
end

end # module
