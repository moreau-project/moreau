# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0

using CUDA
using Moreau
using Moreau_CUDA_jll
using SparseArrays
using Test

@testset "CUDA extension" begin
    @test Base.get_extension(Moreau, :MoreauCUDAExt) !== nothing
    @test Base.get_extension(Moreau, :MoreauCUDAJllExt) !== nothing
    @test !Moreau._cuda_available[] # Backend activation remains lazy after extension imports.
    @test which(Moreau._maybe_free_ptr,
        Tuple{Ptr{Float64},CuVector{Float64}}).module ===
        Base.get_extension(Moreau, :MoreauCUDAExt)
    functional = CUDA.functional()
    required = get(ENV, "MOREAU_TEST_CUDA", "0") == "1"
    required && !functional && error("MOREAU_TEST_CUDA=1 requires a working CUDA device")
    if !functional
        @test_skip functional
    else
        @test Moreau.cuda_available()
        host = [2.0, -3.0]
        device = CuVector(host)
        @test Moreau._to_device(device) == Ptr{Float64}(UInt(pointer(device)))
        copy = Moreau._from_device_like(Moreau._to_device(device), 2, device)
        @test copy isa CuVector{Float64}
        @test Array(copy) == host
        copyto!(copy, zeros(length(host)))
        @test Array(device) == host
        @test Moreau._maybe_free_ptr(Moreau._to_device(device), device) === nothing
        @test isempty(Moreau._from_device_like(Ptr{Float64}(C_NULL), 0, device))

        q_gpu, b_gpu = CuVector([-1.0]), CuVector([6.0])
        one_shot = moreau_solve(1, 1, sparse([2.0;;]), sparse([3.0;;]), q_gpu, b_gpu;
            zero_cones=1, device=:cuda)
        @test Array(one_shot.x) ≈ [2.0] atol=1e-5
        @test Array(q_gpu) == [-1.0]
        @test Array(b_gpu) == [6.0]
        warm = moreau_solve(1, 1, sparse([2.0;;]), sparse([3.0;;]), q_gpu, b_gpu;
            zero_cones=1, device=:cuda,
            warm_x=one_shot.x, warm_z=one_shot.z, warm_s=one_shot.s)
        @test Array(warm.x) ≈ [2.0] atol=1e-5
        @test Array(one_shot.x) ≈ [2.0] atol=1e-5

        solver = CompiledSolver(1, 1, sparse([2.0;;]), sparse([3.0;;]);
            zero_cones=1, device=:cuda, enable_grad=true)
        try
            p_gpu, a_gpu = CuVector([2.0]), CuVector([3.0])
            setup!(solver, p_gpu, a_gpu)
            @test Array(p_gpu) == [2.0]
            @test Array(a_gpu) == [3.0]
            sol = solve!(solver, CuVector([-1.0]), CuVector([6.0]))
            @test sol.x isa CuVector{Float64}
            @test Array(sol.x) ≈ [2.0] atol=1e-5
            @test Array(sol.z) ≈ [-1.0] atol=1e-5
            dx, dz = CuVector([0.7]), CuVector([-0.3])
            grads = backward!(solver, dx; dz)
            @test Array(dx) == [0.7]
            @test Array(dz) == [-0.3]
            @test backward!(solver, dx; dz).dq ≈ grads.dq atol=2e-5
            @test Array(grads.dq) ≈ [0.1] atol=2e-5
            @test Array(grads.db) ≈ [0.3] atol=2e-5
            # Warm starts may use a different array type from q and b.
            host_sol = solve!(solver, [-1.0], [6.0]; warm_start=sol)
            @test host_sol.x ≈ [2.0] atol=1e-5
            @test Array(sol.x) ≈ [2.0] atol=1e-5
            device_sol = solve!(solver, q_gpu, b_gpu; warm_start=host_sol)
            @test Array(device_sol.x) ≈ [2.0] atol=1e-5
            @test Array(q_gpu) == [-1.0]
            @test Array(b_gpu) == [6.0]
            CUDA.synchronize()
        finally
            destroy!(solver)
        end
    end
end
