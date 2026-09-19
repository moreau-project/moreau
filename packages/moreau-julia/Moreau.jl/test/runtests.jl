using Test
using Moreau

@testset "Moreau.jl" begin
    @testset "Libraries" begin
        @test Moreau.Moreau_CPU_jll.is_available()
        @test VersionNumber(Moreau.c_moreau_version()) == pkgversion(Moreau)
        @test Moreau._choose_device(10, 0) == :cpu
        @test !Moreau._cuda_available[]
    end
    include("test_c_wrapper.jl")
    include("test_compiled_solver.jl")
    include("test_sensitivities.jl")
    include("test_moi_wrapper.jl")
    include("test_moi_standard.jl")
end
