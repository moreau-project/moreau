using Test
using TOML
using Moreau

@testset "Moreau.jl" begin
    @testset "Artifacts" begin
        artifacts = TOML.parsefile(joinpath(@__DIR__, "..", "Artifacts.toml"))
        @test all(!get(entry, "lazy", false) for entry in artifacts["moreau_cpu"])
        @test all(get(entry, "lazy", false) for entry in artifacts["moreau_cuda12"])
        @test all(get(entry, "lazy", false) for entry in artifacts["moreau_cuda13"])
        @test all(
            occursin("github.com/moreau-project/moreau/releases/download", download["url"])
            for name in ("moreau_cpu", "moreau_cuda12", "moreau_cuda13")
            for entry in artifacts[name]
            for download in entry["download"]
        )
        @test Moreau._choose_device(10, 0) == :cpu
        @test !Moreau._cuda_available[]
    end
    include("test_moi_wrapper.jl")
    include("test_moi_standard.jl")
end
