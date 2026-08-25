using Test

@testset "Moreau.jl" begin
    include("test_moi_wrapper.jl")
    include("test_moi_standard.jl")
end
