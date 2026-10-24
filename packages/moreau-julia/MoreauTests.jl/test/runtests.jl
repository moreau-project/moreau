using Test
using Moreau

@testset "MoreauTests" begin
    native_tests = joinpath(dirname(pathof(Moreau)), "..", "test")
    include(joinpath(native_tests, "test_c_wrapper.jl"))
    include(joinpath(native_tests, "test_compiled_solver.jl"))
    include(joinpath(native_tests, "test_sensitivities.jl"))
    if get(ENV, "MOREAU_TEST_CUDA", "") == "1"
        include("test_cuda.jl")
    end
end
