using Test

@testset "MoreauTests" begin
    include("test_c_wrapper.jl")
    include("test_compiled_solver.jl")
    include("test_chainrules.jl")
    if get(ENV, "MOREAU_TEST_CUDA", "") == "1"
        include("test_cuda.jl")
    end
end
