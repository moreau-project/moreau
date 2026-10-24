# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0

using Pkg, TOML

length(ARGS) == 3 || error("Usage: test_release_gpu_julia.jl SOURCE CUDA_VERSION REQUIRE_GPU")
source = abspath(ARGS[1])
cuda_version = ARGS[2]
require_gpu = parse(Bool, ARGS[3])
package = joinpath(source, "packages", "moreau-julia", "Moreau.jl")
environment = joinpath(package, "test", "cuda")

open(joinpath(environment, "LocalPreferences.toml"), "w") do io
    TOML.print(io, Dict("CUDA_Runtime_jll" => Dict("version" => cuda_version, "local" => "false")))
end
Pkg.activate(environment)
Pkg.develop(path=package)
Pkg.instantiate()
require_gpu && Pkg.add("JuMP")

ENV["MOREAU_TEST_CUDA"] = require_gpu ? "1" : "0"
include(joinpath(environment, "runtests.jl"))
if require_gpu
    include(joinpath(source, "packages", "moreau-julia", "MoreauTests.jl", "test", "test_cuda.jl"))
end
