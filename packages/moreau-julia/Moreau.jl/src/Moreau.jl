module Moreau

import MathOptInterface as MOI
import SparseArrays: SparseMatrixCSC, sparse, findnz, nnz, spzeros
import LinearAlgebra: Diagonal, diag, dot
import Libdl
import Moreau_CPU_jll: libmoreau
import Moreau_CUDA_jll

include("cuda_mem.jl")
include("c_wrapper.jl")
include("MOI_wrapper.jl")

export moreau_solve, CompiledSolver, Solution, BatchedSolution,
       setup!, solve!, backward!, setup_and_solve!, destroy!,
       cuda_available, set_default_device

end # module
