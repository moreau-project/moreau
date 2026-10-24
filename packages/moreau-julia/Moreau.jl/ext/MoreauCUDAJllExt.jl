# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0

module MoreauCUDAJllExt

import Libdl
import Moreau
import Moreau_CUDA_jll

function load_cuda_library()
    Moreau_CUDA_jll.is_available() || return false
    # Use the runtime selected by the JLL dependency graph, including user
    # CUDA.jl preferences; never mix it with a separately discovered libcudart.
    runtime = Libdl.dlopen(Moreau_CUDA_jll.CUDA_Runtime_jll.libcudart)
    try
        if !Moreau._try_cuda_library(Moreau_CUDA_jll.libmoreau_cuda)
            Libdl.dlclose(runtime)
            return false
        end
    catch
        Libdl.dlclose(runtime)
        rethrow()
    end
    Moreau._cudart_handle[] = runtime
    Moreau._cudart_loaded[] = true
    return true
end

function __init__()
    Moreau._cuda_jll_loader[] = load_cuda_library
    return
end

end
