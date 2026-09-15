# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0

using BinaryBuilder, Pkg

# This recipe is laid out for M/Moreau/Moreau_CUDA in Yggdrasil.
const YGGDRASIL_DIR = "../../.."
include(joinpath(YGGDRASIL_DIR, "fancy_toys.jl"))
include(joinpath(YGGDRASIL_DIR, "platforms", "cuda.jl"))

name = "Moreau_CUDA"
version = v"0.4.0"
sources = [GitSource("https://github.com/moreau-project/moreau.git",
    "36ea4ead046d00b40a3ab703aeb3099bb99667ca")]
script = raw"""
cd ${WORKSPACE}/srcdir/moreau
install_license LICENSE NOTICE
export TMPDIR=${WORKSPACE}/tmpdir
mkdir -p ${TMPDIR}
export CUDA_HOME=${prefix}/cuda
export PATH=${CUDA_HOME}/bin:${PATH}
export CUDACXX=${CUDA_HOME}/bin/nvcc
ln -s lib ${CUDA_HOME}/lib64
cmake -S packages/moreau-cuda -B build \
    -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} \
    -DCMAKE_BUILD_TYPE=Release \
    -DCUDAToolkit_ROOT=${CUDA_HOME} \
    -DCUDSS_ROOT=${prefix} \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDAARCHS}" \
    -DMOREAU_BUILD_C_SHARED=ON \
    -DMOREAU_BUILD_PYTHON=OFF \
    -DMOREAU_BUILD_TESTS=OFF \
    -DMOREAU_BUILD_EXAMPLES=OFF \
    -DMOREAU_VERSION=0.4.0
cmake --build build --target moreau_cuda_shared -j${nproc}
mkdir -p ${libdir} ${includedir}
cp -a build/libmoreau_cuda.so* ${libdir}/
cp packages/moreau-c/include/moreau.h ${includedir}/
"""
products = [LibraryProduct("libmoreau_cuda", :libmoreau_cuda)]
# Retain 0.7.1: Moreau reverted cuDSS 0.8 after performance and determinism regressions.
dependencies = [
    Dependency("CUDSS_jll", v"0.7.1"; compat="=0.7.1"),
    Dependency("CompilerSupportLibraries_jll"),
]
platforms = CUDA.supported_platforms(; min_version=v"12.2", max_version=v"13.1")
# One toolkit per runtime ABI. Keep both ARM CUDA-12 variants selected by Yggdrasil.
filter!(p -> p["cuda"] in ("12.2", "13.0"), platforms)

for platform in platforms
    should_build_platform(triplet(platform)) || continue
    gpu_archs = CUDA.cuda_gpu_archs(platform)
    # Orin needs sm_87 in addition to the generic architecture list.
    arch(platform) == "aarch64" && push!(gpu_archs, "87")
    archs = join(gpu_archs, ";")
    build_tarballs(ARGS, name, version, sources,
        "export CUDAARCHS=\"$archs\"\n" * script, [platform], products,
        [dependencies; CUDA.required_dependencies(platform; static_sdk=true)];
        julia_compat="1.12", preferred_gcc_version=v"11",
        augment_platform_block=CUDA.augment, lazy_artifacts=true, dont_dlopen=true)
end
