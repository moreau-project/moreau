#!/usr/bin/env bash
# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0
#
# Build Moreau from source on an NVIDIA DGX Spark (aarch64, GB10, sm_121) and
# run its C++ and Python tests. .github/workflows/ci-spark.yml runs one stage
# per step; `all` runs every stage, for use by hand on the machine.
#
# The machine needs `devenv` with a `cuda12` environment (CUDA 12, GCC, CMake,
# GoogleTest), uv, Rust, and CUDSS_ROOT pointing at a cuDSS 0.7 build for
# CUDA 12. moreau-cuda is pinned to CUDA 12 and cuDSS 0.7 (see its
# pyproject.toml), so torch comes from PyTorch's CUDA 12.9 index, the first
# with Blackwell code.
set -euo pipefail

usage="Usage: $0 {setup|build|test-cpp|test-python|test-upstream|all} WORK_DIR"
case "${1:-}" in
    -h|--help) echo "$usage"; exit 0 ;;
    setup|build|test-cpp|test-python|test-upstream|all) ;;
    *) echo "$usage" >&2; exit 2 ;;
esac
stage=$1
work=${2:?$usage}
mkdir -p "$work"
work=$(cd "$work" && pwd)
source_dir=$(cd "$(dirname "$0")/.." && pwd)

: "${CUDSS_ROOT:?must point at a cuDSS 0.7 build for CUDA 12}"
export MOREAU_CUDA_ARCH=${MOREAU_CUDA_ARCH:-121}
export UV_LINK_MODE=${UV_LINK_MODE:-copy}
export CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}
venv=$work/venv
python=$venv/bin/python
cuda_env=(devenv run cuda12 openblas-omp --)
# devenv run sets its own PATH, so call uv by its full path inside it.
uv=$(command -v uv)

setup() {
    nvidia-smi
    rm -rf "$venv"
    uv venv --python 3.13 "$venv"
    # torch first, so its PyPI build and CUDA 13 libraries never enter.
    uv pip install --python "$python" torch --index-url https://download.pytorch.org/whl/cu129
    uv pip install --python "$python" "maturin>=1.0,<2.0" numpy scipy "pydantic>=2" pytest \
        hypothesis networkx scikit-build-core nanobind "jax[cuda12]" \
        "cvxpy @ git+https://github.com/cvxpy/cvxpy.git@release/1.8.x" \
        "cvxpylayers @ git+https://github.com/cvxpy/cvxpylayers.git@v1.1.0"
}

build() {
    rm -rf "$work/dist"
    (cd "$source_dir/packages/moreau-cpu" &&
        "$venv/bin/maturin" build --release --interpreter "$python" --out "$work/dist")
    # The C++ tests are built in the same CMake build directory as the wheel.
    # Keeping the cuDSS and CUDA library directories in the RUNPATH lets the
    # extension find the cuDSS that isn't installed from pip.
    "${cuda_env[@]}" "$uv" build --wheel --no-build-isolation --python "$python" \
        --out-dir "$work/dist" \
        -C build-dir="$work/cuda-build" \
        -C cmake.define.CUDSS_ROOT="$CUDSS_ROOT" \
        -C cmake.define.CMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
        -C cmake.define.MOREAU_BUILD_TESTS=ON \
        "$source_dir/packages/moreau-cuda"
    # --no-deps keeps the PyPI moreau-cpu and moreau-cuda from replacing these builds.
    uv pip install --python "$python" --no-deps --reinstall \
        "$work"/dist/moreau_cpu-*.whl "$work"/dist/moreau_cuda-*.whl \
        "$source_dir/packages/moreau"
    "$python" -c 'import moreau
print("devices:", moreau.available_devices())
assert moreau.device_available("cuda"), moreau.device_error("cuda")'
}

test_cpp() {
    cd "$work/cuda-build"
    "${cuda_env[@]}" ctest --output-on-failure -j 8
}

test_python() {
    cd "$source_dir"
    "$python" -m pytest packages/moreau/tests/python packages/moreau-cpu/tests/python \
        packages/moreau-cuda/tests/python --tb=short -rs
}

test_upstream() {
    rm -rf "$work/cvxpy" "$work/cvxpylayers"
    git clone -q --depth 1 --branch release/1.8.x https://github.com/cvxpy/cvxpy.git "$work/cvxpy"
    git clone -q --depth 1 --branch v1.1.0 https://github.com/cvxpy/cvxpylayers.git "$work/cvxpylayers"
    # Run cvxpy's tests from outside its source tree so they import the installed package.
    rm -rf "$work/cvxpy-tests"
    cp -r "$work/cvxpy/cvxpy/tests" "$work/cvxpy-tests"
    cd "$work"
    "$python" -m pytest cvxpy-tests/test_conic_solvers.py --tb=short -k Moreau
    "$python" -m pytest cvxpylayers/tests/test_moreau.py \
        cvxpylayers/tests/test_moreau_dual_variables.py --tb=short
}

case "$stage" in
    setup) setup ;;
    build) build ;;
    test-cpp) test_cpp ;;
    test-python) test_python ;;
    test-upstream) test_upstream ;;
    all)
        setup
        build
        status=0
        (test_cpp) || status=1
        (test_python) || status=1
        (test_upstream) || status=1
        exit $status
        ;;
esac
