#!/usr/bin/env bash
# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage="Usage: $0 TAG [--cuda {12,13}] [--suite {all,python,julia}] [--work-dir DIR]"
case "${1:-}" in
    '') echo "$usage" >&2; exit 2 ;;
    -h|--help) echo "$usage"; exit 0 ;;
esac
tag=$1
shift
cuda=12 suite=all work=
while (( $# )); do
    case "$1" in
        --cuda) cuda=$2; shift 2 ;;
        --suite) suite=$2; shift 2 ;;
        --work-dir) work=$2; shift 2 ;;
        -h|--help) echo "$usage"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
case "$cuda" in
    12) torch_index=cu121; julia_cuda=12.2 ;;
    13) torch_index=cu130; julia_cuda=13.0 ;;
    *) echo '--cuda must be 12 or 13' >&2; exit 2 ;;
esac
case "$suite" in
    all|python|julia) ;;
    *) echo '--suite must be all, python, or julia' >&2; exit 2 ;;
esac

root=$(cd "$(dirname "$0")/.." && pwd)
work=${work:-$(mktemp -d -t moreau-gpu-XXXXXX)}
mkdir -p "$work"
cd "$work"
work=$PWD
echo "GPU test environment: $work"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

git init -q source
git -C source fetch --depth=1 https://github.com/moreau-project/moreau.git "refs/tags/$tag"
git -C source checkout --detach FETCH_HEAD

if [[ $suite != julia ]]; then
    arch=$(uname -m)
    gh release download "$tag" --repo moreau-project/moreau --dir wheels \
        --pattern 'moreau-*-py3-none-any.whl' \
        --pattern "moreau_cpu-*-manylinux*_$arch.whl" \
        --pattern "moreau_cuda$cuda-*-manylinux*_$arch.whl"
    uv venv --python 3.12 python
    python="$work/python/bin/python"
    uv pip install --python "$python" wheels/*.whl
    uv pip install --python "$python" torch \
        --default-index "https://download.pytorch.org/whl/$torch_index"
    uv pip install --python "$python" "jax[cuda$cuda]" pytest numpy scipy hypothesis \
        cvxpy clarabel git+https://github.com/cvxpy/cvxpylayers.git@v1.1.0
    "$python" - <<'PY'
import moreau, torch, jax
assert moreau.device_available('cuda'), moreau.device_error('cuda')
assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable'
assert jax.default_backend() == 'gpu', 'JAX CUDA is unavailable'
PY
    # Separate processes release GPU memory between test files, as in CI.
    for test in source/packages/moreau/tests/python/test_*.py; do
        "$python" -m pytest "$test" -v --tb=short -rs
    done
    git clone --depth=1 --branch release/1.8.x https://github.com/cvxpy/cvxpy.git cvxpy
    git clone --depth=1 --branch v1.1.0 https://github.com/cvxpy/cvxpylayers.git cvxpylayers
    # Copy CVXPY tests away from its checkout to use the installed package.
    cp -r cvxpy/cvxpy/tests cvxpy-tests
    "$python" -m pytest cvxpy-tests/test_conic_solvers.py -v -rs -k Moreau
    "$python" -m pytest cvxpylayers/tests/test_moreau.py \
        cvxpylayers/tests/test_moreau_dual_variables.py -v -rs
fi

if [[ $suite != python ]]; then
    julia --startup-file=no "$root/scripts/test_release_gpu_julia.jl" \
        "$work/source" "$julia_cuda" true
fi
