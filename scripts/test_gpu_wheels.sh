#!/usr/bin/env bash
# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

usage="Usage: $0 WHEEL_DIR [--cuda {12,13}] [--source-dir DIR] [--work-dir DIR]"
case "${1:-}" in
    '') echo "$usage" >&2; exit 2 ;;
    -h|--help) echo "$usage"; exit 0 ;;
esac
wheels=$(cd "$1" && pwd)
shift
source_dir=$(cd "$(dirname "$0")/.." && pwd)
cuda=12 work=
while (( $# )); do
    case "$1" in
        --cuda) cuda=$2; shift 2 ;;
        --source-dir) source_dir=$2; shift 2 ;;
        --work-dir) work=$2; shift 2 ;;
        -h|--help) echo "$usage"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done
arch=$(uname -m)
case "$cuda:$arch" in
    12:x86_64) torch_index=cu121 ;;
    12:aarch64) torch_index=cu128 ;;
    13:x86_64|13:aarch64) torch_index=cu130 ;;
    *) echo "Unsupported CUDA/architecture: $cuda/$arch" >&2; exit 2 ;;
esac
source_dir=$(cd "$source_dir" && pwd)
work=${work:-$(mktemp -d -t moreau-wheel-qa-XXXXXX)}
mkdir -p "$work"
cd "$work"
work=$PWD
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Require exactly one wheel for each package; never silently substitute PyPI.
shopt -s nullglob
wrapper=("$wheels"/moreau-*-py3-none-any.whl)
cpu=("$wheels"/moreau_cpu-*manylinux*"$arch".whl)
gpu=("$wheels"/moreau_cuda"$cuda"-*manylinux*"$arch".whl)
if (( ${#wrapper[@]} != 1 || ${#cpu[@]} != 1 || ${#gpu[@]} != 1 )); then
    echo "Expected one wrapper, CPU, and CUDA $cuda wheel for $arch" >&2
    exit 1
fi
selected=("${wrapper[@]}" "${cpu[@]}" "${gpu[@]}")

# Check the installed packages normally, including their dependency preloading.
# Missing native extensions or an unavailable CUDA backend are failures.
for version in 3.12 3.13 3.14; do
    uv venv --python "$version" "$work/python$version"
    python="$work/python$version/bin/python"
    uv pip install --python "$python" "${selected[@]}"
    "$python" - <<'PY'
import sys
import moreau
import moreau_cpu._cpu_solver
import moreau_cuda

assert moreau.device_available("cuda"), moreau.device_error("cuda")
print(f"Python {sys.version.split()[0]}: CPU/CUDA wheel imports passed")
PY
done
python="$work/python3.12/bin/python"
uv pip install --python "$python" torch \
    --default-index "https://download.pytorch.org/whl/$torch_index"
uv pip install --python "$python" "jax[cuda$cuda]" pytest numpy scipy hypothesis \
    cvxpy clarabel git+https://github.com/cvxpy/cvxpylayers.git@v1.1.0
"$python" - <<'PY'
import moreau
import torch
import jax
from moreau.torch import Solver
from moreau_cuda.jax import ffi_available

assert moreau.device_available("cuda"), moreau.device_error("cuda")
assert torch.cuda.is_available(), "PyTorch CUDA is unavailable"
assert jax.default_backend() == "gpu", "JAX CUDA is unavailable"
assert ffi_available(), "Moreau JAX FFI extension is unavailable"
PY

# Separate processes release GPU memory between files. Run every file even if
# one fails, while permitting pytest's no-tests-collected code for optional APIs.
tests=("$source_dir"/packages/moreau/tests/python/test_*.py)
(( ${#tests[@]} )) || { echo "No Moreau tests found in $source_dir" >&2; exit 1; }
failed=()
for test in "${tests[@]}"; do
    rc=0
    "$python" -m pytest "$test" -v --tb=short -rs || rc=$?
    if (( rc != 0 && rc != 5 )); then
        failed+=("$test")
    fi
done
if (( ${#failed[@]} )); then
    printf 'Failed test file: %s\n' "${failed[@]}" >&2
    exit 1
fi

git clone --depth=1 --branch release/1.8.x https://github.com/cvxpy/cvxpy.git cvxpy
git clone --depth=1 --branch v1.1.0 https://github.com/cvxpy/cvxpylayers.git cvxpylayers
# Copy CVXPY tests away from its checkout to use the installed package.
cp -r cvxpy/cvxpy/tests cvxpy-tests
"$python" -m pytest cvxpy-tests/test_conic_solvers.py -v --tb=short -rs -k Moreau
"$python" -m pytest cvxpylayers/tests/test_moreau.py \
    cvxpylayers/tests/test_moreau_dual_variables.py -v --tb=short -rs
