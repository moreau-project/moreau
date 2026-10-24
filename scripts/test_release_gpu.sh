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
    12) julia_cuda=12.2 ;;
    13) julia_cuda=13.0 ;;
    *) echo '--cuda must be 12 or 13' >&2; exit 2 ;;
esac
case "$suite" in
    all|python|julia) ;;
    *) echo '--suite must be all, python, or julia' >&2; exit 2 ;;
esac

root=$(cd "$(dirname "$0")/.." && pwd)
repo=moreau-project/moreau
commit=$(gh api "repos/$repo/commits/$tag" --jq .sha)
arch=$(uname -m)
report_status() {
    gh api --method POST "repos/$repo/statuses/$commit" --silent \
        -f "context=release-gpu/cuda$cuda/$testing_suite" -f "state=$1" \
        -f "description=Local $arch CUDA $cuda $testing_suite tests: $1"
}

work=${work:-$(mktemp -d -t moreau-gpu-XXXXXX)}
mkdir -p "$work"
cd "$work"
work=$PWD
echo "GPU test environment: $work"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

if [[ $suite != julia ]]; then
    testing_suite=python
    report_status pending
    trap 'report_status failure' EXIT
    gh release download "$tag" --repo moreau-project/moreau --dir wheels \
        --pattern 'moreau-*-py3-none-any.whl' \
        --pattern "moreau_cpu-*-manylinux*_$arch.whl" \
        --pattern "moreau_cuda$cuda-*-manylinux*_$arch.whl"
    bash "$root/scripts/test_gpu_wheels.sh" "$work/wheels" --cuda "$cuda" \
        --work-dir "$work/python-qa"
    report_status success
    trap - EXIT
fi

if [[ $suite != python ]]; then
    testing_suite=julia
    report_status pending
    trap 'report_status failure' EXIT
    git init -q source
    git -C source fetch --depth=1 https://github.com/moreau-project/moreau.git "$commit"
    git -C source checkout --detach FETCH_HEAD
    julia --startup-file=no "$root/scripts/test_release_gpu_julia.jl" \
        "$work/source" "$julia_cuda" true
    report_status success
    trap - EXIT
fi
echo "GPU results: https://github.com/$repo/commit/$commit"
