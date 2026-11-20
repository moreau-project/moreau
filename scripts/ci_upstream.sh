#!/usr/bin/env bash
# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0
# Shared release pins and upstream test setup for CPU, CUDA, and DGX Spark CI.

readonly UPSTREAM_CVXPY_VERSION=1.9.3
readonly UPSTREAM_CVXPY_SHA=fba7e1f41ca5cf0718918d47d02ba2c3614395d2
readonly UPSTREAM_CVXPYLAYERS_VERSION=1.3.0
readonly UPSTREAM_CVXPYLAYERS_SHA=c110a394119477bf573b36fa079fb524058df2cf
UPSTREAM_SCRIPTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
readonly UPSTREAM_SCRIPTS_DIR

prepare_upstream_tests() (
    set -euo pipefail
    local test_root=${1:?upstream test directory required}
    local upstream_python=${2:?Python executable required}

    "$upstream_python" - "$UPSTREAM_CVXPY_VERSION" "$UPSTREAM_CVXPYLAYERS_VERSION" <<'PY'
import sys
from importlib.metadata import version

for package, expected in zip(("cvxpy", "cvxpylayers"), sys.argv[1:]):
    actual = version(package)
    print(f"Installed {package}: {actual}", flush=True)
    if actual != expected:
        raise SystemExit(f"Expected {package}=={expected}, got {actual}")
PY

    mkdir -p "$test_root"
    test_root=$(cd "$test_root" && pwd)
    rm -rf "$test_root/cvxpy" "$test_root/cvxpylayers" "$test_root/cvxpy-tests"
    local package version sha actual
    for package in cvxpy cvxpylayers; do
        if [[ "$package" == cvxpy ]]; then
            version=$UPSTREAM_CVXPY_VERSION
            sha=$UPSTREAM_CVXPY_SHA
        else
            version=$UPSTREAM_CVXPYLAYERS_VERSION
            sha=$UPSTREAM_CVXPYLAYERS_SHA
        fi
        git -c advice.detachedHead=false clone --quiet --depth 1 --branch "v$version" \
            "https://github.com/cvxpy/$package.git" "$test_root/$package"
        actual=$(git -C "$test_root/$package" rev-parse HEAD)
        echo "$package tests: v$version at $actual"
        if [[ "$actual" != "$sha" ]]; then
            echo "Expected $package test commit $sha, got $actual" >&2
            return 1
        fi
    done

    # Backport the merged test-only fix; the installed CVXPY release is unchanged.
    echo "Applying CVXPY #3633 to the v1.9.3 test checkout"
    git -C "$test_root/cvxpy" apply \
        "$UPSTREAM_SCRIPTS_DIR/patches/cvxpy-1.9.3-moreau-presolve.patch"
    # Run outside the checkout so tests import the installed CVXPY package.
    cp -r "$test_root/cvxpy/cvxpy/tests" "$test_root/cvxpy-tests"
)

run_cvxpylayers_tests() (
    set -euo pipefail
    local test_root=${1:?upstream test directory required}
    local upstream_python=${2:?Python executable required}
    shift 2
    local test_file
    # Separate processes release compiled graphs and GPU memory between files.
    for test_file in test_moreau.py test_moreau_dual_variables.py \
        test_moreau_direct_cones.py test_torch_fullgraph.py; do
        "$upstream_python" -m pytest "$test_root/cvxpylayers/tests/$test_file" \
            -v --tb=short "$@"
    done
)
