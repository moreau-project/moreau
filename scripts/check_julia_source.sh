#!/usr/bin/env bash
# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
mkdir "$scratch/tagged" "$scratch/released"
git archive --prefix=Moreau.jl/ HEAD:packages/moreau-julia/Moreau.jl |
    tar -x -C "$scratch/tagged"
tar -xzf "$1" -C "$scratch/released"
diff --no-dereference -r "$scratch/tagged" "$scratch/released"
