#!/usr/bin/env bash
# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 {prepare|build|jll-prs|test|gpu|register|publish} X.Y.Z [options]"
    echo "Options: test --run-gpu-tests; gpu --cuda {12,13} --suite {all,python,julia}"
    exit 0
fi
stage=$1 version=$2
shift 2
tag="v$version"
repo=moreau-project/moreau
root=$(cd "$(dirname "$0")/.." && pwd)

workflow() {
    local name=$1 ref=$2
    shift 2
    run_id=$(gh api --method POST "repos/$repo/actions/workflows/$name/dispatches" \
        -H 'X-GitHub-Api-Version: 2026-03-10' -f "ref=$ref" "$@" --jq .workflow_run_id)
    gh run watch "$run_id" --repo "$repo" --exit-status
}

case "$stage" in
    prepare)
        python "$root/scripts/bump_version.py" "$version" --pin-dependencies
        ;;
    build)
        workflow release.yml main -f "inputs[version]=$version" -f "inputs[release_name]=$tag"
        ;;
    test)
        gpu_tests=false
        if [[ ${1:-} == --run-gpu-tests ]]; then gpu_tests=true; fi
        workflow test-release.yml "$tag" -f "inputs[release_tag]=$tag" \
            -F "inputs[run_gpu_tests]=$gpu_tests"
        ;;
    gpu)
        bash "$root/scripts/test_release_gpu.sh" "$tag" "$@"
        ;;
    jll-prs|register)
        if [[ $stage == jll-prs ]]; then action=prepare-jll-prs; else action=prepare-registration; fi
        workflow julia-release.yml "$tag" -f "inputs[release_tag]=$tag" -f "inputs[stage]=$action"
        commit=$(gh api "repos/$repo/commits/$tag" --jq .sha)
        release_tmp=$(mktemp -d)
        trap 'rm -rf "$release_tmp"' EXIT
        if [[ $stage == register ]]; then
            gh run download "$run_id" --repo "$repo" --name julia-registration-request --dir "$release_tmp"
            gh api "repos/$repo/commits/$commit/comments" \
                -F "body=@$release_tmp/registration-body.txt" --jq .html_url
        else
            fork=$(gh variable get YGGDRASIL_FORK --repo "$repo")
            for backend in CPU CUDA; do
                gh run download "$run_id" --repo "$repo" --name "julia-jll-pr-$backend" --dir "$release_tmp/$backend"
                branch="ptn/moreau-${backend,,}-${version}-${commit:0:8}"
                head="${fork%%/*}:$branch"
                existing=$(gh pr list --repo JuliaPackaging/Yggdrasil --head "$head" \
                    --state all --json url --jq '.[0].url // empty')
                if [[ -n $existing ]]; then
                    echo "$existing"
                else
                    gh pr create --repo JuliaPackaging/Yggdrasil --base master --head "$head" \
                        --title "Moreau_$backend v$version" --body-file "$release_tmp/$backend/jll-pr-body.md"
                fi
            done
        fi
        ;;
    publish)
        workflow publish.yml "$tag" -f "inputs[release_tag]=$tag"
        ;;
    *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
