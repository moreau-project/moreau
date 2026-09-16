#!/usr/bin/env bash
# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

version=$(python -c 'import json; print(json.load(open("julia-handoff/moreau-julia-release.json"))["version"])')
commit=$(python -c 'import json; print(json.load(open("julia-handoff/moreau-julia-release.json"))["source_commit"])')
branch="codex/moreau-${BACKEND,,}-${version}-${commit:0:8}"
recipe="M/Moreau/Moreau_${BACKEND}"
fork_owner="${YGGDRASIL_FORK%%/*}"

git -C yggdrasil config user.name 'Moreau release automation'
git -C yggdrasil config user.email 'actions@users.noreply.github.com'
if git -C yggdrasil ls-remote --exit-code --heads origin "$branch" >/dev/null; then
    git -C yggdrasil fetch origin "$branch"
    git -C yggdrasil checkout -B "$branch" FETCH_HEAD
else
    git -C yggdrasil fetch https://github.com/JuliaPackaging/Yggdrasil.git master
    git -C yggdrasil checkout -b "$branch" FETCH_HEAD
fi
mkdir -p "yggdrasil/$recipe"
cp -a "julia-handoff/yggdrasil/$recipe/." "yggdrasil/$recipe/"
git -C yggdrasil add "$recipe"
if ! git -C yggdrasil diff --cached --quiet; then
    git -C yggdrasil commit -m "ChatGPT generated: build Moreau_${BACKEND} ${version}"
    git -C yggdrasil push origin "HEAD:refs/heads/$branch"
fi

existing=$(gh pr list --repo JuliaPackaging/Yggdrasil --state open \
    --head "$fork_owner:$branch" --json url --jq '.[0].url // empty')
if [ -n "$existing" ]; then
    echo "$existing"
else
    cat > jll-pr-body.md <<EOF
ChatGPT generated:

Build Moreau_${BACKEND} ${version} from moreau-project/moreau commit ${commit}.
The native solver and Julia frontend share this release version. The recipe
is prepared by the monorepo release workflow; Yggdrasil supplies the source
builds, audits, generated JLL package, and registration.
EOF
    gh pr create --repo JuliaPackaging/Yggdrasil --base master --head "$fork_owner:$branch" \
        --title "ChatGPT generated: Moreau_${BACKEND} v${version}" --body-file jll-pr-body.md
fi
