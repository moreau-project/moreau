#!/usr/bin/env bash
# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

version=$(python -c 'import tomllib; print(tomllib.load(open("packages/moreau-julia/Moreau.jl/Project.toml", "rb"))["version"])')
[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
git diff --exit-code -- packages packaging/yggdrasil
commit=$(git rev-parse HEAD)
branch="moreau-${BACKEND,,}-${version}-${commit:0:8}"
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
git show "HEAD:packaging/yggdrasil/$recipe/build_tarballs.jl" |
    sed -E "s/\"[0-9a-f]{40}\"/\"$commit\"/" > "yggdrasil/$recipe/build_tarballs.jl"
git -C yggdrasil add "$recipe"
if ! git -C yggdrasil diff --cached --quiet; then
    git -C yggdrasil commit -m "Build Moreau_${BACKEND} ${version}"
    git -C yggdrasil push origin "HEAD:refs/heads/$branch"
fi

cat > jll-pr-body.md <<EOF
Build Moreau_${BACKEND} ${version} from moreau-project/moreau commit ${commit}.
The native solver and Julia frontend share this release version. The recipe
is prepared by the monorepo release workflow; Yggdrasil supplies the source
builds, audits, generated JLL package, and registration.
EOF
compare="https://github.com/JuliaPackaging/Yggdrasil/compare/master...${fork_owner}:${branch}?expand=1"
cat > jll-pr-request.md <<EOF
Open the [Moreau_${BACKEND} ${version} pull request](${compare}) in your browser.
Use the title **Moreau_${BACKEND} v${version}** and the body below.
If a pull request already exists for this branch, use that existing request.
The workflow pushed only to ${YGGDRASIL_FORK}; it did not open an upstream PR.

---

EOF
cat jll-pr-body.md >> jll-pr-request.md
cat jll-pr-request.md
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    cat jll-pr-request.md >> "$GITHUB_STEP_SUMMARY"
fi
