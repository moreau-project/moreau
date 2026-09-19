# Releases

Run these stages with the release version, using your own `gh` login. Python can
publish before Julia is ready:

```sh
scripts/release.sh prepare X.Y.Z
# Merge the version-bump PR opened by prepare.
scripts/release.sh build X.Y.Z
scripts/release.sh test X.Y.Z
scripts/release.sh gpu X.Y.Z --cuda 12 --suite python
scripts/release.sh gpu X.Y.Z --cuda 13 --suite python
scripts/release.sh publish X.Y.Z
```

Finish Julia independently, using the same release tag:

```sh
scripts/release.sh jll-prs X.Y.Z
# Wait for both JLLs to register in General.
scripts/release.sh test-julia X.Y.Z
scripts/release.sh gpu X.Y.Z --cuda 12 --suite julia
scripts/release.sh gpu X.Y.Z --cuda 13 --suite julia
scripts/release.sh register X.Y.Z
# Wait for General to merge the Moreau registration.
```

The script waits for Actions checks and opens the Yggdrasil PRs and Registrator
comment. Add `--run-gpu-tests` to `test` or `test-julia` to use CI GPU runners
instead of local GPU tests. Local GPU tests require Linux, an NVIDIA GPU, and
`uv` (Python) or Julia 1.12 (Julia); they supplement the required Actions checks.

Local and Actions release GPU runs record `release-gpu/cuda{12,13}/{python,julia}` statuses
on the release tag's commit. A single-suite run updates only that suite's status.
