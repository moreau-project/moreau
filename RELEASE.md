# Releases

Run these stages with the release version, using your own `gh` login:

```sh
scripts/release.sh prepare X.Y.Z
# Merge the version-bump PR opened by prepare.
scripts/release.sh build X.Y.Z
scripts/release.sh jll-prs X.Y.Z
# Wait for both JLLs to register in General.
scripts/release.sh test X.Y.Z
scripts/release.sh register X.Y.Z
# Wait for General to merge the Moreau registration.
scripts/release.sh publish X.Y.Z
```

The script waits for Actions checks and handles the Yggdrasil PRs and Registrator
comment. Add `--run-gpu-tests` to `test` to use CI GPU runners. For local GPU tests:

```sh
scripts/release.sh gpu X.Y.Z --cuda 12
scripts/release.sh gpu X.Y.Z --cuda 13
```

Local GPU tests require Linux, `uv`, Julia 1.12, and an NVIDIA GPU. They do not
replace the required Actions checks. Use `--suite python` or `--suite julia` to
run one suite.
