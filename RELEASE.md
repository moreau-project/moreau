# Releasing Moreau 0.4.0

## Release notes

- Use **Direct Conic Constraint** for constraints of the form `x[J] in K`.
  The API and documentation rename is being prepared in a separate PR and must
  land before the release.
- Correct the row ordering of mixed PSD, exponential, and power cones across CPU
  and CUDA solves, differentiation, equilibration, and chordal decomposition.
- Distribute CPU libraries for Linux x86_64, Linux aarch64, and macOS ARM64, and
  CUDA 12/13 libraries for Linux x86_64 and aarch64.
- Package Moreau.jl independently, using Julia artifacts for native libraries and
  lazy downloads for CUDA. The Julia package has its own version, currently 0.1.0.
- Include C API headers, LICENSE, and NOTICE alongside shared libraries.
- Keep Python metadata, imports, native version banners, dependencies, and lock
  files synchronized for stable, beta, and development builds.

## Local validation (2026-09-04)

- Built and installed the 0.4.0 CPU and wrapper wheels in a fresh Python 3.12 environment.
- Python CPU suites: 869 passed, 401 skipped (CUDA, slow, or optional cases),
  including installed PyTorch and JAX integrations.
- Release tooling: 24 tests passed; workflow syntax and lock-file checks passed.
- Strict Sphinx HTML build passed with external reference inventories available.
- Built the CPU C API library; its version and archive contents passed validation. Scrubbed both wheels and passed metadata and
  hardening validation; 66 targeted tests passed against the scrubbed wheels.
- Julia 1.12.2: 2,761 public/MOI tests and 60 internal integration/differentiation
  tests passed against the new C library via `MOREAU_CPU_LIB`. Artifact downloads
  for stable binaries still require the release build and release QA.
- Stable release matrix builds, GPU runtime QA, and publication remain pending.

## Preparation

Run `./scripts/bump-version.sh 0.4.0` from the repository root. The release workflow
uses the same script with `--pin-dependencies` so every wheel uses matching
Moreau backends. Review the resulting diff and run:

```bash
uv lock --check
pytest scripts/tests/ -q
pytest packages/moreau/tests/python/ packages/moreau-cpu/tests/python/ --device=cpu -rs
sphinx-build -b html -W --keep-going docs docs/_build/html
```

Run the Python tests against freshly built wheels in an isolated environment.
A local CPU build does not validate CUDA or the manylinux/macOS release matrix.

Before final QA, ensure both `gpu-t4` and `gpu-instance` are available. The beta's
full CUDA QA was cancelled while these runners were unavailable. The missing `ghcr.io/moreau-project/moreau/cuda-ci:latest` image was rebuilt
successfully on 2026-09-04 ([image build](https://github.com/moreau-project/moreau/actions/runs/33930637461)).
The previously failing CUDA check on the beta-preparation PR was restarted;
its build passed and its GPU runtime test is queued
([CUDA CI](https://github.com/moreau-project/moreau/actions/runs/33236420586)).
Verify its result before treating CUDA CI as fully resolved. This rerun validates
the earlier beta-preparation commit; the 0.4.0 release commit requires its own CI.

## Build and QA

Commit and review the prepared changes, including the existing Julia packaging
changes and the Direct Conic Constraint API rename. Run the affected tests on
the combined changes and merge both PRs before dispatching the release from `main`:

```bash
gh workflow run release.yml --ref main -f version=0.4.0 -f release_name=v0.4.0
```

The workflow builds eight Python wheels and seven C library archives, generates
an artifact manifest from the actual archives, and creates a GitHub prerelease
for QA. The checked-in Julia manifest intentionally still points at the existing
beta assets; do not substitute stable URLs without regenerating their hashes.
The generated manifest and Julia source tarball contain the new release URLs.

Release QA starts automatically on the release tag. A manual retry must use the
same tag for the workflow and its input:

```bash
gh workflow run test-release.yml --ref v0.4.0 -f release_tag=v0.4.0
```

Require all CPU, CUDA 12, CUDA 13, and Julia jobs to pass. CUDA imports must succeed
on Python 3.12, 3.13, and 3.14. Julia QA must download its CPU artifact without a
`MOREAU_CPU_LIB` override. Linux aarch64 CUDA binaries are build-validated only;
retain that limitation in the release description until ARM GPU runtime QA is
available.

## Publish

After reviewing the complete QA results, publish the existing wheels:

```bash
gh workflow run publish.yml --ref v0.4.0 -f release_tag=v0.4.0
```

Publishing requires the latest QA run for this tag and commit to have succeeded.
It validates wheel metadata and uses the version embedded in the wheels for
PyPI verification. Successful verification advances the `release` docs branch.

After publication and verification, attach the release notes above to the GitHub
release and mark it stable/latest. Copy the generated `moreau-julia-Artifacts.toml`
into the standalone Moreau.jl repository for its separate registration process.
