# Moreau releases

The monorepo owns the native libraries, Python packages, and Julia frontend.
Moreau.jl is registered from `packages/moreau-julia/Moreau.jl` with its existing
name and UUID. All use the same release version and source commit. There is no
standalone frontend synchronization or separate Julia version bump.

Yggdrasil builds and registers `Moreau_CPU_jll` and `Moreau_CUDA_jll`. Its generated
repositories under JuliaBinaryWrappers are build outputs; Moreau does not maintain
parallel source repositories for them.

## One-time GitHub configuration

- Enable the Registrator GitHub app on `moreau-project/moreau`.
- Create a fork of `JuliaPackaging/Yggdrasil` under `moreau-project`, and set
  `YGGDRASIL_FORK` to its full name, such as `moreau-project/Yggdrasil`.
- Create a **fine-grained** personal access token with resource owner
  `moreau-project`, repository access limited to `moreau` and that fork, and
  **Contents: Read and write**. Save it as `JULIA_RELEASE_TOKEN`. This permits
  pushing recipe branches to the fork and commenting on Moreau commits; it does
  not grant write access to other repositories. No classic token or `public_repo`
  scope is needed. Approve the token in the organization if its policy requires it.
  The Registrator caller must be an eligible Moreau collaborator or public
  organization member.
- A maintainer opens the upstream Yggdrasil PRs using the comparison links and
  prepared text from the workflow. The token cannot write to
  `JuliaPackaging/Yggdrasil`, and the workflow never attempts that operation.
- Keep the existing PyPI trusted-publishing configuration. Optional CUDA runtime
  tests use the existing `gpu-t4` and `gpu-instance` runners. CPU and CUDA loading
  tests run on hosted runners.

No settings, tokens, repository permissions, or external repositories are changed
by preparing this branch. The configuration above is required when running the
publication workflows.

## Stable release sequence

1. Run `python scripts/bump_version.py X.Y.Z --pin-dependencies`, review the change,
   and commit it. The script updates Julia, JLL compatibility bounds, and both
   recipe versions alongside Python/native versions. Merge the release changes.
   Stable release dispatch rejects an uncommitted or different frontend/native
   version; the registered package must exist in the tagged source tree.
2. Start the normal release workflow from that commit:

   ```sh
   gh workflow run release.yml --repo moreau-project/moreau --ref main \
     -f version=X.Y.Z -f release_name=vX.Y.Z
   ```

   The workflow builds wheels/C libraries, packages the tracked Julia source,
   and prepares CPU/CUDA Yggdrasil recipes with the full release source commit.
   It attaches `moreau-yggdrasil.tar.gz`, `moreau-julia-release.json`, and the
   registration request text to the candidate release, then dispatches the
   `prepare-jll-prs` stage of `julia-release.yml`. That stage pushes one branch
   per backend to the organization's fork and supplies comparison links, titles,
   and PR bodies in its summary and artifacts. Open the two Yggdrasil PRs in your
   browser, or reuse existing PRs for those branches. Preparing the branches is
   not evidence that either PR has been opened, built, or merged.
3. After Yggdrasil has built, reviewed, merged, and registered both JLLs, resume:

   ```sh
   gh workflow run julia-release.yml --repo moreau-project/moreau --ref vX.Y.Z \
     -f release_tag=vX.Y.Z -f stage=test
   ```

   This checks General for both exact native versions before dispatching release
   QA. `test-release.yml` calls the shared Julia workflow: all seven CPU platform
   variants plus CUDA 12/13 extension loading. With `-f run_gpu_tests=true`, CUDA
   ownership and JuMP/MOI conformance also run on GPU runners and require working
   devices. Python QA remains part of the same release gate. Without GPU runners,
   runtime coverage is explicitly reported as not run; loading checks still run.
4. After the exact tag/commit's QA run succeeds, request frontend registration:

   ```sh
   gh workflow run julia-release.yml --repo moreau-project/moreau --ref vX.Y.Z \
     -f release_tag=vX.Y.Z -f stage=register
   ```

   The workflow verifies the JLLs and latest matching QA, then requests
   `@JuliaRegistrator register subdir=packages/moreau-julia/Moreau.jl` on the
   monorepo release commit. The package remains named **Moreau**. The existing
   native release workflow owns the shared tag; a second TagBot release is not
   required. General/Yggdrasil review and merge remain external stages.
5. After General merges the frontend registration, publish normally:

   ```sh
   gh workflow run publish.yml --repo moreau-project/moreau --ref vX.Y.Z \
     -f release_tag=vX.Y.Z
   ```

   Publication checks successful QA, matching wheel versions, both JLL
   registrations, the monorepo package location, and the exact registered Julia
   tree. It then publishes Python, verifies PyPI installation, updates docs, and
   promotes the GitHub release. Missing registration prevents publication.

All stages can be rerun after an external dependency becomes ready. Do not move an
existing release tag or overwrite a registered package version.

## Initial Julia registration

[General PR #167256](https://github.com/JuliaRegistries/General/pull/167256) is
still an open initial registration for 0.4.0 from the standalone repository.
Retain the name and UUID `c8b129f6-74e5-4f0d-b2d8-2ea10d91a548`, but register the
monorepo subdirectory with the next shared native release, 0.4.1, using the
sequence above. General must point to `moreau-project/moreau` and record
`subdir = "packages/moreau-julia/Moreau.jl"`.

The changed repository and version require a new registration PR. Once that
replacement exists, coordinate closing the old registration with General's
reviewers. Preparing or merging this packaging change does not submit either
registration request.

The native 0.4.0 tag and PyPI packages are already published, and current source
includes later solver fixes. Do not rebuild those changes as 0.4.0, move its tag,
or rerun its PyPI publication. Commit the shared 0.4.1 version bump before
building or registering the next release.

## Development and prerelease builds

Development/prerelease versions continue through the existing native build and
QA pipeline without General or Yggdrasil submission. Julia CPU QA loads the
actual candidate C library through a temporary local dependency shim; the native
version check remains active. These shims are never registered or shipped.
Stable releases always test the real registered JLLs on the full platform matrix.

## Local preparation and verification

```sh
python scripts/julia_release.py prepare --output /tmp/moreau-julia-handoff
python -m pytest scripts/tests/test_release.py scripts/tests/test_julia_release.py -q
actionlint
```

Preparation only writes local handoff files. It does not submit PRs, trigger
registration, or publish anything. The workflow scripts can be validated without
repeating the local native build matrix.

References: [Registrator subdirectory registration](https://github.com/JuliaRegistries/Registrator.jl#registering-a-package-in-a-subdirectory),
[General subdirectory packages](https://github.com/JuliaRegistries/General#how-do-i-move-a-package-into-a-subdirectory-of-a-repository),
[generated JLL packages](https://docs.binarybuilder.org/stable/jll/).
