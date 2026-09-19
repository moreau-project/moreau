# Moreau Julia binary packaging

The recipes under `yggdrasil/M/Moreau/` build the CPU and CUDA C interfaces.
For each release, first commit the version bump, then run
`python scripts/julia_release.py prepare --output /tmp/moreau-julia-handoff`.
This writes recipe copies pinned to that exact source commit. Copy the generated
`yggdrasil/M/Moreau/` directories into the same paths in a Yggdrasil checkout.
The CUDA recipe uses Yggdrasil's shared
`platforms/cuda.jl`, `C/CUDA/common.jl`, and `fancy_toys.jl` helpers.

## Package identities and versions

The Julia frontend remains **Moreau**, with UUID
`c8b129f6-74e5-4f0d-b2d8-2ea10d91a548`. Its version matches the native release.
The BinaryBuilder-generated packages are:

| Package | Generated UUID | Source version |
| --- | --- | --- |
| Moreau_CPU_jll | c0bfae29-27af-5321-a17d-6bf033ce7ea7 | 0.4.0 |
| Moreau_CUDA_jll | c77365f8-1b8a-5c63-97c9-c9362ae64116 | 0.4.0 |

These UUIDs come from BinaryBuilder's generator, and differ from the UUIDs of
the former hand-maintained JLL repositories. A JLL suffix such as `+0` is a
binary rebuild number for the same native release. Moreau.jl pins the native
release in its JLL compatibility bounds and checks the loaded library's version.

The CPU recipe targets x86-64 and ARM64 glibc/musl Linux, Intel/Apple Silicon
macOS, and x86-64 Windows. It retains the native default features, including
FAER and the active-set solver. The native build selects the C++ runtime for the **target** platform when
cross-compiling.

The CUDA recipe targets Linux x86-64 and ARM64, CUDA 12.2 and 13.0. CUDA 12
has separate Jetson and SBSA platform variants. CUDA platform augmentation
uses the runtime selected by CUDA.jl. cuDSS remains pinned to 0.7.1; the 0.8
upgrade was reverted upstream for performance and determinism regressions.

## Build and validation

Run BinaryBuilder with Julia 1.10 (validated with BinaryBuilder 0.6.6); use
Julia 1.12 for Moreau.jl and its tests.
BinaryBuilder's temporary dependency environment can select a TOML version that
is incompatible with Julia 1.12's package-manager precompilation.

From each recipe directory, with BinaryBuilder and JSON3 available:

```sh
julia +1.10 --project=/path/to/binarybuilder-env build_tarballs.jl --verbose --deploy=local
```

`--deploy=local` generates local wrappers and does not publish or register them.
An optional platform triplet limits the build, for example `x86_64-linux-gnu`
for CPU or `x86_64-linux-gnu-cuda+13.0` for CUDA. Use a directory outside an
encrypted home directory. The host must support BinaryBuilder containers.

Test Moreau.jl against the locally generated JLLs before submitting the recipes.
The public package suite covers native calls, batching, MOI, and numerical
pullbacks; `test/cuda/runtests.jl` additionally covers CuVector transfers, device
solutions, and native gradients. Set `MOREAU_TEST_CUDA=1` to require GPU tests
instead of recording an unavailable device as a skip.

For the legacy `MoreauTests.jl` GPU conformance suite, add `Moreau_CUDA_jll`
to that test environment before setting `MOREAU_TEST_CUDA=1`. Its CPU tests
do not require the CUDA package.

## Release integration

See [the release runbook](../RELEASE.md). Moreau.jl is registered directly from
`packages/moreau-julia/Moreau.jl` in the monorepo. No frontend repository sync is
needed. `bump_version.py` updates the frontend, JLL bounds, and recipe versions;
`julia_release.py prepare` pins both recipes to the exact release commit.

The release workflow prepares recipe branches in the organization's Yggdrasil
fork through `julia-release.yml`, using a fine-grained token limited to that fork
and the Moreau monorepo. A maintainer opens the upstream PRs from the generated
comparison links and PR text. After JLL registration, release QA runs the shared
Julia platform matrix. Frontend registration targets the tested monorepo
subdirectory. Stable publication requires
General to contain that exact package tree and both matching JLL versions.

Local source builds passed BinaryBuilder audits for all five non-macOS CPU
targets and both CUDA 13 architectures. The source-built x86-64 Linux libraries
also passed the CPU package suite and GPU regression/conformance suites.
The remaining CUDA 12 and macOS build validation belongs in Yggdrasil CI;
native Windows and ARM execution remains for platform CI before publication.
