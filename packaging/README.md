# Moreau Julia binary packaging

The recipes under `yggdrasil/M/Moreau/` build the CPU and CUDA C interfaces
from the immutable Moreau 0.4.0 source commit. Copy these directories into the
same paths in a Yggdrasil checkout. The CUDA recipe uses Yggdrasil's shared
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
FAER and the active-set solver. The bundled patch selects the C++ runtime for
the **target** platform when cross-compiling.

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

## Release order

1. Build and validate native release artifacts and the matching JLL recipes.
2. Publish/register the generated JLLs through the Yggdrasil process.
3. Run Moreau's release QA against those JLLs. Julia QA must pass before the
   native release is promoted by the publishing workflow.
4. Register the matching Moreau.jl version using its existing name and UUID.

For the pending General PR #167256, retain version **0.4.0** and register the
corrected commit after the JLL dependencies are available. Updating the existing
registration then avoids creating a new package or version registration.

Local source builds passed BinaryBuilder audits for all five non-macOS CPU
targets and both CUDA 13 architectures. The source-built x86-64 Linux libraries
also passed the CPU package suite and GPU regression/conformance suites.
The remaining CUDA 12 and macOS build validation belongs in Yggdrasil CI;
native Windows and ARM execution remains for platform CI before publication.
