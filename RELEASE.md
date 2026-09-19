# Releases

Replace `X.Y.Z` with the stable release version.

1. Run `python scripts/bump_version.py X.Y.Z --pin-dependencies` and merge the
   version bump into `main`.
2. Build the release:

   ```sh
   gh workflow run release.yml --repo moreau-project/moreau --ref main \
     -f version=X.Y.Z -f release_name=vX.Y.Z
   ```

3. Open the CPU and CUDA Yggdrasil PRs using the links in the Julia Release
   workflow summary. Wait for both JLLs to be registered.
4. Run release QA (add `-f run_gpu_tests=true` for GPU runtime tests):

   ```sh
   gh workflow run julia-release.yml --repo moreau-project/moreau --ref vX.Y.Z \
     -f release_tag=vX.Y.Z -f stage=test
   ```

5. After QA passes, prepare Julia registration:

   ```sh
   gh workflow run julia-release.yml --repo moreau-project/moreau --ref vX.Y.Z \
     -f release_tag=vX.Y.Z -f stage=prepare-registration
   ```

   Post the generated Registrator comment on the linked commit. Wait for General
   to merge the registration.
6. Publish:

   ```sh
   gh workflow run publish.yml --repo moreau-project/moreau --ref vX.Y.Z \
     -f release_tag=vX.Y.Z
   ```
