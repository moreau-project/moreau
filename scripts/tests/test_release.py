"""Copyright, the Moreau authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Regression tests for release versioning and wheel metadata validation."""

import zipfile

import bump_version as bump
import pytest
import validate_release_wheels as validate


def snapshot(root):
    return {
        str(path.relative_to(root)): path.read_text() for path in root.rglob("*") if path.is_file()
    }


def test_beta_to_stable_and_idempotence(release_tree):
    bump.bump_version(release_tree, "0.4.0-beta.1")
    assert 'version = "0.4.0b1"' in (release_tree / "uv.lock").read_text()
    bump.bump_version(release_tree, "0.4.0")
    stable = snapshot(release_tree)
    assert all("0.4.0-beta.1" not in text and "0.4.0b1" not in text for text in stable.values())
    for package in ("Moreau.jl", "MoreauTests.jl"):
        assert 'version = "0.4.0"' in stable[f"packages/moreau-julia/{package}/Project.toml"]
    assert 'content: "v0.4.0"' in stable["docs/_static/custom.css"]
    assert 'moreau-cpu>=0.4.0"' in stable["packages/moreau/pyproject.toml"]
    bump.bump_version(release_tree, "0.4.0")
    assert snapshot(release_tree) == stable


@pytest.mark.parametrize("version", ["1.2.3", "1.2.3-beta.4", "1.2.3.dev1", "1.2.3-beta.4.dev1"])
def test_versions_and_exact_dependencies(release_tree, version):
    bump.bump_version(release_tree, version, pin_dependencies=True)
    files = snapshot(release_tree)
    native_version = version.replace(".dev", "-dev")
    for path in (
        "packages/moreau-cpu/Cargo.toml",
        "packages/moreau-julia/Moreau.jl/Project.toml",
        "packages/moreau-julia/MoreauTests.jl/Project.toml",
    ):
        assert f'version = "{native_version}"' in files[path]
    assert f'moreau-cpu=={version}"' in files["packages/moreau/pyproject.toml"]
    assert f'moreau=={version}"' in files["packages/moreau-cuda/pyproject.toml"]
    assert f'__version__ = "{version}"' in files["packages/moreau-cuda/moreau_cuda/__init__.py"]
    assert f'MOREAU_VERSION = "{version}"' in files["packages/moreau-cuda/src/solver/info.cpp"]
    for backend in ("CPU", "CUDA"):
        assert (
            f'Moreau_{backend}_jll = "=1.2.3"'
            in files["packages/moreau-julia/Moreau.jl/Project.toml"]
        )
        recipe = files[f"packaging/yggdrasil/M/Moreau/Moreau_{backend}/build_tarballs.jl"]
        assert f'version = v"{native_version}"' in recipe
        if backend == "CUDA":
            assert f"-DMOREAU_VERSION={version}" in recipe
    assert (
        'Moreau_CUDA_jll = "=1.2.3"'
        in files["packages/moreau-julia/Moreau.jl/test/cuda/Project.toml"]
    )


def test_invalid_version_does_not_modify_files(release_tree):
    before = snapshot(release_tree)
    with pytest.raises(ValueError):
        bump.bump_version(release_tree, '0.4.0"\ninvalid')
    assert snapshot(release_tree) == before


def wheel(tmp_path, name, version, platform="any", *, abi=None, pinned=True):
    interpreter, abi_tag = abi or (
        ("py3", "none")
        if name == "moreau"
        else ("cp39", "abi3") if name == "moreau_cpu" else ("cp312", "abi3")
    )
    path = tmp_path / f"{name}-{version}-{interpreter}-{abi_tag}-{platform}.whl"
    backends = (
        ("moreau-cpu", "moreau-cuda12", "moreau-cuda13")
        if name == "moreau"
        else ("moreau",) if name.startswith("moreau_cuda") else ()
    )
    requirements = "".join(
        f"Requires-Dist: {backend}{'==' if pinned else '>='}{version}\n" for backend in backends
    )
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(
            f"{name}-{version}.dist-info/METADATA",
            f"Name: {name.replace('_', '-')}\nVersion: {version}\n{requirements}",
        )
        zf.writestr(
            f"{name}-{version}.dist-info/WHEEL",
            f"Wheel-Version: 1.0\nTag: {interpreter}-{abi_tag}-{platform}\n",
        )
    return path


def test_release_version_comes_from_wheels(tmp_path):
    wheels = [wheel(tmp_path, name, "0.4.0.dev20260904") for name in ["moreau", "moreau_cpu"]]
    assert validate.release_version(wheels) == "0.4.0.dev20260904"


def test_mixed_release_versions_are_rejected(tmp_path):
    wheels = [wheel(tmp_path, "moreau", "0.4.0"), wheel(tmp_path, "moreau_cpu", "0.4.0b1")]
    with pytest.raises(ValueError, match="Expected one release version"):
        validate.release_version(wheels)


def test_incomplete_release_matrix_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Incomplete release wheel matrix"):
        validate.release_version([wheel(tmp_path, "moreau", "0.4.0")], require_complete=True)


@pytest.fixture
def wheels(tmp_path):
    wheels = [wheel(tmp_path, "moreau", "0.4.0")]
    for platform in ["manylinux_2_28_x86_64", "manylinux_2_28_aarch64", "macosx_11_0_arm64"]:
        wheels.append(wheel(tmp_path, "moreau_cpu", "0.4.0", platform))
        if platform.startswith("manylinux"):
            for name in ["moreau_cuda12", "moreau_cuda13"]:
                wheels.append(wheel(tmp_path, name, "0.4.0", platform))
    return wheels


def test_complete_release_matrix_is_accepted(wheels):
    assert validate.release_version(wheels, require_complete=True) == "0.4.0"


def test_wheel_filename_metadata_mismatch_is_rejected(tmp_path):
    path = wheel(tmp_path, "moreau", "0.4.0")
    renamed = path.rename(tmp_path / "moreau-0.4.1-py3-none-any.whl")
    with pytest.raises(ValueError, match="disagree"):
        validate.release_version([renamed])


@pytest.mark.parametrize("abi", [("cp312", "cp312"), ("cp310", "abi3")])
def test_incorrect_cpu_abi_is_rejected(tmp_path, wheels, abi):
    wheels[1].unlink()
    wheels[1] = wheel(tmp_path, "moreau_cpu", "0.4.0", "manylinux_2_28_x86_64", abi=abi)
    with pytest.raises(ValueError, match="platform/ABI"):
        validate.release_version(wheels, require_complete=True)


def test_duplicate_architecture_cannot_mask_missing_wheel(wheels):
    # Keep eight wheels, but replace the macOS CPU wheel with another Linux wheel.
    wheels[-1] = wheels[1]
    with pytest.raises(ValueError, match="platform/ABI"):
        validate.release_version(wheels, require_complete=True)


def test_release_backend_dependencies_must_be_exact(tmp_path, wheels):
    wheels[0] = wheel(tmp_path, "moreau", "0.4.0", pinned=False)
    with pytest.raises(ValueError, match="must be pinned"):
        validate.release_version(wheels, require_complete=True)


def test_wheel_licenses_are_required(tmp_path):
    path = wheel(tmp_path, "moreau", "0.4.0")
    assert validate.has_wheel_errors(path)
    with zipfile.ZipFile(path, "a") as archive:
        for name in ("LICENSE", "NOTICE"):
            archive.writestr(f"moreau-0.4.0.dist-info/licenses/{name}", "license text")
    assert not validate.has_wheel_errors(path)


def test_wheel_tag_metadata_must_match_filename(tmp_path, wheels):
    wheels[1] = wheels[1].rename(tmp_path / "moreau_cpu-0.4.0-cp310-abi3-manylinux_2_28_x86_64.whl")
    with pytest.raises(ValueError, match="WHEEL tags disagree"):
        validate.release_version(wheels, require_complete=True)
