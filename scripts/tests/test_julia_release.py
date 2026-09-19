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

Tests for release identity, source provenance, registry readiness, and QA gates.
"""

import io
import os
import subprocess
import tarfile
import tomllib

import bump_version
import julia_release as release
import pytest


@pytest.fixture
def source(release_tree):
    bump_version.bump_version(release_tree, "1.2.3", pin_dependencies=True)
    release.git(release_tree, "init", "-q")
    for key, value in {
        "user.name": "Test",
        "user.email": "test@example.invalid",
        "commit.gpgsign": "false",
        "core.hooksPath": "/dev/null",
    }.items():
        release.git(release_tree, "config", key, value)
    release.git(release_tree, "add", ".")
    release.git(release_tree, "commit", "-qm", "release fixture")
    return release_tree


@pytest.fixture
def registry(source, tmp_path):
    registry = tmp_path / "registry"
    tree = release.release_info(source)["julia_tree"]
    for name, uuid in {"Moreau": release.UUID, **release.JLLS}.items():
        directory = registry / ("M/Moreau" if name == "Moreau" else f"jll/M/{name}")
        directory.mkdir(parents=True)
        (directory / "Package.toml").write_text(
            f'name = "{name}"\nuuid = "{uuid}"\n'
            f'repo = "{release.REPO}"\nsubdir = "{release.PACKAGE}"\n'
        )
        version = "1.2.3" if name == "Moreau" else "1.2.3+2"
        (directory / "Versions.toml").write_text(f'["{version}"]\ngit-tree-sha1 = "{tree}"\n')
    return registry


def test_recipes_are_pinned_to_the_release_commit(source, tmp_path):
    output = tmp_path / "handoff"
    junk = source / "packaging/yggdrasil/M/Moreau/Moreau_CPU/build.log"
    junk.write_text("untracked build output must not be submitted")
    info = release.prepare(source, output)
    assert info["source_commit"] == release.git(source, "rev-parse", "HEAD")
    assert info["julia_tree"] == release.git(source, "rev-parse", f"HEAD:{release.PACKAGE}")
    with tarfile.open(output / "moreau-yggdrasil.tar.gz") as archive:
        assert not any(name.endswith("build.log") for name in archive.getnames())
        for backend in ("CPU", "CUDA"):
            text = (
                archive.extractfile(f"M/Moreau/Moreau_{backend}/build_tarballs.jl").read().decode()
            )
            assert info["source_commit"] in text
            assert f'version = v"{info["version"]}"' in text
            assert "atomic_patch" not in text
    assert (output / "register-julia.txt").read_text() == release.registration_body()


def test_dirty_native_source_cannot_be_submitted(source, tmp_path):
    native = source / "packages/moreau-cpu/Cargo.toml"
    native.write_text(native.read_text() + "\n# uncommitted change\n")
    with pytest.raises(ValueError, match="Commit the release sources"):
        release.prepare(source, tmp_path / "handoff")


def test_version_mismatch_is_caught_before_builds(source):
    native = source / "packages/moreau-cpu/Cargo.toml"
    native.write_text(native.read_text().replace('version = "1.2.3"', 'version = "0.5.0"', 1))
    with pytest.raises(ValueError, match="versions disagree"):
        release.release_info(source)


def test_general_requires_both_jlls(source, registry):
    release.check_registry(source, registry)
    (registry / "jll/M/Moreau_CUDA_jll/Versions.toml").write_text(
        '["0.3.0+0"]\ngit-tree-sha1 = "old"\n'
    )
    with pytest.raises(ValueError, match="Moreau_CUDA_jll"):
        release.check_registry(source, registry)


def test_yanked_jll_does_not_satisfy_release(source, registry):
    path = registry / "jll/M/Moreau_CPU_jll/Versions.toml"
    path.write_text(path.read_text() + "yanked = true\n")
    with pytest.raises(ValueError, match="Waiting for General"):
        release.check_registry(source, registry)


def test_registered_source_tree_must_match_the_tested_frontend(source, registry):
    release.check_registry(source, registry, frontend=True)
    path = registry / "M/Moreau/Versions.toml"
    path.write_text(path.read_text().replace(release.release_info(source)["julia_tree"], "b" * 40))
    with pytest.raises(ValueError, match="tested Moreau"):
        release.check_registry(source, registry, frontend=True)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("repo", "https://github.com/moreau-project/Moreau.jl.git", "monorepo"),
        ("subdir", "other/package", "subdirectory"),
        ("uuid", "wrong-uuid", "identity"),
    ],
)
def test_registry_location_and_identity_are_preserved(source, registry, field, value, message):
    path = registry / "M/Moreau/Package.toml"
    info = tomllib.loads(path.read_text())
    path.write_text(path.read_text().replace(f'"{info[field]}"', f'"{value}"'))
    with pytest.raises(ValueError, match=message):
        release.check_registry(source, registry, frontend=True)


def test_prerelease_is_not_submitted_to_general(source):
    path = source / release.PACKAGE / "Project.toml"
    path.write_text(path.read_text().replace('version = "1.2.3"', 'version = "0.4.1-dev123"'))
    with pytest.raises(ValueError, match="stable"):
        release.release_info(source)


def test_latest_matching_qa_is_required():
    good = {
        "displayTitle": "Test Release v0.4.0",
        "status": "completed",
        "conclusion": "success",
    }
    release.check_qa([good], "v0.4.0")
    with pytest.raises(ValueError, match="latest QA"):
        release.check_qa([dict(good, conclusion="failure"), good], "v0.4.0")
    with pytest.raises(ValueError, match="latest QA"):
        release.check_qa([good], "v0.4.1")
    with pytest.raises(ValueError, match="latest QA"):
        release.check_qa([dict(good, status="in_progress")], "v0.4.0")


def test_source_archive_must_be_the_tagged_subtree(source, tmp_path):
    archive = tmp_path / "source.tar.gz"
    release.git(
        source,
        "archive",
        "--format=tar.gz",
        "--prefix=Moreau.jl/",
        f"HEAD:{release.PACKAGE}",
        f"--output={archive}",
    )
    release.check_source(source, archive)
    with tarfile.open(archive, "w:gz") as output:
        entry = tarfile.TarInfo("Moreau.jl/Project.toml")
        data = b'version = "0.4.0"\n'
        entry.size = len(data)
        output.addfile(entry, io.BytesIO(data))
    with pytest.raises(ValueError, match="tagged package tree"):
        release.check_source(source, archive)


def test_native_prerelease_qa_uses_exact_binary_without_general(source, tmp_path):
    package = source / release.PACKAGE
    path = package / "Project.toml"
    path.write_text(path.read_text().replace('version = "1.2.3"', 'version = "0.5.0-dev123"'))
    library = tmp_path / "libmoreau_cpu.so"
    library.touch()
    output = tmp_path / "native-qa"
    release.native_fixture(package, output, library)
    project = tomllib.loads((output / "Project.toml").read_text())
    assert project["uuid"] == release.JLLS["Moreau_CPU_jll"]
    assert project["version"] == "0.5.0+0"
    assert str(library) in (output / "src/Moreau_CPU_jll.jl").read_text()
    with pytest.raises(FileNotFoundError):
        release.native_fixture(package, output, tmp_path / "missing.so")


def test_yggdrasil_pr_preparation_only_pushes_to_fork(source, tmp_path, monkeypatch):
    release.prepare(source, source / "julia-handoff")
    fork = tmp_path / "fork.git"
    release.git(tmp_path, "init", "--bare", "--initial-branch=master", "-q", str(fork))
    release.git(source, "push", str(fork), "HEAD:refs/heads/master")
    release.git(source, "clone", "-q", str(fork), "yggdrasil")
    checkout = source / "yggdrasil"
    release.git(
        checkout,
        "config",
        f"url.{fork}.insteadOf",
        "https://github.com/JuliaPackaging/Yggdrasil.git",
    )
    release.git(checkout, "config", "commit.gpgsign", "false")
    release.git(checkout, "config", "core.hooksPath", "/dev/null")
    # Any accidental API call must fail locally.
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "gh").write_text("#!/bin/sh\nexit 1\n")
    (fake_bin / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", str(fake_bin) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("BACKEND", "CPU")
    monkeypatch.setenv("YGGDRASIL_FORK", "example/Yggdrasil")
    command = ["bash", str(release.ROOT / "scripts/prepare_julia_jll_pr.sh")]
    subprocess.run(command, cwd=source, check=True, capture_output=True)
    first = release.git(source / "yggdrasil", "rev-parse", "HEAD")
    subprocess.run(command, cwd=source, check=True, capture_output=True)
    assert release.git(source / "yggdrasil", "rev-parse", "HEAD") == first
    branch = release.git(source / "yggdrasil", "branch", "--show-current")
    assert release.git(fork, "rev-parse", branch) == first
    request = (source / "jll-pr-request.md").read_text()
    assert f"JuliaPackaging/Yggdrasil/compare/master...example:{branch}?expand=1" in request
    recipe = source / "yggdrasil/M/Moreau/Moreau_CPU/build_tarballs.jl"
    assert release.git(source, "rev-parse", "HEAD") in recipe.read_text()
