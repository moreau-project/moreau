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

Prepare and verify Julia releases from the Moreau monorepo. No network writes.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess
import tarfile
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = pathlib.Path("packages/moreau-julia/Moreau.jl")
REPO = "https://github.com/moreau-project/moreau.git"
UUID = "c8b129f6-74e5-4f0d-b2d8-2ea10d91a548"
JLLS = {
    "Moreau_CPU_jll": "c0bfae29-27af-5321-a17d-6bf033ce7ea7",
    "Moreau_CUDA_jll": "c77365f8-1b8a-5c63-97c9-c9362ae64116",
}


def git(root: pathlib.Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def project(root: pathlib.Path) -> dict:
    return tomllib.loads((root / PACKAGE / "Project.toml").read_text())


def release_info(root: pathlib.Path) -> dict:
    info = project(root)
    version = info["version"]
    if info["name"] != "Moreau" or info["uuid"] != UUID:
        raise ValueError("The registered Moreau name and UUID must be retained")
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("General/Yggdrasil registration requires a stable X.Y.Z release")
    native = tomllib.loads((root / "packages/moreau-cpu/Cargo.toml").read_text())
    if native["package"]["version"] != version:
        raise ValueError("Julia and native release versions disagree; run bump_version.py first")
    for name in JLLS:
        if info["compat"].get(name) != f"={version}":
            raise ValueError(f"{name} must be pinned to ={version}")
    return {
        "version": version,
        "source_commit": git(root, "rev-parse", "HEAD"),
        "julia_tree": git(root, "rev-parse", f"HEAD:{PACKAGE.as_posix()}"),
        "repo": REPO,
        "subdir": PACKAGE.as_posix(),
    }


def prepare(root: pathlib.Path, output: pathlib.Path) -> dict:
    info = release_info(root)
    # Recipes must describe the committed source, not local version edits.
    changed = git(root, "diff", "HEAD", "--", "packages", "packaging/yggdrasil")
    if changed:
        raise ValueError("Commit the release sources and recipes before preparing Yggdrasil")
    output.mkdir(parents=True, exist_ok=True)
    staging = output / "yggdrasil"
    for backend in ("CPU", "CUDA"):
        relative = pathlib.Path(f"M/Moreau/Moreau_{backend}")
        destination = staging / relative
        if destination.exists():
            shutil.rmtree(destination)
        source = pathlib.Path("packaging/yggdrasil") / relative
        for name in git(root, "ls-files", str(source)).splitlines():
            target = destination / pathlib.Path(name).relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        recipe = destination / "build_tarballs.jl"
        text = recipe.read_text()
        text, count = re.subn(
            r'(GitSource\("https://github.com/moreau-project/moreau.git",\s*")[0-9a-f]{40}',
            lambda match: match[1] + info["source_commit"],
            text,
        )
        if count != 1 or f'version = v"{info["version"]}"' not in text:
            raise ValueError(f"Moreau_{backend} recipe version/source is inconsistent")
        recipe.write_text(text)
    (output / "moreau-julia-release.json").write_text(json.dumps(info, indent=2) + "\n")
    (output / "register-julia.txt").write_text(registration_body())
    with tarfile.open(output / "moreau-yggdrasil.tar.gz", "w:gz") as archive:
        archive.add(staging / "M", arcname="M")
    return info


def check_registry(root: pathlib.Path, registry: pathlib.Path, *, frontend: bool = False) -> None:
    info = release_info(root)
    version = info["version"]
    packages = [(f"jll/M/{name}", name, uuid) for name, uuid in JLLS.items()]
    if frontend:
        packages.append(("M/Moreau", "Moreau", UUID))
    for relative, name, uuid in packages:
        directory = registry / relative
        if not (directory / "Package.toml").exists():
            raise ValueError(f"Waiting for General registration: {name} {version}")
        package = tomllib.loads((directory / "Package.toml").read_text())
        versions = tomllib.loads((directory / "Versions.toml").read_text())
        if package["uuid"] != uuid or package["name"] != name:
            raise ValueError(f"Wrong registered identity for {name}")
        if name == "Moreau":
            if package["repo"].removesuffix(".git") != REPO.removesuffix(".git"):
                raise ValueError("General must point Moreau to the monorepo")
            if package.get("subdir") != PACKAGE.as_posix():
                raise ValueError("General must record the Moreau.jl subdirectory")
            entry = versions.get(version, {})
            if entry.get("git-tree-sha1") != info["julia_tree"] or entry.get("yanked", False):
                raise ValueError(f"General has not registered the tested Moreau {version} tree")
        elif not any(
            re.fullmatch(re.escape(version) + r"\+\d+", key) and not entry.get("yanked", False)
            for key, entry in versions.items()
        ):
            raise ValueError(f"Waiting for General registration: {name} {version}+N")


def check_qa(runs: list[dict], tag: str) -> None:
    matching = [run for run in runs if run["displayTitle"] == f"Test Release {tag}"]
    if (
        not matching
        or matching[0]["status"] != "completed"
        or matching[0]["conclusion"] != "success"
    ):
        raise ValueError("The latest QA run for this release tag and commit must pass")


def registration_body() -> str:
    return f"ChatGPT generated:\n\n@JuliaRegistrator register subdir={PACKAGE.as_posix()}\n"


def check_source(root: pathlib.Path, archive_path: pathlib.Path) -> None:
    names = git(root, "ls-files", str(PACKAGE)).splitlines()
    expected = {
        "Moreau.jl/"
        + str(pathlib.Path(name).relative_to(PACKAGE)): subprocess.check_output(
            ["git", "-C", str(root), "show", f"HEAD:{name}"]
        )
        for name in names
    }
    with tarfile.open(archive_path) as archive:
        actual = {}
        for member in archive:
            if member.isdir():
                continue
            if not member.isfile() or member.name in actual:
                raise ValueError("Unexpected Julia source archive entry")
            actual[member.name] = archive.extractfile(member).read()
    if actual != expected:
        raise ValueError("Released Julia source differs from the tagged package tree")


def native_fixture(package: pathlib.Path, output: pathlib.Path, library: pathlib.Path) -> None:
    """Dependency shim for prerelease/PR QA against a real native build, never shipped."""
    library = library.resolve(strict=True)
    version = tomllib.loads((package / "Project.toml").read_text())["version"].split("-", 1)[0]
    name = "Moreau_CPU_jll"
    (output / "src").mkdir(parents=True, exist_ok=True)
    (output / "Project.toml").write_text(
        f'name = "{name}"\nuuid = "{JLLS[name]}"\nversion = "{version}+0"\n'
    )
    (output / "src" / f"{name}.jl").write_text(
        f"module {name}\nconst libmoreau = {json.dumps(str(library))}\n"
        "is_available() = isfile(libmoreau)\nend\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare")
    build.add_argument("--output", type=pathlib.Path, required=True)
    registry = sub.add_parser("check-registry")
    registry.add_argument("registry", type=pathlib.Path)
    registry.add_argument("--frontend", action="store_true")
    qa = sub.add_parser("check-qa")
    qa.add_argument("runs", type=pathlib.Path)
    qa.add_argument("tag")
    fixture = sub.add_parser("native-fixture")
    fixture.add_argument("--output", type=pathlib.Path, required=True)
    fixture.add_argument("--library", type=pathlib.Path, required=True)
    fixture.add_argument("--project", type=pathlib.Path)
    source = sub.add_parser("check-source")
    source.add_argument("archive", type=pathlib.Path)
    version = sub.add_parser("check-version")
    version.add_argument("version")
    sub.add_parser("info")
    sub.add_parser("registration-body")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            print(json.dumps(prepare(args.root, args.output)))
        elif args.command == "check-registry":
            check_registry(args.root, args.registry, frontend=args.frontend)
        elif args.command == "check-qa":
            check_qa(json.loads(args.runs.read_text()), args.tag)
        elif args.command == "native-fixture":
            native_fixture(args.project or args.root / PACKAGE, args.output, args.library)
        elif args.command == "check-source":
            check_source(args.root, args.archive)
        elif args.command == "check-version":
            if (
                re.fullmatch(r"\d+\.\d+\.\d+", args.version)
                and release_info(args.root)["version"] != args.version
            ):
                raise ValueError("Commit bump_version.py changes before starting a stable release")
        elif args.command == "registration-body":
            print(registration_body(), end="")
        else:
            print(json.dumps(release_info(args.root)))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
