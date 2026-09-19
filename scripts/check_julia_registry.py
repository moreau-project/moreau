# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0
"""Check that General contains the release's JLLs and, optionally, frontend."""

import argparse
import pathlib
import re
import subprocess
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


def read_toml(path: pathlib.Path) -> dict:
    return tomllib.loads(path.read_text())


def check_registry(root: pathlib.Path, registry: pathlib.Path, *, frontend: bool = False) -> None:
    version = read_toml(root / PACKAGE / "Project.toml")["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("General registration requires a stable X.Y.Z release")
    packages = [(f"jll/M/{name}", name, uuid) for name, uuid in JLLS.items()]
    if frontend:
        packages.append(("M/Moreau", "Moreau", UUID))
    for relative, name, uuid in packages:
        directory = registry / relative
        if not (directory / "Package.toml").exists():
            raise ValueError(f"Waiting for General registration: {name} {version}")
        package = read_toml(directory / "Package.toml")
        versions = read_toml(directory / "Versions.toml")
        if package["uuid"] != uuid or package["name"] != name:
            raise ValueError(f"Wrong registered identity for {name}")
        if name == "Moreau":
            if package["repo"].removesuffix(".git") != REPO.removesuffix(".git"):
                raise ValueError("General must point Moreau to the monorepo")
            if package.get("subdir") != PACKAGE.as_posix():
                raise ValueError("General must record the Moreau.jl subdirectory")
            entry = versions.get(version, {})
            tree = git(root, "rev-parse", f"HEAD:{PACKAGE.as_posix()}")
            if entry.get("git-tree-sha1") != tree or entry.get("yanked", False):
                raise ValueError(f"General has not registered the tested Moreau {version} tree")
        elif not any(
            re.fullmatch(re.escape(version) + r"\+\d+", key) and not entry.get("yanked", False)
            for key, entry in versions.items()
        ):
            raise ValueError(f"Waiting for General registration: {name} {version}+N")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registry", type=pathlib.Path)
    parser.add_argument("--frontend", action="store_true")
    args = parser.parse_args()
    check_registry(ROOT, args.registry, frontend=args.frontend)
