#!/usr/bin/env python3
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

Keep Moreau's Python, Rust, native, and documentation versions in sync."""

from __future__ import annotations

import argparse
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSION_PATTERN = r"(\d+\.\d+\.\d+)(?:-(alpha|beta|rc)\.(\d+))?(?:\.dev(\d+))?"


def bump_version(root: pathlib.Path, version: str, *, pin_dependencies: bool = False) -> None:
    match = re.fullmatch(VERSION_PATTERN, version)
    if match is None:
        raise ValueError("Use X.Y.Z, X.Y.Z-beta.N, or X.Y.Z.devN (optionally after a prerelease)")
    base, prerelease, number, dev = match.groups()
    python_version = base
    if prerelease:
        python_version += {"alpha": "a", "beta": "b", "rc": "rc"}[prerelease] + number
    if dev:
        python_version += ".dev" + dev
    cargo_version = version.replace(".dev", "-dev")
    updates: dict[pathlib.Path, str] = {}

    def replace(path: str, pattern: str, replacement: str, *, count: int = 0) -> None:
        file = root / path
        content = updates.get(file, file.read_text())
        content, matches = re.subn(pattern, replacement, content, count=count, flags=re.MULTILINE)
        if not matches:
            raise ValueError(f"No version found in {path} for {pattern!r}")
        updates[file] = content

    for path in [
        "pyproject.toml",
        "packages/moreau/pyproject.toml",
        "packages/moreau-cpu/pyproject.toml",
        "packages/moreau-cuda/pyproject.toml",
    ]:
        replace(path, r'^version = "[^"]+"', f'version = "{version}"', count=1)
    replace(
        "packages/moreau-cpu/Cargo.toml",
        r'^version = "[^"]+"',
        f'version = "{cargo_version}"',
        count=1,
    )
    for path in [
        "packages/moreau/python/moreau/__init__.py",
        "packages/moreau-cpu/python/moreau_cpu/__init__.py",
        "packages/moreau-cuda/moreau_cuda/__init__.py",
    ]:
        replace(path, r'__version__ = "[^"]+"', f'__version__ = "{version}"')
    for path, prefix in [
        ("packages/moreau-cuda/bindings/moreau_bindings.cpp", 'm.attr("__version__") = '),
        ("packages/moreau-cuda/src/solver/info.cpp", "MOREAU_VERSION = "),
        ("packages/moreau-cuda/src/solver/solver.cpp", "constexpr const char* VERSION = "),
    ]:
        replace(path, re.escape(prefix) + r'"[^"]+"', prefix + f'"{version}"')

    operator = "==" if pin_dependencies else ">="
    for path, names in [
        ("packages/moreau/pyproject.toml", "moreau-cpu|moreau-cuda12|moreau-cuda13"),
        ("packages/moreau-cuda/pyproject.toml", "moreau"),
    ]:
        replace(path, rf'"({names})(?:>=|==)[^"]+"', rf'"\g<1>{operator}{version}"')

    replace("docs/conf.py", r'^release = "[^"]+"', f'release = "{version}"')
    replace("docs/_static/custom.css", r'content: "v[^"]+"', f'content: "v{version}"')
    replace(
        "docs/installation.md",
        r"(moreau-(?:cpu|cuda12|cuda13) \| >= )[^ |]+",
        rf"\g<1>{version}",
    )
    replace(
        "docs/guide/testing-diagnostics.md",
        r"(✓ moreau(?:-cpu|-cuda)?: )[^\s]+",
        rf"\g<1>{version}",
    )
    replace(
        "packages/moreau-cpu/Cargo.lock",
        r'(name = "moreau-cpu"\nversion = ")[^"]+',
        rf"\g<1>{cargo_version}",
    )
    replace(
        "uv.lock",
        r'(name = "moreau(?:-cpu|-cuda|-workspace)?"\nversion = ")[^"]+',
        rf"\g<1>{python_version}",
    )
    # Preserve the CUDA workspace overrides and registry records; only the
    # wrapper's declared extra requirements track the release version.
    replace(
        "uv.lock",
        r'(name = "moreau-cuda(?:12|13)", marker = [^\n]+specifier = ")[^"]+',
        rf"\g<1>{operator}{python_version}",
    )
    # Compute every edit before writing so invalid input or missing fields cannot
    # leave a partially bumped checkout. Julia has its own package version.
    for file, content in updates.items():
        file.write_text(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    parser.add_argument("--pin-dependencies", action="store_true")
    args = parser.parse_args()
    try:
        bump_version(ROOT, args.version, pin_dependencies=args.pin_dependencies)
    except ValueError as error:
        parser.error(str(error))
    print(f"Version bumped to {args.version}")


if __name__ == "__main__":
    main()
