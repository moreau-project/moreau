#!/usr/bin/env python3
"""Validate release wheels for packaging and hardening invariants."""

from __future__ import annotations

import argparse
from collections import Counter
import email.parser
import io
import pathlib
import sys
import tokenize
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scrub_common import _keep_comment

PATH_MARKERS = (b"/workspace/", b"/Users/runner/work/", b"/home/runner/work/")


def wheel_paths(*dirs: str) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for directory in dirs:
        paths.extend(sorted(pathlib.Path(directory).glob("*.whl")))
    return paths


def has_comment_errors(wheel: pathlib.Path, zf: zipfile.ZipFile) -> bool:
    bad = False
    for name in zf.namelist():
        if not name.endswith(".py"):
            continue
        text = zf.read(name).decode("utf-8")
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT and not _keep_comment(tok):
                print(f"ERROR: {wheel}:{name} still contains Python comments")
                bad = True
                break
    return bad


def has_binary_payload_errors(wheel: pathlib.Path, zf: zipfile.ZipFile) -> bool:
    bad = False
    for name in zf.namelist():
        if not name.endswith((".so", ".dylib", ".pyd")):
            continue
        data = zf.read(name)
        for marker in PATH_MARKERS:
            if marker in data:
                print(f"ERROR: {wheel}:{name} contains build path marker {marker.decode()}")
                bad = True

    return bad


def has_wheel_errors(wheel: pathlib.Path) -> bool:
    """Return True if the wheel has any packaging/hardening errors."""
    bad = False
    with zipfile.ZipFile(wheel) as zf:
        if zf.testzip() is not None:
            print(f"ERROR: {wheel} has a corrupt ZIP member")
            bad = True
        names = zf.namelist()
        sboms = [name for name in names if "/sboms/" in name]
        if sboms:
            print(f"ERROR: {wheel} contains SBOM files: {sboms[:5]}")
            bad = True

        bad |= has_comment_errors(wheel, zf)
        bad |= has_binary_payload_errors(wheel, zf)

    return bad


def release_version(wheels: list[pathlib.Path], *, require_complete: bool = False) -> str:
    """Read the common version from wheel metadata, independent of release tags."""
    versions = set()
    packages = Counter()
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as zf:
            metadata = [name for name in zf.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError(f"{wheel}: expected exactly one METADATA file")
            message = email.parser.Parser().parsestr(zf.read(metadata[0]).decode("utf-8"))
        name, version = message["Name"], message["Version"]
        filename_parts = wheel.name.split("-")
        if not name or not version or filename_parts[:2] != [name.replace("-", "_"), version]:
            raise ValueError(f"{wheel}: filename and package metadata disagree")
        versions.add(version)
        packages[name.replace("-", "_")] += 1
    if require_complete and packages != Counter(
        {"moreau": 1, "moreau_cpu": 3, "moreau_cuda12": 2, "moreau_cuda13": 2}
    ):
        raise ValueError(f"Incomplete release wheel matrix: {dict(packages)}")
    if len(versions) != 1:
        raise ValueError(f"Expected one release version, found {sorted(versions)}")
    return versions.pop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wheel_dirs",
        nargs="*",
        default=["dist-wheels"],
        help="Directories containing built wheels to validate.",
    )
    parser.add_argument("--version-output", type=pathlib.Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    wheels = wheel_paths(*args.wheel_dirs)
    if not wheels:
        print(f"ERROR: no wheels found in {args.wheel_dirs}")
        return 1

    try:
        version = release_version(wheels, require_complete=args.require_complete)
    except ValueError as error:
        print(f"ERROR: {error}")
        return 1

    bad = False
    for wheel in wheels:
        bad |= has_wheel_errors(wheel)

    if bad:
        return 1

    if args.version_output:
        args.version_output.write_text(version + "\n")
    print(f"Release wheel hardening checks passed ({version})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
