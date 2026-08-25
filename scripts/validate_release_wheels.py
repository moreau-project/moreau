#!/usr/bin/env python3
"""Validate release wheels for packaging and hardening invariants."""

from __future__ import annotations

import argparse
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
        names = zf.namelist()
        sboms = [name for name in names if "/sboms/" in name]
        if sboms:
            print(f"ERROR: {wheel} contains SBOM files: {sboms[:5]}")
            bad = True

        bad |= has_comment_errors(wheel, zf)
        bad |= has_binary_payload_errors(wheel, zf)

    return bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wheel_dirs",
        nargs="*",
        default=["dist-wheels"],
        help="Directories containing built wheels to validate.",
    )
    args = parser.parse_args()

    wheels = wheel_paths(*args.wheel_dirs)
    if not wheels:
        print(f"ERROR: no wheels found in {args.wheel_dirs}")
        return 1

    bad = False
    for wheel in wheels:
        bad |= has_wheel_errors(wheel)

    if bad:
        return 1

    print("Release wheel hardening checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
