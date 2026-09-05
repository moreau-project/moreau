#!/usr/bin/env python3
"""Validate release wheels for packaging and hardening invariants."""

from __future__ import annotations

import argparse
from collections import Counter
import email.parser
import io
from itertools import product
import pathlib
import re
import sys
import tarfile
import tokenize
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scrub_common import _keep_comment

RELEASE_WHEEL_TAGS = {
    ("moreau", "py3", "none", "any"),
    ("moreau_cpu", "cp39", "abi3", "manylinux_2_28_x86_64"),
    ("moreau_cpu", "cp39", "abi3", "manylinux_2_28_aarch64"),
    ("moreau_cpu", "cp39", "abi3", "macosx_11_0_arm64"),
    ("moreau_cuda12", "cp312", "abi3", "manylinux_2_28_x86_64"),
    ("moreau_cuda12", "cp312", "abi3", "manylinux_2_28_aarch64"),
    ("moreau_cuda13", "cp312", "abi3", "manylinux_2_28_x86_64"),
    ("moreau_cuda13", "cp312", "abi3", "manylinux_2_28_aarch64"),
}
C_LIBRARY_ARCHIVES = {
    "moreau-cpu-linux-x86_64.tar.gz": "libmoreau_cpu.so",
    "moreau-cpu-linux-aarch64.tar.gz": "libmoreau_cpu.so",
    "moreau-cpu-macos-arm64.tar.gz": "libmoreau_cpu.dylib",
    "moreau-cuda12-linux-x86_64.tar.gz": "libmoreau_cuda.so",
    "moreau-cuda12-linux-aarch64.tar.gz": "libmoreau_cuda.so",
    "moreau-cuda13-linux-x86_64.tar.gz": "libmoreau_cuda.so",
    "moreau-cuda13-linux-aarch64.tar.gz": "libmoreau_cuda.so",
}

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
        for required in ("LICENSE", "NOTICE"):
            if not any(name.endswith("/" + required) for name in names):
                print(f"ERROR: {wheel} is missing {required}")
                bad = True
        sboms = [name for name in names if "/sboms/" in name]
        if sboms:
            print(f"ERROR: {wheel} contains SBOM files: {sboms[:5]}")
            bad = True

        bad |= has_comment_errors(wheel, zf)
        bad |= has_binary_payload_errors(wheel, zf)

    return bad


def _normalize_prerelease(value: str) -> str:
    return re.sub(
        r"-(alpha|beta|rc)\.(\d+)",
        lambda match: {"alpha": "a", "beta": "b", "rc": "rc"}[match[1]] + match[2],
        value,
    )


def release_version(wheels: list[pathlib.Path], *, require_complete: bool = False) -> str:
    """Read the common version from wheel metadata, independent of release tags."""
    versions = set()
    wheel_tags = Counter()
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as zf:
            metadata = [name for name in zf.namelist() if name.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError(f"{wheel}: expected exactly one METADATA file")
            message = email.parser.Parser().parsestr(zf.read(metadata[0]).decode("utf-8"))
            if require_complete:
                wheel_metadata = metadata[0].removesuffix("METADATA") + "WHEEL"
                if wheel_metadata not in zf.namelist():
                    raise ValueError(f"{wheel}: missing WHEEL metadata")
                wheel_info = email.parser.Parser().parsestr(zf.read(wheel_metadata).decode("utf-8"))
                expected_tags = {
                    "-".join(tag)
                    for tag in product(*(part.split(".") for part in wheel.stem.split("-")[-3:]))
                }
                if set(wheel_info.get_all("Tag", [])) != expected_tags:
                    raise ValueError(f"{wheel}: filename and WHEEL tags disagree")
        name, version = message["Name"], message["Version"]
        filename_parts = wheel.name.split("-")
        if not name or not version or filename_parts[:2] != [name.replace("-", "_"), version]:
            raise ValueError(f"{wheel}: filename and package metadata disagree")
        versions.add(version)
        wheel_tags[(name.replace("-", "_"), *wheel.stem.split("-")[-3:])] += 1
        if require_complete:
            requirements = message.get_all("Requires-Dist", [])
            backends = (
                ("moreau-cpu", "moreau-cuda12", "moreau-cuda13")
                if name == "moreau"
                else ("moreau",) if name.replace("-", "_").startswith("moreau_cuda") else ()
            )
            for backend in backends:
                matches = [r for r in requirements if re.match(rf"{backend}(?:[ (=>;]|$)", r)]
                if not matches or any(
                    not re.fullmatch(
                        rf"{backend}\s*\(?=={re.escape(version)}\)?\s*(?:;.*)?",
                        _normalize_prerelease(r),
                    )
                    for r in matches
                ):
                    raise ValueError(f"{wheel}: {backend} must be pinned to =={version}")
    if require_complete and wheel_tags != Counter(RELEASE_WHEEL_TAGS):
        raise ValueError(f"Incomplete release wheel matrix (platform/ABI): {dict(wheel_tags)}")
    if len(versions) != 1:
        raise ValueError(f"Expected one release version, found {sorted(versions)}")
    return versions.pop()


def validate_c_libraries(directory: pathlib.Path) -> None:
    """Require every supported C archive and its public header and licenses."""
    archives = {path.name: path for path in directory.glob("*.tar.gz")}
    if archives.keys() != C_LIBRARY_ARCHIVES.keys():
        raise ValueError(f"Incomplete C library matrix: {sorted(archives)}")
    for filename, library in C_LIBRARY_ARCHIVES.items():
        with tarfile.open(archives[filename]) as archive:
            names = {member.name.removeprefix("./"): member for member in archive.getmembers()}
            required = {"include/moreau.h", "LICENSE", "NOTICE", f"lib/{library}"}
            missing = required - names.keys()
            if missing:
                raise ValueError(f"{filename}: missing {sorted(missing)}")
            for name in required:
                member = names[name]
                if not (member.isfile() or (name.startswith("lib/") and member.issym())):
                    raise ValueError(f"{filename}: {name} is not a file or library symlink")


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
    parser.add_argument("--c-library-dir", type=pathlib.Path)
    args = parser.parse_args()

    wheels = wheel_paths(*args.wheel_dirs)
    if not wheels:
        print(f"ERROR: no wheels found in {args.wheel_dirs}")
        return 1

    try:
        version = release_version(wheels, require_complete=args.require_complete)
        if args.c_library_dir:
            validate_c_libraries(args.c_library_dir)
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
