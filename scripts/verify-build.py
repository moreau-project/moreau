#!/usr/bin/env python3
"""Verify that installed Moreau binaries are up-to-date with source files.

Compares newest source file timestamps against installed .so timestamps.
Reports stale builds with rebuild commands.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PACKAGES = [
    {
        "label": "CPU",
        "src_dirs": [REPO_ROOT / "packages" / "moreau-cpu" / "src"],
        "extensions": {".rs"},
        "module": "moreau._moreau_cpu",
        "rebuild": "cd packages/moreau-cpu && maturin develop --release",
    },
    {
        "label": "CUDA",
        "src_dirs": [
            REPO_ROOT / "packages" / "moreau-cuda" / d for d in ("src", "include", "bindings")
        ],
        "extensions": {".cpp", ".cu", ".hpp", ".h"},
        "module": "moreau_cuda._moreau_cuda",
        "rebuild": "cd packages/moreau-cuda && pip install --no-build-isolation --no-cache-dir --no-deps . --force-reinstall",
    },
]


def newest_source(directories: list[Path], extensions: set[str]) -> tuple[float, Path | None]:
    """Find the newest source file by mtime across directory trees."""
    newest_time = 0.0
    newest_file = None
    for directory in directories:
        if not directory.exists():
            continue
        for root, _, files in os.walk(directory):
            for f in files:
                if Path(f).suffix in extensions:
                    p = Path(root) / f
                    t = p.stat().st_mtime
                    if t > newest_time:
                        newest_time = t
                        newest_file = p
    return newest_time, newest_file


def find_installed_so(module_name: str) -> tuple[float, Path | None]:
    """Find installed .so file for a Python module and return its mtime."""
    try:
        mod = __import__(module_name)
        so_path = Path(mod.__file__)
        return so_path.stat().st_mtime, so_path
    except (ImportError, AttributeError, TypeError):
        return 0.0, None


def check_package(pkg: dict) -> bool:
    """Check if a package's .so is up-to-date with its sources."""
    label = pkg["label"]
    src_time, src_file = newest_source(pkg["src_dirs"], pkg["extensions"])
    so_time, so_path = find_installed_so(pkg["module"])

    if so_path is None:
        print(f"{label:4s}: NOT INSTALLED")
        print(f"  Rebuild: {pkg['rebuild']}")
        return False

    if src_time > so_time:
        fmt = lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{label:4s}: STALE")
        print(f"  Newest source: {src_file} ({fmt(src_time)})")
        print(f"  Installed .so: {so_path} ({fmt(so_time)})")
        print(f"  Rebuild: {pkg['rebuild']}")
        return False

    print(f"{label:4s}: OK ({so_path})")
    return True


def main():
    print("Moreau build verification")
    print("=" * 50)
    results = [check_package(pkg) for pkg in PACKAGES]
    all_ok = all(results)
    print("=" * 50)
    if all_ok:
        print("All builds up-to-date.")
    else:
        print("Stale builds detected — rebuild before benchmarking!")
        sys.exit(1)


if __name__ == "__main__":
    main()
