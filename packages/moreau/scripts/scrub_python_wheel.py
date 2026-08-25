#!/usr/bin/env python3
"""Strip non-runtime Python comments from built wheels."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from scrub_common import strip_comments_in_wheel, verify_no_comments


def main():
    if len(sys.argv) < 2:
        print("Usage: scrub_python_wheel.py <input.whl> [output.whl]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if input_path.suffix != ".whl":
        print("Error: expected a wheel file")
        sys.exit(1)

    strip_comments_in_wheel(input_path, output_path)
    if not verify_no_comments(output_path or input_path):
        sys.exit(1)


if __name__ == "__main__":
    main()
