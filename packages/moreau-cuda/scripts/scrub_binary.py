#!/usr/bin/env python3
"""
Scrub identifying strings from compiled CUDA binaries.

Usage:
    python scrub_binary.py input.so output.so

Or as post-build step on wheel:
    python scrub_binary.py dist/*.whl
"""

import sys
from pathlib import Path

# Allow importing scrub_common from repo root scripts/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from scrub_common import pad_to_length, scrub_main  # noqa: E402

# Patterns to scrub (regex -> replacement function).
# Broad enough to catch any leaked source path regardless of extension, since
# vendored deps can embed __FILE__ strings that don't end in C++ extensions.
SCRUB_PATTERNS = [
    # Relative paths from -ffile-prefix-map (./include/..., ./src/..., ./bindings/...)
    (
        rb"\./(?:include|src|bindings)/[a-zA-Z0-9_/.-]+",
        lambda m: pad_to_length(b"x", len(m.group(0))),
    ),
    # Absolute paths that might have slipped through
    (
        rb"/workspace/[a-zA-Z0-9_/.-]+",
        lambda m: pad_to_length(b"x", len(m.group(0))),
    ),
    (
        rb"/home/[a-zA-Z0-9_/.-]+\.(?:cpp|cu|cuh|hpp|h|c)",
        lambda m: pad_to_length(b"x", len(m.group(0))),
    ),
    # Generic catch-all for any remaining source paths with known extensions
    (
        rb"[a-zA-Z0-9_/.-]+/src/[a-zA-Z0-9_/.-]+\.(?:cpp|cu|cuh|c)",
        lambda m: pad_to_length(b"x", len(m.group(0))),
    ),
    (
        rb"[a-zA-Z0-9_/.-]+/include/[a-zA-Z0-9_/.-]+\.(?:hpp|h)",
        lambda m: pad_to_length(b"x", len(m.group(0))),
    ),
]

VERIFY_PATTERNS = [
    rb"\./src/",
    rb"\./include/",
    rb"/workspace/",
    rb"/[a-zA-Z0-9_]+\.cpp",
    rb"/[a-zA-Z0-9_]+\.cu[^da]",
    rb"/[a-zA-Z0-9_]+\.hpp",
]

if __name__ == "__main__":
    scrub_main(SCRUB_PATTERNS, VERIFY_PATTERNS)
