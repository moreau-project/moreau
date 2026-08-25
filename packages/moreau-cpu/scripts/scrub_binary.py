#!/usr/bin/env python3
"""
Scrub identifying strings from compiled Rust binaries.

Usage:
    python scrub_binary.py input.so output.so

Or as post-build step:
    maturin build --release
    python scrub_binary.py target/wheels/*.whl
"""

import sys
from pathlib import Path

# Allow importing scrub_common from repo root scripts/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from scrub_common import pad_to_length, scrub_main  # noqa: E402

# Patterns to scrub (regex -> replacement function)
# NOTE: Use [a-zA-Z0-9_/.-]+ for path components, NOT [^\x00]+ which is too greedy
# and can match across multiple strings, corrupting the binary.
SCRUB_PATTERNS = [
    # CI/local build output paths that can appear in Mach-O/ELF metadata
    (
        rb"/Users/[a-zA-Z0-9_/.-]+/target/release/deps/lib[a-zA-Z0-9_.-]+\.dylib",
        lambda m: pad_to_length(b"/x/lib.dylib", len(m.group(0))),
    ),
    (
        rb"/home/[a-zA-Z0-9_/.-]+/target/release/deps/lib[a-zA-Z0-9_.-]+\.so",
        lambda m: pad_to_length(b"/x/lib.so", len(m.group(0))),
    ),
    # Crate paths: /home/.../.cargo/registry/src/index.crates.io-.../faer-0.21.9/src/lib.rs
    (
        rb"/[a-zA-Z0-9_/.-]+/\.cargo/registry/src/[^/]+/[a-zA-Z0-9_/.-]+\.rs",
        lambda m: pad_to_length(b"/x/x.rs", len(m.group(0))),
    ),
    # Rust stdlib/deps paths (from backtrace): /rust/deps/gimli-0.32.0/src/...
    (
        rb"/rust/deps/[a-zA-Z0-9_-]+-[\d.]+/src/[a-zA-Z0-9_/.-]+\.rs",
        lambda m: pad_to_length(b"/r/x.rs", len(m.group(0))),
    ),
    # Rustc internal: /rustc/.../library/...
    (
        rb"/rustc/[a-f0-9]+/library/[a-zA-Z0-9_/.-]+\.rs",
        lambda m: pad_to_length(b"/r/x.rs", len(m.group(0))),
    ),
    # Local source: src/solver/core/...
    (
        rb"src/solver/[a-zA-Z0-9_/.-]+\.rs",
        lambda m: pad_to_length(b"s/x.rs", len(m.group(0))),
    ),
    (
        rb"src/algebra/[a-zA-Z0-9_/.-]+\.rs",
        lambda m: pad_to_length(b"s/x.rs", len(m.group(0))),
    ),
    (
        rb"src/python/[a-zA-Z0-9_/.-]+\.rs",
        lambda m: pad_to_length(b"s/x.rs", len(m.group(0))),
    ),
    (
        rb"src/[a-zA-Z0-9_/.-]+\.rs",
        lambda m: pad_to_length(b"s/x.rs", len(m.group(0))),
    ),
    # Crate names with versions (anywhere in binary)
    (
        rb"faer-\d+\.\d+\.\d+",
        lambda m: pad_to_length(b"lib-0.0.0", len(m.group(0))),
    ),
    (
        rb"amd-\d+\.\d+\.\d+",
        lambda m: pad_to_length(b"lib-0.0.0", len(m.group(0))),
    ),
    (
        rb"rayon-\d+\.\d+\.\d+",
        lambda m: pad_to_length(b"lib-0.0.0", len(m.group(0))),
    ),
    (
        rb"gimli-\d+\.\d+\.\d+",
        lambda m: pad_to_length(b"lib-0.0.0", len(m.group(0))),
    ),
    (
        rb"rustc-demangle-\d+\.\d+\.\d+",
        lambda m: pad_to_length(b"lib-0.0.0", len(m.group(0))),
    ),
    (
        rb"crossbeam-[a-z]+-\d+\.\d+\.\d+",
        lambda m: pad_to_length(b"lib-0.0.0", len(m.group(0))),
    ),
    (
        rb"pyo3-\d+\.\d+\.\d*",
        lambda m: pad_to_length(b"lib-0.0.0", len(m.group(0))),
    ),
    # Generic .rs file paths (catch-all, run last)
    (
        rb"/[a-zA-Z0-9_/.]+/src/[a-zA-Z0-9_/]+\.rs",
        lambda m: pad_to_length(b"/x/x.rs", len(m.group(0))),
    ),
    # library/std/src/... patterns from stdlib
    (
        rb"library/[a-zA-Z0-9_/]+\.rs",
        lambda m: pad_to_length(b"lib/x.rs", len(m.group(0))),
    ),
]

VERIFY_PATTERNS = [
    rb"faer-\d+\.\d+",
    rb"amd-\d+\.\d+",
    rb"src/solver/",
    rb"src/algebra/",
    rb"/\.cargo/registry/",
    rb"/Users/runner/work/",
    rb"/home/runner/work/",
]

if __name__ == "__main__":
    scrub_main(SCRUB_PATTERNS, VERIFY_PATTERNS)
