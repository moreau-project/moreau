"""
Shared utilities for scrubbing build artifacts from release wheels and binaries.

Used by packages/moreau-cpu/scripts/scrub_binary.py and
packages/moreau-cuda/scripts/scrub_binary.py.
"""

from __future__ import annotations

import base64
import hashlib
import io
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
import zipfile
from pathlib import Path

ENCODING_COMMENT_RE = re.compile(r"^#.*coding[:=]\s*([-\w.]+)")


def pad_to_length(replacement: bytes, target_length: int) -> bytes:
    """Pad or truncate replacement to exact length (critical for ELF integrity)."""
    if len(replacement) >= target_length:
        return replacement[:target_length]
    return replacement + b" " * (target_length - len(replacement))


def record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def rebuild_wheel_record(root: Path):
    """Regenerate RECORD after modifying wheel contents."""
    record_files = list(root.rglob("*.dist-info/RECORD"))
    if len(record_files) != 1:
        raise RuntimeError(f"Expected exactly one RECORD file, found {len(record_files)}")

    record_path = record_files[0]
    lines = []
    for file in sorted(root.rglob("*")):
        if not file.is_file():
            continue
        rel = file.relative_to(root).as_posix()
        if file == record_path:
            lines.append(f"{rel},,")
            continue
        data = file.read_bytes()
        lines.append(f"{rel},{record_hash(data)},{len(data)}")
    record_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _keep_comment(tok: tokenize.TokenInfo) -> bool:
    """Keep shebangs and encoding declarations, strip everything else."""
    if tok.start == (1, 0) and tok.string.startswith("#!"):
        return True
    if tok.start[0] <= 2 and ENCODING_COMMENT_RE.match(tok.string):
        return True
    return False


def strip_python_comments(text: str) -> tuple[str, int]:
    """Remove non-runtime comments from Python source. Returns (stripped, count)."""
    out_tokens: list[tokenize.TokenInfo] = []
    removed = 0

    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT and not _keep_comment(tok):
            removed += 1
            continue
        out_tokens.append(tok)

    return tokenize.untokenize(out_tokens), removed


def strip_comments_in_wheel(wheel_path: Path, output_path: Path | None = None):
    """Strip Python comments from all .py files in a wheel."""
    if output_path is None:
        output_path = wheel_path

    total_removed = 0

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)

        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(tmpdir)

        for py_file in sorted(tmpdir.rglob("*.py")):
            original = py_file.read_text(encoding="utf-8")
            stripped, removed = strip_python_comments(original)
            if removed:
                rel = py_file.relative_to(tmpdir).as_posix()
                try:
                    compile(stripped, rel, "exec")
                except SyntaxError as e:
                    raise RuntimeError(
                        f"Comment stripping produced invalid Python in {rel}: {e}"
                    ) from e
                py_file.write_text(stripped, encoding="utf-8")
                total_removed += removed

        rebuild_wheel_record(tmpdir)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in tmpdir.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(tmpdir))

    print(f"Stripped {total_removed} Python comments from {output_path}")


def verify_no_comments(wheel_path: Path) -> bool:
    """Verify that no non-runtime comments remain in a wheel's .py files."""
    with zipfile.ZipFile(wheel_path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".py"):
                continue
            text = zf.read(name).decode("utf-8")
            for tok in tokenize.generate_tokens(io.StringIO(text).readline):
                if tok.type == tokenize.COMMENT and not _keep_comment(tok):
                    print(f"ERROR: comment leaked in {name}")
                    return False
    print("Python comment stripping verified")
    return True


def scrub_binary(data: bytes, patterns: list) -> bytes:
    """Apply all scrub patterns to binary data."""
    result = data
    total_replacements = 0

    for pattern, replacer in patterns:

        def do_replace(m, _replacer=replacer):
            nonlocal total_replacements
            total_replacements += 1
            return _replacer(m)

        result = re.sub(pattern, do_replace, result)

    print(f"  Made {total_replacements} replacements")
    return result


def detect_binary_format(path: Path) -> str:
    """Detect binary format by reading magic bytes. Returns 'elf', 'macho', 'pe', or 'unknown'."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)

            if magic == b"\x7fELF":
                return "elf"

            if magic in (
                b"\xfe\xed\xfa\xcf",
                b"\xfe\xed\xfa\xce",
                b"\xcf\xfa\xed\xfe",
                b"\xce\xfa\xed\xfe",
                b"\xca\xfe\xba\xbe",
                b"\xca\xfe\xba\xbf",
                b"\xbe\xba\xfe\xca",
                b"\xbf\xba\xfe\xca",
            ):
                return "macho"

            if magic[:2] == b"MZ":
                return "pe"

            return "unknown"
    except Exception:
        return "unknown"


def strip_elf_metadata(path: Path, remove_rpath: bool = True):
    """Remove identifying ELF sections and debug info."""
    sections = [
        ".comment",
        ".note.gnu.build-id",
        ".note.GNU-stack",
    ]

    args = ["objcopy"]
    for section in sections:
        args.extend(["--remove-section", section])
    args.append(str(path))

    try:
        subprocess.run(args, check=True, capture_output=True)
        print("  Stripped ELF metadata sections")
    except subprocess.CalledProcessError as e:
        print(f"  Warning: objcopy failed: {e.stderr.decode()}")
    except FileNotFoundError:
        print("  Warning: objcopy not found, skipping ELF stripping")

    # Strip DWARF debug info and local symbols from shared library.
    # --strip-unneeded keeps dynamic symbols needed at load time but removes
    # all .debug_* sections (where build paths like /workspace/ get embedded).
    try:
        subprocess.run(["strip", "--strip-unneeded", str(path)], check=True, capture_output=True)
        print("  Stripped debug info and local symbols")
    except subprocess.CalledProcessError as e:
        print(f"  Warning: strip failed: {e.stderr.decode()}")
    except FileNotFoundError:
        print("  Warning: strip not found, skipping debug info stripping")

    # Check whether the binary actually has RPATH/RUNPATH before trying to remove.
    has_rpath = False
    try:
        proc = subprocess.run(
            ["readelf", "-d", str(path)], capture_output=True, text=True, check=False
        )
        has_rpath = proc.returncode == 0 and ("RPATH" in proc.stdout or "RUNPATH" in proc.stdout)
    except FileNotFoundError:
        pass

    if not has_rpath or not remove_rpath:
        return

    for cmd in (
        ["patchelf", "--remove-rpath", str(path)],
        ["chrpath", "-d", str(path)],
    ):
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"  Cleared ELF rpath/runpath with {cmd[0]}")
            return
        except subprocess.CalledProcessError:
            continue
        except FileNotFoundError:
            continue

    raise RuntimeError(
        f"{path}: binary has RPATH/RUNPATH but neither patchelf nor chrpath is available to remove it"
    )


def strip_macho_metadata(path: Path):
    """Remove identifying Mach-O metadata using strip."""
    try:
        subprocess.run(["strip", "-x", str(path)], check=True, capture_output=True)
        print("  Stripped Mach-O metadata")
    except subprocess.CalledProcessError as e:
        print(f"  Warning: strip failed: {e.stderr.decode()}")
    except FileNotFoundError:
        print("  Warning: strip not found, skipping Mach-O stripping")


def strip_pe_metadata(path: Path):
    """Remove identifying PE metadata using strip (from MinGW/LLVM)."""
    for strip_cmd in ["strip", "llvm-strip"]:
        try:
            subprocess.run(
                [strip_cmd, "--strip-all", str(path)],
                check=True,
                capture_output=True,
            )
            print(f"  Stripped PE metadata with {strip_cmd}")
            return
        except subprocess.CalledProcessError as e:
            print(f"  Warning: {strip_cmd} failed: {e.stderr.decode()}")
        except FileNotFoundError:
            continue

    print("  Warning: no strip tool found, skipping PE stripping")


def strip_binary_metadata(path: Path, remove_rpath: bool = True):
    """Strip metadata from binary based on its format.

    Args:
        path: Path to the binary file.
        remove_rpath: If True, remove ELF RPATH/RUNPATH. Set to False for
            auditwheel-repaired wheels where RPATH is needed to find vendored libs.
    """
    fmt = detect_binary_format(path)

    if fmt == "elf":
        strip_elf_metadata(path, remove_rpath=remove_rpath)
    elif fmt == "macho":
        strip_macho_metadata(path)
    elif fmt == "pe":
        strip_pe_metadata(path)
    else:
        print("  Unknown binary format, skipping metadata stripping")


def scrub_so_file(input_path: Path, output_path: Path, patterns: list):
    """Scrub a binary file directly."""
    print(f"Scrubbing {input_path}...")
    data = input_path.read_bytes()
    scrubbed = scrub_binary(data, patterns)
    output_path.write_bytes(scrubbed)
    strip_binary_metadata(output_path)
    print(f"  Wrote {output_path}")


def scrub_wheel(wheel_path: Path, output_path: Path | None, patterns: list):
    """Scrub binary files inside a wheel (zip archive)."""
    if output_path is None:
        output_path = wheel_path

    print(f"Scrubbing wheel {wheel_path}...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(tmpdir)

        for sbom_dir in tmpdir.rglob("sboms"):
            if sbom_dir.is_dir():
                shutil.rmtree(sbom_dir)

        binary_patterns = ["*.so", "*.pyd", "*.dylib"]
        for pattern in binary_patterns:
            for bin_file in tmpdir.rglob(pattern):
                print(f"  Processing {bin_file.name}...")
                data = bin_file.read_bytes()
                scrubbed = scrub_binary(data, patterns)
                bin_file.write_bytes(scrubbed)
                strip_binary_metadata(bin_file, remove_rpath=False)

        rebuild_wheel_record(tmpdir)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in tmpdir.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(tmpdir)
                    zf.write(file, arcname)

    print(f"  Wrote {output_path}")


def verify_scrub(path: Path, verify_patterns: list[bytes]) -> bool:
    """Verify that sensitive strings are removed."""
    if path.suffix == ".whl":
        data = b""
        with zipfile.ZipFile(path, "r") as zf:
            for name in zf.namelist():
                if name.endswith((".so", ".pyd", ".dylib")):
                    data += zf.read(name)
            if any("/sboms/" in name for name in zf.namelist()):
                print("  WARNING: Wheel still contains SBOM files")
                return False
        if not data:
            print("  WARNING: No binary found in wheel to verify")
            return True
    else:
        data = path.read_bytes()

    leaks = []
    for pattern in verify_patterns:
        matches = re.findall(pattern, data)
        if matches:
            leaks.extend(m.decode("utf-8", errors="replace") for m in matches[:3])

    if leaks:
        print(f"  WARNING: Found {len(leaks)} potential leaks: {leaks[:10]}")
        return False
    else:
        print("  OK: No sensitive patterns found")
        return True


def scrub_main(
    patterns: list,
    verify_patterns: list[bytes],
    binary_suffixes: tuple[str, ...] = (".so", ".dylib"),
):
    """Shared main() entry point for scrub_binary.py scripts."""
    if len(sys.argv) < 2:
        print("Usage: scrub_binary.py <input.so|input.whl> [output]")
        print("")
        print("Scrubs identifying strings from compiled binaries.")
        print("If no output specified, overwrites input (for wheels).")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        sys.exit(1)

    if input_path.suffix == ".whl":
        scrub_wheel(input_path, output_path, patterns)
        if not verify_scrub(output_path or input_path, verify_patterns):
            sys.exit(1)
    elif input_path.suffix in binary_suffixes:
        if output_path is None:
            output_path = input_path
        scrub_so_file(input_path, output_path, patterns)
        if not verify_scrub(output_path, verify_patterns):
            sys.exit(1)
    else:
        print(f"Error: Unknown file type {input_path.suffix}")
        sys.exit(1)
