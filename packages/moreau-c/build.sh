#!/usr/bin/env bash
# Build Moreau C API shared libraries for distribution.
#
# Usage:
#   packages/moreau-c/build.sh cpu        # CPU shared lib (current platform)
#   packages/moreau-c/build.sh cuda       # CUDA shared lib (requires CUDA toolkit)
#   packages/moreau-c/build.sh all        # Both CPU and CUDA
#
# Environment variables:
#   MOREAU_CUDA_ARCH    CUDA architectures (default: auto-detect or 75;80;86;89;90)
#   MOREAU_OUT_DIR      Output directory (default: dist/c-api/)
#   MOREAU_BUILD_TYPE   cmake build type (default: Release)
#
# Output structure:
#   dist/c-api/
#   ├── LICENSE                   (Apache-2.0)
#   ├── NOTICE                    (Clarabel/diffqcp/DAQP attribution)
#   ├── include/
#   │   └── moreau.h
#   └── lib/
#       ├── libmoreau_cpu.so      (Linux) or libmoreau_cpu.dylib (macOS)
#       └── libmoreau_cuda.so     (Linux, CUDA build only)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_TYPE="${MOREAU_BUILD_TYPE:-Release}"
OUT_DIR="${MOREAU_OUT_DIR:-$REPO_ROOT/dist/c-api}"

TARGET="${1:-all}"

# Detect platform
OS="$(uname -s)"
ARCH="$(uname -m)"

echo "=== Moreau C API Build ==="
echo "Platform: $OS $ARCH"
echo "Build type: $BUILD_TYPE"
echo "Output: $OUT_DIR"
echo ""

# Prepare output directories
mkdir -p "$OUT_DIR/include" "$OUT_DIR/lib"

# Copy header
cp "$REPO_ROOT/packages/moreau-c/include/moreau.h" "$OUT_DIR/include/"

# Copy license files (Apache-2.0 requires LICENSE + NOTICE travel with the
# distributed binaries, which embed Clarabel/diffqcp (Apache-2.0) and DAQP (MIT) code)
cp "$REPO_ROOT/packages/moreau-c/LICENSE" "$OUT_DIR/"
cp "$REPO_ROOT/packages/moreau-c/NOTICE" "$OUT_DIR/"

# ---------------------------------------------------------------------------
# CPU build (Rust)
# ---------------------------------------------------------------------------
build_cpu() {
    echo "--- Building CPU C API (Rust) ---"
    cd "$REPO_ROOT/packages/moreau-cpu"

    cargo build --release --features c-api

    local target_dir="$REPO_ROOT/packages/moreau-cpu/target/release"
    if [ "$OS" = "Darwin" ]; then
        cp "$target_dir/libmoreau.dylib" "$OUT_DIR/lib/libmoreau_cpu.dylib"
        echo "Built: $OUT_DIR/lib/libmoreau_cpu.dylib"
    else
        cp "$target_dir/libmoreau.so" "$OUT_DIR/lib/libmoreau_cpu.so"
        echo "Built: $OUT_DIR/lib/libmoreau_cpu.so"
    fi

    # Also copy the static lib if available
    if [ -f "$target_dir/libmoreau.a" ]; then
        cp "$target_dir/libmoreau.a" "$OUT_DIR/lib/libmoreau_cpu.a"
    fi
}

# ---------------------------------------------------------------------------
# CUDA build (CMake)
# ---------------------------------------------------------------------------
build_cuda() {
    echo "--- Building CUDA C API (CMake) ---"

    if [ "$OS" = "Darwin" ]; then
        echo "ERROR: CUDA build not supported on macOS"
        exit 1
    fi

    local build_dir="$REPO_ROOT/packages/moreau-cuda/build-c-api"
    mkdir -p "$build_dir"
    cd "$build_dir"

    local cmake_args=(
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
        -DMOREAU_BUILD_C_SHARED=ON
        -DMOREAU_BUILD_PYTHON=OFF
        -DMOREAU_BUILD_TESTS=OFF
        -DMOREAU_BUILD_EXAMPLES=OFF
    )

    if [ -n "${MOREAU_CUDA_ARCH:-}" ]; then
        cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$MOREAU_CUDA_ARCH")
    fi

    cmake .. "${cmake_args[@]}"
    make -j"$(nproc)" moreau_cuda_shared

    # Copy the real file (not symlinks) and recreate symlinks in output
    local real_so
    real_so="$(ls "$build_dir"/libmoreau_cuda.so.*.*.* 2>/dev/null | head -1)"
    if [ -z "$real_so" ]; then
        echo "ERROR: libmoreau_cuda.so not found in $build_dir"
        exit 1
    fi

    cp "$real_so" "$OUT_DIR/lib/"
    # Recreate soname symlinks
    local basename_real
    basename_real="$(basename "$real_so")"
    cd "$OUT_DIR/lib"
    ln -sf "$basename_real" libmoreau_cuda.so.0
    ln -sf libmoreau_cuda.so.0 libmoreau_cuda.so
    cd -

    echo "Built: $OUT_DIR/lib/libmoreau_cuda.so"
}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
case "$TARGET" in
    cpu)
        build_cpu
        ;;
    cuda)
        build_cuda
        ;;
    all)
        build_cpu
        if command -v nvcc &>/dev/null; then
            build_cuda
        else
            echo "Skipping CUDA build (nvcc not found)"
        fi
        ;;
    *)
        echo "Usage: $0 {cpu|cuda|all}"
        exit 1
        ;;
esac

echo ""
echo "=== Output ==="
find "$OUT_DIR" -type f -o -type l | sort
echo ""
echo "Done."
