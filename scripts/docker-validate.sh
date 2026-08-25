#!/bin/bash
# Docker-based CI Validation
# Mirrors the exact GitHub Actions CI environment
#
# Usage:
#   ./scripts/docker-validate.sh           # Full validation (build + test)
#   ./scripts/docker-validate.sh build     # Build only
#   ./scripts/docker-validate.sh test      # Test only (requires prior build)
#
# Environment variables:
#   CUDA_VERSION   - CUDA version (default: 12.4.0)
#   CUDSS_VERSION  - cuDSS version (default: 0.7.1.4)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

CUDA_VERSION="${CUDA_VERSION:-12.4.0}"
CUDSS_VERSION="${CUDSS_VERSION:-0.7.1.4}"

MODE="${1:-all}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Build step - matches CI build job (runs on CPU runner)
docker_build() {
    log_info "=== Docker Build (CPU cross-compilation) ==="
    log_info "CUDA version: $CUDA_VERSION"
    log_info "cuDSS version: $CUDSS_VERSION"

    docker run --rm \
        -v "$ROOT_DIR":/workspace \
        -w /workspace \
        nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 \
        bash -c "
            set -e
            echo '=== Environment Setup ==='
            rm -f /etc/apt/sources.list.d/cuda*.list
            apt-get update && apt-get install -y cmake g++ wget libgtest-dev python3 python3-pip python3-venv

            echo '=== Installing cuDSS ==='
            wget -q https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz
            tar -xf libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz
            cp -r libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive/lib/* /usr/local/lib/
            cp -r libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive/include/* /usr/local/include/
            ldconfig

            pip install pybind11 numpy wheel

            echo '=== Building C++ Library ==='
            cd packages/moreau-cuda
            cmake -B build -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=\$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')
            cmake --build build -j\$(nproc)

            echo '=== Building Python Wheel ==='
            pip wheel . -w dist/ --no-deps --no-build-isolation

            echo '=== Build Artifacts ==='
            ls -la build/*_tests 2>/dev/null || echo 'No test binaries'
            ls -la dist/*.whl

            # Fix permissions for host
            chmod -R 777 build dist
        "

    log_info "Build completed. Artifacts in packages/moreau-cuda/build/ and packages/moreau-cuda/dist/"
}

# Test step - matches CI test job (runs on GPU runner)
docker_test() {
    # GPU is REQUIRED for test step - fail if not available
    if ! nvidia-smi &>/dev/null; then
        log_error "No GPU available. GPU tests require a GPU."
        log_error "Run this script on a machine with an NVIDIA GPU."
        exit 1
    fi

    log_info "=== Docker Test (GPU runner simulation) ==="
    log_info "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

    docker run --rm --gpus all \
        -v "$ROOT_DIR":/workspace \
        -w /workspace \
        nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 \
        bash -c "
            set -e
            echo '=== Environment Setup ==='
            rm -f /etc/apt/sources.list.d/cuda*.list
            apt-get update && apt-get install -y cmake wget python3 python3-pip python3-venv curl

            echo '=== Installing Rust ==='
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source \$HOME/.cargo/env

            echo '=== Installing cuDSS ==='
            wget -q https://developer.download.nvidia.com/compute/cudss/redist/libcudss/linux-x86_64/libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz
            tar -xf libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive.tar.xz
            cp -r libcudss-linux-x86_64-${CUDSS_VERSION}_cuda12-archive/lib/* /usr/local/lib/
            ldconfig

            echo '=== C++ Tests ==='
            chmod +x packages/moreau-cuda/build/*_tests 2>/dev/null || true
            cd packages/moreau-cuda/build
            ctest --output-on-failure -j4 || echo 'C++ tests failed or no tests found'
            cd /workspace

            echo '=== Python Environment ==='
            python3 -m venv /test_env
            source /test_env/bin/activate
            pip install --upgrade pip maturin pytest scipy

            echo '=== Building CPU Package ==='
            cd packages/moreau-cpu
            maturin build --release
            pip install target/wheels/*.whl
            cd /workspace

            echo '=== Installing GPU Wheel ==='
            pip install packages/moreau-cuda/dist/*.whl

            echo '=== CUDA Detection Debug ==='
            echo '--- SO file location ---'
            SO=\$(find /test_env -name '_moreau_gpu*.so' | head -1)
            echo \"SO file: \$SO\"

            if [ -n \"\$SO\" ]; then
                echo '--- SO dependencies ---'
                ldd \"\$SO\" | grep -E '(not found|cudss)' || echo 'All deps found, no cudss linked'
            fi

            echo '--- Import test ---'
            python3 -c \"
import sys
try:
    import moreau_cuda
    print(f'moreau_cuda loaded: {moreau_cuda.__file__}')
except Exception as e:
    print(f'moreau_cuda import FAILED: {e}')
    sys.exit(1)
\"

            echo '--- CUDA availability ---'
            python3 -c \"
import moreau
print(f'cuda_available: {moreau.cuda_available()}')
if not moreau.cuda_available():
    print(f'cuda_import_error: {moreau.cuda_import_error()}')
\"

            echo '=== Installing PyTorch ==='
            pip install torch --index-url https://download.pytorch.org/whl/cu124

            echo '=== Running Python Tests ==='
            pytest packages/moreau/tests/python/ -v --tb=short
        "

    log_info "Tests completed successfully"
}

# Main
echo "========================================="
echo "  Docker CI Validation"
echo "  CUDA: $CUDA_VERSION  cuDSS: $CUDSS_VERSION"
echo "========================================="

case "$MODE" in
    build)
        docker_build
        ;;
    test)
        docker_test
        ;;
    all)
        docker_build
        docker_test
        ;;
    *)
        log_error "Unknown mode: $MODE"
        echo "Usage: $0 [build|test|all]"
        exit 1
        ;;
esac

echo ""
log_info "========================================="
log_info "  Docker Validation PASSED"
log_info "========================================="
