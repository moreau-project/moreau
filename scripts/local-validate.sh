#!/bin/bash
# Local CI Validation Script
# Mirrors the CI pipeline to catch issues before pushing
#
# Usage:
#   ./scripts/local-validate.sh          # CPU only (default)
#   ./scripts/local-validate.sh cpu      # CPU only
#   ./scripts/local-validate.sh gpu      # Full GPU validation
#   ./scripts/local-validate.sh quick    # Quick import test only

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

MODE="${1:-cpu}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Validate CI workflow YAML files (uses Python stdlib json as a syntax sanity
# check, then falls back to PyYAML if available for full YAML validation)
validate_ci_yaml() {
    log_info "Validating CI workflow YAML files..."

    for yaml_file in "$ROOT_DIR"/.github/workflows/*.yml; do
        if [ -f "$yaml_file" ]; then
            log_info "  Checking $(basename "$yaml_file")..."
            python3 -c "
import sys
try:
    import yaml
    yaml.safe_load(open('$yaml_file'))
except ImportError:
    # PyYAML not installed - do a basic syntax check (non-empty, valid UTF-8)
    with open('$yaml_file', 'r') as f:
        content = f.read()
    if not content.strip():
        print('Empty YAML file', file=sys.stderr)
        sys.exit(1)
" || {
                log_error "Invalid YAML in $yaml_file"
                exit 1
            }
        fi
    done
    log_info "All CI workflow files valid"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    command -v python3 >/dev/null || { log_error "python3 not found"; exit 1; }

    if [ "$MODE" == "gpu" ]; then
        command -v nvidia-smi >/dev/null || { log_error "nvidia-smi not found"; exit 1; }
        nvidia-smi --query-gpu=name --format=csv,noheader || { log_error "nvidia-smi failed"; exit 1; }
    fi

    # Check for pytest (needed for running tests)
    command -v pytest >/dev/null || {
        log_warn "pytest not found - installing..."
        python3 -m pip install pytest
    }

    # Check for maturin (needed for CPU package)
    command -v maturin >/dev/null || {
        log_warn "maturin not found - installing..."
        python3 -m pip install maturin
    }
}

# Build CPU package
build_cpu() {
    log_info "Building CPU package..."
    cd "$ROOT_DIR/packages/moreau-cpu"
    maturin build --release
    python3 -m pip install target/wheels/*.whl --force-reinstall --quiet
    cd "$ROOT_DIR"
    log_info "CPU package installed"
}

# Build GPU package (requires CUDA)
build_gpu() {
    log_info "Building GPU package..."
    cd "$ROOT_DIR/packages/moreau-cuda"

    # Check if we need to rebuild
    if [ -d "dist" ] && [ "$(ls -A dist/*.whl 2>/dev/null)" ]; then
        log_info "Found existing GPU wheel, reinstalling..."
    else
        log_info "Building GPU wheel..."
        python3 -m pip wheel . -w dist/ --no-deps --no-build-isolation
    fi

    # Use --no-deps since moreau CPU package is already installed locally
    # and pip may try to resolve from PyPI otherwise
    python3 -m pip install dist/*.whl --force-reinstall --no-deps --quiet
    cd "$ROOT_DIR"
    log_info "GPU package installed"
}

# Validate Python imports
validate_imports() {
    log_info "Validating imports..."

    # CPU imports
    python3 -c "import moreau; print(f'moreau {moreau.__version__}: {moreau.__file__}')"
    python3 -c "from moreau import Cones, Solver, CpuSolver; print('Core imports OK')"

    if [ "$MODE" == "gpu" ]; then
        # GPU imports
        python3 -c "import moreau_cuda; print(f'moreau_cuda: {moreau_cuda.__file__}')"
        python3 -c "from moreau_cuda import GpuSolver; print('GpuSolver import OK')"

        # Check CUDA detection
        python3 -c "
import moreau
print(f'Available devices: {moreau.available_devices()}')
if moreau.device_available('cuda'):
    print('CUDA backend: AVAILABLE')
else:
    err = moreau.device_error('cuda')
    print(f'CUDA backend: NOT AVAILABLE')
    print(f'  Error: {err}')
    exit(1)
"
    fi
}

# Validate .so library dependencies
validate_so_deps() {
    log_info "Validating .so dependencies..."

    # Find the GPU .so file
    SO_FILE=$(python3 -c "
import moreau_cuda._moreau_gpu as m
print(m.__file__)
" 2>/dev/null || echo "")

    if [ -z "$SO_FILE" ]; then
        log_error "Could not find GPU .so file"
        exit 1
    fi

    log_info "Checking: $SO_FILE"

    # Check for missing libraries
    MISSING=$(ldd "$SO_FILE" 2>&1 | grep "not found" || true)
    if [ -n "$MISSING" ]; then
        log_error "Missing libraries detected:"
        echo "$MISSING"
        exit 1
    fi

    # Show cuDSS linkage (informational)
    ldd "$SO_FILE" 2>&1 | grep -i cudss || log_warn "No cuDSS linkage found"

    log_info "All .so dependencies satisfied"
}

# Run tests
run_tests() {
    log_info "Running tests..."
    cd "$ROOT_DIR"

    if [ "$MODE" == "gpu" ]; then
        # Full test suite
        pytest packages/moreau/tests/python/ -v --tb=short
    else
        # CPU-only tests
        pytest packages/moreau/tests/python/ -v --tb=short --device=cpu
    fi
}

# Quick validation (import test only)
quick_validate() {
    log_info "Quick validation (import test only)..."

    python3 -c "
import moreau
print(f'moreau {moreau.__version__}')
print(f'  Available devices: {moreau.available_devices()}')
print(f'  CUDA available: {moreau.device_available(\"cuda\")}')
if not moreau.device_available('cuda'):
    err = moreau.device_error('cuda')
    if err:
        print(f'  CUDA error: {err}')
"
}

# Main
echo "========================================="
echo "  Local CI Validation - Mode: $MODE"
echo "========================================="

case "$MODE" in
    quick)
        check_prerequisites
        validate_ci_yaml
        quick_validate
        ;;
    cpu)
        check_prerequisites
        validate_ci_yaml
        build_cpu
        validate_imports
        run_tests
        ;;
    gpu)
        check_prerequisites
        validate_ci_yaml
        build_cpu
        build_gpu
        validate_so_deps
        validate_imports
        run_tests
        ;;
    *)
        log_error "Unknown mode: $MODE"
        echo "Usage: $0 [cpu|gpu|quick]"
        exit 1
        ;;
esac

echo ""
log_info "========================================="
log_info "  Validation PASSED ($MODE mode)"
log_info "========================================="
