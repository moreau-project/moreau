#!/bin/bash
# ci-build.sh - Build and test script for fresh environment CI
# Run inside Docker container or on self-hosted runner

set -e  # Exit on any error

echo "========================================"
echo "Moreau Fresh Build CI"
echo "========================================"

# Show Python version
echo ""
echo "Python version:"
python3 --version
which python3

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

FAILED=0

# Function to run a step and track failures
run_step() {
    local name="$1"
    shift
    echo ""
    echo "----------------------------------------"
    echo "Running: $name"
    echo "----------------------------------------"
    if "$@"; then
        echo -e "${GREEN}PASSED${NC}: $name"
    else
        echo -e "${RED}FAILED${NC}: $name"
        FAILED=1
    fi
}

# Verify GPU is available
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
    echo "WARNING: No GPU detected. Tests will fail."
}

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Activated virtual environment"
fi

# Run C++ tests
echo ""
echo "========================================"
echo "C++ Tests (ctest)"
echo "========================================"
cd /workspace/build 2>/dev/null || cd build
run_step "C++ unit tests" ctest --output-on-failure

# Run Python tests
echo ""
echo "========================================"
echo "Python Tests (pytest)"
echo "========================================"
cd /workspace 2>/dev/null || cd ..
run_step "Python tests" pytest packages/moreau/tests/python/ -v --tb=short

# Optional: Build and test PyTorch bindings
if [ "${BUILD_PYTORCH:-true}" = "true" ]; then
    echo ""
    echo "========================================"
    echo "PyTorch Extension Tests"
    echo "========================================"

    # Check if PyTorch is installed
    if python -c "import torch" 2>/dev/null; then
        echo "PyTorch detected, running torch tests..."

        # Run tests and capture output (set +e to prevent script exit on failure)
        set +e
        OUTPUT=$(pytest packages/moreau/tests/python/test_torch_integration.py -v --tb=short 2>&1)
        RESULT=$?
        set -e
        echo "$OUTPUT"

        # Check if ALL tests were skipped (indicates build issue)
        if echo "$OUTPUT" | grep -qE "^\s*=+ .* skipped.* =+$" && ! echo "$OUTPUT" | grep -qE "(passed|failed)"; then
            echo -e "${RED}ERROR: All PyTorch tests were skipped!${NC}"
            echo "This indicates _moreau_torch extension was not built correctly."
            echo "Check that BUILD_PYTORCH_EXTENSION=ON was passed to cmake."
            FAILED=1
        elif [ $RESULT -ne 0 ]; then
            echo -e "${RED}FAILED${NC}: PyTorch integration tests"
            FAILED=1
        else
            echo -e "${GREEN}PASSED${NC}: PyTorch integration tests"
        fi
    else
        echo "PyTorch not installed, skipping torch tests"
    fi
fi

# Summary
echo ""
echo "========================================"
echo "Summary"
echo "========================================"
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
