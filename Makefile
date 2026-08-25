# Moreau Build & Validation Makefile
#
# Usage:
#   make validate          # CPU validation (default)
#   make validate-cpu      # CPU validation
#   make validate-gpu      # GPU validation (requires GPU)
#   make docker-test       # Full Docker CI simulation
#   make docker-build      # Docker build only
#   make test              # Run all tests
#   make test-cpu          # Run CPU-only tests
#   make build-cpu         # Build CPU package
#   make build-gpu         # Build GPU package
#   make clean             # Clean build artifacts

.PHONY: all validate validate-cpu validate-gpu docker-test docker-build \
        test test-cpu build-cpu build-gpu install-cpu install-gpu clean verify help

# Default target
all: validate

# Validation targets
validate: validate-cpu

validate-cpu:
	@echo "=== CPU Validation ==="
	./scripts/local-validate.sh cpu

validate-gpu:
	@echo "=== GPU Validation ==="
	./scripts/local-validate.sh gpu

validate-quick:
	@echo "=== Quick Validation ==="
	./scripts/local-validate.sh quick

# Docker-based CI simulation
docker-test:
	@echo "=== Docker CI Test (full) ==="
	./scripts/docker-validate.sh all

docker-build:
	@echo "=== Docker CI Build ==="
	./scripts/docker-validate.sh build

docker-test-only:
	@echo "=== Docker CI Test (reuse build) ==="
	./scripts/docker-validate.sh test

# Test targets
test:
	pytest packages/moreau/tests/python/ -v --tb=short

test-cpu:
	pytest packages/moreau/tests/python/ -v --tb=short --device=cpu

test-gpu:
	pytest packages/moreau/tests/python/ -v --tb=short --device=cuda

# Build targets
build-cpu:
	cd packages/moreau-cpu && maturin build --release

build-gpu:
	cd packages/moreau-cuda && pip wheel . -w dist/ --no-deps --no-build-isolation

# Install targets (development)
install-cpu: build-cpu
	pip install packages/moreau-cpu/target/wheels/*.whl --force-reinstall

install-gpu: build-gpu
	pip install packages/moreau-cuda/dist/*.whl --force-reinstall

install-all: install-cpu install-gpu

# Development install (editable-ish via maturin develop)
dev-cpu:
	cd packages/moreau-cpu && maturin develop --release

# Clean build artifacts
clean:
	rm -rf packages/moreau-cpu/target/wheels/*.whl
	rm -rf packages/moreau-cuda/dist/*.whl
	rm -rf packages/moreau-cuda/build
	rm -rf build_torch build_profile build_test
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Verify builds are up-to-date
verify:
	@python3 scripts/verify-build.py

# Check CUDA status
cuda-status:
	@python3 -c "import moreau; \
		print(f'CUDA available: {moreau.cuda_available()}'); \
		err = moreau.cuda_import_error(); \
		print(f'Import error: {err}' if err else 'No import errors')"

# Help
help:
	@echo "Moreau Build & Validation"
	@echo ""
	@echo "Validation:"
	@echo "  make validate        - Run CPU validation (default)"
	@echo "  make validate-cpu    - Run CPU validation"
	@echo "  make validate-gpu    - Run GPU validation (requires GPU)"
	@echo "  make validate-quick  - Quick import test only"
	@echo ""
	@echo "Docker CI:"
	@echo "  make docker-test     - Full Docker CI simulation"
	@echo "  make docker-build    - Docker build only"
	@echo "  make docker-test-only- Docker test with existing build"
	@echo ""
	@echo "Tests:"
	@echo "  make test            - Run all tests"
	@echo "  make test-cpu        - Run CPU-only tests"
	@echo "  make test-gpu        - Run GPU tests"
	@echo ""
	@echo "Build:"
	@echo "  make build-cpu       - Build CPU wheel"
	@echo "  make build-gpu       - Build GPU wheel"
	@echo "  make install-cpu     - Build and install CPU package"
	@echo "  make install-gpu     - Build and install GPU package"
	@echo "  make install-all     - Install both packages"
	@echo "  make dev-cpu         - Development install CPU (maturin develop)"
	@echo ""
	@echo "Other:"
	@echo "  make verify          - Check if builds are up-to-date"
	@echo "  make cuda-status     - Check CUDA availability"
	@echo "  make clean           - Clean build artifacts"
