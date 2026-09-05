#!/usr/bin/env bash
# Usage: ./scripts/bump-version.sh 0.4.0
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$SCRIPT_DIR/bump_version.py" "$@"
