#!/usr/bin/env bash
# Usage: ./scripts/bump-version.sh 0.1.1
set -euo pipefail
NEW_VERSION="${1:?Usage: bump-version.sh <version>}"

# 8 Python/Rust version locations
sed -i 's/^version = ".*"/version = "'"$NEW_VERSION"'"/' pyproject.toml
sed -i 's/^version = ".*"/version = "'"$NEW_VERSION"'"/' packages/moreau/pyproject.toml
sed -i 's/__version__ = ".*"/__version__ = "'"$NEW_VERSION"'"/' packages/moreau/python/moreau/__init__.py
sed -i 's/^version = ".*"/version = "'"$NEW_VERSION"'"/' packages/moreau-cpu/pyproject.toml
# Cargo.toml: only replace the [package] version (first occurrence), not dependency versions
sed -i '0,/^version = ".*"/{s/^version = ".*"/version = "'"$NEW_VERSION"'"/}' packages/moreau-cpu/Cargo.toml
sed -i 's/__version__ = ".*"/__version__ = "'"$NEW_VERSION"'"/' packages/moreau-cpu/python/moreau_cpu/__init__.py
sed -i 's/^version = ".*"/version = "'"$NEW_VERSION"'"/' packages/moreau-cuda/pyproject.toml
sed -i 's/__version__ = ".*"/__version__ = "'"$NEW_VERSION"'"/' packages/moreau-cuda/moreau_cuda/__init__.py

# 3 C++/CUDA source version strings
sed -i 's/m\.attr("__version__") = ".*"/m.attr("__version__") = "'"$NEW_VERSION"'"/' packages/moreau-cuda/bindings/moreau_bindings.cpp
sed -i 's/MOREAU_VERSION = ".*"/MOREAU_VERSION = "'"$NEW_VERSION"'"/' packages/moreau-cuda/src/solver/info.cpp
sed -i 's/constexpr const char\* VERSION = ".*"/constexpr const char* VERSION = "'"$NEW_VERSION"'"/' packages/moreau-cuda/src/solver/solver.cpp

# 3 docs version strings
sed -i 's/^release = ".*"/release = "'"$NEW_VERSION"'"/' docs/conf.py
sed -i 's/content: "v[0-9][0-9.]*"/content: "v'"$NEW_VERSION"'"/' docs/_static/custom.css
sed -i 's/moreau-cpu | >= [0-9][0-9.]*/moreau-cpu | >= '"$NEW_VERSION"'/' docs/installation.md
sed -i 's/moreau-cuda12 | >= [0-9][0-9.]*/moreau-cuda12 | >= '"$NEW_VERSION"'/' docs/installation.md
sed -i 's/moreau-cuda13 | >= [0-9][0-9.]*/moreau-cuda13 | >= '"$NEW_VERSION"'/' docs/installation.md

# 4 dependency pins
sed -i 's/moreau-cpu>=[0-9][0-9.]*/moreau-cpu>='"$NEW_VERSION"'/' packages/moreau/pyproject.toml
sed -i 's/moreau-cuda12>=[0-9][0-9.]*/moreau-cuda12>='"$NEW_VERSION"'/' packages/moreau/pyproject.toml
sed -i 's/moreau-cuda13>=[0-9][0-9.]*/moreau-cuda13>='"$NEW_VERSION"'/' packages/moreau/pyproject.toml
sed -i 's/"moreau>=[0-9][0-9.]*"/"moreau>='"$NEW_VERSION"'"/' packages/moreau-cuda/pyproject.toml

# Verify
echo "Version bumped to $NEW_VERSION. Verify:"
grep '^version' pyproject.toml packages/*/pyproject.toml packages/moreau-cpu/Cargo.toml
grep '__version__' packages/moreau/python/moreau/__init__.py packages/moreau-cpu/python/moreau_cpu/__init__.py packages/moreau-cuda/moreau_cuda/__init__.py
grep 'moreau-c' packages/moreau/pyproject.toml packages/moreau-cuda/pyproject.toml
grep 'VERSION\|release\|version' packages/moreau-cuda/bindings/moreau_bindings.cpp packages/moreau-cuda/src/solver/info.cpp packages/moreau-cuda/src/solver/solver.cpp docs/conf.py
grep "$NEW_VERSION" docs/_static/custom.css docs/installation.md
