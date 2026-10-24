"""Copyright, the Moreau authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Shared fixtures for release script tests."""

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture
def release_tree(tmp_path):
    paths = [
        "packaging/yggdrasil/M/Moreau/Moreau_CPU/build_tarballs.jl",
        "packaging/yggdrasil/M/Moreau/Moreau_CUDA/build_tarballs.jl",
        "packages/moreau-julia/Moreau.jl/Project.toml",
        "packages/moreau-julia/Moreau.jl/test/cuda/Project.toml",
        "packages/moreau-julia/MoreauTests.jl/Project.toml",
        "pyproject.toml",
        "uv.lock",
        "packages/moreau-cpu/Cargo.toml",
        "packages/moreau-cpu/Cargo.lock",
        "packages/moreau-cuda/bindings/moreau_bindings.cpp",
        "packages/moreau-cuda/src/solver/info.cpp",
        "packages/moreau-cuda/src/solver/solver.cpp",
        "docs/conf.py",
        "docs/_static/custom.css",
        "docs/installation.md",
        "docs/guide/testing-diagnostics.md",
    ]
    for package in ["moreau", "moreau-cpu", "moreau-cuda"]:
        paths.append(f"packages/{package}/pyproject.toml")
        prefix = "" if package == "moreau-cuda" else "python/"
        paths.append(f"packages/{package}/{prefix}{package.replace('-', '_')}/__init__.py")
    for path in paths:
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / path, dest)
    return tmp_path
