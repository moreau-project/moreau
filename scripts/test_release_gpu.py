#!/usr/bin/env python3
# Copyright, the Moreau authors
# SPDX-License-Identifier: Apache-2.0
"""Run the release's GPU tests locally with uv and Julia."""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = "moreau-project/moreau"
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="Release tag, e.g. v0.4.1")
    parser.add_argument("--cuda", choices=("12", "13"), default="12")
    parser.add_argument("--suite", choices=("all", "python", "julia"), default="all")
    parser.add_argument(
        "--work-dir", type=Path, help="Directory for downloaded artifacts and environments"
    )
    args = parser.parse_args()
    work = (
        args.work_dir.resolve() if args.work_dir else Path(tempfile.mkdtemp(prefix="moreau-gpu-"))
    )
    work.mkdir(parents=True, exist_ok=True)
    print(f"GPU test environment: {work}", flush=True)
    env = os.environ | {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}

    def run(*command, **kwargs):
        subprocess.run(list(map(str, command)), cwd=work, env=env, check=True, **kwargs)

    source = work / "source"
    run("git", "init", "-q", source)
    run(
        "git",
        "-C",
        source,
        "fetch",
        "--depth=1",
        f"https://github.com/{REPO}.git",
        f"refs/tags/{args.tag}",
    )
    run("git", "-C", source, "checkout", "--detach", "FETCH_HEAD")
    if args.suite != "julia":
        wheels = work / "wheels"
        arch = platform.machine()
        run(
            "gh",
            "release",
            "download",
            args.tag,
            "--repo",
            REPO,
            "--dir",
            wheels,
            "--pattern",
            "moreau-*-py3-none-any.whl",
            "--pattern",
            f"moreau_cpu-*-manylinux*_{arch}.whl",
            "--pattern",
            f"moreau_cuda{args.cuda}-*-manylinux*_{arch}.whl",
        )
        python = work / "python/bin/python"
        run("uv", "venv", "--python", "3.12", work / "python")
        install = ("uv", "pip", "install", "--python", python)
        run(*install, *sorted(wheels.glob("*.whl")))
        torch_index = "cu121" if args.cuda == "12" else "cu130"
        run(*install, "torch", "--default-index", f"https://download.pytorch.org/whl/{torch_index}")
        run(
            *install,
            f"jax[cuda{args.cuda}]",
            "pytest",
            "numpy",
            "scipy",
            "hypothesis",
            "cvxpy",
            "clarabel",
            "git+https://github.com/cvxpy/cvxpylayers.git@v1.1.0",
        )
        run(
            python,
            "-c",
            """
import moreau, torch, jax
assert moreau.device_available('cuda'), moreau.device_error('cuda')
assert torch.cuda.is_available(), 'PyTorch CUDA is unavailable'
assert jax.default_backend() == 'gpu', 'JAX CUDA is unavailable'
""",
        )
        # Separate processes release GPU memory between test files, as in CI.
        for test in sorted((source / "packages/moreau/tests/python").glob("test_*.py")):
            run(python, "-m", "pytest", test, "-v", "--tb=short", "-rs")
        for repo, branch in (("cvxpy", "release/1.8.x"), ("cvxpylayers", "v1.1.0")):
            run(
                "git",
                "clone",
                "--depth=1",
                "--branch",
                branch,
                f"https://github.com/cvxpy/{repo}.git",
                work / repo,
            )
        # Copy CVXPY tests away from its checkout to use the installed package.
        shutil.copytree(work / "cvxpy/cvxpy/tests", work / "cvxpy-tests")
        run(
            python,
            "-m",
            "pytest",
            work / "cvxpy-tests/test_conic_solvers.py",
            "-v",
            "-rs",
            "-k",
            "Moreau",
        )
        run(
            python,
            "-m",
            "pytest",
            work / "cvxpylayers/tests/test_moreau.py",
            work / "cvxpylayers/tests/test_moreau_dual_variables.py",
            "-v",
            "-rs",
        )
    if args.suite != "python":
        run(
            "julia",
            "--startup-file=no",
            ROOT / "scripts/test_release_gpu_julia.jl",
            source,
            "12.2" if args.cuda == "12" else "13.0",
            "true",
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
