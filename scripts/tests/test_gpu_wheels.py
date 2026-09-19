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

Exercise GPU QA's failure handling without downloading packages or requiring CUDA.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "test_gpu_wheels.sh"


@pytest.fixture
def run_qa(tmp_path):
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    for name in (
        "moreau-0.4.1-py3-none-any.whl",
        "moreau_cpu-0.4.1-cp39-abi3-manylinux_2_28_x86_64.whl",
        "moreau_cuda12-0.4.1-cp312-abi3-manylinux_2_28_x86_64.whl",
        "moreau_cuda13-0.4.1-cp312-abi3-manylinux_2_28_x86_64.whl",
    ):
        (wheels / name).touch()
    source = tmp_path / "source"
    tests = source / "packages/moreau/tests/python"
    tests.mkdir(parents=True)
    for name in ("test_a.py", "test_b.py"):
        (tests / name).touch()

    # Simulate external commands at the process boundary. A failed import,
    # package install, or pytest command must propagate through the real script.
    commands = tmp_path / "bin"
    commands.mkdir()
    shim = commands / "shim"
    shim.write_text(f"#!{sys.executable}\n" + r"""
import json, os, pathlib, shutil, sys
command = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['QA_LOG'], 'a') as log:
    log.write(json.dumps([command, *args]) + '\n')
failure = os.environ.get('QA_FAILURE', '')
if command == 'uname':
    print('x86_64')
elif command == 'uv' and args[0] == 'venv':
    python = pathlib.Path(args[-1]) / 'bin/python'
    python.parent.mkdir(parents=True)
    python.symlink_to(pathlib.Path(__file__).resolve())
elif command == 'uv' and failure == 'install':
    sys.exit(1)
elif command == 'python':
    if args == ['-']:
        sys.stdin.read()
        sys.exit(1 if failure == 'import' else 0)
    if args[:1] == ['-c']:
        print({'cvxpy': '1.9.2', 'cvxpylayers': '1.2.0'}[args[-1]])
    if failure and any(failure in arg for arg in args):
        sys.exit(int(os.environ.get('QA_EXIT_CODE', '1')))
elif command == 'git' and args[0] == 'clone':
    pathlib.Path(args[-1], 'cvxpy/tests').mkdir(parents=True)
elif command == 'gh':
    if args[:2] == ['release', 'download']:
        shutil.copytree(os.environ['QA_WHEELS'], 'wheels')
    elif args[0] == 'api' and '/commits/' in args[1]:
        print('a' * 40)
elif command == 'julia' and failure == 'julia':
    sys.exit(1)
""")
    shim.chmod(0o755)
    for command in ("uv", "git", "uname", "gh", "julia"):
        (commands / command).symlink_to(shim)
    log = tmp_path / "commands.jsonl"

    def run(failure="", exit_code=1, release_suite=None, cuda="12"):
        command = ["bash", str(SCRIPT), str(wheels), "--source-dir", str(source)]
        if release_suite is not None:
            command = [
                "bash",
                str(SCRIPT.with_name("test_release_gpu.sh")),
                "v0.4.1",
                "--suite",
                release_suite,
                "--cuda",
                cuda,
            ]
        result = subprocess.run(
            [*command, "--work-dir", str(tmp_path / "work")],
            env={
                **os.environ,
                "PATH": f"{commands}:{os.environ['PATH']}",
                "QA_LOG": str(log),
                "QA_FAILURE": failure,
                "QA_EXIT_CODE": str(exit_code),
                "QA_WHEELS": str(wheels),
            },
            capture_output=True,
            text=True,
        )
        return result, log.read_text()

    return wheels, run


@pytest.mark.parametrize("wheel_error", ["missing", "duplicate"])
def test_rejects_incomplete_or_ambiguous_wheels(run_qa, wheel_error):
    wheels, run = run_qa
    if wheel_error == "missing":
        next(wheels.glob("moreau_cuda12-*.whl")).unlink()
    else:
        (wheels / "moreau-0.4.0-py3-none-any.whl").touch()
    result, commands = run()
    assert result.returncode != 0
    assert "Expected one wrapper, CPU, and CUDA 12 wheel" in result.stderr
    assert '"uv"' not in commands


@pytest.mark.parametrize("failure", ["install", "import", "test_a.py", "test_conic_solvers"])
def test_failures_fail_qa(run_qa, failure):
    _, run = run_qa
    result, commands = run(failure)
    assert result.returncode != 0
    if failure == "test_a.py":
        assert "test_b.py" in commands  # Still diagnose the remaining Moreau files.
        assert '"git"' not in commands
    elif failure in ("install", "import"):
        assert '"pytest"' not in commands


@pytest.mark.parametrize("optional_module", ["", "test_a.py"])
def test_success_including_optional_empty_module(run_qa, optional_module):
    _, run = run_qa
    result, commands = run(optional_module, exit_code=5)
    assert result.returncode == 0, result.stderr
    assert "python3.14/bin/python" in commands
    assert "test_conic_solvers.py" in commands
    assert "test_moreau_dual_variables.py" in commands
    assert '["git", "clone", "--depth=1", "--branch", "v1.9.2",' in commands
    assert '["git", "clone", "--depth=1", "--branch", "v1.2.0",' in commands


@pytest.mark.parametrize(
    "cuda,suite,failure",
    [
        ("12", "all", ""),
        ("13", "all", ""),
        ("12", "python", ""),
        ("12", "julia", ""),
        ("12", "all", "import"),
        ("12", "all", "julia"),
    ],
)
def test_release_statuses_follow_each_suite_on_tagged_commit(run_qa, cuda, suite, failure):
    _, run = run_qa
    result, commands = run(failure, release_suite=suite, cuda=cuda)
    assert result.returncode == bool(failure), result.stderr
    calls = [json.loads(line) for line in commands.splitlines()]
    updates = [call for call in calls if call[0] == "gh" and "POST" in call]
    expected = []
    for name in ("python", "julia"):
        if suite not in ("all", name):
            continue
        failed = failure == ("import" if name == "python" else "julia")
        expected.extend(
            [
                (f"release-gpu/cuda{cuda}/{name}", "pending"),
                (f"release-gpu/cuda{cuda}/{name}", "failure" if failed else "success"),
            ]
        )
        if failed:
            break
    actual = []
    for call in updates:
        assert f"repos/moreau-project/moreau/statuses/{'a' * 40}" in call
        fields = dict(arg.split("=", 1) for arg in call if "=" in arg)
        actual.append((fields["context"], fields["state"]))
    assert actual == expected
