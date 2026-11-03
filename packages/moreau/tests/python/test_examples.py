"""Run every examples/*.py script in a subprocess; each must exit 0."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[4] / "examples"
OPTIONAL_DEPS = {
    "batch_permutations_torch.py": ["torch"],
    "mpc_cvxpylayers.py": ["torch", "cvxpylayers"],
}
SCRIPTS = sorted(p.name for p in EXAMPLES.glob("*.py") if p.name != "utils.py")


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs(script):
    for dep in OPTIONAL_DEPS.get(script, []):
        if importlib.util.find_spec(dep) is None:
            pytest.skip(f"{dep} not installed")
    result = subprocess.run(
        [sys.executable, script], cwd=EXAMPLES, capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
