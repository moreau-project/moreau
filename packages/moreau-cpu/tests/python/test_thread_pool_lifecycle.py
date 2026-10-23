"""A released CPU pool must not leave a fresh solver using pre-fork workers."""

import gc
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _make_solver():
    from moreau_cpu import _cpu_solver

    settings = _cpu_solver.DefaultSettings()
    settings.verbose = False
    solver = _cpu_solver.CompiledSolver(
        1,
        1,
        [0, 1],
        [0],
        [0, 1],
        [0],
        [_cpu_solver.NonnegativeConeT(1)],
        settings=settings,
        num_threads=2,
    )
    solver.setup([[1.0]], [[1.0]])
    return solver


def _check_solution(solver):
    solution = solver.solve([[-1.0]], [[10.0]])[0]
    assert abs(solution.x[0] - 1.0) < 1e-6


def _solve_in_child():
    _check_solution(_make_solver())


def _fork_after_releasing_solvers():
    first = _make_solver()
    second = _make_solver()
    _check_solution(first)
    del first
    gc.collect()
    # Dropping one owner must not shut down another solver's workers.
    _check_solution(second)
    del second
    gc.collect()

    child = multiprocessing.get_context("fork").Process(target=_solve_in_child)
    child.start()
    try:
        child.join(timeout=10)
        assert not child.is_alive(), "fresh child solver hung after parent solvers were released"
        assert child.exitcode == 0, f"child solver exited with {child.exitcode}"
    finally:
        if child.is_alive():
            child.kill()
        child.join()
        child.close()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="requires the fork start method"
)
def test_fresh_solver_after_releasing_parent_solvers_and_forking():
    # Use a fresh interpreter: the pytest process may have other live solvers,
    # CUDA contexts, or framework threads from earlier tests.
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _fork_after_releasing_solvers()
