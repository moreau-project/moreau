"""Tests for auto-tune on first solve.

Verifies that:
1. auto_tune=False (default) uses heuristic, no benchmarking
2. auto_tune=True triggers benchmarking on first solve
3. Second solve does NOT re-benchmark (auto-tune only fires once)
4. Explicit device + explicit method skips auto-tune entirely
5. device='auto' + explicit method only tunes device
6. Explicit device + method='auto' benchmarks methods on first solve
7. Solve results are correct after auto-tune
8. _tune_result is accessible after auto-tune
9. solver_methods_for_device returns correct candidates
10. torch.Solver and jax.Solver auto-tune via first solve()
"""

import numpy as np
import pytest
import warnings
from unittest.mock import patch

import moreau
from moreau._backend import solver_methods_for_device, available_devices, auto_tune_candidates


@pytest.fixture(autouse=True)
def clear_device_override():
    """Clear set_default_device() so auto-tune tests can fire with device='auto'."""
    moreau.set_default_device(None)
    yield
    moreau.set_default_device(None)


try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def _result_key(device, method):
    """Build the expected result key for a device:method pair."""
    return f"{device}:{method}"


def _make_compiled_solver(device, batch_size=4, method="auto", auto_tune=False):
    """Create a CompiledSolver with setup() already called."""
    n, m = 2, 3
    cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
    ipm = moreau.IPMSettings(direct_solve_method=method)
    settings = moreau.Settings(
        device=device,
        batch_size=batch_size,
        ipm_settings=ipm,
        auto_tune=auto_tune,
        solver="ipm",  # Force IPM so auto-tune tests don't get redirected to active-set
    )

    solver = moreau.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=[0, 1, 2],
        P_col_indices=[0, 1],
        A_row_offsets=[0, 2, 3, 4],
        A_col_indices=[0, 1, 0, 1],
        cones=cones,
        settings=settings,
    )
    solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0, 1.0, 1.0])
    return solver


# ── auto_tune_candidates helper tests ────────────────────────────────


class TestAutoTuneCandidates:
    """Test the auto_tune_candidates() helper."""

    def test_auto_device_returns_all_combos(self):
        """device='auto' returns all devices × methods."""
        combos = auto_tune_candidates("auto", "auto")
        devices = available_devices()
        expected = set()
        for dev in devices:
            for m in solver_methods_for_device(dev):
                expected.add((dev, m))
        assert set(combos) == expected

    def test_auto_device_heuristic_ordering_small(self):
        """Small problem: CPU candidates come first."""
        combos = auto_tune_candidates("auto", "auto", n=10, m=20, nnz_A=50)
        # First candidate should be CPU (small problem)
        assert combos[0][0] == "cpu"

    def test_auto_device_heuristic_ordering_large(self):
        """Large problem: GPU candidates come first (if available)."""
        combos = auto_tune_candidates("auto", "auto", n=1000, m=2000, nnz_A=100_000)
        if len(available_devices()) > 1:
            # GPU should be first for large problem
            assert combos[0][0] != "cpu"

    def test_explicit_device_auto_method(self):
        """Explicit device + method='auto' returns methods for that device."""
        combos = auto_tune_candidates("cpu", "auto")
        expected = set(("cpu", m) for m in solver_methods_for_device("cpu"))
        assert set(combos) == expected

    def test_explicit_device_explicit_method(self):
        """Both explicit → empty list (nothing to benchmark)."""
        combos = auto_tune_candidates("cpu", "qdldl")
        assert combos == []

    def test_auto_device_explicit_method(self):
        """device='auto' + explicit method still returns all combos."""
        combos = auto_tune_candidates("auto", "qdldl")
        # auto device means we try all devices × their methods
        assert len(combos) > 0


class TestSolverMethodsForDevice:
    """Test the solver_methods_for_device helper."""

    def test_cpu_methods(self):
        methods = solver_methods_for_device("cpu")
        assert methods == ["qdldl", "faer"]

    def test_cuda_methods(self):
        methods = solver_methods_for_device("cuda")
        assert methods == ["cudss"]

    def test_unknown_device_returns_qdldl(self):
        methods = solver_methods_for_device("tpu")
        assert methods == ["qdldl"]


# ── Default behavior: auto_tune=False (heuristic, no benchmarking) ───


class TestAutoTuneDisabledByDefault:
    """Test that auto_tune=False (default) uses heuristic without benchmarking."""

    def test_default_auto_tune_is_false(self):
        """Settings defaults to auto_tune=False."""
        settings = moreau.Settings()
        assert settings.auto_tune is False

    def test_auto_device_no_benchmark_by_default(self):
        """device='auto' with auto_tune=False does NOT benchmark."""
        solver = _make_compiled_solver("auto", method="auto", auto_tune=False)
        assert not solver._auto_tuned

        qs = [[2.0, 1.0]] * 4
        bs = [[1.0, 0.7, 0.7]] * 4

        # Should NOT emit a UserWarning about benchmarking
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            solution = solver.solve(qs, bs)

        # No benchmarking occurred
        assert not solver._auto_tuned
        assert solver._tune_result is None

        # But solve still works correctly
        assert solution.x.shape == (4, 2)
        for s in solver.info.status:
            assert s in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)

    def test_auto_device_heuristic_picks_device(self):
        """device='auto' with auto_tune=False resolves to a concrete device."""
        solver = _make_compiled_solver("auto", method="auto", auto_tune=False)
        # Device should already be resolved at construction time
        assert solver.device in available_devices()
        assert solver.device != "auto"

    def test_auto_method_heuristic_picks_method(self):
        """method='auto' with auto_tune=False resolves to a concrete method."""
        solver = _make_compiled_solver("cpu", method="auto", auto_tune=False)
        # Method should be resolved heuristically
        assert solver._settings.ipm_settings.direct_solve_method != "auto"
        assert solver._settings.ipm_settings.direct_solve_method in solver_methods_for_device("cpu")

    def test_explicit_device_and_method_works(self):
        """Explicit device + explicit method works with auto_tune=False."""
        solver = _make_compiled_solver("cpu", method="qdldl", auto_tune=False)
        solution = solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)
        assert solution.x.shape == (4, 2)


# ── CompiledSolver auto-tune tests (auto_tune=True) ─────────────────


class TestAutoTuneOnFirstSolve:
    """Test that first solve() triggers auto-tune when auto_tune=True."""

    def test_auto_device_triggers_tune(self):
        """device='auto' + auto_tune=True triggers auto-tune on first solve."""
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        assert not solver._auto_tuned

        qs = [[2.0, 1.0]] * 4
        bs = [[1.0, 0.7, 0.7]] * 4
        solution = solver.solve(qs, bs)

        assert solver._auto_tuned
        assert solver._tune_result is not None
        # Should have benchmarked all devices × methods
        devices = available_devices()
        expected_keys = set()
        for dev in devices:
            for m in solver_methods_for_device(dev):
                expected_keys.add(f"{dev}:{m}")
        assert set(solver._tune_result.results.keys()) == expected_keys

    def test_explicit_device_auto_method_benchmarks_on_first_solve(self, device):
        """Explicit device + method='auto' + auto_tune=True benchmarks methods."""
        solver = _make_compiled_solver(device, method="auto", auto_tune=True)
        assert not solver._auto_tuned

        qs = [[2.0, 1.0]] * 4
        bs = [[1.0, 0.7, 0.7]] * 4
        solution = solver.solve(qs, bs)

        # Auto-tune fires: benchmarked methods for this device
        assert solver._auto_tuned
        assert solver._tune_result is not None
        # Should have benchmarked a subset of methods for this device

        expected_keys = {f"{device}:{m}" for m in solver_methods_for_device(device)}
        assert set(solver._tune_result.results.keys()) == expected_keys

        # Method should be a concrete value (not 'auto')
        assert solver._settings.ipm_settings.direct_solve_method != "auto"
        assert solver._settings.ipm_settings.direct_solve_method in solver_methods_for_device(
            device
        )
        assert solution.x.shape == (4, 2)

    def test_explicit_device_and_method_skips_tune(self, device):
        """Explicit device + explicit method skips auto-tune even with auto_tune=True."""
        method = "qdldl" if device == "cpu" else "cudss"
        solver = _make_compiled_solver(device, method=method, auto_tune=True)
        assert not solver._auto_tuned

        qs = [[2.0, 1.0]] * 4
        bs = [[1.0, 0.7, 0.7]] * 4
        solution = solver.solve(qs, bs)

        assert not solver._auto_tuned
        assert solver._tune_result is None
        assert solution.x.shape == (4, 2)

    def test_auto_tune_false_does_not_benchmark(self):
        """auto_tune=False with device='auto' does NOT benchmark."""
        solver = _make_compiled_solver("auto", method="auto", auto_tune=False)

        qs = [[2.0, 1.0]] * 4
        bs = [[1.0, 0.7, 0.7]] * 4

        with patch.object(solver, "_auto_tune", wraps=solver._auto_tune) as mock_tune:
            solver.solve(qs, bs)
            mock_tune.assert_not_called()

        assert not solver._auto_tuned
        assert solver._tune_result is None


class TestAutoTuneDoesNotRepeat:
    """Test that auto-tune only fires once."""

    def test_second_solve_skips_tune(self):
        """Second solve() does not re-benchmark."""
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)

        qs = [[2.0, 1.0]] * 4
        bs = [[1.0, 0.7, 0.7]] * 4

        # First solve triggers auto-tune
        solver.solve(qs, bs)
        assert solver._auto_tuned
        first_result = solver._tune_result

        # Second solve should not re-tune
        with patch.object(solver, "_auto_tune", wraps=solver._auto_tune) as mock_tune:
            solver.solve(qs, bs)
            mock_tune.assert_not_called()

        # _tune_result unchanged
        assert solver._tune_result is first_result


class TestAutoTuneResults:
    """Test the quality of auto-tune results."""

    def test_tune_result_has_valid_device(self):
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)

        result = solver._tune_result
        assert result.device == solver.device

    def test_tune_result_has_valid_method(self):
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)

        result = solver._tune_result
        candidates = solver_methods_for_device(solver.device)
        assert result.method in candidates

    def test_tune_result_has_positive_time_limit(self):
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)

        result = solver._tune_result
        assert result.time_limit > 0

    def test_tune_result_contains_all_candidates(self):
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)

        result = solver._tune_result
        # All devices × their methods
        devices = available_devices()
        expected_keys = set()
        for dev in devices:
            for m in solver_methods_for_device(dev):
                expected_keys.add(_result_key(dev, m))
        # Methods incompatible with the problem structure may be skipped
        assert set(result.results.keys()) == expected_keys

    def test_tune_result_per_method_data(self):
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)

        result = solver._tune_result
        for method, data in result.results.items():
            assert "solve_time" in data
            assert "iterations" in data
            assert "status" in data
            assert isinstance(data["solve_time"], float)
            assert data["solve_time"] >= 0

    def test_tune_result_repr(self):
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)

        result = solver._tune_result
        r = repr(result)
        assert "TuneResult" in r
        assert result.method in r
        assert result.device in r


class TestAutoTuneTimeLimits:
    """Test time_limit = max(best_time * margin, original_time_limit)."""

    def test_default_time_limit_inf_preserved(self):
        """Default time_limit is inf, so auto-tune should keep inf."""
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)
        assert solver._tune_result.time_limit == float("inf")

    def test_user_time_limit_never_tightened(self):
        """Auto-tune should never reduce the user's explicit time_limit."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(
            device="auto",
            batch_size=4,
            time_limit=10.0,
            auto_tune=True,
            solver="ipm",
        )

        solver = moreau.CompiledSolver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 2, 3, 4],
            A_col_indices=[0, 1, 0, 1],
            cones=cones,
            settings=settings,
        )
        solver.setup(P_values=[1.0, 1.0], A_values=[1.0, 1.0, 1.0, 1.0])
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)

        assert solver._tune_result.time_limit == 10.0


class TestAutoTuneSolveCorrectness:
    """Test that solve results are correct after auto-tune."""

    def test_solve_after_auto_tune_is_correct(self):
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        qs = [[2.0, 1.0]] * 4
        bs = [[1.0, 0.7, 0.7]] * 4

        solution = solver.solve(qs, bs)
        assert solution.x.shape == (4, 2)
        for s in solver.info.status:
            assert s in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)

    def test_auto_tune_matches_explicit(self, device):
        """Auto-tuned solver produces same results as explicit method."""
        qs = [[2.0, 1.0]] * 4
        bs = [[1.0, 0.7, 0.7]] * 4

        # Explicit method (no auto-tune)
        method = "qdldl" if device == "cpu" else "cudss"
        explicit_solver = _make_compiled_solver(device, method=method)
        sol_explicit = explicit_solver.solve(qs, bs)

        # Auto-tuned (device='auto', resolves to same device via set_default_device)
        auto_solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        sol_auto = auto_solver.solve(qs, bs)

        np.testing.assert_allclose(sol_auto.x, sol_explicit.x, atol=1e-6)
        np.testing.assert_allclose(sol_auto.z, sol_explicit.z, atol=1e-6)
        np.testing.assert_allclose(sol_auto.s, sol_explicit.s, atol=1e-6)

    def test_second_solve_correct(self):
        """Second solve after auto-tune still produces correct results."""
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        qs = [[2.0, 1.0]] * 4
        bs = [[1.0, 0.7, 0.7]] * 4

        sol1 = solver.solve(qs, bs)
        sol2 = solver.solve(qs, bs)

        np.testing.assert_allclose(sol1.x, sol2.x, atol=1e-8)

    def test_heuristic_solve_matches_explicit(self, device):
        """Heuristic (auto_tune=False) produces same results as explicit method."""
        qs = [[2.0, 1.0]] * 4
        bs = [[1.0, 0.7, 0.7]] * 4

        explicit_solver = _make_compiled_solver(device, method="qdldl")
        sol_explicit = explicit_solver.solve(qs, bs)

        heuristic_solver = _make_compiled_solver("auto", method="auto", auto_tune=False)
        sol_heuristic = heuristic_solver.solve(qs, bs)

        np.testing.assert_allclose(sol_heuristic.x, sol_explicit.x, atol=1e-6)
        np.testing.assert_allclose(sol_heuristic.z, sol_explicit.z, atol=1e-6)
        np.testing.assert_allclose(sol_heuristic.s, sol_explicit.s, atol=1e-6)


class TestAutoTuneBatchSizes:
    """Test auto-tune with different batch sizes."""

    def test_batch_size_1(self):
        solver = _make_compiled_solver("auto", batch_size=1, method="auto", auto_tune=True)
        solution = solver.solve([[2.0, 1.0]], [[1.0, 0.7, 0.7]])
        assert solver._auto_tuned
        assert solution.x.shape == (1, 2)

    def test_batch_size_16(self):
        solver = _make_compiled_solver("auto", batch_size=16, method="auto", auto_tune=True)
        qs = [[2.0, 1.0]] * 16
        bs = [[1.0, 0.7, 0.7]] * 16
        solution = solver.solve(qs, bs)
        assert solver._auto_tuned
        assert solution.x.shape == (16, 2)


class TestAutoTunePerBatchValues:
    """Test auto-tune when P/A values vary per batch element."""

    def test_per_batch_P_values(self):
        batch_size = 4
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(
            device="auto",
            batch_size=batch_size,
            auto_tune=True,
            solver="ipm",
        )

        solver = moreau.CompiledSolver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 2, 3, 4],
            A_col_indices=[0, 1, 0, 1],
            cones=cones,
            settings=settings,
        )

        P_values = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        A_values = np.array([[1.0, 1.0, 1.0, 1.0]] * batch_size)
        solver.setup(P_values=P_values, A_values=A_values)

        qs = [[2.0, 1.0]] * batch_size
        bs = [[1.0, 0.7, 0.7]] * batch_size

        solution = solver.solve(qs, bs)
        assert solver._auto_tuned
        assert solution.x.shape == (batch_size, 2)
        for s in solver.info.status:
            assert s in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)


# ── Cross-device auto-tune tests ────────────────────────────────────


class TestCrossDeviceAutoTune:
    """Test auto-tune with device='auto' benchmarks across all devices."""

    def test_auto_device_benchmarks_all_devices(self):
        """device='auto' + auto_tune=True benchmarks all available devices."""
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)

        result = solver._tune_result
        devices = available_devices()
        expected_keys = set()
        for dev in devices:
            for m in solver_methods_for_device(dev):
                expected_keys.add(f"{dev}:{m}")
        # Methods incompatible with the problem structure may be skipped
        assert set(result.results.keys()) == expected_keys

    def test_auto_device_result_has_device_field(self):
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)

        result = solver._tune_result
        assert result.device != ""
        assert result.device in available_devices()

    def test_auto_device_solver_device_updated(self):
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)
        assert solver.device == solver._tune_result.device

    def test_auto_device_settings_updated(self):
        solver = _make_compiled_solver("auto", method="auto", auto_tune=True)
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)

        result = solver._tune_result
        assert solver._settings.ipm_settings.direct_solve_method == result.method

    def test_explicit_device_auto_method_benchmarks_on_first_solve(self, device):
        """Explicit device + method='auto' + auto_tune=True benchmarks methods."""
        solver = _make_compiled_solver(device, method="auto", auto_tune=True)
        solver.solve([[2.0, 1.0]] * 4, [[1.0, 0.7, 0.7]] * 4)

        # Auto-tune fires: benchmarked methods for this device

        assert solver._auto_tuned
        assert solver._tune_result is not None
        expected_keys = {f"{device}:{m}" for m in solver_methods_for_device(device)}
        assert set(solver._tune_result.results.keys()) == expected_keys

        assert solver._settings.ipm_settings.direct_solve_method != "auto"
        assert solver._settings.ipm_settings.direct_solve_method in solver_methods_for_device(
            device
        )


# ── End-to-end workflow tests ────────────────────────────────────────


class TestAutoTuneEndToEnd:
    """Full auto-tune workflow examples as tests."""

    def test_compiled_solver_workflow(self):
        """Standard workflow: construct → setup → solve (auto-tunes) → solve again."""
        batch_size = 64
        solver = _make_compiled_solver("auto", batch_size=batch_size, method="auto", auto_tune=True)

        qs = [[2.0, 1.0]] * batch_size
        bs = [[1.0, 0.7, 0.7]] * batch_size

        # First solve triggers auto-tune
        solution = solver.solve(qs, bs)
        assert solver._auto_tuned
        assert solution.x.shape == (batch_size, 2)
        for s in solver.info.status:
            assert s in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)

        # Second solve uses tuned config directly
        solution2 = solver.solve(qs, bs)
        np.testing.assert_allclose(solution.x[0], solution2.x[0], atol=1e-8)

    def test_auto_device_workflow(self):
        """End-to-end with device='auto' + auto_tune=True."""
        batch_size = 4
        solver = _make_compiled_solver("auto", batch_size=batch_size, method="auto", auto_tune=True)

        qs = [[2.0, 1.0]] * batch_size
        bs = [[1.0, 0.7, 0.7]] * batch_size

        solution = solver.solve(qs, bs)
        assert solver._auto_tuned
        assert solver.device in available_devices()
        assert solution.x.shape == (batch_size, 2)
        for s in solver.info.status:
            assert s in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)

    def test_heuristic_workflow(self):
        """End-to-end with device='auto' + auto_tune=False (default)."""
        batch_size = 4
        solver = _make_compiled_solver(
            "auto", batch_size=batch_size, method="auto", auto_tune=False
        )

        qs = [[2.0, 1.0]] * batch_size
        bs = [[1.0, 0.7, 0.7]] * batch_size

        solution = solver.solve(qs, bs)
        assert not solver._auto_tuned
        assert solver._tune_result is None
        assert solver.device in available_devices()
        assert solution.x.shape == (batch_size, 2)
        for s in solver.info.status:
            assert s in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)


# ── Torch auto-tune tests ───────────────────────────────────────────


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestTorchAutoTune:
    """Test moreau.torch.Solver auto-tune on first solve()."""

    def _make_solver(self, device, method="auto", auto_tune=False):
        from moreau.torch import Solver

        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        ipm = moreau.IPMSettings(direct_solve_method=method)
        settings = moreau.Settings(
            device=device,
            batch_size=4,
            ipm_settings=ipm,
            auto_tune=auto_tune,
            solver="ipm",
        )

        P_ro = torch.tensor([0, 1, 2], dtype=torch.int64)
        P_ci = torch.tensor([0, 1], dtype=torch.int64)
        A_ro = torch.tensor([0, 2, 3, 4], dtype=torch.int64)
        A_ci = torch.tensor([0, 1, 0, 1], dtype=torch.int64)

        solver = Solver(
            n=2,
            m=3,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=settings,
        )

        # For tensor creation, resolve 'auto' to the solver's resolved device
        torch_device = solver.device if solver.device in ("cpu", "cuda") else "cpu"
        P_vals = torch.tensor([[1.0, 1.0]] * 4, dtype=torch.float64, device=torch_device)
        A_vals = torch.tensor([[1.0, 1.0, 1.0, 1.0]] * 4, dtype=torch.float64, device=torch_device)

        q = torch.tensor([[2.0, 1.0]] * 4, dtype=torch.float64, device=torch_device)
        b = torch.tensor([[1.0, 0.7, 0.7]] * 4, dtype=torch.float64, device=torch_device)

        return solver, P_vals, A_vals, q, b

    def test_auto_tune_on_first_solve(self):
        """device='auto' + auto_tune=True triggers auto-tune on first torch solve."""
        solver, P, A, q, b = self._make_solver("auto", method="auto", auto_tune=True)
        assert not solver._auto_tuned
        sol = solver.solve(P, A, q, b)
        assert solver._auto_tuned
        assert sol.x.shape == (4, 2)

    def test_auto_tune_disabled_by_default(self):
        """device='auto' with auto_tune=False does NOT benchmark."""
        solver, P, A, q, b = self._make_solver("auto", method="auto", auto_tune=False)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            sol = solver.solve(P, A, q, b)

        assert not solver._auto_tuned
        assert solver._tune_result is None
        assert sol.x.shape == (4, 2)

    def test_explicit_device_auto_method_benchmarks_on_first_solve(self):
        """Explicit device + method='auto' + auto_tune=True benchmarks methods."""
        solver, P, A, q, b = self._make_solver("cpu", method="auto", auto_tune=True)
        sol = solver.solve(P, A, q, b)
        # Auto-tune fires: benchmarked methods for cpu
        assert solver._auto_tuned
        assert solver._tune_result is not None
        expected_keys = {f"cpu:{m}" for m in solver_methods_for_device("cpu")}
        assert set(solver._tune_result.results.keys()) == expected_keys

        assert solver._settings.ipm_settings.direct_solve_method != "auto"
        assert sol.x.shape == (4, 2)

    def test_second_solve_no_retune(self):
        """Second solve does not re-benchmark."""
        solver, P, A, q, b = self._make_solver("auto", method="auto", auto_tune=True)
        solver.solve(P, A, q, b)
        assert solver._auto_tuned

        with patch.object(solver, "_auto_tune", wraps=solver._auto_tune) as mock:
            solver.solve(P, A, q, b)
            mock.assert_not_called()

    @pytest.mark.skipif(
        not (HAS_TORCH and torch.cuda.is_available()),
        reason="CUDA not available",
    )
    def test_backward_after_auto_tune(self):
        """Gradients still work after auto-tune reconstructs _impl."""
        solver, P, A, q, b = self._make_solver("auto", method="auto", auto_tune=True)
        solver.solve(P, A, q, b)  # triggers auto-tune

        q_grad = q.clone().requires_grad_(True)
        sol = solver.solve(P, A, q_grad, b)
        sol.x.sum().backward()
        assert q_grad.grad is not None
        assert q_grad.grad.shape == q.shape


# ── JAX auto-tune tests ─────────────────────────────────────────────


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestJaxAutoTune:
    """Test moreau.jax.Solver auto-tune on first solve()."""

    def _make_solver(self, device, method="auto", auto_tune=False):
        from moreau.jax import Solver

        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        ipm = moreau.IPMSettings(direct_solve_method=method)
        settings = moreau.Settings(
            device=device, ipm_settings=ipm, auto_tune=auto_tune, solver="ipm"
        )

        solver = Solver(
            n=2,
            m=3,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 2, 3, 4],
            A_col_indices=[0, 1, 0, 1],
            cones=cones,
            settings=settings,
        )

        P_data = jnp.array([1.0, 1.0], dtype=jnp.float64)
        A_data = jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float64)
        solver.setup(P_data, A_data)

        q = jnp.array([2.0, 1.0], dtype=jnp.float64)
        b = jnp.array([1.0, 0.7, 0.7], dtype=jnp.float64)

        return solver, q, b

    def test_auto_tune_on_first_solve(self):
        """device='auto' + auto_tune=True triggers auto-tune on first jax solve."""
        solver, q, b = self._make_solver("auto", method="auto", auto_tune=True)
        assert not solver._auto_tuned
        sol = solver.solve(q, b)
        assert solver._auto_tuned
        assert sol.x.shape == (2,)

    def test_auto_tune_disabled_by_default(self):
        """device='auto' with auto_tune=False does NOT benchmark."""
        solver, q, b = self._make_solver("auto", method="auto", auto_tune=False)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            sol = solver.solve(q, b)

        assert not solver._auto_tuned
        assert solver._tune_result is None
        assert sol.x.shape == (2,)

    def test_explicit_device_auto_method_benchmarks_on_first_solve(self):
        """Explicit device + method='auto' + auto_tune=True benchmarks methods."""
        solver, q, b = self._make_solver("cpu", method="auto", auto_tune=True)
        sol = solver.solve(q, b)
        # Auto-tune fires: benchmarked methods for cpu
        assert solver._auto_tuned
        assert solver._tune_result is not None
        expected_keys = {f"cpu:{m}" for m in solver_methods_for_device("cpu")}
        assert set(solver._tune_result.results.keys()) == expected_keys

        assert solver._settings.ipm_settings.direct_solve_method != "auto"
        assert sol.x.shape == (2,)

    def test_second_solve_no_retune(self):
        """Second solve does not re-benchmark."""
        solver, q, b = self._make_solver("auto", method="auto", auto_tune=True)
        solver.solve(q, b)
        assert solver._auto_tuned

        with patch.object(solver, "_auto_tune", wraps=solver._auto_tune) as mock:
            solver.solve(q, b)
            mock.assert_not_called()

    @pytest.mark.skipif(
        not (HAS_JAX and moreau.device_available("cuda")),
        reason="CUDA not available",
    )
    def test_grad_after_auto_tune(self):
        """Gradients still work after auto-tune reconstructs _impl."""
        solver, q, b = self._make_solver("auto", method="auto", auto_tune=True)
        solver.solve(q, b)  # triggers auto-tune

        grad_fn = jax.grad(lambda q_: solver.solve(q_, b).x.sum())
        g = grad_fn(q)
        assert g.shape == q.shape
