"""Tests for warm starting.

These tests verify:
1. Warm start from a previous solution converges in fewer iterations
2. Warm start with perturbed problem data works
3. Warm start with SOC cones
4. Shape validation and error handling
5. CPU/GPU parity
6. Bad warm start still converges
7. to_warm_start() and WarmStart/BatchedWarmStart types
8. Auto-retry cold on warm start failure
"""

import moreau
import pytest
import warnings
import numpy as np
from unittest.mock import patch


@pytest.fixture
def simple_qp():
    """Simple QP with equality + inequality constraints.

    minimize x'Px + q'x
    subject to x + y = 1, x >= 0, y >= 0, x <= 1, y <= 1
    """
    n = 2
    m = 5

    P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    P_col_indices = np.array([0, 1], dtype=np.int64)
    P_values = np.array([2.0, 2.0])

    A_row_offsets = np.array([0, 2, 3, 4, 5, 6], dtype=np.int64)
    A_col_indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    A_values = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])

    cones = moreau.Cones()
    cones.num_zero_cones = 1
    cones.num_nonneg_cones = 4

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "cones": cones,
    }


@pytest.fixture
def soc_problem():
    """SOC problem: minimize x1 subject to ||(x2, x3)|| <= x1."""
    n = 3
    m = 3

    # P = 0 (LP-like objective)
    P_row_offsets = np.array([0, 0, 0, 0], dtype=np.int64)
    P_col_indices = np.array([], dtype=np.int64)
    P_values = np.array([])

    # A = -I (so constraint is -x + s = 0, i.e., s = x)
    A_row_offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    A_col_indices = np.array([0, 1, 2], dtype=np.int64)
    A_values = np.array([-1.0, -1.0, -1.0])

    cones = moreau.Cones()
    cones.so_cone_dims = [3]

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "cones": cones,
    }


def _make_solver(prob, device, batch_size=1, verbose=False):
    """Create a CompiledSolver for the given problem."""
    method = "qdldl" if device == "cpu" else "cudss"
    ipm = moreau.IPMSettings(direct_solve_method=method)
    settings = moreau.Settings(
        solver="ipm", device=device, batch_size=batch_size, verbose=verbose, ipm_settings=ipm
    )
    solver = moreau.CompiledSolver(
        n=prob["n"],
        m=prob["m"],
        P_row_offsets=prob["P_row_offsets"],
        P_col_indices=prob["P_col_indices"],
        A_row_offsets=prob["A_row_offsets"],
        A_col_indices=prob["A_col_indices"],
        cones=prob["cones"],
        settings=settings,
    )
    solver.setup(P_values=prob["P_values"], A_values=prob["A_values"])
    return solver


@pytest.mark.skipif(not moreau.device_available("cpu"), reason="CPU not available")
def test_nan_warm_start_detected_as_numerical_error_cpu(simple_qp):
    """On CPU, NaN in warm start should be detected as NumericalError and auto-retried."""
    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    # Create a warm start with NaN in x
    ws = moreau.BatchedWarmStart(
        x=np.full((1, prob["n"]), np.nan),
        z=np.ones((1, prob["m"])),
        s=np.ones((1, prob["m"])),
    )

    solver = _make_solver(prob, "cpu")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = solver.solve(qs=q, bs=b, warm_start=ws)

    # CPU detects NaN → NumericalError → auto-retry cold → valid solution
    assert not np.any(np.isnan(sol.x)), "Solution should not contain NaN after retry"
    assert len(w) == 1
    assert "NumericalError" in str(w[0].message)
    assert "without warm start" in str(w[0].message)


def test_warm_start_fewer_iterations(simple_qp, device):
    """Warm starting from a cold-solved solution should converge in fewer iterations."""
    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    # Cold solve
    solver_cold = _make_solver(prob, device)
    sol_cold = solver_cold.solve(qs=q, bs=b)
    info_cold = solver_cold.info
    assert info_cold.status[0] == moreau.SolverStatus.Solved
    iters_cold = (
        info_cold.iterations[0] if isinstance(info_cold.iterations, list) else info_cold.iterations
    )

    # Warm solve from cold solution
    ws = sol_cold.to_warm_start()
    solver_warm = _make_solver(prob, device)
    sol_warm = solver_warm.solve(qs=q, bs=b, warm_start=ws)
    info_warm = solver_warm.info
    assert info_warm.status[0] == moreau.SolverStatus.Solved
    iters_warm = (
        info_warm.iterations[0] if isinstance(info_warm.iterations, list) else info_warm.iterations
    )

    # Warm start should converge in fewer iterations
    assert (
        iters_warm < iters_cold
    ), f"Warm start ({iters_warm} iters) should be faster than cold ({iters_cold} iters)"

    # Solutions should match (warm start converges to same optimum but
    # from a different central path point, so tolerance is relaxed)
    np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4)


def test_warm_start_perturbed_problem(simple_qp, device):
    """Warm start from P1's solution should help P2 converge (slightly perturbed q)."""
    prob = simple_qp
    q1 = np.array([[1.0, -1.0]])
    q2 = np.array([[1.1, -0.9]])  # small perturbation
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    # Solve P1 cold
    solver1 = _make_solver(prob, device)
    sol1 = solver1.solve(qs=q1, bs=b)
    assert solver1.info.status[0] == moreau.SolverStatus.Solved

    # Solve P2 cold
    solver2_cold = _make_solver(prob, device)
    sol2_cold = solver2_cold.solve(qs=q2, bs=b)
    assert solver2_cold.info.status[0] == moreau.SolverStatus.Solved

    # Solve P2 warm from P1's solution
    ws = sol1.to_warm_start()
    solver2_warm = _make_solver(prob, device)
    sol2_warm = solver2_warm.solve(qs=q2, bs=b, warm_start=ws)
    assert solver2_warm.info.status[0] == moreau.SolverStatus.Solved

    # Solutions should match (both P2, relaxed tolerance due to different central path)
    np.testing.assert_allclose(sol2_warm.x, sol2_cold.x, atol=1e-4)


def test_warm_start_soc(soc_problem, device):
    """Test warm start with second-order cone."""
    prob = soc_problem
    q = np.array([[1.0, 0.5, 0.5]])
    b = np.array([[0.0, 0.0, 0.0]])

    # Cold solve
    solver_cold = _make_solver(prob, device)
    sol_cold = solver_cold.solve(qs=q, bs=b)
    assert solver_cold.info.status[0] == moreau.SolverStatus.Solved

    # Warm solve
    ws = sol_cold.to_warm_start()
    solver_warm = _make_solver(prob, device)
    sol_warm = solver_warm.solve(qs=q, bs=b, warm_start=ws)
    assert solver_warm.info.status[0] == moreau.SolverStatus.Solved

    np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-6)


def test_warm_start_bad_point_still_converges(simple_qp, device):
    """A bad warm start (random point) should still converge."""
    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    # Cold solve for reference
    solver_ref = _make_solver(prob, device)
    sol_ref = solver_ref.solve(qs=q, bs=b)
    assert solver_ref.info.status[0] == moreau.SolverStatus.Solved

    # Warm start with a somewhat reasonable but not optimal point
    ws = moreau.WarmStart(
        x=np.array([[0.5, 0.5]]),
        z=np.ones((1, prob["m"])),
        s=np.ones((1, prob["m"])),
    )

    solver_warm = _make_solver(prob, device)
    sol_warm = solver_warm.solve(qs=q, bs=b, warm_start=ws)
    assert solver_warm.info.status[0] in [
        moreau.SolverStatus.Solved,
        moreau.SolverStatus.AlmostSolved,
    ]

    np.testing.assert_allclose(sol_warm.x, sol_ref.x, atol=1e-5)


def test_warm_start_validation_wrong_type(simple_qp, device):
    """Passing a non-WarmStart object should raise TypeError."""
    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    solver = _make_solver(prob, device)

    with pytest.raises(TypeError, match="warm_start must be a WarmStart"):
        solver.solve(qs=q, bs=b, warm_start="not a warm start")


def test_warm_start_validation_wrong_dims(simple_qp, device):
    """Providing wrong-dimension warm start should raise an error."""
    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    solver = _make_solver(prob, device)

    # Wrong x dim
    ws = moreau.WarmStart(
        x=np.zeros((1, prob["n"] + 1)),
        z=np.ones((1, prob["m"])),
        s=np.ones((1, prob["m"])),
    )
    with pytest.raises((ValueError, RuntimeError)):
        solver.solve(qs=q, bs=b, warm_start=ws)


def test_warm_start_batched(simple_qp, device):
    """Test warm start with batched problems (different warm start per problem)."""
    prob = simple_qp
    batch_size = 2

    q_batch = np.array([[1.0, -1.0], [0.5, -0.5]])
    b_batch = np.array([[1.0, 0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0, 1.0]])

    # Cold solve
    solver_cold = _make_solver(prob, device, batch_size=batch_size)
    sol_cold = solver_cold.solve(qs=q_batch, bs=b_batch)
    info_cold = solver_cold.info
    assert all(s == moreau.SolverStatus.Solved for s in info_cold.status)

    # Warm solve from cold solution using to_warm_start()
    ws = sol_cold.to_warm_start()
    assert isinstance(ws, moreau.BatchedWarmStart)
    assert len(ws) == batch_size

    solver_warm = _make_solver(prob, device, batch_size=batch_size)
    sol_warm = solver_warm.solve(qs=q_batch, bs=b_batch, warm_start=ws)
    info_warm = solver_warm.info
    assert all(s == moreau.SolverStatus.Solved for s in info_warm.status)

    # Solutions should match (relaxed tolerance due to different central path)
    np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4)


def test_warm_start_cpu_gpu_parity(simple_qp):
    """Warm start solutions should match between CPU and GPU."""
    if not moreau.device_available("cuda"):
        pytest.skip("CUDA not available")

    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    # Cold solve on CPU for warm start values
    solver_cold = _make_solver(prob, "cpu")
    sol_cold = solver_cold.solve(qs=q, bs=b)
    assert solver_cold.info.status[0] == moreau.SolverStatus.Solved

    ws = sol_cold.to_warm_start()

    # Warm solve on CPU
    solver_cpu = _make_solver(prob, "cpu")
    sol_cpu = solver_cpu.solve(qs=q, bs=b, warm_start=ws)

    # Warm solve on CUDA
    solver_cuda = _make_solver(prob, "cuda")
    sol_cuda = solver_cuda.solve(qs=q, bs=b, warm_start=ws)

    assert solver_cpu.info.status[0] == moreau.SolverStatus.Solved
    assert solver_cuda.info.status[0] == moreau.SolverStatus.Solved

    np.testing.assert_allclose(sol_cpu.x, sol_cuda.x, atol=1e-4)
    np.testing.assert_allclose(sol_cpu.z, sol_cuda.z, atol=1e-4)
    np.testing.assert_allclose(sol_cpu.s, sol_cuda.s, atol=1e-4)


def test_to_warm_start_types(simple_qp, device):
    """Verify to_warm_start() returns the correct types."""
    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    # Single problem -> BatchedSolution.to_warm_start() -> BatchedWarmStart
    solver = _make_solver(prob, device)
    sol = solver.solve(qs=q, bs=b)
    assert isinstance(sol, moreau.BatchedSolution)

    ws = sol.to_warm_start()
    assert isinstance(ws, moreau.BatchedWarmStart)
    assert len(ws) == 1

    # Index into BatchedWarmStart -> WarmStart
    ws0 = ws[0]
    assert isinstance(ws0, moreau.WarmStart)
    assert ws0.x.shape == (prob["n"],)
    assert ws0.z.shape == (prob["m"],)
    assert ws0.s.shape == (prob["m"],)

    # Iteration
    for w in ws:
        assert isinstance(w, moreau.WarmStart)


def test_batched_warm_start_indexing(simple_qp, device):
    """Test BatchedWarmStart indexing and iteration."""
    prob = simple_qp
    batch_size = 3

    q_batch = np.array([[1.0, -1.0], [0.5, -0.5], [0.0, 0.0]])
    b_batch = np.tile([1.0, 0.0, 0.0, 1.0, 1.0], (batch_size, 1))

    solver = _make_solver(prob, device, batch_size=batch_size)
    sol = solver.solve(qs=q_batch, bs=b_batch)

    ws = sol.to_warm_start()
    assert len(ws) == batch_size

    # Positive indexing
    for i in range(batch_size):
        w = ws[i]
        np.testing.assert_array_equal(w.x, sol.x[i])
        np.testing.assert_array_equal(w.z, sol.z[i])
        np.testing.assert_array_equal(w.s, sol.s[i])

    # Negative indexing
    np.testing.assert_array_equal(ws[-1].x, sol.x[batch_size - 1])

    # Out of range
    with pytest.raises(IndexError):
        ws[batch_size]


def test_warm_start_retry_on_failure(simple_qp, device):
    """If warm-started solve fails, it should auto-retry cold and emit a warning."""
    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    solver = _make_solver(prob, device)

    # Cold solve to get a warm start and the expected result
    sol_cold = solver.solve(qs=q, bs=b)
    assert solver.info.status[0] == moreau.SolverStatus.Solved
    ws = sol_cold.to_warm_start()

    # Make a new solver and monkey-patch _impl.solve to fail on warm start
    solver2 = _make_solver(prob, device)
    original_solve = solver2._impl.solve
    call_count = [0]

    def mock_solve(qs_arg, bs_arg, **kwargs):
        call_count[0] += 1
        if kwargs.get("warm_x") is not None:
            # Simulate failure: return NumericalError status
            result = original_solve(qs_arg, bs_arg, **kwargs)
            result["status"] = np.array([int(moreau.SolverStatus.NumericalError)])
            return result
        # Cold call: return normal result
        return original_solve(qs_arg, bs_arg, **kwargs)

    solver2._impl.solve = mock_solve

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = solver2.solve(qs=q, bs=b, warm_start=ws)

    # Should have called solve twice (warm fail + cold retry)
    assert call_count[0] == 2
    # Warning should have been emitted
    assert len(w) == 1
    assert "without warm start" in str(w[0].message)
    assert "NumericalError" in str(w[0].message)
    # Cold retry should succeed
    assert solver2.info.status[0] == moreau.SolverStatus.Solved


def test_warm_start_partial_failure_preserves_successful_warm_results(simple_qp, device):
    """When some batch elements fail warm-start and others succeed, the cold
    re-solve must only patch the failed indices. Successful warm-started
    iterates must be preserved so downstream backward passes use the
    iterate the user expected. (#178)"""
    prob = simple_qp
    batch_size = 4
    q = np.array([[1.0, -1.0]] * batch_size)
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]] * batch_size)

    solver = _make_solver(prob, device, batch_size=batch_size)
    sol_cold = solver.solve(qs=q, bs=b)
    ws = sol_cold.to_warm_start()

    solver2 = _make_solver(prob, device, batch_size=batch_size)
    original_solve = solver2._impl.solve

    failed_index = 2
    warm_result_x_at_2 = [None]

    def mock_solve(qs_arg, bs_arg, **kwargs):
        result = original_solve(qs_arg, bs_arg, **kwargs)
        if kwargs.get("warm_x") is not None:
            # Simulate failure on batch element 2 only
            statuses = np.array([int(moreau.SolverStatus.Solved)] * batch_size)
            statuses[failed_index] = int(moreau.SolverStatus.NumericalError)
            result["status"] = statuses
            # Tag the warm result x at the failing index with a sentinel so
            # we can verify the cold retry replaced it (and didn't replace
            # the others).
            result["x"][failed_index] = np.full_like(result["x"][failed_index], 99.0)
            warm_result_x_at_2[0] = result["x"][failed_index].copy()
        else:
            # Cold call: tag every row distinctively so we can detect overwrite
            result["x"] = result["x"] + 1000.0  # cold sentinel
        return result

    solver2._impl.solve = mock_solve

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = solver2.solve(qs=q, bs=b, warm_start=ws)

    # Successful (warm) elements: not overwritten by cold sentinel.
    for i in range(batch_size):
        if i == failed_index:
            continue
        assert (
            sol.x[i].max() < 1000.0
        ), f"Warm-started element {i} was incorrectly overwritten by cold retry"
    # Failed element: replaced by cold sentinel.
    assert (
        sol.x[failed_index].max() >= 1000.0
    ), f"Failed element {failed_index} should have been replaced by cold retry"
    # Status reflects merged outcome.
    statuses = solver2.info.status
    for i in range(batch_size):
        if i == failed_index:
            assert statuses[i] == moreau.SolverStatus.Solved  # cold retry succeeded
        else:
            assert statuses[i] == moreau.SolverStatus.Solved  # warm succeeded
    # Warning mentions the specific failed index.
    assert any("[2]" in str(msg.message) for msg in w)


def test_warm_start_no_retry_on_success(simple_qp, device):
    """Successful warm-started solve should not trigger a retry."""
    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    solver = _make_solver(prob, device)
    sol_cold = solver.solve(qs=q, bs=b)
    ws = sol_cold.to_warm_start()

    solver2 = _make_solver(prob, device)
    original_solve = solver2._impl.solve
    call_count = [0]

    def counting_solve(qs_arg, bs_arg, **kwargs):
        call_count[0] += 1
        return original_solve(qs_arg, bs_arg, **kwargs)

    solver2._impl.solve = counting_solve

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = solver2.solve(qs=q, bs=b, warm_start=ws)

    # Should have called solve only once (success, no retry)
    assert call_count[0] == 1
    assert len(w) == 0
    assert solver2.info.status[0] == moreau.SolverStatus.Solved


def test_warm_start_no_retry_without_warm_start(simple_qp, device):
    """Cold solve that fails should NOT retry (retry is only for warm start)."""
    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    solver = _make_solver(prob, device)
    original_solve = solver._impl.solve
    call_count = [0]

    def mock_solve(qs_arg, bs_arg, **kwargs):
        call_count[0] += 1
        result = original_solve(qs_arg, bs_arg, **kwargs)
        result["status"] = np.array([int(moreau.SolverStatus.MaxIterations)])
        return result

    solver._impl.solve = mock_solve

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = solver.solve(qs=q, bs=b)  # no warm_start

    # Should have called solve only once (no retry on cold failure)
    assert call_count[0] == 1
    assert len(w) == 0
    assert solver.info.status[0] == moreau.SolverStatus.MaxIterations


def test_warm_start_retry_on_infeasible(simple_qp, device):
    """PrimalInfeasible from warm start should also trigger retry."""
    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    solver = _make_solver(prob, device)
    sol_cold = solver.solve(qs=q, bs=b)
    ws = sol_cold.to_warm_start()

    solver2 = _make_solver(prob, device)
    original_solve = solver2._impl.solve
    call_count = [0]

    def mock_solve(qs_arg, bs_arg, **kwargs):
        call_count[0] += 1
        if kwargs.get("warm_x") is not None:
            result = original_solve(qs_arg, bs_arg, **kwargs)
            result["status"] = np.array([int(moreau.SolverStatus.PrimalInfeasible)])
            return result
        return original_solve(qs_arg, bs_arg, **kwargs)

    solver2._impl.solve = mock_solve

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = solver2.solve(qs=q, bs=b, warm_start=ws)

    assert call_count[0] == 2
    assert len(w) == 1
    assert "PrimalInfeasible" in str(w[0].message)
    assert solver2.info.status[0] == moreau.SolverStatus.Solved


def test_warm_start_no_retry_on_max_iterations(simple_qp, device):
    """MaxIterations should NOT trigger a retry (it's in the no-retry set)."""
    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    solver = _make_solver(prob, device)
    sol_cold = solver.solve(qs=q, bs=b)
    ws = sol_cold.to_warm_start()

    solver2 = _make_solver(prob, device)
    original_solve = solver2._impl.solve
    call_count = [0]

    def mock_solve(qs_arg, bs_arg, **kwargs):
        call_count[0] += 1
        result = original_solve(qs_arg, bs_arg, **kwargs)
        result["status"] = np.array([int(moreau.SolverStatus.MaxIterations)])
        return result

    solver2._impl.solve = mock_solve

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = solver2.solve(qs=q, bs=b, warm_start=ws)

    # MaxIterations is in no-retry set, so no retry
    assert call_count[0] == 1
    assert len(w) == 0
    assert solver2.info.status[0] == moreau.SolverStatus.MaxIterations


def test_warm_start_custom_no_retry(simple_qp, device):
    """Custom warm_start_no_retry in IPMSettings should override the default set."""
    prob = simple_qp
    q = np.array([[1.0, -1.0]])
    b = np.array([[1.0, 0.0, 0.0, 1.0, 1.0]])

    solver = _make_solver(prob, device)
    sol_cold = solver.solve(qs=q, bs=b)
    ws = sol_cold.to_warm_start()

    # Use custom no_retry that includes MaxIterations -- should NOT retry
    method = "qdldl" if device == "cpu" else "cudss"
    custom_ipm = moreau.IPMSettings(
        direct_solve_method=method,
        warm_start_no_retry={moreau.SolverStatus.Solved, moreau.SolverStatus.MaxIterations},
    )
    custom_settings = moreau.Settings(device=device, batch_size=1, ipm_settings=custom_ipm)
    solver2 = moreau.CompiledSolver(
        n=prob["n"],
        m=prob["m"],
        P_row_offsets=prob["P_row_offsets"],
        P_col_indices=prob["P_col_indices"],
        A_row_offsets=prob["A_row_offsets"],
        A_col_indices=prob["A_col_indices"],
        cones=prob["cones"],
        settings=custom_settings,
    )
    solver2.setup(P_values=prob["P_values"], A_values=prob["A_values"])

    original_solve = solver2._impl.solve
    call_count = [0]

    def mock_solve(qs_arg, bs_arg, **kwargs):
        call_count[0] += 1
        if kwargs.get("warm_x") is not None:
            result = original_solve(qs_arg, bs_arg, **kwargs)
            result["status"] = np.array([int(moreau.SolverStatus.MaxIterations)])
            return result
        return original_solve(qs_arg, bs_arg, **kwargs)

    solver2._impl.solve = mock_solve

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = solver2.solve(qs=q, bs=b, warm_start=ws)

    # MaxIterations is in our custom no-retry set, so no retry
    assert call_count[0] == 1
    assert len(w) == 0
    assert solver2.info.status[0] == moreau.SolverStatus.MaxIterations


# --- Solver (single problem) warm start tests --------------------------------


@pytest.fixture
def simple_qp_sparse():
    """Simple QP as scipy sparse matrices for use with Solver."""
    from scipy import sparse

    P = sparse.diags([2.0, 2.0], format="csr")
    q = np.array([1.0, -1.0])
    A = sparse.csr_matrix(
        [
            [1.0, 1.0],  # equality: x + y = 1
            [-1.0, 0.0],  # x >= 0
            [0.0, -1.0],  # y >= 0
            [1.0, 0.0],  # x <= 1
            [0.0, 1.0],  # y <= 1
        ]
    )
    b = np.array([1.0, 0.0, 0.0, 1.0, 1.0])
    cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=4)
    return {"P": P, "q": q, "A": A, "b": b, "cones": cones, "n": 2, "m": 5}


def _make_single_solver(prob, device):
    """Create a Solver (single problem) for the given problem."""
    settings = moreau.Settings(solver="ipm", device=device)
    return moreau.Solver(
        prob["P"],
        prob["q"],
        prob["A"],
        prob["b"],
        cones=prob["cones"],
        settings=settings,
    )


def test_solver_warm_start_fewer_iterations(simple_qp_sparse, device):
    """Solver.solve(warm_start=) should converge in fewer iterations."""
    prob = simple_qp_sparse

    # Cold solve
    solver_cold = _make_single_solver(prob, device)
    sol_cold = solver_cold.solve()
    assert solver_cold.info.status == moreau.SolverStatus.Solved
    iters_cold = solver_cold.info.iterations

    # Warm solve
    ws = sol_cold.to_warm_start()
    assert isinstance(ws, moreau.WarmStart)

    solver_warm = _make_single_solver(prob, device)
    sol_warm = solver_warm.solve(warm_start=ws)
    assert solver_warm.info.status == moreau.SolverStatus.Solved
    iters_warm = solver_warm.info.iterations

    assert (
        iters_warm < iters_cold
    ), f"Warm start ({iters_warm} iters) should be faster than cold ({iters_cold} iters)"
    np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4)


def test_solver_warm_start_type_validation(simple_qp_sparse, device):
    """Solver.solve() should reject non-WarmStart types."""
    solver = _make_single_solver(simple_qp_sparse, device)

    with pytest.raises(TypeError, match="warm_start must be a WarmStart"):
        solver.solve(warm_start="not a warm start")

    # BatchedWarmStart should also be rejected (Solver expects WarmStart, not Batched)
    with pytest.raises(TypeError, match="warm_start must be a WarmStart"):
        solver.solve(
            warm_start=moreau.BatchedWarmStart(
                x=np.zeros((1, 2)),
                z=np.ones((1, 5)),
                s=np.ones((1, 5)),
            )
        )


def test_solver_warm_start_dim_validation(simple_qp_sparse, device):
    """Solver.solve() should reject wrong-dimension WarmStart."""
    solver = _make_single_solver(simple_qp_sparse, device)

    ws_bad = moreau.WarmStart(
        x=np.zeros(3),  # wrong: n=2
        z=np.ones(5),
        s=np.ones(5),
    )
    with pytest.raises(ValueError, match="warm_start.x shape"):
        solver.solve(warm_start=ws_bad)


def test_solver_warm_start_retry_on_failure(simple_qp_sparse, device):
    """Solver auto-retries cold if warm-started solve fails."""
    prob = simple_qp_sparse

    solver = _make_single_solver(prob, device)
    sol_cold = solver.solve()
    ws = sol_cold.to_warm_start()

    solver2 = _make_single_solver(prob, device)
    original_solve = solver2._impl.solve
    call_count = [0]

    def mock_solve(q_arg, b_arg, **kwargs):
        call_count[0] += 1
        if kwargs.get("warm_x") is not None:
            result = original_solve(q_arg, b_arg, **kwargs)
            result["status"] = int(moreau.SolverStatus.NumericalError)
            return result
        return original_solve(q_arg, b_arg, **kwargs)

    solver2._impl.solve = mock_solve

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol = solver2.solve(warm_start=ws)

    assert call_count[0] == 2
    assert len(w) == 1
    assert "NumericalError" in str(w[0].message)
    assert solver2.info.status == moreau.SolverStatus.Solved


def test_solver_to_warm_start_type(simple_qp_sparse, device):
    """Solution.to_warm_start() should return WarmStart with correct shapes."""
    solver = _make_single_solver(simple_qp_sparse, device)
    sol = solver.solve()

    ws = sol.to_warm_start()
    assert isinstance(ws, moreau.WarmStart)
    assert ws.x.shape == (simple_qp_sparse["n"],)
    assert ws.z.shape == (simple_qp_sparse["m"],)
    assert ws.s.shape == (simple_qp_sparse["m"],)
