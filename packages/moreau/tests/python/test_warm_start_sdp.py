"""Tests for warm starting with PSD (SDP) cones.

These tests verify:
1. Warm start from a cold SDP solution converges in fewer iterations
2. Warm start with perturbed SDP problem data works
3. Mixed cones (zero + PSD) with warm start
4. Batched SDP warm start
5. CPU/GPU parity for SDP warm start
6. Bad warm start point still converges for PSD
7. Warm start with multiple PSD cones of different sizes
"""

import moreau
import pytest
import warnings
import numpy as np


def _make_psd_svec(mat_dim, rng):
    """Generate a random PSD matrix and return its svec representation."""
    L = rng.standard_normal((mat_dim, mat_dim)) * 0.5
    M = L @ L.T + 0.1 * np.eye(mat_dim)
    svec = []
    for j in range(mat_dim):
        for i in range(j, mat_dim):
            if i == j:
                svec.append(M[i, j])
            else:
                svec.append(M[i, j] * np.sqrt(2))
    return np.array(svec)


@pytest.fixture
def sdp_problem():
    """Simple SDP: min 0.5*x'Px + q'x  s.t.  x + s = b,  s in PSD(3).

    PSD(3) has svec dimension 6.
    """
    n = 6
    m = 6

    P_row_offsets = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    P_col_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    P_values = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    A_row_offsets = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    A_col_indices = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    A_values = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    cones = moreau.Cones(psd_dims=[3])

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
def mixed_sdp_problem():
    """Mixed cones: 1 zero (equality) + PSD(2).

    n=2 decision vars, m=4 constraints (1 zero + 3 PSD(2) svec).
    """
    n = 2
    m = 4

    P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    P_col_indices = np.array([0, 1], dtype=np.int64)
    P_values = np.array([2.0, 2.0])

    A_row_offsets = np.array([0, 2, 4, 5, 6], dtype=np.int64)
    A_col_indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    A_values = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    cones = moreau.Cones(num_zero_cones=1, psd_dims=[2])

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
def multi_psd_problem():
    """Multiple PSD cones: PSD(2) + PSD(3).

    PSD(2) svec dim = 3, PSD(3) svec dim = 6. Total m = 9.
    """
    n = 9
    m = 9

    P_row_offsets = np.arange(n + 1, dtype=np.int64)
    P_col_indices = np.arange(n, dtype=np.int64)
    P_values = np.full(n, 2.0)

    A_row_offsets = np.arange(m + 1, dtype=np.int64)
    A_col_indices = np.arange(m, dtype=np.int64)
    A_values = np.ones(m)

    cones = moreau.Cones(psd_dims=[2, 3])

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


def _make_solver(prob, device, batch_size=1):
    """Create a CompiledSolver for the given problem."""
    method = "qdldl" if device == "cpu" else "cudss"
    ipm = moreau.IPMSettings(direct_solve_method=method)
    settings = moreau.Settings(
        device=device, batch_size=batch_size, verbose=False, ipm_settings=ipm
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


def _sdp_b(mat_dim, seed=42):
    """Generate a b vector corresponding to a PSD matrix in svec form."""
    rng = np.random.default_rng(seed)
    return _make_psd_svec(mat_dim, rng)


# --- CompiledSolver (batch) warm start tests for SDP ---


def test_warm_start_sdp_fewer_iterations(sdp_problem, device):
    """Warm starting SDP from a cold solution should converge in fewer iterations."""
    prob = sdp_problem
    b = _sdp_b(3, seed=42)
    q = np.zeros((1, prob["n"]))
    bs = b.reshape(1, -1)

    # Cold solve
    solver_cold = _make_solver(prob, device)
    sol_cold = solver_cold.solve(qs=q, bs=bs)
    info_cold = solver_cold.info
    assert info_cold.status[0] == moreau.SolverStatus.Solved
    iters_cold = (
        info_cold.iterations[0] if isinstance(info_cold.iterations, list) else info_cold.iterations
    )

    # Warm solve from cold solution — should not trigger auto-retry warning
    ws = sol_cold.to_warm_start()
    solver_warm = _make_solver(prob, device)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sol_warm = solver_warm.solve(qs=q, bs=bs, warm_start=ws)

    info_warm = solver_warm.info
    assert info_warm.status[0] == moreau.SolverStatus.Solved
    iters_warm = (
        info_warm.iterations[0] if isinstance(info_warm.iterations, list) else info_warm.iterations
    )

    # No auto-retry should have been needed
    assert len(w) == 0, f"Unexpected warning: {w[0].message if w else ''}"

    assert (
        iters_warm < iters_cold
    ), f"Warm start ({iters_warm} iters) should be faster than cold ({iters_cold} iters)"
    np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4)


def test_warm_start_sdp_auto_retry_correct(sdp_problem, device):
    """SDP warm-start with auto-retry produces correct solutions."""
    prob = sdp_problem
    b = _sdp_b(3, seed=42)
    q = np.zeros((1, prob["n"]))
    bs = b.reshape(1, -1)

    # Cold solve for reference
    solver_cold = _make_solver(prob, device)
    sol_cold = solver_cold.solve(qs=q, bs=bs)
    assert solver_cold.info.status[0] == moreau.SolverStatus.Solved

    # Warm solve (may auto-retry, that's OK)
    ws = sol_cold.to_warm_start()
    solver_warm = _make_solver(prob, device)
    sol_warm = solver_warm.solve(qs=q, bs=bs, warm_start=ws)
    assert solver_warm.info.status[0] == moreau.SolverStatus.Solved

    np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4)


def test_warm_start_sdp_perturbed_problem(sdp_problem, device):
    """Warm start from SDP P1's solution should help perturbed P2 converge."""
    prob = sdp_problem
    b1 = _sdp_b(3, seed=42)
    rng = np.random.default_rng(99)
    b2 = b1 + rng.standard_normal(b1.shape) * 0.05

    q = np.zeros((1, prob["n"]))

    # Solve P1 cold
    solver1 = _make_solver(prob, device)
    sol1 = solver1.solve(qs=q, bs=b1.reshape(1, -1))
    assert solver1.info.status[0] == moreau.SolverStatus.Solved

    # Solve P2 cold for reference
    solver2_cold = _make_solver(prob, device)
    sol2_cold = solver2_cold.solve(qs=q, bs=b2.reshape(1, -1))
    assert solver2_cold.info.status[0] == moreau.SolverStatus.Solved

    # Solve P2 warm from P1's solution
    ws = sol1.to_warm_start()
    solver2_warm = _make_solver(prob, device)
    sol2_warm = solver2_warm.solve(qs=q, bs=b2.reshape(1, -1), warm_start=ws)
    assert solver2_warm.info.status[0] == moreau.SolverStatus.Solved

    np.testing.assert_allclose(sol2_warm.x, sol2_cold.x, atol=1e-4)


def test_warm_start_sdp_mixed_cones(mixed_sdp_problem, device):
    """Warm start with mixed zero + PSD cones."""
    prob = mixed_sdp_problem
    q = np.array([[0.5, -0.5]])
    b = np.array([[1.0, 2.0, 0.0, 2.0]])

    # Cold solve
    solver_cold = _make_solver(prob, device)
    sol_cold = solver_cold.solve(qs=q, bs=b)
    assert solver_cold.info.status[0] == moreau.SolverStatus.Solved

    # Warm solve
    ws = sol_cold.to_warm_start()
    solver_warm = _make_solver(prob, device)
    sol_warm = solver_warm.solve(qs=q, bs=b, warm_start=ws)
    assert solver_warm.info.status[0] == moreau.SolverStatus.Solved

    np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4)


def test_warm_start_sdp_multiple_psd_cones(multi_psd_problem, device):
    """Warm start with multiple PSD cones of different sizes."""
    prob = multi_psd_problem
    b_psd2 = _sdp_b(2, seed=10)
    b_psd3 = _sdp_b(3, seed=11)
    b = np.concatenate([b_psd2, b_psd3])

    q = np.zeros((1, prob["n"]))
    bs = b.reshape(1, -1)

    # Cold solve
    solver_cold = _make_solver(prob, device)
    sol_cold = solver_cold.solve(qs=q, bs=bs)
    assert solver_cold.info.status[0] == moreau.SolverStatus.Solved

    # Warm solve
    ws = sol_cold.to_warm_start()
    solver_warm = _make_solver(prob, device)
    sol_warm = solver_warm.solve(qs=q, bs=bs, warm_start=ws)
    assert solver_warm.info.status[0] == moreau.SolverStatus.Solved

    np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4)


def test_warm_start_sdp_bad_point_still_converges(sdp_problem, device):
    """A random warm start point should still converge for SDP."""
    prob = sdp_problem
    b = _sdp_b(3, seed=42)
    q = np.zeros((1, prob["n"]))
    bs = b.reshape(1, -1)

    # Cold solve for reference
    solver_ref = _make_solver(prob, device)
    sol_ref = solver_ref.solve(qs=q, bs=bs)
    assert solver_ref.info.status[0] == moreau.SolverStatus.Solved

    # Warm start with a somewhat reasonable but not optimal point
    ws = moreau.BatchedWarmStart(
        x=np.ones((1, prob["n"])) * 0.1,
        z=np.ones((1, prob["m"])),
        s=np.ones((1, prob["m"])),
    )

    solver_warm = _make_solver(prob, device)
    sol_warm = solver_warm.solve(qs=q, bs=bs, warm_start=ws)
    assert solver_warm.info.status[0] in [
        moreau.SolverStatus.Solved,
        moreau.SolverStatus.AlmostSolved,
    ]

    np.testing.assert_allclose(sol_warm.x, sol_ref.x, atol=1e-4)


def test_warm_start_sdp_batched(sdp_problem, device):
    """Batched SDP warm start: each problem gets its own warm-start point."""
    prob = sdp_problem
    batch_size = 3

    bs = np.array([_sdp_b(3, seed=s) for s in [42, 43, 44]])
    qs = np.zeros((batch_size, prob["n"]))

    # Cold solve
    solver_cold = _make_solver(prob, device, batch_size=batch_size)
    sol_cold = solver_cold.solve(qs=qs, bs=bs)
    info_cold = solver_cold.info
    assert all(s == moreau.SolverStatus.Solved for s in info_cold.status)

    # Warm solve
    ws = sol_cold.to_warm_start()
    assert isinstance(ws, moreau.BatchedWarmStart)
    assert len(ws) == batch_size

    solver_warm = _make_solver(prob, device, batch_size=batch_size)
    sol_warm = solver_warm.solve(qs=qs, bs=bs, warm_start=ws)
    info_warm = solver_warm.info
    assert all(s == moreau.SolverStatus.Solved for s in info_warm.status)

    np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4)


def test_warm_start_sdp_cpu_gpu_parity(sdp_problem):
    """SDP warm start solutions should match between CPU and GPU."""
    if not moreau.device_available("cuda"):
        pytest.skip("CUDA not available")

    prob = sdp_problem
    b = _sdp_b(3, seed=42)
    q = np.zeros((1, prob["n"]))
    bs = b.reshape(1, -1)

    # Cold solve on CPU for warm start values
    solver_cold = _make_solver(prob, "cpu")
    sol_cold = solver_cold.solve(qs=q, bs=bs)
    assert solver_cold.info.status[0] == moreau.SolverStatus.Solved

    ws = sol_cold.to_warm_start()

    # Warm solve on CPU
    solver_cpu = _make_solver(prob, "cpu")
    sol_cpu = solver_cpu.solve(qs=q, bs=bs, warm_start=ws)

    # Warm solve on CUDA
    solver_cuda = _make_solver(prob, "cuda")
    sol_cuda = solver_cuda.solve(qs=q, bs=bs, warm_start=ws)

    assert solver_cpu.info.status[0] == moreau.SolverStatus.Solved
    assert solver_cuda.info.status[0] == moreau.SolverStatus.Solved

    np.testing.assert_allclose(sol_cpu.x, sol_cuda.x, atol=1e-3)
    np.testing.assert_allclose(sol_cpu.z, sol_cuda.z, atol=1e-3)
    np.testing.assert_allclose(sol_cpu.s, sol_cuda.s, atol=1e-3)


def test_warm_start_sdp_cpu_gpu_parity_batched(sdp_problem):
    """Batched SDP warm start parity between CPU and GPU."""
    if not moreau.device_available("cuda"):
        pytest.skip("CUDA not available")

    prob = sdp_problem
    batch_size = 2
    bs = np.array([_sdp_b(3, seed=s) for s in [42, 43]])
    qs = np.zeros((batch_size, prob["n"]))

    # Cold solve on CPU
    solver_cold = _make_solver(prob, "cpu", batch_size=batch_size)
    sol_cold = solver_cold.solve(qs=qs, bs=bs)
    ws = sol_cold.to_warm_start()

    # Warm solve on CPU
    solver_cpu = _make_solver(prob, "cpu", batch_size=batch_size)
    sol_cpu = solver_cpu.solve(qs=qs, bs=bs, warm_start=ws)

    # Warm solve on CUDA
    solver_cuda = _make_solver(prob, "cuda", batch_size=batch_size)
    sol_cuda = solver_cuda.solve(qs=qs, bs=bs, warm_start=ws)

    assert all(s == moreau.SolverStatus.Solved for s in solver_cpu.info.status)
    assert all(s == moreau.SolverStatus.Solved for s in solver_cuda.info.status)

    np.testing.assert_allclose(sol_cpu.x, sol_cuda.x, atol=1e-3)
    np.testing.assert_allclose(sol_cpu.z, sol_cuda.z, atol=1e-3)
    np.testing.assert_allclose(sol_cpu.s, sol_cuda.s, atol=1e-3)


def test_warm_start_sdp_cpu_gpu_parity_mixed_cones(mixed_sdp_problem):
    """Mixed zero+PSD warm start parity between CPU and GPU."""
    if not moreau.device_available("cuda"):
        pytest.skip("CUDA not available")

    prob = mixed_sdp_problem
    q = np.array([[0.5, -0.5]])
    b = np.array([[1.0, 2.0, 0.0, 2.0]])

    # Cold solve on CPU
    solver_cold = _make_solver(prob, "cpu")
    sol_cold = solver_cold.solve(qs=q, bs=b)
    ws = sol_cold.to_warm_start()

    # Warm solve on CPU
    solver_cpu = _make_solver(prob, "cpu")
    sol_cpu = solver_cpu.solve(qs=q, bs=b, warm_start=ws)

    # Warm solve on CUDA
    solver_cuda = _make_solver(prob, "cuda")
    sol_cuda = solver_cuda.solve(qs=q, bs=b, warm_start=ws)

    assert solver_cpu.info.status[0] == moreau.SolverStatus.Solved
    assert solver_cuda.info.status[0] == moreau.SolverStatus.Solved

    np.testing.assert_allclose(sol_cpu.x, sol_cuda.x, atol=1e-3)
    np.testing.assert_allclose(sol_cpu.z, sol_cuda.z, atol=1e-3)
    np.testing.assert_allclose(sol_cpu.s, sol_cuda.s, atol=1e-3)


def test_warm_start_sdp_cpu_gpu_parity_multi_psd(multi_psd_problem):
    """Multiple PSD cones warm start parity between CPU and GPU."""
    if not moreau.device_available("cuda"):
        pytest.skip("CUDA not available")

    prob = multi_psd_problem
    b_psd2 = _sdp_b(2, seed=10)
    b_psd3 = _sdp_b(3, seed=11)
    b = np.concatenate([b_psd2, b_psd3])
    q = np.zeros((1, prob["n"]))
    bs = b.reshape(1, -1)

    # Cold solve on CPU
    solver_cold = _make_solver(prob, "cpu")
    sol_cold = solver_cold.solve(qs=q, bs=bs)
    ws = sol_cold.to_warm_start()

    # Warm solve on CPU
    solver_cpu = _make_solver(prob, "cpu")
    sol_cpu = solver_cpu.solve(qs=q, bs=bs, warm_start=ws)

    # Warm solve on CUDA
    solver_cuda = _make_solver(prob, "cuda")
    sol_cuda = solver_cuda.solve(qs=q, bs=bs, warm_start=ws)

    assert solver_cpu.info.status[0] == moreau.SolverStatus.Solved
    assert solver_cuda.info.status[0] == moreau.SolverStatus.Solved

    np.testing.assert_allclose(sol_cpu.x, sol_cuda.x, atol=1e-3)


# --- Solver (single problem) warm start tests for SDP ---


def test_solver_warm_start_sdp_correct(device):
    """Single Solver warm start with PSD cone produces correct solution."""
    from scipy import sparse

    n, m = 6, 6
    P = sparse.eye(n, format="csr") * 2.0
    A = sparse.eye(m, format="csr")
    b = _sdp_b(3, seed=42)
    q = np.zeros(n)
    cones = moreau.Cones(psd_dims=[3])

    settings = moreau.Settings(device=device)
    settings.ipm_settings.tol_gap_abs = 1e-9
    settings.ipm_settings.tol_feas = 1e-9

    # Cold solve
    solver_cold = moreau.Solver(P, q, A, b, cones, settings=settings)
    sol_cold = solver_cold.solve()
    assert solver_cold.info.status == moreau.SolverStatus.Solved

    # Warm solve (may auto-retry, that's OK)
    ws = sol_cold.to_warm_start()
    assert isinstance(ws, moreau.WarmStart)

    solver_warm = moreau.Solver(P, q, A, b, cones, settings=settings)
    sol_warm = solver_warm.solve(warm_start=ws)
    assert solver_warm.info.status == moreau.SolverStatus.Solved

    np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4)


def test_solver_warm_start_sdp_to_warm_start_types(device):
    """Verify warm start types for SDP single solver."""
    from scipy import sparse

    n, m = 3, 3
    P = sparse.eye(n, format="csr") * 2.0
    A = sparse.eye(m, format="csr")
    b = _sdp_b(2, seed=42)
    q = np.zeros(n)
    cones = moreau.Cones(psd_dims=[2])

    settings = moreau.Settings(device=device)
    solver = moreau.Solver(P, q, A, b, cones, settings=settings)
    sol = solver.solve()

    ws = sol.to_warm_start()
    assert isinstance(ws, moreau.WarmStart)
    assert ws.x.shape == (n,)
    assert ws.z.shape == (m,)
    assert ws.s.shape == (m,)
