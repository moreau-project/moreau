"""Tests for warm starting in PyTorch and JAX wrappers.

These tests verify:
1. Warm start from a previous solution converges in fewer iterations
2. to_warm_start() returns correct types (WarmStart with numpy arrays)
3. warm_start= parameter works for both torch and JAX Solver.solve()
4. Type validation rejects non-WarmStart objects
5. Gradients still work with warm start
6. Batched warm start works
"""

import numpy as np
import pytest
import warnings

import moreau
from moreau._types import WarmStart, BatchedWarmStart

# ─── Shared problem fixtures ─────────────────────────────────────────────────


def _make_problem():
    """Return the shared problem structure (CSR arrays + cones)."""
    n, m = 2, 5
    P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    P_col_indices = np.array([0, 1], dtype=np.int64)
    A_row_offsets = np.array([0, 2, 3, 4, 5, 6], dtype=np.int64)
    A_col_indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
    cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=4)
    return dict(
        n=n,
        m=m,
        P_row_offsets=P_row_offsets,
        P_col_indices=P_col_indices,
        A_row_offsets=A_row_offsets,
        A_col_indices=A_col_indices,
        cones=cones,
    )


P_VALS = np.array([2.0, 2.0])
A_VALS = np.array([1.0, 1.0, -1.0, -1.0, 1.0, 1.0])
Q_VALS = np.array([1.0, -1.0])
B_VALS = np.array([1.0, 0.0, 0.0, 1.0, 1.0])


# ─── PyTorch warm start tests ────────────────────────────────────────────────


try:
    import torch

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

try:
    from moreau.torch import Solver as TorchSolver

    HAS_MOREAU_TORCH = TorchSolver is not None
except ImportError:
    HAS_MOREAU_TORCH = False
    TorchSolver = None


@pytest.fixture(params=["cpu", "cuda"] if HAS_CUDA else ["cpu"])
def torch_device(request):
    """Parametrized torch device."""
    if request.param == "cuda" and not HAS_CUDA:
        pytest.skip("CUDA not available")
    return request.param


def _make_torch_solver(device, batch_size=1, verbose=False):
    """Create a moreau.torch.Solver for the standard problem.

    Returns (solver, P, A) so callers can pass P/A to solve().
    """
    prob = _make_problem()
    settings = moreau.Settings(device=device, batch_size=batch_size, verbose=verbose, solver="ipm")
    solver = TorchSolver(
        n=prob["n"],
        m=prob["m"],
        P_row_offsets=torch.tensor(prob["P_row_offsets"], dtype=torch.int64),
        P_col_indices=torch.tensor(prob["P_col_indices"], dtype=torch.int64),
        A_row_offsets=torch.tensor(prob["A_row_offsets"], dtype=torch.int64),
        A_col_indices=torch.tensor(prob["A_col_indices"], dtype=torch.int64),
        cones=prob["cones"],
        settings=settings,
    )
    if batch_size == 1:
        P = torch.tensor(P_VALS, dtype=torch.float64, device=device)
        A = torch.tensor(A_VALS, dtype=torch.float64, device=device)
    else:
        P = (
            torch.tensor(P_VALS, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        A = (
            torch.tensor(A_VALS, dtype=torch.float64, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
    return solver, P, A


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not HAS_MOREAU_TORCH, reason="moreau torch not available")
class TestTorchWarmStart:
    """PyTorch warm start tests (CPU and CUDA)."""

    def test_fewer_iterations(self, torch_device):
        """Warm starting from a cold-solved solution should converge in fewer iterations."""
        solver_cold, P, A = _make_torch_solver(torch_device)
        q = torch.tensor(Q_VALS, dtype=torch.float64, device=torch_device)
        b = torch.tensor(B_VALS, dtype=torch.float64, device=torch_device)

        # Cold solve
        sol_cold = solver_cold.solve(P, A, q, b)
        info_cold = solver_cold.info
        assert info_cold.status == moreau.SolverStatus.Solved
        iters_cold = info_cold.iterations

        # Warm solve (fresh solver, same problem)
        ws = sol_cold.to_warm_start()
        solver_warm, P_w, A_w = _make_torch_solver(torch_device)
        sol_warm = solver_warm.solve(P_w, A_w, q, b, warm_start=ws)
        info_warm = solver_warm.info
        assert info_warm.status == moreau.SolverStatus.Solved
        iters_warm = info_warm.iterations

        assert (
            iters_warm < iters_cold
        ), f"Warm start ({iters_warm} iters) should be faster than cold ({iters_cold} iters)"
        torch.testing.assert_close(sol_warm.x, sol_cold.x, rtol=1e-4, atol=1e-4)

    def test_to_warm_start_type(self, torch_device):
        """TorchSolution.to_warm_start() should return WarmStart with numpy arrays."""
        solver, P, A = _make_torch_solver(torch_device)
        q = torch.tensor(Q_VALS, dtype=torch.float64, device=torch_device)
        b = torch.tensor(B_VALS, dtype=torch.float64, device=torch_device)

        sol = solver.solve(P, A, q, b)
        ws = sol.to_warm_start()

        assert isinstance(ws, WarmStart)
        assert isinstance(ws.x, np.ndarray)
        assert isinstance(ws.z, np.ndarray)
        assert isinstance(ws.s, np.ndarray)
        assert ws.x.dtype == np.float64
        assert ws.x.shape == (2,)
        assert ws.z.shape == (5,)
        assert ws.s.shape == (5,)

    def test_batched_to_warm_start_type(self, torch_device):
        """TorchBatchedSolution.to_warm_start() should return BatchedWarmStart."""
        batch_size = 2
        solver, P, A = _make_torch_solver(torch_device, batch_size=batch_size)
        q = (
            torch.tensor(Q_VALS, dtype=torch.float64, device=torch_device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        b = (
            torch.tensor(B_VALS, dtype=torch.float64, device=torch_device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )

        sol = solver.solve(P, A, q, b)
        ws = sol.to_warm_start()

        assert isinstance(ws, BatchedWarmStart)
        assert len(ws) == batch_size
        assert ws.x.shape == (batch_size, 2)
        assert ws.z.shape == (batch_size, 5)
        assert ws.s.shape == (batch_size, 5)

        # Indexing should give WarmStart
        w0 = ws[0]
        assert isinstance(w0, WarmStart)

    def test_type_validation(self, torch_device):
        """Passing a non-WarmStart should raise TypeError."""
        solver, P, A = _make_torch_solver(torch_device)
        q = torch.tensor(Q_VALS, dtype=torch.float64, device=torch_device)
        b = torch.tensor(B_VALS, dtype=torch.float64, device=torch_device)

        with pytest.raises(TypeError, match="warm_start must be a WarmStart"):
            solver.solve(P, A, q, b, warm_start="not a warm start")

    def test_gradients_with_warm_start(self, torch_device):
        """Gradients should flow through q even when warm_start is provided."""
        solver, P, A = _make_torch_solver(torch_device)
        q = torch.tensor(Q_VALS, dtype=torch.float64, device=torch_device, requires_grad=True)
        b = torch.tensor(B_VALS, dtype=torch.float64, device=torch_device)

        # Cold solve to get warm start
        sol_cold = solver.solve(P, A, q, b)
        ws = sol_cold.to_warm_start()

        # Detach q, re-enable grad for fresh computation
        q2 = q.detach().clone().requires_grad_(True)

        # Warm solve with gradient tracking
        solver2, P2, A2 = _make_torch_solver(torch_device)
        sol_warm = solver2.solve(P2, A2, q2, b, warm_start=ws)

        loss = sol_warm.x.sum()
        loss.backward()

        assert q2.grad is not None
        assert q2.grad.shape == q2.shape
        assert torch.abs(q2.grad).sum() > 0

    def test_warm_start_perturbed_q(self, torch_device):
        """Warm start from q1's solution should help with slightly perturbed q2."""
        q1 = torch.tensor([1.0, -1.0], dtype=torch.float64, device=torch_device)
        q2 = torch.tensor([1.1, -0.9], dtype=torch.float64, device=torch_device)
        b = torch.tensor(B_VALS, dtype=torch.float64, device=torch_device)

        # Solve q1 cold
        solver1, P1, A1 = _make_torch_solver(torch_device)
        sol1 = solver1.solve(P1, A1, q1, b)
        ws = sol1.to_warm_start()

        # Solve q2 cold
        solver2_cold, P2c, A2c = _make_torch_solver(torch_device)
        sol2_cold = solver2_cold.solve(P2c, A2c, q2, b)
        iters_cold = solver2_cold.info.iterations

        # Solve q2 warm from q1
        solver2_warm, P2w, A2w = _make_torch_solver(torch_device)
        sol2_warm = solver2_warm.solve(P2w, A2w, q2, b, warm_start=ws)
        iters_warm = solver2_warm.info.iterations

        # Warm should use fewer iterations for the perturbed problem
        assert iters_warm <= iters_cold
        # Solutions should match
        torch.testing.assert_close(sol2_warm.x, sol2_cold.x, rtol=1e-4, atol=1e-4)

    def test_batched_warm_start(self, torch_device):
        """Batched warm start should work with batched solver."""
        batch_size = 2
        solver_cold, P, A = _make_torch_solver(torch_device, batch_size=batch_size)
        q = torch.tensor([[1.0, -1.0], [0.5, -0.5]], dtype=torch.float64, device=torch_device)
        b = (
            torch.tensor(B_VALS, dtype=torch.float64, device=torch_device)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )

        sol_cold = solver_cold.solve(P, A, q, b)
        ws = sol_cold.to_warm_start()
        assert isinstance(ws, BatchedWarmStart)

        solver_warm, Pw, Aw = _make_torch_solver(torch_device, batch_size=batch_size)
        sol_warm = solver_warm.solve(Pw, Aw, q, b, warm_start=ws)

        status = solver_warm.info.status
        assert all(s == moreau.SolverStatus.Solved for s in status)
        torch.testing.assert_close(sol_warm.x, sol_cold.x, rtol=1e-4, atol=1e-4)


# ─── JAX warm start tests ────────────────────────────────────────────────────


try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    HAS_JAX = False

try:
    from moreau.jax import Solver as JaxSolver
    from moreau._backend import jax_available

    HAS_MOREAU_JAX = JaxSolver is not None
except ImportError:
    HAS_MOREAU_JAX = False
    JaxSolver = None
    jax_available = lambda d: False


@pytest.fixture(params=["cpu", "cuda"])
def jax_device(request):
    """Parametrized JAX device."""
    device = request.param
    if device == "cuda" and not jax_available("cuda"):
        pytest.skip("CUDA JAX not available")
    return device


def _make_jax_solver(device, jit=False):
    """Create a moreau.jax.Solver for the standard problem."""
    prob = _make_problem()
    method = "qdldl" if device == "cpu" else "cudss"
    ipm = moreau.IPMSettings(direct_solve_method=method)
    settings = moreau.Settings(device=device, ipm_settings=ipm)
    solver = JaxSolver(
        n=prob["n"],
        m=prob["m"],
        P_row_offsets=prob["P_row_offsets"],
        P_col_indices=prob["P_col_indices"],
        A_row_offsets=prob["A_row_offsets"],
        A_col_indices=prob["A_col_indices"],
        cones=prob["cones"],
        settings=settings,
        jit=jit,
    )
    return solver


@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
@pytest.mark.skipif(not HAS_MOREAU_JAX, reason="moreau JAX not available")
class TestJaxWarmStart:
    """JAX warm start tests."""

    def test_fewer_iterations(self, jax_device):
        """Warm starting from a cold-solved solution should converge in fewer iterations."""
        solver_cold = _make_jax_solver(jax_device, jit=False)
        P_data = jnp.array(P_VALS)
        A_data = jnp.array(A_VALS)
        q = jnp.array(Q_VALS)
        b = jnp.array(B_VALS)

        # Cold solve
        sol_cold = solver_cold.solve(P_data, A_data, q, b)
        info_cold = solver_cold.info
        iters_cold = int(jnp.asarray(info_cold.iterations))

        # Warm solve
        ws = sol_cold.to_warm_start()
        solver_warm = _make_jax_solver(jax_device, jit=False)
        sol_warm = solver_warm.solve(P_data, A_data, q, b, warm_start=ws)
        info_warm = solver_warm.info
        iters_warm = int(jnp.asarray(info_warm.iterations))

        assert (
            iters_warm <= iters_cold
        ), f"Warm start ({iters_warm} iters) should be <= cold ({iters_cold} iters)"
        assert jnp.allclose(sol_warm.x, sol_cold.x, atol=1e-4)

    def test_to_warm_start_type(self, jax_device):
        """JaxSolution.to_warm_start() should return WarmStart with numpy arrays."""
        solver = _make_jax_solver(jax_device, jit=False)
        P_data = jnp.array(P_VALS)
        A_data = jnp.array(A_VALS)
        q = jnp.array(Q_VALS)
        b = jnp.array(B_VALS)

        sol = solver.solve(P_data, A_data, q, b)
        ws = sol.to_warm_start()

        assert isinstance(ws, WarmStart)
        assert isinstance(ws.x, np.ndarray)
        assert isinstance(ws.z, np.ndarray)
        assert isinstance(ws.s, np.ndarray)
        assert ws.x.dtype == np.float64
        prob = _make_problem()
        assert ws.x.shape == (prob["n"],)
        assert ws.z.shape == (prob["m"],)
        assert ws.s.shape == (prob["m"],)

    def test_type_validation(self, jax_device):
        """Passing a non-WarmStart should raise TypeError."""
        solver = _make_jax_solver(jax_device, jit=False)
        P_data = jnp.array(P_VALS)
        A_data = jnp.array(A_VALS)
        q = jnp.array(Q_VALS)
        b = jnp.array(B_VALS)

        with pytest.raises(TypeError, match="warm_start must be a WarmStart"):
            solver.solve(P_data, A_data, q, b, warm_start="not a warm start")

    def test_warm_start_two_step_api(self, jax_device):
        """Warm start should work with the two-step setup()/solve() API."""
        solver = _make_jax_solver(jax_device, jit=False)
        P_data = jnp.array(P_VALS)
        A_data = jnp.array(A_VALS)
        q = jnp.array(Q_VALS)
        b = jnp.array(B_VALS)

        # Full-signature cold solve
        sol_cold = solver.solve(P_data, A_data, q, b)
        ws = sol_cold.to_warm_start()

        # Two-step warm solve
        solver2 = _make_jax_solver(jax_device, jit=False)
        sol_warm = solver2.solve(P_data, A_data, q, b, warm_start=ws)

        assert jnp.allclose(sol_warm.x, sol_cold.x, atol=1e-4)

    def test_warm_start_perturbed_q(self, jax_device):
        """Warm start from q1 should help with slightly perturbed q2."""
        solver1 = _make_jax_solver(jax_device, jit=False)
        P_data = jnp.array(P_VALS)
        A_data = jnp.array(A_VALS)
        b = jnp.array(B_VALS)

        q1 = jnp.array([1.0, -1.0])
        q2 = jnp.array([1.1, -0.9])

        # Solve q1 cold
        sol1 = solver1.solve(P_data, A_data, q1, b)
        ws = sol1.to_warm_start()

        # Solve q2 cold
        solver2_cold = _make_jax_solver(jax_device, jit=False)
        sol2_cold = solver2_cold.solve(P_data, A_data, q2, b)
        iters_cold = int(jnp.asarray(solver2_cold.info.iterations))

        # Solve q2 warm
        solver2_warm = _make_jax_solver(jax_device, jit=False)
        sol2_warm = solver2_warm.solve(P_data, A_data, q2, b, warm_start=ws)
        iters_warm = int(jnp.asarray(solver2_warm.info.iterations))

        assert iters_warm <= iters_cold
        assert jnp.allclose(sol2_warm.x, sol2_cold.x, atol=1e-4)

    def test_gradients_with_warm_start(self, jax_device):
        """Gradients should flow through q even when warm_start is provided."""
        solver = _make_jax_solver(jax_device, jit=False)
        P_data = jnp.array(P_VALS)
        A_data = jnp.array(A_VALS)
        q = jnp.array(Q_VALS)
        b = jnp.array(B_VALS)

        # Cold solve for warm start
        sol_cold = solver.solve(P_data, A_data, q, b)
        ws = sol_cold.to_warm_start()

        # Gradient computation with warm start
        solver2 = _make_jax_solver(jax_device, jit=False)

        def loss_fn(q_):
            return jnp.sum(solver2.solve(P_data, A_data, q_, b, warm_start=ws).x)

        grad_q = jax.grad(loss_fn)(q)

        assert not jnp.any(jnp.isnan(grad_q))
        assert grad_q.shape == q.shape
        assert jnp.abs(grad_q).sum() > 0
