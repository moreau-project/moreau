"""Tests for gradient correctness when chaining multiple solves in one autograd graph.

When the same solver is called multiple times in a loop (e.g. MPC rollouts),
each setup() call overwrites internal state. The backward pass must restore
the correct state for each solve in the chain.

Tests run on all available devices (CPU and CUDA when available).
"""

import pytest
import numpy as np

torch = pytest.importorskip("torch")

import moreau
from moreau.torch import Solver


@pytest.fixture
def simple_qp():
    """Simple QP: min (1/2)x'Px + q'x  s.t. Ax + s = b, s >= 0.

    P = diag([2, 2]), A = -I (x >= 0), n=2, m=2.
    """
    n, m = 2, 2
    P_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
    P_col_indices = torch.tensor([0, 1], dtype=torch.int64)
    A_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
    A_col_indices = torch.tensor([0, 1], dtype=torch.int64)
    cones = moreau.Cones(num_nonneg_cones=2)
    return n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones


class TestChainedSolveGradients:
    """Test gradient correctness for chained solves (same solver, multiple calls)."""

    def test_chained_q_varies(self, simple_qp, device):
        """Chain 3 solves where q depends on previous solution. Grad w.r.t. initial q."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)

        q0 = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device, requires_grad=True)

        def chained_loss(q_init):
            q = q_init
            total = 0.0
            for _ in range(3):
                result = solver.solve(P_values, A_values, q, b)
                x = result.x
                total = total + x.sum()
                q = q + 0.1 * x  # next q depends on solution
            return total

        # Compare autograd vs finite differences
        assert torch.autograd.gradcheck(
            chained_loss,
            (q0,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
            nondet_tol=1e-5,
        )

    def test_chained_b_varies(self, simple_qp, device):
        """Chain 3 solves where b depends on previous solution. Grad w.r.t. initial b."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0], dtype=torch.float64, device=device)
        q = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device)

        b0 = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device, requires_grad=True)

        def chained_loss(b_init):
            b_cur = b_init
            total = 0.0
            for _ in range(3):
                result = solver.solve(P_values, A_values, q, b_cur)
                x = result.x
                total = total + x.sum()
                b_cur = b_cur + 0.1 * x
            return total

        assert torch.autograd.gradcheck(
            chained_loss,
            (b0,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
            nondet_tol=1e-5,
        )

    def test_chained_P_varies(self, simple_qp, device):
        """Chain 2 solves where P depends on a learnable parameter. Grad w.r.t. P param."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        A_values = torch.tensor([-1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)

        # Learnable parameter that enters P
        alpha = torch.tensor([2.0], dtype=torch.float64, device=device, requires_grad=True)

        def chained_loss(alpha_):
            P_vals = alpha_.expand(2)  # P = diag([alpha, alpha])
            q = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device)
            total = 0.0
            for _ in range(2):
                result = solver.solve(P_vals, A_values, q, b)
                x = result.x
                total = total + x.sum()
                q = q + 0.1 * x
            return total

        assert torch.autograd.gradcheck(
            chained_loss,
            (alpha,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
            nondet_tol=1e-5,
        )

    def test_chained_A_varies(self, simple_qp, device):
        """Chain 2 solves where A depends on a learnable parameter. Grad w.r.t. A param."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        b = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)

        beta = torch.tensor([-1.0], dtype=torch.float64, device=device, requires_grad=True)

        def chained_loss(beta_):
            A_vals = beta_.expand(2)  # A = diag([beta, beta])
            q = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device)
            total = 0.0
            for _ in range(2):
                result = solver.solve(P_values, A_vals, q, b)
                x = result.x
                total = total + x.sum()
                q = q + 0.1 * x
            return total

        assert torch.autograd.gradcheck(
            chained_loss,
            (beta,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
            nondet_tol=1e-5,
        )

    def test_chained_all_vary(self, simple_qp, device):
        """Chain 2 solves with gradients w.r.t. P, A, q, b simultaneously."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device, requires_grad=True)
        A_values = torch.tensor(
            [-1.0, -1.0], dtype=torch.float64, device=device, requires_grad=True
        )
        q0 = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device, requires_grad=True)
        b0 = torch.tensor([0.5, 0.5], dtype=torch.float64, device=device, requires_grad=True)

        def chained_loss(P_, A_, q_, b_):
            total = 0.0
            q_cur = q_
            b_cur = b_
            for _ in range(2):
                result = solver.solve(P_, A_, q_cur, b_cur)
                x = result.x
                total = total + x.sum()
                q_cur = q_cur + 0.1 * x
                b_cur = b_cur + 0.05 * x
            return total

        assert torch.autograd.gradcheck(
            chained_loss,
            (P_values, A_values, q0, b0),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
            nondet_tol=1e-5,
        )

    def test_chained_batched(self, simple_qp, device):
        """Chain 2 batched solves. Grad w.r.t. q batch."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float64, device=device)

        q0 = torch.tensor(
            [[-1.0, -0.5], [-0.3, -0.8]], dtype=torch.float64, device=device, requires_grad=True
        )

        def chained_loss(q_):
            q_cur = q_
            total = 0.0
            for _ in range(2):
                result = solver.solve(P_values, A_values, q_cur, b)
                x = result.x
                total = total + x.sum()
                q_cur = q_cur + 0.1 * x
            return total

        assert torch.autograd.gradcheck(
            chained_loss,
            (q0,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
            nondet_tol=1e-5,
        )


class TestChainedSolveFiniteDiff:
    """Manual finite-difference checks for chained solves (more readable diagnostics).

    These tests directly compare autograd gradients against finite differences
    and report explicit ratios, making it easy to diagnose the stale-state bug
    where T=1 works but T>=2 gives wrong gradients.
    """

    @pytest.mark.parametrize("T", [1, 2, 3, 5])
    def test_gradient_accuracy_vs_horizon(self, simple_qp, device, T):
        """Verify gradient accuracy at different time horizons.

        The stale-state bug manifests at T>=2: the solver's internal state
        from the last setup()/solve() is used for backward through all
        earlier solves, producing incorrect gradients. T=1 always works
        since there's only one solve.
        """
        n, m, P_ro, P_ci, A_ro, A_ci, cones = simple_qp

        solver = Solver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
        )

        P_values = torch.tensor([2.0, 2.0], dtype=torch.float64, device=device)
        A_values = torch.tensor([-1.0, -1.0], dtype=torch.float64, device=device)
        b = torch.tensor([0.0, 0.0], dtype=torch.float64, device=device)

        def rollout_loss(q_init):
            q = q_init
            total = 0.0
            for _ in range(T):
                result = solver.solve(P_values, A_values, q, b)
                x = result.x
                total = total + x.sum()
                q = q + 0.1 * x
            return total

        q0 = torch.tensor([-1.0, -0.5], dtype=torch.float64, device=device)

        # Autograd gradient
        q_ag = q0.clone().requires_grad_(True)
        loss = rollout_loss(q_ag)
        loss.backward()
        ag_grad = q_ag.grad.clone()

        # Finite difference gradient
        eps = 1e-6
        l0 = rollout_loss(q0).item()
        fd_grad = torch.zeros_like(q0)
        for i in range(n):
            q_pert = q0.clone()
            q_pert[i] += eps
            fd_grad[i] = (rollout_loss(q_pert).item() - l0) / eps

        # Check ratios are close to 1.0
        ratios = ag_grad / fd_grad
        torch.testing.assert_close(
            ratios,
            torch.ones_like(ratios),
            atol=1e-3,
            rtol=1e-3,
            msg=f"T={T}: autograd/fd ratios should be ~1.0, got {ratios.tolist()}",
        )


# ---------------------------------------------------------------------------
# JAX chained-solve gradient tests
# ---------------------------------------------------------------------------

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from moreau.jax import Solver as JaxSolver

    _has_jax = True
except ImportError:
    _has_jax = False


@pytest.mark.skipif(not _has_jax, reason="JAX not installed")
class TestJaxChainedSolveGradients:
    """Test gradient correctness for chained JAX solves (same solver, multiple calls)."""

    @staticmethod
    def _make_solver(cones):
        """Build a JaxSolver for the simple_qp fixture (n=2, m=2, diagonal P/A)."""
        return JaxSolver(
            n=2,
            m=2,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 1, 2],
            A_col_indices=[0, 1],
            cones=cones,
        )

    def test_single_solve_gradient(self, simple_qp, device):
        """Baseline: jax.grad through a single solve matches finite differences."""
        *_, cones = simple_qp
        solver = self._make_solver(cones)

        P_data = jnp.array([2.0, 2.0], dtype=jnp.float64)
        A_data = jnp.array([-1.0, -1.0], dtype=jnp.float64)
        q = jnp.array([-1.0, -0.5], dtype=jnp.float64)
        b = jnp.array([0.0, 0.0], dtype=jnp.float64)

        def loss_fn(q_):
            sol = solver.solve(P_data, A_data, q_, b)
            return sol.x.sum()

        ag_grad = jax.grad(loss_fn)(q)

        # Finite differences
        eps = 1e-5
        l0 = loss_fn(q)
        fd_grad = jnp.zeros_like(q)
        for i in range(2):
            q_pert = q.at[i].add(eps)
            fd_grad = fd_grad.at[i].set((loss_fn(q_pert) - l0) / eps)

        np.testing.assert_allclose(
            np.array(ag_grad),
            np.array(fd_grad),
            atol=1e-4,
            rtol=1e-3,
            err_msg="Single-solve JAX gradient does not match finite differences",
        )

    def test_chained_two_solves_grad_wrt_q1(self, simple_qp, device):
        """Two sequential solves; grad of combined loss w.r.t. q1 is correct.

        sol1 = solve(P, A, q1, b1)
        sol2 = solve(P, A, q2, b2)
        loss = sol1.x.sum() + sol2.x.sum()

        Even though sol2 overwrites solver state, the gradient w.r.t. q1
        must still be correct (the backward pass must use the saved data
        from the first solve, not the stale state from the second).
        """
        *_, cones = simple_qp
        solver = self._make_solver(cones)

        P_data = jnp.array([2.0, 2.0], dtype=jnp.float64)
        A_data = jnp.array([-1.0, -1.0], dtype=jnp.float64)
        q1 = jnp.array([-1.0, -0.5], dtype=jnp.float64)
        q2 = jnp.array([-0.3, -0.8], dtype=jnp.float64)
        b1 = jnp.array([0.0, 0.0], dtype=jnp.float64)
        b2 = jnp.array([0.5, 0.5], dtype=jnp.float64)

        def loss_fn(q1_):
            sol1 = solver.solve(P_data, A_data, q1_, b1)
            sol2 = solver.solve(P_data, A_data, q2, b2)
            return sol1.x.sum() + sol2.x.sum()

        ag_grad = jax.grad(loss_fn)(q1)

        # Finite differences
        eps = 1e-5
        l0 = loss_fn(q1)
        fd_grad = jnp.zeros_like(q1)
        for i in range(2):
            q1_pert = q1.at[i].add(eps)
            fd_grad = fd_grad.at[i].set((loss_fn(q1_pert) - l0) / eps)

        np.testing.assert_allclose(
            np.array(ag_grad),
            np.array(fd_grad),
            atol=1e-4,
            rtol=1e-3,
            err_msg="Chained two-solve JAX gradient w.r.t. q1 does not match finite differences",
        )
