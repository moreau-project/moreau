"""Backward through a problem with no solution must raise (issue #59).

The problem is x >= 1, x <= u with objective x^2 / 2. With u = 2 it is
feasible; with u = 0 it is primal infeasible. Backward raises only when the
loss depends on an infeasible problem, so masking failed problems out of
the loss still works.
"""

import numpy as np
import pytest

import moreau
from moreau._types import SolverStatus, _check_backward_statuses

N, M = 1, 2
P_RO, P_CI = [0, 1], [0]
A_RO, A_CI = [0, 1, 2], [0, 0]
P_VALS = np.array([1.0])
A_VALS = np.array([-1.0, 1.0])
Q = np.array([0.0])
B_FEAS = np.array([-1.0, 2.0])
B_INFEAS = np.array([-1.0, 0.0])
CONES = moreau.Cones(num_nonneg_cones=2)
SOLVERS = ["ipm", "active_set"]
MATCH = "Cannot backpropagate"


def _settings(solver, **kwargs):
    return moreau.Settings(device="cpu", solver=solver, enable_grad=True, **kwargs)


class TestHelper:
    def test_no_failure_ignores_upstream(self):
        _check_backward_statuses([1, 4, 7], np.ones((3, 2)))

    def test_used_failure_raises_with_indices(self):
        dx = np.array([[1.0], [0.0], [2.0]])
        with pytest.raises(RuntimeError, match=r"indices \[2\].*PrimalInfeasible"):
            _check_backward_statuses([1, 2, 2], dx)

    def test_unused_failure_passes(self):
        _check_backward_statuses([1, 9], np.array([[1.0], [0.0]]), None, np.zeros((2, 0)))

    def test_leading_vmap_axes_reduce(self):
        dx = np.zeros((4, 2, 3))
        dx[3, 1, 0] = 1.0
        with pytest.raises(RuntimeError, match=r"indices \[1\]"):
            _check_backward_statuses([1, 3], dx)

    @pytest.mark.parametrize(
        "status",
        [
            SolverStatus.Solved,
            SolverStatus.AlmostSolved,
            SolverStatus.MaxIterations,
            SolverStatus.MaxTime,
            SolverStatus.CallbackTerminated,
        ],
    )
    def test_differentiable_statuses(self, status):
        _check_backward_statuses(int(status), np.ones(2))

    @pytest.mark.parametrize(
        "status",
        [
            SolverStatus.Unsolved,
            SolverStatus.PrimalInfeasible,
            SolverStatus.DualInfeasible,
            SolverStatus.AlmostPrimalInfeasible,
            SolverStatus.AlmostDualInfeasible,
            SolverStatus.NumericalError,
            SolverStatus.InsufficientProgress,
        ],
    )
    def test_no_solution_statuses(self, status):
        with pytest.raises(RuntimeError, match=status.name):
            _check_backward_statuses(int(status), np.ones(2))


@pytest.mark.parametrize("solver_type", SOLVERS)
class TestNumpy:
    def test_single_infeasible_raises(self, solver_type):
        from scipy import sparse

        P = sparse.csr_matrix((P_VALS, P_CI, P_RO), shape=(N, N))
        A = sparse.csr_matrix((A_VALS, A_CI, A_RO), shape=(M, N))
        solver = moreau.Solver(P, Q, A, B_INFEAS, CONES, settings=_settings(solver_type))
        solver.solve()
        assert solver.info.status == SolverStatus.PrimalInfeasible
        with pytest.raises(RuntimeError, match=MATCH):
            solver.backward(np.ones(N))

    def _batched(self, solver_type):
        solver = moreau.CompiledSolver(
            N,
            M,
            P_RO,
            P_CI,
            A_RO,
            A_CI,
            CONES,
            _settings(solver_type, batch_size=2),
        )
        solver.setup(P_VALS, A_VALS)
        solver.solve(qs=np.stack([Q, Q]), bs=np.stack([B_FEAS, B_INFEAS]))
        assert solver.info.status[0] == SolverStatus.Solved
        assert solver.info.status[1] == SolverStatus.PrimalInfeasible
        return solver

    def test_batched_used_failure_raises(self, solver_type):
        solver = self._batched(solver_type)
        with pytest.raises(RuntimeError, match=r"indices \[1\]"):
            solver.backward(np.ones((2, N)))

    def test_batched_masked_failure_passes(self, solver_type):
        solver = self._batched(solver_type)
        grads = solver.backward(np.array([[1.0], [0.0]]))
        # x* = 1 is pinned by x >= 1, so dx*/db[0] = -1.
        np.testing.assert_allclose(grads["db"][0], [-1.0, 0.0], atol=1e-6)

    def test_batched_failure_used_through_dz_raises(self, solver_type):
        solver = self._batched(solver_type)
        dz = np.zeros((2, M))
        dz[1, 0] = 1.0
        with pytest.raises(RuntimeError, match=r"indices \[1\]"):
            solver.backward(np.zeros((2, N)), dz=dz)


torch = pytest.importorskip("torch")


def _torch_problem(solver_type, batched):
    from moreau.torch import Solver

    solver = Solver(N, M, P_RO, P_CI, A_RO, A_CI, CONES, _settings(solver_type))
    P = torch.tensor(P_VALS, dtype=torch.float64)
    A = torch.tensor(A_VALS, dtype=torch.float64)
    if batched:
        # Per-problem P/A: batched active-set backward with shared P/A is a
        # separate bug, unrelated to the status check tested here.
        P, A = P.expand(2, -1).clone(), A.expand(2, -1).clone()
        q = torch.zeros(2, N, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(np.stack([B_FEAS, B_INFEAS]), dtype=torch.float64, requires_grad=True)
    else:
        q = torch.zeros(N, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(B_INFEAS, dtype=torch.float64, requires_grad=True)
    return solver, P, A, q, b


@pytest.mark.parametrize("solver_type", SOLVERS)
class TestTorch:
    def test_single_infeasible_raises(self, solver_type):
        solver, P, A, q, b = _torch_problem(solver_type, batched=False)
        x = solver.solve(P, A, q, b).x
        with pytest.raises(RuntimeError, match=MATCH):
            x.sum().backward()

    def test_batched_used_failure_raises(self, solver_type):
        solver, P, A, q, b = _torch_problem(solver_type, batched=True)
        x = solver.solve(P, A, q, b).x
        with pytest.raises(RuntimeError, match=r"indices \[1\]"):
            x.sum().backward()

    def test_batched_masked_failure_passes(self, solver_type):
        solver, P, A, q, b = _torch_problem(solver_type, batched=True)
        x = solver.solve(P, A, q, b).x
        solved = torch.tensor(
            [s == SolverStatus.Solved for s in solver.info.status], dtype=torch.bool
        )
        x[solved].sum().backward()
        torch.testing.assert_close(
            b.grad, torch.tensor([[-1.0, 0.0], [0.0, 0.0]], dtype=torch.float64), atol=1e-6, rtol=0
        )

    def test_jacrev_through_failure_raises(self, solver_type):
        solver, P, A, q, b = _torch_problem(solver_type, batched=True)
        with pytest.raises(RuntimeError, match=r"indices \[1\]"):
            torch.func.jacrev(lambda b_: solver.solve(P, A, q.detach(), b_).x)(b.detach())

    def test_compiled_used_failure_raises(self, solver_type):
        solver, P, A, q, b = _torch_problem(solver_type, batched=True)
        loss = torch.compile(
            lambda b_: solver.solve(P, A, q.detach(), b_).x.sum(), backend="eager", fullgraph=True
        )
        torch._dynamo.reset()
        with pytest.raises(RuntimeError, match=r"indices \[1\]"):
            loss(b).backward()
        torch._dynamo.reset()

    def test_compiled_masked_failure_passes(self, solver_type):
        solver, P, A, q, b = _torch_problem(solver_type, batched=True)
        loss = torch.compile(
            lambda b_: solver.solve(P, A, q.detach(), b_).x[0].sum(),
            backend="eager",
            fullgraph=True,
        )
        torch._dynamo.reset()
        loss(b).backward()
        torch._dynamo.reset()
        torch.testing.assert_close(
            b.grad, torch.tensor([[-1.0, 0.0], [0.0, 0.0]], dtype=torch.float64), atol=1e-6, rtol=0
        )

    def test_manual_backward_raises(self, solver_type):
        solver, P, A, q, b = _torch_problem(solver_type, batched=True)
        with torch.no_grad():
            solver.solve(P, A, q, b)
        with pytest.raises(RuntimeError, match=r"indices \[1\]"):
            solver.backward(torch.ones(2, N, dtype=torch.float64))


@pytest.fixture
def jax_x64():
    jax = pytest.importorskip("jax")
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", prev)


@pytest.mark.parametrize("solver_type", SOLVERS)
class TestJax:
    def _solver(self, solver_type):
        from moreau.jax import Solver

        return Solver(N, M, P_RO, P_CI, A_RO, A_CI, CONES, _settings(solver_type))

    def test_single_infeasible_raises(self, jax_x64, solver_type):
        jnp = jax_x64.numpy
        solver = self._solver(solver_type)
        P, A, q = jnp.asarray(P_VALS), jnp.asarray(A_VALS), jnp.asarray(Q)
        with pytest.raises(Exception, match=MATCH):
            jax_x64.grad(lambda b: solver.solve(P, A, q, b).x.sum())(jnp.asarray(B_INFEAS))

    def test_vmap_used_failure_raises(self, jax_x64, solver_type):
        jnp = jax_x64.numpy
        solver = self._solver(solver_type)
        solver.setup(jnp.asarray(P_VALS), jnp.asarray(A_VALS))
        qs = jnp.zeros((2, N))
        bs = jnp.asarray(np.stack([B_FEAS, B_INFEAS]))
        with pytest.raises(Exception, match=MATCH):
            jax_x64.grad(lambda b: jax_x64.vmap(solver.solve)(qs, b).x.sum())(bs)

    def test_vmap_masked_failure_passes(self, jax_x64, solver_type):
        jnp = jax_x64.numpy
        solver = self._solver(solver_type)
        solver.setup(jnp.asarray(P_VALS), jnp.asarray(A_VALS))
        qs = jnp.zeros((2, N))
        bs = jnp.asarray(np.stack([B_FEAS, B_INFEAS]))
        db = jax_x64.grad(lambda b: jax_x64.vmap(solver.solve)(qs, b).x[0].sum())(bs)
        np.testing.assert_allclose(np.asarray(db), [[-1.0, 0.0], [0.0, 0.0]], atol=1e-6)
