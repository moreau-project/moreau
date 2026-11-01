"""
Copyright, the Moreau authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

JAX active-set backward differentiates each solve with that solve's own state (#57).
"""

import numpy as np
import pytest

import moreau

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
from moreau.jax import Solver  # noqa: E402

pytestmark = pytest.mark.jax

# min 1/2||x||^2 + q'x  s.t.  x >= 0   (A = -I, b = 0)
#   q1 = (-1, 1): x* = (1, 0), d(sum x)/dq1 = (-1, 0)
#   q2 = (1, -1): x* = (0, 1), d(sum x)/dq2 = (0, -1)
# Arrays are built inside the x64 fixture so they are float64 even when this
# file runs on its own.
P = A = B = Q1 = Q2 = None


@pytest.fixture(autouse=True)
def x64():
    global P, A, B, Q1, Q2
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    P, A, B = jnp.ones(2), -jnp.ones(2), jnp.zeros(2)
    Q1, Q2 = jnp.array([-1.0, 1.0]), jnp.array([1.0, -1.0])
    yield
    jax.config.update("jax_enable_x64", previous)


def _solver(kind):
    return Solver(
        2,
        2,
        np.array([0, 1, 2]),
        np.array([0, 1]),
        np.array([0, 1, 2]),
        np.array([0, 1]),
        moreau.Cones(num_nonneg_cones=2),
        settings=moreau.Settings(solver=kind, device="cpu"),
    )


SOLVERS = ["active_set", "ipm"]


@pytest.mark.parametrize("kind", SOLVERS)
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_two_solves_in_one_grad(kind, jit):
    solver = _solver(kind)

    def loss(a, c):
        return solver.solve(P, A, a, B).x.sum() + solver.solve(P, A, c, B).x.sum()

    grad = jax.grad(loss, argnums=(0, 1))
    g1, g2 = (jax.jit(grad) if jit else grad)(Q1, Q2)
    np.testing.assert_allclose(g1, [-1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(g2, [0.0, -1.0], atol=1e-6)


@pytest.mark.parametrize("kind", SOLVERS)
def test_rollout_matches_finite_differences(kind):
    solver = _solver(kind)

    def rollout(q0):
        q, total = q0, 0.0
        for _ in range(3):
            x = solver.solve(P, A, q, B).x
            total = total + x.sum()
            q = -q + 0.1 * x
        return total

    h = 1e-6
    fd = np.array(
        [(rollout(Q1.at[i].add(h)) - rollout(Q1.at[i].add(-h))) / (2 * h) for i in range(2)]
    )
    np.testing.assert_allclose(jax.grad(rollout)(Q1), fd, atol=1e-5)


@pytest.mark.parametrize("kind", SOLVERS)
def test_jacrev(kind):
    solver = _solver(kind)
    jacobian = jax.jacrev(lambda q: solver.solve(P, A, q, B).x)(Q1)
    np.testing.assert_allclose(jacobian, [[-1.0, 0.0], [0.0, 0.0]], atol=1e-6)


@pytest.mark.parametrize("kind", SOLVERS)
def test_vmap_grad_of_two_batched_solves(kind):
    solver = _solver(kind)

    def loss(q):
        return solver.solve(P, A, q, B).x.sum() + solver.solve(P, A, -q, B).x.sum()

    grads = jax.vmap(jax.grad(loss))(jnp.stack([Q1, Q2]))
    np.testing.assert_allclose(grads, [[-1.0, 1.0], [1.0, -1.0]], atol=1e-6)


@pytest.mark.parametrize("kind", SOLVERS)
def test_P_and_A_grads_of_two_solves(kind):
    # With b = -1/2 (x >= 1/2), the constraint on the coordinate q pushes up is
    # active: d(sum x)/dP = q_i/P_i^2 on free coordinates, d(sum x)/dA = -b_i/A_i^2
    # on active ones.
    solver = _solver(kind)
    b = -0.5 * jnp.ones(2)

    def loss(p1, a1, p2, a2):
        return solver.solve(p1, a1, Q1, b).x.sum() + solver.solve(p2, a2, Q2, b).x.sum()

    dp1, da1, dp2, da2 = jax.grad(loss, argnums=(0, 1, 2, 3))(P, A, P, A)
    np.testing.assert_allclose(dp1, [-1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(da1, [0.0, 0.5], atol=1e-6)
    np.testing.assert_allclose(dp2, [0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(da2, [0.5, 0.0], atol=1e-6)
