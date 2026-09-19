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

Exercise native FFI handlers, not just their deferred registration.
"""

import numpy as np
import pytest

import moreau

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = [pytest.mark.jax, pytest.mark.cuda]


@pytest.mark.parametrize("warm", [False, True], ids=["cold", "warm"])
@pytest.mark.parametrize("mode", ["eager", "jit", "vmap"])
def test_native_ffi_forward_backward(warm, mode):
    from moreau_cuda.jax import JaxSolverCuda, ffi_available

    # FFI requires float64 buffers. Restore the caller's setting after the test.
    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        assert jax.devices("cuda"), "JAX CUDA is unavailable"
        assert ffi_available(), "The native FFI extension must be present"
        solver = JaxSolverCuda(
            2,
            1,
            [0, 1, 2],
            [0, 1],
            [0, 2],
            [0, 1],
            moreau.Cones(num_zero_cones=1),
        )
        assert solver._use_ffi, "This test must not use the Python callback"
        P = jnp.array([2.0, 2.0])
        A = jnp.array([1.0, 1.0])
        b = jnp.array([1.0])
        q = jnp.array([-1.0, -0.5])
        if mode == "vmap":
            q = jnp.stack([q, q + jnp.array([0.1, -0.2])])

        def solve(q):
            if warm:
                solution, _ = solver.solve_warm(
                    P,
                    A,
                    q,
                    b,
                    jnp.array([0.5, 0.5]),
                    jnp.zeros(1),
                    jnp.zeros(1),
                    jnp.empty(0),
                )
            else:
                solution, _ = solver.solve(P, A, q, b)
            return solution.x

        # Equality-constrained diagonal QP: x = (1 + mean(q) - q) / 2.
        reference = (1.0 + q.mean(axis=-1, keepdims=True) - q) / 2.0
        forward = solve
        backward = jax.grad(lambda q: solve(q)[0])
        if mode == "jit":
            forward, backward = jax.jit(forward), jax.jit(backward)
        elif mode == "vmap":
            forward = jax.jit(jax.vmap(forward))
            backward = jax.jit(jax.vmap(backward))
        np.testing.assert_allclose(forward(q), reference, atol=1e-6, rtol=1e-6)
        expected_grad = np.broadcast_to([-0.25, 0.25], q.shape)
        np.testing.assert_allclose(backward(q), expected_grad, atol=1e-6, rtol=1e-6)
    finally:
        jax.config.update("jax_enable_x64", previous_x64)
