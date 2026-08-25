"""Tests for JAX integration.

Tests cover:
- Forward solve correctness
- vmap batching support
- Gradient correctness via check_grads_finite_diff
- custom_vjp works correctly

Tests run on both CPU and CUDA backends (CUDA when available).
"""

import pytest
import numpy as np

jax = pytest.importorskip("jax")

# Enable 64-bit mode for JAX (required for the solver)
jax.config.update("jax_enable_x64", True)

jnp = pytest.importorskip("jax.numpy")

import moreau
from moreau.jax import Solver
from moreau._backend import jax_available

pytestmark = pytest.mark.jax


@pytest.fixture(params=["cpu", "cuda"])
def jax_device(request):
    """Parametrized fixture for JAX device testing."""
    device = request.param
    if device == "cuda" and not jax_available("cuda"):
        pytest.skip("CUDA JAX not available")
    return device


def check_grads_finite_diff(fn, args, eps=1e-5, atol=1e-3, rtol=1e-3, order=1, modes=None):
    """Check gradients against finite differences.

    A simple implementation that doesn't require absl.testing.

    Args:
        fn: Function to check gradients of
        args: Tuple of arguments to pass to fn
        eps: Perturbation size for finite differences
        atol: Absolute tolerance
        rtol: Relative tolerance
        order: Ignored (for compatibility with jax.test_util.check_grads)
        modes: Ignored (for compatibility with jax.test_util.check_grads)
    """
    # Get analytical gradient
    grad_fn = jax.grad(fn)
    grad_analytical = grad_fn(*args)

    # Compute numerical gradient
    x = args[0]
    grad_numerical = jnp.zeros_like(x)

    for i in range(x.size):
        x_flat = x.ravel()
        x_plus = x_flat.at[i].set(x_flat[i] + eps).reshape(x.shape)
        x_minus = x_flat.at[i].set(x_flat[i] - eps).reshape(x.shape)

        f_plus = fn(x_plus)
        f_minus = fn(x_minus)

        grad_numerical = (
            grad_numerical.ravel().at[i].set((f_plus - f_minus) / (2 * eps)).reshape(x.shape)
        )

    # Check match
    abs_diff = jnp.abs(grad_analytical - grad_numerical)
    rel_diff = abs_diff / (jnp.abs(grad_numerical) + 1e-8)

    if not (jnp.all(abs_diff < atol) or jnp.all(rel_diff < rtol)):
        raise AssertionError(
            f"Gradient mismatch:\n"
            f"  analytical={grad_analytical}\n"
            f"  numerical={grad_numerical}\n"
            f"  abs_diff={abs_diff}\n"
            f"  rel_diff={rel_diff}"
        )


def make_simple_qp():
    """Create a simple QP problem structure.

    Problem: min (1/2)x'Px + q'x  s.t. x1 + x2 = b (equality)
    P = diag([2, 2]), A = [1, 1]
    """
    n, m = 2, 1

    P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    P_col_indices = np.array([0, 1], dtype=np.int64)

    A_row_offsets = np.array([0, 2], dtype=np.int64)
    A_col_indices = np.array([0, 1], dtype=np.int64)

    cones = moreau.Cones(num_zero_cones=1)

    return n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones


def make_inequality_qp():
    """Create QP with inequality constraints.

    Problem: min (1/2)x'Px + q'x  s.t. x >= 0
    P = diag([2, 2]), A = -I (for x >= 0)
    """
    n, m = 2, 2

    P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    P_col_indices = np.array([0, 1], dtype=np.int64)

    A_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    A_col_indices = np.array([0, 1], dtype=np.int64)

    cones = moreau.Cones(num_nonneg_cones=2)

    return n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones


def make_soc_problem():
    """Create a problem with a second-order cone constraint (dim=5).

    Problem: min (1/2)x'Px + q'x  s.t.  -x in SOC(5)
    P = diag([2]*5), A = -I(5)
    """
    n, m = 5, 5

    P_row_offsets = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    P_col_indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    A_row_offsets = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    A_col_indices = np.array([0, 1, 2, 3, 4], dtype=np.int64)

    cones = moreau.Cones(so_cone_dims=[5])

    return n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones


class TestJaxSOC:
    """Test JAX integration with second-order cone (dim=5)."""

    def test_soc_forward(self, jax_device):
        """Test solving a SOC problem."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_soc_problem()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
        A_data = jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        q = jnp.array([-3.0, -0.5, -0.3, -0.2, -0.1])
        b = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])

        result = solve(P_data, A_data, q, b)
        info = s.info

        # Solution should satisfy SOC constraint: x[0] >= ||x[1:]||
        assert result.x[0] >= jnp.linalg.norm(result.x[1:]) - 1e-5
        assert not jnp.any(jnp.isnan(result.x))

    def test_soc_grad_q(self, jax_device):
        """Test gradient w.r.t. q for SOC problem."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_soc_problem()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
        A_data = jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        b = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])

        def loss_fn(q):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        q = jnp.array([-3.0, -0.5, -0.3, -0.2, -0.1])
        grad_q = jax.grad(loss_fn)(q)

        assert not jnp.any(jnp.isnan(grad_q))
        assert grad_q.shape == q.shape

    def test_soc_vmap(self, jax_device):
        """Test vmap batching with SOC problem."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_soc_problem()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        batch_size = 3
        P_data = jnp.tile(jnp.array([2.0, 2.0, 2.0, 2.0, 2.0]), (batch_size, 1))
        A_data = jnp.tile(jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0]), (batch_size, 1))
        q = jnp.array(
            [
                [-3.0, -0.5, -0.3, -0.2, -0.1],
                [-2.5, -0.4, -0.6, -0.1, -0.3],
                [-4.0, -0.2, -0.1, -0.3, -0.2],
            ]
        )
        b = jnp.tile(jnp.array([0.0, 0.0, 0.0, 0.0, 0.0]), (batch_size, 1))

        solve_batch = jax.vmap(solve)
        result_batch = solve_batch(P_data, A_data, q, b)
        info_batch = s.info

        assert result_batch.x.shape == (batch_size, n)
        assert not jnp.any(jnp.isnan(result_batch.x))

    def test_soc_check_grads(self, jax_device):
        """Test gradient correctness for SOC with finite differences."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_soc_problem()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
        A_data = jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        b = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])

        def loss_fn(q):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        q = jnp.array([-3.0, -0.5, -0.3, -0.2, -0.1])

        check_grads_finite_diff(
            loss_fn,
            (q,),
            order=1,
            modes=["rev"],
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_soc_grad_does_not_corrupt_later_forwards(self, jax_device):
        """Backward pass should not leave the solver in a stale state."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_soc_problem()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0, 2.0, 2.0, 2.0])
        A_data = jnp.array([-1.0, -1.0, -1.0, -1.0, -1.0])
        b = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        q = jnp.array([-3.0, -0.5, -0.3, -0.2, -0.1])

        def loss_fn(q):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        x_before = solve(P_data, A_data, q, b).x
        _ = jax.grad(loss_fn)(q)
        x_after = solve(P_data, A_data, q, b).x

        assert jnp.allclose(x_after, x_before, atol=1e-8)


class TestJaxForwardSolve:
    """Test forward solve correctness."""

    def test_simple_qp_solve(self, jax_device):
        """Test solving a simple QP."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        # Create solver function
        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        # Problem data
        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        # Solve (returns tuple of JaxSolution and JaxSolveInfo)
        result = solve(P_data, A_data, q, b)
        info = s.info

        # Check solution satisfies equality constraint
        assert jnp.allclose(result.x[0] + result.x[1], 1.0, atol=1e-5)

        # For equality constraint, s should be ~0
        assert jnp.allclose(result.s, 0.0, atol=1e-5)

    def test_inequality_qp_solve(self, jax_device):
        """Test solving QP with inequality constraints."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_inequality_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([-1.0, -1.0])
        q = jnp.array([-2.0, -3.0])  # Push solution positive
        b = jnp.array([0.0, 0.0])

        result = solve(P_data, A_data, q, b)
        info = s.info

        # Solution should be positive (x >= 0)
        assert jnp.all(result.x >= -1e-6)

        # Optimal: x* = -q / P_diag = [1.0, 1.5]
        assert jnp.allclose(result.x, jnp.array([1.0, 1.5]), atol=1e-4)

    @pytest.mark.cpu_only
    def test_agrees_with_cpu_solver(self):
        """Test that JAX wrapper agrees with direct CPU solver."""
        import moreau_cpu

        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        # JAX solve
        settings = moreau.Settings(device="cpu")
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        jax_result = solve(P_data, A_data, q, b)
        jax_info = s.info

        # Direct CPU solve
        cpu_solver = moreau_cpu.Solver(
            n, m, P_ro, P_ci, A_ro, A_ci, cones, moreau_cpu.DefaultSettings(), batch_size=1
        )
        cpu_solver.setup(np.array(P_data).reshape(1, -1), np.array(A_data).reshape(1, -1))
        cpu_result = cpu_solver.solve(np.array(q).reshape(1, -1), np.array(b).reshape(1, -1))

        x_cpu = cpu_result["x"].squeeze()
        z_cpu = cpu_result["z"].squeeze()
        s_cpu = cpu_result["s"].squeeze()

        assert jnp.allclose(jax_result.x, x_cpu, atol=1e-6)
        assert jnp.allclose(jax_result.z, z_cpu, atol=1e-6)
        assert jnp.allclose(jax_result.s, s_cpu, atol=1e-6)


class TestJaxVmap:
    """Test vmap batching support."""

    def test_vmap_solve(self, jax_device):
        """Test solving batch of problems with vmap."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        batch_size = 3

        # Batched inputs (vary q)
        P_data = jnp.tile(jnp.array([2.0, 2.0]), (batch_size, 1))
        A_data = jnp.tile(jnp.array([1.0, 1.0]), (batch_size, 1))
        q = jnp.array(
            [
                [2.0, 1.0],
                [3.0, 1.0],
                [1.0, 2.0],
            ]
        )
        b = jnp.tile(jnp.array([1.0]), (batch_size, 1))

        # vmap solve
        solve_batch = jax.vmap(solve)
        result_batch = solve_batch(P_data, A_data, q, b)
        info_batch = s.info

        # Check shapes
        assert result_batch.x.shape == (batch_size, n)
        assert result_batch.z.shape == (batch_size, m)
        assert result_batch.s.shape == (batch_size, m)

        # Each solution should satisfy constraint
        for i in range(batch_size):
            assert jnp.allclose(result_batch.x[i, 0] + result_batch.x[i, 1], 1.0, atol=1e-5)

    def test_vmap_agrees_with_loop(self, jax_device):
        """Test that vmap produces same results as sequential loop."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        batch_size = 4

        P_data = jnp.tile(jnp.array([2.0, 2.0]), (batch_size, 1))
        A_data = jnp.tile(jnp.array([1.0, 1.0]), (batch_size, 1))
        q = jnp.array(
            [
                [2.0, 1.0],
                [3.0, 0.5],
                [1.0, 2.0],
                [0.5, 0.5],
            ]
        )
        b = jnp.tile(jnp.array([1.0]), (batch_size, 1))

        # vmap solve
        solve_batch = jax.vmap(solve)
        result_vmap = solve_batch(P_data, A_data, q, b)
        info_vmap = s.info

        # Sequential solve
        x_seq = []
        z_seq = []
        s_seq = []
        for i in range(batch_size):
            result_i = solve(P_data[i], A_data[i], q[i], b[i])
            info_i = s.info
            x_seq.append(result_i.x)
            z_seq.append(result_i.z)
            s_seq.append(result_i.s)

        x_seq = jnp.stack(x_seq)
        z_seq = jnp.stack(z_seq)
        s_seq = jnp.stack(s_seq)

        assert jnp.allclose(result_vmap.x, x_seq, atol=1e-6)
        assert jnp.allclose(result_vmap.z, z_seq, atol=1e-6)
        assert jnp.allclose(result_vmap.s, s_seq, atol=1e-6)

    def test_vmap_varying_P(self, jax_device):
        """Test vmap with varying P matrix values."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        batch_size = 3

        # Vary P diagonal
        P_data = jnp.array(
            [
                [2.0, 2.0],
                [4.0, 4.0],
                [1.0, 1.0],
            ]
        )
        A_data = jnp.tile(jnp.array([1.0, 1.0]), (batch_size, 1))
        q = jnp.tile(jnp.array([2.0, 1.0]), (batch_size, 1))
        b = jnp.tile(jnp.array([1.0]), (batch_size, 1))

        solve_batch = jax.vmap(solve)
        result_batch = solve_batch(P_data, A_data, q, b)
        info_batch = s.info

        # Different P should give different solutions
        # (all satisfy constraint but have different optimal x)
        for i in range(batch_size):
            assert jnp.allclose(result_batch.x[i, 0] + result_batch.x[i, 1], 1.0, atol=1e-5)


class TestJaxGradients:
    """Test gradient correctness."""

    def test_grad_q(self, jax_device):
        """Test gradient w.r.t. q using jax.grad."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        b = jnp.array([1.0])

        def loss_fn(q):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        q = jnp.array([2.0, 1.0])
        grad_q = jax.grad(loss_fn)(q)

        # Gradient should not be zero or NaN
        assert not jnp.any(jnp.isnan(grad_q))
        # For this simple problem, gradient should be well-defined
        assert grad_q.shape == q.shape

    def test_grad_b(self, jax_device):
        """Test gradient w.r.t. b."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        q = jnp.array([2.0, 1.0])

        def loss_fn(b):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        b = jnp.array([1.0])
        grad_b = jax.grad(loss_fn)(b)

        assert not jnp.any(jnp.isnan(grad_b))
        assert grad_b.shape == b.shape

    def test_grad_P_values(self, jax_device):
        """Test gradient w.r.t. P matrix values."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        A_data = jnp.array([1.0, 1.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        def loss_fn(P_data):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        P_data = jnp.array([2.0, 2.0])
        grad_P = jax.grad(loss_fn)(P_data)

        assert not jnp.any(jnp.isnan(grad_P))
        assert grad_P.shape == P_data.shape

    def test_grad_A_values(self, jax_device):
        """Test gradient w.r.t. A matrix values."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        def loss_fn(A_data):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        A_data = jnp.array([1.0, 1.0])
        grad_A = jax.grad(loss_fn)(A_data)

        assert not jnp.any(jnp.isnan(grad_A))
        assert grad_A.shape == A_data.shape

    def test_check_grads_q(self, jax_device):
        """Test gradient correctness with check_grads_finite_diff."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        b = jnp.array([1.0])

        def loss_fn(q):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        q = jnp.array([2.0, 1.0])

        # Check gradients match finite differences
        check_grads_finite_diff(
            loss_fn,
            (q,),
            order=1,
            modes=["rev"],  # Only check reverse mode
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_check_grads_b(self, jax_device):
        """Test gradient correctness for b."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        q = jnp.array([2.0, 1.0])

        def loss_fn(b):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        b = jnp.array([1.0])

        check_grads_finite_diff(
            loss_fn,
            (b,),
            order=1,
            modes=["rev"],
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_check_grads_P(self, jax_device):
        """Test gradient correctness for P values."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        A_data = jnp.array([1.0, 1.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        def loss_fn(P_data):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        P_data = jnp.array([2.0, 2.0])

        check_grads_finite_diff(
            loss_fn,
            (P_data,),
            order=1,
            modes=["rev"],
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_check_grads_A(self, jax_device):
        """Test gradient correctness for A values."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        def loss_fn(A_data):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        A_data = jnp.array([1.0, 1.0])

        check_grads_finite_diff(
            loss_fn,
            (A_data,),
            order=1,
            modes=["rev"],
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )


class TestJaxGradientsInequality:
    """Test gradients for inequality constrained problems."""

    def test_grad_q_inequality(self, jax_device):
        """Test gradient w.r.t. q for inequality QP."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_inequality_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([-1.0, -1.0])
        b = jnp.array([0.0, 0.0])

        def loss_fn(q):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        # q chosen for interior solution
        q = jnp.array([-2.0, -3.0])

        check_grads_finite_diff(
            loss_fn,
            (q,),
            order=1,
            modes=["rev"],
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )


class TestJaxGradientsBatched:
    """Test gradients with vmap."""

    def test_vmap_grad(self, jax_device):
        """Test gradient computation with vmap."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        batch_size = 3

        P_data = jnp.tile(jnp.array([2.0, 2.0]), (batch_size, 1))
        A_data = jnp.tile(jnp.array([1.0, 1.0]), (batch_size, 1))
        b = jnp.tile(jnp.array([1.0]), (batch_size, 1))

        def loss_fn(q):
            solve_batch = jax.vmap(solve)
            result = solve_batch(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        q = jnp.array(
            [
                [2.0, 1.0],
                [3.0, 0.5],
                [1.0, 2.0],
            ]
        )

        grad_q = jax.grad(loss_fn)(q)

        assert not jnp.any(jnp.isnan(grad_q))
        assert grad_q.shape == q.shape

    def test_vmap_grad_batch_size_1(self, jax_device):
        """Regression test: vmap with batch_size=1 must return correct 2D shapes.

        When jax.vmap calls the backward pass with batch_size=1, the CPU callback
        returned 1D arrays but JAX expected 2D (1, n) arrays, causing a
        RuntimeError: Incorrect output shape.
        """
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        # batch_size=1 is the key trigger
        P_data = jnp.array([[2.0, 2.0]])
        A_data = jnp.array([[1.0, 1.0]])
        b = jnp.array([[1.0]])

        def loss_fn(q):
            solve_batch = jax.vmap(solve)
            result = solve_batch(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        q = jnp.array([[2.0, 1.0]])

        grad_q = jax.grad(loss_fn)(q)

        assert grad_q.shape == q.shape
        assert not jnp.any(jnp.isnan(grad_q))

    def test_grad_vmap_check_grads(self, jax_device):
        """Test batched gradient correctness."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        batch_size = 2

        P_data = jnp.tile(jnp.array([2.0, 2.0]), (batch_size, 1))
        A_data = jnp.tile(jnp.array([1.0, 1.0]), (batch_size, 1))
        b = jnp.tile(jnp.array([1.0]), (batch_size, 1))

        def loss_fn(q):
            solve_batch = jax.vmap(solve)
            result = solve_batch(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        q = jnp.array(
            [
                [2.0, 1.0],
                [3.0, 0.5],
            ]
        )

        check_grads_finite_diff(
            loss_fn,
            (q,),
            order=1,
            modes=["rev"],
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )


class TestJaxDualSlackGradients:
    """Test gradients through dual and slack variables."""

    def test_grad_through_z(self, jax_device):
        """Test gradient when loss depends on dual variable z."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        b = jnp.array([1.0])

        def loss_fn(q):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.z)

        q = jnp.array([2.0, 1.0])
        grad_q = jax.grad(loss_fn)(q)

        assert not jnp.any(jnp.isnan(grad_q))

    def test_grad_through_s(self, jax_device):
        """Test gradient when loss depends on slack variable s."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_inequality_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([-1.0, -1.0])
        b = jnp.array([0.0, 0.0])

        def loss_fn(q):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.s)

        q = jnp.array([-2.0, -3.0])
        grad_q = jax.grad(loss_fn)(q)

        assert not jnp.any(jnp.isnan(grad_q))

    def test_grad_combined_loss(self, jax_device):
        """Test gradient when loss combines x, z, s."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        b = jnp.array([1.0])

        def loss_fn(q):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x) + 0.5 * jnp.sum(result.z) + 0.1 * jnp.sum(result.s)

        q = jnp.array([2.0, 1.0])

        check_grads_finite_diff(
            loss_fn,
            (q,),
            order=1,
            modes=["rev"],
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )


class TestJaxJIT:
    """Test JIT compilation."""

    def test_jit_solve(self, jax_device):
        """Test that solve can be JIT compiled."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        # JIT compile
        solve_jit = jax.jit(solve)

        # Run twice (second should use cached compilation)
        result1 = solve_jit(P_data, A_data, q, b)
        info1 = s.info
        result2 = solve_jit(P_data, A_data, q, b)
        info2 = s.info

        assert jnp.allclose(result1.x, result2.x)
        assert jnp.allclose(result1.z, result2.z)
        assert jnp.allclose(result1.s, result2.s)

    def test_jit_vmap_solve(self, jax_device):
        """Test that vmap + jit works."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        batch_size = 3

        P_data = jnp.tile(jnp.array([2.0, 2.0]), (batch_size, 1))
        A_data = jnp.tile(jnp.array([1.0, 1.0]), (batch_size, 1))
        q = jnp.array(
            [
                [2.0, 1.0],
                [3.0, 0.5],
                [1.0, 2.0],
            ]
        )
        b = jnp.tile(jnp.array([1.0]), (batch_size, 1))

        solve_batch = jax.jit(jax.vmap(solve))
        result = solve_batch(P_data, A_data, q, b)
        info = s.info

        assert result.x.shape == (batch_size, n)

    def test_jit_grad(self, jax_device):
        """Test that JIT + grad works."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        b = jnp.array([1.0])

        @jax.jit
        def grad_loss(q):
            def loss_fn(q_):
                result = solve(P_data, A_data, q_, b)
                info = s.info
                return jnp.sum(result.x)

            return jax.grad(loss_fn)(q)

        q = jnp.array([2.0, 1.0])
        grad_q = grad_loss(q)

        assert not jnp.any(jnp.isnan(grad_q))
        assert grad_q.shape == q.shape


class TestJaxCaching:
    """Test P/A value caching for performance."""

    def test_cache_invalidation(self, jax_device):
        """Test that changing P/A values produces different solutions."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        A_data = jnp.array([1.0, 1.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        # First solve with P diagonal = [2, 2]
        P_data_1 = jnp.array([2.0, 2.0])
        result1 = solve(P_data_1, A_data, q, b)
        info1 = s.info

        # Second solve with different P diagonal = [10, 10]
        P_data_2 = jnp.array([10.0, 10.0])
        result2 = solve(P_data_2, A_data, q, b)
        info2 = s.info

        # Solutions should be different (cache should be invalidated)
        assert not jnp.allclose(
            result1.x, result2.x, atol=1e-6
        ), "Different P values should produce different solutions"

        # Both should satisfy constraint x1 + x2 = 1
        assert jnp.allclose(result1.x[0] + result1.x[1], 1.0, atol=1e-5)
        assert jnp.allclose(result2.x[0] + result2.x[1], 1.0, atol=1e-5)

    def test_cache_hit(self, jax_device):
        """Test that same P/A values use cache (implicit - solutions match)."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        # Multiple solves with same P/A
        result1 = solve(P_data, A_data, q, b)
        info1 = s.info
        result2 = solve(P_data, A_data, q, b)
        info2 = s.info
        result3 = solve(P_data, A_data, q, b)
        info3 = s.info

        # All should match
        assert jnp.allclose(result1.x, result2.x, atol=1e-10)
        assert jnp.allclose(result2.x, result3.x, atol=1e-10)


class TestJaxSetupSolve:
    """Test two-step setup()/solve() API."""

    def test_setup_solve_basic(self, jax_device):
        """Test basic two-step workflow."""
        from moreau.jax import Solver

        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        # Two-step API
        s.setup(P_data, A_data)
        result = s.solve(q, b)

        # Check solution satisfies constraint
        assert jnp.allclose(result.x[0] + result.x[1], 1.0, atol=1e-5)

    def test_setup_solve_matches_full_signature(self, jax_device):
        """Test that 2-arg solve matches 4-arg solve."""
        from moreau.jax import Solver

        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        # Full signature
        result_full = s.solve(P_data, A_data, q, b)

        # Two-step
        s.setup(P_data, A_data)
        result_setup = s.solve(q, b)

        assert jnp.allclose(result_full.x, result_setup.x, atol=1e-10)
        assert jnp.allclose(result_full.z, result_setup.z, atol=1e-10)
        assert jnp.allclose(result_full.s, result_setup.s, atol=1e-10)

    def test_setup_solve_grad_q(self, jax_device):
        """Test gradients w.r.t. q work with 4-arg solve."""
        from moreau.jax import Solver

        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        method = "qdldl" if jax_device == "cpu" else "cudss"
        ipm = moreau.IPMSettings(direct_solve_method=method)
        settings = moreau.Settings(device=jax_device, ipm_settings=ipm)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])
        b = jnp.array([1.0])

        s.setup(P_data, A_data)

        def loss_fn(q):
            result = s.solve(q, b)
            return jnp.sum(result.x)

        q = jnp.array([2.0, 1.0])
        grad_q = jax.grad(loss_fn)(q)

        assert not jnp.any(jnp.isnan(grad_q))
        assert grad_q.shape == q.shape

    def test_setup_solve_vmap(self, jax_device):
        """Test user can vmap the 4-arg solve."""
        from moreau.jax import Solver

        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        method = "qdldl" if jax_device == "cpu" else "cudss"
        ipm = moreau.IPMSettings(direct_solve_method=method)
        settings = moreau.Settings(device=jax_device, ipm_settings=ipm)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)

        P_data = jnp.array([2.0, 2.0])
        A_data = jnp.array([1.0, 1.0])

        batch_size = 3
        q = jnp.array(
            [
                [2.0, 1.0],
                [3.0, 1.0],
                [1.0, 2.0],
            ]
        )
        b = jnp.tile(jnp.array([1.0]), (batch_size, 1))

        s.setup(P_data, A_data)

        # vmap over q, b
        batched_solve = jax.vmap(s.solve)
        results = batched_solve(q, b)

        assert results.x.shape == (batch_size, n)

        # Each solution should satisfy constraint
        for i in range(batch_size):
            assert jnp.allclose(results.x[i, 0] + results.x[i, 1], 1.0, atol=1e-5)

    def test_setup_twice(self, jax_device):
        """Test calling setup() again with new P/A."""
        from moreau.jax import Solver

        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)

        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        # First setup
        P_data_1 = jnp.array([2.0, 2.0])
        A_data_1 = jnp.array([1.0, 1.0])
        s.setup(P_data_1, A_data_1)
        result1 = s.solve(q, b)

        # Second setup with different P
        P_data_2 = jnp.array([10.0, 10.0])
        A_data_2 = jnp.array([1.0, 1.0])
        s.setup(P_data_2, A_data_2)
        result2 = s.solve(q, b)

        # Solutions should be different
        assert not jnp.allclose(result1.x, result2.x, atol=1e-6)

        # Both should satisfy constraint
        assert jnp.allclose(result1.x[0] + result1.x[1], 1.0, atol=1e-5)
        assert jnp.allclose(result2.x[0] + result2.x[1], 1.0, atol=1e-5)

    def test_solve_without_setup_raises(self, jax_device):
        """Test that 4-arg solve without setup() raises error."""
        from moreau.jax import Solver

        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)

        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        with pytest.raises(RuntimeError, match="setup\\(\\) must be called"):
            s.solve(q, b)

    def test_solve_wrong_args_raises(self, jax_device):
        """Test that wrong number of args raises TypeError."""
        from moreau.jax import Solver

        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)

        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        with pytest.raises(TypeError, match="solve\\(\\) takes 2 or 4 arguments"):
            s.solve(q)  # Only 1 arg

    def test_full_signature_after_setup(self, jax_device):
        """Test 4-arg solve still works after setup() called."""
        from moreau.jax import Solver

        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_simple_qp()

        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)

        P_data_setup = jnp.array([2.0, 2.0])
        A_data_setup = jnp.array([1.0, 1.0])
        s.setup(P_data_setup, A_data_setup)

        # Use 4-arg with DIFFERENT P/A
        P_data_full = jnp.array([10.0, 10.0])
        A_data_full = jnp.array([1.0, 1.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0])

        # 4-arg should use passed P/A, not stored
        result_full = s.solve(P_data_full, A_data_full, q, b)

        # 2-arg should use stored P/A
        result_setup = s.solve(q, b)

        # Should be different since P differs
        assert not jnp.allclose(result_full.x, result_setup.x, atol=1e-6)


# =============================================================================
# Cone type coverage — ensure every cone type works through the JAX FFI path
# =============================================================================


def make_power_cone_problem():
    """Power cone: max z  s.t. z^{2/3} * 1^{1/3} >= |y|,  y >= 4.

    x = (y, z), n=2, m=4
    Cones: nonneg(1) | PowerCone(alpha=2/3)
    Optimal: y=4, z=8.
    """
    n, m = 2, 4
    P_row_offsets = np.array([0, 0, 0], dtype=np.int64)
    P_col_indices = np.array([], dtype=np.int64)
    A_row_offsets = np.array([0, 1, 2, 2, 3], dtype=np.int64)
    A_col_indices = np.array([0, 1, 0], dtype=np.int64)
    cones = moreau.Cones(num_nonneg_cones=1, power_alphas=[2 / 3])
    return n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones


def make_exp_cone_problem():
    """Exponential cone: min x  s.t. exp(x) <= 1, i.e. x <= 0.

    x = (x_var,), n=1, m=3
    ExpCone: (x_var, 1, y) with y >= exp(x_var), we want y <= 1.
    Slack s = b - Ax, so s = [x_var, 1, -x_var] after mapping.
    ExpCone: s[2] >= s[1]*exp(s[0]/s[1]) => -x_var >= 1*exp(x_var/1) doesn't
    work directly. Instead use the standard formulation:

    min t  s.t. exp(x) <= t, x <= 0
    ExpCone K_exp = {(x,y,z) : z >= y*exp(x/y), y>0}
    Constraint: (x, 1, t) in K_exp  =>  t >= exp(x)
    Also x <= 0 via nonneg cone on -x.

    x_vars = (x, t), n=2, m=4
    Cones: nonneg(1) | exp(3)
    Row 0 (nonneg): -x >= 0  =>  A[0] = [-1, 0], b[0] = 0
    Row 1 (exp, pos 0): x  =>  A[1] = [-1, 0], b[1] = 0  => s[0] = x
    Row 2 (exp, pos 1): 1  =>  zero row, b[2] = 1  => s[1] = 1
    Row 3 (exp, pos 2): t  =>  A[3] = [0, -1], b[3] = 0  => s[2] = t
    Objective: min t  =>  q = [0, 1]
    Optimal: x=0, t=1.
    """
    n, m = 2, 4
    P_row_offsets = np.array([0, 0, 0], dtype=np.int64)
    P_col_indices = np.array([], dtype=np.int64)
    A_row_offsets = np.array([0, 1, 2, 2, 3], dtype=np.int64)
    A_col_indices = np.array([0, 0, 1], dtype=np.int64)
    cones = moreau.Cones(num_nonneg_cones=1, num_exp_cones=1)
    return n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones


def make_gen_power_cone_problem():
    """Generalized power cone: min z  s.t. p^0.5 * 1^0.5 >= |z|, p = 4.

    Same structure as the power cone problem but using GenPowerCone.
    GenPowerCone with alphas=[0.5, 0.5], dim2=1 (total dim=3).
    x = (z,), n=1, m=4
    Cones: nonneg(1) | GenPowerCone(alphas=[0.5,0.5], dim2=1)
    Row 0 (nonneg): -b[0] + 0 >= 0, b[0] = 0 (dummy, always satisfied)
    Row 1 (gpc[0]): s[0] = b[1] = 4   (p component, constant)
    Row 2 (gpc[1]): s[1] = b[2] = 1   (constant)
    Row 3 (gpc[2]): s[2] = -z         (w component)
    Optimal: z = -2 (min z s.t. |z| <= sqrt(4)*sqrt(1) = 2).
    """
    n, m = 1, 4
    P_row_offsets = np.array([0, 0], dtype=np.int64)
    P_col_indices = np.array([], dtype=np.int64)
    A_row_offsets = np.array([0, 0, 0, 0, 1], dtype=np.int64)
    A_col_indices = np.array([0], dtype=np.int64)
    cones = moreau.Cones(num_nonneg_cones=1, gen_power_cone_params=[([0.5, 0.5], 1)])
    return n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones


def make_mixed_cone_problem():
    """Problem with zero + nonneg + power cones.

    min z  s.t.
      y = 4                            (zero cone, 1 row)
      z^{2/3} * 1^{1/3} >= |y|        (power cone, 3 rows)

    Reuses the power cone problem but with an equality instead of inequality.
    x = (y, z), n=2, m=4
    Cones: zero(1) | PowerCone(alpha=2/3)
    Row 0 (zero): y = 4  (via Ax + s = b, s=0: -y + b = 0, b = -4)
    Row 1 (pow[0]): z
    Row 2 (pow[1]): 1  (zero row in A)
    Row 3 (pow[2]): y
    Optimal: y=4, z=8.
    """
    n, m = 2, 4
    P_row_offsets = np.array([0, 0, 0], dtype=np.int64)
    P_col_indices = np.array([], dtype=np.int64)
    A_row_offsets = np.array([0, 1, 2, 2, 3], dtype=np.int64)
    A_col_indices = np.array([0, 1, 0], dtype=np.int64)
    cones = moreau.Cones(num_zero_cones=1, power_alphas=[2 / 3])
    return n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones


class TestJaxPowerCone:
    """Test JAX integration with power cones."""

    def test_power_forward(self, jax_device):
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_power_cone_problem()
        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([])
        A_data = jnp.array([-1.0, -1.0, -1.0])
        q = jnp.array([0.0, 1.0])
        b = jnp.array([-4.0, 0.0, 1.0, 0.0])

        result = solve(P_data, A_data, q, b)
        info = s.info
        np.testing.assert_allclose(result.x[0], 4.0, atol=1e-3)
        np.testing.assert_allclose(result.x[1], 8.0, atol=1e-3)

    def test_power_grad_q(self, jax_device):
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_power_cone_problem()
        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        A_data = jnp.array([-1.0, -1.0, -1.0])
        b = jnp.array([-4.0, 0.0, 1.0, 0.0])

        def loss_fn(q):
            result = solve(jnp.array([]), A_data, q, b)
            _ = s.info
            return jnp.sum(result.x)

        q = jnp.array([0.0, 1.0])
        grad_q = jax.grad(loss_fn)(q)
        assert not jnp.any(jnp.isnan(grad_q))
        assert grad_q.shape == q.shape

    def test_power_vmap(self, jax_device):
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_power_cone_problem()
        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        batch = 3
        A_data = jnp.tile(jnp.array([-1.0, -1.0, -1.0]), (batch, 1))
        q = jnp.tile(jnp.array([0.0, 1.0]), (batch, 1))
        b = jnp.stack(
            [
                jnp.array([-4.0, 0.0, 1.0, 0.0]),
                jnp.array([-2.0, 0.0, 1.0, 0.0]),
                jnp.array([-6.0, 0.0, 1.0, 0.0]),
            ]
        )

        result = jax.vmap(solve, in_axes=(0, 0, 0, 0))(jnp.zeros((batch, 0)), A_data, q, b)
        info = s.info
        assert result.x.shape == (batch, n)
        # y=4 => z=8,  y=2 => z=2^(3/2)=2.83,  y=6 => z=6^(3/2)=14.7
        np.testing.assert_allclose(result.x[0, 0], 4.0, atol=1e-3)
        np.testing.assert_allclose(result.x[1, 0], 2.0, atol=1e-3)


class TestJaxExpCone:
    """Test JAX integration with exponential cones."""

    def test_exp_forward(self, jax_device):
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_exp_cone_problem()
        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([])
        A_data = jnp.array([-1.0, -1.0, -1.0])
        q = jnp.array([0.0, 1.0])
        b = jnp.array([0.0, 0.0, 1.0, 0.0])

        result = solve(P_data, A_data, q, b)
        info = s.info
        # x=0, t=exp(0)=1
        np.testing.assert_allclose(result.x[0], 0.0, atol=1e-3)
        np.testing.assert_allclose(result.x[1], 1.0, atol=1e-3)

    def test_exp_grad_q(self, jax_device):
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_exp_cone_problem()
        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        A_data = jnp.array([-1.0, -1.0, -1.0])
        b = jnp.array([0.0, 0.0, 1.0, 0.0])

        def loss_fn(q):
            result = solve(jnp.array([]), A_data, q, b)
            _ = s.info
            return jnp.sum(result.x)

        q = jnp.array([0.0, 1.0])
        grad_q = jax.grad(loss_fn)(q)
        assert not jnp.any(jnp.isnan(grad_q))


class TestJaxGenPowerCone:
    """Test JAX integration with generalized power cones."""

    def test_gen_power_forward(self, jax_device):
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_gen_power_cone_problem()
        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([])
        A_data = jnp.array([-1.0])
        q = jnp.array([1.0])  # min z
        b = jnp.array([0.0, 4.0, 1.0, 0.0])

        result = solve(P_data, A_data, q, b)
        info = s.info
        # |z| <= sqrt(4)*sqrt(1) = 2, min z => z = -2
        np.testing.assert_allclose(result.x[0], -2.0, atol=1e-2)

    def test_gen_power_grad_q(self, jax_device):
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_gen_power_cone_problem()
        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        A_data = jnp.array([-1.0])
        b = jnp.array([0.0, 4.0, 1.0, 0.0])

        def loss_fn(q):
            result = solve(jnp.array([]), A_data, q, b)
            _ = s.info
            return jnp.sum(result.x)

        q = jnp.array([1.0])
        grad_q = jax.grad(loss_fn)(q)
        assert not jnp.any(jnp.isnan(grad_q))


class TestJaxMixedCones:
    """Test JAX integration with multiple cone types in one problem."""

    def test_mixed_forward(self, jax_device):
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_mixed_cone_problem()
        settings = moreau.Settings(device=jax_device)
        s = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([])
        A_data = jnp.array([-1.0, -1.0, -1.0])
        q = jnp.array([0.0, 1.0])
        b = jnp.array([-4.0, 0.0, 1.0, 0.0])

        result = solve(P_data, A_data, q, b)
        info = s.info
        np.testing.assert_allclose(result.x[0], 4.0, atol=1e-3)
        np.testing.assert_allclose(result.x[1], 8.0, atol=1e-3)

    def test_mixed_cpu_gpu_parity(self):
        """Solutions from CPU and CUDA should agree for mixed-cone problems."""
        if not jax_available("cuda"):
            pytest.skip("CUDA not available")
        n, m, P_ro, P_ci, A_ro, A_ci, cones = make_power_cone_problem()

        s_cpu = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, moreau.Settings(device="cpu"))
        s_gpu = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, moreau.Settings(device="cuda"))

        P_data = jnp.array([])
        A_data = jnp.array([-1.0, -1.0, -1.0])
        q = jnp.array([0.0, 1.0])
        b = jnp.array([-4.0, 0.0, 1.0, 0.0])

        res_cpu = s_cpu.solve(P_data, A_data, q, b)
        res_gpu = s_gpu.solve(P_data, A_data, q, b)
        np.testing.assert_allclose(res_cpu.x, res_gpu.x, atol=1e-5)


class TestJaxCudaFallback:
    """Regression tests for #180: when device='cuda' is requested but no
    CUDA JAX backend is registered, the wrapper falls back to CPU. The
    fallback must update self._device and clear any cuda-only
    direct_solve_method (e.g. 'cudss') so the CPU backend doesn't reject
    the settings at solve time."""

    def test_cuda_fallback_updates_device_and_clears_cudss(self):
        if jax_available("cuda"):
            pytest.skip("Test requires no JAX CUDA backend")

        P_ro = np.array([0, 1, 2], dtype=np.int64)
        P_ci = np.array([0, 1], dtype=np.int64)
        A_ro = np.array([0, 2, 3, 4], dtype=np.int64)
        A_ci = np.array([0, 1, 0, 1], dtype=np.int64)
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        ipm = moreau.IPMSettings(direct_solve_method="cudss")

        with pytest.warns(UserWarning, match="falling back to CPU"):
            s = moreau.jax.Solver(
                2,
                3,
                P_ro,
                P_ci,
                A_ro,
                A_ci,
                cones,
                moreau.Settings(device="cuda", solver="ipm", ipm_settings=ipm),
            )

        assert s.device == "cpu", "device must reflect actual backend after fallback"
        # Must not leave 'cudss' in settings — the CPU pybind layer rejects it.
        assert s._settings.ipm_settings.direct_solve_method != "cudss"

        # And the solve actually works (no rejection inside the callback).
        P_data = jnp.array([1.0, 1.0])
        A_data = jnp.array([1.0, 1.0, 1.0, 1.0])
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0, 0.7, 0.7])
        result, _info = s._solve_raw(P_data, A_data, q, b)
        np.testing.assert_allclose(result.x.shape, (2,))


class TestJaxCpuRegistry:
    """Regression tests for the JAX CPU solver registry."""

    def test_registry_drops_collected_solvers(self):
        """JaxSolverCpu wrappers should be removed from the registry on GC."""
        import gc
        from moreau.jax import _cpu_impl

        P_ro = np.array([0, 1, 2], dtype=np.int64)
        P_ci = np.array([0, 1], dtype=np.int64)
        A_ro = np.array([0, 2, 3, 4], dtype=np.int64)
        A_ci = np.array([0, 1, 0, 1], dtype=np.int64)
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        before = len(_cpu_impl._SOLVER_REGISTRY)
        for _ in range(5):
            s = moreau.jax.Solver(
                2,
                3,
                P_ro,
                P_ci,
                A_ro,
                A_ci,
                cones,
                moreau.Settings(solver="ipm", device="cpu"),
            )
            del s
        gc.collect()
        assert len(_cpu_impl._SOLVER_REGISTRY) == before
