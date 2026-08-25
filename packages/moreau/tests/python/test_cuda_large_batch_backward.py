"""
Test for CUDA DiffKKT factorization failure on large batched MPC backward pass.

Reproduces a bug where moreau-cuda backward (DiffKKT factorization) fails when
GPU memory is under pressure — e.g., when JAX has preallocated most of the GPU
memory. The cuDSS factorization in the backward pass runs out of device memory
and raises "DiffKKT factorization failed".

Problem:
    Large MPC trajectory optimization QP:
    - 32 states, 16 controls, horizon 100
    - n=4832 variables, m=6432 constraints
    - batch_size=128
    - Backward pass requires cuDSS factorization of a 11264x11264 KKT system
      for each of 128 problems

The forward solve succeeds, but the backward pass fails with:
    RuntimeError: DiffKKT factorization failed

This test does NOT require JAX — it reproduces the issue by verifying the
backward pass works on CUDA at this problem scale. When GPU memory is limited,
this is the first thing that breaks.
"""

import numpy as np
import pytest
import moreau
from scipy import sparse


def _make_mpc_qp(n_states, n_controls, horizon, batch_size, seed=42):
    """Build a linear MPC problem as a conic QP.

    Variables: [x_0, u_0, x_1, u_1, ..., x_{T-1}, u_{T-1}, x_T]

    Objective:
        min  sum_t (x_t' Q x_t + u_t' R u_t) + x_T' Qf x_T

    Constraints:
        x_{t+1} = A x_t + B u_t   (dynamics, equality)
        x_0 = x0                   (initial condition, equality)
        -u_max <= u_t <= u_max     (control bounds, inequality)
    """
    rng = np.random.default_rng(seed)

    # Random stable linear system
    A_dyn = np.eye(n_states) + 0.1 * rng.standard_normal((n_states, n_states)) * 0.1
    B_dyn = rng.standard_normal((n_states, n_controls)) * 0.1

    n_vars = horizon * (n_states + n_controls) + n_states

    # Cost: P = diag(Q, R, Q, R, ..., Qf)
    P_diag = np.zeros(n_vars)
    for t in range(horizon):
        xs = t * (n_states + n_controls)
        P_diag[xs : xs + n_states] = 1.0  # Q = I
        P_diag[xs + n_states : xs + n_states + n_controls] = 0.1  # R = 0.1*I
    P_diag[horizon * (n_states + n_controls) :] = 10.0  # Qf = 10*I
    P = sparse.diags(P_diag, format="csr")

    # Constraints
    u_max = 1.0
    n_eq_dyn = n_states * horizon  # dynamics
    n_eq_ic = n_states  # initial condition
    n_eq = n_eq_dyn + n_eq_ic
    n_ineq = 2 * n_controls * horizon  # control bounds
    m = n_eq + n_ineq

    rows, cols, vals = [], [], []
    row_idx = 0

    # Dynamics: A x_t + B u_t - x_{t+1} = 0
    for t in range(horizon):
        x_t = t * (n_states + n_controls)
        u_t = x_t + n_states
        if t < horizon - 1:
            x_tp1 = (t + 1) * (n_states + n_controls)
        else:
            x_tp1 = horizon * (n_states + n_controls)

        for i in range(n_states):
            for j in range(n_states):
                if abs(A_dyn[i, j]) > 1e-10:
                    rows.append(row_idx + i)
                    cols.append(x_t + j)
                    vals.append(A_dyn[i, j])
            for j in range(n_controls):
                if abs(B_dyn[i, j]) > 1e-10:
                    rows.append(row_idx + i)
                    cols.append(u_t + j)
                    vals.append(B_dyn[i, j])
            rows.append(row_idx + i)
            cols.append(x_tp1 + i)
            vals.append(-1.0)
        row_idx += n_states

    # Initial condition: x_0 = x0
    for i in range(n_states):
        rows.append(row_idx + i)
        cols.append(i)
        vals.append(1.0)
    row_idx += n_states

    # Control bounds: -u <= u_max, u <= u_max
    for t in range(horizon):
        u_t = t * (n_states + n_controls) + n_states
        for j in range(n_controls):
            rows.append(row_idx)
            cols.append(u_t + j)
            vals.append(-1.0)
            row_idx += 1
        for j in range(n_controls):
            rows.append(row_idx)
            cols.append(u_t + j)
            vals.append(1.0)
            row_idx += 1

    A_mat = sparse.csr_matrix((vals, (rows, cols)), shape=(m, n_vars))

    # Batched RHS: different initial conditions per batch element
    rng_batch = np.random.default_rng(123)
    q_batch = np.zeros((batch_size, n_vars))
    b_batch = np.zeros((batch_size, m))
    for i in range(batch_size):
        x0 = rng_batch.standard_normal(n_states) * 0.5
        b_batch[i, n_eq_dyn : n_eq_dyn + n_eq_ic] = x0
        b_batch[i, n_eq:] = u_max

    cones = moreau.Cones(num_zero_cones=n_eq, num_nonneg_cones=n_ineq)
    return P, q_batch, A_mat, b_batch, cones, n_vars, m


@pytest.mark.cuda
class TestLargeBatchMPCBackward:
    """Test backward pass on large batched MPC problems on CUDA."""

    def test_mpc_backward_batch128_cuda(self):
        """Backward pass should succeed for batch=128 MPC on CUDA.

        This reproduces a DiffKKT factorization failure observed when running
        large batched backward passes on CUDA. The cuDSS factorization of the
        KKT system (11264x11264, 171296 nnz, batch=128) fails under GPU memory
        pressure.
        """
        if not moreau.device_available("cuda"):
            pytest.skip("CUDA not available")

        n_states, n_controls, horizon, batch_size = 32, 16, 100, 128

        P, q_batch, A, b_batch, cones, n, m = _make_mpc_qp(
            n_states, n_controls, horizon, batch_size
        )
        P_csr = P.tocsr()
        A_csr = A.tocsr()

        ipm = moreau.IPMSettings(direct_solve_method="cudss")
        settings = moreau.Settings(
            device="cuda",
            verbose=False,
            enable_grad=True,
            batch_size=batch_size,
            ipm_settings=ipm,
        )
        solver = moreau.CompiledSolver(
            n, m, P_csr.indptr, P_csr.indices, A_csr.indptr, A_csr.indices, cones, settings
        )
        solver.setup(
            np.tile(P_csr.data, (batch_size, 1)),
            np.tile(A_csr.data, (batch_size, 1)),
        )

        # Forward solve should always work
        result = solver.solve(q_batch, b_batch)
        assert result.x.shape == (batch_size, n)

        # Backward pass — this is what fails under memory pressure
        dx = np.ones((batch_size, n))
        dz = np.ones((batch_size, m))
        ds = np.ones((batch_size, m))
        grads = solver.backward(dx, dz, ds)

        # Verify gradients have correct shapes
        assert grads["dq"].shape == (batch_size, n)
        assert grads["db"].shape == (batch_size, m)

    def test_mpc_backward_batch128_cuda_after_jax_memory_pressure(self):
        """Backward pass fails when JAX has preallocated GPU memory.

        JAX grabs most GPU memory at startup. When moreau-cuda then tries to
        run the DiffKKT factorization (cuDSS) for a large batch, it fails
        because cuDSS can't allocate enough device memory.

        This is the exact scenario that occurs in benchmark_trajax.py when
        running trajax (JAX) benchmarks before moreau-cuda benchmarks.
        """
        if not moreau.device_available("cuda"):
            pytest.skip("CUDA not available")

        try:
            import jax
            import jax.numpy as jnp
        except ImportError:
            pytest.skip("JAX not installed")

        # bytes_limit is JAX's pool (75% of total GPU memory by default)
        gpu_pool_gb = jax.devices("gpu")[0].memory_stats()["bytes_limit"] / 1e9
        if gpu_pool_gb < 20:
            pytest.skip(
                f"JAX memory pool is {gpu_pool_gb:.0f}GB, need >=20GB for backward pass after JAX preallocation"
            )

        n_states, n_controls, horizon, batch_size = 32, 16, 100, 128

        # Force JAX to preallocate GPU memory (default behavior)
        x = jnp.ones((256, 256, 256))
        jax.block_until_ready(x)

        P, q_batch, A, b_batch, cones, n, m = _make_mpc_qp(
            n_states, n_controls, horizon, batch_size
        )
        P_csr = P.tocsr()
        A_csr = A.tocsr()

        ipm = moreau.IPMSettings(direct_solve_method="cudss")
        settings = moreau.Settings(
            device="cuda",
            verbose=False,
            enable_grad=True,
            batch_size=batch_size,
            ipm_settings=ipm,
        )
        solver = moreau.CompiledSolver(
            n, m, P_csr.indptr, P_csr.indices, A_csr.indptr, A_csr.indices, cones, settings
        )
        solver.setup(
            np.tile(P_csr.data, (batch_size, 1)),
            np.tile(A_csr.data, (batch_size, 1)),
        )

        result = solver.solve(q_batch, b_batch)
        assert result.x.shape == (batch_size, n)

        # This is the failing case: backward after JAX memory pressure
        dx = np.ones((batch_size, n))
        dz = np.ones((batch_size, m))
        ds = np.ones((batch_size, m))
        grads = solver.backward(dx, dz, ds)

        assert grads["dq"].shape == (batch_size, n)
        assert grads["db"].shape == (batch_size, m)
