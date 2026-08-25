"""Tests for flat-array PyO3 bindings (setup_flat, solve_flat, backward_flat).

These test the Rust bindings directly via moreau_cpu._cpu_solver,
verifying buffer protocol transfer, flat array reshaping, and error handling.
"""

import numpy as np
import pytest

import moreau
from moreau_cpu._cpu import _cpu_solver


@pytest.fixture
def compiled_solver():
    """Create a CompiledSolver for a simple 2-var QP with equality + inequality cones."""
    n, m = 2, 3
    P_ro = [0, 1, 2]
    P_ci = [0, 1]
    A_ro = [0, 2, 3, 4]
    A_ci = [0, 1, 0, 1]
    cones = [_cpu_solver.ZeroConeT(1), _cpu_solver.NonnegativeConeT(2)]
    settings = _cpu_solver.DefaultSettings()
    settings.verbose = False

    solver = _cpu_solver.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=P_ro,
        P_col_indices=P_ci,
        A_row_offsets=A_ro,
        A_col_indices=A_ci,
        cones=cones,
        settings=settings,
        enable_grad=True,
    )
    return solver, n, m, len(P_ci), len(A_ci)


class TestFlatSetup:
    def test_setup_flat_basic(self, compiled_solver):
        solver, n, m, nnz_P, nnz_A = compiled_solver
        batch_size = 3
        P_flat = np.tile([1.0, 1.0], batch_size).astype(np.float64)
        A_flat = np.tile([1.0, 1.0, 1.0, 1.0], batch_size).astype(np.float64)
        solver.setup_flat(P_flat, A_flat, batch_size)

    def test_setup_flat_wrong_length(self, compiled_solver):
        solver, n, m, nnz_P, nnz_A = compiled_solver
        P_flat = np.array([1.0, 1.0], dtype=np.float64)  # only 1 problem
        A_flat = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)  # 2 problems
        with pytest.raises(Exception):
            solver.setup_flat(P_flat, A_flat, 2)

    def test_setup_flat_wrong_dtype(self, compiled_solver):
        solver, n, m, nnz_P, nnz_A = compiled_solver
        P_flat = np.array([1, 1], dtype=np.int32)
        A_flat = np.array([1, 1, 1, 1], dtype=np.int32)
        with pytest.raises(TypeError, match="float64"):
            solver.setup_flat(P_flat, A_flat, 1)


class TestFlatSolve:
    def test_solve_flat_matches_non_flat(self, compiled_solver):
        """Flat solve should produce identical results to non-flat solve."""
        solver, n, m, nnz_P, nnz_A = compiled_solver
        batch_size = 4

        P_vals = [2.0, 2.0]
        A_vals = [1.0, 1.0, -1.0, -1.0]
        q = [1.0, -1.0]
        b = [1.0, 0.0, 0.0]

        # Non-flat path
        solver.setup([P_vals] * batch_size, [A_vals] * batch_size)
        solutions_nf = solver.solve([q] * batch_size, [b] * batch_size)
        x_nf = np.array([np.array(s.x) for s in solutions_nf])

        # Flat path
        P_flat = np.tile(P_vals, batch_size).astype(np.float64)
        A_flat = np.tile(A_vals, batch_size).astype(np.float64)
        solver.setup_flat(P_flat, A_flat, batch_size)
        q_flat = np.tile(q, batch_size).astype(np.float64)
        b_flat = np.tile(b, batch_size).astype(np.float64)
        result = solver.solve_flat(q_flat, b_flat, batch_size)
        x_flat = np.array(result.x).reshape(batch_size, n)

        np.testing.assert_allclose(x_flat, x_nf, atol=1e-10)

    def test_solve_flat_returns_all_fields(self, compiled_solver):
        solver, n, m, nnz_P, nnz_A = compiled_solver
        batch_size = 2

        P_flat = np.tile([2.0, 2.0], batch_size)
        A_flat = np.tile([1.0, 1.0, -1.0, -1.0], batch_size)
        solver.setup_flat(P_flat, A_flat, batch_size)

        q_flat = np.array([1.0, -1.0, 1.0, -1.0])
        b_flat = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        result = solver.solve_flat(q_flat, b_flat, batch_size)

        assert len(result.x) == batch_size * n
        assert len(result.s) == batch_size * m
        assert len(result.z) == batch_size * m
        assert len(result.status) == batch_size
        assert len(result.obj_val) == batch_size
        assert len(result.obj_val_dual) == batch_size
        assert len(result.iterations) == batch_size
        assert result.solve_time > 0

    def test_solve_flat_wrong_q_length(self, compiled_solver):
        solver, n, m, nnz_P, nnz_A = compiled_solver
        batch_size = 2
        P_flat = np.tile([2.0, 2.0], batch_size)
        A_flat = np.tile([1.0, 1.0, -1.0, -1.0], batch_size)
        solver.setup_flat(P_flat, A_flat, batch_size)

        q_flat = np.array([1.0, -1.0])  # only 1 problem, expect 2
        b_flat = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        with pytest.raises(Exception):
            solver.solve_flat(q_flat, b_flat, batch_size)


class TestFlatBackward:
    def test_backward_flat_matches_non_flat(self, compiled_solver):
        """Flat backward should produce identical gradients to non-flat backward."""
        solver, n, m, nnz_P, nnz_A = compiled_solver
        batch_size = 2

        P_vals = [2.0, 2.0]
        A_vals = [1.0, 1.0, -1.0, -1.0]
        q = [1.0, -1.0]
        b = [1.0, 0.0, 0.0]

        # Setup and solve
        solver.setup([P_vals] * batch_size, [A_vals] * batch_size)
        solver.solve([q] * batch_size, [b] * batch_size)

        # Non-flat backward
        upstream = [
            _cpu_solver.UpstreamGradients(dx=[1.0, 0.0], ds=[0.0, 0.0, 0.0], dz=[0.0, 0.0, 0.0])
            for _ in range(batch_size)
        ]
        grads_nf = solver.backward(upstream)
        dq_nf = np.array([np.array(g.dq) for g in grads_nf])
        db_nf = np.array([np.array(g.db) for g in grads_nf])

        # Flat backward
        dx_flat = np.tile([1.0, 0.0], batch_size)
        ds_flat = np.zeros(batch_size * m)
        dz_flat = np.zeros(batch_size * m)
        grads_flat = solver.backward_flat(dx_flat, ds_flat, dz_flat, batch_size)
        dq_flat = np.array(grads_flat.dq).reshape(batch_size, n)
        db_flat = np.array(grads_flat.db).reshape(batch_size, m)

        np.testing.assert_allclose(dq_flat, dq_nf, atol=1e-10)
        np.testing.assert_allclose(db_flat, db_nf, atol=1e-10)

    def test_backward_flat_wrong_batch_size(self, compiled_solver):
        """backward_flat should raise on mismatched batch_size vs array lengths."""
        solver, n, m, nnz_P, nnz_A = compiled_solver
        batch_size = 2

        solver.setup([[2.0, 2.0]] * batch_size, [[1.0, 1.0, -1.0, -1.0]] * batch_size)
        solver.solve([[1.0, -1.0]] * batch_size, [[1.0, 0.0, 0.0]] * batch_size)

        # dx has 2 problems worth of data, but claim batch_size=3
        dx_flat = np.tile([1.0, 0.0], batch_size)
        ds_flat = np.zeros(batch_size * m)
        dz_flat = np.zeros(batch_size * m)
        with pytest.raises(ValueError):
            solver.backward_flat(dx_flat, ds_flat, dz_flat, batch_size + 1)


class TestFlatEndToEnd:
    def test_high_level_api_uses_flat_path(self):
        """Verify the high-level moreau API works end-to-end with flat bindings."""
        n, m = 2, 3
        P_ro = np.array([0, 1, 2], dtype=np.int64)
        P_ci = np.array([0, 1], dtype=np.int64)
        A_ro = np.array([0, 2, 3, 4], dtype=np.int64)
        A_ci = np.array([0, 1, 0, 1], dtype=np.int64)
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(batch_size=4, verbose=False, enable_grad=True)

        solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=cones,
            settings=settings,
        )
        solver.setup(
            P_values=np.tile([2.0, 2.0], (4, 1)),
            A_values=np.tile([1.0, 1.0, -1.0, -1.0], (4, 1)),
        )
        solution = solver.solve(
            qs=np.tile([1.0, -1.0], (4, 1)),
            bs=np.tile([1.0, 0.0, 0.0], (4, 1)),
        )
        assert solution.x.shape == (4, 2)

        # Backward through high-level API
        grads = solver.backward(
            dx=np.ones((4, n)),
            ds=np.zeros((4, m)),
            dz=np.zeros((4, m)),
        )
        assert grads["dq"].shape == (4, n)
        assert grads["db"].shape == (4, m)

    def test_unconstrained_m_zero(self):
        """Regression test: m=0 must not panic in chunks_exact(0)."""
        n, m = 2, 0
        solver = _cpu_solver.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0],
            A_col_indices=[],
            cones=[],
            settings=_cpu_solver.DefaultSettings(),
            enable_grad=True,
        )
        batch_size = 2
        solver.setup_shared([2.0, 2.0], [], batch_size)

        q_flat = np.array([1.0, -1.0, 0.5, -0.5])
        b_flat = np.array([], dtype=np.float64)
        result = solver.solve_flat(q_flat, b_flat, batch_size)

        assert len(result.x) == batch_size * n
        assert len(result.s) == 0
        assert len(result.z) == 0

        # Backward with m=0
        dx_flat = np.ones(batch_size * n)
        ds_flat = np.array([], dtype=np.float64)
        dz_flat = np.array([], dtype=np.float64)
        grads = solver.backward_flat(dx_flat, ds_flat, dz_flat, batch_size)
        assert len(grads.dq) == batch_size * n
