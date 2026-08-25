"""Tests for solver lifecycle, sequential solves, info access, and setup variations.

Covers gaps identified in test audit:
1. Solver.info is None before solve()
2. CompiledSolver.info is None before solve()
3. Multiple sequential solves on same CompiledSolver
4. CompiledSolver setup() with shared (1D) vs per-batch (2D) values
5. P symmetry validation (asymmetric P rejected)
6. P sparsity pattern symmetry validation
7. Solver properties (n, m, device, construction_time)
8. CompiledSolver properties (n, m, batch_size, device, construction_time)
"""

import moreau
import pytest
import numpy as np
from scipy import sparse

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_qp_data():
    """Simple 2D QP data for testing."""
    return {
        "n": 2,
        "m": 3,
        "P_row_offsets": np.array([0, 2, 4], dtype=np.int64),
        "P_col_indices": np.array([0, 1, 0, 1], dtype=np.int64),
        "P_values": np.array([4.0, 1.0, 1.0, 2.0]),
        "A_row_offsets": np.array([0, 2, 3, 4], dtype=np.int64),
        "A_col_indices": np.array([0, 1, 0, 1], dtype=np.int64),
        "A_values": np.array([1.0, 1.0, 1.0, 1.0]),
        "cones": moreau.Cones(num_zero_cones=1, num_nonneg_cones=2),
        "q": np.array([1.0, 1.0]),
        "b": np.array([1.0, 10.0, 10.0]),
    }


# ============================================================================
# Solver info lifecycle
# ============================================================================


class TestSolverInfoLifecycle:
    """Test that Solver.info behaves correctly before/after solve."""

    def test_info_none_before_solve(self, simple_qp_data, device):
        d = simple_qp_data
        P = sparse.csr_array(
            (d["P_values"], d["P_col_indices"], d["P_row_offsets"]), shape=(d["n"], d["n"])
        )
        A = sparse.csr_array(
            (d["A_values"], d["A_col_indices"], d["A_row_offsets"]), shape=(d["m"], d["n"])
        )
        solver = moreau.Solver(
            P,
            d["q"],
            A,
            d["b"],
            cones=d["cones"],
            settings=moreau.Settings(device=device),
        )
        assert solver.info is None

    def test_info_populated_after_solve(self, simple_qp_data, device):
        d = simple_qp_data
        P = sparse.csr_array(
            (d["P_values"], d["P_col_indices"], d["P_row_offsets"]), shape=(d["n"], d["n"])
        )
        A = sparse.csr_array(
            (d["A_values"], d["A_col_indices"], d["A_row_offsets"]), shape=(d["m"], d["n"])
        )
        solver = moreau.Solver(
            P,
            d["q"],
            A,
            d["b"],
            cones=d["cones"],
            settings=moreau.Settings(device=device),
        )
        solver.solve()
        assert solver.info is not None
        assert solver.info.status == moreau.SolverStatus.Solved
        assert solver.info.iterations > 0
        assert solver.info.solve_time >= 0.0
        assert isinstance(solver.info.obj_val, float)

    def test_solver_properties(self, simple_qp_data, device):
        d = simple_qp_data
        P = sparse.csr_array(
            (d["P_values"], d["P_col_indices"], d["P_row_offsets"]), shape=(d["n"], d["n"])
        )
        A = sparse.csr_array(
            (d["A_values"], d["A_col_indices"], d["A_row_offsets"]), shape=(d["m"], d["n"])
        )
        solver = moreau.Solver(
            P,
            d["q"],
            A,
            d["b"],
            cones=d["cones"],
            settings=moreau.Settings(device=device),
        )
        assert solver.n == 2
        assert solver.m == 3
        assert solver.device in ("cpu", "cuda")
        assert solver.construction_time > 0.0


class TestCompiledSolverInfoLifecycle:
    """Test that CompiledSolver.info behaves correctly before/after solve."""

    def test_info_none_before_solve(self, simple_qp_data, device):
        d = simple_qp_data
        settings = moreau.Settings(device=device, batch_size=1)
        solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )
        assert solver.info is None

    def test_info_populated_after_solve(self, simple_qp_data, device):
        d = simple_qp_data
        settings = moreau.Settings(device=device, batch_size=1)
        solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )
        solver.setup(d["P_values"], d["A_values"])
        solver.solve(qs=[d["q"]], bs=[d["b"]])
        assert solver.info is not None
        assert isinstance(solver.info.status, list)
        assert solver.info.status[0] == moreau.SolverStatus.Solved

    def test_compiled_solver_properties(self, simple_qp_data, device):
        d = simple_qp_data
        settings = moreau.Settings(device=device, batch_size=2)
        solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )
        assert solver.n == 2
        assert solver.m == 3
        assert solver.batch_size == 2
        assert solver.device in ("cpu", "cuda")
        assert solver.construction_time > 0.0


# ============================================================================
# Sequential solves
# ============================================================================


class TestSequentialSolves:
    """Test multiple sequential solves on the same CompiledSolver."""

    def test_same_data_gives_same_result(self, simple_qp_data, device):
        d = simple_qp_data
        settings = moreau.Settings(device=device, batch_size=1)
        solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )
        solver.setup(d["P_values"], d["A_values"])

        sol1 = solver.solve(qs=[d["q"]], bs=[d["b"]])
        info1 = solver.info
        sol2 = solver.solve(qs=[d["q"]], bs=[d["b"]])
        info2 = solver.info

        assert info1.status[0] == moreau.SolverStatus.Solved
        assert info2.status[0] == moreau.SolverStatus.Solved
        np.testing.assert_allclose(sol1.x, sol2.x, atol=1e-8)

    def test_different_q_gives_different_result(self, simple_qp_data, device):
        d = simple_qp_data
        settings = moreau.Settings(device=device, batch_size=1)
        solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )
        solver.setup(d["P_values"], d["A_values"])

        sol1 = solver.solve(qs=[[1.0, 1.0]], bs=[d["b"]])
        sol2 = solver.solve(qs=[[2.0, 0.5]], bs=[d["b"]])

        assert solver.info.status[0] == moreau.SolverStatus.Solved
        # Different q should give different x
        assert not np.allclose(sol1.x, sol2.x, atol=1e-4)

    def test_re_setup_and_solve(self, simple_qp_data, device):
        """Test calling setup() again with new values then solving."""
        d = simple_qp_data
        settings = moreau.Settings(device=device, batch_size=1)
        solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )

        # First setup + solve
        solver.setup(d["P_values"], d["A_values"])
        sol1 = solver.solve(qs=[d["q"]], bs=[d["b"]])
        assert solver.info.status[0] == moreau.SolverStatus.Solved

        # Second setup with different P values + solve
        new_P_values = np.array([2.0, 0.0, 0.0, 20.0])  # very different Hessian
        solver.setup(new_P_values, d["A_values"])
        sol2 = solver.solve(qs=[d["q"]], bs=[d["b"]])
        assert solver.info.status[0] == moreau.SolverStatus.Solved

        # Solutions should differ due to different P
        assert not np.allclose(sol1.x, sol2.x, atol=1e-4)

    def test_many_sequential_solves(self, simple_qp_data, device):
        """Stress test: many sequential solves don't accumulate errors."""
        d = simple_qp_data
        settings = moreau.Settings(device=device, batch_size=1)
        solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )
        solver.setup(d["P_values"], d["A_values"])

        # Get reference solution
        ref_sol = solver.solve(qs=[d["q"]], bs=[d["b"]])
        assert solver.info.status[0] == moreau.SolverStatus.Solved

        # Solve many times, verify consistency
        for i in range(20):
            sol = solver.solve(qs=[d["q"]], bs=[d["b"]])
            assert solver.info.status[0] == moreau.SolverStatus.Solved
            np.testing.assert_allclose(
                sol.x,
                ref_sol.x,
                atol=1e-8,
                err_msg=f"Mismatch on iteration {i}",
            )


# ============================================================================
# Setup shared vs per-batch values
# ============================================================================


class TestSetupValueBroadcasting:
    """Test CompiledSolver.setup() with shared (1D) vs per-batch (2D) values."""

    def test_shared_values_1d(self, simple_qp_data, device):
        """1D P_values and A_values should be broadcast to all batch elements."""
        d = simple_qp_data
        batch_size = 3
        settings = moreau.Settings(device=device, batch_size=batch_size)
        solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )

        # 1D values (shared)
        solver.setup(d["P_values"], d["A_values"])

        # All batch elements have same P/A, same q/b -> same solution
        q_batch = np.tile(d["q"], (batch_size, 1))
        b_batch = np.tile(d["b"], (batch_size, 1))
        sol = solver.solve(qs=q_batch, bs=b_batch)

        assert all(s == moreau.SolverStatus.Solved for s in solver.info.status)
        # All solutions should be identical
        for i in range(1, batch_size):
            np.testing.assert_allclose(sol.x[i], sol.x[0], atol=1e-8)

    def test_per_batch_values_2d(self, simple_qp_data, device):
        """2D P_values should be per-problem."""
        d = simple_qp_data
        batch_size = 2
        settings = moreau.Settings(device=device, batch_size=batch_size)
        solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )

        # 2D values (very different per problem to ensure distinct solutions)
        P_batch = np.array(
            [
                [4.0, 1.0, 1.0, 2.0],
                [2.0, 0.0, 0.0, 20.0],
            ]
        )
        A_batch = np.tile(d["A_values"], (batch_size, 1))
        solver.setup(P_batch, A_batch)

        q_batch = np.tile(d["q"], (batch_size, 1))
        b_batch = np.tile(d["b"], (batch_size, 1))
        sol = solver.solve(qs=q_batch, bs=b_batch)

        assert all(s == moreau.SolverStatus.Solved for s in solver.info.status)
        # Different P -> different solutions
        assert not np.allclose(sol.x[0], sol.x[1], atol=1e-4)

    def test_shared_1d_matches_explicit_2d(self, simple_qp_data, device):
        """1D shared setup should give same results as explicit 2D tiled setup."""
        d = simple_qp_data
        batch_size = 2
        settings = moreau.Settings(device=device, batch_size=batch_size)

        # Solver 1: shared (1D) values
        solver1 = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )
        solver1.setup(d["P_values"], d["A_values"])

        # Solver 2: explicit 2D tiled values
        solver2 = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )
        solver2.setup(
            np.tile(d["P_values"], (batch_size, 1)),
            np.tile(d["A_values"], (batch_size, 1)),
        )

        q_batch = np.tile(d["q"], (batch_size, 1))
        b_batch = np.tile(d["b"], (batch_size, 1))

        sol1 = solver1.solve(qs=q_batch, bs=b_batch)
        sol2 = solver2.solve(qs=q_batch, bs=b_batch)

        np.testing.assert_allclose(sol1.x, sol2.x, atol=1e-8)


# ============================================================================
# P symmetry validation
# ============================================================================


class TestPSymmetryValidation:
    """Test that asymmetric P is rejected."""

    def test_asymmetric_P_rejected_by_solver(self, device):
        """Solver should reject a non-symmetric P matrix."""
        n, m = 2, 2
        # Asymmetric P
        P = sparse.csr_array(np.array([[1.0, 2.0], [0.0, 1.0]]))
        q = np.array([1.0, 1.0])
        A = sparse.csr_array(np.array([[-1.0, 0.0], [0.0, -1.0]]))
        b = np.array([0.0, 0.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        with pytest.raises(ValueError, match="not numerically symmetric"):
            moreau.Solver(P, q, A, b, cones, moreau.Settings(device=device))

    def test_symmetric_P_accepted(self, device):
        """Solver should accept a symmetric P matrix."""
        P = sparse.csr_array(np.array([[2.0, 1.0], [1.0, 2.0]]))
        q = np.array([1.0, 1.0])
        A = sparse.csr_array(np.array([[-1.0, 0.0], [0.0, -1.0]]))
        b = np.array([0.0, 0.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        # Should not raise
        solver = moreau.Solver(P, q, A, b, cones, moreau.Settings(device=device))
        assert solver.n == 2

    def test_asymmetric_sparsity_pattern_rejected(self, device):
        """CompiledSolver should reject asymmetric sparsity pattern for P."""
        # P has entry at (0,1) but not (1,0) in sparsity pattern
        with pytest.raises(ValueError, match="sparsity pattern is not symmetric"):
            moreau.CompiledSolver(
                n=2,
                m=2,
                P_row_offsets=[0, 2, 3],  # row 0 has 2 entries, row 1 has 1
                P_col_indices=[0, 1, 1],  # (0,0), (0,1), (1,1) - missing (1,0)
                A_row_offsets=[0, 1, 2],
                A_col_indices=[0, 1],
                cones=moreau.Cones(num_nonneg_cones=2),
                settings=moreau.Settings(device=device),
            )

    def test_diagonal_P_accepted(self, device):
        """Diagonal P (trivially symmetric) should be accepted."""
        solver = moreau.CompiledSolver(
            n=2,
            m=2,
            P_row_offsets=[0, 1, 2],
            P_col_indices=[0, 1],
            A_row_offsets=[0, 1, 2],
            A_col_indices=[0, 1],
            cones=moreau.Cones(num_nonneg_cones=2),
            settings=moreau.Settings(device=device),
        )
        assert solver.n == 2


# ============================================================================
# Warm start validation
# ============================================================================


class TestWarmStartValidation:
    """Test that warm_start parameter is validated."""

    def test_wrong_type_rejected(self, simple_qp_data, device):
        d = simple_qp_data
        settings = moreau.Settings(device=device, batch_size=1)
        solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )
        solver.setup(d["P_values"], d["A_values"])

        with pytest.raises(TypeError, match="warm_start must be"):
            solver.solve(
                qs=[d["q"]],
                bs=[d["b"]],
                warm_start="not_a_warm_start",
            )

    def test_wrong_shape_rejected(self, simple_qp_data, device):
        d = simple_qp_data
        settings = moreau.Settings(device=device, batch_size=1)
        solver = moreau.CompiledSolver(
            n=d["n"],
            m=d["m"],
            P_row_offsets=d["P_row_offsets"],
            P_col_indices=d["P_col_indices"],
            A_row_offsets=d["A_row_offsets"],
            A_col_indices=d["A_col_indices"],
            cones=d["cones"],
            settings=settings,
        )
        solver.setup(d["P_values"], d["A_values"])

        # Create a WarmStart with wrong dimensions
        wrong_ws = moreau.BatchedWarmStart(
            x=np.zeros((1, d["n"] + 1)),
            z=np.zeros((1, d["m"])),
            s=np.zeros((1, d["m"])),
        )
        with pytest.raises(ValueError, match="warm_start.x shape"):
            solver.solve(
                qs=[d["q"]],
                bs=[d["b"]],
                warm_start=wrong_ws,
            )
