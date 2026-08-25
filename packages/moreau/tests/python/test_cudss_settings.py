"""Tests that cuDSS IR and pivot settings propagate correctly through the solver."""

import numpy as np
import pytest

import moreau


@pytest.fixture
def simple_qp():
    """Simple QP: min 0.5*x'Px + q'x s.t. Ax + s = b, s in K."""
    n, m = 2, 3
    P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    P_col_indices = np.array([0, 1], dtype=np.int64)
    P_values = np.array([1.0, 1.0])

    A_row_offsets = np.array([0, 2, 3, 4], dtype=np.int64)
    A_col_indices = np.array([0, 1, 0, 1], dtype=np.int64)
    A_values = np.array([1.0, 1.0, 1.0, 1.0])

    q = np.array([2.0, 1.0])
    b = np.array([1.0, 2.0, 2.0])

    cones = moreau.Cones()
    cones.num_zero_cones = 1
    cones.num_nonneg_cones = 2

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "q": q,
        "b": b,
        "cones": cones,
    }


def _solve_with_cudss(simple_qp, batch_size=1, **ipm_kwargs):
    """Build a CompiledSolver with given cuDSS settings, solve, and return info."""
    ipm = moreau.IPMSettings(direct_solve_method="cudss", **ipm_kwargs)
    settings = moreau.Settings(
        device="cuda", verbose=False, batch_size=batch_size, ipm_settings=ipm
    )
    solver = moreau.CompiledSolver(
        simple_qp["n"],
        simple_qp["m"],
        simple_qp["P_row_offsets"],
        simple_qp["P_col_indices"],
        simple_qp["A_row_offsets"],
        simple_qp["A_col_indices"],
        simple_qp["cones"],
        settings,
    )
    solver.setup(simple_qp["P_values"], simple_qp["A_values"])
    q_batch = np.tile(simple_qp["q"], (batch_size, 1))
    b_batch = np.tile(simple_qp["b"], (batch_size, 1))
    solver.solve(q_batch, b_batch)
    return solver.info


@pytest.mark.skipif(
    not moreau.device_available("cuda"),
    reason="CUDA not available",
)
class TestCudssSettings:
    """Verify cuDSS IR and pivot settings propagate and solve correctly."""

    def test_cudss_default_ir_zero(self, simple_qp):
        """Default cudss_ir_steps=2 produces a valid solution."""
        ipm = moreau.IPMSettings(direct_solve_method="cudss")
        assert ipm.cudss_ir_steps == 2
        assert ipm.cudss_pivot_enable is False

        info = _solve_with_cudss(simple_qp)
        assert info.status[0] == moreau.SolverStatus.Solved

    def test_cudss_with_ir_steps(self, simple_qp):
        """Non-zero cudss_ir_steps produces a valid solution."""
        info = _solve_with_cudss(simple_qp, cudss_ir_steps=3)
        assert info.status[0] == moreau.SolverStatus.Solved

    def test_cudss_with_pivot_enabled(self, simple_qp):
        """cudss_pivot_enable=True produces a valid solution."""
        info = _solve_with_cudss(simple_qp, cudss_ir_steps=0, cudss_pivot_enable=True)
        assert info.status[0] == moreau.SolverStatus.Solved

    def test_cudss_ir_and_pivot_combined(self, simple_qp):
        """Both IR steps and pivot enabled together produce a valid solution."""
        info = _solve_with_cudss(simple_qp, batch_size=4, cudss_ir_steps=2, cudss_pivot_enable=True)
        assert all(s == moreau.SolverStatus.Solved for s in info.status)
