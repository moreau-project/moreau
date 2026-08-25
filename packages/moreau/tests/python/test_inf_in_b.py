"""
Test: solver handles inf in b vector for nonneg cone rows.

Problem: minimize x  subject to  x >= 0, x <= inf
Formulated as: min q'x, Ax + s = b, s in K (nonneg)
  A = [-1; 1], b = [0; inf], q = [1]

The x <= inf constraint is vacuous. The solver sanitizes inf entries
in b before equilibration, zeroing the A row and b entry.
"""

import numpy as np
import pytest
from scipy import sparse

import moreau

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from moreau.torch import Solver as TorchSolver

    HAS_MOREAU_TORCH = TorchSolver is not None
except ImportError:
    HAS_MOREAU_TORCH = False
    TorchSolver = None


def test_inf_in_b_solves(device):
    """Solver should handle inf in b for nonneg cone rows."""
    P = sparse.csr_matrix((1, 1))
    q = np.array([1.0])
    A = sparse.csr_matrix([[-1.0], [1.0]])
    b = np.array([0.0, np.inf])
    cones = moreau.Cones(num_nonneg_cones=2)

    settings = moreau.Settings(verbose=False, device=device)
    solver = moreau.Solver(P, q=q, A=A, b=b, cones=cones, settings=settings)
    solution = solver.solve()

    assert (
        solver.info.status == moreau.SolverStatus.Solved
    ), f"Expected Solved with inf in b, got {solver.info.status}"
    assert abs(solution.x[0]) < 1e-6
    assert abs(solver.info.obj_val) < 1e-6


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
@pytest.mark.skipif(not HAS_MOREAU_TORCH, reason="moreau.torch not available")
def test_inf_in_b_backward(device):
    """Backward pass should produce finite gradients when b contains inf."""
    P_row_offsets = torch.tensor([0, 0], dtype=torch.int64)
    P_col_indices = torch.tensor([], dtype=torch.int64)
    P_values = torch.zeros((1, 0), dtype=torch.float64, device=device)

    A_row_offsets = torch.tensor([0, 1, 2], dtype=torch.int64)
    A_col_indices = torch.tensor([0, 0], dtype=torch.int64)
    A_values = torch.tensor([[-1.0, 1.0]], dtype=torch.float64, device=device, requires_grad=True)

    q = torch.tensor([[1.0]], dtype=torch.float64, device=device, requires_grad=True)
    b = torch.tensor([[0.0, float("inf")]], dtype=torch.float64, device=device, requires_grad=True)

    cones = moreau.Cones(num_nonneg_cones=2)
    settings = moreau.Settings(verbose=False, batch_size=1, device=device)

    solver = TorchSolver(
        n=1,
        m=2,
        P_row_offsets=P_row_offsets,
        P_col_indices=P_col_indices,
        A_row_offsets=A_row_offsets,
        A_col_indices=A_col_indices,
        cones=cones,
        settings=settings,
    )

    result = solver.solve(P_values, A_values, q, b)

    # Forward should solve
    info = solver.info
    status = info.status[0] if isinstance(info.status, list) else info.status
    assert status == moreau.SolverStatus.Solved, f"Expected Solved, got {status}"

    # Backward should produce finite gradients
    result.x.sum().backward()
    assert torch.all(torch.isfinite(q.grad)), f"q.grad not finite: {q.grad}"
    assert torch.all(torch.isfinite(b.grad)), f"b.grad not finite: {b.grad}"
    assert torch.all(torch.isfinite(A_values.grad)), f"A_values.grad not finite: {A_values.grad}"
