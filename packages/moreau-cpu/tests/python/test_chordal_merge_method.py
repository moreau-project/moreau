"""The chordal merge method reaches the CPU solver through settings_to_cpu."""

from types import SimpleNamespace

import moreau_cpu._cpu_solver as moreau
import numpy as np
import pytest
from moreau_cpu._cpu import settings_to_cpu
from scipy import sparse


def banded_psd8():
    """Tridiagonal PSD(8): unit diagonal, x_i on entry (i, i+1).

    The pattern is chordal with 7 cliques of order 2. parent_child merges them
    all (both supernodes have at most 8 vertices); none and clique_graph keep 7.
    """
    k = 8
    pairs = [(i, j) for j in range(k) for i in range(j + 1)]
    rows = [pairs.index((i, i + 1)) for i in range(k - 1)]
    A = sparse.csr_matrix(
        (-np.sqrt(2) * np.ones(k - 1), (rows, range(k - 1))), shape=(len(pairs), k - 1)
    )
    b = np.array([1.0 if i == j else 0.0 for i, j in pairs])
    return sparse.csr_matrix((k - 1, k - 1)), np.ones(k - 1), A, b, [moreau.PSDTriangleConeT(k)]


def kkt_nnz(enable, merge):
    ipm = SimpleNamespace(
        chordal_decomposition_enable=enable, chordal_decomposition_merge_method=merge
    )
    settings = settings_to_cpu(SimpleNamespace(ipm_settings=ipm))
    assert settings.ipm.chordal_decomposition_merge_method == merge
    solver = moreau.DefaultSolver(*banded_psd8(), settings)
    assert solver.solve().status == moreau.SolverStatus.Solved
    return solver.get_info().linsolver.nnzA


def test_merge_method_changes_decomposition():
    undecomposed = kkt_nnz(False, "clique_graph")
    assert kkt_nnz(True, "parent_child") == undecomposed
    split = kkt_nnz(True, "none")
    assert split != undecomposed
    assert kkt_nnz(True, "clique_graph") == split


def test_setting_the_binding_lacks_raises(monkeypatch):
    """A setting the CPU binding doesn't expose fails loudly instead of being dropped."""
    import moreau_cpu._cpu as cpu

    class Settings:
        def __init__(self):
            self.ipm = SimpleNamespace()

    monkeypatch.setattr(cpu, "_cpu_solver", SimpleNamespace(DefaultSettings=Settings))
    ipm = SimpleNamespace(chordal_decomposition_merge_method="none")
    with pytest.raises(AttributeError, match="no attribute 'chordal_decomposition_merge_method'"):
        cpu.settings_to_cpu(SimpleNamespace(ipm_settings=ipm))
