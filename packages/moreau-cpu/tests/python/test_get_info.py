import moreau_cpu._cpu_solver as moreau
import pytest
import numpy as np
from scipy import sparse


@pytest.fixture
def get_info_qp_data():
    # Use CSR format (moreau's expected input format)
    P = sparse.csr_array([[4.0, 1.0], [1.0, 2.0]])
    P = sparse.triu(P).tocsr()

    A = sparse.csr_array(
        [[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )

    q = np.array([1.0, 1.0])
    b = np.array([-1.0, 0.0, 0.0, 1.0, 0.7, 0.7])

    cones = [moreau.NonnegativeConeT(3), moreau.NonnegativeConeT(3)]
    settings = moreau.DefaultSettings()
    return P, q, A, b, cones, settings


def test_get_info(get_info_qp_data):

    P, q, A, b, cones, settings = get_info_qp_data
    ipm = settings.ipm
    ipm.direct_solve_method = "auto"
    settings.ipm = ipm

    solver = moreau.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()
    info = solver.get_info()

    assert solution.status == moreau.SolverStatus.Solved
    assert info.linsolver.name in ("qdldl", "faer")
    assert info.linsolver.threads == 1
    assert info.linsolver.direct
    # nonzeros in upper triangle of KKT matrix
    assert info.linsolver.nnzA == 17
    # nonzeros in KKT L factor
    assert info.linsolver.nnzL == 9
