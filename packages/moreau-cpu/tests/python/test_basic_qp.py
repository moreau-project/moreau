import moreau_cpu._cpu_solver as moreau
import pytest
import numpy as np
from scipy import sparse


@pytest.fixture
def basic_qp_data():
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


@pytest.fixture
def basic_qp_data_dual_inf():
    # Use CSR format (moreau's expected input format)
    P = sparse.csr_array([[1.0, 1.0], [1.0, 1.0]])
    P = sparse.triu(P).tocsr()

    A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0]])

    q = np.array([1.0, -1.0])
    b = np.array([1.0, 1.0])

    cones = [moreau.NonnegativeConeT(2)]
    settings = moreau.DefaultSettings()
    return P, q, A, b, cones, settings


def test_qp_feasible(basic_qp_data):

    P, q, A, b, cones, settings = basic_qp_data

    solver = moreau.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    refsol = np.array([0.3, 0.7])
    refobj = 1.8800000298331538

    assert solution.status == moreau.SolverStatus.Solved
    assert np.allclose(solution.x, refsol)
    assert np.allclose(solution.obj_val, refobj)
    assert np.allclose(solution.obj_val_dual, refobj)


def test_qp_primal_infeasible(basic_qp_data):

    P, q, A, b, cones, settings = basic_qp_data
    b[0] = -1.0
    b[3] = -1.0

    solver = moreau.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    assert solution.status == moreau.SolverStatus.PrimalInfeasible
    assert np.isnan(solution.obj_val)
    assert np.isnan(solution.obj_val_dual)


def test_qp_dual_infeasible(basic_qp_data_dual_inf):

    P, q, A, b, cones, settings = basic_qp_data_dual_inf

    solver = moreau.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    assert solution.status == moreau.SolverStatus.DualInfeasible
    assert np.isnan(solution.obj_val)
    assert np.isnan(solution.obj_val_dual)
