import moreau_cpu._cpu_solver as moreau
import pytest
import numpy as np
from scipy import sparse
from scipy.sparse import vstack


@pytest.fixture
def basic_sdp_data():
    # Use CSR format (moreau's expected input format)
    P = sparse.eye(6).tocsr()
    A = sparse.eye(6).tocsr()

    q = np.zeros(6)
    b = np.array([-3.0, 1.0, 4.0, 1.0, 2.0, 5.0])

    cones = [moreau.PSDTriangleConeT(3)]
    settings = moreau.DefaultSettings()
    return P, q, A, b, cones, settings


@pytest.fixture
def basic_sdp_solution():

    refsol = np.array(
        [
            -3.0729833267361095,
            0.3696004167288786,
            -0.022226685581313674,
            0.31441213129613066,
            -0.026739700851545107,
            -0.016084530571308823,
        ]
    )
    refobj = 4.840076866013861

    return refsol, refobj


def test_sdp_feasible(basic_sdp_data, basic_sdp_solution):

    P, q, A, b, cones, settings = basic_sdp_data
    refsol, refobj = basic_sdp_solution

    solver = moreau.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    assert solution.status == moreau.SolverStatus.Solved
    assert np.allclose(solution.x, refsol)
    assert np.allclose(solution.obj_val, refobj)
    assert np.allclose(solution.obj_val_dual, refobj)


def test_sdp_empty_cone(basic_sdp_data, basic_sdp_solution):
    """Test that zero-dimension PSD cones are correctly rejected.

    PSD cones with dimension 0 are invalid and should raise an error.
    """
    P, q, A, b, cones, settings = basic_sdp_data

    cones = np.append(cones, moreau.PSDTriangleConeT(0))

    # Zero-dimension PSD cone should be rejected
    with pytest.raises(Exception, match="dimension must be positive"):
        moreau.DefaultSolver(P, q, A, b, cones, settings)


def test_sdp_primal_infeasible(basic_sdp_data):

    P, q, A, b, cones, settings = basic_sdp_data

    A = vstack((A, -A)).tocsr()  # Ensure CSR format
    b = np.pad(b, (0, len(b)))
    cones = np.concatenate((cones, cones))

    solver = moreau.DefaultSolver(P, q, A, b, cones, settings)
    solution = solver.solve()

    assert solution.status == moreau.SolverStatus.PrimalInfeasible
    assert np.isnan(solution.obj_val)
    assert np.isnan(solution.obj_val_dual)
