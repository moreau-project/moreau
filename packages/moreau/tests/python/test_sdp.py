"""
Tests for PSD (SDP) cone support via the unified moreau API.

Tests cover:
- Forward solve with PSD cones
- PSD + equality constraints (mixed cones)
- Multiple PSD cones of different sizes
- PSD dim=1 (degenerates to nonneg)
- Backward pass: finite-difference validation of gradients through PSD constraints
"""

import numpy as np
import pytest
from scipy import sparse

import moreau

EPS_FD = 1e-5
TOL_FD = 1e-2
KKT_TOL = 1e-6


def _solve_fresh(n, m, P_dense, A_dense, cones, q, b, device="cpu"):
    """Create a fresh solver, solve, return (x, z, s)."""
    P_sym = 0.5 * (P_dense + P_dense.T)
    P_csr = sparse.csr_matrix(P_sym)
    A_csr = sparse.csr_matrix(A_dense)

    settings = moreau.Settings(
        device=device,
        enable_grad=False,
        verbose=False,
    )
    settings.ipm_settings.tol_gap_abs = 1e-9
    settings.ipm_settings.tol_feas = 1e-9

    solver = moreau.Solver(P_csr, q, A_csr, b, cones, settings=settings)
    solution = solver.solve()
    return np.array(solution.x), np.array(solution.z), np.array(solution.s)


def _backward_moreau(P_dense, A_dense, q, b, cones, dx_bar, dy_bar, ds_bar, device="cpu"):
    """Run moreau backward pass and return (dP_dense, dq, dA_dense, db)."""
    n = P_dense.shape[0]
    m = A_dense.shape[0]
    P_sym = 0.5 * (P_dense + P_dense.T)
    P_csr = sparse.csr_matrix(P_sym)
    A_csr = sparse.csr_matrix(A_dense)

    settings = moreau.Settings(
        device=device,
        enable_grad=True,
        verbose=False,
    )
    settings.ipm_settings.tol_gap_abs = 1e-9
    settings.ipm_settings.tol_feas = 1e-9

    solver = moreau.Solver(P_csr, q, A_csr, b, cones, settings=settings)
    solver.solve()

    grads = solver.backward(dx=dx_bar, dz=dy_bar, ds=ds_bar)

    dq = np.array(grads["dq"])
    db = np.array(grads["db"])

    dP_values = np.array(grads["dP_values"])
    dP_csr = sparse.csr_matrix((dP_values, P_csr.indices, P_csr.indptr), shape=(n, n))
    dP = dP_csr.toarray()

    dA_values = np.array(grads["dA_values"])
    dA_csr = sparse.csr_matrix((dA_values, A_csr.indices, A_csr.indptr), shape=(m, n))
    dA = dA_csr.toarray()

    return dP, dq, dA, db


def _fd_backward_dq(n, m, P, A, cones, q, b, dx_bar, dy_bar, ds_bar):
    """FD for dq via central differences."""
    dq_fd = np.zeros(n)
    for j in range(n):
        qp = q.copy()
        qp[j] += EPS_FD
        qm = q.copy()
        qm[j] -= EPS_FD
        xp, yp, sp = _solve_fresh(n, m, P, A, cones, qp, b)
        xm, ym, sm = _solve_fresh(n, m, P, A, cones, qm, b)
        dq_fd[j] = (dx_bar @ (xp - xm) + dy_bar @ (yp - ym) + ds_bar @ (sp - sm)) / (2 * EPS_FD)
    return dq_fd


def _fd_backward_db(n, m, P, A, cones, q, b, dx_bar, dy_bar, ds_bar):
    """FD for db via central differences."""
    db_fd = np.zeros(m)
    for j in range(m):
        bp = b.copy()
        bp[j] += EPS_FD
        bm = b.copy()
        bm[j] -= EPS_FD
        xp, yp, sp = _solve_fresh(n, m, P, A, cones, q, bp)
        xm, ym, sm = _solve_fresh(n, m, P, A, cones, q, bm)
        db_fd[j] = (dx_bar @ (xp - xm) + dy_bar @ (yp - ym) + ds_bar @ (sp - sm)) / (2 * EPS_FD)
    return db_fd


# PSD cones are now supported on both CPU and CUDA backends.
# No device restriction — tests will run on all available devices.


class TestPSDForward:
    """Test forward solve with PSD cones."""

    def test_basic_sdp(self):
        """Simple SDP: min ||x||^2 s.t. X = mat(Ax+s) is PSD, s in svec form."""
        # 3x3 PSD cone => 6 svec variables
        n = 6
        m = 6

        P = np.eye(n)
        A = np.eye(m)
        q = np.zeros(n)
        b = np.array([-3.0, 1.0, 4.0, 1.0, 2.0, 5.0])

        cones = moreau.Cones(psd_dims=[3])

        x, z, s = _solve_fresh(n, m, P, A, cones, q, b)
        assert x.shape == (n,)
        assert z.shape == (m,)
        assert s.shape == (m,)

    def test_psd_dim1(self):
        """PSD cone of dim 1 should behave like nonneg cone."""
        # 1x1 PSD cone => 1 svec variable
        n = 1
        m = 1

        P = np.eye(n)
        A = np.eye(m)
        q = np.array([2.0])
        b = np.array([0.0])  # s >= 0

        cones = moreau.Cones(psd_dims=[1])

        x, z, s = _solve_fresh(n, m, P, A, cones, q, b)
        # min 0.5*x^2 + 2x s.t. x + s = 0, s >= 0 => x <= 0
        # Solution: x = -2, s = 2 (active at 0 if q=0, but with q=2 it pushes x negative)
        assert x[0] < 0.0 + 1e-6

    def test_psd_dim2(self):
        """PSD cone of dim 2."""
        # 2x2 PSD => 3 svec variables [a, sqrt(2)*b, c] for [[a,b],[b,c]]
        n = 3
        m = 3

        P = np.eye(n)
        A = np.eye(m)
        q = np.zeros(n)
        b = np.array([1.0, 0.0, 1.0])  # identity matrix in svec

        cones = moreau.Cones(psd_dims=[2])

        x, z, s = _solve_fresh(n, m, P, A, cones, q, b)
        assert x.shape == (n,)

    def test_psd_with_equality(self):
        """PSD cone + zero cone (equality constraints)."""
        # 2 vars, 1 equality + 3 PSD(2)
        n = 2
        m = 4  # 1 zero + 3 PSD(2)

        P = np.eye(n)
        q = np.zeros(n)

        # A maps 2 vars to 4 constraints
        # Row 0: equality x[0] + x[1] = 1
        # Rows 1-3: PSD(2) cone, s = b - Ax must be PSD in svec form
        A = np.zeros((m, n))
        A[0, 0] = 1.0
        A[0, 1] = 1.0  # x[0] + x[1] = 1
        A[1, 0] = 1.0  # s[1] = b[1] - x[0]
        A[2, :] = 0.0  # s[2] = 0 (off-diagonal)
        A[3, 1] = 1.0  # s[3] = b[3] - x[1]

        # b chosen so PSD matrix [[2-x0, 0], [0, 2-x1]] is PSD for small x
        b = np.array([1.0, 2.0, 0.0, 2.0])

        cones = moreau.Cones(num_zero_cones=1, psd_dims=[2])

        x, z, s = _solve_fresh(n, m, P, A, cones, q, b)
        # Check equality constraint
        assert abs(x[0] + x[1] - 1.0) < 1e-6

    def test_multiple_psd_cones(self):
        """Multiple PSD cones of different sizes."""
        # Two PSD cones: 2x2 (3 vars) + 3x3 (6 vars) = 9 constraint dims
        n = 9
        m = 9

        P = np.eye(n)
        A = np.eye(m)
        q = np.zeros(n)
        # b for first PSD: identity 2x2 in svec
        # b for second PSD: identity 3x3 in svec
        b = np.array(
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0]  # 2x2 identity svec
        )  # 3x3 identity svec

        cones = moreau.Cones(psd_dims=[2, 3])

        x, z, s = _solve_fresh(n, m, P, A, cones, q, b)
        assert x.shape == (n,)


class TestPSDLogDetSchurRegression:
    """Regression: log-det + Schur-complement PSD MLE (was infeasible).

    Gaussian copula MLE: max log|Lambda| - tr(Lambda @ S) with Sigma = Lambda^-1
    enforced via a Schur-complement PSD block and unit-diagonal equalities.
    Canonicalized via Clarabel (same scaled-svec/exp-cone convention as moreau).
    """

    def test_logdet_schur_psd_solves(self):
        cp = pytest.importorskip("cvxpy")
        d = 3
        S = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.2], [0.3, 0.2, 1.0]])
        Sigma = cp.Variable((d, d), symmetric=True)
        Lambda = cp.Variable((d, d), symmetric=True)
        prob = cp.Problem(
            cp.Maximize(cp.log_det(Lambda) - cp.trace(Lambda @ S)),
            [cp.bmat([[Lambda, np.eye(d)], [np.eye(d), Sigma]]) >> 0, cp.diag(Sigma) == 1],
        )
        data, _, _ = prob.get_problem_data(cp.CLARABEL)
        cd = data["dims"]
        c = np.asarray(data["c"]).ravel()
        A = sparse.csr_matrix(data["A"])
        b = np.asarray(data["b"]).ravel()
        n, m = c.shape[0], b.shape[0]
        cones = moreau.Cones(
            num_zero_cones=cd.zero,
            num_nonneg_cones=cd.nonneg,
            so_cone_dims=list(cd.soc),
            num_exp_cones=cd.exp,
            power_alphas=list(cd.p3d),
            psd_dims=list(cd.psd),
        )
        solver = moreau.Solver(
            sparse.csr_matrix((n, n)), c, A, b, cones, moreau.Settings(device="cpu")
        )
        solver.solve()
        assert solver.info.status in (moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved)
        # Sigma = S is the optimum; SCS/Clarabel objective is -2.6143 (min form +2.6143).
        np.testing.assert_allclose(solver.info.obj_val, 2.6143375, atol=1e-4)


class TestPSDBackward:
    """Test backward pass (differentiation) with PSD cones."""

    def setup_method(self):
        np.random.seed(42)
        # Simple SDP: 3x3 PSD cone (6 svec vars)
        self.n = 6
        self.m = 6

        L = np.random.randn(self.n, self.n)
        self.P = L.T @ L + 1.0 * np.eye(self.n)
        self.A = np.eye(self.m)
        self.q = np.random.randn(self.n) * 0.1
        # Make b correspond to a PSD matrix (identity + small perturbation)
        self.b = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0])  # identity in svec
        self.cones = moreau.Cones(psd_dims=[3])

    def test_backward_dq(self):
        np.random.seed(101)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.zeros(self.m)

        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar
        )
        dq_fd = _fd_backward_dq(
            self.n, self.m, self.P, self.A, self.cones, self.q, self.b, dx_bar, dy_bar, ds_bar
        )
        assert np.allclose(
            dq, dq_fd, atol=TOL_FD
        ), f"PSD dq:\n  analytic: {dq}\n  FD: {dq_fd}\n  diff: {np.abs(dq - dq_fd)}"

    def test_backward_db(self):
        np.random.seed(102)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.zeros(self.m)

        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar
        )
        db_fd = _fd_backward_db(
            self.n, self.m, self.P, self.A, self.cones, self.q, self.b, dx_bar, dy_bar, ds_bar
        )
        assert np.allclose(
            db, db_fd, atol=TOL_FD
        ), f"PSD db:\n  analytic: {db}\n  FD: {db_fd}\n  diff: {np.abs(db - db_fd)}"

    def test_backward_psd_dim1(self):
        """Backward pass for PSD dim=1 (should match nonneg)."""
        np.random.seed(103)
        n, m = 2, 2
        P = np.eye(n) * 2.0
        A = np.eye(m)
        q = np.array([1.0, -1.0])
        b = np.array([0.0, 0.0])
        cones = moreau.Cones(psd_dims=[1, 1])

        dx_bar = np.random.randn(n)
        dy_bar = np.random.randn(m)
        ds_bar = np.zeros(m)

        dP, dq, dA, db = _backward_moreau(P, A, q, b, cones, dx_bar, dy_bar, ds_bar)
        dq_fd = _fd_backward_dq(n, m, P, A, cones, q, b, dx_bar, dy_bar, ds_bar)
        assert np.allclose(
            dq, dq_fd, atol=TOL_FD
        ), f"PSD dim=1 dq:\n  analytic: {dq}\n  FD: {dq_fd}"

    def test_backward_mixed_psd_zero(self):
        """Backward pass for mixed PSD + zero cones."""
        np.random.seed(104)
        n = 4
        m = 4  # 1 zero + 3 PSD(2)
        P = np.eye(n)
        A = np.random.randn(m, n) * 0.5
        A[0, :] = [1, 1, 0, 0]  # equality constraint
        q = np.random.randn(n) * 0.1

        # Make sure problem is feasible
        x0 = np.zeros(n)
        b = A @ x0 + np.array([1.0, 1.0, 0.0, 1.0])  # PSD-feasible RHS

        cones = moreau.Cones(num_zero_cones=1, psd_dims=[2])

        dx_bar = np.random.randn(n)
        dy_bar = np.random.randn(m)
        ds_bar = np.zeros(m)

        dP, dq, dA, db = _backward_moreau(P, A, q, b, cones, dx_bar, dy_bar, ds_bar)
        dq_fd = _fd_backward_dq(n, m, P, A, cones, q, b, dx_bar, dy_bar, ds_bar)
        assert np.allclose(
            dq, dq_fd, atol=TOL_FD
        ), f"Mixed PSD+Zero dq diff: {np.max(np.abs(dq - dq_fd)):.2e}"


class TestPSDValidation:
    """Test validation of PSD cone parameters."""

    def test_psd_dim_zero_rejected(self):
        """PSD dim=0 should be rejected."""
        with pytest.raises(ValueError, match="psd_dims.*must be >= 1"):
            moreau.Cones(psd_dims=[0])

    def test_psd_negative_dim_rejected(self):
        """Negative PSD dim should be rejected."""
        with pytest.raises(ValueError, match="psd_dims.*must be >= 1"):
            moreau.Cones(psd_dims=[-1])

    def test_psd_total_constraints(self):
        """Test total_constraints includes PSD dims."""
        cones = moreau.Cones(num_zero_cones=2, psd_dims=[3, 2])
        # 2 zero + 3*(3+1)/2 + 2*(2+1)/2 = 2 + 6 + 3 = 11
        assert cones.total_constraints() == 11

    def test_psd_degree(self):
        """Test degree includes PSD dims."""
        cones = moreau.Cones(psd_dims=[3, 2])
        # degree for PSD = sum of matrix dims = 3 + 2 = 5
        assert cones.degree() == 5


@pytest.mark.cuda
class TestPSDForwardCUDA:
    """Test forward solve with PSD cones on CUDA backend."""

    def test_basic_sdp_cuda(self):
        """Same as CPU test_basic_sdp but on CUDA."""
        n = 6
        m = 6
        P = np.eye(n)
        A = np.eye(m)
        q = np.zeros(n)
        b = np.array([-3.0, 1.0, 4.0, 1.0, 2.0, 5.0])
        cones = moreau.Cones(psd_dims=[3])

        x, z, s = _solve_fresh(n, m, P, A, cones, q, b, device="cuda")
        assert x.shape == (n,)

        # CPU/GPU use different linear algebra paths; tolerance reflects this
        x_cpu, z_cpu, s_cpu = _solve_fresh(n, m, P, A, cones, q, b, device="cpu")
        assert np.allclose(
            x, x_cpu, atol=5e-2
        ), f"CUDA/CPU x mismatch: max diff = {np.max(np.abs(x - x_cpu)):.2e}"

    def test_psd_dim2_cuda(self):
        """PSD(2) on CUDA."""
        n = 3
        m = 3
        P = np.eye(n)
        A = np.eye(m)
        q = np.zeros(n)
        b = np.array([1.0, 0.0, 1.0])
        cones = moreau.Cones(psd_dims=[2])

        x, z, s = _solve_fresh(n, m, P, A, cones, q, b, device="cuda")
        x_cpu, _, _ = _solve_fresh(n, m, P, A, cones, q, b, device="cpu")
        assert np.allclose(x, x_cpu, atol=5e-2)

    def test_psd_with_equality_cuda(self):
        """PSD + zero cone (equality constraints) on CUDA."""
        n = 2
        m = 4
        P = np.eye(n)
        q = np.zeros(n)
        A = np.zeros((m, n))
        A[0, 0] = 1.0
        A[0, 1] = 1.0
        A[1, 0] = 1.0
        A[3, 1] = 1.0
        b = np.array([1.0, 2.0, 0.0, 2.0])
        cones = moreau.Cones(num_zero_cones=1, psd_dims=[2])

        x, z, s = _solve_fresh(n, m, P, A, cones, q, b, device="cuda")
        assert abs(x[0] + x[1] - 1.0) < 1e-4


@pytest.mark.cuda
class TestPSDBackwardCUDA:
    """Test backward pass with PSD cones on CUDA backend."""

    def setup_method(self):
        np.random.seed(42)
        # PSD(2) is more numerically stable across backends than PSD(3)
        self.n = 3
        self.m = 3
        L = np.random.randn(self.n, self.n)
        self.P = L.T @ L + 2.0 * np.eye(self.n)
        self.A = np.eye(self.m)
        self.q = np.random.randn(self.n) * 0.1
        self.b = np.array([1.0, 0.0, 1.0])  # identity in svec for 2x2
        self.cones = moreau.Cones(psd_dims=[2])

    def test_backward_dq_cuda(self):
        np.random.seed(101)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.zeros(self.m)

        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device="cuda"
        )
        dP_cpu, dq_cpu, dA_cpu, db_cpu = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device="cpu"
        )
        assert np.allclose(
            dq, dq_cpu, atol=5e-2
        ), f"CUDA/CPU dq mismatch: max diff = {np.max(np.abs(dq - dq_cpu)):.2e}"
