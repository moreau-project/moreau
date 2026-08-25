"""
End-to-end differentiation tests for moreau solver (CPU + CUDA).

For each cone type (zero, nonneg, SOC, exp, power, all combined), we verify
backward pass dq, db, dA, dP against central finite differences.

Uses the unified moreau API with `device` fixture to test both CPU and CUDA.
"""

import numpy as np
import pytest
from scipy import sparse

import moreau

EPS_FD = 1e-5
TOL = 1e-3
KKT_TOL = 1e-6


def _make_solver(n, m, P_dense, A_dense, cones, device, enable_grad=True):
    """Create a moreau Solver from dense P and A matrices.

    Returns (solver, P_csr, A_csr).
    """
    P_sym = 0.5 * (P_dense + P_dense.T)
    P_csr = sparse.csr_matrix(P_sym)
    A_csr = sparse.csr_matrix(A_dense)

    settings = moreau.Settings(
        device=device,
        enable_grad=enable_grad,
        verbose=False,
    )
    settings.ipm_settings.tol_gap_abs = 1e-9
    settings.ipm_settings.tol_feas = 1e-9

    solver = moreau.Solver(P_csr, np.zeros(n), A_csr, np.zeros(m), cones, settings=settings)
    return solver, P_csr, A_csr


def _solve_fresh(n, m, P_dense, A_dense, cones, q, b, device):
    """Create a fresh solver, solve, return (x, y, s)."""
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


def _check_kkt(P, q, A, b, x, y, s, label=""):
    """Verify KKT conditions."""
    stat = P @ x + q + A.T @ y
    assert (
        np.linalg.norm(stat) < KKT_TOL
    ), f"[{label}] Stationarity: ||Px+q+A^Ty||={np.linalg.norm(stat):.2e}"
    pfeas = A @ x + s - b
    assert (
        np.linalg.norm(pfeas) < KKT_TOL
    ), f"[{label}] Primal feas: ||Ax+s-b||={np.linalg.norm(pfeas):.2e}"
    comp = abs(s @ y)
    assert comp < KKT_TOL, f"[{label}] Complementarity: s^Ty={comp:.2e}"


def _fd_backward_dq(n, m, P, A, cones, q, b, dx_bar, dy_bar, ds_bar, device):
    """FD for dq via central differences."""
    dq_fd = np.zeros(n)
    for j in range(n):
        qp = q.copy()
        qp[j] += EPS_FD
        qm = q.copy()
        qm[j] -= EPS_FD
        xp, yp, sp = _solve_fresh(n, m, P, A, cones, qp, b, device)
        xm, ym, sm = _solve_fresh(n, m, P, A, cones, qm, b, device)
        dq_fd[j] = (dx_bar @ (xp - xm) + dy_bar @ (yp - ym) + ds_bar @ (sp - sm)) / (2 * EPS_FD)
    return dq_fd


def _fd_backward_db(n, m, P, A, cones, q, b, dx_bar, dy_bar, ds_bar, device):
    """FD for db via central differences."""
    db_fd = np.zeros(m)
    for j in range(m):
        bp = b.copy()
        bp[j] += EPS_FD
        bm = b.copy()
        bm[j] -= EPS_FD
        xp, yp, sp = _solve_fresh(n, m, P, A, cones, q, bp, device)
        xm, ym, sm = _solve_fresh(n, m, P, A, cones, q, bm, device)
        db_fd[j] = (dx_bar @ (xp - xm) + dy_bar @ (yp - ym) + ds_bar @ (sp - sm)) / (2 * EPS_FD)
    return db_fd


def _fd_backward_dP(n, m, P, A, cones, q, b, dx_bar, dy_bar, ds_bar, device):
    """FD for dP via central differences."""
    dP_fd = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            E = np.zeros_like(P)
            E[i, j] = 1.0
            E[j, i] = 1.0
            xp, yp, sp = _solve_fresh(n, m, P + EPS_FD * E, A, cones, q, b, device)
            xm, ym, sm = _solve_fresh(n, m, P - EPS_FD * E, A, cones, q, b, device)
            dirderiv = (dx_bar @ (xp - xm) + dy_bar @ (yp - ym) + ds_bar @ (sp - sm)) / (2 * EPS_FD)
            if i == j:
                dP_fd[i, j] = dirderiv
            else:
                dP_fd[i, j] = dirderiv / 2.0
                dP_fd[j, i] = dirderiv / 2.0
    return dP_fd


def _backward_moreau(
    P_dense, A_dense, q, b, cones, dx_bar, dy_bar, ds_bar, device, diff_method="exact"
):
    """Run moreau backward pass and return (dP_dense, dq, dA_dense, db)."""
    P_sym = 0.5 * (P_dense + P_dense.T)
    P_csr = sparse.csr_matrix(P_sym)
    A_csr = sparse.csr_matrix(A_dense)

    settings = moreau.Settings(
        device=device,
        enable_grad=True,
        verbose=False,
        solver="ipm",
    )
    settings.ipm_settings.tol_gap_abs = 1e-9
    settings.ipm_settings.tol_feas = 1e-9
    settings.ipm_settings.diff_method = diff_method
    # Use very small mu so smoothed ≈ exact, making FD validation valid.
    settings.ipm_settings.diff_smoothing_mu = 1e-8

    solver = moreau.Solver(P_csr, q, A_csr, b, cones, settings=settings)
    solver.solve()

    grads = solver.backward(dx=dx_bar, dz=dy_bar, ds=ds_bar)

    dq = np.array(grads["dq"])
    db = np.array(grads["db"])

    # Reconstruct dense dP from sparse values
    dP_values = np.array(grads["dP_values"])
    dP_csr = sparse.csr_matrix(
        (dP_values, P_csr.indices, P_csr.indptr), shape=(P_csr.shape[0], P_csr.shape[1])
    )
    dP = dP_csr.toarray()

    # Reconstruct dense dA from sparse values
    dA_values = np.array(grads["dA_values"])
    dA_csr = sparse.csr_matrix(
        (dA_values, A_csr.indices, A_csr.indptr), shape=(A_csr.shape[0], A_csr.shape[1])
    )
    dA = dA_csr.toarray()

    return dP, dq, dA, db


# ----------------------------------------------------------------
# Test 1: Zero cone (equality constraints)
# ----------------------------------------------------------------

DIFF_METHODS_LP = ["exact", "smoothed"]


@pytest.mark.parametrize("diff_method", DIFF_METHODS_LP)
class TestZeroCone:
    """QP with equality constraints only: Ax = b (zero cone)."""

    def setup_method(self):
        np.random.seed(100)
        self.n = 4
        self.m = 2
        L = np.random.randn(self.n, self.n)
        self.P = L.T @ L + 1.0 * np.eye(self.n)
        self.q = np.random.randn(self.n)
        self.A = np.random.randn(self.m, self.n)
        x0 = np.linalg.solve(self.P, -self.q)
        self.b = self.A @ x0 + np.random.randn(self.m) * 0.1
        self.cones = moreau.Cones(num_zero_cones=self.m)

    def test_kkt(self, device, diff_method):
        x, y, s = _solve_fresh(self.n, self.m, self.P, self.A, self.cones, self.q, self.b, device)
        _check_kkt(self.P, self.q, self.A, self.b, x, y, s, "Zero")

    def test_backward_dq(self, device, diff_method):
        np.random.seed(101)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device, diff_method
        )
        dq_fd = _fd_backward_dq(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(
            dq, dq_fd, atol=TOL
        ), f"Zero dq:\n  analytic: {dq}\n  FD: {dq_fd}\n  diff: {np.abs(dq - dq_fd)}"

    def test_backward_db(self, device, diff_method):
        np.random.seed(102)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device, diff_method
        )
        db_fd = _fd_backward_db(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(
            db, db_fd, atol=TOL
        ), f"Zero db:\n  analytic: {db}\n  FD: {db_fd}\n  diff: {np.abs(db - db_fd)}"

    def test_backward_dP(self, device, diff_method):
        np.random.seed(104)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device, diff_method
        )
        dP_fd = _fd_backward_dP(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(
            dP, dP_fd, atol=TOL
        ), f"Zero dP max err: {np.max(np.abs(dP - dP_fd)):.2e}"


# ----------------------------------------------------------------
# Test 2: Nonnegative cone (x >= 0)
# ----------------------------------------------------------------


@pytest.mark.parametrize("diff_method", DIFF_METHODS_LP)
class TestNonnegCone:
    """QP with nonnegative cone: x >= 0."""

    def setup_method(self):
        np.random.seed(200)
        self.n = 4
        self.m = self.n
        L = np.random.randn(self.n, self.n)
        self.P = L.T @ L + 0.5 * np.eye(self.n)
        self.q = np.array([1.0, -3.0, 2.0, -1.0])
        self.A = -np.eye(self.n)
        self.b = np.zeros(self.n)
        self.cones = moreau.Cones(num_nonneg_cones=self.n)

    def test_kkt(self, device, diff_method):
        x, y, s = _solve_fresh(self.n, self.m, self.P, self.A, self.cones, self.q, self.b, device)
        _check_kkt(self.P, self.q, self.A, self.b, x, y, s, "Nonneg")

    def test_backward_dq(self, device, diff_method):
        np.random.seed(201)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device, diff_method
        )
        dq_fd = _fd_backward_dq(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(dq, dq_fd, atol=TOL), f"Nonneg dq:\n  analytic: {dq}\n  FD: {dq_fd}"

    def test_backward_db(self, device, diff_method):
        np.random.seed(202)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device, diff_method
        )
        db_fd = _fd_backward_db(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(db, db_fd, atol=TOL), f"Nonneg db:\n  analytic: {db}\n  FD: {db_fd}"

    def test_backward_dP(self, device, diff_method):
        np.random.seed(204)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device, diff_method
        )
        dP_fd = _fd_backward_dP(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(
            dP, dP_fd, atol=TOL
        ), f"Nonneg dP max err: {np.max(np.abs(dP - dP_fd)):.2e}"


# ----------------------------------------------------------------
# Test 3: Second-order cone
# ----------------------------------------------------------------


class TestSOCCone:
    """QP with a second-order cone constraint."""

    def setup_method(self):
        np.random.seed(300)
        self.n = 4
        self.m = 3  # SOC dim=3
        L = np.random.randn(self.n, self.n)
        self.P = L.T @ L + 0.5 * np.eye(self.n)
        self.q = np.random.randn(self.n)
        self.A = np.random.randn(self.m, self.n)
        x_feas = np.linalg.solve(self.P, -self.q)
        s_feas = np.zeros(self.m)
        s_feas[0] = 4.0
        s_feas[1:] = np.random.randn(self.m - 1) * 0.5
        self.b = self.A @ x_feas + s_feas
        self.cones = moreau.Cones(so_cone_dims=[3])

    def test_kkt(self, device):
        x, y, s = _solve_fresh(self.n, self.m, self.P, self.A, self.cones, self.q, self.b, device)
        _check_kkt(self.P, self.q, self.A, self.b, x, y, s, "SOC")

    def test_backward_dq(self, device):
        np.random.seed(301)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        dq_fd = _fd_backward_dq(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(dq, dq_fd, atol=TOL), f"SOC dq:\n  analytic: {dq}\n  FD: {dq_fd}"

    def test_backward_db(self, device):
        np.random.seed(302)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        db_fd = _fd_backward_db(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(db, db_fd, atol=TOL), f"SOC db:\n  analytic: {db}\n  FD: {db_fd}"

    def test_backward_dP(self, device):
        np.random.seed(304)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        dP_fd = _fd_backward_dP(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(dP, dP_fd, atol=TOL), f"SOC dP max err: {np.max(np.abs(dP - dP_fd)):.2e}"


# ----------------------------------------------------------------
# Test 3b: Second-order cone (dim=5, variable dimension)
# ----------------------------------------------------------------


class TestSOCConeVarDim:
    """QP with a variable-dimension second-order cone constraint (dim=5)."""

    def setup_method(self):
        np.random.seed(350)
        self.n = 6
        self.m = 5  # SOC dim=5
        L = np.random.randn(self.n, self.n)
        self.P = L.T @ L + 0.5 * np.eye(self.n)
        self.q = np.random.randn(self.n)
        self.A = np.random.randn(self.m, self.n)
        x_feas = np.linalg.solve(self.P, -self.q)
        s_feas = np.zeros(self.m)
        s_feas[0] = 4.0
        s_feas[1:] = np.random.randn(self.m - 1) * 0.5
        self.b = self.A @ x_feas + s_feas
        self.cones = moreau.Cones(so_cone_dims=[5])

    def test_kkt(self, device):
        x, y, s = _solve_fresh(self.n, self.m, self.P, self.A, self.cones, self.q, self.b, device)
        _check_kkt(self.P, self.q, self.A, self.b, x, y, s, "SOC5")

    def test_backward_dq(self, device):
        np.random.seed(351)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        dq_fd = _fd_backward_dq(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(dq, dq_fd, atol=TOL), f"SOC5 dq:\n  analytic: {dq}\n  FD: {dq_fd}"

    def test_backward_db(self, device):
        np.random.seed(352)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        db_fd = _fd_backward_db(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(db, db_fd, atol=TOL), f"SOC5 db:\n  analytic: {db}\n  FD: {db_fd}"

    def test_backward_dP(self, device):
        np.random.seed(354)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        dP_fd = _fd_backward_dP(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(
            dP, dP_fd, atol=TOL
        ), f"SOC5 dP max err: {np.max(np.abs(dP - dP_fd)):.2e}"


# ----------------------------------------------------------------
# Test 4: Exponential cone
# ----------------------------------------------------------------


class TestExpCone:
    """QP with an exponential cone constraint."""

    def setup_method(self):
        np.random.seed(400)
        self.n = 4
        self.m = 3
        L = np.random.randn(self.n, self.n)
        self.P = L.T @ L + 1.0 * np.eye(self.n)
        self.q = np.random.randn(self.n)
        self.A = np.random.randn(self.m, self.n)
        x_feas = np.linalg.solve(self.P, -self.q)
        s_feas = np.array([0.5, 1.0, 3.0])
        self.b = self.A @ x_feas + s_feas
        self.cones = moreau.Cones(num_exp_cones=1)

    def test_kkt(self, device):
        x, y, s = _solve_fresh(self.n, self.m, self.P, self.A, self.cones, self.q, self.b, device)
        _check_kkt(self.P, self.q, self.A, self.b, x, y, s, "Exp")

    def test_backward_dq(self, device):
        np.random.seed(401)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        dq_fd = _fd_backward_dq(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(dq, dq_fd, atol=TOL), f"Exp dq:\n  analytic: {dq}\n  FD: {dq_fd}"

    def test_backward_db(self, device):
        np.random.seed(402)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        db_fd = _fd_backward_db(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(db, db_fd, atol=TOL), f"Exp db:\n  analytic: {db}\n  FD: {db_fd}"

    def test_backward_dP(self, device):
        np.random.seed(404)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        dP_fd = _fd_backward_dP(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(dP, dP_fd, atol=TOL), f"Exp dP max err: {np.max(np.abs(dP - dP_fd)):.2e}"


# ----------------------------------------------------------------
# Test 5: Power cone
# ----------------------------------------------------------------


class TestPowerCone:
    """QP with a power cone constraint."""

    def setup_method(self):
        np.random.seed(500)
        self.n = 4
        self.m = 3
        self.alpha = 0.3
        L = np.random.randn(self.n, self.n)
        self.P = L.T @ L + 1.0 * np.eye(self.n)
        self.q = np.random.randn(self.n)
        self.A = np.random.randn(self.m, self.n)
        x_feas = np.linalg.solve(self.P, -self.q)
        s_feas = np.array([2.0, 2.0, 0.5])
        self.b = self.A @ x_feas + s_feas
        self.cones = moreau.Cones(power_alphas=[self.alpha])

    def test_kkt(self, device):
        x, y, s = _solve_fresh(self.n, self.m, self.P, self.A, self.cones, self.q, self.b, device)
        _check_kkt(self.P, self.q, self.A, self.b, x, y, s, "Power")

    def test_backward_dq(self, device):
        np.random.seed(501)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        dq_fd = _fd_backward_dq(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(
            dq, dq_fd, atol=TOL
        ), f"Power dq:\n  analytic: {dq}\n  FD: {dq_fd}\n  diff: {np.abs(dq - dq_fd)}"

    def test_backward_db(self, device):
        np.random.seed(502)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        db_fd = _fd_backward_db(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(
            db, db_fd, atol=TOL
        ), f"Power db:\n  analytic: {db}\n  FD: {db_fd}\n  diff: {np.abs(db - db_fd)}"

    def test_backward_dP(self, device):
        np.random.seed(504)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        dP_fd = _fd_backward_dP(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(
            dP, dP_fd, atol=TOL
        ), f"Power dP max err: {np.max(np.abs(dP - dP_fd)):.2e}"


# ----------------------------------------------------------------
# Test 6: All cones combined
# ----------------------------------------------------------------


class TestAllCones:
    """QP with zero + nonneg + SOC + exp + power cones all at once.

    Constraint structure:
      m = 2 (zero) + 3 (nonneg) + 5 (SOC dim=5) + 3 (exp) + 3 (power) = 16
    """

    def setup_method(self):
        np.random.seed(600)
        self.n = 5
        self.m = 16  # 2+3+5+3+3
        L = np.random.randn(self.n, self.n)
        self.P = L.T @ L + 2.0 * np.eye(self.n)
        self.q = np.random.randn(self.n)
        self.A = np.random.randn(self.m, self.n)

        x_feas = np.linalg.solve(self.P, -self.q)
        s_feas = np.zeros(self.m)
        s_feas[0:2] = 0.0  # zero cone
        s_feas[2:5] = np.array([1.5, 2.0, 1.0])  # nonneg
        s_feas[5] = 3.0  # SOC (dim=5): t component
        s_feas[6:10] = np.array([0.5, -0.3, 0.4, -0.2])  # SOC: x components
        s_feas[10:13] = np.array([0.5, 1.0, 3.0])  # exp
        s_feas[13:16] = np.array([2.0, 2.0, 0.5])  # power

        self.b = self.A @ x_feas + s_feas
        self.alpha = 0.4
        self.cones = moreau.Cones(
            num_zero_cones=2,
            num_nonneg_cones=3,
            so_cone_dims=[5],
            num_exp_cones=1,
            power_alphas=[self.alpha],
        )

    def test_kkt(self, device):
        x, y, s = _solve_fresh(self.n, self.m, self.P, self.A, self.cones, self.q, self.b, device)
        _check_kkt(self.P, self.q, self.A, self.b, x, y, s, "All")

    def test_backward_dq(self, device):
        np.random.seed(601)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        dq_fd = _fd_backward_dq(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(dq, dq_fd, atol=TOL), f"All dq:\n  analytic: {dq}\n  FD: {dq_fd}"

    def test_backward_db(self, device):
        np.random.seed(602)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        db_fd = _fd_backward_db(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(db, db_fd, atol=TOL), f"All db:\n  analytic: {db}\n  FD: {db_fd}"

    def test_backward_dP(self, device):
        np.random.seed(604)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        dP_fd = _fd_backward_dP(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(dP, dP_fd, atol=TOL), f"All dP max err: {np.max(np.abs(dP - dP_fd)):.2e}"


# ----------------------------------------------------------------
# Test 7: Power cone MRE (from test_power_mre.py)
# ----------------------------------------------------------------


class TestPowerConeMRE:
    """Power cone MRE with P>>0 (differentiable case).

    minimize  (1/2)x'Px + q'x
    subject to: x2 = 2, x3 = 1  (zero cone)
                (x2, x3, x1) in PowerCone(0.3)
    """

    def setup_method(self):
        self.n = 3
        self.m = 5
        self.alpha = 0.3
        np.random.seed(800)
        L = np.random.randn(self.n, self.n) * 0.3
        self.P = L.T @ L + np.eye(self.n)
        self.q = np.array([-1.0, 0.0, 0.0])
        self.A = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.b = np.array([2.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.cones = moreau.Cones(num_zero_cones=2, power_alphas=[self.alpha])

    def test_kkt(self, device):
        x, y, s = _solve_fresh(self.n, self.m, self.P, self.A, self.cones, self.q, self.b, device)
        _check_kkt(self.P, self.q, self.A, self.b, x, y, s, "PowerMRE")

    def test_backward_dq(self, device):
        np.random.seed(701)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        dq_fd = _fd_backward_dq(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(
            dq, dq_fd, atol=TOL
        ), f"PowerMRE dq:\n  analytic: {dq}\n  FD: {dq_fd}\n  diff: {np.abs(dq - dq_fd)}"

    def test_backward_db(self, device):
        np.random.seed(702)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        db_fd = _fd_backward_db(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(
            db, db_fd, atol=TOL
        ), f"PowerMRE db:\n  analytic: {db}\n  FD: {db_fd}\n  diff: {np.abs(db - db_fd)}"

    def test_backward_dP(self, device):
        np.random.seed(704)
        dx_bar = np.random.randn(self.n)
        dy_bar = np.random.randn(self.m)
        ds_bar = np.random.randn(self.m)
        dP, dq, dA, db = _backward_moreau(
            self.P, self.A, self.q, self.b, self.cones, dx_bar, dy_bar, ds_bar, device
        )
        dP_fd = _fd_backward_dP(
            self.n,
            self.m,
            self.P,
            self.A,
            self.cones,
            self.q,
            self.b,
            dx_bar,
            dy_bar,
            ds_bar,
            device,
        )
        assert np.allclose(
            dP, dP_fd, atol=TOL
        ), f"PowerMRE dP max err: {np.max(np.abs(dP - dP_fd)):.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
