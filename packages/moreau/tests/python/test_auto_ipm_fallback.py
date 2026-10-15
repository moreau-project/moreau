"""
Copyright, the Moreau authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

solver='auto' verifies active-set results and re-solves failures with the IPM.
"""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

import moreau
from moreau._dispatch import _resolve_solver_type

FALLBACK = "fell back to the IPM"


def _solve(P, q, A, b, cones, **settings):
    settings.setdefault("verbose", False)
    settings.setdefault("device", "cpu")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = moreau.Solver(
            sp.csr_matrix(np.asarray(P, float)),
            np.asarray(q, float),
            sp.csr_matrix(np.asarray(A, float)),
            np.asarray(b, float),
            cones,
            moreau.Settings(**settings),
        )
        sol = solver.solve()
    fell_back = any(FALLBACK in str(w.message) for w in caught)
    return solver, sol, fell_back


# Singular P (rank 3), feasible and bounded; active-set alone reports PrimalInfeasible.
_P_SINGULAR = np.array(
    [
        [2.16, 0.7, 0.58, 1.16],
        [0.7, 2.89, -2.65, -1.32],
        [0.58, -2.65, 3.18, 2.12],
        [1.16, -1.32, 2.12, 1.96],
    ]
)
_A_SINGULAR = np.vstack(
    [
        [
            [-1.2, -0.1, -0.6, -1.8],
            [1.0, -0.6, 0.4, -0.2],
            [-0.2, -1.7, 1.4, -0.4],
            [0.1, -0.8, 0.1, -2.2],
        ],
        np.eye(4),
        -np.eye(4),
    ]
)
_B_SINGULAR = np.r_[2.9, -1.3, 0.7, 0.3, np.full(8, 5.0)]
_Q_SINGULAR = np.array([0.2, -0.7, 2.0, 0.8])


class TestAutoResolution:
    cones = moreau.Cones(num_nonneg_cones=3)

    def _resolve(self, settings):
        return _resolve_solver_type(settings, 2, 3, self.cones, "cpu", nnz_P=2)[0]

    def test_auto_enables_fallback(self):
        settings = self._resolve(moreau.Settings())
        assert settings.solver == moreau.SolverType.ACTIVE_SET
        assert settings.active_set_settings.ipm_fallback is True

    def test_explicit_active_set_does_not_enable_fallback(self):
        settings = self._resolve(moreau.Settings(solver="active_set"))
        assert settings.solver == moreau.SolverType.ACTIVE_SET
        assert settings.active_set_settings is None

    def test_explicit_opt_out_is_respected(self):
        as_settings = moreau.ActiveSetSettings(ipm_fallback=False)
        settings = self._resolve(moreau.Settings(active_set_settings=as_settings))
        assert settings.active_set_settings.ipm_fallback is False

    @pytest.mark.parametrize(
        "kwargs",
        [{"ipm_settings": moreau.IPMSettings(diff_method="smoothed")}, {"yolo": True}],
        ids=["smoothed_diff", "yolo"],
    )
    def test_ipm_only_options_select_ipm(self, kwargs):
        # Regression for #66: active-set would silently ignore these.
        assert self._resolve(moreau.Settings(**kwargs)).solver == moreau.SolverType.IPM

    def test_ipm_tolerances_keep_active_set(self):
        # Tolerances don't change what active-set returns; they apply to any fallback.
        settings = self._resolve(moreau.Settings(ipm_settings=moreau.IPMSettings(tol_feas=1e-9)))
        assert settings.solver == moreau.SolverType.ACTIVE_SET


class TestFallbackFixesWrongAnswers:
    def test_unbounded_is_reported_dual_infeasible(self):
        # #43: active-set reported Solved with objective -1e6.
        solver, _, fell_back = _solve(
            np.diag([1.0, 0.0]), [0, -1], [[1.0, 0.0]], [1.0], moreau.Cones(num_nonneg_cones=1)
        )
        assert fell_back
        assert solver.info.status == moreau.SolverStatus.DualInfeasible

    def test_dependent_equalities_are_solved(self):
        # #44: active-set returned NumericalError.
        solver, sol, fell_back = _solve(
            2 * np.eye(2),
            [0, 0],
            [[1, 1], [2, 2], [-1, 0], [0, -1]],
            [1, 2, 0, 0],
            moreau.Cones(num_zero_cones=2, num_nonneg_cones=2),
        )
        assert fell_back
        assert solver.info.status == moreau.SolverStatus.Solved
        np.testing.assert_allclose(sol.x, [0.5, 0.5], atol=1e-7)

    def test_infeasible_result_carries_a_certificate(self):
        # #46: active-set's PrimalInfeasible came with z = 0.
        A, b = np.array([[-1.0], [1.0]]), np.array([-1.0, 0.0])
        solver, sol, fell_back = _solve(np.eye(1), [0], A, b, moreau.Cones(num_nonneg_cones=2))
        assert fell_back
        assert solver.info.status == moreau.SolverStatus.PrimalInfeasible
        assert np.all(sol.z >= -1e-9) and b @ sol.z < 0
        np.testing.assert_allclose(A.T @ sol.z, 0, atol=1e-8)

    def test_singular_P_is_solved(self):
        # #49: active-set reported PrimalInfeasible on a feasible problem.
        solver, sol, fell_back = _solve(
            _P_SINGULAR, _Q_SINGULAR, _A_SINGULAR, _B_SINGULAR, moreau.Cones(num_nonneg_cones=12)
        )
        assert fell_back
        assert solver.info.status == moreau.SolverStatus.Solved
        assert solver.info.obj_val == pytest.approx(-3.300839, abs=1e-5)
        assert np.max(_A_SINGULAR @ sol.x - _B_SINGULAR) <= 1e-8

    def test_non_kkt_solved_result_is_rejected(self):
        # #65: active-set reported Solved with an equality violated by 1.53.
        P = np.array(
            [
                [
                    9.009696376126578e-05,
                    0.0001086530014821886,
                    0.014378836510832691,
                    0.0004367647083981811,
                    0.00013050147095077557,
                    -0.26474686847802714,
                ],
                [
                    0.0001086530014821886,
                    0.00013103077216197886,
                    0.017340248544483287,
                    0.000526719153763063,
                    0.0001573790716656577,
                    -0.319273155190549,
                ],
                [
                    0.014378836510832691,
                    0.017340248544483287,
                    2.294760342347308,
                    0.06970455022657393,
                    0.020827109337408167,
                    -42.25172280707938,
                ],
                [
                    0.0004367647083981811,
                    0.000526719153763063,
                    0.06970455022657393,
                    0.002117312310408408,
                    0.0006326343810694962,
                    -1.2834182638666163,
                ],
                [
                    0.00013050147095077557,
                    0.0001573790716656577,
                    0.020827109337408167,
                    0.0006326343810694962,
                    0.00018902561428643704,
                    -0.3834741407883895,
                ],
                [
                    -0.26474686847802714,
                    -0.319273155190549,
                    -42.25172280707938,
                    -1.2834182638666163,
                    -0.3834741407883895,
                    777.9496826845909,
                ],
            ]
        )
        q = np.array(
            [
                -32.33935966176832,
                -623.9793025476926,
                47.62855790397782,
                -0.8515346209485564,
                -1.1374461750173774,
                499.329513406928,
            ]
        )
        A = np.array(
            [
                [-29.4943010608749, 0.0, 67.15792423323121, 0.0, 0.0, 0.0],
                [0.0, 496.8374074888866, 0.049062795195434133, -0.0023243069954099315, 0.0, 0.0],
                [0.0, 0.0, 6.085125815395794, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.017012878516916494, 0.0, 0.0],
                [0.0, 0.0, -0.0002846581349860103, 0.0, 1.7579573951853116, 0.0],
                [0.0, 51.933078724565945, 0.0, 0.0, 0.0, -7.22535199237225e-05],
                [0.00015231196392685966, -0.2787925916664863, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        b = np.array(
            [
                201.14136132205883,
                -127.14540441247374,
                19.080625876754212,
                -0.008491850672629704,
                4.5996744423270295,
                -13.306335298705548,
                0.4930752074221681,
            ]
        )
        solver, sol, fell_back = _solve(
            P, q, A, b, moreau.Cones(num_zero_cones=3, num_nonneg_cones=4)
        )
        assert fell_back
        assert solver.info.status == moreau.SolverStatus.Solved
        assert solver.info.obj_val == pytest.approx(221.34214599, rel=1e-7)
        np.testing.assert_allclose(A[:3] @ sol.x, b[:3], atol=1e-6)


def test_well_posed_problem_does_not_fall_back():
    solver, sol, fell_back = _solve(
        2 * np.eye(2),
        [1, 1],
        [[1, 1], [1, 0], [0, 1]],
        [1, 0.7, 0.7],
        moreau.Cones(num_zero_cones=1, num_nonneg_cones=2),
    )
    assert solver._settings.solver == moreau.SolverType.ACTIVE_SET
    assert not fell_back
    assert solver.info.status == moreau.SolverStatus.Solved
    np.testing.assert_allclose(sol.x, [0.5, 0.5], atol=1e-9)


def test_well_posed_batch_does_not_fall_back():
    rng = np.random.default_rng(0)
    n, m, batch = 10, 20, 64
    pattern_P, pattern_A = sp.csr_matrix(np.ones((n, n))), sp.csr_matrix(np.ones((m, n)))
    Pv, Av, qs, bs = [], [], [], []
    for _ in range(batch):
        F = rng.normal(size=(n, n))
        A = rng.normal(size=(m, n))
        Pv.append((F.T @ F / n + np.eye(n)).ravel())
        Av.append(A.ravel())
        qs.append(rng.normal(size=n))
        bs.append(A @ rng.normal(size=n) + rng.uniform(0.1, 1.0, m))
    solver = moreau.CompiledSolver(
        n=n,
        m=m,
        P_row_offsets=pattern_P.indptr,
        P_col_indices=pattern_P.indices,
        A_row_offsets=pattern_A.indptr,
        A_col_indices=pattern_A.indices,
        cones=moreau.Cones(num_nonneg_cones=m),
        settings=moreau.Settings(device="cpu", batch_size=batch, verbose=False),
    )
    solver.setup(np.array(Pv), np.array(Av))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        solver.solve(np.array(qs), np.array(bs))
    assert all(st == moreau.SolverStatus.Solved for st in solver.info.status)


def test_explicit_active_set_keeps_its_result():
    # Fallback is opt-in for an explicit solver='active_set'.
    _, _, fell_back = _solve(
        np.diag([1.0, 0.0]),
        [0, -1],
        [[1.0, 0.0]],
        [1.0],
        moreau.Cones(num_nonneg_cones=1),
        solver="active_set",
    )
    assert not fell_back
    solver, _, fell_back = _solve(
        np.diag([1.0, 0.0]),
        [0, -1],
        [[1.0, 0.0]],
        [1.0],
        moreau.Cones(num_nonneg_cones=1),
        solver="active_set",
        active_set_settings=moreau.ActiveSetSettings(ipm_fallback=True),
    )
    assert fell_back
    assert solver.info.status == moreau.SolverStatus.DualInfeasible


def _mixed_batch():
    """Problem 0 falls back (singular P); problem 1 (P + 3I) stays on active-set."""
    pattern_P, pattern_A = sp.csr_matrix(np.ones((4, 4))), sp.csr_matrix(_A_SINGULAR)
    Pv = np.stack([_P_SINGULAR.ravel(), (_P_SINGULAR + 3 * np.eye(4)).ravel()])
    Av = np.stack([pattern_A.data, pattern_A.data])
    return pattern_P, pattern_A, Pv, Av, np.stack([_Q_SINGULAR] * 2), np.stack([_B_SINGULAR] * 2)


def _compiled_grads(solver_name):
    pattern_P, pattern_A, Pv, Av, qs, bs = _mixed_batch()
    solver = moreau.CompiledSolver(
        n=4,
        m=12,
        P_row_offsets=pattern_P.indptr,
        P_col_indices=pattern_P.indices,
        A_row_offsets=pattern_A.indptr,
        A_col_indices=pattern_A.indices,
        cones=moreau.Cones(num_nonneg_cones=12),
        settings=moreau.Settings(
            device="cpu", solver=solver_name, batch_size=2, enable_grad=True, verbose=False
        ),
    )
    solver.setup(Pv, Av)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver.solve(qs, bs)
    w = np.array([0.3, -1.1, 0.7, 0.2])
    grads = solver.backward(np.stack([w, w]))
    return [st for st in solver.info.status], grads


def test_batched_fallback_routes_gradients():
    statuses, auto = _compiled_grads("auto")
    _, ipm = _compiled_grads("ipm")
    _, active_set = _compiled_grads("active_set")
    assert statuses == [moreau.SolverStatus.Solved] * 2
    for key in ("dP_values", "dA_values", "dq", "db"):
        # The fallback problem gets the IPM gradient ...
        np.testing.assert_allclose(auto[key][0], ipm[key][0], atol=1e-7, err_msg=key)
        # ... and the other keeps the active-set gradient exactly.
        np.testing.assert_array_equal(auto[key][1], active_set[key][1], err_msg=key)


def test_torch_fallback_routes_gradients():
    torch = pytest.importorskip("torch")
    from moreau.torch import Solver

    pattern_P, pattern_A, Pv, Av, qs, bs = _mixed_batch()
    w = torch.tensor([0.3, -1.1, 0.7, 0.2], dtype=torch.float64)

    def grads(solver_name):
        solver = Solver(
            4,
            12,
            torch.tensor(pattern_P.indptr),
            torch.tensor(pattern_P.indices),
            torch.tensor(pattern_A.indptr),
            torch.tensor(pattern_A.indices),
            moreau.Cones(num_nonneg_cones=12),
            settings=moreau.Settings(device="cpu", solver=solver_name),
        )
        q = torch.tensor(qs, requires_grad=True)
        b = torch.tensor(bs, requires_grad=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = solver.solve(torch.tensor(Pv), torch.tensor(Av), q, b).x
        (x @ w).sum().backward()
        return q.grad.numpy(), b.grad.numpy()

    auto, ipm, active_set = grads("auto"), grads("ipm"), grads("active_set")
    for a, i, s in zip(auto, ipm, active_set):
        np.testing.assert_allclose(a[0], i[0], atol=1e-9)
        np.testing.assert_array_equal(a[1], s[1])


def test_jax_fallback_routes_gradients():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from moreau.jax import Solver

    previous_x64 = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        pattern_P, pattern_A, Pv, Av, qs, bs = _mixed_batch()
        w = jnp.array([0.3, -1.1, 0.7, 0.2])

        def grads(solver_name):
            solver = Solver(
                4,
                12,
                pattern_P.indptr,
                pattern_P.indices,
                pattern_A.indptr,
                pattern_A.indices,
                moreau.Cones(num_nonneg_cones=12),
                settings=moreau.Settings(device="cpu", solver=solver_name),
            )

            def loss(q, b):
                return (solver.solve(jnp.array(Pv), jnp.array(Av), q, b).x @ w).sum()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return jax.grad(loss, argnums=(0, 1))(jnp.array(qs), jnp.array(bs))

        auto, ipm, active_set = grads("auto"), grads("ipm"), grads("active_set")
        for a, i, s in zip(auto, ipm, active_set):
            np.testing.assert_allclose(np.asarray(a)[0], np.asarray(i)[0], atol=1e-9)
            np.testing.assert_array_equal(np.asarray(a)[1], np.asarray(s)[1])
    finally:
        jax.config.update("jax_enable_x64", previous_x64)


def test_cvxpy_default_options():
    cp = pytest.importorskip("cvxpy")
    if "MOREAU" not in cp.installed_solvers():
        pytest.skip("cvxpy's MOREAU interface is not installed")
    x = cp.Variable(2)
    unbounded = cp.Problem(cp.Minimize(0.5 * cp.square(x[0]) - x[1]), [x[0] <= 1])
    y = cp.Variable(3)
    duplicated = cp.Problem(cp.Minimize(cp.sum_squares(y)), [cp.sum(y) == 1, cp.sum(y) == 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unbounded.solve(solver=cp.MOREAU)
        duplicated.solve(solver=cp.MOREAU)
    assert unbounded.status == "unbounded"
    assert duplicated.status == "optimal"
    assert duplicated.value == pytest.approx(1 / 3, abs=1e-7)
