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

Bad inputs raise clear errors instead of crashing or returning a wrong "Solved".
"""

import numpy as np
import pydantic
import pytest
import scipy.sparse as sp

import moreau

P = sp.csr_array(np.array([[2.0, 1.0], [1.0, 2.0]]))
Q = np.array([1.0, 1.0])
A = sp.csr_array(np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
B = np.array([1.0, 0.7, 0.7])
CONES = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
SOLVERS = ["auto", "ipm"]


def _solver(P_=P, q=Q, A_=A, b=B, cones=CONES, **settings):
    settings.setdefault("device", "cpu")
    settings.setdefault("verbose", False)
    return moreau.Solver(P_, q, A_, b, cones, moreau.Settings(**settings))


def _raw_csr(data, indices, indptr, shape):
    # Build a CSR without scipy's constructor-time checks, as a caller could.
    matrix = sp.csr_array(shape)
    matrix.data, matrix.indices, matrix.indptr = data, indices, indptr
    return matrix


# ---------------------------------------------------------------------------
# #60: malformed and duplicate sparse structure
# ---------------------------------------------------------------------------

MALFORMED_A = {
    "non_monotone_indptr": (np.ones(3), np.array([0, 1, 0]), np.array([0, 2, 1, 3])),
    "column_out_of_range": (np.ones(3), np.array([0, 7, 0]), np.array([0, 1, 2, 3])),
    "negative_column": (np.ones(3), np.array([0, -1, 0]), np.array([0, 1, 2, 3])),
}


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("case", list(MALFORMED_A))
def test_solver_rejects_malformed_csr(case, solver):
    bad = _raw_csr(*MALFORMED_A[case], shape=(3, 2))
    with pytest.raises(ValueError, match="A is not a valid CSR matrix"):
        _solver(A_=bad, solver=solver)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_rejects_duplicate_entries(solver):
    # Same matrix as A, but row 0 stores column 0 twice (0.5 + 0.5).
    duplicated = sp.csr_array(
        (np.array([0.5, 0.5, 1.0, 1.0, 1.0]), np.array([0, 0, 1, 0, 1]), np.array([0, 3, 4, 5])),
        shape=(3, 2),
    )
    with pytest.raises(ValueError, match=r"duplicate entries.*sum_duplicates"):
        _solver(A_=duplicated, solver=solver)
    duplicated.sum_duplicates()
    sol = _solver(A_=duplicated, solver=solver).solve()
    np.testing.assert_allclose(sol.x, [0.5, 0.5], atol=1e-7)


def _structure_kwargs(A_ro, A_ci):
    return dict(
        n=2,
        m=3,
        P_row_offsets=np.array([0, 2, 4]),
        P_col_indices=np.array([0, 1, 0, 1]),
        A_row_offsets=A_ro,
        A_col_indices=A_ci,
        cones=CONES,
    )


BAD_STRUCTURE = {
    "column_out_of_range": (np.array([0, 2, 3, 4]), np.array([0, 5, 0, 1]), "invalid column"),
    "non_monotone_offsets": (np.array([0, 2, 1, 4]), np.array([0, 1, 0, 1]), "monotonically"),
    "duplicate_entry": (np.array([0, 2, 3, 4]), np.array([0, 0, 0, 1]), "duplicate entry"),
}


@pytest.mark.parametrize("case", list(BAD_STRUCTURE))
@pytest.mark.parametrize("interface", ["compiled", "torch", "jax"])
def test_raw_csr_interfaces_validate_structure(interface, case):
    A_ro, A_ci, message = BAD_STRUCTURE[case]
    kwargs = _structure_kwargs(A_ro, A_ci)
    if interface == "compiled":
        make = moreau.CompiledSolver
        kwargs["settings"] = moreau.Settings(device="cpu")
    elif interface == "torch":
        torch = pytest.importorskip("torch")
        make = pytest.importorskip("moreau.torch").Solver
        kwargs = {
            k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
        }
    else:
        make = pytest.importorskip("moreau.jax").Solver
    with pytest.raises(ValueError, match=message):
        make(**kwargs)


def test_unsorted_csr_keeps_caller_order():
    # Reversed column order within each row: gradients must line up with P.data.
    unsorted = P.copy()
    for start, end in zip(unsorted.indptr[:-1], unsorted.indptr[1:]):
        unsorted.indices[start:end] = unsorted.indices[start:end][::-1]
        unsorted.data[start:end] = unsorted.data[start:end][::-1]
    solver = _solver(P_=unsorted, solver="ipm", enable_grad=True)
    solver.solve()
    unsorted_grad = solver.backward(np.array([0.4, -0.8]))["dP_values"].ravel()
    reference = _solver(solver="ipm", enable_grad=True)
    reference.solve()
    sorted_grad = reference.backward(np.array([0.4, -0.8]))["dP_values"].ravel()
    # Each row has two entries, reversed: the gradient must be reversed per row too.
    np.testing.assert_allclose(unsorted_grad, sorted_grad.reshape(2, 2)[:, ::-1].ravel(), atol=1e-9)


# ---------------------------------------------------------------------------
# #61: NaN and Inf in problem data
# ---------------------------------------------------------------------------

NON_FINITE = {
    "q_nan": dict(q=np.array([np.nan, 1.0])),
    "q_inf": dict(q=np.array([np.inf, 1.0])),
    "P_nan": dict(P_=sp.csr_array(np.array([[np.nan, 1.0], [1.0, 2.0]]))),
    "A_nan": dict(A_=sp.csr_array(np.array([[np.nan, 1.0], [1.0, 0.0], [0.0, 1.0]]))),
    "b_nan": dict(b=np.array([np.nan, 0.7, 0.7])),
}


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("case", list(NON_FINITE))
def test_solver_rejects_non_finite_data(case, solver):
    with pytest.raises(ValueError, match="NaN"):
        _solver(solver=solver, **NON_FINITE[case])


def test_inf_in_b_is_still_allowed():
    # +inf on a nonneg row is an absent bound, handled by the solver.
    solver = _solver(b=np.array([1.0, np.inf, 0.7]), solver="ipm")
    solver.solve()
    assert solver.info.status == moreau.SolverStatus.Solved


def test_compiled_solver_rejects_non_finite_values():
    solver = moreau.CompiledSolver(
        **_structure_kwargs(A.indptr, A.indices), settings=moreau.Settings(device="cpu")
    )
    with pytest.raises(ValueError, match="P_values contains NaN"):
        solver.setup(np.array([np.nan, 1.0, 1.0, 2.0]), A.data)
    solver.setup(P.data, A.data)
    with pytest.raises(ValueError, match="qs contains NaN"):
        solver.solve(np.array([np.nan, 1.0]), B)
    with pytest.raises(ValueError, match="bs contains NaN"):
        solver.solve(Q, np.array([np.nan, 0.7, 0.7]))


def test_torch_cpu_rejects_non_finite_values():
    torch = pytest.importorskip("torch")
    from moreau.torch import Solver

    kwargs = _structure_kwargs(A.indptr, A.indices)
    solver = Solver(
        **{k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
    )
    f64 = lambda a: torch.tensor(a, dtype=torch.float64)  # noqa: E731
    with pytest.raises(ValueError, match="q contains NaN"):
        solver.solve(f64(P.data), f64(A.data), f64([np.nan, 1.0]), f64(B))


def test_jax_cpu_rejects_non_finite_values():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    from moreau.jax import Solver

    solver = Solver(
        **_structure_kwargs(A.indptr, A.indices), settings=moreau.Settings(device="cpu")
    )
    with pytest.raises(Exception, match="q contains NaN"):
        jax.block_until_ready(
            solver.solve(
                jnp.asarray(P.data), jnp.asarray(A.data), jnp.array([jnp.nan, 1.0]), jnp.asarray(B)
            ).x
        )


# ---------------------------------------------------------------------------
# #62: misspelled settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda: moreau.Settings(max_iters=3),
        lambda: moreau.IPMSettings(tol_feasibility=1e-3),
        lambda: moreau.ActiveSetSettings(iter_limt=5),
    ],
    ids=["Settings", "IPMSettings", "ActiveSetSettings"],
)
def test_settings_reject_unknown_fields(make):
    with pytest.raises(pydantic.ValidationError, match="Extra inputs are not permitted"):
        make()


# ---------------------------------------------------------------------------
# #63: inputs are copied, not aliased
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", SOLVERS)
def test_solver_copies_inputs(solver):
    P_ = sp.csr_array(np.eye(2))
    q = np.ones(2)
    s = moreau.Solver(
        P_,
        q,
        sp.csr_array(np.eye(2)),
        np.full(2, 10.0),
        moreau.Cones(num_nonneg_cones=2),
        moreau.Settings(device="cpu", solver=solver, verbose=False),
    )
    P_.data *= 4
    q *= 2
    np.testing.assert_allclose(s.solve().x, [-1.0, -1.0], atol=1e-7)


@pytest.mark.parametrize("solver", SOLVERS)
def test_compiled_solver_setup_copies_values(solver):
    compiled = moreau.CompiledSolver(
        n=2,
        m=2,
        P_row_offsets=[0, 1, 2],
        P_col_indices=[0, 1],
        A_row_offsets=[0, 1, 2],
        A_col_indices=[0, 1],
        cones=moreau.Cones(num_nonneg_cones=2),
        settings=moreau.Settings(device="cpu", solver=solver, verbose=False),
    )
    P_values = np.ones(2)
    compiled.setup(P_values, np.ones(2))
    P_values *= 4
    np.testing.assert_allclose(
        compiled.solve(np.ones(2), np.full(2, 10.0)).x.ravel(), [-1.0, -1.0], atol=1e-7
    )


# ---------------------------------------------------------------------------
# #64: clearer errors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["riccati", "cudss", "woodbury"])
def test_cuda_only_methods_on_cpu(method):
    with pytest.raises(ValueError, match="only available on CUDA"):
        _solver(solver="ipm", ipm_settings=moreau.IPMSettings(direct_solve_method=method)).solve()


@pytest.mark.parametrize("cones", [None, {"num_zero_cones": 1}], ids=["None", "dict"])
def test_cones_type_is_checked(cones):
    with pytest.raises(TypeError, match="cones must be a moreau.Cones"):
        _solver(cones=cones)


@pytest.mark.parametrize(
    "q", [np.array([1 + 5j, 1.0]), np.array(["1", "1"])], ids=["complex", "string"]
)
def test_non_real_inputs_are_rejected(q):
    with pytest.raises(TypeError, match="q must be a real numeric array"):
        _solver(q=q)
