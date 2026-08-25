"""
Tests for variable-dimension second-order cone (SOC) support.

SOC cones now support arbitrary dimension >= 2 via so_cone_dims.
Tests cover forward solve, batched solve, gradients, and CPU-GPU parity.
"""

import numpy as np
import pytest
import scipy.sparse as sp

import moreau
from moreau.testing import (
    random_cone_program,
    random_batch,
    sample_cone_interior,
    sample_dual_cone_interior,
)

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

requires_torch = pytest.mark.skipif(
    not (HAS_TORCH and HAS_MOREAU_TORCH),
    reason="Requires torch and moreau.torch",
)

GRADCHECK_EPS = 1e-5
GRADCHECK_ATOL = 1e-3
GRADCHECK_RTOL = 1e-2
GRADCHECK_NONDET_TOL = 1e-5

_SOLVED_STATUSES = [
    moreau.SolverStatus.Solved,
    moreau.SolverStatus.AlmostSolved,
]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestSOCValidation:
    """Test input validation for SOC cone dimensions."""

    def test_rejects_dim_less_than_2(self):
        """Cones rejects SOC dimensions < 2."""
        with pytest.raises(ValueError, match="must be >= 2"):
            moreau.Cones(so_cone_dims=[1])

    def test_rejects_dim_zero(self):
        """Cones rejects SOC dimension of 0."""
        with pytest.raises(ValueError, match="must be >= 2"):
            moreau.Cones(so_cone_dims=[0])

    def test_rejects_negative_dim(self):
        """Cones rejects negative SOC dimension."""
        with pytest.raises(ValueError, match="must be >= 2"):
            moreau.Cones(so_cone_dims=[-1])

    def test_rejects_invalid_in_list(self):
        """Cones rejects list with one invalid dimension."""
        with pytest.raises(ValueError, match="must be >= 2"):
            moreau.Cones(so_cone_dims=[3, 1, 5])

    def test_empty_so_cone_dims(self):
        """Empty so_cone_dims is valid (no SOC cones)."""
        cones = moreau.Cones(so_cone_dims=[])
        assert cones.num_so_cones == 0
        assert cones.so_cone_dims == []

    def test_no_soc_only_nonneg(self, device):
        """Forward solve with zero SOC cones (only nonneg)."""
        n = 3
        P = sp.diags([2.0, 2.0, 2.0], format="csr")
        A = sp.eye(n, format="csr")
        q = np.array([1.0, -1.0, 0.5])
        b = np.zeros(n)
        cones = moreau.Cones(num_nonneg_cones=n)
        assert cones.num_so_cones == 0

        settings = moreau.Settings(device=device, batch_size=1)
        solver = moreau.CompiledSolver(
            n=n,
            m=n,
            P_row_offsets=P.indptr.tolist(),
            P_col_indices=P.indices.tolist(),
            A_row_offsets=A.indptr.tolist(),
            A_col_indices=A.indices.tolist(),
            cones=cones,
            settings=settings,
        )
        solver.setup(P_values=P.data, A_values=A.data)
        sol = solver.solve(qs=q.reshape(1, -1), bs=b.reshape(1, -1))
        assert solver.info.status[0] in _SOLVED_STATUSES

    def test_num_so_cones_read_only(self):
        """num_so_cones is derived from so_cone_dims and not directly settable."""
        cones = moreau.Cones(so_cone_dims=[3, 5])
        assert cones.num_so_cones == 2
        with pytest.raises(AttributeError):
            cones.num_so_cones = 4


def _make_well_conditioned_soc(soc_dims, n_extra=20, seed=42):
    """Build a well-conditioned SOCP: diagonal P, identity-block A.

    Returns (P, q, A, b, cones) where cuDSS converges reliably regardless
    of SOC dimension (random_cone_program generates ill-conditioned A that
    causes cuDSS non-deterministic failures for large sparse SOC).
    """
    m = sum(soc_dims)
    n = m + n_extra
    P = sp.diags([2.0] * n, format="csr")
    A = sp.eye(m, n, format="csr")
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n) * 0.5
    # b: each SOC block needs b[0] > ||b[1:]|| for feasibility
    b = np.empty(m)
    offset = 0
    for d in soc_dims:
        b[offset] = 5.0
        b[offset + 1 : offset + d] = rng.uniform(0.1, 0.5, d - 1)
        offset += d
    cones = moreau.Cones(so_cone_dims=soc_dims)
    return P, q, A, b, cones


# ---------------------------------------------------------------------------
# Forward solve
# ---------------------------------------------------------------------------


class TestSOCVariableDimSolve:
    """Forward solve with variable-dim SOC cones."""

    @pytest.mark.parametrize(
        "dims",
        [
            [2],
            [4],
            [5],
            [10],
            [3, 5, 10],
            [2, 3, 4, 5, 6, 7],
        ],
    )
    def test_various_dims(self, device, dims):
        cones = moreau.Cones(so_cone_dims=dims)
        # n needs enough slack over m for cuDSS to converge reliably;
        # more sparse cones (dim > 4) need proportionally more slack
        num_sparse = sum(1 for d in dims if d > 4)
        n = max(sum(dims) + 12 + num_sparse * 4, 20)
        prob = random_cone_program(n=n, cones=cones, seed=300)
        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones)
        solution = solver.solve()
        assert (
            solver.info.status in _SOLVED_STATUSES
        ), f"Failed with dims={dims}, status={solver.info.status}"

    def test_mixed_cones_with_variable_soc(self, device):
        cones = moreau.Cones(
            num_zero_cones=2,
            num_nonneg_cones=3,
            so_cone_dims=[4, 6],
        )
        prob = random_cone_program(n=15, cones=cones, seed=123)
        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones)
        solution = solver.solve()
        assert solver.info.status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ]

    def test_feasibility_check(self, device):
        """Verify Ax + s = b and s in SOC."""
        cones = moreau.Cones(so_cone_dims=[5])
        prob = random_cone_program(n=10, cones=cones, seed=77)
        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones)
        solution = solver.solve()
        assert solver.info.status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ]

        # Check Ax + s ≈ b
        residual = prob.A @ solution.x + solution.s - prob.b
        np.testing.assert_allclose(residual, 0, atol=1e-6)

        # Check s in SOC: s[0] >= ||s[1:]||
        s = solution.s
        assert s[0] >= np.linalg.norm(s[1:]) - 1e-6

    @pytest.mark.parametrize("soc_dim", [50, 100])
    def test_large_dim_solve(self, device, soc_dim):
        """Forward solve with a single large SOC cone."""
        P, q, A, b, cones = _make_well_conditioned_soc([soc_dim], seed=88)
        settings = moreau.Settings(device=device, max_iter=200)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solution = solver.solve()
        assert (
            solver.info.status in _SOLVED_STATUSES
        ), f"Failed with dim={soc_dim}, status={solver.info.status}"

    def test_large_dim_feasibility(self, device):
        """Verify Ax + s = b and s in SOC for dim=100."""
        P, q, A, b, cones = _make_well_conditioned_soc([100], seed=88)
        settings = moreau.Settings(device=device, max_iter=200)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solution = solver.solve()
        assert (
            solver.info.status in _SOLVED_STATUSES
        ), f"Failed with dim=100, status={solver.info.status}"

        # Check Ax + s ~ b
        residual = A @ solution.x + solution.s - b
        np.testing.assert_allclose(residual, 0, atol=1e-6)

        # Check s in SOC: s[0] >= ||s[1:]||
        s = solution.s
        assert s[0] >= np.linalg.norm(s[1:]) - 1e-6

    def test_large_dim_mixed(self, device):
        """Forward solve with mixed large SOC dims."""
        dims = [10, 30, 50]
        P, q, A, b, cones = _make_well_conditioned_soc(dims, seed=42)
        settings = moreau.Settings(device=device, max_iter=200)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solution = solver.solve()
        assert (
            solver.info.status in _SOLVED_STATUSES
        ), f"Failed with dims={dims}, status={solver.info.status}"

    def test_backward_compat_num_so_cones(self, device):
        """num_so_cones=N still works (creates dim-3 cones)."""
        cones = moreau.Cones(num_so_cones=2)
        assert cones.so_cone_dims == [3, 3]
        prob = random_cone_program(n=10, cones=cones, seed=42)
        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones)
        solution = solver.solve()
        assert solver.info.status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ]


# ---------------------------------------------------------------------------
# Batched solve
# ---------------------------------------------------------------------------


class TestSOCVariableDimBatched:
    """Batched solve with variable-dim SOC cones."""

    @pytest.mark.parametrize(
        "dims,batch_size",
        [
            ([5], 4),
            ([3, 7], 4),
            ([3, 4, 6], 3),
        ],
    )
    def test_batched(self, device, dims, batch_size):
        cones = moreau.Cones(so_cone_dims=dims)
        n = max(sum(dims) + 15, 20)
        first, problems = random_batch(
            n=n,
            cones=cones,
            batch_size=batch_size,
            seed=99,
        )

        m = cones.total_constraints()
        solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=first.P.indptr,
            P_col_indices=first.P.indices,
            A_row_offsets=first.A.indptr,
            A_col_indices=first.A.indices,
            cones=cones,
            settings=moreau.Settings(batch_size=batch_size),
        )

        P_values = [p.P.data for p in problems]
        A_values = [p.A.data for p in problems]
        qs = [p.q for p in problems]
        bs = [p.b for p in problems]

        solver.setup(P_values, A_values)
        solution = solver.solve(qs, bs)

        for i, status in enumerate(solver.info.status):
            assert status in _SOLVED_STATUSES, f"Problem {i} failed with status {status}"

    def test_batched_mixed_cones(self, device):
        cones = moreau.Cones(
            num_zero_cones=1,
            num_nonneg_cones=2,
            so_cone_dims=[4, 6],
        )
        n = 15
        batch_size = 4
        first, problems = random_batch(
            n=n,
            cones=cones,
            batch_size=batch_size,
            seed=99,
        )

        m = cones.total_constraints()
        solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=first.P.indptr,
            P_col_indices=first.P.indices,
            A_row_offsets=first.A.indptr,
            A_col_indices=first.A.indices,
            cones=cones,
            settings=moreau.Settings(batch_size=batch_size),
        )

        solver.setup(
            [p.P.data for p in problems],
            [p.A.data for p in problems],
        )
        solution = solver.solve(
            [p.q for p in problems],
            [p.b for p in problems],
        )

        for i, status in enumerate(solver.info.status):
            assert status in [
                moreau.SolverStatus.Solved,
                moreau.SolverStatus.AlmostSolved,
            ], f"Problem {i} failed with status {status}"


# ---------------------------------------------------------------------------
# Cone interior sampling
# ---------------------------------------------------------------------------


class TestSOCVariableDimSampling:
    """Verify cone interior sampling with variable-dim SOC."""

    def test_sample_interior(self):
        cones = moreau.Cones(so_cone_dims=[2, 5, 10, 100])
        rng = np.random.default_rng(42)
        s = sample_cone_interior(cones, rng)
        assert s.shape == (2 + 5 + 10 + 100,)

        offset = 0
        for dim in [2, 5, 10, 100]:
            s_i = s[offset : offset + dim]
            norm_tail = np.linalg.norm(s_i[1:])
            assert (
                s_i[0] > norm_tail
            ), f"SOC dim={dim}: s[0]={s_i[0]} should be > ||s[1:]||={norm_tail}"
            offset += dim

    def test_sample_dual_interior(self):
        cones = moreau.Cones(so_cone_dims=[3, 7])
        rng = np.random.default_rng(42)
        z = sample_dual_cone_interior(cones, rng)
        assert z.shape == (3 + 7,)

        offset = 0
        for dim in [3, 7]:
            z_i = z[offset : offset + dim]
            norm_tail = np.linalg.norm(z_i[1:])
            assert z_i[0] > norm_tail
            offset += dim


# ---------------------------------------------------------------------------
# Gradient tests (torch.autograd.gradcheck)
# ---------------------------------------------------------------------------


def _make_soc_solver(n, soc_dim, device, extra_nonneg=0):
    """Build a small SOCP with one SOC cone for gradient testing.

    Structure:
    - n variables
    - m = extra_nonneg + soc_dim constraints
    - Diagonal P (PSD)
    - Identity-block A for the SOC rows, diagonal for nonneg
    """
    m = extra_nonneg + soc_dim
    assert n >= soc_dim, "Need n >= soc_dim for identity SOC block"

    # P: diagonal, n entries
    P_row_offsets = list(range(n + 1))
    P_col_indices = list(range(n))
    nnzP = n

    # A: first extra_nonneg rows are diagonal (one col each),
    #    then soc_dim rows mapping to first soc_dim variables
    A_row_offsets = []
    A_col_indices_list = []
    nnz = 0
    # Nonneg rows: A[i, i] = 1
    for i in range(extra_nonneg):
        A_row_offsets.append(nnz)
        A_col_indices_list.append(i)
        nnz += 1
    # SOC rows: A[extra_nonneg + j, j] = 1
    for j in range(soc_dim):
        A_row_offsets.append(nnz)
        A_col_indices_list.append(j)
        nnz += 1
    A_row_offsets.append(nnz)
    nnzA = nnz

    cones = moreau.Cones(
        num_nonneg_cones=extra_nonneg,
        so_cone_dims=[soc_dim],
    )

    settings = moreau.Settings(
        device=device,
        verbose=False,
        max_iter=200,
    )
    settings.ipm_settings.tol_gap_abs = 1e-9
    settings.ipm_settings.tol_feas = 1e-9

    solver = TorchSolver(
        n=n,
        m=m,
        P_row_offsets=torch.tensor(P_row_offsets, dtype=torch.int64),
        P_col_indices=torch.tensor(P_col_indices, dtype=torch.int64),
        A_row_offsets=torch.tensor(A_row_offsets, dtype=torch.int64),
        A_col_indices=torch.tensor(A_col_indices_list, dtype=torch.int64),
        cones=cones,
        settings=settings,
    )
    return solver, nnzP, nnzA, m


class TestSOCVariableDimGradient:
    """Gradient correctness via torch.autograd.gradcheck for variable-dim SOC."""

    @requires_torch
    @pytest.mark.parametrize("soc_dim", [4, 5, 7])
    def test_gradcheck_q(self, device, soc_dim):
        """Gradcheck w.r.t. q with SOC dim > 3."""
        n = soc_dim + 2
        solver, nnzP, nnzA, m = _make_soc_solver(n, soc_dim, device)

        P_values = torch.ones(1, n, dtype=torch.float64, device=device) * 2.0
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device)
        # b: SOC part needs s[0] > ||s[1:]||, so make b[0] large
        b_vals = [3.0] + [0.5] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device)

        q = torch.tensor(
            [[-1.0] + [0.0] * (n - 1)],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def func(q_in):
            return solver.solve(P_values, A_values, q_in, b).x

        assert torch.autograd.gradcheck(
            func,
            q,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        ), f"gradcheck failed for SOC dim={soc_dim}"

    @requires_torch
    def test_gradcheck_b(self, device):
        """Gradcheck w.r.t. b with SOC dim 5."""
        soc_dim = 5
        n = soc_dim + 2
        solver, nnzP, nnzA, m = _make_soc_solver(n, soc_dim, device)

        P_values = torch.ones(1, n, dtype=torch.float64, device=device) * 2.0
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device)
        q = torch.tensor(
            [[-1.0] + [0.0] * (n - 1)],
            dtype=torch.float64,
            device=device,
        )

        b_vals = [3.0] + [0.5] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device, requires_grad=True)

        def func(b_in):
            return solver.solve(P_values, A_values, q, b_in).x

        assert torch.autograd.gradcheck(
            func,
            b,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )

    @requires_torch
    def test_gradcheck_all_inputs(self, device):
        """Gradcheck w.r.t. all inputs with SOC dim 5."""
        soc_dim = 5
        n = soc_dim + 2
        solver, nnzP, nnzA, m = _make_soc_solver(n, soc_dim, device)

        P_values = (torch.ones(1, n, dtype=torch.float64, device=device) * 2.0).requires_grad_(True)
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device).requires_grad_(True)
        q = torch.tensor(
            [[-1.0] + [0.0] * (n - 1)],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        b_vals = [3.0] + [0.5] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device, requires_grad=True)

        def func(P_in, A_in, q_in, b_in):
            return solver.solve(P_in, A_in, q_in, b_in).x

        assert torch.autograd.gradcheck(
            func,
            (P_values, A_values, q, b),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )

    @requires_torch
    def test_gradcheck_mixed_soc_dims(self, device):
        """Gradcheck with mixed SOC dims [3, 5]."""
        soc_dims = [3, 5]
        total_soc = sum(soc_dims)
        n = total_soc + 2
        m = total_soc

        # P: diagonal
        P_row_offsets = list(range(n + 1))
        P_col_indices = list(range(n))

        # A: identity block for first total_soc columns
        A_row_offsets = list(range(m + 1))
        A_col_indices_list = list(range(m))

        cones = moreau.Cones(so_cone_dims=soc_dims)

        settings = moreau.Settings(
            device=device,
            verbose=False,
            max_iter=200,
        )
        settings.ipm_settings.tol_gap_abs = 1e-9
        settings.ipm_settings.tol_feas = 1e-9

        solver = TorchSolver(
            n=n,
            m=m,
            P_row_offsets=torch.tensor(P_row_offsets, dtype=torch.int64),
            P_col_indices=torch.tensor(P_col_indices, dtype=torch.int64),
            A_row_offsets=torch.tensor(A_row_offsets, dtype=torch.int64),
            A_col_indices=torch.tensor(A_col_indices_list, dtype=torch.int64),
            cones=cones,
            settings=settings,
        )

        P_values = torch.ones(1, n, dtype=torch.float64, device=device) * 2.0
        A_values = torch.ones(1, m, dtype=torch.float64, device=device)
        # b: each SOC block needs s[0] large
        b_list = []
        for d in soc_dims:
            b_list.extend([3.0] + [0.5] * (d - 1))
        b = torch.tensor([b_list], dtype=torch.float64, device=device)

        q = torch.randn(1, n, dtype=torch.float64, device=device, requires_grad=True)

        def func(q_in):
            return solver.solve(P_values, A_values, q_in, b).x

        assert torch.autograd.gradcheck(
            func,
            q,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )

    @requires_torch
    def test_gradcheck_z_output(self, device):
        """Gradcheck on dual variables z with SOC dim 5."""
        soc_dim = 5
        n = soc_dim + 2
        solver, nnzP, nnzA, m = _make_soc_solver(n, soc_dim, device)

        P_values = torch.ones(1, n, dtype=torch.float64, device=device) * 2.0
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device)
        b_vals = [3.0] + [0.5] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device)

        q = torch.tensor(
            [[-1.0] + [0.0] * (n - 1)],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def func(q_in):
            return solver.solve(P_values, A_values, q_in, b).z

        assert torch.autograd.gradcheck(
            func,
            q,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )


class TestSparseSocGradient:
    """Gradient correctness for sparse SOC cones (dim > 4).

    These tests exercise the backward path with the sparse expansion KKT
    structure (rank-2 decomposition) used for SOC dim > 4 on CUDA.
    Validates buffer sizes, offset calculations, and safe defaults.
    """

    @requires_torch
    @pytest.mark.parametrize("soc_dim", [10, 20])
    def test_gradcheck_q_large_soc(self, device, soc_dim):
        """Gradcheck w.r.t. q with large sparse SOC (dim > 4)."""
        n = soc_dim + 4
        solver, nnzP, nnzA, m = _make_soc_solver(n, soc_dim, device)

        P_values = torch.ones(1, n, dtype=torch.float64, device=device) * 2.0
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device)
        b_vals = [5.0] + [0.3] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device)

        q = torch.tensor(
            [[-1.0] + [0.1] * (n - 1)],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def func(q_in):
            return solver.solve(P_values, A_values, q_in, b).x

        assert torch.autograd.gradcheck(
            func,
            q,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        ), f"gradcheck failed for sparse SOC dim={soc_dim}"

    @requires_torch
    def test_gradcheck_all_inputs_large_soc(self, device):
        """Gradcheck w.r.t. all inputs (P, A, q, b) with sparse SOC dim=10."""
        soc_dim = 10
        n = soc_dim + 4
        solver, nnzP, nnzA, m = _make_soc_solver(n, soc_dim, device)

        P_values = (torch.ones(1, n, dtype=torch.float64, device=device) * 2.0).requires_grad_(True)
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device).requires_grad_(True)
        q = torch.tensor(
            [[-1.0] + [0.1] * (n - 1)],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        b_vals = [5.0] + [0.3] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device, requires_grad=True)

        def func(P_in, A_in, q_in, b_in):
            return solver.solve(P_in, A_in, q_in, b_in).x

        assert torch.autograd.gradcheck(
            func,
            (P_values, A_values, q, b),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )

    @requires_torch
    def test_gradcheck_multiple_sparse_soc(self, device):
        """Gradcheck with multiple sparse SOC cones [6, 10]."""
        soc_dims = [6, 10]
        total_soc = sum(soc_dims)
        n = total_soc + 4
        m = total_soc

        P_row_offsets = list(range(n + 1))
        P_col_indices = list(range(n))

        A_row_offsets = list(range(m + 1))
        A_col_indices_list = list(range(m))

        cones = moreau.Cones(so_cone_dims=soc_dims)

        settings = moreau.Settings(
            device=device,
            verbose=False,
            max_iter=200,
        )
        settings.ipm_settings.tol_gap_abs = 1e-9
        settings.ipm_settings.tol_feas = 1e-9

        solver = TorchSolver(
            n=n,
            m=m,
            P_row_offsets=torch.tensor(P_row_offsets, dtype=torch.int64),
            P_col_indices=torch.tensor(P_col_indices, dtype=torch.int64),
            A_row_offsets=torch.tensor(A_row_offsets, dtype=torch.int64),
            A_col_indices=torch.tensor(A_col_indices_list, dtype=torch.int64),
            cones=cones,
            settings=settings,
        )

        P_values = torch.ones(1, n, dtype=torch.float64, device=device) * 2.0
        A_values = torch.ones(1, m, dtype=torch.float64, device=device)
        # Each SOC block: b[0] large, rest small
        b_list = []
        for d in soc_dims:
            b_list.extend([5.0] + [0.3] * (d - 1))
        b = torch.tensor([b_list], dtype=torch.float64, device=device)

        q = torch.randn(1, n, dtype=torch.float64, device=device, requires_grad=True)

        def func(q_in):
            return solver.solve(P_values, A_values, q_in, b).x

        assert torch.autograd.gradcheck(
            func,
            q,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )

    @requires_torch
    def test_gradcheck_mixed_dense_sparse_soc(self, device):
        """Gradcheck with mixed dense (dim<=4) and sparse (dim>4) SOC cones [3, 8]."""
        soc_dims = [3, 8]
        total_soc = sum(soc_dims)
        n = total_soc + 4
        m = total_soc

        P_row_offsets = list(range(n + 1))
        P_col_indices = list(range(n))

        A_row_offsets = list(range(m + 1))
        A_col_indices_list = list(range(m))

        cones = moreau.Cones(so_cone_dims=soc_dims)

        settings = moreau.Settings(
            device=device,
            verbose=False,
            max_iter=200,
        )
        settings.ipm_settings.tol_gap_abs = 1e-9
        settings.ipm_settings.tol_feas = 1e-9

        solver = TorchSolver(
            n=n,
            m=m,
            P_row_offsets=torch.tensor(P_row_offsets, dtype=torch.int64),
            P_col_indices=torch.tensor(P_col_indices, dtype=torch.int64),
            A_row_offsets=torch.tensor(A_row_offsets, dtype=torch.int64),
            A_col_indices=torch.tensor(A_col_indices_list, dtype=torch.int64),
            cones=cones,
            settings=settings,
        )

        P_values = torch.ones(1, n, dtype=torch.float64, device=device) * 2.0
        A_values = torch.ones(1, m, dtype=torch.float64, device=device)
        b_list = []
        for d in soc_dims:
            b_list.extend([5.0] + [0.3] * (d - 1))
        b = torch.tensor([b_list], dtype=torch.float64, device=device)

        q = torch.randn(1, n, dtype=torch.float64, device=device, requires_grad=True)
        torch.manual_seed(42)

        def func(q_in):
            return solver.solve(P_values, A_values, q_in, b).x

        assert torch.autograd.gradcheck(
            func,
            q,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )

    @requires_torch
    def test_gradcheck_sparse_soc_with_nonneg(self, device):
        """Gradcheck with nonneg + sparse SOC cones."""
        soc_dim = 8
        extra_nonneg = 3
        n = soc_dim + extra_nonneg + 2
        solver, nnzP, nnzA, m = _make_soc_solver(
            n,
            soc_dim,
            device,
            extra_nonneg=extra_nonneg,
        )

        P_values = (torch.ones(1, n, dtype=torch.float64, device=device) * 2.0).requires_grad_(True)
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device).requires_grad_(True)
        # b: first extra_nonneg for nonneg (positive to be feasible), then SOC
        b_vals = [1.0] * extra_nonneg + [5.0] + [0.3] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device, requires_grad=True)
        q = torch.tensor(
            [[-1.0] + [0.1] * (n - 1)],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def func(P_in, A_in, q_in, b_in):
            return solver.solve(P_in, A_in, q_in, b_in).x

        assert torch.autograd.gradcheck(
            func,
            (P_values, A_values, q, b),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )

    @requires_torch
    def test_gradcheck_sparse_soc_z_output(self, device):
        """Gradcheck on dual variables z with sparse SOC dim=10."""
        soc_dim = 10
        n = soc_dim + 4
        solver, nnzP, nnzA, m = _make_soc_solver(n, soc_dim, device)

        P_values = torch.ones(1, n, dtype=torch.float64, device=device) * 2.0
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device)
        b_vals = [5.0] + [0.3] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device)

        q = torch.tensor(
            [[-1.0] + [0.1] * (n - 1)],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def func(q_in):
            return solver.solve(P_values, A_values, q_in, b).z

        assert torch.autograd.gradcheck(
            func,
            q,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )


class TestLargeSparseSocGradient:
    """Gradient correctness for large SOC cones (dim=50, 100).

    These tests verify the sparse expansion backward path works correctly
    at larger dimensions where dense storage would be expensive.
    """

    @requires_torch
    @pytest.mark.parametrize("soc_dim", [50, 100])
    def test_gradcheck_q_large(self, device, soc_dim):
        """Gradcheck w.r.t. q with large sparse SOC."""
        n = soc_dim + 4
        solver, nnzP, nnzA, m = _make_soc_solver(n, soc_dim, device)

        P_values = torch.ones(1, n, dtype=torch.float64, device=device) * 2.0
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device)
        b_vals = [5.0] + [0.3] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device)

        q = torch.tensor(
            [[-1.0] + [0.1] * (n - 1)],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def func(q_in):
            return solver.solve(P_values, A_values, q_in, b).x

        assert torch.autograd.gradcheck(
            func,
            q,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        ), f"gradcheck failed for large SOC dim={soc_dim}"

    @requires_torch
    @pytest.mark.parametrize("soc_dim", [50, 100])
    def test_gradcheck_b_large(self, device, soc_dim):
        """Gradcheck w.r.t. b with large sparse SOC."""
        n = soc_dim + 4
        solver, nnzP, nnzA, m = _make_soc_solver(n, soc_dim, device)

        P_values = torch.ones(1, n, dtype=torch.float64, device=device) * 2.0
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device)
        q = torch.tensor(
            [[-1.0] + [0.1] * (n - 1)],
            dtype=torch.float64,
            device=device,
        )
        b_vals = [5.0] + [0.3] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device, requires_grad=True)

        def func(b_in):
            return solver.solve(P_values, A_values, q, b_in).x

        assert torch.autograd.gradcheck(
            func,
            b,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        ), f"gradcheck failed for large SOC dim={soc_dim}"

    @requires_torch
    def test_gradcheck_all_inputs_dim50(self, device):
        """Gradcheck w.r.t. all inputs (P, A, q, b) with SOC dim=50."""
        soc_dim = 50
        n = soc_dim + 4
        solver, nnzP, nnzA, m = _make_soc_solver(n, soc_dim, device)

        P_values = (torch.ones(1, n, dtype=torch.float64, device=device) * 2.0).requires_grad_(True)
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device).requires_grad_(True)
        q = torch.tensor(
            [[-1.0] + [0.1] * (n - 1)],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )
        b_vals = [5.0] + [0.3] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device, requires_grad=True)

        def func(P_in, A_in, q_in, b_in):
            return solver.solve(P_in, A_in, q_in, b_in).x

        assert torch.autograd.gradcheck(
            func,
            (P_values, A_values, q, b),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )

    @requires_torch
    def test_gradcheck_multiple_large_soc(self, device):
        """Gradcheck with multiple large sparse SOC cones [20, 30]."""
        soc_dims = [20, 30]
        total_soc = sum(soc_dims)
        n = total_soc + 4
        m = total_soc

        P_row_offsets = list(range(n + 1))
        P_col_indices = list(range(n))

        A_row_offsets = list(range(m + 1))
        A_col_indices_list = list(range(m))

        cones = moreau.Cones(so_cone_dims=soc_dims)
        settings = moreau.Settings(device=device, verbose=False, max_iter=200)
        settings.ipm_settings.tol_gap_abs = 1e-9
        settings.ipm_settings.tol_feas = 1e-9

        solver = TorchSolver(
            n=n,
            m=m,
            P_row_offsets=torch.tensor(P_row_offsets, dtype=torch.int64),
            P_col_indices=torch.tensor(P_col_indices, dtype=torch.int64),
            A_row_offsets=torch.tensor(A_row_offsets, dtype=torch.int64),
            A_col_indices=torch.tensor(A_col_indices_list, dtype=torch.int64),
            cones=cones,
            settings=settings,
        )

        P_values = torch.ones(1, n, dtype=torch.float64, device=device) * 2.0
        A_values = torch.ones(1, m, dtype=torch.float64, device=device)
        b_list = []
        for d in soc_dims:
            b_list.extend([5.0] + [0.3] * (d - 1))
        b = torch.tensor([b_list], dtype=torch.float64, device=device)

        q = torch.randn(1, n, dtype=torch.float64, device=device, requires_grad=True)
        torch.manual_seed(42)

        def func(q_in):
            return solver.solve(P_values, A_values, q_in, b).x

        assert torch.autograd.gradcheck(
            func,
            q,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )

    @requires_torch
    def test_gradcheck_mixed_large_with_nonneg(self, device):
        """Gradcheck with nonneg + large sparse SOC dim=50."""
        soc_dim = 50
        extra_nonneg = 5
        n = soc_dim + extra_nonneg + 4
        solver, nnzP, nnzA, m = _make_soc_solver(
            n,
            soc_dim,
            device,
            extra_nonneg=extra_nonneg,
        )

        P_values = (torch.ones(1, n, dtype=torch.float64, device=device) * 2.0).requires_grad_(True)
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device).requires_grad_(True)
        b_vals = [1.0] * extra_nonneg + [5.0] + [0.3] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device, requires_grad=True)
        q = torch.tensor(
            [[-1.0] + [0.1] * (n - 1)],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def func(P_in, A_in, q_in, b_in):
            return solver.solve(P_in, A_in, q_in, b_in).x

        assert torch.autograd.gradcheck(
            func,
            (P_values, A_values, q, b),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )

    @requires_torch
    def test_gradcheck_z_output_dim50(self, device):
        """Gradcheck on dual variables z with large sparse SOC dim=50."""
        soc_dim = 50
        n = soc_dim + 4
        solver, nnzP, nnzA, m = _make_soc_solver(n, soc_dim, device)

        P_values = torch.ones(1, n, dtype=torch.float64, device=device) * 2.0
        A_values = torch.ones(1, nnzA, dtype=torch.float64, device=device)
        b_vals = [5.0] + [0.3] * (soc_dim - 1)
        b = torch.tensor([b_vals], dtype=torch.float64, device=device)

        q = torch.tensor(
            [[-1.0] + [0.1] * (n - 1)],
            dtype=torch.float64,
            device=device,
            requires_grad=True,
        )

        def func(q_in):
            return solver.solve(P_values, A_values, q_in, b).z

        assert torch.autograd.gradcheck(
            func,
            q,
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )


# ---------------------------------------------------------------------------
# CPU-GPU parity
# ---------------------------------------------------------------------------


class TestSOCVariableDimParity:
    """CPU vs CUDA parity for variable-dim SOC."""

    @pytest.mark.skipif(
        not moreau.device_available("cuda"),
        reason="CUDA not available",
    )
    @pytest.mark.parametrize(
        "dims",
        [
            [5],
            [3, 7],
            [2, 4, 6, 8],
            [5, 10],
        ],
    )
    def test_parity(self, dims):
        cones = moreau.Cones(so_cone_dims=dims)
        n = max(15, sum(dims) + 15)
        prob = random_cone_program(n=n, cones=cones, seed=42)

        results = {}
        for dev in ["cpu", "cuda"]:
            settings = moreau.Settings(device=dev, max_iter=200)
            solver = moreau.Solver(
                prob.P,
                prob.q,
                prob.A,
                prob.b,
                prob.cones,
                settings,
            )
            sol = solver.solve()
            assert solver.info.status in _SOLVED_STATUSES, f"Failed on {dev}: {solver.info.status}"
            results[dev] = sol

        # Sparse SOC expansion (dim > 4) uses a different KKT structure on GPU
        # vs CPU, so numerical agreement is looser than dense cones.
        has_sparse = any(d > 4 for d in dims)
        if has_sparse:
            rtol, atol = 2e-3, 1e-4
        else:
            rtol, atol = 1e-4, 1e-6

        np.testing.assert_allclose(
            results["cpu"].x,
            results["cuda"].x,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            results["cpu"].z,
            results["cuda"].z,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            results["cpu"].s,
            results["cuda"].s,
            rtol=rtol,
            atol=atol,
        )

    @pytest.mark.skipif(
        not moreau.device_available("cuda"),
        reason="CUDA not available",
    )
    @pytest.mark.parametrize(
        "dims",
        [
            [2, 5],
            [3, 7],
        ],
    )
    def test_compiled_setup_update_parity(self, dims):
        """CompiledSolver setup()+solve() parity across repeated q/b updates."""
        cones = moreau.Cones(so_cone_dims=dims)
        n = max(16, sum(dims) + 4)
        prob = random_cone_program(n=n, cones=cones, seed=456)
        m = cones.total_constraints()

        settings_cpu = moreau.Settings(device="cpu", verbose=False)
        settings_cuda = moreau.Settings(device="cuda", verbose=False)

        cpu_solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=prob.P.indptr,
            P_col_indices=prob.P.indices,
            A_row_offsets=prob.A.indptr,
            A_col_indices=prob.A.indices,
            cones=cones,
            settings=settings_cpu,
        )
        cuda_solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=prob.P.indptr,
            P_col_indices=prob.P.indices,
            A_row_offsets=prob.A.indptr,
            A_col_indices=prob.A.indices,
            cones=cones,
            settings=settings_cuda,
        )

        cpu_solver.setup(prob.P.data, prob.A.data)
        cuda_solver.setup(prob.P.data, prob.A.data)

        q2 = prob.q.copy()
        b2 = prob.b.copy()
        q3 = prob.q.copy()
        b3 = prob.b.copy()
        rng = np.random.default_rng(123)
        q3 += 0.1 * rng.normal(size=n)
        b3 += 0.1 * rng.normal(size=m)

        base_cpu = cpu_solver.solve(q2, b2)
        base_cpu_status = cpu_solver.info.status[0]
        base_cuda = cuda_solver.solve(q2, b2)
        base_cuda_status = cuda_solver.info.status[0]

        cpu_solver.setup(prob.P.data, prob.A.data)
        cuda_solver.setup(prob.P.data, prob.A.data)
        updated_cpu = cpu_solver.solve(q3, b3)
        updated_cpu_status = cpu_solver.info.status[0]
        updated_cuda = cuda_solver.solve(q3, b3)
        updated_cuda_status = cuda_solver.info.status[0]

        assert base_cpu_status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ]
        assert base_cuda_status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ]
        assert updated_cpu_status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ]
        assert updated_cuda_status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ]

        # Sparse SOC expansion (dim > 4) uses a different KKT structure on GPU
        # vs CPU, so numerical agreement is looser than dense cones.
        # Large-dim sparse SOC (dim >= 50) needs even looser tolerances.
        has_sparse = any(d > 4 for d in dims)
        has_large = any(d >= 50 for d in dims)
        if has_large:
            rtol, atol = 5e-3, 1e-3
        elif has_sparse:
            rtol, atol = 2e-3, 1e-4
        else:
            rtol, atol = 1e-4, 1e-6

        np.testing.assert_allclose(
            base_cpu.x,
            base_cuda.x,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            updated_cpu.x,
            updated_cuda.x,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            base_cpu.s,
            base_cuda.s,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            updated_cpu.s,
            updated_cuda.s,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            base_cpu.z,
            base_cuda.z,
            rtol=rtol,
            atol=atol,
        )
        np.testing.assert_allclose(
            updated_cpu.z,
            updated_cuda.z,
            rtol=rtol,
            atol=atol,
        )


# ---------------------------------------------------------------------------
# Warm start with variable-dim SOC
# ---------------------------------------------------------------------------


class TestSOCVariableDimWarmStart:
    """Warm start with variable-dim SOC cones."""

    def _settings(self, device):
        return moreau.Settings(device=device, max_iter=200)

    def test_warm_start_single_soc(self, device):
        """Warm start with a single large SOC cone."""
        P, q, A, b, cones = _make_well_conditioned_soc([50], seed=42)
        settings = self._settings(device)

        solver = moreau.Solver(P, q, A, b, cones, settings)
        sol1 = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

        # Perturb q slightly and warm start
        rng = np.random.default_rng(123)
        q2 = q + 0.05 * rng.normal(size=len(q))

        solver2 = moreau.Solver(P, q2, A, b, cones, settings)
        ws = sol1.to_warm_start()
        sol2 = solver2.solve(warm_start=ws)
        assert solver2.info.status in _SOLVED_STATUSES

    def test_warm_start_mixed_soc_dims(self, device):
        """Warm start with mixed dense/sparse SOC cones."""
        cones = moreau.Cones(so_cone_dims=[3, 5, 10])
        n = 25
        prob1 = random_cone_program(n=n, cones=cones, seed=456)
        settings = self._settings(device)

        solver = moreau.Solver(prob1.P, prob1.q, prob1.A, prob1.b, prob1.cones, settings)
        sol1 = solver.solve()
        assert solver.info.status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ]

        rng = np.random.default_rng(456)
        q2 = prob1.q + 0.05 * rng.normal(size=n)

        solver2 = moreau.Solver(prob1.P, q2, prob1.A, prob1.b, prob1.cones, settings)
        ws = sol1.to_warm_start()
        sol2 = solver2.solve(warm_start=ws)
        assert solver2.info.status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ]

    def test_warm_start_batched_variable_soc(self, device):
        """Batched warm start with variable-dim SOC cones."""
        cones = moreau.Cones(so_cone_dims=[3, 4])
        n = 15
        batch_size = 3
        first, problems = random_batch(
            n=n,
            cones=cones,
            batch_size=batch_size,
            seed=99,
        )

        m = cones.total_constraints()
        settings = moreau.Settings(batch_size=batch_size, device=device)
        solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=first.P.indptr,
            P_col_indices=first.P.indices,
            A_row_offsets=first.A.indptr,
            A_col_indices=first.A.indices,
            cones=cones,
            settings=settings,
        )

        P_values = [p.P.data for p in problems]
        A_values = [p.A.data for p in problems]
        qs = [p.q for p in problems]
        bs = [p.b for p in problems]

        solver.setup(P_values, A_values)
        sol1 = solver.solve(qs, bs)

        for i, status in enumerate(solver.info.status):
            assert status in [
                moreau.SolverStatus.Solved,
                moreau.SolverStatus.AlmostSolved,
            ], f"First solve: problem {i} failed with status {status}"

        # Warm start with perturbed q
        ws = sol1.to_warm_start()
        rng = np.random.default_rng(789)
        qs2 = [q + 0.05 * rng.normal(size=n) for q in qs]

        sol2 = solver.solve(qs2, bs, warm_start=ws)

        for i, status in enumerate(solver.info.status):
            assert status in [
                moreau.SolverStatus.Solved,
                moreau.SolverStatus.AlmostSolved,
            ], f"Warm start solve: problem {i} failed with status {status}"

    def test_warm_start_mixed_cones_with_soc(self, device):
        """Warm start with zero + nonneg + mixed-dim SOC + exp cones."""
        cones = moreau.Cones(
            num_zero_cones=2,
            num_nonneg_cones=3,
            so_cone_dims=[3, 8],
            num_exp_cones=1,
        )
        n = 30
        prob1 = random_cone_program(n=n, cones=cones, seed=42)
        settings = self._settings(device)

        solver = moreau.Solver(prob1.P, prob1.q, prob1.A, prob1.b, prob1.cones, settings)
        sol1 = solver.solve()
        assert solver.info.status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ]

        rng = np.random.default_rng(321)
        q2 = prob1.q + 0.05 * rng.normal(size=n)

        solver2 = moreau.Solver(prob1.P, q2, prob1.A, prob1.b, prob1.cones, settings)
        ws = sol1.to_warm_start()
        sol2 = solver2.solve(warm_start=ws)
        assert solver2.info.status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ], f"status={solver2.info.status}"


# ---------------------------------------------------------------------------
# Multiple SOC cones: many cones with mixed dense/sparse dims + other cones
# ---------------------------------------------------------------------------


class TestSOCMultipleConesMixed:
    """Tests with many SOC cones, mixed dense/sparse, combined with other cone types."""

    def _settings(self, device):
        return moreau.Settings(device=device, max_iter=200)

    def test_many_small_soc_cones(self, device):
        """Many dim-2 SOC cones (all dense path)."""
        dims = [2] * 20
        cones = moreau.Cones(so_cone_dims=dims)
        prob = random_cone_program(n=50, cones=cones, seed=42)
        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones, self._settings(device))
        solution = solver.solve()
        assert solver.info.status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ]

    def test_many_large_soc_cones(self, device):
        """Many sparse SOC cones (all dim > 4)."""
        dims = [5] * 10
        cones = moreau.Cones(so_cone_dims=dims)
        n = sum(dims) + 20
        prob = random_cone_program(n=n, cones=cones, seed=43)
        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones, self._settings(device))
        solution = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

    def test_mixed_dense_sparse_soc(self, device):
        """Mix of dense (dim <= 4) and sparse (dim > 4) SOC cones."""
        dims = [2, 3, 4, 5, 8]
        cones = moreau.Cones(so_cone_dims=dims)
        n = sum(dims) + 10
        prob = random_cone_program(n=n, cones=cones, seed=42)
        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones, self._settings(device))
        solution = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

    def test_all_cone_types_with_variable_soc(self, device):
        """Zero + nonneg + mixed-dim SOC + exp + power cones."""
        cones = moreau.Cones(
            num_zero_cones=3,
            num_nonneg_cones=5,
            so_cone_dims=[2, 4, 5],
            num_exp_cones=2,
            power_alphas=[0.3, 0.7],
        )
        n = cones.total_constraints() + 15
        prob = random_cone_program(n=n, cones=cones, seed=42)
        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones, self._settings(device))
        solution = solver.solve()
        assert solver.info.status in [
            moreau.SolverStatus.Solved,
            moreau.SolverStatus.AlmostSolved,
        ], f"status={solver.info.status}"

    def test_all_cone_types_batched(self, device):
        """Batched solve with all cone types + variable SOC."""
        cones = moreau.Cones(
            num_zero_cones=2,
            num_nonneg_cones=3,
            so_cone_dims=[3, 5],
            num_exp_cones=1,
        )
        n = cones.total_constraints() + 5
        batch_size = 3
        first, problems = random_batch(
            n=n,
            cones=cones,
            batch_size=batch_size,
            seed=99,
        )

        m = cones.total_constraints()
        settings = moreau.Settings(batch_size=batch_size, device=device)
        solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=first.P.indptr,
            P_col_indices=first.P.indices,
            A_row_offsets=first.A.indptr,
            A_col_indices=first.A.indices,
            cones=cones,
            settings=settings,
        )

        solver.setup(
            [p.P.data for p in problems],
            [p.A.data for p in problems],
        )
        solution = solver.solve(
            [p.q for p in problems],
            [p.b for p in problems],
        )

        for i, status in enumerate(solver.info.status):
            assert status in [
                moreau.SolverStatus.Solved,
                moreau.SolverStatus.AlmostSolved,
            ], f"Problem {i} failed with status {status}"

    @pytest.mark.skipif(
        not moreau.device_available("cuda"),
        reason="CUDA not available",
    )
    def test_mixed_dense_sparse_parity(self):
        """CPU-GPU parity with mix of dense and sparse SOC + other cones."""
        cones = moreau.Cones(
            num_nonneg_cones=4,
            so_cone_dims=[2, 3, 5],
        )
        n = cones.total_constraints() + 5
        prob = random_cone_program(n=n, cones=cones, seed=456)

        results = {}
        for dev in ["cpu", "cuda"]:
            settings = moreau.Settings(device=dev)
            solver = moreau.Solver(
                prob.P,
                prob.q,
                prob.A,
                prob.b,
                prob.cones,
                settings,
            )
            sol = solver.solve()
            assert solver.info.status in [
                moreau.SolverStatus.Solved,
                moreau.SolverStatus.AlmostSolved,
            ], f"Failed on {dev}: {solver.info.status}"
            results[dev] = sol

        # Sparse SOC expansion uses different KKT structure on CPU vs GPU
        np.testing.assert_allclose(
            results["cpu"].x,
            results["cuda"].x,
            rtol=2e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            results["cpu"].z,
            results["cuda"].z,
            rtol=2e-3,
            atol=1e-3,
        )
        np.testing.assert_allclose(
            results["cpu"].s,
            results["cuda"].s,
            rtol=2e-3,
            atol=1e-3,
        )


# ---------------------------------------------------------------------------
# CPU vs GPU backward parity (sparse expansion, dim > 4)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not moreau.device_available("cuda"),
    reason="CUDA not available",
)
class TestSparseSocBackwardParity:
    """CPU vs GPU backward gradient parity for SOC dim > 4 (sparse path)."""

    @pytest.mark.parametrize("soc_dim", [10, 20, 50, 100])
    def test_backward_dq_parity(self, soc_dim):
        """CPU and GPU dq gradients agree for sparse SOC."""
        P, q, A, b, cones = _make_well_conditioned_soc([soc_dim], seed=42)
        n = P.shape[0]

        results = {}
        for dev in ["cpu", "cuda"]:
            solver = moreau.Solver(
                P,
                q,
                A,
                b,
                cones,
                moreau.Settings(device=dev, enable_grad=True, verbose=False),
            )
            sol = solver.solve()
            assert solver.info.status in _SOLVED_STATUSES, f"Failed on {dev}: {solver.info.status}"
            dx_bar = np.ones(n)
            grads = solver.backward(dx_bar)
            results[dev] = grads

        np.testing.assert_allclose(
            results["cpu"]["dq"],
            results["cuda"]["dq"],
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"SOC dim={soc_dim} dq CPU/GPU mismatch",
        )
        np.testing.assert_allclose(
            results["cpu"]["db"],
            results["cuda"]["db"],
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"SOC dim={soc_dim} db CPU/GPU mismatch",
        )

    def test_backward_parity_mixed_sparse(self):
        """CPU and GPU backward agree with multiple sparse SOC cones."""
        P, q, A, b, cones = _make_well_conditioned_soc([6, 10, 20], seed=99)
        n = P.shape[0]

        results = {}
        for dev in ["cpu", "cuda"]:
            solver = moreau.Solver(
                P,
                q,
                A,
                b,
                cones,
                moreau.Settings(device=dev, enable_grad=True, verbose=False),
            )
            sol = solver.solve()
            assert solver.info.status in _SOLVED_STATUSES, f"Failed on {dev}: {solver.info.status}"
            dx_bar = np.ones(n)
            grads = solver.backward(dx_bar)
            results[dev] = grads

        np.testing.assert_allclose(
            results["cpu"]["dq"],
            results["cuda"]["dq"],
            rtol=1e-3,
            atol=1e-4,
            err_msg="Mixed sparse SOC dq CPU/GPU mismatch",
        )
        np.testing.assert_allclose(
            results["cpu"]["db"],
            results["cuda"]["db"],
            rtol=1e-3,
            atol=1e-4,
            err_msg="Mixed sparse SOC db CPU/GPU mismatch",
        )

    def test_backward_parity_mixed_dense_sparse_with_nonneg(self):
        """CPU and GPU backward agree with nonneg + dense + sparse SOC."""
        soc_dims = [3, 4, 10]
        m = sum(soc_dims) + 5  # 5 nonneg
        n = m + 15
        P = sp.diags([2.0] * n, format="csr")
        A = sp.eye(m, n, format="csr")
        rng = np.random.default_rng(77)
        q = rng.standard_normal(n) * 0.5
        b = np.empty(m)
        # nonneg part
        b[:5] = rng.uniform(1.0, 3.0, 5)
        # SOC part
        offset = 5
        for d in soc_dims:
            b[offset] = 5.0
            b[offset + 1 : offset + d] = rng.uniform(0.1, 0.5, d - 1)
            offset += d
        cones = moreau.Cones(num_nonneg_cones=5, so_cone_dims=soc_dims)

        results = {}
        for dev in ["cpu", "cuda"]:
            solver = moreau.Solver(
                P,
                q,
                A,
                b,
                cones,
                moreau.Settings(device=dev, enable_grad=True, verbose=False),
            )
            sol = solver.solve()
            assert solver.info.status in _SOLVED_STATUSES, f"Failed on {dev}: {solver.info.status}"
            dx_bar = np.ones(n)
            grads = solver.backward(dx_bar)
            results[dev] = grads

        np.testing.assert_allclose(
            results["cpu"]["dq"],
            results["cuda"]["dq"],
            rtol=1e-3,
            atol=1e-4,
            err_msg="Mixed nonneg+SOC dq CPU/GPU mismatch",
        )
        np.testing.assert_allclose(
            results["cpu"]["db"],
            results["cuda"]["db"],
            rtol=1e-3,
            atol=1e-4,
            err_msg="Mixed nonneg+SOC db CPU/GPU mismatch",
        )

    @pytest.mark.parametrize(
        "soc_dims",
        [
            [8, 3],  # large before small (unsorted, triggers sort permutation)
            [5, 3, 10],  # mixed unsorted order
            [10, 4, 7],  # sparse-dense-sparse unsorted
        ],
    )
    def test_backward_parity_unsorted_soc(self, soc_dims):
        """CPU and GPU backward agree with unsorted SOC dims.

        This exercises the sort permutation remapping in the backward pass:
        the solver sorts cones by dimension for warp coherence, so derivative
        data arrives in sorted order while the KKT rows follow original order.
        """
        P, q, A, b, cones = _make_well_conditioned_soc(soc_dims, seed=123)
        n = P.shape[0]

        results = {}
        for dev in ["cpu", "cuda"]:
            solver = moreau.Solver(
                P,
                q,
                A,
                b,
                cones,
                moreau.Settings(device=dev, enable_grad=True, verbose=False),
            )
            sol = solver.solve()
            assert solver.info.status in _SOLVED_STATUSES, f"Failed on {dev}: {solver.info.status}"
            dx_bar = np.ones(n)
            grads = solver.backward(dx_bar)
            results[dev] = grads

        np.testing.assert_allclose(
            results["cpu"]["dq"],
            results["cuda"]["dq"],
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Unsorted SOC {soc_dims} dq CPU/GPU mismatch",
        )
        np.testing.assert_allclose(
            results["cpu"]["db"],
            results["cuda"]["db"],
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Unsorted SOC {soc_dims} db CPU/GPU mismatch",
        )


# ---------------------------------------------------------------------------
# Infeasible / unbounded SOC problems
# ---------------------------------------------------------------------------


class TestSOCVariableDimInfeasible:
    """Infeasibility and unboundedness detection with variable-dim SOC."""

    def test_primal_infeasible_soc_dim5(self, device):
        """SOC(5) with contradictory equality: t=0.1, ||x||<=t, sum(x)=100.

        Force t to be small (t=0.1) while requiring the body components to sum
        to a huge value — primal infeasible since ||x|| <= t = 0.1 can't satisfy.
        Cone ordering: zero cones first, then SOC.
        """
        # Variables: [t, x1, x2, x3, x4]
        n = 5
        m = 2 + 5  # 2 zero (equalities) + 5 SOC
        P = sp.diags([1.0] * n, format="csr")
        A = sp.vstack(
            [
                # Zero cone rows (equalities) — must come first
                # t = 0.1
                sp.csr_matrix(([1.0], ([0], [0])), shape=(1, n)),
                # x1 + x2 + x3 + x4 = 100 (impossible under SOC with t=0.1)
                sp.csr_matrix(([1.0, 1.0, 1.0, 1.0], ([0, 0, 0, 0], [1, 2, 3, 4])), shape=(1, n)),
                # SOC constraint: -[t, x1..x4] + s = 0 => s = [t, x1..x4] in SOC
                -sp.eye(5, n, format="csr"),
            ],
            format="csr",
        )
        q = np.zeros(n)
        b = np.array([0.1, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cones = moreau.Cones(num_zero_cones=2, so_cone_dims=[5])

        settings = moreau.Settings(verbose=False, max_iter=200, device=device)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solver.solve()
        assert solver.info.status in (
            moreau.SolverStatus.PrimalInfeasible,
            moreau.SolverStatus.AlmostPrimalInfeasible,
            moreau.SolverStatus.MaxIterations,
        )

    def test_primal_infeasible_soc_dim10(self, device):
        """Larger SOC dim=10, still primal infeasible."""
        d = 10
        n = d
        m = 2 + d  # 2 equalities + SOC
        P = sp.diags([1.0] * n, format="csr")
        # Equalities first (zero cones), then SOC
        A_blocks = [
            # t = 0.1
            sp.csr_matrix(([1.0], ([0], [0])), shape=(1, n)),
            # sum(x_body) = 100
            sp.csr_matrix(
                (np.ones(d - 1), (np.zeros(d - 1, dtype=int), np.arange(1, d))), shape=(1, n)
            ),
            # SOC
            -sp.eye(d, format="csr"),
        ]
        A = sp.vstack(A_blocks, format="csr")
        q = np.zeros(n)
        b = np.zeros(m)
        b[0] = 0.1  # t = 0.1
        b[1] = 100.0  # sum(x_body) = 100 — impossible under ||x_body|| <= 0.1
        # b[2:] = 0 for SOC slack
        cones = moreau.Cones(num_zero_cones=2, so_cone_dims=[d])

        settings = moreau.Settings(verbose=False, max_iter=200, device=device)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solver.solve()
        assert solver.info.status in (
            moreau.SolverStatus.PrimalInfeasible,
            moreau.SolverStatus.AlmostPrimalInfeasible,
            moreau.SolverStatus.MaxIterations,
        )

    def test_dual_infeasible_soc_dim5(self, device):
        """Unbounded SOCP: min -t  s.t. [t, x1..x3] in SOC(4).

        P=0 (LP), no upper bound on t => dual infeasible.
        """
        d = 4
        n = d
        P = sp.csr_matrix((n, n), dtype=np.float64)
        A = -sp.eye(d, format="csr")  # s = x in SOC
        q = np.zeros(n)
        q[0] = -1.0  # minimize -t
        b = np.zeros(d)
        cones = moreau.Cones(so_cone_dims=[d])

        settings = moreau.Settings(verbose=False, max_iter=200, device=device)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solver.solve()
        assert solver.info.status in (
            moreau.SolverStatus.DualInfeasible,
            moreau.SolverStatus.AlmostDualInfeasible,
            moreau.SolverStatus.MaxIterations,
        )

    def test_dual_infeasible_soc_dim8(self, device):
        """Unbounded SOCP with larger SOC dim=8."""
        d = 8
        n = d
        P = sp.csr_matrix((n, n), dtype=np.float64)
        A = -sp.eye(d, format="csr")
        q = np.zeros(n)
        q[0] = -1.0  # minimize -t
        b = np.zeros(d)
        cones = moreau.Cones(so_cone_dims=[d])

        settings = moreau.Settings(verbose=False, max_iter=200, device=device)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solver.solve()
        assert solver.info.status in (
            moreau.SolverStatus.DualInfeasible,
            moreau.SolverStatus.AlmostDualInfeasible,
            moreau.SolverStatus.MaxIterations,
        )

    def test_infeasible_mixed_soc_nonneg(self, device):
        """Mixed SOC + nonneg where equality contradicts SOC feasibility.

        t = -1 (equality), [t, x1, x2] in SOC(3) => infeasible since SOC needs t >= 0.
        Cone ordering: zero, nonneg, SOC.
        """
        n = 3  # [t, x1, x2]
        m = 1 + 2 + 3  # zero(1) + nonneg(2) + SOC(3)
        P = sp.diags([1.0] * n, format="csr")
        A = sp.vstack(
            [
                # Zero cone: t = -1
                sp.csr_matrix(([1.0], ([0], [0])), shape=(1, 3)),
                # Nonneg: x1, x2 >= 0 => -x1 + s = 0, -x2 + s = 0
                sp.csr_matrix(([-1.0, -1.0], ([0, 1], [1, 2])), shape=(2, 3)),
                # SOC: s = [t, x1, x2] in SOC(3)
                -sp.eye(3, format="csr"),
            ],
            format="csr",
        )
        q = np.zeros(n)
        b = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2, so_cone_dims=[3])

        settings = moreau.Settings(verbose=False, max_iter=200, device=device)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solver.solve()
        assert solver.info.status in (
            moreau.SolverStatus.PrimalInfeasible,
            moreau.SolverStatus.AlmostPrimalInfeasible,
            moreau.SolverStatus.MaxIterations,
        )

    def test_infeasible_multiple_variable_soc(self, device):
        """Multiple variable-dim SOC cones with contradictory constraints.

        Cone ordering: zero first, then SOC.
        """
        n = 6
        soc_dims = [3, 3]
        m_eq = 2
        m_soc = sum(soc_dims)
        m = m_eq + m_soc

        P = sp.diags([1.0] * n, format="csr")
        A = sp.vstack(
            [
                # Zero cone rows (equalities) — must come first
                # x0 = 0.01 (tiny t for SOC1)
                sp.csr_matrix(([1.0], ([0], [0])), shape=(1, n)),
                # x1 + x2 = 100 (impossible under SOC1 with t=0.01)
                sp.csr_matrix(([1.0, 1.0], ([0, 0], [1, 2])), shape=(1, n)),
                # SOC1: s1 = [x0, x1, x2]
                sp.csr_matrix((-np.ones(3), (np.arange(3), np.arange(3))), shape=(3, n)),
                # SOC2: s2 = [x3, x4, x5]
                sp.csr_matrix((-np.ones(3), (np.arange(3), np.arange(3, 6))), shape=(3, n)),
            ],
            format="csr",
        )
        q = np.zeros(n)
        b = np.zeros(m)
        b[0] = 0.01
        b[1] = 100.0
        cones = moreau.Cones(num_zero_cones=m_eq, so_cone_dims=soc_dims)

        settings = moreau.Settings(verbose=False, max_iter=200, device=device)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solver.solve()
        assert solver.info.status in (
            moreau.SolverStatus.PrimalInfeasible,
            moreau.SolverStatus.AlmostPrimalInfeasible,
            moreau.SolverStatus.MaxIterations,
        )


# ---------------------------------------------------------------------------
# SOC boundary behavior
# ---------------------------------------------------------------------------


class TestSOCBoundaryBehavior:
    """Test behavior at or near the SOC boundary (t = ||x||)."""

    @pytest.mark.parametrize("soc_dim", [2, 3, 5, 10])
    def test_solution_on_boundary(self, device, soc_dim):
        """Push solution toward SOC boundary: min -t s.t. [t, x] in SOC, t <= 1.

        Optimal: t=1, x=0 => exactly on boundary (t = ||x|| = 0 at face).
        But with quadratic regularization: min t^2 - 2t + ||x||^2.
        """
        d = soc_dim
        n = d
        # m: SOC(d) + 1 nonneg (1 - t >= 0 i.e. t <= 1)
        m = d + 1
        P = sp.diags([2.0] * n, format="csr")
        q = np.zeros(n)
        q[0] = -2.0  # push t toward 1

        A = sp.vstack(
            [
                -sp.eye(d, format="csr"),  # SOC constraint
                sp.csr_matrix(([1.0], ([0], [0])), shape=(1, n)),  # t <= 1 via nonneg: t + s = 1
            ],
            format="csr",
        )
        b = np.zeros(m)
        b[d] = 1.0  # 1 - t = s >= 0
        cones = moreau.Cones(so_cone_dims=[d], num_nonneg_cones=1)

        settings = moreau.Settings(verbose=False, device=device)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solution = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

        # t should be close to 1, x_body close to 0
        s = np.array(solution.s)
        t_soc = s[0]  # SOC slack: t component
        x_body = s[1:d]
        assert (
            t_soc >= np.linalg.norm(x_body) - 1e-5
        ), f"SOC violated: t={t_soc}, ||x||={np.linalg.norm(x_body)}"

    @pytest.mark.parametrize("soc_dim", [3, 5, 8])
    def test_near_boundary_interior(self, device, soc_dim):
        """Solution should land in SOC interior (strict ineq) for a well-posed problem."""
        P, q, A, b, cones = _make_well_conditioned_soc([soc_dim], seed=77)
        settings = moreau.Settings(verbose=False, device=device)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solution = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

        s = np.array(solution.s)
        t = s[0]
        x_body = s[1:soc_dim]
        # Should be strictly in interior
        gap = t - np.linalg.norm(x_body)
        assert gap > -1e-6, f"SOC gap negative: {gap}"


# ---------------------------------------------------------------------------
# E2E backward (FD) for variable-dim SOC (extending test_diff_e2e patterns)
# ---------------------------------------------------------------------------


class TestSOCVariableDimE2EBackward:
    """End-to-end backward pass vs finite differences for variable-dim SOC.

    Covers dims at the dense/sparse boundary (dim=4 dense, dim=5 sparse)
    and larger dims, testing dq, db, dP, dA.
    """

    EPS_FD = 1e-5
    TOL = 1e-3
    KKT_TOL = 1e-6

    @staticmethod
    def _solve_fresh(P_dense, A_dense, cones, q, b, device):
        P_sym = 0.5 * (P_dense + P_dense.T)
        P_csr = sp.csr_matrix(P_sym)
        A_csr = sp.csr_matrix(A_dense)
        settings = moreau.Settings(device=device, enable_grad=False, verbose=False)
        settings.ipm_settings.tol_gap_abs = 1e-9
        settings.ipm_settings.tol_feas = 1e-9
        solver = moreau.Solver(P_csr, q, A_csr, b, cones, settings=settings)
        solution = solver.solve()
        return np.array(solution.x), np.array(solution.z), np.array(solution.s)

    @staticmethod
    def _backward_moreau(P_dense, A_dense, cones, q, b, dx_bar, dy_bar, ds_bar, device):
        P_sym = 0.5 * (P_dense + P_dense.T)
        P_csr = sp.csr_matrix(P_sym)
        A_csr = sp.csr_matrix(A_dense)
        settings = moreau.Settings(device=device, enable_grad=True, verbose=False)
        settings.ipm_settings.tol_gap_abs = 1e-9
        settings.ipm_settings.tol_feas = 1e-9
        solver = moreau.Solver(P_csr, q, A_csr, b, cones, settings=settings)
        solver.solve()
        grads = solver.backward(dx=dx_bar, dz=dy_bar, ds=ds_bar)
        dq = np.array(grads["dq"])
        db = np.array(grads["db"])
        dP_values = np.array(grads["dP_values"])
        dP_csr = sp.csr_matrix((dP_values, P_csr.indices, P_csr.indptr), shape=P_csr.shape)
        dA_values = np.array(grads["dA_values"])
        dA_csr = sp.csr_matrix((dA_values, A_csr.indices, A_csr.indptr), shape=A_csr.shape)
        return dP_csr.toarray(), dq, dA_csr.toarray(), db

    def _make_soc_problem(self, soc_dims, seed=42):
        """Build well-conditioned SOCP with dense P/A for FD tests."""
        m = sum(soc_dims)
        n = m + 4  # small n_extra for tractable FD
        rng = np.random.default_rng(seed)
        L = rng.standard_normal((n, n)) * 0.3
        P = L.T @ L + 2.0 * np.eye(n)
        q = rng.standard_normal(n) * 0.5
        A = rng.standard_normal((m, n))
        # Build feasible b: solve for x, then pick s in SOC interior
        x_feas = np.linalg.solve(P, -q)
        s_feas = np.zeros(m)
        offset = 0
        for d in soc_dims:
            s_feas[offset] = 4.0  # t component
            s_feas[offset + 1 : offset + d] = rng.uniform(-0.3, 0.3, d - 1)
            offset += d
        b = A @ x_feas + s_feas
        cones = moreau.Cones(so_cone_dims=soc_dims)
        return P, q, A, b, cones, n, m

    @pytest.mark.parametrize(
        "soc_dims",
        [
            [2],  # smallest valid SOC
            [4],  # largest dense path
            [5],  # smallest sparse path
            [3, 5],  # mixed dense + sparse
            [7],  # sparse
        ],
    )
    def test_backward_dq(self, device, soc_dims):
        P, q, A, b, cones, n, m = self._make_soc_problem(soc_dims, seed=700)
        rng = np.random.default_rng(701)
        dx_bar = rng.standard_normal(n)
        dy_bar = rng.standard_normal(m)
        ds_bar = rng.standard_normal(m)

        _, dq, _, _ = self._backward_moreau(P, A, cones, q, b, dx_bar, dy_bar, ds_bar, device)

        dq_fd = np.zeros(n)
        for j in range(n):
            qp = q.copy()
            qp[j] += self.EPS_FD
            qm = q.copy()
            qm[j] -= self.EPS_FD
            xp, yp, sp_ = self._solve_fresh(P, A, cones, qp, b, device)
            xm, ym, sm = self._solve_fresh(P, A, cones, qm, b, device)
            dq_fd[j] = (dx_bar @ (xp - xm) + dy_bar @ (yp - ym) + ds_bar @ (sp_ - sm)) / (
                2 * self.EPS_FD
            )

        assert np.allclose(
            dq, dq_fd, atol=self.TOL
        ), f"SOC{soc_dims} dq max err: {np.max(np.abs(dq - dq_fd)):.2e}"

    @pytest.mark.parametrize(
        "soc_dims",
        [
            [2],
            [4],
            [5],
            [3, 5],
        ],
    )
    def test_backward_db(self, device, soc_dims):
        P, q, A, b, cones, n, m = self._make_soc_problem(soc_dims, seed=710)
        rng = np.random.default_rng(711)
        dx_bar = rng.standard_normal(n)
        dy_bar = rng.standard_normal(m)
        ds_bar = rng.standard_normal(m)

        _, _, _, db = self._backward_moreau(P, A, cones, q, b, dx_bar, dy_bar, ds_bar, device)

        db_fd = np.zeros(m)
        for j in range(m):
            bp = b.copy()
            bp[j] += self.EPS_FD
            bm = b.copy()
            bm[j] -= self.EPS_FD
            xp, yp, sp_ = self._solve_fresh(P, A, cones, q, bp, device)
            xm, ym, sm = self._solve_fresh(P, A, cones, q, bm, device)
            db_fd[j] = (dx_bar @ (xp - xm) + dy_bar @ (yp - ym) + ds_bar @ (sp_ - sm)) / (
                2 * self.EPS_FD
            )

        assert np.allclose(
            db, db_fd, atol=self.TOL
        ), f"SOC{soc_dims} db max err: {np.max(np.abs(db - db_fd)):.2e}"

    @pytest.mark.parametrize(
        "soc_dims",
        [
            [2],
            [5],
        ],
    )
    def test_backward_dP(self, device, soc_dims):
        P, q, A, b, cones, n, m = self._make_soc_problem(soc_dims, seed=720)
        rng = np.random.default_rng(721)
        dx_bar = rng.standard_normal(n)
        dy_bar = rng.standard_normal(m)
        ds_bar = rng.standard_normal(m)

        dP, _, _, _ = self._backward_moreau(P, A, cones, q, b, dx_bar, dy_bar, ds_bar, device)

        dP_fd = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                E = np.zeros_like(P)
                E[i, j] = 1.0
                E[j, i] = 1.0
                xp, yp, sp_ = self._solve_fresh(P + self.EPS_FD * E, A, cones, q, b, device)
                xm, ym, sm = self._solve_fresh(P - self.EPS_FD * E, A, cones, q, b, device)
                dirderiv = (dx_bar @ (xp - xm) + dy_bar @ (yp - ym) + ds_bar @ (sp_ - sm)) / (
                    2 * self.EPS_FD
                )
                if i == j:
                    dP_fd[i, j] = dirderiv
                else:
                    dP_fd[i, j] = dirderiv / 2.0
                    dP_fd[j, i] = dirderiv / 2.0

        assert np.allclose(
            dP, dP_fd, atol=self.TOL
        ), f"SOC{soc_dims} dP max err: {np.max(np.abs(dP - dP_fd)):.2e}"


# ---------------------------------------------------------------------------
# Mixed cones E2E backward: zero + nonneg + variable SOC
# ---------------------------------------------------------------------------


class TestMixedConesWithSOCBackward:
    """E2E backward for problems mixing zero, nonneg, and variable-dim SOC."""

    EPS_FD = 1e-5
    TOL = 1e-2  # mixed-cone FD needs looser tol than single-cone

    @staticmethod
    def _solve_fresh(P_dense, A_dense, cones, q, b, device):
        P_sym = 0.5 * (P_dense + P_dense.T)
        P_csr = sp.csr_matrix(P_sym)
        A_csr = sp.csr_matrix(A_dense)
        settings = moreau.Settings(device=device, enable_grad=False, verbose=False)
        settings.ipm_settings.tol_gap_abs = 1e-9
        settings.ipm_settings.tol_feas = 1e-9
        solver = moreau.Solver(P_csr, q, A_csr, b, cones, settings=settings)
        solution = solver.solve()
        return np.array(solution.x), np.array(solution.z), np.array(solution.s)

    @staticmethod
    def _backward_moreau(P_dense, A_dense, cones, q, b, dx_bar, dy_bar, ds_bar, device):
        P_sym = 0.5 * (P_dense + P_dense.T)
        P_csr = sp.csr_matrix(P_sym)
        A_csr = sp.csr_matrix(A_dense)
        settings = moreau.Settings(device=device, enable_grad=True, verbose=False)
        settings.ipm_settings.tol_gap_abs = 1e-9
        settings.ipm_settings.tol_feas = 1e-9
        solver = moreau.Solver(P_csr, q, A_csr, b, cones, settings=settings)
        solver.solve()
        grads = solver.backward(dx=dx_bar, dz=dy_bar, ds=ds_bar)
        return np.array(grads["dq"]), np.array(grads["db"])

    def test_zero_nonneg_soc5_backward_dq(self, device):
        """Zero(2) + Nonneg(3) + SOC(5) — test dq."""
        rng = np.random.default_rng(800)
        n = 12
        m = 2 + 3 + 5  # 10
        L = rng.standard_normal((n, n)) * 0.3
        P = L.T @ L + 2.0 * np.eye(n)
        q = rng.standard_normal(n) * 0.5
        A = rng.standard_normal((m, n))
        x_feas = np.linalg.solve(P, -q)
        s_feas = np.zeros(m)
        s_feas[0:2] = 0.0  # zero cone
        s_feas[2:5] = [1.5, 2.0, 1.0]  # nonneg
        s_feas[5] = 4.0  # SOC t
        s_feas[6:10] = rng.uniform(-0.3, 0.3, 4)
        b = A @ x_feas + s_feas
        cones = moreau.Cones(num_zero_cones=2, num_nonneg_cones=3, so_cone_dims=[5])

        dx_bar = rng.standard_normal(n)
        dy_bar = rng.standard_normal(m)
        ds_bar = rng.standard_normal(m)

        dq, db = self._backward_moreau(P, A, cones, q, b, dx_bar, dy_bar, ds_bar, device)

        dq_fd = np.zeros(n)
        for j in range(n):
            qp = q.copy()
            qp[j] += self.EPS_FD
            qm = q.copy()
            qm[j] -= self.EPS_FD
            xp, yp, sp_ = self._solve_fresh(P, A, cones, qp, b, device)
            xm, ym, sm = self._solve_fresh(P, A, cones, qm, b, device)
            dq_fd[j] = (dx_bar @ (xp - xm) + dy_bar @ (yp - ym) + ds_bar @ (sp_ - sm)) / (
                2 * self.EPS_FD
            )

        assert np.allclose(
            dq, dq_fd, atol=self.TOL
        ), f"Mixed dq max err: {np.max(np.abs(dq - dq_fd)):.2e}"

    def test_nonneg_soc_mixed_dims_backward_db(self, device):
        """Nonneg(2) + SOC(3) + SOC(6) — test db."""
        rng = np.random.default_rng(810)
        n = 14
        m = 2 + 3 + 6  # 11
        L = rng.standard_normal((n, n)) * 0.3
        P = L.T @ L + 2.0 * np.eye(n)
        q = rng.standard_normal(n) * 0.5
        A = rng.standard_normal((m, n))
        x_feas = np.linalg.solve(P, -q)
        s_feas = np.zeros(m)
        s_feas[0:2] = [1.5, 2.0]  # nonneg
        s_feas[2] = 4.0  # SOC1 t
        s_feas[3:5] = [0.3, -0.2]  # SOC1 body
        s_feas[5] = 5.0  # SOC2 t
        s_feas[6:11] = rng.uniform(-0.3, 0.3, 5)  # SOC2 body
        b = A @ x_feas + s_feas
        cones = moreau.Cones(num_nonneg_cones=2, so_cone_dims=[3, 6])

        dx_bar = rng.standard_normal(n)
        dy_bar = rng.standard_normal(m)
        ds_bar = rng.standard_normal(m)

        dq, db = self._backward_moreau(P, A, cones, q, b, dx_bar, dy_bar, ds_bar, device)

        db_fd = np.zeros(m)
        for j in range(m):
            bp = b.copy()
            bp[j] += self.EPS_FD
            bm = b.copy()
            bm[j] -= self.EPS_FD
            xp, yp, sp_ = self._solve_fresh(P, A, cones, q, bp, device)
            xm, ym, sm = self._solve_fresh(P, A, cones, q, bm, device)
            db_fd[j] = (dx_bar @ (xp - xm) + dy_bar @ (yp - ym) + ds_bar @ (sp_ - sm)) / (
                2 * self.EPS_FD
            )

        assert np.allclose(
            db, db_fd, atol=self.TOL
        ), f"Mixed db max err: {np.max(np.abs(db - db_fd)):.2e}"


# ---------------------------------------------------------------------------
# JAX autodiff tests for variable-dim SOC
# ---------------------------------------------------------------------------


try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from moreau.jax import Solver as JaxSolver
    from moreau._backend import jax_available as _jax_available

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    _jax_available = lambda d: False

requires_jax = pytest.mark.skipif(not HAS_JAX, reason="Requires JAX")


def _check_grads_finite_diff(fn, args, eps=1e-5, atol=1e-3, rtol=1e-3):
    """Check gradients against finite differences."""
    grad_fn = jax.grad(fn)
    grad_analytical = grad_fn(*args)
    x = args[0]
    grad_numerical = jnp.zeros_like(x)
    for i in range(x.size):
        x_flat = x.ravel()
        x_plus = x_flat.at[i].set(x_flat[i] + eps).reshape(x.shape)
        x_minus = x_flat.at[i].set(x_flat[i] - eps).reshape(x.shape)
        f_plus = fn(x_plus)
        f_minus = fn(x_minus)
        grad_numerical = (
            grad_numerical.ravel().at[i].set((f_plus - f_minus) / (2 * eps)).reshape(x.shape)
        )
    abs_diff = jnp.abs(grad_analytical - grad_numerical)
    rel_diff = abs_diff / (jnp.abs(grad_numerical) + 1e-8)
    if not (jnp.all(abs_diff < atol) or jnp.all(rel_diff < rtol)):
        raise AssertionError(
            f"Gradient mismatch:\n"
            f"  analytical={grad_analytical}\n"
            f"  numerical={grad_numerical}\n"
            f"  abs_diff={abs_diff}\n"
            f"  rel_diff={rel_diff}"
        )


def _make_jax_soc_problem(soc_dims):
    """Create a SOC problem for JAX tests.

    Returns (n, m, P_ro, P_ci, A_ro, A_ci, cones) with
    diagonal P and -I constraint matrix for SOC(sum(dims)).
    """
    m = sum(soc_dims)
    n = m
    P_ro = np.arange(n + 1, dtype=np.int64)
    P_ci = np.arange(n, dtype=np.int64)
    A_ro = np.arange(m + 1, dtype=np.int64)
    A_ci = np.arange(m, dtype=np.int64)
    cones = moreau.Cones(so_cone_dims=soc_dims)
    return n, m, P_ro, P_ci, A_ro, A_ci, cones


@requires_jax
class TestJaxVariableDimSOC:
    """JAX integration tests for variable-dim SOC."""

    @pytest.fixture(params=["cpu", "cuda"])
    def jax_device(self, request):
        device = request.param
        if device == "cuda":
            if not _jax_available("cuda"):
                pytest.skip("CUDA JAX not available")
            try:
                jax.devices("cuda")
            except RuntimeError:
                pytest.skip("JAX CUDA backend not available")
        return device

    @pytest.mark.parametrize("soc_dims", [[5], [3, 4], [8]])
    def test_forward(self, jax_device, soc_dims):
        n, m, P_ro, P_ci, A_ro, A_ci, cones = _make_jax_soc_problem(soc_dims)
        settings = moreau.Settings(device=jax_device)
        s = JaxSolver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0] * n)
        A_data = jnp.array([-1.0] * m)
        rng = np.random.default_rng(42)
        q = jnp.array(rng.standard_normal(n) * -1.0 - 1.0)  # push into interior
        b = jnp.zeros(m)

        result = solve(P_data, A_data, q, b)
        info = s.info
        assert not jnp.any(jnp.isnan(result.x))

        # Check SOC constraints on slack
        offset = 0
        for d in soc_dims:
            s_block = result.s[offset : offset + d]
            assert s_block[0] >= jnp.linalg.norm(s_block[1:]) - 1e-5
            offset += d

    @pytest.mark.parametrize("soc_dims", [[5], [3, 5], [8]])
    def test_grad_q(self, jax_device, soc_dims):
        n, m, P_ro, P_ci, A_ro, A_ci, cones = _make_jax_soc_problem(soc_dims)
        settings = moreau.Settings(device=jax_device)
        s = JaxSolver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0] * n)
        A_data = jnp.array([-1.0] * m)
        b = jnp.zeros(m)

        def loss_fn(q):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        rng = np.random.default_rng(43)
        q = jnp.array(rng.standard_normal(n) * -1.0 - 1.0)
        grad_q = jax.grad(loss_fn)(q)
        assert not jnp.any(jnp.isnan(grad_q))
        assert grad_q.shape == q.shape

    @pytest.mark.parametrize("soc_dims", [[5], [8]])
    def test_check_grads_q(self, jax_device, soc_dims):
        """Gradient correctness via finite differences for variable-dim SOC."""
        n, m, P_ro, P_ci, A_ro, A_ci, cones = _make_jax_soc_problem(soc_dims)
        settings = moreau.Settings(device=jax_device)
        s = JaxSolver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0] * n)
        A_data = jnp.array([-1.0] * m)
        b = jnp.zeros(m)

        def loss_fn(q):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        rng = np.random.default_rng(44)
        q = jnp.array(rng.standard_normal(n) * -1.0 - 1.0)
        _check_grads_finite_diff(loss_fn, (q,), eps=1e-5, atol=1e-3, rtol=1e-3)

    def test_vmap_variable_soc(self, jax_device):
        """vmap batching with variable-dim SOC."""
        soc_dims = [5]
        n, m, P_ro, P_ci, A_ro, A_ci, cones = _make_jax_soc_problem(soc_dims)
        settings = moreau.Settings(device=jax_device)
        s = JaxSolver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        batch_size = 3
        P_data = jnp.tile(jnp.array([2.0] * n), (batch_size, 1))
        A_data = jnp.tile(jnp.array([-1.0] * m), (batch_size, 1))
        rng = np.random.default_rng(45)
        q = jnp.array(rng.standard_normal((batch_size, n)) * -1.0 - 1.0)
        b = jnp.zeros((batch_size, m))

        result_batch = jax.vmap(solve)(P_data, A_data, q, b)
        info_batch = s.info
        assert result_batch.x.shape == (batch_size, n)
        assert not jnp.any(jnp.isnan(result_batch.x))

    def test_grad_b_variable_soc(self, jax_device):
        """Gradient w.r.t. b for variable-dim SOC."""
        soc_dims = [6]
        n, m, P_ro, P_ci, A_ro, A_ci, cones = _make_jax_soc_problem(soc_dims)
        settings = moreau.Settings(device=jax_device)
        s = JaxSolver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0] * n)
        A_data = jnp.array([-1.0] * m)
        q = jnp.array([-3.0, -0.5, -0.3, -0.2, -0.1, -0.4])

        def loss_fn(b):
            result = solve(P_data, A_data, q, b)
            info = s.info
            return jnp.sum(result.x)

        b = jnp.zeros(m)
        grad_b = jax.grad(loss_fn)(b)
        assert not jnp.any(jnp.isnan(grad_b))
        assert grad_b.shape == b.shape

    def test_jit_grad_variable_soc(self, jax_device):
        """JIT + grad for variable-dim SOC."""
        soc_dims = [5]
        n, m, P_ro, P_ci, A_ro, A_ci, cones = _make_jax_soc_problem(soc_dims)
        settings = moreau.Settings(device=jax_device)
        s = JaxSolver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0] * n)
        A_data = jnp.array([-1.0] * m)
        b = jnp.zeros(m)

        @jax.jit
        def grad_loss(q):
            def loss_fn(q_):
                result = solve(P_data, A_data, q_, b)
                info = s.info
                return jnp.sum(result.x)

            return jax.grad(loss_fn)(q)

        q = jnp.array([-3.0, -0.5, -0.3, -0.2, -0.1])
        grad_q = grad_loss(q)
        assert not jnp.any(jnp.isnan(grad_q))

    def test_mixed_cones_jax(self, jax_device):
        """JAX with mixed zero + nonneg + SOC cones."""
        n = 7
        m = 1 + 2 + 4  # zero(1) + nonneg(2) + SOC(4)
        P_ro = np.arange(n + 1, dtype=np.int64)
        P_ci = np.arange(n, dtype=np.int64)
        # A: identity-like to map into constraint space
        A_ro = np.arange(m + 1, dtype=np.int64)
        A_ci = np.arange(m, dtype=np.int64)
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2, so_cone_dims=[4])

        settings = moreau.Settings(device=jax_device)
        s = JaxSolver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
        solve = s.solve

        P_data = jnp.array([2.0] * n)
        A_data = jnp.array([1.0] + [-1.0] * 2 + [-1.0] * 4)
        q = jnp.array([-1.0, -0.5, -0.3, -2.0, -0.1, -0.2, -0.1])
        b = jnp.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = solve(P_data, A_data, q, b)
        info = s.info
        assert not jnp.any(jnp.isnan(result.x))


# ---------------------------------------------------------------------------
# Dim=2 edge case (smallest valid SOC)
# ---------------------------------------------------------------------------


class TestSOCDim2EdgeCase:
    """Targeted tests for SOC dim=2 — the smallest valid SOC."""

    def test_dim2_forward(self, device):
        """SOC(2): [t, x] with t >= |x|."""
        P, q, A, b, cones = _make_well_conditioned_soc([2], seed=900)
        settings = moreau.Settings(verbose=False, device=device)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solution = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES
        s = np.array(solution.s)
        assert s[0] >= abs(s[1]) - 1e-6

    def test_dim2_many_cones(self, device):
        """20 SOC(2) cones — many small cones."""
        soc_dims = [2] * 20
        P, q, A, b, cones = _make_well_conditioned_soc(soc_dims, n_extra=10, seed=901)
        settings = moreau.Settings(verbose=False, device=device)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solution = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

    @requires_torch
    def test_dim2_gradcheck(self, device):
        """Torch gradcheck for SOC(2)."""
        soc_dims = [2]
        P, q, A, b, cones = _make_well_conditioned_soc(soc_dims, n_extra=6, seed=902)
        P_csr = sp.csr_matrix(P)
        A_csr = sp.csr_matrix(A)
        settings = moreau.Settings(device=device, verbose=False)
        ts = TorchSolver(
            n=P_csr.shape[0],
            m=A_csr.shape[0],
            P_row_offsets=torch.tensor(P_csr.indptr),
            P_col_indices=torch.tensor(P_csr.indices),
            A_row_offsets=torch.tensor(A_csr.indptr),
            A_col_indices=torch.tensor(A_csr.indices),
            cones=cones,
            settings=settings,
        )
        P_t = torch.tensor(P_csr.data, dtype=torch.float64)
        A_t = torch.tensor(A_csr.data, dtype=torch.float64)
        q_t = torch.tensor(q, dtype=torch.float64, requires_grad=True)
        b_t = torch.tensor(b, dtype=torch.float64)

        def fn(q_in):
            sol = ts.solve(P_t, A_t, q_in, b_t)
            return sol.x.sum()

        torch.autograd.gradcheck(
            fn,
            (q_t,),
            eps=GRADCHECK_EPS,
            atol=GRADCHECK_ATOL,
            rtol=GRADCHECK_RTOL,
            nondet_tol=GRADCHECK_NONDET_TOL,
        )


# ---------------------------------------------------------------------------
# CVXPY integration (native SOC dims, no reformulation)
# ---------------------------------------------------------------------------

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

requires_cvxpy = pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")


@pytest.mark.skip(reason="CVXPY adapter passes batch_size=None (pre-existing bug in cvxpy-fork)")
class TestSOCVariableDimCVXPY:
    """Test that CVXPY passes native SOC dims to Moreau."""

    @requires_cvxpy
    def test_higher_dim_soc_via_cvxpy(self, device):
        """CVXPY SOC with dim > 3 should solve natively."""
        n = 5
        x = cp.Variable(n)
        # min x[0]  s.t.  ||x[1:]|| <= x[0], x[1]=1, x[2]=0, x[3]=0, x[4]=0
        # This is a 5-dim SOC
        constraints = [
            cp.SOC(x[0], x[1:]),
            x[1] == 1,
            x[2] == 0,
            x[3] == 0,
            x[4] == 0,
        ]
        prob = cp.Problem(cp.Minimize(x[0]), constraints)

        # Solve with Clarabel for reference
        prob.solve(solver=cp.CLARABEL)
        clarabel_obj = prob.value

        # Solve with Moreau
        prob2 = cp.Problem(cp.Minimize(x[0]), constraints)
        prob2.solve(solver=cp.MOREAU)
        moreau_obj = prob2.value

        assert prob2.status == "optimal"
        np.testing.assert_allclose(moreau_obj, clarabel_obj, rtol=1e-4, atol=1e-6)

    @requires_cvxpy
    def test_norm_constraint_high_dim(self, device):
        """Norm constraint on high-dim vector uses native SOC."""
        n = 10
        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [cp.norm(x) <= 1],
        )

        prob.solve(solver=cp.CLARABEL)
        clarabel_obj = prob.value

        prob2 = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [cp.norm(x) <= 1],
        )
        prob2.solve(solver=cp.MOREAU)

        assert prob2.status == "optimal"
        np.testing.assert_allclose(prob2.value, clarabel_obj, rtol=1e-4, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
