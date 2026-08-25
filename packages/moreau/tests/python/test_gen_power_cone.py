"""
Tests for generalized power cone (GenPowerCone) support.

GenPowerCone: K = {(p,w) : prod(p_i^alpha_i) >= ||w||_2, p_i >= 0}
where alpha_i > 0, sum(alpha_i) = 1.

Tests cover:
- Forward solve (CPU, CUDA, both devices)
- Backward pass / gradient correctness (torch.autograd.gradcheck)
- Warm starting
- CPU/GPU parity (forward + backward)
- Batched solves
- Variable dimensions (different dim1, dim2 combos)
- Mixed cones (GenPowerCone + other cone types)
"""

import numpy as np
import pytest
from scipy import sparse

import moreau
from moreau.testing import random_cone_program, random_batch

try:
    import torch

    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    HAS_CUDA = False

try:
    from moreau.torch import Solver as TorchSolver

    HAS_MOREAU_TORCH = TorchSolver is not None
except ImportError:
    HAS_MOREAU_TORCH = False
    TorchSolver = None

_SOLVED_STATUSES = {moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved}

# Gradcheck parameters
GRADCHECK_EPS = 1e-5
GRADCHECK_ATOL = 1e-3
GRADCHECK_RTOL = 1e-2
GRADCHECK_NONDET_TOL = 1e-2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_genpow_problem(alphas, dim2, n=None, seed=42):
    """Construct a small QP with one GenPowerCone constraint.

    minimize  (1/2)x'Px + q'x
    subject to  -Ix + s = 0  (s = x projected into cone)
                sum(x) <= C   (nonneg slack)
                (p, w) in GenPowerCone(alphas, dim2)

    Returns dict with solver construction params.
    """
    dim1 = len(alphas)
    cone_dim = dim1 + dim2
    if n is None:
        n = cone_dim
    m = cone_dim + 1  # cone_dim for genpow + 1 nonneg

    rng = np.random.default_rng(seed)

    # P = diagonal PSD
    P_row_offsets = np.arange(n + 1, dtype=np.int64)
    P_col_indices = np.arange(n, dtype=np.int64)
    P_values = rng.uniform(0.5, 2.0, size=n)

    # A: first cone_dim rows = -I (mapping x -> cone slack)
    # last row = -[1,...,1] for sum constraint
    rows, cols, vals = [], [], []

    for i in range(min(cone_dim, n)):
        rows.append(i)
        cols.append(i)
        vals.append(-1.0)

    for j in range(n):
        rows.append(cone_dim)
        cols.append(j)
        vals.append(-1.0)

    A_dense = sparse.csr_array((vals, (rows, cols)), shape=(m, n))
    A_row_offsets = A_dense.indptr.astype(np.int64)
    A_col_indices = A_dense.indices.astype(np.int64)
    A_values = A_dense.data.astype(np.float64)

    cones = moreau.Cones(
        gen_power_cone_params=[(list(alphas), dim2)],
        num_nonneg_cones=1,
    )

    q = rng.uniform(-0.5, 0.5, size=n)
    b = np.zeros(m)
    b[-1] = -5.0  # sum(x) <= 5

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "cones": cones,
        "q": q,
        "b": b,
    }


def _make_compiled_solver(prob, device, batch_size=1):
    """Create CompiledSolver from problem dict."""
    settings = moreau.Settings(device=device, batch_size=batch_size)
    solver = moreau.CompiledSolver(
        n=prob["n"],
        m=prob["m"],
        P_row_offsets=prob["P_row_offsets"],
        P_col_indices=prob["P_col_indices"],
        A_row_offsets=prob["A_row_offsets"],
        A_col_indices=prob["A_col_indices"],
        cones=prob["cones"],
        settings=settings,
    )
    solver.setup(P_values=prob["P_values"], A_values=prob["A_values"])
    return solver


def _make_torch_solver(prob, device, batch_size=1):
    """Create TorchSolver from problem dict."""
    settings = moreau.Settings(
        device=device,
        batch_size=batch_size,
        verbose=False,
        max_iter=200,
    )
    settings.ipm_settings.tol_gap_abs = 1e-9
    settings.ipm_settings.tol_feas = 1e-9

    solver = TorchSolver(
        n=prob["n"],
        m=prob["m"],
        P_row_offsets=torch.tensor(prob["P_row_offsets"], dtype=torch.int64),
        P_col_indices=torch.tensor(prob["P_col_indices"], dtype=torch.int64),
        A_row_offsets=torch.tensor(prob["A_row_offsets"], dtype=torch.int64),
        A_col_indices=torch.tensor(prob["A_col_indices"], dtype=torch.int64),
        cones=prob["cones"],
        settings=settings,
    )
    return solver


# ---------------------------------------------------------------------------
# Warm-start gradcheck
# ---------------------------------------------------------------------------
#
# `torch.autograd.gradcheck`'s default cold-FD evaluates the IPM at q±eps in
# isolation. Near a borderline-active cone constraint the IPM's line-search
# step length is ratio-test-limited by z[i]/Δz[i] for the near-zero
# component i; tiny input perturbations move that ratio enough to change
# step length by ~20%, leaving each cold solve at a slightly different
# point inside the IPM's tol_gap_abs (1e-9) ball — typically ~5e-6 apart in
# x even though the true gradient predicts ~5e-7. Cold FD then disagrees
# with the analytical backward by O(1e-1), even though the analytical is
# correct.
#
# `_gradcheck_warm` warm-starts every FD solve from the unperturbed solution.
# That collapses the IPM trajectory across the FD points to a few
# refinement iterations from the same anchor, so the only iterate variation
# is the (genuine) sensitivity of the optimum to the input. Warm-FD then
# matches the analytical Jacobian to ~1e-4 — the gradcheck signal we
# actually want.


def _build_dense_PA(prob):
    """Construct dense scipy.sparse P (full symmetric) and A from the problem dict."""
    n, m = prob["n"], prob["m"]
    P_csr = sparse.csr_array(
        (prob["P_values"], prob["P_col_indices"], prob["P_row_offsets"]),
        shape=(n, n),
    )
    P_sym = P_csr + P_csr.T - sparse.diags(P_csr.diagonal())
    A_csr = sparse.csr_array(
        (prob["A_values"], prob["A_col_indices"], prob["A_row_offsets"]),
        shape=(m, n),
    )
    return sparse.csr_matrix(P_sym), sparse.csr_matrix(A_csr)


_GRADCHECK_TOL = 1e-11


def _make_gradcheck_settings(*, batch_size=1, device="cpu"):
    """Solver settings tightened for gradcheck use (tol=1e-11 instead of 1e-9).

    Tighter than the default test tol so warm-FD's per-FD-point noise floor
    drops below the FD truncation error — the FD step h=1e-4 has truncation
    O(h²)=1e-8, so we want IPM precision better than 1e-9 to keep FD signal
    above noise on multi-cone configurations.
    """
    settings = moreau.Settings(
        device=device,
        batch_size=batch_size,
        verbose=False,
        max_iter=400,
    )
    settings.ipm_settings.tol_gap_abs = _GRADCHECK_TOL
    settings.ipm_settings.tol_feas = _GRADCHECK_TOL
    return settings


def _make_gradcheck_torch_solver(prob):
    """TorchSolver with the gradcheck-tight settings."""
    settings = _make_gradcheck_settings()
    return TorchSolver(
        n=prob["n"],
        m=prob["m"],
        P_row_offsets=torch.tensor(prob["P_row_offsets"], dtype=torch.int64),
        P_col_indices=torch.tensor(prob["P_col_indices"], dtype=torch.int64),
        A_row_offsets=torch.tensor(prob["A_row_offsets"], dtype=torch.int64),
        A_col_indices=torch.tensor(prob["A_col_indices"], dtype=torch.int64),
        cones=prob["cones"],
        settings=settings,
    )


def _make_single_solver(P, q, A, b, cones):
    """Single-problem moreau.Solver with the gradcheck-tight settings."""
    return moreau.Solver(P, q, A, b, cones, _make_gradcheck_settings())


def _gradcheck_warm(prob, *, perturb, eps=1e-4, atol=1e-3, rtol=1e-2):
    """Assert analytical Jacobian (TorchSolver autograd) agrees with warm-FD.

    Parameters
    ----------
    prob : dict from `_make_genpow_problem` (or equivalent fields)
    perturb : one of {'q', 'b', 'P_values', 'A_values'} — which input to differentiate
    eps : FD step size. Default 1e-4 matches the optimal central-difference
        step for the gradcheck IPM tol = 1e-11: truncation O(h²) = 1e-8 is
        below atol; warm-FD precision floor is O(tol/h) ≈ 1e-7, also
        comfortably below atol.
    atol, rtol : ``np.testing.assert_allclose`` tolerances.

    Returns ``True`` on success; otherwise raises through ``assert_allclose``.
    """
    if not (HAS_TORCH and HAS_MOREAU_TORCH):
        pytest.skip("Requires torch + moreau.torch")

    P_sp, A_sp = _build_dense_PA(prob)
    cones = prob["cones"]
    q_np = np.asarray(prob["q"], dtype=np.float64)
    b_np = np.asarray(prob["b"], dtype=np.float64)

    # 1. Cold-solve once to get the warm-start anchor.
    anchor = _make_single_solver(P_sp, q_np, A_sp, b_np, cones)
    sol_anchor = anchor.solve()
    assert (
        anchor.info.status in _SOLVED_STATUSES
    ), f"Anchor solve failed: status={anchor.info.status}"
    warm = sol_anchor.to_warm_start()
    n_out = prob["n"]

    # 2. Analytical Jacobian via TorchSolver autograd at the unperturbed input.
    tsolver = _make_gradcheck_torch_solver(prob)
    P_t = torch.tensor(prob["P_values"], dtype=torch.float64).unsqueeze(0)
    A_t = torch.tensor(prob["A_values"], dtype=torch.float64).unsqueeze(0)
    q_t = torch.tensor(q_np, dtype=torch.float64).unsqueeze(0)
    b_t = torch.tensor(b_np, dtype=torch.float64).unsqueeze(0)
    if perturb == "q":
        q_t = q_t.detach().requires_grad_(True)
        inp_t = q_t
    elif perturb == "b":
        b_t = b_t.detach().requires_grad_(True)
        inp_t = b_t
    elif perturb == "P_values":
        P_t = P_t.detach().requires_grad_(True)
        inp_t = P_t
    elif perturb == "A_values":
        A_t = A_t.detach().requires_grad_(True)
        inp_t = A_t
    else:
        raise ValueError(f"unknown perturb={perturb!r}")
    sol_t = tsolver.solve(P_t, A_t, q_t, b_t)
    n_in = inp_t.numel()
    J_an = np.zeros((n_out, n_in))
    for i in range(n_out):
        g = torch.autograd.grad(sol_t.x[0, i], inp_t, retain_graph=True)[0]
        J_an[i, :] = g.detach().numpy().reshape(-1)

    # 3. Warm-FD Jacobian.
    inp_np = inp_t.detach().numpy().reshape(-1)
    J_fd = np.zeros((n_out, n_in))

    def _solve_with(P_v, q_v, A_v, b_v):
        s = _make_single_solver(P_v, q_v, A_v, b_v, cones)
        sol = s.solve(warm_start=warm)
        return np.asarray(sol.x)

    for j in range(n_in):
        plus = inp_np.copy()
        plus[j] += eps
        minus = inp_np.copy()
        minus[j] -= eps
        if perturb == "q":
            x_p = _solve_with(P_sp, plus, A_sp, b_np)
            x_m = _solve_with(P_sp, minus, A_sp, b_np)
        elif perturb == "b":
            x_p = _solve_with(P_sp, q_np, A_sp, plus)
            x_m = _solve_with(P_sp, q_np, A_sp, minus)
        elif perturb == "P_values":
            P_p = sparse.csr_matrix(
                (plus, prob["P_col_indices"], prob["P_row_offsets"]), shape=(n_out, n_out)
            )
            P_p = P_p + P_p.T - sparse.diags(P_p.diagonal())
            P_m = sparse.csr_matrix(
                (minus, prob["P_col_indices"], prob["P_row_offsets"]), shape=(n_out, n_out)
            )
            P_m = P_m + P_m.T - sparse.diags(P_m.diagonal())
            x_p = _solve_with(P_p, q_np, A_sp, b_np)
            x_m = _solve_with(P_m, q_np, A_sp, b_np)
        else:  # A_values
            A_p = sparse.csr_matrix(
                (plus, prob["A_col_indices"], prob["A_row_offsets"]), shape=(prob["m"], n_out)
            )
            A_m = sparse.csr_matrix(
                (minus, prob["A_col_indices"], prob["A_row_offsets"]), shape=(prob["m"], n_out)
            )
            x_p = _solve_with(P_sp, q_np, A_p, b_np)
            x_m = _solve_with(P_sp, q_np, A_m, b_np)
        J_fd[:, j] = (x_p - x_m) / (2 * eps)

    np.testing.assert_allclose(
        J_an,
        J_fd,
        atol=atol,
        rtol=rtol,
        err_msg=f"warm-FD vs analytical Jacobian disagrees for perturb={perturb!r}",
    )
    return True


# ---------------------------------------------------------------------------
# Phase 4.1: Forward solve tests
# ---------------------------------------------------------------------------


class TestGenPowerConeForwardSolve:
    """Test forward solves with GenPowerCone constraints."""

    def test_basic_genpow_alpha_03_07(self, device):
        """Simple GenPowerCone with alpha=[0.3, 0.7], dim2=2."""
        prob = _make_genpow_problem([0.3, 0.7], dim2=2, seed=42)
        solver = _make_compiled_solver(prob, device)
        sol = solver.solve(
            qs=prob["q"].reshape(1, -1),
            bs=prob["b"].reshape(1, -1),
        )
        assert (
            solver.info.status[0] in _SOLVED_STATUSES
        ), f"Expected Solved, got {solver.info.status[0]}"

    def test_basic_genpow_alpha_05_05(self, device):
        """GenPowerCone with alpha=[0.5, 0.5], dim2=1 (reduces to 3D power cone)."""
        prob = _make_genpow_problem([0.5, 0.5], dim2=1, seed=100)
        solver = _make_compiled_solver(prob, device)
        sol = solver.solve(
            qs=prob["q"].reshape(1, -1),
            bs=prob["b"].reshape(1, -1),
        )
        assert solver.info.status[0] in _SOLVED_STATUSES

    def test_genpow_three_alphas(self, device):
        """GenPowerCone with dim1=3 alphas, dim2=2 (total dim=5)."""
        prob = _make_genpow_problem([0.2, 0.3, 0.5], dim2=2, seed=200)
        solver = _make_compiled_solver(prob, device)
        sol = solver.solve(
            qs=prob["q"].reshape(1, -1),
            bs=prob["b"].reshape(1, -1),
        )
        assert solver.info.status[0] in _SOLVED_STATUSES

    def test_genpow_four_alphas(self, device):
        """GenPowerCone with dim1=4, dim2=3 (total dim=7)."""
        prob = _make_genpow_problem([0.1, 0.2, 0.3, 0.4], dim2=3, seed=300)
        solver = _make_compiled_solver(prob, device)
        sol = solver.solve(
            qs=prob["q"].reshape(1, -1),
            bs=prob["b"].reshape(1, -1),
        )
        assert solver.info.status[0] in _SOLVED_STATUSES

    def test_genpow_single_alpha(self, device):
        """GenPowerCone with dim1=1 (alpha=[1.0]), dim2=1."""
        prob = _make_genpow_problem([1.0], dim2=1, seed=400)
        solver = _make_compiled_solver(prob, device)
        sol = solver.solve(
            qs=prob["q"].reshape(1, -1),
            bs=prob["b"].reshape(1, -1),
        )
        assert solver.info.status[0] in _SOLVED_STATUSES

    def test_genpow_random_program(self, device):
        """Test using random_cone_program with GenPowerCone."""
        cones = moreau.Cones(gen_power_cone_params=[([0.3, 0.7], 2)])
        prob = random_cone_program(n=6, cones=cones, seed=42)
        solver = moreau.Solver(
            prob.P,
            prob.q,
            prob.A,
            prob.b,
            prob.cones,
            moreau.Settings(device=device),
        )
        sol = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

    def test_genpow_random_three_alphas(self, device):
        """Random program with 3-alpha GenPowerCone."""
        cones = moreau.Cones(gen_power_cone_params=[([0.2, 0.3, 0.5], 2)])
        prob = random_cone_program(n=8, cones=cones, seed=100)
        solver = moreau.Solver(
            prob.P,
            prob.q,
            prob.A,
            prob.b,
            prob.cones,
            moreau.Settings(device=device),
        )
        sol = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

    @pytest.mark.parametrize("seed", [1, 42, 100, 200, 300])
    def test_genpow_multiple_seeds(self, device, seed):
        """Multiple random problems with GenPowerCone."""
        cones = moreau.Cones(gen_power_cone_params=[([0.4, 0.6], 2)])
        prob = random_cone_program(n=6, cones=cones, seed=seed)
        solver = moreau.Solver(
            prob.P,
            prob.q,
            prob.A,
            prob.b,
            prob.cones,
            moreau.Settings(device=device),
        )
        sol = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

    def test_genpow_high_dim_64(self, device):
        """High-dim GenPowerCone: dim1=40, dim2=24 (total=64).

        Verifies that the Python API works end-to-end for dimensions that
        previously would have been blocked by the dim=32 cap.
        Uses _make_genpow_problem (A=-I, diagonal P) for numerical stability.
        """
        alphas = [1.0 / 40] * 40
        prob = _make_genpow_problem(alphas, dim2=24, seed=42)
        settings = moreau.Settings(device=device, batch_size=1, max_iter=500)
        solver = moreau.CompiledSolver(
            n=prob["n"],
            m=prob["m"],
            P_row_offsets=prob["P_row_offsets"],
            P_col_indices=prob["P_col_indices"],
            A_row_offsets=prob["A_row_offsets"],
            A_col_indices=prob["A_col_indices"],
            cones=prob["cones"],
            settings=settings,
        )
        solver.setup(P_values=prob["P_values"], A_values=prob["A_values"])
        sol = solver.solve(
            qs=prob["q"].reshape(1, -1),
            bs=prob["b"].reshape(1, -1),
        )
        assert solver.info.status[0] in _SOLVED_STATUSES


class TestGenPowerConeMultiple:
    """Test multiple GenPowerCone constraints in one problem."""

    def test_two_genpow_cones(self, device):
        """Two GenPowerCones of same dimension."""
        cones = moreau.Cones(
            gen_power_cone_params=[
                ([0.3, 0.7], 2),
                ([0.5, 0.5], 2),
            ]
        )
        prob = random_cone_program(n=10, cones=cones, seed=42)
        solver = moreau.Solver(
            prob.P,
            prob.q,
            prob.A,
            prob.b,
            prob.cones,
            moreau.Settings(device=device),
        )
        sol = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

    def test_two_genpow_different_dims(self, device):
        """Two GenPowerCones of different dimensions."""
        cones = moreau.Cones(
            gen_power_cone_params=[
                ([0.3, 0.7], 1),  # total dim = 3
                ([0.2, 0.3, 0.5], 2),  # total dim = 5
            ]
        )
        prob = random_cone_program(n=10, cones=cones, seed=123)
        solver = moreau.Solver(
            prob.P,
            prob.q,
            prob.A,
            prob.b,
            prob.cones,
            moreau.Settings(device=device),
        )
        sol = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES


class TestGenPowerConeMixed:
    """Test GenPowerCone mixed with other cone types."""

    def test_genpow_with_nonneg(self, device):
        """GenPowerCone + nonneg cones."""
        cones = moreau.Cones(
            num_nonneg_cones=3,
            gen_power_cone_params=[([0.3, 0.7], 2)],
        )
        prob = random_cone_program(n=8, cones=cones, seed=42)
        solver = moreau.Solver(
            prob.P,
            prob.q,
            prob.A,
            prob.b,
            prob.cones,
            moreau.Settings(device=device),
        )
        sol = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

    def test_genpow_with_zero_nonneg(self, device):
        """GenPowerCone + zero + nonneg cones."""
        cones = moreau.Cones(
            num_zero_cones=2,
            num_nonneg_cones=3,
            gen_power_cone_params=[([0.4, 0.6], 1)],
        )
        prob = random_cone_program(n=10, cones=cones, seed=100)
        solver = moreau.Solver(
            prob.P,
            prob.q,
            prob.A,
            prob.b,
            prob.cones,
            moreau.Settings(device=device),
        )
        sol = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

    def test_genpow_with_soc(self, device):
        """GenPowerCone + SOC cones."""
        cones = moreau.Cones(
            so_cone_dims=[3],
            gen_power_cone_params=[([0.3, 0.7], 2)],
        )
        prob = random_cone_program(n=10, cones=cones, seed=200)
        solver = moreau.Solver(
            prob.P,
            prob.q,
            prob.A,
            prob.b,
            prob.cones,
            moreau.Settings(device=device),
        )
        sol = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

    def test_genpow_with_power(self, device):
        """GenPowerCone + power cones (3D)."""
        cones = moreau.Cones(
            power_alphas=[0.5],
            gen_power_cone_params=[([0.3, 0.7], 2)],
        )
        prob = random_cone_program(n=8, cones=cones, seed=300)
        solver = moreau.Solver(
            prob.P,
            prob.q,
            prob.A,
            prob.b,
            prob.cones,
            moreau.Settings(device=device),
        )
        sol = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES

    def test_genpow_all_cone_types(self, device):
        """GenPowerCone + zero + nonneg + SOC + exp + power."""
        cones = moreau.Cones(
            num_zero_cones=1,
            num_nonneg_cones=2,
            so_cone_dims=[3],
            num_exp_cones=1,
            power_alphas=[0.5],
            gen_power_cone_params=[([0.3, 0.7], 1)],
        )
        prob = random_cone_program(n=15, cones=cones, seed=42)
        solver = moreau.Solver(
            prob.P,
            prob.q,
            prob.A,
            prob.b,
            prob.cones,
            moreau.Settings(device=device),
        )
        sol = solver.solve()
        assert solver.info.status in _SOLVED_STATUSES


class TestGenPowerConeBatched:
    """Test batched solves with GenPowerCone."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_batched_genpow(self, device, batch_size):
        """Batched GenPowerCone solve."""
        cones = moreau.Cones(gen_power_cone_params=[([0.3, 0.7], 2)])
        n = 6
        first, problems = random_batch(n=n, cones=cones, batch_size=batch_size, seed=42)

        m = cones.total_constraints()
        solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=first.P.indptr,
            P_col_indices=first.P.indices,
            A_row_offsets=first.A.indptr,
            A_col_indices=first.A.indices,
            cones=cones,
            settings=moreau.Settings(device=device, batch_size=batch_size),
        )
        solver.setup(
            [p.P.data for p in problems],
            [p.A.data for p in problems],
        )
        sol = solver.solve(
            [p.q for p in problems],
            [p.b for p in problems],
        )

        for i, status in enumerate(solver.info.status):
            assert status in _SOLVED_STATUSES, f"Batch {i}: {status}"

    def test_batched_genpow_mixed(self, device):
        """Batched solve with GenPowerCone + other cones."""
        cones = moreau.Cones(
            num_nonneg_cones=2,
            gen_power_cone_params=[([0.4, 0.6], 1)],
        )
        n = 6
        batch_size = 3
        first, problems = random_batch(n=n, cones=cones, batch_size=batch_size, seed=100)

        m = cones.total_constraints()
        solver = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=first.P.indptr,
            P_col_indices=first.P.indices,
            A_row_offsets=first.A.indptr,
            A_col_indices=first.A.indices,
            cones=cones,
            settings=moreau.Settings(device=device, batch_size=batch_size),
        )
        solver.setup(
            [p.P.data for p in problems],
            [p.A.data for p in problems],
        )
        sol = solver.solve(
            [p.q for p in problems],
            [p.b for p in problems],
        )

        for i, status in enumerate(solver.info.status):
            assert status in _SOLVED_STATUSES, f"Batch {i}: {status}"


# ---------------------------------------------------------------------------
# Phase 4.2: Backward / gradient tests
# ---------------------------------------------------------------------------

requires_torch = pytest.mark.skipif(
    not (HAS_TORCH and HAS_MOREAU_TORCH), reason="Requires torch and moreau.torch"
)


@requires_torch
class TestGenPowerConeGradcheck:
    """torch.autograd.gradcheck for GenPowerCone problems."""

    def test_gradcheck_genpow_q(self, device):
        """Gradcheck w.r.t. q for GenPowerCone (warm-FD)."""
        if device != "cpu":
            pytest.skip("warm-start FD is CPU-only; CUDA gradcheck via test_diff.cpp")
        prob = _make_genpow_problem([0.3, 0.7], dim2=2, seed=42)
        assert _gradcheck_warm(prob, perturb="q")

    def test_gradcheck_genpow_b(self, device):
        """Gradcheck w.r.t. b for GenPowerCone (warm-FD)."""
        if device != "cpu":
            pytest.skip("warm-start FD is CPU-only; CUDA gradcheck via test_diff.cpp")
        prob = _make_genpow_problem([0.3, 0.7], dim2=2, seed=42)
        assert _gradcheck_warm(prob, perturb="b")

    def test_gradcheck_genpow_three_alphas_q(self, device):
        """Gradcheck w.r.t. q for 3-alpha GenPowerCone (warm-FD).

        Seed 200 produces a well-conditioned interior solution.
        """
        if device != "cpu":
            pytest.skip("warm-start FD is CPU-only; CUDA gradcheck via test_diff.cpp")
        prob = _make_genpow_problem([0.2, 0.3, 0.5], dim2=2, seed=200)
        assert _gradcheck_warm(prob, perturb="q")

    def test_gradcheck_genpow_mixed_q(self, device):
        """Gradcheck w.r.t. q for GenPowerCone + nonneg cones (warm-FD)."""
        if device != "cpu":
            pytest.skip("warm-start FD is CPU-only; CUDA gradcheck via test_diff.cpp")
        prob = _make_genpow_problem([0.4, 0.6], dim2=1, seed=300)
        assert _gradcheck_warm(prob, perturb="q")

    def test_gradcheck_genpow_P_values(self, device):
        """Gradcheck w.r.t. P_values for GenPowerCone (warm-FD)."""
        if device != "cpu":
            pytest.skip("warm-start FD is CPU-only; CUDA gradcheck via test_diff.cpp")
        prob = _make_genpow_problem([0.3, 0.7], dim2=2, seed=42)
        assert _gradcheck_warm(prob, perturb="P_values")

    def test_gradcheck_genpow_A_values(self, device):
        """Gradcheck w.r.t. A_values for GenPowerCone (warm-FD)."""
        if device != "cpu":
            pytest.skip("warm-start FD is CPU-only; CUDA gradcheck via test_diff.cpp")
        prob = _make_genpow_problem([0.3, 0.7], dim2=2, seed=42)
        assert _gradcheck_warm(prob, perturb="A_values")


# ---------------------------------------------------------------------------
# Phase 4.3: Warm start tests
# ---------------------------------------------------------------------------


class TestGenPowerConeWarmStart:
    """Warm starting tests for GenPowerCone."""

    def test_warm_start_solution_correct(self, device):
        """Warm start should produce a correct solution (same as cold)."""
        prob = _make_genpow_problem([0.3, 0.7], dim2=2, seed=42)
        q = prob["q"].reshape(1, -1)
        b = prob["b"].reshape(1, -1)

        # Cold solve
        solver_cold = _make_compiled_solver(prob, device)
        sol_cold = solver_cold.solve(qs=q, bs=b)
        info_cold = solver_cold.info
        assert info_cold.status[0] in _SOLVED_STATUSES

        # Warm solve (may fall back to cold for GenPowerCone)
        solver_warm = _make_compiled_solver(prob, device)
        sol_warm = solver_warm.solve(
            qs=q,
            bs=b,
            warm_start=sol_cold.to_warm_start(),
        )
        info_warm = solver_warm.info
        assert info_warm.status[0] in _SOLVED_STATUSES
        # Warm/cold land within the IPM tolerance ball (~1e-5 apart), not 1e-8.
        np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4, rtol=1e-4)

    def test_warm_start_perturbed_q(self, device):
        """Warm start from solution of nearby problem (perturbed q)."""
        prob = _make_genpow_problem([0.3, 0.7], dim2=2, seed=42)
        q1 = prob["q"].reshape(1, -1)
        q2 = (prob["q"] + np.array([0.1, -0.1, 0.05, 0.0])).reshape(1, -1)
        b = prob["b"].reshape(1, -1)

        # Cold solve P1
        solver1 = _make_compiled_solver(prob, device)
        sol1 = solver1.solve(qs=q1, bs=b)
        assert solver1.info.status[0] in _SOLVED_STATUSES

        # Warm solve P2
        solver2_warm = _make_compiled_solver(prob, device)
        sol2_warm = solver2_warm.solve(
            qs=q2,
            bs=b,
            warm_start=sol1.to_warm_start(),
        )
        assert solver2_warm.info.status[0] in _SOLVED_STATUSES

        # Cold solve P2 for reference
        solver2_cold = _make_compiled_solver(prob, device)
        sol2_cold = solver2_cold.solve(qs=q2, bs=b)
        assert solver2_cold.info.status[0] in _SOLVED_STATUSES

        # Warm/cold land within the IPM tolerance ball (~1e-5 apart), not 1e-8.
        np.testing.assert_allclose(sol2_warm.x, sol2_cold.x, atol=1e-4, rtol=1e-4)

    def test_warm_start_random_genpow(self, device):
        """Warm start on random GenPowerCone program."""
        cones = moreau.Cones(
            num_nonneg_cones=2,
            gen_power_cone_params=[([0.4, 0.6], 2)],
        )
        prob = random_cone_program(n=6, cones=cones, seed=42)

        n = prob.q.shape[0]
        m = cones.total_constraints()
        solver_cold = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=prob.P.indptr,
            P_col_indices=prob.P.indices,
            A_row_offsets=prob.A.indptr,
            A_col_indices=prob.A.indices,
            cones=cones,
            settings=moreau.Settings(device=device, batch_size=1),
        )
        solver_cold.setup(prob.P.data, prob.A.data)
        sol_cold = solver_cold.solve(
            qs=prob.q.reshape(1, -1),
            bs=prob.b.reshape(1, -1),
        )
        assert solver_cold.info.status[0] in _SOLVED_STATUSES

        solver_warm = moreau.CompiledSolver(
            n=n,
            m=m,
            P_row_offsets=prob.P.indptr,
            P_col_indices=prob.P.indices,
            A_row_offsets=prob.A.indptr,
            A_col_indices=prob.A.indices,
            cones=cones,
            settings=moreau.Settings(device=device, batch_size=1),
        )
        solver_warm.setup(prob.P.data, prob.A.data)
        sol_warm = solver_warm.solve(
            qs=prob.q.reshape(1, -1),
            bs=prob.b.reshape(1, -1),
            warm_start=sol_cold.to_warm_start(),
        )
        # Warm start may fall back to cold solve for GenPowerCone
        assert solver_warm.info.status[0] in _SOLVED_STATUSES
        # Warm/cold land within the IPM tolerance ball (~1e-5 apart), not 1e-8.
        np.testing.assert_allclose(sol_warm.x, sol_cold.x, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Phase 4.4: CPU/GPU parity tests
# ---------------------------------------------------------------------------

pytestmark_cuda = pytest.mark.skipif(
    not moreau.device_available("cuda"),
    reason="CUDA not available",
)


@pytestmark_cuda
class TestGenPowerConeCpuGpuParity:
    """CPU vs GPU parity tests for GenPowerCone.

    GenPowerCone is a nonsymmetric cone with nonlinear barriers. CPU (faer) and
    GPU (cuDSS) KKT backends can converge to different solutions for the same
    problem. We verify both reach Solved status, produce finite solutions,
    and satisfy KKT conditions independently.
    """

    def _solve_on_device(self, prob, device_name):
        """Solve random program on specified device."""
        solver = moreau.Solver(
            prob.P,
            prob.q,
            prob.A,
            prob.b,
            prob.cones,
            moreau.Settings(device=device_name),
        )
        sol = solver.solve()
        return sol, solver.info

    def _check_kkt_residuals(self, prob, sol, label, feas_tol=1e-5):
        """Verify solution satisfies KKT conditions."""
        x, s, z = np.asarray(sol.x), np.asarray(sol.s), np.asarray(sol.z)
        assert np.all(np.isfinite(x)), f"{label}: x has non-finite values"
        assert np.all(np.isfinite(s)), f"{label}: s has non-finite values"
        assert np.all(np.isfinite(z)), f"{label}: z has non-finite values"

        # Primal feasibility: ||Ax + s - b||_inf
        r_prim = np.max(np.abs(prob.A @ x + s - prob.b))
        assert r_prim < feas_tol, f"{label}: primal residual {r_prim:.2e} >= {feas_tol}"

        # Dual feasibility: ||Px + q + A'z||_inf
        r_dual = np.max(np.abs(prob.P @ x + prob.q + prob.A.T @ z))
        assert r_dual < feas_tol, f"{label}: dual residual {r_dual:.2e} >= {feas_tol}"

    def _check_parity(self, prob, cpu_sol, cpu_info, gpu_sol, gpu_info):
        """Common parity assertions for all tests."""
        assert cpu_info.status in _SOLVED_STATUSES
        assert gpu_info.status in _SOLVED_STATUSES
        assert np.isfinite(cpu_info.obj_val) and np.isfinite(gpu_info.obj_val)
        self._check_kkt_residuals(prob, cpu_sol, "CPU")
        self._check_kkt_residuals(prob, gpu_sol, "GPU")

    def test_forward_parity_basic(self):
        """CPU and GPU should both solve."""
        cones = moreau.Cones(gen_power_cone_params=[([0.3, 0.7], 2)])
        prob = random_cone_program(n=6, cones=cones, seed=300)

        cpu_sol, cpu_info = self._solve_on_device(prob, "cpu")
        gpu_sol, gpu_info = self._solve_on_device(prob, "cuda")

        self._check_parity(prob, cpu_sol, cpu_info, gpu_sol, gpu_info)

    def test_forward_parity_three_alphas(self):
        """CPU/GPU parity with 3-alpha GenPowerCone."""
        cones = moreau.Cones(gen_power_cone_params=[([0.2, 0.3, 0.5], 2)])
        prob = random_cone_program(n=8, cones=cones, seed=300)

        cpu_sol, cpu_info = self._solve_on_device(prob, "cpu")
        gpu_sol, gpu_info = self._solve_on_device(prob, "cuda")

        self._check_parity(prob, cpu_sol, cpu_info, gpu_sol, gpu_info)

    def test_forward_parity_mixed_cones(self):
        """CPU/GPU parity with GenPowerCone + other cones."""
        cones = moreau.Cones(
            num_zero_cones=1,
            num_nonneg_cones=2,
            gen_power_cone_params=[([0.3, 0.7], 1)],
        )
        prob = random_cone_program(n=8, cones=cones, seed=200)

        cpu_sol, cpu_info = self._solve_on_device(prob, "cpu")
        gpu_sol, gpu_info = self._solve_on_device(prob, "cuda")

        self._check_parity(prob, cpu_sol, cpu_info, gpu_sol, gpu_info)

    def test_forward_parity_two_genpow(self):
        """CPU/GPU parity with two GenPowerCones."""
        cones = moreau.Cones(
            gen_power_cone_params=[
                ([0.3, 0.7], 2),
                ([0.5, 0.5], 1),
            ]
        )
        prob = random_cone_program(n=10, cones=cones, seed=300)

        cpu_sol, cpu_info = self._solve_on_device(prob, "cpu")
        gpu_sol, gpu_info = self._solve_on_device(prob, "cuda")

        self._check_parity(prob, cpu_sol, cpu_info, gpu_sol, gpu_info)

    @pytest.mark.parametrize("seed", [1, 42, 100, 200, 300, 500, 700])
    def test_forward_parity_multiple_seeds(self, seed):
        """CPU/GPU parity across multiple random seeds."""
        cones = moreau.Cones(gen_power_cone_params=[([0.4, 0.6], 2)])
        prob = random_cone_program(n=6, cones=cones, seed=seed)

        cpu_sol, cpu_info = self._solve_on_device(prob, "cpu")
        gpu_sol, gpu_info = self._solve_on_device(prob, "cuda")

        self._check_parity(prob, cpu_sol, cpu_info, gpu_sol, gpu_info)


@pytestmark_cuda
@requires_torch
class TestGenPowerConeGpuGradients:
    """Test GPU backward/gradient correctness for GenPowerCone."""

    def _solve_with_grads(self, prob, device):
        """Solve on given device and compute autograd gradients."""
        n, m = prob.q.shape[0], prob.b.shape[0]

        P_ro = torch.tensor(prob.P.indptr, dtype=torch.int64)
        P_ci = torch.tensor(prob.P.indices, dtype=torch.int64)
        A_ro = torch.tensor(prob.A.indptr, dtype=torch.int64)
        A_ci = torch.tensor(prob.A.indices, dtype=torch.int64)

        P_vals = (
            torch.tensor(prob.P.data, dtype=torch.float64, device=device)
            .clone()
            .requires_grad_(True)
        )
        A_vals = (
            torch.tensor(prob.A.data, dtype=torch.float64, device=device)
            .clone()
            .requires_grad_(True)
        )
        q = torch.tensor(prob.q, dtype=torch.float64, device=device).clone().requires_grad_(True)
        b = torch.tensor(prob.b, dtype=torch.float64, device=device).clone().requires_grad_(True)

        # Tight tol so CPU and GPU converge to the same optimum within
        # FD-comparable precision; default 1e-8 leaves the two backends
        # at points that differ by enough in `x` to make their analytical
        # backward gradients disagree at the FD-noise floor.
        settings = moreau.Settings(device=device)
        settings.ipm_settings.tol_gap_abs = 1e-11
        settings.ipm_settings.tol_feas = 1e-11
        solver = TorchSolver(
            n=n,
            m=m,
            P_row_offsets=P_ro,
            P_col_indices=P_ci,
            A_row_offsets=A_ro,
            A_col_indices=A_ci,
            cones=prob.cones,
            settings=settings,
        )

        solution = solver.solve(
            P_vals.unsqueeze(0), A_vals.unsqueeze(0), q.unsqueeze(0), b.unsqueeze(0)
        )

        loss = solution.x.sum()
        loss.backward()

        return {
            "x": solution.x.detach().cpu().numpy().squeeze(),
            "dq": q.grad.detach().cpu().numpy() if q.grad is not None else None,
            "db": b.grad.detach().cpu().numpy() if b.grad is not None else None,
        }

    def _fd_gradient_check(self, prob, device, eps=1e-4):
        """Validate backward gradients against finite differences on a given device.

        FD step h=1e-4 matches the optimal central-difference step for the
        gradcheck IPM tol = 1e-11 (truncation O(h²) = 1e-8 is well below
        the comparison rtol=1e-2 / atol=1e-3).
        """
        n, m = prob.q.shape[0], prob.b.shape[0]

        s_settings = moreau.Settings(device=device)
        s_settings.ipm_settings.tol_gap_abs = 1e-11
        s_settings.ipm_settings.tol_feas = 1e-11

        def solve_x_sum(q_np):
            solver = moreau.CompiledSolver(
                n=n,
                m=m,
                P_row_offsets=prob.P.indptr.tolist(),
                P_col_indices=prob.P.indices.tolist(),
                A_row_offsets=prob.A.indptr.tolist(),
                A_col_indices=prob.A.indices.tolist(),
                cones=prob.cones,
                settings=s_settings,
            )
            solver.setup(
                P_values=prob.P.data.tolist(),
                A_values=prob.A.data.tolist(),
            )
            sol = solver.solve(qs=[q_np.tolist()], bs=[prob.b.tolist()])
            return np.array(sol.x[0]).sum()

        analytical = self._solve_with_grads(prob, device)
        dq_fd = np.zeros(n)
        for i in range(n):
            qp = prob.q.copy()
            qp[i] += eps
            qm = prob.q.copy()
            qm[i] -= eps
            dq_fd[i] = (solve_x_sum(qp) - solve_x_sum(qm)) / (2 * eps)

        np.testing.assert_allclose(
            analytical["dq"],
            dq_fd,
            rtol=1e-2,
            atol=1e-3,
            err_msg=f"GenPowerCone {device} dq gradient vs finite diff",
        )
        return analytical

    def test_gpu_backward_parity_with_cpu(self):
        """GPU backward gradients should match CPU backward gradients.

        CPU and GPU forward solves converge to slightly different optimal points
        (different linear algebra backends: QDLDL vs cuDSS), so backward
        gradients are evaluated at different solution points.  We first verify
        each backend's gradients are correct via finite differences, then check
        cross-device agreement with tolerances that account for this.
        """
        cones = moreau.Cones(gen_power_cone_params=[([0.3, 0.7], 2)])
        prob = random_cone_program(n=6, cones=cones, seed=42)

        # Validate each device's backward pass independently via FD
        cpu_result = self._fd_gradient_check(prob, "cpu")
        gpu_result = self._fd_gradient_check(prob, "cuda")

        assert cpu_result["dq"] is not None and gpu_result["dq"] is not None
        assert cpu_result["db"] is not None and gpu_result["db"] is not None

        # Cross-device comparison (relaxed: forward solution differences
        # propagate through nonlinear cone derivatives)
        np.testing.assert_allclose(
            gpu_result["dq"],
            cpu_result["dq"],
            rtol=1e-2,
            atol=1e-3,
            err_msg="GenPowerCone dq gradient mismatch between CPU and GPU",
        )
        np.testing.assert_allclose(
            gpu_result["db"],
            cpu_result["db"],
            rtol=1e-2,
            atol=1e-3,
            err_msg="GenPowerCone db gradient mismatch between CPU and GPU",
        )

    def test_gpu_backward_parity_three_alphas(self):
        """GPU backward parity for 3-alpha GenPowerCone (dim=5)."""
        cones = moreau.Cones(gen_power_cone_params=[([0.2, 0.3, 0.5], 2)])
        prob = random_cone_program(n=7, cones=cones, seed=200)

        cpu_result = self._solve_with_grads(prob, "cpu")
        gpu_result = self._solve_with_grads(prob, "cuda")

        assert cpu_result["dq"] is not None and gpu_result["dq"] is not None
        np.testing.assert_allclose(
            gpu_result["dq"],
            cpu_result["dq"],
            rtol=1e-2,
            atol=1e-3,
            err_msg="3-alpha GenPowerCone dq gradient mismatch between CPU and GPU",
        )
        np.testing.assert_allclose(
            gpu_result["db"],
            cpu_result["db"],
            rtol=1e-2,
            atol=1e-3,
            err_msg="3-alpha GenPowerCone db gradient mismatch between CPU and GPU",
        )


# ---------------------------------------------------------------------------
# Phase 4.4: CUDA batch parity
# ---------------------------------------------------------------------------


@pytestmark_cuda
class TestGenPowerConeBatchParity:
    """Test CPU/GPU parity on batched GenPowerCone solves."""

    def test_batch_genpow_parity(self):
        """Batched GenPowerCone: both CPU and GPU should solve."""
        cones = moreau.Cones(gen_power_cone_params=[([0.3, 0.7], 2)])
        n = 6
        batch_size = 3
        first, problems = random_batch(n=n, cones=cones, batch_size=batch_size, seed=42)

        m = cones.total_constraints()

        def _solve_batch(device_name):
            solver = moreau.CompiledSolver(
                n=n,
                m=m,
                P_row_offsets=first.P.indptr,
                P_col_indices=first.P.indices,
                A_row_offsets=first.A.indptr,
                A_col_indices=first.A.indices,
                cones=cones,
                settings=moreau.Settings(device=device_name, batch_size=batch_size),
            )
            solver.setup(
                [p.P.data for p in problems],
                [p.A.data for p in problems],
            )
            sol = solver.solve(
                [p.q for p in problems],
                [p.b for p in problems],
            )
            return sol, solver.info

        cpu_sol, cpu_info = _solve_batch("cpu")
        gpu_sol, gpu_info = _solve_batch("cuda")

        for i in range(batch_size):
            assert cpu_info.status[i] in _SOLVED_STATUSES
            assert gpu_info.status[i] in _SOLVED_STATUSES

        # Compare objectives and solutions between CPU and GPU.
        # CPU (faer) and GPU (cuDSS) can converge along slightly different
        # numerical paths for nonsymmetric cones.
        np.testing.assert_allclose(
            cpu_info.obj_val,
            gpu_info.obj_val,
            rtol=1e-3,
            atol=1e-2,
            err_msg="Objective mismatch between CPU and GPU",
        )

        # Compare x solutions (use moderate tolerance for nonsymmetric cones)
        for i in range(batch_size):
            np.testing.assert_allclose(
                cpu_sol.x[i],
                gpu_sol.x[i],
                rtol=1e-2,
                atol=1e-3,
                err_msg=f"x mismatch between CPU and GPU at batch {i}",
            )


# ---------------------------------------------------------------------------
# Phase 4.5: High-dim & edge-case gradient tests
# ---------------------------------------------------------------------------


@requires_torch
class TestGenPowerConeHighDimGradcheck:
    """Gradient checks for high-dimensional and edge-case GenPowerCone problems."""

    def test_gradcheck_high_dim_10_alphas(self, device):
        """Gradcheck w.r.t. q for high-dim GenPowerCone: 10 alphas, dim2=5 (total=15) (warm-FD)."""
        if device != "cpu":
            pytest.skip("warm-start FD is CPU-only; CUDA gradcheck via test_diff.cpp")
        prob = _make_genpow_problem([0.1] * 10, dim2=5, seed=42)
        assert _gradcheck_warm(prob, perturb="q")

    def test_gradcheck_mixed_size_cones(self, device):
        """Gradcheck with two GenPowerCones of different sizes (warm-FD)."""
        if device != "cpu":
            pytest.skip("warm-start FD is CPU-only; CUDA gradcheck via test_diff.cpp")
        dim1a, dim2a = 2, 1  # total = 3
        dim1b, dim2b = 3, 2  # total = 5
        cone_dim = (dim1a + dim2a) + (dim1b + dim2b)  # 8
        n = cone_dim
        m = cone_dim + 1  # +1 nonneg

        rng = np.random.default_rng(42)

        P_row_offsets = np.arange(n + 1, dtype=np.int64)
        P_col_indices = np.arange(n, dtype=np.int64)
        P_values = rng.uniform(0.5, 2.0, size=n)

        rows, cols, vals = [], [], []
        for i in range(n):
            rows.append(i)
            cols.append(i)
            vals.append(-1.0)
        for j in range(n):
            rows.append(cone_dim)
            cols.append(j)
            vals.append(-1.0)

        A_dense = sparse.csr_array((vals, (rows, cols)), shape=(m, n))
        A_row_offsets = A_dense.indptr.astype(np.int64)
        A_col_indices = A_dense.indices.astype(np.int64)
        A_values = A_dense.data.astype(np.float64)

        cones = moreau.Cones(
            gen_power_cone_params=[([0.3, 0.7], dim2a), ([0.2, 0.3, 0.5], dim2b)],
            num_nonneg_cones=1,
        )
        q = rng.uniform(-0.5, 0.5, size=n)
        b = np.zeros(m)
        b[-1] = -5.0

        prob = {
            "n": n,
            "m": m,
            "P_row_offsets": P_row_offsets,
            "P_col_indices": P_col_indices,
            "P_values": P_values,
            "A_row_offsets": A_row_offsets,
            "A_col_indices": A_col_indices,
            "A_values": A_values,
            "cones": cones,
            "q": q,
            "b": b,
        }
        assert _gradcheck_warm(prob, perturb="q")

    def test_gradcheck_dim2_equals_1(self, device):
        """Gradcheck edge case: dim2=1, alpha=[0.5, 0.5] (minimal w dimension) (warm-FD)."""
        if device != "cpu":
            pytest.skip("warm-start FD is CPU-only; CUDA gradcheck via test_diff.cpp")
        prob = _make_genpow_problem([0.5, 0.5], dim2=1, seed=100)
        assert _gradcheck_warm(prob, perturb="q")

    def test_gradcheck_random_A(self, device):
        """Gradcheck with a random (non-identity) A matrix for realistic coverage (warm-FD)."""
        if device != "cpu":
            pytest.skip("warm-start FD is CPU-only; CUDA gradcheck via test_diff.cpp")
        alphas = [0.4, 0.6]
        dim2 = 2
        dim1 = len(alphas)
        cone_dim = dim1 + dim2
        n = cone_dim
        m = cone_dim + 1

        rng = np.random.default_rng(777)

        P_row_offsets = np.arange(n + 1, dtype=np.int64)
        P_col_indices = np.arange(n, dtype=np.int64)
        P_values = rng.uniform(1.0, 3.0, size=n)

        # Random A: cone rows have diagonal dominance for feasibility,
        # last row is sum constraint
        A_dense = np.zeros((m, n))
        for i in range(cone_dim):
            A_dense[i, i] = -rng.uniform(0.8, 1.2)
            for j in range(n):
                if j != i:
                    A_dense[i, j] = rng.uniform(-0.1, 0.1)
        A_dense[-1, :] = -1.0

        A_sp = sparse.csr_array(A_dense)
        A_row_offsets = A_sp.indptr.astype(np.int64)
        A_col_indices = A_sp.indices.astype(np.int64)
        A_values = A_sp.data.astype(np.float64)

        cones = moreau.Cones(
            gen_power_cone_params=[(list(alphas), dim2)],
            num_nonneg_cones=1,
        )

        q = rng.uniform(-0.3, 0.3, size=n)
        b = np.zeros(m)
        b[-1] = -5.0

        prob = {
            "n": n,
            "m": m,
            "P_row_offsets": P_row_offsets,
            "P_col_indices": P_col_indices,
            "P_values": P_values,
            "A_row_offsets": A_row_offsets,
            "A_col_indices": A_col_indices,
            "A_values": A_values,
            "cones": cones,
            "q": q,
            "b": b,
        }
        assert _gradcheck_warm(prob, perturb="q")


# ---------------------------------------------------------------------------
# Validation / error handling
# ---------------------------------------------------------------------------
class TestGenPowerConeValidation:
    """Tests that invalid GenPowerCone parameters are rejected."""

    def test_alphas_not_summing_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            moreau.Cones(gen_power_cone_params=[([0.3, 0.6], 2)])

    def test_alpha_not_positive(self):
        with pytest.raises(ValueError, match="must be > 0"):
            moreau.Cones(gen_power_cone_params=[([0.0, 1.0], 2)])

    def test_alpha_negative(self):
        with pytest.raises(ValueError, match="must be > 0"):
            moreau.Cones(gen_power_cone_params=[([-0.5, 1.5], 2)])

    def test_dim2_zero(self):
        with pytest.raises(ValueError, match="dim2 must be an integer >= 1"):
            moreau.Cones(gen_power_cone_params=[([0.5, 0.5], 0)])

    def test_empty_alphas(self):
        with pytest.raises(ValueError, match="non-empty"):
            moreau.Cones(gen_power_cone_params=[([], 2)])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
