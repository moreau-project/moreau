"""Integration tests for direct-x cones — features other than plain
forward/backward parity.

Covers:
- CUDA torch wrapper threading per-kind params (alpha, alphas, dim2)
  for asymmetric direct-x cones (exp, power, gen_power).
- Warm-start behaviour: iter reduction with z_x, fallback when z_x
  is omitted, CUDA torch warm-start path.
- Active-set solver rejecting direct-x cones (and exotic slack cones)
  instead of silently dropping them.
- CPU torch auto-resolver routing direct-x problems to IPM, not
  active-set.
- PSD slack cone combined with nonneg direct-x via moreau.Solver and
  CompiledSolver (chordal default and chordal-disabled paths).
- CUDA torch backward zero-copy through `dz_x_ptr` matches numpy.
- Woodbury KKT solver with nonneg direct-x agreeing with cuDSS.
"""

import numpy as np
import pytest
from scipy import sparse

import moreau


@pytest.mark.parametrize(
    "kind,extra_kwargs,q_vals",
    [
        ("exp", {}, [0.0, -1.0, -5.0]),
        ("power", {"alpha": 0.4}, [-2.0, -3.0, -1.0]),
        ("gen_power", {"alphas": [0.4, 0.6], "dim2": 1}, [-2.0, -3.0, -1.0]),
    ],
)
def test_cuda_torch_asymmetric_direct_x_kinds(kind, extra_kwargs, q_vals):
    """CUDA torch wrapper accepts all asymmetric direct-x kinds and threads
    per-kind parameters (alpha for Power, alphas/dim2 for GenPower) through.

    Regression: the torch wrapper's kind_map originally only listed
    'nonneg' and 'soc', raising NotImplementedError on Exp/Power/GenPower
    even though the CUDA backend natively supports them.
    """
    pytest.importorskip("torch")
    import torch
    from moreau.torch import Solver
    from moreau._backend import device_available

    if not device_available("cuda"):
        pytest.skip("CUDA backend not available")

    n = 3
    P_ro = torch.tensor([0, 1, 2, 3])
    P_ci = torch.tensor([0, 1, 2])
    A_ro = torch.tensor([0])
    A_ci = torch.tensor([], dtype=torch.int64)
    cones = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind=kind, indices=[0, 1, 2], **extra_kwargs)],
    )
    settings = moreau.Settings(
        device="cuda",
        verbose=False,
        ipm_settings=moreau.IPMSettings(equilibrate_enable=False),
    )
    solver = Solver(
        n=n,
        m=0,
        P_row_offsets=P_ro,
        P_col_indices=P_ci,
        A_row_offsets=A_ro,
        A_col_indices=A_ci,
        cones=cones,
        settings=settings,
    )
    P_values = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
    A_values = torch.tensor([], dtype=torch.float64)
    q = torch.tensor(q_vals, dtype=torch.float64)
    b = torch.tensor([], dtype=torch.float64)

    sol = solver.solve(P_values, A_values, q, b)
    # Must produce a finite primal x in the cone — the wrapper used to drop
    # alpha/alphas/dim2, producing wrong solutions even when it didn't crash.
    assert sol.x.shape == (n,)
    x = sol.x.detach().cpu().numpy()
    assert np.all(np.isfinite(x)), f"{kind} produced non-finite x: {x}"


def test_direct_x_warm_start_reduces_iters(device):
    """Warm-starting a direct-x problem with its own solution must converge
    in **fewer** iterations than a cold solve. Regression for the
    `warm_z_x` binding plumbing on both backends — without a `warm_z_x`
    parameter the user-supplied direct-x dual is silently dropped, so
    warm-start gives the same iter count as cold (best case) or worse
    (typical, because warm_x is used but z_x stays at default-init)."""
    from moreau._backend import device_available

    if not device_available(device):
        pytest.skip(f"{device} backend not available")

    ipm = moreau.IPMSettings(presolve_enable=False, equilibrate_enable=False)
    n = 3
    P = sparse.eye(n, format="csr") * 0.1
    q = np.array([-0.5, -0.5, 5.0])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind="gen_power", indices=[0, 1, 2], alphas=[0.3, 0.7], dim2=1)]
    )

    settings = moreau.Settings(
        device=device,
        solver="ipm",
        ipm_settings=ipm,
        verbose=False,
        max_iter=300,
    )
    cold = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    sol_cold = cold.solve()
    iter_cold = cold.info.iterations
    assert cold.info.status.name == "Solved"

    ws = sol_cold.to_warm_start()
    assert (
        ws.z_x is not None and len(ws.z_x) == 3
    ), "Solution.to_warm_start() must carry z_x for direct-x problems"

    warm = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    warm.solve(warm_start=ws)
    iter_warm = warm.info.iterations
    assert warm.info.status.name == "Solved"

    # The warm point IS the optimal solution; the IPM should converge in
    # substantially fewer iters. A small margin (~2) absorbs implementation
    # noise; pre-fix this was iter_warm >= iter_cold for direct-x.
    assert iter_warm < iter_cold - 2, (
        f"Direct-x warm-start did not reduce iter count: " f"cold={iter_cold}, warm={iter_warm}"
    )


def test_direct_x_warm_start_without_z_x_uses_default_init(device):
    """Calling `solve(warm_start=WarmStart(x, z, s))` on a direct-x problem
    WITHOUT `warm_z_x` must fall back to default direct-x cone-block
    initialisation — not reuse z_x from a prior solve. Pre-fix the
    warm-start only refreshed the slack `(z, s)` smoothing, leaving the
    direct-x dual `z_x` (and a boundary-valued warm `x`) stranded on the
    cone face, so a second warm-start solve failed to converge.
    """
    from moreau._backend import device_available

    if not device_available(device):
        pytest.skip(f"{device} backend not available")

    ipm = moreau.IPMSettings(presolve_enable=False, equilibrate_enable=False)
    n = 3
    P = sparse.eye(n, format="csr") * 0.1
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind="gen_power", indices=[0, 1, 2], alphas=[0.3, 0.7], dim2=1)]
    )
    settings = moreau.Settings(
        device=device,
        solver="ipm",
        ipm_settings=ipm,
        verbose=False,
        max_iter=300,
    )

    # Cold solve on problem A — produces a z_x that would be wrong for
    # problem B.
    q_a = np.array([-0.5, -0.5, 5.0])
    sol_a = moreau.Solver(P, q_a, A, b, cones=cones, settings=settings).solve()
    assert sol_a is not None

    # Warm-start problem B with x/z/s only — no warm_z_x. The fallback
    # must re-initialise z_x; we must still converge cleanly.
    q_b = np.array([1.0, 1.0, 2.0])
    ws_no_zx = moreau.WarmStart(
        x=np.asarray(sol_a.x).reshape(-1),
        z=np.asarray(sol_a.z).reshape(-1) if sol_a.z.size else np.zeros(0),
        s=np.asarray(sol_a.s).reshape(-1) if sol_a.s.size else np.zeros(0),
        z_x=None,
    )
    warm_b = moreau.Solver(P, q_b, A, b, cones=cones, settings=settings)
    warm_b.solve(warm_start=ws_no_zx)
    assert warm_b.info.status.name in ("Solved", "AlmostSolved"), (
        f"Warm-start without z_x failed to converge: {warm_b.info.status.name}. "
        "The warm-start fallback must cold-init the direct-x cone block "
        "rather than reuse stale state from a prior solve."
    )


def test_active_set_solver_rejects_x_cones():
    """ActiveSetSolver only knows zero + nonneg slack cones. If a user
    explicitly forces ``solver='active_set'`` on a problem that carries
    `cones.x_cones`, the solver previously silently dropped the direct-x
    cones and returned a wrong (constraint-violating) solution. Validate
    that the constructor now errors loudly so the bug cannot recur.
    """
    n = 3
    cones = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=[0, 1, 2])],
    )
    with pytest.raises(ValueError, match="does not support direct-x cones"):
        moreau.CompiledSolver(
            n=n,
            m=0,
            P_row_offsets=[0, 1, 2, 3],
            P_col_indices=[0, 1, 2],
            A_row_offsets=[0],
            A_col_indices=[],
            cones=cones,
            settings=moreau.Settings(
                device="cpu",
                solver="active_set",
                batch_size=2,
            ),
        )


def test_active_set_solver_rejects_exotic_slack_cones():
    """ActiveSetSolver must also reject SOC / Exp / Power / PSD / GenPower
    slack cones it cannot handle, instead of silently dropping them."""
    n = 3
    cones = moreau.Cones(num_zero_cones=1, num_exp_cones=1)
    with pytest.raises(ValueError, match="does not support"):
        moreau.CompiledSolver(
            n=n,
            m=4,
            P_row_offsets=[0, 1, 2, 3],
            P_col_indices=[0, 1, 2],
            A_row_offsets=[0, 1, 2, 3, 4],
            A_col_indices=[0, 0, 1, 2],
            cones=cones,
            settings=moreau.Settings(
                device="cpu",
                solver="active_set",
                batch_size=1,
            ),
        )


def test_cpu_torch_auto_solver_skips_active_set_when_x_cones():
    """`Settings(device='cpu')` (auto) must NOT route x_cones problems
    through the active-set CPU solver — active-set ignores x_cones and
    silently produces a constraint-violating solution with empty z_x.
    Regression: the previous auto-resolver only checked slack cone
    counts; direct-x cones with zero+nonneg slack passed the QP-only
    test and went to active-set."""
    pytest.importorskip("torch")
    import torch
    from moreau.torch import Solver as TorchSolver

    n = 4
    cones = moreau.Cones(
        num_zero_cones=1,
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=list(range(n)))],
    )
    solver = TorchSolver(
        n=n,
        m=1,
        P_row_offsets=torch.tensor(list(range(n + 1)), dtype=torch.int64),
        P_col_indices=torch.tensor(list(range(n)), dtype=torch.int64),
        A_row_offsets=torch.tensor([0, n], dtype=torch.int64),
        A_col_indices=torch.tensor(list(range(n)), dtype=torch.int64),
        cones=cones,
        settings=moreau.Settings(device="cpu", enable_grad=True),  # solver='auto'
    )
    Pv = torch.tensor([1.0] * n, dtype=torch.float64, requires_grad=True)
    Av = torch.tensor([1.0] * n, dtype=torch.float64, requires_grad=True)
    qt = torch.tensor([2.0, 1.0, -1.0, 0.5], dtype=torch.float64, requires_grad=True)
    bt = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
    sol = solver.solve(Pv, Av, qt, bt)

    # IPM (correct) returns z_x of length n; active-set silently drops it.
    assert sol.z_x.numel() == n, "auto routed direct-x to active-set (z_x missing)"
    # Solution must respect the nonneg constraint (active-set produces
    # negative entries for this problem with sum(x)=1; q[2]=-1 wins).
    x_np = sol.x.detach().numpy()
    assert (x_np >= -1e-6).all(), f"x violates nonneg constraint: {x_np}"


# ---------- PSD slack + direct-x mixed ----------


def _build_psd_dx_problem():
    """min 0.5||x||² + q'x  s.t.  x[:3] forms a 2x2 PSD svec, x[3] >= 0,
    with q[3] = -1 driving x[3] to 1.0."""
    n = 4
    P = sparse.diags([1.0] * n, format="csr")
    q = np.array([0.0, 0.0, 0.0, -1.0])
    A = sparse.csr_matrix(np.hstack([np.eye(3), np.zeros((3, 1))]))
    b = np.zeros(3)
    cones = moreau.Cones(
        psd_dims=[2],
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=[3])],
    )
    return P, q, A, b, cones


def test_psd_slack_plus_direct_x_solver_chordal_default():
    """Slack PSD + nonneg direct-x via moreau.Solver, chordal on (default).
    Regression: previously rejected with `direct-x cones combined with PSD
    slack cones not yet supported`."""
    P, q, A, b, cones = _build_psd_dx_problem()
    s = moreau.Solver(P, q, A, b, cones)
    sol = s.solve()
    assert abs(sol.x[3] - 1.0) < 1e-6
    # PSD entries should be ~0 (interior of PSD with q[:3]=0 → svec=0).
    assert max(abs(v) for v in sol.x[:3]) < 1e-4


def test_psd_slack_plus_direct_x_compiled_solver():
    """Slack PSD + nonneg direct-x via CompiledSolver."""
    P, q, A, b, cones = _build_psd_dx_problem()
    settings = moreau.Settings(batch_size=1)
    s = moreau.CompiledSolver(
        n=4,
        m=3,
        P_row_offsets=[0, 1, 2, 3, 4],
        P_col_indices=[0, 1, 2, 3],
        A_row_offsets=[0, 1, 2, 3],
        A_col_indices=[0, 1, 2],
        cones=cones,
        settings=settings,
    )
    s.setup(P_values=[1.0] * 4, A_values=[1.0] * 3)
    sol = s.solve(qs=[q.tolist()], bs=[b.tolist()])
    assert abs(sol.x[0][3] - 1.0) < 1e-6


def test_psd_slack_plus_direct_x_chordal_disabled():
    """Same problem with chordal disabled to lock in the workaround
    knob exposed via IPMSettings."""
    P, q, A, b, cones = _build_psd_dx_problem()
    settings = moreau.Settings(
        ipm_settings=moreau.IPMSettings(chordal_decomposition_enable=False),
    )
    s = moreau.Solver(P, q, A, b, cones, settings)
    sol = s.solve()
    assert abs(sol.x[3] - 1.0) < 1e-6


# ---------- CUDA torch zero-copy + warm-start + Woodbury ----------


def test_cuda_torch_warm_start_with_direct_x():
    """CUDA torch warm-start path threads direct-x z_x through correctly.
    Regression: previously rejected at torch_wrapper module with
    `Warm-started solve with direct-x cones is not yet supported`."""
    pytest.importorskip("torch")
    pytest.importorskip("moreau_cuda")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from moreau.torch import Solver as TorchSolver

    n = 4
    cones = moreau.Cones(
        num_zero_cones=1,
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=list(range(n)))],
    )
    solver = TorchSolver(
        n=n,
        m=1,
        P_row_offsets=torch.tensor(list(range(n + 1)), dtype=torch.int64),
        P_col_indices=torch.tensor(list(range(n)), dtype=torch.int64),
        A_row_offsets=torch.tensor([0, n], dtype=torch.int64),
        A_col_indices=torch.tensor(list(range(n)), dtype=torch.int64),
        cones=cones,
        settings=moreau.Settings(device="cuda"),
    )
    P_values = torch.tensor([1.0] * n, device="cuda", dtype=torch.float64)
    A_values = torch.tensor([1.0] * n, device="cuda", dtype=torch.float64)
    q1 = torch.tensor([2.0, 1.0, -1.0, 0.5], device="cuda", dtype=torch.float64)
    q2 = torch.tensor([2.0, 1.1, -1.0, 0.5], device="cuda", dtype=torch.float64)
    b = torch.tensor([1.0], device="cuda", dtype=torch.float64)

    sol1 = solver.solve(P_values, A_values, q1, b)
    assert sol1.z_x is not None and sol1.z_x.numel() == n
    ws = sol1.to_warm_start()
    assert ws.z_x is not None
    sol2 = solver.solve(P_values, A_values, q2, b, warm_start=ws)
    # Cold reference for the perturbed problem
    sol2_cold = solver.solve(P_values, A_values, q2, b)
    assert (sol2.x - sol2_cold.x).abs().max().item() < 1e-5


def test_cuda_torch_dz_x_zerocopy_matches_numpy():
    """CUDA torch backward with non-trivial dz_x flows zero-copy through
    `backward_to_device_pointers(..., dz_x_ptr)` (rather than the prior
    host-array round trip) and matches the numpy reference.
    Regression: torch_wrapper module used to copy to CPU and back."""
    pytest.importorskip("torch")
    pytest.importorskip("moreau_cuda")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    n = 4
    P = sparse.diags([1.0] * n, format="csr")
    A = sparse.csr_matrix(np.array([[1.0] * n]))
    b = np.array([1.0])
    q = np.array([2.0, 1.0, -1.0, 0.5])
    cones = moreau.Cones(
        num_zero_cones=1,
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=list(range(n)))],
    )

    # Numpy reference
    sNp = moreau.Solver(P, q, A, b, cones, moreau.Settings(enable_grad=True))
    sNp.solve()
    res_np = sNp.backward(dx=np.zeros(n), dz=np.zeros(1), ds=np.zeros(1), dz_x=np.ones(n))

    # CUDA torch path
    from moreau.torch import Solver as TorchSolver

    sCu = TorchSolver(
        n=n,
        m=1,
        P_row_offsets=torch.tensor(list(range(n + 1)), dtype=torch.int64),
        P_col_indices=torch.tensor(list(range(n)), dtype=torch.int64),
        A_row_offsets=torch.tensor([0, n], dtype=torch.int64),
        A_col_indices=torch.tensor(list(range(n)), dtype=torch.int64),
        cones=cones,
        settings=moreau.Settings(device="cuda", enable_grad=True),
    )
    Pv = torch.tensor([1.0] * n, device="cuda", requires_grad=True, dtype=torch.float64)
    Av = torch.tensor([1.0] * n, device="cuda", requires_grad=True, dtype=torch.float64)
    qt = torch.tensor(q.tolist(), device="cuda", requires_grad=True, dtype=torch.float64)
    bt = torch.tensor(b.tolist(), device="cuda", requires_grad=True, dtype=torch.float64)
    sol = sCu.solve(Pv, Av, qt, bt)
    sol.z_x.sum().backward()  # dz_x = ones

    dq_cu = qt.grad.cpu().numpy()
    db_cu = bt.grad.cpu().numpy()
    dP_cu = Pv.grad.cpu().numpy()
    dA_cu = Av.grad.cpu().numpy()
    assert np.max(np.abs(res_np["dq"].squeeze() - dq_cu)) < 1e-6
    assert np.max(np.abs(res_np["db"].squeeze() - db_cu)) < 1e-6
    assert np.max(np.abs(res_np["dP_values"].squeeze() - dP_cu)) < 1e-6
    assert np.max(np.abs(res_np["dA_values"].squeeze() - dA_cu)) < 1e-6


def test_woodbury_with_nonneg_direct_x_matches_cudss():
    """Woodbury solver with nonneg direct-x cones (diagonal P).
    Woodbury exploits the same structure that nonneg direct-x preserves
    (diagonal contribution to (1,1)) — should give an answer matching
    the general cuDSS path within IPM tolerance.

    Regression: previously rejected at WoodburyKKTData::isCompatible
    with `requires diagonal P, zero + nonneg cones only, k_total < n`.
    """
    pytest.importorskip("moreau_cuda")
    n = 4
    P = sparse.diags([1.0] * n, format="csr")
    q = np.array([2.0, 1.0, -1.0, 0.5])
    A = sparse.csr_matrix(np.array([[1.0, 1.0, 1.0, 1.0]]))
    b = np.array([1.0])
    cones = moreau.Cones(
        num_zero_cones=1,
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=list(range(n)))],
    )
    s_cudss = moreau.Solver(
        P,
        q,
        A,
        b,
        cones,
        moreau.Settings(
            device="cuda", ipm_settings=moreau.IPMSettings(direct_solve_method="cudss")
        ),
    )
    sol_cudss = s_cudss.solve()
    s_wb = moreau.Solver(
        P,
        q,
        A,
        b,
        cones,
        moreau.Settings(
            device="cuda", ipm_settings=moreau.IPMSettings(direct_solve_method="woodbury")
        ),
    )
    sol_wb = s_wb.solve()
    # Tolerance accounts for different solvers' iterate trajectories.
    assert max(abs(a - b) for a, b in zip(sol_cudss.x, sol_wb.x)) < 1e-5
