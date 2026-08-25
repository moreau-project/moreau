"""Backward-pass tests for direct-x cones.

Covers:
- moreau.Solver.backward(): direct-x dP/dq parity with the equivalent
  slack form, for nonneg + PSD.
- CompiledSolver.backward() in batch mode vs per-problem reference.
- Finite-difference parity for d(z_x[j])/dq.
- torch.Solver autograd through Solution.z_x (CPU + CUDA).
- jax.Solver autograd through JaxSolution.z_x (CPU + CUDA).
"""

import numpy as np
import pytest
from scipy import sparse

import moreau


def test_solver_with_x_cones_backward_matches_slack():
    # Backward via the unfold path: dP and dq should match the slack-
    # form reference. dA / db for a pure direct-x problem (no slack
    # rows) are empty.
    P = sparse.diags([2.0, 2.0, 2.0], format="csr")
    q = np.array([-1.0, 2.0, -0.5])

    # Direct-x form (no slack rows)
    cones_direct = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=[0, 1, 2])],
    )
    A_dx = sparse.csr_matrix(np.zeros((0, 3)))
    b_dx = np.array([])
    # Force IPM on slack reference so both paths use the same solver.
    settings = moreau.Settings(
        device="cpu",
        verbose=False,
        solver="ipm",
        enable_grad=True,
    )
    solver_direct = moreau.Solver(P, q, A_dx, b_dx, cones=cones_direct, settings=settings)
    solver_direct.solve()
    dx = np.array([1.0, 1.0, 1.0])
    grads_direct = solver_direct.backward(dx)

    # Slack reference
    A_slack = sparse.csr_matrix(-np.eye(3))
    b_slack = np.zeros(3)
    cones_slack = moreau.Cones(num_nonneg_cones=3)
    solver_slack = moreau.Solver(P, q, A_slack, b_slack, cones=cones_slack, settings=settings)
    solver_slack.solve()
    grads_slack = solver_slack.backward(dx)

    np.testing.assert_allclose(grads_direct["dq"], grads_slack["dq"], atol=1e-5)
    np.testing.assert_allclose(
        grads_direct["dP_values"],
        grads_slack["dP_values"],
        atol=1e-5,
    )
    assert len(grads_direct["db"]) == 0
    assert len(grads_direct["dA_values"]) == 0


def test_solver_with_x_cones_cpu_psd_backward_matches_slack():
    # Backward via unfold: PSD direct-x must match slack dP/dq.
    P = sparse.diags([1.0, 1.0, 1.0], format="csr")
    q = np.array([0.0, -1.0, 0.0])

    cones_direct = moreau.Cones(
        x_cones=[
            moreau.XConeSpec(
                kind="psd_triangle",
                indices=[0, 1, 2],
                psd_k=2,
            )
        ],
    )
    A_dx = sparse.csr_matrix(np.zeros((0, 3)))
    b_dx = np.array([])
    settings = moreau.Settings(
        device="cpu",
        verbose=False,
        solver="ipm",
        enable_grad=True,
    )
    solver_direct = moreau.Solver(P, q, A_dx, b_dx, cones=cones_direct, settings=settings)
    solver_direct.solve()
    dx = np.array([1.0, 1.0, 1.0])
    grads_direct = solver_direct.backward(dx)

    A_slack = sparse.csr_matrix(-np.eye(3))
    b_slack = np.zeros(3)
    cones_slack = moreau.Cones(psd_dims=[2])
    solver_slack = moreau.Solver(P, q, A_slack, b_slack, cones=cones_slack, settings=settings)
    solver_slack.solve()
    grads_slack = solver_slack.backward(dx)

    np.testing.assert_allclose(grads_direct["dq"], grads_slack["dq"], atol=1e-5)
    np.testing.assert_allclose(
        grads_direct["dP_values"],
        grads_slack["dP_values"],
        atol=1e-5,
    )
    assert len(grads_direct["db"]) == 0
    assert len(grads_direct["dA_values"]) == 0


def _batched_backward_matches_per_problem(device):
    """Helper: batched CompiledSolver.backward() with direct-x cones must
    match per-problem reference gradients on the given device."""
    from moreau._backend import device_available

    if not device_available(device):
        pytest.skip(f"{device} backend not available")

    n = 3
    cones = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=[0, 1, 2])],
    )
    P_row_offsets = [0, 1, 2, 3]
    P_col_indices = [0, 1, 2]
    A_row_offsets = [0]
    A_col_indices = []

    settings = moreau.Settings(
        device=device,
        batch_size=2,
        enable_grad=True,
        verbose=False,
    )
    solver = moreau.CompiledSolver(
        n=n,
        m=0,
        P_row_offsets=P_row_offsets,
        P_col_indices=P_col_indices,
        A_row_offsets=A_row_offsets,
        A_col_indices=A_col_indices,
        cones=cones,
        settings=settings,
    )

    # Two problems sharing P structure; first leaves cone interior, second
    # binds asymmetrically — exercises per-batch H_x dependence.
    P_values = np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])
    A_values = np.zeros((2, 0))
    qs = np.array([[-1.0, -1.0, -1.0], [-0.5, -0.5, 1.0]])
    bs = np.zeros((2, 0))

    solver.setup(P_values, A_values)
    solver.solve(qs, bs)
    statuses = [s.name for s in solver.info.status]
    assert all(st in ("Solved", "AlmostSolved") for st in statuses), statuses

    dxs = np.ones((2, n))
    dzs = np.zeros((2, 0))
    dss = np.zeros((2, 0))
    grads = solver.backward(dxs, dzs, dss)

    # Per-problem reference using single-problem Solver.
    P_dense = np.diag([2.0, 2.0, 2.0])
    P_sparse = sparse.csr_matrix(P_dense)
    A_ref = sparse.csr_matrix(np.zeros((0, n)))
    b_ref = np.array([])
    for b_idx in range(2):
        ref_settings = moreau.Settings(
            device="cpu",
            solver="ipm",
            enable_grad=True,
            verbose=False,
        )
        ref = moreau.Solver(P_sparse, qs[b_idx], A_ref, b_ref, cones=cones, settings=ref_settings)
        ref.solve()
        ref_grads = ref.backward(dxs[b_idx])
        np.testing.assert_allclose(
            grads["dq"][b_idx],
            ref_grads["dq"],
            atol=1e-4,
            err_msg=f"batch {b_idx} dq disagreement on {device}",
        )


def test_compiled_solver_xcone_backward_batch_cpu():
    _batched_backward_matches_per_problem("cpu")


def test_compiled_solver_xcone_backward_batch_cuda():
    _batched_backward_matches_per_problem("cuda")


def _check_dz_x_finite_difference(device):
    """Helper: backward(dz_x=e_j) must match central FD of z_x[j] on the
    given device. Active-boundary problem keeps the test well-conditioned."""
    from moreau._backend import device_available

    if not device_available(device):
        pytest.skip(f"{device} backend not available")

    n = 3
    P = sparse.diags([1.0, 1.0, 1.0], format="csr")
    q = np.array([-0.5, -0.5, 1.0])  # active boundary on x[2] = 0
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=[0, 1, 2])],
    )
    # Equilibration is a non-smooth rescaling that produces asymmetric
    # finite differences at the cone boundary; disable it to keep the FD
    # reference clean. The IFT-direct math is invariant under uniform
    # per-cone equilibration.
    ipm = moreau.IPMSettings(equilibrate_enable=False)
    settings = moreau.Settings(
        device=device,
        solver="ipm",
        enable_grad=True,
        verbose=False,
        ipm_settings=ipm,
    )

    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    sol = solver.solve()
    assert sol.z_x.shape == (n,)

    # Analytical d(z_x[j])/dq via backward with dz_x = e_j.
    j = 2
    dx = np.zeros(n)
    dz_x = np.zeros(n)
    dz_x[j] = 1.0
    grads = solver.backward(dx, dz_x=dz_x)
    analytic = grads["dq"]

    eps = 1e-6
    fd = np.zeros(n)
    for k in range(n):
        q_plus = q.copy()
        q_plus[k] += eps
        s_plus = moreau.Solver(P, q_plus, A, b, cones=cones, settings=settings)
        sol_plus = s_plus.solve()
        q_minus = q.copy()
        q_minus[k] -= eps
        s_minus = moreau.Solver(P, q_minus, A, b, cones=cones, settings=settings)
        sol_minus = s_minus.solve()
        fd[k] = (sol_plus.z_x[j] - sol_minus.z_x[j]) / (2 * eps)

    np.testing.assert_allclose(analytic, fd, atol=1e-3)


def test_solver_dz_x_finite_difference_cpu():
    """Upstream gradient on z_x must match finite differences of z_x_orig.

    Mirrors the Rust integration test (`xcone_dz_x_backward`) but at the
    Python API level.
    """
    _check_dz_x_finite_difference("cpu")


def test_solver_dz_x_finite_difference_cuda():
    """Same dz_x parity check as the CPU test, but on CUDA."""
    _check_dz_x_finite_difference("cuda")


def _check_torch_autograd_through_z_x(device):
    """torch.Solver: backprop through Solution.z_x[2] (active boundary)
    yields dq[2] ≈ 1 and zeros elsewhere on the given device."""
    pytest.importorskip("torch")
    import torch
    from moreau._backend import device_available
    from moreau.torch import Solver

    if not device_available(device):
        pytest.skip(f"{device} backend not available")

    n = 3
    P_ro = torch.tensor([0, 1, 2, 3])
    P_ci = torch.tensor([0, 1, 2])
    A_ro = torch.tensor([0])
    A_ci = torch.tensor([], dtype=torch.int64)
    cones = moreau.Cones(x_cones=[moreau.XConeSpec(kind="nonneg", indices=[0, 1, 2])])
    ipm = moreau.IPMSettings(equilibrate_enable=False)
    settings = moreau.Settings(
        device=device,
        enable_grad=True,
        verbose=False,
        ipm_settings=ipm,
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
    q = torch.tensor([-0.5, -0.5, 1.0], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([], dtype=torch.float64)

    sol = solver.solve(P_values, A_values, q, b)
    assert sol.z_x.shape == (n,)
    sol.z_x[2].backward()
    expected = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    torch.testing.assert_close(q.grad.cpu(), expected, atol=1e-3, rtol=1e-3)


def test_torch_solver_direct_x_autograd_through_z_x_cpu():
    """torch.Solver: backprop through Solution.z_x produces correct dq."""
    _check_torch_autograd_through_z_x("cpu")


def test_torch_solver_direct_x_autograd_through_z_x_cuda():
    _check_torch_autograd_through_z_x("cuda")


def _check_jax_autograd_through_z_x(device):
    """JAX direct-x autograd: forward exposes `z_x` on JaxSolution; the
    custom_vjp backward threads `dz_x` through. Asserts gradients match
    a numpy moreau.Solver reference (dz_x = ones)."""
    pytest.importorskip("jax")
    import jax
    import jax.numpy as jnp
    from moreau._backend import device_available
    from moreau.jax import Solver as JaxSolver

    if not device_available(device):
        pytest.skip(f"{device} backend not available")
    if device == "cuda":
        if not any(d.platform == "gpu" for d in jax.devices()):
            pytest.skip("CUDA JAX device not available")

    n = 4
    cones = moreau.Cones(
        num_zero_cones=1,
        x_cones=[moreau.XConeSpec(kind="nonneg", indices=list(range(n)))],
    )
    solver = JaxSolver(
        n=n,
        m=1,
        P_row_offsets=jnp.array(list(range(n + 1)), dtype=jnp.int64),
        P_col_indices=jnp.array(list(range(n)), dtype=jnp.int64),
        A_row_offsets=jnp.array([0, n], dtype=jnp.int64),
        A_col_indices=jnp.array(list(range(n)), dtype=jnp.int64),
        cones=cones,
        settings=moreau.Settings(device=device, solver="ipm"),
    )
    if device == "cuda":
        assert solver.device == "cuda"

    P_values = jnp.array([1.0] * n)
    A_values = jnp.array([1.0] * n)
    q = jnp.array([2.0, 1.0, -1.0, 0.5])
    b = jnp.array([1.0])

    sol = solver.solve(P_values, A_values, q, b)
    assert sol.z_x.shape == (n,)

    def loss(q_):
        return jnp.sum(solver.solve(P_values, A_values, q_, b).z_x)

    dq = np.asarray(jax.grad(loss)(q))

    P = sparse.diags([1.0] * n, format="csr")
    A = sparse.csr_matrix(np.array([[1.0] * n]))
    sNp = moreau.Solver(
        P, np.asarray(q), A, np.asarray(b), cones, moreau.Settings(enable_grad=True)
    )
    sNp.solve()
    res = sNp.backward(dx=np.zeros(n), dz=np.zeros(1), ds=np.zeros(1), dz_x=np.ones(n))
    dq_ref = np.asarray(res["dq"]).squeeze()
    assert np.max(np.abs(dq - dq_ref)) < 1e-5


def test_cuda_jax_solver_direct_x_autograd_through_z_x():
    """CUDA JAX direct-x autograd: forward exposes `z_x` on JaxSolution
    via the FFI z_x output buffer; the custom_vjp backward threads `dz_x`
    through the FFI dz_x input. Regression: CUDA JAX FFI previously had
    no x_cone slots and silently skipped direct-x cones."""
    pytest.importorskip("moreau_cuda")
    _check_jax_autograd_through_z_x("cuda")


def test_jax_solver_direct_x_autograd_through_z_x():
    """JAX (CPU) direct-x autograd. Regression: previously
    `JaxSolver(cones=Cones(x_cones=...))` raised `NotImplementedError`."""
    _check_jax_autograd_through_z_x("cpu")
