"""Forward-solve tests for direct-x cones.

Compares moreau.Solver with `dir_cones=...` against the equivalent slack
form across CPU + CUDA backends, and covers the `enable_grad=True`
construction flag for each supported direct-x kind. Backward parity
(slack reference + finite differences + torch/jax autograd) lives in
test_xcone_backward.py.
"""

import numpy as np
import pytest
from scipy import sparse

import moreau


def test_solver_with_dir_cones_cpu_matches_slack():
    # CPU direct-x forward pass is wired via DirectXSolverCpu. Compare
    # `min 0.5x'x + q'x s.t. x >= 0, [A|b constraint]` against the
    # equivalent slack form.
    P = sparse.diags([1.0, 1.0], format="csr")
    q = np.array([-0.5, 1.0])  # unconstrained opt is (0.5, -1); clipped to (0.5, 0)
    A = sparse.csr_array([[1.0, 1.0]])
    b = np.array([1.0])

    cones_direct = moreau.Cones(
        num_zero_cones=1,
        dir_cones=[moreau.DirectConeSpec(kind="nonneg", indices=[0, 1])],
    )
    settings = moreau.Settings(device="cpu", verbose=False)
    solver_direct = moreau.Solver(P, q, A, b, cones=cones_direct, settings=settings)
    sol_direct = solver_direct.solve()

    # Slack reference: add two nonneg rows `-x[i] + s_i = 0` with s_i >= 0.
    A_slack = sparse.vstack([A, sparse.csr_array([[-1.0, 0.0], [0.0, -1.0]])])
    b_slack = np.concatenate([b, [0.0, 0.0]])
    cones_slack = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
    solver_slack = moreau.Solver(P, q, A_slack, b_slack, cones=cones_slack, settings=settings)
    sol_slack = solver_slack.solve()

    np.testing.assert_allclose(sol_direct.x, sol_slack.x, atol=1e-6)


def test_solver_with_dir_cones_cuda_nonneg_scalar():
    # CUDA nonneg direct-x forward. Scalar nonneg QP:
    # min 0.5 x^2 - 4x  s.t. x ≥ 0. Optimum is x = 4 (interior).
    from moreau._backend import device_available

    if not device_available("cuda"):
        pytest.skip("CUDA backend not available")

    P = sparse.diags([1.0], format="csr")
    q = np.array([-4.0])
    A = sparse.csr_matrix(np.zeros((0, 1)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[moreau.DirectConeSpec(kind="nonneg", indices=[0])],
    )
    settings = moreau.Settings(device="cuda", verbose=False)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    solution = solver.solve()
    np.testing.assert_allclose(solution.x, [4.0], atol=1e-5)


def test_solver_with_dir_cones_cuda_soc_dense_boundary_active():
    # CUDA direct-x SOC (dense, dim ≤ 4) landed alongside the nonneg path.
    # Mirrors CPU test_soc_constraint_active_boundary: q=(1, 2, 0) pushes
    # x[0] negative; optimum lies on the SOC boundary at x* = (0.5, -0.5, 0).
    from moreau._backend import device_available

    if not device_available("cuda"):
        pytest.skip("CUDA backend not available")

    P = sparse.diags([1.0, 1.0, 1.0], format="csr")
    q = np.array([1.0, 2.0, 0.0])
    A = sparse.csr_matrix(np.zeros((0, 3)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[moreau.DirectConeSpec(kind="soc", indices=[0, 1, 2])],
    )
    settings = moreau.Settings(device="cuda", verbose=False)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    solution = solver.solve()
    np.testing.assert_allclose(solution.x, [0.5, -0.5, 0.0], atol=1e-4)


def test_solver_with_dir_cones_cuda_soc_rank2_boundary_active():
    # Rank-2 sparse SOC (dim > 4) on CUDA uses the u/v expansion path
    # end-to-end. q pulls the scalar negative, optimum sits on the SOC
    # boundary with ||x[1..]|| = x[0].
    from moreau._backend import device_available

    if not device_available("cuda"):
        pytest.skip("CUDA backend not available")

    n = 6
    P = sparse.eye(n, format="csr")
    q = np.array([2.0, -1.0, 1.5, -0.5, 0.3, -0.2])
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[moreau.DirectConeSpec(kind="soc", indices=list(range(n)))],
    )
    settings = moreau.Settings(device="cuda", verbose=False)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    solution = solver.solve()
    # SOC boundary: x[0] = ||x[1..]||
    tail = np.linalg.norm(solution.x[1:])
    np.testing.assert_allclose(solution.x[0], tail, atol=1e-4)
    assert solution.x[0] > 0.0


def test_solver_with_dir_cones_cuda_soc_large_dim_boundary_active():
    # Large-dim (> stack-friendly) direct-x SOC on CUDA — kernels stream
    # over cone entries with no dim cap. Exercise dim=40.
    from moreau._backend import device_available

    if not device_available("cuda"):
        pytest.skip("CUDA backend not available")

    n = 40
    P = sparse.eye(n, format="csr")
    # Bias q to make the SOC constraint active at the optimum.
    rng = np.random.default_rng(0)
    q = rng.standard_normal(n)
    q[0] = 3.0  # push scalar of SOC negative via -q
    A = sparse.csr_matrix(np.zeros((0, n)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[moreau.DirectConeSpec(kind="soc", indices=list(range(n)))],
    )
    settings = moreau.Settings(device="cuda", verbose=False)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    solution = solver.solve()
    tail = float(np.linalg.norm(solution.x[1:]))
    np.testing.assert_allclose(solution.x[0], tail, atol=1e-4)
    assert solution.x[0] > 0.0


def test_solver_with_dir_cones_cuda_nonneg_enable_grad_supported():
    # Native CUDA direct-x backward (nonneg) is supported via the
    # IFT-direct augmented-system path; constructing a solver with
    # enable_grad=True on CUDA must not raise.
    P = sparse.diags([1.0], format="csr")
    q = np.array([-4.0])
    A = sparse.csr_matrix(np.zeros((0, 1)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[moreau.DirectConeSpec(kind="nonneg", indices=[0])],
    )
    settings = moreau.Settings(device="cuda", verbose=False, enable_grad=True)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    sol = solver.solve()
    np.testing.assert_allclose(sol.x, [4.0], atol=1e-5)


def test_solver_with_dir_cones_cuda_soc_enable_grad_supported():
    # Native CUDA direct-x SOC backward via IFT-direct: same SOC-active
    # boundary problem as the forward-only test. Constructing the solver
    # with enable_grad=True must not raise.
    P = sparse.diags([1.0, 1.0, 1.0], format="csr")
    q = np.array([1.0, 2.0, 0.0])
    A = sparse.csr_matrix(np.zeros((0, 3)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[moreau.DirectConeSpec(kind="soc", indices=[0, 1, 2])],
    )
    settings = moreau.Settings(device="cuda", verbose=False, enable_grad=True)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    sol = solver.solve()
    np.testing.assert_allclose(sol.x, [0.5, -0.5, 0.0], atol=1e-5)


def test_solver_with_dir_cones_cuda_psd_enable_grad_supported():
    # CUDA direct-x PSD backward is wired natively via cuSOLVER
    # eigendecomp + Ω-matrix Jacobian construction. Constructing a
    # solver with enable_grad=True must not raise.
    P = sparse.diags([1.0, 1.0, 1.0], format="csr")
    q = np.array([0.0, -1.0, 0.0])
    A = sparse.csr_matrix(np.zeros((0, 3)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[moreau.DirectConeSpec(kind="psd_triangle", indices=[0, 1, 2], psd_k=2)],
    )
    settings = moreau.Settings(device="cuda", verbose=False, enable_grad=True)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    solver.solve()
    assert solver.info.status.name in ("Solved", "AlmostSolved")


def test_solver_with_dir_cones_cpu_psd_matches_slack():
    # Direct-x PSD forward on CPU. PSD(2): svec dim 3. Push the
    # unconstrained optimum outside the PSD cone so the constraint is
    # active, then confirm direct-x and slack agree.
    #
    # min 0.5 ||X||_F^2 - X[0,1]  over X ⪰ 0 (2x2).
    # svec ordering: x = (a, b*sqrt(2), c); unconstrained opt (0, 1, 0) is
    # not PSD.
    P = sparse.diags([1.0, 1.0, 1.0], format="csr")
    q = np.array([0.0, -1.0, 0.0])

    cones_direct = moreau.Cones(
        dir_cones=[
            moreau.DirectConeSpec(
                kind="psd_triangle",
                indices=[0, 1, 2],
                psd_k=2,
            )
        ],
    )
    A_dx = sparse.csr_matrix(np.zeros((0, 3)))
    b_dx = np.array([])
    settings = moreau.Settings(device="cpu", verbose=False, solver="ipm")
    solver_direct = moreau.Solver(P, q, A_dx, b_dx, cones=cones_direct, settings=settings)
    sol_direct = solver_direct.solve()

    # Slack reference: `-x + s = 0, s ∈ PSD_svec(2)`.
    A_slack = sparse.csr_matrix(-np.eye(3))
    b_slack = np.zeros(3)
    cones_slack = moreau.Cones(psd_dims=[2])
    solver_slack = moreau.Solver(P, q, A_slack, b_slack, cones=cones_slack, settings=settings)
    sol_slack = solver_slack.solve()

    np.testing.assert_allclose(sol_direct.x, sol_slack.x, atol=1e-6)

    # PSD sanity: det(X) >= 0.
    a, b_scaled, c = sol_direct.x
    b = b_scaled / np.sqrt(2.0)
    assert a + 1e-8 >= 0.0
    assert c + 1e-8 >= 0.0
    assert a * c - b * b + 1e-8 >= 0.0


def test_solver_with_dir_cones_cuda_psd_supported():
    # Direct-x PSD on CUDA is wired. Solve must succeed and
    # produce the same result as CPU.
    from moreau._backend import device_available

    if not device_available("cuda"):
        pytest.skip("CUDA backend not available")

    P = sparse.diags([1.0, 1.0, 1.0], format="csr")
    # q drives optimum to svec(2I) which is strictly interior PSD.
    q = np.array([-2.0, 0.0, -2.0])
    A = sparse.csr_matrix(np.zeros((0, 3)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[
            moreau.DirectConeSpec(
                kind="psd_triangle",
                indices=[0, 1, 2],
                psd_k=2,
            )
        ],
    )
    settings_cuda = moreau.Settings(device="cuda", verbose=False)
    sol_cuda = moreau.Solver(P, q, A, b, cones=cones, settings=settings_cuda).solve()

    settings_cpu = moreau.Settings(device="cpu", verbose=False)
    sol_cpu = moreau.Solver(P, q, A, b, cones=cones, settings=settings_cpu).solve()

    np.testing.assert_allclose(sol_cuda.x, sol_cpu.x, atol=1e-3)


def test_solver_with_dir_cones_cuda_psd_enable_grad_runs():
    # CUDA direct-x PSD backward is wired natively. Smoke-test that
    # enable_grad=True solves without raising. Numerical parity vs the
    # CPU IFT-direct path is covered by C++ gtests.
    from moreau._backend import device_available

    if not device_available("cuda"):
        pytest.skip("CUDA backend not available")

    P = sparse.diags([1.0, 1.0, 1.0], format="csr")
    q = np.array([0.0, -1.0, 0.0])
    A = sparse.csr_matrix(np.zeros((0, 3)))
    b = np.array([])
    cones = moreau.Cones(
        dir_cones=[
            moreau.DirectConeSpec(
                kind="psd_triangle",
                indices=[0, 1, 2],
                psd_k=2,
            )
        ],
    )
    settings = moreau.Settings(device="cuda", verbose=False, enable_grad=True)
    solver = moreau.Solver(P, q, A, b, cones=cones, settings=settings)
    solver.solve()
    assert solver.info.status.name in ("Solved", "AlmostSolved")
