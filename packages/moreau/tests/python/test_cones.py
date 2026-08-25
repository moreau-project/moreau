"""
Unit tests for Moreau cone support with CVXPY comparison.

Tests SOC, exponential, and power cones against CVXPY/Clarabel reference.
"""

import numpy as np
import pytest
from scipy import sparse

import moreau

try:
    import cvxpy as cp

    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False
    cp = None

pytestmark = pytest.mark.skipif(not HAS_CVXPY, reason="cvxpy not installed")


def cvxpy_to_moreau_data(prob: "cp.Problem"):
    """
    Extract problem data from CVXPY in Moreau-compatible format.

    Returns dict with CSR matrices, vectors, and cone specification.
    """
    # Get problem data in Clarabel format (closest to Moreau)
    data, chain, inverse_data = prob.get_problem_data(solver=cp.CLARABEL)

    # Extract matrices - CVXPY uses 'c' for linear cost (not 'q')
    # P may not be present for LPs (no quadratic term)
    A = data["A"]  # scipy sparse
    c = data["c"]  # numpy array (linear cost, called 'q' in Moreau)
    b = data["b"]  # numpy array

    n = len(c)  # Number of variables

    # Handle P matrix - may be None or missing for LPs
    if "P" in data and data["P"] is not None:
        P = data["P"]
        P_csr = sparse.csr_array(P) if not sparse.isspmatrix_csr(P) else P
    else:
        # Create empty P matrix for LP
        P_csr = sparse.csr_array((n, n), dtype=np.float64)

    # Convert A to CSR format
    A_csr = sparse.csr_array(A) if not sparse.isspmatrix_csr(A) else A

    # Extract CSR components
    P_row_offsets = P_csr.indptr.astype(np.int64)
    P_col_indices = P_csr.indices.astype(np.int64)
    P_values = P_csr.data.astype(np.float64)

    A_row_offsets = A_csr.indptr.astype(np.int64)
    A_col_indices = A_csr.indices.astype(np.int64)
    A_values = A_csr.data.astype(np.float64)

    # Extract cone info from CVXPY ConeDims
    dims = data["dims"]

    # Build Moreau cones object
    cones = moreau.Cones()
    cones.num_zero_cones = dims.zero
    cones.num_nonneg_cones = dims.nonneg
    cones.num_exp_cones = dims.exp

    # SOC cones - CVXPY gives a list of sizes (arbitrary dim >= 2)
    if dims.soc:
        cones.so_cone_dims = list(dims.soc)

    # Power cones - CVXPY gives a list of alpha values
    if dims.p3d:
        cones.power_alphas = list(dims.p3d)

    # n already computed above
    m = A_csr.shape[0]

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "q": c.astype(np.float64),  # Moreau calls it 'q'
        "b": b.astype(np.float64),
        "cones": cones,
        "inverse_data": inverse_data,
        "chain": chain,
    }


def solve_with_moreau(data, verbose=False, max_iter=200):
    """Solve problem with Moreau solver."""
    from scipy import sparse

    settings = moreau.Settings(max_iter=max_iter, verbose=verbose)

    # Convert to sparse matrices
    P = sparse.csr_array(
        (data["P_values"], data["P_col_indices"], data["P_row_offsets"]),
        shape=(data["n"], data["n"]),
    )
    A = sparse.csr_array(
        (data["A_values"], data["A_col_indices"], data["A_row_offsets"]),
        shape=(data["m"], data["n"]),
    )

    # Create bounds (unbounded)

    # Use new API: Solver(P, q, A, b, cones)
    solver = moreau.Solver(P, data["q"], A, data["b"], data["cones"], settings)
    result = solver.solve()
    info = solver.info

    return result, info


class TestSecondOrderCone:
    """Tests for second-order cone (SOC) constraints."""

    def test_soc_basic(self):
        """Test basic 3D SOC constraint: ||x[1:]|| <= x[0]."""
        # min x[0] s.t. ||x[1:]|| <= x[0], x[1] = 1
        x = cp.Variable(3)
        constraints = [
            cp.SOC(x[0], x[1:]),  # ||x[1:]|| <= x[0]
            x[1] == 1,
            x[2] == 0,
        ]
        prob = cp.Problem(cp.Minimize(x[0]), constraints)
        prob.solve(solver=cp.CLARABEL)
        cvxpy_x = x.value
        cvxpy_obj = prob.value

        # Solve with Moreau
        data = cvxpy_to_moreau_data(prob)
        result, info = solve_with_moreau(data, verbose=False)

        # Check convergence
        assert info.status in [
            moreau.SolverStatus.Solved.value,
            moreau.SolverStatus.AlmostSolved.value,
        ], f"Expected Solved, got {info.status}"

        # Compare solutions (moreau x is in transformed space, need to compare objectives)
        moreau_obj = info.obj_val
        np.testing.assert_allclose(moreau_obj, cvxpy_obj, rtol=1e-4, atol=1e-6)

    def test_soc_norm_minimization(self):
        """Test norm minimization: min ||x - target||."""
        n = 3
        target = np.array([1.0, 2.0, 3.0])

        # min ||x - target||^2 s.t. sum(x) = 3
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x - target)), [cp.sum(x) == 3])
        prob.solve(solver=cp.CLARABEL)
        cvxpy_obj = prob.value

        # Solve with Moreau
        data = cvxpy_to_moreau_data(prob)
        result, info = solve_with_moreau(data, verbose=False)

        assert info.status in [
            moreau.SolverStatus.Solved.value,
            moreau.SolverStatus.AlmostSolved.value,
        ]

        np.testing.assert_allclose(info.obj_val, cvxpy_obj, rtol=1e-4, atol=1e-6)

    def test_soc_with_linear(self):
        """Test SOC combined with linear inequalities."""
        x = cp.Variable(3)
        constraints = [
            cp.SOC(x[0], x[1:]),  # ||x[1:]|| <= x[0]
            x >= 0,  # Linear inequalities
            x[0] >= 1,  # x[0] >= 1
        ]
        prob = cp.Problem(cp.Minimize(x[0] + x[1] + x[2]), constraints)
        prob.solve(solver=cp.CLARABEL)
        cvxpy_obj = prob.value

        data = cvxpy_to_moreau_data(prob)
        result, info = solve_with_moreau(data, verbose=False)

        assert info.status in [
            moreau.SolverStatus.Solved.value,
            moreau.SolverStatus.AlmostSolved.value,
        ]

        np.testing.assert_allclose(info.obj_val, cvxpy_obj, rtol=1e-4, atol=1e-6)


class TestExponentialCone:
    """Tests for exponential cone constraints."""

    def test_exp_log(self):
        """Test problem with log constraint."""
        # min x s.t. log(x) >= 1, x > 0
        # Solution: x = e (the smallest x satisfying log(x) >= 1)
        x = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x), [cp.log(x) >= 1])
        prob.solve(solver=cp.CLARABEL)
        cvxpy_obj = prob.value

        data = cvxpy_to_moreau_data(prob)
        result, info = solve_with_moreau(data, verbose=False)

        assert info.status in [
            moreau.SolverStatus.Solved.value,
            moreau.SolverStatus.AlmostSolved.value,
        ], f"Expected Solved, got status {info.status}"

        # Objective should be approximately e
        np.testing.assert_allclose(info.obj_val, cvxpy_obj, rtol=1e-3, atol=1e-5)

    def test_exp_entropy(self):
        """Test entropy maximization problem."""
        n = 3
        # max -sum(x * log(x)) s.t. sum(x) = 1, x >= 0
        # This is equivalent to min sum(x * log(x))
        x = cp.Variable(n, pos=True)
        prob = cp.Problem(cp.Maximize(cp.sum(cp.entr(x))), [cp.sum(x) == 1])
        prob.solve(solver=cp.CLARABEL)
        cvxpy_obj = prob.value

        data = cvxpy_to_moreau_data(prob)
        result, info = solve_with_moreau(data, verbose=False)

        assert info.status in [
            moreau.SolverStatus.Solved.value,
            moreau.SolverStatus.AlmostSolved.value,
        ]

        # Entropy of uniform distribution is log(n)
        np.testing.assert_allclose(-info.obj_val, cvxpy_obj, rtol=1e-3, atol=1e-5)

    def test_exp_with_linear(self):
        """Test exponential cone with linear constraints."""
        x = cp.Variable(2, pos=True)
        prob = cp.Problem(
            cp.Minimize(-cp.log(x[0]) - cp.log(x[1])), [x[0] + x[1] <= 2, x[0] >= 0.5]
        )
        prob.solve(solver=cp.CLARABEL)
        cvxpy_obj = prob.value

        data = cvxpy_to_moreau_data(prob)
        result, info = solve_with_moreau(data, verbose=False)

        assert info.status in [
            moreau.SolverStatus.Solved.value,
            moreau.SolverStatus.AlmostSolved.value,
        ]

        np.testing.assert_allclose(info.obj_val, cvxpy_obj, rtol=1e-3, atol=1e-5)


class TestPowerCone:
    """Tests for power cone constraints."""

    def test_power_geometric_mean(self):
        """Test geometric mean constraint."""
        # max sqrt(x * y) s.t. x + y <= 2, x, y >= 0
        # Solution: x = y = 1, geo_mean = 1
        x = cp.Variable(2, nonneg=True)
        prob = cp.Problem(cp.Maximize(cp.geo_mean(x)), [cp.sum(x) <= 2])
        prob.solve(solver=cp.CLARABEL)
        cvxpy_obj = prob.value

        data = cvxpy_to_moreau_data(prob)
        result, info = solve_with_moreau(data, verbose=False)

        assert info.status in [
            moreau.SolverStatus.Solved.value,
            moreau.SolverStatus.AlmostSolved.value,
        ]

        np.testing.assert_allclose(-info.obj_val, cvxpy_obj, rtol=1e-3, atol=1e-5)

    def test_power_p_norm(self):
        """Test p-norm constraint (uses power cones internally)."""
        n = 3
        x = cp.Variable(n)
        target = np.array([1.0, 2.0, 1.0])

        # min ||x - target||_1.5 s.t. sum(x) = 3
        prob = cp.Problem(cp.Minimize(cp.pnorm(x - target, 1.5)), [cp.sum(x) == 3])
        prob.solve(solver=cp.CLARABEL)
        cvxpy_obj = prob.value

        data = cvxpy_to_moreau_data(prob)
        result, info = solve_with_moreau(data, verbose=False, max_iter=500)

        assert info.status in [
            moreau.SolverStatus.Solved.value,
            moreau.SolverStatus.AlmostSolved.value,
        ]

        np.testing.assert_allclose(info.obj_val, cvxpy_obj, rtol=1e-2, atol=1e-4)


class TestMixedCones:
    """Tests for problems with multiple cone types."""

    def test_mixed_zero_nonneg_soc(self):
        """Test problem with equality, inequality, and SOC constraints."""
        x = cp.Variable(3)
        constraints = [
            x[0] + x[1] == 2,  # Equality (zero cone)
            x >= 0,  # Nonnegative cone
            cp.SOC(x[0], x[1:]),  # SOC
        ]
        prob = cp.Problem(cp.Minimize(x[0] - x[1] + x[2]), constraints)
        prob.solve(solver=cp.CLARABEL)
        cvxpy_obj = prob.value

        data = cvxpy_to_moreau_data(prob)
        result, info = solve_with_moreau(data, verbose=False)

        assert info.status in [
            moreau.SolverStatus.Solved.value,
            moreau.SolverStatus.AlmostSolved.value,
        ]

        np.testing.assert_allclose(info.obj_val, cvxpy_obj, rtol=1e-4, atol=1e-6)

    def test_qp_with_soc(self):
        """Test QP with SOC constraint (portfolio-like)."""
        n = 3
        # Minimize risk (quadratic) subject to return constraint and norm bound
        returns = np.array([0.1, 0.2, 0.15])

        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(x)),
            [
                returns @ x >= 0.12,  # Return constraint
                cp.sum(x) == 1,  # Budget constraint
                x >= 0,  # No short selling
            ],
        )
        prob.solve(solver=cp.CLARABEL)
        cvxpy_obj = prob.value

        data = cvxpy_to_moreau_data(prob)
        result, info = solve_with_moreau(data, verbose=False)

        assert info.status in [
            moreau.SolverStatus.Solved.value,
            moreau.SolverStatus.AlmostSolved.value,
        ]

        np.testing.assert_allclose(info.obj_val, cvxpy_obj, rtol=1e-4, atol=1e-6)


# PyTorch variants
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


def solve_with_moreau_torch(data, batch_size=1, verbose=False, max_iter=200):
    """Solve problem with Moreau PyTorch solver."""
    import torch

    # Use the default device from moreau settings
    device = moreau.default_device()

    settings = moreau.Settings()
    settings.max_iter = max_iter
    settings.verbose = verbose

    # Build cones for torch (uses same Cones class)
    # Note: num_power_cones is computed from power_alphas, not set directly
    cones = moreau.Cones(
        num_zero_cones=data["cones"].num_zero_cones,
        num_nonneg_cones=data["cones"].num_nonneg_cones,
        so_cone_dims=list(data["cones"].so_cone_dims),
        num_exp_cones=data["cones"].num_exp_cones,
        power_alphas=list(data["cones"].power_alphas) if data["cones"].num_power_cones > 0 else [],
    )

    # TorchSolver uses CompiledSolver API with pattern
    solver = TorchSolver(
        n=data["n"],
        m=data["m"],
        P_row_offsets=torch.tensor(data["P_row_offsets"], dtype=torch.int64),
        P_col_indices=torch.tensor(data["P_col_indices"], dtype=torch.int64),
        A_row_offsets=torch.tensor(data["A_row_offsets"], dtype=torch.int64),
        A_col_indices=torch.tensor(data["A_col_indices"], dtype=torch.int64),
        cones=cones,
        settings=settings,
    )

    # Convert to tensors with batch dimension, respecting default device
    P_values = torch.tensor(data["P_values"], dtype=torch.float64, device=device).unsqueeze(0)
    A_values = torch.tensor(data["A_values"], dtype=torch.float64, device=device).unsqueeze(0)
    q = torch.tensor(data["q"], dtype=torch.float64, device=device).unsqueeze(0)
    b = torch.tensor(data["b"], dtype=torch.float64, device=device).unsqueeze(0)

    result = solver.solve(P_values, A_values, q, b)
    info = solver.info

    # Result is a TorchSolution or TorchBatchedSolution object
    x = result.x
    z = result.z
    s = result.s
    status = info.status
    obj_val = info.obj_val

    # Convert to numpy
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(z, "cpu"):
        z = z.cpu()
    if hasattr(s, "cpu"):
        s = s.cpu()
    if hasattr(status, "value"):
        status = status.value
    elif isinstance(status, list):
        status = status[0].value if hasattr(status[0], "value") else status[0]

    return {
        "x": x.numpy().squeeze() if hasattr(x, "numpy") else x,
        "z": z.numpy().squeeze() if hasattr(z, "numpy") else z,
        "s": s.numpy().squeeze() if hasattr(s, "numpy") else s,
        "status": status,
        "obj_val": obj_val,
    }


requires_torch = pytest.mark.skipif(
    not (HAS_TORCH and HAS_MOREAU_TORCH), reason="Requires torch and moreau.torch"
)

requires_torch_cuda = pytest.mark.skipif(
    not (HAS_TORCH and HAS_CUDA and HAS_MOREAU_TORCH),
    reason="Requires torch, CUDA, and moreau_torch",
)


class TestSecondOrderConeTorch:
    """PyTorch variants of SOC tests."""

    @requires_torch
    def test_soc_basic_torch(self):
        """Test basic SOC with PyTorch tensors."""
        import torch

        x = cp.Variable(3)
        constraints = [
            cp.SOC(x[0], x[1:]),
            x[1] == 1,
            x[2] == 0,
        ]
        prob = cp.Problem(cp.Minimize(x[0]), constraints)
        prob.solve(solver=cp.CLARABEL)
        cvxpy_obj = prob.value

        data = cvxpy_to_moreau_data(prob)
        result = solve_with_moreau_torch(data, verbose=False)

        assert result["status"] in [
            moreau.SolverStatus.Solved.value,
            moreau.SolverStatus.AlmostSolved.value,
        ]

        # Convert obj_val to CPU if it's a tensor
        obj_val = result["obj_val"]
        if isinstance(obj_val, torch.Tensor):
            obj_val = obj_val.cpu().numpy()
        np.testing.assert_allclose(obj_val, cvxpy_obj, rtol=1e-4, atol=1e-6)

    @requires_torch_cuda
    def test_soc_output_on_gpu(self):
        """Verify SOC solve outputs stay on GPU."""
        import torch

        x = cp.Variable(3)
        constraints = [
            cp.SOC(x[0], x[1:]),
            x[1] == 1,
            x[2] == 0,
        ]
        prob = cp.Problem(cp.Minimize(x[0]), constraints)
        prob.solve(solver=cp.CLARABEL)

        data = cvxpy_to_moreau_data(prob)

        # Build cones
        cones = moreau.Cones()
        cones.num_zero_cones = data["cones"].num_zero_cones
        cones.num_nonneg_cones = data["cones"].num_nonneg_cones
        cones.so_cone_dims = list(data["cones"].so_cone_dims)

        settings = moreau.Settings(device="cuda")
        settings.verbose = False

        # Note: batch_size is inferred from inputs (lazy initialization)
        solver = TorchSolver(
            n=data["n"],
            m=data["m"],
            P_row_offsets=torch.tensor(data["P_row_offsets"], dtype=torch.int64),
            P_col_indices=torch.tensor(data["P_col_indices"], dtype=torch.int64),
            A_row_offsets=torch.tensor(data["A_row_offsets"], dtype=torch.int64),
            A_col_indices=torch.tensor(data["A_col_indices"], dtype=torch.int64),
            cones=cones,
            settings=settings,
        )

        P_values = torch.tensor(data["P_values"], dtype=torch.float64, device="cuda").unsqueeze(0)
        A_values = torch.tensor(data["A_values"], dtype=torch.float64, device="cuda").unsqueeze(0)
        q = torch.tensor(data["q"], dtype=torch.float64, device="cuda").unsqueeze(0)
        b = torch.tensor(data["b"], dtype=torch.float64, device="cuda").unsqueeze(0)

        result = solver.solve(P_values, A_values, q, b)
        info = solver.info

        # Verify outputs are on CUDA
        assert result.x.device.type == "cuda", "x should be on CUDA"
        assert result.z.device.type == "cuda", "z should be on CUDA"
        assert result.s.device.type == "cuda", "s should be on CUDA"


class TestExponentialConeTorch:
    """PyTorch variants of exponential cone tests."""

    @requires_torch
    def test_exp_log_torch(self):
        """Test log constraint with PyTorch tensors."""
        import torch

        x = cp.Variable(pos=True)
        prob = cp.Problem(cp.Minimize(x), [cp.log(x) >= 1])  # min x (not -x to avoid unbounded)
        prob.solve(solver=cp.CLARABEL)
        cvxpy_obj = prob.value

        data = cvxpy_to_moreau_data(prob)
        result = solve_with_moreau_torch(data, verbose=False)

        assert result["status"] in [
            moreau.SolverStatus.Solved.value,
            moreau.SolverStatus.AlmostSolved.value,
        ], f"Expected Solved, got status {result['status']}"

        # Convert obj_val to CPU if it's a tensor
        obj_val = result["obj_val"]
        if isinstance(obj_val, torch.Tensor):
            obj_val = obj_val.cpu().numpy()
        np.testing.assert_allclose(obj_val, cvxpy_obj, rtol=1e-3, atol=1e-5)


class TestMixedConesTorchBatched:
    """Test batched solving with mixed cones."""

    @requires_torch
    def test_batched_qp_with_soc(self):
        """Test batched QP with SOC constraint."""
        import torch

        # Use default device
        device = moreau.default_device()

        n = 3
        returns = np.array([0.1, 0.2, 0.15])

        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Minimize(cp.sum_squares(x)),
            [
                returns @ x >= 0.12,
                cp.sum(x) == 1,
                x >= 0,
            ],
        )
        prob.solve(solver=cp.CLARABEL)
        cvxpy_obj = prob.value

        data = cvxpy_to_moreau_data(prob)

        # Build cones
        cones = moreau.Cones()
        cones.num_zero_cones = data["cones"].num_zero_cones
        cones.num_nonneg_cones = data["cones"].num_nonneg_cones

        settings = moreau.Settings()
        settings.verbose = False

        batch_size = 3

        # Note: batch_size is inferred from inputs (lazy initialization)
        solver = TorchSolver(
            n=data["n"],
            m=data["m"],
            P_row_offsets=torch.tensor(data["P_row_offsets"], dtype=torch.int64),
            P_col_indices=torch.tensor(data["P_col_indices"], dtype=torch.int64),
            A_row_offsets=torch.tensor(data["A_row_offsets"], dtype=torch.int64),
            A_col_indices=torch.tensor(data["A_col_indices"], dtype=torch.int64),
            cones=cones,
            settings=settings,
        )

        # Create batched inputs using default device
        P_values = torch.tensor(data["P_values"], dtype=torch.float64, device=device)
        A_values = torch.tensor(data["A_values"], dtype=torch.float64, device=device)
        q = torch.tensor(data["q"], dtype=torch.float64, device=device)
        b = torch.tensor(data["b"], dtype=torch.float64, device=device)

        P_batched = P_values.unsqueeze(0).expand(batch_size, -1).contiguous()
        A_batched = A_values.unsqueeze(0).expand(batch_size, -1).contiguous()
        q_batched = q.unsqueeze(0).expand(batch_size, -1).contiguous()
        b_batched = b.unsqueeze(0).expand(batch_size, -1).contiguous()

        result = solver.solve(P_batched, A_batched, q_batched, b_batched)
        info = solver.info

        # Result is a TorchBatchedSolution object
        x_out = result.x
        z = result.z

        # Check shapes
        assert x_out.shape == (batch_size, data["n"])
        assert z.shape == (batch_size, data["m"])

        # All batch elements should give same solution
        x_np = x_out.cpu().numpy() if hasattr(x_out, "cpu") else x_out.numpy()
        for i in range(batch_size):
            np.testing.assert_allclose(x_np[i], x_np[0], rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
