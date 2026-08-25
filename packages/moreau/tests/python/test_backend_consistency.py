"""
Tests for CPU/CUDA backend consistency.

Verifies that CPU and CUDA backends produce identical results for the same problems.
Also compares against Clarabel reference solver when available.
"""

import numpy as np
import pytest

import moreau

try:
    import moreau.torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import clarabel

    HAS_CLARABEL = True
except ImportError:
    HAS_CLARABEL = False


@pytest.fixture
def simple_qp():
    """Simple QP problem for consistency testing."""
    n, m = 2, 3

    # P matrix (2x2 identity) in CSR format
    P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    P_col_indices = np.array([0, 1], dtype=np.int64)
    P_values = np.array([1.0, 1.0])

    # A matrix in CSR format
    A_row_offsets = np.array([0, 2, 3, 4], dtype=np.int64)
    A_col_indices = np.array([0, 1, 0, 1], dtype=np.int64)
    A_values = np.array([1.0, 1.0, 1.0, 1.0])

    q = np.array([2.0, 1.0])
    b = np.array([1.0, 2.0, 2.0])

    cones = moreau.Cones()
    cones.num_zero_cones = 1
    cones.num_nonneg_cones = 2

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "q": q,
        "b": b,
        "cones": cones,
    }


@pytest.fixture
def soc_problem():
    """Second-order cone problem for consistency testing."""
    # min 0.5 * x'Px + q'x
    # s.t. Ax + s = b, s in SOC(3)
    n, m = 3, 3

    # P = I
    P_row_offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    P_col_indices = np.array([0, 1, 2], dtype=np.int64)
    P_values = np.array([1.0, 1.0, 1.0])

    # A = I
    A_row_offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    A_col_indices = np.array([0, 1, 2], dtype=np.int64)
    A_values = np.array([1.0, 1.0, 1.0])

    q = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.5, 0.5])

    cones = moreau.Cones()
    cones.so_cone_dims = [3]  # One 3-dimensional SOC

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "q": q,
        "b": b,
        "cones": cones,
    }


@pytest.fixture
def exp_cone_problem():
    """Exponential cone problem for consistency testing.

    Exponential cone: (x, y, z) such that y * exp(x/y) <= z, y > 0
    """
    n, m = 3, 3

    # P = I
    P_row_offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    P_col_indices = np.array([0, 1, 2], dtype=np.int64)
    P_values = np.array([1.0, 1.0, 1.0])

    # A = I
    A_row_offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    A_col_indices = np.array([0, 1, 2], dtype=np.int64)
    A_values = np.array([1.0, 1.0, 1.0])

    q = np.array([0.0, 0.0, -1.0])  # Minimize -z (maximize z)
    b = np.array([1.0, 2.0, 0.0])  # x=1, y=2, find z

    cones = moreau.Cones()
    cones.num_exp_cones = 1

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "q": q,
        "b": b,
        "cones": cones,
    }


@pytest.fixture
def power_cone_problem():
    """Power cone problem for consistency testing.

    Power cone with alpha=0.5: (x, y, z) such that x^alpha * y^(1-alpha) >= |z|, x,y >= 0
    """
    n, m = 3, 3

    # P = I
    P_row_offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    P_col_indices = np.array([0, 1, 2], dtype=np.int64)
    P_values = np.array([1.0, 1.0, 1.0])

    # A = I
    A_row_offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    A_col_indices = np.array([0, 1, 2], dtype=np.int64)
    A_values = np.array([1.0, 1.0, 1.0])

    q = np.array([0.0, 0.0, -1.0])  # Minimize -z
    b = np.array([1.0, 1.0, 0.0])  # x=1, y=1

    cones = moreau.Cones()
    cones.power_alphas = [0.5]

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "q": q,
        "b": b,
        "cones": cones,
    }


@pytest.fixture
def mixed_cone_problem():
    """Problem with zero + nonneg + SOC cones.

    Minimizes 0.5*x'Px + q'x subject to:
    - x0 = 2              (zero cone - equality)
    - x1 <= 1, x2 <= 1    (nonneg cone - inequalities)
    - ||(x1, x2)|| <= x0  (SOC - second order cone)
    """
    # 1 zero (equality), 2 nonneg, 1 SOC(3) = 6 constraints total
    n, m = 3, 6

    # P = I (3x3)
    P_row_offsets = np.array([0, 1, 2, 3], dtype=np.int64)
    P_col_indices = np.array([0, 1, 2], dtype=np.int64)
    P_values = np.array([1.0, 1.0, 1.0])

    # A: 6x3 matrix
    # Row 0: x0 = 2 (equality, zero cone)
    # Rows 1-2: x1 + s[1] = 1, x2 + s[2] = 1 (s >= 0 means x <= 1)
    # Rows 3-5: SOC via -I so s = (x0, x1, x2), then s in SOC means x0 >= ||(x1,x2)||
    A_row_offsets = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    A_col_indices = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    A_values = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])

    q = np.array([1.0, 1.0, 1.0])
    b = np.array([2.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    cones = moreau.Cones()
    cones.num_zero_cones = 1
    cones.num_nonneg_cones = 2
    cones.so_cone_dims = [3]

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "q": q,
        "b": b,
        "cones": cones,
    }


@pytest.fixture
def pure_lp():
    """Pure linear program (no quadratic term)."""
    # min q'x s.t. Ax <= b
    n, m = 3, 4

    # P = 0 (empty/zero)
    P_row_offsets = np.array([0, 0, 0, 0], dtype=np.int64)
    P_col_indices = np.array([], dtype=np.int64)
    P_values = np.array([])

    # A: 4x3 inequality constraints
    A_row_offsets = np.array([0, 3, 6, 9, 12], dtype=np.int64)
    A_col_indices = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)
    A_values = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, -1.0])

    q = np.array([-1.0, -2.0, -1.0])  # Maximize x + 2y + z
    b = np.array([1.0, 1.0, 1.0, -0.5])  # x,y,z <= 1, x+y+z >= 0.5

    cones = moreau.Cones()
    cones.num_nonneg_cones = 4

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "q": q,
        "b": b,
        "cones": cones,
    }


@pytest.fixture
def multiple_soc_problem():
    """Problem with multiple second-order cones."""
    # Two SOC(3) constraints = 6 constraints
    n, m = 4, 6

    # P = I
    P_row_offsets = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    P_col_indices = np.array([0, 1, 2, 3], dtype=np.int64)
    P_values = np.array([1.0, 1.0, 1.0, 1.0])

    # A: maps variables to two SOC constraints
    # SOC1: (x0, x1, x2) in SOC(3)
    # SOC2: (x0, x2, x3) in SOC(3)
    A_row_offsets = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    A_col_indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.int64)
    A_values = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    q = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([1.0, 0.3, 0.3, 1.0, 0.3, 0.3])

    cones = moreau.Cones()
    cones.so_cone_dims = [3, 3]

    return {
        "n": n,
        "m": m,
        "P_row_offsets": P_row_offsets,
        "P_col_indices": P_col_indices,
        "P_values": P_values,
        "A_row_offsets": A_row_offsets,
        "A_col_indices": A_col_indices,
        "A_values": A_values,
        "q": q,
        "b": b,
        "cones": cones,
    }


def _solve_and_compare(p, rtol=1e-6, atol=1e-6):
    """Helper to solve with both backends and compare results."""
    from scipy import sparse

    # Convert to sparse matrices
    P = sparse.csr_array(
        (p["P_values"], p["P_col_indices"], p["P_row_offsets"]), shape=(p["n"], p["n"])
    )
    A = sparse.csr_array(
        (p["A_values"], p["A_col_indices"], p["A_row_offsets"]), shape=(p["m"], p["n"])
    )

    # Create bounds (unbounded)

    cpu_settings = moreau.Settings(device="cpu")
    cpu_solver = moreau.Solver(P, p["q"], A, p["b"], p["cones"], cpu_settings)

    gpu_settings = moreau.Settings(device="cuda")
    gpu_solver = moreau.Solver(P, p["q"], A, p["b"], p["cones"], gpu_settings)

    cpu_result = cpu_solver.solve()
    cpu_info = cpu_solver.info
    gpu_result = gpu_solver.solve()
    gpu_info = gpu_solver.info

    np.testing.assert_allclose(
        cpu_result.x, gpu_result.x, rtol=rtol, atol=atol, err_msg="x mismatch between CPU and GPU"
    )
    np.testing.assert_allclose(
        cpu_result.s, gpu_result.s, rtol=rtol, atol=atol, err_msg="s mismatch between CPU and GPU"
    )
    np.testing.assert_allclose(
        cpu_result.z, gpu_result.z, rtol=rtol, atol=atol, err_msg="z mismatch between CPU and GPU"
    )
    np.testing.assert_allclose(
        cpu_info.obj_val,
        gpu_info.obj_val,
        rtol=rtol,
        err_msg="obj_val mismatch between CPU and GPU",
    )

    return cpu_result, gpu_result


class TestCpuGpuConsistency:
    """Test that CPU and GPU backends produce matching results."""

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_simple_qp_consistency(self, simple_qp):
        """Test CPU and GPU produce same results for simple QP."""
        _solve_and_compare(simple_qp)

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_soc_consistency(self, soc_problem):
        """Test CPU and CUDA produce same results for SOC problem."""
        _solve_and_compare(soc_problem)

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_exp_cone_consistency(self, exp_cone_problem):
        """Test CPU and CUDA produce same results for exponential cone."""
        _solve_and_compare(exp_cone_problem)

    def test_power_cone_cpu_only(self, power_cone_problem):
        """Test power cone works on CPU (CUDA doesn't support power cones)."""
        from scipy import sparse

        p = power_cone_problem

        # Convert to sparse matrices
        P = sparse.csr_array(
            (p["P_values"], p["P_col_indices"], p["P_row_offsets"]), shape=(p["n"], p["n"])
        )
        A = sparse.csr_array(
            (p["A_values"], p["A_col_indices"], p["A_row_offsets"]), shape=(p["m"], p["n"])
        )

        # Create bounds (unbounded)

        settings = moreau.Settings(device="cpu")
        cpu_solver = moreau.Solver(P, p["q"], A, p["b"], p["cones"], settings)
        result = cpu_solver.solve()
        info = cpu_solver.info
        assert hasattr(result, "x")
        # For power cone with alpha=0.5, x=y=1 => z can be up to 1
        assert info.obj_val < 0  # Minimizing -z, so should be negative

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_mixed_cone_consistency(self, mixed_cone_problem):
        """Test CPU and CUDA produce same results for mixed cones."""
        _solve_and_compare(mixed_cone_problem)

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_pure_lp_consistency(self, pure_lp):
        """Test CPU and CUDA produce same results for pure LP."""
        _solve_and_compare(pure_lp)

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_multiple_soc_consistency(self, multiple_soc_problem):
        """Test CPU and CUDA produce same results for multiple SOCs."""
        _solve_and_compare(multiple_soc_problem)

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_batched_consistency(self, simple_qp):
        """Test CPU and CUDA produce same results for batched problems."""
        p = simple_qp
        batch_size = 4

        # Create batched inputs with slight variations
        np.random.seed(42)
        P_batch = np.tile(p["P_values"], (batch_size, 1))
        A_batch = np.tile(p["A_values"], (batch_size, 1))
        q_batch = np.tile(p["q"], (batch_size, 1))
        q_batch += np.random.randn(batch_size, p["n"]) * 0.1  # Small perturbations
        b_batch = np.tile(p["b"], (batch_size, 1))

        # Create bounds (unbounded)

        cpu_settings = moreau.Settings(device="cpu", batch_size=batch_size)
        cpu_solver = moreau.CompiledSolver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=cpu_settings,
        )

        gpu_settings = moreau.Settings(device="cuda", batch_size=batch_size)
        gpu_solver = moreau.CompiledSolver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=gpu_settings,
        )

        cpu_solver.setup(P_batch, A_batch)
        cpu_result = cpu_solver.solve(q_batch, b_batch)
        cpu_info = cpu_solver.info
        gpu_solver.setup(P_batch, A_batch)
        gpu_result = gpu_solver.solve(q_batch, b_batch)
        gpu_info = gpu_solver.info

        np.testing.assert_allclose(
            cpu_result.x,
            gpu_result.x,
            rtol=1e-5,
            atol=1e-5,
            err_msg="Batched x mismatch between CPU and GPU",
        )
        np.testing.assert_allclose(
            cpu_info.obj_val,
            gpu_info.obj_val,
            rtol=1e-5,
            err_msg="Batched obj_val mismatch between CPU and GPU",
        )


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestDerivativeConsistency:
    """Test that CPU and GPU derivatives match."""

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_backward_gradient_consistency(self, simple_qp):
        """Test CPU and CUDA backward passes produce same gradients."""
        torch = pytest.importorskip("torch")

        p = simple_qp

        # GPU solver with autograd support
        gpu_settings = moreau.Settings(device="cuda", batch_size=1)
        gpu_solver = moreau.torch.Solver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=torch.tensor(p["P_row_offsets"]),
            P_col_indices=torch.tensor(p["P_col_indices"]),
            A_row_offsets=torch.tensor(p["A_row_offsets"]),
            A_col_indices=torch.tensor(p["A_col_indices"]),
            cones=p["cones"],
            settings=gpu_settings,
        )

        # Prepare CUDA tensors with requires_grad (batched)
        # Create with correct shape first, then enable grad (to ensure leaf tensor)
        P_gpu = (
            torch.tensor(p["P_values"], dtype=torch.float64, device="cuda")
            .unsqueeze(0)
            .clone()
            .requires_grad_(True)
        )
        A_gpu = (
            torch.tensor(p["A_values"], dtype=torch.float64, device="cuda")
            .unsqueeze(0)
            .clone()
            .requires_grad_(True)
        )
        q_gpu = (
            torch.tensor(p["q"], dtype=torch.float64, device="cuda")
            .unsqueeze(0)
            .clone()
            .requires_grad_(True)
        )
        b_gpu = (
            torch.tensor(p["b"], dtype=torch.float64, device="cuda")
            .unsqueeze(0)
            .clone()
            .requires_grad_(True)
        )

        # GPU forward and backward
        solution_gpu = gpu_solver.solve(P_gpu, A_gpu, q_gpu, b_gpu)
        x_gpu, z_gpu, s_gpu = solution_gpu.x, solution_gpu.z, solution_gpu.s
        x_gpu.sum().backward()
        gpu_dq = q_gpu.grad.cpu().numpy().squeeze()
        gpu_db = b_gpu.grad.cpu().numpy().squeeze()

        # CPU solver - same problem with CPU device
        cpu_settings = moreau.Settings(device="cpu", batch_size=1)
        cpu_solver = moreau.torch.Solver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=torch.tensor(p["P_row_offsets"]),
            P_col_indices=torch.tensor(p["P_col_indices"]),
            A_row_offsets=torch.tensor(p["A_row_offsets"]),
            A_col_indices=torch.tensor(p["A_col_indices"]),
            cones=p["cones"],
            settings=cpu_settings,
        )

        # Prepare CPU tensors with requires_grad (batched)
        # Create with correct shape first, then enable grad (to ensure leaf tensor)
        P_cpu = (
            torch.tensor(p["P_values"], dtype=torch.float64)
            .unsqueeze(0)
            .clone()
            .requires_grad_(True)
        )
        A_cpu = (
            torch.tensor(p["A_values"], dtype=torch.float64)
            .unsqueeze(0)
            .clone()
            .requires_grad_(True)
        )
        q_cpu = torch.tensor(p["q"], dtype=torch.float64).unsqueeze(0).clone().requires_grad_(True)
        b_cpu = torch.tensor(p["b"], dtype=torch.float64).unsqueeze(0).clone().requires_grad_(True)

        # CPU forward and backward
        solution_cpu = cpu_solver.solve(P_cpu, A_cpu, q_cpu, b_cpu)
        x_cpu, z_cpu, s_cpu = solution_cpu.x, solution_cpu.z, solution_cpu.s
        x_cpu.sum().backward()
        cpu_dq = q_cpu.grad.numpy().squeeze()
        cpu_db = b_cpu.grad.numpy().squeeze()

        # Compare gradients
        np.testing.assert_allclose(
            cpu_dq, gpu_dq, rtol=1e-5, atol=1e-5, err_msg="dq gradient mismatch between CPU and GPU"
        )
        np.testing.assert_allclose(
            cpu_db, gpu_db, rtol=1e-5, atol=1e-5, err_msg="db gradient mismatch between CPU and GPU"
        )

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_soc_backward_consistency(self, soc_problem):
        """Test CPU and CUDA backward passes match for SOC problem."""
        torch = pytest.importorskip("torch")

        p = soc_problem

        # GPU solver with autograd support
        gpu_settings = moreau.Settings(device="cuda", batch_size=1)
        gpu_solver = moreau.torch.Solver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=torch.tensor(p["P_row_offsets"]),
            P_col_indices=torch.tensor(p["P_col_indices"]),
            A_row_offsets=torch.tensor(p["A_row_offsets"]),
            A_col_indices=torch.tensor(p["A_col_indices"]),
            cones=p["cones"],
            settings=gpu_settings,
        )

        # Create with correct shape first, then enable grad (to ensure leaf tensor)
        P_gpu = torch.tensor(p["P_values"], dtype=torch.float64, device="cuda").unsqueeze(0)
        A_gpu = torch.tensor(p["A_values"], dtype=torch.float64, device="cuda").unsqueeze(0)
        q_gpu = (
            torch.tensor(p["q"], dtype=torch.float64, device="cuda")
            .unsqueeze(0)
            .clone()
            .requires_grad_(True)
        )
        b_gpu = torch.tensor(p["b"], dtype=torch.float64, device="cuda").unsqueeze(0)

        solution_gpu = gpu_solver.solve(P_gpu, A_gpu, q_gpu, b_gpu)
        x_gpu, z_gpu, s_gpu = solution_gpu.x, solution_gpu.z, solution_gpu.s
        x_gpu.sum().backward()
        gpu_dq = q_gpu.grad.cpu().numpy().squeeze()

        # CPU solver - same problem
        cpu_settings = moreau.Settings(device="cpu", batch_size=1)
        cpu_solver = moreau.torch.Solver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=torch.tensor(p["P_row_offsets"]),
            P_col_indices=torch.tensor(p["P_col_indices"]),
            A_row_offsets=torch.tensor(p["A_row_offsets"]),
            A_col_indices=torch.tensor(p["A_col_indices"]),
            cones=p["cones"],
            settings=cpu_settings,
        )

        # Create with correct shape first, then enable grad (to ensure leaf tensor)
        P_cpu = torch.tensor(p["P_values"], dtype=torch.float64).unsqueeze(0)
        A_cpu = torch.tensor(p["A_values"], dtype=torch.float64).unsqueeze(0)
        q_cpu = torch.tensor(p["q"], dtype=torch.float64).unsqueeze(0).clone().requires_grad_(True)
        b_cpu = torch.tensor(p["b"], dtype=torch.float64).unsqueeze(0)

        solution_cpu = cpu_solver.solve(P_cpu, A_cpu, q_cpu, b_cpu)
        x_cpu, z_cpu, s_cpu = solution_cpu.x, solution_cpu.z, solution_cpu.s
        x_cpu.sum().backward()
        cpu_dq = q_cpu.grad.numpy().squeeze()

        np.testing.assert_allclose(
            cpu_dq,
            gpu_dq,
            rtol=1e-4,
            atol=1e-4,
            err_msg="SOC dq gradient mismatch between CPU and GPU",
        )

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_exp_cone_backward_consistency(self, exp_cone_problem):
        """Test CPU and CUDA backward passes match for an exp-cone problem.

        Regression for #175: until that fix, CUDA exp-cone backward used a
        looser ``invert_4x4`` pivot floor (1e-8 vs CPU 1e-12) and a coarser
        ``s_eff`` floor (1e-6 vs CPU 1e-10), and silently fell back to
        identity on NaN/Inf — which makes the dual derivative
        ``I - block`` equal to zero, so gradients quietly vanished at the
        cone boundary. This test catches those divergences against the CPU
        reference.
        """
        torch = pytest.importorskip("torch")

        p = exp_cone_problem

        def _solve_grad(device):
            settings = moreau.Settings(device=device, batch_size=1)
            solver = moreau.torch.Solver(
                n=p["n"],
                m=p["m"],
                P_row_offsets=torch.tensor(p["P_row_offsets"]),
                P_col_indices=torch.tensor(p["P_col_indices"]),
                A_row_offsets=torch.tensor(p["A_row_offsets"]),
                A_col_indices=torch.tensor(p["A_col_indices"]),
                cones=p["cones"],
                settings=settings,
            )
            P_t = torch.tensor(p["P_values"], dtype=torch.float64, device=device).unsqueeze(0)
            A_t = torch.tensor(p["A_values"], dtype=torch.float64, device=device).unsqueeze(0)
            q_t = (
                torch.tensor(p["q"], dtype=torch.float64, device=device)
                .unsqueeze(0)
                .clone()
                .requires_grad_(True)
            )
            b_t = (
                torch.tensor(p["b"], dtype=torch.float64, device=device)
                .unsqueeze(0)
                .clone()
                .requires_grad_(True)
            )

            solution = solver.solve(P_t, A_t, q_t, b_t)
            solution.x.sum().backward()
            return (q_t.grad.cpu().numpy().squeeze(), b_t.grad.cpu().numpy().squeeze())

        cpu_dq, cpu_db = _solve_grad("cpu")
        gpu_dq, gpu_db = _solve_grad("cuda")

        # Exp-cone backward is more sensitive to floor/tolerance choices than
        # SOC (we depend on a 4x4 inversion), so the tolerance is a notch
        # looser than the QP test's 1e-5, matching the SOC test's 1e-4.
        np.testing.assert_allclose(
            cpu_dq,
            gpu_dq,
            rtol=1e-4,
            atol=1e-4,
            err_msg="exp-cone dq gradient mismatch between CPU and GPU",
        )
        np.testing.assert_allclose(
            cpu_db,
            gpu_db,
            rtol=1e-4,
            atol=1e-4,
            err_msg="exp-cone db gradient mismatch between CPU and GPU",
        )

        # Sanity: neither side should silently produce zero gradients.
        # (The pre-fix CUDA path's identity fallback would give 0 here.)
        assert np.max(np.abs(gpu_dq)) > 0 or np.max(np.abs(gpu_db)) > 0, (
            "GPU exp-cone backward returned all-zero gradients — likely the "
            "identity-fallback regression of #175."
        )

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    @pytest.mark.parametrize("b_scale", [1.0, 1e-2, 1e-4])
    def test_exp_cone_backward_near_boundary(self, exp_cone_problem, b_scale):
        """Push the exp-cone projection toward the boundary (small ``y``) so
        ``s ≈ 0`` after projection — the regime where the CUDA s_eff floor
        of 1e-6 disagreed with CPU's 1e-10 by orders of magnitude.

        Regression #175. Loosened tolerances vs the basic test because the
        backward is genuinely ill-conditioned near the vertex; we just want
        the two backends to agree."""
        torch = pytest.importorskip("torch")

        p = dict(exp_cone_problem)
        # Shrink y toward zero so r/s_eff is large and alpha = exp(...) huge.
        p["b"] = np.array([p["b"][0], p["b"][1] * b_scale, p["b"][2]])

        def _solve_grad(device):
            settings = moreau.Settings(device=device, batch_size=1)
            solver = moreau.torch.Solver(
                n=p["n"],
                m=p["m"],
                P_row_offsets=torch.tensor(p["P_row_offsets"]),
                P_col_indices=torch.tensor(p["P_col_indices"]),
                A_row_offsets=torch.tensor(p["A_row_offsets"]),
                A_col_indices=torch.tensor(p["A_col_indices"]),
                cones=p["cones"],
                settings=settings,
            )
            P_t = torch.tensor(p["P_values"], dtype=torch.float64, device=device).unsqueeze(0)
            A_t = torch.tensor(p["A_values"], dtype=torch.float64, device=device).unsqueeze(0)
            q_t = (
                torch.tensor(p["q"], dtype=torch.float64, device=device)
                .unsqueeze(0)
                .clone()
                .requires_grad_(True)
            )
            b_t = (
                torch.tensor(p["b"], dtype=torch.float64, device=device)
                .unsqueeze(0)
                .clone()
                .requires_grad_(True)
            )

            solution = solver.solve(P_t, A_t, q_t, b_t)
            solution.x.sum().backward()
            return (q_t.grad.cpu().numpy().squeeze(), b_t.grad.cpu().numpy().squeeze())

        cpu_dq, cpu_db = _solve_grad("cpu")
        gpu_dq, gpu_db = _solve_grad("cuda")

        # Looser tol near the boundary; we mainly want to catch O(1) drift.
        np.testing.assert_allclose(
            cpu_dq,
            gpu_dq,
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"near-boundary (b_scale={b_scale}) " f"dq gradient mismatch",
        )
        np.testing.assert_allclose(
            cpu_db,
            gpu_db,
            rtol=1e-3,
            atol=1e-3,
            err_msg=f"near-boundary (b_scale={b_scale}) " f"db gradient mismatch",
        )


class TestClarabelReference:
    """Test against Clarabel reference solver."""

    @pytest.mark.skipif(not HAS_CLARABEL, reason="clarabel not installed")
    def test_simple_qp_vs_clarabel(self, simple_qp):
        """Test moreau CPU matches Clarabel reference."""
        from scipy import sparse

        p = simple_qp

        # Build scipy sparse matrices for Clarabel
        P_csr = sparse.csr_array(
            (p["P_values"], p["P_col_indices"], p["P_row_offsets"]), shape=(p["n"], p["n"])
        )
        A_csr = sparse.csr_array(
            (p["A_values"], p["A_col_indices"], p["A_row_offsets"]), shape=(p["m"], p["n"])
        )

        # Clarabel uses CSC format
        P_csc = sparse.triu(P_csr).tocsc()
        A_csc = A_csr.tocsc()

        # Clarabel cones
        clarabel_cones = [
            clarabel.ZeroConeT(p["cones"].num_zero_cones),
            clarabel.NonnegativeConeT(p["cones"].num_nonneg_cones),
        ]

        # Solve with Clarabel
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        clarabel_solver = clarabel.DefaultSolver(
            P_csc, p["q"], A_csc, p["b"], clarabel_cones, settings
        )
        clarabel_solution = clarabel_solver.solve()

        # Solve with moreau CPU
        # Convert to sparse matrices
        P_moreau = sparse.csr_array(
            (p["P_values"], p["P_col_indices"], p["P_row_offsets"]), shape=(p["n"], p["n"])
        )
        A_moreau = sparse.csr_array(
            (p["A_values"], p["A_col_indices"], p["A_row_offsets"]), shape=(p["m"], p["n"])
        )

        # Create bounds (unbounded)

        moreau_settings = moreau.Settings(device="cpu")
        moreau_solver = moreau.Solver(
            P_moreau, p["q"], A_moreau, p["b"], p["cones"], moreau_settings
        )
        moreau_result = moreau_solver.solve()
        moreau_info = moreau_solver.info

        # Compare
        np.testing.assert_allclose(
            moreau_result.x,
            clarabel_solution.x,
            rtol=1e-6,
            atol=1e-6,
            err_msg="x mismatch vs Clarabel",
        )
        np.testing.assert_allclose(
            moreau_result.s,
            clarabel_solution.s,
            rtol=1e-6,
            atol=1e-6,
            err_msg="s mismatch vs Clarabel",
        )
        np.testing.assert_allclose(
            moreau_result.z,
            clarabel_solution.z,
            rtol=1e-6,
            atol=1e-6,
            err_msg="z mismatch vs Clarabel",
        )
        np.testing.assert_allclose(
            moreau_info.obj_val,
            clarabel_solution.obj_val,
            rtol=1e-6,
            err_msg="obj_val mismatch vs Clarabel",
        )

    @pytest.mark.skipif(not HAS_CLARABEL, reason="clarabel not installed")
    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_cuda_vs_clarabel(self, simple_qp):
        """Test moreau CUDA matches Clarabel reference."""
        from scipy import sparse

        p = simple_qp

        # Build scipy sparse matrices for Clarabel
        P_csr = sparse.csr_array(
            (p["P_values"], p["P_col_indices"], p["P_row_offsets"]), shape=(p["n"], p["n"])
        )
        A_csr = sparse.csr_array(
            (p["A_values"], p["A_col_indices"], p["A_row_offsets"]), shape=(p["m"], p["n"])
        )

        P_csc = sparse.triu(P_csr).tocsc()
        A_csc = A_csr.tocsc()

        clarabel_cones = [
            clarabel.ZeroConeT(p["cones"].num_zero_cones),
            clarabel.NonnegativeConeT(p["cones"].num_nonneg_cones),
        ]

        settings = clarabel.DefaultSettings()
        settings.verbose = False
        clarabel_solver = clarabel.DefaultSolver(
            P_csc, p["q"], A_csc, p["b"], clarabel_cones, settings
        )
        clarabel_solution = clarabel_solver.solve()

        # Solve with moreau GPU
        # Convert to sparse matrices
        P_moreau = sparse.csr_array(
            (p["P_values"], p["P_col_indices"], p["P_row_offsets"]), shape=(p["n"], p["n"])
        )
        A_moreau = sparse.csr_array(
            (p["A_values"], p["A_col_indices"], p["A_row_offsets"]), shape=(p["m"], p["n"])
        )

        # Create bounds (unbounded)

        moreau_settings = moreau.Settings(device="cuda")
        moreau_solver = moreau.Solver(
            P_moreau, p["q"], A_moreau, p["b"], p["cones"], moreau_settings
        )
        moreau_result = moreau_solver.solve()
        moreau_info = moreau_solver.info

        np.testing.assert_allclose(
            moreau_result.x,
            clarabel_solution.x,
            rtol=1e-6,
            atol=1e-6,
            err_msg="GPU x mismatch vs Clarabel",
        )
        np.testing.assert_allclose(
            moreau_info.obj_val,
            clarabel_solution.obj_val,
            rtol=1e-6,
            err_msg="GPU obj_val mismatch vs Clarabel",
        )


class TestAutoBackendSelection:
    """Test automatic backend selection."""

    def test_auto_uses_smart_device_selection(self, simple_qp):
        """Test that device='auto' uses smart heuristics for selection.

        For small problems (n < 500), CPU should be selected even when CUDA
        is available, since CUDA overhead outweighs parallelism benefits.
        """
        from scipy import sparse

        p = simple_qp

        # Convert to sparse matrices
        P = sparse.csr_array(
            (p["P_values"], p["P_col_indices"], p["P_row_offsets"]), shape=(p["n"], p["n"])
        )
        A = sparse.csr_array(
            (p["A_values"], p["A_col_indices"], p["A_row_offsets"]), shape=(p["m"], p["n"])
        )

        solver = moreau.Solver(
            P,
            p["q"],
            A,
            p["b"],
            p["cones"],
            settings=moreau.Settings(device="auto"),
        )

        # For small problems (n < 500), CPU should be chosen regardless of CUDA availability
        assert solver.device == "cpu"

    def test_auto_selects_cuda_for_large_problems(self):
        """Test that device='auto' selects CUDA for large problems when available."""
        from scipy import sparse

        # Create a large problem (n >= 750) that should use CUDA
        n, m = 800, 400
        P = sparse.diags([1.0] * n, format="csr")
        A = sparse.eye(m, n, format="csr")
        q = np.ones(n)
        b = np.ones(m)
        cones = moreau.Cones(num_nonneg_cones=m)

        solver = moreau.Solver(
            P,
            q,
            A,
            b,
            cones,
            settings=moreau.Settings(device="auto"),
        )

        expected_device = "cuda" if moreau.device_available("cuda") else "cpu"
        assert solver.device == expected_device

    def test_cpu_backend_explicit(self, simple_qp):
        """Test explicit CPU backend selection."""
        from scipy import sparse

        p = simple_qp

        # Convert to sparse matrices
        P = sparse.csr_array(
            (p["P_values"], p["P_col_indices"], p["P_row_offsets"]), shape=(p["n"], p["n"])
        )
        A = sparse.csr_array(
            (p["A_values"], p["A_col_indices"], p["A_row_offsets"]), shape=(p["m"], p["n"])
        )

        solver = moreau.Solver(
            P,
            p["q"],
            A,
            p["b"],
            p["cones"],
            settings=moreau.Settings(device="cpu"),
        )

        assert solver.device == "cpu"
        result = solver.solve()
        info = solver.info
        assert hasattr(result, "x")

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_cuda_backend_explicit(self, simple_qp):
        """Test explicit CUDA backend selection."""
        from scipy import sparse

        p = simple_qp

        # Convert to sparse matrices
        P = sparse.csr_array(
            (p["P_values"], p["P_col_indices"], p["P_row_offsets"]), shape=(p["n"], p["n"])
        )
        A = sparse.csr_array(
            (p["A_values"], p["A_col_indices"], p["A_row_offsets"]), shape=(p["m"], p["n"])
        )

        solver = moreau.Solver(
            P,
            p["q"],
            A,
            p["b"],
            p["cones"],
            settings=moreau.Settings(device="cuda"),
        )

        assert solver.device == "cuda"
        result = solver.solve()
        info = solver.info
        assert hasattr(result, "x")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
