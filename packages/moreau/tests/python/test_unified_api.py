"""Tests for the unified Moreau API.

Tests cover:
- moreau.Solver with device='cpu' and device='cuda'
- moreau.CompiledSolver with device='cpu' and device='cuda'
- moreau.torch.Solver with device='cpu' and device='cuda'
- moreau.torch.MoreauLayer with device='cpu' and device='cuda'
- moreau.Settings unified settings class
- moreau.Cones unified cone specification
"""

import pytest
import numpy as np
from scipy import sparse

import moreau
from moreau_cpu._cpu import settings_to_cpu
from moreau_cpu import _cpu_solver

# Check for optional dependencies
torch = pytest.importorskip("torch", reason="PyTorch not installed")


@pytest.fixture
def simple_qp_matrices():
    """Setup for simple QP problem using scipy sparse matrices.

    Problem: minimize 0.5 * (x^2 + y^2) + x + y
             subject to x + y = 1, x >= 0, y >= 0

    Solution: x = y = 0.5, obj_val = 1.25
    """
    n, m = 2, 3

    # P matrix (diagonal): [[1, 0], [0, 1]]
    P = sparse.diags([1.0, 1.0], format="csr")

    # A matrix: [[1, 1], [-1, 0], [0, -1]]
    A = sparse.csr_array(
        [
            [1.0, 1.0],  # x + y = 1 (equality)
            [-1.0, 0.0],  # -x <= 0 => x >= 0
            [0.0, -1.0],  # -y <= 0 => y >= 0
        ]
    )

    # Objective and constraints
    q = np.array([1.0, 1.0], dtype=np.float64)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Cones
    cones = moreau.Cones()
    cones.num_zero_cones = 1  # equality constraint
    cones.num_nonneg_cones = 2  # non-negativity constraints

    return {
        "n": n,
        "m": m,
        "P": P,
        "q": q,
        "A": A,
        "b": b,
        "cones": cones,
        "expected_x": np.array([0.5, 0.5]),
        "expected_obj": 1.25,
    }


@pytest.fixture
def simple_qp_csr():
    """Setup for simple QP problem in CSR format for CompiledSolver.

    Same problem as simple_qp_matrices but in CSR format.
    """
    n, m = 2, 3

    # P matrix (diagonal): [[1, 0], [0, 1]] in CSR
    P_row_offsets = np.array([0, 1, 2], dtype=np.int64)
    P_col_indices = np.array([0, 1], dtype=np.int64)
    P_values = np.array([1.0, 1.0], dtype=np.float64)

    # A matrix: [[1, 1], [-1, 0], [0, -1]] in CSR
    A_row_offsets = np.array([0, 2, 3, 4], dtype=np.int64)
    A_col_indices = np.array([0, 1, 0, 1], dtype=np.int64)
    A_values = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)

    # Objective and constraints
    q = np.array([1.0, 1.0], dtype=np.float64)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # Cones
    cones = moreau.Cones()
    cones.num_zero_cones = 1  # equality constraint
    cones.num_nonneg_cones = 2  # non-negativity constraints

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
        "expected_x": np.array([0.5, 0.5]),
        "expected_obj": 1.25,
    }


class TestSettings:
    """Tests for unified Settings class."""

    def test_settings_defaults(self):
        """Test Settings has correct default values."""
        settings = moreau.Settings()
        assert settings.max_iter == 200
        assert settings.verbose == False
        assert settings.ipm_settings.tol_gap_abs == 1e-8
        assert settings.ipm_settings.tol_feas == 1e-8

    def test_settings_modification(self):
        """Test Settings can be modified."""
        ipm = moreau.IPMSettings(tol_gap_abs=1e-6)
        settings = moreau.Settings(ipm_settings=ipm)
        settings.max_iter = 500
        settings.verbose = True

        assert settings.max_iter == 500
        assert settings.verbose == True
        assert settings.ipm_settings.tol_gap_abs == 1e-6

    def test_settings_to_cpu(self):
        """Test Settings conversion to CPU format."""
        settings = moreau.Settings()
        settings.max_iter = 100
        settings.verbose = True

        cpu_settings = settings.to_device_settings("cpu")
        assert cpu_settings.max_iter == 100
        assert cpu_settings.verbose == True

    def test_settings_to_cpu_converter(self):
        """Regression test: settings_to_cpu must convert Settings to DefaultSettings.

        This tests the direct converter function used by CpuSolver and _TorchSolverCpu.
        Failure here means TorchSolver with CPU fallback would crash with:
        TypeError: 'Settings' object cannot be converted to 'DefaultSettings'
        """
        ipm = moreau.IPMSettings(tol_feas=1e-6)
        settings = moreau.Settings(max_iter=500, verbose=True, ipm_settings=ipm)
        cpu_settings = settings_to_cpu(settings)

        assert isinstance(cpu_settings, _cpu_solver.DefaultSettings)
        assert cpu_settings.max_iter == 500
        assert cpu_settings.verbose == True
        assert cpu_settings.ipm.tol_feas == 1e-6

    def test_settings_to_cpu_passthrough(self):
        """Test that DefaultSettings passes through unchanged."""
        original = _cpu_solver.DefaultSettings()
        original.max_iter = 300

        result = settings_to_cpu(original)

        assert result is original  # Same object, not a copy
        assert result.max_iter == 300


class TestCones:
    """Tests for unified Cones class."""

    def test_cones_creation(self):
        """Test Cones can be created."""
        cones = moreau.Cones()
        assert cones.num_zero_cones == 0
        assert cones.num_nonneg_cones == 0

    def test_cones_basic(self):
        """Test basic cone specification."""
        cones = moreau.Cones()
        cones.num_zero_cones = 1
        cones.num_nonneg_cones = 5

        assert cones.num_zero_cones == 1
        assert cones.num_nonneg_cones == 5
        assert cones.total_constraints() == 6

    def test_cones_soc(self):
        """Test second-order cone specification."""
        cones = moreau.Cones()
        cones.so_cone_dims = [3] * 3

        assert cones.num_so_cones == 3
        assert cones.total_constraints() == 9  # 3 cones * 3 dims each

    def test_cones_power(self):
        """Test power cone specification."""
        cones = moreau.Cones()
        cones.power_alphas = [0.5, 0.3]

        assert cones.num_power_cones == 2
        assert cones.total_constraints() == 6  # 2 * 3


class TestSolverNumpy:
    """Tests for unified numpy Solver (matrix-based interface)."""

    def test_solver_cpu_creation(self, simple_qp_matrices):
        """Test CPU solver can be created."""
        d = simple_qp_matrices
        settings = moreau.Settings(device="cpu")
        solver = moreau.Solver(
            d["P"],
            d["q"],
            d["A"],
            d["b"],
            d["cones"],
            settings,
        )
        assert solver.device == "cpu"
        assert solver.n == d["n"]
        assert solver.m == d["m"]

    def test_solver_cpu_solve(self, simple_qp_matrices):
        """Test CPU solver solves correctly."""
        d = simple_qp_matrices
        settings = moreau.Settings(device="cpu")
        solver = moreau.Solver(
            d["P"],
            d["q"],
            d["A"],
            d["b"],
            d["cones"],
            settings,
        )

        result = solver.solve()
        info = solver.info

        assert info.status == moreau.SolverStatus.Solved
        assert np.allclose(result.x, d["expected_x"], atol=1e-4)
        assert np.isclose(info.obj_val, d["expected_obj"], atol=1e-4)

    def test_compiled_solver_cpu(self, simple_qp_csr):
        """Test CompiledSolver handles batch correctly."""
        d = simple_qp_csr

        # Batch of 3 identical problems
        batch_size = 3

        settings = moreau.Settings(device="cpu", batch_size=batch_size)
        solver = moreau.CompiledSolver(
            d["n"],
            d["m"],
            d["P_row_offsets"],
            d["P_col_indices"],
            d["A_row_offsets"],
            d["A_col_indices"],
            d["cones"],
            settings=settings,
        )

        P_values = np.tile(d["P_values"], (batch_size, 1))
        A_values = np.tile(d["A_values"], (batch_size, 1))
        q = np.tile(d["q"], (batch_size, 1))
        b = np.tile(d["b"], (batch_size, 1))

        solver.setup(P_values, A_values)
        result = solver.solve(q, b)
        info = solver.info

        assert result.x.shape == (batch_size, d["n"])
        for i in range(batch_size):
            assert np.allclose(result.x[i], d["expected_x"], atol=1e-4)

    def test_solver_with_settings(self, simple_qp_matrices):
        """Test solver respects settings."""
        d = simple_qp_matrices
        # Tolerances are in IPMSettings, not Settings
        ipm_settings = moreau.IPMSettings(tol_gap_abs=1e-6)
        settings = moreau.Settings(device="cpu", max_iter=50, ipm_settings=ipm_settings)

        solver = moreau.Solver(
            d["P"],
            d["q"],
            d["A"],
            d["b"],
            d["cones"],
            settings,
        )

        result = solver.solve()
        info = solver.info
        assert info.status == moreau.SolverStatus.Solved

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_solver_cuda_solve(self, simple_qp_matrices):
        """Test CUDA solver solves correctly."""
        d = simple_qp_matrices
        settings = moreau.Settings(device="cuda")
        solver = moreau.Solver(
            d["P"],
            d["q"],
            d["A"],
            d["b"],
            d["cones"],
            settings,
        )

        assert solver.device == "cuda"

        result = solver.solve()
        info = solver.info

        assert info.status == moreau.SolverStatus.Solved
        assert np.allclose(result.x, d["expected_x"], atol=1e-4)

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_solver_auto_uses_smart_device_selection(self, simple_qp_matrices):
        """Test auto device selection uses smart heuristics.

        For small problems, CPU is selected even when CUDA is available.
        """
        d = simple_qp_matrices
        settings = moreau.Settings(device="auto")
        solver = moreau.Solver(
            d["P"],
            d["q"],
            d["A"],
            d["b"],
            d["cones"],
            settings,
        )

        # For small problems (n < 500), expect CPU
        assert solver.device == "cpu"

    def test_auto_uses_ipm_for_small_qp(self):
        """Regression for #43/#48: auto sent small QPs to active-set, which
        reported an unbounded QP as Solved. min 1/2 x0^2 - x1 s.t. x0 <= 1."""
        solver = moreau.Solver(
            sparse.csr_matrix(np.diag([1.0, 0.0])),
            np.array([0.0, -1.0]),
            sparse.csr_matrix([[1.0, 0.0]]),
            np.array([1.0]),
            moreau.Cones(num_nonneg_cones=1),
            moreau.Settings(device="cpu"),
        )
        solver.solve()
        assert solver._settings.solver == moreau.SolverType.IPM
        assert solver.info.status == moreau.SolverStatus.DualInfeasible

    def test_explicit_active_set_is_kept(self, simple_qp_matrices):
        d = simple_qp_matrices
        settings = moreau.Settings(solver="active_set", device="cpu")
        solver = moreau.Solver(d["P"], d["q"], d["A"], d["b"], d["cones"], settings)
        assert solver._settings.solver == moreau.SolverType.ACTIVE_SET


class TestTorchSolver:
    """Tests for moreau.torch.Solver."""

    def test_torch_solver_cpu_creation(self, simple_qp_csr):
        """Test torch CPU solver can be created."""
        from moreau.torch import Solver

        d = simple_qp_csr
        settings = moreau.Settings(device="cpu")
        solver = Solver(
            d["n"],
            d["m"],
            torch.tensor(d["P_row_offsets"]),
            torch.tensor(d["P_col_indices"]),
            torch.tensor(d["A_row_offsets"]),
            torch.tensor(d["A_col_indices"]),
            d["cones"],
            settings=settings,
        )

        assert solver.device == "cpu"
        assert solver.n == d["n"]
        assert solver.m == d["m"]

    def test_torch_solver_cpu_solve(self, simple_qp_csr):
        """Test torch CPU solver solves correctly."""
        from moreau.torch import Solver

        d = simple_qp_csr
        settings = moreau.Settings(device="cpu")
        solver = Solver(
            d["n"],
            d["m"],
            torch.tensor(d["P_row_offsets"]),
            torch.tensor(d["P_col_indices"]),
            torch.tensor(d["A_row_offsets"]),
            torch.tensor(d["A_col_indices"]),
            d["cones"],
            settings=settings,
        )

        result = solver.solve(
            torch.tensor(d["P_values"]),
            torch.tensor(d["A_values"]),
            torch.tensor(d["q"]),
            torch.tensor(d["b"]),
        )
        info = solver.info
        x = result.x

        assert info.status == moreau.SolverStatus.Solved
        assert torch.allclose(x, torch.tensor(d["expected_x"]), atol=1e-4)

    def test_torch_solver_cpu_batch(self, simple_qp_csr):
        """Test torch CPU solver handles batch correctly."""
        from moreau.torch import Solver

        d = simple_qp_csr
        settings = moreau.Settings(device="cpu")
        solver = Solver(
            d["n"],
            d["m"],
            torch.tensor(d["P_row_offsets"]),
            torch.tensor(d["P_col_indices"]),
            torch.tensor(d["A_row_offsets"]),
            torch.tensor(d["A_col_indices"]),
            d["cones"],
            settings=settings,
        )

        batch_size = 4
        P_values = torch.tensor(d["P_values"]).unsqueeze(0).expand(batch_size, -1)
        A_values = torch.tensor(d["A_values"]).unsqueeze(0).expand(batch_size, -1)
        q = torch.tensor(d["q"]).unsqueeze(0).expand(batch_size, -1)
        b = torch.tensor(d["b"]).unsqueeze(0).expand(batch_size, -1)

        result = solver.solve(
            P_values.contiguous(),
            A_values.contiguous(),
            q.contiguous(),
            b.contiguous(),
        )
        x = result.x

        assert x.shape == (batch_size, d["n"])
        for i in range(batch_size):
            assert torch.allclose(x[i], torch.tensor(d["expected_x"]), atol=1e-4)

    def test_torch_solver_cpu_1d_input(self, simple_qp_csr):
        """Test torch CPU solver handles 1D input (single problem)."""
        from moreau.torch import Solver

        d = simple_qp_csr
        settings = moreau.Settings(device="cpu")
        solver = Solver(
            d["n"],
            d["m"],
            torch.tensor(d["P_row_offsets"]),
            torch.tensor(d["P_col_indices"]),
            torch.tensor(d["A_row_offsets"]),
            torch.tensor(d["A_col_indices"]),
            d["cones"],
            settings=settings,
        )

        # 1D inputs (single problem)
        result = solver.solve(
            torch.tensor(d["P_values"]),  # 1D
            torch.tensor(d["A_values"]),  # 1D
            torch.tensor(d["q"]),  # 1D
            torch.tensor(d["b"]),  # 1D
        )
        x = result.x

        # Output should be 1D for 1D input
        assert x.dim() == 1
        assert x.shape == (d["n"],)
        assert torch.allclose(x, torch.tensor(d["expected_x"]), atol=1e-4)

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_torch_solver_cuda_solve(self, simple_qp_csr):
        """Test torch CUDA solver solves correctly."""
        from moreau.torch import Solver

        d = simple_qp_csr
        device = "cuda"
        settings = moreau.Settings(device=device)

        # Set torch defaults for this test
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float64)

        try:
            solver = Solver(
                d["n"],
                d["m"],
                torch.tensor(d["P_row_offsets"]),
                torch.tensor(d["P_col_indices"]),
                torch.tensor(d["A_row_offsets"]),
                torch.tensor(d["A_col_indices"]),
                d["cones"],
                settings=settings,
            )

            assert solver.device == "cuda"

            result = solver.solve(
                torch.tensor(d["P_values"]),
                torch.tensor(d["A_values"]),
                torch.tensor(d["q"]),
                torch.tensor(d["b"]),
            )
            x = result.x

            # CUDA outputs are on GPU
            assert x.device.type == "cuda"
            assert torch.allclose(x.cpu(), torch.tensor(d["expected_x"], device="cpu"), atol=1e-4)
        finally:
            # Reset torch defaults
            torch.set_default_device("cpu")
            torch.set_default_dtype(torch.float32)


class TestTorchFunctionalSolver:
    """Tests for moreau.torch.solver functional API."""

    def test_solver_cpu_creation(self, simple_qp_csr):
        """Test functional solver can be created."""
        from moreau.torch import Solver

        d = simple_qp_csr
        settings = moreau.Settings(device="cpu")
        s = Solver(
            d["n"],
            d["m"],
            torch.tensor(d["P_row_offsets"]),
            torch.tensor(d["P_col_indices"]),
            torch.tensor(d["A_row_offsets"]),
            torch.tensor(d["A_col_indices"]),
            d["cones"],
            settings=settings,
        )
        solve = s.solve

        assert callable(solve)

    def test_solver_cpu_solve(self, simple_qp_csr):
        """Test functional solver solve."""
        from moreau.torch import Solver

        d = simple_qp_csr
        settings = moreau.Settings(device="cpu")
        s = Solver(
            d["n"],
            d["m"],
            torch.tensor(d["P_row_offsets"]),
            torch.tensor(d["P_col_indices"]),
            torch.tensor(d["A_row_offsets"]),
            torch.tensor(d["A_col_indices"]),
            d["cones"],
            settings=settings,
        )
        solve = s.solve

        P_values = torch.tensor(d["P_values"])
        A_values = torch.tensor(d["A_values"])
        q = torch.tensor(d["q"])
        b = torch.tensor(d["b"])

        solution = solve(P_values, A_values, q, b)
        info = s.info

        assert solution.x.shape == (d["n"],)
        assert torch.allclose(solution.x, torch.tensor(d["expected_x"]), atol=1e-4)

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    def test_solver_cuda_solve(self, simple_qp_csr):
        """Test functional solver with CUDA."""
        from moreau.torch import Solver

        d = simple_qp_csr
        device = "cuda"
        settings = moreau.Settings(device=device)

        # Set torch defaults for this test
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float64)

        try:
            s = Solver(
                d["n"],
                d["m"],
                torch.tensor(d["P_row_offsets"]),
                torch.tensor(d["P_col_indices"]),
                torch.tensor(d["A_row_offsets"]),
                torch.tensor(d["A_col_indices"]),
                d["cones"],
                settings=settings,
            )
            solve = s.solve

            P_values = torch.tensor(d["P_values"])
            A_values = torch.tensor(d["A_values"])
            q = torch.tensor(d["q"])
            b = torch.tensor(d["b"])

            solution = solve(P_values, A_values, q, b)
            info = s.info

            assert solution.x.device.type == "cuda"
            assert torch.allclose(
                solution.x.cpu(), torch.tensor(d["expected_x"], device="cpu"), atol=1e-4
            )
        finally:
            # Reset torch defaults
            torch.set_default_device("cpu")
            torch.set_default_dtype(torch.float32)


class TestBackward:
    """Tests for backward pass (gradient computation)."""

    def test_torch_solver_cpu_backward(self, simple_qp_csr):
        """Test torch CPU solver backward pass."""
        from moreau.torch import Solver

        d = simple_qp_csr
        settings = moreau.Settings(device="cpu")
        solver = Solver(
            d["n"],
            d["m"],
            torch.tensor(d["P_row_offsets"]),
            torch.tensor(d["P_col_indices"]),
            torch.tensor(d["A_row_offsets"]),
            torch.tensor(d["A_col_indices"]),
            d["cones"],
            settings=settings,
        )

        # Forward
        result = solver.solve(
            torch.tensor(d["P_values"]),
            torch.tensor(d["A_values"]),
            torch.tensor(d["q"]),
            torch.tensor(d["b"]),
        )
        x, z, s = result.x, result.z, result.s

        # Backward
        dx = torch.ones_like(x)
        dz = torch.zeros_like(z)
        ds = torch.zeros_like(s)

        dP, dq, dA, db = solver.backward(dx, dz, ds)

        assert dP.shape == d["P_values"].shape
        assert dq.shape == d["q"].shape
        assert dA.shape == d["A_values"].shape
        assert db.shape == d["b"].shape

        # Gradients should be finite
        assert torch.isfinite(dP).all()
        assert torch.isfinite(dq).all()
        assert torch.isfinite(dA).all()
        assert torch.isfinite(db).all()


class TestDeviceSelection:
    """Tests for device selection behavior."""

    @pytest.mark.skipif(moreau.device_available("cuda"), reason="CUDA is available")
    def test_cuda_unavailable_raises(self, simple_qp_matrices):
        """Test requesting CUDA when unavailable raises error."""
        d = simple_qp_matrices
        settings = moreau.Settings(device="cuda")

        with pytest.raises(RuntimeError, match="not available"):
            moreau.Solver(
                d["P"],
                d["q"],
                d["A"],
                d["b"],
                d["cones"],
                settings,
            )

    def test_cuda_available_returns_bool(self):
        """Test cuda_available returns boolean."""
        result = moreau.device_available("cuda")
        assert isinstance(result, bool)

    def test_explicit_cpu_device(self, simple_qp_matrices):
        """Test explicit device='cpu' works even when CUDA is available."""
        d = simple_qp_matrices
        settings = moreau.Settings(device="cpu")

        solver = moreau.Solver(
            d["P"],
            d["q"],
            d["A"],
            d["b"],
            d["cones"],
            settings,
        )
        assert solver.device == "cpu"

        result = solver.solve()
        info = solver.info
        assert info.status == moreau.SolverStatus.Solved
