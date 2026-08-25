"""
API Contract Tests - Ensures CPU and GPU backends have identical APIs.

Tests run on ALL available devices via the `device` fixture from conftest.py.
These tests prevent API drift between backends.

Run with:
    pytest packages/moreau/tests/python/test_api_contract.py -v
    pytest packages/moreau/tests/python/test_api_contract.py -v --device=cpu
    pytest packages/moreau/tests/python/test_api_contract.py -v --device=cuda
"""

import numpy as np
import pytest

import moreau

try:
    import moreau.torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.fixture
def simple_problem():
    """Simple QP problem for API testing."""
    return {
        "n": 2,
        "m": 3,
        "P_row_offsets": np.array([0, 1, 2], dtype=np.int64),
        "P_col_indices": np.array([0, 1], dtype=np.int64),
        "P_values": np.array([1.0, 1.0]),
        "A_row_offsets": np.array([0, 2, 3, 4], dtype=np.int64),
        "A_col_indices": np.array([0, 1, 0, 1], dtype=np.int64),
        "A_values": np.array([1.0, 1.0, 1.0, 1.0]),
        "q": np.array([2.0, 1.0]),
        "b": np.array([1.0, 0.7, 0.7]),
        "cones": moreau.Cones(num_zero_cones=1, num_nonneg_cones=2),
    }


class TestBackwardReturnKeys:
    """backward() must return identical keys on all backends."""

    def test_backward_returns_six_keys_single(self, device, simple_problem):
        """Single-problem backward() returns all 6 gradient keys."""
        p = simple_problem

        settings = moreau.Settings(device=device, batch_size=1, enable_grad=True)
        solver = moreau.CompiledSolver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings,
        )

        # Single problem (1D arrays passed as batch of 1)
        solver.setup(p["P_values"].reshape(1, -1), p["A_values"].reshape(1, -1))
        solver.solve(
            p["q"].reshape(1, -1),
            p["b"].reshape(1, -1),
        )

        result = solver.backward(
            dx=np.array([[1.0, 0.0]]), dz=np.zeros((1, p["m"])), ds=np.zeros((1, p["m"]))
        )

        expected_keys = {"dP_values", "dq", "dA_values", "db"}
        actual_keys = set(result.keys())
        missing_keys = expected_keys - actual_keys

        assert missing_keys == set(), f"backward() missing keys on {device}: {missing_keys}"

    def test_backward_returns_six_keys_batch(self, device, simple_problem):
        """Batched backward() returns all 6 gradient keys."""
        p = simple_problem
        batch_size = 3

        settings = moreau.Settings(device=device, batch_size=batch_size, enable_grad=True)
        solver = moreau.CompiledSolver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings,
        )

        # Batched problem
        P_batch = np.tile(p["P_values"], (batch_size, 1))
        A_batch = np.tile(p["A_values"], (batch_size, 1))
        q_batch = np.tile(p["q"], (batch_size, 1))
        b_batch = np.tile(p["b"], (batch_size, 1))

        solver.setup(P_batch, A_batch)
        solver.solve(q_batch, b_batch)

        result = solver.backward(
            dx=np.ones((batch_size, p["n"])),
            dz=np.zeros((batch_size, p["m"])),
            ds=np.zeros((batch_size, p["m"])),
        )

        expected_keys = {"dP_values", "dq", "dA_values", "db"}
        actual_keys = set(result.keys())
        missing_keys = expected_keys - actual_keys

        assert missing_keys == set(), f"backward() missing keys on {device} (batch): {missing_keys}"

    def test_backward_single_vs_batch_same_keys(self, device, simple_problem):
        """Single and batch backward must return identical key sets."""
        p = simple_problem

        # Single problem
        settings_single = moreau.Settings(device=device, batch_size=1, enable_grad=True)
        solver_single = moreau.CompiledSolver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings_single,
        )
        solver_single.setup(p["P_values"].reshape(1, -1), p["A_values"].reshape(1, -1))
        solver_single.solve(
            p["q"].reshape(1, -1),
            p["b"].reshape(1, -1),
        )
        result_single = solver_single.backward(
            dx=np.array([[1.0, 0.0]]), dz=np.zeros((1, p["m"])), ds=np.zeros((1, p["m"]))
        )

        # Batch problem
        batch_size = 3
        settings_batch = moreau.Settings(device=device, batch_size=batch_size, enable_grad=True)
        solver_batch = moreau.CompiledSolver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings_batch,
        )
        solver_batch.setup(
            np.tile(p["P_values"], (batch_size, 1)), np.tile(p["A_values"], (batch_size, 1))
        )
        solver_batch.solve(
            np.tile(p["q"], (batch_size, 1)),
            np.tile(p["b"], (batch_size, 1)),
        )
        result_batch = solver_batch.backward(
            dx=np.ones((batch_size, p["n"])),
            dz=np.zeros((batch_size, p["m"])),
            ds=np.zeros((batch_size, p["m"])),
        )

        assert set(result_single.keys()) == set(
            result_batch.keys()
        ), f"Single and batch backward() return different keys on {device}"


class TestSolveSignature:
    """solve() signature must be consistent across backends."""

    def test_solve_works(self, device, simple_problem):
        """solve() works correctly."""
        p = simple_problem

        settings = moreau.Settings(device=device)
        solver = moreau.CompiledSolver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings,
        )
        solver.setup(p["P_values"].reshape(1, -1), p["A_values"].reshape(1, -1))

        result = solver.solve(
            p["q"].reshape(1, -1),
            p["b"].reshape(1, -1),
        )

        assert hasattr(result, "x")
        # status is accessed via solver.info, not on result
        info = solver.info
        assert hasattr(info, "status")


class TestSolutionFields:
    """Solution must have consistent fields across backends."""

    def test_solution_has_required_fields(self, device, simple_problem):
        """Solution must have x, z, s fields; status, obj_val are on solver.info."""
        p = simple_problem

        settings = moreau.Settings(device=device)
        solver = moreau.CompiledSolver(
            n=p["n"],
            m=p["m"],
            P_row_offsets=p["P_row_offsets"],
            P_col_indices=p["P_col_indices"],
            A_row_offsets=p["A_row_offsets"],
            A_col_indices=p["A_col_indices"],
            cones=p["cones"],
            settings=settings,
        )
        solver.setup(p["P_values"].reshape(1, -1), p["A_values"].reshape(1, -1))
        result = solver.solve(
            p["q"].reshape(1, -1),
            p["b"].reshape(1, -1),
        )

        # Solution fields (primal/dual variables)
        solution_fields = ["x", "z", "s"]
        for field in solution_fields:
            assert hasattr(result, field), f"Solution missing '{field}' field on {device}"

        # Info fields (status, objective, etc.) are on solver.info
        info = solver.info
        info_fields = ["status", "obj_val"]
        for field in info_fields:
            assert hasattr(info, field), f"SolveInfo missing '{field}' field on {device}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestTorchBackward:
    """PyTorch backward tests."""

    @pytest.mark.torch
    def test_torch_backward_basic(self, device, simple_problem):
        """PyTorch backward() computes gradients correctly."""
        import torch

        p = simple_problem

        # Determine torch device
        torch_device = "cuda" if device == "cuda" else "cpu"

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

        P = (
            torch.tensor(p["P_values"], dtype=torch.float64, device=torch_device)
            .unsqueeze(0)
            .clone()
            .requires_grad_(True)
        )
        A = (
            torch.tensor(p["A_values"], dtype=torch.float64, device=torch_device)
            .unsqueeze(0)
            .clone()
            .requires_grad_(True)
        )
        q = (
            torch.tensor(p["q"], dtype=torch.float64, device=torch_device)
            .unsqueeze(0)
            .clone()
            .requires_grad_(True)
        )
        b = torch.tensor(p["b"], dtype=torch.float64, device=torch_device).unsqueeze(0)

        solution = solver.solve(P, A, q, b)
        x, z, s = solution.x, solution.z, solution.s

        loss = x.sum()
        loss.backward()

        assert q.grad is not None, f"Gradient not computed for q on {device}"
        # Verify gradients are not all zeros (sanity check)
        assert not torch.all(q.grad == 0), f"Gradients are all zero on {device}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestGradientConsistency:
    """Gradients must be consistent across backends."""

    @pytest.mark.skipif(not moreau.device_available("cuda"), reason="CUDA not available")
    @pytest.mark.torch
    def test_cpu_gpu_gradient_agreement(self, simple_problem):
        """CPU and GPU must produce matching gradients."""
        import torch

        p = simple_problem

        # GPU solve
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
        b_gpu = torch.tensor(p["b"], dtype=torch.float64, device="cuda").unsqueeze(0)

        solution_gpu = gpu_solver.solve(P_gpu, A_gpu, q_gpu, b_gpu)
        solution_gpu.x.sum().backward()
        gpu_dq = q_gpu.grad.cpu().numpy().squeeze()

        # CPU solve
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
        b_cpu = torch.tensor(p["b"], dtype=torch.float64).unsqueeze(0)

        solution_cpu = cpu_solver.solve(P_cpu, A_cpu, q_cpu, b_cpu)
        solution_cpu.x.sum().backward()
        cpu_dq = q_cpu.grad.numpy().squeeze()

        np.testing.assert_allclose(
            cpu_dq, gpu_dq, rtol=1e-5, atol=1e-5, err_msg="dq gradient mismatch between CPU and GPU"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
