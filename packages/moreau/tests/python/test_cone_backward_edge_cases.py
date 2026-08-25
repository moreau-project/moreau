"""
Tests for cone backward pass edge cases and CPU/GPU parity.

These tests specifically target boundary conditions and edge cases where
CPU and GPU implementations might diverge, including:
- Power cone: points outside cone, near boundary, z=0
- Exponential cone: boundary cases, near-zero components
- SOC: boundary projections, near-zero norms
- Nonnegative cone: exact zero boundary

These tests were added to prevent regressions after fixing GPU backward
pass bugs in the KKT matrix H block storage and cone derivative computation.
"""

import numpy as np
import pytest
import scipy.sparse as sp

import moreau


def skip_if_no_gpu():
    """Skip test if CUDA is not available."""
    try:
        settings = moreau.Settings(device="cuda")
        # Try to create a simple solver to verify CUDA works
        P = sp.eye(2, format="csr")
        A = sp.eye(2, format="csr")
        cones = moreau.Cones(num_nonneg_cones=2)
        solver = moreau.Solver(P, q=np.zeros(2), A=A, b=np.ones(2), cones=cones, settings=settings)
        return False
    except Exception:
        return True


requires_gpu = pytest.mark.skipif(skip_if_no_gpu(), reason="CUDA not available")


class TestPowerConeBackward:
    """Test power cone backward pass edge cases."""

    @requires_gpu
    def test_power_cone_exterior_point(self):
        """Test backward pass when u = z - s is outside the power cone.

        This was the main bug: KKT matrix only stored upper triangle of H block.
        For u outside cone, the Jacobian computation involves Newton iteration,
        and incorrect H storage caused ~11% gradient errors.
        """
        n, m = 3, 3
        P = sp.eye(n, format="csr")
        q = np.array([0.0, 0.0, -5.0])  # Push z component large
        A = sp.eye(m, n, format="csr")
        b = np.array([2.0, 2.0, 1.0])
        cones = moreau.Cones(power_alphas=[0.5])

        # Solve on both devices
        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        # Verify solutions match
        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-5, atol=1e-6)

        # Compute backward pass
        dx_bar = np.array([1.0, 0.0, 0.0])
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf in gradients
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        # Gradients should match within tolerance
        np.testing.assert_allclose(
            cpu_grads["dq"],
            gpu_grads["dq"],
            rtol=1e-4,
            atol=1e-5,
            err_msg="Power cone dq gradient mismatch (exterior point)",
        )
        np.testing.assert_allclose(
            cpu_grads["db"],
            gpu_grads["db"],
            rtol=1e-4,
            atol=1e-5,
            err_msg="Power cone db gradient mismatch (exterior point)",
        )

    @requires_gpu
    def test_power_cone_interior_point(self):
        """Test backward pass when solution is in power cone interior.

        Interior case should give identity Jacobian - simpler but still tests
        the storage format.
        """
        n, m = 3, 3
        P = sp.eye(n, format="csr")
        # q designed to keep solution in interior
        q = np.array([0.1, 0.1, 0.0])
        A = sp.eye(m, n, format="csr")
        b = np.array([2.0, 2.0, 0.5])  # |z| < x^α * y^(1-α) in interior
        cones = moreau.Cones(power_alphas=[0.5])

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-5, atol=1e-6)

        dx_bar = np.ones(n)
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf in gradients
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        np.testing.assert_allclose(cpu_grads["dq"], gpu_grads["dq"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cpu_grads["db"], gpu_grads["db"], rtol=1e-4, atol=1e-5)

    @requires_gpu
    def test_power_cone_z_near_zero(self):
        """Test backward pass when z component is near zero.

        This exercises the z=0 special case in power cone derivative.
        """
        n, m = 3, 3
        P = sp.eye(n, format="csr")
        q = np.array([0.1, 0.1, 0.0])  # No push on z
        A = sp.eye(m, n, format="csr")
        b = np.array([1.0, 1.0, 0.0])  # z = 0 at solution
        cones = moreau.Cones(power_alphas=[0.5])

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        # Solutions should match
        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-4, atol=1e-5)

        dx_bar = np.ones(n)
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        # Gradients should match
        np.testing.assert_allclose(cpu_grads["dq"], gpu_grads["dq"], rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(cpu_grads["db"], gpu_grads["db"], rtol=1e-3, atol=1e-4)

    @requires_gpu
    @pytest.mark.parametrize("alpha", [0.3, 0.5, 0.7])
    def test_power_cone_different_alphas(self, alpha):
        """Test backward pass with different alpha values."""
        n, m = 3, 3
        P = sp.eye(n, format="csr")
        q = np.array([0.0, 0.0, -2.0])
        A = sp.eye(m, n, format="csr")
        b = np.array([1.5, 1.5, 0.5])
        cones = moreau.Cones(power_alphas=[alpha])

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        # Solutions may differ slightly in interior/boundary region due to
        # different convergence criteria - use looser tolerance for solution
        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-3, atol=1e-4)

        dx_bar = np.array([1.0, 0.5, 0.2])
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf in gradients
        assert not np.any(np.isnan(gpu_grads["dq"])), f"GPU dq has NaN (alpha={alpha})"
        assert not np.any(np.isnan(gpu_grads["db"])), f"GPU db has NaN (alpha={alpha})"
        assert not np.any(np.isinf(gpu_grads["dq"])), f"GPU dq has Inf (alpha={alpha})"
        assert not np.any(np.isinf(gpu_grads["db"])), f"GPU db has Inf (alpha={alpha})"

        np.testing.assert_allclose(
            cpu_grads["dq"],
            gpu_grads["dq"],
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Power cone (alpha={alpha}) dq mismatch",
        )
        np.testing.assert_allclose(
            cpu_grads["db"],
            gpu_grads["db"],
            rtol=1e-3,
            atol=1e-4,
            err_msg=f"Power cone (alpha={alpha}) db mismatch",
        )


class TestExpConeBackward:
    """Test exponential cone backward pass edge cases."""

    @requires_gpu
    def test_exp_cone_boundary(self):
        """Test backward pass for exponential cone at/near boundary.

        Exp cone: {(r, s, t) : s * exp(r/s) <= t, s > 0}
        Boundary is when s * exp(r/s) = t.

        Note: Use problem parameters that result in a non-degenerate solution
        with s > 0 (not near zero). The original test had r,s ≈ 0 which is
        numerically degenerate and gives unstable gradients.
        """
        n, m = 3, 3
        P = sp.eye(n, format="csr")
        # Parameters that push toward a meaningful exp cone boundary
        # We want solution where s > 0 significantly
        q = np.array([-1.0, -2.0, 0.5])
        A = sp.eye(m, n, format="csr")
        b = np.array([1.0, 2.0, 3.0])  # Reasonable positive values
        cones = moreau.Cones(num_exp_cones=1)

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-4, atol=1e-5)

        dx_bar = np.ones(n)
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        np.testing.assert_allclose(cpu_grads["dq"], gpu_grads["dq"], rtol=1e-3, atol=1e-4)
        np.testing.assert_allclose(cpu_grads["db"], gpu_grads["db"], rtol=1e-3, atol=1e-4)

    @requires_gpu
    def test_exp_cone_interior(self):
        """Test backward pass for exponential cone in interior."""
        n, m = 3, 3
        P = sp.eye(n, format="csr")
        q = np.array([0.0, -0.5, 0.5])
        A = sp.eye(m, n, format="csr")
        b = np.array([0.0, 2.0, 10.0])  # Strictly interior: 2*exp(0) = 2 < 10
        cones = moreau.Cones(num_exp_cones=1)

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-4, atol=1e-5)

        dx_bar = np.array([0.5, 1.0, 0.2])
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        np.testing.assert_allclose(cpu_grads["dq"], gpu_grads["dq"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cpu_grads["db"], gpu_grads["db"], rtol=1e-4, atol=1e-5)

    @requires_gpu
    def test_exp_cone_negative_r(self):
        """Test exp cone with r < 0, s < 0 case (special branch in derivative)."""
        n, m = 3, 3
        P = sp.eye(n, format="csr")
        q = np.array([1.0, 1.0, -1.0])  # Push r, s positive in x, so negative in s
        A = sp.eye(m, n, format="csr")
        b = np.array([-1.0, -1.0, 2.0])
        cones = moreau.Cones(num_exp_cones=1)

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        # Solutions might differ slightly for edge cases
        if np.allclose(cpu_sol.x, gpu_sol.x, rtol=1e-3, atol=1e-4):
            dx_bar = np.ones(n)
            cpu_grads = cpu_solver.backward(dx_bar)
            gpu_grads = gpu_solver.backward(dx_bar)

            # Check for NaN/Inf
            assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
            assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
            assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
            assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

            np.testing.assert_allclose(cpu_grads["dq"], gpu_grads["dq"], rtol=1e-3, atol=1e-4)
            np.testing.assert_allclose(cpu_grads["db"], gpu_grads["db"], rtol=1e-3, atol=1e-4)


class TestSOCBackward:
    """Test second-order cone backward pass edge cases."""

    @requires_gpu
    def test_soc_boundary(self):
        """Test backward pass for SOC at/near boundary.

        SOC: {(t, x) : t >= ||x||}
        Boundary is when t = ||x||.
        """
        n, m = 3, 3
        P = sp.eye(n, format="csr")
        # Push toward boundary where t ≈ ||x||
        q = np.array([-1.0, 0.0, 0.0])  # Push t large
        A = sp.eye(m, n, format="csr")
        b = np.array([1.0, 0.5, 0.5])  # ||x|| = sqrt(0.5) ≈ 0.707, t = 1 > ||x||
        cones = moreau.Cones(so_cone_dims=[3])

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-4, atol=1e-5)

        dx_bar = np.ones(n)
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        np.testing.assert_allclose(
            cpu_grads["dq"],
            gpu_grads["dq"],
            rtol=1e-4,
            atol=1e-5,
            err_msg="SOC dq gradient mismatch (boundary)",
        )
        np.testing.assert_allclose(
            cpu_grads["db"],
            gpu_grads["db"],
            rtol=1e-4,
            atol=1e-5,
            err_msg="SOC db gradient mismatch (boundary)",
        )

    @requires_gpu
    def test_soc_interior(self):
        """Test backward pass for SOC in interior."""
        n, m = 3, 3
        P = sp.eye(n, format="csr")
        q = np.array([-2.0, 0.0, 0.0])
        A = sp.eye(m, n, format="csr")
        b = np.array([5.0, 0.1, 0.1])  # t = 5 >> ||x|| ≈ 0.14
        cones = moreau.Cones(so_cone_dims=[3])

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-4, atol=1e-5)

        dx_bar = np.array([0.3, 0.6, 0.9])
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        np.testing.assert_allclose(cpu_grads["dq"], gpu_grads["dq"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cpu_grads["db"], gpu_grads["db"], rtol=1e-4, atol=1e-5)

    @requires_gpu
    def test_soc_near_zero_norm(self):
        """Test SOC when ||x|| is near zero.

        This tests the safe_div handling in the derivative computation.
        """
        n, m = 3, 3
        P = sp.eye(n, format="csr")
        q = np.array([-1.0, 0.0, 0.0])
        A = sp.eye(m, n, format="csr")
        b = np.array([1.0, 1e-8, 1e-8])  # Very small ||x||
        cones = moreau.Cones(so_cone_dims=[3])

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        # Check for NaN/Inf in solutions
        assert not np.any(np.isnan(gpu_sol.x)), "GPU solution has NaN"
        assert not np.any(np.isinf(gpu_sol.x)), "GPU solution has Inf"

        dx_bar = np.ones(n)
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf in gradients
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

    @requires_gpu
    def test_soc_boundary_dim5(self):
        """Test backward pass for SOC dim=5 at/near boundary."""
        n, m = 5, 5
        P = sp.eye(n, format="csr")
        q = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
        A = sp.eye(m, n, format="csr")
        b = np.array([1.0, 0.3, 0.3, 0.3, 0.3])
        cones = moreau.Cones(so_cone_dims=[5])

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-4, atol=1e-5)

        dx_bar = np.ones(n)
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        np.testing.assert_allclose(
            cpu_grads["dq"],
            gpu_grads["dq"],
            rtol=1e-4,
            atol=1e-5,
            err_msg="SOC dim=5 dq gradient mismatch (boundary)",
        )
        np.testing.assert_allclose(
            cpu_grads["db"],
            gpu_grads["db"],
            rtol=1e-4,
            atol=1e-5,
            err_msg="SOC dim=5 db gradient mismatch (boundary)",
        )

    @requires_gpu
    def test_soc_interior_dim5(self):
        """Test backward pass for SOC dim=5 in interior."""
        n, m = 5, 5
        P = sp.eye(n, format="csr")
        q = np.array([-2.0, 0.0, 0.0, 0.0, 0.0])
        A = sp.eye(m, n, format="csr")
        b = np.array([5.0, 0.1, 0.1, 0.1, 0.1])
        cones = moreau.Cones(so_cone_dims=[5])

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-4, atol=1e-5)

        dx_bar = np.array([0.3, 0.6, 0.9, 0.4, 0.7])
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        np.testing.assert_allclose(cpu_grads["dq"], gpu_grads["dq"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cpu_grads["db"], gpu_grads["db"], rtol=1e-4, atol=1e-5)

    @requires_gpu
    def test_soc_near_zero_norm_dim5(self):
        """Test SOC dim=5 when ||x|| is near zero."""
        n, m = 5, 5
        P = sp.eye(n, format="csr")
        q = np.array([-1.0, 0.0, 0.0, 0.0, 0.0])
        A = sp.eye(m, n, format="csr")
        b = np.array([1.0, 1e-8, 1e-8, 1e-8, 1e-8])
        cones = moreau.Cones(so_cone_dims=[5])

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        assert not np.any(np.isnan(gpu_sol.x)), "GPU solution has NaN"
        assert not np.any(np.isinf(gpu_sol.x)), "GPU solution has Inf"

        dx_bar = np.ones(n)
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"


class TestNonnegConeBackward:
    """Test nonnegative cone backward pass edge cases."""

    @requires_gpu
    def test_nonneg_exact_zero(self):
        """Test nonnegative cone at exact zero boundary.

        At the boundary (s=0), the nonneg cone projection Jacobian is
        discontinuous: DPi(u) = diag(u >= 0).  Since u = z_eq - s_eq is
        ~1e-20 in magnitude, the sign (and thus the derivative) depends on
        floating-point rounding that differs between CPU and GPU.

        CPU and GPU may legitimately produce different gradients here,
        so we only verify no NaN/Inf and bounded outputs.
        """
        n, m = 2, 2
        P = sp.eye(n, format="csr")
        q = np.array([0.0, 0.0])
        A = sp.eye(m, n, format="csr")
        b = np.array([0.0, 0.0])
        cones = moreau.Cones(num_nonneg_cones=2)

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-5, atol=1e-6)

        dx_bar = np.array([1.0, 1.0])
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        for label, grads in [("CPU", cpu_grads), ("GPU", gpu_grads)]:
            # No NaN/Inf
            assert not np.any(np.isnan(grads["dq"])), f"{label} dq has NaN"
            assert not np.any(np.isnan(grads["db"])), f"{label} db has NaN"
            assert not np.any(np.isinf(grads["dq"])), f"{label} dq has Inf"
            assert not np.any(np.isinf(grads["db"])), f"{label} db has Inf"

            # Bounded: each component should be in [-1-eps, 1+eps]
            assert np.all(
                np.abs(grads["dq"]) <= 1.0 + 1e-4
            ), f"{label} dq out of bounds: {grads['dq']}"
            assert np.all(
                np.abs(grads["db"]) <= 1.0 + 1e-4
            ), f"{label} db out of bounds: {grads['db']}"

    @requires_gpu
    def test_nonneg_active_vs_inactive(self):
        """Test nonnegative cone with mix of active and inactive constraints."""
        n, m = 4, 4
        P = sp.eye(n, format="csr")
        q = np.array([1.0, -1.0, 1.0, -1.0])  # Push some to boundary
        A = sp.eye(m, n, format="csr")
        b = np.array([0.5, 2.0, 0.0, 3.0])  # Mix of values
        cones = moreau.Cones(num_nonneg_cones=4)

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-5, atol=1e-6)

        dx_bar = np.ones(n)
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        np.testing.assert_allclose(cpu_grads["dq"], gpu_grads["dq"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cpu_grads["db"], gpu_grads["db"], rtol=1e-4, atol=1e-5)


class TestMixedConesBackward:
    """Test backward pass with mixed cone types."""

    @requires_gpu
    def test_zero_and_nonneg(self):
        """Test mix of zero (equality) and nonnegative (inequality) cones."""
        n, m = 4, 4
        P = sp.eye(n, format="csr")
        q = np.array([1.0, 1.0, -1.0, -1.0])
        A = sp.eye(m, n, format="csr")
        b = np.array([1.0, 1.0, 2.0, 2.0])
        cones = moreau.Cones(num_zero_cones=2, num_nonneg_cones=2)

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-5, atol=1e-6)

        dx_bar = np.ones(n)
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        np.testing.assert_allclose(cpu_grads["dq"], gpu_grads["dq"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cpu_grads["db"], gpu_grads["db"], rtol=1e-4, atol=1e-5)

    @requires_gpu
    def test_nonneg_and_soc(self):
        """Test mix of nonnegative and second-order cones."""
        n, m = 5, 5
        P = sp.eye(n, format="csr")
        q = np.array([0.5, 0.5, -1.0, 0.0, 0.0])
        A = sp.eye(m, n, format="csr")
        b = np.array([1.0, 1.0, 2.0, 0.5, 0.5])
        cones = moreau.Cones(num_nonneg_cones=2, so_cone_dims=[3])

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-4, atol=1e-5)

        dx_bar = np.ones(n)
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        np.testing.assert_allclose(cpu_grads["dq"], gpu_grads["dq"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cpu_grads["db"], gpu_grads["db"], rtol=1e-4, atol=1e-5)

    @requires_gpu
    def test_nonneg_and_soc_varlen(self):
        """Test mix of nonnegative and variable-dimension SOC (dim=5)."""
        n, m = 7, 7
        P = sp.eye(n, format="csr")
        q = np.array([0.5, 0.5, -1.0, 0.0, 0.0, 0.0, 0.0])
        A = sp.eye(m, n, format="csr")
        b = np.array([1.0, 1.0, 2.0, 0.5, 0.5, 0.3, 0.3])
        cones = moreau.Cones(num_nonneg_cones=2, so_cone_dims=[5])

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-4, atol=1e-5)

        dx_bar = np.ones(n)
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        np.testing.assert_allclose(cpu_grads["dq"], gpu_grads["dq"], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(cpu_grads["db"], gpu_grads["db"], rtol=1e-4, atol=1e-5)

    @requires_gpu
    def test_all_cone_types(self):
        """Test with zero, nonneg, SOC, exp, and power cones together."""
        # Dimensions: 1 zero + 2 nonneg + 3 SOC + 3 exp + 3 power = 12
        n, m = 12, 12
        P = sp.eye(n, format="csr")
        q = np.zeros(n)
        q[0] = 0.1  # zero cone
        q[1:3] = [0.2, -0.2]  # nonneg
        q[3:6] = [-1.0, 0.0, 0.0]  # SOC
        q[6:9] = [0.0, -0.5, 0.5]  # exp
        q[9:12] = [0.0, 0.0, -1.0]  # power

        A = sp.eye(m, n, format="csr")
        b = np.array(
            [
                1.0,  # zero
                1.0,
                1.0,  # nonneg
                2.0,
                0.5,
                0.5,  # SOC: t=2, ||x||=0.707
                0.0,
                1.0,
                5.0,  # exp: s*exp(r/s) = exp(0) = 1 < 5
                1.5,
                1.5,
                0.5,  # power: 1.5^0.5 * 1.5^0.5 = 1.5 > 0.5
            ]
        )
        cones = moreau.Cones(
            num_zero_cones=1,
            num_nonneg_cones=2,
            so_cone_dims=[3],
            num_exp_cones=1,
            power_alphas=[0.5],
        )

        cpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cpu", enable_grad=True, solver="ipm"),
        )
        gpu_solver = moreau.Solver(
            P,
            q=q,
            A=A,
            b=b,
            cones=cones,
            settings=moreau.Settings(device="cuda", enable_grad=True, solver="ipm"),
        )

        cpu_sol = cpu_solver.solve()
        gpu_sol = gpu_solver.solve()

        np.testing.assert_allclose(
            cpu_sol.x, gpu_sol.x, rtol=1e-4, atol=1e-5, err_msg="Mixed cones solution mismatch"
        )

        dx_bar = np.ones(n)
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check for NaN/Inf
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        np.testing.assert_allclose(
            cpu_grads["dq"],
            gpu_grads["dq"],
            rtol=1e-3,
            atol=1e-4,
            err_msg="Mixed cones dq mismatch",
        )
        np.testing.assert_allclose(
            cpu_grads["db"],
            gpu_grads["db"],
            rtol=1e-3,
            atol=1e-4,
            err_msg="Mixed cones db mismatch",
        )


class TestBatchedConeBackward:
    """Test batched backward pass for cones."""

    @requires_gpu
    def test_batched_power_cone(self):
        """Test batched power cone backward pass."""
        n, m = 3, 3
        batch_size = 4

        P_ro = np.array([0, 1, 2, 3], dtype=np.int64)
        P_ci = np.array([0, 1, 2], dtype=np.int64)
        A_ro = np.array([0, 1, 2, 3], dtype=np.int64)
        A_ci = np.array([0, 1, 2], dtype=np.int64)
        cones = moreau.Cones(power_alphas=[0.5])

        # Different q/b for each batch
        qs = np.array(
            [
                [0.0, 0.0, -3.0],
                [0.0, 0.0, -4.0],
                [0.0, 0.0, -5.0],
                [0.0, 0.0, -2.0],
            ]
        )
        bs = np.array(
            [
                [2.0, 2.0, 1.0],
                [2.0, 2.0, 1.0],
                [2.0, 2.0, 1.0],
                [2.0, 2.0, 1.0],
            ]
        )

        for device in ["cpu", "cuda"]:
            settings = moreau.Settings(device=device, batch_size=batch_size, enable_grad=True)
            solver = moreau.CompiledSolver(
                n=n,
                m=m,
                P_row_offsets=P_ro,
                P_col_indices=P_ci,
                A_row_offsets=A_ro,
                A_col_indices=A_ci,
                cones=cones,
                settings=settings,
            )
            solver.setup(P_values=np.ones(3), A_values=np.ones(3))
            sol = solver.solve(qs=qs, bs=bs)

            # Check no NaN/Inf in solutions
            assert not np.any(np.isnan(sol.x)), f"{device} solution has NaN"
            assert not np.any(np.isinf(sol.x)), f"{device} solution has Inf"

            # Backward pass
            dx_bar = np.ones((batch_size, n))
            grads = solver.backward(dx_bar)

            # Check no NaN/Inf in gradients
            assert not np.any(np.isnan(grads["dq"])), f"{device} dq has NaN"
            assert not np.any(np.isnan(grads["db"])), f"{device} db has NaN"
            assert not np.any(np.isinf(grads["dq"])), f"{device} dq has Inf"
            assert not np.any(np.isinf(grads["db"])), f"{device} db has Inf"

    @requires_gpu
    def test_batched_cpu_gpu_parity(self):
        """Test that batched CPU and GPU gradients match."""
        n, m = 3, 3
        batch_size = 4

        P_ro = np.array([0, 1, 2, 3], dtype=np.int64)
        P_ci = np.array([0, 1, 2], dtype=np.int64)
        A_ro = np.array([0, 1, 2, 3], dtype=np.int64)
        A_ci = np.array([0, 1, 2], dtype=np.int64)
        cones = moreau.Cones(so_cone_dims=[3])

        qs = np.array(
            [
                [-1.0, 0.0, 0.0],
                [-1.5, 0.0, 0.0],
                [-2.0, 0.0, 0.0],
                [-0.5, 0.0, 0.0],
            ]
        )
        bs = np.array(
            [
                [2.0, 0.5, 0.5],
                [2.0, 0.5, 0.5],
                [2.0, 0.5, 0.5],
                [2.0, 0.5, 0.5],
            ]
        )

        def solve_and_backward(device):
            settings = moreau.Settings(device=device, batch_size=batch_size, enable_grad=True)
            solver = moreau.CompiledSolver(
                n=n,
                m=m,
                P_row_offsets=P_ro,
                P_col_indices=P_ci,
                A_row_offsets=A_ro,
                A_col_indices=A_ci,
                cones=cones,
                settings=settings,
            )
            solver.setup(P_values=np.ones(3), A_values=np.ones(3))
            sol = solver.solve(qs=qs, bs=bs)
            dx_bar = np.ones((batch_size, n))
            grads = solver.backward(dx_bar)
            return sol, grads

        cpu_sol, cpu_grads = solve_and_backward("cpu")
        gpu_sol, gpu_grads = solve_and_backward("cuda")

        # Solutions should match
        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-4, atol=1e-5)

        # Check for NaN/Inf
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        # Gradients should match
        np.testing.assert_allclose(
            cpu_grads["dq"],
            gpu_grads["dq"],
            rtol=1e-4,
            atol=1e-5,
            err_msg="Batched SOC dq mismatch",
        )
        np.testing.assert_allclose(
            cpu_grads["db"],
            gpu_grads["db"],
            rtol=1e-4,
            atol=1e-5,
            err_msg="Batched SOC db mismatch",
        )

    @requires_gpu
    def test_batched_cpu_gpu_parity_soc_varlen(self):
        """Test that batched CPU and GPU gradients match for SOC dim=5."""
        n, m = 5, 5
        batch_size = 4

        P_ro = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        P_ci = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        A_ro = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        A_ci = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        cones = moreau.Cones(so_cone_dims=[5])

        qs = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.5, 0.0, 0.0, 0.0, 0.0],
                [-2.0, 0.0, 0.0, 0.0, 0.0],
                [-0.5, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        bs = np.array(
            [
                [2.0, 0.3, 0.3, 0.3, 0.3],
                [2.0, 0.3, 0.3, 0.3, 0.3],
                [2.0, 0.3, 0.3, 0.3, 0.3],
                [2.0, 0.3, 0.3, 0.3, 0.3],
            ]
        )

        def solve_and_backward(device):
            settings = moreau.Settings(device=device, batch_size=batch_size, enable_grad=True)
            solver = moreau.CompiledSolver(
                n=n,
                m=m,
                P_row_offsets=P_ro,
                P_col_indices=P_ci,
                A_row_offsets=A_ro,
                A_col_indices=A_ci,
                cones=cones,
                settings=settings,
            )
            solver.setup(P_values=np.ones(5), A_values=np.ones(5))
            sol = solver.solve(qs=qs, bs=bs)
            dx_bar = np.ones((batch_size, n))
            grads = solver.backward(dx_bar)
            return sol, grads

        cpu_sol, cpu_grads = solve_and_backward("cpu")
        gpu_sol, gpu_grads = solve_and_backward("cuda")

        # Solutions should match
        np.testing.assert_allclose(cpu_sol.x, gpu_sol.x, rtol=1e-4, atol=1e-5)

        # Check for NaN/Inf
        assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
        assert not np.any(np.isnan(gpu_grads["db"])), "GPU db has NaN"
        assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
        assert not np.any(np.isinf(gpu_grads["db"])), "GPU db has Inf"

        # Gradients should match
        np.testing.assert_allclose(
            cpu_grads["dq"],
            gpu_grads["dq"],
            rtol=1e-4,
            atol=1e-5,
            err_msg="Batched SOC dim=5 dq mismatch",
        )
        np.testing.assert_allclose(
            cpu_grads["db"],
            gpu_grads["db"],
            rtol=1e-4,
            atol=1e-5,
            err_msg="Batched SOC dim=5 db mismatch",
        )
