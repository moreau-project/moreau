"""Tests for _types.py data types, containers, and validation.

Covers gaps identified in test audit:
1. BatchedSolution: __getitem__, __len__, __iter__, negative indexing, out-of-bounds
2. TorchBatchedSolution: same container protocol
3. Cones: total_constraints(), degree(), power_alphas validation
4. Settings: validation edge cases (invalid device, solver type, tolerances)
5. IPMSettings: validation (tolerances > 0, direct_solve_method)
6. SolveInfo/BatchedSolveInfo: __repr__, fields
7. _normalize_status: various input types
8. Solution/BatchedSolution: __repr__
"""

import pytest
import numpy as np

from moreau._types import (
    Cones,
    Settings,
    IPMSettings,
    SolverType,
    SolverStatus,
    SolveInfo,
    Solution,
    BatchedSolveInfo,
    BatchedSolution,
    TorchSolveInfo,
    TorchSolution,
    TorchBatchedSolveInfo,
    TorchBatchedSolution,
    _normalize_status,
)

# ============================================================================
# Cones tests
# ============================================================================


class TestConesValidation:
    """Test Cones construction and validation."""

    def test_default_cones(self):
        c = Cones()
        assert c.num_zero_cones == 0
        assert c.num_nonneg_cones == 0
        assert c.num_so_cones == 0
        assert c.num_exp_cones == 0
        assert c.power_alphas == []
        assert c.num_power_cones == 0

    def test_total_constraints_zero_only(self):
        c = Cones(num_zero_cones=5)
        assert c.total_constraints() == 5

    def test_total_constraints_nonneg_only(self):
        c = Cones(num_nonneg_cones=3)
        assert c.total_constraints() == 3

    def test_total_constraints_soc(self):
        c = Cones(so_cone_dims=[3, 3])
        assert c.total_constraints() == 6  # 2 * 3

    def test_total_constraints_exp(self):
        c = Cones(num_exp_cones=4)
        assert c.total_constraints() == 12  # 4 * 3

    def test_total_constraints_power(self):
        c = Cones(power_alphas=[0.5, 0.3])
        assert c.total_constraints() == 6  # 2 * 3

    def test_total_constraints_mixed(self):
        c = Cones(
            num_zero_cones=2,
            num_nonneg_cones=3,
            so_cone_dims=[3],
            num_exp_cones=1,
            power_alphas=[0.5],
        )
        assert c.total_constraints() == 2 + 3 + 3 + 3 + 3  # 14

    def test_degree_zero_cone_not_counted(self):
        c = Cones(num_zero_cones=10)
        assert c.degree() == 0

    def test_degree_nonneg(self):
        c = Cones(num_nonneg_cones=5)
        assert c.degree() == 5

    def test_degree_soc(self):
        c = Cones(so_cone_dims=[3, 3, 3])
        assert c.degree() == 3

    def test_degree_exp(self):
        c = Cones(num_exp_cones=2)
        assert c.degree() == 6  # 3 * 2

    def test_degree_power(self):
        c = Cones(power_alphas=[0.5, 0.7])
        assert c.degree() == 6  # 3 * 2

    def test_degree_mixed(self):
        c = Cones(
            num_zero_cones=5,
            num_nonneg_cones=3,
            so_cone_dims=[3, 3],
            num_exp_cones=1,
            power_alphas=[0.4],
        )
        # 0 + 3 + 2 + 3 + 3 = 11
        assert c.degree() == 11

    def test_num_power_cones(self):
        c = Cones(power_alphas=[0.1, 0.5, 0.9])
        assert c.num_power_cones == 3

    def test_power_alpha_zero_rejected(self):
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            Cones(power_alphas=[0.0])

    def test_power_alpha_one_rejected(self):
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            Cones(power_alphas=[1.0])

    def test_power_alpha_negative_rejected(self):
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            Cones(power_alphas=[-0.5])

    def test_power_alpha_greater_than_one_rejected(self):
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            Cones(power_alphas=[1.5])

    def test_negative_cone_count_rejected(self):
        with pytest.raises(ValueError):
            Cones(num_zero_cones=-1)

    def test_validate_assignment(self):
        """Cones uses validate_assignment so reassignment should also validate."""
        c = Cones()
        with pytest.raises(ValueError):
            c.num_nonneg_cones = -1


# ============================================================================
# Settings validation tests
# ============================================================================


class TestSettingsValidation:
    """Test Settings construction and validation edge cases."""

    def test_default_settings(self):
        s = Settings()
        assert s.device == "auto"
        assert s.batch_size == 1
        assert s.max_iter == 200
        assert s.time_limit == float("inf")
        assert s.verbose is False
        assert s.enable_grad is False
        assert s.solver == SolverType.AUTO
        assert s.ipm_settings is not None  # auto-created for auto/ipm

    def test_invalid_device_rejected(self):
        with pytest.raises(ValueError, match="device must be one of"):
            Settings(device="tpu")

    def test_invalid_device_gpu_rejected(self):
        with pytest.raises(ValueError, match="device must be one of"):
            Settings(device="gpu")

    def test_batch_size_zero_rejected(self):
        with pytest.raises(ValueError):
            Settings(batch_size=0)

    def test_batch_size_negative_rejected(self):
        with pytest.raises(ValueError):
            Settings(batch_size=-1)

    def test_max_iter_zero_rejected(self):
        with pytest.raises(ValueError):
            Settings(max_iter=0)

    def test_time_limit_zero_rejected(self):
        with pytest.raises(ValueError):
            Settings(time_limit=0.0)

    def test_time_limit_negative_rejected(self):
        with pytest.raises(ValueError):
            Settings(time_limit=-1.0)

    def test_time_limit_positive(self):
        s = Settings(time_limit=5.0)
        assert s.time_limit == 5.0

    def test_solver_string_normalized(self):
        s = Settings(solver="ipm")
        assert s.solver == SolverType.IPM

    def test_solver_string_case_insensitive(self):
        s = Settings(solver="IPM")
        assert s.solver == SolverType.IPM

    def test_ipm_settings_auto_created(self):
        s = Settings()
        assert isinstance(s.ipm_settings, IPMSettings)

    def test_explicit_ipm_settings_used(self):
        ipm = IPMSettings(tol_gap_abs=1e-4)
        s = Settings(ipm_settings=ipm)
        assert s.ipm_settings.tol_gap_abs == 1e-4


class TestIPMSettingsValidation:
    """Test IPMSettings validation."""

    def test_defaults(self):
        ipm = IPMSettings()
        assert ipm.tol_gap_abs == 1e-8
        assert ipm.tol_feas == 1e-8
        assert ipm.equilibrate_enable is True
        assert ipm.direct_solve_method == "auto"

    def test_tolerance_zero_rejected(self):
        with pytest.raises(ValueError):
            IPMSettings(tol_gap_abs=0.0)

    def test_tolerance_negative_rejected(self):
        with pytest.raises(ValueError):
            IPMSettings(tol_feas=-1e-8)

    def test_max_step_fraction_zero_rejected(self):
        with pytest.raises(ValueError):
            IPMSettings(max_step_fraction=0.0)

    def test_max_step_fraction_above_one_rejected(self):
        with pytest.raises(ValueError):
            IPMSettings(max_step_fraction=1.01)

    def test_max_step_fraction_one_ok(self):
        ipm = IPMSettings(max_step_fraction=1.0)
        assert ipm.max_step_fraction == 1.0

    def test_invalid_direct_solve_method(self):
        with pytest.raises(ValueError, match="direct_solve_method must be one of"):
            IPMSettings(direct_solve_method="cholmod")

    def test_qdldl_method_accepted(self):
        ipm = IPMSettings(direct_solve_method="qdldl")
        assert ipm.direct_solve_method == "qdldl"


# ============================================================================
# Solution container tests
# ============================================================================


class TestSolutionRepr:
    """Test Solution __repr__."""

    def test_repr(self):
        sol = Solution(x=np.array([1.0, 2.0]), z=np.array([3.0]), s=np.array([4.0]))
        r = repr(sol)
        assert "Solution" in r
        assert "n=2" in r


class TestBatchedSolutionContainer:
    """Test BatchedSolution __getitem__, __len__, __iter__."""

    @pytest.fixture
    def batch_sol(self):
        return BatchedSolution(
            x=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            z=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            s=np.array([[0.7, 0.8], [0.9, 1.0], [1.1, 1.2]]),
        )

    def test_len(self, batch_sol):
        assert len(batch_sol) == 3

    def test_getitem_positive(self, batch_sol):
        sol = batch_sol[0]
        assert isinstance(sol, Solution)
        np.testing.assert_array_equal(sol.x, [1.0, 2.0])
        np.testing.assert_array_equal(sol.z, [0.1, 0.2])

    def test_getitem_last(self, batch_sol):
        sol = batch_sol[2]
        np.testing.assert_array_equal(sol.x, [5.0, 6.0])

    def test_getitem_negative(self, batch_sol):
        sol = batch_sol[-1]
        np.testing.assert_array_equal(sol.x, [5.0, 6.0])

    def test_getitem_negative_first(self, batch_sol):
        sol = batch_sol[-3]
        np.testing.assert_array_equal(sol.x, [1.0, 2.0])

    def test_getitem_out_of_bounds(self, batch_sol):
        with pytest.raises(IndexError, match="out of range"):
            batch_sol[3]

    def test_getitem_negative_out_of_bounds(self, batch_sol):
        with pytest.raises(IndexError, match="out of range"):
            batch_sol[-4]

    def test_iter(self, batch_sol):
        solutions = list(batch_sol)
        assert len(solutions) == 3
        assert all(isinstance(s, Solution) for s in solutions)
        np.testing.assert_array_equal(solutions[0].x, [1.0, 2.0])
        np.testing.assert_array_equal(solutions[2].x, [5.0, 6.0])

    def test_repr(self, batch_sol):
        r = repr(batch_sol)
        assert "BatchedSolution" in r
        assert "batch_size=3" in r


torch = pytest.importorskip("torch")


class TestTorchBatchedSolutionContainer:
    """Test TorchBatchedSolution.__getitem__ threads z_x for direct-x problems."""

    def test_getitem_preserves_z_x(self):
        batch = TorchBatchedSolution(
            x=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
            z=torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float64),
            s=torch.tensor([[0.7, 0.8], [0.9, 1.0]], dtype=torch.float64),
            z_x=torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float64),
        )
        sol = batch[1]
        assert isinstance(sol, TorchSolution)
        torch.testing.assert_close(sol.z_x, torch.tensor([7.0, 8.0], dtype=torch.float64))

    def test_getitem_empty_z_x(self):
        batch = TorchBatchedSolution(
            x=torch.tensor([[1.0, 2.0]], dtype=torch.float64),
            z=torch.tensor([[0.1, 0.2]], dtype=torch.float64),
            s=torch.tensor([[0.7, 0.8]], dtype=torch.float64),
        )
        assert batch[0].z_x.numel() == 0


# ============================================================================
# SolveInfo tests
# ============================================================================


class TestSolveInfo:
    """Test SolveInfo dataclass."""

    def test_construction(self):
        info = SolveInfo(
            status=SolverStatus.Solved,
            obj_val=1.5,
            iterations=10,
            solve_time=0.001,
        )
        assert info.status == SolverStatus.Solved
        assert info.obj_val == 1.5
        assert info.iterations == 10
        assert info.setup_time == 0.0
        assert info.construction_time == 0.0

    def test_repr(self):
        info = SolveInfo(
            status=SolverStatus.Solved,
            obj_val=1.5,
            iterations=10,
            solve_time=0.001,
        )
        r = repr(info)
        assert "SolveInfo" in r
        assert "Solved" in r

    def test_with_timing(self):
        info = SolveInfo(
            status=SolverStatus.MaxIterations,
            obj_val=2.0,
            iterations=200,
            solve_time=1.5,
            setup_time=0.1,
            construction_time=0.05,
        )
        assert info.setup_time == 0.1
        assert info.construction_time == 0.05


class TestBatchedSolveInfo:
    """Test BatchedSolveInfo dataclass."""

    def test_construction(self):
        info = BatchedSolveInfo(
            status=[SolverStatus.Solved, SolverStatus.Solved],
            obj_val=[1.0, 2.0],
            iterations=[10, 12],
            solve_time=0.002,
        )
        assert len(info.status) == 2
        assert info.status[0] == SolverStatus.Solved

    def test_repr(self):
        info = BatchedSolveInfo(
            status=[SolverStatus.Solved, SolverStatus.Solved],
            obj_val=[1.0, 2.0],
            iterations=[10, 12],
            solve_time=0.002,
        )
        r = repr(info)
        assert "BatchedSolveInfo" in r
        assert "batch_size=2" in r


# ============================================================================
# _normalize_status tests
# ============================================================================


class TestNormalizeStatus:
    """Test _normalize_status with various input types."""

    def test_already_solver_status(self):
        result = _normalize_status(SolverStatus.Solved)
        assert result == SolverStatus.Solved

    def test_int(self):
        result = _normalize_status(1)
        assert result == SolverStatus.Solved

    def test_numpy_integer(self):
        result = _normalize_status(np.int64(2))
        assert result == SolverStatus.PrimalInfeasible

    def test_numpy_array(self):
        result = _normalize_status(np.array([1, 2, 3]))
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0] == SolverStatus.Solved
        assert result[1] == SolverStatus.PrimalInfeasible
        assert result[2] == SolverStatus.DualInfeasible

    def test_python_list(self):
        result = _normalize_status([1, 7])
        assert isinstance(result, list)
        assert result[0] == SolverStatus.Solved
        assert result[1] == SolverStatus.MaxIterations

    def test_all_status_values(self):
        for status in SolverStatus:
            result = _normalize_status(int(status))
            assert result == status


# ============================================================================
# SolverStatus tests
# ============================================================================


class TestSolverStatus:
    """Test SolverStatus enum properties."""

    def test_int_comparison(self):
        assert SolverStatus.Solved == 1
        assert SolverStatus.PrimalInfeasible == 2

    def test_name(self):
        assert SolverStatus.Solved.name == "Solved"
        assert SolverStatus.MaxIterations.name == "MaxIterations"

    def test_value(self):
        assert SolverStatus.Solved.value == 1
        assert SolverStatus.NumericalError.value == 9

    def test_all_statuses_exist(self):
        expected = [
            "Unsolved",
            "Solved",
            "PrimalInfeasible",
            "DualInfeasible",
            "AlmostSolved",
            "AlmostPrimalInfeasible",
            "AlmostDualInfeasible",
            "MaxIterations",
            "MaxTime",
            "NumericalError",
            "InsufficientProgress",
            "CallbackTerminated",
        ]
        for name in expected:
            assert hasattr(SolverStatus, name), f"Missing status: {name}"
