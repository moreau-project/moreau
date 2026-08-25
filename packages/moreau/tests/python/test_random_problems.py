"""Tests for moreau.testing random problem generation."""

import numpy as np
import pytest
from scipy import sparse

import moreau
from moreau.testing import (
    random_cone_program,
    random_cone_program_with_pattern,
    random_batch,
    sample_cone_interior,
    sample_dual_cone_interior,
    RandomProblem,
)


class TestSampleConeInterior:
    """Test cone interior sampling functions."""

    def test_zero_cone(self):
        """Zero cone interior is just {0}."""
        rng = np.random.default_rng(42)
        cones = moreau.Cones(num_zero_cones=5)
        s = sample_cone_interior(cones, rng)
        assert s.shape == (5,)
        np.testing.assert_array_equal(s, np.zeros(5))

    def test_nonneg_cone(self):
        """Nonnegative cone interior has s > 0."""
        rng = np.random.default_rng(42)
        cones = moreau.Cones(num_nonneg_cones=10)
        s = sample_cone_interior(cones, rng)
        assert s.shape == (10,)
        assert np.all(s > 0), "Nonnegative cone interior requires s > 0"

    def test_soc_cone(self):
        """SOC interior has s[0] > ||s[1:]||."""
        rng = np.random.default_rng(42)
        cones = moreau.Cones(so_cone_dims=[3, 3, 3])
        s = sample_cone_interior(cones, rng)
        assert s.shape == (9,)  # 3 cones * 3 dims each

        # Check each SOC
        for i in range(3):
            s_i = s[3 * i : 3 * i + 3]
            norm_tail = np.linalg.norm(s_i[1:])
            assert s_i[0] > norm_tail, f"SOC {i}: s[0]={s_i[0]} should be > ||s[1:]||={norm_tail}"

    def test_exp_cone(self):
        """Exp cone interior has y > 0, z > y*exp(x/y)."""
        rng = np.random.default_rng(42)
        cones = moreau.Cones(num_exp_cones=2)
        s = sample_cone_interior(cones, rng)
        assert s.shape == (6,)  # 2 cones * 3 dims each

        for i in range(2):
            x, y, z = s[3 * i : 3 * i + 3]
            assert y > 0, f"Exp cone {i}: y={y} should be > 0"
            bound = y * np.exp(x / y)
            assert z > bound, f"Exp cone {i}: z={z} should be > y*exp(x/y)={bound}"

    def test_power_cone(self):
        """Power cone interior has x,y > 0, x^a * y^(1-a) > |z|."""
        rng = np.random.default_rng(42)
        cones = moreau.Cones(power_alphas=[0.3, 0.7])
        s = sample_cone_interior(cones, rng)
        assert s.shape == (6,)  # 2 cones * 3 dims each

        for i, alpha in enumerate(cones.power_alphas):
            x, y, z = s[3 * i : 3 * i + 3]
            assert x > 0, f"Power cone {i}: x={x} should be > 0"
            assert y > 0, f"Power cone {i}: y={y} should be > 0"
            bound = (x**alpha) * (y ** (1 - alpha))
            assert abs(z) < bound, f"Power cone {i}: |z|={abs(z)} should be < x^a*y^(1-a)={bound}"

    def test_mixed_cones(self):
        """Test sampling from mixed cone product."""
        rng = np.random.default_rng(42)
        cones = moreau.Cones(
            num_zero_cones=2,
            num_nonneg_cones=3,
            so_cone_dims=[3],
            num_exp_cones=1,
            power_alphas=[0.5],
        )
        s = sample_cone_interior(cones, rng)
        expected_dim = 2 + 3 + 3 + 3 + 3  # 14
        assert s.shape == (expected_dim,)


class TestSampleDualConeInterior:
    """Test dual cone interior sampling."""

    def test_zero_cone_dual(self):
        """Dual of zero cone is R^n (free)."""
        rng = np.random.default_rng(42)
        cones = moreau.Cones(num_zero_cones=5)
        z = sample_dual_cone_interior(cones, rng)
        assert z.shape == (5,)
        # Should be finite (any value works)
        assert np.all(np.isfinite(z))

    def test_nonneg_cone_dual(self):
        """Nonnegative cone is self-dual: z >= 0."""
        rng = np.random.default_rng(42)
        cones = moreau.Cones(num_nonneg_cones=10)
        z = sample_dual_cone_interior(cones, rng)
        assert z.shape == (10,)
        assert np.all(z > 0), "Nonneg dual interior requires z > 0"

    def test_soc_cone_dual(self):
        """SOC is self-dual: z[0] > ||z[1:]||."""
        rng = np.random.default_rng(42)
        cones = moreau.Cones(so_cone_dims=[3, 3])
        z = sample_dual_cone_interior(cones, rng)
        assert z.shape == (6,)

        for i in range(2):
            z_i = z[3 * i : 3 * i + 3]
            norm_tail = np.linalg.norm(z_i[1:])
            assert z_i[0] > norm_tail


class TestRandomConeProgram:
    """Test random cone program generation."""

    def test_basic_qp(self, device):
        """Generate and solve a basic QP."""
        cones = moreau.Cones(num_zero_cones=2, num_nonneg_cones=5)
        prob = random_cone_program(n=10, cones=cones, seed=42)

        assert isinstance(prob, RandomProblem)
        assert prob.P.shape == (10, 10)
        assert prob.A.shape == (7, 10)
        assert prob.q.shape == (10,)
        assert prob.b.shape == (7,)
        assert prob.x_feas.shape == (10,)
        assert prob.s_feas.shape == (7,)

        # Verify feasibility: b = A @ x_feas + s_feas
        np.testing.assert_allclose(
            prob.A @ prob.x_feas + prob.s_feas,
            prob.b,
            rtol=1e-10,
        )

        # Solve it
        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones)
        solution = solver.solve()
        info = solver.info
        assert info.status in [moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved]

    def test_with_soc(self, device):
        """Generate and solve problem with SOC constraints."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2, so_cone_dims=[3, 3])
        prob = random_cone_program(n=8, cones=cones, seed=123)

        expected_m = 1 + 2 + 6  # 9 constraints
        assert prob.A.shape == (expected_m, 8)

        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones)
        solution = solver.solve()
        info = solver.info
        assert info.status in [moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved]

    def test_with_exp_cones(self, device):
        """Generate and solve problem with exponential cones."""
        cones = moreau.Cones(num_zero_cones=1, num_exp_cones=2)
        prob = random_cone_program(n=6, cones=cones, seed=456)

        expected_m = 1 + 6  # 7 constraints
        assert prob.A.shape == (expected_m, 6)

        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones)
        solution = solver.solve()
        info = solver.info
        assert info.status in [moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved]

    def test_with_power_cones(self, device):
        """Generate and solve problem with power cones."""
        cones = moreau.Cones(num_zero_cones=1, power_alphas=[0.3, 0.7])
        prob = random_cone_program(n=6, cones=cones, seed=789)

        expected_m = 1 + 6  # 7 constraints
        assert prob.A.shape == (expected_m, 6)

        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones)
        solution = solver.solve()
        info = solver.info
        assert info.status in [moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved]

    def test_reproducibility(self):
        """Same seed produces identical problems."""
        cones = moreau.Cones(num_zero_cones=2, num_nonneg_cones=3)

        prob1 = random_cone_program(n=5, cones=cones, seed=42)
        prob2 = random_cone_program(n=5, cones=cones, seed=42)

        np.testing.assert_array_equal(prob1.q, prob2.q)
        np.testing.assert_array_equal(prob1.b, prob2.b)
        np.testing.assert_array_equal(prob1.x_feas, prob2.x_feas)
        np.testing.assert_array_equal(prob1.P.toarray(), prob2.P.toarray())
        np.testing.assert_array_equal(prob1.A.toarray(), prob2.A.toarray())


class TestRandomConeProgramWithPattern:
    """Test random problem generation with specified sparsity pattern."""

    def test_respects_pattern(self, device):
        """Generated problem uses specified sparsity pattern."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        n, m = 4, 3

        # Define specific patterns
        P_row_offsets = np.array([0, 1, 2, 3, 4])  # Diagonal P
        P_col_indices = np.array([0, 1, 2, 3])
        A_row_offsets = np.array([0, 2, 3, 4])  # Sparse A
        A_col_indices = np.array([0, 1, 2, 3])

        prob = random_cone_program_with_pattern(
            n=n,
            cones=cones,
            P_row_offsets=P_row_offsets,
            P_col_indices=P_col_indices,
            A_row_offsets=A_row_offsets,
            A_col_indices=A_col_indices,
            seed=42,
        )

        # Check shapes
        assert prob.P.shape == (n, n)
        assert prob.A.shape == (m, n)

        # Check that patterns match
        np.testing.assert_array_equal(prob.A.indptr, A_row_offsets)
        np.testing.assert_array_equal(prob.A.indices, A_col_indices)

        # Should be solvable
        solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones)
        solution = solver.solve()
        info = solver.info
        assert info.status in [moreau.SolverStatus.Solved, moreau.SolverStatus.AlmostSolved]


class TestRandomBatch:
    """Test batch problem generation."""

    def test_shared_pattern(self, device):
        """All problems in batch share same sparsity pattern."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        first, problems = random_batch(n=5, cones=cones, batch_size=4, seed=42)

        assert len(problems) == 4
        assert first is problems[0]

        # All should have same pattern
        for prob in problems[1:]:
            np.testing.assert_array_equal(prob.P.indptr, first.P.indptr)
            np.testing.assert_array_equal(prob.P.indices, first.P.indices)
            np.testing.assert_array_equal(prob.A.indptr, first.A.indptr)
            np.testing.assert_array_equal(prob.A.indices, first.A.indices)

        # But different values
        assert not np.allclose(problems[0].q, problems[1].q)

    def test_batch_solve(self, device):
        """Solve batch of generated problems."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=3)
        first, problems = random_batch(n=6, cones=cones, batch_size=4, seed=42)

        # Use CompiledSolver for batch
        solver = moreau.CompiledSolver(
            n=6,
            m=4,
            P_row_offsets=first.P.indptr,
            P_col_indices=first.P.indices,
            A_row_offsets=first.A.indptr,
            A_col_indices=first.A.indices,
            cones=cones,
            settings=moreau.Settings(batch_size=4),
        )

        P_values = [prob.P.data for prob in problems]
        A_values = [prob.A.data for prob in problems]
        qs = [prob.q for prob in problems]
        bs = [prob.b for prob in problems]

        solver.setup(P_values, A_values)
        solution = solver.solve(qs, bs)
        info = solver.info

        # All should solve
        for i, status in enumerate(info.status):
            assert status in [
                moreau.SolverStatus.Solved,
                moreau.SolverStatus.AlmostSolved,
            ], f"Problem {i} failed with status {status}"


class TestMixedConesBatched:
    """Test batch solving with mixed cone types."""

    def test_soc_batch(self, device):
        """Batch solve with SOC cones."""
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2, so_cone_dims=[3])
        first, problems = random_batch(n=5, cones=cones, batch_size=3, seed=42)

        m = cones.total_constraints()  # 1 + 2 + 3 = 6

        solver = moreau.CompiledSolver(
            n=5,
            m=m,
            P_row_offsets=first.P.indptr,
            P_col_indices=first.P.indices,
            A_row_offsets=first.A.indptr,
            A_col_indices=first.A.indices,
            cones=cones,
            settings=moreau.Settings(batch_size=3),
        )

        P_values = [prob.P.data for prob in problems]
        A_values = [prob.A.data for prob in problems]
        qs = [prob.q for prob in problems]
        bs = [prob.b for prob in problems]

        solver.setup(P_values, A_values)
        solution = solver.solve(qs, bs)
        info = solver.info

        for i, status in enumerate(info.status):
            assert status in [
                moreau.SolverStatus.Solved,
                moreau.SolverStatus.AlmostSolved,
            ], f"Problem {i} failed with status {status}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
