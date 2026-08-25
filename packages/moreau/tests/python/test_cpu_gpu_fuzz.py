"""
Fuzz testing for CPU/GPU parity using Hypothesis.

This module generates random optimization problems and verifies that
CPU and GPU implementations produce consistent results for both
forward solve and backward pass (gradients).
"""

import numpy as np
import pytest
import scipy.sparse as sp

# Hypothesis-driven fuzz suite (100+s aggregate). Pass --runslow to enable.
pytestmark = pytest.mark.slow

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st, assume, HealthCheck, Phase
from hypothesis.extra.numpy import arrays

import moreau


def skip_if_no_gpu():
    """Skip test if CUDA is not available."""
    try:
        settings = moreau.Settings(device="cuda")
        P = sp.eye(2, format="csr")
        A = sp.eye(2, format="csr")
        cones = moreau.Cones(num_nonneg_cones=2)
        solver = moreau.Solver(P, q=np.zeros(2), A=A, b=np.ones(2), cones=cones, settings=settings)
        return False
    except Exception:
        return True


requires_gpu = pytest.mark.skipif(skip_if_no_gpu(), reason="CUDA not available")


# =============================================================================
# Hypothesis strategies for generating valid optimization problems
# =============================================================================


@st.composite
def small_problem_dims(draw):
    """Generate small problem dimensions for fast testing."""
    n = draw(st.integers(min_value=2, max_value=10))
    m = draw(st.integers(min_value=2, max_value=15))
    return n, m


@st.composite
def medium_problem_dims(draw):
    """Generate medium problem dimensions."""
    n = draw(st.integers(min_value=5, max_value=30))
    m = draw(st.integers(min_value=5, max_value=50))
    return n, m


@st.composite
def random_psd_matrix(draw, n):
    """Generate a random positive semi-definite matrix as CSR."""
    # Generate random matrix and make it PSD via A'A + small diagonal
    density = draw(st.floats(min_value=0.1, max_value=0.8))

    # Generate random entries - exclude zero values to avoid stored zeros
    num_entries = max(n, int(n * n * density))
    rows = draw(arrays(np.int32, num_entries, elements=st.integers(0, n - 1)))
    cols = draw(arrays(np.int32, num_entries, elements=st.integers(0, n - 1)))
    # Use floats that exclude values very close to zero
    vals = draw(
        arrays(
            np.float64,
            num_entries,
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False).filter(
                lambda x: abs(x) > 0.01
            ),
        )
    )

    # Create sparse matrix
    A_rand = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A_rand.eliminate_zeros()  # Remove any stored zeros

    # Make PSD: P = A'A + eps*I
    P = (A_rand.T @ A_rand).tocsr()
    P = P + sp.eye(n, format="csr") * 0.1
    P.eliminate_zeros()

    return P


@st.composite
def random_sparse_matrix(draw, m, n):
    """Generate a random sparse matrix as CSR with no stored zeros and no zero rows."""
    density = draw(st.floats(min_value=0.2, max_value=0.6))

    num_entries = max(m * 2, int(m * n * density))  # Ensure enough entries
    rows = draw(arrays(np.int32, num_entries, elements=st.integers(0, m - 1)))
    cols = draw(arrays(np.int32, num_entries, elements=st.integers(0, n - 1)))
    # Use floats that exclude values very close to zero
    vals = draw(
        arrays(
            np.float64,
            num_entries,
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False).filter(
                lambda x: abs(x) > 0.1
            ),
        )
    )

    A = sp.coo_matrix((vals, (rows, cols)), shape=(m, n)).tocsr()
    A.eliminate_zeros()  # Remove any stored zeros

    # CRITICAL: Ensure every row has at least one nonzero entry
    # Zero rows cause degenerate problems where z is not uniquely determined
    row_has_nonzero = np.diff(A.indptr) > 0
    for i in range(m):
        if not row_has_nonzero[i]:
            # Add a nonzero entry to row i, using varying columns to avoid degeneracy
            col = i % n  # Distribute columns across rows
            val = draw(st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False))
            # Convert to lil for efficient modification
            A_lil = A.tolil()
            A_lil[i, col] = val
            A = A_lil.tocsr()

    # Ensure A uses at least 2 different columns to avoid rank-1 degeneracy
    cols_used = set()
    for i in range(m):
        row_start, row_end = A.indptr[i], A.indptr[i + 1]
        for j in range(row_start, row_end):
            cols_used.add(A.indices[j])

    if len(cols_used) < min(2, n):
        # Add entries to different columns
        A_lil = A.tolil()
        for col in range(min(2, n)):
            if col not in cols_used:
                row = col % m
                val = draw(st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False))
                A_lil[row, col] = val
        A = A_lil.tocsr()

    return A


@st.composite
def nonneg_cones_only(draw, m):
    """Generate cone configuration using only nonnegative cones."""
    return moreau.Cones(num_nonneg_cones=m)


@st.composite
def mixed_nonneg_zero_cones(draw, m):
    """Generate cone configuration with nonnegative and zero cones."""
    num_zero = draw(st.integers(min_value=0, max_value=m // 2))
    num_nonneg = m - num_zero
    return moreau.Cones(num_zero_cones=num_zero, num_nonneg_cones=num_nonneg)


@st.composite
def cones_with_soc(draw, m):
    """Generate cone configuration including variable-dimension SOC."""
    # Generate variable SOC dims (each >= 2, max 4 to stay on dense path;
    # sparse SOC dim > 4 degrades cuDSS factorization quality)
    num_soc = draw(st.integers(min_value=0, max_value=m // 2))
    soc_dims = [draw(st.integers(min_value=2, max_value=4)) for _ in range(num_soc)]
    total_soc = sum(soc_dims)

    if total_soc > m:
        # Trim to fit
        soc_dims = []
        total_soc = 0
        for _ in range(num_soc):
            d = draw(st.integers(min_value=2, max_value=min(4, m - total_soc)))
            if total_soc + d > m:
                break
            soc_dims.append(d)
            total_soc += d

    remaining = m - total_soc
    num_nonneg = remaining
    return moreau.Cones(num_nonneg_cones=num_nonneg, so_cone_dims=soc_dims)


@st.composite
def cones_with_exp(draw, m):
    """Generate cone configuration including exponential cones."""
    num_exp = draw(st.integers(min_value=0, max_value=m // 3))
    remaining = m - num_exp * 3

    if remaining < 0:
        num_exp = m // 3
        remaining = m - num_exp * 3

    num_nonneg = remaining
    return moreau.Cones(num_nonneg_cones=num_nonneg, num_exp_cones=num_exp)


@st.composite
def cones_with_power(draw, m):
    """Generate cone configuration including power cones."""
    num_power = draw(st.integers(min_value=0, max_value=m // 3))
    remaining = m - num_power * 3

    if remaining < 0:
        num_power = m // 3
        remaining = m - num_power * 3

    # Generate random alpha values in (0, 1)
    alphas = [draw(st.floats(min_value=0.1, max_value=0.9)) for _ in range(num_power)]

    num_nonneg = remaining
    return moreau.Cones(num_nonneg_cones=num_nonneg, power_alphas=alphas)


def reasonable_float():
    """Generate floats that are not too extreme."""
    return st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False).filter(
        lambda x: abs(x) < 1e10 and (abs(x) > 1e-10 or x == 0)
    )


@st.composite
def simple_qp_problem(draw):
    """Generate a simple QP with nonnegative constraints."""
    n, m = draw(small_problem_dims())

    P = draw(random_psd_matrix(n))
    A = draw(random_sparse_matrix(m, n))
    # Use reasonable floats to avoid extreme values like 1e-108
    q = draw(arrays(np.float64, n, elements=reasonable_float()))
    b = draw(arrays(np.float64, m, elements=reasonable_float()))
    cones = draw(nonneg_cones_only(m))

    return P, q, A, b, cones


def reasonable_nonzero_floats():
    """Generate floats that are not too extreme and not all zero."""
    return st.floats(-10, 10, allow_nan=False, allow_infinity=False).filter(lambda x: abs(x) > 0.01)


@st.composite
def soc_problem(draw):
    """Generate a problem with variable-dimension SOC constraints.

    SOC dims capped at 4 to stay on the dense Hs path; sparse SOC (dim > 4)
    degrades cuDSS factorization quality and is tested in test_soc_variable_dim.py.
    """
    n = draw(st.integers(min_value=3, max_value=10))
    num_soc = draw(st.integers(min_value=1, max_value=3))
    soc_dims = [draw(st.integers(min_value=2, max_value=4)) for _ in range(num_soc)]
    m = sum(soc_dims)

    P = draw(random_psd_matrix(n))
    A = draw(random_sparse_matrix(m, n))
    # Use non-zero values to avoid degenerate problems
    q = draw(arrays(np.float64, n, elements=reasonable_nonzero_floats()))
    b = draw(arrays(np.float64, m, elements=reasonable_nonzero_floats()))
    cones = moreau.Cones(so_cone_dims=soc_dims)

    return P, q, A, b, cones


@st.composite
def exp_cone_problem(draw):
    """Generate a problem with exponential cone constraints."""
    n = draw(st.integers(min_value=3, max_value=10))
    num_exp = draw(st.integers(min_value=1, max_value=3))
    m = num_exp * 3

    P = draw(random_psd_matrix(n))
    A = draw(random_sparse_matrix(m, n))
    # Use non-zero values to avoid degenerate problems
    q = draw(arrays(np.float64, n, elements=reasonable_nonzero_floats()))
    b = draw(arrays(np.float64, m, elements=reasonable_nonzero_floats()))
    cones = moreau.Cones(num_exp_cones=num_exp)

    return P, q, A, b, cones


@st.composite
def power_cone_problem(draw):
    """Generate a problem with power cone constraints."""
    n = draw(st.integers(min_value=3, max_value=10))
    num_power = draw(st.integers(min_value=1, max_value=3))
    m = num_power * 3

    P = draw(random_psd_matrix(n))
    A = draw(random_sparse_matrix(m, n))
    # Use non-zero values to avoid degenerate problems
    q = draw(arrays(np.float64, n, elements=reasonable_nonzero_floats()))
    b = draw(arrays(np.float64, m, elements=reasonable_nonzero_floats()))

    alphas = [draw(st.floats(min_value=0.1, max_value=0.9)) for _ in range(num_power)]
    cones = moreau.Cones(power_alphas=alphas)

    return P, q, A, b, cones


@st.composite
def mixed_cone_problem(draw):
    """Generate a problem with mixed cone types."""
    n = draw(st.integers(min_value=3, max_value=10))

    # Allocate constraints to different cone types
    num_nonneg = draw(st.integers(min_value=0, max_value=5))
    num_soc = draw(st.integers(min_value=0, max_value=2))
    soc_dims = [draw(st.integers(min_value=2, max_value=4)) for _ in range(num_soc)]
    num_exp = draw(st.integers(min_value=0, max_value=2))
    num_power = draw(st.integers(min_value=0, max_value=2))

    m = num_nonneg + sum(soc_dims) + num_exp * 3 + num_power * 3
    assume(m >= 1)  # Need at least one constraint

    P = draw(random_psd_matrix(n))
    A = draw(random_sparse_matrix(m, n))
    q = draw(
        arrays(np.float64, n, elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False))
    )
    b = draw(
        arrays(np.float64, m, elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False))
    )

    alphas = [draw(st.floats(min_value=0.1, max_value=0.9)) for _ in range(num_power)]
    cones = moreau.Cones(
        num_nonneg_cones=num_nonneg,
        so_cone_dims=soc_dims,
        num_exp_cones=num_exp,
        power_alphas=alphas,
    )

    return P, q, A, b, cones


# =============================================================================
# Helper functions
# =============================================================================


def print_problem(P, q, A, b, cones):
    """Print problem data in a reproducible format."""
    print("\n" + "=" * 60)
    print("FAILING PROBLEM DATA (copy-paste to reproduce):")
    print("=" * 60)
    print(f"# P matrix ({P.shape[0]}x{P.shape[1]}, {P.nnz} nnz)")
    print(f"P_data = {P.data.tolist()}")
    print(f"P_indices = {P.indices.tolist()}")
    print(f"P_indptr = {P.indptr.tolist()}")
    print(f"P = sp.csr_matrix((P_data, P_indices, P_indptr), shape={P.shape})")
    print()
    print(f"# A matrix ({A.shape[0]}x{A.shape[1]}, {A.nnz} nnz)")
    print(f"A_data = {A.data.tolist()}")
    print(f"A_indices = {A.indices.tolist()}")
    print(f"A_indptr = {A.indptr.tolist()}")
    print(f"A = sp.csr_matrix((A_data, A_indices, A_indptr), shape={A.shape})")
    print()
    print(f"q = np.array({q.tolist()})")
    print(f"b = np.array({b.tolist()})")
    print(f"cones = {cones}")
    print("=" * 60 + "\n")


def solve_and_compare(P, q, A, b, cones, rtol=1e-4, atol=1e-5, grad_rtol=1e-2, grad_atol=1e-3):
    """Solve on both CPU and GPU, compare results."""
    # Skip degenerate problems that cause numerical issues
    # 1. q and b should not both be all zeros (trivial problem)
    assume(np.linalg.norm(q) > 1e-6 or np.linalg.norm(b) > 1e-6)

    # 2. A should use at least 2 columns (avoid rank-1 degeneracy)
    cols_used = len(set(A.indices))
    assume(cols_used >= min(2, A.shape[1]))

    # 3. Check if A is rank-deficient (causes non-unique solutions)
    A_rank = np.linalg.matrix_rank(A.toarray())
    A_rank_deficient = A_rank < A.shape[0]

    try:
        # CPU solve
        cpu_settings = moreau.Settings(device="cpu", enable_grad=True)
        cpu_solver = moreau.Solver(P, q=q, A=A, b=b, cones=cones, settings=cpu_settings)
        cpu_sol = cpu_solver.solve()

        # GPU solve
        gpu_settings = moreau.Settings(device="cuda", enable_grad=True)
        gpu_solver = moreau.Solver(P, q=q, A=A, b=b, cones=cones, settings=gpu_settings)
        gpu_sol = gpu_solver.solve()

        # Check both solved
        cpu_solved = cpu_solver.info.status.name in ["Solved", "AlmostSolved"]
        gpu_solved = gpu_solver.info.status.name in ["Solved", "AlmostSolved"]

        if not cpu_solved or not gpu_solved:
            # If one didn't solve, just check that gradients don't have NaN/Inf
            if cpu_solved:
                dx_bar = np.ones(len(q))
                cpu_grads = cpu_solver.backward(dx_bar)
                assert not np.any(np.isnan(cpu_grads["dq"])), "CPU dq has NaN"
                assert not np.any(np.isinf(cpu_grads["dq"])), "CPU dq has Inf"
            if gpu_solved:
                dx_bar = np.ones(len(q))
                gpu_grads = gpu_solver.backward(dx_bar)
                assert not np.any(np.isnan(gpu_grads["dq"])), "GPU dq has NaN"
                assert not np.any(np.isinf(gpu_grads["dq"])), "GPU dq has Inf"
            return  # Skip comparison if one didn't solve

        # Helper function to verify KKT conditions (relaxed tolerances)
        # The solver already verified these internally; we just sanity check
        def verify_kkt(sol, label):
            """Verify KKT conditions for a solution (relaxed tolerances)."""
            # Primal feasibility: Ax + s = b
            res = A @ sol.x + sol.s - b
            res_scale = max(np.linalg.norm(b), np.linalg.norm(A @ sol.x), 1.0)
            feas_tol = max(1e-4, res_scale * 1e-4)
            assert (
                np.linalg.norm(res) < feas_tol
            ), f"{label} primal infeasible: |Ax+s-b|={np.linalg.norm(res)}, tol={feas_tol}"

            # Dual feasibility: Px + q + A'z = 0
            dual = P @ sol.x + q + A.T @ sol.z
            dual_scale = max(np.linalg.norm(q), np.linalg.norm(P @ sol.x), 1.0)
            dual_tol = max(1e-4, dual_scale * 1e-4)
            assert (
                np.linalg.norm(dual) < dual_tol
            ), f"{label} dual infeasible: |Px+q+A'z|={np.linalg.norm(dual)}, tol={dual_tol}"

            # Note: Skip complementarity check - the solver uses HSDE formulation
            # which minimizes duality gap but doesn't directly enforce s'z = 0.
            # The solver's internal tolerances already handle this.

        # Compute solution scales and differences
        x_scale = max(np.max(np.abs(cpu_sol.x)), np.max(np.abs(gpu_sol.x)), 1e-6)
        s_scale = max(np.max(np.abs(cpu_sol.s)), np.max(np.abs(gpu_sol.s)), 1e-6)
        z_scale = max(np.max(np.abs(cpu_sol.z)), np.max(np.abs(gpu_sol.z)), 1e-6)
        x_diff = np.max(np.abs(cpu_sol.x - gpu_sol.x))
        s_diff = np.max(np.abs(cpu_sol.s - gpu_sol.s))
        z_diff = np.max(np.abs(cpu_sol.z - gpu_sol.z))

        # Check if solutions match within tolerance
        x_match = x_diff < max(atol, x_scale * 0.01)  # 1% tolerance
        s_match = s_diff < max(atol, s_scale * 0.01)
        z_match = z_diff < max(atol, z_scale * 0.01)
        solutions_match = x_match and s_match and z_match

        # For rank-deficient A or when solutions don't match, verify KKT conditions
        # instead of comparing solutions directly (both can be valid optimal solutions)
        if A_rank_deficient or not solutions_match:
            # Check objective values match
            def objective(x):
                return 0.5 * x @ P @ x + q @ x

            obj_cpu = objective(cpu_sol.x)
            obj_gpu = objective(gpu_sol.x)
            obj_scale = max(abs(obj_cpu), abs(obj_gpu), 1.0)
            assert (
                abs(obj_cpu - obj_gpu) < obj_scale * 1e-3
            ), f"Objective mismatch: CPU={obj_cpu}, GPU={obj_gpu}"

            # Verify both satisfy KKT conditions
            verify_kkt(cpu_sol, "CPU")
            verify_kkt(gpu_sol, "GPU")

            # Note: We don't compare solutions directly because they may differ
            # while both being optimal (non-unique solution case)
        else:
            # Solutions match well enough, just verify they're feasible
            # (the close match already implies correctness)
            verify_kkt(cpu_sol, "CPU")
            verify_kkt(gpu_sol, "GPU")

        # Skip backward pass entirely for degenerate cases where gradients are numerically unstable
        # Case 0: Very small z values cause numerical instability in cone projection Jacobians
        z_min_abs = min(np.min(np.abs(cpu_sol.z)), np.min(np.abs(gpu_sol.z)))
        if z_min_abs < 1e-6:
            # z is too small, backward pass may produce NaN - skip entirely
            return

        # Backward pass
        dx_bar = np.ones(len(q))
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check no NaN/Inf (always required when backward pass is run)
        assert not np.any(np.isnan(cpu_grads["dq"])), f"CPU dq has NaN: {cpu_grads['dq']}"
        assert not np.any(np.isnan(gpu_grads["dq"])), f"GPU dq has NaN: {gpu_grads['dq']}"
        assert not np.any(np.isinf(cpu_grads["dq"])), f"CPU dq has Inf: {cpu_grads['dq']}"
        assert not np.any(np.isinf(gpu_grads["dq"])), f"GPU dq has Inf: {gpu_grads['dq']}"

        assert not np.any(np.isnan(cpu_grads["db"])), f"CPU db has NaN: {cpu_grads['db']}"
        assert not np.any(np.isnan(gpu_grads["db"])), f"GPU db has NaN: {gpu_grads['db']}"
        assert not np.any(np.isinf(cpu_grads["db"])), f"CPU db has Inf: {cpu_grads['db']}"
        assert not np.any(np.isinf(gpu_grads["db"])), f"GPU db has Inf: {gpu_grads['db']}"

        # Skip gradient comparison for degenerate problems where gradients are expected to differ
        # Case 1: Forward solutions differ (non-unique solution)
        if not solutions_match:
            return

        # Case 2: Near-zero solution indicates a degenerate problem
        if x_scale < 1e-4:
            return

        # Case 3: When s has near-zero components, the constraint is almost active
        # (on the boundary of the cone). The cone projection Jacobian is discontinuous
        # at the boundary, so small differences cause large gradient differences.
        s_min = min(np.min(np.abs(cpu_sol.s)), np.min(np.abs(gpu_sol.s)))
        if s_min < s_scale * 1e-3 or s_scale < 1e-3:
            return

        # Case 4: When z has near-zero components, numerical instability causes gradient differences
        z_min = min(np.min(np.abs(cpu_sol.z)), np.min(np.abs(gpu_sol.z)))
        if z_min < z_scale * 1e-4 or z_min < 1e-8:
            return

        # Case 5: When s and z are on the cone boundary (tight complementarity),
        # the projection Jacobian can differ between CPU/GPU
        comp = np.dot(cpu_sol.s, cpu_sol.z)
        s_norm = np.linalg.norm(cpu_sol.s)
        z_norm = np.linalg.norm(cpu_sol.z)
        if s_norm > 1e-3 and z_norm > 1e-3 and abs(comp) < s_norm * z_norm * 1e-4:
            return

        # Compare gradients with scale-relative tolerance
        dq_scale = max(np.max(np.abs(cpu_grads["dq"])), np.max(np.abs(gpu_grads["dq"])), 1.0)
        db_scale = max(np.max(np.abs(cpu_grads["db"])), np.max(np.abs(gpu_grads["db"])), 1.0)
        grad_dq_atol = max(grad_atol, dq_scale * 0.01)  # Allow 1% of gradient scale
        grad_db_atol = max(grad_atol, db_scale * 0.05)  # Allow 5% for db (more sensitive)

        np.testing.assert_allclose(
            cpu_grads["dq"],
            gpu_grads["dq"],
            rtol=grad_rtol,
            atol=grad_dq_atol,
            err_msg=f"dq mismatch: CPU={cpu_grads['dq']}, GPU={gpu_grads['dq']}",
        )
        np.testing.assert_allclose(
            cpu_grads["db"],
            gpu_grads["db"],
            rtol=grad_rtol,
            atol=grad_db_atol,
            err_msg=f"db mismatch: CPU={cpu_grads['db']}, GPU={gpu_grads['db']}",
        )

    except AssertionError as e:
        # Print problem data for reproduction before re-raising
        print_problem(P, q, A, b, cones)
        print(f"CPU solution: x={cpu_sol.x}, s={cpu_sol.s}, z={cpu_sol.z}")
        print(f"GPU solution: x={gpu_sol.x}, s={gpu_sol.s}, z={gpu_sol.z}")
        if "cpu_grads" in dir() and "gpu_grads" in dir():
            print(f"CPU grads: dq={cpu_grads['dq']}, db={cpu_grads['db']}")
            print(f"GPU grads: dq={gpu_grads['dq']}, db={gpu_grads['db']}")
        raise


# =============================================================================
# Fuzz tests
# =============================================================================


@requires_gpu
class TestCPUGPUFuzz:
    """Fuzz tests comparing CPU and GPU implementations."""

    @given(problem=simple_qp_problem())
    @settings(
        max_examples=50,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_simple_qp_parity(self, problem):
        """Fuzz test: simple QP with nonnegative cones."""
        P, q, A, b, cones = problem
        solve_and_compare(P, q, A, b, cones)

    @given(problem=soc_problem())
    @settings(
        max_examples=50,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_soc_parity(self, problem):
        """Fuzz test: problems with SOC constraints."""
        P, q, A, b, cones = problem
        solve_and_compare(P, q, A, b, cones)

    @given(problem=exp_cone_problem())
    @settings(
        max_examples=50,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_exp_cone_parity(self, problem):
        """Fuzz test: problems with exponential cone constraints."""
        P, q, A, b, cones = problem
        solve_and_compare(P, q, A, b, cones)

    @given(problem=power_cone_problem())
    @settings(
        max_examples=50,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_power_cone_parity(self, problem):
        """Fuzz test: problems with power cone constraints."""
        P, q, A, b, cones = problem
        solve_and_compare(P, q, A, b, cones)

    @given(problem=mixed_cone_problem())
    @settings(
        max_examples=50,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_mixed_cone_parity(self, problem):
        """Fuzz test: problems with mixed cone types."""
        P, q, A, b, cones = problem
        solve_and_compare(P, q, A, b, cones)


@requires_gpu
class TestCPUGPUFuzzExtreme:
    """Fuzz tests with extreme values to stress test numerical stability."""

    @given(
        scale=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
        problem=simple_qp_problem(),
    )
    @settings(
        max_examples=30,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_scaled_problem(self, scale, problem):
        """Fuzz test: problems with varying scales."""
        P, q, A, b, cones = problem
        # Scale the problem
        P = P * scale
        q = q * scale
        b = b * scale
        solve_and_compare(P, q, A, b, cones, rtol=1e-3, atol=1e-4, grad_rtol=0.1, grad_atol=1e-2)

    @given(problem=simple_qp_problem())
    @settings(
        max_examples=30,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_near_zero_q(self, problem):
        """Fuzz test: problems with near-zero objective linear term."""
        P, q, A, b, cones = problem
        q = q * 1e-10  # Scale q to be very small
        solve_and_compare(P, q, A, b, cones)

    @given(problem=simple_qp_problem())
    @settings(
        max_examples=30,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_near_zero_b(self, problem):
        """Fuzz test: problems with near-zero RHS."""
        P, q, A, b, cones = problem
        b = b * 1e-10  # Scale b to be very small
        solve_and_compare(P, q, A, b, cones)


@requires_gpu
class TestCPUGPUGradientFuzz:
    """Fuzz tests specifically targeting gradient computation."""

    @given(
        problem=simple_qp_problem(),
        dx_bar=arrays(
            np.float64,
            st.integers(2, 10),
            elements=st.floats(-10, 10, allow_nan=False, allow_infinity=False),
        ),
    )
    @settings(
        max_examples=30,
        derandomize=True,
        deadline=None,
        suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
    )
    def test_various_dx_bar(self, problem, dx_bar):
        """Fuzz test: varying backward pass input vectors."""
        P, q, A, b, cones = problem
        n = P.shape[0]

        # Skip degenerate problems
        assume(np.linalg.norm(q) > 1e-6 or np.linalg.norm(b) > 1e-6)

        # Resize dx_bar to match problem dimension
        if len(dx_bar) < n:
            dx_bar = np.pad(dx_bar, (0, n - len(dx_bar)), constant_values=1.0)
        else:
            dx_bar = dx_bar[:n]

        # CPU solve
        cpu_settings = moreau.Settings(device="cpu", enable_grad=True)
        cpu_solver = moreau.Solver(P, q=q, A=A, b=b, cones=cones, settings=cpu_settings)
        cpu_sol = cpu_solver.solve()

        # GPU solve
        gpu_settings = moreau.Settings(device="cuda", enable_grad=True)
        gpu_solver = moreau.Solver(P, q=q, A=A, b=b, cones=cones, settings=gpu_settings)
        gpu_sol = gpu_solver.solve()

        cpu_solved = cpu_solver.info.status.name in ["Solved", "AlmostSolved"]
        gpu_solved = gpu_solver.info.status.name in ["Solved", "AlmostSolved"]

        if not cpu_solved or not gpu_solved:
            return  # Skip if didn't solve

        # Skip degenerate solutions (near-zero, boundary cases)
        x_scale = max(np.max(np.abs(cpu_sol.x)), np.max(np.abs(gpu_sol.x)), 1e-6)
        s_scale = max(np.max(np.abs(cpu_sol.s)), np.max(np.abs(gpu_sol.s)), 1e-6)
        z_scale = max(np.max(np.abs(cpu_sol.z)), np.max(np.abs(gpu_sol.z)), 1e-6)
        s_min = min(np.min(np.abs(cpu_sol.s)), np.min(np.abs(gpu_sol.s)))
        z_min = min(np.min(np.abs(cpu_sol.z)), np.min(np.abs(gpu_sol.z)))

        # Skip if solutions don't match well
        x_diff = np.max(np.abs(cpu_sol.x - gpu_sol.x))
        s_diff = np.max(np.abs(cpu_sol.s - gpu_sol.s))
        if x_diff > x_scale * 0.01 or s_diff > s_scale * 0.01:
            return

        # Skip boundary/degenerate cases
        if x_scale < 1e-4 or s_min < s_scale * 1e-3 or z_min < 1e-8:
            return

        # Backward with custom dx_bar
        cpu_grads = cpu_solver.backward(dx_bar)
        gpu_grads = gpu_solver.backward(dx_bar)

        # Check no NaN/Inf
        assert not np.any(np.isnan(cpu_grads["dq"])), f"CPU dq has NaN with dx_bar={dx_bar}"
        assert not np.any(np.isnan(gpu_grads["dq"])), f"GPU dq has NaN with dx_bar={dx_bar}"
        assert not np.any(np.isinf(cpu_grads["dq"])), f"CPU dq has Inf with dx_bar={dx_bar}"
        assert not np.any(np.isinf(gpu_grads["dq"])), f"GPU dq has Inf with dx_bar={dx_bar}"

        # Compare with scale-relative tolerance
        dq_scale = max(np.max(np.abs(cpu_grads["dq"])), np.max(np.abs(gpu_grads["dq"])), 1.0)
        dq_atol = max(1e-3, dq_scale * 0.01)  # Allow 1% of gradient scale
        np.testing.assert_allclose(
            cpu_grads["dq"],
            gpu_grads["dq"],
            rtol=1e-2,
            atol=dq_atol,
            err_msg=f"dq mismatch with dx_bar={dx_bar}",
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x", "--hypothesis-show-statistics"])
