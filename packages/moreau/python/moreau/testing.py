"""Random problem generation utilities for testing.

This module provides functions to generate random feasible conic programs
with specified dimensions, sparsity patterns, and cones.

The key insight is: given s ∈ int(K) and any x, setting b = Ax + s
guarantees that (x, s) is a strictly feasible primal point.

Example:
    >>> from moreau.testing import random_cone_program
    >>> import moreau
    >>>
    >>> # Generate a random QP with SOC and nonneg cones
    >>> cones = moreau.Cones(num_zero_cones=2, num_nonneg_cones=5, so_cone_dims=[3, 3])
    >>> problem = random_cone_program(n=10, cones=cones, seed=42)
    >>>
    >>> # Solve it
    >>> solver = moreau.Solver(problem.P, problem.q, problem.A, problem.b, problem.cones)
    >>> solver.solve()
    >>> assert solver.info.status == moreau.SolverStatus.Solved
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from scipy import sparse

from moreau._types import Cones


@dataclass
class RandomProblem:
    """Container for a randomly generated cone program.

    Attributes:
        P: Quadratic objective matrix (n x n), CSR format, symmetric PSD
        q: Linear objective vector (n,)
        A: Constraint matrix (m x n), CSR format
        b: Constraint RHS (m,)
        cones: Cone specification
        x_feas: A known feasible primal point (n,)
        s_feas: A known feasible slack point in int(K) (m,)
        z_feas: A known feasible dual point in int(K*) (m,), if available
    """

    P: sparse.csr_array
    q: np.ndarray
    A: sparse.csr_array
    b: np.ndarray
    cones: Cones
    x_feas: np.ndarray
    s_feas: np.ndarray
    z_feas: Optional[np.ndarray] = None


def sample_cone_interior(cones: Cones, rng: np.random.Generator) -> np.ndarray:
    """Sample a point strictly inside the cone K.

    For each cone type:
    - Zero cone: s = 0 (the only point)
    - Nonnegative: s > 0 (uniform on [0.1, 1])
    - SOC: s[0] > ||s[1:]|| (s[0] = 1 + ||s[1:]||, s[1:] uniform)
    - Exp: (x, y, z) with y > 0, z > y*exp(x/y) (sample y, x, compute z)
    - Power: (x, y, z) with x, y > 0, |z| < x^alpha * y^(1-alpha)

    Args:
        cones: Cone specification
        rng: NumPy random generator

    Returns:
        Array of shape (m,) in the interior of K
    """
    parts = []

    # Zero cone: s = 0
    if cones.num_zero_cones > 0:
        parts.append(np.zeros(cones.num_zero_cones))

    # Nonnegative cone: s > 0
    if cones.num_nonneg_cones > 0:
        s_nonneg = rng.uniform(0.1, 1.0, size=cones.num_nonneg_cones)
        parts.append(s_nonneg)

    # Second-order cones: s[0] > ||s[1:]||
    for dim in cones.so_cone_dims:
        s_tail = rng.uniform(-0.5, 0.5, size=dim - 1)
        norm_tail = np.linalg.norm(s_tail)
        s_head = norm_tail + rng.uniform(0.1, 0.5)  # Strictly greater
        parts.append(np.concatenate([[s_head], s_tail]))

    # Exponential cones: K_exp = {(x,y,z) : y*exp(x/y) <= z, y > 0}
    # Interior: y > 0, z > y*exp(x/y)
    for _ in range(cones.num_exp_cones):
        y = rng.uniform(0.5, 2.0)
        x = rng.uniform(-1.0, 1.0)
        z_min = y * np.exp(x / y)
        z = z_min + rng.uniform(0.1, 0.5)
        parts.append(np.array([x, y, z]))

    # Power cones: K_pow = {(x,y,z) : x^alpha * y^(1-alpha) >= |z|, x,y >= 0}
    # Interior: x, y > 0, x^alpha * y^(1-alpha) > |z|
    for alpha in cones.power_alphas:
        x = rng.uniform(0.5, 2.0)
        y = rng.uniform(0.5, 2.0)
        bound = (x**alpha) * (y ** (1 - alpha))
        z = rng.uniform(-0.9 * bound, 0.9 * bound)
        parts.append(np.array([x, y, z]))

    # Generalized power cones: K = {(p,w) : prod(p_i^alpha_i) >= ||w||, p_i >= 0}
    # Interior: p_i > 0, prod(p_i^alpha_i) > ||w||
    for alphas, dim2 in cones.gen_power_cone_params:
        dim1 = len(alphas)
        p = rng.uniform(0.5, 2.0, size=dim1)
        bound = np.prod(p ** np.array(alphas))
        w = rng.uniform(-0.5, 0.5, size=dim2)
        w_norm = np.linalg.norm(w)
        if w_norm > 0:
            w = w * (0.9 * bound / w_norm)  # Scale so ||w|| < bound
        parts.append(np.concatenate([p, w]))

    if not parts:
        return np.array([])
    return np.concatenate(parts)


def sample_dual_cone_interior(cones: Cones, rng: np.random.Generator) -> np.ndarray:
    """Sample a point strictly inside the dual cone K*.

    For self-dual cones (nonneg, SOC), K* = K.
    For exp cone, K* = cl{(u,v,w) : -u*exp(v/u-1) <= w, u < 0} ∪ {(0,v,w) : v >= 0, w >= 0}
    For power cone, K*_pow(alpha) = K_pow(1-alpha) scaled.

    Args:
        cones: Cone specification
        rng: NumPy random generator

    Returns:
        Array of shape (m,) in the interior of K*
    """
    parts = []

    # Zero cone dual is R^n (free), sample anything
    if cones.num_zero_cones > 0:
        parts.append(rng.uniform(-1.0, 1.0, size=cones.num_zero_cones))

    # Nonnegative cone is self-dual: z >= 0
    if cones.num_nonneg_cones > 0:
        z_nonneg = rng.uniform(0.1, 1.0, size=cones.num_nonneg_cones)
        parts.append(z_nonneg)

    # SOC is self-dual: z[0] > ||z[1:]||
    for dim in cones.so_cone_dims:
        z_tail = rng.uniform(-0.5, 0.5, size=dim - 1)
        norm_tail = np.linalg.norm(z_tail)
        z_head = norm_tail + rng.uniform(0.1, 0.5)
        parts.append(np.concatenate([[z_head], z_tail]))

    # Dual of exp cone: K*_exp = cl{(u,v,w) : -u*exp(v/u-1) <= w, u < 0}
    # Interior: u < 0, -u*exp(v/u - 1) < w
    for _ in range(cones.num_exp_cones):
        u = rng.uniform(-2.0, -0.5)  # u < 0
        v = rng.uniform(-1.0, 1.0)
        w_min = -u * np.exp(v / u - 1)
        w = w_min + rng.uniform(0.1, 0.5)
        parts.append(np.array([u, v, w]))

    # Dual of power cone K_pow(alpha) is K_pow(1-alpha) with scaling
    # K*_pow(alpha) = {(u,v,w) : (u/alpha)^alpha * (v/(1-alpha))^(1-alpha) >= |w|, u,v >= 0}
    for alpha in cones.power_alphas:
        u = rng.uniform(0.5, 2.0)
        v = rng.uniform(0.5, 2.0)
        bound = ((u / alpha) ** alpha) * ((v / (1 - alpha)) ** (1 - alpha))
        w = rng.uniform(-0.9 * bound, 0.9 * bound)
        parts.append(np.array([u, v, w]))

    # Dual of generalized power cone:
    # K*_genpow(alpha) = {(u,w) : prod((u_i/alpha_i)^alpha_i) >= ||w||, u_i >= 0}
    for alphas, dim2 in cones.gen_power_cone_params:
        dim1 = len(alphas)
        alphas_arr = np.array(alphas)
        u = rng.uniform(0.5, 2.0, size=dim1)
        bound = np.prod((u / alphas_arr) ** alphas_arr)
        w = rng.uniform(-0.5, 0.5, size=dim2)
        w_norm = np.linalg.norm(w)
        if w_norm > 0:
            w = w * (0.9 * bound / w_norm)
        parts.append(np.concatenate([u, w]))

    if not parts:
        return np.array([])
    return np.concatenate(parts)


def random_sparse_matrix(
    m: int,
    n: int,
    density: float,
    rng: np.random.Generator,
    symmetric: bool = False,
    psd: bool = False,
) -> sparse.csr_array:
    """Generate a random sparse matrix.

    Args:
        m: Number of rows
        n: Number of columns
        density: Fraction of nonzeros (0 to 1)
        rng: NumPy random generator
        symmetric: If True, generate symmetric matrix (requires m == n)
        psd: If True, generate positive semi-definite matrix (implies symmetric)

    Returns:
        CSR sparse matrix (full symmetric storage for symmetric/psd matrices)

    Note:
        For psd=True, generates FULL symmetric matrix (not just upper triangle)
        because the GPU backend requires full matrix storage for SpMV.
    """
    if psd:
        symmetric = True
    if symmetric:
        assert m == n, "Symmetric matrix requires m == n"

    if psd:
        # Generate A, then return A.T @ A + diagonal for strict PSD
        # Use full symmetric storage (not just upper triangle) for GPU compatibility
        nnz = max(1, int(m * n * np.sqrt(density)))  # Approximate
        k = max(1, int(np.sqrt(nnz)))  # Inner dimension
        A_inner = sparse.random(
            m, k, density=min(1.0, density * 2), random_state=rng, format="csr", dtype=np.float64
        )
        A_inner.data[:] = rng.uniform(-1, 1, size=len(A_inner.data))
        result = A_inner @ A_inner.T
        # Add diagonal for strict positive definiteness
        result = result + sparse.eye(m, format="csr") * 0.5
        # Ensure full symmetric storage (result already is symmetric)
        return sparse.csr_array(result)

    if symmetric:
        # Generate upper triangle, then symmetrize to get FULL matrix
        upper = sparse.random(
            m, n, density=density / 2, random_state=rng, format="csr", dtype=np.float64
        )
        upper.data[:] = rng.uniform(-1, 1, size=len(upper.data))
        result = upper + upper.T
        return sparse.csr_array(result)

    # General matrix
    A = sparse.random(m, n, density=density, random_state=rng, format="csr", dtype=np.float64)
    A.data[:] = rng.uniform(-1, 1, size=len(A.data))
    return sparse.csr_array(A)


def random_cone_program(
    n: int,
    cones: Cones,
    density: float = 0.3,
    seed: Optional[int] = None,
) -> RandomProblem:
    """Generate a random feasible cone program.

    Creates a conic optimization problem:
        minimize    (1/2)x'Px + q'x
        subject to  Ax + s = b
                    s ∈ K

    The problem is guaranteed feasible by construction:
    1. Sample x_feas uniformly
    2. Sample s_feas in int(K)
    3. Generate random A
    4. Set b = A @ x_feas + s_feas

    Args:
        n: Number of variables
        cones: Cone specification (determines m via cones.total_constraints())
        density: Sparsity of A and P (0 to 1)
        seed: Random seed for reproducibility

    Returns:
        RandomProblem with all problem data and known feasible point

    Example:
        >>> cones = moreau.Cones(num_zero_cones=2, num_nonneg_cones=5)
        >>> prob = random_cone_program(n=10, cones=cones, seed=42)
        >>> solver = moreau.Solver(prob.P, prob.q, prob.A, prob.b, prob.cones)
        >>> solver.solve()
        >>> assert solver.info.status == moreau.SolverStatus.Solved
    """
    rng = np.random.default_rng(seed)
    m = cones.total_constraints()

    if m == 0:
        raise ValueError("cones.total_constraints() must be > 0")

    # Generate random PSD matrix P
    P = random_sparse_matrix(n, n, density, rng, psd=True)

    # Generate random linear objective
    q = rng.uniform(-1, 1, size=n)

    # Generate random constraint matrix A
    # Ensure each row has at least one nonzero for well-posedness
    A = random_sparse_matrix(m, n, density, rng)
    # Add diagonal-ish entries to ensure no empty rows
    for i in range(m):
        if A[i, :].nnz == 0:
            j = rng.integers(0, n)
            A[i, j] = rng.uniform(0.5, 1.5)

    # Sample feasible point
    x_feas = rng.uniform(-1, 1, size=n)
    s_feas = sample_cone_interior(cones, rng)

    # Set b to make (x_feas, s_feas) feasible
    b = A @ x_feas + s_feas

    # Optionally generate dual feasible point
    z_feas = sample_dual_cone_interior(cones, rng)

    return RandomProblem(
        P=P,
        q=q,
        A=A,
        b=b,
        cones=cones,
        x_feas=x_feas,
        s_feas=s_feas,
        z_feas=z_feas,
    )


def random_cone_program_with_pattern(
    n: int,
    cones: Cones,
    P_row_offsets: np.ndarray,
    P_col_indices: np.ndarray,
    A_row_offsets: np.ndarray,
    A_col_indices: np.ndarray,
    seed: Optional[int] = None,
) -> RandomProblem:
    """Generate a random feasible cone program with specified sparsity pattern.

    Like random_cone_program, but uses the given CSR sparsity patterns for P and A
    instead of generating random patterns.

    Args:
        n: Number of variables
        cones: Cone specification
        P_row_offsets: CSR row pointers for P (length n+1)
        P_col_indices: CSR column indices for P
        A_row_offsets: CSR row pointers for A (length m+1)
        A_col_indices: CSR column indices for A
        seed: Random seed for reproducibility

    Returns:
        RandomProblem with specified sparsity pattern
    """
    rng = np.random.default_rng(seed)
    m = cones.total_constraints()

    if m == 0:
        raise ValueError("cones.total_constraints() must be > 0")

    # Validate patterns
    P_row_offsets = np.asarray(P_row_offsets, dtype=np.int64)
    P_col_indices = np.asarray(P_col_indices, dtype=np.int64)
    A_row_offsets = np.asarray(A_row_offsets, dtype=np.int64)
    A_col_indices = np.asarray(A_col_indices, dtype=np.int64)

    assert len(P_row_offsets) == n + 1, f"P_row_offsets length {len(P_row_offsets)} != n+1={n+1}"
    assert len(A_row_offsets) == m + 1, f"A_row_offsets length {len(A_row_offsets)} != m+1={m+1}"

    nnz_P = int(P_row_offsets[-1])
    nnz_A = int(A_row_offsets[-1])

    # Generate symmetric PSD P with the given pattern
    # Strategy: build symmetric values first, then make diagonally dominant
    P_data = np.zeros(nnz_P, dtype=np.float64)

    # Build a mapping from (i,j) -> index in data array
    ij_to_idx = {}
    for i in range(n):
        for k in range(P_row_offsets[i], P_row_offsets[i + 1]):
            j = P_col_indices[k]
            ij_to_idx[(i, j)] = k

    # Generate symmetric values: for (i,j) and (j,i), use same value
    processed = set()
    for i in range(n):
        for k in range(P_row_offsets[i], P_row_offsets[i + 1]):
            j = P_col_indices[k]
            if (i, j) in processed:
                continue
            if i == j:
                # Diagonal - will be set later for dominance
                processed.add((i, j))
            else:
                val = rng.uniform(-0.5, 0.5)
                P_data[k] = val
                processed.add((i, j))
                # Set symmetric entry if it exists
                if (j, i) in ij_to_idx:
                    P_data[ij_to_idx[(j, i)]] = val
                    processed.add((j, i))

    # Make diagonal dominant for PSD
    for i in range(n):
        row_start = P_row_offsets[i]
        row_end = P_row_offsets[i + 1]
        diag_idx = None
        off_diag_sum = 0.0

        for k in range(row_start, row_end):
            if P_col_indices[k] == i:
                diag_idx = k
            else:
                off_diag_sum += abs(P_data[k])

        if diag_idx is not None:
            # Make strictly diagonally dominant: P[i,i] > sum of |off-diagonal|
            P_data[diag_idx] = off_diag_sum + rng.uniform(0.5, 1.5)

    P = sparse.csr_array(
        (P_data.astype(np.float64), P_col_indices.copy(), P_row_offsets.copy()), shape=(n, n)
    )

    # Generate random values for A
    A_data = rng.uniform(-1, 1, size=nnz_A)
    A = sparse.csr_array((A_data, A_col_indices, A_row_offsets), shape=(m, n))

    # Generate random q
    q = rng.uniform(-1, 1, size=n)

    # Sample feasible point
    x_feas = rng.uniform(-1, 1, size=n)
    s_feas = sample_cone_interior(cones, rng)

    # Set b for feasibility
    b = A @ x_feas + s_feas

    z_feas = sample_dual_cone_interior(cones, rng)

    return RandomProblem(
        P=P,
        q=q,
        A=A,
        b=b,
        cones=cones,
        x_feas=x_feas,
        s_feas=s_feas,
        z_feas=z_feas,
    )


def random_batch(
    n: int,
    cones: Cones,
    batch_size: int,
    density: float = 0.3,
    seed: Optional[int] = None,
    shared_pattern: bool = True,
) -> Tuple[RandomProblem, List[RandomProblem]]:
    """Generate a batch of random feasible cone programs with shared structure.

    All problems share the same sparsity pattern but have different values.
    This is useful for testing batched solvers.

    Args:
        n: Number of variables
        cones: Cone specification
        batch_size: Number of problems to generate
        density: Sparsity of A and P
        seed: Random seed
        shared_pattern: If True, all problems share P/A sparsity pattern

    Returns:
        Tuple of (first_problem, list_of_all_problems)
        The first problem can be used to extract the shared pattern.
    """
    rng = np.random.default_rng(seed)

    # Generate first problem with random pattern
    first = random_cone_program(n, cones, density, seed=rng.integers(0, 2**31))

    if not shared_pattern:
        problems = [first]
        for _ in range(batch_size - 1):
            prob = random_cone_program(n, cones, density, seed=rng.integers(0, 2**31))
            problems.append(prob)
        return first, problems

    # Generate remaining problems with same pattern
    P_row_offsets = first.P.indptr
    P_col_indices = first.P.indices
    A_row_offsets = first.A.indptr
    A_col_indices = first.A.indices

    problems = [first]
    for _ in range(batch_size - 1):
        prob = random_cone_program_with_pattern(
            n,
            cones,
            P_row_offsets,
            P_col_indices,
            A_row_offsets,
            A_col_indices,
            seed=rng.integers(0, 2**31),
        )
        problems.append(prob)

    return first, problems


__all__ = [
    "RandomProblem",
    "random_cone_program",
    "random_cone_program_with_pattern",
    "random_batch",
    "sample_cone_interior",
    "sample_dual_cone_interior",
]
