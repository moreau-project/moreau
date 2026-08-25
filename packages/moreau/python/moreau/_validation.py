"""Input validation helpers for moreau (private).

Pure validation routines used by Solver / CompiledSolver. No solver logic
lives here — these functions only check shapes, dtypes, and structural
consistency, raising ValueError on mismatch.
"""

from typing import Optional

import numpy as np
from scipy import sparse


def _to_csr(matrix):
    """Convert matrix to CSR format, extracting row_offsets, col_indices, values."""
    if sparse.issparse(matrix):
        csr = matrix.tocsr()
    else:
        csr = sparse.csr_array(matrix)
    return (
        np.asarray(csr.indptr, dtype=np.int64),
        np.asarray(csr.indices, dtype=np.int64),
        np.asarray(csr.data, dtype=np.float64),
    )


def _validate_problem_dimensions(P, q, A, b, cones):
    """Validate that problem dimensions are consistent.

    Args:
        P: Quadratic objective matrix (n x n)
        q: Linear objective vector (n,)
        A: Constraint matrix (m x n)
        b: Constraint RHS vector (m,)
        cones: Cone specification

    Raises:
        ValueError: If dimensions are inconsistent
    """
    # Get dimensions
    q_arr = np.asarray(q)
    b_arr = np.asarray(b)
    n = q_arr.shape[-1] if q_arr.ndim > 0 else 1
    m = b_arr.shape[-1] if b_arr.ndim > 0 else 1

    # Validate q is 1D
    if q_arr.ndim != 1:
        raise ValueError(f"q must be a 1D array, got shape {q_arr.shape}")

    # Validate b is 1D
    if b_arr.ndim != 1:
        raise ValueError(f"b must be a 1D array, got shape {b_arr.shape}")

    # Validate P dimensions
    if sparse.issparse(P):
        P_shape = P.shape
    else:
        P_arr = np.asarray(P)
        P_shape = P_arr.shape

    if len(P_shape) != 2:
        raise ValueError(f"P must be a 2D matrix, got shape {P_shape}")
    if P_shape[0] != P_shape[1]:
        raise ValueError(f"P must be square, got shape {P_shape}")
    if P_shape[0] != n:
        raise ValueError(
            f"P has shape {P_shape} but q has length {n}. "
            f"P must be (n x n) where n = len(q) = {n}"
        )

    # Validate A dimensions
    if sparse.issparse(A):
        A_shape = A.shape
    else:
        A_arr = np.asarray(A)
        A_shape = A_arr.shape

    if len(A_shape) != 2:
        raise ValueError(f"A must be a 2D matrix, got shape {A_shape}")
    if A_shape[1] != n:
        raise ValueError(
            f"A has shape {A_shape} but problem has n={n} variables (from q). "
            f"A must have {n} columns."
        )
    if A_shape[0] != m:
        raise ValueError(
            f"A has shape {A_shape} but b has length {m}. " f"A must have {m} rows to match b."
        )

    # Validate cone dimensions sum to m
    cone_total = cones.total_constraints()
    if cone_total != m:
        raise ValueError(
            f"Cone dimensions sum to {cone_total} but b has length {m}. "
            f"Total cone constraints must equal m = len(b) = {m}."
        )

    # Validate direct-x cone indices lie in [0, n)
    if hasattr(cones, "validate_x_cone_indices"):
        cones.validate_x_cone_indices(n)


def _validate_csr_structure(
    n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones
):
    """Validate CSR structure is consistent with problem dimensions.

    Args:
        n: Number of primal variables
        m: Number of constraints
        P_row_offsets: CSR row pointers for P (length n+1)
        P_col_indices: CSR column indices for P
        A_row_offsets: CSR row pointers for A (length m+1)
        A_col_indices: CSR column indices for A
        cones: Cone specification

    Raises:
        ValueError: If CSR structure is invalid
    """
    P_ro = np.asarray(P_row_offsets)
    P_ci = np.asarray(P_col_indices)
    A_ro = np.asarray(A_row_offsets)
    A_ci = np.asarray(A_col_indices)

    # Validate P_row_offsets length
    if len(P_ro) != n + 1:
        raise ValueError(
            f"P_row_offsets has length {len(P_ro)} but must have length n+1 = {n + 1} "
            f"for a matrix with n={n} rows."
        )

    # Validate A_row_offsets length
    if len(A_ro) != m + 1:
        raise ValueError(
            f"A_row_offsets has length {len(A_ro)} but must have length m+1 = {m + 1} "
            f"for a matrix with m={m} rows."
        )

    # Validate P_row_offsets starts at 0 and is monotonic
    if P_ro[0] != 0:
        raise ValueError(f"P_row_offsets must start at 0, got {P_ro[0]}")
    if not np.all(np.diff(P_ro) >= 0):
        raise ValueError("P_row_offsets must be monotonically non-decreasing")

    # Validate A_row_offsets starts at 0 and is monotonic
    if A_ro[0] != 0:
        raise ValueError(f"A_row_offsets must start at 0, got {A_ro[0]}")
    if not np.all(np.diff(A_ro) >= 0):
        raise ValueError("A_row_offsets must be monotonically non-decreasing")

    # Validate P_col_indices length matches P_row_offsets[-1] (nnz_P)
    nnz_P = int(P_ro[-1])
    if len(P_ci) != nnz_P:
        raise ValueError(
            f"P_col_indices has length {len(P_ci)} but P_row_offsets indicates "
            f"nnz_P = {nnz_P} non-zeros."
        )

    # Validate A_col_indices length matches A_row_offsets[-1] (nnz_A)
    nnz_A = int(A_ro[-1])
    if len(A_ci) != nnz_A:
        raise ValueError(
            f"A_col_indices has length {len(A_ci)} but A_row_offsets indicates "
            f"nnz_A = {nnz_A} non-zeros."
        )

    # Validate P column indices are in range [0, n)
    if len(P_ci) > 0:
        if np.any(P_ci < 0) or np.any(P_ci >= n):
            bad_idx = P_ci[(P_ci < 0) | (P_ci >= n)]
            raise ValueError(
                f"P_col_indices contains invalid column index(es): {bad_idx[:5].tolist()}... "
                f"Valid range is [0, {n - 1}]."
            )

    # Validate A column indices are in range [0, n)
    if len(A_ci) > 0:
        if np.any(A_ci < 0) or np.any(A_ci >= n):
            bad_idx = A_ci[(A_ci < 0) | (A_ci >= n)]
            raise ValueError(
                f"A_col_indices contains invalid column index(es): {bad_idx[:5].tolist()}... "
                f"Valid range is [0, {n - 1}]."
            )

    # Validate direct-x cone indices lie in [0, n)
    if hasattr(cones, "validate_x_cone_indices"):
        cones.validate_x_cone_indices(n)

    # Validate cone dimensions sum to m
    cone_total = cones.total_constraints()
    if cone_total != m:
        raise ValueError(
            f"Cone dimensions sum to {cone_total} but m={m}. "
            f"Total cone constraints must equal m."
        )


def _validate_smoothed_diff_cones(settings, cones):
    """Validate that smoothed differentiation is only used with supported cones.

    Smoothed diff requires zero + nonneg + SOC cones only (no exp or power).
    Raises ValueError if diff_method='smoothed' with unsupported cones.
    """
    if settings is None or not hasattr(settings, "ipm_settings"):
        return
    ipm = settings.ipm_settings
    if ipm is None or getattr(ipm, "diff_method", "auto") != "smoothed":
        return

    unsupported = []
    if cones.num_exp_cones > 0:
        unsupported.append("exponential")
    if len(cones.power_alphas) > 0:
        unsupported.append("power")
    if hasattr(cones, "gen_power_cone_params") and len(cones.gen_power_cone_params) > 0:
        unsupported.append("generalized power")

    if unsupported:
        raise ValueError(
            f"diff_method='smoothed' only supports zero + nonneg + SOC cones, "
            f"but problem has {', '.join(unsupported)} cone(s). "
            f"Use diff_method='exact' or 'auto' instead."
        )


def _validate_setup_values(batch_size, nnz_P, nnz_A, P_values, A_values):
    """Validate P_values and A_values dimensions for setup().

    Args:
        batch_size: Expected batch size
        nnz_P: Number of non-zeros in P
        nnz_A: Number of non-zeros in A
        P_values: P matrix values array
        A_values: A matrix values array

    Raises:
        ValueError: If dimensions don't match
    """
    P_vals = np.asarray(P_values)
    A_vals = np.asarray(A_values)

    # Handle 1D (shared) vs 2D (per-batch) P_values
    if P_vals.ndim == 1:
        if P_vals.shape[0] != nnz_P:
            raise ValueError(
                f"P_values has length {P_vals.shape[0]} but expected nnz_P = {nnz_P}. "
                f"For shared values, shape should be ({nnz_P},)."
            )
    elif P_vals.ndim == 2:
        if P_vals.shape[0] != batch_size:
            raise ValueError(
                f"P_values has batch dimension {P_vals.shape[0]} but solver has "
                f"batch_size = {batch_size}."
            )
        if P_vals.shape[1] != nnz_P:
            raise ValueError(
                f"P_values has {P_vals.shape[1]} values per problem but expected "
                f"nnz_P = {nnz_P}. Shape should be ({batch_size}, {nnz_P})."
            )
    else:
        raise ValueError(
            f"P_values must be 1D (shared) or 2D (per-batch), got shape {P_vals.shape}"
        )

    # Handle 1D (shared) vs 2D (per-batch) A_values
    if A_vals.ndim == 1:
        if A_vals.shape[0] != nnz_A:
            raise ValueError(
                f"A_values has length {A_vals.shape[0]} but expected nnz_A = {nnz_A}. "
                f"For shared values, shape should be ({nnz_A},)."
            )
    elif A_vals.ndim == 2:
        if A_vals.shape[0] != batch_size:
            raise ValueError(
                f"A_values has batch dimension {A_vals.shape[0]} but solver has "
                f"batch_size = {batch_size}."
            )
        if A_vals.shape[1] != nnz_A:
            raise ValueError(
                f"A_values has {A_vals.shape[1]} values per problem but expected "
                f"nnz_A = {nnz_A}. Shape should be ({batch_size}, {nnz_A})."
            )
    else:
        raise ValueError(
            f"A_values must be 1D (shared) or 2D (per-batch), got shape {A_vals.shape}"
        )


def _validate_solve_inputs(batch_size, n, m, qs, bs):
    """Validate qs and bs dimensions for solve().

    Args:
        batch_size: Expected batch size
        n: Number of primal variables
        m: Number of constraints
        qs: q vectors (linear cost)
        bs: b vectors (constraint RHS)

    Raises:
        ValueError: If dimensions don't match
    """
    qs_arr = np.asarray(qs)
    bs_arr = np.asarray(bs)

    # Validate qs shape
    if qs_arr.ndim == 1:
        if batch_size != 1:
            raise ValueError(
                f"qs is 1D but solver has batch_size = {batch_size}. "
                f"Expected shape ({batch_size}, {n})."
            )
        if qs_arr.shape[0] != n:
            raise ValueError(f"qs has length {qs_arr.shape[0]} but problem has n = {n} variables.")
    elif qs_arr.ndim == 2:
        if qs_arr.shape[0] != batch_size:
            raise ValueError(
                f"qs has batch dimension {qs_arr.shape[0]} but solver has "
                f"batch_size = {batch_size}."
            )
        if qs_arr.shape[1] != n:
            raise ValueError(
                f"qs has {qs_arr.shape[1]} elements per problem but expected "
                f"n = {n}. Shape should be ({batch_size}, {n})."
            )
    else:
        raise ValueError(f"qs must be 1D (batch_size=1) or 2D, got shape {qs_arr.shape}")

    # Validate bs shape
    if bs_arr.ndim == 1:
        if batch_size != 1:
            raise ValueError(
                f"bs is 1D but solver has batch_size = {batch_size}. "
                f"Expected shape ({batch_size}, {m})."
            )
        if bs_arr.shape[0] != m:
            raise ValueError(
                f"bs has length {bs_arr.shape[0]} but problem has m = {m} constraints."
            )
    elif bs_arr.ndim == 2:
        if bs_arr.shape[0] != batch_size:
            raise ValueError(
                f"bs has batch dimension {bs_arr.shape[0]} but solver has "
                f"batch_size = {batch_size}."
            )
        if bs_arr.shape[1] != m:
            raise ValueError(
                f"bs has {bs_arr.shape[1]} elements per problem but expected "
                f"m = {m}. Shape should be ({batch_size}, {m})."
            )
    else:
        raise ValueError(f"bs must be 1D (batch_size=1) or 2D, got shape {bs_arr.shape}")


def _max_abs_asymmetry(P) -> float:
    """Return max|P - P^T| as a scalar; handles sparse or dense P."""
    if sparse.issparse(P):
        P_csr = P.tocsr()
        diff = P_csr - P_csr.T
        return float(abs(diff).max()) if diff.nnz > 0 else 0.0
    P_arr = np.asarray(P)
    return float(np.max(np.abs(P_arr - P_arr.T)))


def _validate_P_symmetry(P, tol: float = 1e-10, *, hint: Optional[str] = None):
    """Validate that P matrix is numerically symmetric.

    P must be provided as full symmetric matrix (not just upper or lower triangle).

    Args:
        P: The P matrix (scipy sparse or numpy array)
        tol: Tolerance for symmetry check (default 1e-10)
        hint: Optional remediation hint appended to the error message
            (defaults to suggesting `P = (P + P.T) / 2`).

    Raises:
        ValueError: If P is not numerically symmetric
    """
    max_diff = _max_abs_asymmetry(P)
    if max_diff > tol:
        suffix = hint or (
            "Use P = (P + P.T) / 2 to symmetrize, or provide the full symmetric matrix."
        )
        raise ValueError(
            f"P matrix is not numerically symmetric (max|P - P^T| = {max_diff:.2e}). "
            f"P must be full symmetric, not upper or lower triangle only. "
            f"{suffix}"
        )


def _validate_P_sparsity_pattern_symmetric(n: int, P_row_offsets, P_col_indices):
    """Validate that the P sparsity pattern is symmetric.

    For full symmetric P matrix requirement, the sparsity pattern must include
    both (i,j) and (j,i) for every off-diagonal entry.

    Args:
        n: Number of columns/rows in P
        P_row_offsets: CSR row pointers (length n+1)
        P_col_indices: CSR column indices

    Raises:
        ValueError: If sparsity pattern is not symmetric
    """
    P_ro = np.asarray(P_row_offsets)
    P_ci = np.asarray(P_col_indices)

    # Build set of (row, col) pairs from CSR structure
    entries = set()
    for row in range(n):
        start, end = P_ro[row], P_ro[row + 1]
        for idx in range(start, end):
            col = P_ci[idx]
            entries.add((row, col))

    # Check each off-diagonal entry has its transpose
    missing = []
    for row, col in entries:
        if row != col and (col, row) not in entries:
            missing.append((row, col))
            if len(missing) >= 5:  # Limit number of examples
                break

    if missing:
        examples = ", ".join(f"({r},{c})" for r, c in missing[:3])
        raise ValueError(
            f"P sparsity pattern is not symmetric. "
            f"Found entries without their transpose: {examples}{'...' if len(missing) > 3 else ''}. "
            f"P must be a full symmetric matrix (both upper and lower triangles). "
            f"Ensure both P[i,j] and P[j,i] positions are included in the sparsity pattern."
        )


def _validate_P_values_symmetry(n: int, P_row_offsets, P_col_indices, P_values, tol: float = 1e-10):
    """Validate P matrix symmetry from CSR components.

    P must be full symmetric (not upper/lower triangle only).
    This validates a single problem's P_values.

    Args:
        n: Number of columns/rows in P
        P_row_offsets: CSR row pointers
        P_col_indices: CSR column indices
        P_values: CSR values (for one problem, shape (nnz,))
        tol: Tolerance for symmetry check

    Raises:
        ValueError: If P is not numerically symmetric
    """
    P_csr = sparse.csr_array((P_values, P_col_indices, P_row_offsets), shape=(n, n))
    _validate_P_symmetry(
        P_csr,
        tol,
        hint="Ensure P_values contains both P[i,j] and P[j,i] entries.",
    )
