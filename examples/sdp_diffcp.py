"""
GPU SDP benchmark based on diffcp's SDP example.

Solves:  minimize    trace(C @ X)
         subject to  trace(A_i @ X) == b_i,  i = 1..p
                     X >= 0 (PSD)

In moreau's standard form (min 1/2 x'Px + q'x, Ax + s = b, s ∈ K):
  - Decision variable x = svec(X), dimension n(n+1)/2
  - P = 0 (linear objective)
  - q = svec(C)
  - A has p rows: A[i,:] = svec(A_i)' (equality constraints via zero cone)
  - Plus identity block for PSD cone: [A_eq; I] x + s = [b; 0]
  - Cones: p zero (equalities) + PSD(n)

Reference: "Differentiating Through a Cone Program" (Agrawal et al., 2019)
           https://github.com/cvxgrp/diffcp/blob/master/examples/sdp.py
"""

import numpy as np
import time
from scipy import sparse
import argparse


def mat_to_svec(M):
    """Convert symmetric matrix to svec (scaled upper triangle, column-major)."""
    n = M.shape[0]
    svec_dim = n * (n + 1) // 2
    v = np.zeros(svec_dim)
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            if i == j:
                v[idx] = M[i, j]
            else:
                v[idx] = M[i, j] * np.sqrt(2)
            idx += 1
    return v


def svec_to_mat(v, n):
    """Convert svec back to symmetric matrix."""
    M = np.zeros((n, n))
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            if i == j:
                M[i, j] = v[idx]
            else:
                M[i, j] = v[idx] / np.sqrt(2)
                M[j, i] = M[i, j]
            idx += 1
    return M


def randn_symm(n, rng):
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2


def randn_psd(n, rng):
    A = rng.standard_normal((n, n)) / 10
    return A @ A.T


def build_sdp_problem(n, p, seed=0):
    """Build SDP problem data in moreau format.

    Returns: (n_var, m_con, P_ro, P_ci, P_vals, A_ro, A_ci, A_vals, q, b, cones_kwargs)
    """
    rng = np.random.default_rng(seed)

    # Generate problem data
    C = randn_psd(n, rng)
    As = [randn_symm(n, rng) for _ in range(p)]
    Bs = rng.standard_normal(p)

    svec_dim = n * (n + 1) // 2

    # q = svec(C)
    q = mat_to_svec(C)

    # Build A matrix (CSR):
    # Top p rows: equality constraints trace(A_i @ X) = b_i
    #   => A_eq[i, :] = svec(A_i)' (but with sqrt(2) handled by svec)
    # Bottom svec_dim rows: identity (PSD cone)
    m_eq = p
    m_psd = svec_dim
    m_total = m_eq + m_psd
    n_var = svec_dim

    # Build equality constraint rows
    A_eq = np.zeros((m_eq, n_var))
    for i in range(p):
        A_eq[i, :] = mat_to_svec(As[i])

    # Full A = [A_eq; -I]  (negative identity so s = x, and s ∈ PSD means x ∈ PSD)
    A_full = np.vstack([A_eq, -np.eye(n_var)])

    # b = [Bs; 0]
    b = np.concatenate([Bs, np.zeros(m_psd)])

    # Convert to CSR
    A_csr = sparse.csr_matrix(A_full)

    # P = εI (small regularization for GPU numerical stability)
    # Without this, the LP (P=0) case triggers a GPU initialization bug
    # where the initial point appears converged at iteration 0.
    reg = 1e-6
    P_diag = sparse.eye(n_var, format="csr") * reg
    P_ro = np.asarray(P_diag.indptr, dtype=np.int64)
    P_ci = np.asarray(P_diag.indices, dtype=np.int64)
    P_vals = np.asarray(P_diag.data, dtype=np.float64)

    A_ro = np.asarray(A_csr.indptr, dtype=np.int64)
    A_ci = np.asarray(A_csr.indices, dtype=np.int64)
    A_vals = np.asarray(A_csr.data, dtype=np.float64)

    cones_kwargs = dict(num_zero_cones=m_eq, psd_dims=[n])

    return n_var, m_total, P_ro, P_ci, P_vals, A_ro, A_ci, A_vals, q, b, cones_kwargs


def solve_moreau(n, p, device="auto", batch_size=1, enable_grad=False, verbose=True, seed=0):
    """Solve the SDP problem using moreau."""
    import moreau

    n_var, m, P_ro, P_ci, P_vals, A_ro, A_ci, A_vals, q, b, cones_kwargs = build_sdp_problem(
        n, p, seed=seed
    )

    svec_dim = n * (n + 1) // 2
    print(f"\n{'='*60}")
    print(f"SDP Problem: n={n} (matrix dim), p={p} (constraints)")
    print(f"  Variables (svec): {svec_dim}")
    print(f"  Constraints: {m} ({p} equality + {svec_dim} PSD)")
    print(f"  A nnz: {len(A_vals)}")
    print(f"  Device: {device}, batch_size={batch_size}, grad={enable_grad}")
    print(f"{'='*60}")

    cones = moreau.Cones(**cones_kwargs)

    # cuDSS default max_lu_nnz is 100*nnz which OOMs for dense SDP. Use 5*nnz.
    A_nnz = len(A_vals)
    kkt_nnz = len(P_vals) + A_nnz + n_var + m
    settings = moreau.Settings(
        device=device,
        batch_size=batch_size,
        enable_grad=enable_grad,
        verbose=verbose,
        ipm_settings=moreau.IPMSettings(max_lu_nnz=5 * kkt_nnz),
    )

    # Construction
    t0 = time.perf_counter()
    solver = moreau.CompiledSolver(
        n=n_var,
        m=m,
        P_row_offsets=P_ro,
        P_col_indices=P_ci,
        A_row_offsets=A_ro,
        A_col_indices=A_ci,
        cones=cones,
        settings=settings,
    )
    t_construct = time.perf_counter() - t0

    # Setup
    t0 = time.perf_counter()
    solver.setup(P_values=P_vals, A_values=A_vals)
    t_setup = time.perf_counter() - t0

    # Tile q/b for batch
    if batch_size > 1:
        qs = np.tile(q, (batch_size, 1))
        bs = np.tile(b, (batch_size, 1))
    else:
        qs = q.reshape(1, -1)
        bs = b.reshape(1, -1)

    # Solve
    t0 = time.perf_counter()
    solution = solver.solve(qs=qs, bs=bs)
    t_solve = time.perf_counter() - t0

    info = solver.info
    print(f"\nResults:")
    print(f"  Status: {info.status}")
    print(f"  Objective: {info.obj_val}")
    print(f"  Iterations: {info.iterations}")
    print(f"  Construction: {t_construct:.4f}s")
    print(f"  Setup: {t_setup:.4f}s")
    print(f"  Solve: {t_solve:.4f}s")
    print(f"  Total: {t_construct + t_setup + t_solve:.4f}s")

    # Verify solution is PSD
    x = solution.x[0] if solution.x.ndim > 1 else solution.x
    s = solution.s[0] if solution.s.ndim > 1 else solution.s
    # s_psd is the PSD cone slack (last svec_dim entries of s)
    s_psd = s[p:]
    X_from_s = svec_to_mat(s_psd, n)
    X_from_x = svec_to_mat(x[:svec_dim], n)
    eigvals_s = np.linalg.eigvalsh(X_from_s)
    eigvals_x = np.linalg.eigvalsh(X_from_x)
    print(f"  X (from s) min eig: {eigvals_s[0]:.2e}, max eig: {eigvals_s[-1]:.2e}")
    print(f"  X (from x) min eig: {eigvals_x[0]:.2e}, max eig: {eigvals_x[-1]:.2e}")
    print(f"  |x|: {np.linalg.norm(x):.4e}, |s|: {np.linalg.norm(s):.4e}")

    if enable_grad:
        # Backward pass
        dx = np.random.default_rng(42).standard_normal(solution.x.shape) * 0.1
        t0 = time.perf_counter()
        grads = solver.backward(dx=dx)
        t_backward = time.perf_counter() - t0
        print(f"  Backward: {t_backward:.4f}s")
        if isinstance(grads, dict):
            dq = grads["dq"]
            db = grads["db"]
        else:
            dq = grads.dq
            db = grads.db
        print(f"  |dq|: {np.linalg.norm(dq):.4e}")
        print(f"  |db|: {np.linalg.norm(db):.4e}")

    return solution


def main():
    parser = argparse.ArgumentParser(description="GPU SDP benchmark (diffqcp style)")
    parser.add_argument("--n", type=int, default=50, help="PSD matrix dimension")
    parser.add_argument("--p", type=int, default=25, help="Number of equality constraints")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--grad", action="store_true", help="Enable backward pass")
    parser.add_argument("--quiet", action="store_true", help="Suppress solver output")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    solve_moreau(
        n=args.n,
        p=args.p,
        device=args.device,
        batch_size=args.batch,
        enable_grad=args.grad,
        verbose=not args.quiet,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
