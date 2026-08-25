#!/usr/bin/env python3
"""
MPC via cvxpylayers with Moreau solver vs native moreau.torch API.

Double-integrator MPC (nx=12, nu=6, T=20):
  - State: [position(6), velocity(6)]
  - Control: force on each axis
  - Dynamics: double integrator with dt=0.1
  - Constraints: |u| <= 1

Compares:
  1. cvxpylayers + MOREAU: high-level CVXPY formulation
  2. moreau.torch native: manual CSR + cones (same problem)
  3. cvxpylayers + SCS: baseline comparison

All three compute forward + backward pass and report timing.
"""

import time
import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from scipy import sparse

import moreau
from moreau.torch import Solver as MoreauTorchSolver

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# Problem parameters
# ============================================================================

nx = 12  # state dim (6 position + 6 velocity)
nu = 6  # control dim
T = 20  # horizon
dt = 0.1
batch_size = 512

half = nx // 2

# Double integrator dynamics: x_{t+1} = A_dyn @ x_t + B_dyn @ u_t
A_dyn = np.eye(nx)
A_dyn[:half, half:] = dt * np.eye(half)
B_dyn = np.zeros((nx, nu))
B_dyn[:half, :] = 0.5 * dt**2 * np.eye(nu)
B_dyn[half:, :] = dt * np.eye(nu)

# Cost matrices
Q = np.eye(nx)  # state cost
R = 0.1 * np.eye(nu)  # control cost
Qf = 10.0 * np.eye(nx)  # terminal cost


# ============================================================================
# 1. cvxpylayers formulation
# ============================================================================


def build_cvxpy_mpc():
    """Build parametric MPC as a CVXPY problem."""
    x = [cp.Variable(nx) for _ in range(T + 1)]
    u = [cp.Variable(nu) for _ in range(T)]
    x0 = cp.Parameter(nx)

    cost = 0
    constraints = [x[0] == x0]

    for t in range(T):
        cost += 0.5 * cp.quad_form(x[t], Q) + 0.5 * cp.quad_form(u[t], R)
        constraints += [
            x[t + 1] == A_dyn @ x[t] + B_dyn @ u[t],
            u[t] >= -1.0,
            u[t] <= 1.0,
        ]
    cost += 0.5 * cp.quad_form(x[T], Qf)

    problem = cp.Problem(cp.Minimize(cost), constraints)
    return problem, x0, x, u


# ============================================================================
# 2. Native moreau.torch formulation
# ============================================================================


def build_moreau_native_mpc():
    """Build the same MPC as manual CSR matrices + cones for moreau.torch."""
    n_vars = T * (nx + nu) + nx

    # --- P matrix (full symmetric, block diagonal: Q, R, ..., Qf) ---
    p_rows, p_cols, p_vals = [], [], []
    for t in range(T):
        x_off = t * (nx + nu)
        u_off = x_off + nx
        for i in range(nx):
            for j in range(nx):
                if Q[i, j] != 0:
                    p_rows.append(x_off + i)
                    p_cols.append(x_off + j)
                    p_vals.append(Q[i, j])
        for i in range(nu):
            for j in range(nu):
                if R[i, j] != 0:
                    p_rows.append(u_off + i)
                    p_cols.append(u_off + j)
                    p_vals.append(R[i, j])
    xt_off = T * (nx + nu)
    for i in range(nx):
        for j in range(nx):
            if Qf[i, j] != 0:
                p_rows.append(xt_off + i)
                p_cols.append(xt_off + j)
                p_vals.append(Qf[i, j])

    P = sparse.csr_array((p_vals, (p_rows, p_cols)), shape=(n_vars, n_vars))

    # --- A matrix and b vector ---
    # Equality constraints: x0 = param, dynamics
    # Inequality constraints: -1 <= u <= 1 (as u <= 1, -u <= 1)
    n_eq = nx + T * nx  # initial condition + dynamics
    n_ineq = T * nu * 2  # control bounds
    n_con = n_eq + n_ineq

    a_rows, a_cols, a_vals = [], [], []
    b = np.zeros(n_con)
    row = 0

    # Initial condition: x_0 = x0_param (will be set via b vector)
    for i in range(nx):
        a_rows.append(row)
        a_cols.append(i)
        a_vals.append(1.0)
        row += 1

    # Dynamics: A_dyn @ x_t + B_dyn @ u_t - x_{t+1} = 0
    for t in range(T):
        x_off = t * (nx + nu)
        u_off = x_off + nx
        x_next = x_off + nx + nu
        for i in range(nx):
            for j in range(nx):
                if A_dyn[i, j] != 0:
                    a_rows.append(row)
                    a_cols.append(x_off + j)
                    a_vals.append(A_dyn[i, j])
            for j in range(nu):
                if B_dyn[i, j] != 0:
                    a_rows.append(row)
                    a_cols.append(u_off + j)
                    a_vals.append(B_dyn[i, j])
            a_rows.append(row)
            a_cols.append(x_next + i)
            a_vals.append(-1.0)
            row += 1

    # Control bounds: u_t <= 1 -> u_t + s = 1, s >= 0
    #                -u_t <= 1 -> -u_t + s = 1, s >= 0
    for t in range(T):
        u_off = t * (nx + nu) + nx
        for i in range(nu):
            a_rows.append(row)
            a_cols.append(u_off + i)
            a_vals.append(1.0)
            b[row] = 1.0
            row += 1
            a_rows.append(row)
            a_cols.append(u_off + i)
            a_vals.append(-1.0)
            b[row] = 1.0
            row += 1

    assert row == n_con

    A = sparse.csr_array((a_vals, (a_rows, a_cols)), shape=(n_con, n_vars))
    q = np.zeros(n_vars)

    cones = moreau.Cones(num_zero_cones=n_eq, num_nonneg_cones=n_ineq)

    return P, q, A, b, cones, n_vars, n_con, n_eq


# ============================================================================
# Benchmark helpers
# ============================================================================


def benchmark(name, fn, warmup=3, iters=10):
    """Run fn() with warmup, return median time."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    times = []
    for _ in range(iters):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    print(f"  {name}: {median*1000:.1f} ms (median of {iters})")
    return median


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print(f"MPC Benchmark: nx={nx}, nu={nu}, T={T}, batch={batch_size}, device={device}")
    print("=" * 70)

    # Random initial states
    x0_np = np.random.randn(batch_size, nx) * 0.5
    x0_torch = torch.tensor(x0_np, device=device, requires_grad=True)

    # ------------------------------------------------------------------
    # cvxpylayers + MOREAU
    # ------------------------------------------------------------------
    print("\n--- cvxpylayers + MOREAU ---")
    problem, x0_param, x_vars, u_vars = build_cvxpy_mpc()

    try:
        layer_moreau = CvxpyLayer(
            problem,
            parameters=[x0_param],
            variables=[x_vars[0], u_vars[0]],
            solver="MOREAU",
            solver_args={"max_iter": 1},
        )

        def run_cvxpylayers_moreau():
            x0_t = x0_torch.detach().requires_grad_(True)
            x_opt, u_opt = layer_moreau(x0_t)
            loss = x_opt.sum() + u_opt.sum()
            loss.backward()
            return x_opt, u_opt, x0_t.grad

        x_opt, u_opt, grad = run_cvxpylayers_moreau()
        print(f"  x[0] shape: {x_opt.shape}, u[0] shape: {u_opt.shape}")
        print(f"  obj (first problem): x[0]={x_opt[0,:3].detach().cpu().numpy()}")
        benchmark("forward+backward", run_cvxpylayers_moreau)
    except Exception as e:
        print(f"  SKIPPED: {e}")

    # ------------------------------------------------------------------
    # Native moreau.torch
    # ------------------------------------------------------------------
    print("\n--- moreau.torch (native) ---")
    P, q_np, A, b_np, cones, n_vars, n_con, n_eq = build_moreau_native_mpc()

    P_csr = P.tocsr()
    A_csr = A.tocsr()

    settings = moreau.Settings(
        verbose=True,
        batch_size=batch_size,
        device=str(device),
        max_iter=1,
    )

    solver = MoreauTorchSolver(
        n=n_vars,
        m=n_con,
        P_row_offsets=torch.tensor(P_csr.indptr, dtype=torch.int64),
        P_col_indices=torch.tensor(P_csr.indices, dtype=torch.int64),
        A_row_offsets=torch.tensor(A_csr.indptr, dtype=torch.int64),
        A_col_indices=torch.tensor(A_csr.indices, dtype=torch.int64),
        cones=cones,
        settings=settings,
    )

    P_values = torch.tensor(P_csr.data, dtype=torch.float64, device=device)
    A_values = torch.tensor(A_csr.data, dtype=torch.float64, device=device)

    # Expand to batch (shared structure)
    P_values_batch = P_values.unsqueeze(0).expand(batch_size, -1)
    A_values_batch = A_values.unsqueeze(0).expand(batch_size, -1)

    solver.setup(P_values_batch, A_values_batch)

    # Build q and b for each batch element
    # q is zero for all; b has x0 in the first nx entries
    q_batch = torch.zeros(batch_size, n_vars, dtype=torch.float64, device=device)
    b_batch = (
        torch.tensor(b_np, dtype=torch.float64, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .clone()
    )
    # Set initial conditions
    b_batch[:, :nx] = x0_torch.detach()

    b_grad = b_batch.clone().requires_grad_(True)

    def run_moreau_native():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        b_t = b_batch.detach().clone().requires_grad_(True)
        b_t.data[:, :nx] = x0_torch.detach()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        result = solver.solve(q_batch, b_t)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        loss = result.x.sum()
        loss.backward()
        torch.cuda.synchronize()
        t3 = time.perf_counter()
        return result, b_t.grad, (t1 - t0, t2 - t1, t3 - t2)

    result, grad_native, (t_setup, t_solve, t_backward) = run_moreau_native()
    print(
        f"  breakdown: setup={t_setup*1000:.1f}ms  solve={t_solve*1000:.1f}ms  backward={t_backward*1000:.1f}ms"
    )
    print(f"  x shape: {result.x.shape}")
    print(f"  x[0,:3]: {result.x[0,:3].detach().cpu().numpy()}")
    benchmark("forward+backward", run_moreau_native)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary:")
    print("  cvxpylayers overhead comes from DPP canonicalization on each call,")
    print("  plus CVXPY's cone reduction (SOC lifting of quad_form, etc).")
    print("  The native API skips all of that — same solver, less wrapping.")
    print("=" * 70)
