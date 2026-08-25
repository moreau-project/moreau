"""
SOC projection in 10 dimensions: verify forward + backward against analytical.

Projects v onto SOC_10 = {(t, x) : ||x|| <= t} by solving:
    min  0.5 ||x - v||^2   s.t.  x in SOC_10

Formulation: P = I, q = -v, A = -I, b = 0, cone = SOC(10).

Three cases:
  1. Interior  (||x|| < t)   -> proj = v,  Jacobian = I
  2. Polar     (||x|| <= -t) -> proj = 0,  Jacobian = 0
  3. Boundary  (||x|| > t)   -> proj = 0.5*(1 + t/||x||) * [||x||; x]
"""

import numpy as np
from scipy import sparse
import moreau

n = 10
P_csr = sparse.eye(n, format="csr")
A_csr = -sparse.eye(n, format="csr")
b = np.zeros(n)
cones = moreau.Cones(so_cone_dims=[n])


# ── Analytical SOC projection ────────────────────────────────────────────────


def proj_soc(v):
    t, x = v[0], v[1:]
    nx = np.linalg.norm(x)
    if nx <= t:
        return v.copy()
    elif nx <= -t:
        return np.zeros_like(v)
    else:
        alpha = 0.5 * (1 + t / nx)
        out = np.zeros_like(v)
        out[0] = alpha * nx
        out[1:] = alpha * x
        return out


# ── Analytical gradient of sum(proj(v)) w.r.t. v ────────────────────────────


def grad_sum_proj(v):
    """Returns d(sum proj(v))/dv for SOC projection."""
    t, x = v[0], v[1:]
    nx = np.linalg.norm(x)
    if nx <= t:
        return np.ones(n)
    elif nx <= -t:
        return np.zeros(n)
    else:
        sx = np.sum(x)
        dsum_dt = 0.5 + 0.5 * sx / nx
        dsum_dx = 0.5 * (1 + t / nx) * np.ones(n - 1) - 0.5 * t * x * sx / nx**3 + 0.5 * x / nx
        return np.concatenate([[dsum_dt], dsum_dx])


# ── Solver wrapper ───────────────────────────────────────────────────────────


def solve_and_grad(v, device):
    settings = moreau.Settings(device=device, batch_size=1, enable_grad=True)
    solver = moreau.CompiledSolver(
        n=n,
        m=n,
        P_row_offsets=P_csr.indptr.tolist(),
        P_col_indices=P_csr.indices.tolist(),
        A_row_offsets=A_csr.indptr.tolist(),
        A_col_indices=A_csr.indices.tolist(),
        cones=cones,
        settings=settings,
    )
    solver.setup(P_values=P_csr.data, A_values=A_csr.data)
    sol = solver.solve(qs=(-v).reshape(1, -1), bs=b.reshape(1, -1))
    grads = solver.backward(dx=np.ones((1, n)))
    return sol.x.flatten(), grads["dq"].flatten()


# ── Test cases ───────────────────────────────────────────────────────────────

np.random.seed(42)

cases = {
    "Interior (||x|| < t, proj = identity)": lambda: (
        np.concatenate([[5.0], np.random.randn(n - 1) * 0.3])
    ),
    "Polar (||x|| <= -t, proj = zero)": lambda: (
        np.concatenate([[-5.0], np.random.randn(n - 1) * 0.3])
    ),
    "Boundary (||x|| > t, proj onto boundary)": lambda: (
        np.concatenate(
            [[1.0], 3.0 * np.random.randn(n - 1) / np.linalg.norm(np.random.randn(n - 1))]
        )
    ),
}

all_pass = True
for label, make_v in cases.items():
    v = make_v()
    t, x = v[0], v[1:]
    nx = np.linalg.norm(x)

    x_anal = proj_soc(v)
    g_anal = grad_sum_proj(v)

    print(f"=== {label} ===")
    print(f"  v[0] (t) = {t:.4f},  ||v[1:]|| = {nx:.4f}")

    for device in ("cpu", "cuda"):
        x_sol, dq_sol = solve_and_grad(v, device)
        # dq = d(sum x)/dq = -d(sum x)/dv  since q = -v
        g_sol = -dq_sol

        fwd_err = np.max(np.abs(x_sol - x_anal))
        grad_err = np.max(np.abs(g_sol - g_anal))

        ok_fwd = fwd_err < 1e-6
        ok_grad = grad_err < 1e-4
        status = "PASS" if (ok_fwd and ok_grad) else "FAIL"
        if status == "FAIL":
            all_pass = False

        print(f"  {device:4s}  fwd_err={fwd_err:.2e}  grad_err={grad_err:.2e}  [{status}]")

    print()

print("ALL PASSED" if all_pass else "SOME FAILED")
