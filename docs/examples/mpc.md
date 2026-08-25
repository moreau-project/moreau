# Model Predictive Control

This example demonstrates using Moreau for Model Predictive Control (MPC).

## Problem Formulation

Linear MPC with quadratic cost:

$$
\begin{aligned}
\text{minimize} \quad & \sum_{t=0}^{N-1} \bigl[x_t^\top Q x_t + u_t^\top R u_t\bigr] + x_N^\top Q_f x_N \\
\text{subject to} \quad & x_{t+1} = A x_t + B u_t \\
& x_{\min} \le x_t \le x_{\max} \\
& u_{\min} \le u_t \le u_{\max} \\
& x_0 = x_{\text{init}}
\end{aligned}
$$

Where:
- $x_t$ is the state at time $t$
- $u_t$ is the control input at time $t$
- $A, B$ are system dynamics matrices
- $Q, R, Q_f$ are cost matrices
- $N$ is the prediction horizon

## Single MPC Step

```python
import moreau
import numpy as np
from scipy import sparse

def build_mpc_problem(A, B, Q, R, Q_f, N, x_min, x_max, u_min, u_max):
    """
    Build the QP matrices for a linear MPC problem.

    Args:
        A: State transition matrix (nx, nx)
        B: Input matrix (nx, nu)
        Q: State cost matrix (nx, nx)
        R: Input cost matrix (nu, nu)
        Q_f: Terminal cost matrix (nx, nx)
        N: Prediction horizon
        x_min, x_max: State bounds (nx,)
        u_min, u_max: Input bounds (nu,)

    Returns:
        P, A_eq, A_ineq matrices and dimensions
    """
    nx, nu = B.shape

    # Decision variables: [x_0, u_0, x_1, u_1, ..., x_N]
    n_vars = (N + 1) * nx + N * nu

    # Build cost matrix P (block diagonal)
    P_blocks = []
    for t in range(N):
        P_blocks.append(Q)  # x_t cost
        P_blocks.append(R)  # u_t cost
    P_blocks.append(Q_f)  # x_N cost

    P = sparse.block_diag(P_blocks, format='csr')

    # Dynamics constraints: x_{t+1} = Ax_t + Bu_t
    # Rewritten as: Ax_t + Bu_t - x_{t+1} = 0
    eq_rows = []
    for t in range(N):
        row = sparse.lil_matrix((nx, n_vars))
        x_t_idx = t * (nx + nu)
        u_t_idx = t * (nx + nu) + nx
        x_tp1_idx = (t + 1) * (nx + nu) if t < N - 1 else N * (nx + nu)

        row[:, x_t_idx:x_t_idx+nx] = A
        row[:, u_t_idx:u_t_idx+nu] = B
        row[:, x_tp1_idx:x_tp1_idx+nx] = -np.eye(nx)
        eq_rows.append(row.tocsr())

    A_dynamics = sparse.vstack(eq_rows, format='csr')

    # Initial condition: x_0 = x_init (will set b later)
    A_init = sparse.lil_matrix((nx, n_vars))
    A_init[:, :nx] = np.eye(nx)
    A_init = A_init.tocsr()

    # State bounds: x_min <= x_t <= x_max for all t
    # Rewritten as: -x_t <= -x_min, x_t <= x_max
    state_bound_rows = []
    for t in range(N + 1):
        if t < N:
            x_idx = t * (nx + nu)
        else:
            x_idx = N * (nx + nu)

        # -x_t <= -x_min
        row_lower = sparse.lil_matrix((nx, n_vars))
        row_lower[:, x_idx:x_idx+nx] = -np.eye(nx)
        state_bound_rows.append(row_lower.tocsr())

        # x_t <= x_max
        row_upper = sparse.lil_matrix((nx, n_vars))
        row_upper[:, x_idx:x_idx+nx] = np.eye(nx)
        state_bound_rows.append(row_upper.tocsr())

    # Input bounds: u_min <= u_t <= u_max for all t
    input_bound_rows = []
    for t in range(N):
        u_idx = t * (nx + nu) + nx

        # -u_t <= -u_min
        row_lower = sparse.lil_matrix((nu, n_vars))
        row_lower[:, u_idx:u_idx+nu] = -np.eye(nu)
        input_bound_rows.append(row_lower.tocsr())

        # u_t <= u_max
        row_upper = sparse.lil_matrix((nu, n_vars))
        row_upper[:, u_idx:u_idx+nu] = np.eye(nu)
        input_bound_rows.append(row_upper.tocsr())

    A_state_bounds = sparse.vstack(state_bound_rows, format='csr')
    A_input_bounds = sparse.vstack(input_bound_rows, format='csr')

    return {
        'P': P,
        'A_eq': sparse.vstack([A_dynamics, A_init], format='csr'),
        'A_ineq': sparse.vstack([A_state_bounds, A_input_bounds], format='csr'),
        'nx': nx,
        'nu': nu,
        'N': N,
        'n_vars': n_vars,
        'n_eq': N * nx + nx,  # dynamics + initial
        'n_ineq': 2 * (N + 1) * nx + 2 * N * nu,
        'x_min': x_min,
        'x_max': x_max,
        'u_min': u_min,
        'u_max': u_max,
    }


def solve_mpc(problem, x_init):
    """Solve MPC for given initial state."""
    nx, nu, N = problem['nx'], problem['nu'], problem['N']

    # Full constraint matrix
    A = sparse.vstack([problem['A_eq'], problem['A_ineq']], format='csr')

    # RHS: dynamics=0, init=x_init, bounds
    b_eq = np.concatenate([
        np.zeros(N * nx),  # dynamics
        x_init,            # initial condition
    ])

    b_ineq = np.concatenate([
        np.tile(-problem['x_min'], N + 1),  # -x <= -x_min
        np.tile(problem['x_max'], N + 1),   # x <= x_max
        np.tile(-problem['u_min'], N),       # -u <= -u_min
        np.tile(problem['u_max'], N),        # u <= u_max
    ])

    b = np.concatenate([b_eq, b_ineq])
    q = np.zeros(problem['n_vars'])

    cones = moreau.Cones(
        num_zero_cones=problem['n_eq'],
        num_nonneg_cones=problem['n_ineq'],
    )

    solver = moreau.Solver(problem['P'], q, A, b, cones=cones)
    solution = solver.solve()

    # Tip: For MPC problems, the Riccati solver exploits block-tridiagonal
    # structure and is auto-selected when detected. To force it:
    # settings = moreau.Settings(ipm_settings=moreau.IPMSettings(direct_solve_method='riccati'))

    # Extract first control input
    u_0 = solution.x[nx:nx+nu]
    return u_0, solution, solver.info


# Example: Double integrator
nx, nu = 4, 2  # 2D position + velocity
N = 10

# Dynamics: x_{t+1} = Ax_t + Bu_t
dt = 0.1
A = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])
B = np.array([
    [0.5*dt**2, 0],
    [0, 0.5*dt**2],
    [dt, 0],
    [0, dt],
])

# Costs
Q = np.diag([1, 1, 0.1, 0.1])
R = np.eye(nu) * 0.1
Q_f = Q * 10

# Bounds
x_min = np.array([-10, -10, -2, -2])
x_max = np.array([10, 10, 2, 2])
u_min = np.array([-1, -1])
u_max = np.array([1, 1])

# Build problem
problem = build_mpc_problem(A, B, Q, R, Q_f, N, x_min, x_max, u_min, u_max)

# Solve for initial state
x_init = np.array([5.0, 3.0, 0.0, 0.0])
u_opt, solution, info = solve_mpc(problem, x_init)
print(f"Optimal control: {u_opt}")
print(f"Status: {info.status}")
```

## Batched MPC

For parallel MPC across multiple initial states:

```python
import moreau
import numpy as np
from scipy import sparse

def batched_mpc(problem, x_inits):
    """
    Solve MPC for multiple initial states in parallel.

    Args:
        problem: MPC problem structure
        x_inits: Initial states, shape (batch, nx)

    Returns:
        Optimal controls, shape (batch, nu)
    """
    batch_size = len(x_inits)
    nx, nu, N = problem['nx'], problem['nu'], problem['N']

    # Full constraint matrix
    A = sparse.vstack([problem['A_eq'], problem['A_ineq']], format='csr')

    # CSR structure
    P_ro = problem['P'].indptr.tolist()
    P_ci = problem['P'].indices.tolist()
    A_ro = A.indptr.tolist()
    A_ci = A.indices.tolist()

    cones = moreau.Cones(
        num_zero_cones=problem['n_eq'],
        num_nonneg_cones=problem['n_ineq'],
    )
    settings = moreau.Settings(batch_size=batch_size)

    solver = moreau.CompiledSolver(
        n=problem['n_vars'],
        m=problem['n_eq'] + problem['n_ineq'],
        P_row_offsets=P_ro,
        P_col_indices=P_ci,
        A_row_offsets=A_ro,
        A_col_indices=A_ci,
        cones=cones,
        settings=settings,
    )

    # Matrix values (shared)
    solver.setup(problem['P'].data, A.data)

    # Build batched b vectors (only initial condition differs)
    bs = []
    for x_init in x_inits:
        b_eq = np.concatenate([np.zeros(N * nx), x_init])
        b_ineq = np.concatenate([
            np.tile(-problem['x_min'], N + 1),
            np.tile(problem['x_max'], N + 1),
            np.tile(-problem['u_min'], N),
            np.tile(problem['u_max'], N),
        ])
        bs.append(np.concatenate([b_eq, b_ineq]))

    qs = np.zeros((batch_size, problem['n_vars']))
    bs = np.array(bs)

    solution = solver.solve(qs, bs)

    # Extract first control inputs
    u_opts = solution.x[:, nx:nx+nu]
    return u_opts

# Example: 100 random initial states
batch_size = 100
x_inits = np.random.uniform(-5, 5, (batch_size, nx))
x_inits[:, 2:] = 0  # Zero initial velocity

u_opts = batched_mpc(problem, x_inits)
print(f"Batch shape: {u_opts.shape}")
print(f"First control: {u_opts[0]}")
```
