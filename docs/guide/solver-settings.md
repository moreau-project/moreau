# Solver Settings

Configure solver behavior through the `Settings`, `IPMSettings`, and `ActiveSetSettings` classes.

## Settings Overview

```python
import moreau

settings = moreau.Settings(
    solver='auto',        # Solver algorithm ('auto', 'ipm', 'active_set')
    device='auto',        # Device selection ('auto', 'cpu', 'cuda')
    device_id=-1,         # CUDA device ID (-1 = current device, ignored on CPU)
    batch_size=1,         # Batch size for CompiledSolver
    enable_grad=False,    # Enable gradient computation
    auto_tune=False,      # Benchmark solver configs on first solve (opt-in)
    max_iter=200,         # Maximum IPM iterations
    time_limit=float('inf'),  # Time limit in seconds
    verbose=False,        # Print solver progress
    yolo=False,           # Fixed-iteration mode, no convergence checking
    yolo_num_iters=15,    # Number of iterations in YOLO mode
    ipm_settings=None,    # Fine-grained IPM control (auto-created if None)
    active_set_settings=None,  # Active-set solver control (optional)
)
```

---

## Solver Selection

The `solver` parameter controls which algorithm is used:

| Solver | Cones | Device | Best For |
|--------|-------|--------|----------|
| `'auto'` | any | auto | Default — picks best solver for the problem |
| `'ipm'` | all | CPU / CUDA | General conic problems, high accuracy |
| `'active_set'` | zero + nonneg only | CPU | Small QPs, very fast forward/backward |

```python
# Auto-select (default): uses active-set for small QPs, IPM otherwise
settings = moreau.Settings(solver='auto')

# Force IPM (interior-point method)
settings = moreau.Settings(solver='ipm')

# Force active-set (zero + nonneg cones only)
settings = moreau.Settings(solver='active_set')
```

When `solver='auto'`, the active-set solver is selected for problems with:
- Only zero and nonneg cones (pure QPs/LPs)
- n <= 500 variables
- m <= max(500, 2n) constraints

Otherwise the IPM solver is used.

!!! warning "Auto + gradients: gradient quality changes at the threshold"
    Active-set produces non-smooth (combinatorial) derivatives. When
    `solver='auto'` picks active-set, gradient quality changes
    discontinuously the moment a problem grows past the n=500 /
    m=max(500, 2n) threshold (above which auto picks IPM, with smooth
    gradients). For ML training loops where problem size varies along a
    sweep, that's a silent footgun: bumping `n` from 499 to 501 invisibly
    switches you between non-smooth active-set gradients and smooth IPM
    gradients, changing outer-loop convergence behaviour.

    When `solver='auto'` resolves to active-set on a problem with
    `enable_grad=True`, Moreau emits a one-time `UserWarning` naming the
    cliff and the workaround. For smooth gradients regardless of size, pass
    `solver='ipm'` explicitly. If you've already chosen
    `solver='active_set'` deliberately, the warning won't fire.

---

## Device Settings

```python
# Automatic device selection (default)
settings = moreau.Settings(device='auto')

# Force CPU
settings = moreau.Settings(device='cpu')

# Force CUDA GPU
settings = moreau.Settings(device='cuda')

# Force a specific CUDA device
settings = moreau.Settings(device='cuda', device_id=1)
```

See [Device Selection](device-selection.md) for more details.

---

## Batch Settings

For `CompiledSolver`, specify batch size:

```python
settings = moreau.Settings(batch_size=64)

solver = moreau.CompiledSolver(
    n=2, m=3,
    P_row_offsets=[0, 1, 2], P_col_indices=[0, 1],
    A_row_offsets=[0, 2, 3, 4], A_col_indices=[0, 1, 0, 1],
    cones=cones,
    settings=settings,
)
```

---

## Iteration & Time Control

```python
settings = moreau.Settings(
    max_iter=200,               # Maximum iterations (default: 200)
    time_limit=30.0,            # Time limit in seconds (default: infinity)
    verbose=True,               # Print iteration progress
)
```

---

## Gradient Settings

Enable gradients for differentiation through the solver:

```python
# Core API (NumPy)
settings = moreau.Settings(enable_grad=True)
solver = moreau.CompiledSolver(..., settings=settings)

# PyTorch (enable_grad is True by default)
from moreau.torch import Solver
solver = Solver(...)  # Gradients enabled automatically
```

---

## IPM Settings

Fine-tune the interior-point method:

```python
ipm_settings = moreau.IPMSettings(
    # Convergence tolerances
    tol_gap_abs=1e-8,     # Absolute duality gap tolerance
    tol_gap_rel=1e-8,     # Relative duality gap tolerance
    tol_feas=1e-8,        # Primal/dual feasibility tolerance
    tol_infeas_abs=1e-8,  # Infeasibility detection tolerance
    tol_infeas_rel=1e-8,  # Relative infeasibility tolerance
    tol_ktratio=1e-6,     # KKT ratio tolerance

    # Algorithm control
    max_step_fraction=0.99,    # Maximum step size fraction (0, 1]
    equilibrate_enable=True,   # Enable matrix equilibration

    # KKT solver method
    direct_solve_method='auto',  # See table below
)

settings = moreau.Settings(ipm_settings=ipm_settings)
```

### KKT Solver Method

The `direct_solve_method` controls which linear algebra backend solves the KKT
system at each IPM iteration:

| Method | Device | Best For |
|--------|--------|----------|
| `'auto'` | any | Heuristic or auto-tuned (default) |
| `'qdldl'` | CPU | Small/sparse KKT systems |
| `'faer'` | CPU | Large CPU problems (multi-threaded) |
| `'faer-1t'` | CPU | Large CPU problems (single-threaded) |
| `'faer-nt'` | CPU | Large CPU problems (automatic thread count) |
| `'cudss'` | CUDA | Large GPU problems |
| `'riccati'` | CPU / CUDA | Block-tridiagonal problems (MPC/LQR) |
| `'woodbury'` | CUDA | Portfolio-type (diagonal P + low-rank constraints) |

When set to `'auto'` (the default), a heuristic picks the best method for the
device without benchmarking. To enable benchmarking on the first `solve()` call,
set `auto_tune=True`:

```python
# Heuristic selection (default, no benchmarking)
settings = moreau.Settings(device='auto')

# Benchmarking on first solve
settings = moreau.Settings(device='auto', auto_tune=True)
```

Set both device and method explicitly to skip all automatic selection.

### Reduced Tolerances

When the solver cannot meet the primary tolerances, it checks against a set of
relaxed "reduced" tolerances. If those are met, the status is `AlmostSolved`
(or `AlmostPrimalInfeasible`/`AlmostDualInfeasible`):

```python
ipm_settings = moreau.IPMSettings(
    reduced_tol_gap_abs=5e-5,       # default 5e-5
    reduced_tol_gap_rel=5e-5,       # default 5e-5
    reduced_tol_feas=1e-4,          # default 1e-4
    reduced_tol_infeas_abs=5e-12,   # default 5e-12
    reduced_tol_infeas_rel=5e-5,    # default 5e-5
    reduced_tol_ktratio=1e-4,       # default 1e-4
)
```

### Warm Start Retry Control

By default, if a warm-started solve returns a "bad" status, the solver
automatically retries cold. The `warm_start_no_retry` parameter controls which
statuses are considered "acceptable" (i.e. do **not** trigger a retry):

```python
from moreau import IPMSettings, SolverStatus

# Default behavior (None): {Solved, AlmostSolved, MaxIterations, CallbackTerminated}
ipm = moreau.IPMSettings()

# Custom: also accept MaxTime without retrying cold
ipm = moreau.IPMSettings(
    warm_start_no_retry=frozenset({
        SolverStatus.Solved,
        SolverStatus.AlmostSolved,
        SolverStatus.MaxTime,
        SolverStatus.MaxIterations,
        SolverStatus.CallbackTerminated,
    })
)
```

### Tolerance Guidelines

| Tolerance | Effect |
|-----------|--------|
| `tol_gap_abs/rel` | Lower = more accurate objective |
| `tol_feas` | Lower = stricter constraint satisfaction |
| Higher tolerances | Faster convergence, less accuracy |
| Lower tolerances | More iterations, higher accuracy |

Default tolerances (1e-8) are suitable for most applications.

---

## Active-Set Settings

Fine-tune the active-set QP solver (used when `solver='active_set'`):

```python
as_settings = moreau.ActiveSetSettings(
    # Convergence tolerances
    primal_tol=1e-6,       # Primal feasibility tolerance
    dual_tol=1e-12,        # Dual feasibility tolerance
    zero_tol=1e-11,        # Zero tolerance for numerical checks
    pivot_tol=1e-6,        # Pivot tolerance for LDL factorization
    progress_tol=1e-14,    # Progress tolerance for cycle detection

    # Limits
    iter_limit=10000,      # Maximum iterations
    cycle_tol=10,          # Cycle detection tolerance
    fval_bound=1e30,       # Objective bound for infeasibility detection

    # Differentiation
    diff_method='exact',       # 'exact' or 'smoothed'
    diff_smoothing_mu=1e-4,    # Smoothing parameter (for 'smoothed' mode)
)

settings = moreau.Settings(
    solver='active_set',
    enable_grad=True,
    active_set_settings=as_settings,
)
```

### Active-Set Differentiation

The active-set solver supports two differentiation modes:

- **`'exact'`** (default): Hard active-set KKT adjoint. Fast and accurate, but
  discontinuous at constraint transitions (when a constraint switches between
  active and inactive).

- **`'smoothed'`**: Barrier-smoothed KKT using central-path regularization.
  Produces C^∞ gradients through active-set transitions, matching the IPM's
  smoothed differentiation behavior. Controlled by `diff_smoothing_mu` — larger
  values give smoother gradients at the cost of accuracy.

```python
# Smoothed differentiation for active-set solver
settings = moreau.Settings(
    solver='active_set',
    enable_grad=True,
    active_set_settings=moreau.ActiveSetSettings(
        diff_method='smoothed',
        diff_smoothing_mu=1e-3,
    ),
)
```

### Active-Set Warm Starting

The active-set solver supports warm starting from a previous solution. When warm
starting, the active set from the previous solution is used as the initial
working set, which can significantly reduce iteration count for similar problems:

```python
solver = moreau.CompiledSolver(..., settings=moreau.Settings(solver='active_set'))
solver.setup(P_values, A_values)

# First solve
solution = solver.solve(qs=q1, bs=b1)
ws = solution.to_warm_start()

# Warm-started solve (typically fewer iterations)
solution2 = solver.solve(qs=q2, bs=b2, warm_start=ws)
```

---

## Common Configurations

### High Accuracy

```python
ipm_settings = moreau.IPMSettings(
    tol_gap_abs=1e-10,
    tol_gap_rel=1e-10,
    tol_feas=1e-10,
)
settings = moreau.Settings(
    max_iter=500,
    ipm_settings=ipm_settings,
)
```

### Fast/Approximate

```python
ipm_settings = moreau.IPMSettings(
    tol_gap_abs=1e-6,
    tol_gap_rel=1e-6,
    tol_feas=1e-6,
)
settings = moreau.Settings(
    max_iter=100,
    ipm_settings=ipm_settings,
)
```

### GPU Batched with Time Limit

```python
settings = moreau.Settings(
    device='cuda',
    batch_size=1024,
    enable_grad=True,
    time_limit=60.0,
)
```

### YOLO Mode (Fixed-Iteration, Zero-Sync) — Experimental

> **Experimental:** YOLO mode is under active development. Its API and behavior may change.

YOLO mode runs exactly `yolo_num_iters` IPM iterations with **no convergence
checking** and **no GPU-to-host synchronization** in the iteration loop. This is
useful for latency-sensitive workloads where you know roughly how many iterations
are needed (e.g., warm-starting, real-time control, approximate solutions).

```python
settings = moreau.Settings(
    device='cuda',
    batch_size=128,
    yolo=True,
    yolo_num_iters=10,  # default: 15
)
```

All batches return `SolverStatus.MaxIterations`. The solver automatically
preserves the last numerically valid (non-NaN) iterate, so overshooting
`yolo_num_iters` past convergence is safe — you get the best solution found
rather than NaN.

**Constraints:**

- Incompatible with `enable_grad=True` (backward pass needs convergence data)
- Forces `verbose=False`
- Works on both CPU and CUDA backends

**Choosing `yolo_num_iters`:** Run the problem once with normal settings and
check `solver.info.iterations` to see how many iterations convergence takes.
Set `yolo_num_iters` to that value or slightly higher.
