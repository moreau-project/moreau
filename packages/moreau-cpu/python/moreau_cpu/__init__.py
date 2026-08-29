"""Moreau CPU solver: Conic interior-point optimization in Rust.

This package provides the CPU backend for moreau. It's typically installed as
a dependency of the main moreau package, not used directly.

For end users, install moreau instead:
    pip install moreau           # CPU only
    pip install moreau[cuda]     # CPU + GPU

Direct usage (advanced):
    import moreau_cpu
    from moreau_cpu import Solver

    # Unified solver (auto-detects single vs batch from input shape)
    solver = Solver(n, m, P_ro, P_ci, A_ro, A_ci, cones, settings)
    result = solver.solve(P_values, A_values, q, b)  # 1D=single, 2D=batch
"""

__version__ = "0.4.0-beta.1"

# Import the Rust extension module
from ._cpu_solver import *
from . import _cpu_solver

# Re-export the low-level module's __all__
if hasattr(_cpu_solver, "__all__"):
    __all__ = list(_cpu_solver.__all__)
else:
    __all__ = []

# Initialize BLAS/LAPACK
_cpu_solver.force_load_blas_lapack()

# Export module reference for type access
_cpu_solver_module = _cpu_solver

# High-level unified Solver wrapper
from ._cpu import (
    Solver,
    ActiveSetSolver,
    Cones,
    SolverStatus,
    cones_to_cpu,
    settings_to_cpu,
)

__all__.extend(
    [
        "__version__",
        # Solvers
        "Solver",
        "ActiveSetSolver",
        # Types
        "Cones",
        "SolverStatus",
        "cones_to_cpu",
        "settings_to_cpu",
    ]
)
