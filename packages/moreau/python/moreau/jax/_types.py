"""JAX-specific solution and info types for moreau.

These use NamedTuple for JAX pytree compatibility (jit, vmap, grad). They
live here so the JAX integration owns its own types.
"""

from typing import NamedTuple

from moreau._types import WarmStart


class JaxSolveInfo(NamedTuple):
    """Solver metadata from a JAX solve operation.

    Uses NamedTuple for JAX pytree compatibility (jit, vmap, grad).

    Attributes:
        status: Solver status as float (SolverStatus enum value)
        obj_val: Objective value at solution
        iterations: Number of iterations taken
        solve_time: Time spent in solve phase (seconds)
        setup_time: Time spent setting matrix values (seconds)
        construction_time: Time spent constructing solver structure (seconds)
    """

    status: float  # SolverStatus as float for JAX compatibility
    obj_val: float
    iterations: int
    solve_time: float
    setup_time: float
    construction_time: float


class JaxSolution(NamedTuple):
    """Solution vectors for a JAX conic optimization problem.

    Uses NamedTuple for JAX pytree compatibility (jit, vmap, grad).
    Contains only the optimization variables (primal and dual solutions).

    Attributes:
        x: Primal solution array
        z: Dual variable array (Lagrange multipliers)
        s: Slack variable array
        z_x: Direct-x cone duals (length sum |J|, or zero-length when no
            direct-x cones are present). Carrying it as the 4th field keeps
            JAX pytree-flatten / unflatten consistent across direct-x and
            slack-only problems.
    """

    x: "jnp.ndarray"
    z: "jnp.ndarray"
    s: "jnp.ndarray"
    z_x: "jnp.ndarray"

    def to_warm_start(self) -> WarmStart:
        """Create a WarmStart from this solution (converts JAX arrays to numpy)."""
        import numpy as np

        zx_arr = None
        try:
            if self.z_x.size > 0:
                zx_arr = np.asarray(self.z_x, dtype=np.float64)
        except (AttributeError, TypeError):
            pass
        return WarmStart(
            x=np.asarray(self.x, dtype=np.float64),
            z=np.asarray(self.z, dtype=np.float64),
            s=np.asarray(self.s, dtype=np.float64),
            z_x=zx_arr,
        )


__all__ = ["JaxSolveInfo", "JaxSolution"]
