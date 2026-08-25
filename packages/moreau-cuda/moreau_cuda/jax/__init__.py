"""CUDA JAX implementation using XLA FFI for true zero-copy GPU integration.

This provides a JAX-compatible interface to the CUDA solver with:
- jax.vmap support for batching (via custom_vmap rule)
- jax.grad support via custom_vjp
- True zero-copy GPU tensor sharing via XLA FFI

Requires JAX >= 0.6.0 for pure_callback vmap_method parameter.
"""

from ._ffi import ffi_available, clear_ffi_cache
from ._solver import JaxSolverCuda

__all__ = ["JaxSolverCuda", "ffi_available", "clear_ffi_cache"]
