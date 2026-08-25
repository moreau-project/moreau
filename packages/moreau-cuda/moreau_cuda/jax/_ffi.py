"""Load the JAX FFI shared library and register FFI targets."""

import ctypes
import logging
import os

import jax

# Enable 64-bit mode for JAX (required for the solver's numerical precision)
jax.config.update("jax_enable_x64", True)


# =============================================================================
# Load the FFI shared library
# =============================================================================

_FFI_LIB = None
_FFI_AVAILABLE = False


def _get_ffi_lib():
    """Load the JAX FFI shared library."""
    global _FFI_LIB, _FFI_AVAILABLE
    if _FFI_LIB is not None:
        return _FFI_LIB

    # Find the library in the moreau_cuda package directory
    lib_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_path = os.path.join(lib_dir, "lib_moreau_jax_ffi.so")

    if not os.path.exists(lib_path):
        _FFI_AVAILABLE = False
        logging.getLogger(__name__).warning(
            "JAX FFI library not found at %s; falling back to pure_callback "
            "(slower). Rebuild with: MOREAU_CUDA_ARCH=<arch> pip install -e "
            "packages/moreau-cuda --no-build-isolation",
            lib_path,
        )
        return None

    try:
        _FFI_LIB = ctypes.CDLL(lib_path)

        # Set up function signatures for cache clearing (for testing)
        if hasattr(_FFI_LIB, "moreau_jax_clear_cache"):
            _FFI_LIB.moreau_jax_clear_cache.argtypes = []
            _FFI_LIB.moreau_jax_clear_cache.restype = None

        # Register FFI targets with JAX (stateless handlers).
        # Register under both "CUDA" (JAX docs convention) and "cuda" (canonical
        # platform name) to work across JAX versions — some versions only look
        # up handlers by the canonical lowercase name.
        _fwd_capsule = jax.ffi.pycapsule(_FFI_LIB.MoreauSolveFwd)
        _bwd_capsule = jax.ffi.pycapsule(_FFI_LIB.MoreauSolveBwd)
        for _platform in ("CUDA", "cuda"):
            jax.ffi.register_ffi_target(
                "moreau_solve_fwd",
                _fwd_capsule,
                platform=_platform,
            )
            jax.ffi.register_ffi_target(
                "moreau_solve_bwd",
                _bwd_capsule,
                platform=_platform,
            )

        # Warm-start forward handler
        if hasattr(_FFI_LIB, "MoreauSolveFwdWarm"):
            _warm_capsule = jax.ffi.pycapsule(_FFI_LIB.MoreauSolveFwdWarm)
            for _platform in ("CUDA", "cuda"):
                jax.ffi.register_ffi_target(
                    "moreau_solve_fwd_warm",
                    _warm_capsule,
                    platform=_platform,
                )

        _FFI_AVAILABLE = True
        return _FFI_LIB

    except Exception as e:
        _FFI_AVAILABLE = False
        return None


def ffi_available() -> bool:
    """Check if XLA FFI zero-copy is available."""
    _get_ffi_lib()
    return _FFI_AVAILABLE


def clear_ffi_cache():
    """Clear the internal FFI solver cache (useful for testing)."""
    lib = _get_ffi_lib()
    if lib is not None and hasattr(lib, "moreau_jax_clear_cache"):
        lib.moreau_jax_clear_cache()
