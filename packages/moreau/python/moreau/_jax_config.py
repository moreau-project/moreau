"""Shared precision policy for Moreau's JAX integrations."""

import jax
import numpy as np

if not hasattr(jax.config, "jax_explicit_x64_dtypes"):
    raise ImportError("Moreau's JAX integration requires JAX >= 0.8.0.")

jax.config.update("jax_explicit_x64_dtypes", "allow")


def _check_jax_precision():
    """Reject settings that would truncate the solver's float64 buffers."""
    if not jax.config.jax_enable_x64 and jax.config.jax_explicit_x64_dtypes.name != "ALLOW":
        raise ValueError(
            "Moreau requires 64-bit JAX arrays. Set "
            "jax.config.update('jax_explicit_x64_dtypes', 'allow') or "
            "jax.config.update('jax_enable_x64', True)."
        )


def _pure_callback(callback, result_shape_dtypes, *args, **kwargs):
    _check_jax_precision()
    if jax.config.jax_enable_x64:
        return jax.pure_callback(callback, result_shape_dtypes, *args, **kwargs)

    # pure_callback rejects float64 results and canonicalizes its runtime
    # buffers even with explicit x64 allowed. Carry float64 as pairs of uint32
    # words across that boundary, preserving every bit without changing defaults.
    arg_dtypes = jax.tree.map(lambda x: x.dtype, args)

    def pack_device(x):
        return jax.lax.bitcast_convert_type(x, np.uint32) if x.dtype == np.float64 else x

    def transport_shape(spec):
        if spec.dtype == np.float64:
            return jax.ShapeDtypeStruct((*spec.shape, 2), np.uint32)
        return spec

    def host_callback(*values):
        def unpack(x, dtype):
            if dtype == np.float64:
                return np.asarray(x).view(np.float64).reshape(x.shape[:-1])
            return x

        def pack(x):
            x = np.asarray(x)
            if x.dtype == np.float64:
                return np.ascontiguousarray(x).reshape(-1).view(np.uint32).reshape((*x.shape, 2))
            return x

        values = jax.tree.map(unpack, values, arg_dtypes)
        return jax.tree.map(pack, callback(*values))

    result = jax.pure_callback(
        host_callback,
        jax.tree.map(transport_shape, result_shape_dtypes),
        *jax.tree.map(pack_device, args),
        **kwargs,
    )
    return jax.tree.map(
        lambda x, spec: (
            jax.lax.bitcast_convert_type(x, np.float64) if spec.dtype == np.float64 else x
        ),
        result,
        result_shape_dtypes,
    )
