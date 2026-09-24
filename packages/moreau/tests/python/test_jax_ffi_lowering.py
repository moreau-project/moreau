"""Stage CUDA FFI graphs without requiring a CUDA device or native library."""

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

pytestmark = pytest.mark.jax


@pytest.fixture(scope="module")
def lowering():
    path = Path(__file__).resolve().parents[3] / "moreau-cuda/moreau_cuda/jax/_lowering.py"
    # Import the Python lowering independently of moreau_cuda's native extension.
    spec = importlib.util.spec_from_file_location("moreau_ffi_lowering", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_ffi_with_float32_defaults(lowering, warm, batched, mixed):
    with jax.enable_x64(False):
        factory = lowering._make_ffi_solve_warm_fn if warm else lowering._make_ffi_solve_fn
        structure = {
            "P_row_offsets_gpu": [0, 1, 2],
            "P_col_indices_gpu": [0, 1],
            "A_row_offsets_gpu": [0, 2],
            "A_col_indices_gpu": [0, 1],
        }
        kwargs = {}
        for name in inspect.signature(factory).parameters:
            if name.endswith("_gpu"):
                kwargs[name] = jnp.array(
                    structure.get(name, []),
                    dtype=jnp.float64 if "alphas" in name else jnp.int64,
                )
            else:
                kwargs[name] = {
                    "n": 2,
                    "m": 1,
                    "num_zero": 1,
                    "wrapper_construction_time": 0.0,
                }.get(name, 0)
        solve = factory(**kwargs)
        warm_values = tuple(jnp.zeros(shape) for shape in [(2,), (1,), (1,), (0,)]) if warm else ()

        def loss(P, A, q, b):
            return solve(P, A, q, b, *warm_values)[0].x.sum()

        dtypes = (
            (jnp.float32, jnp.float64, jnp.float32, jnp.float64) if mixed else (jnp.float32,) * 4
        )
        inputs = tuple(
            jnp.array(values, dtype=dtype)
            for values, dtype in zip(([2.0, 3.0], [1.0, 2.0], [0.4, -0.2], [0.8]), dtypes)
        )
        fn = jax.value_and_grad(loss, argnums=(0, 1, 2, 3))
        if batched:
            fn = jax.vmap(fn)
            inputs = tuple(jnp.stack([x, x]) for x in inputs)
        fn = jax.jit(fn)
        trace = jax.make_jaxpr(fn)(*inputs)
        assert [v.aval.dtype for v in trace.jaxpr.outvars] == [np.result_type(*dtypes), *dtypes]
        ir = str(fn.lower(*inputs).compiler_ir())
        assert "pure_callback" not in str(trace)
        assert "python_cpu_callback" not in ir
        assert ("@moreau_solve_fwd_warm" if warm else "@moreau_solve_fwd") in ir
        assert "@moreau_solve_bwd" in ir
        assert not jax.config.jax_enable_x64
