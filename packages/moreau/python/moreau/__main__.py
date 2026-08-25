"""
Diagnostic command for moreau package.

Usage:
    python -m moreau check
"""

import argparse
import sys
from typing import Optional


def check_package_version(name: str) -> tuple[bool, Optional[str]]:
    """Check if a package is installed and return version."""
    try:
        import importlib.metadata

        version = importlib.metadata.version(name)
        return True, version
    except importlib.metadata.PackageNotFoundError:
        return False, None


def check_import(name: str) -> tuple[bool, Optional[str]]:
    """Check if a module can be imported."""
    try:
        module = __import__(name)
        version = getattr(module, "__version__", None)
        return True, version
    except ImportError as e:
        return False, str(e)


def test_cpu_solver() -> tuple[bool, str]:
    """Test CPU solver with a simple QP."""
    try:
        import numpy as np
        from scipy import sparse
        import moreau

        # Simple QP: min 0.5*x'Px + q'x s.t. Ax + s = b, s >= 0
        P = sparse.diags([1.0, 1.0], format="csr")
        q = np.array([2.0, 1.0])
        A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 0.7, 0.7])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        settings = moreau.Settings(device="cpu", verbose=False)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solver.solve()
        info = solver.info

        if info.status.name == "Solved":
            return True, f"Solved in {info.iterations} iterations, obj={info.obj_val:.6f}"
        else:
            return False, f"Status: {info.status.name}"
    except Exception as e:
        return False, str(e)


def test_cuda_solver() -> tuple[bool, str]:
    """Test CUDA solver with a simple QP."""
    try:
        import numpy as np
        from scipy import sparse
        import moreau

        if not moreau.device_available("cuda"):
            return False, "CUDA device not available"

        P = sparse.diags([1.0, 1.0], format="csr")
        q = np.array([2.0, 1.0])
        A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        b = np.array([1.0, 0.7, 0.7])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        settings = moreau.Settings(device="cuda", verbose=False)
        solver = moreau.Solver(P, q, A, b, cones, settings)
        solver.solve()
        info = solver.info

        if info.status.name == "Solved":
            return True, f"Solved in {info.iterations} iterations, obj={info.obj_val:.6f}"
        else:
            return False, f"Status: {info.status.name}"
    except Exception as e:
        return False, str(e)


def test_torch_interface() -> tuple[bool, str]:
    """Test PyTorch interface."""
    try:
        import torch
        from scipy import sparse
        import moreau
        from moreau.torch import Solver as TorchSolver

        device = "cuda" if moreau.device_available("cuda") and torch.cuda.is_available() else "cpu"

        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        # batch_size goes in Settings, not as separate arg
        settings = moreau.Settings(device=device, verbose=False, batch_size=1)
        solver = TorchSolver(
            n=2,
            m=3,
            P_row_offsets=torch.tensor(P.indptr, dtype=torch.int32),
            P_col_indices=torch.tensor(P.indices, dtype=torch.int32),
            A_row_offsets=torch.tensor(A.indptr, dtype=torch.int32),
            A_col_indices=torch.tensor(A.indices, dtype=torch.int32),
            cones=cones,
            settings=settings,
        )

        P_vals = torch.tensor(P.data, dtype=torch.float64, device=device).unsqueeze(0)
        A_vals = torch.tensor(A.data, dtype=torch.float64, device=device).unsqueeze(0)

        q = torch.tensor([[2.0, 1.0]], dtype=torch.float64, device=device)
        b = torch.tensor([[1.0, 0.7, 0.7]], dtype=torch.float64, device=device)

        result = solver.solve(P_vals, A_vals, q, b)
        info = solver.info

        # info.status is a list of SolverStatus enums for batched mode
        if info.status[0].name == "Solved":
            return True, f"Solved on {device}, x={result.x[0].tolist()}"
        else:
            return False, f"Status: {info.status[0].name}"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, str(e)


def test_torch_autograd() -> tuple[bool, str]:
    """Test PyTorch autograd support."""
    try:
        import torch
        from scipy import sparse
        import moreau
        from moreau.torch import Solver as TorchSolver

        device = "cuda" if moreau.device_available("cuda") and torch.cuda.is_available() else "cpu"

        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        # enable_grad is automatically True for torch.Solver
        settings = moreau.Settings(device=device, verbose=False, batch_size=1)
        solver = TorchSolver(
            n=2,
            m=3,
            P_row_offsets=torch.tensor(P.indptr, dtype=torch.int32),
            P_col_indices=torch.tensor(P.indices, dtype=torch.int32),
            A_row_offsets=torch.tensor(A.indptr, dtype=torch.int32),
            A_col_indices=torch.tensor(A.indices, dtype=torch.int32),
            cones=cones,
            settings=settings,
        )

        P_vals = torch.tensor(
            P.data, dtype=torch.float64, device=device, requires_grad=True
        ).unsqueeze(0)
        A_vals = torch.tensor(
            A.data, dtype=torch.float64, device=device, requires_grad=True
        ).unsqueeze(0)

        q = torch.tensor([[2.0, 1.0]], dtype=torch.float64, device=device, requires_grad=True)
        b = torch.tensor([[1.0, 0.7, 0.7]], dtype=torch.float64, device=device, requires_grad=True)

        result = solver.solve(P_vals, A_vals, q, b)
        loss = result.x.sum()
        loss.backward()

        has_grads = q.grad is not None and b.grad is not None
        if has_grads:
            return True, f"Autograd working on {device}, dL/dq={q.grad[0].tolist()}"
        else:
            return False, "Gradients not computed"
    except ImportError:
        return False, "PyTorch not installed"
    except Exception as e:
        return False, str(e)


def test_jax_interface() -> tuple[bool, str]:
    """Test JAX interface."""
    try:
        import jax
        import jax.numpy as jnp
        from scipy import sparse
        import moreau
        from moreau.jax import Solver as JaxSolver

        jax_has_gpu = any(d.platform == "gpu" for d in jax.devices())
        device = "cuda" if moreau.device_available("cuda") and jax_has_gpu else "cpu"

        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)

        settings = moreau.Settings(device=device, verbose=False, batch_size=1)
        solver = JaxSolver(
            n=2,
            m=3,
            P_row_offsets=jnp.array(P.indptr),
            P_col_indices=jnp.array(P.indices),
            A_row_offsets=jnp.array(A.indptr),
            A_col_indices=jnp.array(A.indices),
            cones=cones,
            settings=settings,
        )

        P_values = jnp.array(P.data)
        A_values = jnp.array(A.data)
        q = jnp.array([2.0, 1.0])
        b = jnp.array([1.0, 0.7, 0.7])

        solution = solver.solve(P_values, A_values, q, b)
        info = solver.info

        if int(info.status) == moreau.SolverStatus.Solved:
            return True, f"Solved on {device}, x={solution.x.tolist()}"
        else:
            return False, f"Status: {int(info.status)}"
    except ImportError:
        return False, "JAX not installed"
    except Exception as e:
        return False, str(e)


def test_cupy_interface() -> tuple[bool, str]:
    """Test the zero-copy CuPy path.

    Passes CuPy device pointers (``ndarray.data.ptr``) straight to the CUDA
    solver's ``solve_to_device_pointers`` entry point — the same zero-copy
    interface PyTorch and JAX use via ``tensor.data_ptr()`` — with no host
    round-trip.
    """
    try:
        import cupy as cp
    except ImportError:
        return False, "CuPy not installed"
    try:
        from moreau_cuda import CudaSolver, _cones_to_cuda, _settings_to_cuda
    except ImportError:
        return False, "moreau-cuda not installed (CuPy interop needs the CUDA backend)"
    try:
        import numpy as np
        from scipy import sparse
        import moreau

        # Sparsity structure lives on the host (set once at construction).
        P = sparse.diags([1.0, 1.0], format="csr")
        A = sparse.csr_array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
        cones = moreau.Cones(num_zero_cones=1, num_nonneg_cones=2)
        settings = moreau.Settings(device="cuda", verbose=False)

        solver = CudaSolver(
            2,
            3,
            1,  # n, m, batch_size
            np.asarray(P.indptr, dtype=np.int64),
            np.asarray(P.indices, dtype=np.int64),
            np.asarray(A.indptr, dtype=np.int64),
            np.asarray(A.indices, dtype=np.int64),
            _cones_to_cuda(cones),
            _settings_to_cuda(settings),
            False,  # enable_grad
        )

        # Problem values and parameters live on the GPU as CuPy arrays.
        P_vals = cp.asarray(P.data, dtype=cp.float64).reshape(1, -1)
        A_vals = cp.asarray(A.data, dtype=cp.float64).reshape(1, -1)
        q = cp.asarray([[2.0, 1.0]], dtype=cp.float64)
        b = cp.asarray([[1.0, 0.7, 0.7]], dtype=cp.float64)
        x = cp.empty((1, 2), dtype=cp.float64)
        z = cp.empty((1, 3), dtype=cp.float64)
        s = cp.empty((1, 3), dtype=cp.float64)
        status = cp.empty(1, dtype=cp.int32)
        obj = cp.empty(1, dtype=cp.float64)

        # Zero-copy: only device pointers cross into C++ (no host transfer).
        solver.solve_to_device_pointers(
            int(P_vals.data.ptr),
            int(A_vals.data.ptr),
            int(q.data.ptr),
            int(b.data.ptr),
            int(x.data.ptr),
            int(z.data.ptr),
            int(s.data.ptr),
            int(status.data.ptr),
            int(obj.data.ptr),
        )

        status_code = int(status[0])
        if status_code != int(moreau.SolverStatus.Solved):
            return False, f"Status code {status_code} (expected Solved)"
        return True, f"Zero-copy solve, x={cp.asnumpy(x)[0].tolist()}"
    except Exception as e:
        return False, str(e)


def print_status(name: str, ok: bool, detail: str = "", indent: int = 0):
    """Print a status line."""
    prefix = "  " * indent
    symbol = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
    if detail:
        print(f"{prefix}{symbol} {name}: {detail}")
    else:
        print(f"{prefix}{symbol} {name}")


def run_check(verbose: bool = False):
    """Run all diagnostic checks."""
    print("=" * 60)
    print("Moreau Solver Diagnostics")
    print("=" * 60)
    print()

    # Core packages
    print("Core Packages:")
    print("-" * 40)

    ok, version = check_package_version("moreau")
    print_status("moreau", ok, version or "not installed", indent=1)

    ok, version = check_package_version("moreau-cpu")
    print_status("moreau-cpu", ok, version or "not installed", indent=1)

    ok, version = check_package_version("moreau-cuda")
    cuda_available = ok
    print_status("moreau-cuda", ok, version or "not installed", indent=1)

    print()

    # Backend availability
    print("Device Backends:")
    print("-" * 40)

    try:
        import moreau

        devices = moreau.available_devices()
        default = moreau.default_device()
        print_status("Available devices", True, ", ".join(devices), indent=1)
        print_status("Default device", True, default, indent=1)
    except Exception as e:
        print_status("Device detection", False, str(e), indent=1)

    print()

    # Optional dependencies
    print("Optional Dependencies:")
    print("-" * 40)

    ok, version = check_import("torch")
    torch_available = ok
    print_status("PyTorch", ok, version or "not installed", indent=1)

    if torch_available:
        try:
            import torch

            cuda_torch = torch.cuda.is_available()
            if cuda_torch:
                device_name = torch.cuda.get_device_name(0)
                print_status("  CUDA available", True, device_name, indent=1)
            else:
                print_status("  CUDA available", False, "no CUDA devices", indent=1)
        except Exception as e:
            print_status("  CUDA check", False, str(e), indent=1)

    ok, version = check_import("cupy")
    print_status("CuPy", ok, version or "not installed", indent=1)

    ok, version = check_import("jax")
    print_status("JAX", ok, version or "not installed", indent=1)

    ok, _ = check_import("scipy")
    print_status("SciPy", ok, "installed" if ok else "not installed", indent=1)

    ok, _ = check_import("numpy")
    if ok:
        import numpy as np

        print_status("NumPy", ok, np.__version__, indent=1)
    else:
        print_status("NumPy", ok, "not installed", indent=1)

    print()

    # Solver tests
    print("Solver Tests:")
    print("-" * 40)

    ok, msg = test_cpu_solver()
    print_status("CPU solver", ok, msg, indent=1)

    if cuda_available:
        ok, msg = test_cuda_solver()
        print_status("CUDA solver", ok, msg, indent=1)
    else:
        print_status("CUDA solver", False, "moreau-cuda not installed", indent=1)

    print()

    # Interface tests
    print("Interface Tests:")
    print("-" * 40)

    if torch_available:
        ok, msg = test_torch_interface()
        print_status("PyTorch interface", ok, msg, indent=1)

        ok, msg = test_torch_autograd()
        print_status("PyTorch autograd", ok, msg, indent=1)
    else:
        print_status("PyTorch interface", False, "PyTorch not installed", indent=1)
        print_status("PyTorch autograd", False, "PyTorch not installed", indent=1)

    ok, version = check_import("jax")
    if ok:
        ok, msg = test_jax_interface()
        print_status("JAX interface", ok, msg, indent=1)
    else:
        print_status("JAX interface", False, "JAX not installed", indent=1)

    cupy_ok, _ = check_import("cupy")
    if cupy_ok:
        ok, msg = test_cupy_interface()
        print_status("CuPy interface", ok, msg, indent=1)
    else:
        print_status("CuPy interface", False, "CuPy not installed", indent=1)

    print()
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        prog="python -m moreau",
        description="Moreau solver diagnostics and utilities",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check command
    check_parser = subparsers.add_parser("check", help="Run diagnostic checks")
    check_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.command == "check":
        run_check(verbose=args.verbose)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
