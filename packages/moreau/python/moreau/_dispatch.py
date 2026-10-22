"""Solver-selection and direct cone dispatch helpers for moreau (private).

Resolves the solver and device, and validates that a direct cone
configuration is supported on the requested device.
"""

from ._types import Settings, SolverType
from ._backend import (
    _choose_device,
    available_devices,
    device_available,
    device_error,
)


def _resolve_settings_and_device(
    settings,
    *,
    n: int,
    nnz_A: int,
    batch_size: int,
):
    """Resolve settings/device for Solver and CompiledSolver init.

    Defaults settings, resolves solver='auto' (active-set forces
    device='cpu'), resolves device='auto' to a concrete device via
    `_choose_device`, then validates the result. Single source of
    truth — the two top-level __init__s only differ in how they
    measure `nnz_A` / `batch_size`.

    Returns:
        (settings, device) — `settings` is a fresh copy with `solver`
        resolved; `device` is a concrete device name (never 'auto').

    Raises:
        RuntimeError: If the resolved device isn't available.
    """
    if settings is None:
        settings = Settings()
    settings, device = _resolve_solver_type(settings, settings.device)
    if device == "auto":
        device = _choose_device(n, nnz_A, batch_size=batch_size)
    if not device_available(device):
        error = device_error(device) or ""
        raise RuntimeError(
            f"Device '{device}' is not available. {error}\n"
            f"Available devices: {available_devices()}"
        )
    return settings, device


# Direct cone kinds with native CUDA forward + backward support.
_CUDA_SUPPORTED_DIR_CONE_KINDS = frozenset(
    {
        "nonneg",
        "soc",
        "psd_triangle",
        "exp",
        "power",
        "gen_power",
    }
)


def _resolve_solver_type(settings, device):
    """Resolve solver='auto' to the IPM; active-set (opt-in only) forces device='cpu'.

    Returns a copy of settings with solver resolved, and the device.
    """
    settings = settings.model_copy(deep=True)
    if settings.solver == SolverType.AUTO or settings.solver == "auto":
        settings.solver = SolverType.IPM
    if settings.solver == SolverType.ACTIVE_SET:
        device = "cpu"
    return settings, device


def _require_dir_cones_compatible(cones, settings=None) -> None:
    """Validate that a direct configuration is supported.

    CompiledSolver + CPU: routes through DirectXSolverCpu (batched via
    per-problem loop, or single-problem direct invocation).
    CompiledSolver + CUDA: routes through the device CompiledSolver
    which already threads dir_cones. Forward and backward are supported
    natively on CUDA for all direct cone kinds (nonneg, SOC, PSD,
    exp, power, gen_power) via the IFT-direct augmented-system path.
    """
    dir_cones = getattr(cones, "dir_cones", None)
    if not dir_cones:
        return

    device = getattr(settings, "device", None) if settings is not None else None
    enable_grad = getattr(settings, "enable_grad", False) if settings is not None else False
    _ = enable_grad  # All CUDA-supported kinds also support enable_grad.

    if device == "cuda":
        if not _all_dir_cones_cuda_supported(cones):
            raise NotImplementedError(
                "CUDA direct cones support kinds " f"{sorted(_CUDA_SUPPORTED_DIR_CONE_KINDS)}."
            )


def _has_dir_cones(cones) -> bool:
    """True if the cones object carries any direct cone specs."""
    dir_cones = getattr(cones, "dir_cones", None)
    return bool(dir_cones)


def _all_dir_cones_cuda_supported(cones) -> bool:
    """True if every x-cone kind is supported on the CUDA backend.

    CUDA forward + backward is wired for: nonneg, SOC (any dim),
    psd_triangle, exp, power, and gen_power.
    """
    dir_cones = getattr(cones, "dir_cones", None) or []
    for xc in dir_cones:
        kind = getattr(xc, "kind", None)
        if kind not in _CUDA_SUPPORTED_DIR_CONE_KINDS:
            return False
    return True


def _require_cpu_device_for_dir_cones(settings, cones) -> None:
    """Validate direct device + enable_grad support.

    CUDA supports nonneg, SOC, psd_triangle, exp, power, and gen_power
    direct forward and backward via the IFT-direct augmented-system path.
    """
    if settings is None:
        return
    device = getattr(settings, "device", None)
    if device in (None, "auto", "cpu"):
        return
    if device == "cuda" and _all_dir_cones_cuda_supported(cones):
        return
    raise NotImplementedError(
        f"Direct cones (Cones.dir_cones) are not yet supported on "
        f"device={device!r} with the current cone mix. CUDA supports: "
        f"{sorted(_CUDA_SUPPORTED_DIR_CONE_KINDS)}."
    )
