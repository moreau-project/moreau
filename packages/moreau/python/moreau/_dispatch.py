"""Solver-selection and direct-x cone dispatch helpers for moreau (private).

Decides between active-set and IPM, and validates that a direct-x cone
configuration is supported on the requested device.
"""

import warnings

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
    m: int,
    cones,
    nnz_P: int,
    nnz_A: int,
    batch_size: int,
):
    """Resolve settings/device for Solver and CompiledSolver init.

    Defaults settings, resolves solver='auto' (may force device='cpu'
    for active-set), resolves device='auto' to a concrete device via
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
    settings, device = _resolve_solver_type(
        settings,
        n,
        m,
        cones,
        settings.device,
        nnz_P=nnz_P,
    )
    if device == "auto":
        device = _choose_device(n, nnz_A, batch_size=batch_size)
    if not device_available(device):
        error = device_error(device) or ""
        raise RuntimeError(
            f"Device '{device}' is not available. {error}\n"
            f"Available devices: {available_devices()}"
        )
    return settings, device


# Direct-x cone kinds with native CUDA forward + backward support.
_CUDA_SUPPORTED_XCONE_KINDS = frozenset(
    {
        "nonneg",
        "soc",
        "psd_triangle",
        "exp",
        "power",
        "gen_power",
    }
)

_auto_active_set_with_grad_warning_issued = False


def _resolve_solver_type(settings, n, m, cones, device, nnz_P=None):
    """Resolve solver='auto' to active-set or IPM.

    Returns a (possibly copied) settings object with solver resolved,
    and the device (forced to 'cpu' for active-set).

    Args:
        nnz_P: Number of nonzeros in P. If 0, the problem is an LP and
            active-set is not used. If None, assumed nonzero (QP).

    When ``solver='auto'`` resolves to ACTIVE_SET for a problem with
    ``enable_grad=True``, a one-time UserWarning is emitted. Active-set
    produces non-smooth derivatives whose quality changes discontinuously
    at the n/m threshold below. For smooth gradients, pass ``solver='ipm'``
    explicitly.
    """
    if settings.solver == SolverType.AUTO or settings.solver == "auto":
        is_qp_only = (
            cones.num_exp_cones == 0
            and len(cones.power_alphas) == 0
            and len(cones.so_cone_dims) == 0
            and len(getattr(cones, "gen_power_cone_params", [])) == 0
            and len(getattr(cones, "psd_dims", [])) == 0
        )
        # Active-set CPU solver only knows zero+nonneg slack cones; direct-x
        # cones (any kind) require IPM. Routing direct-x problems through
        # active-set silently drops `x_cones`, returning a wrong (constraint-
        # violating) solution and zero z_x.
        has_x_cones = bool(getattr(cones, "x_cones", None))
        has_quadratic = nnz_P is None or nnz_P > 0
        m_limit = max(500, 2 * n)
        if (
            is_qp_only
            and not has_x_cones
            and has_quadratic
            and n <= 500
            and m > 0
            and m <= m_limit
            and device != "cuda"
        ):
            settings = settings.model_copy(deep=True)
            settings.solver = SolverType.ACTIVE_SET
            if settings.enable_grad:
                _warn_auto_active_set_with_grad()
        else:
            settings = settings.model_copy(deep=True)
            settings.solver = SolverType.IPM

    if settings.solver == SolverType.ACTIVE_SET:
        device = "cpu"

    return settings, device


def _warn_auto_active_set_with_grad():
    """Warn once when solver='auto' picks ACTIVE_SET with gradients enabled."""
    global _auto_active_set_with_grad_warning_issued
    if _auto_active_set_with_grad_warning_issued:
        return
    _auto_active_set_with_grad_warning_issued = True
    warnings.warn(
        "solver='auto' resolved to ACTIVE_SET for a problem with "
        "enable_grad=True. Active-set gradients are non-smooth "
        "(combinatorial), and gradient quality changes discontinuously when "
        "the problem crosses the n=500 / m=max(500, 2n) threshold (above "
        "which auto picks IPM, with smooth gradients). For smooth gradients "
        "regardless of size, pass solver='ipm' explicitly. (#185)",
        UserWarning,
        stacklevel=3,
    )


def _require_x_cones_compatible(cones, settings=None) -> None:
    """Validate that a direct-x configuration is supported.

    CompiledSolver + CPU: routes through DirectXSolverCpu (batched via
    per-problem loop, or single-problem direct invocation).
    CompiledSolver + CUDA: routes through the device CompiledSolver
    which already threads x_cones. Forward and backward are supported
    natively on CUDA for all direct-x cone kinds (nonneg, SOC, PSD,
    exp, power, gen_power) via the IFT-direct augmented-system path.
    """
    x_cones = getattr(cones, "x_cones", None)
    if not x_cones:
        return

    device = getattr(settings, "device", None) if settings is not None else None
    enable_grad = getattr(settings, "enable_grad", False) if settings is not None else False
    _ = enable_grad  # All CUDA-supported kinds also support enable_grad.

    if device == "cuda":
        if not _all_x_cones_cuda_supported(cones):
            raise NotImplementedError(
                "CUDA direct-x cones support kinds " f"{sorted(_CUDA_SUPPORTED_XCONE_KINDS)}."
            )


def _has_x_cones(cones) -> bool:
    """True if the cones object carries any direct-x cone specs."""
    x_cones = getattr(cones, "x_cones", None)
    return bool(x_cones)


def _all_x_cones_cuda_supported(cones) -> bool:
    """True if every x-cone kind is supported on the CUDA backend.

    CUDA forward + backward is wired for: nonneg, SOC (any dim),
    psd_triangle, exp, power, and gen_power.
    """
    x_cones = getattr(cones, "x_cones", None) or []
    for xc in x_cones:
        kind = getattr(xc, "kind", None)
        if kind not in _CUDA_SUPPORTED_XCONE_KINDS:
            return False
    return True


def _require_cpu_device_for_x_cones(settings, cones) -> None:
    """Validate direct-x device + enable_grad support.

    CUDA supports nonneg, SOC, psd_triangle, exp, power, and gen_power
    direct-x forward and backward via the IFT-direct augmented-system path.
    """
    if settings is None:
        return
    device = getattr(settings, "device", None)
    if device in (None, "auto", "cpu"):
        return
    if device == "cuda" and _all_x_cones_cuda_supported(cones):
        return
    raise NotImplementedError(
        f"Direct-x cones (Cones.x_cones) are not yet supported on "
        f"device={device!r} with the current cone mix. CUDA supports: "
        f"{sorted(_CUDA_SUPPORTED_XCONE_KINDS)}."
    )
