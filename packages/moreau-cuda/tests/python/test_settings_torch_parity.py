"""Regression test for #183: torch wrapper IPMSettings forwarding parity.

The torch_wrapper used to maintain its own copy of `_settings_to_cuda`
that silently dropped reduced_tol_*, max_lu_nnz, yolo, yolo_num_iters,
and device_id. This test asserts that every IPMSettings field round-trips
through both the unified converter and the torch wrapper converter so
that future divergence is caught immediately.
"""

import pytest

moreau_cuda = pytest.importorskip("moreau_cuda", reason="moreau_cuda not installed")

from moreau import Settings, IPMSettings


def _ipm_settings_with_distinctive_values() -> IPMSettings:
    """Construct an IPMSettings with non-default values on every field
    that the converter touches."""
    return IPMSettings(
        tol_gap_abs=1.5e-7,
        tol_gap_rel=2.5e-7,
        tol_feas=3.5e-7,
        tol_infeas_abs=4.5e-7,
        tol_infeas_rel=5.5e-7,
        tol_ktratio=6.5e-7,
        max_step_fraction=0.987,
        equilibrate_enable=False,
        reduced_tol_gap_abs=1.5e-5,
        reduced_tol_gap_rel=2.5e-5,
        reduced_tol_feas=3.5e-5,
        reduced_tol_infeas_abs=4.5e-5,
        reduced_tol_infeas_rel=5.5e-5,
        reduced_tol_ktratio=6.5e-5,
        diff_method="smoothed",
        diff_smoothing_mu=1.234e-3,
        diff_smoothing_step_factor=0.42,
        max_lu_nnz=987654,
        cudss_ir_steps=3,
        cudss_pivot_enable=False,
    )


def _cuda_settings_to_dict(cuda_settings) -> dict:
    """Pull every readable attribute off a _CudaSettings (top-level + ipm)."""
    out = {}
    for name in dir(cuda_settings):
        if name.startswith("_"):
            continue
        try:
            val = getattr(cuda_settings, name)
        except Exception:
            continue
        if callable(val):
            continue
        if name == "ipm":
            for ipm_name in dir(val):
                if ipm_name.startswith("_"):
                    continue
                try:
                    iv = getattr(val, ipm_name)
                except Exception:
                    continue
                if callable(iv):
                    continue
                out[f"ipm.{ipm_name}"] = iv
        else:
            out[name] = val
    return out


def test_torch_wrapper_settings_parity_with_unified():
    """torch_wrapper._settings_to_cuda and the unified _settings_to_cuda
    must produce identical _CudaSettings for the same input."""
    from moreau_cuda import _settings_to_cuda as unified_converter
    from moreau_cuda.torch_wrapper import _settings_to_cuda as torch_converter

    settings = Settings(
        max_iter=42,
        time_limit=12.5,
        verbose=True,
        device_id=1,
        yolo=True,
        yolo_num_iters=7,
        ipm_settings=_ipm_settings_with_distinctive_values(),
    )

    unified_dict = _cuda_settings_to_dict(unified_converter(settings))
    torch_dict = _cuda_settings_to_dict(torch_converter(settings))

    # Same set of keys
    assert set(unified_dict.keys()) == set(torch_dict.keys()), (
        f"Field set differs: unified-only={set(unified_dict)-set(torch_dict)}, "
        f"torch-only={set(torch_dict)-set(unified_dict)}"
    )

    # Same values per key
    mismatches = {
        k: (unified_dict[k], torch_dict[k])
        for k in unified_dict
        if unified_dict[k] != torch_dict[k]
    }
    assert not mismatches, f"Settings drift: {mismatches}"
