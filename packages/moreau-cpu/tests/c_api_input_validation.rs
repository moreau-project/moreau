//! Integration tests for the Moreau CPU C API input validation.
//!
//! Previously, malformed cone inputs (negative SOC dim, alpha outside (0,1),
//! etc.) reached `convert_cones` and caused
//! OOM aborts or undefined behaviour downstream. These tests exercise the
//! C entry points exactly as a raw C/Julia caller would, and verify they
//! reject malformed input with `MOREAU_ERROR_INVALID_ARGUMENT`.
//!
//! Bidirectional check: with the validation patch removed, the negative
//! tests below would crash the test process (OOM panic from
//! `vec![T::zero(); usize::MAX]` for SOC dim, or undefined behaviour in
//! cone projection for out-of-range alphas) instead of returning the
//! expected `MOREAU_ERROR_INVALID_ARGUMENT`.
//!
//! Calls go through the same `#[no_mangle] extern "C"` entry points the C
//! header advertises; we just dispatch via the Rust path because Rust
//! integration tests link against the rlib (which `--gc-sections`-strips
//! the FFI symbols by default).

#![allow(non_snake_case)]

use std::ptr;

use moreau::c_api::{
    moreau_last_error, moreau_settings_default, moreau_solver_create, moreau_solver_destroy,
    MoreauConesT, MoreauSettingsT, MoreauSolverInner,
};

const MOREAU_OK: i32 = 0;
const MOREAU_ERROR_INVALID_ARGUMENT: i32 = 1;

// -- Test fixture -------------------------------------------------------------

const N: i64 = 2;

fn default_settings() -> MoreauSettingsT {
    let mut s: MoreauSettingsT = unsafe { std::mem::zeroed() };
    moreau_settings_default(&mut s);
    s.verbose = 0;
    s
}

fn empty_cones() -> MoreauConesT {
    MoreauConesT {
        num_zero_cones: 0,
        num_nonneg_cones: 0,
        num_soc_cones: 0,
        soc_dims: ptr::null(),
        num_exp_cones: 0,
        num_power_cones: 0,
        power_alphas: ptr::null(),
    }
}

/// Try to create a solver with `cones`; returns the C error code. Cleans up
/// any successfully-created solver.
///
/// Builds a 2-variable identity-shaped P (so `n = N = 2`) and an A with `m`
/// rows whose nonzeros pattern is one entry in column 0 per row (so the
/// solver can build a valid CSR even when the cones span >1 row).
fn try_create(cones: &MoreauConesT, n: i64, m: i64) -> i32 {
    let p_ro: [i64; 3] = [0, 1, 2];
    let p_ci: [i64; 2] = [0, 1];

    // A: each row has exactly one nonzero at column 0.
    let a_ro: Vec<i64> = (0..=m).collect();
    let a_ci: Vec<i64> = vec![0; m as usize];
    let nnz_A = m;

    let settings = default_settings();
    let mut solver: *mut MoreauSolverInner = ptr::null_mut();
    let code = moreau_solver_create(
        &mut solver,
        n,
        m,
        p_ro.as_ptr(),
        p_ci.as_ptr(),
        2,
        a_ro.as_ptr(),
        if nnz_A > 0 {
            a_ci.as_ptr()
        } else {
            ptr::null()
        },
        nnz_A,
        cones,
        &settings,
    );
    if !solver.is_null() {
        moreau_solver_destroy(solver);
    }
    code
}

fn last_error_str() -> Option<String> {
    let p = moreau_last_error();
    if p.is_null() {
        return None;
    }
    let cstr = unsafe { std::ffi::CStr::from_ptr(p) };
    Some(cstr.to_string_lossy().into_owned())
}

// -- SOC dim validation -------------------------------------------------------

#[test]
fn negative_soc_dim_rejected() {
    let dims: [i64; 1] = [-3];
    let cones = MoreauConesT {
        num_soc_cones: 1,
        soc_dims: dims.as_ptr(),
        ..empty_cones()
    };
    let code = try_create(&cones, N, 3);
    assert_eq!(
        code,
        MOREAU_ERROR_INVALID_ARGUMENT,
        "negative SOC dim must be rejected; last_error: {:?}",
        last_error_str()
    );
    let err = last_error_str().unwrap_or_default();
    assert!(
        err.contains("soc_dims") || err.contains("SOC"),
        "error should mention soc_dims, got: {}",
        err
    );
}

#[test]
fn soc_dim_one_rejected() {
    // SOC requires dim >= 2; dim=1 is invalid.
    let dims: [i64; 1] = [1];
    let cones = MoreauConesT {
        num_soc_cones: 1,
        soc_dims: dims.as_ptr(),
        ..empty_cones()
    };
    let code = try_create(&cones, N, 1);
    assert_eq!(
        code,
        MOREAU_ERROR_INVALID_ARGUMENT,
        "SOC dim=1 must be rejected; last_error: {:?}",
        last_error_str()
    );
}

#[test]
fn soc_dims_null_with_count_rejected() {
    let cones = MoreauConesT {
        num_soc_cones: 1,
        soc_dims: ptr::null(),
        ..empty_cones()
    };
    let code = try_create(&cones, N, 0);
    assert_eq!(code, MOREAU_ERROR_INVALID_ARGUMENT);
}

#[test]
fn negative_num_soc_cones_rejected() {
    let cones = MoreauConesT {
        num_soc_cones: -1,
        ..empty_cones()
    };
    let code = try_create(&cones, N, 0);
    assert_eq!(code, MOREAU_ERROR_INVALID_ARGUMENT);
}

// -- Power cone alpha validation ----------------------------------------------

#[test]
fn alpha_zero_rejected() {
    let alphas: [f64; 1] = [0.0];
    let cones = MoreauConesT {
        num_power_cones: 1,
        power_alphas: alphas.as_ptr(),
        ..empty_cones()
    };
    let code = try_create(&cones, N, 3);
    assert_eq!(
        code,
        MOREAU_ERROR_INVALID_ARGUMENT,
        "alpha=0 must be rejected; last_error: {:?}",
        last_error_str()
    );
}

#[test]
fn alpha_one_rejected() {
    let alphas: [f64; 1] = [1.0];
    let cones = MoreauConesT {
        num_power_cones: 1,
        power_alphas: alphas.as_ptr(),
        ..empty_cones()
    };
    let code = try_create(&cones, N, 3);
    assert_eq!(code, MOREAU_ERROR_INVALID_ARGUMENT);
}

#[test]
fn alpha_negative_rejected() {
    let alphas: [f64; 1] = [-0.5];
    let cones = MoreauConesT {
        num_power_cones: 1,
        power_alphas: alphas.as_ptr(),
        ..empty_cones()
    };
    let code = try_create(&cones, N, 3);
    assert_eq!(code, MOREAU_ERROR_INVALID_ARGUMENT);
}

#[test]
fn alpha_nan_rejected() {
    let alphas: [f64; 1] = [f64::NAN];
    let cones = MoreauConesT {
        num_power_cones: 1,
        power_alphas: alphas.as_ptr(),
        ..empty_cones()
    };
    let code = try_create(&cones, N, 3);
    assert_eq!(
        code,
        MOREAU_ERROR_INVALID_ARGUMENT,
        "alpha=NaN must be rejected; last_error: {:?}",
        last_error_str()
    );
}

#[test]
fn alpha_inf_rejected() {
    let alphas: [f64; 1] = [f64::INFINITY];
    let cones = MoreauConesT {
        num_power_cones: 1,
        power_alphas: alphas.as_ptr(),
        ..empty_cones()
    };
    let code = try_create(&cones, N, 3);
    assert_eq!(code, MOREAU_ERROR_INVALID_ARGUMENT);
}

#[test]
fn power_alphas_null_with_count_rejected() {
    let cones = MoreauConesT {
        num_power_cones: 1,
        power_alphas: ptr::null(),
        ..empty_cones()
    };
    let code = try_create(&cones, N, 0);
    assert_eq!(code, MOREAU_ERROR_INVALID_ARGUMENT);
}

// -- Other count validations --------------------------------------------------

#[test]
fn negative_num_zero_cones_rejected() {
    let cones = MoreauConesT {
        num_zero_cones: -1,
        ..empty_cones()
    };
    let code = try_create(&cones, N, 0);
    assert_eq!(code, MOREAU_ERROR_INVALID_ARGUMENT);
}

#[test]
fn negative_num_nonneg_cones_rejected() {
    let cones = MoreauConesT {
        num_nonneg_cones: -1,
        ..empty_cones()
    };
    let code = try_create(&cones, N, 0);
    assert_eq!(code, MOREAU_ERROR_INVALID_ARGUMENT);
}

#[test]
fn negative_num_exp_cones_rejected() {
    let cones = MoreauConesT {
        num_exp_cones: -1,
        ..empty_cones()
    };
    let code = try_create(&cones, N, 0);
    assert_eq!(code, MOREAU_ERROR_INVALID_ARGUMENT);
}

#[test]
fn negative_num_power_cones_rejected() {
    let cones = MoreauConesT {
        num_power_cones: -1,
        ..empty_cones()
    };
    let code = try_create(&cones, N, 0);
    assert_eq!(code, MOREAU_ERROR_INVALID_ARGUMENT);
}

// -- CSR offset / index sanity (i64_slice_to_usize_vec) -----------------------

#[test]
fn negative_csr_offset_rejected() {
    // Build a 1x1 problem with a malformed P_row_offsets (negative entry).
    let bad_p_ro: [i64; 2] = [0, -5]; // row offsets must be non-negative
    let p_ci: [i64; 0] = [];
    let a_ro: [i64; 1] = [0];
    let cones = empty_cones();
    let settings = default_settings();
    let mut solver: *mut MoreauSolverInner = ptr::null_mut();
    let code = moreau_solver_create(
        &mut solver,
        1,
        0,
        bad_p_ro.as_ptr(),
        p_ci.as_ptr(),
        0,
        a_ro.as_ptr(),
        ptr::null(),
        0,
        &cones,
        &settings,
    );
    if !solver.is_null() {
        moreau_solver_destroy(solver);
    }
    assert_eq!(
        code,
        MOREAU_ERROR_INVALID_ARGUMENT,
        "negative CSR row offset must be rejected; last_error: {:?}",
        last_error_str()
    );
}

// -- Positive controls: valid input still works ------------------------------

#[test]
fn valid_inputs_accepted() {
    // Valid SOC dim (3) and valid alpha (0.5)
    let dims: [i64; 1] = [3];
    let alphas: [f64; 1] = [0.5];
    let cones = MoreauConesT {
        num_zero_cones: 0,
        num_nonneg_cones: 1,
        num_soc_cones: 1,
        soc_dims: dims.as_ptr(),
        num_exp_cones: 0,
        num_power_cones: 1,
        power_alphas: alphas.as_ptr(),
    };
    // m = 1 (nonneg) + 3 (SOC) + 3 (power) = 7
    let code = try_create(&cones, N, 7);
    assert_eq!(
        code,
        MOREAU_OK,
        "valid input must be accepted; last_error: {:?}",
        last_error_str()
    );
}

#[test]
fn alpha_just_inside_unit_interval_accepted() {
    // Exercise edge-near-boundary values that should still pass.
    for &a in &[1e-12_f64, 0.5, 1.0 - 1e-12] {
        let alphas = [a];
        let cones = MoreauConesT {
            num_power_cones: 1,
            power_alphas: alphas.as_ptr(),
            ..empty_cones()
        };
        let code = try_create(&cones, N, 3);
        assert_eq!(
            code,
            MOREAU_OK,
            "alpha={} should be accepted; last_error: {:?}",
            a,
            last_error_str()
        );
    }
}
