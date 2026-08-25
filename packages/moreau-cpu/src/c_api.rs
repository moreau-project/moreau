//! C API for Moreau CPU solver
//!
//! Implements the functions declared in `include/moreau.h`.
//! Gated behind the `c-api` feature.

#![allow(non_snake_case)]
#![allow(private_interfaces)]
#![allow(dead_code)]

use crate::solver::implementations::default::{
    CompiledSolver, DefaultSettings, DefaultSolution, IPMSettings, SolverError, UpstreamGradients,
};
use crate::solver::{SolverStatus, SupportedConeT};
use std::cell::RefCell;
use std::panic;
use std::ptr;
use std::slice;

// ---------------------------------------------------------------------------
// Error codes (must match moreau.h moreau_error_t)
// ---------------------------------------------------------------------------
const MOREAU_OK: i32 = 0;
const MOREAU_ERROR_INVALID_ARGUMENT: i32 = 1;
const MOREAU_ERROR_NOT_SETUP: i32 = 2;
const MOREAU_ERROR_NUMERICAL: i32 = 4;
const MOREAU_ERROR_OUT_OF_MEMORY: i32 = 5;
#[allow(dead_code)]
const MOREAU_ERROR_CUDA: i32 = 6;
const MOREAU_ERROR_INTERNAL: i32 = 99;

// ---------------------------------------------------------------------------
// Thread-local last error
// ---------------------------------------------------------------------------
thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = RefCell::new(None);
}

fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = Some(msg);
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------
pub struct MoreauSolverInner {
    solver: CompiledSolver<f64>,
    solutions: Vec<DefaultSolution<f64>>,
    n: usize,
    m: usize,
    nnz_P: usize,
    nnz_A: usize,
    batch_size: usize,
    enable_grad: bool,
    is_setup: bool,
    has_solution: bool,
}

// ---------------------------------------------------------------------------
// C types (must match moreau.h)
//
// `pub` so Rust integration tests can construct them and call the C-ABI
// entry points directly (the symbols themselves are exported via
// `#[no_mangle]` for FFI, but Rust integration tests link against the
// crate's rlib and need the Rust-level path).
// ---------------------------------------------------------------------------
#[repr(C)]
pub struct MoreauConesT {
    pub num_zero_cones: i64,
    pub num_nonneg_cones: i64,
    pub num_soc_cones: i64,
    pub soc_dims: *const i64,
    pub num_exp_cones: i64,
    pub num_power_cones: i64,
    pub power_alphas: *const f64,
}

#[repr(C)]
pub struct MoreauIpmSettingsT {
    pub tol_gap_abs: f64,
    pub tol_gap_rel: f64,
    pub tol_feas: f64,
    pub tol_infeas_abs: f64,
    pub tol_infeas_rel: f64,
    pub tol_ktratio: f64,
    pub reduced_tol_gap_abs: f64,
    pub reduced_tol_gap_rel: f64,
    pub reduced_tol_feas: f64,
    pub reduced_tol_infeas_abs: f64,
    pub reduced_tol_infeas_rel: f64,
    pub reduced_tol_ktratio: f64,
    pub equilibrate_enable: i32,
    pub equilibrate_max_iter: i32,
    pub equilibrate_min_scaling: f64,
    pub equilibrate_max_scaling: f64,
    pub max_step_fraction: f64,
    pub linesearch_backtrack_step: f64,
    pub min_switch_step_length: f64,
    pub min_terminate_step_length: f64,
    pub direct_solve_method: i32, // moreau_direct_solve_method_t
    pub static_regularization_enable: i32,
    pub static_regularization_constant: f64,
    pub static_regularization_proportional: f64,
    pub dynamic_regularization_enable: i32,
    pub dynamic_regularization_eps: f64,
    pub dynamic_regularization_delta: f64,
}

#[repr(C)]
pub struct MoreauSettingsT {
    pub batch_size: i64,
    pub max_iter: u32,
    pub time_limit: f64,
    pub verbose: i32,
    pub enable_grad: i32,
    pub ipm: MoreauIpmSettingsT,
}

#[repr(C)]
pub struct MoreauSolutionT {
    pub x: *const f64,
    pub z: *const f64,
    pub s: *const f64,
    pub status: i32, // moreau_status_t
    pub obj_val: f64,
    pub obj_val_dual: f64,
    pub solve_time: f64,
    pub iterations: i32,
    pub r_prim: f64,
    pub r_dual: f64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn solver_status_to_c(status: SolverStatus) -> i32 {
    match status {
        SolverStatus::Unsolved => 0,
        SolverStatus::Solved => 1,
        SolverStatus::PrimalInfeasible => 2,
        SolverStatus::DualInfeasible => 3,
        SolverStatus::AlmostSolved => 4,
        SolverStatus::AlmostPrimalInfeasible => 5,
        SolverStatus::AlmostDualInfeasible => 6,
        SolverStatus::MaxIterations => 7,
        SolverStatus::MaxTime => 8,
        SolverStatus::NumericalError => 9,
        SolverStatus::InsufficientProgress => 10,
        SolverStatus::CallbackTerminated => 11,
    }
}

/// Convert a `MoreauConesT` (raw C-input) to the Rust cone enum vector,
/// validating every field that comes from user memory.
///
/// Validated invariants (defense-in-depth — the Python wrapper validates
/// these too, but raw C/Julia callers reach this code directly):
///
/// | Field                  | Source            | Valid range                                      |
/// |------------------------|-------------------|--------------------------------------------------|
/// | `num_zero_cones`       | i64 by value      | `>= 0` (0 means "no zero cone added")            |
/// | `num_nonneg_cones`     | i64 by value      | `>= 0`                                           |
/// | `num_soc_cones`        | i64 by value      | `>= 0`                                           |
/// | `soc_dims[i]`          | i64 in user buf   | `>= 2` (SOC requires dim >= 2)                   |
/// | `num_exp_cones`        | i64 by value      | `>= 0`                                           |
/// | `num_power_cones`      | i64 by value      | `>= 0`                                           |
/// | `power_alphas[i]`      | f64 in user buf   | finite and in `(0, 1)`                           |
///
/// Returns `Err` with a human-readable message on the first invalid field.
fn convert_cones(cones: &MoreauConesT) -> Result<Vec<SupportedConeT<f64>>, String> {
    let mut result = Vec::new();

    if cones.num_zero_cones < 0 {
        return Err(format!(
            "num_zero_cones must be >= 0, got {}",
            cones.num_zero_cones
        ));
    }
    if cones.num_nonneg_cones < 0 {
        return Err(format!(
            "num_nonneg_cones must be >= 0, got {}",
            cones.num_nonneg_cones
        ));
    }
    if cones.num_soc_cones < 0 {
        return Err(format!(
            "num_soc_cones must be >= 0, got {}",
            cones.num_soc_cones
        ));
    }
    if cones.num_exp_cones < 0 {
        return Err(format!(
            "num_exp_cones must be >= 0, got {}",
            cones.num_exp_cones
        ));
    }
    if cones.num_power_cones < 0 {
        return Err(format!(
            "num_power_cones must be >= 0, got {}",
            cones.num_power_cones
        ));
    }

    if cones.num_zero_cones > 0 {
        result.push(SupportedConeT::ZeroConeT(cones.num_zero_cones as usize));
    }
    if cones.num_nonneg_cones > 0 {
        result.push(SupportedConeT::NonnegativeConeT(
            cones.num_nonneg_cones as usize,
        ));
    }
    if cones.num_soc_cones > 0 {
        if cones.soc_dims.is_null() {
            return Err("soc_dims is NULL but num_soc_cones > 0".into());
        }
        // SAFETY: null-checked; caller must pass `num_soc_cones` i64s.
        let dims = unsafe { slice::from_raw_parts(cones.soc_dims, cones.num_soc_cones as usize) };
        for (i, &dim) in dims.iter().enumerate() {
            if dim < 2 {
                return Err(format!(
                    "soc_dims[{}] = {} must be >= 2 (SOC requires dim >= 2)",
                    i, dim
                ));
            }
            result.push(SupportedConeT::SecondOrderConeT(dim as usize));
        }
    }
    for _ in 0..cones.num_exp_cones {
        result.push(SupportedConeT::ExponentialConeT());
    }
    if cones.num_power_cones > 0 {
        if cones.power_alphas.is_null() {
            return Err("power_alphas is NULL but num_power_cones > 0".into());
        }
        // SAFETY: null-checked; caller must pass `num_power_cones` f64s.
        let alphas =
            unsafe { slice::from_raw_parts(cones.power_alphas, cones.num_power_cones as usize) };
        for (i, &alpha) in alphas.iter().enumerate() {
            if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
                return Err(format!(
                    "power_alphas[{}] = {} must be finite and in (0, 1)",
                    i, alpha
                ));
            }
            result.push(SupportedConeT::PowerConeT(alpha));
        }
    }
    Ok(result)
}

fn convert_settings(s: &MoreauSettingsT) -> (DefaultSettings<f64>, bool) {
    let direct_solve_method = match s.ipm.direct_solve_method {
        0 => "auto".to_string(),
        1 => "qdldl".to_string(),
        2 => "faer".to_string(),
        _ => "auto".to_string(),
    };

    let ipm = IPMSettings {
        max_step_fraction: s.ipm.max_step_fraction,
        tol_gap_abs: s.ipm.tol_gap_abs,
        tol_gap_rel: s.ipm.tol_gap_rel,
        tol_feas: s.ipm.tol_feas,
        tol_infeas_abs: s.ipm.tol_infeas_abs,
        tol_infeas_rel: s.ipm.tol_infeas_rel,
        tol_ktratio: s.ipm.tol_ktratio,
        reduced_tol_gap_abs: s.ipm.reduced_tol_gap_abs,
        reduced_tol_gap_rel: s.ipm.reduced_tol_gap_rel,
        reduced_tol_feas: s.ipm.reduced_tol_feas,
        reduced_tol_infeas_abs: s.ipm.reduced_tol_infeas_abs,
        reduced_tol_infeas_rel: s.ipm.reduced_tol_infeas_rel,
        reduced_tol_ktratio: s.ipm.reduced_tol_ktratio,
        equilibrate_enable: s.ipm.equilibrate_enable != 0,
        equilibrate_max_iter: s.ipm.equilibrate_max_iter as u32,
        equilibrate_min_scaling: s.ipm.equilibrate_min_scaling,
        equilibrate_max_scaling: s.ipm.equilibrate_max_scaling,
        linesearch_backtrack_step: s.ipm.linesearch_backtrack_step,
        min_switch_step_length: s.ipm.min_switch_step_length,
        min_terminate_step_length: s.ipm.min_terminate_step_length,
        direct_solve_method,
        static_regularization_enable: s.ipm.static_regularization_enable != 0,
        static_regularization_constant: s.ipm.static_regularization_constant,
        static_regularization_proportional: s.ipm.static_regularization_proportional,
        dynamic_regularization_enable: s.ipm.dynamic_regularization_enable != 0,
        dynamic_regularization_eps: s.ipm.dynamic_regularization_eps,
        dynamic_regularization_delta: s.ipm.dynamic_regularization_delta,
        ..IPMSettings::default()
    };

    let settings = DefaultSettings {
        max_iter: s.max_iter,
        time_limit: s.time_limit,
        verbose: s.verbose != 0,
        enable_grad: s.enable_grad != 0,
        ipm,
        ..DefaultSettings::default()
    };

    (settings, s.enable_grad != 0)
}

/// Read an i64 slice from raw memory and copy to an owned `Vec<usize>`.
///
/// The previous version returned `&'static [usize]`, which was a lifetime
/// lie — the slice was actually borrowed from the caller's memory. We now
/// copy eagerly to avoid any unsoundness; callers were already calling
/// `to_vec()` on the slice immediately, so no perf regression.
///
/// Returns an error if any element is negative (an i64 cast directly to
/// usize would silently wrap, producing a huge index that triggers OOM
/// or memory corruption downstream).
unsafe fn i64_slice_to_usize_vec(ptr: *const i64, len: usize) -> Result<Vec<usize>, String> {
    // SAFETY: caller must pass a valid `ptr` to at least `len` i64s.
    assert_eq!(
        std::mem::size_of::<i64>(),
        std::mem::size_of::<usize>(),
        "C API requires 64-bit platform"
    );
    if len == 0 {
        return Ok(Vec::new());
    }
    let raw = slice::from_raw_parts(ptr, len);
    let mut out = Vec::with_capacity(len);
    for (i, &v) in raw.iter().enumerate() {
        if v < 0 {
            return Err(format!("index[{}] = {} must be >= 0", i, v));
        }
        out.push(v as usize);
    }
    Ok(out)
}

/// Wrap a closure that may panic, returning MOREAU_ERROR_INTERNAL on panic.
fn catch_panic<F: FnOnce() -> i32>(f: F) -> i32 {
    match panic::catch_unwind(panic::AssertUnwindSafe(f)) {
        Ok(code) => code,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else {
                "Unknown panic".to_string()
            };
            set_last_error(format!("Internal error (panic): {}", msg));
            MOREAU_ERROR_INTERNAL
        }
    }
}

/// Slice a raw pointer into a Vec of per-batch Vec<f64>.
/// Returns vec of empty vecs if dim == 0.
unsafe fn chunk_batched(ptr: *const f64, dim: usize, batch_size: usize) -> Vec<Vec<f64>> {
    // SAFETY: caller must pass a valid `ptr` to at least `batch_size * dim` f64s.
    if dim > 0 {
        slice::from_raw_parts(ptr, batch_size * dim)
            .chunks(dim)
            .map(|c| c.to_vec())
            .collect()
    } else {
        vec![vec![]; batch_size]
    }
}

/// Handle solve/solve_warm result: store solutions or classify the error.
fn handle_solve_result(
    inner: &mut MoreauSolverInner,
    result: Result<Vec<DefaultSolution<f64>>, SolverError>,
) -> i32 {
    match result {
        Ok(solutions) => {
            inner.solutions = solutions;
            inner.has_solution = true;
            MOREAU_OK
        }
        Err(e) => {
            set_last_error(format!("{}", e));
            MOREAU_ERROR_NUMERICAL
        }
    }
}

// ---------------------------------------------------------------------------
// C API functions
// ---------------------------------------------------------------------------

/// Return the library version string.
#[no_mangle]
pub extern "C" fn moreau_version() -> *const std::os::raw::c_char {
    // Use a static CStr to ensure the pointer is valid for the library's lifetime.
    static VERSION_CSTR: std::sync::LazyLock<std::ffi::CString> =
        std::sync::LazyLock::new(|| std::ffi::CString::new(crate::VERSION).unwrap());
    VERSION_CSTR.as_ptr()
}

/// Return a description of the last error (thread-local).
#[no_mangle]
pub extern "C" fn moreau_last_error() -> *const std::os::raw::c_char {
    thread_local! {
        static LAST_ERROR_CSTR: RefCell<Option<std::ffi::CString>> = RefCell::new(None);
    }
    LAST_ERROR.with(|e| {
        let err = e.borrow();
        match err.as_ref() {
            Some(msg) => {
                let cstr = std::ffi::CString::new(msg.clone()).unwrap_or_default();
                LAST_ERROR_CSTR.with(|c| {
                    let mut slot = c.borrow_mut();
                    *slot = Some(cstr);
                    slot.as_ref().unwrap().as_ptr()
                })
            }
            None => ptr::null(),
        }
    })
}

/// Fill settings with safe defaults.
#[no_mangle]
pub extern "C" fn moreau_settings_default(settings: *mut MoreauSettingsT) {
    if settings.is_null() {
        return;
    }
    // SAFETY: null-checked; caller must pass a valid MoreauSettingsT pointer.
    let s = unsafe { &mut *settings };
    s.batch_size = 1;
    s.max_iter = 200;
    s.time_limit = f64::INFINITY;
    s.verbose = 1;
    s.enable_grad = 0;

    let ipm = &mut s.ipm;
    ipm.tol_gap_abs = 1e-8;
    ipm.tol_gap_rel = 1e-8;
    ipm.tol_feas = 1e-8;
    ipm.tol_infeas_abs = 1e-8;
    ipm.tol_infeas_rel = 1e-8;
    ipm.tol_ktratio = 1e-6;
    ipm.reduced_tol_gap_abs = 5e-5;
    ipm.reduced_tol_gap_rel = 5e-5;
    ipm.reduced_tol_feas = 1e-4;
    ipm.reduced_tol_infeas_abs = 5e-12;
    ipm.reduced_tol_infeas_rel = 5e-5;
    ipm.reduced_tol_ktratio = 1e-4;
    ipm.equilibrate_enable = 1;
    ipm.equilibrate_max_iter = 10;
    ipm.equilibrate_min_scaling = 1e-4;
    ipm.equilibrate_max_scaling = 1e4;
    ipm.max_step_fraction = 0.99;
    ipm.linesearch_backtrack_step = 0.8;
    ipm.min_switch_step_length = 0.1;
    ipm.min_terminate_step_length = 1e-4;
    ipm.direct_solve_method = 0; // MOREAU_DIRECT_SOLVE_AUTO
    ipm.static_regularization_enable = 1;
    ipm.static_regularization_constant = 1e-8;
    ipm.static_regularization_proportional = f64::EPSILON * f64::EPSILON;
    ipm.dynamic_regularization_enable = 1;
    ipm.dynamic_regularization_eps = 1e-13;
    ipm.dynamic_regularization_delta = 2e-7;
}

/// Create a solver.
#[no_mangle]
pub extern "C" fn moreau_solver_create(
    solver_out: *mut *mut MoreauSolverInner,
    n: i64,
    m: i64,
    P_row_offsets: *const i64,
    P_col_indices: *const i64,
    nnz_P: i64,
    A_row_offsets: *const i64,
    A_col_indices: *const i64,
    nnz_A: i64,
    cones_ptr: *const MoreauConesT,
    settings_ptr: *const MoreauSettingsT,
) -> i32 {
    catch_panic(|| {
        clear_last_error();

        if solver_out.is_null() {
            set_last_error("solver_out is NULL".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        if n < 0 || m < 0 || nnz_P < 0 || nnz_A < 0 {
            set_last_error("Negative dimension".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        if P_row_offsets.is_null() || A_row_offsets.is_null() || cones_ptr.is_null() {
            set_last_error("NULL pointer for required argument".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        // Allow P_col_indices to be NULL if nnz_P == 0, same for A
        if nnz_P > 0 && P_col_indices.is_null() {
            set_last_error("P_col_indices is NULL but nnz_P > 0".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }
        if nnz_A > 0 && A_col_indices.is_null() {
            set_last_error("A_col_indices is NULL but nnz_A > 0".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        let n = n as usize;
        let m = m as usize;
        let nnz_P = nnz_P as usize;
        let nnz_A = nnz_A as usize;

        // Convert settings
        let (settings, enable_grad, batch_size) = if settings_ptr.is_null() {
            (DefaultSettings::default(), false, 1usize)
        } else {
            // SAFETY: null-checked; caller must pass a valid MoreauSettingsT pointer.
            let s = unsafe { &*settings_ptr };
            let (settings, enable_grad) = convert_settings(s);
            let bs = if s.batch_size < 1 {
                1
            } else {
                s.batch_size as usize
            };
            (settings, enable_grad, bs)
        };

        // Convert cones (validates every user-supplied field)
        // SAFETY: null-checked; caller must pass a valid MoreauConesT pointer.
        let cones_c = unsafe { &*cones_ptr };
        let cones = match convert_cones(cones_c) {
            Ok(c) => c,
            Err(e) => {
                set_last_error(format!("Invalid cones argument: {}", e));
                return MOREAU_ERROR_INVALID_ARGUMENT;
            }
        };

        // Copy i64 arrays to Vec<usize> (rejects negative entries that would wrap to huge indices)
        let p_ro = match unsafe { i64_slice_to_usize_vec(P_row_offsets, n + 1) } {
            Ok(v) => v,
            Err(e) => {
                set_last_error(format!("Invalid P_row_offsets: {}", e));
                return MOREAU_ERROR_INVALID_ARGUMENT;
            }
        };
        let p_ci = match unsafe { i64_slice_to_usize_vec(P_col_indices, nnz_P) } {
            Ok(v) => v,
            Err(e) => {
                set_last_error(format!("Invalid P_col_indices: {}", e));
                return MOREAU_ERROR_INVALID_ARGUMENT;
            }
        };
        let a_ro = match unsafe { i64_slice_to_usize_vec(A_row_offsets, m + 1) } {
            Ok(v) => v,
            Err(e) => {
                set_last_error(format!("Invalid A_row_offsets: {}", e));
                return MOREAU_ERROR_INVALID_ARGUMENT;
            }
        };
        let a_ci = match unsafe { i64_slice_to_usize_vec(A_col_indices, nnz_A) } {
            Ok(v) => v,
            Err(e) => {
                set_last_error(format!("Invalid A_col_indices: {}", e));
                return MOREAU_ERROR_INVALID_ARGUMENT;
            }
        };

        // Auto-detect thread count
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(batch_size);

        match CompiledSolver::new(
            n,
            m,
            &p_ro,
            &p_ci,
            &a_ro,
            &a_ci,
            &cones,
            settings,
            num_threads,
            enable_grad,
        ) {
            Ok(solver) => {
                let inner = Box::new(MoreauSolverInner {
                    solver,
                    solutions: Vec::new(),
                    n,
                    m,
                    nnz_P,
                    nnz_A,
                    batch_size,
                    enable_grad,
                    is_setup: false,
                    has_solution: false,
                });
                // SAFETY: null-checked; caller must pass a valid out-handle pointer.
                unsafe { *solver_out = Box::into_raw(inner) };
                MOREAU_OK
            }
            Err(e) => {
                set_last_error(format!("Failed to create solver: {}", e));
                MOREAU_ERROR_INTERNAL
            }
        }
    })
}

/// Set matrix values.
#[no_mangle]
pub extern "C" fn moreau_solver_setup(
    solver: *mut MoreauSolverInner,
    P_values: *const f64,
    P_count: i64,
    A_values: *const f64,
    A_count: i64,
) -> i32 {
    catch_panic(|| {
        clear_last_error();

        if solver.is_null() {
            set_last_error("solver is NULL".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: null-checked; caller must pass a live handle from the constructor.
        let inner = unsafe { &mut *solver };

        if P_values.is_null() && inner.nnz_P > 0 {
            set_last_error("P_values is NULL".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }
        if A_values.is_null() && inner.nnz_A > 0 {
            set_last_error("A_values is NULL".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        let p_count = P_count as usize;
        let a_count = A_count as usize;

        let p_shared = p_count == inner.nnz_P;
        let a_shared = a_count == inner.nnz_A;

        if !p_shared && p_count != inner.batch_size * inner.nnz_P {
            set_last_error(format!(
                "P_count ({}) must be nnz_P ({}) or batch_size * nnz_P ({})",
                p_count,
                inner.nnz_P,
                inner.batch_size * inner.nnz_P
            ));
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }
        if !a_shared && a_count != inner.batch_size * inner.nnz_A {
            set_last_error(format!(
                "A_count ({}) must be nnz_A ({}) or batch_size * nnz_A ({})",
                a_count,
                inner.nnz_A,
                inner.batch_size * inner.nnz_A
            ));
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        // Both shared or both per-batch (simplify logic)
        let shared = p_shared && a_shared;

        // SAFETY: P_values/A_values are null-checked above; caller must pass
        // pointers to at least the indicated element counts.
        if shared {
            let p_vals = if inner.nnz_P > 0 {
                unsafe { slice::from_raw_parts(P_values, inner.nnz_P) }
            } else {
                &[]
            };
            let a_vals = if inner.nnz_A > 0 {
                unsafe { slice::from_raw_parts(A_values, inner.nnz_A) }
            } else {
                &[]
            };
            inner.solver.setup_shared(p_vals, a_vals, inner.batch_size);
        } else {
            // Per-batch: chunk into Vec<Vec<f64>>
            let all_p = if inner.nnz_P > 0 {
                unsafe { slice::from_raw_parts(P_values, p_count) }
            } else {
                &[]
            };
            let all_a = if inner.nnz_A > 0 {
                unsafe { slice::from_raw_parts(A_values, a_count) }
            } else {
                &[]
            };

            let p_batch: Vec<Vec<f64>> = if p_shared {
                vec![all_p.to_vec(); inner.batch_size]
            } else {
                all_p.chunks(inner.nnz_P).map(|c| c.to_vec()).collect()
            };
            let a_batch: Vec<Vec<f64>> = if a_shared {
                vec![all_a.to_vec(); inner.batch_size]
            } else {
                all_a.chunks(inner.nnz_A).map(|c| c.to_vec()).collect()
            };

            if let Err(e) = inner.solver.setup(&p_batch, &a_batch) {
                set_last_error(e.to_string());
                return MOREAU_ERROR_INVALID_ARGUMENT;
            }
        }

        inner.is_setup = true;
        inner.has_solution = false;
        MOREAU_OK
    })
}

/// Solve.
#[no_mangle]
pub extern "C" fn moreau_solver_solve(
    solver: *mut MoreauSolverInner,
    q: *const f64,
    b: *const f64,
) -> i32 {
    catch_panic(|| {
        clear_last_error();

        if solver.is_null() {
            set_last_error("solver is NULL".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: null-checked; caller must pass a live handle from the constructor.
        let inner = unsafe { &mut *solver };

        if !inner.is_setup {
            set_last_error("solve() called before setup()".into());
            return MOREAU_ERROR_NOT_SETUP;
        }

        if (q.is_null() && inner.n > 0) || (b.is_null() && inner.m > 0) {
            set_last_error("q or b is NULL".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        let qs = unsafe { chunk_batched(q, inner.n, inner.batch_size) };
        let bs = unsafe { chunk_batched(b, inner.m, inner.batch_size) };

        let result = inner.solver.solve(&qs, &bs);
        handle_solve_result(inner, result)
    })
}

/// Solve with warm start.
#[no_mangle]
pub extern "C" fn moreau_solver_solve_warm(
    solver: *mut MoreauSolverInner,
    q: *const f64,
    b: *const f64,
    warm_x: *const f64,
    warm_z: *const f64,
    warm_s: *const f64,
) -> i32 {
    catch_panic(|| {
        clear_last_error();

        if solver.is_null() {
            set_last_error("solver is NULL".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: null-checked; caller must pass a live handle from the constructor.
        let inner = unsafe { &mut *solver };

        if !inner.is_setup {
            set_last_error("solve() called before setup()".into());
            return MOREAU_ERROR_NOT_SETUP;
        }

        if (q.is_null() && inner.n > 0)
            || (b.is_null() && inner.m > 0)
            || (warm_x.is_null() && inner.n > 0)
            || (warm_z.is_null() && inner.m > 0)
            || (warm_s.is_null() && inner.m > 0)
        {
            set_last_error("NULL pointer for required argument".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        let qs = unsafe { chunk_batched(q, inner.n, inner.batch_size) };
        let bs = unsafe { chunk_batched(b, inner.m, inner.batch_size) };
        let wxs = unsafe { chunk_batched(warm_x, inner.n, inner.batch_size) };
        let wzs = unsafe { chunk_batched(warm_z, inner.m, inner.batch_size) };
        let wss = unsafe { chunk_batched(warm_s, inner.m, inner.batch_size) };

        // Direct-x warm `z_x` is not exposed through the C API yet.
        let result =
            inner
                .solver
                .solve_with_warm_start(&qs, &bs, Some(&wxs), Some(&wzs), Some(&wss), None);
        handle_solve_result(inner, result)
    })
}

/// Destroy solver.
#[no_mangle]
pub extern "C" fn moreau_solver_destroy(solver: *mut MoreauSolverInner) {
    if !solver.is_null() {
        unsafe {
            drop(Box::from_raw(solver));
        }
    }
}

/// Get solution for a batch index.
#[no_mangle]
pub extern "C" fn moreau_solver_get_solution(
    solver: *const MoreauSolverInner,
    batch_idx: i64,
    sol: *mut MoreauSolutionT,
) -> i32 {
    catch_panic(|| {
        clear_last_error();

        if solver.is_null() || sol.is_null() {
            set_last_error("NULL pointer".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: null-checked; caller must pass a live handle from the constructor.
        let inner = unsafe { &*solver };

        if !inner.has_solution {
            set_last_error("No solution available (call solve() first)".into());
            return MOREAU_ERROR_NOT_SETUP;
        }

        let idx = batch_idx as usize;
        if idx >= inner.solutions.len() {
            set_last_error(format!(
                "batch_idx {} out of range (0..{})",
                idx,
                inner.solutions.len()
            ));
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        let solution = &inner.solutions[idx];
        // SAFETY: null-checked; caller must pass a valid output struct pointer.
        let out = unsafe { &mut *sol };
        out.x = solution.x.as_ptr();
        out.z = solution.z.as_ptr();
        out.s = solution.s.as_ptr();
        out.status = solver_status_to_c(solution.status);
        out.obj_val = solution.obj_val;
        out.obj_val_dual = solution.obj_val_dual;
        out.solve_time = solution.solve_time;
        out.iterations = solution.iterations as i32;
        out.r_prim = solution.r_prim;
        out.r_dual = solution.r_dual;

        MOREAU_OK
    })
}

/// Copy solution vectors to caller-provided buffers.
#[no_mangle]
pub extern "C" fn moreau_solver_copy_solution(
    solver: *const MoreauSolverInner,
    batch_idx: i64,
    x_out: *mut f64,
    z_out: *mut f64,
    s_out: *mut f64,
) -> i32 {
    catch_panic(|| {
        clear_last_error();

        if solver.is_null() {
            set_last_error("solver is NULL".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: null-checked; caller must pass a live handle from the constructor.
        let inner = unsafe { &*solver };

        if !inner.has_solution {
            set_last_error("No solution available".into());
            return MOREAU_ERROR_NOT_SETUP;
        }

        let idx = batch_idx as usize;
        if idx >= inner.solutions.len() {
            set_last_error(format!(
                "batch_idx {} out of range (0..{})",
                idx,
                inner.solutions.len()
            ));
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        let solution = &inner.solutions[idx];

        if !x_out.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(solution.x.as_ptr(), x_out, inner.n);
            }
        }
        if !z_out.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(solution.z.as_ptr(), z_out, inner.m);
            }
        }
        if !s_out.is_null() {
            unsafe {
                ptr::copy_nonoverlapping(solution.s.as_ptr(), s_out, inner.m);
            }
        }

        MOREAU_OK
    })
}

/// Get solver status for a batch index.
#[no_mangle]
pub extern "C" fn moreau_solver_get_status(
    solver: *const MoreauSolverInner,
    batch_idx: i64,
) -> i32 {
    // Return 0 (Unsolved) as safe default on panic, since this returns a status
    // code rather than a moreau_error_t.
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        if solver.is_null() {
            return 0; // Unsolved
        }
        // SAFETY: null-checked; caller must pass a live handle from the constructor.
        let inner = unsafe { &*solver };

        if !inner.has_solution {
            return 0; // Unsolved
        }

        let idx = batch_idx as usize;
        if idx >= inner.solutions.len() {
            return 0;
        }

        solver_status_to_c(inner.solutions[idx].status)
    }))
    .unwrap_or(0)
}

/// Backward pass.
#[no_mangle]
pub extern "C" fn moreau_solver_backward(
    solver: *mut MoreauSolverInner,
    dx: *const f64,
    dz: *const f64,
    ds: *const f64,
    dP_out: *mut f64,
    dA_out: *mut f64,
    dq_out: *mut f64,
    db_out: *mut f64,
) -> i32 {
    catch_panic(|| {
        clear_last_error();

        if solver.is_null() {
            set_last_error("solver is NULL".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: null-checked; caller must pass a live handle from the constructor.
        let inner = unsafe { &mut *solver };

        if !inner.enable_grad {
            set_last_error("backward() requires enable_grad=1 in settings".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        if !inner.has_solution {
            set_last_error("backward() requires solve() to be called first".into());
            return MOREAU_ERROR_NOT_SETUP;
        }

        if dx.is_null() {
            set_last_error("dx is NULL".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        let batch_size = inner.batch_size;
        let n = inner.n;
        let m = inner.m;

        // SAFETY: null-checked above; caller must pass `batch_size * n` f64s.
        let dx_all = unsafe { slice::from_raw_parts(dx, batch_size * n) };

        let dz_all: Vec<f64> = if dz.is_null() {
            vec![0.0; batch_size * m]
        } else {
            // SAFETY: non-null; caller must pass `batch_size * m` f64s.
            unsafe { slice::from_raw_parts(dz, batch_size * m) }.to_vec()
        };

        let ds_all: Vec<f64> = if ds.is_null() {
            vec![0.0; batch_size * m]
        } else {
            // SAFETY: non-null; caller must pass `batch_size * m` f64s.
            unsafe { slice::from_raw_parts(ds, batch_size * m) }.to_vec()
        };

        let upstream_grads: Vec<UpstreamGradients<f64>> = (0..batch_size)
            .map(|i| UpstreamGradients {
                dx: dx_all[i * n..(i + 1) * n].to_vec(),
                dz: dz_all[i * m..(i + 1) * m].to_vec(),
                ds: ds_all[i * m..(i + 1) * m].to_vec(),
                dz_x: Vec::new(), // C API does not yet expose direct-x cones
            })
            .collect();

        match inner.solver.backward(&upstream_grads) {
            Ok(grads) => {
                // SAFETY: each copy is null-checked; caller must pass output
                // buffers large enough for `batch_size` blocks of each gradient.
                for (i, g) in grads.iter().enumerate() {
                    if !dP_out.is_null() {
                        unsafe {
                            ptr::copy_nonoverlapping(
                                g.dP_values.as_ptr(),
                                dP_out.add(i * inner.nnz_P),
                                inner.nnz_P,
                            );
                        }
                    }
                    if !dA_out.is_null() {
                        unsafe {
                            ptr::copy_nonoverlapping(
                                g.dA_values.as_ptr(),
                                dA_out.add(i * inner.nnz_A),
                                inner.nnz_A,
                            );
                        }
                    }
                    if !dq_out.is_null() {
                        unsafe {
                            ptr::copy_nonoverlapping(g.dq.as_ptr(), dq_out.add(i * n), n);
                        }
                    }
                    if !db_out.is_null() {
                        unsafe {
                            ptr::copy_nonoverlapping(g.db.as_ptr(), db_out.add(i * m), m);
                        }
                    }
                }
                MOREAU_OK
            }
            Err(e) => {
                set_last_error(format!("backward() failed: {}", e));
                MOREAU_ERROR_NUMERICAL
            }
        }
    })
}

/// Query problem dimensions.
#[no_mangle]
pub extern "C" fn moreau_solver_get_dims(
    solver: *const MoreauSolverInner,
    n_out: *mut i64,
    m_out: *mut i64,
    batch_size_out: *mut i64,
    nnz_P_out: *mut i64,
    nnz_A_out: *mut i64,
) -> i32 {
    catch_panic(|| {
        clear_last_error();

        if solver.is_null() {
            set_last_error("solver is NULL".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }
        // SAFETY: null-checked; caller must pass a live handle from the constructor.
        let inner = unsafe { &*solver };

        // SAFETY: each write is null-checked; caller must pass valid out pointers.
        if !n_out.is_null() {
            unsafe { *n_out = inner.n as i64 };
        }
        if !m_out.is_null() {
            unsafe { *m_out = inner.m as i64 };
        }
        if !batch_size_out.is_null() {
            unsafe { *batch_size_out = inner.batch_size as i64 };
        }
        if !nnz_P_out.is_null() {
            unsafe { *nnz_P_out = inner.nnz_P as i64 };
        }
        if !nnz_A_out.is_null() {
            unsafe { *nnz_A_out = inner.nnz_A as i64 };
        }

        MOREAU_OK
    })
}

/// Query memory usage (not fully implemented for CPU).
#[no_mangle]
pub extern "C" fn moreau_solver_memory_usage(
    solver: *const MoreauSolverInner,
    bytes_out: *mut usize,
) -> i32 {
    catch_panic(|| {
        clear_last_error();

        if solver.is_null() || bytes_out.is_null() {
            set_last_error("NULL pointer".into());
            return MOREAU_ERROR_INVALID_ARGUMENT;
        }

        // Approximate memory usage — exact tracking not available for CPU backend
        // SAFETY: null-checked; caller must pass a live handle from the constructor.
        let inner = unsafe { &*solver };
        let approx = (inner.n + inner.m) * inner.batch_size * 8 * 10; // rough estimate
                                                                      // SAFETY: null-checked; caller must pass a valid out pointer.
        unsafe { *bytes_out = approx };

        MOREAU_OK
    })
}
