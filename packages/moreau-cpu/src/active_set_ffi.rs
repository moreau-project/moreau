//! FFI bindings to the C++ active-set QP solver.
//!
//! Wraps the C API from `active_set_c_api.h` with safe Rust types.

use std::ffi::CStr;
use std::ptr;

// ============================================================================
// Raw C FFI declarations
// ============================================================================

#[repr(C)]
#[allow(non_camel_case_types)]
pub enum moreau_as_error_t {
    MOREAU_AS_OK = 0,
    MOREAU_AS_ERROR_INVALID_ARGUMENT = 1,
    MOREAU_AS_ERROR_NOT_SETUP = 2,
    MOREAU_AS_ERROR_NOT_SOLVED = 3,
    MOREAU_AS_ERROR_GRAD_DISABLED = 4,
    MOREAU_AS_ERROR_INTERNAL = 99,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum moreau_as_diff_method_t {
    MOREAU_AS_DIFF_EXACT = 0,
    MOREAU_AS_DIFF_SMOOTHED = 1,
}

#[repr(C)]
#[derive(Debug, Clone)]
#[allow(non_camel_case_types)]
pub struct moreau_as_settings_t {
    pub primal_tol: f64,
    pub dual_tol: f64,
    pub zero_tol: f64,
    pub pivot_tol: f64,
    pub progress_tol: f64,
    pub fval_bound: f64,
    pub iter_limit: i32,
    pub time_limit: f64,
    pub cycle_tol: i32,
    pub diff_method: moreau_as_diff_method_t,
    pub diff_smoothing_mu: f64,
}

#[repr(C)]
#[derive(Debug)]
#[allow(non_camel_case_types)]
pub struct moreau_as_solution_t {
    pub x: *const f64,
    pub z: *const f64,
    pub s: *const f64,
    pub status: i32,
    pub obj_val: f64,
    pub iterations: i32,
}

#[repr(C)]
#[derive(Debug)]
#[allow(non_camel_case_types)]
pub struct moreau_as_backward_t {
    pub dP_values: *const f64,
    pub dA_values: *const f64,
    pub dq: *const f64,
    pub db: *const f64,
}

#[repr(C)]
#[derive(Debug)]
#[allow(non_camel_case_types)]
pub struct moreau_as_backward_state_t {
    pub rinv: *const f64,
    pub rinv_diag: *const f64,
    pub ws: *const i32,
    pub sense: *const i32,
    pub lam_star: *const f64,
    pub n_active: i32,
    pub use_rinv_diag: i32,
}

#[allow(non_camel_case_types)]
pub enum moreau_as_solver_s {}

extern "C" {
    pub fn moreau_as_last_error() -> *const std::ffi::c_char;
    pub fn moreau_as_settings_default(settings: *mut moreau_as_settings_t);

    pub fn moreau_as_create(
        solver_out: *mut *mut moreau_as_solver_s,
        n: i64,
        m: i64,
        batch_size: i64,
        P_row_offsets: *const i64,
        P_col_indices: *const i64,
        nnz_P: i64,
        A_row_offsets: *const i64,
        A_col_indices: *const i64,
        nnz_A: i64,
        num_zero_cones: i64,
        num_nonneg_cones: i64,
        settings: *const moreau_as_settings_t,
        enable_grad: i32,
        verbose: i32,
    ) -> moreau_as_error_t;

    pub fn moreau_as_setup(
        solver: *mut moreau_as_solver_s,
        P_values: *const f64,
        A_values: *const f64,
        shared: i32,
    ) -> moreau_as_error_t;

    pub fn moreau_as_solve(
        solver: *mut moreau_as_solver_s,
        q: *const f64,
        b: *const f64,
    ) -> moreau_as_error_t;

    pub fn moreau_as_solve_warm(
        solver: *mut moreau_as_solver_s,
        q: *const f64,
        b: *const f64,
        warm_x: *const f64,
        warm_z: *const f64,
        warm_s: *const f64,
    ) -> moreau_as_error_t;

    pub fn moreau_as_backward(
        solver: *mut moreau_as_solver_s,
        dx: *const f64,
        dz: *const f64,
        ds: *const f64,
    ) -> moreau_as_error_t;

    pub fn moreau_as_backward_with_data(
        solver: *mut moreau_as_solver_s,
        dx: *const f64,
        dz: *const f64,
        ds: *const f64,
        P_values: *const f64,
        A_values: *const f64,
        shared: i32,
        q: *const f64,
        b: *const f64,
        x: *const f64,
        z: *const f64,
        s: *const f64,
        state_rinv: *const f64,
        state_rinv_diag: *const f64,
        state_use_rinv_diag: *const i32,
        state_n_active: *const i32,
        state_ws: *const i32,
        state_sense: *const i32,
        state_lam_star: *const f64,
    ) -> moreau_as_error_t;

    pub fn moreau_as_destroy(solver: *mut moreau_as_solver_s);

    pub fn moreau_as_get_solution(
        solver: *const moreau_as_solver_s,
        batch_idx: i64,
        sol: *mut moreau_as_solution_t,
    ) -> moreau_as_error_t;

    pub fn moreau_as_get_solve_time(
        solver: *const moreau_as_solver_s,
        time_out: *mut f64,
    ) -> moreau_as_error_t;

    pub fn moreau_as_get_backward(
        solver: *const moreau_as_solver_s,
        batch_idx: i64,
        grad: *mut moreau_as_backward_t,
    ) -> moreau_as_error_t;

    pub fn moreau_as_get_backward_state(
        solver: *const moreau_as_solver_s,
        batch_idx: i64,
        state: *mut moreau_as_backward_state_t,
    ) -> moreau_as_error_t;

    pub fn moreau_as_get_dims(
        solver: *const moreau_as_solver_s,
        n_out: *mut i64,
        m_out: *mut i64,
        batch_size_out: *mut i64,
        nnz_P_out: *mut i64,
        nnz_A_out: *mut i64,
    ) -> moreau_as_error_t;
}

// ============================================================================
// Safe Rust wrappers
// ============================================================================

fn last_error_string() -> String {
    unsafe {
        let ptr = moreau_as_last_error();
        if ptr.is_null() {
            "Unknown error".to_string()
        } else {
            // SAFETY: non-null; C side returns a valid NUL-terminated string.
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

fn check(err: moreau_as_error_t) -> Result<(), String> {
    match err {
        moreau_as_error_t::MOREAU_AS_OK => Ok(()),
        _ => Err(last_error_string()),
    }
}

/// Active-set solver settings
#[derive(Debug, Clone)]
pub struct ActiveSetSettings {
    inner: moreau_as_settings_t,
}

impl Default for ActiveSetSettings {
    fn default() -> Self {
        let mut inner = moreau_as_settings_t {
            primal_tol: 0.0,
            dual_tol: 0.0,
            zero_tol: 0.0,
            pivot_tol: 0.0,
            progress_tol: 0.0,
            fval_bound: 0.0,
            iter_limit: 0,
            time_limit: 0.0,
            cycle_tol: 0,
            diff_method: moreau_as_diff_method_t::MOREAU_AS_DIFF_EXACT,
            diff_smoothing_mu: 0.0,
        };
        unsafe { moreau_as_settings_default(&mut inner) };
        Self { inner }
    }
}

impl ActiveSetSettings {
    /// Get a reference to the raw C settings struct
    pub fn as_raw(&self) -> &moreau_as_settings_t {
        &self.inner
    }

    /// Get a mutable reference to the raw C settings struct
    pub fn as_raw_mut(&mut self) -> &mut moreau_as_settings_t {
        &mut self.inner
    }
}

/// Active-set QP solver wrapping the C++ implementation
pub struct ActiveSetSolver {
    handle: *mut moreau_as_solver_s,
    n: i64,
    m: i64,
    batch_size: i64,
    nnz_p: i64,
    nnz_a: i64,
}

// Safety: ActiveSetSolver owns its handle and the C++ solver is thread-safe
// for non-overlapping calls (same as any mutable reference).
unsafe impl Send for ActiveSetSolver {}
unsafe impl Sync for ActiveSetSolver {}

impl ActiveSetSolver {
    /// Create a new active-set solver.
    pub fn new(
        n: i64,
        m: i64,
        batch_size: i64,
        p_row_offsets: &[i64],
        p_col_indices: &[i64],
        nnz_p: i64,
        a_row_offsets: &[i64],
        a_col_indices: &[i64],
        nnz_a: i64,
        num_zero_cones: i64,
        num_nonneg_cones: i64,
        settings: &ActiveSetSettings,
        enable_grad: bool,
        verbose: bool,
    ) -> Result<Self, String> {
        let mut handle: *mut moreau_as_solver_s = ptr::null_mut();
        let err = unsafe {
            moreau_as_create(
                &mut handle,
                n,
                m,
                batch_size,
                p_row_offsets.as_ptr(),
                p_col_indices.as_ptr(),
                nnz_p,
                a_row_offsets.as_ptr(),
                a_col_indices.as_ptr(),
                nnz_a,
                num_zero_cones,
                num_nonneg_cones,
                settings.as_raw(),
                enable_grad as i32,
                verbose as i32,
            )
        };
        check(err)?;
        Ok(Self {
            handle,
            n,
            m,
            batch_size,
            nnz_p,
            nnz_a,
        })
    }

    /// Set P and A matrix values.
    pub fn setup(
        &mut self,
        p_values: &[f64],
        a_values: &[f64],
        shared: bool,
    ) -> Result<(), String> {
        let err = unsafe {
            moreau_as_setup(
                self.handle,
                p_values.as_ptr(),
                a_values.as_ptr(),
                shared as i32,
            )
        };
        check(err)
    }

    /// Solve with given q and b.
    pub fn solve(&mut self, q: &[f64], b: &[f64]) -> Result<(), String> {
        let err = unsafe { moreau_as_solve(self.handle, q.as_ptr(), b.as_ptr()) };
        check(err)
    }

    /// Solve with warm start.
    pub fn solve_warm(
        &mut self,
        q: &[f64],
        b: &[f64],
        warm_x: &[f64],
        warm_z: &[f64],
        warm_s: &[f64],
    ) -> Result<(), String> {
        let err = unsafe {
            moreau_as_solve_warm(
                self.handle,
                q.as_ptr(),
                b.as_ptr(),
                warm_x.as_ptr(),
                warm_z.as_ptr(),
                warm_s.as_ptr(),
            )
        };
        check(err)
    }

    /// Compute backward pass.
    pub fn backward(&mut self, dx: &[f64], dz: &[f64], ds: &[f64]) -> Result<(), String> {
        let err = unsafe { moreau_as_backward(self.handle, dx.as_ptr(), dz.as_ptr(), ds.as_ptr()) };
        check(err)
    }

    pub fn backward_with_data(
        &mut self,
        dx: &[f64],
        dz: &[f64],
        ds: &[f64],
        p_values: &[f64],
        a_values: &[f64],
        shared: bool,
        q: &[f64],
        b: &[f64],
        x: &[f64],
        z: &[f64],
        s: &[f64],
        state_rinv: &[f64],
        state_rinv_diag: &[f64],
        state_use_rinv_diag: &[i32],
        state_n_active: &[i32],
        state_ws: &[i32],
        state_sense: &[i32],
        state_lam_star: &[f64],
    ) -> Result<(), String> {
        let err = unsafe {
            moreau_as_backward_with_data(
                self.handle,
                dx.as_ptr(),
                dz.as_ptr(),
                ds.as_ptr(),
                p_values.as_ptr(),
                a_values.as_ptr(),
                shared as i32,
                q.as_ptr(),
                b.as_ptr(),
                x.as_ptr(),
                z.as_ptr(),
                s.as_ptr(),
                state_rinv.as_ptr(),
                state_rinv_diag.as_ptr(),
                state_use_rinv_diag.as_ptr(),
                state_n_active.as_ptr(),
                state_ws.as_ptr(),
                state_sense.as_ptr(),
                state_lam_star.as_ptr(),
            )
        };
        check(err)
    }

    pub fn get_backward_state(&self, batch_idx: i64) -> Result<ActiveSetBackwardState, String> {
        let mut state = moreau_as_backward_state_t {
            rinv: ptr::null(),
            rinv_diag: ptr::null(),
            ws: ptr::null(),
            sense: ptr::null(),
            lam_star: ptr::null(),
            n_active: 0,
            use_rinv_diag: 0,
        };
        let err = unsafe { moreau_as_get_backward_state(self.handle, batch_idx, &mut state) };
        check(err)?;

        let n = self.n as usize;
        let m = self.m as usize;
        let packed_rinv = n * (n + 1) / 2;
        // SAFETY: C side populated `state` with arrays of these lengths.
        Ok(ActiveSetBackwardState {
            rinv: unsafe { std::slice::from_raw_parts(state.rinv, packed_rinv).to_vec() },
            rinv_diag: unsafe { std::slice::from_raw_parts(state.rinv_diag, n).to_vec() },
            ws: unsafe { std::slice::from_raw_parts(state.ws, m).to_vec() },
            sense: unsafe { std::slice::from_raw_parts(state.sense, m).to_vec() },
            lam_star: unsafe { std::slice::from_raw_parts(state.lam_star, m).to_vec() },
            n_active: state.n_active,
            use_rinv_diag: state.use_rinv_diag != 0,
        })
    }

    /// Get solution for a batch index. Returns (x, z, s, status, obj_val, iterations).
    pub fn get_solution(&self, batch_idx: i64) -> Result<ActiveSetSolution, String> {
        let mut sol = moreau_as_solution_t {
            x: ptr::null(),
            z: ptr::null(),
            s: ptr::null(),
            status: 0,
            obj_val: 0.0,
            iterations: 0,
        };
        let err = unsafe { moreau_as_get_solution(self.handle, batch_idx, &mut sol) };
        check(err)?;

        let n = self.n as usize;
        let m = self.m as usize;
        // SAFETY: C side populated `sol` with arrays of these lengths.
        Ok(ActiveSetSolution {
            x: unsafe { std::slice::from_raw_parts(sol.x, n).to_vec() },
            z: unsafe { std::slice::from_raw_parts(sol.z, m).to_vec() },
            s: unsafe { std::slice::from_raw_parts(sol.s, m).to_vec() },
            status: sol.status,
            obj_val: sol.obj_val,
            iterations: sol.iterations,
        })
    }

    /// Get solve time in seconds.
    pub fn solve_time(&self) -> Result<f64, String> {
        let mut t = 0.0;
        let err = unsafe { moreau_as_get_solve_time(self.handle, &mut t) };
        check(err)?;
        Ok(t)
    }

    /// Get backward results for a batch index.
    pub fn get_backward(&self, batch_idx: i64) -> Result<ActiveSetBackward, String> {
        let mut grad = moreau_as_backward_t {
            dP_values: ptr::null(),
            dA_values: ptr::null(),
            dq: ptr::null(),
            db: ptr::null(),
        };
        let err = unsafe { moreau_as_get_backward(self.handle, batch_idx, &mut grad) };
        check(err)?;

        let n = self.n as usize;
        let m = self.m as usize;
        let nnz_p = self.nnz_p as usize;
        let nnz_a = self.nnz_a as usize;
        // SAFETY: C side populated `grad` with arrays of these lengths.
        Ok(ActiveSetBackward {
            dp_values: unsafe { std::slice::from_raw_parts(grad.dP_values, nnz_p).to_vec() },
            da_values: unsafe { std::slice::from_raw_parts(grad.dA_values, nnz_a).to_vec() },
            dq: unsafe { std::slice::from_raw_parts(grad.dq, n).to_vec() },
            db: unsafe { std::slice::from_raw_parts(grad.db, m).to_vec() },
        })
    }

    /// Dimensions
    pub fn n(&self) -> i64 {
        self.n
    }
    /// Dimensions
    pub fn m(&self) -> i64 {
        self.m
    }
    /// Dimensions
    pub fn batch_size(&self) -> i64 {
        self.batch_size
    }
    /// Dimensions
    pub fn nnz_p(&self) -> i64 {
        self.nnz_p
    }
    /// Dimensions
    pub fn nnz_a(&self) -> i64 {
        self.nnz_a
    }
}

impl Drop for ActiveSetSolver {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe { moreau_as_destroy(self.handle) };
            self.handle = ptr::null_mut();
        }
    }
}

/// Solution data from a single batch element
#[derive(Debug, Clone)]
pub struct ActiveSetSolution {
    /// Primal solution
    pub x: Vec<f64>,
    /// Dual solution
    pub z: Vec<f64>,
    /// Slack variables
    pub s: Vec<f64>,
    /// DAQP exit flag (1 = optimal)
    pub status: i32,
    /// Objective value
    pub obj_val: f64,
    /// Number of iterations
    pub iterations: i32,
}

/// Backward gradient data from a single batch element
#[derive(Debug, Clone)]
pub struct ActiveSetBackward {
    /// Gradient w.r.t. P values (sparse CSR)
    pub dp_values: Vec<f64>,
    /// Gradient w.r.t. A values (sparse CSR)
    pub da_values: Vec<f64>,
    /// Gradient w.r.t. q
    pub dq: Vec<f64>,
    /// Gradient w.r.t. b
    pub db: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ActiveSetBackwardState {
    pub rinv: Vec<f64>,
    pub rinv_diag: Vec<f64>,
    pub ws: Vec<i32>,
    pub sense: Vec<i32>,
    pub lam_star: Vec<f64>,
    pub n_active: i32,
    pub use_rinv_diag: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_qp() {
        // minimize    0.5*(x1^2 + x2^2) + x1 + x2
        // subject to  x1 + x2 = 1, x1 >= 0, x2 >= 0
        let n = 2i64;
        let m = 3i64;
        let p_ro = vec![0i64, 1, 2];
        let p_ci = vec![0i64, 1];
        let a_ro = vec![0i64, 2, 3, 4];
        let a_ci = vec![0i64, 1, 0, 1];

        let settings = ActiveSetSettings::default();
        let mut solver = ActiveSetSolver::new(
            n, m, 1, &p_ro, &p_ci, 2, &a_ro, &a_ci, 4, 1, 2, // 1 zero cone, 2 nonneg cones
            &settings, false, false,
        )
        .unwrap();

        solver
            .setup(&[1.0, 1.0], &[1.0, 1.0, -1.0, -1.0], true)
            .unwrap();
        solver.solve(&[1.0, 1.0], &[1.0, 0.0, 0.0]).unwrap();

        let sol = solver.get_solution(0).unwrap();
        // status_vec stores SolverStatus::Solved which is 1
        assert_eq!(sol.status, 1);
        assert!((sol.x[0] - 0.5).abs() < 1e-6);
        assert!((sol.x[1] - 0.5).abs() < 1e-6);
    }
}
