//! Python bindings for the active-set QP solver via pyo3.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::compiled_solver_py::extract_f64_buffer;
use crate::active_set_ffi;

#[pyclass(name = "ActiveSetBackwardState")]
#[derive(Debug, Clone)]
pub struct PyActiveSetBackwardState {
    pub(crate) inner: Vec<active_set_ffi::ActiveSetBackwardState>,
}

#[pymethods]
impl PyActiveSetBackwardState {
    fn _to_flat_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let batch = self.inner.len();
        let packed_rinv: usize = self
            .inner
            .iter()
            .map(|state| state.rinv.len())
            .max()
            .unwrap_or(0);
        let n: usize = self
            .inner
            .iter()
            .map(|state| state.rinv_diag.len())
            .max()
            .unwrap_or(0);
        let m: usize = self
            .inner
            .iter()
            .map(|state| state.ws.len())
            .max()
            .unwrap_or(0);

        let mut rinv_flat = Vec::with_capacity(batch * packed_rinv);
        let mut rinv_diag_flat = Vec::with_capacity(batch * n);
        let mut use_rinv_diag_flat = Vec::with_capacity(batch);
        let mut n_active_flat = Vec::with_capacity(batch);
        let mut ws_flat = Vec::with_capacity(batch * m);
        let mut sense_flat = Vec::with_capacity(batch * m);
        let mut lam_star_flat = Vec::with_capacity(batch * m);

        for state in &self.inner {
            rinv_flat.extend_from_slice(&state.rinv);
            rinv_diag_flat.extend_from_slice(&state.rinv_diag);
            use_rinv_diag_flat.push(state.use_rinv_diag as i32);
            n_active_flat.push(state.n_active);
            ws_flat.extend_from_slice(&state.ws);
            sense_flat.extend_from_slice(&state.sense);
            lam_star_flat.extend_from_slice(&state.lam_star);
        }

        dict.set_item("rinv", rinv_flat)?;
        dict.set_item("rinv_diag", rinv_diag_flat)?;
        dict.set_item("use_rinv_diag", use_rinv_diag_flat)?;
        dict.set_item("n_active", n_active_flat)?;
        dict.set_item("ws", ws_flat)?;
        dict.set_item("sense", sense_flat)?;
        dict.set_item("lam_star", lam_star_flat)?;
        Ok(dict)
    }
}

/// Active-set solver settings exposed to Python.
#[pyclass(name = "ActiveSetSettings")]
#[derive(Debug, Clone)]
pub struct PyActiveSetSettings {
    pub(crate) inner: active_set_ffi::ActiveSetSettings,
}

#[pymethods]
impl PyActiveSetSettings {
    #[new]
    #[pyo3(signature = (
        primal_tol=None, dual_tol=None, zero_tol=None, pivot_tol=None,
        progress_tol=None, fval_bound=None, iter_limit=None, time_limit=None,
        cycle_tol=None, diff_method=None, diff_smoothing_mu=None
    ))]
    fn new(
        primal_tol: Option<f64>,
        dual_tol: Option<f64>,
        zero_tol: Option<f64>,
        pivot_tol: Option<f64>,
        progress_tol: Option<f64>,
        fval_bound: Option<f64>,
        iter_limit: Option<i32>,
        time_limit: Option<f64>,
        cycle_tol: Option<i32>,
        diff_method: Option<&str>,
        diff_smoothing_mu: Option<f64>,
    ) -> pyo3::PyResult<Self> {
        let mut settings = active_set_ffi::ActiveSetSettings::default();
        let raw = settings.as_raw_mut();
        if let Some(v) = primal_tol {
            raw.primal_tol = v;
        }
        if let Some(v) = dual_tol {
            raw.dual_tol = v;
        }
        if let Some(v) = zero_tol {
            raw.zero_tol = v;
        }
        if let Some(v) = pivot_tol {
            raw.pivot_tol = v;
        }
        if let Some(v) = progress_tol {
            raw.progress_tol = v;
        }
        if let Some(v) = fval_bound {
            raw.fval_bound = v;
        }
        if let Some(v) = iter_limit {
            raw.iter_limit = v;
        }
        if let Some(v) = time_limit {
            raw.time_limit = v;
        }
        if let Some(v) = cycle_tol {
            raw.cycle_tol = v;
        }
        if let Some(dm) = diff_method {
            raw.diff_method = match dm {
                "exact" => active_set_ffi::moreau_as_diff_method_t::MOREAU_AS_DIFF_EXACT,
                "smoothed" => active_set_ffi::moreau_as_diff_method_t::MOREAU_AS_DIFF_SMOOTHED,
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid diff_method '{}'. Must be 'exact' or 'smoothed'.",
                        other
                    )))
                }
            };
        }
        if let Some(v) = diff_smoothing_mu {
            if v <= 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "diff_smoothing_mu must be positive, got {}",
                    v
                )));
            }
            raw.diff_smoothing_mu = v;
        }
        Ok(Self { inner: settings })
    }
}

/// Active-set QP solver exposed to Python.
///
/// Wraps the C++ dual active-set solver via the C API.
/// Supports batched solving, warm starting, and backward differentiation.
#[pyclass(name = "ActiveSetSolver")]
pub struct PyActiveSetSolver {
    solver: active_set_ffi::ActiveSetSolver,
    n: i64,
    m: i64,
    batch_size: i64,
}

#[pymethods]
impl PyActiveSetSolver {
    #[new]
    #[pyo3(signature = (
        n, m, batch_size,
        P_row_offsets, P_col_indices, nnz_P,
        A_row_offsets, A_col_indices, nnz_A,
        num_zero_cones, num_nonneg_cones,
        settings=None, enable_grad=false, verbose=false
    ))]
    fn new(
        n: i64,
        m: i64,
        batch_size: i64,
        P_row_offsets: Vec<i64>,
        P_col_indices: Vec<i64>,
        nnz_P: i64,
        A_row_offsets: Vec<i64>,
        A_col_indices: Vec<i64>,
        nnz_A: i64,
        num_zero_cones: i64,
        num_nonneg_cones: i64,
        settings: Option<&PyActiveSetSettings>,
        enable_grad: bool,
        verbose: bool,
    ) -> PyResult<Self> {
        let default_settings = active_set_ffi::ActiveSetSettings::default();
        let s = settings.map(|s| &s.inner).unwrap_or(&default_settings);

        let solver = active_set_ffi::ActiveSetSolver::new(
            n,
            m,
            batch_size,
            &P_row_offsets,
            &P_col_indices,
            nnz_P,
            &A_row_offsets,
            &A_col_indices,
            nnz_A,
            num_zero_cones,
            num_nonneg_cones,
            s,
            enable_grad,
            verbose,
        )
        .map_err(|e| PyRuntimeError::new_err(e))?;

        Ok(Self {
            solver,
            n,
            m,
            batch_size,
        })
    }

    /// Set P and A matrix values. Pass flat contiguous f64 arrays.
    fn setup(
        &mut self,
        P_values: &Bound<'_, PyAny>,
        A_values: &Bound<'_, PyAny>,
        shared: bool,
    ) -> PyResult<()> {
        let p = extract_f64_buffer(P_values)?;
        let a = extract_f64_buffer(A_values)?;
        let nnz_p = self.solver.nnz_p() as usize;
        let nnz_a = self.solver.nnz_a() as usize;
        let bs = self.batch_size as usize;
        let expect_p = if shared { nnz_p } else { bs * nnz_p };
        let expect_a = if shared { nnz_a } else { bs * nnz_a };
        if p.len() != expect_p {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "P_values has {} elements, expected {}",
                p.len(),
                expect_p
            )));
        }
        if a.len() != expect_a {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "A_values has {} elements, expected {}",
                a.len(),
                expect_a
            )));
        }
        self.solver
            .setup(&p, &a, shared)
            .map_err(|e| PyRuntimeError::new_err(e))
    }

    /// Solve the QP. Returns a dict with x, z, s, status, obj_val, etc.
    #[pyo3(signature = (q, b, warm_x=None, warm_z=None, warm_s=None))]
    fn solve<'py>(
        &mut self,
        py: Python<'py>,
        q: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        warm_x: Option<&Bound<'_, PyAny>>,
        warm_z: Option<&Bound<'_, PyAny>>,
        warm_s: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let q_vec = extract_f64_buffer(q)?;
        let b_vec = extract_f64_buffer(b)?;
        let n = self.n as usize;
        let m = self.m as usize;
        let bs = self.batch_size as usize;
        if q_vec.len() != bs * n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q has {} elements, expected {}",
                q_vec.len(),
                bs * n
            )));
        }
        if b_vec.len() != bs * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "b has {} elements, expected {}",
                b_vec.len(),
                bs * m
            )));
        }

        if let (Some(wx), Some(wz), Some(ws)) = (warm_x, warm_z, warm_s) {
            let wx_vec = extract_f64_buffer(wx)?;
            let wz_vec = extract_f64_buffer(wz)?;
            let ws_vec = extract_f64_buffer(ws)?;
            if wx_vec.len() != bs * n || wz_vec.len() != bs * m || ws_vec.len() != bs * m {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "warm start arrays have wrong dimensions",
                ));
            }
            py.allow_threads(|| {
                self.solver
                    .solve_warm(&q_vec, &b_vec, &wx_vec, &wz_vec, &ws_vec)
            })
            .map_err(|e| PyRuntimeError::new_err(e))?;
        } else {
            py.allow_threads(|| self.solver.solve(&q_vec, &b_vec))
                .map_err(|e| PyRuntimeError::new_err(e))?;
        }

        let solve_time = self.solver.solve_time().unwrap_or(0.0);
        let dict = PyDict::new(py);

        let n = self.n as usize;
        let m = self.m as usize;
        let bs = self.batch_size as usize;

        let mut x_flat = Vec::with_capacity(bs * n);
        let mut z_flat = Vec::with_capacity(bs * m);
        let mut s_flat = Vec::with_capacity(bs * m);
        let mut statuses = Vec::with_capacity(bs);
        let mut obj_vals = Vec::with_capacity(bs);
        let mut iters_vec = Vec::with_capacity(bs);

        for i in 0..bs {
            let sol = self
                .solver
                .get_solution(i as i64)
                .map_err(|e| PyRuntimeError::new_err(e))?;
            x_flat.extend_from_slice(&sol.x);
            z_flat.extend_from_slice(&sol.z);
            s_flat.extend_from_slice(&sol.s);
            statuses.push(sol.status as i64);
            obj_vals.push(sol.obj_val);
            iters_vec.push(sol.iterations as i64);
        }

        dict.set_item("x", x_flat)?;
        dict.set_item("z", z_flat)?;
        dict.set_item("s", s_flat)?;
        dict.set_item("status", &statuses)?;
        dict.set_item("obj_val", &obj_vals)?;
        dict.set_item("dual_obj_val", &obj_vals)?;
        dict.set_item("iterations", &iters_vec)?;
        dict.set_item("construction_time", 0.0)?;
        dict.set_item("setup_time", 0.0)?;
        dict.set_item("solve_time", solve_time)?;

        Ok(dict)
    }

    /// Compute backward pass. Returns a dict with dP_values, dA_values, dq, db.
    fn backward<'py>(
        &mut self,
        py: Python<'py>,
        dx: &Bound<'_, PyAny>,
        dz: &Bound<'_, PyAny>,
        ds: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dx_vec = extract_f64_buffer(dx)?;
        let dz_vec = extract_f64_buffer(dz)?;
        let ds_vec = extract_f64_buffer(ds)?;
        let n = self.n as usize;
        let m = self.m as usize;
        let bs = self.batch_size as usize;
        if dx_vec.len() != bs * n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dx has {} elements, expected {}",
                dx_vec.len(),
                bs * n
            )));
        }
        if dz_vec.len() != bs * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dz has {} elements, expected {}",
                dz_vec.len(),
                bs * m
            )));
        }
        if ds_vec.len() != bs * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ds has {} elements, expected {}",
                ds_vec.len(),
                bs * m
            )));
        }

        py.allow_threads(|| self.solver.backward(&dx_vec, &dz_vec, &ds_vec))
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let bs = self.batch_size as usize;
        let nnz_p = self.solver.nnz_p() as usize;
        let nnz_a = self.solver.nnz_a() as usize;
        let n = self.n as usize;
        let m = self.m as usize;

        let mut dp_flat = Vec::with_capacity(bs * nnz_p);
        let mut da_flat = Vec::with_capacity(bs * nnz_a);
        let mut dq_flat = Vec::with_capacity(bs * n);
        let mut db_flat = Vec::with_capacity(bs * m);

        for i in 0..bs {
            let grad = self
                .solver
                .get_backward(i as i64)
                .map_err(|e| PyRuntimeError::new_err(e))?;
            dp_flat.extend_from_slice(&grad.dp_values);
            da_flat.extend_from_slice(&grad.da_values);
            dq_flat.extend_from_slice(&grad.dq);
            db_flat.extend_from_slice(&grad.db);
        }

        let dict = PyDict::new(py);
        dict.set_item("dP_values", dp_flat)?;
        dict.set_item("dA_values", da_flat)?;
        dict.set_item("dq", dq_flat)?;
        dict.set_item("db", db_flat)?;
        Ok(dict)
    }

    #[pyo3(signature = (
        dx_flat, ds_flat, dz_flat,
        P_values_flat, A_values_flat,
        q_flat, b_flat,
        x_flat, z_flat, s_flat,
        backward_state,
        shared=true
    ))]
    fn backward_with_data_flat<'py>(
        &mut self,
        py: Python<'py>,
        dx_flat: &Bound<'_, PyAny>,
        ds_flat: &Bound<'_, PyAny>,
        dz_flat: &Bound<'_, PyAny>,
        P_values_flat: &Bound<'_, PyAny>,
        A_values_flat: &Bound<'_, PyAny>,
        q_flat: &Bound<'_, PyAny>,
        b_flat: &Bound<'_, PyAny>,
        x_flat: &Bound<'_, PyAny>,
        z_flat: &Bound<'_, PyAny>,
        s_flat: &Bound<'_, PyAny>,
        backward_state: &PyActiveSetBackwardState,
        shared: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let dx_vec = extract_f64_buffer(dx_flat)?;
        let ds_vec = extract_f64_buffer(ds_flat)?;
        let dz_vec = extract_f64_buffer(dz_flat)?;
        let p_vec = extract_f64_buffer(P_values_flat)?;
        let a_vec = extract_f64_buffer(A_values_flat)?;
        let q_vec = extract_f64_buffer(q_flat)?;
        let b_vec = extract_f64_buffer(b_flat)?;
        let x_vec = extract_f64_buffer(x_flat)?;
        let z_vec = extract_f64_buffer(z_flat)?;
        let s_vec = extract_f64_buffer(s_flat)?;
        let bs = self.batch_size as usize;
        let n = self.n as usize;
        let m = self.m as usize;
        let nnz_p = self.solver.nnz_p() as usize;
        let nnz_a = self.solver.nnz_a() as usize;
        let packed_rinv = n * (n + 1) / 2;
        if backward_state.inner.len() != bs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "backward_state has {} batch elements, expected {}",
                backward_state.inner.len(),
                bs
            )));
        }
        let mut state_rinv_vec = Vec::with_capacity(bs * packed_rinv);
        let mut state_rinv_diag_vec = Vec::with_capacity(bs * n);
        let mut state_use_rinv_diag_vec = Vec::with_capacity(bs);
        let mut state_n_active_vec = Vec::with_capacity(bs);
        let mut state_ws_vec = Vec::with_capacity(bs * m);
        let mut state_sense_vec = Vec::with_capacity(bs * m);
        let mut state_lam_star_vec = Vec::with_capacity(bs * m);
        for state in &backward_state.inner {
            state_rinv_vec.extend_from_slice(&state.rinv);
            state_rinv_diag_vec.extend_from_slice(&state.rinv_diag);
            state_use_rinv_diag_vec.push(state.use_rinv_diag as i32);
            state_n_active_vec.push(state.n_active);
            state_ws_vec.extend_from_slice(&state.ws);
            state_sense_vec.extend_from_slice(&state.sense);
            state_lam_star_vec.extend_from_slice(&state.lam_star);
        }

        if dx_vec.len() != bs * n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dx_flat has {} elements, expected {}",
                dx_vec.len(),
                bs * n
            )));
        }
        if dz_vec.len() != bs * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dz_flat has {} elements, expected {}",
                dz_vec.len(),
                bs * m
            )));
        }
        if ds_vec.len() != bs * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ds_flat has {} elements, expected {}",
                ds_vec.len(),
                bs * m
            )));
        }
        let expect_p = if shared { nnz_p } else { bs * nnz_p };
        let expect_a = if shared { nnz_a } else { bs * nnz_a };
        if p_vec.len() != expect_p {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "P_values_flat has {} elements, expected {}",
                p_vec.len(),
                expect_p
            )));
        }
        if a_vec.len() != expect_a {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "A_values_flat has {} elements, expected {}",
                a_vec.len(),
                expect_a
            )));
        }
        if q_vec.len() != bs * n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q_flat has {} elements, expected {}",
                q_vec.len(),
                bs * n
            )));
        }
        if b_vec.len() != bs * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "b_flat has {} elements, expected {}",
                b_vec.len(),
                bs * m
            )));
        }
        if x_vec.len() != bs * n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "x_flat has {} elements, expected {}",
                x_vec.len(),
                bs * n
            )));
        }
        if z_vec.len() != bs * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "z_flat has {} elements, expected {}",
                z_vec.len(),
                bs * m
            )));
        }
        if s_vec.len() != bs * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "s_flat has {} elements, expected {}",
                s_vec.len(),
                bs * m
            )));
        }
        if state_rinv_vec.len() != bs * packed_rinv {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "state_rinv_flat has {} elements, expected {}",
                state_rinv_vec.len(),
                bs * packed_rinv
            )));
        }
        if state_rinv_diag_vec.len() != bs * n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "state_rinv_diag_flat has {} elements, expected {}",
                state_rinv_diag_vec.len(),
                bs * n
            )));
        }
        if state_use_rinv_diag_vec.len() != bs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "state_use_rinv_diag_flat has {} elements, expected {}",
                state_use_rinv_diag_vec.len(),
                bs
            )));
        }
        if state_n_active_vec.len() != bs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "state_n_active_flat has {} elements, expected {}",
                state_n_active_vec.len(),
                bs
            )));
        }
        if state_ws_vec.len() != bs * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "state_ws_flat has {} elements, expected {}",
                state_ws_vec.len(),
                bs * m
            )));
        }
        if state_sense_vec.len() != bs * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "state_sense_flat has {} elements, expected {}",
                state_sense_vec.len(),
                bs * m
            )));
        }
        if state_lam_star_vec.len() != bs * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "state_lam_star_flat has {} elements, expected {}",
                state_lam_star_vec.len(),
                bs * m
            )));
        }

        py.allow_threads(|| {
            self.solver.backward_with_data(
                &dx_vec,
                &dz_vec,
                &ds_vec,
                &p_vec,
                &a_vec,
                shared,
                &q_vec,
                &b_vec,
                &x_vec,
                &z_vec,
                &s_vec,
                &state_rinv_vec,
                &state_rinv_diag_vec,
                &state_use_rinv_diag_vec,
                &state_n_active_vec,
                &state_ws_vec,
                &state_sense_vec,
                &state_lam_star_vec,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(e))?;

        let mut dp_flat = Vec::with_capacity(bs * nnz_p);
        let mut da_flat = Vec::with_capacity(bs * nnz_a);
        let mut dq_flat = Vec::with_capacity(bs * n);
        let mut db_flat = Vec::with_capacity(bs * m);

        for i in 0..bs {
            let grad = self
                .solver
                .get_backward(i as i64)
                .map_err(|e| PyRuntimeError::new_err(e))?;
            dp_flat.extend_from_slice(&grad.dp_values);
            da_flat.extend_from_slice(&grad.da_values);
            dq_flat.extend_from_slice(&grad.dq);
            db_flat.extend_from_slice(&grad.db);
        }

        let dict = PyDict::new(py);
        dict.set_item("dP_values", dp_flat)?;
        dict.set_item("dA_values", da_flat)?;
        dict.set_item("dq", dq_flat)?;
        dict.set_item("db", db_flat)?;
        Ok(dict)
    }

    fn get_backward_state(&self) -> PyResult<PyActiveSetBackwardState> {
        let mut states = Vec::with_capacity(self.batch_size as usize);
        for i in 0..self.batch_size {
            let state = self
                .solver
                .get_backward_state(i)
                .map_err(|e| PyRuntimeError::new_err(e))?;
            states.push(state);
        }
        Ok(PyActiveSetBackwardState { inner: states })
    }

    #[pyo3(signature = (
        state_rinv_flat, state_rinv_diag_flat,
        state_use_rinv_diag_flat, state_n_active_flat,
        state_ws_flat, state_sense_flat, state_lam_star_flat
    ))]
    fn _make_backward_state_from_flat(
        &self,
        state_rinv_flat: &Bound<'_, PyAny>,
        state_rinv_diag_flat: &Bound<'_, PyAny>,
        state_use_rinv_diag_flat: &Bound<'_, PyAny>,
        state_n_active_flat: &Bound<'_, PyAny>,
        state_ws_flat: &Bound<'_, PyAny>,
        state_sense_flat: &Bound<'_, PyAny>,
        state_lam_star_flat: &Bound<'_, PyAny>,
    ) -> PyResult<PyActiveSetBackwardState> {
        let rinv = extract_f64_buffer(state_rinv_flat)?;
        let rinv_diag = extract_f64_buffer(state_rinv_diag_flat)?;
        let use_rinv_diag = state_use_rinv_diag_flat.extract::<Vec<i32>>()?;
        let n_active = state_n_active_flat.extract::<Vec<i32>>()?;
        let ws = state_ws_flat.extract::<Vec<i32>>()?;
        let sense = state_sense_flat.extract::<Vec<i32>>()?;
        let lam_star = extract_f64_buffer(state_lam_star_flat)?;

        let bs = self.batch_size as usize;
        let n = self.n as usize;
        let m = self.m as usize;
        let packed_rinv = n * (n + 1) / 2;

        if rinv.len() != bs * packed_rinv
            || rinv_diag.len() != bs * n
            || use_rinv_diag.len() != bs
            || n_active.len() != bs
            || ws.len() != bs * m
            || sense.len() != bs * m
            || lam_star.len() != bs * m
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "flat backward-state arrays have inconsistent dimensions",
            ));
        }

        let mut states = Vec::with_capacity(bs);
        for i in 0..bs {
            states.push(active_set_ffi::ActiveSetBackwardState {
                rinv: rinv[i * packed_rinv..(i + 1) * packed_rinv].to_vec(),
                rinv_diag: rinv_diag[i * n..(i + 1) * n].to_vec(),
                ws: ws[i * m..(i + 1) * m].to_vec(),
                sense: sense[i * m..(i + 1) * m].to_vec(),
                lam_star: lam_star[i * m..(i + 1) * m].to_vec(),
                n_active: n_active[i],
                use_rinv_diag: use_rinv_diag[i] != 0,
            });
        }

        Ok(PyActiveSetBackwardState { inner: states })
    }
}
