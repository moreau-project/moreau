// Python wrappers and interface for the CompiledSolver
// implementation and its related types.

#![allow(non_snake_case)]
use super::*;
use crate::solver::implementations::default::*;
use pyo3::prelude::*;
use std::os::raw::{c_char, c_int, c_void};

// ----------------------------------
// BatchProblem
// ----------------------------------

/// A single problem in a batch - stores only values, not structure
#[pyclass(name = "BatchProblem")]
pub struct PyBatchProblem {
    /// Values for P matrix (CSR order, must match pattern structure)
    #[pyo3(get, set)]
    pub P_values: Vec<f64>,
    /// Objective vector q
    #[pyo3(get, set)]
    pub q: Vec<f64>,
    /// Values for A matrix (CSR order, must match pattern structure)
    #[pyo3(get, set)]
    pub A_values: Vec<f64>,
    /// Constraint vector b
    #[pyo3(get, set)]
    pub b: Vec<f64>,
}

#[pymethods]
impl PyBatchProblem {
    #[new]
    fn new(P_values: Vec<f64>, q: Vec<f64>, A_values: Vec<f64>, b: Vec<f64>) -> Self {
        Self {
            P_values,
            q,
            A_values,
            b,
        }
    }

    /// Create from scipy sparse matrices (extracts only values)
    #[staticmethod]
    fn from_matrices(
        P: Py<PyAny>,
        q: Vec<f64>,
        A: Py<PyAny>,
        b: Vec<f64>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let P_values: Vec<f64> = P.bind(py).getattr("data")?.extract()?;
        let A_values: Vec<f64> = A.bind(py).getattr("data")?.extract()?;
        Ok(Self {
            P_values,
            q,
            A_values,
            b,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchProblem(n={}, m={}, nnz_P={}, nnz_A={})",
            self.q.len(),
            self.b.len(),
            self.P_values.len(),
            self.A_values.len()
        )
    }
}

// Convert Python BatchProblem to Rust BatchProblem
impl PyBatchProblem {
    fn to_rust(&self) -> BatchProblem<f64> {
        BatchProblem {
            P_values: self.P_values.clone(),
            q: self.q.clone(),
            A_values: self.A_values.clone(),
            b: self.b.clone(),
        }
    }
}

// ----------------------------------
// UpstreamGradients
// ----------------------------------

#[derive(Clone)]
#[pyclass(name = "UpstreamGradients")]
pub struct PyUpstreamGradients {
    /// Gradient w.r.t. primal variables x (dL/dx)
    #[pyo3(get, set)]
    pub dx: Vec<f64>,
    /// Gradient w.r.t. slack variables s (dL/ds)
    #[pyo3(get, set)]
    pub ds: Vec<f64>,
    /// Gradient w.r.t. dual variables z (dL/dz)
    #[pyo3(get, set)]
    pub dz: Vec<f64>,
    /// Gradient w.r.t. direct-x cone duals z_x (dL/dz_x). Empty if the
    /// solver has no direct-x cones; otherwise must match the flat
    /// `Solution.z_x` length (sum of cone dimensions, in spec order).
    #[pyo3(get, set)]
    pub dz_x: Vec<f64>,
}

#[pymethods]
impl PyUpstreamGradients {
    #[new]
    #[pyo3(signature = (dx, ds, dz, dz_x=None))]
    fn new(dx: Vec<f64>, ds: Vec<f64>, dz: Vec<f64>, dz_x: Option<Vec<f64>>) -> Self {
        Self {
            dx,
            ds,
            dz,
            dz_x: dz_x.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "UpstreamGradients(n={}, m={})",
            self.dx.len(),
            self.ds.len()
        )
    }
}

impl From<PyUpstreamGradients> for UpstreamGradients<f64> {
    fn from(py_grads: PyUpstreamGradients) -> Self {
        Self {
            dx: py_grads.dx,
            ds: py_grads.ds,
            dz: py_grads.dz,
            dz_x: py_grads.dz_x,
        }
    }
}

// ----------------------------------
// ComputedGradients
// ----------------------------------

#[derive(Clone)]
#[pyclass(name = "ComputedGradients")]
pub struct PyComputedGradients {
    /// Gradient values w.r.t. P matrix (CSR order, matches P sparsity pattern)
    #[pyo3(get)]
    pub dP_values: Vec<f64>,
    /// Gradient w.r.t. q vector
    #[pyo3(get)]
    pub dq: Vec<f64>,
    /// Gradient values w.r.t. A matrix (CSR order, matches A sparsity pattern)
    #[pyo3(get)]
    pub dA_values: Vec<f64>,
    /// Gradient w.r.t. b vector
    #[pyo3(get)]
    pub db: Vec<f64>,
    #[cfg(debug_assertions)]
    #[pyo3(get)]
    pub debug_smoothing_x: Vec<f64>,
    #[cfg(debug_assertions)]
    #[pyo3(get)]
    pub debug_smoothing_z: Vec<f64>,
    #[cfg(debug_assertions)]
    #[pyo3(get)]
    pub debug_smoothing_s: Vec<f64>,
}

#[pymethods]
impl PyComputedGradients {
    fn __repr__(&self) -> String {
        format!(
            "ComputedGradients(n={}, m={}, nnz_P={}, nnz_A={})",
            self.dq.len(),
            self.db.len(),
            self.dP_values.len(),
            self.dA_values.len()
        )
    }
}

// Buffer protocol FFI — in stable ABI since Python 3.2 but pyo3-ffi
// gates behind Py_3_11. We define the minimal subset directly.
const PY_BUF_FORMAT: c_int = 0x0004;
const PY_BUF_ND: c_int = 0x0008;
const PY_BUF_STRIDES: c_int = 0x0010;
const PY_BUF_C_CONTIGUOUS: c_int = PY_BUF_ND | PY_BUF_STRIDES | 0x0020;

#[repr(C)]
struct Py_buffer {
    buf: *mut c_void,
    obj: *mut pyo3::ffi::PyObject,
    len: pyo3::ffi::Py_ssize_t,
    itemsize: pyo3::ffi::Py_ssize_t,
    readonly: c_int,
    ndim: c_int,
    format: *mut c_char,
    shape: *mut pyo3::ffi::Py_ssize_t,
    strides: *mut pyo3::ffi::Py_ssize_t,
    suboffsets: *mut pyo3::ffi::Py_ssize_t,
    internal: *mut c_void,
}

unsafe extern "C" {
    fn PyObject_GetBuffer(
        obj: *mut pyo3::ffi::PyObject,
        view: *mut Py_buffer,
        flags: c_int,
    ) -> c_int;
    fn PyBuffer_Release(view: *mut Py_buffer);
}

/// Read a contiguous f64 buffer from a Python object via buffer protocol.
pub(crate) fn extract_f64_buffer(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    unsafe {
        let mut view: Py_buffer = std::mem::zeroed();
        if PyObject_GetBuffer(obj.as_ptr(), &mut view, PY_BUF_C_CONTIGUOUS | PY_BUF_FORMAT) != 0 {
            return Err(PyErr::fetch(obj.py()));
        }
        let fmt = if view.format.is_null() {
            b"B" as &[u8]
        } else {
            std::ffi::CStr::from_ptr(view.format).to_bytes()
        };
        if fmt != b"d" {
            PyBuffer_Release(&mut view);
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "Expected float64 buffer",
            ));
        }
        let len = view.len as usize / 8;
        let data = std::slice::from_raw_parts(view.buf as *const f64, len).to_vec();
        PyBuffer_Release(&mut view);
        Ok(data)
    }
}

/// Batched solution with flat arrays.
#[pyclass(name = "BatchedSolutionFlat")]
pub struct PyBatchedSolutionFlat {
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub s: Vec<f64>,
    #[pyo3(get)]
    pub z: Vec<f64>,
    /// Direct-x cone duals, flat (length = `batch_size * total_xn`).
    /// Empty for slack-only problems.
    #[pyo3(get)]
    pub z_x: Vec<f64>,
    #[pyo3(get)]
    pub status: Vec<i32>,
    #[pyo3(get)]
    pub obj_val: Vec<f64>,
    #[pyo3(get)]
    pub obj_val_dual: Vec<f64>,
    #[pyo3(get)]
    pub iterations: Vec<u32>,
    #[pyo3(get)]
    pub construction_time: f64,
    #[pyo3(get)]
    pub setup_time: f64,
    #[pyo3(get)]
    pub solve_time: f64,
}

/// Batched gradients with flat arrays.
#[pyclass(name = "BatchedGradientsFlat")]
pub struct PyBatchedGradientsFlat {
    #[pyo3(get)]
    pub dP_values: Vec<f64>,
    #[pyo3(get)]
    pub dq: Vec<f64>,
    #[pyo3(get)]
    pub dA_values: Vec<f64>,
    #[pyo3(get)]
    pub db: Vec<f64>,
}

// ----------------------------------
// CompiledSolver
// ----------------------------------

/// Parallel batch solver using two-step API pattern.
///
/// Constructor takes CSR structure (row_offsets, col_indices), solve() takes values.
/// All values (input and output) are in CSR order.
#[pyclass(name = "CompiledSolver")]
pub struct PyCompiledSolver {
    inner: CompiledSolver<f64>,
}

#[pymethods]
impl PyCompiledSolver {
    /// Create a new batch solver.
    ///
    /// Two-step API: structure at construction, values at solve().
    ///
    /// # Arguments
    /// * `n` - Number of primal variables
    /// * `m` - Number of constraints
    /// * `P_row_offsets` - CSR row pointers for P matrix
    /// * `P_col_indices` - CSR column indices for P matrix
    /// * `A_row_offsets` - CSR row pointers for A matrix
    /// * `A_col_indices` - CSR column indices for A matrix
    /// * `cones` - Cone constraints specification
    /// * `settings` - Optional solver settings
    /// * `num_threads` - Number of threads for parallel solve (None = auto-detect)
    /// * `enable_grad` - If True, pre-compute gradient structures for backward() (default: False)
    #[new]
    #[pyo3(signature = (n, m, P_row_offsets, P_col_indices, A_row_offsets, A_col_indices, cones, settings=None, num_threads=None, enable_grad=false, x_cones=None, b_sparsity_pattern=None))]
    fn new(
        n: usize,
        m: usize,
        P_row_offsets: Vec<usize>,
        P_col_indices: Vec<usize>,
        A_row_offsets: Vec<usize>,
        A_col_indices: Vec<usize>,
        cones: Vec<PySupportedCone>,
        settings: Option<PyDefaultSettings>,
        num_threads: Option<usize>,
        enable_grad: bool,
        x_cones: Option<Vec<super::cones_py::PySupportedXCone>>,
        b_sparsity_pattern: Option<Vec<bool>>,
    ) -> PyResult<Self> {
        let cones = _py_to_native_cones(cones);
        let x_cones: Vec<crate::solver::core::cones::SupportedXConeT> = x_cones
            .unwrap_or_default()
            .into_iter()
            .map(Into::into)
            .collect();

        let settings = match settings {
            Some(s) => s.to_internal()?,
            None => DefaultSettings::default(),
        };

        // Auto-detect num_threads if not specified
        let num_threads = num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4)
        });

        let inner = CompiledSolver::new_with_b_nnz_mask_and_xcones(
            n,
            m,
            &P_row_offsets,
            &P_col_indices,
            &A_row_offsets,
            &A_col_indices,
            &cones,
            &x_cones,
            settings,
            num_threads,
            enable_grad,
            b_sparsity_pattern.as_deref(),
        )?;

        Ok(Self { inner })
    }

    /// Check if gradient computation is enabled
    fn grad_enabled(&self) -> bool {
        self.inner.grad_enabled()
    }

    /// Solve a batch of problems in parallel
    ///
    /// Args:
    ///     problems: List of BatchProblem instances with values in CSR order
    ///
    /// Returns:
    ///     List of DefaultSolution instances
    fn solve_batch_parallel(
        &self,
        py: Python<'_>,
        problems: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<Vec<PyDefaultSolution>> {
        // Convert Python list to Rust types (values are already in CSR order)
        let rust_problems: Vec<BatchProblem<f64>> = problems
            .iter()
            .map(|item| {
                let py_problem: PyRef<PyBatchProblem> = item.extract()?;
                Ok(py_problem.to_rust())
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Run batch processing (release GIL during computation)
        let solutions = py.allow_threads(|| self.inner.solve_batch_parallel(&rust_problems))?;

        // Convert solutions back to Python types
        Ok(solutions
            .iter()
            .map(|s| PyDefaultSolution::from(s))
            .collect())
    }

    /// Compute gradients using cached state from the last solve.
    ///
    /// This method uses the cached solutions and equilibration factors from
    /// the last solve_batch_parallel call, avoiding the need to re-send
    /// problem data or re-solve.
    ///
    /// Args:
    ///     upstream_grads: List of UpstreamGradients (dx, ds, dz)
    ///
    /// Returns:
    ///     List of ComputedGradients with dP_values and dA_values in CSR order
    ///
    /// Raises:
    ///     RuntimeError: If solve_batch_parallel was not called first
    fn backward(
        &self,
        py: Python<'_>,
        upstream_grads: &Bound<'_, pyo3::types::PyList>,
    ) -> PyResult<Vec<PyComputedGradients>> {
        let rust_grads: Vec<UpstreamGradients<f64>> = upstream_grads
            .iter()
            .map(|item| {
                let py_grads: PyRef<PyUpstreamGradients> = item.extract()?;
                Ok(UpstreamGradients {
                    dx: py_grads.dx.clone(),
                    ds: py_grads.ds.clone(),
                    dz: py_grads.dz.clone(),
                    dz_x: py_grads.dz_x.clone(),
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        // Run backward pass (release GIL during computation)
        let computed = py.allow_threads(|| self.inner.backward(&rust_grads))?;

        // Convert results back to Python types (gradients are already in CSR order)
        Ok(computed
            .into_iter()
            .map(|grads| PyComputedGradients {
                dP_values: grads.dP_values,
                dq: grads.dq,
                dA_values: grads.dA_values,
                db: grads.db,
                #[cfg(debug_assertions)]
                debug_smoothing_x: grads.debug_smoothing_x,
                #[cfg(debug_assertions)]
                debug_smoothing_z: grads.debug_smoothing_z,
                #[cfg(debug_assertions)]
                debug_smoothing_s: grads.debug_smoothing_s,
            })
            .collect())
    }

    /// Set P and A matrix values for a batch and precompute equilibration.
    ///
    /// This is the primary setup method. Must be called before solve().
    /// Pass a list of P values and A values, one per problem in the batch.
    ///
    /// Args:
    ///     P_values_batch: List of P matrix values (one Vec per problem, CSR order)
    ///     A_values_batch: List of A matrix values (one Vec per problem, CSR order)
    fn setup(
        &mut self,
        P_values_batch: Vec<Vec<f64>>,
        A_values_batch: Vec<Vec<f64>>,
    ) -> PyResult<()> {
        self.inner
            .setup(&P_values_batch, &A_values_batch)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Solve a batch of problems with optional warm starting.
    ///
    /// This is the primary solve interface. Requires setup() to be called first.
    ///
    /// Args:
    ///     qs: List of q vectors (linear cost), one per problem
    ///     bs: List of b vectors (constraint RHS), one per problem
    ///     warm_x: Optional warm start primal variables (batch × n)
    ///     warm_z: Optional warm start dual variables (batch × m)
    ///     warm_s: Optional warm start slack variables (batch × m)
    ///
    /// Returns:
    ///     List of DefaultSolution instances
    #[pyo3(signature = (qs, bs, warm_x=None, warm_z=None, warm_s=None, warm_z_x=None))]
    fn solve(
        &self,
        py: Python<'_>,
        qs: Vec<Vec<f64>>,
        bs: Vec<Vec<f64>>,
        warm_x: Option<Vec<Vec<f64>>>,
        warm_z: Option<Vec<Vec<f64>>>,
        warm_s: Option<Vec<Vec<f64>>>,
        warm_z_x: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Vec<PyDefaultSolution>> {
        let solutions = py.allow_threads(|| {
            self.inner.solve_with_warm_start(
                &qs,
                &bs,
                warm_x.as_deref(),
                warm_z.as_deref(),
                warm_s.as_deref(),
                warm_z_x.as_deref(),
            )
        })?;
        Ok(solutions
            .iter()
            .map(|s| PyDefaultSolution::from(s))
            .collect())
    }

    /// Setup with shared P and A matrices for all problems in a batch.
    ///
    /// Use this when all problems share the same P and A matrices but have
    /// different q and b vectors. More efficient than per-problem setup.
    ///
    /// Args:
    ///     P_values: Non-zero values for shared P matrix (CSR order)
    ///     A_values: Non-zero values for shared A matrix (CSR order)
    ///     batch_size: Number of problems to solve
    fn setup_shared(&mut self, P_values: Vec<f64>, A_values: Vec<f64>, batch_size: usize) {
        self.inner.setup_shared(&P_values, &A_values, batch_size);
    }

    /// Solve a batch of problems with shared P and A matrices.
    ///
    /// Requires setup_shared() to be called first. All problems share the same
    /// P and A matrices (set via setup_shared) but have different q and b.
    ///
    /// Args:
    ///     q_batch: Linear cost vectors flattened, shape (batch * n,)
    ///     b_batch: Constraint RHS vectors flattened, shape (batch * m,)
    ///     batch_size: Number of problems
    ///
    /// Returns:
    ///     List of DefaultSolution instances
    fn solve_batch_shared(
        &self,
        py: Python<'_>,
        q_batch: Vec<f64>,
        b_batch: Vec<f64>,
        batch_size: usize,
    ) -> PyResult<Vec<PyDefaultSolution>> {
        let solutions = py.allow_threads(|| {
            self.inner
                .solve_batch_shared(&q_batch, &b_batch, batch_size)
        })?;
        Ok(solutions
            .iter()
            .map(|s| PyDefaultSolution::from(s))
            .collect())
    }

    /// Precompute equilibration for solves where only q and b change.
    ///
    /// It caches equilibration factors that are reused across multiple solves.
    ///
    /// Args:
    ///     P_values: Non-zero values for P matrix (CSR order)
    ///     q: Objective vector
    ///     A_values: Non-zero values for A matrix (CSR order)
    ///     b: Initial constraint RHS vector
    fn precompute_equilibration(
        &mut self,
        P_values: Vec<f64>,
        q: Vec<f64>,
        A_values: Vec<f64>,
        b: Vec<f64>,
    ) {
        self.inner
            .precompute_equilibration(&P_values, &q, &A_values, &b);
    }

    /// Check if there is a valid cache from a previous solve
    fn has_cache(&self) -> bool {
        self.inner.has_cache()
    }

    /// Get the size of the cached batch (0 if no cache)
    fn cache_size(&self) -> usize {
        self.inner.cache_size()
    }

    /// Get the number of threads configured for this batch solver
    fn num_threads(&self) -> usize {
        self.inner.num_threads()
    }

    /// Get problem dimensions (n, m)
    fn dims(&self) -> (usize, usize) {
        self.inner.dims()
    }

    /// Get number of non-zeros in P
    fn nnz_P(&self) -> usize {
        self.inner.nnz_P()
    }

    /// Get number of non-zeros in A
    fn nnz_A(&self) -> usize {
        self.inner.nnz_A()
    }

    /// Setup from flat contiguous numpy arrays via buffer protocol.
    fn setup_flat(
        &mut self,
        P_values_flat: &Bound<'_, PyAny>,
        A_values_flat: &Bound<'_, PyAny>,
        batch_size: usize,
    ) -> PyResult<()> {
        let p = extract_f64_buffer(P_values_flat)?;
        let a = extract_f64_buffer(A_values_flat)?;
        self.inner
            .setup_flat(&p, &a, batch_size)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Solve from flat contiguous numpy arrays, return flat results.
    #[pyo3(signature = (qs_flat, bs_flat, batch_size, warm_x_flat=None, warm_z_flat=None, warm_s_flat=None, warm_z_x_flat=None))]
    fn solve_flat(
        &self,
        py: Python<'_>,
        qs_flat: &Bound<'_, PyAny>,
        bs_flat: &Bound<'_, PyAny>,
        batch_size: usize,
        warm_x_flat: Option<&Bound<'_, PyAny>>,
        warm_z_flat: Option<&Bound<'_, PyAny>>,
        warm_s_flat: Option<&Bound<'_, PyAny>>,
        warm_z_x_flat: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyBatchedSolutionFlat> {
        let q = extract_f64_buffer(qs_flat)?;
        let b = extract_f64_buffer(bs_flat)?;
        let wx = warm_x_flat.map(|v| extract_f64_buffer(v)).transpose()?;
        let wz = warm_z_flat.map(|v| extract_f64_buffer(v)).transpose()?;
        let ws = warm_s_flat.map(|v| extract_f64_buffer(v)).transpose()?;
        let wzx = warm_z_x_flat.map(|v| extract_f64_buffer(v)).transpose()?;

        let solutions = py.allow_threads(|| {
            self.inner.solve_flat(
                &q,
                &b,
                batch_size,
                wx.as_deref(),
                wz.as_deref(),
                ws.as_deref(),
                wzx.as_deref(),
            )
        })?;

        let (n, m) = self.inner.dims();
        let mut x = Vec::with_capacity(batch_size * n);
        let mut s = Vec::with_capacity(batch_size * m);
        let mut z = Vec::with_capacity(batch_size * m);
        let mut z_x: Vec<f64> = Vec::new();
        let mut status = Vec::with_capacity(batch_size);
        let mut obj_val = Vec::with_capacity(batch_size);
        let mut obj_val_dual = Vec::with_capacity(batch_size);
        let mut iterations = Vec::with_capacity(batch_size);
        for sol in &solutions {
            x.extend_from_slice(&sol.x);
            s.extend_from_slice(&sol.s);
            z.extend_from_slice(&sol.z);
            z_x.extend_from_slice(&sol.z_x);
            status.push(sol.status as i32);
            obj_val.push(sol.obj_val);
            obj_val_dual.push(sol.obj_val_dual);
            iterations.push(sol.iterations);
        }
        // Index into solutions[0] would panic on batch_size=0; with
        // panic="abort" set in the release profile this aborts the whole
        // Python interpreter (including Jupyter kernels). Surface as a
        // PyValueError instead. (#189)
        let first = solutions.first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(
                "solve_flat called with batch_size=0; need at least one problem to solve",
            )
        })?;
        Ok(PyBatchedSolutionFlat {
            x,
            s,
            z,
            z_x,
            status,
            obj_val,
            obj_val_dual,
            iterations,
            construction_time: first.construction_time,
            setup_time: first.setup_time,
            solve_time: first.solve_time,
        })
    }

    /// Backward pass from flat contiguous numpy arrays, return flat gradients.
    ///
    /// `dz_x_flat` is `None` for slack-only problems; for direct-x cones it
    /// must have shape `(batch_size * total_xn,)` matching the order returned
    /// by `solve_flat`'s `z_x` field.
    #[pyo3(signature = (dx_flat, ds_flat, dz_flat, batch_size, dz_x_flat=None))]
    fn backward_flat(
        &self,
        py: Python<'_>,
        dx_flat: &Bound<'_, PyAny>,
        ds_flat: &Bound<'_, PyAny>,
        dz_flat: &Bound<'_, PyAny>,
        batch_size: usize,
        dz_x_flat: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyBatchedGradientsFlat> {
        let (n, m) = self.inner.dims();
        let dx = extract_f64_buffer(dx_flat)?;
        let ds = extract_f64_buffer(ds_flat)?;
        let dz = extract_f64_buffer(dz_flat)?;
        let dz_x_buf = dz_x_flat.map(|v| extract_f64_buffer(v)).transpose()?;

        if dx.len() != batch_size * n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dx_flat length {} != batch_size ({}) * n ({})",
                dx.len(),
                batch_size,
                n
            )));
        }
        if ds.len() != batch_size * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ds_flat length {} != batch_size ({}) * m ({})",
                ds.len(),
                batch_size,
                m
            )));
        }
        if dz.len() != batch_size * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dz_flat length {} != batch_size ({}) * m ({})",
                dz.len(),
                batch_size,
                m
            )));
        }

        // Validate dz_x_flat length and per-batch slicing. xn (per-problem
        // direct-x dimension) is implicit: total_len / batch_size.
        let xn_per: usize = match &dz_x_buf {
            Some(v) if !v.is_empty() => {
                if v.len() % batch_size != 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "dz_x_flat length {} not divisible by batch_size {}",
                        v.len(),
                        batch_size
                    )));
                }
                v.len() / batch_size
            }
            _ => 0,
        };

        let upstream: Vec<UpstreamGradients<f64>> = (0..batch_size)
            .map(|i| {
                let dx_i = dx[i * n..(i + 1) * n].to_vec();
                let ds_i = if m > 0 {
                    ds[i * m..(i + 1) * m].to_vec()
                } else {
                    vec![]
                };
                let dz_i = if m > 0 {
                    dz[i * m..(i + 1) * m].to_vec()
                } else {
                    vec![]
                };
                let dz_x_i = match &dz_x_buf {
                    Some(v) if xn_per > 0 => v[i * xn_per..(i + 1) * xn_per].to_vec(),
                    _ => vec![],
                };
                UpstreamGradients {
                    dx: dx_i,
                    ds: ds_i,
                    dz: dz_i,
                    dz_x: dz_x_i,
                }
            })
            .collect();

        let computed = py.allow_threads(|| self.inner.backward(&upstream))?;

        let (nnz_p, nnz_a) = (self.inner.nnz_P(), self.inner.nnz_A());
        let batch = computed.len();
        let mut dp = Vec::with_capacity(batch * nnz_p);
        let mut dq = Vec::with_capacity(batch * n);
        let mut da = Vec::with_capacity(batch * nnz_a);
        let mut db = Vec::with_capacity(batch * m);
        for g in &computed {
            dp.extend_from_slice(&g.dP_values);
            dq.extend_from_slice(&g.dq);
            da.extend_from_slice(&g.dA_values);
            db.extend_from_slice(&g.db);
        }
        Ok(PyBatchedGradientsFlat {
            dP_values: dp,
            dq,
            dA_values: da,
            db,
        })
    }

    /// Compute gradients from explicitly provided problem data and solution.
    ///
    /// Unlike backward_flat(), this does not require a prior solve() call.
    /// All inputs are flat contiguous arrays: batch_size * per-problem-size.
    ///
    /// Args:
    ///     dx_flat: upstream gradient w.r.t. x, shape (batch_size * n,)
    ///     ds_flat: upstream gradient w.r.t. s, shape (batch_size * m,)
    ///     dz_flat: upstream gradient w.r.t. z, shape (batch_size * m,)
    ///     P_values_flat: P matrix values in CSR order, shape (batch_size * nnz_P,)
    ///     A_values_flat: A matrix values in CSR order, shape (batch_size * nnz_A,)
    ///     q_flat: linear cost vector, shape (batch_size * n,)
    ///     b_flat: constraint RHS, shape (batch_size * m,)
    ///     x_flat: primal solution, shape (batch_size * n,)
    ///     z_flat: dual solution, shape (batch_size * m,)
    ///     s_flat: slack solution, shape (batch_size * m,)
    ///     batch_size: number of problems
    #[pyo3(signature = (
        dx_flat, ds_flat, dz_flat, P_values_flat, A_values_flat, q_flat,
        b_flat, x_flat, z_flat, s_flat, batch_size,
        z_x_flat=None, dz_x_flat=None
    ))]
    fn backward_with_data_flat(
        &self,
        py: Python<'_>,
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
        batch_size: usize,
        z_x_flat: Option<&Bound<'_, PyAny>>,
        dz_x_flat: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyBatchedGradientsFlat> {
        let (n, m) = self.inner.dims();
        let nnz_p = self.inner.nnz_P();
        let nnz_a = self.inner.nnz_A();

        let dx = extract_f64_buffer(dx_flat)?;
        let ds = extract_f64_buffer(ds_flat)?;
        let dz = extract_f64_buffer(dz_flat)?;
        let p_vals = extract_f64_buffer(P_values_flat)?;
        let a_vals = extract_f64_buffer(A_values_flat)?;
        let q_vals = extract_f64_buffer(q_flat)?;
        let b_vals = extract_f64_buffer(b_flat)?;
        let x_vals = extract_f64_buffer(x_flat)?;
        let z_vals = extract_f64_buffer(z_flat)?;
        let s_vals = extract_f64_buffer(s_flat)?;

        // Validate sizes
        if dx.len() != batch_size * n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dx_flat length {} != batch_size ({}) * n ({})",
                dx.len(),
                batch_size,
                n
            )));
        }
        if ds.len() != batch_size * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "ds_flat length {} != batch_size ({}) * m ({})",
                ds.len(),
                batch_size,
                m
            )));
        }
        if dz.len() != batch_size * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "dz_flat length {} != batch_size ({}) * m ({})",
                dz.len(),
                batch_size,
                m
            )));
        }
        if p_vals.len() != batch_size * nnz_p {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "P_values_flat length {} != batch_size ({}) * nnz_P ({})",
                p_vals.len(),
                batch_size,
                nnz_p
            )));
        }
        if a_vals.len() != batch_size * nnz_a {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "A_values_flat length {} != batch_size ({}) * nnz_A ({})",
                a_vals.len(),
                batch_size,
                nnz_a
            )));
        }
        if q_vals.len() != batch_size * n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "q_flat length {} != batch_size ({}) * n ({})",
                q_vals.len(),
                batch_size,
                n
            )));
        }
        if b_vals.len() != batch_size * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "b_flat length {} != batch_size ({}) * m ({})",
                b_vals.len(),
                batch_size,
                m
            )));
        }
        if x_vals.len() != batch_size * n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "x_flat length {} != batch_size ({}) * n ({})",
                x_vals.len(),
                batch_size,
                n
            )));
        }
        if z_vals.len() != batch_size * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "z_flat length {} != batch_size ({}) * m ({})",
                z_vals.len(),
                batch_size,
                m
            )));
        }
        if s_vals.len() != batch_size * m {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "s_flat length {} != batch_size ({}) * m ({})",
                s_vals.len(),
                batch_size,
                m
            )));
        }

        // Optional direct-x batches: solution z_x and upstream gradient dz_x.
        let xn = self.inner.total_xcone_dim();
        let z_x_buf = z_x_flat.map(|v| extract_f64_buffer(v)).transpose()?;
        let dz_x_buf = dz_x_flat.map(|v| extract_f64_buffer(v)).transpose()?;
        if let Some(zx) = &z_x_buf {
            if !zx.is_empty() && zx.len() != batch_size * xn {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "z_x_flat length {} != batch_size ({}) * total_xn ({})",
                    zx.len(),
                    batch_size,
                    xn
                )));
            }
        }
        if let Some(dzx) = &dz_x_buf {
            if !dzx.is_empty() && dzx.len() != batch_size * xn {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "dz_x_flat length {} != batch_size ({}) * total_xn ({})",
                    dzx.len(),
                    batch_size,
                    xn
                )));
            }
        }

        // Chunk into per-problem Vecs
        let upstream: Vec<UpstreamGradients<f64>> = (0..batch_size)
            .map(|i| {
                let dx_i = dx[i * n..(i + 1) * n].to_vec();
                let ds_i = if m > 0 {
                    ds[i * m..(i + 1) * m].to_vec()
                } else {
                    vec![]
                };
                let dz_i = if m > 0 {
                    dz[i * m..(i + 1) * m].to_vec()
                } else {
                    vec![]
                };
                let dz_x_i = match &dz_x_buf {
                    Some(v) if v.len() == batch_size * xn && xn > 0 => {
                        v[i * xn..(i + 1) * xn].to_vec()
                    }
                    _ => vec![],
                };
                UpstreamGradients {
                    dx: dx_i,
                    ds: ds_i,
                    dz: dz_i,
                    dz_x: dz_x_i,
                }
            })
            .collect();

        // Helper: chunks_exact panics on chunk_size=0, so produce empty vecs instead
        let chunk = |data: &[f64], sz: usize| -> Vec<Vec<f64>> {
            if sz == 0 {
                vec![vec![]; batch_size]
            } else {
                data.chunks_exact(sz).map(|c| c.to_vec()).collect()
            }
        };
        let p_batch = chunk(&p_vals, nnz_p);
        let a_batch = chunk(&a_vals, nnz_a);
        let q_batch = chunk(&q_vals, n);
        let b_batch = chunk(&b_vals, m);
        let x_batch = chunk(&x_vals, n);
        let z_batch = chunk(&z_vals, m);
        let s_batch = chunk(&s_vals, m);
        let z_x_batch_v: Vec<Vec<f64>> = match &z_x_buf {
            Some(v) if !v.is_empty() && xn > 0 => chunk(v, xn),
            _ => Vec::new(),
        };

        let computed = py.allow_threads(|| {
            self.inner.backward_with_data_and_z_x(
                &upstream,
                &p_batch,
                &a_batch,
                &q_batch,
                &b_batch,
                &x_batch,
                &z_batch,
                &s_batch,
                &z_x_batch_v,
            )
        })?;

        let batch = computed.len();
        let mut dp = Vec::with_capacity(batch * nnz_p);
        let mut dq_out = Vec::with_capacity(batch * n);
        let mut da = Vec::with_capacity(batch * nnz_a);
        let mut db_out = Vec::with_capacity(batch * m);
        for g in &computed {
            dp.extend_from_slice(&g.dP_values);
            dq_out.extend_from_slice(&g.dq);
            da.extend_from_slice(&g.dA_values);
            db_out.extend_from_slice(&g.db);
        }
        Ok(PyBatchedGradientsFlat {
            dP_values: dp,
            dq: dq_out,
            dA_values: da,
            db: db_out,
        })
    }

    fn __repr__(&self) -> String {
        let (n, m) = self.inner.dims();
        format!(
            "CompiledSolver(n={}, m={}, threads={})",
            n,
            m,
            self.inner.num_threads()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_batch_problem_fields() {
        let problem = PyBatchProblem {
            P_values: vec![2.0, 2.0],
            q: vec![0.0, 0.0],
            A_values: vec![1.0, 1.0, 1.0, 1.0],
            b: vec![1.0, 0.0, 0.0],
        };

        assert_eq!(problem.P_values.len(), 2);
        assert_eq!(problem.q.len(), 2);
        assert_eq!(problem.A_values.len(), 4);
        assert_eq!(problem.b.len(), 3);
    }

    #[test]
    fn test_py_batch_problem_to_rust() {
        let problem = PyBatchProblem {
            P_values: vec![2.0, 2.0],
            q: vec![0.0, 0.0],
            A_values: vec![1.0, 1.0, 1.0, 1.0],
            b: vec![1.0, 0.0, 0.0],
        };

        let rust_problem = problem.to_rust();
        assert_eq!(rust_problem.P_values.len(), 2);
        assert_eq!(rust_problem.q.len(), 2);
        assert_eq!(rust_problem.A_values.len(), 4);
        assert_eq!(rust_problem.b.len(), 3);
    }
}
