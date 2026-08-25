// Python wrappers and interface for the Default solver
// implementation and its related types.

#![allow(non_snake_case)]
use super::*;
use crate::{
    algebra::CsrMatrix,
    solver::{
        core::{
            kktsolvers::LinearSolverInfo,
            traits::{InfoPrint, Settings},
            IPSolver, SettingsError, SolverStatus,
        },
        implementations::default::{diff::BackwardResult, IPMSettings, SolverType, *},
    },
};
use derive_more::with_trait::Debug;
use pyo3::{exceptions::PyException, prelude::*};
use std::fmt::Write;

//Here we end up repeating several datatypes defined internally
//in the Clarabel default implementation.   We would prefer
//to just apply the PyO3 macros to autoderive these types,
//except there are currently problems using cfg_attr with
//the PyO3 get/set attribute.  Pyo3 also does not seem to
//support autoderivation of python types from Rust structs
//that use generics.   See here:
//
// https://github.com/PyO3/pyo3/issues/780
// https://github.com/PyO3/pyo3/issues/1003
// https://github.com/PyO3/pyo3/issues/1088

// ----------------------------------
// DefaultInfo
// ----------------------------------

#[derive(Clone)]
#[pyclass(name = "LinearSolverInfo")]
pub struct PyLinearSolverInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub threads: usize,
    #[pyo3(get)]
    pub direct: bool,
    #[pyo3(get)]
    pub nnzA: usize,
    #[pyo3(get)]
    pub nnzL: usize,
}

impl From<&LinearSolverInfo> for PyLinearSolverInfo {
    fn from(info: &LinearSolverInfo) -> Self {
        Self {
            name: info.name.clone(),
            threads: info.threads,
            direct: info.direct,
            nnzA: info.nnzA,
            nnzL: info.nnzL,
        }
    }
}

// Must directly implement debug for this so that it appears
// as a nested "LinearSolverInfo" object in the debug output of
// DefaultInfo.   For other types we can just drop the leading
// "Py" prefix when implement __repr__

impl Debug for PyLinearSolverInfo {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("LinearSolverInfo")
            .field("name", &self.name)
            .field("threads", &self.threads)
            .field("direct", &self.direct)
            .field("nnzA", &self.nnzA)
            .field("nnzL", &self.nnzL)
            .finish()
    }
}

#[pymethods]
impl PyLinearSolverInfo {
    pub fn __repr__(&self) -> String {
        let mut s = String::new();
        write!(s, "{:#?}", self).unwrap();
        s
    }
}

#[derive(Debug, Clone)]
#[pyclass(name = "DefaultInfo")]
pub struct PyDefaultInfo {
    #[pyo3(get)]
    pub mu: f64,
    #[pyo3(get)]
    pub sigma: f64,
    #[pyo3(get)]
    pub step_length: f64,
    #[pyo3(get)]
    pub iterations: u32,
    #[pyo3(get)]
    pub cost_primal: f64,
    #[pyo3(get)]
    pub cost_dual: f64,
    #[pyo3(get)]
    pub res_primal: f64,
    #[pyo3(get)]
    pub res_dual: f64,
    #[pyo3(get)]
    pub res_primal_inf: f64,
    #[pyo3(get)]
    pub res_dual_inf: f64,
    #[pyo3(get)]
    pub gap_abs: f64,
    #[pyo3(get)]
    pub gap_rel: f64,
    #[pyo3(get)]
    pub ktratio: f64,
    //
    // prev iterate values deliberately excluded
    // since they are pub(crate) in the solver
    //
    /// Time spent constructing solver structure (seconds)
    #[pyo3(get)]
    pub construction_time: f64,
    /// Time spent setting matrix values (seconds)
    #[pyo3(get)]
    pub setup_time: f64,
    /// Time spent in IPM iterations (seconds)
    #[pyo3(get)]
    pub solve_time: f64,
    #[pyo3(get)]
    pub status: PySolverStatus,
    #[pyo3(get)]
    pub linsolver: PyLinearSolverInfo,
    // print stream intentionally excluded
}

impl From<&DefaultInfo<f64>> for PyDefaultInfo {
    fn from(info: &DefaultInfo<f64>) -> Self {
        let status = (&info.status).into();
        let linsolver = (&info.linsolver).into();
        Self {
            mu: info.mu,
            sigma: info.sigma,
            step_length: info.step_length,
            iterations: info.iterations,
            cost_primal: info.cost_primal,
            cost_dual: info.cost_dual,
            res_primal: info.res_primal,
            res_dual: info.res_dual,
            res_primal_inf: info.res_primal_inf,
            res_dual_inf: info.res_dual_inf,
            gap_abs: info.gap_abs,
            gap_rel: info.gap_rel,
            ktratio: info.ktratio,
            construction_time: info.construction_time,
            setup_time: info.setup_time,
            solve_time: info.solve_time,
            status,
            linsolver,
        }
    }
}

#[pymethods]
impl PyDefaultInfo {
    pub fn __repr__(&self) -> String {
        let mut s = String::new();
        write!(s, "{:#?}", self).unwrap();
        s.replacen("PyDefaultInfo", "DefaultInfo", 1)
    }
}

// ----------------------------------
// DefaultSolution
// ----------------------------------

#[derive(Clone)]
#[pyclass(name = "DefaultSolution")]
pub struct PyDefaultSolution {
    #[pyo3(get)]
    pub x: Vec<f64>,
    #[pyo3(get)]
    pub s: Vec<f64>,
    #[pyo3(get)]
    pub z: Vec<f64>,
    /// Direct-x cone duals (length = sum of `Cones.x_cones` dimensions).
    /// Empty when the solver has no direct-x cones.
    #[pyo3(get)]
    pub z_x: Vec<f64>,
    #[pyo3(get)]
    pub status: PySolverStatus,
    #[pyo3(get)]
    pub obj_val: f64,
    #[pyo3(get)]
    pub obj_val_dual: f64,
    /// Time spent constructing solver structure (seconds)
    #[pyo3(get)]
    pub construction_time: f64,
    /// Time spent setting matrix values (seconds)
    #[pyo3(get)]
    pub setup_time: f64,
    /// Time spent in IPM iterations (seconds)
    #[pyo3(get)]
    pub solve_time: f64,
    #[pyo3(get)]
    pub iterations: u32,
    #[pyo3(get)]
    pub r_prim: f64,
    #[pyo3(get)]
    pub r_dual: f64,
}

impl From<&DefaultSolution<f64>> for PyDefaultSolution {
    fn from(result: &DefaultSolution<f64>) -> Self {
        let x = result.x.clone();
        let s = result.s.clone();
        let z = result.z.clone();
        let z_x = result.z_x.clone();
        let status = (&result.status).into();
        Self {
            x,
            s,
            z,
            z_x,
            obj_val: result.obj_val,
            obj_val_dual: result.obj_val_dual,
            status,
            construction_time: result.construction_time,
            setup_time: result.setup_time,
            solve_time: result.solve_time,
            iterations: result.iterations,
            r_prim: result.r_prim,
            r_dual: result.r_dual,
        }
    }
}

// ----------------------------------
// BackwardResult
// ----------------------------------

#[derive(Clone, Debug)]
#[pyclass(name = "BackwardResult")]
pub struct PyBackwardResult {
    #[pyo3(get)]
    pub dP_values: Vec<f64>,
    #[pyo3(get)]
    pub dq: Vec<f64>,
    #[pyo3(get)]
    pub dA_values: Vec<f64>,
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

impl From<BackwardResult<f64>> for PyBackwardResult {
    fn from(result: BackwardResult<f64>) -> Self {
        Self {
            dP_values: result.dP.nzval,
            dq: result.dq,
            dA_values: result.dA.nzval,
            db: result.db,
            #[cfg(debug_assertions)]
            debug_smoothing_x: result.debug_smoothing_x,
            #[cfg(debug_assertions)]
            debug_smoothing_z: result.debug_smoothing_z,
            #[cfg(debug_assertions)]
            debug_smoothing_s: result.debug_smoothing_s,
        }
    }
}

#[pymethods]
impl PyBackwardResult {
    pub fn __repr__(&self) -> String {
        format!(
            "BackwardResult(dP_values=[{}], dq=[{}], dA_values=[{}], db=[{}])",
            self.dP_values.len(),
            self.dq.len(),
            self.dA_values.len(),
            self.db.len()
        )
    }
}

struct TruncatedSlice<'a, T>(&'a [T]);
impl<T> Debug for TruncatedSlice<'_, T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let MAX = 5;
        if self.0.len() > MAX {
            let truncated: Vec<String> = self.0[..MAX - 2]
                .iter()
                .map(|v| format!("{:?}", v))
                .collect();
            write!(
                fmt,
                "[{} ... ({} more)]",
                truncated.join(", "),
                self.0.len() - truncated.len()
            )
        } else {
            write!(fmt, "{:?}", self.0)
        }
    }
}

impl Debug for PyDefaultSolution {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("PyDefaultSolution")
            .field("x", &TruncatedSlice(&self.x))
            .field("s", &TruncatedSlice(&self.s))
            .field("z", &TruncatedSlice(&self.z))
            .field("status", &self.status)
            .field("obj_val", &self.obj_val)
            .field("obj_val_dual", &self.obj_val_dual)
            .field("construction_time", &self.construction_time)
            .field("setup_time", &self.setup_time)
            .field("solve_time", &self.solve_time)
            .field("iterations", &self.iterations)
            .field("r_prim", &self.r_prim)
            .field("r_dual", &self.r_dual)
            .finish()
    }
}

#[pymethods]
impl PyDefaultSolution {
    pub fn __repr__(&self) -> String {
        let mut s = String::new();
        write!(s, "{:#?}", self).unwrap();
        s.replacen("PyDefaultSolution", "DefaultSolution", 1)
    }
}

// ----------------------------------
// Solver Status
// ----------------------------------

impl From<SolverError> for PyErr {
    fn from(err: SolverError) -> Self {
        PyException::new_err(err.to_string())
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
#[pyclass(eq, eq_int, name = "SolverStatus")]
pub enum PySolverStatus {
    Unsolved = 0,
    Solved,
    PrimalInfeasible,
    DualInfeasible,
    AlmostSolved,
    AlmostPrimalInfeasible,
    AlmostDualInfeasible,
    MaxIterations,
    MaxTime,
    NumericalError,
    InsufficientProgress,
    CallbackTerminated,
}

impl From<&SolverStatus> for PySolverStatus {
    fn from(status: &SolverStatus) -> Self {
        match status {
            SolverStatus::Unsolved => PySolverStatus::Unsolved,
            SolverStatus::Solved => PySolverStatus::Solved,
            SolverStatus::PrimalInfeasible => PySolverStatus::PrimalInfeasible,
            SolverStatus::DualInfeasible => PySolverStatus::DualInfeasible,
            SolverStatus::AlmostSolved => PySolverStatus::AlmostSolved,
            SolverStatus::AlmostPrimalInfeasible => PySolverStatus::AlmostPrimalInfeasible,
            SolverStatus::AlmostDualInfeasible => PySolverStatus::AlmostDualInfeasible,
            SolverStatus::MaxIterations => PySolverStatus::MaxIterations,
            SolverStatus::MaxTime => PySolverStatus::MaxTime,
            SolverStatus::NumericalError => PySolverStatus::NumericalError,
            SolverStatus::InsufficientProgress => PySolverStatus::InsufficientProgress,
            SolverStatus::CallbackTerminated => PySolverStatus::CallbackTerminated,
        }
    }
}

#[pymethods]
impl PySolverStatus {
    pub fn __repr__(&self) -> String {
        match self {
            PySolverStatus::Unsolved => "Unsolved",
            PySolverStatus::Solved => "Solved",
            PySolverStatus::PrimalInfeasible => "PrimalInfeasible",
            PySolverStatus::DualInfeasible => "DualInfeasible",
            PySolverStatus::AlmostSolved => "AlmostSolved",
            PySolverStatus::AlmostPrimalInfeasible => "AlmostPrimalInfeasible",
            PySolverStatus::AlmostDualInfeasible => "AlmostDualInfeasible",
            PySolverStatus::MaxIterations => "MaxIterations",
            PySolverStatus::MaxTime => "MaxTime",
            PySolverStatus::NumericalError => "NumericalError",
            PySolverStatus::InsufficientProgress => "InsufficientProgress",
            PySolverStatus::CallbackTerminated => "CallbackTerminated",
        }
        .to_string()
    }

    // mapping of solver status to CVXPY keys is done via a hash
    pub fn __hash__(&self) -> u32 {
        *self as u32
    }
}

// ----------------------------------
// Solver Settings
// ----------------------------------

impl From<SettingsError> for PyErr {
    fn from(err: SettingsError) -> Self {
        PyException::new_err(err.to_string())
    }
}

// ----------------------------------
// SolverType enum
// ----------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[pyclass(eq, eq_int, name = "SolverType")]
pub enum PySolverType {
    IPM = 0,
}

impl From<&SolverType> for PySolverType {
    fn from(solver_type: &SolverType) -> Self {
        match solver_type {
            SolverType::IPM => PySolverType::IPM,
        }
    }
}

impl From<&PySolverType> for SolverType {
    fn from(solver_type: &PySolverType) -> Self {
        match solver_type {
            PySolverType::IPM => SolverType::IPM,
        }
    }
}

#[pymethods]
impl PySolverType {
    pub fn __repr__(&self) -> String {
        match self {
            PySolverType::IPM => "SolverType.IPM",
        }
        .to_string()
    }
}

// ----------------------------------
// IPMSettings (nested in DefaultSettings)
// ----------------------------------

#[derive(Debug, Clone)]
#[pyclass(name = "IPMSettings")]
pub struct PyIPMSettings {
    #[pyo3(get, set)]
    pub max_step_fraction: f64,

    //full accuracy solution tolerances
    #[pyo3(get, set)]
    pub tol_gap_abs: f64,
    #[pyo3(get, set)]
    pub tol_gap_rel: f64,
    #[pyo3(get, set)]
    pub tol_feas: f64,
    #[pyo3(get, set)]
    pub tol_infeas_abs: f64,
    #[pyo3(get, set)]
    pub tol_infeas_rel: f64,
    #[pyo3(get, set)]
    pub tol_ktratio: f64,

    //reduced accuracy solution tolerances
    #[pyo3(get, set)]
    pub reduced_tol_gap_abs: f64,
    #[pyo3(get, set)]
    pub reduced_tol_gap_rel: f64,
    #[pyo3(get, set)]
    pub reduced_tol_feas: f64,
    #[pyo3(get, set)]
    pub reduced_tol_infeas_abs: f64,
    #[pyo3(get, set)]
    pub reduced_tol_infeas_rel: f64,
    #[pyo3(get, set)]
    pub reduced_tol_ktratio: f64,

    // data equilibration
    #[pyo3(get, set)]
    pub equilibrate_enable: bool,
    #[pyo3(get, set)]
    pub equilibrate_max_iter: u32,
    #[pyo3(get, set)]
    pub equilibrate_min_scaling: f64,
    #[pyo3(get, set)]
    pub equilibrate_max_scaling: f64,

    //step size settings
    #[pyo3(get, set)]
    pub linesearch_backtrack_step: f64,
    #[pyo3(get, set)]
    pub min_switch_step_length: f64,
    #[pyo3(get, set)]
    pub min_terminate_step_length: f64,

    // KKT settings
    #[pyo3(get, set)]
    pub max_threads: u32,
    #[pyo3(get, set)]
    pub direct_kkt_solver: bool,
    #[pyo3(get, set)]
    pub direct_solve_method: String,

    // static regularization parameters
    #[pyo3(get, set)]
    pub static_regularization_enable: bool,
    #[pyo3(get, set)]
    pub static_regularization_constant: f64,
    #[pyo3(get, set)]
    pub static_regularization_proportional: f64,

    // dynamic regularization parameters
    #[pyo3(get, set)]
    pub dynamic_regularization_enable: bool,
    #[pyo3(get, set)]
    pub dynamic_regularization_eps: f64,
    #[pyo3(get, set)]
    pub dynamic_regularization_delta: f64,

    // iterative refinement (for QDLDL)
    #[pyo3(get, set)]
    pub iterative_refinement_enable: bool,
    #[pyo3(get, set)]
    pub iterative_refinement_reltol: f64,
    #[pyo3(get, set)]
    pub iterative_refinement_abstol: f64,
    #[pyo3(get, set)]
    pub iterative_refinement_max_iter: u32,
    #[pyo3(get, set)]
    pub iterative_refinement_stop_ratio: f64,

    // differentiation method
    #[pyo3(get, set)]
    pub diff_method: String,
    #[pyo3(get, set)]
    pub diff_smoothing_mu: f64,
    #[pyo3(get, set)]
    pub diff_smoothing_step_factor: f64,

    // preprocessing
    #[pyo3(get, set)]
    pub presolve_enable: bool,

    #[pyo3(get, set)]
    pub input_sparse_dropzeros: bool,

    //chordal decomposition
    #[pyo3(get, set)]
    pub chordal_decomposition_enable: bool,
    pub chordal_decomposition_merge_method: String,
    pub chordal_decomposition_compact: bool,
    pub chordal_decomposition_complete_dual: bool,

    // pardiso settings (requires pardiso features enabled)
    #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
    #[debug("iparm array [i32; 64]")]
    #[pyo3(get, set)]
    pub pardiso_iparm: [i32; 64],
    #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
    #[pyo3(get, set)]
    pub pardiso_verbose: bool,
}

#[pymethods]
impl PyIPMSettings {
    #[new]
    pub fn new() -> Self {
        (&IPMSettings::<f64>::default()).into()
    }

    #[staticmethod]
    #[pyo3(name = "default")]
    pub fn py_default() -> Self {
        PyIPMSettings::new()
    }

    pub fn __repr__(&self) -> String {
        let mut s = String::new();
        write!(s, "{:#?}", self).unwrap();
        s.replacen("PyIPMSettings", "IPMSettings", 1)
    }
}

impl Default for PyIPMSettings {
    fn default() -> Self {
        PyIPMSettings::new()
    }
}

impl From<&IPMSettings<f64>> for PyIPMSettings {
    fn from(set: &IPMSettings<f64>) -> Self {
        PyIPMSettings {
            max_step_fraction: set.max_step_fraction,
            tol_gap_abs: set.tol_gap_abs,
            tol_gap_rel: set.tol_gap_rel,
            tol_feas: set.tol_feas,
            tol_infeas_abs: set.tol_infeas_abs,
            tol_infeas_rel: set.tol_infeas_rel,
            tol_ktratio: set.tol_ktratio,
            reduced_tol_gap_abs: set.reduced_tol_gap_abs,
            reduced_tol_gap_rel: set.reduced_tol_gap_rel,
            reduced_tol_feas: set.reduced_tol_feas,
            reduced_tol_infeas_abs: set.reduced_tol_infeas_abs,
            reduced_tol_infeas_rel: set.reduced_tol_infeas_rel,
            reduced_tol_ktratio: set.reduced_tol_ktratio,
            equilibrate_enable: set.equilibrate_enable,
            equilibrate_max_iter: set.equilibrate_max_iter,
            equilibrate_min_scaling: set.equilibrate_min_scaling,
            equilibrate_max_scaling: set.equilibrate_max_scaling,
            linesearch_backtrack_step: set.linesearch_backtrack_step,
            min_switch_step_length: set.min_switch_step_length,
            min_terminate_step_length: set.min_terminate_step_length,
            max_threads: set.max_threads,
            direct_kkt_solver: set.direct_kkt_solver,
            direct_solve_method: set.direct_solve_method.clone(),
            static_regularization_enable: set.static_regularization_enable,
            static_regularization_constant: set.static_regularization_constant,
            static_regularization_proportional: set.static_regularization_proportional,
            dynamic_regularization_enable: set.dynamic_regularization_enable,
            dynamic_regularization_eps: set.dynamic_regularization_eps,
            dynamic_regularization_delta: set.dynamic_regularization_delta,
            iterative_refinement_enable: set.iterative_refinement_enable,
            iterative_refinement_reltol: set.iterative_refinement_reltol,
            iterative_refinement_abstol: set.iterative_refinement_abstol,
            iterative_refinement_max_iter: set.iterative_refinement_max_iter,
            iterative_refinement_stop_ratio: set.iterative_refinement_stop_ratio,
            diff_method: match set.diff_method {
                DiffMethod::Auto => "auto".to_string(),
                DiffMethod::Smoothed => "smoothed".to_string(),
                DiffMethod::Exact => "exact".to_string(),
            },
            diff_smoothing_mu: set.diff_smoothing_mu,
            diff_smoothing_step_factor: set.diff_smoothing_step_factor,
            presolve_enable: set.presolve_enable,
            input_sparse_dropzeros: set.input_sparse_dropzeros,
            #[cfg(feature = "sdp")]
            chordal_decomposition_enable: set.chordal_decomposition_enable,
            #[cfg(feature = "sdp")]
            chordal_decomposition_merge_method: set.chordal_decomposition_merge_method.clone(),
            #[cfg(feature = "sdp")]
            chordal_decomposition_compact: set.chordal_decomposition_compact,
            #[cfg(feature = "sdp")]
            chordal_decomposition_complete_dual: set.chordal_decomposition_complete_dual,
            #[cfg(not(feature = "sdp"))]
            chordal_decomposition_enable: false,
            #[cfg(not(feature = "sdp"))]
            chordal_decomposition_merge_method: String::new(),
            #[cfg(not(feature = "sdp"))]
            chordal_decomposition_compact: false,
            #[cfg(not(feature = "sdp"))]
            chordal_decomposition_complete_dual: false,
            #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
            pardiso_iparm: set.pardiso_iparm,
            #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
            pardiso_verbose: set.pardiso_verbose,
        }
    }
}

impl PyIPMSettings {
    pub(crate) fn to_internal(&self) -> IPMSettings<f64> {
        IPMSettings {
            max_step_fraction: self.max_step_fraction,
            tol_gap_abs: self.tol_gap_abs,
            tol_gap_rel: self.tol_gap_rel,
            tol_feas: self.tol_feas,
            tol_infeas_abs: self.tol_infeas_abs,
            tol_infeas_rel: self.tol_infeas_rel,
            tol_ktratio: self.tol_ktratio,
            reduced_tol_gap_abs: self.reduced_tol_gap_abs,
            reduced_tol_gap_rel: self.reduced_tol_gap_rel,
            reduced_tol_feas: self.reduced_tol_feas,
            reduced_tol_infeas_abs: self.reduced_tol_infeas_abs,
            reduced_tol_infeas_rel: self.reduced_tol_infeas_rel,
            reduced_tol_ktratio: self.reduced_tol_ktratio,
            equilibrate_enable: self.equilibrate_enable,
            equilibrate_max_iter: self.equilibrate_max_iter,
            equilibrate_min_scaling: self.equilibrate_min_scaling,
            equilibrate_max_scaling: self.equilibrate_max_scaling,
            linesearch_backtrack_step: self.linesearch_backtrack_step,
            min_switch_step_length: self.min_switch_step_length,
            min_terminate_step_length: self.min_terminate_step_length,
            max_threads: self.max_threads,
            direct_kkt_solver: self.direct_kkt_solver,
            direct_solve_method: self.direct_solve_method.clone(),
            static_regularization_enable: self.static_regularization_enable,
            static_regularization_constant: self.static_regularization_constant,
            static_regularization_proportional: self.static_regularization_proportional,
            dynamic_regularization_enable: self.dynamic_regularization_enable,
            dynamic_regularization_eps: self.dynamic_regularization_eps,
            dynamic_regularization_delta: self.dynamic_regularization_delta,
            iterative_refinement_enable: self.iterative_refinement_enable,
            iterative_refinement_reltol: self.iterative_refinement_reltol,
            iterative_refinement_abstol: self.iterative_refinement_abstol,
            iterative_refinement_max_iter: self.iterative_refinement_max_iter,
            iterative_refinement_stop_ratio: self.iterative_refinement_stop_ratio,
            diff_method: match self.diff_method.as_str() {
                "smoothed" => DiffMethod::Smoothed,
                "exact" => DiffMethod::Exact,
                _ => DiffMethod::Auto,
            },
            diff_smoothing_mu: self.diff_smoothing_mu,
            diff_smoothing_step_factor: self.diff_smoothing_step_factor,
            presolve_enable: self.presolve_enable,
            input_sparse_dropzeros: self.input_sparse_dropzeros,
            #[cfg(feature = "sdp")]
            chordal_decomposition_enable: self.chordal_decomposition_enable,
            #[cfg(feature = "sdp")]
            chordal_decomposition_merge_method: self.chordal_decomposition_merge_method.clone(),
            #[cfg(feature = "sdp")]
            chordal_decomposition_compact: self.chordal_decomposition_compact,
            #[cfg(feature = "sdp")]
            chordal_decomposition_complete_dual: self.chordal_decomposition_complete_dual,
            #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
            pardiso_iparm: self.pardiso_iparm,
            #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
            pardiso_verbose: self.pardiso_verbose,
        }
    }
}

// ----------------------------------
// DefaultSettings (with nested ipm)
// ----------------------------------

#[derive(Debug, Clone)]
#[pyclass(name = "DefaultSettings")]
pub struct PyDefaultSettings {
    #[pyo3(get, set)]
    pub solver: PySolverType,
    #[pyo3(get, set)]
    pub max_iter: u32,
    #[pyo3(get, set)]
    pub time_limit: f64,
    #[pyo3(get, set)]
    pub verbose: bool,
    #[pyo3(get, set)]
    pub yolo: bool,
    #[pyo3(get, set)]
    pub yolo_num_iters: u32,
    #[pyo3(get, set)]
    pub ipm: PyIPMSettings,
}

#[pymethods]
impl PyDefaultSettings {
    #[new]
    pub fn new() -> Self {
        (&DefaultSettings::<f64>::default()).into()
    }

    #[staticmethod]
    #[pyo3(name = "default")]
    pub fn py_default() -> Self {
        PyDefaultSettings::default()
    }

    pub fn __repr__(&self) -> String {
        let mut s = String::new();
        write!(s, "{:#?}", self).unwrap();
        s.replacen("PyDefaultSettings", "DefaultSettings", 1)
    }
}

//Default not really necessary, but keeps clippy happy....
impl Default for PyDefaultSettings {
    fn default() -> Self {
        PyDefaultSettings::new()
    }
}

impl From<&DefaultSettings<f64>> for PyDefaultSettings {
    fn from(set: &DefaultSettings<f64>) -> Self {
        PyDefaultSettings {
            solver: (&set.solver).into(),
            max_iter: set.max_iter,
            time_limit: set.time_limit,
            verbose: set.verbose,
            yolo: set.yolo,
            yolo_num_iters: set.yolo_num_iters,
            ipm: (&set.ipm).into(),
        }
    }
}

impl PyDefaultSettings {
    pub(crate) fn to_internal(&self) -> Result<DefaultSettings<f64>, PyErr> {
        let settings = DefaultSettings::<f64> {
            solver: (&self.solver).into(),
            max_iter: self.max_iter,
            time_limit: self.time_limit,
            verbose: self.verbose,
            enable_grad: false,
            yolo: self.yolo,
            yolo_num_iters: self.yolo_num_iters,
            ipm: self.ipm.to_internal(),
        };
        Ok(settings)
    }
}

// ----------------------------------
// Solver
// ----------------------------------

#[pyclass(name = "DefaultSolver")]
pub struct PyDefaultSolver {
    inner: DefaultSolver<f64>,
}

#[pymethods]
impl PyDefaultSolver {
    /// Create a new solver.
    ///
    /// # Arguments
    /// * `P` - Quadratic objective matrix (CSR format from scipy.sparse.csr_matrix)
    /// * `q` - Linear objective vector
    /// * `A` - Constraint matrix (CSR format from scipy.sparse.csr_matrix)
    /// * `b` - Constraint RHS vector
    /// * `cones` - Cone constraints specification
    /// * `settings` - Solver settings
    #[new]
    fn new(
        P: PyCsrMatrix,
        q: Vec<f64>,
        A: PyCsrMatrix,
        b: Vec<f64>,
        cones: Vec<PySupportedCone>,
        settings: PyDefaultSettings,
    ) -> PyResult<Self> {
        let cones = _py_to_native_cones(cones);
        let settings = settings.to_internal()?;

        // Convert CSR to CSC for internal solver
        let P_csr: CsrMatrix<f64> = P.into();
        let A_csr: CsrMatrix<f64> = A.into();
        let P_csc = P_csr.to_csc();
        let A_csc = A_csr.to_csc();

        // Handle the Result returned by DefaultSolver::new
        let solver = DefaultSolver::new(&P_csc, &q, &A_csc, &b, &cones, settings)?;

        Ok(Self { inner: solver })
    }

    /// Create a new solver with direct-x cones. Equivalent to `new` but
    /// additionally accepts `x_cones` — a list of direct-x cone specs
    /// (e.g. `NonnegativeXConeT(indices)` or `SecondOrderXConeT(indices)`)
    /// constraining sub-vectors of the primal `x` to cones directly.
    #[staticmethod]
    fn new_with_xcones(
        P: PyCsrMatrix,
        q: Vec<f64>,
        A: PyCsrMatrix,
        b: Vec<f64>,
        cones: Vec<PySupportedCone>,
        x_cones: Vec<super::cones_py::PySupportedXCone>,
        settings: PyDefaultSettings,
    ) -> PyResult<Self> {
        let cones = _py_to_native_cones(cones);
        let x_cones = super::cones_py::_py_to_native_x_cones(x_cones);
        let settings = settings.to_internal()?;

        let P_csr: CsrMatrix<f64> = P.into();
        let A_csr: CsrMatrix<f64> = A.into();
        let P_csc = P_csr.to_csc();
        let A_csc = A_csr.to_csc();

        let solver =
            DefaultSolver::new_with_xcones(&P_csc, &q, &A_csc, &b, &cones, &x_cones, settings)?;

        Ok(Self { inner: solver })
    }

    fn solve(&mut self, py: Python<'_>) -> PyResult<PyDefaultSolution> {
        py.allow_threads(|| self.inner.solve());
        Ok(self.get_solution())
    }

    pub fn __repr__(&self) -> String {
        "Moreau solver with Float precision: f64".to_string()
    }

    fn print_configuration(&mut self) {
        // force a print of the configuration regardless
        // of the verbosity settings.   Save them here first.
        let verbose = self.inner.settings.core().verbose;

        self.inner.settings.core_mut().verbose = true;
        self.inner
            .info
            .print_configuration(&self.inner.settings, &self.inner.data, &self.inner.cones)
            .unwrap();

        // revert back to user option
        self.inner.settings.core_mut().verbose = verbose;
    }

    fn print_timers(&self) {
        match &self.inner.timers {
            Some(timers) => timers.print(),
            None => println!("no timers enabled"),
        };
    }

    // return the currently configured settings of a solver.  If settings
    // are to be overridden, modify this object then pass back using kwargs
    // update(settings=settings)
    fn get_settings(&self) -> PyDefaultSettings {
        (&self.inner.settings).into()
    }

    fn get_info(&self) -> PyDefaultInfo {
        (&self.inner.info).into()
    }

    fn get_solution(&self) -> PyDefaultSolution {
        (&self.inner.solution).into()
    }

    /// Compute gradients via implicit differentiation.
    ///
    /// # Arguments
    /// * `dx` - Upstream gradient w.r.t. primal solution x
    /// * `ds` - Upstream gradient w.r.t. slack variables s
    /// * `dz` - Upstream gradient w.r.t. dual variables z
    ///
    /// # Returns
    /// BackwardResult containing gradients (dP_values, dq, dA_values, db)
    #[pyo3(signature = (dx, ds, dz, dz_x=None))]
    fn backward_batch(
        &mut self,
        dx: Vec<f64>,
        ds: Vec<f64>,
        dz: Vec<f64>,
        dz_x: Option<Vec<f64>>,
    ) -> PyResult<PyBackwardResult> {
        let diff_method = self.inner.settings.ipm.diff_method;
        let dz_x_slice: &[f64] = dz_x.as_deref().unwrap_or(&[]);
        let result =
            self.inner
                .backward_batch_with_dz_x(&dx, &ds, &dz, dz_x_slice, None, diff_method)?;
        Ok(result.into())
    }
}
