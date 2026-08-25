use crate::algebra::*;
use crate::solver::core::ffi::*;
use crate::solver::DefaultSettings;

/// FFI interface for [`DefaultSettings`](crate::solver::implementations::default::DefaultSettings)
///
/// This is a flat structure for C FFI compatibility. The Rust DefaultSettings
/// has a nested structure (common settings + ipm: IPMSettings), but the FFI
/// flattens this for easier C interop.
#[allow(missing_docs)]
#[derive(Debug, Clone)]
#[repr(C)]
pub struct DefaultSettingsFFI<T: FloatT> {
    // Common settings (top-level in DefaultSettings)
    pub solver: SolverTypeFFI,
    pub max_iter: u32,
    pub time_limit: f64,
    pub verbose: bool,
    pub enable_grad: bool,

    // IPM-specific settings (from DefaultSettings.ipm)
    pub max_step_fraction: T,

    // Full accuracy settings
    pub tol_gap_abs: T,
    pub tol_gap_rel: T,
    pub tol_feas: T,
    pub tol_infeas_abs: T,
    pub tol_infeas_rel: T,
    pub tol_ktratio: T,

    // Reduced accuracy settings
    pub reduced_tol_gap_abs: T,
    pub reduced_tol_gap_rel: T,
    pub reduced_tol_feas: T,
    pub reduced_tol_infeas_abs: T,
    pub reduced_tol_infeas_rel: T,
    pub reduced_tol_ktratio: T,

    // data equilibration settings
    pub equilibrate_enable: bool,
    pub equilibrate_max_iter: u32,
    pub equilibrate_min_scaling: T,
    pub equilibrate_max_scaling: T,

    // Step size settings
    pub linesearch_backtrack_step: T,
    pub min_switch_step_length: T,
    pub min_terminate_step_length: T,

    // Linear solver settings
    pub max_threads: u32,
    pub direct_kkt_solver: bool,
    pub direct_solve_method: DirectSolveMethodsFFI,

    // static regularization parameters
    pub static_regularization_enable: bool,
    pub static_regularization_constant: T,
    pub static_regularization_proportional: T,

    // dynamic regularization parameters
    pub dynamic_regularization_enable: bool,
    pub dynamic_regularization_eps: T,
    pub dynamic_regularization_delta: T,

    // iterative refinement (for direct solves)
    pub iterative_refinement_enable: bool,
    pub iterative_refinement_reltol: T,
    pub iterative_refinement_abstol: T,
    pub iterative_refinement_max_iter: u32,
    pub iterative_refinement_stop_ratio: T,

    // differentiation method
    pub diff_method: DiffMethodFFI,
    pub diff_smoothing_mu: T,
    pub diff_smoothing_step_factor: T,

    // preprocessing
    pub presolve_enable: bool,
    pub input_sparse_dropzeros: bool,

    // chordal decomposition
    #[cfg(feature = "sdp")]
    pub chordal_decomposition_enable: bool,
    #[cfg(feature = "sdp")]
    pub chordal_decomposition_merge_method: CliqueMergeMethodsFFI,
    #[cfg(feature = "sdp")]
    pub chordal_decomposition_compact: bool,
    #[cfg(feature = "sdp")]
    pub chordal_decomposition_complete_dual: bool,

    //pardiso settings
    #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
    pub pardiso_iparm: [i32; 64],
    #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
    pub pardiso_verbose: bool,
}

// DefaultSettingsFFI -> DefaultSettings
impl<T> From<DefaultSettingsFFI<T>> for DefaultSettings<T>
where
    T: FloatT,
{
    fn from(ffi: DefaultSettingsFFI<T>) -> Self {
        use crate::solver::implementations::default::IPMSettings;

        let ipm = IPMSettings {
            max_step_fraction: ffi.max_step_fraction,
            tol_gap_abs: ffi.tol_gap_abs,
            tol_gap_rel: ffi.tol_gap_rel,
            tol_feas: ffi.tol_feas,
            tol_infeas_abs: ffi.tol_infeas_abs,
            tol_infeas_rel: ffi.tol_infeas_rel,
            tol_ktratio: ffi.tol_ktratio,
            reduced_tol_gap_abs: ffi.reduced_tol_gap_abs,
            reduced_tol_gap_rel: ffi.reduced_tol_gap_rel,
            reduced_tol_feas: ffi.reduced_tol_feas,
            reduced_tol_infeas_abs: ffi.reduced_tol_infeas_abs,
            reduced_tol_infeas_rel: ffi.reduced_tol_infeas_rel,
            reduced_tol_ktratio: ffi.reduced_tol_ktratio,
            equilibrate_enable: ffi.equilibrate_enable,
            equilibrate_max_iter: ffi.equilibrate_max_iter,
            equilibrate_min_scaling: ffi.equilibrate_min_scaling,
            equilibrate_max_scaling: ffi.equilibrate_max_scaling,
            linesearch_backtrack_step: ffi.linesearch_backtrack_step,
            min_switch_step_length: ffi.min_switch_step_length,
            min_terminate_step_length: ffi.min_terminate_step_length,
            max_threads: ffi.max_threads,
            direct_kkt_solver: ffi.direct_kkt_solver,
            direct_solve_method: ffi.direct_solve_method.into(),
            static_regularization_enable: ffi.static_regularization_enable,
            static_regularization_constant: ffi.static_regularization_constant,
            static_regularization_proportional: ffi.static_regularization_proportional,
            dynamic_regularization_enable: ffi.dynamic_regularization_enable,
            dynamic_regularization_eps: ffi.dynamic_regularization_eps,
            dynamic_regularization_delta: ffi.dynamic_regularization_delta,
            iterative_refinement_enable: ffi.iterative_refinement_enable,
            iterative_refinement_reltol: ffi.iterative_refinement_reltol,
            iterative_refinement_abstol: ffi.iterative_refinement_abstol,
            iterative_refinement_max_iter: ffi.iterative_refinement_max_iter,
            iterative_refinement_stop_ratio: ffi.iterative_refinement_stop_ratio,
            diff_method: ffi.diff_method.into(),
            diff_smoothing_mu: ffi.diff_smoothing_mu,
            diff_smoothing_step_factor: ffi.diff_smoothing_step_factor,
            presolve_enable: ffi.presolve_enable,
            input_sparse_dropzeros: ffi.input_sparse_dropzeros,
            #[cfg(feature = "sdp")]
            chordal_decomposition_enable: ffi.chordal_decomposition_enable,
            #[cfg(feature = "sdp")]
            chordal_decomposition_merge_method: ffi.chordal_decomposition_merge_method.into(),
            #[cfg(feature = "sdp")]
            chordal_decomposition_compact: ffi.chordal_decomposition_compact,
            #[cfg(feature = "sdp")]
            chordal_decomposition_complete_dual: ffi.chordal_decomposition_complete_dual,
            #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
            pardiso_iparm: ffi.pardiso_iparm,
            #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
            pardiso_verbose: ffi.pardiso_verbose,
        };

        DefaultSettings {
            solver: ffi.solver.into(),
            max_iter: ffi.max_iter,
            time_limit: ffi.time_limit,
            verbose: ffi.verbose,
            enable_grad: ffi.enable_grad,
            // YOLO mode is not exposed via the C API (FFI). C API callers
            // should use the Python or Rust API for YOLO mode.
            yolo: false,
            yolo_num_iters: 15,
            ipm,
        }
    }
}

// DefaultSettings -> DefaultSettingsFFI
impl<T> From<DefaultSettings<T>> for DefaultSettingsFFI<T>
where
    T: FloatT,
{
    fn from(settings: DefaultSettings<T>) -> Self {
        Self {
            solver: settings.solver.into(),
            max_iter: settings.max_iter,
            time_limit: settings.time_limit,
            verbose: settings.verbose,
            enable_grad: settings.enable_grad,
            max_step_fraction: settings.ipm.max_step_fraction,
            tol_gap_abs: settings.ipm.tol_gap_abs,
            tol_gap_rel: settings.ipm.tol_gap_rel,
            tol_feas: settings.ipm.tol_feas,
            tol_infeas_abs: settings.ipm.tol_infeas_abs,
            tol_infeas_rel: settings.ipm.tol_infeas_rel,
            tol_ktratio: settings.ipm.tol_ktratio,
            reduced_tol_gap_abs: settings.ipm.reduced_tol_gap_abs,
            reduced_tol_gap_rel: settings.ipm.reduced_tol_gap_rel,
            reduced_tol_feas: settings.ipm.reduced_tol_feas,
            reduced_tol_infeas_abs: settings.ipm.reduced_tol_infeas_abs,
            reduced_tol_infeas_rel: settings.ipm.reduced_tol_infeas_rel,
            reduced_tol_ktratio: settings.ipm.reduced_tol_ktratio,
            equilibrate_enable: settings.ipm.equilibrate_enable,
            equilibrate_max_iter: settings.ipm.equilibrate_max_iter,
            equilibrate_min_scaling: settings.ipm.equilibrate_min_scaling,
            equilibrate_max_scaling: settings.ipm.equilibrate_max_scaling,
            linesearch_backtrack_step: settings.ipm.linesearch_backtrack_step,
            min_switch_step_length: settings.ipm.min_switch_step_length,
            min_terminate_step_length: settings.ipm.min_terminate_step_length,
            max_threads: settings.ipm.max_threads,
            direct_kkt_solver: settings.ipm.direct_kkt_solver,
            direct_solve_method: settings.ipm.direct_solve_method.into(),
            static_regularization_enable: settings.ipm.static_regularization_enable,
            static_regularization_constant: settings.ipm.static_regularization_constant,
            static_regularization_proportional: settings.ipm.static_regularization_proportional,
            dynamic_regularization_enable: settings.ipm.dynamic_regularization_enable,
            dynamic_regularization_eps: settings.ipm.dynamic_regularization_eps,
            dynamic_regularization_delta: settings.ipm.dynamic_regularization_delta,
            iterative_refinement_enable: settings.ipm.iterative_refinement_enable,
            iterative_refinement_reltol: settings.ipm.iterative_refinement_reltol,
            iterative_refinement_abstol: settings.ipm.iterative_refinement_abstol,
            iterative_refinement_max_iter: settings.ipm.iterative_refinement_max_iter,
            iterative_refinement_stop_ratio: settings.ipm.iterative_refinement_stop_ratio,
            diff_method: settings.ipm.diff_method.into(),
            diff_smoothing_mu: settings.ipm.diff_smoothing_mu,
            diff_smoothing_step_factor: settings.ipm.diff_smoothing_step_factor,
            presolve_enable: settings.ipm.presolve_enable,
            input_sparse_dropzeros: settings.ipm.input_sparse_dropzeros,
            #[cfg(feature = "sdp")]
            chordal_decomposition_enable: settings.ipm.chordal_decomposition_enable,
            #[cfg(feature = "sdp")]
            chordal_decomposition_merge_method: settings
                .ipm
                .chordal_decomposition_merge_method
                .into(),
            #[cfg(feature = "sdp")]
            chordal_decomposition_compact: settings.ipm.chordal_decomposition_compact,
            #[cfg(feature = "sdp")]
            chordal_decomposition_complete_dual: settings.ipm.chordal_decomposition_complete_dual,
            #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
            pardiso_iparm: settings.ipm.pardiso_iparm,
            #[cfg(any(feature = "pardiso-mkl", feature = "pardiso-panua"))]
            pardiso_verbose: settings.ipm.pardiso_verbose,
        }
    }
}

#[test]
fn test_settings_ffi() {
    use super::*;

    let settings = DefaultSettings::<f64> {
        max_iter: 123,
        ..Default::default()
    };
    let settings_ffi: DefaultSettingsFFI<f64> = settings.clone().into();

    assert_eq!(settings.max_iter, settings_ffi.max_iter);

    // Test round-trip
    let settings_back: DefaultSettings<f64> = settings_ffi.into();
    assert_eq!(settings.max_iter, settings_back.max_iter);
    assert_eq!(settings.ipm.tol_gap_abs, settings_back.ipm.tol_gap_abs);
}
