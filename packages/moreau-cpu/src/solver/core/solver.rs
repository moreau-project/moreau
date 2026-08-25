use self::internal::*;
use super::callbacks::{Callback, CallbackFcnFFI, MoreauCallbackFn};
use super::cones::Cone;
use super::{traits::*, SettingsError};
use crate::algebra::*;
use crate::solver::core::callbacks::SolverCallbacks;
use crate::solver::core::ffi::*;
use crate::timers::*;
use crate::utils::banner;
use crate::utils::debug::{debug_block, debug_println};
use std::io::Write;

// ---------------------------------
// Solver status type
// ---------------------------------

/// Status of solver at termination
#[repr(C)]
#[derive(PartialEq, Eq, Clone, Debug, Copy, Default)]
pub enum SolverStatus {
    /// Problem is not solved (solver hasn't run).
    #[default]
    Unsolved = 0,
    /// Solver terminated with a solution.
    Solved,
    /// Problem is primal infeasible.  Solution returned is a certificate of primal infeasibility.
    PrimalInfeasible,
    /// Problem is dual infeasible.  Solution returned is a certificate of dual infeasibility.
    DualInfeasible,
    /// Solver terminated with a solution (reduced accuracy)
    AlmostSolved,
    /// Problem is primal infeasible.  Solution returned is a certificate of primal infeasibility (reduced accuracy).
    AlmostPrimalInfeasible,
    /// Problem is dual infeasible.  Solution returned is a certificate of dual infeasibility (reduced accuracy).
    AlmostDualInfeasible,
    /// Iteration limit reached before solution or infeasibility certificate found.
    MaxIterations,
    /// Time limit reached before solution or infeasibility certificate found.
    MaxTime,
    /// Solver terminated with a numerical error
    NumericalError,
    /// Solver terminated due to lack of progress.
    InsufficientProgress,
    /// Solver terminated by user callback
    CallbackTerminated,
}

impl SolverStatus {
    pub(crate) fn is_infeasible(&self) -> bool {
        matches!(
            *self,
            |SolverStatus::PrimalInfeasible| SolverStatus::DualInfeasible
                | SolverStatus::AlmostPrimalInfeasible
                | SolverStatus::AlmostDualInfeasible
        )
    }

    pub(crate) fn is_errored(&self) -> bool {
        // status is any of the error codes
        matches!(
            *self,
            SolverStatus::NumericalError | SolverStatus::InsufficientProgress
        )
    }
}

#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum StepDirection {
    Affine,
    Combined,
    /// Centering step without HSDE τ/κ coupling.
    /// Uses the combined RHS but skips the Δτ correction
    /// to (Δx, Δz), keeping the step in the original-space
    /// central path direction.
    Centering,
}

/// Scaling strategy used by the solver when
/// linearizing centrality conditions.
#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum ScalingStrategy {
    PrimalDual,
    Dual,
}

/// An enum for reporting strategy checkpointing
#[repr(u32)]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
enum StrategyCheckpoint {
    Update(ScalingStrategy), // Checkpoint is suggesting a new ScalingStrategy
    NoUpdate,                // Checkpoint recommends no change to ScalingStrategy
    Fail,                    // Checkpoint found a problem but no more ScalingStrategies to try
}

impl std::fmt::Display for SolverStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

// ---------------------------------
// top level solver container type
// ---------------------------------

// The top-level solver.

// This trait is defined with a collection of mutually interacting associated types.
// See the [`DefaultSolver`](crate::solver::implementations::default) for an example.

pub struct Solver<T, D, V, R, K, C, I, SO, SE>
where
    I: MoreauFFI<I>,
    SE: Settings<T>,
    T: FloatT,
{
    pub data: D,
    pub variables: V,
    pub residuals: R,
    pub kktsystem: K,
    pub cones: C,
    pub step_lhs: V,
    pub step_rhs: V,
    pub prev_vars: V,
    /// Snapshot of the most recent iterate that met reduced tolerances.
    /// Used as a fallback on InsufficientProgress / NumericalError / MaxIter /
    /// MaxTime terminations so the solver can report AlmostSolved instead.
    pub best_vars: V,
    pub info: I,
    pub solution: SO,
    pub(crate) settings: SE, // not public to avoid unchecked modifications
    pub timers: Option<Timers>,
    pub(crate) callbacks: SolverCallbacks<I, I::FFI>,
    pub(crate) phantom: std::marker::PhantomData<T>,
    /// Skip default_start() initialization when true (reserved for internal use)
    pub(crate) skip_default_start: bool,
    /// Cached central-path iterate for smoothed differentiation.
    /// Pre-allocated at construction; populated during the IPM loop
    /// when μ crosses `diff_smoothing_mu`.
    pub(crate) smoothing_vars: V,
    /// Whether `smoothing_vars` has been populated in the current solve.
    pub(crate) smoothing_cached: bool,
}

fn _print_setup_info(
    out: &mut dyn Write,
    is_verbose: bool,
    nnz_kkt: usize,
    nnz_factor: usize,
    setup_time: std::time::Duration,
) -> std::io::Result<()> {
    if !is_verbose {
        return std::io::Result::Ok(());
    }

    let fill_ratio = if nnz_kkt > 0 {
        nnz_factor as f64 / nnz_kkt as f64
    } else {
        1.0
    };

    writeln!(out)?;
    writeln!(out, "  \x1b[38;5;243m▸\x1b[0m \x1b[38;5;250msetup:\x1b[0m nnz(KKT)={} nnz(L)={} fill={:.1}x \x1b[38;5;243m•\x1b[0m {:?}",
        nnz_kkt, nnz_factor, fill_ratio, setup_time)?;
    std::io::Result::Ok(())
}

impl<T, D, V, R, K, C, I, SO, SE> Solver<T, D, V, R, K, C, I, SO, SE>
where
    I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
    SE: Settings<T>,
    C: Cone<T>,
    T: FloatT,
{
    /// Create a new solver object
    pub fn set_termination_callback(&mut self, callback: impl MoreauCallbackFn<I> + 'static) {
        self.callbacks.termination_callback = Callback::Rust(Box::new(callback));
    }

    pub fn set_termination_callback_c(
        &mut self,
        callback: CallbackFcnFFI<I::FFI>,
        data_ptr: *mut std::ffi::c_void,
    ) {
        self.callbacks.termination_callback = Callback::new_c(callback, data_ptr);
    }

    pub fn unset_termination_callback(&mut self) {
        self.callbacks.termination_callback = Callback::None;
    }

    pub fn settings(&self) -> &SE {
        &self.settings
    }

    pub fn update_settings(&mut self, settings: SE) -> Result<(), SettingsError> {
        settings.validate_as_update(&self.settings)?;
        settings.validate_with_cones(&self.cones)?;
        self.settings = settings;
        Ok(())
    }
}

// ---------------------------------
// IPSolver trait and its standard implementation.
// ---------------------------------

/// An interior point solver implementing a predictor-corrector scheme
//
// Only the main solver function lives in IPSolver, since this is the
// only publicly facing trait we want to give the solver.   Additional
// internal functionality for the top level solver object is implemented
// for the IPSolverUtilities trait below, upon which IPSolver depends
pub trait IPSolver<T, D, V, R, K, C, I, SO, SE> {
    /// Run the solver
    fn solve(&mut self);
}

impl<T, D, V, R, K, C, I, SO, SE> IPSolver<T, D, V, R, K, C, I, SO, SE>
    for Solver<T, D, V, R, K, C, I, SO, SE>
where
    T: FloatT,
    D: ProblemData<T, V = V>,
    V: Variables<T, D = D, R = R, C = C, SE = SE>,
    R: Residuals<T, D = D, V = V>,
    K: KKTSystem<T, D = D, V = V, C = C, SE = SE>,
    C: Cone<T>,
    I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
    SO: Solution<T, D = D, V = V, I = I, SE = SE>,
    SE: Settings<T>,
{
    fn solve(&mut self) {
        // various initializations
        let mut iter: u32 = 0;
        let mut σ = T::one();
        let mut α = T::zero();
        let mut μ;

        //timers is stored as an option so that
        //we can swap it out here and avoid
        //borrow conflicts with other fields.
        let mut timers = self.timers.take().unwrap();

        // solver release info, solver config
        // problem dimensions, cone types etc
        notimeit! {timers; {
            banner::print_banner(
                self.info.print_target(),
                self.settings.core().verbose,
                "CPU Conic Solver",
            )
            .unwrap();
            self.info.print_configuration(&self.settings, &self.data, &self.cones).unwrap();
            self.info.print_status_header(&self.settings).unwrap();
        }}

        self.info.reset(&mut timers);
        self.smoothing_cached = false;

        timeit! {timers => "solve"; {

        // initialize variables to some reasonable starting point
        if !self.skip_default_start {
            timeit!{timers => "default start"; {
                self.default_start();
            }}
        }
        self.skip_default_start = false; // reset for next solve

        // YOLO mode: seed prev_vars from initial variables so that a first-iteration
        // NaN restores the actual starting point instead of synthetic zeros.
        if self.settings.core().yolo {
            self.prev_vars.copy_from(&self.variables);
        }

        timeit!{timers => "IP iteration"; {

        // ----------
        // main loop
        // ----------

        let mut scaling = {
            if self.cones.allows_primal_dual_scaling() {ScalingStrategy::PrimalDual}
            else {ScalingStrategy::Dual}
        };

        loop {

            //update the residuals
            //--------------
            self.residuals.update(&self.variables, &self.data);

            //calculate duality gap (scaled)
            //--------------
            μ = self
                .variables
                .calc_mu(&self.residuals, &self.cones, self.kktsystem.x_cones_ref());

            debug_println!("μ: {:?}", μ);

            let yolo_mode = self.settings.core().yolo;

            if !yolo_mode {
                // record scalar values from most recent iteration.
                // This captures μ at iteration zero.
                self.info.save_scalars(μ, α, σ, iter);

                // convergence check and printing
                // --------------
                self.info.update(
                    &mut self.data,
                    &self.variables,
                    &self.residuals,&timers);

                // Snapshot this iterate if it meets reduced tolerances. If the
                // solve later diverges (e.g. bad step from near-convergent KKT
                // noise) we can fall back to this state instead of reporting
                // MaxIterations / InsufficientProgress with a degraded iterate.
                self.info.try_save_best_iterate(
                    &self.variables,
                    &mut self.best_vars,
                    &self.settings,
                );

                notimeit!{timers; {
                    self.info.print_status(&self.settings).unwrap();
                }}

                // termination checks
                // --------------

                // user defined termination checks
                if self.callbacks.check_termination(&self.info) {
                    self.info.set_status(SolverStatus::CallbackTerminated);
                    break;
                }
                // internal termination checks
                let is_done = self.info.check_termination(&self.residuals, &self.settings, iter);

                // check for termination due to slow progress and update strategy
                if is_done{
                        match self.strategy_checkpoint_insufficient_progress(scaling){
                            StrategyCheckpoint::NoUpdate | StrategyCheckpoint::Fail => {break}
                            StrategyCheckpoint::Update(s) => {scaling = s; continue}
                        }
                }  // allows continuation if new strategy provided
            } else {
                // YOLO mode: skip convergence, break at fixed iteration count
                if iter >= self.settings.core().yolo_num_iters {
                    self.info.set_status(SolverStatus::MaxIterations);
                    break;
                }
            }

            // update the scalings
            // --------------
            let is_scaling_success;
            timeit!{timers => "scale cones"; {
                is_scaling_success = self.variables.scale_cones(
                    &mut self.cones,
                    self.kktsystem.x_cones_mut(),
                    μ,
                    scaling,
                );
            }}
            if !yolo_mode {
                // check whether variables are interior points
                // Note: YOLO mode skips this check. A scaling failure that produces
                // finite (non-NaN) garbage will not be caught — the NaN snapshot only
                // guards against NaN, not arbitrary numerical corruption. This is an
                // accepted trade-off for zero-overhead iteration in YOLO mode.
                match self.strategy_checkpoint_is_scaling_success(is_scaling_success,scaling){
                    StrategyCheckpoint::Fail => {break}
                    StrategyCheckpoint::NoUpdate => {} // we only expect NoUpdate or Fail here
                    StrategyCheckpoint::Update(_) => {unreachable!()}
                }
            }

            //increment counter here because we only count
            //iterations that produce a KKT update
            iter += 1;

            // Update the KKT system and the constant parts of its solution.
            // Keep track of the success of each step that calls KKT
            // --------------
            //This should be a Result in Rust, but needs changes down
            //into the KKT solvers to do that.
            let mut is_kkt_solve_success : bool;
            timeit!{timers => "kkt update"; {
                is_kkt_solve_success = self.kktsystem.update(&self.data, &mut self.cones, &self.settings);
            }} // end "kkt update" timer

            // calculate the affine step
            // --------------
            self.step_rhs.affine_step_rhs(
                &self.residuals,
                &self.variables,
                &self.cones,
                self.kktsystem.x_cones_ref(),
            );

            debug_block! {
                eprintln!("--- CPU affine_step_rhs ---");
                eprintln!("CPU: step_rhs={:?}", self.step_rhs);
            }

            timeit!{timers => "kkt solve"; {
                is_kkt_solve_success = is_kkt_solve_success &&
                self.kktsystem.solve(
                    &mut self.step_lhs,
                    &self.step_rhs,
                    &self.data,
                    &self.variables,
                    &mut self.cones,
                    StepDirection::Affine,
                    &self.settings,
                );
            }}  //end "kkt solve affine" timer

            debug_block! {
                eprintln!("--- CPU step_lhs (affine) ---");
                eprintln!("CPU: step_lhs={:?}", self.step_lhs);
            }

            // combined step only on affine step success
            if is_kkt_solve_success {

                //calculate step length and centering parameter
                // --------------
                α = self.get_step_length(StepDirection::Affine, scaling);
                debug_println!("CPU: α (affine)={:.16e}", α.to_f64().unwrap());
                σ = self.centering_parameter(α);
                debug_println!("CPU: σ={:.16e}", σ.to_f64().unwrap());

                // make a reduced Mehrotra correction in the first iteration
                // to accommodate badly centred starting points
                let m = if iter > 1 {T::one()} else {α};
                debug_println!("CPU: m (Mehrotra)={:.16e}", m.to_f64().unwrap());

                // calculate the combined step and length
                // --------------
                self.step_rhs.combined_step_rhs(
                    &self.residuals,
                    &self.variables,
                    &mut self.cones,
                    self.kktsystem.x_cones_mut(),
                    &mut self.step_lhs,
                    σ,
                    μ,
                    m
                );

                debug_block! {
                    eprintln!("--- CPU combined_step_rhs ---");
                    eprintln!("CPU: step_rhs={:?}", self.step_rhs);
                }

                timeit!{timers => "kkt solve" ; {
                    is_kkt_solve_success =
                    self.kktsystem.solve(
                        &mut self.step_lhs,
                        &self.step_rhs,
                        &self.data,
                        &self.variables,
                        &mut self.cones,
                        StepDirection::Combined,
                        &self.settings,
                    );
                }} //end "kkt solve"

                debug_block! {
                    eprintln!("--- CPU step_lhs (combined) ---");
                    eprintln!("CPU: step_lhs={:?}", self.step_lhs);
                }
            }

            if !yolo_mode {
                // check for numerical failure and update strategy
                match self.strategy_checkpoint_numerical_error(is_kkt_solve_success,scaling) {
                    StrategyCheckpoint::NoUpdate => {}
                    StrategyCheckpoint::Update(s) => {α = T::zero(); scaling = s; continue}
                    StrategyCheckpoint::Fail => {α = T::zero(); break}
                }
            }


            // compute final step length and update the current iterate
            // --------------
            α = self.get_step_length(StepDirection::Combined,scaling);

            debug_println!("CPU: α (combined)={:.16e}", α.to_f64().unwrap());

            if !yolo_mode {
                // check for undersized step and update strategy
                match self.strategy_checkpoint_small_step(α, scaling) {
                    StrategyCheckpoint::NoUpdate => {}
                    StrategyCheckpoint::Update(s) => {α = T::zero(); scaling = s; continue}
                    StrategyCheckpoint::Fail => {α = T::zero(); break}
                }

                // Copy previous iterate in case the next one is a dud
                self.info.save_prev_iterate(&self.variables,&mut self.prev_vars);
            }

            self.variables.add_step(&self.step_lhs, α);

            // YOLO mode: save prev_vars as snapshot if variables are NaN-free
            if yolo_mode && !self.variables.has_nan() {
                self.prev_vars.copy_from(&self.variables);
            }
        } //end loop
        // ----------
        // ----------

        }} //end "IP iteration" timer

        }} // end "solve" timer

        let yolo_mode_final = self.settings.core().yolo;

        // YOLO mode: if variables ended up NaN, restore from last valid snapshot
        if yolo_mode_final && self.variables.has_nan() {
            self.variables.copy_from(&self.prev_vars);
        }

        // YOLO mode: save iteration count (info.update was never called)
        if yolo_mode_final {
            self.info.save_scalars(μ, α, σ, iter);
        }

        // Check if we terminated without taking a final step (iter > 0 means we ran at least one).
        // If so, recapture the scalars and print one last line showing the final state.
        // This handles cases like insufficient progress where α becomes zero mid-solve.
        // Don't print for iter=0 since that was already printed in the loop.
        if !yolo_mode_final && α.is_zero() && iter > 0 {
            self.info.save_scalars(μ, α, σ, iter);
            notimeit! {timers; {self.info.print_status(&self.settings).unwrap();}}
        }

        // Refine the cached smoothing iterate to converge to exactly μ_target.
        // This ensures every problem in the batch gets the same μ, producing
        // smooth gradients across parameter sweeps.
        self.refine_smoothing_iterate();

        timeit! {timers => "post-process"; {
            // YOLO mode: skip convergence re-check (always MaxIterations).
            // Non-YOLO: check for "almost" convergence and then extract solution.
            if !yolo_mode_final {
                // Best-iterate fallback runs before post_process. If it
                // restores, status becomes AlmostSolved so the subsequent
                // post_process check is a no-op. If no restore, post_process
                // may still upgrade based on current-iterate metrics.
                self.info.try_restore_best_iterate(
                    &mut self.variables,
                    &self.best_vars,
                );
                self.info.post_process(&self.residuals, &self.settings);
            }
            self.solution
                .post_process(&self.data, &mut self.variables, &self.info, &self.settings);
        }}

        //halt timers
        self.info.finalize(&mut timers);
        self.solution.finalize(&self.info);

        self.info.print_footer(&self.settings).unwrap();

        //stow the timers back into Option in the solver struct
        self.timers.replace(timers);
    }
}

// Encapsulate the internal helpers trait in a private module
// so it doesn't get exported
mod internal {
    use super::super::traits::*;
    use super::*;

    pub(super) trait IPSolverInternals<T, D, V, R, K, C, I, SO, SE> {
        /// Find an initial condition
        fn default_start(&mut self);

        /// Compute a centering parameter
        fn centering_parameter(&self, α: T) -> T;

        /// Compute the current step length
        fn get_step_length(&mut self, step_direction: StepDirection, scaling: ScalingStrategy)
            -> T;

        /// backtrack a step direction to the barrier
        fn backtrack_step_to_barrier(&mut self, αinit: T) -> T;

        /// Scaling strategy checkpointing functions
        fn strategy_checkpoint_insufficient_progress(
            &mut self,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        fn strategy_checkpoint_numerical_error(
            &mut self,
            is_kkt_solve_success: bool,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        fn strategy_checkpoint_small_step(
            &mut self,
            α: T,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        fn strategy_checkpoint_is_scaling_success(
            &mut self,
            is_scaling_success: bool,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint;

        /// Refine the cached smoothing iterate to converge to exactly μ_target
        fn refine_smoothing_iterate(&mut self);
    }

    impl<T, D, V, R, K, C, I, SO, SE> IPSolverInternals<T, D, V, R, K, C, I, SO, SE>
        for Solver<T, D, V, R, K, C, I, SO, SE>
    where
        T: FloatT,
        D: ProblemData<T, V = V>,
        V: Variables<T, D = D, R = R, C = C, SE = SE>,
        R: Residuals<T, D = D, V = V>,
        K: KKTSystem<T, D = D, V = V, C = C, SE = SE>,
        C: Cone<T>,
        I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
        SO: Solution<T, D = D, V = V, I = I>,
        SE: Settings<T>,
    {
        fn default_start(&mut self) {
            if self.cones.is_symmetric() && self.kktsystem.x_cones_ref().is_symmetric() {
                // set all scalings to identity (or zero for the zero cone)
                self.cones.set_identity_scaling();
                // Direct-x cones also start at identity scaling so the KKT
                // (1,1) contribution `Σ E' Hs E` is well-defined on the
                // first factorization.
                for entry in self.kktsystem.x_cones_mut().iter_mut() {
                    entry.cone.set_identity_scaling();
                }
                // Refactor
                self.kktsystem
                    .update(&self.data, &mut self.cones, &self.settings);
                // solve for primal/dual initial points via KKT
                self.kktsystem
                    .solve_initial_point(&mut self.variables, &self.data, &self.settings);
                // fix up (z,s) so that they are in the cone
                self.variables
                    .symmetric_initialization(&mut self.cones, self.kktsystem.x_cones_mut());
            } else {
                // Assigns unit (z,s) and zeros the primal variables
                self.variables
                    .unit_initialization(&self.cones, self.kktsystem.x_cones_ref());
            }
        }

        fn centering_parameter(&self, α: T) -> T {
            T::powi(T::one() - α, 3)
        }

        fn get_step_length(
            &mut self,
            step_direction: StepDirection,
            scaling: ScalingStrategy,
        ) -> T {
            //step length to stay within the cones
            let mut α = self.variables.calc_step_length(
                &self.step_lhs,
                &mut self.cones,
                self.kktsystem.x_cones_mut(),
                &self.settings,
                step_direction,
            );

            // additional barrier function limits for asymmetric cones
            if !(self.cones.is_symmetric() && self.kktsystem.x_cones_ref().is_symmetric())
                && step_direction == StepDirection::Combined
                && scaling == ScalingStrategy::Dual
            {
                let αinit = α;
                α = self.backtrack_step_to_barrier(αinit);
            }
            α
        }

        fn backtrack_step_to_barrier(&mut self, αinit: T) -> T {
            let step = self.settings.core().ipm.linesearch_backtrack_step;
            let mut α = αinit;

            for _ in 0..50 {
                let barrier = self.variables.barrier(
                    &self.step_lhs,
                    α,
                    &mut self.cones,
                    self.kktsystem.x_cones_mut(),
                );
                if barrier < T::one() {
                    return α;
                } else {
                    α = step * α;
                }
            }
            α
        }

        fn strategy_checkpoint_insufficient_progress(
            &mut self,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            let output;
            if self.info.get_status() != SolverStatus::InsufficientProgress {
                // there is no problem, so nothing to do
                output = StrategyCheckpoint::NoUpdate;
            } else {
                // recover old iterate since "insufficient progress" often
                // involves actual degradation of results
                self.info
                    .reset_to_prev_iterate(&mut self.variables, &self.prev_vars);

                // If problem is asymmetric, we can try to continue with the dual-only strategy
                if !(self.cones.is_symmetric() && self.kktsystem.x_cones_ref().is_symmetric())
                    && (scaling == ScalingStrategy::PrimalDual)
                {
                    self.info.set_status(SolverStatus::Unsolved);
                    output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
                } else {
                    output = StrategyCheckpoint::Fail;
                }
            }
            output
        }

        fn strategy_checkpoint_numerical_error(
            &mut self,
            is_kkt_solve_success: bool,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            let output;
            // No update if kkt updates successfully
            if is_kkt_solve_success {
                output = StrategyCheckpoint::NoUpdate;
            }
            // If problem is asymmetric, we can try to continue with the dual-only strategy
            else if !(self.cones.is_symmetric() && self.kktsystem.x_cones_ref().is_symmetric())
                && (scaling == ScalingStrategy::PrimalDual)
            {
                output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
            } else {
                // out of tricks.  Bail out with an error
                self.info.set_status(SolverStatus::NumericalError);
                output = StrategyCheckpoint::Fail;
            }
            output
        }

        fn strategy_checkpoint_small_step(
            &mut self,
            α: T,
            scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            let output;

            if !(self.cones.is_symmetric() && self.kktsystem.x_cones_ref().is_symmetric())
                && scaling == ScalingStrategy::PrimalDual
                && α < self.settings.core().ipm.min_switch_step_length
            {
                output = StrategyCheckpoint::Update(ScalingStrategy::Dual);
            } else if α
                <= T::max(
                    T::zero(),
                    self.settings.core().ipm.min_terminate_step_length,
                )
            {
                self.info.set_status(SolverStatus::InsufficientProgress);
                output = StrategyCheckpoint::Fail;
            } else {
                output = StrategyCheckpoint::NoUpdate;
            }

            output
        }

        fn strategy_checkpoint_is_scaling_success(
            &mut self,
            is_scaling_success: bool,
            _scaling: ScalingStrategy,
        ) -> StrategyCheckpoint {
            if is_scaling_success {
                StrategyCheckpoint::NoUpdate
            } else {
                self.info.set_status(SolverStatus::NumericalError);
                StrategyCheckpoint::Fail
            }
        }
        fn refine_smoothing_iterate(&mut self) {
            use crate::solver::implementations::default::DiffMethod;

            let core = self.settings.core();
            if !core.enable_grad {
                return;
            }
            if core.ipm.diff_method == DiffMethod::Exact {
                return;
            }
            if !self.cones.supports_smoothed() {
                return;
            }
            // Zero-cone-only problems (pure QP equality) use a separate
            // backward path that doesn't need the smoothing iterate.
            // Also avoids division by zero: degree=0 → μ_target=0.
            if self.cones.degree() == 0 {
                return;
            }
            // Only refine when the solve produced a usable iterate.
            // Infeasibility certificates or numerical failures don't
            // give a meaningful central-path iterate to refine from.
            if !matches!(
                self.info.get_status(),
                SolverStatus::Solved
                    | SolverStatus::AlmostSolved
                    | SolverStatus::MaxIterations
                    | SolverStatus::CallbackTerminated
            ) {
                return;
            }

            let μ_target_comp = core.ipm.diff_smoothing_mu;
            let tol_gap_abs = core.ipm.tol_gap_abs;

            // Skip refinement if μ_target is already at or below convergence tolerance
            // (the iterate is already past the target on the central path).
            if μ_target_comp <= tol_gap_abs {
                return;
            }

            let eff_factor = core.ipm.diff_smoothing_step_factor;

            // Fixed iteration count: enough steps to reach μ_target from
            // tol_gap_abs (the convergence tolerance, which bounds initial μ).
            // Extra iterations after convergence are nearly free (step ≈ 0).
            let ratio = μ_target_comp.to_f64().unwrap() / tol_gap_abs.to_f64().unwrap();
            let ln_factor = eff_factor.to_f64().unwrap().ln();
            let n_iters = std::cmp::min(50, (ratio.ln() / ln_factor).ceil() as u32 * 2 + 2);

            // Save the converged solution before modifying variables.
            self.smoothing_vars.copy_from(&self.variables);

            // Normalize HSDE variables before refinement by dividing by
            // max(τ, κ).  For a converged solve τ ≈ 1 and κ ≈ 0, so this
            // effectively sets τ = 1.  Without this, τ varies across batch
            // elements and the 1/τ unscaling in the cache extraction
            // distorts the original-space μ.
            self.variables.rescale();

            let scaling = if self.cones.allows_primal_dual_scaling() {
                ScalingStrategy::PrimalDual
            } else {
                ScalingStrategy::Dual
            };

            // Walk up the central path from the converged μ to μ_target
            // using pure centering steps.  Each iteration targets
            // μ_step = min(μ * eff_factor, μ_target_comp).
            for _iter in 0..n_iters {
                self.residuals.update(&self.variables, &self.data);
                let μ = self.variables.calc_mu(
                    &self.residuals,
                    &self.cones,
                    self.kktsystem.x_cones_ref(),
                );

                let μ_step = T::min(μ * eff_factor, μ_target_comp);

                if !self.variables.scale_cones(
                    &mut self.cones,
                    self.kktsystem.x_cones_mut(),
                    μ,
                    scaling,
                ) {
                    break;
                }

                if !self
                    .kktsystem
                    .update(&self.data, &mut self.cones, &self.settings)
                {
                    break;
                }

                self.step_rhs.affine_step_rhs(
                    &self.residuals,
                    &self.variables,
                    &self.cones,
                    self.kktsystem.x_cones_ref(),
                );

                self.step_rhs.combined_step_rhs(
                    &self.residuals,
                    &self.variables,
                    &mut self.cones,
                    self.kktsystem.x_cones_mut(),
                    &mut self.step_lhs,
                    T::one(),
                    μ_step,
                    T::zero(),
                );

                if !self.kktsystem.solve(
                    &mut self.step_lhs,
                    &self.step_rhs,
                    &self.data,
                    &self.variables,
                    &mut self.cones,
                    StepDirection::Centering,
                    &self.settings,
                ) {
                    break;
                }

                let α = self.get_step_length(StepDirection::Centering, scaling);

                // Sanitize: NaN or negative → 0 (step is skipped, next
                // iteration retries naturally with updated residuals).
                let α = if α.is_nan() || α <= T::zero() {
                    T::zero()
                } else {
                    α
                };

                self.variables.add_step(&self.step_lhs, α);
            }

            // Always mark as cached — the fixed iteration count is designed
            // to be sufficient for convergence. Even if some steps were
            // skipped (α=0), later iterations recover.
            self.smoothing_cached = true;

            // Swap: refined → smoothing_vars, converged → variables.
            std::mem::swap(&mut self.variables, &mut self.smoothing_vars);

            // Restore residuals for the converged solution (used by post_process).
            self.residuals.update(&self.variables, &self.data);
        }
    } // end trait impl
} //end internals module

impl<T, D, V, R, K, C, I, SO, SE> Solver<T, D, V, R, K, C, I, SO, SE>
where
    T: FloatT,
    D: ProblemData<T, V = V>,
    V: Variables<T, D = D, R = R, C = C, SE = SE>,
    R: Residuals<T, D = D, V = V>,
    K: KKTSystem<T, D = D, V = V, C = C, SE = SE>,
    C: Cone<T>,
    I: Info<T, D = D, V = V, R = R, C = C, SE = SE>,
    SO: Solution<T, D = D, V = V, I = I>,
    SE: Settings<T>,
{
    /// Regenerate a smoothing iterate from an externally-supplied solution.
    ///
    /// `backward_with_data` provides problem data and solution directly without
    /// running a solve, so `info.status` is `Unsolved` and the normal status
    /// guard in `refine_smoothing_iterate` would bail early.  Setting status to
    /// `Solved` here reflects that the caller has verified the solution.
    pub(crate) fn refine_smoothing_for_backward(&mut self) {
        self.info.set_status(SolverStatus::Solved);
        self.refine_smoothing_iterate();
    }
}
