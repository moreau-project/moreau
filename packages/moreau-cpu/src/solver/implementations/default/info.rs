use super::*;
use crate::algebra::*;
use crate::io::PrintTarget;
use crate::solver::core::ffi::*;
use crate::solver::core::kktsolvers::LinearSolverInfo;
use crate::solver::core::{traits::Info, SolverStatus};
use crate::solver::traits::Variables;
use crate::timers::*;

/// Standard-form solver type implementing the [`Info`](crate::solver::core::traits::Info) and [`InfoPrint`](crate::solver::core::traits::InfoPrint) traits
#[repr(C)]
#[derive(Debug, Clone)]
pub struct DefaultInfo<T> {
    /// interior point path parameter μ
    pub mu: T,
    /// interior point path parameter reduction ratio σ
    pub sigma: T,
    /// step length for the current iteration
    pub step_length: T,
    /// number of iterations
    pub iterations: u32,
    /// primal objective value
    pub cost_primal: T,
    /// dual objective value
    pub cost_dual: T,
    /// primal residual
    pub res_primal: T,
    /// dual residual
    pub res_dual: T,
    /// primal infeasibility residual
    pub res_primal_inf: T,
    /// dual infeasibility residual
    pub res_dual_inf: T,
    /// absolute duality gap
    pub gap_abs: T,
    /// relative duality gap
    pub gap_rel: T,
    /// κ/τ ratio
    pub ktratio: T,

    // previous iterate
    /// primal object value from previous iteration
    pub(crate) prev_cost_primal: T,
    /// dual objective value from previous iteration
    pub(crate) prev_cost_dual: T,
    /// primal residual from previous iteration
    pub(crate) prev_res_primal: T,
    /// dual residual from previous iteration
    pub(crate) prev_res_dual: T,
    /// absolute duality gap from previous iteration
    pub(crate) prev_gap_abs: T,
    /// relative duality gap from previous iteration
    pub(crate) prev_gap_rel: T,

    // Best-iterate tracking (reduced-tolerance snapshot).
    /// `true` iff a reduced-tolerance iterate has been saved to the
    /// solver's `best_vars` buffer.
    pub(crate) best_saved: bool,
    /// Iteration number at which `best_vars` was last updated.
    pub(crate) best_iter: u32,
    /// Metric snapshots paired with `best_vars`.  Restored together when
    /// the solve ends in InsufficientProgress / NumericalError / MaxIter /
    /// MaxTime after a reduced-tolerance iterate was observed.
    pub(crate) best_cost_primal: T,
    pub(crate) best_cost_dual: T,
    pub(crate) best_res_primal: T,
    pub(crate) best_res_dual: T,
    pub(crate) best_gap_abs: T,
    pub(crate) best_gap_rel: T,
    pub(crate) best_ktratio: T,
    /// construction time (solver structure allocation)
    pub construction_time: f64,
    /// setup time (matrix values, equilibration)
    pub setup_time: f64,
    /// solve time (IPM iterations)
    pub solve_time: f64,
    /// solver status
    pub status: SolverStatus,

    /// linear solver information
    pub linsolver: LinearSolverInfo,

    // target stream for printing
    pub(crate) stream: PrintTarget,
}

impl<T> Default for DefaultInfo<T>
where
    T: FloatT,
{
    fn default() -> Self {
        Self {
            mu: T::zero(),
            sigma: T::zero(),
            step_length: T::zero(),
            iterations: 0,
            cost_primal: T::zero(),
            cost_dual: T::zero(),
            res_primal: T::zero(),
            res_dual: T::zero(),
            res_primal_inf: T::zero(),
            res_dual_inf: T::zero(),
            gap_abs: T::zero(),
            gap_rel: T::zero(),
            ktratio: T::zero(),
            // Initialize prev_* to represent "no previous iterate" so
            // InsufficientProgress checks don't false-trigger
            prev_cost_primal: T::infinity(),
            prev_cost_dual: -T::infinity(),
            prev_res_primal: T::infinity(),
            prev_res_dual: T::infinity(),
            prev_gap_abs: T::infinity(),
            prev_gap_rel: T::infinity(),
            best_saved: false,
            best_iter: 0,
            best_cost_primal: T::zero(),
            best_cost_dual: T::zero(),
            best_res_primal: T::zero(),
            best_res_dual: T::zero(),
            best_gap_abs: T::zero(),
            best_gap_rel: T::zero(),
            best_ktratio: T::zero(),
            construction_time: 0f64,
            setup_time: 0f64,
            solve_time: 0f64,
            status: SolverStatus::default(),
            linsolver: LinearSolverInfo::default(),
            stream: PrintTarget::default(),
        }
    }
}

impl<T> DefaultInfo<T>
where
    T: FloatT,
{
    /// creates a new `DefaultInfo` object
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` iff a reduced-tolerance iterate was snapshotted during the
    /// most recent solve.  Used by the IPM loop to decide whether to fall
    /// back to the snapshot on non-convergent termination; exposed for
    /// tests and diagnostics.
    pub fn best_saved(&self) -> bool {
        self.best_saved
    }

    /// Iteration number at which the best-iterate snapshot was last taken.
    /// Only meaningful when [`best_saved`](Self::best_saved) is `true`.
    pub fn best_iter(&self) -> u32 {
        self.best_iter
    }
}

impl<T: FloatT> MoreauFFI<Self> for DefaultInfo<T> {
    type FFI = super::ffi::DefaultInfoFFI<T>;
}

impl<T> Info<T> for DefaultInfo<T>
where
    T: FloatT,
{
    type V = DefaultVariables<T>;
    type R = DefaultResiduals<T>;

    fn reset(&mut self, timers: &mut Timers) {
        self.status = SolverStatus::Unsolved;
        self.iterations = 0;
        self.solve_time = 0f64;
        self.best_saved = false;
        self.best_iter = 0;

        timers.reset_timer("solve");
    }

    fn post_process(&mut self, residuals: &DefaultResiduals<T>, settings: &DefaultSettings<T>) {
        // if there was an error or we ran out of time
        // or iterations, check for partial convergence

        if self.status.is_errored()
            || matches!(self.status, SolverStatus::MaxIterations)
            || matches!(self.status, SolverStatus::MaxTime)
        {
            self.check_convergence_almost(residuals, settings);
        }
    }

    fn finalize(&mut self, timers: &mut Timers) {
        // Extract individual timer values
        self.construction_time = timers
            .get_timer("setup")
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        self.solve_time = timers
            .get_timer("solve")
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        // setup_time is currently 0 for CPU since matrix values are set in constructor
        // For the three-step API, this would be populated when setup() is called
        self.setup_time = 0.0;
    }

    fn update(
        &mut self,
        data: &mut DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        residuals: &DefaultResiduals<T>,
        timers: &Timers,
    ) {
        // optimality termination check should be computed w.r.t
        // the pre-homogenization x and z variables.
        let τinv = T::recip(variables.τ);

        // unscaled linear term norms
        let normb = data.get_normb();
        let normq = data.get_normq();

        // shortcuts for the equilibration matrices
        let d = &data.equilibration.d;
        let e = &data.equilibration.e;
        let dinv = &data.equilibration.dinv;
        let einv = &data.equilibration.einv;
        let cinv = T::recip(data.equilibration.c);

        // primal and dual costs. dot products are invariant w.r.t
        // equilibration, but we still need to back out the overall
        // objective scaling term c

        let xPx_τinvsq_over2 = residuals.dot_xPx * τinv * τinv / (2.).as_T();
        self.cost_primal = (residuals.dot_qx * τinv + xPx_τinvsq_over2) * cinv;
        self.cost_dual = (-residuals.dot_bz * τinv - xPx_τinvsq_over2) * cinv;

        // variables norms, undoing the equilibration.  Do not unscale
        // by τ yet because the infeasibility residuals are ratios of
        // terms that have no affine parts anyway
        let mut normx = variables.x.norm_scaled(d);
        let mut normz = variables.z.norm_scaled(e) * cinv;
        let mut norms = variables.s.norm_scaled(einv);

        // Direct-x dual `z_x` contributes to the primal-infeasibility
        // certificate `‖A^T z − Σ_J E_J^T z_x‖ → 0` and pairs with `x[J]`
        // (so it unscales by `d[J]`, not `e`). Folding `‖z_x‖_{d|J} · cinv`
        // into the relative-residual denominator keeps the certificate test
        // comparable across slack-only vs. direct-x problems — without it,
        // a certificate with small `‖z‖` but large `‖z_x‖` can inflate the
        // relative residual past `tol_infeas_rel · |b^T z|` and miss firing.
        let norm_zx_d_cinv = if data.dir_cones.is_empty() {
            T::zero()
        } else {
            let mut sumsq = T::zero();
            let mut off = 0usize;
            for xcone in &data.dir_cones {
                let indices = xcone.indices();
                for (k, &idx) in indices.iter().enumerate() {
                    let val = variables.z_x[off + k] * d[idx];
                    sumsq += val * val;
                }
                off += indices.len();
            }
            sumsq.sqrt() * cinv
        };

        // primal and dual infeasibility residuals.
        self.res_primal_inf =
            (residuals.rx_inf.norm_scaled(dinv) * cinv) / T::max(T::one(), normz + norm_zx_d_cinv);
        self.res_dual_inf = T::max(
            residuals.Px.norm_scaled(dinv) / T::max(T::one(), normx),
            residuals.rz_inf.norm_scaled(einv) / T::max(T::one(), normx + norms),
        );

        // now back out the τ scaling so we can normalize the unscaled primal / dual errors
        normx *= τinv;
        normz *= τinv;
        norms *= τinv;

        // primal and dual relative residuals.
        self.res_primal =
            residuals.rz.norm_scaled(einv) * τinv / T::max(T::one(), normb + normx + norms);
        self.res_dual =
            residuals.rx.norm_scaled(dinv) * τinv * cinv / T::max(T::one(), normq + normx + normz);

        // absolute and relative gaps
        self.gap_abs = T::abs(self.cost_primal - self.cost_dual);
        self.gap_rel = self.gap_abs
            / T::max(
                T::one(),
                T::min(T::abs(self.cost_primal), T::abs(self.cost_dual)),
            );

        // κ/τ ratio (scaled)
        self.ktratio = variables.κ * τinv;

        // solve time so far (IPM iterations only)
        self.solve_time = timers
            .get_timer("solve")
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
    }

    fn check_termination(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
        iter: u32,
    ) -> bool {
        //  optimality or infeasibility
        // ---------------------
        self.check_convergence_full(residuals, settings);

        //  poor progress
        // ----------------------
        if self.status == SolverStatus::Unsolved
            && iter > 1u32
            && (self.res_dual > self.prev_res_dual || self.res_primal > self.prev_res_primal)
        {
            // Poor progress at high tolerance.
            if self.ktratio < T::epsilon() * (100.).as_T()
                && (self.prev_gap_abs < settings.ipm.tol_gap_abs
                    || self.prev_gap_rel < settings.ipm.tol_gap_rel)
            {
                self.status = SolverStatus::InsufficientProgress;
            }

            // Going backwards. Stop immediately if residuals diverge out of feasibility tolerance.
            #[allow(clippy::collapsible_if)] // nested if for readability
            if self.ktratio < T::one() {
                if (self.res_dual > settings.ipm.tol_feas * (100.).as_T()
                    && self.res_dual > self.prev_res_dual * (100.).as_T())
                    || (self.res_primal > settings.ipm.tol_feas * (100.).as_T()
                        && self.res_primal > self.prev_res_primal * (100.).as_T())
                {
                    self.status = SolverStatus::InsufficientProgress;
                }
            }
        }

        // Gap regression — mirror of the residual-divergence check above.
        // Catches the HSDE pathology where a bad step collapses τ, blows
        // up gap_abs via the 1/τ scaling, while residuals drift downward.
        // Guarded on prev_gap_abs < reduced_tol so this fires only on
        // "near-optimum then blew up", not on early-iterate oscillation.
        if self.status == SolverStatus::Unsolved
            && iter > 1u32
            && self.ktratio < T::one()
            && self.prev_gap_abs < settings.ipm.reduced_tol_gap_abs
            && self.gap_abs > settings.ipm.reduced_tol_gap_abs * (100.).as_T()
            && self.gap_abs > self.prev_gap_abs * (100.).as_T()
        {
            self.status = SolverStatus::InsufficientProgress;
        }

        // In-loop AlmostSolved on stagnation: iterate has been within reduced
        // tolerances for two consecutive iters, hasn't reached tight tolerance,
        // and hasn't moved meaningfully. Prevents the solver from running to
        // MaxIter (and being ruined by a late bad step) when KKT-solve noise
        // pins gap_rel slightly above the tight tolerance. The "prev also in
        // reduced" guard avoids firing on trivial problems that converge
        // straight through the reduced zone to tight in a single iteration.
        if self.status == SolverStatus::Unsolved && iter > 1u32 {
            let reduced_converged = (self.gap_abs < settings.ipm.reduced_tol_gap_abs
                || self.gap_rel < settings.ipm.reduced_tol_gap_rel)
                && self.res_primal < settings.ipm.reduced_tol_feas
                && self.res_dual < settings.ipm.reduced_tol_feas
                && self.ktratio <= settings.ipm.reduced_tol_ktratio;
            let prev_was_reduced = self.prev_gap_abs < settings.ipm.reduced_tol_gap_abs
                || self.prev_gap_rel < settings.ipm.reduced_tol_gap_rel;
            let not_in_tight = self.gap_abs >= settings.ipm.tol_gap_abs
                && self.gap_rel >= settings.ipm.tol_gap_rel;
            if reduced_converged && prev_was_reduced && not_in_tight {
                let denom = self.prev_gap_abs.abs().max((1e-300).as_T());
                let gap_rel_change = (self.gap_abs - self.prev_gap_abs).abs() / denom;
                // Threshold 1e-4: distinguishes "noise-floor frozen" from
                // "slow but real" convergence near the cone boundary.
                // - cuDSS noise on PSD pins gap changes to ~1e-8 per iter
                //   (4 orders below this threshold) — fires correctly.
                // - GenPowerCone lacks its Mehrotra higher-order correction
                //   (see genpowcone.rs combined_ds_shift), giving linear
                //   ~0.5% per-iter convergence near the boundary (3 orders
                //   above this threshold) — does not fire, letting the
                //   solver run to MaxIter and refine the iterate for the
                //   ill-conditioned backward pass.
                if gap_rel_change < (1e-4).as_T() {
                    self.status = SolverStatus::AlmostSolved;
                }
            }
        }

        // time or iteration limits
        // ----------------------
        if self.status == SolverStatus::Unsolved {
            if settings.max_iter == self.iterations {
                self.status = SolverStatus::MaxIterations;
            } else if self.solve_time > settings.time_limit {
                self.status = SolverStatus::MaxTime;
            }
        }

        // return TRUE if we settled on a final status
        self.status != SolverStatus::Unsolved
    }

    fn save_prev_iterate(&mut self, variables: &Self::V, prev_variables: &mut Self::V) {
        self.prev_cost_primal = self.cost_primal;
        self.prev_cost_dual = self.cost_dual;
        self.prev_res_primal = self.res_primal;
        self.prev_res_dual = self.res_dual;
        self.prev_gap_abs = self.gap_abs;
        self.prev_gap_rel = self.gap_rel;

        prev_variables.copy_from(variables);
    }

    fn reset_to_prev_iterate(&mut self, variables: &mut Self::V, prev_variables: &Self::V) {
        self.cost_primal = self.prev_cost_primal;
        self.cost_dual = self.prev_cost_dual;
        self.res_primal = self.prev_res_primal;
        self.res_dual = self.prev_res_dual;
        self.gap_abs = self.prev_gap_abs;
        self.gap_rel = self.prev_gap_rel;

        variables.copy_from(prev_variables);
    }

    fn try_save_best_iterate(
        &mut self,
        variables: &Self::V,
        best_vars: &mut Self::V,
        settings: &Self::SE,
    ) {
        // Snapshot the iterate that is closest to tight tolerance: among
        // iterates inside the reduced-tolerance zone, keep the one that
        // minimises `max(res_primal / tol_feas, res_dual / tol_feas,
        // gap_abs / tol_gap_abs)` — i.e., the one whose worst convergence
        // metric (relative to the tight tolerance) is smallest.
        //
        // The IPM trades off primal/dual/gap as it pushes toward tight
        // tolerance, so a later in-zone iterate can be strictly worse on
        // any one component than an earlier one. "Latest wins" would
        // restore the late, drifted iterate; "lowest worst-ratio" restores
        // the iterate that came closest to fully-tight convergence.
        let ipm = &settings.ipm;
        let reduced_met = (self.gap_abs < ipm.reduced_tol_gap_abs
            || self.gap_rel < ipm.reduced_tol_gap_rel)
            && self.res_primal < ipm.reduced_tol_feas
            && self.res_dual < ipm.reduced_tol_feas
            && self.ktratio <= ipm.reduced_tol_ktratio;
        if !reduced_met {
            return;
        }
        let score = |rp: T, rd: T, gap: T| -> T {
            let pf = ipm.tol_feas.max(T::epsilon());
            let gf = ipm.tol_gap_abs.max(T::epsilon());
            (rp / pf).max(rd / pf).max(gap / gf)
        };
        let cur = score(self.res_primal, self.res_dual, self.gap_abs);
        if self.best_saved {
            let prev = score(self.best_res_primal, self.best_res_dual, self.best_gap_abs);
            if cur >= prev {
                return;
            }
        }
        self.best_saved = true;
        self.best_iter = self.iterations;
        self.best_cost_primal = self.cost_primal;
        self.best_cost_dual = self.cost_dual;
        self.best_res_primal = self.res_primal;
        self.best_res_dual = self.res_dual;
        self.best_gap_abs = self.gap_abs;
        self.best_gap_rel = self.gap_rel;
        self.best_ktratio = self.ktratio;
        best_vars.copy_from(variables);
    }

    fn try_restore_best_iterate(&mut self, variables: &mut Self::V, best_vars: &Self::V) -> bool {
        // Safety net: if the solve landed on an error / limit status but we
        // did observe a reduced-tol iterate earlier, restore that iterate
        // and report AlmostSolved instead.  The caller guarantees this runs
        // after the main loop, so in-loop convergence checks have already
        // had a chance to promote to Solved/AlmostSolved normally.
        let eligible = self.best_saved
            && (self.status.is_errored()
                || matches!(
                    self.status,
                    SolverStatus::MaxIterations | SolverStatus::MaxTime
                ));
        if !eligible {
            return false;
        }
        self.cost_primal = self.best_cost_primal;
        self.cost_dual = self.best_cost_dual;
        self.res_primal = self.best_res_primal;
        self.res_dual = self.best_res_dual;
        self.gap_abs = self.best_gap_abs;
        self.gap_rel = self.best_gap_rel;
        self.ktratio = self.best_ktratio;
        self.iterations = self.best_iter;
        self.status = SolverStatus::AlmostSolved;
        variables.copy_from(best_vars);
        true
    }

    fn save_scalars(&mut self, μ: T, α: T, σ: T, iter: u32) {
        self.mu = μ;
        self.step_length = α;
        self.sigma = σ;
        self.iterations = iter;
    }

    fn get_status(&self) -> SolverStatus {
        self.status
    }

    fn set_status(&mut self, status: SolverStatus) {
        self.status = status;
    }
}

// Utility functions for convergence checkiing

impl<T> DefaultInfo<T>
where
    T: FloatT,
{
    fn check_convergence_full(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
    ) {
        // "full" tolerances
        let tol_gap_abs = settings.ipm.tol_gap_abs;
        let tol_gap_rel = settings.ipm.tol_gap_rel;
        let tol_feas = settings.ipm.tol_feas;
        let tol_infeas_abs = settings.ipm.tol_infeas_abs;
        let tol_infeas_rel = settings.ipm.tol_infeas_rel;
        let tol_ktratio = settings.ipm.tol_ktratio;

        let solved_status = SolverStatus::Solved;
        let pinf_status = SolverStatus::PrimalInfeasible;
        let dinf_status = SolverStatus::DualInfeasible;

        self.check_convergence(
            residuals,
            tol_gap_abs,
            tol_gap_rel,
            tol_feas,
            tol_infeas_abs,
            tol_infeas_rel,
            tol_ktratio,
            solved_status,
            pinf_status,
            dinf_status,
        );
    }

    fn check_convergence_almost(
        &mut self,
        residuals: &DefaultResiduals<T>,
        settings: &DefaultSettings<T>,
    ) {
        // "almost" tolerances
        let tol_gap_abs = settings.ipm.reduced_tol_gap_abs;
        let tol_gap_rel = settings.ipm.reduced_tol_gap_rel;
        let tol_feas = settings.ipm.reduced_tol_feas;
        let tol_infeas_abs = settings.ipm.reduced_tol_infeas_abs;
        let tol_infeas_rel = settings.ipm.reduced_tol_infeas_rel;
        let tol_ktratio = settings.ipm.reduced_tol_ktratio;

        let solved_status = SolverStatus::AlmostSolved;
        let pinf_status = SolverStatus::AlmostPrimalInfeasible;
        let dinf_status = SolverStatus::AlmostDualInfeasible;

        self.check_convergence(
            residuals,
            tol_gap_abs,
            tol_gap_rel,
            tol_feas,
            tol_infeas_abs,
            tol_infeas_rel,
            tol_ktratio,
            solved_status,
            pinf_status,
            dinf_status,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn check_convergence(
        &mut self,
        residuals: &DefaultResiduals<T>,
        tol_gap_abs: T,
        tol_gap_rel: T,
        tol_feas: T,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
        tol_ktratio: T,
        solved_status: SolverStatus,
        pinf_status: SolverStatus,
        dinf_status: SolverStatus,
    ) {
        if self.ktratio <= T::one() && self.is_solved(tol_gap_abs, tol_gap_rel, tol_feas) {
            self.status = solved_status;
        //hardcoded factor 1000 here should be fixed
        } else if self.ktratio > tol_ktratio.recip() * (1000.0).as_T() {
            if self.is_primal_infeasible(residuals, tol_infeas_abs, tol_infeas_rel) {
                self.status = pinf_status;
            } else if self.is_dual_infeasible(residuals, tol_infeas_abs, tol_infeas_rel) {
                self.status = dinf_status;
            }
        }
    }

    fn is_solved(&self, tol_gap_abs: T, tol_gap_rel: T, tol_feas: T) -> bool {
        ((self.gap_abs < tol_gap_abs) || (self.gap_rel < tol_gap_rel))
            && (self.res_primal < tol_feas)
            && (self.res_dual < tol_feas)
    }

    fn is_primal_infeasible(
        &self,
        residuals: &DefaultResiduals<T>,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
    ) -> bool {
        (residuals.dot_bz < -tol_infeas_abs)
            && (self.res_primal_inf < -tol_infeas_rel * residuals.dot_bz)
    }

    fn is_dual_infeasible(
        &self,
        residuals: &DefaultResiduals<T>,
        tol_infeas_abs: T,
        tol_infeas_rel: T,
    ) -> bool {
        (residuals.dot_qx < -tol_infeas_abs)
            && (self.res_dual_inf < -tol_infeas_rel * residuals.dot_qx)
    }
}

#[cfg(test)]
mod best_iterate_tests {
    use super::*;
    use crate::solver::core::traits::Info;

    fn reduced_feas(settings: &DefaultSettings<f64>) -> f64 {
        settings.ipm.reduced_tol_feas
    }

    fn reduced_gap_abs(settings: &DefaultSettings<f64>) -> f64 {
        settings.ipm.reduced_tol_gap_abs
    }

    fn set_metrics(info: &mut DefaultInfo<f64>, gap: f64, feas: f64, kt: f64) {
        info.gap_abs = gap;
        info.gap_rel = gap;
        info.res_primal = feas;
        info.res_dual = feas;
        info.ktratio = kt;
        info.cost_primal = 1.0;
        info.cost_dual = 1.0;
    }

    /// save_best only snapshots once the iterate meets reduced tolerances,
    /// and prefers the iterate with the smallest worst-ratio to tight
    /// tolerance — a later but worse in-zone iterate must NOT overwrite an
    /// earlier, tighter one.
    #[test]
    fn save_keeps_iterate_closest_to_tight_tol() {
        let settings = DefaultSettings::<f64>::default();
        let mut info = DefaultInfo::<f64>::new();
        let vars = DefaultVariables::<f64>::new(2, 2);
        let mut best = DefaultVariables::<f64>::new(2, 2);

        // Not in zone yet.
        set_metrics(
            &mut info,
            reduced_gap_abs(&settings) * 10.0,
            reduced_feas(&settings) * 10.0,
            0.1,
        );
        info.iterations = 1;
        info.try_save_best_iterate(&vars, &mut best, &settings);
        assert!(!info.best_saved);

        // First in-zone iterate, very close to tight tolerance.
        set_metrics(
            &mut info,
            reduced_gap_abs(&settings) / 1e3,
            reduced_feas(&settings) / 1e3,
            1e-6,
        );
        info.iterations = 5;
        info.try_save_best_iterate(&vars, &mut best, &settings);
        assert!(info.best_saved);
        assert_eq!(info.best_iter, 5);
        let saved_res_p = info.best_res_primal;

        // Later in-zone iterate, but strictly worse on every metric. Must
        // NOT overwrite — best_iter stays at 5.
        set_metrics(
            &mut info,
            reduced_gap_abs(&settings) / 10.0,
            reduced_feas(&settings) / 10.0,
            1e-6,
        );
        info.iterations = 7;
        info.try_save_best_iterate(&vars, &mut best, &settings);
        assert_eq!(info.best_iter, 5);
        assert_eq!(info.best_res_primal, saved_res_p);

        // Still later in-zone iterate that improves to even tighter
        // tolerances — overwrites because it has a smaller worst-ratio.
        set_metrics(
            &mut info,
            reduced_gap_abs(&settings) / 1e6,
            reduced_feas(&settings) / 1e6,
            1e-6,
        );
        info.iterations = 9;
        info.try_save_best_iterate(&vars, &mut best, &settings);
        assert_eq!(info.best_iter, 9);
    }

    /// After an in-zone save, a MaxIterations termination should restore
    /// the snapshot and flip status to AlmostSolved.
    #[test]
    fn restore_promotes_maxiter_to_almost_solved() {
        let settings = DefaultSettings::<f64>::default();
        let mut info = DefaultInfo::<f64>::new();
        let vars = DefaultVariables::<f64>::new(2, 2);
        let mut best = DefaultVariables::<f64>::new(2, 2);

        // Save one good snapshot.
        set_metrics(
            &mut info,
            reduced_gap_abs(&settings) / 10.0,
            reduced_feas(&settings) / 10.0,
            1e-6,
        );
        info.iterations = 11;
        info.try_save_best_iterate(&vars, &mut best, &settings);

        // Degrade the live metrics to simulate iterations drifting back out.
        let bad_gap = reduced_gap_abs(&settings) * 1e3;
        set_metrics(&mut info, bad_gap, reduced_feas(&settings) * 1e3, 0.5);
        info.iterations = 200;
        info.status = SolverStatus::MaxIterations;

        let mut live_vars = DefaultVariables::<f64>::new(2, 2);
        let restored = info.try_restore_best_iterate(&mut live_vars, &best);

        assert!(restored);
        assert_eq!(info.status, SolverStatus::AlmostSolved);
        assert_eq!(info.iterations, 11);
        assert_eq!(info.gap_abs, reduced_gap_abs(&settings) / 10.0);
    }

    /// When nothing was ever snapshotted, restore must be a no-op regardless
    /// of terminal status.  Status must stay as-is.
    #[test]
    fn restore_noop_without_snapshot() {
        let mut info = DefaultInfo::<f64>::new();
        let best = DefaultVariables::<f64>::new(2, 2);
        let mut live = DefaultVariables::<f64>::new(2, 2);
        info.status = SolverStatus::InsufficientProgress;
        assert!(!info.try_restore_best_iterate(&mut live, &best));
        assert_eq!(info.status, SolverStatus::InsufficientProgress);
    }

    /// A Solved status must not be overwritten by the restore pathway —
    /// the solver converged at full tolerance and we keep that result.
    #[test]
    fn restore_noop_when_solved() {
        let settings = DefaultSettings::<f64>::default();
        let mut info = DefaultInfo::<f64>::new();
        let vars = DefaultVariables::<f64>::new(2, 2);
        let mut best = DefaultVariables::<f64>::new(2, 2);

        set_metrics(
            &mut info,
            reduced_gap_abs(&settings) / 10.0,
            reduced_feas(&settings) / 10.0,
            1e-6,
        );
        info.iterations = 12;
        info.try_save_best_iterate(&vars, &mut best, &settings);

        info.status = SolverStatus::Solved;
        info.iterations = 42;
        let mut live = DefaultVariables::<f64>::new(2, 2);
        assert!(!info.try_restore_best_iterate(&mut live, &best));
        assert_eq!(info.status, SolverStatus::Solved);
        assert_eq!(info.iterations, 42);
    }

    /// reset() must clear best_saved so a reused solver doesn't restore
    /// a stale snapshot from the previous solve.
    #[test]
    fn reset_clears_best_saved() {
        let settings = DefaultSettings::<f64>::default();
        let mut info = DefaultInfo::<f64>::new();
        let vars = DefaultVariables::<f64>::new(2, 2);
        let mut best = DefaultVariables::<f64>::new(2, 2);

        set_metrics(
            &mut info,
            reduced_gap_abs(&settings) / 10.0,
            reduced_feas(&settings) / 10.0,
            1e-6,
        );
        info.try_save_best_iterate(&vars, &mut best, &settings);
        assert!(info.best_saved);

        let mut timers = Timers::default();
        info.reset(&mut timers);
        assert!(!info.best_saved);
    }
}
