use super::*;
use crate::solver::core::callbacks::SolverCallbacks;
use crate::solver::traits::Settings;
use crate::{
    io::ConfigurablePrintTarget,
    solver::core::{
        cones::{CompositeCone, SupportedConeT, SupportedXConeT},
        kktsolvers::HasLinearSolverInfo,
        traits::ProblemData,
        SettingsError, Solver,
    },
};
use thiserror::Error;

use crate::algebra::*;
use crate::timers::*;

/// Solver for problems in standard conic program form
pub type DefaultSolver<T = f64> = Solver<
    T,
    DefaultProblemData<T>,
    DefaultVariables<T>,
    DefaultResiduals<T>,
    DefaultKKTSystem<T>,
    CompositeCone<T>,
    DefaultInfo<T>,
    DefaultSolution<T>,
    DefaultSettings<T>,
>;

/// Error types returned by the DefaultSolver

#[derive(Error, Debug)]
/// Error type returned by settings validation
pub enum SolverError {
    /// An error attributable to one of the fields
    #[error("Bad input data: {0}")]
    BadInputData(&'static str),

    /// Error from settings validation with details
    #[error("Bad settings: {0}")]
    SettingsError(#[from] SettingsError),

    /// Error from I/O operations
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Validate that diff_method is compatible with the problem's cones.
/// Smoothed differentiation only supports zero, nonneg, and SOC cones.
fn validate_diff_method_cones<T: FloatT>(
    cones: &[SupportedConeT<T>],
    settings: &DefaultSettings<T>,
) -> Result<(), SolverError> {
    if settings.ipm.diff_method == DiffMethod::Smoothed && !super::diff::supports_smoothed(cones) {
        return Err(SolverError::BadInputData(
            "diff_method='smoothed' requires only zero, nonneg, and SOC cones",
        ));
    }
    Ok(())
}

impl<T> DefaultSolver<T>
where
    T: FloatT,
{
    /// Create a new solver.
    ///
    /// # Arguments
    /// * `P` - Quadratic objective matrix (upper triangular)
    /// * `q` - Linear objective vector
    /// * `A` - Constraint matrix
    /// * `b` - Constraint RHS vector
    /// * `cones` - Cone constraints specification
    /// * `settings` - Solver settings
    pub fn new(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
    ) -> Result<Self, SolverError> {
        Self::new_with_xcones(P, q, A, b, cones, &[], settings)
    }

    /// Create a new solver with direct-x cones constraining sub-vectors of
    /// `x` to cones directly (in addition to the slack cones via `A`).
    ///
    /// When `x_cones` is empty this is identical to [`DefaultSolver::new`].
    pub fn new_with_xcones(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        x_cones: &[SupportedXConeT],
        settings: DefaultSettings<T>,
    ) -> Result<Self, SolverError> {
        //sanity check problem dimensions
        check_dimensions(P, q, A, b, cones)?;
        //sanity check settings
        settings.validate()?;
        validate_diff_method_cones(cones, &settings)?;

        let mut timers = Timers::default();
        let mut output;
        let mut info = DefaultInfo::<T>::new();

        timeit! {timers => "setup"; {

        // user facing results go here.
        let solution = DefaultSolution::<T>::new(A.n, A.m);

        // presolve / chordal decomposition if needed,
        // then take an internal copy of the problem data
        let mut data;
        timeit!{timers => "presolve"; {
            data = DefaultProblemData::<T>::new_with_xcones(P,q,A,b,cones,x_cones,&settings);
        }}

        let cones = CompositeCone::<T>::new(&data.cones);
        assert_eq!(cones.numel, data.m);
        let xn: usize = data.x_cones.iter().map(|c| c.indices().len()).sum();
        let variables = DefaultVariables::<T>::new_with_xn(data.n,data.m,xn);
        let residuals = DefaultResiduals::<T>::new(data.n,data.m);

        // equilibrate problem data immediately on setup.
        // this prevents multiple equlibrations if solve!
        // is called more than once.
        timeit!{timers => "equilibration"; {
            data.equilibrate(&cones,&settings);
        }}

        let kktsystem;
        timeit!{timers => "kktinit"; {
            kktsystem = DefaultKKTSystem::<T>::new(&data,&cones,&settings);
        }}
        info.linsolver = kktsystem.linear_solver_info();

        // work variables for assembling step direction LHS/RHS
        let step_rhs  = DefaultVariables::<T>::new_with_xn(data.n,data.m,xn);
        let step_lhs  = DefaultVariables::<T>::new_with_xn(data.n,data.m,xn);
        let prev_vars = DefaultVariables::<T>::new_with_xn(data.n,data.m,xn);
        let best_vars = DefaultVariables::<T>::new_with_xn(data.n,data.m,xn);
        let smoothing_vars = DefaultVariables::<T>::new_with_xn(data.n,data.m,xn);

        // configure empty user callbacks

        output = Self{
            data,variables,residuals,kktsystem,
            step_lhs,step_rhs,prev_vars,best_vars,info,
            solution,cones,settings,
            timers: None,
            callbacks: SolverCallbacks::default(),
            phantom: std::marker::PhantomData,
            skip_default_start: false,
            smoothing_vars,
            smoothing_cached: false,
        };

        }} //end "setup" timer.

        //now that the timer is finished we can swap our
        //timer object into the solver structure
        output.timers.replace(timers);

        Ok(output)
    }

    /// Create a new solver from CSR matrices.
    pub fn new_from_csr(
        P: &CsrMatrix<T>,
        q: &[T],
        A: &CsrMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
    ) -> Result<Self, SolverError> {
        let P_csc = P.to_csc();
        let A_csc = A.to_csc();
        Self::new(&P_csc, q, &A_csc, b, cones, settings)
    }

    /// Create a solver with symbolic factorization only, using zero-valued sparsity patterns.
    /// This is useful for batch processing where the sparsity pattern is fixed but values change.
    ///
    /// # Arguments
    /// * `n` - Number of decision variables
    /// * `m` - Number of constraints
    /// * `P_pattern` - Sparsity pattern for objective matrix (values can be zeros)
    /// * `A_pattern` - Sparsity pattern for constraint matrix (values can be zeros)
    /// * `cones` - Cone constraints (must match dimension m)
    /// * `settings` - Solver settings (will be modified for batch mode)
    ///
    /// # Batch Mode Settings
    /// The following settings are automatically enforced for batch processing:
    /// - `presolve_enable = false`
    /// - `chordal_decomposition_enable = false` (if SDP feature enabled)
    /// - `input_sparse_dropzeros = false`
    /// - Equilibration is performed per-problem in `solve_batch()`
    ///
    /// # Notes
    /// - SDP cones are disabled in batch mode (all other cone types supported)
    /// - DPi (projection derivative) uses diagonal approximation for some cones
    pub fn new_symbolic(
        n: usize,
        m: usize,
        P_pattern: &CscMatrix<T>,
        A_pattern: &CscMatrix<T>,
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
    ) -> Result<Self, SolverError> {
        Self::new_symbolic_with_xcones_and_perm(
            n,
            m,
            P_pattern,
            A_pattern,
            cones,
            &[],
            settings,
            None,
        )
    }

    /// Create a solver with symbolic factorization, reusing a cached AMD permutation.
    pub fn new_symbolic_with_perm(
        n: usize,
        m: usize,
        P_pattern: &CscMatrix<T>,
        A_pattern: &CscMatrix<T>,
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
        perm: Option<Vec<usize>>,
    ) -> Result<Self, SolverError> {
        Self::new_symbolic_with_xcones_and_perm(
            n,
            m,
            P_pattern,
            A_pattern,
            cones,
            &[],
            settings,
            perm,
        )
    }

    /// Create a solver with symbolic factorization and direct-x cones,
    /// reusing a cached AMD permutation. Generalizes `new_symbolic_with_perm`
    /// to support `x_cones`; pass an empty slice for the slack-only case.
    pub fn new_symbolic_with_xcones_and_perm(
        n: usize,
        m: usize,
        P_pattern: &CscMatrix<T>,
        A_pattern: &CscMatrix<T>,
        cones: &[SupportedConeT<T>],
        x_cones: &[SupportedXConeT],
        mut settings: DefaultSettings<T>,
        perm: Option<Vec<usize>>,
    ) -> Result<Self, SolverError> {
        // Create dummy vectors with correct dimensions
        let q_zeros = vec![T::zero(); n];
        let b_zeros = vec![T::zero(); m];

        // Validate cone constraints for batch mode
        validate_batch_cones(cones)?;

        // Enforce batch mode settings
        settings.ipm.presolve_enable = false;
        settings.ipm.input_sparse_dropzeros = false;
        // Note: equilibration is enabled and will be done per-problem in solve_batch
        #[cfg(feature = "sdp")]
        {
            settings.ipm.chordal_decomposition_enable = false;
        }

        // Check dimensions
        check_dimensions(P_pattern, &q_zeros, A_pattern, &b_zeros, cones)?;
        settings.validate()?;
        validate_diff_method_cones(cones, &settings)?;

        let mut timers = Timers::default();
        let mut output;
        let mut info = DefaultInfo::<T>::new();

        timeit! {timers => "setup"; {

        let solution = DefaultSolution::<T>::new(n, m);

        // Create problem data (with direct-x cones threaded through) with
        // zero-valued patterns. Empty x_cones reduces to the slack-only path.
        let data;
        timeit!{timers => "presolve"; {
            data = DefaultProblemData::<T>::new_with_xcones(
                P_pattern, &q_zeros, A_pattern, &b_zeros, cones, x_cones, &settings,
            );
        }}

        let cones = CompositeCone::<T>::new(&data.cones);
        assert_eq!(cones.numel, data.m);
        let xn: usize = data.x_cones.iter().map(|c| c.indices().len()).sum();
        let variables = DefaultVariables::<T>::new_with_xn(data.n, data.m, xn);
        let residuals = DefaultResiduals::<T>::new(data.n, data.m);

        // Skip equilibration for symbolic factorization

        // This performs the symbolic factorization (AMD ordering, memory allocation)
        let kktsystem;
        timeit!{timers => "kktinit"; {
            kktsystem = DefaultKKTSystem::<T>::new_with_perm(&data, &cones, &settings, perm);
        }}
        info.linsolver = kktsystem.linear_solver_info();

        let step_rhs  = DefaultVariables::<T>::new_with_xn(data.n, data.m, xn);
        let step_lhs  = DefaultVariables::<T>::new_with_xn(data.n, data.m, xn);
        let prev_vars = DefaultVariables::<T>::new_with_xn(data.n, data.m, xn);
        let best_vars = DefaultVariables::<T>::new_with_xn(data.n, data.m, xn);
        let smoothing_vars = DefaultVariables::<T>::new_with_xn(data.n, data.m, xn);

        output = Self{
            data, variables, residuals, kktsystem,
            step_lhs, step_rhs, prev_vars, best_vars, info,
            solution, cones, settings,
            timers: None,
            callbacks: SolverCallbacks::default(),
            phantom: std::marker::PhantomData,
            skip_default_start: false,
            smoothing_vars,
            smoothing_cached: false,
        };

        }} //end "setup" timer.

        output.timers.replace(timers);

        Ok(output)
    }

    // NOTE: new_symbolic_with_bounds removed - bounds no longer supported

    /// Solve an optimization problem with new data, reusing the symbolic factorization.
    ///
    /// # Arguments
    /// * `P` - Objective matrix (must have same sparsity pattern as initialization)
    /// * `q` - Objective vector
    /// * `A` - Constraint matrix (must have same sparsity pattern as initialization)
    /// * `b` - Constraint vector
    ///
    /// # Notes
    /// - This assumes P and A have the same sparsity pattern as used in `new_symbolic()`
    /// - Equilibration is performed automatically per-problem
    /// - The solution is returned in the original (unequilibrated) space
    /// Convert the stored original-space solution to equilibrated coordinates.
    fn solution_to_equilibrated(&self) -> (Vec<T>, Vec<T>, Vec<T>) {
        let dinv = &self.data.equilibration.dinv;
        let e = &self.data.equilibration.e;
        let einv = &self.data.equilibration.einv;
        let c = self.data.equilibration.c;

        let x_eq: Vec<T> = self
            .solution
            .x
            .iter()
            .zip(dinv.iter())
            .map(|(&xi, &di)| xi * di)
            .collect();
        let z_eq: Vec<T> = self
            .solution
            .z
            .iter()
            .zip(einv.iter())
            .map(|(&zi, &ei)| zi * ei * c)
            .collect();
        let s_eq: Vec<T> = self
            .solution
            .s
            .iter()
            .zip(e.iter())
            .map(|(&si, &ei)| si * ei)
            .collect();

        (x_eq, z_eq, s_eq)
    }

    /// Load new problem data, reset equilibration, re-equilibrate, and update the KKT system.
    pub(crate) fn load_and_equilibrate(
        &mut self,
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
    ) {
        self.data.P.nzval.copy_from_slice(&P.nzval);
        self.data.q.copy_from_slice(q);
        self.data.A.nzval.copy_from_slice(&A.nzval);
        self.data.b.copy_from_slice(b);

        // Clamp b to the infinity bound, matching DefaultProblemData::new().
        // Without this, inf values in b propagate through equilibration and
        // produce a singular KKT system in the backward pass.
        let infbound = crate::get_infinity().as_T();
        self.data.b.scalarop(|x| T::min(x, infbound));

        self.data.equilibration.d.fill(T::one());
        self.data.equilibration.dinv.fill(T::one());
        self.data.equilibration.e.fill(T::one());
        self.data.equilibration.einv.fill(T::one());
        self.data.equilibration.c = T::one();

        self.data.clear_normq();
        self.data.clear_normb();

        self.data.equilibrate(&self.cones, &self.settings);

        self.kktsystem.update_P(&self.data.P);
        self.kktsystem.update_A(&self.data.A);
    }

    pub fn solve_batch(
        &mut self,
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
    ) -> Result<(), SolverError> {
        use crate::solver::IPSolver;

        self.load_and_equilibrate(P, q, A, b);

        self.info.status = crate::solver::core::SolverStatus::Unsolved;
        self.solve();

        Ok(())
    }

    /// Precompute equilibration for repeated solves where only b changes.
    pub fn precompute_equilibration(
        &mut self,
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
    ) {
        self.load_and_equilibrate(P, q, A, b);
    }

    /// Solve with precomputed equilibration, only updating b.
    pub fn solve_batch_preequilibrated(&mut self, b: &[T]) -> Result<(), SolverError> {
        use crate::solver::IPSolver;

        for (bi, (&b_new, &e)) in self
            .data
            .b
            .iter_mut()
            .zip(b.iter().zip(self.data.equilibration.e.iter()))
        {
            *bi = b_new * e;
        }

        // Clear cached normb since b has changed
        self.data.clear_normb();

        self.info.status = crate::solver::core::SolverStatus::Unsolved;

        self.solve();

        Ok(())
    }

    /// Compute forward pass: solution perturbations given problem data perturbations.
    ///
    /// Given perturbations to the problem data (dP, dq, dA, db) in the ORIGINAL
    /// (unequilibrated) space, computes the corresponding perturbations to the
    /// solution (dx, dz, ds) also in the original space.
    ///
    /// # Arguments
    /// * `dP` - Perturbation to objective matrix P (original space)
    /// * `dq` - Perturbation to objective vector q (original space)
    /// * `dA` - Perturbation to constraint matrix A (original space)
    /// * `db` - Perturbation to constraint vector b (original space)
    ///
    /// # Returns
    /// Tuple (dx, dz, ds) of solution perturbations in original space
    ///
    /// # Equilibration
    /// This function handles the equilibration scaling internally:
    /// - Converts input perturbations from original to equilibrated space
    /// - Performs differentiation in equilibrated space
    /// - Converts output perturbations back to original space
    pub fn forward_batch(
        &self,
        dP: &CscMatrix<T>,
        dq: &[T],
        dA: &CscMatrix<T>,
        db: &[T],
        diff_method: super::DiffMethod,
        mu: T,
    ) -> Result<(Vec<T>, Vec<T>, Vec<T>), SolverError> {
        // Get equilibration factors
        let d = &self.data.equilibration.d;
        let e = &self.data.equilibration.e;
        let einv = &self.data.equilibration.einv;
        let c = self.data.equilibration.c;

        let (x_eq, z_eq, s_eq) = self.solution_to_equilibrated();

        // Convert input perturbations from original to equilibrated space:
        // dP_eq = c * D * dP * D
        // dq_eq = c * D * dq
        // dA_eq = E * dA * D
        // db_eq = E * db
        let mut dP_eq = dP.clone();
        for (j, col_range) in dP_eq.colptr.windows(2).enumerate() {
            for idx in col_range[0]..col_range[1] {
                let i = dP_eq.rowval[idx];
                dP_eq.nzval[idx] = dP_eq.nzval[idx] * c * d[i] * d[j];
            }
        }

        let dq_eq: Vec<T> = dq
            .iter()
            .zip(d.iter())
            .map(|(&qi, &di)| qi * c * di)
            .collect();

        let mut dA_eq = dA.clone();
        for (j, col_range) in dA_eq.colptr.windows(2).enumerate() {
            for idx in col_range[0]..col_range[1] {
                let i = dA_eq.rowval[idx];
                dA_eq.nzval[idx] = dA_eq.nzval[idx] * e[i] * d[j];
            }
        }

        let db_eq: Vec<T> = db.iter().zip(e.iter()).map(|(&bi, &ei)| bi * ei).collect();

        // τ = 1 because we differentiate in de-homogenized coordinates:
        // variables.unscale() divides x, z, s by τ before storing in solution,
        // so the equilibrated iterate used here already has τ factored out.
        let tau = T::one();

        // Call the differentiation routine in equilibrated space
        let result = super::diff::differentiate(
            &self.data.P,
            &self.data.q,
            &self.data.A,
            &self.data.b,
            &self.data.cones,
            &x_eq,
            &s_eq,
            &z_eq,
            tau,
            &dP_eq,
            &dq_eq,
            &dA_eq,
            &db_eq,
            diff_method,
            mu,
        );

        // Convert output perturbations from equilibrated to original space:
        // dx_orig = D * dx_eq
        // dz_orig = E * dz_eq / c
        // ds_orig = E^{-1} * ds_eq
        let dx: Vec<T> = result
            .dx
            .iter()
            .zip(d.iter())
            .map(|(&dxi, &di)| dxi * di)
            .collect();
        let dz: Vec<T> = result
            .dz
            .iter()
            .zip(e.iter())
            .map(|(&dzi, &ei)| dzi * ei / c)
            .collect();
        let ds: Vec<T> = result
            .ds
            .iter()
            .zip(einv.iter())
            .map(|(&dsi, &ei)| dsi * ei)
            .collect();

        Ok((dx, dz, ds))
    }

    /// Compute gradients of the solution with respect to problem data using the adjoint method.
    ///
    /// Given upstream gradients (dx_bar, dz_bar, ds_bar) w.r.t. the solution in
    /// the ORIGINAL (unequilibrated) space, computes the gradients w.r.t. the
    /// problem data (dP, dq, dA, db) also in the original space.
    ///
    /// # Arguments
    /// * `dx` - Upstream gradient w.r.t. primal solution x (original space)
    /// * `ds` - Upstream gradient w.r.t. slack variables s (original space)
    /// * `dz` - Upstream gradient w.r.t. dual variables z (original space)
    /// * `grad_state` - Optional pre-computed gradient state for efficient backward pass.
    ///                  If provided, uses cached symbolic factorization. If None, performs
    ///                  fresh factorization (slower on first call).
    ///
    /// # Returns
    /// BackwardResult containing gradients (dP, dq, dA, db) in original space
    ///
    /// # Equilibration
    /// This function handles the equilibration scaling internally:
    /// - Converts upstream gradients from original to equilibrated space
    /// - Performs adjoint differentiation in equilibrated space
    /// - Converts output gradients back to original space
    pub fn backward_batch(
        &self,
        dx: &[T],
        ds: &[T],
        dz: &[T],
        grad_state: Option<&mut super::diff::GradState<T>>,
        diff_method: super::DiffMethod,
    ) -> Result<BackwardResult<T>, SolverError> {
        // Slack-only convenience entry-point: drop dz_x.
        self.backward_batch_with_dz_x(dx, ds, dz, &[], grad_state, diff_method)
    }

    /// `backward_batch` variant that also accepts an upstream gradient on the
    /// direct-x dual `z_x` (in original space). For slack-only or when
    /// gradients on `z_x` are not needed, pass an empty slice.
    pub fn backward_batch_with_dz_x(
        &self,
        dx: &[T],
        ds: &[T],
        dz: &[T],
        dz_x: &[T],
        grad_state: Option<&mut super::diff::GradState<T>>,
        diff_method: super::DiffMethod,
    ) -> Result<BackwardResult<T>, SolverError> {
        // Direct-x cones go through `differentiate_adjoint_hsde_with_xcones`
        // (or the `GradState`-cached equivalent when a state is supplied),
        // which augments the HSDE Jacobian directly with `xn` extra rows +
        // a `du_x` column block carrying the direct-x cone-projection
        // Jacobian `H_x`. Same algebraic system the slack unfold gives, but
        // without rebuilding `A`, `cones`, or the iterate at backward time.
        // Per v1 scope, direct-x dual gradients are dropped (zero on the
        // direct-x slot of the adjoint RHS).
        let has_x_cones = !self.data.x_cones.is_empty();

        // Resolve Auto using the converged μ and cone types.
        let diff_method =
            super::diff::resolve_diff_method(diff_method, &self.data.cones, self.info.mu);

        // Get equilibration factors
        let d = &self.data.equilibration.d;
        let e = &self.data.equilibration.e;
        let einv = &self.data.equilibration.einv;
        let c = self.data.equilibration.c;

        // For Smoothed differentiation with a cached refinement iterate,
        // use the smoothing_vars directly — they are already in equilibrated
        // HSDE coordinates with τ=1 from the refinement loop.
        // For Exact or when no smoothing iterate is cached, convert the
        // original-space solution back to equilibrated coordinates.
        let use_smoothing = diff_method == super::DiffMethod::Smoothed && self.smoothing_cached;

        let (x_eq, z_eq, s_eq) = if use_smoothing {
            let tau_inv = T::recip(self.smoothing_vars.τ);
            let x_eq: Vec<T> = self
                .smoothing_vars
                .x
                .iter()
                .map(|&xi| xi * tau_inv)
                .collect();
            let z_eq: Vec<T> = self
                .smoothing_vars
                .z
                .iter()
                .map(|&zi| zi * tau_inv)
                .collect();
            let s_eq: Vec<T> = self
                .smoothing_vars
                .s
                .iter()
                .map(|&si| si * tau_inv)
                .collect();
            (x_eq, z_eq, s_eq)
        } else {
            self.solution_to_equilibrated()
        };

        // Compute μ from the equilibrated iterate (used for Auto resolution
        // and as the smoothing parameter in H_μ).
        // Use degree (not m) since zero-cone s·z contributions are always 0
        // and shouldn't dilute the complementarity average.
        let degree: usize = self.data.cones.iter().map(|c| c.degree()).sum();
        let dot_sz: T = s_eq
            .iter()
            .zip(z_eq.iter())
            .map(|(&si, &zi)| si * zi)
            .fold(T::zero(), |a, b| a + b);
        let mu = if degree > 0 {
            dot_sz / T::from(degree).unwrap()
        } else {
            dot_sz
        };

        // Convert upstream gradients from original to equilibrated space.
        // dL/dx_eq = D * dx_bar
        // dL/dz_eq = E/c * dz_bar
        // dL/ds_eq = E^{-1} * ds_bar
        let dx_eq: Vec<T> = dx
            .iter()
            .zip(d.iter())
            .map(|(&dxi, &di)| dxi * di)
            .collect();
        let dz_eq: Vec<T> = dz
            .iter()
            .zip(e.iter())
            .map(|(&dzi, &ei)| dzi * ei / c)
            .collect();
        let ds_eq: Vec<T> = ds
            .iter()
            .zip(einv.iter())
            .map(|(&dsi, &ei)| dsi * ei)
            .collect();

        // τ = 1 because we differentiate in de-homogenized coordinates:
        // variables.unscale() divides x, z, s by τ before storing in solution,
        // so the equilibrated iterate used here already has τ factored out.
        let tau = T::one();

        let n = self.data.n;
        let m = self.data.m;
        let _ = n;
        let _ = m;

        // Direct-x dual `z_x_eq` in the solver's equilibrated frame:
        // unscale runs `z_x_orig = z_x_eq * d[J] / c`, so the inverse is
        // `z_x_eq[k] = z_x_orig[k] * c / d[J[k]]`.
        let xcone_indices: Vec<Vec<usize>> = self
            .data
            .x_cones
            .iter()
            .map(|xc| xc.indices().to_vec())
            .collect();
        let z_x_eq: Vec<T> = if has_x_cones {
            let mut out = Vec::with_capacity(xcone_indices.iter().map(|ix| ix.len()).sum());
            let mut zx_off = 0usize;
            for ix in &xcone_indices {
                for &idx in ix {
                    out.push(self.variables.z_x[zx_off] * c / d[idx]);
                    zx_off += 1;
                }
            }
            out
        } else {
            Vec::new()
        };

        // Convert the user-frame dz_x into equilibrated frame:
        //   z_x_orig = z_x_eq * d[J] / c  ⇒  dz_x_eq = dz_x_orig * d[J] / c.
        // Empty input means "no upstream gradient on z_x" — same as before
        // task-5 plumbing. xn-zero in this case.
        let xn_total: usize = xcone_indices.iter().map(|ix| ix.len()).sum();
        let dz_x_eq: Vec<T> = if has_x_cones && !dz_x.is_empty() {
            if dz_x.len() != xn_total {
                return Err(SolverError::BadInputData(
                    "dz_x length must equal total direct-x dimension",
                ));
            }
            let mut out = Vec::with_capacity(xn_total);
            let mut k = 0usize;
            for ix in &xcone_indices {
                for &idx in ix {
                    out.push(dz_x[k] * d[idx] / c);
                    k += 1;
                }
            }
            out
        } else {
            vec![T::zero(); xn_total]
        };

        let mut result = if let Some(gs) = grad_state {
            // Cached path: slack-only when `x_cones` were empty at construction,
            // otherwise reuses the augmented symbolic factorization.
            gs.solve_adjoint(
                &self.data.P,
                &self.data.q,
                &self.data.A,
                &self.data.b,
                &self.data.cones,
                &x_eq,
                &s_eq,
                &z_eq,
                &z_x_eq,
                tau,
                &dx_eq,
                &dz_eq,
                &ds_eq,
                &dz_x_eq,
                diff_method,
                mu,
            )
        } else {
            super::diff::differentiate_adjoint_hsde_with_xcones(
                &self.data.P,
                &self.data.q,
                &self.data.A,
                &self.data.b,
                &self.data.cones,
                &self.data.x_cones,
                &xcone_indices,
                &x_eq,
                &s_eq,
                &z_eq,
                &z_x_eq,
                tau,
                &dx_eq,
                &dz_eq,
                &ds_eq,
                &dz_x_eq,
                diff_method,
                mu,
            )
        };

        // The IFT-direct path returns gradients sized for the original
        // (P, A, q, b); no slicing needed. Skip the post-slicing block.
        if false {
            // dA: keep only the first m rows. Iterate columns and drop
            // any entries with row >= m.
            let dA_old = result.dA;
            let mut new_colptr = vec![0usize; dA_old.colptr.len()];
            let mut new_rowval = Vec::new();
            let mut new_nzval = Vec::new();
            for j in 0..n {
                new_colptr[j] = new_rowval.len();
                for idx in dA_old.colptr[j]..dA_old.colptr[j + 1] {
                    let r = dA_old.rowval[idx];
                    if r < m {
                        new_rowval.push(r);
                        new_nzval.push(dA_old.nzval[idx]);
                    }
                }
            }
            new_colptr[n] = new_rowval.len();
            result.dA = CscMatrix::new(m, n, new_colptr, new_rowval, new_nzval);
            result.db.truncate(m);
        }

        // Convert output gradients from equilibrated to original space.
        // P_eq = c * D * P * D  =>  dL/dP = c * D * dL/dP_eq * D
        // q_eq = c * D * q      =>  dL/dq = c * D * dL/dq_eq
        // A_eq = E * A * D      =>  dL/dA = E * dL/dA_eq * D
        // b_eq = E * b          =>  dL/db = E * dL/db_eq
        for (j, col_range) in result.dP.colptr.windows(2).enumerate() {
            for idx in col_range[0]..col_range[1] {
                let i = result.dP.rowval[idx];
                result.dP.nzval[idx] = result.dP.nzval[idx] * c * d[i] * d[j];
            }
        }

        for (dqi, &di) in result.dq.iter_mut().zip(d.iter()) {
            *dqi = *dqi * c * di;
        }

        // Convert dA from equilibrated to original space
        for (j, col_range) in result.dA.colptr.windows(2).enumerate() {
            for idx in col_range[0]..col_range[1] {
                let row = result.dA.rowval[idx];
                result.dA.nzval[idx] = result.dA.nzval[idx] * e[row] * d[j];
            }
        }

        // Convert db from equilibrated to original space
        let db: Vec<T> = result
            .db
            .iter()
            .zip(e.iter())
            .map(|(&dbi, &ei)| dbi * ei)
            .collect();

        Ok(BackwardResult {
            dP: result.dP,
            dq: result.dq,
            dA: result.dA,
            db,
            // Convert debug smoothing iterates from equilibrated to original space.
            // x_eq = dinv * x_orig  =>  x_orig = d * x_eq
            // z_eq = einv * c * z_orig  =>  z_orig = e * z_eq / c
            // s_eq = e * s_orig  =>  s_orig = einv * s_eq
            #[cfg(debug_assertions)]
            debug_smoothing_x: x_eq
                .iter()
                .zip(d.iter())
                .map(|(&xi, &di)| xi * di)
                .collect(),
            #[cfg(debug_assertions)]
            debug_smoothing_z: z_eq
                .iter()
                .zip(e.iter())
                .map(|(&zi, &ei)| zi * ei / c)
                .collect(),
            #[cfg(debug_assertions)]
            debug_smoothing_s: s_eq
                .iter()
                .zip(einv.iter())
                .map(|(&si, &ei)| si * ei)
                .collect(),
        })
    }
}

// Re-export BackwardResult from diff module
pub use super::diff::BackwardResult;

fn check_dimensions<T: FloatT>(
    P: &CscMatrix<T>,
    q: &[T],
    A: &CscMatrix<T>,
    b: &[T],
    cone_types: &[SupportedConeT<T>],
) -> Result<(), SolverError> {
    let m = b.len();
    let n = q.len();
    let p = cone_types.iter().fold(0, |acc, cone| acc + cone.nvars());

    // Check P dimensions
    if !P.is_square() {
        return Err(SolverError::BadInputData("P must be square"));
    }
    if P.nrows() != n {
        return Err(SolverError::BadInputData(
            "P dimensions incompatible with q (expected n x n)",
        ));
    }

    // Check P is symmetric (or upper triangular, which we treat as symmetric)
    if !P.is_triu() && !P.is_symmetric() {
        return Err(SolverError::BadInputData(
            "P must be symmetric (or upper triangular)",
        ));
    }

    // Check A dimensions
    if A.nrows() != m {
        return Err(SolverError::BadInputData(
            "A row dimension incompatible with b",
        ));
    }
    if A.ncols() != n {
        return Err(SolverError::BadInputData(
            "A column dimension incompatible with q",
        ));
    }

    // Check cone dimensions sum to m
    if p != m {
        return Err(SolverError::BadInputData(
            "Cone dimensions must sum to number of constraints (m)",
        ));
    }

    // Validate individual cone parameters
    validate_cones(cone_types)?;

    Ok(())
}

fn validate_cones<T: FloatT>(cones: &[SupportedConeT<T>]) -> Result<(), SolverError> {
    for cone in cones {
        match cone {
            SupportedConeT::ZeroConeT(dim) => {
                if *dim == 0 {
                    return Err(SolverError::BadInputData(
                        "ZeroCone dimension must be positive",
                    ));
                }
            }
            SupportedConeT::NonnegativeConeT(dim) => {
                if *dim == 0 {
                    return Err(SolverError::BadInputData(
                        "NonnegativeCone dimension must be positive",
                    ));
                }
            }
            SupportedConeT::SecondOrderConeT(dim) => {
                if *dim < 2 {
                    return Err(SolverError::BadInputData(
                        "SecondOrderCone dimension must be >= 2",
                    ));
                }
            }
            SupportedConeT::ExponentialConeT() => {
                // Exponential cone is always 3D, no validation needed
            }
            SupportedConeT::PowerConeT(alpha) => {
                if *alpha <= T::zero() || *alpha >= T::one() {
                    return Err(SolverError::BadInputData(
                        "PowerCone alpha must be in (0, 1)",
                    ));
                }
            }
            SupportedConeT::GenPowerConeT(ref alpha, dim2) => {
                if alpha.is_empty() {
                    return Err(SolverError::BadInputData(
                        "GenPowerCone: alpha must have at least one element",
                    ));
                }
                if *dim2 < 1 {
                    return Err(SolverError::BadInputData("GenPowerCone: dim2 must be >= 1"));
                }
                if !alpha.iter().all(|&a| a > T::zero()) {
                    return Err(SolverError::BadInputData(
                        "GenPowerCone: all alpha values must be > 0",
                    ));
                }
                let sum: T = alpha.iter().fold(T::zero(), |acc, &a| acc + a);
                let base: T = (1e-8).as_T();
                let tol = base * alpha.len().as_T();
                if (sum - T::one()).abs() > tol {
                    return Err(SolverError::BadInputData(
                        "GenPowerCone: alpha values must sum to 1",
                    ));
                }
            }
            #[cfg(feature = "sdp")]
            SupportedConeT::PSDTriangleConeT(dim) => {
                if *dim == 0 {
                    return Err(SolverError::BadInputData(
                        "PSDTriangleCone dimension must be positive",
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_batch_cones<T: FloatT>(_cones: &[SupportedConeT<T>]) -> Result<(), SolverError> {
    // All cone restrictions (SOC dim >= 2, no PSDTriangleCone, GenPowerCone alpha validation)
    // are enforced in validate_cones(), so no additional batch-specific validation is needed.
    Ok(())
}

impl<T> ConfigurablePrintTarget for DefaultSolver<T>
where
    T: FloatT,
{
    fn print_to_stdout(&mut self) {
        self.info.print_to_stdout();
    }
    fn print_to_file(&mut self, file: std::fs::File) {
        self.info.print_to_file(file)
    }
    fn print_to_stream(&mut self, stream: Box<dyn std::io::Write + Send + Sync>) {
        self.info.print_to_stream(stream)
    }
    fn print_to_sink(&mut self) {
        self.info.print_to_sink()
    }
    fn print_to_buffer(&mut self) {
        self.info.print_to_buffer();
    }
    fn get_print_buffer(&mut self) -> std::io::Result<String> {
        self.info.get_print_buffer()
    }
}

#[cfg(test)]
mod input_validation_tests {
    use super::*;

    fn make_test_data() -> (CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>) {
        let P = CscMatrix::<f64>::zeros((4, 4));
        let q = vec![0.; 4];
        let A = CscMatrix::<f64>::zeros((6, 4));
        let b = vec![0.; 6];
        (P, q, A, b)
    }

    #[test]
    fn test_asymmetric_p_rejected() {
        let q = vec![0.; 4];
        let A = CscMatrix::<f64>::zeros((6, 4));
        let b = vec![0.; 6];
        let cones = vec![
            SupportedConeT::ZeroConeT(1),
            SupportedConeT::NonnegativeConeT(2),
            SupportedConeT::NonnegativeConeT(3),
        ];

        // Non-symmetric, non-upper-triangular P: P[0,1]=2.0 but P[1,0]=3.0
        let P_asym = CscMatrix::new(
            4,
            4,
            vec![0, 2, 3, 4, 5],
            vec![0, 1, 0, 2, 3],
            vec![1.0, 3.0, 2.0, 1.0, 1.0],
        );
        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P_asym, &q, &A, &b, &cones, settings).is_err());
    }

    #[test]
    fn test_symmetric_p_accepted() {
        // Full symmetric P matrix
        let P = CscMatrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![2.0, 1.0, 1.0, 2.0], // Symmetric: P[0,1] = P[1,0] = 1.0
        );
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((2, 2));
        let b = vec![0.; 2];
        let cones = vec![SupportedConeT::NonnegativeConeT(2)];

        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_ok());
    }

    #[test]
    fn test_upper_triangular_p_accepted() {
        // Upper triangular P (implicit symmetric)
        let P = CscMatrix::new(
            4,
            4,
            vec![0, 1, 2, 3, 4],
            vec![0, 0, 2, 3],
            vec![1.0, 2.0, 1.0, 1.0],
        );
        let q = vec![0.; 4];
        let A = CscMatrix::<f64>::zeros((6, 4));
        let b = vec![0.; 6];
        let cones = vec![
            SupportedConeT::ZeroConeT(1),
            SupportedConeT::NonnegativeConeT(2),
            SupportedConeT::NonnegativeConeT(3),
        ];

        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_ok());
    }

    #[test]
    fn test_zero_dim_zero_cone_rejected() {
        let (P, q, A, b) = make_test_data();
        let cones = vec![
            SupportedConeT::ZeroConeT(0),
            SupportedConeT::NonnegativeConeT(6),
        ];
        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
    }

    #[test]
    fn test_zero_dim_nonneg_cone_rejected() {
        let (P, q, A, b) = make_test_data();
        let cones = vec![
            SupportedConeT::NonnegativeConeT(0),
            SupportedConeT::ZeroConeT(6),
        ];
        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
    }

    #[test]
    fn test_soc_dim_one_rejected() {
        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((1, 2));
        let b = vec![0.; 1];
        let cones = vec![SupportedConeT::SecondOrderConeT(1)];

        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
    }

    #[test]
    fn test_soc_dim_two_accepted() {
        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((2, 2));
        let b = vec![0.; 2];
        // SOC dim=2 is now accepted (minimum dim is 2)
        let cones = vec![SupportedConeT::SecondOrderConeT(2)];

        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_ok());
    }

    #[test]
    fn test_soc_dim_three_accepted() {
        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((3, 2));
        let b = vec![0.; 3];
        let cones = vec![SupportedConeT::SecondOrderConeT(3)];

        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_ok());
    }

    #[test]
    fn test_power_cone_alpha_zero_rejected() {
        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((3, 2));
        let b = vec![0.; 3];
        let cones = vec![SupportedConeT::PowerConeT(0.0)];

        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
    }

    #[test]
    fn test_power_cone_alpha_one_rejected() {
        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((3, 2));
        let b = vec![0.; 3];
        let cones = vec![SupportedConeT::PowerConeT(1.0)];

        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
    }

    #[test]
    fn test_power_cone_valid_alpha_accepted() {
        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((3, 2));
        let b = vec![0.; 3];
        let cones = vec![SupportedConeT::PowerConeT(0.5)];

        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_ok());
    }

    #[test]
    fn test_genpow_valid_accepted() {
        // Valid GenPowerCone: alpha=[0.5, 0.5], dim2=2 => total dim=4
        let P = CscMatrix::<f64>::zeros((3, 3));
        let q = vec![0.; 3];
        let A = CscMatrix::<f64>::zeros((4, 3));
        let b = vec![0.; 4];
        let cones = vec![SupportedConeT::GenPowerConeT(vec![0.5, 0.5], 2)];

        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_ok());
    }

    #[test]
    fn test_genpow_bad_alpha_rejected() {
        // GenPowerCone with alphas not summing to 1
        let P = CscMatrix::<f64>::zeros((3, 3));
        let q = vec![0.; 3];
        let A = CscMatrix::<f64>::zeros((4, 3));
        let b = vec![0.; 4];
        let cones = vec![SupportedConeT::GenPowerConeT(vec![0.3, 0.3], 2)];
        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());

        // GenPowerCone with negative alpha
        let cones = vec![SupportedConeT::GenPowerConeT(vec![-0.5, 1.5], 2)];
        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());

        // GenPowerCone with empty alpha
        let cones = vec![SupportedConeT::GenPowerConeT(vec![], 2)];
        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
    }

    #[test]
    fn test_p_wrong_size_rejected() {
        let P = CscMatrix::<f64>::zeros((3, 3)); // Wrong: should be 4x4
        let q = vec![0.; 4];
        let A = CscMatrix::<f64>::zeros((6, 4));
        let b = vec![0.; 6];
        let cones = vec![
            SupportedConeT::ZeroConeT(1),
            SupportedConeT::NonnegativeConeT(2),
            SupportedConeT::NonnegativeConeT(3),
        ];

        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
    }

    #[test]
    fn test_a_wrong_cols_rejected() {
        let P = CscMatrix::<f64>::zeros((4, 4));
        let q = vec![0.; 4];
        let A = CscMatrix::<f64>::zeros((6, 3)); // Wrong: should be 6x4
        let b = vec![0.; 6];
        let cones = vec![
            SupportedConeT::ZeroConeT(1),
            SupportedConeT::NonnegativeConeT(2),
            SupportedConeT::NonnegativeConeT(3),
        ];

        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
    }

    #[test]
    fn test_a_wrong_rows_rejected() {
        let P = CscMatrix::<f64>::zeros((4, 4));
        let q = vec![0.; 4];
        let A = CscMatrix::<f64>::zeros((5, 4)); // Wrong: should be 6x4
        let b = vec![0.; 6];
        let cones = vec![
            SupportedConeT::ZeroConeT(1),
            SupportedConeT::NonnegativeConeT(2),
            SupportedConeT::NonnegativeConeT(3),
        ];

        let settings = DefaultSettings::default();
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
    }

    #[test]
    fn test_smoothed_with_soc_accepted() {
        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((3, 2));
        let b = vec![0.; 3];
        let cones = vec![SupportedConeT::SecondOrderConeT(3)];

        let mut settings = DefaultSettings::default();
        settings.ipm.diff_method = DiffMethod::Smoothed;
        // SOC with smoothed should now be accepted (not rejected)
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_ok());
    }

    #[test]
    fn test_smoothed_with_exp_rejected() {
        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((3, 2));
        let b = vec![0.; 3];
        let cones = vec![SupportedConeT::ExponentialConeT()];

        let mut settings = DefaultSettings::default();
        settings.ipm.diff_method = DiffMethod::Smoothed;
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
    }

    #[test]
    fn test_smoothed_with_power_rejected() {
        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((3, 2));
        let b = vec![0.; 3];
        let cones = vec![SupportedConeT::PowerConeT(0.5)];

        let mut settings = DefaultSettings::default();
        settings.ipm.diff_method = DiffMethod::Smoothed;
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_err());
    }

    #[test]
    fn test_smoothed_with_lp_only_accepted() {
        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((3, 2));
        let b = vec![0.; 3];
        let cones = vec![
            SupportedConeT::ZeroConeT(1),
            SupportedConeT::NonnegativeConeT(2),
        ];

        let mut settings = DefaultSettings::default();
        settings.ipm.diff_method = DiffMethod::Smoothed;
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_ok());
    }

    #[test]
    fn test_auto_with_soc_accepted() {
        // Auto is always allowed — it resolves at runtime
        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((3, 2));
        let b = vec![0.; 3];
        let cones = vec![SupportedConeT::SecondOrderConeT(3)];

        let mut settings = DefaultSettings::default();
        settings.ipm.diff_method = DiffMethod::Auto;
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_ok());
    }

    #[test]
    fn test_exact_with_soc_accepted() {
        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((3, 2));
        let b = vec![0.; 3];
        let cones = vec![SupportedConeT::SecondOrderConeT(3)];

        let mut settings = DefaultSettings::default();
        settings.ipm.diff_method = DiffMethod::Exact;
        assert!(DefaultSolver::new(&P, &q, &A, &b, &cones, settings).is_ok());
    }

    #[test]
    fn test_update_settings_smoothed_with_soc_accepted() {
        use crate::solver::IPSolver;

        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((3, 2));
        let b = vec![0.; 3];
        let cones = vec![SupportedConeT::SecondOrderConeT(3)];

        let settings = DefaultSettings::default();
        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();

        // Update to Smoothed — should now be accepted for SOC
        let mut new_settings = solver.settings().clone();
        new_settings.ipm.diff_method = DiffMethod::Smoothed;
        assert!(solver.update_settings(new_settings).is_ok());
    }

    #[test]
    fn test_update_settings_smoothed_with_lp_accepted() {
        use crate::solver::IPSolver;

        let P = CscMatrix::<f64>::zeros((2, 2));
        let q = vec![0.; 2];
        let A = CscMatrix::<f64>::zeros((2, 2));
        let b = vec![0.; 2];
        let cones = vec![SupportedConeT::NonnegativeConeT(2)];

        let settings = DefaultSettings::default();
        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();

        // Update to Smoothed with LP-only cones — should succeed
        let mut new_settings = solver.settings().clone();
        new_settings.ipm.diff_method = DiffMethod::Smoothed;
        assert!(solver.update_settings(new_settings).is_ok());
    }
}
