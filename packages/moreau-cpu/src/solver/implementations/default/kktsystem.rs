use super::*;
use crate::solver::core::{
    cones::{CompositeCone, CompositeXCone, Cone},
    kktsolvers::{direct::*, *},
    traits::{KKTSystem, Settings},
    StepDirection,
};
use crate::utils::debug::debug_block;

use crate::algebra::*;

// We require Send/Sync here to allow pyo3 builds to share
// solver objects between threads.

type BoxedKKTSolver<T> = Box<dyn KKTSolver<T> + Send + Sync>;

/// Standard-form solver type implementing the [`KKTSystem`](crate::solver::core::traits::KKTSystem) trait
pub struct DefaultKKTSystem<T: FloatT> {
    kktsolver: BoxedKKTSolver<T>,

    // Direct-x cone state. Holds NT-scaling (`w` for nonneg, etc.) across
    // IPM iterations, paralleling the slack `CompositeCone`. Empty when the
    // problem has no direct-x cones — in that case it costs nothing.
    x_cones: CompositeXCone<T>,

    // solution vector for constant part of KKT solves
    x1: Vec<T>,
    z1: Vec<T>,

    // solution vector for general KKT solves
    x2: Vec<T>,
    z2: Vec<T>,

    // work vectors for assembling/dissambling vectors
    workx: Vec<T>,
    workz: Vec<T>,
    work_conic: Vec<T>,

    // Per-cone direct-x scratch, sized to `max(x_cones[i].indices.len())`.
    // Allocated once and reused across IPM iterations to satisfy the
    // "zero allocations inside iteration loops" rule. All consumers
    // truncate-or-resize to the current cone's k via `[..k]` slicing or
    // `resize`; never reallocate.
    xcone_scratch_c_j: Vec<T>,
    xcone_scratch_work: Vec<T>,
    xcone_scratch_gather: Vec<T>,
    xcone_scratch_hs_dx: Vec<T>,
    xcone_scratch_work2: Vec<T>,
}

impl<T> DefaultKKTSystem<T>
where
    T: FloatT,
{
    /// Create a new KKT system solver
    pub fn new(
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> Self {
        Self::new_with_perm(data, cones, settings, None)
    }

    /// Create a new KKT system solver with an optional cached AMD permutation
    pub fn new_with_perm(
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        settings: &DefaultSettings<T>,
        perm: Option<Vec<usize>>,
    ) -> Self {
        let x_cones = data.composite_x_cones();
        Self::new_with_perm_and_xcones(data, cones, x_cones, settings, perm)
    }

    /// Create a new KKT system solver including direct-x cones and an
    /// optional cached AMD permutation. Ownership of `x_cones` moves to
    /// the KKT system so NT-scaling state persists across IPM iterations.
    pub fn new_with_perm_and_xcones(
        data: &DefaultProblemData<T>,
        cones: &CompositeCone<T>,
        x_cones: CompositeXCone<T>,
        settings: &DefaultSettings<T>,
        perm: Option<Vec<usize>>,
    ) -> Self {
        let (m, n) = (data.m, data.n);

        //here we allow scope for different KKT solvers, e.g.
        //direct vs indirect, different QR based direct methods
        //etc.
        let kktsolver: BoxedKKTSolver<T> = if !settings.ipm.direct_kkt_solver {
            panic!("Indirect and other solve strategies not yet supported.");
        } else {
            // NOTE: Riccati (block-tridiagonal) detection is intentionally
            // disabled on CPU.  Benchmarks show that faer's sparse Cholesky
            // (which implicitly exploits banded structure via fill-reducing
            // orderings) outperforms the dense-block Riccati recursion on
            // CPU for all tested MPC problem sizes.  The Riccati path is
            // valuable on GPU where batched dense BLAS parallelises across
            // problems, but on CPU each problem runs on its own thread and
            // sparse factorisation wins.
            Box::new(DirectLDLKKTSolver::<T>::new_with_perm_and_xcones(
                &data.P,
                &data.A,
                cones,
                &x_cones,
                m,
                n,
                settings.core(),
                perm,
            ))
        };

        //the LHS constant part of the reduced solve
        let x1 = vec![T::zero(); n];
        let z1 = vec![T::zero(); m];

        //the LHS for other solves
        let x2 = vec![T::zero(); n];
        let z2 = vec![T::zero(); m];

        //workspace compatible with (x,z)
        let workx = vec![T::zero(); n];
        let workz = vec![T::zero(); m];

        //additional conic workspace vector compatible with s and z
        let work_conic = vec![T::zero(); m];

        // Per-cone direct-x scratch sized to the largest cone's k. Empty
        // when there are no direct-x cones.
        let max_xcone_k = x_cones.iter().map(|e| e.indices.len()).max().unwrap_or(0);

        Self {
            kktsolver,
            x_cones,
            x1,
            z1,
            x2,
            z2,
            workx,
            workz,
            work_conic,
            xcone_scratch_c_j: vec![T::zero(); max_xcone_k],
            xcone_scratch_work: vec![T::zero(); max_xcone_k],
            xcone_scratch_gather: vec![T::zero(); max_xcone_k],
            xcone_scratch_hs_dx: vec![T::zero(); max_xcone_k],
            xcone_scratch_work2: vec![T::zero(); max_xcone_k],
        }
    }
}

impl<T> HasLinearSolverInfo for DefaultKKTSystem<T>
where
    T: FloatT,
{
    fn linear_solver_info(&self) -> LinearSolverInfo {
        self.kktsolver.linear_solver_info()
    }
}

impl<T> KKTSystem<T> for DefaultKKTSystem<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type C = CompositeCone<T>;
    type SE = DefaultSettings<T>;

    fn x_cones_mut(&mut self) -> &mut CompositeXCone<T> {
        &mut self.x_cones
    }

    fn x_cones_ref(&self) -> &CompositeXCone<T> {
        &self.x_cones
    }

    fn update(
        &mut self,
        data: &DefaultProblemData<T>,
        cones: &mut CompositeCone<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        // update the linear solver with new cones
        let is_success = self
            .kktsolver
            .update(cones, &mut self.x_cones, settings.core());

        if !is_success {
            return is_success;
        }

        // calculate KKT solution for constant terms
        self.solve_constant_rhs(data, settings.core())
    }

    fn solve(
        &mut self,
        lhs: &mut DefaultVariables<T>,
        rhs: &DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        variables: &DefaultVariables<T>,
        cones: &mut CompositeCone<T>,
        step_direction: StepDirection,
        settings: &DefaultSettings<T>,
    ) -> bool {
        let (x1, z1) = (&mut self.x1, &mut self.z1);
        let (x2, z2) = (&self.x2, &self.z2); //from constant solve, so not mut
        let (workx, workz) = (&mut self.workx, &mut self.workz);

        // solve for (x1,z1)
        // -----------
        workx.copy_from(&rhs.x);

        // compute the vector c in the step equation HₛΔz + Δs = -c,
        // with shortcut in affine case
        let Δs_const_term = &mut self.work_conic;

        match step_direction {
            StepDirection::Affine => {
                Δs_const_term.copy_from(&variables.s);
            }
            StepDirection::Combined | StepDirection::Centering => {
                cones.Δs_from_Δz_offset(Δs_const_term, &rhs.s, &mut lhs.z, &variables.z);
            }
        }

        workz.waxpby(T::one(), Δs_const_term, -T::one(), &rhs.z);

        // Direct-x workx adjustment. The reduced x-gradient row is
        //   (P + Σ_J E_J' Hs_inv_J E_J) Δx + A'Δz = rx - Σ_J E_J' c_J
        // where Hs_inv_J is already baked into the KKT (1,1) block by
        // `refresh_hx_blocks` via the swapped `update_scaling` convention,
        // and c_J is the direct-x analog of `Δs_const_term` (with `z_J`
        // playing the role of cone's primal slack "s"). For Affine:
        // `c_J = variables.z_x[rng]`; for Combined/Centering: `c_J =
        // cone.Δs_from_Δz_offset(rhs.z_x[rng], x[J])` (cone's z = x[J]).
        if !self.x_cones.is_empty() {
            let x_cones = &mut self.x_cones;
            let c_j_buf = &mut self.xcone_scratch_c_j;
            let work_buf = &mut self.xcone_scratch_work;
            let gather_buf = &mut self.xcone_scratch_gather;
            let mut off = 0usize;
            for entry in x_cones.iter_mut() {
                let k = entry.indices.len();
                let c_j = &mut c_j_buf[..k];
                match step_direction {
                    StepDirection::Affine => {
                        entry
                            .cone
                            .direct_x_affine_offset(c_j, &variables.z_x[off..off + k]);
                    }
                    StepDirection::Combined | StepDirection::Centering => {
                        let x_gather = &mut gather_buf[..k];
                        for (i, &idx) in entry.indices.iter().enumerate() {
                            x_gather[i] = variables.x[idx];
                        }
                        let work = &mut work_buf[..k];
                        entry.cone.direct_x_combined_offset(
                            c_j,
                            &rhs.z_x[off..off + k],
                            work,
                            x_gather,
                        );
                    }
                }
                for (i, &idx) in entry.indices.iter().enumerate() {
                    workx[idx] -= c_j[i];
                }
                off += k;
            }
        }

        debug_block! {
            eprintln!("CPU affine RHS x: {:?}", workx);
            eprintln!("CPU affine RHS z (workz): {:?}", workz);
        }

        // ---------------------------------------------------
        // this solves the variable part of reduced KKT system
        self.kktsolver.setrhs(workx, workz);
        let is_success = self.kktsolver.solve(Some(x1), Some(z1), settings.core());
        if !is_success {
            return false;
        }

        debug_block! {
            eprintln!("x1: {:?}", x1);
            eprintln!("z1: {:?}", z1);
            eprintln!("x2: {:?}", x2);
            eprintln!("z2: {:?}", z2);
        }

        if step_direction == StepDirection::Centering {
            // Centering mode: skip the HSDE Δτ correction.
            // Use (x1, z1) directly as the step direction, which is
            // the reduced KKT solution without homogeneous coupling.
            lhs.τ = T::zero();
            lhs.κ = T::zero();

            lhs.x.copy_from(x1);
            lhs.z.copy_from(z1);

            // solve for Δs from Δz
            cones.mul_Hs(&mut lhs.s, &lhs.z, workz);
            lhs.s.axpby(-T::one(), Δs_const_term, -T::one());

            // Direct-x Δz_J recovery. The complementarity linearization
            //   Hs_inv_J · Δx[J] + Δz_J = -c_J
            // gives Δz_J = -Hs_inv_J (E_J Δx) - c_J, mirroring the slack
            // recovery `Δs = -Hs Δz - c_slack` with primal/dual swapped.
            recover_direct_x_dual(
                &mut self.x_cones,
                &mut lhs.z_x,
                &lhs.x,
                rhs,
                variables,
                step_direction,
                &mut self.xcone_scratch_c_j,
                &mut self.xcone_scratch_work,
                &mut self.xcone_scratch_gather,
                &mut self.xcone_scratch_hs_dx,
                &mut self.xcone_scratch_work2,
            );
        } else {
            // solve for Δτ.
            // -----------
            // Numerator first
            let ξ = workx;
            ξ.axpby(T::recip(variables.τ), &variables.x, T::zero());

            let two: T = (2.).as_T();
            let tau_num = rhs.τ - rhs.κ / variables.τ
                + data.q.dot(x1)
                + data.b.dot(z1)
                + two * data.P.sym_up().quad_form(ξ, x1);

            // offset ξ for the quadratic form in the denominator
            let ξ_minus_x2 = ξ; //alias to ξ, same as workx
            ξ_minus_x2.axpby(-T::one(), x2, T::one());

            let mut tau_den = variables.κ / variables.τ - data.q.dot(x2) - data.b.dot(z2);
            tau_den += data.P.sym_up().quad_form(ξ_minus_x2, ξ_minus_x2)
                - data.P.sym_up().quad_form(x2, x2);

            // solve for (Δx,Δz)
            // -----------
            lhs.τ = tau_num / tau_den;

            lhs.x.waxpby(T::one(), x1, lhs.τ, x2);
            lhs.z.waxpby(T::one(), z1, lhs.τ, z2);

            // solve for Δs
            // -------------
            //  compute the linear term HₛΔz, where Hs = WᵀW for symmetric
            //  cones and Hs = μH(z) for asymmetric cones
            cones.mul_Hs(&mut lhs.s, &lhs.z, workz);
            lhs.s.axpby(-T::one(), Δs_const_term, -T::one()); // lhs.s = -(lhs.s+Δs_const_term);

            // solve for Δκ
            // --------------
            lhs.κ = -(rhs.κ + variables.κ * lhs.τ) / variables.τ;

            // Direct-x Δz_J recovery — mirror of slack `Δs = -Hs Δz - c`
            // with primal/dual swapped.
            recover_direct_x_dual(
                &mut self.x_cones,
                &mut lhs.z_x,
                &lhs.x,
                rhs,
                variables,
                step_direction,
                &mut self.xcone_scratch_c_j,
                &mut self.xcone_scratch_work,
                &mut self.xcone_scratch_gather,
                &mut self.xcone_scratch_hs_dx,
                &mut self.xcone_scratch_work2,
            );
        }

        // we don't check the validity of anything
        // after the KKT solve, so just return is_success
        // without further validation
        is_success
    }

    fn solve_initial_point(
        &mut self,
        variables: &mut DefaultVariables<T>,
        data: &DefaultProblemData<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        let is_success;

        if data.P.nnz() == 0 {
            // LP initialization
            // solve with [0;b] as a RHS to get (x,-s) initializers
            // zero out any sparse cone variables at end
            self.workx.fill(T::zero());
            self.workz.copy_from(&data.b);
            self.kktsolver.setrhs(&self.workx, &self.workz);
            is_success = self.kktsolver.solve(
                Some(&mut variables.x),
                Some(&mut variables.s),
                settings.core(),
            );

            variables.s.negate();

            if !is_success {
                return is_success;
            }

            // solve with [-q;0] as a RHS to get z initializer
            // zero out any sparse cone variables at end
            self.workx.axpby(-T::one(), &data.q, T::zero());
            self.workz.fill(T::zero());

            self.kktsolver.setrhs(&self.workx, &self.workz);
            let mut xtmp = vec![T::zero(); data.n];
            let is_success =
                self.kktsolver
                    .solve(Some(&mut xtmp), Some(&mut variables.z), settings.core());
            return is_success;
        } else {
            //QP initialization
            self.workx.scalarop_from(|q| -q, &data.q);
            self.workz.copy_from(&data.b);

            self.kktsolver.setrhs(&self.workx, &self.workz);
            is_success = self.kktsolver.solve(
                Some(&mut variables.x),
                Some(&mut variables.z),
                settings.core(),
            );

            variables.s.scalarop_from(|z| -z, &variables.z);
        }

        is_success
    }
}

/// Recover the direct-x dual step `Δz_J` after the reduced KKT solve.
///
/// For each direct-x cone the linearized complementarity is
/// `Hs_inv_J Δx[J] + Δz_J = -c_J`, which rearranges to
/// `Δz_J = -Hs_inv_J (E_J Δx) - c_J`. `cone.mul_Hs` applied to gathered
/// `Δx[J]` gives `Hs_inv_J Δx[J]` because `update_scaling` was called
/// with swapped (z_J, x[J]) args (so cone's Hs = z/x = Hs_inv).
#[allow(clippy::too_many_arguments)]
fn recover_direct_x_dual<T: FloatT>(
    x_cones: &mut CompositeXCone<T>,
    lhs_z_x: &mut [T],
    lhs_x: &[T],
    rhs: &DefaultVariables<T>,
    variables: &DefaultVariables<T>,
    step_direction: StepDirection,
    // Preallocated scratch sized to max(x_cones[i].indices.len()); sliced
    // to [..k] per cone. See DefaultKKTSystem::xcone_scratch_*.
    c_j_buf: &mut [T],
    work_buf: &mut [T],
    gather_buf: &mut [T],
    hs_dx_buf: &mut [T],
    work2_buf: &mut [T],
) {
    if x_cones.is_empty() {
        return;
    }
    let mut off = 0usize;
    for entry in x_cones.iter_mut() {
        let k = entry.indices.len();
        let dx_gather = &mut gather_buf[..k];
        for (i, &idx) in entry.indices.iter().enumerate() {
            dx_gather[i] = lhs_x[idx];
        }
        let hs_dx = &mut hs_dx_buf[..k];
        let work = &mut work_buf[..k];
        entry.cone.direct_x_mul_Hs(hs_dx, dx_gather, work);

        // c_J per step_direction
        let c_j = &mut c_j_buf[..k];
        match step_direction {
            StepDirection::Affine => {
                entry
                    .cone
                    .direct_x_affine_offset(c_j, &variables.z_x[off..off + k]);
            }
            StepDirection::Combined | StepDirection::Centering => {
                // Reuse `dx_gather` slot for `x_gather` — these uses don't
                // overlap (dx_gather is consumed by mul_Hs above before this).
                let x_gather = dx_gather;
                for (i, &idx) in entry.indices.iter().enumerate() {
                    x_gather[i] = variables.x[idx];
                }
                let work2 = &mut work2_buf[..k];
                entry
                    .cone
                    .direct_x_combined_offset(c_j, &rhs.z_x[off..off + k], work2, x_gather);
            }
        }

        // lhs.z_x[rng] = -Hs · Δx - c_j  (mirror of slack
        // `Δs = -Hs·Δz - c_slack`)
        for i in 0..k {
            lhs_z_x[off + i] = -hs_dx[i] - c_j[i];
        }
        off += k;
    }
}

impl<T> DefaultKKTSystem<T>
where
    T: FloatT,
{
    fn solve_constant_rhs(
        &mut self,
        data: &DefaultProblemData<T>,
        settings: &DefaultSettings<T>,
    ) -> bool {
        self.workx.axpby(-T::one(), &data.q, T::zero()); //workx .= -q
        debug_block! {
            eprintln!("CPU constant RHS x: {:?}", self.workx);
            eprintln!("CPU constant RHS z (b): {:?}", data.b);
        }
        self.kktsolver.setrhs(&self.workx, &data.b);
        let is_success =
            self.kktsolver
                .solve(Some(&mut self.x2), Some(&mut self.z2), settings.core());

        is_success
    }

    pub(crate) fn update_P(&mut self, P: &CscMatrix<T>) {
        self.kktsolver.update_P(P);
    }

    pub(crate) fn update_A(&mut self, A: &CscMatrix<T>) {
        self.kktsolver.update_A(A);
    }

    /// Extract the AMD permutation from the underlying KKT solver.
    pub(crate) fn get_amd_perm(&self) -> Option<Vec<usize>> {
        self.kktsolver.get_amd_perm()
    }
}
