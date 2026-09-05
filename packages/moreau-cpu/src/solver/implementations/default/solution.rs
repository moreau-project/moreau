#![allow(unused_variables)]

use super::*;
use crate::{
    algebra::*,
    solver::core::{traits::Solution, SolverStatus},
};

/// Standard-form solver type implementing the [`Solution`](crate::solver::core::traits::Solution) trait
#[derive(Debug)]
pub struct DefaultSolution<T> {
    /// primal solution
    pub x: Vec<T>,
    /// dual solution (in dual cone)
    pub z: Vec<T>,
    /// vector of slacks (in primal cone)
    pub s: Vec<T>,
    /// direct-x cone duals, flat across all `dir_cones` in spec order
    /// (length = `Σ |J_xc|`; empty when no direct-x cones).
    pub z_x: Vec<T>,
    /// final solver status
    pub status: SolverStatus,
    /// primal objective value
    pub obj_val: T,
    /// dual objective value
    pub obj_val_dual: T,
    /// construction time in seconds (solver structure allocation)
    pub construction_time: f64,
    /// setup time in seconds (matrix values, equilibration)
    pub setup_time: f64,
    /// solve time in seconds (IPM iterations)
    pub solve_time: f64,
    /// number of iterations
    pub iterations: u32,
    /// primal residual
    pub r_prim: T,
    /// dual residual
    pub r_dual: T,
}

impl<T> DefaultSolution<T>
where
    T: FloatT,
{
    /// Create a new `DefaultSolution` object (slack-only, `z_x` empty).
    pub fn new(n: usize, m: usize) -> Self {
        Self::new_with_xn(n, m, 0)
    }

    /// Create a new `DefaultSolution` object including a direct-x dual
    /// vector of length `xn` (sum of direct-x cone dimensions).
    pub fn new_with_xn(n: usize, m: usize, xn: usize) -> Self {
        let x = vec![T::zero(); n];
        let z = vec![T::zero(); m];
        let s = vec![T::zero(); m];
        let z_x = vec![T::zero(); xn];

        Self {
            x,
            z,
            s,
            z_x,
            status: SolverStatus::Unsolved,
            obj_val: T::nan(),
            obj_val_dual: T::nan(),
            construction_time: 0f64,
            setup_time: 0f64,
            solve_time: 0f64,
            iterations: 0,
            r_prim: T::nan(),
            r_dual: T::nan(),
        }
    }
}

impl<T> Solution<T> for DefaultSolution<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type V = DefaultVariables<T>;
    type I = DefaultInfo<T>;
    type SE = DefaultSettings<T>;

    fn post_process(
        &mut self,
        data: &DefaultProblemData<T>,
        variables: &mut DefaultVariables<T>,
        info: &DefaultInfo<T>,
        settings: &DefaultSettings<T>,
    ) {
        self.status = info.status;
        let is_infeasible = info.status.is_infeasible();

        if is_infeasible {
            self.obj_val = T::nan();
            self.obj_val_dual = T::nan();
        } else {
            self.obj_val = info.cost_primal;
            self.obj_val_dual = info.cost_dual;
        }

        self.iterations = info.iterations;
        self.r_prim = info.res_primal;
        self.r_dual = info.res_dual;

        // unscale the variables to get a solution
        // to the internal problem as we solved it
        variables.unscale(data, is_infeasible);

        // unwind the chordal decomp and presolve, in the
        // reverse of the order in which they were applied
        #[cfg(feature = "sdp")]
        let tmp = data
            .chordal_info
            .as_ref()
            .map(|chordal_info| chordal_info.decomp_reverse(variables, &data.cones, settings));
        #[cfg(feature = "sdp")]
        let variables = tmp.as_ref().unwrap_or(variables);

        if let Some(ref presolver) = data.presolver {
            presolver.reverse_presolve(self, variables);
        } else {
            self.x.copy_from(&variables.x);
            self.z.copy_from(&variables.z);
            self.s.copy_from(&variables.s);
        }

        // Direct-x dual `z_x` is unscaled in `Variables::unscale`; copy out
        // into the user-facing `Solution`. The chordal+presolve reverses
        // above don't touch direct-x, so this can stand alone.
        if !variables.z_x.is_empty() {
            if self.z_x.len() != variables.z_x.len() {
                self.z_x.resize(variables.z_x.len(), T::zero());
            }
            self.z_x.copy_from_slice(&variables.z_x);
        }
    }

    fn finalize(&mut self, info: &DefaultInfo<T>) {
        self.construction_time = info.construction_time;
        self.setup_time = info.setup_time;
        self.solve_time = info.solve_time;
    }
}
