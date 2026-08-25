use crate::{algebra::*, solver::core::kktsolvers::HasLinearSolverInfo};

//ldl linear solvers kept in a submodule (not flattened)
pub mod ldlsolvers;

//flatten direct KKT module structure
mod datamaps;
mod directldlkktsolver;
pub(crate) mod kkt_assembly;
use datamaps::*;
pub use directldlkktsolver::*;
use kkt_assembly::*;

pub trait DirectLDLSolverReqs {
    fn required_matrix_shape() -> MatrixTriangle
    where
        Self: Sized;
}
pub trait DirectLDLSolver<T: FloatT>: DirectLDLSolverReqs + HasLinearSolverInfo {
    fn update_values(&mut self, index: &[usize], values: &[T]);
    fn scale_values(&mut self, index: &[usize], scale: T);
    #[allow(dead_code)] //could be removed.
    fn offset_values(&mut self, index: &[usize], offset: T, signs: &[i8]);
    /// Replace the cached diagonal sign vector. Lets sparse-expansion cones
    /// switch the sign of a diagonal slot per iteration (see GenPow rank-9
    /// PD scaling). Length must equal the KKT dimension.
    fn update_dsigns(&mut self, signs: &[i8]);
    fn solve(&mut self, kkt: &CscMatrix<T>, x: &mut [T], b: &mut [T]);
    fn refactor(&mut self, kkt: &CscMatrix<T>) -> bool;
    /// Extract the AMD permutation vector used by this solver.
    /// Returns None if the solver does not use or expose a permutation.
    fn get_perm(&self) -> Option<Vec<usize>> {
        None
    }
}
