#![allow(non_snake_case)]
use super::{
    cones::{CompositeCone, CompositeXCone},
    CoreSettings,
};
use crate::algebra::*;

pub mod direct;

pub trait KKTSolver<T: FloatT>: HasLinearSolverInfo {
    /// Refresh the KKT numeric values from current cone scalings and
    /// refactor. The direct-x `dir_cones` contribute additive Hs blocks to
    /// the (1,1) block; pass an empty `CompositeXCone` when the problem
    /// has no direct-x constraints.
    fn update(
        &mut self,
        cones: &mut CompositeCone<T>,
        dir_cones: &mut CompositeXCone<T>,
        settings: &CoreSettings<T>,
    ) -> bool;
    fn setrhs(&mut self, x: &[T], z: &[T]);
    fn solve(
        &mut self,
        x: Option<&mut [T]>,
        z: Option<&mut [T]>,
        settings: &CoreSettings<T>,
    ) -> bool;
    fn update_P(&mut self, P: &CscMatrix<T>);
    fn update_A(&mut self, A: &CscMatrix<T>);
    /// Extract the AMD permutation vector from the underlying linear solver.
    fn get_amd_perm(&self) -> Option<Vec<usize>> {
        None
    }
}

pub trait HasLinearSolverInfo {
    fn linear_solver_info(&self) -> LinearSolverInfo;
}
#[repr(C)]
#[derive(Debug, Default, Clone)]
/// Linear subsolver information.
///
pub struct LinearSolverInfo {
    /// Name of the linear solver that was used
    pub name: String,
    /// Number of threads used by solver
    pub threads: usize,
    /// Whether the solver used a direct factorisation method
    pub direct: bool,
    /// Number of nonzeros in the linear system
    pub nnzA: usize, // nnz in A for A = LDL^T
    /// Number of nonzeros in the factored system
    pub nnzL: usize, // nnz in L for A = LDL^T
}
