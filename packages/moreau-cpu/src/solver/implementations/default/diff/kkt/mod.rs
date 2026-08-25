//! KKT system construction and solving for differentiation.
//!
//! This module handles the linear systems that arise from implicit
//! differentiation of the KKT conditions.

use super::cones::{
    get_central_path_derivative_sparse, get_cone_derivative_sparse, ConeDerivativeBlock,
};
use super::{BackwardResult, ForwardResult};
use crate::algebra::{
    AsFloatT, CscMatrix, FloatT, MatrixVectorMultiply, ShapedMatrix, SymMatrixMath,
};
use crate::solver::core::cones::SupportedConeT;
use crate::solver::core::kktsolvers::direct::BoxedDirectLDLSolver;
use crate::solver::core::kktsolvers::direct::LDLConfiguration;
use crate::solver::implementations::default::settings::DiffMethod;
use crate::solver::CoreSettings;
use crate::utils::debug::debug_block;

mod adjoint;
mod assembly;
mod differentiate;

#[cfg(test)]
mod tests;

// Re-export the forward/adjoint differentiation entry points used by the
// `diff` module root and the public `api` submodule.
pub use differentiate::{
    differentiate_adjoint_hsde, differentiate_adjoint_hsde_with_xcones,
    differentiate_adjoint_qp_eq, differentiate_hsde, differentiate_qp_eq,
};
// Bring the assembly / adjoint helpers into the `kkt` namespace so the
// submodules (and the test module) can reach them via `use super::*`.
use adjoint::*;
use assembly::*;

/// Regularization for the KKT system
/// QDLDL needs a quasi-definite system, so we use moderate regularization
const REG_GENERAL: f64 = 1e-8;

/// Count the number of expansion variables for the diff KKT system.
/// Returns 2 per SocSparse H block (rank-2 expansion) + 3 per GenPowerSparse (rank-3).
fn count_expansion_vars<T>(H_blocks: &[ConeDerivativeBlock<T>]) -> usize {
    H_blocks
        .iter()
        .map(|b| match b {
            ConeDerivativeBlock::SocSparse { .. } => 2,
            ConeDerivativeBlock::GenPowerSparse { .. } => 3,
            _ => 0,
        })
        .sum()
}

/// Pre-computed state for gradient computation.
///
/// This struct stores the symbolic factorization of the differentiation KKT system,
/// allowing efficient re-use across multiple backward passes. Call `enable_grad()`
/// on the solver to create this state.
///
/// The KKT matrix structure is fixed for a given problem (P, A, cones), but
/// the numerical values change per backward pass based on the solution.
/// By pre-computing the symbolic factorization (AMD ordering, workspace allocation),
/// we avoid this O(nnz) work on each backward call.
pub struct GradState<T: FloatT> {
    /// LDL solver matching the forward pass solver type
    pub(crate) factor: BoxedDirectLDLSolver<T>,
    /// KKT matrix (stored for refactor/solve calls)
    kkt: CscMatrix<T>,
    /// Problem dimensions: n (primal variables)
    n: usize,
    /// Problem dimensions: m (constraints)
    m: usize,
    /// Direct-x cone specs (empty for slack-only).
    x_cones: Vec<crate::solver::core::cones::SupportedXConeT>,
    /// Per-cone indices for direct-x cones (parallel to `x_cones`).
    xcone_indices: Vec<Vec<usize>>,
}

impl<T: FloatT + LDLConfiguration> GradState<T> {
    /// Create a new GradState by performing symbolic factorization.
    ///
    /// This pre-computes the sparsity pattern and AMD ordering for the
    /// differentiation KKT system. The symbolic factorization can be reused
    /// for multiple backward passes.
    ///
    /// Uses the same LDL solver type as the forward pass (based on settings).
    pub fn new(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: &CoreSettings<T>,
    ) -> Result<Self, crate::solver::SolverError> {
        Self::new_with_xcones(P, q, A, b, cones, &[], settings)
    }

    /// Create a new GradState with direct-x cones. The augmented KKT system
    /// includes `xn = sum(|J_xc|)` extra rows + `du_x` columns carrying the
    /// direct-x cone-projection Jacobian `H_x`. Pass an empty `x_cones`
    /// slice for the slack-only path.
    pub fn new_with_xcones(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        x_cones: &[crate::solver::core::cones::SupportedXConeT],
        settings: &CoreSettings<T>,
    ) -> Result<Self, crate::solver::SolverError> {
        let n = q.len();
        let m = b.len();

        let xcone_indices: Vec<Vec<usize>> =
            x_cones.iter().map(|xc| xc.indices().to_vec()).collect();
        let xn: usize = xcone_indices.iter().map(|ix| ix.len()).sum();

        // Build HSDE KKT system with placeholder values.
        let H_blocks = build_placeholder_H_blocks(cones);
        let H_x_blocks = build_placeholder_H_x_blocks::<T>(x_cones);
        let c1 = vec![T::one(); n];
        let c2 = vec![T::one(); m];
        let c3 = T::one();
        let rhs = vec![T::zero(); n + 2 * m + xn + 1];

        let (kkt, _) = build_hsde_augmented_system_sparse_full(
            P,
            A,
            q,
            b,
            &H_blocks,
            cones,
            &xcone_indices,
            &H_x_blocks,
            &c1,
            &c2,
            c3,
            &rhs,
            n,
            m,
            false,
        );

        // Compute diagonal signs for the augmented system:
        // [I J; J' -εI] is quasi-definite: top jdim rows positive, bottom jdim rows negative.
        // Direct-x adds `xn` rows/cols to the base J dim; slack and direct-x
        // sparse expansion vars (SocSparse / GenPowerSparse) extend it further.
        let p_exp_slack = count_expansion_vars(&H_blocks);
        let p_exp_xcone = count_expansion_vars(&H_x_blocks);
        let jdim = n + 2 * m + xn + 1 + p_exp_slack + p_exp_xcone;
        let augdim = 2 * jdim;
        let mut dsigns = vec![1_i8; augdim];
        for i in jdim..augdim {
            dsigns[i] = -1;
        }

        // Use the same solver type as the forward pass
        let (_kktshape, ldl_ctor) = T::get_ldlsolver_config(settings);
        let factor = ldl_ctor(&kkt, &dsigns, settings, None);

        Ok(Self {
            factor,
            kkt,
            n,
            m,
            x_cones: x_cones.to_vec(),
            xcone_indices,
        })
    }

    /// Solve the adjoint system using the pre-computed factorization.
    ///
    /// This updates the numerical values in the factorization and solves,
    /// reusing the pre-computed AMD ordering and workspace.
    ///
    /// Slack-only path: pass empty slices for `z_x_eq` and `dz_x_bar_eq`.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_adjoint(
        &mut self,
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        x: &[T],
        s: &[T],
        z: &[T],
        z_x_eq: &[T],
        tau: T,
        dx_bar: &[T],
        dz_bar: &[T],
        ds_bar: &[T],
        dz_x_bar_eq: &[T],
        diff_method: DiffMethod,
        mu: T,
    ) -> BackwardResult<T> {
        solve_adjoint_hsde_cached(
            self,
            P,
            q,
            A,
            b,
            cones,
            x,
            s,
            z,
            z_x_eq,
            tau,
            dx_bar,
            dz_bar,
            ds_bar,
            dz_x_bar_eq,
            diff_method,
            mu,
        )
    }
}
