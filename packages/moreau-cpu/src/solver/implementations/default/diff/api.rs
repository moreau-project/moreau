//! Forward and adjoint differentiation entry points.
//!
//! This submodule holds the public `differentiate` /
//! `differentiate_adjoint` API and the cone-validation helpers.

use super::super::settings::DiffMethod;
use super::kkt;
use crate::algebra::{AsFloatT, CscMatrix, FloatT};
use crate::solver::core::cones::SupportedConeT;

/// Tolerance for cone boundary detection
pub const CONE_TOL: f64 = 1e-8;

/// Result type for forward differentiation
#[derive(Debug, Clone)]
pub struct ForwardResult<T> {
    /// Derivative of primal solution x
    pub dx: Vec<T>,
    /// Derivative of dual solution z (y in some formulations)
    pub dz: Vec<T>,
    /// Derivative of slack variables s
    pub ds: Vec<T>,
}

/// Result type for backward (adjoint) differentiation
#[derive(Debug, Clone)]
pub struct BackwardResult<T> {
    /// Gradient w.r.t. objective matrix P (same sparsity pattern)
    pub dP: CscMatrix<T>,
    /// Gradient w.r.t. objective vector q
    pub dq: Vec<T>,
    /// Gradient w.r.t. constraint matrix A (same sparsity pattern)
    pub dA: CscMatrix<T>,
    /// Gradient w.r.t. constraint vector b
    pub db: Vec<T>,
    /// Debug: smoothed iterate x (equilibrated, τ-normalized)
    #[cfg(debug_assertions)]
    pub debug_smoothing_x: Vec<T>,
    /// Debug: smoothed iterate z (equilibrated, τ-normalized)
    #[cfg(debug_assertions)]
    pub debug_smoothing_z: Vec<T>,
    /// Debug: smoothed iterate s (equilibrated, τ-normalized)
    #[cfg(debug_assertions)]
    pub debug_smoothing_s: Vec<T>,
}

/// Forward differentiation: compute solution derivatives given problem perturbations.
///
/// Given a solved conic optimization problem and perturbations to the problem data,
/// computes the derivatives of the solution (x, z, s) with respect to those perturbations.
///
/// # Mathematical Background
///
/// The conic problem is:
/// ```text
/// minimize    (1/2)x'Px + q'x
/// subject to  Ax + s = b
///             x ∈ K1,  s ∈ K2
/// ```
/// K2 constrains the slack s; K1 constrains x directly (direct-x cones).
///
/// At optimality, the KKT conditions hold. By differentiating these conditions
/// with respect to problem parameters, we can compute how the solution changes
/// when the problem data is perturbed.
///
/// # Arguments
/// * `P` - Objective matrix (n x n, symmetric positive semidefinite)
/// * `q` - Objective vector (n)
/// * `A` - Constraint matrix (m x n)
/// * `b` - Constraint vector (m)
/// * `cones` - Cone constraints
/// * `x` - Primal solution
/// * `s` - Slack variables (in primal cone)
/// * `z` - Dual solution (in dual cone)
/// * `tau` - Homogenization scalar (typically 1.0 at solution)
/// * `dP` - Perturbation to P
/// * `dq` - Perturbation to q
/// * `dA` - Perturbation to A
/// * `db` - Perturbation to b
///
/// # Returns
/// * `ForwardResult` containing (dx, dz, ds)
pub fn differentiate<T: FloatT>(
    P: &CscMatrix<T>,
    q: &[T],
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    x: &[T],
    s: &[T],
    z: &[T],
    tau: T,
    dP: &CscMatrix<T>,
    dq: &[T],
    dA: &CscMatrix<T>,
    db: &[T],
    diff_method: DiffMethod,
    mu: T,
) -> ForwardResult<T> {
    // Check if we have only zero cones (pure equality constraints)
    if is_all_zero_cones(cones) {
        return kkt::differentiate_qp_eq(P, q, A, b, x, z, dP, dq, dA, db);
    }

    // Validate cones are supported for differentiation
    validate_diff_cones(cones);

    // General case: use HSDE differentiation
    kkt::differentiate_hsde(
        P,
        q,
        A,
        b,
        cones,
        x,
        s,
        z,
        tau,
        dP,
        dq,
        dA,
        db,
        diff_method,
        mu,
    )
}

/// Adjoint (backward) differentiation: compute parameter gradients from output gradients.
///
/// Given upstream gradients with respect to the solution (dx_bar, dz_bar, ds_bar),
/// computes the gradients with respect to the problem data (P, q, A, b).
///
/// This is the key operation for training neural networks that include
/// conic optimization as a layer.
///
/// # Mathematical Background
///
/// Using the adjoint method (reverse-mode automatic differentiation),
/// we solve a linear system involving the transpose of the KKT matrix
/// to propagate gradients backward through the optimization.
///
/// # Arguments
/// * `P` - Objective matrix
/// * `q` - Objective vector
/// * `A` - Constraint matrix
/// * `b` - Constraint vector
/// * `cones` - Cone constraints
/// * `x` - Primal solution
/// * `s` - Slack variables
/// * `z` - Dual solution
/// * `tau` - Homogenization scalar
/// * `dx_bar` - Upstream gradient w.r.t. x
/// * `dz_bar` - Upstream gradient w.r.t. z
/// * `ds_bar` - Upstream gradient w.r.t. s
///
/// # Returns
/// * `BackwardResult` containing (dP, dq, dA, db)
pub fn differentiate_adjoint<T: FloatT>(
    P: &CscMatrix<T>,
    q: &[T],
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    x: &[T],
    s: &[T],
    z: &[T],
    tau: T,
    dx_bar: &[T],
    dz_bar: &[T],
    ds_bar: &[T],
    diff_method: DiffMethod,
    mu: T,
) -> BackwardResult<T> {
    // Check if we have only zero cones (pure equality constraints)
    if is_all_zero_cones(cones) {
        return kkt::differentiate_adjoint_qp_eq(P, q, A, b, x, z, dx_bar, dz_bar, ds_bar);
    }

    // For any problem with inequality constraints (nonneg cones), we must use HSDE
    // to properly handle the complementarity conditions and cone projections.
    // The QP equality path doesn't account for active/inactive constraints.

    // Validate cones are supported for differentiation
    validate_diff_cones(cones);

    // General case: use HSDE adjoint differentiation
    kkt::differentiate_adjoint_hsde(
        P,
        q,
        A,
        b,
        cones,
        x,
        s,
        z,
        tau,
        dx_bar,
        dz_bar,
        ds_bar,
        diff_method,
        mu,
    )
}

/// Check if all cones support smoothed differentiation (zero + nonneg + SOC).
pub(crate) fn supports_smoothed<T: FloatT>(cones: &[SupportedConeT<T>]) -> bool {
    cones.iter().all(|c| {
        matches!(
            c,
            SupportedConeT::ZeroConeT(_)
                | SupportedConeT::NonnegativeConeT(_)
                | SupportedConeT::SecondOrderConeT(_)
        )
    })
}

/// Resolve `DiffMethod::Auto` into `Exact` or `Smoothed`.
///
/// Uses `Smoothed` for supported cones with μ above the threshold.
pub(crate) fn resolve_diff_method<T: FloatT>(
    diff_method: DiffMethod,
    cones: &[SupportedConeT<T>],
    mu: T,
) -> DiffMethod {
    match diff_method {
        DiffMethod::Auto => {
            if supports_smoothed(cones) && mu >= super::AUTO_SMOOTHED_MU_THRESHOLD.as_T() {
                DiffMethod::Smoothed
            } else {
                DiffMethod::Exact
            }
        }
        other => other,
    }
}

/// Check if all cones are zero cones (equality constraints)
pub(super) fn is_all_zero_cones<T: FloatT>(cones: &[SupportedConeT<T>]) -> bool {
    cones
        .iter()
        .all(|c| matches!(c, SupportedConeT::ZeroConeT(_)))
}

/// Validate cones are supported for differentiation.
///
/// # Panics
/// Panics if any cone is not supported:
/// - PSDTriangleCone: not implemented
fn validate_diff_cones<T: FloatT>(cones: &[SupportedConeT<T>]) {
    for cone in cones {
        match cone {
            #[cfg(feature = "sdp")]
            SupportedConeT::PSDTriangleConeT(_) => {
                // PSD cone differentiation is supported
            }
            _ => {}
        }
    }
}
