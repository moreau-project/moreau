//! Differentiation module for conic optimization.
//!
//! This module provides forward and adjoint (backward) differentiation
//! for conic optimization problems solved by the moreau solver.
//!
//! The differentiation is based on implicit differentiation of the
//! KKT conditions using the Homogeneous Self-Dual Embedding (HSDE).

mod api;
mod cones;
mod kkt;

#[cfg(test)]
mod tests;

pub use cones::*;
// kkt functions are internal, used through differentiate and differentiate_adjoint
// Export GradState for use in solver
pub use kkt::differentiate_adjoint_hsde_with_xcones;
pub use kkt::GradState;

// Re-export the public differentiation API and shared helpers.
pub use api::{differentiate, differentiate_adjoint, BackwardResult, ForwardResult, CONE_TOL};
pub(crate) use api::{resolve_diff_method, supports_smoothed};

// Re-exported so the `api` submodule can reach it via `super::`.
pub(crate) use super::AUTO_SMOOTHED_MU_THRESHOLD;
