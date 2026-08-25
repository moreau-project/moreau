//! # Moreau - CPU Conic Solver
//!
//! Moreau is an interior point numerical solver for convex optimization problems.
//! Based on the Clarabel.rs algorithm with a novel homogeneous embedding.
//!
//! Moreau solves the following problem:
//!
//! $$
//! \begin{array}{rl}
//! \text{minimize} & \frac{1}{2}x^T P x + q^T x\\\\\[2ex\]
//!  \text{subject to} & Ax + s = b \\\\\[1ex\]
//!         & x \in \mathcal{K}_1, \; s \in \mathcal{K}_2
//!  \end{array}
//! $$
//!
//! with decision variables
//! $x \in \mathbb{R}^n$,
//! $s \in \mathbb{R}^m$
//! and data matrices
//! $P=P^\top \succeq 0$,
//! $q \in \mathbb{R}^n$,
//! $A \in \mathbb{R}^{m \times n}$, and
//! $b \in \mathbb{R}^m$.
//! The convex sets $\mathcal{K}_2$ (on the slack $s$) and $\mathcal{K}_1$ (on
//! $x$ directly — "direct-x" cones) are each a composition of convex cones.
//!
//! ## Features
//!
//! * __Versatile__: Solves linear programs (LPs), quadratic programs (QPs), second-order cone
//!   programs (SOCPs) and semidefinite programs (SDPs). Also handles exponential, power cone
//!   and generalized power cone constraints.
//!
//! * __Quadratic objectives__: Handles quadratic objectives without epigraphical reformulation,
//!   significantly faster than standard HSDE-based solvers for QP problems.
//!
//! * __Infeasibility detection__: Uses homogeneous embedding technique.

//Rust hates greek characters
#![allow(confusable_idents)]
#![warn(missing_docs)]

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod algebra;
pub mod io;
pub mod qdldl;
pub mod solver;
pub mod timers;

pub(crate) mod utils;
pub use crate::utils::infbounds::*;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "active-set")]
pub mod active_set_ffi;

#[cfg(feature = "c-api")]
pub mod c_api;

#[allow(unused_macros)]
macro_rules! printbuildenv {
    ($tag:expr) => {
        if let Some(opt) = option_env!(concat!("VERGEN_", $tag)) {
            writeln!(crate::io::stdout(), "{}: {}", $tag, opt).unwrap();
        }
    };
}

/// print detailed build configuration info to stdout
#[allow(clippy::explicit_write)]
pub fn buildinfo() {
    use std::io::Write;

    #[cfg(feature = "buildinfo")]
    {
        printbuildenv!("BUILD_TIMESTAMP");
        printbuildenv!("CARGO_DEBUG");
        printbuildenv!("CARGO_FEATURES");
        printbuildenv!("CARGO_OPT_LEVEL");
        printbuildenv!("CARGO_TARGET_TRIPLE");
        printbuildenv!("RUSTC_CHANNEL");
        printbuildenv!("RUSTC_COMMIT_DATE");
        printbuildenv!("RUSTC_COMMIT_HASH");
        printbuildenv!("RUSTC_HOST_TRIPLE");
        printbuildenv!("RUSTC_LLVM_VERSION");
        printbuildenv!("RUSTC_SEMVER");
        printbuildenv!("SYSINFO_NAME");
        printbuildenv!("SYSINFO_OS_VERSION");
        printbuildenv!("SYSINFO_TOTAL_MEMORY");
        printbuildenv!("SYSINFO_CPU_VENDOR");
        printbuildenv!("SYSINFO_CPU_CORE_COUNT");
        printbuildenv!("SYSINFO_CPU_BRAND");
        printbuildenv!("SYSINFO_CPU_FREQUENCY");
    }
    #[cfg(not(feature = "buildinfo"))]
    writeln!(crate::io::stdout(), "no build info available").unwrap();
}

pub(crate) const _INFINITY_DEFAULT: f64 = 1e20;

#[test]
fn test_buildinfo() {
    buildinfo();
}
