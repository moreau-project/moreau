#![allow(non_snake_case)]
//! Infeasibility / unboundedness detection with direct-x cones.
//!
//! With direct-x cones, the HSDE certificates differ from the slack-only
//! case: the primal-infeasibility residual must include the direct-x dual
//! contribution `Σ_J E_J^T z_x` so that the certificate test
//! `‖A^T z − Σ_J E_J^T z_x‖ → 0, b^T z < 0` actually fires. These tests
//! pin the CPU path on small instances where the slack analogue is known
//! to converge to the right status.

use moreau::solver::SupportedXConeT::{NonnegativeXConeT, SecondOrderXConeT};
use moreau::{algebra::*, solver::*};

fn nonneg_pinf_settings() -> DefaultSettings<f64> {
    let mut s = DefaultSettings::default();
    s.ipm.presolve_enable = false;
    s.verbose = false;
    s
}

/// `min 0 s.t. x = -1, x ≥ 0` is primal-infeasible.
/// Direct-x form: `Ax = -1` (zero cone, m=1) plus `x ∈ R+` (direct-x).
#[test]
fn nonneg_direct_x_primal_infeasible() {
    let n = 1;
    let P = CscMatrix::<f64>::zeros((n, n));
    let q = vec![0.0; n];
    let A = CscMatrix::<f64>::identity(n);
    let b = vec![-1.0; 1];

    let cones = vec![ZeroConeT(1)];
    let x_cones = vec![NonnegativeXConeT(vec![0])];

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A, &b, &cones, &x_cones, nonneg_pinf_settings())
            .unwrap();
    solver.solve();

    assert_eq!(
        solver.solution.status,
        SolverStatus::PrimalInfeasible,
        "direct-x nonneg infeasible problem reported {:?} instead of PrimalInfeasible",
        solver.solution.status
    );
}

/// `min -x s.t. x ≥ 0` is unbounded below ⇒ DualInfeasible.
/// Direct-x form: no slack constraints, only `x ∈ R+` direct-x.
#[test]
fn nonneg_direct_x_dual_infeasible() {
    let n = 1;
    let P = CscMatrix::<f64>::zeros((n, n));
    let q = vec![-1.0; n];
    let A = CscMatrix::<f64>::zeros((0, n));
    let b: Vec<f64> = vec![];

    let cones: Vec<SupportedConeT<f64>> = vec![];
    let x_cones = vec![NonnegativeXConeT(vec![0])];

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A, &b, &cones, &x_cones, nonneg_pinf_settings())
            .unwrap();
    solver.solve();

    assert_eq!(
        solver.solution.status,
        SolverStatus::DualInfeasible,
        "direct-x nonneg unbounded problem reported {:?} instead of DualInfeasible",
        solver.solution.status
    );
}

/// Direct-x SOC primal infeasible: `(t, v) ∈ SOC_3` requires `t ≥ ‖v‖`,
/// so forcing `t = -1` via a zero-cone equality has no feasible point.
#[test]
fn soc_direct_x_primal_infeasible() {
    let n = 3;
    let P = CscMatrix::<f64>::zeros((n, n));
    let q = vec![0.0; n];
    // Single equality: x[0] = -1.
    let A = CscMatrix::<f64>::from(&[[1.0, 0.0, 0.0]]);
    let b = vec![-1.0];

    let cones = vec![ZeroConeT(1)];
    let x_cones = vec![SecondOrderXConeT(vec![0, 1, 2])];

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A, &b, &cones, &x_cones, nonneg_pinf_settings())
            .unwrap();
    solver.solve();

    assert_eq!(
        solver.solution.status,
        SolverStatus::PrimalInfeasible,
        "direct-x SOC infeasible problem reported {:?} instead of PrimalInfeasible",
        solver.solution.status
    );
}

/// Direct-x SOC dual infeasible: `min -t s.t. (t, v) ∈ SOC_3` is unbounded
/// (set `v = 0`, `t → ∞`).
#[test]
fn soc_direct_x_dual_infeasible() {
    let n = 3;
    let P = CscMatrix::<f64>::zeros((n, n));
    let q = vec![-1.0, 0.0, 0.0];
    let A = CscMatrix::<f64>::zeros((0, n));
    let b: Vec<f64> = vec![];

    let cones: Vec<SupportedConeT<f64>> = vec![];
    let x_cones = vec![SecondOrderXConeT(vec![0, 1, 2])];

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A, &b, &cones, &x_cones, nonneg_pinf_settings())
            .unwrap();
    solver.solve();

    assert_eq!(
        solver.solution.status,
        SolverStatus::DualInfeasible,
        "direct-x SOC unbounded problem reported {:?} instead of DualInfeasible",
        solver.solution.status
    );
}

/// Mixed: direct-x nonneg + slack equality forcing infeasibility.
/// `x ∈ R+², a^T x = -2 (a > 0)` ⇒ infeasible.
#[test]
fn nonneg_direct_x_mixed_primal_infeasible() {
    let n = 2;
    let P = CscMatrix::<f64>::zeros((n, n));
    let q = vec![1.0, 1.0];
    let A = CscMatrix::<f64>::from(&[[1.0, 1.0]]);
    let b = vec![-2.0];

    let cones = vec![ZeroConeT(1)];
    let x_cones = vec![NonnegativeXConeT(vec![0, 1])];

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A, &b, &cones, &x_cones, nonneg_pinf_settings())
            .unwrap();
    solver.solve();

    assert_eq!(
        solver.solution.status,
        SolverStatus::PrimalInfeasible,
        "mixed direct-x infeasible problem reported {:?} instead of PrimalInfeasible",
        solver.solution.status
    );
}

/// Direct-x PSD primal infeasible: a 2×2 PSD matrix's (1,1) entry must be
/// nonneg, so forcing svec(X)[0] = -1 has no feasible point.
#[cfg(feature = "sdp")]
#[test]
fn psd_direct_x_primal_infeasible() {
    use moreau::solver::SupportedXConeT::PSDTriangleXConeT;

    // X is 2×2 ⇒ svec length 3 (a, b√2, c).
    let n = 3;
    let k = 2;
    let P = CscMatrix::<f64>::zeros((n, n));
    let q = vec![0.0; n];
    // Equality forcing a = -1 (the (1,1) diagonal of X).
    let A = CscMatrix::<f64>::from(&[[1.0, 0.0, 0.0]]);
    let b = vec![-1.0];

    let cones = vec![ZeroConeT(1)];
    let x_cones = vec![PSDTriangleXConeT(vec![0, 1, 2], k)];

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A, &b, &cones, &x_cones, nonneg_pinf_settings())
            .unwrap();
    solver.solve();

    assert_eq!(
        solver.solution.status,
        SolverStatus::PrimalInfeasible,
        "direct-x PSD infeasible problem reported {:?} instead of PrimalInfeasible",
        solver.solution.status
    );
}
