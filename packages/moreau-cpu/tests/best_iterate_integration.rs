//! End-to-end tests for the best-iterate safety net wired into the IPM loop.
//!
//! Verifies that a solve which enters the reduced-tolerance zone but then
//! terminates on a non-convergent status (e.g. MaxIterations) is promoted to
//! AlmostSolved with the best observed iterate restored, rather than being
//! returned as the degraded final iterate.

#![allow(non_snake_case)]

use moreau::{algebra::*, solver::*};

/// Tiny LP with strict interior solution so the IPM reaches tight tolerance
/// in just a handful of iterations — the iterates must pass through the
/// reduced-tol zone on their way, triggering `best_saved`.
fn build_lp() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    // minimize x0 + x1  s.t.  x0 + x1 = 1,  x0 >= 0,  x1 >= 0
    let n = 2;
    let P = CscMatrix {
        m: n,
        n,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let q = vec![1.0, 1.0];

    // A encodes: [1 1; -1 0; 0 -1] as CSC
    let A = CscMatrix {
        m: 3,
        n,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 2],
        nzval: vec![1.0, -1.0, 1.0, -1.0],
    };
    let b = vec![1.0, 0.0, 0.0];
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];
    (P, q, A, b, cones)
}

/// A standard solve runs to Solved and snapshots at least one reduced-tol
/// iterate along the way.
#[test]
fn best_saved_on_normal_solve() {
    let (P, q, A, b, cones) = build_lp();
    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!(
        solver.info.best_saved(),
        "IPM passed through reduced-tol zone; best_saved should be true"
    );
    assert!(solver.info.best_iter() > 0);
    assert!(solver.info.best_iter() <= solver.info.iterations);
}

/// A very low iteration cap forces MaxIterations *after* the iterate has
/// entered the reduced-tol zone. The best-iterate fallback must promote the
/// status to AlmostSolved and restore the saved metrics.
#[test]
fn maxiter_promoted_when_reduced_zone_was_reached() {
    let (P, q, A, b, cones) = build_lp();
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    // Tight tol so the solver spends multiple iterations in the reduced zone
    // before reaching tight convergence; low max_iter stops it mid-polish.
    settings.ipm.tol_gap_abs = 1e-14;
    settings.ipm.tol_gap_rel = 1e-14;
    settings.ipm.tol_feas = 1e-14;

    let mut reached_reduced = false;
    // Search for a max_iter that lands in the reduced zone but before tight.
    for cap in 3..=12u32 {
        let mut s = settings.clone();
        s.max_iter = cap;
        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, s).unwrap();
        solver.solve();
        if solver.info.best_saved() && solver.solution.status == SolverStatus::AlmostSolved {
            reached_reduced = true;
            // Restored iterate metrics should satisfy reduced tolerances.
            assert!(solver.info.gap_abs < 5e-5 || solver.info.gap_rel < 5e-5);
            assert!(solver.info.res_primal < 1e-4);
            assert!(solver.info.res_dual < 1e-4);
            break;
        }
    }
    assert!(
        reached_reduced,
        "expected some max_iter cap in 3..=12 to trigger best-iterate restore"
    );
}
