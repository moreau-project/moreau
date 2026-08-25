#![allow(non_snake_case)]
//! Cached `GradState` direct-x equivalence: the cached symbolic factorization
//! must produce the same gradients as the per-call (non-cached) IFT-direct
//! path for nonneg, SOC, and PSD direct-x cones.

use moreau::solver::implementations::default::diff::GradState;
use moreau::solver::traits::Settings;
use moreau::{algebra::*, solver::*};

fn build_soc_active_problem() -> (CscMatrix<f64>, Vec<f64>, usize) {
    // 3D SOC active-boundary QP: q = (1, 2, 0) pulls x[0] negative so the
    // cone constraint binds.
    let n = 3usize;
    let P = CscMatrix::<f64>::new(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0, 1.0, 1.0],
    );
    let q = vec![1.0, 2.0, 0.0];
    (P, q, n)
}

fn run_direct_x<F: Fn() -> Vec<SupportedXConeT>>(
    P: &CscMatrix<f64>,
    q: &[f64],
    n: usize,
    make_xcones: F,
    cached: bool,
) -> BackwardResult<f64> {
    let A = CscMatrix::<f64>::zeros((0, n));
    let b: Vec<f64> = vec![];
    let cones: Vec<SupportedConeT<f64>> = vec![];
    let x_cones = make_xcones();

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;

    let mut solver =
        DefaultSolver::new_with_xcones(P, q, &A, &b, &cones, &x_cones, settings.clone()).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let dx = vec![1.0; n];
    let dz: Vec<f64> = vec![];
    let ds: Vec<f64> = vec![];

    if cached {
        let mut grad_state =
            GradState::<f64>::new_with_xcones(P, q, &A, &b, &cones, &x_cones, settings.core())
                .unwrap();
        solver
            .backward_batch(&dx, &ds, &dz, Some(&mut grad_state), DiffMethod::Exact)
            .unwrap()
    } else {
        solver
            .backward_batch(&dx, &ds, &dz, None, DiffMethod::Exact)
            .unwrap()
    }
}

fn assert_results_close(a: &BackwardResult<f64>, b: &BackwardResult<f64>, tol: f64) {
    assert_eq!(a.dq.len(), b.dq.len());
    for i in 0..a.dq.len() {
        assert!(
            (a.dq[i] - b.dq[i]).abs() < tol,
            "dq[{}] cached={} vs non-cached={}",
            i,
            b.dq[i],
            a.dq[i]
        );
    }
    assert_eq!(a.dP.nzval.len(), b.dP.nzval.len());
    for k in 0..a.dP.nzval.len() {
        assert!(
            (a.dP.nzval[k] - b.dP.nzval[k]).abs() < tol,
            "dP[{}] cached={} vs non-cached={}",
            k,
            b.dP.nzval[k],
            a.dP.nzval[k]
        );
    }
}

#[test]
fn test_cached_grad_state_nonneg_direct_x() {
    // Nonneg has diagonal H_x — exercises the Diagonal branch.
    let n = 3usize;
    let P = CscMatrix::<f64>::new(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0, 1.0, 1.0],
    );
    let q = vec![-0.5, -0.5, 1.0]; // first two unconstrained; third pushed against cone
    let make = || vec![SupportedXConeT::NonnegativeXConeT((0..n).collect())];
    let cached = run_direct_x(&P, &q, n, make, true);
    let non_cached = run_direct_x(&P, &q, n, make, false);
    assert_results_close(&cached, &non_cached, 1e-9);
}

#[test]
fn test_cached_grad_state_soc_direct_x() {
    // SOC at active boundary — exercises the dense H_x branch via SocSparse
    // → to_dense() expansion in the augmented system builder.
    let (P, q, n) = build_soc_active_problem();
    let make = || vec![SupportedXConeT::SecondOrderXConeT((0..n).collect())];
    let cached = run_direct_x(&P, &q, n, make, true);
    let non_cached = run_direct_x(&P, &q, n, make, false);
    assert_results_close(&cached, &non_cached, 1e-9);
}

#[cfg(feature = "sdp")]
#[test]
fn test_cached_grad_state_psd_direct_x() {
    // PSD at active boundary: 2x2 PSD with q chosen so the optimal X has a
    // zero eigenvalue (boundary).
    let n = 3usize; // svec_dim of 2x2
    let P = CscMatrix::<f64>::new(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0, 1.0, 1.0],
    );
    let q = vec![-0.5, 1.0, -0.5];
    let make = || vec![SupportedXConeT::PSDTriangleXConeT((0..n).collect(), 2)];
    let cached = run_direct_x(&P, &q, n, make, true);
    let non_cached = run_direct_x(&P, &q, n, make, false);
    assert_results_close(&cached, &non_cached, 1e-9);
}
