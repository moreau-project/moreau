#![allow(non_snake_case)]
//! Integration tests for direct-x nonneg cones: the forward pass should
//! produce the same primal solution as the slack-form equivalent, within
//! solver tolerance.

use moreau::{algebra::*, solver::*};

/// Solve `min 0.5 x'Px + q'x s.t. x ≥ 0` via the slack form
/// `min 0.5 x'Px + q'x s.t. -x + s = 0, s ∈ R+^n`.
fn solve_slack_nonneg_qp(P: &CscMatrix<f64>, q: &[f64], n: usize) -> DefaultSolution<f64> {
    // A = -I (n×n), b = 0, single nonneg cone of dim n.
    let colptr: Vec<usize> = (0..=n).collect();
    let rowval: Vec<usize> = (0..n).collect();
    let nzval: Vec<f64> = vec![-1.0; n];
    let A = CscMatrix::new(n, n, colptr, rowval, nzval);
    let b = vec![0.0f64; n];
    let cones = vec![NonnegativeConeT(n)];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    let mut solver = DefaultSolver::new(P, q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    solver.solution
}

/// Solve `min 0.5 x'Px + q'x s.t. x ≥ 0` via direct-x nonneg on all of x.
fn solve_direct_x_nonneg_qp(P: &CscMatrix<f64>, q: &[f64], n: usize) -> DefaultSolution<f64> {
    // No slack cones, A is an empty 0×n matrix, b is empty.
    let A = CscMatrix::<f64>::zeros((0, n));
    let b: Vec<f64> = vec![];
    let cones: Vec<SupportedConeT<f64>> = vec![];

    let indices: Vec<usize> = (0..n).collect();
    let x_cones = vec![SupportedXConeT::NonnegativeXConeT(indices)];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    let mut solver =
        DefaultSolver::new_with_xcones(P, q, &A, &b, &cones, &x_cones, settings).unwrap();
    solver.solve();
    solver.solution
}

#[test]
fn test_scalar_nonneg_unconstrained_optimum_positive() {
    // min 0.5 x^2 - 4x s.t. x >= 0; unconstrained opt at x=4 satisfies x>=0
    let P = CscMatrix::<f64>::from(&[[1.0]]);
    let q = vec![-4.0];

    let slack = solve_slack_nonneg_qp(&P, &q, 1);
    let direct = solve_direct_x_nonneg_qp(&P, &q, 1);

    assert_eq!(slack.status, SolverStatus::Solved, "slack form failed");
    assert_eq!(direct.status, SolverStatus::Solved, "direct-x form failed");
    assert!(
        (slack.x[0] - direct.x[0]).abs() < 1e-6,
        "x disagree: slack={}, direct={}",
        slack.x[0],
        direct.x[0]
    );
    assert!(
        (direct.x[0] - 4.0).abs() < 1e-6,
        "expected x≈4, got {}",
        direct.x[0]
    );
}

#[test]
fn test_scalar_nonneg_constraint_active() {
    // min 0.5 x^2 + 2x s.t. x >= 0; constrained opt at x=0
    let P = CscMatrix::<f64>::from(&[[1.0]]);
    let q = vec![2.0];

    let slack = solve_slack_nonneg_qp(&P, &q, 1);
    let direct = solve_direct_x_nonneg_qp(&P, &q, 1);

    assert_eq!(slack.status, SolverStatus::Solved);
    assert_eq!(direct.status, SolverStatus::Solved);
    assert!(
        (slack.x[0] - direct.x[0]).abs() < 1e-6,
        "x disagree: slack={}, direct={}",
        slack.x[0],
        direct.x[0]
    );
    assert!(
        direct.x[0].abs() < 1e-6,
        "expected x≈0, got {}",
        direct.x[0]
    );
}

#[test]
fn test_nonneg_qp_with_off_diagonal_P() {
    // P has off-diagonal coupling between x[0] and x[1]; optimum is a
    // nontrivial tradeoff between the linear term and the cone boundary.
    let P = CscMatrix::<f64>::from(&[[2.0, 0.5], [0.5, 2.0]]);
    let q = vec![-1.0, 1.5];

    let slack = solve_slack_nonneg_qp(&P, &q, 2);
    let direct = solve_direct_x_nonneg_qp(&P, &q, 2);

    assert_eq!(slack.status, SolverStatus::Solved);
    assert_eq!(direct.status, SolverStatus::Solved);
    for i in 0..2 {
        assert!(
            (slack.x[i] - direct.x[i]).abs() < 1e-6,
            "x[{}] disagree: slack={}, direct={}",
            i,
            slack.x[i],
            direct.x[i]
        );
    }
}

#[test]
fn test_partial_direct_x_nonneg() {
    // 3 variables; only x[0] and x[2] are direct-x nonneg, x[1] free.
    // Compare against the slack form where x[0], x[2] have slack-form
    // constraints.
    let P = CscMatrix::<f64>::from(&[[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]);
    let q = vec![2.0, -1.0, 0.5]; // x1*: -1, x2*: -0.25 (clipped to 0)
    let n = 3usize;

    // Direct-x form: nonneg on indices [0, 2]; no slack cones, no A.
    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let x_cones_dx = vec![SupportedXConeT::NonnegativeXConeT(vec![0, 2])];

    let mut settings_dx = DefaultSettings::default();
    settings_dx.ipm.presolve_enable = false;
    let mut solver_dx =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &x_cones_dx, settings_dx)
            .unwrap();
    solver_dx.solve();
    let direct = solver_dx.solution;

    // Slack form: A has rows for -x[0] + s0 = 0, -x[2] + s1 = 0;
    // single nonneg cone of dim 2.
    let A_slack = CscMatrix::<f64>::from(&[[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]);
    let b_slack = vec![0.0, 0.0];
    let cones_slack = vec![NonnegativeConeT(2)];

    let mut settings_slack = DefaultSettings::default();
    settings_slack.ipm.presolve_enable = false;
    let mut solver_slack =
        DefaultSolver::new(&P, &q, &A_slack, &b_slack, &cones_slack, settings_slack).unwrap();
    solver_slack.solve();
    let slack = solver_slack.solution;

    assert_eq!(slack.status, SolverStatus::Solved);
    assert_eq!(direct.status, SolverStatus::Solved);
    for i in 0..n {
        assert!(
            (slack.x[i] - direct.x[i]).abs() < 1e-6,
            "x[{}] disagree: slack={}, direct={}",
            i,
            slack.x[i],
            direct.x[i]
        );
    }
}

#[test]
fn test_backward_direct_x_matches_slack() {
    // Backward pass via UNFOLD: direct-x nonneg gradients should match
    // the slack form gradients (dP, dq, dA, db) to solver tolerance.
    // We compare the unified moreau.Solver path can't be used here (no
    // backward yet), so use DefaultSolver + DefaultSolver::new_with_xcones.
    let n = 3usize;
    let P = CscMatrix::<f64>::new(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        vec![2.0, 2.0, 2.0],
    );
    let q = vec![-1.0, 2.0, -0.5]; // unconstrained (0.5, -1, 0.25); clipped (0.5, 0, 0.25)

    let dx = vec![1.0, 1.0, 1.0]; // arbitrary upstream gradient

    // Slack form gradients
    let slack_result = {
        let colptr: Vec<usize> = (0..=n).collect();
        let rowval: Vec<usize> = (0..n).collect();
        let nzval: Vec<f64> = vec![-1.0; n];
        let A = CscMatrix::new(n, n, colptr, rowval, nzval);
        let b = vec![0.0; n];
        let cones = vec![NonnegativeConeT(n)];
        let mut settings = DefaultSettings::default();
        settings.ipm.presolve_enable = false;
        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::Solved);
        let dz = vec![0.0; n];
        let ds = vec![0.0; n];
        solver
            .backward_batch(&dx, &ds, &dz, None, DiffMethod::Exact)
            .unwrap()
    };

    // Direct-x form gradients (via unfold inside backward_batch).
    let direct_result = {
        let A = CscMatrix::<f64>::zeros((0, n));
        let b: Vec<f64> = vec![];
        let cones: Vec<SupportedConeT<f64>> = vec![];
        let x_cones = vec![SupportedXConeT::NonnegativeXConeT((0..n).collect())];
        let mut settings = DefaultSettings::default();
        settings.ipm.presolve_enable = false;
        let mut solver =
            DefaultSolver::new_with_xcones(&P, &q, &A, &b, &cones, &x_cones, settings).unwrap();
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::Solved);
        let dz: Vec<f64> = vec![];
        let ds: Vec<f64> = vec![];
        solver
            .backward_batch(&dx, &ds, &dz, None, DiffMethod::Exact)
            .unwrap()
    };

    // dq must match (same n components).
    for i in 0..n {
        assert!(
            (slack_result.dq[i] - direct_result.dq[i]).abs() < 1e-5,
            "dq[{}] disagree: slack={}, direct={}",
            i,
            slack_result.dq[i],
            direct_result.dq[i]
        );
    }

    // dP must match.
    assert_eq!(slack_result.dP.nzval.len(), direct_result.dP.nzval.len());
    for k in 0..slack_result.dP.nzval.len() {
        assert!(
            (slack_result.dP.nzval[k] - direct_result.dP.nzval[k]).abs() < 1e-5,
            "dP[{}] disagree: slack={}, direct={}",
            k,
            slack_result.dP.nzval[k],
            direct_result.dP.nzval[k]
        );
    }

    // db — direct form has m=0, so nothing to compare.
    // dA — direct form has 0 rows, nothing to compare (the slack rows
    // don't correspond to user parameters in the direct-x form).
    assert!(direct_result.db.is_empty());
    assert_eq!(direct_result.dA.m, 0);
}

#[test]
fn test_small_nonneg_qp_equivalence() {
    // 3-dim QP: min 0.5 x'Px + q'x s.t. x >= 0 where P is 2I and q = [-1, 2, -0.5]
    // Unconstrained opt is (0.5, -1, 0.25); clipping to x>=0 gives (0.5, 0, 0.25).
    let P = CscMatrix::<f64>::from(&[
        [2.0, 0.0, 0.0], //
        [0.0, 2.0, 0.0], //
        [0.0, 0.0, 2.0], //
    ]);
    let q = vec![-1.0, 2.0, -0.5];

    let slack = solve_slack_nonneg_qp(&P, &q, 3);
    let direct = solve_direct_x_nonneg_qp(&P, &q, 3);

    assert_eq!(slack.status, SolverStatus::Solved);
    assert_eq!(direct.status, SolverStatus::Solved);
    for i in 0..3 {
        assert!(
            (slack.x[i] - direct.x[i]).abs() < 1e-6,
            "x[{}] disagree: slack={}, direct={}",
            i,
            slack.x[i],
            direct.x[i]
        );
    }
}
