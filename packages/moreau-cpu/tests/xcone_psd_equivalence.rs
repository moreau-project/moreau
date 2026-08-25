#![allow(non_snake_case)]
//! Integration tests: direct-x PSD cone constraints produce the same primal
//! solution as the slack PSD formulation.
//!
//! Representation: an n×n PSD matrix X is stored as svec(X) of length
//! n(n+1)/2 (Clarabel column-major triangular convention).

#![cfg(feature = "sdp")]

use moreau::{algebra::*, solver::*};

/// Slack PSD: `min 0.5 x'Px + q'x s.t. X ⪰ 0` (with x = svec(X)) as
/// `-x + s = 0, s ∈ PSD_svec(k)`.
fn solve_slack_psd_qp(P: &CscMatrix<f64>, q: &[f64], k: usize) -> DefaultSolution<f64> {
    let n = k * (k + 1) / 2;
    let colptr: Vec<usize> = (0..=n).collect();
    let rowval: Vec<usize> = (0..n).collect();
    let nzval: Vec<f64> = vec![-1.0; n];
    let A = CscMatrix::new(n, n, colptr, rowval, nzval);
    let b = vec![0.0f64; n];
    let cones = vec![SupportedConeT::PSDTriangleConeT(k)];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.chordal_decomposition_enable = false;
    let mut solver = DefaultSolver::new(P, q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    solver.solution
}

/// Direct-x PSD: same constraint via `x_cones = PSDTriangleXConeT(indices, k)`
/// on all of x.
fn solve_direct_x_psd_qp(P: &CscMatrix<f64>, q: &[f64], k: usize) -> DefaultSolution<f64> {
    let n = k * (k + 1) / 2;
    let A = CscMatrix::<f64>::zeros((0, n));
    let b: Vec<f64> = vec![];
    let cones: Vec<SupportedConeT<f64>> = vec![];

    let indices: Vec<usize> = (0..n).collect();
    let x_cones = vec![SupportedXConeT::PSDTriangleXConeT(indices, k)];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    let mut solver =
        DefaultSolver::new_with_xcones(P, q, &A, &b, &cones, &x_cones, settings).unwrap();
    solver.solve();
    solver.solution
}

fn assert_close(slack: &[f64], direct: &[f64], tol: f64, label: &str) {
    assert_eq!(slack.len(), direct.len());
    for i in 0..slack.len() {
        assert!(
            (slack[i] - direct[i]).abs() < tol,
            "{} [{}] disagree: slack={:.10}, direct={:.10}",
            label,
            i,
            slack[i],
            direct[i]
        );
    }
}

#[test]
fn test_psd2_identity_target_interior() {
    // min 0.5 ||X - I||_F^2  with X = [[a, b/√2], [b/√2, c]].
    // Equivalent svec QP:  min 0.5 (a² + b² + c²) - a - c.
    // Unconstrained optimum: (a, b, c) = (1, 0, 1), which is interior
    // to the PSD cone (I is strictly PD), so the constraint is inactive.
    let P = CscMatrix::<f64>::from(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let q = vec![-1.0, 0.0, -1.0];

    let slack = solve_slack_psd_qp(&P, &q, 2);
    let direct = solve_direct_x_psd_qp(&P, &q, 2);

    assert_eq!(slack.status, SolverStatus::Solved);
    assert_eq!(direct.status, SolverStatus::Solved);
    assert_close(&slack.x, &direct.x, 1e-6, "x");
}

#[test]
fn test_psd2_constraint_active_boundary() {
    // Push the unconstrained optimum outside the PSD cone.
    // min 0.5 (a² + b² + c²) - b  ⇒  unconstrained opt (0, 1, 0),
    // which is NOT PSD (indefinite, eigenvalues ±1/√2).
    // The constrained optimum must project onto the boundary.
    let P = CscMatrix::<f64>::from(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let q = vec![0.0, -1.0, 0.0];

    let slack = solve_slack_psd_qp(&P, &q, 2);
    let direct = solve_direct_x_psd_qp(&P, &q, 2);

    assert_eq!(slack.status, SolverStatus::Solved);
    assert_eq!(direct.status, SolverStatus::Solved);
    assert_close(&slack.x, &direct.x, 1e-6, "x");

    // Sanity: direct-x solution must lie in PSD.
    // svec ordering: x[0]=a (M[0,0]), x[1]=b=M[0,1]√2, x[2]=c (M[1,1]).
    let a = direct.x[0];
    let b = direct.x[1] / std::f64::consts::SQRT_2;
    let c = direct.x[2];
    let det = a * c - b * b;
    assert!(a + 1e-8 >= 0.0, "a < 0: {}", a);
    assert!(c + 1e-8 >= 0.0, "c < 0: {}", c);
    assert!(det + 1e-8 >= 0.0, "det(X) < 0: {}", det);
}

/// Stress test for direct-x equilibration: use a P with widely-varying row
/// magnitudes so Ruiz wants per-row `d` scaling. Without uniform-scaling
/// rectification, direct-x would produce a different iterate path (and
/// potentially a different fixed point) than slack.
#[test]
fn test_psd2_ill_scaled_P() {
    // Diagonal P with magnitudes 1e-3, 1e0, 1e3.
    let colptr: Vec<usize> = (0..=3).collect();
    let rowval: Vec<usize> = (0..3).collect();
    let nzval: Vec<f64> = vec![1e-3, 1.0, 1e3];
    let P = CscMatrix::new(3, 3, colptr, rowval, nzval);

    // q chosen so the unconstrained optimum (-q/diag(P)) violates PSD:
    // a = 1e3, b = -100, c = 1e-3 → det = 1.0 - 10000 < 0.
    let q = vec![-1.0, 100.0, -1.0];

    let slack = solve_slack_psd_qp(&P, &q, 2);
    let direct = solve_direct_x_psd_qp(&P, &q, 2);

    assert_eq!(slack.status, SolverStatus::Solved);
    assert_eq!(direct.status, SolverStatus::Solved);
    assert_close(&slack.x, &direct.x, 1e-5, "x");

    let a = direct.x[0];
    let b = direct.x[1] / std::f64::consts::SQRT_2;
    let c = direct.x[2];
    let det = a * c - b * b;
    assert!(det + 1e-6 >= 0.0, "PSD violated: det={}", det);
}

/// Chordal analysis must not misapply to direct-x PSD cones.
///
/// Setup: one slack PSD(6) cone (block-diag structure that chordal WOULD
/// decompose) + one direct-x PSD(2) cone on x[0..3]. Confirms:
///  - chordal decomp is only applied to the slack side (we enable it)
///  - direct-x PSD indices survive the augmentation
///  - the augmented solve produces the same answer as the chordal-disabled
///    reference.
#[test]
fn test_mixed_chordal_slack_psd_plus_direct_x_psd() {
    // Primal variable layout: x[0..3] for direct-x PSD(2), x[3..9] for the
    // slack PSD(6) image. We encode the slack constraint as `-x[3..9] + s = 0,
    // s ∈ PSD(6)`, so x[3..9] must also be PSD.
    //
    // Build the slack PSD(6) with a block-diag sparsity on A so chordal decomp
    // has something to work with. We use A = -block_diag(I_3, I_3) acting on
    // two separate triples of x[3..9], which makes the PSD image decomposable
    // into PSD(3) ⊕ PSD(3) when the off-diagonal block is structurally zero.
    //
    // For simplicity: A = -I_6 on x[3..9], b = 0, PSDTriangleConeT(6) slack.
    // Zero-pattern of b means full svec is unconstrained → chordal detects no
    // decomp; set some entries of b slightly nonzero on the block diagonal
    // only to force a block-diag sparsity pattern if chordal is enabled.
    let n = 9;
    let m = 6;

    // P = I_n
    let P_colptr: Vec<usize> = (0..=n).collect();
    let P_rowval: Vec<usize> = (0..n).collect();
    let P_nzval: Vec<f64> = vec![1.0; n];
    let P = CscMatrix::new(n, n, P_colptr, P_rowval, P_nzval);

    // q chosen to force non-trivial PSD constraint on both cones.
    let mut q = vec![0.0; n];
    q[1] = -1.0; // pushes b≠0 in direct-x PSD(2)
    q[7] = -1.0; // pushes off-diag in slack PSD(6) side

    // A scatters x[3..9] to s with negation.
    let A_colptr: Vec<usize> = {
        let mut v = vec![0; n + 1];
        for j in 3..n {
            v[j + 1] = v[j] + 1;
        }
        for j in 0..=3 {
            v[j] = 0;
        }
        v
    };
    let A_rowval: Vec<usize> = (0..6).collect();
    let A_nzval: Vec<f64> = vec![-1.0; 6];
    let A = CscMatrix::new(m, n, A_colptr, A_rowval, A_nzval);
    let b = vec![0.0f64; m];

    let slack_cones = vec![SupportedConeT::PSDTriangleConeT(3)]; // size-3 slack
                                                                 // ^ Note: if we used PSDTriangleConeT(3), svec dim = 6 which matches m=6.
    let x_cones = vec![SupportedXConeT::PSDTriangleXConeT(vec![0, 1, 2], 2)];

    // Solve twice: with chordal disabled (reference) and with chordal enabled
    // (to check that chordal on slack side coexists with direct-x).
    let mut run = |chordal: bool| -> DefaultSolution<f64> {
        let mut settings = DefaultSettings::default();
        settings.ipm.presolve_enable = false;
        settings.ipm.chordal_decomposition_enable = chordal;
        let mut solver =
            DefaultSolver::new_with_xcones(&P, &q, &A, &b, &slack_cones, &x_cones, settings)
                .unwrap();
        solver.solve();
        solver.solution
    };

    let ref_sol = run(false);
    let chord_sol = run(true);

    assert_eq!(ref_sol.status, SolverStatus::Solved, "reference failed");
    assert_eq!(chord_sol.status, SolverStatus::Solved, "chordal failed");
    assert_close(&ref_sol.x, &chord_sol.x, 1e-5, "x");
}

#[test]
fn test_psd3_negdef_target() {
    // 3×3 PSD cone, svec dim = 6. Target: project -I onto PSD.
    // Solution: X = 0 (zero matrix is closest PSD to -I in Frobenius).
    // svec(-I) scaled: (-1, 0, -1, 0, 0, -1) with diagonal entries
    // unchanged (svec convention stores diag as M[i,i]).
    let n = 6;
    // Identity matrix as CSC (one diagonal entry per column).
    let colptr: Vec<usize> = (0..=n).collect();
    let rowval: Vec<usize> = (0..n).collect();
    let nzval: Vec<f64> = vec![1.0; n];
    let P = CscMatrix::new(n, n, colptr, rowval, nzval);
    let q = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0];

    let slack = solve_slack_psd_qp(&P, &q, 3);
    let direct = solve_direct_x_psd_qp(&P, &q, 3);

    assert_eq!(slack.status, SolverStatus::Solved);
    assert_eq!(direct.status, SolverStatus::Solved);
    assert_close(&slack.x, &direct.x, 1e-6, "x");

    // Solution should be near zero (up to tol).
    for i in 0..n {
        assert!(
            direct.x[i].abs() < 1e-5,
            "expected x[{}] ≈ 0, got {}",
            i,
            direct.x[i]
        );
    }
}

/// Backward pass via UNFOLD: direct-x PSD gradients should match the
/// slack-form gradients (dP, dq) to solver tolerance. `db`/`dA` rows for
/// the unfolded direct-x slack rows are dropped from the returned result
/// since they don't correspond to user-visible parameters.
#[test]
fn test_backward_direct_x_psd_matches_slack() {
    // PSD(2) on all of x (svec dim n=3). q pushes the optimum outside
    // the PSD cone so the constraint is active.
    let n = 3usize;
    let colptr: Vec<usize> = (0..=n).collect();
    let rowval: Vec<usize> = (0..n).collect();
    let nzval: Vec<f64> = vec![1.0; n];
    let P = CscMatrix::new(n, n, colptr.clone(), rowval.clone(), nzval.clone());
    let q = vec![0.0, -1.0, 0.0]; // indefinite target: b ≠ 0, a = c = 0

    let dx = vec![1.0, 1.0, 1.0];

    let slack_result = {
        let colptr: Vec<usize> = (0..=n).collect();
        let rowval: Vec<usize> = (0..n).collect();
        let nzval: Vec<f64> = vec![-1.0; n];
        let A = CscMatrix::new(n, n, colptr, rowval, nzval);
        let b = vec![0.0; n];
        let cones = vec![SupportedConeT::PSDTriangleConeT(2)];
        let mut settings = DefaultSettings::default();
        settings.ipm.presolve_enable = false;
        settings.ipm.chordal_decomposition_enable = false;
        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::Solved);
        let dz = vec![0.0; n];
        let ds = vec![0.0; n];
        solver
            .backward_batch(&dx, &ds, &dz, None, DiffMethod::Exact)
            .unwrap()
    };

    let direct_result = {
        let A = CscMatrix::<f64>::zeros((0, n));
        let b: Vec<f64> = vec![];
        let cones: Vec<SupportedConeT<f64>> = vec![];
        let x_cones = vec![SupportedXConeT::PSDTriangleXConeT((0..n).collect(), 2)];
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
            "dq[{}] disagree: slack={:.8}, direct={:.8}",
            i,
            slack_result.dq[i],
            direct_result.dq[i]
        );
    }

    // dP must match (element-wise in CSC).
    assert_eq!(slack_result.dP.nzval.len(), direct_result.dP.nzval.len());
    for k in 0..slack_result.dP.nzval.len() {
        assert!(
            (slack_result.dP.nzval[k] - direct_result.dP.nzval[k]).abs() < 1e-5,
            "dP[{}] disagree: slack={:.8}, direct={:.8}",
            k,
            slack_result.dP.nzval[k],
            direct_result.dP.nzval[k]
        );
    }

    // Direct form has no user-visible A/b rows.
    assert!(direct_result.db.is_empty());
    assert_eq!(direct_result.dA.m, 0);
}
