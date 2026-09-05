#![allow(non_snake_case)]
//! Integration tests: direct-x second-order cone constraints produce the
//! same primal solution as the slack SOC formulation.

use moreau::{algebra::*, solver::*};

/// Slack SOC: `min 0.5 x'Px + q'x s.t. x ∈ SOC_n` is represented as
/// `-x + s = 0, s ∈ SOC_n`.
fn solve_slack_soc_qp(P: &CscMatrix<f64>, q: &[f64], n: usize) -> DefaultSolution<f64> {
    let colptr: Vec<usize> = (0..=n).collect();
    let rowval: Vec<usize> = (0..n).collect();
    let nzval: Vec<f64> = vec![-1.0; n];
    let A = CscMatrix::new(n, n, colptr, rowval, nzval);
    let b = vec![0.0f64; n];
    let cones = vec![SecondOrderConeT(n)];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    let mut solver = DefaultSolver::new(P, q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    solver.solution
}

/// Direct-x SOC: same constraint via dir_cones = SecondOrderXConeT on all of x.
fn solve_direct_x_soc_qp(P: &CscMatrix<f64>, q: &[f64], n: usize) -> DefaultSolution<f64> {
    let A = CscMatrix::<f64>::zeros((0, n));
    let b: Vec<f64> = vec![];
    let cones: Vec<SupportedConeT<f64>> = vec![];

    let indices: Vec<usize> = (0..n).collect();
    let dir_cones = vec![SupportedXConeT::SecondOrderXConeT(indices)];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    let mut solver =
        DefaultSolver::new_with_xcones(P, q, &A, &b, &cones, &dir_cones, settings).unwrap();
    solver.solve();
    solver.solution
}

#[test]
fn test_soc_unconstrained_optimum_interior() {
    // x ∈ SOC_3 means x[0] >= ||x[1..3]||_2.
    // min 0.5 x'x - x[0]  ⇒  unconstrained opt is x = (1, 0, 0),
    // which is interior to SOC_3 so the constraint is inactive.
    let P = CscMatrix::<f64>::from(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let q = vec![-1.0, 0.0, 0.0];

    let slack = solve_slack_soc_qp(&P, &q, 3);
    let direct = solve_direct_x_soc_qp(&P, &q, 3);

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

#[test]
fn test_soc_constraint_active_boundary() {
    // min 0.5 x'x + q'x with q pulling us outside the SOC.
    // q = [1, 2, 0] pushes x[0] negative; optimum is on the SOC boundary.
    let P = CscMatrix::<f64>::from(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
    let q = vec![1.0, 2.0, 0.0];

    let slack = solve_slack_soc_qp(&P, &q, 3);
    let direct = solve_direct_x_soc_qp(&P, &q, 3);

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
    // Sanity: x must lie in the SOC.
    let norm_tail = (direct.x[1].powi(2) + direct.x[2].powi(2)).sqrt();
    assert!(
        direct.x[0] + 1e-7 >= norm_tail,
        "direct-x violated SOC: x[0]={}, ||x[1..]||={}",
        direct.x[0],
        norm_tail
    );
}

#[test]
fn test_soc_dim_20_sparse_expansion() {
    // dim > 4 triggers the rank-2 sparse expansion for direct-x SOC.
    // Verify equivalence with slack form.
    let n = 20usize;
    // Diagonal PSD P with varying entries.
    let colptr: Vec<usize> = (0..=n).collect();
    let rowval: Vec<usize> = (0..n).collect();
    let nzval: Vec<f64> = (0..n).map(|i| 1.0 + (i as f64) * 0.05).collect();
    let P = CscMatrix::<f64>::new(n, n, colptr, rowval, nzval);

    let mut q = vec![0.0; n];
    q[0] = 2.0;
    for i in 1..n {
        q[i] = 0.1 * (i as f64) - 0.5;
    }

    let slack = solve_slack_soc_qp(&P, &q, n);
    let direct = solve_direct_x_soc_qp(&P, &q, n);

    assert_eq!(slack.status, SolverStatus::Solved);
    assert_eq!(direct.status, SolverStatus::Solved);
    // Objectives match to IPM tolerance (1e-8 gap); individual x
    // components differ by O(1e-5) on the SOC boundary because direct-x
    // rank-2 expansion and slack rank-2 expansion converge to slightly
    // different boundary iterates (same objective, nearby x vectors).
    let obj_gap = (slack.obj_val - direct.obj_val).abs();
    assert!(obj_gap < 1e-6, "objective gap {} too large", obj_gap);
    for i in 0..n {
        assert!(
            (slack.x[i] - direct.x[i]).abs() < 5e-5,
            "x[{}] disagree: slack={}, direct={}",
            i,
            slack.x[i],
            direct.x[i]
        );
    }
}

#[test]
fn test_soc_dim_4_active() {
    // 4-dim SOC, constraint active.
    let P = CscMatrix::<f64>::from(&[
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    let q = vec![2.0, -1.0, 1.5, -0.5];

    let slack = solve_slack_soc_qp(&P, &q, 4);
    let direct = solve_direct_x_soc_qp(&P, &q, 4);

    assert_eq!(slack.status, SolverStatus::Solved);
    assert_eq!(direct.status, SolverStatus::Solved);
    for i in 0..4 {
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
fn test_backward_direct_x_soc_matches_slack() {
    // Backward parity for SOC direct-x via the IFT-direct path.
    // Same QP as the active-boundary forward test (q pulls x[0] negative
    // so the SOC constraint binds), backward with arbitrary upstream dx.
    // Equilibration off so reference math is exact (the IFT-direct path
    // is invariant under uniform per-cone equilibration anyway).
    let n = 3usize;
    let P = CscMatrix::<f64>::new(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0, 1.0, 1.0],
    );
    let q = vec![1.0, 2.0, 0.0];
    let dx = vec![1.0, 1.0, 1.0];

    let slack_result = {
        let colptr: Vec<usize> = (0..=n).collect();
        let rowval: Vec<usize> = (0..n).collect();
        let nzval: Vec<f64> = vec![-1.0; n];
        let A = CscMatrix::new(n, n, colptr, rowval, nzval);
        let b = vec![0.0; n];
        let cones = vec![SecondOrderConeT(n)];
        let mut settings = DefaultSettings::default();
        settings.ipm.presolve_enable = false;
        settings.ipm.equilibrate_enable = false;
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
        let dir_cones = vec![SupportedXConeT::SecondOrderXConeT((0..n).collect())];
        let mut settings = DefaultSettings::default();
        settings.ipm.presolve_enable = false;
        settings.ipm.equilibrate_enable = false;
        let mut solver =
            DefaultSolver::new_with_xcones(&P, &q, &A, &b, &cones, &dir_cones, settings).unwrap();
        solver.solve();
        assert_eq!(solver.solution.status, SolverStatus::Solved);
        let dz: Vec<f64> = vec![];
        let ds: Vec<f64> = vec![];
        solver
            .backward_batch(&dx, &ds, &dz, None, DiffMethod::Exact)
            .unwrap()
    };

    for i in 0..n {
        assert!(
            (slack_result.dq[i] - direct_result.dq[i]).abs() < 1e-5,
            "dq[{}] disagree: slack={}, direct={}",
            i,
            slack_result.dq[i],
            direct_result.dq[i]
        );
    }
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
    assert!(direct_result.db.is_empty());
    assert_eq!(direct_result.dA.m, 0);
}

#[test]
fn test_backward_direct_x_soc_high_dim_matches_slack() {
    // Backward parity at dim=20 — exercises the rank-2 sparse expansion
    // at scale where the dense fallback would be O(dim²) = 400 entries
    // per cone vs O(dim + 2*dim) ≈ 60 with sparse expansion.
    let n = 20usize;
    let P = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![1.0; n]);
    let mut q = vec![0.0; n];
    q[0] = -2.0; // Pull x[0] toward -2 → SOC binding at boundary.
    let dx = vec![1.0; n];

    let slack_result = {
        let A = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![-1.0; n]);
        let b = vec![0.0; n];
        let cones = vec![SecondOrderConeT(n)];
        let mut settings = DefaultSettings::default();
        settings.ipm.presolve_enable = false;
        settings.ipm.equilibrate_enable = false;
        settings.verbose = false;
        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
        solver.solve();
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
        let dir_cones = vec![SupportedXConeT::SecondOrderXConeT((0..n).collect())];
        let mut settings = DefaultSettings::default();
        settings.ipm.presolve_enable = false;
        settings.ipm.equilibrate_enable = false;
        settings.verbose = false;
        let mut solver =
            DefaultSolver::new_with_xcones(&P, &q, &A, &b, &cones, &dir_cones, settings).unwrap();
        solver.solve();
        let dz: Vec<f64> = vec![];
        let ds: Vec<f64> = vec![];
        solver
            .backward_batch(&dx, &ds, &dz, None, DiffMethod::Exact)
            .unwrap()
    };

    for i in 0..n {
        assert!(
            (slack_result.dq[i] - direct_result.dq[i]).abs() < 1e-4,
            "dq[{}] disagree: slack={}, direct={}",
            i,
            slack_result.dq[i],
            direct_result.dq[i]
        );
    }
    assert_eq!(slack_result.dP.nzval.len(), direct_result.dP.nzval.len());
    for k in 0..slack_result.dP.nzval.len() {
        assert!(
            (slack_result.dP.nzval[k] - direct_result.dP.nzval[k]).abs() < 1e-4,
            "dP[{}] disagree: slack={}, direct={}",
            k,
            slack_result.dP.nzval[k],
            direct_result.dP.nzval[k]
        );
    }
}
