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

/// Direct-x PSD: same constraint via `dir_cones = PSDTriangleXConeT(indices, k)`
/// on all of x.
fn solve_direct_x_psd_qp(P: &CscMatrix<f64>, q: &[f64], k: usize) -> DefaultSolution<f64> {
    let n = k * (k + 1) / 2;
    let A = CscMatrix::<f64>::zeros((0, n));
    let b: Vec<f64> = vec![];
    let cones: Vec<SupportedConeT<f64>> = vec![];

    let indices: Vec<usize> = (0..n).collect();
    let dir_cones = vec![SupportedXConeT::PSDTriangleXConeT(indices, k)];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    let mut solver =
        DefaultSolver::new_with_xcones(P, q, &A, &b, &cones, &dir_cones, settings).unwrap();
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

/// A slack PSD(4) genuinely splits into two PSD(2) blocks, alongside a
/// direct PSD(2). Check original-coordinate duals as well as primal values.
#[test]
fn test_mixed_chordal_slack_psd_plus_direct_x_psd() {
    let n = 9;
    let m = 10;
    let diagonal: Vec<f64> = vec![4., 0.5, 2., 1., 3., 2., 5., 2., 0.4];
    let P = CscMatrix::new(n, n, (0..=n).collect(), (0..n).collect(), diagonal.clone());
    let mut q = vec![0.; n];
    q[1] = -1.;
    q[4] = -1.;
    q[7] = -1.;
    // Only the two diagonal 2x2 blocks of the 4x4 slack matrix occur.
    let A = CscMatrix::new(
        m,
        n,
        vec![0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
        vec![0, 1, 2, 5, 8, 9],
        vec![-1.; 6],
    );
    let b = vec![0.; m];
    let slack_cones = vec![SupportedConeT::PSDTriangleConeT(4)];
    let dir_cones = vec![SupportedXConeT::PSDTriangleXConeT(vec![0, 1, 2], 2)];
    let run = |chordal| {
        let mut settings = DefaultSettings::default();
        settings.verbose = false;
        settings.ipm.presolve_enable = false;
        settings.ipm.chordal_decomposition_enable = chordal;
        settings.ipm.chordal_decomposition_merge_method = "none".to_string();
        settings.ipm.tol_gap_abs = 1e-11;
        settings.ipm.tol_gap_rel = 1e-11;
        settings.ipm.tol_feas = 1e-11;
        let mut solver = DefaultSolver::new_with_xcones(
            &P,
            &q,
            &A,
            &b,
            &slack_cones,
            &dir_cones,
            settings.clone(),
        )
        .unwrap();
        if chordal {
            assert_ne!(solver.data.m, m, "fixture must actually decompose");
        } else {
            assert_eq!(solver.data.m, m);
        }
        solver.solve();
        let sol = solver.solution;
        assert_eq!(sol.status, SolverStatus::Solved);
        assert!(sol.z_x.iter().any(|v| v.abs() > 0.1));
        for j in 0..n {
            let dual = if j < 3 {
                -sol.z_x[j]
            } else {
                -sol.z[[0, 1, 2, 5, 8, 9][j - 3]]
            };
            assert!((diagonal[j] * sol.x[j] + q[j] + dual).abs() < 1e-6);
        }
        assert!(
            sol.x[..3]
                .iter()
                .zip(&sol.z_x)
                .map(|(x, z)| x * z)
                .sum::<f64>()
                .abs()
                < 1e-6
        );

        let p = CsrMatrix::from_csc(&P);
        let a = CsrMatrix::from_csc(&A);
        let mut compiled = CompiledSolver::new_with_b_nnz_mask_and_xcones(
            n,
            m,
            &p.rowptr,
            &p.colval,
            &a.rowptr,
            &a.colval,
            &slack_cones,
            &dir_cones,
            settings,
            1,
            true,
            Some(&[false; 10]),
        )
        .unwrap();
        compiled
            .setup(&[p.nzval.clone()], &[a.nzval.clone()])
            .unwrap();
        let compiled_sol = compiled.solve(&[q.clone()], &[b.clone()]).unwrap();
        assert_close(&sol.z_x, &compiled_sol[0].z_x, 1e-5, "compiled z_x");
        let upstream = [UpstreamGradients {
            dx: vec![0.7; n],
            ds: vec![0.2; m],
            dz: vec![0.1; m],
            dz_x: vec![0.3; 3],
        }];
        let grads = compiled.backward(&upstream).unwrap();
        // Autograd may restore an earlier forward result after solver reuse.
        compiled
            .solve(&[q.iter().map(|v| v * 0.9).collect()], &[b.clone()])
            .unwrap();
        let saved = &compiled_sol[0];
        let explicit = compiled
            .backward_with_data_and_z_x(
                &upstream,
                &[p.nzval],
                &[a.nzval],
                &[q.clone()],
                &[b.clone()],
                &[saved.x.clone()],
                &[saved.z.clone()],
                &[saved.s.clone()],
                &[saved.z_x.clone()],
            )
            .unwrap();
        assert_close(&grads[0].dq, &explicit[0].dq, 1e-4, "external dq");
        assert_close(
            &grads[0].dP_values,
            &explicit[0].dP_values,
            1e-4,
            "external dP",
        );
        assert_close(
            &grads[0].dA_values,
            &explicit[0].dA_values,
            1e-4,
            "external dA",
        );
        for i in [0, 1, 2, 5, 8, 9] {
            assert!((grads[0].db[i] - explicit[0].db[i]).abs() < 1e-4);
        }
        (sol, grads.into_iter().next().unwrap())
    };
    let (reference, ref_grad) = run(false);
    let (decomposed, dec_grad) = run(true);
    assert_close(&reference.x, &decomposed.x, 1e-5, "x");
    assert_close(&reference.s, &decomposed.s, 1e-5, "s");
    assert_close(&reference.z, &decomposed.z, 1e-5, "z");
    assert_close(&reference.z_x, &decomposed.z_x, 1e-5, "z_x");
    assert_close(&ref_grad.dq, &dec_grad.dq, 1e-4, "dq");
    assert_close(&ref_grad.dP_values, &dec_grad.dP_values, 1e-4, "dP");
    assert_close(&ref_grad.dA_values, &dec_grad.dA_values, 1e-4, "dA");
    // Only structurally present rows of b are parameters of this decomposition.
    for i in [0, 1, 2, 5, 8, 9] {
        assert!((ref_grad.db[i] - dec_grad.db[i]).abs() < 1e-4);
    }
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
        let dir_cones = vec![SupportedXConeT::PSDTriangleXConeT((0..n).collect(), 2)];
        let mut settings = DefaultSettings::default();
        settings.ipm.presolve_enable = false;
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
