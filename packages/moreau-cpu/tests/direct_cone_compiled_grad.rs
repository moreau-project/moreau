//! `CompiledSolver` batched backward with `enable_grad=true` and direct-x
//! cones. Verifies that the cached-grad-state path produces gradients
//! matching the single-problem `DefaultSolver::backward_batch` reference.

#![allow(non_snake_case)]

use moreau::algebra::CscMatrix;
use moreau::solver::*;

#[test]
fn test_compiled_solver_grad_state_direct_x_nonneg_batch() {
    // Direct-x nonneg constraint on x[0..3]; batch of 2 problems with
    // distinct q vectors, shared P. Compare CompiledSolver.backward()
    // to single-problem DefaultSolver.backward_batch.
    let n = 3usize;
    let m = 0usize;

    let P_ro: Vec<usize> = (0..=n).collect();
    let P_ci: Vec<usize> = (0..n).collect();
    let A_ro: Vec<usize> = vec![0; m + 1];
    let A_ci: Vec<usize> = vec![];
    let cones: Vec<SupportedConeT<f64>> = vec![];
    let dir_cones = vec![SupportedXConeT::NonnegativeXConeT((0..n).collect())];

    let P_values = vec![1.0, 1.0, 1.0];
    let A_values: Vec<f64> = vec![];
    // q1: unconstrained optimum has all components nonneg → cone inactive.
    // q2: pushes x[2] negative → cone active on third component.
    let q1 = vec![-1.0, -1.0, -1.0];
    let q2 = vec![-0.5, -0.5, 1.0];
    let b: Vec<f64> = vec![];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.ipm.equilibrate_enable = false;

    let mut solver = CompiledSolver::new_with_xcones(
        n,
        m,
        &P_ro,
        &P_ci,
        &A_ro,
        &A_ci,
        &cones,
        &dir_cones,
        settings.clone(),
        1,
        true,
    )
    .unwrap();

    solver.setup(
        &[P_values.clone(), P_values.clone()],
        &[A_values.clone(), A_values.clone()],
    );
    let solutions = solver
        .solve(&[q1.clone(), q2.clone()], &[b.clone(), b.clone()])
        .unwrap();
    assert_eq!(solutions[0].status, SolverStatus::Solved);
    assert_eq!(solutions[1].status, SolverStatus::Solved);

    let upstream = vec![
        UpstreamGradients {
            dx: vec![1.0; n],
            ds: vec![],
            dz: vec![],
            dz_x: vec![],
        },
        UpstreamGradients {
            dx: vec![1.0; n],
            ds: vec![],
            dz: vec![],
            dz_x: vec![],
        },
    ];
    let batched = solver.backward(&upstream).unwrap();

    // Reference: per-problem DefaultSolver backward.
    let P = CscMatrix::<f64>::new(n, n, P_ro.clone(), P_ci.clone(), P_values.clone());
    let A = CscMatrix::<f64>::zeros((m, n));
    for (qb, expected) in [(q1, &batched[0]), (q2, &batched[1])] {
        let mut s =
            DefaultSolver::new_with_xcones(&P, &qb, &A, &b, &cones, &dir_cones, settings.clone())
                .unwrap();
        s.solve();
        assert_eq!(s.solution.status, SolverStatus::Solved);
        let dx = vec![1.0; n];
        let ref_grad = s
            .backward_batch(&dx, &[], &[], None, DiffMethod::Exact)
            .unwrap();
        for i in 0..n {
            assert!(
                (expected.dq[i] - ref_grad.dq[i]).abs() < 1e-7,
                "dq[{}] batched={} vs ref={}",
                i,
                expected.dq[i],
                ref_grad.dq[i]
            );
        }
        // dP_values are returned in CSR order; for our diagonal P each entry
        // matches the corresponding diagonal slot.
        assert_eq!(expected.dP_values.len(), ref_grad.dP.nzval.len());
        for k in 0..ref_grad.dP.nzval.len() {
            assert!(
                (expected.dP_values[k] - ref_grad.dP.nzval[k]).abs() < 1e-7,
                "dP[{}] batched={} vs ref={}",
                k,
                expected.dP_values[k],
                ref_grad.dP.nzval[k]
            );
        }
    }
}

#[test]
fn test_compiled_solver_grad_state_direct_x_soc_batch() {
    // Direct-x SOC on all 3 dims, batch of 2 with active-boundary q vectors.
    let n = 3usize;
    let m = 0usize;

    let P_ro: Vec<usize> = (0..=n).collect();
    let P_ci: Vec<usize> = (0..n).collect();
    let A_ro: Vec<usize> = vec![0; m + 1];
    let A_ci: Vec<usize> = vec![];
    let cones: Vec<SupportedConeT<f64>> = vec![];
    let dir_cones = vec![SupportedXConeT::SecondOrderXConeT((0..n).collect())];

    let P_values = vec![1.0, 1.0, 1.0];
    let q1 = vec![1.0, 2.0, 0.0]; // SOC active boundary
    let q2 = vec![0.5, 0.5, 0.5];
    let b: Vec<f64> = vec![];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.ipm.equilibrate_enable = false;

    let mut solver = CompiledSolver::new_with_xcones(
        n,
        m,
        &P_ro,
        &P_ci,
        &A_ro,
        &A_ci,
        &cones,
        &dir_cones,
        settings.clone(),
        1,
        true,
    )
    .unwrap();

    solver.setup(&[P_values.clone(), P_values.clone()], &[vec![], vec![]]);
    let solutions = solver
        .solve(&[q1.clone(), q2.clone()], &[b.clone(), b.clone()])
        .unwrap();
    assert_eq!(solutions[0].status, SolverStatus::Solved);
    assert_eq!(solutions[1].status, SolverStatus::Solved);

    let upstream = vec![
        UpstreamGradients {
            dx: vec![1.0; n],
            ds: vec![],
            dz: vec![],
            dz_x: vec![],
        },
        UpstreamGradients {
            dx: vec![1.0; n],
            ds: vec![],
            dz: vec![],
            dz_x: vec![],
        },
    ];
    let batched = solver.backward(&upstream).unwrap();

    let P = CscMatrix::<f64>::new(n, n, P_ro.clone(), P_ci.clone(), P_values.clone());
    let A = CscMatrix::<f64>::zeros((m, n));
    for (qb, expected) in [(q1, &batched[0]), (q2, &batched[1])] {
        let mut s =
            DefaultSolver::new_with_xcones(&P, &qb, &A, &b, &cones, &dir_cones, settings.clone())
                .unwrap();
        s.solve();
        assert_eq!(s.solution.status, SolverStatus::Solved);
        let dx = vec![1.0; n];
        let ref_grad = s
            .backward_batch(&dx, &[], &[], None, DiffMethod::Exact)
            .unwrap();
        for i in 0..n {
            assert!(
                (expected.dq[i] - ref_grad.dq[i]).abs() < 1e-7,
                "dq[{}] batched={} vs ref={}",
                i,
                expected.dq[i],
                ref_grad.dq[i]
            );
        }
    }
}
