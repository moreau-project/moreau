//! Tests for CompiledSolver::backward_with_data().
//!
//! Verifies that backward_with_data() (which accepts problem data and solution
//! externally) produces identical gradients to the normal backward() path
//! (which uses cached state from a prior solve).

#![allow(non_snake_case)]

use moreau::solver::*;

/// Helper: assert two slices are element-wise close.
fn assert_vecs_close(a: &[f64], b: &[f64], tol: f64, label: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{}: length mismatch {} vs {}",
        label,
        a.len(),
        b.len()
    );
    for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (ai - bi).abs() < tol,
            "{} mismatch at index {}: {} vs {} (diff={:.2e})",
            label,
            i,
            ai,
            bi,
            (ai - bi).abs()
        );
    }
}

/// Simple 2-variable QP with nonneg cones.
///
///   minimize  (1/2) x'Px + q'x
///   s.t.      Ax + s = b, s >= 0
///
/// P = [[2, 0], [0, 2]]  (identity scaled by 2)
/// A = [[1, 0], [0, 1], [1, 1]]  (3 constraints)
/// cones = NonnegativeCone(3)
fn simple_qp_structure() -> (
    usize,
    usize, // n, m
    Vec<usize>,
    Vec<usize>, // P_ro, P_ci
    Vec<usize>,
    Vec<usize>,               // A_ro, A_ci
    Vec<SupportedConeT<f64>>, // cones
) {
    let n = 2;
    let m = 3;

    // P in CSR: 2x2 diagonal, full symmetric (both triangles — here it's diagonal so same)
    let P_ro = vec![0, 1, 2];
    let P_ci = vec![0, 1];

    // A in CSR: 3x2
    // row 0: [1, 0]
    // row 1: [0, 1]
    // row 2: [1, 1]
    let A_ro = vec![0, 1, 2, 4];
    let A_ci = vec![0, 1, 0, 1];

    let cones = vec![NonnegativeConeT(3)];

    (n, m, P_ro, P_ci, A_ro, A_ci, cones)
}

#[test]
fn test_backward_with_data_matches_backward_simple_qp() {
    let (n, m, P_ro, P_ci, A_ro, A_ci, cones) = simple_qp_structure();

    let P_values = vec![2.0, 2.0];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];
    let q = vec![-3.0, -2.0];
    let b = vec![1.5, 1.5, 2.0];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    // --- Solve with enable_grad=true to populate cache ---
    let mut solver = CompiledSolver::new(
        n,
        m,
        &P_ro,
        &P_ci,
        &A_ro,
        &A_ci,
        &cones,
        settings.clone(),
        1,
        true,
    )
    .unwrap();

    solver.setup(&[P_values.clone()], &[A_values.clone()]);
    let results = solver.solve(&[q.clone()], &[b.clone()]).unwrap();
    assert_eq!(results[0].status, SolverStatus::Solved);

    let x = results[0].x.clone();
    let z = results[0].z.clone();
    let s = results[0].s.clone();

    // Upstream gradients: dx = [1, 1], ds = 0, dz = 0
    let upstream = vec![UpstreamGradients {
        dx: vec![1.0, 1.0],
        ds: vec![0.0; m],
        dz: vec![0.0; m],
        dz_x: vec![],
    }];

    // Normal backward (uses cached state)
    let normal_grads = solver.backward(&upstream).unwrap();

    // backward_with_data (uses externally-provided data)
    let data_grads = solver
        .backward_with_data(
            &upstream,
            &[P_values.clone()],
            &[A_values.clone()],
            &[q.clone()],
            &[b.clone()],
            &[x.clone()],
            &[z.clone()],
            &[s.clone()],
        )
        .unwrap();

    // backward_with_data re-equilibrates from scratch, so floating-point
    // differences in equilibration can cause small discrepancies.
    let tol = 1e-5;
    assert_vecs_close(
        &normal_grads[0].dP_values,
        &data_grads[0].dP_values,
        tol,
        "dP_values",
    );
    assert_vecs_close(&normal_grads[0].dq, &data_grads[0].dq, tol, "dq");
    assert_vecs_close(
        &normal_grads[0].dA_values,
        &data_grads[0].dA_values,
        tol,
        "dA_values",
    );
    assert_vecs_close(&normal_grads[0].db, &data_grads[0].db, tol, "db");
}

#[test]
fn test_backward_with_data_batch() {
    let (n, m, P_ro, P_ci, A_ro, A_ci, cones) = simple_qp_structure();

    let P_values = vec![2.0, 2.0];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];

    // Two slightly different q vectors for batch
    let q1 = vec![-3.0, -2.0];
    let q2 = vec![-2.5, -1.5];
    let b = vec![1.5, 1.5, 2.0];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver = CompiledSolver::new(
        n,
        m,
        &P_ro,
        &P_ci,
        &A_ro,
        &A_ci,
        &cones,
        settings.clone(),
        2,
        true,
    )
    .unwrap();

    solver.setup(
        &[P_values.clone(), P_values.clone()],
        &[A_values.clone(), A_values.clone()],
    );
    let results = solver
        .solve(&[q1.clone(), q2.clone()], &[b.clone(), b.clone()])
        .unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    assert_eq!(results[1].status, SolverStatus::Solved);

    let x_batch = vec![results[0].x.clone(), results[1].x.clone()];
    let z_batch = vec![results[0].z.clone(), results[1].z.clone()];
    let s_batch = vec![results[0].s.clone(), results[1].s.clone()];

    let upstream = vec![
        UpstreamGradients {
            dx: vec![1.0, 0.0],
            ds: vec![0.0; m],
            dz: vec![0.0; m],
            dz_x: vec![],
        },
        UpstreamGradients {
            dx: vec![0.0, 1.0],
            ds: vec![0.0; m],
            dz: vec![0.0; m],
            dz_x: vec![],
        },
    ];

    // Normal backward
    let normal_grads = solver.backward(&upstream).unwrap();

    // backward_with_data
    let data_grads = solver
        .backward_with_data(
            &upstream,
            &[P_values.clone(), P_values.clone()],
            &[A_values.clone(), A_values.clone()],
            &[q1.clone(), q2.clone()],
            &[b.clone(), b.clone()],
            &x_batch,
            &z_batch,
            &s_batch,
        )
        .unwrap();

    // backward_with_data re-equilibrates from scratch, so floating-point
    // differences in equilibration can cause slightly larger discrepancies.
    let tol = 1e-5;
    for batch_idx in 0..2 {
        let label = format!("batch[{}]", batch_idx);
        assert_vecs_close(
            &normal_grads[batch_idx].dP_values,
            &data_grads[batch_idx].dP_values,
            tol,
            &format!("{} dP_values", label),
        );
        assert_vecs_close(
            &normal_grads[batch_idx].dq,
            &data_grads[batch_idx].dq,
            tol,
            &format!("{} dq", label),
        );
        assert_vecs_close(
            &normal_grads[batch_idx].dA_values,
            &data_grads[batch_idx].dA_values,
            tol,
            &format!("{} dA_values", label),
        );
        assert_vecs_close(
            &normal_grads[batch_idx].db,
            &data_grads[batch_idx].db,
            tol,
            &format!("{} db", label),
        );
    }
}
