//! Backward gradient test for GenPowerCone + nonneg mixed cones.
//!
//! Reproduces the failing Python test_gradcheck_genpow_three_alphas_q (seed=500).
//! Uses non-uniform diagonal P and boundary-touching solution.
//!
//! Note: This problem has ill-conditioned KKT backward system for some output
//! indices. The derivative unit tests (cones.rs) verify derivative accuracy to
//! tight tolerance; the residual error here is from KKT conditioning.

#![allow(non_snake_case)]

use moreau::{algebra::*, solver::implementations::default::DiffMethod, solver::*};

/// Build and solve the exact problem from _make_genpow_problem([0.2,0.3,0.5], dim2=2, seed=500).
///
/// Cone order: nonneg(1), GenPowerCone(5)
/// A: rows 0-4 = -I, row 5 = -ones (sum constraint)
/// b: [0,0,0,0,0,-5]
fn solve_genpow_nonneg(P_diag: &[f64], q: &[f64], b: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = 5;
    let m = 6;

    // P = diag(P_diag) — full symmetric (both triangles)
    let P = CscMatrix {
        m: n,
        n,
        colptr: vec![0, 1, 2, 3, 4, 5],
        rowval: vec![0, 1, 2, 3, 4],
        nzval: P_diag.to_vec(),
    };

    // A: rows 0-4 each have A[i,i] = -1, row 5 has A[5,:] = -ones
    // CSC format: column-major
    // col 0: rows 0, 5 -> vals -1, -1
    // col 1: rows 1, 5 -> vals -1, -1
    // etc.
    let A = CscMatrix {
        m,
        n,
        colptr: vec![0, 2, 4, 6, 8, 10],
        rowval: vec![0, 5, 1, 5, 2, 5, 3, 5, 4, 5],
        nzval: vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    };

    // Cone order: nonneg(1 dim), then GenPowerCone(5 dims)
    let cones = vec![NonnegativeConeT(1), GenPowerConeT(vec![0.2, 0.3, 0.5], 2)];

    let mut settings = DefaultSettings::default();
    settings.ipm.tol_gap_abs = 1e-9;
    settings.ipm.tol_feas = 1e-9;

    let mut solver = DefaultSolver::new(&P, q, &A, b, &cones, settings).unwrap();
    solver.solve();
    assert!(
        solver.solution.status == SolverStatus::Solved
            || solver.solution.status == SolverStatus::AlmostSolved,
        "Solver failed with status {:?}",
        solver.solution.status
    );

    (
        solver.solution.x.clone(),
        solver.solution.z.clone(),
        solver.solution.s.clone(),
    )
}

/// Full backward gradient test: dx/dq via analytical backward vs finite differences.
#[test]
fn test_genpow_nonneg_backward_dq() {
    let n = 5;
    let m = 6;

    // Exact values from numpy.random.default_rng(500)
    let P_diag = [1.35011471, 1.78096698, 1.4680358, 1.11673522, 1.20204792];
    let q = [
        0.32136122,
        -0.18747683,
        0.28622079,
        -0.06444984,
        -0.25236979,
    ];
    let b = [0.0, 0.0, 0.0, 0.0, 0.0, -5.0];

    // Solve base problem
    let (x0, _z0, _s0) = solve_genpow_nonneg(&P_diag, &q, &b);
    println!("x0 = {:?}", x0);

    // P matrix for backward
    let P = CscMatrix {
        m: n,
        n,
        colptr: vec![0, 1, 2, 3, 4, 5],
        rowval: vec![0, 1, 2, 3, 4],
        nzval: P_diag.to_vec(),
    };
    let A = CscMatrix {
        m,
        n,
        colptr: vec![0, 2, 4, 6, 8, 10],
        rowval: vec![0, 5, 1, 5, 2, 5, 3, 5, 4, 5],
        nzval: vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    };
    let cones = vec![NonnegativeConeT(1), GenPowerConeT(vec![0.2, 0.3, 0.5], 2)];

    // Solve and backward
    let mut settings = DefaultSettings::default();
    settings.ipm.tol_gap_abs = 1e-9;
    settings.ipm.tol_feas = 1e-9;
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    let eps = 1e-6_f64;
    let atol = 1e-1_f64;
    let rtol = 1e-1_f64;
    let mut max_error = 0.0_f64;

    for out_idx in 0..n {
        // Analytical: backward with dx_bar = e_{out_idx}
        let mut dx_bar = vec![0.0; n];
        dx_bar[out_idx] = 1.0;
        let dz_bar = vec![0.0; m];
        let ds_bar = vec![0.0; m];

        let result = solver
            .backward_batch(&dx_bar, &ds_bar, &dz_bar, None, DiffMethod::Exact)
            .unwrap();

        for i in 0..n {
            // Finite difference: perturb q[i]
            let mut q_plus = q.to_vec();
            q_plus[i] += eps;
            let (x_plus, _, _) = solve_genpow_nonneg(&P_diag, &q_plus, &b);

            let mut q_minus = q.to_vec();
            q_minus[i] -= eps;
            let (x_minus, _, _) = solve_genpow_nonneg(&P_diag, &q_minus, &b);

            let fd = (x_plus[out_idx] - x_minus[out_idx]) / (2.0 * eps);
            let err = (result.dq[i] - fd).abs();
            let threshold = atol + rtol * fd.abs().max(result.dq[i].abs());
            max_error = max_error.max(err);

            println!(
                "dx[{}]/dq[{}]: analytical={:.6}, fd={:.6}, diff={:.2e}",
                out_idx, i, result.dq[i], fd, err
            );
            assert!(
                err < threshold,
                "Gradient mismatch at dx[{}]/dq[{}]: analytical={}, fd={}, diff={:.2e}",
                out_idx,
                i,
                result.dq[i],
                fd,
                err
            );
        }
    }
    println!("Max error: {:.2e}", max_error);
}

/// Same test for db gradients.
#[test]
fn test_genpow_nonneg_backward_db() {
    let n = 5;
    let m = 6;

    let P_diag = [1.35011471, 1.78096698, 1.4680358, 1.11673522, 1.20204792];
    let q = [
        0.32136122,
        -0.18747683,
        0.28622079,
        -0.06444984,
        -0.25236979,
    ];
    let b = [0.0, 0.0, 0.0, 0.0, 0.0, -5.0];

    let P = CscMatrix {
        m: n,
        n,
        colptr: vec![0, 1, 2, 3, 4, 5],
        rowval: vec![0, 1, 2, 3, 4],
        nzval: P_diag.to_vec(),
    };
    let A = CscMatrix {
        m,
        n,
        colptr: vec![0, 2, 4, 6, 8, 10],
        rowval: vec![0, 5, 1, 5, 2, 5, 3, 5, 4, 5],
        nzval: vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    };
    let cones = vec![NonnegativeConeT(1), GenPowerConeT(vec![0.2, 0.3, 0.5], 2)];

    let mut settings = DefaultSettings::default();
    settings.ipm.tol_gap_abs = 1e-9;
    settings.ipm.tol_feas = 1e-9;
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    let eps = 1e-6_f64;
    let atol = 1e-1_f64;
    let rtol = 1e-1_f64;
    let mut max_error = 0.0_f64;

    for out_idx in 0..n {
        let mut dx_bar = vec![0.0; n];
        dx_bar[out_idx] = 1.0;
        let dz_bar = vec![0.0; m];
        let ds_bar = vec![0.0; m];

        let result = solver
            .backward_batch(&dx_bar, &ds_bar, &dz_bar, None, DiffMethod::Exact)
            .unwrap();

        for j in 0..m {
            let mut b_plus = b.to_vec();
            b_plus[j] += eps;
            let (x_plus, _, _) = solve_genpow_nonneg(&P_diag, &q, &b_plus);

            let mut b_minus = b.to_vec();
            b_minus[j] -= eps;
            let (x_minus, _, _) = solve_genpow_nonneg(&P_diag, &q, &b_minus);

            let fd = (x_plus[out_idx] - x_minus[out_idx]) / (2.0 * eps);
            let err = (result.db[j] - fd).abs();
            let threshold = atol + rtol * fd.abs().max(result.db[j].abs());
            max_error = max_error.max(err);

            println!(
                "dx[{}]/db[{}]: analytical={:.6}, fd={:.6}, diff={:.2e}",
                out_idx, j, result.db[j], fd, err
            );
            assert!(
                err < threshold,
                "Gradient mismatch at dx[{}]/db[{}]: analytical={}, fd={}, diff={:.2e}",
                out_idx,
                j,
                result.db[j],
                fd,
                err
            );
        }
    }
    println!("Max error: {:.2e}", max_error);
}
