//! Test for power cone dz backward gradients
//!
//! This test verifies that the backward pass correctly computes dz/dq and dz/db
//! for power cone problems.
//!
//! NOTE: Uses P = I (not P = 0). With P = 0 (LP), dual variables are not uniquely
//! differentiable because the F matrix in the HSDE formulation becomes rank-deficient.
//! See POWER_CONE_BACKWARD_FINDINGS.md for details.

#![allow(non_snake_case)]

use moreau::{algebra::*, solver::*};

/// Compare analytical backward gradients with finite differences for power cone
#[test]
fn test_power_cone_dz_backward() {
    // Problem: min (1/2)x'Px + q'x s.t. Ax + s = b, s in PowerCone(0.3) + ZeroCone
    // We use zero cones to fix x[1] = 2 and x[2] = 1, then optimize x[0]
    let n = 3;
    let m = 5;

    // P = I (PD to ensure unique, differentiable dual solution)
    let P = CscMatrix::identity(n);

    // q = [-1, 0, 0] - maximize x[0]
    let q_base = vec![-1.0, 0.0, 0.0];

    // A = [[0, 1, 0], [0, 0, 1], [0, -1, 0], [0, 0, -1], [-1, 0, 0]]
    // First two rows: zero cone to fix x[1], x[2]
    // Last three rows: power cone on (-x[1], -x[2], -x[0])
    let A = CscMatrix::from(&[
        [0.0, 1.0, 0.0],  // x[1] = b[0]
        [0.0, 0.0, 1.0],  // x[2] = b[1]
        [0.0, -1.0, 0.0], // power cone
        [0.0, 0.0, -1.0], // power cone
        [-1.0, 0.0, 0.0], // power cone
    ]);

    let b_base = vec![2.0, 1.0, 0.0, 0.0, 0.0];

    let cones = vec![ZeroConeT(2), PowerConeT(0.3)];

    let eps = 1e-6;

    // Helper to solve problem and return (x, z, s)
    let solve = |q: &[f64], b: &[f64]| {
        let settings = DefaultSettings::default();
        let mut solver = DefaultSolver::new(&P, q, &A, b, &cones, settings).unwrap();
        solver.solve();
        (
            solver.solution.x.clone(),
            solver.solution.z.clone(),
            solver.solution.s.clone(),
        )
    };

    // Solve base problem
    let (x0, z0, s0) = solve(&q_base, &b_base);

    println!("x = {:?}", x0);
    println!("z = {:?}", z0);
    println!("s = {:?}", s0);

    // Compute finite difference dz/dq
    let mut fd_dz_dq = vec![vec![0.0; n]; m];
    for j in 0..n {
        let mut q_plus = q_base.clone();
        q_plus[j] += eps;
        let (_, z_plus, _) = solve(&q_plus, &b_base);

        let mut q_minus = q_base.clone();
        q_minus[j] -= eps;
        let (_, z_minus, _) = solve(&q_minus, &b_base);

        for i in 0..m {
            fd_dz_dq[i][j] = (z_plus[i] - z_minus[i]) / (2.0 * eps);
        }
    }

    // Compute analytical dz/dq using backward pass
    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &q_base, &A, &b_base, &cones, settings).unwrap();
    solver.solve();

    let mut analytical_dz_dq = vec![vec![0.0; n]; m];
    for i in 0..m {
        let mut dz_bar = vec![0.0; m];
        dz_bar[i] = 1.0;
        let dx_bar = vec![0.0; n];
        let ds_bar = vec![0.0; m];

        let result = solver
            .backward_batch(&dx_bar, &ds_bar, &dz_bar, None, DiffMethod::Exact)
            .unwrap();
        for j in 0..n {
            analytical_dz_dq[i][j] = result.dq[j];
        }
    }

    // Compare
    println!("\n=== dz/dq ===");
    let mut max_error = 0.0f64;
    for i in 0..m {
        for j in 0..n {
            let err = (analytical_dz_dq[i][j] - fd_dz_dq[i][j]).abs();
            max_error = max_error.max(err);
        }
        println!(
            "dz[{}]/dq: analytical={:?}, fd={:?}",
            i, analytical_dz_dq[i], fd_dz_dq[i]
        );
    }
    println!("Max error dz/dq: {:.2e}", max_error);

    // Compute finite difference dz/db
    let mut fd_dz_db = vec![vec![0.0; m]; m];
    for j in 0..m {
        let mut b_plus = b_base.clone();
        b_plus[j] += eps;
        let (_, z_plus, _) = solve(&q_base, &b_plus);

        let mut b_minus = b_base.clone();
        b_minus[j] -= eps;
        let (_, z_minus, _) = solve(&q_base, &b_minus);

        for i in 0..m {
            fd_dz_db[i][j] = (z_plus[i] - z_minus[i]) / (2.0 * eps);
        }
    }

    // Compute analytical dz/db
    let mut analytical_dz_db = vec![vec![0.0; m]; m];
    for i in 0..m {
        let mut dz_bar = vec![0.0; m];
        dz_bar[i] = 1.0;
        let dx_bar = vec![0.0; n];
        let ds_bar = vec![0.0; m];

        let result = solver
            .backward_batch(&dx_bar, &ds_bar, &dz_bar, None, DiffMethod::Exact)
            .unwrap();
        for j in 0..m {
            analytical_dz_db[i][j] = result.db[j];
        }
    }

    // Compare
    println!("\n=== dz/db ===");
    let mut max_error_db = 0.0f64;
    for i in 0..m {
        for j in 0..m {
            let err = (analytical_dz_db[i][j] - fd_dz_db[i][j]).abs();
            max_error_db = max_error_db.max(err);
        }
        println!(
            "dz[{}]/db: analytical={:?}, fd={:?}",
            i, analytical_dz_db[i], fd_dz_db[i]
        );
    }
    println!("Max error dz/db: {:.2e}", max_error_db);

    // Assert tolerance
    let tol = 1e-3;
    assert!(
        max_error < tol,
        "dz/dq max error {:.2e} exceeds tolerance {:.2e}",
        max_error,
        tol
    );
    assert!(
        max_error_db < tol,
        "dz/db max error {:.2e} exceeds tolerance {:.2e}",
        max_error_db,
        tol
    );
}
