//! Regression: PSD(2) QP via the batched CompiledSolver.
//! min 0.5*||x||^2 + q'x s.t. x + s = 0, s in PSD(2)
//! svec: 3 vars, P=I, A=I, q=[-3,0,-2], b=[0,0,0]

#![allow(non_snake_case)]
#![cfg(feature = "sdp")]

use moreau::solver::implementations::default::{CompiledSolver, DefaultSettings};
use moreau::solver::{SolverStatus, SupportedConeT};

#[test]
fn psd2_qp_compiled_solver_solves() {
    let n = 3;
    let m = 3;

    // P = I (CSR: diagonal)
    let P_row_offsets = vec![0, 1, 2, 3];
    let P_col_indices = vec![0, 1, 2];
    let P_values = vec![1.0, 1.0, 1.0];

    // A = I (CSR: diagonal)
    let A_row_offsets = vec![0, 1, 2, 3];
    let A_col_indices = vec![0, 1, 2];
    let A_values = vec![1.0, 1.0, 1.0];

    let q = vec![-3.0, 0.0, -2.0];
    let b = vec![0.0, 0.0, 0.0];

    let cones = vec![SupportedConeT::PSDTriangleConeT(2)];

    let settings = DefaultSettings::<f64>::default();

    let mut solver = CompiledSolver::new(
        n,
        m,
        &P_row_offsets,
        &P_col_indices,
        &A_row_offsets,
        &A_col_indices,
        &cones,
        settings,
        1,
        false,
    )
    .expect("construction failed");

    solver.setup(&[P_values], &[A_values]);
    let solutions = solver.solve(&[q], &[b]).expect("solve failed");

    let sol = &solutions[0];
    assert_eq!(sol.status, SolverStatus::Solved);
    // x should be near 0 (PSD constraint forces negative semidefiniteness)
    for i in 0..n {
        assert!(
            sol.x[i].abs() < 1e-4,
            "x[{}] = {} should be ~0",
            i,
            sol.x[i]
        );
    }
}
