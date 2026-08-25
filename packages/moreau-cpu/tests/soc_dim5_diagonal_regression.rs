#![allow(non_snake_case)]

//! Regression: small diagonal SOC dim=5 QP the CPU solver must solve.

use moreau::{algebra::*, solver::*};

#[test]
fn soc_dim5_diagonal_qp_solves() {
    let n = 5;

    // P = 2*I (diagonal CSC)
    let P = CscMatrix::new(
        n,
        n,
        vec![0, 1, 2, 3, 4, 5],
        vec![0, 1, 2, 3, 4],
        vec![2.0, 2.0, 2.0, 2.0, 2.0],
    );

    // A = -I (diagonal CSC)
    let A = CscMatrix::new(
        n,
        n,
        vec![0, 1, 2, 3, 4, 5],
        vec![0, 1, 2, 3, 4],
        vec![-1.0, -1.0, -1.0, -1.0, -1.0],
    );

    // q chosen so solution is in SOC interior
    let q = vec![-3.0, -0.5, -0.3, -0.2, -0.1];
    let b = vec![0.0, 0.0, 0.0, 0.0, 0.0];

    let cones = vec![SecondOrderConeT(5)];

    let settings = DefaultSettings::<f64>::default();

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert!(
        solver.solution.status == SolverStatus::Solved
            || solver.solution.status == SolverStatus::AlmostSolved,
        "expected Solved/AlmostSolved, got {:?}",
        solver.solution.status
    );

    // s must lie in the second-order cone: s[0] >= ||s[1:]||.
    let s = &solver.solution.s;
    let tail_norm: f64 = s[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        s[0] >= tail_norm - 1e-6,
        "SOC violated: s[0]={}, ||s[1:]||={}",
        s[0],
        tail_norm
    );
}
