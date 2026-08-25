#![allow(non_snake_case)]

//! Box Constraint Example using moreau's high-level API
//!
//! Solves:
//!   minimize    (1/2)||x||^2 + 1'x
//!   subject to  Ax + s = b, s >= 0  (box constraints via A = [I; -I])

use moreau::algebra::{BlockConcatenate, CscMatrix, MatrixMathMut};
use moreau::solver::api::{Cones, Settings, Solver};

fn problem_data() -> (CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>) {
    let n = 200;

    // P = I (identity)
    let P = CscMatrix::identity(n);

    // Construct A = [I; -I] for box constraints
    let I1 = CscMatrix::<f64>::identity(n);
    let mut I2 = CscMatrix::<f64>::identity(n);
    I2.negate();

    let A = CscMatrix::vcat(&I1, &I2).unwrap();

    let q = vec![1.; n];
    let b = vec![1.; 2 * n];

    (P, q, A, b)
}

fn main() {
    let (P, q, A, b) = problem_data();

    // 2*n inequality constraints (box constraints)
    let cones = Cones {
        num_nonneg: b.len(),
        ..Default::default()
    };

    let mut settings = Settings::default();
    settings.max_iter = 50;
    settings.ipm.equilibrate_enable = true;

    // Create solver with all problem data
    let mut solver = Solver::new(&P, &q, &A, &b, &cones, Some(settings)).unwrap();

    // Solve
    let solution = solver.solve().unwrap();

    println!("Solution = {:?}", solution.x);
}
