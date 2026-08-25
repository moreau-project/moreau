#![allow(non_snake_case)]

//! LP Example using moreau's high-level API
//!
//! Solves:
//!   minimize    c'x
//!   subject to  Ax + s = b, s >= 0

use moreau::algebra::CscMatrix;
use moreau::solver::api::{Cones, Settings, Solver};

fn main() {
    // No quadratic term (P = 0)
    let P: CscMatrix<f64> = CscMatrix::zeros((2, 2));

    // A 2-d box constraint: -1 <= x,y <= 1
    // Represented as: [I; -I] * [x;y] <= [1;1;1;1]
    let A = CscMatrix::from(&[
        [1., 0.],  //  x <= 1
        [0., 1.],  //  y <= 1
        [-1., 0.], // -x <= 1  =>  x >= -1
        [0., -1.], // -y <= 1  =>  y >= -1
    ]);

    let q = vec![1., -1.];
    let b = vec![1.; 4];

    let cones = Cones {
        num_nonneg: 4,
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
