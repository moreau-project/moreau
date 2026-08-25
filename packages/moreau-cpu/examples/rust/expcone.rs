#![allow(non_snake_case)]

//! Exponential Cone Example using moreau's high-level API
//!
//! Solves:
//!   max  x
//!   s.t. y * exp(x / y) <= z
//!        y == 1, z == exp(5)
//!
//! The optimal x should be 5.

use moreau::algebra::CscMatrix;
use moreau::solver::api::{Cones, Settings, Solver};

fn main() {
    // Problem data
    let P = CscMatrix::zeros((3, 3));

    let A = CscMatrix::from(&[
        [-1., 0., 0.],
        [0., -1., 0.],
        [0., 0., -1.],
        [0., 1., 0.],
        [0., 0., 1.],
    ]);

    let q = vec![-1., 0., 0.];
    let b = vec![0., 0., 0., 1., (5f64).exp()];

    // One exponential cone (3D) + two equality constraints
    let cones = Cones {
        num_exp: 1,
        num_zero: 2,
        ..Default::default()
    };

    let settings = Settings {
        verbose: true,
        ..Default::default()
    };

    // Create solver with all problem data
    let mut solver = Solver::new(&P, &q, &A, &b, &cones, Some(settings)).unwrap();

    // Solve
    let solution = solver.solve().unwrap();

    println!("Solution = {:?}", solution.x);
    // Expected: x ≈ 5.0
}
