#![allow(non_snake_case)]

//! SOCP Example using moreau's high-level API
//!
//! Solves a second-order cone program.

use moreau::algebra::CscMatrix;
use moreau::solver::api::{Cones, Solver};

fn main() {
    // Problem data
    let P = CscMatrix::from(&[
        [0., 0.], //
        [0., 2.], //
    ]);

    let A = CscMatrix::from(&[
        [0., 0.],  //
        [-2., 0.], //
        [0., -1.], //
    ]);

    let q = vec![0., 0.];
    let b = vec![1., -2., -2.];

    // One second-order cone of dimension 3
    let cones = Cones {
        soc_dims: vec![3],
        ..Default::default()
    };

    // Create solver with all problem data
    let mut solver = Solver::new(&P, &q, &A, &b, &cones, None).unwrap();

    // Solve
    let solution = solver.solve().unwrap();

    println!("Solution = {:?}", solution.x);
}
