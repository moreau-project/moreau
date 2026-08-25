#![allow(non_snake_case)]

//! QP Example using moreau's high-level API
//!
//! Solves:
//!   minimize    (1/2)x'Px + q'x
//!   subject to  Ax + s = b, s in K
//!
//! where K is the product of a zero cone (equality) and nonnegative cone.

use moreau::algebra::CscMatrix;
use moreau::solver::api::{Cones, Solver};

fn main() {
    // Quadratic objective: P = diag([6, 4])
    let P = CscMatrix::from(&[
        [6., 0.], //
        [0., 4.], //
    ]);

    // Constraint matrix:
    // [1, -2]   equality (zero cone)
    // [1,  0]   x <= 1  (nonnegative cone)
    // [0,  1]   y <= 1
    // [-1, 0]   x >= -1
    // [0, -1]   y >= -1
    let A = CscMatrix::from(&[
        [1., -2.], // <-- LHS of equality constraint
        [1., 0.],  // <-- LHS of inequality constraint (upper bound)
        [0., 1.],  // <-- LHS of inequality constraint (upper bound)
        [-1., 0.], // <-- LHS of inequality constraint (lower bound)
        [0., -1.], // <-- LHS of inequality constraint (lower bound)
    ]);

    let q = vec![-1., -4.];
    let b = vec![0., 1., 1., 1., 1.];

    // 1 equality constraint (zero cone), 4 inequality constraints (nonneg cone)
    let cones = Cones {
        num_zero: 1,
        num_nonneg: 4,
        ..Default::default()
    };

    // Create solver with all problem data
    let mut solver = Solver::new(&P, &q, &A, &b, &cones, None).unwrap();

    // Solve
    let solution = solver.solve().unwrap();

    println!("Solution (x)    = {:?}", solution.x);
    println!("Multipliers (z) = {:?}", solution.z);
    println!("Slacks (s)      = {:?}", solution.s);
}
