#![allow(non_snake_case)]

//! Power Cone Example using moreau's high-level API
//!
//! Solves:
//!   max  x1^0.6 * y^0.4 + x2^0.1
//!   s.t. x1, y, x2 >= 0
//!        x1 + 2y + 3x2 == 3
//!
//! This is equivalent to:
//!   max z1 + z2
//!   s.t. (x1, y, z1) in K_pow(0.6)
//!        (x2, 1, z2) in K_pow(0.1)
//!        x1 + 2y + 3x2 == 3
//!
//! Variables: [x1, y, z1, x2, w, z2] where w is fixed to 1 for second cone.
//!
//! Cone ordering in moreau (differs from Clarabel examples):
//!   1. Zero cone (equalities) - 2 rows
//!   2. Power cones - 6 rows (2 cones × 3)

use moreau::algebra::CscMatrix;
use moreau::solver::api::{Cones, Settings, Solver};

fn main() {
    let P = CscMatrix::zeros((6, 6));

    // Constraint matrix A (8 rows × 6 cols)
    // Rows 0-1: Zero cone (equality constraints)
    //   Row 0: x1 + 2y + 3x2 = 3
    //   Row 1: w = 1 (fix w for second power cone)
    // Rows 2-4: First power cone (x1, y, z1)
    // Rows 5-7: Second power cone (x2, w, z2)
    let A = CscMatrix::from(&[
        // Zero cone rows (equalities)
        [1., 2., 0., 3., 0., 0.], // x1 + 2y + 3x2 = 3
        [0., 0., 0., 0., 1., 0.], // w = 1
        // Power cone 1: (x1, y, z1)
        [-1., 0., 0., 0., 0., 0.], // -x1 + s0 = 0 => s0 = x1
        [0., -1., 0., 0., 0., 0.], // -y + s1 = 0  => s1 = y
        [0., 0., -1., 0., 0., 0.], // -z1 + s2 = 0 => s2 = z1
        // Power cone 2: (x2, w, z2)
        [0., 0., 0., -1., 0., 0.], // -x2 + s3 = 0 => s3 = x2
        [0., 0., 0., 0., -1., 0.], // -w + s4 = 0  => s4 = w
        [0., 0., 0., 0., 0., -1.], // -z2 + s5 = 0 => s5 = z2
    ]);

    // Objective: max z1 + z2 = min -z1 - z2
    let q = vec![0., 0., -1., 0., 0., -1.];

    // RHS: equalities then zeros for power cones
    let b = vec![3., 1., 0., 0., 0., 0., 0., 0.];

    // Moreau cone order: zero first, then power cones
    let cones = Cones {
        num_zero: 2,
        power_alphas: vec![0.6, 0.1],
        ..Default::default()
    };

    let settings = Settings {
        verbose: true,
        max_iter: 100,
        ..Default::default()
    };

    // Create solver with all problem data
    let mut solver = Solver::new(&P, &q, &A, &b, &cones, Some(settings)).unwrap();

    // Solve
    let solution = solver.solve().unwrap();

    println!("\nStatus = {:?}", solution.status);
    println!("Solution x = {:?}", solution.x);
    println!("  x1 = {:.6}", solution.x[0]);
    println!("  y  = {:.6}", solution.x[1]);
    println!("  z1 = {:.6} (x1^0.6 * y^0.4)", solution.x[2]);
    println!("  x2 = {:.6}", solution.x[3]);
    println!("  w  = {:.6} (should be 1)", solution.x[4]);
    println!("  z2 = {:.6} (x2^0.1)", solution.x[5]);
    println!("Objective (z1 + z2) = {:.6}", solution.x[2] + solution.x[5]);
}
