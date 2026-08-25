#![allow(non_snake_case)]

//! Batch Solver Example using moreau's two-step API pattern
//!
//! Demonstrates solving multiple QP problems in parallel using CompiledSolver.
//! Each problem shares the same structure (sparsity pattern) but has different values.
//!
//! Problem form:
//!   minimize    (1/2)x'Px + q'x
//!   subject to  Ax + s = b, s in K
//!
//! All batch problems must share the same dimensions (n, m) and sparsity patterns.

use moreau::solver::implementations::default::{BatchProblem, CompiledSolver, DefaultSettings};
use moreau::solver::SupportedConeT;

fn main() {
    // === Step 1: Define shared problem structure ===

    // All problems: 2 variables, 1 equality constraint
    let n = 2;
    let m = 1;

    // P matrix structure: diagonal 2x2 (CSR format)
    // P = [[p1, 0], [0, p2]] - values will vary per problem
    let P_row_offsets = vec![0, 1, 2];
    let P_col_indices = vec![0, 1];

    // A matrix structure: [1, 1] (CSR format)
    // Constraint: x[0] + x[1] = b
    let A_row_offsets = vec![0, 2];
    let A_col_indices = vec![0, 1];

    // One equality constraint (zero cone)
    let cones: Vec<SupportedConeT<f64>> = vec![SupportedConeT::ZeroConeT(1)];

    // Solver settings
    let mut settings = DefaultSettings::default();
    settings.verbose = true; // Show batch summary

    let compiled_solver = CompiledSolver::new(
        n,
        m,
        &P_row_offsets,
        &P_col_indices,
        &A_row_offsets,
        &A_col_indices,
        &cones,
        settings,
        4,
        false,
    )
    .expect("Failed to create batch solver");

    // === Step 2: Create batch of problems with different values ===

    // Problem 1: P = diag(2, 2), q = [-1, -1], A = [1, 1], b = 2
    // Solution: x = [1, 1] (minimize ||x - [0.5, 0.5]||^2 s.t. x[0]+x[1]=2)
    let problem1 = BatchProblem {
        P_values: vec![2.0, 2.0],
        q: vec![-1.0, -1.0],
        A_values: vec![1.0, 1.0],
        b: vec![2.0],
    };

    // Problem 2: Same structure, different b
    // Solution: x = [2, 2] (minimize ||x - [0.5, 0.5]||^2 s.t. x[0]+x[1]=4)
    let problem2 = BatchProblem {
        P_values: vec![2.0, 2.0],
        q: vec![-1.0, -1.0],
        A_values: vec![1.0, 1.0],
        b: vec![4.0],
    };

    // Problem 3: Different P scaling
    // Solution: x = [0.6, 2.4] (x[0] has higher cost, so solution favors x[1])
    let problem3 = BatchProblem {
        P_values: vec![10.0, 2.0],
        q: vec![0.0, 0.0],
        A_values: vec![1.0, 1.0],
        b: vec![3.0],
    };

    // Problem 4: Different q (linear cost)
    let problem4 = BatchProblem {
        P_values: vec![2.0, 2.0],
        q: vec![2.0, -4.0], // Favor increasing x[1]
        A_values: vec![1.0, 1.0],
        b: vec![2.0],
    };

    let problems = vec![problem1, problem2, problem3, problem4];

    // === Step 3: Solve batch in parallel ===

    println!("\nSolving {} problems in parallel...\n", problems.len());

    let solutions = compiled_solver
        .solve_batch_parallel(&problems)
        .expect("Batch solve failed");

    // === Display results ===

    println!("\n=== Results ===\n");
    for (i, sol) in solutions.iter().enumerate() {
        println!(
            "Problem {}: x = [{:.4}, {:.4}], obj = {:.6}, status = {:?}",
            i + 1,
            sol.x[0],
            sol.x[1],
            sol.obj_val,
            sol.status
        );
    }

    // Verify expected solutions
    println!("\n=== Verification ===\n");

    // Problem 1: x = [1, 1]
    assert!((solutions[0].x[0] - 1.0).abs() < 1e-4, "Problem 1 failed");
    assert!((solutions[0].x[1] - 1.0).abs() < 1e-4, "Problem 1 failed");
    println!("Problem 1: PASS (x = [1, 1] as expected)");

    // Problem 2: x = [2, 2]
    assert!((solutions[1].x[0] - 2.0).abs() < 1e-4, "Problem 2 failed");
    assert!((solutions[1].x[1] - 2.0).abs() < 1e-4, "Problem 2 failed");
    println!("Problem 2: PASS (x = [2, 2] as expected)");

    // Problem 3: x[0] + x[1] = 3, x[0] < x[1] due to higher P[0,0]
    assert!(
        (solutions[2].x[0] + solutions[2].x[1] - 3.0).abs() < 1e-4,
        "Problem 3 constraint failed"
    );
    assert!(
        solutions[2].x[0] < solutions[2].x[1],
        "Problem 3: x[0] should be less than x[1]"
    );
    println!("Problem 3: PASS (constraint satisfied, x[0] < x[1])");

    // Problem 4: x[0] + x[1] = 2, x[1] > x[0] due to negative q[1]
    assert!(
        (solutions[3].x[0] + solutions[3].x[1] - 2.0).abs() < 1e-4,
        "Problem 4 constraint failed"
    );
    assert!(
        solutions[3].x[1] > solutions[3].x[0],
        "Problem 4: x[1] should be greater than x[0]"
    );
    println!("Problem 4: PASS (constraint satisfied, x[1] > x[0])");

    println!("\nAll batch problems solved correctly!");
    println!("Total batch time: {:.4}s", solutions[0].solve_time);
}
