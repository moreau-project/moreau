//! Tests for warm starting in the CPU solver (CompiledSolver).
//!
//! These tests verify:
//! 1. Warm start from a previous solution converges in fewer iterations
//! 2. Warm start with perturbed problem data works
//! 3. Warm start with SOC cones
//! 4. Bad warm start still converges
//! 5. Validation: partial warm start raises error
//! 6. Validation: wrong dimensions raises error
//! 7. Batched warm start

#![allow(non_snake_case)]

use moreau::solver::*;

// ============================================================================
// HELPERS
// ============================================================================

fn is_close(a: f64, b: f64, atol: f64) -> bool {
    (a - b).abs() <= atol
}

/// Simple QP: minimize x'Px + q'x
/// subject to x + y = 1, x >= 0, y >= 0, x <= 1, y <= 1
fn simple_qp() -> (
    Vec<usize>, // P_row_offsets
    Vec<usize>, // P_col_indices
    Vec<f64>,   // P_values
    Vec<usize>, // A_row_offsets
    Vec<usize>, // A_col_indices
    Vec<f64>,   // A_values
    Vec<SupportedConeT<f64>>,
    usize, // n
    usize, // m
) {
    let n = 2;
    let m = 5;

    // P = diag(2, 2) in CSR
    let P_row_offsets = vec![0, 1, 2];
    let P_col_indices = vec![0, 1];
    let P_values = vec![2.0, 2.0];

    // A in CSR (5 rows, 2 cols):
    // Row 0: [1, 1]     (equality: x + y = 1)
    // Row 1: [-1, 0]    (nonneg: x >= 0)
    // Row 2: [0, -1]    (nonneg: y >= 0)
    // Row 3: [1, 0]     (nonneg: x <= 1)
    // Row 4: [0, 1]     (nonneg: y <= 1)
    let A_row_offsets = vec![0, 2, 3, 4, 5, 6];
    let A_col_indices = vec![0, 1, 0, 1, 0, 1];
    let A_values = vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0];

    let cones = vec![ZeroConeT(1), NonnegativeConeT(4)];

    (
        P_row_offsets,
        P_col_indices,
        P_values,
        A_row_offsets,
        A_col_indices,
        A_values,
        cones,
        n,
        m,
    )
}

/// SOC problem: minimize x1 subject to ||(x2, x3)|| <= x1
fn soc_problem() -> (
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
    usize,
    usize,
) {
    let n = 3;
    let m = 3;

    // P = 0 (LP-like objective)
    let P_row_offsets = vec![0, 0, 0, 0];
    let P_col_indices = vec![];
    let P_values = vec![];

    // A = -I (so constraint is -x + s = 0, i.e., s = x)
    let A_row_offsets = vec![0, 1, 2, 3];
    let A_col_indices = vec![0, 1, 2];
    let A_values = vec![-1.0, -1.0, -1.0];

    let cones = vec![SecondOrderConeT(3)];

    (
        P_row_offsets,
        P_col_indices,
        P_values,
        A_row_offsets,
        A_col_indices,
        A_values,
        cones,
        n,
        m,
    )
}

fn make_compiled_solver(
    P_ro: &[usize],
    P_ci: &[usize],
    A_ro: &[usize],
    A_ci: &[usize],
    cones: &[SupportedConeT<f64>],
    n: usize,
    m: usize,
    batch_size: usize,
) -> CompiledSolver<f64> {
    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    CompiledSolver::new(
        n, m, P_ro, P_ci, A_ro, A_ci, cones, settings, batch_size, false,
    )
    .unwrap()
}

// ============================================================================
// WARM START TESTS
// ============================================================================

/// Warm starting from a cold-solved solution should converge in fewer iterations
#[test]
fn test_warm_start_fewer_iterations() {
    let (P_ro, P_ci, P_val, A_ro, A_ci, A_val, cones, n, m) = simple_qp();

    let q = vec![1.0, -1.0];
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0];

    // Cold solve
    let mut solver_cold = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, 1);
    solver_cold.setup(&[P_val.clone()], &[A_val.clone()]);
    let sol_cold = solver_cold.solve(&[q.clone()], &[b.clone()]).unwrap();

    assert_eq!(sol_cold[0].status, SolverStatus::Solved);
    let iters_cold = sol_cold[0].iterations;

    // Warm solve from cold solution
    let mut solver_warm = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, 1);
    solver_warm.setup(&[P_val.clone()], &[A_val.clone()]);
    let sol_warm = solver_warm
        .solve_with_warm_start(
            &[q.clone()],
            &[b.clone()],
            Some(&[sol_cold[0].x.clone()]),
            Some(&[sol_cold[0].z.clone()]),
            Some(&[sol_cold[0].s.clone()]),
            None,
        )
        .unwrap();

    assert_eq!(sol_warm[0].status, SolverStatus::Solved);
    let iters_warm = sol_warm[0].iterations;

    assert!(
        iters_warm < iters_cold,
        "Warm start ({} iters) should be faster than cold ({} iters)",
        iters_warm,
        iters_cold
    );

    // Solutions should match
    for i in 0..n {
        assert!(
            is_close(sol_warm[0].x[i], sol_cold[0].x[i], 1e-4),
            "x[{}] mismatch: warm={} vs cold={}",
            i,
            sol_warm[0].x[i],
            sol_cold[0].x[i]
        );
    }
}

/// Warm start from P1's solution should help P2 converge (slightly perturbed q)
#[test]
fn test_warm_start_perturbed_problem() {
    let (P_ro, P_ci, P_val, A_ro, A_ci, A_val, cones, n, m) = simple_qp();

    let q1 = vec![1.0, -1.0];
    let q2 = vec![1.1, -0.9]; // small perturbation
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0];

    // Solve P1 cold
    let mut solver1 = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, 1);
    solver1.setup(&[P_val.clone()], &[A_val.clone()]);
    let sol1 = solver1.solve(&[q1], &[b.clone()]).unwrap();
    assert_eq!(sol1[0].status, SolverStatus::Solved);

    // Solve P2 cold
    let mut solver2_cold = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, 1);
    solver2_cold.setup(&[P_val.clone()], &[A_val.clone()]);
    let sol2_cold = solver2_cold.solve(&[q2.clone()], &[b.clone()]).unwrap();
    assert_eq!(sol2_cold[0].status, SolverStatus::Solved);

    // Solve P2 warm from P1's solution
    let mut solver2_warm = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, 1);
    solver2_warm.setup(&[P_val.clone()], &[A_val.clone()]);
    let sol2_warm = solver2_warm
        .solve_with_warm_start(
            &[q2],
            &[b],
            Some(&[sol1[0].x.clone()]),
            Some(&[sol1[0].z.clone()]),
            Some(&[sol1[0].s.clone()]),
            None,
        )
        .unwrap();
    assert_eq!(sol2_warm[0].status, SolverStatus::Solved);

    // Solutions should match
    for i in 0..n {
        assert!(
            is_close(sol2_warm[0].x[i], sol2_cold[0].x[i], 1e-4),
            "x[{}] mismatch: warm={} vs cold={}",
            i,
            sol2_warm[0].x[i],
            sol2_cold[0].x[i]
        );
    }
}

/// Test warm start with second-order cone
#[test]
fn test_warm_start_soc() {
    let (P_ro, P_ci, P_val, A_ro, A_ci, A_val, cones, n, m) = soc_problem();

    let q = vec![1.0, 0.5, 0.5];
    let b = vec![0.0, 0.0, 0.0];

    // Cold solve
    let mut solver_cold = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, 1);
    solver_cold.setup(&[P_val.clone()], &[A_val.clone()]);
    let sol_cold = solver_cold.solve(&[q.clone()], &[b.clone()]).unwrap();
    assert_eq!(sol_cold[0].status, SolverStatus::Solved);

    // Warm solve
    let mut solver_warm = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, 1);
    solver_warm.setup(&[P_val.clone()], &[A_val.clone()]);
    let sol_warm = solver_warm
        .solve_with_warm_start(
            &[q],
            &[b],
            Some(&[sol_cold[0].x.clone()]),
            Some(&[sol_cold[0].z.clone()]),
            Some(&[sol_cold[0].s.clone()]),
            None,
        )
        .unwrap();
    assert_eq!(sol_warm[0].status, SolverStatus::Solved);

    for i in 0..n {
        assert!(
            is_close(sol_warm[0].x[i], sol_cold[0].x[i], 1e-6),
            "x[{}] mismatch: warm={} vs cold={}",
            i,
            sol_warm[0].x[i],
            sol_cold[0].x[i]
        );
    }
}

/// A bad warm start (somewhat reasonable point) should still converge
#[test]
fn test_warm_start_bad_point_still_converges() {
    let (P_ro, P_ci, P_val, A_ro, A_ci, A_val, cones, n, m) = simple_qp();

    let q = vec![1.0, -1.0];
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0];

    // Cold solve for reference
    let mut solver_ref = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, 1);
    solver_ref.setup(&[P_val.clone()], &[A_val.clone()]);
    let sol_ref = solver_ref.solve(&[q.clone()], &[b.clone()]).unwrap();
    assert_eq!(sol_ref[0].status, SolverStatus::Solved);

    // Warm start with a somewhat reasonable but not optimal point
    let warm_x = vec![0.5, 0.5];
    let warm_z = vec![1.0; m];
    let warm_s = vec![1.0; m];

    let mut solver_warm = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, 1);
    solver_warm.setup(&[P_val.clone()], &[A_val.clone()]);
    let sol_warm = solver_warm
        .solve_with_warm_start(
            &[q],
            &[b],
            Some(&[warm_x]),
            Some(&[warm_z]),
            Some(&[warm_s]),
            None,
        )
        .unwrap();

    assert!(
        sol_warm[0].status == SolverStatus::Solved
            || sol_warm[0].status == SolverStatus::AlmostSolved,
        "Bad warm start should still converge, got {:?}",
        sol_warm[0].status
    );

    for i in 0..n {
        assert!(
            is_close(sol_warm[0].x[i], sol_ref[0].x[i], 1e-5),
            "x[{}] mismatch: warm={} vs ref={}",
            i,
            sol_warm[0].x[i],
            sol_ref[0].x[i]
        );
    }
}

/// Providing only some of warm_x/z/s should raise an error
#[test]
fn test_warm_start_validation_partial() {
    let (P_ro, P_ci, P_val, A_ro, A_ci, A_val, cones, n, m) = simple_qp();

    let q = vec![1.0, -1.0];
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0];

    let mut solver = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, 1);
    solver.setup(&[P_val], &[A_val]);

    // Only warm_x (missing z and s)
    let result = solver.solve_with_warm_start(
        &[q.clone()],
        &[b.clone()],
        Some(&[vec![0.0; n]]),
        None,
        None,
        None,
    );
    assert!(result.is_err(), "Providing only warm_x should error");

    // Only warm_z and warm_s (missing x)
    let result = solver.solve_with_warm_start(
        &[q],
        &[b],
        None,
        Some(&[vec![1.0; m]]),
        Some(&[vec![1.0; m]]),
        None,
    );
    assert!(result.is_err(), "Providing only warm_z/s should error");
}

/// Providing wrong-dimension warm start should raise an error
#[test]
fn test_warm_start_validation_wrong_dims() {
    let (P_ro, P_ci, P_val, A_ro, A_ci, A_val, cones, n, m) = simple_qp();

    let q = vec![1.0, -1.0];
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0];

    let mut solver = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, 1);
    solver.setup(&[P_val], &[A_val]);

    // Wrong x dim
    let result = solver.solve_with_warm_start(
        &[q],
        &[b],
        Some(&[vec![0.0; n + 1]]), // wrong!
        Some(&[vec![1.0; m]]),
        Some(&[vec![1.0; m]]),
        None,
    );
    assert!(result.is_err(), "Wrong x dimension should error");
}

/// Test warm start with batched problems
#[test]
fn test_warm_start_batched() {
    let (P_ro, P_ci, P_val, A_ro, A_ci, A_val, cones, n, m) = simple_qp();
    let batch_size = 2;

    let q_batch = vec![vec![1.0, -1.0], vec![0.5, -0.5]];
    let b_batch = vec![vec![1.0, 0.0, 0.0, 1.0, 1.0], vec![1.0, 0.0, 0.0, 1.0, 1.0]];

    // Cold solve
    let mut solver_cold =
        make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, batch_size);
    solver_cold.setup(
        &vec![P_val.clone(); batch_size],
        &vec![A_val.clone(); batch_size],
    );
    let sol_cold = solver_cold.solve(&q_batch, &b_batch).unwrap();

    for i in 0..batch_size {
        assert_eq!(sol_cold[i].status, SolverStatus::Solved);
    }

    // Warm solve from cold solution
    let warm_xs: Vec<Vec<f64>> = sol_cold.iter().map(|s| s.x.clone()).collect();
    let warm_zs: Vec<Vec<f64>> = sol_cold.iter().map(|s| s.z.clone()).collect();
    let warm_ss: Vec<Vec<f64>> = sol_cold.iter().map(|s| s.s.clone()).collect();

    let mut solver_warm =
        make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, batch_size);
    solver_warm.setup(&vec![P_val; batch_size], &vec![A_val; batch_size]);
    let sol_warm = solver_warm
        .solve_with_warm_start(
            &q_batch,
            &b_batch,
            Some(&warm_xs),
            Some(&warm_zs),
            Some(&warm_ss),
            None,
        )
        .unwrap();

    for i in 0..batch_size {
        assert_eq!(sol_warm[i].status, SolverStatus::Solved);
        for j in 0..n {
            assert!(
                is_close(sol_warm[i].x[j], sol_cold[i].x[j], 1e-4),
                "Batch {} x[{}] mismatch: warm={} vs cold={}",
                i,
                j,
                sol_warm[i].x[j],
                sol_cold[i].x[j]
            );
        }
    }
}
