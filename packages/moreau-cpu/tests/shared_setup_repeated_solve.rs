//! Regression test: shared setup + repeated batched solve.
//!
//! Before the fix in compiled_solver.rs, calling `setup_shared()` followed by
//! two `solve()` calls would panic on the second solve with
//! "index out of bounds: the len is 0".
//!
//! The root cause was that `setup_shared()` cleared `base_equilibrated_P_values`
//! but never populated it. The first solve populated it during equilibration;
//! the second solve expected it to be populated (matrices_already_equilibrated)
//! but it was empty.

#![allow(non_snake_case)]

use moreau::solver::*;

fn is_close(a: f64, b: f64, atol: f64) -> bool {
    (a - b).abs() <= atol
}

/// Simple QP: minimize x'Px + q'x  s.t.  Ax + s = b, s in K
/// P = diag(2,2), A encodes x+y=1, x>=0, y>=0, x<=1, y<=1
fn simple_qp() -> (
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
    let n = 2;
    let m = 5;

    let P_row_offsets = vec![0, 1, 2];
    let P_col_indices = vec![0, 1];
    let P_values = vec![2.0, 2.0];

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

/// Shared setup + two consecutive solves should produce identical results.
/// This is the core regression: the second solve used to panic.
#[test]
fn test_shared_setup_repeated_solve_batch() {
    let (P_ro, P_ci, P_val, A_ro, A_ci, A_val, cones, n, m) = simple_qp();
    let batch_size = 4;

    let mut solver = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, batch_size);

    // Shared setup: single P/A values broadcast to all batch entries
    solver.setup_shared(&P_val, &A_val, batch_size);

    let q = vec![1.0, -1.0];
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0];
    let qs: Vec<Vec<f64>> = vec![q.clone(); batch_size];
    let bs: Vec<Vec<f64>> = vec![b.clone(); batch_size];

    // First solve
    let sol1 = solver.solve(&qs, &bs).unwrap();
    for i in 0..batch_size {
        assert_eq!(
            sol1[i].status,
            SolverStatus::Solved,
            "batch[{}] first solve status: {:?}",
            i,
            sol1[i].status
        );
    }

    // Second solve — this used to panic with "index out of bounds"
    let sol2 = solver.solve(&qs, &bs).unwrap();
    for i in 0..batch_size {
        assert_eq!(
            sol2[i].status,
            SolverStatus::Solved,
            "batch[{}] second solve status: {:?}",
            i,
            sol2[i].status
        );
    }

    // Results should be identical
    for i in 0..batch_size {
        for j in 0..n {
            assert!(
                is_close(sol1[i].x[j], sol2[i].x[j], 1e-6),
                "batch[{}] x[{}] mismatch: solve1={} vs solve2={}",
                i,
                j,
                sol1[i].x[j],
                sol2[i].x[j]
            );
        }
    }
}

/// Shared setup + repeated solve with different q/b values each time.
#[test]
fn test_shared_setup_solve_different_data() {
    let (P_ro, P_ci, P_val, A_ro, A_ci, A_val, cones, n, m) = simple_qp();
    let batch_size = 2;

    let mut solver = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, batch_size);
    solver.setup_shared(&P_val, &A_val, batch_size);

    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0];

    // Solve with q1
    let qs1 = vec![vec![1.0, -1.0]; batch_size];
    let bs = vec![b.clone(); batch_size];
    let sol1 = solver.solve(&qs1, &bs).unwrap();

    // Solve with q2 (perturbed)
    let qs2 = vec![vec![-1.0, 1.0]; batch_size];
    let sol2 = solver.solve(&qs2, &bs).unwrap();

    // Both should solve
    for i in 0..batch_size {
        assert_eq!(sol1[i].status, SolverStatus::Solved);
        assert_eq!(sol2[i].status, SolverStatus::Solved);
    }

    // Solutions should differ (different q)
    assert!(
        !is_close(sol1[0].x[0], sol2[0].x[0], 1e-4),
        "Solutions should differ with different q: sol1.x={:?} sol2.x={:?}",
        sol1[0].x,
        sol2[0].x
    );
}

/// Shared setup + warm-started second solve should work.
#[test]
fn test_shared_setup_warm_start_second_solve() {
    let (P_ro, P_ci, P_val, A_ro, A_ci, A_val, cones, n, m) = simple_qp();
    let batch_size = 2;

    let mut solver = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, batch_size);
    solver.setup_shared(&P_val, &A_val, batch_size);

    let q = vec![1.0, -1.0];
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0];
    let qs: Vec<Vec<f64>> = vec![q.clone(); batch_size];
    let bs: Vec<Vec<f64>> = vec![b.clone(); batch_size];

    // Cold solve
    let sol_cold = solver.solve(&qs, &bs).unwrap();
    for i in 0..batch_size {
        assert_eq!(sol_cold[i].status, SolverStatus::Solved);
    }

    // Warm-started solve with slightly perturbed q
    let qs2: Vec<Vec<f64>> = vec![vec![1.1, -0.9]; batch_size];
    let xs: Vec<Vec<f64>> = sol_cold.iter().map(|s| s.x.clone()).collect();
    let zs: Vec<Vec<f64>> = sol_cold.iter().map(|s| s.z.clone()).collect();
    let ss: Vec<Vec<f64>> = sol_cold.iter().map(|s| s.s.clone()).collect();

    let sol_warm = solver
        .solve_with_warm_start(&qs2, &bs, Some(&xs), Some(&zs), Some(&ss), None)
        .unwrap();

    for i in 0..batch_size {
        assert_eq!(
            sol_warm[i].status,
            SolverStatus::Solved,
            "batch[{}] warm solve status: {:?}",
            i,
            sol_warm[i].status
        );
    }
}

/// Three consecutive solves to ensure base_equilibrated_P_values
/// stays consistent across multiple reuses.
#[test]
fn test_shared_setup_three_solves() {
    let (P_ro, P_ci, P_val, A_ro, A_ci, A_val, cones, n, m) = simple_qp();
    let batch_size = 3;

    let mut solver = make_compiled_solver(&P_ro, &P_ci, &A_ro, &A_ci, &cones, n, m, batch_size);
    solver.setup_shared(&P_val, &A_val, batch_size);

    let q = vec![1.0, -1.0];
    let b = vec![1.0, 0.0, 0.0, 1.0, 1.0];
    let qs: Vec<Vec<f64>> = vec![q.clone(); batch_size];
    let bs: Vec<Vec<f64>> = vec![b.clone(); batch_size];

    let sol1 = solver.solve(&qs, &bs).unwrap();
    let sol2 = solver.solve(&qs, &bs).unwrap();
    let sol3 = solver.solve(&qs, &bs).unwrap();

    for i in 0..batch_size {
        assert_eq!(sol1[i].status, SolverStatus::Solved);
        assert_eq!(sol2[i].status, SolverStatus::Solved);
        assert_eq!(sol3[i].status, SolverStatus::Solved);

        for j in 0..n {
            assert!(
                is_close(sol1[i].x[j], sol3[i].x[j], 1e-6),
                "batch[{}] x[{}]: solve1={} vs solve3={}",
                i,
                j,
                sol1[i].x[j],
                sol3[i].x[j]
            );
        }
    }
}
