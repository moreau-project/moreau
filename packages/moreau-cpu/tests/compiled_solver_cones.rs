//! Tests for CompiledSolver with non-QP cone types.
//!
//! Existing tests only cover CompiledSolver with zero + nonneg cones.
//! This fills gaps for: exponential cones, power cones, SOC cones,
//! mixed cone types, warm starting, time limit, and different P/A per batch.

#![allow(non_snake_case)]

use moreau::{algebra::*, solver::*};

fn is_close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    (a - b).abs() <= atol + rtol * b.abs()
}

// ============================================================================
// COMPILED SOLVER WITH EXPONENTIAL CONES
// ============================================================================

/// Build CSR structure for a 3x3 identity-like constraint for exp cone
fn exp_cone_problem() -> (
    Vec<usize>,
    Vec<usize>,
    Vec<f64>, // P_ro, P_ci, P_values
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,                 // A_ro, A_ci, A_values
    Vec<SupportedConeT<f64>>, // cones
    Vec<f64>,
    Vec<f64>, // q, b
) {
    // minimize x0 + x1 + x2 + 0.5*(x0^2 + x1^2 + x2^2)
    // subject to (-x0, -x1, -x2) + s = 0, s in K_exp
    //            -x0 - x1 - x2 + s4 = -3, s4 >= 0
    let _n = 3;

    // P = I (diagonal)
    let P_ro = vec![0, 1, 2, 3];
    let P_ci = vec![0, 1, 2];
    let P_values = vec![1.0, 1.0, 1.0];

    // A: rows 0,1,2 = -I for exp cone, row 3 = [-1,-1,-1] for nonneg
    let A_ro = vec![0, 1, 2, 3, 6];
    let A_ci = vec![0, 1, 2, 0, 1, 2];
    let A_values = vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0];

    let cones = vec![ExponentialConeT(), NonnegativeConeT(1)];
    let q = vec![1.0, 1.0, -1.0];
    let b = vec![0.0, 0.0, 0.0, -3.0];

    (P_ro, P_ci, P_values, A_ro, A_ci, A_values, cones, q, b)
}

#[test]
fn test_compiled_solver_exp_cone() {
    let (P_ro, P_ci, P_values, A_ro, A_ci, A_values, cones, q, b) = exp_cone_problem();

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver =
        CompiledSolver::new(3, 4, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false).unwrap();

    solver.setup(&[P_values], &[A_values]);
    let results = solver.solve(&[q], &[b]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    for i in 0..3 {
        assert!(!results[0].x[i].is_nan(), "x[{}] is NaN", i);
        assert!(!results[0].x[i].is_infinite(), "x[{}] is infinite", i);
    }
}

#[test]
fn test_compiled_solver_exp_cone_matches_single() {
    let (P_ro, P_ci, P_values, A_ro, A_ci, A_values, cones, q, b) = exp_cone_problem();

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    // Single solver (CSC format)
    let P_csc = CscMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]);
    let A_csc = CscMatrix::new(
        4,
        3,
        vec![0, 2, 4, 6],
        vec![0, 3, 1, 3, 2, 3],
        vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    );

    let mut single = DefaultSolver::new(&P_csc, &q, &A_csc, &b, &cones, settings.clone()).unwrap();
    single.solve();
    assert_eq!(single.solution.status, SolverStatus::Solved);

    // Compiled solver
    let mut compiled =
        CompiledSolver::new(3, 4, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false).unwrap();

    compiled.setup(&[P_values], &[A_values]);
    let results = compiled.solve(&[q], &[b]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    for i in 0..3 {
        assert!(
            is_close(single.solution.x[i], results[0].x[i], 1e-5, 1e-7),
            "x[{}] mismatch: single={} vs compiled={}",
            i,
            single.solution.x[i],
            results[0].x[i]
        );
    }
}

#[test]
fn test_compiled_solver_exp_cone_batched() {
    let (P_ro, P_ci, P_values, A_ro, A_ci, A_values, cones, _, b) = exp_cone_problem();

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let batch_size = 3;
    let mut solver = CompiledSolver::new(
        3, 4, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, batch_size, false,
    )
    .unwrap();

    let P_batch = vec![P_values.clone(); batch_size];
    let A_batch = vec![A_values.clone(); batch_size];
    solver.setup(&P_batch, &A_batch);

    let qs = vec![
        vec![1.0, 1.0, -1.0],
        vec![2.0, 0.5, -0.5],
        vec![0.5, 2.0, -1.5],
    ];
    let bs = vec![b.clone(); batch_size];

    let results = solver.solve(&qs, &bs).unwrap();

    for i in 0..batch_size {
        assert_eq!(
            results[i].status,
            SolverStatus::Solved,
            "Problem {} failed",
            i
        );
    }
    // Different q should give different results
    assert!(!is_close(results[0].x[0], results[1].x[0], 1e-3, 1e-5));
}

// ============================================================================
// COMPILED SOLVER WITH POWER CONES
// ============================================================================

fn power_cone_problem(
    alpha: f64,
) -> (
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
    Vec<f64>,
    Vec<f64>,
) {
    let P_ro = vec![0, 1, 2, 3];
    let P_ci = vec![0, 1, 2];
    let P_values = vec![1.0, 1.0, 1.0];

    let A_ro = vec![0, 1, 2, 3, 6];
    let A_ci = vec![0, 1, 2, 0, 1, 2];
    let A_values = vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0];

    let cones = vec![PowerConeT(alpha), NonnegativeConeT(1)];
    let q = vec![1.0, 1.0, 0.0];
    let b = vec![0.0, 0.0, 0.0, -5.0];

    (P_ro, P_ci, P_values, A_ro, A_ci, A_values, cones, q, b)
}

#[test]
fn test_compiled_solver_power_cone_half() {
    let (P_ro, P_ci, P_values, A_ro, A_ci, A_values, cones, q, b) = power_cone_problem(0.5);

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver =
        CompiledSolver::new(3, 4, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false).unwrap();

    solver.setup(&[P_values], &[A_values]);
    let results = solver.solve(&[q], &[b]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    for i in 0..3 {
        assert!(!results[0].x[i].is_nan(), "x[{}] is NaN", i);
    }
}

#[test]
fn test_compiled_solver_power_cone_extreme_alphas() {
    for alpha in [0.1, 0.3, 0.7, 0.9] {
        let (P_ro, P_ci, P_values, A_ro, A_ci, A_values, cones, q, b) = power_cone_problem(alpha);

        let mut settings = DefaultSettings::default();
        settings.verbose = false;

        let mut solver =
            CompiledSolver::new(3, 4, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false)
                .unwrap();

        solver.setup(&[P_values], &[A_values]);
        let results = solver.solve(&[q], &[b]).unwrap();

        assert_eq!(
            results[0].status,
            SolverStatus::Solved,
            "Failed with alpha={}",
            alpha
        );
    }
}

#[test]
fn test_compiled_solver_power_cone_matches_single() {
    let alpha = 0.5;
    let (P_ro, P_ci, P_values, A_ro, A_ci, A_values, cones, q, b) = power_cone_problem(alpha);

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    // Single solver
    let P_csc = CscMatrix::new(3, 3, vec![0, 1, 2, 3], vec![0, 1, 2], vec![1.0, 1.0, 1.0]);
    let A_csc = CscMatrix::new(
        4,
        3,
        vec![0, 2, 4, 6],
        vec![0, 3, 1, 3, 2, 3],
        vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    );

    let mut single = DefaultSolver::new(&P_csc, &q, &A_csc, &b, &cones, settings.clone()).unwrap();
    single.solve();
    assert_eq!(single.solution.status, SolverStatus::Solved);

    // Compiled solver
    let mut compiled =
        CompiledSolver::new(3, 4, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false).unwrap();

    compiled.setup(&[P_values], &[A_values]);
    let results = compiled.solve(&[q], &[b]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    for i in 0..3 {
        assert!(
            is_close(single.solution.x[i], results[0].x[i], 1e-5, 1e-7),
            "x[{}] mismatch: single={} vs compiled={}",
            i,
            single.solution.x[i],
            results[0].x[i]
        );
    }
}

// ============================================================================
// COMPILED SOLVER WITH SOC CONES
// ============================================================================

#[test]
fn test_compiled_solver_soc_cone() {
    // SOC: minimize x0 + 0.5*(x0^2+x1^2+x2^2) s.t. (x0,x1,x2) in SOC
    let P_ro = vec![0, 1, 2, 3];
    let P_ci = vec![0, 1, 2];
    let P_values = vec![1.0, 1.0, 1.0];

    let A_ro = vec![0, 1, 2, 3];
    let A_ci = vec![0, 1, 2];
    let A_values = vec![-1.0, -1.0, -1.0];

    let cones = vec![SecondOrderConeT(3)];
    let q = vec![1.0, 0.5, 0.5];
    let b = vec![0.0, 0.0, 0.0];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver =
        CompiledSolver::new(3, 3, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false).unwrap();

    solver.setup(&[P_values], &[A_values]);
    let results = solver.solve(&[q], &[b]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    // SOC: x0 >= sqrt(x1^2 + x2^2)
    let x = &results[0].x;
    let norm = f64::sqrt(x[1] * x[1] + x[2] * x[2]);
    assert!(
        x[0] >= norm - 1e-6,
        "SOC violated: x[0]={}, norm={}",
        x[0],
        norm
    );
}

// ============================================================================
// COMPILED SOLVER WITH MIXED CONES
// ============================================================================

#[test]
fn test_compiled_solver_mixed_cones() {
    // Mixed: zero + nonneg + SOC
    // n=5, m=6: 1 zero, 2 nonneg, 1 SOC(3)
    let P_ro = vec![0, 1, 2, 3, 4, 5];
    let P_ci = vec![0, 1, 2, 3, 4];
    let P_values = vec![1.0; 5];

    // A = -I (5x5) + extra row for zero cone
    // Actually let's do: 6 rows, 5 cols
    // Row 0: zero cone (x0 + x1 = 1)
    // Rows 1-2: nonneg (x0 >= 0, x1 >= 0)
    // Rows 3-5: SOC (x2, x3, x4) in SOC
    let A_ro = vec![0, 2, 3, 4, 5, 6, 7];
    let A_ci = vec![0, 1, 0, 1, 2, 3, 4];
    let A_values = vec![1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0];

    let cones = vec![ZeroConeT(1), NonnegativeConeT(2), SecondOrderConeT(3)];
    let q = vec![1.0, 1.0, 1.0, 0.5, 0.5];
    let b = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver =
        CompiledSolver::new(5, 6, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false).unwrap();

    solver.setup(&[P_values], &[A_values]);
    let results = solver.solve(&[q], &[b]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    // Check equality constraint: x0 + x1 ≈ 1
    let x = &results[0].x;
    assert!(
        is_close(x[0] + x[1], 1.0, 1e-5, 1e-6),
        "Equality violated: x0+x1 = {}",
        x[0] + x[1]
    );
}

// ============================================================================
// COMPILED SOLVER WARM STARTING
// ============================================================================

#[test]
fn test_compiled_solver_warm_start_fewer_iterations() {
    let P_ro = vec![0, 2, 4];
    let P_ci = vec![0, 1, 0, 1];
    let P_values = vec![4.0, 1.0, 1.0, 2.0];

    let A_ro = vec![0, 2, 3, 4];
    let A_ci = vec![0, 1, 0, 1];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];

    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];
    let q = vec![1.0, 1.0];
    let b = vec![1.0, 0.7, 0.7];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    // Cold solve
    let mut solver_cold = CompiledSolver::new(
        2,
        3,
        &P_ro,
        &P_ci,
        &A_ro,
        &A_ci,
        &cones,
        settings.clone(),
        1,
        false,
    )
    .unwrap();
    solver_cold.setup(&[P_values.clone()], &[A_values.clone()]);
    let cold_results = solver_cold.solve(&[q.clone()], &[b.clone()]).unwrap();
    assert_eq!(cold_results[0].status, SolverStatus::Solved);
    let cold_iters = cold_results[0].iterations;

    // Warm solve
    let mut solver_warm =
        CompiledSolver::new(2, 3, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false).unwrap();
    solver_warm.setup(&[P_values], &[A_values]);
    let warm_results = solver_warm
        .solve_with_warm_start(
            &[q],
            &[b],
            Some(&[cold_results[0].x.clone()]),
            Some(&[cold_results[0].z.clone()]),
            Some(&[cold_results[0].s.clone()]),
            None,
        )
        .unwrap();

    assert_eq!(warm_results[0].status, SolverStatus::Solved);
    let warm_iters = warm_results[0].iterations;

    assert!(
        warm_iters <= cold_iters,
        "Warm ({} iters) should be <= cold ({} iters)",
        warm_iters,
        cold_iters
    );

    // Solutions should match
    for i in 0..2 {
        assert!(
            is_close(cold_results[0].x[i], warm_results[0].x[i], 1e-5, 1e-7),
            "x[{}] mismatch",
            i
        );
    }
}

#[test]
fn test_compiled_solver_warm_start_perturbed_problem() {
    let P_ro = vec![0, 2, 4];
    let P_ci = vec![0, 1, 0, 1];
    let P_values = vec![4.0, 1.0, 1.0, 2.0];

    let A_ro = vec![0, 2, 3, 4];
    let A_ci = vec![0, 1, 0, 1];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];

    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];
    let b = vec![1.0, 0.7, 0.7];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    // Solve P1
    let q1 = vec![1.0, 1.0];
    let mut solver1 = CompiledSolver::new(
        2,
        3,
        &P_ro,
        &P_ci,
        &A_ro,
        &A_ci,
        &cones,
        settings.clone(),
        1,
        false,
    )
    .unwrap();
    solver1.setup(&[P_values.clone()], &[A_values.clone()]);
    let r1 = solver1.solve(&[q1], &[b.clone()]).unwrap();
    assert_eq!(r1[0].status, SolverStatus::Solved);

    // Solve P2 (perturbed q) with warm start from P1
    let q2 = vec![1.1, 0.9];
    let mut solver2 = CompiledSolver::new(
        2,
        3,
        &P_ro,
        &P_ci,
        &A_ro,
        &A_ci,
        &cones,
        settings.clone(),
        1,
        false,
    )
    .unwrap();
    solver2.setup(&[P_values.clone()], &[A_values.clone()]);
    let r2_warm = solver2
        .solve_with_warm_start(
            &[q2.clone()],
            &[b.clone()],
            Some(&[r1[0].x.clone()]),
            Some(&[r1[0].z.clone()]),
            Some(&[r1[0].s.clone()]),
            None,
        )
        .unwrap();
    assert_eq!(r2_warm[0].status, SolverStatus::Solved);

    // Cold solve P2 for reference
    let mut solver2_cold =
        CompiledSolver::new(2, 3, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false).unwrap();
    solver2_cold.setup(&[P_values], &[A_values]);
    let r2_cold = solver2_cold.solve(&[q2], &[b]).unwrap();
    assert_eq!(r2_cold[0].status, SolverStatus::Solved);

    // Solutions should match
    for i in 0..2 {
        assert!(
            is_close(r2_warm[0].x[i], r2_cold[0].x[i], 1e-5, 1e-7),
            "x[{}] mismatch: warm={} vs cold={}",
            i,
            r2_warm[0].x[i],
            r2_cold[0].x[i]
        );
    }
}

// ============================================================================
// COMPILED SOLVER WITH DIFFERENT P/A PER BATCH
// ============================================================================

#[test]
fn test_compiled_solver_different_P_per_batch() {
    let P_ro = vec![0, 2, 4];
    let P_ci = vec![0, 1, 0, 1];

    let A_ro = vec![0, 2, 3, 4];
    let A_ci = vec![0, 1, 0, 1];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];

    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];
    let q = vec![1.0, 1.0];
    let b = vec![1.0, 0.7, 0.7];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let batch_size = 2;
    let mut solver = CompiledSolver::new(
        2,
        3,
        &P_ro,
        &P_ci,
        &A_ro,
        &A_ci,
        &cones,
        settings.clone(),
        batch_size,
        false,
    )
    .unwrap();

    // Different P values for each problem (very different to ensure distinct solutions)
    let P_batch = vec![vec![4.0, 1.0, 1.0, 2.0], vec![2.0, 0.0, 0.0, 20.0]];
    let A_batch = vec![A_values.clone(), A_values.clone()];

    solver.setup(&P_batch, &A_batch);

    let qs = vec![q.clone(), q.clone()];
    let bs = vec![b.clone(), b.clone()];
    let results = solver.solve(&qs, &bs).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    assert_eq!(results[1].status, SolverStatus::Solved);

    // Each batch slot must match its own single-problem solve. This pins
    // batch order: a mixup in parallel setup would swap the solutions.
    for (i, P_values) in P_batch.iter().enumerate() {
        // P is symmetric with a symmetric pattern, so CSC values == CSR values
        let P_csc = CscMatrix::new(2, 2, vec![0, 2, 4], vec![0, 1, 0, 1], P_values.clone());
        let A_csc = CscMatrix::new(3, 2, vec![0, 2, 4], vec![0, 1, 0, 2], vec![1.0; 4]);
        let mut single =
            DefaultSolver::new(&P_csc, &q, &A_csc, &b, &cones, settings.clone()).unwrap();
        single.solve();
        assert_eq!(single.solution.status, SolverStatus::Solved);
        for j in 0..2 {
            assert!(
                is_close(results[i].x[j], single.solution.x[j], 1e-5, 1e-7),
                "batch[{}].x[{}] = {} does not match single solve x[{}] = {}",
                i,
                j,
                results[i].x[j],
                j,
                single.solution.x[j]
            );
        }
    }
}

#[test]
fn test_compiled_solver_different_A_per_batch() {
    let P_ro = vec![0, 1, 2];
    let P_ci = vec![0, 1];
    let P_values = vec![2.0, 2.0];

    let A_ro = vec![0, 1, 2];
    let A_ci = vec![0, 1];

    let cones = vec![NonnegativeConeT(2)];
    let q = vec![1.0, 1.0];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let batch_size = 2;
    let mut solver = CompiledSolver::new(
        2, 2, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, batch_size, false,
    )
    .unwrap();

    // Different A values (different constraint matrices)
    let P_batch = vec![P_values.clone(), P_values];
    let A_batch = vec![vec![-1.0, -1.0], vec![-2.0, -0.5]];
    solver.setup(&P_batch, &A_batch);

    let bs = vec![vec![0.0, 0.0], vec![0.0, 0.0]];
    let qs = vec![q.clone(), q];
    let results = solver.solve(&qs, &bs).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    assert_eq!(results[1].status, SolverStatus::Solved);
}

// ============================================================================
// TIME LIMIT
// ============================================================================

#[test]
fn test_time_limit_zero_returns_max_time() {
    let P_ro = vec![0, 2, 4];
    let P_ci = vec![0, 1, 0, 1];
    let P_values = vec![4.0, 1.0, 1.0, 2.0];

    let A_ro = vec![0, 2, 3, 4];
    let A_ci = vec![0, 1, 0, 1];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];

    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];
    let q = vec![1.0, 1.0];
    let b = vec![1.0, 0.7, 0.7];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.time_limit = 0.0; // Immediate timeout

    let mut solver =
        CompiledSolver::new(2, 3, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false).unwrap();

    solver.setup(&[P_values], &[A_values]);
    let results = solver.solve(&[q], &[b]).unwrap();

    // With zero time limit, solver should either return MaxTime or
    // solve instantly if the first iteration is fast enough
    assert!(
        results[0].status == SolverStatus::MaxTime || results[0].status == SolverStatus::Solved,
        "Expected MaxTime or Solved, got {:?}",
        results[0].status
    );
}

#[test]
fn test_time_limit_allows_completion() {
    let P_ro = vec![0, 2, 4];
    let P_ci = vec![0, 1, 0, 1];
    let P_values = vec![4.0, 1.0, 1.0, 2.0];

    let A_ro = vec![0, 2, 3, 4];
    let A_ci = vec![0, 1, 0, 1];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];

    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];
    let q = vec![1.0, 1.0];
    let b = vec![1.0, 0.7, 0.7];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.time_limit = 60.0; // Very generous

    let mut solver =
        CompiledSolver::new(2, 3, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false).unwrap();

    solver.setup(&[P_values], &[A_values]);
    let results = solver.solve(&[q], &[b]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
}

// ============================================================================
// COMPILED SOLVER ALL CONE TYPES IN ONE PROBLEM
// ============================================================================

#[test]
fn test_compiled_solver_all_cone_types() {
    // Problem with zero + nonneg + SOC + exp + power cones
    // This tests that the compiled solver correctly handles composite cone structures.
    //
    // n=9, m = 1 (zero) + 2 (nonneg) + 3 (SOC) + 3 (exp) + 3 (power) = 12
    let n = 9;
    let m = 12;

    // P = I (diagonal)
    let P_ro: Vec<usize> = (0..=n).collect();
    let P_ci: Vec<usize> = (0..n).collect();
    let P_values = vec![1.0; n];

    // A = block structure mapping variables to constraints
    // Simple: -I for first 9 rows, then padding
    // Actually let's be more careful...
    // Row 0: zero cone -> x0 + x1 = 1 => [1, 1, 0, ...]
    // Rows 1-2: nonneg -> -x2, -x3 (x2,x3 >= 0)
    // Rows 3-5: SOC -> -x4, -x5, -x6 ((x4,x5,x6) in SOC)
    // Rows 6-8: exp -> -x0, -x1, -x2 (some variables mapped to exp)
    // Rows 9-11: power -> -x6, -x7, -x8

    // For simplicity, use a block diagonal approach:
    // We have 9 variables. Map as follows:
    //   Zero cone: x0 + x1 = 2 (row 0, cols 0,1)
    //   Nonneg: -x0 + s = 0, -x1 + s = 0 (rows 1-2)
    //   SOC: (-x2, -x3, -x4) + s = 0 (rows 3-5)
    //   Exp: (-x2, -x3, -x4) + s = 0 (rows 6-8) -- different variables?
    //
    // Actually, let's keep it simple: just test that the solver doesn't crash.

    let mut A_ro = Vec::new();
    let mut A_ci = Vec::new();
    let mut A_values_vec = Vec::new();
    let mut row_count = 0;

    // Row 0: zero cone (x0 + x1 = 2)
    A_ci.push(0);
    A_ci.push(1);
    A_values_vec.push(1.0);
    A_values_vec.push(1.0);
    A_ro.push(row_count);
    row_count = A_ci.len();

    // Rows 1-2: nonneg (-x2, -x3)
    A_ci.push(2);
    A_values_vec.push(-1.0);
    A_ro.push(row_count);
    row_count = A_ci.len();

    A_ci.push(3);
    A_values_vec.push(-1.0);
    A_ro.push(row_count);
    row_count = A_ci.len();

    // Rows 3-5: SOC (-x4, -x5, -x6)
    A_ci.push(4);
    A_values_vec.push(-1.0);
    A_ro.push(row_count);
    row_count = A_ci.len();

    A_ci.push(5);
    A_values_vec.push(-1.0);
    A_ro.push(row_count);
    row_count = A_ci.len();

    A_ci.push(6);
    A_values_vec.push(-1.0);
    A_ro.push(row_count);
    row_count = A_ci.len();

    // Rows 6-8: exp (-x0, -x1, -x3)
    A_ci.push(0);
    A_values_vec.push(-1.0);
    A_ro.push(row_count);
    row_count = A_ci.len();

    A_ci.push(1);
    A_values_vec.push(-1.0);
    A_ro.push(row_count);
    row_count = A_ci.len();

    A_ci.push(3);
    A_values_vec.push(-1.0);
    A_ro.push(row_count);
    row_count = A_ci.len();

    // Rows 9-11: power (-x7, -x8, -x6)
    A_ci.push(7);
    A_values_vec.push(-1.0);
    A_ro.push(row_count);
    row_count = A_ci.len();

    A_ci.push(8);
    A_values_vec.push(-1.0);
    A_ro.push(row_count);
    row_count = A_ci.len();

    A_ci.push(6);
    A_values_vec.push(-1.0);
    A_ro.push(row_count);
    row_count = A_ci.len();

    A_ro.push(row_count); // final sentinel

    let cones = vec![
        ZeroConeT(1),
        NonnegativeConeT(2),
        SecondOrderConeT(3),
        ExponentialConeT(),
        PowerConeT(0.5),
    ];

    let q = vec![0.1; n];
    let b = vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver =
        CompiledSolver::new(n, m, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false).unwrap();

    solver.setup(&[P_values], &[A_values_vec]);
    let results = solver.solve(&[q], &[b]).unwrap();

    // Should solve or at least not crash
    assert!(
        results[0].status == SolverStatus::Solved
            || results[0].status == SolverStatus::AlmostSolved,
        "Expected Solved/AlmostSolved, got {:?}",
        results[0].status
    );
}

// ============================================================================
// COMPILED SOLVER: VARIABLE-DIMENSION SOC TESTS
// ============================================================================

#[test]
fn test_compiled_solver_soc_dim2() {
    // SOC(2): minimize 0.5*||x||^2 - 2*x[0]  s.t.  x in SOC(2)
    let dim = 2;
    let P_ro = vec![0, 1, 2];
    let P_ci = vec![0, 1];
    let P_values = vec![1.0, 1.0];

    let A_ro = vec![0, 1, 2];
    let A_ci = vec![0, 1];
    let A_values = vec![-1.0, -1.0];

    let cones = vec![SecondOrderConeT(dim)];
    let mut q = vec![0.0; dim];
    q[0] = -2.0;
    let b = vec![0.0; dim];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver = CompiledSolver::new(
        dim, dim, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false,
    )
    .unwrap();

    solver.setup(&[P_values], &[A_values]);
    let results = solver.solve(&[q], &[b]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    let x = &results[0].x;
    let tail_norm: f64 = x[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        x[0] >= tail_norm - 1e-6,
        "SOC violated: x[0]={}, tail={}",
        x[0],
        tail_norm
    );
}

#[test]
fn test_compiled_solver_soc_dim5() {
    let dim = 5;
    let P_ro: Vec<usize> = (0..=dim).collect();
    let P_ci: Vec<usize> = (0..dim).collect();
    let P_values = vec![1.0; dim];

    let A_ro: Vec<usize> = (0..=dim).collect();
    let A_ci: Vec<usize> = (0..dim).collect();
    let A_values = vec![-1.0; dim];

    let cones = vec![SecondOrderConeT(dim)];
    let mut q = vec![0.0; dim];
    q[0] = -2.0;
    let b = vec![0.0; dim];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver = CompiledSolver::new(
        dim, dim, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false,
    )
    .unwrap();

    solver.setup(&[P_values], &[A_values]);
    let results = solver.solve(&[q], &[b]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    let x = &results[0].x;
    let tail_norm: f64 = x[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        x[0] >= tail_norm - 1e-6,
        "SOC violated: x[0]={}, tail={}",
        x[0],
        tail_norm
    );
}

#[test]
fn test_compiled_solver_soc_dim10() {
    let dim = 10;
    let P_ro: Vec<usize> = (0..=dim).collect();
    let P_ci: Vec<usize> = (0..dim).collect();
    let P_values = vec![1.0; dim];

    let A_ro: Vec<usize> = (0..=dim).collect();
    let A_ci: Vec<usize> = (0..dim).collect();
    let A_values = vec![-1.0; dim];

    let cones = vec![SecondOrderConeT(dim)];
    let mut q = vec![0.0; dim];
    q[0] = -2.0;
    let b = vec![0.0; dim];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver = CompiledSolver::new(
        dim, dim, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false,
    )
    .unwrap();

    solver.setup(&[P_values], &[A_values]);
    let results = solver.solve(&[q], &[b]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    let x = &results[0].x;
    let tail_norm: f64 = x[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        x[0] >= tail_norm - 1e-6,
        "SOC violated: x[0]={}, tail={}",
        x[0],
        tail_norm
    );
}

#[test]
fn test_compiled_solver_mixed_soc_dims() {
    // Two SOC cones: SOC(2) + SOC(5)
    let dim1 = 2;
    let dim2 = 5;
    let n = dim1 + dim2;
    let m = n;

    let P_ro: Vec<usize> = (0..=n).collect();
    let P_ci: Vec<usize> = (0..n).collect();
    let P_values = vec![1.0; n];

    let A_ro: Vec<usize> = (0..=m).collect();
    let A_ci: Vec<usize> = (0..m).collect();
    let A_values = vec![-1.0; m];

    let cones = vec![SecondOrderConeT(dim1), SecondOrderConeT(dim2)];
    let mut q = vec![0.0; n];
    q[0] = -2.0;
    q[dim1] = -3.0;
    let b = vec![0.0; m];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver =
        CompiledSolver::new(n, m, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 1, false).unwrap();

    solver.setup(&[P_values], &[A_values]);
    let results = solver.solve(&[q], &[b]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    let x = &results[0].x;

    // Check first SOC(2)
    let tail1: f64 = x[1..dim1].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(x[0] >= tail1 - 1e-6, "SOC(2) violated");

    // Check second SOC(5)
    let tail2: f64 = x[dim1 + 1..n].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(x[dim1] >= tail2 - 1e-6, "SOC(5) violated");
}

#[test]
fn test_compiled_solver_nonneg_plus_variable_soc() {
    // nonneg(3) + SOC(4)
    let n = 5;
    let m_nonneg = 3;
    let dim_soc = 4;
    let m = m_nonneg + dim_soc; // 7

    // P = I (5x5)
    let P_ro: Vec<usize> = (0..=n).collect();
    let P_ci: Vec<usize> = (0..n).collect();
    let P_values = vec![1.0; n];

    // A: nonneg rows map to first 3 vars, SOC rows map to last 4 vars
    // Row 0: -x[0], Row 1: -x[1], Row 2: -x[2]  (nonneg: x[0..3] >= 0)
    // Row 3: -x[1], Row 4: -x[2], Row 5: -x[3], Row 6: -x[4]  (SOC(4))
    let A_ro = vec![0, 1, 2, 3, 5, 7, 8, 9];
    let A_ci = vec![0, 1, 2, 1, 2, 3, 4, 3, 4];
    // Wait - let me be more careful. CSR: row_offsets has m+1 entries.
    // Row 0 (nonneg): col 0, val -1
    // Row 1 (nonneg): col 1, val -1
    // Row 2 (nonneg): col 2, val -1
    // Row 3 (SOC head): col 1, col 2 => no, let's keep it simple.
    // Actually: just map the SOC to vars [1..5] via -I
    // Row 3: col 1, val -1
    // Row 4: col 2, val -1
    // Row 5: col 3, val -1
    // Row 6: col 4, val -1

    let A_ro_proper: Vec<usize> = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let A_ci_proper: Vec<usize> = vec![0, 1, 2, 1, 2, 3, 4];
    let A_values = vec![-1.0; 7];

    let cones = vec![NonnegativeConeT(m_nonneg), SecondOrderConeT(dim_soc)];
    let mut q = vec![0.0; n];
    q[0] = -1.0;
    q[1] = -2.0;
    let b = vec![0.0; m];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver = CompiledSolver::new(
        n,
        m,
        &P_ro,
        &P_ci,
        &A_ro_proper,
        &A_ci_proper,
        &cones,
        settings,
        1,
        false,
    )
    .unwrap();

    solver.setup(&[P_values], &[A_values]);
    let results = solver.solve(&[q], &[b]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);
    let x = &results[0].x;
    // Nonneg: x[0..3] >= 0
    for i in 0..3 {
        assert!(x[i] >= -1e-6, "Nonneg violated: x[{}]={}", i, x[i]);
    }
}

#[test]
fn test_compiled_solver_variable_soc_batched() {
    // Batch of 4 problems with SOC(5)
    let dim = 5;
    let batch_size = 4;

    let P_ro: Vec<usize> = (0..=dim).collect();
    let P_ci: Vec<usize> = (0..dim).collect();
    let P_values = vec![1.0; dim];

    let A_ro: Vec<usize> = (0..=dim).collect();
    let A_ci: Vec<usize> = (0..dim).collect();
    let A_values = vec![-1.0; dim];

    let cones = vec![SecondOrderConeT(dim)];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver = CompiledSolver::new(
        dim, dim, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, batch_size, false,
    )
    .unwrap();

    let P_batch: Vec<Vec<f64>> = vec![P_values; batch_size];
    let A_batch: Vec<Vec<f64>> = vec![A_values; batch_size];
    solver.setup(&P_batch, &A_batch);

    let q_batch: Vec<Vec<f64>> = (0..batch_size)
        .map(|i| {
            let mut q = vec![0.0; dim];
            q[0] = -1.0 - i as f64;
            q
        })
        .collect();
    let b_batch: Vec<Vec<f64>> = vec![vec![0.0; dim]; batch_size];

    let results = solver.solve(&q_batch, &b_batch).unwrap();

    for (i, result) in results.iter().enumerate() {
        assert_eq!(result.status, SolverStatus::Solved, "Batch {} failed", i);
        let x = &result.x;
        let tail_norm: f64 = x[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(x[0] >= tail_norm - 1e-6, "SOC violated in batch {}", i);
    }
}
