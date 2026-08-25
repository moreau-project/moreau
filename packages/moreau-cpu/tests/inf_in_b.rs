#![allow(non_snake_case)]

use moreau::{algebra::*, solver::*};

// ============================================================================
// Helper: create a CompiledSolver, setup, and solve
// ============================================================================

fn compiled_solve(
    n: usize,
    m: usize,
    P_row_offsets: &[usize],
    P_col_indices: &[usize],
    A_row_offsets: &[usize],
    A_col_indices: &[usize],
    cones: &[SupportedConeT<f64>],
    P_values: &[f64],
    A_values: &[f64],
    q: &[f64],
    b: &[f64],
) -> DefaultSolution<f64> {
    let settings = DefaultSettings::default();
    let mut solver = CompiledSolver::new(
        n,
        m,
        P_row_offsets,
        P_col_indices,
        A_row_offsets,
        A_col_indices,
        cones,
        settings,
        1,
        false,
    )
    .unwrap();

    solver.setup(&[P_values.to_vec()], &[A_values.to_vec()]);
    let solutions = solver.solve(&[q.to_vec()], &[b.to_vec()]).unwrap();
    solutions.into_iter().next().unwrap()
}

// ============================================================================
// DefaultSolver (presolve enabled)
// ============================================================================

/// DefaultSolver handles +inf in b via presolve (strips the vacuous constraint).
#[test]
fn test_default_solver_nonneg_pos_inf() {
    // min x s.t. x >= 0, x <= inf
    let P = CscMatrix::<f64>::zeros((1, 1));
    let c = vec![1.0];
    let A = CscMatrix::new(2, 1, vec![0, 2], vec![0, 1], vec![-1.0, 1.0]);
    let b = vec![0.0, f64::INFINITY];
    let cones = vec![NonnegativeConeT(2)];

    let mut solver =
        DefaultSolver::new(&P, &c, &A, &b, &cones, DefaultSettings::default()).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
    assert!(f64::abs(solver.solution.x[0]) <= 1e-6);
    assert!(f64::abs(solver.solution.obj_val) <= 1e-6);
}

// ============================================================================
// CompiledSolver: +inf in nonneg cone (vacuous, should solve)
// ============================================================================

/// Simple: min x s.t. x >= 0, x <= inf → x = 0
#[test]
fn test_compiled_nonneg_pos_inf_simple() {
    let sol = compiled_solve(
        1,
        2,
        &[0, 0],
        &[],
        &[0, 1, 2],
        &[0, 0],
        &[NonnegativeConeT(2)],
        &[],
        &[-1.0, 1.0],
        &[1.0],
        &[0.0, f64::INFINITY],
    );
    assert_eq!(sol.status, SolverStatus::Solved);
    assert!(f64::abs(sol.x[0]) <= 1e-6);
}

/// With active lower bound: min x s.t. x >= 0.5, x <= inf → x = 0.5
#[test]
fn test_compiled_nonneg_pos_inf_with_lower_bound() {
    let sol = compiled_solve(
        1,
        2,
        &[0, 0],
        &[],
        &[0, 1, 2],
        &[0, 0],
        &[NonnegativeConeT(2)],
        &[],
        &[-1.0, 1.0],
        &[1.0],
        &[-0.5, f64::INFINITY],
    );
    assert_eq!(sol.status, SolverStatus::Solved);
    assert!(
        (sol.x[0] - 0.5).abs() <= 1e-6,
        "x = {}, expected 0.5",
        sol.x[0]
    );
}

/// 2D with multiple inf upper bounds:
/// min x1 + x2 s.t. x1 >= 0.5, x2 >= 0.3, x1 <= inf, x2 <= inf
#[test]
fn test_compiled_nonneg_pos_inf_2d() {
    // P = 2I (diagonal QP so solution is at bounds)
    let sol = compiled_solve(
        2,
        4,
        &[0, 1, 2],
        &[0, 1], // P = diag(2, 2)
        &[0, 1, 2, 3, 4],
        &[0, 1, 0, 1], // A: [-I; I]
        &[NonnegativeConeT(4)],
        &[2.0, 2.0],
        &[-1.0, -1.0, 1.0, 1.0],
        &[1.0, -1.0],
        &[-0.5, -0.3, f64::INFINITY, f64::INFINITY],
    );
    assert_eq!(sol.status, SolverStatus::Solved);
    assert!(
        (sol.x[0] - 0.5).abs() <= 1e-5,
        "x[0] = {}, expected 0.5",
        sol.x[0]
    );
    assert!(
        (sol.x[1] - 0.5).abs() <= 1e-5,
        "x[1] = {}, expected 0.5",
        sol.x[1]
    );
}

// ============================================================================
// CompiledSolver: -inf in nonneg cone (infeasible)
// ============================================================================

/// b = -inf in nonneg cone → PrimalInfeasible
#[test]
fn test_compiled_nonneg_neg_inf_infeasible() {
    let sol = compiled_solve(
        1,
        1,
        &[0, 0],
        &[],
        &[0, 1],
        &[0],
        &[NonnegativeConeT(1)],
        &[],
        &[-1.0],
        &[1.0],
        &[-f64::INFINITY],
    );
    assert_eq!(sol.status, SolverStatus::PrimalInfeasible);
    assert!(sol.obj_val.is_nan());
}

/// Mixed: one finite constraint, one -inf → PrimalInfeasible
#[test]
fn test_compiled_nonneg_neg_inf_mixed() {
    let sol = compiled_solve(
        1,
        2,
        &[0, 0],
        &[],
        &[0, 1, 2],
        &[0, 0],
        &[NonnegativeConeT(2)],
        &[],
        &[-1.0, 1.0],
        &[1.0],
        &[0.0, -f64::INFINITY],
    );
    assert_eq!(sol.status, SolverStatus::PrimalInfeasible);
}

// ============================================================================
// CompiledSolver: ±inf in zero cone (equality, always infeasible)
// ============================================================================

/// b = +inf in zero cone (a'x = inf) → PrimalInfeasible
#[test]
fn test_compiled_zero_cone_pos_inf_infeasible() {
    let sol = compiled_solve(
        1,
        1,
        &[0, 0],
        &[],
        &[0, 1],
        &[0],
        &[ZeroConeT(1)],
        &[],
        &[1.0],
        &[1.0],
        &[f64::INFINITY],
    );
    assert_eq!(sol.status, SolverStatus::PrimalInfeasible);
    assert!(sol.obj_val.is_nan());
}

/// b = -inf in zero cone (a'x = -inf) → PrimalInfeasible
#[test]
fn test_compiled_zero_cone_neg_inf_infeasible() {
    let sol = compiled_solve(
        1,
        1,
        &[0, 0],
        &[],
        &[0, 1],
        &[0],
        &[ZeroConeT(1)],
        &[],
        &[1.0],
        &[1.0],
        &[-f64::INFINITY],
    );
    assert_eq!(sol.status, SolverStatus::PrimalInfeasible);
    assert!(sol.obj_val.is_nan());
}

// ============================================================================
// CompiledSolver: mixed cones with inf
// ============================================================================

/// Zero cone (equality) + nonneg with +inf → should solve
/// min x s.t. x = 1 (zero cone), x <= inf (nonneg)
#[test]
fn test_compiled_mixed_zero_nonneg_pos_inf() {
    let sol = compiled_solve(
        1,
        2,
        &[0, 0],
        &[],
        &[0, 1, 2],
        &[0, 0],
        &[ZeroConeT(1), NonnegativeConeT(1)],
        &[],
        &[1.0, 1.0],
        &[0.0],
        &[1.0, f64::INFINITY],
    );
    assert_eq!(sol.status, SolverStatus::Solved);
    assert!(
        (sol.x[0] - 1.0).abs() <= 1e-6,
        "x = {}, expected 1.0",
        sol.x[0]
    );
}

/// Zero cone with inf + nonneg with finite → infeasible (zero cone has inf)
#[test]
fn test_compiled_mixed_zero_inf_nonneg_finite_infeasible() {
    let sol = compiled_solve(
        1,
        2,
        &[0, 0],
        &[],
        &[0, 1, 2],
        &[0, 0],
        &[ZeroConeT(1), NonnegativeConeT(1)],
        &[],
        &[1.0, -1.0],
        &[1.0],
        &[f64::INFINITY, 0.0],
    );
    assert_eq!(sol.status, SolverStatus::PrimalInfeasible);
}

// ============================================================================
// CompiledSolver: batch with mixed feasible/infeasible
// ============================================================================

/// Batch of 2: first problem feasible (+inf), second infeasible (-inf in nonneg)
#[test]
fn test_compiled_batch_mixed_feasibility() {
    let n = 1usize;
    let m = 1usize;
    let P_ro = vec![0, 0];
    let P_ci: Vec<usize> = vec![];
    let A_ro = vec![0, 1];
    let A_ci = vec![0];
    let cones = vec![NonnegativeConeT(1)];

    let settings = DefaultSettings::default();
    let mut solver =
        CompiledSolver::new(n, m, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, 2, false).unwrap();

    // Both problems share A = [-1]
    solver.setup(&[vec![], vec![]], &[vec![-1.0], vec![-1.0]]);

    let qs = vec![vec![1.0], vec![1.0]];
    // Problem 0: b = 0 (x >= 0), should solve with x = 0
    // Problem 1: b = -inf (impossible), should be PrimalInfeasible
    let bs = vec![vec![0.0], vec![-f64::INFINITY]];
    let solutions = solver.solve(&qs, &bs).unwrap();

    assert_eq!(solutions[0].status, SolverStatus::Solved);
    assert!(f64::abs(solutions[0].x[0]) <= 1e-6);

    assert_eq!(solutions[1].status, SolverStatus::PrimalInfeasible);
    assert!(solutions[1].obj_val.is_nan());
}

// ============================================================================
// CompiledSolver: no inf (regression — normal problems still work)
// ============================================================================

/// Normal problem without any inf in b (regression test).
#[test]
fn test_compiled_no_inf_regression() {
    // min (1/2)x'Px + q'x s.t. x >= 0
    let sol = compiled_solve(
        1,
        1,
        &[0, 1],
        &[0],
        &[0, 1],
        &[0],
        &[NonnegativeConeT(1)],
        &[2.0],
        &[-1.0],
        &[1.0],
        &[0.0],
    );
    assert_eq!(sol.status, SolverStatus::Solved);
    assert!(f64::abs(sol.x[0]) <= 1e-6);
}
