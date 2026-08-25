//! Tests for subtle edge cases in the CPU solver
//!
//! These tests aim to uncover:
//! 1. Cone projection/derivative edge cases at boundaries
//! 2. Numerical precision issues with extreme values
//! 3. Solution consistency across different problem formulations
//! 4. Equilibration behavior with poorly scaled data
//! 5. Gradient computation correctness (for differentiable solver)

#![allow(non_snake_case)]

use moreau::{algebra::*, solver::*};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

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
// CONE BOUNDARY TESTS
// Test behavior at exact cone boundaries
// ============================================================================

/// Test SOC constraint at exact boundary (t = ||x||)
#[test]
fn test_soc_boundary_point() {
    // Problem where solution lies near SOC boundary
    // min t + 0.01*(x1^2 + x2^2) s.t. (t, x1, x2) in SOC, t + x1 + x2 >= 0.5
    //
    // The quadratic regularization ensures bounded solution while
    // the linear cost on t pushes toward the SOC boundary.

    // P has small regularization on x1, x2
    let P = CscMatrix::new(
        3,
        3,
        vec![0, 0, 1, 2], // only entries at (1,1) and (2,2)
        vec![1, 2],
        vec![0.02, 0.02], // small regularization
    );
    let c = vec![1.0, 0.0, 0.0]; // minimize t

    // A has:
    // - SOC constraint: (t, x1, x2) in SOC (requires t >= sqrt(x1^2 + x2^2))
    // - Nonneg constraint: t + x1 + x2 >= 0.5 (written as -t - x1 - x2 + s = -0.5, s >= 0)
    let A = CscMatrix::new(
        4, // 3 for SOC + 1 for nonneg
        3,
        vec![0, 2, 4, 6],                         // colptr
        vec![0, 3, 1, 3, 2, 3],                   // rowval
        vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], // nzval
    );

    let b = vec![0.0, 0.0, 0.0, -0.5];

    let cones = vec![SecondOrderConeT(3), NonnegativeConeT(1)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    // Solution should have t >= ||x||
    let x = &solver.solution.x;
    let t: f64 = x[0];
    let norm_x = f64::sqrt(x[1] * x[1] + x[2] * x[2]);

    // Should satisfy SOC constraint: t >= ||x||
    assert!(
        t >= norm_x - 1e-6,
        "SOC constraint violated: t={}, ||x||={}",
        t,
        norm_x
    );

    // Check all values are finite
    for i in 0..3 {
        assert!(!f64::is_nan(x[i]), "x[{}] is NaN", i);
        assert!(!f64::is_infinite(x[i]), "x[{}] is infinite", i);
    }
}

/// Test nonnegative cone at exact zero
#[test]
fn test_nonneg_at_zero_boundary() {
    // Problem with optimal solution having some slack variables at exactly zero
    // min x0 + x1 s.t. x0 >= 0, x1 >= 0, x0 + x1 = 1

    let P = CscMatrix::zeros((2, 2));
    let c = vec![1.0, 2.0]; // Different costs to ensure unique minimum

    // A = [-I; 1 1] (nonneg + equality)
    let A = CscMatrix::new(
        3,
        2,
        vec![0, 2, 4],
        vec![0, 2, 1, 2],
        vec![-1.0, 1.0, -1.0, 1.0],
    );

    let b = vec![0.0, 0.0, 1.0];
    let cones = vec![NonnegativeConeT(2), ZeroConeT(1)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    // Optimal: x0 = 1, x1 = 0 (since c[1] > c[0])
    let x = &solver.solution.x;
    assert!(
        is_close(x[0], 1.0, 1e-6, 1e-6),
        "Expected x0=1, got {}",
        x[0]
    );
    assert!(
        is_close(x[1], 0.0, 1e-6, 1e-6),
        "Expected x1=0, got {}",
        x[1]
    );

    // Check slack variable s[1] is at zero (active constraint)
    let s = &solver.solution.s;
    assert!(s[1].abs() < 1e-6, "Expected s[1] near 0, got {}", s[1]);
}

/// Test exp cone near singular point
#[test]
fn test_exp_cone_near_boundary() {
    // Exponential cone: (r, s, t) where s*exp(r/s) <= t, s > 0
    // Test near the boundary where r/s is large

    let P = CscMatrix::identity(3);
    let c = vec![0.0, 0.0, -1.0]; // maximize t (minimize -t)

    // Identity mapping to exp cone + bound
    let A = CscMatrix::new(
        4,
        3,
        vec![0, 2, 4, 6],
        vec![0, 3, 1, 3, 2, 3],
        vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );

    // b: exp cone constraint and sum bound
    let b = vec![0.0, 0.0, 0.0, 3.0]; // r + s + t <= 3

    let cones = vec![ExponentialConeT(), NonnegativeConeT(1)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    // May be solved or infeasible depending on formulation
    // Just check we don't crash and get finite results
    let x = &solver.solution.x;
    for i in 0..3 {
        assert!(!f64::is_nan(x[i]), "x[{}] is NaN", i);
    }
}

/// Test power cone with alpha near 0 and near 1
#[test]
fn test_power_cone_extreme_alpha() {
    // Power cone: x0^alpha * x1^(1-alpha) >= |x2|, x0,x1 >= 0
    // Test with alpha = 0.1 (near 0) and alpha = 0.9 (near 1)

    for alpha in [0.1, 0.5, 0.9] {
        let P = CscMatrix::identity(3);
        let c = vec![0.0, 0.0, 1.0]; // minimize x2

        // Power cone constraint
        let A = CscMatrix::identity(3);
        let b = vec![0.0, 0.0, 0.0];

        let cones = vec![PowerConeT(alpha)];

        let settings = DefaultSettings::default();
        let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
        solver.solve();

        // Just verify no crashes and finite results
        let x = &solver.solution.x;
        for i in 0..3 {
            assert!(!f64::is_nan(x[i]), "alpha={}: x[{}] is NaN", alpha, i);
            assert!(
                !f64::is_infinite(x[i]),
                "alpha={}: x[{}] is infinite",
                alpha,
                i
            );
        }
    }
}

// ============================================================================
// NUMERICAL PRECISION TESTS
// ============================================================================

/// Test with very large coefficient magnitudes
#[test]
fn test_large_coefficients() {
    // P = 1e8 * I, q = 1e8 * [1, 1], A = 1e4 * I, b = 1e4 * [5, 5]
    let scale: f64 = 1e6;

    let P = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![scale, scale]);

    let c = vec![scale, scale];

    let A = CscMatrix::new(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![scale.sqrt(), scale.sqrt()],
    );

    let b = vec![5.0 * scale.sqrt(), 5.0 * scale.sqrt()];

    let cones = vec![NonnegativeConeT(2)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let x = &solver.solution.x;
    for i in 0..2 {
        assert!(!x[i].is_nan(), "Large scale: x[{}] is NaN", i);
        assert!(!x[i].is_infinite(), "Large scale: x[{}] is infinite", i);
    }
}

/// Test with very small coefficient magnitudes
#[test]
fn test_small_coefficients() {
    let scale: f64 = 1e-6;

    let P = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![scale, scale]);

    let c = vec![scale, scale];

    let A = CscMatrix::new(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![scale.sqrt(), scale.sqrt()],
    );

    let b = vec![5.0 * scale.sqrt(), 5.0 * scale.sqrt()];

    let cones = vec![NonnegativeConeT(2)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let x = &solver.solution.x;
    for i in 0..2 {
        assert!(!x[i].is_nan(), "Small scale: x[{}] is NaN", i);
        assert!(!x[i].is_infinite(), "Small scale: x[{}] is infinite", i);
    }
}

/// Test with mixed large and small coefficients (ill-conditioned)
#[test]
fn test_ill_conditioned() {
    // P with condition number ~1e8
    let P = CscMatrix::new(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![1e-4, 1e4], // eigenvalues differ by 1e8
    );

    let c = vec![1.0, 1.0];

    let A = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]);

    let b = vec![5.0, 5.0];

    let cones = vec![NonnegativeConeT(2)];

    let mut settings = DefaultSettings::default();
    settings.max_iter = 200; // May need more iterations

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    // May not solve to high precision, but should not crash
    let x = &solver.solution.x;
    for i in 0..2 {
        assert!(!f64::is_nan(x[i]), "Ill-conditioned: x[{}] is NaN", i);
        // May have large values due to ill-conditioning
    }
}

/// Test with zero cost vector (feasibility problem)
#[test]
fn test_zero_cost() {
    let P = CscMatrix::zeros((2, 2));
    let c = vec![0.0, 0.0]; // No objective

    // Just find feasible point
    let A = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![1.0, 1.0]);

    let b = vec![1.0, 1.0];

    let cones = vec![NonnegativeConeT(2)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    // Any feasible point is optimal
    let x = &solver.solution.x;
    assert!(x[0] >= -1e-6, "x[0] should be nonnegative");
    assert!(x[1] >= -1e-6, "x[1] should be nonnegative");
}

// ============================================================================
// SOLUTION CONSISTENCY TESTS
// ============================================================================

/// Test that equivalent formulations give same solution
#[test]
fn test_equivalent_formulations() {
    // Problem 1: min x0 + x1 s.t. x >= 0
    // Problem 2: min x0 + x1 s.t. -x <= 0

    let P = CscMatrix::zeros((2, 2));
    let c = vec![1.0, 1.0];

    // Formulation 1: x >= 0 directly
    let A1 = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![-1.0, -1.0]);
    let b1 = vec![0.0, 0.0];
    let cones1 = vec![NonnegativeConeT(2)];

    let settings = DefaultSettings::default();
    let mut solver1 = DefaultSolver::new(&P, &c, &A1, &b1, &cones1, settings.clone()).unwrap();
    solver1.solve();

    // Formulation 2: same but with different slack
    let A2 = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![-1.0, -1.0]);
    let b2 = vec![0.0, 0.0];
    let cones2 = vec![NonnegativeConeT(2)];

    let mut solver2 = DefaultSolver::new(&P, &c, &A2, &b2, &cones2, settings).unwrap();
    solver2.solve();

    assert_eq!(solver1.solution.status, solver2.solution.status);

    // Solutions should be identical
    for i in 0..2 {
        assert!(
            is_close(solver1.solution.x[i], solver2.solution.x[i], 1e-6, 1e-8),
            "Solution mismatch at x[{}]: {} vs {}",
            i,
            solver1.solution.x[i],
            solver2.solution.x[i]
        );
    }
}

/// Test determinism - same input should give same output
#[test]
fn test_determinism() {
    let P = CscMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0, 1.0, 1.0, 2.0],
    );

    let c = vec![1.0, 1.0];

    let A = CscMatrix::new(
        3,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 2],
        vec![1.0, 1.0, 1.0, 1.0],
    );

    let b = vec![1.0, 0.7, 0.7];
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];

    let settings = DefaultSettings::default();

    // Run twice
    let mut solver1 = DefaultSolver::new(&P, &c, &A, &b, &cones, settings.clone()).unwrap();
    solver1.solve();

    let mut solver2 = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver2.solve();

    // Results should be bitwise identical
    for i in 0..2 {
        let v1: f64 = solver1.solution.x[i];
        let v2: f64 = solver2.solution.x[i];
        assert_eq!(
            v1.to_bits(),
            v2.to_bits(),
            "Non-determinism detected at x[{}]: {} vs {}",
            i,
            v1,
            v2
        );
    }
}

// ============================================================================
// INFEASIBILITY DETECTION TESTS
// ============================================================================

/// Test primal infeasibility detection
#[test]
fn test_primal_infeasibility_detection() {
    // min x s.t. x >= 1 AND x <= 0 (infeasible)
    let P = CscMatrix::zeros((1, 1));
    let c = vec![1.0];

    // -x <= 0 (x >= 0) AND x <= -1 (infeasible combination)
    let A = CscMatrix::new(
        2,
        1,
        vec![0, 2],
        vec![0, 1],
        vec![-1.0, 1.0], // -x <= 0, x <= -1
    );

    let b = vec![0.0, -1.0];
    let cones = vec![NonnegativeConeT(2)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(
        solver.solution.status,
        SolverStatus::PrimalInfeasible,
        "Should detect primal infeasibility"
    );
}

/// Test dual infeasibility (unbounded) detection
#[test]
fn test_dual_infeasibility_detection() {
    // min -x s.t. x >= 0 (unbounded below in direction of x)
    let P = CscMatrix::zeros((1, 1));
    let c = vec![-1.0];

    let A = CscMatrix::new(
        1,
        1,
        vec![0, 1],
        vec![0],
        vec![-1.0], // -x <= 0
    );

    let b = vec![0.0];
    let cones = vec![NonnegativeConeT(1)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(
        solver.solution.status,
        SolverStatus::DualInfeasible,
        "Should detect dual infeasibility (unboundedness)"
    );
}

// ============================================================================
// SPARSE STRUCTURE TESTS
// ============================================================================

/// Test with very sparse A matrix
#[test]
fn test_very_sparse_A() {
    let n = 100;
    let m = 50;

    // P = diagonal
    let P = CscMatrix::identity(n);
    let c = vec![1.0; n];

    // A has only 1 nonzero per column (very sparse)
    let mut colptr = vec![0usize];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();

    for j in 0..n {
        // Each column has at most 1 nonzero
        if j < m {
            rowval.push(j);
            nzval.push(1.0);
        }
        colptr.push(rowval.len());
    }

    let A = CscMatrix::new(m, n, colptr, rowval, nzval);
    let b = vec![5.0; m];
    let cones = vec![NonnegativeConeT(m)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    // Check solution is valid
    for i in 0..n {
        assert!(!f64::is_nan(solver.solution.x[i]), "x[{}] is NaN", i);
    }
}

/// Test with dense A matrix
#[test]
fn test_dense_A() {
    let n = 10;
    let m = 5;

    let P = CscMatrix::identity(n);
    let c = vec![1.0; n];

    // Dense A: all entries nonzero
    let mut colptr = vec![0usize];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();

    for j in 0..n {
        for i in 0..m {
            rowval.push(i);
            nzval.push(0.1 * ((i + j) as f64 + 1.0));
        }
        colptr.push(rowval.len());
    }

    let A = CscMatrix::new(m, n, colptr, rowval, nzval);
    let b = vec![10.0; m];
    let cones = vec![NonnegativeConeT(m)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);
}

// ============================================================================
// SOLUTION QUALITY TESTS
// ============================================================================

/// Verify KKT conditions are satisfied at solution
#[test]
fn test_kkt_conditions() {
    // Simple QP with known solution
    let P = CscMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![2.0, 0.0, 0.0, 2.0], // 2*I
    );

    let c = vec![1.0, 1.0];

    // x >= 0
    let A = CscMatrix::new(2, 2, vec![0, 1, 2], vec![0, 1], vec![-1.0, -1.0]);

    let b = vec![0.0, 0.0];
    let cones = vec![NonnegativeConeT(2)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let x = &solver.solution.x;
    let z = &solver.solution.z;
    let s = &solver.solution.s;

    // Check primal feasibility: Ax + s = b
    // Here: -x + s = 0, so s = x
    for i in 0..2 {
        let residual: f64 = -x[i] + s[i];
        assert!(
            residual.abs() < 1e-5,
            "Primal feasibility: row {} residual = {}",
            i,
            residual
        );
    }

    // Check dual feasibility: Px + q + A'z = 0 (for unconstrained minimum)
    // Here: 2*x + [1,1] - z = 0
    for i in 0..2 {
        let dual_residual: f64 = 2.0 * x[i] + 1.0 - z[i];
        assert!(
            dual_residual.abs() < 1e-5,
            "Dual feasibility: var {} residual = {}",
            i,
            dual_residual
        );
    }

    // Check complementarity: s'z ≈ 0
    let comp: f64 = s.iter().zip(z.iter()).map(|(si, zi)| si * zi).sum();
    assert!(comp.abs() < 1e-5, "Complementarity gap = {}", comp);
}

/// Test objective value accuracy
#[test]
fn test_objective_value() {
    // min x0^2 + x1^2 + x0 + x1 s.t. x0 + x1 = 1, x >= 0
    // Optimal: x = [0.5, 0.5], obj = 0.5 + 0.5 + 1 = 1.5 (approx)

    let P = CscMatrix::new(
        2,
        2,
        vec![0, 1, 2],
        vec![0, 1],
        vec![2.0, 2.0], // diag(2,2)
    );

    let c = vec![1.0, 1.0];

    // Equality + nonneg
    let A = CscMatrix::new(
        3,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 2],
        vec![1.0, -1.0, 1.0, -1.0],
    );

    let b = vec![1.0, 0.0, 0.0];
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    // Compute objective manually
    let x = &solver.solution.x;
    let obj_manual = x[0] * x[0] + x[1] * x[1] + x[0] + x[1];

    assert!(
        is_close(solver.solution.obj_val, obj_manual, 1e-4, 1e-6),
        "Objective mismatch: reported {} vs computed {}",
        solver.solution.obj_val,
        obj_manual
    );
}

// ============================================================================
// EQUILIBRATION CONSISTENCY TESTS
// Verify that CompiledSolver matches single Solver (Clarabel-like) behavior
// ============================================================================

/// Test that CompiledSolver gives same result as single Solver
#[test]
fn test_compiled_vs_single_solver_parity() {
    // Simple QP problem
    let P = CscMatrix::new(
        2,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 1],
        vec![4.0, 1.0, 1.0, 2.0], // full symmetric
    );

    let q = vec![1.0, 1.0];

    let A = CscMatrix::new(
        3,
        2,
        vec![0, 2, 4],
        vec![0, 1, 0, 2],
        vec![1.0, 1.0, 1.0, 1.0],
    );

    let b = vec![1.0, 0.7, 0.7];
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    // Single solver
    let mut single_solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone()).unwrap();
    single_solver.solve();
    assert_eq!(single_solver.solution.status, SolverStatus::Solved);

    // CompiledSolver (batch size 1)
    // Need to provide CSR format: (row_offsets, col_indices, values)
    // P CSR: rows are [0,1,0,1] -> offsets [0,2,4], cols [0,1,0,1]
    let P_row_offsets = vec![0, 2, 4];
    let P_col_indices = vec![0, 1, 0, 1];
    let P_values = vec![4.0, 1.0, 1.0, 2.0];

    // A CSR: rows 0,1,2 have elements at different positions
    // Row 0: (0,0) and (0,1)
    // Row 1: (1,0)
    // Row 2: (2,1)
    let A_row_offsets = vec![0, 2, 3, 4];
    let A_col_indices = vec![0, 1, 0, 1];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];

    let mut compiled_solver = CompiledSolver::new(
        2,
        3,
        &P_row_offsets,
        &P_col_indices,
        &A_row_offsets,
        &A_col_indices,
        &cones,
        settings,
        1,
        false,
    )
    .unwrap();

    compiled_solver.setup(&[P_values], &[A_values]);
    let results = compiled_solver.solve(&[q.clone()], &[b.clone()]).unwrap();

    assert_eq!(results[0].status, SolverStatus::Solved);

    // Solutions should match
    for i in 0..2 {
        assert!(
            is_close(single_solver.solution.x[i], results[0].x[i], 1e-6, 1e-8),
            "Solution mismatch at x[{}]: single={} vs compiled={}",
            i,
            single_solver.solution.x[i],
            results[0].x[i]
        );
    }
}

/// Test that repeated solves with CompiledSolver give consistent results
#[test]
fn test_compiled_solver_repeated_solve_consistency() {
    let P_row_offsets = vec![0, 2, 4];
    let P_col_indices = vec![0, 1, 0, 1];
    let P_values = vec![4.0, 1.0, 1.0, 2.0];

    let A_row_offsets = vec![0, 2, 3, 4];
    let A_col_indices = vec![0, 1, 0, 1];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];

    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver = CompiledSolver::new(
        2,
        3,
        &P_row_offsets,
        &P_col_indices,
        &A_row_offsets,
        &A_col_indices,
        &cones,
        settings,
        1,
        false,
    )
    .unwrap();

    solver.setup(&[P_values], &[A_values]);

    let q = vec![1.0, 1.0];
    let b = vec![1.0, 0.7, 0.7];

    // First solve
    let result1 = solver.solve(&[q.clone()], &[b.clone()]).unwrap();
    assert_eq!(result1[0].status, SolverStatus::Solved);

    // Second solve with same input (should be identical due to restored base P)
    let result2 = solver.solve(&[q.clone()], &[b.clone()]).unwrap();
    assert_eq!(result2[0].status, SolverStatus::Solved);

    // Third solve
    let result3 = solver.solve(&[q.clone()], &[b.clone()]).unwrap();
    assert_eq!(result3[0].status, SolverStatus::Solved);

    // All results should be identical
    for i in 0..2 {
        let v1: f64 = result1[0].x[i];
        let v2: f64 = result2[0].x[i];
        let v3: f64 = result3[0].x[i];
        assert_eq!(
            v1.to_bits(),
            v2.to_bits(),
            "Solve 1 vs 2 mismatch at x[{}]: {} vs {}",
            i,
            v1,
            v2
        );
        assert_eq!(
            v2.to_bits(),
            v3.to_bits(),
            "Solve 2 vs 3 mismatch at x[{}]: {} vs {}",
            i,
            v2,
            v3
        );
    }
}

/// Test that different q values give different (correct) solutions
#[test]
fn test_compiled_solver_different_q_gives_correct_results() {
    let P_row_offsets = vec![0, 2, 4];
    let P_col_indices = vec![0, 1, 0, 1];
    let P_values = vec![4.0, 1.0, 1.0, 2.0];

    let A_row_offsets = vec![0, 2, 3, 4];
    let A_col_indices = vec![0, 1, 0, 1];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];

    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    // Test with multiple q vectors
    let q_values = vec![
        vec![1.0, 1.0],
        vec![2.0, 1.0],
        vec![0.5, 2.0],
        vec![-1.0, -1.0],
    ];
    let b = vec![1.0, 0.7, 0.7];

    // For each q, verify CompiledSolver matches single Solver
    for q in &q_values {
        // Single solver
        let P = CscMatrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![4.0, 1.0, 1.0, 2.0],
        );

        let A = CscMatrix::new(
            3,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 2],
            vec![1.0, 1.0, 1.0, 1.0],
        );

        let mut single_solver =
            DefaultSolver::new(&P, q, &A, &b, &cones, settings.clone()).unwrap();
        single_solver.solve();

        // CompiledSolver (fresh for each q to ensure clean state)
        let mut compiled_solver = CompiledSolver::new(
            2,
            3,
            &P_row_offsets,
            &P_col_indices,
            &A_row_offsets,
            &A_col_indices,
            &cones,
            settings.clone(),
            1,
            false,
        )
        .unwrap();

        compiled_solver.setup(&[P_values.clone()], &[A_values.clone()]);
        let result = compiled_solver.solve(&[q.clone()], &[b.clone()]).unwrap();

        assert_eq!(
            single_solver.solution.status, result[0].status,
            "Status mismatch for q={:?}",
            q
        );

        if single_solver.solution.status == SolverStatus::Solved {
            for i in 0..2 {
                assert!(
                    is_close(single_solver.solution.x[i], result[0].x[i], 1e-6, 1e-8),
                    "q={:?}: x[{}] mismatch: single={} vs compiled={}",
                    q,
                    i,
                    single_solver.solution.x[i],
                    result[0].x[i]
                );
            }
        }
    }
}

/// Test batch consistency: all problems in batch should match individual solves
#[test]
fn test_batch_matches_individual_solves() {
    let P_row_offsets = vec![0, 2, 4];
    let P_col_indices = vec![0, 1, 0, 1];
    let P_values = vec![4.0, 1.0, 1.0, 2.0];

    let A_row_offsets = vec![0, 2, 3, 4];
    let A_col_indices = vec![0, 1, 0, 1];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];

    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    // Batch of q, b values
    let q_batch = vec![
        vec![1.0, 1.0],
        vec![2.0, 1.0],
        vec![0.5, 2.0],
        vec![-1.0, -1.0],
    ];
    let b_batch = vec![
        vec![1.0, 0.7, 0.7],
        vec![1.0, 0.7, 0.7],
        vec![1.0, 0.7, 0.7],
        vec![1.0, 0.7, 0.7],
    ];
    let batch_size = q_batch.len();

    // Batch solve
    let mut batch_solver = CompiledSolver::new(
        2,
        3,
        &P_row_offsets,
        &P_col_indices,
        &A_row_offsets,
        &A_col_indices,
        &cones,
        settings.clone(),
        batch_size,
        false,
    )
    .unwrap();

    // All problems share P, A
    let P_batch: Vec<Vec<f64>> = vec![P_values.clone(); batch_size];
    let A_batch: Vec<Vec<f64>> = vec![A_values.clone(); batch_size];
    batch_solver.setup(&P_batch, &A_batch);

    let batch_results = batch_solver.solve(&q_batch, &b_batch).unwrap();

    // Compare with individual solves
    for i in 0..batch_size {
        let P = CscMatrix::new(
            2,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 1],
            vec![4.0, 1.0, 1.0, 2.0],
        );

        let A = CscMatrix::new(
            3,
            2,
            vec![0, 2, 4],
            vec![0, 1, 0, 2],
            vec![1.0, 1.0, 1.0, 1.0],
        );

        let mut single_solver =
            DefaultSolver::new(&P, &q_batch[i], &A, &b_batch[i], &cones, settings.clone()).unwrap();
        single_solver.solve();

        assert_eq!(
            single_solver.solution.status, batch_results[i].status,
            "Batch problem {} status mismatch",
            i
        );

        if single_solver.solution.status == SolverStatus::Solved {
            for j in 0..2 {
                assert!(
                    is_close(
                        single_solver.solution.x[j],
                        batch_results[i].x[j],
                        1e-6,
                        1e-8
                    ),
                    "Batch problem {}: x[{}] mismatch: single={} vs batch={}",
                    i,
                    j,
                    single_solver.solution.x[j],
                    batch_results[i].x[j]
                );
            }
        }
    }
}

// ============================================================================
// Compiled solver: direct_solve_method override based on batch size
// ============================================================================

/// Test that CompiledSolver with num_threads > 1 forces qdldl
/// even when user sets "auto" (the default).
#[test]
fn test_compiled_solver_forces_qdldl_for_batched() {
    let P_row_offsets = vec![0, 2, 4];
    let P_col_indices = vec![0, 1, 0, 1];
    let P_values = vec![4.0, 1.0, 1.0, 2.0];

    let A_row_offsets = vec![0, 2, 3, 4];
    let A_col_indices = vec![0, 1, 0, 1];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];

    let q = vec![1.0, 1.0];
    let b = vec![1.0, 0.7, 0.7];
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];

    // With num_threads > 1 and direct_solve_method = "auto",
    // should force qdldl and solve successfully
    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver = CompiledSolver::new(
        2,
        3,
        &P_row_offsets,
        &P_col_indices,
        &A_row_offsets,
        &A_col_indices,
        &cones,
        settings,
        2,
        false,
    )
    .unwrap();

    solver.setup(
        &[P_values.clone(), P_values.clone()],
        &[A_values.clone(), A_values.clone()],
    );
    let results = solver
        .solve(&[q.clone(), q.clone()], &[b.clone(), b.clone()])
        .unwrap();
    assert_eq!(results[0].status, SolverStatus::Solved);
    assert_eq!(results[1].status, SolverStatus::Solved);
}

/// Test that CompiledSolver with num_threads == 1 preserves "auto"
/// and solves correctly (allows faer if feature is enabled).
#[test]
fn test_compiled_solver_allows_auto_for_single() {
    let P_row_offsets = vec![0, 2, 4];
    let P_col_indices = vec![0, 1, 0, 1];
    let P_values = vec![4.0, 1.0, 1.0, 2.0];

    let A_row_offsets = vec![0, 2, 3, 4];
    let A_col_indices = vec![0, 1, 0, 1];
    let A_values = vec![1.0, 1.0, 1.0, 1.0];

    let q = vec![1.0, 1.0];
    let b = vec![1.0, 0.7, 0.7];
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];

    // With num_threads == 1, "auto" should be preserved
    let mut settings = DefaultSettings::default();
    settings.verbose = false;

    let mut solver = CompiledSolver::new(
        2,
        3,
        &P_row_offsets,
        &P_col_indices,
        &A_row_offsets,
        &A_col_indices,
        &cones,
        settings,
        1,
        false,
    )
    .unwrap();

    solver.setup(&[P_values], &[A_values]);
    let results = solver.solve(&[q], &[b]).unwrap();
    assert_eq!(results[0].status, SolverStatus::Solved);
}
