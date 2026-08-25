#![allow(non_snake_case)]
//! Integration tests for 3D PowerCone direct-x.
//!
//! PowerCone (with parameter α ∈ (0, 1)) is asymmetric (`F ≠ F*`), so the
//! symmetric primal↔dual swap defaults that the other direct-x cones rely
//! on are wrong here. The cone overrides `direct_x_*` to use the
//! primal-barrier Hessian `μ·∇²F_primal(x)` directly in the augmented
//! (1,1) KKT block (same pattern as ExpCone direct-x — see
//! `xcone_exp_equivalence.rs`).

use moreau::{algebra::*, solver::*};

const ALPHA: f64 = 0.4;

fn build_simple_pow_qp() -> (CscMatrix<f64>, Vec<f64>) {
    // min 0.5 ||x - target||^2 s.t. (x[0], x[1], x[2]) ∈ Pow(α).
    // Power cone constraint: x[0]^α · x[1]^(1-α) >= |x[2]|, x[0], x[1] >= 0.
    // Choose target inside the cone: (2, 3, 1) with α=0.4 satisfies
    // 2^0.4 · 3^0.6 ≈ 1.32 · 1.93 ≈ 2.55 >= 1. Interior.
    let n = 3usize;
    let P = CscMatrix::<f64>::new(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0, 1.0, 1.0],
    );
    let q = vec![-2.0, -3.0, -1.0];
    (P, q)
}

#[test]
fn test_pow_direct_x_scaffolding_does_not_crash() {
    let (P, q) = build_simple_pow_qp();
    let n = 3usize;

    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let x_cones = vec![SupportedXConeT::PowerXConeT((0..n).collect(), ALPHA)];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &x_cones, settings)
            .expect("PowerCone direct-x construction must succeed");
    solver.solve();

    // Whatever the final status, the iterate must be primal-feasible.
    let x = &solver.solution.x;
    assert!(x[0] > -1e-6, "x[0] must remain >= 0, got {}", x[0]);
    assert!(x[1] > -1e-6, "x[1] must remain >= 0, got {}", x[1]);
    if x[0] > 1e-6 && x[1] > 1e-6 {
        let lhs = x[0].powf(ALPHA) * x[1].powf(1.0 - ALPHA);
        assert!(
            lhs + 1e-3 >= x[2].abs(),
            "power cone violation: x[0]^α x[1]^(1-α) = {}, |x[2]| = {}",
            lhs,
            x[2].abs(),
        );
    }
}

#[test]
fn test_pow_direct_x_solves_simple_problem() {
    let (P, q) = build_simple_pow_qp();
    let n = 3usize;
    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let x_cones = vec![SupportedXConeT::PowerXConeT((0..n).collect(), ALPHA)];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &x_cones, settings)
            .unwrap();
    solver.solve();
    assert!(
        matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "expected Solved/AlmostSolved, got {:?}",
        solver.solution.status,
    );
}

/// Many stacked PowerCone direct-x cones must all converge — a
/// regression guard for the closed-form `update_primal_grad_H` fix.
/// The previous `hessian_primal_3x3` (an "approximation" with a
/// hardcoded `h[(2,2)] = 1.0` fallback at `s[2]≈0` and zero
/// off-diagonals) made the augmented (1,1) Hessian wrong; the IPM
/// landed on bad step directions and stalled at K≥10 with status
/// `InsufficientProgress`. Single-cone (K=1) hid the bug because
/// the wrong-by-O(1) Hessian was still PD enough for one cone.
#[test]
fn test_pow_direct_x_stacked_solves() {
    const K: usize = 100;
    const ALPHA: f64 = 0.4;
    let n = 3 * K;
    let P = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![1.0; n]);
    let mut q = vec![0.0; n];
    for k in 0..K {
        q[3 * k] = -2.0;
        q[3 * k + 1] = -3.0;
        q[3 * k + 2] = -1.0;
    }

    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let x_cones: Vec<SupportedXConeT> = (0..K)
        .map(|k| SupportedXConeT::PowerXConeT(vec![3 * k, 3 * k + 1, 3 * k + 2], ALPHA))
        .collect();

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &x_cones, settings)
            .unwrap();
    solver.solve();
    assert!(
        matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "K={K} stacked PowerCone direct-x expected Solved/AlmostSolved, got {:?}",
        solver.solution.status,
    );
}

#[test]
fn test_pow_direct_x_matches_slack() {
    // Same problem solved via slack form. Both should give the same primal x.
    let (P, q) = build_simple_pow_qp();
    let n = 3usize;

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    // Slack form: -x + s = 0, s ∈ PowerCone(α).
    let A_slack = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![-1.0; n]);
    let b_slack = vec![0.0; n];
    let cones_slack = vec![PowerConeT(ALPHA)];
    let mut s_solver =
        DefaultSolver::new(&P, &q, &A_slack, &b_slack, &cones_slack, settings.clone()).unwrap();
    s_solver.solve();
    assert!(
        matches!(
            s_solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "slack form failed to solve",
    );

    // Direct-x form.
    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let x_cones = vec![SupportedXConeT::PowerXConeT((0..n).collect(), ALPHA)];
    let mut d_solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &x_cones, settings)
            .unwrap();
    d_solver.solve();
    assert!(
        matches!(
            d_solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "direct-x form failed: {:?}",
        d_solver.solution.status,
    );

    for i in 0..n {
        assert!(
            (s_solver.solution.x[i] - d_solver.solution.x[i]).abs() < 1e-3,
            "x[{}] disagrees: slack={}, direct-x={}",
            i,
            s_solver.solution.x[i],
            d_solver.solution.x[i],
        );
    }
}
