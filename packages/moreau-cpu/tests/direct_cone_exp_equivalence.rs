#![allow(non_snake_case)]
//! Integration tests for ExpCone direct-x.
//!
//! ExpCone is asymmetric (`F ≠ F*`), so the symmetric primal↔dual swap
//! defaults that the other direct-x cones rely on are wrong here. The
//! cone overrides `direct_x_*` to use the primal-barrier Hessian
//! `μ·∇²F_primal(x)` directly in the augmented (1,1) KKT block — this
//! is the natural primal-IPM augmentation, not slack's `μ·H_dual(z)`
//! (which slack uses inside `A^T·Hs⁻¹·A`, a different role). Direct-x
//! ExpCone uses dual-only scaling; slack's primal-dual NT scaling
//! formula does not port to direct-x by argument substitution.

use moreau::{algebra::*, solver::*};

fn build_simple_exp_qp() -> (CscMatrix<f64>, Vec<f64>) {
    // min 0.5 ||x - target||^2 s.t. x ∈ ExpCone
    // Choose target inside the cone: x_target = (1, 1, e) ⇒ w = v·exp(u/v).
    // Pick something boring well inside: x_target = (0, 1, 5).
    let n = 3usize;
    let P = CscMatrix::<f64>::new(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0, 1.0, 1.0],
    );
    let q = vec![0.0, -1.0, -5.0];
    (P, q)
}

#[test]
fn test_exp_direct_x_scaffolding_does_not_crash() {
    // Smoke test: ExpCone direct-x scaffolding (variant + factory + trait
    // overrides) should at least construct, attempt to solve without
    // panicking on `unreachable!()`, and produce an iterate that lies in
    // the cone (primal feasible) — even if it doesn't yet drive μ → 0.
    let (P, q) = build_simple_exp_qp();
    let n = 3usize;

    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let dir_cones = vec![SupportedXConeT::ExponentialXConeT((0..n).collect())];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &dir_cones, settings)
            .expect("ExpCone direct-x construction must succeed");
    solver.solve();

    // Whatever the final status, the iterate must be primal-feasible
    // (it's never updated past a point that violates the cone).
    let x = &solver.solution.x;
    let (u, v, w) = (x[0], x[1], x[2]);
    assert!(v > -1e-6, "v must remain ≥ 0, got {}", v);
    if v > 1e-6 {
        let rhs = v * (u / v).exp();
        assert!(
            w + 1e-3 >= rhs,
            "exp cone violation: w={}, v·exp(u/v)={}",
            w,
            rhs,
        );
    }
}

#[test]
fn test_exp_direct_x_solves_simple_problem() {
    let (P, q) = build_simple_exp_qp();
    let n = 3usize;
    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let dir_cones = vec![SupportedXConeT::ExponentialXConeT((0..n).collect())];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &dir_cones, settings)
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

/// Many stacked ExpCone direct-x cones must all converge cleanly — a
/// regression guard for the closed-form `update_primal_grad_H` fix.
/// The previous implementation used the Karimi-Tunçel Wright-Omega
/// gradient/Hessian (matching slack), but mixing K-T derivatives with
/// direct-x's `+Hs` augmentation made the IPM slower (12→15 iters at
/// K=100). The closed-form 3-LB barrier
/// `F(s) = -log(s[1]·log(s[2]/s[1]) - s[0]) - log(s[1]) - log(s[2])`
/// matches slack's iter count and beats it on time.
#[test]
fn test_exp_direct_x_stacked_solves() {
    const K: usize = 100;
    let n = 3 * K;
    let P = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![1.0; n]);
    let mut q = vec![0.0; n];
    for k in 0..K {
        q[3 * k] = 0.0;
        q[3 * k + 1] = -1.0;
        q[3 * k + 2] = -5.0;
    }

    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let dir_cones: Vec<SupportedXConeT> = (0..K)
        .map(|k| SupportedXConeT::ExponentialXConeT(vec![3 * k, 3 * k + 1, 3 * k + 2]))
        .collect();

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &dir_cones, settings)
            .unwrap();
    solver.solve();
    assert!(
        matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "K={K} stacked ExpCone direct-x expected Solved/AlmostSolved, got {:?}",
        solver.solution.status,
    );
}

#[test]
fn test_exp_direct_x_matches_slack() {
    // Same problem solved via slack form. Both should give the same
    // primal x.
    let (P, q) = build_simple_exp_qp();
    let n = 3usize;

    // Slack form: -x + s = 0, s ∈ ExpCone.
    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    let A_slack = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![-1.0; n]);
    let b_slack = vec![0.0; n];
    let cones_slack = vec![ExponentialConeT()];
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
    let dir_cones = vec![SupportedXConeT::ExponentialXConeT((0..n).collect())];
    let mut d_solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &dir_cones, settings)
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
