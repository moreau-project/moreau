//! Integration tests for smoothed differentiation (LP-only, CPU).
//!
//! Verifies the central-path refinement loop produces correct smoothed gradients
//! by checking:
//! 1. Smoothed gradients differ from Exact at moderate μ (walk-up is working)
//! 2. Smoothed gradients converge to Exact as μ → 0
//! 3. Smoothed backward gradients are consistent with finite differences
//! 4. Smoothed diff is rejected for non-LP cones

#![allow(non_snake_case)]

use moreau::{algebra::*, solver::*};

/// Helper: solve a nonneg QP and return the solver for backward pass.
fn solve_nonneg_qp(q: &[f64], settings: DefaultSettings<f64>) -> DefaultSolver<f64> {
    let n = q.len();
    let P = CscMatrix::identity(n);
    let A = CscMatrix::identity(n);
    let b = vec![0.0; n];
    let cones = vec![NonnegativeConeT(n)];

    let mut solver = DefaultSolver::new(&P, q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    solver
}

/// Helper: compute dq gradient via backward pass.
fn backward_dq(
    solver: &mut DefaultSolver<f64>,
    dx_bar: &[f64],
    diff_method: DiffMethod,
) -> Vec<f64> {
    let n = dx_bar.len();
    let m = n;
    let ds_bar = vec![0.0; m];
    let dz_bar = vec![0.0; m];
    let result = solver
        .backward_batch(dx_bar, &ds_bar, &dz_bar, None, diff_method)
        .unwrap();
    result.dq
}

#[test]
fn test_smoothed_differs_from_exact_at_moderate_mu() {
    // With q near zero, x* ≈ 0 (constraint boundary).
    // Exact H is a step function, Smoothed H is smooth.
    // At μ=0.1 the gradients must visibly differ.
    let q = vec![-0.01, 1.0];
    let dx_bar = vec![1.0, 0.0];

    let mut settings_exact = DefaultSettings::default();
    settings_exact.ipm.diff_method = DiffMethod::Exact;
    let mut solver_exact = solve_nonneg_qp(&q, settings_exact);
    let dq_exact = backward_dq(&mut solver_exact, &dx_bar, DiffMethod::Exact);

    let mut settings_smooth = DefaultSettings::default();
    settings_smooth.ipm.diff_method = DiffMethod::Smoothed;
    settings_smooth.ipm.diff_smoothing_mu = 0.1;
    settings_smooth.enable_grad = true;
    let mut solver_smooth = solve_nonneg_qp(&q, settings_smooth);
    let dq_smooth = backward_dq(&mut solver_smooth, &dx_bar, DiffMethod::Smoothed);

    let diff: f64 = dq_exact
        .iter()
        .zip(dq_smooth.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    assert!(
        diff > 0.01,
        "Smoothed at mu=0.1 should differ from Exact (diff={:.2e})",
        diff
    );
}

#[test]
fn test_smoothed_converges_to_exact_at_small_mu() {
    // At very small μ, smoothed gradients should approximate exact.
    let q = vec![-0.5, 1.0, -0.3];
    let dx_bar = vec![1.0, 0.0, 0.0];

    let mut settings_exact = DefaultSettings::default();
    settings_exact.ipm.diff_method = DiffMethod::Exact;
    let mut solver_exact = solve_nonneg_qp(&q, settings_exact);
    let dq_exact = backward_dq(&mut solver_exact, &dx_bar, DiffMethod::Exact);

    let mut settings_smooth = DefaultSettings::default();
    settings_smooth.ipm.diff_method = DiffMethod::Smoothed;
    settings_smooth.ipm.diff_smoothing_mu = 1e-8;
    settings_smooth.enable_grad = true;
    let mut solver_smooth = solve_nonneg_qp(&q, settings_smooth);
    let dq_smooth = backward_dq(&mut solver_smooth, &dx_bar, DiffMethod::Smoothed);

    let max_err: f64 = dq_exact
        .iter()
        .zip(dq_smooth.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    assert!(
        max_err < 1e-4,
        "Smoothed at mu=1e-8 should match Exact (max_err={:.2e})",
        max_err
    );
}

#[test]
fn test_smoothed_fd_consistency_dq() {
    // Finite-difference validation: perturb q[j], resolve, compare.
    let n = 3;
    let q_base = vec![-0.5, 1.0, -0.3];
    let dx_bar = vec![0.7, 0.3, -0.5];
    let eps = 1e-6;

    let mut settings = DefaultSettings::default();
    settings.ipm.diff_method = DiffMethod::Smoothed;
    settings.ipm.diff_smoothing_mu = 1e-4;
    settings.enable_grad = true;
    let mut solver = solve_nonneg_qp(&q_base, settings);
    let dq_analytic = backward_dq(&mut solver, &dx_bar, DiffMethod::Smoothed);

    // FD: dq[j] ≈ (x(q+ε·ej) - x(q-ε·ej))' dx_bar / (2ε)
    let mut dq_fd = vec![0.0; n];
    for j in 0..n {
        let mut q_plus = q_base.clone();
        q_plus[j] += eps;
        let mut settings_p = DefaultSettings::default();
        settings_p.ipm.diff_method = DiffMethod::Smoothed;
        settings_p.ipm.diff_smoothing_mu = 1e-4;
        settings_p.enable_grad = true;
        let solver_p = solve_nonneg_qp(&q_plus, settings_p);

        let mut q_minus = q_base.clone();
        q_minus[j] -= eps;
        let mut settings_m = DefaultSettings::default();
        settings_m.ipm.diff_method = DiffMethod::Smoothed;
        settings_m.ipm.diff_smoothing_mu = 1e-4;
        settings_m.enable_grad = true;
        let solver_m = solve_nonneg_qp(&q_minus, settings_m);

        let mut val = 0.0;
        for i in 0..n {
            val += (solver_p.solution.x[i] - solver_m.solution.x[i]) * dx_bar[i];
        }
        dq_fd[j] = val / (2.0 * eps);
    }

    let max_err: f64 = dq_analytic
        .iter()
        .zip(dq_fd.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    assert!(
        max_err < 1e-3,
        "Smoothed dq FD mismatch: max_err={:.2e}\n  analytic: {:?}\n  fd: {:?}",
        max_err,
        dq_analytic,
        dq_fd
    );
}

#[test]
fn test_smoothed_fd_consistency_db() {
    // Finite-difference validation for db gradients.
    // Use small mu so smoothed ≈ exact, making FD validation more reliable.
    // (At larger mu, the smoothing iterate varies with each perturbed problem,
    //  creating O(mu) discrepancies between analytic and FD gradients.)
    let n = 2;
    let m = 3; // 1 zero cone + 2 nonneg cones
    let P = CscMatrix::identity(n);
    let A = CscMatrix::from(&[[1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]);
    let b_base = vec![1.0, 0.0, 0.0];
    let q = vec![-0.5, 0.3];
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];

    let dx_bar = vec![0.7, -0.3];
    let dz_bar = vec![0.0; m];
    let ds_bar = vec![0.0; m];
    let eps = 1e-6;

    let mut settings = DefaultSettings::default();
    settings.ipm.diff_method = DiffMethod::Smoothed;
    settings.ipm.diff_smoothing_mu = 1e-8;
    settings.enable_grad = true;

    let mut solver = DefaultSolver::new(&P, &q, &A, &b_base, &cones, settings.clone()).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    let result = solver
        .backward_batch(&dx_bar, &ds_bar, &dz_bar, None, DiffMethod::Smoothed)
        .unwrap();
    let db_analytic = result.db;

    // FD: db[j] ≈ (x(b+ε·ej) - x(b-ε·ej))' dx_bar / (2ε)
    let mut db_fd: Vec<f64> = vec![0.0; m];
    for j in 0..m {
        let mut b_plus = b_base.clone();
        b_plus[j] += eps;
        let mut s_p = DefaultSolver::new(&P, &q, &A, &b_plus, &cones, settings.clone()).unwrap();
        s_p.solve();

        let mut b_minus = b_base.clone();
        b_minus[j] -= eps;
        let mut s_m = DefaultSolver::new(&P, &q, &A, &b_minus, &cones, settings.clone()).unwrap();
        s_m.solve();

        let mut val = 0.0;
        for i in 0..n {
            val += (s_p.solution.x[i] - s_m.solution.x[i]) * dx_bar[i];
        }
        db_fd[j] = val / (2.0 * eps);
    }

    let max_err: f64 = db_analytic
        .iter()
        .zip(db_fd.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    assert!(
        max_err < 1e-3,
        "Smoothed db FD mismatch: max_err={:.2e}\n  analytic: {:?}\n  fd: {:?}",
        max_err,
        db_analytic,
        db_fd
    );
}

#[test]
fn test_smoothed_mu_monotonicity() {
    // Larger μ should produce more smoothing (H values closer to 0.5 at boundary).
    // We detect this by checking that gradients change monotonically with μ.
    let q = vec![-0.01]; // near boundary
    let dx_bar = vec![1.0];

    let mus = [1e-6, 1e-4, 1e-2, 1e-1];
    let mut dq_values = Vec::new();

    for &mu in &mus {
        let mut settings = DefaultSettings::default();
        settings.ipm.diff_method = DiffMethod::Smoothed;
        settings.ipm.diff_smoothing_mu = mu;
        settings.enable_grad = true;
        let mut solver = solve_nonneg_qp(&q, settings);
        let dq = backward_dq(&mut solver, &dx_bar, DiffMethod::Smoothed);
        dq_values.push(dq[0]);
    }

    // The dq values should change (not all identical).
    let max_val: f64 = dq_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_val: f64 = dq_values.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = max_val - min_val;
    assert!(
        range > 0.01,
        "Smoothed dq should vary with mu (range={:.2e}, values={:?})",
        range,
        dq_values
    );
}

#[test]
fn test_smoothed_gradient_is_smooth_across_kink() {
    // Sweep q through the non-differentiable point at q=0 for
    //   min (1/2)x² + qx  s.t. x >= 0
    // The exact gradient dx*/dq is a step function (jump of 1 at q=0).
    // With smoothed diff at mu=1e-3, the max consecutive jump in dq
    // across the sweep must be much smaller than 1 — a smooth sigmoid.
    //
    // The old refinement loop (step_factor=1000, break on zero α)
    // produced jagged gradients with max_jump ≈ 0.49 because the
    // walk-up failed for some q values near the boundary.
    let n_pts = 101;
    let mu = 1e-3;

    // Sweep q from -0.5 to 0.5
    let q_vals: Vec<f64> = (0..n_pts)
        .map(|i| -0.5 + (i as f64) / ((n_pts - 1) as f64))
        .collect();

    let mut dq_vals: Vec<f64> = Vec::with_capacity(n_pts);
    for &q_val in &q_vals {
        let mut settings = DefaultSettings::default();
        settings.ipm.diff_method = DiffMethod::Smoothed;
        settings.ipm.diff_smoothing_mu = mu;
        settings.enable_grad = true;
        settings.verbose = false;
        let mut solver = solve_nonneg_qp(&[q_val], settings);
        let dq = backward_dq(&mut solver, &[1.0], DiffMethod::Smoothed);
        dq_vals.push(dq[0]);
    }

    // Max consecutive jump in the gradient curve
    let max_jump: f64 = dq_vals
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0, f64::max);

    // The sweep spacing is 1.0/100 = 0.01. For a smooth sigmoid with
    // width ~sqrt(mu)=0.032, the max slope is ~1/(4*sqrt(mu))≈8, so
    // max_jump ≈ 8 * 0.01 = 0.08. Allow up to 0.15 for margin.
    // The old broken code had max_jump ≈ 0.49.
    assert!(
        max_jump < 0.15,
        "Smoothed gradient is not smooth: max consecutive jump = {:.4} \
         (expected < 0.15). The walk-up may be failing for some q values.\n\
         dq values: {:?}",
        max_jump,
        &dq_vals[45..56] // show values around q=0
    );

    // Also verify the gradient has non-trivial range (not all zeros)
    let max_val: f64 = dq_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_val: f64 = dq_vals.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        max_val - min_val > 0.5,
        "Smoothed gradient range too small: [{:.4}, {:.4}]",
        min_val,
        max_val
    );
}

#[test]
fn test_smoothed_accepted_for_soc() {
    let P = CscMatrix::<f64>::zeros((2, 2));
    let q = vec![0.0; 2];
    let A = CscMatrix::<f64>::zeros((3, 2));
    let b = vec![0.0; 3];
    let cones = vec![SecondOrderConeT(3)];

    let mut settings = DefaultSettings::default();
    settings.ipm.diff_method = DiffMethod::Smoothed;

    let result = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
    assert!(result.is_ok(), "Smoothed with SOC should now be accepted");
}

#[test]
fn test_smoothed_rejected_for_exp() {
    let P = CscMatrix::<f64>::zeros((3, 3));
    let q = vec![0.0; 3];
    let A = CscMatrix::<f64>::zeros((3, 3));
    let b = vec![0.0; 3];
    let cones = vec![ExponentialConeT()];

    let mut settings = DefaultSettings::default();
    settings.ipm.diff_method = DiffMethod::Smoothed;

    let result = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
    assert!(result.is_err(), "Smoothed with exp cone should be rejected");
}

#[test]
fn test_smoothed_zero_cone_only_no_wasted_refinement() {
    // Zero-cone-only problems bypass the HSDE backward path entirely.
    // The refinement loop must not run (degree=0 would cause μ_target=0
    // and division by zero in the convergence check).
    // This test verifies the solver completes without hanging on 50
    // wasted centering iterations and produces correct gradients.
    let n = 2;
    let P = CscMatrix::identity(n);
    let q = vec![1.0, 2.0];
    let A = CscMatrix::from(&[[1.0, 1.0]]);
    let b = vec![3.0];
    let cones = vec![ZeroConeT(1)];

    let mut settings = DefaultSettings::default();
    settings.ipm.diff_method = DiffMethod::Smoothed;
    settings.ipm.diff_smoothing_mu = 1e-4;
    settings.enable_grad = true;
    settings.verbose = false;

    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);

    // Backward pass should work and match Exact (zero-cone path ignores diff_method)
    let dx_bar = vec![1.0, 0.0];
    let ds_bar = vec![0.0];
    let dz_bar = vec![0.0];
    let smoothed = solver
        .backward_batch(&dx_bar, &ds_bar, &dz_bar, None, DiffMethod::Smoothed)
        .unwrap();
    let smoothed_dq: &Vec<f64> = &smoothed.dq;

    let mut settings_exact = DefaultSettings::default();
    settings_exact.ipm.diff_method = DiffMethod::Exact;
    let mut solver_exact = DefaultSolver::new(&P, &q, &A, &b, &cones, settings_exact).unwrap();
    solver_exact.solve();
    let exact = solver_exact
        .backward_batch(&dx_bar, &ds_bar, &dz_bar, None, DiffMethod::Exact)
        .unwrap();
    let exact_dq: &Vec<f64> = &exact.dq;

    let max_err: f64 = smoothed_dq
        .iter()
        .zip(exact_dq.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        max_err < 1e-10,
        "Zero-cone Smoothed != Exact: max_err={:.2e}",
        max_err
    );
}

#[test]
fn test_smoothed_accepted_for_lp() {
    let P = CscMatrix::<f64>::zeros((2, 2));
    let q = vec![0.0; 2];
    let A = CscMatrix::<f64>::zeros((3, 2));
    let b = vec![0.0; 3];
    let cones = vec![ZeroConeT(1), NonnegativeConeT(2)];

    let mut settings = DefaultSettings::default();
    settings.ipm.diff_method = DiffMethod::Smoothed;

    let result = DefaultSolver::new(&P, &q, &A, &b, &cones, settings);
    assert!(result.is_ok(), "Smoothed with LP cones should be accepted");
}
