#![allow(non_snake_case)]
//! Direct-x dual gradients: an upstream gradient on `z_x` must produce a
//! gradient on (P, q) that matches finite differences of `z_x_orig`.
//!
//! Without this plumbing, the backward path silently dropped any upstream
//! `dz_x` (the v1 scope ignored direct-x dual gradients). This test
//! verifies that `backward_batch_with_dz_x(dz_x = e_j)` gives the column j
//! of the Jacobian d(z_x_orig) / dq, computed via finite differences.

use moreau::{algebra::*, solver::*};

const FD_EPS: f64 = 1e-6;
const TOL: f64 = 1e-3;

fn solve_and_backward(
    P: &CscMatrix<f64>,
    q: &[f64],
    n: usize,
    j: usize,
    dir_cones: &[SupportedXConeT],
) -> (Vec<f64>, Vec<f64>) {
    let A = CscMatrix::<f64>::zeros((0, n));
    let b: Vec<f64> = vec![];
    let cones: Vec<SupportedConeT<f64>> = vec![];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;

    let mut solver =
        DefaultSolver::new_with_xcones(P, q, &A, &b, &cones, dir_cones, settings).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let z_x_orig = solver.variables.z_x.clone();
    let mut dz_x = vec![0.0; z_x_orig.len()];
    dz_x[j] = 1.0;
    let dx = vec![0.0; n];
    let result = solver
        .backward_batch_with_dz_x(&dx, &[], &[], &dz_x, None, DiffMethod::Exact)
        .unwrap();
    (z_x_orig, result.dq)
}

fn fd_dq_for_z_x_j<F: Fn(&[f64]) -> Vec<f64>>(q: &[f64], j: usize, solve_z_x: F) -> Vec<f64> {
    let n_q = q.len();
    let mut fd = vec![0.0; n_q];
    for k in 0..n_q {
        let mut q_plus = q.to_vec();
        q_plus[k] += FD_EPS;
        let z_plus = solve_z_x(&q_plus);

        let mut q_minus = q.to_vec();
        q_minus[k] -= FD_EPS;
        let z_minus = solve_z_x(&q_minus);

        fd[k] = (z_plus[j] - z_minus[j]) / (2.0 * FD_EPS);
    }
    fd
}

#[test]
fn test_dz_x_dq_matches_finite_difference_nonneg() {
    // Active-boundary nonneg: q[2] = 1.0 binds x[2] at zero, so z_x[2] is
    // strictly positive (an active dual). This makes the sensitivity
    // d(z_x_orig) / dq non-trivial along the boundary direction.
    let n = 3usize;
    let P = CscMatrix::<f64>::new(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0, 1.0, 1.0],
    );
    let q = vec![-0.5, -0.5, 1.0];
    let dir_cones = vec![SupportedXConeT::NonnegativeXConeT((0..n).collect())];
    let j = 2usize;

    let (_z_x_base, analytic_dq) = solve_and_backward(&P, &q, n, j, &dir_cones);
    let fd_dq = fd_dq_for_z_x_j(&q, j, |q_perturbed| {
        let (z_x, _) = solve_and_backward(&P, q_perturbed, n, 0, &dir_cones);
        z_x
    });

    for k in 0..q.len() {
        assert!(
            (analytic_dq[k] - fd_dq[k]).abs() < TOL,
            "dq[{}] analytic={:.6} fd={:.6}",
            k,
            analytic_dq[k],
            fd_dq[k],
        );
    }
}

#[test]
fn test_dz_x_dq_matches_finite_difference_psd() {
    // PSD direct-x at boundary: 2x2 PSD svec (n=3). q chosen so the optimum
    // hits the cone boundary, so z_x has a non-trivial active-boundary
    // structure. Exercises the dense H_x branch via PSD eigendecomp.
    let n = 3usize;
    let P = CscMatrix::<f64>::new(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0, 1.0, 1.0],
    );
    let q = vec![-0.5, 1.0, -0.5];
    let dir_cones = vec![SupportedXConeT::PSDTriangleXConeT((0..n).collect(), 2)];
    let j = 1usize;

    let (_z_x_base, analytic_dq) = solve_and_backward(&P, &q, n, j, &dir_cones);
    let fd_dq = fd_dq_for_z_x_j(&q, j, |q_perturbed| {
        let (z_x, _) = solve_and_backward(&P, q_perturbed, n, 0, &dir_cones);
        z_x
    });

    for k in 0..q.len() {
        assert!(
            (analytic_dq[k] - fd_dq[k]).abs() < TOL,
            "dq[{}] analytic={:.6} fd={:.6}",
            k,
            analytic_dq[k],
            fd_dq[k],
        );
    }
}

#[test]
fn test_dz_x_dq_matches_finite_difference_soc() {
    // SOC active-boundary: q = (1, 2, 0) makes the SOC constraint bind, so
    // z_x is non-zero across the cone (dense H_x at the boundary).
    let n = 3usize;
    let P = CscMatrix::<f64>::new(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0, 1.0, 1.0],
    );
    let q = vec![1.0, 2.0, 0.0];
    let dir_cones = vec![SupportedXConeT::SecondOrderXConeT((0..n).collect())];
    let j = 0usize; // dz_x on the cone-axis dual.

    let (_z_x_base, analytic_dq) = solve_and_backward(&P, &q, n, j, &dir_cones);
    let fd_dq = fd_dq_for_z_x_j(&q, j, |q_perturbed| {
        let (z_x, _) = solve_and_backward(&P, q_perturbed, n, 0, &dir_cones);
        z_x
    });

    for k in 0..q.len() {
        assert!(
            (analytic_dq[k] - fd_dq[k]).abs() < TOL,
            "dq[{}] analytic={:.6} fd={:.6}",
            k,
            analytic_dq[k],
            fd_dq[k],
        );
    }
}

/// Direct-x backward in `DiffMethod::Smoothed` mode: the same FD gradient
/// check on `d(z_x_orig)/dq[j]`, except the analytic backward routes through
/// the central-path Jacobian (`get_central_path_derivative_sparse_xcones`).
/// Regression: the prior code at kkt module silently fell back to Exact
/// for the direct-x portion regardless of `diff_method`.
#[test]
fn test_dz_x_dq_matches_finite_difference_nonneg_smoothed() {
    let n = 3usize;
    let P = CscMatrix::<f64>::new(
        n,
        n,
        (0..=n).collect(),
        (0..n).collect(),
        vec![1.0, 1.0, 1.0],
    );
    let q = vec![-0.5, -0.5, 1.0];
    let dir_cones = vec![SupportedXConeT::NonnegativeXConeT((0..n).collect())];
    let j = 2usize;

    let solve_smoothed = |q_in: &[f64]| -> (Vec<f64>, Vec<f64>) {
        let A = CscMatrix::<f64>::zeros((0, n));
        let b: Vec<f64> = vec![];
        let cones: Vec<SupportedConeT<f64>> = vec![];
        let mut settings = DefaultSettings::default();
        settings.ipm.presolve_enable = false;
        settings.ipm.equilibrate_enable = false;
        settings.ipm.diff_method = DiffMethod::Smoothed;
        settings.ipm.diff_smoothing_mu = 1e-8;
        settings.enable_grad = true;
        settings.verbose = false;
        let mut solver =
            DefaultSolver::new_with_xcones(&P, q_in, &A, &b, &cones, &dir_cones, settings).unwrap();
        solver.solve();
        let z_x_orig = solver.variables.z_x.clone();
        let mut dz_x = vec![0.0; z_x_orig.len()];
        dz_x[j] = 1.0;
        let dx = vec![0.0; n];
        let result = solver
            .backward_batch_with_dz_x(&dx, &[], &[], &dz_x, None, DiffMethod::Smoothed)
            .unwrap();
        (z_x_orig, result.dq)
    };

    let (_z_x_base, analytic_dq) = solve_smoothed(&q);
    let fd_dq = fd_dq_for_z_x_j(&q, j, |q_perturbed| {
        let (z_x, _) = solve_smoothed(q_perturbed);
        z_x
    });

    // Smoothed mode adds an O(μ) bias; the FD check tolerates μ-level error.
    let tol = 5e-3;
    for k in 0..q.len() {
        assert!(
            (analytic_dq[k] - fd_dq[k]).abs() < tol,
            "dq[{}] analytic={:.6} fd={:.6} (smoothed)",
            k,
            analytic_dq[k],
            fd_dq[k],
        );
    }
}
