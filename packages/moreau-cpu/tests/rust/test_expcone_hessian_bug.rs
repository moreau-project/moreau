//! Test for division-by-zero bug in exponential cone Hessian computation.
//!
//! The bug is in update_dual_grad_H where r = z[1] - z[0]*(1 + log(-z[2]/z[0]))
//! can be zero, causing c2 = r.recip() to produce infinity.
//!
//! This test exercises the cone with various problem configurations to check
//! for NaN/Inf in the solution.

#![allow(non_snake_case)]

use moreau::{algebra::*, solver::*};

/// Compute r value for given z to check if we're near the singularity.
/// For the dual exp cone: z[0] < 0, z[2] > 0
/// r = z[1] - z[0] - z[0]*log(-z[2]/z[0])
/// Note: -z[2]/z[0] = -z[2]/z[0] where z[0] < 0 and z[2] > 0, so -z[2]/z[0] > 0
fn compute_r(z: &[f64]) -> f64 {
    // Dual cone requires z[0] < 0, z[2] > 0
    if z[0] >= 0.0 || z[2] <= 0.0 {
        return f64::NAN;
    }
    let arg = -z[2] / z[0]; // This is positive when z[0] < 0 and z[2] > 0
    if arg <= 0.0 {
        return f64::NAN;
    }
    let l = arg.ln();
    z[1] - z[0] - z[0] * l
}

/// Create exp cone problem: max x s.t. y*exp(x/y) <= z, y == y_val, z == z_val
fn create_expcone_problem(
    y_val: f64,
    z_val: f64,
) -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let P = CscMatrix::<f64>::zeros((3, 3));
    let c = vec![-1., 0., 0.]; // maximize x

    // Exp cone constraint: -I * x + s = 0, s in K_exp
    let mut A1 = CscMatrix::<f64>::identity(3);
    A1.negate();
    let b1 = vec![0.; 3];

    // Equality constraints: y = y_val, z = z_val
    let A2 = CscMatrix::new(
        2,                // m
        3,                // n
        vec![0, 0, 1, 2], // colptr
        vec![0, 1],       // rowval
        vec![1., 1.],     // nzval
    );
    let b2 = vec![y_val, z_val];

    let A = CscMatrix::vcat(&A1, &A2).unwrap();
    let b = [b1, b2].concat();

    let cones = vec![ExponentialConeT(), ZeroConeT(2)];

    (P, c, A, b, cones)
}

#[test]
fn test_expcone_r_computation() {
    // Test the r computation helper function
    // For r = 0: z[1] = z[0] * (1 + log(-z[2]/z[0]))
    // With z[0] = -1, z[2] = -e: z[1] = -1 * (1 + 1) = -2

    let test_cases = [
        // (z[0], z[1], z[2], expected_r)
        // Dual cone: z[0] < 0, z[2] > 0
        // r = z[1] - z[0] - z[0] * ln(-z[2]/z[0])
        // With z[0]=-1, z[2]=e: -z[2]/z[0] = e, ln(e)=1
        (-1.0, -2.0, std::f64::consts::E, 0.0),      // l=1, r = -2 - (-1) - (-1)*1 = -2+1+1 = 0
        (-1.0, -1.5, std::f64::consts::E, 0.5),      // l=1, r = -1.5 + 1 + 1 = 0.5
        (-1.0, -2.5, std::f64::consts::E, -0.5),     // l=1, r = -2.5 + 1 + 1 = -0.5
        (-2.0, -4.0, 2.0 * std::f64::consts::E, 0.0), // l=1, r = -4 + 2 + 2 = 0
    ];

    for (z0, z1, z2, expected_r) in test_cases {
        let r = compute_r(&[z0, z1, z2]);
        println!(
            "z = ({:.6}, {:.6}, {:.6}), r = {:.10}, expected = {:.6}",
            z0, z1, z2, r, expected_r
        );
        assert!(
            (r - expected_r).abs() < 1e-10,
            "Expected r ≈ {} but got r = {}",
            expected_r,
            r
        );
    }
}

#[test]
fn test_expcone_boundary_approach() {
    // Sequence of problems that approach the cone boundary
    // max x s.t. y * exp(x/y) <= z, y = 1, z = e^t
    // Solution: x = t (on boundary)

    for &t in &[1.0_f64, 5.0, 10.0, 20.0, 50.0] {
        println!("\n=== t = {} ===", t);

        let (P, c, A, b, cones) = create_expcone_problem(1.0, t.exp());

        let settings = DefaultSettings::default();
        let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
        solver.solve();

        println!(
            "Status: {:?}, iterations: {}, obj: {}",
            solver.solution.status, solver.info.iterations, solver.info.cost_primal
        );

        // Check for NaN/Inf
        let has_nan = solver.solution.x.iter().any(|&v| v.is_nan())
            || solver.solution.z.iter().any(|&v| v.is_nan());
        let has_inf = solver.solution.x.iter().any(|&v| v.is_infinite())
            || solver.solution.z.iter().any(|&v| v.is_infinite());

        if has_nan || has_inf {
            println!("BUG: NaN/Inf detected!");
            println!("x = {:?}", solver.solution.x);
            println!("z = {:?}", solver.solution.z);
        }

        assert!(!has_nan, "NaN detected at t = {}", t);
        assert!(!has_inf, "Inf detected at t = {}", t);

        // Solution should give x ≈ t
        if solver.solution.status == SolverStatus::Solved {
            let x_opt = solver.solution.x[0];
            let tol = 1e-3 + 1e-6 * t; // looser tol for larger t
            assert!(
                (x_opt - t).abs() < tol,
                "Expected x ≈ {}, got {}",
                t,
                x_opt
            );
        }
    }
}

#[test]
fn test_expcone_tight_tolerances() {
    // Test with very tight tolerances that might expose numerical issues

    let (P, c, A, b, cones) = create_expcone_problem(1.0, 5.0_f64.exp());

    for &tol in &[1e-8, 1e-10, 1e-12] {
        println!("\n=== tol = {} ===", tol);

        let mut settings = DefaultSettings::default();
        settings.ipm.tol_gap_abs = tol;
        settings.ipm.tol_gap_rel = tol;
        settings.ipm.tol_feas = tol;
        settings.max_iter = 500;
        settings.verbose = false;

        let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
        solver.solve();

        println!(
            "Status: {:?}, iterations: {}",
            solver.solution.status, solver.info.iterations
        );

        // Check for NaN
        let has_nan = solver.solution.x.iter().any(|&v| v.is_nan());
        assert!(!has_nan, "NaN at tol = {}", tol);

        // Compute r for the dual exp cone part
        if solver.solution.z.len() >= 3 {
            let r = compute_r(&solver.solution.z[0..3]);
            println!("Dual r = {}", r);
        }
    }
}

#[test]
fn test_expcone_scaled_problems() {
    // Test problems with different scales

    for &scale in &[1e-6, 1e-3, 1.0, 1e3, 1e6] {
        println!("\n=== scale = {} ===", scale);

        let t = 5.0_f64;
        let (P, c, A, b, cones) = create_expcone_problem(scale, scale * t.exp());

        let settings = DefaultSettings::default();
        let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
        solver.solve();

        let has_nan = solver.solution.x.iter().any(|&v| v.is_nan())
            || solver.solution.z.iter().any(|&v| v.is_nan());

        println!(
            "Status: {:?}, iter: {}, NaN: {}",
            solver.solution.status, solver.info.iterations, has_nan
        );

        assert!(!has_nan, "NaN at scale = {}", scale);
    }
}

#[test]
fn test_expcone_dual_infeasible_no_nan() {
    // Dual infeasible problem should not produce NaN

    let P = CscMatrix::<f64>::zeros((3, 3));
    let c = vec![-1., 0., 0.];

    let mut A = CscMatrix::<f64>::identity(3);
    A.negate();
    let b = vec![0.; 3];
    let cones = vec![ExponentialConeT()];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    println!("Status: {:?}", solver.solution.status);
    assert_eq!(solver.solution.status, SolverStatus::DualInfeasible);

    // Even for infeasibility certificates, check no NaN in the ray
    // (Inf is expected in rays)
    let has_nan = solver.solution.x.iter().any(|&v| v.is_nan());
    assert!(!has_nan, "NaN in dual infeasible certificate");
}

#[test]
fn test_expcone_primal_infeasible_no_nan() {
    // Primal infeasible problem (z < 0) should not produce NaN

    let (P, c, A, mut b, cones) = create_expcone_problem(1.0, 5.0_f64.exp());
    b[4] = -1.0; // z = -1 is infeasible

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    println!("Status: {:?}", solver.solution.status);
    assert_eq!(solver.solution.status, SolverStatus::PrimalInfeasible);

    // Check no NaN in infeasibility certificate
    let has_nan = solver.solution.z.iter().any(|&v| v.is_nan());
    assert!(!has_nan, "NaN in primal infeasible certificate");
}
