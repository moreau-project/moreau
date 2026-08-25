//! Integration and numerical-gradient tests for the diff module.

use super::super::settings::DiffMethod;
use super::api::is_all_zero_cones;
use super::*;
use crate::algebra::{CscMatrix, ShapedMatrix};
use crate::solver::core::cones::SupportedConeT;

// Default diff params for existing tests (projection method, mu unused)
const TEST_DIFF_METHOD: DiffMethod = DiffMethod::Exact;
const TEST_MU: f64 = 0.0;

/// Materialize a GenPowerSparse (or Dense) block into a dense dim×dim matrix for testing.
fn genpow_sparse_to_dense(block: &ConeDerivativeBlock<f64>, dim: usize) -> Vec<f64> {
    match block {
        ConeDerivativeBlock::Dense { dim: d, data } => {
            assert_eq!(*d, dim);
            data.clone()
        }
        ConeDerivativeBlock::GenPowerSparse {
            dim: d,
            diag,
            left1,
            right1,
            left2,
            right2,
            left3,
            c3,
        } => {
            assert_eq!(*d, dim);
            let mut jac = vec![0.0_f64; dim * dim];
            for i in 0..dim {
                for j in 0..dim {
                    jac[i * dim + j] = if i == j { diag[i] } else { 0.0 }
                        + left1[i] * right1[j]
                        + left2[i] * right2[j]
                        + c3 * left3[i] * left3[j];
                }
            }
            jac
        }
        _ => panic!("Expected Dense or GenPowerSparse block for GenPowerCone"),
    }
}

// ========================================================================
// Integration tests for differentiate and differentiate_adjoint
// ========================================================================

/// Helper to create identity matrix in CSC format
fn identity_csc(n: usize) -> CscMatrix<f64> {
    let mut colptr = vec![0usize; n + 1];
    let mut rowval = Vec::with_capacity(n);
    let mut nzval = Vec::with_capacity(n);

    for i in 0..n {
        colptr[i + 1] = i + 1;
        rowval.push(i);
        nzval.push(1.0);
    }

    CscMatrix {
        m: n,
        n: n,
        colptr,
        rowval,
        nzval,
    }
}

/// Helper to create zero matrix in CSC format
fn zero_csc(m: usize, n: usize) -> CscMatrix<f64> {
    CscMatrix {
        m,
        n,
        colptr: vec![0; n + 1],
        rowval: vec![],
        nzval: vec![],
    }
}

// ========================================================================
// QP with equality constraints (ZeroCone)
// ========================================================================

#[test]
fn test_integration_qp_equality_forward() {
    // min (1/2)||x||^2 s.t. sum(x) = 2
    // Solution: x = [1, 1]

    let P = identity_csc(2);
    let q = vec![0.0, 0.0];
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b = vec![2.0];
    let cones = vec![SupportedConeT::ZeroConeT(1)];

    let x = vec![1.0, 1.0];
    let s = vec![0.0];
    let z = vec![0.0];
    let tau = 1.0;

    // Perturb q
    let dP = zero_csc(2, 2);
    let dq = vec![1.0, 0.0];
    let dA = zero_csc(1, 2);
    let db = vec![0.0];

    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // dx should satisfy: dx[0] + dx[1] = 0 (constraint preserved)
    assert!((result.dx[0] + result.dx[1]).abs() < 1e-6);
    // ds should be 0 for equality constraints
    assert!(result.ds[0].abs() < 1e-10);
}

#[test]
fn test_integration_qp_equality_backward() {
    // Same setup
    let P = identity_csc(2);
    let q = vec![0.0, 0.0];
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b = vec![2.0];
    let cones = vec![SupportedConeT::ZeroConeT(1)];

    let x = vec![1.0, 1.0];
    let s = vec![0.0];
    let z = vec![0.0];
    let tau = 1.0;

    // Gradient w.r.t. x[0]
    let dx_bar = vec![1.0, 0.0];
    let dz_bar = vec![0.0];
    let ds_bar = vec![0.0];

    let result = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // Check gradients have correct dimensions
    assert_eq!(result.dq.len(), 2);
    assert_eq!(result.db.len(), 1);
    assert_eq!(result.dP.nrows(), 2);
    assert_eq!(result.dA.nrows(), 1);
}

#[test]
fn test_integration_qp_forward_adjoint_consistency() {
    // Test that <forward(d), bar> = <d, adjoint(bar)>

    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![2.0, 3.0],
    };
    let q = vec![1.0, 1.0];
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b = vec![2.0];
    let cones = vec![SupportedConeT::ZeroConeT(1)];

    let x = vec![1.0, 1.0];
    let s = vec![0.0];
    let z = vec![-2.5];
    let tau = 1.0;

    // Forward perturbations
    let dP = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![0.1, 0.2],
    };
    let dq = vec![0.3, 0.4];
    let dA = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![0.5, 0.6],
    };
    let db = vec![0.7];

    let forward = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // Adjoint directions
    let dx_bar = vec![0.8, 0.9];
    let dz_bar = vec![1.0];
    let ds_bar = vec![0.0];

    let adjoint = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // LHS: <forward, bar>
    let lhs: f64 =
        forward.dx[0] * dx_bar[0] + forward.dx[1] * dx_bar[1] + forward.dz[0] * dz_bar[0];

    // RHS: <d, adjoint>
    let mut rhs: f64 = 0.0;
    for k in 0..dP.nnz() {
        rhs += dP.nzval[k] * adjoint.dP.nzval[k];
    }
    for i in 0..dq.len() {
        rhs += dq[i] * adjoint.dq[i];
    }
    for k in 0..dA.nnz() {
        rhs += dA.nzval[k] * adjoint.dA.nzval[k];
    }
    for i in 0..db.len() {
        rhs += db[i] * adjoint.db[i];
    }

    assert!(
        (lhs - rhs).abs() < 1e-5,
        "Forward-adjoint consistency: LHS = {}, RHS = {}",
        lhs,
        rhs
    );
}

// ========================================================================
// LP with nonnegative cone
// ========================================================================

#[test]
fn test_integration_lp_nonneg_forward() {
    // min q'x s.t. Ax + s = b, s >= 0
    let P = zero_csc(2, 2);
    let q = vec![1.0, 1.0];
    let A = identity_csc(2);
    let b = vec![1.0, 1.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(2)];

    // Solution at bounds: x = [0, 0], s = [1, 1], z = [1, 1]
    let x = vec![0.0, 0.0];
    let s = vec![1.0, 1.0];
    let z = vec![1.0, 1.0];
    let tau = 1.0;

    let dP = zero_csc(2, 2);
    let dq = vec![0.1, 0.0];
    let dA = zero_csc(2, 2);
    let db = vec![0.0, 0.0];

    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dx.len(), 2);
    assert_eq!(result.dz.len(), 2);
    assert_eq!(result.ds.len(), 2);
}

#[test]
fn test_integration_lp_nonneg_backward() {
    let P = zero_csc(2, 2);
    let q = vec![1.0, 1.0];
    let A = identity_csc(2);
    let b = vec![1.0, 1.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(2)];

    let x = vec![0.0, 0.0];
    let s = vec![1.0, 1.0];
    let z = vec![1.0, 1.0];
    let tau = 1.0;

    let dx_bar = vec![1.0, 0.0];
    let dz_bar = vec![0.0, 0.0];
    let ds_bar = vec![0.0, 0.0];

    let result = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dq.len(), 2);
    assert_eq!(result.db.len(), 2);
}

// ========================================================================
// Mixed cones
// ========================================================================

#[test]
fn test_integration_mixed_cones_forward() {
    // Zero cone + Nonnegative cone
    let P = identity_csc(2);
    let q = vec![0.0, 0.0];
    let A = CscMatrix {
        m: 3,
        n: 2,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 2],
        nzval: vec![1.0, 1.0, 1.0, 1.0],
    };
    let b = vec![1.0, 1.0, 1.0];
    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
    ];

    let x = vec![0.5, 0.5];
    let s = vec![0.0, 0.5, 0.5];
    let z = vec![1.0, 0.0, 0.0];
    let tau = 1.0;

    let dP = zero_csc(2, 2);
    let dq = vec![0.1, 0.0];
    let dA = zero_csc(3, 2);
    let db = vec![0.0, 0.0, 0.0];

    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dx.len(), 2);
    assert_eq!(result.dz.len(), 3);
    assert_eq!(result.ds.len(), 3);
}

// ========================================================================
// SOC cone
// ========================================================================

#[test]
fn test_integration_soc_forward() {
    // SOC constraint: s[0] >= ||s[1:]||
    let n = 2;
    let m = 3;

    let P = identity_csc(n);
    let q = vec![0.0, 0.0];
    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 2],
        nzval: vec![1.0, 1.0, 0.0, 1.0],
    };
    let b = vec![2.0, 0.5, 0.5];
    let cones = vec![SupportedConeT::SecondOrderConeT(3)];

    // Solution inside SOC
    let x = vec![1.0, 0.5];
    let s = vec![1.0, 0.5, 0.0]; // t=1, ||x||=0.5 < 1
    let z = vec![0.5, 0.25, 0.0];
    let tau = 1.0;

    let dP = zero_csc(n, n);
    let dq = vec![0.0, 0.1];
    let dA = zero_csc(m, n);
    let db = vec![0.0, 0.0, 0.0];

    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dx.len(), n);
    assert_eq!(result.dz.len(), m);
    assert_eq!(result.ds.len(), m);
}

#[test]
fn test_integration_soc_backward() {
    let n = 2;
    let m = 3;

    let P = identity_csc(n);
    let q = vec![0.0, 0.0];
    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 2],
        nzval: vec![1.0, 1.0, 0.0, 1.0],
    };
    let b = vec![2.0, 0.5, 0.5];
    let cones = vec![SupportedConeT::SecondOrderConeT(3)];

    let x = vec![1.0, 0.5];
    let s = vec![1.0, 0.5, 0.0];
    let z = vec![0.5, 0.25, 0.0];
    let tau = 1.0;

    let dx_bar = vec![1.0, 0.0];
    let dz_bar = vec![0.0, 0.0, 0.0];
    let ds_bar = vec![0.0, 0.0, 0.0];

    let result = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dq.len(), n);
    assert_eq!(result.db.len(), m);
}

// ========================================================================
// is_all_zero_cones tests
// ========================================================================

#[test]
fn test_is_all_zero_cones_true() {
    let cones: Vec<SupportedConeT<f64>> =
        vec![SupportedConeT::ZeroConeT(2), SupportedConeT::ZeroConeT(3)];
    assert!(is_all_zero_cones(&cones));
}

#[test]
fn test_is_all_zero_cones_false() {
    let cones: Vec<SupportedConeT<f64>> = vec![
        SupportedConeT::ZeroConeT(2),
        SupportedConeT::NonnegativeConeT(3),
    ];
    assert!(!is_all_zero_cones(&cones));
}

#[test]
fn test_is_all_zero_cones_empty() {
    let cones: Vec<SupportedConeT<f64>> = vec![];
    assert!(is_all_zero_cones(&cones));
}

// ========================================================================
// Numerical gradient verification tests
// These tests verify analytical derivatives match finite-difference approx
// ========================================================================

/// Numerical verification of forward derivative for QP equality case
#[test]
fn test_numerical_gradient_qp_dq() {
    // Verify d(x)/d(q) via finite differences
    // Problem: min (1/2)x'Px + q'x s.t. sum(x) = 2

    let P = identity_csc(2);
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b = vec![2.0];
    let cones = vec![SupportedConeT::ZeroConeT(1)];

    let q = vec![0.0, 0.0];
    let x = vec![1.0, 1.0];
    let s = vec![0.0];
    let z = vec![0.0];
    let tau = 1.0;

    // Analytical derivative: perturb q[0] by epsilon
    let dP = zero_csc(2, 2);
    let dq = vec![1.0, 0.0]; // direction of perturbation
    let dA = zero_csc(1, 2);
    let db = vec![0.0];

    let analytical = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // Finite difference verification
    // For this problem, we can solve analytically:
    // KKT: Px + A'z = -q, Ax = b
    // With P = I, A = [1,1], b = 2:
    // x + lambda*[1;1] = -q
    // x[0] + x[1] = 2
    //
    // From x = -q - lambda*[1;1]:
    // -q[0] - lambda + (-q[1] - lambda) = 2
    // -q[0] - q[1] - 2*lambda = 2
    // lambda = -(q[0] + q[1] + 2)/2
    //
    // x[0] = -q[0] + (q[0] + q[1] + 2)/2 = (q[1] - q[0] + 2)/2
    // x[1] = -q[1] + (q[0] + q[1] + 2)/2 = (q[0] - q[1] + 2)/2
    //
    // d(x[0])/d(q[0]) = -1/2
    // d(x[1])/d(q[0]) = 1/2

    let tol = 1e-5;
    assert!(
        (analytical.dx[0] - (-0.5)).abs() < tol,
        "dx[0] = {}, expected -0.5",
        analytical.dx[0]
    );
    assert!(
        (analytical.dx[1] - 0.5).abs() < tol,
        "dx[1] = {}, expected 0.5",
        analytical.dx[1]
    );
}

/// Numerical verification of forward derivative for db perturbation
#[test]
fn test_numerical_gradient_qp_db() {
    let P = identity_csc(2);
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b = vec![2.0];
    let cones = vec![SupportedConeT::ZeroConeT(1)];

    let q = vec![0.0, 0.0];
    let x = vec![1.0, 1.0];
    let s = vec![0.0];
    let z = vec![0.0];
    let tau = 1.0;

    // Perturb b by 1
    let dP = zero_csc(2, 2);
    let dq = vec![0.0, 0.0];
    let dA = zero_csc(1, 2);
    let db = vec![1.0];

    let analytical = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // From the solution formula:
    // x[0] = (q[1] - q[0] + b)/2, x[1] = (q[0] - q[1] + b)/2
    // d(x[0])/d(b) = 0.5
    // d(x[1])/d(b) = 0.5

    let tol = 1e-5;
    assert!(
        (analytical.dx[0] - 0.5).abs() < tol,
        "dx[0] = {}, expected 0.5",
        analytical.dx[0]
    );
    assert!(
        (analytical.dx[1] - 0.5).abs() < tol,
        "dx[1] = {}, expected 0.5",
        analytical.dx[1]
    );
}

/// Verify backward gradient using forward-backward identity
/// This is a stronger test: solve both forward and backward,
/// then check inner product equality
#[test]
fn test_numerical_gradient_backward_identity() {
    // Use a non-trivial problem with known solution
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![2.0, 3.0],
    };
    let q = vec![1.0, 1.0];
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b = vec![2.0];
    let cones = vec![SupportedConeT::ZeroConeT(1)];

    // At optimality: 2*x[0] + lambda = -1, 3*x[1] + lambda = -1, x[0]+x[1] = 2
    // Solving: x[0] = 7/5, x[1] = 3/5, lambda = -19/5
    let x = vec![7.0 / 5.0, 3.0 / 5.0];
    let s = vec![0.0];
    let z = vec![-19.0 / 5.0];
    let tau = 1.0;

    // Random perturbation direction
    let dP = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![0.1, 0.2],
    };
    let dq = vec![0.3, 0.4];
    let dA = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![0.5, 0.6],
    };
    let db = vec![0.7];

    // Forward pass
    let forward = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // Random output gradient direction
    let dx_bar = vec![0.11, 0.22];
    let dz_bar = vec![0.33];
    let ds_bar = vec![0.0];

    // Backward pass
    let backward = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // Identity: <forward, bar> = <d, backward>
    let lhs: f64 = forward
        .dx
        .iter()
        .zip(dx_bar.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + forward
            .dz
            .iter()
            .zip(dz_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    let rhs: f64 = dP
        .nzval
        .iter()
        .zip(backward.dP.nzval.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + dq.iter()
            .zip(backward.dq.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
        + dA.nzval
            .iter()
            .zip(backward.dA.nzval.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
        + db.iter()
            .zip(backward.db.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    let tol = 1e-5;
    assert!(
        (lhs - rhs).abs() < tol,
        "Forward-backward identity: LHS = {}, RHS = {}",
        lhs,
        rhs
    );
}

/// Verify gradient for diagonal P perturbation
#[test]
fn test_numerical_gradient_dP_diagonal() {
    // Perturb P[0,0] and verify derivative
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![2.0, 2.0],
    };
    let q = vec![0.0, 0.0];
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b = vec![2.0];
    let cones = vec![SupportedConeT::ZeroConeT(1)];

    // With P = 2I, q = 0, A = [1,1], b = 2:
    // Solution: x = [1, 1], lambda = 0
    let x = vec![1.0, 1.0];
    let s = vec![0.0];
    let z = vec![0.0];
    let tau = 1.0;

    // Perturb P[0,0]
    let dP = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 1], // Only P[0,0] is nonzero
        rowval: vec![0],
        nzval: vec![1.0],
    };
    let dq = vec![0.0, 0.0];
    let dA = zero_csc(1, 2);
    let db = vec![0.0];

    let analytical = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // Verify constraint still holds: A*dx = 0
    let constraint_violation = analytical.dx[0] + analytical.dx[1];
    assert!(
        constraint_violation.abs() < 1e-6,
        "Constraint violation: {}",
        constraint_violation
    );
}

/// Regression test: dP with off-diagonal entries must use sym_up().symv()
/// not gemv(). With upper-triangular P, gemv() misses lower-triangle
/// contributions, producing wrong dPx products and therefore wrong gradients.
#[test]
fn test_numerical_gradient_dP_off_diagonal() {
    // Problem: min (1/2)x'Px + q'x s.t. x[0] + x[1] = 3
    // P = [[2, 1], [1, 2]] stored as upper-triangular CSC
    // q = [1, 0] (asymmetric to produce asymmetric x*)
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 3],
        rowval: vec![0, 0, 1],
        nzval: vec![2.0, 1.0, 2.0],
    };
    let q = vec![1.0, 0.0];
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b = vec![3.0];
    let cones = vec![SupportedConeT::ZeroConeT(1)];

    // KKT: Px + A'z = -q, Ax = b
    // [2 1][x0] + lam*[1] = [-1]    x0+x1=3
    // [1 2][x1]       [1]   [ 0]
    // 2x0+x1+lam = -1, x0+2x1+lam = 0, x0+x1 = 3
    // Subtract: x0-x1 = -1 → x1 = x0+1
    // x0 + x0+1 = 3 → x0 = 1, x1 = 2
    // lam = -1 - 2*1 - 2 = -5
    let x = vec![1.0, 2.0];
    let s = vec![0.0];
    let z = vec![-5.0];
    let tau = 1.0;

    // Perturb P[0,1] (upper-triangular entry)
    let dP = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 0, 1],
        rowval: vec![0],
        nzval: vec![1.0],
    };
    let dq = vec![0.0, 0.0];
    let dA = zero_csc(1, 2);
    let db = vec![0.0];

    let analytical = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Exact,
        0.0,
    );

    // Numerical finite difference: perturb P[0,1] (and P[1,0]) by eps
    // P(e) = [[2, 1+e], [1+e, 2]], q = [1, 0], Ax = 3
    // KKT: 2x0+(1+e)x1+lam = -1, (1+e)x0+2x1+lam = 0, x0+x1 = 3
    // Subtract: (1-e)x0-(1-e)x1 = -1 → x0-x1 = -1/(1-e)
    // x0+x1=3, x0-x1=-1/(1-e) → x0 = (3 - 1/(1-e))/2, x1 = (3 + 1/(1-e))/2
    // At e=0: x0=1, x1=2. dx0/de = -1/(2*(1-e)^2)|_{e=0} = -0.5, dx1/de = 0.5
    let expected_dx = [-0.5, 0.5];

    let tol = 1e-5;
    assert!(
        (analytical.dx[0] - expected_dx[0]).abs() < tol,
        "dx[0] = {}, expected {}",
        analytical.dx[0],
        expected_dx[0]
    );
    assert!(
        (analytical.dx[1] - expected_dx[1]).abs() < tol,
        "dx[1] = {}, expected {}",
        analytical.dx[1],
        expected_dx[1]
    );
}

/// Regression test for HSDE path: non-diagonal P with off-diagonal dP perturbation.
/// Tests dP.sym_up().symv() in differentiate_hsde (inequality constraints force HSDE path).
#[test]
fn test_numerical_gradient_dP_off_diagonal_hsde() {
    // Same problem as above but with inequality constraints to force HSDE path.
    // min (1/2)x'Px + q'x s.t. x[0]+x[1]=3, x[0]>=0, x[1]>=0
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 3],
        rowval: vec![0, 0, 1],
        nzval: vec![2.0, 1.0, 2.0],
    };
    let q = vec![1.0, 0.0];
    let A = CscMatrix {
        m: 3,
        n: 2,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 2],
        nzval: vec![1.0, 1.0, 1.0, 1.0],
    };
    let b = vec![3.0, 0.0, 0.0];
    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
    ];

    // x = [1, 2] (inactive inequality constraints)
    let x = vec![1.0, 2.0];
    let s = vec![0.0, 1.0, 2.0];
    let z = vec![-5.0, 0.0, 0.0];
    let tau = 1.0;

    // Perturb P[0,1]
    let dP = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 0, 1],
        rowval: vec![0],
        nzval: vec![1.0],
    };
    let dq = vec![0.0, 0.0];
    let dA = zero_csc(3, 2);
    let db = vec![0.0, 0.0, 0.0];

    let analytical = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Exact,
        0.0,
    );

    // Same analytical result: dx/dP[0,1] = [-0.5, 0.5]
    let tol = 1e-4;
    assert!(
        (analytical.dx[0] - (-0.5)).abs() < tol,
        "HSDE dx[0] = {}, expected -0.5",
        analytical.dx[0]
    );
    assert!(
        (analytical.dx[1] - 0.5).abs() < tol,
        "HSDE dx[1] = {}, expected 0.5",
        analytical.dx[1]
    );
}

// ========================================================================
// Exponential cone integration tests
// ========================================================================

#[test]
fn test_integration_exp_cone_forward() {
    // Problem with exponential cone constraint
    // min q'x s.t. Ax + s = b, s ∈ K_exp
    // Exponential cone: (r, s, t) where s*exp(r/s) <= t, s > 0
    let n = 3;
    let m = 3;

    let P = zero_csc(n, n);
    let q = vec![1.0, 1.0, 1.0];
    let A = identity_csc(n); // Ax + s = b => x + s = b
    let b = vec![0.0, 1.0, 3.0]; // Solution: s = (0, 1, 3) which is inside exp cone

    // Solution at a point inside the exp cone
    // s = [0, 1, 3] means 1*exp(0/1) = 1 <= 3, so it's in the cone interior
    let x = vec![0.0, 0.0, 0.0];
    let s = vec![0.0, 1.0, 3.0];
    let z = vec![1.0, 1.0, 1.0]; // Dual variables
    let tau = 1.0;

    let cones = vec![SupportedConeT::ExponentialConeT()];

    let dP = zero_csc(n, n);
    let dq = vec![0.1, 0.0, 0.0];
    let dA = zero_csc(m, n);
    let db = vec![0.0, 0.0, 0.0];

    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dx.len(), n);
    assert_eq!(result.dz.len(), m);
    assert_eq!(result.ds.len(), m);
}

#[test]
fn test_integration_exp_cone_backward() {
    let n = 3;
    let m = 3;

    let P = zero_csc(n, n);
    let q = vec![1.0, 1.0, 1.0];
    let A = identity_csc(n);
    let b = vec![0.0, 1.0, 3.0];

    let x = vec![0.0, 0.0, 0.0];
    let s = vec![0.0, 1.0, 3.0];
    let z = vec![1.0, 1.0, 1.0];
    let tau = 1.0;

    let cones = vec![SupportedConeT::ExponentialConeT()];

    let dx_bar = vec![1.0, 0.0, 0.0];
    let dz_bar = vec![0.0, 0.0, 0.0];
    let ds_bar = vec![0.0, 0.0, 0.0];

    let result = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dq.len(), n);
    assert_eq!(result.db.len(), m);
}

#[test]
fn test_integration_exp_cone_forward_adjoint_consistency() {
    // Test forward-adjoint consistency for exponential cone
    //
    // CRITICAL: We must use a point (x, s, z) that satisfies KKT conditions!
    // This test uses actual solution data obtained by solving:
    //   min (1/2)x'Px + q'x s.t. Ax + s = b, s in K_exp
    // with P = I, A = I (identity), q = [-1, -2, -3], b = [0, 2, 10]
    //
    // The solver converges to a point satisfying:
    //   1. Primal feasibility: Ax + s = b (x + s = b)
    //   2. Dual feasibility: Px + q + A'z = 0 (x + q + z = 0)
    //   3. Complementarity: <s, z> ≈ 0
    let n = 3;
    let m = 3;

    // P = I (diagonal)
    let P = CscMatrix {
        m: n,
        n: n,
        colptr: vec![0, 1, 2, 3],
        rowval: vec![0, 1, 2],
        nzval: vec![1.0, 1.0, 1.0],
    };
    let q = vec![-1.0, -2.0, -3.0];
    let A = identity_csc(n);
    let b = vec![0.0, 2.0, 10.0];

    // Solution from solver (converged to tolerance 1e-8)
    let x = vec![1.000000001799601, 1.999924838047244, 3.000000001553104];
    let s = vec![-1.000000001799601, 0.000075161951437, 6.999999998446897];
    let z = vec![-0.000000001271995, 0.000075163006650, 0.000000000029715];
    let tau = 1.0;

    let cones = vec![SupportedConeT::ExponentialConeT()];

    // Forward perturbations
    let dP = zero_csc(n, n);
    let dq = vec![0.1, 0.2, 0.3];
    let dA = zero_csc(m, n);
    let db = vec![0.4, 0.5, 0.6];

    let forward = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // Adjoint directions
    let dx_bar = vec![1.0, 0.0, 0.0];
    let dz_bar = vec![0.0, 0.0, 0.0];
    let ds_bar = vec![0.0, 0.0, 0.0];

    let adjoint = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // LHS: <forward, bar>
    let lhs: f64 = forward
        .dx
        .iter()
        .zip(dx_bar.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + forward
            .dz
            .iter()
            .zip(dz_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
        + forward
            .ds
            .iter()
            .zip(ds_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    // RHS: <d, adjoint>
    let rhs: f64 = dq
        .iter()
        .zip(adjoint.dq.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + db.iter()
            .zip(adjoint.db.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    assert!(
        (lhs - rhs).abs() < 1e-4,
        "Forward-adjoint consistency: LHS = {}, RHS = {}",
        lhs,
        rhs
    );
}

// ========================================================================
// Power cone integration tests
// ========================================================================

#[test]
fn test_integration_power_cone_forward() {
    // Problem with power cone constraint
    // Power cone: x^α * y^(1-α) >= |z|, x,y >= 0
    let n = 3;
    let m = 3;

    let P = zero_csc(n, n);
    let q = vec![1.0, 1.0, 1.0];
    let A = identity_csc(n);
    let b = vec![4.0, 4.0, 1.0]; // Solution inside power cone: 4^0.5 * 4^0.5 = 4 > 1

    let x = vec![0.0, 0.0, 0.0];
    let s = vec![4.0, 4.0, 1.0]; // Inside power cone with alpha=0.5
    let z = vec![0.5, 0.5, 0.5];
    let tau = 1.0;

    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let dP = zero_csc(n, n);
    let dq = vec![0.1, 0.0, 0.0];
    let dA = zero_csc(m, n);
    let db = vec![0.0, 0.0, 0.0];

    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dx.len(), n);
    assert_eq!(result.dz.len(), m);
    assert_eq!(result.ds.len(), m);
}

#[test]
fn test_integration_power_cone_backward() {
    let n = 3;
    let m = 3;

    let P = zero_csc(n, n);
    let q = vec![1.0, 1.0, 1.0];
    let A = identity_csc(n);
    let b = vec![4.0, 4.0, 1.0];

    let x = vec![0.0, 0.0, 0.0];
    let s = vec![4.0, 4.0, 1.0];
    let z = vec![0.5, 0.5, 0.5];
    let tau = 1.0;

    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let dx_bar = vec![1.0, 0.0, 0.0];
    let dz_bar = vec![0.0, 0.0, 0.0];
    let ds_bar = vec![0.0, 0.0, 0.0];

    let result = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dq.len(), n);
    assert_eq!(result.db.len(), m);
}

#[test]
fn test_integration_power_cone_alpha_third() {
    // Power cone with alpha = 1/3
    let n = 3;
    let m = 3;

    let P = zero_csc(n, n);
    let q = vec![1.0, 1.0, 1.0];
    let A = identity_csc(n);
    let b = vec![8.0, 8.0, 1.0]; // 8^(1/3) * 8^(2/3) = 2 * 4 = 8 > 1

    let x = vec![0.0, 0.0, 0.0];
    let s = vec![8.0, 8.0, 1.0];
    let z = vec![0.5, 0.5, 0.5];
    let tau = 1.0;

    let cones = vec![SupportedConeT::PowerConeT(1.0 / 3.0)];

    let dP = zero_csc(n, n);
    let dq = vec![0.1, 0.1, 0.1];
    let dA = zero_csc(m, n);
    let db = vec![0.0, 0.0, 0.0];

    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dx.len(), n);
    assert_eq!(result.dz.len(), m);
    assert_eq!(result.ds.len(), m);
}

#[test]
fn test_integration_power_cone_forward_adjoint_consistency() {
    // Test forward-adjoint consistency for power cone
    //
    // CRITICAL: We must use a point (x, s, z) that satisfies KKT conditions!
    // This test uses actual solution data obtained by solving:
    //   min (1/2)x'Px + q'x s.t. Ax + s = b, s in K_pow(0.5)
    // with P = I, A = I (identity), q = [-1, -1, -0.5], b = [2, 2, 1]
    let n = 3;
    let m = 3;

    // P = I (diagonal)
    let P = CscMatrix {
        m: n,
        n: n,
        colptr: vec![0, 1, 2, 3],
        rowval: vec![0, 1, 2],
        nzval: vec![1.0, 1.0, 1.0],
    };
    let q = vec![-1.0, -1.0, -0.5];
    let A = identity_csc(n);
    let b = vec![2.0, 2.0, 1.0];

    // Solution from solver (converged to tolerance 1e-8)
    let x = vec![0.999999991503699, 0.999999991503699, 0.500000012743991];
    let s = vec![1.000000008495949, 1.000000008495949, 0.499999987255597];
    let z = vec![0.000000008533425, 0.000000008533425, -0.000000012725664];
    let tau = 1.0;

    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    // Forward perturbations
    let dP = zero_csc(n, n);
    let dq = vec![0.1, 0.2, 0.3];
    let dA = zero_csc(m, n);
    let db = vec![0.4, 0.5, 0.6];

    let forward = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // Adjoint directions
    let dx_bar = vec![1.0, 0.0, 0.0];
    let dz_bar = vec![0.0, 0.0, 0.0];
    let ds_bar = vec![0.0, 0.0, 0.0];

    let adjoint = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // LHS: <forward, bar>
    let lhs: f64 = forward
        .dx
        .iter()
        .zip(dx_bar.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + forward
            .dz
            .iter()
            .zip(dz_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
        + forward
            .ds
            .iter()
            .zip(ds_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    // RHS: <d, adjoint>
    let rhs: f64 = dq
        .iter()
        .zip(adjoint.dq.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + db.iter()
            .zip(adjoint.db.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    assert!(
        (lhs - rhs).abs() < 1e-4,
        "Forward-adjoint consistency: LHS = {}, RHS = {}",
        lhs,
        rhs
    );
}

// ========================================================================
// All cones together integration tests
// ========================================================================

#[test]
fn test_integration_all_cones_forward() {
    // Problem with all cone types: Zero + Nonneg + SOC + Exp + Power
    // Constraint dimensions: 1 (zero) + 2 (nonneg) + 3 (soc) + 3 (exp) + 3 (power) = 12
    let n = 4;
    let m = 12;

    let P = identity_csc(n);
    let q = vec![0.0; n];

    // Build A matrix: maps 4 variables to 12 constraint dimensions
    // Simple: each row of A has one nonzero
    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 3, 6, 9, 12],
        rowval: vec![
            0, 3, 6, // x[0] affects rows 0, 3, 6
            1, 4, 7, // x[1] affects rows 1, 4, 7
            2, 5, 8, // x[2] affects rows 2, 5, 8
            9, 10, 11, // x[3] affects rows 9, 10, 11
        ],
        nzval: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    };

    let b = vec![
        0.0, // Zero cone (s = 0)
        1.0, 1.0, // Nonneg cone (s >= 0)
        2.0, 0.5, 0.5, // SOC (t >= ||x||)
        0.0, 1.0, 3.0, // Exp cone interior
        4.0, 4.0, 1.0, // Power cone interior (alpha=0.5)
    ];

    let x = vec![0.0, 0.0, 0.0, 0.0];
    let s = vec![
        0.0, // Zero cone
        1.0, 1.0, // Nonneg
        2.0, 0.5, 0.5, // SOC interior
        0.0, 1.0, 3.0, // Exp interior
        4.0, 4.0, 1.0, // Power interior
    ];
    let z = vec![0.1; m];
    let tau = 1.0;

    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::SecondOrderConeT(3),
        SupportedConeT::ExponentialConeT(),
        SupportedConeT::PowerConeT(0.5),
    ];

    let dP = zero_csc(n, n);
    let dq = vec![0.1, 0.0, 0.0, 0.0];
    let dA = zero_csc(m, n);
    let db = vec![0.0; m];

    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dx.len(), n);
    assert_eq!(result.dz.len(), m);
    assert_eq!(result.ds.len(), m);
}

#[test]
fn test_integration_all_cones_backward() {
    let n = 4;
    let m = 12;

    let P = identity_csc(n);
    let q = vec![0.0; n];

    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 3, 6, 9, 12],
        rowval: vec![0, 3, 6, 1, 4, 7, 2, 5, 8, 9, 10, 11],
        nzval: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    };

    let b = vec![0.0, 1.0, 1.0, 2.0, 0.5, 0.5, 0.0, 1.0, 3.0, 4.0, 4.0, 1.0];

    let x = vec![0.0, 0.0, 0.0, 0.0];
    let s = vec![0.0, 1.0, 1.0, 2.0, 0.5, 0.5, 0.0, 1.0, 3.0, 4.0, 4.0, 1.0];
    let z = vec![0.1; m];
    let tau = 1.0;

    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::SecondOrderConeT(3),
        SupportedConeT::ExponentialConeT(),
        SupportedConeT::PowerConeT(0.5),
    ];

    let dx_bar = vec![1.0, 0.0, 0.0, 0.0];
    let dz_bar = vec![0.0; m];
    let ds_bar = vec![0.0; m];

    let result = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dq.len(), n);
    assert_eq!(result.db.len(), m);
}

#[test]
fn test_integration_zero_nonneg_soc_forward() {
    // Simpler mixed cone: Zero + Nonneg + SOC
    let n = 3;
    let m = 6; // 1 + 2 + 3

    let P = identity_csc(n);
    let q = vec![0.0; n];

    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 2, 4, 6],
        rowval: vec![
            0, 3, // x[0] -> zero, soc
            1, 4, // x[1] -> nonneg, soc
            2, 5, // x[2] -> nonneg, soc
        ],
        nzval: vec![1.0; 6],
    };

    let b = vec![
        0.0, // Zero
        1.0, 1.0, // Nonneg
        2.0, 0.5, 0.5, // SOC
    ];

    let x = vec![0.0, 0.0, 0.0];
    let s = vec![0.0, 1.0, 1.0, 2.0, 0.5, 0.5];
    let z = vec![0.1; m];
    let tau = 1.0;

    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::SecondOrderConeT(3),
    ];

    let dP = zero_csc(n, n);
    let dq = vec![0.1, 0.0, 0.0];
    let dA = zero_csc(m, n);
    let db = vec![0.0; m];

    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dx.len(), n);
    assert_eq!(result.dz.len(), m);
    assert_eq!(result.ds.len(), m);
}

#[test]
fn test_integration_exp_power_combined() {
    // Combined exponential and power cones
    let n = 3;
    let m = 6; // 3 (exp) + 3 (power)

    let P = zero_csc(n, n);
    let q = vec![1.0, 1.0, 1.0];

    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 2, 4, 6],
        rowval: vec![
            0, 3, // x[0] -> exp[0], pow[0]
            1, 4, // x[1] -> exp[1], pow[1]
            2, 5, // x[2] -> exp[2], pow[2]
        ],
        nzval: vec![1.0; 6],
    };

    let b = vec![
        0.0, 1.0, 3.0, // Exp cone interior
        4.0, 4.0, 1.0, // Power cone interior
    ];

    let x = vec![0.0, 0.0, 0.0];
    let s = vec![0.0, 1.0, 3.0, 4.0, 4.0, 1.0];
    let z = vec![0.5; m];
    let tau = 1.0;

    let cones = vec![
        SupportedConeT::ExponentialConeT(),
        SupportedConeT::PowerConeT(0.5),
    ];

    let dP = zero_csc(n, n);
    let dq = vec![0.1, 0.1, 0.1];
    let dA = zero_csc(m, n);
    let db = vec![0.0; m];

    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dx.len(), n);
    assert_eq!(result.dz.len(), m);
    assert_eq!(result.ds.len(), m);
}

#[test]
fn test_integration_multiple_same_cone() {
    // Multiple instances of the same cone type
    // 2 exponential cones + 2 power cones
    let n = 4;
    let m = 12; // 3 + 3 + 3 + 3

    let P = zero_csc(n, n);
    let q = vec![1.0; n];

    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 3, 6, 9, 12],
        rowval: vec![
            0, 3, 6, // x[0]
            1, 4, 7, // x[1]
            2, 5, 8, // x[2]
            9, 10, 11, // x[3]
        ],
        nzval: vec![1.0; 12],
    };

    let b = vec![
        0.0, 1.0, 3.0, // Exp cone 1 interior
        0.0, 2.0, 8.0, // Exp cone 2 interior
        4.0, 4.0, 1.0, // Power cone 1 interior
        9.0, 9.0, 2.0, // Power cone 2 interior
    ];

    let x = vec![0.0; n];
    let s = b.clone();
    let z = vec![0.1; m];
    let tau = 1.0;

    let cones = vec![
        SupportedConeT::ExponentialConeT(),
        SupportedConeT::ExponentialConeT(),
        SupportedConeT::PowerConeT(0.5),
        SupportedConeT::PowerConeT(0.5),
    ];

    let dP = zero_csc(n, n);
    let dq = vec![0.1; n];
    let dA = zero_csc(m, n);
    let db = vec![0.0; m];

    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(result.dx.len(), n);
    assert_eq!(result.dz.len(), m);
    assert_eq!(result.ds.len(), m);
}

// ========================================================================
// Validation tests for unsupported cones
// ========================================================================

#[test]
fn test_genpow_cone_forward_does_not_panic() {
    // GenPowerCone differentiation is now supported
    let n = 3;
    let m = 4;

    let P = zero_csc(n, n);
    let q = vec![1.0; n];
    let A = CscMatrix {
        m,
        n,
        colptr: vec![0, 1, 2, 4],
        rowval: vec![0, 1, 2, 3],
        nzval: vec![1.0; 4],
    };
    let b = vec![1.0; m];

    let x = vec![0.0; n];
    let s = vec![1.0; m];
    let z = vec![0.1; m];
    let tau = 1.0;

    let cones = vec![SupportedConeT::GenPowerConeT(vec![0.5, 0.5], 2)];

    let dP = zero_csc(n, n);
    let dq = vec![0.1; n];
    let dA = zero_csc(m, n);
    let db = vec![0.0; m];

    // Should not panic (GenPowerCone is now supported)
    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );
    assert!(result.dx.len() == n);
}

#[test]
fn test_genpow_cone_high_dim_forward_adjoint_consistency() {
    // High-dim GenPowerCone: 10 alphas, dim2=5, total dim=15.
    // Exercises the larger Jacobian computation.
    let dim1 = 10;
    let dim2 = 5;
    let m = dim1 + dim2; // 15
    let n = m;

    let P = identity_csc(n);
    let q = vec![0.0; n];
    let A = identity_csc(n);

    // Build b so that s = b is strictly interior to GenPowerCone:
    // p_i > 0 and prod(p_i^alpha_i) > ||w||
    let mut b = vec![0.0; m];
    for i in 0..dim1 {
        b[i] = 2.0 + (i as f64) * 0.1; // p = [2.0, 2.1, ..., 2.9]
    }
    for i in 0..dim2 {
        b[dim1 + i] = 0.1 * (i as f64 + 1.0); // w = [0.1, 0.2, 0.3, 0.4, 0.5]
    }

    let x = vec![0.0; n];
    let s = b.clone();
    let z = vec![0.1; m];
    let tau = 1.0;

    let alphas: Vec<f64> = vec![0.1; dim1]; // uniform, sum = 1
    let cones = vec![SupportedConeT::GenPowerConeT(alphas, dim2)];

    // Forward perturbations
    let dP = zero_csc(n, n);
    let dq: Vec<f64> = (0..n).map(|i| 0.01 * (i as f64 + 1.0)).collect();
    let dA = zero_csc(m, n);
    let db: Vec<f64> = (0..m).map(|i| 0.005 * (i as f64 + 1.0)).collect();

    let forward = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(forward.dx.len(), n);
    assert_eq!(forward.dz.len(), m);
    assert_eq!(forward.ds.len(), m);

    // Adjoint directions
    let dx_bar: Vec<f64> = (0..n).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let dz_bar = vec![0.0; m];
    let ds_bar = vec![0.0; m];

    let adjoint = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    // LHS: <forward, bar>
    let lhs: f64 = forward
        .dx
        .iter()
        .zip(dx_bar.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + forward
            .dz
            .iter()
            .zip(dz_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
        + forward
            .ds
            .iter()
            .zip(ds_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    // RHS: <d, adjoint>
    let rhs: f64 = dq
        .iter()
        .zip(adjoint.dq.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + db.iter()
            .zip(adjoint.db.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    assert!(
        (lhs - rhs).abs() < 1e-3,
        "Forward-adjoint consistency: LHS = {}, RHS = {}, diff = {}",
        lhs,
        rhs,
        (lhs - rhs).abs()
    );
}

#[test]
fn test_genpow_cone_w_zero_forward_adjoint_consistency() {
    // Edge case: w = 0 (the dim2 portion of s/z is zero).
    // This tests the singularity where p, r vectors in the Hessian
    // have zero dim2 components.
    let dim1 = 2;
    let dim2 = 2;
    let m = dim1 + dim2; // 4
    let n = m;

    let P = identity_csc(n);
    let q = vec![0.0; n];
    let A = identity_csc(n);

    // s interior to cone with w = 0: prod(p_i^alpha_i) > 0 = ||w||
    let b = vec![3.0, 2.0, 0.0, 0.0]; // p = [3, 2], w = [0, 0]
    let x = vec![0.0; n];
    let s = b.clone();
    // z must be in dual cone interior: prod((z_i/alpha_i)^alpha_i) > ||w_z||
    let z = vec![1.0, 1.0, 0.1, 0.1];
    let tau = 1.0;

    let cones = vec![SupportedConeT::GenPowerConeT(vec![0.5, 0.5], dim2)];

    let dP = zero_csc(n, n);
    let dq = vec![0.1, 0.2, 0.3, 0.4];
    let dA = zero_csc(m, n);
    let db = vec![0.5, 0.6, 0.7, 0.8];

    let forward = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(forward.dx.len(), n);
    // Verify no NaN/Inf in results
    for v in forward
        .dx
        .iter()
        .chain(forward.dz.iter())
        .chain(forward.ds.iter())
    {
        assert!(v.is_finite(), "Non-finite value in forward result with w=0");
    }

    let dx_bar = vec![1.0, 0.0, 0.0, 0.0];
    let dz_bar = vec![0.0; m];
    let ds_bar = vec![0.0; m];

    let adjoint = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    for v in adjoint.dq.iter().chain(adjoint.db.iter()) {
        assert!(v.is_finite(), "Non-finite value in adjoint result with w=0");
    }

    // Forward-adjoint consistency
    let lhs: f64 = forward
        .dx
        .iter()
        .zip(dx_bar.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + forward
            .dz
            .iter()
            .zip(dz_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
        + forward
            .ds
            .iter()
            .zip(ds_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    let rhs: f64 = dq
        .iter()
        .zip(adjoint.dq.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + db.iter()
            .zip(adjoint.db.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    assert!(
        (lhs - rhs).abs() < 1e-3,
        "Forward-adjoint consistency with w=0: LHS = {}, RHS = {}, diff = {}",
        lhs,
        rhs,
        (lhs - rhs).abs()
    );
}

#[test]
fn test_genpow_cone_near_boundary_forward_adjoint_consistency() {
    // Edge case: s is near the cone boundary (ζ is small).
    // prod(p_i^alpha_i) ≈ ||w||, so the Hessian has large condition number.
    let dim1 = 2;
    let dim2 = 1;
    let m = dim1 + dim2; // 3
    let n = m;

    let P = identity_csc(n);
    let q = vec![0.0; n];
    let A = identity_csc(n);

    // s near boundary: prod(p_i^alpha_i) slightly > ||w||
    // p = [1.0, 1.0], alpha = [0.5, 0.5] => prod = 1.0^0.5 * 1.0^0.5 = 1.0
    // w = [0.99] => ||w|| = 0.99, so ζ = 1.0 - 0.99 = 0.01
    let b = vec![1.0, 1.0, 0.99];
    let x = vec![0.0; n];
    let s = b.clone();
    let z = vec![1.0, 1.0, 0.1]; // interior of dual cone
    let tau = 1.0;

    let cones = vec![SupportedConeT::GenPowerConeT(vec![0.5, 0.5], dim2)];

    let dP = zero_csc(n, n);
    let dq = vec![0.1, 0.2, 0.3];
    let dA = zero_csc(m, n);
    let db = vec![0.4, 0.5, 0.6];

    let forward = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(forward.dx.len(), n);
    for v in forward
        .dx
        .iter()
        .chain(forward.dz.iter())
        .chain(forward.ds.iter())
    {
        assert!(
            v.is_finite(),
            "Non-finite value in forward result near boundary"
        );
    }

    let dx_bar = vec![1.0, 0.0, 0.0];
    let dz_bar = vec![0.0; m];
    let ds_bar = vec![0.0; m];

    let adjoint = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    for v in adjoint.dq.iter().chain(adjoint.db.iter()) {
        assert!(
            v.is_finite(),
            "Non-finite value in adjoint result near boundary"
        );
    }

    let lhs: f64 = forward
        .dx
        .iter()
        .zip(dx_bar.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + forward
            .dz
            .iter()
            .zip(dz_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
        + forward
            .ds
            .iter()
            .zip(ds_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    let rhs: f64 = dq
        .iter()
        .zip(adjoint.dq.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + db.iter()
            .zip(adjoint.db.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    // Looser tolerance near boundary due to ill-conditioning
    assert!(
        (lhs - rhs).abs() < 1e-2,
        "Forward-adjoint consistency near boundary: LHS = {}, RHS = {}, diff = {}",
        lhs,
        rhs,
        (lhs - rhs).abs()
    );
}

#[test]
fn test_genpow_cone_extreme_scale_forward_adjoint_consistency() {
    // Edge case: very large p values to test overflow resilience.
    let dim1 = 3;
    let dim2 = 2;
    let m = dim1 + dim2; // 5
    let n = m;

    let P = identity_csc(n);
    let q = vec![0.0; n];
    let A = identity_csc(n);

    // Large p values: p = [100, 200, 150], w = [1, 1]
    // prod(p_i^alpha_i) = 100^0.2 * 200^0.3 * 150^0.5 ≈ 2.51 * 5.85 * 12.25 ≈ 179.9
    // ||w|| = sqrt(2) ≈ 1.41, so well interior
    let b = vec![100.0, 200.0, 150.0, 1.0, 1.0];
    let x = vec![0.0; n];
    let s = b.clone();
    let z = vec![0.1, 0.1, 0.1, 0.05, 0.05];
    let tau = 1.0;

    let cones = vec![SupportedConeT::GenPowerConeT(vec![0.2, 0.3, 0.5], dim2)];

    let dP = zero_csc(n, n);
    let dq = vec![0.01, 0.02, 0.03, 0.04, 0.05];
    let dA = zero_csc(m, n);
    let db = vec![0.0; m];

    let forward = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    assert_eq!(forward.dx.len(), n);
    for v in forward
        .dx
        .iter()
        .chain(forward.dz.iter())
        .chain(forward.ds.iter())
    {
        assert!(
            v.is_finite(),
            "Non-finite value in forward result with extreme scale"
        );
    }

    let dx_bar = vec![1.0, 0.0, 0.0, 0.0, 0.0];
    let dz_bar = vec![0.0; m];
    let ds_bar = vec![0.0; m];

    let adjoint = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );

    for v in adjoint.dq.iter().chain(adjoint.db.iter()) {
        assert!(
            v.is_finite(),
            "Non-finite value in adjoint result with extreme scale"
        );
    }

    let lhs: f64 = forward
        .dx
        .iter()
        .zip(dx_bar.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + forward
            .dz
            .iter()
            .zip(dz_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
        + forward
            .ds
            .iter()
            .zip(ds_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    let rhs: f64 = dq
        .iter()
        .zip(adjoint.dq.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + db.iter()
            .zip(adjoint.db.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    assert!(
        (lhs - rhs).abs() < 1e-3,
        "Forward-adjoint consistency extreme scale: LHS = {}, RHS = {}, diff = {}",
        lhs,
        rhs,
        (lhs - rhs).abs()
    );
}

#[test]
fn test_genpow_cone_backward_does_not_panic() {
    // GenPowerCone differentiation is now supported
    let n = 3;
    let m = 4;

    let P = zero_csc(n, n);
    let q = vec![1.0; n];
    let A = CscMatrix {
        m,
        n,
        colptr: vec![0, 1, 2, 4],
        rowval: vec![0, 1, 2, 3],
        nzval: vec![1.0; 4],
    };
    let b = vec![1.0; m];

    let x = vec![0.0; n];
    let s = vec![1.0; m];
    let z = vec![0.1; m];
    let tau = 1.0;

    let cones = vec![SupportedConeT::GenPowerConeT(vec![0.5, 0.5], 2)];

    let dx_bar = vec![1.0; n];
    let dz_bar = vec![0.0; m];
    let ds_bar = vec![0.0; m];

    // Should not panic (GenPowerCone is now supported)
    let result = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        TEST_DIFF_METHOD,
        TEST_MU,
    );
    assert!(result.dq.len() == n);
}

#[test]
fn test_genpow_cone_projection_correctness() {
    // Test projection properties for GenPowerCone:
    //  1. Interior points project to themselves
    //  2. Projected point satisfies cone constraint
    //  3. Moreau decomposition: v = Π_K(v) + Π_{K°}(v)
    //     equivalently: v = Π_K(v) - Π_{K*}(-v)

    let alpha = vec![0.3, 0.7];
    let dim2 = 2;
    let cones = vec![SupportedConeT::GenPowerConeT(alpha.clone(), dim2)];
    let dim = alpha.len() + dim2;

    // Helper: check prod(p_i^alpha_i) >= ||w||
    let in_cone = |proj: &[f64]| -> bool {
        let p = &proj[..alpha.len()];
        let w = &proj[alpha.len()..dim];
        if p.iter().any(|&pi| pi < -1e-10) {
            return false;
        }
        let prod: f64 = p
            .iter()
            .zip(alpha.iter())
            .map(|(&pi, &ai)| pi.max(0.0).powf(ai))
            .product();
        let norm_w: f64 = w.iter().map(|&wi| wi * wi).sum::<f64>().sqrt();
        prod >= norm_w - 1e-8
    };

    // Case 1: Interior point — should project to itself
    // p = [5, 3], alpha = [0.3, 0.7] => prod = 5^0.3 * 3^0.7 ≈ 1.62 * 2.27 ≈ 3.68
    // w = [0.5, 0.5] => ||w|| ≈ 0.71, well interior
    let v_interior = vec![5.0, 3.0, 0.5, 0.5];
    let proj = get_cone_projection(&v_interior, &cones, false);
    for i in 0..dim {
        assert!(
            (proj[i] - v_interior[i]).abs() < 1e-6,
            "Interior point should project to itself: proj[{}] = {}, v[{}] = {}",
            i,
            proj[i],
            i,
            v_interior[i]
        );
    }

    // Case 2: Point outside cone — projection must land in cone
    let v_outside = vec![-1.0, -2.0, 5.0, 5.0];
    let proj = get_cone_projection(&v_outside, &cones, false);
    assert!(
        in_cone(&proj),
        "Projection of exterior point must be in cone, got p={:?}, w={:?}",
        &proj[..alpha.len()],
        &proj[alpha.len()..dim]
    );

    // Case 3: Moreau decomposition for multiple test points
    // v = Π_K(v) + Π_{K°}(v) = Π_K(v) - Π_{K*}(-v)
    let test_points = vec![
        vec![5.0, 3.0, 0.5, 0.5],   // interior
        vec![-1.0, -2.0, 5.0, 5.0], // outside
        vec![1.0, 1.0, 0.0, 0.0],   // w = 0
        vec![0.1, 0.1, 10.0, 10.0], // far outside
        vec![2.0, 3.0, 1.0, -1.0],  // mixed signs in w
    ];

    for v in &test_points {
        let primal_proj = get_cone_projection(v, &cones, false);
        let neg_v: Vec<f64> = v.iter().map(|&x| -x).collect();
        let dual_proj_neg = get_cone_projection(&neg_v, &cones, true);
        // Moreau: v = Π_K(v) - Π_{K*}(-v)
        // i.e., v = primal_proj - dual_proj_neg
        for i in 0..dim {
            let reconstructed = primal_proj[i] - dual_proj_neg[i];
            assert!(
                (reconstructed - v[i]).abs() < 1e-6,
                "Moreau decomposition failed for v={:?}: \
                 v[{}]={}, Π_K(v)[{}]={}, Π_K*(-v)[{}]={}, reconstructed={}",
                v,
                i,
                v[i],
                i,
                primal_proj[i],
                i,
                dual_proj_neg[i],
                reconstructed
            );
        }
    }

    // Case 4: High-dim projection correctness (dim1=10, dim2=5)
    let alpha_high: Vec<f64> = vec![0.1; 10];
    let dim2_high = 5;
    let cones_high = vec![SupportedConeT::GenPowerConeT(alpha_high.clone(), dim2_high)];
    let dim_high = 10 + dim2_high;

    // Exterior point: small p, large w
    let mut v_high = vec![0.5; dim_high];
    for i in 10..dim_high {
        v_high[i] = 10.0;
    }
    let proj = get_cone_projection(&v_high, &cones_high, false);
    // Check cone membership
    let p = &proj[..10];
    let w = &proj[10..dim_high];
    assert!(
        p.iter().all(|&pi| pi >= -1e-10),
        "p must be nonneg after projection"
    );
    let prod: f64 = p
        .iter()
        .zip(alpha_high.iter())
        .map(|(&pi, &ai)| pi.max(0.0).powf(ai))
        .product();
    let norm_w: f64 = w.iter().map(|&wi| wi * wi).sum::<f64>().sqrt();
    assert!(
        prod >= norm_w - 1e-6,
        "High-dim projection must satisfy cone constraint: prod={}, ||w||={}",
        prod,
        norm_w
    );

    // Moreau check for high-dim
    let neg_v_high: Vec<f64> = v_high.iter().map(|&x| -x).collect();
    let primal_proj = get_cone_projection(&v_high, &cones_high, false);
    let dual_proj_neg = get_cone_projection(&neg_v_high, &cones_high, true);
    for i in 0..dim_high {
        let reconstructed = primal_proj[i] - dual_proj_neg[i];
        assert!(
            (reconstructed - v_high[i]).abs() < 1e-6,
            "High-dim Moreau decomposition failed at index {}",
            i
        );
    }
}

// Note: test_soc_dim_not_3_forward_panics and test_soc_dim_not_3_backward_panics
// were removed because SOC dim < 2 is now rejected at cone creation time (via
// validate_cones), not at differentiation time. The original tests expected a
// panic during differentiation, but with the minimum dim=2 restriction,
// such cones cannot even be created.

// ========================================================================
// BarrierProx forward-adjoint consistency tests
// ========================================================================

const TEST_BARRIER_PROX_MU: f64 = 1e-6;

#[test]
fn test_integration_qp_equality_forward_barrier_prox() {
    // Zero-cone-only problem: bypasses diff_method entirely (uses QP equality path).
    // Should produce the same result regardless of diff_method.
    let P = identity_csc(2);
    let q = vec![0.0, 0.0];
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b = vec![2.0];
    let cones = vec![SupportedConeT::ZeroConeT(1)];

    let x = vec![1.0, 1.0];
    let s = vec![0.0];
    let z = vec![0.0];
    let tau = 1.0;

    let dP = zero_csc(2, 2);
    let dq = vec![1.0, 0.0];
    let dA = zero_csc(1, 2);
    let db = vec![0.0];

    let result_proj = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Exact,
        0.0,
    );
    let result_bp = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Smoothed,
        TEST_BARRIER_PROX_MU,
    );

    for i in 0..2 {
        assert!(
            (result_proj.dx[i] - result_bp.dx[i]).abs() < 1e-10,
            "QP eq forward: dx[{}] proj={} bp={}",
            i,
            result_proj.dx[i],
            result_bp.dx[i]
        );
    }
}

#[test]
fn test_integration_nonneg_forward_adjoint_barrier_prox() {
    // Nonneg cone with BarrierProx: check forward-adjoint identity
    let P = zero_csc(2, 2);
    let q = vec![1.0, 1.0];
    let A = identity_csc(2);
    let b = vec![1.0, 1.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(2)];

    let x = vec![0.0, 0.0];
    let s = vec![1.0, 1.0];
    let z = vec![1.0, 1.0];
    let tau = 1.0;

    let dP = zero_csc(2, 2);
    let dq = vec![0.3, 0.4];
    let dA = zero_csc(2, 2);
    let db = vec![0.5, 0.6];

    let forward = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Smoothed,
        TEST_BARRIER_PROX_MU,
    );

    let dx_bar = vec![0.7, 0.8];
    let dz_bar = vec![0.9, 1.0];
    let ds_bar = vec![0.1, 0.2];

    let adjoint = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Smoothed,
        TEST_BARRIER_PROX_MU,
    );

    let lhs: f64 = forward
        .dx
        .iter()
        .zip(dx_bar.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + forward
            .dz
            .iter()
            .zip(dz_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
        + forward
            .ds
            .iter()
            .zip(ds_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    let rhs: f64 = dq
        .iter()
        .zip(adjoint.dq.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + db.iter()
            .zip(adjoint.db.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    assert!(
        (lhs - rhs).abs() < 1e-4,
        "Nonneg BarrierProx forward-adjoint: LHS = {}, RHS = {}",
        lhs,
        rhs
    );
}

// ========================================================================
// Unsupported cone panic test
// ========================================================================

#[test]
#[should_panic(expected = "Smoothed differentiation not yet implemented")]
fn test_smoothed_unsupported_cone_panics() {
    // Smoothed differentiation should panic for exponential cones (not yet implemented)
    let n = 2;
    let m = 3;

    let P = identity_csc(n);
    let q = vec![0.0, 0.0];
    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 2],
        nzval: vec![1.0, 1.0, 0.0, 1.0],
    };
    let b = vec![2.0, 0.5, 0.5];
    let cones = vec![SupportedConeT::ExponentialConeT()];

    let x = vec![1.0, 0.5];
    let s = vec![1.0, 0.5, 0.0];
    let z = vec![0.5, 0.25, 0.0];
    let tau = 1.0;

    let dP = zero_csc(n, n);
    let dq = vec![0.1, 0.2];
    let dA = zero_csc(m, n);
    let db = vec![0.3, 0.4, 0.5];

    // This should panic because Exp smoothed diff is not implemented
    // (the test problem uses ExponentialConeT; SOC smoothed diff IS supported).
    differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Smoothed,
        TEST_BARRIER_PROX_MU,
    );
}

// ========================================================================
// LP (zero + nonneg) Smoothed differentiation tests
// ========================================================================

#[test]
fn test_smoothed_matches_exact_at_convergence_lp() {
    // For a fully-converged LP (nonneg cone), Smoothed with small μ
    // should give nearly the same gradients as Exact.
    let n = 2;
    let m = 3;

    // min x1 + x2  s.t.  x1 + x2 = 1, x1 >= 0, x2 >= 0
    // P = 0, q = [1, 1]
    // A = [[1,1],[1,0],[0,1]], b = [1, 0, 0]
    // cones: ZeroCone(1), NonnegCone(2)
    // Solution: x = [0.5, 0.5] (by symmetry, degenerate)
    // Actually: any x1+x2=1, x1,x2>=0 is optimal. z=[1,0,0], s=[0,0.5,0.5]
    let P = zero_csc(n, n);
    let q = vec![1.0, 1.0];
    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 2],
        nzval: vec![1.0, 1.0, 1.0, 1.0],
    };
    let b = vec![1.0, 0.0, 0.0];
    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
    ];

    // Use a non-degenerate solution point with active constraints away from boundary
    let x = vec![0.5, 0.5];
    let s = vec![0.0, 0.5, 0.5];
    let z = vec![-1.0, 0.0, 0.0]; // dual: -1 for equality, 0 for inactive nonneg
    let tau = 1.0;

    let dx_bar = vec![1.0, 0.0];
    let dz_bar = vec![0.0, 0.0, 0.0];
    let ds_bar = vec![0.0, 0.0, 0.0];

    let exact = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Exact,
        0.0,
    );

    // Small mu: Smoothed should approximate Exact
    let small_mu = 1e-8;
    let smoothed = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Smoothed,
        small_mu,
    );

    for i in 0..n {
        assert!(
            (exact.dq[i] - smoothed.dq[i]).abs() < 1e-4,
            "dq[{}]: exact={}, smoothed={}",
            i,
            exact.dq[i],
            smoothed.dq[i]
        );
    }
    for i in 0..m {
        assert!(
            (exact.db[i] - smoothed.db[i]).abs() < 1e-4,
            "db[{}]: exact={}, smoothed={}",
            i,
            exact.db[i],
            smoothed.db[i]
        );
    }
}

#[test]
fn test_genpow_cone_jacobian_finite_difference() {
    // Verify that the analytical Jacobian from get_cone_derivative_sparse
    // matches numerical finite-difference Jacobian of get_cone_projection.

    let eps = 1e-7;
    let tol = 1e-4; // FD tolerance

    // Test case 1: General point (alpha=[0.3,0.7], dim2=2)
    // Point outside cone so projection is nontrivial
    {
        let alpha = vec![0.3, 0.7];
        let dim2 = 2;
        let cones = vec![SupportedConeT::GenPowerConeT(alpha.clone(), dim2)];
        let dim = alpha.len() + dim2;

        // Point that lands on the boundary after projection
        let z = vec![0.5, 0.8, 3.0, 2.0];

        // Analytical Jacobian
        let blocks = get_cone_derivative_sparse(&z, &cones, false);
        let jac_data = genpow_sparse_to_dense(&blocks[0], dim);

        // Numerical Jacobian via central FD
        let mut jac_fd = vec![0.0_f64; dim * dim];
        for j in 0..dim {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();
            z_plus[j] += eps;
            z_minus[j] -= eps;
            let proj_plus = get_cone_projection(&z_plus, &cones, false);
            let proj_minus = get_cone_projection(&z_minus, &cones, false);
            for i in 0..dim {
                jac_fd[i * dim + j] = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
            }
        }

        // Compare
        let mut max_diff = 0.0_f64;
        for k in 0..dim * dim {
            let diff = (jac_data[k] - jac_fd[k]).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff < tol,
            "Jacobian FD mismatch (case 1): max_diff = {:.2e}, tol = {:.2e}\n\
             Analytical:\n{:?}\nFD:\n{:?}",
            max_diff,
            tol,
            jac_data,
            jac_fd
        );
    }

    // Test case 2: High-dim (alpha=[0.1]*6, dim2=3, total=9)
    {
        let alpha = vec![0.1, 0.15, 0.2, 0.15, 0.25, 0.15];
        let dim2 = 3;
        let cones = vec![SupportedConeT::GenPowerConeT(alpha.clone(), dim2)];
        let dim = alpha.len() + dim2;

        // Point outside cone
        let mut z = vec![0.0_f64; dim];
        for i in 0..alpha.len() {
            z[i] = 0.3 + 0.1 * (i as f64);
        }
        for i in alpha.len()..dim {
            z[i] = 2.0 + 0.5 * (i as f64);
        }

        let blocks = get_cone_derivative_sparse(&z, &cones, false);
        let jac_data = genpow_sparse_to_dense(&blocks[0], dim);

        let mut jac_fd = vec![0.0_f64; dim * dim];
        for j in 0..dim {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();
            z_plus[j] += eps;
            z_minus[j] -= eps;
            let proj_plus = get_cone_projection(&z_plus, &cones, false);
            let proj_minus = get_cone_projection(&z_minus, &cones, false);
            for i in 0..dim {
                jac_fd[i * dim + j] = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
            }
        }

        let mut max_diff = 0.0_f64;
        for k in 0..dim * dim {
            let diff = (jac_data[k] - jac_fd[k]).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff < tol,
            "Jacobian FD mismatch (case 2, high-dim): max_diff = {:.2e}",
            max_diff
        );
    }

    // Test case 3: Dual cone Jacobian
    {
        let alpha = vec![0.4, 0.6];
        let dim2 = 1;
        let cones = vec![SupportedConeT::GenPowerConeT(alpha.clone(), dim2)];
        let dim = alpha.len() + dim2;

        let z = vec![1.0, 0.5, -3.0];

        let blocks = get_cone_derivative_sparse(&z, &cones, true);
        let jac_data = genpow_sparse_to_dense(&blocks[0], dim);

        let mut jac_fd = vec![0.0_f64; dim * dim];
        for j in 0..dim {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();
            z_plus[j] += eps;
            z_minus[j] -= eps;
            let proj_plus = get_cone_projection(&z_plus, &cones, true);
            let proj_minus = get_cone_projection(&z_minus, &cones, true);
            for i in 0..dim {
                jac_fd[i * dim + j] = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
            }
        }

        let mut max_diff = 0.0_f64;
        for k in 0..dim * dim {
            let diff = (jac_data[k] - jac_fd[k]).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff < tol,
            "Dual Jacobian FD mismatch (case 3): max_diff = {:.2e}",
            max_diff
        );
    }

    // Test case 4: dim2=1 edge case
    {
        let alpha = vec![0.5, 0.5];
        let dim2 = 1;
        let cones = vec![SupportedConeT::GenPowerConeT(alpha.clone(), dim2)];
        let dim = alpha.len() + dim2;

        let z = vec![0.3, 0.4, 5.0]; // outside cone

        let blocks = get_cone_derivative_sparse(&z, &cones, false);
        let jac_data = genpow_sparse_to_dense(&blocks[0], dim);

        let mut jac_fd = vec![0.0_f64; dim * dim];
        for j in 0..dim {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();
            z_plus[j] += eps;
            z_minus[j] -= eps;
            let proj_plus = get_cone_projection(&z_plus, &cones, false);
            let proj_minus = get_cone_projection(&z_minus, &cones, false);
            for i in 0..dim {
                jac_fd[i * dim + j] = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
            }
        }

        let mut max_diff = 0.0_f64;
        for k in 0..dim * dim {
            let diff = (jac_data[k] - jac_fd[k]).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(
            max_diff < tol,
            "Jacobian FD mismatch (case 4, dim2=1): max_diff = {:.2e}",
            max_diff
        );
    }

    // Test case 5: 3 alphas dual (matching failing Python test)
    {
        let alpha = vec![0.2, 0.3, 0.5];
        let dim2 = 2;
        let cones = vec![SupportedConeT::GenPowerConeT(alpha.clone(), dim2)];
        let dim = alpha.len() + dim2;

        let z = vec![0.5, 0.8, 0.3, 3.0, 2.0];

        for dual in [false, true] {
            let blocks = get_cone_derivative_sparse(&z, &cones, dual);
            let jac_data = genpow_sparse_to_dense(&blocks[0], dim);

            let mut jac_fd = vec![0.0_f64; dim * dim];
            for j in 0..dim {
                let mut z_plus = z.clone();
                let mut z_minus = z.clone();
                z_plus[j] += eps;
                z_minus[j] -= eps;
                let proj_plus = get_cone_projection(&z_plus, &cones, dual);
                let proj_minus = get_cone_projection(&z_minus, &cones, dual);
                for i in 0..dim {
                    jac_fd[i * dim + j] = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
                }
            }

            let mut max_diff = 0.0_f64;
            for k in 0..dim * dim {
                let diff = (jac_data[k] - jac_fd[k]).abs();
                if diff > max_diff {
                    max_diff = diff;
                    if diff > tol {
                        let r = k / dim;
                        let c = k % dim;
                        eprintln!(
                            "  Jac[{},{}]: analytical={:.8e}, fd={:.8e}, diff={:.2e}",
                            r, c, jac_data[k], jac_fd[k], diff
                        );
                    }
                }
            }
            assert!(
                max_diff < tol,
                "Jacobian FD mismatch (case 5, 3 alphas, dual={}): max_diff = {:.2e}",
                dual,
                max_diff
            );
        }
    }
}

#[test]
fn test_lp_forward_adjoint_consistency_smoothed() {
    // LP with zero + nonneg cones: verify <forward(d), bar> = <d, adjoint(bar)>
    // using Smoothed differentiation (HSDE path, not QP-eq path).
    let n = 2;
    let m = 3;

    let P = zero_csc(n, n);
    let q = vec![1.0, 2.0];
    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 2],
        nzval: vec![1.0, 1.0, 1.0, 1.0],
    };
    let b = vec![1.0, 0.0, 0.0];
    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
    ];

    let x = vec![0.5, 0.5];
    let s = vec![0.0, 0.5, 0.5];
    let z = vec![-1.5, 0.5, 0.5];
    let tau = 1.0;
    let mu = 1e-6;

    // Forward perturbations
    let dP = zero_csc(n, n);
    let dq = vec![0.3, 0.4];
    let dA = zero_csc(m, n);
    let db = vec![0.5, 0.6, 0.7];

    let forward = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Smoothed,
        mu,
    );

    // Adjoint directions
    let dx_bar = vec![0.8, 0.9];
    let dz_bar = vec![1.0, 0.1, 0.2];
    let ds_bar = vec![0.3, 0.4, 0.5];

    let adjoint = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Smoothed,
        mu,
    );

    let lhs: f64 = forward
        .dx
        .iter()
        .zip(dx_bar.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + forward
            .dz
            .iter()
            .zip(dz_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
        + forward
            .ds
            .iter()
            .zip(ds_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    let rhs: f64 = dq
        .iter()
        .zip(adjoint.dq.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + db.iter()
            .zip(adjoint.db.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    assert!(
        (lhs - rhs).abs() < 1e-4,
        "LP Smoothed forward-adjoint: LHS = {}, RHS = {}, diff = {}",
        lhs,
        rhs,
        (lhs - rhs).abs()
    );
}

#[test]
fn test_qp_nonneg_forward_adjoint_consistency_smoothed() {
    // QP with nonneg constraints: verify forward-adjoint identity
    // using Smoothed differentiation.
    let n = 2;
    let m = 2;

    let P = identity_csc(n);
    let q = vec![-1.0, -1.0];
    let A = identity_csc(n);
    let b = vec![0.0, 0.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(m)];

    // Solution: x = [1, 1], s = [1, 1], z = [0, 0] (inactive constraints)
    // Actually for min 0.5||x||^2 - x  s.t. x >= 0, solution is x=[1,1], z=0
    let x = vec![1.0, 1.0];
    let s = vec![1.0, 1.0];
    let z = vec![0.001, 0.001]; // slightly positive dual (near-inactive)
    let tau = 1.0;
    let mu = 1e-4;

    let dP = zero_csc(n, n);
    let dq = vec![0.2, 0.3];
    let dA = zero_csc(m, n);
    let db = vec![0.4, 0.5];

    let forward = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Smoothed,
        mu,
    );

    let dx_bar = vec![0.6, 0.7];
    let dz_bar = vec![0.8, 0.9];
    let ds_bar = vec![0.1, 0.2];

    let adjoint = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Smoothed,
        mu,
    );

    let lhs: f64 = forward
        .dx
        .iter()
        .zip(dx_bar.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + forward
            .dz
            .iter()
            .zip(dz_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
        + forward
            .ds
            .iter()
            .zip(ds_bar.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    let rhs: f64 = dq
        .iter()
        .zip(adjoint.dq.iter())
        .map(|(a, b)| a * b)
        .sum::<f64>()
        + db.iter()
            .zip(adjoint.db.iter())
            .map(|(a, b)| a * b)
            .sum::<f64>();

    assert!(
        (lhs - rhs).abs() < 1e-4,
        "QP nonneg Smoothed forward-adjoint: LHS = {}, RHS = {}",
        lhs,
        rhs
    );
}

#[test]
fn test_diff_method_auto_resolves_correctly() {
    // DiffMethod::Auto should resolve to Exact when mu < 1e-6,
    // and Smoothed when mu >= 1e-6.
    let n = 2;
    let m = 2;

    let P = identity_csc(n);
    let q = vec![-1.0, -1.0];
    let A = identity_csc(n);
    let b = vec![0.0, 0.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(m)];

    let x = vec![1.0, 1.0];
    let s = vec![1.0, 1.0];
    let z = vec![0.5, 0.5];
    let tau = 1.0;

    let dx_bar = vec![1.0, 0.0];
    let dz_bar = vec![0.0, 0.0];
    let ds_bar = vec![0.0, 0.0];

    // Auto with small mu should match Exact
    let auto_small = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Auto,
        1e-7,
    );
    let exact = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Exact,
        1e-7,
    );

    for i in 0..n {
        assert!(
            (auto_small.dq[i] - exact.dq[i]).abs() < 1e-12,
            "Auto(small mu) != Exact: dq[{}]",
            i
        );
    }

    // Auto with larger mu should match Smoothed
    let auto_large = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Auto,
        1e-4,
    );
    let smoothed = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Smoothed,
        1e-4,
    );

    for i in 0..n {
        assert!(
            (auto_large.dq[i] - smoothed.dq[i]).abs() < 1e-12,
            "Auto(large mu) != Smoothed: dq[{}]",
            i
        );
    }
}

// ========================================================================
// Smoothed differentiation edge cases
// ========================================================================

#[test]
#[should_panic(expected = "Smoothed differentiation not yet implemented")]
fn test_smoothed_exp_cone_panics() {
    let n = 3;
    let m = 3;
    let P = identity_csc(n);
    let q = vec![0.0; n];
    let A = identity_csc(n);
    let b = vec![0.0, 1.0, 3.0];
    let cones = vec![SupportedConeT::ExponentialConeT()];

    let x = vec![0.0; n];
    let s = vec![0.0, 1.0, 3.0];
    let z = vec![1.0, 1.0, 1.0];
    let tau = 1.0;

    let dP = zero_csc(n, n);
    let dq = vec![0.1; n];
    let dA = zero_csc(m, n);
    let db = vec![0.0; m];

    differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Smoothed,
        1e-6,
    );
}

#[test]
#[should_panic(expected = "Smoothed differentiation not yet implemented")]
fn test_smoothed_power_cone_panics() {
    let n = 3;
    let m = 3;
    let P = identity_csc(n);
    let q = vec![0.0; n];
    let A = identity_csc(n);
    let b = vec![4.0, 4.0, 1.0];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let x = vec![0.0; n];
    let s = vec![4.0, 4.0, 1.0];
    let z = vec![0.5, 0.5, 0.5];
    let tau = 1.0;

    let dP = zero_csc(n, n);
    let dq = vec![0.1; n];
    let dA = zero_csc(m, n);
    let db = vec![0.0; m];

    differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Smoothed,
        1e-6,
    );
}

#[test]
fn test_smoothed_nonneg_z_at_zero() {
    // When z[i] = 0 (active constraint boundary), smoothed H should be ~0
    // (full projection), not NaN.
    let n = 2;
    let m = 2;

    let P = identity_csc(n);
    let q = vec![-1.0, -1.0];
    let A = identity_csc(n);
    let b = vec![0.0, 0.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(m)];

    let x = vec![1.0, 1.0];
    let s = vec![1.0, 1.0];
    let z = vec![0.0, 1e-15]; // z at/near zero = active constraint
    let tau = 1.0;
    let mu = 1e-4;

    let dx_bar = vec![1.0, 0.0];
    let dz_bar = vec![0.0, 0.0];
    let ds_bar = vec![0.0, 0.0];

    let result = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Smoothed,
        mu,
    );

    // Must produce finite, non-NaN gradients
    for i in 0..n {
        assert!(
            result.dq[i].is_finite(),
            "dq[{}] is not finite: {}",
            i,
            result.dq[i]
        );
    }
    for i in 0..m {
        assert!(
            result.db[i].is_finite(),
            "db[{}] is not finite: {}",
            i,
            result.db[i]
        );
    }
}

#[test]
fn test_smoothed_large_mu_increases_smoothing() {
    // Larger μ should produce H values closer to 0 for nonneg cone
    // (more smoothing = less sharp). Test via forward-adjoint: larger μ
    // should change the gradients compared to small μ.
    let n = 2;
    let m = 2;

    let P = identity_csc(n);
    let q = vec![-1.0, -1.0];
    let A = identity_csc(n);
    let b = vec![0.0, 0.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(m)];

    let x = vec![1.0, 1.0];
    let s = vec![1.0, 1.0];
    let z = vec![0.1, 0.1]; // small z: smoothing effect is more visible
    let tau = 1.0;

    let dx_bar = vec![1.0, 0.0];
    let dz_bar = vec![0.0, 0.0];
    let ds_bar = vec![0.0, 0.0];

    let small = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Smoothed,
        1e-6,
    );
    let large = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Smoothed,
        1.0,
    );

    // With small z=0.1, H[i] = 0.01/(0.01+μ). For μ=1e-6, H≈1. For μ=1, H≈0.01.
    // So the gradients must differ noticeably.
    let diff: f64 = small
        .dq
        .iter()
        .zip(large.dq.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-3,
        "Large and small mu should produce different gradients, diff = {:.2e}",
        diff
    );
}

#[test]
fn test_smoothed_negative_mu_clamped() {
    // Negative μ should be clamped to epsilon internally, producing valid output
    let n = 2;
    let m = 2;

    let P = identity_csc(n);
    let q = vec![-1.0, -1.0];
    let A = identity_csc(n);
    let b = vec![0.0, 0.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(m)];

    let x = vec![1.0, 1.0];
    let s = vec![1.0, 1.0];
    let z = vec![0.5, 0.5];
    let tau = 1.0;

    let dx_bar = vec![1.0, 0.0];
    let dz_bar = vec![0.0, 0.0];
    let ds_bar = vec![0.0, 0.0];

    let result = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Smoothed,
        -1.0,
    );

    for i in 0..n {
        assert!(
            result.dq[i].is_finite(),
            "Negative mu: dq[{}] not finite",
            i
        );
    }
}

#[test]
fn test_smoothed_zero_cones_bypasses_hsde() {
    // All-zero-cone problems should produce identical results regardless of
    // DiffMethod, since they bypass the HSDE path entirely.
    let P = identity_csc(2);
    let q = vec![0.0, 0.0];
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b = vec![2.0];
    let cones = vec![SupportedConeT::ZeroConeT(1)];

    let x = vec![1.0, 1.0];
    let s = vec![0.0];
    let z = vec![0.0];
    let tau = 1.0;

    let dx_bar = vec![1.0, 0.0];
    let dz_bar = vec![0.0];
    let ds_bar = vec![0.0];

    let exact = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Exact,
        0.0,
    );
    let smoothed = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Smoothed,
        1.0,
    );
    let auto = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Auto,
        1.0,
    );

    for i in 0..2 {
        assert!(
            (exact.dq[i] - smoothed.dq[i]).abs() < 1e-12,
            "Zero-cone: Exact vs Smoothed dq[{}]",
            i
        );
        assert!(
            (exact.dq[i] - auto.dq[i]).abs() < 1e-12,
            "Zero-cone: Exact vs Auto dq[{}]",
            i
        );
    }
}

#[test]
fn test_smoothed_mixed_zero_nonneg_forward() {
    // Smoothed forward differentiation with mixed zero+nonneg cones.
    // Verifies the forward path (not just adjoint) works.
    let n = 2;
    let m = 3;

    let P = identity_csc(n);
    let q = vec![1.0, 2.0];
    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 2],
        nzval: vec![1.0, 1.0, 1.0, 1.0],
    };
    let b = vec![1.0, 0.0, 0.0];
    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
    ];

    let x = vec![0.5, 0.5];
    let s = vec![0.0, 0.5, 0.5];
    let z = vec![-1.5, 0.5, 0.5];
    let tau = 1.0;
    let mu = 1e-4;

    let dP = zero_csc(n, n);
    let dq = vec![0.3, 0.4];
    let dA = zero_csc(m, n);
    let db = vec![0.5, 0.6, 0.7];

    let result = differentiate(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Smoothed,
        mu,
    );

    assert_eq!(result.dx.len(), n);
    assert_eq!(result.dz.len(), m);
    assert_eq!(result.ds.len(), m);

    // All values must be finite
    for i in 0..n {
        assert!(result.dx[i].is_finite(), "dx[{}] not finite", i);
    }
    for i in 0..m {
        assert!(result.dz[i].is_finite(), "dz[{}] not finite", i);
        assert!(result.ds[i].is_finite(), "ds[{}] not finite", i);
    }
}

#[test]
fn test_supports_smoothed_empty() {
    let cones: Vec<SupportedConeT<f64>> = vec![];
    assert!(supports_smoothed(&cones));
}

#[test]
fn test_supports_smoothed_zero_only() {
    let cones: Vec<SupportedConeT<f64>> = vec![SupportedConeT::ZeroConeT(5)];
    assert!(supports_smoothed(&cones));
}

#[test]
fn test_supports_smoothed_nonneg_only() {
    let cones: Vec<SupportedConeT<f64>> = vec![SupportedConeT::NonnegativeConeT(3)];
    assert!(supports_smoothed(&cones));
}

#[test]
fn test_supports_smoothed_mixed_zero_nonneg() {
    let cones: Vec<SupportedConeT<f64>> = vec![
        SupportedConeT::ZeroConeT(2),
        SupportedConeT::NonnegativeConeT(3),
    ];
    assert!(supports_smoothed(&cones));
}

#[test]
fn test_supports_smoothed_with_soc() {
    let cones: Vec<SupportedConeT<f64>> = vec![
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::SecondOrderConeT(3),
    ];
    assert!(supports_smoothed(&cones));
}

#[test]
fn test_supports_smoothed_rejects_exp() {
    let cones: Vec<SupportedConeT<f64>> = vec![SupportedConeT::ExponentialConeT()];
    assert!(!supports_smoothed(&cones));
}

#[test]
fn test_supports_smoothed_rejects_power() {
    let cones: Vec<SupportedConeT<f64>> = vec![SupportedConeT::PowerConeT(0.5)];
    assert!(!supports_smoothed(&cones));
}

#[test]
fn test_smoothed_adjoint_works_for_mixed_with_soc() {
    // Smoothed with mixed cones including SOC should succeed.
    let n = 2;
    let m = 5; // 1 zero + 1 nonneg + 3 SOC

    let P = identity_csc(n);
    let q = vec![0.0; n];
    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 3, 5],
        rowval: vec![0, 1, 2, 3, 4],
        nzval: vec![1.0, 1.0, 1.0, 1.0, 1.0],
    };
    let b = vec![0.0, 1.0, 2.0, 0.5, 0.5];
    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(1),
        SupportedConeT::SecondOrderConeT(3),
    ];

    let x = vec![0.0; n];
    let s = vec![0.0, 1.0, 2.0, 0.5, 0.5];
    let z = vec![0.1; m];
    let tau = 1.0;

    let dx_bar = vec![1.0, 0.0];
    let dz_bar = vec![0.0; m];
    let ds_bar = vec![0.0; m];

    // Should succeed now that SOC is supported
    let result = differentiate_adjoint(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Smoothed,
        1e-4,
    );
    // Check that we get finite results
    assert!(
        result.dq.iter().all(|v| v.is_finite()),
        "dq should be finite"
    );
    assert!(
        result.db.iter().all(|v| v.is_finite()),
        "db should be finite"
    );
}
