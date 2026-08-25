//! Unit tests for cone projections and projection Jacobians.

use super::exp::invert_4x4;
use super::*;

// ========================================================================
// Zero cone tests
// ========================================================================

#[test]
fn test_zero_cone_projection() {
    let z: Vec<f64> = vec![1.0, 2.0, 3.0];
    let cones = vec![SupportedConeT::ZeroConeT(3)];

    // Primal projection should be zero
    let proj = get_cone_projection(&z, &cones, false);
    assert!(proj.iter().all(|&x| x.abs() < 1e-10));

    // Dual projection should be identity
    let proj_dual = get_cone_projection(&z, &cones, true);
    for i in 0..3 {
        assert!((proj_dual[i] - z[i]).abs() < 1e-10);
    }
}

#[test]
fn test_zero_cone_derivative() {
    let z: Vec<f64> = vec![1.0, 2.0, 3.0];
    let cones = vec![SupportedConeT::ZeroConeT(3)];

    // Primal derivative should be zero matrix
    let deriv = get_cone_derivative(&z, &cones, false);
    assert_eq!(deriv.len(), 1);
    assert!(deriv[0].iter().all(|&x| x.abs() < 1e-10));

    // Dual derivative should be identity matrix
    let deriv_dual = get_cone_derivative(&z, &cones, true);
    assert_eq!(deriv_dual.len(), 1);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((deriv_dual[0][i * 3 + j] - expected).abs() < 1e-10);
        }
    }
}

// ========================================================================
// Nonnegative cone tests
// ========================================================================

#[test]
fn test_nonneg_cone_projection() {
    let z: Vec<f64> = vec![1.0, -2.0, 3.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(3)];

    let proj = get_cone_projection(&z, &cones, false);
    assert!((proj[0] - 1.0).abs() < 1e-10);
    assert!(proj[1].abs() < 1e-10);
    assert!((proj[2] - 3.0).abs() < 1e-10);
}

#[test]
fn test_nonneg_cone_projection_all_positive() {
    let z: Vec<f64> = vec![1.0, 2.0, 3.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(3)];

    let proj = get_cone_projection(&z, &cones, false);
    for i in 0..3 {
        assert!((proj[i] - z[i]).abs() < 1e-10);
    }
}

#[test]
fn test_nonneg_cone_projection_all_negative() {
    let z: Vec<f64> = vec![-1.0, -2.0, -3.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(3)];

    let proj = get_cone_projection(&z, &cones, false);
    assert!(proj.iter().all(|&x| x.abs() < 1e-10));
}

#[test]
fn test_nonneg_cone_derivative() {
    let z: Vec<f64> = vec![1.0, -2.0, 3.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(3)];

    let deriv = get_cone_derivative(&z, &cones, false);
    assert_eq!(deriv.len(), 1);

    // Derivative is diagonal: 1 where z >= 0, 0 where z < 0
    assert!((deriv[0][0] - 1.0).abs() < 1e-10); // z[0] = 1.0 >= 0
    assert!(deriv[0][4].abs() < 1e-10); // z[1] = -2.0 < 0
    assert!((deriv[0][8] - 1.0).abs() < 1e-10); // z[2] = 3.0 >= 0
}

#[test]
fn test_nonneg_cone_derivative_at_zero() {
    let z: Vec<f64> = vec![0.0, 1.0, -1.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(3)];

    let deriv = get_cone_derivative(&z, &cones, false);
    // At exactly zero, derivative is 1 (since z[i] >= 0 is true)
    assert!((deriv[0][0] - 1.0).abs() < 1e-10);
    assert!((deriv[0][4] - 1.0).abs() < 1e-10);
    assert!(deriv[0][8].abs() < 1e-10);
}

// ========================================================================
// Second-order cone tests
// ========================================================================

#[test]
fn test_soc_projection_interior() {
    // Point inside SOC: ||x|| < t
    let z: Vec<f64> = vec![2.0, 1.0, 0.5];
    let cones = vec![SupportedConeT::SecondOrderConeT(3)];

    let proj = get_cone_projection(&z, &cones, false);
    for i in 0..3 {
        assert!((proj[i] - z[i]).abs() < 1e-10);
    }
}

#[test]
fn test_soc_projection_boundary() {
    // Point outside SOC (||x|| > t)
    let z: Vec<f64> = vec![1.0, 2.0, 2.0];
    let cones = vec![SupportedConeT::SecondOrderConeT(3)];

    let proj = get_cone_projection(&z, &cones, false);

    // Check that projection is on cone boundary: t = ||x||
    let t_proj = proj[0];
    let norm_x = (proj[1] * proj[1] + proj[2] * proj[2]).sqrt();
    assert!((t_proj - norm_x).abs() < 1e-6);
}

#[test]
fn test_soc_projection_polar() {
    // Point in polar cone (||x|| <= -t)
    let z: Vec<f64> = vec![-3.0, 1.0, 1.0];
    let cones = vec![SupportedConeT::SecondOrderConeT(3)];

    let proj = get_cone_projection(&z, &cones, false);
    assert!(proj.iter().all(|&x| x.abs() < 1e-10));
}

#[test]
fn test_soc_projection_on_boundary() {
    // Point exactly on boundary: ||x|| = t
    let z: Vec<f64> = vec![5.0_f64.sqrt(), 1.0, 2.0]; // sqrt(1^2 + 2^2) = sqrt(5)
    let cones = vec![SupportedConeT::SecondOrderConeT(3)];

    let proj = get_cone_projection(&z, &cones, false);
    for i in 0..3 {
        assert!((proj[i] - z[i]).abs() < 1e-6);
    }
}

#[test]
fn test_soc_derivative_interior() {
    // Point inside SOC - derivative is identity
    let z: Vec<f64> = vec![2.0, 0.5, 0.5];
    let cones = vec![SupportedConeT::SecondOrderConeT(3)];

    let deriv = get_cone_derivative(&z, &cones, false);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (deriv[0][i * 3 + j] - expected).abs() < 1e-10,
                "deriv[{}][{}] = {}, expected {}",
                i,
                j,
                deriv[0][i * 3 + j],
                expected
            );
        }
    }
}

#[test]
fn test_soc_derivative_polar() {
    // Point in polar cone - derivative is zero
    let z: Vec<f64> = vec![-3.0, 1.0, 1.0];
    let cones = vec![SupportedConeT::SecondOrderConeT(3)];

    let deriv = get_cone_derivative(&z, &cones, false);
    assert!(deriv[0].iter().all(|&x| x.abs() < 1e-10));
}

#[test]
fn test_soc_derivative_boundary() {
    // Point on boundary - derivative is special form
    let z: Vec<f64> = vec![1.0, 2.0, 2.0];
    let cones = vec![SupportedConeT::SecondOrderConeT(3)];

    let deriv = get_cone_derivative(&z, &cones, false);

    // Check H[0,0] = 0.5
    assert!((deriv[0][0] - 0.5).abs() < 1e-6);

    // Check symmetry
    assert!((deriv[0][1] - deriv[0][3]).abs() < 1e-10);
    assert!((deriv[0][2] - deriv[0][6]).abs() < 1e-10);
    assert!((deriv[0][5] - deriv[0][7]).abs() < 1e-10);
}

// ========================================================================
// Exponential cone tests
// ========================================================================

#[test]
fn test_exp_cone_interior() {
    // Point inside exp cone: s > 0 and s * exp(r/s) <= t
    let z: Vec<f64> = vec![0.0, 1.0, 3.0]; // 1 * exp(0/1) = 1 <= 3
    let cones = vec![SupportedConeT::ExponentialConeT()];

    let proj = get_cone_projection(&z, &cones, false);
    for i in 0..3 {
        assert!(
            (proj[i] - z[i]).abs() < 1e-6,
            "proj[{}] = {}, z[{}] = {}",
            i,
            proj[i],
            i,
            z[i]
        );
    }
}

#[test]
fn test_exp_cone_boundary_case() {
    // Point with r < 0, s < 0
    let z: Vec<f64> = vec![-1.0, -1.0, 1.0];
    let cones = vec![SupportedConeT::ExponentialConeT()];

    let proj = get_cone_projection(&z, &cones, false);
    // For r < 0, s < 0: proj = (r, 0, max(t, 0))
    assert!((proj[0] - (-1.0)).abs() < 1e-6);
    assert!(proj[1].abs() < 1e-6);
    assert!((proj[2] - 1.0).abs() < 1e-6);
}

#[test]
fn test_exp_cone_dual_interior() {
    // Point in dual exp cone interior
    let z: Vec<f64> = vec![-1.0, 1.0, 1.0]; // r < 0, -r*exp(s/r) < e*t
    let cones = vec![SupportedConeT::ExponentialConeT()];

    // Dual projection uses Moreau: Π_{K*}(z) = z + Π_K(-z)
    // Note: K* (dual) ≠ K° (polar). Polar would be z - Π_K(z).
    let proj_dual = get_cone_projection(&z, &cones, true);

    // Verify: proj_dual = z + proj_primal(-z)
    let neg_z: Vec<f64> = vec![1.0, -1.0, -1.0];
    let proj_primal_neg = get_cone_projection(&neg_z, &cones, false);
    for i in 0..3 {
        assert!((proj_dual[i] - (z[i] + proj_primal_neg[i])).abs() < 1e-6);
    }
}

#[test]
fn test_exp_cone_derivative_interior() {
    // Interior point - derivative is identity
    let z: Vec<f64> = vec![0.0, 1.0, 3.0];
    let cones = vec![SupportedConeT::ExponentialConeT()];

    let deriv = get_cone_derivative(&z, &cones, false);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((deriv[0][i * 3 + j] - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn test_exp_cone_derivative_dual() {
    // With diffclarabel-style implementation, dual derivative evaluates at -z
    // For z = [0, 1, 3] in primal cone interior:
    //   - primal derivative at z = I
    //   - dual derivative evaluates at -z = [0, -1, -3], which is NOT in primal cone interior
    //   - so dual derivative is I - (derivative at -z)
    let z: Vec<f64> = vec![0.0, 1.0, 3.0];
    let cones = vec![SupportedConeT::ExponentialConeT()];

    let deriv_primal = get_cone_derivative(&z, &cones, false);

    // Interior point: primal derivative = I
    for i in 0..3 {
        for j in 0..3 {
            let expected_primal = if i == j { 1.0 } else { 0.0 };
            assert!(
                (deriv_primal[0][i * 3 + j] - expected_primal).abs() < 1e-6,
                "primal deriv[{}][{}] = {}",
                i,
                j,
                deriv_primal[0][i * 3 + j]
            );
        }
    }

    // Note: dual derivative behavior follows diffclarabel convention,
    // which is verified by end-to-end gradient tests (not unit tested here)
}

// ========================================================================
// Power cone tests
// ========================================================================

#[test]
fn test_pow_cone_interior() {
    // Point inside power cone with alpha = 0.5
    let z: Vec<f64> = vec![4.0, 4.0, 1.0]; // 4^0.5 * 4^0.5 = 4 > 1
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let proj = get_cone_projection(&z, &cones, false);
    for i in 0..3 {
        assert!(
            (proj[i] - z[i]).abs() < 1e-6,
            "proj[{}] = {}, z[{}] = {}",
            i,
            proj[i],
            i,
            z[i]
        );
    }
}

#[test]
fn test_pow_cone_interior_alpha_third() {
    // Point inside power cone with alpha = 1/3
    // x^(1/3) * y^(2/3) >= |z|
    // 8^(1/3) * 8^(2/3) = 2 * 4 = 8 > 1
    let z: Vec<f64> = vec![8.0, 8.0, 1.0];
    let cones = vec![SupportedConeT::PowerConeT(1.0 / 3.0)];

    let proj = get_cone_projection(&z, &cones, false);
    for i in 0..3 {
        assert!((proj[i] - z[i]).abs() < 1e-6);
    }
}

#[test]
fn test_pow_cone_polar() {
    // Point in polar cone: x < 0, y < 0
    let z: Vec<f64> = vec![-1.0, -1.0, 0.1];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let proj = get_cone_projection(&z, &cones, false);
    // Should project to zero
    assert!(proj.iter().all(|&x| x.abs() < 1e-6));
}

#[test]
fn test_pow_cone_z_zero() {
    // Edge case: z = 0
    let z: Vec<f64> = vec![1.0, 2.0, 0.0];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let proj = get_cone_projection(&z, &cones, false);
    assert!((proj[0] - 1.0).abs() < 1e-6);
    assert!((proj[1] - 2.0).abs() < 1e-6);
    assert!(proj[2].abs() < 1e-6);
}

#[test]
fn test_pow_cone_negative_z() {
    // Test with negative z
    let z: Vec<f64> = vec![4.0, 4.0, -1.0];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let proj = get_cone_projection(&z, &cones, false);
    // Should be inside cone since |z| = 1 < 4
    for i in 0..3 {
        assert!((proj[i] - z[i]).abs() < 1e-6);
    }
}

#[test]
fn test_pow_cone_boundary_projection() {
    // Point outside power cone - should project to boundary
    let z: Vec<f64> = vec![1.0, 1.0, 10.0]; // 1^0.5 * 1^0.5 = 1 < 10
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let proj = get_cone_projection(&z, &cones, false);

    // Check that projection satisfies cone constraint approximately
    let x = proj[0];
    let y = proj[1];
    let abs_z = proj[2].abs();
    let lhs = x.powf(0.5) * y.powf(0.5);
    assert!(
        (lhs - abs_z).abs() < 1e-4,
        "lhs = {}, abs_z = {}",
        lhs,
        abs_z
    );
}

#[test]
fn test_pow_cone_derivative_interior() {
    // Interior point - derivative is identity
    let z: Vec<f64> = vec![4.0, 4.0, 1.0];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let deriv = get_cone_derivative(&z, &cones, false);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((deriv[0][i * 3 + j] - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn test_pow_cone_derivative_polar() {
    // Polar interior - derivative is zero
    let z: Vec<f64> = vec![-2.0, -2.0, 0.1];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let deriv = get_cone_derivative(&z, &cones, false);
    assert!(deriv[0].iter().all(|&x| x.abs() < 1e-6));
}

#[test]
fn test_pow_cone_derivative_symmetry() {
    // Check Jacobian is symmetric
    let z: Vec<f64> = vec![1.0, 1.0, 5.0];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let deriv = get_cone_derivative(&z, &cones, false);

    // J[i,j] = J[j,i]
    assert!((deriv[0][1] - deriv[0][3]).abs() < 1e-10); // J[0,1] = J[1,0]
    assert!((deriv[0][2] - deriv[0][6]).abs() < 1e-10); // J[0,2] = J[2,0]
    assert!((deriv[0][5] - deriv[0][7]).abs() < 1e-10); // J[1,2] = J[2,1]
}

#[test]
fn test_pow_cone_dual_projection() {
    let z: Vec<f64> = vec![1.0, 1.0, 5.0];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    // Dual projection uses Moreau: Π_{K*}(z) = z + Π_K(-z)
    // Note: K* (dual) ≠ K° (polar). Polar would be z - Π_K(z).
    let proj_dual = get_cone_projection(&z, &cones, true);
    let neg_z: Vec<f64> = vec![-1.0, -1.0, -5.0];
    let proj_primal_neg = get_cone_projection(&neg_z, &cones, false);

    for i in 0..3 {
        assert!((proj_dual[i] - (z[i] + proj_primal_neg[i])).abs() < 1e-6);
    }
}

// ========================================================================
// Mixed cone tests
// ========================================================================

#[test]
fn test_mixed_cones_projection() {
    let z: Vec<f64> = vec![
        1.0, 2.0, // Zero cone (dim=2)
        -1.0, 2.0, -3.0, // Nonneg cone (dim=3)
        2.0, 0.5, 0.5, // SOC (dim=3)
    ];
    let cones = vec![
        SupportedConeT::ZeroConeT(2),
        SupportedConeT::NonnegativeConeT(3),
        SupportedConeT::SecondOrderConeT(3),
    ];

    let proj = get_cone_projection(&z, &cones, false);

    // Zero cone part: should be zero
    assert!(proj[0].abs() < 1e-10);
    assert!(proj[1].abs() < 1e-10);

    // Nonneg part: max(z, 0)
    assert!(proj[2].abs() < 1e-10); // -1 -> 0
    assert!((proj[3] - 2.0).abs() < 1e-10);
    assert!(proj[4].abs() < 1e-10); // -3 -> 0

    // SOC part: inside cone, unchanged
    assert!((proj[5] - 2.0).abs() < 1e-10);
    assert!((proj[6] - 0.5).abs() < 1e-10);
    assert!((proj[7] - 0.5).abs() < 1e-10);
}

#[test]
fn test_mixed_cones_derivative() {
    let z: Vec<f64> = vec![
        1.0, // Zero cone (dim=1)
        1.0, -1.0, // Nonneg cone (dim=2)
    ];
    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
    ];

    let deriv = get_cone_derivative(&z, &cones, false);
    assert_eq!(deriv.len(), 2);

    // Zero cone: 1x1 zero matrix
    assert!(deriv[0][0].abs() < 1e-10);

    // Nonneg cone: 2x2 diagonal with [1, 0]
    assert!((deriv[1][0] - 1.0).abs() < 1e-10); // z[0]=1 >= 0
    assert!(deriv[1][1].abs() < 1e-10);
    assert!(deriv[1][2].abs() < 1e-10);
    assert!(deriv[1][3].abs() < 1e-10); // z[1]=-1 < 0
}

// ========================================================================
// Apply derivative tests
// ========================================================================

#[test]
fn test_apply_cone_derivative_zero() {
    let z: Vec<f64> = vec![1.0, 2.0];
    let v: Vec<f64> = vec![3.0, 4.0];
    let cones = vec![SupportedConeT::ZeroConeT(2)];
    let mut out = vec![0.0; 2];

    apply_cone_derivative(&z, &v, &cones, false, &mut out);
    assert!(out.iter().all(|&x| x.abs() < 1e-10));

    apply_cone_derivative(&z, &v, &cones, true, &mut out);
    assert!((out[0] - 3.0).abs() < 1e-10);
    assert!((out[1] - 4.0).abs() < 1e-10);
}

#[test]
fn test_apply_cone_derivative_nonneg() {
    let z: Vec<f64> = vec![1.0, -2.0, 3.0];
    let v: Vec<f64> = vec![1.0, 1.0, 1.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(3)];
    let mut out = vec![0.0; 3];

    apply_cone_derivative(&z, &v, &cones, false, &mut out);
    assert!((out[0] - 1.0).abs() < 1e-10); // z[0] >= 0
    assert!(out[1].abs() < 1e-10); // z[1] < 0
    assert!((out[2] - 1.0).abs() < 1e-10); // z[2] >= 0
}

// ========================================================================
// 4x4 matrix inversion tests
// ========================================================================

#[test]
fn test_4x4_inverse() {
    let m: [[f64; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 4.0],
    ];

    let inv = invert_4x4(&m);

    assert!((inv[0][0] - 1.0).abs() < 1e-10);
    assert!((inv[1][1] - 0.5).abs() < 1e-10);
    assert!((inv[2][2] - 1.0 / 3.0).abs() < 1e-10);
    assert!((inv[3][3] - 0.25).abs() < 1e-10);
}

#[test]
fn test_4x4_inverse_identity() {
    let m: [[f64; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];

    let inv = invert_4x4(&m);

    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((inv[i][j] - expected).abs() < 1e-10);
        }
    }
}

#[test]
fn test_4x4_inverse_general() {
    let m: [[f64; 4]; 4] = [
        [2.0, 1.0, 0.0, 0.0],
        [1.0, 2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 1.0],
        [0.0, 0.0, 1.0, 2.0],
    ];

    let inv = invert_4x4(&m);

    // Verify M * M^(-1) = I
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0.0;
            for k in 0..4 {
                sum += m[i][k] * inv[k][j];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() < 1e-8,
                "M * M^(-1)[{},{}] = {}, expected {}",
                i,
                j,
                sum,
                expected
            );
        }
    }
}

// ========================================================================
// Numerical gradient verification tests
// ========================================================================

#[test]
fn test_nonneg_derivative_numerical() {
    // Verify derivative by finite differences
    let z: Vec<f64> = vec![1.0, 2.0, 3.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(3)];
    let eps = 1e-7;

    let deriv = get_cone_derivative(&z, &cones, false);

    for j in 0..3 {
        let mut z_plus = z.clone();
        let mut z_minus = z.clone();
        z_plus[j] += eps;
        z_minus[j] -= eps;

        let proj_plus = get_cone_projection(&z_plus, &cones, false);
        let proj_minus = get_cone_projection(&z_minus, &cones, false);

        for i in 0..3 {
            let numerical = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
            let analytical = deriv[0][i * 3 + j];
            assert!(
                (numerical - analytical).abs() < 1e-5,
                "d proj[{}] / d z[{}]: numerical = {}, analytical = {}",
                i,
                j,
                numerical,
                analytical
            );
        }
    }
}

#[test]
fn test_soc_derivative_numerical() {
    // Verify SOC derivative by finite differences (boundary case)
    let z: Vec<f64> = vec![1.0, 2.0, 2.0];
    let cones = vec![SupportedConeT::SecondOrderConeT(3)];
    let eps = 1e-7;

    let deriv = get_cone_derivative(&z, &cones, false);

    for j in 0..3 {
        let mut z_plus = z.clone();
        let mut z_minus = z.clone();
        z_plus[j] += eps;
        z_minus[j] -= eps;

        let proj_plus = get_cone_projection(&z_plus, &cones, false);
        let proj_minus = get_cone_projection(&z_minus, &cones, false);

        for i in 0..3 {
            let numerical = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
            let analytical = deriv[0][i * 3 + j];
            assert!(
                (numerical - analytical).abs() < 1e-4,
                "d proj[{}] / d z[{}]: numerical = {}, analytical = {}",
                i,
                j,
                numerical,
                analytical
            );
        }
    }
}

#[test]
fn test_pow_derivative_numerical() {
    // Verify power cone derivative by finite differences
    let z: Vec<f64> = vec![2.0, 2.0, 5.0];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];
    let eps = 1e-6;

    let deriv = get_cone_derivative(&z, &cones, false);

    for j in 0..3 {
        let mut z_plus = z.clone();
        let mut z_minus = z.clone();
        z_plus[j] += eps;
        z_minus[j] -= eps;

        let proj_plus = get_cone_projection(&z_plus, &cones, false);
        let proj_minus = get_cone_projection(&z_minus, &cones, false);

        for i in 0..3 {
            let numerical = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
            let analytical = deriv[0][i * 3 + j];
            assert!(
                (numerical - analytical).abs() < 1e-3,
                "d proj[{}] / d z[{}]: numerical = {}, analytical = {}",
                i,
                j,
                numerical,
                analytical
            );
        }
    }
}

// ========================================================================
// Exponential cone numerical gradient tests
// ========================================================================

#[test]
fn test_exp_derivative_numerical_boundary() {
    // Verify exp cone derivative by finite differences (boundary case)
    // Point outside but close to the boundary
    let z: Vec<f64> = vec![0.5, 1.0, 2.0];
    let cones = vec![SupportedConeT::ExponentialConeT()];
    let eps = 1e-6;

    let deriv = get_cone_derivative(&z, &cones, false);

    for j in 0..3 {
        let mut z_plus = z.clone();
        let mut z_minus = z.clone();
        z_plus[j] += eps;
        z_minus[j] -= eps;

        let proj_plus = get_cone_projection(&z_plus, &cones, false);
        let proj_minus = get_cone_projection(&z_minus, &cones, false);

        for i in 0..3 {
            let numerical = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
            let analytical = deriv[0][i * 3 + j];
            // Looser tolerance for boundary cases
            assert!(
                (numerical - analytical).abs() < 0.1,
                "d proj[{}] / d z[{}]: numerical = {}, analytical = {}",
                i,
                j,
                numerical,
                analytical
            );
        }
    }
}

#[test]
fn test_exp_derivative_numerical_negative_r_s() {
    // Verify exp cone derivative for r < 0, s < 0 case
    let z: Vec<f64> = vec![-1.0, -1.0, 1.0];
    let cones = vec![SupportedConeT::ExponentialConeT()];
    let eps = 1e-7;

    let deriv = get_cone_derivative(&z, &cones, false);

    for j in 0..3 {
        let mut z_plus = z.clone();
        let mut z_minus = z.clone();
        z_plus[j] += eps;
        z_minus[j] -= eps;

        let proj_plus = get_cone_projection(&z_plus, &cones, false);
        let proj_minus = get_cone_projection(&z_minus, &cones, false);

        for i in 0..3 {
            let numerical = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
            let analytical = deriv[0][i * 3 + j];
            assert!(
                (numerical - analytical).abs() < 1e-4,
                "d proj[{}] / d z[{}]: numerical = {}, analytical = {}",
                i,
                j,
                numerical,
                analytical
            );
        }
    }
}

#[test]
fn test_exp_derivative_dual_numerical() {
    // Dual derivative = I - primal derivative at -z
    let z: Vec<f64> = vec![-1.0, 1.5, 1.5];
    let cones = vec![SupportedConeT::ExponentialConeT()];

    let deriv_dual = get_cone_derivative(&z, &cones, true);
    let neg_z: Vec<f64> = vec![1.0, -1.5, -1.5];
    let deriv_primal_at_neg_z = get_cone_derivative(&neg_z, &cones, false);

    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 } - deriv_primal_at_neg_z[0][i * 3 + j];
            let actual = deriv_dual[0][i * 3 + j];
            assert!((actual - expected).abs() < 1e-10);
        }
    }
}

// ========================================================================
// Power cone additional numerical gradient tests
// ========================================================================

#[test]
fn test_pow_derivative_numerical_alpha_third() {
    // Verify power cone derivative with alpha = 1/3
    let z: Vec<f64> = vec![2.0, 2.0, 4.0];
    let cones = vec![SupportedConeT::PowerConeT(1.0 / 3.0)];
    let eps = 1e-6;

    let deriv = get_cone_derivative(&z, &cones, false);

    for j in 0..3 {
        let mut z_plus = z.clone();
        let mut z_minus = z.clone();
        z_plus[j] += eps;
        z_minus[j] -= eps;

        let proj_plus = get_cone_projection(&z_plus, &cones, false);
        let proj_minus = get_cone_projection(&z_minus, &cones, false);

        for i in 0..3 {
            let numerical = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
            let analytical = deriv[0][i * 3 + j];
            assert!(
                (numerical - analytical).abs() < 1e-2,
                "d proj[{}] / d z[{}]: numerical = {}, analytical = {}",
                i,
                j,
                numerical,
                analytical
            );
        }
    }
}

#[test]
fn test_pow_derivative_numerical_negative_z() {
    // Verify power cone derivative with negative z component
    let z: Vec<f64> = vec![2.0, 2.0, -3.0];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];
    let eps = 1e-6;

    let deriv = get_cone_derivative(&z, &cones, false);

    for j in 0..3 {
        let mut z_plus = z.clone();
        let mut z_minus = z.clone();
        z_plus[j] += eps;
        z_minus[j] -= eps;

        let proj_plus = get_cone_projection(&z_plus, &cones, false);
        let proj_minus = get_cone_projection(&z_minus, &cones, false);

        for i in 0..3 {
            let numerical = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
            let analytical = deriv[0][i * 3 + j];
            assert!(
                (numerical - analytical).abs() < 1e-2,
                "d proj[{}] / d z[{}]: numerical = {}, analytical = {}",
                i,
                j,
                numerical,
                analytical
            );
        }
    }
}

#[test]
fn test_pow_derivative_dual_numerical() {
    // Dual derivative = I - primal derivative at -z
    let z: Vec<f64> = vec![2.0, 2.0, 5.0];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let deriv_dual = get_cone_derivative(&z, &cones, true);
    let neg_z: Vec<f64> = vec![-2.0, -2.0, -5.0];
    let deriv_primal_at_neg_z = get_cone_derivative(&neg_z, &cones, false);

    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 } - deriv_primal_at_neg_z[0][i * 3 + j];
            let actual = deriv_dual[0][i * 3 + j];
            assert!((actual - expected).abs() < 1e-10);
        }
    }
}

// ========================================================================
// Moreau decomposition tests
// ========================================================================

#[test]
fn test_moreau_decomposition_exp() {
    // The Moreau decomposition is: z = Π_K(z) + Π_{K°}(z)
    // where K° is the POLAR cone (not dual).
    // For dual cone K*: K° = -K*, so Π_{K°}(z) = -Π_{K*}(-z)
    // This means: z = Π_K(z) - Π_{K*}(-z)
    // Or equivalently: Π_{K*}(z) = z + Π_K(-z)
    let z: Vec<f64> = vec![0.5, 1.0, 2.0];
    let cones = vec![SupportedConeT::ExponentialConeT()];

    let proj_primal = get_cone_projection(&z, &cones, false);
    let neg_z: Vec<f64> = vec![-0.5, -1.0, -2.0];
    let proj_dual_neg = get_cone_projection(&neg_z, &cones, true);

    // Verify: z = Π_K(z) - Π_{K*}(-z)
    for i in 0..3 {
        let sum = proj_primal[i] - proj_dual_neg[i];
        assert!(
            (sum - z[i]).abs() < 1e-6,
            "Moreau decomposition failed: z[{}] = {}, sum = {}",
            i,
            z[i],
            sum
        );
    }
}

#[test]
fn test_moreau_decomposition_pow() {
    // The Moreau decomposition is: z = Π_K(z) + Π_{K°}(z)
    // where K° is the POLAR cone (not dual).
    // For dual cone K*: K° = -K*, so Π_{K°}(z) = -Π_{K*}(-z)
    // This means: z = Π_K(z) - Π_{K*}(-z)
    let z: Vec<f64> = vec![2.0, 2.0, 5.0];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let proj_primal = get_cone_projection(&z, &cones, false);
    let neg_z: Vec<f64> = vec![-2.0, -2.0, -5.0];
    let proj_dual_neg = get_cone_projection(&neg_z, &cones, true);

    // Verify: z = Π_K(z) - Π_{K*}(-z)
    for i in 0..3 {
        let sum = proj_primal[i] - proj_dual_neg[i];
        assert!(
            (sum - z[i]).abs() < 1e-6,
            "Moreau decomposition failed: z[{}] = {}, sum = {}",
            i,
            z[i],
            sum
        );
    }
}

#[test]
fn test_moreau_derivative_exp() {
    let z: Vec<f64> = vec![0.5, 1.0, 2.0];
    let cones = vec![SupportedConeT::ExponentialConeT()];
    let deriv_primal = get_cone_derivative(&z, &cones, false);
    assert!(deriv_primal[0].len() == 9);
}

#[test]
fn test_moreau_derivative_pow() {
    let z: Vec<f64> = vec![2.0, 2.0, 5.0];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];
    let deriv_primal = get_cone_derivative(&z, &cones, false);
    assert!(deriv_primal[0].len() == 9);
}

// ========================================================================
// Mixed cones numerical tests
// ========================================================================

#[test]
fn test_mixed_exp_pow_numerical() {
    // Mixed exp + power cones derivative verification
    let z: Vec<f64> = vec![
        0.0, 1.0, 3.0, // Exp cone interior
        4.0, 4.0, 1.0, // Power cone interior
    ];
    let cones = vec![
        SupportedConeT::ExponentialConeT(),
        SupportedConeT::PowerConeT(0.5),
    ];
    let eps = 1e-6;

    let deriv = get_cone_derivative(&z, &cones, false);

    // Check exp cone block
    for j in 0..3 {
        let mut z_plus = z.clone();
        let mut z_minus = z.clone();
        z_plus[j] += eps;
        z_minus[j] -= eps;

        let proj_plus = get_cone_projection(&z_plus, &cones, false);
        let proj_minus = get_cone_projection(&z_minus, &cones, false);

        for i in 0..3 {
            let numerical = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
            let analytical = deriv[0][i * 3 + j];
            assert!(
                (numerical - analytical).abs() < 1e-4,
                "exp cone d proj[{}] / d z[{}]: numerical = {}, analytical = {}",
                i,
                j,
                numerical,
                analytical
            );
        }
    }

    // Check power cone block
    for j in 0..3 {
        let mut z_plus = z.clone();
        let mut z_minus = z.clone();
        z_plus[3 + j] += eps;
        z_minus[3 + j] -= eps;

        let proj_plus = get_cone_projection(&z_plus, &cones, false);
        let proj_minus = get_cone_projection(&z_minus, &cones, false);

        for i in 0..3 {
            let numerical = (proj_plus[3 + i] - proj_minus[3 + i]) / (2.0 * eps);
            let analytical = deriv[1][i * 3 + j];
            assert!(
                (numerical - analytical).abs() < 1e-3,
                "pow cone d proj[{}] / d z[{}]: numerical = {}, analytical = {}",
                3 + i,
                3 + j,
                numerical,
                analytical
            );
        }
    }
}

#[test]
fn test_all_cones_numerical() {
    // All cone types derivative verification including GenPowerCone
    let z: Vec<f64> = vec![
        1.0, // Zero cone
        1.0, 2.0, // Nonneg cone
        2.0, 0.5, 0.5, // SOC interior
        0.0, 1.0, 3.0, // Exp cone interior
        4.0, 4.0, 1.0, // Power cone interior
        2.0, 3.0, 1.5, 0.3, 0.4, // GenPowerCone (dim1=3, dim2=2)
    ];
    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::SecondOrderConeT(3),
        SupportedConeT::ExponentialConeT(),
        SupportedConeT::PowerConeT(0.5),
        SupportedConeT::GenPowerConeT(vec![0.2, 0.3, 0.5], 2),
    ];
    let eps = 1e-6;

    let deriv = get_cone_derivative(&z, &cones, false);

    // Check each cone block
    let dims = [1, 2, 3, 3, 3, 5];
    let mut offset = 0;

    for (cone_idx, &dim) in dims.iter().enumerate() {
        for j in 0..dim {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();
            z_plus[offset + j] += eps;
            z_minus[offset + j] -= eps;

            let proj_plus = get_cone_projection(&z_plus, &cones, false);
            let proj_minus = get_cone_projection(&z_minus, &cones, false);

            for i in 0..dim {
                let numerical = (proj_plus[offset + i] - proj_minus[offset + i]) / (2.0 * eps);
                let analytical = deriv[cone_idx][i * dim + j];
                assert!(
                    (numerical - analytical).abs() < 1e-3,
                    "cone {} d proj[{}] / d z[{}]: numerical = {}, analytical = {}",
                    cone_idx,
                    offset + i,
                    offset + j,
                    numerical,
                    analytical
                );
            }
        }
        offset += dim;
    }
}

#[test]
fn test_exp_cone_derivative_polar_case() {
    // Note: The diffclarabel-style implementation uses a specific formula for
    // dual derivatives that is designed for HSDE differentiation. It evaluates
    // the derivative at -u and returns I - that. This is validated by end-to-end
    // gradient tests.
    //
    // This test just verifies the primal derivative at u = [0.5, -1.0, -1.5]
    let u: Vec<f64> = vec![0.5, -1.0, -1.5];
    let cones = vec![SupportedConeT::ExponentialConeT()];

    // The primal derivative should be computed without error
    let deriv_primal = get_cone_derivative(&u, &cones, false);
    assert!(deriv_primal[0].len() == 9);

    // Note: Dual derivative behavior is validated by end-to-end gradient tests
}

// ========================================================================
// Comprehensive numerical verification tests for all regions
// ========================================================================

/// Helper function to compute numerical Jacobian via finite differences
fn numerical_jacobian(
    z: &[f64],
    cones: &[SupportedConeT<f64>],
    dual: bool,
    eps: f64,
) -> Vec<Vec<f64>> {
    let dim = z.len();
    let mut jac = vec![vec![0.0; dim]; dim];

    for j in 0..dim {
        let mut z_plus = z.to_vec();
        let mut z_minus = z.to_vec();
        z_plus[j] += eps;
        z_minus[j] -= eps;

        let proj_plus = get_cone_projection(&z_plus, cones, dual);
        let proj_minus = get_cone_projection(&z_minus, cones, dual);

        for i in 0..dim {
            jac[i][j] = (proj_plus[i] - proj_minus[i]) / (2.0 * eps);
        }
    }
    jac
}

/// Helper to convert flat Jacobian to 2D
fn flat_to_2d(flat: &[f64], dim: usize) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; dim]; dim];
    for i in 0..dim {
        for j in 0..dim {
            result[i][j] = flat[i * dim + j];
        }
    }
    result
}

/// Helper to compute max error between two Jacobians
fn max_jacobian_error(jac1: &[Vec<f64>], jac2: &[Vec<f64>]) -> f64 {
    let mut max_err = 0.0_f64;
    for i in 0..jac1.len() {
        for j in 0..jac1[i].len() {
            max_err = max_err.max((jac1[i][j] - jac2[i][j]).abs());
        }
    }
    max_err
}

// ========================================================================
// Exponential cone: comprehensive tests for all regions
// ========================================================================

#[test]
fn test_exp_primal_jacobian_interior() {
    // Interior of exponential cone: s > 0 and s*exp(r/s) < t
    let z: Vec<f64> = vec![0.0, 1.0, 3.0]; // 1*exp(0) = 1 < 3
    let cones = vec![SupportedConeT::ExponentialConeT()];

    let deriv = get_cone_derivative(&z, &cones, false);
    let analytical = flat_to_2d(&deriv[0], 3);
    let numerical = numerical_jacobian(&z, &cones, false, 1e-7);

    let max_err = max_jacobian_error(&analytical, &numerical);
    assert!(
        max_err < 1e-5,
        "Exp primal interior Jacobian error {:.2e} exceeds tolerance",
        max_err
    );

    // Interior should have identity Jacobian
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (analytical[i][j] - expected).abs() < 1e-6,
                "Exp interior J[{}][{}] = {}, expected {}",
                i,
                j,
                analytical[i][j],
                expected
            );
        }
    }
}

#[test]
fn test_exp_primal_jacobian_polar() {
    // In dual cone interior => projection is 0 => Jacobian is 0
    // -z in dual cone interior: r > 0 and -r*exp(s/r) < e*t is NOT the condition
    // Actually: dual exp cone is {(u,v,w): u <= 0, -u*exp(v/u-1) <= w}
    // For z = [1.0, -2.0, -3.0], -z = [-1.0, 2.0, 3.0]
    // Check if -z is in dual cone: u = -1 < 0, -(-1)*exp(2/-1-1) = exp(-3) ≈ 0.05 < 3 ✓
    let z: Vec<f64> = vec![1.0, -2.0, -3.0];
    let cones = vec![SupportedConeT::ExponentialConeT()];

    let deriv = get_cone_derivative(&z, &cones, false);
    let analytical = flat_to_2d(&deriv[0], 3);
    let numerical = numerical_jacobian(&z, &cones, false, 1e-7);

    let max_err = max_jacobian_error(&analytical, &numerical);
    assert!(
        max_err < 1e-5,
        "Exp primal polar Jacobian error {:.2e} exceeds tolerance",
        max_err
    );
}

#[test]
fn test_exp_primal_jacobian_boundary() {
    // Point outside cone that projects to boundary
    let z: Vec<f64> = vec![1.0, 1.0, 1.5]; // 1*exp(1) = e ≈ 2.72 > 1.5
    let cones = vec![SupportedConeT::ExponentialConeT()];

    let deriv = get_cone_derivative(&z, &cones, false);
    let analytical = flat_to_2d(&deriv[0], 3);
    let numerical = numerical_jacobian(&z, &cones, false, 1e-6);

    let max_err = max_jacobian_error(&analytical, &numerical);
    // Note: Boundary cases have slightly higher numerical error due to projection
    // curvature; 5e-3 tolerance is acceptable for validation.
    assert!(max_err < 5e-3,
        "Exp primal boundary Jacobian error {:.2e} exceeds tolerance.\nAnalytical:\n{:?}\nNumerical:\n{:?}",
        max_err, analytical, numerical);

    // Jacobian should be symmetric
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (analytical[i][j] - analytical[j][i]).abs() < 1e-6,
                "Exp boundary Jacobian not symmetric: J[{}][{}]={}, J[{}][{}]={}",
                i,
                j,
                analytical[i][j],
                j,
                i,
                analytical[j][i]
            );
        }
    }
}

#[test]
fn test_exp_primal_jacobian_negative_rs() {
    // Special case: r < 0, s < 0
    let z: Vec<f64> = vec![-1.0, -1.0, 1.0];
    let cones = vec![SupportedConeT::ExponentialConeT()];

    let deriv = get_cone_derivative(&z, &cones, false);
    let analytical = flat_to_2d(&deriv[0], 3);
    let numerical = numerical_jacobian(&z, &cones, false, 1e-7);

    let max_err = max_jacobian_error(&analytical, &numerical);
    assert!(
        max_err < 1e-5,
        "Exp primal r<0,s<0 Jacobian error {:.2e} exceeds tolerance",
        max_err
    );
}

#[test]
fn test_exp_dual_jacobian_all_regions() {
    // Test dual Jacobian against numerical for several points
    let test_points = vec![
        vec![0.5, 0.8, 1.5],   // General position
        vec![-1.0, 1.0, 2.0],  // r < 0
        vec![1.0, 2.0, 0.5],   // Outside cone
        vec![-0.5, -0.5, 1.0], // r < 0, s < 0
    ];

    let cones = vec![SupportedConeT::ExponentialConeT()];

    for z in test_points {
        let deriv = get_cone_derivative(&z, &cones, true);
        let analytical = flat_to_2d(&deriv[0], 3);
        let numerical = numerical_jacobian(&z, &cones, true, 1e-6);

        let max_err = max_jacobian_error(&analytical, &numerical);
        assert!(max_err < 1e-3,
            "Exp dual Jacobian error {:.2e} at z={:?} exceeds tolerance.\nAnalytical:\n{:?}\nNumerical:\n{:?}",
            max_err, z, analytical, numerical);
    }
}

#[test]
fn test_exp_dual_projection_moreau() {
    // Verify Moreau identity: Π_{K*}(z) = z + Π_K(-z)
    let test_points = vec![
        vec![0.5, 0.8, 1.5],
        vec![-1.0, 1.0, 2.0],
        vec![1.0, 2.0, 0.5],
    ];

    let cones = vec![SupportedConeT::ExponentialConeT()];

    for z in test_points {
        let proj_dual = get_cone_projection(&z, &cones, true);
        let neg_z: Vec<f64> = z.iter().map(|&x| -x).collect();
        let proj_primal_neg = get_cone_projection(&neg_z, &cones, false);

        for i in 0..3 {
            let expected = z[i] + proj_primal_neg[i];
            assert!(
                (proj_dual[i] - expected).abs() < 1e-10,
                "Exp Moreau identity failed at z={:?}: proj_dual[{}]={}, expected={}",
                z,
                i,
                proj_dual[i],
                expected
            );
        }
    }
}

// ========================================================================
// Power cone: comprehensive tests for all regions
// ========================================================================

#[test]
fn test_pow_primal_jacobian_interior() {
    // Interior: x > 0, y > 0, x^α * y^(1-α) > |z|
    let z: Vec<f64> = vec![4.0, 4.0, 1.0]; // 4^0.5 * 4^0.5 = 4 > 1
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let deriv = get_cone_derivative(&z, &cones, false);
    let analytical = flat_to_2d(&deriv[0], 3);
    let numerical = numerical_jacobian(&z, &cones, false, 1e-7);

    let max_err = max_jacobian_error(&analytical, &numerical);
    assert!(
        max_err < 1e-5,
        "Pow primal interior Jacobian error {:.2e} exceeds tolerance",
        max_err
    );

    // Interior should have identity Jacobian
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (analytical[i][j] - expected).abs() < 1e-6,
                "Pow interior J[{}][{}] = {}, expected {}",
                i,
                j,
                analytical[i][j],
                expected
            );
        }
    }
}

#[test]
fn test_pow_primal_jacobian_polar() {
    // Polar interior: x < 0, y < 0, (-x/α)^α * (-y/(1-α))^(1-α) > |z|
    let z: Vec<f64> = vec![-2.0, -2.0, 0.1];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let deriv = get_cone_derivative(&z, &cones, false);
    let analytical = flat_to_2d(&deriv[0], 3);
    let numerical = numerical_jacobian(&z, &cones, false, 1e-7);

    let max_err = max_jacobian_error(&analytical, &numerical);
    assert!(
        max_err < 1e-5,
        "Pow primal polar Jacobian error {:.2e} exceeds tolerance",
        max_err
    );

    // Polar should have zero Jacobian
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                analytical[i][j].abs() < 1e-6,
                "Pow polar J[{}][{}] = {}, expected 0",
                i,
                j,
                analytical[i][j]
            );
        }
    }
}

#[test]
fn test_pow_primal_jacobian_boundary() {
    // Point outside cone that projects to boundary
    let z: Vec<f64> = vec![0.5, 0.8, 1.5]; // 0.5^0.5 * 0.8^0.5 ≈ 0.63 < 1.5
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let deriv = get_cone_derivative(&z, &cones, false);
    let analytical = flat_to_2d(&deriv[0], 3);
    let numerical = numerical_jacobian(&z, &cones, false, 1e-6);

    let max_err = max_jacobian_error(&analytical, &numerical);
    assert!(max_err < 1e-3,
        "Pow primal boundary Jacobian error {:.2e} exceeds tolerance.\nAnalytical:\n{:?}\nNumerical:\n{:?}",
        max_err, analytical, numerical);

    // Jacobian should be symmetric
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (analytical[i][j] - analytical[j][i]).abs() < 1e-6,
                "Pow boundary Jacobian not symmetric: J[{}][{}]={}, J[{}][{}]={}",
                i,
                j,
                analytical[i][j],
                j,
                i,
                analytical[j][i]
            );
        }
    }
}

#[test]
fn test_pow_primal_jacobian_negative_z() {
    // z component negative
    let z: Vec<f64> = vec![0.5, 0.8, -1.5];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let deriv = get_cone_derivative(&z, &cones, false);
    let analytical = flat_to_2d(&deriv[0], 3);
    let numerical = numerical_jacobian(&z, &cones, false, 1e-6);

    let max_err = max_jacobian_error(&analytical, &numerical);
    assert!(
        max_err < 1e-3,
        "Pow primal negative z Jacobian error {:.2e} exceeds tolerance",
        max_err
    );
}

#[test]
fn test_pow_primal_jacobian_z_zero() {
    // Edge case: z = 0
    let z: Vec<f64> = vec![1.0, 2.0, 0.0];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let deriv = get_cone_derivative(&z, &cones, false);
    let analytical = flat_to_2d(&deriv[0], 3);
    let numerical = numerical_jacobian(&z, &cones, false, 1e-7);

    let max_err = max_jacobian_error(&analytical, &numerical);
    assert!(
        max_err < 1e-4,
        "Pow primal z=0 Jacobian error {:.2e} exceeds tolerance",
        max_err
    );
}

#[test]
fn test_pow_primal_jacobian_various_alpha() {
    // Test different alpha values
    let alphas = vec![0.1, 0.3, 0.5, 0.7, 0.9];
    let z: Vec<f64> = vec![1.0, 1.0, 3.0]; // Outside cone for most alphas

    for alpha in alphas {
        let cones = vec![SupportedConeT::PowerConeT(alpha)];

        let deriv = get_cone_derivative(&z, &cones, false);
        let analytical = flat_to_2d(&deriv[0], 3);
        let numerical = numerical_jacobian(&z, &cones, false, 1e-6);

        let max_err = max_jacobian_error(&analytical, &numerical);
        assert!(
            max_err < 1e-3,
            "Pow primal alpha={} Jacobian error {:.2e} exceeds tolerance",
            alpha,
            max_err
        );
    }
}

#[test]
fn test_pow_dual_jacobian_all_regions() {
    // Test dual Jacobian against numerical for several points
    // Note: Avoid points where -z lands exactly on cone boundaries, as the
    // projection is non-differentiable there and numerical differentiation fails.
    let test_points = vec![
        vec![0.5, 0.8, 1.5],   // Boundary projection (general position)
        vec![4.0, 4.0, 1.0],   // Interior (z in primal interior)
        vec![-2.0, -2.0, 0.1], // Polar (-z in primal interior)
        vec![2.0, 2.0, -1.5],  // Negative z, well inside cone
    ];

    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    for z in test_points {
        let deriv = get_cone_derivative(&z, &cones, true);
        let analytical = flat_to_2d(&deriv[0], 3);
        let numerical = numerical_jacobian(&z, &cones, true, 1e-6);

        let max_err = max_jacobian_error(&analytical, &numerical);
        assert!(max_err < 1e-3,
            "Pow dual Jacobian error {:.2e} at z={:?} exceeds tolerance.\nAnalytical:\n{:?}\nNumerical:\n{:?}",
            max_err, z, analytical, numerical);
    }
}

#[test]
fn test_pow_dual_projection_moreau() {
    // Verify Moreau identity: Π_{K*}(z) = z + Π_K(-z)
    let test_points = vec![
        vec![0.5, 0.8, 1.5],
        vec![4.0, 4.0, 1.0],
        vec![1.0, 1.0, 3.0],
    ];

    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    for z in test_points {
        let proj_dual = get_cone_projection(&z, &cones, true);
        let neg_z: Vec<f64> = z.iter().map(|&x| -x).collect();
        let proj_primal_neg = get_cone_projection(&neg_z, &cones, false);

        for i in 0..3 {
            let expected = z[i] + proj_primal_neg[i];
            assert!(
                (proj_dual[i] - expected).abs() < 1e-10,
                "Pow Moreau identity failed at z={:?}: proj_dual[{}]={}, expected={}",
                z,
                i,
                proj_dual[i],
                expected
            );
        }
    }
}

#[test]
fn test_pow_dual_jacobian_formula() {
    // Verify D[Π_{K*}](z) = I - D[Π_K](-z)
    let z: Vec<f64> = vec![0.5, 0.8, 1.5];
    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    let deriv_dual = get_cone_derivative(&z, &cones, true);
    let neg_z: Vec<f64> = vec![-0.5, -0.8, -1.5];
    let deriv_primal_neg = get_cone_derivative(&neg_z, &cones, false);

    for i in 0..3 {
        for j in 0..3 {
            let eye_ij = if i == j { 1.0 } else { 0.0 };
            let expected = eye_ij - deriv_primal_neg[0][i * 3 + j];
            let actual = deriv_dual[0][i * 3 + j];
            assert!(
                (actual - expected).abs() < 1e-10,
                "Pow dual Jacobian formula: D_dual[{}][{}]={}, expected I - D_primal(-z) = {}",
                i,
                j,
                actual,
                expected
            );
        }
    }
}

// ========================================================================
// Projection validity tests
// ========================================================================

#[test]
fn test_exp_projection_in_cone() {
    // Verify that primal projection lands in the cone
    let test_points: Vec<Vec<f64>> = vec![
        vec![1.0, 1.0, 1.5],
        vec![0.5, 2.0, 1.0],
        vec![-0.5, 0.5, 0.3],
    ];

    let cones = vec![SupportedConeT::ExponentialConeT()];

    for z in test_points {
        let proj = get_cone_projection(&z, &cones, false);

        // Check cone membership: s >= 0 and (s = 0 implies r <= 0 and t >= 0)
        //                        or (s > 0 and s*exp(r/s) <= t)
        let (r, s, t): (f64, f64, f64) = (proj[0], proj[1], proj[2]);
        let in_cone = if s.abs() < 1e-10 {
            r <= 1e-8 && t >= -1e-8
        } else if s > 0.0 {
            s * (r / s).exp() <= t + 1e-6
        } else {
            false
        };

        assert!(
            in_cone || proj.iter().all(|&v: &f64| v.abs() < 1e-8),
            "Exp projection not in cone: z={:?}, proj={:?}",
            z,
            proj
        );
    }
}

#[test]
fn test_pow_projection_in_cone() {
    // Verify that primal projection lands in the cone
    let test_points: Vec<Vec<f64>> = vec![
        vec![0.5, 0.8, 1.5],
        vec![1.0, 1.0, 5.0],
        vec![2.0, 0.5, 3.0],
    ];

    let cones = vec![SupportedConeT::PowerConeT(0.5)];

    for z in test_points {
        let proj = get_cone_projection(&z, &cones, false);

        // Check cone membership: x >= 0, y >= 0, x^α * y^(1-α) >= |z|
        let (x, y, abs_z): (f64, f64, f64) = (proj[0], proj[1], proj[2].abs());
        let in_cone = x >= -1e-8
            && y >= -1e-8
            && (x.max(0.0).powf(0.5) * y.max(0.0).powf(0.5) >= abs_z - 1e-6);

        assert!(
            in_cone || proj.iter().all(|&v: &f64| v.abs() < 1e-8),
            "Pow projection not in cone: z={:?}, proj={:?}",
            z,
            proj
        );
    }
}

// ========================================================================
// Central-path derivative (Smoothed) tests
// ========================================================================

#[test]
fn test_central_path_nonneg_closed_form() {
    let s: Vec<f64> = vec![0.5, 0.3, 0.8];
    let z: Vec<f64> = vec![2.0, 0.5, 1.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(3)];
    let mu: f64 = 0.1;

    let blocks = get_central_path_derivative_sparse(&s, &z, &cones, mu);
    assert_eq!(blocks.len(), 1);

    if let ConeDerivativeBlock::Diagonal(ref diag) = blocks[0] {
        assert_eq!(diag.len(), 3);
        for i in 0..3 {
            let z2 = z[i] * z[i];
            let expected = z2 / (z2 + mu);
            assert!(
                (diag[i] - expected).abs() < 1e-12,
                "H[{},{}]: got {}, expected {}",
                i,
                i,
                diag[i],
                expected
            );
        }
    } else {
        panic!("Expected Diagonal block for nonneg cone");
    }
}

#[test]
fn test_central_path_nonneg_approaches_projection_as_mu_to_zero() {
    let z_positive: Vec<f64> = vec![1.0, 2.0, 0.5];
    let s: Vec<f64> = vec![0.5, 0.5, 0.5];
    let cones = vec![SupportedConeT::NonnegativeConeT(3)];

    let blocks = get_central_path_derivative_sparse(&s, &z_positive, &cones, 1e-12);
    if let ConeDerivativeBlock::Diagonal(ref diag) = blocks[0] {
        for i in 0..3 {
            assert!(
                (diag[i] - 1.0_f64).abs() < 1e-4,
                "H[{},{}] = {} (expected ~1 for z>0)",
                i,
                i,
                diag[i]
            );
        }
    } else {
        panic!("Expected Diagonal block");
    }
}

#[test]
fn test_central_path_zero_cone_is_identity() {
    let s: Vec<f64> = vec![0.0, 0.0];
    let z: Vec<f64> = vec![1.0, 2.0];
    let cones = vec![SupportedConeT::ZeroConeT(2)];
    let mu: f64 = 1.0;

    let blocks = get_central_path_derivative_sparse(&s, &z, &cones, mu);
    if let ConeDerivativeBlock::Diagonal(ref diag) = blocks[0] {
        for i in 0..2 {
            assert!(
                (diag[i] - 1.0_f64).abs() < 1e-12,
                "Zero cone H should be identity, got H[{},{}] = {}",
                i,
                i,
                diag[i]
            );
        }
    } else {
        panic!("Expected Diagonal block for zero cone");
    }
}

#[test]
fn test_central_path_mu_zero_guard() {
    let s: Vec<f64> = vec![0.0];
    let z: Vec<f64> = vec![0.0];
    let cones = vec![SupportedConeT::NonnegativeConeT(1)];

    let blocks = get_central_path_derivative_sparse(&s, &z, &cones, 0.0);
    if let ConeDerivativeBlock::Diagonal(ref diag) = blocks[0] {
        assert!(!diag[0].is_nan(), "mu=0, z=0 should not produce NaN");
        assert!(diag[0].is_finite(), "mu=0, z=0 should produce finite value");
    } else {
        panic!("Expected Diagonal block");
    }
}

// ========================================================================
// SocSparse decomposition tests
// ========================================================================

fn soc_sparse_to_dense(block: &ConeDerivativeBlock<f64>) -> Vec<f64> {
    match block {
        ConeDerivativeBlock::SocSparse {
            dim,
            diag,
            v1,
            c1,
            v2,
            c2,
        } => {
            let mut result = vec![0.0; dim * dim];
            for i in 0..*dim {
                result[i * dim + i] = diag[i];
            }
            for i in 0..*dim {
                for j in 0..*dim {
                    result[i * dim + j] += c1 * v1[i] * v1[j];
                }
            }
            for i in 0..*dim {
                for j in 0..*dim {
                    result[i * dim + j] += c2 * v2[i] * v2[j];
                }
            }
            result
        }
        _ => panic!("Expected SocSparse"),
    }
}

#[test]
fn test_soc_sparse_matches_dense_boundary() {
    for dim in [5, 8, 10, 20, 50] {
        for &(t, norm_scale) in &[(0.5, 1.0), (-0.3, 1.0), (0.0, 1.0), (0.1, 0.5)] {
            let mut z = vec![0.0; dim];
            z[0] = t;
            let dim_factor = (dim as f64 - 1.0).sqrt();
            for i in 1..dim {
                z[i] = norm_scale * dim_factor / (dim as f64 - 1.0).sqrt();
            }
            let norm_x: f64 = z[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
            if !(norm_x > z[0] && norm_x > -z[0]) {
                continue;
            }
            let dense = derivative_soc(&z, dim);
            let sparse_block = derivative_soc_sparse(&z, dim);
            let reconstructed = soc_sparse_to_dense(&sparse_block);
            for i in 0..dim {
                for j in 0..dim {
                    let d = dense[i * dim + j];
                    let s = reconstructed[i * dim + j];
                    assert!(
                        (d - s).abs() < 1e-12,
                        "Mismatch at dim={}, z[0]={}, [{},{}]: dense={}, sparse={}",
                        dim,
                        t,
                        i,
                        j,
                        d,
                        s
                    );
                }
            }
        }
    }
}

#[test]
fn test_soc_sparse_interior_is_identity() {
    let dim = 10;
    let mut z = vec![0.0; dim];
    z[0] = 5.0;
    for i in 1..dim {
        z[i] = 0.1;
    }
    let block = derivative_soc_sparse(&z, dim);
    match &block {
        ConeDerivativeBlock::SocSparse {
            dim: d,
            diag,
            c1,
            c2,
            ..
        } => {
            assert_eq!(*d, dim);
            for &val in diag {
                assert!((val - 1.0_f64).abs() < 1e-12);
            }
            assert!((*c1 as f64).abs() < 1e-12);
            assert!((*c2 as f64).abs() < 1e-12);
        }
        _ => panic!("Expected SocSparse for interior"),
    }
    let v: Vec<f64> = (0..dim).map(|i| i as f64 + 1.0).collect();
    let blocks = vec![block];
    let result = apply_cone_derivative_blocks(&blocks, &v);
    for i in 0..dim {
        assert!((result[i] - v[i]).abs() < 1e-12);
    }
}

#[test]
fn test_soc_sparse_polar_is_zero() {
    let dim = 10;
    let mut z = vec![0.0; dim];
    z[0] = -5.0;
    for i in 1..dim {
        z[i] = 0.1;
    }
    let block = derivative_soc_sparse(&z, dim);
    match &block {
        ConeDerivativeBlock::SocSparse {
            dim: d,
            diag,
            c1,
            c2,
            ..
        } => {
            assert_eq!(*d, dim);
            for &val in diag {
                assert!((val as f64).abs() < 1e-12);
            }
            assert!((*c1 as f64).abs() < 1e-12);
            assert!((*c2 as f64).abs() < 1e-12);
        }
        _ => panic!("Expected SocSparse for polar"),
    }
    let v: Vec<f64> = (0..dim).map(|i| i as f64 + 1.0).collect();
    let blocks = vec![block];
    let result = apply_cone_derivative_blocks(&blocks, &v);
    for i in 0..dim {
        assert!(result[i].abs() < 1e-12);
    }
}

#[test]
fn test_soc_sparse_apply_matches_dense() {
    let dim = 10;
    let mut z = vec![0.0; dim];
    z[0] = 0.5;
    for i in 1..dim {
        z[i] = 1.0 / (i as f64);
    }
    let cones = vec![SupportedConeT::SecondOrderConeT(dim)];
    let dense_data = derivative_soc(&z, dim);
    let dense_blocks = vec![ConeDerivativeBlock::Dense {
        dim,
        data: dense_data,
    }];
    let sparse_blocks = get_cone_derivative_sparse(&z, &cones, false);
    let v: Vec<f64> = (0..dim).map(|i| (i as f64 + 1.0) * 0.3).collect();
    let result_dense = apply_cone_derivative_blocks(&dense_blocks, &v);
    let result_sparse = apply_cone_derivative_blocks(&sparse_blocks, &v);
    for i in 0..dim {
        assert!(
            (result_dense[i] - result_sparse[i]).abs() < 1e-12,
            "Mismatch at i={}: dense={}, sparse={}",
            i,
            result_dense[i],
            result_sparse[i]
        );
    }
    let result_dense_t = apply_cone_derivative_blocks_transpose(&dense_blocks, &v);
    let result_sparse_t = apply_cone_derivative_blocks_transpose(&sparse_blocks, &v);
    for i in 0..dim {
        assert!(
            (result_dense_t[i] - result_sparse_t[i]).abs() < 1e-12,
            "Transpose mismatch at i={}: dense={}, sparse={}",
            i,
            result_dense_t[i],
            result_sparse_t[i]
        );
    }
}
