//! Exponential cone projection and projection-Jacobian kernels.

use crate::algebra::{AsFloatT, FloatT};
use crate::solver::implementations::default::diff::CONE_TOL;

const EXP_MAX_ITERS: usize = 200;

// ============================================================================
// Exponential cone
// ============================================================================

fn in_exp<T: FloatT>(x: &[T; 3]) -> bool {
    let (r, s, t) = (x[0], x[1], x[2]);
    let tol: T = CONE_TOL.as_T();

    (r <= T::zero() && s.abs() <= tol && t >= T::zero())
        || (s > T::zero() && s * (r / s).exp() - t <= tol)
}

fn in_exp_dual<T: FloatT>(x: &[T; 3]) -> bool {
    let (r, s, t) = (x[0], x[1], x[2]);
    let tol: T = CONE_TOL.as_T();
    let e: T = std::f64::consts::E.as_T();

    // Dual exp cone: {(r, s, t) : r <= 0 and -r * exp(s/r - 1) <= e * t}
    // Special case: r = 0 requires s >= 0 and t >= 0
    (r.abs() <= tol && s >= T::zero() && t >= T::zero())
        || (r < T::zero() && -r * (s / r - T::one()).exp() - e * t <= tol)
}

fn in_exp_interior<T: FloatT>(x: &[T; 3], margin: T) -> bool {
    let (r, s, t) = (x[0], x[1], x[2]);
    s > margin && s * (r / s).exp() + margin < t
}

fn in_exp_dual_interior<T: FloatT>(x: &[T; 3], margin: T) -> bool {
    let (r, s, t) = (x[0], x[1], x[2]);
    let e: T = std::f64::consts::E.as_T();
    // Dual exp cone: r < 0 and -r * exp(s/r - 1) <= e * t
    // Interior: strict inequality with margin
    r < -margin && -r * (s / r - T::one()).exp() + margin < e * t
}

fn exp_newton_one_d<T: FloatT>(rho: T, y_hat: T, z_hat: T) -> T {
    let mut t = T::max(-z_hat, (1e-6).as_T());
    let tol: T = CONE_TOL.as_T();

    for _ in 0..EXP_MAX_ITERS {
        let f = t * (t + z_hat) / rho / rho - y_hat / rho + (t / rho).ln() + T::one();
        let fp = ((t + t) + z_hat) / rho / rho + T::one() / t;
        t -= f / fp;

        if t <= -z_hat {
            return T::zero();
        } else if t <= T::zero() {
            return z_hat;
        } else if f.abs() < tol {
            break;
        }
    }
    t + z_hat
}

fn exp_solve_for_x_with_rho<T: FloatT>(v: &[T; 3], rho: T) -> [T; 3] {
    let x2 = exp_newton_one_d(rho, v[1], v[2]);
    let x1 = (x2 - v[2]) * x2 / rho;
    let x0 = v[0] - rho;
    [x0, x1, x2]
}

fn exp_calc_grad<T: FloatT>(v: &[T; 3], rho: T) -> T {
    let x = exp_solve_for_x_with_rho(v, rho);
    let small: T = (1e-12).as_T();
    if x[1] <= small {
        x[0]
    } else {
        x[0] + x[1] * (x[1] / x[2]).ln()
    }
}

fn exp_get_rho_ub<T: FloatT>(v: &[T; 3]) -> (T, T) {
    let mut lb = T::zero();
    let mut ub: T = (0.125).as_T();

    while exp_calc_grad(v, ub) > T::zero() {
        lb = ub;
        ub *= (2.0).as_T();
    }
    (lb, ub)
}

pub(super) fn project_exp_cone_primal<T: FloatT>(v: &[T; 3], out: &mut [T]) {
    let (r, s, t) = (v[0], v[1], v[2]);

    if in_exp(v) {
        out[0] = v[0];
        out[1] = v[1];
        out[2] = v[2];
        return;
    }

    let neg_v = [-v[0], -v[1], -v[2]];
    if in_exp_dual(&neg_v) {
        out[0] = T::zero();
        out[1] = T::zero();
        out[2] = T::zero();
        return;
    }

    if r < T::zero() && s < T::zero() {
        out[0] = r;
        out[1] = T::zero();
        out[2] = T::max(t, T::zero());
        return;
    }

    // Bisection
    let (mut lb, mut ub) = exp_get_rho_ub(v);
    let mut rho = T::zero();
    let tol: T = CONE_TOL.as_T();

    for _ in 0..EXP_MAX_ITERS {
        rho = (ub + lb) / (2.0).as_T();
        let g = exp_calc_grad(v, rho);
        if g > T::zero() {
            lb = rho;
        } else {
            ub = rho;
        }
        if ub - lb < tol {
            break;
        }
    }

    let result = exp_solve_for_x_with_rho(v, rho);
    out[0] = result[0];
    out[1] = result[1];
    out[2] = result[2];
}
pub(super) fn derivative_exp_cone<T: FloatT>(z: &[T; 3], dual: bool) -> Vec<T> {
    // Following diffclarabel: when dual=True, evaluate checks at -z
    let xi = if dual {
        [-z[0], -z[1], -z[2]]
    } else {
        [z[0], z[1], z[2]]
    };
    let margin: T = (1e-6).as_T();

    let block = if in_exp_interior(&xi, margin) {
        // xi is in primal cone interior: derivative at xi = I
        [
            [T::one(), T::zero(), T::zero()],
            [T::zero(), T::one(), T::zero()],
            [T::zero(), T::zero(), T::one()],
        ]
    } else if in_exp_dual_interior(&[-xi[0], -xi[1], -xi[2]], margin) {
        // -xi is in dual cone interior => Π_K(xi) = 0
        [[T::zero(); 3]; 3]
    } else if xi[0] < -margin && xi[1] < -margin {
        // Special case: r < 0, s < 0 (with tolerance to avoid discontinuity at xi ≈ 0)
        let mut b = [[T::zero(); 3]; 3];
        b[0][0] = T::one();
        if xi[2] >= T::zero() {
            b[2][2] = T::one();
        }
        b
    } else {
        // Boundary case - compute Jacobian via 4x4 system
        let mut rs = [T::zero(); 3];
        project_exp_cone_primal(&xi, &mut rs);

        let (r, s, t) = (rs[0], rs[1], rs[2]);
        // When s is zero, negative, or very small, the Jacobian computation is ill-defined
        // because the exponential cone requires s > 0. In this degenerate case,
        // use a small positive value to regularize the computation.
        // BUG FIX: Previously used s.abs() > s_min which allowed negative s values
        // to be used, producing incorrect gradients.
        let s_min: T = (1e-10).as_T();
        let s_eff = if s > s_min { s } else { s_min };
        let l = t - xi[2];
        let alpha = (r / s_eff).exp();
        let beta = l * r / (s_eff * s_eff) * alpha;

        // Build 4x4 system
        let mut j_inv = [[T::zero(); 4]; 4];
        j_inv[0][0] = alpha;
        j_inv[0][1] = (-r + s_eff) / s_eff * alpha;
        j_inv[0][2] = -T::one();
        j_inv[1][0] = T::one() + l / s_eff * alpha;
        j_inv[1][1] = -beta;
        j_inv[1][3] = alpha;
        j_inv[2][0] = -beta;
        j_inv[2][1] = T::one() + beta * r / s_eff;
        j_inv[2][3] = (T::one() - r / s_eff) * alpha;
        j_inv[3][2] = T::one();
        j_inv[3][3] = -T::one();

        // Invert 4x4 matrix
        let j_full = invert_4x4(&j_inv);

        // Extract 3x3 submatrix
        [
            [j_full[0][1], j_full[0][2], j_full[0][3]],
            [j_full[1][1], j_full[1][2], j_full[1][3]],
            [j_full[2][1], j_full[2][2], j_full[2][3]],
        ]
    };

    // Convert to flat vector
    let mut result = vec![T::zero(); 9];

    if dual {
        // Moreau decomposition: D_{K*}(z) = I - D_K(z)
        for i in 0..3 {
            for j in 0..3 {
                let delta_ij = if i == j { T::one() } else { T::zero() };
                result[i * 3 + j] = delta_ij - block[i][j];
            }
        }
    } else {
        for i in 0..3 {
            for j in 0..3 {
                result[i * 3 + j] = block[i][j];
            }
        }
    }

    result
}
/// 4x4 matrix inversion (for exponential cone Jacobian)
pub(super) fn invert_4x4<T: FloatT>(m: &[[T; 4]; 4]) -> [[T; 4]; 4] {
    // Use Gaussian elimination with partial pivoting
    let mut a = [[T::zero(); 8]; 4];

    // Augment with identity
    for i in 0..4 {
        for j in 0..4 {
            a[i][j] = m[i][j];
            a[i][j + 4] = if i == j { T::one() } else { T::zero() };
        }
    }

    // Forward elimination
    for col in 0..4 {
        // Find pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in col + 1..4 {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            for j in 0..8 {
                let tmp = a[col][j];
                a[col][j] = a[max_row][j];
                a[max_row][j] = tmp;
            }
        }

        // Check for singular matrix - use regularization instead of silent failure
        let eps: T = (1e-12).as_T();
        if a[col][col].abs() < eps {
            // Regularize the pivot instead of returning wrong identity matrix
            // This provides a more stable numerical result for nearly-singular cases
            a[col][col] = eps.copysign(a[col][col]);
        }

        // Eliminate
        let pivot = a[col][col];
        for j in 0..8 {
            a[col][j] /= pivot;
        }

        for row in 0..4 {
            if row != col {
                let factor = a[row][col];
                for j in 0..8 {
                    a[row][j] -= factor * a[col][j];
                }
            }
        }
    }

    // Extract inverse
    let mut result = [[T::zero(); 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = a[i][j + 4];
        }
    }
    result
}
