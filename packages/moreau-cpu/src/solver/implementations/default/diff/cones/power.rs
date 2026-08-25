//! Power cone projection and projection-Jacobian kernels.

use crate::algebra::{AsFloatT, FloatT};
use crate::solver::implementations::default::diff::CONE_TOL;

const POW_MAX_ITERS: usize = 50;

// ============================================================================
// Power cone
// ============================================================================

fn in_pow_cone_interior<T: FloatT>(x: T, y: T, abs_z: T, alpha: T, margin: T) -> bool {
    if x <= margin || y <= margin {
        return false;
    }
    x.powf(alpha) * y.powf(T::one() - alpha) >= abs_z + margin
}

fn in_pow_polar_cone_interior<T: FloatT>(u: T, v: T, abs_w: T, alpha: T, margin: T) -> bool {
    if u >= -margin || v >= -margin {
        return false;
    }
    let lhs = (-u / alpha).powf(alpha) * (-v / (T::one() - alpha)).powf(T::one() - alpha);
    lhs >= abs_w + margin
}

fn pow_calc_xi<T: FloatT>(ri: T, x: T, abs_z: T, alpha: T) -> T {
    let tol: T = CONE_TOL.as_T();
    let four: T = (4.0).as_T();
    let half: T = (0.5).as_T();
    let val = x * x + four * alpha * (abs_z - ri) * ri;
    let xi = half * (x + T::max(val, T::zero()).sqrt());
    T::max(xi, tol)
}

fn pow_calc_f<T: FloatT>(ri: T, xi: T, yi: T, alpha: T) -> T {
    xi.powf(alpha) * yi.powf(T::one() - alpha) - ri
}

fn pow_calc_dxi_dr<T: FloatT>(ri: T, xi: T, x: T, abs_z: T, alpha: T) -> T {
    let tol: T = CONE_TOL.as_T();
    let two: T = (2.0).as_T();
    let denom: T = two * xi - x;
    if denom.abs() < tol {
        return T::zero();
    }
    alpha * (abs_z - two * ri) / denom
}

fn pow_calc_fp<T: FloatT>(xi: T, yi: T, dxidri: T, dyidri: T, alpha: T) -> T {
    let alphac = T::one() - alpha;
    let tol: T = CONE_TOL.as_T();

    // Safe division - avoid NaN/Inf at cone boundaries where xi or yi approach zero
    let term_x = if xi.abs() > tol {
        alpha * dxidri / xi
    } else {
        T::zero()
    };
    let term_y = if yi.abs() > tol {
        alphac * dyidri / yi
    } else {
        T::zero()
    };

    xi.powf(alpha) * yi.powf(alphac) * (term_x + term_y) - T::one()
}

/// Project onto primal power cone and compute Jacobian
fn project_pow_cone_with_jacobian<T: FloatT>(v: &[T; 3], alpha: T) -> ([T; 3], [[T; 3]; 3]) {
    let (x, y, z) = (v[0], v[1], v[2]);
    let abs_z = z.abs();
    let margin: T = (1e-6).as_T();
    let tol: T = CONE_TOL.as_T();

    // Interior case
    if in_pow_cone_interior(x, y, abs_z, alpha, margin) {
        return (
            [x, y, z],
            [
                [T::one(), T::zero(), T::zero()],
                [T::zero(), T::one(), T::zero()],
                [T::zero(), T::zero(), T::one()],
            ],
        );
    }

    // Polar interior case
    if in_pow_polar_cone_interior(x, y, abs_z, alpha, margin) {
        return ([T::zero(), T::zero(), T::zero()], [[T::zero(); 3]; 3]);
    }

    // z = 0 case
    if abs_z <= tol {
        let proj = [T::max(x, T::zero()), T::max(y, T::zero()), T::zero()];
        let mut jac = [[T::zero(); 3]; 3];

        let half: T = (0.5).as_T();
        // BUG FIX: Previously used exact comparison (x != T::zero()) which fails for tiny
        // floating-point values. Now use tolerance-based comparison for numerical stability.
        jac[0][0] = if x.abs() > tol {
            half * (x.signum() + T::one())
        } else {
            half
        };
        jac[1][1] = if y.abs() > tol {
            half * (y.signum() + T::one())
        } else {
            half
        };

        let two: T = (2.0).as_T();
        if x > T::zero() && y < T::zero() {
            if alpha > half {
                jac[2][2] = T::one();
            } else if alpha < half {
                jac[2][2] = T::zero();
            } else {
                // BUG FIX: Use tolerance instead of exact > T::zero() comparison
                jac[2][2] = if y.abs() > tol {
                    x / (two * y.abs() + x)
                } else {
                    T::one()
                };
            }
        } else if y > T::zero() && x < T::zero() {
            if alpha < half {
                jac[2][2] = T::one();
            } else if alpha > half {
                jac[2][2] = T::zero();
            } else {
                // BUG FIX: Use tolerance instead of exact > T::zero() comparison
                jac[2][2] = if x.abs() > tol {
                    y / (two * x.abs() + y)
                } else {
                    T::one()
                };
            }
        }

        return (proj, jac);
    }

    // General case: Newton iteration
    let mut r = abs_z / (2.0).as_T();
    let mut xi = T::zero();
    let mut yi = T::zero();

    for _ in 0..POW_MAX_ITERS {
        xi = pow_calc_xi(r, x, abs_z, alpha);
        yi = pow_calc_xi(r, y, abs_z, T::one() - alpha);

        let f = pow_calc_f(r, xi, yi, alpha);

        if f.abs() < tol {
            break;
        }

        let dxdr = pow_calc_dxi_dr(r, xi, x, abs_z, alpha);
        let dydr = pow_calc_dxi_dr(r, yi, y, abs_z, T::one() - alpha);
        let fp = pow_calc_fp(xi, yi, dxdr, dydr, alpha);

        if fp.abs() < tol {
            break;
        }

        r -= f / fp;
        r = T::max(r, T::zero());
        r = T::min(r, abs_z);
    }

    let x_star = xi;
    let y_star = yi;
    let z_star = if z >= T::zero() { r } else { -r };

    // Compute Jacobian using implicit differentiation
    //
    // The projection satisfies the fixed-point equations:
    //   x* = 0.5 * (x + sqrt(x^2 + 4*alpha*(|z| - r*)*r*))
    //   y* = 0.5 * (y + sqrt(y^2 + 4*(1-alpha)*(|z| - r*)*r*))
    //   z* = sign(z) * r*
    //   phi(r*) = x*^alpha * y*^(1-alpha) - r* = 0
    //
    // Using implicit differentiation on phi = 0:
    //   dr*/dx = -dphi/dx / dphi/dr
    //   dr*/dy = -dphi/dy / dphi/dr
    //   dr*/d|z| = -dphi/d|z| / dphi/dr
    //
    // Then the full Jacobian is computed from chain rule.
    //
    // NOTE: The diffqcp analytical formula is WRONG (verified against JAX autodiff).
    // This implementation uses the correct implicit differentiation formula.

    let a = alpha;
    let ac = T::one() - alpha;
    let r_star = r;
    let sign_z: T = if z != T::zero() { z.signum() } else { T::one() };

    let two: T = (2.0).as_T();
    let half: T = (0.5).as_T();

    // Compute sqrt terms
    let four: T = (4.0).as_T();
    let gx = x * x + four * a * (abs_z - r_star) * r_star;
    let gy = y * y + four * ac * (abs_z - r_star) * r_star;
    let sqrt_gx = T::max(gx, tol).sqrt();
    let sqrt_gy = T::max(gy, tol).sqrt();

    // Partial derivatives of xi w.r.t. x, |z|, r (holding others constant)
    let dxi_dx = half + x / (two * sqrt_gx);
    let dxi_dz = a * r_star / sqrt_gx;
    let dxi_dr = a * (abs_z - two * r_star) / sqrt_gx;

    // Partial derivatives of yi w.r.t. y, |z|, r
    let dyi_dy = half + y / (two * sqrt_gy);
    let dyi_dz = ac * r_star / sqrt_gy;
    let dyi_dr = ac * (abs_z - two * r_star) / sqrt_gy;

    // At solution, phi = x*^a * y*^(1-a) - r* = 0, so x*^a * y*^(1-a) = r*
    // Partial derivatives of phi
    let dphi_dx = a * r_star * dxi_dx / x_star;
    let dphi_dy = ac * r_star * dyi_dy / y_star;
    let dphi_dz = a * r_star * dxi_dz / x_star + ac * r_star * dyi_dz / y_star;
    let dphi_dr = a * r_star * dxi_dr / x_star + ac * r_star * dyi_dr / y_star - T::one();

    // From phi = 0: dr*/d(...) = -dphi/d(...) / dphi/dr
    let (dr_dx, dr_dy, dr_dz_abs) = if dphi_dr.abs() > tol {
        (-dphi_dx / dphi_dr, -dphi_dy / dphi_dr, -dphi_dz / dphi_dr)
    } else {
        (T::zero(), T::zero(), T::zero())
    };

    // Full Jacobian d[x*, y*, z*] / d[x, y, z]
    let mut jac = [[T::zero(); 3]; 3];

    // dx*/dx = dxi_dx + dxi_dr * dr_dx
    jac[0][0] = dxi_dx + dxi_dr * dr_dx;
    // dx*/dy = dxi_dr * dr_dy (xi doesn't depend on y directly)
    jac[0][1] = dxi_dr * dr_dy;
    // dx*/dz = (dxi_dz + dxi_dr * dr_dz_abs) * sign_z
    jac[0][2] = (dxi_dz + dxi_dr * dr_dz_abs) * sign_z;

    // dy*/dx = dyi_dr * dr_dx (yi doesn't depend on x directly)
    jac[1][0] = dyi_dr * dr_dx;
    // dy*/dy = dyi_dy + dyi_dr * dr_dy
    jac[1][1] = dyi_dy + dyi_dr * dr_dy;
    // dy*/dz = (dyi_dz + dyi_dr * dr_dz_abs) * sign_z
    jac[1][2] = (dyi_dz + dyi_dr * dr_dz_abs) * sign_z;

    // dz*/dx = sign_z * dr_dx
    jac[2][0] = sign_z * dr_dx;
    // dz*/dy = sign_z * dr_dy
    jac[2][1] = sign_z * dr_dy;
    // dz*/dz = dr_dz_abs (for z != 0)
    jac[2][2] = dr_dz_abs;

    ([x_star, y_star, z_star], jac)
}

pub(super) fn project_pow_cone_primal<T: FloatT>(v: &[T; 3], out: &mut [T], alpha: T) {
    let (proj, _) = project_pow_cone_with_jacobian(v, alpha);
    out[0] = proj[0];
    out[1] = proj[1];
    out[2] = proj[2];
}
pub(super) fn derivative_pow_cone<T: FloatT>(z: &[T; 3], alpha: T, dual: bool) -> Vec<T> {
    // Following diffclarabel: when dual=True, evaluate checks at -z
    let xi = if dual {
        [-z[0], -z[1], -z[2]]
    } else {
        [z[0], z[1], z[2]]
    };
    let abs_xi = xi[2].abs();
    let margin: T = (1e-6).as_T();

    let block = if in_pow_cone_interior(xi[0], xi[1], abs_xi, alpha, margin) {
        // xi is in primal cone interior: derivative at xi = I
        [
            [T::one(), T::zero(), T::zero()],
            [T::zero(), T::one(), T::zero()],
            [T::zero(), T::zero(), T::one()],
        ]
    } else if in_pow_polar_cone_interior(xi[0], xi[1], abs_xi, alpha, margin) {
        // xi is in polar cone interior: derivative at xi = 0
        [[T::zero(); 3]; 3]
    } else {
        // Boundary: compute Jacobian from projection at xi
        let (_, jac) = project_pow_cone_with_jacobian(&xi, alpha);
        jac
    };

    // Convert to flat vector
    let mut result = vec![T::zero(); 9];

    if dual {
        // D = I - block
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
