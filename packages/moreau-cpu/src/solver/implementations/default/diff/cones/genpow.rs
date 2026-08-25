//! Generalized power cone projection and projection-Jacobian kernels.

use super::ConeDerivativeBlock;
use crate::algebra::{AsFloatT, FloatT};
use crate::solver::implementations::default::diff::CONE_TOL;

const GENPOW_MAX_ITERS: usize = 200;

// ============================================================================
// Generalized Power Cone projection
// ============================================================================

/// Check if point is in GenPowerCone interior with margin
fn in_genpow_cone_interior<T: FloatT>(p: &[T], w: &[T], alpha: &[T], margin: T) -> bool {
    let two: T = (2.0).as_T();

    if !p.iter().all(|&pi| pi > margin) {
        return false;
    }

    let log_prod: T = alpha
        .iter()
        .zip(p.iter())
        .fold(T::zero(), |acc, (&ai, &pi)| acc + two * ai * pi.ln());
    let prod = log_prod.exp();
    let norm_w_sq: T = w.iter().fold(T::zero(), |acc, &wi| acc + wi * wi);

    prod >= norm_w_sq + margin
}

/// Check if point is in polar cone of GenPowerCone interior with margin
/// Polar cone: {(u,v) : ∏ (-u_i/αi)^αi ≥ ||v||₂, u ≤ 0}
fn in_genpow_polar_interior<T: FloatT>(p: &[T], w: &[T], alpha: &[T], margin: T) -> bool {
    let two: T = (2.0).as_T();

    if !p.iter().all(|&pi| pi < -margin) {
        return false;
    }

    let log_prod: T = alpha
        .iter()
        .zip(p.iter())
        .fold(T::zero(), |acc, (&ai, &pi)| {
            acc + two * ai * (-pi / ai).ln()
        });
    let prod = log_prod.exp();
    let norm_w_sq: T = w.iter().fold(T::zero(), |acc, &wi| acc + wi * wi);

    prod >= norm_w_sq + margin
}

/// Compute p_i* given scalar r and original p_i:
///   p_i* = 0.5 * (p_i + sqrt(p_i² + 4αi * r * (||w|| - r)))
fn genpow_calc_pi<T: FloatT>(r: T, pi: T, norm_w: T, ai: T) -> T {
    let tol: T = CONE_TOL.as_T();
    let four: T = (4.0).as_T();
    let half: T = (0.5).as_T();
    let val = pi * pi + four * ai * r * (norm_w - r);
    let result = half * (pi + T::max(val, T::zero()).sqrt());
    T::max(result, tol)
}

/// Compute d(p_i*)/dr with soft regularization to avoid discontinuities
fn genpow_calc_dpi_dr<T: FloatT>(r: T, pi_star: T, pi: T, norm_w: T, ai: T) -> T {
    let tol: T = CONE_TOL.as_T();
    let two: T = (2.0).as_T();
    let denom = two * pi_star - pi;
    ai * (norm_w - two * r) / (denom + tol.copysign(denom))
}

/// Project onto primal generalized power cone.
///
/// Uses Newton iteration on the scalar equation:
///   φ(r) = ∏ p_i*(r)^αi - r = 0
/// where r is the norm of the projected w-component.
pub(super) fn project_genpow_cone_primal<T: FloatT>(
    v: &[T],
    out: &mut [T],
    alpha: &[T],
    dim2: usize,
) {
    let dim1 = alpha.len();
    let dim = dim1 + dim2;
    let margin: T = (1e-6).as_T();
    let tol: T = CONE_TOL.as_T();

    let p = &v[..dim1];
    let w = &v[dim1..dim];

    // Interior case
    if in_genpow_cone_interior(p, w, alpha, margin) {
        out[..dim].copy_from_slice(&v[..dim]);
        return;
    }

    // Polar interior: v is in polar cone → projection is 0
    if in_genpow_polar_interior(p, w, alpha, margin) {
        for i in 0..dim {
            out[i] = T::zero();
        }
        return;
    }

    // Compute ||w||
    let norm_w: T = w.iter().fold(T::zero(), |acc, &wi| acc + wi * wi).sqrt();

    // Special case: w = 0 → project p onto nonneg orthant
    if norm_w <= tol {
        for i in 0..dim1 {
            out[i] = T::max(p[i], T::zero());
        }
        for i in dim1..dim {
            out[i] = T::zero();
        }
        return;
    }

    // General case: Newton iteration on φ(r) = ∏ p_i*(r)^αi - r = 0
    let half: T = (0.5).as_T();
    let proj_tol: T = (1e-12).as_T();
    let proj_fp_tol: T = (1e-14).as_T();

    // Initialize r
    let mut r = norm_w * half;
    let mut pi_stars: Vec<T> = vec![T::zero(); dim1];

    for _ in 0..GENPOW_MAX_ITERS {
        // Compute all p_i*(r)
        for i in 0..dim1 {
            pi_stars[i] = genpow_calc_pi(r, p[i], norm_w, alpha[i]);
        }

        // φ(r) = ∏ p_i*^αi - r
        let log_prod: T = alpha
            .iter()
            .zip(pi_stars.iter())
            .fold(T::zero(), |acc, (&ai, &psi)| acc + ai * psi.ln());
        let prod = log_prod.exp();
        let f = prod - r;

        if f.abs() < proj_tol {
            break;
        }

        // φ'(r) = (∏ p_i*^αi) * Σ(αi * dp_i*/dr / p_i*) - 1
        let mut fp = -T::one();
        for i in 0..dim1 {
            let dpi_dr = genpow_calc_dpi_dr(r, pi_stars[i], p[i], norm_w, alpha[i]);
            if pi_stars[i].abs() > tol {
                fp += prod * alpha[i] * dpi_dr / pi_stars[i];
            }
        }

        if fp.abs() < proj_fp_tol {
            break;
        }

        r -= f / fp;
        r = T::max(r, T::zero());
        r = T::min(r, norm_w);
    }

    // Write projection output
    for i in 0..dim1 {
        out[i] = genpow_calc_pi(r, p[i], norm_w, alpha[i]);
    }
    // w* = (r / ||w||) * w
    let scale = r / norm_w;
    for i in 0..dim2 {
        out[dim1 + i] = scale * w[i];
    }
}
/// GenPowerCone projection derivative in sparse form: diagonal + rank-3.
///
/// The Jacobian decomposes as:
///   J = D + a⊗b^T + e⊗f^T + c_ww * g⊗g^T
///
/// Where:
///   D = diag([dpi_dpi[0..dim1]; (r/||w||) * ones(dim2)])
///   a = [dpi_dr[0..dim1]; ŵ[0..dim2]]
///   b = [dr_dp[0..dim1]; dr_dnw * ŵ[0..dim2]]
///   e = [dpi_dnw[0..dim1]; zeros(dim2)]
///   f = [zeros(dim1); ŵ[0..dim2]]
///   g = [zeros(dim1); w[0..dim2]]
///   c_ww = -r/||w||³  (the dr_dnw/||w||² term is captured by a⊗b^T)
///
/// For dual: J_dual = (I - D) - a⊗b^T - e⊗f^T - c_ww * g⊗g^T
pub(crate) fn derivative_genpow_cone_sparse<T: FloatT>(
    z: &[T],
    alpha: &[T],
    dim2: usize,
    dual: bool,
) -> ConeDerivativeBlock<T> {
    let dim1 = alpha.len();
    let dim = dim1 + dim2;
    let tol: T = CONE_TOL.as_T();
    let margin: T = (1e-6).as_T();

    // When dual=True, evaluate at -z for primal projection
    let xi: Vec<T> = if dual {
        z.iter().map(|&x| -x).collect()
    } else {
        z[..dim].to_vec()
    };

    let p = &xi[..dim1];
    let w = &xi[dim1..dim];

    if in_genpow_cone_interior(p, w, alpha, margin) {
        // Interior: J = I  →  D = I, rank terms zero
        // For dual: I - I = 0, but we still keep GenPowerSparse for consistent sparsity
        if dual {
            return ConeDerivativeBlock::GenPowerSparse {
                dim,
                diag: vec![T::zero(); dim],
                left1: vec![T::zero(); dim],
                right1: vec![T::zero(); dim],
                left2: vec![T::zero(); dim],
                right2: vec![T::zero(); dim],
                left3: vec![T::zero(); dim],
                c3: T::zero(),
            };
        }
        return ConeDerivativeBlock::GenPowerSparse {
            dim,
            diag: vec![T::one(); dim],
            left1: vec![T::zero(); dim],
            right1: vec![T::zero(); dim],
            left2: vec![T::zero(); dim],
            right2: vec![T::zero(); dim],
            left3: vec![T::zero(); dim],
            c3: T::zero(),
        };
    }

    if in_genpow_polar_interior(p, w, alpha, margin) {
        // Polar: J = 0
        // For dual: I - 0 = I
        if dual {
            return ConeDerivativeBlock::GenPowerSparse {
                dim,
                diag: vec![T::one(); dim],
                left1: vec![T::zero(); dim],
                right1: vec![T::zero(); dim],
                left2: vec![T::zero(); dim],
                right2: vec![T::zero(); dim],
                left3: vec![T::zero(); dim],
                c3: T::zero(),
            };
        }
        return ConeDerivativeBlock::GenPowerSparse {
            dim,
            diag: vec![T::zero(); dim],
            left1: vec![T::zero(); dim],
            right1: vec![T::zero(); dim],
            left2: vec![T::zero(); dim],
            right2: vec![T::zero(); dim],
            left3: vec![T::zero(); dim],
            c3: T::zero(),
        };
    }

    let norm_w: T = w.iter().fold(T::zero(), |acc, &wi| acc + wi * wi).sqrt();

    // Special case: w = 0
    if norm_w <= tol {
        let half: T = (0.5).as_T();
        let mut diag_vec = vec![T::zero(); dim];
        for i in 0..dim1 {
            diag_vec[i] = if p[i].abs() > tol {
                half * (p[i].signum() + T::one())
            } else {
                half
            };
        }
        if dual {
            for i in 0..dim {
                diag_vec[i] = T::one() - diag_vec[i];
            }
        }
        return ConeDerivativeBlock::GenPowerSparse {
            dim,
            diag: diag_vec,
            left1: vec![T::zero(); dim],
            right1: vec![T::zero(); dim],
            left2: vec![T::zero(); dim],
            right2: vec![T::zero(); dim],
            left3: vec![T::zero(); dim],
            c3: T::zero(),
        };
    }

    // Boundary case: Newton iteration with tight tolerance for derivative accuracy
    let _two: T = (2.0).as_T();
    let half: T = (0.5).as_T();
    let four: T = (4.0).as_T();
    let deriv_tol: T = (1e-12).as_T();
    let deriv_fp_tol: T = (1e-14).as_T();

    let mut r = norm_w * half;
    let mut pi_stars: Vec<T> = vec![T::zero(); dim1];

    for _ in 0..GENPOW_MAX_ITERS {
        for i in 0..dim1 {
            pi_stars[i] = genpow_calc_pi(r, p[i], norm_w, alpha[i]);
        }

        let log_prod: T = alpha
            .iter()
            .zip(pi_stars.iter())
            .fold(T::zero(), |acc, (&ai, &psi)| acc + ai * psi.ln());
        let prod = log_prod.exp();
        let f = prod - r;

        if f.abs() < deriv_tol {
            break;
        }

        let mut fp = -T::one();
        for i in 0..dim1 {
            let dpi_dr = genpow_calc_dpi_dr(r, pi_stars[i], p[i], norm_w, alpha[i]);
            if pi_stars[i].abs() > tol {
                fp += prod * alpha[i] * dpi_dr / pi_stars[i];
            }
        }

        if fp.abs() < deriv_fp_tol {
            break;
        }

        r -= f / fp;
        r = T::max(r, T::zero());
        r = T::min(r, norm_w);
    }

    // Recompute pi_stars at converged r
    for i in 0..dim1 {
        pi_stars[i] = genpow_calc_pi(r, p[i], norm_w, alpha[i]);
    }

    // Compute intermediate quantities (same as dense version)
    let log_prod: T = alpha
        .iter()
        .zip(pi_stars.iter())
        .fold(T::zero(), |acc, (&ai, &psi)| acc + ai * psi.ln());
    let prod = log_prod.exp();

    let mut sqrt_gi: Vec<T> = Vec::with_capacity(dim1);
    for i in 0..dim1 {
        let gi = p[i] * p[i] + four * alpha[i] * r * (norm_w - r);
        sqrt_gi.push(T::max(gi, tol).sqrt());
    }

    let mut dpi_dpi_vec: Vec<T> = Vec::with_capacity(dim1);
    for i in 0..dim1 {
        // Use pi_star/sqrt_gi to avoid catastrophic cancellation when p[i] ≈ 0
        dpi_dpi_vec.push(pi_stars[i] / sqrt_gi[i]);
    }

    let mut dpi_dr_vec: Vec<T> = Vec::with_capacity(dim1);
    for i in 0..dim1 {
        dpi_dr_vec.push(genpow_calc_dpi_dr(r, pi_stars[i], p[i], norm_w, alpha[i]));
    }

    let mut dpi_dnw_vec: Vec<T> = Vec::with_capacity(dim1);
    for i in 0..dim1 {
        dpi_dnw_vec.push(alpha[i] * r / sqrt_gi[i]);
    }

    // dphi/dr, dphi/dp_i, dphi/d(||w||)
    let mut dphi_dr = -T::one();
    for i in 0..dim1 {
        if pi_stars[i].abs() > tol {
            dphi_dr += prod * alpha[i] * dpi_dr_vec[i] / pi_stars[i];
        }
    }

    let mut dphi_dp: Vec<T> = vec![T::zero(); dim1];
    for i in 0..dim1 {
        if pi_stars[i].abs() > tol {
            dphi_dp[i] = prod * alpha[i] * dpi_dpi_vec[i] / pi_stars[i];
        }
    }

    let mut dphi_dnw = T::zero();
    for i in 0..dim1 {
        if pi_stars[i].abs() > tol {
            dphi_dnw += prod * alpha[i] * dpi_dnw_vec[i] / pi_stars[i];
        }
    }

    let mut dr_dp: Vec<T> = vec![T::zero(); dim1];
    let dphi_dr_reg = dphi_dr + tol.copysign(dphi_dr);
    for i in 0..dim1 {
        dr_dp[i] = -dphi_dp[i] / dphi_dr_reg;
    }
    let dr_dnw = -dphi_dnw / dphi_dr_reg;

    // Build decomposition vectors
    let r_over_nw = r / norm_w;

    // D = diag([dpi_dpi[0..dim1]; r/||w|| * ones(dim2)])
    let mut diag_vec = vec![T::zero(); dim];
    for i in 0..dim1 {
        diag_vec[i] = dpi_dpi_vec[i];
    }
    for i in 0..dim2 {
        diag_vec[dim1 + i] = r_over_nw;
    }

    // a = [dpi_dr[0..dim1]; ŵ[0..dim2]]  where ŵ = w/||w||
    let mut left1_vec = vec![T::zero(); dim];
    for i in 0..dim1 {
        left1_vec[i] = dpi_dr_vec[i];
    }
    for i in 0..dim2 {
        left1_vec[dim1 + i] = w[i] / norm_w;
    }

    // b = [dr_dp[0..dim1]; dr_dnw * ŵ[0..dim2]]
    let mut right1_vec = vec![T::zero(); dim];
    for i in 0..dim1 {
        right1_vec[i] = dr_dp[i];
    }
    for i in 0..dim2 {
        right1_vec[dim1 + i] = dr_dnw * w[i] / norm_w;
    }

    // e = [dpi_dnw[0..dim1]; zeros(dim2)]
    let mut left2_vec = vec![T::zero(); dim];
    for i in 0..dim1 {
        left2_vec[i] = dpi_dnw_vec[i];
    }

    // f = [zeros(dim1); ŵ[0..dim2]]
    let mut right2_vec = vec![T::zero(); dim];
    for i in 0..dim2 {
        right2_vec[dim1 + i] = w[i] / norm_w;
    }

    // g = [zeros(dim1); w[0..dim2]]
    let mut left3_vec = vec![T::zero(); dim];
    for i in 0..dim2 {
        left3_vec[dim1 + i] = w[i];
    }

    // For dim2=1, the c3 term (-r/||w||³ * g⊗g') blows up as ||w||→0.
    // Since dim2=1, g⊗g' contributes only to the scalar ww entry: c3*w² = -r/||w||.
    // Fold this into the diagonal and set c3=0 to avoid the blowup.
    let c_ww = if dim2 == 1 {
        // diag was r/||w||, add c3*w² = -r*w²/||w||³ = -r/||w|| (for dim2=1, ||w||=|w|)
        diag_vec[dim1] = T::zero(); // diag + c3*w² cancel; left1*right1 gives dr_dnw
        left3_vec[dim1] = T::zero();
        T::zero()
    } else {
        // c_ww = -r/||w||³
        // The dr_dnw/||w||² contribution to dw*/dw is already captured by a⊗b^T.
        -r / (norm_w * norm_w * norm_w)
    };

    if dual {
        // J_dual = (I - D) - a⊗b^T - e⊗f^T - c_ww * g⊗g^T
        for i in 0..dim {
            diag_vec[i] = T::one() - diag_vec[i];
        }
        for i in 0..dim {
            left1_vec[i] = -left1_vec[i];
        }
        for i in 0..dim {
            left2_vec[i] = -left2_vec[i];
        }
        ConeDerivativeBlock::GenPowerSparse {
            dim,
            diag: diag_vec,
            left1: left1_vec,
            right1: right1_vec,
            left2: left2_vec,
            right2: right2_vec,
            left3: left3_vec,
            c3: -c_ww,
        }
    } else {
        ConeDerivativeBlock::GenPowerSparse {
            dim,
            diag: diag_vec,
            left1: left1_vec,
            right1: right1_vec,
            left2: left2_vec,
            right2: right2_vec,
            left3: left3_vec,
            c3: c_ww,
        }
    }
}
pub(super) fn derivative_genpow_cone<T: FloatT>(
    z: &[T],
    alpha: &[T],
    dim2: usize,
    dual: bool,
) -> Vec<T> {
    // Delegate to sparse representation and expand to dense
    let block = derivative_genpow_cone_sparse(z, alpha, dim2, dual);
    block.to_dense()
}
