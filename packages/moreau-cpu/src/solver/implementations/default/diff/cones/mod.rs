//! Cone projections and derivatives for differentiation.
//!
//! This module implements projections onto cones and their Jacobians,
//! which are needed for implicit differentiation of conic programs.
//!
//! The implementation closely follows the Python reference in diffclarabel.
//!
//! Per-cone-family kernels live in the `soc`, `exp`, `power`, `psd` and
//! `genpow` submodules; this module root holds the shared
//! `ConeDerivativeBlock` representation and the per-product-cone dispatchers.

#[cfg(feature = "sdp")]
use crate::algebra::*;
use crate::algebra::{AsFloatT, FloatT};
use crate::solver::core::cones::SupportedConeT;

mod exp;
mod genpow;
mod power;
#[cfg(feature = "sdp")]
mod psd;
mod soc;

#[cfg(test)]
mod tests;

use exp::{derivative_exp_cone, project_exp_cone_primal};
use genpow::{derivative_genpow_cone, project_genpow_cone_primal};
use power::{derivative_pow_cone, project_pow_cone_primal};
#[cfg(feature = "sdp")]
use psd::{derivative_psd_cone, project_psd_cone};
use soc::project_soc;

// `derivative_soc_sparse` and `derivative_genpow_cone_sparse` are also used
// by the diff KKT module, so re-export them at `pub(crate)` (this also brings
// the names into scope for the dispatchers below).
pub(crate) use genpow::derivative_genpow_cone_sparse;
pub(crate) use soc::derivative_soc_sparse;
// `derivative_soc` (dense) is only referenced from test code — the test-only
// `derivative_cone` dispatcher and the KKT test module.
#[cfg(test)]
pub(crate) use soc::derivative_soc;

/// Efficient representation of cone derivative (Jacobian).
///
/// For diagonal cones (Zero, Nonnegative), we only store the diagonal.
/// For dense cones (SOC, Exp, Power), we store the full matrix.
#[derive(Clone, Debug)]
pub enum ConeDerivativeBlock<T> {
    /// Zero matrix (derivative is 0) - no storage needed
    Zero(usize),
    /// Diagonal matrix - only store diagonal entries (O(n) storage)
    Diagonal(Vec<T>),
    /// Dense matrix in row-major order (O(n²) storage) - only for small cones
    Dense {
        /// Matrix dimension
        dim: usize,
        /// Matrix data in row-major order
        data: Vec<T>,
    },
    /// SOC sparse expansion: diagonal + rank-2 correction.
    /// The full matrix is:
    ///   diag(diag) + c1 * v1 * v1^T + c2 * v2 * v2^T
    /// Used for all SOC cones. The KKT system gets 2 extra columns/rows
    /// for this expansion, keeping the factorization sparse.
    SocSparse {
        /// Cone dimension
        dim: usize,
        /// Diagonal entries [dim]
        diag: Vec<T>,
        /// First expansion vector [dim]
        v1: Vec<T>,
        /// Coefficient for first rank-1 term
        c1: T,
        /// Second expansion vector [dim]
        v2: Vec<T>,
        /// Coefficient for second rank-1 term
        c2: T,
    },
    /// GenPowerCone sparse expansion: diagonal + rank-3 correction.
    /// The full matrix is:
    ///   diag(diag) + left1 * right1^T + left2 * right2^T + c3 * left3 * left3^T
    /// Terms 1-2 are asymmetric (left ≠ right); term 3 is symmetric.
    /// Used for GenPowerCone. The KKT system gets 3 extra columns/rows per cone.
    GenPowerSparse {
        /// Cone dimension
        dim: usize,
        /// Diagonal entries [dim]
        diag: Vec<T>,
        /// First left vector [dim] (a)
        left1: Vec<T>,
        /// First right vector [dim] (b)
        right1: Vec<T>,
        /// Second left vector [dim] (e)
        left2: Vec<T>,
        /// Second right vector [dim] (f)
        right2: Vec<T>,
        /// Third vector [dim] (g) — symmetric rank-1 term
        left3: Vec<T>,
        /// Coefficient for third rank-1 term (c_ww)
        c3: T,
    },
}

impl<T: FloatT> ConeDerivativeBlock<T> {
    /// Expand to a dense dim×dim matrix in row-major order.
    pub fn to_dense(&self) -> Vec<T> {
        match self {
            ConeDerivativeBlock::Zero(dim) => vec![T::zero(); dim * dim],
            ConeDerivativeBlock::Diagonal(diag) => {
                let dim = diag.len();
                let mut m = vec![T::zero(); dim * dim];
                for i in 0..dim {
                    m[i * dim + i] = diag[i];
                }
                m
            }
            ConeDerivativeBlock::Dense { dim: _, data } => data.clone(),
            ConeDerivativeBlock::SocSparse {
                dim,
                diag,
                v1,
                c1,
                v2,
                c2,
            } => {
                let d = *dim;
                let mut m = vec![T::zero(); d * d];
                for i in 0..d {
                    m[i * d + i] = diag[i];
                }
                for i in 0..d {
                    for j in 0..d {
                        m[i * d + j] = m[i * d + j] + *c1 * v1[i] * v1[j] + *c2 * v2[i] * v2[j];
                    }
                }
                m
            }
            ConeDerivativeBlock::GenPowerSparse {
                dim,
                diag,
                left1,
                right1,
                left2,
                right2,
                left3,
                c3,
            } => {
                let d = *dim;
                let mut m = vec![T::zero(); d * d];
                for i in 0..d {
                    m[i * d + i] = diag[i];
                }
                for i in 0..d {
                    for j in 0..d {
                        m[i * d + j] = m[i * d + j]
                            + left1[i] * right1[j]
                            + left2[i] * right2[j]
                            + *c3 * left3[i] * left3[j];
                    }
                }
                m
            }
        }
    }
}

/// Compute the projection of a vector onto a product of cones.
///
/// # Arguments
/// * `z` - Input vector to project
/// * `cones` - List of cones defining the product cone
/// * `dual` - If true, project onto dual cone; otherwise primal cone
///
/// # Returns
/// Projected vector (same length as z)
pub fn get_cone_projection<T: FloatT>(z: &[T], cones: &[SupportedConeT<T>], dual: bool) -> Vec<T> {
    let mut result = vec![T::zero(); z.len()];
    let mut offset = 0;

    for cone in cones {
        let dim = cone.nvars();
        let z_slice = &z[offset..offset + dim];
        let r_slice = &mut result[offset..offset + dim];

        project_cone(z_slice, r_slice, cone, dual);
        offset += dim;
    }

    result
}

/// Compute the Jacobian of cone projection with efficient storage.
///
/// Returns a vector of derivative blocks using sparse representation where possible.
/// Zero and Nonnegative cones use diagonal storage (O(n)), while SOC, Exp, Power
/// use dense storage (but these are always small).
///
/// # Arguments
/// * `z` - Point at which to evaluate the Jacobian
/// * `cones` - List of cones
/// * `dual` - If true, derivative of dual cone projection
///
/// # Returns
/// Vector of ConeDerivativeBlock (one per cone)
pub fn get_cone_derivative_sparse<T: FloatT>(
    z: &[T],
    cones: &[SupportedConeT<T>],
    dual: bool,
) -> Vec<ConeDerivativeBlock<T>> {
    let mut blocks = Vec::with_capacity(cones.len());
    let mut offset = 0;

    for cone in cones {
        let dim = cone.nvars();
        let z_slice = &z[offset..offset + dim];

        let block = derivative_cone_sparse(z_slice, cone, dual);
        blocks.push(block);
        offset += dim;
    }

    blocks
}

/// Compute cone-projection Jacobian blocks `H_x = DΠ_{K_J}(u_J)` for each
/// direct-x cone, mirroring [`get_cone_derivative_sparse`] for slack cones.
///
/// For symmetric cones (nonneg, SOC, PSD) the projection Jacobian is the
/// same function as the slack version — only the input changes from the
/// flat slack `(s, z)` pair to a gathered `(x[J], z_x_J)` pair per cone.
///
/// `u_x_flat` is `z_x_flat - x_J_flat` for `Exact` mode where `x_J_flat`
/// is the gathered primal `x[J]` across all direct-x cones in the same
/// flat layout as `z_x_flat`. Caller is responsible for the gather and
/// for the smoothed-mode equivalent.
pub fn get_cone_derivative_sparse_xcones<T: FloatT>(
    u_x_flat: &[T],
    x_cones: &[crate::solver::core::cones::SupportedXConeT],
    dual: bool,
) -> Vec<ConeDerivativeBlock<T>> {
    use crate::solver::core::cones::SupportedXConeT;
    let mut blocks = Vec::with_capacity(x_cones.len());
    let mut offset = 0usize;
    for xc in x_cones {
        let dim = xc.indices().len();
        let u_slice = &u_x_flat[offset..offset + dim];
        // Map direct-x cone variant to the equivalent slack cone variant
        // and reuse the existing per-cone derivative.
        let block = match xc {
            SupportedXConeT::NonnegativeXConeT(_) => {
                let slack = SupportedConeT::<T>::NonnegativeConeT(dim);
                derivative_cone_sparse(u_slice, &slack, dual)
            }
            SupportedXConeT::SecondOrderXConeT(_) => {
                let slack = SupportedConeT::<T>::SecondOrderConeT(dim);
                derivative_cone_sparse(u_slice, &slack, dual)
            }
            SupportedXConeT::ExponentialXConeT(_) => {
                debug_assert_eq!(dim, 3, "ExpXCone must have exactly 3 indices");
                // Exp cone projection Jacobian (dense 3x3) is the same
                // function regardless of slack vs direct-x context — we
                // dispatch through the slack-form variant for the actual
                // Jacobian computation. The asymmetric primal/dual swap
                // matters for the FORWARD IPM (cone scaling), not for the
                // backward projection Jacobian.
                let slack = SupportedConeT::<T>::ExponentialConeT();
                derivative_cone_sparse(u_slice, &slack, dual)
            }
            SupportedXConeT::PowerXConeT(_, α) => {
                debug_assert_eq!(dim, 3, "PowerXCone must have exactly 3 indices");
                let slack = SupportedConeT::<T>::PowerConeT(T::from(*α).unwrap());
                derivative_cone_sparse(u_slice, &slack, dual)
            }
            SupportedXConeT::GenPowerXConeT(_, alphas, dim2) => {
                debug_assert_eq!(dim, alphas.len() + dim2, "GenPowerXCone dim mismatch");
                let alphas_t: Vec<T> = alphas.iter().map(|&a| T::from(a).unwrap()).collect();
                let slack = SupportedConeT::<T>::GenPowerConeT(alphas_t, *dim2);
                derivative_cone_sparse(u_slice, &slack, dual)
            }
            #[cfg(feature = "sdp")]
            SupportedXConeT::PSDTriangleXConeT(_, k) => {
                let slack = SupportedConeT::<T>::PSDTriangleConeT(*k);
                derivative_cone_sparse(u_slice, &slack, dual)
            }
        };
        blocks.push(block);
        offset += dim;
    }
    blocks
}

/// Apply cone derivative H (from sparse blocks) to a vector: result = H @ v
///
/// This computes the matrix-vector product using the block structure of H.
/// Use this with H_blocks from `get_cone_derivative_sparse`.
pub fn apply_cone_derivative_blocks<T: FloatT>(
    H_blocks: &[ConeDerivativeBlock<T>],
    v: &[T],
) -> Vec<T> {
    let total_dim = v.len();
    let mut result = vec![T::zero(); total_dim];
    let mut offset = 0;

    for block in H_blocks {
        match block {
            ConeDerivativeBlock::Zero(dim) => {
                // H = 0, so H @ v = 0 (already zero-initialized)
                offset += dim;
            }
            ConeDerivativeBlock::Diagonal(diag) => {
                let dim = diag.len();
                for i in 0..dim {
                    result[offset + i] = diag[i] * v[offset + i];
                }
                offset += dim;
            }
            ConeDerivativeBlock::Dense { dim, data } => {
                // Dense matrix in row-major: result[i] = sum_j data[i*dim + j] * v[j]
                for i in 0..*dim {
                    let mut sum = T::zero();
                    for j in 0..*dim {
                        sum += data[i * dim + j] * v[offset + j];
                    }
                    result[offset + i] = sum;
                }
                offset += dim;
            }
            ConeDerivativeBlock::SocSparse {
                dim,
                diag,
                v1,
                c1,
                v2,
                c2,
            } => {
                // H = diag + c1 * v1*v1^T + c2 * v2*v2^T
                // result = diag .* v_slice + c1*(v1^T v_slice)*v1 + c2*(v2^T v_slice)*v2
                let v_slice = &v[offset..offset + dim];
                let mut dot1 = T::zero();
                let mut dot2 = T::zero();
                for i in 0..*dim {
                    dot1 += v1[i] * v_slice[i];
                    dot2 += v2[i] * v_slice[i];
                }
                for i in 0..*dim {
                    result[offset + i] =
                        diag[i] * v_slice[i] + *c1 * dot1 * v1[i] + *c2 * dot2 * v2[i];
                }
                offset += dim;
            }
            ConeDerivativeBlock::GenPowerSparse {
                dim,
                diag,
                left1,
                right1,
                left2,
                right2,
                left3,
                c3,
            } => {
                // H = diag + left1*right1^T + left2*right2^T + c3*left3*left3^T
                let v_slice = &v[offset..offset + dim];
                let mut dot1 = T::zero();
                let mut dot2 = T::zero();
                let mut dot3 = T::zero();
                for i in 0..*dim {
                    dot1 += right1[i] * v_slice[i];
                    dot2 += right2[i] * v_slice[i];
                    dot3 += left3[i] * v_slice[i];
                }
                for i in 0..*dim {
                    result[offset + i] = diag[i] * v_slice[i]
                        + dot1 * left1[i]
                        + dot2 * left2[i]
                        + *c3 * dot3 * left3[i];
                }
                offset += dim;
            }
        }
    }

    result
}

/// Apply cone derivative transpose H' (from sparse blocks) to a vector: result = H' @ v
///
/// This computes the matrix-vector product using the block structure of H.
/// For symmetric cones, H = H', but for non-symmetric cones (exp, power),
/// this correctly transposes the dense blocks.
/// Use this with H_blocks from `get_cone_derivative_sparse`.
pub fn apply_cone_derivative_blocks_transpose<T: FloatT>(
    H_blocks: &[ConeDerivativeBlock<T>],
    v: &[T],
) -> Vec<T> {
    let total_dim = v.len();
    let mut result = vec![T::zero(); total_dim];
    let mut offset = 0;

    for block in H_blocks {
        match block {
            ConeDerivativeBlock::Zero(dim) => {
                // H' = 0, so H' @ v = 0 (already zero-initialized)
                offset += dim;
            }
            ConeDerivativeBlock::Diagonal(diag) => {
                // Diagonal is symmetric: H' = H
                let dim = diag.len();
                for i in 0..dim {
                    result[offset + i] = diag[i] * v[offset + i];
                }
                offset += dim;
            }
            ConeDerivativeBlock::Dense { dim, data } => {
                // Transpose: result[j] = sum_i data[i*dim + j] * v[i]
                // i.e., result[j] = sum_i H[i,j] * v[i] = (H' @ v)[j]
                for j in 0..*dim {
                    let mut sum = T::zero();
                    for i in 0..*dim {
                        sum += data[i * dim + j] * v[offset + i];
                    }
                    result[offset + j] = sum;
                }
                offset += dim;
            }
            ConeDerivativeBlock::SocSparse {
                dim,
                diag,
                v1,
                c1,
                v2,
                c2,
            } => {
                // Symmetric matrix: transpose is same as forward
                let v_slice = &v[offset..offset + dim];
                let mut dot1 = T::zero();
                let mut dot2 = T::zero();
                for i in 0..*dim {
                    dot1 += v1[i] * v_slice[i];
                    dot2 += v2[i] * v_slice[i];
                }
                for i in 0..*dim {
                    result[offset + i] =
                        diag[i] * v_slice[i] + *c1 * dot1 * v1[i] + *c2 * dot2 * v2[i];
                }
                offset += dim;
            }
            ConeDerivativeBlock::GenPowerSparse {
                dim,
                diag,
                left1,
                right1,
                left2,
                right2,
                left3,
                c3,
            } => {
                // Transpose: H' = diag + right1*left1^T + right2*left2^T + c3*left3*left3^T
                let v_slice = &v[offset..offset + dim];
                let mut dot1 = T::zero();
                let mut dot2 = T::zero();
                let mut dot3 = T::zero();
                for i in 0..*dim {
                    dot1 += left1[i] * v_slice[i];
                    dot2 += left2[i] * v_slice[i];
                    dot3 += left3[i] * v_slice[i];
                }
                for i in 0..*dim {
                    result[offset + i] = diag[i] * v_slice[i]
                        + dot1 * right1[i]
                        + dot2 * right2[i]
                        + *c3 * dot3 * left3[i];
                }
                offset += dim;
            }
        }
    }

    result
}

/// Compute the Jacobian of cone projection as dense blocks.
///
/// Used in tests for direct comparison with expected dense matrices.
/// Production code uses `get_cone_derivative_sparse` for better memory efficiency.
#[cfg(test)]
pub fn get_cone_derivative<T: FloatT>(
    z: &[T],
    cones: &[SupportedConeT<T>],
    dual: bool,
) -> Vec<Vec<T>> {
    let mut blocks = Vec::with_capacity(cones.len());
    let mut offset = 0;

    for cone in cones {
        let dim = cone.nvars();
        let z_slice = &z[offset..offset + dim];

        let block = derivative_cone(z_slice, cone, dual);
        blocks.push(block);
        offset += dim;
    }

    blocks
}

/// Apply the cone derivative (Jacobian) to a vector.
#[cfg(test)]
pub fn apply_cone_derivative<T: FloatT>(
    z: &[T],
    v: &[T],
    cones: &[SupportedConeT<T>],
    dual: bool,
    out: &mut [T],
) {
    let mut offset = 0;

    for cone in cones {
        let dim = cone.nvars();
        let z_slice = &z[offset..offset + dim];
        let v_slice = &v[offset..offset + dim];
        let out_slice = &mut out[offset..offset + dim];

        apply_derivative_cone(z_slice, v_slice, cone, dual, out_slice);
        offset += dim;
    }
}
// ============================================================================
// Individual cone projections
// ============================================================================

fn project_cone<T: FloatT>(z: &[T], out: &mut [T], cone: &SupportedConeT<T>, dual: bool) {
    match cone {
        SupportedConeT::ZeroConeT(dim) => {
            if dual {
                out[..*dim].copy_from_slice(&z[..*dim]);
            } else {
                for i in 0..*dim {
                    out[i] = T::zero();
                }
            }
        }
        SupportedConeT::NonnegativeConeT(dim) => {
            // Self-dual
            for i in 0..*dim {
                out[i] = T::max(z[i], T::zero());
            }
        }
        SupportedConeT::SecondOrderConeT(dim) => {
            project_soc(z, out, *dim);
        }
        SupportedConeT::ExponentialConeT() => {
            if dual {
                // Dual cone projection via Moreau: Π_{K*}(z) = z + Π_K(-z)
                // Note: K* (dual) ≠ K° (polar). The polar formula would be z - Π_K(z).
                let mut primal_proj = [T::zero(); 3];
                project_exp_cone_primal(&[-z[0], -z[1], -z[2]], &mut primal_proj);
                for i in 0..3 {
                    out[i] = z[i] + primal_proj[i];
                }
            } else {
                project_exp_cone_primal(&[z[0], z[1], z[2]], out);
            }
        }
        SupportedConeT::PowerConeT(alpha) => {
            if dual {
                // Dual cone projection via Moreau: Π_{K*}(z) = z + Π_K(-z)
                // Note: K* (dual) ≠ K° (polar). The polar formula would be z - Π_K(z).
                let mut primal_proj = [T::zero(); 3];
                project_pow_cone_primal(&[-z[0], -z[1], -z[2]], &mut primal_proj, *alpha);
                for i in 0..3 {
                    out[i] = z[i] + primal_proj[i];
                }
            } else {
                project_pow_cone_primal(&[z[0], z[1], z[2]], out, *alpha);
            }
        }
        SupportedConeT::GenPowerConeT(alpha, dim2) => {
            let dim1 = alpha.len();
            let dim = dim1 + *dim2;
            if dual {
                // Dual cone projection via Moreau: Π_{K*}(z) = z + Π_K(-z)
                let mut neg_z = vec![T::zero(); dim];
                for i in 0..dim {
                    neg_z[i] = -z[i];
                }
                let mut primal_proj = vec![T::zero(); dim];
                project_genpow_cone_primal(&neg_z, &mut primal_proj, alpha, *dim2);
                for i in 0..dim {
                    out[i] = z[i] + primal_proj[i];
                }
            } else {
                project_genpow_cone_primal(z, out, alpha, *dim2);
            }
        }
        #[cfg(feature = "sdp")]
        SupportedConeT::PSDTriangleConeT(mat_dim) => {
            // PSD cone is self-dual: Π_K(z) = Π_{K*}(z)
            project_psd_cone(z, out, *mat_dim);
        }
        _ => {
            // Unsupported cone - identity
            out.copy_from_slice(z);
        }
    }
}
// ============================================================================
// Derivatives
// ============================================================================

#[cfg(test)]
fn derivative_cone<T: FloatT>(z: &[T], cone: &SupportedConeT<T>, dual: bool) -> Vec<T> {
    match cone {
        SupportedConeT::ZeroConeT(dim) => {
            let d = *dim;
            let mut result = vec![T::zero(); d * d];
            if dual {
                for i in 0..d {
                    result[i * d + i] = T::one();
                }
            }
            result
        }
        SupportedConeT::NonnegativeConeT(dim) => {
            let d = *dim;
            let mut result = vec![T::zero(); d * d];
            // Self-dual
            for i in 0..d {
                if z[i] >= T::zero() {
                    result[i * d + i] = T::one();
                }
            }
            result
        }
        SupportedConeT::SecondOrderConeT(dim) => derivative_soc(z, *dim),
        SupportedConeT::ExponentialConeT() => derivative_exp_cone(&[z[0], z[1], z[2]], dual),
        SupportedConeT::PowerConeT(alpha) => derivative_pow_cone(&[z[0], z[1], z[2]], *alpha, dual),
        SupportedConeT::GenPowerConeT(alpha, dim2) => derivative_genpow_cone(z, alpha, *dim2, dual),
        #[cfg(feature = "sdp")]
        SupportedConeT::PSDTriangleConeT(mat_dim) => {
            // PSD cone is self-dual: same derivative for primal and dual
            derivative_psd_cone(z, *mat_dim)
        }
        _ => {
            let dim = cone.nvars();
            let mut result = vec![T::zero(); dim * dim];
            for i in 0..dim {
                result[i * dim + i] = T::one();
            }
            result
        }
    }
}
/// Compute cone derivative with efficient sparse representation.
///
/// For diagonal cones (Zero, Nonnegative), returns Diagonal variant with O(n) storage.
/// For small dense cones (SOC, Exp, Power), returns Dense variant.
fn derivative_cone_sparse<T: FloatT>(
    z: &[T],
    cone: &SupportedConeT<T>,
    dual: bool,
) -> ConeDerivativeBlock<T> {
    match cone {
        SupportedConeT::ZeroConeT(dim) => {
            let d = *dim;
            if dual {
                // Dual of zero cone: derivative is I (identity)
                ConeDerivativeBlock::Diagonal(vec![T::one(); d])
            } else {
                // Primal zero cone: derivative is 0
                ConeDerivativeBlock::Zero(d)
            }
        }
        SupportedConeT::NonnegativeConeT(dim) => {
            let d = *dim;
            // Self-dual: diagonal with 1 where z >= 0, 0 otherwise
            let mut diag = vec![T::zero(); d];
            for i in 0..d {
                if z[i] >= T::zero() {
                    diag[i] = T::one();
                }
            }
            ConeDerivativeBlock::Diagonal(diag)
        }
        SupportedConeT::SecondOrderConeT(dim) => {
            // Sparse expansion: diagonal + rank-2, avoids O(dim²) KKT fill
            derivative_soc_sparse(z, *dim)
        }
        SupportedConeT::ExponentialConeT() => {
            // Always 3x3
            let data = derivative_exp_cone(&[z[0], z[1], z[2]], dual);
            ConeDerivativeBlock::Dense { dim: 3, data }
        }
        SupportedConeT::PowerConeT(alpha) => {
            // Always 3x3
            let data = derivative_pow_cone(&[z[0], z[1], z[2]], *alpha, dual);
            ConeDerivativeBlock::Dense { dim: 3, data }
        }
        SupportedConeT::GenPowerConeT(alpha, dim2) => {
            // Dense vs sparse selection mirrors the forward path
            // (`N_DENSE_GENPOW` in `genpowcone.rs`). For dense cones
            // materialise the dim×dim Jacobian; for large cones use
            // the rank-3 sparse expansion.
            use crate::solver::core::cones::N_DENSE_GENPOW;
            let total = alpha.len() + *dim2;
            if total <= N_DENSE_GENPOW {
                let data = derivative_genpow_cone(z, alpha, *dim2, dual);
                ConeDerivativeBlock::Dense { dim: total, data }
            } else {
                derivative_genpow_cone_sparse(z, alpha, *dim2, dual)
            }
        }
        #[cfg(feature = "sdp")]
        SupportedConeT::PSDTriangleConeT(mat_dim) => {
            // PSD cone is self-dual: same derivative for primal and dual
            let nvars = triangular_number(*mat_dim);
            let data = derivative_psd_cone(z, *mat_dim);
            ConeDerivativeBlock::Dense { dim: nvars, data }
        }
        _ => {
            // Fallback: identity matrix as diagonal
            let dim = cone.nvars();
            ConeDerivativeBlock::Diagonal(vec![T::one(); dim])
        }
    }
}
#[cfg(test)]
fn apply_derivative_cone<T: FloatT>(
    z: &[T],
    v: &[T],
    cone: &SupportedConeT<T>,
    dual: bool,
    out: &mut [T],
) {
    match cone {
        SupportedConeT::ZeroConeT(dim) => {
            if dual {
                out[..*dim].copy_from_slice(&v[..*dim]);
            } else {
                for i in 0..*dim {
                    out[i] = T::zero();
                }
            }
        }
        SupportedConeT::NonnegativeConeT(dim) => {
            for i in 0..*dim {
                out[i] = if z[i] >= T::zero() { v[i] } else { T::zero() };
            }
        }
        _ => {
            // Full matrix-vector multiplication
            let jac = derivative_cone(z, cone, dual);
            let dim = cone.nvars();
            for i in 0..dim {
                out[i] = T::zero();
                for j in 0..dim {
                    out[i] = out[i] + jac[i * dim + j] * v[j];
                }
            }
        }
    }
}
// ============================================================================
// Barrier proximal derivatives: H = (I + μ ∇²φ*(z_prox))⁻¹
// ============================================================================

/// Compute the central-path derivative for a product of cones.
///
/// Given a central-path iterate z (already satisfying z + μ∇φ*(z) = u),
/// computes H = (I + μ ∇²φ*(z))⁻¹ directly — no Newton solve needed.
///
/// This is used for the Smoothed differentiation method, where z is the
/// smoothing iterate produced by the post-convergence refinement loop.
pub fn get_central_path_derivative_sparse<T: FloatT>(
    s: &[T],
    z: &[T],
    cones: &[SupportedConeT<T>],
    mu: T,
) -> Vec<ConeDerivativeBlock<T>> {
    let mut blocks = Vec::with_capacity(cones.len());
    let mut offset = 0;

    for cone in cones {
        let dim = cone.nvars();
        let _s_slice = &s[offset..offset + dim];
        let z_slice = &z[offset..offset + dim];

        let block = central_path_derivative_cone(z_slice, cone, mu);
        blocks.push(block);
        offset += dim;
    }

    blocks
}

/// Smoothed-mode central-path derivative for direct-x cones.
///
/// Mirror of [`get_cone_derivative_sparse_xcones`] for `DiffMethod::Smoothed`:
/// every direct-x cone is mapped to its slack-form equivalent and dispatched
/// through `central_path_derivative_cone(z_x, slack_cone, mu)`. The H block
/// produced enters the augmented (1,1) KKT slot exactly the way the Exact
/// projection Jacobian does — only the per-cone math differs.
pub fn get_central_path_derivative_sparse_xcones<T: FloatT>(
    _s_x_flat: &[T],
    z_x_flat: &[T],
    x_cones: &[crate::solver::core::cones::SupportedXConeT],
    mu: T,
) -> Vec<ConeDerivativeBlock<T>> {
    use crate::solver::core::cones::SupportedXConeT;
    let mut blocks = Vec::with_capacity(x_cones.len());
    let mut offset = 0usize;
    for xc in x_cones {
        let dim = xc.indices().len();
        let z_slice = &z_x_flat[offset..offset + dim];
        let block = match xc {
            SupportedXConeT::NonnegativeXConeT(_) => {
                let slack = SupportedConeT::<T>::NonnegativeConeT(dim);
                central_path_derivative_cone(z_slice, &slack, mu)
            }
            SupportedXConeT::SecondOrderXConeT(_) => {
                let slack = SupportedConeT::<T>::SecondOrderConeT(dim);
                central_path_derivative_cone(z_slice, &slack, mu)
            }
            SupportedXConeT::ExponentialXConeT(_) => {
                debug_assert_eq!(dim, 3, "ExpXCone must have exactly 3 indices");
                let slack = SupportedConeT::<T>::ExponentialConeT();
                central_path_derivative_cone(z_slice, &slack, mu)
            }
            SupportedXConeT::PowerXConeT(_, α) => {
                debug_assert_eq!(dim, 3, "PowerXCone must have exactly 3 indices");
                let slack = SupportedConeT::<T>::PowerConeT(T::from(*α).unwrap());
                central_path_derivative_cone(z_slice, &slack, mu)
            }
            SupportedXConeT::GenPowerXConeT(_, alphas, dim2) => {
                debug_assert_eq!(dim, alphas.len() + dim2, "GenPowerXCone dim mismatch");
                let alphas_t: Vec<T> = alphas.iter().map(|&a| T::from(a).unwrap()).collect();
                let slack = SupportedConeT::<T>::GenPowerConeT(alphas_t, *dim2);
                central_path_derivative_cone(z_slice, &slack, mu)
            }
            #[cfg(feature = "sdp")]
            SupportedXConeT::PSDTriangleXConeT(_, k) => {
                let slack = SupportedConeT::<T>::PSDTriangleConeT(*k);
                central_path_derivative_cone(z_slice, &slack, mu)
            }
        };
        blocks.push(block);
        offset += dim;
    }
    blocks
}
/// Compute the central-path derivative block for a single cone.
///
/// `_s` is unused for LP cones (zero + nonneg) where H depends only on z and μ.
/// It is included in the signature for future SOC/exp/power cone support where
/// both s and z are needed to evaluate ∇²φ*(z).
fn central_path_derivative_cone<T: FloatT>(
    z: &[T],
    cone: &SupportedConeT<T>,
    mu: T,
) -> ConeDerivativeBlock<T> {
    // Guard against mu=0: when z[i]=0 the formula z²/(z²+μ) becomes 0/0.
    // Clamp to a small positive value to avoid NaN.
    let mu = if mu <= T::zero() { T::epsilon() } else { mu };

    match cone {
        SupportedConeT::ZeroConeT(dim) => {
            // Zero cone (free dual): no barrier, H = I
            ConeDerivativeBlock::Diagonal(vec![T::one(); *dim])
        }
        SupportedConeT::NonnegativeConeT(dim) => {
            // For nonneg cone, φ*(z) = -sum(log(z_i)), so ∇²φ*(z) = diag(1/z_i²)
            // H[i,i] = 1 / (1 + μ/z_i²) = z_i² / (z_i² + μ)
            let mut diag = vec![T::zero(); *dim];
            for i in 0..*dim {
                let z2 = z[i] * z[i];
                diag[i] = z2 / (z2 + mu);
            }
            ConeDerivativeBlock::Diagonal(diag)
        }
        SupportedConeT::SecondOrderConeT(dim) => {
            // SOC barrier: φ*(z) = -½ log(ζ), ζ = z₀² - ‖z₁:‖²
            // ∇²φ*(z) = -(1/ζ)J + (2/ζ²)(Jz)(Jz)ᵀ   where J = diag(1,-1,...,-1)
            //
            // M = I + μ∇²φ* = di·I + c·vvᵀ + (d0-di)·e₀e₀ᵀ   where:
            //   d0 = 1 - μ/ζ,  di = 1 + μ/ζ  (di > 1 always)
            //   c = 2μ/ζ²,  v = Jz,  d0-di = -2μ/ζ
            //
            // Two-step Sherman-Morrison (numerically stable even when d0 ≈ 0):
            //   Step 1: N = di·I + c·vvᵀ
            //           N⁻¹ = (1/di)I - β·vvᵀ,  β = c/(di·(di + c·‖v‖²))
            //   Step 2: M = N + (d0-di)·e₀e₀ᵀ
            //           M⁻¹ = N⁻¹ - γ·wwᵀ,  w = N⁻¹e₀,  γ = (d0-di)/(1 + (d0-di)·N⁻¹₀₀)
            //
            // Result: M⁻¹ = (1/di)I + (-β)·v·vᵀ + (-γ)·w·wᵀ  (SocSparse format)
            let dim = *dim;
            let z0 = z[0];
            let mut norm_sq = T::zero();
            for i in 1..dim {
                norm_sq += z[i] * z[i];
            }
            let zeta = z0 * z0 - norm_sq;

            // Guard: if ζ ≤ 0, z is not in the interior. Return identity.
            if zeta <= T::zero() {
                return ConeDerivativeBlock::SocSparse {
                    dim,
                    diag: vec![T::one(); dim],
                    v1: vec![T::zero(); dim],
                    c1: T::zero(),
                    v2: vec![T::zero(); dim],
                    c2: T::zero(),
                };
            }

            let two: T = (2.0).as_T();
            let di = T::one() + mu / zeta;
            let inv_di = T::one() / di;
            let c = two * mu / (zeta * zeta);
            let v_norm_sq = z0 * z0 + norm_sq;

            // Step 1: β = c / (di · (di + c·‖v‖²))
            let beta = c / (di * (di + c * v_norm_sq));

            // Step 2: w = N⁻¹e₀, where N⁻¹ = (1/di)I - β·vvᵀ
            // w[0] = 1/di - β·v₀² = 1/di - β·z₀²
            // w[i] = -β·v₀·v_i = -β·z₀·(-z_i) = β·z₀·z_i  (for i > 0)
            let n_inv_00 = inv_di - beta * z0 * z0;
            let beta_z0 = beta * z0;

            // γ = (d0 - di) / (1 + (d0 - di)·N⁻¹₀₀),  d0 - di = -2μ/ζ
            let d0_minus_di = -two * mu / zeta;
            let gamma = d0_minus_di / (T::one() + d0_minus_di * n_inv_00);

            // Build SocSparse: M⁻¹ = (1/di)I + (-β)·v·vᵀ + (-γ)·w·wᵀ
            let diag = vec![inv_di; dim];

            // v1 = Jz = (z₀, -z₁, -z₂, ...)
            let mut v1 = vec![T::zero(); dim];
            v1[0] = z0;
            for i in 1..dim {
                v1[i] = -z[i];
            }

            // v2 = w = N⁻¹e₀
            let mut v2 = vec![T::zero(); dim];
            v2[0] = n_inv_00;
            for i in 1..dim {
                v2[i] = beta_z0 * z[i];
            }

            ConeDerivativeBlock::SocSparse {
                dim,
                diag,
                v1,
                c1: -beta,
                v2,
                c2: -gamma,
            }
        }
        _ => {
            panic!(
                "Smoothed differentiation not yet implemented for {:?}. Use diff_method='exact'.",
                cone
            );
        }
    }
}
