//! Second-order cone projection and projection-Jacobian kernels.

use super::ConeDerivativeBlock;
use crate::algebra::{AsFloatT, FloatT};

/// Second-order cone projection (self-dual)
pub(super) fn project_soc<T: FloatT>(z: &[T], out: &mut [T], dim: usize) {
    if dim < 2 {
        out[..dim].copy_from_slice(&z[..dim]);
        return;
    }

    let t = z[0];

    // Compute ||x||
    let mut norm_z = T::zero();
    for i in 1..dim {
        norm_z += z[i] * z[i];
    }
    norm_z = norm_z.sqrt();

    if norm_z <= t {
        // Inside cone
        out[..dim].copy_from_slice(&z[..dim]);
    } else if norm_z <= -t {
        // In polar cone
        for i in 0..dim {
            out[i] = T::zero();
        }
    } else {
        // On boundary
        let half: T = (0.5).as_T();
        let scale = half * (T::one() + t / norm_z);
        out[0] = norm_z * scale;
        for i in 1..dim {
            out[i] = z[i] * scale;
        }
    }
}
// Dense SOC derivative; reference for the sparse form in tests only.
#[cfg(test)]
pub(crate) fn derivative_soc<T: FloatT>(z: &[T], dim: usize) -> Vec<T> {
    let mut result = vec![T::zero(); dim * dim];

    if dim < 2 {
        result[0] = T::one();
        return result;
    }

    let t = z[0];

    // Compute ||x||
    let mut norm_z = T::zero();
    for i in 1..dim {
        norm_z = norm_z + z[i] * z[i];
    }
    norm_z = norm_z.sqrt();

    if norm_z <= t {
        // Interior: identity
        for i in 0..dim {
            result[i * dim + i] = T::one();
        }
    } else if norm_z <= -t {
        // Polar: zero
    } else {
        // Boundary
        let half: T = (0.5).as_T();

        // u = x / ||x||
        let mut u = vec![T::zero(); dim - 1];
        for i in 0..dim - 1 {
            u[i] = z[i + 1] / norm_z;
        }

        // H[0,0] = 0.5
        result[0] = half;

        // H[0,1:] = 0.5 * u, H[1:,0] = 0.5 * u
        for i in 0..dim - 1 {
            result[i + 1] = half * u[i];
            result[(i + 1) * dim] = half * u[i];
        }

        // H[1:,1:] = ((t + norm_z) * I - t * u*u') / (2*norm_z)
        let two: T = (2.0).as_T();
        let scale = (t + norm_z) / (two * norm_z);
        let scale2 = t / (two * norm_z);

        for i in 0..dim - 1 {
            for j in 0..dim - 1 {
                let delta_ij = if i == j { T::one() } else { T::zero() };
                result[(i + 1) * dim + (j + 1)] = scale * delta_ij - scale2 * u[i] * u[j];
            }
        }
    }

    result
}
/// SOC projection derivative in sparse form: diagonal + rank-2.
///
/// Instead of O(dim²) dense storage, we store:
///   H = diag(d) + c1 * v1 * v1^T + c2 * v2 * v2^T
///
/// Boundary case decomposition (||x|| > |t|, ||x|| > 0):
///   d = (0, α, α, ..., α)     where α = (t + ||x||) / (2||x||)
///   v1 = (1, û)                c1 = 0.5
///   v2 = (0, û)                c2 = -α
///   (û = x/||x||)
///
/// This matches the dense matrix exactly:
///   H[0,0] = 0 + 0.5*1 + 0 = 0.5
///   H[0,i] = 0 + 0.5*û_i + 0 = 0.5*û_i
///   H[i,j] = α*δ_ij + 0.5*û_i*û_j - α*û_i*û_j = α*δ_ij - β*û_i*û_j
///   where β = t/(2||x||) = α - 0.5
pub(crate) fn derivative_soc_sparse<T: FloatT>(z: &[T], dim: usize) -> ConeDerivativeBlock<T> {
    debug_assert!(dim >= 2);

    if dim < 2 {
        return ConeDerivativeBlock::Diagonal(vec![T::one(); dim]);
    }

    let t = z[0];

    // Compute ||x||
    let mut norm_z = T::zero();
    for i in 1..dim {
        norm_z += z[i] * z[i];
    }
    norm_z = norm_z.sqrt();

    if norm_z <= t {
        // Interior: identity — must keep SocSparse for consistent sparsity pattern
        ConeDerivativeBlock::SocSparse {
            dim,
            diag: vec![T::one(); dim],
            v1: vec![T::zero(); dim],
            c1: T::zero(),
            v2: vec![T::zero(); dim],
            c2: T::zero(),
        }
    } else if norm_z <= -t {
        // Polar: zero — must keep SocSparse for consistent sparsity pattern
        ConeDerivativeBlock::SocSparse {
            dim,
            diag: vec![T::zero(); dim],
            v1: vec![T::zero(); dim],
            c1: T::zero(),
            v2: vec![T::zero(); dim],
            c2: T::zero(),
        }
    } else {
        // Boundary: H = diag(d) + c1 * v1 * v1^T + c2 * v2 * v2^T
        let two: T = (2.0).as_T();
        let half: T = (0.5).as_T();

        // û = x / ||x||
        let u_hat: Vec<T> = z[1..dim].iter().map(|&zi| zi / norm_z).collect();

        let alpha = (t + norm_z) / (two * norm_z);

        // Diagonal: (0, α, α, ..., α)
        let mut diag = vec![alpha; dim];
        diag[0] = T::zero();

        // v1 = (1, û), c1 = 0.5
        let mut v1 = vec![T::zero(); dim];
        v1[0] = T::one();
        v1[1..].copy_from_slice(&u_hat);
        let c1 = half;

        // v2 = (0, û), c2 = -α (clone v1 and zero the first element)
        let mut v2 = v1.clone();
        v2[0] = T::zero();
        let c2 = -alpha;

        ConeDerivativeBlock::SocSparse {
            dim,
            diag,
            v1,
            c1,
            v2,
            c2,
        }
    }
}
