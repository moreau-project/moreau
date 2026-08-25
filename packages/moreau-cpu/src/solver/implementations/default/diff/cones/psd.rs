//! PSD (SDP) cone projection and projection-Jacobian kernels.
//!
//! This entire submodule is gated behind the `sdp` feature.

use crate::algebra::*;

// ============================================================================
// PSD (SDP) cone projection and derivative
// ============================================================================

/// Project onto the PSD cone in svec space.
///
/// PSD cone is self-dual (K* = K), so primal and dual projections are the same.
#[cfg(feature = "sdp")]
pub(super) fn project_psd_cone<T: FloatT>(z: &[T], out: &mut [T], mat_dim: usize) {
    let nvars = triangular_number(mat_dim);
    debug_assert_eq!(z.len(), nvars);
    debug_assert_eq!(out.len(), nvars);

    let mut S = Matrix::<T>::zeros((mat_dim, mat_dim));
    svec_to_mat(&mut S, z);

    let mut eig = EigEngine::<T>::new(mat_dim);
    eig.eigen(&mut S).expect("Eigval error in PSD projection");

    // Project eigenvalues: λ_+ = max(λ, 0)
    let Q = eig
        .V
        .as_ref()
        .expect("Eigenvectors required for PSD projection");

    // Reconstruct: S_+ = Q diag(λ_+) Q'
    let mut QΛ = Matrix::<T>::zeros((mat_dim, mat_dim));
    QΛ.data_mut().copy_from_slice(Q.data());
    for j in 0..mat_dim {
        let λ_plus = T::max(eig.λ[j], T::zero());
        for i in 0..mat_dim {
            QΛ[(i, j)] *= λ_plus;
        }
    }

    let mut result = Matrix::<T>::zeros((mat_dim, mat_dim));
    result.mul(&QΛ, &Q.t(), T::one(), T::zero());
    mat_to_svec(out, &result);
}

/// Compute the dense Jacobian of PSD cone projection in svec space.
///
/// The Jacobian is: D_Π_K(s)[ds] = svec(Q * (Ω ∘ (Q' * mat(ds) * Q)) * Q')
/// where Ω_ij = (max(λ_i,0) - max(λ_j,0)) / (λ_i - λ_j) for λ_i ≠ λ_j,
///       Ω_ii = 1 if λ_i ≥ 0, else 0.
///
/// PSD cone is self-dual, so dual derivative is the same as primal.
///
/// Returns the nvars×nvars Jacobian in row-major order.
#[cfg(feature = "sdp")]
pub(super) fn derivative_psd_cone<T: FloatT>(z: &[T], mat_dim: usize) -> Vec<T> {
    let nvars = triangular_number(mat_dim);

    let mut S = Matrix::<T>::zeros((mat_dim, mat_dim));
    svec_to_mat(&mut S, z);

    let mut eig = EigEngine::<T>::new(mat_dim);
    eig.eigen(&mut S).expect("Eigval error in PSD derivative");

    let Q = eig
        .V
        .as_ref()
        .expect("Eigenvectors required for PSD derivative");
    let λ = &eig.λ;

    // Build Ω matrix using the continuous extension of the spectral derivative.
    // For f(λ) = max(λ, 0), same-sign pairs have the exact CE values:
    //   both ≥ 0:  Ω = 1
    //   both ≤ 0:  Ω = 0
    //   mixed:     Ω = (max(λ_i,0) - max(λ_j,0)) / (λ_i - λ_j), stable because
    //              the denominator is bounded below by |λ+| + |λ-|.
    // Matches the CUDA implementation in diff_psd.cu build_omega_kernel.
    let mut Ω = Matrix::<T>::zeros((mat_dim, mat_dim));
    for i in 0..mat_dim {
        for j in 0..mat_dim {
            if i == j {
                Ω[(i, j)] = if λ[i] >= T::zero() {
                    T::one()
                } else {
                    T::zero()
                };
            } else if λ[i] >= T::zero() && λ[j] >= T::zero() {
                Ω[(i, j)] = T::one();
            } else if λ[i] <= T::zero() && λ[j] <= T::zero() {
                Ω[(i, j)] = T::zero();
            } else {
                let λi_plus = T::max(λ[i], T::zero());
                let λj_plus = T::max(λ[j], T::zero());
                Ω[(i, j)] = (λi_plus - λj_plus) / (λ[i] - λ[j]);
            }
        }
    }

    // Build the Jacobian by applying the linear map to each svec basis vector.
    // For each basis vector e_k in svec space:
    //   J[:,k] = svec(Q * (Ω ∘ (Q' * mat(e_k) * Q)) * Q')
    let mut jacobian = vec![T::zero(); nvars * nvars];
    let mut e_k = vec![T::zero(); nvars];
    let mut E_k = Matrix::<T>::zeros((mat_dim, mat_dim));
    let mut QtE = Matrix::<T>::zeros((mat_dim, mat_dim));
    let mut QtEQ = Matrix::<T>::zeros((mat_dim, mat_dim));
    let mut ΩH = Matrix::<T>::zeros((mat_dim, mat_dim));
    let mut QΩH = Matrix::<T>::zeros((mat_dim, mat_dim));
    let mut result_mat = Matrix::<T>::zeros((mat_dim, mat_dim));
    let mut result_vec = vec![T::zero(); nvars];

    for k in 0..nvars {
        // Set up basis vector
        e_k[k] = T::one();

        // Convert to matrix
        svec_to_mat(&mut E_k, &e_k);

        // Q' * E_k
        QtE.mul(&Q.t(), &E_k, T::one(), T::zero());

        // Q' * E_k * Q
        QtEQ.mul(&QtE, Q, T::one(), T::zero());

        // Ω ∘ (Q' * E_k * Q)  (Hadamard product)
        for i in 0..mat_dim {
            for j in 0..mat_dim {
                ΩH[(i, j)] = Ω[(i, j)] * QtEQ[(i, j)];
            }
        }

        // Q * (Ω ∘ ...)
        QΩH.mul(Q, &ΩH, T::one(), T::zero());

        // Q * (Ω ∘ ...) * Q'
        result_mat.mul(&QΩH, &Q.t(), T::one(), T::zero());

        // Convert back to svec
        mat_to_svec(&mut result_vec, &result_mat);

        // Store as column k of Jacobian (row-major storage)
        for i in 0..nvars {
            jacobian[i * nvars + k] = result_vec[i];
        }

        // Reset basis vector
        e_k[k] = T::zero();
    }

    jacobian
}
