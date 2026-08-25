use crate::{algebra::*, solver::core::ScalingStrategy};

// --------------------------------------
// Traits and blanket implementations for Exponential, 3D Power and ND Power Cones
// -------------------------------------
// Operations supported on all nonsymmetric cones
pub(crate) trait NonsymmetricCone<T: FloatT> {
    // Returns true if s is primal feasible
    fn is_primal_feasible(&self, s: &[T]) -> bool;

    // Returns true if z is dual feasible
    fn is_dual_feasible(&self, z: &[T]) -> bool;

    fn barrier_primal(&mut self, s: &[T]) -> T;

    fn barrier_dual(&mut self, z: &[T]) -> T;

    fn higher_correction(&mut self, η: &mut [T], ds: &[T], v: &[T]);

    /// Direct entry into the primal-side η formula given `u, v` already
    /// in x-space (no SMW recovery). For the direct-x IPM the natural
    /// Mehrotra correction is `0.5·D³F(x)[step_x, step_x, ·]` so we
    /// have the directions directly without solving `H·u = ds`. Cap
    /// applies the K=7 ∞-norm bound against `cap_ds`.
    fn higher_correction_primal_direct(&mut self, η: &mut [T], _u: &[T], _v: &[T], _cap_ds: &[T]) {
        for v in η.iter_mut() {
            *v = T::zero();
        }
    }

    fn update_dual_grad_H(&mut self, z: &[T]);
}

// --------------------------------------
// Trait and blanket utlity implementations for Exponential and 3D Power Cones
// -------------------------------------
#[allow(clippy::too_many_arguments)]
pub(crate) trait Nonsymmetric3DCone<T: FloatT> {
    fn gradient_primal(&self, s: &[T]) -> [T; 3];

    #[allow(dead_code)]
    fn hessian_primal_3x3(&self, s: &[T]) -> DenseMatrixSym3<T>;

    fn split_borrow_mut(
        &mut self,
    ) -> (
        &mut DenseMatrixSym3<T>,
        &mut DenseMatrixSym3<T>,
        &mut [T; 3],
        &mut [T; 3],
    );
}

pub(crate) trait Nonsymmetric3DConeUtils<T: FloatT> {
    fn update_Hs(&mut self, s: &[T], z: &[T], μ: T, scaling_strategy: ScalingStrategy);

    fn use_dual_scaling(&mut self, μ: T);

    fn use_primal_dual_scaling(&mut self, s: &[T], z: &[T]);
}

impl<T, C> Nonsymmetric3DConeUtils<T> for C
where
    T: FloatT,
    C: Nonsymmetric3DCone<T>,
{
    fn update_Hs(&mut self, s: &[T], z: &[T], μ: T, scaling_strategy: ScalingStrategy) {
        // Choose the scaling strategy
        if scaling_strategy == ScalingStrategy::Dual {
            // Dual scaling: Hs = μ*H
            self.use_dual_scaling(μ);
        } else {
            self.use_primal_dual_scaling(s, z);
        }
    }

    // implements dual only scaling
    fn use_dual_scaling(&mut self, μ: T) {
        let (H_dual, Hs, _, _) = self.split_borrow_mut();
        Hs.scaled_from(μ, H_dual);
    }

    fn use_primal_dual_scaling(&mut self, s: &[T], z: &[T]) {
        let three: T = (3.).as_T();

        let zt: [T; 3] = self.gradient_primal(s);

        let (H_dual, Hs, grad, _) = self.split_borrow_mut();

        let st = grad;
        let mut δs = [T::zero(); 3];
        let mut tmp = [T::zero(); 3];

        // compute zt,st,μt locally
        // NB: zt,st have different sign convention wrt Mosek paper
        let dot_sz = s.dot(z);
        let μ = dot_sz / three;
        let μt = st[..].dot(&zt[..]) / three;

        // δs = s + μ*st
        // δz = z + μ*zt
        let mut δz = tmp;
        for i in 0..3 {
            δs[i] = s[i] + μ * st[i];
            δz[i] = z[i] + μ * zt[i];
        }
        let dot_δsz = δs[..].dot(&δz[..]);

        let de1 = μ * μt - T::one();
        let de2 = H_dual.quad_form(&zt, &zt) - three * μt * μt;

        // use the primal-dual scaling
        if T::abs(de1) > T::sqrt(T::epsilon()) &&      // too close to central path
           T::abs(de2) > T::epsilon()          &&      // others for numerical stability
           dot_sz > T::zero()                  &&
           dot_δsz > T::zero()
        {
            // compute t
            // tmp = μt*st - H*zt
            H_dual.mul(&mut tmp, &zt);
            for i in 0..3 {
                tmp[i] = μt * st[i] - tmp[i];
            }

            // Hs as a workspace (only need to write the upper triangle)
            Hs.copy_from(H_dual);
            for i in 0..3 {
                for j in i..3 {
                    Hs[(i, j)] -= st[i] * st[j] / three + tmp[i] * tmp[j] / de2;
                }
            }
            let t = μ * Hs.norm_fro(); //Frobenius norm

            // generate the remaining axis
            // axis_z = cross(z,zt)
            let mut axis_z = tmp;
            axis_z[0] = z[1] * zt[2] - z[2] * zt[1];
            axis_z[1] = z[2] * zt[0] - z[0] * zt[2];
            axis_z[2] = z[0] * zt[1] - z[1] * zt[0];
            axis_z.normalize();

            // Hs = s*s'/⟨s,z⟩ + δs*δs'/⟨δs,δz⟩ + t*axis_z*axis_z'
            // (only need to write the upper triangle)
            for i in 0..3 {
                for j in i..3 {
                    Hs[(i, j)] =
                        s[i] * s[j] / dot_sz + δs[i] * δs[j] / dot_δsz + t * axis_z[i] * axis_z[j];
                }
            }

        // use the dual scaling
        } else {
            // Hs = μH when s,z are on the central path
            self.use_dual_scaling(μ);
        }
    }
}

// --------------------------------------
// Traits for general ND cones
// -------------------------------------

// Operations supported on ND nonsymmetrics only.  Note this
// differs from the 3D cone in particular because we don't
// return a 3D tuple for the primal gradient.
pub(crate) trait NonsymmetricNDCone<T: FloatT> {
    // Compute the primal gradient of f(s) at s
    fn gradient_primal(&self, grad: &mut [T], s: &[T]);
}

// --------------------------------------
// ND Mosek-Tunçel primal-dual scaling
// --------------------------------------
//
// Both helpers below construct Hs satisfying the primal-dual secant
// conditions (S1) Hs·z = s and (S2) Hs·δz = δs (the latter exact for the
// dense path, approximate for the sparse path), where
//     δs = s + μ · ∇F*(z),   δz = z + μ · ∇F(s).
//
// They rely on the orthogonality identities ⟨δs,z⟩ = ⟨s,δz⟩ = 0 — both
// follow from log-homogeneity ⟨g, x⟩ = -ν and hold for any (s, z) in the
// cone interior, not just on the central path.

/// Sparse **rank-6** expansion of the full Mosek-Tunçel scaling
/// `Hs = ss'/⟨s,z⟩ + δs δs'/⟨δs,δz⟩ + P_⊥·μH·P_⊥` (both secants exact).
///
/// **Why rank-6 and not rank-9.** The earlier rank-9 form decomposed
/// `P_⊥·μH·P_⊥ − μH` into 7 rank-1 axes via the
/// `ab' + ba' = (a+b)(a+b)' − aa' − bb'` identity. The 7 axes' coefficients
/// (`inv_z2`, `inv_dzp2`, `coef_z`, `coef_dzp`, `coef_zdzp`) blow up
/// individually as `‖dzp‖ → 0` near the cone boundary while their *sum*
/// stays bounded — the cancellation is exact in math but only `O(ulp)`
/// in floating-point. Each huge axis enters the augmented LDL as an
/// off-diagonal column scaled by `√|coef|`, and the cancellation has
/// to happen *during* the Schur elimination, where every step adds
/// rounding error proportional to the (now massive) intermediate
/// pivot magnitudes. End result: the augmented LDL produces Newton
/// steps with `~3e-4` relative error vs the dense path, sparse takes
/// 4–12× more IPM iters on skewed-α boundary problems.
///
/// **Rank-6 fix.** The projector correction `P_⊥·μH·P_⊥ − μH` has rank
/// at most 4 (it's supported on `span{e_z, e_dzp, h_z, h_dzp}` where
/// `e_z = z/‖z‖`, `e_dzp = dzp/‖dzp‖`, `h_z = μH·e_z`, `h_dzp = μH·e_dzp`).
/// QR-factorising those 4 vectors gives an orthonormal `Q` (n×4 unit
/// vectors) and an `R` (4×4) that contains the divergent magnitudes
/// (`1/‖dzp‖`, etc.). Compute the 4×4 representation
/// `M = Q' · (P_⊥·μH·P_⊥ − μH) · Q` directly from the closed-form
/// projector identity (no explicit `1/‖dzp‖²` factors), eigendecompose
/// `M`, and the resulting 4 eigenpairs `(λ_k, u_k)` give 4 axes
/// `Q · u_k` (unit) with bounded coefficients (`|λ_k| ≤ ‖μH‖`).
/// Plus 2 bounded secant axes (`s/√⟨s,z⟩`, `δs/√⟨δs,δz⟩`). Total
/// rank-6 expansion, all axes unit, all coefs bounded — augmented
/// LDL pivots stay bounded.
///
/// Slot layout:
/// ```text
/// 0..3 : projector axes  (Q · u_k for k=0..3)   sign = sign(λ_k)
/// 4    : s     (secant)                          sign = +1
/// 5    : δs    (secant)                          sign = +1
/// ```
pub(crate) struct PdScalingNdQr6<T: FloatT> {
    /// 6 axis vectors, each length `n`. Slots 0..3 are the projector
    /// eigenvector axes; slots 4, 5 are the s and δs secant axes.
    pub axes: Vec<Vec<T>>,
    /// 6 axis coefficient magnitudes (`|coef|` ≥ 0). Each axis contributes
    /// `signs[k] · coefs[k] · axes[k]·axes[k]'` to `Hs - μH`.
    pub coefs: Vec<T>,
    /// 6 sign values (`±1`). All can flip per iteration based on the
    /// eigenvalues of the 4×4 projector representation; the LDL
    /// backend ingests `signs` per iteration via `update_dsigns`.
    pub signs: Vec<i8>,
    /// True iff numerical guards passed; false ⇒ caller falls back to
    /// dual-only scaling for this iteration.
    pub ok: bool,
}

/// Default signs at construction (all `+1`). They get overwritten with
/// `sign(eigenvalue_k)` per iteration in `pd_scaling_nd_qr6`.
pub(crate) const PD_QR6_DEFAULT_SIGNS: [i8; 6] = [1, 1, 1, 1, 1, 1];

impl<T: FloatT> PdScalingNdQr6<T> {
    pub fn zero(n: usize) -> Self {
        Self {
            axes: (0..6).map(|_| vec![T::zero(); n]).collect(),
            coefs: vec![T::zero(); 6],
            signs: PD_QR6_DEFAULT_SIGNS.to_vec(),
            ok: false,
        }
    }

    /// Reset to the "zeroed" state in-place — clears `axes` (preserves
    /// allocation), zeros `coefs`, resets `signs` to default, sets
    /// `ok = false`. Used by callers that hold a preallocated instance
    /// and want to avoid per-iter `vec!` traffic.
    pub fn reset(&mut self) {
        for ax in self.axes.iter_mut() {
            for v in ax.iter_mut() {
                *v = T::zero();
            }
        }
        for c in self.coefs.iter_mut() {
            *c = T::zero();
        }
        self.signs.copy_from_slice(&PD_QR6_DEFAULT_SIGNS);
        self.ok = false;
    }
}

/// Per-iteration scratch for the rank-6 + dense PD scaling builders.
/// All buffers are sized to the cone dimension `n` and reused across IPM
/// iterations — see `GenPowerConeData::pd_scratch`. Splitting by purpose
/// (not by which builder consumes which) keeps the borrow story simple
/// at every call site.
pub(crate) struct PdScalingWorkspace<T: FloatT> {
    pub delta_s: Vec<T>,
    pub delta_z: Vec<T>,
    pub e_z: Vec<T>,
    pub e_dzp: Vec<T>,
    pub h_z: Vec<T>,
    pub h_dzp: Vec<T>,
    pub q3: Vec<T>,
    pub q4: Vec<T>,
    pub col_buf: Vec<T>,
    pub basis: Vec<T>,
}

impl<T: FloatT> PdScalingWorkspace<T> {
    pub fn new(n: usize) -> Self {
        let mk = || vec![T::zero(); n];
        Self {
            delta_s: mk(),
            delta_z: mk(),
            e_z: mk(),
            e_dzp: mk(),
            h_z: mk(),
            h_dzp: mk(),
            q3: mk(),
            q4: mk(),
            col_buf: mk(),
            basis: mk(),
        }
    }
}

/// Build the rank-6 sparse expansion of the ND PD scaling. Sister of
/// `pd_scaling_nd_dense` for cones where dim is too large to afford the
/// dense Hs block.
///
/// `mul_μH(out, x)` computes `out = μ·H_dual·x` (caller-supplied to
/// stay agnostic of the cone's rank-3 representation).
///
/// Returns `result.ok = false` when any of the secant denominators
/// (`⟨s,z⟩`, `⟨δs,δz⟩`, `‖z‖²`, `‖dzp‖²`) collapse below `√ε`; the
/// caller then falls back to dual-only scaling.
///
/// See `PdScalingNdQr6` doc for the algorithm overview.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pd_scaling_nd_qr6<T, MulMuH>(
    s: &[T],
    z: &[T],
    gz: &[T],
    gs: &[T],
    μ: T,
    mut mul_μH: MulMuH,
    ws: &mut PdScalingWorkspace<T>,
    r: &mut PdScalingNdQr6<T>,
) where
    T: FloatT,
    MulMuH: FnMut(&mut [T], &[T]),
{
    let n = s.len();
    debug_assert_eq!(z.len(), n);
    debug_assert_eq!(gz.len(), n);
    debug_assert_eq!(gs.len(), n);

    let eps_sqrt = T::sqrt(T::epsilon());
    r.reset();

    let sz = s.dot(z);
    if sz <= eps_sqrt {
        return;
    }
    let z_norm = T::sqrt(z.dot(z));
    if z_norm <= eps_sqrt {
        return;
    }

    // δs = s + μ·gz, δz = z + μ·gs.
    let delta_s = &mut ws.delta_s[..n];
    for i in 0..n {
        delta_s[i] = s[i] + μ * gz[i];
    }
    let delta_z = &mut ws.delta_z[..n];
    for i in 0..n {
        delta_z[i] = z[i] + μ * gs[i];
    }
    let δs_δz = delta_s.dot(&*delta_z);
    if δs_δz <= eps_sqrt {
        return;
    }

    // -------------------------------------------------------------
    // Step 1: Orthonormal basis e_z, e_dzp for span{z, δz}.
    //
    // e_z = z/‖z‖. dzp = δz − ⟨δz,e_z⟩·e_z (Gram-Schmidt with one
    // reorthogonalisation pass — Kahan's "twice is enough" — to
    // recover orthogonality when δz ≈ z near the central path).
    // e_dzp = dzp/‖dzp‖.
    // -------------------------------------------------------------
    let e_z = &mut ws.e_z[..n];
    for i in 0..n {
        e_z[i] = z[i] / z_norm;
    }
    let e_dzp = &mut ws.e_dzp[..n];
    let dot1 = delta_z.dot(&*e_z);
    for i in 0..n {
        e_dzp[i] = delta_z[i] - dot1 * e_z[i];
    }
    let dot2 = e_dzp.dot(&*e_z);
    for i in 0..n {
        e_dzp[i] -= dot2 * e_z[i];
    }
    let dzp_norm = T::sqrt(e_dzp.dot(&*e_dzp));
    if dzp_norm <= eps_sqrt {
        return;
    }
    for i in 0..n {
        e_dzp[i] /= dzp_norm;
    }
    let _ = dot1; // silence

    // -------------------------------------------------------------
    // Step 2: h_z = μH·e_z, h_dzp = μH·e_dzp.
    // q11 = ⟨e_z, h_z⟩, q12 = ⟨e_z, h_dzp⟩, q22 = ⟨e_dzp, h_dzp⟩.
    // (q12 = ⟨e_dzp, h_z⟩ by symmetry of μH.)
    // -------------------------------------------------------------
    let h_z = &mut ws.h_z[..n];
    mul_μH(h_z, e_z);
    let h_dzp = &mut ws.h_dzp[..n];
    mul_μH(h_dzp, e_dzp);
    let q11 = e_z.dot(&*h_z);
    let q12 = e_z.dot(&*h_dzp);
    let q22 = e_dzp.dot(&*h_dzp);

    // -------------------------------------------------------------
    // Step 3: Extend {e_z, e_dzp} to an orthonormal basis Q for
    // span{e_z, e_dzp, h_z, h_dzp} via modified Gram-Schmidt.
    //
    //   q3 = (h_z − q11·e_z − q12·e_dzp) / ‖·‖           (norm = α)
    //   q4 = (h_dzp − q12·e_z − q22·e_dzp − β·q3) / ‖·‖   (norm = γ)
    //
    // Both with one reorthogonalisation pass. If a residual norm
    // collapses, the basis is rank-deficient (the corresponding axis
    // is redundant) and we set the axis coef to zero downstream.
    // -------------------------------------------------------------
    let zero = T::zero();
    let q3 = &mut ws.q3[..n];
    for i in 0..n {
        q3[i] = h_z[i] - q11 * e_z[i] - q12 * e_dzp[i];
    }
    let r1 = q3.dot(&*e_z);
    let r2 = q3.dot(&*e_dzp);
    for i in 0..n {
        q3[i] -= r1 * e_z[i] + r2 * e_dzp[i];
    }
    let alpha = T::sqrt(q3.dot(&*q3));
    let q3_ok = alpha > eps_sqrt;
    if q3_ok {
        for i in 0..n {
            q3[i] /= alpha;
        }
    }

    let q4 = &mut ws.q4[..n];
    let mut beta = zero;
    for i in 0..n {
        q4[i] = h_dzp[i] - q12 * e_z[i] - q22 * e_dzp[i];
    }
    if q3_ok {
        beta = q4.dot(&*q3);
        for i in 0..n {
            q4[i] -= beta * q3[i];
        }
    }
    let r3 = q4.dot(&*e_z);
    let r4 = q4.dot(&*e_dzp);
    for i in 0..n {
        q4[i] -= r3 * e_z[i] + r4 * e_dzp[i];
    }
    if q3_ok {
        let r5 = q4.dot(&*q3);
        for i in 0..n {
            q4[i] -= r5 * q3[i];
        }
    }
    let gamma = T::sqrt(q4.dot(&*q4));
    let q4_ok = gamma > eps_sqrt;
    if q4_ok {
        for i in 0..n {
            q4[i] /= gamma;
        }
    }

    // -------------------------------------------------------------
    // Step 4: Closed-form 4×4 representation of the projector
    // correction `C = P_⊥·μH·P_⊥ − μH` in the Q = [e_z, e_dzp, q3, q4]
    // basis. With C = -h_z e_z' - e_z h_z' - h_dzp e_dzp' - e_dzp h_dzp'
    //              + q11 e_z e_z' + q22 e_dzp e_dzp' + q12 (e_z e_dzp' + e_dzp e_z'),
    //
    // and noting e_z, e_dzp ⊥ q3, q4, while
    //   h_z   = q11·e_z + q12·e_dzp + α·q3
    //   h_dzp = q12·e_z + q22·e_dzp + β·q3 + γ·q4
    //
    // the 4×4 M = Q'·C·Q comes out as:
    //
    //   M = [[ -q11, -q12,  -α,   0  ],
    //        [ -q12, -q22,  -β,  -γ  ],
    //        [ -α,   -β,     0,   0  ],
    //        [  0,   -γ,     0,   0  ]]
    //
    // (saddle-point form; rank ≤ 4). Eigendecompose for the 4 axes.
    // -------------------------------------------------------------
    let mut m = [[zero; 4]; 4];
    m[0][0] = -q11;
    m[0][1] = -q12;
    m[1][0] = -q12;
    m[1][1] = -q22;
    if q3_ok {
        m[0][2] = -alpha;
        m[2][0] = -alpha;
        m[1][2] = -beta;
        m[2][1] = -beta;
    }
    if q4_ok {
        m[1][3] = -gamma;
        m[3][1] = -gamma;
    }

    // -------------------------------------------------------------
    // Step 5: Symmetric Jacobi eigendecomposition of the 4×4 M.
    // Robust, ~5 sweeps. Outputs eigvals on `m`'s diagonal and eigvecs
    // accumulated in `v` (initially identity).
    // -------------------------------------------------------------
    let mut v = [[zero; 4]; 4];
    for i in 0..4 {
        v[i][i] = T::one();
    }
    jacobi_4x4(&mut m, &mut v);
    let lambda = [m[0][0], m[1][1], m[2][2], m[3][3]];

    // -------------------------------------------------------------
    // Step 6: Materialise the 4 projector axes ax_k = Q · v[:,k]
    // (each a unit vector of length n). Set coef = |λ_k|, sign = sign(λ_k).
    // Skip axes whose |λ| is below `T::epsilon()` (truly zero — eg
    // the rank-deficient case where some Q column collapsed). Anything
    // larger we keep; sparse-rank dropping with a generous threshold
    // (1e-12) trades exactness for one fewer rank-1 column and tends
    // to perturb the Schur complement enough to cost extra IPM iters
    // — not worth it.
    // -------------------------------------------------------------
    let coef_threshold = T::epsilon();
    for k in 0..4 {
        let l = lambda[k];
        let mag = l.abs();
        if mag <= coef_threshold {
            r.coefs[k] = zero;
            // axis vector zeroed out — off-diag column is zero, no Schur term.
            for i in 0..n {
                r.axes[k][i] = zero;
            }
            continue;
        }
        r.coefs[k] = mag;
        r.signs[k] = if l >= zero { 1 } else { -1 };
        // axis_k = v[0,k]·e_z + v[1,k]·e_dzp + v[2,k]·q3 + v[3,k]·q4
        let v0 = v[0][k];
        let v1 = v[1][k];
        let v2 = if q3_ok { v[2][k] } else { zero };
        let v3 = if q4_ok { v[3][k] } else { zero };
        for i in 0..n {
            r.axes[k][i] = v0 * e_z[i] + v1 * e_dzp[i] + v2 * q3[i] + v3 * q4[i];
        }
    }

    // -------------------------------------------------------------
    // Step 7: Two secant axes (s, δs) with bounded coefs.
    // -------------------------------------------------------------
    for i in 0..n {
        r.axes[4][i] = s[i];
        r.axes[5][i] = delta_s[i];
    }
    r.coefs[4] = T::one() / sz;
    r.coefs[5] = T::one() / δs_δz;
    // signs[4], signs[5] default to +1.

    r.ok = true;
}

/// In-place Jacobi eigendecomposition of a 4×4 symmetric matrix.
/// On exit, `m` is diagonal (eigenvalues on the diagonal) and `v` is
/// the matrix of right eigenvectors (orthogonal). Caller initialises
/// `v` to the identity.
fn jacobi_4x4<T: FloatT>(m: &mut [[T; 4]; 4], v: &mut [[T; 4]; 4]) {
    let two = T::from(2.0).unwrap();
    let half = T::from(0.5).unwrap();
    let one = T::one();
    let max_sweeps = 30; // 4×4 converges in ~5 sweeps; cap for safety
    let tol = T::from(1e-15).unwrap();
    for _sweep in 0..max_sweeps {
        // Largest off-diagonal magnitude.
        let mut max_off = T::zero();
        for p in 0..4 {
            for q in (p + 1)..4 {
                let a = m[p][q].abs();
                if a > max_off {
                    max_off = a;
                }
            }
        }
        if max_off < tol {
            break;
        }
        for p in 0..4 {
            for q in (p + 1)..4 {
                let mpq = m[p][q];
                if mpq.abs() < tol {
                    continue;
                }
                let mpp = m[p][p];
                let mqq = m[q][q];
                // Compute rotation angle θ s.t. tan(2θ) = 2·mpq / (mpp − mqq).
                let theta = (mqq - mpp) / (two * mpq);
                let t = if theta >= T::zero() {
                    one / (theta + T::sqrt(one + theta * theta))
                } else {
                    one / (theta - T::sqrt(one + theta * theta))
                };
                let c = one / T::sqrt(one + t * t);
                let sn = t * c;
                // Update m: rotate rows/columns p, q.
                let new_pp = mpp - t * mpq;
                let new_qq = mqq + t * mpq;
                m[p][p] = new_pp;
                m[q][q] = new_qq;
                m[p][q] = T::zero();
                m[q][p] = T::zero();
                for i in 0..4 {
                    if i != p && i != q {
                        let mip = m[i][p];
                        let miq = m[i][q];
                        m[i][p] = c * mip - sn * miq;
                        m[p][i] = m[i][p];
                        m[i][q] = sn * mip + c * miq;
                        m[q][i] = m[i][q];
                    }
                }
                // Update v: accumulate rotation.
                for i in 0..4 {
                    let vip = v[i][p];
                    let viq = v[i][q];
                    v[i][p] = c * vip - sn * viq;
                    v[i][q] = sn * vip + c * viq;
                }
                let _ = half; // silence unused
            }
        }
    }
}

/// Build the dense ND PD-scaling matrix Hs (`n × n`, row-major).
///
/// `Hs = (s s')/⟨s,z⟩ + (δs δs')/⟨δs,δz⟩ + P_⊥ · (μ·H_dual) · P_⊥`,
/// where `P_⊥ = I − e1 e1' − e2 e2'` and `(e1, e2)` is an orthonormal
/// basis for `span{z, δz}`.
///
/// `mul_μH` computes `out = μ · H_dual · x` (same interface as the sparse
/// variant). Returns `false` if the guards fail and the caller should
/// fall back to dual-only scaling.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pd_scaling_nd_dense<T, MulMuH>(
    s: &[T],
    z: &[T],
    gz: &[T],
    gs: &[T],
    μ: T,
    mut mul_μH: MulMuH,
    ws: &mut PdScalingWorkspace<T>,
    hs_out: &mut [T],
) -> bool
where
    T: FloatT,
    MulMuH: FnMut(&mut [T], &[T]),
{
    let n = s.len();
    debug_assert_eq!(z.len(), n);
    debug_assert_eq!(hs_out.len(), n * n);

    let eps_sqrt = T::sqrt(T::epsilon());

    let sz = s.dot(z);
    if sz <= eps_sqrt {
        return false;
    }

    // δs, δz.
    let delta_s = &mut ws.delta_s[..n];
    for i in 0..n {
        delta_s[i] = s[i] + μ * gz[i];
    }
    let delta_z = &mut ws.delta_z[..n];
    for i in 0..n {
        delta_z[i] = z[i] + μ * gs[i];
    }
    let delta_inner = delta_s.dot(&*delta_z);
    if delta_inner <= eps_sqrt {
        return false;
    }

    // Gram-Schmidt orthonormal basis (e1, e2) for span{z, δz}.
    let z_norm = T::sqrt(z.dot(z));
    if z_norm <= eps_sqrt {
        return false;
    }
    let e1 = &mut ws.e_z[..n];
    for i in 0..n {
        e1[i] = z[i] / z_norm;
    }
    let dz_dot_e1 = delta_z.dot(&*e1);
    let e2 = &mut ws.e_dzp[..n];
    for i in 0..n {
        e2[i] = delta_z[i] - dz_dot_e1 * e1[i];
    }
    let e2_e1_residual = e2.dot(&*e1);
    for i in 0..n {
        e2[i] -= e2_e1_residual * e1[i];
    }
    let e2_norm = T::sqrt(e2.dot(&*e2));
    if e2_norm <= eps_sqrt {
        return false;
    }
    for v in e2.iter_mut() {
        *v /= e2_norm;
    }

    // Compute h1 = μ·H·e1, h2 = μ·H·e2 using the supplied closure.
    let h1 = &mut ws.h_z[..n];
    mul_μH(h1, e1);
    let h2 = &mut ws.h_dzp[..n];
    mul_μH(h2, e2);
    let q11 = e1.dot(&*h1);
    let q22 = e2.dot(&*h2);
    let q12 = e1.dot(&*h2); // = e2.dot(&h1) by symmetry of μ·H

    // We never materialise μ·H — instead build (P_⊥ μH P_⊥) cleanly via
    // the projector identity:
    //   P_⊥·H·P_⊥ = H − h1 e1' − e1 h1' − h2 e2' − e2 h2'
    //               + q11·e1 e1' + q22·e2 e2' + q12·(e1 e2' + e2 e1').
    // We materialise the FINAL Hs only — μ·H itself goes into the dense
    // block by computing μ·H·e_k for each canonical basis vector e_k.
    //
    // O(n³) total: n applies of mul_μH (n × O(n²) for rank-3 mul) + O(n²)
    // for the rank-update cleanup.
    let col_buf = &mut ws.col_buf[..n];
    let basis = &mut ws.basis[..n];
    for v in basis.iter_mut() {
        *v = T::zero();
    }
    for k in 0..n {
        if k > 0 {
            basis[k - 1] = T::zero();
        }
        basis[k] = T::one();
        mul_μH(col_buf, basis);
        for i in 0..n {
            hs_out[i * n + k] = col_buf[i];
        }
    }

    // Subtract h1·e1' + e1·h1' + h2·e2' + e2·h2'.
    for i in 0..n {
        for j in 0..n {
            hs_out[i * n + j] =
                hs_out[i * n + j] - h1[i] * e1[j] - e1[i] * h1[j] - h2[i] * e2[j] - e2[i] * h2[j];
        }
    }

    // Add q11·e1 e1' + q22·e2 e2' + q12·(e1 e2' + e2 e1').
    for i in 0..n {
        for j in 0..n {
            hs_out[i * n + j] = hs_out[i * n + j]
                + q11 * e1[i] * e1[j]
                + q22 * e2[i] * e2[j]
                + q12 * (e1[i] * e2[j] + e2[i] * e1[j]);
        }
    }

    // Add (1/⟨s,z⟩)·s s' + (1/⟨δs,δz⟩)·δs δs'.
    let c_sz = T::one() / sz;
    let c_dd = T::one() / delta_inner;
    for i in 0..n {
        for j in 0..n {
            hs_out[i * n + j] =
                hs_out[i * n + j] + c_sz * s[i] * s[j] + c_dd * delta_s[i] * delta_s[j];
        }
    }

    // Force exact symmetry. The construction above is mathematically
    // symmetric, but each rank-1 update uses a slightly different
    // float-rounding path on the (i,j) and (j,i) entries (especially
    // the `μH` columns produced via `mul_μH(e_k)`). An asymmetric Hs
    // breaks the IPM's quasidefinite KKT factorisation in subtle
    // ways — the LDL solver still succeeds but the resulting Newton
    // step is no longer a descent direction, so the IPM converges to
    // a non-optimal point that still passes residual checks.
    let half: T = (0.5).as_T();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = half * (hs_out[i * n + j] + hs_out[j * n + i]);
            hs_out[i * n + j] = avg;
            hs_out[j * n + i] = avg;
        }
    }

    true
}

#[cfg(test)]
mod nd_pd_scaling_tests {
    //! Tests use a real GenPower cone for the inputs because synthetic
    //! vectors don't satisfy the log-homogeneity orthogonality identities
    //! (`⟨δs, z⟩ = 0`, `⟨s, δz⟩ = 0`) the secant proofs depend on.
    use super::*;
    use crate::solver::core::cones::genpowcone::GenPowerCone;
    use crate::solver::core::cones::nonsymmetric_common::NonsymmetricNDCone;
    use crate::solver::core::cones::Cone;

    fn build_cone_state(
        alphas: Vec<f64>,
        dim2: usize,
        s: Vec<f64>,
        z: Vec<f64>,
    ) -> (
        GenPowerCone<f64>,
        Vec<f64>, // gz = ∇F*(z) (dual barrier gradient)
        Vec<f64>, // gs = ∇F(s)  (primal barrier gradient)
        f64,      // μ
        Vec<f64>, // s
        Vec<f64>, // z
    ) {
        let mut cone = GenPowerCone::new(alphas.clone(), dim2);
        // μ chosen so that ⟨s, z⟩ ≈ ν·μ (on central path the IPM picks this).
        let nu = (alphas.len() + 1) as f64;
        let μ = s.iter().zip(z.iter()).map(|(a, b)| a * b).sum::<f64>() / nu;
        // Populate gz = ∇F*(z) via update_dual_grad_H (writes into self.data.grad).
        let _ok = cone.update_scaling(&s, &z, μ, crate::solver::core::ScalingStrategy::Dual);
        let gz: Vec<f64> = cone.dual_grad().to_vec();
        // gs = ∇F(s) via gradient_primal.
        let mut gs = vec![0.0; s.len()];
        cone.gradient_primal(&mut gs, &s);
        (cone, gz, gs, μ, s, z)
    }

    #[test]
    fn dense_satisfies_both_secants_on_genpow() {
        let dim1 = 3;
        let dim2 = 2;
        let alphas = vec![1.0 / dim1 as f64; dim1];
        let s = vec![1.5_f64, 1.2, 0.9, 0.05, -0.04];
        let z = vec![1.4_f64, 1.3, 1.0, 0.04, -0.03];
        let (mut cone, gz, gs, μ, s, z) = build_cone_state(alphas, dim2, s, z);
        let n = s.len();

        let mut work = vec![0.0; n];
        let mul_μH = |out: &mut [f64], x: &[f64]| {
            cone.mul_Hs(out, x, &mut work);
        };

        let mut hs = vec![0.0; n * n];
        let mut ws = PdScalingWorkspace::<f64>::new(n);
        let ok = pd_scaling_nd_dense(&s, &z, &gz, &gs, μ, mul_μH, &mut ws, &mut hs);
        assert!(ok);

        // δs = s + μ·gz, δz = z + μ·gs.
        let delta_s: Vec<f64> = s
            .iter()
            .zip(gz.iter())
            .map(|(&si, &g)| si + μ * g)
            .collect();
        let delta_z: Vec<f64> = z
            .iter()
            .zip(gs.iter())
            .map(|(&zi, &g)| zi + μ * g)
            .collect();

        let mat_vec = |hs: &[f64], x: &[f64]| -> Vec<f64> {
            let mut y = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    y[i] += hs[i * n + j] * x[j];
                }
            }
            y
        };
        let hs_z = mat_vec(&hs, &z);
        let hs_dz = mat_vec(&hs, &delta_z);

        for i in 0..n {
            assert!(
                (hs_z[i] - s[i]).abs() < 1e-8,
                "(S1) Hs·z[{}] = {} expected {}",
                i,
                hs_z[i],
                s[i]
            );
            assert!(
                (hs_dz[i] - delta_s[i]).abs() < 1e-8,
                "(S2) Hs·δz[{}] = {} expected {}",
                i,
                hs_dz[i],
                delta_s[i]
            );
        }
    }

    #[test]
    fn qr6_satisfies_both_secants_on_genpow() {
        // Build Hs from the rank-6 QR sparse axes and verify Hs·z = s
        // and Hs·δz = δs (both exact under log-homogeneity).
        let dim1 = 3;
        let dim2 = 2;
        let alphas = vec![1.0 / dim1 as f64; dim1];
        let s = vec![1.5_f64, 1.2, 0.9, 0.05, -0.04];
        let z = vec![1.4_f64, 1.3, 1.0, 0.04, -0.03];
        let (mut cone, gz, gs, μ, s, z) = build_cone_state(alphas, dim2, s, z);
        let n = s.len();

        let mut work = vec![0.0; n];
        let mul_μH = |out: &mut [f64], x: &[f64]| {
            cone.mul_Hs(out, x, &mut work);
        };
        let mut ws = PdScalingWorkspace::<f64>::new(n);
        let mut r = PdScalingNdQr6::<f64>::zero(n);
        pd_scaling_nd_qr6(&s, &z, &gz, &gs, μ, mul_μH, &mut ws, &mut r);
        assert!(
            r.ok,
            "rank-6 axes should be ok on a strictly interior point"
        );

        // Build Hs explicitly: Hs = μH + Σ sign · coef · axis · axis'.
        let mut hs = vec![0.0_f64; n * n];
        let mut basis = vec![0.0; n];
        let mut col = vec![0.0; n];
        let mut work2 = vec![0.0; n];
        for k in 0..n {
            for v in basis.iter_mut() {
                *v = 0.0;
            }
            basis[k] = 1.0;
            cone.mul_Hs(&mut col, &basis, &mut work2);
            for i in 0..n {
                hs[i * n + k] = col[i];
            }
        }
        for k in 0..6 {
            let sign = r.signs[k] as f64;
            let c = sign * r.coefs[k];
            for i in 0..n {
                for j in 0..n {
                    hs[i * n + j] += c * r.axes[k][i] * r.axes[k][j];
                }
            }
        }

        let mat_vec = |hs: &[f64], x: &[f64]| -> Vec<f64> {
            let mut y = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    y[i] += hs[i * n + j] * x[j];
                }
            }
            y
        };

        // (S1) Hs·z = s.
        let hs_z = mat_vec(&hs, &z);
        for i in 0..n {
            assert!(
                (hs_z[i] - s[i]).abs() < 1e-8,
                "(S1) Hs·z[{}] = {} expected {}",
                i,
                hs_z[i],
                s[i]
            );
        }

        // (S2) Hs·δz = δs.
        let δs: Vec<f64> = s.iter().zip(gz.iter()).map(|(&a, &b)| a + μ * b).collect();
        let δz: Vec<f64> = z.iter().zip(gs.iter()).map(|(&a, &b)| a + μ * b).collect();
        let hs_δz = mat_vec(&hs, &δz);
        for i in 0..n {
            assert!(
                (hs_δz[i] - δs[i]).abs() < 1e-8,
                "(S2) Hs·δz[{}] = {} expected {}",
                i,
                hs_δz[i],
                δs[i]
            );
        }
    }

    #[test]
    fn dense_and_sparse_produce_same_hs() {
        // Both `pd_scaling_nd_dense` and `pd_scaling_nd_qr6` claim
        // to materialize the same Mosek-Tunçel scaling
        //   Hs = ss'/⟨s,z⟩ + δs δs'/⟨δs,δz⟩ + P_⊥·μH·P_⊥.
        // If they don't, the iter-count gap between the two paths is
        // a derivation bug, not KKT noise. This test compares them
        // directly, entry by entry.
        let dim1 = 3;
        let dim2 = 2;
        let alphas = vec![1.0 / dim1 as f64; dim1];
        let s = vec![1.5_f64, 1.2, 0.9, 0.05, -0.04];
        let z = vec![1.4_f64, 1.3, 1.0, 0.04, -0.03];
        let (mut cone, gz, gs, μ, s, z) = build_cone_state(alphas, dim2, s, z);
        let n = s.len();

        // Dense Hs.
        let mut hs_dense = vec![0.0; n * n];
        let mut work_a = vec![0.0; n];
        let mut ws_dense = PdScalingWorkspace::<f64>::new(n);
        {
            let mul = |out: &mut [f64], x: &[f64]| {
                cone.mul_Hs(out, x, &mut work_a);
            };
            assert!(pd_scaling_nd_dense(
                &s,
                &z,
                &gz,
                &gs,
                μ,
                mul,
                &mut ws_dense,
                &mut hs_dense
            ));
        }

        // Sparse Hs = μH (col-by-col) + Σ sign·coef·v·v'.
        let mut hs_sparse = vec![0.0; n * n];
        let mut basis = vec![0.0; n];
        let mut col = vec![0.0; n];
        let mut work_b = vec![0.0; n];
        for k in 0..n {
            for v in basis.iter_mut() {
                *v = 0.0;
            }
            basis[k] = 1.0;
            cone.mul_Hs(&mut col, &basis, &mut work_b);
            for i in 0..n {
                hs_sparse[i * n + k] = col[i];
            }
        }
        let mut work_c = vec![0.0; n];
        let mut ws = PdScalingWorkspace::<f64>::new(n);
        let mut r = PdScalingNdQr6::<f64>::zero(n);
        {
            let mul = |out: &mut [f64], x: &[f64]| {
                cone.mul_Hs(out, x, &mut work_c);
            };
            pd_scaling_nd_qr6(&s, &z, &gz, &gs, μ, mul, &mut ws, &mut r);
        }
        assert!(r.ok);
        for k in 0..6 {
            let sign = r.signs[k] as f64;
            let c = sign * r.coefs[k];
            for i in 0..n {
                for j in 0..n {
                    hs_sparse[i * n + j] += c * r.axes[k][i] * r.axes[k][j];
                }
            }
        }

        // Compare. Allow a tight tolerance — both materializations are
        // pure floating-point operations on the same inputs, so they
        // should agree to a few ulps.
        let mut max_diff = 0.0_f64;
        for i in 0..n {
            for j in 0..n {
                let d = (hs_dense[i * n + j] - hs_sparse[i * n + j]).abs();
                max_diff = max_diff.max(d);
            }
        }
        assert!(
            max_diff < 1e-10,
            "dense and sparse Hs disagree by {} — derivation bug",
            max_diff
        );
    }
}

// --------------------------------------
// utility functions for nonsymmetric cones
// --------------------------------------

// find the maximum step length α≥0 so that
// q + α*dq stays in an exponential or power
// cone, or their respective dual cones.
pub(crate) fn backtrack_search<T>(
    dq: &[T],
    q: &[T],
    α_init: T,
    α_min: T,
    step: T,
    is_in_cone_fcn: impl Fn(&[T]) -> bool,
    work: &mut [T],
) -> T
where
    T: FloatT,
{
    let mut α = α_init;

    loop {
        // work = q + α*dq
        work.waxpby(T::one(), q, α, dq);

        if is_in_cone_fcn(work) {
            break;
        }
        α *= step;
        if α < α_min {
            α = T::zero();
            break;
        }
    }
    α
}
pub(crate) fn newton_raphson_onesided<T>(x0: T, f0: impl Fn(T) -> T, f1: impl Fn(T) -> T) -> T
where
    T: FloatT,
{
    // implements NR method from a starting point assumed to be to the
    // left of the true value.   Once a negative step is encountered
    // this function will halt regardless of the calculated correction.

    let mut x = x0;
    let mut iter = 0;

    while iter < 100 {
        iter += 1;
        let dfdx = f1(x);
        let dx = -f0(x) / dfdx;

        if (dx < T::epsilon())
            || (T::abs(dx / x) < T::sqrt(T::epsilon()))
            || (T::abs(dfdx) < T::epsilon())
        {
            break;
        }
        x += dx;
    }

    x
}
