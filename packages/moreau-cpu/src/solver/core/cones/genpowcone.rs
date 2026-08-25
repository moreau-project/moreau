use super::*;
use crate::algebra::*;
use itertools::izip;
use std::iter::zip;

// -------------------------------------
// Generalized Power Cone
// -------------------------------------

pub struct GenPowerConeData<T: FloatT> {
    // gradient of the dual barrier at z
    grad: Vec<T>,
    // holds copy of z at scaling point
    z: Vec<T>,

    // central path parameter
    pub μ: T,

    // vectors for rank 3 update representation of Hs
    pub p: Vec<T>,
    pub q: Vec<T>,
    pub r: Vec<T>,
    pub d1: Vec<T>,

    // additional scalar terms for rank-2 rep
    d2: T,
    // additional constant for initialization in the Newton-Raphson method
    ψ: T,

    //work vector length dim, e.g. for line searches
    work: Vec<T>,
    //work vectors for Mehrotra higher_correction (length dim each)
    correction_u: Vec<T>,
    correction_gpsi: Vec<T>,

    // Persistent workspace for smoothing Newton iteration
    smoothing_res: Vec<T>,
    smoothing_hmat: Vec<T>,
    smoothing_lmat: Vec<T>,
    smoothing_delta: Vec<T>,

    // ND Mosek-Tunçel primal-dual scaling — rank-6 sparse expansion
    // (4 projector axes from QR+eigendecomposition + 2 secant axes).
    // See `nonsymmetric_common::PdScalingNdQr6` for layout details.
    // When `pd_active` is true the cone's `Hs` is
    //   Hs = μ·H_dual + Σ_{k=0..6} pd_signs[k] · pd_coefs[k] · pd_axes[k] · pd_axes[k]'
    // (`pd_coefs[k] ≥ 0`; sign carried in `pd_signs[k]`). Otherwise
    // these are zero and `Hs = μ·H_dual` (dual-only fallback).
    pub pd_axes: Vec<Vec<T>>,
    pub pd_coefs: Vec<T>,
    pub pd_signs: [i8; 6],
    pub pd_active: bool,

    // Dense Hs storage (row-major, length `dim*dim`). Used when the
    // cone uses the dense KKT path (`is_dense_path == true` on the
    // outer cone struct), holding either `μ·H_dual` (Dual scaling) or
    // the full Mosek-Tunçel form satisfying both (S1) and (S2)
    // (PrimalDual scaling). Empty (length 0) on the sparse path.
    pub dense_hs: Vec<T>,

    // Direct-x mode: ∇F_primal(x) at the scaling point. Used by the
    // centering-shift formula `σμ·∇F(x)`.
    pub grad_primal: Vec<T>,
    // Direct-x mode: copy of x at the scaling point.
    pub x_pt: Vec<T>,

    // Per-IPM-iter scratch for the rank-6 + dense PD scaling builders.
    // Reused across iterations: no allocations inside iteration loops.
    pub(crate) pd_scratch: crate::solver::core::cones::nonsymmetric_common::PdScalingWorkspace<T>,
    // Preallocated rank-6 output reused by pd_scaling_nd_qr6 — avoids
    // allocating 6×n axis vectors per IPM iter per genpow cone.
    pub(crate) pd_qr6_out: crate::solver::core::cones::nonsymmetric_common::PdScalingNdQr6<T>,
    // Preallocated n*n scratch for `pd_scaling_nd_dense` Hs output.
    // Lives as a sibling of `pd_scratch` so a single call site can hold
    // `&mut pd_scratch` and `&mut pd_dense_hs_tmp[..n*n]` simultaneously
    // via disjoint-field borrow rules.
    pub pd_dense_hs_tmp: Vec<T>,
}

impl<T> GenPowerConeData<T>
where
    T: FloatT,
{
    pub fn new(α: &[T], dim2: usize, dense_path: bool) -> Self {
        let dim1 = α.len();
        let dim = dim1 + dim2;

        // Defense-in-depth: solver.rs validates these before construction
        debug_assert!(α.iter().all(|r| *r > T::zero()));
        debug_assert!({
            let base: T = (1e-8).as_T();
            (T::one() - α.sum()).abs() < base * α.len().as_T()
        });

        Self {
            grad: vec![T::zero(); dim],
            z: vec![T::zero(); dim],
            μ: T::one(),
            p: vec![T::zero(); dim],
            q: vec![T::zero(); dim1],
            r: vec![T::zero(); dim2],
            d1: vec![T::zero(); dim1],
            d2: T::zero(),
            ψ: T::one() / (α.sumsq()),
            work: vec![T::zero(); dim],
            correction_u: vec![T::zero(); dim],
            correction_gpsi: vec![T::zero(); dim],
            // Persistent workspace for smoothing Newton iteration (avoids per-call allocation)
            smoothing_res: vec![T::zero(); dim],
            smoothing_hmat: vec![T::zero(); dim * dim],
            smoothing_lmat: vec![T::zero(); dim * dim],
            smoothing_delta: vec![T::zero(); dim],
            // PD-scaling axes — start zeroed/inactive. Signs default
            // to all `+1`; per-iteration sign flips happen in
            // `try_compute_pd_axes` based on eigenvalue signs.
            pd_axes: (0..6).map(|_| vec![T::zero(); dim]).collect(),
            pd_coefs: vec![T::zero(); 6],
            pd_signs: crate::solver::core::cones::nonsymmetric_common::PD_QR6_DEFAULT_SIGNS,
            pd_active: false,
            // Dense Hs only allocated for the dense-path cone.
            dense_hs: if dense_path {
                vec![T::zero(); dim * dim]
            } else {
                Vec::new()
            },
            // Direct-x state (allocated upfront; unused in slack mode).
            grad_primal: vec![T::zero(); dim],
            x_pt: vec![T::zero(); dim],
            // Preallocated workspace for the PD scaling builders.
            pd_scratch: crate::solver::core::cones::nonsymmetric_common::PdScalingWorkspace::new(
                dim,
            ),
            pd_qr6_out: crate::solver::core::cones::nonsymmetric_common::PdScalingNdQr6::zero(dim),
            pd_dense_hs_tmp: vec![T::zero(); dim * dim],
        }
    }
}

/// Threshold below which a GenPow cone uses a *dense* `Hs` block
/// (full `n × n` Mosek-Tunçel matrix, both secants exact) instead of
/// the rank-3 sparse expansion plus rank-9 PD axes.
pub(crate) const N_DENSE_GENPOW: usize = 64;

fn dense_genpow_threshold() -> usize {
    N_DENSE_GENPOW
}

pub struct GenPowerCone<T: FloatT> {
    pub α: Vec<T>, // power defining the cone.  length determines dim1
    dim2: usize,   // dimension of w
    /// True when this cone routes its Hs through the dense KKT block.
    /// Determined at construction time by `dim() ≤ N_DENSE_GENPOW`; this
    /// is the only setting that gets the full Mosek-Tunçel primal-dual
    /// scaling.
    is_dense_path: bool,
    pub data: Box<GenPowerConeData<T>>, // Boxed so that the enum_dispatch variant isn't huge
}

impl<T> GenPowerCone<T>
where
    T: FloatT,
{
    pub fn new(α: Vec<T>, dim2: usize) -> Self {
        let dim = α.len() + dim2;
        let is_dense_path = dim <= dense_genpow_threshold();
        let data = Box::new(GenPowerConeData::<T>::new(&α, dim2, is_dense_path));
        Self {
            α,
            dim2,
            is_dense_path,
            data,
        }
    }

    pub fn dim1(&self) -> usize {
        self.α.len()
    }
    pub fn dim2(&self) -> usize {
        self.dim2
    }
    pub fn dim(&self) -> usize {
        self.dim1() + self.dim2()
    }

    #[cfg(test)]
    pub fn dual_grad(&self) -> &[T] {
        &self.data.grad
    }

    /// Try to compute the rank-9 sparse PD-scaling axes at `(s, z)`.
    /// Assumes `update_dual_grad_H(z)`, `data.μ`, and `data.z` have
    /// already been refreshed by the caller. Returns `true` on success;
    /// on numerical failure returns `false` and the caller should call
    /// [`zero_pd_axes`] to keep the cone in a clean dual-only state.
    ///
    /// **Local μ for δs/δz only.** The Mosek-Tunçel orthogonality
    /// `⟨δs,z⟩ = 0` follows from log-homogeneity (`⟨gz,z⟩ = -ν`)
    /// only when `μ_local = ⟨s,z⟩/ν`. moreau's IPM passes the HSDE
    /// `μ = (⟨s,z⟩ + τκ)/(ν+1)` into `update_scaling`, which is off
    /// by a factor of roughly `1/(ν+1)` — enough to push the secant
    /// residual to ~1e-3 on dim-100+ problems. We recompute
    /// `μ_local` here and use it for `δs`, `δz`. The `μ·H` filler
    /// (`R = P_⊥·μH·P_⊥`) stays at the IPM's `μ` because `R·z = 0`
    /// regardless — and that's what the cone's `mul_Hs` actually
    /// uses, so consistency requires it. The 3D Mosek-Tunçel code
    /// uses the same trick (`nonsymmetric_common.rs:85-86`).
    fn try_compute_pd_axes(&mut self, s: &[T]) -> bool {
        use crate::solver::core::cones::nonsymmetric_common::pd_scaling_nd_qr6;

        let dim1 = self.dim1();
        let dim2 = self.dim2();
        let n = dim1 + dim2;

        let mut gs = vec![T::zero(); n];
        self.gradient_primal(&mut gs, s);

        let z = self.data.z.clone();
        let gz = self.data.grad.clone();
        let μ_ipm = self.data.μ;
        let ν = T::from_usize(self.degree()).unwrap();
        let sz = s.dot(&z[..]);
        if sz <= T::zero() {
            return false;
        }
        let μ_local = sz / ν;

        // Disjoint-field borrows: pd_scratch + pd_qr6_out (mut) and the
        // rank-3 components p/q/r/d1/d2 (imm) borrow distinct fields of
        // `*self.data` so they coexist. Closure captures the field refs
        // (not `&self.data` as a whole), routing through
        // `mul_μH_rank3_parts` instead of `mul_μH_rank3`.
        {
            let p_ax = &self.data.p;
            let q_ax = &self.data.q;
            let r_ax = &self.data.r;
            let d1 = &self.data.d1;
            let d2_v = self.data.d2;
            let scratch = &mut self.data.pd_scratch;
            let out_qr6 = &mut self.data.pd_qr6_out;
            let mul_μH = |out: &mut [T], x: &[T]| {
                mul_μH_rank3_parts(p_ax, q_ax, r_ax, d1, d2_v, dim1, μ_ipm, out, x);
            };
            pd_scaling_nd_qr6(s, &z, &gz, &gs, μ_local, mul_μH, scratch, out_qr6);
        }
        if !self.data.pd_qr6_out.ok {
            return false;
        }
        // Bind `r` to the preallocated output so the rest of the function
        // (which reads r.coefs / r.axes / r.signs) works unchanged.
        let r = &self.data.pd_qr6_out;

        // Verify both secants Hs·z = s and Hs·δz = δs. The check
        // re-materialises Hs via the rank-1 outer-product sum, which
        // accumulates FP error proportional to `max|λ_k|` (eigenvalues
        // of the 4×4 projector matrix M). At late IPM iters near the
        // cone boundary |λ_k| grows ∝ ‖μH‖ even as `μ → 0`, so the
        // tolerance scales with `ulp · n · max|λ_k|`: catches actual
        // math errors but doesn't trip on FP noise from huge λ.
        // When the guard fires we fall back to dual scaling for this
        // iter — the IPM gets back on track in subsequent iters once
        // the iterate moves away from the bad-conditioning region.
        let mut s_norm_sq = T::zero();
        for &v in s.iter() {
            s_norm_sq += v * v;
        }
        let s_norm = T::sqrt(s_norm_sq);

        let mut δs = vec![T::zero(); n];
        let mut δz = vec![T::zero(); n];
        for i in 0..n {
            δs[i] = s[i] + μ_local * gz[i];
            δz[i] = z[i] + μ_local * gs[i];
        }
        let mut δs_norm_sq = T::zero();
        for &v in δs.iter() {
            δs_norm_sq += v * v;
        }
        let δs_norm = T::sqrt(δs_norm_sq);

        let mut hs_z = vec![T::zero(); n];
        mul_μH_rank3(&self.data, dim1, dim2, μ_ipm, &mut hs_z, &z);
        let mut max_coef = T::zero();
        for k in 0..6 {
            let coef = r.coefs[k];
            if coef > max_coef {
                max_coef = coef;
            }
            if coef == T::zero() {
                continue;
            }
            let dot = r.axes[k].dot(&z[..]);
            let sign: T = T::from_i8(r.signs[k]).unwrap();
            let scale = sign * coef * dot;
            for i in 0..n {
                hs_z[i] += scale * r.axes[k][i];
            }
        }
        let mut err_z_sq = T::zero();
        for i in 0..n {
            let d = hs_z[i] - s[i];
            err_z_sq += d * d;
        }

        let mut hs_dz = vec![T::zero(); n];
        mul_μH_rank3(&self.data, dim1, dim2, μ_ipm, &mut hs_dz, &δz);
        for k in 0..6 {
            let coef = r.coefs[k];
            if coef == T::zero() {
                continue;
            }
            let dot = r.axes[k].dot(&δz[..]);
            let sign: T = T::from_i8(r.signs[k]).unwrap();
            let scale = sign * coef * dot;
            for i in 0..n {
                hs_dz[i] += scale * r.axes[k][i];
            }
        }
        let mut err_dz_sq = T::zero();
        for i in 0..n {
            let d = hs_dz[i] - δs[i];
            err_dz_sq += d * d;
        }

        // Tolerance: max(1e-7, √n · ulp · max|λ|). The first term is
        // the "real" math-error threshold; the second is the FP noise
        // floor of the rank-1 reconstruction. Per-entry rounding is
        // O(ulp · max|λ|); summing n entries gives O(√n · ulp · max|λ|)
        // by the standard random-walk bound on accumulated FP error.
        let n_t = T::from_usize(n).unwrap();
        let fp_floor = T::sqrt(n_t) * T::epsilon() * max_coef;
        let tol_secant: T = (1e-7).as_T();
        let tol = if fp_floor > tol_secant {
            fp_floor
        } else {
            tol_secant
        };
        let r_z = T::sqrt(err_z_sq) / s_norm.max(T::one());
        let r_dz = T::sqrt(err_dz_sq) / δs_norm.max(T::one());
        if r_z > tol {
            return false;
        }
        if r_dz > tol {
            return false;
        }

        // Stash axes, coefficients, and per-iteration signs on the cone.
        for k in 0..6 {
            self.data.pd_axes[k].copy_from_slice(&r.axes[k]);
            self.data.pd_coefs[k] = r.coefs[k];
            self.data.pd_signs[k] = r.signs[k];
        }
        self.data.pd_active = true;
        true
    }

    fn zero_pd_axes(&mut self) {
        let n = self.dim();
        for k in 0..6 {
            for i in 0..n {
                self.data.pd_axes[k][i] = T::zero();
            }
            self.data.pd_coefs[k] = T::zero();
        }
        // Reset signs to defaults (all +1). pd_active = false means
        // the LDL diagonal slots carry only the structural ±1 with no
        // off-diagonal contribution, so signs don't affect the math.
        self.data.pd_signs = crate::solver::core::cones::nonsymmetric_common::PD_QR6_DEFAULT_SIGNS;
        self.data.pd_active = false;
    }

    /// Direct-x analog of `try_compute_pd_axes` operating on the *primal*
    /// barrier. The Mosek-Tunçel rank-6 expansion is identical in structure
    /// to slack's; only the substitution map changes:
    ///
    /// ```text
    ///   slack:    s, z,  gz = ∇F*(z),  gs = ∇F(s),   mul_μH_dual
    ///   direct-x: x, z_x, g_zx = ∇F*(z_x), g_x = ∇F(x), mul_μH_primal
    /// ```
    ///
    /// Both forms enforce the secant identities `H·x = z_x` (S1) and
    /// `H·δx = δz_x` (S2). After this returns true, `(d1, d2, p, q, r,
    /// grad_primal)` hold the primal-barrier rank-3 decomposition at `x`
    /// and `pd_axes/pd_coefs/pd_signs/pd_active` carry the 6 axes that
    /// extend `μ·H_primal` to the full Mosek-Tunçel scaling.
    ///
    /// On numerical failure (any secant denominator collapses, projector
    /// basis is rank-deficient, or secant verification trips) returns
    /// false; `(d1, ..., grad_primal)` are still populated for the
    /// dual-only fallback (`Hs = μ·H_primal`), and `pd_active` is left
    /// false.
    fn try_compute_primal_pd_axes(&mut self, x: &[T], z_x: &[T], μ_ipm: T) -> bool {
        use crate::solver::core::cones::nonsymmetric_common::pd_scaling_nd_qr6;

        let dim1 = self.dim1();
        let dim2 = self.dim2();
        let n = dim1 + dim2;
        let two: T = (2.).as_T();

        // ---------- Step A: g_zx = ∇F*(z_x) into a local buffer. ----------
        // Same formula as `update_dual_grad_H`'s gradient block, but written
        // out here so we don't clobber `(d1, d2, p, q, r, grad)` — those
        // fields will be re-used by `update_primal_grad_H(x)` below for the
        // primal-barrier rank-3 decomposition.
        let mut g_zx = vec![T::zero(); n];
        {
            let α = &self.α;
            let log_phi =
                zip(α, z_x).fold(T::zero(), |acc, (&αi, &zi)| acc + two * αi * (zi / αi).ln());
            let phi = log_phi.exp();
            let norm2w = z_x[dim1..].sumsq();
            let ζ = phi - norm2w;
            if !(ζ > T::zero()) {
                return false;
            }
            for (g, &αi, &zi) in izip!(g_zx[..dim1].iter_mut(), α, &z_x[..dim1]) {
                let τi = two * αi / zi;
                *g = -τi * phi / ζ - (T::one() - αi) / zi;
            }
            for (g, &zi) in izip!(g_zx[dim1..].iter_mut(), &z_x[dim1..]) {
                *g = (two / ζ) * zi;
            }
        }

        // ---------- Step B: primal rank-3 decomposition at x. ----------
        // Populates (d1, d2, p, q, r, grad_primal) with the primal barrier
        // formulas — the role-flipped analog of slack's update_dual_grad_H.
        if !self.update_primal_grad_H(x) {
            return false;
        }
        let g_x = self.data.grad_primal.clone();

        // ---------- Step C: μ_local = ⟨x, z_x⟩ / ν ----------
        // Same HSDE-μ correction as slack — see the long comment above
        // `try_compute_pd_axes` for why we use μ_local instead of the
        // IPM's HSDE μ for δx, δz_x. The `μ·H` filler still uses μ_ipm.
        let xz = x.dot(z_x);
        if !(xz > T::zero()) {
            return false;
        }
        let ν = T::from_usize(self.degree()).unwrap();
        let μ_local = xz / ν;

        // ---------- Step D: rank-6 PD scaling builder (primal side). ----------
        // Direct-x swap: helper expects slack args (s, z, gz, gs); we feed
        // (z_x, x, g_x, g_zx) so it produces Hs satisfying Hs·x = z_x
        // (the right Newton-coefficient direction for direct-x; see
        // `try_build_dense_pd_hs_primal` for the full justification).
        {
            let p_ax = &self.data.p;
            let q_ax = &self.data.q;
            let r_ax = &self.data.r;
            let d1 = &self.data.d1;
            let d2_v = self.data.d2;
            let scratch = &mut self.data.pd_scratch;
            let out_qr6 = &mut self.data.pd_qr6_out;
            let mul_μH_primal = |out: &mut [T], y: &[T]| {
                mul_μH_rank3_parts(p_ax, q_ax, r_ax, d1, d2_v, dim1, μ_ipm, out, y);
            };
            pd_scaling_nd_qr6(
                z_x,
                x,
                &g_x,
                &g_zx,
                μ_local,
                mul_μH_primal,
                scratch,
                out_qr6,
            );
        }
        if !self.data.pd_qr6_out.ok {
            return false;
        }
        let r = &self.data.pd_qr6_out;
        let _ = dim2; // silence

        // ---------- Step E: secant verification: H·x = z_x and H·δx = δz_x. ----------
        let mut z_norm_sq = T::zero();
        for &v in z_x.iter() {
            z_norm_sq += v * v;
        }
        let z_norm = T::sqrt(z_norm_sq);

        let mut δx = vec![T::zero(); n];
        let mut δz_x = vec![T::zero(); n];
        for i in 0..n {
            δx[i] = x[i] + μ_local * g_zx[i];
            δz_x[i] = z_x[i] + μ_local * g_x[i];
        }
        let mut δz_norm_sq = T::zero();
        for &v in δz_x.iter() {
            δz_norm_sq += v * v;
        }
        let δz_norm = T::sqrt(δz_norm_sq);

        // H·x: μ·H_primal·x first, then add PD-axis contributions.
        let mut h_x = vec![T::zero(); n];
        mul_μH_rank3(&self.data, dim1, dim2, μ_ipm, &mut h_x, x);
        let mut max_coef = T::zero();
        for k in 0..6 {
            let coef = r.coefs[k];
            if coef > max_coef {
                max_coef = coef;
            }
            if coef == T::zero() {
                continue;
            }
            let dot = r.axes[k].dot(x);
            let sign: T = T::from_i8(r.signs[k]).unwrap();
            let scale = sign * coef * dot;
            for i in 0..n {
                h_x[i] += scale * r.axes[k][i];
            }
        }
        let mut err_x_sq = T::zero();
        for i in 0..n {
            let d = h_x[i] - z_x[i];
            err_x_sq += d * d;
        }

        // H·δx.
        let mut h_dx = vec![T::zero(); n];
        mul_μH_rank3(&self.data, dim1, dim2, μ_ipm, &mut h_dx, &δx);
        for k in 0..6 {
            let coef = r.coefs[k];
            if coef == T::zero() {
                continue;
            }
            let dot = r.axes[k].dot(&δx);
            let sign: T = T::from_i8(r.signs[k]).unwrap();
            let scale = sign * coef * dot;
            for i in 0..n {
                h_dx[i] += scale * r.axes[k][i];
            }
        }
        let mut err_dx_sq = T::zero();
        for i in 0..n {
            let d = h_dx[i] - δz_x[i];
            err_dx_sq += d * d;
        }

        // FP tolerance: max(1e-7, √n · ulp · max|λ|). Same logic as slack.
        let n_t = T::from_usize(n).unwrap();
        let fp_floor = T::sqrt(n_t) * T::epsilon() * max_coef;
        let tol_secant: T = (1e-7).as_T();
        let tol = if fp_floor > tol_secant {
            fp_floor
        } else {
            tol_secant
        };
        let r_x = T::sqrt(err_x_sq) / z_norm.max(T::one());
        let r_dx = T::sqrt(err_dx_sq) / δz_norm.max(T::one());
        if r_x > tol {
            return false;
        }
        if r_dx > tol {
            return false;
        }

        // ---------- Step F: stash axes/coefs/signs. ----------
        for k in 0..6 {
            self.data.pd_axes[k].copy_from_slice(&r.axes[k]);
            self.data.pd_coefs[k] = r.coefs[k];
            self.data.pd_signs[k] = r.signs[k];
        }
        self.data.pd_active = true;
        true
    }

    /// Build `data.dense_hs = μ·H_primal(x) + Σ pd_signs[k]·pd_coefs[k]·
    /// axes_k axes_k'` from the rank-3 fields populated by
    /// [`Self::update_primal_grad_H`] (or by [`Self::try_compute_primal_pd_axes`],
    /// which calls `update_primal_grad_H` internally) and the optional
    /// rank-6 PD axes.
    ///
    /// Direct-x mirror of [`Self::build_dense_dual_hs`]. The body is
    /// identical because the rank-3 form `H = D + p p' − q q' − r r'` has
    /// the same structural shape on both sides; only the *source*
    /// barrier (and hence the populated values of `(d1, d2, p, q, r)`)
    /// differs. When `pd_active`, the rank-6 axes contribute the same
    /// `Σ sign·coef·v v'` correction as in [`Self::mul_Hs`].
    ///
    /// Required because `direct_x_update_scaling` populates only the
    /// rank-3 fields; without this call, dense-path direct-x GenPow
    /// (cone dim ≤ `N_DENSE_GENPOW`) would feed a zeroed `dense_hs`
    /// into `direct_x_get_Hs`, giving the IPM a wrong (1,1) Hs block.
    fn build_dense_primal_hs(&mut self) {
        let n = self.dim();
        let dim1 = self.dim1();
        let dim2 = self.dim2();
        let μ = self.data.μ;
        let data = &mut self.data;

        for v in data.dense_hs.iter_mut() {
            *v = T::zero();
        }
        for i in 0..dim1 {
            data.dense_hs[i * n + i] = μ * data.d1[i];
        }
        for i in 0..dim2 {
            data.dense_hs[(dim1 + i) * n + (dim1 + i)] = μ * data.d2;
        }
        for i in 0..n {
            for j in 0..n {
                data.dense_hs[i * n + j] += μ * data.p[i] * data.p[j];
            }
        }
        for i in 0..dim1 {
            for j in 0..dim1 {
                data.dense_hs[i * n + j] -= μ * data.q[i] * data.q[j];
            }
        }
        for i in 0..dim2 {
            for j in 0..dim2 {
                data.dense_hs[(dim1 + i) * n + (dim1 + j)] -= μ * data.r[i] * data.r[j];
            }
        }

        // Rank-6 PD axes contribution when active.
        if data.pd_active {
            for k in 0..6 {
                let coef = data.pd_coefs[k];
                if coef == T::zero() {
                    continue;
                }
                let sign = T::from_i8(data.pd_signs[k]).unwrap();
                let scale = sign * coef;
                for i in 0..n {
                    let ai = data.pd_axes[k][i];
                    if ai == T::zero() {
                        continue;
                    }
                    for j in 0..n {
                        data.dense_hs[i * n + j] += scale * ai * data.pd_axes[k][j];
                    }
                }
            }
        }
    }

    /// Build the dense `data.dense_hs = μ · H_dual` from the rank-3
    /// representation. Used on the dual scaling fallback.
    fn build_dense_dual_hs(&mut self) {
        let n = self.dim();
        let dim1 = self.dim1();
        let dim2 = self.dim2();
        let μ = self.data.μ;
        let data = &mut self.data;

        // Zero, then write μ·D on the diagonal.
        for v in data.dense_hs.iter_mut() {
            *v = T::zero();
        }
        for i in 0..dim1 {
            data.dense_hs[i * n + i] = μ * data.d1[i];
        }
        for i in 0..dim2 {
            data.dense_hs[(dim1 + i) * n + (dim1 + i)] = μ * data.d2;
        }
        // + μ·p p'
        for i in 0..n {
            for j in 0..n {
                data.dense_hs[i * n + j] += μ * data.p[i] * data.p[j];
            }
        }
        // − μ·q q'  (q has length dim1)
        for i in 0..dim1 {
            for j in 0..dim1 {
                data.dense_hs[i * n + j] -= μ * data.q[i] * data.q[j];
            }
        }
        // − μ·r r'  (r has length dim2; placed in bottom-right block)
        for i in 0..dim2 {
            for j in 0..dim2 {
                data.dense_hs[(dim1 + i) * n + (dim1 + j)] -= μ * data.r[i] * data.r[j];
            }
        }
        data.pd_active = false;
    }

    /// Try to build the dense Mosek-Tunçel Hs satisfying both secants.
    /// Returns false on numerical failure (caller falls back to dual).
    /// See `try_compute_pd_axes` for why we use `μ_local = ⟨s,z⟩/ν`
    /// for `δs`/`δz` rather than the IPM's HSDE μ.
    fn try_build_dense_pd_hs(&mut self, s: &[T]) -> bool {
        use crate::solver::core::cones::nonsymmetric_common::pd_scaling_nd_dense;

        let dim1 = self.dim1();
        let dim2 = self.dim2();
        let n = dim1 + dim2;

        // gz = ∇F*(z) is in self.data.grad; gs = ∇F(s) via NR.
        let mut gs = vec![T::zero(); n];
        self.gradient_primal(&mut gs, s);

        let z = self.data.z.clone();
        let gz = self.data.grad.clone();
        let μ_ipm = self.data.μ;
        let ν = T::from_usize(self.degree()).unwrap();
        let sz = s.dot(&z[..]);
        if sz <= T::zero() {
            return false;
        }
        let μ_local = sz / ν;

        // Disjoint-field borrows so `pd_scratch`, the n*n `pd_dense_hs_tmp`
        // output buffer, and imm refs to the rank-3 components coexist.
        // Closure uses `mul_μH_rank3_parts` to capture field refs (p/q/r/
        // d1/d2) rather than `&self.data`, so it stays compatible with the
        // mut workspace borrow.
        let ok = {
            let p_ax = &self.data.p;
            let q_ax = &self.data.q;
            let r_ax = &self.data.r;
            let d1 = &self.data.d1;
            let d2_v = self.data.d2;
            let scratch = &mut self.data.pd_scratch;
            let hs_tmp = &mut self.data.pd_dense_hs_tmp[..n * n];
            let mul_μH = |out: &mut [T], x: &[T]| {
                mul_μH_rank3_parts(p_ax, q_ax, r_ax, d1, d2_v, dim1, μ_ipm, out, x);
            };
            pd_scaling_nd_dense(s, &z, &gz, &gs, μ_local, mul_μH, scratch, hs_tmp)
        };
        if !ok {
            return false;
        }
        let _ = dim2; // silence

        // Verify both secants. If `Hs·z` deviates from `s` by more than
        // ~√ε relative, the dense form is numerically off and we'd
        // rather take the dual-only fallback than feed a bad Newton
        // step into the IPM (which can stall the solver at a non-optimal
        // point that still passes residual checks).
        let mut s_norm_sq = T::zero();
        for &v in s.iter() {
            s_norm_sq += v * v;
        }
        let s_norm = T::sqrt(s_norm_sq);

        let mut δs = vec![T::zero(); n];
        let mut δz = vec![T::zero(); n];
        for i in 0..n {
            δs[i] = s[i] + μ_local * gz[i];
            δz[i] = z[i] + μ_local * gs[i];
        }
        let mut δs_norm_sq = T::zero();
        for &v in δs.iter() {
            δs_norm_sq += v * v;
        }
        let δs_norm = T::sqrt(δs_norm_sq);
        let tol_secant: T = (1e-7).as_T();

        let mut hs_z_err = T::zero();
        for i in 0..n {
            let mut row = T::zero();
            for j in 0..n {
                row += self.data.pd_dense_hs_tmp[i * n + j] * z[j];
            }
            let d = row - s[i];
            hs_z_err += d * d;
        }
        let hs_z_err = T::sqrt(hs_z_err);
        if hs_z_err > tol_secant * s_norm.max(T::one()) {
            return false;
        }

        let mut hs_δz_err = T::zero();
        for i in 0..n {
            let mut row = T::zero();
            for j in 0..n {
                row += self.data.pd_dense_hs_tmp[i * n + j] * δz[j];
            }
            let d = row - δs[i];
            hs_δz_err += d * d;
        }
        let hs_δz_err = T::sqrt(hs_δz_err);
        if hs_δz_err > tol_secant * δs_norm.max(T::one()) {
            return false;
        }

        // Disjoint-field copy: dense_hs and pd_dense_hs_tmp are sibling
        // fields of self.data, so &mut + & against them coexist.
        let data = &mut *self.data;
        data.dense_hs[..n * n].copy_from_slice(&data.pd_dense_hs_tmp[..n * n]);
        self.data.pd_active = true;
        true
    }

    /// Direct-x dense PD-scaling builder. Used when the cone
    /// dim ≤ N_DENSE_GENPOW.
    ///
    /// Slack form: `pd_scaling_nd_dense` builds Hs satisfying `Hs·z = s`
    /// and `Hs·δz = δs`, with base `P_⊥·(μ·H_dual)·P_⊥`. The slack KKT
    /// (1,1) block uses `A^T·Hs⁻¹·A`, so the IPM sees `Hs⁻¹` and the
    /// natural identity is `Hs⁻¹·s = z` — primal-to-dual.
    ///
    /// Direct-x form: the (1,1) block uses `Hs` directly with no inverse,
    /// and the centrality linearisation `δz_x + μ·H_F(x)·δx = …` makes
    /// `μ·H_F(x)` the natural Newton coefficient. The PD scaling we want
    /// satisfies `Hs·x = z_x` and `Hs·δx = δz_x` (primal-to-dual,
    /// matching the linearisation), with base `P_⊥·(μ·H_primal)·P_⊥`
    /// where `P_⊥` projects orthogonal to span{x, δx}.
    ///
    /// We get this by calling the slack helper with the swap
    /// `(s, z, gz, gs) → (z_x, x, g_x, g_zx)` and `mul_μH = μ·H_primal`.
    /// Under that swap the helper builds:
    ///   Hs = (z_x z_x')/⟨z_x, x⟩ + (δz_x δz_x')/⟨δz_x, δx⟩
    ///        + P_⊥·(μ·H_primal)·P_⊥,   P_⊥ ⊥ span{x, δx}
    /// satisfying both secants in the right direction.
    ///
    /// Empirically the previous (wrong-direction) call stalled the IPM
    /// on 3D dense-path GenPow (any α) and 5D dense-path GenPow with
    /// skewed α: gap fell ~6% per iter with α-step ping-ponging between
    /// 0.5 and 0.99 — a classic sign of a wrong Newton coefficient.
    fn try_build_dense_pd_hs_primal(&mut self, x: &[T], z_x: &[T], μ_ipm: T) -> bool {
        use crate::solver::core::cones::nonsymmetric_common::pd_scaling_nd_dense;

        let dim1 = self.dim1();
        let dim2 = self.dim2();
        let n = dim1 + dim2;
        let two: T = (2.).as_T();

        // g_zx = ∇F*(z_x) (gradient of conjugate dual barrier at z_x).
        let mut g_zx = vec![T::zero(); n];
        {
            let α = &self.α;
            let log_phi =
                zip(α, z_x).fold(T::zero(), |acc, (&αi, &zi)| acc + two * αi * (zi / αi).ln());
            let phi = log_phi.exp();
            let norm2w = z_x[dim1..].sumsq();
            let ζ = phi - norm2w;
            if !(ζ > T::zero()) {
                return false;
            }
            for (g, &αi, &zi) in izip!(g_zx[..dim1].iter_mut(), α, &z_x[..dim1]) {
                let τi = two * αi / zi;
                *g = -τi * phi / ζ - (T::one() - αi) / zi;
            }
            for (g, &zi) in izip!(g_zx[dim1..].iter_mut(), &z_x[dim1..]) {
                *g = (two / ζ) * zi;
            }
        }

        // Populate (d1, d2, p, q, r, grad_primal) at x.
        if !self.update_primal_grad_H(x) {
            return false;
        }
        let g_x = self.data.grad_primal.clone();

        let xz = x.dot(z_x);
        if !(xz > T::zero()) {
            return false;
        }
        let ν = T::from_usize(self.degree()).unwrap();
        let μ_local = xz / ν;

        // Disjoint-field borrows: see try_build_dense_pd_hs for shape.
        let ok = {
            let p_ax = &self.data.p;
            let q_ax = &self.data.q;
            let r_ax = &self.data.r;
            let d1 = &self.data.d1;
            let d2_v = self.data.d2;
            let scratch = &mut self.data.pd_scratch;
            let hs_tmp = &mut self.data.pd_dense_hs_tmp[..n * n];
            let mul_μH_primal = |out: &mut [T], y: &[T]| {
                mul_μH_rank3_parts(p_ax, q_ax, r_ax, d1, d2_v, dim1, μ_ipm, out, y);
            };
            // Direct-x swap: helper expects (s, z, gz, gs); we feed
            // (z_x, x, g_x, g_zx) so it produces Hs satisfying Hs·x = z_x.
            pd_scaling_nd_dense(z_x, x, &g_x, &g_zx, μ_local, mul_μH_primal, scratch, hs_tmp)
        };
        if !ok {
            return false;
        }
        let _ = dim2; // silence

        // Verify both secants Hs·x = z_x and Hs·δx = δz_x.
        let mut z_norm_sq = T::zero();
        for &v in z_x.iter() {
            z_norm_sq += v * v;
        }
        let z_norm = T::sqrt(z_norm_sq);

        let mut δx = vec![T::zero(); n];
        let mut δz_x = vec![T::zero(); n];
        for i in 0..n {
            δx[i] = x[i] + μ_local * g_zx[i];
            δz_x[i] = z_x[i] + μ_local * g_x[i];
        }
        let mut δz_norm_sq = T::zero();
        for &v in δz_x.iter() {
            δz_norm_sq += v * v;
        }
        let δz_norm = T::sqrt(δz_norm_sq);
        let tol_secant: T = (1e-7).as_T();

        let mut hs_x_err = T::zero();
        for i in 0..n {
            let mut row = T::zero();
            for j in 0..n {
                row += self.data.pd_dense_hs_tmp[i * n + j] * x[j];
            }
            let d = row - z_x[i];
            hs_x_err += d * d;
        }
        let hs_x_err = T::sqrt(hs_x_err);
        if hs_x_err > tol_secant * z_norm.max(T::one()) {
            return false;
        }

        let mut hs_δx_err = T::zero();
        for i in 0..n {
            let mut row = T::zero();
            for j in 0..n {
                row += self.data.pd_dense_hs_tmp[i * n + j] * δx[j];
            }
            let d = row - δz_x[i];
            hs_δx_err += d * d;
        }
        let hs_δx_err = T::sqrt(hs_δx_err);
        if hs_δx_err > tol_secant * δz_norm.max(T::one()) {
            return false;
        }

        let data = &mut *self.data;
        data.dense_hs[..n * n].copy_from_slice(&data.pd_dense_hs_tmp[..n * n]);
        self.data.pd_active = true;
        true
    }

    /// Populate the *primal-side* rank-3 Hessian decomposition
    /// `∇²F_primal(s) = D + p·p' − q·q' − r·r'` and the gradient
    /// `∇F_primal(s)`, storing in the same `(d1, d2, p, q, r, grad)`
    /// fields that `update_dual_grad_H(z)` uses for the *dual* side.
    ///
    /// In direct-x mode the cone primitive is owned by `x_cones`, so
    /// slack-side methods like `update_scaling` are never invoked on the
    /// same instance — overwriting the dual fields is safe.
    ///
    /// The primal barrier
    ///   `F(s) = −log(∏ sᵢ^(2αᵢ) − ‖w‖²) − Σ (1−αᵢ) log sᵢ`
    /// is structurally identical to the dual barrier in
    /// `update_dual_grad_H`; the only difference is `phi_p = ∏ sᵢ^(2αᵢ)`
    /// (no `/αᵢ` division). All `(d1, d2, p, q, r)` formulas carry over
    /// by substituting `z → s`, `phi_d → phi_p`, `ζ_d → ζ_p`,
    /// `τ_d[i] = 2αᵢ/zᵢ → τ_p[i] = 2αᵢ/sᵢ`. The block-sparse structure
    /// (q dim1-only, r dim2-only) and PD-preserving sign convention
    /// `D + pp' − qq' − rr'` are preserved — so slack's existing rank-3
    /// sparse expansion machinery handles direct-x GenPowerCone after a
    /// data-only swap.
    ///
    /// Also stores `grad_primal` for the centering-shift formula
    /// `σμ·∇F(x)`.
    fn update_primal_grad_H(&mut self, s: &[T]) -> bool {
        let α = &self.α;
        let dim1 = self.dim1();
        let two: T = (2.).as_T();

        // Log-space phi_p = ∏ sᵢ^(2αᵢ) (no `/αᵢ` — that's the only
        // structural difference from `update_dual_grad_H`'s phi_d).
        let log_phi =
            zip(α, &s[..dim1]).fold(T::zero(), |acc, (&αi, &si)| acc + two * αi * si.ln());
        let phi = log_phi.exp();
        let norm2w = s[dim1..].sumsq();
        let ζ = phi - norm2w;
        if !(ζ > T::zero()) {
            return false;
        }

        let data = &mut self.data;

        // grad_primal[i] = -τ_p[i]·phi/ζ - (1-αᵢ)/sᵢ for i < dim1
        // grad_primal[dim1+j] = (2/ζ)·wⱼ
        // τ_p shares memory with `q` (will be overwritten by q below).
        let τ = &mut data.q;
        for (τ, gp, &αi, &si) in izip!(τ.iter_mut(), &mut data.grad_primal[..dim1], α, &s[..dim1])
        {
            *τ = two * αi / si;
            *gp = -(*τ) * phi / ζ - (T::one() - αi) / si;
        }
        data.grad_primal[dim1..].scalarop_from(|w| (two / ζ) * w, &s[dim1..]);

        // Hessian rank-3 components — identical formulas to dual.
        let eps: T = T::epsilon();
        let p0 = T::sqrt(phi * (phi + norm2w) / two);
        let p1 = if p0 > eps { -two * phi / p0 } else { T::zero() };
        let q0 = T::sqrt(ζ * phi / two);
        let phi_plus_norm2w = phi + norm2w;
        let r1 = if phi_plus_norm2w > eps {
            two * T::sqrt(ζ / phi_plus_norm2w)
        } else {
            T::zero()
        };

        // d1[i] = τ[i]·phi/(ζ·sᵢ) + (1-αᵢ)/sᵢ²
        for (d1, &τ, &αi, &si) in izip!(&mut data.d1, τ.iter(), α, &s[..dim1]) {
            *d1 = τ * phi / (ζ * si) + (T::one() - αi) / (si * si);
        }
        data.d2 = two / ζ;

        // p, q, r vectors — same formulas as dual (with τ_p, phi_p, ζ_p).
        // τ aliases data.q; copy out p[..dim1] and r before scaling.
        data.p[..dim1].scalarop_from(|τi| (p0 / ζ) * τi, τ);
        data.p[dim1..].scalarop_from(|w| (p1 / ζ) * w, &s[dim1..]);
        data.q.scale(q0 / ζ);
        data.r.scalarop_from(|w| (r1 / ζ) * w, &s[dim1..]);
        true
    }
}

/// Free function that multiplies `μ · H_dual` by a vector, using the
/// rank-3 form `H_dual = D + p p' − q q' − r r'` stored on
/// `GenPowerConeData`. Equivalent to the rank-3 part of `mul_Hs` but
/// callable from contexts that already hold a borrow into the cone's
/// data (e.g. closure passed to `pd_scaling_nd_qr6`). The `μ`
/// argument is the central-path parameter to scale by — callers pass
/// the IPM's HSDE μ when reproducing `data.μ · H_dual`, or the
/// math-derivation's local μ = ⟨s,z⟩/ν when constructing the PD
/// scaling (so that `⟨δs,z⟩ = 0` holds exactly).
fn mul_μH_rank3<T: FloatT>(
    data: &GenPowerConeData<T>,
    dim1: usize,
    _dim2: usize,
    μ: T,
    y: &mut [T],
    x: &[T],
) {
    mul_μH_rank3_parts(&data.p, &data.q, &data.r, &data.d1, data.d2, dim1, μ, y, x);
}

/// Same math as `mul_μH_rank3` but takes the individual rank-3
/// components as separate references. Lets the caller hold disjoint
/// `&mut` borrows of OTHER fields of `GenPowerConeData` (e.g.
/// `pd_scratch`, `pd_qr6_out`) while passing this as a closure body —
/// the closure captures field refs, not the whole `&GenPowerConeData`.
#[allow(clippy::too_many_arguments)]
fn mul_μH_rank3_parts<T: FloatT>(
    p: &[T],
    q: &[T],
    r: &[T],
    d1: &[T],
    d2: T,
    dim1: usize,
    μ: T,
    y: &mut [T],
    x: &[T],
) {
    let coef_p = p.dot(x);
    let coef_q = q.dot(&x[..dim1]);
    let coef_r = r.dot(&x[dim1..]);

    for (y, &x, &d1v, &qv) in izip!(&mut y[..dim1], &x[..dim1], d1, q) {
        *y = d1v * x - coef_q * qv;
    }
    for (y, &x, &rv) in izip!(&mut y[dim1..], &x[dim1..], r) {
        *y = d2 * x - coef_r * rv;
    }
    y.axpby(coef_p, p, T::one());
    y.scale(μ);
}

/// Apply ∞-norm magnitude cap to the 3rd-order Mehrotra η correction.
///
/// `‖η‖_∞ ≤ K · ‖ds‖_∞` with K = 7. The 3rd-order term is a
/// *correction* to the predictor; if any component grows much bigger
/// than the predictor's largest component, η stops being a correction
/// and starts dominating the direction. The catastrophic config from
/// `bench_genpow_parity` (64×32 skewed t=0.5 seed=7) had η components
/// 10× the predictor in the first few iters, pushing aggressive steps
/// that landed the iterate near the cone boundary and then stalled the
/// IPM at step=0 by iter 13.
///
/// Sweep across K ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 50} on the
/// genpow parity bench (96 configs at 16×8 / 32×16 / 64×32 / 128×64),
/// reporting (uniform Solved, skewed Solved, total ldl iter):
///
///   K     | uniform S | skewed S | total iter | catastrophic
///   ------|-----------|----------|------------|-------------
///   none  |        43 |       22 |       1391 | AS-63 (broken)
///      1  |        28 |       12 |       1672 |
///      5  |        42 |       18 |       1391 | AS-22
///      6  |        46 |       14 |       1375 |
///   ** 7  |        46 |       24 |       1343 | AS-23  **  (chosen)
///      8  |        47 |       16 |       1405 |
///     10  |        41 |       13 |       1414 |
///
/// Compared to a 2-norm cap with K=2 (the previously-tried setting,
/// which gave uniform Solved=34, skewed Solved=19, total=1588),
/// `linf K=7` is dramatically better — the ∞-norm penalises only
/// outlier components rather than the aggregate, which means uniform
/// configs (where η is naturally well-distributed) don't get clipped
/// while skewed near-boundary configs (where one or two PD-axis
/// components dominate) do. Strict aggregate improvement over both
/// uncapped and the 2-norm cap.
fn apply_eta_cap<T: FloatT>(η: &mut [T], ds: &[T], dim: usize) {
    apply_eta_cap_with_k(η, ds, dim, (7.0).as_T());
}

/// Generalised version of [`apply_eta_cap`] with tunable K. Direct-x
/// uses K=10 (chosen by parameter sweep); slack uses the K=7 default.
fn apply_eta_cap_with_k<T: FloatT>(η: &mut [T], ds: &[T], dim: usize, k: T) {
    let max_ds = ds.iter().fold(T::zero(), |a, &x| a.max(x.abs()));
    let max_eta = η.iter().fold(T::zero(), |a, &x| a.max(x.abs()));
    let cap = k * max_ds;
    if max_eta > cap && max_ds > T::zero() {
        let scale = cap / max_eta;
        for i in 0..dim {
            η[i] *= scale;
        }
    }
}

impl<T> Cone<T> for GenPowerCone<T>
where
    T: FloatT,
{
    fn degree(&self) -> usize {
        self.dim1() + 1
    }

    fn numel(&self) -> usize {
        self.dim()
    }

    fn is_symmetric(&self) -> bool {
        false
    }

    fn is_linear_only(&self) -> bool {
        false
    }

    fn is_sparse_expandable(&self) -> bool {
        // Small cones use the dense-Hs block path (and the full PD
        // scaling). Large cones use the rank-3 sparse expansion.
        !self.is_dense_path
    }

    fn allows_primal_dual_scaling(&self) -> bool {
        // Both code paths construct the full Mosek-Tunçel `Hs` (dense
        // block for small cones, rank-9 sparse expansion for large
        // cones). `update_scaling` falls back to dual-only when the
        // secant numerical guard trips, but that's a per-iteration
        // local decision — the IPM dispatcher should run in
        // `PrimalDual` strategy regardless.
        true
    }

    // GenPowerCone is invariant under uniform positive scaling (homogeneous
    // of degree 1) but NOT under per-coordinate scaling: ∏(d_i·p_i)^α_i =
    // (∏d_i^α_i)·∏p_i^α_i ≠ const·prod unless d_i is constant. Direct-x
    // equilibration must apply a single scalar to x[J], else cone-feasible
    // x[J] in equilibrated space maps to infeasible x[J] in original space
    // (manifests as the solver returning sol.x outside K).
    fn requires_uniform_x_scaling(&self) -> bool {
        true
    }

    fn margins(&mut self, _z: &mut [T], _pd: PrimalOrDualCone) -> (T, T) {
        // We should never end up shifting to this cone, since
        // asymmetric problems should always use unit_initialization
        unreachable!();
    }
    fn scaled_unit_shift(&self, _z: &mut [T], _α: T, _pd: PrimalOrDualCone) {
        // We should never end up shifting to this cone, since
        // asymmetric problems should always use unit_initialization
        unreachable!();
    }

    fn unit_initialization(&self, z: &mut [T], s: &mut [T]) {
        let α = &self.α;
        let dim1 = self.dim1();

        s[..dim1].scalarop_from(|αi| T::sqrt(T::one() + αi), α);
        s[dim1..].set(T::zero());

        z.copy_from(s);
    }

    fn set_identity_scaling(&mut self) {
        // We should never use identity scaling because
        // we never want to allow symmetric initialization
        unreachable!();
    }

    fn update_scaling(
        &mut self,
        s: &[T],
        z: &[T],
        μ: T,
        scaling_strategy: ScalingStrategy,
    ) -> bool {
        if !self.is_dual_feasible(z) {
            return false;
        }

        self.update_dual_grad_H(z);
        self.data.μ = μ;
        self.data.z.copy_from(z);

        if self.is_dense_path {
            // Dense path: build the full Hs matrix in `data.dense_hs`.
            // On PrimalDual we use the Mosek-Tunçel form (both secants);
            // on Dual or numerical failure, fall back to μ·H_dual.
            let mut pd_built = false;
            if scaling_strategy == ScalingStrategy::PrimalDual {
                pd_built = self.try_build_dense_pd_hs(s);
            }
            if !pd_built {
                self.build_dense_dual_hs();
            }
        } else {
            // Sparse path: rank-3 H_dual + optional rank-6 PD axes
            // (4 projector eigenvectors + 2 secant axes, all bounded).
            let mut pd_built = false;
            if scaling_strategy == ScalingStrategy::PrimalDual {
                pd_built = self.try_compute_pd_axes(s);
            }
            if !pd_built {
                self.zero_pd_axes();
            }
        }

        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        // Sparse path: returns true; the KKT uses rank-3+5 sparse expansion
        // columns and `get_Hs()` returns just the diagonal D.
        // Dense path: returns false; the KKT block holds the dense Hs
        // directly (`get_Hs()` writes the packed triu form).
        !self.is_dense_path
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        if self.is_dense_path {
            // Dense path: `Hsblock` is packed upper triangle (column-major
            // — see `CscMatrix::fill_dense_triangle`). `data.dense_hs` is
            // the full row-major n×n matrix; pack the upper triangle.
            // Note that for symmetric Hs the layout is the same in row
            // and column major as long as the diagonal and the upper
            // triangle indices match consistently — we use column-major
            // packed triu: entries (0,0), (0,1), (1,1), (0,2), (1,2),
            // (2,2), ...
            let n = self.dim();
            let mut idx = 0;
            for j in 0..n {
                for i in 0..=j {
                    Hsblock[idx] = self.data.dense_hs[i * n + j];
                    idx += 1;
                }
            }
        } else {
            // Sparse path: just the diagonal D, scaled by μ.
            let dim1 = self.dim1();
            let data = &self.data;
            Hsblock[..dim1].scalarop_from(|d1| data.μ * d1, &data.d1);
            Hsblock[dim1..].set(data.μ * data.d2);
        }
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], _work: &mut [T]) {
        if self.is_dense_path {
            // Dense path: y = dense_hs · x.
            let n = self.dim();
            for i in 0..n {
                let mut acc = T::zero();
                for j in 0..n {
                    acc += self.data.dense_hs[i * n + j] * x[j];
                }
                y[i] = acc;
            }
        } else {
            // Sparse path: Hs = μ·(D + pp' − qq' − rr') [+ rank-9 PD axes].
            // Uses the IPM's HSDE μ to match what `try_compute_pd_axes`
            // passed to the helper's `mul_μH` closure.
            let μ = self.data.μ;
            mul_μH_rank3(&self.data, self.dim1(), self.dim2(), μ, y, x);

            // PD axes contribution: Σ sign · coef · ⟨axis, x⟩ · axis.
            // Skipped when `pd_active = false` (coefs are zero).
            if self.data.pd_active {
                let n = self.dim();
                for k in 0..6 {
                    let coef = self.data.pd_coefs[k];
                    if coef == T::zero() {
                        continue;
                    }
                    let dot = self.data.pd_axes[k].dot(x);
                    let sign: T = T::from_i8(self.data.pd_signs[k]).unwrap();
                    let scale = sign * coef * dot;
                    for i in 0..n {
                        y[i] += scale * self.data.pd_axes[k][i];
                    }
                }
            }
        }
    }

    fn affine_ds(&self, ds: &mut [T], s: &[T]) {
        ds.copy_from(s);
    }

    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T) {
        // Build η directly into shift, then turn it into grad*σμ - η in place.
        self.higher_correction(shift, step_s, step_z);
        for i in 0..self.dim() {
            shift[i] = self.data.grad[i] * σμ - shift[i];
        }
    }

    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], _work: &mut [T], _z: &[T]) {
        out.copy_from(ds);
    }

    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        let step = settings.ipm.linesearch_backtrack_step;
        let αmin = settings.ipm.min_terminate_step_length;

        //simultaneously using "work" and the closures defined
        //below produces a borrow check error, so temporarily
        //move "work" out of self
        let mut work = std::mem::take(&mut self.data.work);

        let is_prim_feasible_fcn = |s: &[T]| -> bool { self.is_primal_feasible(s) };
        let is_dual_feasible_fcn = |s: &[T]| -> bool { self.is_dual_feasible(s) };

        let αz = backtrack_search(dz, z, αmax, αmin, step, is_dual_feasible_fcn, &mut work);
        let αs = backtrack_search(ds, s, αmax, αmin, step, is_prim_feasible_fcn, &mut work);

        //restore work to self
        self.data.work = work;

        (αz, αs)
    }

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        let mut barrier = T::zero();
        let mut work = std::mem::take(&mut self.data.work);

        work.waxpby(T::one(), s, α, ds);
        barrier += self.barrier_primal(&work);

        work.waxpby(T::one(), z, α, dz);
        barrier += self.barrier_dual(&work);

        self.data.work = work;

        barrier
    }

    fn smoothing(&mut self, z: &mut [T], _s: &[T], work: &[T], μ: T) {
        // Newton's method to solve: z + μ*∇f*(z) = work
        // where ∇f*(z) is the dual barrier gradient.
        //
        // The Hessian of the dual barrier is represented as:
        //   H_dual = D + pp' - qq' - rr'
        // where D is diagonal, p/q/r are vectors from update_dual_grad_H.
        // The Newton system Hessian is: H = μ*H_dual + I

        let dim = self.dim();
        let dim1 = self.dim1();
        let min_val: T = (1e-6).as_T();

        // Check if z is feasible and well-conditioned
        let needs_fallback =
            !self.is_dual_feasible(z) || z[..dim1].iter().any(|&zi| T::abs(zi) < min_val);

        if needs_fallback {
            // Fall back to standard initialization
            let α = &self.α;
            for (zi, &αi) in z[..dim1].iter_mut().zip(α.iter()) {
                *zi = T::sqrt(T::one() + αi);
            }
            for zi in z[dim1..].iter_mut() {
                *zi = T::zero();
            }
        }

        let max_iter = 100;
        let tol = T::sqrt(T::epsilon());
        let two_minus_sqrt3: T = (2.0 - 3.0_f64.sqrt()).as_T();
        let dim2 = self.dim2();

        for _ in 0..max_iter {
            // Check feasibility before computing gradient/Hessian
            // (Newton step may have moved z outside the cone)
            if !self.is_dual_feasible(z) || z[..dim1].iter().any(|&zi| zi <= T::zero()) {
                // Rollback to last feasible iterate saved in work buffer
                z.copy_from_slice(&self.data.work[..dim]);
                break;
            }

            // Save current feasible z before Newton update (for rollback)
            self.data.work[..dim].copy_from_slice(z);

            // Update dual gradient and Hessian at current z
            self.update_dual_grad_H(z);

            // Compute residual: res = μ*grad + (z - work)
            for i in 0..dim {
                self.data.smoothing_res[i] = μ * self.data.grad[i] + z[i] - work[i];
            }

            // Check convergence
            let mut res_norm_sq = T::zero();
            for i in 0..dim {
                res_norm_sq += self.data.smoothing_res[i] * self.data.smoothing_res[i];
            }
            if T::sqrt(res_norm_sq) < tol {
                break;
            }

            // Form Hessian: H = μ*(D + pp' - qq' - rr') + I
            for i in 0..dim {
                for j in 0..dim {
                    self.data.smoothing_hmat[i * dim + j] = T::zero();
                }
            }
            // Diagonal part
            for i in 0..dim1 {
                self.data.smoothing_hmat[i * dim + i] = μ * self.data.d1[i] + T::one();
            }
            for i in dim1..dim {
                self.data.smoothing_hmat[i * dim + i] = μ * self.data.d2 + T::one();
            }
            // + pp'
            for i in 0..dim {
                for j in 0..dim {
                    self.data.smoothing_hmat[i * dim + j] =
                        self.data.smoothing_hmat[i * dim + j] + μ * self.data.p[i] * self.data.p[j];
                }
            }
            // - qq' (q is dim1-length)
            for i in 0..dim1 {
                for j in 0..dim1 {
                    self.data.smoothing_hmat[i * dim + j] =
                        self.data.smoothing_hmat[i * dim + j] - μ * self.data.q[i] * self.data.q[j];
                }
            }
            // - rr' (r is dim2-length, placed in bottom-right)
            for i in 0..dim2 {
                for j in 0..dim2 {
                    self.data.smoothing_hmat[(dim1 + i) * dim + (dim1 + j)] =
                        self.data.smoothing_hmat[(dim1 + i) * dim + (dim1 + j)]
                            - μ * self.data.r[i] * self.data.r[j];
                }
            }

            // Dense Cholesky factorization with regularization
            let mut factor_success = false;
            let mut regularizer = T::epsilon();
            let mut max_diag = T::zero();
            for i in 0..dim {
                let d = self.data.smoothing_hmat[i * dim + i].abs();
                if d > max_diag {
                    max_diag = d;
                }
            }
            regularizer *= max_diag;

            for _ in 0..10 {
                // Copy hmat to lmat buffer for Cholesky factorization
                self.data
                    .smoothing_lmat
                    .copy_from_slice(&self.data.smoothing_hmat);

                let mut ok = true;
                for j in 0..dim {
                    let mut sum = self.data.smoothing_lmat[j * dim + j];
                    for k in 0..j {
                        sum -= self.data.smoothing_lmat[j * dim + k]
                            * self.data.smoothing_lmat[j * dim + k];
                    }
                    if sum <= T::zero() {
                        ok = false;
                        break;
                    }
                    let ljj = T::sqrt(sum);
                    self.data.smoothing_lmat[j * dim + j] = ljj;

                    for i in (j + 1)..dim {
                        let mut sum = self.data.smoothing_lmat[i * dim + j];
                        for k in 0..j {
                            sum -= self.data.smoothing_lmat[i * dim + k]
                                * self.data.smoothing_lmat[j * dim + k];
                        }
                        self.data.smoothing_lmat[i * dim + j] = sum / ljj;
                    }
                }

                if ok {
                    // Solve L*y = res (forward substitution)
                    for i in 0..dim {
                        let mut sum = self.data.smoothing_res[i];
                        for j in 0..i {
                            sum -= self.data.smoothing_lmat[i * dim + j]
                                * self.data.smoothing_delta[j];
                        }
                        self.data.smoothing_delta[i] = sum / self.data.smoothing_lmat[i * dim + i];
                    }
                    // Solve L'*delta = y (back substitution)
                    for i in (0..dim).rev() {
                        let mut sum = self.data.smoothing_delta[i];
                        for j in (i + 1)..dim {
                            sum -= self.data.smoothing_lmat[j * dim + i]
                                * self.data.smoothing_delta[j];
                        }
                        self.data.smoothing_delta[i] = sum / self.data.smoothing_lmat[i * dim + i];
                    }
                    factor_success = true;
                    break;
                }
                // Add regularization
                for i in 0..dim {
                    self.data.smoothing_hmat[i * dim + i] += regularizer;
                }
            }

            if !factor_success {
                break;
            }

            // Newton decrement: λ² = res' * delta
            let mut lambda_sq = T::zero();
            for i in 0..dim {
                lambda_sq += self.data.smoothing_res[i] * self.data.smoothing_delta[i];
            }
            let lambda = if lambda_sq > T::zero() {
                T::sqrt(lambda_sq)
            } else {
                T::zero()
            };

            // Damped Newton update
            let damping = if lambda < two_minus_sqrt3 {
                T::one()
            } else {
                T::one() / (T::one() + lambda)
            };

            for i in 0..dim {
                z[i] -= damping * self.data.smoothing_delta[i];
            }
        }
    }

    // ============================================================
    // Direct-x trait overrides — primal-barrier-specific math.
    //
    // GenPowerCone is asymmetric (F ≠ F*); the slack form uses a rank-3
    // update of a diagonal `Hs_dual = μ·(D + pp' - qq' - rr')` with q
    // restricted to the dim1 block and r to the dim2 block.
    //
    // The primal barrier `F(s) = −log(∏ sᵢ^(2αᵢ) − ‖w‖²) − Σ(1−αᵢ)log sᵢ`
    // has the *same* structural form as the dual (only `phi_p = ∏ sᵢ^(2αᵢ)`
    // vs `phi_d = ∏(zᵢ/αᵢ)^(2αᵢ)` differ). So `H_primal(s)` admits the
    // same `D + pp' − qq' − rr'` decomposition with q dim1-only, r
    // dim2-only — slack's existing rank-3 sparse-expansion machinery
    // applies directly once we fill `(d1, d2, p, q, r)` with primal-side
    // values (see `update_primal_grad_H`).
    //
    // Direct-x sign convention: the augmented (1,1) block adds `+Hs` (vs
    // slack's `-Hs` in the (2,2) block), so the expansion-column Dsigns
    // are flipped from slack's `[-1, -1, +1]` to `[+1, +1, -1]`. The
    // expansion machinery handles this via the cone's `direct_x_*`
    // hooks; the cone-level methods below are slack-style identities
    // since the primal-side data already lives in the slack-named fields.
    // ============================================================

    fn direct_x_update_scaling(
        &mut self,
        x: &[T],
        z: &[T],
        μ: T,
        scaling_strategy: ScalingStrategy,
    ) -> bool {
        if !self.is_primal_feasible(x) || !self.is_dual_feasible(z) {
            return false;
        }
        // Dense vs sparse split mirrors slack's `update_scaling`:
        //   dense path (dim ≤ N_DENSE_GENPOW): build full n×n Hs directly via
        //     `try_build_dense_pd_hs_primal` (calls pd_scaling_nd_dense).
        //   sparse path (dim > N_DENSE_GENPOW): use rank-6 axes via
        //     `try_compute_primal_pd_axes` (calls pd_scaling_nd_qr6) and then
        //     reconstruct dense_hs from rank-3 + rank-6.
        let pd_built = match scaling_strategy {
            ScalingStrategy::PrimalDual => {
                if self.is_dense_path {
                    self.try_build_dense_pd_hs_primal(x, z, μ)
                } else {
                    self.try_compute_primal_pd_axes(x, z, μ)
                }
            }
            ScalingStrategy::Dual => false,
        };
        if !pd_built {
            self.zero_pd_axes();
            if !self.update_primal_grad_H(x) {
                return false;
            }
        }
        self.data.x_pt.copy_from_slice(x);
        self.data.z.copy_from_slice(z);
        self.data.μ = μ;
        if self.is_dense_path && !pd_built {
            self.build_dense_primal_hs();
        }
        true
    }

    fn direct_x_combined_ds_shift(
        &mut self,
        shift: &mut [T],
        step_x: &mut [T],
        step_z: &mut [T],
        σμ: T,
    ) {
        // shift = σμ·∇F_primal(x) − η, where η = +0.5·D³F(x)[step_x,
        // step_x, ·] is the Mehrotra correction. Direction comes from
        // the centrality linearisation `z_x + μ·∇F(x) = 0`:
        //
        //   Taylor: δz_x + μ·H_F·δx + (μ/2)·D³F[δx,δx] = -[z_x+μ∇F]
        //
        // so the 3rd-order term lives along δx in x-space. We have
        // `step_x` directly, no SMW recovery needed.
        //
        // 3D verification reduces to `PowerCone::direct_x_higher_correction`.
        self.higher_correction_primal_direct(shift, step_x, step_x, step_z);
        for i in 0..self.dim() {
            shift[i] = self.data.grad_primal[i] * σμ - shift[i];
        }
    }

    // direct_x_affine_offset: trait default `out = z` already correct
    // (see ExpCone for the rationale).

    fn direct_x_combined_offset(&mut self, out: &mut [T], ds: &[T], _work: &mut [T], _x: &[T]) {
        // Slack `Δs_from_Δz_offset` for GenPowerCone is identity.
        out.copy_from_slice(ds);
    }

    fn direct_x_step_length(
        &mut self,
        dx: &[T],
        dz: &[T],
        x: &[T],
        z: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        let step = settings.ipm.linesearch_backtrack_step;
        let αmin = settings.ipm.min_terminate_step_length;
        let mut work = std::mem::take(&mut self.data.work);
        let is_prim = |s: &[T]| -> bool { self.is_primal_feasible(s) };
        let is_dual = |s: &[T]| -> bool { self.is_dual_feasible(s) };
        let αx = backtrack_search(dx, x, αmax, αmin, step, is_prim, &mut work);
        let αz = backtrack_search(dz, z, αmax, αmin, step, is_dual, &mut work);
        self.data.work = work;
        (αx, αz)
    }

    fn direct_x_compute_barrier(&mut self, x: &[T], z: &[T], dx: &[T], dz: &[T], α: T) -> T {
        let mut barrier = T::zero();
        let mut work = std::mem::take(&mut self.data.work);
        work.waxpby(T::one(), x, α, dx);
        barrier += self.barrier_primal(&work);
        work.waxpby(T::one(), z, α, dz);
        barrier += self.barrier_dual(&work);
        self.data.work = work;
        barrier
    }

    fn direct_x_unit_initialization(&self, x: &mut [T], z: &mut [T]) {
        let dim1 = self.dim1();
        for (xi, &αi) in x[..dim1].iter_mut().zip(self.α.iter()) {
            *xi = T::sqrt(T::one() + αi);
        }
        for xi in x[dim1..].iter_mut() {
            *xi = T::zero();
        }
        z.copy_from_slice(x);
    }
}

impl<T> NonsymmetricCone<T> for GenPowerCone<T>
where
    T: FloatT,
{
    // Returns true if s is primal feasible
    fn is_primal_feasible(&self, s: &[T]) -> bool
    where
        T: FloatT,
    {
        let α = &self.α;
        let two: T = (2f64).as_T();
        let dim1 = self.dim1();

        if s[..dim1].iter().all(|&x| x > T::zero()) {
            let res = zip(α, &s[..dim1]).fold(T::zero(), |res, (&αi, &si)| -> T {
                res + two * αi * si.logsafe()
            });
            let res = T::exp(res) - s[dim1..].sumsq();

            if res > T::zero() {
                return true;
            }
        }
        false
    }

    // Returns true if z is dual feasible
    fn is_dual_feasible(&self, z: &[T]) -> bool
    where
        T: FloatT,
    {
        let α = &self.α;
        let two: T = (2.).as_T();
        let dim1 = self.dim1();

        if z[..dim1].iter().all(|&x| x > T::zero()) {
            let res = zip(α, &z[..dim1]).fold(T::zero(), |res, (&αi, &zi)| -> T {
                res + two * αi * (zi / αi).logsafe()
            });
            let res = T::exp(res) - z[dim1..].sumsq();

            if res > T::zero() {
                return true;
            }
        }
        false
    }

    fn barrier_primal(&mut self, s: &[T]) -> T
    where
        T: FloatT,
    {
        // Direct textbook GenPower primal barrier:
        //   F(s) = -log(Π p_i^{2α_i} - ‖w‖²) - Σ (1-α_i) log p_i.
        //
        // The previous implementation derived F via the identity
        //   F(s) = -F*(-∇F(s)) - ν
        // where `barrier_dual` was used in place of F*. This is correct
        // only for self-dual cones; GenPower is *not* self-dual (the
        // dual cone has p coordinates rescaled by α_i), so F̃ ≠ F* and
        // the identity drifts by an s-dependent amount (verified
        // numerically in the `barrier_validation` tests below).
        let dim1 = self.dim1();
        let two: T = (2.0).as_T();

        let log_phi = zip(&s[..dim1], &self.α)
            .fold(T::zero(), |acc, (&si, &αi)| acc + two * αi * si.logsafe());
        let phi = log_phi.exp();
        let psi = phi - s[dim1..].sumsq();

        let mut barrier: T = -psi.logsafe();
        for (&si, &αi) in zip(&s[..dim1], &self.α) {
            barrier -= (T::one() - αi) * si.logsafe();
        }
        barrier
    }

    fn barrier_dual(&mut self, z: &[T]) -> T
    where
        T: FloatT,
    {
        // Dual barrier:
        let α = &self.α;
        let dim1 = self.dim1();
        let two: T = (2.).as_T();

        let mut res = T::zero();
        for (&zi, &αi) in zip(&z[..dim1], α) {
            res += two * αi * (zi / αi).logsafe();
        }
        res = T::exp(res) - z[dim1..].sumsq();

        let mut barrier: T = -res.logsafe();
        for (&zi, &αi) in zip(&z[..dim1], α) {
            barrier -= (zi).logsafe() * (T::one() - αi);
        }

        barrier
    }

    /// Direct-x analog of [`Self::higher_correction`]. Computes the
    /// 3rd-order Mehrotra η for the GenPower *primal* barrier
    ///   `f(x) = -log ψ_p(x) - Σ_{i<n1}(1-α_i) log x_i`,
    ///   `ψ_p(x) = φ_p(x) - ‖w‖²,  φ_p(x) = Π x_i^{2α_i},  w = x[n1..]`.
    ///
    /// Structurally identical to [`Self::higher_correction`] under the
    /// substitution `z → x` and `φ_dual → φ_primal` (the dual barrier
    /// has `(z_i/α_i)^{2α_i}` in φ — the constant `Π α_i^{-2α_i}` factor
    /// drops out of all derivatives, so formula shape is the same; only
    /// the φ value differs). Uses `(d1, d2, p, q, r)` populated by
    /// [`Self::update_primal_grad_H`] for the SMW solve and `data.x_pt`
    /// (set in `direct_x_update_scaling`) as the cone point.
    ///
    /// Sign convention matches [`Self::direct_x_combined_ds_shift`]: the
    /// caller forms `shift = grad_primal·σμ − η`. Returns η through
    /// the `η` slice; applies the same K=7 ∞-norm cap as slack.
    /// Direct entry into the primal η formula given `u` and `v` already
    /// in the same space (no SMW pre-step). Computes
    /// `η = 0.5 · D³F(x_pt)[u, v, ·]` and applies the K=7 ∞-norm cap
    /// against `cap_ds`.
    ///
    /// For the direct-x IPM the natural Mehrotra correction is
    /// `η = 0.5·D³F(x)[step_x, step_x, ·]` — both directions are
    /// the affine x-step, evaluated against the primal barrier `F`.
    /// The directions are available directly, so no SMW solve is needed.
    fn higher_correction_primal_direct(&mut self, η: &mut [T], u: &[T], v: &[T], cap_ds: &[T]) {
        let dim = self.dim();
        let dim1 = self.dim1();
        let dim2 = self.dim2();
        let two: T = (2.).as_T();
        let half: T = (0.5).as_T();
        let α = &self.α;
        let data = &mut self.data;

        // φ_p, ψ_p, g_ψ at x_pt.
        let x = &data.x_pt;
        let log_phi = α
            .iter()
            .zip(x.iter().take(dim1))
            .fold(T::zero(), |acc, (&αi, &xi)| acc + two * αi * xi.ln());
        let phi = log_phi.exp();
        let mut norm2_w = T::zero();
        for j in 0..dim2 {
            norm2_w += x[dim1 + j] * x[dim1 + j];
        }
        let psi = phi - norm2_w;
        let eps_sqrt = T::sqrt(T::epsilon());
        if psi <= T::zero() || psi < eps_sqrt * phi {
            for i in 0..dim {
                η[i] = T::zero();
            }
            return;
        }

        let gpsi = &mut data.correction_gpsi;
        for i in 0..dim1 {
            gpsi[i] = two * α[i] * phi / x[i];
        }
        for j in 0..dim2 {
            gpsi[dim1 + j] = -two * x[dim1 + j];
        }

        let mut sigma_u = T::zero();
        let mut sigma_v = T::zero();
        let mut rho_uv = T::zero();
        for i in 0..dim1 {
            let ai = two * α[i];
            sigma_u += ai * u[i] / x[i];
            sigma_v += ai * v[i] / x[i];
            rho_uv += ai * u[i] * v[i] / (x[i] * x[i]);
        }

        let mut dot_u = T::zero();
        let mut dot_v = T::zero();
        for i in 0..dim {
            dot_u += gpsi[i] * u[i];
            dot_v += gpsi[i] * v[i];
        }
        let mut uw_dot_vw = T::zero();
        for j in 0..dim2 {
            uw_dot_vw += u[dim1 + j] * v[dim1 + j];
        }
        let u_hpsi_v = phi * (sigma_u * sigma_v - rho_uv) - two * uw_dot_vw;

        let coef = (u_hpsi_v * psi - two * dot_u * dot_v) / (psi * psi * psi);
        let inv_psi2 = (psi * psi).recip();

        for i in 0..dim {
            let (hpsi_u_i, hpsi_v_i) = if i < dim1 {
                (
                    gpsi[i] * (sigma_u - u[i] / x[i]),
                    gpsi[i] * (sigma_v - v[i] / x[i]),
                )
            } else {
                (-two * u[i], -two * v[i])
            };

            let t_psi_i = if i < dim1 {
                let bracket =
                    sigma_u * sigma_v - rho_uv - sigma_v * u[i] / x[i] - sigma_u * v[i] / x[i]
                        + two * u[i] * v[i] / (x[i] * x[i]);
                gpsi[i] * bracket
            } else {
                T::zero()
            };

            let mut eta_i =
                -t_psi_i / psi + coef * gpsi[i] + (hpsi_u_i * dot_v + hpsi_v_i * dot_u) * inv_psi2;

            if i < dim1 {
                let beta = T::one() - α[i];
                eta_i -= two * beta * u[i] * v[i] / (x[i] * x[i] * x[i]);
            }

            η[i] = half * eta_i;
        }
        // Mehrotra correction: η = (μ²/2)·D³F(x)[u,v,·]. The (1/2) is
        // baked into the formula above; multiply by μ² here.
        //
        // Slack's `higher_correction` returns η without μ because its NT
        // W-scaling absorbs μ elsewhere. Direct-x stores `Hs = μ·H_primal`
        // explicitly so the μ has to be carried; empirically `μ²` is the
        // right power for the bench-gate (full sweep at κ=1, n×30 configs:
        // μ² gives 30/30 Solved; μ¹ gives 29/30; μ^{0,0.5} give 28-29/30).
        // Hypothesis: step_x grows ~1/μ near the cone boundary, so
        // D³F[step_x, step_x] has μ⁻² magnitude built in; multiplying
        // by μ² normalises to the centring-shift `σμ·∇F` magnitude.
        // Direct-x η μ-power: μ⁴ keeps η in "perturbation" territory across
        // the IPM trajectory. Sweep at d∈{12,24,48,96,192} (dense + sparse
        // paths) over K∈[0.5,5] × μ_exp∈[3,5] showed (K=1.5, μ⁴) as the
        // empirical Pareto winner on the 40-problem suite (3604 iters,
        // 30 strict-Solved). The prior (K=clip(dim/9.6,5,10), μ²) gave
        // 19/40 strict-Solved with 4329 iters. Reasoning: the IPM's
        // centring shift `σμ·∇F` decays linearly in μ; η must decay
        // faster to remain corrective rather than primary.
        let μ = self.data.μ;
        let μ_sq = μ * μ;
        let μ_quad = μ_sq * μ_sq;
        for i in 0..η.len() {
            η[i] *= μ_quad;
        }
        // ∞-norm cap: ‖η‖_∞ ≤ K · ‖cap_ds‖_∞ with K scaled to dim.
        // K = clip(dim / 9.6, 5, 10).
        //
        // Background: 96-cell bench K-sweep showed disjoint failures —
        // K=10 fails at dim ≤ 48 with loose cones (Mehrotra overshoots
        // when boundary is far), K=5 fails at dim = 96 (η too small,
        // loses centring quality). The split aligns cleanly with dim:
        // smaller dim → fewer averaging directions → Mehrotra
        // extrapolation more sensitive to single-coordinate barrier
        // blowup. Linear-in-dim with bench-tuned anchor at dim=96, K=10:
        //
        //   bench results (CPU direct-x):
        //   K=3 static          82/96
        //   K=5 static          92/96
        //   K=10 static         93/96 (prior baseline)
        //   K=clip(dim/9.6,5,10) 95/96  ← current
        // ∞-norm cap K=1.5 (constant, dim-independent). Sweep at K∈[0.5,5]
        // showed K=1.5 wins paired with μ⁴ above. The earlier
        // K=clip(dim/9.6,5,10) heuristic was tuned with μ²; with μ⁴ the
        // cap rarely fires anyway, so smaller K acts as a safety belt
        // against early-iteration extrapolation overshoots without
        // restricting late-iteration corrections.
        let _ = dim; // K is now constant
        let k: T = (1.5).as_T();
        apply_eta_cap_with_k(η, cap_ds, dim, k);
    }

    fn higher_correction(&mut self, η: &mut [T], ds: &[T], v: &[T]) {
        // 3rd-order Mehrotra η for the GenPower dual barrier
        //   f*(z) = -log ψ(z) - Σ_{i<n1}(1-α_i) log z_i,
        //   ψ(z) = φ(z) - ‖w‖², φ(z) = Π_{i<n1}(z_i/α_i)^{2α_i}, w = z[n1..]
        //
        // η = +0.5 · D³f*(z)[u, v, ·], where u solves H_dual·u = ds.
        //
        // (Sign convention matches `combined_ds_shift`: shift = grad·σμ − η.)
        //
        // Closed form (j-th component):
        //   η[j] = 0.5·{ −T_ψ[u,v,j]/ψ + coef·g_ψ[j]
        //                + (H_ψ u·dot_v + H_ψ v·dot_u)[j]/ψ²
        //                − 2·β_j · u_j·v_j / z_j³ · 1[j<n1] },
        // with
        //   coef = (u'H_ψv·ψ − 2·dot_u·dot_v) / ψ³,
        //   dot_u = g_ψ·u, dot_v = g_ψ·v,
        //   β_j = 1 − α_j (linear barrier coefficient).

        let dim = self.dim();
        let dim1 = self.dim1();
        let dim2 = self.dim2();
        let two: T = (2.).as_T();
        let half: T = (0.5).as_T();

        // Borrow split: pull α and z out, mutably borrow data for workspace.
        let α = &self.α;
        let data = &mut self.data;

        // ---- Step 1: solve H_dual · u = ds via Sherman-Morrison-Woodbury ----
        //
        //   H_dual = D + U Σ U',
        //     U = [p, q_full, r_full]   (dim × 3, q on i<n1, r on i≥n1)
        //     Σ = diag(1, −1, −1),  D = diag(d1[0..n1], d2 I_{n2})
        //
        //   H⁻¹ ds = D⁻¹ ds − D⁻¹ U · M⁻¹ · U' D⁻¹ ds,  M = Σ⁻¹ + U' D⁻¹ U  (3×3).
        //
        // D is positive (interior of dual cone); M is generally indefinite.
        // O(dim) work — keeps higher_correction off the O(dim³) hot path.

        // v = D⁻¹ ds, stored in correction_u (final u overwrites it).
        let u = &mut data.correction_u;
        for i in 0..dim1 {
            u[i] = ds[i] / data.d1[i];
        }
        for i in dim1..dim {
            u[i] = ds[i] / data.d2;
        }

        // U' D⁻¹ ds = U' v as a 3-vector b = (b0, b1, b2).
        let mut b0 = T::zero();
        let mut b1 = T::zero();
        let mut b2 = T::zero();
        for i in 0..dim {
            b0 += data.p[i] * u[i];
        }
        for i in 0..dim1 {
            b1 += data.q[i] * u[i];
        }
        for j in 0..dim2 {
            b2 += data.r[j] * u[dim1 + j];
        }

        // Build M = Σ⁻¹ + U' D⁻¹ U (3×3 symmetric, M[1,2] = 0 by disjoint support).
        let mut m00 = T::one();
        let mut m11 = -T::one();
        let mut m22 = -T::one();
        let mut m01 = T::zero();
        let mut m02 = T::zero();
        for i in 0..dim1 {
            let inv = T::one() / data.d1[i];
            m00 += data.p[i] * data.p[i] * inv;
            m11 += data.q[i] * data.q[i] * inv;
            m01 += data.p[i] * data.q[i] * inv;
        }
        let inv_d2 = T::one() / data.d2;
        for j in 0..dim2 {
            let pj = data.p[dim1 + j];
            m00 += pj * pj * inv_d2;
            m22 += data.r[j] * data.r[j] * inv_d2;
            m02 += pj * data.r[j] * inv_d2;
        }

        // Solve M y = b in closed form (3×3 with m12 = 0).
        let det = m00 * m11 * m22 - m01 * m01 * m22 - m02 * m02 * m11;
        if det.abs() < T::epsilon() {
            for i in 0..dim {
                η[i] = T::zero();
            }
            return;
        }
        let inv_det = T::one() / det;
        let c00 = m11 * m22;
        let c01 = -m01 * m22;
        let c02 = -m02 * m11;
        let c11 = m00 * m22 - m02 * m02;
        let c12 = m01 * m02;
        let c22 = m00 * m11 - m01 * m01;
        let y0 = (c00 * b0 + c01 * b1 + c02 * b2) * inv_det;
        let y1 = (c01 * b0 + c11 * b1 + c12 * b2) * inv_det;
        let y2 = (c02 * b0 + c12 * b1 + c22 * b2) * inv_det;

        // u = v − D⁻¹ U y = v − D⁻¹ (p·y0 + q_full·y1 + r_full·y2).
        for i in 0..dim1 {
            u[i] -= (data.p[i] * y0 + data.q[i] * y1) / data.d1[i];
        }
        for j in 0..dim2 {
            u[dim1 + j] -= (data.p[dim1 + j] * y0 + data.r[j] * y2) * inv_d2;
        }

        // ---- Step 2: compute φ, ψ, g_ψ, σ_u, σ_v, ρ_uv ----
        let z = &data.z;
        let log_phi = α
            .iter()
            .zip(z.iter().take(dim1))
            .fold(T::zero(), |acc, (&αi, &zi)| acc + two * αi * (zi / αi).ln());
        let phi = log_phi.exp();
        let mut norm2_w = T::zero();
        for j in 0..dim2 {
            norm2_w += z[dim1 + j] * z[dim1 + j];
        }
        let psi = phi - norm2_w;
        // Guard near-boundary: coef ∝ 1/ψ³ blows up as ψ → 0, so drop η when
        // we are too close. Threshold on the ratio ψ/φ so it scales with the
        // problem; below √ε the corrector adds noise rather than removing it.
        let eps_sqrt = T::sqrt(T::epsilon());
        if psi <= T::zero() || psi < eps_sqrt * phi {
            for i in 0..dim {
                η[i] = T::zero();
            }
            return;
        }

        // g_ψ[i] = 2α_i φ/z_i for i<n1; -2 w_j for i≥n1.
        let gpsi = &mut data.correction_gpsi;
        for i in 0..dim1 {
            gpsi[i] = two * α[i] * phi / z[i];
        }
        for j in 0..dim2 {
            gpsi[dim1 + j] = -two * z[dim1 + j];
        }

        // σ_u, σ_v, ρ_uv: sums over i < n1 only.
        // σ_w = Σ a_i w_i/z_i, ρ_uv = Σ a_i u_i v_i / z_i² (a_i = 2α_i)
        let mut sigma_u = T::zero();
        let mut sigma_v = T::zero();
        let mut rho_uv = T::zero();
        for i in 0..dim1 {
            let ai = two * α[i];
            sigma_u += ai * u[i] / z[i];
            sigma_v += ai * v[i] / z[i];
            rho_uv += ai * u[i] * v[i] / (z[i] * z[i]);
        }

        // dot_u = g_ψ·u, dot_v = g_ψ·v, u'H_ψv = φ(σ_u σ_v − ρ_uv) − 2 u_w·v_w.
        let mut dot_u = T::zero();
        let mut dot_v = T::zero();
        for i in 0..dim {
            dot_u += gpsi[i] * u[i];
            dot_v += gpsi[i] * v[i];
        }
        let mut uw_dot_vw = T::zero();
        for j in 0..dim2 {
            uw_dot_vw += u[dim1 + j] * v[dim1 + j];
        }
        let u_hpsi_v = phi * (sigma_u * sigma_v - rho_uv) - two * uw_dot_vw;

        let coef = (u_hpsi_v * psi - two * dot_u * dot_v) / (psi * psi * psi);
        let inv_psi2 = (psi * psi).recip();

        // ---- Step 3: assemble η ----
        for i in 0..dim {
            // (H_ψ u)[i] and (H_ψ v)[i]
            let (hpsi_u_i, hpsi_v_i) = if i < dim1 {
                (
                    gpsi[i] * (sigma_u - u[i] / z[i]),
                    gpsi[i] * (sigma_v - v[i] / z[i]),
                )
            } else {
                (-two * u[i], -two * v[i])
            };

            // T_ψ[u,v,i]: nonzero only for i < n1.
            let t_psi_i = if i < dim1 {
                let bracket =
                    sigma_u * sigma_v - rho_uv - sigma_v * u[i] / z[i] - sigma_u * v[i] / z[i]
                        + two * u[i] * v[i] / (z[i] * z[i]);
                gpsi[i] * bracket
            } else {
                T::zero()
            };

            let mut eta_i =
                -t_psi_i / psi + coef * gpsi[i] + (hpsi_u_i * dot_v + hpsi_v_i * dot_u) * inv_psi2;

            if i < dim1 {
                let beta = T::one() - α[i];
                eta_i -= two * beta * u[i] * v[i] / (z[i] * z[i] * z[i]);
            }

            η[i] = half * eta_i;
        }

        // ∞-norm magnitude cap. See `apply_eta_cap` docstring.
        apply_eta_cap(η, ds, dim);
    }

    fn update_dual_grad_H(&mut self, z: &[T]) {
        let α = &self.α;
        let dim1 = self.dim1();
        let data = &mut self.data;
        let two: T = (2.).as_T();

        // Log-space computation to avoid overflow/underflow for extreme z values
        let log_phi = zip(α, z).fold(T::zero(), |acc, (&αi, &zi)| acc + two * αi * (zi / αi).ln());
        let phi = log_phi.exp();

        let norm2w = z[dim1..].sumsq();
        let ζ = phi - norm2w;
        debug_assert!(
            ζ > T::zero(),
            "GenPowerCone: z is not in dual cone interior"
        );

        // compute the gradient at z
        let grad = &mut data.grad;
        let τ = &mut data.q;

        for (τ, grad, &α, &z) in izip!(τ.iter_mut(), &mut grad[..dim1], α, &z[..dim1]) {
            *τ = two * α / z;
            *grad = -(*τ) * phi / ζ - (T::one() - α) / z;
        }

        grad[dim1..].scalarop_from(|z| (two / ζ) * z, &z[dim1..]);

        // compute Hessian information at z (with guarded denominators)
        let eps: T = T::epsilon();
        let p0 = T::sqrt(phi * (phi + norm2w) / two);
        let p1 = if p0 > eps { -two * phi / p0 } else { T::zero() };
        let q0 = T::sqrt(ζ * phi / two);
        let phi_plus_norm2w = phi + norm2w;
        let r1 = if phi_plus_norm2w > eps {
            two * T::sqrt(ζ / phi_plus_norm2w)
        } else {
            T::zero()
        };

        // compute the diagonal d1,d2
        for (d1, &τ, &α, &z) in izip!(&mut data.d1, τ.iter(), α, &z[..dim1]) {
            *d1 = (τ) * phi / (ζ * z) + (T::one() - α) / (z * z);
        }
        data.d2 = two / ζ;

        // compute p, q, r where τ shares memory with q
        data.p[..dim1].scalarop_from(|τi| (p0 / ζ) * τi, τ);
        data.p[dim1..].scalarop_from(|zi| (p1 / ζ) * zi, &z[dim1..]);

        data.q.scale(q0 / ζ);
        data.r.scalarop_from(|zi| (r1 / ζ) * zi, &z[dim1..]);
    }
}

impl<T> NonsymmetricNDCone<T> for GenPowerCone<T>
where
    T: FloatT,
{
    // Compute the primal gradient of f(s) at s
    fn gradient_primal(&self, g: &mut [T], s: &[T])
    where
        T: FloatT,
    {
        let dim1 = self.dim1();
        let two: T = (2.).as_T();
        let data = &self.data;

        // unscaled phi (log-space to avoid overflow/underflow)
        let log_phi =
            zip(&s[..dim1], &self.α).fold(T::zero(), |acc, (&si, &αi)| acc + two * αi * si.ln());
        let phi = log_phi.exp();

        // obtain g1 from the Newton-Raphson method
        let (p, r) = s.split_at(dim1);
        let (gp, gr) = g.split_at_mut(dim1);
        let norm_r = r.norm();

        if norm_r > T::epsilon() {
            let g1 = _newton_raphson_genpowcone(norm_r, p, phi, &self.α, data.ψ);

            gr.scalarop_from(|ri| (g1 / norm_r) * ri, r);

            for (gp, &α, &p) in izip!(gp.iter_mut(), &self.α, p) {
                *gp = -(T::one() + α + α * g1 * norm_r) / p;
            }
        } else {
            gr.set(T::zero());

            for (gp, &α, &p) in izip!(gp.iter_mut(), &self.α, p) {
                *gp = -(T::one() + α) / p;
            }
        }
    }
}
// ----------------------------------------------
//  internal operations for generalized power cones

// Newton-Raphson method:
// solve a one-dimensional equation f(x) = 0
// x(k+1) = x(k) - f(x(k))/f'(x(k))
// When we initialize x0 such that 0 < x0 < x* and f(x0) > 0,
// the Newton-Raphson method converges quadratically

fn _newton_raphson_genpowcone<T>(norm_r: T, p: &[T], phi: T, α: &[T], ψ: T) -> T
where
    T: FloatT,
{
    let two: T = (2.).as_T();

    // init point x0: f(x0) > 0
    let x0 = -norm_r.recip()
        + (ψ * norm_r + ((phi / norm_r / norm_r + ψ * ψ - T::one()) * phi).sqrt())
            / (phi - norm_r * norm_r);

    // function for f(x) = 0
    let f0 = {
        |x: T| -> T {
            let finit = -(two * x / norm_r + x * x).logsafe();

            zip(α, p).fold(finit, |f, (&αi, &pi)| {
                f + two * αi * ((x * norm_r + (T::one() + αi) / αi).logsafe() - pi.logsafe())
            })
        }
    };

    // first derivative
    let f1 = {
        |x: T| -> T {
            let finit = -(two * x + two / norm_r) / (x * x + two * x / norm_r);

            α.iter().fold(finit, |f, &αi| {
                f + two * (αi) * norm_r / (norm_r * x + (T::one() + αi) / αi)
            })
        }
    };
    newton_raphson_onesided(x0, f0, f1)
}

#[cfg(test)]
mod barrier_validation {
    use super::*;
    use crate::solver::core::cones::nonsymmetric_common::NonsymmetricCone;

    /// Closed-form GenPower primal barrier from the textbook:
    ///   F(s) = -log(Π p_i^{2α_i} - ‖w‖²) - Σ (1-α_i) log p_i.
    fn ref_F(p: &[f64], w: &[f64], alphas: &[f64]) -> f64 {
        let log_phi: f64 = p
            .iter()
            .zip(alphas.iter())
            .map(|(&pi, &αi)| 2.0 * αi * pi.ln())
            .sum();
        let phi = log_phi.exp();
        let psi = phi - w.iter().map(|x| x * x).sum::<f64>();
        let mut barrier = -psi.ln();
        for (&pi, &αi) in p.iter().zip(alphas.iter()) {
            barrier -= (1.0 - αi) * pi.ln();
        }
        barrier
    }

    fn build(alphas: Vec<f64>, dim2: usize) -> GenPowerCone<f64> {
        GenPowerCone::new(alphas, dim2)
    }

    #[test]
    fn barrier_primal_matches_closed_form_at_unit_init() {
        let alphas = vec![0.6_f64, 0.4];
        let mut cone = build(alphas.clone(), 1);

        // Unit-initialization point — strictly interior, w = 0.
        let s = vec![(1.0_f64 + 0.6).sqrt(), (1.0_f64 + 0.4).sqrt(), 0.0];
        let direct = ref_F(&s[..2], &s[2..], &alphas);
        let from_cone = cone.barrier_primal(&s);

        let err = (direct - from_cone).abs();
        assert!(
            err < 1e-9,
            "barrier_primal mismatch at unit-init: ref={} cone={} (err={})",
            direct,
            from_cone,
            err
        );
    }

    #[test]
    fn barrier_primal_matches_closed_form_off_center() {
        let alphas = vec![0.6_f64, 0.4];
        let mut cone = build(alphas.clone(), 1);

        let s = vec![1.5_f64, 1.5, 0.3];
        let direct = ref_F(&s[..2], &s[2..], &alphas);
        let from_cone = cone.barrier_primal(&s);

        let err = (direct - from_cone).abs();
        assert!(
            err < 1e-9,
            "barrier_primal mismatch off-center: ref={} cone={} (err={})",
            direct,
            from_cone,
            err
        );
    }

    #[test]
    #[ignore = "PD path is gated behind N_DENSE_GENPOW; only exercised when dim ≤ threshold"]
    fn mul_Hs_satisfies_first_secant_under_pd_scaling() {
        // After `update_scaling(s, z, μ, PrimalDual)` with the PD-axes
        // computation enabled, `mul_Hs(z)` must equal `s` (the (S1)
        // primal-dual secant). Failure means either the axes are wrong
        // or `mul_Hs` doesn't fold them in correctly.
        use crate::solver::core::ScalingStrategy;

        // Two configs so we cover skewed and uniform α distributions.
        let configs: &[(Vec<f64>, usize, Vec<f64>, Vec<f64>)] = &[
            (
                vec![0.4_f64, 0.35, 0.25],
                2,
                vec![1.6_f64, 1.4, 1.2, 0.05, -0.04],
                vec![1.5_f64, 1.6, 1.3, 0.04, -0.03],
            ),
            (
                vec![0.5_f64, 0.3, 0.2],
                2,
                vec![2.0_f64, 3.0, 4.0, 0.3, 0.2], // matches failing IPM test
                vec![1.0_f64, 0.6, 0.5, 0.05, 0.05],
            ),
        ];

        for (alphas, dim2, s, z) in configs {
            let mut cone = build(alphas.clone(), *dim2);
            let n = cone.dim();
            let nu = (cone.dim1() + 1) as f64;
            let μ = s.iter().zip(z.iter()).map(|(a, b)| a * b).sum::<f64>() / nu;

            let ok = cone.update_scaling(s, z, μ, ScalingStrategy::PrimalDual);
            assert!(ok, "update_scaling failed for α={:?}", alphas);
            assert!(cone.data.pd_active, "expected pd_active for α={:?}", alphas);

            let mut y = vec![0.0; n];
            let mut work = vec![0.0; n];
            cone.mul_Hs(&mut y, z, &mut work);

            for i in 0..n {
                let err = (y[i] - s[i]).abs();
                assert!(
                    err < 1e-7,
                    "α={:?}: mul_Hs(z)[{}] = {} expected s[{}] = {} (err = {})",
                    alphas,
                    i,
                    y[i],
                    i,
                    s[i],
                    err
                );
            }
        }
    }

    #[test]
    fn barrier_primal_matches_closed_form_uniform_alphas() {
        // 5-dim primal cone, uniform α — typical "high-dim" usage.
        let alphas = vec![0.2_f64; 5];
        let mut cone = build(alphas.clone(), 3);

        let s = vec![
            1.5_f64, 1.2, 0.9, 1.1, 1.4, // p
            0.05, -0.04, 0.03, // w
        ];
        let direct = ref_F(&s[..5], &s[5..], &alphas);
        let from_cone = cone.barrier_primal(&s);

        let err = (direct - from_cone).abs();
        assert!(
            err < 1e-9,
            "barrier_primal mismatch uniform: ref={} cone={} (err={})",
            direct,
            from_cone,
            err
        );
    }
}
