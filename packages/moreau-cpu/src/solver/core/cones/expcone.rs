use crate::{
    algebra::{AsFloatT, DenseMatrixSym3, FloatT, ScalarMath, VectorMath},
    solver::{core::ScalingStrategy, CoreSettings},
};

use super::{
    nonsymmetric_common::{backtrack_search, Nonsymmetric3DCone, NonsymmetricCone},
    Cone, Nonsymmetric3DConeUtils, PrimalOrDualCone,
};

//use super::*;
//use crate::algebra::*;

// Direct-x Mehrotra η tunables for 3D Exp/Power, chosen by parameter sweep.
pub(super) const DIRECT_X_ETA_CAP_K: f64 = 0.5;
pub(super) const DIRECT_X_ETA_MU_EXP: f64 = 4.0;

// -------------------------------------
// Exponential Cone
// -------------------------------------

pub struct ExponentialCone<T> {
    // Hessian of the dual barrier at z (used by slack-form scaling and
    // by symmetric `direct_x_*` defaults — *wrong* for asymmetric
    // direct-x; see the direct-x overrides below).
    H_dual: DenseMatrixSym3<T>,

    // Hessian of the primal barrier at x (used by direct-x scaling).
    // Lazily populated in `direct_x_update_scaling`.
    H_primal: DenseMatrixSym3<T>,

    // Scaling matrix, i.e. μ·H. For slack mode this stores μ·H_dual(z);
    // for direct-x mode this stores μ·H_primal(x).
    Hs: DenseMatrixSym3<T>,

    // Gradient of the dual barrier at z (slack mode).
    grad: [T; 3],

    // Gradient of the primal barrier at x (direct-x mode).
    grad_primal: [T; 3],

    // Holds copy of z at scaling point (slack mode) or copy of z_x (direct-x mode).
    z: [T; 3],

    // Holds copy of x at direct-x scaling point (direct-x mode only).
    x_pt: [T; 3],

    // IPM barrier parameter at last `direct_x_update_scaling` call
    // (direct-x mode only). Needed by the primal-direct Mehrotra
    // corrector to scale η to the centring-shift magnitude.
    μ_ipm: T,
}

#[allow(clippy::new_without_default)]
impl<T> ExponentialCone<T>
where
    T: FloatT,
{
    pub fn new() -> Self {
        Self {
            H_dual: DenseMatrixSym3::zeros(),
            H_primal: DenseMatrixSym3::zeros(),
            Hs: DenseMatrixSym3::zeros(),
            grad: [T::zero(); 3],
            grad_primal: [T::zero(); 3],
            z: [T::zero(); 3],
            x_pt: [T::zero(); 3],
            μ_ipm: T::zero(),
        }
    }

    /// Compute the primal barrier gradient ∇F(x) and Hessian ∇²F(x) at `x`,
    /// storing into `self.grad_primal` and `self.H_primal`. Used by the
    /// direct-x scaling override.
    ///
    /// Uses the *standard* closed-form 3-LB primal barrier
    ///   F(s) = -log(u) - log(s[1]) - log(s[2])
    /// where `u = s[1]·log(s[2]/s[1]) - s[0]`. The slack form's
    /// `gradient_primal` and `hessian_primal_3x3` use the alternative
    /// Karimi-Tunçel implicit construction (Wright-Omega), which is a
    /// different (also valid) self-concordant barrier — different
    /// numerical behavior under direct-x's `+Hs` augmentation. Mirrors
    /// the same fix made for PowerCone direct-x.
    fn update_primal_grad_H(&mut self, s: &[T]) {
        let u = s[1] * (s[2] / s[1]).logsafe() - s[0];
        let v = (s[2] / s[1]).logsafe() - T::one();
        let w = s[1] / s[2];
        let inv_u = T::recip(u);
        let inv_u2 = inv_u * inv_u;

        // ∇F = -(1/u)·∇u - { 0, 1/s[1], 1/s[2] }, where ∇u = (-1, v, w).
        self.grad_primal[0] = inv_u;
        self.grad_primal[1] = -v * inv_u - T::recip(s[1]);
        self.grad_primal[2] = -w * inv_u - T::recip(s[2]);

        // H = (1/u²)·∇u·∇u' - (1/u)·∇²u + diag{0, 1/s[1]², 1/s[2]²}.
        // ∇²u: only (1,1) = -1/s[1], (1,2) = 1/s[2], (2,2) = -s[1]/s[2]².
        let h = &mut self.H_primal;
        h[(0, 0)] = inv_u2;
        h[(0, 1)] = -v * inv_u2;
        h[(0, 2)] = -w * inv_u2;
        h[(1, 1)] = v * v * inv_u2 + inv_u / s[1] + T::recip(s[1] * s[1]);
        h[(1, 2)] = v * w * inv_u2 - inv_u / s[2];
        h[(2, 2)] = w * w * inv_u2 + (s[1] * inv_u) / (s[2] * s[2]) + T::recip(s[2] * s[2]);
    }

    /// Closed-form primal barrier value at `s`. Companion to
    /// `update_primal_grad_H` — both use the standard 3-LB barrier.
    fn barrier_primal_closed_form(&self, s: &[T]) -> T {
        let u = s[1] * (s[2] / s[1]).logsafe() - s[0];
        -u.logsafe() - s[1].logsafe() - s[2].logsafe()
    }

    /// Primal-direct Mehrotra 3rd-order correction for direct-x ExpCone:
    ///   η = (1/2) · D³F(x)[step_x, step_x]   (vector indexed by cone slot)
    /// where F is the closed-form primal barrier
    ///   F(x) = -log(u) - log(x[1]) - log(x[2]),  u = x[1]·log(x[2]/x[1]) - x[0].
    ///
    /// Both contraction slots are the affine x-step — same form GenPower
    /// uses (`higher_correction_primal_direct`). The earlier transcription
    /// of slack's (H_dual⁻¹·Δs, Δz) into (H_primal⁻¹·Δx, Δz) was wrong for
    /// direct-x: in primal form, the central-path equation `z + μ·∇F(x)=0`
    /// linearizes only in Δx, not jointly with Δz, so the 2nd-order
    /// Taylor remainder contracts D³F with Δx twice.
    ///
    /// Decomposition:
    ///   D³F = D³(-log u) + D³(-log x[1]) + D³(-log x[2]).
    /// The -log(u) part splits into the ∇u-aligned and residual H_u/D³u
    /// pieces; the -log(x_i) parts contribute a single diagonal
    /// `-2 a_i b_i / x_i³` term to η[i].
    fn direct_x_higher_correction(&mut self, η: &mut [T], step_x: &[T])
    where
        T: FloatT,
    {
        let x = &self.x_pt;
        let a = step_x;
        let b = step_x;

        let eps = T::epsilon().sqrt();
        if x[1].abs() < eps || x[2].abs() < eps {
            η.set(T::zero());
            return;
        }

        let l = (x[2] / x[1]).logsafe();
        let ucone = x[1] * l - x[0];
        if ucone.abs() < eps {
            η.set(T::zero());
            return;
        }

        // ∇u(x) = (-1, log(x[2]/x[1]) - 1, x[1]/x[2]).
        η[0] = -T::one();
        η[1] = l - T::one();
        η[2] = x[1] / x[2];

        let dot_a = a[0] * η[0] + a[1] * η[1] + a[2] * η[2]; // a · ∇u
        let dot_b = b[0] * η[0] + b[1] * η[1] + b[2] * η[2]; // b · ∇u

        let two: T = (2.).as_T();

        // a^T H_u b. H_u has nonzero entries (1,1)=-1/x[1], (1,2)=(2,1)=1/x[2],
        // (2,2)=-x[1]/x[2]², all others zero.
        let q_u = -a[1] * b[1] / x[1] + (a[1] * b[2] + a[2] * b[1]) / x[2]
            - x[1] * a[2] * b[2] / (x[2] * x[2]);

        // ∇u-aligned coefficient: η = coef · ∇u where
        //   coef = (q_u·u - 2·(a·∇u)·(b·∇u)) / u³
        let coef = (q_u * ucone - two * dot_a * dot_b) / (ucone * ucone * ucone);
        η.scale(coef);

        let inv_u = ucone.recip();
        let inv_u2 = inv_u * inv_u;

        // η[0] additions: zero (no x[0] in non-∇u terms).

        // η[1]: H_u/D³u residual + diagonal -log(x[1]).
        let h_u_a_1 = -a[1] / x[1] + a[2] / x[2];
        let h_u_b_1 = -b[1] / x[1] + b[2] / x[2];
        let d3u_1 = a[1] * b[1] / (x[1] * x[1]) - a[2] * b[2] / (x[2] * x[2]);
        η[1] += dot_b * h_u_a_1 * inv_u2 + dot_a * h_u_b_1 * inv_u2
            - d3u_1 * inv_u
            - two * a[1] * b[1] / (x[1] * x[1] * x[1]);

        // η[2]: H_u/D³u residual + diagonal -log(x[2]).
        let h_u_a_2 = a[1] / x[2] - a[2] * x[1] / (x[2] * x[2]);
        let h_u_b_2 = b[1] / x[2] - b[2] * x[1] / (x[2] * x[2]);
        let d3u_2 = -(a[1] * b[2] + a[2] * b[1]) / (x[2] * x[2])
            + two * x[1] * a[2] * b[2] / (x[2] * x[2] * x[2]);
        η[2] += dot_b * h_u_a_2 * inv_u2 + dot_a * h_u_b_2 * inv_u2
            - d3u_2 * inv_u
            - two * a[2] * b[2] / (x[2] * x[2] * x[2]);

        η.scale((0.5).as_T());
    }
}

impl<T> Cone<T> for ExponentialCone<T>
where
    T: FloatT,
{
    fn degree(&self) -> usize {
        3
    }

    fn numel(&self) -> usize {
        3
    }

    fn is_symmetric(&self) -> bool {
        false
    }

    fn is_linear_only(&self) -> bool {
        false
    }

    fn is_sparse_expandable(&self) -> bool {
        false
    }

    fn allows_primal_dual_scaling(&self) -> bool {
        true
    }

    // ExpCone is invariant under uniform positive scaling (the cone
    // {(x,y,z) : y·exp(x/y) ≤ z, y > 0} scales: (αx, αy, αz) ∈ K iff
    // (x,y,z) ∈ K). NOT invariant under per-coordinate scaling — direct-x
    // equilibration on x must use a single scalar across the cone's index
    // set or cone-feasibility fails on unscale.
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
        s[0] = (-1.051_383_945_322_714).as_T();
        s[1] = (0.556_409_619_469_370).as_T();
        s[2] = (1.258_967_884_768_947).as_T();

        (z[0], z[1], z[2]) = (s[0], s[1], s[2]);
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
        // update both gradient and Hessian for function f*(z) at the point z
        self.update_dual_grad_H(z);

        // update the scaling matrix Hs
        self.update_Hs(s, z, μ, scaling_strategy);

        // K.z .= z
        self.z.copy_from(z);

        true
    }

    fn Hs_is_diagonal(&self) -> bool {
        false
    }

    fn get_Hs(&self, Hsblock: &mut [T]) {
        // Hs data is already in packed triu form, so just copy
        Hsblock.copy_from(&self.Hs.data);
    }

    fn mul_Hs(&mut self, y: &mut [T], x: &[T], _work: &mut [T]) {
        self.Hs.mul(y, x);
    }

    fn affine_ds(&self, ds: &mut [T], s: &[T]) {
        ds.copy_from(s);
    }

    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T) {
        //3rd order correction requires input variables.z

        let mut η = [T::zero(); 3];
        self.higher_correction(&mut η, step_s, step_z);

        for i in 0..3 {
            shift[i] = self.grad[i] * σμ - η[i];
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
        let mut work = [T::zero(); 3];

        let _is_prim_feasible_fcn = |s: &[T]| -> bool { self.is_primal_feasible(s) };
        let _is_dual_feasible_fcn = |s: &[T]| -> bool { self.is_dual_feasible(s) };

        let αz = backtrack_search(dz, z, αmax, αmin, step, _is_dual_feasible_fcn, &mut work);
        let αs = backtrack_search(ds, s, αmax, αmin, step, _is_prim_feasible_fcn, &mut work);

        (αz, αs)
    }

    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T {
        let mut barrier = T::zero();

        let cur_z = [z[0] + α * dz[0], z[1] + α * dz[1], z[2] + α * dz[2]];
        let cur_s = [s[0] + α * ds[0], s[1] + α * ds[1], s[2] + α * ds[2]];

        barrier += self.barrier_dual(&cur_z);
        barrier += self.barrier_primal(&cur_s);

        barrier
    }

    fn smoothing(&mut self, z: &mut [T], _s: &[T], work: &[T], μ: T) {
        // Newton's method to solve: z + μ*∇f*(z) = work
        // where ∇f*(z) is the DUAL barrier gradient
        // This follows Clarabel.jl's newton_smoothing implementation
        //
        // Starting point: use the input z if it's sufficiently in the interior
        // For numerical stability, require z to be well inside the cone
        let min_val: T = (1e-6).as_T();
        let needs_fallback =
            !self.is_dual_feasible(z) || T::abs(z[0]) < min_val || T::abs(z[2]) < min_val;

        if needs_fallback {
            // Fall back to the standard initialization point for exp cone
            let init_z = [
                (-1.051_383_945_322_714).as_T(),
                (0.556_409_619_469_370).as_T(),
                (1.258_967_884_768_947).as_T(),
            ];
            z.copy_from(&init_z);
        }

        let max_iter = 100;
        let tol = T::sqrt(T::epsilon());
        let two_minus_sqrt3: T = (2.0 - 3.0_f64.sqrt()).as_T();

        for _ in 0..max_iter {
            // Update dual gradient and Hessian at current z
            self.update_dual_grad_H(z);

            // Compute residual: res = μ*grad + (z - work)
            let mut res = [T::zero(); 3];
            for i in 0..3 {
                res[i] = μ * self.grad[i] + z[i] - work[i];
            }

            // Check convergence on residual norm
            let res_norm = T::sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2]);
            if res_norm < tol {
                break;
            }

            // Form Hessian: H = μ*H_dual + I
            let mut mat = DenseMatrixSym3::zeros();
            mat[(0, 0)] = μ * self.H_dual[(0, 0)] + T::one();
            mat[(0, 1)] = μ * self.H_dual[(0, 1)];
            mat[(0, 2)] = μ * self.H_dual[(0, 2)];
            mat[(1, 1)] = μ * self.H_dual[(1, 1)] + T::one();
            mat[(1, 2)] = μ * self.H_dual[(1, 2)];
            mat[(2, 2)] = μ * self.H_dual[(2, 2)] + T::one();

            // Factorize with regularization for numerical stability
            let mut chol = DenseMatrixSym3::zeros();
            let regularizer = T::epsilon() * mat.max_element();
            let mut mat_reg = mat;
            let mut factor_success = false;

            for _ in 0..10 {
                if chol.cholesky_3x3_explicit_factor(&mat_reg).is_ok() {
                    factor_success = true;
                    break;
                }
                // Add more regularization
                mat_reg[(0, 0)] += regularizer;
                mat_reg[(1, 1)] += regularizer;
                mat_reg[(2, 2)] += regularizer;
            }

            if !factor_success {
                break;
            }

            // Newton step: solve H * Δ = res for Δ
            let mut delta = [T::zero(); 3];
            chol.cholesky_3x3_explicit_solve(&mut delta, &res);

            // Compute Newton decrement: λ = sqrt(Δ' * H * Δ) = sqrt(res' * Δ)
            let lambda_sq = res[0] * delta[0] + res[1] * delta[1] + res[2] * delta[2];
            let lambda = if lambda_sq > T::zero() {
                T::sqrt(lambda_sq)
            } else {
                T::zero()
            };

            // Damped Newton update: damping_ratio = 1 if λ < 2-√3, else 1/(1+λ)
            let damping_ratio = if lambda < two_minus_sqrt3 {
                T::one()
            } else {
                T::one() / (T::one() + lambda)
            };

            // Update: z = z - damping_ratio * Δ
            for i in 0..3 {
                z[i] -= damping_ratio * delta[i];
            }
        }
    }

    // ============================================================
    // Direct-x trait overrides — primal-barrier-specific math.
    //
    // The `Cone<T>` defaults delegate to slack with a primal↔dual swap,
    // exploiting self-duality. ExpCone is asymmetric (F ≠ F*), so the
    // swap is wrong: it would compute the dual barrier Hessian at the
    // primal point x. These overrides use the primal barrier formulas.
    //
    // Restrictions vs slack ExpCone:
    //  - Dual-only scaling: `Hs = μ·∇²F_primal(x)` is the natural
    //    augmentation of the (1,1) KKT block (the IPM's barrier
    //    Hessian). Slack's primal-dual NT scaling formula assumes
    //    `Hs` is consumed via `A^T·Hs⁻¹·A` — opposite role from
    //    direct-x — and does not port by argument substitution.
    //  - `direct_x_combined_ds_shift` drops the higher-order correction
    //    (the slack version uses a dual-side correction that has no
    //    primal-side analog without separate cone-specific math).
    // ============================================================
    fn direct_x_update_scaling(
        &mut self,
        x: &[T],
        z: &[T],
        μ: T,
        scaling_strategy: ScalingStrategy,
    ) -> bool {
        // Primal feasibility of x and dual feasibility of z is required
        // for the barrier gradient/Hessian formulas to be well-defined.
        if !self.is_primal_feasible(x) || !self.is_dual_feasible(z) {
            return false;
        }

        // For direct-x, the augmented (1,1) KKT block adds
        //   E_J^T · (μ · ∇²F_primal(x)) · E_J
        // — the natural primal-IPM barrier Hessian. This is *not* slack's
        // `Hs ≈ μ·H_dual(z)` because slack uses Hs in `A^T·Hs⁻¹·A`
        // (opposite role from direct-x's `E_J^T·Hs·E_J`). The slack
        // primal-dual NT scaling formula in `nonsymmetric_common.rs`
        // therefore does *not* port to direct-x by argument substitution
        // — it computes a different matrix. For asymmetric direct-x we
        // restrict to dual-only scaling (`Hs = μ·H_primal(x)`); a proper
        // direct-x primal-dual scaling derivation is future work.
        let _ = scaling_strategy;
        self.update_primal_grad_H(x);
        self.update_dual_grad_H(z);
        self.x_pt.copy_from_slice(x);
        self.z.copy_from_slice(z);
        self.Hs.scaled_from(μ, &self.H_primal);
        self.μ_ipm = μ;
        true
    }

    fn direct_x_combined_ds_shift(
        &mut self,
        shift: &mut [T],
        step_x: &mut [T],
        _step_z: &mut [T],
        σμ: T,
    ) {
        // shift = σμ·∇F(x) - η; primal-direct η = ½·D³F(x)[Δx, Δx].
        let mut η = [T::zero(); 3];
        self.direct_x_higher_correction(&mut η, step_x);

        let mu_exp: T = T::from(DIRECT_X_ETA_MU_EXP).unwrap();
        let μ_scale = self.μ_ipm.powf(mu_exp);
        for v in η.iter_mut() {
            *v *= μ_scale;
        }

        let cap_k: T = T::from(DIRECT_X_ETA_CAP_K).unwrap();
        let max_ds = step_x.iter().fold(T::zero(), |a, &x| a.max(x.abs()));
        let max_eta = η.iter().fold(T::zero(), |a, &x| a.max(x.abs()));
        let cap = cap_k * max_ds;
        if max_eta > cap && max_ds > T::zero() {
            let scale = cap / max_eta;
            for v in η.iter_mut() {
                *v *= scale;
            }
        }

        for i in 0..3 {
            shift[i] = self.grad_primal[i] * σμ - η[i];
        }
    }

    // direct_x_affine_offset: cone-equation linearization is symmetric
    // in (primal, dual) labels, so the trait default `out = z` is
    // already correct for asymmetric direct-x ExpCone — no override
    // needed. Same story in PowerCone and GenPowerCone.

    fn direct_x_combined_offset(&mut self, out: &mut [T], ds: &[T], _work: &mut [T], _x: &[T]) {
        // Slack `Δs_from_Δz_offset` for ExpCone is identity (`out = ds`),
        // so the direct-x analog is the same — no swap consequence.
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
        // Walk x in the PRIMAL cone and z in the DUAL cone — no swap.
        let step = settings.ipm.linesearch_backtrack_step;
        let αmin = settings.ipm.min_terminate_step_length;
        let mut work = [T::zero(); 3];

        let is_prim = |s: &[T]| -> bool { self.is_primal_feasible(s) };
        let is_dual = |s: &[T]| -> bool { self.is_dual_feasible(s) };

        let αx = backtrack_search(dx, x, αmax, αmin, step, is_prim, &mut work);
        let αz = backtrack_search(dz, z, αmax, αmin, step, is_dual, &mut work);

        // Return order matches `step_length`'s `(αz, αs)` pattern: the
        // direct-x equivalent is `(αx, αz)` since direct-x's primal is x
        // and dual is z.
        (αx, αz)
    }

    fn direct_x_compute_barrier(&mut self, x: &[T], z: &[T], dx: &[T], dz: &[T], α: T) -> T {
        // Use the closed-form primal barrier (matches the closed-form
        // Hessian/gradient stored by `update_primal_grad_H`). The slack
        // form's `barrier_primal` uses the Karimi-Tunçel construction
        // (Wright-Omega) — different barrier, different value, can't
        // mix here.
        let cur_x = [x[0] + α * dx[0], x[1] + α * dx[1], x[2] + α * dx[2]];
        let cur_z = [z[0] + α * dz[0], z[1] + α * dz[1], z[2] + α * dz[2]];
        self.barrier_primal_closed_form(&cur_x) + self.barrier_dual(&cur_z)
    }

    fn direct_x_unit_initialization(&self, x: &mut [T], z: &mut [T]) {
        // The standard ExpCone unit-init point lies in BOTH the primal
        // and the dual cone (it's self-conjugate at this specific
        // point), so initializing x and z to the same point is feasible.
        // Match the slack default's behavior literally.
        let pt = [
            (-1.051_383_945_322_714).as_T(),
            (0.556_409_619_469_370).as_T(),
            (1.258_967_884_768_947).as_T(),
        ];
        x.copy_from(&pt);
        z.copy_from(&pt);
    }
}

impl<T> NonsymmetricCone<T> for ExponentialCone<T>
where
    T: FloatT,
{
    // -----------------------------------------
    // internal operations for exponential cones
    //
    // Primal exponential cone: s3 ≥ s2*e^(s1/s2), s3,s2 > 0
    // Dual exponential cone: z3 ≥ -z1*e^(z2/z1 - 1), z3 > 0, z1 < 0
    // ----------------------------------------

    // Returns true if s is primal feasible
    fn is_primal_feasible(&self, s: &[T]) -> bool
    where
        T: FloatT,
    {
        if s[2] > T::zero() && s[1] > T::zero() {
            //feasible
            let res = s[1] * (s[2] / s[1]).logsafe() - s[0];
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
        if z[2] > T::zero() && z[0] < T::zero() {
            let res = z[1] - z[0] - z[0] * (-z[2] / z[0]).logsafe();
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
        // Primal barrier:
        // f(s) = ⟨s,g(s)⟩ - f*(-g(s))
        //      = -2*log(s2) - log(s3) - log((1-barω)^2/barω) - 3,
        // where barω = ω(1 - s1/s2 - log(s2) - log(s3))
        // NB: ⟨s,g(s)⟩ = -3 = - ν

        let ω = _wright_omega(T::one() - s[0] / s[1] - (s[1] / s[2]).logsafe());

        let ω = (ω - T::one()) * (ω - T::one()) / ω;

        -ω.logsafe() - (s[1].logsafe()) * ((2.).as_T()) - s[2].logsafe() - (3.).as_T()
    }

    fn barrier_dual(&mut self, z: &[T]) -> T
    where
        T: FloatT,
    {
        // Dual barrier:
        // f*(z) = -log(z2 - z1 - z1*log(z3/-z1)) - log(-z1) - log(z3)
        // -----------------------------------------
        //
        // Guard against z[0] ≈ 0 which causes:
        //   1. l = log(-z[2]/z[0]) → ±∞
        //   2. z[0] * l → 0 * ∞ = NaN
        // This can happen at cone boundaries during IPM iterations.
        let eps = T::epsilon().sqrt();
        if z[0].abs() < eps {
            // Return a large but finite barrier value to indicate near-boundary
            return T::max_value() / (4.0).as_T();
        }

        let l = (-z[2] / z[0]).logsafe();

        // Guard against NaN from log computation
        if !l.is_finite() {
            return T::max_value() / (4.0).as_T();
        }

        -(-z[2] * z[0]).logsafe() - (z[1] - z[0] - z[0] * l).logsafe()
    }

    fn higher_correction(&mut self, η: &mut [T], ds: &[T], v: &[T])
    where
        T: FloatT,
    {
        // u for H^{-1}*Δs
        let H = &self.H_dual;
        let mut u = [T::zero(); 3];
        let z = &self.z;

        //Fine to use symmetric here because the upper
        //triangle is ignored anyway
        let mut cholH = DenseMatrixSym3::zeros();

        // solve H*u = ds
        let is_success = cholH.cholesky_3x3_explicit_factor(H).is_ok();
        if is_success {
            cholH.cholesky_3x3_explicit_solve(&mut u[..], ds);
        } else {
            η.set(T::zero());
            return;
        }

        // Guard against near-zero divisors that can occur at cone boundaries
        let eps = T::epsilon().sqrt();
        if z[0].abs() < eps || z[2].abs() < eps {
            η.set(T::zero());
            return;
        }

        η[1] = T::one();
        η[2] = -z[0] / z[2]; // gradient of ψ
        η[0] = η[2].logsafe();

        let ψ = z[0] * η[0] - z[0] + z[1];

        // Guard against near-zero ψ
        if ψ.abs() < eps {
            η.set(T::zero());
            return;
        }

        let dotψu = u.dot(η);
        let dotψv = v.dot(η);

        let two: T = (2.).as_T();
        let coef =
            ((u[0] * (v[0] / z[0] - v[2] / z[2]) + u[2] * (z[0] * v[2] / z[2] - v[0]) / z[2]) * ψ
                - two * dotψu * dotψv)
                / (ψ * ψ * ψ);

        η.scale(coef);

        let inv_ψ2 = (ψ * ψ).recip();

        // efficient implementation for η above
        η[0] += (ψ.recip() - two / z[0]) * u[0] * v[0] / (z[0] * z[0])
            - u[2] * v[2] / (z[2] * z[2]) / ψ
            + dotψu * inv_ψ2 * (v[0] / z[0] - v[2] / z[2])
            + dotψv * inv_ψ2 * (u[0] / z[0] - u[2] / z[2]);
        η[2] += two * (z[0] / ψ - T::one()) * u[2] * v[2] / (z[2] * z[2] * z[2])
            - (u[2] * v[0] + u[0] * v[2]) / (z[2] * z[2]) / ψ
            + dotψu * inv_ψ2 * (z[0] * v[2] / (z[2] * z[2]) - v[0] / z[2])
            + dotψv * inv_ψ2 * (z[0] * u[2] / (z[2] * z[2]) - u[0] / z[2]);

        η[..].scale((0.5).as_T());
    }

    // 3rd-order correction at the point z.  Output is η.
    //
    // η = -0.5*[(dot(u,Hψ,v)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)*gψ +
    //      dotψu/(ψ*ψ)*Hψv + dotψv/(ψ*ψ)*Hψu - dotψuv/ψ + dothuv]
    //
    // where :
    // Hψ = [  1/z[1]    0   -1/z[3];
    //           0       0   0;
    //         -1/z[3]   0   z[1]/(z[3]*z[3]);]
    // dotψuv = [-u[1]*v[1]/(z[1]*z[1]) + u[3]*v[3]/(z[3]*z[3]);
    //            0;
    //           (u[3]*v[1]+u[1]*v[3])/(z[3]*z[3]) - 2*z[1]*u[3]*v[3]/(z[3]*z[3]*z[3])]
    //
    // dothuv = [-2*u[1]*v[1]/(z[1]*z[1]*z[1]) ;
    //            0;
    //           -2*u[3]*v[3]/(z[3]*z[3]*z[3])]
    // Hψv = Hψ*v
    // Hψu = Hψ*u
    // gψ is used inside η

    fn update_dual_grad_H(&mut self, z: &[T]) {
        let grad = &mut self.grad;
        let H = &mut self.H_dual;

        // Hessian computation, compute μ locally
        let l = (-z[2] / z[0]).logsafe();
        let r = -z[0] * l - z[0] + z[1];

        // compute the gradient at z
        let c2 = r.recip();

        grad[0] = c2 * l - z[0].recip();
        grad[1] = -c2;
        grad[2] = (c2 * z[0] - T::one()) / z[2];

        // compute_Hessian(K,z,H).   Type is symmetric, so
        // only need to assign upper triangle.
        H[(0, 0)] = (r * r - z[0] * r + l * l * z[0] * z[0]) / (r * z[0] * z[0] * r);
        H[(0, 1)] = -l / (r * r);
        H[(1, 1)] = (r * r).recip();
        H[(0, 2)] = (z[1] - z[0]) / (r * r * z[2]);
        H[(1, 2)] = -z[0] / (r * r * z[2]);
        H[(2, 2)] = (r * r - z[0] * r + z[0] * z[0]) / (r * r * z[2] * z[2]);
    }
}

impl<T> Nonsymmetric3DCone<T> for ExponentialCone<T>
where
    T: FloatT,
{
    // Compute the primal gradient of f(s) at s
    fn gradient_primal(&self, s: &[T]) -> [T; 3]
    where
        T: FloatT,
    {
        let mut g = [T::zero(); 3];
        let ω = _wright_omega(T::one() - s[0] / s[1] - (s[1] / s[2]).logsafe());

        g[0] = T::one() / ((ω - T::one()) * s[1]);
        g[1] = g[0] + g[0] * ((ω * s[1] / s[2]).logsafe()) - T::one() / s[1];
        g[2] = ω / ((T::one() - ω) * s[2]);
        g
    }

    // Compute the primal Hessian of f(s) at s
    fn hessian_primal_3x3(&self, s: &[T]) -> DenseMatrixSym3<T>
    where
        T: FloatT,
    {
        let mut h = DenseMatrixSym3::zeros();
        let ω = _wright_omega(T::one() - s[0] / s[1] - (s[1] / s[2]).logsafe());
        let ω_minus_1 = ω - T::one();

        // ∂²f/∂s₁² = ω / ((ω-1)² * s₂² * (1+ω))
        h[(0, 0)] = ω / (ω_minus_1 * ω_minus_1 * s[1] * s[1] * (T::one() + ω));

        // ∂²f/∂s₁∂s₂ = -1 / ((ω-1)² * s₂² * (1+ω))
        h[(0, 1)] = -T::one() / (ω_minus_1 * ω_minus_1 * s[1] * s[1] * (T::one() + ω));

        // ∂²f/∂s₁∂s₃ = 0
        h[(0, 2)] = T::zero();

        // ∂²f/∂s₂²
        let log_term = (ω * s[1] / s[2]).logsafe();
        let a = ω / (ω_minus_1 * ω_minus_1 * s[1] * s[1] * (T::one() + ω));
        let b = T::one() / (s[1] * s[1]);
        h[(1, 1)] = a * (T::one() + log_term * log_term) + b;

        // ∂²f/∂s₂∂s₃ = -ω / ((ω-1)² * s₂ * s₃ * (1+ω))
        h[(1, 2)] = -ω / (ω_minus_1 * ω_minus_1 * s[1] * s[2] * (T::one() + ω));

        // ∂²f/∂s₃² = ω² / ((ω-1)² * s₃² * (1+ω))
        h[(2, 2)] = ω * ω / (ω_minus_1 * ω_minus_1 * s[2] * s[2] * (T::one() + ω));

        h
    }

    //getters
    fn split_borrow_mut(
        &mut self,
    ) -> (
        &mut DenseMatrixSym3<T>,
        &mut DenseMatrixSym3<T>,
        &mut [T; 3],
        &mut [T; 3],
    ) {
        (&mut self.H_dual, &mut self.Hs, &mut self.grad, &mut self.z)
    }
}

// ω(z) is the Wright-Omega function
// Computes the value ω(z) defined as the solution y to
// y+log(y) = z for reals z>=1.
//
// Follows Algorithm 4, §8.4 of thesis of Santiago Serrango:
//  Algorithms for Unsymmetric Cone Optimization and an
//  Implementation for Problems with the Exponential Cone
//  https://web.stanford.edu/group/SOL/dissertations/ThesisAkleAdobe-augmented.pdf

fn _wright_omega<T>(z: T) -> T
where
    T: FloatT,
{
    // Guard against invalid input - return a safe default instead of panicking.
    // This can happen during IPM iterations when numerical instability pushes
    // variables slightly outside the cone. Returning 1.0 (omega(1) = 1) provides
    // a reasonable approximation that allows the solver to continue.
    if z < T::zero() {
        return T::one();
    }

    let mut p: T;
    let mut w: T;
    if z < T::one() + T::PI() {
        //Initialize with the taylor series
        let zm1 = z - T::one();
        p = zm1; //(z-1)
        w = T::one() + p * ((0.5).as_T());
        p *= zm1; //(z-1)^2
        w += p * (1. / 16.0).as_T();
        p *= zm1; //(z-1)^3
        w -= p * (1. / 192.0).as_T();
        p *= zm1; //(z-1)^4
        w -= p * (1. / 3072.0).as_T();
        p *= zm1; //(z-1)^5
        w += p * (13. / 61440.0).as_T();
    } else {
        // Initialize with:
        // w(z) = z - log(z) +
        //        log(z)/z +
        //        log(z)/z^2(log(z)/2-1) +
        //        log(z)/z^3(1/3log(z)^2-3/2log(z)+1)

        let logz = z.logsafe();
        let zinv = z.recip();
        w = z - logz;

        // add log(z)/z
        let mut q = logz * zinv; // log(z)/z
        w += q;

        // add log(z)/z^2(log(z)/2-1)
        q *= zinv; // log(z)/(z^2)
        w += q * (logz / (2.).as_T() - T::one());

        // add log(z)/z^3(1/3log(z)^2-3/2log(z)+1)
        q *= zinv; // log(z)/(z^3)
        w += q * (logz * logz / (3.).as_T() - logz * (1.5).as_T() + T::one());
    }

    // Initialize the residual
    let mut r = z - w - w.logsafe();

    // Santiago suggests two refinement iterations only
    for _ in 0..2 {
        let wp1 = w + T::one();
        let t = wp1 * (wp1 + (r * (2.).as_T()) / (3.0).as_T());
        w *= T::one() + (r / wp1) * (t - r * (0.5).as_T()) / (t - r);

        let r_4th = r * r * r * r;
        let wp1_6th = wp1 * wp1 * wp1 * wp1 * wp1 * wp1;
        r = (w * w * (2.).as_T() - w * (8.).as_T() - T::one()) / (wp1_6th * (72.0).as_T()) * r_4th;
    }

    w
}

// internal unit tests
#[test]
fn test_wright_omega() {
    // y = ω(z) should solve y + ln(y) = z.
    let pts = [1e-7, 1e-5, 1e-3, 1e-1, 1e1, 1e3, 1e5, 1e7, 1e9];

    for z in pts {
        let y = _wright_omega(z);
        let zsolved = y + f64::ln(y);
        let err = f64::abs(z - zsolved);
        assert!((err / z) < 1e-9);
    }
}
