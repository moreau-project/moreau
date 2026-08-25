use super::expcone::{DIRECT_X_ETA_CAP_K, DIRECT_X_ETA_MU_EXP};
use super::*;
use crate::algebra::*;

// -------------------------------------
// Power Cone
// -------------------------------------

pub struct PowerCone<T> {
    // power defining the cone
    α: T,
    // Hessian of the dual barrier at z (used by slack-form scaling and
    // by symmetric `direct_x_*` defaults — *wrong* for asymmetric
    // direct-x; see the direct-x overrides below).
    H_dual: DenseMatrixSym3<T>,

    // Hessian of the primal barrier at x (used by direct-x scaling).
    // Lazily populated in `direct_x_update_scaling`.
    H_primal: DenseMatrixSym3<T>,

    // scaling matrix, i.e. μH. For slack mode this stores μ·H_dual(z);
    // for direct-x mode this stores μ·H_primal(x).
    Hs: DenseMatrixSym3<T>,

    // gradient of the dual barrier at z (slack mode).
    grad: [T; 3],

    // gradient of the primal barrier at x (direct-x mode).
    grad_primal: [T; 3],

    // holds copy of z at scaling point (slack mode) or copy of z_x (direct-x mode).
    z: [T; 3],

    // holds copy of x at direct-x scaling point (direct-x mode only).
    x_pt: [T; 3],

    // IPM barrier parameter at last `direct_x_update_scaling` call
    // (direct-x mode only). See ExpCone for usage in primal-direct η.
    μ_ipm: T,
}

impl<T> PowerCone<T>
where
    T: FloatT,
{
    pub fn new(α: T) -> Self {
        Self {
            α,
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
    /// Uses the *standard* primal barrier
    ///   F(s) = -log(phi - s[2]²) - (1-α)·log(s[0]) - α·log(s[1])
    /// where `phi = s[0]^(2α) · s[1]^(2(1-α))`. This is the closed-form
    /// barrier that matches GenPowerCone(α=[α,1−α], dim2=1). The
    /// alternative "conjugate-dual" formulation in `gradient_primal` /
    /// `hessian_primal_3x3` (Karimi-Tunçel) requires a Newton-Raphson
    /// solve and the matching `hessian_primal_3x3` was only an
    /// "approximation" — its (2,2) and off-diagonal entries are wrong
    /// at general iterates, which caused the IPM to land on bad step
    /// directions and stall once K stacked PowerCone direct-x cones
    /// shared the augmented (1,1) block.
    fn update_primal_grad_H(&mut self, s: &[T]) {
        let α = self.α;
        let two: T = (2.).as_T();
        let phi = s[0].powf(two * α) * s[1].powf(two * (T::one() - α));
        let ψ = phi - s[2] * s[2];

        // gψ = (1/ψ)·∇ψ — same shape as GenPow's primal helper.
        let gψ_0 = two * α * phi / (s[0] * ψ);
        let gψ_1 = two * (T::one() - α) * phi / (s[1] * ψ);
        let gψ_2 = -two * s[2] / ψ;

        // ∇F = -gψ - { (1-α)/s[0], α/s[1], 0 }
        self.grad_primal[0] = -gψ_0 - (T::one() - α) / s[0];
        self.grad_primal[1] = -gψ_1 - α / s[1];
        self.grad_primal[2] = -gψ_2; // = +2 s[2]/ψ

        // H = gψ·gψ' - (1/ψ)·∇²ψ + diag terms.
        let h = &mut self.H_primal;
        h[(0, 0)] = gψ_0 * gψ_0 - two * α * (two * α - T::one()) * phi / (s[0] * s[0] * ψ)
            + (T::one() - α) / (s[0] * s[0]);
        h[(0, 1)] = gψ_0 * gψ_1 - (two * two) * α * (T::one() - α) * phi / (s[0] * s[1] * ψ);
        h[(1, 1)] = gψ_1 * gψ_1
            - two * (T::one() - α) * (T::one() - two * α) * phi / (s[1] * s[1] * ψ)
            + α / (s[1] * s[1]);
        h[(0, 2)] = gψ_0 * gψ_2;
        h[(1, 2)] = gψ_1 * gψ_2;
        h[(2, 2)] = gψ_2 * gψ_2 + two / ψ;
    }

    /// Closed-form primal barrier value at `s`. Companion to
    /// `update_primal_grad_H` — both use the standard
    /// `F(s) = -log(phi - s[2]²) - (1-α)·log(s[0]) - α·log(s[1])`
    /// formulation. Used by `direct_x_compute_barrier` for the
    /// line-search backtrack.
    fn barrier_primal_closed_form(&self, s: &[T]) -> T {
        let α = self.α;
        let two: T = (2.).as_T();
        let phi = s[0].powf(two * α) * s[1].powf(two * (T::one() - α));
        let ψ = phi - s[2] * s[2];
        -ψ.logsafe() - (T::one() - α) * s[0].logsafe() - α * s[1].logsafe()
    }

    /// Primal-direct Mehrotra 3rd-order correction for direct-x PowerCone:
    ///   η = (1/2) · D³F(s)[step_x, step_x]
    /// where F(s) = -log(ψ) - (1-α)·log(s[0]) - α·log(s[1]),
    /// ψ(s) = s[0]^(2α)·s[1]^(2(1-α)) - s[2]².
    ///
    /// Both contraction slots are the affine x-step — see ExpCone's
    /// `direct_x_higher_correction` for the rationale.
    fn direct_x_higher_correction(&mut self, η: &mut [T], step_x: &[T])
    where
        T: FloatT,
    {
        let a = step_x;
        let b = step_x;
        let s = &self.x_pt;
        let α = self.α;

        let eps = T::epsilon().sqrt();
        if s[0].abs() < eps || s[1].abs() < eps {
            η.set(T::zero());
            return;
        }

        let two: T = (2.).as_T();
        let phi = s[0].powf(two * α) * s[1].powf(two * (T::one() - α));
        let ψ = phi - s[2] * s[2];
        if ψ.abs() < eps {
            η.set(T::zero());
            return;
        }

        // ∇ψ = (2α·φ/s[0], 2(1-α)·φ/s[1], -2 s[2])
        η[0] = two * α * phi / s[0];
        η[1] = two * (T::one() - α) * phi / s[1];
        η[2] = -two * s[2];

        let dot_a = a[0] * η[0] + a[1] * η[1] + a[2] * η[2];
        let dot_b = b[0] * η[0] + b[1] * η[1] + b[2] * η[2];

        // H_ψ entries (Hessian of ψ):
        //   H_ψ[0,0] = 2α(2α-1)·φ/s[0]²
        //   H_ψ[0,1] = 4α(1-α)·φ/(s[0]·s[1])
        //   H_ψ[1,1] = 2(1-α)(1-2α)·φ/s[1]²
        //   H_ψ[2,2] = -2,  others zero
        let h00 = two * α * (two * α - T::one()) * phi / (s[0] * s[0]);
        let h01 = (two * two) * α * (T::one() - α) * phi / (s[0] * s[1]);
        let h11 = two * (T::one() - α) * (T::one() - two * α) * phi / (s[1] * s[1]);
        let h22 = -two;

        // a^T H_ψ b
        let q_ψ =
            a[0] * (h00 * b[0] + h01 * b[1]) + a[1] * (h01 * b[0] + h11 * b[1]) + a[2] * h22 * b[2];

        // First contribution: η = coef · ∇ψ
        let coef = (q_ψ * ψ - two * dot_a * dot_b) / (ψ * ψ * ψ);
        η.scale(coef);

        let inv_ψ = ψ.recip();
        let inv_ψ2 = inv_ψ * inv_ψ;

        // (H_ψ a)_k for k=0,1,2:
        let h_ψ_a_0 = h00 * a[0] + h01 * a[1];
        let h_ψ_a_1 = h01 * a[0] + h11 * a[1];
        let h_ψ_a_2 = h22 * a[2];
        let h_ψ_b_0 = h00 * b[0] + h01 * b[1];
        let h_ψ_b_1 = h01 * b[0] + h11 * b[1];
        let h_ψ_b_2 = h22 * b[2];

        // (D³ψ[a,b])_k. Tensor symmetric in indices; only nonzero when no
        // index is 2 (since ψ depends linearly on s[2]² with -2 second
        // derivative, third derivative w.r.t. s[2] is 0; cross-terms with
        // s[2] are also zero because φ doesn't depend on s[2]).
        //   ψ_000 = 2α(2α-1)(2α-2)·φ/s[0]³
        //   ψ_001 = ψ_010 = ψ_100 = 4α(2α-1)(1-α)·φ/(s[0]² s[1])
        //   ψ_011 = ψ_101 = ψ_110 = 4α(1-α)(1-2α)·φ/(s[0] s[1]²)
        //   ψ_111 = -4α(1-α)(1-2α)·φ/s[1]³  (note: 2(1-α)(1-2α)·(2(1-α)-2) = -4α(1-α)(1-2α))
        //   ψ_ijk = 0 if any index is 2.
        let three: T = (3.).as_T();
        let ψ_000 = two * α * (two * α - T::one()) * (two * α - two) * phi / (s[0] * s[0] * s[0]);
        let ψ_001 =
            (two * two) * α * (two * α - T::one()) * (T::one() - α) * phi / (s[0] * s[0] * s[1]);
        let ψ_011 =
            (two * two) * α * (T::one() - α) * (T::one() - two * α) * phi / (s[0] * s[1] * s[1]);
        let ψ_111 =
            -(two * two) * α * (T::one() - α) * (T::one() - two * α) * phi / (s[1] * s[1] * s[1]);
        let _ = three; // keep for clarity

        let d3ψ_0 =
            a[0] * b[0] * ψ_000 + (a[0] * b[1] + a[1] * b[0]) * ψ_001 + a[1] * b[1] * ψ_011;
        let d3ψ_1 =
            a[0] * b[0] * ψ_001 + (a[0] * b[1] + a[1] * b[0]) * ψ_011 + a[1] * b[1] * ψ_111;
        // d3ψ_2 = 0

        // η[0] additions: H_ψ residual + -log(s[0]) diagonal (coef=1-α)
        η[0] += dot_b * h_ψ_a_0 * inv_ψ2 + dot_a * h_ψ_b_0 * inv_ψ2
            - d3ψ_0 * inv_ψ
            - two * (T::one() - α) * a[0] * b[0] / (s[0] * s[0] * s[0]);

        // η[1] additions: H_ψ residual + -log(s[1]) diagonal (coef=α)
        η[1] += dot_b * h_ψ_a_1 * inv_ψ2 + dot_a * h_ψ_b_1 * inv_ψ2
            - d3ψ_1 * inv_ψ
            - two * α * a[1] * b[1] / (s[1] * s[1] * s[1]);

        // η[2] additions: only H_ψ residual (no D³ψ at index 2, no diagonal).
        η[2] += dot_b * h_ψ_a_2 * inv_ψ2 + dot_a * h_ψ_b_2 * inv_ψ2;

        η.scale((0.5).as_T());
    }
}

impl<T> Cone<T> for PowerCone<T>
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

    // PowerCone is invariant under uniform positive scaling but NOT under
    // per-coordinate scaling. Direct-x equilibration on x must use a single
    // scalar across the cone's index set or cone-feasibility fails on
    // unscale.
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
        let α = self.α;

        s[0] = (T::one() + α).sqrt();
        s[1] = (T::one() + (T::one() - α)).sqrt();
        s[2] = T::zero();

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
            !self.is_dual_feasible(z) || T::abs(z[0]) < min_val || T::abs(z[1]) < min_val;

        if needs_fallback {
            // Fall back to the standard initialization point for power cone
            let α = self.α;
            z[0] = (T::one() + α).sqrt();
            z[1] = (T::one() + (T::one() - α)).sqrt();
            z[2] = T::zero();
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
    // exploiting self-duality. PowerCone is asymmetric (F ≠ F*), so the
    // swap is wrong: it would compute the dual barrier Hessian at the
    // primal point x. These overrides use the primal barrier formulas.
    //
    // Restrictions vs slack PowerCone (same as ExpCone direct-x):
    //  - Dual-only scaling: `Hs = μ·∇²F_primal(x)` is the natural
    //    augmentation of the (1,1) KKT block. Slack's primal-dual NT
    //    scaling formula does not port to direct-x (different role).
    //  - `direct_x_combined_ds_shift` drops the higher-order correction
    //    (the slack version uses a dual-side correction; primal-side
    //    analog requires separate cone-specific math).
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
        // shift = σμ·∇F(s) - η; primal-direct η = ½·D³F(s)[Δx, Δx].
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

    // direct_x_affine_offset: trait default `out = z` already correct
    // (see ExpCone for the rationale).

    fn direct_x_combined_offset(&mut self, out: &mut [T], ds: &[T], _work: &mut [T], _x: &[T]) {
        // Slack `Δs_from_Δz_offset` for PowerCone is identity, so the
        // direct-x analog is the same.
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
        (αx, αz)
    }

    fn direct_x_compute_barrier(&mut self, x: &[T], z: &[T], dx: &[T], dz: &[T], α: T) -> T {
        // Use the closed-form primal barrier (matches the closed-form
        // Hessian/gradient stored by `update_primal_grad_H`). The slack
        // form's `barrier_primal` uses the conjugate-dual formulation
        // — different barrier, different value, can't mix here.
        let cur_x = [x[0] + α * dx[0], x[1] + α * dx[1], x[2] + α * dx[2]];
        let cur_z = [z[0] + α * dz[0], z[1] + α * dz[1], z[2] + α * dz[2]];
        self.barrier_primal_closed_form(&cur_x) + self.barrier_dual(&cur_z)
    }

    fn direct_x_unit_initialization(&self, x: &mut [T], z: &mut [T]) {
        // Standard PowerCone unit-init point. From slack
        // `unit_initialization`: s[0] = sqrt(1+α), s[1] = sqrt(2-α),
        // s[2] = 0; (z = s). The point lies in BOTH primal and dual
        // power cones for any α ∈ (0, 1).
        let α = self.α;
        let pt = [
            (T::one() + α).sqrt(),
            (T::one() + (T::one() - α)).sqrt(),
            T::zero(),
        ];
        x.copy_from(&pt);
        z.copy_from(&pt);
    }
}

//-------------------------------------
// primal-dual scaling
//-------------------------------------

impl<T> NonsymmetricCone<T> for PowerCone<T>
where
    T: FloatT,
{
    // Returns true if s is primal feasible
    fn is_primal_feasible(&self, s: &[T]) -> bool
    where
        T: FloatT,
    {
        let α = self.α;
        let two: T = (2f64).as_T();
        if s[0] > T::zero() && s[1] > T::zero() {
            let res = T::exp(two * α * s[0].logsafe() + two * (T::one() - α) * s[1].logsafe())
                - s[2] * s[2];
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
        let α = self.α;
        let two: T = (2.).as_T();

        if z[0] > T::zero() && z[1] > T::zero() {
            let res = T::exp(
                (α * two) * (z[0] / α).logsafe()
                    + (T::one() - α) * (z[1] / (T::one() - α)).logsafe() * two,
            ) - z[2] * z[2];
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
        // Primal barrier: f(s) = ⟨s,g(s)⟩ - f*(-g(s))
        // NB: ⟨s,g(s)⟩ = -3 = - ν

        let α = self.α;
        let two: T = (2.).as_T();
        let three: T = (3.).as_T();

        let g = self.gradient_primal(s);

        let mut out = T::zero();

        out += ((-g[0] / α).powf(two * α) * (-g[1] / (T::one() - α)).powf(two - α * two)
            - g[2] * g[2])
            .logsafe();
        out += (T::one() - α) * (-g[0]).logsafe();
        out += α * (-g[1]).logsafe() - three;
        out
    }

    fn barrier_dual(&mut self, z: &[T]) -> T
    where
        T: FloatT,
    {
        // Dual barrier:
        // f*(z) = -log((z1/α)^{2α} * (z2/(1-α))^{2(1-α)} - z3*z3) - (1-α)*log(z1) - α*log(z2):
        let α = self.α;
        let two: T = (2.).as_T();
        let arg1 =
            (z[0] / α).powf(two * α) * (z[1] / (T::one() - α)).powf(two - two * α) - z[2] * z[2];

        -arg1.logsafe() - (T::one() - α) * z[0].logsafe() - α * z[1].logsafe()
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

        let α = self.α;
        let two: T = (2.).as_T();
        let four: T = (4.).as_T();

        let phi = (z[0] / α).powf(two * α) * (z[1] / (T::one() - α)).powf(two - two * α);
        let ψ = phi - z[2] * z[2];

        // Reuse cholH memory for further computation
        let Hψ = &mut cholH;

        η[0] = two * α * phi / z[0];
        η[1] = two * (T::one() - α) * phi / z[1];
        η[2] = -two * z[2];

        // we only need to assign the upper triangle
        // for our 3x3 symmetric type
        Hψ[(0, 1)] = four * α * (T::one() - α) * phi / (z[0] * z[1]);
        Hψ[(0, 0)] = two * α * (two * α - T::one()) * phi / (z[0] * z[0]);
        Hψ[(0, 2)] = T::zero();
        Hψ[(1, 1)] = two * (T::one() - α) * (T::one() - two * α) * phi / (z[1] * z[1]);
        Hψ[(1, 2)] = T::zero();
        Hψ[(2, 2)] = -two;

        let dotψu = u.dot(η);
        let dotψv = v.dot(η);

        let mut Hψv = [T::zero(); 3];
        Hψ.mul(&mut Hψv, v);

        let coef = (u.dot(&Hψv) * ψ - two * dotψu * dotψv) / (ψ * ψ * ψ);
        let coef2 = four
            * α
            * (two * α - T::one())
            * (T::one() - α)
            * phi
            * (u[0] / z[0] - u[1] / z[1])
            * (v[0] / z[0] - v[1] / z[1])
            / ψ;
        let inv_ψ2 = (ψ * ψ).recip();

        η[0] = coef * η[0] - two * (T::one() - α) * u[0] * v[0] / (z[0] * z[0] * z[0])
            + coef2 / z[0]
            + Hψv[0] * dotψu * inv_ψ2;

        η[1] = coef * η[1] - two * α * u[1] * v[1] / (z[1] * z[1] * z[1]) - coef2 / z[1]
            + Hψv[1] * dotψu * inv_ψ2;

        η[2] = coef * η[2] + Hψv[2] * dotψu * inv_ψ2;

        // reuse vector Hψv
        let Hψu = &mut Hψv;
        Hψ.mul(Hψu, &u);

        // @. η <= (η + Hψu*dotψv*inv_ψ2)/2
        η[..].axpby(dotψv * inv_ψ2, Hψu, T::one());
        η[..].scale((0.5).as_T());
    }

    // 3rd-order correction at the point z.  Output is η.
    //
    // 3rd order correction:
    // η = -0.5*[(dot(u,Hψ,v)*ψ - 2*dotψu*dotψv)/(ψ*ψ*ψ)*gψ +
    //            dotψu/(ψ*ψ)*Hψv + dotψv/(ψ*ψ)*Hψu -
    //            dotψuv/ψ + dothuv]
    // where:
    // Hψ = [  2*α*(2*α-1)*ϕ/(z1*z1)     4*α*(1-α)*ϕ/(z1*z2)       0;
    //         4*α*(1-α)*ϕ/(z1*z2)     2*(1-α)*(1-2*α)*ϕ/(z2*z2)   0;
    //         0                       0                          -2;]

    fn update_dual_grad_H(&mut self, z: &[T]) {
        let H = &mut self.H_dual;
        let α = self.α;
        let two: T = (2.).as_T();
        let four: T = (4.).as_T();

        let phi = (z[0] / α).powf(two * α) * (z[1] / (T::one() - α)).powf(two - two * α);
        let ψ = phi - z[2] * z[2];

        // use K.grad as a temporary workspace
        let gψ = &mut self.grad;
        gψ[0] = two * α * phi / (z[0] * ψ);
        gψ[1] = two * (T::one() - α) * phi / (z[1] * ψ);
        gψ[2] = -two * z[2] / ψ;

        // compute_Hessian(K,z,H).   Type is symmetric, so
        // only need to assign upper triangle.
        H[(0, 0)] = gψ[0] * gψ[0] - two * α * (two * α - T::one()) * phi / (z[0] * z[0] * ψ)
            + (T::one() - α) / (z[0] * z[0]);
        H[(0, 1)] = gψ[0] * gψ[1] - four * α * (T::one() - α) * phi / (z[0] * z[1] * ψ);
        H[(1, 1)] = gψ[1] * gψ[1]
            - two * (T::one() - α) * (T::one() - two * α) * phi / (z[1] * z[1] * ψ)
            + α / (z[1] * z[1]);
        H[(0, 2)] = gψ[0] * gψ[2];
        H[(1, 2)] = gψ[1] * gψ[2];
        H[(2, 2)] = gψ[2] * gψ[2] + two / ψ;

        // compute the gradient at z
        let grad = &mut self.grad;
        grad[0] = -two * α * phi / (z[0] * ψ) - (T::one() - α) / z[0];
        grad[1] = -two * (T::one() - α) * phi / (z[1] * ψ) - α / z[1];
        grad[2] = two * z[2] / ψ;
    }
}

impl<T> Nonsymmetric3DCone<T> for PowerCone<T>
where
    T: FloatT,
{
    // Compute the primal gradient of f(s) at s
    fn gradient_primal(&self, s: &[T]) -> [T; 3]
    where
        T: FloatT,
    {
        let α = self.α;
        let mut g = [T::zero(); 3];
        let two: T = (2.).as_T();

        // unscaled ϕ
        let phi = (s[0]).powf(two * α) * (s[1]).powf(two - α * two);

        // obtain last element of g from the Newton-Raphson method
        let abs_s = s[2].abs();
        if abs_s > T::epsilon() {
            g[2] = _newton_raphson_powcone(abs_s, phi, α);
            if s[2] < T::zero() {
                g[2] = -g[2];
            }
            g[0] = -(α * g[2] * s[2] + T::one() + α) / s[0];
            g[1] = -((T::one() - α) * g[2] * s[2] + two - α) / s[1];
        } else {
            g[2] = T::zero();
            g[0] = -(T::one() + α) / s[0];
            g[1] = -(two - α) / s[1];
        }
        g
    }

    /// Trait-required stub — direct-x uses the closed-form primal Hessian
    /// inlined in `update_primal_grad_H`. Slack does not call this.
    /// The previous body was an "approximation based on barrier
    /// structure" that returned `h[(0,1)] = 0`, missed factors of
    /// `2·phi/ψ` on `h[(0,2)]`/`h[(1,2)]`, and hardcoded
    /// `h[(2,2)] = 1.0` at `s[2]≈0` (correct value: `2/ψ`). Wrong by
    /// O(1) — caused the IPM to stall when K stacked PowerCone direct-x
    /// cones shared the augmented (1,1) block. Do not reintroduce.
    fn hessian_primal_3x3(&self, _s: &[T]) -> DenseMatrixSym3<T>
    where
        T: FloatT,
    {
        unreachable!(
            "PowerCone::hessian_primal_3x3 is unused — direct-x uses the \
             closed-form Hessian inlined in update_primal_grad_H, slack \
             does not need this method"
        );
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

// ----------------------------------------------
//  internal operations for power cones
//
// Primal Power cone: s1^{α}s2^{1-α} ≥ s3, s1,s2 ≥ 0
// Dual Power cone: (z1/α)^{α} * (z2/(1-α))^{1-α} ≥ z3, z1,z2 ≥ 0

// Newton-Raphson method:
// solve a one-dimensional equation f(x) = 0
// x(k+1) = x(k) - f(x(k))/f'(x(k))
// When we initialize x0 such that 0 < x0 < x*,
// the Newton-Raphson method converges quadratically

fn _newton_raphson_powcone<T>(s3: T, phi: T, α: T) -> T
where
    T: FloatT,
{
    let two: T = (2.).as_T();
    let three: T = (3.).as_T();

    // init point x0: since our dual barrier has an additional
    // shift -2α*log(α) - 2(1-α)*log(1-α) > 0 in f(x),
    // the previous selection is still feasible, i.e. f(x0) > 0

    let x0 =
        -s3.recip() + (s3 * two + T::sqrt((phi * phi) / (s3 * s3) + phi * three)) / (phi - s3 * s3);

    // additional shift due to the choice of dual barrier
    let t0 = -two * α * (α.logsafe()) - two * (T::one() - α) * (T::one() - α).logsafe();

    // function for f(x) = 0
    let f0 = {
        |x: T| -> T {
            let two = (2.).as_T();
            let t1 = x * x;
            let t2 = (x * two) / s3;
            two * α * (two * α * t1 + (T::one() + α) * t2).logsafe()
                + two * (T::one() - α) * (two * (T::one() - α) * t1 + (two - α) * t2).logsafe()
                - phi.logsafe()
                - (t1 + t2).logsafe()
                - two * t2.logsafe()
                + t0
        }
    };

    // first derivative
    let f1 = {
        |x: T| -> T {
            let two = (2.).as_T();
            let t1 = x * x;
            let t2 = (two * x) / s3;
            (α * α * two) / (α * x + (T::one() + α) / s3)
                + ((T::one() - α) * two) * (T::one() - α) / ((T::one() - α) * x + (two - α) / s3)
                - ((x + s3.recip()) * two) / (t1 + t2)
        }
    };
    newton_raphson_onesided(x0, f0, f1)
}
