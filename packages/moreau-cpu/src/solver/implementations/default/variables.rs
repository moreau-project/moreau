use super::*;
use crate::algebra::*;
use crate::solver::core::{
    cones::{CompositeCone, CompositeXCone, Cone, PrimalOrDualCone},
    traits::{Settings, Variables},
    ScalingStrategy, StepDirection,
};
use crate::utils::debug::{debug_block, debug_println};

// ---------------
// Variables type for default problem format
// ---------------

/// Standard-form solver type implementing the [`Variables`](crate::solver::core::traits::Variables) trait
pub struct DefaultVariables<T> {
    /// scaled primal variables
    pub x: Vec<T>,
    /// slack variables
    pub s: Vec<T>,
    /// scaled dual variables
    pub z: Vec<T>,
    /// direct-x cone dual variables (one entry per index across all
    /// direct-x cones, in the order they appear in `x_cones`). The
    /// corresponding primal `s_x` is implicit: `s_x = x[J]` for each
    /// direct-x cone J.
    pub z_x: Vec<T>,
    /// homogenization scalar τ
    pub τ: T,
    /// homogenization scalar κ
    pub κ: T,
}

impl<T: std::fmt::Display + std::fmt::Debug> std::fmt::Debug for DefaultVariables<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "x: {:?}\ns: {:?}\nz: {:?}\nz_x: {:?}\nτ: {:?}\nκ: {:?}\n",
            self.x, self.s, self.z, self.z_x, self.τ, self.κ
        )
    }
}

impl<T> DefaultVariables<T>
where
    T: FloatT,
{
    /// Create a new `DefaultVariables` object with no direct-x cones.
    pub fn new(n: usize, m: usize) -> Self {
        Self::new_with_xn(n, m, 0)
    }

    /// Create a new `DefaultVariables` with a given direct-x dual length.
    /// `xn` is the total number of scalar entries across all direct-x
    /// cones (`Σ_J indices_J.len()`).
    pub fn new_with_xn(n: usize, m: usize, xn: usize) -> Self {
        let x = vec![T::zero(); n];
        let s = vec![T::zero(); m];
        let z = vec![T::zero(); m];
        let z_x = vec![T::zero(); xn];
        let τ = T::one();
        let κ = T::one();

        Self {
            x, s, z, z_x, τ, κ
        }
    }
}

impl<T> Variables<T> for DefaultVariables<T>
where
    T: FloatT,
{
    type D = DefaultProblemData<T>;
    type R = DefaultResiduals<T>;
    type C = CompositeCone<T>;
    type SE = DefaultSettings<T>;

    fn calc_mu(
        &mut self,
        residuals: &DefaultResiduals<T>,
        cones: &CompositeCone<T>,
        x_cones: &CompositeXCone<T>,
    ) -> T {
        // Direct-x cones contribute `<x[J], z_J>` to the primal-dual gap
        // and `cone.degree()` to the central-path normalization.
        let mut extra_sz = T::zero();
        let mut off = 0usize;
        for entry in x_cones.iter() {
            let k = entry.indices.len();
            let zx = &self.z_x[off..off + k];
            for (i, &idx) in entry.indices.iter().enumerate() {
                extra_sz += self.x[idx] * zx[i];
            }
            off += k;
        }
        let denom = T::from(cones.degree() + x_cones.degree() + 1).unwrap();
        let mu = (residuals.dot_sz + extra_sz + self.τ * self.κ) / denom;

        // Debug output for CPU/CUDA comparison (set MOREAU_DEBUG=1 to enable)
        debug_block! {
            eprintln!("CPU: μ={:.16e}", mu.to_f64().unwrap());
            eprintln!("CPU: τ={:.16e}", self.τ.to_f64().unwrap());
            eprintln!("CPU: κ={:.16e}", self.κ.to_f64().unwrap());
            eprintln!("CPU: x={:?}", self.x);
            eprintln!("CPU: s={:?}", self.s);
            eprintln!("CPU: z={:?}", self.z);
            eprintln!("CPU: residuals.rx={:?}", residuals.rx);
            eprintln!("CPU: residuals.rz={:?}", residuals.rz);
            eprintln!("CPU: residuals.rτ={:.16e}", residuals.rτ.to_f64().unwrap());
            eprintln!("CPU: residuals.dot_sz={:.16e}", residuals.dot_sz.to_f64().unwrap());
            eprintln!("---");
        }

        mu
    }

    fn affine_step_rhs(
        &mut self,
        residuals: &DefaultResiduals<T>,
        variables: &Self,
        cones: &CompositeCone<T>,
        x_cones: &CompositeXCone<T>,
    ) {
        self.x.copy_from(&residuals.rx);
        self.z.copy_from(&residuals.rz);
        cones.affine_ds(&mut self.s, &variables.s);
        // Direct-x: fill `z_x`-slot (the direct-x step RHS) via the
        // cone's direct-x affine_ds. Primal x[J] is not needed here —
        // for symmetric cones the affine shortcut depends only on z.
        let mut off = 0usize;
        for entry in x_cones.iter() {
            let k = entry.indices.len();
            entry
                .cone
                .direct_x_affine_ds(&mut self.z_x[off..off + k], &variables.z_x[off..off + k]);
            off += k;
        }
        self.τ = residuals.rτ;
        self.κ = variables.τ * variables.κ;

        debug_println!("self after affine_step_rhs: {:?}", self);
    }

    fn combined_step_rhs(
        &mut self,
        residuals: &DefaultResiduals<T>,
        variables: &Self,
        cones: &mut CompositeCone<T>,
        x_cones: &mut CompositeXCone<T>,
        step: &mut Self,
        σ: T,
        μ: T,
        m: T,
    ) {
        let dotσμ = σ * μ;

        self.x.axpby(T::one() - σ, &residuals.rx, T::zero()); //self.x  = (1 - σ)*rx
        self.τ = (T::one() - σ) * residuals.rτ;
        self.κ = -dotσμ + m * step.τ * step.κ + variables.τ * variables.κ;

        // ds is different for symmetric and asymmetric cones:
        // Symmetric cones: d.s = λ ◦ λ + W⁻¹Δs ∘ WΔz − σμe
        // Asymmetric cones: d.s = s + σμ*g(z)

        // we want to scale the Mehotra correction in the symmetric
        // case by M, so just scale step_z by M.  This is an unnecessary
        // vector operation (since it amounts to M*z'*s), but it
        // doesn't happen very often
        if m != T::one() {
            step.z.scale(m);
            step.z_x.scale(m);
        }

        cones.combined_ds_shift(&mut self.z, &mut step.z, &mut step.s, dotσμ);

        //We are relying on d.s = affine_ds already here
        self.s.axpby(T::one(), &self.z, T::one());

        // now we copy the scaled res for rz and d.z is no longer work
        self.z.axpby(T::one() - σ, &residuals.rz, T::zero());

        // Direct-x: mirror the slack combined-step RHS composition via
        // `direct_x_combined_ds_shift(shift, step_x, step_z, σμ)`. The
        // shift is accumulated into `self.z_x` which already holds
        // affine_ds from affine_step_rhs.
        let mut off = 0usize;
        for entry in x_cones.iter_mut() {
            let k = entry.indices.len();
            let mut step_x_gather = vec![T::zero(); k];
            for (i, &idx) in entry.indices.iter().enumerate() {
                step_x_gather[i] = step.x[idx];
            }
            let mut shift = vec![T::zero(); k];
            entry.cone.direct_x_combined_ds_shift(
                &mut shift,
                &mut step_x_gather,
                &mut step.z_x[off..off + k],
                dotσμ,
            );
            for i in 0..k {
                self.z_x[off + i] += shift[i];
            }
            off += k;
        }
    }

    fn calc_step_length(
        &self,
        step: &Self,
        cones: &mut CompositeCone<T>,
        x_cones: &mut CompositeXCone<T>,
        settings: &DefaultSettings<T>,
        step_direction: StepDirection,
    ) -> T {
        let ατ = {
            if step.τ < T::zero() {
                -self.τ / step.τ
            } else {
                T::max_value()
            }
        };

        let ακ = {
            if step.κ < T::zero() {
                -self.κ / step.κ
            } else {
                T::max_value()
            }
        };

        let α = [ατ, ακ, T::one()].minimum();

        let (αz, αs) = cones.step_length(&step.z, &step.s, &self.z, &self.s, settings.core(), α);

        // itself only allows for a single maximum value.
        // To enable split lengths, we need to also pass a
        // tuple of limits to the step_length function of
        // every cone
        let mut α = T::min(αz, αs);

        // Direct-x step length — `direct_x_step_length(dx, dz, x, z, ...)`
        // takes the natural primal-first ordering.
        let mut off = 0usize;
        for entry in x_cones.iter_mut() {
            let k = entry.indices.len();
            let mut x_gather = vec![T::zero(); k];
            let mut dx_gather = vec![T::zero(); k];
            for (i, &idx) in entry.indices.iter().enumerate() {
                x_gather[i] = self.x[idx];
                dx_gather[i] = step.x[idx];
            }
            let (αz_x, αs_x) = entry.cone.direct_x_step_length(
                &dx_gather,
                &step.z_x[off..off + k],
                &x_gather,
                &self.z_x[off..off + k],
                settings.core(),
                α,
            );
            α = T::min(α, T::min(αz_x, αs_x));
            off += k;
        }

        if step_direction == StepDirection::Combined || step_direction == StepDirection::Centering {
            α *= settings.core().ipm.max_step_fraction;
        }

        α
    }

    fn add_step(&mut self, step: &Self, α: T) {
        self.x.axpby(α, &step.x, T::one());
        self.s.axpby(α, &step.s, T::one());
        self.z.axpby(α, &step.z, T::one());
        self.z_x.axpby(α, &step.z_x, T::one());
        self.τ += α * step.τ;
        self.κ += α * step.κ;
    }

    fn symmetric_initialization(
        &mut self,
        cones: &mut CompositeCone<T>,
        x_cones: &mut CompositeXCone<T>,
    ) {
        _shift_to_cone_interior(&mut self.s, cones, PrimalOrDualCone::PrimalCone);
        _shift_to_cone_interior(&mut self.z, cones, PrimalOrDualCone::DualCone);

        // Shift direct-x slots into the cone interior with the correct
        // primal/dual markers: `x[J]` is in the primal cone, `z_J` is
        // in the dual cone. The markers are ignored by symmetric cones
        // (where margins are the same formula) but future asymmetric
        // direct-x cones need the distinction.
        let mut off = 0usize;
        for entry in x_cones.iter_mut() {
            let k = entry.indices.len();
            let mut xj = vec![T::zero(); k];
            for (i, &idx) in entry.indices.iter().enumerate() {
                xj[i] = self.x[idx];
            }
            _shift_single_cone_to_interior(&mut xj, &mut entry.cone, PrimalOrDualCone::PrimalCone);
            _shift_single_cone_to_interior(
                &mut self.z_x[off..off + k],
                &mut entry.cone,
                PrimalOrDualCone::DualCone,
            );
            for (i, &idx) in entry.indices.iter().enumerate() {
                self.x[idx] = xj[i];
            }
            off += k;
        }

        self.τ = T::one();
        self.κ = T::one();

        debug_println!("self after symmetric initialization: {:?}", self);
    }

    fn unit_initialization(&mut self, cones: &CompositeCone<T>, x_cones: &CompositeXCone<T>) {
        cones.unit_initialization(&mut self.z, &mut self.s);

        self.x.set(T::zero());

        // Direct-x: `direct_x_unit_initialization(x, z)` — primal-first
        // ordering. Gather `x[J]` into a local buffer, initialize, then
        // scatter back.
        let mut off = 0usize;
        for entry in x_cones.iter() {
            let k = entry.indices.len();
            let mut xj = vec![T::zero(); k];
            entry
                .cone
                .direct_x_unit_initialization(&mut xj, &mut self.z_x[off..off + k]);
            for (i, &idx) in entry.indices.iter().enumerate() {
                self.x[idx] = xj[i];
            }
            off += k;
        }

        self.τ = T::one();
        self.κ = T::one();
    }

    fn copy_from(&mut self, src: &Self) {
        self.x.copy_from(&src.x);
        self.s.copy_from(&src.s);
        self.z.copy_from(&src.z);
        self.z_x.copy_from(&src.z_x);
        self.τ = src.τ;
        self.κ = src.κ;
    }

    fn scale_cones(
        &self,
        cones: &mut CompositeCone<T>,
        x_cones: &mut CompositeXCone<T>,
        μ: T,
        scaling_strategy: ScalingStrategy,
    ) -> bool {
        let ok = cones.update_scaling(&self.s, &self.z, μ, scaling_strategy);
        if !ok {
            return false;
        }
        // Direct-x: update NT scaling via `direct_x_update_scaling(x, z)`
        // — primal-first ordering. For symmetric cones (nonneg, SOC, PSD)
        // the default impl produces Hs = Hs_inv which matches commit 3's
        // additive contribution to (1,1). Asymmetric cones override.
        let mut off = 0usize;
        for entry in x_cones.iter_mut() {
            let k = entry.indices.len();
            let mut xj = vec![T::zero(); k];
            for (i, &idx) in entry.indices.iter().enumerate() {
                xj[i] = self.x[idx];
            }
            if !entry.cone.direct_x_update_scaling(
                &xj,
                &self.z_x[off..off + k],
                μ,
                scaling_strategy,
            ) {
                return false;
            }
            off += k;
        }
        true
    }

    fn barrier(
        &self,
        step: &Self,
        α: T,
        cones: &mut CompositeCone<T>,
        x_cones: &mut CompositeXCone<T>,
    ) -> T {
        let central_coef = (cones.degree() + x_cones.degree() + 1).as_T();

        let cur_τ = self.τ + α * step.τ;
        let cur_κ = self.κ + α * step.κ;

        // compute current μ (include direct-x contributions to `s·z`).
        let sz_slack = <[T] as VectorMath<T>>::dot_shifted(&self.z, &self.s, &step.z, &step.s, α);
        let mut sz_direct = T::zero();
        {
            let mut off = 0usize;
            for entry in x_cones.iter() {
                let k = entry.indices.len();
                for (i, &idx) in entry.indices.iter().enumerate() {
                    let x_cur = self.x[idx] + α * step.x[idx];
                    let z_cur = self.z_x[off + i] + α * step.z_x[off + i];
                    sz_direct += x_cur * z_cur;
                }
                off += k;
            }
        }
        let sz = sz_slack + sz_direct;
        let μ = (sz + cur_τ * cur_κ) / central_coef;

        // barrier terms from gap and scalars
        let mut barrier = central_coef * μ.logsafe() - cur_τ.logsafe() - cur_κ.logsafe();

        // barriers from the slack cones
        let (z, s) = (&self.z, &self.s);
        let (dz, ds) = (&step.z, &step.s);

        barrier += cones.compute_barrier(z, s, dz, ds, α);

        // barriers from direct-x cones via `direct_x_compute_barrier(
        // x, z, dx, dz, α)` — primal-first ordering.
        let mut off = 0usize;
        for entry in x_cones.iter_mut() {
            let k = entry.indices.len();
            let mut xj = vec![T::zero(); k];
            let mut dxj = vec![T::zero(); k];
            for (i, &idx) in entry.indices.iter().enumerate() {
                xj[i] = self.x[idx];
                dxj[i] = step.x[idx];
            }
            barrier += entry.cone.direct_x_compute_barrier(
                &xj,
                &self.z_x[off..off + k],
                &dxj,
                &step.z_x[off..off + k],
                α,
            );
            off += k;
        }

        barrier
    }

    fn rescale(&mut self) {
        let scale = T::max(self.τ, self.κ);
        let invscale = scale.recip();

        self.x.scale(invscale);
        self.z.scale(invscale);
        self.z_x.scale(invscale);
        self.s.scale(invscale);
        self.τ *= invscale;
        self.κ *= invscale;
    }

    fn has_nan(&self) -> bool {
        self.τ.is_nan()
            || self.κ.is_nan()
            || self.x.iter().any(|v| v.is_nan())
            || self.s.iter().any(|v| v.is_nan())
            || self.z.iter().any(|v| v.is_nan())
            || self.z_x.iter().any(|v| v.is_nan())
    }
}

fn _shift_single_cone_to_interior<T>(
    z: &mut [T],
    cone: &mut crate::solver::core::cones::SupportedCone<T>,
    pd: PrimalOrDualCone,
) where
    T: FloatT,
{
    let degree = cone.degree().as_T();
    let (min_margin, pos_margin) = cone.margins(z, pd);
    let target = T::max(T::one(), (pos_margin * (0.1).as_T()) / degree);

    if min_margin <= T::zero() {
        cone.scaled_unit_shift(z, -min_margin, pd);
        cone.scaled_unit_shift(z, target, pd);
    } else if min_margin < target {
        cone.scaled_unit_shift(z, target - min_margin, pd);
    } else {
        cone.scaled_unit_shift(z, T::zero(), pd);
    }
}

fn _shift_to_cone_interior<T>(z: &mut [T], cones: &mut CompositeCone<T>, pd: PrimalOrDualCone)
where
    T: FloatT,
{
    let degree = cones.degree().as_T();
    let (min_margin, pos_margin) = cones.margins(z, pd);
    let target = T::max(T::one(), (pos_margin * (0.1).as_T()) / degree);

    if min_margin <= T::zero() {
        // at least some component is outside its cone
        // done in two stages since otherwise (1-α) = -α for
        // large α, which makes z exactly 0. (or worse, -0.0 )
        cones.scaled_unit_shift(z, -min_margin, pd);
        cones.scaled_unit_shift(z, target, pd);
    } else if min_margin < target {
        // margin is positive but small.
        cones.scaled_unit_shift(z, target - min_margin, pd);
    } else {
        // good margin, but still shift explicitly by
        // zero to catch any elements in the zero cone
        // that need to be forced to zero
        cones.scaled_unit_shift(z, T::zero(), pd);
    }
}

impl<T> DefaultVariables<T>
where
    T: FloatT,
{
    /// Re-initialise the direct-x cone block — primal `x[J]` and dual
    /// `z_x` — to the cold interior start, leaving slack `(z, s)`
    /// untouched. Used by the warm-start path when no warm `z_x` is
    /// supplied: pairing a default `z_x` with a boundary-valued warm `x`
    /// would strand the IPM on the cone face. `symmetric` selects the
    /// margin-shift vs. unit-init path, matching `default_start`.
    pub(crate) fn reinit_direct_x_cone_block(
        &mut self,
        x_cones: &mut CompositeXCone<T>,
        symmetric: bool,
    ) {
        let mut off = 0usize;
        for entry in x_cones.iter_mut() {
            let k = entry.indices.len();
            let mut xj = vec![T::zero(); k];
            if symmetric {
                for (i, &idx) in entry.indices.iter().enumerate() {
                    xj[i] = self.x[idx];
                }
                _shift_single_cone_to_interior(
                    &mut xj,
                    &mut entry.cone,
                    PrimalOrDualCone::PrimalCone,
                );
                _shift_single_cone_to_interior(
                    &mut self.z_x[off..off + k],
                    &mut entry.cone,
                    PrimalOrDualCone::DualCone,
                );
            } else {
                entry
                    .cone
                    .direct_x_unit_initialization(&mut xj, &mut self.z_x[off..off + k]);
            }
            for (i, &idx) in entry.indices.iter().enumerate() {
                self.x[idx] = xj[i];
            }
            off += k;
        }
    }

    pub(crate) fn unscale(&mut self, data: &DefaultProblemData<T>, is_infeasible: bool) {
        // if we have an infeasible problem, normalize
        // using κ to get an infeasibility certificate.
        // Otherwise use τ to get an unscaled solution.
        let scaleinv = {
            if is_infeasible {
                T::recip(self.κ)
            } else {
                T::recip(self.τ)
            }
        };

        // also undo the equilibration
        let d = &data.equilibration.d;
        let (e, einv) = (&data.equilibration.e, &data.equilibration.einv);
        let cinv = T::recip(data.equilibration.c);

        self.x.hadamard(d).scale(scaleinv);
        self.z.hadamard(e).scale(scaleinv * cinv);
        self.s.hadamard(einv).scale(scaleinv);

        // Direct-x dual z_x pairs with x[J], so it scales with D (not E) —
        // `d[J]` at the cone's index set. A dedicated scatter pass is
        // required since `z_x` is a flat stacked Vec while `d` is length-n.
        if !self.z_x.is_empty() {
            let mut off = 0usize;
            for xcone in &data.x_cones {
                let indices = xcone.indices();
                for (k, &idx) in indices.iter().enumerate() {
                    self.z_x[off + k] = self.z_x[off + k] * d[idx] * scaleinv * cinv;
                }
                off += indices.len();
            }
        }

        self.τ *= scaleinv;
        self.κ *= scaleinv;
    }

    #[cfg_attr(not(feature = "sdp"), allow(dead_code))]
    pub(crate) fn dims(&self) -> (usize, usize) {
        (self.x.len(), self.s.len())
    }
}
