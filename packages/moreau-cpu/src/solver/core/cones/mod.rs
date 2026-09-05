#![allow(non_snake_case)]

use crate::algebra::{FloatT, VectorMath};
use crate::solver::{core::ScalingStrategy, CoreSettings};
use enum_dispatch::*;

// the supported cone wrapper type for primitives
// and the composite cone
mod compositecone;
mod compositedirectcone;
mod supportedcone;
// primitive cone types
mod expcone;
mod genpowcone;
mod nonnegativecone;
mod powcone;
mod socone;
mod zerocone;
// partially specialized traits and blanket implementataions
mod nonsymmetric_common;
mod symmetric_common;

//re-export everything to appear as one module
use nonsymmetric_common::*;
pub use {
    compositecone::*, compositedirectcone::*, expcone::*, genpowcone::*, nonnegativecone::*,
    powcone::*, socone::*, supportedcone::*, symmetric_common::*, zerocone::*,
};

// only use PSD cones with SDP/Blas enabled
#[cfg(feature = "sdp")]
mod psdtrianglecone;
#[cfg(feature = "sdp")]
pub use psdtrianglecone::*;

// marker for primal / dual distinctions
#[derive(Eq, PartialEq, Clone, Debug, Copy)]
pub enum PrimalOrDualCone {
    PrimalCone,
    DualCone,
}

#[enum_dispatch]
pub trait Cone<T>
where
    T: FloatT,
{
    // functions relating to basic sizing
    fn degree(&self) -> usize;
    fn numel(&self) -> usize;

    //Can the cone provide a sparse expanded representation?
    fn is_sparse_expandable(&self) -> bool;

    // is the cone symmetric?  NB: zero cone still reports true
    fn is_symmetric(&self) -> bool;

    // does the cone contain only linear cones (zero + nonneg)?
    // For these problems, unit_initialization works as well as
    // the expensive KKT-based initial point computation.
    fn is_linear_only(&self) -> bool;

    // report false here if only dual scaling is implemented (e.g. GenPowerCone)
    fn allows_primal_dual_scaling(&self) -> bool;

    // true when all cones support smoothed differentiation (zero + nonneg + SOC).
    // Only meaningful on CompositeCone; individual cones return false.
    fn supports_smoothed(&self) -> bool {
        false
    }

    // Converts an elementwise scaling into one that preserves cone
    // membership. Default: scalar (mean) equilibration — `δ = mean(e) / e`
    // — which is correct for every non-linear primitive cone (SOC, Exp,
    // Power, GenPow, PSD). Zero and Nonneg override to leave the
    // elementwise scaling alone (return false, write δ=1).
    fn rectify_equilibration(&self, δ: &mut [T], e: &[T]) -> bool {
        δ.copy_from(e).recip().scale(e.mean());
        true // scalar equilibration
    }

    // Does a direct-x constraint `x[J] ∈ K_J` require a uniform scalar
    // scaling over `d[J]` to remain invariant under Ruiz equilibration?
    //
    // Nonneg / PSD: `false` — positive-diagonal scaling preserves both.
    // SOC: `true` — the cone is not invariant under per-entry positive
    // scaling; `d[J]` must be replaced with its geometric mean.
    //
    // This method exists only for the direct-x equilibration path; slack
    // cones go through `rectify_equilibration` as before.
    fn requires_uniform_x_scaling(&self) -> bool {
        false
    }

    // returns (α,β) such that:
    // z - α⋅e is just on the cone boundary, with value
    // α >=0 indicates z \in cone, i.e. negative margin ===
    // outside of the cone.
    //
    // β is the sum of the margins that are positive.   For most
    // cones this will just be β = max(0.,α), but for cones that
    // are composites (e.g. the R_n^+), it is the sum of all of
    // the positive margin terms.
    fn margins(&mut self, z: &mut [T], pd: PrimalOrDualCone) -> (T, T);

    // functions relating to unit vectors and cone initialization
    fn scaled_unit_shift(&self, z: &mut [T], α: T, pd: PrimalOrDualCone);
    fn unit_initialization(&self, z: &mut [T], s: &mut [T]);

    // Compute scaling points
    fn set_identity_scaling(&mut self);
    fn update_scaling(
        &mut self, s: &[T], z: &[T], μ: T, scaling_strategy: ScalingStrategy
    ) -> bool;

    // operations on the Hessian of the centrality condition
    // : W^TW for symmmetric cones
    // : μH(s) for nonsymmetric cones
    fn Hs_is_diagonal(&self) -> bool;
    fn get_Hs(&self, Hsblock: &mut [T]);
    fn mul_Hs(&mut self, y: &mut [T], x: &[T], work: &mut [T]);

    // ---------------------------------------------------------
    // Linearized centrality condition functions
    //
    // For nonsymmetric cones:
    // -----------------------
    //
    // The centrality condition is : s = -μg(z)
    //
    // The linearized version is :
    //     Δs + μH(z)Δz = -ds = -(affine_ds + combined_ds_shift)
    //
    // The affine term (computed in affine_ds!) is s
    // The shift term is μg(z) plus any higher order corrections
    //
    // # To recover Δs from Δz, we can write
    //     Δs = - (ds + μHΔz)
    // The "offset" in Δs_from_Δz_offset is then just ds
    //
    // For symmetric cones:
    // --------------------
    //
    // The centrality condition is : (W(z + Δz) ∘ W⁻ᵀ(s + Δs) = μe
    //
    // The linearized version is :
    //     λ ∘ (WΔz + WᵀΔs) = -ds = - (affine_ds + combined_ds_shift)
    //
    // The affine term (computed in affine_ds!) is λ ∘ λ
    // The shift term is W⁻¹Δs_aff ∘ WΔz_aff - σμe, where the terms
    // Δs_aff an Δz_aff are from the affine KKT solve, i.e. they
    // are the Mehrotra correction terms.
    //
    // To recover Δs from Δz, we can write
    //     Δs = - ( Wᵀ(λ \ ds) + WᵀW Δz)
    // The "offset" in Δs_from_Δz_offset is then Wᵀ(λ \ ds)
    //
    // Note that the Δs_from_Δz_offset function is only needed in the
    // general combined step direction.   In the affine step direction,
    // we have the identity Wᵀ(λ \ (λ ∘ λ )) = s.  The symmetric and
    // nonsymmetric cases coincide and offset is taken directly as s.
    //
    // The affine step directions terms steps_z and step_s are
    // passed to combined_ds_shift as mutable.  Once they have been
    // used to compute the combined ds shift they are no longer needed,
    // so may be modified in place as workspace.
    // ---------------------------------------------------------
    fn affine_ds(&self, ds: &mut [T], s: &[T]);
    fn combined_ds_shift(&mut self, shift: &mut [T], step_z: &mut [T], step_s: &mut [T], σμ: T);
    fn Δs_from_Δz_offset(&mut self, out: &mut [T], ds: &[T], work: &mut [T], z: &[T]);

    // Find the maximum step length in some search direction
    fn step_length(
        &mut self,
        dz: &[T],
        ds: &[T],
        z: &[T],
        s: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T);

    // return the barrier function at (z+αdz,s+αds)
    fn compute_barrier(&mut self, z: &[T], s: &[T], dz: &[T], ds: &[T], α: T) -> T;

    // ---------------------------------------------------------
    // Direct-x cone primitives.
    //
    // # Convention
    //
    // In direct-x a sub-vector `x[J]` is constrained to the cone directly,
    // with `z_J` the associated dual. We call the pair `(x, z)` here —
    // `x` is the PRIMAL cone variable, `z` is its DUAL. Note this is the
    // OPPOSITE of slack naming (`s` primal, `z` dual): direct-x's `z`
    // plays the role slack's `s` plays, and direct-x's `x` plays the
    // role slack's `z` plays. Every default below encodes that swap.
    //
    // # Why the swap is mathematically justified (symmetric cones)
    //
    // For symmetric cones (nonneg, SOC, PSD) the cone K is self-dual
    // (K == K*) and the log-barrier `F` is self-conjugate up to a sign:
    // `F*(z) = F(z) + const`. The Nesterov–Todd scaling `W` depends on
    // the pair `(s, z)` via
    //     W·s = W⁻ᵀ·z = λ
    // and carries the invariant `∇²F(s) = W²` only when `(s, z)` are
    // plugged in with `s` on the primal side. Swapping the two arguments
    // yields the same NT triple but against the *dual* barrier `F*`, which
    // for self-dual K means `W` is replaced by `W⁻ᵀ` and therefore the
    // stored Hessian flips: `Hs ↔ Hs_inv`.
    //
    // The slack KKT (1,1) block needs the primal-side `Hs` (mapping Δz
    // to Δs); the direct-x (1,1) block needs the DUAL-side `Hs_inv`
    // (mapping Δx to Δz for post-KKT recovery, and Σ_J E_J' Hs_inv_J E_J
    // added to P). So we get the direct-x Hessian from the slack machinery
    // by calling it with `(s, z)` reversed. The scalar view makes this
    // obvious: nonneg slack stores `w² = s/z`; feeding it `(z_x, x)`
    // produces `z_x/x[J] = Hs_inv` — exactly what the (1,1) block needs.
    //
    // Every default below is a one-line delegation that applies this
    // primal↔dual swap at the slack-call boundary. Nothing else in the
    // direct-x pipeline needs to know about the swap — w, λ, η, Hs are
    // all stored with the dual interpretation baked in, and downstream
    // step-math uses them consistently.
    //
    // # Asymmetric cones (exp, power, genpow) — out of v1 scope
    //
    // For asymmetric cones `F ≠ F*` and `K ≠ K*`. Swapping args doesn't
    // give the dual barrier; the correct direct-x Hessian needs primal-
    // barrier-specific math. A cone that wants direct-x membership of an
    // asymmetric variable must override the defaults below. The slack-
    // delegation defaults would silently compute the DUAL barrier
    // Hessian applied to an x-step, which is wrong.
    // ---------------------------------------------------------

    /// Whether the direct-x augmented (1,1) block uses sparse expansion
    /// columns. Defaults to the slack value (`is_sparse_expandable`); a
    /// cone may override if its direct-x usage demands a different
    /// representation than its slack usage. Asymmetric direct-x cones
    /// that store `μ·H_primal(x)` instead of slack's rank-3
    /// `μ·H_dual(z)` should override to `false` and use the Dense block
    /// path (since the slack rank-3 expansion encodes
    /// `(d1, d2, p, q, r)` from the dual barrier — *wrong* for direct-x
    /// primal-IPM augmentation).
    fn direct_x_is_sparse_expandable(&self) -> bool {
        self.is_sparse_expandable()
    }

    /// Whether the direct-x augmented (1,1) block contribution from this
    /// cone is diagonal. Defaults to the slack value (`Hs_is_diagonal`).
    /// Cones whose direct-x Hs differs in sparsity from slack (e.g.
    /// GenPowerCone — slack diagonal+rank-3, direct-x dense) should
    /// override.
    fn direct_x_Hs_is_diagonal(&self) -> bool {
        self.Hs_is_diagonal()
    }

    /// Update the NT scaling for a direct-x `(x, z)` primal-dual pair.
    /// `x` is constrained to the cone; `z` is its dual.
    fn direct_x_update_scaling(
        &mut self,
        x: &[T],
        z: &[T],
        μ: T,
        scaling_strategy: ScalingStrategy,
    ) -> bool {
        // Symmetric default: apply the primal↔dual swap (see block
        // comment above). Slack's `update_scaling(s, z)` becomes our
        // `update_scaling(z, x)` — the stored W² picks up the dual-side
        // interpretation, i.e. Hs_inv in the direct-x sense.
        self.update_scaling(z, x, μ, scaling_strategy)
    }

    /// Fill the Hs block with the direct-x (1,1) contribution. After
    /// `direct_x_update_scaling` this is numerically `Hs_inv` of the
    /// primal barrier (see block comment above).
    fn direct_x_get_Hs(&self, Hsblock: &mut [T]) {
        self.get_Hs(Hsblock)
    }

    /// Compute `y = Hs · x_step` in the direct-x sense. Used in post-KKT
    /// recovery `Δz = -Hs·Δx[J] - c_J` where `Hs` here is the direct-x
    /// Hessian (i.e. `Hs_inv` of the primal barrier; see block comment).
    fn direct_x_mul_Hs(&mut self, y: &mut [T], x_step: &[T], work: &mut [T]) {
        self.mul_Hs(y, x_step, work)
    }

    /// Fill the direct-x z_x-slot with the affine centrality residual
    /// (analog of slack `affine_ds`).
    fn direct_x_affine_ds(&self, out: &mut [T], z: &[T]) {
        self.affine_ds(out, z)
    }

    /// Combined-step shift for the direct-x centrality linearization.
    /// `step_x`, `step_z` are the affine primal/dual steps.
    fn direct_x_combined_ds_shift(
        &mut self,
        shift: &mut [T],
        step_x: &mut [T],
        step_z: &mut [T],
        σμ: T,
    ) {
        // Symmetric default: apply the primal↔dual swap (see block
        // comment above). Slack `combined_ds_shift(shift, step_z, step_s)`
        // becomes our `combined_ds_shift(shift, step_x, step_z)` — the
        // primal step `step_x` takes slack's `step_z` (z-side W-action)
        // slot and the dual step `step_z` takes slack's `step_s` slot.
        self.combined_ds_shift(shift, step_x, step_z, σμ)
    }

    /// Affine-step offset `c_J` for the direct-x KKT RHS adjustment
    /// `workx -= Σ_J E_J' c_J` (analog of slack's
    /// `Δs_const_term = variables.s`).
    fn direct_x_affine_offset(&self, out: &mut [T], z: &[T]) {
        // Symmetric default (per block comment): c_J_affine for slack
        // is `s` (primal slot). Applying the primal↔dual swap gives
        // c_J_affine = z on the direct-x side — the dual slot plays
        // slack's primal role under self-duality.
        out.copy_from_slice(z);
    }

    /// Combined/centering-step offset `c_J` for the direct-x KKT RHS
    /// adjustment.
    fn direct_x_combined_offset(&mut self, out: &mut [T], ds: &[T], work: &mut [T], x: &[T]) {
        // Symmetric default: apply the primal↔dual swap (see block
        // comment above). Slack `Δs_from_Δz_offset(out, ds, work, z)`
        // uses `z` as the dual-slot argument; direct-x's dual is `x[J]`,
        // so we pass `x` into that slot.
        self.Δs_from_Δz_offset(out, ds, work, x)
    }

    /// Step length along (Δx, Δz) for a direct-x cone.
    fn direct_x_step_length(
        &mut self,
        dx: &[T],
        dz: &[T],
        x: &[T],
        z: &[T],
        settings: &CoreSettings<T>,
        αmax: T,
    ) -> (T, T) {
        // Symmetric default: apply the primal↔dual swap (see block
        // comment). Slack `step_length(dz, ds, z, s)` — which walks
        // along the (z-dual, s-primal) pair — is invoked with direct-x
        // primal `(dx, x)` in slack's `(dz, z)` slot and direct-x dual
        // `(dz, z)` in slack's `(ds, s)` slot.
        self.step_length(dx, dz, x, z, settings, αmax)
    }

    /// Evaluate the barrier at (x+αΔx, z+αΔz) for direct-x.
    fn direct_x_compute_barrier(&mut self, x: &[T], z: &[T], dx: &[T], dz: &[T], α: T) -> T {
        // Symmetric default: apply the primal↔dual swap (see block
        // comment). Slack `compute_barrier(z, s, dz, ds)` reads from
        // (z-dual, s-primal); we pass direct-x primal `(x, dx)` into
        // slack's z-slot and direct-x dual `(z, dz)` into slack's s-slot.
        // Self-duality of the barrier makes this value invariant under
        // the swap.
        self.compute_barrier(x, z, dx, dz, α)
    }

    /// Initialize `(x, z)` to unit values for a direct-x cone.
    fn direct_x_unit_initialization(&self, x: &mut [T], z: &mut [T]) {
        // Symmetric default: apply the primal↔dual swap (see block
        // comment). Slack `unit_initialization(z, s)` initializes the
        // (dual, primal) pair; direct-x's (primal, dual) is `(x, z)`,
        // so we swap into slack's (dual=x's role filled by z, primal=
        // z's role filled by x) convention.
        self.unit_initialization(z, x)
    }

    /// Smoothing function (reserved for internal use).
    ///
    /// Projects (z, s) to cone interiors while preserving w = z - s.
    /// The input `work` contains w = z - s. After calling:
    /// - `z` is updated to the smoothed z value
    /// - `s` should be recovered as s = z - w
    ///
    /// The smoothing ensures both z and s are strictly in the cone interior
    /// with margin proportional to μ.
    fn smoothing(&mut self, z: &mut [T], s: &[T], work: &[T], μ: T);
}

#[test]
fn numel_degree() {
    use crate::solver::core::cones::*;

    let zcone = ZeroCone::<f64>::new(5);
    let nncone = NonnegativeCone::<f64>::new(5);
    let scone = SecondOrderCone::<f64>::new(3);
    let expcone = ExponentialCone::<f64>::new();
    let powcone = PowerCone::<f64>::new(0.5);

    assert_eq!(zcone.numel(), 5);
    assert_eq!(zcone.degree(), 0);
    assert_eq!(nncone.numel(), 5);
    assert_eq!(nncone.degree(), 5);
    assert_eq!(scone.numel(), 3);
    assert_eq!(scone.degree(), 1);
    assert_eq!(expcone.numel(), 3);
    assert_eq!(expcone.degree(), 3);
    assert_eq!(powcone.numel(), 3);
    assert_eq!(powcone.degree(), 3);

    #[cfg(feature = "sdp")]
    {
        let sdpcone = PSDTriangleCone::<f64>::new(5);
        assert_eq!(sdpcone.numel(), 15);
        assert_eq!(sdpcone.degree(), 5);
    }
}
