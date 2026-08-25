#![allow(non_snake_case)]
use super::*;
use crate::algebra::triangular_number;
use std::ops::Range;

// -------------------------------------
// Direct-x composite cone type
//
// Parallel to `CompositeCone`, but stores cones that constrain subvectors
// of `x` directly (rather than slack variables `s`). Each entry pairs a
// cone primitive with a list of indices into `x`. The (1,1) KKT block
// becomes `P + Σ_J E_J' H_J E_J`, where `E_J` scatters indices `J` into
// `x`.
//
// This module provides only the types and collection layout. Callers in
// the KKT assembly / solver are wired up in later commits.
// -------------------------------------

/// User-facing API type describing a direct-x cone constraint.
///
/// Each variant constrains a subvector `x[indices]` to a cone. Indices
/// must be disjoint across all direct-x cones in a problem (enforced at
/// the Python/FFI boundary).
#[derive(Clone, Debug, PartialEq)]
pub enum SupportedXConeT {
    /// Nonnegative orthant on `x[indices]`.
    NonnegativeXConeT(Vec<usize>),
    /// Second-order cone on `x[indices]`. First entry of `indices` maps
    /// to the scalar "t", remaining entries to the vector "v".
    SecondOrderXConeT(Vec<usize>),
    /// Exponential cone on `x[indices]` (length 3). Asymmetric: F ≠ F*,
    /// so the slack-form `direct_x_*` defaults that swap (s, z) → (z, x)
    /// are wrong. `ExponentialCone` overrides those defaults to use the
    /// primal-barrier Hessian at `x` instead.
    ExponentialXConeT(Vec<usize>),
    /// 3D power cone on `x[indices]` with parameter α ∈ (0, 1). Same
    /// asymmetry caveat as ExpCone: `PowerCone` overrides the direct-x
    /// defaults to use the primal-barrier Hessian at `x`.
    PowerXConeT(Vec<usize>, f64),
    /// Generalized power cone on `x[indices]` with weights `alphas`
    /// (length `dim1`, all positive, summing to 1) and tail dimension
    /// `dim2`. Total length: `dim1 + dim2 == indices.len()`. Asymmetric
    /// direct-x: `GenPowerCone` overrides the direct-x defaults to
    /// store `μ·∇²F_primal(x)` as a *dense* dim×dim block (slack form
    /// uses a rank-3 sparse expansion of `μ·∇²F_dual(z)` instead).
    GenPowerXConeT(Vec<usize>, Vec<f64>, usize),
    /// PSD triangle on `x[indices]` with side dimension `psd_k`. Requires
    /// `indices.len() == psd_k * (psd_k + 1) / 2`.
    #[cfg(feature = "sdp")]
    PSDTriangleXConeT(Vec<usize>, usize),
}

impl SupportedXConeT {
    pub fn indices(&self) -> &[usize] {
        match self {
            SupportedXConeT::NonnegativeXConeT(ix) => ix,
            SupportedXConeT::SecondOrderXConeT(ix) => ix,
            SupportedXConeT::ExponentialXConeT(ix) => ix,
            SupportedXConeT::PowerXConeT(ix, _) => ix,
            SupportedXConeT::GenPowerXConeT(ix, _, _) => ix,
            #[cfg(feature = "sdp")]
            SupportedXConeT::PSDTriangleXConeT(ix, _) => ix,
        }
    }

    pub fn numel(&self) -> usize {
        self.indices().len()
    }

    pub fn degree(&self) -> usize {
        match self {
            SupportedXConeT::NonnegativeXConeT(ix) => ix.len(),
            SupportedXConeT::SecondOrderXConeT(_) => 1,
            SupportedXConeT::ExponentialXConeT(_) => 3,
            SupportedXConeT::PowerXConeT(_, _) => 3,
            SupportedXConeT::GenPowerXConeT(_, alphas, _) => alphas.len() + 1,
            #[cfg(feature = "sdp")]
            SupportedXConeT::PSDTriangleXConeT(_, k) => *k,
        }
    }
}

impl SupportedConeAsTag for SupportedXConeT {
    fn as_tag(&self) -> SupportedConeTag {
        match self {
            SupportedXConeT::NonnegativeXConeT(_) => SupportedConeTag::NonnegativeCone,
            SupportedXConeT::SecondOrderXConeT(_) => SupportedConeTag::SecondOrderCone,
            SupportedXConeT::ExponentialXConeT(_) => SupportedConeTag::ExponentialCone,
            SupportedXConeT::PowerXConeT(_, _) => SupportedConeTag::PowerCone,
            SupportedXConeT::GenPowerXConeT(_, _, _) => SupportedConeTag::GenPowerCone,
            #[cfg(feature = "sdp")]
            SupportedXConeT::PSDTriangleXConeT(_, _) => SupportedConeTag::PSDTriangleCone,
        }
    }
}

/// Instantiate the internal cone primitive for a direct-x cone spec.
///
/// The primitive is the same `SupportedCone<T>` used for slack cones —
/// cone math (`update_scaling`, `get_Hs`, `mul_Hs`, ...) is identical;
/// only the KKT-assembly plumbing differs.
pub fn make_x_cone<T: FloatT>(cone: &SupportedXConeT) -> SupportedCone<T> {
    match cone {
        SupportedXConeT::NonnegativeXConeT(ix) => NonnegativeCone::<T>::new(ix.len()).into(),
        SupportedXConeT::SecondOrderXConeT(ix) => {
            // SOC direct-x uses the same rank-2 sparse expansion as slack
            // SOC for large cones (dim > 4). The expansion stores two
            // dense scattered columns + 2 diag entries instead of a
            // k(k+1)/2 dense triu block, turning O(k²) factor work back
            // into O(k). See kkt_assembly.rs for the scatter layout.
            SecondOrderCone::<T>::new(ix.len()).into()
        }
        SupportedXConeT::ExponentialXConeT(_) => {
            // Exp cone is asymmetric and 3D; the cone primitive is the
            // same as slack ExpCone, but the direct-x trait methods are
            // overridden on `ExponentialCone` to use the primal-barrier
            // Hessian at the constrained primal `x` (rather than the
            // dual-barrier Hessian via slack-form swap, which is wrong
            // for asymmetric cones).
            ExponentialCone::<T>::new().into()
        }
        SupportedXConeT::PowerXConeT(_, α) => {
            // Power cone is asymmetric and 3D — same direct-x story as
            // ExpCone. `α` is the cone parameter ∈ (0, 1).
            PowerCone::<T>::new(T::from(*α).unwrap()).into()
        }
        SupportedXConeT::GenPowerXConeT(_, alphas, dim2) => {
            // Generalized power cone: dim = alphas.len() + dim2,
            // alphas > 0 summing to 1. Same primitive as slack but
            // direct-x trait overrides force the Dense (1,1) block.
            let alphas_t: Vec<T> = alphas.iter().map(|&a| T::from(a).unwrap()).collect();
            GenPowerCone::<T>::new(alphas_t, *dim2).into()
        }
        #[cfg(feature = "sdp")]
        SupportedXConeT::PSDTriangleXConeT(_, k) => PSDTriangleCone::<T>::new(*k).into(),
    }
}

/// A single direct-x cone entry.
///
/// Holds the cone math object, the indices into `x` that it constrains,
/// and the range into the stacked Hx buffer where its `get_Hs()` output
/// lives.
pub struct XConeEntry<T: FloatT> {
    pub cone: SupportedCone<T>,
    pub indices: Vec<usize>,
    pub rng_block: Range<usize>,
    /// Sorted copy of `indices` (strictly ascending). Required for
    /// writing the rank-2 sparse-expansion columns — the KKT column
    /// must be row-sorted for CSC validity.
    pub sorted_indices: Vec<usize>,
    /// `cone_pos_for_sorted[p]` is the original cone-internal position
    /// (0..k-1) whose index ended up at sorted position `p`, i.e.
    /// `sorted_indices[p] == indices[cone_pos_for_sorted[p]]`. Used to
    /// fetch `sparse_data.u[cone_pos_for_sorted[p]]` when filling the
    /// KKT column in sorted order.
    pub cone_pos_for_sorted: Vec<usize>,
}

impl<T> XConeEntry<T>
where
    T: FloatT,
{
    pub fn numel(&self) -> usize {
        self.indices.len()
    }
}

/// Collection of direct-x cones.
pub struct CompositeXCone<T: FloatT = f64> {
    entries: Vec<XConeEntry<T>>,
    pub(crate) numel: usize,
    pub(crate) degree: usize,
    pub(crate) hx_block_len: usize,
}

impl<T> CompositeXCone<T>
where
    T: FloatT,
{
    pub fn new(types: &[SupportedXConeT]) -> Self {
        let mut entries = Vec::with_capacity(types.len());
        let mut numel = 0usize;
        let mut degree = 0usize;
        let mut hx_block_len = 0usize;

        for t in types.iter() {
            let cone = make_x_cone::<T>(t);
            let ni = cone.numel();
            assert_eq!(
                t.indices().len(),
                ni,
                "SupportedXConeT indices length ({}) must match cone dimension ({})",
                t.indices().len(),
                ni
            );
            let block_size = if cone.direct_x_Hs_is_diagonal() {
                ni
            } else {
                triangular_number(ni)
            };
            let start = hx_block_len;
            hx_block_len += block_size;
            numel += ni;
            degree += cone.degree();

            // Build the (sorted indices, permutation) pair used by the
            // direct-x sparse expansion. The permutation maps sorted
            // positions back to the cone's internal order so we can
            // fetch `sparse_data.u[cone_pos]` values when refreshing
            // scattered KKT columns each iteration.
            let indices = t.indices().to_vec();
            let mut pairs: Vec<(usize, usize)> =
                indices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
            pairs.sort_by_key(|&(v, _)| v);
            let sorted_indices: Vec<usize> = pairs.iter().map(|&(v, _)| v).collect();
            let cone_pos_for_sorted: Vec<usize> = pairs.iter().map(|&(_, i)| i).collect();

            entries.push(XConeEntry {
                cone,
                indices,
                rng_block: start..hx_block_len,
                sorted_indices,
                cone_pos_for_sorted,
            });
        }

        Self {
            entries,
            numel,
            degree,
            hx_block_len,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn iter(&self) -> std::slice::Iter<'_, XConeEntry<T>> {
        self.entries.iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, XConeEntry<T>> {
        self.entries.iter_mut()
    }

    pub fn numel(&self) -> usize {
        self.numel
    }

    /// True iff every direct-x cone is symmetric (self-dual). The default
    /// `direct_x_*` trait impls assume symmetry; asymmetric cones (exp,
    /// power, gen_power) override the trait methods explicitly. The IPM
    /// branches on overall symmetry — slack composite ∧ direct-x
    /// composite — to choose between identity-scaling and unit-init paths.
    pub fn is_symmetric(&self) -> bool {
        self.entries.iter().all(|e| e.cone.is_symmetric())
    }
    pub fn degree(&self) -> usize {
        self.degree
    }
    pub fn hx_block_len(&self) -> usize {
        self.hx_block_len
    }

    /// Total pdim across all direct-x cones that use sparse expansion.
    /// SOC adds rank-2 (2 cols), GenPow adds rank-9 (3 base + 6 PD axes).
    pub fn x_sparse_pdim(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.cone.direct_x_is_sparse_expandable())
            .map(|e| match &e.cone {
                SupportedCone::SecondOrderCone(_) => 2,
                // GenPow: rank-3 base (q, r, p) + rank-6 PD axes (4 projector
                // + 2 secant from `pd_scaling_nd_qr6`). Mirrors slack's
                // `GenPowExpansionMap::pdim() = 9`.
                SupportedCone::GenPowerCone(_) => 9,
                _ => panic!("direct-x sparse expansion not implemented for this cone"),
            })
            .sum()
    }

    /// Populate the stacked Hx buffer from each entry's direct-x Hessian
    /// contribution to the (1,1) KKT block. Semantically this is the
    /// primal-barrier Hessian on `x[J]` (for symmetric cones equal to
    /// the slack dual-barrier Hessian after the self-dual swap — see
    /// `Cone::direct_x_update_scaling`).
    pub fn get_Hs(&self, hx_block: &mut [T]) {
        assert_eq!(hx_block.len(), self.hx_block_len);
        for entry in &self.entries {
            entry
                .cone
                .direct_x_get_Hs(&mut hx_block[entry.rng_block.clone()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_composite_xcone() {
        let cx = CompositeXCone::<f64>::new(&[]);
        assert_eq!(cx.len(), 0);
        assert!(cx.is_empty());
        assert_eq!(cx.numel(), 0);
        assert_eq!(cx.degree(), 0);
        assert_eq!(cx.hx_block_len(), 0);
    }

    #[test]
    fn test_single_nonneg_xcone_layout() {
        let specs = vec![SupportedXConeT::NonnegativeXConeT(vec![0, 2, 4])];
        let cx = CompositeXCone::<f64>::new(&specs);
        assert_eq!(cx.len(), 1);
        assert_eq!(cx.numel(), 3);
        assert_eq!(cx.degree(), 3);
        // nonneg Hs is diagonal -> 3 entries
        assert_eq!(cx.hx_block_len(), 3);
        let e = cx.iter().next().unwrap();
        assert_eq!(e.indices, vec![0, 2, 4]);
        assert_eq!(e.rng_block, 0..3);
    }

    #[test]
    fn test_soc_xcone_layout_nondiagonal() {
        // SOC Hs is non-diagonal -> triangular_number(n) entries
        let specs = vec![SupportedXConeT::SecondOrderXConeT(vec![1, 3, 5])];
        let cx = CompositeXCone::<f64>::new(&specs);
        assert_eq!(cx.numel(), 3);
        assert_eq!(cx.degree(), 1);
        assert_eq!(cx.hx_block_len(), triangular_number(3)); // = 6
    }

    #[test]
    fn test_mixed_xcones_stacked_layout() {
        let specs = vec![
            SupportedXConeT::NonnegativeXConeT(vec![0, 1]),
            SupportedXConeT::SecondOrderXConeT(vec![2, 3, 4]),
            SupportedXConeT::NonnegativeXConeT(vec![7]),
        ];
        let cx = CompositeXCone::<f64>::new(&specs);
        assert_eq!(cx.len(), 3);
        assert_eq!(cx.numel(), 6);
        // degrees: 2 + 1 + 1
        assert_eq!(cx.degree(), 4);
        // block layout: nonneg(2) + soc_triangle(6) + nonneg(1) = 9
        assert_eq!(cx.hx_block_len(), 2 + 6 + 1);

        let rngs: Vec<_> = cx.iter().map(|e| e.rng_block.clone()).collect();
        assert_eq!(rngs, vec![0..2, 2..8, 8..9]);
    }

    #[test]
    fn test_supported_xconet_accessors() {
        let s = SupportedXConeT::NonnegativeXConeT(vec![3, 7, 11]);
        assert_eq!(s.indices(), &[3, 7, 11]);
        assert_eq!(s.numel(), 3);
        assert_eq!(s.degree(), 3);

        let s = SupportedXConeT::SecondOrderXConeT(vec![0, 1, 2, 3]);
        assert_eq!(s.numel(), 4);
        assert_eq!(s.degree(), 1);
    }

    #[cfg(feature = "sdp")]
    #[test]
    fn test_psd_triangle_xcone_layout() {
        // k=3 -> vech length = 6, Hs dense triangle = triangular_number(6) = 21
        let specs = vec![SupportedXConeT::PSDTriangleXConeT(
            vec![0, 1, 2, 3, 4, 5],
            3,
        )];
        let cx = CompositeXCone::<f64>::new(&specs);
        assert_eq!(cx.numel(), 6);
        assert_eq!(cx.degree(), 3);
        assert_eq!(cx.hx_block_len(), triangular_number(6));
    }
}
