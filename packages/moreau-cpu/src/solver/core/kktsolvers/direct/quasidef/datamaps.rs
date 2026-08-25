#![allow(non_snake_case)]

use super::*;
use crate::solver::core::cones::*;
use enum_dispatch::*;

#[enum_dispatch(SparseExpansionMapTrait)]
pub(crate) enum SparseExpansionMap {
    SOCExpansionMap(SOCExpansionMap),
    GenPowExpansionMap(GenPowExpansionMap),
}

#[enum_dispatch(SparseExpansionConeTrait<T>)]
pub(crate) enum SparseExpansionCone<'a, T>
where
    T: FloatT,
{
    SecondOrderCone(&'a SecondOrderCone<T>),
    GenPowerCone(&'a GenPowerCone<T>),
}

impl<T> SupportedCone<T>
where
    T: FloatT,
{
    pub(crate) fn to_sparse_expansion(&self) -> Option<SparseExpansionCone<'_, T>> {
        match self {
            SupportedCone::SecondOrderCone(sc) => Some(SparseExpansionCone::SecondOrderCone(sc)),
            SupportedCone::GenPowerCone(sc) => Some(SparseExpansionCone::GenPowerCone(sc)),
            _ => None,
        }
    }
}

#[enum_dispatch]
pub(crate) trait SparseExpansionMapTrait {
    fn pdim(&self) -> usize;
    fn nnz_vec(&self) -> usize;
    fn Dsigns(&self) -> &[i8];
    /// True if this cone wants its expansion-variable rows eliminated
    /// *first* in the LDL (bypassing AMD's choice). Necessary for
    /// rank-r expansions where individual axes can carry huge magnitudes
    /// (GenPow's rank-6 PD expansion at near-boundary iterates) — AMD
    /// orders things to minimise fill but produces an LDL pivot
    /// trajectory with `min|D|` near zero, ruining FP precision of
    /// intermediate Schur corrections. SOC's rank-2 expansion has
    /// bounded axis magnitudes and is well-behaved under AMD, so it
    /// keeps its default `false`.
    fn wants_expansion_first(&self) -> bool {
        false
    }
}

impl SparseExpansionMapTrait for Vec<SparseExpansionMap> {
    fn pdim(&self) -> usize {
        self.iter().fold(0, |pdim, map| pdim + map.pdim())
    }
    fn nnz_vec(&self) -> usize {
        self.iter().fold(0, |nnz, map| nnz + map.nnz_vec())
    }
    fn Dsigns(&self) -> &[i8] {
        unreachable!()
    }
    fn wants_expansion_first(&self) -> bool {
        self.iter().any(|m| m.wants_expansion_first())
    }
}

type UpdateFcn<T> = fn(&mut BoxedDirectLDLSolver<T>, &mut CscMatrix<T>, &[usize], &[T]) -> ();
type ScaleFcn<T> = fn(&mut BoxedDirectLDLSolver<T>, &mut CscMatrix<T>, &[usize], T) -> ();

#[enum_dispatch]
pub(crate) trait SparseExpansionConeTrait<T>
where
    T: FloatT,
{
    fn expansion_map(&self) -> SparseExpansionMap;
    fn csc_colcount_sparsecone(
        &self,
        map: &SparseExpansionMap,
        K: &mut CscMatrix<T>,
        row: usize,
        col: usize,
        shape: MatrixTriangle,
    );
    fn csc_fill_sparsecone(
        &self,
        map: &mut SparseExpansionMap,
        K: &mut CscMatrix<T>,
        row: usize,
        col: usize,
        shape: MatrixTriangle,
    );
    fn csc_update_sparsecone(
        &self,
        map: &SparseExpansionMap,
        ldl: &mut BoxedDirectLDLSolver<T>,
        K: &mut CscMatrix<T>,
        updateFcn: UpdateFcn<T>,
        scaleFcn: ScaleFcn<T>,
    );
    /// Fill the current expected diagonal signs for this cone's expansion
    /// rows. `out.len()` must equal the cone's `pdim()`. Static-sign cones
    /// just copy from the map's `Dsigns()`; PD-scaled cones whose signs
    /// depend on per-iteration coefficients return the live values here.
    fn fill_dsigns(&self, map: &SparseExpansionMap, out: &mut [i8]);
}

macro_rules! impl_map_recover {
    ($CONE:ident,$MAP:ident) => {
        impl<'a, T: FloatT> $CONE<T> {
            pub(crate) fn recover_map(&self, map: &'a SparseExpansionMap) -> &'a $MAP {
                match map {
                    SparseExpansionMap::$MAP(map) => map,
                    _ => panic!(),
                }
            }
            pub(crate) fn recover_map_mut(&self, map: &'a mut SparseExpansionMap) -> &'a mut $MAP {
                match map {
                    SparseExpansionMap::$MAP(map) => map,
                    _ => panic!(),
                }
            }
        }
    };
}

//--------------------------------------
// Second order cone data map
//--------------------------------------

pub(crate) struct SOCExpansionMap {
    pub(crate) u: Vec<usize>, //off diag dense columns u
    pub(crate) v: Vec<usize>, //off diag dense columns v
    pub(crate) D: [usize; 2], //diag D
}

impl SOCExpansionMap {
    pub fn new<T: FloatT>(cone: &SecondOrderCone<T>) -> Self {
        let u = vec![0; cone.numel()];
        let v = vec![0; cone.numel()];
        let D = [0; 2];
        Self { u, v, D }
    }
}

impl SparseExpansionMapTrait for SOCExpansionMap {
    fn pdim(&self) -> usize {
        2
    }
    fn nnz_vec(&self) -> usize {
        2 * self.v.len()
    }
    fn Dsigns(&self) -> &[i8] {
        &[-1, 1]
    }
}

impl_map_recover!(SecondOrderCone, SOCExpansionMap);

impl<T> SparseExpansionConeTrait<T> for &'_ SecondOrderCone<T>
where
    T: FloatT,
{
    fn expansion_map(&self) -> SparseExpansionMap {
        SparseExpansionMap::SOCExpansionMap(SOCExpansionMap::new(self))
    }

    fn csc_colcount_sparsecone(
        &self,
        map: &SparseExpansionMap,
        K: &mut CscMatrix<T>,
        row: usize,
        col: usize,
        shape: MatrixTriangle,
    ) {
        let map = self.recover_map(map);
        let nvars = self.numel();

        match shape {
            MatrixTriangle::Triu => {
                K.colcount_colvec(nvars, row, col); // u column
                K.colcount_colvec(nvars, row, col + 1); // v column
            }
            MatrixTriangle::Tril => {
                K.colcount_rowvec(nvars, col, row); // u row
                K.colcount_rowvec(nvars, col + 1, row); // v row
            }
        }
        K.colcount_diag(col, map.pdim());
    }

    fn csc_fill_sparsecone(
        &self,
        map: &mut SparseExpansionMap,
        K: &mut CscMatrix<T>,
        row: usize,
        col: usize,
        shape: MatrixTriangle,
    ) {
        let map = self.recover_map_mut(map);

        // fill structural zeros for u and v columns for this cone
        // note v is the first extra row/column, u is second
        match shape {
            MatrixTriangle::Triu => {
                K.fill_colvec(&mut map.v, row, col); //u
                K.fill_colvec(&mut map.u, row, col + 1); //v
            }
            MatrixTriangle::Tril => {
                K.fill_rowvec(&mut map.v, col, row); //u
                K.fill_rowvec(&mut map.u, col + 1, row); //v
            }
        }
        let pdim = map.pdim();
        K.fill_diag(&mut map.D, col, pdim);
    }

    fn csc_update_sparsecone(
        &self,
        map: &SparseExpansionMap,
        ldl: &mut BoxedDirectLDLSolver<T>,
        K: &mut CscMatrix<T>,
        updateFcn: UpdateFcn<T>,
        scaleFcn: ScaleFcn<T>,
    ) {
        let sparse_data = self.sparse_data.as_ref().unwrap();

        let map = self.recover_map(map);
        let η2 = self.η * self.η;

        // off diagonal columns (or rows)
        updateFcn(ldl, K, &map.u, &sparse_data.u);
        updateFcn(ldl, K, &map.v, &sparse_data.v);
        scaleFcn(ldl, K, &map.u, -η2);
        scaleFcn(ldl, K, &map.v, -η2);

        //set diagonal to η^2*(-1,1) in the extended rows/cols
        updateFcn(ldl, K, &map.D, &[-η2, η2]);
    }

    fn fill_dsigns(&self, map: &SparseExpansionMap, out: &mut [i8]) {
        out.copy_from_slice(map.Dsigns());
    }
}

//--------------------------------------
// Generalized power cone data map
//--------------------------------------

pub(crate) struct GenPowExpansionMap {
    // Existing rank-3 form `H_dual = D + p p' - q q' - r r'`.
    pub(crate) p: Vec<usize>, //off diag dense columns p
    pub(crate) q: Vec<usize>, //off diag dense columns q
    pub(crate) r: Vec<usize>, //off diag dense columns r
    // PD-scaling rank-6 QR-based sparse expansion (4 projector axes
    // + 2 secant axes, see `nonsymmetric_common::PdScalingNdQr6`).
    // All 6 slots have dynamic signs (signs flip with eigenvalue
    // signs from the 4×4 projector eigendecomposition; secant axes
    // default to `+`). Each axis is length `numel()`. Zero-filled
    // until `update_scaling` populates them on `PrimalDual` strategy.
    pub(crate) pd_axes: [Vec<usize>; 6],
    pub(crate) D: [usize; 9], //diag D — order: q, r, p, then 6 PD axes
}

impl GenPowExpansionMap {
    pub fn new<T: FloatT>(cone: &GenPowerCone<T>) -> Self {
        let p = vec![0; cone.numel()];
        let q = vec![0; cone.dim1()];
        let r = vec![0; cone.dim2()];
        let pd_axes: [Vec<usize>; 6] = std::array::from_fn(|_| vec![0; cone.numel()]);
        let D = [0; 9];
        Self {
            p,
            q,
            r,
            pd_axes,
            D,
        }
    }
}

impl SparseExpansionMapTrait for GenPowExpansionMap {
    fn pdim(&self) -> usize {
        9
    }
    fn nnz_vec(&self) -> usize {
        let pd_nnz: usize = self.pd_axes.iter().map(|v| v.len()).sum();
        self.p.len() + self.q.len() + self.r.len() + pd_nnz
    }
    fn Dsigns(&self) -> &[i8] {
        // Default signs at construction; all 6 PD slots are dynamic
        // (eigenvalue signs from the 4×4 projector decomp + secant
        // axes default `+`). Order: q (−), r (−), p (+), then 6 PD
        // slots laid out as in `PdScalingNdQr6`.
        &[
            -1, -1, 1, // q, r, p
            1, 1, 1, 1, // 4 projector eigenvector axes (default +)
            1, 1, // s, δs (secant axes)
        ]
    }
    fn wants_expansion_first(&self) -> bool {
        // The rank-6 PD expansion's eigenvalues `λ_k` blow up
        // (∝ `‖μH‖`) at near-boundary iterates. AMD on the augmented
        // matrix gives an LDL pivot trajectory with `min|D|` collapsing
        // to ~`1e-8` (vs `1.0` for the equivalent dense block), making
        // intermediate Schur corrections lose FP precision. Eliminating
        // expansion variables first restores `min|D| ≈ 1.0`.
        true
    }
}

impl_map_recover!(GenPowerCone, GenPowExpansionMap);

impl<T> SparseExpansionConeTrait<T> for &'_ GenPowerCone<T>
where
    T: FloatT,
{
    fn expansion_map(&self) -> SparseExpansionMap {
        SparseExpansionMap::GenPowExpansionMap(GenPowExpansionMap::new(self))
    }

    fn csc_colcount_sparsecone(
        &self,
        map: &SparseExpansionMap,
        K: &mut CscMatrix<T>,
        row: usize,
        col: usize,
        shape: MatrixTriangle,
    ) {
        let map = self.recover_map(map);
        let nvars = self.numel();
        let dim1 = self.dim1();
        let dim2 = self.dim2();

        match shape {
            MatrixTriangle::Triu => {
                K.colcount_colvec(dim1, row, col); //q column
                K.colcount_colvec(dim2, row + dim1, col + 1); //r column
                K.colcount_colvec(nvars, row, col + 2); //p column
                for k in 0..6 {
                    K.colcount_colvec(nvars, row, col + 3 + k); // PD axis k
                }
            }
            MatrixTriangle::Tril => {
                K.colcount_rowvec(dim1, col, row); //q row
                K.colcount_rowvec(dim2, col + 1, row + dim1); //r row
                K.colcount_rowvec(nvars, col + 2, row); //p row
                for k in 0..6 {
                    K.colcount_rowvec(nvars, col + 3 + k, row); // PD axis k
                }
            }
        }
        K.colcount_diag(col, map.pdim());
    }

    fn csc_fill_sparsecone(
        &self,
        map: &mut SparseExpansionMap,
        K: &mut CscMatrix<T>,
        row: usize,
        col: usize,
        shape: MatrixTriangle,
    ) {
        let map = self.recover_map_mut(map);
        let dim1 = self.dim1();

        match shape {
            MatrixTriangle::Triu => {
                K.fill_colvec(&mut map.q, row, col); //q column
                K.fill_colvec(&mut map.r, row + dim1, col + 1); //r column
                K.fill_colvec(&mut map.p, row, col + 2); //p column
                for k in 0..6 {
                    K.fill_colvec(&mut map.pd_axes[k], row, col + 3 + k);
                }
            }
            MatrixTriangle::Tril => {
                K.fill_rowvec(&mut map.q, col, row); //q row
                K.fill_rowvec(&mut map.r, col + 1, row + dim1); //r row
                K.fill_rowvec(&mut map.p, col + 2, row); //p row
                for k in 0..6 {
                    K.fill_rowvec(&mut map.pd_axes[k], col + 3 + k, row);
                }
            }
        }
        let pdim = map.pdim();
        K.fill_diag(&mut map.D, col, pdim);
    }

    fn csc_update_sparsecone(
        &self,
        map: &SparseExpansionMap,
        ldl: &mut BoxedDirectLDLSolver<T>,
        K: &mut CscMatrix<T>,
        updateFcn: UpdateFcn<T>,
        scaleFcn: ScaleFcn<T>,
    ) {
        let map = self.recover_map(map);
        let data = &self.data;
        let sqrtμ = data.μ.sqrt();

        // ---------- Adaptive per-axis equilibration ----------
        //
        // Schur on the augmented sentinel block gives K = K_top − E^T D⁻¹ E.
        // For each axis with E_col = c·v and sentinel d, the rank-1
        // contribution is (-c²/d)·v v'. Preserving the math means any
        // (c, d) with `-c²/d = λ` (the axis's signed weight) works.
        //
        // The original convention `c² = μ` (or `|coef_k|`), `d = ±1`
        // works fine when the per-axis weight `μ·||v||²` (≈ |E_col|²) is
        // moderate — the off-diagonal magnitudes match the sentinel and
        // AMD-LDL has no trouble. It only breaks on genpow's near-boundary
        // iterates where `||v||²` reaches 1e10+ and the off-diagonals
        // dwarf the ±1 sentinels.
        //
        // We equilibrate axis-by-axis based on the per-axis weight `w`
        // (the |E_col|² magnitude under the original convention):
        //   • Below threshold: leave alone — original ±sqrtμ·v / ±1
        //     convention is already well-balanced. Switching to
        //     equilibrated entries makes the matrix have *mixed*
        //     magnitudes (some equilibrated, some not), which is worse
        //     for AMD than uniformly leaving everything alone.
        //   • Above threshold: rescale to unit-norm off-diag with
        //     sentinel ±1/w, bounding both sides of the matrix.
        //
        // Threshold = 1e12 — the sweet spot from a sweep across 12
        // values on the genpow parity bench. Across 48 skewed configs
        // at dim 16×8 / 32×16 / 64×32 / 128×64:
        //
        //   threshold | Solved | AS | IP | NumErr | total iter
        //   ----------|--------|----|----|--------|-----------
        //         1e0 |      3 | 22 | 23 |      0 |     3110
        //         1e8 |      8 | 17 | 23 |      0 |     2565
        //        1e11 |     13 | 31 |  4 |      0 |      955
        //   **  1e12  |     16 | 32 |  0 |      0 |      944  **
        //        1e13 |     16 | 31 |  1 |      0 |      978
        //   ~baseline |     16 | 19 | 13 |      0 |     3461
        //
        // At 1e12, every config terminates Solved or AlmostSolved (no
        // InsufficientProgress, no NumericalError) in 3.7× fewer
        // iterations than the un-equilibrated baseline. Below 1e10 the
        // threshold catches moderate axes too — the resulting mixed
        // (some equilibrated, some original) matrix is *worse* than
        // either uniform extreme because AMD struggles with
        // wildly-varying entry magnitudes within the same factor.
        let threshold = T::from_f64(1e12).unwrap();
        let q_norm_sq = data.q.iter().fold(T::zero(), |a, &x| a + x * x);
        let r_norm_sq = data.r.iter().fold(T::zero(), |a, &x| a + x * x);
        let p_norm_sq = data.p.iter().fold(T::zero(), |a, &x| a + x * x);
        let q_w = data.μ * q_norm_sq;
        let r_w = data.μ * r_norm_sq;
        let p_w = data.μ * p_norm_sq;

        let one = T::one();
        let safe_inv_sqrt = |x: T| -> T {
            if x > T::epsilon() {
                T::one() / x.sqrt()
            } else {
                T::zero()
            }
        };

        updateFcn(ldl, K, &map.q, &data.q);
        updateFcn(ldl, K, &map.r, &data.r);
        updateFcn(ldl, K, &map.p, &data.p);
        let q_scale = if q_w > threshold {
            -safe_inv_sqrt(q_norm_sq)
        } else {
            -sqrtμ
        };
        let r_scale = if r_w > threshold {
            -safe_inv_sqrt(r_norm_sq)
        } else {
            -sqrtμ
        };
        let p_scale = if p_w > threshold {
            -safe_inv_sqrt(p_norm_sq)
        } else {
            -sqrtμ
        };
        scaleFcn(ldl, K, &map.q, q_scale);
        scaleFcn(ldl, K, &map.r, r_scale);
        scaleFcn(ldl, K, &map.p, p_scale);

        // PD-scaling rank-6 axes: same per-axis test with weight
        // `|coef_k|·||pd_axes[k]||²`.
        let mut pd_norm_sq = [T::zero(); 6];
        let mut pd_w = [T::zero(); 6];
        for k in 0..6 {
            let n_sq = data.pd_axes[k].iter().fold(T::zero(), |a, &x| a + x * x);
            pd_norm_sq[k] = n_sq;
            let coef_abs = data.pd_coefs[k].abs();
            pd_w[k] = coef_abs * n_sq;
            let scale = if pd_w[k] > threshold {
                -safe_inv_sqrt(n_sq)
            } else {
                -coef_abs.sqrt()
            };
            updateFcn(ldl, K, &map.pd_axes[k], &data.pd_axes[k]);
            scaleFcn(ldl, K, &map.pd_axes[k], scale);
        }

        // Sentinels match the per-axis decision: ±1 (original) below
        // threshold, ±1/w (equilibrated) above. Sign is preserved.
        let signs = &data.pd_signs;
        let q_sent = if q_w > threshold { -one / q_w } else { -one };
        let r_sent = if r_w > threshold { -one / r_w } else { -one };
        let p_sent = if p_w > threshold { one / p_w } else { one };
        let mut pd_sents = [T::zero(); 6];
        for k in 0..6 {
            let sign = T::from_i8(signs[k]).unwrap();
            pd_sents[k] = if pd_w[k] > threshold {
                sign / pd_w[k]
            } else {
                sign
            };
        }
        updateFcn(
            ldl,
            K,
            &map.D,
            &[
                q_sent,
                r_sent,
                p_sent,
                pd_sents[0],
                pd_sents[1],
                pd_sents[2],
                pd_sents[3],
                pd_sents[4],
                pd_sents[5],
            ],
        );
    }

    fn fill_dsigns(&self, map: &SparseExpansionMap, out: &mut [i8]) {
        // q, r, p signs are static; the 6 PD slots come from the cone's
        // current per-iteration sign cache.
        debug_assert_eq!(out.len(), map.pdim());
        let _ = map;
        out[0] = -1;
        out[1] = -1;
        out[2] = 1;
        out[3..9].copy_from_slice(&self.data.pd_signs);
    }
}

//--------------------------------------
// Direct-x SOC rank-2 expansion map
//--------------------------------------

/// Per-direct-x-sparse-cone map into the KKT `nzval` buffer.
///
/// SOC variant: rank-2 expansion (mirrors [`SOCExpansionMap`]).
/// `u`/`v` entries are placed at scattered rows in the (1,1) block (the
/// direct-x cone's `x`-indices) instead of contiguous rows in the (2,2)
/// slack block. The sign convention is opposite of slack's (direct-x
/// adds `+Hs` to the (1,1) block, slack adds `-Hs` to the (2,2)).
///
/// GenPow variant: rank-3 expansion. `q` (length dim1) covers the
/// "p"-block of cone indices, `r` (length dim2) the "w"-block, and `p_v`
/// (length dim) covers all indices. The cone primitive's `direct_x_*`
/// hooks fill `(d1, d2, p, q, r)` from the *primal* barrier so the
/// rank-3 update represents `H_primal(x)`.
pub(crate) enum DirectXSparseMap {
    SOC(DirectXSparseMapSOC),
    GenPow(DirectXSparseMapGenPow),
}

pub(crate) struct DirectXSparseMapSOC {
    /// nzval indices of the `u` column (k entries) in sorted-row order.
    pub u: Vec<usize>,
    /// nzval indices of the `v` column in sorted-row order.
    pub v: Vec<usize>,
    /// nzval indices of the two diagonal entries of the expansion cols.
    pub D: [usize; 2],
    /// `cone_pos_for_sorted[p]` = cone-internal position whose index
    /// ended up at sorted position `p`.
    pub cone_pos_for_sorted: Vec<usize>,
}

pub(crate) struct DirectXSparseMapGenPow {
    /// nzval indices of the `q` column (dim1 entries, scattered into
    /// the cone's first dim1 indices in ascending row order).
    pub q: Vec<usize>,
    /// nzval indices of the `r` column (dim2 entries, scattered into
    /// the cone's last dim2 indices in ascending row order).
    pub r: Vec<usize>,
    /// nzval indices of the `p` column (dim entries, scattered into all
    /// cone indices in ascending row order).
    pub p_v: Vec<usize>,
    /// nzval indices of the 6 PD-scaling rank-6 axes (4 projector + 2
    /// secant from `pd_scaling_nd_qr6`). Each axis is a dim-long
    /// column scattered into all cone indices in ascending row order.
    /// Zero-filled when `pd_active = false`. Mirrors slack's
    /// [`GenPowExpansionMap::pd_axes`].
    pub pd_axes: [Vec<usize>; 6],
    /// nzval indices of the 9 diagonal entries of the expansion cols
    /// (q, r, p, then 6 PD axes).
    pub D: [usize; 9],
    /// Permutation: `cone_pos_for_sorted_dim1[i]` = cone-internal
    /// position (in [0, dim1)) of the index at sorted-dim1 slot `i`.
    pub cone_pos_for_sorted_dim1: Vec<usize>,
    /// Permutation: `cone_pos_for_sorted_dim2[i]` = cone-internal
    /// position (in [dim1, dim1+dim2)) of the index at sorted-dim2 slot `i`.
    pub cone_pos_for_sorted_dim2: Vec<usize>,
    /// Permutation: `cone_pos_for_sorted_all[i]` = cone-internal
    /// position (in [0, dim)) of the index at sorted-all slot `i`.
    pub cone_pos_for_sorted_all: Vec<usize>,
}

impl DirectXSparseMap {
    pub(crate) fn pdim(&self) -> usize {
        match self {
            DirectXSparseMap::SOC(_) => 2,
            DirectXSparseMap::GenPow(_) => 9,
        }
    }
    /// Default sign vector (length `pdim()`) for the expansion-column
    /// entries of the LDL `Dsigns` array. Direct-x adds `+Hs` to the
    /// (1,1) block, so signs are flipped from slack's expansion.
    ///
    /// For GenPow rank-9: the first 3 (q, r, p) are static; the last 6
    /// (PD axes) are dynamic — flipping with `data.pd_signs` per
    /// iteration. The default returned here corresponds to the init
    /// state (`pd_signs` all `+1`, sentinel = `-pd_signs[k] = -1`); the
    /// per-iteration refresh is in `directldlkktsolver::update`.
    pub(crate) fn dsigns(&self) -> &'static [i8] {
        match self {
            DirectXSparseMap::SOC(_) => &[1, -1],
            DirectXSparseMap::GenPow(_) => &[
                1, 1, -1, // q, r, p (rank-3 base, static)
                -1, -1, -1, -1, // 4 projector eigenvector axes (default −)
                -1, -1, // 2 secant axes (default −)
            ],
        }
    }
}

impl DirectXSparseMapSOC {
    pub(crate) fn new(k: usize, cone_pos_for_sorted: Vec<usize>) -> Self {
        Self {
            u: vec![0; k],
            v: vec![0; k],
            D: [0; 2],
            cone_pos_for_sorted,
        }
    }
}

impl DirectXSparseMapGenPow {
    pub(crate) fn new(
        dim1: usize,
        dim2: usize,
        cone_pos_for_sorted_dim1: Vec<usize>,
        cone_pos_for_sorted_dim2: Vec<usize>,
        cone_pos_for_sorted_all: Vec<usize>,
    ) -> Self {
        let dim = dim1 + dim2;
        let pd_axes: [Vec<usize>; 6] = std::array::from_fn(|_| vec![0; dim]);
        Self {
            q: vec![0; dim1],
            r: vec![0; dim2],
            p_v: vec![0; dim],
            pd_axes,
            D: [0; 9],
            cone_pos_for_sorted_dim1,
            cone_pos_for_sorted_dim2,
            cone_pos_for_sorted_all,
        }
    }
}

//--------------------------------------
// LDL Data Map
//--------------------------------------

pub(crate) struct LDLDataMap {
    pub P: Vec<usize>,
    pub A: Vec<usize>,
    pub Hsblocks: Vec<usize>, //indices of the lower RHS blocks (by cone)
    pub sparse_maps: Vec<SparseExpansionMap>, //sparse cone expansion terms

    // Flat indices into KKT.nzval for the stacked Hx buffer produced by the
    // direct-x composite cone (one entry per Hs block position, ordered by
    // cone then by the cone's get_Hs layout). Empty when no direct-x cones.
    // Note: these indices may coincide with entries in `P` when the direct-x
    // footprint overlaps the user-supplied P pattern.
    pub Hxblocks: Vec<usize>,

    // Per-Hxblocks-entry pointer into `P.nzval` when the direct-x footprint
    // coincides with a user-supplied P entry, else `None`. Same length as
    // `Hxblocks`. Used to maintain the `KKT = P + Σ E_J' H_J E_J` invariant
    // at overlap slots: after `update_P`, the Hx slot value is reset to the
    // fresh P value, and `refresh_hx_blocks` additively re-applies the Hs
    // contribution on top.
    pub Hx_to_P: Vec<Option<usize>>,

    // Direct-x SOC rank-2 expansion maps, parallel to `sparse_maps` for
    // slack cones but scattered into `x`-space columns. One entry per
    // direct-x cone (in xcone order); `None` for cones that use the
    // dense Hxblocks path (nonneg, small SOC). Each `Some` entry carries
    // the KKT nzval indices for the expansion's `u`/`v` scattered
    // columns, `D` extra-col diagonals, and the permutation needed to
    // fetch `sparse_data.u[cone_pos]` when refreshing the (sorted)
    // column each iteration.
    pub x_sparse_maps: Vec<Option<DirectXSparseMap>>,

    // all of above terms should be disjoint and their union
    // should cover all of the user data in the KKT matrix.  Now
    // we make two last redundant indices that will tell us where
    // the whole diagonal is, including structural zeros.
    pub diagP: Vec<usize>,
    pub diag_full: Vec<usize>,
}

impl LDLDataMap {
    pub fn new<T: FloatT>(
        Pmat: &CscMatrix<T>,
        Amat: &CscMatrix<T>,
        cones: &CompositeCone<T>,
    ) -> Self {
        let (m, n) = (Amat.nrows(), Pmat.nrows());
        let P = vec![0; Pmat.nnz()];
        let A = vec![0; Amat.nnz()];

        // the diagonal of the ULHS KKT block P.
        // NB : we fill in structural zeros here even if the matrix
        // P is empty (e.g. as in an LP), so we can have entries in
        // index Pdiag that are not present in the index P
        let diagP = vec![0; n];

        // make an index for each of the Hs blocks for each cone
        let Hsblocks = allocate_kkt_Hsblocks::<T, usize>(cones);

        // now do the sparse cone expansion pieces
        let nsparse = cones.iter().filter(|&c| c.is_sparse_expandable()).count();
        let mut sparse_maps = Vec::with_capacity(nsparse);

        for cone in cones.iter() {
            if cone.is_sparse_expandable() {
                let sc = cone.to_sparse_expansion().unwrap();
                sparse_maps.push(sc.expansion_map());
            }
        }

        let diag_full = vec![0; m + n + sparse_maps.pdim()];

        Self {
            P,
            A,
            Hsblocks,
            sparse_maps,
            Hxblocks: Vec::new(),
            Hx_to_P: Vec::new(),
            x_sparse_maps: Vec::new(),
            diagP,
            diag_full,
        }
    }
}
