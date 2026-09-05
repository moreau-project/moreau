#![allow(non_snake_case)]
use itertools::izip;

use super::*;
use crate::algebra::*;
use crate::solver::core::{
    cones::{CompositeCone, CompositeXCone, Cone, SupportedXConeT},
    traits::ProblemData,
};
use crate::solver::SupportedConeT;

#[cfg(feature = "sdp")]
use crate::solver::chordal::ChordalInfo;

// ---------------
// Data type for default problem format
// ---------------

/// Standard-form solver type implementing the [`ProblemData`](crate::solver::core::traits::ProblemData) trait
pub struct DefaultProblemData<T> {
    /// The matrix P in the quadratic objective term
    pub P: CscMatrix<T>,
    /// The vector q in the quadratic objective term
    pub q: Vec<T>,
    /// The matrix A in the constraints
    pub A: CscMatrix<T>,
    /// The vector b in the constraints
    pub b: Vec<T>,
    /// Vector of cones in the problem
    pub cones: Vec<SupportedConeT<T>>,
    /// Direct-x cones constraining sub-vectors of `x` to cones directly.
    /// Empty when the problem has only slack cones.
    pub dir_cones: Vec<SupportedXConeT>,
    /// Number of variables
    pub n: usize,
    /// Number of constraints
    pub m: usize,
    /// Equilibration data for the problem
    pub equilibration: DefaultEquilibrationData<T>,

    // unscaled inf norms of linear terms.  Set to "None"
    // during data updating to allow for multiple updates, and
    // then recalculated during solve if needed
    normq: Option<T>,
    normb: Option<T>,

    pub(crate) presolver: Option<Presolver<T>>,

    #[cfg(feature = "sdp")]
    pub(crate) chordal_info: Option<ChordalInfo<T>>,
}

impl<T> DefaultProblemData<T>
where
    T: FloatT,
{
    /// Create a new `DefaultProblemData` object
    pub fn new(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        settings: &DefaultSettings<T>,
    ) -> Self {
        Self::new_with_xcones(P, q, A, b, cones, &[], settings)
    }

    /// Create a new `DefaultProblemData` object including direct-x cones.
    pub fn new_with_xcones(
        P: &CscMatrix<T>,
        q: &[T],
        A: &CscMatrix<T>,
        b: &[T],
        cones: &[SupportedConeT<T>],
        dir_cones: &[SupportedXConeT],
        settings: &DefaultSettings<T>,
    ) -> Self {
        // clean up the cones by consolidating repeated NNs,
        // eliminate empty cones, transform singletons etc
        // this makes a locally owned copy of the cones
        let cones = SupportedConeT::new_collapsed(cones);

        // some caution is required to ensure we take a minimal,
        // but nonzero, number of data copies during presolve steps

        let mut P_new: Option<CscMatrix<T>> = None;
        #[allow(unused_mut)] // mut q_new only needed with chordal
        let mut q_new: Option<Vec<T>> = None;
        let mut A_new: Option<CscMatrix<T>> = None;
        let mut b_new: Option<Vec<T>> = None;
        let mut cones_new: Option<Vec<SupportedConeT<T>>> = None;

        if !P.is_triu() {
            P_new = Some(P.to_triu());
        }

        // presolve : return nothing if disabled or no reduction
        // --------------------------------------
        let presolver = try_presolver(A, b, &cones, settings);

        if let Some(ref presolver) = presolver {
            let (_A_new, _b_new, _cones_new) = presolver.presolve(A, b, &cones);
            (A_new, b_new, cones_new) = (Some(_A_new), Some(_b_new), Some(_cones_new));
        }

        // chordal decomposition : return nothing if disabled or no decomp
        // --------------------------------------
        #[cfg(feature = "sdp")]
        let mut chordal_info = try_chordal_info(A, b, &cones, settings);
        #[cfg(feature = "sdp")]
        if let Some(ref mut chordal_info) = chordal_info {
            let (_P_new, _q_new, _A_new, _b_new, _cones_new) = chordal_info.decomp_augment(
                P_new.as_ref().unwrap_or(P),
                unwrap_and_slice_or_else(&q_new, || q),
                A_new.as_ref().unwrap_or(A),
                unwrap_and_slice_or_else(&b_new, || b),
                settings,
            );
            (P_new, q_new, A_new, b_new, cones_new) = (
                Some(_P_new),
                Some(_q_new),
                Some(_A_new),
                Some(_b_new),
                Some(_cones_new),
            );
        }

        // now make sure we have a clean copy of everything if we
        // haven't made one already.   Necessary since we will scale
        // the internal copy and don't want to step on the user

        let mut P_new = P_new.unwrap_or_else(|| P.clone());
        let q_new = q_new.unwrap_or_else(|| q.to_vec());
        let mut A_new = A_new.unwrap_or_else(|| A.clone());
        let mut b_new = b_new.unwrap_or_else(|| b.to_vec());

        // cones was already copied, so can just pass through without cloning
        let cones_new = cones_new.unwrap_or(cones);

        //cap entries in b at INFINITY.  This is important
        //for inf values that were not in a reduced cone
        //this is not considered part of the "presolve", so
        //can always happen regardless of user settings
        let infbound = crate::get_infinity().as_T();
        b_new.scalarop(|x| T::min(x, infbound));

        // this ensures m is the *reduced* size m
        let (m, n) = A_new.size();

        // explicitly dropzeros on the copied data, since dropzeros
        // operates in place.  revisit this order of operations
        // once a proper presolver is implemented, since it might
        // be preferable to dropzeros then presolve
        if settings.ipm.input_sparse_dropzeros {
            P_new.dropzeros();
            A_new.dropzeros();
        }

        let equilibration = DefaultEquilibrationData::<T>::new(n, m);

        let normq = Some(q_new.norm_inf());
        let normb = Some(b_new.norm_inf());

        Self {
            P: P_new,
            q: q_new,
            A: A_new,
            b: b_new,
            cones: cones_new,
            dir_cones: dir_cones.to_vec(),
            n,
            m,
            equilibration,
            normq,
            normb,
            presolver,
            #[cfg(feature = "sdp")]
            chordal_info,
        }
    }

    /// Build a `CompositeXCone<T>` from the stored direct-x cone specs.
    pub fn composite_dir_cones(&self) -> CompositeXCone<T> {
        CompositeXCone::<T>::new(&self.dir_cones)
    }

    pub(crate) fn get_normq(&mut self) -> T {
        if let Some(norm) = self.normq {
            norm
        } else {
            let dinv = &self.equilibration.dinv;
            let cinv = T::recip(self.equilibration.c);
            let norm = self.q.norm_inf_scaled(dinv) * cinv;
            self.normq = Some(norm);
            norm
        }
    }

    pub(crate) fn get_normb(&mut self) -> T {
        if let Some(norm) = self.normb {
            norm
        } else {
            let einv = &self.equilibration.einv;
            let norm = self.b.norm_inf_scaled(einv);
            self.normb = Some(norm);
            norm
        }
    }

    pub(crate) fn clear_normq(&mut self) {
        self.normq = None;
    }

    pub(crate) fn clear_normb(&mut self) {
        self.normb = None;
    }
}

impl<T> ProblemData<T> for DefaultProblemData<T>
where
    T: FloatT,
{
    type V = DefaultVariables<T>;
    type C = CompositeCone<T>;
    type SE = DefaultSettings<T>;

    fn equilibrate(&mut self, cones: &CompositeCone<T>, settings: &DefaultSettings<T>) {
        let data = self;
        let equil = &mut data.equilibration;

        // if equilibration is disabled, just return.  Note that
        // the default equilibration structure initializes with
        // identity scaling already.
        if !settings.ipm.equilibrate_enable {
            return;
        }

        // references to scaling matrices from workspace
        let (d, e) = (&mut equil.d, &mut equil.e);

        // use the inverse scalings as work vectors
        let dwork = &mut equil.dinv;
        let ework = &mut equil.einv;

        // references to problem data
        // note that P may be triu, but it shouldn't matter
        let (P, A, q, b) = (&mut data.P, &mut data.A, &mut data.q, &mut data.b);

        let scale_min = settings.ipm.equilibrate_min_scaling;
        let scale_max = settings.ipm.equilibrate_max_scaling;

        // perform scaling operations for a fixed number of steps
        for _iter in 0..settings.ipm.equilibrate_max_iter {
            kkt_col_norms(P, A, dwork, ework);

            //zero rows or columns should not get scaled
            dwork.scalarop(|x| if x == T::zero() { T::one() } else { x });
            ework.scalarop(|x| if x == T::zero() { T::one() } else { x });

            dwork.rsqrt();
            ework.rsqrt();

            // bound the cumulative scaling
            for (dwork, &d) in izip!(dwork.iter_mut(), d.iter()) {
                *dwork = T::clip(dwork, scale_min / d, scale_max / d);
            }
            for (ework, &e) in izip!(ework.iter_mut(), e.iter()) {
                *ework = T::clip(ework, scale_min / e, scale_max / e);
            }

            // Scale the problem data and update the
            // equilibration matrices
            scale_data(P, A, q, b, Some(dwork), ework);
            d.hadamard(dwork);
            e.hadamard(ework);

            // now use the Dwork array to hold the
            // column norms of the newly scaled P
            // so that we can compute the mean
            P.col_norms(dwork);
            let mean_col_norm_P = dwork.mean();
            let inf_norm_q = q.norm_inf();

            if mean_col_norm_P != T::zero() && inf_norm_q != T::zero() {
                let scale_cost = T::max(inf_norm_q, mean_col_norm_P);
                let ctmp = T::recip(scale_cost);
                let ctmp = T::clip(&ctmp, scale_min / equil.c, scale_max / equil.c);

                // scale the penalty terms and overall scaling
                P.scale(ctmp);
                q.scale(ctmp);
                equil.c *= ctmp;
            }
        } //end Ruiz scaling loop

        // fix scalings in cones for which elementwise
        // scaling can't be applied. Rectification should
        //either do nothing or take a convex combination of
        //scalings over a cone, so shouldn't need to check
        //bounds on the scalings here
        if cones.rectify_equilibration(ework, e) {
            // only rescale again if some cones were rectified
            scale_data(P, A, q, b, None, ework);
            e.hadamard(ework);
        }

        // Direct-x cones with `requires_uniform_x_scaling() == true`
        // (SOC, and eventually any future cone that isn't invariant
        // under per-entry diagonal scaling on `x`) need `d[J]` replaced
        // with a single scalar — the geometric mean of the per-entry
        // Ruiz scalings — to keep `x[J] ∈ K_J` invariant under the
        // equilibration `x̃ = D⁻¹ x`. Rebuild `dwork` as the rectification
        // factor (`geom_mean(d[J]) / d[J]` on the cone's indices, 1
        // elsewhere) and rescale the data through scale_data with an
        // all-ones `e` so only the D direction moves.
        let mut rectified_any = false;
        dwork.fill(T::one());
        for xcone in &data.dir_cones {
            let x_cone_entry = crate::solver::core::cones::make_x_cone::<T>(xcone);
            if !x_cone_entry.requires_uniform_x_scaling() {
                continue;
            }
            let indices = xcone.indices();
            if indices.is_empty() {
                continue;
            }
            // geometric mean of d[indices], computed via log to avoid
            // overflow/underflow on products.
            let ln_sum: T = indices
                .iter()
                .map(|&i| d[i].logsafe())
                .fold(T::zero(), |a, b| a + b);
            let geom_mean = (ln_sum / T::from(indices.len()).unwrap()).exp();
            for &i in indices {
                dwork[i] = geom_mean / d[i];
            }
            rectified_any = true;
        }
        if rectified_any {
            ework.fill(T::one());
            scale_data(P, A, q, b, Some(dwork), ework);
            d.hadamard(dwork);
        }

        // update the inverse scaling data
        equil.dinv.scalarop_from(T::recip, d);
        equil.einv.scalarop_from(T::recip, e);
    }
}

// ---------------
// utilities
// ---------------

fn kkt_col_norms<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    norm_LHS: &mut [T],
    norm_RHS: &mut [T],
) {
    P.col_norms_sym(norm_LHS); // P can be triu
    A.col_norms_no_reset(norm_LHS); // incrementally from P norms
    A.row_norms(norm_RHS); // same as column norms of A'
}

fn scale_data<T: FloatT>(
    P: &mut CscMatrix<T>,
    A: &mut CscMatrix<T>,
    q: &mut [T],
    b: &mut [T],
    d: Option<&[T]>,
    e: &[T],
) {
    match d {
        Some(d) => {
            P.lrscale(d, d); // P[:,:] = Ds*P*Ds
            A.lrscale(e, d); // A[:,:] = Es*A*Ds
            q.hadamard(d);
        }
        None => {
            A.lscale(e); // A[:,:] = Es*A
        }
    }
    b.hadamard(e);
}

#[cfg(feature = "sdp")]
fn try_chordal_info<T>(
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    settings: &DefaultSettings<T>,
) -> Option<ChordalInfo<T>>
where
    T: FloatT,
{
    if !settings.ipm.chordal_decomposition_enable {
        return None;
    }

    // nothing to do if there are no PSD cones or they are all small
    if !cones
        .iter()
        .any(|c| matches!(c, SupportedConeT::PSDTriangleConeT(dim) if *dim > 3))
    {
        return None;
    }

    let chordal_info = ChordalInfo::new(A, b, cones, settings);

    // no decomposition possible
    if !chordal_info.is_decomposed() {
        return None;
    }

    Some(chordal_info)
}

fn try_presolver<T>(
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    settings: &DefaultSettings<T>,
) -> Option<Presolver<T>>
where
    T: FloatT,
{
    if !settings.ipm.presolve_enable {
        return None;
    }

    let presolver = Presolver::new(A, b, cones, settings);

    if !presolver.is_reduced() {
        return None;
    }

    Some(presolver)
}

// -- utility function that tries to unwrap and slice a vector, or return
// an alternative.   Necessary since the Options for q and b are &Vec, but
// the user supplied data is a slice &[T]
#[cfg(feature = "sdp")]
pub(crate) fn unwrap_and_slice_or_else<'a, T, F>(opt: &'a Option<Vec<T>>, f: F) -> &'a [T]
where
    F: FnOnce() -> &'a [T],
    T: FloatT,
{
    if opt.is_some() {
        opt.as_ref().unwrap().as_slice()
    } else {
        f()
    }
}

#[cfg(test)]
mod xcone_equilibration_tests {
    use super::*;

    /// After equilibration with a direct-x SOC cone, all `d[J]` entries
    /// on the SOC's index set must be equal (uniform scaling), otherwise
    /// the constraint `x[J] ∈ SOC` is not preserved by `x̃ = D⁻¹ x`.
    #[test]
    fn direct_x_soc_gets_uniform_d() {
        // Toy problem whose Ruiz scaling would naturally produce
        // non-uniform `d` on indices 0..3: use a diagonal P with very
        // different column norms on those indices.
        let P = CscMatrix::<f64>::from(&[
            [100.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]);
        let q = vec![1.0, 1.0, 1.0, 1.0];
        // One equality constraint; irrelevant here.
        let A = CscMatrix::<f64>::from(&[[1.0, 1.0, 1.0, 1.0]]);
        let b = vec![1.0];
        let cones = vec![SupportedConeT::<f64>::ZeroConeT(1)];
        let dir_cones = vec![SupportedXConeT::SecondOrderXConeT(vec![0, 1, 2])];

        let settings = DefaultSettings::default();
        let mut data =
            DefaultProblemData::<f64>::new_with_xcones(&P, &q, &A, &b, &cones, &dir_cones, &settings);
        let composite = CompositeCone::<f64>::new(&data.cones);
        data.equilibrate(&composite, &settings);

        let d = &data.equilibration.d;
        // d[0..3] should be uniform to near machine precision.
        let d0 = d[0];
        for i in 1..3 {
            assert!(
                (d[i] - d0).abs() / d0 < 1e-12,
                "d[{}]={} vs d[0]={} (non-uniform on SOC direct-x indices)",
                i,
                d[i],
                d0
            );
        }
        // d[3] is NOT in the SOC, so it can differ.
    }

    /// Nonneg direct-x does NOT require uniform scaling; per-entry Ruiz
    /// should still run freely on the cone's indices.
    #[test]
    fn direct_x_nonneg_keeps_nonuniform_d() {
        let P = CscMatrix::<f64>::from(&[[100.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.01]]);
        let q = vec![1.0, 1.0, 1.0];
        let A = CscMatrix::<f64>::from(&[[1.0, 1.0, 1.0]]);
        let b = vec![1.0];
        let cones = vec![SupportedConeT::<f64>::ZeroConeT(1)];
        let dir_cones = vec![SupportedXConeT::NonnegativeXConeT(vec![0, 1, 2])];

        let settings = DefaultSettings::default();
        let mut data =
            DefaultProblemData::<f64>::new_with_xcones(&P, &q, &A, &b, &cones, &dir_cones, &settings);
        let composite = CompositeCone::<f64>::new(&data.cones);
        data.equilibrate(&composite, &settings);

        let d = &data.equilibration.d;
        // At least one pair must differ — Ruiz must actually run on the
        // highly anisotropic P above.
        let max_ratio = d
            .iter()
            .enumerate()
            .map(|(i, &di)| di / d[(i + 1) % d.len()])
            .map(|r| if r > 1.0 { r } else { 1.0 / r })
            .fold(1.0_f64, f64::max);
        assert!(
            max_ratio > 1.1,
            "nonneg direct-x unexpectedly produced uniform d: {:?}",
            d
        );
    }
}
