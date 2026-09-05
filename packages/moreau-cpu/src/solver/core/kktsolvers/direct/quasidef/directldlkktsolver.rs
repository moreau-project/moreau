#![allow(non_snake_case)]

use super::ldlsolvers::config::LDLConfiguration;
use super::*;
use crate::solver::core::kktsolvers::{HasLinearSolverInfo, KKTSolver, LinearSolverInfo};
use crate::solver::core::{cones::*, CoreSettings};
use crate::utils::debug::debug_block;
use std::iter::zip;

// -------------------------------------
// KKTSolver using direct LDL factorisation
// -------------------------------------

// We require Send/Sync here to allow pyo3 builds to share
// solver objects between threads.

pub(crate) type BoxedDirectLDLSolver<T> = Box<dyn DirectLDLSolver<T> + Send + Sync>;

pub struct DirectLDLKKTSolver<T> {
    // problem dimensions
    m: usize,
    n: usize,
    p: usize,

    // Left and right hand sides for solves
    x: Vec<T>,
    b: Vec<T>,

    // internal workspace for IR scheme
    // and static offsetting of KKT
    work1: Vec<T>,
    work2: Vec<T>,

    // KKT mapping from problem data to KKT
    map: LDLDataMap,

    // the expected signs of D in KKT = LDL^T
    dsigns: Vec<i8>,

    // a vector for storing the entries of Hs blocks
    // on the KKT matrix block diagonal
    Hsblocks: Vec<T>,

    // Baseline values of P at each direct-x Hx-block position. Non-zero only
    // where the Hx footprint overlaps the user P pattern. Refreshed by
    // `update_P`; consumed by `refresh_hx_blocks` which writes
    // `KKT.nzval[Hxblocks[i]] = px_baseline[i] + hx_values[i]`.
    px_baseline: Vec<T>,

    // Scratch for the stacked direct-x Hs buffer. Populated by
    // `CompositeXCone::get_Hs` inside `refresh_hx_blocks`.
    hx_values: Vec<T>,

    // original diagonal of KKT (without regularization)
    diag_P: Vec<T>,

    // unpermuted KKT matrix
    KKT: CscMatrix<T>,

    // triangular storage shape for KKT
    KKTuplo: MatrixTriangle,

    // the direct linear LDL solver
    ldlsolver: BoxedDirectLDLSolver<T>,

    // the diagonal regularizer currently applied
    diagonal_regularizer: T,

    // debug iteration counter
    debug_iter: usize,
}

impl<T> DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    pub fn new_with_perm_and_xcones(
        P: &CscMatrix<T>,
        A: &CscMatrix<T>,
        cones: &CompositeCone<T>,
        dir_cones: &CompositeXCone<T>,
        m: usize,
        n: usize,
        settings: &CoreSettings<T>,
        perm: Option<Vec<usize>>,
    ) -> Self {
        // get a constructor for the LDL solver we should use,
        // and also the matrix shape it requires
        let (kktshape, ldl_ctor) = T::get_ldlsolver_config(settings);

        //construct a KKT matrix of the right shape
        let (KKT, map) = assemble_kkt_matrix_with_xcones(P, A, cones, dir_cones, kktshape);

        // Total expansion dimension = slack (SOC/genpow) + direct-x
        // sparse (SOC). Direct-x sparse cols live at the end of the KKT.
        let p_slack = map.sparse_maps.pdim();
        let p_x: usize = map
            .x_sparse_maps
            .iter()
            .filter_map(|m| m.as_ref())
            .map(|m| m.pdim())
            .sum();
        let p = p_slack + p_x;

        // LHS/RHS/work for iterative refinement
        let x = vec![T::zero(); n + m + p];
        let b = vec![T::zero(); n + m + p];
        let work1 = vec![T::zero(); n + m + p];
        let work2 = vec![T::zero(); n + m + p];

        // the expected signs of D in LDL
        let mut dsigns = vec![1_i8; n + m + p];
        _fill_signs(&mut dsigns, m, n, &map);

        // updates to the diagonal of KKT will be
        // assigned here before updating matrix entries
        let mut Hsblocks = allocate_kkt_Hsblocks::<T, T>(cones);
        cones.get_Hs(&mut Hsblocks);

        // Direct-x baseline and scratch. `px_baseline[i]` holds the user-P
        // value at the i-th Hx block position when the footprint overlaps P,
        // else zero. Together with `hx_values` (get_Hs output) it reconstructs
        // `KKT[(r,c)] = P[(r,c)] + H_J[(r,c)]` each solver update.
        let hx_len = dir_cones.hx_block_len();
        let mut px_baseline = vec![T::zero(); hx_len];
        for (i, entry) in map.Hx_to_P.iter().enumerate() {
            if let Some(k) = entry {
                px_baseline[i] = P.nzval[*k];
            }
        }
        let hx_values = vec![T::zero(); hx_len];

        let diagonal_regularizer = T::zero();

        // Store original diagonal of KKT
        let mut diag_P = vec![T::zero(); n + m + p];
        for (d, idx) in zip(&mut diag_P, &map.diag_full) {
            *d = KKT.nzval[*idx];
        }

        // AMD preserves spectral conditioning (symmetric permutation),
        // but the *LDL pivot trajectory* depends on elimination order
        // and can have intermediate `min|D|` much smaller than the
        // matrix's true min eigenvalue — bad for FP-precision of
        // intermediate Schur corrections. For GenPow's sparse rank-6
        // PD expansion, AMD's choice gives `min|D| = 1e-8` at late
        // iters (vs `1.0` for the equivalent dense block) and
        // catastrophic Newton-step error. Forcing expansion variables
        // to be eliminated first restores `min|D| ≈ 1.0`.
        //
        // Per-cone opt-in via `wants_expansion_first()` — SOC's
        // bounded rank-2 expansion is well-behaved under AMD and is
        // ~10–20× slower at large dim under exp-first. When mixed
        // (GenPow + SOC), only the cones that opted in get their
        // expansion vars pulled to the front; SOC's stay where AMD
        // placed them.
        //
        // Set `KKT_PERM_NATURAL=1` to fall back to plain AMD for
        // diagnosis.
        // Direct-x GenPow rank-9 expansion has the same AMD-conditioning
        // problem as slack's: the 6 PD-axis eigenvalues blow up at near-
        // boundary iterates, and AMD interleaving the axis rows with the
        // (1,1) cone block gives `min|D| ≈ 1e-8` mid-LDL. Pull them to the
        // front, same as slack. SOC rank-2 is well-behaved under AMD and
        // is left alone.
        let any_x_genpow = map
            .x_sparse_maps
            .iter()
            .any(|m| matches!(m, Some(DirectXSparseMap::GenPow(_))));
        let do_reorder = (map.sparse_maps.wants_expansion_first() || any_x_genpow)
            && p > 0
            && std::env::var("KKT_PERM_NATURAL").is_err();
        let perm = if perm.is_some() || !do_reorder {
            perm
        } else {
            // Mark which expansion rows want first elimination.
            let mut wants_first = vec![false; n + m + p];
            let mut p_off = n + m;
            for sm in map.sparse_maps.iter() {
                let pd = sm.pdim();
                if sm.wants_expansion_first() {
                    for k in p_off..p_off + pd {
                        wants_first[k] = true;
                    }
                }
                p_off += pd;
            }
            for opt_xm in map.x_sparse_maps.iter() {
                let Some(xm) = opt_xm else {
                    continue;
                };
                let pd = xm.pdim();
                if let DirectXSparseMap::GenPow(_) = xm {
                    for k in p_off..p_off + pd {
                        wants_first[k] = true;
                    }
                }
                p_off += pd;
            }
            // AMD on the full augmented matrix.
            let (amd_perm, _, _) = crate::qdldl::get_amd_ordering(&KKT, 1.5);
            // Pull wants-first indices to the front (preserving AMD-relative
            // order among them); append the rest in AMD order.
            let mut new_perm = Vec::with_capacity(n + m + p);
            for &orig in &amd_perm {
                if wants_first[orig] {
                    new_perm.push(orig);
                }
            }
            for &orig in &amd_perm {
                if !wants_first[orig] {
                    new_perm.push(orig);
                }
            }
            Some(new_perm)
        };

        // now make the LDL linear solver engine
        let ldlsolver = ldl_ctor(&KKT, &dsigns, settings, perm);

        Self {
            m,
            n,
            p,
            x,
            b,
            work1,
            work2,
            map,
            dsigns,
            Hsblocks,
            px_baseline,
            hx_values,
            diag_P,
            KKT,
            KKTuplo: kktshape,
            ldlsolver,
            diagonal_regularizer,
            debug_iter: 0,
        }
    }

    /// Refresh the direct-x Hs contributions in the (1,1) KKT block.
    ///
    /// Each Hxblocks entry is written as `px_baseline[i] + hs[i]` where
    /// `hs` is populated by `dir_cones.get_Hs`. At overlap slots with the
    /// user P pattern, this restores the structural invariant
    /// `KKT[(r,c)] = P[(r,c)] + H_J[(r,c)]` after a fresh `update_P`. At
    /// non-overlap slots `px_baseline` is zero.
    ///
    /// Callers must invoke `update_P` (or equivalent) before this whenever
    /// P has changed — `px_baseline` is a cache of user P values at Hx
    /// positions.
    pub(crate) fn refresh_hx_blocks(&mut self, dir_cones: &CompositeXCone<T>) {
        if !self.map.Hxblocks.is_empty() {
            assert_eq!(self.hx_values.len(), self.map.Hxblocks.len());
            dir_cones.get_Hs(&mut self.hx_values);
            for i in 0..self.hx_values.len() {
                self.hx_values[i] += self.px_baseline[i];
            }
            _update_values(
                &mut self.ldlsolver,
                &mut self.KKT,
                &self.map.Hxblocks,
                &self.hx_values,
            );
        }

        // Refresh direct-x sparse expansion columns. Direct-x adds `+Hs`
        // to the (1,1) block, so the signs are opposite of slack's `-Hs`
        // convention. Values are fetched in sorted-row order via the map's
        // `cone_pos_for_sorted_*` permutations.
        if self.map.x_sparse_maps.iter().any(|m| m.is_some()) {
            for (entry, opt_map) in dir_cones.iter().zip(self.map.x_sparse_maps.iter()) {
                let Some(xm) = opt_map else {
                    continue;
                };
                match (xm, &entry.cone) {
                    (DirectXSparseMap::SOC(xm_soc), SupportedCone::SecondOrderCone(soc)) => {
                        // SOC: u col, v col, diag [+η², -η²].
                        let sparse_data = soc
                            .sparse_data
                            .as_ref()
                            .expect("sparse-expandable SOC must have sparse_data");
                        let η2 = soc.η * soc.η;
                        let k = entry.indices.len();
                        let mut u_scaled = vec![T::zero(); k];
                        let mut v_scaled = vec![T::zero(); k];
                        for p in 0..k {
                            let cone_pos = xm_soc.cone_pos_for_sorted[p];
                            u_scaled[p] = η2 * sparse_data.u[cone_pos];
                            v_scaled[p] = η2 * sparse_data.v[cone_pos];
                        }
                        _update_values(&mut self.ldlsolver, &mut self.KKT, &xm_soc.u, &u_scaled);
                        _update_values(&mut self.ldlsolver, &mut self.KKT, &xm_soc.v, &v_scaled);
                        _update_values(&mut self.ldlsolver, &mut self.KKT, &xm_soc.D, &[η2, -η2]);
                    }
                    (DirectXSparseMap::GenPow(xm_gp), SupportedCone::GenPowerCone(gpc)) => {
                        // GenPow rank-9: rank-3 base (q, r, p) + rank-6 PD
                        // axes. Adaptive per-axis equilibration mirrors
                        // slack's `csc_update_sparsecone`: when an axis's
                        // weight `w` exceeds the AMD-conditioning threshold
                        // (1e12), rescale to unit-norm off-diag with
                        // sentinel `±1/w`. Direct-x sentinels are slack
                        // sentinels negated (direct-x adds `+Hs` to the
                        // (1,1) block; slack adds `-Hs` to (2,2)).
                        let dim1 = gpc.dim1();
                        let dim2 = gpc.dim2();
                        let data = &gpc.data;
                        let sqrtμ = data.μ.sqrt();
                        let one = T::one();
                        let safe_inv_sqrt = |x: T| -> T {
                            if x > T::epsilon() {
                                one / x.sqrt()
                            } else {
                                T::zero()
                            }
                        };
                        let threshold = T::from_f64(1e12).unwrap();

                        // Per-axis weight `w = μ·||axis||²` (rank-3 base).
                        let q_norm_sq = data.q.iter().fold(T::zero(), |a, &x| a + x * x);
                        let r_norm_sq = data.r.iter().fold(T::zero(), |a, &x| a + x * x);
                        let p_norm_sq = data.p.iter().fold(T::zero(), |a, &x| a + x * x);
                        let q_w = data.μ * q_norm_sq;
                        let r_w = data.μ * r_norm_sq;
                        let p_w = data.μ * p_norm_sq;

                        // Off-diag scale: `-sqrtμ` (low-w) or `-1/||axis||`
                        // (high-w). Sign is symmetric — only `c²` enters
                        // Schur — so this matches slack exactly.
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

                        let mut q_vals = vec![T::zero(); dim1];
                        for i in 0..dim1 {
                            let cone_pos = xm_gp.cone_pos_for_sorted_dim1[i];
                            q_vals[i] = q_scale * data.q[cone_pos];
                        }
                        let mut r_vals = vec![T::zero(); dim2];
                        for i in 0..dim2 {
                            let cone_pos = xm_gp.cone_pos_for_sorted_dim2[i] - dim1;
                            r_vals[i] = r_scale * data.r[cone_pos];
                        }
                        let mut p_vals = vec![T::zero(); dim1 + dim2];
                        for i in 0..(dim1 + dim2) {
                            let cone_pos = xm_gp.cone_pos_for_sorted_all[i];
                            p_vals[i] = p_scale * data.p[cone_pos];
                        }
                        _update_values(&mut self.ldlsolver, &mut self.KKT, &xm_gp.q, &q_vals);
                        _update_values(&mut self.ldlsolver, &mut self.KKT, &xm_gp.r, &r_vals);
                        _update_values(&mut self.ldlsolver, &mut self.KKT, &xm_gp.p_v, &p_vals);

                        // Sentinels: direct-x flip of slack's. Slack uses
                        // q_sent=-1, r_sent=-1, p_sent=+1 (low-w); the
                        // direct-x flip is +1, +1, -1.
                        let q_sent = if q_w > threshold { one / q_w } else { one };
                        let r_sent = if r_w > threshold { one / r_w } else { one };
                        let p_sent = if p_w > threshold { -one / p_w } else { -one };

                        // ---------- Rank-6 PD axes ----------
                        // Direct-x adds `+Σ pd_signs[k]·pd_coefs[k]·v_k v_k'`
                        // to (1,1). Schur `-(c²/d) v v' = pd_signs·pd_coefs·v v'`
                        // with `c² = pd_coefs[k]` ⇒ sentinel `d = -pd_signs[k]`
                        // (low-w) or `-pd_signs[k]/pd_w[k]` (high-w).
                        let mut pd_norm_sq = [T::zero(); 6];
                        let mut pd_w = [T::zero(); 6];
                        let signs = &data.pd_signs;
                        let mut pd_sents = [T::zero(); 6];
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
                            // Permute axis values into KKT (sorted) row order.
                            let mut axis_vals = vec![T::zero(); dim1 + dim2];
                            for i in 0..(dim1 + dim2) {
                                let cone_pos = xm_gp.cone_pos_for_sorted_all[i];
                                axis_vals[i] = scale * data.pd_axes[k][cone_pos];
                            }
                            _update_values(
                                &mut self.ldlsolver,
                                &mut self.KKT,
                                &xm_gp.pd_axes[k],
                                &axis_vals,
                            );

                            // Direct-x sentinel = -(slack sentinel) = -sign[k]
                            // (low-w) or -sign[k]/w[k] (high-w).
                            let sign = T::from_i8(signs[k]).unwrap();
                            pd_sents[k] = if pd_w[k] > threshold {
                                -sign / pd_w[k]
                            } else {
                                -sign
                            };
                        }

                        _update_values(
                            &mut self.ldlsolver,
                            &mut self.KKT,
                            &xm_gp.D,
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
                    _ => panic!("direct-x sparse expansion: cone/map kind mismatch"),
                }
            }
        }
    }
}

impl<T> HasLinearSolverInfo for DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    fn linear_solver_info(&self) -> LinearSolverInfo {
        self.ldlsolver.linear_solver_info()
    }
}

impl<T> KKTSolver<T> for DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    fn update(
        &mut self,
        cones: &mut CompositeCone<T>,
        dir_cones: &mut CompositeXCone<T>,
        settings: &CoreSettings<T>,
    ) -> bool {
        // Set the elements the W^tW blocks in the KKT matrix.
        cones.get_Hs(&mut self.Hsblocks);

        {
            let (values, index) = (&mut self.Hsblocks, &self.map.Hsblocks);
            // change signs to get -W^TW
            values.negate();
            _update_values(&mut self.ldlsolver, &mut self.KKT, index, values);
        }

        {
            let map = &self.map;
            let mut sparse_map_iter = map.sparse_maps.iter();
            let ldl = &mut self.ldlsolver;
            let KKT = &mut self.KKT;

            for cone in cones.iter() {
                if cone.is_sparse_expandable() {
                    let sc = cone.to_sparse_expansion().unwrap();
                    let thismap = sparse_map_iter.next().unwrap();
                    sc.csc_update_sparsecone(thismap, ldl, KKT, _update_values, _scale_values);
                }
            }
        }

        // Refresh expansion-row sign expectations from the cones (sparse
        // PD scaling has dynamic-sign slots that flip per iteration) and
        // push the new sign vector into the LDL backend before refactor
        // — sign-aware regularisation needs the live values.
        let mut p_off = self.n + self.m;
        let mut sparse_map_iter = self.map.sparse_maps.iter();
        for cone in cones.iter() {
            if cone.is_sparse_expandable() {
                let sc = cone.to_sparse_expansion().unwrap();
                let thismap = sparse_map_iter.next().unwrap();
                let pdim = thismap.pdim();
                sc.fill_dsigns(thismap, &mut self.dsigns[p_off..p_off + pdim]);
                p_off += pdim;
            }
        }
        // Direct-x sparse expansion dynamic signs. SOC dsigns are static
        // (`[+1, -1]`); GenPow PD slots flip with `data.pd_signs`. The
        // direct-x sentinel for axis k is `-pd_signs[k]` (slack's sentinel
        // negated — direct-x adds `+Hs` to (1,1), slack adds `-Hs` to (2,2)),
        // so the LDL pivot expected sign is also `-pd_signs[k]`.
        for (entry, opt_map) in dir_cones.iter().zip(self.map.x_sparse_maps.iter()) {
            let Some(xm) = opt_map else {
                continue;
            };
            let xpdim = xm.pdim();
            match (xm, &entry.cone) {
                (DirectXSparseMap::SOC(_), _) => {
                    self.dsigns[p_off..p_off + xpdim].copy_from_slice(xm.dsigns());
                }
                (DirectXSparseMap::GenPow(_), SupportedCone::GenPowerCone(gpc)) => {
                    let signs = &gpc.data.pd_signs;
                    self.dsigns[p_off] = 1; // q
                    self.dsigns[p_off + 1] = 1; // r
                    self.dsigns[p_off + 2] = -1; // p
                    for k in 0..6 {
                        self.dsigns[p_off + 3 + k] = -signs[k];
                    }
                }
                _ => panic!("direct-x sparse expansion: cone/map kind mismatch"),
            }
            p_off += xpdim;
        }
        self.ldlsolver.update_dsigns(&self.dsigns);

        // Direct-x cones additively contribute H_J to (1,1). Skipped cheaply
        // when `dir_cones` is empty (fast path matches the slack-only solver).
        self.refresh_hx_blocks(dir_cones);

        let success = self.regularize_and_refactor(settings);

        self.debug_iter += 1;
        debug_block! {
            eprintln!("\n=== CPU KKT iter {} ===", self.debug_iter);
            self.debug_dump_kkt_dense();
        }

        success
    }

    fn setrhs(&mut self, rhsx: &[T], rhsz: &[T]) {
        let (m, n, p) = (self.m, self.n, self.p);

        self.b[0..n].copy_from(rhsx);
        self.b[n..(n + m)].copy_from(rhsz);
        self.b[n + m..(n + m + p)].fill(T::zero());
    }

    fn solve(
        &mut self,
        lhsx: Option<&mut [T]>,
        lhsz: Option<&mut [T]>,
        settings: &CoreSettings<T>,
    ) -> bool {
        debug_block! {
            let dim = self.n + self.m + self.p;
            eprintln!("CPU KKT RHS iter {} ({}):", self.debug_iter, dim);
            for i in 0..dim {
                let block = if i < self.n { "P" } else if i < self.n + self.m { "H" } else { "E" };
                eprintln!("  [{:3}] ({}) {:+.16e}", i, block, self.b[i].to_f64().unwrap());
            }
        }

        self.ldlsolver.solve(&self.KKT, &mut self.x, &mut self.b);

        debug_block! {
            let dim = self.n + self.m + self.p;
            eprintln!("CPU KKT SOL iter {} ({}):", self.debug_iter, dim);
            for i in 0..dim {
                let block = if i < self.n { "P" } else if i < self.n + self.m { "H" } else { "E" };
                eprintln!("  [{:3}] ({}) {:+.16e}", i, block, self.x[i].to_f64().unwrap());
            }

            // Compute solve residual: r = KKT*x - b
            let kkt_sym = self.KKT.sym(self.KKTuplo);
            let mut kkt_x = vec![T::zero(); dim];
            kkt_sym.symv(&mut kkt_x, &self.x, T::one(), T::zero());
            let mut max_res = 0.0_f64;
            for i in 0..dim {
                let r = (kkt_x[i] - self.b[i]).to_f64().unwrap().abs();
                max_res = max_res.max(r);
            }
            let norm_b: f64 = self.b.iter().map(|v| v.to_f64().unwrap().abs()).fold(0.0, f64::max);
            eprintln!("CPU KKT solve residual: abs={:.4e}, rel={:.4e}", max_res, max_res / norm_b.max(1e-20));
        }

        let is_success = {
            if settings.ipm.iterative_refinement_enable {
                self.iterative_refinement(settings)
            } else {
                self.x.is_finite()
            }
        };

        if is_success {
            self.getlhs(lhsx, lhsz);
        }

        is_success
    }

    fn update_P(&mut self, P: &CscMatrix<T>) {
        _update_values(&mut self.ldlsolver, &mut self.KKT, &self.map.P, &P.nzval);

        // Refresh direct-x baselines at overlap slots so the next
        // `refresh_hx_blocks` reconstructs KKT = P + H_J correctly.
        // Non-overlap entries stay at zero (they were zero at construction
        // and P overlap structure is fixed for the lifetime of the solver).
        for (i, entry) in self.map.Hx_to_P.iter().enumerate() {
            if let Some(k) = entry {
                self.px_baseline[i] = P.nzval[*k];
            }
        }

        // Refresh cached diagonal of the top-left (P) block.
        self.diag_P[0..self.n].fill(T::zero());
        let n = self.n;
        for j in 0..n {
            for idx in P.colptr[j]..P.colptr[j + 1] {
                let i = P.rowval[idx];
                if i == j {
                    self.diag_P[j] = P.nzval[idx];
                    break;
                }
            }
        }
    }

    fn update_A(&mut self, A: &CscMatrix<T>) {
        _update_values(&mut self.ldlsolver, &mut self.KKT, &self.map.A, &A.nzval);
    }

    fn get_amd_perm(&self) -> Option<Vec<usize>> {
        self.ldlsolver.get_perm()
    }
}

impl<T> DirectLDLKKTSolver<T>
where
    T: FloatT,
{
    // extra helper functions, not required for KKTSolver trait
    fn getlhs(&self, lhsx: Option<&mut [T]>, lhsz: Option<&mut [T]>) {
        let x = &self.x;
        let (m, n) = (self.m, self.n);

        if let Some(v) = lhsx {
            v.copy_from(&x[0..n]);
        }
        if let Some(v) = lhsz {
            v.copy_from(&x[n..(n + m)]);
        }
    }

    fn regularize_and_refactor(&mut self, settings: &CoreSettings<T>) -> bool {
        let map = &self.map;
        let KKT = &mut self.KKT;
        let dsigns = &self.dsigns;
        let diag_kkt = &mut self.work1;
        let diag_shifted = &mut self.work2;

        if settings.ipm.static_regularization_enable {
            // hold a copy of the true KKT diagonal
            for (d, idx) in zip(&mut *diag_kkt, &map.diag_full) {
                *d = KKT.nzval[*idx];
            }

            let eps = _compute_regularizer(diag_kkt, settings);

            // compute an offset version, accounting for signs
            diag_shifted.copy_from(diag_kkt);

            zip(&mut *diag_shifted, dsigns).for_each(|(shift, &sign)| {
                if sign == 1 {
                    *shift += eps;
                } else {
                    *shift -= eps;
                }
            });

            // overwrite the diagonal of KKT and within the ldlsolver
            _update_values(&mut self.ldlsolver, KKT, &map.diag_full, diag_shifted);

            // remember the value we used.  Not needed,
            // but possibly useful for debugging
            self.diagonal_regularizer = eps;
        }

        //refactor with new data
        let is_success = self.ldlsolver.refactor(KKT);

        if settings.ipm.static_regularization_enable {
            // put our internal copy of the KKT matrix back the way
            // it was. Not necessary to fix the ldlsolver copy because
            // this is only needed for our post-factorization IR scheme
            _update_values_KKT(KKT, &map.diag_full, diag_kkt);
        }

        is_success
    }

    fn debug_dump_kkt_dense(&self) {
        let dim = self.n + self.m + self.p;
        let kkt = &self.KKT;
        let _is_triu = matches!(self.KKTuplo, MatrixTriangle::Triu);

        // Reconstruct full symmetric dense matrix from CSC upper/lower triangle
        let mut dense = vec![0.0_f64; dim * dim];
        for j in 0..dim {
            for idx in kkt.colptr[j]..kkt.colptr[j + 1] {
                let i = kkt.rowval[idx];
                let v: f64 = kkt.nzval[idx].to_f64().unwrap();
                dense[i * dim + j] = v;
                dense[j * dim + i] = v; // symmetric
            }
        }

        eprintln!(
            "CPU KKT dense ({0}x{0}), n={1}, m={2}, p={3}:",
            dim, self.n, self.m, self.p
        );
        for i in 0..dim {
            eprint!("  [");
            for j in 0..dim {
                if j > 0 {
                    eprint!(", ");
                }
                eprint!("{:+.16e}", dense[i * dim + j]);
            }
            eprintln!("]");
        }
        eprintln!("CPU KKT diagonal:");
        for i in 0..dim {
            let block = if i < self.n {
                "P"
            } else if i < self.n + self.m {
                "H"
            } else {
                "E"
            };
            eprintln!("  [{:3}] ({}) {:+.16e}", i, block, dense[i * dim + i]);
        }
    }

    fn iterative_refinement(&mut self, settings: &CoreSettings<T>) -> bool {
        let (x, b) = (&mut self.x, &self.b);
        let (e, dx) = (&mut self.work1, &mut self.work2);

        // iterative refinement params
        let reltol = settings.ipm.iterative_refinement_reltol;
        let abstol = settings.ipm.iterative_refinement_abstol;
        let maxiter = settings.ipm.iterative_refinement_max_iter;
        let stopratio = settings.ipm.iterative_refinement_stop_ratio;

        let KKT = &self.KKT;
        let KKTsym = KKT.sym(self.KKTuplo);

        let normb = b.norm_inf();

        //compute the initial error
        let mut norme = _get_refine_error(e, b, &KKTsym, x);

        if !norme.is_finite() {
            return false;
        }

        for _ in 0..maxiter {
            if norme <= (abstol + reltol * normb) {
                //within tolerance.  Exit
                break;
            }

            let lastnorme = norme;

            //make a refinement
            self.ldlsolver.solve(KKT, dx, e);

            //prospective solution is x + dx.  Use dx space to
            // hold it for a check before applying to x
            dx.axpby(T::one(), x, T::one());

            norme = _get_refine_error(e, b, &KKTsym, dx);

            if !norme.is_finite() {
                return false;
            }

            let improved_ratio = lastnorme / norme;
            if improved_ratio < stopratio {
                //insufficient improvement.  Exit
                if improved_ratio > T::one() {
                    std::mem::swap(x, dx);
                }
                break;
            }
            std::mem::swap(x, dx);
        }
        //NB: "success" means only that we had a finite valued result
        true
    }
}

fn _compute_regularizer<T: FloatT>(diag_kkt: &[T], settings: &CoreSettings<T>) -> T {
    let maxdiag = diag_kkt.norm_inf();
    settings.ipm.static_regularization_constant
        + settings.ipm.static_regularization_proportional * maxdiag
}

//  computes e = b - Kξ, overwriting the first argument
//  and returning its norm

fn _get_refine_error<T: FloatT>(
    e: &mut [T],
    b: &[T],
    KKTsym: &Symmetric<CscMatrix<T>>,
    ξ: &mut [T],
) -> T {
    // Note that K is only triu data, so need to
    // be careful when computing the residual here

    e.copy_from(b);
    KKTsym.symv(e, ξ, -T::one(), T::one()); //#  e = b - Kξ

    e.norm_inf()
}

// update entries of the KKT matrix using the given index into its CSC representation.
// applied to both the unpermuted matrix of the kktsolver and also to the ldlsolver
fn _update_values<T: FloatT>(
    ldlsolver: &mut BoxedDirectLDLSolver<T>,
    KKT: &mut CscMatrix<T>,
    index: &[usize],
    values: &[T],
) {
    //Update values in the KKT matrix K
    _update_values_KKT(KKT, index, values);

    // give the LDL subsolver an opportunity to update the same
    // values if needed.   This latter is useful for QDLDL since
    // it stores its own permuted copy internally
    ldlsolver.update_values(index, values);
}

fn _update_values_KKT<T: FloatT>(KKT: &mut CscMatrix<T>, index: &[usize], values: &[T]) {
    for (idx, v) in zip(index, values) {
        KKT.nzval[*idx] = *v;
    }
}

fn _scale_values<T: FloatT>(
    ldlsolver: &mut BoxedDirectLDLSolver<T>,
    KKT: &mut CscMatrix<T>,
    index: &[usize],
    scale: T,
) {
    //Update values in the KKT matrix K
    _scale_values_KKT(KKT, index, scale);

    // ...and in the LDL subsolver if needed
    ldlsolver.scale_values(index, scale);
}

//scales KKT matrix values
fn _scale_values_KKT<T: FloatT>(KKT: &mut CscMatrix<T>, index: &[usize], scale: T) {
    for idx in index.iter() {
        KKT.nzval[*idx] *= scale;
    }
}

#[cfg(test)]
mod xcone_refresh_tests {
    use super::*;
    use crate::solver::core::traits::Settings;
    use crate::solver::{core::cones::SupportedXConeT, DefaultSettings};

    fn kkt_at(kkt: &CscMatrix<f64>, row: usize, col: usize) -> f64 {
        // Symmetric dense lookup — the KKT is triu so (row, col) with
        // row <= col, (col, row) otherwise.
        let (r, c) = if row <= col { (row, col) } else { (col, row) };
        for k in kkt.colptr[c]..kkt.colptr[c + 1] {
            if kkt.rowval[k] == r {
                return kkt.nzval[k];
            }
        }
        0.0
    }

    #[test]
    fn refresh_hx_blocks_overlap_and_update_p_cycle() {
        // P has a single entry at (0,0)=3.0. Direct-x nonneg on [0, 1],
        // so slot 0 overlaps P and slot 1 is a new structural entry.
        // Confirm that after refresh_hx_blocks:
        //   KKT[0,0] = P[0,0] + Hs[0]
        //   KKT[1,1] = 0       + Hs[1]
        // And that update_P with new P followed by refresh_hx_blocks
        // tracks the new baseline.
        let P = CscMatrix::<f64>::from(&[[3., 0.], [0., 0.]]);
        let A = CscMatrix::<f64>::from(&[[1., 1.]]);
        let cones = CompositeCone::<f64>::new(&[SupportedConeT::ZeroConeT(1)]);
        let dir_cones = CompositeXCone::<f64>::new(&[SupportedXConeT::NonnegativeXConeT(vec![0, 1])]);

        let settings = DefaultSettings::<f64>::default();
        let mut ldl = DirectLDLKKTSolver::<f64>::new_with_perm_and_xcones(
            &P,
            &A,
            &cones,
            &dir_cones,
            1,
            2,
            settings.core(),
            None,
        );

        // Prime the direct-x cone scaling so `get_Hs` emits a non-zero,
        // deterministic block. NonnegativeCone's Hs is `w*w`, and
        // `set_identity_scaling` sets w = 1, so Hs[i] = 1.
        let mut dir_cones_mut = dir_cones;
        for entry in dir_cones_mut.iter_mut() {
            entry.cone.set_identity_scaling();
        }
        ldl.refresh_hx_blocks(&dir_cones_mut);
        assert_eq!(kkt_at(&ldl.KKT, 0, 0), 3.0 + 1.0);
        assert_eq!(kkt_at(&ldl.KKT, 1, 1), 0.0 + 1.0);

        // Change P[0,0] to 7.0, call update_P, then refresh — the overlap
        // slot should track the new baseline.
        let P_new = CscMatrix::<f64>::from(&[[7., 0.], [0., 0.]]);
        ldl.update_P(&P_new);
        ldl.refresh_hx_blocks(&dir_cones_mut);
        assert_eq!(kkt_at(&ldl.KKT, 0, 0), 7.0 + 1.0);
        assert_eq!(kkt_at(&ldl.KKT, 1, 1), 0.0 + 1.0);
    }
}

fn _fill_signs(signs: &mut [i8], m: usize, n: usize, map: &LDLDataMap) {
    signs.fill(1);

    //flip expected negative signs of D in LDL
    signs[n..(n + m)].iter_mut().for_each(|x| *x = -*x);

    let mut p = m + n;
    // assign D signs for slack sparse expansion cones
    for thismap in map.sparse_maps.iter() {
        let thisp = thismap.pdim();
        signs[p..(p + thisp)].copy_from_slice(thismap.Dsigns());
        p += thisp;
    }
    // assign D signs for direct-x sparse expansion cones.
    // Direct-x adds +Hs to the (1,1) block (positive side), so the sign
    // convention is the OPPOSITE of slack's. SOC: [+1, -1] (slack has
    // [-1, +1]). GenPow: [+1, +1, -1] (slack has [-1, -1, +1]).
    for xm in map.x_sparse_maps.iter().flatten() {
        let thisp = xm.pdim();
        signs[p..(p + thisp)].copy_from_slice(xm.dsigns());
        p += thisp;
    }
}
