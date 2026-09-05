#![allow(non_snake_case)]

use super::*;
use crate::algebra::*;
#[cfg(feature = "sdp")]
use crate::solver::chordal::{ChordalInfo, CompactAugmentedProblem};
use crate::solver::core::cones::Cone;
use crate::solver::core::cones::SupportedXConeT;
use crate::solver::core::traits::*;
use crate::solver::SupportedConeT;
use crate::utils::banner;
use rayon::prelude::*;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, RwLock};
use std::time::Instant;

/// A single problem in a batch - stores only values, not structure
///
/// The sparsity pattern is shared across all problems in the batch and
/// stored in the CompiledSolver. This saves memory and is more efficient.
#[derive(Clone)]
pub struct BatchProblem<T: FloatT> {
    /// Values for objective matrix P (length = nnz(P), CSR order)
    pub P_values: Vec<T>,
    /// Objective vector q (length = n)
    pub q: Vec<T>,
    /// Values for constraint matrix A (length = nnz(A), CSR order)
    pub A_values: Vec<T>,
    /// Constraint vector b (length = m)
    pub b: Vec<T>,
}

/// Upstream gradients for backward pass
pub struct UpstreamGradients<T: FloatT> {
    /// Gradient w.r.t. primal variables x (dL/dx)
    pub dx: Vec<T>,
    /// Gradient w.r.t. slack variables s (dL/ds)
    pub ds: Vec<T>,
    /// Gradient w.r.t. dual variables z (dL/dz)
    pub dz: Vec<T>,
    /// Gradient w.r.t. direct-x cone duals z_x (dL/dz_x). Empty for
    /// problems without direct-x cones; otherwise must match the flat
    /// `Solution.z_x` length.
    pub dz_x: Vec<T>,
}

/// Computed gradients from backward pass - stores only values, not structure
pub struct ComputedGradients<T: FloatT> {
    /// Gradient values w.r.t. P (length = nnz(P), CSR order)
    pub dP_values: Vec<T>,
    /// Gradient w.r.t. q (length = n)
    pub dq: Vec<T>,
    /// Gradient values w.r.t. A (length = nnz(A), CSR order)
    pub dA_values: Vec<T>,
    /// Gradient w.r.t. b (length = m)
    pub db: Vec<T>,
    #[cfg(debug_assertions)]
    pub debug_smoothing_x: Vec<T>,
    #[cfg(debug_assertions)]
    pub debug_smoothing_z: Vec<T>,
    #[cfg(debug_assertions)]
    pub debug_smoothing_s: Vec<T>,
}

/// Cached state from solve for backward pass
struct CachedState<T: FloatT> {
    x: Vec<T>,
    z: Vec<T>,
    s: Vec<T>,
    /// Direct-x dual `z_x` (length `xn`) in the solver-internal frame, kept
    /// alongside the rest of the variables snapshot for backward(). Empty
    /// when the solver has no direct-x cones.
    z_x: Vec<T>,
    d: Vec<T>,
    dinv: Vec<T>,
    e: Vec<T>,
    einv: Vec<T>,
    c: T,
    P_values: Vec<T>,
    q: Vec<T>,
    A_values: Vec<T>,
    b: Vec<T>,
    mu: T,
    /// Whether a smoothing iterate was cached (for smoothed differentiation).
    smoothing_cached: bool,
    /// Smoothing iterate in equilibrated HSDE coordinates (with τ=1).
    smoothing_x: Vec<T>,
    smoothing_z: Vec<T>,
    smoothing_s: Vec<T>,
    smoothing_tau: T,
}

impl<T: FloatT> CachedState<T> {
    /// Create an empty cached state (used when grad is disabled to avoid allocations)
    fn empty() -> Self {
        Self {
            x: Vec::new(),
            z: Vec::new(),
            s: Vec::new(),
            z_x: Vec::new(),
            d: Vec::new(),
            dinv: Vec::new(),
            e: Vec::new(),
            einv: Vec::new(),
            c: T::zero(),
            P_values: Vec::new(),
            q: Vec::new(),
            A_values: Vec::new(),
            b: Vec::new(),
            mu: T::zero(),
            smoothing_cached: false,
            smoothing_x: Vec::new(),
            smoothing_z: Vec::new(),
            smoothing_s: Vec::new(),
            smoothing_tau: T::zero(),
        }
    }

    /// Capture solver state for backward pass.
    fn from_solver(solver: &DefaultSolver<T>) -> Self {
        Self {
            x: solver.solution.x.clone(),
            z: solver.solution.z.clone(),
            s: solver.solution.s.clone(),
            z_x: solver.variables.z_x.clone(),
            d: solver.data.equilibration.d.clone(),
            dinv: solver.data.equilibration.dinv.clone(),
            e: solver.data.equilibration.e.clone(),
            einv: solver.data.equilibration.einv.clone(),
            c: solver.data.equilibration.c,
            P_values: solver.data.P.nzval.clone(),
            q: solver.data.q.clone(),
            A_values: solver.data.A.nzval.clone(),
            b: solver.data.b.clone(),
            mu: solver.info.mu,
            smoothing_cached: solver.smoothing_cached,
            smoothing_x: solver.smoothing_vars.x.clone(),
            smoothing_z: solver.smoothing_vars.z.clone(),
            smoothing_s: solver.smoothing_vars.s.clone(),
            smoothing_tau: solver.smoothing_vars.τ,
        }
    }
}

/// Parallel batch solver with two usage patterns:
///
/// **Pattern 1: Shared P, A (three-step API)**
/// When all problems share the same P and A matrices:
/// 1. `new()` - Define problem structure (symbolic factorization done here)
/// 2. `setup(P, A)` - Set shared P and A values (does equilibration once)
/// 3. `solve_batch(q_vectors, b_vectors)` - Solve with different q, b per problem
///
/// **Pattern 2: Different P, A per problem**
/// When P, A, q, b all vary per problem:
/// 1. `new()` - Define problem structure (symbolic factorization done here)
/// 2. `solve_batch_parallel(problems)` - Solve all problems (does equilibration per-problem)
///
/// CSR input format: Constructor takes (row_offsets, col_indices), methods take values.
/// Internally converts to CSC for the solver.
pub struct CompiledSolver<T: FloatT> {
    /// Number of variables (API/original dimensions)
    n: usize,
    /// Number of constraints (API/original dimensions)
    m: usize,
    /// Internal number of variables (reduced if presolve active)
    n_internal: usize,
    /// Internal number of constraints (reduced if presolve active)
    m_internal: usize,
    /// CSR row offsets for P (length n+1)
    #[allow(dead_code)]
    P_row_offsets: Vec<usize>,
    /// CSR column indices for P
    P_col_indices: Vec<usize>,
    /// CSR row offsets for A (length m+1)
    #[allow(dead_code)]
    A_row_offsets: Vec<usize>,
    /// CSR column indices for A
    A_col_indices: Vec<usize>,
    /// CSR to CSC index mapping for P
    P_csr_to_csc: Vec<usize>,
    /// CSR to CSC index mapping for A
    A_csr_to_csc: Vec<usize>,
    /// CSC pattern for P (for solver initialization)
    P_csc_pattern: CscMatrix<T>,
    /// CSC pattern for A (for solver initialization, reduced if presolve active)
    A_csc_pattern: CscMatrix<T>,
    /// Cone constraints
    cones: Vec<SupportedConeT<T>>,
    /// Direct-x cone constraints (empty = slack-only)
    dir_cones: Vec<SupportedXConeT>,
    /// Solver settings
    settings: DefaultSettings<T>,
    /// Thread pool for parallel processing
    thread_pool: rayon::ThreadPool,
    /// Pre-allocated solvers - one per problem slot for correct solution caching
    solver_pool: Vec<Mutex<DefaultSolver<T>>>,
    /// Current batch size (number of solvers in pool)
    #[allow(dead_code)]
    batch_size: usize,
    /// Number of threads for parallel execution
    #[allow(dead_code)]
    num_threads: usize,
    /// Whether equilibration has been precomputed
    equilibration_precomputed: bool,
    /// Cached solutions for backward pass
    solutions_cache: RwLock<Vec<CachedState<T>>>,
    /// Pre-computed gradient state for efficient backward pass (one per thread)
    grad_states: Vec<Mutex<super::diff::GradState<T>>>,
    /// Whether gradient computation is enabled
    grad_enabled: bool,
    /// Cached AMD permutation from first solver (reused for pool expansion)
    cached_amd_perm: Option<Vec<usize>>,
    /// Setup time from last setup() or setup_shared() call
    setup_time: f64,
    /// Original (un-equilibrated) P values per problem in CSC order
    /// Stored so we can restore before re-equilibrating on each solve
    original_P_values: Vec<Vec<T>>,
    /// Original (un-equilibrated) A values per problem in CSC order
    original_A_values: Vec<Vec<T>>,
    /// Base equilibrated P values (after d,e scaling, before cost scaling c)
    /// Used in vectors-only path to restore P before applying new cost scaling.
    /// Wrapped in RwLock for interior mutability — populated after first solve.
    base_equilibrated_P_values: RwLock<Vec<Vec<T>>>,
    /// Whether matrices (P, A) are already equilibrated in the solver pool
    /// When true, solve() only equilibrates q and b (not P and A)
    /// Set to false by setup()/setup_shared(), set to true after first solve()
    matrices_equilibrated: AtomicBool,
    /// Chordal decomposition info (Some iff PSD cones were decomposed at construction)
    #[cfg(feature = "sdp")]
    chordal_info: Option<ChordalInfo<T>>,
    /// Precomputed augmented problem structure (Some iff chordal_info is Some)
    #[cfg(feature = "sdp")]
    augmented_problem: Option<CompactAugmentedProblem<T>>,
    /// NNZ of original (pre-chordal) P in upper-triangular CSC format
    P_nnz_orig: usize,
    /// NNZ of original (pre-chordal) A in CSC format
    A_nnz_orig: usize,
    /// Row range for nonneg cone constraints: [nonneg_row_start, nonneg_row_end)
    /// Precomputed at construction time from cone layout (zero cones first, then nonneg).
    nonneg_row_start: usize,
    nonneg_row_end: usize,
    /// For each nonneg row, the CSC indices in A that correspond to that row's entries.
    nonneg_row_csc_indices: Vec<Vec<usize>>,
}

impl<T: FloatT> CompiledSolver<T> {
    /// Create a new batch solver with CSR structure.
    ///
    /// Two-step API: structure at construction, values at solve().
    ///
    /// # Arguments
    ///
    /// * `n` - Number of variables
    /// * `m` - Number of constraints
    /// * `P_row_offsets` - CSR row pointers for P matrix (length n+1)
    /// * `P_col_indices` - CSR column indices for P matrix
    /// * `A_row_offsets` - CSR row pointers for A matrix (length m+1)
    /// * `A_col_indices` - CSR column indices for A matrix
    /// * `cones` - Cone constraints
    /// * `settings` - Solver settings (will be modified for batch mode)
    /// * `num_threads` - Number of threads for parallel processing
    /// * `enable_grad` - If true, pre-compute gradient structures (required for backward())
    ///
    /// # Returns
    ///
    /// A configured CompiledSolver ready for parallel processing
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        n: usize,
        m: usize,
        P_row_offsets: &[usize],
        P_col_indices: &[usize],
        A_row_offsets: &[usize],
        A_col_indices: &[usize],
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
        num_threads: usize,
        enable_grad: bool,
    ) -> Result<Self, SolverError> {
        Self::new_with_b_nnz_mask_and_xcones(
            n,
            m,
            P_row_offsets,
            P_col_indices,
            A_row_offsets,
            A_col_indices,
            cones,
            &[],
            settings,
            num_threads,
            enable_grad,
            None,
        )
    }

    /// Same as `new` but with direct-x cones constraining sub-vectors of `x`
    /// to cones (in addition to the slack cones). Pass an empty slice for
    /// the slack-only case.
    ///
    /// `enable_grad=true` with non-empty `dir_cones` routes through the cached
    /// `backward()` path. `backward_with_data()` (the stateless autograd
    /// API) does not yet support direct-x — see `backward_with_data` for
    /// the rejection.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_xcones(
        n: usize,
        m: usize,
        P_row_offsets: &[usize],
        P_col_indices: &[usize],
        A_row_offsets: &[usize],
        A_col_indices: &[usize],
        cones: &[SupportedConeT<T>],
        dir_cones: &[SupportedXConeT],
        settings: DefaultSettings<T>,
        num_threads: usize,
        enable_grad: bool,
    ) -> Result<Self, SolverError> {
        Self::new_with_b_nnz_mask_and_xcones(
            n,
            m,
            P_row_offsets,
            P_col_indices,
            A_row_offsets,
            A_col_indices,
            cones,
            dir_cones,
            settings,
            num_threads,
            enable_grad,
            None,
        )
    }

    /// Same as `new` but with an explicit b sparsity pattern for chordal
    /// decomposition heuristics on PSD problems. `None` means dense b.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_b_nnz_mask(
        n: usize,
        m: usize,
        P_row_offsets: &[usize],
        P_col_indices: &[usize],
        A_row_offsets: &[usize],
        A_col_indices: &[usize],
        cones: &[SupportedConeT<T>],
        settings: DefaultSettings<T>,
        num_threads: usize,
        enable_grad: bool,
        b_sparsity_pattern: Option<&[bool]>,
    ) -> Result<Self, SolverError> {
        Self::new_with_b_nnz_mask_and_xcones(
            n,
            m,
            P_row_offsets,
            P_col_indices,
            A_row_offsets,
            A_col_indices,
            cones,
            &[],
            settings,
            num_threads,
            enable_grad,
            b_sparsity_pattern,
        )
    }

    /// Core constructor: combines direct-x cones (`dir_cones`) with PSD
    /// chordal-decomposition `b_sparsity_pattern`. The other constructors
    /// are thin wrappers around this one.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_b_nnz_mask_and_xcones(
        n: usize,
        m: usize,
        P_row_offsets: &[usize],
        P_col_indices: &[usize],
        A_row_offsets: &[usize],
        A_col_indices: &[usize],
        cones: &[SupportedConeT<T>],
        dir_cones: &[SupportedXConeT],
        mut settings: DefaultSettings<T>,
        num_threads: usize,
        enable_grad: bool,
        b_sparsity_pattern: Option<&[bool]>,
    ) -> Result<Self, SolverError> {
        // Direct-x cones with chordal-decomposed PSD slack: chordal
        // augmentation only changes the slack/row dimension; the primal x
        // (and thus the dir_cones index set) is preserved. We thread dir_cones
        // through the augmented solver below.

        // Create a local thread pool (not global)
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|_e| SolverError::BadInputData("Failed to configure thread pool"))?;

        // Ensure settings are appropriate for batch mode.
        settings.core_mut().ipm.presolve_enable = false;
        // Propagate enable_grad to inner solver settings so that the
        // smoothing iterate is cached during the solve loop.
        settings.core_mut().enable_grad = enable_grad;
        // For batch mode, auto uses faer single-threaded (best for
        // concurrent solves on separate OS threads).
        cfg_if::cfg_if! {
            if #[cfg(feature = "faer-sparse")] {
                if settings.core().ipm.direct_solve_method == "auto" {
                    settings.core_mut().ipm.direct_solve_method = "faer-1t".to_string();
                }
            } else {
                if settings.core().ipm.direct_solve_method == "auto" {
                    settings.core_mut().ipm.direct_solve_method = "qdldl".to_string();
                }
            }
        }

        // Build CSR matrices with dummy values to get structure
        let nnz_P = P_col_indices.len();
        let nnz_A = A_col_indices.len();
        let P_csr = CsrMatrix::new(
            n,
            n,
            P_row_offsets.to_vec(),
            P_col_indices.to_vec(),
            vec![T::zero(); nnz_P],
        );
        let A_csr = CsrMatrix::new(
            m,
            n,
            A_row_offsets.to_vec(),
            A_col_indices.to_vec(),
            vec![T::zero(); nnz_A],
        );

        // Convert to CSC with mapping
        let (P_csc_full, _P_csr_to_csc_full) = P_csr.to_csc_with_mapping();
        let (A_csc, A_csr_to_csc) = A_csr.to_csc_with_mapping();

        // Convert P to upper-triangular (solver internally uses triu form)
        let (P_csc, P_csr_to_csc) = if !P_csc_full.is_triu() {
            let P_triu = P_csc_full.to_triu();
            let mut mapping = vec![usize::MAX; nnz_P];
            let mut csr_idx = 0;
            for row in 0..n {
                let row_start = P_row_offsets[row];
                let row_end = P_row_offsets[row + 1];
                for k in row_start..row_end {
                    let col = P_col_indices[k];
                    if col >= row {
                        let csc_col_start = P_triu.colptr[col];
                        let csc_col_end = P_triu.colptr[col + 1];
                        let mut found = false;
                        for csc_idx in csc_col_start..csc_col_end {
                            if P_triu.rowval[csc_idx] == row {
                                mapping[csr_idx] = csc_idx;
                                found = true;
                                break;
                            }
                        }
                        assert!(found,
                            "P CSR-to-CSC mapping failed: upper-tri entry ({},{}) not found in triu CSC pattern",
                            row, col);
                    }
                    csr_idx += 1;
                }
            }
            (P_triu, mapping)
        } else {
            (P_csc_full, _P_csr_to_csc_full)
        };

        let mut cones_internal = cones.to_vec();
        let mut n_internal = n;
        let mut m_internal = m;
        let P_nnz_orig = P_csc.nnz();
        let A_nnz_orig = A_csc.nnz();
        let mut P_csc_solver = P_csc.clone();
        let mut A_csc_solver = A_csc.clone();

        // Run chordal decomposition at construction time if PSD cones are present.
        // This produces augmented P/A/cone dimensions that all batch solvers share.
        // Individual solvers have chordal disabled (via solver.rs) to prevent double decomp.
        #[cfg(feature = "sdp")]
        let (chordal_info, augmented_problem) = {
            let has_psd = cones_internal
                .iter()
                .any(|c| matches!(c, SupportedConeT::PSDTriangleConeT(_)));

            if has_psd && settings.core().ipm.chordal_decomposition_enable {
                let mut info = ChordalInfo::new_from_pattern(
                    &A_csc_solver,
                    b_sparsity_pattern,
                    m_internal,
                    &cones_internal,
                    settings.core(),
                );

                if info.is_decomposed() {
                    let aug = info.decomp_augment_pattern(&P_csc_solver, &A_csc_solver);

                    if settings.core().verbose {
                        let n_psd_orig = cones_internal
                            .iter()
                            .filter(|c| matches!(c, SupportedConeT::PSDTriangleConeT(_)))
                            .count();
                        let n_psd_aug = aug
                            .cones
                            .iter()
                            .filter(|c| matches!(c, SupportedConeT::PSDTriangleConeT(_)))
                            .count();
                        eprintln!("chordal decomposition (batch):");
                        eprintln!("  PSD cones initial             = {}", n_psd_orig);
                        eprintln!("  PSD cones after decomposition = {}", n_psd_aug);
                        eprintln!("  m: {} -> {}", aug.m_orig, aug.A.m);
                        eprintln!("  n: {} -> {}", aug.n_orig, aug.P.n);
                    }

                    // Override dimensions and patterns for solver pool construction
                    n_internal = aug.P.n;
                    m_internal = aug.A.m;
                    P_csc_solver = aug.P.clone();
                    A_csc_solver = aug.A.clone();
                    cones_internal = aug.cones.clone();

                    (Some(info), Some(aug))
                } else {
                    (None, None)
                }
            } else {
                (None, None)
            }
        };
        #[cfg(not(feature = "sdp"))]
        let () = ();

        // Start with num_threads solvers using internal (potentially augmented) dimensions.
        // The first solver computes AMD ordering; subsequent ones reuse it.
        let mut solver_pool = Vec::with_capacity(num_threads);
        let first_solver = DefaultSolver::new_symbolic_with_xcones_and_perm(
            n_internal,
            m_internal,
            &P_csc_solver,
            &A_csc_solver,
            &cones_internal,
            dir_cones,
            settings.clone(),
            None,
        )?;
        let cached_amd_perm = first_solver.kktsystem.get_amd_perm();
        solver_pool.push(Mutex::new(first_solver));
        for _ in 1..num_threads {
            let solver = DefaultSolver::new_symbolic_with_xcones_and_perm(
                n_internal,
                m_internal,
                &P_csc_solver,
                &A_csc_solver,
                &cones_internal,
                dir_cones,
                settings.clone(),
                cached_amd_perm.clone(),
            )?;
            solver_pool.push(Mutex::new(solver));
        }

        // Pre-compute gradient states if requested (using internal/augmented dimensions)
        // GradState::new now builds the AUGMENTED system for backward pass (m + 2*n)
        let grad_states = if enable_grad {
            let q_placeholder = vec![T::one(); n_internal];
            let b_placeholder = vec![T::one(); m_internal];

            let mut states = Vec::with_capacity(num_threads);
            for _ in 0..num_threads {
                let state = super::diff::GradState::new_with_xcones(
                    &P_csc_solver,
                    &q_placeholder,
                    &A_csc_solver,
                    &b_placeholder,
                    &cones_internal,
                    dir_cones,
                    settings.core(),
                )?;
                states.push(Mutex::new(state));
            }
            states
        } else {
            Vec::new()
        };

        // Precompute nonneg cone row range and CSC indices for inf-in-b handling.
        // Cone layout: zero cones first, then nonneg, then SOC, exp, power.
        let mut nonneg_row_start = 0usize;
        let mut nonneg_row_end = 0usize;
        {
            let mut offset = 0usize;
            for cone in cones.iter() {
                match cone {
                    SupportedConeT::ZeroConeT(d) => {
                        offset += d;
                        nonneg_row_start = offset;
                    }
                    SupportedConeT::NonnegativeConeT(d) => {
                        offset += d;
                        nonneg_row_end = offset;
                    }
                    SupportedConeT::SecondOrderConeT(d) => {
                        offset += d;
                    }
                    SupportedConeT::ExponentialConeT() => {
                        offset += 3;
                    }
                    SupportedConeT::PowerConeT(_) => {
                        offset += 3;
                    }
                    SupportedConeT::GenPowerConeT(alpha, dim2) => {
                        offset += alpha.len() + dim2;
                    }
                    #[cfg(feature = "sdp")]
                    SupportedConeT::PSDTriangleConeT(d) => {
                        offset += d * (d + 1) / 2;
                    }
                };
            }
            // If no nonneg cone, end == start (empty range)
            if nonneg_row_end == 0 {
                nonneg_row_end = nonneg_row_start;
            }
        }

        // For each nonneg row, find the CSC indices in A corresponding to that row.
        let nonneg_row_csc_indices = {
            let mut indices = Vec::with_capacity(nonneg_row_end - nonneg_row_start);
            for row in nonneg_row_start..nonneg_row_end {
                let csr_start = A_row_offsets[row];
                let csr_end = A_row_offsets[row + 1];
                let csc_idxs: Vec<usize> = (csr_start..csr_end)
                    .map(|csr_idx| A_csr_to_csc[csr_idx])
                    .collect();
                indices.push(csc_idxs);
            }
            indices
        };

        Ok(Self {
            n,
            m,
            n_internal,
            m_internal,
            P_row_offsets: P_row_offsets.to_vec(),
            P_col_indices: P_col_indices.to_vec(),
            A_row_offsets: A_row_offsets.to_vec(),
            A_col_indices: A_col_indices.to_vec(),
            P_csr_to_csc,
            A_csr_to_csc,
            P_csc_pattern: P_csc_solver,
            A_csc_pattern: A_csc_solver,
            cones: cones_internal,
            dir_cones: dir_cones.to_vec(),
            settings,
            thread_pool,
            solver_pool,
            batch_size: num_threads,
            num_threads,
            equilibration_precomputed: false,
            solutions_cache: RwLock::new(Vec::new()),
            grad_states,
            grad_enabled: enable_grad,
            cached_amd_perm,
            setup_time: 0.0,
            original_P_values: Vec::new(),
            original_A_values: Vec::new(),
            base_equilibrated_P_values: RwLock::new(Vec::new()),
            matrices_equilibrated: AtomicBool::new(false),
            #[cfg(feature = "sdp")]
            chordal_info,
            #[cfg(feature = "sdp")]
            augmented_problem,
            P_nnz_orig,
            A_nnz_orig,
            nonneg_row_start,
            nonneg_row_end,
            nonneg_row_csc_indices,
        })
    }

    /// Create a batch solver with equilibration enabled.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_equilibration(
        n: usize,
        m: usize,
        P_row_offsets: &[usize],
        P_col_indices: &[usize],
        A_row_offsets: &[usize],
        A_col_indices: &[usize],
        cones: &[SupportedConeT<T>],
        mut settings: DefaultSettings<T>,
        num_threads: usize,
    ) -> Result<Self, SolverError> {
        settings.core_mut().ipm.equilibrate_enable = true;
        // Equilibration solver doesn't need grad
        Self::new(
            n,
            m,
            P_row_offsets,
            P_col_indices,
            A_row_offsets,
            A_col_indices,
            cones,
            settings,
            num_threads,
            false,
        )
    }

    // NOTE: new_with_bounds() has been removed - variable bounds are no longer supported.
    // Use conic constraints instead (e.g., Ax + s = b with s >= 0).

    /// Check whether gradient computation is enabled.
    pub fn grad_enabled(&self) -> bool {
        self.grad_enabled
    }

    /// Convert CSR values to CSC order using precomputed mapping
    fn csr_to_csc_values(&self, csr_values: &[T], mapping: &[usize], target_size: usize) -> Vec<T> {
        let mut csc_values = vec![T::zero(); target_size];
        for (csr_idx, &csc_idx) in mapping.iter().enumerate() {
            // Skip lower-triangular entries (marked with usize::MAX)
            if csc_idx != usize::MAX {
                csc_values[csc_idx] = csr_values[csr_idx];
            }
        }
        csc_values
    }

    /// Convert CSC values back to CSR order using precomputed mapping
    fn csc_to_csr_values(&self, csc_values: &[T], mapping: &[usize]) -> Vec<T> {
        let mut csr_values = vec![T::zero(); mapping.len()];
        for (csr_idx, &csc_idx) in mapping.iter().enumerate() {
            // Skip lower-triangular entries (marked with usize::MAX)
            if csc_idx != usize::MAX {
                csr_values[csr_idx] = csc_values[csc_idx];
            }
        }
        csr_values
    }

    /// Convert P gradient from CSC (upper-tri) back to CSR (full symmetric).
    ///
    /// For symmetric P, the gradient dP is also symmetric: dP[i,j] = dP[j,i].
    /// The backward pass computes gradients in upper-triangular CSC format.
    /// When the input P was full symmetric (both triangles stored), we need to
    /// copy the gradient to both the upper and lower triangle positions.
    fn csc_to_csr_values_symmetric_P(&self, csc_values: &[T], mapping: &[usize]) -> Vec<T> {
        let mut csr_values = vec![T::zero(); mapping.len()];

        // First pass: copy values from CSC to CSR for entries that have direct mappings
        // (upper-triangle entries that map to upper-triangle positions)
        let mut csr_idx = 0;
        for row in 0..self.n {
            let row_start = self.P_row_offsets[row];
            let row_end = self.P_row_offsets[row + 1];
            for k in row_start..row_end {
                let col = self.P_col_indices[k];
                let csc_idx = mapping[csr_idx];

                if csc_idx != usize::MAX {
                    // This entry maps directly (upper-tri in CSR maps to upper-tri in CSC)
                    csr_values[csr_idx] = csc_values[csc_idx];
                } else {
                    // This is a lower-triangle entry (row > col) - find the transpose entry
                    // The transpose is at (col, row) in CSC format
                    // For CSC, we find column 'row' and look for row index 'col'
                    let trans_col = row;
                    let trans_row = col;
                    // Look up (trans_row, trans_col) in the CSC structure
                    // CSC: colptr[j]..colptr[j+1] gives entries in column j, rowval[k] gives row
                    let col_start = self.P_csc_pattern.colptr[trans_col];
                    let col_end = self.P_csc_pattern.colptr[trans_col + 1];
                    let mut found = false;
                    for kk in col_start..col_end {
                        if self.P_csc_pattern.rowval[kk] == trans_row {
                            csr_values[csr_idx] = csc_values[kk];
                            found = true;
                            break;
                        }
                    }
                    assert!(found,
                        "P sparsity pattern inconsistency: lower-tri entry ({},{}) has no upper-tri counterpart ({},{})",
                        row, col, col, row);
                }
                csr_idx += 1;
            }
        }

        csr_values
    }

    /// Sanitize inf entries in b and detect infeasibility.
    ///
    /// For +inf in nonneg rows: zero the A row and set b=1 (not 0, because
    /// b=0 forces s=0 and the IPM barrier -log(s) diverges).
    /// For -inf in nonneg rows or ±inf in zero cone rows: return infeasible.
    fn sanitize_inf_b(
        nonneg_row_start: usize,
        nonneg_row_end: usize,
        nonneg_row_csc_indices: &[Vec<usize>],
        b: &mut [T],
        a_nzval: &mut [T],
    ) -> bool {
        // Check zero cone rows (rows 0..nonneg_row_start) for any inf
        for row in 0..nonneg_row_start {
            if b[row].is_infinite() {
                return true;
            }
        }

        // Check nonneg cone rows
        for row in nonneg_row_start..nonneg_row_end {
            if b[row].is_infinite() {
                if b[row] > T::zero() {
                    b[row] = T::one();
                    let local_idx = row - nonneg_row_start;
                    for &csc_idx in &nonneg_row_csc_indices[local_idx] {
                        a_nzval[csc_idx] = T::zero();
                    }
                } else {
                    return true;
                }
            }
        }

        false
    }

    /// Create an infeasible solution for early return.
    fn infeasible_solution(n: usize, m: usize, setup_time: f64) -> DefaultSolution<T> {
        DefaultSolution {
            x: vec![T::nan(); n],
            z: vec![T::nan(); m],
            s: vec![T::nan(); m],
            z_x: Vec::new(),
            status: crate::solver::core::SolverStatus::PrimalInfeasible,
            obj_val: T::nan(),
            obj_val_dual: T::nan(),
            construction_time: 0.0,
            setup_time,
            solve_time: 0.0,
            iterations: 0,
            r_prim: T::nan(),
            r_dual: T::nan(),
        }
    }

    /// Set P and A matrix values for a batch and precompute equilibration.
    ///
    /// Must be called before `solve_batch()`. Can be called multiple times
    /// to update values for repeated solves with the same structure.
    /// This method performs equilibration of P and A matrices so that subsequent
    /// `solve()` calls only need to apply scaling to q and b vectors.
    ///
    /// The equilibration computes scaling factors d, e from P and A norms using
    /// Ruiz scaling. These factors are cached and reused for all subsequent solves.
    ///
    /// # Arguments
    /// * `P_values_batch` - Non-zero values for P matrix for each problem (Vec of CSR format)
    /// * `A_values_batch` - Non-zero values for A matrix for each problem (Vec of CSR format)
    pub fn setup(
        &mut self,
        P_values_batch: &[Vec<T>],
        A_values_batch: &[Vec<T>],
    ) -> Result<(), SolverError> {
        if P_values_batch.len() != A_values_batch.len() {
            return Err(SolverError::BadInputData(
                "P_values_batch and A_values_batch must have the same length",
            ));
        }

        let batch_size = P_values_batch.len();
        let nnz_p_csr = self.P_csr_to_csc.len();
        let nnz_a_csr = self.A_csr_to_csc.len();

        // Flatten into contiguous arrays and delegate to setup_flat.
        // Previously this used .expect() on the delegated call, which
        // aborted the Python interpreter under panic="abort" on any
        // user-controlled length mismatch (#189). Return the error instead.
        let mut p_flat = Vec::with_capacity(batch_size * nnz_p_csr);
        let mut a_flat = Vec::with_capacity(batch_size * nnz_a_csr);
        for (p, a) in P_values_batch.iter().zip(A_values_batch.iter()) {
            p_flat.extend_from_slice(p);
            a_flat.extend_from_slice(a);
        }
        self.setup_flat(&p_flat, &a_flat, batch_size)
    }

    /// Setup from flat contiguous arrays (zero-copy slicing, no per-problem allocation).
    ///
    /// # Arguments
    /// * `P_values_flat` - All P matrix values concatenated, length = batch_size * nnz_P
    /// * `A_values_flat` - All A matrix values concatenated, length = batch_size * nnz_A
    /// * `batch_size` - Number of problems
    pub fn setup_flat(
        &mut self,
        P_values_flat: &[T],
        A_values_flat: &[T],
        batch_size: usize,
    ) -> Result<(), SolverError> {
        // Use original (pre-chordal) NNZ counts for CSR-to-CSC conversion.
        // P_csc_pattern/A_csc_pattern are augmented when chordal is active,
        // but csr_to_csc mappings reference original dimensions.
        let nnz_p = self.P_nnz_orig;
        let nnz_a = self.A_nnz_orig;
        let nnz_p_csr = self.P_csr_to_csc.len();
        let nnz_a_csr = self.A_csr_to_csc.len();

        if P_values_flat.len() != batch_size * nnz_p_csr {
            return Err(SolverError::BadInputData(
                "P_values_flat length must be batch_size * nnz_P",
            ));
        }
        if A_values_flat.len() != batch_size * nnz_a_csr {
            return Err(SolverError::BadInputData(
                "A_values_flat length must be batch_size * nnz_A",
            ));
        }

        let setup_start = Instant::now();

        // Expand solver pool if needed
        while self.solver_pool.len() < batch_size {
            let solver = DefaultSolver::new_symbolic_with_xcones_and_perm(
                self.n_internal,
                self.m_internal,
                &self.P_csc_pattern,
                &self.A_csc_pattern,
                &self.cones,
                &self.dir_cones,
                self.settings.clone(),
                self.cached_amd_perm.clone(),
            )
            .expect("Failed to create solver");
            self.solver_pool.push(Mutex::new(solver));
        }

        // Clear and resize storage
        self.original_P_values.clear();
        self.original_A_values.clear();
        {
            let mut base_p = self.base_equilibrated_P_values.write().unwrap();
            base_p.clear();
            base_p.reserve(batch_size);
        }
        self.original_P_values.reserve(batch_size);
        self.original_A_values.reserve(batch_size);

        // Setup and equilibrate P, A for each problem, in parallel across
        // the batch (each problem owns its pooled solver). Collect preserves
        // batch order.
        let per_problem: Vec<(Vec<T>, Vec<T>, Vec<T>)> = self.thread_pool.install(|| {
            (0..batch_size)
                .into_par_iter()
                .map(|i| {
                    // Slice into flat buffer — zero allocation for the input
                    let P_values = &P_values_flat[i * nnz_p_csr..(i + 1) * nnz_p_csr];
                    let A_values = &A_values_flat[i * nnz_a_csr..(i + 1) * nnz_a_csr];

                    // Convert CSR to CSC (maps to original/user dimensions)
                    let P_csc_values_orig =
                        self.csr_to_csc_values(P_values, &self.P_csr_to_csc, nnz_p);
                    let A_csc_values_orig =
                        self.csr_to_csc_values(A_values, &self.A_csr_to_csc, nnz_a);

                    // If chordal is active, augment P/A values to solver (augmented) dimensions
                    #[cfg(feature = "sdp")]
                    let (P_csc_values, A_csc_values) = if let Some(ref aug) = self.augmented_problem
                    {
                        let (p_aug, _q, a_aug, _b) =
                            aug.augment_values(&P_csc_values_orig, &[], &A_csc_values_orig, &[]);
                        (p_aug, a_aug)
                    } else {
                        (P_csc_values_orig, A_csc_values_orig)
                    };
                    #[cfg(not(feature = "sdp"))]
                    let (P_csc_values, A_csc_values) = (P_csc_values_orig, A_csc_values_orig);

                    let mut solver = self.solver_pool[i].lock().unwrap();
                    solver.data.P.nzval.copy_from_slice(&P_csc_values);
                    solver.data.A.nzval.copy_from_slice(&A_csc_values);

                    Self::equilibrate_matrices(&mut solver);
                    let base_equilibrated_p = solver.data.P.nzval.clone();

                    // Original (un-equilibrated, augmented if chordal) values
                    (P_csc_values, A_csc_values, base_equilibrated_p)
                })
                .collect()
        });

        let mut base_p = self.base_equilibrated_P_values.write().unwrap();
        for (p_orig, a_orig, base_equilibrated_p) in per_problem {
            self.original_P_values.push(p_orig);
            self.original_A_values.push(a_orig);
            base_p.push(base_equilibrated_p);
        }
        drop(base_p);

        self.equilibration_precomputed = true;
        self.matrices_equilibrated.store(true, Ordering::SeqCst);
        self.batch_size = batch_size;
        self.setup_time = setup_start.elapsed().as_secs_f64();
        Ok(())
    }

    /// Solve from flat contiguous arrays (zero-copy slicing, no per-problem allocation).
    ///
    /// Slices the flat buffers into per-problem `&[T]` references and delegates
    /// to `solve_with_warm_start`. The only allocation is the slice-of-references
    /// vectors (~8 bytes × batch_size each).
    pub fn solve_flat(
        &self,
        q_flat: &[T],
        b_flat: &[T],
        batch_size: usize,
        warm_x_flat: Option<&[T]>,
        warm_z_flat: Option<&[T]>,
        warm_s_flat: Option<&[T]>,
        warm_z_x_flat: Option<&[T]>,
    ) -> Result<Vec<DefaultSolution<T>>, SolverError> {
        let n = self.n;
        let m = self.m;
        let xn: usize = self.dir_cones.iter().map(|c| c.indices().len()).sum();

        if q_flat.len() != batch_size * n {
            return Err(SolverError::BadInputData(
                "q_flat length must be batch_size * n",
            ));
        }
        if b_flat.len() != batch_size * m {
            return Err(SolverError::BadInputData(
                "b_flat length must be batch_size * m",
            ));
        }
        if let Some(flat) = warm_x_flat {
            if flat.len() != batch_size * n {
                return Err(SolverError::BadInputData(
                    "warm_x_flat length must be batch_size * n",
                ));
            }
        }
        if let Some(flat) = warm_z_flat {
            if flat.len() != batch_size * m {
                return Err(SolverError::BadInputData(
                    "warm_z_flat length must be batch_size * m",
                ));
            }
        }
        if let Some(flat) = warm_s_flat {
            if flat.len() != batch_size * m {
                return Err(SolverError::BadInputData(
                    "warm_s_flat length must be batch_size * m",
                ));
            }
        }
        if let Some(flat) = warm_z_x_flat {
            if flat.len() != batch_size * xn {
                return Err(SolverError::BadInputData(
                    "warm_z_x_flat length must be batch_size * xn",
                ));
            }
        }

        // chunks_exact panics on chunk size 0, which would abort the Python
        // interpreter under panic="abort". Reject n=0 here (LP/QP with zero
        // primal variables is degenerate anyway). (#189)
        if n == 0 {
            return Err(SolverError::BadInputData(
                "n=0 (zero primal variables) is not supported",
            ));
        }
        let qs: Vec<Vec<T>> = q_flat.chunks_exact(n).map(|c| c.to_vec()).collect();
        let bs: Vec<Vec<T>> = if m > 0 {
            b_flat.chunks_exact(m).map(|c| c.to_vec()).collect()
        } else {
            vec![vec![]; batch_size]
        };

        let warm_xs: Option<Vec<Vec<T>>> =
            warm_x_flat.map(|flat| flat.chunks_exact(n).map(|c| c.to_vec()).collect());
        let warm_zs: Option<Vec<Vec<T>>> = warm_z_flat.map(|flat| {
            if m > 0 {
                flat.chunks_exact(m).map(|c| c.to_vec()).collect()
            } else {
                vec![vec![]; batch_size]
            }
        });
        let warm_ss: Option<Vec<Vec<T>>> = warm_s_flat.map(|flat| {
            if m > 0 {
                flat.chunks_exact(m).map(|c| c.to_vec()).collect()
            } else {
                vec![vec![]; batch_size]
            }
        });
        let warm_z_xs: Option<Vec<Vec<T>>> = warm_z_x_flat.map(|flat| {
            if xn > 0 {
                flat.chunks_exact(xn).map(|c| c.to_vec()).collect()
            } else {
                vec![vec![]; batch_size]
            }
        });

        self.solve_with_warm_start(
            &qs,
            &bs,
            warm_xs.as_deref(),
            warm_zs.as_deref(),
            warm_ss.as_deref(),
            warm_z_xs.as_deref(),
        )
    }

    /// Equilibrate P and A matrices, computing d and e scaling factors.
    /// This is the Ruiz scaling part that only depends on matrix norms.
    /// Cost scaling (c) is computed later in solve() when q is available.
    fn equilibrate_matrices(solver: &mut DefaultSolver<T>) {
        use itertools::izip;

        let settings = &solver.settings;
        if !settings.ipm.equilibrate_enable {
            return;
        }

        let data = &mut solver.data;
        let equil = &mut data.equilibration;
        let cones = &solver.cones;

        // Reset equilibration to identity
        equil.d.fill(T::one());
        equil.dinv.fill(T::one());
        equil.e.fill(T::one());
        equil.einv.fill(T::one());
        equil.c = T::one();

        let (d, e) = (&mut equil.d, &mut equil.e);
        let dwork = &mut equil.dinv;
        let ework = &mut equil.einv;

        let (P, A) = (&mut data.P, &mut data.A);

        let scale_min = settings.ipm.equilibrate_min_scaling;
        let scale_max = settings.ipm.equilibrate_max_scaling;

        // Ruiz scaling iterations - computes d, e from P, A norms only
        for _iter in 0..settings.ipm.equilibrate_max_iter {
            // Compute column norms of [P; A] and row norms of A
            P.col_norms_sym(dwork);
            A.col_norms_no_reset(dwork);
            A.row_norms(ework);

            // Zero rows/columns should not get scaled
            dwork.scalarop(|x| if x == T::zero() { T::one() } else { x });
            ework.scalarop(|x| if x == T::zero() { T::one() } else { x });

            dwork.rsqrt();
            ework.rsqrt();

            // Bound the cumulative scaling
            for (dw, &dv) in izip!(dwork.iter_mut(), d.iter()) {
                *dw = T::clip(dw, scale_min / dv, scale_max / dv);
            }
            for (ew, &ev) in izip!(ework.iter_mut(), e.iter()) {
                *ew = T::clip(ew, scale_min / ev, scale_max / ev);
            }

            // Scale P and A (not q, b - those are scaled in solve())
            P.lrscale(dwork, dwork);
            A.lrscale(ework, dwork);

            // Update cumulative scaling factors
            d.hadamard(dwork);
            e.hadamard(ework);
        }

        // Rectify cone equilibration (e.g., for SOC cones)
        if cones.rectify_equilibration(ework, e) {
            // Only rescale A if cones were rectified (P unaffected)
            A.lscale(ework);
            e.hadamard(ework);
        }

        // Direct-x uniform-scaling rectification. Mirrors
        // `DefaultProblemData::equilibrate` (problemdata.rs): for any
        // direct-x cone with `requires_uniform_x_scaling()` (SOC/Exp/
        // Power/GenPower/PSD x-cones), replace `d[J]` with the geometric
        // mean over the cone's indices so `x[J] ∈ K_J` is preserved
        // under `x̃ = D⁻¹ x`. Without this, IPM trajectories on x-cone
        // problems diverge from the DefaultSolver direct path.
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
            // Geometric mean of d[indices] via log-mean to avoid
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
            // Rescale P (column+row) and A (column) by `dwork`. This is
            // the d-direction-only analogue of `scale_data(P, A, q, b,
            // Some(dwork), ones)` (we don't have q,b here in batch
            // mode — they're scaled per-problem in solve_internal).
            P.lrscale(dwork, dwork);
            A.lrscale(&vec![T::one(); A.nrows()], dwork);
            d.hadamard(dwork);
        }

        // Compute inverse scaling data
        equil.dinv.scalarop_from(T::recip, d);
        equil.einv.scalarop_from(T::recip, e);

        // Note: c (cost scaling) is NOT computed here - it depends on q
        // and will be computed in solve() when q is available
    }

    /// Set shared P and A matrix values for all problems in a batch.
    ///
    /// Use this when all problems share the same P and A matrices but have
    /// different q and b vectors. This is more efficient than `setup()` because
    /// equilibration is computed once and shared across all problems.
    ///
    /// Must be called before `solve_batch_shared()`.
    ///
    /// # Arguments
    /// * `P_values` - Non-zero values for shared P matrix (CSR format)
    /// * `A_values` - Non-zero values for shared A matrix (CSR format)
    /// * `batch_size` - Number of problems to solve (for pre-allocating solvers)
    pub fn setup_shared(&mut self, P_values: &[T], A_values: &[T], batch_size: usize) {
        let setup_start = Instant::now();

        // Expand solver pool if needed (reuses cached AMD perm)
        while self.solver_pool.len() < batch_size {
            let solver = DefaultSolver::new_symbolic_with_xcones_and_perm(
                self.n_internal,
                self.m_internal,
                &self.P_csc_pattern,
                &self.A_csc_pattern,
                &self.cones,
                &self.dir_cones,
                self.settings.clone(),
                self.cached_amd_perm.clone(),
            )
            .expect("Failed to create solver");
            self.solver_pool.push(Mutex::new(solver));
        }

        // Convert CSR values to CSC order (maps to original/user dimensions)
        let P_csc_values_orig =
            self.csr_to_csc_values(P_values, &self.P_csr_to_csc, self.P_nnz_orig);
        let A_csc_values_orig =
            self.csr_to_csc_values(A_values, &self.A_csr_to_csc, self.A_nnz_orig);

        // If chordal is active, augment P/A values to solver (augmented) dimensions
        #[cfg(feature = "sdp")]
        let (P_csc_values, A_csc_values) = if let Some(ref aug) = self.augmented_problem {
            let (p_aug, _q, a_aug, _b) =
                aug.augment_values(&P_csc_values_orig, &[], &A_csc_values_orig, &[]);
            (p_aug, a_aug)
        } else {
            (P_csc_values_orig, A_csc_values_orig)
        };
        #[cfg(not(feature = "sdp"))]
        let (P_csc_values, A_csc_values) = (P_csc_values_orig, A_csc_values_orig);

        // Store original values - same for all problems
        self.original_P_values.clear();
        self.original_A_values.clear();
        self.base_equilibrated_P_values.write().unwrap().clear();
        self.original_P_values.reserve(batch_size);
        self.original_A_values.reserve(batch_size);
        for _ in 0..batch_size {
            self.original_P_values.push(P_csc_values.clone());
            self.original_A_values.push(A_csc_values.clone());
        }

        // Equilibrate ONCE on first solver, then copy d,e to all others
        {
            let mut solver0 = self.solver_pool[0].lock().unwrap();
            solver0.data.P.nzval.copy_from_slice(&P_csc_values);
            solver0.data.A.nzval.copy_from_slice(&A_csc_values);
            Self::equilibrate_matrices(&mut solver0);
        }

        // Read equilibrated state from solver 0
        let solver0 = self.solver_pool[0].lock().unwrap();
        let eq_P = solver0.data.P.nzval.clone();
        let eq_A = solver0.data.A.nzval.clone();
        let eq_d = solver0.data.equilibration.d.clone();
        let eq_dinv = solver0.data.equilibration.dinv.clone();
        let eq_e = solver0.data.equilibration.e.clone();
        let eq_einv = solver0.data.equilibration.einv.clone();
        drop(solver0);

        // Store base equilibrated P for all problems (same since P/A shared)
        {
            let mut base_p = self.base_equilibrated_P_values.write().unwrap();
            for _ in 0..batch_size {
                base_p.push(eq_P.clone());
            }
        }

        // Copy equilibrated matrices and d,e to all solvers
        for i in 0..batch_size {
            let mut solver = self.solver_pool[i].lock().unwrap();
            solver.data.P.nzval.copy_from_slice(&eq_P);
            solver.data.A.nzval.copy_from_slice(&eq_A);
            solver.data.equilibration.d.copy_from_slice(&eq_d);
            solver.data.equilibration.dinv.copy_from_slice(&eq_dinv);
            solver.data.equilibration.e.copy_from_slice(&eq_e);
            solver.data.equilibration.einv.copy_from_slice(&eq_einv);
            solver.data.equilibration.c = T::one(); // c depends on q, computed at solve time
        }

        self.equilibration_precomputed = true;
        self.matrices_equilibrated.store(true, Ordering::SeqCst); // matrices are now equilibrated
        self.batch_size = batch_size;
        self.setup_time = setup_start.elapsed().as_secs_f64();
    }

    /// Solve a batch of problems.
    ///
    /// This is the primary solve interface for CompiledSolver. Requires `setup()`
    /// to be called first to set P and A matrix values.
    ///
    /// # Arguments
    /// * `qs` - Vector of q arrays (linear cost), one per problem
    /// * `bs` - Vector of b arrays (constraint RHS), one per problem
    ///
    /// # Example
    /// ```no_run
    /// # use moreau::solver::implementations::default::{CompiledSolver, DefaultSettings};
    /// # use moreau::solver::SupportedConeT;
    /// # let cones = vec![SupportedConeT::ZeroConeT::<f64>(1)];
    /// # let settings = DefaultSettings::default();
    /// # let mut solver = CompiledSolver::new(2, 1, &[0,1,2], &[0,1], &[0,2], &[0,1], &cones, settings, 2, false).unwrap();
    /// # let P_values = vec![vec![1.0, 1.0]; 2];
    /// # let A_values = vec![vec![1.0, 1.0]; 2];
    /// solver.setup(&P_values, &A_values);
    ///
    /// let qs = vec![vec![1.0, 2.0]; 2];
    /// let bs = vec![vec![1.0]; 2];
    /// let solutions = solver.solve(&qs, &bs).unwrap();
    /// ```
    pub fn solve(
        &self,
        qs: &[Vec<T>],
        bs: &[Vec<T>],
    ) -> Result<Vec<DefaultSolution<T>>, SolverError> {
        self.solve_with_warm_start(qs, bs, None, None, None, None)
    }

    /// Solve a batch of problems with optional warm starting.
    ///
    /// If warm start vectors are provided (all three must be given together),
    /// the solver uses them as an initial point and applies central-path smoothing
    /// to accelerate convergence. The warm start point should be in the original
    /// (unscaled) problem space — typically a solution from a related problem.
    ///
    /// # Arguments
    /// * `qs` - Linear cost vectors, one per problem
    /// * `bs` - Constraint RHS vectors, one per problem
    /// * `warm_xs` - Optional warm start primal variables (shape: batch × n)
    /// * `warm_zs` - Optional warm start dual variables (shape: batch × m)
    /// * `warm_ss` - Optional warm start slack variables (shape: batch × m)
    pub fn solve_with_warm_start(
        &self,
        qs: &[Vec<T>],
        bs: &[Vec<T>],
        warm_xs: Option<&[Vec<T>]>,
        warm_zs: Option<&[Vec<T>]>,
        warm_ss: Option<&[Vec<T>]>,
        warm_z_xs: Option<&[Vec<T>]>,
    ) -> Result<Vec<DefaultSolution<T>>, SolverError> {
        if !self.equilibration_precomputed {
            return Err(SolverError::BadInputData(
                "Must call setup() before solve()",
            ));
        }

        // Validate input lengths
        let batch_size = qs.len();
        if bs.len() != batch_size {
            return Err(SolverError::BadInputData(
                "qs and bs must have the same length",
            ));
        }
        if batch_size != self.batch_size {
            return Err(SolverError::BadInputData(
                "Batch size must match the size provided to setup()",
            ));
        }

        // Validate warm start: all three must be provided or all None
        let has_warm_start = match (warm_xs, warm_zs, warm_ss) {
            (Some(xs), Some(zs), Some(ss)) => {
                if xs.len() != batch_size || zs.len() != batch_size || ss.len() != batch_size {
                    return Err(SolverError::BadInputData(
                        "Warm start arrays must have the same batch size as qs/bs",
                    ));
                }
                for i in 0..batch_size {
                    if xs[i].len() != self.n {
                        return Err(SolverError::BadInputData("warm_x dimensions must match n"));
                    }
                    if zs[i].len() != self.m || ss[i].len() != self.m {
                        return Err(SolverError::BadInputData(
                            "warm_z and warm_s dimensions must match m",
                        ));
                    }
                }
                true
            }
            (None, None, None) => false,
            _ => {
                return Err(SolverError::BadInputData(
                    "warm_xs, warm_zs, warm_ss must all be provided or all be None",
                ));
            }
        };

        // Warm `z_x` is optional even when (x, z, s) are warm-started.
        if let Some(zxs) = warm_z_xs {
            if !has_warm_start {
                return Err(SolverError::BadInputData(
                    "warm_z_xs requires warm_xs/warm_zs/warm_ss to also be provided",
                ));
            }
            if zxs.len() != batch_size {
                return Err(SolverError::BadInputData(
                    "warm_z_xs must have the same batch size as qs/bs",
                ));
            }
            let xn: usize = self.dir_cones.iter().map(|c| c.indices().len()).sum();
            for zx in zxs {
                if zx.len() != xn {
                    return Err(SolverError::BadInputData(
                        "warm_z_x dimension must match total direct-x cone size",
                    ));
                }
            }
        }

        self.solve_internal(qs, bs, has_warm_start, warm_xs, warm_zs, warm_ss, warm_z_xs)
    }

    /// Internal solve implementation
    fn solve_internal(
        &self,
        qs: &[Vec<T>],
        bs: &[Vec<T>],
        has_warm_start: bool,
        warm_xs: Option<&[Vec<T>]>,
        warm_zs: Option<&[Vec<T>]>,
        warm_ss: Option<&[Vec<T>]>,
        warm_z_xs: Option<&[Vec<T>]>,
    ) -> Result<Vec<DefaultSolution<T>>, SolverError> {
        let start = Instant::now();
        let setup_time_for_closure = self.setup_time;
        let batch_size = qs.len();

        // Check if matrices are already equilibrated (subsequent solve after first)
        let matrices_already_equilibrated = self.matrices_equilibrated.load(Ordering::SeqCst);

        // Pre-allocate base_equilibrated_P_values for first solve so parallel
        // threads can write by index without growing the vec.
        if !matrices_already_equilibrated {
            let mut base_p = self.base_equilibrated_P_values.write().unwrap();
            base_p.clear();
            base_p.resize(batch_size, Vec::new());
        }

        let results: Result<Vec<_>, _> = self.thread_pool.install(|| {
            qs.par_iter()
                .zip(bs.par_iter())
                .enumerate()
                .map(|(i, (q, b))| {
                    let mut solver = self.solver_pool[i].lock().unwrap();
                    solver.info.setup_time = setup_time_for_closure;

                    // Augment q and b if chordal is active, otherwise copy directly
                    #[cfg(feature = "sdp")]
                    let (q_solve, b_solve) = if let Some(ref aug) = self.augmented_problem {
                        let (_, q_aug, _, b_aug) = aug.augment_values(&[], q, &[], b);
                        (q_aug, b_aug)
                    } else {
                        (q.to_vec(), b.to_vec())
                    };
                    #[cfg(feature = "sdp")]
                    {
                        solver.data.q.copy_from_slice(&q_solve);
                        solver.data.b.copy_from_slice(&b_solve);
                    }
                    #[cfg(not(feature = "sdp"))]
                    {
                        solver.data.q.copy_from_slice(q);
                        solver.data.b.copy_from_slice(b);
                    }

                    if !matrices_already_equilibrated {
                        // First solve: restore original P/A and do full equilibration
                        solver
                            .data
                            .P
                            .nzval
                            .copy_from_slice(&self.original_P_values[i]);
                        solver
                            .data
                            .A
                            .nzval
                            .copy_from_slice(&self.original_A_values[i]);

                        // Sanitize inf entries in b and detect infeasibility.
                        let infeasible = {
                            let data = &mut solver.data;
                            Self::sanitize_inf_b(
                                self.nonneg_row_start,
                                self.nonneg_row_end,
                                &self.nonneg_row_csc_indices,
                                &mut data.b,
                                &mut data.A.nzval,
                            )
                        };
                        if infeasible {
                            return Ok((
                                i,
                                Self::infeasible_solution(self.n, self.m, setup_time_for_closure),
                                CachedState::empty(),
                            ));
                        }

                        // Reset equilibration to identity
                        solver.data.equilibration.d.fill(T::one());
                        solver.data.equilibration.dinv.fill(T::one());
                        solver.data.equilibration.e.fill(T::one());
                        solver.data.equilibration.einv.fill(T::one());
                        solver.data.equilibration.c = T::one();
                        solver.data.clear_normq();
                        solver.data.clear_normb();

                        // Full equilibration (modifies P, A, q, b and computes d, e, c)
                        {
                            use crate::solver::core::traits::ProblemData;
                            let solver_ref = &mut *solver;
                            solver_ref
                                .data
                                .equilibrate(&solver_ref.cones, &solver_ref.settings);
                        }

                        // Store base equilibrated P (after d,e scaling, before c scaling)
                        // by undoing the cost scaling c that equilibrate() applied.
                        {
                            let c = solver.data.equilibration.c;
                            let base_p = if c != T::one() && c != T::zero() {
                                let cinv = T::recip(c);
                                solver.data.P.nzval.iter().map(|&v| v * cinv).collect()
                            } else {
                                solver.data.P.nzval.clone()
                            };
                            self.base_equilibrated_P_values.write().unwrap()[i] = base_p;
                        }

                        // Update KKT system with equilibrated matrices
                        {
                            let solver_ref = &mut *solver;
                            solver_ref.kktsystem.update_P(&solver_ref.data.P);
                            solver_ref.kktsystem.update_A(&solver_ref.data.A);
                        }
                    } else {
                        // Subsequent solve: P and A already equilibrated with d,e
                        // Restore P to base equilibrated state (before cost scaling c)
                        // This ensures each solve starts from the same state
                        let base_p = self.base_equilibrated_P_values.read().unwrap();
                        solver.data.P.nzval.copy_from_slice(&base_p[i]);

                        // Sanitize inf entries in b before scaling and detect infeasibility.
                        // NOTE: A rows zeroed by sanitize_inf_b on the first solve are NOT
                        // restored here. If the inf pattern in b changes between solves
                        // (e.g. b[i] goes from +inf to finite), call setup() again to
                        // reload original A values and re-equilibrate.
                        {
                            let infeasible = {
                                let data = &mut solver.data;
                                Self::sanitize_inf_b(
                                    self.nonneg_row_start,
                                    self.nonneg_row_end,
                                    &self.nonneg_row_csc_indices,
                                    &mut data.b,
                                    &mut data.A.nzval,
                                )
                            };
                            if infeasible {
                                return Ok((
                                    i,
                                    Self::infeasible_solution(
                                        self.n,
                                        self.m,
                                        setup_time_for_closure,
                                    ),
                                    CachedState::empty(),
                                ));
                            }
                        }

                        // Reset cost scaling to 1 (will be recomputed below)
                        solver.data.equilibration.c = T::one();

                        // Apply stored d scaling to new q: q = d * q
                        // Apply stored e scaling to new b: b = e * b
                        {
                            let data = &mut solver.data;
                            data.q.hadamard(&data.equilibration.d);
                            data.b.hadamard(&data.equilibration.e);
                        }

                        // Compute cost scaling based on base-equilibrated P and new q
                        let equilibrate_enable = solver.settings.ipm.equilibrate_enable;
                        let nnz_p = solver.data.P.nnz();
                        if equilibrate_enable && nnz_p > 0 {
                            let scale_min = solver.settings.ipm.equilibrate_min_scaling;
                            let scale_max = solver.settings.ipm.equilibrate_max_scaling;

                            // Compute P column norms and mean from base-equilibrated P
                            let n = solver.data.P.n;
                            let mut dwork = vec![T::zero(); n];
                            solver.data.P.col_norms(&mut dwork);
                            let mean_col_norm_P =
                                dwork.iter().copied().fold(T::zero(), |a, b| a + b)
                                    / T::from(dwork.len()).unwrap();
                            let inf_norm_q = solver.data.q.norm_inf();

                            if mean_col_norm_P != T::zero() && inf_norm_q != T::zero() {
                                let scale_cost = T::max(inf_norm_q, mean_col_norm_P);
                                let ctmp = T::recip(scale_cost);
                                // Clip relative to c=1 (since we reset c above)
                                let ctmp = T::clip(&ctmp, scale_min, scale_max);

                                // Scale P, q, and c
                                solver.data.P.scale(ctmp);
                                solver.data.q.scale(ctmp);
                                solver.data.equilibration.c = ctmp;
                            }
                        }

                        // Clear cached norms so they get recomputed
                        solver.data.clear_normq();
                        solver.data.clear_normb();

                        // Update KKT with equilibrated P (cost scaling applied)
                        {
                            let solver_ref = &mut *solver;
                            solver_ref.kktsystem.update_P(&solver_ref.data.P);
                            solver_ref.kktsystem.update_A(&solver_ref.data.A);
                        }
                    }

                    // Apply warm start if provided
                    // Follows the algorithm from arXiv:2512.00693 and Clarabel.jl
                    // yc/warmstart branch: smoothing is done directly in
                    // equilibrated space (valid because rectify_equilibration
                    // ensures uniform scaling within each cone block).
                    if has_warm_start {
                        let warm_x_orig = &warm_xs.unwrap()[i];
                        let warm_z_orig = &warm_zs.unwrap()[i];
                        let warm_s_orig = &warm_ss.unwrap()[i];

                        // If chordal is active, augment warm-start vectors from
                        // user dimensions to solver (augmented) dimensions.
                        #[cfg(feature = "sdp")]
                        let (warm_x, warm_z, warm_s);
                        #[cfg(feature = "sdp")]
                        {
                            if let Some(ref aug) = self.augmented_problem {
                                let mut x_aug = vec![T::zero(); self.n_internal];
                                x_aug[..self.n].copy_from_slice(warm_x_orig);
                                let mut z_aug = vec![T::zero(); self.m_internal];
                                let mut s_aug = vec![T::zero(); self.m_internal];
                                for (aug_row, &orig_row) in aug.b_row_map.iter().enumerate() {
                                    if orig_row != usize::MAX {
                                        z_aug[aug_row] = warm_z_orig[orig_row];
                                        s_aug[aug_row] = warm_s_orig[orig_row];
                                    }
                                }
                                warm_x = x_aug;
                                warm_z = z_aug;
                                warm_s = s_aug;
                            } else {
                                warm_x = warm_x_orig.to_vec();
                                warm_z = warm_z_orig.to_vec();
                                warm_s = warm_s_orig.to_vec();
                            }
                        }
                        #[cfg(not(feature = "sdp"))]
                        let (warm_x, warm_z, warm_s) = (
                            warm_x_orig.to_vec(),
                            warm_z_orig.to_vec(),
                            warm_s_orig.to_vec(),
                        );

                        // Step 1: Copy warm values and scale into equilibrated
                        // space so we can compute residuals/info.
                        solver.variables.x.copy_from_slice(&warm_x);
                        solver.variables.z.copy_from_slice(&warm_z);
                        solver.variables.s.copy_from_slice(&warm_s);
                        solver.variables.τ = T::one();
                        solver.variables.κ = T::one();

                        {
                            let solver_ref = &mut *solver;
                            let equil = &solver_ref.data.equilibration;
                            let vars = &mut solver_ref.variables;
                            let c = equil.c;
                            for j in 0..vars.x.len() {
                                vars.x[j] *= equil.dinv[j];
                            }
                            for j in 0..vars.z.len() {
                                vars.z[j] *= equil.einv[j] * c;
                            }
                            for j in 0..vars.s.len() {
                                vars.s[j] *= equil.e[j];
                            }
                        }

                        // Direct-x dual: equilibrate the user-supplied
                        // warm z_x, or — when it is omitted — cold-init
                        // the whole direct-x cone block (x[J] and z_x).
                        // Resetting z_x alone would pair a default dual
                        // with a boundary-valued warm x and strand the
                        // IPM on the cone face.
                        if !solver.variables.z_x.is_empty() {
                            match warm_z_xs {
                                Some(zxs) => {
                                    let warm_z_x = &zxs[i];
                                    let solver_ref = &mut *solver;
                                    let c = solver_ref.data.equilibration.c;
                                    let d = &solver_ref.data.equilibration.d;
                                    let mut off = 0usize;
                                    for xcone in &solver_ref.data.dir_cones {
                                        for (k, &idx) in xcone.indices().iter().enumerate() {
                                            solver_ref.variables.z_x[off + k] =
                                                warm_z_x[off + k] * c / d[idx];
                                        }
                                        off += xcone.indices().len();
                                    }
                                }
                                None => {
                                    let symmetric = solver.cones.is_symmetric()
                                        && solver.kktsystem.dir_cones_ref().is_symmetric();
                                    let solver_ref = &mut *solver;
                                    solver_ref.variables.reinit_direct_x_cone_block(
                                        solver_ref.kktsystem.dir_cones_mut(),
                                        symmetric,
                                    );
                                }
                            }
                        }

                        // Step 2: Compute residuals at the scaled warm point.
                        {
                            let solver_ref = &mut *solver;
                            solver_ref
                                .residuals
                                .update(&solver_ref.variables, &solver_ref.data);
                        }

                        // Step 3: Compute warmness ratio from info (same as
                        // info.update()).
                        let mu_warm = {
                            let τinv = T::recip(solver.variables.τ);
                            let normb = solver.data.get_normb();
                            let normq = solver.data.get_normq();
                            let d = &solver.data.equilibration.d;
                            let e = &solver.data.equilibration.e;
                            let dinv = &solver.data.equilibration.dinv;
                            let einv = &solver.data.equilibration.einv;
                            let cinv = T::recip(solver.data.equilibration.c);

                            let xPx_τinvsq_over2 =
                                solver.residuals.dot_xPx * τinv * τinv / (2.).as_T();
                            let cost_primal =
                                (solver.residuals.dot_qx * τinv + xPx_τinvsq_over2) * cinv;
                            let cost_dual =
                                (-solver.residuals.dot_bz * τinv - xPx_τinvsq_over2) * cinv;

                            let normx = solver.variables.x.norm_scaled(d) * τinv;
                            let normz = solver.variables.z.norm_scaled(e) * cinv * τinv;
                            let norms = solver.variables.s.norm_scaled(einv) * τinv;

                            let res_primal = solver.residuals.rz.norm_scaled(einv) * τinv
                                / T::max(T::one(), normb + normx + norms);
                            let res_dual = solver.residuals.rx.norm_scaled(dinv) * τinv * cinv
                                / T::max(T::one(), normq + normx + normz);

                            let gap_abs = T::abs(cost_primal - cost_dual);
                            let gap_rel = gap_abs
                                / T::max(T::one(), T::min(T::abs(cost_primal), T::abs(cost_dual)));

                            T::max(T::max(res_primal, res_dual), T::min(gap_abs, gap_rel))
                                .max((1e-6).as_T())
                        };

                        // Step 4: Set τ=1, κ=mu_warm
                        solver.variables.τ = T::one();
                        solver.variables.κ = mu_warm;

                        // Step 5: Smooth z_eq, s_eq directly in equilibrated space.
                        let m = solver.variables.z.len();
                        let mut work = vec![T::zero(); m];
                        for j in 0..m {
                            work[j] = solver.variables.z[j] - solver.variables.s[j];
                        }

                        {
                            let solver_ref = &mut *solver;
                            solver_ref.cones.smoothing(
                                &mut solver_ref.variables.z,
                                &solver_ref.variables.s,
                                &work,
                                mu_warm,
                            );
                        }

                        for j in 0..m {
                            solver.variables.s[j] = solver.variables.z[j] - work[j];
                        }

                        // Skip default_start in inner solver
                        solver.skip_default_start = true;
                    }

                    // Solve
                    use crate::solver::IPSolver;
                    solver.info.status = crate::solver::core::SolverStatus::Unsolved;
                    solver.solve();

                    // Cache state for backward pass — store the augmented solution
                    // (before reverse-mapping) so the backward pass operates in augmented space
                    let cached_state = if self.grad_enabled {
                        CachedState::from_solver(&solver)
                    } else {
                        CachedState::empty()
                    };

                    // Reverse-map solution from augmented to original space when chordal is active
                    #[cfg(feature = "sdp")]
                    let (sol_x, sol_z, sol_s) = if let Some(ref chordal_info) = self.chordal_info {
                        use crate::solver::DefaultVariables;
                        let aug_vars = DefaultVariables {
                            x: solver.solution.x.clone(),
                            z: solver.solution.z.clone(),
                            s: solver.solution.s.clone(),
                            z_x: solver.variables.z_x.clone(),
                            τ: solver.variables.τ,
                            κ: solver.variables.κ,
                        };
                        let orig_vars =
                            chordal_info.decomp_reverse(&aug_vars, &self.cones, &self.settings);
                        (orig_vars.x, orig_vars.z, orig_vars.s)
                    } else {
                        (
                            solver.solution.x.clone(),
                            solver.solution.z.clone(),
                            solver.solution.s.clone(),
                        )
                    };
                    #[cfg(not(feature = "sdp"))]
                    let (sol_x, sol_z, sol_s) = (
                        solver.solution.x.clone(),
                        solver.solution.z.clone(),
                        solver.solution.s.clone(),
                    );

                    // Build solution
                    let solution = DefaultSolution {
                        x: sol_x,
                        z: sol_z,
                        s: sol_s,
                        z_x: solver.solution.z_x.clone(),
                        status: solver.solution.status,
                        obj_val: solver.solution.obj_val,
                        obj_val_dual: solver.solution.obj_val_dual,
                        construction_time: 0.0,
                        setup_time: 0.0,
                        solve_time: 0.0,
                        iterations: solver.solution.iterations,
                        r_prim: solver.solution.r_prim,
                        r_dual: solver.solution.r_dual,
                    };

                    Ok((i, solution, cached_state))
                })
                .collect()
        });

        // Process results
        let solve_time = start.elapsed().as_secs_f64();
        let setup_time = self.setup_time;
        results.map(|mut indexed_results| {
            indexed_results.sort_by_key(|(i, _, _)| *i);

            let mut solutions = Vec::with_capacity(batch_size);
            let mut cache = if self.grad_enabled {
                Vec::with_capacity(batch_size)
            } else {
                Vec::new()
            };

            for (_, mut sol, cached_state) in indexed_results {
                sol.construction_time = 0.0;
                sol.setup_time = setup_time;
                sol.solve_time = solve_time;
                solutions.push(sol);
                if self.grad_enabled {
                    cache.push(cached_state);
                }
            }

            // Store cache for backward pass
            if self.grad_enabled {
                *self.solutions_cache.write().unwrap() = cache;
            }

            // Mark matrices as equilibrated for subsequent solves
            // (only vectors q/b will be equilibrated on next solve)
            if !matrices_already_equilibrated {
                self.matrices_equilibrated.store(true, Ordering::SeqCst);
            }

            solutions
        })
    }

    /// Precompute equilibration once for repeated solves where only b changes.
    ///
    /// Deprecated: Use setup() instead for three-step API.
    pub fn precompute_equilibration(&mut self, P_values: &[T], q: &[T], A_values: &[T], b: &[T]) {
        // Convert CSR values to CSC order
        let P_csc_values =
            self.csr_to_csc_values(P_values, &self.P_csr_to_csc, self.P_csc_pattern.nnz());
        let A_csc_values =
            self.csr_to_csc_values(A_values, &self.A_csr_to_csc, self.A_csc_pattern.nnz());
        let q_for_solver = q.to_vec();
        let b_for_solver = b.to_vec();

        let mut P = self.P_csc_pattern.clone();
        P.nzval.copy_from_slice(&P_csc_values);

        let mut A = self.A_csc_pattern.clone();
        A.nzval.copy_from_slice(&A_csc_values);

        for solver_mutex in &self.solver_pool {
            let mut solver = solver_mutex.lock().unwrap();
            solver.precompute_equilibration(&P, &q_for_solver, &A, &b_for_solver);
        }

        self.equilibration_precomputed = true;
        // All solvers share the same P, A - batch_size is pool size
        self.batch_size = self.solver_pool.len();
    }

    /// Solve a batch of problems using precomputed equilibration.
    ///
    /// Requires `setup()` to be called first with matching batch size.
    /// Each problem uses its corresponding equilibrated P, A from `setup()`.
    ///
    /// # Arguments
    /// * `q_vectors` - Vector of q arrays, one per problem
    /// * `b_vectors` - Vector of b arrays, one per problem
    pub fn solve_batch(
        &self,
        q_vectors: &[Vec<T>],
        b_vectors: &[Vec<T>],
    ) -> Result<Vec<DefaultSolution<T>>, SolverError> {
        self.solve(q_vectors, b_vectors)
    }

    /// Solve a batch of problems using shared P and A from setup_shared().
    ///
    /// Use this after calling setup_shared() when all problems share the same
    /// P and A matrices but have different q and b vectors.
    ///
    /// # Arguments
    /// * `q_batch` - Linear cost vectors, shape (batch, n) flattened to slice
    /// * `b_batch` - Constraint RHS vectors, shape (batch, m) flattened to slice
    /// * `batch_size` - Number of problems (must match setup_shared batch_size)
    ///
    /// # Returns
    /// Vector of solutions, one per problem
    pub fn solve_batch_shared(
        &self,
        q_batch: &[T],
        b_batch: &[T],
        batch_size: usize,
    ) -> Result<Vec<DefaultSolution<T>>, SolverError> {
        if !self.equilibration_precomputed {
            return Err(SolverError::BadInputData(
                "Must call setup_shared() before solve_batch_shared()",
            ));
        }
        if batch_size != self.batch_size {
            return Err(SolverError::BadInputData(
                "Batch size must match the size provided to setup_shared()",
            ));
        }

        // Validate input sizes
        let expected_q_size = batch_size * self.n;
        let expected_b_size = batch_size * self.m;
        if q_batch.len() != expected_q_size {
            return Err(SolverError::BadInputData(
                "q_batch size mismatch: expected batch_size * n elements",
            ));
        }
        if b_batch.len() != expected_b_size {
            return Err(SolverError::BadInputData(
                "b_batch size mismatch: expected batch_size * m elements",
            ));
        }

        let n = self.n;
        let m = self.m;
        let qs: Vec<Vec<T>> = q_batch.chunks_exact(n).map(|c| c.to_vec()).collect();
        let bs: Vec<Vec<T>> = if m > 0 {
            b_batch.chunks_exact(m).map(|c| c.to_vec()).collect()
        } else {
            vec![vec![]; batch_size]
        };
        self.solve(&qs, &bs)
    }

    // NOTE: solve_batch_b_only and solve_batch_q_b removed - equilibration now happens at solve time
    // NOTE: solve_batch_with_bounds and solve_batch_q_b_l_u removed - bounds no longer supported

    /// Process a batch of problems in parallel (solve only, no gradients)
    ///
    /// # Arguments
    ///
    /// * `problems` - Vector of problems to solve (values in CSR order)
    ///
    /// # Returns
    ///
    /// Vector of solutions. All solutions share the same `solve_time` field
    /// representing the total batch wall-clock time (consistent with GPU behavior).
    /// Per-problem `iterations` are still available for debugging.
    pub fn solve_batch_parallel(
        &self,
        problems: &[BatchProblem<T>],
    ) -> Result<Vec<DefaultSolution<T>>, SolverError> {
        use crate::solver::SolverStatus;

        let is_verbose = self.settings.verbose;

        // Print header once if verbose (same format as regular solver)
        if is_verbose {
            let mut stdout = std::io::stdout();
            banner::print_banner(&mut stdout, true, "CPU Batch Solver")?;
            writeln!(stdout)?;
            writeln!(stdout, "batch:")?;
            writeln!(stdout, "  ├ problems = {}", problems.len())?;
            writeln!(stdout, "  ├ threads  = {}", self.solver_pool.len())?;
            writeln!(stdout, "  └ n = {}, m = {}", self.n, self.m)?;
            writeln!(stdout)?;
        }

        let start = Instant::now();

        // Process in parallel using pre-allocated solvers from the pool
        let num_solvers = self.solver_pool.len();
        let results: Result<Vec<_>, _> = self.thread_pool.install(|| {
            problems
                .par_iter()
                .enumerate()
                .map(|(i, problem)| {
                    // Get solver from pool (round-robin assignment)
                    let solver_idx = i % num_solvers;
                    let mut solver = self.solver_pool[solver_idx].lock().unwrap();

                    // Silence individual solver output (we print batch summary instead)
                    let original_verbose = solver.settings.verbose;
                    solver.settings.verbose = false;

                    // Convert CSR values to original CSC order, then augment if chordal is active.
                    let P_csc_values_orig = self.csr_to_csc_values(
                        &problem.P_values,
                        &self.P_csr_to_csc,
                        self.P_nnz_orig,
                    );
                    let A_csc_values_orig = self.csr_to_csc_values(
                        &problem.A_values,
                        &self.A_csr_to_csc,
                        self.A_nnz_orig,
                    );

                    #[cfg(feature = "sdp")]
                    let (P_csc_values, q_solve, A_csc_values, b_solve) =
                        if let Some(ref aug) = self.augmented_problem {
                            aug.augment_values(
                                &P_csc_values_orig,
                                &problem.q,
                                &A_csc_values_orig,
                                &problem.b,
                            )
                        } else {
                            (
                                P_csc_values_orig,
                                problem.q.clone(),
                                A_csc_values_orig,
                                problem.b.clone(),
                            )
                        };
                    #[cfg(not(feature = "sdp"))]
                    let (P_csc_values, q_solve, A_csc_values, b_solve) = (
                        P_csc_values_orig,
                        problem.q.clone(),
                        A_csc_values_orig,
                        problem.b.clone(),
                    );

                    // Reconstruct P matrix from pattern + values
                    let mut P = self.P_csc_pattern.clone();
                    P.nzval.copy_from_slice(&P_csc_values);

                    // Reconstruct A matrix from pattern + values
                    let mut A = self.A_csc_pattern.clone();
                    A.nzval.copy_from_slice(&A_csc_values);

                    // Solve with actual problem data
                    let solve_result = solver.solve_batch(&P, &q_solve, &A, &b_solve);

                    // Restore verbose setting
                    solver.settings.verbose = original_verbose;

                    solve_result?;

                    // Cache state for backward pass (skip allocations when grad disabled)
                    let cached_state = if self.grad_enabled {
                        CachedState::from_solver(&solver)
                    } else {
                        CachedState::empty()
                    };

                    #[cfg(feature = "sdp")]
                    let (sol_x, sol_z, sol_s) = if let Some(ref chordal_info) = self.chordal_info {
                        use crate::solver::DefaultVariables;
                        let aug_vars = DefaultVariables {
                            x: solver.solution.x.clone(),
                            z: solver.solution.z.clone(),
                            s: solver.solution.s.clone(),
                            z_x: solver.variables.z_x.clone(),
                            τ: solver.variables.τ,
                            κ: solver.variables.κ,
                        };
                        let orig_vars =
                            chordal_info.decomp_reverse(&aug_vars, &self.cones, &self.settings);
                        (orig_vars.x, orig_vars.z, orig_vars.s)
                    } else {
                        (
                            solver.solution.x.clone(),
                            solver.solution.z.clone(),
                            solver.solution.s.clone(),
                        )
                    };
                    #[cfg(not(feature = "sdp"))]
                    let (sol_x, sol_z, sol_s) = (
                        solver.solution.x.clone(),
                        solver.solution.z.clone(),
                        solver.solution.s.clone(),
                    );

                    // Copy solution (times set to batch time after parallel section)
                    let solution = DefaultSolution {
                        x: sol_x,
                        z: sol_z,
                        s: sol_s,
                        z_x: solver.solution.z_x.clone(),
                        status: solver.solution.status,
                        obj_val: solver.solution.obj_val,
                        obj_val_dual: solver.solution.obj_val_dual,
                        construction_time: 0.0,
                        setup_time: 0.0,
                        solve_time: 0.0,
                        iterations: solver.solution.iterations,
                        r_prim: solver.solution.r_prim,
                        r_dual: solver.solution.r_dual,
                    };

                    Ok((i, solution, cached_state))
                })
                .collect()
        });

        // Process results: separate solutions and cache states
        let elapsed = start.elapsed();
        let batch_len = problems.len();

        let processed_results = results.map(|mut indexed_results| {
            // Sort by index for consistent ordering
            indexed_results.sort_by_key(|(i, _, _)| *i);

            // Extract solutions and cache
            let mut solutions = Vec::with_capacity(batch_len);
            let mut cache = if self.grad_enabled {
                Vec::with_capacity(batch_len)
            } else {
                Vec::new()
            };

            for (_, mut sol, cached_state) in indexed_results {
                sol.construction_time = elapsed.as_secs_f64();
                sol.setup_time = 0.0;
                sol.solve_time = elapsed.as_secs_f64();
                solutions.push(sol);
                if self.grad_enabled {
                    cache.push(cached_state);
                }
            }

            // Store cache for backward pass
            if self.grad_enabled {
                *self.solutions_cache.write().unwrap() = cache;
            }

            solutions
        });

        // Print batch solve summary if verbose
        if is_verbose {
            if let Ok(ref solutions) = processed_results {
                let solved = solutions
                    .iter()
                    .filter(|s| s.status == SolverStatus::Solved)
                    .count();
                let almost = solutions
                    .iter()
                    .filter(|s| s.status == SolverStatus::AlmostSolved)
                    .count();
                let failed = solutions.len() - solved - almost;
                let total_iters: u32 = solutions.iter().map(|s| s.iterations).sum();
                let per_problem = elapsed.as_secs_f64() / batch_len as f64;

                println!("─────────────────────────────────────────────────────────────────────────────────────────────");
                println!(
                    "✓ Batch complete: {} solved, {} almost, {} failed",
                    solved, almost, failed
                );
                println!(
                    "solve time = {:?} ({:.2}ms/problem, {} total iters)",
                    elapsed,
                    per_problem * 1000.0,
                    total_iters
                );
            }
        }

        processed_results
    }

    /// Compute gradients for a batch of problems via backward pass using cached state.
    ///
    /// This method requires that solve_batch_parallel was called first. It uses the
    /// cached solutions and equilibration factors from the last solve to compute
    /// gradients without re-solving.
    ///
    /// # Arguments
    ///
    /// * `upstream_grads` - Upstream gradients for each problem in the batch
    ///
    /// # Returns
    ///
    /// Vector of computed gradients (dP, dq, dA, db) for each problem.
    /// Gradients are returned in CSR order (same as input values).
    ///
    /// # Errors
    ///
    /// Returns error if no cached state exists (solve_batch_parallel not called first)
    pub fn backward(
        &self,
        upstream_grads: &[UpstreamGradients<T>],
    ) -> Result<Vec<ComputedGradients<T>>, SolverError> {
        // Require enable_grad=true in constructor
        if !self.grad_enabled {
            return Err(SolverError::BadInputData(
                "backward() requires enable_grad=true in CompiledSolver constructor",
            ));
        }

        let start = Instant::now();
        let batch_size = upstream_grads.len();
        let num_solvers = self.solver_pool.len();

        // Verify cache exists and matches batch size
        let cache = self.solutions_cache.read().unwrap();
        if cache.is_empty() {
            return Err(SolverError::BadInputData(
                "backward() requires solve_batch_parallel() to be called first",
            ));
        }
        if cache.len() != batch_size {
            return Err(SolverError::BadInputData(
                "Batch size mismatch: upstream_grads length must match cached batch size",
            ));
        }
        drop(cache); // Release lock before parallel section

        let num_grad_states = self.grad_states.len();
        let results: Result<Vec<_>, _> = self.thread_pool.install(|| {
            upstream_grads
                .par_iter()
                .enumerate()
                .map(|(i, grads)| {
                    // Get solver from pool (round-robin)
                    let solver_idx = i % num_solvers;
                    let mut solver = self.solver_pool[solver_idx].lock().unwrap();

                    // Get grad_state from pool if available (round-robin, same index as solver)
                    let mut grad_state_guard = if num_grad_states > 0 {
                        Some(
                            self.grad_states[solver_idx % num_grad_states]
                                .lock()
                                .unwrap(),
                        )
                    } else {
                        None
                    };

                    // Use cached state - no re-solving needed
                    let cache = self.solutions_cache.read().unwrap();
                    let cached = &cache[i];

                    let diff_method = self.settings.ipm.diff_method;

                    // Restore converged solution and info
                    solver.solution.x.copy_from_slice(&cached.x);
                    solver.solution.z.copy_from_slice(&cached.z);
                    solver.solution.s.copy_from_slice(&cached.s);
                    if !cached.z_x.is_empty() {
                        solver.variables.z_x.copy_from_slice(&cached.z_x);
                    }
                    solver.info.mu = cached.mu;

                    // Restore equilibration
                    solver.data.equilibration.d.copy_from_slice(&cached.d);
                    solver.data.equilibration.dinv.copy_from_slice(&cached.dinv);
                    solver.data.equilibration.e.copy_from_slice(&cached.e);
                    solver.data.equilibration.einv.copy_from_slice(&cached.einv);
                    solver.data.equilibration.c = cached.c;

                    // Restore problem data (equilibrated)
                    solver.data.P.nzval.copy_from_slice(&cached.P_values);
                    solver.data.q.copy_from_slice(&cached.q);
                    solver.data.A.nzval.copy_from_slice(&cached.A_values);
                    solver.data.b.copy_from_slice(&cached.b);

                    // Restore smoothing iterate (equilibrated HSDE coordinates)
                    solver.smoothing_cached = cached.smoothing_cached;
                    solver.smoothing_vars.x.copy_from_slice(&cached.smoothing_x);
                    solver.smoothing_vars.z.copy_from_slice(&cached.smoothing_z);
                    solver.smoothing_vars.s.copy_from_slice(&cached.smoothing_s);
                    solver.smoothing_vars.τ = cached.smoothing_tau;

                    drop(cache); // Release lock after reading

                    // Map upstream gradients from original to augmented space if chordal is active
                    #[cfg(feature = "sdp")]
                    let (dx_solve, ds_solve, dz_solve) =
                        if let Some(ref chordal_info) = self.chordal_info {
                            let aug = self.augmented_problem.as_ref().unwrap();
                            let m_aug = aug.A.m;

                            // dx: pad with zeros for overlap variables
                            let mut dx_aug = vec![T::zero(); aug.P.n];
                            dx_aug[..aug.n_orig].copy_from_slice(&grads.dx);

                            // ds, dz: adjoint of decomp_reverse_compact
                            let (ds_aug, dz_aug) = chordal_info.adjoint_decomp_reverse_compact(
                                &grads.ds,
                                &grads.dz,
                                &self.cones,
                                m_aug,
                            );

                            (dx_aug, ds_aug, dz_aug)
                        } else {
                            (grads.dx.clone(), grads.ds.clone(), grads.dz.clone())
                        };
                    #[cfg(not(feature = "sdp"))]
                    let (dx_solve, ds_solve, dz_solve) =
                        (grads.dx.clone(), grads.ds.clone(), grads.dz.clone());

                    // backward_batch_with_dz_x: identical to backward_batch
                    // when grads.dz_x is empty (slack-only); for direct-x
                    // problems it threads the upstream z_x gradient into
                    // the IFT-direct adjoint RHS.
                    let backward_result = solver.backward_batch_with_dz_x(
                        &dx_solve,
                        &ds_solve,
                        &dz_solve,
                        &grads.dz_x,
                        grad_state_guard.as_deref_mut(),
                        diff_method,
                    )?;

                    // Map output gradients from augmented to original space if chordal is active
                    #[cfg(feature = "sdp")]
                    let (dP_values, dq_out, dA_values, db_out) =
                        if let Some(ref aug) = self.augmented_problem {
                            // dq: first n_orig entries (overlap variables discarded)
                            let dq_orig = backward_result.dq[..aug.n_orig].to_vec();

                            // db: reverse the b_row_map permutation (adjoint of scatter)
                            let mut db_orig = vec![T::zero(); aug.m_orig];
                            for (aug_i, &orig_i) in aug.b_row_map.iter().enumerate() {
                                if orig_i != usize::MAX {
                                    db_orig[orig_i] += backward_result.db[aug_i];
                                }
                            }

                            // dP: first P_nnz_orig entries of augmented P gradient (CSC)
                            let dP_csc_orig = backward_result.dP.nzval[..aug.P_nnz_orig].to_vec();
                            // dA: first A_nnz_orig entries of augmented A gradient (CSC)
                            let dA_csc_orig = backward_result.dA.nzval[..aug.A_nnz_orig].to_vec();

                            let dP_csr = self
                                .csc_to_csr_values_symmetric_P(&dP_csc_orig, &self.P_csr_to_csc);
                            let dA_csr = self.csc_to_csr_values(&dA_csc_orig, &self.A_csr_to_csc);

                            (dP_csr, dq_orig, dA_csr, db_orig)
                        } else {
                            let dP_values = self.csc_to_csr_values_symmetric_P(
                                &backward_result.dP.nzval,
                                &self.P_csr_to_csc,
                            );
                            let dA_values = self
                                .csc_to_csr_values(&backward_result.dA.nzval, &self.A_csr_to_csc);
                            (dP_values, backward_result.dq, dA_values, backward_result.db)
                        };
                    #[cfg(not(feature = "sdp"))]
                    let (dP_values, dq_out, dA_values, db_out) = {
                        let dP_values = self.csc_to_csr_values_symmetric_P(
                            &backward_result.dP.nzval,
                            &self.P_csr_to_csc,
                        );
                        let dA_values =
                            self.csc_to_csr_values(&backward_result.dA.nzval, &self.A_csr_to_csc);
                        (dP_values, backward_result.dq, dA_values, backward_result.db)
                    };

                    Ok(ComputedGradients {
                        dP_values,
                        dq: dq_out,
                        dA_values,
                        db: db_out,
                        #[cfg(debug_assertions)]
                        debug_smoothing_x: backward_result.debug_smoothing_x,
                        #[cfg(debug_assertions)]
                        debug_smoothing_z: backward_result.debug_smoothing_z,
                        #[cfg(debug_assertions)]
                        debug_smoothing_s: backward_result.debug_smoothing_s,
                    })
                })
                .collect()
        });

        // Print backward pass summary if verbose
        if self.settings.verbose {
            let elapsed = start.elapsed();
            let per_problem = elapsed.as_secs_f64() / batch_size as f64;
            println!("─────────────────────────────────────────────────────────────────────────────────────────────");
            println!("◀ Backward pass: {} problems (cached)", batch_size);
            println!(
                "backward time = {:?} ({:.2}ms/problem)",
                elapsed,
                per_problem * 1000.0
            );
        }

        results
    }

    /// Compute gradients from explicitly provided problem data and solution.
    ///
    /// Unlike `backward()`, this does not require a prior `solve_batch_parallel()` call.
    /// Instead, all problem data (P, A, q, b) and the solution (x, z, s) are passed
    /// directly. The method equilibrates the data internally, then calls the adjoint
    /// KKT differentiation.
    ///
    /// This is the recommended API for PyTorch/JAX autograd where each forward pass
    /// saves its own (P, A, q, b, x, z, s) tensors and backward must use exactly
    /// those values — even when the same solver is reused across multiple forward
    /// calls in one autograd graph (e.g. MPC rollouts).
    ///
    /// # Arguments
    ///
    /// * `upstream_grads` - Upstream gradients (dx, ds, dz) per problem
    /// * `P_values_batch` - P matrix values in CSR order per problem (original space)
    /// * `A_values_batch` - A matrix values in CSR order per problem (original space)
    /// * `q_batch` - Linear cost vector per problem (original space)
    /// * `b_batch` - Constraint RHS per problem (original space)
    /// * `x_batch` - Primal solution per problem (original space)
    /// * `z_batch` - Dual solution per problem (original space)
    /// * `s_batch` - Slack solution per problem (original space)
    ///
    /// # Returns
    ///
    /// Vector of computed gradients (dP, dq, dA, db) for each problem.
    /// Gradients are returned in CSR order (same as input values).
    pub fn backward_with_data(
        &self,
        upstream_grads: &[UpstreamGradients<T>],
        P_values_batch: &[Vec<T>],
        A_values_batch: &[Vec<T>],
        q_batch: &[Vec<T>],
        b_batch: &[Vec<T>],
        x_batch: &[Vec<T>],
        z_batch: &[Vec<T>],
        s_batch: &[Vec<T>],
    ) -> Result<Vec<ComputedGradients<T>>, SolverError> {
        // Slack-only entry-point: pass empty z_x_batch.
        self.backward_with_data_and_z_x(
            upstream_grads,
            P_values_batch,
            A_values_batch,
            q_batch,
            b_batch,
            x_batch,
            z_batch,
            s_batch,
            &[],
        )
    }

    /// Stateless backward variant that also accepts the per-problem
    /// direct-x dual `z_x` (in user/original frame). Pass an empty slice
    /// for `z_x_batch` (or for each Vec) when the problem has no direct-x
    /// cones; otherwise each `z_x_batch[i]` must match the flat
    /// `Solution.z_x` length.
    #[allow(clippy::too_many_arguments)]
    pub fn backward_with_data_and_z_x(
        &self,
        upstream_grads: &[UpstreamGradients<T>],
        P_values_batch: &[Vec<T>],
        A_values_batch: &[Vec<T>],
        q_batch: &[Vec<T>],
        b_batch: &[Vec<T>],
        x_batch: &[Vec<T>],
        z_batch: &[Vec<T>],
        s_batch: &[Vec<T>],
        z_x_batch: &[Vec<T>],
    ) -> Result<Vec<ComputedGradients<T>>, SolverError> {
        if !self.grad_enabled {
            return Err(SolverError::BadInputData(
                "backward_with_data() requires enable_grad=true in CompiledSolver constructor",
            ));
        }

        let has_dir_cones = !self.dir_cones.is_empty();
        let xn_total: usize = self.dir_cones.iter().map(|xc| xc.indices().len()).sum();
        // If the solver has direct-x cones, z_x_batch must be supplied with
        // matching dimensions. (Empty slice means "no z_x" — only valid for
        // slack-only solvers.)
        if has_dir_cones && z_x_batch.is_empty() {
            return Err(SolverError::BadInputData(
                "backward_with_data() requires z_x_batch when the solver has direct-x cones",
            ));
        }

        let start = Instant::now();
        let batch_size = upstream_grads.len();
        let num_solvers = self.solver_pool.len();

        // Validate batch sizes
        if P_values_batch.len() != batch_size
            || A_values_batch.len() != batch_size
            || q_batch.len() != batch_size
            || b_batch.len() != batch_size
            || x_batch.len() != batch_size
            || z_batch.len() != batch_size
            || s_batch.len() != batch_size
            || (!z_x_batch.is_empty() && z_x_batch.len() != batch_size)
        {
            return Err(SolverError::BadInputData(
                "All batch arguments must have the same length",
            ));
        }
        if has_dir_cones {
            for zx in z_x_batch {
                if zx.len() != xn_total {
                    return Err(SolverError::BadInputData(
                        "z_x_batch[i] length must equal total direct-x dimension",
                    ));
                }
            }
        }

        let num_grad_states = self.grad_states.len();
        let diff_method = self.settings.ipm.diff_method;

        let results: Result<Vec<_>, _> = self.thread_pool.install(|| {
            upstream_grads
                .par_iter()
                .enumerate()
                .map(|(i, grads)| {
                    let solver_idx = i % num_solvers;
                    let mut solver = self.solver_pool[solver_idx].lock().unwrap();

                    let mut grad_state_guard = if num_grad_states > 0 {
                        Some(
                            self.grad_states[solver_idx % num_grad_states]
                                .lock()
                                .unwrap(),
                        )
                    } else {
                        None
                    };

                    // Convert CSR values to CSC order
                    let P_csc_values = self.csr_to_csc_values(
                        &P_values_batch[i],
                        &self.P_csr_to_csc,
                        self.P_csc_pattern.nnz(),
                    );
                    let A_csc_values = self.csr_to_csc_values(
                        &A_values_batch[i],
                        &self.A_csr_to_csc,
                        self.A_csc_pattern.nnz(),
                    );

                    // Reconstruct CSC matrices from pattern + values
                    let mut P = self.P_csc_pattern.clone();
                    P.nzval.copy_from_slice(&P_csc_values);
                    let mut A = self.A_csc_pattern.clone();
                    A.nzval.copy_from_slice(&A_csc_values);

                    // Save solver state that load_and_equilibrate will overwrite.
                    // The forward solve path caches equilibration (d, e, c, P, A, q, b)
                    // inside the solver; we must restore it after backward so that
                    // subsequent forward solves see the correct cached state.
                    let saved_equil_d = solver.data.equilibration.d.clone();
                    let saved_equil_dinv = solver.data.equilibration.dinv.clone();
                    let saved_equil_e = solver.data.equilibration.e.clone();
                    let saved_equil_einv = solver.data.equilibration.einv.clone();
                    let saved_equil_c = solver.data.equilibration.c;
                    let saved_P_nzval = solver.data.P.nzval.clone();
                    let saved_A_nzval = solver.data.A.nzval.clone();
                    let saved_q = solver.data.q.clone();
                    let saved_b = solver.data.b.clone();
                    let saved_mu = solver.info.mu;
                    let saved_smoothing_cached = solver.smoothing_cached;

                    // Load data, reset equilibration, re-equilibrate, and update KKT.
                    // Uses the solver's own method which handles Rust borrow rules.
                    // (The KKT update is unnecessary for backward but harmless.)
                    solver.load_and_equilibrate(&P, &q_batch[i], &A, &b_batch[i]);

                    // Set solution from provided data
                    solver.solution.x.copy_from_slice(&x_batch[i]);
                    solver.solution.z.copy_from_slice(&z_batch[i]);
                    solver.solution.s.copy_from_slice(&s_batch[i]);
                    if has_dir_cones {
                        // Restore direct-x dual z_x (user frame, post-unscale).
                        // backward_batch_with_dz_x reads `solver.variables.z_x`
                        // and converts it to the equilibrated frame internally.
                        if solver.variables.z_x.len() != xn_total {
                            solver.variables.z_x.resize(xn_total, T::zero());
                        }
                        solver.variables.z_x.copy_from_slice(&z_x_batch[i]);
                    }

                    // Compute mu in equilibrated space for diff_method resolution.
                    // The normal solve path stores mu in equilibrated coordinates,
                    // so resolve_diff_method expects equilibrated-space mu.
                    // s_eq = einv * s, z_eq = e * z / c
                    let degree: usize = solver.data.cones.iter().map(|c| c.degree()).sum();
                    let e = &solver.data.equilibration.e;
                    let einv = &solver.data.equilibration.einv;
                    let c = solver.data.equilibration.c;
                    let dot_sz_eq: T = s_batch[i]
                        .iter()
                        .zip(z_batch[i].iter())
                        .zip(einv.iter().zip(e.iter()))
                        .map(|((&si, &zi), (&ei_inv, &ei))| (si * ei_inv) * (zi * ei / c))
                        .fold(T::zero(), |a, b| a + b);
                    solver.info.mu = if degree > 0 {
                        dot_sz_eq / T::from(degree).unwrap()
                    } else {
                        dot_sz_eq
                    };

                    // No smoothing iterate from external data — regenerate
                    // if the user requested smoothed differentiation.
                    solver.smoothing_cached = false;
                    if diff_method == super::DiffMethod::Smoothed {
                        solver.refine_smoothing_for_backward();
                    }

                    // Compute gradients (forwards `dz_x` when present so
                    // direct-x dual gradients flow into dq/dP).
                    let backward_result = solver.backward_batch_with_dz_x(
                        &grads.dx,
                        &grads.ds,
                        &grads.dz,
                        &grads.dz_x,
                        grad_state_guard.as_deref_mut(),
                        diff_method,
                    )?;

                    // Restore solver state so subsequent forward solves are not corrupted.
                    solver.data.equilibration.d = saved_equil_d;
                    solver.data.equilibration.dinv = saved_equil_dinv;
                    solver.data.equilibration.e = saved_equil_e;
                    solver.data.equilibration.einv = saved_equil_einv;
                    solver.data.equilibration.c = saved_equil_c;
                    solver.data.P.nzval.copy_from_slice(&saved_P_nzval);
                    solver.data.A.nzval.copy_from_slice(&saved_A_nzval);
                    solver.data.q.copy_from_slice(&saved_q);
                    solver.data.b.copy_from_slice(&saved_b);
                    solver.info.mu = saved_mu;
                    solver.smoothing_cached = saved_smoothing_cached;

                    // Convert CSC gradients back to CSR order
                    let dP_values = self.csc_to_csr_values_symmetric_P(
                        &backward_result.dP.nzval,
                        &self.P_csr_to_csc,
                    );
                    let dA_values =
                        self.csc_to_csr_values(&backward_result.dA.nzval, &self.A_csr_to_csc);

                    Ok(ComputedGradients {
                        dP_values,
                        dq: backward_result.dq,
                        dA_values,
                        db: backward_result.db,
                        #[cfg(debug_assertions)]
                        debug_smoothing_x: backward_result.debug_smoothing_x,
                        #[cfg(debug_assertions)]
                        debug_smoothing_z: backward_result.debug_smoothing_z,
                        #[cfg(debug_assertions)]
                        debug_smoothing_s: backward_result.debug_smoothing_s,
                    })
                })
                .collect()
        });

        if self.settings.verbose {
            let elapsed = start.elapsed();
            let per_problem = elapsed.as_secs_f64() / batch_size as f64;
            println!("─────────────────────────────────────────────────────────────────────────────────────────────");
            println!("◀ Backward pass: {} problems (with_data)", batch_size);
            println!(
                "backward time = {:?} ({:.2}ms/problem)",
                elapsed,
                per_problem * 1000.0
            );
        }

        results
    }

    /// Check if there is a valid cache from a previous solve
    pub fn has_cache(&self) -> bool {
        !self.solutions_cache.read().unwrap().is_empty()
    }

    /// Get the size of the cached batch (0 if no cache)
    pub fn cache_size(&self) -> usize {
        self.solutions_cache.read().unwrap().len()
    }

    /// Get the number of threads configured
    pub fn num_threads(&self) -> usize {
        self.thread_pool.current_num_threads()
    }

    /// Get problem dimensions
    pub fn dims(&self) -> (usize, usize) {
        (self.n, self.m)
    }

    /// Get number of non-zeros in P
    pub fn nnz_P(&self) -> usize {
        self.P_col_indices.len()
    }

    /// Get number of non-zeros in A
    pub fn nnz_A(&self) -> usize {
        self.A_col_indices.len()
    }

    /// Total direct-x cone dimension (sum over all dir_cones). Zero for
    /// slack-only solvers.
    pub fn total_xcone_dim(&self) -> usize {
        self.dir_cones.iter().map(|xc| xc.indices().len()).sum()
    }
}
