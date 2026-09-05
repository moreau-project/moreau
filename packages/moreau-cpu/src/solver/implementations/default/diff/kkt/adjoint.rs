//\! Adjoint (cached) HSDE differentiation pipeline.
//\!
//\! Placeholder/runtime H-block builders, HSDE coefficient assembly and
//\! the cached adjoint solve that backs `GradState::solve_adjoint`.

use super::*;

/// Build placeholder H blocks for symbolic factorization.
/// These have the correct sparsity pattern but placeholder values.
/// IMPORTANT: Must match the sparsity of `get_cone_derivative_sparse(..., dual=true)` since
/// the backward pass always uses dual=true for DΠ computation.
pub(super) fn build_placeholder_H_blocks<T: FloatT>(
    cones: &[SupportedConeT<T>],
) -> Vec<ConeDerivativeBlock<T>> {
    cones
        .iter()
        .map(|cone| {
            let dim = cone.nvars();
            match cone {
                // Zero cone with dual=true returns Identity (diagonal with 1s)
                SupportedConeT::ZeroConeT(_) => ConeDerivativeBlock::Diagonal(vec![T::one(); dim]),
                SupportedConeT::NonnegativeConeT(_) => {
                    ConeDerivativeBlock::Diagonal(vec![T::one(); dim])
                }
                SupportedConeT::SecondOrderConeT(_) if dim > 4 => {
                    // Large SOC: sparse expansion (diagonal + rank-2)
                    ConeDerivativeBlock::SocSparse {
                        dim,
                        diag: vec![T::one(); dim],
                        v1: vec![T::one(); dim],
                        c1: T::one(),
                        v2: vec![T::one(); dim],
                        c2: T::one(),
                    }
                }
                SupportedConeT::SecondOrderConeT(_) => {
                    // Sparse expansion (diagonal + rank-2) for all SOC cones
                    ConeDerivativeBlock::SocSparse {
                        dim,
                        diag: vec![T::one(); dim],
                        v1: vec![T::one(); dim],
                        c1: T::one(),
                        v2: vec![T::one(); dim],
                        c2: T::one(),
                    }
                }
                SupportedConeT::ExponentialConeT() => {
                    // Exp cone has dense 3x3 derivative
                    ConeDerivativeBlock::Dense {
                        dim,
                        data: vec![T::one(); dim * dim],
                    }
                }
                SupportedConeT::PowerConeT(_) => {
                    // Power cone has dense 3x3 derivative
                    ConeDerivativeBlock::Dense {
                        dim,
                        data: vec![T::one(); dim * dim],
                    }
                }
                SupportedConeT::GenPowerConeT(alpha, dim2) => {
                    // GenPower cone: choose Dense vs sparse rank-3 expansion
                    // by the same `N_DENSE_GENPOW` threshold the *forward*
                    // cone uses. Cones routed through the dense forward
                    // KKT block must also have a Dense backward block.
                    use crate::solver::core::cones::N_DENSE_GENPOW;
                    let total = alpha.len() + *dim2;
                    if total <= N_DENSE_GENPOW {
                        ConeDerivativeBlock::Dense {
                            dim,
                            data: vec![T::one(); dim * dim],
                        }
                    } else {
                        ConeDerivativeBlock::GenPowerSparse {
                            dim,
                            diag: vec![T::one(); dim],
                            left1: vec![T::one(); dim],
                            right1: vec![T::one(); dim],
                            left2: vec![T::one(); dim],
                            right2: vec![T::one(); dim],
                            left3: vec![T::one(); dim],
                            c3: T::one(),
                        }
                    }
                }
                #[cfg(feature = "sdp")]
                SupportedConeT::PSDTriangleConeT(_) => {
                    // SDP cone not supported in batch mode, but provide dense pattern
                    ConeDerivativeBlock::Dense {
                        dim,
                        data: vec![T::one(); dim * dim],
                    }
                }
            }
        })
        .collect()
}

/// Build placeholder H_x blocks for direct-x cones. The shape must produce
/// the same nonzero pattern as the runtime block produced by
/// `get_cone_derivative_sparse_xcones`, so the cached symbolic factorization
/// is a superset of the runtime numeric pattern. All-1 placeholder values
/// guarantee the augmented system builder takes the all-slot branch (no
/// zero-skipping) regardless of which `ConeDerivativeBlock` variant.
pub(super) fn build_placeholder_H_x_blocks<T: FloatT>(
    dir_cones: &[crate::solver::core::cones::SupportedXConeT],
) -> Vec<ConeDerivativeBlock<T>> {
    use crate::solver::core::cones::SupportedXConeT;
    dir_cones
        .iter()
        .map(|xc| {
            let dim = xc.indices().len();
            match xc {
                SupportedXConeT::NonnegativeXConeT(_) => {
                    ConeDerivativeBlock::Diagonal(vec![T::one(); dim])
                }
                SupportedXConeT::SecondOrderXConeT(_) => {
                    // Direct-x SOC uses the rank-2 sparse expansion in the
                    // augmented KKT (mirrors slack `SocSparse`). Placeholder
                    // values are arbitrary — they fix the symbolic sparsity
                    // pattern; real values come from
                    // `derivative_cone_sparse(u_x, SecondOrderConeT(dim), dual)`.
                    ConeDerivativeBlock::SocSparse {
                        dim,
                        diag: vec![T::one(); dim],
                        v1: vec![T::one(); dim],
                        c1: T::one(),
                        v2: vec![T::one(); dim],
                        c2: T::one(),
                    }
                }
                SupportedXConeT::ExponentialXConeT(_) => {
                    // Exp cone is 3D dense — same shape as the slack Dense block.
                    ConeDerivativeBlock::Dense {
                        dim: 3,
                        data: vec![T::one(); 9],
                    }
                }
                SupportedXConeT::PowerXConeT(_, _) => {
                    // Power cone is 3D dense — same shape as the slack Dense block.
                    ConeDerivativeBlock::Dense {
                        dim: 3,
                        data: vec![T::one(); 9],
                    }
                }
                SupportedXConeT::GenPowerXConeT(_, alphas, dim2) => {
                    // Direct-x GenPowerCone uses the rank-3 sparse expansion
                    // (mirrors slack `GenPowerSparse`). The forward (1,1)
                    // block uses μ·H_primal; the BACKWARD projection
                    // Jacobian factors as diag + 3 rank-1 outer products.
                    // Placeholder values are arbitrary — they only fix the
                    // symbolic sparsity; real values come from
                    // `derivative_cone_sparse(u_x, GenPowerConeT(...), dual)`.
                    let d = alphas.len() + dim2;
                    ConeDerivativeBlock::GenPowerSparse {
                        dim: d,
                        diag: vec![T::one(); d],
                        left1: vec![T::one(); d],
                        right1: vec![T::one(); d],
                        left2: vec![T::one(); d],
                        right2: vec![T::one(); d],
                        left3: vec![T::one(); d],
                        c3: T::one(),
                    }
                }
                #[cfg(feature = "sdp")]
                SupportedXConeT::PSDTriangleXConeT(_, k) => {
                    let svec_dim = k * (k + 1) / 2;
                    ConeDerivativeBlock::Dense {
                        dim: svec_dim,
                        data: vec![T::one(); svec_dim * svec_dim],
                    }
                }
            }
        })
        .collect()
}

/// Resolve DiffMethod::Auto and compute cone derivative blocks H.
///
/// For Exact: H = DΠ_{K*}(u), the Jacobian of the dual cone projection at u = z - s.
/// For Smoothed: H = (I + μ ∇²φ*(z))⁻¹, computed from the central-path iterate.
pub(super) fn compute_H_blocks<T: FloatT>(
    s: &[T],
    z: &[T],
    cones: &[SupportedConeT<T>],
    diff_method: DiffMethod,
    mu: T,
) -> Vec<ConeDerivativeBlock<T>> {
    let resolved = super::super::resolve_diff_method(diff_method, cones, mu);

    match resolved {
        DiffMethod::Exact => {
            let u: Vec<T> = z.iter().zip(s.iter()).map(|(&zi, &si)| zi - si).collect();
            get_cone_derivative_sparse(&u, cones, true)
        }
        DiffMethod::Smoothed => get_central_path_derivative_sparse(s, z, cones, mu),
        DiffMethod::Auto => unreachable!(),
    }
}

/// Compute the projection-Jacobian blocks for direct-x cones, dispatching
/// on `diff_method`.
///
/// Mirrors [`compute_H_blocks`] but operates on the direct-x partitioning
/// `(x[J], z_x_eq)` per cone. The slack-equivalent SupportedConeT for each
/// `SupportedXConeT` provides the actual cone-specific Jacobian formulas;
/// direct-x just routes the gathered (s_x, z_x) into them. `diff_method`
/// resolution uses the *slack* cone list because `Auto` already factors in
/// the global cone mix; direct-x cones don't change the auto-pick.
pub(super) fn compute_H_x_blocks<T: FloatT>(
    x: &[T],
    z_x_eq: &[T],
    xcone_indices: &[Vec<usize>],
    dir_cones: &[crate::solver::core::cones::SupportedXConeT],
    slack_cones: &[SupportedConeT<T>],
    diff_method: DiffMethod,
    mu: T,
) -> Vec<ConeDerivativeBlock<T>> {
    let resolved = super::super::resolve_diff_method(diff_method, slack_cones, mu);
    let xn: usize = xcone_indices.iter().map(|ix| ix.len()).sum();
    match resolved {
        DiffMethod::Exact => {
            let mut u_x = Vec::with_capacity(xn);
            let mut off = 0usize;
            for ix in xcone_indices {
                for (k, &i) in ix.iter().enumerate() {
                    u_x.push(z_x_eq[off + k] - x[i]);
                }
                off += ix.len();
            }
            super::super::cones::get_cone_derivative_sparse_xcones(&u_x, dir_cones, true)
        }
        DiffMethod::Smoothed => {
            // Gather s_x = x[J] alongside z_x_eq; central_path_derivative_cone
            // only inspects z_x in the implementations that have one (LP
            // cones), but the symmetric-cone/asymmetric paths take both.
            let mut s_x = Vec::with_capacity(xn);
            let mut off = 0usize;
            for ix in xcone_indices {
                for &i in ix {
                    s_x.push(x[i]);
                }
                off += ix.len();
            }
            super::super::cones::get_central_path_derivative_sparse_xcones(
                &s_x, z_x_eq, dir_cones, mu,
            )
        }
        DiffMethod::Auto => unreachable!(),
    }
}

/// Compute the coefficients c1, c2, c3 for the last row of the HSDE Jacobian.
///
/// c1[i] = -(2/τ)*Px[i] - q[i]
/// c2[i] = -b[i]
/// c3 = x'Px / τ²
pub(super) fn compute_hsde_coefficients<T: FloatT>(
    P: &CscMatrix<T>,
    q: &[T],
    b: &[T],
    x: &[T],
    tau: T,
    n: usize,
    _m: usize,
) -> (Vec<T>, Vec<T>, T) {
    let two: T = (2.0).as_T();
    let mut c1 = vec![T::zero(); n];
    let mut Px = vec![T::zero(); n];
    P.sym_up().symv(&mut Px, x, T::one(), T::zero());
    for i in 0..n {
        c1[i] = -(two / tau) * Px[i] - q[i];
    }

    let c2: Vec<T> = b.iter().map(|&bi| -bi).collect();

    let mut xPx = T::zero();
    for i in 0..n {
        xPx += x[i] * Px[i];
    }
    let c3 = xPx / (tau * tau);

    (c1, c2, c3)
}

/// Build the adjoint RHS for the augmented HSDE system.
///
/// `dz_x_bar_eq` is the equilibrated upstream gradient on the direct-x dual;
/// it enters at both the `du_x` slot (positive) and the corresponding
/// primal-x positions (negative), since internally `z_x = du_x − x[J]`.
/// `xcone_indices_flat` lists the gathered x indices in the same order as
/// `dz_x_bar_eq` and `z_x_eq`. Pass empty slices for the slack-only case.
pub(super) fn build_adjoint_rhs_full<T: FloatT>(
    dx_bar: &[T],
    dz_bar: &[T],
    ds_bar: &[T],
    dz_x_bar_eq: &[T],
    x: &[T],
    s: &[T],
    z: &[T],
    z_x_eq: &[T],
    xcone_indices_flat: &[usize],
    n: usize,
    m: usize,
    xn: usize,
) -> Vec<T> {
    debug_assert_eq!(
        dz_x_bar_eq.len(),
        xn,
        "dz_x_bar_eq length ({}) != xn ({})",
        dz_x_bar_eq.len(),
        xn
    );
    debug_assert_eq!(
        xcone_indices_flat.len(),
        xn,
        "xcone_indices_flat length ({}) != xn ({})",
        xcone_indices_flat.len(),
        xn
    );
    debug_assert_eq!(
        z_x_eq.len(),
        xn,
        "z_x_eq length ({}) != xn ({})",
        z_x_eq.len(),
        xn
    );

    let dim = n + 2 * m + xn + 1;
    let mut rhs_bar = vec![T::zero(); dim];

    for i in 0..n {
        rhs_bar[i] = dx_bar[i];
    }
    for i in 0..m {
        rhs_bar[n + i] = dz_bar[i] + ds_bar[i];
    }
    for i in 0..m {
        rhs_bar[n + m + i] = -ds_bar[i];
    }
    // Direct-x slot. The augmented variable `du_x` in this codebase encodes
    // `du_x = z_x − x[J]` (the Moreau-decomposition input to the dual cone
    // projection, mirroring slack's `du = z − s`). So
    //   z_x_internal = du_x + x[J]  (at τ=1)
    // and an upstream gradient on z_x flows positively to BOTH the `du_x`
    // slot and the corresponding primal-x slot.
    for k in 0..xn {
        rhs_bar[n + 2 * m + k] = dz_x_bar_eq[k];
        let xj = xcone_indices_flat[k];
        rhs_bar[xj] += dz_x_bar_eq[k];
    }
    let mut dt_bar = T::zero();
    for i in 0..n {
        dt_bar -= dx_bar[i] * x[i];
    }
    for i in 0..m {
        dt_bar -= dz_bar[i] * z[i];
        dt_bar -= ds_bar[i] * s[i];
    }
    for k in 0..xn {
        dt_bar -= dz_x_bar_eq[k] * z_x_eq[k];
    }
    rhs_bar[dim - 1] = dt_bar;

    rhs_bar
}

pub(super) fn extract_hsde_gradients_full<T: FloatT>(
    full_sol: &[T],
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    x: &[T],
    z: &[T],
    tau: T,
    n: usize,
    m: usize,
    xn: usize,
) -> BackwardResult<T> {
    // base_j_dim layout: [x; w; du_slack; du_x; τ]; the τ slot is index n+2m+xn.
    let lam1 = &full_sol[..n];
    let lam2 = &full_sol[n..n + m];
    let lam4 = full_sol[n + 2 * m + xn];

    let mut dq_bar = vec![T::zero(); n];
    for i in 0..n {
        dq_bar[i] = -tau * lam1[i] + lam4 * x[i];
    }

    let mut db_bar = vec![T::zero(); m];
    for i in 0..m {
        db_bar[i] = tau * lam2[i] + lam4 * z[i];
    }

    let dP_bar = compute_gradient_P_hsde(P, lam1, lam4, x, tau);
    let dA_bar = compute_gradient_A_hsde(A, lam1, lam2, x, z, tau);

    BackwardResult {
        dP: dP_bar,
        dq: dq_bar,
        dA: dA_bar,
        db: db_bar,
        #[cfg(debug_assertions)]
        debug_smoothing_x: vec![],
        #[cfg(debug_assertions)]
        debug_smoothing_z: vec![],
        #[cfg(debug_assertions)]
        debug_smoothing_s: vec![],
    }
}

/// Solve adjoint for HSDE using cached factorization. Handles both slack-only
/// (`state.dir_cones` empty, `z_x_eq` and `dz_x_bar_eq` empty) and direct-x
/// augmented systems.
#[allow(clippy::too_many_arguments)]
pub(super) fn solve_adjoint_hsde_cached<T: FloatT>(
    state: &mut GradState<T>,
    P: &CscMatrix<T>,
    q: &[T],
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    x: &[T],
    s: &[T],
    z: &[T],
    z_x_eq: &[T],
    tau: T,
    dx_bar: &[T],
    dz_bar: &[T],
    ds_bar: &[T],
    dz_x_bar_eq: &[T],
    diff_method: DiffMethod,
    mu: T,
) -> BackwardResult<T> {
    let n = state.n;
    let m = state.m;
    let xn: usize = state.xcone_indices.iter().map(|ix| ix.len()).sum();

    // Debug assertions to verify dimensions match
    debug_assert_eq!(x.len(), n, "x.len() != n");
    debug_assert_eq!(z.len(), m, "z.len() != m (z={}, m={})", z.len(), m);
    debug_assert_eq!(s.len(), m, "s.len() != m");
    debug_assert_eq!(A.nrows(), m, "A.nrows() != m");
    debug_assert_eq!(b.len(), m, "b.len() != m");
    debug_assert_eq!(
        z_x_eq.len(),
        xn,
        "z_x_eq length ({}) does not match cached xn ({})",
        z_x_eq.len(),
        xn
    );
    let cones_dim: usize = cones.iter().map(|c| c.nvars()).sum();
    debug_assert_eq!(cones_dim, m, "cones dim != m");

    let H_blocks = compute_H_blocks(s, z, cones, diff_method, mu);
    let H_x_blocks = if state.dir_cones.is_empty() {
        Vec::new()
    } else {
        compute_H_x_blocks(
            x,
            z_x_eq,
            &state.xcone_indices,
            &state.dir_cones,
            cones,
            diff_method,
            mu,
        )
    };
    let (c1, c2, c3) = compute_hsde_coefficients(P, q, b, x, tau, n, m);

    let xcone_indices_flat: Vec<usize> = state
        .xcone_indices
        .iter()
        .flat_map(|ix| ix.iter().copied())
        .collect();
    let rhs_bar = build_adjoint_rhs_full(
        dx_bar,
        dz_bar,
        ds_bar,
        dz_x_bar_eq,
        x,
        s,
        z,
        z_x_eq,
        &xcone_indices_flat,
        n,
        m,
        xn,
    );

    // Build augmented system with actual values
    let (kkt, aug_rhs) = build_hsde_augmented_system_sparse_full(
        P,
        A,
        q,
        b,
        &H_blocks,
        cones,
        &state.xcone_indices,
        &H_x_blocks,
        &c1,
        &c2,
        c3,
        &rhs_bar,
        n,
        m,
        true,
    );

    // Verify sparsity matches between symbolic and numeric matrices
    let numeric_nnz = kkt.nnz();
    let stored_nnz = state.kkt.nnz();
    debug_assert_eq!(
        stored_nnz, numeric_nnz,
        "Sparsity mismatch: stored nnz={}, numeric nnz={}",
        stored_nnz, numeric_nnz
    );

    // Update stored KKT values and factorization
    let indices: Vec<usize> = (0..numeric_nnz).collect();
    state.factor.update_values(&indices, &kkt.nzval);
    state.kkt.nzval.copy_from_slice(&kkt.nzval);

    let ok = state.factor.refactor(&state.kkt);
    if !ok {
        panic!(
            "LDL refactor failed in backward pass. This typically indicates a numerically \
             singular or ill-conditioned KKT system. Check that the problem is well-posed."
        );
    }

    // Solve
    let mut full_sol = vec![T::zero(); aug_rhs.len()];
    let mut rhs_buf = aug_rhs;
    state.factor.solve(&state.kkt, &mut full_sol, &mut rhs_buf);

    extract_hsde_gradients_full(&full_sol, P, A, x, z, tau, n, m, xn)
}
