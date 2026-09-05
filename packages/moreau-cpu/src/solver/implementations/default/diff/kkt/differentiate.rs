//\! Forward and adjoint differentiation entry points (QP-eq and HSDE).

use super::*;

// ============================================================================
// Forward differentiation (QP with equality constraints only)
// ============================================================================

/// Forward differentiation for QP with only equality constraints.
///
/// The KKT system for equality-constrained QP is:
/// ```text
/// [P  A'] [x]   [q]
/// [A  0 ] [y] = [b]
/// ```
///
/// Differentiating:
/// ```text
/// [P  A'] [dx]   [dq + dP*x + dA'*y]
/// [A  0 ] [dy] = [db + dA*x       ]
/// ```
pub fn differentiate_qp_eq<T: FloatT>(
    P: &CscMatrix<T>,
    _q: &[T],
    A: &CscMatrix<T>,
    _b: &[T],
    x: &[T],
    z: &[T], // y in QP formulation
    dP: &CscMatrix<T>,
    dq: &[T],
    dA: &CscMatrix<T>,
    db: &[T],
) -> ForwardResult<T> {
    let n = x.len();
    let m = z.len();

    // Build RHS: [dq + dP*x + dA'*z; db + dA*x]
    let mut rhs = vec![T::zero(); n + m];

    // rhs[0:n] = dq + dP*x + dA'*z
    rhs[..n].copy_from_slice(dq);

    // Add dP*x
    let mut dPx = vec![T::zero(); n];
    dP.sym_up().symv(&mut dPx, x, T::one(), T::zero());
    for i in 0..n {
        rhs[i] += dPx[i];
    }

    // Add dA'*z
    let mut dAtz = vec![T::zero(); n];
    dA.t().gemv(&mut dAtz, z, T::one(), T::zero());
    for i in 0..n {
        rhs[i] += dAtz[i];
    }

    // rhs[n:n+m] = db - dA*x (note: NOT negated, unlike r1)
    rhs[n..n + m].copy_from_slice(db);
    let mut dAx = vec![T::zero(); m];
    dA.gemv(&mut dAx, x, T::one(), T::zero());
    for i in 0..m {
        rhs[n + i] -= dAx[i];
    }

    // Negate only r1 (first n elements)
    for i in 0..n {
        rhs[i] = -rhs[i];
    }

    // Build and solve KKT system
    let kkt = build_qp_kkt_matrix(P, A, REG_GENERAL.as_T());
    let sol = solve_linear_system(&kkt, &rhs);

    ForwardResult {
        dx: sol[..n].to_vec(),
        dz: sol[n..n + m].to_vec(),
        ds: vec![T::zero(); m], // s = 0 for equality constraints
    }
}

/// Debug helper to print H_blocks
fn debug_print_h_blocks<T: FloatT>(blocks: &[ConeDerivativeBlock<T>]) {
    if !crate::utils::debug::is_debug_mode() {
        return;
    }
    eprintln!("\n=== DEBUG: H_blocks ===");
    for (i, block) in blocks.iter().enumerate() {
        match block {
            ConeDerivativeBlock::Zero(dim) => eprintln!("H_block[{}]: Zero({})", i, dim),
            ConeDerivativeBlock::Diagonal(diag) => {
                eprintln!("H_block[{}]: Diagonal({:?})", i, diag)
            }
            ConeDerivativeBlock::Dense { dim, data } => {
                eprintln!("H_block[{}]: Dense {}x{}:", i, dim, dim);
                for row in 0..*dim {
                    eprint!("  ");
                    for col in 0..*dim {
                        eprint!("{:.6} ", data[row * dim + col]);
                    }
                    eprintln!();
                }
            }
            ConeDerivativeBlock::SocSparse {
                dim, diag, c1, c2, ..
            } => {
                eprintln!(
                    "H_block[{}]: SocSparse dim={}, c1={:.6}, c2={:.6}",
                    i, dim, c1, c2
                );
                eprintln!("  diag: {:?}", &diag[..std::cmp::min(8, diag.len())]);
            }
            ConeDerivativeBlock::GenPowerSparse { dim, .. } => {
                eprintln!("H_block[{}]: GenPowerSparse dim={}", i, dim);
            }
        }
    }
}

/// Debug helper to print a vector
fn debug_print_vec<T: FloatT>(name: &str, v: &[T], limit: usize) {
    if !crate::utils::debug::is_debug_mode() {
        return;
    }
    eprint!("  {} [{}]: ", name, v.len());
    for (_i, val) in v.iter().enumerate().take(limit) {
        eprint!("{:.8e} ", val.to_f64().unwrap());
    }
    if v.len() > limit {
        eprint!("...");
    }
    eprintln!();
}

/// Adjoint differentiation for QP with only equality constraints.
pub fn differentiate_adjoint_qp_eq<T: FloatT>(
    P: &CscMatrix<T>,
    _q: &[T],
    A: &CscMatrix<T>,
    _b: &[T],
    x: &[T],
    z: &[T],
    dx_bar: &[T],
    dz_bar: &[T],
    _ds_bar: &[T],
) -> BackwardResult<T> {
    let n = x.len();
    let m = z.len();

    debug_block! {
        eprintln!("\n=== CPU differentiate_adjoint_qp_eq ===");
        eprintln!("n = {}, m = {}", n, m);
        debug_print_vec("x", x, 8);
        debug_print_vec("z", z, 8);
        debug_print_vec("dx_bar", dx_bar, 8);
        debug_print_vec("dz_bar", dz_bar, 8);
    }

    // Build RHS for adjoint system: [dx_bar; dz_bar]
    let mut rhs = vec![T::zero(); n + m];
    rhs[..n].copy_from_slice(dx_bar);
    rhs[n..n + m].copy_from_slice(dz_bar);

    debug_block! {
        eprintln!("\nStep 1: Build adjoint RHS");
        debug_print_vec("rhs", &rhs, 8);
    }

    // Solve K^T * lambda = rhs
    // For symmetric KKT, K^T = K
    let kkt = build_qp_kkt_matrix(P, A, REG_GENERAL.as_T());

    debug_block! {
        eprintln!("\nStep 2: Build KKT matrix");
        eprintln!("  KKT dim = {} x {}, nnz = {}", kkt.nrows(), kkt.ncols(), kkt.nnz());
        debug_print_vec("P.nzval", &P.nzval, 8);
        debug_print_vec("A.nzval", &A.nzval, 8);
        debug_print_vec("KKT.nzval", &kkt.nzval, 16);
    }

    let lambda = solve_linear_system(&kkt, &rhs);

    debug_block! {
        eprintln!("\nStep 3: KKT solve");
        debug_print_vec("lambda (sol)", &lambda, 8);
    }

    let lam_x = &lambda[..n];
    let lam_z = &lambda[n..n + m];

    debug_block! {
        debug_print_vec("lam_x", lam_x, 8);
        debug_print_vec("lam_z", lam_z, 8);
    }

    // Compute gradients (following diffclarabel's _differentiate_adjoint_qp_eq)
    // dq_bar = -lam_x
    let dq_bar: Vec<T> = lam_x.iter().map(|&v| -v).collect();

    // db_bar = lam_z
    let db_bar = lam_z.to_vec();

    // dP_bar: only at nonzero positions of P
    // dP_bar[i,j] = -lam_x[i] * x[j] (for symmetric P, symmetrize)
    let dP_bar = compute_gradient_P(P, lam_x, x);

    // dA_bar: only at nonzero positions of A
    // dA_bar = -lam_z * x^T - lam_x * z^T (for the terms involving A)
    let dA_bar = compute_gradient_A(A, lam_x, lam_z, x, z);

    debug_block! {
        eprintln!("\nStep 4: Compute gradients");
        debug_print_vec("dq", &dq_bar, 8);
        debug_print_vec("db", &db_bar, 8);
        debug_print_vec("dP.nzval", &dP_bar.nzval, 8);
        debug_print_vec("dA.nzval", &dA_bar.nzval, 8);
        eprintln!("=== CPU backward complete ===\n");
    }

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

// ============================================================================
// Forward differentiation (HSDE - general cones)
// ============================================================================

/// Forward differentiation using HSDE formulation.
///
/// The augmented KKT system is:
/// ```text
/// [P    A'   0   q ] [dz_x]   [r1]
/// [A    I   -I  -b ] [w   ] = [r2]
/// [0    I   -H   0 ] [du  ]   [0 ]
/// [c1   c2   0  c3 ] [dt  ]   [r3]
/// ```
///
/// where:
/// - H = DΠ_{K*}(u), u = z - s
/// - w = H * du (auxiliary variable for sparsity)
/// - Recovery: dx = dz_x - dt*x, dz = w - dt*z, ds = w - du - dt*s
pub fn differentiate_hsde<T: FloatT>(
    P: &CscMatrix<T>,
    q: &[T],
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    x: &[T],
    s: &[T],
    z: &[T],
    tau: T,
    dP: &CscMatrix<T>,
    dq: &[T],
    dA: &CscMatrix<T>,
    db: &[T],
    diff_method: DiffMethod,
    mu: T,
) -> ForwardResult<T> {
    let n = x.len();
    let m = z.len();

    let H_blocks = compute_H_blocks(s, z, cones, diff_method, mu);

    // Compute dP*x once (reused in r1 and r3)
    let mut dPx = vec![T::zero(); n];
    dP.sym_up().symv(&mut dPx, x, T::one(), T::zero());

    // r1 = -(dP*x + dA'*z + dq*tau)
    let mut r1 = vec![T::zero(); n];
    for i in 0..n {
        r1[i] = -dPx[i];
    }
    let mut tmp = vec![T::zero(); n];
    dA.t().gemv(&mut tmp, z, T::one(), T::zero());
    for i in 0..n {
        r1[i] = r1[i] - tmp[i] - dq[i] * tau;
    }

    // r2 = -(dA*x - db*tau)
    let mut r2 = vec![T::zero(); m];
    dA.gemv(&mut r2, x, -T::one(), T::zero());
    for i in 0..m {
        r2[i] += db[i] * tau;
    }

    // r3 = -(- x'*dP*x/tau - dq'*x - db'*z)
    let mut xdPx = T::zero();
    for i in 0..n {
        xdPx += x[i] * dPx[i];
    }

    let mut dqx = T::zero();
    for i in 0..n {
        dqx += dq[i] * x[i];
    }

    let mut db_z = T::zero();
    for i in 0..m {
        db_z += db[i] * z[i];
    }

    let r3 = xdPx / tau + dqx + db_z;

    let (c1, c2, c3) = compute_hsde_coefficients(P, q, b, x, tau, n, m);

    // Build and solve the augmented system
    // K = [I J; J' -reg*I], rhs = [r1,r2,0,r3; 0]
    // Solution is [y; sol] where sol is what we want
    let (kkt, rhs) = build_hsde_kkt_system(
        P, q, b, A, &H_blocks, cones, &c1, &c2, c3, &r1, &r2, r3, n, m,
    );

    let full_sol = solve_linear_system(&kkt, &rhs);

    // Extract solution from second half of augmented solution
    // j_dim includes expansion variables; solution components are at original positions
    let p_exp = count_expansion_vars(&H_blocks);
    let j_dim = n + 2 * m + 1 + p_exp;
    let sol = &full_sol[j_dim..];
    let dz_x = &sol[..n];
    let w = &sol[n..n + m];
    let du = &sol[n + m..n + 2 * m];
    let dt = sol[n + 2 * m];

    // Recover dx, dz, ds
    // The system constraint in row block (2) is: w = H @ du
    // So:
    //   dx = dz_x - dt * x
    //   dz = w - dt * z = H @ du - dt * z
    //   ds = w - du - dt * s = (H - I) @ du - dt * s
    // This matches diffqcp's recovery if we identify du with their dz_m.
    let mut dx = vec![T::zero(); n];
    let mut dz = vec![T::zero(); m];
    let mut ds = vec![T::zero(); m];

    for i in 0..n {
        dx[i] = dz_x[i] - dt * x[i];
    }
    for i in 0..m {
        dz[i] = w[i] - dt * z[i];
        ds[i] = w[i] - du[i] - dt * s[i];
    }

    ForwardResult { dx, dz, ds }
}

/// Adjoint differentiation using HSDE formulation.
pub fn differentiate_adjoint_hsde<T: FloatT>(
    P: &CscMatrix<T>,
    q: &[T],
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    x: &[T],
    s: &[T],
    z: &[T],
    tau: T,
    dx_bar: &[T],
    dz_bar: &[T],
    ds_bar: &[T],
    diff_method: DiffMethod,
    mu: T,
) -> BackwardResult<T> {
    differentiate_adjoint_hsde_with_xcones(
        P,
        q,
        A,
        b,
        cones,
        &[],
        &[],
        x,
        s,
        z,
        &[],
        tau,
        dx_bar,
        dz_bar,
        ds_bar,
        &[],
        diff_method,
        mu,
    )
}

/// IFT-direct adjoint differentiation: HSDE Jacobian augmented with
/// direct-x cone-projection rows + a `du_x` column block carrying H_x.
///
/// `dir_cones` and `xcone_indices` describe each direct-x cone (their slack
/// equivalents and which positions of `x` are constrained). `z_x_eq` is
/// the equilibrated direct-x dual flat across all cones in the same order
/// as `dir_cones`.
///
/// When `dir_cones` is empty this reduces to the slack-only HSDE backward.
#[allow(clippy::too_many_arguments)]
pub fn differentiate_adjoint_hsde_with_xcones<T: FloatT>(
    P: &CscMatrix<T>,
    q: &[T],
    A: &CscMatrix<T>,
    b: &[T],
    cones: &[SupportedConeT<T>],
    dir_cones: &[crate::solver::core::cones::SupportedXConeT],
    xcone_indices: &[Vec<usize>],
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
    let n = x.len();
    let m = z.len();
    debug_assert_eq!(
        dir_cones.len(),
        xcone_indices.len(),
        "dir_cones and xcone_indices must have the same length"
    );

    let H_blocks = compute_H_blocks(s, z, cones, diff_method, mu);

    // Direct-x H_x blocks: gather x[J] per cone and call the same per-cone
    // projection-Jacobian kernel used for slack cones.
    let xn: usize = xcone_indices.iter().map(|ix| ix.len()).sum();
    debug_assert_eq!(
        z_x_eq.len(),
        xn,
        "z_x_eq length ({}) does not match total direct-x dim ({})",
        z_x_eq.len(),
        xn
    );
    let H_x_blocks = if dir_cones.is_empty() {
        Vec::new()
    } else {
        compute_H_x_blocks(x, z_x_eq, xcone_indices, dir_cones, cones, diff_method, mu)
    };

    debug_block! { debug_print_h_blocks(&H_blocks); }

    let (c1, c2, c3) = compute_hsde_coefficients(P, q, b, x, tau, n, m);

    let xcone_indices_flat: Vec<usize> = xcone_indices
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

    // Build and solve the augmented adjoint system (with direct-x rows/cols).
    let (kkt, aug_rhs) = build_hsde_augmented_system_sparse_full(
        P,
        A,
        q,
        b,
        &H_blocks,
        cones,
        xcone_indices,
        &H_x_blocks,
        &c1,
        &c2,
        c3,
        &rhs_bar,
        n,
        m,
        true,
    );

    let full_sol = solve_linear_system(&kkt, &aug_rhs);

    extract_hsde_gradients_full(&full_sol, P, A, x, z, tau, n, m, xn)
}
