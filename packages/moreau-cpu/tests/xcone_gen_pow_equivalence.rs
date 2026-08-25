#![allow(non_snake_case)]
//! Integration tests for GenPowerCone direct-x.
//!
//! GenPowerCone is asymmetric (`F ≠ F*`) and variable-dimension. Slack
//! form uses a rank-3 sparse expansion of `μ·∇²F_dual(z)` in the (1,1)
//! KKT block; direct-x stores the natural primal-IPM augmentation
//! `μ·∇²F_primal(x)` as a *dense* dim×dim block. The cone overrides
//! `direct_x_is_sparse_expandable` and `direct_x_Hs_is_diagonal` to
//! `false` to opt out of the slack expansion.
//!
//! `H_primal(x)` is computed via Legendre conjugacy:
//! `∇²F_primal(x) = (∇²F_dual(z_eff))⁻¹` where `z_eff = -∇F_primal(x)`.

use moreau::{algebra::*, solver::*};

fn build_simple_genpow_qp(n: usize) -> (CscMatrix<f64>, Vec<f64>) {
    // min 0.5 ||x - target||^2 s.t. (p, w) ∈ GenPow(α=[0.4, 0.6], dim2=2).
    // Target [2, 3, 1, 1] satisfies 2^0.4 * 3^0.6 ≈ 2.55 >= ||(1,1)|| ≈ 1.41.
    // Fully interior.
    let P = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![1.0; n]);
    let q = vec![-2.0, -3.0, -1.0, -1.0];
    (P, q)
}

#[test]
fn test_gen_pow_direct_x_scaffolding_does_not_crash() {
    let n = 4usize;
    let dim1 = 2usize;
    let dim2 = 2usize;
    let alphas = vec![0.4, 0.6];
    let (P, q) = build_simple_genpow_qp(n);

    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let x_cones = vec![SupportedXConeT::GenPowerXConeT(
        (0..n).collect(),
        alphas.clone(),
        dim2,
    )];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &x_cones, settings)
            .expect("GenPowerCone direct-x construction must succeed");
    solver.solve();

    // Whatever the final status, the iterate must be primal-feasible.
    let x = &solver.solution.x;
    for i in 0..dim1 {
        assert!(
            x[i] > -1e-6,
            "x[{}] (p coord) must remain >= 0, got {}",
            i,
            x[i]
        );
    }
}

#[test]
fn test_gen_pow_direct_x_solves_simple_problem() {
    let n = 4usize;
    let dim2 = 2usize;
    let alphas = vec![0.4, 0.6];
    let (P, q) = build_simple_genpow_qp(n);

    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let x_cones = vec![SupportedXConeT::GenPowerXConeT(
        (0..n).collect(),
        alphas,
        dim2,
    )];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &x_cones, settings)
            .unwrap();
    solver.solve();
    assert!(
        matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "expected Solved/AlmostSolved, got {:?}",
        solver.solution.status,
    );
}

/// Stress test: a higher-dim GenPowerCone exercises the rank-3 sparse
/// expansion path (without dense (1,1) block bloat). dim=20 here would
/// take 400 dense entries vs ~63 with rank-3 expansion.
#[test]
fn test_gen_pow_direct_x_high_dim_solves() {
    let dim1 = 4usize;
    let dim2 = 16usize;
    let n = dim1 + dim2;
    let alphas: Vec<f64> = (0..dim1).map(|i| (i + 1) as f64).collect();
    let asum: f64 = alphas.iter().sum();
    let alphas: Vec<f64> = alphas.iter().map(|a| a / asum).collect();

    let P = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![1.0; n]);
    let mut q = vec![0.0; n];
    for i in 0..dim1 {
        q[i] = -2.0_f64;
    }
    q[dim1] = -1.0;

    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let x_cones = vec![SupportedXConeT::GenPowerXConeT(
        (0..n).collect(),
        alphas,
        dim2,
    )];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    let mut solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &x_cones, settings)
            .unwrap();
    solver.solve();
    assert!(
        matches!(
            solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "high-dim GenPow direct-x failed: {:?}",
        solver.solution.status,
    );
}

#[test]
fn test_gen_pow_direct_x_matches_slack() {
    // Same problem solved via slack form. Both should give the same primal x.
    let n = 4usize;
    let dim2 = 2usize;
    let alphas = vec![0.4, 0.6];
    let (P, q) = build_simple_genpow_qp(n);

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    // Slack form: -x + s = 0, s ∈ GenPower(alphas, dim2).
    let A_slack = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![-1.0; n]);
    let b_slack = vec![0.0; n];
    let cones_slack = vec![GenPowerConeT(alphas.clone(), dim2)];
    let mut s_solver =
        DefaultSolver::new(&P, &q, &A_slack, &b_slack, &cones_slack, settings.clone()).unwrap();
    s_solver.solve();
    assert!(
        matches!(
            s_solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "slack form failed to solve: {:?}",
        s_solver.solution.status,
    );

    // Direct-x form.
    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let x_cones = vec![SupportedXConeT::GenPowerXConeT(
        (0..n).collect(),
        alphas,
        dim2,
    )];
    let mut d_solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &x_cones, settings)
            .unwrap();
    d_solver.solve();
    assert!(
        matches!(
            d_solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "direct-x form failed: {:?}",
        d_solver.solution.status,
    );

    for i in 0..n {
        assert!(
            (s_solver.solution.x[i] - d_solver.solution.x[i]).abs() < 1e-3,
            "x[{}] disagrees: slack={}, direct-x={}",
            i,
            s_solver.solution.x[i],
            d_solver.solution.x[i],
        );
    }
}

/// Parity test for the direct-x GenPow rank-9 PD-scaling sparse expansion
/// (`DirectXSparseMapGenPow.pd_axes`). Cone dim > `N_DENSE_GENPOW` (=64),
/// so the cone takes the *sparse* path: `Hs = μ·H_primal(x) + Σ pd_signs[k]·
/// pd_coefs[k]·v_k v_k'` enters the (1,1) block via 6 PD-axis expansion
/// columns (mirror of slack's `GenPowExpansionMap.pd_axes`).
///
/// Asserts the direct-x sparse PD path produces the same minimiser as the
/// slack form on the same problem. Pre-rank-9, the PD axes were computed
/// inside `try_compute_primal_pd_axes` but discarded by KKT assembly
/// (rank-3-only); only the rank-3 base entered the augmented (1,1) block.
#[test]
fn test_gen_pow_direct_x_sparse_pd_matches_slack() {
    // dim = 80 > N_DENSE_GENPOW (64), so this hits the sparse expansion.
    let dim1 = 16usize;
    let dim2 = 64usize;
    let n = dim1 + dim2;
    // Uniform alphas — interior, well-conditioned starting point.
    let alphas = vec![1.0_f64 / dim1 as f64; dim1];

    let P = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![1.0; n]);
    let mut q = vec![0.0_f64; n];
    for i in 0..dim1 {
        q[i] = -2.0;
    }
    for i in dim1..n {
        q[i] = -0.5;
    }

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    // Slack form.
    let A_slack = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![-1.0; n]);
    let b_slack = vec![0.0; n];
    let cones_slack = vec![GenPowerConeT(alphas.clone(), dim2)];
    let mut s_solver =
        DefaultSolver::new(&P, &q, &A_slack, &b_slack, &cones_slack, settings.clone()).unwrap();
    s_solver.solve();
    assert!(
        matches!(
            s_solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "slack form failed: {:?}",
        s_solver.solution.status,
    );

    // Direct-x form.
    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let x_cones = vec![SupportedXConeT::GenPowerXConeT(
        (0..n).collect(),
        alphas,
        dim2,
    )];
    let mut d_solver =
        DefaultSolver::new_with_xcones(&P, &q, &A_dx, &b_dx, &cones_dx, &x_cones, settings)
            .unwrap();
    d_solver.solve();
    assert!(
        matches!(
            d_solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "direct-x sparse form failed: {:?}",
        d_solver.solution.status,
    );

    for i in 0..n {
        assert!(
            (s_solver.solution.x[i] - d_solver.solution.x[i]).abs() < 1e-3,
            "x[{}] disagrees (sparse PD): slack={}, direct-x={}",
            i,
            s_solver.solution.x[i],
            d_solver.solution.x[i],
        );
    }
}

/// Dense ↔ sparse path parity for direct-x GenPow. The same problem is
/// solved twice: once with the default `N_DENSE_GENPOW=64` (cone dim 20
/// → dense path → `data.dense_hs`, fed via `direct_x_get_Hs`) and once
/// with `MOREAU_N_DENSE_GENPOW=0` (forces sparse path → rank-9 KKT
/// expansion via `DirectXSparseMap::GenPow`). Both must converge to the
/// same primal `x` within IPM tolerance.
///
/// Catches drift between the two implementations of `+H_primal(x)` in
/// the augmented (1,1) block: dense-path materialises the full Hs into
/// `dense_hs`, sparse-path encodes it as `D + p p' − q q' − r r'` plus
/// the rank-6 PD axes via expansion columns. Same math, different
/// machinery.
///
/// Marked `#[serial]` via single-threaded run requirement: the env var
/// is read at cone construction. Uses `cargo test -- --test-threads=1`
/// or the test framework's serial guard, but we're conservative and
/// restore the var even on panic (Rust's drop guarantees).
#[test]
fn test_gen_pow_direct_x_dense_sparse_parity() {
    // dim = 20, well below default N_DENSE_GENPOW (=64) → dense path.
    let dim1 = 4usize;
    let dim2 = 16usize;
    let n = dim1 + dim2;
    let alphas = vec![1.0_f64 / dim1 as f64; dim1];

    let P = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![1.0; n]);
    let mut q = vec![0.0_f64; n];
    for i in 0..dim1 {
        q[i] = -2.0;
    }
    for i in dim1..n {
        q[i] = -0.5;
    }
    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    /// RAII guard for `MOREAU_N_DENSE_GENPOW`: sets on construction,
    /// restores prior value (or unsets) on drop, so the env var leaves
    /// no residue across tests even if the body panics.
    struct EnvGuard {
        prev: Option<String>,
    }
    impl EnvGuard {
        fn set(value: &str) -> Self {
            let prev = std::env::var("MOREAU_N_DENSE_GENPOW").ok();
            std::env::set_var("MOREAU_N_DENSE_GENPOW", value);
            Self { prev }
        }
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("MOREAU_N_DENSE_GENPOW", v),
                None => std::env::remove_var("MOREAU_N_DENSE_GENPOW"),
            }
        }
    }

    // Default threshold → dense path.
    let x_cones_dense = vec![SupportedXConeT::GenPowerXConeT(
        (0..n).collect(),
        alphas.clone(),
        dim2,
    )];
    let mut dense_solver = DefaultSolver::new_with_xcones(
        &P,
        &q,
        &A_dx,
        &b_dx,
        &cones_dx,
        &x_cones_dense,
        settings.clone(),
    )
    .unwrap();
    dense_solver.solve();
    assert!(
        matches!(
            dense_solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "dense path failed: {:?}",
        dense_solver.solution.status,
    );

    // Threshold = 0 → forces sparse path even at small dim.
    let _guard = EnvGuard::set("0");
    let x_cones_sparse = vec![SupportedXConeT::GenPowerXConeT(
        (0..n).collect(),
        alphas,
        dim2,
    )];
    let mut sparse_solver = DefaultSolver::new_with_xcones(
        &P,
        &q,
        &A_dx,
        &b_dx,
        &cones_dx,
        &x_cones_sparse,
        settings.clone(),
    )
    .unwrap();
    sparse_solver.solve();
    assert!(
        matches!(
            sparse_solver.solution.status,
            SolverStatus::Solved | SolverStatus::AlmostSolved
        ),
        "sparse path failed: {:?}",
        sparse_solver.solution.status,
    );

    for i in 0..n {
        let d = dense_solver.solution.x[i];
        let s = sparse_solver.solution.x[i];
        assert!(
            (d - s).abs() < 1e-4,
            "x[{}] dense vs sparse disagree: dense={}, sparse={}",
            i,
            d,
            s,
        );
    }

    // Iter-count parity: both paths apply the same `+H_primal(x)` to
    // (1,1), so the IPM Newton trajectory must match. AMD permutes the
    // augmented matrix differently (dense has no expansion columns;
    // sparse adds 9), so a few iters of wiggle from rounding in
    // different orderings is acceptable. Anything larger says the
    // operators have actually drifted.
    let d_iter = dense_solver.solution.iterations as i32;
    let s_iter = sparse_solver.solution.iterations as i32;
    let diff = (d_iter - s_iter).abs();
    assert!(
        diff <= 2,
        "iter-count drift too large: dense={}, sparse={} (diff={}); same \
         operator should give same Newton trajectory within FP noise",
        d_iter,
        s_iter,
        diff,
    );

    // Near-boundary stress: pin p, push w outside the cone so the
    // optimum lies on the boundary. Same problem at dense and sparse
    // path — the IPM is harder here (PD scaling matters more), so any
    // operator drift between the paths shows up as a larger iter delta.
    let mut q_active = vec![0.0_f64; n];
    for j in 0..dim2 {
        // w̄ scaled so optimum sits on the GenPow boundary at α=uniform.
        q_active[dim1 + j] = -10.0;
    }

    let x_cones_active_dense = vec![SupportedXConeT::GenPowerXConeT(
        (0..n).collect(),
        vec![1.0_f64 / dim1 as f64; dim1],
        dim2,
    )];
    let mut dense_active = DefaultSolver::new_with_xcones(
        &P,
        &q_active,
        &A_dx,
        &b_dx,
        &cones_dx,
        &x_cones_active_dense,
        settings.clone(),
    )
    .unwrap();
    dense_active.solve();

    let _guard = EnvGuard::set("0");
    let x_cones_active_sparse = vec![SupportedXConeT::GenPowerXConeT(
        (0..n).collect(),
        vec![1.0_f64 / dim1 as f64; dim1],
        dim2,
    )];
    let mut sparse_active = DefaultSolver::new_with_xcones(
        &P,
        &q_active,
        &A_dx,
        &b_dx,
        &cones_dx,
        &x_cones_active_sparse,
        settings,
    )
    .unwrap();
    sparse_active.solve();

    let da = dense_active.solution.iterations as i32;
    let sa = sparse_active.solution.iterations as i32;
    let diff_a = (da - sa).abs();
    assert_eq!(
        format!("{:?}", dense_active.solution.status),
        format!("{:?}", sparse_active.solution.status),
        "active-boundary status mismatch: dense={:?}, sparse={:?}",
        dense_active.solution.status,
        sparse_active.solution.status,
    );
    assert!(
        diff_a <= 2,
        "active-boundary iter drift too large: dense={}, sparse={} \
         (diff={})",
        da,
        sa,
        diff_a,
    );
    for i in 0..n {
        let d = dense_active.solution.x[i];
        let s = sparse_active.solution.x[i];
        assert!(
            (d - s).abs() < 1e-3,
            "active x[{}] dense vs sparse disagree: dense={}, sparse={}",
            i,
            d,
            s,
        );
    }
}

#[test]
fn test_backward_direct_x_gen_pow_matches_slack() {
    // Backward parity for GenPower direct-x via the IFT-direct path.
    // Exercises the rank-3 sparse expansion in the augmented KKT.
    // Same QP solved two ways with arbitrary upstream `dx`, asserting
    // dq/dP agree.
    let n = 4usize;
    let dim2 = 2usize;
    let alphas = vec![0.4, 0.6];
    let (P, q) = build_simple_genpow_qp(n);
    let dx = vec![1.0, 1.0, 1.0, 1.0];

    let mut settings = DefaultSettings::default();
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = false;
    settings.verbose = false;
    settings.max_iter = 200;

    let slack_result = {
        let A = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![-1.0; n]);
        let b = vec![0.0; n];
        let cones = vec![GenPowerConeT(alphas.clone(), dim2)];
        let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings.clone()).unwrap();
        solver.solve();
        let dz = vec![0.0; n];
        let ds = vec![0.0; n];
        solver
            .backward_batch(&dx, &ds, &dz, None, DiffMethod::Exact)
            .unwrap()
    };

    let direct_result = {
        let A = CscMatrix::<f64>::zeros((0, n));
        let b: Vec<f64> = vec![];
        let cones: Vec<SupportedConeT<f64>> = vec![];
        let x_cones = vec![SupportedXConeT::GenPowerXConeT(
            (0..n).collect(),
            alphas.clone(),
            dim2,
        )];
        let mut solver =
            DefaultSolver::new_with_xcones(&P, &q, &A, &b, &cones, &x_cones, settings).unwrap();
        solver.solve();
        let dz: Vec<f64> = vec![];
        let ds: Vec<f64> = vec![];
        solver
            .backward_batch(&dx, &ds, &dz, None, DiffMethod::Exact)
            .unwrap()
    };

    for i in 0..n {
        assert!(
            (slack_result.dq[i] - direct_result.dq[i]).abs() < 1e-4,
            "dq[{}] disagree: slack={}, direct={}",
            i,
            slack_result.dq[i],
            direct_result.dq[i]
        );
    }
    assert_eq!(slack_result.dP.nzval.len(), direct_result.dP.nzval.len());
    for k in 0..slack_result.dP.nzval.len() {
        assert!(
            (slack_result.dP.nzval[k] - direct_result.dP.nzval[k]).abs() < 1e-4,
            "dP[{}] disagree: slack={}, direct={}",
            k,
            slack_result.dP.nzval[k],
            direct_result.dP.nzval[k]
        );
    }
}
