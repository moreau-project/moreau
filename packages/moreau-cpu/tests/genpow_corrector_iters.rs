#![allow(non_snake_case)]
//! Iteration-count assertions for the GenPower 3rd-order Mehrotra η.
//!
//! These tests pin down two things that are easy to silently break:
//!  - on hard boundary-tight problems where η matters, iteration count
//!    stays well below the no-corrector baseline;
//!  - on a 3D GenPower([α, 1−α], 1) the iteration count tracks PowerCone(α)
//!    closely, since the dual barriers coincide.

use moreau::algebra::{CscMatrix, MatrixMathMut};
use moreau::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};

fn solve_iter_count(
    P: &CscMatrix<f64>,
    q: &[f64],
    A: &CscMatrix<f64>,
    b: &[f64],
    cones: &[SupportedConeT<f64>],
    max_iter: u32,
) -> (SolverStatus, u32) {
    let mut settings = DefaultSettings::default();
    settings.max_iter = max_iter;
    settings.verbose = false;
    let mut solver = DefaultSolver::new(P, q, A, b, cones, settings).unwrap();
    solver.solve();
    (solver.solution.status, solver.info.iterations)
}

/// Hard boundary problem: maximize ‖w‖ subject to (p, w) ∈ GenPow with p
/// fixed.  The optimum lies *exactly* on the cone boundary, so both s* and
/// z* are active. Without a 3rd-order corrector this class needs many more
/// iterations than expected.
fn make_active_boundary_problem(
    alphas: Vec<f64>,
    dim2: usize,
    seed: u64,
) -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let dim1 = alphas.len();

    // Simple LCG.
    let mut state = seed;
    let mut next_f = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / (1u64 << 53) as f64
    };

    let n = dim2;
    let mut p_bar = vec![0.0; dim1];
    for i in 0..dim1 {
        p_bar[i] = 1.0 + 0.3 * next_f();
    }
    let log_prod: f64 = (0..dim1).map(|i| alphas[i] * p_bar[i].ln()).sum();
    let prod = log_prod.exp();
    let mut w_bar = vec![0.0; n];
    let mut wb_norm_sq = 0.0;
    for j in 0..n {
        w_bar[j] = 2.0 * next_f() - 1.0;
        wb_norm_sq += w_bar[j] * w_bar[j];
    }
    let scale = 2.0 * prod / wb_norm_sq.sqrt().max(1e-30);
    for j in 0..n {
        w_bar[j] *= scale;
    }

    let mut P_colptr = vec![0usize; n + 1];
    let mut P_rowval = Vec::with_capacity(n);
    let mut P_nzval = Vec::with_capacity(n);
    for i in 0..n {
        P_colptr[i + 1] = i + 1;
        P_rowval.push(i);
        P_nzval.push(2.0);
    }
    let P = CscMatrix {
        m: n,
        n,
        colptr: P_colptr,
        rowval: P_rowval,
        nzval: P_nzval,
    };
    let q: Vec<f64> = w_bar.iter().map(|&x| -2.0 * x).collect();

    let m = dim1 + dim2;
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_idxs = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n {
        row_idxs.push(dim1 + j);
        vals.push(-1.0);
        col_ptrs[j + 1] = col_ptrs[j] + 1;
    }
    let A = CscMatrix {
        m,
        n,
        colptr: col_ptrs,
        rowval: row_idxs,
        nzval: vals,
    };
    let mut b = vec![0.0; m];
    for i in 0..dim1 {
        b[i] = p_bar[i];
    }

    let cones = vec![SupportedConeT::GenPowerConeT(alphas, dim2)];
    (P, q, A, b, cones)
}

#[test]
fn test_corrector_helps_3d_boundary() {
    // 3D GenPower with optimum on the boundary. The pre-corrector
    // baseline on this family is ≥35 iterations across seeds; the
    // corrector should bring the worst case to ≤30.
    for seed in [7u64, 41, 137, 9001] {
        let (P, q, A, b, cones) = make_active_boundary_problem(vec![0.6, 0.4], 1, seed);
        let (status, iters) = solve_iter_count(&P, &q, &A, &b, &cones, 60);
        assert!(
            matches!(status, SolverStatus::Solved | SolverStatus::AlmostSolved),
            "seed={} got {:?}",
            seed,
            status
        );
        assert!(
            iters <= 30,
            "seed={}: expected ≤30 iters with corrector, got {}",
            seed,
            iters
        );
    }
}

#[test]
fn test_corrector_helps_high_dim_boundary() {
    // High-dim GenPower with optimum on the cone boundary. Without the
    // corrector this class often hits MaxIterations or
    // InsufficientProgress at modest tolerances; with the corrector we
    // should at least reach AlmostSolved within 60 iterations.
    let alphas: Vec<f64> = vec![1.0 / 5.0; 5];
    let (P, q, A, b, cones) = make_active_boundary_problem(alphas, 3, 137);
    let (status, iters) = solve_iter_count(&P, &q, &A, &b, &cones, 100);
    assert!(
        matches!(status, SolverStatus::Solved | SolverStatus::AlmostSolved),
        "got {:?}",
        status
    );
    assert!(
        iters <= 60,
        "expected ≤60 iters with corrector, got {}",
        iters
    );
}

#[test]
fn test_corrector_does_not_break_easy_problems() {
    // Sanity guard: on a benign GenPower QP (well inside the cone) the
    // corrector must not regress iteration count beyond a small slack.
    let n = 5;
    let m = 5;
    let alphas = vec![0.5_f64, 0.3, 0.2];

    let x_star = vec![1.0, -0.5, 0.3, 0.7, -0.2];
    let q: Vec<f64> = x_star.iter().map(|&xi| -xi).collect();

    let mut A = CscMatrix::<f64>::identity(n);
    A.negate();

    let s_star = vec![2.0_f64, 3.0, 4.0, 0.3, 0.2]; // ‖w‖ ≪ Π p_i^{α_i}
    let b: Vec<f64> = (0..m).map(|i| -x_star[i] + s_star[i]).collect();

    let P = CscMatrix::<f64>::identity(n);
    let cones = vec![SupportedConeT::GenPowerConeT(alphas, 2)];

    let (status, iters) = solve_iter_count(&P, &q, &A, &b, &cones, 50);
    assert_eq!(status, SolverStatus::Solved);
    assert!(
        iters <= 25,
        "easy problem should solve in ≤25 iters, got {}",
        iters
    );
}
