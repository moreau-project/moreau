#![allow(non_snake_case)]

//! Benchmark: slack vs direct-x for asymmetric cones (Exp, Power, GenPow).
//!
//! Same QP `min 0.5||x − target||²` solved two ways:
//!   slack:    -x + s = 0, s ∈ K        (n extra rows, n extra slack vars)
//!   direct-x: x ∈ K (no slack)         (cone augmentation in (1,1) block)
//!
//! Decides whether mirroring asymmetric direct-x onto CUDA is worth it.
//!
//! Usage:
//!   cargo run --release --example bench_direct_cone_asymmetric

use moreau::algebra::CscMatrix;
use moreau::solver::{
    DefaultSettings, DefaultSolver, ExponentialConeT, GenPowerConeT, IPSolver, PowerConeT,
    SolverStatus, SupportedConeT, SupportedXConeT,
};
use std::time::Instant;

const POW_ALPHA: f64 = 0.4;

fn diag_P(n: usize) -> CscMatrix<f64> {
    CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![1.0; n])
}

// Negated identity for the slack equality `-x + s = 0`.
fn neg_eye(n: usize) -> CscMatrix<f64> {
    CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), vec![-1.0; n])
}

fn settings() -> DefaultSettings<f64> {
    let mut s = DefaultSettings::default();
    s.verbose = false;
    s.ipm.presolve_enable = false;
    s.ipm.equilibrate_enable = false;
    s.ipm.direct_solve_method = "qdldl".to_string();
    s.max_iter = 200;
    s
}

fn time_solve_slack(
    P: &CscMatrix<f64>,
    q: &[f64],
    A: &CscMatrix<f64>,
    b: &[f64],
    cones: &[SupportedConeT<f64>],
) -> (f64, u32, SolverStatus) {
    let mut solver = DefaultSolver::new(P, q, A, b, cones, settings()).unwrap();
    let t = Instant::now();
    solver.solve();
    (
        t.elapsed().as_secs_f64() * 1e6,
        solver.solution.iterations,
        solver.solution.status,
    )
}

fn time_solve_dx(
    P: &CscMatrix<f64>,
    q: &[f64],
    dir_cones: &[SupportedXConeT],
) -> (f64, u32, SolverStatus) {
    let n = q.len();
    let A_dx = CscMatrix::<f64>::zeros((0, n));
    let b_dx: Vec<f64> = vec![];
    let cones_dx: Vec<SupportedConeT<f64>> = vec![];
    let mut solver =
        DefaultSolver::new_with_xcones(P, q, &A_dx, &b_dx, &cones_dx, dir_cones, settings())
            .unwrap();
    let t = Instant::now();
    solver.solve();
    (
        t.elapsed().as_secs_f64() * 1e6,
        solver.solution.iterations,
        solver.solution.status,
    )
}

// ─── Workload generators ─────────────────────────────────────────────

/// K stacked ExpCones. Target `(0, 1, 5)` per cone — well inside.
fn make_exp_problem(
    K: usize,
) -> (
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
    Vec<SupportedXConeT>,
) {
    let n = 3 * K;
    let P = diag_P(n);
    let mut q = vec![0.0; n];
    for k in 0..K {
        q[3 * k] = 0.0;
        q[3 * k + 1] = -1.0;
        q[3 * k + 2] = -5.0;
    }
    let cones_slack: Vec<_> = (0..K).map(|_| ExponentialConeT()).collect();
    let dir_cones: Vec<_> = (0..K)
        .map(|k| SupportedXConeT::ExponentialXConeT(vec![3 * k, 3 * k + 1, 3 * k + 2]))
        .collect();
    (P, q, cones_slack, dir_cones)
}

/// K stacked 3D PowerCones. Target `(2, 3, 1)` per cone (interior for α=0.4).
fn make_pow_problem(
    K: usize,
) -> (
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
    Vec<SupportedXConeT>,
) {
    let n = 3 * K;
    let P = diag_P(n);
    let mut q = vec![0.0; n];
    for k in 0..K {
        q[3 * k] = -2.0;
        q[3 * k + 1] = -3.0;
        q[3 * k + 2] = -1.0;
    }
    let cones_slack: Vec<_> = (0..K).map(|_| PowerConeT(POW_ALPHA)).collect();
    let dir_cones: Vec<_> = (0..K)
        .map(|k| SupportedXConeT::PowerXConeT(vec![3 * k, 3 * k + 1, 3 * k + 2], POW_ALPHA))
        .collect();
    (P, q, cones_slack, dir_cones)
}

/// One GenPowerCone with `dim1` p-coords (uniform alphas) and `dim2` w-coords.
/// Target: p_i = 2, w_j = 0.5 — well inside since 2^1 = 2 ≥ ‖w‖ = 0.5·√dim2 (small dim2)
/// and we keep dim2 small relative to dim1's product capacity.
fn make_genpow_problem(
    dim1: usize,
    dim2: usize,
) -> (
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
    Vec<SupportedXConeT>,
) {
    let n = dim1 + dim2;
    let P = diag_P(n);
    let alphas: Vec<f64> = vec![1.0 / (dim1 as f64); dim1];
    let mut q = vec![0.0; n];
    for i in 0..dim1 {
        q[i] = -2.0;
    }
    for j in 0..dim2 {
        q[dim1 + j] = -0.5;
    }
    let cones_slack = vec![GenPowerConeT(alphas.clone(), dim2)];
    let dir_cones = vec![SupportedXConeT::GenPowerXConeT(
        (0..n).collect(),
        alphas,
        dim2,
    )];
    (P, q, cones_slack, dir_cones)
}

// ─── Bench harness ───────────────────────────────────────────────────

fn bench_one(
    label: &str,
    n: usize,
    nnz_kkt_slack: usize,
    nnz_kkt_dx: usize,
    slack_runs: &[(f64, u32, SolverStatus)],
    dx_runs: &[(f64, u32, SolverStatus)],
) {
    let mut s_times: Vec<f64> = slack_runs.iter().map(|(t, _, _)| *t).collect();
    let mut d_times: Vec<f64> = dx_runs.iter().map(|(t, _, _)| *t).collect();
    s_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    d_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let s_med = s_times[s_times.len() / 2];
    let d_med = d_times[d_times.len() / 2];
    let s_iter = slack_runs[0].1;
    let d_iter = dx_runs[0].1;
    let s_st = format!("{:?}", slack_runs[0].2);
    let d_st = format!("{:?}", dx_runs[0].2);
    let speedup = s_med / d_med;
    println!(
        "  {label:<14} n={n:<5} kkt(s/dx)={nnz_kkt_slack:>6}/{nnz_kkt_dx:<6}  slack={s_med:>9.0}us(it={s_iter},{s_st})  dx={d_med:>9.0}us(it={d_iter},{d_st})  speedup={speedup:.2}x"
    );
}

fn run_bench(
    label: &str,
    n: usize,
    P: &CscMatrix<f64>,
    q: &[f64],
    cones_slack: &[SupportedConeT<f64>],
    dir_cones: &[SupportedXConeT],
    iters: usize,
) {
    let A_slack = neg_eye(n);
    let b_slack = vec![0.0; n];
    let nnz_kkt_slack = P.nnz() + A_slack.nnz() + n + n; // P upper + A + diag(P) + diag(neg-block)
                                                         // Direct-x KKT nnz is harder to estimate accurately (cone-dependent expansion);
                                                         // report P nnz as a lower bound for transparency.
    let nnz_kkt_dx = P.nnz();

    // Warmup
    let _ = time_solve_slack(P, q, &A_slack, &b_slack, cones_slack);
    let _ = time_solve_dx(P, q, dir_cones);

    let mut s_runs = Vec::with_capacity(iters);
    let mut d_runs = Vec::with_capacity(iters);
    for _ in 0..iters {
        s_runs.push(time_solve_slack(P, q, &A_slack, &b_slack, cones_slack));
        d_runs.push(time_solve_dx(P, q, dir_cones));
    }

    bench_one(label, n, nnz_kkt_slack, nnz_kkt_dx, &s_runs, &d_runs);
}

fn main() {
    println!("=== Asymmetric direct-x vs slack benchmark (CPU, qdldl) ===");
    println!("Problem: min 0.5||x − target||²  s.t.  x in K  (slack:  -x+s=0, s∈K)\n");

    let iters = 5;

    println!("ExpCone (3D, K stacked):");
    for &K in &[1usize, 10, 100, 1000] {
        let (P, q, cs, xs) = make_exp_problem(K);
        run_bench(&format!("Exp K={K}"), 3 * K, &P, &q, &cs, &xs, iters);
    }

    println!("\nPowerCone (3D, α=0.4, K stacked):");
    for &K in &[1usize, 10, 100, 1000] {
        let (P, q, cs, xs) = make_pow_problem(K);
        run_bench(&format!("Pow K={K}"), 3 * K, &P, &q, &cs, &xs, iters);
    }

    println!("\nGenPowerCone (uniform α, single cone, varying dim):");
    for &(dim1, dim2) in &[(2usize, 2usize), (4, 16), (8, 64), (16, 256), (32, 1024)] {
        let n = dim1 + dim2;
        let (P, q, cs, xs) = make_genpow_problem(dim1, dim2);
        run_bench(&format!("GenPow {dim1}/{dim2}"), n, &P, &q, &cs, &xs, iters);
    }

    println!("\n3D GenPowerCone stacked (mathematically same shape as Pow, different code path):");
    for &K in &[1usize, 10, 100, 1000] {
        let n = 3 * K;
        let alphas = vec![POW_ALPHA, 1.0 - POW_ALPHA];
        let dim2 = 1usize;
        let P = diag_P(n);
        let mut q = vec![0.0; n];
        for k in 0..K {
            q[3 * k] = -2.0;
            q[3 * k + 1] = -3.0;
            q[3 * k + 2] = -1.0;
        }
        let cones_slack: Vec<SupportedConeT<f64>> = (0..K)
            .map(|_| GenPowerConeT(alphas.clone(), dim2))
            .collect();
        let dir_cones: Vec<SupportedXConeT> = (0..K)
            .map(|k| {
                SupportedXConeT::GenPowerXConeT(
                    vec![3 * k, 3 * k + 1, 3 * k + 2],
                    alphas.clone(),
                    dim2,
                )
            })
            .collect();
        run_bench(
            &format!("GenPow3D K={K}"),
            n,
            &P,
            &q,
            &cones_slack,
            &dir_cones,
            iters,
        );
    }
}
