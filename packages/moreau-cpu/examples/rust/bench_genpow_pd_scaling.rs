#![allow(non_snake_case)]

//! Benchmark: GenPower Mehrotra 3rd-order corrector on hard boundary problems.
//!
//! This bench drives both the primal and dual GenPower constraints toward
//! activity at optimum, which is where the 3rd-order η matters most. Two
//! problem classes are exercised:
//!
//!   1. `make_active_genpow` — forces the GenPow inequality to be active at
//!      optimum, so the optimal slack lies *exactly* on the cone boundary
//!      and the dual is correspondingly active.
//!
//!   2. `make_skewed_genpow` — uses heavily skewed α distributions
//!      ([0.97, 0.01, 0.01, ...]) so the cone is nearly degenerate; this
//!      stresses the asymmetric IPM where the primal and dual barriers
//!      have very different curvatures.
//!
//! Each row prints `(dim1, dim2, kind, status, iter, time_ms)`. Compare
//! against a `git stash`'d baseline to see the corrector's effect.
//!
//! Usage:
//!   cargo run --release --features sdp-openblas --example bench_genpow_pd_scaling

use moreau::algebra::{CscMatrix, MatrixMathMut};
use moreau::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus, SupportedConeT};
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn signed(&mut self) -> f64 {
        2.0 * self.uniform() - 1.0
    }
}

/// Active-constraint GenPower: maximize ‖w‖² subject to w lying in a
/// GenPow cone whose p side is bounded above and below. The optimum
/// rides the GenPow boundary, so both `s*` and `z*` are active.
///
///   maximize    Σ w_j²        i.e.  minimize −Σ w_j²
///   subject to  s = (p, w) ∈ GenPower(α, dim2)
///               p_i = p̄_i  (zero-cone)
///
/// Reformulated as a QP via slack variables since Moreau requires a PSD P.
/// We expose a "hardness" `tightness ∈ (0,1)` controlling how close the
/// initial point sits to the boundary along p.
fn make_active_genpow(
    alphas: Vec<f64>,
    dim2: usize,
    tightness: f64,
    seed: u64,
) -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let dim1 = alphas.len();
    let n_w = dim2;
    let n = n_w; // decision variable = w
    let mut rng = Rng::new(seed);

    // p̄_i target.
    let mut p_bar = vec![0.0; dim1];
    for i in 0..dim1 {
        p_bar[i] = 1.0 + 0.3 * rng.uniform();
    }
    let log_prod: f64 = (0..dim1).map(|i| alphas[i] * p_bar[i].ln()).sum();
    let prod = log_prod.exp();
    // Box-constrain ‖w‖ < tightness · prod via the cone constraint itself
    // (the optimizer will push to that boundary).

    // Primal: −Σ w_j²  → P = -2 I (PSD requirement violated). To stay
    // PSD-compliant we instead minimize Σ (w_j − w̄_j)²  with w̄ chosen so
    // optimum lies on cone boundary. Set w̄ to a vector with ‖w̄‖ = β · prod
    // for β > 1 (infeasible without constraint), so optimum projects onto
    // the cone surface.
    let beta = 2.0_f64.max(2.0 / tightness);
    let mut w_bar = vec![0.0; n_w];
    let mut wb_norm_sq = 0.0;
    for j in 0..n_w {
        w_bar[j] = rng.signed();
        wb_norm_sq += w_bar[j] * w_bar[j];
    }
    let scale = beta * prod / wb_norm_sq.sqrt().max(1e-30);
    for j in 0..n_w {
        w_bar[j] *= scale;
    }

    // P = 2I, q = -2 w̄  (so (1/2) x' P x + q' x = ‖w − w̄‖² − ‖w̄‖²).
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

    // Constraints: A x + s = b
    //   ZeroCone(dim1): −p_bar + s_p = 0   i.e. s_p = p_bar  (rows 0..dim1, cols=none → A_top = 0 (dim1 × n_w))
    //   GenPower(α, dim2): A_genpow x + s_genpow = b_genpow with
    //       p-block: s = p_bar (constant)
    //       w-block: −w + s_w = 0  ⇒  s_w = w
    //
    // We model this as a single GenPower cone of size (dim1 + dim2):
    //   first dim1 rows: A = 0 (dim1 × n_w),  b = p_bar
    //   next  dim2 rows: A = -I (dim2 × n_w), b = 0
    let m = dim1 + dim2;
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_idxs = Vec::new();
    let mut vals = Vec::new();
    for j in 0..n_w {
        // Column j: only row (dim1 + j) has -1.
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

/// Skewed-α: one dominant exponent, the rest tiny. Drives the cone toward
/// degeneracy along the p-direction.
fn make_skewed_genpow(
    dim1: usize,
    dim2: usize,
    tightness: f64,
    seed: u64,
) -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let mut alphas = vec![0.01 / (dim1 as f64 - 1.0).max(1.0); dim1];
    alphas[0] = 0.99;
    // Renormalize.
    let s: f64 = alphas.iter().sum();
    for a in &mut alphas {
        *a /= s;
    }
    make_active_genpow(alphas, dim2, tightness, seed)
}

fn run_one(
    P: &CscMatrix<f64>,
    q: &[f64],
    A: &CscMatrix<f64>,
    b: &[f64],
    cones: &[SupportedConeT<f64>],
    max_iter: u32,
) -> (SolverStatus, u32, f64) {
    let mut settings = DefaultSettings::default();
    settings.max_iter = max_iter;
    settings.verbose = std::env::var("BENCH_VERBOSE").is_ok();
    if std::env::var("BENCH_NO_DYNREG").is_ok() {
        settings.ipm.dynamic_regularization_enable = false;
    }
    if std::env::var("BENCH_NO_STATICREG").is_ok() {
        settings.ipm.static_regularization_enable = false;
    }
    if let Ok(method) = std::env::var("BENCH_LDL") {
        settings.ipm.direct_solve_method = method;
    }
    if let Ok(v) = std::env::var("BENCH_IR_MAX") {
        settings.ipm.iterative_refinement_max_iter = v.parse().unwrap_or(10);
    }
    if let Ok(v) = std::env::var("BENCH_IR_RELTOL") {
        settings.ipm.iterative_refinement_reltol = v.parse().unwrap_or(1e-13);
    }
    if std::env::var("BENCH_FOCUS").is_ok() {
        eprintln!(
            "[bench] q[0..3]={:.10},{:.10},{:.10} b[0..3]={:.10},{:.10},{:.10}",
            q[0], q[1], q[2], b[0], b[1], b[2]
        );
    }
    let mut solver = DefaultSolver::new(P, q, A, b, cones, settings).unwrap();
    let t0 = Instant::now();
    solver.solve();
    let elapsed = t0.elapsed().as_secs_f64() * 1e3;
    (solver.solution.status, solver.info.iterations, elapsed)
}

fn main() {
    println!(
        "{:>5} {:>5} {:>10}  {:>14}  {:>20}  {:>5}  {:>10}",
        "dim1", "dim2", "tightness", "kind", "status", "iter", "time (ms)"
    );
    println!("{:-<82}", "");

    let configs: Vec<(usize, usize)> = std::env::var("BENCH_DIMS")
        .ok()
        .map(|s| {
            s.split(',')
                .map(|p| {
                    let mut it = p.split('x');
                    let d1: usize = it.next().unwrap().parse().unwrap();
                    let d2: usize = it.next().unwrap().parse().unwrap();
                    (d1, d2)
                })
                .collect()
        })
        .unwrap_or_else(|| {
            if std::env::var("BENCH_FOCUS").is_ok() {
                vec![(128, 64)]
            } else {
                vec![(3, 1), (8, 4), (16, 8), (32, 16), (64, 32), (128, 64)]
            }
        });
    let tightnesses: &[f64] = if std::env::var("BENCH_FOCUS").is_ok() {
        &[0.999]
    } else {
        &[0.5, 0.9, 0.99, 0.999]
    };
    let seeds: &[u64] = if std::env::var("BENCH_FOCUS").is_ok() {
        &[137]
    } else {
        &[7, 41, 137]
    };

    for &(dim1, dim2) in &configs {
        for &kind in &["uniform", "skewed"] {
            for &tightness in tightnesses {
                let mut tot_iters = 0u32;
                let mut tot_ms = 0.0;
                let mut last_status = SolverStatus::Unsolved;
                for &seed in seeds {
                    let (P, q, A, b, cones) = if kind == "uniform" {
                        let alphas = vec![1.0 / dim1 as f64; dim1];
                        make_active_genpow(alphas, dim2, tightness, seed)
                    } else {
                        make_skewed_genpow(dim1, dim2, tightness, seed)
                    };
                    let (status, iters, ms) = run_one(&P, &q, &A, &b, &cones, 1000);
                    eprintln!(
                        "  [seed={:3}] iter={:4} status={:?} ms={:.1}",
                        seed, iters, status, ms
                    );
                    tot_iters += iters;
                    tot_ms += ms;
                    last_status = status;
                }
                let n_runs = seeds.len() as f64;
                println!(
                    "{:>5} {:>5} {:>10.3}  {:>14}  {:>20?}  {:>5.1}  {:>10.2}",
                    dim1,
                    dim2,
                    tightness,
                    kind,
                    last_status,
                    tot_iters as f64 / n_runs,
                    tot_ms / n_runs,
                );
            }
        }
        println!();
    }
}
