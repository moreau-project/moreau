#![allow(non_snake_case)]
//! Direct-x GenPow PD-scaling bench: hard near-boundary problems.
//!
//! Mirrors the active-boundary problems from `bench_genpow_pd_scaling.rs`
//! but solves them via direct-x. Used to bench-gate the rank-9 PD-scaling
//! KKT expansion against the rank-3-only baseline.
//!
//! Each problem: cone vector x ∈ GenPower(α, dim2), with the first dim1
//! coordinates fixed to a random `p_bar` via a slack ZeroCone and the
//! last dim2 free (with `-w + s = 0` regression: but here we put w into
//! x directly). Target is set to push the optimum onto the cone boundary.
//!
//!   uniform: α = (1/dim1, ..., 1/dim1)
//!   skewed:  α₀ = 0.99, α_i = 0.01/(dim1-1) for i ≥ 1
//!
//! Tightness ∈ {0.5, 0.9, 0.99, 0.999} — closer to 1 = harder.
//!
//! Usage: cargo run --release --example bench_xcone_genpow_pd

use moreau::algebra::CscMatrix;
use moreau::solver::{
    DefaultSettings, DefaultSolver, GenPowerConeT, IPSolver, SolverStatus, SupportedConeT,
    SupportedXConeT, ZeroConeT,
};
use std::time::Instant;

/// Splitmix64 — same as bench_genpow_pd_scaling for reproducibility.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }
    fn signed(&mut self) -> f64 {
        2.0 * self.uniform() - 1.0
    }
}

/// Build a hard active-boundary direct-x GenPow problem.
///
/// Decision variable x has length dim1 + dim2 (the full cone vector).
/// First dim1 coords pinned to p_bar via a ZeroCone; last dim2 are free
/// w. Objective minimises ‖w − w̄‖² with w̄ scaled to lie outside the
/// cone, so the optimum sits on the boundary.
fn make_skewed_directx(
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
    Vec<SupportedXConeT>,
) {
    let mut alphas = vec![0.01 / (dim1 as f64 - 1.0).max(1.0); dim1];
    alphas[0] = 0.99;
    let s: f64 = alphas.iter().sum();
    for a in &mut alphas {
        *a /= s;
    }
    make_active_directx(alphas, dim2, tightness, seed)
}

fn make_uniform_directx(
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
    Vec<SupportedXConeT>,
) {
    let alphas = vec![1.0 / dim1 as f64; dim1];
    make_active_directx(alphas, dim2, tightness, seed)
}

fn make_active_directx(
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
    Vec<SupportedXConeT>,
) {
    let dim1 = alphas.len();
    let n = dim1 + dim2; // x = (p, w)
    let mut rng = Rng::new(seed);

    let mut p_bar = vec![0.0; dim1];
    for i in 0..dim1 {
        p_bar[i] = 1.0 + 0.3 * rng.uniform();
    }
    let log_prod: f64 = (0..dim1).map(|i| alphas[i] * p_bar[i].ln()).sum();
    let prod = log_prod.exp();

    let beta = 2.0_f64.max(2.0 / tightness);
    let mut w_bar = vec![0.0; dim2];
    let mut wb_norm_sq = 0.0;
    for j in 0..dim2 {
        w_bar[j] = rng.signed();
        wb_norm_sq += w_bar[j] * w_bar[j];
    }
    let scale = beta * prod / wb_norm_sq.sqrt().max(1e-30);
    for j in 0..dim2 {
        w_bar[j] *= scale;
    }

    // P diagonal — by default 2·I (κ=1). `BENCH_P_KAPPA` env var
    // generates a diagonal with condition number κ: d_i = 2 · κ^(i/(n-1))
    // ranging from 2 to 2·κ. κ=1 reproduces the well-conditioned default.
    let kappa: f64 = std::env::var("BENCH_P_KAPPA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let mut p_diag = vec![0.0_f64; n];
    if (kappa - 1.0).abs() < 1e-9 {
        for v in &mut p_diag {
            *v = 2.0;
        }
    } else {
        let log_k = kappa.ln();
        for i in 0..n {
            let t = (i as f64) / ((n - 1).max(1) as f64);
            p_diag[i] = 2.0 * (t * log_k).exp();
        }
    }
    let P = CscMatrix::<f64>::new(n, n, (0..=n).collect(), (0..n).collect(), p_diag);

    // q: −2 w̄ on the w-block, 0 on the p-block (objective is ‖w−w̄‖²
    // minus a constant — the p-coords are pinned by ZeroCone).
    let mut q = vec![0.0_f64; n];
    for j in 0..dim2 {
        q[dim1 + j] = -2.0 * w_bar[j];
    }

    // Slack constraints: x[0..dim1] = p_bar via ZeroCone(dim1).
    //   A: dim1 × n with -I in the first dim1 columns (so −x_p + s = 0
    //   with s = -p_bar pins x_p = p_bar).  Actually use A = I, b = p_bar
    //   so I·x_p − p_bar + s = 0  ⇒  s = p_bar − x_p, which equals 0 in
    //   ZeroCone, fixing x_p = p_bar.
    let mut col_ptrs = vec![0usize; n + 1];
    let mut row_idxs = Vec::new();
    let mut vals = Vec::new();
    for col in 0..dim1 {
        row_idxs.push(col);
        vals.push(1.0);
        col_ptrs[col + 1] = col_ptrs[col] + 1;
    }
    for col in dim1..n {
        col_ptrs[col + 1] = col_ptrs[col];
    }
    let A = CscMatrix::<f64>::new(dim1, n, col_ptrs, row_idxs, vals);
    let b = p_bar.clone();
    let cones_slack = vec![ZeroConeT(dim1)];
    let dir_cones = vec![SupportedXConeT::GenPowerXConeT(
        (0..n).collect(),
        alphas,
        dim2,
    )];
    (P, q, A, b, cones_slack, dir_cones)
}

fn run_directx(
    P: &CscMatrix<f64>,
    q: &[f64],
    A: &CscMatrix<f64>,
    b: &[f64],
    cones: &[SupportedConeT<f64>],
    dir_cones: &[SupportedXConeT],
    max_iter: u32,
) -> (SolverStatus, u32, f64) {
    let mut settings = DefaultSettings::default();
    settings.max_iter = max_iter;
    settings.verbose = false;
    settings.ipm.presolve_enable = false;
    settings.ipm.equilibrate_enable = std::env::var("BENCH_EQ").is_ok();

    let mut solver = DefaultSolver::new_with_xcones(P, q, A, b, cones, dir_cones, settings)
        .expect("solver construction must succeed");
    let t = Instant::now();
    solver.solve();
    let ms = t.elapsed().as_secs_f64() * 1e3;
    (solver.solution.status, solver.solution.iterations, ms)
}

fn main() {
    println!("=== Direct-x GenPow PD-scaling bench (active boundary) ===");
    println!("Problem: pinned p, push w to cone boundary. Compares iter counts");
    println!("at the rank-9 wiring on D vs the rank-3 baseline.\n");

    let configs: Vec<(usize, usize)> = vec![(8, 4), (16, 8), (32, 16), (64, 32), (128, 64)];
    let tightnesses: &[f64] = &[0.9, 0.99, 0.999];
    let seeds: &[u64] = &[7, 41, 137];

    println!(" dim1  dim2 tightness          kind                status  iter(avg)   ms(avg)");
    println!("------------------------------------------------------------------------");

    for &(dim1, dim2) in &configs {
        for &kind in &["uniform", "skewed"] {
            for &tightness in tightnesses {
                let mut tot_iters = 0u32;
                let mut tot_ms = 0.0;
                let mut last_status = SolverStatus::Unsolved;
                for &seed in seeds {
                    let (P, q, A, b, cs, xs) = if kind == "uniform" {
                        make_uniform_directx(dim1, dim2, tightness, seed)
                    } else {
                        make_skewed_directx(dim1, dim2, tightness, seed)
                    };
                    let (status, iters, ms) = run_directx(&P, &q, &A, &b, &cs, &xs, 1000);
                    tot_iters += iters;
                    tot_ms += ms;
                    last_status = status;
                }
                let n = seeds.len() as f64;
                println!(
                    "{:>5} {:>5} {:>10.3}  {:>14}  {:>20?}  {:>5.1}  {:>10.2}",
                    dim1,
                    dim2,
                    tightness,
                    kind,
                    last_status,
                    tot_iters as f64 / n,
                    tot_ms / n,
                );
            }
        }
        println!();
    }
}
