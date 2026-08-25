#![allow(non_snake_case)]

//! GenPower dense / sparse-LDL parity check.
//!
//! For each (dim, kind, tightness, seed) configuration this bench runs the
//! same problem under both CPU paths and reports:
//!   - status, iter count, objective value
//!   - max abs difference of x against the dense reference
//!   - per-path wall-clock time
//!
//! Paths exercised (selected via `MOREAU_N_DENSE_GENPOW`):
//!   1. `dense`     : MOREAU_N_DENSE_GENPOW = ∞, qdldl
//!   2. `sparse-ldl`: MOREAU_N_DENSE_GENPOW = 0, qdldl  (uses the
//!     adaptive equilibration in `csc_update_sparsecone` to bound
//!     the augmented matrix entries on near-boundary genpow iterates).
//!
//! Usage:
//!   cargo run --release --example bench_genpow_parity
//!   BENCH_DIMS="32x16,64x32,128x64" cargo run --release --example bench_genpow_parity
//!   BENCH_TIGHTNESSES="0.99,0.999" cargo run --release --example bench_genpow_parity

use moreau::algebra::CscMatrix;
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
    let n = n_w;
    let mut rng = Rng::new(seed);

    let mut p_bar = vec![0.0; dim1];
    for i in 0..dim1 {
        p_bar[i] = 1.0 + 0.3 * rng.uniform();
    }
    let log_prod: f64 = (0..dim1).map(|i| alphas[i] * p_bar[i].ln()).sum();
    let prod = log_prod.exp();

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
    for j in 0..n_w {
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
    let s: f64 = alphas.iter().sum();
    for a in &mut alphas {
        *a /= s;
    }
    make_active_genpow(alphas, dim2, tightness, seed)
}

#[derive(Clone, Copy)]
enum Path {
    Dense,
    SparseLdl,
}

impl Path {
    fn apply_env(&self) {
        // SAFETY: single-threaded benchmark. We mutate the process env var
        // before constructing each solver so the lazy `dense_genpow_threshold()`
        // reader (called in `GenPowerCone::new`) picks up the new value.
        match self {
            Path::Dense => unsafe {
                std::env::set_var("MOREAU_N_DENSE_GENPOW", "1000000");
            },
            Path::SparseLdl => unsafe {
                std::env::set_var("MOREAU_N_DENSE_GENPOW", "0");
            },
        }
    }
}

struct RunResult {
    status: SolverStatus,
    iters: u32,
    obj_val: f64,
    x: Vec<f64>,
    ms: f64,
}

fn run_one(
    P: &CscMatrix<f64>,
    q: &[f64],
    A: &CscMatrix<f64>,
    b: &[f64],
    cones: &[SupportedConeT<f64>],
    path: Path,
) -> RunResult {
    path.apply_env();
    let mut settings = DefaultSettings::default();
    settings.max_iter = 1000;
    settings.verbose = false;
    settings.ipm.direct_solve_method = "qdldl".to_string();
    let mut solver = DefaultSolver::new(P, q, A, b, cones, settings).unwrap();
    let t0 = Instant::now();
    solver.solve();
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    RunResult {
        status: solver.solution.status,
        iters: solver.info.iterations,
        obj_val: solver.info.cost_primal,
        x: solver.solution.x.clone(),
        ms,
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

fn parse_dims() -> Vec<(usize, usize)> {
    std::env::var("BENCH_DIMS")
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
        .unwrap_or_else(|| vec![(16, 8), (32, 16), (64, 32), (128, 64)])
}

fn parse_tightnesses() -> Vec<f64> {
    std::env::var("BENCH_TIGHTNESSES")
        .ok()
        .map(|s| s.split(',').map(|p| p.parse().unwrap()).collect())
        .unwrap_or_else(|| vec![0.5, 0.9, 0.99, 0.999])
}

fn main() {
    let dims = parse_dims();
    let tightnesses = parse_tightnesses();
    let seeds: &[u64] = &[7, 41, 137];
    let kinds: &[&str] = &["uniform", "skewed"];

    println!(
        "{:>4} {:>4} {:>8} {:>9} {:>5}  {:>11} {:>4} {:>14} {:>9}  {:>11} {:>4} {:>14} {:>9} {:>10}",
        "d1",
        "d2",
        "kind",
        "tight",
        "seed",
        "dn-status",
        "iter",
        "obj",
        "ms",
        "ldl-status",
        "iter",
        "obj",
        "ms",
        "Δx",
    );
    println!("{:-<140}", "");

    let mut total_match_obj = 0;
    let mut total_match_x = 0;
    let mut total_runs = 0;
    let obj_tol = 1e-6;
    let x_tol = 1e-5;

    for &(d1, d2) in &dims {
        for &kind in kinds {
            for &tightness in &tightnesses {
                for &seed in seeds {
                    let (P, q, A, b, cones_template) = if kind == "uniform" {
                        let alphas = vec![1.0 / d1 as f64; d1];
                        make_active_genpow(alphas, d2, tightness, seed)
                    } else {
                        make_skewed_genpow(d1, d2, tightness, seed)
                    };
                    // Each run consumes `cones`, so clone for each.
                    let r_dn = run_one(&P, &q, &A, &b, &cones_template.clone(), Path::Dense);
                    let r_ldl = run_one(&P, &q, &A, &b, &cones_template, Path::SparseLdl);

                    let dx_ldl = max_abs_diff(&r_ldl.x, &r_dn.x);

                    let obj_match = (r_ldl.obj_val - r_dn.obj_val).abs() < obj_tol;
                    let x_match = dx_ldl < x_tol;
                    if obj_match {
                        total_match_obj += 1;
                    }
                    if x_match {
                        total_match_x += 1;
                    }
                    total_runs += 1;

                    println!(
                        "{:>4} {:>4} {:>8} {:>9.3} {:>5}  {:>11?} {:>4} {:>14.6e} {:>9.2}  {:>11?} {:>4} {:>14.6e} {:>9.2} {:>10.2e}",
                        d1,
                        d2,
                        kind,
                        tightness,
                        seed,
                        r_dn.status,
                        r_dn.iters,
                        r_dn.obj_val,
                        r_dn.ms,
                        r_ldl.status,
                        r_ldl.iters,
                        r_ldl.obj_val,
                        r_ldl.ms,
                        dx_ldl,
                    );
                }
            }
        }
        println!();
    }

    println!(
        "PARITY: obj_match {}/{} ({}%), x_match {}/{} ({}%) (obj_tol={:.0e}, x_tol={:.0e})",
        total_match_obj,
        total_runs,
        100 * total_match_obj / total_runs.max(1),
        total_match_x,
        total_runs,
        100 * total_match_x / total_runs.max(1),
        obj_tol,
        x_tol,
    );
}
