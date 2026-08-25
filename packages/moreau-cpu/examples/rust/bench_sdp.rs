#![allow(non_snake_case)]

//! SDP Benchmark: chordal decomposition speedup and backward pass overhead
//!
//! Compares solve time for block-diagonal PSD problems (chordal decomposition active)
//! vs dense PSD problems (no decomposition). Also measures backward pass overhead.
//!
//! Usage: cargo run --release --features sdp-openblas --example bench_sdp [mat_dim] [batch_size] [n_runs]

use moreau::solver::implementations::default::{
    CompiledSolver, DefaultSettings, UpstreamGradients,
};
use moreau::solver::{SolverStatus, SupportedConeT};
use std::time::Instant;

fn triangular_number(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Simple PRNG (xorshift64).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0
    }
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.next_u64() as f64 / u64::MAX as f64) * (hi - lo)
    }
    fn randn_vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.next_f64() * scale).collect()
    }
}

/// Build svec of a block-diagonal PD matrix.
fn make_block_diag_svec(mat_dim: usize, block_dim: usize, rng: &mut Rng) -> Vec<f64> {
    let svec_dim = triangular_number(mat_dim);
    let mut svec = vec![0.0f64; svec_dim];
    let mut idx = 0;
    for j in 0..mat_dim {
        for i in 0..=j {
            let bi = i / block_dim;
            let bj = j / block_dim;
            if bi == bj {
                if i == j {
                    svec[idx] = rng.next_range(2.0, 5.0);
                } else {
                    svec[idx] = rng.next_f64() * 0.1 * 2f64.sqrt();
                }
            }
            idx += 1;
        }
    }
    svec
}

/// Build block-diagonal A sparsity (CSR).
fn build_block_diag_A(mat_dim: usize, block_dim: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let mut ro = vec![0usize];
    let mut ci = vec![];
    let mut vals = vec![];
    let mut idx = 0;
    for j in 0..mat_dim {
        for i in 0..=j {
            if i / block_dim == j / block_dim {
                ci.push(idx);
                vals.push(1.0);
                ro.push(ro.last().unwrap() + 1);
            } else {
                ro.push(*ro.last().unwrap());
            }
            idx += 1;
        }
    }
    (ro, ci, vals)
}

/// Build full-A (dense identity) sparsity (CSR).
fn build_dense_A(m: usize) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let ro: Vec<usize> = (0..=m).collect();
    let ci: Vec<usize> = (0..m).collect();
    let vals = vec![1.0; m];
    (ro, ci, vals)
}

/// Build PD svec for dense (non-decomposable) case.
fn make_dense_svec(mat_dim: usize, rng: &mut Rng) -> Vec<f64> {
    let svec_dim = triangular_number(mat_dim);
    let mut svec = vec![0.0f64; svec_dim];
    let mut idx = 0;
    for j in 0..mat_dim {
        for i in 0..=j {
            if i == j {
                svec[idx] = rng.next_range(2.0, 5.0);
            } else {
                svec[idx] = rng.next_f64() * 0.05 * 2f64.sqrt();
            }
            idx += 1;
        }
    }
    svec
}

struct BenchResult {
    setup_ms: f64,
    solve_ms: f64,
    backward_ms: f64,
    num_solved: usize,
    total_iters: usize,
}

fn run_bench(
    label: &str,
    mat_dim: usize,
    batch_size: usize,
    n_runs: usize,
    block_diag: bool,
    block_dim: usize,
    enable_grad: bool,
) -> BenchResult {
    let svec_dim = triangular_number(mat_dim);
    let n = svec_dim;
    let m = svec_dim;

    let P_ro: Vec<usize> = (0..=n).collect();
    let P_ci: Vec<usize> = (0..n).collect();
    let P_vals = vec![1.0; n];

    let (A_ro, A_ci, A_vals) = if block_diag {
        build_block_diag_A(mat_dim, block_dim)
    } else {
        build_dense_A(m)
    };

    let cones = vec![SupportedConeT::PSDTriangleConeT(mat_dim)];

    let mut settings = DefaultSettings::<f64>::default();
    settings.verbose = false;

    let mut compiled = CompiledSolver::new(
        n,
        m,
        &P_ro,
        &P_ci,
        &A_ro,
        &A_ci,
        &cones,
        settings,
        batch_size,
        enable_grad,
    )
    .expect("construction failed");

    let mut rng = Rng::new(42);

    // Generate batch problems
    let bs: Vec<Vec<f64>> = (0..batch_size)
        .map(|_| {
            if block_diag {
                make_block_diag_svec(mat_dim, block_dim, &mut rng)
            } else {
                make_dense_svec(mat_dim, &mut rng)
            }
        })
        .collect();
    let qs: Vec<Vec<f64>> = bs
        .iter()
        .map(|b| b.iter().map(|v| -2.0 * v + rng.next_f64() * 0.1).collect())
        .collect();

    // Upstream grads for backward
    let upstreams: Vec<UpstreamGradients<f64>> = (0..batch_size)
        .map(|_| UpstreamGradients {
            dx: rng.randn_vec(n, 0.1),
            ds: vec![0.0; m],
            dz: rng.randn_vec(m, 0.1),
            dz_x: vec![],
        })
        .collect();

    // Warmup
    compiled.setup_shared(&P_vals, &A_vals, batch_size);
    let _ = compiled.solve(&qs, &bs).unwrap();
    if enable_grad {
        let _ = compiled.backward(&upstreams).unwrap();
    }

    // Benchmark
    let mut setup_times = Vec::with_capacity(n_runs);
    let mut solve_times = Vec::with_capacity(n_runs);
    let mut backward_times = Vec::with_capacity(n_runs);
    let mut last_solutions = None;

    for _ in 0..n_runs {
        let t0 = Instant::now();
        compiled.setup_shared(&P_vals, &A_vals, batch_size);
        setup_times.push(t0.elapsed().as_secs_f64() * 1000.0);

        let t1 = Instant::now();
        let solutions = compiled.solve(&qs, &bs).unwrap();
        solve_times.push(t1.elapsed().as_secs_f64() * 1000.0);

        if enable_grad {
            let t2 = Instant::now();
            let _ = compiled.backward(&upstreams).unwrap();
            backward_times.push(t2.elapsed().as_secs_f64() * 1000.0);
        }

        last_solutions = Some(solutions);
    }

    setup_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    solve_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    backward_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let median = |v: &[f64]| if v.is_empty() { 0.0 } else { v[v.len() / 2] };

    let solutions = last_solutions.unwrap();
    let num_solved = solutions
        .iter()
        .filter(|s| s.status == SolverStatus::Solved || s.status == SolverStatus::AlmostSolved)
        .count();
    let total_iters: u32 = solutions.iter().map(|s| s.iterations).sum();

    let result = BenchResult {
        setup_ms: median(&setup_times),
        solve_ms: median(&solve_times),
        backward_ms: median(&backward_times),
        num_solved,
        total_iters: total_iters as usize,
    };

    if !label.is_empty() {
        let per_prob_solve = result.solve_ms / batch_size as f64;
        let per_prob_bwd = result.backward_ms / batch_size as f64;

        println!("{}", label);
        println!(
            "  PSD({}) svec_dim={} | batch={} | {} runs (median)",
            mat_dim, svec_dim, batch_size, n_runs
        );
        println!("  setup:    {:.3} ms", result.setup_ms);
        println!(
            "  solve:    {:.3} ms total, {:.3} ms/problem ({} solved, {} total iters)",
            result.solve_ms, per_prob_solve, num_solved, total_iters
        );
        if enable_grad {
            println!(
                "  backward: {:.3} ms total, {:.3} ms/problem",
                result.backward_ms, per_prob_bwd
            );
            println!("  bwd/fwd:  {:.1}x", result.backward_ms / result.solve_ms);
        }
    }

    result
}

/// Same as run_bench but no output.
fn run_bench_quiet(
    mat_dim: usize,
    batch_size: usize,
    n_runs: usize,
    block_diag: bool,
    block_dim: usize,
    enable_grad: bool,
) -> BenchResult {
    run_bench(
        "",
        mat_dim,
        batch_size,
        n_runs,
        block_diag,
        block_dim,
        enable_grad,
    )
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mat_dim: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(8);
    let batch_size: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(32);
    let n_runs: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          SDP Benchmark: Chordal vs Dense                    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // ── Forward only ─────────────────────────────────────────────────

    println!("━━━ Forward Solve Only ━━━\n");

    let dense_fwd = run_bench(
        "Dense (no chordal):",
        mat_dim,
        batch_size,
        n_runs,
        false,
        mat_dim,
        false,
    );
    println!();

    let block_dim = 2.min(mat_dim);
    let chordal_fwd = run_bench(
        &format!("Chordal (block_dim={}):", block_dim),
        mat_dim,
        batch_size,
        n_runs,
        true,
        block_dim,
        false,
    );
    println!();

    let speedup = dense_fwd.solve_ms / chordal_fwd.solve_ms;
    println!("  ▸ Chordal speedup (solve): {:.2}x\n", speedup);

    // ── Forward + Backward ───────────────────────────────────────────

    println!("━━━ Forward + Backward ━━━\n");

    let dense_grad = run_bench(
        "Dense (no chordal) + backward:",
        mat_dim,
        batch_size,
        n_runs,
        false,
        mat_dim,
        true,
    );
    println!();

    let chordal_grad = run_bench(
        &format!("Chordal (block_dim={}) + backward:", block_dim),
        mat_dim,
        batch_size,
        n_runs,
        true,
        block_dim,
        true,
    );
    println!();

    let total_dense = dense_grad.solve_ms + dense_grad.backward_ms;
    let total_chordal = chordal_grad.solve_ms + chordal_grad.backward_ms;
    println!(
        "  ▸ Chordal speedup (solve+backward): {:.2}x",
        total_dense / total_chordal
    );

    // ── Scaling ──────────────────────────────────────────────────────

    println!("\n━━━ Scaling (chordal, block_dim={}) ━━━\n", block_dim);
    println!(
        "{:>8} {:>8} {:>10} {:>10} {:>10} {:>8}",
        "mat_dim", "svec", "solve/ms", "bwd/ms", "total/ms", "iters"
    );
    println!("{}", "─".repeat(64));

    for &md in &[4, 6, 8, 10, 12, 16] {
        if md < 2 * block_dim {
            continue;
        }
        let r = run_bench_quiet(md, batch_size, n_runs, true, block_dim, true);
        let total = r.solve_ms + r.backward_ms;
        println!(
            "{:>8} {:>8} {:>10.3} {:>10.3} {:>10.3} {:>8}",
            md,
            triangular_number(md),
            r.solve_ms,
            r.backward_ms,
            total,
            r.total_iters
        );
    }
}
