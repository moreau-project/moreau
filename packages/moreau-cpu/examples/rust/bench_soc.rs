#![allow(non_snake_case)]

//! Second-order-cone bench. Times a portfolio-style QP with one big SOC
//! at varying dim, exercised with both the default (expansion-first)
//! KKT permutation and the prior AMD-on-augmented permutation
//! (`KKT_PERM_NATURAL=1`). Lets us check whether the GenPow-motivated
//! permutation change broke SOC perf.

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

/// SOCP: minimize ½ x' P x + q' x   s.t. ‖A1 x + b1‖ ≤ c1' x + d1
/// We model this with one SOC of dimension `dim_soc + 1`.
fn make_socp(
    n_var: usize,
    dim_soc: usize,
    seed: u64,
) -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    let mut rng = Rng::new(seed);
    // P = I (PD).
    let mut p_colptr = vec![0usize; n_var + 1];
    let mut p_rowval = Vec::new();
    let mut p_nzval = Vec::new();
    for i in 0..n_var {
        p_colptr[i + 1] = i + 1;
        p_rowval.push(i);
        p_nzval.push(1.0);
    }
    let p = CscMatrix {
        m: n_var,
        n: n_var,
        colptr: p_colptr,
        rowval: p_rowval,
        nzval: p_nzval,
    };
    // q = random.
    let q: Vec<f64> = (0..n_var).map(|_| rng.signed()).collect();

    // Build A row by row, then convert to CSC.
    // SOC of dim `dim_soc + 1`: rows are [c'x + d; A1·x + b1].
    // We'll set c = 1·e_0, A1 = -[I; 0] (selecting first dim_soc components).
    let m = dim_soc + 1;
    let mut entries: Vec<(usize, usize, f64)> = Vec::new();
    // Row 0 (the t in (t, w) ∈ SOC): t = -c'x → c = e_0
    entries.push((0, 0, -1.0));
    // Rows 1..dim_soc+1: w_i = -A1·x = -x_i for i in 0..min(dim_soc, n_var)
    for i in 0..dim_soc.min(n_var) {
        entries.push((1 + i, i, 1.0));
    }
    // CSC build
    let mut col_counts = vec![0usize; n_var + 1];
    for &(_, c, _) in &entries {
        col_counts[c + 1] += 1;
    }
    for i in 1..=n_var {
        col_counts[i] += col_counts[i - 1];
    }
    let mut col_offset = col_counts.clone();
    let nnz = entries.len();
    let mut row_idxs = vec![0usize; nnz];
    let mut vals = vec![0.0_f64; nnz];
    for &(r, c, v) in &entries {
        let pos = col_offset[c];
        row_idxs[pos] = r;
        vals[pos] = v;
        col_offset[c] += 1;
    }
    let a = CscMatrix {
        m,
        n: n_var,
        colptr: col_counts,
        rowval: row_idxs,
        nzval: vals,
    };
    let b = vec![0.0; m];

    let cones = vec![SupportedConeT::SecondOrderConeT(dim_soc + 1)];
    (p, q, a, b, cones)
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
    let mut solver = DefaultSolver::new(P, q, A, b, cones, settings).unwrap();
    let t0 = Instant::now();
    solver.solve();
    let elapsed = t0.elapsed().as_secs_f64() * 1e3;
    (solver.solution.status, solver.info.iterations, elapsed)
}

fn main() {
    println!(
        "{:>6} {:>6}  {:>20}  {:>5}  {:>10}",
        "n_var", "dim_soc", "status", "iter", "time (ms)"
    );
    println!("{:-<60}", "");
    let configs: &[(usize, usize)] = &[
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
    ];
    let seeds = [7u64, 41, 137];
    for &(n_var, dim_soc) in configs {
        let mut tot_iters = 0u32;
        let mut tot_ms = 0.0;
        let mut last_status = SolverStatus::Unsolved;
        for &seed in &seeds {
            let (p, q, a, b, cones) = make_socp(n_var, dim_soc, seed);
            // Warm up jit/cache once.
            let _ = run_one(&p, &q, &a, &b, &cones, 200);
            let (status, iters, ms) = run_one(&p, &q, &a, &b, &cones, 200);
            tot_iters += iters;
            tot_ms += ms;
            last_status = status;
        }
        let n = seeds.len() as f64;
        println!(
            "{:>6} {:>6}  {:>20?}  {:>5.1}  {:>10.2}",
            n_var,
            dim_soc,
            last_status,
            tot_iters as f64 / n,
            tot_ms / n,
        );
    }
}
