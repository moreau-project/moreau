#![allow(non_snake_case)]

//! Benchmark: faer-1t vs qdldl across problem sizes and structures
//!
//! Compares solve time for both LDL solver backends on QP problems
//! with different sparsity patterns.
//!
//! Usage:
//!   cargo run --release --features faer-sparse --example bench_solvers

use moreau::algebra::{BlockConcatenate, CscMatrix, MatrixMathMut};
use moreau::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus};
use std::time::Instant;

/// Simple LCG PRNG (no rand dependency needed for examples)
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
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ─── Problem generators ──────────────────────────────────────────

/// Box-constrained QP with diagonal P: KKT is nearly diagonal (best case for simplicial).
fn make_box_qp(n: usize) -> (CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>) {
    let mut P_colptr = vec![0usize; n + 1];
    let mut P_rowval = Vec::with_capacity(n);
    let mut P_nzval = Vec::with_capacity(n);
    for i in 0..n {
        P_colptr[i + 1] = i + 1;
        P_rowval.push(i);
        P_nzval.push(1.0 + (i as f64) * 0.01);
    }
    let P = CscMatrix {
        m: n,
        n,
        colptr: P_colptr,
        rowval: P_rowval,
        nzval: P_nzval,
    };
    let q: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let I1 = CscMatrix::<f64>::identity(n);
    let mut I2 = CscMatrix::<f64>::identity(n);
    I2.negate();
    let A = CscMatrix::vcat(&I1, &I2).unwrap();
    let b = vec![1.0; 2 * n];
    (P, q, A, b)
}

/// Sparse random QP: P has ~5*n nonzeros (banded), A has ~5*n nonzeros.
/// This creates fill-in in the KKT factorization where supernodal helps.
fn make_sparse_qp(n: usize, seed: u64) -> (CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let m = n; // equal number of constraints

    // Build P as upper-triangular banded: bandwidth = min(5, n-1)
    let bw = 5.min(n - 1);
    let mut P_colptr = vec![0usize; n + 1];
    let mut P_rowval = Vec::new();
    let mut P_nzval = Vec::new();

    for j in 0..n {
        let row_start = if j > bw { j - bw } else { 0 };
        for i in row_start..=j {
            P_rowval.push(i);
            if i == j {
                // Diagonal: make it dominant
                P_nzval.push(10.0 + rng.next_f64());
            } else {
                P_nzval.push(rng.next_f64() * 0.5);
            }
        }
        P_colptr[j + 1] = P_rowval.len();
    }
    let P = CscMatrix {
        m: n,
        n,
        colptr: P_colptr,
        rowval: P_rowval,
        nzval: P_nzval,
    };

    // Build A: each row has ~5 random entries (sparse constraint matrix)
    let entries_per_row = 5.min(n);
    let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..m {
        // Pick `entries_per_row` columns for this row (strided to avoid duplicates)
        let step = n.max(1) / entries_per_row.max(1);
        for k in 0..entries_per_row {
            let j = (i + k * step.max(1)) % n;
            triplets.push((i, j, rng.next_f64() * 2.0 - 1.0));
        }
    }

    // Build CSC from triplets
    // Sort by (col, row)
    triplets.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    // Dedup (row, col)
    triplets.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    let mut A_colptr = vec![0usize; n + 1];
    let mut A_rowval = Vec::new();
    let mut A_nzval = Vec::new();
    let mut idx = 0;
    for j in 0..n {
        while idx < triplets.len() && triplets[idx].1 == j {
            A_rowval.push(triplets[idx].0);
            A_nzval.push(triplets[idx].2);
            idx += 1;
        }
        A_colptr[j + 1] = A_rowval.len();
    }
    let A = CscMatrix {
        m,
        n,
        colptr: A_colptr,
        rowval: A_rowval,
        nzval: A_nzval,
    };

    let q: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
    let b = vec![10.0; m]; // generous RHS so problem is feasible

    (P, q, A, b)
}

/// Dense-constraint QP: A is fully dense (m x n), P is banded.
/// This is the worst case for simplicial and best case for supernodal.
fn make_dense_constraint_qp(
    n: usize,
    seed: u64,
) -> (CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let m = n / 2; // half as many constraints as variables

    // P: tridiagonal
    let mut P_colptr = vec![0usize; n + 1];
    let mut P_rowval = Vec::new();
    let mut P_nzval = Vec::new();
    for j in 0..n {
        let start = if j > 0 { j - 1 } else { 0 };
        for i in start..=j {
            P_rowval.push(i);
            if i == j {
                P_nzval.push(4.0 + rng.next_f64());
            } else {
                P_nzval.push(0.5);
            }
        }
        P_colptr[j + 1] = P_rowval.len();
    }
    let P = CscMatrix {
        m: n,
        n,
        colptr: P_colptr,
        rowval: P_rowval,
        nzval: P_nzval,
    };

    // A: fully dense m x n
    let mut A_colptr = vec![0usize; n + 1];
    let mut A_rowval = Vec::new();
    let mut A_nzval = Vec::new();
    for j in 0..n {
        for i in 0..m {
            A_rowval.push(i);
            A_nzval.push(rng.next_f64() * 2.0 - 1.0);
        }
        A_colptr[j + 1] = A_rowval.len();
    }
    let A = CscMatrix {
        m,
        n,
        colptr: A_colptr,
        rowval: A_rowval,
        nzval: A_nzval,
    };

    let q: Vec<f64> = (0..n).map(|_| rng.next_f64() * 2.0 - 1.0).collect();
    let b = vec![100.0; m];

    (P, q, A, b)
}

// ─── Benchmark harness ───────────────────────────────────────────

fn solve_with_method(
    P: &CscMatrix<f64>,
    q: &[f64],
    A: &CscMatrix<f64>,
    b: &[f64],
    method: &str,
) -> (f64, u32, SolverStatus) {
    let cones = vec![moreau::solver::SupportedConeT::NonnegativeConeT(b.len())];

    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.ipm.direct_solve_method = method.to_string();

    let mut solver = DefaultSolver::new(P, q, A, b, &cones, settings).unwrap();

    let start = Instant::now();
    solver.solve();
    let elapsed = start.elapsed();

    (
        elapsed.as_secs_f64() * 1e6,
        solver.solution.iterations,
        solver.solution.status,
    )
}

fn bench_suite(
    label: &str,
    sizes: &[usize],
    gen: impl Fn(usize) -> (CscMatrix<f64>, Vec<f64>, CscMatrix<f64>, Vec<f64>),
) {
    let warmup_iters = 2;
    let bench_iters = 5;

    println!("\n{}", label);
    println!("{}", "=".repeat(label.len()));
    println!(
        "{:>6} {:>6}  {:>10} {:>5} {:>6}  {:>10} {:>5} {:>6}  {:>7}",
        "n", "nnzKKT", "faer(us)", "iter", "stat", "qdldl(us)", "iter", "stat", "speedup"
    );
    println!("{}", "-".repeat(80));

    for &n in sizes {
        let (P, q, A, b) = gen(n);

        // Estimate KKT nnz: nnz(P) + nnz(A) + n + m  (rough)
        let nnz_kkt = P.nnz() + A.nnz() + n + b.len();

        // Warmup
        for _ in 0..warmup_iters {
            solve_with_method(&P, &q, &A, &b, "faer-1t");
            solve_with_method(&P, &q, &A, &b, "qdldl");
        }

        let mut faer_times = Vec::with_capacity(bench_iters);
        let mut qdldl_times = Vec::with_capacity(bench_iters);
        let mut faer_iters = 0u32;
        let mut qdldl_iters = 0u32;
        let mut faer_status = SolverStatus::Unsolved;
        let mut qdldl_status = SolverStatus::Unsolved;

        for _ in 0..bench_iters {
            let (t, it, st) = solve_with_method(&P, &q, &A, &b, "faer-1t");
            faer_times.push(t);
            faer_iters = it;
            faer_status = st;

            let (t, it, st) = solve_with_method(&P, &q, &A, &b, "qdldl");
            qdldl_times.push(t);
            qdldl_iters = it;
            qdldl_status = st;
        }

        faer_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        qdldl_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let faer_median = faer_times[bench_iters / 2];
        let qdldl_median = qdldl_times[bench_iters / 2];
        let speedup = qdldl_median / faer_median;

        let st = |s: SolverStatus| match s {
            SolverStatus::Solved => "OK",
            SolverStatus::AlmostSolved => "~OK",
            _ => "FAIL",
        };

        println!(
            "{:>6} {:>6}  {:>10.0} {:>5} {:>6}  {:>10.0} {:>5} {:>6}  {:>6.2}x",
            n,
            nnz_kkt,
            faer_median,
            faer_iters,
            st(faer_status),
            qdldl_median,
            qdldl_iters,
            st(qdldl_status),
            speedup,
        );
    }
    println!("(median of {} runs, {} warmup)", bench_iters, warmup_iters);
}

fn main() {
    println!("Moreau LDL Solver Benchmark: faer-1t vs qdldl");
    println!("==============================================");

    // 1. Box QP (diagonal P, identity A) — best case for simplicial
    bench_suite(
        "Box QP (diagonal P, A=[I;-I]) — minimal fill-in",
        &[2, 5, 10, 50, 200, 1000],
        |n| make_box_qp(n),
    );

    // 2. Sparse random QP — moderate fill-in
    bench_suite(
        "Sparse QP (banded P, ~5 entries/row A) — moderate fill-in",
        &[5, 10, 50, 200, 500, 1000],
        |n| make_sparse_qp(n, 42),
    );

    // 3. Dense constraint QP — heavy fill-in (supernodal sweet spot)
    bench_suite(
        "Dense-constraint QP (tridiag P, dense A) — heavy fill-in",
        &[5, 10, 50, 100, 200, 500],
        |n| make_dense_constraint_qp(n, 42),
    );
}
