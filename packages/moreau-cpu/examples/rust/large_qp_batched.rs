#![allow(non_snake_case)]

//! Large Batched QP Benchmark
//!
//! Benchmarks solving many QP problems in parallel using CompiledSolver.
//! Matches the C++ large_qp_batched example for comparison.
//!
//! Usage: cargo run --example large_qp_batched --release [batch_size] [n] [m] [density]
//!
//! Default: batch_size=128, n=5000, m=2500, density=0.05

use moreau::solver::implementations::default::{BatchProblem, CompiledSolver, DefaultSettings};
use moreau::solver::SupportedConeT;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Default problem size (matches C++ defaults)
    let batch_size: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(128);
    let n: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5000);
    let m: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2500);
    let density: f64 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0.05);

    println!("Creating large batched QP problem:");
    println!("  Variables: {}", n);
    println!("  Constraints: {}", m);
    println!("  Batch size: {}", batch_size);
    println!("  Target density: {:.1}%", density * 100.0);

    let mut rng = StdRng::seed_from_u64(42);

    // P matrix: diagonal (n x n)
    let P_row_offsets: Vec<usize> = (0..=n).collect();
    let P_col_indices: Vec<usize> = (0..n).collect();
    let nnz_P = n;

    // A matrix: sparse random (m x n), with specified density
    let entries_per_row = std::cmp::max(1, std::cmp::min(n, (density * n as f64 + 0.5) as usize));
    let mut A_row_offsets: Vec<usize> = vec![0];
    let mut A_col_indices: Vec<usize> = Vec::new();

    for _ in 0..m {
        // Sample random columns
        let mut cols: Vec<usize> = (0..n).collect();
        // Fisher-Yates shuffle first entries_per_row elements
        for i in 0..entries_per_row {
            let j = rng.gen_range(i..n);
            cols.swap(i, j);
        }
        let mut selected: Vec<usize> = cols[..entries_per_row].to_vec();
        selected.sort();

        for col in selected {
            A_col_indices.push(col);
        }
        A_row_offsets.push(A_col_indices.len());
    }

    let nnz_A = A_col_indices.len();
    println!(
        "A matrix sparsity: {} nonzeros ({:.1}% dense)",
        nnz_A,
        100.0 * nnz_A as f64 / (n * m) as f64
    );

    // Cones: all nonnegative (simple box constraints)
    let cones: Vec<SupportedConeT<f64>> = vec![SupportedConeT::NonnegativeConeT(m)];

    // Settings
    let mut settings = DefaultSettings::default();
    settings.verbose = false;
    settings.max_iter = 50;

    println!("Creating solver...");

    let compiled_solver = CompiledSolver::new(
        n,
        m,
        &P_row_offsets,
        &P_col_indices,
        &A_row_offsets,
        &A_col_indices,
        &cones,
        settings,
        batch_size,
        false,
    )
    .expect("Failed to create batch solver");

    // Generate batch of problems with different values
    let mut problems = Vec::with_capacity(batch_size);

    for _ in 0..batch_size {
        // P values: random diagonal in [0.1, 2.0]
        let P_values: Vec<f64> = (0..nnz_P).map(|_| rng.gen_range(0.1..2.0)).collect();

        // q values: random in [-1, 1]
        let q: Vec<f64> = (0..n).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // A values: random in [0.1, 2.0]
        let A_values: Vec<f64> = (0..nnz_A).map(|_| rng.gen_range(0.1..2.0)).collect();

        // b values: random in [-2, 2]
        let b: Vec<f64> = (0..m).map(|_| rng.gen_range(-2.0..2.0)).collect();

        problems.push(BatchProblem {
            P_values,
            q,
            A_values,
            b,
        });
    }

    println!("\nSolving {} problems in parallel...", batch_size);

    let start = Instant::now();
    let solutions = compiled_solver
        .solve_batch_parallel(&problems)
        .expect("Batch solve failed");
    let elapsed = start.elapsed();

    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let per_problem_ms = total_ms / batch_size as f64;

    println!("\n=== BENCHMARK ===");
    println!("Total solve() time: {:.1} ms", total_ms);
    println!("Time per problem: {:.3} ms", per_problem_ms);
    println!("Throughput: {:.4} problems/second", 1000.0 / per_problem_ms);

    // Print first solution snippet
    println!("\n=== SOLUTION (first problem, first 10 values) ===");
    print!("x = [");
    for i in 0..std::cmp::min(10, n) {
        print!("{:.6}", solutions[0].x[i]);
        if i < 9 {
            print!(", ");
        }
    }
    println!(", ...]");

    // Count statuses
    let mut num_solved = 0;
    let mut num_almost = 0;
    let mut num_failed = 0;

    for sol in &solutions {
        match sol.status {
            moreau::solver::SolverStatus::Solved => num_solved += 1,
            moreau::solver::SolverStatus::AlmostSolved => num_almost += 1,
            _ => num_failed += 1,
        }
    }

    println!("\n=== BATCH RESULTS ===");
    println!("Solved: {} / {}", num_solved, batch_size);
    println!("Almost solved: {} / {}", num_almost, batch_size);
    println!("Failed: {} / {}", num_failed, batch_size);
}
