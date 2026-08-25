#![allow(non_snake_case)]

//! Batch SDP with chordal decomposition example
//!
//! Demonstrates CompiledSolver with a PSD(6) cone where the A matrix has
//! block-diagonal sparsity that triggers chordal decomposition.
//!
//! The 6x6 PSD matrix has a 2-block structure: entries coupling block 0 (rows 0-2)
//! to block 1 (rows 3-5) are zero. Chordal decomposition detects this and splits
//! the single PSD(6) into two PSD(3) cones, which is more efficient to solve.
//!
//! Problem:
//!   minimize    (1/2)||x||^2 - 2*b'x
//!   subject to  x + s = b,  s in PSD(6)
//!
//! Optimal: x = b (unconstrained min at x = 2b, but s = b - x >= 0 in PSD sense
//! constrains x <= b; s = 0 at optimum).

use moreau::solver::implementations::default::{CompiledSolver, DefaultSettings};
use moreau::solver::SupportedConeT;

fn triangular_number(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Build svec of a scaled block-diagonal PD matrix: diag(scale_a * I, scale_b * I)
/// with small off-diagonal entries within each block.
fn make_block_diag_svec(block_dim: usize, scale_a: f64, scale_b: f64) -> Vec<f64> {
    let mat_dim = 2 * block_dim;
    let svec_dim = triangular_number(mat_dim);
    let mut svec = vec![0.0f64; svec_dim];

    let mut idx = 0;
    for j in 0..mat_dim {
        for i in 0..=j {
            let in_block_a = i < block_dim && j < block_dim;
            let in_block_b = i >= block_dim && j >= block_dim;
            if i == j {
                svec[idx] = if in_block_a { scale_a } else { scale_b };
            } else if in_block_a || in_block_b {
                svec[idx] = 0.1 * 2f64.sqrt(); // small off-diagonal (scaled for svec)
            }
            // cross-block entries remain 0
            idx += 1;
        }
    }
    svec
}

fn main() {
    let mat_dim = 6usize;
    let block_dim = mat_dim / 2; // 3
    let svec_dim = triangular_number(mat_dim); // 21
    let n = svec_dim;
    let m = svec_dim;
    let batch_size = 3;

    // P = I (identity, CSR, full symmetric)
    let P_row_offsets: Vec<usize> = (0..=n).collect();
    let P_col_indices: Vec<usize> = (0..n).collect();
    let P_values: Vec<f64> = vec![1.0; n];

    // A = block-diagonal sparse +I (positive identity on within-block entries):
    // Only include rows/columns corresponding to within-block svec entries.
    // Cross-block entries get empty rows in A (those components of s are free = 0).
    // This block-diagonal sparsity is what triggers chordal decomposition.
    let mut A_row_offsets = vec![0usize];
    let mut A_col_indices = vec![];
    let mut A_values_shared = vec![];

    let mut idx = 0;
    for j in 0..mat_dim {
        for i in 0..=j {
            let in_block_a = i < block_dim && j < block_dim;
            let in_block_b = i >= block_dim && j >= block_dim;
            if in_block_a || in_block_b {
                // Within-block entry: A maps x -> +x for this svec entry
                A_col_indices.push(idx);
                A_values_shared.push(1.0); // +I, not -I
                A_row_offsets.push(A_row_offsets.last().unwrap() + 1);
            } else {
                // Cross-block entry: empty row (s_i free, forced to 0 via b_i = 0)
                A_row_offsets.push(*A_row_offsets.last().unwrap());
            }
            idx += 1;
        }
    }
    assert_eq!(A_row_offsets.len(), m + 1);

    let nnz_A = A_col_indices.len();
    let nnz_A_dense = m; // I would have m nonzeros
    println!(
        "Block-diagonal sparsity: nnz(A) = {} / {} = {:.0}% of dense",
        nnz_A,
        nnz_A_dense,
        100.0 * nnz_A as f64 / nnz_A_dense as f64
    );
    println!(
        "Expected: PSD({}) -> 2x PSD({}) after chordal decomposition\n",
        mat_dim, block_dim
    );

    let cones: Vec<SupportedConeT<f64>> = vec![SupportedConeT::PSDTriangleConeT(mat_dim)];

    let mut settings = DefaultSettings::<f64>::default();
    settings.verbose = true;

    let mut compiled = CompiledSolver::new(
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
    .expect("CompiledSolver construction failed");

    // Build 3 batch problems with different block scales
    let scales = [(2.0, 3.0), (1.5, 4.0), (3.0, 1.0)];
    let bs: Vec<Vec<f64>> = scales
        .iter()
        .map(|(sa, sb)| make_block_diag_svec(block_dim, *sa, *sb))
        .collect();
    // q = -2*b so that unconstrained optimum is at x = 2*b, but PSD constraint
    // forces x = b at optimum (s = b - x = 0 at boundary of PSD cone).
    let qs: Vec<Vec<f64>> = bs
        .iter()
        .map(|b| b.iter().map(|v| -2.0 * v).collect())
        .collect();

    // Debug: print b vector for problem 0
    println!("b[0] (first few): {:?}", &bs[0][..6]);

    compiled.setup_shared(&P_values, &A_values_shared, batch_size);
    let solutions = compiled.solve(&qs, &bs).expect("solve failed");

    println!("\n=== Results ===");
    for (i, sol) in solutions.iter().enumerate() {
        let obj: f64 = sol
            .x
            .iter()
            .enumerate()
            .map(|(j, v)| 0.5 * v * v + qs[i][j] * v)
            .sum();
        println!("Problem {}: obj = {:.6}, status = {:?}", i, obj, sol.status);

        // At optimum, x should equal b (the PSD constraint is active: s = b - x = 0)
        let max_err = sol
            .x
            .iter()
            .zip(bs[i].iter())
            .map(|(x, b)| (x - b).abs())
            .fold(0.0f64, f64::max);
        println!("         max |x - b| = {:.2e}", max_err);

        // Print diagonal entries specifically
        for k in 0..mat_dim {
            let svec_diag = triangular_number(k + 1) - 1;
            println!(
                "         x[{}] = {:.6}, b[{}] = {:.6}",
                svec_diag, sol.x[svec_diag], svec_diag, bs[i][svec_diag]
            );
        }
    }
}
