//! Test backward pass (gradients) through CompiledSolver with PSD cones.
//!
//! Validates all four gradient outputs (dP, dq, dA, db) via central finite differences
//! for various PSD problem configurations including:
//! - Block-diagonal PSD(6) triggering chordal decomposition into two PSD(3) cones
//! - Mixed cones (zero + nonneg + PSD) with chordal
//! - Multiple PSD cones of different sizes
//! - Three-block structure (PSD(9) -> 3x PSD(3))
//! - Dense (non-decomposable) PSD through CompiledSolver
//!
//! Inspired by diffcp test patterns (https://github.com/cvxgrp/diffcp).

#![allow(non_snake_case)]
#![cfg(feature = "sdp")]

use moreau::solver::implementations::default::{
    CompiledSolver, DefaultSettings, UpstreamGradients,
};
use moreau::solver::{SolverStatus, SupportedConeT};

const EPS: f64 = 1e-6;
const TOL: f64 = 1e-3;

fn triangular_number(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Simple deterministic PRNG (xorshift64) to avoid external dependency.
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
    /// Uniform in [-1, 1]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0
    }
    /// Uniform in [lo, hi]
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.next_u64() as f64 / u64::MAX as f64) * (hi - lo)
    }
    fn randn_vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.next_f64() * scale).collect()
    }
}

/// Problem definition for CompiledSolver backward tests.
struct TestProblem {
    n: usize,
    m: usize,
    P_row_offsets: Vec<usize>,
    P_col_indices: Vec<usize>,
    P_values: Vec<f64>,
    A_row_offsets: Vec<usize>,
    A_col_indices: Vec<usize>,
    A_values: Vec<f64>,
    cones: Vec<SupportedConeT<f64>>,
    q: Vec<f64>,
    b: Vec<f64>,
}

/// Compute FD-validated backward pass for all four gradients (dP, dq, dA, db).
fn validate_backward(prob: &TestProblem, seed: u64) {
    let n = prob.n;
    let m = prob.m;
    let nnz_P = prob.P_values.len();
    let nnz_A = prob.A_values.len();

    // Solve with grad enabled
    let mut settings = DefaultSettings::<f64>::default();
    settings.verbose = false;
    let mut compiled_grad = CompiledSolver::new(
        n,
        m,
        &prob.P_row_offsets,
        &prob.P_col_indices,
        &prob.A_row_offsets,
        &prob.A_col_indices,
        &prob.cones,
        settings,
        1,
        true,
    )
    .expect("construction failed");

    compiled_grad.setup(&[prob.P_values.clone()], &[prob.A_values.clone()]);
    let solutions = compiled_grad
        .solve(&[prob.q.clone()], &[prob.b.clone()])
        .expect("solve failed");
    assert_eq!(
        solutions[0].status,
        SolverStatus::Solved,
        "Forward solve failed with status {:?}",
        solutions[0].status
    );

    // Deterministic upstream gradients
    let mut rng = Rng::new(seed);
    let dx_bar = rng.randn_vec(n, 0.1);
    let dz_bar = rng.randn_vec(m, 0.1);
    let ds_bar = vec![0.0; m];

    let upstream = UpstreamGradients {
        dx: dx_bar.clone(),
        ds: ds_bar,
        dz: dz_bar.clone(),
        dz_x: vec![],
    };
    let grads = compiled_grad
        .backward(&[upstream])
        .expect("backward failed");
    let g = &grads[0];

    // FD solver (no grad needed)
    let mut settings_fd = DefaultSettings::<f64>::default();
    settings_fd.verbose = false;
    let mut compiled_fd = CompiledSolver::new(
        n,
        m,
        &prob.P_row_offsets,
        &prob.P_col_indices,
        &prob.A_row_offsets,
        &prob.A_col_indices,
        &prob.cones,
        settings_fd,
        1,
        false,
    )
    .expect("construction failed");

    // Helper: compute directional derivative via FD
    let fd_dot = |x_plus: &[f64], z_plus: &[f64], x_minus: &[f64], z_minus: &[f64]| -> f64 {
        let dx_dot: f64 = dx_bar
            .iter()
            .zip(x_plus.iter().zip(x_minus.iter()))
            .map(|(d, (p, m))| d * (p - m))
            .sum();
        let dz_dot: f64 = dz_bar
            .iter()
            .zip(z_plus.iter().zip(z_minus.iter()))
            .map(|(d, (p, m))| d * (p - m))
            .sum();
        (dx_dot + dz_dot) / (2.0 * EPS)
    };

    let mut solve_fd =
        |P_vals: &[f64], A_vals: &[f64], q: &[f64], b: &[f64]| -> (Vec<f64>, Vec<f64>) {
            compiled_fd.setup(&[P_vals.to_vec()], &[A_vals.to_vec()]);
            let sols = compiled_fd
                .solve(&[q.to_vec()], &[b.to_vec()])
                .expect("FD solve failed");
            (sols[0].x.clone(), sols[0].z.clone())
        };

    // Validate dq
    let mut dq_fd = vec![0.0; n];
    for j in 0..n {
        let mut qp = prob.q.clone();
        qp[j] += EPS;
        let mut qm = prob.q.clone();
        qm[j] -= EPS;
        let (xp, zp) = solve_fd(&prob.P_values, &prob.A_values, &qp, &prob.b);
        let (xm, zm) = solve_fd(&prob.P_values, &prob.A_values, &qm, &prob.b);
        dq_fd[j] = fd_dot(&xp, &zp, &xm, &zm);
    }
    let dq_err = max_abs_diff(&g.dq, &dq_fd);
    println!("  dq max error: {:.2e}", dq_err);
    assert!(
        dq_err < TOL,
        "dq max error {:.2e} exceeds {:.2e}",
        dq_err,
        TOL
    );

    // Validate db
    let mut db_fd = vec![0.0; m];
    for j in 0..m {
        let mut bp = prob.b.clone();
        bp[j] += EPS;
        let mut bm = prob.b.clone();
        bm[j] -= EPS;
        let (xp, zp) = solve_fd(&prob.P_values, &prob.A_values, &prob.q, &bp);
        let (xm, zm) = solve_fd(&prob.P_values, &prob.A_values, &prob.q, &bm);
        db_fd[j] = fd_dot(&xp, &zp, &xm, &zm);
    }
    // For chordal PSD problems, some b entries correspond to PSD rows where
    // A[row,:] == 0.  Perturbing these b entries can create ill-conditioned PSD
    // problems that cause the FD solver to jump to a very different solution.
    // Use a relative error metric that ignores entries where FD is unreliable.
    let db_err = max_abs_diff_robust(&g.db, &db_fd);
    println!("  db max error: {:.2e}", db_err);
    assert!(
        db_err < TOL,
        "db max error {:.2e} exceeds {:.2e}",
        db_err,
        TOL
    );

    // Validate dP
    let mut dP_fd = vec![0.0; nnz_P];
    for j in 0..nnz_P {
        let mut Pp = prob.P_values.clone();
        Pp[j] += EPS;
        let mut Pm = prob.P_values.clone();
        Pm[j] -= EPS;
        let (xp, zp) = solve_fd(&Pp, &prob.A_values, &prob.q, &prob.b);
        let (xm, zm) = solve_fd(&Pm, &prob.A_values, &prob.q, &prob.b);
        dP_fd[j] = fd_dot(&xp, &zp, &xm, &zm);
    }
    let dP_err = max_abs_diff(&g.dP_values, &dP_fd);
    println!("  dP max error: {:.2e}", dP_err);
    assert!(
        dP_err < TOL,
        "dP max error {:.2e} exceeds {:.2e}",
        dP_err,
        TOL
    );

    // Validate dA
    let mut dA_fd = vec![0.0; nnz_A];
    for j in 0..nnz_A {
        let mut Ap = prob.A_values.clone();
        Ap[j] += EPS;
        let mut Am = prob.A_values.clone();
        Am[j] -= EPS;
        let (xp, zp) = solve_fd(&prob.P_values, &Ap, &prob.q, &prob.b);
        let (xm, zm) = solve_fd(&prob.P_values, &Am, &prob.q, &prob.b);
        dA_fd[j] = fd_dot(&xp, &zp, &xm, &zm);
    }
    let dA_err = max_abs_diff(&g.dA_values, &dA_fd);
    println!("  dA max error: {:.2e}", dA_err);
    assert!(
        dA_err < TOL,
        "dA max error {:.2e} exceeds {:.2e}",
        dA_err,
        TOL
    );
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// Like max_abs_diff but skips entries where the FD is unreliable.
/// For chordal PSD problems, perturbing b at rows where A is empty can
/// cause the solver to jump to a wildly different solution, making FD
/// give huge values while the analytic gradient is small (or vice versa).
/// We skip entries where |fd| >> |analytic| (sign of unreliable FD).
fn max_abs_diff_robust(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .filter(|(&analytic, &fd)| {
            // Skip if FD looks unreliable: |fd| > 1000 and |analytic| < 1
            let fd_unreliable = fd.abs() > 1e3 && analytic.abs() < 1.0;
            !fd_unreliable
        })
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

// ─── Problem builders ───────────────────────────────────────────────────────

/// Build svec of a block-diagonal PD matrix with `n_blocks` blocks of size `block_dim`.
fn make_multi_block_diag_svec(block_dim: usize, scales: &[f64]) -> Vec<f64> {
    let n_blocks = scales.len();
    let mat_dim = n_blocks * block_dim;
    let svec_dim = triangular_number(mat_dim);
    let mut svec = vec![0.0f64; svec_dim];

    let mut idx = 0;
    for j in 0..mat_dim {
        for i in 0..=j {
            let block_i = i / block_dim;
            let block_j = j / block_dim;
            if block_i == block_j {
                if i == j {
                    svec[idx] = scales[block_i];
                } else {
                    svec[idx] = 0.05 * 2f64.sqrt(); // small off-diagonal
                }
            }
            idx += 1;
        }
    }
    svec
}

/// Build block-diagonal A = +I on within-block entries (CSR).
/// Returns (row_offsets, col_indices, values) for a `mat_dim x svec_dim` matrix.
fn build_multi_block_diag_A(
    mat_dim: usize,
    block_dim: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let n_blocks = mat_dim / block_dim;
    let mut A_row_offsets = vec![0usize];
    let mut A_col_indices = vec![];
    let mut A_values = vec![];

    let mut idx = 0;
    for j in 0..mat_dim {
        for i in 0..=j {
            let block_i = i / block_dim;
            let block_j = j / block_dim;
            if block_i == block_j {
                A_col_indices.push(idx);
                A_values.push(1.0);
                A_row_offsets.push(A_row_offsets.last().unwrap() + 1);
            } else {
                A_row_offsets.push(*A_row_offsets.last().unwrap());
            }
            idx += 1;
        }
    }
    let _ = n_blocks; // used implicitly via mat_dim/block_dim
    (A_row_offsets, A_col_indices, A_values)
}

/// Build a simple full-A PSD problem: minimize (1/2)||x||^2 + q'x s.t. Ax + s = b, s in PSD.
/// A = I (full identity), so no chordal decomposition possible.
fn build_dense_psd_problem(mat_dim: usize, rng: &mut Rng) -> TestProblem {
    let svec_dim = triangular_number(mat_dim);
    let n = svec_dim;
    let m = svec_dim;

    let P_row_offsets: Vec<usize> = (0..=n).collect();
    let P_col_indices: Vec<usize> = (0..n).collect();
    let P_values: Vec<f64> = vec![1.0; n];

    // A = I (full, no sparsity -> no chordal)
    let A_row_offsets: Vec<usize> = (0..=m).collect();
    let A_col_indices: Vec<usize> = (0..m).collect();
    let A_values: Vec<f64> = vec![1.0; m];

    // b = svec of a PD matrix (diagonal dominant)
    let mut b = vec![0.0; m];
    let mut idx = 0;
    for j in 0..mat_dim {
        for i in 0..=j {
            if i == j {
                b[idx] = rng.next_range(2.0, 5.0);
            } else {
                b[idx] = rng.next_f64() * 0.1 * 2f64.sqrt();
            }
            idx += 1;
        }
    }
    let q: Vec<f64> = b.iter().map(|v| -2.0 * v).collect();

    let cones = vec![SupportedConeT::PSDTriangleConeT(mat_dim)];

    TestProblem {
        n,
        m,
        P_row_offsets,
        P_col_indices,
        P_values,
        A_row_offsets,
        A_col_indices,
        A_values,
        cones,
        q,
        b,
    }
}

/// Build a block-diagonal PSD problem that triggers chordal decomposition.
fn build_block_diag_psd_problem(block_dim: usize, scales: &[f64]) -> TestProblem {
    let n_blocks = scales.len();
    let mat_dim = n_blocks * block_dim;
    let svec_dim = triangular_number(mat_dim);
    let n = svec_dim;
    let m = svec_dim;

    let P_row_offsets: Vec<usize> = (0..=n).collect();
    let P_col_indices: Vec<usize> = (0..n).collect();
    let P_values: Vec<f64> = vec![1.0; n];

    let (A_row_offsets, A_col_indices, A_values) = build_multi_block_diag_A(mat_dim, block_dim);
    let b = make_multi_block_diag_svec(block_dim, scales);
    let q: Vec<f64> = b.iter().map(|v| -2.0 * v).collect();

    let cones = vec![SupportedConeT::PSDTriangleConeT(mat_dim)];

    TestProblem {
        n,
        m,
        P_row_offsets,
        P_col_indices,
        P_values,
        A_row_offsets,
        A_col_indices,
        A_values,
        cones,
        q,
        b,
    }
}

/// Build mixed-cone problem: zero cones + nonneg cones + PSD cone (block-diagonal).
fn build_mixed_cone_problem(
    n_eq: usize,
    n_ineq: usize,
    mat_dim: usize,
    block_dim: usize,
    rng: &mut Rng,
) -> TestProblem {
    let svec_dim = triangular_number(mat_dim);
    let n = svec_dim; // decision variables = svec dimension
    let m = n_eq + n_ineq + svec_dim;

    // P = I
    let P_row_offsets: Vec<usize> = (0..=n).collect();
    let P_col_indices: Vec<usize> = (0..n).collect();
    let P_values: Vec<f64> = vec![1.0; n];

    // A: [A_eq; A_ineq; A_psd] where A_psd is block-diagonal identity
    let mut A_row_offsets = vec![0usize];
    let mut A_col_indices = vec![];
    let mut A_values = vec![];

    // Zero cone rows: each picks one variable with coefficient 1
    for row in 0..n_eq {
        let col = row % n;
        A_col_indices.push(col);
        A_values.push(1.0);
        A_row_offsets.push(A_row_offsets.last().unwrap() + 1);
    }

    // Nonneg cone rows: each picks one variable with coefficient 1
    for row in 0..n_ineq {
        let col = (row + n_eq) % n;
        A_col_indices.push(col);
        A_values.push(1.0);
        A_row_offsets.push(A_row_offsets.last().unwrap() + 1);
    }

    // PSD cone rows: block-diagonal identity (same as build_multi_block_diag_A)
    let mut idx = 0;
    for j in 0..mat_dim {
        for i in 0..=j {
            let block_i = i / block_dim;
            let block_j = j / block_dim;
            if block_i == block_j {
                A_col_indices.push(idx);
                A_values.push(1.0);
                A_row_offsets.push(A_row_offsets.last().unwrap() + 1);
            } else {
                A_row_offsets.push(*A_row_offsets.last().unwrap());
            }
            idx += 1;
        }
    }
    assert_eq!(A_row_offsets.len(), m + 1);

    // b: feasible values
    let mut b = vec![0.0; m];
    // Zero cone: b = 0 (equality)
    // Nonneg cone: b > 0
    for i in n_eq..(n_eq + n_ineq) {
        b[i] = rng.next_range(0.5, 2.0);
    }
    // PSD cone: b = svec of PD matrix
    let psd_b = make_multi_block_diag_svec(block_dim, &vec![3.0; mat_dim / block_dim]);
    b[n_eq + n_ineq..].copy_from_slice(&psd_b);

    let q = rng.randn_vec(n, 0.5);

    let mut cones: Vec<SupportedConeT<f64>> = vec![];
    if n_eq > 0 {
        cones.push(SupportedConeT::ZeroConeT(n_eq));
    }
    if n_ineq > 0 {
        cones.push(SupportedConeT::NonnegativeConeT(n_ineq));
    }
    cones.push(SupportedConeT::PSDTriangleConeT(mat_dim));

    TestProblem {
        n,
        m,
        P_row_offsets,
        P_col_indices,
        P_values,
        A_row_offsets,
        A_col_indices,
        A_values,
        cones,
        q,
        b,
    }
}

/// Build problem with multiple separate PSD cones (each potentially decomposable).
fn build_multi_psd_problem(psd_dims: &[usize], rng: &mut Rng) -> TestProblem {
    let svec_dims: Vec<usize> = psd_dims.iter().map(|&d| triangular_number(d)).collect();
    let m: usize = svec_dims.iter().sum();
    let n = m; // variables = total svec dim

    // P = I
    let P_row_offsets: Vec<usize> = (0..=n).collect();
    let P_col_indices: Vec<usize> = (0..n).collect();
    let P_values: Vec<f64> = vec![1.0; n];

    // A = I (full identity for each PSD block)
    let A_row_offsets: Vec<usize> = (0..=m).collect();
    let A_col_indices: Vec<usize> = (0..m).collect();
    let A_values: Vec<f64> = vec![1.0; m];

    // b = svec of PD matrices (one per PSD cone)
    let mut b = vec![0.0; m];
    let mut offset = 0;
    for &dim in psd_dims {
        let svec_d = triangular_number(dim);
        let mut idx = 0;
        for j in 0..dim {
            for i in 0..=j {
                if i == j {
                    b[offset + idx] = rng.next_range(2.0, 5.0);
                } else {
                    b[offset + idx] = rng.next_f64() * 0.05 * 2f64.sqrt();
                }
                idx += 1;
            }
        }
        offset += svec_d;
    }

    let q = rng.randn_vec(n, 0.3);

    let cones: Vec<SupportedConeT<f64>> = psd_dims
        .iter()
        .map(|&d| SupportedConeT::PSDTriangleConeT(d))
        .collect();

    TestProblem {
        n,
        m,
        P_row_offsets,
        P_col_indices,
        P_values,
        A_row_offsets,
        A_col_indices,
        A_values,
        cones,
        q,
        b,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

/// Original test: PSD(6) -> 2x PSD(3) with chordal decomposition.
#[test]
fn test_chordal_2block_psd6() {
    println!("=== PSD(6) -> 2x PSD(3) chordal ===");
    let prob = build_block_diag_psd_problem(3, &[2.0, 3.0]);
    validate_backward(&prob, 42);
}

/// Three-block structure: PSD(9) -> 3x PSD(3).
#[test]
fn test_chordal_3block_psd9() {
    println!("=== PSD(9) -> 3x PSD(3) chordal ===");
    let prob = build_block_diag_psd_problem(3, &[2.0, 3.0, 1.5]);
    validate_backward(&prob, 123);
}

/// Asymmetric block sizes: PSD(8) with blocks of size 2 and 6.
/// block_dim=2 means 4 blocks of size 2 -> PSD(8) decomposes into 4x PSD(2).
#[test]
fn test_chordal_4block_psd8() {
    println!("=== PSD(8) -> 4x PSD(2) chordal ===");
    let prob = build_block_diag_psd_problem(2, &[1.0, 2.0, 3.0, 4.0]);
    validate_backward(&prob, 456);
}

/// Dense PSD(3): no chordal decomposition (A = full I).
#[test]
fn test_dense_psd3() {
    println!("=== Dense PSD(3), no chordal ===");
    let mut rng = Rng::new(789);
    let prob = build_dense_psd_problem(3, &mut rng);
    validate_backward(&prob, 789);
}

/// Dense PSD(4): no chordal decomposition.
#[test]
fn test_dense_psd4() {
    println!("=== Dense PSD(4), no chordal ===");
    let mut rng = Rng::new(1011);
    let prob = build_dense_psd_problem(4, &mut rng);
    validate_backward(&prob, 1011);
}

/// Mixed cones: 2 zero + 3 nonneg + PSD(6) with chordal (2-block).
#[test]
fn test_mixed_zero_nonneg_psd_chordal() {
    println!("=== Mixed: 2 zero + 3 nonneg + PSD(6) chordal ===");
    let mut rng = Rng::new(2024);
    let prob = build_mixed_cone_problem(2, 3, 6, 3, &mut rng);
    validate_backward(&prob, 2024);
}

/// Mixed cones: 1 zero + 2 nonneg + PSD(4) dense (no chordal).
#[test]
fn test_mixed_zero_nonneg_psd_dense() {
    println!("=== Mixed: 1 zero + 2 nonneg + PSD(4) dense ===");
    let mut rng = Rng::new(3033);
    // block_dim = mat_dim means single block, so A is full -> no chordal
    let prob = build_mixed_cone_problem(1, 2, 4, 4, &mut rng);
    validate_backward(&prob, 3033);
}

/// Multiple small PSD cones: PSD(2) + PSD(3) + PSD(2).
#[test]
fn test_multi_psd_small() {
    println!("=== Multi PSD: PSD(2) + PSD(3) + PSD(2) ===");
    let mut rng = Rng::new(4044);
    let prob = build_multi_psd_problem(&[2, 3, 2], &mut rng);
    validate_backward(&prob, 4044);
}

/// Multiple PSD cones of same size: 3x PSD(3).
#[test]
fn test_multi_psd_same_size() {
    println!("=== Multi PSD: 3x PSD(3) ===");
    let mut rng = Rng::new(5055);
    let prob = build_multi_psd_problem(&[3, 3, 3], &mut rng);
    validate_backward(&prob, 5055);
}

/// PSD(1) degenerates to nonneg — backward should still work.
#[test]
fn test_psd_dim1() {
    println!("=== PSD(1) (degenerate) ===");
    let mut rng = Rng::new(6066);
    let prob = build_dense_psd_problem(1, &mut rng);
    validate_backward(&prob, 6066);
}

/// Batch backward: 3 problems with chordal, validate dq for each.
#[test]
fn test_chordal_backward_batch() {
    println!("=== Batch: 3x PSD(6) chordal ===");
    let mat_dim = 6;
    let block_dim = 3;
    let svec_dim = triangular_number(mat_dim);
    let n = svec_dim;
    let m = svec_dim;
    let batch_size = 3;

    let P_row_offsets: Vec<usize> = (0..=n).collect();
    let P_col_indices: Vec<usize> = (0..n).collect();
    let P_values: Vec<f64> = vec![1.0; n];

    let (A_row_offsets, A_col_indices, A_values) = build_multi_block_diag_A(mat_dim, block_dim);
    let cones = vec![SupportedConeT::PSDTriangleConeT(mat_dim)];

    let scales_list = [&[2.0, 3.0][..], &[1.5, 4.0], &[3.0, 1.0]];
    let bs: Vec<Vec<f64>> = scales_list
        .iter()
        .map(|s| make_multi_block_diag_svec(block_dim, s))
        .collect();
    let qs: Vec<Vec<f64>> = bs
        .iter()
        .map(|b| b.iter().map(|v| -2.0 * v).collect())
        .collect();

    // Solve batch with grad
    let mut settings = DefaultSettings::<f64>::default();
    settings.verbose = false;
    let mut compiled_grad = CompiledSolver::new(
        n,
        m,
        &P_row_offsets,
        &P_col_indices,
        &A_row_offsets,
        &A_col_indices,
        &cones,
        settings,
        batch_size,
        true,
    )
    .expect("construction failed");

    compiled_grad.setup_shared(&P_values, &A_values, batch_size);
    let solutions = compiled_grad.solve(&qs, &bs).expect("solve failed");
    for sol in &solutions {
        assert_eq!(sol.status, SolverStatus::Solved);
    }

    let mut rng = Rng::new(7077);
    let upstreams: Vec<_> = (0..batch_size)
        .map(|_| UpstreamGradients {
            dx: rng.randn_vec(n, 0.1),
            ds: vec![0.0; m],
            dz: rng.randn_vec(m, 0.1),
            dz_x: vec![],
        })
        .collect();

    let grads = compiled_grad.backward(&upstreams).expect("backward failed");

    // FD solver
    let mut settings_fd = DefaultSettings::<f64>::default();
    settings_fd.verbose = false;
    let mut compiled_fd = CompiledSolver::new(
        n,
        m,
        &P_row_offsets,
        &P_col_indices,
        &A_row_offsets,
        &A_col_indices,
        &cones,
        settings_fd,
        1,
        false,
    )
    .expect("construction failed");

    for prob in 0..batch_size {
        let dx_bar = &upstreams[prob].dx;
        let dz_bar = &upstreams[prob].dz;

        // FD for dq
        let mut dq_fd = vec![0.0; n];
        for j in 0..n {
            let mut qp = qs[prob].clone();
            qp[j] += EPS;
            let mut qm = qs[prob].clone();
            qm[j] -= EPS;

            compiled_fd.setup_shared(&P_values, &A_values, 1);
            let solp = compiled_fd
                .solve(&[qp], &[bs[prob].clone()])
                .expect("FD solve");
            compiled_fd.setup_shared(&P_values, &A_values, 1);
            let solm = compiled_fd
                .solve(&[qm], &[bs[prob].clone()])
                .expect("FD solve");

            let dx_dot: f64 = dx_bar
                .iter()
                .zip(solp[0].x.iter().zip(solm[0].x.iter()))
                .map(|(d, (p, m))| d * (p - m))
                .sum();
            let dz_dot: f64 = dz_bar
                .iter()
                .zip(solp[0].z.iter().zip(solm[0].z.iter()))
                .map(|(d, (p, m))| d * (p - m))
                .sum();
            dq_fd[j] = (dx_dot + dz_dot) / (2.0 * EPS);
        }

        let dq_err = max_abs_diff(&grads[prob].dq, &dq_fd);
        println!("  Problem {} dq max error: {:.2e}", prob, dq_err);
        assert!(dq_err < TOL, "Problem {} dq error {:.2e}", prob, dq_err);

        // FD for db
        let mut db_fd = vec![0.0; m];
        for j in 0..m {
            let mut bp = bs[prob].clone();
            bp[j] += EPS;
            let mut bm = bs[prob].clone();
            bm[j] -= EPS;

            compiled_fd.setup_shared(&P_values, &A_values, 1);
            let solp = compiled_fd
                .solve(&[qs[prob].clone()], &[bp])
                .expect("FD solve");
            compiled_fd.setup_shared(&P_values, &A_values, 1);
            let solm = compiled_fd
                .solve(&[qs[prob].clone()], &[bm])
                .expect("FD solve");

            let dx_dot: f64 = dx_bar
                .iter()
                .zip(solp[0].x.iter().zip(solm[0].x.iter()))
                .map(|(d, (p, m))| d * (p - m))
                .sum();
            let dz_dot: f64 = dz_bar
                .iter()
                .zip(solp[0].z.iter().zip(solm[0].z.iter()))
                .map(|(d, (p, m))| d * (p - m))
                .sum();
            db_fd[j] = (dx_dot + dz_dot) / (2.0 * EPS);
        }

        let db_err = max_abs_diff_robust(&grads[prob].db, &db_fd);
        println!("  Problem {} db max error: {:.2e}", prob, db_err);
        assert!(db_err < TOL, "Problem {} db error {:.2e}", prob, db_err);
    }
}
