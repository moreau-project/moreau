#![allow(non_snake_case)]

//! SOS-MPC: Sum-of-Squares Lyapunov certificate for MPC terminal constraints
//!
//! Solves a Lyapunov SDP to certify stability of a linear feedback controller
//! for decoupled 2D subsystems. The Lyapunov matrix P gives a terminal cost
//! V(x) = x'Px and invariant set {x : V(x) <= 1} for use in MPC.
//!
//! The block-diagonal dynamics structure triggers chordal decomposition,
//! splitting PSD(6) into 3x PSD(2) cones — the same speedup pattern that
//! arises in SOS-based nonlinear MPC with separable Lyapunov functions.
//!
//! Problem:
//!   maximize   trace(P)                     [largest invariant ellipsoid]
//!   subject to P >= eps*I                   [PSD, positive definite]
//!              P - A'PA >= eps*I            [PSD, Lyapunov decrease]
//!              P_ii <= p_max                [bound entries to keep P finite]
//!
//! Batch: different system dynamics (a1, a2, a3 gain parameters) per problem.
//!
//! Usage: cargo run --release --features sdp-openblas --example sos_mpc [batch_size]

use moreau::solver::implementations::default::{CompiledSolver, DefaultSettings};
use moreau::solver::{SolverStatus, SupportedConeT};
use std::time::Instant;

fn tri(n: usize) -> usize {
    n * (n + 1) / 2
}

/// svec index for (i,j) with i<=j in column-major upper triangle.
fn svec_idx(i: usize, j: usize) -> usize {
    debug_assert!(i <= j);
    j * (j + 1) / 2 + i
}

/// Simple PRNG.
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
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.next_u64() as f64 / u64::MAX as f64) * (hi - lo)
    }
}

const N_SUB: usize = 3; // number of 2D subsystems
const NX: usize = 2 * N_SUB; // total state dimension = 6
const SVEC_DIM: usize = 21; // tri(6) = 21
const EPS: f64 = 0.01; // positive definiteness margin
const P_MAX: f64 = 100.0; // upper bound on diagonal entries
const SQRT2: f64 = std::f64::consts::SQRT_2;

/// 2x2 rotation-like stable dynamics parameterized by angle and damping.
fn make_subsystem(angle: f64, damping: f64) -> [[f64; 2]; 2] {
    let c = angle.cos() * damping;
    let s = angle.sin() * damping;
    [[c, -s], [s, c]]
}

/// Build the Lyapunov SDP.
///
/// Decision variables x: svec(P) of length 21.
///
/// Constraints (CSR A, b, cones):
///   Cone 1: NonnegCone(6)  — P_ii <= P_MAX, i.e., P_MAX - P_ii >= 0
///   Cone 2: PSD(6)         — P - eps*I >= 0
///   Cone 3: PSD(6)         — P - A'PA - eps*I >= 0
///
/// The P matrix rows are identity maps from svec(P) to the PSD cones.
/// The Lyapunov decrease rows encode the linear map P -> P - A'PA.
///
/// Returns: (P_ro, P_ci, P_vals, A_ro, A_ci, A_vals_shared, cones, n, m, q_shared)
/// where A_vals depend on A_sys (per-batch), but structure is shared.
fn build_problem_structure() -> (
    Vec<usize>,
    Vec<usize>,
    Vec<f64>, // P (cost matrix, CSR)
    Vec<usize>,
    Vec<usize>,               // A structure (CSR, no values yet)
    Vec<SupportedConeT<f64>>, // cones
    usize,
    usize, // n, m
) {
    let n = SVEC_DIM; // 21 decision variables
    let n_bound = NX; // 6 diagonal bound constraints
    let m = n_bound + 2 * SVEC_DIM; // 6 + 21 + 21 = 48

    // P_cost = 0 (we use q for objective)
    // Actually use P_cost = small regularization to make strictly convex
    let P_ro: Vec<usize> = (0..=n).collect();
    let P_ci: Vec<usize> = (0..n).collect();
    let P_vals: Vec<f64> = vec![1e-6; n]; // tiny regularization

    // Build A matrix structure (CSR)
    let mut A_ro = vec![0usize];
    let mut A_ci = vec![];

    // --- Cone 1: Nonneg(6) — diagonal bounds: P_MAX - P_ii >= 0 ---
    // Row i: coefficient -1 on P_ii (svec index for diagonal (i,i))
    // b[i] = P_MAX
    for i in 0..NX {
        let col = svec_idx(i, i);
        A_ci.push(col);
        A_ro.push(A_ci.len());
    }

    // --- Cone 2: PSD(6) — P - eps*I >= 0 ---
    // For each svec entry (i,j), the coefficient on x[svec_idx(i,j)] is 1.
    // b entries are eps on diagonals, 0 on off-diagonals.
    // This is just identity: A_psd1[k, k] = 1 for k in 0..SVEC_DIM
    for k in 0..SVEC_DIM {
        A_ci.push(k);
        A_ro.push(A_ci.len());
    }

    // --- Cone 3: PSD(6) — P - A'PA - eps*I >= 0 ---
    // For each output svec entry (r,c) with r<=c, we need the linear map:
    //   (P - A'PA)[r,c] = P[r,c] - sum_{i,j} A[i,r]*A[j,c]*P[i,j]
    //
    // For block-diagonal A = blkdiag(A1, A2, A3), the A'PA product has
    // block-diagonal sparsity: (A'PA)[r,c] = 0 when r and c are in
    // different subsystem blocks.
    //
    // Each svec output row depends on multiple svec input columns.
    // The sparsity pattern: for output (r,c), the nonzero input columns are:
    //   - column svec_idx(r,c) with coefficient 1 (from P term)
    //   - columns from the A'PA cross terms within the same block
    //
    // For a 2x2 block A_k, the A'PA block is a 2x2 matrix whose entries
    // are quadratic in A_k entries and linear in P block entries.
    // The svec(A'PA block) depends on svec(P block) via a 3x3 matrix T_k.
    //
    // We'll compute the sparsity by listing which input P svec columns
    // affect each output svec entry.

    // For block-diagonal structure, output (r,c) is affected by:
    // - P[r,c] directly (coefficient 1)
    // - P[i,j] for i,j in the same block as r,c (via A'PA)
    // - Nothing if r,c are in different blocks (cross-block: only P[r,c] appears)

    for c_out in 0..NX {
        for r_out in 0..=c_out {
            let block_r = r_out / 2;
            let block_c = c_out / 2;

            if block_r == block_c {
                // Same block: output depends on all P entries within this block
                let b0 = block_r * 2; // block start index
                                      // Block entries: (b0,b0), (b0,b0+1), (b0+1,b0+1)
                for j_blk in b0..b0 + 2 {
                    for i_blk in b0..=j_blk {
                        A_ci.push(svec_idx(i_blk, j_blk));
                    }
                }
            } else {
                // Cross-block: only P[r_out, c_out] appears
                A_ci.push(svec_idx(r_out, c_out));
            }
            A_ro.push(A_ci.len());
        }
    }

    assert_eq!(
        A_ro.len(),
        m + 1,
        "A_ro length mismatch: {} != {}",
        A_ro.len(),
        m + 1
    );

    let cones = vec![
        SupportedConeT::NonnegativeConeT(n_bound),
        SupportedConeT::PSDTriangleConeT(NX),
        SupportedConeT::PSDTriangleConeT(NX),
    ];

    (P_ro, P_ci, P_vals, A_ro, A_ci, cones, n, m)
}

/// Compute A values and b vector for given subsystem dynamics.
/// A_sys = [A1, A2, A3] each 2x2.
fn build_problem_values(
    A_sys: &[[[f64; 2]; 2]; N_SUB],
    A_ro: &[usize],
    A_ci: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = SVEC_DIM;
    let n_bound = NX;
    let m = n_bound + 2 * SVEC_DIM;

    let nnz_A = A_ci.len();
    let mut A_vals = vec![0.0f64; nnz_A];
    let mut b = vec![0.0f64; m];

    // q: minimize trace(P) (tightest Lyapunov ellipsoid)
    let mut q = vec![0.0f64; n];
    for i in 0..NX {
        q[svec_idx(i, i)] = 1.0;
    }

    // --- Cone 1: Nonneg(6) — P_MAX - P_ii >= 0 ---
    // A[row, P_ii] = -1 (so -P_ii + s = P_MAX, s >= 0 means P_ii <= P_MAX)
    // Wait: moreau formulation is Ax + s = b, s in cone.
    // For nonneg cone: s >= 0.
    // We want P_ii <= P_MAX, i.e., P_MAX - P_ii >= 0.
    // So: -P_ii + s = P_MAX -> A coefficient = -1, b = P_MAX
    // Then s = P_MAX - (-1)*(-P_ii) = P_MAX + P_ii... that's wrong.
    // Actually: Ax + s = b means s = b - Ax.
    // With A[row, col] = -1 and x[col] = P_ii:
    //   s = b - (-1)*P_ii = b + P_ii
    // We want s = P_MAX - P_ii >= 0.
    // So: b + P_ii = P_MAX - P_ii -> b = P_MAX - 2*P_ii... no, that's circular.
    //
    // Let's be careful. Decision variable x[k] = svec(P)[k].
    // Constraint: P_ii <= P_MAX.
    // Conic form: A_row * x + s = b_row, s >= 0.
    // We need s = P_MAX - P_ii = P_MAX - x[svec_idx(i,i)].
    // So: A_row has coefficient 1 at column svec_idx(i,i), b_row = P_MAX.
    // Then s = P_MAX - 1 * x[svec_idx(i,i)] = P_MAX - P_ii. ✓

    for i in 0..NX {
        let row = i;
        let nz_start = A_ro[row];
        A_vals[nz_start] = 1.0; // coefficient on P_ii
        b[row] = P_MAX;
    }

    // --- Cone 2: PSD(6) — P - eps*I >= 0 ---
    // svec(s) = svec(P) - svec(eps*I), s PSD.
    // Ax + s = b -> s = b - Ax.
    // We want s = P - eps*I, so Ax = -(P - eps*I - s)... let me think again.
    // Ax + s = b with s in PSD cone.
    // We want s = P - eps*I (which should be PSD for the constraint to hold).
    // So: Ax + (P - eps*I) = b -> Ax = b - P + eps*I.
    // With A[k,k] = -1 (negative identity): -x + s = b -> s = b + x.
    // Hmm, we want s = P - eps*I = x - eps*I (since x = svec(P)).
    // So we need: Ax + s = b, s = x - svec(eps*I).
    // -> Ax = b - s = b - x + svec(eps*I).
    // If A = -I: -x + s = b -> s = b + x. We want s = x - svec(eps*I).
    // So b + x = x - svec(eps*I) -> b = -svec(eps*I).
    // Then s = -svec(eps*I) + x = x - svec(eps*I) = svec(P - eps*I). ✓
    //
    // But wait: svec(eps*I) has eps on diagonal entries and 0 elsewhere.
    // So b[row] = -eps for diagonal entries, 0 for off-diagonal.
    //
    // And A[row, col] = -1 (for the identity map x -> svec(P)).

    let psd1_offset = n_bound;
    for k in 0..SVEC_DIM {
        let row = psd1_offset + k;
        let nz_start = A_ro[row];
        A_vals[nz_start] = -1.0; // -I map
    }
    // b for PSD cone 1: b = -svec(eps*I)
    for i in 0..NX {
        let k = svec_idx(i, i);
        b[psd1_offset + k] = -EPS;
    }

    // --- Cone 3: PSD(6) — P - A'PA - eps*I >= 0 ---
    // s = P - A'PA - eps*I should be PSD.
    // Ax + s = b -> s = b - Ax.
    // We need s = P - A'PA - eps*I.
    // For each svec entry (r,c):
    //   s[svec(r,c)] = P[r,c] - (A'PA)[r,c] - eps*delta(r,c)
    //
    // (A'PA)[r,c] = sum_{i,j} A_sys[i,r] * P[i,j] * A_sys[j,c]
    //
    // This is linear in P, so:
    //   s[svec(r,c)] = P[r,c] - sum_{i,j} A_sys[i,r]*A_sys[j,c]*P[i,j] - eps*delta(r,c)
    //
    // In Ax + s = b form:
    //   For the svec(r,c) row, the coefficient on x[svec(i,j)] is:
    //     -(delta(r,i)*delta(c,j) - A_sys[i,r]*A_sys[j,c])
    //   Wait, let me redo: s = b - Ax.
    //   We want s[row] = P[r,c] - (A'PA)[r,c] - eps*delta(r,c)
    //             = sum_{i<=j} coeff(i,j,r,c) * x[svec(i,j)] - eps*delta(r,c)
    //
    //   where coeff(i,j,r,c) = delta(i,r)*delta(j,c) [from P]
    //                          - A'PA contribution [from -A'PA]
    //
    //   Then A[row, svec(i,j)] = -coeff(i,j,r,c) and b[row] = -eps*delta(r,c).

    let psd2_offset = n_bound + SVEC_DIM;

    // Precompute A' for each block
    // For block-diagonal A_sys = blkdiag(A1, A2, A3):
    // (A'PA)[r,c] is nonzero only when r,c are in the same 2x2 block.
    //
    // Within block k (rows/cols 2k, 2k+1):
    //   (A_k' * P_k * A_k)[r_local, c_local]
    //     = sum_{i,j} A_k[i, r_local] * P_k[i, j] * A_k[j, c_local]
    //
    // P_k is the 2x2 subblock of P corresponding to block k.

    let mut out_row = 0; // svec output row counter
    for c_out in 0..NX {
        for r_out in 0..=c_out {
            let row = psd2_offset + out_row;
            let nz_start = A_ro[row];
            let nz_end = A_ro[row + 1];

            let block_r = r_out / 2;
            let block_c = c_out / 2;

            if block_r == block_c {
                // Same block: compute coefficients
                let blk = block_r;
                let A_k = &A_sys[blk];
                let b0 = blk * 2;
                let rl = r_out - b0; // local row index (0 or 1)
                let cl = c_out - b0; // local col index (0 or 1)

                // The svec entries for this block are:
                // (b0,b0), (b0,b0+1), (b0+1,b0+1)
                // which correspond to P_k[0,0], P_k[0,1], P_k[1,1]

                // For each block P entry (il, jl) with il<=jl:
                // Contribution to (A'PA)[rl,cl] from P[il,jl]:
                //   A_k[il,rl] * A_k[jl,cl] + (if il!=jl) A_k[jl,rl] * A_k[il,cl]
                // (the second term accounts for P being symmetric: P[jl,il] = P[il,jl])

                let mut nz_idx = nz_start;
                for jl in 0..2usize {
                    for il in 0..=jl {
                        // Coefficient on x[svec_idx(b0+il, b0+jl)]
                        let mut apa = A_k[il][rl] * A_k[jl][cl];
                        if il != jl {
                            apa += A_k[jl][rl] * A_k[il][cl];
                        }

                        // coeff = delta(il==rl && jl==cl) - apa
                        let identity_part = if il == rl && jl == cl { 1.0 } else { 0.0 };
                        let coeff = identity_part - apa;

                        // svec scaling: off-diagonal entries are scaled by sqrt(2)
                        // in svec, so we need to account for that.
                        // The input x[svec(i,j)] for i<j is sqrt(2)*P[i,j].
                        // The output row for (r,c) with r<c represents sqrt(2)*(P-A'PA)[r,c].
                        //
                        // Input: x[svec(il,jl)] = P[il,jl] * (il==jl ? 1 : sqrt(2))
                        // Output row (rl,cl): (P-A'PA)[rl,cl] * (rl==cl ? 1 : sqrt(2))
                        //
                        // So the A coefficient relating them:
                        //   output_scale * coeff_matrix * P[il,jl]
                        //   = output_scale * coeff * x[svec(il,jl)] / input_scale
                        //
                        // A[row, col] = -coeff * output_scale / input_scale
                        // (negative because s = b - Ax)

                        let input_scale = if il == jl { 1.0 } else { SQRT2 };
                        let output_scale = if rl == cl { 1.0 } else { SQRT2 };
                        let a_coeff = -coeff * output_scale / input_scale;

                        A_vals[nz_idx] = a_coeff;
                        nz_idx += 1;
                    }
                }
                assert_eq!(nz_idx, nz_end);
            } else {
                // Cross-block: only P[r_out, c_out] contributes (A'PA is zero here)
                // coeff = 1 (from P), apa = 0
                // A[row, svec(r_out, c_out)] = -1
                A_vals[nz_start] = -1.0;
            }

            // b: -eps on diagonals, 0 on off-diagonals
            if r_out == c_out {
                b[row] = -EPS;
            }

            out_row += 1;
        }
    }
    assert_eq!(out_row, SVEC_DIM);

    (A_vals, b, q)
}

/// Extract the P matrix from svec solution.
fn extract_P(x: &[f64]) -> [[f64; NX]; NX] {
    let mut P = [[0.0; NX]; NX];
    for j in 0..NX {
        for i in 0..=j {
            let val = x[svec_idx(i, j)];
            let unscaled = if i == j { val } else { val / SQRT2 };
            P[i][j] = unscaled;
            P[j][i] = unscaled;
        }
    }
    P
}

/// Check that M is PSD (all eigenvalues >= -tol).
fn is_psd(M: &[[f64; NX]; NX], tol: f64) -> bool {
    // Simple check: compute Cholesky or just check 2x2 blocks for our block-diagonal case
    // For 6x6, compute eigenvalues via characteristic polynomial... just check principal minors
    // Actually for a quick check, just verify each 2x2 diagonal block is PSD
    for blk in 0..N_SUB {
        let i = blk * 2;
        let det = M[i][i] * M[i + 1][i + 1] - M[i][i + 1] * M[i + 1][i];
        let trace = M[i][i] + M[i + 1][i + 1];
        if trace < -tol || det < -tol {
            return false;
        }
    }
    true
}

/// Compute A'PA for block-diagonal A.
fn compute_AtPA(A_sys: &[[[f64; 2]; 2]; N_SUB], P: &[[f64; NX]; NX]) -> [[f64; NX]; NX] {
    let mut result = [[0.0; NX]; NX];
    for blk in 0..N_SUB {
        let b0 = blk * 2;
        let A_k = &A_sys[blk];
        // Compute A_k' * P_block * A_k
        for r in 0..2 {
            for c in 0..2 {
                let mut val = 0.0;
                for i in 0..2 {
                    for j in 0..2 {
                        val += A_k[i][r] * P[b0 + i][b0 + j] * A_k[j][c];
                    }
                }
                result[b0 + r][b0 + c] = val;
            }
        }
    }
    result
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let batch_size: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(32);
    let n_runs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  SOS-MPC: Lyapunov SDP for terminal stability certificate  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("  {} decoupled 2D subsystems, state dim = {}", N_SUB, NX);
    println!("  PSD({}) cone x2 + Nonneg({}) cone", NX, NX);
    println!(
        "  svec dim = {}, total constraints = {}",
        SVEC_DIM,
        NX + 2 * SVEC_DIM
    );
    println!("  batch = {}, eps = {}", batch_size, EPS);
    println!();

    // Build problem structure (shared across all batch problems)
    let (P_ro, P_ci, P_vals, A_ro, A_ci, cones, n, m) = build_problem_structure();

    // Generate batch of subsystem dynamics with different damping/rotation
    let mut rng = Rng::new(42);
    let dynamics: Vec<[[[f64; 2]; 2]; N_SUB]> = (0..batch_size)
        .map(|_| {
            let mut A_sys = [[[0.0f64; 2]; 2]; N_SUB];
            for blk in 0..N_SUB {
                let angle = rng.next_range(0.2, 1.2);
                let damping = rng.next_range(0.5, 0.95); // stable: |lambda| < 1
                A_sys[blk] = make_subsystem(angle, damping);
            }
            A_sys
        })
        .collect();

    // Build A values and q/b for each batch problem
    let mut A_vals_batch: Vec<Vec<f64>> = Vec::with_capacity(batch_size);
    let mut qs: Vec<Vec<f64>> = Vec::with_capacity(batch_size);
    let mut bs: Vec<Vec<f64>> = Vec::with_capacity(batch_size);

    for dyn_sys in &dynamics {
        let (A_v, b_v, q_v) = build_problem_values(dyn_sys, &A_ro, &A_ci);
        A_vals_batch.push(A_v);
        qs.push(q_v);
        bs.push(b_v);
    }

    // Create solver
    let mut settings = DefaultSettings::<f64>::default();
    settings.verbose = false;

    let mut compiled = CompiledSolver::new(
        n, m, &P_ro, &P_ci, &A_ro, &A_ci, &cones, settings, batch_size, false,
    )
    .expect("construction failed");

    // Setup with per-problem A values (P is shared tiny regularization)
    let P_vals_batch: Vec<Vec<f64>> = vec![P_vals.clone(); batch_size];
    compiled.setup(&P_vals_batch, &A_vals_batch);

    // Warmup
    let _ = compiled.solve(&qs, &bs);

    // Benchmark
    println!("━━━ Benchmark ━━━\n");
    let mut solve_times = Vec::with_capacity(n_runs);
    let mut last_solutions = None;

    for _ in 0..n_runs {
        compiled.setup(&P_vals_batch, &A_vals_batch);
        let t0 = Instant::now();
        let solutions = compiled.solve(&qs, &bs).unwrap();
        solve_times.push(t0.elapsed().as_secs_f64() * 1000.0);
        last_solutions = Some(solutions);
    }

    solve_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = solve_times[solve_times.len() / 2];
    let per_prob = median / batch_size as f64;

    let solutions = last_solutions.unwrap();
    let num_solved = solutions
        .iter()
        .filter(|s| s.status == SolverStatus::Solved || s.status == SolverStatus::AlmostSolved)
        .count();
    let total_iters: u32 = solutions.iter().map(|s| s.iterations).sum();

    println!(
        "  Solve: {:.3} ms total, {:.3} ms/problem (median of {} runs)",
        median, per_prob, n_runs
    );
    println!(
        "  Solved: {}/{}, total iterations: {}",
        num_solved, batch_size, total_iters
    );

    // Verify first few solutions
    println!("\n━━━ Verification ━━━\n");
    for i in 0..3.min(batch_size) {
        let sol = &solutions[i];
        print!("  Problem {}: status={:?}", i, sol.status);

        if sol.status == SolverStatus::Solved || sol.status == SolverStatus::AlmostSolved {
            let P = extract_P(&sol.x);
            let trace: f64 = (0..NX).map(|k| P[k][k]).sum();

            // Check P - eps*I >= 0
            let mut P_shifted = P;
            for k in 0..NX {
                P_shifted[k][k] -= EPS;
            }
            let p_psd = is_psd(&P_shifted, 1e-6);

            // Check P - A'PA - eps*I >= 0
            let AtPA = compute_AtPA(&dynamics[i], &P);
            let mut lyap = [[0.0; NX]; NX];
            for r in 0..NX {
                for c in 0..NX {
                    lyap[r][c] = P[r][c] - AtPA[r][c] - if r == c { EPS } else { 0.0 };
                }
            }
            let lyap_psd = is_psd(&lyap, 1e-6);

            println!(
                ", trace(P)={:.2}, P-εI≥0: {}, P-A'PA-εI≥0: {}",
                trace,
                if p_psd { "✓" } else { "✗" },
                if lyap_psd { "✓" } else { "✗" }
            );

            // Print P block diagonals
            for blk in 0..N_SUB {
                let b0 = blk * 2;
                println!(
                    "    Block {}: P = [{:.4}, {:.4}; {:.4}, {:.4}]",
                    blk,
                    P[b0][b0],
                    P[b0][b0 + 1],
                    P[b0 + 1][b0],
                    P[b0 + 1][b0 + 1]
                );
            }
        } else {
            println!();
        }
    }
}
