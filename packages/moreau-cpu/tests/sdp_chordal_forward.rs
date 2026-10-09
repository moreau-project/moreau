//! Forward-correctness tests for chordal PSD decomposition.
//!
//! Solves the same problem with `chordal_decomposition_enable = true` and
//! `false`, asserts the solutions match. Decoupled from the backward-pass
//! tests in `sdp_chordal_backward.rs`, which only exercise this code path
//! transitively.

#![allow(non_snake_case)]
#![cfg(feature = "sdp")]

use moreau::solver::implementations::default::{CompiledSolver, DefaultSettings};
use moreau::solver::{SolverStatus, SupportedConeT};

const TOL: f64 = 1e-5;

fn solve(
    enable_chordal: bool,
    n: usize,
    m: usize,
    P_row_offsets: &[usize],
    P_col_indices: &[usize],
    P_values: &[f64],
    A_row_offsets: &[usize],
    A_col_indices: &[usize],
    A_values: &[f64],
    q: &[f64],
    b: &[f64],
    cones: &[SupportedConeT<f64>],
) -> (Vec<f64>, f64, SolverStatus) {
    solve_with_b_mask(
        enable_chordal,
        n,
        m,
        P_row_offsets,
        P_col_indices,
        P_values,
        A_row_offsets,
        A_col_indices,
        A_values,
        q,
        b,
        cones,
        None,
    )
}

/// `solve` with an explicit b sparsity pattern. Without one, the chordal
/// analysis assumes b is nonzero on every row where A is empty.
fn solve_with_b_mask(
    enable_chordal: bool,
    n: usize,
    m: usize,
    P_row_offsets: &[usize],
    P_col_indices: &[usize],
    P_values: &[f64],
    A_row_offsets: &[usize],
    A_col_indices: &[usize],
    A_values: &[f64],
    q: &[f64],
    b: &[f64],
    cones: &[SupportedConeT<f64>],
    b_nnz_mask: Option<&[bool]>,
) -> (Vec<f64>, f64, SolverStatus) {
    let mut settings = DefaultSettings::<f64>::default();
    settings.verbose = false;
    settings.ipm.chordal_decomposition_enable = enable_chordal;

    let mut solver = CompiledSolver::new_with_b_nnz_mask(
        n,
        m,
        P_row_offsets,
        P_col_indices,
        A_row_offsets,
        A_col_indices,
        cones,
        settings,
        1,
        false,
        b_nnz_mask,
    )
    .expect("construction failed");

    solver.setup(&[P_values.to_vec()], &[A_values.to_vec()]);
    let solutions = solver
        .solve(&[q.to_vec()], &[b.to_vec()])
        .expect("solve failed");

    let sol = &solutions[0];
    (sol.x.clone(), sol.obj_val, sol.status)
}

fn assert_match(
    label: &str,
    on: &(Vec<f64>, f64, SolverStatus),
    off: &(Vec<f64>, f64, SolverStatus),
) {
    let (x_on, obj_on, st_on) = on;
    let (x_off, obj_off, st_off) = off;

    assert_eq!(
        *st_on,
        SolverStatus::Solved,
        "{}: chordal-on did not solve ({:?})",
        label,
        st_on
    );
    assert_eq!(
        *st_off,
        SolverStatus::Solved,
        "{}: chordal-off did not solve ({:?})",
        label,
        st_off
    );

    assert_eq!(x_on.len(), x_off.len(), "{}: x dim mismatch", label);
    for i in 0..x_on.len() {
        let diff = (x_on[i] - x_off[i]).abs();
        assert!(
            diff < TOL,
            "{}: x[{}] mismatch: chordal-on={}, chordal-off={}, diff={}",
            label,
            i,
            x_on[i],
            x_off[i],
            diff
        );
    }

    let obj_diff = (obj_on - obj_off).abs();
    assert!(
        obj_diff < TOL,
        "{}: obj_val mismatch: chordal-on={}, chordal-off={}",
        label,
        obj_on,
        obj_off
    );
}

/// 4×4 block-diagonal PSD (two 2×2 blocks). Chordal decomposition should
/// split this into two PSD(2) cones; without chordal, the solver treats
/// it as a single dense PSD(4). Both should yield the same x*.
///
/// svec ordering (col-major upper-tri) for a 4×4 matrix:
///   indices 0,1,2,5,8,9 are in-block; 3,4,6,7 are cross-block (zero).
#[test]
fn chordal_on_off_match_2block_psd4() {
    // Problem: min 0.5||x||^2 + q'x  s.t.  x + s = b, s in PSD(4)
    // q encourages x[0,2,5,9] = 1 (the four diagonal svec entries),
    // b is svec of diag(5,5,5,5), so x* = e_diag is feasible (s* PD).
    let n = 10; // svec dim of 4×4
    let m = 10;

    // P = I
    let P_row_offsets: Vec<usize> = (0..=n).collect();
    let P_col_indices: Vec<usize> = (0..n).collect();
    let P_values = vec![1.0; n];

    // A = identity on in-block svec rows {0,1,2,5,8,9}, zero on cross-block {3,4,6,7}
    let in_block: [usize; 6] = [0, 1, 2, 5, 8, 9];
    let mut A_row_offsets = vec![0usize];
    let mut A_col_indices = Vec::new();
    let mut A_values = Vec::new();
    for r in 0..m {
        if in_block.contains(&r) {
            A_col_indices.push(r);
            A_values.push(1.0);
            A_row_offsets.push(A_row_offsets.last().unwrap() + 1);
        } else {
            A_row_offsets.push(*A_row_offsets.last().unwrap());
        }
    }

    // q = -e_diag: pushes x toward (1,0,1,0,0,1,0,0,0,1)
    let mut q = vec![0.0; n];
    for &i in &[0, 2, 5, 9] {
        q[i] = -1.0;
    }

    // b = svec(5·I_4): diagonal-only, in-block.
    let mut b = vec![0.0; m];
    for &i in &[0, 2, 5, 9] {
        b[i] = 5.0;
    }

    let cones = vec![SupportedConeT::PSDTriangleConeT(4)];

    let res_on = solve(
        true,
        n,
        m,
        &P_row_offsets,
        &P_col_indices,
        &P_values,
        &A_row_offsets,
        &A_col_indices,
        &A_values,
        &q,
        &b,
        &cones,
    );
    let res_off = solve(
        false,
        n,
        m,
        &P_row_offsets,
        &P_col_indices,
        &P_values,
        &A_row_offsets,
        &A_col_indices,
        &A_values,
        &q,
        &b,
        &cones,
    );

    assert_match("2block_psd4", &res_on, &res_off);

    // Sanity: x* should be roughly e_diag = (1,0,1,0,0,1,0,0,0,1)
    for &i in &[0, 2, 5, 9] {
        assert!(
            (res_on.0[i] - 1.0).abs() < 1e-3,
            "x[{}] = {} (want 1)",
            i,
            res_on.0[i]
        );
    }
}

/// Multi-PSD problem: two separate PSD(3) cones in the cone list.
/// Chordal flag should not affect the answer — each cone is already
/// at its smallest decomposable form.
#[test]
fn chordal_on_off_match_multi_psd3() {
    // Two PSD(3) cones, each with svec dim 6 → total m = 12.
    // Problem: min 0.5||x||^2 + q'x  s.t.  x + s = b, s in PSD(3) × PSD(3)
    let n = 12;
    let m = 12;

    let P_row_offsets: Vec<usize> = (0..=n).collect();
    let P_col_indices: Vec<usize> = (0..n).collect();
    let P_values = vec![1.0; n];

    // A = I (full identity)
    let A_row_offsets: Vec<usize> = (0..=m).collect();
    let A_col_indices: Vec<usize> = (0..m).collect();
    let A_values = vec![1.0; m];

    // q = -e_diag for both 3×3 blocks. svec(3×3) diag = indices 0, 2, 5.
    let mut q = vec![0.0; n];
    for &i in &[0, 2, 5, 6, 8, 11] {
        q[i] = -1.0;
    }

    // b = svec of 3·I_3 for both blocks
    let mut b = vec![0.0; m];
    for &i in &[0, 2, 5, 6, 8, 11] {
        b[i] = 3.0;
    }

    let cones = vec![
        SupportedConeT::PSDTriangleConeT(3),
        SupportedConeT::PSDTriangleConeT(3),
    ];

    let res_on = solve(
        true,
        n,
        m,
        &P_row_offsets,
        &P_col_indices,
        &P_values,
        &A_row_offsets,
        &A_col_indices,
        &A_values,
        &q,
        &b,
        &cones,
    );
    let res_off = solve(
        false,
        n,
        m,
        &P_row_offsets,
        &P_col_indices,
        &P_values,
        &A_row_offsets,
        &A_col_indices,
        &A_values,
        &q,
        &b,
        &cones,
    );

    assert_match("multi_psd3", &res_on, &res_off);
}

/// Mixed cones: one zero cone, one nonneg cone, one block-diagonal PSD
/// cone. Chordal should split the PSD; non-PSD cones unaffected.
#[test]
fn chordal_on_off_match_mixed_zero_nonneg_psd() {
    // Layout: 1 zero + 2 nonneg + PSD(4) (2-block) → m = 1 + 2 + 10 = 13
    // n = 13 (one variable per row, A = I-like)
    let n = 13;
    let m = 13;

    let P_row_offsets: Vec<usize> = (0..=n).collect();
    let P_col_indices: Vec<usize> = (0..n).collect();
    let P_values = vec![1.0; n];

    // A: identity on rows 0..3 (zero + nonneg), block-diag pattern on rows 3..13 (PSD)
    // PSD in-block rows are 3 + {0,1,2,5,8,9}, cross-block rows are 3 + {3,4,6,7}
    let psd_inblock: [usize; 6] = [0, 1, 2, 5, 8, 9];
    let mut A_row_offsets = vec![0usize];
    let mut A_col_indices = Vec::new();
    let mut A_values = Vec::new();
    for r in 0..m {
        let is_present = if r < 3 {
            true // zero + nonneg rows: identity
        } else {
            psd_inblock.contains(&(r - 3))
        };
        if is_present {
            A_col_indices.push(r);
            A_values.push(1.0);
            A_row_offsets.push(A_row_offsets.last().unwrap() + 1);
        } else {
            A_row_offsets.push(*A_row_offsets.last().unwrap());
        }
    }

    let mut q = vec![0.0; n];
    q[0] = 1.0; // pin equality to 0: x[0] = 0
    q[1] = -2.0;
    q[2] = -2.0; // push nonneg slacks active
                 // PSD diag indices in absolute terms: 3+0, 3+2, 3+5, 3+9
    for &i in &[3 + 0, 3 + 2, 3 + 5, 3 + 9] {
        q[i] = -1.0;
    }

    let mut b = vec![0.0; m];
    b[1] = 1.0;
    b[2] = 1.0;
    for &i in &[3 + 0, 3 + 2, 3 + 5, 3 + 9] {
        b[i] = 5.0;
    }

    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
        SupportedConeT::PSDTriangleConeT(4),
    ];

    let res_on = solve(
        true,
        n,
        m,
        &P_row_offsets,
        &P_col_indices,
        &P_values,
        &A_row_offsets,
        &A_col_indices,
        &A_values,
        &q,
        &b,
        &cones,
    );
    let res_off = solve(
        false,
        n,
        m,
        &P_row_offsets,
        &P_col_indices,
        &P_values,
        &A_row_offsets,
        &A_col_indices,
        &A_values,
        &q,
        &b,
        &cones,
    );

    assert_match("mixed_zero_nonneg_psd", &res_on, &res_off);
}

/// (A_row_offsets, A_col_indices, A_values, q, b, cones)
type AffinePsd = (
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
);

/// Conic data for `maximize x[n-1]` subject to `x[0] = t0` and
/// `X(x) = I + sum_c x[c] F_c ⪰ 0`, where `Fs[c]` lists the upper-triangle
/// entries `((i, j), v)`, i < j, of the zero-diagonal symmetric matrix F_c.
///
/// Rows: one zero cone, then svec(X) (upper triangle, column-major, √2 on
/// off-diagonals).
fn affine_psd(k: usize, t0: f64, Fs: &[&[((usize, usize), f64)]]) -> AffinePsd {
    let n = Fs.len();
    let m = 1 + k * (k + 1) / 2;
    let svec = |i: usize, j: usize| 1 + j * (j + 1) / 2 + i;

    let mut dense = vec![vec![0.0; n]; m];
    dense[0][0] = 1.0;
    for (c, F) in Fs.iter().enumerate() {
        for &((i, j), v) in F.iter() {
            dense[svec(i, j)][c] -= std::f64::consts::SQRT_2 * v;
        }
    }

    let mut A_row_offsets = vec![0usize];
    let mut A_col_indices = Vec::new();
    let mut A_values = Vec::new();
    for row in &dense {
        for (c, &v) in row.iter().enumerate() {
            if v != 0.0 {
                A_col_indices.push(c);
                A_values.push(v);
            }
        }
        A_row_offsets.push(A_col_indices.len());
    }

    let mut q = vec![0.0; n];
    q[n - 1] = -1.0;
    let mut b = vec![0.0; m];
    b[0] = t0;
    for i in 0..k {
        b[svec(i, i)] = 1.0;
    }

    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::PSDTriangleConeT(k),
    ];
    (A_row_offsets, A_col_indices, A_values, q, b, cones)
}

/// Solve an `affine_psd` problem with chordal on and off, check both
/// against each other and check the last variable against `s_opt`.
fn check_affine_psd(label: &str, k: usize, t0: f64, Fs: &[&[((usize, usize), f64)]], s_opt: f64) {
    let (A_row_offsets, A_col_indices, A_values, q, b, cones) = affine_psd(k, t0, Fs);
    let n = q.len();
    let m = b.len();
    let P_row_offsets = vec![0usize; n + 1];
    let b_nnz_mask: Vec<bool> = b.iter().map(|&v| v != 0.0).collect();

    let run = |chordal: bool| {
        solve_with_b_mask(
            chordal,
            n,
            m,
            &P_row_offsets,
            &[],
            &[],
            &A_row_offsets,
            &A_col_indices,
            &A_values,
            &q,
            &b,
            &cones,
            Some(&b_nnz_mask),
        )
    };
    let res_on = run(true);
    let res_off = run(false);

    assert_match(label, &res_on, &res_off);
    assert!(
        (res_on.0[n - 1] - s_opt).abs() < TOL,
        "{}: s = {} (want {})",
        label,
        res_on.0[n - 1],
        s_opt
    );
}

/// Tridiagonal PSD(4) in which t has unequal coefficients on (0,1) and
/// (1,2): X = [[1, t, 0, 0], [t, 1, 2t, 0], [0, 2t, 1, s], [0, 0, s, 1]],
/// maximize s with t = 0.3. det X = 0.55 - 0.91 s², so s* = √(55/91).
///
/// The decomposition reorders the rows of t's column in A. Filling A's
/// values in the original order exchanges t's two coefficients, which
/// gives s = √(55/64) at a point where X is not PSD.
#[test]
fn chordal_on_off_match_psd4_unequal_values_in_column() {
    let t_entries: [((usize, usize), f64); 2] = [((0, 1), 1.0), ((1, 2), 2.0)];
    let s_entries: [((usize, usize), f64); 1] = [((2, 3), 1.0)];
    check_affine_psd(
        "psd4_unequal_values_in_column",
        4,
        0.3,
        &[&t_entries, &s_entries],
        (55.0f64 / 91.0).sqrt(),
    );
}

/// PSD(3) with X = [[1, t+s, 0], [t+s, 1, 2t], [0, 2t, 1]], maximize s
/// with t = 0.3, so s* = 0.5. Exchanging t's coefficients gives the
/// feasible, suboptimal s = √0.91 - 0.6.
#[test]
fn chordal_on_off_match_psd3_unequal_values_in_column() {
    let t_entries: [((usize, usize), f64); 2] = [((0, 1), 1.0), ((1, 2), 2.0)];
    let s_entries: [((usize, usize), f64); 1] = [((0, 1), 1.0)];
    check_affine_psd(
        "psd3_unequal_values_in_column",
        3,
        0.3,
        &[&t_entries, &s_entries],
        0.5,
    );
}
