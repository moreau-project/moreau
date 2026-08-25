#![allow(non_snake_case)]

//! Benchmark: Riccati (block-tridiagonal) vs LDL on MPC-structured QPs
//!
//! Measures both total time (construction + solve) and solve-only time.
//!
//! Usage:
//!   cargo run --release --example bench_riccati

use moreau::algebra::CscMatrix;
use moreau::solver::{DefaultSettings, DefaultSolver, IPSolver, SolverStatus};
use std::time::Instant;

/// Build an MPC-structured QP for a double integrator.
fn build_mpc_qp(
    horizon: usize,
    nx: usize,
    nu: usize,
    dt: f64,
) -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<moreau::solver::SupportedConeT<f64>>,
) {
    use moreau::solver::SupportedConeT::*;

    let n_vars = horizon * (nx + nu) + nx;

    let mut p_rows = Vec::new();
    let mut p_cols = Vec::new();
    let mut p_vals = Vec::new();
    for t in 0..horizon {
        let x_off = t * (nx + nu);
        let u_off = x_off + nx;
        for i in 0..nx {
            p_rows.push(x_off + i);
            p_cols.push(x_off + i);
            p_vals.push(1.0);
        }
        for i in 0..nu {
            p_rows.push(u_off + i);
            p_cols.push(u_off + i);
            p_vals.push(0.1);
        }
    }
    let xt_off = horizon * (nx + nu);
    for i in 0..nx {
        p_rows.push(xt_off + i);
        p_cols.push(xt_off + i);
        p_vals.push(10.0);
    }
    let P = CscMatrix::new_from_triplets(n_vars, n_vars, p_rows, p_cols, p_vals);
    let q = vec![0.0; n_vars];

    let n_eq_init = nx;
    let n_eq_dyn = horizon * nx;
    let n_ineq = horizon * nu * 2;
    let n_con = n_eq_init + n_eq_dyn + n_ineq;

    let mut a_rows = Vec::new();
    let mut a_cols = Vec::new();
    let mut a_vals = Vec::new();
    let mut b = vec![0.0; n_con];

    let mut row = 0;
    for i in 0..nx {
        a_rows.push(row);
        a_cols.push(i);
        a_vals.push(1.0);
        b[row] = if i == 0 { 1.0 } else { 0.0 };
        row += 1;
    }

    let half = nx / 2;
    for t in 0..horizon {
        let x_off = t * (nx + nu);
        let u_off = x_off + nx;
        let x_next = x_off + nx + nu;

        for i in 0..half {
            a_rows.push(row);
            a_cols.push(x_off + i);
            a_vals.push(1.0);
            a_rows.push(row);
            a_cols.push(x_off + half + i);
            a_vals.push(dt);
            if i < nu {
                a_rows.push(row);
                a_cols.push(u_off + i);
                a_vals.push(0.5 * dt * dt);
            }
            a_rows.push(row);
            a_cols.push(x_next + i);
            a_vals.push(-1.0);
            row += 1;

            a_rows.push(row);
            a_cols.push(x_off + half + i);
            a_vals.push(1.0);
            if i < nu {
                a_rows.push(row);
                a_cols.push(u_off + i);
                a_vals.push(dt);
            }
            a_rows.push(row);
            a_cols.push(x_next + half + i);
            a_vals.push(-1.0);
            row += 1;
        }
    }

    for t in 0..horizon {
        let u_off = t * (nx + nu) + nx;
        for i in 0..nu {
            a_rows.push(row);
            a_cols.push(u_off + i);
            a_vals.push(1.0);
            b[row] = 1.0;
            row += 1;
            a_rows.push(row);
            a_cols.push(u_off + i);
            a_vals.push(-1.0);
            b[row] = 1.0;
            row += 1;
        }
    }
    assert_eq!(row, n_con);

    let A = CscMatrix::new_from_triplets(n_con, n_vars, a_rows, a_cols, a_vals);
    let cones = vec![ZeroConeT(n_eq_init + n_eq_dyn), NonnegativeConeT(n_ineq)];

    (P, q, A, b, cones)
}

struct BenchResult {
    total_ms: f64,
    solve_ms: f64,
    iters: u32,
    status: SolverStatus,
}

fn bench_solver(
    method: &str,
    P: &CscMatrix<f64>,
    q: &[f64],
    A: &CscMatrix<f64>,
    b: &[f64],
    cones: &[moreau::solver::SupportedConeT<f64>],
    n_runs: usize,
) -> BenchResult {
    let mut total_total = 0.0;
    let mut total_solve = 0.0;
    let mut iters: u32 = 0;
    let mut status = SolverStatus::Solved;

    for _ in 0..n_runs {
        let mut settings = DefaultSettings::default();
        settings.verbose = false;
        settings.ipm.direct_solve_method = method.to_string();

        let t0 = Instant::now();
        let mut solver = DefaultSolver::new(P, q, A, b, cones, settings).unwrap();
        solver.solve();
        total_total += t0.elapsed().as_secs_f64();
        total_solve += solver.solution.solve_time;
        iters = solver.solution.iterations;
        status = solver.solution.status;
    }

    BenchResult {
        total_ms: total_total / n_runs as f64 * 1000.0,
        solve_ms: total_solve / n_runs as f64 * 1000.0,
        iters,
        status,
    }
}

fn print_header() {
    println!(
        "{:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>8} {:>8} {:>8}",
        "T/nx", "n_vars", "ric(ms)", "ric_solv", "ldl(ms)", "ldl_solv", "speedup", "r_it", "l_it"
    );
}

fn print_row(label: usize, n_vars: usize, r: &BenchResult, l: &BenchResult) {
    let status_r = if r.status != SolverStatus::Solved {
        "!"
    } else {
        ""
    };
    let status_l = if l.status != SolverStatus::Solved {
        "!"
    } else {
        ""
    };
    println!(
        "{:>8} {:>8} {:>9.3}{} {:>10.3} {:>9.3}{} {:>10.3} {:>7.1}x {:>8} {:>8}",
        label,
        n_vars,
        r.total_ms,
        status_r,
        r.solve_ms,
        l.total_ms,
        status_l,
        l.solve_ms,
        l.solve_ms / r.solve_ms,
        r.iters,
        l.iters,
    );
}

fn main() {
    println!("Riccati vs QDLDL benchmark on MPC-structured QPs");
    println!("==================================================");
    println!("  ric(ms)    = total wall time (construction + solve)");
    println!("  ric_solv   = IPM solve time only (from solver internals)");
    println!("  speedup    = ldl_solv / ric_solv\n");

    let n_runs = 10;

    println!("── Varying horizon (nx=6, nu=3) ──");
    print_header();
    for &horizon in &[10, 20, 50, 100, 200, 500] {
        let nx = 6;
        let nu = 3;
        let (P, q, A, b, cones) = build_mpc_qp(horizon, nx, nu, 0.1);
        let n_vars = horizon * (nx + nu) + nx;

        let r = bench_solver("auto", &P, &q, &A, &b, &cones, n_runs);
        let l = bench_solver("qdldl", &P, &q, &A, &b, &cones, n_runs);
        print_row(horizon, n_vars, &r, &l);
    }

    println!();
    println!("── Varying state dim (T=50, nu=nx/2) ──");
    print_header();
    for &nx in &[4, 8, 12, 16, 24, 32] {
        let horizon = 50;
        let nu = nx / 2;
        let (P, q, A, b, cones) = build_mpc_qp(horizon, nx, nu, 0.1);
        let n_vars = horizon * (nx + nu) + nx;

        let r = bench_solver("auto", &P, &q, &A, &b, &cones, n_runs);
        let l = bench_solver("qdldl", &P, &q, &A, &b, &cones, n_runs);
        print_row(nx, n_vars, &r, &l);
    }
}
