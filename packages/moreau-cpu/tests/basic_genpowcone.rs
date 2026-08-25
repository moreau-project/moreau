#![allow(non_snake_case)]

use moreau::{algebra::*, solver::*};

/// Helper to generate a random GenPowerCone problem with known solution.
///
/// Creates: minimize (1/2)||x||^2 + q'x  s.t. Ax + s = b, s ∈ GenPowerCone(alphas, dim2)
/// where q = -x_star and s_star is strictly in the cone interior.
fn gen_genpow_problem(
    n: usize,
    alphas: Vec<f64>,
    dim2: usize,
    seed: u64,
) -> (
    CscMatrix<f64>,           // P
    Vec<f64>,                 // q
    CscMatrix<f64>,           // A
    Vec<f64>,                 // b
    Vec<SupportedConeT<f64>>, // cones
    Vec<f64>,                 // x_star
) {
    let dim1 = alphas.len();
    let m = dim1 + dim2;

    // Simple deterministic pseudo-random using seed
    let mut state = seed;
    let mut next_f64 = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as f64) / (1u64 << 31) as f64 - 1.0
    };

    // x_star
    let x_star: Vec<f64> = (0..n).map(|_| next_f64() * 0.5).collect();
    let q: Vec<f64> = x_star.iter().map(|&xi| -xi).collect();

    // Random A (dense, m×n)
    let mut a_data = vec![0.0; m * n];
    for val in a_data.iter_mut() {
        *val = next_f64();
    }

    // Build A as CscMatrix from column-major
    let mut col_ptr = vec![0usize; n + 1];
    let mut row_idx = Vec::with_capacity(m * n);
    let mut vals = Vec::with_capacity(m * n);
    for j in 0..n {
        for i in 0..m {
            row_idx.push(i);
            vals.push(a_data[i * n + j]);
        }
        col_ptr[j + 1] = col_ptr[j] + m;
    }
    let A = CscMatrix::new(m, n, col_ptr, row_idx, vals);

    // s_star strictly in GenPowerCone
    let mut s_star = vec![0.0; m];
    for i in 0..dim1 {
        s_star[i] = 1.0 + (next_f64()).abs(); // p_i > 0
    }
    // Compute ∏ p_i^αi
    let prod: f64 = (0..dim1).map(|i| s_star[i].powf(alphas[i])).product();
    // w with ||w|| < prod * 0.3
    let max_norm = prod * 0.3;
    let mut w_norm_sq = 0.0;
    for i in 0..dim2 {
        s_star[dim1 + i] = next_f64();
        w_norm_sq += s_star[dim1 + i] * s_star[dim1 + i];
    }
    if w_norm_sq > 0.0 && dim2 > 0 {
        let scale = max_norm / (w_norm_sq.sqrt() + 1e-12);
        for i in 0..dim2 {
            s_star[dim1 + i] *= scale;
        }
    }

    // b = A*x_star + s_star
    let mut b = s_star.clone();
    for i in 0..m {
        for j in 0..n {
            b[i] += a_data[i * n + j] * x_star[j];
        }
    }

    let P = CscMatrix::<f64>::identity(n);
    let cones = vec![GenPowerConeT(alphas, dim2)];

    (P, q, A, b, cones, x_star)
}

#[test]
fn test_genpowcone_two_alphas_dim2_one() {
    // GenPowerCone with α = [0.6, 0.4], dim2 = 1
    // This is essentially the same as PowerCone(0.6) but via the generalized API.
    //
    // max z
    // s.t. (x, y, z) ∈ GenPowerCone([0.6, 0.4], 1)
    //      x + 2y == 3

    let n = 3;
    let P = CscMatrix::<f64>::zeros((n, n));
    let c = vec![0., 0., -1.]; // minimize -z => maximize z

    // (x, y, z) ∈ GenPowerCone([0.6, 0.4], 1)
    let mut A1 = CscMatrix::<f64>::identity(n);
    A1.negate();
    let b1 = vec![0.; n];
    let cones1 = vec![GenPowerConeT(vec![0.6, 0.4], 1)];

    // x + 2y == 3
    let A2 = CscMatrix::from(&[[1., 2., 0.]]);
    let b2 = vec![3.];
    let cones2 = vec![ZeroConeT(1)];

    let A = CscMatrix::vcat(&A1, &A2).unwrap();
    let b = [b1, b2].concat();
    let cones = [cones1, cones2].concat();

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let x = solver.solution.x[0];
    let y = solver.solution.x[1];
    let z = solver.solution.x[2];
    assert!(x > 0.0, "x should be positive, got {}", x);
    assert!(y > 0.0, "y should be positive, got {}", y);
    assert!(z > 0.0, "z should be positive, got {}", z);

    // Verify cone constraint at optimality
    let prod = x.powf(0.6) * y.powf(0.4);
    assert!(
        (prod - z).abs() < 1e-4,
        "Cone constraint should be active at optimality: prod={}, z={}",
        prod,
        z
    );
}

#[test]
fn test_genpowcone_three_alphas_dim2_two() {
    // GenPowerCone with α = [0.5, 0.3, 0.2], dim2 = 2
    // Known solution approach: q = -x_star, s_star strictly in cone interior
    //
    // minimize (1/2)||x||^2 + q'x
    // s.t. Ax + s = b, s ∈ GenPowerCone([0.5, 0.3, 0.2], 2)

    let n = 5; // match cone dimension
    let m = 5; // dim1(3) + dim2(2)

    let x_star = vec![1.0, -0.5, 0.3, 0.7, -0.2];
    let q: Vec<f64> = x_star.iter().map(|&xi| -xi).collect();

    // A = -I
    let mut A = CscMatrix::<f64>::identity(n);
    A.negate();

    // s_star strictly in GenPowerCone([0.5, 0.3, 0.2], 2):
    // p = (2, 3, 4), w = (0.3, 0.2)
    // prod = 2^0.5 * 3^0.3 * 4^0.2 ≈ 2.59, ||w|| ≈ 0.36 << 2.59
    let s_star = vec![2.0, 3.0, 4.0, 0.3, 0.2];

    // b = A*x_star + s_star = -x_star + s_star
    let b: Vec<f64> = (0..m).map(|i| -x_star[i] + s_star[i]).collect();

    let P = CscMatrix::<f64>::identity(n);
    let cones = vec![GenPowerConeT(vec![0.5, 0.3, 0.2], 2)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    for i in 0..n {
        assert!(
            (solver.solution.x[i] - x_star[i]).abs() < 1e-4,
            "x[{}]: expected {}, got {}",
            i,
            x_star[i],
            solver.solution.x[i]
        );
    }
}

#[test]
fn test_genpowcone_multiple_cones() {
    // Two GenPowerCones of different dimensions
    // Cone 1: GenPowerCone([0.6, 0.4], 1) — dim=3
    // Cone 2: GenPowerCone([0.5, 0.3, 0.2], 2) — dim=5
    // Total m = 8

    let n = 8;
    let m = 8;

    let x_star = vec![0.5, -0.3, 0.8, -0.2, 0.6, -0.1, 0.4, -0.7];
    let q: Vec<f64> = x_star.iter().map(|&xi| -xi).collect();

    // A = -I (8×8)
    let mut A = CscMatrix::<f64>::identity(m);
    A.negate();

    // s_star: cone 1 block (dim=3), cone 2 block (dim=5), both strictly interior
    // Cone 1: p=(2, 3), w=(0.5). prod = 2^0.6 * 3^0.4 ≈ 1.516*1.552 ≈ 2.35 >> 0.5
    // Cone 2: p=(2, 3, 4), w=(0.3, 0.2). prod ≈ 2.59, ||w|| ≈ 0.36
    let s_star = vec![2.0, 3.0, 0.5, 2.0, 3.0, 4.0, 0.3, 0.2];

    let b: Vec<f64> = (0..m).map(|i| -x_star[i] + s_star[i]).collect();

    let P = CscMatrix::<f64>::identity(n);
    let cones = vec![
        GenPowerConeT(vec![0.6, 0.4], 1),
        GenPowerConeT(vec![0.5, 0.3, 0.2], 2),
    ];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    for i in 0..n {
        assert!(
            (solver.solution.x[i] - x_star[i]).abs() < 1e-4,
            "x[{}]: expected {}, got {}",
            i,
            x_star[i],
            solver.solution.x[i]
        );
    }
}

#[test]
fn test_genpowcone_mixed_with_other_cones() {
    // Mix of ZeroCone, NonnegCone, and GenPowerCone
    // All with strictly feasible slack at the known solution.
    //
    // minimize (1/2)||x||^2 + c'x
    // s.t. Ax + s = b
    //   s[0..2] ∈ ZeroCone(2)
    //   s[2..4] ∈ NonnegCone(2)
    //   s[4..7] ∈ GenPowerCone([0.6, 0.4], 1)

    let n = 4;
    let m = 7; // 2 + 2 + 3
    let P = CscMatrix::<f64>::identity(n);

    // x_star chosen so constraint is slack
    let x_star = vec![0.3, 0.4, 0.2, 0.1];
    let c: Vec<f64> = x_star.iter().map(|&xi| -xi).collect();

    #[rustfmt::skip]
    let A_dense: [[f64; 4]; 7] = [
        // ZeroCone(2): equality
        [1., 0., 1., 0.],
        [0., 1., 0., 1.],
        // NonnegCone(2): inequality
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        // GenPowerCone([0.6, 0.4], 1)
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [1., 1., 0., 0.],
    ];

    let A = CscMatrix::from(&A_dense);

    // s_star: zero for ZeroCone, positive for NonnegCone, interior for GenPowerCone
    // GenPowerCone block: p=(2, 3), w=(0.3). prod = 2^0.6 * 3^0.4 ≈ 2.35 >> 0.3
    let s_star = vec![0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 0.3];

    // b = A*x_star + s_star
    let mut b = vec![0.0; m];
    for i in 0..m {
        for j in 0..n {
            b[i] += A_dense[i][j] * x_star[j];
        }
        b[i] += s_star[i];
    }

    let cones = vec![
        ZeroConeT(2),
        NonnegativeConeT(2),
        GenPowerConeT(vec![0.6, 0.4], 1),
    ];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert!(
        solver.solution.status == SolverStatus::Solved
            || solver.solution.status == SolverStatus::AlmostSolved,
        "Expected Solved or AlmostSolved, got {:?}",
        solver.solution.status
    );
}

#[test]
fn test_genpowcone_known_solution() {
    // Known solution with strictly interior slack.
    //
    // minimize (1/2)||x||^2 + q'x
    // s.t. Ax + s = b, s ∈ GenPowerCone([0.5, 0.3, 0.2], 1)

    let n = 6;
    let dim2 = 1;
    let m = 4; // dim1(3) + dim2(1)
    let alphas = vec![0.5, 0.3, 0.2];

    let x_star = vec![0.5, -0.3, 0.8, -0.2, 0.6, -0.1];
    let q: Vec<f64> = x_star.iter().map(|&xi| -xi).collect();

    #[rustfmt::skip]
    let A_dense: [[f64; 6]; 4] = [
        [ 1.0,  0.5, -0.3,  0.2, -0.1,  0.4],
        [ 0.3,  1.0,  0.2, -0.5,  0.1, -0.2],
        [-0.2,  0.4,  1.0,  0.3, -0.4,  0.1],
        [ 0.1, -0.3,  0.5,  1.0,  0.2, -0.3],
    ];

    let A = CscMatrix::from(&A_dense);

    // s_star strictly in GenPowerCone([0.5, 0.3, 0.2], 1)
    // p = (2, 3, 4), w = (0.5)
    // prod = 2^0.5 * 3^0.3 * 4^0.2 ≈ 2.59 >> 0.5
    let s_star = vec![2.0, 3.0, 4.0, 0.5];

    let mut b = vec![0.0; m];
    for i in 0..m {
        for j in 0..n {
            b[i] += A_dense[i][j] * x_star[j];
        }
        b[i] += s_star[i];
    }

    let P = CscMatrix::<f64>::identity(n);
    let cones = vec![GenPowerConeT(alphas, dim2)];

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    for i in 0..n {
        assert!(
            (solver.solution.x[i] - x_star[i]).abs() < 1e-4,
            "x[{}]: expected {}, got {}",
            i,
            x_star[i],
            solver.solution.x[i]
        );
    }
}

#[test]
fn test_genpowcone_high_dim_16() {
    // 10 alphas (uniform), dim2=6 => total dim=16
    let dim1 = 10;
    let dim2 = 6;
    let n = 20;
    let alphas: Vec<f64> = vec![1.0 / dim1 as f64; dim1];

    let (P, q, A, b, cones, x_star) = gen_genpow_problem(n, alphas, dim2, 42);

    let mut settings = DefaultSettings::default();
    settings.max_iter = 500;
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert!(
        solver.solution.status == SolverStatus::Solved
            || solver.solution.status == SolverStatus::AlmostSolved,
        "dim=16: expected Solved/AlmostSolved, got {:?}",
        solver.solution.status
    );

    for i in 0..n {
        assert!(
            (solver.solution.x[i] - x_star[i]).abs() < 1e-3,
            "dim=16 x[{}]: expected {}, got {}",
            i,
            x_star[i],
            solver.solution.x[i]
        );
    }
}

#[test]
fn test_genpowcone_high_dim_32() {
    // 20 alphas (uniform), dim2=12 => total dim=32
    let dim1 = 20;
    let dim2 = 12;
    let n = 40;
    let alphas: Vec<f64> = vec![1.0 / dim1 as f64; dim1];

    let (P, q, A, b, cones, x_star) = gen_genpow_problem(n, alphas, dim2, 123);

    let mut settings = DefaultSettings::default();
    settings.max_iter = 500;
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert!(
        solver.solution.status == SolverStatus::Solved
            || solver.solution.status == SolverStatus::AlmostSolved,
        "dim=32: expected Solved/AlmostSolved, got {:?}",
        solver.solution.status
    );

    for i in 0..n {
        assert!(
            (solver.solution.x[i] - x_star[i]).abs() < 1e-3,
            "dim=32 x[{}]: expected {}, got {}",
            i,
            x_star[i],
            solver.solution.x[i]
        );
    }
}

#[test]
fn test_genpowcone_high_dim_100() {
    // 60 alphas (uniform), dim2=40 => total dim=100
    // CPU only — CUDA caps at dim=32
    let dim1 = 60;
    let dim2 = 40;
    let n = 120;
    let alphas: Vec<f64> = vec![1.0 / dim1 as f64; dim1];

    let (P, q, A, b, cones, x_star) = gen_genpow_problem(n, alphas, dim2, 999);

    let mut settings = DefaultSettings::default();
    settings.max_iter = 1000;
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();

    assert!(
        solver.solution.status == SolverStatus::Solved
            || solver.solution.status == SolverStatus::AlmostSolved,
        "dim=100: expected Solved/AlmostSolved, got {:?}",
        solver.solution.status
    );

    for i in 0..n {
        assert!(
            (solver.solution.x[i] - x_star[i]).abs() < 1e-2,
            "dim=100 x[{}]: expected {}, got {}",
            i,
            x_star[i],
            solver.solution.x[i]
        );
    }
}
