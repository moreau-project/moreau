#![allow(non_snake_case)]

use moreau::{algebra::*, solver::*};

#[allow(clippy::type_complexity)]
fn basic_socp_data() -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    // P matrix data taken from corresponding Julia unit test.
    // These nzvals form a 3x3 positive definite matrix
    let nzval = vec![
        1.4652521089139698,
        0.6137176286085666,
        -1.1527861771130112,
        0.6137176286085666,
        2.219109946678485,
        -1.4400420548730628,
        -1.1527861771130112,
        -1.4400420548730628,
        1.6014483534926371,
    ];

    let P = CscMatrix::new(
        3,                               // m
        3,                               // n
        vec![0, 3, 6, 9],                // colptr
        vec![0, 1, 2, 0, 1, 2, 0, 1, 2], // rowval
        nzval,                           // nzval
    );

    // A = [2I;-2I;I]
    let I1 = CscMatrix::<f64>::identity(3);
    let mut I2 = CscMatrix::<f64>::identity(3);
    I2.negate();
    let mut A = CscMatrix::vcat(&I1, &I2).unwrap();
    A.scale(2.);
    let A = CscMatrix::vcat(&A, &I1).unwrap();

    let c = vec![0.1, -2.0, 1.0];
    let b = vec![1., 1., 1., 1., 1., 1., 0., 0., 0.];

    let cones = vec![
        NonnegativeConeT(3),
        NonnegativeConeT(3),
        SecondOrderConeT(3),
    ];

    (P, c, A, b, cones)
}

#[test]
fn test_socp_feasible() {
    let (P, c, A, b, cones) = basic_socp_data();

    let settings = DefaultSettings::<f64>::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let refsol = vec![-0.5, 0.435603, -0.245459];
    assert!(solver.solution.x.dist(&refsol) <= 1e-4);

    let refobj = -8.4590e-01;
    assert!(f64::abs(solver.solution.obj_val - refobj) <= 1e-4);
    assert!(f64::abs(solver.solution.obj_val_dual - refobj) <= 1e-4);
}

// Variable-dimension SOC tests: SOC dim >= 2 is now supported

/// Helper: build a problem with a single SOC(dim) cone.
/// minimize 0.5 * ||x||^2 + q'x  s.t.  -x in SOC(dim)
/// i.e. s = -Ax + b = x, s in SOC => x in SOC
#[allow(clippy::type_complexity)]
fn socp_variable_dim_data(
    dim: usize,
) -> (
    CscMatrix<f64>,
    Vec<f64>,
    CscMatrix<f64>,
    Vec<f64>,
    Vec<SupportedConeT<f64>>,
) {
    // P = I (dim x dim)
    let P = CscMatrix::<f64>::identity(dim);
    // A = -I
    let mut A = CscMatrix::<f64>::identity(dim);
    A.negate();
    // q: push towards interior of SOC
    let mut q = vec![0.0; dim];
    q[0] = -2.0; // minimize x[0], so x[0] will be positive and large
    let b = vec![0.0; dim];
    let cones = vec![SecondOrderConeT(dim)];
    (P, q, A, b, cones)
}

#[test]
fn test_socp_dim2() {
    let (P, q, A, b, cones) = socp_variable_dim_data(2);
    let settings = DefaultSettings::<f64>::default();
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    let x = &solver.solution.x;
    // Check SOC membership: x[0] >= ||x[1:]||
    let tail_norm: f64 = x[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(x[0] >= tail_norm - 1e-6, "SOC violated: x[0]={}, tail_norm={}", x[0], tail_norm);
}

#[test]
fn test_socp_dim5() {
    let (P, q, A, b, cones) = socp_variable_dim_data(5);
    let settings = DefaultSettings::<f64>::default();
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    let x = &solver.solution.x;
    let tail_norm: f64 = x[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(x[0] >= tail_norm - 1e-6, "SOC violated: x[0]={}, tail_norm={}", x[0], tail_norm);
}

#[test]
fn test_socp_dim10() {
    let (P, q, A, b, cones) = socp_variable_dim_data(10);
    let settings = DefaultSettings::<f64>::default();
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);
    let x = &solver.solution.x;
    let tail_norm: f64 = x[1..].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(x[0] >= tail_norm - 1e-6, "SOC violated: x[0]={}, tail_norm={}", x[0], tail_norm);
}

#[test]
fn test_socp_mixed_dims() {
    // Two SOC cones with different dimensions: SOC(2) and SOC(5)
    // n = 7, m = 7: s = -x (A = -I)
    let dim1 = 2;
    let dim2 = 5;
    let n = dim1 + dim2;

    let P = CscMatrix::<f64>::identity(n);
    let mut A = CscMatrix::<f64>::identity(n);
    A.negate();
    let mut q = vec![0.0; n];
    q[0] = -2.0; // push SOC(2) head
    q[dim1] = -3.0; // push SOC(5) head
    let b = vec![0.0; n];
    let cones = vec![SecondOrderConeT(dim1), SecondOrderConeT(dim2)];

    let settings = DefaultSettings::<f64>::default();
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let x = &solver.solution.x;
    // Check first SOC
    let tail1: f64 = x[1..dim1].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(x[0] >= tail1 - 1e-6, "SOC(2) violated: x[0]={}, tail={}", x[0], tail1);
    // Check second SOC
    let tail2: f64 = x[dim1+1..n].iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(x[dim1] >= tail2 - 1e-6, "SOC(5) violated: x[{}]={}, tail={}", dim1, x[dim1], tail2);
}

#[test]
fn test_socp_mixed_cones_with_variable_soc() {
    // 1 zero + 2 nonneg + SOC(4) + SOC(2)
    let dim_soc1 = 4;
    let dim_soc2 = 2;
    let n = 5;
    let m = 1 + 2 + dim_soc1 + dim_soc2; // 9

    // P = I (n x n)
    let P = CscMatrix::<f64>::identity(n);

    // A: [1,1,0,0,0; -1,0,0,0,0; 0,-1,0,0,0; 0,0,-I4; 0,0,0,0,-I2]
    // Build A as dense then convert
    let mut a_dense = vec![vec![0.0; n]; m];
    // Zero cone row: x0 + x1 = 1
    a_dense[0][0] = 1.0;
    a_dense[0][1] = 1.0;
    // Nonneg rows: -x0 >= 0, -x1 >= 0 (so x0 <= 0, x1 <= 0... wait that conflicts)
    // Actually nonneg cone: s >= 0 where s = Ax + b. So A*x + b >= 0
    // Let's do x0 >= 0, x1 >= 0: A rows = [-1,0,...; 0,-1,...], b = [0,0]
    a_dense[1][0] = -1.0;
    a_dense[2][1] = -1.0;
    // SOC(4): (x1, x2, x3, x4) mapped via -I
    for i in 0..dim_soc1 {
        a_dense[3 + i][1 + i] = -1.0;
    }
    // SOC(2): (x3, x4) mapped via -I
    for i in 0..dim_soc2 {
        a_dense[3 + dim_soc1 + i][3 + i] = -1.0;
    }

    // Convert to CSC
    let mut colptr = vec![0i64];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();
    for j in 0..n {
        for i in 0..m {
            if a_dense[i][j] != 0.0 {
                rowval.push(i);
                nzval.push(a_dense[i][j]);
            }
        }
        colptr.push(rowval.len() as i64);
    }
    let colptr: Vec<usize> = colptr.iter().map(|&v| v as usize).collect();
    let A = CscMatrix::new(m, n, colptr, rowval, nzval);

    let q = vec![1.0, 1.0, 0.5, 0.5, 0.5];
    let b = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    let cones = vec![
        ZeroConeT(1),
        NonnegativeConeT(2),
        SecondOrderConeT(dim_soc1),
        SecondOrderConeT(dim_soc2),
    ];

    let settings = DefaultSettings::<f64>::default();
    let mut solver = DefaultSolver::new(&P, &q, &A, &b, &cones, settings).unwrap();
    solver.solve();
    assert_eq!(solver.solution.status, SolverStatus::Solved);

    let x = &solver.solution.x;
    // Check equality: x0 + x1 ≈ 1
    assert!((x[0] + x[1] - 1.0).abs() < 1e-5, "Equality violated: {}", x[0] + x[1]);
}

#[test]
fn test_socp_infeasible() {
    let (P, c, A, mut b, cones) = basic_socp_data();

    //make the cone constraint unsatisfiable
    b[6] = -10.;

    let settings = DefaultSettings::default();

    let mut solver = DefaultSolver::new(&P, &c, &A, &b, &cones, settings).unwrap();

    solver.solve();

    assert_eq!(solver.solution.status, SolverStatus::PrimalInfeasible);
    assert!(solver.solution.obj_val.is_nan());
    assert!(solver.solution.obj_val_dual.is_nan());
}
