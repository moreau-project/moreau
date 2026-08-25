//! Integration tests for the differentiation KKT systems.

use super::*;

/// Convert dense matrix to CSC format (full matrix) - only used in tests
#[cfg(test)]
fn dense_to_csc<T: FloatT>(dense: &[T], nrows: usize, ncols: usize) -> CscMatrix<T> {
    let mut col_ptr = vec![0usize; ncols + 1];
    let mut row_idx = Vec::new();
    let mut values = Vec::new();

    for j in 0..ncols {
        col_ptr[j] = row_idx.len();
        for i in 0..nrows {
            let val = dense[i * ncols + j];
            if val.abs() > T::epsilon() {
                row_idx.push(i);
                values.push(val);
            }
        }
    }
    col_ptr[ncols] = row_idx.len();

    CscMatrix {
        m: nrows,
        n: ncols,
        colptr: col_ptr,
        rowval: row_idx,
        nzval: values,
    }
}

/// Convert dense symmetric matrix to upper triangular CSC format (for QDLDL) - only used in tests
#[cfg(test)]
fn dense_to_csc_upper<T: FloatT>(dense: &[T], dim: usize) -> CscMatrix<T> {
    let mut col_ptr = vec![0usize; dim + 1];
    let mut row_idx = Vec::new();
    let mut values = Vec::new();

    for j in 0..dim {
        col_ptr[j] = row_idx.len();
        // Only include upper triangular part (i <= j)
        for i in 0..=j {
            let val = dense[i * dim + j];
            if val.abs() > T::epsilon() {
                row_idx.push(i);
                values.push(val);
            }
        }
    }
    col_ptr[dim] = row_idx.len();

    CscMatrix {
        m: dim,
        n: dim,
        colptr: col_ptr,
        rowval: row_idx,
        nzval: values,
    }
}

// ========================================================================
// Test-only dense construction functions (for comparison tests)
// ========================================================================

/// Build KKT matrix using dense construction - test only
fn build_qp_kkt_dense(P: &CscMatrix<f64>, A: &CscMatrix<f64>, reg: f64) -> CscMatrix<f64> {
    let n = P.nrows();
    let m = A.nrows();
    let dim = n + m;

    let mut dense = vec![0.0; dim * dim];

    // Upper-left: P
    for j in 0..n {
        for k in P.colptr[j]..P.colptr[j + 1] {
            let i = P.rowval[k];
            dense[i * dim + j] = P.nzval[k];
            dense[j * dim + i] = P.nzval[k];
        }
    }

    // Upper-right: A'
    for j in 0..n {
        for k in A.colptr[j]..A.colptr[j + 1] {
            let i = A.rowval[k];
            dense[j * dim + (n + i)] = A.nzval[k];
            dense[(n + i) * dim + j] = A.nzval[k];
        }
    }

    // Lower-right: -reg * I
    for i in 0..m {
        dense[(n + i) * dim + (n + i)] = -reg;
    }

    dense_to_csc(&dense, dim, dim)
}

/// Build QP KKT matrix (upper triangular) - test only
fn build_qp_kkt_dense_upper(P: &CscMatrix<f64>, A: &CscMatrix<f64>, reg: f64) -> CscMatrix<f64> {
    let n = P.nrows();
    let m = A.nrows();
    let dim = n + m;

    let mut dense = vec![0.0; dim * dim];

    // Upper-left: P + reg*I
    for j in 0..n {
        for k in P.colptr[j]..P.colptr[j + 1] {
            let i = P.rowval[k];
            dense[i * dim + j] = P.nzval[k];
            if i != j {
                dense[j * dim + i] = P.nzval[k];
            }
        }
        dense[j * dim + j] = dense[j * dim + j] + reg;
    }

    // Upper-right: A'
    for j in 0..n {
        for k in A.colptr[j]..A.colptr[j + 1] {
            let i = A.rowval[k];
            dense[j * dim + (n + i)] = A.nzval[k];
            dense[(n + i) * dim + j] = A.nzval[k];
        }
    }

    // Lower-right: -reg*I
    for i in 0..m {
        dense[(n + i) * dim + (n + i)] = -reg;
    }

    dense_to_csc_upper(&dense, dim)
}

// ========================================================================
// Dense to CSC conversion tests
// ========================================================================

#[test]
fn test_dense_to_csc() {
    let dense = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0];

    let csc = dense_to_csc(&dense, 3, 3);

    assert_eq!(csc.nrows(), 3);
    assert_eq!(csc.ncols(), 3);
    assert_eq!(csc.nnz(), 5);
}

#[test]
fn test_dense_to_csc_identity() {
    let dense: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    let csc = dense_to_csc(&dense, 3, 3);

    assert_eq!(csc.nnz(), 3);
    // Check column pointers
    assert_eq!(csc.colptr, vec![0, 1, 2, 3]);
    // Check row indices
    assert_eq!(csc.rowval, vec![0, 1, 2]);
    // Check values
    assert!((csc.nzval[0] - 1.0).abs() < 1e-10);
    assert!((csc.nzval[1] - 1.0).abs() < 1e-10);
    assert!((csc.nzval[2] - 1.0).abs() < 1e-10);
}

#[test]
fn test_dense_to_csc_full() {
    let dense: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

    let csc = dense_to_csc(&dense, 2, 2);

    assert_eq!(csc.nnz(), 4);
    assert_eq!(csc.colptr, vec![0, 2, 4]);
}

#[test]
fn test_dense_to_csc_empty() {
    let dense: Vec<f64> = vec![0.0; 9];

    let csc = dense_to_csc(&dense, 3, 3);

    assert_eq!(csc.nnz(), 0);
    assert_eq!(csc.colptr, vec![0, 0, 0, 0]);
}

#[test]
fn test_dense_to_csc_rectangular() {
    let dense: Vec<f64> = vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0];

    let csc = dense_to_csc(&dense, 2, 3);

    assert_eq!(csc.nrows(), 2);
    assert_eq!(csc.ncols(), 3);
    assert_eq!(csc.nnz(), 3);
}

// ========================================================================
// QP KKT matrix construction tests
// ========================================================================

#[test]
fn test_build_qp_kkt_dense_simple() {
    // P = [1 0; 0 2], A = [1 1]
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![1.0, 2.0],
    };

    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };

    let kkt = build_qp_kkt_dense(&P, &A, 1e-8);

    // KKT matrix should be 3x3
    assert_eq!(kkt.nrows(), 3);
    assert_eq!(kkt.ncols(), 3);
}

#[test]
fn test_build_qp_kkt_dense_symmetry() {
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 2, 3],
        rowval: vec![0, 1, 1],
        nzval: vec![2.0, 1.0, 3.0], // P = [2 1; 1 3]
    };

    let A = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 1],
        nzval: vec![1.0, 0.0, 0.0, 1.0],
    };

    let kkt = build_qp_kkt_dense(&P, &A, 1e-8);

    // Convert back to dense to check symmetry
    let dim = kkt.nrows();
    let mut dense: Vec<f64> = vec![0.0; dim * dim];
    for j in 0..dim {
        for k in kkt.colptr[j]..kkt.colptr[j + 1] {
            let i = kkt.rowval[k];
            dense[i * dim + j] = kkt.nzval[k];
        }
    }

    // Check symmetry
    for i in 0..dim {
        for j in 0..dim {
            assert!(
                (dense[i * dim + j] - dense[j * dim + i]).abs() < 1e-10,
                "KKT[{},{}] = {} != {} = KKT[{},{}]",
                i,
                j,
                dense[i * dim + j],
                dense[j * dim + i],
                j,
                i
            );
        }
    }
}

// ========================================================================
// Linear system solver tests
// ========================================================================

#[test]
fn test_solve_linear_system_identity() {
    // Solve I * x = b
    let identity: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let kkt = dense_to_csc(&identity, 3, 3);
    let rhs: Vec<f64> = vec![1.0, 2.0, 3.0];

    let x = solve_linear_system(&kkt, &rhs);

    for i in 0..3 {
        assert!((x[i] - rhs[i]).abs() < 1e-10);
    }
}

#[test]
fn test_solve_linear_system_diagonal() {
    // Solve D * x = b where D = diag(2, 3, 4)
    let diag: Vec<f64> = vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
    let kkt = dense_to_csc(&diag, 3, 3);
    let rhs: Vec<f64> = vec![2.0, 6.0, 8.0];

    let x = solve_linear_system(&kkt, &rhs);

    assert!((x[0] - 1.0).abs() < 1e-10);
    assert!((x[1] - 2.0).abs() < 1e-10);
    assert!((x[2] - 2.0).abs() < 1e-10);
}

#[test]
fn test_solve_linear_system_symmetric() {
    // Solve K * x = b where K is SPD
    // QDLDL needs upper triangular format
    let mat: Vec<f64> = vec![4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0];
    let kkt = dense_to_csc_upper(&mat, 3);
    let rhs: Vec<f64> = vec![5.0, 6.0, 5.0];

    let x = solve_linear_system(&kkt, &rhs);

    // Verify K * x = b
    let mut residual = vec![0.0; 3];
    residual[0] = 4.0 * x[0] + 1.0 * x[1] - rhs[0];
    residual[1] = 1.0 * x[0] + 4.0 * x[1] + 1.0 * x[2] - rhs[1];
    residual[2] = 1.0 * x[1] + 4.0 * x[2] - rhs[2];

    for i in 0..3 {
        assert!(
            residual[i].abs() < 1e-10,
            "residual[{}] = {}",
            i,
            residual[i]
        );
    }
}

// ========================================================================
// QP forward differentiation tests
// ========================================================================

#[test]
fn test_differentiate_qp_eq_simple() {
    // Simple QP: min (1/2)x'Px + q'x s.t. Ax = b
    // P = I, q = 0, A = [1 1], b = [2]
    // Solution: x = [1, 1], z = 0

    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![1.0, 1.0],
    };
    let q: Vec<f64> = vec![0.0, 0.0];
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b: Vec<f64> = vec![2.0];

    let x: Vec<f64> = vec![1.0, 1.0];
    let z: Vec<f64> = vec![0.0];

    // Debug: print the KKT matrix
    let reg: f64 = REG_GENERAL;
    let kkt = build_qp_kkt_dense_upper(&P, &A, reg);
    println!("KKT matrix (upper triangular CSC):");
    println!("  colptr: {:?}", kkt.colptr);
    println!("  rowval: {:?}", kkt.rowval);
    println!("  nzval:  {:?}", kkt.nzval);

    // Reconstruct to dense for inspection
    let dim = kkt.nrows();
    let mut dense: Vec<f64> = vec![0.0; dim * dim];
    for j in 0..dim {
        for k in kkt.colptr[j]..kkt.colptr[j + 1] {
            let i = kkt.rowval[k];
            dense[i * dim + j] = kkt.nzval[k];
            dense[j * dim + i] = kkt.nzval[k];
        }
    }
    println!("Reconstructed KKT matrix:");
    for i in 0..dim {
        for j in 0..dim {
            print!("{:12.2e} ", dense[i * dim + j]);
        }
        println!();
    }

    // Perturb q
    let dP = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let dq: Vec<f64> = vec![1.0, 0.0];
    let dA = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let db: Vec<f64> = vec![0.0];

    let result = differentiate_qp_eq(&P, &q, &A, &b, &x, &z, &dP, &dq, &dA, &db);

    println!("Result: dx = {:?}, dz = {:?}", result.dx, result.dz);

    // dx should be nonzero due to perturbation in q
    // The constraint Ax = b (sum = 2) must still hold: dx[0] + dx[1] = 0
    assert!(
        (result.dx[0] + result.dx[1]).abs() < 1e-6,
        "dx[0] + dx[1] = {} should be 0",
        result.dx[0] + result.dx[1]
    );

    // ds should be zero for equality constraints
    assert!(result.ds.iter().all(|&x| x.abs() < 1e-10));
}

#[test]
fn test_differentiate_qp_eq_perturb_b() {
    // Same QP, perturb b
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![1.0, 1.0],
    };
    let q: Vec<f64> = vec![0.0, 0.0];
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b: Vec<f64> = vec![2.0];

    let x: Vec<f64> = vec![1.0, 1.0];
    let z: Vec<f64> = vec![0.0];

    let dP = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let dq: Vec<f64> = vec![0.0, 0.0];
    let dA = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let db: Vec<f64> = vec![1.0]; // Increase constraint by 1

    let result = differentiate_qp_eq(&P, &q, &A, &b, &x, &z, &dP, &dq, &dA, &db);

    // New constraint: x[0] + x[1] = 3, so each should increase by 0.5
    assert!(
        (result.dx[0] - 0.5).abs() < 1e-6,
        "dx[0] = {}",
        result.dx[0]
    );
    assert!(
        (result.dx[1] - 0.5).abs() < 1e-6,
        "dx[1] = {}",
        result.dx[1]
    );
}

// ========================================================================
// QP adjoint differentiation tests
// ========================================================================

#[test]
fn test_differentiate_adjoint_qp_eq_simple() {
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![1.0, 1.0],
    };
    let q: Vec<f64> = vec![0.0, 0.0];
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b: Vec<f64> = vec![2.0];

    let x: Vec<f64> = vec![1.0, 1.0];
    let z: Vec<f64> = vec![0.0];

    // Gradient w.r.t. x[0]
    let dx_bar: Vec<f64> = vec![1.0, 0.0];
    let dz_bar: Vec<f64> = vec![0.0];
    let ds_bar: Vec<f64> = vec![0.0];

    let result = differentiate_adjoint_qp_eq(&P, &q, &A, &b, &x, &z, &dx_bar, &dz_bar, &ds_bar);

    // Check that gradients have correct sparsity pattern
    assert_eq!(result.dP.nnz(), P.nnz());
    assert_eq!(result.dA.nnz(), A.nnz());
    assert_eq!(result.dq.len(), 2);
    assert_eq!(result.db.len(), 1);
}

// ========================================================================
// Gradient computation helper tests
// ========================================================================

#[test]
fn test_compute_gradient_P() {
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![1.0, 2.0],
    };
    let lam_x: Vec<f64> = vec![1.0, 2.0];
    let x: Vec<f64> = vec![3.0, 4.0];

    let dP = compute_gradient_P(&P, &lam_x, &x);

    assert_eq!(dP.nnz(), 2);
    // dP[i,j] = -lam_x[i] * x[j]
    // dP[0,0] = -lam_x[0] * x[0] = -1 * 3 = -3
    assert!((dP.nzval[0] + 3.0).abs() < 1e-10);
    // dP[1,1] = -lam_x[1] * x[1] = -2 * 4 = -8
    assert!((dP.nzval[1] + 8.0).abs() < 1e-10);
}

#[test]
fn test_compute_gradient_A() {
    let A = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 1],
        nzval: vec![1.0, 2.0, 3.0, 4.0],
    };
    let lam_x: Vec<f64> = vec![1.0, 2.0];
    let lam_z: Vec<f64> = vec![3.0, 4.0];
    let x: Vec<f64> = vec![5.0, 6.0];
    let z: Vec<f64> = vec![7.0, 8.0];

    let dA = compute_gradient_A(&A, &lam_x, &lam_z, &x, &z);

    assert_eq!(dA.nnz(), 4);
    // Each entry: dA[i,j] = -lam_z[i] * x[j] - lam_x[j] * z[i]
    // dA[0,0] = -3 * 5 - 1 * 7 = -15 - 7 = -22
    assert!((dA.nzval[0] + 22.0).abs() < 1e-10);
    // dA[1,0] = -4 * 5 - 1 * 8 = -20 - 8 = -28
    assert!((dA.nzval[1] + 28.0).abs() < 1e-10);
}

// ========================================================================
// HSDE differentiation tests
// ========================================================================

#[test]
fn test_differentiate_hsde_nonneg() {
    // Simple LP with nonnegative constraints
    // min q'x s.t. Ax + s = b, s >= 0
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let q: Vec<f64> = vec![1.0, 1.0];
    let A = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 1],
        nzval: vec![1.0, 0.0, 0.0, 1.0],
    };
    let b: Vec<f64> = vec![1.0, 1.0];

    let x: Vec<f64> = vec![0.0, 0.0]; // At bounds
    let s: Vec<f64> = vec![1.0, 1.0];
    let z: Vec<f64> = vec![1.0, 1.0];
    let tau = 1.0;

    let cones = vec![SupportedConeT::NonnegativeConeT(2)];

    let dP = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let dq: Vec<f64> = vec![0.1, 0.0];
    let dA = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let db: Vec<f64> = vec![0.0, 0.0];

    let result = differentiate_hsde(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Exact,
        0.0,
    );

    // Result should have correct dimensions
    assert_eq!(result.dx.len(), 2);
    assert_eq!(result.dz.len(), 2);
    assert_eq!(result.ds.len(), 2);
}

#[test]
fn test_differentiate_adjoint_hsde_nonneg() {
    // Same LP setup
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let q: Vec<f64> = vec![1.0, 1.0];
    let A = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 1],
        nzval: vec![1.0, 0.0, 0.0, 1.0],
    };
    let b: Vec<f64> = vec![1.0, 1.0];

    let x: Vec<f64> = vec![0.0, 0.0];
    let s: Vec<f64> = vec![1.0, 1.0];
    let z: Vec<f64> = vec![1.0, 1.0];
    let tau = 1.0;

    let cones = vec![SupportedConeT::NonnegativeConeT(2)];

    let dx_bar: Vec<f64> = vec![1.0, 0.0];
    let dz_bar: Vec<f64> = vec![0.0, 0.0];
    let ds_bar: Vec<f64> = vec![0.0, 0.0];

    let result = differentiate_adjoint_hsde(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dx_bar,
        &dz_bar,
        &ds_bar,
        DiffMethod::Exact,
        0.0,
    );

    // Result should have correct dimensions
    assert_eq!(result.dq.len(), 2);
    assert_eq!(result.db.len(), 2);
}

// ========================================================================
// Forward-adjoint consistency test
// ========================================================================

#[test]
fn test_forward_adjoint_consistency_qp() {
    // Test that forward and adjoint are consistent
    // <forward(dP, dq, dA, db), (dx_bar, dz_bar)> = <(dP, dq, dA, db), adjoint(dx_bar, dz_bar)>

    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![2.0, 3.0],
    };
    let q: Vec<f64> = vec![1.0, 1.0];
    let A = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![1.0, 1.0],
    };
    let b: Vec<f64> = vec![2.0];

    let x: Vec<f64> = vec![1.0, 1.0];
    let z: Vec<f64> = vec![-2.5]; // Lagrange multiplier

    // Forward direction
    let dP = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![0.1, 0.2],
    };
    let dq: Vec<f64> = vec![0.3, 0.4];
    let dA = CscMatrix {
        m: 1,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 0],
        nzval: vec![0.5, 0.6],
    };
    let db: Vec<f64> = vec![0.7];

    let forward = differentiate_qp_eq(&P, &q, &A, &b, &x, &z, &dP, &dq, &dA, &db);

    // Adjoint direction
    let dx_bar: Vec<f64> = vec![0.8, 0.9];
    let dz_bar: Vec<f64> = vec![1.0];
    let ds_bar: Vec<f64> = vec![0.0];

    let adjoint = differentiate_adjoint_qp_eq(&P, &q, &A, &b, &x, &z, &dx_bar, &dz_bar, &ds_bar);

    // Compute inner products
    // LHS: <forward, bar>
    let mut lhs = 0.0;
    for i in 0..2 {
        lhs += forward.dx[i] * dx_bar[i];
    }
    for i in 0..1 {
        lhs += forward.dz[i] * dz_bar[i];
    }

    // RHS: <perturbation, adjoint>
    let mut rhs = 0.0;
    // dP contribution
    for k in 0..dP.nnz() {
        rhs += dP.nzval[k] * adjoint.dP.nzval[k];
    }
    // dq contribution
    for i in 0..2 {
        rhs += dq[i] * adjoint.dq[i];
    }
    // dA contribution
    for k in 0..dA.nnz() {
        rhs += dA.nzval[k] * adjoint.dA.nzval[k];
    }
    // db contribution
    for i in 0..1 {
        rhs += db[i] * adjoint.db[i];
    }

    // They should match (up to numerical precision)
    assert!(
        (lhs - rhs).abs() < 1e-5,
        "Forward-adjoint consistency: LHS = {}, RHS = {}",
        lhs,
        rhs
    );
}

// ========================================================================
// HSDE system construction tests
// ========================================================================

#[test]
fn test_build_hsde_kkt_dimensions() {
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![1.0, 1.0],
    };
    let q: Vec<f64> = vec![0.0, 0.0];
    let b: Vec<f64> = vec![1.0, 1.0];
    let A = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 1],
        nzval: vec![1.0, 0.0, 0.0, 1.0],
    };

    let H_blocks: Vec<ConeDerivativeBlock<f64>> = vec![
        ConeDerivativeBlock::Diagonal(vec![1.0, 1.0]), // 2x2 identity as diagonal
    ];
    let cones = vec![SupportedConeT::NonnegativeConeT(2)];

    let c1: Vec<f64> = vec![1.0, 1.0];
    let c2: Vec<f64> = vec![1.0, 1.0];
    let c3: f64 = 1.0;

    let r1: Vec<f64> = vec![0.0, 0.0];
    let r2: Vec<f64> = vec![0.0, 0.0];
    let r3: f64 = 0.0;

    let n = 2;
    let m = 2;

    let (kkt, rhs) = build_hsde_kkt_system(
        &P, &q, &b, &A, &H_blocks, &cones, &c1, &c2, c3, &r1, &r2, r3, n, m,
    );

    // Augmented system size: 2 * (n + 2m + 1)
    let base_dim = n + 2 * m + 1;
    let expected_dim = 2 * base_dim;
    assert_eq!(kkt.nrows(), expected_dim);
    assert_eq!(kkt.ncols(), expected_dim);
    assert_eq!(rhs.len(), expected_dim);
}

#[test]
fn test_build_hsde_kkt_symmetry() {
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![1.0, 1.0],
    };
    let q: Vec<f64> = vec![0.0, 0.0];
    let b: Vec<f64> = vec![1.0, 1.0];
    let A = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 1],
        nzval: vec![1.0, 0.0, 0.0, 1.0],
    };

    let H_blocks: Vec<ConeDerivativeBlock<f64>> = vec![
        ConeDerivativeBlock::Diagonal(vec![0.5, 0.5]), // Diagonal block
    ];
    let cones = vec![SupportedConeT::NonnegativeConeT(2)];

    let c1: Vec<f64> = vec![1.0, 2.0];
    let c2: Vec<f64> = vec![3.0, 4.0];
    let c3: f64 = 5.0;

    let r1: Vec<f64> = vec![0.0, 0.0];
    let r2: Vec<f64> = vec![0.0, 0.0];
    let r3: f64 = 0.0;

    let (kkt, _rhs) = build_hsde_kkt_system(
        &P, &q, &b, &A, &H_blocks, &cones, &c1, &c2, c3, &r1, &r2, r3, 2, 2,
    );

    // Convert upper triangular CSC to full dense matrix
    let dim = kkt.nrows();
    let mut dense: Vec<f64> = vec![0.0; dim * dim];
    for j in 0..dim {
        for k in kkt.colptr[j]..kkt.colptr[j + 1] {
            let i = kkt.rowval[k];
            // Upper triangular: i <= j
            dense[i * dim + j] = kkt.nzval[k];
            // Mirror to lower triangular for symmetry
            dense[j * dim + i] = kkt.nzval[k];
        }
    }

    // Check that upper triangular part is non-empty
    let mut has_offdiag = false;
    for j in 0..dim {
        for k in kkt.colptr[j]..kkt.colptr[j + 1] {
            let i = kkt.rowval[k];
            if i < j {
                has_offdiag = true;
            }
        }
    }
    assert!(
        has_offdiag,
        "Upper triangular should have off-diagonal entries"
    );

    // Now verify the reconstructed matrix is symmetric
    for i in 0..dim {
        for j in i..dim {
            assert!(
                (dense[i * dim + j] - dense[j * dim + i]).abs() < 1e-10,
                "Reconstructed HSDE KKT[{},{}] = {} != {} = KKT[{},{}]",
                i,
                j,
                dense[i * dim + j],
                dense[j * dim + i],
                j,
                i
            );
        }
    }
}

// ========================================================================
// Mixed cone HSDE tests
// ========================================================================

#[test]
fn test_differentiate_hsde_mixed_cones() {
    // Problem with zero cone and nonnegative cone
    let P = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![1.0, 1.0],
    };
    let q: Vec<f64> = vec![0.0, 0.0];
    let A = CscMatrix {
        m: 3,
        n: 2,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 2],
        nzval: vec![1.0, 1.0, 1.0, 1.0],
    };
    let b: Vec<f64> = vec![1.0, 1.0, 1.0];

    let x: Vec<f64> = vec![0.5, 0.5];
    let s: Vec<f64> = vec![0.0, 0.5, 0.5]; // Zero cone: s=0, Nonneg: s>=0
    let z: Vec<f64> = vec![1.0, 0.0, 0.0];
    let tau = 1.0;

    let cones = vec![
        SupportedConeT::ZeroConeT(1),
        SupportedConeT::NonnegativeConeT(2),
    ];

    let dP = CscMatrix {
        m: 2,
        n: 2,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let dq: Vec<f64> = vec![0.1, 0.0];
    let dA = CscMatrix {
        m: 3,
        n: 2,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let db: Vec<f64> = vec![0.0, 0.0, 0.0];

    let result = differentiate_hsde(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Exact,
        0.0,
    );

    assert_eq!(result.dx.len(), 2);
    assert_eq!(result.dz.len(), 3);
    assert_eq!(result.ds.len(), 3);
}

#[test]
fn test_differentiate_hsde_soc() {
    // Problem with second-order cone
    let n = 2;
    let m = 3;

    let P = CscMatrix {
        m: n,
        n: n,
        colptr: vec![0, 1, 2],
        rowval: vec![0, 1],
        nzval: vec![1.0, 1.0],
    };
    let q: Vec<f64> = vec![0.0, 0.0];
    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 2, 4],
        rowval: vec![0, 1, 0, 2],
        nzval: vec![1.0, 1.0, 0.0, 1.0],
    };
    let b: Vec<f64> = vec![2.0, 0.5, 0.5];

    // Solution inside SOC
    let x: Vec<f64> = vec![1.0, 0.5];
    let s: Vec<f64> = vec![1.0, 0.5, 0.0]; // t=1, ||x||=0.5 < 1
    let z: Vec<f64> = vec![0.5, 0.25, 0.0];
    let tau = 1.0;

    let cones = vec![SupportedConeT::SecondOrderConeT(3)];

    let dP = CscMatrix {
        m: n,
        n: n,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let dq: Vec<f64> = vec![0.0, 0.1];
    let dA = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 0, 0],
        rowval: vec![],
        nzval: vec![],
    };
    let db: Vec<f64> = vec![0.0, 0.0, 0.0];

    let result = differentiate_hsde(
        &P,
        &q,
        &A,
        &b,
        &cones,
        &x,
        &s,
        &z,
        tau,
        &dP,
        &dq,
        &dA,
        &db,
        DiffMethod::Exact,
        0.0,
    );

    assert_eq!(result.dx.len(), n);
    assert_eq!(result.dz.len(), m);
    assert_eq!(result.ds.len(), m);
}

#[test]
fn test_qdldl_saddle_point() {
    use crate::qdldl::QDLDLFactorisation;
    // Debug test: Check QDLDL behavior on saddle-point system
    // Matrix: [P+reg, A'; A, -reg] where P=I, A=[1,1], reg=1e-8
    // [1+reg,    0,      1   ]
    // [0,        1+reg,  1   ]
    // [1,        1,     -reg ]

    let reg = 1e-8_f64;
    let kkt = CscMatrix {
        m: 3,
        n: 3,
        colptr: vec![0, 1, 2, 5],
        rowval: vec![0, 1, 0, 1, 2],
        nzval: vec![1.0 + reg, 1.0 + reg, 1.0, 1.0, -reg],
    };

    let rhs = vec![-1.0, 0.0, 0.0];

    println!("\n=== QDLDL Debug Test ===");
    println!("Matrix: [1+e, 0, 1; 0, 1+e, 1; 1, 1, -e] where e={}", reg);

    // Factorize
    let factor = QDLDLFactorisation::new(&kkt, None).expect("QDLDL factorization failed");

    println!("QDLDL factorization:");
    println!("  L.colptr: {:?}", factor.L.colptr);
    println!("  L.rowval: {:?}", factor.L.rowval);
    println!("  L.nzval:  {:?}", factor.L.nzval);
    println!("  D: {:?}", factor.D);
    println!("  Dinv: {:?}", factor.Dinv);
    println!("  perm: {:?}", factor.perm);

    // Solve
    let x = solve_linear_system(&kkt, &rhs);

    println!("\nRHS: {:?}", rhs);
    println!("Solution: {:?}", x);

    // Verify K*x - rhs
    let K = [
        [1.0 + reg, 0.0, 1.0],
        [0.0, 1.0 + reg, 1.0],
        [1.0, 1.0, -reg],
    ];

    let mut result = [0.0; 3];
    for i in 0..3 {
        for j in 0..3 {
            result[i] += K[i][j] * x[j];
        }
    }
    println!("K*x = {:?}", result);

    let mut residual = [0.0; 3];
    for i in 0..3 {
        residual[i] = result[i] - rhs[i];
    }
    println!("K*x - rhs = {:?}", residual);

    // Expected solution (from scipy): [-0.5, 0.5, -0.5]
    println!("\nExpected: [-0.5, 0.5, -0.5]");

    // Assert reasonable bounds
    for i in 0..3 {
        assert!(
            residual[i].abs() < 1e-6,
            "residual[{}] = {} is too large",
            i,
            residual[i]
        );
    }
}

/// Test that the HSDE KKT system with SocSparse blocks has correct dimensions
/// and matches the dense-block KKT for the same problem.
#[test]
fn test_soc_sparse_kkt_matches_dense() {
    // Build a small problem with a dim-5 SOC cone
    // This triggers SocSparse in the backward path
    let n = 3;
    let soc_dim = 5;
    let m = soc_dim;

    // P = I (diagonal)
    let P = CscMatrix {
        m: n,
        n: n,
        colptr: vec![0, 1, 2, 3],
        rowval: vec![0, 1, 2],
        nzval: vec![1.0, 1.0, 1.0],
    };
    let q = vec![1.0, 0.5, -0.3];
    let b = vec![3.0, 0.5, 0.5, 0.5, 0.5];

    // A: identity-like mapping first 3 columns to first 3 SOC rows
    // and zeros for the rest
    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 1, 2, 3],
        rowval: vec![0, 1, 2],
        nzval: vec![1.0, 1.0, 1.0],
    };

    let cones = vec![SupportedConeT::SecondOrderConeT(soc_dim)];

    // Create a boundary-case SOC derivative block
    // z = (0.5, 0.3, 0.3, 0.3, 0.3) — boundary since ||x|| > |t|
    let z = vec![0.5, 0.3, 0.3, 0.3, 0.3];
    use crate::solver::implementations::default::diff::cones::{
        derivative_soc, derivative_soc_sparse,
    };
    let H_sparse: Vec<ConeDerivativeBlock<f64>> = vec![derivative_soc_sparse(&z, soc_dim)];
    let H_dense_data = derivative_soc(&z, soc_dim);
    let H_dense: Vec<ConeDerivativeBlock<f64>> = vec![ConeDerivativeBlock::Dense {
        dim: soc_dim,
        data: H_dense_data,
    }];

    let c1 = vec![1.0; n];
    let c2 = vec![1.0; m];
    let c3 = 1.0;
    // Use a non-trivial RHS to get meaningful solution values
    let mut rhs_base = vec![0.0; n + 2 * m + 1];
    for i in 0..rhs_base.len() {
        rhs_base[i] = (i as f64 + 1.0) * 0.1;
    }

    // Build dense KKT (no expansion)
    let (kkt_dense, aug_rhs_dense) = build_hsde_augmented_system_sparse(
        &P, &A, &q, &b, &H_dense, &cones, &c1, &c2, c3, &rhs_base, n, m, false,
    );

    // Build sparse KKT (with expansion)
    let (kkt_sparse, aug_rhs_sparse) = build_hsde_augmented_system_sparse(
        &P, &A, &q, &b, &H_sparse, &cones, &c1, &c2, c3, &rhs_base, n, m, false,
    );

    // Dense system dim: 2*(n+2m+1) = 2*14 = 28
    // Sparse system dim: 2*(n+2m+1+2) = 2*16 = 32 (2 expansion vars for 1 SOC cone)
    let base_dim = n + 2 * m + 1;
    assert_eq!(kkt_dense.nrows(), 2 * base_dim, "Dense KKT dim wrong");
    assert_eq!(
        kkt_sparse.nrows(),
        2 * (base_dim + 2),
        "Sparse KKT dim wrong"
    );

    // Solve both systems
    let sol_dense = solve_linear_system(&kkt_dense, &aug_rhs_dense);
    let sol_sparse = solve_linear_system(&kkt_sparse, &aug_rhs_sparse);

    // Compare: the first base_dim components of sol should match
    // (the expansion vars in sol_sparse are auxiliary and should be discarded)
    println!(
        "Dense sol len={}, Sparse sol len={}",
        sol_dense.len(),
        sol_sparse.len()
    );
    for i in 0..base_dim {
        let d = sol_dense[i];
        let s = sol_sparse[i];
        println!(
            "  sol[{}]: dense={:.10e}, sparse={:.10e}, diff={:.4e}",
            i,
            d,
            s,
            (d - s).abs()
        );
    }
    // Use absolute tolerance since we're comparing solutions of regularized systems
    let mut max_abs_diff = 0.0_f64;
    for i in 0..base_dim {
        let d = sol_dense[i];
        let s = sol_sparse[i];
        let diff = (d - s).abs();
        max_abs_diff = max_abs_diff.max(diff);
    }
    println!("Max absolute diff: {:.10e}", max_abs_diff);
    assert!(
        max_abs_diff < 1e-5_f64,
        "Solution mismatch too large: {:.10e}",
        max_abs_diff
    );
}

#[test]
fn test_genpow_sparse_kkt_matches_dense() {
    // Build a small problem with a GenPowerCone (3 alphas, dim2=2 → total dim=5)
    // This triggers GenPowerSparse in the backward path
    let n = 3;
    let dim1 = 3;
    let dim2 = 2;
    let genpow_dim = dim1 + dim2;
    let m = genpow_dim;
    let alpha = vec![0.2, 0.3, 0.5];

    let P = CscMatrix {
        m: n,
        n: n,
        colptr: vec![0, 1, 2, 3],
        rowval: vec![0, 1, 2],
        nzval: vec![1.0, 1.0, 1.0],
    };
    let q = vec![1.0, 0.5, -0.3];
    let b = vec![2.0, 3.0, 1.5, 0.3, 0.4]; // in cone interior

    let A = CscMatrix {
        m: m,
        n: n,
        colptr: vec![0, 1, 2, 3],
        rowval: vec![0, 1, 2],
        nzval: vec![1.0, 1.0, 1.0],
    };

    let cones = vec![SupportedConeT::GenPowerConeT(alpha.clone(), dim2)];

    // Create a boundary-case derivative block
    let z = vec![2.0, 3.0, 1.5, 0.3, 0.4];
    use crate::solver::implementations::default::diff::cones::derivative_genpow_cone_sparse;
    let H_sparse: Vec<ConeDerivativeBlock<f64>> =
        vec![derivative_genpow_cone_sparse(&z, &alpha, dim2, true)];
    // Dense version: expand sparse to dense
    let dense_data = H_sparse[0].to_dense();
    let H_dense: Vec<ConeDerivativeBlock<f64>> = vec![ConeDerivativeBlock::Dense {
        dim: genpow_dim,
        data: dense_data,
    }];

    let c1 = vec![1.0; n];
    let c2 = vec![1.0; m];
    let c3 = 1.0;
    let mut rhs_base = vec![0.0; n + 2 * m + 1];
    for i in 0..rhs_base.len() {
        rhs_base[i] = (i as f64 + 1.0) * 0.1;
    }

    // Build dense KKT (no expansion)
    let (kkt_dense, aug_rhs_dense) = build_hsde_augmented_system_sparse(
        &P, &A, &q, &b, &H_dense, &cones, &c1, &c2, c3, &rhs_base, n, m, false,
    );

    // Build sparse KKT (with 3-column expansion for GenPowerSparse)
    let (kkt_sparse, aug_rhs_sparse) = build_hsde_augmented_system_sparse(
        &P, &A, &q, &b, &H_sparse, &cones, &c1, &c2, c3, &rhs_base, n, m, false,
    );

    // Dense system dim: 2*(n+2m+1) = 2*14 = 28
    // Sparse system dim: 2*(n+2m+1+3) = 2*17 = 34 (3 expansion vars for 1 GenPow cone)
    let base_dim = n + 2 * m + 1;
    assert_eq!(kkt_dense.nrows(), 2 * base_dim, "Dense KKT dim wrong");
    assert_eq!(
        kkt_sparse.nrows(),
        2 * (base_dim + 3),
        "Sparse KKT dim wrong"
    );

    // Solve both systems
    let sol_dense = solve_linear_system(&kkt_dense, &aug_rhs_dense);
    let sol_sparse = solve_linear_system(&kkt_sparse, &aug_rhs_sparse);

    // Compare: the first base_dim components should match
    let mut max_abs_diff = 0.0_f64;
    for i in 0..base_dim {
        let d = sol_dense[i];
        let s = sol_sparse[i];
        let diff = (d - s).abs();
        max_abs_diff = max_abs_diff.max(diff);
    }
    println!("GenPow sparse KKT max absolute diff: {:.10e}", max_abs_diff);
    assert!(
        max_abs_diff < 1e-5_f64,
        "GenPow KKT solution mismatch too large: {:.10e}",
        max_abs_diff
    );
}
