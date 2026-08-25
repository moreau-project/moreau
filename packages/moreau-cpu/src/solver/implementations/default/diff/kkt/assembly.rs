//\! KKT-matrix assembly and gradient-extraction helpers.

use super::*;

// ============================================================================
// Helper functions
// ============================================================================

/// Build QP KKT matrix directly in sparse format (upper triangular for QDLDL)
///
/// The KKT matrix is quasi-definite:
/// ```text
/// [P + εI    A' ]
/// [A       -εI  ]
/// ```
/// where ε is small regularization.
pub(super) fn build_qp_kkt_matrix<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    reg: T,
) -> CscMatrix<T> {
    let n = P.nrows();
    let m = A.nrows();
    let dim = n + m;

    // Count nonzeros for upper triangular part:
    // - P's upper triangular entries (including diagonal with regularization)
    // - n diagonal entries for regularization (may overlap with P diagonal)
    // - A entries (these form the upper-right block since row < n, col >= n)
    // - m diagonal entries for -reg*I in lower-right

    // Build column by column for upper triangular CSC
    let mut colptr = vec![0usize; dim + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();

    // Columns 0..n: P's upper triangle + regularization + A' (columns n..n+m contributions)
    for j in 0..n {
        let col_start = rowval.len();

        // Collect P entries for column j (only upper triangle: row <= j)
        let mut p_entries: Vec<(usize, T)> = Vec::new();
        for k in P.colptr[j]..P.colptr[j + 1] {
            let i = P.rowval[k];
            if i <= j {
                p_entries.push((i, P.nzval[k]));
            }
        }
        // Also handle symmetric entries from P (if P stores lower triangle too)
        // P should be symmetric, so we may have entries where i > j stored in column i
        // For safety, also check if there are entries in other columns that belong here
        for col in 0..j {
            for k in P.colptr[col]..P.colptr[col + 1] {
                if P.rowval[k] == j {
                    // This is P[j, col] which by symmetry equals P[col, j]
                    // For upper triangular, we want (col, j) where col < j
                    p_entries.push((col, P.nzval[k]));
                }
            }
        }

        // Sort by row index and merge duplicates (add regularization to diagonal)
        p_entries.sort_by_key(|&(i, _)| i);

        let mut has_diagonal = false;
        let mut prev_row: Option<usize> = None;
        for (i, v) in p_entries {
            if prev_row == Some(i) {
                // Merge with previous (shouldn't happen with proper P, but be safe)
                let last_idx = nzval.len() - 1;
                nzval[last_idx] += v;
            } else {
                // Check if we need to insert diagonal before this entry
                if i == j {
                    has_diagonal = true;
                    rowval.push(i);
                    nzval.push(v + reg); // Add regularization
                } else {
                    rowval.push(i);
                    nzval.push(v);
                }
                prev_row = Some(i);
            }
        }

        // Add diagonal regularization if not already present
        if !has_diagonal {
            // Find correct position to insert diagonal
            let insert_pos = rowval[col_start..].iter().position(|&r| r > j);
            match insert_pos {
                Some(pos) => {
                    rowval.insert(col_start + pos, j);
                    nzval.insert(col_start + pos, reg);
                }
                None => {
                    rowval.push(j);
                    nzval.push(reg);
                }
            }
        }

        // Add A' entries: A[i, j] goes to KKT[j, n+i] for upper triangular
        // But for column j (< n), we need entries where row < j, which means
        // nothing from A' since A' entries are in columns n..n+m
        // Actually, A' has shape n x m, so A'[j, i] = A[i, j]
        // In KKT column j, A' contributes nothing (those go to columns n+i)

        colptr[j + 1] = rowval.len();
    }

    // Columns n..n+m: A (as lower-left, but upper triangular stores it in upper-right)
    // and -reg*I diagonal
    for j in 0..m {
        let kkt_col = n + j;

        // A entries: A[j, k] contributes to KKT[k, n+j] where k < n < n+j, so upper triangular
        // A is m x n, A[j, :] is row j of A
        // We need A[:, k] entries where rowval == j
        for k in 0..n {
            for idx in A.colptr[k]..A.colptr[k + 1] {
                if A.rowval[idx] == j {
                    rowval.push(k);
                    nzval.push(A.nzval[idx]);
                }
            }
        }

        // Add diagonal -reg
        rowval.push(kkt_col);
        nzval.push(-reg);

        colptr[kkt_col + 1] = rowval.len();
    }

    CscMatrix {
        m: dim,
        n: dim,
        colptr,
        rowval,
        nzval,
    }
}

/// Build the HSDE augmented system directly in sparse format.
///
/// The augmented system K = [I  J; J' -reg*I] is built directly in sparse CSC format.
/// This avoids O(n²) dense construction.
///
/// The Jacobian J has the structure (with p expansion variables for SocSparse cones):
/// ```text
/// [P,      A',     0,   q,   0       ]   (n rows)
/// [A,      I,     -I,  -b,   0       ]   (m rows)
/// [0,      I,  -diag,   0,  -c*v     ]   (m rows)
/// [c1,     c2,     0,   c3,  0       ]   (1 row)
/// [0,      0,   -v^T,   0,   I       ]   (p rows)
/// ```
///
/// For non-SocSparse cones, the -H block is dense or diagonal as before.
/// For SocSparse cones, -H is decomposed as -diag + rank-2 expansion.
pub(super) fn build_hsde_augmented_system_sparse<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    q: &[T],
    b: &[T],
    H_blocks: &[ConeDerivativeBlock<T>],
    cones: &[SupportedConeT<T>],
    c1: &[T],
    c2: &[T],
    c3: T,
    rhs: &[T],
    n: usize,
    m: usize,
    for_adjoint: bool,
) -> (CscMatrix<T>, Vec<T>) {
    build_hsde_augmented_system_sparse_full(
        P,
        A,
        q,
        b,
        H_blocks,
        cones,
        &[],
        &[],
        c1,
        c2,
        c3,
        rhs,
        n,
        m,
        for_adjoint,
    )
}

/// IFT-direct version: also accepts direct-x cone indices and `H_x` blocks
/// for direct-x cones. Adds `xn = Σ_J |J|` rows + cols to the augmented J,
/// representing the direct-x cone-projection equations
/// `Π_{K_J}(x[J] + z_x) − x[J] = 0`. The HSDE row/col stays at the very end.
///
/// Layout of `J` columns (0-indexed):
/// - `0..n`               : x
/// - `n..n+m`             : w (slack dual)
/// - `n+m..n+2m`          : du_slack
/// - `n+2m..n+2m+xn`      : du_x
/// - `n+2m+xn`            : τ
/// - after `base_j_dim`   : slack expansion vars (SocSparse, GenPowerSparse)
pub(super) fn build_hsde_augmented_system_sparse_full<T: FloatT>(
    P: &CscMatrix<T>,
    A: &CscMatrix<T>,
    q: &[T],
    b: &[T],
    H_blocks: &[ConeDerivativeBlock<T>],
    cones: &[SupportedConeT<T>],
    xcone_indices: &[Vec<usize>],
    H_x_blocks: &[ConeDerivativeBlock<T>],
    c1: &[T],
    c2: &[T],
    c3: T,
    rhs: &[T],
    n: usize,
    m: usize,
    for_adjoint: bool,
) -> (CscMatrix<T>, Vec<T>) {
    debug_assert_eq!(
        xcone_indices.len(),
        H_x_blocks.len(),
        "xcone_indices and H_x_blocks must have the same number of cones"
    );
    let reg: T = REG_GENERAL.as_T();

    // Count expansion variables: 2 per SocSparse cone, 3 per GenPowerSparse cone
    let mut p_expansion = 0usize;
    // Build mapping: for each sparse cone, its expansion column offset (relative to base j_dim)
    // (cone_idx, cone_offset, exp_col_base, num_exp_vars)
    let mut expansion_map: Vec<(usize, usize, usize, usize)> = Vec::new();
    {
        let mut cone_offset = 0;
        for (cone_idx, block) in H_blocks.iter().enumerate() {
            match block {
                ConeDerivativeBlock::SocSparse { dim, .. } => {
                    expansion_map.push((cone_idx, cone_offset, p_expansion, 2));
                    p_expansion += 2;
                    cone_offset += dim;
                }
                ConeDerivativeBlock::GenPowerSparse { dim, .. } => {
                    expansion_map.push((cone_idx, cone_offset, p_expansion, 3));
                    p_expansion += 3;
                    cone_offset += dim;
                }
                _ => {
                    let dim = match block {
                        ConeDerivativeBlock::Zero(d) => *d,
                        ConeDerivativeBlock::Diagonal(d) => d.len(),
                        ConeDerivativeBlock::Dense { dim, .. } => *dim,
                        _ => unreachable!(),
                    };
                    cone_offset += dim;
                }
            }
        }
    }

    // Direct-x cone offsets within the [n+2m, n+2m+xn) du_x block.
    let mut xn: usize = 0;
    let mut xcone_offsets: Vec<usize> = Vec::with_capacity(xcone_indices.len());
    for ix in xcone_indices {
        xcone_offsets.push(xn);
        xn += ix.len();
    }

    // Count direct-x expansion variables: 2 per SocSparse direct-x cone,
    // 3 per GenPowerSparse direct-x cone. These columns live AFTER the
    // slack expansion cols. The augmented system has structure
    //   [base_j_dim cols .. | slack expansion .. | direct-x expansion]
    // so a direct-x cone xc with rank-r expansion uses cols
    //   base_j_dim + p_expansion (slack total) + xcone_exp_base[xc] .. + r
    // For each direct-x sparse cone we record (xc_idx, xc_du_offset,
    // exp_col_base_relative_to_xcone_block, num_exp_vars).
    let mut p_xcone_expansion = 0usize;
    let mut xcone_expansion_map: Vec<(usize, usize, usize, usize)> = Vec::new();
    for (xc_idx, block) in H_x_blocks.iter().enumerate() {
        match block {
            ConeDerivativeBlock::SocSparse { .. } => {
                xcone_expansion_map.push((xc_idx, xcone_offsets[xc_idx], p_xcone_expansion, 2));
                p_xcone_expansion += 2;
            }
            ConeDerivativeBlock::GenPowerSparse { .. } => {
                xcone_expansion_map.push((xc_idx, xcone_offsets[xc_idx], p_xcone_expansion, 3));
                p_xcone_expansion += 3;
            }
            _ => {}
        }
    }
    // For each primal index, find which direct-x cone it belongs to and its
    // local position within that cone (None if not in any direct-x cone).
    // Used by the stationarity-row column build to scatter `−H_x` entries.
    let mut x_to_xcone: Vec<Option<(usize, usize)>> = vec![None; n];
    for (xc_idx, ix) in xcone_indices.iter().enumerate() {
        for (k, &i) in ix.iter().enumerate() {
            debug_assert!(
                x_to_xcone[i].is_none(),
                "direct-x indices must be disjoint across cones (idx {} reused)",
                i
            );
            x_to_xcone[i] = Some((xc_idx, k));
        }
    }

    let base_j_dim = n + 2 * m + xn + 1; // J dimension without expansion vars
                                         // Slack expansion vars come first (cols base_j_dim..base_j_dim+p_expansion),
                                         // then direct-x expansion vars. `xcone_exp_base_abs` = offset where
                                         // direct-x expansion cols start within the augmented system.
    let xcone_exp_base_abs = base_j_dim + p_expansion;
    let j_dim = base_j_dim + p_expansion + p_xcone_expansion;
    let aug_dim = 2 * j_dim;

    // Build upper triangular part of K column by column
    let mut colptr = vec![0usize; aug_dim + 1];
    let mut rowval = Vec::new();
    let mut nzval = Vec::new();

    // First j_dim columns: [I; J'] — only diagonal I entries (J' goes in second half columns)
    for col in 0..j_dim {
        rowval.push(col);
        nzval.push(T::one());
        colptr[col + 1] = rowval.len();
    }

    // Columns j_dim..aug_dim: [J; -reg*I]
    for jcol in 0..j_dim {
        let aug_col = j_dim + jcol;

        // Collect J column entries
        let mut j_col_entries: Vec<(usize, T)> = Vec::new();

        if jcol < n {
            // Column in first n columns of J
            // Block (0,0): P column jcol (full symmetric)
            for k in P.colptr[jcol]..P.colptr[jcol + 1] {
                let row = P.rowval[k];
                j_col_entries.push((row, P.nzval[k]));
            }
            for col in (jcol + 1)..n {
                for k in P.colptr[col]..P.colptr[col + 1] {
                    if P.rowval[k] == jcol {
                        j_col_entries.push((col, P.nzval[k]));
                    }
                }
            }

            // Block (1,0): A column jcol
            for k in A.colptr[jcol]..A.colptr[jcol + 1] {
                let i = A.rowval[k];
                j_col_entries.push((n + i, A.nzval[k]));
            }

            // Direct-x rows: if `jcol` is one of the gathered indices `J[k]`,
            // then F_4_k = x[J[k]] − (I − H_x)·du_x has +1 at that x column.
            if let Some((xc_idx, k)) = x_to_xcone[jcol] {
                let xc_row = n + 2 * m + xcone_offsets[xc_idx] + k;
                j_col_entries.push((xc_row, T::one()));
            }

            // Block (3,0): c1[jcol] — tau row (now n + 2m + xn)
            j_col_entries.push((n + 2 * m + xn, c1[jcol]));
        } else if jcol < n + m {
            // Column in n..n+m of J (w columns)
            let w_idx = jcol - n;

            // Block (0,1): A' column w_idx
            for pcol in 0..n {
                for k in A.colptr[pcol]..A.colptr[pcol + 1] {
                    if A.rowval[k] == w_idx {
                        j_col_entries.push((pcol, A.nzval[k]));
                    }
                }
            }

            // Block (1,1): I diagonal
            j_col_entries.push((n + w_idx, T::one()));

            // Block (2,1): I
            j_col_entries.push((n + m + w_idx, T::one()));

            // Block (3,1): c2[w_idx] — tau row (now n + 2m + xn)
            j_col_entries.push((n + 2 * m + xn, c2[w_idx]));
        } else if jcol < n + 2 * m {
            // Column in n+m..n+2m of J (du columns)
            let u_idx = jcol - n - m;

            // Block (1,2): -I
            j_col_entries.push((n + u_idx, -T::one()));

            // Block (2,2): -H (diagonal part only for SocSparse, full for Dense)
            let mut offset = 0;
            for (cone_idx, cone) in cones.iter().enumerate() {
                let cone_dim = cone.nvars();
                if u_idx >= offset && u_idx < offset + cone_dim {
                    let local_col = u_idx - offset;
                    let H_block = &H_blocks[cone_idx];

                    match H_block {
                        ConeDerivativeBlock::Zero(_) => {
                            // -H = 0, nothing to add
                        }
                        ConeDerivativeBlock::Diagonal(diag) => {
                            let val = -diag[local_col];
                            j_col_entries.push((n + m + offset + local_col, val));
                        }
                        ConeDerivativeBlock::Dense { dim, data } => {
                            for local_row in 0..*dim {
                                let val = -data[local_row * dim + local_col];
                                j_col_entries.push((n + m + offset + local_row, val));
                            }
                        }
                        ConeDerivativeBlock::SocSparse { dim: _, diag, .. } => {
                            // Only diagonal entry for the du column
                            let val = -diag[local_col];
                            j_col_entries.push((n + m + offset + local_col, val));

                            // Expansion row entries: -v^T in the expansion rows
                            for &(exp_cone_idx, _, exp_base, _) in &expansion_map {
                                if exp_cone_idx == cone_idx {
                                    let H_block_ref = &H_blocks[cone_idx];
                                    if let ConeDerivativeBlock::SocSparse { v1, v2, .. } =
                                        H_block_ref
                                    {
                                        j_col_entries.push((base_j_dim + exp_base, -v1[local_col]));
                                        j_col_entries
                                            .push((base_j_dim + exp_base + 1, -v2[local_col]));
                                    }
                                    break;
                                }
                            }
                        }
                        ConeDerivativeBlock::GenPowerSparse { dim: _, diag, .. } => {
                            // Only diagonal entry for the du column
                            let val = -diag[local_col];
                            j_col_entries.push((n + m + offset + local_col, val));

                            // Expansion row entries: -right^T in the expansion rows
                            for &(exp_cone_idx, _, exp_base, _) in &expansion_map {
                                if exp_cone_idx == cone_idx {
                                    let H_block_ref = &H_blocks[cone_idx];
                                    if let ConeDerivativeBlock::GenPowerSparse {
                                        right1,
                                        right2,
                                        left3,
                                        ..
                                    } = H_block_ref
                                    {
                                        // Row 0: -right1[local_col] (b vector)
                                        j_col_entries
                                            .push((base_j_dim + exp_base, -right1[local_col]));
                                        // Row 1: -right2[local_col] (f vector)
                                        j_col_entries
                                            .push((base_j_dim + exp_base + 1, -right2[local_col]));
                                        // Row 2: -left3[local_col] (g vector, right3 = left3 for symmetric term)
                                        j_col_entries
                                            .push((base_j_dim + exp_base + 2, -left3[local_col]));
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    break;
                }
                offset += cone_dim;
            }
        } else if jcol < n + 2 * m + xn {
            // du_x column block: jcol in [n+2m, n+2m+xn).
            // Locate which direct-x cone this column belongs to.
            let local = jcol - n - 2 * m;
            let xc_idx = xcone_offsets
                .iter()
                .rposition(|&off| off <= local)
                .expect("xcone_offsets must cover all du_x cols");
            let l = local - xcone_offsets[xc_idx];
            let dim_xc = xcone_indices[xc_idx].len();
            let xc_row_base = n + 2 * m + xcone_offsets[xc_idx];

            // Direct-x F_4 row: E_J x + (I − H_x) du_x = 0. The du_x col
            // contributes -H_x to the stationarity rows (at primal index
            // J[k]) and +(I − H_x) to the direct-x rows (xc_row_base + k).
            // The (I − H_x) sign matters: with the opposite sign, the
            // reduced stationarity has -H_x(I−H_x)^{-1} instead of the
            // slack form's +H_x(I−H_x)^{-1}, which flips off-diagonal
            // coupling sign for non-diagonal H_x (SOC, PSD).
            match &H_x_blocks[xc_idx] {
                ConeDerivativeBlock::Zero(_) => {
                    // H_x = 0: F_4 = E_J x + I du_x → +1 on the diagonal.
                    j_col_entries.push((xc_row_base + l, T::one()));
                }
                ConeDerivativeBlock::Diagonal(diag) => {
                    // Stationarity row J[l]: -H_x[l, l] (only diagonal contributes).
                    // Always emit the slot so cached symbolic factorizations
                    // see a fixed nnz pattern even when h_ll == 0 at runtime.
                    let row_x = xcone_indices[xc_idx][l];
                    let h_ll = diag[l];
                    j_col_entries.push((row_x, -h_ll));
                    // Direct-x row at local k=l: 1 - diag[l].
                    j_col_entries.push((xc_row_base + l, T::one() - h_ll));
                }
                ConeDerivativeBlock::Dense { dim: _, data } => {
                    // Always emit every slot in the dense block so the
                    // symbolic factorization is a superset of the runtime
                    // numeric pattern.
                    for k in 0..dim_xc {
                        let val = data[k * dim_xc + l];
                        j_col_entries.push((xcone_indices[xc_idx][k], -val));
                        let dx_val = if k == l { T::one() - val } else { -val };
                        j_col_entries.push((xc_row_base + k, dx_val));
                    }
                }
                ConeDerivativeBlock::SocSparse {
                    dim: _,
                    diag,
                    v1,
                    v2,
                    ..
                } => {
                    // Rank-2 sparse expansion. The du_x col l of cone xc
                    // emits only the diagonal contribution at the local
                    // row l (stat + direct-x), plus rank coefficients
                    // -v_i[l] in the two direct-x expansion rows.
                    // Off-diagonal rank-1 contributions are folded
                    // through the expansion variables.
                    let h_ll = diag[l];
                    j_col_entries.push((xcone_indices[xc_idx][l], -h_ll));
                    j_col_entries.push((xc_row_base + l, T::one() - h_ll));
                    // Direct-x expansion rows for this cone.
                    for &(exp_xc, _, exp_base, _) in &xcone_expansion_map {
                        if exp_xc == xc_idx {
                            j_col_entries.push((xcone_exp_base_abs + exp_base, -v1[l]));
                            j_col_entries.push((xcone_exp_base_abs + exp_base + 1, -v2[l]));
                            break;
                        }
                    }
                }
                ConeDerivativeBlock::GenPowerSparse {
                    dim: _,
                    diag,
                    right1,
                    right2,
                    left3,
                    ..
                } => {
                    // Rank-3 sparse expansion. Same shape as SocSparse but
                    // with three rank components (r3 = c3 * left3 ⊗ left3
                    // is symmetric so right3 == left3).
                    let h_ll = diag[l];
                    j_col_entries.push((xcone_indices[xc_idx][l], -h_ll));
                    j_col_entries.push((xc_row_base + l, T::one() - h_ll));
                    for &(exp_xc, _, exp_base, _) in &xcone_expansion_map {
                        if exp_xc == xc_idx {
                            j_col_entries.push((xcone_exp_base_abs + exp_base, -right1[l]));
                            j_col_entries.push((xcone_exp_base_abs + exp_base + 1, -right2[l]));
                            j_col_entries.push((xcone_exp_base_abs + exp_base + 2, -left3[l]));
                            break;
                        }
                    }
                }
            }
        } else if jcol == n + 2 * m + xn {
            // Tau column (now n + 2m + xn)
            // Block (0,3): q
            for i in 0..n {
                j_col_entries.push((i, q[i]));
            }

            // Block (1,3): -b
            for i in 0..m {
                j_col_entries.push((n + i, -b[i]));
            }

            // Block (3,3): c3 — tau row (now n + 2m + xn)
            j_col_entries.push((n + 2 * m + xn, c3));
        } else if jcol < xcone_exp_base_abs {
            // Slack expansion columns: jcol in base_j_dim..xcone_exp_base_abs
            let exp_idx = jcol - base_j_dim;

            // Find which sparse cone and which expansion vector this belongs to
            for &(exp_cone_idx, cone_offset, exp_base, num_exp) in &expansion_map {
                if exp_idx >= exp_base && exp_idx < exp_base + num_exp {
                    let local_exp = exp_idx - exp_base;
                    let H_block = &H_blocks[exp_cone_idx];

                    match H_block {
                        ConeDerivativeBlock::SocSparse {
                            dim,
                            v1,
                            c1: coeff1,
                            v2,
                            c2: coeff2,
                            ..
                        } => {
                            let (vec_ref, coeff) = if local_exp == 0 {
                                (v1, *coeff1)
                            } else {
                                (v2, *coeff2)
                            };

                            // Entries in H-block rows: -c * v[i] at rows n+m+cone_offset+i
                            for i in 0..*dim {
                                j_col_entries.push((n + m + cone_offset + i, -coeff * vec_ref[i]));
                            }

                            // Diagonal entry for this expansion variable: 1
                            j_col_entries.push((jcol, T::one()));
                        }
                        ConeDerivativeBlock::GenPowerSparse {
                            dim,
                            left1,
                            left2,
                            left3,
                            c3,
                            ..
                        } => {
                            // 3 expansion columns:
                            //   0: left column = -left1[i] (a vector, coeff = 1)
                            //   1: left column = -left2[i] (e vector, coeff = 1)
                            //   2: left column = -c3*left3[i] (g vector, coeff = c3)
                            let (vec_ref, coeff) = match local_exp {
                                0 => (left1, T::one()),
                                1 => (left2, T::one()),
                                2 => (left3, *c3),
                                _ => unreachable!(),
                            };

                            for i in 0..*dim {
                                j_col_entries.push((n + m + cone_offset + i, -coeff * vec_ref[i]));
                            }

                            // Diagonal entry for this expansion variable: 1
                            j_col_entries.push((jcol, T::one()));
                        }
                        _ => unreachable!(),
                    }
                    break;
                }
            }
        } else {
            // Direct-x expansion columns: jcol in
            //   xcone_exp_base_abs .. xcone_exp_base_abs + p_xcone_expansion
            // Each rank-r component of cone xc contributes -c * left[k]
            // entries to BOTH the stationarity rows J_xc[k] AND the
            // direct-x rows xc_row_base + k (the rank-1 contribution
            // shows up as -H_x[k,l] in the stat row and -H_x[k,l] in
            // the direct-x row, off-diagonally).
            let dx_exp_idx = jcol - xcone_exp_base_abs;
            for &(exp_xc_idx, _xc_du_off, exp_base, num_exp) in &xcone_expansion_map {
                if dx_exp_idx >= exp_base && dx_exp_idx < exp_base + num_exp {
                    let local_exp = dx_exp_idx - exp_base;
                    let dim_xc = xcone_indices[exp_xc_idx].len();
                    let xc_row_base = n + 2 * m + xcone_offsets[exp_xc_idx];
                    let H_block = &H_x_blocks[exp_xc_idx];
                    match H_block {
                        ConeDerivativeBlock::SocSparse {
                            dim: _,
                            v1,
                            c1: coeff1,
                            v2,
                            c2: coeff2,
                            ..
                        } => {
                            let (vec_ref, coeff) = if local_exp == 0 {
                                (v1, *coeff1)
                            } else {
                                (v2, *coeff2)
                            };
                            for k in 0..dim_xc {
                                let val = -coeff * vec_ref[k];
                                j_col_entries.push((xcone_indices[exp_xc_idx][k], val));
                                j_col_entries.push((xc_row_base + k, val));
                            }
                            j_col_entries.push((jcol, T::one()));
                        }
                        ConeDerivativeBlock::GenPowerSparse {
                            dim: _,
                            left1,
                            left2,
                            left3,
                            c3,
                            ..
                        } => {
                            let (vec_ref, coeff) = match local_exp {
                                0 => (left1, T::one()),
                                1 => (left2, T::one()),
                                2 => (left3, *c3),
                                _ => unreachable!(),
                            };
                            for k in 0..dim_xc {
                                let val = -coeff * vec_ref[k];
                                j_col_entries.push((xcone_indices[exp_xc_idx][k], val));
                                j_col_entries.push((xc_row_base + k, val));
                            }
                            j_col_entries.push((jcol, T::one()));
                        }
                        _ => {
                            unreachable!("direct-x expansion col emitted for non-sparse H_x_block")
                        }
                    }
                    break;
                }
            }
        }

        // Sort and deduplicate
        j_col_entries.sort_by_key(|&(r, _)| r);

        let mut prev_row: Option<usize> = None;
        for (row, val) in j_col_entries {
            if prev_row == Some(row) {
                let last_idx = nzval.len() - 1;
                nzval[last_idx] += val;
            } else {
                rowval.push(row);
                nzval.push(val);
                prev_row = Some(row);
            }
        }

        // Add diagonal -reg for lower-right block
        rowval.push(aug_col);
        nzval.push(-reg);

        colptr[aug_col + 1] = rowval.len();
    }

    // Build augmented RHS: rhs is length base_j_dim, expand to j_dim with zeros for expansion vars
    let mut aug_rhs = vec![T::zero(); aug_dim];
    if for_adjoint {
        // RHS goes into second half, first base_j_dim entries of that half
        aug_rhs[j_dim..j_dim + base_j_dim].copy_from_slice(rhs);
    } else {
        // RHS goes into first half, first base_j_dim entries
        aug_rhs[..base_j_dim].copy_from_slice(rhs);
    }

    (
        CscMatrix {
            m: aug_dim,
            n: aug_dim,
            colptr,
            rowval,
            nzval,
        },
        aug_rhs,
    )
}

/// Build HSDE KKT system using sparse construction
pub(super) fn build_hsde_kkt_system<T: FloatT>(
    P: &CscMatrix<T>,
    q: &[T],
    b: &[T],
    A: &CscMatrix<T>,
    H_blocks: &[ConeDerivativeBlock<T>],
    cones: &[SupportedConeT<T>],
    c1: &[T],
    c2: &[T],
    c3: T,
    r1: &[T],
    r2: &[T],
    r3: T,
    n: usize,
    m: usize,
) -> (CscMatrix<T>, Vec<T>) {
    let dim = n + 2 * m + 1;

    // Build the RHS
    let mut rhs = vec![T::zero(); dim];
    rhs[..n].copy_from_slice(r1);
    rhs[n..n + m].copy_from_slice(r2);
    // rhs[n+m..n+2*m] = 0 (constraint w = H*du)
    rhs[dim - 1] = r3;

    // Build augmented system directly in sparse format
    build_hsde_augmented_system_sparse(P, A, q, b, H_blocks, cones, c1, c2, c3, &rhs, n, m, false)
}

/// Solve a linear system using QDLDL
pub(super) fn solve_linear_system<T: FloatT>(kkt: &CscMatrix<T>, rhs: &[T]) -> Vec<T> {
    use crate::qdldl::{QDLDLFactorisation, QDLDLSettingsBuilder};

    // Disable QDLDL's internal regularization - we handle regularization externally
    // in the KKT matrix construction
    let settings = QDLDLSettingsBuilder::default()
        .regularize_enable(false)
        .build()
        .expect("Failed to build QDLDL settings");

    let mut factor =
        QDLDLFactorisation::new(kkt, Some(settings)).expect("QDLDL factorization failed");

    let mut x = rhs.to_vec();
    factor.solve(&mut x);

    x
}

/// Compute gradient w.r.t. P (at nonzero positions only)
///
/// P is expected as a full symmetric matrix (both upper and lower triangle entries present).
/// For symmetric P, when we compute Px in the objective 0.5*x'Px, each off-diagonal pair
/// (i,j) and (j,i) both contribute to the result. The gradient formula from diffclarabel is:
///
///   dP[i,j] = -0.5 * (lam_x[i] * x[j] + lam_x[j] * x[i])
///
/// This gives the same value for both P[i,j] and P[j,i] (as expected for symmetric P).
/// For diagonal entries, this simplifies to: dP[i,i] = -lam_x[i] * x[i]
pub(super) fn compute_gradient_P<T: FloatT>(
    P: &CscMatrix<T>,
    lam_x: &[T],
    x: &[T],
) -> CscMatrix<T> {
    let n = P.nrows();
    let half = T::from_f64(0.5).unwrap();

    let mut nzval = vec![T::zero(); P.nnz()];

    for j in 0..n {
        for k in P.colptr[j]..P.colptr[j + 1] {
            let i = P.rowval[k];
            // Use diffclarabel formula: -0.5 * (lam[i]*x[j] + lam[j]*x[i])
            // For diagonal (i==j): -0.5 * 2 * lam[i] * x[i] = -lam[i] * x[i]
            // For off-diagonal: -0.5 * (lam[i]*x[j] + lam[j]*x[i])
            nzval[k] = -half * (lam_x[i] * x[j] + lam_x[j] * x[i]);
        }
    }

    CscMatrix {
        m: n,
        n,
        colptr: P.colptr.clone(),
        rowval: P.rowval.clone(),
        nzval,
    }
}

/// Compute gradient w.r.t. A (at nonzero positions only)
pub(super) fn compute_gradient_A<T: FloatT>(
    A: &CscMatrix<T>,
    lam_x: &[T],
    lam_z: &[T],
    x: &[T],
    z: &[T],
) -> CscMatrix<T> {
    let m = A.nrows();
    let n = A.ncols();

    // dA[i,j] = -lam_z[i] * x[j] - lam_x[j] * z[i]
    // Only populate at A's nonzero positions (following diffclarabel)

    let mut nzval = vec![T::zero(); A.nnz()];

    for j in 0..n {
        for k in A.colptr[j]..A.colptr[j + 1] {
            let i = A.rowval[k];
            nzval[k] = -lam_z[i] * x[j] - lam_x[j] * z[i];
        }
    }

    CscMatrix {
        m,
        n,
        colptr: A.colptr.clone(),
        rowval: A.rowval.clone(),
        nzval,
    }
}

/// Compute gradient w.r.t. P for HSDE formulation
///
/// P is expected as a full symmetric matrix (both upper and lower triangle entries present).
/// The formula from diffclarabel is:
///   dP[i,j] = -0.5 * (lam1[i] * x[j] + lam1[j] * x[i]) + lam4 * x[i] * x[j] / tau
///
/// This gives the same value for both P[i,j] and P[j,i] (as expected for symmetric P).
pub(super) fn compute_gradient_P_hsde<T: FloatT>(
    P: &CscMatrix<T>,
    lam1: &[T],
    lam4: T,
    x: &[T],
    tau: T,
) -> CscMatrix<T> {
    let n = P.nrows();
    let half = T::from_f64(0.5).unwrap();

    let mut nzval = vec![T::zero(); P.nnz()];

    for j in 0..n {
        for k in P.colptr[j]..P.colptr[j + 1] {
            let i = P.rowval[k];
            // Use diffclarabel formula: -0.5 * (lam1[i]*x[j] + lam1[j]*x[i]) + lam4*x[i]*x[j]/tau
            // For diagonal (i==j): -0.5 * 2 * lam1[i] * x[i] + lam4*x[i]^2/tau = -lam1[i]*x[i] + lam4*x[i]^2/tau
            // For off-diagonal: -0.5 * (lam1[i]*x[j] + lam1[j]*x[i]) + lam4*x[i]*x[j]/tau
            nzval[k] = -half * (lam1[i] * x[j] + lam1[j] * x[i]) + lam4 * x[i] * x[j] / tau;
        }
    }

    CscMatrix {
        m: n,
        n,
        colptr: P.colptr.clone(),
        rowval: P.rowval.clone(),
        nzval,
    }
}

/// Compute gradient w.r.t. A for HSDE formulation
pub(super) fn compute_gradient_A_hsde<T: FloatT>(
    A: &CscMatrix<T>,
    lam1: &[T],
    lam2: &[T],
    x: &[T],
    z: &[T],
    _tau: T,
) -> CscMatrix<T> {
    let m = A.nrows();
    let n = A.ncols();

    // dA[i,j] = -lam1[j] * z[i] - lam2[i] * x[j]
    // Only at nonzero positions

    let mut nzval = vec![T::zero(); A.nnz()];

    for j in 0..n {
        for k in A.colptr[j]..A.colptr[j + 1] {
            let i = A.rowval[k];
            nzval[k] = -lam1[j] * z[i] - lam2[i] * x[j];
        }
    }

    CscMatrix {
        m,
        n,
        colptr: A.colptr.clone(),
        rowval: A.rowval.clone(),
        nzval,
    }
}
